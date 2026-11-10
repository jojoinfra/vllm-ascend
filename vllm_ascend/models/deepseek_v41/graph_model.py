"""FULL_DECODE_ONLY target with vLLM-owned graph execution and external host Engram staging."""

import json
import logging
from copy import copy
from pathlib import Path

import torch
from torch import nn
from vllm.forward_context import get_forward_context

from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.models.deepseek_v41.backend import TextBackbone, make_identity_pre_mix
from vllm_ascend.models.deepseek_v41.engram import HostEngramEmbedding

from .decode_state import DecodeInputs, PagedHistory
from .graph_ops import LayerPages, paged_attention
from .model import AscendDeepseekV41ForCausalLM

LOG = logging.getLogger(__name__)


class StagedEmbedding(HostEngramEmbedding):
    """The standard loader binds the original shared mmap; capture only reads fixed device rows."""

    def forward(self, placeholder):
        width = placeholder.size(1)
        raw, scales = self.stage.engrams[self.index]
        values = self._value_lut[raw[:width].long()].unflatten(-1, (-1, 32))
        values = values * self._scale_lut[scales[:width].long()].unsqueeze(-1)
        return values.flatten(-2).bfloat16().unsqueeze(0)


class PagedACLGraphWrapper(ACLGraphWrapper):
    """Stage the model's host rows before the platform captures or replays this target."""

    def __call__(self, *args, **kwargs):
        ctx = get_forward_context()
        if ctx.cudagraph_runtime_mode != self.runtime_mode:
            return super().__call__(*args, **kwargs)
        ids = kwargs["input_ids"] if "input_ids" in kwargs else args[0]
        pos = kwargs["positions"] if "positions" in kwargs else args[1]
        desc = ctx.batch_descriptor
        old_entry = self.concrete_aclgraph_entries.get(desc)
        replay = old_entry is not None and old_entry.aclgraph is not None
        model = self.runnable
        _, live, max_block = model._prepare(ids, pos)
        model._inside_full_wrapper = True
        try:
            output = super().__call__(*args, **kwargs)
        finally:
            model._inside_full_wrapper = False
        graph = self.concrete_aclgraph_entries[desc].aclgraph
        if not isinstance(graph, torch.npu.NPUGraph):
            raise RuntimeError("The platform did not produce the expected native NPU graph")
        if replay:
            if model.replay_count == 0:
                torch.npu.current_stream().synchronize()
            model.replay_count += 1
        model._record(
            "replay" if replay else "capture",
            ids.numel(),
            live=live,
            max_physical_block=max_block,
            replays=model.replay_count,
        )
        return output


class DeepseekV41Stage2ForCausalLM(AscendDeepseekV41ForCausalLM):
    ascend_full_graph_wrapper_cls = PagedACLGraphWrapper
    supports_eagle3 = True
    has_own_embed_tokens = True
    has_own_lm_head = True

    def __init__(self, *, vllm_config, prefix=""):
        # The registered type's eager initializer owns a single-sequence cache.
        # The paged variant builds its own attention and history storage below.
        nn.Module.__init__(self)
        from vllm.distributed.parallel_state import get_dp_group, get_ep_group
        from vllm.model_executor.layers.attention import Attention
        from vllm.tokenizers import cached_tokenizer_from_config

        p = vllm_config.parallel_config
        m = vllm_config.model_config
        cache = vllm_config.cache_config
        if (p.data_parallel_size, p.tensor_parallel_size, p.pipeline_parallel_size) != (8, 1, 1):
            raise ValueError("V4.1 full-graph stage requires DP8, TP1, PP1")
        if not p.enable_expert_parallel or getattr(p, "prefill_context_parallel_size", 1) != 1:
            raise ValueError("V4.1 full-graph stage requires EP8 and PCP1")
        if cache.enable_prefix_caching:
            raise ValueError("Stage two text n-grams require prefix caching disabled")
        if cache.num_gpu_blocks_override is None:
            raise ValueError("The model and scheduler require the same bounded physical page pool")
        group = get_ep_group()
        if group.world_size != 8:
            raise ValueError("Expected an eight-device expert group")
        self.dp_rank = get_dp_group().rank_in_group
        local = m.hf_text_config.n_routed_experts // group.world_size
        self.expert_range = (group.rank_in_group * local, (group.rank_in_group + 1) * local)
        tokenizer = cached_tokenizer_from_config(model_config=m)
        device = torch.device("npu", torch.npu.current_device())
        self.backbone = TextBackbone(
            m.hf_text_config,
            tokenizer,
            device=device,
            max_seq_len=m.max_model_len,
            ep_group=group.device_group,
            engram_embedding_factory=lambda _, rows, dim: StagedEmbedding(rows, dim),
        )
        self.metadata_key = f"{prefix + '.' if prefix else ''}model.layers.0.self_attn.attn"
        # Compressed KV can select blocks outside the 128-token raw attention
        # window. Its scheduling group must retain every page of the prefix.
        full_history_cache = copy(cache)
        full_history_cache.sliding_window = None
        self.metadata_layer = Attention(
            1, 128, 128**-0.5, num_kv_heads=1, cache_config=full_history_cache, prefix=self.metadata_key
        )
        self.dim = self.backbone.dim
        self.history = PagedHistory(m.max_model_len, cache.block_size, cache.num_gpu_blocks_override)
        self.engram_layers = [b.engram for b in self.backbone.layers if b.engram is not None]
        self.stage = DecodeInputs(
            self.history, vllm_config.scheduler_config.max_num_batched_tokens, device, self.engram_layers
        )
        for layer in self.engram_layers:
            layer.embed.stage = self.stage
            layer.embed.index = layer.layer_hash_index
        self.layer_pages = [LayerPages(block.attn, self.stage) for block in self.backbone.layers]
        self.target_layers = tuple(m.hf_text_config.dspark_target_layer_ids)
        self.aux_layers = self.target_layers if vllm_config.speculative_config is not None else ()
        self.finalized = False
        self.replay_count = 0
        self.receipt = None
        self.moe_prefix = f"{prefix + '.' if prefix else ''}model.layers"
        self._inside_full_wrapper = False

    def finalize_after_load(self, receipt_dir=None):
        from .fused_moe import install_native_layers

        if self.finalized:
            return
        if receipt_dir is not None:
            path = Path(receipt_dir)
            path.mkdir(parents=True, exist_ok=True)
            self.receipt = path / f"graph-rank-{self.dp_rank}.jsonl"
        install_native_layers(self.backbone, prefix=self.moe_prefix, receipt_dir=receipt_dir)
        self.finalized = True

    def _record(self, event, width, **fields):
        record = dict(
            event=event,
            rank=self.dp_rank,
            device=str(self.backbone.device),
            api="torch.npu.NPUGraph",
            width=width,
            **fields,
        )
        if self.receipt is not None:
            record["requests"] = self._scheduled_geometry
            with self.receipt.open("a") as f:
                f.write(json.dumps(record) + "\n")
        elif event == "capture" or self.replay_count == 1:
            LOG.info("V4.1 graph %s", record)

    def _prepare(self, ids, pos):
        plan = self.history.from_context(get_forward_context(), self.metadata_key, pos)
        values = ids.detach().cpu().numpy() if plan.active.any() else None
        context = self.stage.stage(plan, values, self.backbone.engram_hash, self.engram_layers)
        live = int(plan.active.sum())
        max_block = int(plan.slots[plan.active].max()) // self.history.block_size if live else -1
        if self.receipt is not None:
            # Prefix caching is disabled. A request owns its first physical page
            # until release; its logical positions distinguish reuse and rewinds.
            geometry = {}
            for block, position, slot in zip(
                plan.token_pages[plan.active, 0], plan.positions[plan.active], plan.slots[plan.active]
            ):
                row = geometry.setdefault(int(block), dict(first_block=int(block), start=int(position), tokens=0))
                row["tokens"] += 1
                row["last_position"] = int(position)
                row["last_slot"] = int(slot)
            self._scheduled_geometry = list(geometry.values())
        return context, live, max_block

    def set_aux_hidden_state_layers(self, layers):
        if tuple(layers) != self.target_layers:
            raise ValueError(f"DSpark expects the checkpoint's target block inputs {self.target_layers}")
        self.aux_layers = tuple(layers)

    def get_eagle3_default_aux_hidden_state_layers(self):
        return self.target_layers

    def embed_input_ids(self, input_ids):
        return self.backbone.embed(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        if input_ids is None or inputs_embeds is not None or intermediate_tensors is not None:
            raise ValueError("Expected text token IDs and one pipeline stage")
        if not self.finalized:
            raise RuntimeError("Load the target's native FusedMoE layers before running the model")
        width = input_ids.numel()
        if positions.numel() != width or not 0 < width <= self.stage.max_tokens:
            raise ValueError("Inconsistent input shape or a batch wider than the configured stage")
        if torch.npu.is_current_stream_capturing():
            if not self._inside_full_wrapper:
                raise RuntimeError("Full capture requires the V4.1 paged graph wrapper")
            context = self.history.max_len
        else:
            context, _, _ = self._prepare(input_ids, positions)
        safe_ids = torch.where(self.stage.active[:width], input_ids, 0)
        h = self.backbone.embed(safe_ids).reshape(1, width, 1, self.dim).expand(-1, -1, self.backbone.hc_mult, -1)
        pre = make_identity_pre_mix(h, self.backbone.hc_mult)
        shared, aux = {}, []
        for block, pages in zip(self.backbone.layers, self.layer_pages):
            if block.engram is not None:
                h = block.engram(h, input_ids.reshape(1, width), self.stage.active[:width].unsqueeze(0))
            if block.layer_id in self.aux_layers:
                aux.append(h.mean(2).squeeze(0))
            residual = h
            a_pre, a_post, a_comb = block.hc_mixes(h, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
            x = block.attn_norm(block.hc_pre(h, pre))
            x = paged_attention(block.attn, x, self.stage, pages, shared, context)
            h = block.hc_post(x, residual, a_post, a_comb)
            residual = h
            pre, f_post, f_comb = block.hc_mixes(h, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base)
            x = block.ffn_norm(block.hc_pre(h, a_pre))
            h = block.hc_post(block.ffn(x, None), residual, f_post, f_comb)
        final = self.backbone.norm(block.hc_pre(h, pre)).squeeze(0)
        return (final, aux) if self.aux_layers else final

    def compute_logits(self, hidden_states):
        return self.backbone.head(hidden_states, full_logits=True)
