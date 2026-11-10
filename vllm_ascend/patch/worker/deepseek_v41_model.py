# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plugin-side V4.1 backbone, composed from paged attention and Ascend MoE.

Architecture follows the pinned official HF implementation. The delayed mHC
pre-mix, Engram history and auxiliary target inputs are V4.1-specific; expert
execution and tensor-parallel linear layers use the existing Ascend backend.
"""

from contextlib import contextmanager
from pathlib import Path

import regex as re
import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn
from transformers import AutoTokenizer
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsEagle3, SupportsMultiModal
from vllm.model_executor.models.utils import (
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.models.common.ops.sequence_parallel import sp_all_gather
from vllm_ascend.models.deepseek_v4.model import DeepseekV2MixtureOfExperts, DeepseekV4MoE
from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method
from vllm_ascend.patch.worker.deepseek_v41_attention import DeepseekV41Attention, DeepseekV41AttentionContext
from vllm_ascend.patch.worker.deepseek_v41_attention_math import DeepseekV41Rotary
from vllm_ascend.patch.worker.deepseek_v41_cache import DeepseekV41TokenHistory
from vllm_ascend.patch.worker.deepseek_v41_cache_plan import build_attention_cache_plan
from vllm_ascend.patch.worker.deepseek_v41_compressor import DeepseekV41RMSNorm
from vllm_ascend.patch.worker.deepseek_v41_decoder import DeepseekV41DecoderLayer, hc_collapse
from vllm_ascend.patch.worker.deepseek_v41_embedding import ShardedEngramEmbedding
from vllm_ascend.patch.worker.deepseek_v41_engram import EngramHasher, build_engram_token_map
from vllm_ascend.patch.worker.deepseek_v41_engram_layer import DeepseekV41Engram
from vllm_ascend.patch.worker.deepseek_v41_processor import (
    IMAGE_PLACEHOLDER,
    DeepseekV41DummyInputsBuilder,
    DeepseekV41MultiModalProcessor,
    DeepseekV41ProcessingInfo,
)
from vllm_ascend.patch.worker.deepseek_v41_router import DeepseekV41ImageRouter
from vllm_ascend.patch.worker.deepseek_v41_vision import DeepseekV41Aligner, DeepseekV41Vision, make_image_span


def capture_moe_communication():
    # The base runner replaces its process-wide communication table each time
    # a MoE layer is constructed. V4.1 target and draft have different expert
    # counts (384/128) and top-k (6/3), so retain each model's own table.
    return {kind: get_moe_comm_method(kind) for kind in MoECommType}


@contextmanager
def use_moe_communication(methods):
    previous = _EXTRA_CTX.moe_comm_method
    _EXTRA_CTX.moe_comm_method = methods[_EXTRA_CTX.moe_comm_type]
    try:
        yield
    finally:
        _EXTRA_CTX.moe_comm_method = previous


class DeepseekV41MoE(DeepseekV4MoE):
    def __init__(self, config, parallel_config, quant_config=None, prefix=""):
        super().__init__(config, parallel_config, quant_config, prefix)
        # Keep the checkpoint's image selection bias distinct from text bias.
        self.gate.bias_vl = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32), requires_grad=False)
        router = DeepseekV41ImageRouter(self.experts.router, self.gate.bias_vl)
        self.experts.router = router
        self.experts.routed_experts.router = router

    def forward(self, hidden_states, *, image_mask=None):
        if not self.is_sequence_parallel:
            return super().forward(hidden_states, input_ids=image_mask)
        # Paged attention and delayed mHC retain full DP-local token order.
        # The sequence-parallel expert runner instead consumes one TP shard;
        # keep its image-routing side channel on the identical token shard.
        num_tokens = hidden_states.shape[0]
        hidden_states = sequence_parallel_chunk(hidden_states)
        if image_mask is not None:
            image_mask = sequence_parallel_chunk(image_mask.reshape(-1, 1)).reshape(-1)
        output = super().forward(hidden_states, input_ids=image_mask)
        return sp_all_gather(output)[:num_tokens]


class DeepseekV41Model(nn.Module):
    def __init__(self, *, vllm_config, prefix):
        super().__init__()
        self.config = config = vllm_config.model_config.hf_text_config
        composite = vllm_config.model_config.hf_config
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("V4.1 cross-layer cache ownership requires pipeline_parallel_size=1")
        self.start_layer, self.end_layer = 0, config.num_hidden_layers
        self.embed = VocabParallelEmbedding(config.vocab_size, config.hidden_size, prefix=prefix + ".embed")
        self.norm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
        device = self.embed.weight.device
        self.window_rotary = DeepseekV41Rotary(config, vllm_config.model_config.max_model_len, compressed=False)
        self.compressed_rotary = DeepseekV41Rotary(config, vllm_config.model_config.max_model_len, compressed=True)
        self.ep_group = get_ep_group()
        self.aux_hidden_state_layers = ()
        rotation_config = getattr(composite, "engram_rotation_config", {})
        rotation = None
        if rotation_config.get("value_projection_rotated", False):
            with safe_open(
                Path(vllm_config.model_config.model) / "optional/quarot.safetensors", framework="pt"
            ) as handle:
                rotation = handle.get_tensor("global_rotation").to(device=device, dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(vllm_config.model_config.tokenizer, trust_remote_code=False)
        token_map = build_engram_token_map(tokenizer, config.engram_compressed_vocab_size)
        self.hasher = EngramHasher(
            config, config.engram_compressed_vocab_size, int(token_map[config.engram_pad_token_id])
        )
        self.history = DeepseekV41TokenHistory(
            # Cache binding orders entries by a numeric decoder-layer index.
            # History is written before layer 0 and shared by both Engrams.
            vllm_config,
            token_map.to(device),
            config.engram_max_ngram_size,
            prefix + ".layers.0.history",
        )
        self.layers = nn.ModuleList()
        for owner in build_attention_cache_plan(config):
            layer_prefix = f"{prefix}.layers.{owner.layer_id}"
            attention = DeepseekV41Attention(
                vllm_config,
                config,
                owner,
                layer_prefix + ".attn",
                self.compressed_rotary if owner.compress_ratio else self.window_rotary,
            )
            ffn = DeepseekV41MoE(config, vllm_config.parallel_config, vllm_config.quant_config, layer_prefix + ".ffn")
            layer = DeepseekV41DecoderLayer(config, attention, ffn)
            layer.engram = None
            if owner.layer_id in config.engram_layer_ids:
                hash_index = config.engram_layer_ids.index(owner.layer_id)
                table = ShardedEngramEmbedding(
                    config.engram_num_embeddings[hash_index],
                    config.engram_head_dim,
                    rank=self.ep_group.rank_in_group,
                    world_size=self.ep_group.world_size,
                    device=device,
                    quant_type="INT8_SYM",
                )
                layer.engram = DeepseekV41Engram(
                    config, table, device=device, rotation=rotation, rotation_config=rotation_config
                )
                layer.engram.layer_hash_index = hash_index
            self.layers.append(layer)

    def embed_input_ids(self, input_ids):
        return self.embed(input_ids)

    @property
    def embed_tokens(self):
        # The common speculative proposer discovers the target embedding here.
        return self.embed

    def forward(self, input_ids, positions, inputs_embeds=None, *, image_mask=None):
        hidden = self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        hidden = hidden.unsqueeze(1).expand(-1, self.config.hc_mult, -1).contiguous()
        pre_mix = hidden.new_zeros(hidden.shape[:2], dtype=torch.float32)
        pre_mix[:, 0] = 1
        context = get_forward_context()
        history = self.history(input_ids, positions, image_mask)
        hashes = self.hasher(history)
        # DP batches differ. Only the Engram collective uses a common shape;
        # token padding comes from the existing host-side DP synchronization.
        common_tokens = getattr(context, "max_tokens_across_dp", None) or hidden.shape[0]
        common_tokens = max(int(common_tokens), hidden.shape[0])
        padded_hashes = F.pad(hashes, (0, 0, 0, 0, 0, common_tokens - hidden.shape[0]), value=-1)
        attention_context = DeepseekV41AttentionContext()
        auxiliary = []
        for layer_id, layer in enumerate(self.layers):
            if layer.engram is not None:
                # Engram's all-gather/reduce-scatter also includes padded ranks.
                padded_hidden = F.pad(hidden, (0, 0, 0, 0, 0, common_tokens - hidden.shape[0]))
                token_mask = torch.arange(common_tokens, device=hidden.device) < hidden.shape[0]
                if image_mask is not None:
                    token_mask = token_mask & ~F.pad(image_mask, (0, common_tokens - hidden.shape[0]))
                hidden = layer.engram(
                    padded_hidden,
                    padded_hashes[:, layer.engram.layer_hash_index],
                    token_mask=token_mask,
                    group=self.ep_group,
                )[: hidden.shape[0]]
            if layer_id in self.aux_hidden_state_layers:
                # Official DSpark reads the post-Engram, pre-attention mean.
                auxiliary.append(hidden.mean(dim=1))
            hidden, pre_mix = layer(
                positions, hidden, pre_mix, image_mask=image_mask, attention_context=attention_context
            )
        hidden = self.norm(hc_collapse(hidden, pre_mix))
        return (hidden, auxiliary) if self.aux_hidden_state_layers else hidden


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV41MultiModalProcessor,
    info=DeepseekV41ProcessingInfo,
    dummy_inputs=DeepseekV41DummyInputsBuilder,
)
class DeepseekV41ForCausalLM(nn.Module, DeepseekV2MixtureOfExperts, SupportsEagle3, SupportsMultiModal):
    requires_raw_input_tokens = True
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "head.": "lm_head.",
        },
        orig_to_new_regex={
            re.compile(r"(\.ffn\.(?:experts\.\d+|shared_experts))\.w1\."): r"\1.gate_proj.",
            re.compile(r"(\.ffn\.(?:experts\.\d+|shared_experts))\.w2\."): r"\1.down_proj.",
            re.compile(r"(\.ffn\.(?:experts\.\d+|shared_experts))\.w3\."): r"\1.up_proj.",
        },
    )
    checkpoint_prefixes = ("layers.", "embed.", "norm.", "head.")

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.config = vllm_config.model_config.hf_text_config
        composite = vllm_config.model_config.hf_config
        self.image_token_id = composite.image_token_id
        self.vision = None
        self.aligner = None
        if vllm_config.model_config.multimodal_config.get_limit_per_prompt("image") > 0:
            self.vision = DeepseekV41Vision(composite.vision_config)
            self.aligner = DeepseekV41Aligner(composite.vision_config, self.config.hidden_size)
            self.image_start = nn.Parameter(torch.empty(self.config.hidden_size), requires_grad=False)
            self.image_newline = nn.Parameter(torch.empty(self.config.hidden_size), requires_grad=False)
            self.image_end = nn.Parameter(torch.empty(self.config.hidden_size), requires_grad=False)
            self.checkpoint_prefixes = (*self.checkpoint_prefixes, "vision.", "aligner.", "image_")
        self.model = DeepseekV41Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            params_dtype=torch.float32,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.moe_mlp_layers = [layer.ffn for layer in self.model.layers]
        self.moe_layers = [layer.experts for layer in self.moe_mlp_layers]
        self.expert_weights = []
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.extract_moe_parameters(self.moe_mlp_layers[0])
        self.moe_comm_methods = capture_moe_communication()

    @classmethod
    def get_placeholder_str(cls, modality, i):
        if modality.startswith("image"):
            return IMAGE_PLACEHOLDER
        raise ValueError(f"V4.1 does not support modality {modality}")

    def get_language_model(self):
        return self

    def embed_multimodal(self, **kwargs):
        patches = kwargs.get("pixel_values")
        if patches is None:
            return []
        if self.vision is None:
            raise ValueError("Image inputs are disabled by limit_mm_per_prompt")
        grids = kwargs["image_grid_hw"]
        output = []
        for pixels, grid in zip(patches, grids, strict=True):
            rows, columns = grid.tolist()  # Processor retains small grids on CPU.
            hidden = self.vision(pixels, rows, columns)
            features = self.aligner(hidden, rows, columns)
            output.append(
                make_image_span(
                    features, rows, columns, self.aligner.ratio, self.image_start, self.image_newline, self.image_end
                )
            )
        return output

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        hidden = self.model.embed_input_ids(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return hidden
        if is_multimodal is None:
            raise ValueError("Image embeddings require an explicit multimodal token mask")
        return _merge_multimodal_embeddings(hidden, multimodal_embeddings, is_multimodal)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if intermediate_tensors is not None:
            raise ValueError("V4.1 does not accept pipeline intermediate tensors")
        with use_moe_communication(self.moe_comm_methods):
            image_mask = (
                input_ids == self.image_token_id if self.vision is not None and inputs_embeds is not None else None
            )
            return self.model(input_ids, positions, inputs_embeds, image_mask=image_mask)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    def set_aux_hidden_state_layers(self, layers):
        # vLLM's Eagle interface numbers states after the embedding (layer+1).
        # The official DSpark IDs name pre-attention decoder inputs directly.
        self.model.aux_hidden_state_layers = tuple(layer - 1 for layer in layers)

    def get_eagle3_default_aux_hidden_state_layers(self):
        return tuple(layer + 1 for layer in self.config.dspark_target_layer_ids)

    def get_expert_mapping(self):
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_checkpoint_tables(self, checkpoint):
        for layer_id in self.config.engram_layer_ids:
            checkpoint.load_engram(layer_id, self.model.layers[layer_id].engram.embed)

    def load_weights(self, weights):
        parameters = dict(self.named_parameters())
        expert_mapping = self.get_expert_mapping()
        loaded = set()
        for name, value in self.hf_to_vllm_mapper.apply(weights):
            if ".gate.bias" in name and not name.endswith(".bias_vl"):
                name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
            if ".experts." in name:
                for parameter_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name not in name:
                        continue
                    mapped = name.replace(weight_name, parameter_name)
                    parameter = parameters[mapped]
                    if parameter.weight_loader(
                        parameter, value, mapped, shard_id=shard_id, expert_id=expert_id, return_success=True
                    ):
                        loaded.add(mapped)
                        break
                else:
                    # An EP rank intentionally has no storage for remote experts.
                    continue
            elif ".shared_experts.gate_proj." in name or ".shared_experts.up_proj." in name:
                shard = 0 if ".gate_proj." in name else 1
                mapped = name.replace(".gate_proj.", ".gate_up_proj.").replace(".up_proj.", ".gate_up_proj.")
                parameter = parameters[mapped]
                parameter.weight_loader(parameter, value, shard)
                loaded.add(mapped)
            else:
                parameter = parameters[name]
                getattr(parameter, "weight_loader", default_weight_loader)(parameter, value)
                loaded.add(name)
        missing = set(parameters) - loaded
        if missing:
            raise ValueError(f"V4.1 checkpoint did not initialize parameters: {sorted(missing)}")
        return loaded
