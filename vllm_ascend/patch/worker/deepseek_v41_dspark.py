# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 block drafter using the common Ascend DSpark execution contract.

The pinned official HF forward defines delayed mHC, context KV projection,
noncausal draft blocks, and FP32 Markov/confidence heads. Sampling and rejected
token handling stay in the existing proposer and scheduler.
"""

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix

from vllm_ascend.models.deepseek_v4.dspark import DSparkDeepseekV4ForCausalLM
from vllm_ascend.patch.worker.deepseek_v41_attention import (
    QUERY_CHUNK_SIZE,
    DeepseekV41Attention,
    partition_context_tokens,
)
from vllm_ascend.patch.worker.deepseek_v41_attention_math import DeepseekV41Rotary, fake_quant_fp8, sparse_attention
from vllm_ascend.patch.worker.deepseek_v41_cache import read_paged, write_paged
from vllm_ascend.patch.worker.deepseek_v41_cache_plan import AttentionCacheOwner
from vllm_ascend.patch.worker.deepseek_v41_compressor import DeepseekV41RMSNorm
from vllm_ascend.patch.worker.deepseek_v41_decoder import DeepseekV41DecoderLayer, hc_collapse
from vllm_ascend.patch.worker.deepseek_v41_linear import apply_linear
from vllm_ascend.patch.worker.deepseek_v41_model import (
    DeepseekV41ForCausalLM,
    DeepseekV41MoE,
    capture_moe_communication,
    use_moe_communication,
)


def draft_key_positions(positions, window_size, block_size):
    """Use the reference's circular context order, followed by all draft keys.

    Pack short contexts before the draft tail. Leading masked keys change the
    online softmax's chunk boundaries and therefore its BF16 probability rounding.
    """
    query_offset = torch.arange(positions.shape[0], device=positions.device) % block_size
    first_query = positions - query_offset
    count = first_query.clamp(0, window_size)[:, None]
    offsets = torch.arange(window_size + block_size, device=positions.device)
    latest = first_query[:, None] - 1
    context_positions = latest - (latest - offsets).remainder(window_size)
    draft_positions = first_query[:, None] + offsets - count
    keys = torch.where(offsets < count, context_positions, draft_positions)
    return torch.where(offsets < count + block_size, keys, -1)


class DeepseekV41DSparkAttention(DeepseekV41Attention):
    def __init__(self, vllm_config, config, layer_id, prefix, rotary):
        owner = AttentionCacheOwner(layer_id, 0, None, None, None)
        super().__init__(
            vllm_config,
            config,
            owner,
            prefix,
            rotary,
            # Reserving the draft tail must not evict the oldest context token.
            cache_window=config.sliding_window + config.dspark_block_size,
        )

    def project_context(self, hidden, positions):
        return fake_quant_fp8(self.rotary(self.kv_norm(apply_linear(self.wkv, hidden)), positions))

    def forward(self, positions, hidden, context=None):
        qr = self.q_norm(apply_linear(self.wq_a, hidden))
        query = self.rotary(apply_linear(self.wq_b, qr).unflatten(-1, (self.heads, self.config.head_dim)), positions)
        draft_kv = self.project_context(hidden, positions)
        cache, metadata = self.window_cache()
        if metadata is None:
            return self.project_output(query, positions)
        write_paged(cache, metadata.slot_mapping, draft_kv)
        # Derive block membership before CP slicing: a rank's first query
        # may lie inside a draft block rather than at its first token.
        keys = draft_key_positions(positions, self.config.sliding_window, self.config.dspark_block_size)
        requests = metadata.request_indices
        if self.context_parallel:
            query = partition_context_tokens(get_tp_group().all_gather(query, dim=1), self.tp_rank, self.tp_size)
            keys = partition_context_tokens(keys, self.tp_rank, self.tp_size, padding_value=-1)
            requests = partition_context_tokens(requests, self.tp_rank, self.tp_size, padding_value=-1)
        outputs = []
        for start in range(0, query.shape[0], QUERY_CHUNK_SIZE):
            end = min(start + QUERY_CHUNK_SIZE, query.shape[0])
            selected, valid = read_paged(
                cache,
                keys[start:end],
                requests[start:end],
                metadata.block_table,
                logical_block_size=metadata.logical_block_size,
            )
            outputs.append(
                sparse_attention(query[start:end], selected, valid, self.attn_sink, self.config.head_dim**-0.5)
            )
        output = torch.cat(outputs)
        if self.context_parallel:
            output = get_tp_group().all_gather(output, dim=0)[: hidden.shape[0]]
            output = output[:, self.tp_rank * self.heads : (self.tp_rank + 1) * self.heads].contiguous()
        return self.project_output(output, positions)


class DeepseekV41MarkovHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Replicate the small-rank head to avoid collectives at each Markov step.
        self.embed = nn.Embedding(config.vocab_size, config.dspark_markov_rank)
        self.head = nn.Linear(config.dspark_markov_rank, config.vocab_size, bias=False, dtype=torch.float32)

    def bias(self, embedding):
        return self.head(embedding.float())


class DeepseekV41ConfidenceHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size + config.dspark_markov_rank, 1, bias=False, dtype=torch.float32)

    def forward(self, hidden, embedding):
        return self.proj(torch.cat((hidden, embedding), dim=-1).float()).squeeze(-1)


class DeepseekV41DSparkModel(nn.Module):
    def __init__(self, *, vllm_config, prefix):
        super().__init__()
        self.config = config = vllm_config.speculative_config.draft_model_config.hf_config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=prefix + ".embed_tokens"
        )
        rotary = DeepseekV41Rotary(config, vllm_config.model_config.max_model_len, compressed=False)
        self.mtp = nn.ModuleList()
        for stage in range(config.n_mtp_layers):
            layer_prefix = f"{prefix}.mtp.{stage}"
            attention = DeepseekV41DSparkAttention(vllm_config, config, stage, layer_prefix + ".attn", rotary)
            ffn = DeepseekV41MoE(config, vllm_config.parallel_config, vllm_config.quant_config, layer_prefix + ".ffn")
            self.mtp.append(DeepseekV41DecoderLayer(config, attention, ffn))
        first, last = self.mtp[0], self.mtp[-1]
        first.main_proj = nn.Linear(
            config.hidden_size * len(config.dspark_target_layer_ids), config.hidden_size, bias=False
        )
        first.main_norm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
        last.norm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
        last.markov_head = DeepseekV41MarkovHead(config)
        last.confidence_head = DeepseekV41ConfidenceHead(config)

    @property
    def confidence_head(self):
        return self.mtp[-1].confidence_head

    def get_draft_kv_cache_layer_names(self):
        return [layer.attn.window_cache.prefix for layer in self.mtp]

    def combine_hidden_states(self, hidden):
        first = self.mtp[0]
        return first.main_norm(apply_linear(first.main_proj, hidden))

    def precompute_and_store_context_kv(self, context_states, context_positions, context_slot_mapping=None):
        if context_states.numel() == 0 or context_slot_mapping is None:
            return
        for layer, slots in zip(self.mtp, context_slot_mapping, strict=True):
            if slots is not None:
                kv = layer.attn.project_context(context_states, context_positions)
                write_paged(layer.attn.window_cache.kv_cache, slots, kv)

    def forward(self, input_ids, positions):
        hidden = self.embed_tokens(input_ids).unsqueeze(1).expand(-1, self.config.hc_mult, -1).contiguous()
        pre_mix = hidden.new_zeros(hidden.shape[:2], dtype=torch.float32)
        pre_mix[:, 0] = 1
        for layer in self.mtp:
            hidden, pre_mix = layer(positions, hidden, pre_mix)
        return hc_collapse(hidden, pre_mix)

    def compute_logits(self, hidden, lm_head, logits_processor):
        return logits_processor(lm_head, self.mtp[-1].norm(hidden).float())

    def markov_embed(self, token_ids):
        return self.mtp[-1].markov_head.embed(token_ids)

    def markov_bias(self, embedding):
        return self.mtp[-1].markov_head.bias(embedding)


# The common Ascend DSpark proposer deliberately runs its draft eagerly.
# Keep the inherited call wrapper initialized without Dynamo serializing the
# model's live MoE communication handles; target ACLGraph remains independent.
@support_torch_compile(enable_if=lambda _: False)
class DeepseekV41DSparkForCausalLM(DSparkDeepseekV4ForCausalLM):
    """Reuse the V4 proposer interface while replacing its decoder mathematics."""

    packed_modules_mapping = DeepseekV41ForCausalLM.packed_modules_mapping
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "mtp.0.embed.": "model.embed_tokens.",
            "mtp.2.head.": "lm_head.",
            "mtp.": "model.mtp.",
        },
        orig_to_new_substr={".w1.": ".gate_proj.", ".w2.": ".down_proj.", ".w3.": ".up_proj."},
    )
    checkpoint_prefixes = ("mtp.",)

    def __init__(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.model = DeepseekV41DSparkModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            params_dtype=torch.float32,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.moe_mlp_layers = [layer.ffn for layer in self.model.mtp]
        self.moe_layers = [layer.experts for layer in self.moe_mlp_layers]
        self.expert_weights = []
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.extract_moe_parameters(self.moe_mlp_layers[0])
        self.moe_comm_methods = capture_moe_communication()

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, inputs_embeds: torch.Tensor | None = None
    ) -> torch.Tensor:
        with use_moe_communication(self.moe_comm_methods):
            return self.model(input_ids, positions)

    def get_expert_mapping(self):
        return DeepseekV41ForCausalLM.get_expert_mapping(self)

    def load_checkpoint_tables(self, checkpoint):
        # The draft has no Engram tables; mtp.* is independently indexed.
        pass

    def load_weights(self, weights):
        loaded = DeepseekV41ForCausalLM.load_weights(self, weights)
        # SupportsEagle3 defaults these flags to False. Without updating them,
        # the proposer replaces the freshly loaded MTP weights with the target's.
        self.has_own_embed_tokens = "model.embed_tokens.weight" in loaded
        self.has_own_lm_head = "lm_head.weight" in loaded
        return loaded
