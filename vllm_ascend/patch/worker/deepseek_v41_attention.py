# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged V4.1 attention with explicit per-forward cross-layer ownership."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size, get_tp_group
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear

from vllm_ascend.patch.worker.deepseek_v41_attention_math import fake_quant_fp4, fake_quant_fp8, sparse_attention
from vllm_ascend.patch.worker.deepseek_v41_cache import DeepseekV41PagedCache, read_paged, write_paged
from vllm_ascend.patch.worker.deepseek_v41_compressor import DeepseekV41Compressor, DeepseekV41RMSNorm
from vllm_ascend.patch.worker.deepseek_v41_linear import apply_linear, grouped_output_projection
from vllm_ascend.utils import enable_dsa_cp

QUERY_CHUNK_SIZE = 16


@dataclass
class DeepseekV41AttentionContext:
    """Created anew for each forward, never shared across concurrent calls."""

    kv_owner: int | None = None
    index_owner: int | None = None
    compressed_kv: tuple | None = None
    index_keys: tuple | None = None
    topk: torch.Tensor | None = None
    candidates: torch.Tensor | None = None


def load_tp_rows(parameter, loaded_weight):
    parameter.data.copy_(loaded_weight.narrow(0, parameter.tp_row_start, parameter.shape[0]))


def partition_context_tokens(value, rank, size, *, padding_value=0):
    """Take an equally padded token shard without changing trailing axes."""
    shape = value.shape
    flat = value.reshape(shape[0], -1)
    padded = F.pad(flat, (0, 0, 0, -shape[0] % size), value=padding_value)
    count = padded.shape[0] // size
    return padded.narrow(0, rank * count, count).reshape(count, *shape[1:]).contiguous()


def select_candidates(scores, visible_lengths, *, block_size, topk_blocks):
    """Keep highest scoring blocks and always pin the newest visible block."""
    grouped = F.pad(scores, (0, -scores.shape[-1] % block_size), value=-torch.inf)
    grouped = grouped.unflatten(-1, (-1, block_size)).amax(-1)
    last = (visible_lengths - 1) // block_size
    blocks = torch.arange(grouped.shape[-1], device=scores.device)
    grouped = grouped.masked_fill(blocks.unsqueeze(0) == last.unsqueeze(-1), torch.inf)
    values, indices = grouped.topk(min(topk_blocks, grouped.shape[-1]), dim=-1)
    return torch.zeros_like(grouped, dtype=torch.bool).scatter_(-1, indices, values > -torch.inf)


def select_positions(scores, count):
    width = min(count, scores.shape[-1])
    indices = scores.topk(width, dim=-1, sorted=False).indices.sort(dim=-1).values
    reachable = torch.isfinite(scores.gather(-1, indices))
    return F.pad(torch.where(reachable, indices, -1), (0, count - width), value=-1)


class DeepseekV41Indexer(nn.Module):
    def __init__(self, vllm_config, config, owner, prefix, rotary):
        super().__init__()
        self.owner = owner
        self.config = config
        self.rotary = rotary
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.context_parallel = self.tp_size > 1 and enable_dsa_cp()
        self.heads = config.index_n_heads // self.tp_size
        self.wq_b = ColumnParallelLinear(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=prefix + ".wq_b",
            return_bias=False,
        )
        self.weights_proj = ColumnParallelLinear(
            config.hidden_size,
            config.index_n_heads,
            bias=False,
            quant_config=None,
            prefix=prefix + ".weights_proj",
            return_bias=False,
        )
        if owner.layer_id == owner.kv_layer_id:
            self.wk = ReplicatedLinear(
                config.head_dim,
                config.index_head_dim,
                bias=False,
                quant_config=None,
                prefix=prefix + ".wk",
                return_bias=False,
            )
            self.k_norm = DeepseekV41RMSNorm(config.index_head_dim, config.rms_norm_eps)
            self.cache = DeepseekV41PagedCache(
                vllm_config,
                prefix + ".cache",
                config.index_head_dim,
                torch.bfloat16,
                compress_ratio=owner.compress_ratio,
            )

    def forward(self, hidden, qr, positions, latent, complete, context):
        """Publish replicated keys and select candidates for this query shard."""
        ratio = self.owner.compress_ratio
        if self.owner.layer_id == self.owner.kv_layer_id:
            cache, metadata = self.cache()
            key = self.k_norm(apply_linear(self.wk, latent))
            key = fake_quant_fp4(self.rotary(key, positions + 1 - ratio))
            slots = torch.where(complete & (metadata.slot_mapping >= 0), metadata.slot_mapping // ratio, -1)
            write_paged(cache, slots, key)
            context.index_keys = (cache, metadata)
        if context.index_keys is None or context.kv_owner != self.owner.kv_layer_id:
            raise RuntimeError("V4.1 indexer read before its KV owner published keys")
        cache, metadata = context.index_keys
        q = apply_linear(self.wq_b, qr).unflatten(-1, (self.heads, self.config.index_head_dim))
        q = fake_quant_fp4(self.rotary(q, positions))
        weights = apply_linear(self.weights_proj, hidden) * (
            self.config.index_head_dim**-0.5 * self.config.index_n_heads**-0.5
        )
        requests = metadata.request_indices
        if self.context_parallel:
            q = partition_context_tokens(get_tp_group().all_gather(q, dim=1), self.tp_rank, self.tp_size)
            weights = partition_context_tokens(get_tp_group().all_gather(weights, dim=1), self.tp_rank, self.tp_size)
            positions = partition_context_tokens(positions, self.tp_rank, self.tp_size)
            requests = partition_context_tokens(requests, self.tp_rank, self.tp_size, padding_value=-1)
        extent = max(1, metadata.max_seq_len // ratio)
        compressed_positions = torch.arange(extent, device=positions.device)
        selected, candidates = [], []
        for start in range(0, q.shape[0], QUERY_CHUNK_SIZE):
            end = min(start + QUERY_CHUNK_SIZE, q.shape[0])
            requested = requests[start:end]
            key_positions = (compressed_positions * ratio).expand(end - start, -1)
            keys, readable = read_paged(
                cache,
                key_positions,
                requested,
                metadata.block_table,
                logical_block_size=metadata.logical_block_size,
                compress_ratio=ratio,
            )
            # Match the reference BF16 einsum, weight product and head sum.
            scores = (torch.bmm(q[start:end], keys.transpose(1, 2)).relu() * weights[start:end, :, None]).sum(1)
            if self.tp_size > 1 and not self.context_parallel:
                scores = get_tp_group().all_reduce(scores)
            visible = (positions[start:end] + 1) // ratio
            valid = readable & (compressed_positions[None, :] < visible[:, None])
            scores = scores.float().masked_fill(~valid, -torch.inf)
            if self.owner.layer_id == self.config.candidate_source_layer_id:
                candidates.append(
                    select_candidates(
                        scores,
                        visible,
                        block_size=self.config.candidate_block_size,
                        topk_blocks=self.config.candidate_topk_blocks,
                    )
                )
            elif self.owner.candidate_layer_id is not None:
                if context.candidates is None:
                    raise RuntimeError("V4.1 indexer read before candidate blocks were published")
                selected_blocks = context.candidates[
                    start:end, compressed_positions // self.config.candidate_block_size
                ]
                scores = scores.masked_fill(~selected_blocks, -torch.inf)
            selected.append(select_positions(scores, self.config.index_topk))
        if candidates:
            context.candidates = torch.cat(candidates)
        context.index_owner = self.owner.layer_id
        context.topk = torch.cat(selected)


class DeepseekV41Attention(nn.Module):
    def __init__(self, vllm_config, config, owner, prefix, rotary, *, cache_window=None):
        super().__init__()
        self.owner = owner
        self.config = config
        self.rotary = rotary
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.context_parallel = self.tp_size > 1 and enable_dsa_cp()
        self.heads = config.num_attention_heads // self.tp_size
        self.groups = config.o_groups // self.tp_size
        if not self.groups or config.o_groups % self.tp_size:
            raise ValueError("Tensor parallel size must divide the attention output groups")
        linear_kwargs = dict(bias=False, quant_config=vllm_config.quant_config, return_bias=False)
        self.wq_a = ReplicatedLinear(config.hidden_size, config.q_lora_rank, prefix=prefix + ".wq_a", **linear_kwargs)
        self.wq_b = ColumnParallelLinear(
            config.q_lora_rank, config.num_attention_heads * config.head_dim, prefix=prefix + ".wq_b", **linear_kwargs
        )
        self.wkv = ReplicatedLinear(config.hidden_size, config.head_dim, prefix=prefix + ".wkv", **linear_kwargs)
        self.q_norm = DeepseekV41RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.kv_norm = DeepseekV41RMSNorm(config.head_dim, config.rms_norm_eps)
        self.wo_a = nn.Linear(
            config.num_attention_heads * config.head_dim // config.o_groups,
            self.groups * config.o_lora_rank,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.wo_a.weight.tp_row_start = self.tp_rank * self.wo_a.weight.shape[0]
        self.wo_a.weight.weight_loader = load_tp_rows
        self.wo_b = RowParallelLinear(
            config.o_groups * config.o_lora_rank, config.hidden_size, prefix=prefix + ".wo_b", **linear_kwargs
        )
        sink_heads = config.num_attention_heads if self.context_parallel else self.heads
        self.attn_sink = nn.Parameter(torch.empty(sink_heads, dtype=torch.float32), requires_grad=False)
        self.attn_sink.tp_row_start = 0 if self.context_parallel else self.tp_rank * self.heads
        self.attn_sink.weight_loader = load_tp_rows
        self.window_cache = DeepseekV41PagedCache(
            vllm_config,
            prefix + ".window_cache",
            config.head_dim,
            torch.bfloat16,
            sliding_window=config.sliding_window if cache_window is None else cache_window,
        )
        if owner.layer_id == owner.kv_layer_id:
            self.compressor = DeepseekV41Compressor(vllm_config, config, owner.compress_ratio, prefix + ".compressor")
            self.compressed_cache = DeepseekV41PagedCache(
                vllm_config,
                prefix + ".compressed_cache",
                config.head_dim,
                torch.bfloat16,
                compress_ratio=owner.compress_ratio,
            )
        if owner.layer_id == owner.index_layer_id:
            self.indexer = DeepseekV41Indexer(vllm_config, config, owner, prefix + ".indexer", rotary)

    def forward(self, positions, hidden, context):
        """Keep paged writes replicated; DSA-CP distributes query tokens.

        Linear weights stay TP sharded. Only projected query heads and the
        attention result cross the TP group, so CP does not replicate the
        large query/output projection weights or alter cache ownership.
        """
        qr = self.q_norm(apply_linear(self.wq_a, hidden))
        query = self.rotary(apply_linear(self.wq_b, qr).unflatten(-1, (self.heads, self.config.head_dim)), positions)
        window_kv = fake_quant_fp8(self.rotary(self.kv_norm(apply_linear(self.wkv, hidden)), positions))
        window_cache, window_metadata = self.window_cache()
        if window_metadata is None:
            # Initial memory profiling has no allocated KV cache. Still run
            # the large projections so their activation memory is measured.
            return self.project_output(query, positions)
        write_paged(window_cache, window_metadata.slot_mapping, window_kv)
        ratio = self.owner.compress_ratio
        if ratio:
            latent = complete = None
            if self.owner.layer_id == self.owner.kv_layer_id:
                latent, complete = self.compressor(hidden, positions)
                context.kv_owner = self.owner.layer_id
                context.compressed_kv = self.compressed_cache()
            if self.owner.layer_id == self.owner.index_layer_id:
                self.indexer(hidden, qr, positions, latent, complete, context)
            if context.kv_owner != self.owner.kv_layer_id or context.index_owner != self.owner.index_layer_id:
                raise RuntimeError("V4.1 compressed attention cache ownership mismatch")
            if latent is not None:
                cache, metadata = context.compressed_kv
                compressed = fake_quant_fp4(self.rotary(latent, positions + 1 - ratio), group_size=16, e4m3_scale=True)
                slots = torch.where(complete & (metadata.slot_mapping >= 0), metadata.slot_mapping // ratio, -1)
                write_paged(cache, slots, compressed)

        query_positions = positions
        window_requests = window_metadata.request_indices
        compressed_requests = context.compressed_kv[1].request_indices if ratio else None
        if self.context_parallel:
            query = partition_context_tokens(get_tp_group().all_gather(query, dim=1), self.tp_rank, self.tp_size)
            query_positions = partition_context_tokens(positions, self.tp_rank, self.tp_size)
            window_requests = partition_context_tokens(window_requests, self.tp_rank, self.tp_size, padding_value=-1)
            if compressed_requests is not None:
                compressed_requests = partition_context_tokens(
                    compressed_requests, self.tp_rank, self.tp_size, padding_value=-1
                )
        outputs = []
        for start in range(0, query.shape[0], QUERY_CHUNK_SIZE):
            end = min(start + QUERY_CHUNK_SIZE, query.shape[0])
            window_positions = query_positions[start:end, None] + torch.arange(
                1 - self.config.sliding_window, 1, device=positions.device
            )
            selected, valid = read_paged(
                window_cache,
                window_positions,
                window_requests[start:end],
                window_metadata.block_table,
                logical_block_size=window_metadata.logical_block_size,
            )
            if ratio:
                cache, metadata = context.compressed_kv
                selected_positions = context.topk[start:end] * ratio
                compressed, reachable = read_paged(
                    cache,
                    selected_positions,
                    compressed_requests[start:end],
                    metadata.block_table,
                    logical_block_size=metadata.logical_block_size,
                    compress_ratio=ratio,
                )
                selected = torch.cat((selected, compressed), dim=1)
                valid = torch.cat((valid, reachable), dim=1)
            outputs.append(
                sparse_attention(query[start:end], selected, valid, self.attn_sink, self.config.head_dim**-0.5)
            )
        output = torch.cat(outputs)
        if self.context_parallel:
            output = get_tp_group().all_gather(output, dim=0)[: hidden.shape[0]]
            output = output[:, self.tp_rank * self.heads : (self.tp_rank + 1) * self.heads].contiguous()
        return self.project_output(output, positions)

    def project_output(self, output, positions):
        output = self.rotary(output, positions, inverse=True)
        output = fake_quant_fp8(output.reshape(output.shape[0], self.groups, -1))
        weight = self.wo_a.weight.view(self.groups, self.config.o_lora_rank, -1)
        projected = grouped_output_projection(output, weight)
        return apply_linear(self.wo_b, projected.flatten(1))
