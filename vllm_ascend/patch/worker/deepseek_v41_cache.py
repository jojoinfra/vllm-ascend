# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-owned paged state for the official V4.1 model.

Block zero is vLLM BlockPool's permanent null block. Invalid/padded writes
target only that block; valid reads never expose its contents. Indexing keeps
the block dimension explicit, including when hybrid cache pages are padded.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend, AttentionCGSupport, AttentionMetadataBuilder

from vllm_ascend.core.kv_cache_interface import AscendSlidingTokenCacheSpec, AscendTokenCacheSpec


def write_paged(cache: torch.Tensor, slots: torch.Tensor, values: torch.Tensor) -> None:
    """Write [tokens, width] to [blocks, physical_rows, 1, width]."""
    block_size = cache.shape[1]
    valid = (slots >= block_size) & (slots < cache.shape[0] * block_size)
    safe = torch.where(valid, slots, 0).long()
    cache[safe // block_size, safe % block_size, 0] = torch.where(valid[:, None], values, 0)


def read_paged(
    cache: torch.Tensor,
    positions: torch.Tensor,
    requests: torch.Tensor,
    block_table: torch.Tensor,
    *,
    logical_block_size: int,
    compress_ratio: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather logical token positions; return values and a validity mask.

    ``positions`` is [tokens, lookback] and ``requests`` is [tokens]. The
    caller additionally masks future/uncompleted compressed groups.
    """
    logical_blocks = positions.clamp_min(0) // logical_block_size
    safe_requests = requests.clamp(0, block_table.shape[0] - 1).long()
    blocks = block_table[safe_requests[:, None], logical_blocks.clamp_max(block_table.shape[1] - 1).long()]
    valid = (
        (positions >= 0)
        & (logical_blocks < block_table.shape[1])
        & (requests[:, None] >= 0)
        & (requests[:, None] < block_table.shape[0])
        & (blocks > 0)
        & (blocks < cache.shape[0])
    )
    safe_blocks = torch.where(valid, blocks, 0).long()
    offsets = positions.remainder(logical_block_size) // compress_ratio
    values = cache[safe_blocks, offsets.long(), 0]
    return torch.where(valid[..., None], values, 0), valid


@dataclass
class DeepseekV41CacheMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    request_indices: torch.Tensor
    logical_block_size: int
    compress_ratio: int
    max_seq_len: int = 0


class DeepseekV41CacheMetadataBuilder(AttentionMetadataBuilder[DeepseekV41CacheMetadata]):
    # All runtime-dependent addressing is device-side. Whole-model graph
    # validation remains necessary before claiming the serving graph gate.
    _cudagraph_support = AttentionCGSupport.ALWAYS

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_indices = torch.arange(max_tokens, dtype=torch.int32, device=device)
        self.request_indices = torch.empty(max_tokens, dtype=torch.int64, device=device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        common = common_attn_metadata
        count = common.num_input_tokens
        tokens = self.token_indices[:count]
        # query_start_loc is authoritative on device, including async/spec.
        requests = torch.searchsorted(common.query_start_loc[1:], tokens, right=True).long()
        valid = (tokens < common.num_actual_tokens) & (common.slot_mapping[:count] >= 0)
        self.request_indices[:count].copy_(torch.where(valid, requests, -1))
        return DeepseekV41CacheMetadata(
            common.block_table_tensor,
            common.slot_mapping[:count],
            self.request_indices[:count],
            self.kv_cache_spec.block_size,
            self.kv_cache_spec.compress_ratio,
            getattr(common, "max_seq_len", 0),
        )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        metadata = self.build(0, common_attn_metadata)
        # Captured index search must reach later decode positions. A capture
        # made at length 1 must not freeze a one-token key search forever.
        metadata.max_seq_len = self.vllm_config.model_config.max_model_len
        return metadata


class DeepseekV41CacheBackend(AttentionBackend):
    # Captured tensor operations read persistent device metadata directly.
    # There is no FIA task handle or host sequence-length list to update.
    requires_graph_param_update = False

    @staticmethod
    def get_name():
        return "ASCEND_V41_PAGED_STATE"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("Paged state is read and written by the model, not an attention kernel")

    @staticmethod
    def get_builder_cls():
        return DeepseekV41CacheMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
        return num_blocks, block_size, num_kv_heads, head_size

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        # Optional logical layout is [layers, blocks, rows, heads, channels].
        return (1, 0, 2, 3, 4) if include_num_layers_dimension else (0, 1, 2, 3)


class DeepseekV41PagedCache(nn.Module, AttentionLayerBase):
    def __init__(self, vllm_config, prefix, width, dtype, *, compress_ratio=1, sliding_window=None, block_size=None):
        super().__init__()
        self.prefix = prefix
        block_size = block_size or vllm_config.cache_config.block_size
        if compress_ratio not in (1, 2) or block_size % compress_ratio:
            raise ValueError("V4.1 cache blocks must be divisible by compression ratio 1 or 2")
        spec_args = dict(
            block_size=block_size,
            num_kv_heads=1,
            head_size=width,
            dtype=dtype,
            compress_ratio=compress_ratio,
            alignment=None,
        )
        self.spec = (
            AscendTokenCacheSpec(**spec_args)
            if sliding_window is None
            else AscendSlidingTokenCacheSpec(**spec_args, sliding_window=sliding_window)
        )
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate V4.1 cache: {prefix}")
        context[prefix] = self
        self.kv_cache = torch.empty(0)

    def get_attn_backend(self):
        return DeepseekV41CacheBackend

    def get_kv_cache_spec(self, vllm_config):
        return self.spec

    def forward(self):
        metadata = get_forward_context().attn_metadata
        return self.kv_cache, None if metadata is None else metadata[self.prefix]


class DeepseekV41TokenHistory(nn.Module):
    """Map IDs and gather Engram windows from a scheduler-owned full cache."""

    def __init__(self, vllm_config, token_map, max_ngram_size, prefix):
        super().__init__()
        self.register_buffer("token_map", token_map, persistent=False)
        self.max_ngram_size = max_ngram_size
        self.cache = DeepseekV41PagedCache(vllm_config, prefix, 1, torch.int64)

    def forward(self, token_ids, positions, image_mask=None):
        cache, metadata = self.cache()
        compressed = self.token_map[token_ids.long()]
        if image_mask is not None:
            compressed = torch.where(image_mask, -1, compressed)
        if metadata is None:
            # vLLM's initial memory profile runs before cache allocation.
            return F.pad(compressed[:, None], (0, self.max_ngram_size - 1), value=-1)
        write_paged(cache, metadata.slot_mapping, compressed[:, None])
        history_positions = positions[:, None] - torch.arange(self.max_ngram_size, device=positions.device)
        values, valid = read_paged(
            cache,
            history_positions,
            metadata.request_indices,
            metadata.block_table,
            logical_block_size=metadata.logical_block_size,
        )
        return torch.where(valid, values.squeeze(-1), -1)
