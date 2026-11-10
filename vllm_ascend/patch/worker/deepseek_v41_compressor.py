# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ratio 1/2 compression from the pinned official V4.1 model equations."""

import torch
from torch import nn

from vllm_ascend.patch.worker.deepseek_v41_cache import DeepseekV41PagedCache, read_paged, write_paged
from vllm_ascend.patch.worker.deepseek_v41_linear import apply_linear


class DeepseekV41RMSNorm(nn.Module):
    def __init__(self, width, eps, *, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, dtype=dtype), requires_grad=False)
        self.eps = eps

    def forward(self, hidden):
        normalized = hidden.float() * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(hidden.dtype)


class DeepseekV41Compressor(nn.Module):
    """One output row per scheduled token, masked until its group completes.

    Projection state is paged by original token position, so arbitrary chunk
    cuts, shared prefixes and speculative overwrites use the scheduler's
    existing block ownership. There is no circular per-request mutable state.
    Consumers write only rows selected by the returned completion mask.
    """

    def __init__(self, vllm_config, config, ratio, prefix):
        super().__init__()
        if ratio not in (1, 2):
            raise ValueError("V4.1 compressor supports ratios 1 and 2")
        self.ratio = ratio
        self.head_dim = config.head_dim
        self.wkv = nn.Linear(
            config.hidden_size,
            config.head_dim,
            bias=False,
            dtype=torch.float32 if ratio == 2 else torch.bfloat16,
        )
        self.norm = DeepseekV41RMSNorm(config.head_dim, config.rms_norm_eps)
        if ratio == 2:
            self.wgate = nn.Linear(config.hidden_size, config.head_dim, bias=False, dtype=torch.float32)
            self.state = DeepseekV41PagedCache(
                vllm_config,
                prefix + ".state",
                2 * config.head_dim,
                torch.float32,
                sliding_window=ratio,
                block_size=2 * ratio,
            )

    def forward(self, hidden, positions):
        if self.ratio == 1:
            return self.norm(apply_linear(self.wkv, hidden)), torch.ones_like(positions, dtype=torch.bool)

        cache, metadata = self.state()
        projected = torch.cat(
            (apply_linear(self.wkv, hidden.float()), apply_linear(self.wgate, hidden.float())), dim=-1
        )
        write_paged(cache, metadata.slot_mapping, projected)
        group_positions = positions[:, None] - torch.arange(self.ratio - 1, -1, -1, device=positions.device)
        grouped, readable = read_paged(
            cache,
            group_positions,
            metadata.request_indices,
            metadata.block_table,
            logical_block_size=metadata.logical_block_size,
        )
        values, scores = grouped.split(self.head_dim, dim=-1)
        pooled = (values * scores.softmax(dim=1)).sum(dim=1).to(hidden.dtype)
        complete = ((positions + 1) % self.ratio == 0) & readable.all(dim=-1)
        return self.norm(pooled), complete
