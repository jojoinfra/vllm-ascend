# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep floating-point projection rounding independent of token batching."""

import torch
import torch.nn.functional as F

PROJECTION_ROWS = 64


def apply_linear(linear, hidden):
    """Preserve the native linear's TP/weight handling at a fixed GEMM shape.

    A3 floating-point GEMMs can change BF16 rounding when prefill is split
    differently by a prefix hit. Quantized linears retain their native path;
    floating-point linears use equally sized tiles, including the last tile.
    The supplied linear must return a tensor (``return_bias=False`` in vLLM).
    """
    weight = getattr(linear, "weight", None)
    if weight is None or not weight.is_floating_point() or hidden.numel() == 0:
        return linear(hidden)
    shape = hidden.shape
    flat = hidden.reshape(-1, shape[-1])
    rows = flat.shape[0]
    padded = F.pad(flat, (0, 0, 0, -rows % PROJECTION_ROWS))
    output = torch.cat(
        [linear(padded[start : start + PROJECTION_ROWS]) for start in range(0, len(padded), PROJECTION_ROWS)]
    )
    return output[:rows].reshape(*shape[:-1], output.shape[-1])


def grouped_output_projection(hidden, weight):
    """Apply independent output groups: [tokens, groups, dim] x [groups, rank, dim]."""
    rows = hidden.shape[0]
    if rows == 0:
        return torch.einsum("tgd,grd->tgr", hidden, weight)
    padded = F.pad(hidden, (0, 0, 0, 0, 0, -rows % PROJECTION_ROWS))
    return torch.cat(
        [
            torch.einsum("tgd,grd->tgr", padded[start : start + PROJECTION_ROWS], weight)
            for start in range(0, len(padded), PROJECTION_ROWS)
        ]
    )[:rows]
