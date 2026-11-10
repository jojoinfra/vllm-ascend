# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Delayed Hyper-Connections, following the official V4.1 Block equations.

Unlike V4, a sublayer consumes the preceding sublayer's pre-mix. Parameter
names retain the checkpoint layout; attention and Ascend FusedMoE are supplied
by the model constructor rather than replaced by a second routing engine.
"""

import torch
import torch.nn.functional as F
from torch import nn

from vllm_ascend.patch.worker.deepseek_v41_compressor import DeepseekV41RMSNorm

_HC_PROJECTION_ROWS = 64


def hc_collapse(hidden, pre_mix):
    return (hidden.float() * pre_mix.unsqueeze(-1)).sum(-2).to(hidden.dtype)


def hc_expand(output, residual, post_mix, combination):
    # combination[source, destination]; transposing these axes is observable
    # even though Sinkhorn makes both row and column sums approach one.
    mixed_residual = (combination.unsqueeze(-1) * residual.float().unsqueeze(-2)).sum(-3)
    return (post_mix.unsqueeze(-1) * output.float().unsqueeze(-2) + mixed_residual).to(output.dtype)


def hc_coefficients(hidden, weight, scale, base, *, sinkhorn_iters, norm_eps, hc_eps):
    copies = hidden.shape[-2]
    flattened = hidden.flatten(-2).float()
    token_shape = flattened.shape[:-1]
    flattened = flattened.reshape(-1, flattened.shape[-1])
    rows = flattened.shape[0]
    # A3 FP32 GEMM and vector kernels change rounding with the row count.
    # Those tiny differences can cross a BF16 boundary after delayed mixing,
    # making prefix hits differ from full prefill. Keep each GEMM at 64 rows
    # and vector operations aligned, including the final partial tile.
    flattened = F.pad(flattened, (0, 0, 0, -rows % _HC_PROJECTION_ROWS))
    projections = [
        F.linear(flattened[start : start + _HC_PROJECTION_ROWS], weight)
        for start in range(0, flattened.shape[0], _HC_PROJECTION_ROWS)
    ]
    projection = torch.cat(projections) if projections else F.linear(flattened, weight)
    mixes = projection * torch.rsqrt(flattened.square().mean(-1, keepdim=True) + norm_eps)
    pre = torch.sigmoid(mixes[..., :copies] * scale[0] + base[:copies]) + hc_eps
    post = 2 * torch.sigmoid(mixes[..., copies : 2 * copies] * scale[1] + base[copies : 2 * copies])
    matrix = (mixes[..., 2 * copies :] * scale[2] + base[2 * copies :]).unflatten(-1, (copies, copies))
    matrix = matrix.softmax(-1) + hc_eps
    matrix = matrix / (matrix.sum(-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        matrix = matrix / (matrix.sum(-1, keepdim=True) + hc_eps)
        matrix = matrix / (matrix.sum(-2, keepdim=True) + hc_eps)
    return (
        pre[:rows].reshape(*token_shape, copies),
        post[:rows].reshape(*token_shape, copies),
        matrix[:rows].reshape(*token_shape, copies, copies),
    )


class DeepseekV41DecoderLayer(nn.Module):
    def __init__(self, config, attention, ffn):
        super().__init__()
        self.attn = attention
        self.ffn = ffn
        self.attn_norm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.hc_sinkhorn_iters
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        if self.sinkhorn_iters < 1:
            raise ValueError("Hyper-Connections requires at least one Sinkhorn iteration")
        coefficients = (2 + self.hc_mult) * self.hc_mult
        for sublayer in ("attn", "ffn"):
            self.register_parameter(
                f"hc_{sublayer}_fn",
                nn.Parameter(
                    torch.empty(coefficients, self.hc_mult * config.hidden_size, dtype=torch.float32),
                    requires_grad=False,
                ),
            )
            self.register_parameter(
                f"hc_{sublayer}_base", nn.Parameter(torch.empty(coefficients, dtype=torch.float32), requires_grad=False)
            )
            self.register_parameter(
                f"hc_{sublayer}_scale", nn.Parameter(torch.empty(3, dtype=torch.float32), requires_grad=False)
            )

    def _coefficients(self, hidden, sublayer):
        return hc_coefficients(
            hidden,
            getattr(self, f"hc_{sublayer}_fn"),
            getattr(self, f"hc_{sublayer}_scale"),
            getattr(self, f"hc_{sublayer}_base"),
            sinkhorn_iters=self.sinkhorn_iters,
            norm_eps=self.norm_eps,
            hc_eps=self.hc_eps,
        )

    def forward(self, positions, hidden, pre_mix, *, image_mask=None, attention_context=None):
        attn_pre, attn_post, attn_combination = self._coefficients(hidden, "attn")
        attention_input = self.attn_norm(hc_collapse(hidden, pre_mix))
        output = self.attn(positions, attention_input, attention_context)
        hidden = hc_expand(output, hidden, attn_post, attn_combination)
        ffn_pre, ffn_post, ffn_combination = self._coefficients(hidden, "ffn")
        output = self.ffn(self.ffn_norm(hc_collapse(hidden, attn_pre)), image_mask=image_mask)
        return hc_expand(output, hidden, ffn_post, ffn_combination), ffn_pre
