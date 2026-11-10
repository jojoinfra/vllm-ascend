# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 attention numerics from the pinned official model.py/kernel.py.

A3 stores dequantized BF16 KV, retaining the reference FP4/FP8 rounding before
storage. Floating point formats are emulated with device arithmetic, avoiding
unsupported float8/float4 casts and CPU fallbacks on A3.
"""

import math

import torch
from torch import nn

FP32_MANTISSA_BITS = 23
FP32_EXPONENT_BIAS = 127
FP8_MAX = 448.0
FP4_MAX = 6.0
ATTENTION_TILE_SIZE = 64


def ceil_power_of_two(value):
    """The official fast_round_scale for positive normal float32 values."""
    bits = value.float().contiguous().view(torch.int32)
    rounded = (bits + (1 << FP32_MANTISSA_BITS) - 1) & ~((1 << FP32_MANTISSA_BITS) - 1)
    return rounded.view(torch.float32)


def round_e4m3(value):
    absolute = value.float().abs().clamp_max(FP8_MAX)
    # The A3 scalar-shift fallback can reuse a shift operand across captured
    # and eager calls. A later eager shift must not change the graph's grid.
    # Absolute values have no sign bit, so integer division extracts the
    # biased exponent directly, without a scalar shift or exponent mask.
    exponent_unit = 1 << FP32_MANTISSA_BITS
    exponent = torch.div(absolute.contiguous().view(torch.int32), exponent_unit, rounding_mode="floor")
    exponent = exponent - FP32_EXPONENT_BIAS
    # E4M3 has three mantissa bits; subnormals use the fixed 2^-9 grid.
    precision = (exponent - 3).clamp_min(-9)
    # A3 exp2(-6) is one FP32 ULP below 2^-6. That error changes halfway
    # rounding, so build the exactly representable powers from exponent bits.
    step = ((precision + FP32_EXPONENT_BIAS) * exponent_unit).view(torch.float32)
    inverse = ((FP32_EXPONENT_BIAS - precision) * exponent_unit).view(torch.float32)
    return torch.round(value.float().clamp(-FP8_MAX, FP8_MAX) * inverse) * step


def fake_quant_fp8(value, group_size=32):
    grouped = value.float().unflatten(-1, (-1, group_size))
    scale = ceil_power_of_two(grouped.abs().amax(-1, keepdim=True).clamp_min(1e-4) * (1 / FP8_MAX))
    inverse = ((2 * FP32_EXPONENT_BIAS << FP32_MANTISSA_BITS) - scale.view(torch.int32)).view(torch.float32)
    return (round_e4m3(grouped * inverse) * scale).flatten(-2).to(value.dtype)


def fake_quant_fp4(value, *, group_size=32, e4m3_scale=False):
    grouped = value.float().unflatten(-1, (-1, group_size))
    maximum = grouped.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        scale = round_e4m3(maximum.clamp_min(FP4_MAX * 2**-9) / FP4_MAX)
    else:
        scale = ceil_power_of_two(maximum.clamp_min(FP4_MAX * 2**-126) * (1 / FP4_MAX))
    # Compare in the original value domain: division by an E4M3 scale can
    # perturb exact halfway values. Each threshold * scale is exact in FP32.
    absolute = grouped.abs()
    quantized = torch.where(absolute > 0.25 * scale, 0.5, 0.0)
    quantized = torch.where(absolute >= 0.75 * scale, 1.0, quantized)
    quantized = torch.where(absolute > 1.25 * scale, 1.5, quantized)
    quantized = torch.where(absolute >= 1.75 * scale, 2.0, quantized)
    quantized = torch.where(absolute > 2.5 * scale, 3.0, quantized)
    quantized = torch.where(absolute >= 3.5 * scale, 4.0, quantized)
    quantized = torch.where(absolute > 5.0 * scale, 6.0, quantized)
    quantized = torch.where(grouped.contiguous().view(torch.int32) < 0, -quantized, quantized)
    return (quantized * scale).flatten(-2).to(value.dtype)


class DeepseekV41Rotary(nn.Module):
    def __init__(self, config, max_positions, *, compressed):
        super().__init__()
        self.rotary_dim = config.qk_rope_head_dim
        base = config.compress_rope_theta if compressed else config.rope_theta
        frequencies = 1 / (base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))
        if compressed:
            rope = config.rope_parameters
            original = rope["original_max_position_embeddings"]

            def corrected_dim(rotations):
                return self.rotary_dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

            low = max(math.floor(corrected_dim(rope["beta_fast"])), 0)
            high = min(math.ceil(corrected_dim(rope["beta_slow"])), self.rotary_dim - 1)
            ramp = ((torch.arange(self.rotary_dim // 2) - low) / max(high - low, 1e-3)).clamp(0, 1)
            frequencies = frequencies / rope["factor"] * ramp + frequencies * (1 - ramp)
        angles = torch.outer(torch.arange(max_positions, dtype=torch.float32), frequencies)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, value, positions, *, inverse=False):
        tail = value[..., -self.rotary_dim :].float().unflatten(-1, (-1, 2))
        cos, sin = self.cos[positions.clamp_min(0).long()], self.sin[positions.clamp_min(0).long()]
        if value.ndim == 3:
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        if inverse:
            sin = -sin
        real, imag = tail[..., 0], tail[..., 1]
        rotated = torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1).flatten(-2)
        return torch.cat((value[..., : -self.rotary_dim], rotated.to(value.dtype)), dim=-1)


def sparse_attention(query, selected_kv, valid, sinks, scale):
    """Official 64-key online softmax, including BF16 probability rounding.

    Shapes: query [tokens, heads, dim], selected_kv [tokens, keys, dim]. The
    caller bounds token chunks, keeping the gather workspace independent of
    prefill length. Invalid rows, including an entirely padded query, yield 0.
    """
    maximum = torch.full(query.shape[:2], -1e30, dtype=torch.float32, device=query.device)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros_like(query, dtype=torch.float32)
    for start in range(0, selected_kv.shape[1], ATTENTION_TILE_SIZE):
        kv = selected_kv[:, start : start + ATTENTION_TILE_SIZE]
        scores = torch.bmm(query.float(), kv.float().transpose(1, 2)) * scale
        scores = scores.masked_fill(~valid[:, None, start : start + ATTENTION_TILE_SIZE], -torch.inf)
        next_maximum = torch.maximum(maximum, scores.amax(-1))
        correction = torch.exp(maximum - next_maximum)
        probabilities = torch.exp(scores - next_maximum.unsqueeze(-1))
        numerator = numerator * correction.unsqueeze(-1) + torch.bmm(probabilities.to(query.dtype).float(), kv.float())
        denominator = denominator * correction + probabilities.sum(-1)
        maximum = next_maximum
    denominator = denominator + torch.exp(sinks.float().unsqueeze(0) - maximum)
    return (numerator / denominator.unsqueeze(-1)).to(query.dtype)
