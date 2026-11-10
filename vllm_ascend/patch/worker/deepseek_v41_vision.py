# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vision tower and image layout from the pinned official V4.1 reference."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageOps
from torch import nn

from vllm_ascend.patch.worker.deepseek_v41_compressor import DeepseekV41RMSNorm

VISION_NORM_EPS = 1e-6
IMAGE_PAD_COLOR = (127, 127, 127)


def image_grid(width, height, config):
    """Return ViT rows/columns under the reference's complete-span token budget."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    patch, ratio = config.patch_size, config.downsample_ratio
    if config.max_wh_ratio is not None and width > height * config.max_wh_ratio:
        width = height * config.max_wh_ratio
    if width * height < config.min_pixels:
        scale = (config.min_pixels / (width * height)) ** 0.5
        width, height = int(width * scale), int(height * scale)
    columns, rows = math.ceil(width / patch), math.ceil(height / patch)
    if image_span_size(rows, columns, ratio) > config.max_image_tokens:
        aspect = height / width
        max_columns = math.sqrt((config.max_image_tokens - 2) / aspect + 0.25) - 0.5
        max_rows = max_columns * aspect
        cell = patch * ratio
        if max_columns < 1:
            rows, columns = (config.max_image_tokens - 2) // 2 * ratio, ratio
        elif max_rows < 1:
            rows, columns = ratio, (config.max_image_tokens - 3) * ratio
        else:
            scale = min(math.floor(max_columns) * cell / width, math.floor(max_rows) * cell / height)
            rows, columns = math.floor(height * scale / patch), math.floor(width * scale / patch)
    if image_span_size(rows, columns, ratio) > config.max_image_tokens:
        raise ValueError("Image grid exceeds the configured image token budget")
    return rows, columns


def image_span_size(rows, columns, ratio):
    return math.ceil(rows / ratio) * (math.ceil(columns / ratio) + 1) + 2


def image_patches(image, config):
    """Normalize a PIL image and arrange channel-first patches in reading order."""
    rows, columns = image_grid(image.width, image.height, config)
    patch = config.patch_size
    size = (columns * patch, rows * patch)
    image = image.convert("RGB")
    if config.max_wh_ratio is not None and image.width >= config.max_wh_ratio * image.height:
        image = image.resize(size)
    else:
        image = ImageOps.pad(image, size, color=IMAGE_PAD_COLOR)
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    pixels = ((pixels - 0.5) / 0.5).to(torch.bfloat16)
    patches = pixels.reshape(3, rows, patch, columns, patch).permute(1, 3, 0, 2, 4)
    return patches.reshape(rows * columns, 3, patch, patch), rows, columns


def vision_rotary_factors(rows, columns, head_dim, theta, device):
    # Each spatial axis contributes half of the split-half RoPE frequencies.
    axis_dim = head_dim // 2
    inverse = 1.0 / (theta ** (torch.arange(0, axis_dim, 2, device=device).float() / axis_dim))
    heights = torch.arange(rows, device=device)[:, None].expand(rows, columns)
    widths = torch.arange(columns, device=device)[None, :].expand(rows, columns)
    phases = torch.stack((heights, widths), dim=-1).reshape(-1, 2, 1).float() * inverse
    phases = phases.flatten(1).unsqueeze(1)
    return phases.cos(), phases.sin()


def apply_vision_rotary(hidden, cosine, sine):
    first, second = hidden.float().chunk(2, dim=-1)
    return torch.cat((first * cosine - second * sine, second * cosine + first * sine), -1).to(hidden.dtype)


def vision_attention(query, key, value):
    """Unmasked inference attention with the official 3D math-SDPA arithmetic.

    Preserve full head/query shapes and FP32 intermediates. Reusing the score
    buffer avoids SDPA's second quadratic allocation and unused returned
    attention weights. There are no masked rows or dropout in this tower.
    """
    scale = query.shape[-1] ** (-0.25)
    scores = torch.matmul(query.float() * scale, key.float().transpose(-2, -1) * scale)
    torch.softmax(scores, dim=-1, out=scores)
    return torch.matmul(scores, value.float()).to(query.dtype)


class DeepseekV41PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(3 * config.patch_size**2, config.hidden_size)

    def forward(self, patches):
        return self.proj(patches.flatten(1))


class DeepseekV41VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.heads
        self.wqkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden, cosine, sine):
        query, key, value = (part.unflatten(-1, (self.heads, self.head_dim)) for part in self.wqkv(hidden).chunk(3, -1))
        query = apply_vision_rotary(query, cosine, sine)
        key = apply_vision_rotary(key, cosine, sine)
        # Preserve the official 3D SDPA layout. FIA and query/head chunking
        # produce BF16 rounding differences amplified by the last vision block.
        output = vision_attention(query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1))
        return self.wo(output.transpose(0, 1).reshape(hidden.shape))


class DeepseekV41VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden):
        gate, up = self.w1(hidden).chunk(2, -1)
        return self.w2(F.silu(gate) * up)


class DeepseekV41VisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = DeepseekV41RMSNorm(config.hidden_size, VISION_NORM_EPS, dtype=torch.float32)
        self.attn = DeepseekV41VisionAttention(config)
        self.norm2 = DeepseekV41RMSNorm(config.hidden_size, VISION_NORM_EPS, dtype=torch.float32)
        self.mlp = DeepseekV41VisionMLP(config)

    def forward(self, hidden, cosine, sine):
        hidden = hidden + self.attn(self.norm1(hidden), cosine, sine)
        return hidden + self.mlp(self.norm2(hidden))


class DeepseekV41Vision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embed = DeepseekV41PatchEmbed(config)
        self.blocks = nn.ModuleList(DeepseekV41VisionBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = DeepseekV41RMSNorm(config.hidden_size, VISION_NORM_EPS, dtype=torch.float32)

    def forward(self, patches, rows, columns):
        hidden = self.patch_embed(patches)
        cosine, sine = vision_rotary_factors(
            rows,
            columns,
            self.config.hidden_size // self.config.num_attention_heads,
            self.config.rope_theta,
            hidden.device,
        )
        for block in self.blocks:
            hidden = block(hidden, cosine, sine)
        return self.norm(hidden)


class DeepseekV41Aligner(nn.Module):
    def __init__(self, config, text_hidden_size):
        super().__init__()
        self.ratio = config.downsample_ratio
        self.w1 = nn.Linear(config.hidden_size * self.ratio**2, text_hidden_size)
        self.w2 = nn.Linear(text_hidden_size, text_hidden_size)

    def forward(self, hidden, rows, columns):
        hidden = hidden.reshape(rows, columns, -1).permute(2, 0, 1)
        hidden = F.pad(hidden, (0, -columns % self.ratio, 0, -rows % self.ratio))
        # Preserve channel-major pixels within each cell, as in F.unfold.
        hidden = F.unfold(hidden.unsqueeze(0), self.ratio, stride=self.ratio).squeeze(0).T
        return self.w2(F.gelu(self.w1(hidden)))


def make_image_span(features, rows, columns, ratio, start, newline, end):
    height, width = math.ceil(rows / ratio), math.ceil(columns / ratio)
    features = features.reshape(height, width, -1)
    lines = torch.cat((features, newline.to(features.dtype).expand(height, 1, -1)), dim=1).flatten(0, 1)
    return torch.cat((start.to(features.dtype)[None], lines, end.to(features.dtype)[None]))
