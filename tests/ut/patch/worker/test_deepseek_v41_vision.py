# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from vllm_ascend.patch.worker.deepseek_v41_vision import (
    DeepseekV41Aligner,
    DeepseekV41VisionAttention,
    apply_vision_rotary,
    image_grid,
    image_patches,
    image_span_size,
    make_image_span,
    vision_attention,
    vision_rotary_factors,
)


def processor_config(**overrides):
    values = dict(patch_size=14, downsample_ratio=3, min_pixels=544**2, max_image_tokens=1024, max_wh_ratio=None)
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize("size", [(1, 1), (544, 544), (4096, 3072), (1, 100000), (100000, 1)])
def test_image_grid_respects_full_span_budget_and_positive_dimensions(size):
    config = processor_config()
    rows, columns = image_grid(*size, config)
    assert rows > 0 and columns > 0
    assert image_span_size(rows, columns, config.downsample_ratio) <= config.max_image_tokens
    if size[0] == size[1]:
        assert (rows, columns) == (39, 39)


def test_image_grid_rejects_empty_image():
    with pytest.raises(ValueError, match="positive"):
        image_grid(0, 3, processor_config())


def test_image_patches_preserve_channel_and_spatial_order():
    pixels = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    config = processor_config(patch_size=2, downsample_ratio=1, min_pixels=1)
    patches, rows, columns = image_patches(Image.fromarray(pixels), config)
    assert (rows, columns) == (2, 3)
    reconstructed = torch.cat([torch.cat(list(patches[i * columns : (i + 1) * columns]), -1) for i in range(rows)], -2)
    expected = ((torch.tensor(pixels).permute(2, 0, 1).float() / 255 - 0.5) / 0.5).bfloat16()
    torch.testing.assert_close(reconstructed, expected, rtol=0, atol=0)


def test_vision_rotary_uses_separate_spatial_axes_and_preserves_norm():
    cosine, sine = vision_rotary_factors(2, 3, 8, 10000, "cpu")
    hidden = torch.randn(6, 2, 8)
    rotated = apply_vision_rotary(hidden, cosine, sine)
    torch.testing.assert_close(rotated[0], hidden[0], rtol=0, atol=0)
    torch.testing.assert_close(rotated.square().sum(-1), hidden.square().sum(-1))
    # Moving right changes only width frequencies; moving down only height.
    torch.testing.assert_close(cosine[0, :, :2], cosine[1, :, :2], rtol=0, atol=0)
    torch.testing.assert_close(cosine[0, :, 2:], cosine[3, :, 2:], rtol=0, atol=0)
    assert not torch.equal(cosine[0], cosine[1])
    assert not torch.equal(cosine[0], cosine[3])


def test_aligner_keeps_channel_major_cells_and_zero_pads_partial_edges():
    config = SimpleNamespace(hidden_size=2, downsample_ratio=2)
    aligner = DeepseekV41Aligner(config, 8)
    aligner.w1 = aligner.w2 = nn.Identity()
    hidden = torch.stack((torch.arange(6), torch.arange(10, 16)), -1).float()
    with patch("vllm_ascend.patch.worker.deepseek_v41_vision.F.gelu", side_effect=lambda values: values):
        result = aligner(hidden, 2, 3)
    expected = torch.tensor([[0, 1, 3, 4, 10, 11, 13, 14], [2, 0, 5, 0, 12, 0, 15, 0]]).float()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_image_span_keeps_delimiters_and_line_breaks_in_reading_order():
    features = torch.arange(4).float()[:, None]
    start, newline, end = (torch.tensor([value]) for value in (10.0, 20.0, 30.0))
    span = make_image_span(features, 4, 3, 2, start, newline, end)
    expected = torch.tensor([10, 0, 1, 20, 2, 3, 20, 30]).float()[:, None]
    torch.testing.assert_close(span, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_vision_attention_is_bidirectional_with_independent_heads():
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2)
    attention = DeepseekV41VisionAttention(config)
    with torch.no_grad():
        attention.wqkv.weight.copy_(torch.eye(8).repeat(3, 1))
        attention.wqkv.bias.zero_()
        attention.wo.weight.copy_(torch.eye(8))
        attention.wo.bias.zero_()
    hidden = torch.arange(24).reshape(3, 8).float() / 20
    cosine, sine = torch.ones(3, 1, 2), torch.zeros(3, 1, 2)
    actual = attention(hidden, cosine, sine)
    heads = hidden.reshape(3, 2, 4).transpose(0, 1)
    scores = (heads @ heads.transpose(-1, -2)) / 2
    expected = (scores.softmax(-1) @ heads).transpose(0, 1).reshape(3, 8)
    torch.testing.assert_close(actual, expected)
    changed = hidden.clone()
    changed[-1] += 1
    assert not torch.equal(actual[0], attention(changed, cosine, sine)[0])


@pytest.mark.parametrize("shape", [(2, 15, 8), (3, 22, 16)])
@torch.inference_mode()
def test_vision_attention_matches_unmodified_math_sdpa_with_strided_values(shape):
    generator = torch.Generator().manual_seed(41)
    tensors = [
        torch.randn(shape[1], shape[0], shape[2], generator=generator).bfloat16().transpose(0, 1) for _ in range(3)
    ]
    actual = vision_attention(*tensors)
    expected = F.scaled_dot_product_attention(*tensors)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
