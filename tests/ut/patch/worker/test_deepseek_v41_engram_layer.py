# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_ascend.patch.worker.deepseek_v41_engram_layer import DeepseekV41Engram


class FixedEmbedding(nn.Module):
    def forward(self, ids, group=None):
        return torch.ones((*ids.shape, 1), dtype=torch.bfloat16)


def make_layer(rotation=None):
    config = SimpleNamespace(
        hidden_size=4,
        hc_mult=2,
        rms_norm_eps=1e-20,
        engram_max_ngram_size=2,
        engram_n_heads=1,
        engram_head_dim=1,
    )
    rotation_config = (
        None
        if rotation is None
        else dict(
            value_projection_rotated=True,
            value_basis="quarot_global",
            key_and_gate_basis="original",
            runtime_delta_rotation=False,
        )
    )
    layer = DeepseekV41Engram(
        config, FixedEmbedding(), device="cpu", rotation=rotation, rotation_config=rotation_config
    )
    with torch.no_grad():
        # Opposite keys for the two HC copies, and nonconstant value.
        layer.wkv.weight[:, 0] = torch.tensor([1] * 4 + [-1] * 4 + [1, 2, 3, 4])
    return layer


def test_signed_gate_per_hc_copy_and_image_mask():
    layer = make_layer()
    hidden = torch.ones(2, 2, 4)
    result = layer(hidden, torch.zeros(2, 1, dtype=torch.int64), token_mask=torch.tensor([True, False]))
    positive_gate = 1 / (1 + math.exp(-math.sqrt(2)))
    negative_gate = 1 / (1 + math.exp(math.sqrt(2)))
    expected = hidden.clone()
    expected[0, 0] += positive_gate * torch.tensor([1, 2, 3, 4])
    expected[0, 1] += negative_gate * torch.tensor([1, 2, 3, 4])
    torch.testing.assert_close(result, expected)


def test_rotated_values_are_not_rotated_twice():
    # An exact orthogonal signed permutation makes the basis test independent
    # of rounding and detects applying the rotation in the wrong direction.
    rotation = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [-1.0, 0.0, 0.0, 0.0]])
    original, rotated = make_layer(), make_layer(rotation)
    with torch.no_grad():
        rotated.q_weight.copy_(original.q_weight)
        rotated.k_weight.copy_(original.k_weight)
        rotated.wkv.weight[-4:, 0] = original.wkv.weight[-4:, 0].float() @ rotation
    hidden = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [-2.0, 1.0, 4.0, 3.0]]])
    ids = torch.zeros(1, 1, dtype=torch.int64)
    torch.testing.assert_close(rotated(hidden @ rotation, ids), original(hidden, ids) @ rotation)


def test_rotation_metadata_is_required():
    config = SimpleNamespace(
        hidden_size=4, hc_mult=2, rms_norm_eps=1e-20, engram_max_ngram_size=2, engram_n_heads=1, engram_head_dim=1
    )
    with pytest.raises(ValueError, match="global rotation"):
        DeepseekV41Engram(
            config,
            FixedEmbedding(),
            device="cpu",
            rotation_config={
                "value_projection_rotated": True,
                "value_basis": "quarot_global",
                "key_and_gate_basis": "original",
            },
        )
