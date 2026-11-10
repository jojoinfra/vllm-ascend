# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from vllm_ascend.patch.worker.deepseek_v41_decoder import DeepseekV41DecoderLayer, hc_coefficients, hc_expand


@pytest.mark.parametrize("token_shape", [(0,), (1,), (3,), (63,), (64,), (65,), (2, 3), (129,)])
def test_sinkhorn_matches_double_precision_scalar_equations(token_shape):
    rng = np.random.default_rng(17)
    hidden = rng.normal(size=(*token_shape, 4, 7)).astype(np.float32)
    weight = rng.normal(size=(24, 28)).astype(np.float32)
    scale = np.array([0.7, 0.2, 0.6], dtype=np.float32)
    base = rng.normal(size=24).astype(np.float32)
    eps = 1e-6
    pre, post, combination = hc_coefficients(
        torch.from_numpy(hidden),
        torch.from_numpy(weight),
        torch.from_numpy(scale),
        torch.from_numpy(base),
        sinkhorn_iters=20,
        norm_eps=1e-20,
        hc_eps=eps,
    )
    assert pre.shape == post.shape == (*token_shape, 4)
    assert combination.shape == (*token_shape, 4, 4)
    hidden = hidden.reshape(-1, 4, 7)
    pre, post, combination = pre.reshape(-1, 4), post.reshape(-1, 4), combination.reshape(-1, 4, 4)
    for token in range(len(hidden)):
        flat = hidden[token].astype(np.float64).reshape(-1)
        mixes = weight.astype(np.float64) @ flat / np.sqrt(np.mean(flat**2) + 1e-20)
        expected_pre = 1 / (1 + np.exp(-(mixes[:4] * scale[0] + base[:4]))) + eps
        expected_post = 2 / (1 + np.exp(-(mixes[4:8] * scale[1] + base[4:8])))
        matrix = (mixes[8:] * scale[2] + base[8:]).reshape(4, 4)
        for row in range(4):
            matrix[row] = np.exp(matrix[row] - matrix[row].max())
            matrix[row] = matrix[row] / matrix[row].sum() + eps
        for iteration in range(20):
            if iteration:
                for row in range(4):
                    matrix[row] /= matrix[row].sum() + eps
            for column in range(4):
                matrix[:, column] /= matrix[:, column].sum() + eps
        np.testing.assert_allclose(pre[token], expected_pre, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(post[token], expected_post, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(combination[token], matrix, rtol=2e-6, atol=2e-7)


def test_hc_expand_uses_source_destination_orientation():
    residual = torch.tensor([[[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]]])
    permutation = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]])
    got = hc_expand(torch.zeros(1, 2), residual, torch.zeros(1, 3), permutation)
    torch.testing.assert_close(got, residual[:, [2, 0, 1]])


class RecordAttention(nn.Module):
    def forward(self, positions, hidden, attention_context):
        self.input = hidden.clone()
        return hidden * 0


class RecordMoE(nn.Module):
    def forward(self, hidden, *, image_mask):
        self.input = hidden.clone()
        self.image_mask = image_mask
        return hidden * 0


def test_layer_uses_preceding_mix_and_passes_next_mix_and_image_mask():
    config = SimpleNamespace(hidden_size=3, hc_mult=2, hc_sinkhorn_iters=20, rms_norm_eps=1e-20, hc_eps=1e-6)
    attention, ffn = RecordAttention(), RecordMoE()
    layer = DeepseekV41DecoderLayer(config, attention, ffn)
    with torch.no_grad():
        layer.hc_attn_fn.zero_()
        layer.hc_ffn_fn.zero_()
        layer.hc_attn_scale.zero_()
        layer.hc_ffn_scale.zero_()
        # Almost-identity residual mixing; attn_pre intentionally selects a
        # different HC combination than the incoming one-hot pre_mix.
        layer.hc_attn_base.copy_(torch.tensor([-2.0, 2.0, 0.0, 0.0, 12.0, -12.0, -12.0, 12.0]))
        layer.hc_ffn_base.copy_(torch.tensor([3.0, -3.0, 0.0, 0.0, 12.0, -12.0, -12.0, 12.0]))
    hidden = torch.tensor([[[1.0, 2.0, 4.0], [5.0, 2.0, -1.0]]])
    pre_mix = torch.tensor([[1.0, 0.0]])
    image_mask = torch.tensor([True])
    result, next_pre = layer(torch.tensor([7]), hidden, pre_mix, image_mask=image_mask)
    torch.testing.assert_close(attention.input, layer.attn_norm(hidden[:, 0]))
    attn_pre = torch.sigmoid(layer.hc_attn_base[:2]) + config.hc_eps
    # With the near-identity residual matrix, this discriminates delayed
    # mixing from using ffn_pre immediately, even allowing Sinkhorn epsilon.
    expected_ffn = layer.ffn_norm((hidden * attn_pre[None, :, None]).sum(1))
    torch.testing.assert_close(ffn.input, expected_ffn, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(next_pre[0], torch.sigmoid(layer.hc_ffn_base[:2]) + config.hc_eps)
    assert ffn.image_mask is image_mask
    assert result.shape == hidden.shape
