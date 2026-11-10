# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from torch import nn

from vllm_ascend.patch.worker.deepseek_v41_linear import apply_linear, grouped_output_projection


def test_parameterless_projection_preserves_its_native_contract():
    hidden = torch.randn(3, 5)
    assert apply_linear(nn.Identity(), hidden) is hidden


@pytest.mark.parametrize("shape", [(0, 17), (1, 17), (63, 17), (65, 17), (2, 65, 17)])
def test_linear_preserves_leading_axes_bias_and_dot_products(shape):
    generator = torch.Generator().manual_seed(41)
    hidden = torch.randn(shape, generator=generator)
    linear = nn.Linear(17, 11)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(linear.weight.shape, generator=generator))
        linear.bias.copy_(torch.randn(linear.bias.shape, generator=generator))
    actual = apply_linear(linear, hidden)
    expected = hidden.numpy().astype(np.float64) @ linear.weight.detach().numpy().astype(np.float64).T
    expected += linear.bias.detach().numpy()
    torch.testing.assert_close(actual.double(), torch.from_numpy(expected), rtol=1e-5, atol=2e-6)
    assert actual.shape == (*shape[:-1], 11)


@pytest.mark.parametrize("rows", [0, 1, 63, 64, 65, 129])
def test_grouped_projection_preserves_group_ownership(rows):
    generator = torch.Generator().manual_seed(42)
    hidden = torch.randn(rows, 3, 17, generator=generator)
    weight = torch.randn(3, 7, 17, generator=generator)
    actual = grouped_output_projection(hidden, weight)
    expected = np.einsum("tgd,grd->tgr", hidden.numpy().astype(np.float64), weight.numpy().astype(np.float64))
    torch.testing.assert_close(actual.double(), torch.from_numpy(expected), rtol=1e-5, atol=2e-6)


def test_quantized_linear_retains_its_native_batch_contract():
    class Quantized(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(3, 5, dtype=torch.int8), requires_grad=False)
            self.seen = []

        def forward(self, hidden):
            self.seen.append(hidden.shape)
            return hidden.sum(-1, keepdim=True).expand(*hidden.shape[:-1], 3)

    linear = Quantized()
    hidden = torch.randn(7, 5)
    expected = hidden.sum(-1, keepdim=True).expand(7, 3)
    assert torch.equal(apply_linear(linear, hidden), expected)
    assert linear.seen == [hidden.shape]
