# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn

from vllm_ascend.models.deepseek_v41.native_ops import NativeMXLinear, quantize_activation


def test_quantize_activation_preserves_scale_pairs_and_bin_boundaries():
    x = torch.tensor([448.0, 449.0, -224.0, 0.0]).repeat_interleave(32).reshape(2, 64)
    encoded, scales = quantize_activation(x)
    assert scales.shape == (2, 1, 2)
    assert scales.flatten()[:3].tolist() == [127, 128, 126]
    decoded = encoded.float().reshape(2, 2, 32) * scales.view(torch.float8_e8m0fnu).float().reshape(2, 2, 1)
    torch.testing.assert_close(decoded[:, :, 0], torch.tensor([[448.0, 448.0], [-224.0, 0.0]]))


def test_projection_pairs_one_scale_row_per_output_channel():
    source = nn.Module()
    source.weight = nn.Parameter(torch.empty(64, 64), requires_grad=False)
    source.scale = nn.Parameter(torch.tensor([[121, 122], [125, 126]], dtype=torch.uint8), requires_grad=False)
    source.packed_fp4 = False
    projection = NativeMXLinear(source)
    assert projection.weight is source.weight
    assert projection.paired_scale.shape == (1, 64, 2)
    torch.testing.assert_close(projection.paired_scale[0], source.scale.repeat_interleave(32, dim=0))
