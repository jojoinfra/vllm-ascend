# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact reference rounding on NPU, including scales at E4M3 midpoints."""

import pytest
import torch
import torch_npu

from vllm_ascend.patch.worker.deepseek_v41_attention_math import fake_quant_fp4, fake_quant_fp8, round_e4m3


def test_e4m3_normal_subnormal_and_halfway_values_on_npu():
    torch.npu.set_device(0)
    values = torch.arange(-65536, 65537, dtype=torch.float32) / 128
    values = torch.cat((values, torch.arange(-1024, 1025) / 1024)).clamp(-448, 448)
    actual = round_e4m3(values.npu()).cpu()
    torch.testing.assert_close(actual, values.to(torch.float8_e4m3fn).float(), rtol=0, atol=0)


@pytest.mark.parametrize("e4m3_scale", [False, True])
def test_fp4_quantization_matches_independent_grid_and_changed_graph_replay(e4m3_scale):
    torch.manual_seed(41)
    values = torch.randn(16, 512).bfloat16()
    # Real checkpoint latent exposed these exact midpoints. max/6=0.265625
    # must round to E4M3 scale .25, and -.78125/.15625=-5 rounds to -4.
    values[0, :16] = 0
    values[0, :3] = torch.tensor([1.59375, -0.78125, 0.3515625])
    values[1, :16] = 0
    values[1, :2] = torch.tensor([0.9375, -0.78125])

    def oracle(value):
        grouped = value.float().unflatten(-1, (-1, 16))
        maximum = grouped.abs().amax(-1, keepdim=True)
        if e4m3_scale:
            scale = (maximum.clamp_min(6 * 2**-9) / 6).to(torch.float8_e4m3fn).float()
        else:
            scale = torch.exp2((maximum.clamp_min(6 * 2**-126) / 6).log2().ceil())
        normalized = grouped / scale
        grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        distance = (normalized.abs().unsqueeze(-1) - grid).abs()
        code = torch.where(distance == distance.amin(-1, keepdim=True), torch.arange(8), 100)
        even = torch.where(code % 2 == 0, code, 100).amin(-1)
        code = torch.where(even < 100, even, code.amin(-1))
        return (torch.copysign(grid[code], normalized) * scale).flatten(-2).to(value.dtype)

    device_input = values.npu()
    for _ in range(3):
        fake_quant_fp4(device_input, group_size=16, e4m3_scale=e4m3_scale)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = fake_quant_fp4(device_input, group_size=16, e4m3_scale=e4m3_scale)
    for value in (values, values.flip(0)):
        device_input.copy_(value)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), oracle(value), rtol=0, atol=0)


def test_fp8_group_scales_match_native_format_on_npu():
    torch.manual_seed(8)
    values = torch.randn(16, 512).bfloat16()
    grouped = values.float().unflatten(-1, (-1, 32))
    scale = torch.exp2((grouped.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448).log2().ceil())
    expected = ((grouped / scale).to(torch.float8_e4m3fn).float() * scale).flatten(-2).bfloat16()
    torch.testing.assert_close(fake_quant_fp8(values.npu()).cpu(), expected, rtol=0, atol=0)


@torch.inference_mode()
def test_fp8_graph_is_independent_of_interleaved_eager_scalar_shifts(request):
    torch.npu.set_device(0)
    # Match the serving runner. This option has a setter but no Python
    # getter, so preserve its native value for pytest's teardown.
    previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    request.addfinalizer(lambda: setattr(torch.npu.config, "allow_internal_format", previous == b"enable"))
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(412)
    sample = torch.randn(1, 512).bfloat16()
    device_input = torch.zeros_like(sample, device="npu")
    for _ in range(3):
        fake_quant_fp8(device_input)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = fake_quant_fp8(device_input)

    device_input.copy_(sample)
    grouped = sample.float().unflatten(-1, (-1, 32))
    scale = torch.exp2((grouped.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448).log2().ceil())
    expected = ((grouped / scale).to(torch.float8_e4m3fn).float() * scale).flatten(-2).bfloat16()
    for rows in (1, 128):
        integers = torch.full((rows, 16, 32), 1 << 30, dtype=torch.int32, device="npu")
        # Retain the unrelated eager output across replay, as its caller
        # may still be using it. Its scalar must not change the graph's grid.
        for left_shift in (False, True):
            for shift in (23, 0, 32, 23):
                _eager_output = integers << shift if left_shift else integers >> shift
                graph.replay()
                torch.npu.synchronize()
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
