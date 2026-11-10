# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401
from torch import nn

from vllm_ascend.patch.worker.deepseek_v41_linear import apply_linear, grouped_output_projection


@pytest.mark.parametrize("grouped", [False, True])
@torch.inference_mode()
def test_projection_row_invariance_and_changed_graph_replays(grouped):
    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(4103)
    if grouped:
        weight = torch.randn(2, 1024, 4096, generator=generator).bfloat16().npu()
        token = torch.randn(2, 4096, generator=generator).bfloat16()

        def project(value):
            return grouped_output_projection(value, weight)
    else:
        linear = nn.Linear(5120, 1280, bias=False, dtype=torch.bfloat16, device="npu:0")
        linear.weight.copy_(torch.randn(1280, 5120, generator=generator).bfloat16())
        token = torch.randn(5120, generator=generator).bfloat16()

        def project(value):
            return apply_linear(linear, value)

    reference = project(token.unsqueeze(0).npu())[0].cpu()
    for rows in (1, 6, 24, 63, 64, 65, 94, 110, 112, 129):
        hidden = torch.randn(rows, *token.shape, generator=generator).bfloat16()
        indices = sorted({0, rows // 2, rows - 1})
        hidden[indices] = token
        actual = project(hidden.npu()).cpu()
        for index in indices:
            assert torch.equal(actual[index], reference)

    buffer = torch.randn(6, *token.shape, generator=generator).bfloat16().npu()
    for _ in range(3):
        project(buffer)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = project(buffer)
    for _ in range(3):
        buffer.copy_(torch.randn(buffer.shape, generator=generator).bfloat16().npu())
        expected = project(buffer)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(output, expected)
