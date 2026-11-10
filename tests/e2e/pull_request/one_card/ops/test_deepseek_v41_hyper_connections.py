# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""mHC coefficients must not change with prefill or speculative batch shape."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.patch.worker.deepseek_v41_decoder import hc_coefficients


@pytest.mark.parametrize("seed", [41, 412])
@torch.inference_mode()
def test_mhc_batch_shape_and_changed_graph_replay(seed):
    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(seed)
    copies, width = 4, 5120
    samples = torch.randn(2, copies, width, generator=generator).bfloat16().npu()
    weight = (torch.randn(24, copies * width, generator=generator) / (copies * width) ** 0.5).npu()
    scale = torch.tensor([0.7, 0.2, 0.6], device="npu")
    base = torch.randn(24, generator=generator).npu()

    def coefficients(hidden):
        return hc_coefficients(hidden, weight, scale, base, sinkhorn_iters=20, norm_eps=1e-6, hc_eps=1e-6)

    reference = [tuple(x.cpu() for x in coefficients(row[None])) for row in samples]
    for rows in (1, 6, 24, 63, 64, 65, 94, 110, 128, 129):
        hidden = torch.randn(rows, copies, width, generator=generator).bfloat16().npu()
        for index in sorted({0, rows // 2, rows - 1}):
            hidden[index].copy_(samples[0])
        actual = coefficients(hidden)
        for index in sorted({0, rows // 2, rows - 1}):
            for value, expected in zip(actual, reference[0]):
                torch.testing.assert_close(value[index].cpu(), expected[0], rtol=0, atol=0)

    static = samples[0:1].expand(24, -1, -1).contiguous()
    for _ in range(3):
        coefficients(static)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = coefficients(static)
    for revision in (1, 0, 1):
        static.copy_(samples[revision : revision + 1].expand_as(static))
        graph.replay()
        torch.npu.synchronize()
        for value, expected in zip(captured, reference[revision]):
            torch.testing.assert_close(value.cpu(), expected.expand_as(value), rtol=0, atol=0)
