# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import attention
from vllm_ascend.models.deepseek_v41.attention_ops import fp8_quant_dequant


@pytest.mark.parametrize("device", ["cpu", "npu"])
@pytest.mark.parametrize("rows", [0, 6, 256, 513])
def test_wo_quantization_matches_dense_and_bounds_temporaries(monkeypatch, device, rows):
    if device == "npu":
        if importlib.util.find_spec("torch_npu") is None:
            pytest.skip("Requires Ascend NPU")
        importlib.import_module("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("Requires Ascend NPU")
    generator = torch.Generator().manual_seed(820)
    sample = torch.randn(rows, 2, 64, generator=generator, dtype=torch.bfloat16).to(device)
    expected = fp8_quant_dequant(sample.clone(), 32)
    chunks = []

    def record(tile, block_size, **kwargs):
        chunks.append(tile.shape[0])
        return fp8_quant_dequant(tile, block_size, **kwargs)

    monkeypatch.setattr(attention, "fp8_quant_dequant", record)
    actual = attention._quantize_projection_input(sample)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert sum(chunks) == rows
    assert max(chunks) <= attention._OUTPUT_QDQ_TILE_ROWS
    if rows > attention._OUTPUT_QDQ_TILE_ROWS:
        assert actual.data_ptr() == sample.data_ptr()


def test_cannbot_fp8_npu_graph_and_noncontiguous_inplace():
    pytest.importorskip("cannbotdsl")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Requires Ascend NPU")
    from vllm_ascend.models.deepseek_v41.cannbot_quant import FP8QuantDequant

    quantizer = FP8QuantDequant()
    with torch.inference_mode():
        original = torch.randn(3, 128, dtype=torch.bfloat16, device="npu")
        backing = original.clone()
        view = backing[:, ::2]
        assert quantizer(view, inplace=True) is view
        torch.testing.assert_close(view, fp8_quant_dequant(original[:, ::2]), rtol=0, atol=0)
        torch.testing.assert_close(backing[:, 1::2], original[:, 1::2], rtol=0, atol=0)
        x = torch.zeros(5, 64, dtype=torch.bfloat16, device="npu")
        quantizer(x)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = quantizer(x)
        for amplitude in (0, 1e-6, 1, 100):
            x.copy_(torch.randn_like(x) * amplitude)
            graph.replay()
            torch.testing.assert_close(output, fp8_quant_dequant(x), rtol=0, atol=0)
        assert quantizer.compiled.cache_info().misses == 1
