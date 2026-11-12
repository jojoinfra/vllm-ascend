# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate experimental FP4 QDQ against the initial equations and A5 path."""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import torch
from benchmark_cache import graph_us
from validate_hc import load_module


def equal_values(actual, expected):
    actual, expected = actual.cpu(), expected.cpu()
    # NaN payloads need not match; finite values, signed zero and infinities do.
    nan = expected.isnan()
    if not torch.equal(actual.isnan(), nan):
        indices = (actual.isnan() != nan).nonzero()[:8]
        examples = [(i.tolist(), float(actual[tuple(i)]), float(expected[tuple(i)])) for i in indices]
        raise AssertionError(f"Nonfinite classification differs: {examples}")
    mismatch = (actual.view(torch.int16) != expected.view(torch.int16)) & ~nan
    if mismatch.any():
        indices = mismatch.nonzero()[:8]
        examples = [(i.tolist(), float(actual[tuple(i)]), float(expected[tuple(i)])) for i in indices]
        raise AssertionError(f"QDQ bit mismatch: count={int(mismatch.sum())}, examples={examples}")


def validate(report, group, scale_format):
    import torch_npu  # noqa: F401 -- explicitly register the hardware backend

    if torch.npu.device_count() != 1:
        raise RuntimeError("QDQ validation requires exactly one isolated NPU")
    torch.npu.set_device(0)
    torch.manual_seed(741)
    reference = partial(load_module("attention_ops").fp4_quant_dequant, block_size=group, scale_format=scale_format)
    candidate = load_module("cannbot_fp4").FP4QuantDequant(group, scale_format)
    report.update(numerical=[], graph_replay=[], graph_microbenchmark=[])
    with torch.inference_mode():
        for shape in (
            (0, 512),
            (1, group),
            (3, group),
            (65, group),
            (1, 512),
            (3, 128),
            (12, 512),
            (48, 8192),
            (512, 512),
        ):
            for amplitude in (0.0, 1e-6, 1e-4, 1.0, 100.0, 1e20):
                x = (torch.randn(shape) * amplitude).to(torch.bfloat16)
                equal_values(candidate(x.npu()), reference(x))
                report["numerical"].append(dict(shape=list(shape), amplitude=amplitude))
        # Every BF16 encoding, including signed zeros, subnormals, infinities
        # and NaNs, participates in a real per-group scale computation.
        exhaustive = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        exhaustive = exhaustive.reshape(-1, group)
        # The initial A5 path flushes certain FP32 subnormal results. Compare
        # against that unchanged device reference and retain CPU differences.
        device_input = exhaustive.npu()
        device_reference = reference(device_input).cpu()
        cpu_reference = reference(exhaustive)
        equal_values(candidate(device_input), device_reference)
        difference = (device_reference.view(torch.int16) != cpu_reference.view(torch.int16)) & ~cpu_reference.isnan()
        report["initial_npu_vs_cpu_non_nan_bit_differences"] = int(difference.sum())
        report["exhaustive_reference"] = "Initial attention_ops.fp4_quant_dequant on the same NPU"
        report["numerical"].append(dict(case="all_bf16_encodings", elements=exhaustive.numel()))
        edge = torch.zeros(8, group, dtype=torch.bfloat16)
        edge[:, :8] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.0], dtype=torch.bfloat16)
        edge[0, -1], edge[1, -1], edge[2, -1] = float("nan"), float("inf"), -float("inf")
        edge[3:, -1] = torch.tensor([448.0, 224.0, 112.0, 1e-4, 1e-6], dtype=torch.bfloat16)
        equal_values(candidate(edge.npu()), reference(edge))
        report["numerical"].append(dict(case="mixed_nonfinite_and_scale_boundaries"))
        original = torch.randn(12, 1024, dtype=torch.bfloat16)
        backing = original.npu()
        view = backing[:, ::2]
        golden = reference(original[:, ::2])
        returned = candidate(view, inplace=True)
        if returned is not view:
            raise AssertionError("In-place QDQ must return the original view")
        equal_values(view, golden)
        equal_values(backing[:, 1::2], original[:, 1::2])
        report["numerical"].append(dict(case="noncontiguous_inplace_alias"))
        for shape in ((12, 512), (48, 8192)):
            x = torch.zeros(shape, dtype=torch.bfloat16, device="npu")
            candidate(x)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = candidate(x)
            for replay, amplitude in enumerate((0.0, 1e-6, 1.0, 100.0, 1e20)):
                values = (torch.randn(shape) * amplitude).to(torch.bfloat16)
                x.copy_(values)
                graph.replay()
                equal_values(output, reference(values))
                report["graph_replay"].append(dict(shape=list(shape), replay=replay, amplitude=amplitude))
        cache = candidate.compiled.cache_info()
        if cache.misses != 1:
            raise AssertionError(f"Shape changes recompiled QDQ: {cache}")
        report["compile_cache_misses"] = cache.misses
        for shape in ((2, 128), (8, 128), (8, 512), (2, 32, 128)):
            x = torch.randn(shape, dtype=torch.bfloat16, device="npu")
            equal_values(candidate(x), reference(x))
            row = dict(
                shape=list(shape),
                reference_us=graph_us(partial(reference, x)),
                candidate_us=graph_us(partial(candidate, x)),
            )
            report["graph_microbenchmark"].append(row)
            print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", type=int, choices=(16, 32), required=True)
    parser.add_argument("--scale-format", choices=("e8m0", "e4m3"), required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/cannbot_quant.py"
    report = dict(
        status="running",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope="Numerical and operator graph tests only; no end-to-end speedup claim",
    )
    with args.output.open("x") as stream:
        json.dump(report, stream)
    try:
        validate(report, args.group, args.scale_format)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
