# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate an unintegrated RMS prototype with a declared one-BF16-ULP bound."""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import torch
from benchmark_cache import graph_us
from validate_hc import load_module


def compare(actual, expected):
    actual, expected = actual.cpu(), expected.cpu()
    if not torch.equal(actual.isnan(), expected.isnan()) or not torch.equal(actual.isinf(), expected.isinf()):
        raise AssertionError("RMS nonfinite classification differs")
    finite = expected.isfinite()
    left, right = actual.view(torch.int16).int(), expected.view(torch.int16).int()
    left = torch.where(left < 0, -32768 - left, left)
    right = torch.where(right < 0, -32768 - right, right)
    ulp = (left - right).abs()[finite]
    maximum = int(ulp.max()) if ulp.numel() else 0
    if maximum > 1:
        raise AssertionError(f"RMS exceeds declared one-BF16-ULP bound: {maximum}")
    torch.testing.assert_close(actual[~finite], expected[~finite], rtol=0, atol=0, equal_nan=True)
    return dict(max_bf16_ulp=maximum, changed_finite_elements=int((ulp != 0).sum()), finite_elements=int(finite.sum()))


def validate(report):
    import torch_npu  # noqa: F401

    if torch.npu.device_count() != 1:
        raise RuntimeError("Requires exactly one isolated NPU")
    torch.npu.set_device(0)
    torch.manual_seed(822)
    cls = load_module("cannbot_norm").RMSNorm
    reference = load_module("core").rms_norm
    report.update(numerical=[], graph_replay=[], microbenchmark=[])
    with torch.inference_mode():
        for width in (128, 512, 1280, 5120):
            for dtype in (torch.bfloat16, torch.float32):
                for eps in (1e-20, 1e-6):
                    candidate = cls(width, eps, dtype)
                    weight = torch.randn(width, dtype=dtype, device="npu")
                    for rows in (1, 2, 8, 48):
                        for amplitude in (0.0, 1e-12, 1.0, 100.0, 1e20):
                            x = (torch.randn(rows, width, device="npu") * amplitude).bfloat16()
                            stats = compare(candidate(x, weight), reference(x, weight, eps))
                            report["numerical"].append(
                                dict(width=width, dtype=str(dtype), eps=eps, rows=rows, amplitude=amplitude, **stats)
                            )
                    x = torch.randn(8, width, dtype=torch.bfloat16, device="npu")
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        output = candidate(x, weight)
                    for replay in range(5):
                        x.copy_(torch.randn_like(x))
                        weight.copy_(torch.randn_like(weight))
                        graph.replay()
                        stats = compare(output, reference(x, weight, eps))
                        report["graph_replay"].append(
                            dict(width=width, dtype=str(dtype), eps=eps, replay=replay, **stats)
                        )
                    if candidate.compiled.cache_info().misses != 1:
                        raise AssertionError("RMS shape changes recompiled the kernel")
                    if eps == 1e-20 and dtype == torch.bfloat16:
                        for rows in (2, 48):
                            x = torch.randn(rows, width, dtype=torch.bfloat16, device="npu")
                            row = dict(
                                width=width,
                                rows=rows,
                                reference_us=graph_us(partial(reference, x, weight, eps)),
                                candidate_us=graph_us(partial(candidate, x, weight)),
                            )
                            report["microbenchmark"].append(row)
                            print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/cannbot_norm.py"
    report = dict(
        status="running",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        numerical_contract="At most one BF16 ULP per finite output; exact nonfinite classification",
        scope="Unintegrated operator prototype; no end-to-end or bit-exactness claim",
    )
    with args.output.open("x") as stream:
        json.dump(report, stream)
    try:
        validate(report)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], numerical=len(report["numerical"]))), flush=True)


if __name__ == "__main__":
    main()
