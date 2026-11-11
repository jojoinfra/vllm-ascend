# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate unintegrated HC4 residual mixing against the original NPU path."""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import torch
from benchmark_cache import graph_us
from validate_hc import load_module
from validate_quant import equal_values


def validate(report):
    import torch_npu  # noqa: F401

    if torch.npu.device_count() != 1:
        raise RuntimeError("HC residual validation requires one isolated NPU")
    torch.npu.set_device(0)
    torch.manual_seed(9302)
    module, core = load_module("cannbot_hc_residual"), load_module("core")
    report.update(numerical=[], graph_replay=[], microbenchmark=[])
    with torch.inference_mode():
        for width in (64, 128, 5120):
            kernels = module.HCPre(width), module.HCPost(width)
            for rows in (0, 1, 2, 8, 48, 128):
                for amplitude in (0.0, 1e-30, 1.0, 1e10, 1e30):
                    residual = torch.randn(rows, 4, width, device="npu", dtype=torch.bfloat16) * amplitude
                    sublayer = torch.randn(rows, width, device="npu", dtype=torch.bfloat16) * amplitude
                    pre, post = torch.randn(rows, 4, device="npu"), torch.randn(rows, 4, device="npu")
                    combination = torch.randn(rows, 4, 4, device="npu")
                    for operation, kernel, reference, args in (
                        ("pre", kernels[0], core.hc_pre, (residual, pre)),
                        ("post", kernels[1], core.hc_post, (sublayer, residual, post, combination)),
                    ):
                        equal_values(kernel(*args), reference(*args))
                        report["numerical"].append(
                            dict(operation=operation, width=width, rows=rows, amplitude=amplitude)
                        )
            rows = 2
            residual = torch.randn(rows, 4, width, device="npu", dtype=torch.bfloat16)
            sublayer = torch.randn(rows, width, device="npu", dtype=torch.bfloat16)
            pre, post = torch.randn(rows, 4, device="npu"), torch.randn(rows, 4, device="npu")
            combination = torch.randn(rows, 4, 4, device="npu")
            for operation, kernel, reference, args in (
                ("pre", kernels[0], core.hc_pre, (residual, pre)),
                ("post", kernels[1], core.hc_post, (sublayer, residual, post, combination)),
            ):
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    output = kernel(*args)
                for replay in range(10):
                    for arg in args:
                        arg.normal_()
                    if replay == 0:
                        residual[0, 0, 0], residual[1, 0, 0] = float("nan"), float("inf")
                    graph.replay()
                    equal_values(output, reference(*args))
                    report["graph_replay"].append(dict(operation=operation, width=width, replay=replay))
                if kernel.compiled.cache_info().misses != 1:
                    raise AssertionError("Row changes recompiled HC residual operation")
            if width == 5120:
                for rows in (2, 8, 12, 48, 128):
                    residual = torch.randn(rows, 4, width, device="npu", dtype=torch.bfloat16)
                    sublayer = torch.randn(rows, width, device="npu", dtype=torch.bfloat16)
                    pre, post = torch.randn(rows, 4, device="npu"), torch.randn(rows, 4, device="npu")
                    combination = torch.randn(rows, 4, 4, device="npu")
                    for operation, kernel, reference, args in (
                        ("pre", kernels[0], core.hc_pre, (residual, pre)),
                        ("post", kernels[1], core.hc_post, (sublayer, residual, post, combination)),
                    ):
                        row = dict(
                            operation=operation,
                            width=width,
                            rows=rows,
                            reference_us=graph_us(partial(reference, *args)),
                            candidate_us=graph_us(partial(kernel, *args)),
                        )
                        report["microbenchmark"].append(row)
                        print(json.dumps(row), flush=True)
        bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        residual = bits.reshape(256, 4, 64).npu()
        sublayer = residual[:, 0, :]
        pre, post = torch.randn(256, 8, device="npu")[:, ::2], torch.randn(256, 8, device="npu")[:, ::2]
        combination = torch.randn(256, 4, 8, device="npu")[:, :, ::2]
        for operation, kernel, reference, args in (
            ("pre", module.HCPre(64), core.hc_pre, (residual, pre)),
            ("post", module.HCPost(64), core.hc_post, (sublayer, residual, post, combination)),
        ):
            equal_values(kernel(*args), reference(*args))
            report["numerical"].append(dict(operation=operation, case="all_bf16_and_strided_coefficients"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/cannbot_hc_residual.py"
    report = dict(
        status="running",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope="Exact finite bits and nonfinite classification against initial NPU HC pre/post",
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
