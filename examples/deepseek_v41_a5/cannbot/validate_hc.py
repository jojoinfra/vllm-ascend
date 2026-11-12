# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile or numerically validate HC4 against the initial branch's equations.

Run inside an isolated container exposing exactly one reserved NPU. Compilation
mode requires no NPU devices and does not claim numerical/runtime validation.
"""

import argparse
import hashlib
import importlib.util
import json
import time
from functools import partial
from pathlib import Path

import torch


def load_module(name):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"v41_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(actual, expected):
    errors = []
    for value, golden in zip(actual, expected, strict=True):
        value, golden = value.cpu(), golden.cpu()
        torch.testing.assert_close(value, golden, rtol=1e-5, atol=1e-6)
        errors.append(float((value - golden).abs().max()) if value.numel() else 0.0)
    return errors


def elapsed_us(fn, repeats=20):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1e6 / repeats


def validate_npu(report):
    import torch_npu  # noqa: F401 -- register NPU backend only in hardware mode

    if torch.npu.device_count() != 1:
        raise RuntimeError("This validator requires exactly one isolated NPU")
    torch.npu.set_device(0)
    torch.manual_seed(741)
    golden = load_module("core").hc_split_sinkhorn
    HC4Sinkhorn = load_module("cannbot_hc").HC4Sinkhorn
    report["numerical"] = []
    report["graph_replay"] = []
    report["microbenchmark"] = []
    report["compile_cache"] = []
    with torch.inference_mode():
        for iterations, eps in ((1, 0.0), (2, 1e-6), (20, 1e-6), (20, 1e-3)):
            op = HC4Sinkhorn(iterations, eps)
            for tokens in (0, 1, 3, 8, 24, 48, 65, 512):
                # eps=0 can produce 0/0 for underflowed entire columns. The
                # finite-reference cases test large logits with positive eps.
                for amplitude in (0.0, 1.0, 100.0 if eps else 2.0):
                    mixes = (torch.randn(tokens, 24) * amplitude).npu()
                    # Match checkpoint-loaded nn.Parameters: inference_mode
                    # disables autograd but does not clear requires_grad.
                    scale = torch.nn.Parameter(torch.tensor([0.7, -1.3, 1.5]).npu())
                    base = torch.nn.Parameter(torch.randn(24).npu())
                    expected = golden(mixes, scale, base, 4, iterations, eps)
                    errors = compare(op(mixes, scale, base), expected)
                    report["numerical"].append(
                        dict(tokens=tokens, iterations=iterations, eps=eps, amplitude=amplitude, max_errors=errors)
                    )
            cache = op.compiled.cache_info()
            if cache.misses != 1:
                raise AssertionError(f"Token count changes recompiled the kernel: {cache}")
            report["compile_cache"].append(dict(iterations=iterations, eps=eps, misses=cache.misses))
            # Compile/warm up outside capture; change all inputs between replays.
            mixes = torch.randn(24, 24).npu()
            scale, base = torch.ones(3).npu(), torch.zeros(24).npu()
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                for _ in range(3):
                    op(mixes, scale, base)
            torch.npu.current_stream().wait_stream(stream)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                captured = op(mixes, scale, base)
            for replay in range(5):
                mixes.copy_(torch.randn_like(mixes) * (replay + 1))
                scale.copy_(torch.randn_like(scale))
                base.copy_(torch.randn_like(base))
                graph.replay()
                errors = compare(captured, golden(mixes, scale, base, 4, iterations, eps))
                report["graph_replay"].append(dict(iterations=iterations, eps=eps, replay=replay, max_errors=errors))
            if iterations == 20 and eps == 1e-6:
                for tokens in (24, 48):
                    mixes = torch.randn(tokens, 24).npu()
                    baseline_us = elapsed_us(partial(golden, mixes, scale, base, 4, iterations, eps))
                    candidate_us = elapsed_us(partial(op, mixes, scale, base))
                    report["microbenchmark"].append(
                        dict(
                            tokens=tokens,
                            baseline_us=baseline_us,
                            candidate_us=candidate_us,
                            scope="eager HC4 operator only; not end-to-end speedup",
                        )
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("compile", "npu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"mode": args.mode, "status": "running", "end_to_end_speedup": None}
    root = Path(__file__).resolve().parents[3]
    report["source_sha256"] = {
        name: hashlib.sha256((root / "vllm_ascend/models/deepseek_v41" / name).read_bytes()).hexdigest()
        for name in ("core.py", "cannbot_hc.py")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output)
    try:
        if args.mode == "compile":
            op = load_module("cannbot_hc").HC4Sinkhorn()
            op.compiled()
            report["compiled_contract"] = "symbolic leading token dimension >= 1; no shape specialization"
        else:
            validate_npu(report)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
