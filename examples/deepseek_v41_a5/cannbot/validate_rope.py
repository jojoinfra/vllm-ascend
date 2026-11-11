# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the unintegrated rotary application against the initial NPU path."""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import torch
from benchmark_cache import graph_us
from validate_hc import load_module
from validate_quant import equal_values


def reference(fn, x, cosine, sine, inverse):
    fn(x[..., -64:], cosine, sine, inverse=inverse)
    return x


def validate(report):
    import torch_npu  # noqa: F401

    if torch.npu.device_count() != 1:
        raise RuntimeError("RoPE validation requires one isolated NPU")
    torch.npu.set_device(0)
    torch.manual_seed(934)
    cls = load_module("cannbot_rope").RotaryApply
    original = load_module("attention_ops").apply_rotary_emb_real
    report.update(numerical=[], graph_replay=[], microbenchmark=[])
    with torch.inference_mode():
        for pitch, heads in ((512, 1), (512, 16), (512, 64), (128, 8), (128, 32)):
            for inverse in (False, True):
                kernel = cls(pitch, heads, inverse)
                for tokens in (0, 1, 2, 12, 48, 128):
                    x = torch.randn(1, tokens, heads, pitch, dtype=torch.bfloat16, device="npu")
                    angles = torch.randn(tokens, 32, device="npu") * 100
                    cosine, sine = angles.cos(), angles.sin()
                    expected = reference(original, x.clone(), cosine, sine, inverse)
                    identity = kernel(x, cosine, sine)
                    if identity is not x:
                        raise AssertionError("RoPE must return its original tensor")
                    equal_values(x, expected)
                    report["numerical"].append(dict(pitch=pitch, heads=heads, inverse=inverse, tokens=tokens))
                x = torch.randn(1, 2, heads, pitch, dtype=torch.bfloat16, device="npu")
                angles = torch.randn(2, 32, device="npu")
                cosine, sine = angles.cos(), angles.sin()
                kernel(x, cosine, sine)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    kernel(x, cosine, sine)
                for replay in range(8):
                    x.normal_()
                    angles.normal_()
                    cosine.copy_(angles.cos())
                    sine.copy_(angles.sin())
                    expected = reference(original, x.clone(), cosine, sine, inverse)
                    graph.replay()
                    equal_values(x, expected)
                    report["graph_replay"].append(dict(pitch=pitch, heads=heads, inverse=inverse, replay=replay))
                if kernel.compiled.cache_info().misses != 1:
                    raise AssertionError("Token shape changes recompiled RoPE")
                if not inverse:
                    for tokens in (2, 12, 128):
                        x = torch.randn(1, tokens, heads, pitch, dtype=torch.bfloat16, device="npu")
                        baseline = x.clone()
                        angles = torch.randn(tokens, 32, device="npu")
                        cosine, sine = angles.cos(), angles.sin()
                        row = dict(
                            pitch=pitch,
                            heads=heads,
                            tokens=tokens,
                            reference_us=graph_us(partial(reference, original, baseline, cosine, sine, inverse)),
                            candidate_us=graph_us(partial(kernel, x, cosine, sine)),
                        )
                        report["microbenchmark"].append(row)
                        print(json.dumps(row), flush=True)
        x = torch.zeros(1, 1024, 1, 128, dtype=torch.bfloat16, device="npu")
        bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        x[..., -64:].copy_(bits.reshape(1, 1024, 1, 64))
        cosine = torch.ones(1024, 32, dtype=torch.float32, device="npu")
        sine = torch.zeros_like(cosine)
        expected = reference(original, x.clone(), cosine, sine, False)
        cls(128, 1)(x, cosine, sine)
        equal_values(x, expected)
        report["numerical"].append(dict(case="all_bf16_encodings_in_rotated_tail", elements=65536))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/cannbot_rope.py"
    report = dict(
        status="running",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope="Unintegrated operation; exact finite bits and nonfinite classification against initial NPU path",
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
