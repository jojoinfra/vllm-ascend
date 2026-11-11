# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical and graph validation of the unintegrated paged index prototype."""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import torch
from benchmark_cache import graph_us, original_read
from validate_hc import load_module


def reference(backing, query, weights, table, lengths, active, capacity, ratio):
    positions = ((torch.arange(capacity, device=query.device) + 1) * ratio - 1).expand(query.shape[0], -1)
    keys = original_read(backing, positions, table, lengths, active.bool(), 512, 128)
    per_head = torch.einsum("bhd,btd->bht", query, keys)
    return (per_head.relu_() * weights.unsqueeze(-1)).sum(1)


def candidate(kernel, backing, query, weights, table, lengths, active, output):
    output.zero_()
    kernel(backing, query, weights, table, lengths, active, output)
    return output


def compare(actual, expected):
    actual, expected = actual.cpu(), expected.cpu()
    mismatch = (actual.view(torch.int16) != expected.view(torch.int16)) & ~((actual == 0) & (expected == 0))
    if mismatch.any():
        indices = mismatch.nonzero()[:8]
        examples = [(i.tolist(), float(actual[tuple(i)]), float(expected[tuple(i)])) for i in indices]
        raise AssertionError(f"Index score bit mismatch: count={int(mismatch.sum())}, examples={examples}")


def validate(report):
    import torch_npu  # noqa: F401

    if torch.npu.device_count() != 1:
        raise RuntimeError("Requires exactly one isolated NPU")
    torch.npu.set_device(0)
    torch.manual_seed(583)
    module = load_module("cannbot_index")
    qdq = load_module("attention_ops").fp4_quant_dequant
    report.update(numerical=[], graph_replay=[], microbenchmark=[])
    with torch.inference_mode():
        for heads, ratio in ((8, 2), (32, 2), (32, 1), (32, 128)):
            kernel = module.PagedIndexScores(1024, 512, heads, ratio=ratio).compile()
            for rows, capacity in ((1, 8), (3, 128), (12, 65536 // ratio)):
                backing = qdq(torch.randn(2048, 1024, dtype=torch.bfloat16, device="npu"))
                query = qdq(torch.randn(rows, heads, 128, dtype=torch.bfloat16, device="npu"))
                weights = torch.randn(rows, heads, dtype=torch.bfloat16, device="npu")
                table = torch.randint(0, 16, (rows, 512), dtype=torch.int32, device="npu")
                # Invalid physical blocks must produce zero scores like the reference.
                table[:, 1] = -1
                lengths = torch.arange(rows, dtype=torch.int64, device="npu") * 127 + ratio
                active = torch.ones(rows, dtype=torch.int32, device="npu")
                if rows > 1:
                    active[0] = 0
                    lengths[-1] = 0
                output = torch.empty(rows, capacity, dtype=torch.bfloat16, device="npu")
                run = partial(candidate, kernel, backing, query, weights, table, lengths, active, output)
                golden = partial(reference, backing, query, weights, table, lengths, active, capacity, ratio)
                compare(run(), golden())
                report["numerical"].append(dict(heads=heads, ratio=ratio, rows=rows, capacity=capacity))
                if rows != 12:
                    continue
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    observed = run()
                for replay in range(12):
                    query.copy_(qdq(torch.randn_like(query)))
                    weights.copy_(torch.randn_like(weights))
                    lengths.copy_(torch.randint(0, 4096, lengths.shape, dtype=torch.int64, device="npu"))
                    active.copy_(torch.randint(0, 2, active.shape, dtype=torch.int32, device="npu"))
                    table.copy_(torch.randint(0, 16, table.shape, dtype=torch.int32, device="npu"))
                    graph.replay()
                    compare(observed, golden())
                    report["graph_replay"].append(dict(heads=heads, ratio=ratio, replay=replay))
                for context in (256, 4096, 65536):
                    lengths.fill_(context)
                    active.fill_(1)
                    backing.copy_(qdq(torch.randn_like(backing)))
                    graph.replay()
                    compare(observed, golden())
                    report["graph_replay"].append(dict(heads=heads, ratio=ratio, context=context, changed_backing=True))
                    row = dict(
                        heads=heads,
                        ratio=ratio,
                        capacity=capacity,
                        context=context,
                        reference_us=graph_us(golden),
                        candidate_us=graph_us(run),
                    )
                    report["microbenchmark"].append(row)
                    print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/cannbot_index.py"
    report = dict(
        status="running",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope="Unintegrated prototype; no end-to-end performance claim",
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
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
