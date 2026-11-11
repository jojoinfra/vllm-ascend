# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare original cache access with CANNBot under repeated graph replay."""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import torch
from validate_cache import equal_bits, original_field_write
from validate_hc import elapsed_us, load_module


def original_read(backing, positions, table, lengths, active, offset, width):
    pos = positions.clamp(0, 65535)
    blocks = table.gather(1, pos // 128).long()
    slots = blocks * 128 + pos % 128
    page = backing[:, offset : offset + width]
    valid = active[:, None] & (positions >= 0) & (positions < lengths[:, None])
    valid = valid & (slots >= 0) & (slots < page.shape[0])
    safe = torch.where(valid, slots, 0)
    values = page.index_select(0, safe.flatten()).reshape(*positions.shape, width)
    return torch.where(valid[..., None], values, 0)


def candidate_read(reader, backing, positions, table, lengths, active, output):
    output.zero_()
    reader(backing, positions, table, lengths, active, output)
    return output


def graph_us(fn, repetitions=8):
    fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(repetitions):
            fn()
    return elapsed_us(graph.replay, repeats=10) / repetitions


def main():
    import torch_npu  # noqa: F401
    from cannbotdsl import dtypes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.npu.device_count() != 1:
        raise RuntimeError("Graph microbenchmark requires one isolated NPU")
    torch.npu.set_device(0)
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/cannbot_cache.py"
    report = dict(status="running", source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), results=[])
    with args.output.open("x") as stream:
        json.dump(report, stream)
    module = load_module("cannbot_cache")
    try:
        with torch.inference_mode():
            for dtype, dsl_dtype in ((torch.bfloat16, dtypes.bfloat16), (torch.float32, dtypes.float32)):
                for width in (128, 512):
                    offset = 512
                    backing = torch.randn(103296, 1024, dtype=dtype, device="npu")
                    values = torch.randn(12, width, dtype=dtype, device="npu")
                    slots = torch.arange(12, dtype=torch.int64, device="npu")
                    active = torch.ones(12, dtype=torch.int32, device="npu")
                    writer = module.CacheFieldWrite(1024, width, offset, dsl_dtype).compile()
                    reference = partial(original_field_write, backing, values, slots, offset, width)
                    candidate = partial(writer, backing, values, slots, active)
                    row = dict(
                        op="write",
                        dtype=str(dtype),
                        width=width,
                        reference_us=graph_us(reference),
                        candidate_us=graph_us(candidate),
                    )
                    report["results"].append(row)
                    print(json.dumps(row), flush=True)
                    reader = module.CacheFieldRead(1024, width, offset, dtype=dsl_dtype).compile()
                    lengths = torch.full((12,), 128, dtype=torch.int32, device="npu")
                    table = torch.zeros((12, 512), dtype=torch.int32, device="npu")
                    live = active.bool()
                    for columns in (8, 640, 8192):
                        positions = torch.arange(columns, dtype=torch.int64, device="npu").expand(12, -1).contiguous()
                        output = torch.empty((12, columns, width), dtype=dtype, device="npu")
                        reference = partial(original_read, backing, positions, table, lengths, live, offset, width)
                        candidate = partial(candidate_read, reader, backing, positions, table, lengths, active, output)
                        equal_bits(candidate(), reference())
                        row = dict(
                            op="read",
                            dtype=str(dtype),
                            width=width,
                            columns=columns,
                            reference_us=graph_us(reference),
                            candidate_us=graph_us(candidate),
                        )
                        report["results"].append(row)
                        print(json.dumps(row), flush=True)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["scope"] = "Operator graph microbenchmark only; not end-to-end throughput"
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
