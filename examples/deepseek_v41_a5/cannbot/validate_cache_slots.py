# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate unintegrated vector cache read-slot metadata against the initial path."""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import torch
from benchmark_cache import graph_us
from validate_hc import load_module


def reference(positions, table, lengths, active, max_length, storage_rows):
    pos = positions.clamp(0, max_length - 1)
    blocks = table.gather(1, pos // 128).long()
    slots = blocks * 128 + pos % 128
    valid = active[:, None] & (positions >= 0) & (positions < lengths[:, None])
    valid = valid & (slots >= 0) & (slots < storage_rows)
    return torch.where(valid, slots, 0).to(torch.int32), valid


def compare(actual, expected):
    for value, golden in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, golden, rtol=0, atol=0)


def validate(report):
    import torch_npu  # noqa: F401

    if torch.npu.device_count() != 1:
        raise RuntimeError("Cache slot validation requires one isolated NPU")
    torch.npu.set_device(0)
    torch.manual_seed(8501)
    cls = load_module("cannbot_cache_slots").CacheReadSlots
    report.update(numerical=[], graph_replay=[], microbenchmark=[])
    with torch.inference_mode():
        for table_width, max_length, storage_rows in ((7, 830, 893), (512, 65536, 65536)):
            kernel = cls(table_width, max_length, storage_rows)
            golden = partial(reference, max_length=max_length, storage_rows=storage_rows)
            for rows in (0, 1, 2, 8, 48):
                for columns in (64, 128, 640, 8192):
                    positions = torch.randint(
                        -max_length, 2 * max_length, (rows, columns), dtype=torch.int64, device="npu"
                    )
                    table = torch.randint(
                        -1, (storage_rows + 127) // 128 + 1, (rows, table_width), dtype=torch.int32, device="npu"
                    )
                    lengths = torch.randint(-1, 2 * max_length, (rows,), dtype=torch.int32, device="npu")
                    active = torch.rand(rows, device="npu") > 0.25
                    if rows:
                        positions[0, :8] = torch.tensor(
                            [-(2**63), -1, 0, 127, 128, 2**32, 2**32 + 127, 2**63 - 1], device="npu"
                        )
                        table[0, :2] = torch.tensor([-(2**31), 2**31 - 1], dtype=torch.int32, device="npu")
                    for length_dtype in (torch.int32, torch.int64):
                        typed_lengths = lengths.to(length_dtype)
                        compare(
                            kernel(positions, table, typed_lengths, active),
                            golden(positions, table, typed_lengths, active),
                        )
                        report["numerical"].append(
                            dict(table_width=table_width, rows=rows, columns=columns, length_dtype=str(length_dtype))
                        )
            rows, columns = 2, 640
            positions = torch.randint(0, max_length, (rows, columns), device="npu")
            table = torch.randint(0, (storage_rows + 127) // 128, (rows, table_width), dtype=torch.int32, device="npu")
            lengths = torch.full((rows,), max_length, dtype=torch.int64, device="npu")
            active = torch.ones(rows, dtype=torch.bool, device="npu")
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = kernel(positions, table, lengths, active)
            for replay in range(12):
                positions.random_(-max_length, 2 * max_length)
                table.random_(-1, (storage_rows + 127) // 128 + 1)
                lengths.random_(-1, 2 * max_length)
                active.logical_not_()
                graph.replay()
                compare(output, golden(positions, table, lengths, active))
                report["graph_replay"].append(dict(table_width=table_width, replay=replay))
            if kernel.compiled.cache_info().misses != 1:
                raise AssertionError("Changing rows/columns recompiled cache slot metadata")
            active.fill_(True)
            positions.random_(0, max_length)
            table.random_(0, (storage_rows + 127) // 128)
            lengths.fill_(max_length)
            row = dict(
                table_width=table_width,
                rows=rows,
                columns=columns,
                reference_us=graph_us(partial(golden, positions, table, lengths, active)),
                candidate_us=graph_us(partial(kernel, positions, table, lengths, active)),
            )
            report["microbenchmark"].append(row)
            print(json.dumps(row), flush=True)
            table.zero_()
            positions[:, :8] = torch.tensor([-(2**63), -1, 0, 127, 2**31, 2**32, 2**32 + 127, 2**63 - 1], device="npu")
            for boundary in (-(2**63), -1, 0, 1, 2**31 - 1, 2**31, 2**32, 2**32 + 128, 2**63 - 1):
                lengths.fill_(boundary)
                compare(kernel(positions, table, lengths, active), golden(positions, table, lengths, active))
                report["numerical"].append(dict(table_width=table_width, case="int64_length_boundary", length=boundary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/cannbot_cache_slots.py"
    report = dict(
        status="running",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope="Exact read-slot and valid mask equality; no integration",
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
