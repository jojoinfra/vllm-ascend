# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize exported Ascend kernel records separately for every rank."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


def analyze(path):
    groups = defaultdict(lambda: dict(count=0, duration_sum_us=0.0, max_us=0.0))
    intervals = []
    hc = []
    cache_write = []
    cache_read = []
    fp8 = []
    index = []
    extra_kernels = {key: [] for key in ("fp4_quant", "rms_norm", "index_cube", "index_reduce")}
    devices = set()
    with path.open() as stream:
        for row in csv.DictReader(stream):
            devices.add(row["Device_id"])
            # Preserve sub-microsecond precision at epoch-sized timestamps.
            start = int(Decimal(row["Start Time(us)"].strip()) * 1000)
            duration = float(row["Duration(us)"])
            intervals.append((start, start + int(Decimal(row["Duration(us)"]) * 1000)))
            key = (row["Type"], row["Accelerator Core"], row["Input Shapes"], row["Input Data Types"])
            item = groups[key]
            item["count"] += 1
            item["duration_sum_us"] += duration
            item["max_us"] = max(item["max_us"], duration)
            item.setdefault("example_name", row["Name"])
            if "hc4sinkhorn" in row["Name"].lower() or "hc4sinkhorn" in row["Type"].lower():
                hc.append(duration)
            if "cachefieldwrite" in row["Name"].lower() or "cachefieldwrite" in row["Type"].lower():
                cache_write.append(duration)
            if "cachefieldread" in row["Name"].lower() or "cachefieldread" in row["Type"].lower():
                cache_read.append(duration)
            if "fp8quantdequant" in row["Name"].lower() or "fp8quantdequant" in row["Type"].lower():
                fp8.append(duration)
            if "pagedindexscores" in row["Name"].lower() or "pagedindexscores" in row["Type"].lower():
                index.append(duration)
            for key, name in (
                ("fp4_quant", "fp4quantdequant"),
                ("rms_norm", "rmsnormkernel"),
                ("index_cube", "pagedindexcubekernel"),
                ("index_reduce", "indexheadreducekernel"),
            ):
                if name in row["Name"].lower() or name in row["Type"].lower():
                    extra_kernels[key].append(duration)
    if not intervals:
        raise ValueError(f"Empty kernel trace: {path}")
    union = []
    for start, end in sorted(intervals):
        if union and start <= union[-1][1]:
            union[-1][1] = max(union[-1][1], end)
        else:
            union.append([start, end])
    ranked = [dict(op_type=key[0], core=key[1], shapes=key[2], dtypes=key[3], **value) for key, value in groups.items()]
    ranked.sort(key=lambda item: item["duration_sum_us"], reverse=True)
    by_type = defaultdict(lambda: dict(count=0, duration_sum_us=0.0))
    for row in ranked:
        by_type[row["op_type"]]["count"] += row["count"]
        by_type[row["op_type"]]["duration_sum_us"] += row["duration_sum_us"]
    return dict(
        source=str(path.resolve()),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        devices=sorted(devices),
        kernel_count=len(intervals),
        kernel_span_us=(union[-1][1] - union[0][0]) / 1000,
        kernel_union_us=sum(end - start for start, end in union) / 1000,
        duration_sum_us=sum(row["duration_sum_us"] for row in ranked),
        hc4_sinkhorn_count=len(hc),
        hc4_sinkhorn_duration_sum_us=sum(hc),
        cache_write_count=len(cache_write),
        cache_write_duration_sum_us=sum(cache_write),
        cache_read_count=len(cache_read),
        cache_read_duration_sum_us=sum(cache_read),
        fp8_quant_count=len(fp8),
        fp8_quant_duration_sum_us=sum(fp8),
        index_score_count=len(index),
        index_score_duration_sum_us=sum(index),
        **{f"{key}_count": len(values) for key, values in extra_kernels.items()},
        **{f"{key}_duration_sum_us": sum(values) for key, values in extra_kernels.items()},
        top_shape_groups=ranked[:40],
        op_types=sorted(
            [dict(op_type=key, **value) for key, value in by_type.items()],
            key=lambda row: row["duration_sum_us"],
            reverse=True,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.profile_dir.rglob("kernel_details.csv"))
    if len(paths) != 4:
        raise ValueError(f"Expected four complete rank exports; found {len(paths)}")
    report = dict(
        scope="All kernel records within this separate profiled capture, one report per exported rank",
        usable_as_unprofiled_performance=False,
        limitations=[
            "Duration sums can overlap and are not end-to-end latency or potential speedup.",
            "Rank durations must not be added together.",
            "Kernel records alone do not measure host stalls, transfer utilization or the critical path.",
            "Missing HC names do not prove absence: retain the trace for name/correlation inspection.",
        ],
        ranks=[analyze(path) for path in paths],
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    for rank in report["ranks"]:
        print(json.dumps({key: rank[key] for key in ("devices", "kernel_count", "hc4_sinkhorn_count")}))
        print(json.dumps(rank["op_types"][:12]))


if __name__ == "__main__":
    main()
