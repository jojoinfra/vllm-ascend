# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check direct cache access against CPU indexing, including changing graph inputs."""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import torch
from validate_hc import elapsed_us, load_module


def equal_bits(actual, expected):
    integer = torch.int16 if expected.dtype == torch.bfloat16 else torch.int32
    actual, expected = actual.cpu().view(integer), expected.cpu().view(integer)
    if not torch.equal(actual, expected):
        mismatch = (actual != expected).nonzero()
        selected = torch.cat((mismatch[:4], mismatch[-4:]))
        examples = [(index.tolist(), int(actual[tuple(index)]), int(expected[tuple(index)])) for index in selected]
        raise AssertionError(f"Cache bits differ: {len(mismatch)} values; examples={examples}")


def read_reference(backing, positions, table, lengths, active, offset, width, block_size=128):
    output = torch.zeros((*positions.shape, width), dtype=backing.dtype)
    for row in range(positions.shape[0]):
        for column in range(positions.shape[1]):
            position = int(positions[row, column])
            if not active[row] or not 0 <= position < lengths[row]:
                continue
            block = position // block_size
            if block >= table.shape[1]:
                continue
            slot = int(table[row, block]) * block_size + position % block_size
            if 0 <= slot < backing.shape[0]:
                output[row, column] = backing[slot, offset : offset + width]
    return output


def write_reference(backing, values, slots, enabled, offset):
    result = backing.clone()
    for row in range(values.shape[0]):
        slot = int(slots[row])
        if enabled[row] and 0 <= slot < backing.shape[0]:
            result[slot, offset : offset + values.shape[1]] = values[row]
    return result


def original_field_write(backing, values, slots, offset, width):
    backing[:, offset : offset + width].index_copy_(0, slots, values)


def validate(report):
    import torch_npu  # noqa: F401 -- register backend only for hardware tests
    from cannbotdsl import dtypes

    if torch.npu.device_count() != 1:
        raise RuntimeError("Cache validation requires exactly one isolated NPU")
    torch.npu.set_device(0)
    torch.manual_seed(741)
    module = load_module("cannbot_cache")
    report.update(numerical=[], graph_replay=[], microbenchmark=[])
    with torch.inference_mode():
        for dtype, dsl_dtype in ((torch.bfloat16, dtypes.bfloat16), (torch.float32, dtypes.float32)):
            for width, offset in ((128, 512), (512, 0), (512, 512)):
                backend = module.CacheFieldBackend(128)
                backend.prepare(1024, width, offset, dtype)
                writer, reader = backend.fields[(1024, width, offset, dtype)]
                for rows, columns in ((1, 1), (3, 8), (12, 8), (12, 640), (48, 8)):
                    backing = torch.randn(1024, 1024, dtype=dtype)
                    backing[0, offset] = -0.0
                    backing[0, offset + 1] = float("nan")
                    values = torch.randn(rows, width, dtype=dtype)
                    slots = torch.arange(rows, dtype=torch.int64)
                    enabled = (torch.arange(rows) % 3 != 1).to(torch.int32)
                    # Inactive aliases of a live slot zero must not overwrite it.
                    slots[enabled == 0] = 0
                    if rows > 3:
                        slots[-1] = 1025
                        slots[-2] = -1
                    table = torch.randint(0, 8, (rows, 8), dtype=torch.int32)
                    table[:, 0] = 0
                    table[:, -1] = -1
                    positions = torch.randint(-4, 1028, (rows, columns), dtype=torch.int64)
                    positions[:, 0] = 0
                    lengths = torch.randint(1, 1025, (rows,), dtype=torch.int32)
                    active = enabled.clone()
                    expected = write_reference(backing, values, slots, enabled, offset)
                    device = [x.npu() for x in (backing, values, slots, enabled, positions, table, lengths, active)]
                    db, dv, ds, de, dp, dt, dl, da = device
                    equal_bits(db, backing)
                    writer(db, dv, ds, de)
                    equal_bits(db, expected)
                    assert backend.write(db, dv, ds, de.bool(), offset)
                    equal_bits(db, expected)
                    output = torch.zeros((rows, columns, width), dtype=dtype, device="npu")
                    reader(db, dp, dt, dl, da, output)
                    equal_bits(output, read_reference(expected, positions, table, lengths, active, offset, width))
                    wrapped = backend.read(db, dp, dt, dl, da.bool(), offset, width)
                    if columns <= module.MAX_SHORT_READ_POSITIONS:
                        equal_bits(wrapped, output)
                    else:
                        assert wrapped is None
                    report["numerical"].append(
                        dict(dtype=str(dtype), width=width, offset=offset, rows=rows, columns=columns)
                    )
                    if rows != 12:
                        continue
                    # Capture read and write together; replace all metadata
                    # between replays to catch stale scalar-cache reads.
                    stream = torch.npu.Stream()
                    graph = torch.npu.NPUGraph()
                    torch.npu.synchronize()
                    with torch.npu.graph(graph, stream=stream):
                        backend.write(db, dv, ds, de.bool(), offset)
                        if columns <= module.MAX_SHORT_READ_POSITIONS:
                            output = backend.read(db, dp, dt, dl, da.bool(), offset, width)
                        else:
                            output.zero_()
                            reader(db, dp, dt, dl, da, output)
                    for replay in range(6):
                        slots = torch.roll(torch.arange(rows, dtype=torch.int64), replay)
                        enabled = ((torch.arange(rows) + replay) % 3 != 1).to(torch.int32)
                        if replay == 5:
                            enabled.zero_()
                        slots[enabled == 0] = 0
                        active = enabled.clone()
                        positions = torch.roll(positions, 1, dims=1)
                        lengths = torch.full((rows,), 128 * (replay + 1), dtype=torch.int32)
                        table = torch.roll(table, 1, dims=1)
                        values = torch.randn_like(values)
                        expected = write_reference(backing, values, slots, enabled, offset)
                        for target, source in zip(
                            device, (backing, values, slots, enabled, positions, table, lengths, active), strict=True
                        ):
                            target.copy_(source)
                        graph.replay()
                        equal_bits(db, expected)
                        equal_bits(output, read_reference(expected, positions, table, lengths, active, offset, width))
                        report["graph_replay"].append(dict(dtype=str(dtype), width=width, offset=offset, replay=replay))
                # Operator-only timing includes no host data transfers. The
                # initial branch's field slice plus index_copy is the baseline.
                db = torch.randn(103296, 1024, dtype=dtype, device="npu")
                dv = torch.randn(12, width, dtype=dtype, device="npu")
                ds = torch.arange(12, dtype=torch.int64, device="npu")
                de = torch.ones(12, dtype=torch.int32, device="npu")
                reference = partial(original_field_write, db, dv, ds, offset, width)
                candidate = partial(writer, db, dv, ds, de)
                report["microbenchmark"].append(
                    dict(
                        dtype=str(dtype),
                        width=width,
                        offset=offset,
                        reference_us=elapsed_us(reference),
                        candidate_us=elapsed_us(candidate),
                    )
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/cannbot_cache.py"
    report = dict(
        status="running",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope="Operator correctness and microbenchmark only",
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
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
