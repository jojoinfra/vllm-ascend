# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check integrated index wrapper and changed-input graph replay on one NPU."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from validate_hc import load_module
from validate_index import compare, reference


def main():
    import torch_npu  # noqa: F401

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.npu.device_count() != 1:
        raise RuntimeError("Requires one isolated NPU")
    torch.npu.set_device(0)
    torch.manual_seed(535)
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/cannbot_index.py"
    report = dict(status="running", source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), graph_replay=[])
    with args.output.open("x") as stream:
        json.dump(report, stream)
    try:
        backend_cls = load_module("cannbot_index").PagedIndexBackend
        qdq = load_module("attention_ops").fp4_quant_dequant
        with torch.inference_mode():
            for heads, ratio in ((8, 2), (32, 2), (32, 1)):
                backend = backend_cls(1024, 512, heads, 128, ratio)
                backing = qdq(torch.randn(2048, 1024, device="npu", dtype=torch.bfloat16))
                query = qdq(torch.randn(3, heads, 128, device="npu", dtype=torch.bfloat16))
                weights = torch.randn(3, heads, device="npu", dtype=torch.bfloat16)
                table = torch.randint(0, 16, (3, 512), device="npu", dtype=torch.int32)
                lengths = torch.full((3,), 256, device="npu", dtype=torch.int32)
                active = torch.ones(3, device="npu", dtype=torch.bool)
                inputs = (backing, query, weights, table, lengths, active, 512, 512)
                actual = backend(*inputs)
                compare(actual, reference(backing, query, weights, table, lengths, active, 512, ratio))
                assert backend(backing, query.float(), weights, table, lengths, active, 512, 512) is None
                assert backend(backing, query, weights, table, lengths, active, 512, 511) is None
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    output = backend(*inputs)
                for replay in range(4):
                    lengths.copy_(torch.tensor([0, 1, 512, 1024][replay], device="npu", dtype=torch.int32))
                    active.copy_(torch.tensor([False, True, True], device="npu"))
                    backing.copy_(qdq(torch.randn_like(backing)))
                    query.copy_(qdq(torch.randn_like(query)))
                    graph.replay()
                    compare(output, reference(backing, query, weights, table, lengths, active, 512, ratio))
                    report["graph_replay"].append(dict(heads=heads, ratio=ratio, replay=replay))
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
