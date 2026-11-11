# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile the frozen requests, keeping profiled latency out of performance claims."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from run_fixed import check_devices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--scenario-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / "benchmark-fixed.py"
    spec = importlib.util.spec_from_file_location("frozen_benchmark", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    raw = (args.scenario_dir / "scenario.json").read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if checksum != (args.scenario_dir / "scenario.sha256").read_text().split()[0]:
        raise ValueError("Frozen scenario checksum mismatch")
    scenario = json.loads(raw)
    report = dict(
        status="running",
        usable_as_unprofiled_performance=False,
        scenario_sha256=checksum,
        profile_scope="Worker steps selected by profiler_config; not necessarily the complete request interval",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream)
    started = False
    original = bench.request

    def traced_request(base_url, scenario, sample, output_tokens, barrier=None):
        nonlocal started
        result = original(base_url, scenario, sample, output_tokens, barrier)
        if sample["id"] == "warmup":
            with bench.fetch(base_url, "/start_profile", {}) as response:
                response.read()
            started = True
        return result

    try:
        check_devices(args.container)
        report["runtime"] = bench.runtime_metadata(args.container, scenario)
        bench.check_tokenization(args.base_url, scenario)
        bench.request = traced_request
        report.update(bench.measure(args.base_url, scenario))
        report["status"] = "profiled_valid_workload"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        bench.request = original
        try:
            if started:
                with bench.fetch(args.base_url, "/stop_profile", {}) as response:
                    response.read()
                report["profiler_stopped"] = True
        except Exception as error:
            report.update(status="failed", stop_error=str(error))
            raise
        finally:
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "usable_as_unprofiled_performance", "profiler_stopped")}))


if __name__ == "__main__":
    main()
