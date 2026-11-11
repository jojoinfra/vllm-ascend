# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observe one service and retain every round of the frozen end-to-end workload."""

import argparse
import http.client
import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def container_state(name):
    return json.loads(subprocess.check_output(["docker", "inspect", name], text=True))[0]


def check_devices(name):
    info = container_state(name)
    devices = {item["PathOnHost"] for item in info["HostConfig"]["Devices"]}
    devices = {x for x in devices if x.removeprefix("/dev/davinci").isdigit()}
    if devices != {"/dev/davinci1", "/dev/davinci2", "/dev/davinci3", "/dev/davinci4"}:
        raise ValueError("Experiment must use only the four reserved physical cards 1-4")
    live = subprocess.check_output(["docker", "ps", "-q"], text=True).split()
    for other in json.loads(subprocess.check_output(["docker", "inspect", *live], text=True)) if live else []:
        if other["Id"] == info["Id"]:
            continue
        exposed = {item["PathOnHost"] for item in other["HostConfig"].get("Devices") or []}
        if devices & exposed:
            raise RuntimeError(f"Overlapping NPU container: {other['Name']}")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--scenario-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--ready-timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.rounds < 1 or args.ready_timeout <= 0:
        raise ValueError("Round count and readiness observation timeout must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"container": args.container, "status": "waiting", "rounds": [], "goal_10x_met": False}
    summary = args.output_dir / "summary.json"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + args.ready_timeout
    summary.write_text(json.dumps(report, indent=2) + "\n")
    try:
        while True:
            info = check_devices(args.container)
            if not info["State"]["Running"]:
                raise RuntimeError(f"Observed service terminal: {info['State']}")
            try:
                with opener.open(args.base_url.rstrip("/") + "/health", timeout=5) as response:
                    if response.status == 200:
                        break
            except (OSError, http.client.HTTPException):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Readiness observation expired; container retained, re-observe the same container")
            time.sleep(5)
        report.update(status="measuring", image_id=info["Image"], physical_devices=[1, 2, 3, 4])
        print("Service ready; starting frozen workload", flush=True)
        script = Path(__file__).resolve().parents[1] / "benchmark-fixed.py"
        for index in range(args.rounds):
            check_devices(args.container)
            path = args.output_dir / f"round-{index + 1}.json"
            with (args.output_dir / f"round-{index + 1}.log").open("x") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "--container",
                        args.container,
                        "--base-url",
                        args.base_url,
                        "--scenario-dir",
                        str(args.scenario_dir),
                        "--output",
                        str(path),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            row = json.loads(path.read_text())
            if row["status"] != "valid":
                raise RuntimeError(f"Invalid workload round: {path}")
            report["rounds"].append(
                {
                    "path": str(path),
                    "tokens_per_second": row["aggregate_output_tokens_per_second"],
                    "mean_ttft_seconds": row["mean_ttft_seconds"],
                    "mean_latency_seconds": row["mean_latency_seconds"],
                }
            )
            summary.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report["rounds"][-1]), flush=True)
        throughput = [x["tokens_per_second"] for x in report["rounds"]]
        report.update(
            status="passed",
            mean_tokens_per_second=statistics.mean(throughput),
            median_tokens_per_second=statistics.median(throughput),
            minimum=min(throughput),
            maximum=max(throughput),
        )
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        summary.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
