# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Start a four-card experiment only on this task's reserved physical NPU1-4.

Refuse overlaps and existing names. Never stop or replace another container.
The unchanged initial serve.sh supplies all workload/runtime defaults.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

DEVICES = (1, 2, 3, 4)
SUPPORT_DEVICES = ("davinci_manager", "hisi_hdc", "uburma", "ubcore", "ubmempfd", "ubagg", "ummu")
TEMPLATE_SHA256 = "d959d804d5101b79a49b6ff1bf3c54cd5affa9a5a78c503d925f3090040b31c3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--parallel-layout", choices=("tp4", "dp4"), default="tp4")
    parser.add_argument("--enable-hc", action="store_true")
    parser.add_argument("--enable-cache", action="store_true")
    parser.add_argument("--enable-fp8", action="store_true")
    parser.add_argument("--enable-index", action="store_true")
    parser.add_argument("--profile-dir", type=Path, help="New output directory for a separate profiling run")
    parser.add_argument("--profile-delay-iterations", type=int, default=10)
    parser.add_argument("--profile-max-iterations", type=int, default=4, help="Use zero to capture until stop_profile")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("Port must be in 1..65535")
    if args.profile_delay_iterations < 0 or args.profile_max_iterations < 0:
        raise ValueError("Profiler iteration controls must be nonnegative")
    if subprocess.run(["docker", "container", "inspect", args.name], capture_output=True).returncode == 0:
        raise ValueError("Container name exists; retain it and use a new stage name")
    live = subprocess.check_output(["docker", "ps", "-q"], text=True).split()
    reserved = {f"/dev/davinci{i}" for i in DEVICES}
    if live:
        for container in json.loads(subprocess.check_output(["docker", "inspect", *live], text=True)):
            exposed = {item["PathOnHost"] for item in (container["HostConfig"].get("Devices") or [])}
            overlap = reserved & exposed
            if overlap:
                raise RuntimeError(f"Reserved cards still exposed by {container['Name']}: {sorted(overlap)}")
    deployment = args.deployment.resolve()
    template = deployment / "reference/chat_template.jinja"
    if hashlib.sha256(template.read_bytes()).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("Chat template differs from the initial branch")
    repo = Path(__file__).resolve().parents[3]
    initial_script = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "show",
            "4c50751f2204e202c739baf090112727be7e3621:examples/deepseek_v41_a5/serve.sh",
        ]
    )
    if (deployment / "serve.sh").read_bytes() != initial_script:
        raise ValueError("serve.sh must match the initial branch exactly")
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        args.name,
        "--label",
        "owner=jojo",
        "--label",
        "workload=dsv41-cannbot10x",
        "--log-opt",
        "max-size=100m",
        "--log-opt",
        "max-file=3",
        "--ulimit",
        "memlock=-1:-1",
        "--cap-add",
        "IPC_LOCK",
        "--security-opt",
        "label=disable",
        "--shm-size",
        "64g",
        "-p",
        f"127.0.0.1:{args.port}:8000",
    ]
    for path in sorted(reserved) + [f"/dev/{name}" for name in SUPPORT_DEVICES]:
        command += ["--device", path]
    for source, target in (
        ("/usr/local/Ascend/driver", "/usr/local/Ascend/driver"),
        ("/usr/local/Ascend/firmware", "/usr/local/Ascend/firmware"),
        ("/usr/local/bin/npu-smi", "/usr/local/bin/npu-smi"),
        ("/usr/lib64/urma", "/usr/lib64/urma"),
        ("/mnt/weight/DeepSeek-V4.1-Flash-1017", "/models/deepseek-v41"),
        (str(deployment), "/deployment"),
    ):
        command += ["-v", f"{source}:{target}:ro"]
    if args.profile_dir:
        args.profile_dir.mkdir(parents=True, exist_ok=False)
        command += ["-v", f"{args.profile_dir.resolve()}:/profiles"]
    command += ["--entrypoint", "bash", args.image, "/deployment/serve.sh"]
    if args.profile_dir:
        command += [
            "--profiler-config",
            json.dumps(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": "/profiles",
                    "torch_profiler_with_stack": False,
                    "delay_iterations": args.profile_delay_iterations,
                    "max_iterations": args.profile_max_iterations,
                }
            ),
        ]
    if args.parallel_layout == "dp4":
        command += [
            "--tensor-parallel-size",
            "1",
            "--data-parallel-size",
            "4",
            "--compilation-config",
            json.dumps(dict(mode=0, cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[6, 12, 24, 48])),
        ]
    if args.enable_hc or args.enable_cache or args.enable_fp8 or args.enable_index or args.parallel_layout == "dp4":
        additional = dict(
            enable_dsa_cp=args.parallel_layout == "tp4", enable_flashcomm1=False, enable_shared_expert_dp=True
        )
        if args.enable_hc:
            additional["v41_cannbot_hc"] = True
        if args.enable_cache:
            additional["v41_cannbot_cache"] = True
        if args.enable_fp8:
            additional["v41_cannbot_fp8"] = True
        if args.enable_index:
            additional["v41_cannbot_index"] = True
        command += [
            "--additional-config",
            json.dumps(additional),
        ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
