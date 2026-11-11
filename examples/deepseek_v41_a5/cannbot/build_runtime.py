# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build an offline CANNBot candidate from the exact initial A5 runtime."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

BASE_REF = "4c50751f2204e202c739baf090112727be7e3621"
BASE_IMAGE = "jojo/vllm-ascend:d4ac431-a5-fp8-fp4"
BASE_IMAGE_ID = "sha256:70586cf570b3780fdb5c3c8a23e5e4c250d397240abf4df79e25f86292ac1e8e"
WHEEL_NAME = "cannbotdsl-0.6.0+g91e7c8f-cp312-cp312-manylinux_2_28_aarch64.whl"
WHEEL_SHA256 = "65b0957dd5462ac97e9318c688d95e9e4e71815832ef3b8c73caf0211998c51f"
BASE_RUNTIME_FILES = (
    "vllm_ascend/ascend_config.py",
    "vllm_ascend/models/deepseek_v41/core.py",
    "vllm_ascend/models/deepseek_v41/model.py",
    "vllm_ascend/models/deepseek_v41/dspark.py",
    "vllm_ascend/models/deepseek_v41/attention.py",
    "vllm_ascend/models/deepseek_v41/engram.py",
)
RUNTIME_FILES = BASE_RUNTIME_FILES + (
    "vllm_ascend/models/deepseek_v41/cannbot_hc.py",
    "vllm_ascend/models/deepseek_v41/cannbot_cache.py",
    "vllm_ascend/models/deepseek_v41/cannbot_quant.py",
    "vllm_ascend/models/deepseek_v41/cannbot_fp4.py",
    "vllm_ascend/models/deepseek_v41/cannbot_index.py",
    "vllm_ascend/models/deepseek_v41/cannbot_index_cube.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def image_hashes(image, paths):
    code = (
        "import hashlib,json,pathlib; "
        "root=pathlib.Path('/opt/d4ac431/vllm-ascend'); "
        f"print(json.dumps({{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in {paths!r}}}))"
    )
    return json.loads(
        subprocess.check_output(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "/opt/d4ac431/.venv/bin/python3",
                image,
                "-c",
                code,
            ],
            text=True,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument(
        "--context", type=Path, required=True, help="New directory; existing artifacts are never replaced"
    )
    parser.add_argument("--tag", required=True, help="New local image tag")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", BASE_REF, "HEAD"], check=True)
    if args.wheel.name != WHEEL_NAME or digest(args.wheel.read_bytes()) != WHEEL_SHA256:
        raise ValueError("Use the pinned CANNBotDSL wheel; name or SHA256 differs")
    base_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", BASE_IMAGE], text=True
    ).strip()
    if base_id != BASE_IMAGE_ID:
        raise ValueError("Initial runtime image differs; audit and update the base pin before building")
    if subprocess.run(["docker", "image", "inspect", args.tag], capture_output=True).returncode == 0:
        raise ValueError("Candidate image tag already exists; choose a new version")
    parent_hashes = {
        path: digest(subprocess.check_output(["git", "-C", str(repo), "show", f"{BASE_REF}:{path}"]))
        for path in BASE_RUNTIME_FILES
    }
    if image_hashes(BASE_IMAGE, BASE_RUNTIME_FILES) != parent_hashes:
        raise ValueError("Initial image runtime sources differ from the requested initial branch")
    args.context.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in RUNTIME_FILES:
        destination = args.context / "source" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / path, destination)
        hashes[path] = digest(destination.read_bytes())
    shutil.copyfile(args.wheel, args.context / WHEEL_NAME)
    (args.context / "Dockerfile").write_text(
        f"FROM {BASE_IMAGE}\n"
        f"COPY {WHEEL_NAME} /tmp/{WHEEL_NAME}\n"
        f"RUN /opt/d4ac431/.venv/bin/python3 -m pip install --no-index --no-deps /tmp/{WHEEL_NAME} "
        f"&& rm /tmp/{WHEEL_NAME}\n"
        "COPY source/ /opt/d4ac431/vllm-ascend/\n"
        'LABEL source.stage="cannbot-experimental"\n'
    )
    manifest = dict(
        base_ref=BASE_REF,
        base_image_id=base_id,
        wheel_sha256=WHEEL_SHA256,
        parent_hashes=parent_hashes,
        source_sha256=hashes,
        tag=args.tag,
        npu_validated=False,
    )
    manifest_path = args.context / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run(["docker", "build", "--network", "none", "-t", args.tag, str(args.context)], check=True)
    if image_hashes(args.tag, RUNTIME_FILES) != hashes:
        raise RuntimeError("Built image failed runtime source verification")
    manifest["image_id"] = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", args.tag], text=True
    ).strip()
    manifest["runtime_source_verified"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
