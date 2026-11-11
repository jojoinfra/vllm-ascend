# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded GSM8K and image regression after unprofiled throughput measurements."""

import argparse
import hashlib
import json
import time
import urllib.request
import uuid
from pathlib import Path

import regex as re


def post(base_url, body):
    request = urllib.request.Request(
        base_url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=600) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = (args.reference / "gsm8k-test.jsonl").read_bytes()
    report = dict(
        status="running",
        dataset_sha256=hashlib.sha256(dataset).hexdigest(),
        results=[],
        scope="First eight GSM8K samples and frozen image; bounded integration regression only",
    )
    with args.output.open("x") as stream:
        json.dump(report, stream)
    try:
        for index, line in enumerate(dataset.splitlines()[:8]):
            sample = json.loads(line)
            body = dict(
                model="deepseek-v4.1",
                temperature=0,
                seed=0,
                max_tokens=512,
                chat_template_kwargs=dict(thinking_mode="chat"),
                cache_salt=uuid.uuid4().hex,
                messages=[
                    dict(
                        role="user",
                        content=sample["question"]
                        + "\nPlease solve the problem and end with #### followed by the answer.",
                    )
                ],
            )
            started = time.monotonic()
            response = post(args.base_url, body)
            answer = response["choices"][0]["message"]["content"]
            expected = sample["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
            extracted = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer)
            predicted = extracted[-1].replace(",", "") if extracted else None
            row = dict(
                index=index,
                request=body,
                response=response,
                expected=expected,
                predicted=predicted,
                passed=predicted == expected,
                seconds=time.monotonic() - started,
            )
            report["results"].append(row)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps(dict(index=index, passed=row["passed"], predicted=predicted)), flush=True)
        image_request = json.loads((args.reference / "vision-request.json").read_text())
        image_request["cache_salt"] = uuid.uuid4().hex
        response = post(args.base_url, image_request)
        answer = response["choices"][0]["message"]["content"].strip()
        report["vision"] = dict(response=response, expected="5", passed=answer == "5")
        report["gsm8k_first8_score"] = sum(x["passed"] for x in report["results"]) / 8
        report["status"] = "passed" if report["vision"]["passed"] and report["gsm8k_first8_score"] == 1 else "failed"
        if report["status"] != "passed":
            raise AssertionError("Bounded service regression failed")
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status=report["status"],
                gsm8k_first8_score=report["gsm8k_first8_score"],
                vision_passed=report["vision"]["passed"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
