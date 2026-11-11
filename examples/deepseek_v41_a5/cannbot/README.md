# CANNBot A5 optimization experiment

Base: `dsv41-a5-fp8-fp4` at `4c50751f2204e202c739baf090112727be7e3621`.
The goal is at least 10x end-to-end throughput with 1–4 NPUs and the original
FP8/FP4 checkpoint. The current experiment reserves physical NPU **1, 2, 3, 4**.
Do not use physical 0, 5, 6, 7; another experiment owns those cards.

The first candidate replaces HC4 gates and the 20-step Sinkhorn normalization
with a CANNBotDSL kernel. The projection and all model equations remain in
the initial implementation. `v41_cannbot_hc` defaults to false; setting it to
true enables this candidate for both target and DSpark blocks. It requires
CANNBotDSL `0.6.0+g91e7c8f`, HC width 4 and contiguous FP32 coefficients.
Compilation is cached per block with a symbolic token dimension, avoiding
shape compilation in the request path. Checkpoint parameters are passed as
detached tensor aliases to match DSL 0.6's strict DLPack provider type. The
kernel does not transfer tensor values to the host.

The next candidate adds `v41_cannbot_cache`, also false by default. It writes
live decode rows directly into the cache's backing storage and reads compressor
groups of up to eight positions. This avoids materializing a full cache field
for each small update. Large window/index gathers and scalar token history use
the original implementation. Field kernels compile during model construction;
device metadata loads bypass the scalar cache so graph replays see new slots,
lengths and live masks. The existing invariant of unique live write slots is
required; inactive rows perform no writes, including aliases of slot zero.

## Validation gates

1. Run the core/model/DSpark CPU regression tests.
2. Compile the CANNBot kernel, then run its numerical and graph replay checks
   on one reserved NPU. The validator compares against this branch's `core.py`,
   including multiple iteration counts, epsilon values, saturation, empty and
   non-multiple token counts. A compile-only pass is not numerical validation.
3. Run the initial image and candidate serially on the same four physical cards.
   Preserve fixed prompts, tokenization hashes, eight concurrent requests,
   256 output tokens per request, warmup, sampling and zero-cache-hit checks.
4. Keep every raw round and report aggregate repeated results, TTFT and latency.
   Profile separately; exclude profiled rounds from the performance comparison.
5. Check API, text and image correctness. Audit runtime source hashes and verify
   the CANNBot kernel executes before attributing any performance change to it.

No 10x performance result has been established. Operator microbenchmarks are
not end-to-end results. Historical data in `../fixed-throughput/` remains
unchanged. The NPU1–4 experiment uses a separate scenario directory with the
same requests and a fresh initial-image baseline; pass it with `--scenario-dir`
to `../benchmark-fixed.py`.

## Tools

`build_runtime.py` checks the pinned local initial image and wheel, verifies
the parent source hashes against the base commit, builds offline without any
NPU devices, and verifies the candidate image's source hashes. It refuses an
existing image tag or build context. Its manifest identifies the exact source
snapshot, even before a commit.

```bash
python3 examples/deepseek_v41_a5/cannbot/build_runtime.py \
  --wheel /home/jojo/deepseek-v41-4-7/wheels/cannbotdsl-0.6.0+g91e7c8f-cp312-cp312-manylinux_2_28_aarch64.whl \
  --context /path/to/new-build-directory \
  --tag jojo/vllm-ascend:new-cannbot-candidate
```

In the matching CANN environment, `validate_hc.py --mode compile --output FILE`
performs device-free compilation. `--mode npu` requires a container exposing
exactly one reserved NPU, plus the standard A5 support devices (`uburma`,
`ubcore`, `ubmempfd`, `ubagg`, `ummu`, `davinci_manager`, `hisi_hdc`), driver,
firmware and URMA mounts. Both modes refuse an existing result file.

`start_runtime.py` starts a fresh four-card container on physical NPU1–4 and
refuses any card overlap with a running container. Supply `--image`, `--name`
and `--deployment` containing the unchanged initial `serve.sh` and fixed
`reference/chat_template.jinja`. Add `--enable-hc` only after numerical checks.
The default local service port is 8010. It never stops or replaces containers.
Add `--enable-cache` to select the direct cache access candidate. Its
`validate_cache.py --output FILE` checks complete backing-storage bits and
changing metadata during graph replay on one isolated NPU. The separate
`benchmark_cache.py --output FILE` compares graph replay costs, including the
large gathers that deliberately retain the original implementation. These
operator measurements are not end-to-end performance results.

For a separate profiling service, also supply `--profile-dir` with a new output
directory. By default sampling starts after ten worker steps and stops after
the counter exceeds four recorded steps, limiting expensive trace export.
Override with `--profile-delay-iterations` and `--profile-max-iterations`; a zero
maximum captures until the stop request. After the service is healthy,
`capture_fixed.py --container NAME
--scenario-dir DIRECTORY --output FILE` runs the frozen warmup, starts profiling,
runs the fixed requests, and stops profiling. Its output explicitly excludes
profiled throughput from performance claims. Keep the native traces alongside
the capture report; a successful HTTP response alone does not establish that
every rank exported a complete trace.
Daemon workers can require offline parsing with
`torch_npu.profiler.profiler.analyse`. Keep original data if export is incomplete.
`analyze_profile.py --profile-dir DIRECTORY --output FILE` requires all four
`kernel_details.csv` exports and reports per-rank duration sums and unions;
these cannot be added into end-to-end latency.

## First measured candidate

The HC v3 image passed 96 numerical cases, 20 graph replays, 71 CPU regressions
and the eight-question GSM8K integration regression. On physical NPU1–4, three
unprofiled initial-image rounds averaged 75.33 output tokens/s; HC v3 averaged
76.54 output tokens/s. The ratio of means is 1.016, while the ratio of medians
is 0.991. The ranges overlap, so these rounds do not establish a stable gain.
The other experiment on the shared host was not controlled. The 10x target
against this fresh baseline is 753.27 output tokens/s and remains unmet.

Raw rounds and the comparison are retained under `baseline-rounds-v2/`,
`hc-v3-rounds/`, and `hc-v3-comparison.json` in the evidence directory below.
The GSM8K check covers only eight examples, not the full benchmark. The
candidate remains opt-in while profiling and further optimization continue.

Local evidence and the separate NPU1–4 scenario are under
`/home/jojo/dsv41-cannbot-10x-20261111/`. Check current JSON/log results before
using a candidate; a successful build manifest alone is not an acceptance gate.
