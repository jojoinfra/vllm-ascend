# CANNBot A5 optimization experiment

Base: `dsv41-a5-fp8-fp4` at `4c50751f2204e202c739baf090112727be7e3621`.
The goal is at least 10x end-to-end throughput with 1–4 NPUs and the original
FP8/FP4 checkpoint. The current experiment reserves physical NPU **1, 2, 3, 4**.
Do not use physical 0, 5, 6, 7; another experiment owns those cards.

## Measured result and scope

The additional **16-concurrent-request scenario exceeds 10x**: three initial
rounds average **16.52 output tokens/s**, and three optimized rounds average
**277.04 output tokens/s**, an observed **16.77x** improvement. The original
eight-request scenario remains below the target. These are complete batch
measurements, including prefill, on the same TP4/EP4 cards and FP8/FP4 weights,
with approximately 100 input tokens and exactly 256 output tokens per request.

| Workload and candidate configuration | Initial mean tokens/s | Candidate mean tokens/s | Observed ratio |
| --- | ---: | ---: | ---: |
| c16, graph16 v1, captures `[4,8,16]` | 16.52 | 277.04 | 16.77x |
| c8, same graph16 v1 image and configuration | 75.33 | 167.94 | 2.23x |
| c8, earlier cache-slots v3, captures `[4,8]` | 75.33 | 175.50 | 2.33x |

The c16 baseline rounds are 16.03, 16.84 and 16.70 tokens/s; the candidate
rounds are 276.00, 276.08 and 279.05. Both initial and candidate services passed
16 simultaneous text checks (two copies of the same first eight GSM8K problems)
and the frozen image check. This is bounded integration coverage, not a full
accuracy evaluation. All throughput rounds require zero cached prompt tokens.

The improvement combines CANNBot kernels, direct cache access, disabling DSpark
speculation and extending ordinary decode graph coverage to 16 requests. The
initial service already permits 16 sequences, but its largest graph covers only
eight speculative requests; larger batches fall back to eager execution. The
candidate log confirms actual 16-active-row replay on all four workers, with
at least 512 replays recorded per worker. The result is not a claim that any
single CANNBot kernel provides 16.77x end-to-end speedup.

Evidence is retained under `/home/jojo/dsv41-cannbot-10x-20261111/`:
`baseline-c16-rounds-v1/`, `graph16-v1-rounds/`, `graph16-v1-c8-rounds/`,
`baseline-c16-quality-v1.json`, `graph16-v1-quality.json` and
`graph16-v1-service.log`. A fresh initial-image confirmation after the candidate
measured **15.97 tokens/s**, giving **17.35x** against the candidate mean.
Even the slowest candidate round divided by the fastest initial round is
**16.39x**. `graph16-v1-results.json` audits scenario hashes, unchanged workloads,
runtime differences, output counts, cache hits, quality and active graph replay.
These remain bounded shared-host measurements, not a universal performance
guarantee. The image is
`jojo/vllm-ascend:dsv41-cannbot10x-graph16-v1`, digest
`sha256:e2b8af3808106cd45a7d2af0c9994f7cbbfe4504860fb2166046fc57245216d1`;
`build-graph16-v1/manifest.json` records exact runtime source hashes.

To reproduce the optimized configuration with the pinned local image and
deployment, choose a new container name while the reserved cards are free:

```bash
python3 examples/deepseek_v41_a5/cannbot/start_runtime.py \
  --image jojo/vllm-ascend:dsv41-cannbot10x-graph16-v1 \
  --name jojo-cannbot10x-graph16-reproduce-npu1-4 \
  --deployment /home/jojo/dsv41-cannbot-10x-20261111/deployment \
  --decode-mode autoregressive --autoregressive-max-requests 16 \
  --enable-hc --enable-cache --enable-cache-slots \
  --enable-fp8 --enable-fp4 --enable-index \
  --enable-norm --enable-rope --enable-residual

python3 examples/deepseek_v41_a5/cannbot/run_fixed.py \
  --container jojo-cannbot10x-graph16-reproduce-npu1-4 \
  --scenario-dir /home/jojo/dsv41-cannbot-10x-20261111/scenario-npu1-4-c16-autoregressive-v3 \
  --output-dir /path/to/new-results-directory
```

## Optimization history

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
   Preserve each frozen scenario's prompts, tokenization hashes, concurrency,
   256 output tokens per request, warmup, sampling and zero-cache-hit checks.
4. Keep every raw round and report aggregate repeated results, TTFT and latency.
   Profile separately; exclude profiled rounds from the performance comparison.
5. Check API, text and image correctness. Audit runtime source hashes and verify
   the CANNBot kernel executes before attributing any performance change to it.

The original c8 scenario has not reached 10x; the additional c16 result and its
scope are reported above. Operator microbenchmarks are not end-to-end results.
Historical data in `../fixed-throughput/` remains
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

With direct cache access enabled alongside HC, the next three rounds measured
84.32, 83.67 and 83.96 output tokens/s (mean 83.99). The ratio to the initial
mean is 1.115; this is an observed sequential comparison, with no alternating
A/B confirmation on the shared host. All eight GSM8K checks and the image smoke
test passed. See `cache-v1-rounds/` and `cache-v1-comparison.json`. The cache
backend also passed 30 bitwise cases, 72 changing-input graph replays and 99 CPU
regressions. The 10x goal is still unmet.

Local evidence and the separate NPU1–4 scenario are under
`/home/jojo/dsv41-cannbot-10x-20261111/`. Check current JSON/log results before
using a candidate; a successful build manifest alone is not an acceptance gate.

## FP8 activation candidate and parallel-layout experiment

`v41_cannbot_fp8` (launcher `--enable-fp8`) selects BF16 activation QDQ
for target and DSpark attention; other dtypes retain the reference path.
The kernel preserves 32-value E4M3 groups, the FP32 amax floor, upward E8M0
scale rounding, ties-to-even conversion, signed zeros and the original
bit-based NaN scale behavior. `validate_quant.py --output FILE` checks all
65,536 BF16 encodings, ordinary amplitudes, mixed nonfinite groups, in-place
views and changing-input graph replay. Initial operator measurements show
2.4–13x speedups across the measured shapes; this is not a model speedup.

`start_runtime.py --parallel-layout dp4` explicitly changes TP4/DP1/EP4 to
TP1/DP4/EP4 on the same physical cards. It disables DSA-CP and adds graph
buckets 6 and 12 for smaller per-replica batches. Use a separate scenario
with `tensor_parallel_size: 1` and `data_parallel_size: 4`, retaining all
requests, token hashes, sampling, output lengths, cache checks and checkpoint.
The default remains the original TP4 layout. Results from this experiment
must identify the topology change when compared with the initial baseline.

Runtime metadata now records the **last** occurrence of CLI options, matching
launch overrides. Older retained TP4 result files recorded the first
`--additional-config`, omitting opt-in flags from that metadata field. Their
actual startup arguments and device traces independently establish HC/cache
execution. Keep those original result files; do not rewrite historical data.

The FP8 v2 TP4 integration measured **101.45, 97.73, 102.36 output tokens/s**,
mean **100.51** versus the initial mean **75.33** (observed **1.334x**).
It passed 114 CPU regressions, the FP8 numerical/graph validator, GSM8K first
8/8 and the image smoke check. This is a sequential shared-host comparison;
alternating A/B confirmation remains outstanding. The 10x target is not met.
The cache-only TP1/DP4 attempt failed startup with about **−3.4 GiB** available
KV-cache budget per card; it has no throughput result.

## Experimental paged index scores

`--enable-index` sets `v41_cannbot_index`. The initial CANNBot vector kernel reads only
live compressed positions directly from their physical cache pages, preserving
BF16 rounding after dot products, weighted products and head reduction. The
original masking, candidate selection, deterministic top-k and TP reduction
remain in place. Unsupported layouts use the original path. Output rows must
be aligned to 16 BF16 elements; one core owns each row to avoid concurrent
sub-cacheline writes. Signed zeros in masked/zero scores are numerically
interchangeable for the subsequent selection.

The vector-only v2 image is **not suitable as a general performance default**. At actual
compression ratios 1 and 2 it passed numerical checks and graph replay through
65,536 tokens, but its vector dot products regress substantially for long
contexts. At 256-token contexts the measured index operation improved about
5–10x; at 65,536 tokens it was about 22–45x slower. These are operator timings,
not end-to-end results. Keep the option disabled unless explicitly evaluating
short workloads. The cube follow-up below addresses this limitation.

`validate_index.py` checks page mappings, active masks, changing queries,
weights and cache contents, full context lengths, and repeated graph execution.
`validate_index_wrapper.py` checks the prepared backend, metadata conversion,
unsupported-input fallbacks and graph replay through the model-facing API.
The image also replaces Engram's unsupported tensor shift with exact INT32
multiplication, avoiding a CPU fallback when staging embedding rows; all 256
E8M0 scale encodings were checked on NPU.

The index v2 image (HC/cache/FP8/vector index plus Engram shift removal)
measured **106.09, 101.06, 106.36 output tokens/s**, mean **104.50**, with
GSM8K first8 8/8 and image checks passing. This is an observed **1.387x**
versus the initial mean, and about 4% beyond FP8 v2; the contributions of the
two added changes are not separately measured.

`--decode-mode autoregressive` disables DSpark and selects graph buckets
`[4,8]` for TP4. Use a separate scenario declaring
`runtime_variant: {speculation: disabled, capture_sizes: [4,8]}` and retain
all original input/output and quality requirements. The benchmark rejects
mismatching final runtime flags. The original workload's measured draft-token
acceptance was about 18%; this experiment tests whether speculative overhead
outweighs that acceptance.

The first autoregressive launch used an unsupported 16-row capture bucket;
it failed the existing model guard (maximum eight one-token requests). Its
failed result is retained. The corrected variant uses buckets `[4,8]`.

The corrected autoregressive variant measured **121.64, 123.51, 123.39 output
tokens/s**, mean **122.84**: an observed **1.631x** versus the initial branch,
and **1.176x** versus the same image with DSpark. GSM8K first8 and the image
check passed. Inputs, outputs, checkpoint and physical cards remain identical;
the decoding policy and capture buckets change explicitly. These sequential
shared-host rounds do not establish an alternating A/B result. The 10x target
remains unmet. See `index-autoreg-v2-rounds/`,
`index-autoreg-v2-comparison.json` and `index-autoreg-v2-quality.json`.

## Cube index and gathered-cache follow-up

The experimental backend now selects a cube matrix kernel for block size 128,
compression ratios 1/2 and 8/32 heads. Its separate reduction keeps BF16
per-head dots and products. Graph shapes retain their scratch allocations,
and explicit synchronization protects reused buffers. Capacities not divisible
by 64 use the original reference path. Other supported layouts retain the
vector implementation; the option remains disabled by default.

`index-cube-npu-v3.json` records nine numerical cases and 45 changing-input
replays. At 65,536 tokens, measured cube index times were 241, 345 and 673 us
for (8 heads, ratio2), (32 heads, ratio2) and (32 heads, ratio1), versus reference
359, 383 and 629 us. The last case still regresses about 7%; there is no
universal speedup claim. The model-facing wrapper passed 12 additional replays.

With cache optimization enabled, larger reads gather requested backing rows
before slicing fields when this moves fewer elements than slicing the entire
cache. Otherwise they retain the original ordering. The isolated 12-row,
640-column, 512-wide BF16 read improved from about 150 to 66 us. Large 8192-column
reads can regress and are excluded by the element-count guard. CPU regression
covers both BF16 and mixed FP32 cache fields. These are operator measurements;
the combined candidate measured 143.91, 147.35 and 149.86 output tokens/s,
mean **147.04** (observed **1.952x** initial, **1.197x** previous autoregressive
variant). GSM8K first8 8/8 and the image check passed. The shared-host comparison is
sequential; the 10x target remains unmet.

The separate autoregressive profile exported all four ranks. Each rank records
320 FP8 QDQ launches, 32 vector index launches, 320 HC launches, 220 cache
writes and 24 direct short reads. The profile is not included in throughput
results. See `profile-index-autoreg-v1-summary.json` for counts and limitations.

## FP4 activation candidate

`--enable-fp4` selects BF16 QDQ for index Q/K (group32/E8M0) and compressed
KV (group16/E4M3). Other dtypes and formats use the reference. Both scale rules,
nearest-even E2M1 conversion and BF16 output remain explicit; pairing adjacent
16-value groups prevents concurrent writes to the same output cache line.

`validate_fp4.py --group 32 --scale-format e8m0 --output FILE` and the
corresponding `--group 16 --scale-format e4m3` command each passed 57 numerical
checks and ten changing-input graph replays. Every BF16 encoding is compared
with the unchanged initial NPU reference. That reference itself differs from
CPU on 190 E8M0 subnormal outputs, flushing them to signed zero; the fusion
matches the initial NPU exactly for those inputs. Ordinary amplitudes, mixed
nonfinite values, strided in-place views and graph updates also retain CPU
reference checks. No CPU difference occurred for the exhaustive E4M3 case.

The measured index-query QDQ shape `[2,32,128]` improved from about 50 to
3.6 us (E8M0) and 62 to 4.7 us (E4M3). These are isolated operation timings.
Model integration passed 120 CPU regressions and two NPU dispatch tests.
The FP4 image measured **148.96, 149.66, 150.93 output tokens/s**, mean
**149.85** (observed **1.989x** initial and **1.019x** cube/gather). GSM8K first8
and the image check passed. The small incremental gain is not established by
alternating A/B runs; all comparisons remain sequential on a shared host.
The 10x target remains unmet.

## RMS normalization candidate

`--enable-norm` enables CANNBot BF16 RMS normalization in compatible target and
DSpark normalization modules. Weight dtype may be BF16 or FP32; unsupported
devices, input dtypes and widths retain the reference. The option defaults off.
Reduction order and reciprocal-square-root evaluation can change rounding:
this candidate is **not bit-exact**. Its declared limit is one BF16 ULP per
finite output with matching nonfinite classifications.

`norm-npu-v1.json` passed 320 numerical cases and 80 changing-input graph
replays. Across 9,433,600 finite outputs, 18 differed by one BF16 ULP and the
rest matched. The 5120-wide, two-row operation measured about 14.7 us for the
reference and 3.0 us for the candidate. These are isolated operation timings.
Integration passed 121 CPU regressions and two NPU tests covering noncontiguous
views, empty inputs, NaN/Inf, changed weights and repeated graph execution.
Three fixed rounds measured **155.17, 158.30, 157.95 output tokens/s**, mean
**157.14** (observed **2.086x** initial and **1.049x** FP4 v1). GSM8K first8
8/8 and the image check passed. This remains a sequential shared-host comparison;
the 10x target is unmet.

The follow-up constant-input audit from 1e-40 through 1e20 also matched the
initial NPU path, including the FP32 square-sum overflow at 1e18 and above.
The four-rank profile confirms 676 RMS, 64 FP4, 320 FP8 and 32 cube index
launches per rank in its captured window; see `profile-norm-v1-summary.json`.

## Rotary application candidate

`--enable-rope` fuses the adjacent-pair BF16 rotation arithmetic while retaining
the original FP32 phase, sine and cosine operations. It supports contiguous
single-batch inputs with a 64-wide rotary tail, pitch128 with 1/8/32 heads or
pitch512 with 1/16/64 heads, and at most 48 tokens. Other layouts and dtypes use
the reference. Both target and DSpark attention initialize this default-off
backend; each instance caches kernels by pitch, head count and direction.

`validate_rope.py` passed 61 numerical cases and 80 changing-input graph replays,
checking exact finite output bits and nonfinite classifications against the
initial NPU operation. This includes all BF16 encodings and unchanged prefix
elements. Two-token operation timings improved from about 15-23 us to 3.4 us.
The 128-token, 64-head case had little gain, so integration conservatively
retains the reference above 48 tokens. These are isolated operation timings.
Integration passed 122 CPU regressions and four NPU tests covering forward and
inverse graph replays, rank3/rank4 dispatch and unsupported-input fallbacks.
Three fixed rounds measured **162.47, 164.45, 164.30 output tokens/s**, mean
**163.74** (observed **2.174x** initial and **1.042x** normalization v1). This
retains the separately declared autoregressive decoding policy and the frozen
requests, weights, token counts, physical cards and zero cache-hit checks.
GSM8K first8 8/8 and the image check passed. The comparison is sequential on a
shared host; the 10x target remains unmet.

## HC residual mixing candidate

`--enable-residual` enables BF16 HC4 pre/post mixing in 5120-wide target and
DSpark blocks for up to 128 rows. Unsupported sizes, dtypes and devices retain
the reference. The default is off. The reduction explicitly computes
`(a+c)+(b+d)`, matching the initial NPU path: sequential summation changed
22 of 655,360 BF16 outputs in the order audit, while the chosen order matched
all of them.

The 256-element tile version passed 182 numerical checks and 60 changing-input
graph replays, including all BF16 encodings and strided coefficients. Checks
require exact finite output bits and matching nonfinite classifications.
At two rows, pre-mixing measured about 10.0 to 3.3 us and post-mixing 13.8 to
4.7 us. The larger tiles also improved all measured row counts through 128;
the earlier 64-element version regressed for 48/128 rows and is not deployed.
These are operator measurements. Integration passed 123 CPU regressions and
one NPU test covering eight graph replays, strided residuals, NaN/Inf and
dtype/row-count fallbacks. End-to-end measurement is recorded separately.
Three fixed rounds measured **166.03, 170.11, 170.76 output tokens/s**, mean
**168.97** (observed **2.243x** initial and **1.032x** RoPE v1). GSM8K first8
8/8 and the image check passed. The decoding policy and workload remain as
declared for the previous autoregressive candidate. These sequential runs on
a shared host do not establish a controlled A/B effect; the 10x target is unmet.
The complete four-rank profile contains 40,976 kernel records per rank, including
544 rotary applications and 320 each of HC pre/post mixing, confirming that the
new paths execute. Profile latency is excluded from throughput comparisons.

## Cache read metadata candidate

`--enable-cache-slots` combines read-slot lookup and validity masking for
128-token pages and position matrices whose column count is a multiple of 64.
It supports tables up to 512 blocks, positive storage sizes fitting int32,
int64 positions, int32 page tables, int32/int64 lengths and boolean active flags. Unsupported
inputs retain the original operations. The option defaults off and leaves the
cache values and field layouts unchanged.

Both halves of every int64 position participate in validity checks, so large
or negative positions cannot wrap to valid low 32-bit positions. Both halves of
int64 lengths are also compared, retaining the original signed comparison and
position-clamping behavior. Page bounds
are checked before multiplying physical pages into int32 slots. Each cache
owns a bounded compilation cache; lengths, page tables and active flags remain
dynamic graph inputs.

`cache-slots-npu-v1.json` passed 40 exact numerical checks and 24 changing-input
graph replays, including int64 position extremes, int32 page extremes, partial
physical pages and empty batches. At two rows and 640 columns, the metadata
operations measured about 27 us for the reference and 3.4 us for the fusion.
Integration passed 124 CPU regressions and six NPU cache-read tests with 36
graph replays across raw, compressed and index fields for compression ratios
one and two. Both BF16 and FP32 storage and nonaligned fallback are covered.
These are operation/integration checks; full-model results are recorded separately.

The first full-model image retained the fallback because the real metadata
builder uses int64 lengths while v1 admitted only int32. Its mean 166.44 tokens/s
is **not a measurement of this fusion**; `cache-slots-v1-audit.json` retains that
finding. The corrected kernel passed 98 numerical cases (including int64 length
boundaries) and 24 graph replays. Twelve NPU field-read tests now cover both
length dtypes with 72 replays; 13 focused CPU regressions also passed. The
int64-length metadata operation measured about 26.6 to 3.3 us for a 512-block
table. Image v3 adds a once-per-process admission log showing actual dtypes;
the v2 build without that diagnostic was not served.

Image v3 admitted the fusion on all four workers with int64 positions and
lengths, 512 table columns and maximum length65536. Three fixed c8 rounds
measured **175.07, 176.89, 174.55 output tokens/s**, mean **175.50**, or
**2.330x** the initial branch and **1.039x** HC residual v1. GSM8K first8
8/8 and the image check passed. These remain sequential shared-host results
with the explicitly declared autoregressive policy; the 10x target is unmet.

## Sixteen-request graph experiment

The initial deployment permits 16 sequences, but its graph capture sizes cover
only eight speculative requests. A separate c16 workload repeats each original
c8 prompt twice with unique request IDs and cache salts. It retains the same
prompt/output lengths, weights, hardware and maximum sequences. Measure the
unchanged initial image on this scenario before comparing a candidate; c8
results remain separate.

`--decode-mode autoregressive --autoregressive-max-requests 16` explicitly
selects capture sizes `[4,8,16]` with TP4. The launcher still defaults to eight
requests. Only the ordinary one-token decode graph limit is expanded; DSpark
retains its eight-request bound. This experiment requires a newly built image
containing the expanded limit. The change passed 125 related CPU regressions,
including sixteen-row Engram staging, reordered live rows and inactive rows.
One additional focused test covers all sixteen live histories, request
reordering and page boundaries; the twelve focused graph/staging tests passed.
Full-service c16/c8 throughput and concurrent quality results are reported above.

For quality checks at the expanded batch size, `validate_service.py
--concurrency 16` submits two copies of each original first-eight GSM8K sample
at once, with independent cache salts, then runs the frozen image request.
The default remains eight serial text requests. The report records concurrency,
repetitions, all sixteen answers and their original sample indices. This is
still an eight-problem regression, not a broader accuracy evaluation.
