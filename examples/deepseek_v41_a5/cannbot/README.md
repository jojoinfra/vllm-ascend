# CANNBot fusion on the A5 adaptation baseline

This branch starts at `dsv41-a5-fp8-fp4`, commit
`4c50751f2204e202c739baf090112727be7e3621`, before the Triton fusion
experiments. It adds optional CANNBot kernels and their integration tests in one
commit. It does not include the later Engram arithmetic optimization,
16-request decode graph extension, TP1 deployment changes or experiment history.

## Enable the kernels

Install CANNBotDSL `0.6.0+g91e7c8f` in the existing A5 runtime. All switches default
to false, and the optional DSL modules are imported only when enabled. Keep the
baseline model, quantization, parallelism, graph and DSpark settings. Add the
required switches to the existing `vllm serve --additional-config` JSON:

```json
{
  "enable_dsa_cp": true,
  "enable_flashcomm1": false,
  "enable_shared_expert_dp": true,
  "v41_cannbot_hc": true,
  "v41_cannbot_cache": true,
  "v41_cannbot_cache_slots": true,
  "v41_cannbot_fp8": true,
  "v41_cannbot_fp4": true,
  "v41_cannbot_norm": true,
  "v41_cannbot_rope": true,
  "v41_cannbot_residual": true,
  "v41_cannbot_index": true
}
```

| Switch | Fused operation |
| --- | --- |
| `v41_cannbot_hc` | HC4 gates and Sinkhorn normalization |
| `v41_cannbot_cache` | Direct decode cache writes and short reads |
| `v41_cannbot_cache_slots` | Paged read slots and validity masks |
| `v41_cannbot_fp8` | BF16 activation FP8 quantization/dequantization |
| `v41_cannbot_fp4` | BF16 activation FP4 quantization/dequantization |
| `v41_cannbot_norm` | BF16 RMS normalization |
| `v41_cannbot_rope` | Short BF16 rotary application |
| `v41_cannbot_residual` | HC4 pre/post residual mixing |
| `v41_cannbot_index` | Paged index scoring |

Shape, dtype and device guards retain the original fallback paths. HC gate
fusion requires HC width four. RMS normalization permits at most one BF16 ULP
of finite-output difference in the numerical validator. Paged index scoring
remains experimental; performance depends on context length and supported
shapes. Enable and measure the switches needed for a workload.

## Validation and build tools

The `validate_*.py` scripts check numerical outputs and changed-input NPU graph
replay against the baseline equations. Run hardware validators in an isolated
container exposing one available NPU. `validate_hc.py` also supports a
`--mode compile` check without hardware; compilation alone is not numerical
validation. `benchmark_cache.py` measures cache operations separately from
end-to-end service throughput.

The unit tests in `tests/ut/models/deepseek_v41/` cover optional dependency
loading, CPU fallbacks, cache metadata, quantization, normalization and rotary
integration. Hardware cases require Ascend NPU and the pinned DSL package.

`build_runtime.py` builds an offline image from the pinned A5 baseline image and
DSL wheel. It verifies baseline and candidate source hashes and refuses existing
image tags or build directories:

```bash
python3 examples/deepseek_v41_a5/cannbot/build_runtime.py \
  --wheel /path/to/cannbotdsl-0.6.0+g91e7c8f-cp312-cp312-manylinux_2_28_aarch64.whl \
  --context /path/to/new-build-directory \
  --tag jojo/vllm-ascend:cannbot-only
```

The kernels are extracted unchanged from the validated experimental branch.
Its throughput results also included other runtime changes, so those results
must not be attributed to this fusion-only branch. Benchmark this branch with
an unchanged baseline serving configuration for a controlled comparison.
