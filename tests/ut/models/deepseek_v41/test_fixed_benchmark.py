# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark metadata must describe actual final CLI options and topology."""

import importlib.util
import sys
from pathlib import Path

import pytest


def benchmark_module():
    source = Path(__file__).resolve().parents[4] / "examples/deepseek_v41_a5/benchmark-fixed.py"
    spec = importlib.util.spec_from_file_location("fixed_benchmark", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_cli_options_override_earlier_values():
    module = benchmark_module()
    settings = module.final_cli_settings(
        [
            "serve",
            "--tensor-parallel-size",
            "4",
            "--additional-config",
            '{"old":true}',
            "--tensor-parallel-size=1",
            "--additional-config",
            '{"new":true}',
        ],
        ("--tensor-parallel-size", "--additional-config"),
    )
    assert settings == {"--tensor-parallel-size": "1", "--additional-config": '{"new":true}'}


@pytest.mark.parametrize("tail", [["--dtype"], ["--dtype", "--other"]])
def test_missing_cli_value_rejected(tail):
    with pytest.raises(ValueError, match="Missing value"):
        benchmark_module().final_cli_settings(tail, ("--dtype",))


@pytest.mark.parametrize("tp,dp", [(4, 1), (1, 4)])
def test_matching_four_card_layout(tp, dp):
    hardware = dict(physical_devices=[1, 2, 3, 4], tensor_parallel_size=tp, expert_parallel_size=4)
    settings = {"--tensor-parallel-size": str(tp), "--enable-expert-parallel": True}
    if dp != 1:
        hardware["data_parallel_size"] = dp
        settings["--data-parallel-size"] = str(dp)
    benchmark_module().validate_parallel_layout(settings, hardware)


@pytest.mark.parametrize(
    "override",
    [
        {"--tensor-parallel-size": "1"},
        {"--data-parallel-size": "4"},
        {"--pipeline-parallel-size": "2"},
        {"--enable-expert-parallel": False},
    ],
)
def test_layout_mismatch_rejected(override):
    settings = {"--tensor-parallel-size": "4", "--enable-expert-parallel": True} | override
    hardware = dict(physical_devices=[1, 2, 3, 4], tensor_parallel_size=4, expert_parallel_size=4)
    with pytest.raises(RuntimeError, match="Scenario requires"):
        benchmark_module().validate_parallel_layout(settings, hardware)


def test_declared_autoregressive_variant_rejects_speculation_and_wrong_buckets():
    module = benchmark_module()
    scenario = dict(runtime_variant=dict(speculation="disabled", capture_sizes=[4, 8, 16]))
    settings = {"--speculative-config": "null", "--compilation-config": '{"cudagraph_capture_sizes":[4,8,16]}'}
    module.validate_runtime_variant(settings, scenario)
    with pytest.raises(RuntimeError, match="speculative decoding"):
        module.validate_runtime_variant(settings | {"--speculative-config": '{"method":"dspark"}'}, scenario)
    with pytest.raises(RuntimeError, match="Graph buckets"):
        module.validate_runtime_variant(
            settings | {"--compilation-config": '{"cudagraph_capture_sizes":[24,48]}'}, scenario
        )


def test_declared_memory_fraction_requires_matching_runtime_value():
    module = benchmark_module()
    scenario = dict(runtime_variant=dict(gpu_memory_utilization=0.94))
    module.validate_runtime_variant({"--gpu-memory-utilization": "0.94"}, scenario)
    for settings in ({}, {"--gpu-memory-utilization": "0.93"}, {"--gpu-memory-utilization": "nan"}):
        with pytest.raises(RuntimeError, match="GPU memory utilization"):
            module.validate_runtime_variant(settings, scenario)


@pytest.mark.parametrize("fraction", ["0", "-0.1", "1.01", "nan", "inf"])
def test_launcher_rejects_invalid_memory_fraction_before_docker(monkeypatch, fraction):
    source = Path(__file__).resolve().parents[4] / "examples/deepseek_v41_a5/cannbot/start_runtime.py"
    spec = importlib.util.spec_from_file_location("cannbot_start_runtime", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(source),
            "--image",
            "unused",
            "--name",
            "unused",
            "--deployment",
            "/unused",
            "--gpu-memory-utilization",
            fraction,
        ],
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: pytest.fail("Docker must not be invoked"))
    with pytest.raises(ValueError, match="GPU memory utilization"):
        module.main()
