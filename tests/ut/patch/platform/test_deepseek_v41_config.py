# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig
from vllm.config.model import _get_head_dtype
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

from vllm_ascend.patch.platform.patch_deepseek_v41_config import (
    DeepseekV41Config,
    DeepseekV41TextConfig,
    DeepseekV41VisionConfig,
)
from vllm_ascend.patch.platform.patch_speculative_config import _normalize_legacy_qwen3_dspark_config


def test_nested_config_roundtrip(tmp_path):
    # W8A8 checkpoint metadata must survive independently of the text config.
    config = DeepseekV41Config(
        architectures=["DeepseekV41ForCausalLM"],
        quantization_config={"quant_method": "ascend"},
        engram_rotation_config={"value_projection_rotated": True, "runtime_delta_rotation": False},
    )
    config.save_pretrained(tmp_path)
    loaded = AutoConfig.from_pretrained(tmp_path, trust_remote_code=False)
    assert isinstance(loaded, DeepseekV41Config)
    assert isinstance(loaded.get_text_config(), DeepseekV41TextConfig)
    assert isinstance(loaded.vision_config, DeepseekV41VisionConfig)
    assert loaded.text_config.num_hidden_layers == 40
    assert len(loaded.text_config.compress_ratios) == 43
    assert loaded.text_config.rope_parameters["rope_type"] == "yarn"
    assert loaded.quantization_config == {"quant_method": "ascend"}
    assert loaded.engram_rotation_config == config.engram_rotation_config
    assert loaded.image_token_id == 129264
    assert loaded.text_config.image_token_index == loaded.image_token_id
    assert loaded.architectures == ["DeepseekV41ForCausalLM"]
    assert loaded.head_dtype == loaded.text_config.head_dtype == "float32"


def test_overrides_preserve_input_and_are_independent():
    text = {"hidden_size": 64, "rope_theta": 20000, "compress_ratios": [0, 2, 1]}
    original = copy.deepcopy(text)
    config = DeepseekV41Config(text_config=text, vision_config={"hidden_size": 32})
    assert config.text_config.hidden_size == 64
    assert config.vision_config.hidden_size == 32
    assert config.text_config.rope_theta == 20000
    assert text == original
    first, second = DeepseekV41Config(), DeepseekV41Config()
    first.text_config.compress_ratios[0] = 2
    first.text_config.rope_parameters["factor"] = 2
    assert second.text_config.compress_ratios[0] == 0
    assert second.text_config.rope_parameters["factor"] == 16


def test_subconfig_instances_and_standalone_registration():
    text = AutoConfig.for_model("deepseek_v41_text")
    vision = AutoConfig.for_model("deepseek_v41_vision")
    config = DeepseekV41Config(text_config=text, vision_config=vision)
    assert config.get_text_config() is text
    assert config.vision_config is vision


def test_explicit_rope_parameters_override_default_scaling():
    rope = {"rope_type": "yarn", "factor": 8, "original_max_position_embeddings": 32768}
    config = DeepseekV41TextConfig(rope_parameters=rope)
    assert config.rope_parameters == rope


def test_draft_config_uses_own_experts_without_mutating_composite_target():
    target = DeepseekV41Config(quantization_config={"quant_method": "ascend"})
    draft = _normalize_legacy_qwen3_dspark_config(target)
    assert draft.n_routed_experts == 128
    assert draft.num_experts_per_tok == 3
    assert draft.n_mtp_layers == 3
    assert draft.n_predict == 5
    assert draft.ptd_token_id == 128799
    assert draft.quantization_config == {"quant_method": "ascend"}
    assert draft.architectures == ["DeepseekV41DSparkForCausalLM"]
    assert target.text_config.n_routed_experts == 384
    assert target.text_config.num_experts_per_tok == 6


def test_image_token_override_propagates_to_language_and_draft_configs():
    target = DeepseekV41Config(image_token_id=1234)
    draft = _normalize_legacy_qwen3_dspark_config(target)
    assert target.text_config.image_token_index == 1234
    assert draft.image_token_index == 1234


@pytest.mark.parametrize("draft", [False, True])
def test_official_fp32_head_survives_native_logits_dtype_selection(draft):
    target = DeepseekV41Config()
    config = _normalize_legacy_qwen3_dspark_config(target) if draft else target
    dtype = _get_head_dtype(config, torch.bfloat16, "generate")
    processor = LogitsProcessor.__new__(LogitsProcessor)
    nn.Module.__init__(processor)
    processor.head_dtype = dtype
    # The target checkpoint is BF16 and the draft head is FP32. Both must
    # produce FP32 logits, including values that BF16 would round away.
    torch.manual_seed(43)
    head = nn.Module()
    head.weight = nn.Parameter(torch.randn(11, 8).to(torch.float32 if draft else torch.bfloat16))
    head.quant_method = UnquantizedEmbeddingMethod()
    hidden = torch.randn(3, 8).to(torch.float32 if draft else torch.bfloat16)
    actual = LogitsProcessor._apply_head(processor, head, hidden, None)
    expected = F.linear(hidden.float(), head.weight.float())
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(expected, expected.bfloat16().float())


def test_explicit_head_dtype_override_remains_native_configuration():
    config = DeepseekV41Config(head_dtype="model")
    assert _get_head_dtype(config, torch.bfloat16, "generate") == torch.bfloat16
