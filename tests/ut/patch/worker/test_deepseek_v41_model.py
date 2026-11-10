# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from vllm.model_executor.models.utils import sequence_parallel_chunk_impl

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.models.deepseek_v4.model import DeepseekV4MoE
from vllm_ascend.patch.worker.deepseek_v41_model import (
    DeepseekV41ForCausalLM,
    DeepseekV41Model,
    DeepseekV41MoE,
    capture_moe_communication,
    use_moe_communication,
)
from vllm_ascend.patch.worker.patch_bind_kv_cache import bind_kv_cache
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig, get_packed_modules_mapping


@pytest.mark.parametrize("tp_size", [4, 8])
@pytest.mark.parametrize("num_tokens", [1, 7, 16])
@pytest.mark.parametrize("with_images", [False, True])
def test_moe_sequence_parallel_preserves_token_order_and_image_alignment(tp_size, num_tokens, with_images):
    model = DeepseekV41MoE.__new__(DeepseekV41MoE)
    nn.Module.__init__(model)
    model.is_sequence_parallel = True
    hidden = torch.arange(num_tokens * 3, dtype=torch.float32).reshape(num_tokens, 3) + 1
    mask = torch.arange(num_tokens) % 3 == 0 if with_images else None
    expected = hidden.square() + (mask[:, None].float() if mask is not None else 0)
    shards = []
    module = "vllm_ascend.patch.worker.deepseek_v41_model."
    utils = "vllm.model_executor.models.utils."

    def experts(self, local_hidden, input_ids=None):
        assert local_hidden.shape[0] == (num_tokens + tp_size - 1) // tp_size
        assert (input_ids is not None) == with_images
        if input_ids is not None:
            assert input_ids.shape == local_hidden.shape[:1]
        output = local_hidden.square() + (input_ids[:, None].float() if input_ids is not None else 0)
        shards.append(output)
        return output

    # Exercise the real upstream shard implementation on every TP rank,
    # including ranks containing only padding and an uneven final shard.
    padded_output = torch.nn.functional.pad(expected, (0, 0, 0, -num_tokens % tp_size))
    for rank in range(tp_size):
        with (
            patch(utils + "get_tensor_model_parallel_world_size", return_value=tp_size),
            patch(utils + "get_tensor_model_parallel_rank", return_value=rank),
            patch(module + "sequence_parallel_chunk", side_effect=sequence_parallel_chunk_impl),
            patch.object(DeepseekV4MoE, "forward", autospec=True, side_effect=experts),
            patch(module + "sp_all_gather", return_value=padded_output) as gather,
        ):
            torch.testing.assert_close(model(hidden, image_mask=mask), expected, rtol=0, atol=0)
            assert gather.call_args.args[0] is shards[-1]
    torch.testing.assert_close(torch.cat(shards), padded_output, rtol=0, atol=0)


def test_moe_without_sequence_parallel_passes_full_inputs_without_gather():
    model = DeepseekV41MoE.__new__(DeepseekV41MoE)
    nn.Module.__init__(model)
    model.is_sequence_parallel = False
    hidden = torch.ones(3, 2)
    mask = torch.tensor([False, True, False])
    with (
        patch.object(DeepseekV4MoE, "forward", autospec=True, return_value=hidden) as forward,
        patch("vllm_ascend.patch.worker.deepseek_v41_model.sp_all_gather") as gather,
    ):
        assert model(hidden, image_mask=mask) is hidden
        forward.assert_called_once_with(model, hidden, input_ids=mask)
        gather.assert_not_called()


def test_target_and_draft_keep_distinct_moe_communication_and_restore_context():
    module = "vllm_ascend.patch.worker.deepseek_v41_model."
    target = SimpleNamespace(num_experts=384, top_k=6)
    draft = SimpleNamespace(num_experts=128, top_k=3)
    with patch(module + "get_moe_comm_method", return_value=target):
        target_methods = capture_moe_communication()
    with patch(module + "get_moe_comm_method", return_value=draft):
        draft_methods = capture_moe_communication()
    context = SimpleNamespace(moe_comm_type=MoECommType.ALLTOALL, moe_comm_method=draft)
    with patch(module + "_EXTRA_CTX", context):
        with use_moe_communication(target_methods):
            assert context.moe_comm_method is target
            with pytest.raises(RuntimeError, match="draft failure"), use_moe_communication(draft_methods):
                assert context.moe_comm_method is draft
                raise RuntimeError("draft failure")
            assert context.moe_comm_method is target
        assert context.moe_comm_method is draft


def test_backbone_shared_history_uses_real_cache_binding_contract():
    config = SimpleNamespace(
        num_hidden_layers=0,
        vocab_size=8,
        hidden_size=4,
        rms_norm_eps=1e-6,
        engram_compressed_vocab_size=8,
        engram_pad_token_id=0,
        engram_max_ngram_size=4,
    )
    context = {}
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config, hf_config=config, max_model_len=32, tokenizer="unused"),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        compilation_config=SimpleNamespace(static_forward_context=context),
        cache_config=SimpleNamespace(block_size=4),
    )
    module = "vllm_ascend.patch.worker.deepseek_v41_model."
    with ExitStack() as stack:
        for name, result in {
            "VocabParallelEmbedding": nn.Embedding(8, 4),
            "DeepseekV41Rotary": nn.Identity(),
            "get_ep_group": SimpleNamespace(),
            "AutoTokenizer.from_pretrained": object(),
            "build_engram_token_map": torch.arange(8),
            "EngramHasher": nn.Identity(),
            "build_attention_cache_plan": [],
        }.items():
            stack.enter_context(patch(module + name, return_value=result))
        model = DeepseekV41Model(vllm_config=vllm_config, prefix="model")
    cache = torch.zeros(3, 4, 1, 1, dtype=torch.int64)
    runner_caches = []
    bind_kv_cache({name: cache for name in context}, context, runner_caches)
    assert runner_caches == [cache]
    assert model.history.cache.kv_cache is cache


def test_quantizer_selector_loads_layer_description_sidecar(tmp_path):
    description = {"layers.0.attn.wq_a.weight": "FLOAT", "layers.0.attn.wq_b.weight": "W8A8_DYNAMIC"}
    sidecar = tmp_path / "quant_model_description.json"
    sidecar.write_text(json.dumps(description))
    quant = AscendModelSlimConfig({"quant_method": "ascend"})
    with patch("vllm_ascend.quantization.utils.get_model_file", return_value=str(sidecar)) as find:
        quant.maybe_update_config(str(tmp_path))
        assert quant.quant_description == description
        quant.maybe_update_config(str(tmp_path))
        find.assert_called_once()


def test_checkpoint_quantization_names_preserve_attention_and_pack_shared_experts():
    quant = AscendModelSlimConfig(
        {
            "layers.2.attn.wq_b.weight": "W8A8_DYNAMIC",
            "layers.2.attn.wq_a.weight": "FLOAT",
            "layers.2.ffn.shared_experts.w1.weight": "W8A8_DYNAMIC",
            "layers.2.ffn.shared_experts.w3.weight": "W8A8_DYNAMIC",
            "layers.2.ffn.experts.0.w1.weight": "W8A8_DYNAMIC",
            "layers.2.ffn.experts.0.w2.weight": "W8A8_DYNAMIC",
            "layers.2.ffn.experts.0.w3.weight": "W8A8_DYNAMIC",
            "head.weight": "FLOAT",
        }
    )
    quant.apply_vllm_mapper(DeepseekV41ForCausalLM.hf_to_vllm_mapper)
    mapping = get_packed_modules_mapping("deepseek_v41")
    assert not quant.is_layer_skipped_ascend("model.layers.2.attn.wq_b", mapping)
    assert quant.is_layer_skipped_ascend("model.layers.2.attn.wq_a", mapping)
    assert not quant.is_layer_skipped_ascend("model.layers.2.ffn.shared_experts.gate_up_proj", mapping)
    assert not quant.is_layer_skipped_ascend("model.layers.2.ffn.experts", mapping)
    assert quant.is_layer_skipped_ascend("lm_head", mapping)


def empty_language_model():
    model = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
    nn.Module.__init__(model)
    model.lm_head = nn.Linear(3, 5, bias=False)
    model.model = nn.Module()
    model.model.norm = nn.LayerNorm(3, bias=False)
    return model


def test_auxiliary_interface_translates_eagle_numbering_to_official_input_layers():
    model = empty_language_model()
    model.config = SimpleNamespace(dspark_target_layer_ids=[37, 38, 39])
    layers = model.get_eagle3_default_aux_hidden_state_layers()
    assert layers == (38, 39, 40)
    model.set_aux_hidden_state_layers(layers)
    assert model.model.aux_hidden_state_layers == (37, 38, 39)


def test_weight_loader_rejects_missing_or_unexpected_parameters():
    model = empty_language_model()
    with patch.object(model, "get_expert_mapping", return_value=[]):
        with pytest.raises(ValueError, match="model.norm.weight"):
            model.load_weights([("head.weight", torch.ones(5, 3))])
        with pytest.raises(KeyError, match="extra"):
            model.load_weights([("extra.weight", torch.ones(3))])
        loaded = model.load_weights([("head.weight", torch.ones(5, 3)), ("norm.weight", torch.full((3,), 2.0))])
    assert loaded == {"lm_head.weight", "model.norm.weight"}
    torch.testing.assert_close(model.model.norm.weight, torch.full((3,), 2.0))


class EngramRecorder(nn.Module):
    layer_hash_index = 0

    def forward(self, hidden, hashes, *, token_mask, group):
        self.last_input = (hidden.shape, hashes.clone(), token_mask.clone())
        return hidden + 3 * token_mask[:, None, None]


class DecoderStub(nn.Module):
    def __init__(self, increment, engram=None):
        super().__init__()
        self.increment = increment
        self.engram = engram

    def forward(self, positions, hidden, pre_mix, **kwargs):
        return hidden + self.increment, pre_mix


class HasherStub(nn.Module):
    def forward(self, history):
        return history[:, None, :]


class HistoryStub(nn.Module):
    def forward(self, tokens, positions, image_mask):
        return tokens[:, None]


def test_backbone_collects_post_engram_pre_attention_auxiliary_states_with_dp_padding():
    model = DeepseekV41Model.__new__(DeepseekV41Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hc_mult=2)
    model.embed = nn.Embedding.from_pretrained(torch.arange(8).float()[:, None].expand(-1, 3))
    model.norm = nn.Identity()
    model.history = HistoryStub()
    model.hasher = HasherStub()
    model.ep_group = SimpleNamespace(world_size=4)
    engram = EngramRecorder()
    model.layers = nn.ModuleList([DecoderStub(1, engram), DecoderStub(2)])
    model.aux_hidden_state_layers = (0, 1)
    tokens = torch.tensor([1, 4, 6])
    with patch(
        "vllm_ascend.patch.worker.deepseek_v41_model.get_forward_context",
        return_value=SimpleNamespace(max_tokens_across_dp=5),
    ):
        output, auxiliary = model(tokens, torch.arange(3))
    torch.testing.assert_close(auxiliary[0], (tokens + 3).float()[:, None].expand(-1, 3))
    torch.testing.assert_close(auxiliary[1], (tokens + 4).float()[:, None].expand(-1, 3))
    torch.testing.assert_close(output, (tokens + 6).float()[:, None].expand(-1, 3))
    shape, hashes, mask = engram.last_input
    assert shape == (5, 2, 3)
    assert hashes[:, 0].tolist() == [1, 4, 6, -1, -1]
    assert mask.tolist() == [True, True, True, False, False]
