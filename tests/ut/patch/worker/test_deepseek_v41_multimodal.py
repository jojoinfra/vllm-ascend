# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.patch.worker.deepseek_v41_model import DeepseekV41ForCausalLM
from vllm_ascend.patch.worker.deepseek_v41_router import DeepseekV41ImageRouter


class ScoreRouter:
    top_k = 2
    global_num_experts = 3
    eplb_state = tid2eid = None
    routing_method_type = "test"
    e_score_correction_bias = torch.tensor([2.0, 1.0, -2.0])

    def _compute_routing(self, hidden, logits, indices_type):
        scores = F.softplus(logits.float()).sqrt()
        ids = (scores + self.e_score_correction_bias).topk(2, dim=-1).indices
        weights = scores.gather(1, ids)
        return weights / weights.sum(-1, keepdim=True), ids.to(torch.int32)


@pytest.mark.parametrize("communication_kind", [MoECommType.ALLGATHER, MoECommType.ALLTOALL])
def test_image_router_aligns_masks_and_selects_bias_without_changing_weights(communication_kind):
    router = DeepseekV41ImageRouter(ScoreRouter(), torch.tensor([-2.0, 1.0, 2.0]))
    communication = Mock()
    # Simulate DP padding/gather or native MoE token partitioning. Negative
    # padded IDs must be interpreted as text rather than a truthy image flag.
    aligned = torch.tensor([0, 1, -1])
    communication.prepare_finalize.all_gather_input_ids.return_value = aligned
    communication.pad_and_split_input_ids.return_value = aligned
    context = SimpleNamespace(moe_comm_type=communication_kind, moe_comm_method=communication)
    logits = torch.tensor([[0.0, 1.0, 2.0]]).repeat(3, 1)
    with patch("vllm_ascend.patch.worker.deepseek_v41_router._EXTRA_CTX", context):
        weights, ids = router._select_experts(logits, logits, torch.int32, input_ids=torch.tensor([0, 1]))
    assert ids.tolist() == [[0, 1], [2, 1], [0, 1]]
    raw = F.softplus(logits).sqrt().gather(1, ids.long())
    torch.testing.assert_close(weights, raw / raw.sum(-1, keepdim=True))
    if communication_kind == MoECommType.ALLGATHER:
        communication.prepare_finalize.all_gather_input_ids.assert_called_once()
    else:
        communication.pad_and_split_input_ids.assert_called_once()


def test_image_router_applies_sequence_partition_after_communication():
    router = DeepseekV41ImageRouter(ScoreRouter(), torch.tensor([-2.0, 1.0, 2.0]))
    communication = Mock()
    communication.pad_and_split_input_ids.return_value = torch.tensor([0, 1, 0, 1])
    context = SimpleNamespace(moe_comm_type=MoECommType.ALLTOALL, moe_comm_method=communication)
    logits = torch.zeros(2, 3)
    with (
        patch("vllm_ascend.patch.worker.deepseek_v41_router._EXTRA_CTX", context),
        patch("vllm_ascend.patch.worker.deepseek_v41_router.sequence_parallel_chunk", side_effect=lambda x: x[2:]),
    ):
        _, ids = router._select_experts(logits, logits, torch.int32, input_ids=torch.tensor([0, 1, 0, 1]))
    assert ids.tolist() == [[0, 1], [2, 1]]


def test_image_router_text_only_delegates_without_mask_collectives():
    text = ScoreRouter()
    router = DeepseekV41ImageRouter(text, torch.ones(3))
    logits = torch.randn(4, 3)
    actual = router._select_experts(logits, logits, torch.int32)
    expected = text._compute_routing(logits, logits, torch.int32)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def empty_multimodal_model():
    model = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.embed_input_ids = lambda ids: ids.float()[:, None].expand(-1, 2).clone()
    return model


def test_multimodal_merge_preserves_raw_tokens_and_text_embeddings():
    model = empty_multimodal_model()
    tokens = torch.tensor([7, 129264, 129264, 9])
    embeddings = [torch.tensor([[2.0, 3.0], [4.0, 5.0]])]
    hidden = model.embed_input_ids(tokens, embeddings, is_multimodal=tokens == 129264)
    torch.testing.assert_close(hidden, torch.tensor([[7.0, 7.0], [2.0, 3.0], [4.0, 5.0], [9.0, 9.0]]))
    assert tokens.tolist() == [7, 129264, 129264, 9]
    assert model.requires_raw_input_tokens
    assert model.get_language_model() is model
    with pytest.raises(ValueError, match="explicit multimodal token mask"):
        model.embed_input_ids(tokens, embeddings)


def test_vision_and_aligner_weights_are_not_remapped_as_moe_projections():
    mapper = DeepseekV41ForCausalLM.hf_to_vllm_mapper
    names = [
        "vision.blocks.0.mlp.w1.weight",
        "aligner.w2.bias",
        "image_start",
        "layers.2.ffn.experts.12.w1.weight",
        "layers.2.ffn.shared_experts.w2.weight",
    ]
    assert mapper.apply_list(names) == [
        *names[:3],
        "model.layers.2.ffn.experts.12.gate_proj.weight",
        "model.layers.2.ffn.shared_experts.down_proj.weight",
    ]


@pytest.mark.parametrize("image_enabled", [False, True])
def test_model_forward_passes_image_spans_to_history_and_moe(image_enabled):
    model = empty_multimodal_model()
    model.vision = nn.Identity() if image_enabled else None
    model.model.forward = Mock()
    model.image_token_id = 129264
    model.moe_comm_methods = {}
    tokens = torch.tensor([3, 129264, 129264, 4])
    embeddings = torch.randn(4, 2)
    with patch("vllm_ascend.patch.worker.deepseek_v41_model.use_moe_communication"):
        model(tokens, torch.arange(4), inputs_embeds=embeddings)
    mask = model.model.forward.call_args.kwargs["image_mask"]
    if image_enabled:
        torch.testing.assert_close(mask, torch.tensor([False, True, True, False]))
    else:
        assert mask is None
