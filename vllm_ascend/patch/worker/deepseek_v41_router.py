# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-token image bias with the existing Ascend expert-selection kernels."""

from copy import copy

import torch
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.model_executor.models.utils import sequence_parallel_chunk

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType


class DeepseekV41ImageRouter(BaseRouter):
    def __init__(self, text_router, image_bias):
        super().__init__(text_router.top_k, text_router.global_num_experts, text_router.eplb_state)
        if text_router.tid2eid is not None:
            raise ValueError("V4.1 image routing cannot use token-hash expert selection")
        self.text_router = text_router
        self.image_router = copy(text_router)
        self.image_router.e_score_correction_bias = image_bias

    @property
    def routing_method_type(self):
        return self.text_router.routing_method_type

    def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
        # input_ids carries the image-span mask through the MoE runner's
        # existing token-side channel. Align it exactly like hash-router IDs.
        text_weights, text_ids = self.text_router._compute_routing(hidden_states, router_logits, indices_type)
        if input_ids is None:
            return text_weights, text_ids
        mask = input_ids.to(torch.int64)
        communication = _EXTRA_CTX.moe_comm_method
        if _EXTRA_CTX.moe_comm_type == MoECommType.ALLGATHER:
            mask = communication.prepare_finalize.all_gather_input_ids(mask)
        else:
            mask = communication.pad_and_split_input_ids(mask)
            if mask.numel() != router_logits.shape[0]:
                mask = sequence_parallel_chunk(mask.reshape(-1, 1)).reshape(-1)
        image_weights, image_ids = self.image_router._compute_routing(hidden_states, router_logits, indices_type)
        mask = (mask > 0).reshape(-1, 1)
        # Bias affects selection only. Both kernel calls return weights from
        # the original sqrtsoftplus scores, preserving the text kernel path.
        return torch.where(mask, image_weights, text_weights), torch.where(mask, image_ids, text_ids)
