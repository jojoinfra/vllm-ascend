# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram residual update, composed with a separately sharded hash table.

Equations follow the official inference/model.py Engram.forward. The W8A8
checkpoint's value projection is already in the global QuaRot basis; keys and
gate weights remain in the original basis. Never rotate its value twice.
"""

import torch
import torch.nn.functional as F
from torch import nn

from vllm_ascend.patch.worker.deepseek_v41_linear import apply_linear


class DeepseekV41Engram(nn.Module):
    GATE_CLAMP = 1e-6

    def __init__(self, config, embedding, *, device, rotation=None, rotation_config=None):
        super().__init__()
        self.embed = embedding
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        hash_columns = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        self.wkv = nn.Linear(
            hash_columns * config.engram_head_dim,
            (self.hc_mult + 1) * self.hidden_size,
            bias=False,
            dtype=torch.bfloat16,
            device=device,
        )
        self.q_weight = nn.Parameter(torch.ones(self.hc_mult, self.hidden_size, dtype=torch.bfloat16, device=device))
        self.k_weight = nn.Parameter(torch.ones_like(self.q_weight))
        rotation_config = rotation_config or {}
        if rotation_config.get("runtime_delta_rotation", False):
            raise ValueError("Engram runtime delta rotation is not supported by this checkpoint adapter")
        rotated = rotation_config.get("value_projection_rotated", False)
        if rotated:
            if (
                rotation_config.get("value_basis") != "quarot_global"
                or rotation_config.get("key_and_gate_basis") != "original"
            ):
                raise ValueError("Unrecognized Engram projection/gate basis")
            if rotation is None or tuple(rotation.shape) != (self.hidden_size, self.hidden_size):
                raise ValueError("Rotated Engram values require the checkpoint global rotation")
        elif rotation is not None:
            raise ValueError("A rotation cannot be applied to an unrotated Engram checkpoint")
        # The model may share one immutable rotation tensor across Engram layers.
        self.register_buffer("rotation", rotation, persistent=False)

    def forward(self, hidden_states, hash_ids, *, token_mask=None, group=None):
        embeddings = self.embed(hash_ids, group).flatten(-2)
        projected = apply_linear(self.wkv, embeddings)
        key, value = projected.split((self.hc_mult * self.hidden_size, self.hidden_size), dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.hidden_size))
        residual = hidden_states.float()
        # x_rotated = x_original @ Q, hence x_original = x_rotated @ Q.T.
        gate_input = F.linear(residual, self.rotation) if self.rotation is not None else residual
        gate_weight = self.q_weight.float() * self.k_weight.float()
        inverse_norm = torch.rsqrt(gate_input.square().mean(-1) + self.norm_eps)
        inverse_norm = inverse_norm * torch.rsqrt(key.square().mean(-1) + self.norm_eps)
        dot = (gate_input * gate_weight * key).sum(-1) * inverse_norm * self.hidden_size**-0.5
        magnitude = dot.abs().clamp_min(self.GATE_CLAMP).sqrt()
        # copysign falls back to CPU on A3. Inspect the FP32 sign bit instead,
        # preserving negative zero without a host synchronization.
        signed_root = torch.where(dot.view(torch.int32) < 0, -magnitude, magnitude)
        gate = torch.sigmoid(signed_root)
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (residual + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden_states.dtype)
