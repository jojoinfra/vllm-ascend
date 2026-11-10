# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_ascend.patch.worker.deepseek_v41_cache_plan import build_attention_cache_plan


def flash_topology(**overrides):
    fields = dict(
        num_hidden_layers=40,
        compress_ratios=[0] * 2 + [2] * 18 + [1] * 20 + [0] * 3,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
    )
    return SimpleNamespace(**(fields | overrides))


def test_official_flash_ownership():
    plan = build_attention_cache_plan(flash_topology())
    assert len(plan) == 40  # Three DSpark layers do not own backbone caches.
    assert [p.kv_layer_id for p in plan] == [None] * 2 + [2] * 6 + [8] * 6 + [14] * 6 + [20] * 20
    assert [p.index_layer_id for p in plan] == (
        [None] * 2 + [2] * 6 + [8] * 6 + [14] * 6 + [20] * 4 + [24] * 4 + [28] * 4 + [32] * 4 + [36] * 4
    )
    assert [p.candidate_layer_id for p in plan] == [None] * 24 + [20] * 16
    # Layer 24 has its own query/index selection, but reads layer 20's K.
    assert plan[24].kv_layer_id == 20 and plan[24].index_layer_id == 24


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"compress_ratios": [0]}, "cover every"),
        ({"kv_source_layer_ids": [8, 2]}, "unique and ordered"),
        ({"index_source_layer_ids": [2, 2]}, "unique and ordered"),
        ({"kv_source_layer_ids": [2, 8, 14, 40]}, "outside"),
        ({"kv_source_layer_ids": [0, 2, 8, 14, 20]}, "positive compression"),
        ({"index_source_layer_ids": [2, 8, 20]}, "produce index keys"),
        ({"kv_source_layer_ids": [8, 14, 20]}, "before a source"),
        ({"kv_source_layer_ids": [2, 8, 14]}, "changes compression"),
        ({"candidate_source_layer_id": 21}, "Candidate source"),
        ({"candidate_block_size": 0}, "positive block"),
        ({"compress_ratios": [0, 0] + [4] * 38}, "Unsupported"),
    ],
)
def test_invalid_topology_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        build_attention_cache_plan(flash_topology(**overrides))


def test_candidates_can_be_disabled():
    plan = build_attention_cache_plan(flash_topology(candidate_source_layer_id=-1))
    assert all(layer.candidate_layer_id is None for layer in plan)
