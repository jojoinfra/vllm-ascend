# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import MagicMock, patch

import torch
from safetensors.torch import save_file

from vllm_ascend.patch.worker.deepseek_v41_embedding import ShardedEngramEmbedding
from vllm_ascend.patch.worker.deepseek_v41_loader import DeepseekV41Checkpoint


def make_checkpoint(root):
    tensors = {
        "layers.1.engram.embed.weight": torch.arange(8 * 32, dtype=torch.int32).reshape(8, 32).to(torch.int8),
        "layers.1.engram.embed.scale": torch.full((8, 1), 127, dtype=torch.uint8),
        "layers.1.ffn.experts.0.w1.weight": torch.ones(2, 3),
        "layers.1.ffn.experts.1.w1.weight": torch.full((2, 3), 2.0),
        "layers.1.attn.wq_a.weight": torch.zeros(2, 3),
        "mtp.0.ffn.experts.0.w1.weight": torch.ones(2, 3),
        "vision.layers.0.weight": torch.ones(2, 3),
    }
    save_file(dict(tensors, unindexed_copy=torch.zeros(7)), root / "weights.safetensors")
    (root / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "weights.safetensors" for name in tensors}})
    )
    return tensors


def test_expert_and_architecture_filtering_happens_before_materializing(tmp_path):
    make_checkpoint(tmp_path)
    checkpoint = DeepseekV41Checkpoint(tmp_path)
    handle = MagicMock()
    handle.__enter__.return_value = handle
    with patch("vllm_ascend.patch.worker.deepseek_v41_loader.safe_open", return_value=handle):
        rows = list(checkpoint.iter_weights(local_expert_ids={1}, include_prefixes=("layers.",)))
    assert [name for name, _ in rows] == ["layers.1.ffn.experts.1.w1.weight", "layers.1.attn.wq_a.weight"]
    assert [call.args[0] for call in handle.get_tensor.call_args_list] == [name for name, _ in rows]
    handle.get_slice.assert_not_called()


def test_engram_only_reads_rank_slice_and_preserves_other_weight_names(tmp_path):
    tensors = make_checkpoint(tmp_path)
    checkpoint = DeepseekV41Checkpoint(tmp_path)
    table = ShardedEngramEmbedding(8, 32, rank=1, world_size=3, device="cpu", quant_type="INT8_SYM")
    checkpoint.load_engram(1, table)
    torch.testing.assert_close(table.weight, tensors["layers.1.engram.embed.weight"][3:6])
    loaded = dict(checkpoint.iter_weights())
    assert "unindexed_copy" not in loaded
    assert "layers.1.engram.embed.weight" not in loaded
    assert "vision.layers.0.weight" in loaded and "mtp.0.ffn.experts.0.w1.weight" in loaded
    torch.testing.assert_close(loaded["layers.1.ffn.experts.1.w1.weight"], torch.full((2, 3), 2.0))


def test_engram_weight_and_scale_can_be_in_separate_shards(tmp_path):
    tensors = make_checkpoint(tmp_path)
    scale = "layers.1.engram.embed.scale"
    save_file({scale: tensors[scale]}, tmp_path / "scale.safetensors")
    checkpoint = DeepseekV41Checkpoint(tmp_path)
    checkpoint.weight_map[scale] = "scale.safetensors"
    table = ShardedEngramEmbedding(8, 32, rank=2, world_size=3, device="cpu", quant_type="INT8_SYM")
    checkpoint.load_engram(1, table)
    torch.testing.assert_close(table.weight[:2], tensors["layers.1.engram.embed.weight"][6:8])
    assert not table.weight[2].count_nonzero()
