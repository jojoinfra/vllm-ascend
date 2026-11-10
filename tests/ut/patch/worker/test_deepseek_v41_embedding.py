# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.patch.worker.deepseek_v41_embedding import ShardedEngramEmbedding


def make_table(rank=0, world_size=1):
    return ShardedEngramEmbedding(7, 64, rank=rank, world_size=world_size, device="cpu", quant_type="INT8_SYM")


def test_exact_signed_int8_and_e8m0_values():
    weights = torch.arange(-64, 64, dtype=torch.int8).reshape(2, 64).repeat(4, 1)[:7]
    scales = torch.tensor([[126, 128]], dtype=torch.uint8).repeat(7, 1)
    table = make_table()
    table.load_rows(weights, scales, chunk_rows=2)
    result = table(torch.tensor([0, 1, 6, -1, 7]))
    expected = weights[[0, 1, 6]].float() * torch.tensor([0.5] * 32 + [2.0] * 32)
    torch.testing.assert_close(result[:3], expected.bfloat16())
    assert torch.count_nonzero(result[3:]) == 0


@pytest.mark.parametrize("world_size", [2, 4, 8, 16])
def test_uneven_and_empty_partitions_sum_to_whole_table(world_size):
    weights = torch.arange(7 * 64).remainder(127).to(torch.int8).reshape(7, 64)
    scales = torch.full((7, 2), 127, dtype=torch.uint8)
    indices = torch.tensor([[6, 0, -1], [3, 5, 2], [1, 4, -1]])
    contributions = []
    for rank in range(world_size):
        table = make_table(rank, world_size)
        table.load_rows(weights, scales, chunk_rows=2)
        contributions.append(table.local_lookup(indices))
    expected = weights[indices.clamp_min(0)].bfloat16().masked_fill((indices < 0).unsqueeze(-1), 0)
    torch.testing.assert_close(sum(contributions), expected)


def test_loader_never_reads_whole_table():
    class SliceOnly:
        def __init__(self, tensor):
            self.tensor = tensor
            self.reads = []

        def get_shape(self):
            return self.tensor.shape

        def __getitem__(self, key):
            self.reads.append((key.start, key.stop))
            return self.tensor[key]

    weights = SliceOnly(torch.ones(7, 64, dtype=torch.int8))
    scales = SliceOnly(torch.full((7, 2), 127, dtype=torch.uint8))
    table = make_table(rank=1, world_size=2)
    table.load_rows(weights, scales, chunk_rows=2)
    assert weights.reads == scales.reads == [(4, 6), (6, 7)]
    assert torch.count_nonzero(table.weight[-1]) == 0


def test_loader_rejects_wrong_encoding():
    table = make_table()
    weights = torch.ones(7, 64, dtype=torch.int8)
    scales = torch.full((7, 2), 127, dtype=torch.uint8)
    with pytest.raises(ValueError, match="int8 storage"):
        table.load_rows(weights.float(), scales)
    with pytest.raises(ValueError, match="E8M0 bytes"):
        table.load_rows(weights, scales.float())
    scales[0, 0] = 255
    with pytest.raises(ValueError, match="NaN"):
        table.load_rows(weights, scales)
    with pytest.raises(ValueError, match="quantization"):
        ShardedEngramEmbedding(7, 64, rank=0, world_size=1, device="cpu", quant_type="FP8")


def test_sharded_lookup_requires_matching_group():
    table = make_table(rank=1, world_size=2)
    with pytest.raises(ValueError, match="same group"):
        table(torch.tensor([1]))


def test_layer_hash_view_is_contiguous_before_collective():
    class Group:
        rank_in_group = 0
        world_size = 2

        def all_gather(self, value, dim):
            assert value.is_contiguous()
            return torch.cat((value, value), dim=dim)

        def reduce_scatter(self, value, dim):
            # This fixture selects only rank0 rows; rank1 contributes zeros.
            return value.chunk(2, dim=dim)[0]

    table = make_table(rank=0, world_size=2)
    weights = torch.arange(7 * 64).remainder(127).to(torch.int8).reshape(7, 64)
    table.load_rows(weights, torch.full((7, 2), 127, dtype=torch.uint8))
    all_layer_hashes = torch.tensor([[[1, 2], [6, 0]], [[3, -1], [4, 5]]])
    indices = all_layer_hashes[:, 0]
    assert not indices.is_contiguous()
    output = table(indices, Group())
    expected = weights[indices.clamp_min(0)].bfloat16().masked_fill((indices < 0).unsqueeze(-1), 0)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
