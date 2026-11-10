# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_ascend.patch.worker.deepseek_v41_engram import paged_engram_history


def run_history(cache, blocks, tokens, positions, requests, slots):
    return paged_engram_history(
        torch.tensor(tokens),
        torch.tensor(positions),
        torch.tensor(requests),
        torch.tensor(blocks),
        torch.tensor(slots),
        cache,
        block_size=4,
        max_ngram_size=4,
    )


def test_chunk_boundary_request_order_and_image_tokens():
    # Nonconsecutive physical pages and interleaved requests.
    cache = torch.full((17,), -1, dtype=torch.int64)
    blocks = [[2, 0], [3, 1]]
    run_history(
        cache, blocks, [1, 2, -1, 4, 9, 8, 7, 6], [0, 1, 2, 3] * 2, [0] * 4 + [1] * 4, [8, 9, 10, 11, 12, 13, 14, 15]
    )
    got = run_history(cache, blocks, [5, 5, 6, 4], [4, 4, 5, 5], [0, 1, 0, 1], [0, 4, 1, 5])
    torch.testing.assert_close(got, torch.tensor([[5, 4, -1, 2], [5, 6, 7, 8], [6, 5, 4, -1], [4, 5, 6, 7]]))


def test_full_prefix_block_shared_by_two_requests():
    cache = torch.full((13,), -1, dtype=torch.int64)
    run_history(cache, [[0, 1]], [10, 11, 12, 13], [0, 1, 2, 3], [0] * 4, [0, 1, 2, 3])
    prefix_before = cache[:4].clone()
    got = run_history(cache, [[0, 1], [0, 2]], [14, 24], [4, 4], [0, 1], [4, 8])
    torch.testing.assert_close(got, torch.tensor([[14, 13, 12, 11], [24, 13, 12, 11]]))
    torch.testing.assert_close(cache[:4], prefix_before)


def test_speculative_rejection_overwrites_only_corrected_positions():
    cache = torch.full((9,), -1, dtype=torch.int64)
    blocks = [[0, 1]]
    run_history(cache, blocks, [1, 2, 3, 99, 98], [0, 1, 2, 3, 4], [0] * 5, [0, 1, 2, 3, 4])
    # Tokens 99/98 were rejected; accept a replacement at position3, followed
    # by a new draft. Position4 must never leak into position3's history.
    got = run_history(cache, blocks, [4, 5], [3, 4], [0, 0], [3, 4])
    torch.testing.assert_close(got, torch.tensor([[4, 3, 2, 1], [5, 4, 3, 2]]))


def test_padding_does_not_overwrite_real_slot_zero():
    cache = torch.full((9,), -1, dtype=torch.int64)
    got = run_history(cache, [[0, 1]], [7, 100, 101], [0, 0, 0], [0, -1, -1], [0, -1, -1])
    torch.testing.assert_close(got, torch.tensor([[7, -1, -1, -1], [-1] * 4, [-1] * 4]))
    assert cache[0] == 7


def test_recycled_block_does_not_leak_previous_request():
    cache = torch.arange(9, dtype=torch.int64) + 1000
    got = run_history(cache, [[1, 0]], [1, 2], [0, 1], [0, 0], [4, 5])
    torch.testing.assert_close(got, torch.tensor([[1, -1, -1, -1], [2, 1, -1, -1]]))
