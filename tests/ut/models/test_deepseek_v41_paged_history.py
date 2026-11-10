# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest import TestCase

import numpy as np
import torch

from vllm_ascend.models.deepseek_v41.decode_state import PagedHistory


class TestPagedHistory(TestCase):
    def test_interleaved_chunks_rejection_and_page_reuse(self):
        state = PagedHistory(max_len=8, block_size=2, num_blocks=10)
        reference = SimpleNamespace(
            layout=SimpleNamespace(max_ngram_size=3),
            token_map=torch.arange(32),
            multipliers=torch.tensor([[3, 5, 7]]),
            primes=torch.tensor([[[11], [13]]]),
            offsets=torch.tensor([[0, 11]]),
            pad_id=0,
        )
        tables = {"a": [4, 7, 1, 9], "b": [2, 6, 3, 8], "c": [4, 7, 1, 9]}
        sequences = {key: [] for key in tables}

        def compare(chunks):
            ids, starts, slots, ends, pages, expected = [], [], [], [], [], []
            for req, start, new in chunks:
                prefix = sequences[req][:start]
                self.assertEqual(len(prefix), start)
                for token in new:
                    pos = len(prefix)
                    two = 3 * token ^ 5 * (prefix[-1] if prefix else 0)
                    three = two ^ 7 * (prefix[-2] if len(prefix) >= 2 else 0)
                    expected.append([two % 11, 11 + three % 13])
                    ids.append(token)
                    starts.append(pos)
                    slots.append(tables[req][pos // 2] * 2 + pos % 2)
                    prefix.append(token)
                pages.append(tables[req])
                ends.append(len(ids))
                sequences[req] = prefix
            plan = state.plan(starts + [0], slots + [-1], pages, ends)
            hashes = state.hash_ids(reference, plan, ids + [0])[:, 0]
            np.testing.assert_array_equal(hashes, expected)

        compare([("a", 0, [1, 2]), ("b", 0, [3])])
        compare([("b", 1, [4, 5]), ("a", 2, [6, 7])])
        compare([("a", 4, [8, 9, 10]), ("b", 3, [11])])
        compare([("b", 4, [13]), ("a", 5, [14])])
        compare([("c", 0, [15, 16, 17]), ("b", 5, [18])])
        before = state.tokens.copy()
        idle = state.plan([0], [-1], [], [])
        self.assertFalse(idle.active.any())
        np.testing.assert_array_equal(before, state.tokens)

    def test_inconsistent_physical_slot_rejected(self):
        state = PagedHistory(max_len=8, block_size=2, num_blocks=10)
        with self.assertRaisesRegex(ValueError, "disagree"):
            state.plan([0], [2], [[4, 7]], [1])
        with self.assertRaisesRegex(ValueError, "outside"):
            state.plan([0], [20], [[10]], [1])
