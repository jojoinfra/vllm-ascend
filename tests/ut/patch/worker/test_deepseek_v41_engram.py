# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.patch.worker.deepseek_v41_engram import EngramHasher, build_engram_token_map


class TokenizerWithPartialBytes:
    def __init__(self):
        self.backend_tokenizer = self
        self.tokens = [" The", "the", "THE", "thé", " ", "", "\ufffd", "\ufffd", "Ａ"]

    def __len__(self):
        return len(self.tokens)

    def decode(self, ids, skip_special_tokens=False):
        assert not skip_special_tokens
        return self.tokens[ids[0]]

    def id_to_token(self, token_id):
        return {6: "<0xC3>", 7: "<0xA9>"}[token_id]


def test_token_normalization_and_partial_bytes():
    tokenizer = TokenizerWithPartialBytes()
    mapping = build_engram_token_map(tokenizer, expected_vocab_size=6)
    assert mapping.tolist() == [0, 0, 0, 0, 1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="refusing to rehash"):
        build_engram_token_map(tokenizer, expected_vocab_size=7)


@pytest.fixture
def oracle():
    return json.loads(Path(__file__).with_name("deepseek_v41_engram_oracle.json").read_text())


def test_official_prefill_and_decode_hashes(oracle):
    hasher = EngramHasher(SimpleNamespace(**oracle["config"]), 10, 0)
    history = torch.tensor(oracle["history"])
    expected = torch.tensor(oracle["expected"])
    torch.testing.assert_close(hasher(history), expected)
    torch.testing.assert_close(hasher(torch.tensor(oracle["decode_history"])), torch.tensor(oracle["expected_decode"]))
    # Scheduler splitting and permutation must not mix request histories.
    for chunk_size in (1, 3, 8):
        actual = torch.cat([hasher(chunk) for chunk in history.split(chunk_size)])
        torch.testing.assert_close(actual, expected)
    order = torch.arange(len(history) - 1, -1, -1)
    torch.testing.assert_close(hasher(history[order]), expected[order])


def test_image_boundary_blocks_older_tokens(oracle):
    hasher = EngramHasher(SimpleNamespace(**oracle["config"]), 10, 0)
    # Image/dead marker blocks the whole older tail, even if it contains text.
    history = torch.tensor([[5, 4, -1, 2], [5, 4, -1, 9], [5, 4, -1, -1]])
    hashes = hasher(history)
    torch.testing.assert_close(hashes[0], hashes[1])
    torch.testing.assert_close(hashes[0], hashes[2])
    # No speculative history is stored in the module itself: replaying an
    # accepted window after a rejected window produces the original result.
    expected = hasher(history[:1]).clone()
    hasher(torch.tensor([[9, 8, 7, 6]]))
    torch.testing.assert_close(hasher(history[:1]), expected)


def test_invalid_checkpoint_layout(oracle):
    config = SimpleNamespace(**oracle["config"])
    with pytest.raises(ValueError, match="vocabulary size"):
        EngramHasher(config, 11, 0)
    with pytest.raises(ValueError, match="padding ID"):
        EngramHasher(config, 10, 10)
    config.engram_num_embeddings[0] += 1
    with pytest.raises(ValueError, match="bucket layout"):
        EngramHasher(config, 10, 0)
