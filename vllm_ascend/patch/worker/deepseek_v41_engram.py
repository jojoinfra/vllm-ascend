# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram hashing over explicit request-local history, without mutable globals.

Hash arithmetic and prime ordering follow the official DeepSeek V4.1 reference:
https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/engram.py
"""

import numpy as np
import torch
from sympy import nextprime
from tokenizers import Regex, normalizers
from torch import nn


def paged_engram_history(
    compressed_ids: torch.Tensor,
    positions: torch.Tensor,
    request_indices: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    cache: torch.Tensor,
    *,
    block_size: int,
    max_ngram_size: int,
) -> torch.Tensor:
    """Write scheduled IDs and gather request-local, newest-first windows.

    The scheduler owns physical blocks (including prefix reuse/copy-on-write).
    ``cache`` contains one int64 per token slot and one final dummy-write slot.
    Rejected speculative positions need no host rollback: the next forward
    overwrites them at corrected positions before any history read. Image
    positions arrive as -1, which the hasher treats as a lookback boundary.

    This primitive does not allocate/manage blocks or infer acceptance lengths;
    callers must supply the scheduler's current device-side positions and slots.
    """
    if cache.ndim != 1 or cache.dtype != torch.int64:
        raise ValueError("Engram history cache must be a flat int64 token cache")
    if block_size < 1 or max_ngram_size < 2 or (cache.numel() - 1) % block_size:
        raise ValueError("Engram history needs complete blocks plus one dummy slot")
    dummy_slot = cache.numel() - 1
    writable = (slot_mapping >= 0) & (slot_mapping < dummy_slot)
    slots = torch.where(writable, slot_mapping, dummy_slot).long()
    cache.index_copy_(0, slots, torch.where(writable, compressed_ids, -1))

    history_positions = positions.unsqueeze(-1) - torch.arange(max_ngram_size, device=positions.device)
    logical_blocks = history_positions.clamp_min(0) // block_size
    request_valid = (request_indices >= 0) & (request_indices < block_table.shape[0])
    safe_requests = request_indices.clamp(0, block_table.shape[0] - 1).long()
    physical_blocks = block_table[
        safe_requests.unsqueeze(-1), logical_blocks.clamp_max(block_table.shape[1] - 1).long()
    ]
    history_slots = physical_blocks * block_size + history_positions.remainder(block_size)
    readable = (
        writable.unsqueeze(-1)
        & request_valid.unsqueeze(-1)
        & (history_positions >= 0)
        & (logical_blocks < block_table.shape[1])
        & (physical_blocks >= 0)
        & (history_slots < dummy_slot)
    )
    history_slots = torch.where(readable, history_slots, dummy_slot).long()
    return torch.where(readable, cache[history_slots], -1)


def build_engram_token_map(tokenizer, expected_vocab_size: int) -> torch.Tensor:
    """Build the training-compatible compressed vocabulary once on the host.

    Decode with the Rust backend: transformers cleanup would change hash IDs.
    Partial UTF-8 tokens use their raw token spelling instead of replacement
    characters. A single space must survive stripping as its own token.
    """
    space_marker = "\ue000"
    normalize = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), space_marker),
            normalizers.Strip(),
            normalizers.Replace(space_marker, " "),
        ]
    )
    vocabulary = {}
    mapping = []
    backend = tokenizer.backend_tokenizer
    for token_id in range(len(tokenizer)):
        decoded = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in decoded:
            key = backend.id_to_token(token_id)
        else:
            key = normalize.normalize_str(decoded) or decoded
        if key not in vocabulary:
            vocabulary[key] = len(vocabulary)
        mapping.append(vocabulary[key])
    if len(vocabulary) != expected_vocab_size:
        raise ValueError(
            f"Compressed vocabulary has {len(vocabulary)} entries, expected {expected_vocab_size}; "
            "refusing to rehash the checkpoint"
        )
    return torch.tensor(mapping, dtype=torch.int64)


class EngramHasher(nn.Module):
    """Hash compressed token windows supplied by the request's cache manager.

    Input shape is ``[num_tokens, max_ngram_size]``, newest token first.
    Negative entries represent a sequence start or image token. A boundary
    blocks all older tokens, including text before an image. This module owns
    only immutable hash constants, not request or speculative-decoding state.
    """

    HASH_SEED_MULTIPLIER = 10007

    def __init__(self, config, compressed_vocab_size: int, pad_id: int):
        super().__init__()
        if compressed_vocab_size != config.engram_compressed_vocab_size:
            raise ValueError("Compressed vocabulary size does not match the Engram checkpoint")
        if not 0 <= pad_id < compressed_vocab_size:
            raise ValueError("Compressed padding ID must be inside the compressed vocabulary")
        self.pad_id = pad_id
        self.max_ngram_size = config.engram_max_ngram_size
        if self.max_ngram_size < 2 or config.engram_n_heads < 1 or not config.engram_layer_ids:
            raise ValueError("Engram requires layers, heads and n-grams of size at least two")
        if len(config.engram_num_embeddings) != len(config.engram_layer_ids):
            raise ValueError("Each Engram layer requires its table row count")

        prime_rows, offset_rows, multiplier_rows = [], [], []
        used_primes = set()
        multiplier_bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
        for layer_id, table_rows in zip(config.engram_layer_ids, config.engram_num_embeddings):
            primes, offsets, offset = [], [], 0
            for _ in range(self.max_ngram_size - 1):
                current = config.engram_vocab_size - 1
                for _ in range(config.engram_n_heads):
                    current = int(nextprime(current))
                    while current in used_primes:
                        current = int(nextprime(current))
                    used_primes.add(current)
                    primes.append(current)
                    offsets.append(offset)
                    offset += current
            if offset != table_rows:
                raise ValueError(f"Engram layer {layer_id} bucket layout does not match the checkpoint table")
            generator = np.random.default_rng(self.HASH_SEED_MULTIPLIER * layer_id)
            multipliers = generator.integers(0, multiplier_bound, size=self.max_ngram_size, dtype=np.int64)
            multiplier_rows.append(multipliers * 2 + 1)
            prime_rows.append(primes)
            offset_rows.append(offsets)

        self.register_buffer("primes", torch.tensor(prime_rows, dtype=torch.int64), persistent=False)
        self.register_buffer("offsets", torch.tensor(offset_rows, dtype=torch.int64), persistent=False)
        self.register_buffer("multipliers", torch.tensor(np.array(multiplier_rows)), persistent=False)

    def forward(self, compressed_history: torch.Tensor) -> torch.Tensor:
        if compressed_history.ndim != 2 or compressed_history.shape[1] != self.max_ngram_size:
            raise ValueError("Expected flattened tokens with newest-first Engram history")
        # No device-to-host reads: the loop length is a model constant.
        blocked = torch.zeros_like(compressed_history[:, 0], dtype=torch.bool)
        products = []
        for shift in range(self.max_ngram_size):
            source = compressed_history[:, shift]
            blocked = blocked | (source < 0)
            token = torch.where(blocked, self.pad_id, source)
            products.append(token.unsqueeze(-1) * self.multipliers[:, shift])
        rolling, hashes = products[0], []
        heads = self.primes.shape[1] // (self.max_ngram_size - 1)
        for shift in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[shift])
            bucket = slice((shift - 1) * heads, shift * heads)
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, bucket] + self.offsets[:, bucket])
        return torch.cat(hashes, dim=-1)
