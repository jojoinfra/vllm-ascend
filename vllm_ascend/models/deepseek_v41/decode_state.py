# SPDX-License-Identifier: Apache-2.0
"""Paged text history and the bounded host-to-device inputs for a single DP rank."""

import math
from dataclasses import dataclass

import numpy as np
import torch


def _numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _field(obj, *names):
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


@dataclass
class PagePlan:
    positions: np.ndarray
    slots: np.ndarray
    token_pages: np.ndarray
    active: np.ndarray


class PagedHistory:
    """Only real scheduler pages own state. Stale speculative suffixes are never read."""

    def __init__(self, max_len, block_size, num_blocks):
        if min(max_len, block_size, num_blocks) <= 0 or block_size % 2:
            raise ValueError("Expected a positive, even KV page size")
        self.max_len = max_len
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_pages = math.ceil(max_len / block_size)
        self.num_slots = block_size * num_blocks
        self.tokens = np.full(self.num_slots, -1, dtype=np.int64)

    def plan(self, positions, slots, block_tables, query_ends):
        positions = np.asarray(positions, dtype=np.int64).reshape(-1)
        slots = np.asarray(slots, dtype=np.int64).reshape(-1)
        width = positions.size
        if slots.size < width:
            slots = np.pad(slots, (0, width - slots.size), constant_values=-1)
        slots = slots[:width]
        active = slots >= 0
        positions = np.where(active, positions, 0)
        if active.any() and (slots.max() >= self.num_slots or positions.min() < 0 or positions.max() >= self.max_len):
            raise ValueError("Scheduler token address is outside the configured page pool")
        token_pages = np.zeros((width, self.num_pages), dtype=np.int64)
        if active.any():
            ends = np.asarray(query_ends, dtype=np.int64).reshape(-1)
            tables = np.asarray(block_tables, dtype=np.int64)
            if ends.size == 0 or tables.ndim != 2:
                raise ValueError("Missing scheduler row lengths or physical page table")
            # Padded request rows in some backends end in zero. Only query
            # offsets through the last real token participate in the mapping.
            last = np.flatnonzero(active)[-1]
            enough = np.flatnonzero(ends > last)
            if not enough.size:
                raise ValueError("Scheduler query lengths do not cover active token slots")
            ends = ends[: enough[0] + 1]
            if (np.diff(ends) < 0).any() or (ends < 0).any() or ends.size > len(tables):
                raise ValueError("Invalid scheduler query offsets")
            rows = np.searchsorted(ends, np.flatnonzero(active), side="right")
            lengths = positions[active] // self.block_size + 1
            if lengths.max() > tables.shape[1]:
                raise ValueError("Scheduler page table does not cover the live context")
            columns = min(tables.shape[1], self.num_pages)
            needed = np.arange(columns) < lengths[:, None]
            pages = np.where(needed, tables[rows, :columns], 0)
            if pages.min() < 0 or pages.max() >= self.num_blocks:
                raise ValueError("Physical block ID is outside --num-gpu-blocks-override")
            token_pages[np.flatnonzero(active), :columns] = pages
            physical = pages[np.arange(len(rows)), lengths - 1] * self.block_size
            physical += positions[active] % self.block_size
            if not np.array_equal(physical, slots[active]) or np.unique(physical).size != physical.size:
                raise ValueError("Block table, positions and scheduler slots disagree")
        return PagePlan(positions, slots, token_pages, active)

    def from_context(self, ctx, key, positions):
        width = positions.numel()
        metadata = ctx.attn_metadata
        mappings = getattr(ctx, "slot_mapping", None)
        data = metadata.get(key) if isinstance(metadata, dict) else None
        slots = mappings.get(key) if isinstance(mappings, dict) else None
        if slots is None and data is not None:
            slots = _field(data, "slot_mapping")
        if data is None:
            if slots is not None and (_numpy(slots)[:width] >= 0).any():
                raise RuntimeError("Active slots with no scheduler attention metadata")
            return self.plan(np.zeros(width), np.full(width, -1), [], [])
        if slots is None:
            raise RuntimeError(f"Missing slot mapping on {type(data).__name__}")
        slots = _numpy(slots).reshape(-1)[:width]
        if not (slots >= 0).any():
            return self.plan(np.zeros(width), slots, [], [])
        starts = _field(data, "query_start_loc_cpu", "query_start_loc")
        ends = _numpy(starts)[1:] if starts is not None else _field(data, "actual_seq_lengths_q")
        tables = _field(data, "block_tables", "block_table", "block_table_tensor")
        if ends is None or tables is None:
            raise RuntimeError(f"Missing logical query boundaries or physical block table on {type(data).__name__}")
        return self.plan(_numpy(positions), slots, _numpy(tables), _numpy(ends))

    def hash_ids(self, reference, plan, input_ids):
        """Use exactly HF's compressed IDs, int64 products, XOR, primes and per-head offsets."""
        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)[plan.active]
        active = np.flatnonzero(plan.active)
        pos = plan.positions[active]
        token_map = reference.token_map.numpy()
        if ids.size and (ids.min() < 0 or ids.max() >= token_map.size):
            raise ValueError("Input ID outside the checkpoint tokenizer")
        self.tokens[plan.slots[active]] = token_map[ids]
        tokens = []
        for shift in range(reference.layout.max_ngram_size):
            at = np.maximum(pos - shift, 0)
            addr = plan.token_pages[active, at // self.block_size] * self.block_size + at % self.block_size
            value = self.tokens[addr]
            if (value[pos >= shift] < 0).any():
                raise RuntimeError("Engram history was read before the scheduler filled its KV pages")
            tokens.append(np.where(pos >= shift, value, reference.pad_id))
        products = np.stack(tokens, axis=-1)[:, None, :] * reference.multipliers.numpy()
        rolling, hashes = products[..., 0], []
        for i in range(1, reference.layout.max_ngram_size):
            rolling = np.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling[..., None] % reference.primes.numpy()[:, i - 1])
        return np.concatenate(hashes, axis=-1) + reference.offsets.numpy()


class DecodeInputs:
    """Storage addresses stay constant across capture, prefill, speculative verification and replay."""

    def __init__(self, history, max_tokens, device, engrams):
        self.history = history
        self.max_tokens = max_tokens
        self.device = device
        self.pos = torch.zeros(max_tokens, dtype=torch.long, device=device)
        self.active = torch.zeros(max_tokens, dtype=torch.bool, device=device)
        self.slots = torch.zeros(max_tokens, dtype=torch.long, device=device)
        self.pages = torch.zeros((max_tokens, history.num_pages), dtype=torch.long, device=device)
        self.serial = torch.arange(max_tokens, device=device)
        self.fp4_lut = torch.tensor(
            (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
            device=device,
            dtype=torch.float32,
        )
        self.slots.copy_(self.serial + history.num_slots)
        self.engrams = {}
        for layer in engrams:
            cols = layer.wkv.weight.shape[1] // layer.embed.dim
            shape = (max_tokens, cols, layer.embed.dim)
            self.engrams[layer.layer_hash_index] = (
                torch.zeros(shape, device=device, dtype=torch.uint8),
                torch.zeros((*shape[:2], shape[2] // 32), device=device, dtype=torch.uint8),
            )

    def stage(self, plan, ids, reference, engrams):
        width = len(plan.positions)
        if not 0 < width <= self.max_tokens:
            raise ValueError("Scheduled width exceeds the static input allocation")
        live = np.flatnonzero(plan.active)
        slots = np.where(plan.active, plan.slots, self.history.num_slots + np.arange(width))
        for target, source in (
            (self.pos, plan.positions),
            (self.active, plan.active),
            (self.slots, slots),
            (self.pages, plan.token_pages),
        ):
            target[:width].copy_(torch.from_numpy(source))
        if len(live):
            hashes = self.history.hash_ids(reference, plan, ids)
            for layer in engrams:
                embed = layer.embed
                if embed._weight is None or embed._scale is None:
                    raise RuntimeError("Engram checkpoint must be bound before scheduling")
                raw_dest, scale_dest = self.engrams[layer.layer_hash_index]
                layer_ids = hashes[:, layer.layer_hash_index]
                # Keep file-backed random reads and transient host payload bounded.
                step = max(1, embed.transfer_rows // layer_ids.shape[1])
                for begin in range(0, len(live), step):
                    rows = layer_ids[begin : begin + step]
                    idx = live[begin : begin + step]
                    raw = torch.from_numpy(embed._weight[rows])
                    scales = torch.from_numpy(embed._scale[rows])
                    if idx[-1] - idx[0] + 1 == len(idx):
                        raw_dest[idx[0] : idx[-1] + 1].copy_(raw)
                        scale_dest[idx[0] : idx[-1] + 1].copy_(scales)
                    else:
                        loc = torch.from_numpy(idx).to(self.device)
                        raw_dest.index_copy_(0, loc, raw.to(self.device))
                        scale_dest.index_copy_(0, loc, scales.to(self.device))
        return int(plan.positions.max()) + 1 if len(live) else 1

    def physical(self, positions, rows, ratio=1):
        """Positions are last-token absolute addresses of completed groups when ratio > 1."""
        pages = self.pages[rows].gather(1, positions // self.history.block_size)
        return (pages * self.history.block_size + positions % self.history.block_size) // ratio

    def write_addresses(self, width, ratio=1):
        live = self.active[:width] & ((self.pos[:width] + 1) % ratio == 0)
        return torch.where(live, self.slots[:width] // ratio, self.history.num_slots // ratio + self.serial[:width])
