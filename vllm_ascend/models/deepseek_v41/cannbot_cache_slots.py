# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unintegrated vectorized cache read-slot and validity calculation."""

from functools import lru_cache

import torch
from cannbotdsl import Dim, MemLoc, Tensor, TensorSpec, dtypes
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.scalar import load_bypass
from cannbotdsl.ops.sync import vec_sync_all
from cannbotdsl.tensor import tile_slice

LANES = 64
BLOCK_SIZE = 128


@kernel
class CacheReadSlotsKernel:
    def __init__(self, table_width: int, max_length: int, storage_rows: int):
        self.table_width, self.max_length, self.storage_rows = table_width, max_length, storage_rows
        self.positions = Buffer(MemLoc.UB, (2 * LANES,), dtypes.int32)
        self.length = Buffer(MemLoc.UB, (LANES,), dtypes.int32)
        self.table = Buffer(MemLoc.UB, ((table_width + LANES - 1) // LANES * LANES,), dtypes.int32)
        self.slots = Buffer(MemLoc.UB, (LANES,), dtypes.int32)
        self.valid = Buffer(MemLoc.UB, (LANES,), dtypes.uint8)

    def __call__(self, positions: Tensor, table: Tensor, lengths: Tensor, active: Tensor, slots: Tensor, valid: Tensor):
        tiles = positions.shape[1] // LANES
        for job in range(get_block_idx(), positions.shape[0] * tiles, get_block_num()):
            row, column = job // tiles, job % tiles
            enabled = load_bypass(active.ptr(row))
            mem_copy(tile_slice(self.length, (2,), (0,)), lengths[row, None].view(2))
            mem_copy(self.positions, tile_slice(positions, (1, LANES, 2), (row, column, 0)).view(2 * LANES))
            mem_copy(tile_slice(self.table, (self.table_width,), (0,)), table[row, None].view(self.table_width))
            with vf(mode="raw"):
                full = rr.full_mask()
                zero = rr.vdups(0, dtypes.uint32, mask=full)
                one = rr.vdups(1, dtypes.uint32, mask=full)
                # Retain both halves: out-of-range int64 positions must never
                # wrap into apparently valid low 32-bit positions.
                low, high = rr.vload_deinterleave(self.positions, 0, width="b32")
                low, high = rr.vreinterpret(low, dtypes.uint32), rr.vreinterpret(high, dtypes.uint32)
                length_words = rr.vload(self.length, 0)
                length_low = rr.vreinterpret(rr.vgather_reg(length_words, zero), dtypes.uint32)
                length_high = rr.vgather_reg(length_words, one)
                signed_high = rr.vreinterpret(high, dtypes.int32)
                live = rr.vselect(one, zero, cond_mask=rr.vlt(low, length_low, mask=full))
                live = rr.vselect(live, zero, cond_mask=rr.veq(signed_high, length_high, mask=full))
                live = rr.vselect(one, live, cond_mask=rr.vlt(signed_high, length_high, mask=full))
                live = rr.vselect(
                    live,
                    zero,
                    cond_mask=rr.vge(signed_high, rr.vdups(0, dtypes.int32, mask=full), mask=full),
                )
                live = rr.vselect(
                    live, zero, cond_mask=rr.vne(rr.vdups(enabled, dtypes.uint32, mask=full), zero, mask=full)
                )
                clamped = rr.vmin(low, rr.vdups(self.max_length - 1, dtypes.uint32, mask=full), mask=full)
                clamped = rr.vselect(
                    rr.vdups(self.max_length - 1, dtypes.uint32, mask=full),
                    clamped,
                    cond_mask=rr.vne(high, zero, mask=full),
                )
                block = rr.vshr(clamped, 7, mask=full)
                block = rr.vselect(block, zero, cond_mask=rr.vne(live, zero, mask=full))
                physical = rr.vgather(self.table, block, mask=full)
                live = rr.vselect(
                    live, zero, cond_mask=rr.vge(physical, rr.vdups(0, dtypes.int32, mask=full), mask=full)
                )
                live = rr.vselect(
                    live,
                    zero,
                    cond_mask=rr.vlt(
                        physical,
                        rr.vdups((self.storage_rows + BLOCK_SIZE - 1) // BLOCK_SIZE, dtypes.int32, mask=full),
                        mask=full,
                    ),
                )
                physical = rr.vselect(
                    rr.vreinterpret(physical, dtypes.uint32), zero, cond_mask=rr.vne(live, zero, mask=full)
                )
                offset = rr.vbitwise_and(clamped, rr.vdups(BLOCK_SIZE - 1, dtypes.uint32, mask=full), mask=full)
                slot = rr.vadd(rr.vmuls(physical, BLOCK_SIZE, mask=full), offset, mask=full)
                live = rr.vselect(
                    live, zero, cond_mask=rr.vlt(slot, rr.vdups(self.storage_rows, dtypes.uint32, mask=full), mask=full)
                )
                safe = rr.vselect(slot, zero, cond_mask=rr.vne(live, zero, mask=full))
                rr.vstore(self.slots, 0, rr.vreinterpret(safe, dtypes.int32), full)
                rr.vstore_pack(self.valid, 0, live, full, pack_mode=rr.PackMode.B32_TO_B8)
            mem_copy(tile_slice(slots, (1, LANES), (row, column)).view(LANES), self.slots)
            mem_copy(tile_slice(valid, (1, LANES), (row, column)).view(LANES), self.valid)
            vec_sync_all()


class CacheReadSlots:
    def __init__(self, table_width, max_length, storage_rows, blocks=64):
        if not 0 < table_width <= 512 or not 0 < max_length <= table_width * BLOCK_SIZE:
            raise ValueError("Cache slot prototype requires a covering table up to 512 blocks")
        if not 0 < storage_rows < 2**31:
            raise ValueError("Cache slots must fit in nonnegative int32")
        self.table_width, self.max_length, self.storage_rows = table_width, max_length, storage_rows
        self.blocks = blocks
        self.compiled = lru_cache(maxsize=1)(self._compile)

    @jit
    def run(self, positions, table, lengths, active, slots, valid):
        op = CacheReadSlotsKernel(self.table_width, self.max_length, self.storage_rows)
        op[self.blocks](positions, table, lengths, active, slots, valid)

    def _compile(self):
        rows, columns = Dim("rows", min=1), Dim("columns", min=LANES)
        return self.run.compile(
            TensorSpec((rows, columns, 2), dtypes.int32),
            TensorSpec((rows, self.table_width), dtypes.int32),
            TensorSpec((rows, 2), dtypes.int32),
            TensorSpec((rows,), dtypes.uint8),
            TensorSpec((rows, columns), dtypes.int32),
            TensorSpec((rows, columns), dtypes.uint8),
        )

    def __call__(self, positions, table, lengths, active):
        if positions.ndim != 2 or positions.shape[1] % LANES:
            raise ValueError("Cache slot prototype requires a 64-aligned position matrix")
        rows = positions.shape[0]
        if table.shape != (rows, self.table_width) or lengths.shape != (rows,) or active.shape != (rows,):
            raise ValueError("Cache slot metadata shapes do not match")
        if (
            positions.dtype != torch.int64
            or table.dtype != torch.int32
            or lengths.dtype not in (torch.int32, torch.int64)
            or active.dtype != torch.bool
        ):
            raise ValueError(
                "Cache slots require int64 positions, int32 tables, integer lengths and boolean active flags"
            )
        if positions.device.type not in ("npu", "privateuseone") or any(
            x.device != positions.device for x in (table, lengths, active)
        ):
            raise ValueError("Cache slot inputs must share an NPU")
        slots = torch.empty_like(positions, dtype=torch.int32, memory_format=torch.contiguous_format)
        valid = torch.empty_like(positions, dtype=torch.uint8, memory_format=torch.contiguous_format)
        if positions.numel():
            self.compiled()(
                positions.contiguous().view(torch.int32).view(rows, positions.shape[1], 2),
                table.contiguous(),
                lengths.to(torch.int64).contiguous().view(torch.int32).view(rows, 2),
                active.contiguous().view(torch.uint8),
                slots,
                valid,
            )
        return slots, valid.view(torch.bool)
