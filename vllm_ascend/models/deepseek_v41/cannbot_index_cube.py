# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental cube implementation for device-bounded ratio-1/2 index scores.

Keep per-head BF16 rounding, then BF16 weighted products and head reduction.
Explicit synchronization protects reusable on-chip buffers during graph replay.
"""

import torch
from cannbotdsl import Dim, MemLoc, Tensor, TensorSpec, dtypes
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.scalar import load_bypass
from cannbotdsl.ops.sync import cube_sync_all, vec_sync_all
from cannbotdsl.tensor import tile_slice

TILE = 64
DIMENSION = 128
MAX_GRAPH_SHAPES = 8


@kernel
class PagedIndexCubeKernel:
    def __init__(self, stride: int, offset: int, heads: int, ratio: int):
        self.stride, self.offset, self.heads, self.ratio = stride, offset, heads, ratio
        self.l1q = Buffer(MemLoc.L1, (heads, DIMENSION), dtypes.bfloat16)
        self.l1k = Buffer(MemLoc.L1, (TILE, DIMENSION), dtypes.bfloat16)
        self.l0q = Buffer(MemLoc.L0A, (heads, DIMENSION), dtypes.bfloat16)
        self.l0k = Buffer(MemLoc.L0B, (TILE, DIMENSION), dtypes.bfloat16)
        self.acc = Buffer(MemLoc.L0C, (heads, TILE), dtypes.float32)

    def __call__(
        self, backing: Tensor, query: Tensor, table: Tensor, lengths: Tensor, active: Tensor, per_head: Tensor
    ):
        tiles = per_head.shape[2] // TILE
        for row in range(query.shape[0]):
            count = load_bypass(lengths.ptr(row)) // self.ratio
            live = load_bypass(active.ptr(row))
            if live != 0 and count > 0:
                mem_copy(self.l1q, query[row, None, None].view(self.heads, DIMENSION))
                mem_copy(self.l0q, self.l1q)
                for tile in range(get_block_idx(), tiles, get_block_num()):
                    if tile * TILE < count:
                        position = (tile * TILE + 1) * self.ratio - 1
                        block = position // 128
                        if block < table.shape[1]:
                            physical = load_bypass(table.ptr((row, block)))
                            slot = physical * 128 + position % 128
                            if slot >= 0 and slot + (TILE - 1) * self.ratio < backing.shape[0] * self.ratio:
                                keys = backing[
                                    slot // self.ratio : slot // self.ratio + TILE,
                                    self.offset : self.offset + DIMENSION,
                                ]
                                mem_copy(self.l1k, keys)
                                mem_copy(self.l0k, self.l1k)
                                matmul(self.acc, self.l0q, self.l0k)
                                mem_copy(
                                    tile_slice(
                                        per_head[row, None, None].view(self.heads, per_head.shape[2]),
                                        (self.heads, TILE),
                                        (0, tile),
                                    ),
                                    self.acc,
                                )
                                cube_sync_all()


@kernel
class IndexHeadReduceKernel:
    def __init__(self, heads: int, ratio: int):
        self.heads, self.ratio = heads, ratio
        self.input = Buffer(MemLoc.UB, (heads, TILE), dtypes.bfloat16)
        self.weights = Buffer(MemLoc.UB, (TILE,), dtypes.bfloat16)
        self.weighted = Buffer(MemLoc.UB, (heads * TILE,), dtypes.float32)
        self.output = Buffer(MemLoc.UB, (TILE,), dtypes.bfloat16)

    def __call__(
        self,
        backing: Tensor,
        per_head: Tensor,
        weights: Tensor,
        table: Tensor,
        lengths: Tensor,
        active: Tensor,
        output: Tensor,
    ):
        tiles = output.shape[1] // TILE
        for row in range(output.shape[0]):
            count = load_bypass(lengths.ptr(row)) // self.ratio
            live = load_bypass(active.ptr(row))
            if live != 0 and count > 0:
                mem_copy(tile_slice(self.weights, (self.heads,), (0,)), weights[row, None].view(self.heads))
                for tile in range(get_block_idx(), tiles, get_block_num()):
                    if tile * TILE < count:
                        position = (tile * TILE + 1) * self.ratio - 1
                        block = position // 128
                        if block < table.shape[1]:
                            physical = load_bypass(table.ptr((row, block)))
                            if physical >= 0 and (physical + 1) * 128 <= backing.shape[0] * self.ratio:
                                mem_copy(
                                    self.input,
                                    tile_slice(
                                        per_head[row, None, None].view(self.heads, per_head.shape[2]),
                                        (self.heads, TILE),
                                        (0, tile),
                                    ),
                                )
                                with vf(mode="raw"):
                                    full = rr.full_mask()
                                    hm = rr.update_mask(self.heads, elem_bits=32)[0]
                                    w = rr.vcast(
                                        rr.vload_unpack(self.weights, 0, unpack_mode=rr.UnpackMode.B16_TO_B32),
                                        dtypes.float32,
                                        mask=hm,
                                    )
                                    for head in range(self.heads):
                                        value = rr.vcast(
                                            rr.vload_unpack(
                                                self.input, head * TILE, unpack_mode=rr.UnpackMode.B16_TO_B32
                                            ),
                                            dtypes.float32,
                                            mask=full,
                                        )
                                        weight = rr.vgather_reg(w, rr.vdups(head, dtypes.uint32, mask=full))
                                        value = rr.vmul(rr.vmaxs(value, 0.0, mask=full), weight, mask=full)
                                        value = rr.vcast(
                                            rr.vcast(value, dtypes.bfloat16, mask=full), dtypes.float32, mask=full
                                        )
                                        rr.vstore(self.weighted, head * TILE, value, full)
                                vec_sync_all()
                                with vf(mode="raw"):
                                    full = rr.full_mask()
                                    hm = rr.update_mask(self.heads, elem_bits=32)[0]
                                    lanes = rr.varange(0, dtypes.uint32)
                                    head_offsets = rr.vmuls(lanes, TILE, mask=full)
                                    result = rr.vdups(0.0, dtypes.float32, mask=full)
                                    for column in range(TILE):
                                        values = rr.vgather(
                                            self.weighted, rr.vadds(head_offsets, column, mask=full), mask=hm
                                        )
                                        total = rr.vdup(rr.vreduce_sum(values, mask=hm), mask=full)
                                        result = rr.vselect(
                                            total,
                                            result,
                                            cond_mask=rr.veq(
                                                lanes, rr.vdups(column, dtypes.uint32, mask=full), mask=full
                                            ),
                                        )
                                    remaining = rr.vdups(count - tile * TILE, dtypes.uint32, mask=full)
                                    result = rr.vselect(
                                        result,
                                        rr.vdups(0.0, dtypes.float32, mask=full),
                                        cond_mask=rr.vlt(lanes, remaining, mask=full),
                                    )
                                    rr.vstore_pack(
                                        self.output,
                                        0,
                                        rr.vcast(result, dtypes.bfloat16, mask=full),
                                        full,
                                        pack_mode=rr.PackMode.B32_TO_B16,
                                    )
                                mem_copy(
                                    tile_slice(output[row, None].view(output.shape[1]), (TILE,), (tile,)), self.output
                                )
                                vec_sync_all()


class PagedIndexScores:
    def __init__(self, stride, offset, heads, block_size=128, ratio=2, blocks=32):
        if ratio not in (1, 2) or block_size != 128 or heads not in (8, 32):
            raise ValueError("Cube prototype requires ratio 1/2, block 128 and 8/32 heads")
        self.stride, self.offset, self.heads, self.ratio, self.blocks = stride, offset, heads, ratio, blocks
        self.scratch = {}

    @jit
    def run(self, backing, query, weights, table, lengths, active, output, partial):
        cube = PagedIndexCubeKernel(self.stride, self.offset, self.heads, self.ratio)
        reduce = IndexHeadReduceKernel(self.heads, self.ratio)
        cube[self.blocks](backing, query, table, lengths, active, partial)
        reduce[self.blocks * 2](backing, partial, weights, table, lengths, active, output)

    def compile(self):
        slots, rows, pages, columns = (Dim(name, min=1) for name in ("slots", "rows", "pages", "columns"))
        compiled = self.run.compile(
            TensorSpec((slots, self.stride), dtypes.bfloat16, stride=(self.ratio * self.stride, 1)),
            TensorSpec((rows, self.heads, DIMENSION), dtypes.bfloat16),
            TensorSpec((rows, self.heads), dtypes.bfloat16),
            TensorSpec((rows, pages), dtypes.int64),
            TensorSpec((rows,), dtypes.int64),
            TensorSpec((rows,), dtypes.int32),
            TensorSpec((rows, columns), dtypes.bfloat16),
            TensorSpec((rows, self.heads, columns), dtypes.bfloat16),
        )

        def call(backing, query, weights, table, lengths, active, output):
            if output.shape[1] % TILE:
                raise ValueError("Cube prototype capacity must be a multiple of 64")
            key = (query.shape[0], output.shape[1], str(query.device))
            if key not in self.scratch:
                if len(self.scratch) >= MAX_GRAPH_SHAPES:
                    raise ValueError("Create a separate backend for more than eight graph shapes")
                self.scratch[key] = torch.empty(
                    (query.shape[0], self.heads, output.shape[1]), dtype=query.dtype, device=query.device
                )
            partial = self.scratch[key]
            selected = torch.as_strided(
                backing,
                (backing.shape[0] // self.ratio, self.stride),
                (self.ratio * self.stride, 1),
                storage_offset=backing.storage_offset() + (self.ratio - 1) * self.stride,
            )
            compiled(selected, query, weights, table.long(), lengths.long(), active, output, partial)

        call.raw = compiled
        return call
