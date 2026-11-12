# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental device-bounded paged index scores for short decode workloads.

Skip unpopulated compressed positions without host synchronization. Preserve
BF16 rounding after each head dot product, weighted product, and head sum.
The caller supplies zeroed scores and applies the original validity/top-k mask.
"""

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
from cannbotdsl.tensor import tile_slice

INDEX_DIM = 128
VECTOR_LANES = 64


@kernel
class PagedIndexScoresKernel:
    def __init__(self, heads: int, offset: int, block_size: int, ratio: int):
        self.heads = heads
        self.offset = offset
        self.block_size = block_size
        self.ratio = ratio
        self.query = Buffer(MemLoc.UB, (heads * INDEX_DIM,), dtypes.bfloat16)
        self.weights = Buffer(MemLoc.UB, (VECTOR_LANES,), dtypes.bfloat16)
        self.key = Buffer(MemLoc.UB, (INDEX_DIM,), dtypes.bfloat16)
        self.score = Buffer(MemLoc.UB, (VECTOR_LANES,), dtypes.bfloat16)

    def __call__(
        self,
        backing: Tensor,
        query: Tensor,
        weights: Tensor,
        table: Tensor,
        lengths: Tensor,
        active: Tensor,
        output: Tensor,
    ):
        flat = backing.view(backing.shape[0] * backing.shape[1])
        for row in range(get_block_idx(), query.shape[0], get_block_num()):
            enabled = load_bypass(active.ptr(row))
            length = load_bypass(lengths.ptr(row))
            count = length // self.ratio
            if count > output.shape[1]:
                count = output.shape[1]
            if enabled != 0 and count > 0:
                mem_copy(self.query, query[row, None, None].view(self.heads * INDEX_DIM))
                mem_copy(tile_slice(self.weights, (self.heads,), (0,)), weights[row, None].view(self.heads))
                for column in range(count):
                    position = (column + 1) * self.ratio - 1
                    block = position // self.block_size
                    if block < table.shape[1]:
                        physical = load_bypass(table.ptr((row, block)))
                        slot = physical * self.block_size + position % self.block_size
                        if slot >= 0 and slot < backing.shape[0]:
                            mem_copy(
                                self.key,
                                tile_slice(flat, (INDEX_DIM,), ((slot * backing.shape[1] + self.offset) // INDEX_DIM,)),
                            )
                            with vf(mode="raw"):
                                full = rr.full_mask()
                                one = rr.update_mask(1, elem_bits=32)[0]
                                heads_mask = rr.update_mask(self.heads, elem_bits=32)[0]
                                k0 = rr.vcast(
                                    rr.vload_unpack(self.key, 0, unpack_mode=rr.UnpackMode.B16_TO_B32),
                                    dtypes.float32,
                                    mask=full,
                                )
                                k1 = rr.vcast(
                                    rr.vload_unpack(self.key, VECTOR_LANES, unpack_mode=rr.UnpackMode.B16_TO_B32),
                                    dtypes.float32,
                                    mask=full,
                                )
                                weight = rr.vcast(
                                    rr.vload_unpack(self.weights, 0, unpack_mode=rr.UnpackMode.B16_TO_B32),
                                    dtypes.float32,
                                    mask=heads_mask,
                                )
                                partials = rr.vdups(0.0, dtypes.float32, mask=full)
                                lanes = rr.varange(0, dtypes.uint32)
                                for head in range(self.heads):
                                    q0 = rr.vcast(
                                        rr.vload_unpack(
                                            self.query, head * INDEX_DIM, unpack_mode=rr.UnpackMode.B16_TO_B32
                                        ),
                                        dtypes.float32,
                                        mask=full,
                                    )
                                    q1 = rr.vcast(
                                        rr.vload_unpack(
                                            self.query,
                                            head * INDEX_DIM + VECTOR_LANES,
                                            unpack_mode=rr.UnpackMode.B16_TO_B32,
                                        ),
                                        dtypes.float32,
                                        mask=full,
                                    )
                                    product = rr.vadd(rr.vmul(q0, k0, mask=full), rr.vmul(q1, k1, mask=full), mask=full)
                                    dot = rr.vreduce_sum(product, mask=full)
                                    dot = rr.vcast(rr.vcast(dot, dtypes.bfloat16, mask=one), dtypes.float32, mask=one)
                                    w = rr.vgather_reg(weight, rr.vdups(head, dtypes.uint32, mask=one))
                                    weighted = rr.vmul(rr.vmaxs(dot, 0.0, mask=one), w, mask=one)
                                    weighted = rr.vcast(
                                        rr.vcast(weighted, dtypes.bfloat16, mask=one), dtypes.float32, mask=one
                                    )
                                    partials = rr.vselect(
                                        rr.vdup(weighted, mask=full),
                                        partials,
                                        cond_mask=rr.veq(lanes, rr.vdups(head, dtypes.uint32, mask=full), mask=full),
                                    )
                                total = rr.vreduce_sum(partials, mask=heads_mask)
                                rr.vstore_pack(
                                    self.score,
                                    0,
                                    rr.vcast(total, dtypes.bfloat16, mask=one),
                                    one,
                                    pack_mode=rr.PackMode.B32_TO_B16,
                                )
                            mem_copy(
                                tile_slice(
                                    output.view(output.shape[0] * output.shape[1]),
                                    (1,),
                                    (row * output.shape[1] + column,),
                                ),
                                tile_slice(self.score, (1,), (0,)),
                            )


class PagedIndexScores:
    def __init__(self, stride, offset, heads, block_size=128, ratio=8, blocks=64):
        if heads not in (8, 16, 32, 64):
            raise ValueError("Supported index head counts are 8, 16, 32, 64")
        if stride % INDEX_DIM or offset % INDEX_DIM or not 0 <= offset <= stride - INDEX_DIM:
            raise ValueError("Index field must be aligned within its backing row")
        if min(block_size, ratio, blocks) <= 0:
            raise ValueError("Block size, compression ratio and core count must be positive")
        self.stride, self.offset, self.heads = stride, offset, heads
        self.block_size, self.ratio, self.blocks = block_size, ratio, blocks

    @jit
    def run(self, backing, query, weights, table, lengths, active, output):
        op = PagedIndexScoresKernel(self.heads, self.offset, self.block_size, self.ratio)
        op[self.blocks](backing, query, weights, table, lengths, active, output)

    def compile(self):
        slots = Dim("slots", min=1)
        rows = Dim("rows", min=1)
        pages = Dim("pages", min=1)
        columns = Dim("columns", min=1)
        return self.run.compile(
            TensorSpec((slots, self.stride), dtypes.bfloat16),
            TensorSpec((rows, self.heads, INDEX_DIM), dtypes.bfloat16),
            TensorSpec((rows, self.heads), dtypes.bfloat16),
            TensorSpec((rows, pages), dtypes.int32),
            TensorSpec((rows,), dtypes.int64),
            TensorSpec((rows,), dtypes.int32),
            TensorSpec((rows, columns), dtypes.bfloat16),
        )


class PagedIndexBackend:
    """Own one prepared layout; unsupported inputs retain the original path."""

    def __init__(self, stride, offset, heads, block_size, ratio):
        self.stride, self.offset, self.heads = stride, offset, heads
        self.alignment = 16
        kernel_type = PagedIndexScores
        if ratio in (1, 2) and block_size == 128 and heads in (8, 32):
            from vllm_ascend.models.deepseek_v41.cannbot_index_cube import PagedIndexScores as CubeIndexScores

            kernel_type = CubeIndexScores
            self.alignment = 64
        self.kernel = kernel_type(stride, offset, heads, block_size, ratio).compile()

    def __call__(self, backing, query, weights, table, lengths, active, offset, capacity):
        if (
            backing.dtype != torch.bfloat16
            or query.dtype != torch.bfloat16
            or weights.dtype != torch.bfloat16
            or backing.device.type not in ("npu", "privateuseone")
            or backing.shape[1] != self.stride
            or offset != self.offset
            or query.shape[1:] != (self.heads, INDEX_DIM)
            or not backing.is_contiguous()
            or capacity % self.alignment
        ):
            return None
        output = query.new_zeros((query.shape[0], capacity))
        if output.numel():
            self.kernel(
                backing,
                query.detach().contiguous(),
                weights.detach().contiguous(),
                table.to(torch.int32).contiguous(),
                lengths.to(torch.int64).contiguous(),
                active.to(torch.int32).contiguous(),
                output,
            )
        return output
