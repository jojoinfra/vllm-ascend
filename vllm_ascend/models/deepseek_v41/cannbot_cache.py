# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in direct paged-cache access for V4.1 decode.

The initial NPU field slices can materialize the complete cache. This kernel
copies only live rows directly into the selected field of the backing storage.
Metadata loads bypass the scalar cache so graph replays observe new values.
Live destinations must be unique, as required by V4.1 decode metadata.
"""

import torch
from cannbotdsl import Dim, MemLoc, Tensor, TensorSpec, dtypes
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.scalar import load_bypass
from cannbotdsl.tensor import tile_slice

MAX_SHORT_READ_POSITIONS = 8


@kernel
class CacheFieldWriteKernel:
    def __init__(self, width: int, offset: int, dtype):
        self.width = width
        self.offset = offset
        self.row = Buffer(MemLoc.UB, (width,), dtype)

    def __call__(self, backing: Tensor, values: Tensor, slots: Tensor, enabled: Tensor):
        flat = backing.view(backing.shape[0] * backing.shape[1])
        for row in range(get_block_idx(), values.shape[0], get_block_num()):
            active = load_bypass(enabled.ptr(row))
            slot = load_bypass(slots.ptr(row))
            if active != 0 and slot >= 0 and slot < backing.shape[0]:
                mem_copy(self.row, values[row, None].view(self.width))
                mem_copy(
                    # tile_slice coordinates count tiles, not elements.
                    tile_slice(flat, (self.width,), ((slot * backing.shape[1] + self.offset) // self.width,)),
                    self.row,
                )


class CacheFieldWrite:
    """Compile one field layout offline or before graph capture."""

    def __init__(self, stride, width, offset, dtype=dtypes.bfloat16, blocks=64):
        if not 0 <= offset < stride or not 0 < width <= stride - offset:
            raise ValueError("Cache field must fit within the backing row")
        if width % 16 or offset % width or stride % width or blocks <= 0:
            raise ValueError("Cache writes require aligned field tiles and positive block counts")
        self.stride = stride
        self.width = width
        self.offset = offset
        self.dtype = dtype
        self.blocks = blocks

    @jit
    def run(self, backing, values, slots, enabled):
        op = CacheFieldWriteKernel(self.width, self.offset, self.dtype)
        op[self.blocks](backing, values, slots, enabled)

    def compile(self):
        pages = Dim("pages", min=1)
        rows = Dim("rows", min=1)
        return self.run.compile(
            TensorSpec((pages, self.stride), self.dtype),
            TensorSpec((rows, self.width), self.dtype),
            TensorSpec((rows,), dtypes.int64),
            TensorSpec((rows,), dtypes.int32),
        )


@kernel
class CacheFieldReadKernel:
    def __init__(self, width: int, offset: int, block_size: int, dtype):
        self.width = width
        self.offset = offset
        self.block_size = block_size
        self.row = Buffer(MemLoc.UB, (width,), dtype)

    def __call__(
        self, backing: Tensor, positions: Tensor, table: Tensor, lengths: Tensor, active: Tensor, output: Tensor
    ):
        columns = positions.shape[1]
        flat = backing.view(backing.shape[0] * backing.shape[1])
        for index in range(get_block_idx(), positions.shape[0] * columns, get_block_num()):
            row = index // columns
            column = index % columns
            enabled = load_bypass(active.ptr(row))
            length = load_bypass(lengths.ptr(row))
            position = load_bypass(positions.ptr((row, column)))
            if enabled != 0 and position >= 0 and position < length:
                block = position // self.block_size
                if block < table.shape[1]:
                    physical = load_bypass(table.ptr((row, block)))
                    slot = physical * self.block_size + position % self.block_size
                    if slot >= 0 and slot < backing.shape[0]:
                        mem_copy(
                            self.row,
                            tile_slice(flat, (self.width,), ((slot * backing.shape[1] + self.offset) // self.width,)),
                        )
                        mem_copy(output[row, column, None].view(self.width), self.row)


class CacheFieldRead:
    """The caller must zero the output before every invocation/replay."""

    def __init__(self, stride, width, offset, block_size=128, dtype=dtypes.bfloat16, blocks=64):
        if not 0 <= offset < stride or not 0 < width <= stride - offset:
            raise ValueError("Cache field must fit within the backing row")
        if width % 16 or offset % width or stride % width or block_size <= 0 or blocks <= 0:
            raise ValueError("Cache reads require aligned fields and positive block counts")
        self.stride = stride
        self.width = width
        self.offset = offset
        self.block_size = block_size
        self.dtype = dtype
        self.blocks = blocks

    @jit
    def run(self, backing, positions, table, lengths, active, output):
        op = CacheFieldReadKernel(self.width, self.offset, self.block_size, self.dtype)
        op[self.blocks](backing, positions, table, lengths, active, output)

    def compile(self):
        pages = Dim("pages", min=1)
        rows = Dim("rows", min=1)
        columns = Dim("columns", min=1)
        blocks = Dim("table_blocks", min=1)
        return self.run.compile(
            TensorSpec((pages, self.stride), self.dtype),
            TensorSpec((rows, columns), dtypes.int64),
            TensorSpec((rows, blocks), dtypes.int32),
            TensorSpec((rows,), dtypes.int32),
            TensorSpec((rows,), dtypes.int32),
            TensorSpec((rows, columns, self.width), self.dtype),
        )


class CacheFieldBackend:
    """Own a fixed set of field kernels prepared outside graph capture.

    Graph measurements favor direct writes and short compressor-group reads.
    Large index/window gathers retain the original implementation.
    """

    def __init__(self, block_size):
        self.block_size = block_size
        self.fields = {}

    def prepare(self, stride, width, offset, dtype):
        key = (stride, width, offset, dtype)
        if key in self.fields:
            return
        if width % 16 or offset % width or stride % width:
            return
        if dtype not in (torch.bfloat16, torch.float32):
            return
        dsl_dtype = dtypes.bfloat16 if dtype == torch.bfloat16 else dtypes.float32
        self.fields[key] = (
            CacheFieldWrite(stride, width, offset, dsl_dtype).compile(),
            CacheFieldRead(stride, width, offset, self.block_size, dsl_dtype).compile(),
        )

    def lookup(self, backing, width, offset):
        if backing.device.type not in ("npu", "privateuseone") or not backing.is_contiguous():
            return None
        return self.fields.get((backing.shape[1], width, offset, backing.dtype))

    def write(self, backing, values, slots, enabled, offset):
        kernels = self.lookup(backing, values.shape[-1], offset)
        if kernels is None:
            return False
        if values.shape[0]:
            kernels[0](
                backing,
                values.to(backing.dtype).contiguous(),
                slots.to(torch.int64).contiguous(),
                enabled.to(torch.int32).contiguous(),
            )
        return True

    def read(self, backing, positions, table, lengths, active, offset, width):
        if positions.shape[1] > MAX_SHORT_READ_POSITIONS:
            return None
        kernels = self.lookup(backing, width, offset)
        if kernels is None:
            return None
        output = backing.new_zeros((*positions.shape, width))
        if output.numel():
            kernels[1](
                backing,
                positions.to(torch.int64).contiguous(),
                table.to(torch.int32).contiguous(),
                lengths.to(torch.int32).contiguous(),
                active.to(torch.int32).contiguous(),
                output,
            )
        return output
