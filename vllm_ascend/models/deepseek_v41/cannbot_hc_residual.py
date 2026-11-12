# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unintegrated HC4 residual mixing with the initial NPU pairwise reduction."""

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
from cannbotdsl.ops.sync import vec_sync_all
from cannbotdsl.tensor import tile_slice

TILE = 64
COPIES = 4


@kernel
class HCPreKernel:
    def __init__(self, width: int):
        self.width = width
        self.tile = 256 if width % 256 == 0 else TILE
        self.input = Buffer(MemLoc.UB, (COPIES * self.tile,), dtypes.bfloat16)
        self.coefficients = Buffer(MemLoc.UB, (self.tile,), dtypes.float32)
        self.output = Buffer(MemLoc.UB, (self.tile,), dtypes.bfloat16)

    def __call__(self, x: Tensor, coefficients: Tensor, output: Tensor):
        tiles = self.width // self.tile
        for job in range(get_block_idx(), x.shape[0] * tiles, get_block_num()):
            row, column = job // tiles, job % tiles
            for channel in range(COPIES):
                mem_copy(
                    tile_slice(self.input, (self.tile,), (channel,)),
                    tile_slice(x, (1, 1, self.tile), (row, channel, column)).view(self.tile),
                )
            mem_copy(tile_slice(self.coefficients, (COPIES,), (0,)), coefficients[row, None].view(COPIES))
            with vf(mode="raw"):
                full = rr.full_mask()
                coeff = rr.vload(self.coefficients, 0)
                for offset in range(0, self.tile, TILE):
                    a = rr.vcast(
                        rr.vload_unpack(self.input, offset, unpack_mode=rr.UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=full,
                    )
                    b = rr.vcast(
                        rr.vload_unpack(self.input, self.tile + offset, unpack_mode=rr.UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=full,
                    )
                    c = rr.vcast(
                        rr.vload_unpack(self.input, 2 * self.tile + offset, unpack_mode=rr.UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=full,
                    )
                    d = rr.vcast(
                        rr.vload_unpack(self.input, 3 * self.tile + offset, unpack_mode=rr.UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=full,
                    )
                    a = rr.vmul(a, rr.vgather_reg(coeff, rr.vdups(0, dtypes.uint32, mask=full)), mask=full)
                    b = rr.vmul(b, rr.vgather_reg(coeff, rr.vdups(1, dtypes.uint32, mask=full)), mask=full)
                    c = rr.vmul(c, rr.vgather_reg(coeff, rr.vdups(2, dtypes.uint32, mask=full)), mask=full)
                    d = rr.vmul(d, rr.vgather_reg(coeff, rr.vdups(3, dtypes.uint32, mask=full)), mask=full)
                    value = rr.vadd(rr.vadd(a, c, mask=full), rr.vadd(b, d, mask=full), mask=full)
                    rr.vstore_pack(
                        self.output,
                        offset,
                        rr.vcast(value, dtypes.bfloat16, mask=full),
                        full,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )
            mem_copy(tile_slice(output, (1, self.tile), (row, column)).view(self.tile), self.output)
            vec_sync_all()


class HCPre:
    def __init__(self, width=5120, blocks=64):
        if type(width) is not int or width <= 0 or width % TILE:
            raise ValueError("HC pre prototype requires a positive 64-aligned width")
        self.width, self.blocks = width, blocks
        self.compiled = lru_cache(maxsize=1)(self._compile)

    @jit
    def run(self, x, coefficients, output):
        op = HCPreKernel(self.width)
        op[self.blocks](x, coefficients, output)

    def _compile(self):
        rows = Dim("rows", min=1)
        return self.run.compile(
            TensorSpec((rows, COPIES, self.width), dtypes.bfloat16),
            TensorSpec((rows, COPIES), dtypes.float32),
            TensorSpec((rows, self.width), dtypes.bfloat16),
        )

    def __call__(self, x, coefficients):
        if x.ndim != 3 or x.shape[1:] != (COPIES, self.width) or coefficients.shape != x.shape[:2]:
            raise ValueError("HC pre input/coefficient shape mismatch")
        if x.dtype != torch.bfloat16 or coefficients.dtype != torch.float32 or coefficients.device != x.device:
            raise ValueError("HC pre requires BF16 states and FP32 coefficients on the same device")
        if x.device.type not in ("npu", "privateuseone"):
            raise ValueError("HC pre prototype requires an NPU")
        if torch.is_grad_enabled() and (x.requires_grad or coefficients.requires_grad):
            raise ValueError("HC pre prototype is inference-only")
        output = torch.empty((x.shape[0], self.width), dtype=x.dtype, device=x.device)
        if x.numel():
            self.compiled()(x.detach().contiguous(), coefficients.detach().contiguous(), output)
        return output


@kernel
class HCPostKernel:
    def __init__(self, width: int):
        self.width = width
        self.tile = 256 if width % 256 == 0 else TILE
        self.residual = Buffer(MemLoc.UB, (COPIES * self.tile,), dtypes.bfloat16)
        self.sublayer = Buffer(MemLoc.UB, (self.tile,), dtypes.bfloat16)
        self.post = Buffer(MemLoc.UB, (self.tile,), dtypes.float32)
        self.combination = Buffer(MemLoc.UB, (self.tile,), dtypes.float32)
        self.output = Buffer(MemLoc.UB, (COPIES * self.tile,), dtypes.bfloat16)

    def __call__(self, sublayer: Tensor, residual: Tensor, post: Tensor, combination: Tensor, output: Tensor):
        tiles = self.width // self.tile
        for job in range(get_block_idx(), residual.shape[0] * tiles, get_block_num()):
            row, column = job // tiles, job % tiles
            for channel in range(COPIES):
                mem_copy(
                    tile_slice(self.residual, (self.tile,), (channel,)),
                    tile_slice(residual, (1, 1, self.tile), (row, channel, column)).view(self.tile),
                )
            mem_copy(self.sublayer, tile_slice(sublayer, (1, self.tile), (row, column)).view(self.tile))
            mem_copy(tile_slice(self.post, (COPIES,), (0,)), post[row, None].view(COPIES))
            mem_copy(
                tile_slice(self.combination, (COPIES * COPIES,), (0,)), combination[row, None].view(COPIES * COPIES)
            )
            with vf(mode="raw"):
                full = rr.full_mask()
                coeff = rr.vload(self.combination, 0)
                post_coeff = rr.vload(self.post, 0)
                for offset in range(0, self.tile, TILE):
                    y = rr.vcast(
                        rr.vload_unpack(self.sublayer, offset, unpack_mode=rr.UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=full,
                    )
                    r0 = rr.vcast(
                        rr.vload_unpack(self.residual, offset, unpack_mode=rr.UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=full,
                    )
                    r1 = rr.vcast(
                        rr.vload_unpack(self.residual, self.tile + offset, unpack_mode=rr.UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=full,
                    )
                    r2 = rr.vcast(
                        rr.vload_unpack(self.residual, 2 * self.tile + offset, unpack_mode=rr.UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=full,
                    )
                    r3 = rr.vcast(
                        rr.vload_unpack(self.residual, 3 * self.tile + offset, unpack_mode=rr.UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=full,
                    )
                    for channel in range(COPIES):
                        a = rr.vmul(r0, rr.vgather_reg(coeff, rr.vdups(channel, dtypes.uint32, mask=full)), mask=full)
                        b = rr.vmul(
                            r1, rr.vgather_reg(coeff, rr.vdups(COPIES + channel, dtypes.uint32, mask=full)), mask=full
                        )
                        c = rr.vmul(
                            r2,
                            rr.vgather_reg(coeff, rr.vdups(2 * COPIES + channel, dtypes.uint32, mask=full)),
                            mask=full,
                        )
                        d = rr.vmul(
                            r3,
                            rr.vgather_reg(coeff, rr.vdups(3 * COPIES + channel, dtypes.uint32, mask=full)),
                            mask=full,
                        )
                        mixed = rr.vadd(rr.vadd(a, c, mask=full), rr.vadd(b, d, mask=full), mask=full)
                        weighted = rr.vmul(
                            y, rr.vgather_reg(post_coeff, rr.vdups(channel, dtypes.uint32, mask=full)), mask=full
                        )
                        result = rr.vadd(weighted, mixed, mask=full)
                        rr.vstore_pack(
                            self.output,
                            channel * self.tile + offset,
                            rr.vcast(result, dtypes.bfloat16, mask=full),
                            full,
                            pack_mode=rr.PackMode.B32_TO_B16,
                        )
            for channel in range(COPIES):
                mem_copy(
                    tile_slice(output, (1, 1, self.tile), (row, channel, column)).view(self.tile),
                    tile_slice(self.output, (self.tile,), (channel,)),
                )
            vec_sync_all()


class HCPost:
    def __init__(self, width=5120, blocks=64):
        if type(width) is not int or width <= 0 or width % TILE:
            raise ValueError("HC post prototype requires a positive 64-aligned width")
        self.width, self.blocks = width, blocks
        self.compiled = lru_cache(maxsize=1)(self._compile)

    @jit
    def run(self, sublayer, residual, post, combination, output):
        op = HCPostKernel(self.width)
        op[self.blocks](sublayer, residual, post, combination, output)

    def _compile(self):
        rows = Dim("rows", min=1)
        return self.run.compile(
            TensorSpec((rows, self.width), dtypes.bfloat16),
            TensorSpec((rows, COPIES, self.width), dtypes.bfloat16),
            TensorSpec((rows, COPIES), dtypes.float32),
            TensorSpec((rows, COPIES * COPIES), dtypes.float32),
            TensorSpec((rows, COPIES, self.width), dtypes.bfloat16),
        )

    def __call__(self, sublayer, residual, post, combination):
        if residual.ndim != 3 or residual.shape[1:] != (COPIES, self.width):
            raise ValueError("HC post residual shape mismatch")
        rows = residual.shape[0]
        if (
            sublayer.shape != (rows, self.width)
            or post.shape != (rows, COPIES)
            or combination.shape != (rows, COPIES, COPIES)
        ):
            raise ValueError("HC post sublayer/coefficient shape mismatch")
        if (
            residual.dtype != torch.bfloat16
            or sublayer.dtype != torch.bfloat16
            or post.dtype != torch.float32
            or combination.dtype != torch.float32
        ):
            raise ValueError("HC post requires BF16 states and FP32 coefficients")
        if residual.device.type not in ("npu", "privateuseone") or any(
            x.device != residual.device for x in (sublayer, post, combination)
        ):
            raise ValueError("HC post inputs must share an NPU")
        if torch.is_grad_enabled() and any(x.requires_grad for x in (sublayer, residual, post, combination)):
            raise ValueError("HC post prototype is inference-only")
        output = torch.empty_like(residual, memory_format=torch.contiguous_format)
        if residual.numel():
            self.compiled()(
                sublayer.detach().contiguous(),
                residual.detach().contiguous(),
                post.detach().contiguous(),
                combination.detach().contiguous().view(rows, COPIES * COPIES),
                output,
            )
        return output
