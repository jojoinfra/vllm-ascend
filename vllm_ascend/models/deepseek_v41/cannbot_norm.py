# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unintegrated CANNBot BF16 RMS normalization prototype."""

import math
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

VECTOR_WIDTH = 64
MAX_DIMENSION = 16384


@kernel
class RMSNormKernel:
    def __init__(self, width: int, eps: float, weight_dtype):
        self.width, self.eps, self.weight_dtype = width, eps, weight_dtype
        self.input = Buffer(MemLoc.UB, (width,), dtypes.bfloat16)
        self.weight = Buffer(MemLoc.UB, (width,), weight_dtype)
        self.output = Buffer(MemLoc.UB, (width,), dtypes.bfloat16)

    def __call__(self, x: Tensor, weight: Tensor, output: Tensor):
        mem_copy(self.weight, weight)
        for row in range(get_block_idx(), x.shape[0], get_block_num()):
            mem_copy(self.input, x[row, None].view(self.width))
            with vf(mode="raw"):
                mask = rr.full_mask()
                total = rr.vdups(0.0, dtypes.float32, mask=mask)
                for offset in range(0, self.width, VECTOR_WIDTH):
                    value = rr.vcast(
                        rr.vload_unpack(self.input, offset, unpack_mode=rr.UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=mask,
                    )
                    total = rr.vadd(total, rr.vmul(value, value, mask=mask), mask=mask)
                mean = rr.vmuls(rr.vdup(rr.vreduce_sum(total, mask=mask), mask=mask), 1.0 / self.width, mask=mask)
                root = rr.vsqrt(rr.vadds(mean, self.eps, mask=mask), mask=mask)
                inverse = rr.vdiv(rr.vdups(1.0, dtypes.float32, mask=mask), root, mask=mask)
                for offset in range(0, self.width, VECTOR_WIDTH):
                    value = rr.vcast(
                        rr.vload_unpack(self.input, offset, unpack_mode=rr.UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=mask,
                    )
                    scale = rr.vdups(0.0, dtypes.float32, mask=mask)
                    if self.weight_dtype == dtypes.bfloat16:
                        scale = rr.vcast(
                            rr.vload_unpack(self.weight, offset, unpack_mode=rr.UnpackMode.B16_TO_B32),
                            dtypes.float32,
                            mask=mask,
                        )
                    else:
                        scale = rr.vload(self.weight, offset)
                    result = rr.vmul(scale, rr.vmul(value, inverse, mask=mask), mask=mask)
                    rr.vstore_pack(
                        self.output,
                        offset,
                        rr.vcast(result, dtypes.bfloat16, mask=mask),
                        mask,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )
            mem_copy(output[row, None].view(self.width), self.output)


class RMSNorm:
    def __init__(self, width, eps=1e-20, weight_dtype=torch.bfloat16, blocks=64):
        if width <= 0 or width % VECTOR_WIDTH or width > MAX_DIMENSION:
            raise ValueError("RMS prototype requires a positive 64-aligned width up to 16384")
        if weight_dtype not in (torch.bfloat16, torch.float32) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("RMS prototype requires BF16/FP32 weights and positive finite epsilon")
        self.width, self.eps, self.weight_dtype, self.blocks = width, eps, weight_dtype, blocks
        self.dsl_weight_dtype = dtypes.bfloat16 if weight_dtype == torch.bfloat16 else dtypes.float32
        self.compiled = lru_cache(maxsize=1)(self._compile)

    @jit
    def run(self, x, weight, output):
        op = RMSNormKernel(self.width, self.eps, self.dsl_weight_dtype)
        op[self.blocks](x, weight, output)

    def _compile(self):
        rows = Dim("rows", min=1)
        return self.run.compile(
            TensorSpec((rows, self.width), dtypes.bfloat16),
            TensorSpec((self.width,), self.dsl_weight_dtype),
            TensorSpec((rows, self.width), dtypes.bfloat16),
        )

    def __call__(self, x, weight):
        if x.ndim < 1 or x.shape[-1] != self.width or weight.shape != (self.width,):
            raise ValueError("RMS input/weight shape mismatch")
        if x.dtype != torch.bfloat16 or weight.dtype != self.weight_dtype or weight.device != x.device:
            raise ValueError("RMS input/weight dtype or device mismatch")
        if x.device.type not in ("npu", "privateuseone"):
            raise ValueError("RMS prototype requires NPU tensors")
        if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
            raise ValueError("RMS prototype is inference-only")
        flat = x.detach().contiguous().view(-1, self.width)
        output = torch.empty_like(flat)
        if flat.numel():
            self.compiled()(flat, weight.detach().contiguous(), output)
        return output.view(x.shape)
