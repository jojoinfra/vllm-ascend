# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional CANNBot BF16 E4M3 QDQ for A5 attention activations.

Match attention_ops.py's 32-element groups, FP32 amax floor, upward-rounded
power-of-two scale, ties-to-even E4M3 conversion and final BF16 rounding.
"""

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
from cannbotdsl.tensor import tile_slice

GROUP_WIDTH = 32
VECTOR_WIDTH = 64
EXPONENT_MASK = 0x7F800000
MANTISSA_MASK = 0x007FFFFF
SIGN_MASK = 0x80000000
MIN_NORMAL_E4M3_EXPONENT_BITS = (127 - 6) << 23
E4M3_MANTISSA_EXPONENT_BITS = 3 << 23


@kernel
class FP8QuantDequantKernel:
    def __init__(self):
        self.input = Buffer(MemLoc.UB, (VECTOR_WIDTH,), dtypes.bfloat16)
        self.output = Buffer(MemLoc.UB, (VECTOR_WIDTH,), dtypes.bfloat16)

    def __call__(self, x: Tensor, output: Tensor):
        for group in range(get_block_idx(), x.shape[0], get_block_num()):
            mem_copy(tile_slice(self.input, (GROUP_WIDTH,), (0,)), x[group, None].view(GROUP_WIDTH))
            with vf(mode="raw"):
                mask = rr.update_mask(GROUP_WIDTH, elem_bits=32)[0]
                value = rr.vcast(
                    rr.vload_unpack(self.input, 0, unpack_mode=rr.UnpackMode.B16_TO_B32),
                    dtypes.float32,
                    mask=mask,
                )
                magnitude = rr.vabs(value, mask=mask)
                maximum = rr.vdup(rr.vreduce_max(magnitude, mask=mask), mask=mask)
                raw_scale = rr.vmuls(rr.vmaxs(maximum, 1e-4, mask=mask), 1.0 / 448.0, mask=mask)
                exponent_mask = rr.vdups(EXPONENT_MASK, dtypes.uint32, mask=mask)
                scale_bits = rr.vbitwise_and(
                    rr.vadds(rr.vreinterpret(raw_scale, dtypes.uint32), MANTISSA_MASK, mask=mask),
                    exponent_mask,
                    mask=mask,
                )
                scale = rr.vreinterpret(scale_bits, dtypes.float32)
                normalized = rr.vdiv(value, scale, mask=mask)
                absolute = rr.vmins(rr.vabs(normalized, mask=mask), 448.0, mask=mask)
                exponent = rr.vbitwise_and(rr.vreinterpret(absolute, dtypes.uint32), exponent_mask, mask=mask)
                quantum_bits = rr.vadds(
                    rr.vmaxs(rr.vreinterpret(exponent, dtypes.int32), MIN_NORMAL_E4M3_EXPONENT_BITS, mask=mask),
                    -E4M3_MANTISSA_EXPONENT_BITS,
                    mask=mask,
                )
                quantum = rr.vreinterpret(quantum_bits, dtypes.float32)
                units = rr.vcast(rr.vdiv(absolute, quantum, mask=mask), dtypes.int32, mask=mask)
                rounded = rr.vmul(rr.vcast(units, dtypes.float32, mask=mask), quantum, mask=mask)
                sign = rr.vbitwise_and(
                    rr.vreinterpret(value, dtypes.uint32), rr.vdups(SIGN_MASK, dtypes.uint32, mask=mask), mask=mask
                )
                rounded = rr.vreinterpret(
                    rr.vbitwise_or(rr.vreinterpret(rounded, dtypes.uint32), sign, mask=mask), dtypes.float32
                )
                result = rr.vmul(rounded, scale, mask=mask)
                # Preserve the reference bit-based scale rounding even for
                # NaN amax: exponent 256 wraps to negative zero. Nonzero
                # non-NaN values then return signed zero; zero and NaN inputs
                # remain NaN. Inf amax without NaNs produces all NaNs.
                source_bits = rr.vreinterpret(magnitude, dtypes.uint32)
                source_nan = rr.vgt(source_bits, exponent_mask, mask=mask)
                source_bad = rr.vge(source_bits, exponent_mask, mask=mask)
                zero = rr.vdups(0.0, dtypes.float32, mask=mask)
                one = rr.vdups(1.0, dtypes.float32, mask=mask)
                has_nan = rr.vdup(rr.vreduce_max(rr.vselect(one, zero, cond_mask=source_nan), mask=mask), mask=mask)
                has_bad = rr.vdup(rr.vreduce_max(rr.vselect(one, zero, cond_mask=source_bad), mask=mask), mask=mask)
                nan = rr.vreinterpret(rr.vdups(0x7FC00000, dtypes.uint32, mask=mask), dtypes.float32)
                nan_scale_result = rr.vselect(nan, rr.vreinterpret(sign, dtypes.float32), cond_mask=source_nan)
                nan_scale_result = rr.vselect(nan, nan_scale_result, cond_mask=rr.veq(magnitude, zero, mask=mask))
                result = rr.vselect(nan, result, cond_mask=rr.vne(has_bad, zero, mask=mask))
                result = rr.vselect(nan_scale_result, result, cond_mask=rr.vne(has_nan, zero, mask=mask))
                rr.vstore_pack(
                    self.output,
                    0,
                    rr.vcast(result, dtypes.bfloat16, mask=mask),
                    mask,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
            mem_copy(output[group, None].view(GROUP_WIDTH), tile_slice(self.output, (GROUP_WIDTH,), (0,)))


class FP8QuantDequant:
    def __init__(self, blocks=64):
        if type(blocks) is not int or blocks < 1:
            raise ValueError("blocks must be a positive integer")
        self.blocks = blocks
        self.compiled = lru_cache(maxsize=1)(self._compile)

    @jit
    def run(self, x, output):
        op = FP8QuantDequantKernel()
        op[self.blocks](x, output)

    def _compile(self):
        groups = Dim("groups", min=1)
        return self.run.compile(
            TensorSpec((groups, GROUP_WIDTH), dtypes.bfloat16),
            TensorSpec((groups, GROUP_WIDTH), dtypes.bfloat16),
        )

    def __call__(self, x, *, inplace=False):
        if x.ndim < 1 or x.shape[-1] % GROUP_WIDTH:
            raise ValueError("Last dimension must be divisible by 32")
        if x.dtype != torch.bfloat16 or x.device.type not in ("npu", "privateuseone"):
            raise ValueError("Experimental FP8 QDQ requires BF16 on NPU")
        if torch.is_grad_enabled() and x.requires_grad:
            raise ValueError("FP8 QDQ is inference-only")
        flat = x.detach().contiguous().view(-1, GROUP_WIDTH)
        output = torch.empty_like(flat)
        if flat.numel():
            self.compiled()(flat, output)
        result = output.view(x.shape)
        if inplace:
            x.copy_(result)
            return x
        return result
