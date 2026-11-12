# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in in-place BF16 RoPE application using precomputed FP32 angles."""

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

ROPE_WIDTH = 64
PAIRS = ROPE_WIDTH // 2


@kernel
class RotaryApplyKernel:
    def __init__(self, pitch: int, heads: int, inverse: bool):
        self.pitch, self.heads, self.inverse = pitch, heads, inverse
        self.input = Buffer(MemLoc.UB, (ROPE_WIDTH,), dtypes.bfloat16)
        self.output = Buffer(MemLoc.UB, (ROPE_WIDTH,), dtypes.bfloat16)
        self.cosine = Buffer(MemLoc.UB, (ROPE_WIDTH,), dtypes.float32)
        self.sine = Buffer(MemLoc.UB, (ROPE_WIDTH,), dtypes.float32)

    def __call__(self, x: Tensor, cosine: Tensor, sine: Tensor):
        for row in range(get_block_idx(), x.shape[0], get_block_num()):
            tail = tile_slice(x, (1, ROPE_WIDTH), (row, self.pitch // ROPE_WIDTH - 1)).view(ROPE_WIDTH)
            mem_copy(self.input, tail)
            mem_copy(tile_slice(self.cosine, (PAIRS,), (0,)), cosine[row // self.heads, None].view(PAIRS))
            mem_copy(tile_slice(self.sine, (PAIRS,), (0,)), sine[row // self.heads, None].view(PAIRS))
            with vf(mode="raw"):
                full = rr.full_mask()
                half = rr.update_mask(PAIRS, elem_bits=32)[0]
                lanes = rr.varange(0, dtypes.uint32)
                pair = rr.vbitwise_and(lanes, rr.vdups(PAIRS - 1, dtypes.uint32, mask=full), mask=full)
                even_index = rr.vmuls(pair, 2, mask=full)
                value = rr.vcast(
                    rr.vload_unpack(self.input, 0, unpack_mode=rr.UnpackMode.B16_TO_B32),
                    dtypes.float32,
                    mask=full,
                )
                even = rr.vgather_reg(value, even_index)
                odd = rr.vgather_reg(value, rr.vadds(even_index, 1, mask=full))
                cosine_value = rr.vload(self.cosine, 0)
                sine_value = rr.vload(self.sine, 0)
                if self.inverse:
                    sine_value = rr.vmuls(sine_value, -1.0, mask=half)
                real = rr.vsub(rr.vmul(even, cosine_value, mask=half), rr.vmul(odd, sine_value, mask=half), mask=half)
                imaginary = rr.vadd(
                    rr.vmul(even, sine_value, mask=half), rr.vmul(odd, cosine_value, mask=half), mask=half
                )
                source = rr.vshr(lanes, 1, mask=full)
                real, imaginary = rr.vgather_reg(real, source), rr.vgather_reg(imaginary, source)
                parity = rr.vbitwise_and(lanes, rr.vdups(1, dtypes.uint32, mask=full), mask=full)
                result = rr.vselect(
                    real,
                    imaginary,
                    cond_mask=rr.veq(parity, rr.vdups(0, dtypes.uint32, mask=full), mask=full),
                )
                rr.vstore_pack(
                    self.output,
                    0,
                    rr.vcast(result, dtypes.bfloat16, mask=full),
                    full,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
            mem_copy(tail, self.output)
            vec_sync_all()


class RotaryApply:
    def __init__(self, pitch, heads, inverse=False, blocks=64):
        if pitch not in (128, 512) or type(heads) is not int or heads < 1:
            raise ValueError("RoPE prototype requires pitch128/512 and positive heads")
        self.pitch, self.heads, self.inverse, self.blocks = pitch, heads, inverse, blocks
        self.compiled = lru_cache(maxsize=1)(self._compile)

    @jit
    def run(self, x, cosine, sine):
        op = RotaryApplyKernel(self.pitch, self.heads, self.inverse)
        op[self.blocks](x, cosine, sine)

    def _compile(self):
        rows, tokens = Dim("rows", min=1), Dim("tokens", min=1)
        return self.run.compile(
            TensorSpec((rows, self.pitch), dtypes.bfloat16),
            TensorSpec((tokens, PAIRS), dtypes.float32),
            TensorSpec((tokens, PAIRS), dtypes.float32),
        )

    def __call__(self, x, cosine, sine):
        if x.ndim not in (3, 4) or x.shape[0] != 1 or x.shape[-1] != self.pitch or not x.is_contiguous():
            raise ValueError("RoPE prototype requires contiguous [1,tokens,(heads),pitch]")
        if (x.shape[2] if x.ndim == 4 else 1) != self.heads:
            raise ValueError("RoPE head count differs from the prepared kernel")
        if cosine.shape != (x.shape[1], PAIRS) or sine.shape != cosine.shape:
            raise ValueError("RoPE angles do not match input tokens")
        if x.dtype != torch.bfloat16 or cosine.dtype != torch.float32 or sine.dtype != torch.float32:
            raise ValueError("RoPE prototype requires BF16 input and FP32 angles")
        if x.device.type not in ("npu", "privateuseone") or cosine.device != x.device or sine.device != x.device:
            raise ValueError("RoPE input and angles must share one NPU")
        if torch.is_grad_enabled() and (x.requires_grad or cosine.requires_grad or sine.requires_grad):
            raise ValueError("RoPE prototype is inference-only")
        if x.numel():
            self.compiled()(x.detach().view(-1, self.pitch), cosine.detach().contiguous(), sine.detach().contiguous())
        return x
