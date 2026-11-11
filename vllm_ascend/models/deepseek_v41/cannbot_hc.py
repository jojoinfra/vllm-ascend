# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in experimental CANNBot HC4 gates and Sinkhorn (DSL 0.6.0).

Each vector core processes one token at a time. The 4x4 residual matrix stays
in registers for all iterations. XOR shuffles reduce rows/columns without GM
temporaries. The FP32 equations and epsilon placement follow core.py exactly.
"""

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
from cannbotdsl.tensor import tile_slice

HC_MULT = 4
MIX_WIDTH = 24
FP32_VECTOR_LANES = 64


@kernel
class HC4SinkhornKernel:
    def __init__(self, iterations: int, eps: float):
        self.iterations = iterations
        self.eps = eps
        self.mixes = Buffer(MemLoc.UB, (FP32_VECTOR_LANES,), dtypes.float32)
        self.base = Buffer(MemLoc.UB, (FP32_VECTOR_LANES,), dtypes.float32)
        self.scale = Buffer(MemLoc.UB, (FP32_VECTOR_LANES,), dtypes.float32)
        self.pre = Buffer(MemLoc.UB, (FP32_VECTOR_LANES,), dtypes.float32)
        self.post = Buffer(MemLoc.UB, (FP32_VECTOR_LANES,), dtypes.float32)
        self.comb = Buffer(MemLoc.UB, (FP32_VECTOR_LANES,), dtypes.float32)

    def __call__(self, mixes: Tensor, scale: Tensor, base: Tensor, pre: Tensor, post: Tensor, comb: Tensor):
        mem_copy(tile_slice(self.scale, (3,), (0,)), scale)
        mem_copy(tile_slice(self.base, (MIX_WIDTH,), (0,)), base)
        for token in range(get_block_idx(), mixes.shape[0], get_block_num()):
            mem_copy(tile_slice(self.mixes, (MIX_WIDTH,), (0,)), mixes[token, None].view(MIX_WIDTH))
            with vf(mode="raw"):
                m4 = rr.update_mask(4, elem_bits=32)[0]
                m16 = rr.update_mask(16, elem_bits=32)[0]
                full = rr.full_mask()
                lane = rr.varange(0, dtypes.uint32)
                x = rr.vload(self.mixes, 0)
                b = rr.vload(self.base, 0)
                s = rr.vload(self.scale, 0)
                s0 = rr.vgather_reg(s, rr.vdups(0, dtypes.uint32, mask=full))
                s1 = rr.vgather_reg(s, rr.vdups(1, dtypes.uint32, mask=full))
                s2 = rr.vgather_reg(s, rr.vdups(2, dtypes.uint32, mask=full))
                one = rr.vdups(1.0, dtypes.float32, mask=full)
                p = rr.vadd(rr.vmul(x, s0, mask=m4), b, mask=m4)
                p = rr.vdiv(one, rr.vadds(rr.vexp(rr.vmuls(p, -1.0, mask=m4), mask=m4), 1.0, mask=m4), mask=m4)
                rr.vstore(self.pre, 0, rr.vadds(p, self.eps, mask=m4), m4)
                post_idx = rr.vadds(lane, 4, mask=full)
                q = rr.vadd(rr.vmul(rr.vgather_reg(x, post_idx), s1, mask=m4), rr.vgather_reg(b, post_idx), mask=m4)
                q = rr.vdiv(one, rr.vadds(rr.vexp(rr.vmuls(q, -1.0, mask=m4), mask=m4), 1.0, mask=m4), mask=m4)
                rr.vstore(self.post, 0, rr.vmuls(q, 2.0, mask=m4), m4)
                comb_idx = rr.vadds(lane, 8, mask=full)
                c = rr.vadd(rr.vmul(rr.vgather_reg(x, comb_idx), s2, mask=m16), rr.vgather_reg(b, comb_idx), mask=m16)
                row1 = rr.vbitwise_xor(lane, rr.vdups(1, dtypes.uint32, mask=full), mask=full)
                row2 = rr.vbitwise_xor(lane, rr.vdups(2, dtypes.uint32, mask=full), mask=full)
                col1 = rr.vbitwise_xor(lane, rr.vdups(4, dtypes.uint32, mask=full), mask=full)
                col2 = rr.vbitwise_xor(lane, rr.vdups(8, dtypes.uint32, mask=full), mask=full)
                maximum = rr.vmax(c, rr.vgather_reg(c, row1), mask=m16)
                maximum = rr.vmax(maximum, rr.vgather_reg(maximum, row2), mask=m16)
                c = rr.vexp(rr.vsub(c, maximum, mask=m16), mask=m16)
                total = rr.vadd(c, rr.vgather_reg(c, row1), mask=m16)
                total = rr.vadd(total, rr.vgather_reg(total, row2), mask=m16)
                c = rr.vadds(rr.vdiv(c, total, mask=m16), self.eps, mask=m16)
                total = rr.vadd(c, rr.vgather_reg(c, col1), mask=m16)
                total = rr.vadd(total, rr.vgather_reg(total, col2), mask=m16)
                c = rr.vdiv(c, rr.vadds(total, self.eps, mask=m16), mask=m16)
                for _ in range(self.iterations - 1):
                    total = rr.vadd(c, rr.vgather_reg(c, row1), mask=m16)
                    total = rr.vadd(total, rr.vgather_reg(total, row2), mask=m16)
                    c = rr.vdiv(c, rr.vadds(total, self.eps, mask=m16), mask=m16)
                    total = rr.vadd(c, rr.vgather_reg(c, col1), mask=m16)
                    total = rr.vadd(total, rr.vgather_reg(total, col2), mask=m16)
                    c = rr.vdiv(c, rr.vadds(total, self.eps, mask=m16), mask=m16)
                rr.vstore(self.comb, 0, c, m16)
            mem_copy(pre[token, None].view(4), tile_slice(self.pre, (4,), (0,)))
            mem_copy(post[token, None].view(4), tile_slice(self.post, (4,), (0,)))
            mem_copy(comb[token, None].view(16), tile_slice(self.comb, (16,), (0,)))


class HC4Sinkhorn:
    """Own a bounded compilation cache; construct once outside graph capture."""

    def __init__(self, iterations=20, eps=1e-6, max_blocks=64):
        if type(iterations) is not int or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not math.isfinite(eps) or eps < 0:
            raise ValueError("eps must be finite and nonnegative")
        if type(max_blocks) is not int or max_blocks < 1:
            raise ValueError("max_blocks must be a positive integer")
        self.iterations = iterations
        self.eps = float(eps)
        self.max_blocks = max_blocks
        self.compiled = lru_cache(maxsize=1)(self._compile)

    @jit
    def run(self, mixes, scale, base, pre, post, comb):
        op = HC4SinkhornKernel(self.iterations, self.eps)
        op[self.max_blocks](mixes, scale, base, pre, post, comb)

    def _compile(self):
        # Prefill and packed decode lengths vary at runtime. A symbolic leading
        # dimension avoids compiling new programs in the request hot path.
        tokens = Dim("tokens", min=1)
        return self.run.compile(
            TensorSpec((tokens, MIX_WIDTH), dtypes.float32),
            TensorSpec((3,), dtypes.float32),
            TensorSpec((MIX_WIDTH,), dtypes.float32),
            TensorSpec((tokens, HC_MULT), dtypes.float32),
            TensorSpec((tokens, HC_MULT), dtypes.float32),
            TensorSpec((tokens, HC_MULT * HC_MULT), dtypes.float32),
        )

    def __call__(self, mixes, scale, base):
        if mixes.ndim < 2 or mixes.shape[-1] != MIX_WIDTH:
            raise ValueError("mixes must have shape [..., 24]")
        if scale.shape != (3,) or base.shape != (MIX_WIDTH,):
            raise ValueError("HC4 scale/base must have shapes [3]/[24]")
        for tensor in (mixes, scale, base):
            if tensor.dtype != torch.float32 or not tensor.is_contiguous():
                raise ValueError("HC4 requires contiguous FP32 tensors")
            if tensor.device != mixes.device or tensor.device.type not in ("npu", "privateuseone"):
                raise ValueError("HC4 inputs must be on the same NPU")
            # Check execution mode, not only the Parameter flag: checkpoint
            # parameters retain requires_grad=True under inference_mode().
            if torch.is_grad_enabled() and tensor.requires_grad:
                raise ValueError("HC4 is inference-only")
        lead = mixes.shape[:-1]
        flat = mixes.view(-1, MIX_WIDTH)
        tokens = flat.shape[0]
        pre = mixes.new_empty((tokens, HC_MULT))
        post = torch.empty_like(pre)
        comb = mixes.new_empty((tokens, HC_MULT * HC_MULT))
        if tokens:
            # DSL 0.6's persisted DLPack ABI requires exactly torch.Tensor,
            # not its Parameter subclass. detach is an alias, not a copy.
            self.compiled()(flat, scale.detach(), base.detach(), pre, post, comb)
        return pre.view(*lead, HC_MULT), post.view(*lead, HC_MULT), comb.view(*lead, HC_MULT, HC_MULT)
