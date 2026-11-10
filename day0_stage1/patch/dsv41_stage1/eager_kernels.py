"""Stage-one eager math for the frozen DeepSeek HF kernels (inference/kernel.py).

Keep checkpoint matrices compact; A5 uses MXFP8 and MXFP8 × MXFP4 ND matmul.
Activations, including the two different cache formats, are rounded before use.
"""
import torch
from torch.nn import functional as F


def power_of_two_scale(amax, max_value):
    # Compute ceil(log2(amax / max_value)) without log2 rounding at bin edges.
    scaled = amax * (1. / max_value)
    bits = scaled.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 255) - 127
    ceil = exponent + ((bits & 0x7fffff) != 0).int()
    return torch.exp2(ceil.float())


def act_quant(x, block_size=32, scale_fmt="ue8m0", scale_dtype=None, inplace=False):
    groups = x.float().unflatten(-1, (-1, block_size))
    maximum = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scales = power_of_two_scale(maximum, 448.) if scale_fmt is not None else maximum / 448.
    code = (groups / scales).clamp(-448., 448.).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((code.float() * scales).flatten(-2))
        return x
    return code.flatten(-2).contiguous(), scales.squeeze(-1)


def _fp4_level(x):
    # E2M1 has eight unsigned magnitudes. The ordered midpoint comparisons encode
    # round-to-nearest-even, including exact ties (0.25, .75, 1.25, ... 5.0).
    value = x.abs()
    code = ((value > .25).int() + (value >= .75).int() + (value > 1.25).int()
            + (value >= 1.75).int() + (value > 2.5).int()
            + (value >= 3.5).int() + (value > 5.).int())
    return code | (torch.signbit(x).int() << 3)


def _fp4_values(codes):
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                        -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=torch.float32, device=codes.device)
    return lut[codes.long()]


def fp4_act_quant(x, block_size=32, inplace=True, scale_dtype=torch.float8_e8m0fnu):
    groups = x.float().unflatten(-1, (-1, block_size))
    maximum = groups.abs().amax(dim=-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        scales = (maximum.clamp_min(6 * 2.**-9) / 6.).to(scale_dtype).float()
    else:
        scales = power_of_two_scale(maximum.clamp_min(6 * 2.**-126), 6.)
    codes = _fp4_level((groups / scales).clamp(-6, 6))
    if inplace:
        x.copy_((_fp4_values(codes) * scales).flatten(-2))
        return x
    codes = codes.flatten(-2)
    return (codes[..., ::2] | (codes[..., 1::2] << 4)).to(torch.uint8), scales.squeeze(-1)


def _scale_values(raw):
    # UE8M0 exponent byte 0 represents 2^-127, 255 represents NaN.
    exp = raw.view(torch.uint8).int()
    return torch.exp2((exp - 127).float()).masked_fill(exp == 255, torch.nan)


def _npu_mx_linear(q, sx, weight, scale, packed_fp4):
    """A5 ND MX GEMM. Pair adjacent scale bytes; transpose weights and their scale together."""
    import torch_npu

    k = q.size(-1)
    n = weight.size(0)
    # sx contains exact powers of two, so its IEEE exponent is the UE8M0 byte.
    sa = ((sx.contiguous().view(torch.int32) >> 23) & 255).byte().reshape(-1, k // 64, 2)
    if packed_fp4:
        sw = scale
        extra = {"x2_dtype": torch_npu.float4_e2m1fn_x2}
    else:
        sw = scale.repeat_interleave(32, dim=0)[:n]
        extra = {}
    sw = sw.reshape(n, k // 64, 2).transpose(0, 1)
    y = torch_npu.npu_quant_matmul(
        q.reshape(-1, k), weight.T, sw, pertoken_scale=sa,
        scale_dtype=torch_npu.float8_e8m0fnu, pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        output_dtype=torch.bfloat16, group_sizes=[1, 1, 32], **extra,
    )
    return y.reshape(*q.shape[:-1], n)


def compact_linear(x, weight, scale, *, packed_fp4):
    q, sx = act_quant(x)
    if x.device.type == "npu" and x.size(-1) % 64 == 0:
        return _npu_mx_linear(q, sx, weight, scale, packed_fp4)
    groups = q.float().unflatten(-1, (-1, 32))
    # The HF kernel reduces each 32-element dot product in FP32, applies the
    # corresponding scales, then accumulates over K blocks in order. Dequantizing
    # the entire matrix before a single GEMM changes BF16 results near a tie.
    output = x.new_empty(*x.shape[:-1], weight.size(0), dtype=torch.bfloat16)
    rows = max(32, (8 * 1024 * 1024 // x.size(-1) // 32) * 32)
    for start in range(0, weight.size(0), rows):
        end = min(start + rows, weight.size(0))
        if packed_fp4:
            w = weight[start:end].view(torch.uint8).int()
            codes = torch.stack((w & 15, w >> 4), dim=-1).flatten(-2)
            w = _fp4_values(codes)
            sw = _scale_values(scale[start:end])
        else:
            w = weight[start:end].view(torch.float8_e4m3fn).float()
            sw = _scale_values(scale[start // 32:(end + 31) // 32]).repeat_interleave(32, 0)[:end-start]
        w = w.unflatten(-1, (-1, 32))
        accum = x.new_zeros(*x.shape[:-1], end-start, dtype=torch.float32)
        for block in range(w.size(1)):
            dot = F.linear(groups[..., block, :], w[:, block, :])
            accum.add_(dot * sx[..., block, None] * sw[:, block])
        output[..., start:end] = accum
    return output


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    # Follow the reference's 64-key online softmax, including its BF16 rounding
    # of each tile's unnormalized exponentials before multiplying values.
    batch, seq, heads, dim = q.shape
    output = torch.empty_like(q)
    batches = torch.arange(batch, device=q.device).view(batch, 1, 1)
    for query in range(0, seq, 16):
        qq = q[:, query:query+16].float()
        shape = (*qq.shape[:-1], 1)
        maximum = qq.new_full(shape, -1e30)
        denom = qq.new_zeros(shape)
        numer = torch.zeros_like(qq)
        for begin in range(0, topk_idxs.size(-1), 64):
            ids = topk_idxs[:, query:query+16, begin:begin+64].long()
            live = ids >= 0
            values = kv[batches, ids.clamp_min(0)].float().masked_fill(~live.unsqueeze(-1), 0.)
            scores = torch.matmul(qq, values.transpose(-2, -1)) * softmax_scale
            scores.masked_fill_(~live.unsqueeze(-2), -torch.inf)
            new_max = torch.maximum(maximum, scores.amax(dim=-1, keepdim=True))
            previous = torch.exp(maximum - new_max)
            probability = torch.exp(scores - new_max)
            numer = numer * previous + torch.matmul(probability.bfloat16().float(), values)
            denom = denom * previous + probability.sum(dim=-1, keepdim=True)
            maximum = new_max
        denom += (attn_sink.view(1, 1, heads, 1) - maximum).exp()
        output[:, query:query+16] = numer / denom
    return output


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    h = hc_mult
    pre = torch.sigmoid(mixes[..., :h] * hc_scale[0] + hc_base[:h]) + eps
    post = 2 * torch.sigmoid(mixes[..., h:2*h] * hc_scale[1] + hc_base[h:2*h])
    comb = (mixes[..., 2*h:] * hc_scale[2] + hc_base[2*h:]).unflatten(-1, (h, h))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
