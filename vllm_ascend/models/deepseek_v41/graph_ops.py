"""Tensor-only V4.1 attention over vLLM physical pages. Based on the frozen MIT HF inference model."""

import torch

from vllm_ascend.models.deepseek_v41.backend import apply_rotary_emb, select_candidate_blocks
from vllm_ascend.models.deepseek_v41.eager_kernels import _fp4_level, act_quant, power_of_two_scale


def _rotate(x, freq, dim, inverse=False):
    apply_rotary_emb(x[..., -dim:].unsqueeze(0), freq, inverse)
    return x


def _fp4(x, lut, width=32, e4scale=False):
    groups = x.float().unflatten(-1, (-1, width))
    maximum = groups.abs().amax(-1, keepdim=True)
    if e4scale:
        scales = (maximum.clamp_min(6 * 2.0**-9) / 6.0).to(torch.float8_e4m3fn).float()
    else:
        scales = power_of_two_scale(maximum.clamp_min(6 * 2.0**-126), 6.0)
    codes = _fp4_level((groups / scales).clamp(-6.0, 6.0))
    x.copy_((lut[codes.long()] * scales).flatten(-2))


class LayerPages:
    @staticmethod
    def layout(attn):
        """Yield table name, tokens per stored row, row width and element type."""
        yield "window", 1, attn.head_dim, torch.bfloat16
        if attn.is_kv_source:
            yield "compressed", attn.compress_ratio, attn.head_dim, torch.bfloat16
            yield "index", attn.compress_ratio, attn.indexer.index_head_dim, torch.bfloat16
            if attn.compress_ratio == 2:
                yield "first_kv", 2, attn.head_dim, torch.float32
                yield "first_score", 2, attn.head_dim, torch.float32

    def __init__(self, attn, inputs, *, physical_slots=None):
        count = inputs.history.num_slots if physical_slots is None else physical_slots
        for name, ratio, width, dtype in self.layout(attn):
            rows = count // ratio + inputs.max_tokens
            setattr(self, name, torch.zeros((rows, width), device=inputs.device, dtype=dtype))

    @classmethod
    def bytes_per_block(cls, attn, block_size):
        return sum(block_size // ratio * width * dtype.itemsize for _, ratio, width, dtype in cls.layout(attn))


def _store_source(attn, x, state, pages):
    count = x.shape[0]
    ratio = attn.compress_ratio
    compressor = attn.compressor
    if ratio == 1:
        latent = compressor.norm(compressor.wkv(x))
    elif ratio == 2:
        value = compressor.wkv(x.float())
        score = compressor.wgate(x.float())
        first = state.active[:count] & (state.pos[:count] % ratio == 0)
        group = state.slots[:count] // ratio
        scratch = state.serial[:count] + state.history.num_slots // ratio
        first_addr = torch.where(first, group, scratch)
        pages.first_kv.index_copy_(0, first_addr, value)
        pages.first_score.index_copy_(0, first_addr, score)
        prior_addr = torch.where(state.active[:count], group, scratch)
        pair = torch.stack((pages.first_kv.index_select(0, prior_addr), value), dim=1)
        scores = torch.stack((pages.first_score.index_select(0, prior_addr), score), dim=1)
        latent = compressor.norm((pair * scores.softmax(1)).sum(1).to(x.dtype))
    else:
        raise ValueError("Expected checkpoint compression ratio one or two")
    dest = state.write_addresses(count, ratio)
    pos = (state.pos[:count] // ratio) * ratio
    freq = attn.freqs_cis.index_select(0, pos)
    idx = attn.indexer
    key = _rotate(idx.k_norm(idx.wk(latent)), freq, idx.rope_head_dim)
    _fp4(key, state.fp4_lut)
    pages.index.index_copy_(0, dest, key)
    _rotate(latent, freq, attn.rope_head_dim)
    _fp4(latent, state.fp4_lut, 16, True)
    pages.compressed.index_copy_(0, dest, latent)


def _select(attn, x, qr, freq, state, source, context, candidate):
    count = x.shape[0]
    idx = attn.indexer
    ratio = attn.compress_ratio
    width = max(1, context // ratio)
    sequence = torch.arange(width, device=x.device)
    positions = state.pos[:count]
    lengths = (positions[:, None] + 1) // ratio
    reachable = sequence < lengths
    # For short contexts the sorted top-k is exactly the reachable prefix,
    # and the source's block budget includes every visible block.
    if width <= idx.index_topk and width <= idx.candidate_topk_blocks * idx.candidate_block_size:
        logical = sequence.expand(count, -1)
        return torch.where(reachable, logical, -1), (reachable if idx.is_candidate_source else candidate)
    q = idx.wq_b(qr).unflatten(-1, (idx.n_local_heads, idx.index_head_dim))
    _rotate(q, freq, idx.rope_head_dim)
    _fp4(q, state.fp4_lut)
    weight = idx.weights_proj(x) * (idx.softmax_scale * idx.n_heads**-0.5)
    selected, masks = [], []
    for start in range(0, count, 32):
        rows = slice(start, min(start + 32, count))
        last_token = ((sequence + 1) * ratio - 1).expand(q[rows].size(0), -1)
        coords = state.physical(last_token, rows, ratio)
        keys = source.index[coords]
        score = torch.bmm(q[rows], keys.transpose(1, 2))
        score = (score.relu_() * weight[rows].unsqueeze(-1)).sum(1)
        score.masked_fill_(~reachable[rows], -torch.inf)
        if idx.is_candidate_source:
            if width > idx.candidate_topk_blocks * idx.candidate_block_size:
                keep = select_candidate_blocks(
                    score, lengths[rows], idx.candidate_topk_blocks, idx.candidate_block_size
                )
            else:
                keep = reachable[rows]
            masks.append(keep)
        elif idx.uses_candidates:
            score.masked_fill_(~candidate[rows], -torch.inf)
        indices = score.topk(min(width, idx.index_topk), dim=-1, sorted=False).indices
        indices = indices.sort(dim=-1).values
        selected.append(torch.where(indices < lengths[rows], indices, -1))
    return torch.cat(selected), torch.cat(masks) if idx.is_candidate_source else candidate


def _sparse(q, window, compressed, win_ids, comp_ids, attn):
    """HF 64-key online softmax, including the BF16 exponent rounding and sink logit."""
    count, heads, dim = q.shape
    result = torch.empty_like(q)
    for start in range(0, count, 32):
        rows = slice(start, min(start + 32, count))
        w = win_ids[rows]
        masks = [w >= 0]
        values = [window[w.clamp_min(0)]]
        if comp_ids is not None:
            c = comp_ids[rows]
            masks.append(c >= 0)
            values.append(compressed[c.clamp_min(0)])
        valid = torch.cat(masks, dim=1) if len(masks) > 1 else masks[0]
        v = torch.cat(values, dim=1) if len(values) > 1 else values[0]
        v = v.float().masked_fill(~valid.unsqueeze(-1), 0.0)
        qq = q[rows].float()
        scores = torch.bmm(qq, v.transpose(1, 2)) * attn.softmax_scale
        scores.masked_fill_(~valid.unsqueeze(1), -torch.inf)
        maximum = qq.new_full((qq.size(0), heads, 1), -1e30)
        denom = torch.zeros_like(maximum)
        numer = torch.zeros_like(qq)
        for begin in range(0, scores.size(-1), 64):
            tile = scores[:, :, begin : begin + 64]
            new_max = torch.maximum(maximum, tile.amax(-1, keepdim=True))
            prev = (maximum - new_max).exp()
            prob = (tile - new_max).exp()
            numer = numer * prev + torch.bmm(prob.bfloat16().float(), v[:, begin : begin + 64])
            denom = denom * prev + prob.sum(-1, keepdim=True)
            maximum = new_max
        denom += (attn.attn_sink.view(1, heads, 1) - maximum).exp()
        result[rows] = numer / denom
    return result


def paged_attention(attn, x, state, pages, shared, context):
    """A single token-major pass, including mixed prefill, greedy decode and verified draft tokens."""
    count = x.size(1)
    x = x[0]
    pos = state.pos[:count]
    freq = attn.freqs_cis.index_select(0, pos)
    qr = attn.q_norm(attn.wq_a(x))
    q = attn.wq_b(qr).unflatten(-1, (attn.n_local_heads, attn.head_dim))
    _rotate(q, freq, attn.rope_head_dim)
    kv = _rotate(attn.kv_norm(attn.wkv(x)), freq, attn.rope_head_dim)
    act_quant(kv, 32, inplace=True)
    pages.window.index_copy_(0, state.slots[:count], kv)
    width = min(context, attn.window_size)
    win_pos = (pos[:, None] - attn.window_size + 1).clamp_min(0) + torch.arange(width, device=x.device)
    active = state.active[:count, None]
    win_valid = (win_pos <= pos[:, None]) & active
    win_ids = torch.where(win_valid, state.physical(win_pos.clamp_max(state.history.max_len - 1), slice(0, count)), -1)
    comp_ids = compressed = None
    if attn.compress_ratio:
        ratio = attn.compress_ratio
        if attn.is_kv_source:
            _store_source(attn, x, state, pages)
            shared["source"] = pages
        source = shared["source"]
        if attn.is_index_source:
            shared["selected"], shared["candidate"] = _select(
                attn, x, qr, freq, state, source, context, shared.get("candidate")
            )
        indices = shared["selected"]
        last = ((indices.clamp_min(0) + 1) * ratio - 1).clamp_max(state.history.max_len - 1)
        comp_ids = torch.where((indices >= 0) & active, state.physical(last, slice(0, count), ratio), -1)
        compressed = source.compressed
    out = _sparse(q, pages.window, compressed, win_ids, comp_ids, attn)
    _rotate(out, freq, attn.rope_head_dim, True)
    out = out.reshape(count, attn.n_local_groups, -1)
    act_quant(out, 32, inplace=True)
    wo_a = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
    out = torch.einsum("tgd,grd->tgr", out, wo_a)
    return attn.wo_b(out.flatten(1)).unsqueeze(0)
