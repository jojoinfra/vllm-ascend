# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import torch

from vllm_ascend.models.deepseek_v41.decode_state import DecodeInputs, PagedHistory
from vllm_ascend.models.deepseek_v41.graph_ops import _raw_window, _sparse


def stage(positions, prompts, *, padding=0):
    size = 128
    history = PagedHistory(512, size, len(positions) * 4 + 1)
    history.tokens[:] = 0
    state = DecodeInputs(history, len(positions) + padding, "cpu", [])
    tables = np.arange(1, 1 + len(positions) * 4).reshape(-1, 4)
    slots = [int(table[pos // size] * size + pos % size) for table, pos in zip(tables, positions, strict=True)]
    data = SimpleNamespace(
        query_start_loc=torch.arange(len(positions) + 1),
        block_tables=tables,
        v41_prompt_lengths=np.array(prompts),
    )
    ctx = SimpleNamespace(attn_metadata={"target": data}, slot_mapping={"target": torch.tensor(slots + [-1] * padding)})
    plan = history.from_context(ctx, "target", torch.tensor(positions + [0] * padding))
    hash_state = SimpleNamespace(
        token_map=torch.arange(4),
        pad_id=0,
        layout=SimpleNamespace(max_ngram_size=2),
        multipliers=torch.tensor([[1, 3]]),
        primes=torch.tensor([[[7]]]),
        offsets=torch.tensor([[0]]),
    )
    context = state.stage(plan, [1] * len(plan.active), hash_state, [])
    return state, tables, context


def reference(q, keys, mask, sink, scale):
    maximum = torch.full((q.size(0), 1), -1e30)
    denom, numer = torch.zeros_like(maximum), torch.zeros_like(q, dtype=torch.float32)
    for tile, valid in zip(keys.split(64), mask.split(64), strict=True):
        scores = (q.float() @ tile.float().T) * scale
        scores.masked_fill_(~valid, -torch.inf)
        new_max = torch.maximum(maximum, scores.amax(-1, keepdim=True))
        prior = (maximum - new_max).exp()
        prob = (scores - new_max).exp()
        numer = numer * prior + prob.bfloat16().float() @ tile.float()
        denom = denom * prior + prob.sum(-1, keepdim=True)
        maximum = new_max
    return (numer / (denom + (sink[:, None] - maximum).exp())).bfloat16()


def test_mixed_rows_match_independent_prompt_and_decode_attention():
    poses = [14, 20, 15, 63, 64, 69, 95, 126, 127, 137, 201]
    prompts = [15, 26, 15, 16, 16, 16, 1000, 15, 15, 16, 15]
    state, tables, context = stage(poses, prompts, padding=1)
    count = len(poses) + 1
    raw, spans = _raw_window(state, count, context, 128)
    assert raw.shape == (count, 128)
    assert spans.flatten().tolist() == [min(n, 128) if p < n else 128 for p, n in zip(poses, prompts, strict=True)] + [
        128
    ]
    rng = torch.Generator().manual_seed(2015)

    def values(*shape):
        return torch.randn(shape, generator=rng).bfloat16()

    window, comp = values(state.history.num_slots, 32), values(state.history.num_slots // 2, 32)
    query = values(count, 3, 32)
    sink = torch.tensor([0.1, -0.7, 0.5])
    attn = SimpleNamespace(softmax_scale=32**-0.5, attn_sink=sink)
    comp_ids = torch.full((count, 128), -1)
    expected = []
    for i, (p, n, table) in enumerate(zip(poses, prompts, tables, strict=True)):

        def address(j, table=table):
            return int(table[j // 128] * 128 + j % 128)

        if p < n:
            begin = max(0, p - 127)
            raw_ids = [address(j) if j <= p else -1 for j in range(begin, begin + min(n, 128))]
        else:
            raw_ids = [address(j) if j >= 0 else -1 for j in range(p - 127, p + 1)]
        assert raw[i, : len(raw_ids)].tolist() == raw_ids
        assert (raw[i, len(raw_ids) :] == -1).all()
        cs = [address(j) // 2 for j in range(1, p + 1, 2)]
        comp_ids[i, : len(cs)] = torch.tensor(cs)
        w = torch.tensor(raw_ids)
        keys = torch.cat((window[w.clamp_min(0)], comp[cs]))
        valid = torch.cat((w >= 0, torch.ones(len(cs), dtype=torch.bool)))
        expected.append(reference(query[i], keys, valid, sink, attn.softmax_scale))
    out = _sparse(query, window, comp, raw, comp_ids, attn, spans)
    assert torch.equal(out[:-1], torch.stack(expected))
    assert torch.count_nonzero(out[-1]) == 0


def test_single_token_continuation_preserves_ring_and_short_prefill_stays_short():
    for p, n, w in ((14, 15, 15), (15, 15, 128), (95, 200, 128)):
        state, _, context = stage([p], [n])
        raw, span = _raw_window(state, 1, context, 128)
        assert raw.shape[1] == span.item() == w
        if p == 15:
            assert (raw[0, :112] == -1).all()
            assert raw[0, 112:].tolist() == list(range(128, 144))
