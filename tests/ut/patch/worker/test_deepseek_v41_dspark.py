# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from threading import Barrier, Lock, local
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import CompilationMode, CUDAGraphMode

from vllm_ascend.patch.platform.patch_deepseek_v41_config import DeepseekV41Config
from vllm_ascend.patch.worker.deepseek_v41_attention_math import fake_quant_fp8
from vllm_ascend.patch.worker.deepseek_v41_cache import DeepseekV41CacheMetadata, write_paged
from vllm_ascend.patch.worker.deepseek_v41_dspark import (
    DeepseekV41ConfidenceHead,
    DeepseekV41DSparkAttention,
    DeepseekV41DSparkForCausalLM,
    DeepseekV41DSparkModel,
    DeepseekV41MarkovHead,
    draft_key_positions,
)
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


def test_common_proposer_resolves_image_token_from_target_language_config():
    target = DeepseekV41Config()
    image_token = AscendSpecDecodeBaseProposer._get_multimodal_image_token_index(
        "DeepseekV41ForCausalLM", target.get_text_config()
    )
    assert image_token == target.image_token_id


def test_draft_window_is_shared_by_all_queries_and_includes_future_drafts():
    positions = torch.tensor([2, 3, 4, 9, 10, 11])
    expected = torch.tensor([[0, 1, 2, 3, 4, -1, -1]] * 3 + [[8, 5, 6, 7, 9, 10, 11]] * 3)
    assert torch.equal(draft_key_positions(positions, 4, 3), expected)


class CacheStub(nn.Module):
    def __init__(self, cache, metadata):
        super().__init__()
        self.kv_cache, self.metadata = cache, metadata

    def forward(self):
        return self.kv_cache, self.metadata


class RotaryStub(nn.Module):
    def forward(self, hidden, positions, inverse=False):
        return hidden


@pytest.mark.parametrize("first_position", [2, 7])
def test_paged_draft_attention_matches_dense_noncausal_oracle(first_position):
    torch.manual_seed(41)
    block_size, width, draft_size, window = 4, 32, 3, 4
    attention = DeepseekV41DSparkAttention.__new__(DeepseekV41DSparkAttention)
    nn.Module.__init__(attention)
    attention.config = SimpleNamespace(head_dim=width, dspark_block_size=draft_size, sliding_window=window)
    attention.heads = 1
    attention.context_parallel = False
    attention.wq_a = attention.q_norm = attention.wq_b = attention.wkv = attention.kv_norm = nn.Identity()
    attention.rotary = RotaryStub()
    attention.attn_sink = nn.Parameter(torch.tensor([0.25]))
    # Two requests with noncontiguous physical pages and distinct contexts.
    block_table = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]], dtype=torch.int32)
    cache = torch.zeros(9, block_size, 1, width, dtype=torch.bfloat16)
    hidden = torch.randn(2, first_position + draft_size, width).bfloat16()
    context = fake_quant_fp8(hidden[:, :first_position].reshape(-1, width)).view(2, first_position, width)
    for request in range(2):
        offsets = torch.arange(first_position)
        slots = block_table[request, offsets // block_size] * block_size + offsets % block_size
        write_paged(cache, slots, context[request])
    positions = torch.arange(first_position, first_position + draft_size).repeat(2)
    requests = torch.arange(2).repeat_interleave(draft_size)
    slots = block_table[requests, positions // block_size] * block_size + positions % block_size
    metadata = DeepseekV41CacheMetadata(block_table, slots, requests, block_size, 1)
    attention.window_cache = CacheStub(cache, metadata)
    queries = hidden[:, first_position:].reshape(-1, width)
    with patch.object(attention, "project_output", side_effect=lambda output, positions: output.flatten(1)):
        actual = attention(positions, queries).view(2, draft_size, width)
        # Rewrite the rejected tail with a changed draft, retaining context.
        changed = (queries * -0.75).bfloat16()
        replay = attention(positions, changed).view_as(actual)
    for result, draft in [(actual, queries), (replay, changed)]:
        draft = draft.view(2, draft_size, width)
        for request in range(2):
            kv = torch.cat((context[request, max(0, first_position - window) :], fake_quant_fp8(draft[request])))
            scores = draft[request].float() @ kv.float().T / width**0.5
            maximum = scores.amax(-1, keepdim=True)
            weights = (scores - maximum).exp()
            expected = weights.bfloat16().float() @ kv.float()
            expected /= weights.sum(-1, keepdim=True) + (attention.attn_sink - maximum).exp()
            torch.testing.assert_close(result[request], expected.bfloat16(), rtol=0.01, atol=0.01)


def test_markov_and_confidence_heads_keep_fp32_accumulation():
    torch.manual_seed(11)
    config = SimpleNamespace(vocab_size=13, hidden_size=8, dspark_markov_rank=4)
    markov = DeepseekV41MarkovHead(config)
    confidence = DeepseekV41ConfidenceHead(config)
    embedding = markov.embed(torch.tensor([1, 7])).bfloat16()
    hidden = torch.randn(2, 8).bfloat16()
    bias = markov.bias(embedding)
    scores = confidence(hidden, embedding)
    assert bias.dtype == scores.dtype == torch.float32
    torch.testing.assert_close(bias, F.linear(embedding.float(), markov.head.weight))
    torch.testing.assert_close(
        scores, F.linear(torch.cat((hidden, embedding), -1).float(), confidence.proj.weight).squeeze(-1)
    )


@pytest.mark.parametrize("size", [4, 8])
@pytest.mark.parametrize("first_position", [2, 7])
def test_context_parallel_draft_preserves_blocks_and_rejected_tail(size, first_position):
    """Real CPU rank collectives vs dense noncausal attention for two requests.

    Five-token blocks straddle TP shards; padded rows must not read request0.
    Rewriting all draft tokens models a rejection without replacing context.
    """
    torch.manual_seed(19)
    width, block_size, window, draft_size, heads = 32, 4, 4, 5, 8
    block_table = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]], dtype=torch.int32)
    hidden = torch.randn(2, first_position + draft_size, width).bfloat16()
    context = fake_quant_fp8(hidden[:, :first_position].reshape(-1, width)).reshape(2, first_position, width)
    positions = torch.arange(first_position, first_position + draft_size).repeat(2)
    requests = torch.arange(2).repeat_interleave(draft_size)
    slots = block_table[requests, positions // block_size] * block_size + positions % block_size
    metadata = DeepseekV41CacheMetadata(block_table, slots, requests, block_size, 1)
    query_input = hidden[:, first_position:].reshape(-1, width)
    scales = torch.linspace(0.5, 1.5, heads).bfloat16()
    sinks = torch.linspace(-1, 1, heads)
    state = local()
    barrier, lock, exchanges = Barrier(size, timeout=30), Lock(), {}

    class Group:
        def __init__(self, rank):
            self.rank, self.step = rank, 0

        def all_gather(self, value, dim):
            step = self.step
            self.step += 1
            with lock:
                exchanges[step, self.rank] = value.clone()
            barrier.wait()
            return torch.cat([exchanges[step, rank] for rank in range(size)], dim=dim)

    class QueryProjection(nn.Module):
        def __init__(self, rank):
            super().__init__()
            self.scales = scales.chunk(size)[rank]

        def forward(self, value):
            return (value[:, None, :] * self.scales[None, :, None]).flatten(1)

    layers, groups = [], [Group(rank) for rank in range(size)]
    for rank in range(size):
        attention = DeepseekV41DSparkAttention.__new__(DeepseekV41DSparkAttention)
        nn.Module.__init__(attention)
        attention.config = SimpleNamespace(head_dim=width, dspark_block_size=draft_size, sliding_window=window)
        attention.heads, attention.tp_rank, attention.tp_size = heads // size, rank, size
        attention.context_parallel = True
        attention.wq_a = attention.q_norm = attention.wkv = attention.kv_norm = nn.Identity()
        attention.wq_b = QueryProjection(rank)
        attention.rotary = RotaryStub()
        attention.attn_sink = nn.Parameter(sinks.clone())
        cache = torch.zeros(9, block_size, 1, width, dtype=torch.bfloat16)
        for request in range(2):
            offsets = torch.arange(first_position)
            context_slots = block_table[request, offsets // block_size] * block_size + offsets % block_size
            write_paged(cache, context_slots, context[request])
        attention.window_cache = CacheStub(cache, metadata)
        attention.project_output = lambda output, positions: output
        layers.append(attention)

    def run_rank(rank):
        state.rank = rank
        attention = layers[rank]
        return [attention(positions, query_input), attention(positions, (query_input * -0.75).bfloat16())]

    with (
        patch(
            "vllm_ascend.patch.worker.deepseek_v41_dspark.get_tp_group",
            side_effect=lambda: groups[state.rank],
            create=True,
        ),
        ThreadPoolExecutor(max_workers=size) as executor,
    ):
        actual = list(executor.map(run_rank, range(size)))
    for attempt, draft in enumerate([query_input, (query_input * -0.75).bfloat16()]):
        combined = torch.cat([rank[attempt] for rank in actual], dim=1).reshape(2, draft_size, heads, width)
        draft = draft.reshape(2, draft_size, width)
        for request in range(2):
            kv = torch.cat((context[request, max(0, first_position - window) :], fake_quant_fp8(draft[request])))
            query = (draft[request, :, None] * scales[None, :, None]).float()
            logits = query @ kv.float().T / width**0.5
            maximum = logits.amax(-1, keepdim=True)
            probabilities = (logits - maximum).exp()
            expected = probabilities.bfloat16().float() @ kv.float()
            expected /= probabilities.sum(-1, keepdim=True) + (sinks[None, :, None] - maximum).exp()
            torch.testing.assert_close(combined[request], expected.bfloat16(), rtol=0.01, atol=0.01)


class DelayedLayer(nn.Module):
    def __init__(self, increment, next_mix):
        super().__init__()
        self.increment, self.next_mix = increment, next_mix

    def forward(self, positions, hidden, pre_mix):
        self.seen_mix = pre_mix.clone()
        # Deliberately asymmetric copies so a wrong final collapse is visible.
        return hidden + self.increment, self.next_mix.expand_as(pre_mix)


def test_draft_preserves_delayed_mix_between_stages_and_at_head():
    model = DeepseekV41DSparkModel.__new__(DeepseekV41DSparkModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hc_mult=2)
    model.embed_tokens = nn.Embedding.from_pretrained(torch.ones(5, 3))
    model.mtp = nn.ModuleList(
        [
            DelayedLayer(torch.tensor([[[1.0], [2.0]]]), torch.tensor([[0.2, 0.8]])),
            DelayedLayer(torch.tensor([[[4.0], [8.0]]]), torch.tensor([[0.7, 0.3]])),
            DelayedLayer(torch.tensor([[[2.0], [5.0]]]), torch.tensor([[0.4, 0.6]])),
        ]
    )
    result = model(torch.tensor([1, 2]), torch.tensor([4, 5]))
    for layer, mix in zip(model.mtp, [[1.0, 0.0], [0.2, 0.8], [0.7, 0.3]], strict=True):
        torch.testing.assert_close(layer.seen_mix, torch.tensor([mix, mix]))
    torch.testing.assert_close(result, torch.full((2, 3), 8 * 0.4 + 16 * 0.6))


def test_draft_checkpoint_mapping_keeps_only_mtp_and_preserves_head_names():
    mapper = DeepseekV41DSparkForCausalLM.hf_to_vllm_mapper
    assert mapper._map_name("mtp.0.embed.weight") == "model.embed_tokens.weight"
    assert mapper._map_name("mtp.2.head.weight") == "lm_head.weight"
    assert mapper._map_name("mtp.2.markov_head.head.weight") == "model.mtp.2.markov_head.head.weight"
    assert mapper._map_name("mtp.0.ffn.experts.3.w1.weight") == "model.mtp.0.ffn.experts.3.gate_proj.weight"
    assert DeepseekV41DSparkForCausalLM.checkpoint_prefixes == ("mtp.",)


def test_loaded_draft_vocab_weights_survive_proposer_sharing_checks():
    draft = DeepseekV41DSparkForCausalLM.__new__(DeepseekV41DSparkForCausalLM)
    nn.Module.__init__(draft)
    draft.model = nn.Module()
    draft.model.embed_tokens = nn.Embedding(8, 4)
    draft.lm_head = nn.Linear(4, 8, bias=False)
    embed, head = draft.model.embed_tokens, draft.lm_head
    with patch.object(draft, "get_expert_mapping", return_value=[]):
        draft.load_weights(
            [("mtp.0.embed.weight", torch.full((8, 4), 2.0)), ("mtp.2.head.weight", torch.full((8, 4), 3.0))]
        )
    target = SimpleNamespace(model=SimpleNamespace(embed_tokens=nn.Embedding(8, 4)), lm_head=nn.Linear(4, 8))
    proposer = SimpleNamespace(
        method="dspark",
        model=draft,
        vllm_config=SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)),
    )
    with patch("vllm_ascend.spec_decode.llm_base_proposer.get_pp_group", return_value=SimpleNamespace(world_size=1)):
        AscendSpecDecodeBaseProposer._maybe_share_embeddings(proposer, target)
        AscendSpecDecodeBaseProposer._maybe_share_lm_head(proposer, target)
    assert draft.model.embed_tokens is embed
    assert draft.lm_head is head
    assert draft.has_own_embed_tokens and draft.has_own_lm_head
    torch.testing.assert_close(embed.weight, torch.full((8, 4), 2.0))
    torch.testing.assert_close(head.weight, torch.full((8, 4), 3.0))


class DraftCoreStub(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.Module()
        layer.ffn = nn.Module()
        layer.ffn.experts = nn.Identity()
        self.mtp = nn.ModuleList([layer])

    def forward(self, input_ids, positions):
        return (input_ids + positions).float()[:, None]


@pytest.mark.parametrize("mode", [CompilationMode.NONE, CompilationMode.VLLM_COMPILE])
def test_draft_constructor_initializes_inherited_compile_call_contract(mode):
    config = SimpleNamespace(vocab_size=8, hidden_size=4)
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=config)),
        compilation_config=SimpleNamespace(mode=mode),
    )
    module = "vllm_ascend.patch.worker.deepseek_v41_dspark."
    with ExitStack() as stack:
        stack.enter_context(patch(module + "DeepseekV41DSparkModel", return_value=DraftCoreStub()))
        stack.enter_context(patch(module + "ParallelLMHead", return_value=nn.Linear(4, 8)))
        stack.enter_context(patch(module + "LogitsProcessor", return_value=nn.Identity()))
        stack.enter_context(patch.object(DeepseekV41DSparkForCausalLM, "extract_moe_parameters"))
        draft = DeepseekV41DSparkForCausalLM(vllm_config=vllm_config)
    assert draft.do_not_compile
    with patch(module + "use_moe_communication", return_value=nullcontext()):
        result = draft(input_ids=torch.tensor([1, 2]), positions=torch.tensor([3, 4]))
    torch.testing.assert_close(result, torch.tensor([[4.0], [6.0]]))
