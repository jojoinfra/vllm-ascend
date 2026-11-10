# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from threading import Barrier, Lock, local
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_ascend.patch.worker.deepseek_v41_attention import (
    DeepseekV41Attention,
    DeepseekV41AttentionContext,
    partition_context_tokens,
    select_candidates,
    select_positions,
)
from vllm_ascend.patch.worker.deepseek_v41_attention_math import (
    DeepseekV41Rotary,
    ceil_power_of_two,
    fake_quant_fp4,
    fake_quant_fp8,
    round_e4m3,
    sparse_attention,
)
from vllm_ascend.patch.worker.deepseek_v41_cache import DeepseekV41CacheMetadataBuilder
from vllm_ascend.patch.worker.deepseek_v41_cache_plan import build_attention_cache_plan


def test_fp8_rounding_matches_native_cast_including_ties_and_subnormals():
    torch.manual_seed(41)
    values = torch.cat((torch.randn(16384) * 100, torch.arange(-1024, 1024) / 1024)).clamp(-448, 448)
    torch.testing.assert_close(round_e4m3(values), values.to(torch.float8_e4m3fn).float(), rtol=0, atol=0)
    positive = torch.tensor([2**-126, 0.125, 0.1250001, 1.0, 1.00001, 16.0])
    torch.testing.assert_close(ceil_power_of_two(positive), torch.exp2(positive.log2().ceil()), rtol=0, atol=0)


@pytest.mark.parametrize("e4m3_scale", [False, True])
def test_fp4_ties_round_to_even_and_zero_groups_remain_zero(e4m3_scale):
    # Include magnitude 6 so both scale encodings choose exactly 1.
    values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, -5.0])
    expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 6.0, -4.0])
    torch.testing.assert_close(fake_quant_fp4(values, group_size=16, e4m3_scale=e4m3_scale), expected, rtol=0, atol=0)
    assert torch.equal(fake_quant_fp4(torch.zeros(16), group_size=16, e4m3_scale=e4m3_scale), torch.zeros(16))


def test_fp8_group_scales_and_quantized_values_match_native_cpu_format():
    torch.manual_seed(9)
    values = torch.randn(11, 512).bfloat16()
    grouped = values.float().view(11, 16, 32)
    scale = torch.exp2((grouped.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448).log2().ceil())
    expected = ((grouped / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).view_as(values).bfloat16()
    torch.testing.assert_close(fake_quant_fp8(values), expected, rtol=0, atol=0)


def test_sparse_attention_mask_sink_and_probability_rounding():
    torch.manual_seed(16)
    q, kv = torch.randn(2, 3, 32).bfloat16(), torch.randn(2, 64, 32).bfloat16()
    valid = torch.ones(2, 64, dtype=torch.bool)
    valid[0, 9:] = False
    valid[1] = False
    sink = torch.tensor([-2.0, 0.0, 3.0])
    got = sparse_attention(q, kv, valid, sink, 32**-0.5)
    logits = (q[0].double() @ kv[0, :9].double().T) * 32**-0.5
    maximum = logits.amax(-1, keepdim=True)
    exp = (logits - maximum).exp()
    expected = (exp.bfloat16().double() @ kv[0, :9].double()) / (
        exp.sum(-1, keepdim=True) + (sink[:, None] - maximum).exp()
    )
    torch.testing.assert_close(got[0], expected.bfloat16(), rtol=0.01, atol=0.01)
    assert torch.equal(got[1], torch.zeros_like(got[1]))


def test_candidates_pin_partial_latest_block_without_filtering_source_topk():
    scores = torch.tensor([[100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 1.0, -torch.inf], [-torch.inf] * 8])
    visible = torch.tensor([7, 0])
    candidates = select_candidates(scores, visible, block_size=2, topk_blocks=2)
    assert candidates.tolist() == [[True, False, False, True], [False] * 4]
    assert select_positions(scores, 4)[0].tolist() == [0, 1, 2, 3]
    filtered = scores.masked_fill(~candidates.repeat_interleave(2, dim=1), -torch.inf)
    assert select_positions(filtered, 3)[0].tolist() == [0, 1, 6]
    assert select_positions(filtered, 3)[1].tolist() == [-1] * 3


class LocalLinear(nn.Linear):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__(input_size, output_size, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return F.linear(x.float(), self.weight.float()).to(x.dtype)


def small_config():
    return SimpleNamespace(
        hidden_size=32,
        num_attention_heads=2,
        head_dim=32,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        o_groups=2,
        o_lora_rank=8,
        rms_norm_eps=1e-20,
        sliding_window=4,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=2,
        candidate_block_size=2,
        candidate_topk_blocks=1,
        candidate_source_layer_id=3,
        num_hidden_layers=6,
        compress_ratios=[0, 2, 2, 1, 1, 1],
        kv_source_layer_ids=[1, 3],
        index_source_layer_ids=[1, 3, 5],
        rope_theta=10000,
        compress_rope_theta=160000,
        rope_parameters=dict(original_max_position_embeddings=8, beta_fast=32, beta_slow=1, factor=16),
    )


def test_paged_attention_chunking_and_cross_layer_reuse_match_prefill():
    torch.manual_seed(12)
    config = small_config()
    vconfig = SimpleNamespace(
        quant_config=None,
        cache_config=SimpleNamespace(block_size=4),
        model_config=SimpleNamespace(max_model_len=16),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    module_name = "vllm_ascend.patch.worker.deepseek_v41_attention."
    with ExitStack() as stack:
        stack.enter_context(patch(module_name + "get_tensor_model_parallel_world_size", return_value=1))
        stack.enter_context(patch(module_name + "get_tensor_model_parallel_rank", return_value=0))
        for name in ("ReplicatedLinear", "ColumnParallelLinear", "RowParallelLinear"):
            stack.enter_context(patch(module_name + name, LocalLinear))
        layers = [
            DeepseekV41Attention(
                vconfig,
                config,
                owner,
                f"layers.{owner.layer_id}.attn",
                DeepseekV41Rotary(config, 16, compressed=bool(owner.compress_ratio)),
            )
            for owner in build_attention_cache_plan(config)
        ]
    for layer in layers:
        for parameter in layer.parameters():
            parameter.data.normal_(0, 0.15)
    builders = {}
    for name, cache in vconfig.compilation_config.static_forward_context.items():
        cache.bind_kv_cache(
            torch.zeros(5, cache.spec.storage_block_size, 1, cache.spec.head_size, dtype=cache.spec.dtype)
        )
        builders[name] = DeepseekV41CacheMetadataBuilder(cache.spec, [name], vconfig, torch.device("cpu"))
    inputs = torch.randn(6, 9, 32).bfloat16()

    def execute(parts):
        outputs = [[] for _ in layers]
        for begin, end in parts:
            metadata = {}
            positions = torch.arange(begin, end)
            for name, builder in builders.items():
                metadata[name] = builder.build(
                    0,
                    SimpleNamespace(
                        num_input_tokens=end - begin,
                        num_actual_tokens=end - begin,
                        max_seq_len=end,
                        slot_mapping=positions + 4,
                        block_table_tensor=torch.tensor([[1, 2, 3, 4]]),
                        query_start_loc=torch.tensor([0, end - begin], dtype=torch.int32),
                    ),
                )
            context = DeepseekV41AttentionContext()
            with patch(
                "vllm_ascend.patch.worker.deepseek_v41_cache.get_forward_context",
                return_value=SimpleNamespace(attn_metadata=metadata),
            ):
                for index, layer in enumerate(layers):
                    outputs[index].append(layer(positions, inputs[index, begin:end], context))
            assert context.kv_owner == 3 and context.index_owner == 5
        return [torch.cat(parts) for parts in outputs]

    full = execute([(0, 9)])
    for cache in vconfig.compilation_config.static_forward_context.values():
        cache.kv_cache.zero_()
    chunks = execute([(0, 3), (3, 6), (6, 9)])
    for expected, actual in zip(full, chunks):
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
    assert not hasattr(layers[2], "compressor")
    assert not hasattr(layers[2], "indexer")
    assert not hasattr(layers[5].indexer, "wk")


def test_capture_metadata_covers_future_decode_positions():
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4), model_config=SimpleNamespace(max_model_len=153600)
    )
    spec = SimpleNamespace(block_size=128, compress_ratio=2)
    builder = DeepseekV41CacheMetadataBuilder(spec, ["cache"], config, "cpu")
    common = SimpleNamespace(
        num_input_tokens=1,
        num_actual_tokens=1,
        max_seq_len=1,
        slot_mapping=torch.tensor([128]),
        block_table_tensor=torch.tensor([[1]]),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
    )
    assert builder.build(0, common).max_seq_len == 1
    assert builder.build_for_cudagraph_capture(common).max_seq_len == 153600


@pytest.mark.parametrize("size", [4, 8])
@pytest.mark.parametrize("tokens", [1, 7, 16])
@pytest.mark.parametrize("trailing", [(), (3,), (2, 3)])
def test_context_partition_preserves_axes_and_masks_padding(size, tokens, trailing):
    width = 1
    for dimension in trailing:
        width *= dimension
    value = torch.arange(tokens * width).reshape(tokens, *trailing)
    shards = [partition_context_tokens(value, rank, size, padding_value=-1) for rank in range(size)]
    assert all(shard.shape == ((tokens + size - 1) // size, *trailing) for shard in shards)
    restored = torch.cat(shards)
    torch.testing.assert_close(restored[:tokens], value, rtol=0, atol=0)
    assert (restored[tokens:] == -1).all()


@pytest.mark.parametrize("size", [4, 8])
@pytest.mark.parametrize("tokens", [1, 7, 16])
def test_context_parallel_paged_queries_match_full_head_reference(size, tokens):
    """Use real CPU collectives across rank threads, including idle shards.

    The full-head reference has no partition/gather calls. Each rank owns
    independent paged caches and context; sparse-attention outputs must match
    before the existing TP output projection. Cover ratio0/1/2, candidate
    reuse, shared keys/index and two requests with independent positions.
    """
    torch.manual_seed(52)
    config = small_config()
    config.num_attention_heads = config.index_n_heads = config.o_groups = 8
    module_name = "vllm_ascend.patch.worker.deepseek_v41_attention."
    state = local()
    barrier, lock, exchanges = Barrier(size, timeout=30), Lock(), {}

    class Group:
        def __init__(self, rank):
            self.rank, self.step = rank, 0

        def all_gather(self, value, dim):
            step = self.step
            self.step += 1
            with lock:
                exchanges[step, self.rank] = (dim, value.detach().clone())
            barrier.wait()
            parts = [exchanges[step, rank] for rank in range(size)]
            assert all(part[0] == dim for part in parts)
            return torch.cat([part[1] for part in parts], dim=dim)

    def make_layers(rank, world):
        vconfig = SimpleNamespace(
            quant_config=None,
            cache_config=SimpleNamespace(block_size=4),
            model_config=SimpleNamespace(max_model_len=16),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
            compilation_config=SimpleNamespace(static_forward_context={}),
        )

        class Column(LocalLinear):
            def __init__(self, input_size, output_size, **kwargs):
                super().__init__(input_size, output_size // world)

        class Row(LocalLinear):
            def __init__(self, input_size, output_size, **kwargs):
                super().__init__(input_size // world, output_size)

        with ExitStack() as stack:
            stack.enter_context(patch(module_name + "get_tensor_model_parallel_world_size", return_value=world))
            stack.enter_context(patch(module_name + "get_tensor_model_parallel_rank", return_value=rank))
            stack.enter_context(patch(module_name + "enable_dsa_cp", return_value=True))
            for name, cls in (
                ("ReplicatedLinear", LocalLinear),
                ("ColumnParallelLinear", Column),
                ("RowParallelLinear", Row),
            ):
                stack.enter_context(patch(module_name + name, cls))
            layers = [
                DeepseekV41Attention(
                    vconfig,
                    config,
                    owner,
                    f"layers.{owner.layer_id}.attn",
                    DeepseekV41Rotary(config, 16, compressed=bool(owner.compress_ratio)),
                )
                for owner in build_attention_cache_plan(config)
            ]
        first = (tokens + 1) // 2
        positions = torch.cat((torch.arange(first), torch.arange(tokens - first)))
        slots = torch.cat((torch.arange(first) + 4, torch.arange(tokens - first) + 24))
        metadata = {}
        for name, cache in vconfig.compilation_config.static_forward_context.items():
            cache.bind_kv_cache(
                torch.zeros(11, cache.spec.storage_block_size, 1, cache.spec.head_size, dtype=cache.spec.dtype)
            )
            builder = DeepseekV41CacheMetadataBuilder(cache.spec, [name], vconfig, torch.device("cpu"))
            metadata[name] = builder.build(
                0,
                SimpleNamespace(
                    num_input_tokens=tokens,
                    num_actual_tokens=tokens,
                    max_seq_len=first,
                    slot_mapping=slots,
                    block_table_tensor=torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
                    query_start_loc=torch.tensor([0, first, tokens], dtype=torch.int32),
                ),
            )
        for layer in layers:
            layer.project_output = lambda output, positions: output
        return layers, metadata, positions

    reference, ref_metadata, positions = make_layers(0, 1)
    for layer in reference:
        for parameter in layer.parameters():
            parameter.data.normal_(0, 0.15)
    distributed = [make_layers(rank, size) for rank in range(size)]
    for rank, (layers, _, _) in enumerate(distributed):
        for source, target in zip(reference, layers):
            for name, parameter in target.named_parameters():
                full = dict(source.named_parameters())[name]
                if parameter.shape != full.shape:
                    axis = 1 if name == "wo_b.weight" else 0
                    full = full.chunk(size, dim=axis)[rank]
                parameter.data.copy_(full)
            assert target.attn_sink.shape == source.attn_sink.shape
            assert target.attn_sink.tp_row_start == 0
    hidden = torch.randn(len(reference), tokens, config.hidden_size).bfloat16()
    state.metadata = ref_metadata
    groups = [Group(rank) for rank in range(size)]

    def run_rank(rank):
        state.rank = rank
        layers, state.metadata, _ = distributed[rank]
        context = DeepseekV41AttentionContext()
        outputs = [layer(positions, hidden[index], context) for index, layer in enumerate(layers)]
        assert context.topk.shape[0] == (tokens + size - 1) // size
        return outputs

    with (
        patch(module_name + "get_tp_group", side_effect=lambda: groups[state.rank]),
        patch(
            "vllm_ascend.patch.worker.deepseek_v41_cache.get_forward_context",
            side_effect=lambda: SimpleNamespace(attn_metadata=state.metadata),
        ),
    ):
        context = DeepseekV41AttentionContext()
        expected = [layer(positions, hidden[index], context) for index, layer in enumerate(reference)]
        with ThreadPoolExecutor(max_workers=size) as executor:
            actual = list(executor.map(run_rank, range(size)))
    for layer_index, full in enumerate(expected):
        combined = torch.cat([rank[layer_index] for rank in actual], dim=1)
        torch.testing.assert_close(combined, full, rtol=0.02, atol=0.002)
