# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendSlidingTokenCacheSpec, AscendTokenCacheSpec
from vllm_ascend.patch.worker.deepseek_v41_cache import (
    DeepseekV41CacheBackend,
    DeepseekV41CacheMetadataBuilder,
    DeepseekV41TokenHistory,
    read_paged,
    write_paged,
)
from vllm_ascend.patch.worker.deepseek_v41_compressor import DeepseekV41Compressor


def make_config(block_size=4):
    return SimpleNamespace(
        speculative_config=None,
        cache_config=SimpleNamespace(block_size=block_size),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


@pytest.mark.parametrize("mixed_backend", [False, True])
def test_device_only_backend_replays_without_fia_parameter_update(mixed_backend):
    from vllm.config import CUDAGraphMode

    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    config = make_config()
    history = DeepseekV41TokenHistory(config, torch.arange(100), 4, "model.layers.0.history")
    groups = [KVCacheGroupSpec(layer_names=[history.cache.prefix], kv_cache_spec=history.cache.spec)]
    if mixed_backend:

        class TaskUpdateBackend(DeepseekV41CacheBackend):
            requires_graph_param_update = True

        other = DeepseekV41TokenHistory(config, torch.arange(100), 4, "model.layers.1.history")
        other.cache.get_attn_backend = lambda: TaskUpdateBackend
        groups.append(KVCacheGroupSpec(layer_names=[other.cache.prefix], kv_cache_spec=other.cache.spec))
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.vllm_config = config
    runner.device = torch.device("cpu")
    runner.attn_groups = []
    runner._check_and_update_cudagraph_mode = Mock()
    runner.calculate_reorder_batch_threshold = Mock()
    runner.initialize_attn_backend(KVCacheConfig(num_blocks=4, kv_cache_tensors=[], kv_cache_groups=groups))
    assert runner.requires_graph_param_update is mixed_backend
    runner.use_sparse = runner.use_compress = runner.enable_enpu = False
    runner.attn_backend = Mock()
    runner.update_stream = None
    runner.speculative_config = None
    output = torch.randn(1, 4)
    runner.model = Mock(return_value=output)
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL, capturing=False)
    with (
        patch("vllm_ascend.worker.model_runner_v1.get_forward_context", return_value=context),
        patch("vllm_ascend.worker.model_runner_v1.update_full_graph_params") as update,
    ):
        assert runner._model_forward(1, input_ids=torch.tensor([5]), positions=torch.tensor([9])) is output
    runner.model.assert_called_once()
    assert context.cudagraph_runtime_mode == CUDAGraphMode.FULL
    assert update.call_count == int(mixed_backend)


def make_metadata(builder, blocks, slots, query_starts):
    return builder.build(
        0,
        SimpleNamespace(
            num_input_tokens=len(slots),
            num_actual_tokens=query_starts[-1],
            slot_mapping=torch.tensor(slots),
            block_table_tensor=torch.tensor(blocks),
            query_start_loc=torch.tensor(query_starts, dtype=torch.int32),
        ),
    )


def test_strided_pages_null_block_and_compressed_addressing():
    storage = torch.full((5, 40), -999.0)
    cache = storage[:, :8].view(5, 2, 1, 4)
    write_paged(cache, torch.tensor([4, 5, -1, 0]), torch.arange(16.0).view(4, 4))
    got, valid = read_paged(
        cache,
        torch.tensor([[0, 1, 2, 3, 4, -1]]),
        torch.tensor([0]),
        torch.tensor([[2, 0]]),
        logical_block_size=4,
        compress_ratio=2,
    )
    torch.testing.assert_close(got[0, :4], torch.tensor([[0.0, 1.0, 2.0, 3.0]] * 2 + [[4.0, 5.0, 6.0, 7.0]] * 2))
    assert valid.tolist() == [[True, True, True, True, False, False]]
    assert torch.all(storage[:, 8:] == -999)


def test_metadata_buffer_stays_stable_across_request_order_and_padding():
    spec = AscendTokenCacheSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.int64)
    builder = DeepseekV41CacheMetadataBuilder(spec, ["history"], make_config(), "cpu")
    first = make_metadata(builder, [[1], [2]], [4, 5, 8, -1], [0, 2, 3])
    pointer = first.request_indices.data_ptr()
    assert first.request_indices.tolist() == [0, 0, 1, -1]
    second = make_metadata(builder, [[2], [1]], [8, 4, 5, -1], [0, 1, 3])
    assert second.request_indices.data_ptr() == pointer
    assert first.request_indices.tolist() == [0, 1, 1, -1]


def test_history_forward_uses_bound_cache_and_masks_image_padding_rejection():
    config = make_config()
    history = DeepseekV41TokenHistory(config, torch.arange(100), 4, "history")
    # Extra page bytes must never be flattened into token history.
    backing = torch.full((5, 8), -777, dtype=torch.int64)
    history.cache.bind_kv_cache(backing[:, :4].view(5, 4, 1, 1))
    builder = DeepseekV41CacheMetadataBuilder(history.cache.spec, ["history"], config, "cpu")

    def run(tokens, positions, slots, starts, blocks, image_mask=None):
        metadata = make_metadata(builder, blocks, slots, starts)
        with patch(
            "vllm_ascend.patch.worker.deepseek_v41_cache.get_forward_context",
            return_value=SimpleNamespace(attn_metadata={"history": metadata}),
        ):
            return history(torch.tensor(tokens), torch.tensor(positions), image_mask)

    run([1, 2, 3, 99], [0, 1, 2, 3], [4, 5, 6, 7], [0, 4], [[1, 2]])
    result = run([4, 5, 6, 7], [3, 4, 5, 0], [7, 8, 9, -1], [0, 3], [[1, 2]], torch.tensor([False, True, False, False]))
    assert result.tolist() == [[4, 3, 2, 1], [-1, 4, 3, 2], [6, -1, 4, 3], [-1] * 4]
    # Two suffixes share the same full prefix page, then diverge.
    result = run([10, 20], [4, 4], [8, 12], [0, 1, 2], [[1, 2], [1, 3]])
    assert result.tolist() == [[10, 4, 3, 2], [20, 4, 3, 2]]
    assert torch.all(backing[:, 4:] == -777)


@pytest.mark.parametrize("ratio", [1, 2])
def test_compressor_chunk_boundary_and_speculative_overwrite(ratio):
    torch.manual_seed(19)
    config = make_config()
    model_config = SimpleNamespace(hidden_size=6, head_dim=4, rms_norm_eps=1e-20)
    layer = DeepseekV41Compressor(config, model_config, ratio, "compressor")
    hidden = torch.randn(8, 6).bfloat16()
    if ratio == 1:
        got, complete = layer(hidden, torch.arange(8))
        projected = hidden.float() @ layer.wkv.weight.float().T
        expected = layer.norm(projected.to(hidden.dtype))
        torch.testing.assert_close(got, expected, rtol=0.02, atol=0.02)
        assert complete.all()
        return
    layer.state.bind_kv_cache(torch.zeros(5, 4, 1, 8))
    builder = DeepseekV41CacheMetadataBuilder(layer.state.spec, ["compressor.state"], config, "cpu")
    results = []
    for begin, end in [(0, 3), (3, 5), (5, 8)]:
        positions = torch.arange(begin, end)
        slots = [4 + pos if pos < 4 else 12 + pos % 4 for pos in range(begin, end)]
        metadata = make_metadata(builder, [[1, 3]], slots, [0, end - begin])
        with patch(
            "vllm_ascend.patch.worker.deepseek_v41_cache.get_forward_context",
            return_value=SimpleNamespace(attn_metadata={"compressor.state": metadata}),
        ):
            output, complete = layer(hidden[begin:end], positions)
        results.append(output[complete])
    # Independent grouping equation over the whole request, without cache.
    values = (hidden.float() @ layer.wkv.weight.T).view(4, 2, 4)
    scores = (hidden.float() @ layer.wgate.weight.T).view(4, 2, 4)
    expected = layer.norm((values * scores.softmax(1)).sum(1).bfloat16())
    torch.testing.assert_close(torch.cat(results), expected)
    # Reject positions 5..7, replacing 5 must retain accepted position 4.
    replacement = hidden[5:6] + 2
    metadata = make_metadata(builder, [[1, 3]], [13], [0, 1])
    with patch(
        "vllm_ascend.patch.worker.deepseek_v41_cache.get_forward_context",
        return_value=SimpleNamespace(attn_metadata={"compressor.state": metadata}),
    ):
        output, complete = layer(replacement, torch.tensor([5]))
    pair = torch.cat([hidden[4:5], replacement]).float()
    expected = layer.norm(((pair @ layer.wkv.weight.T) * (pair @ layer.wgate.weight.T).softmax(0)).sum(0).bfloat16())
    torch.testing.assert_close(output[0], expected)
    assert complete.tolist() == [True]


@pytest.mark.parametrize("sliding", [False, True])
def test_runner_allocation_preserves_single_tensor_page_stride(sliding):
    # Exercise the actual runner methods, with no NPU init or whole model.
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    spec_cls = AscendSlidingTokenCacheSpec if sliding else AscendTokenCacheSpec
    spec = spec_cls(
        block_size=4,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        compress_ratio=2,
        page_size_padded=64,
        **({"sliding_window": 4} if sliding else {}),
    )
    name = "model.state"
    config = KVCacheConfig(
        num_blocks=5,
        kv_cache_tensors=[KVCacheTensor(size=320, shared_by=[name])],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec)],
    )
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.vllm_config = SimpleNamespace(kv_transfer_config=None)
    runner.device = torch.device("cpu")
    runner.runner_only_attn_layers = set()
    runner._kv_cache_spec_attn_group_iterator = lambda: [
        SimpleNamespace(kv_cache_spec=spec, backend=DeepseekV41CacheBackend, layer_names=[name])
    ]
    raw = runner._allocate_kv_cache_tensors(config)
    cache = runner._reshape_kv_cache_tensors(config, raw)[name]
    assert cache.shape == (5, 2, 1, 2)
    assert cache.stride() == (16, 2, 2, 1)
    cache[2, 1, 0, 1] = 17
    assert raw[name].view(torch.float32)[35] == 17
    assert cache.untyped_storage().data_ptr() == raw[name].untyped_storage().data_ptr()


def test_full_specs_can_share_scheduler_block_ids_with_different_row_sizes():
    specs = {
        "history": AscendTokenCacheSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.int64),
        "latent": AscendTokenCacheSpec(
            block_size=4, num_kv_heads=1, head_size=8, dtype=torch.bfloat16, compress_ratio=2
        ),
    }
    merged = UniformTypeKVCacheSpecs.from_specs(specs)
    assert merged is not None
    assert merged.kv_cache_specs == specs


def test_real_cache_manager_prefix_hit_reuses_history_bytes():
    from vllm import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    config = make_config()
    layer = DeepseekV41TokenHistory(config, torch.arange(100), 4, "history")
    cache_config = KVCacheConfig(
        num_blocks=12,
        kv_cache_tensors=[KVCacheTensor(size=12 * 32, shared_by=["history"])],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["history"], kv_cache_spec=layer.cache.spec)],
    )
    manager = KVCacheManager(cache_config, max_model_len=32, scheduler_block_size=4, hash_block_size=4)
    assert manager.block_pool.null_block.block_id == 0
    hash_function = lambda value: hashlib.sha256(repr(value).encode()).digest()
    init_none_hash(hash_function)
    hasher = get_request_block_hasher(4, hash_function)
    layer.cache.bind_kv_cache(torch.zeros(12, 4, 1, 1, dtype=torch.int64))
    builder = DeepseekV41CacheMetadataBuilder(layer.cache.spec, ["history"], config, "cpu")

    def forward(tokens, positions, block_ids):
        slots = [block_ids[p // 4] * 4 + p % 4 for p in positions]
        metadata = make_metadata(builder, [block_ids], slots, [0, len(tokens)])
        with patch(
            "vllm_ascend.patch.worker.deepseek_v41_cache.get_forward_context",
            return_value=SimpleNamespace(attn_metadata={"history": metadata}),
        ):
            return layer(torch.tensor(tokens), torch.tensor(positions))

    first = Request("first", [1, 2, 3, 4, 5], SamplingParams(max_tokens=2), None, block_hasher=hasher)
    assert manager.allocate_slots(first, 5) is not None
    first_blocks = manager.get_blocks(first.request_id).get_block_ids()[0]
    forward(first.prompt_token_ids, list(range(5)), first_blocks)
    manager.cache_blocks(first, 5)
    manager.free(first)
    second = Request("second", [1, 2, 3, 4, 9], SamplingParams(max_tokens=2), None, block_hasher=hasher)
    computed, hit_tokens, _ = manager.get_computed_blocks(second)
    assert hit_tokens == 4
    assert (
        manager.allocate_slots(second, 1, num_new_computed_tokens=hit_tokens, new_computed_blocks=computed) is not None
    )
    second_blocks = manager.get_blocks(second.request_id).get_block_ids()[0]
    assert second_blocks[0] == first_blocks[0]
    assert forward([9], [4], second_blocks).tolist() == [[9, 4, 3, 2]]


def test_full_topology_groups_and_pool_budget_use_real_vllm_entrypoints():
    from vllm import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
        get_request_block_hasher,
        init_none_hash,
    )
    from vllm.v1.request import Request

    from vllm_ascend.core.token_cache import unwrap_group_specs
    from vllm_ascend.patch.worker.deepseek_v41_cache import DeepseekV41PagedCache

    config = make_config(128)
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.cache_config.num_gpu_blocks_override = None
    specs = {}

    def add(name, width, dtype, **kwargs):
        specs[name] = DeepseekV41PagedCache(config, name, width, dtype, **kwargs).spec

    add("history", 1, torch.int64)
    for layer in range(40):
        add(f"window.{layer}", 512, torch.bfloat16, sliding_window=128)
    for layer, ratio in ((2, 2), (8, 2), (14, 2), (20, 1)):
        add(f"latent.{layer}", 512, torch.bfloat16, compress_ratio=ratio)
        add(f"index.{layer}", 128, torch.bfloat16, compress_ratio=ratio)
        if ratio == 2:
            add(f"state.{layer}", 1024, torch.float32, sliding_window=2, block_size=4)
    groups = get_kv_cache_groups(config, specs)
    assert len(groups) == 44  # one full-cache group and 43 recyclable groups
    full_group = groups[0]
    assert set(full_group.layer_names) == {
        "history",
        "latent.2",
        "latent.8",
        "latent.14",
        "latent.20",
        "index.2",
        "index.8",
        "index.14",
        "index.20",
    }
    available_memory = 8 * 1024**3
    layout = get_kv_cache_config_from_groups(config, groups, available_memory)
    assert sum(tensor.size for tensor in layout.kv_cache_tensors) <= available_memory
    assert sum(tensor.size for tensor in layout.kv_cache_tensors) == layout.num_blocks * 410624
    assert len(layout.kv_cache_tensors) == 9
    assert all(spec.page_size_padded is None for spec in unwrap_group_specs(groups))
    # Full caches in one group cannot alias; windows may alias that group's
    # same-size pool only because their scheduler block IDs are independent.
    for tensor in layout.kv_cache_tensors:
        assert len(set(tensor.shared_by) & set(full_group.layer_names)) == 1
        assert len({specs[name].page_size_bytes for name in tensor.shared_by}) == 1
    scheduler_layout = generate_scheduler_kv_cache_config([layout])
    manager = KVCacheManager(
        scheduler_layout,
        max_model_len=153600,
        max_in_flight_tokens=8192,
        scheduler_block_size=128,
        hash_block_size=4,
    )
    assert manager.num_kv_cache_groups == 44
    hash_function = lambda value: hashlib.sha256(repr(value).encode()).digest()
    init_none_hash(hash_function)
    hasher = get_request_block_hasher(4, hash_function)
    first = Request("first", list(range(129)), SamplingParams(max_tokens=2), None, block_hasher=hasher)
    assert manager.allocate_slots(first, 129) is not None
    manager.cache_blocks(first, 129)
    manager.free(first)
    second = Request("second", list(range(128)) + [200], SamplingParams(max_tokens=2), None, block_hasher=hasher)
    computed, hit_tokens, _ = manager.get_computed_blocks(second)
    assert hit_tokens == 128
    assert (
        manager.allocate_slots(second, 1, num_new_computed_tokens=hit_tokens, new_computed_blocks=computed) is not None
    )
