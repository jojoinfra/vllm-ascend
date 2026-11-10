# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Group single-tensor token caches without padding tiny history pages.

Full caches with the same logical block size share scheduler block IDs.
Sliding caches each use their own group, so recycling a window does not
retain a full-context allocation. Physical pools are shared across groups
only when their page byte sizes match; scheduler IDs are globally unique.
"""

from collections import defaultdict

from vllm.v1.core.kv_cache_utils import may_override_num_blocks
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendSlidingTokenCacheSpec, AscendTokenCacheSpec


def is_token_cache_layout(specs):
    specs = tuple(specs)
    return bool(specs) and all(isinstance(spec, (AscendTokenCacheSpec, AscendSlidingTokenCacheSpec)) for spec in specs)


def group_token_caches(specs):
    full_by_block_size = defaultdict(dict)
    sliding = []
    for name, spec in specs.items():
        if isinstance(spec, AscendTokenCacheSpec):
            full_by_block_size[spec.block_size][name] = spec
        else:
            sliding.append({name: spec})
    grouped = [full_by_block_size[size] for size in sorted(full_by_block_size)] + sliding
    result = [UniformTypeKVCacheSpecs.from_specs(group) for group in grouped]
    if any(group is None for group in result):
        raise ValueError("Token cache group contains incompatible scheduler block ownership")
    return result


def token_cache_config(vllm_config, groups, available_memory):
    # Pool [page_bytes][layer_index] can serve one layer from each independent
    # group. Layers within one group must have distinct physical buffers.
    pools = defaultdict(list)
    for group in groups:
        spec = group.kv_cache_spec
        per_layer = (
            spec.kv_cache_specs
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else {name: spec for name in group.layer_names}
        )
        used = defaultdict(int)
        for name in group.layer_names:
            page_bytes = per_layer[name].page_size_bytes
            index = used[page_bytes]
            if index == len(pools[page_bytes]):
                pools[page_bytes].append([])
            pools[page_bytes][index].append(name)
            used[page_bytes] += 1
    bytes_per_block = sum(size * len(entries) for size, entries in pools.items())
    num_blocks = may_override_num_blocks(vllm_config, available_memory // bytes_per_block)
    tensors = [
        KVCacheTensor(size=size * num_blocks, shared_by=names) for size, entries in pools.items() for names in entries
    ]
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=tensors, kv_cache_groups=groups)


def unwrap_group_specs(groups):
    return [
        spec
        for group in groups
        for spec in (
            group.kv_cache_spec.kv_cache_specs.values()
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            else (group.kv_cache_spec,)
        )
    ]


def token_cache_groups(grouped_specs):
    return [KVCacheGroupSpec(layer_names=list(spec.kv_cache_specs), kv_cache_spec=spec) for spec in grouped_specs]
