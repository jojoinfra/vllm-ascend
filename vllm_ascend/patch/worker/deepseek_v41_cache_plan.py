# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static cache ownership derived from the official V4.1 forward ordering.

Each layer has its own sliding window. Compressed KV and index keys belong
to the most recent KV source, while top-k positions belong to the most recent
index source. Keep these identities distinct when allocating paged caches.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionCacheOwner:
    layer_id: int
    compress_ratio: int
    kv_layer_id: int | None
    index_layer_id: int | None
    candidate_layer_id: int | None


def build_attention_cache_plan(config) -> tuple[AttentionCacheOwner, ...]:
    """Resolve backbone owners; draft layers use separate sliding windows.

    Resolve once at construction, never by mutable process-global state during
    forward. Reject a topology that would silently reuse another source's data.
    """
    num_layers = config.num_hidden_layers
    ratios = config.compress_ratios
    if len(ratios) < num_layers:
        raise ValueError("compress_ratios must cover every backbone layer")
    kv_sources = tuple(config.kv_source_layer_ids)
    index_sources = tuple(config.index_source_layer_ids)
    for name, sources in (("KV", kv_sources), ("index", index_sources)):
        if sources != tuple(sorted(set(sources))):
            raise ValueError(f"{name} source layers must be unique and ordered")
        if any(layer < 0 or layer >= num_layers for layer in sources):
            raise ValueError(f"{name} source layer is outside the backbone")
        if any(ratios[layer] <= 0 for layer in sources):
            raise ValueError(f"{name} source must have a positive compression ratio")
    if not set(kv_sources).issubset(index_sources):
        raise ValueError("Every KV source must also produce index keys")
    candidate_source = getattr(config, "candidate_source_layer_id", -1)
    if candidate_source >= 0:
        if candidate_source not in index_sources:
            raise ValueError("Candidate source must be an index source")
        if config.candidate_topk_blocks <= 0 or config.candidate_block_size <= 0:
            raise ValueError("Candidate selection requires positive block sizes and counts")

    kv_owner = index_owner = None
    plan = []
    for layer, ratio in enumerate(ratios[:num_layers]):
        if ratio not in (0, 1, 2):
            raise ValueError(f"Unsupported V4.1 compression ratio: {ratio}")
        if layer in kv_sources:
            kv_owner = layer
        if layer in index_sources:
            index_owner = layer
        if ratio:
            if kv_owner is None or index_owner is None:
                raise ValueError(f"Layer {layer} reads compressed attention before a source")
            if ratios[kv_owner] != ratio or ratios[index_owner] != ratio:
                raise ValueError(f"Layer {layer} changes compression ratio without a matching source")
            if candidate_source >= 0 and layer > candidate_source and ratios[candidate_source] != ratio:
                raise ValueError("Candidate positions and consumer must use the same compression ratio")
        plan.append(
            AttentionCacheOwner(
                layer,
                ratio,
                kv_owner if ratio else None,
                index_owner if ratio else None,
                # The candidate source's own top-k is unfiltered; its immediate
                # followers reuse that result. Filtering starts at the next
                # index source, not merely the next decoder layer.
                candidate_source if ratio and index_owner is not None and 0 <= candidate_source < index_owner else None,
            )
        )
    return tuple(plan)
