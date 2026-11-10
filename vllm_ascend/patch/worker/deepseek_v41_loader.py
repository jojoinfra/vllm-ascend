# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexed loading that never materializes a full Engram table on any rank."""

import json
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import regex as re
from safetensors import safe_open
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

EXPERT_WEIGHT_PATTERN = re.compile(r"(?:^|\.)experts\.(\d+)\.")
ENGRAM_TABLE_PATTERN = re.compile(r"^layers\.(\d+)\.engram\.embed\.(weight|scale)$")


class DeepseekV41Checkpoint:
    def __init__(self, model_path):
        self.root = Path(model_path)
        index = self.root / "quant_model_weights.safetensors.index.json"
        if not index.is_file():
            index = self.root / "model.safetensors.index.json"
        self.weight_map = json.loads(index.read_text())["weight_map"]

    def iter_weights(self, *, local_expert_ids=None, include_prefixes=None):
        """Filter by indexed tensor name before opening or reading its data.

        Each physical file is opened once. Unindexed copies of weights are
        ignored even when they reside in the same shard. The caller explicitly
        chooses target/draft/vision prefixes; no architecture is silently lost.
        """
        shards = defaultdict(list)
        for name, filename in self.weight_map.items():
            if ENGRAM_TABLE_PATTERN.fullmatch(name):
                continue
            if include_prefixes is not None and not name.startswith(tuple(include_prefixes)):
                continue
            expert = EXPERT_WEIGHT_PATTERN.search(name)
            if expert is not None and local_expert_ids is not None and int(expert[1]) not in local_expert_ids:
                continue
            shards[filename].append(name)
        for filename, names in shards.items():
            with safe_open(self.root / filename, framework="pt", device="cpu") as handle:
                for name in names:
                    yield name, handle.get_tensor(name)

    def load_engram(self, layer_id, embedding):
        names = [f"layers.{layer_id}.engram.embed.{part}" for part in ("weight", "scale")]
        with ExitStack() as stack:
            handles = {
                filename: stack.enter_context(safe_open(self.root / filename, framework="pt", device="cpu"))
                for filename in {self.weight_map[name] for name in names}
            }
            slices = [handles[self.weight_map[name]].get_slice(name) for name in names]
            embedding.load_rows(*slices)


class DeepseekV41ModelLoader(DefaultModelLoader):
    """Explicit --load-format deepseek_v41 for the indexed W8A8 checkpoint.

    Keep DefaultModelLoader's quantization postprocessing and EP placement,
    replacing only weight iteration so giant tables never enter get_tensor.
    """

    def download_model(self, model_config):
        DeepseekV41Checkpoint(model_config.model)

    def get_all_weights(self, model_config, model):
        checkpoint = DeepseekV41Checkpoint(model_config.model)
        self.counter_before_loading_weights = time.perf_counter()
        model.load_checkpoint_tables(checkpoint)
        yield from checkpoint.iter_weights(
            local_expert_ids=self.local_expert_ids,
            include_prefixes=model.checkpoint_prefixes,
        )
