# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM image fields and placeholders for the official V4.1 image layout."""

import math

import torch
from transformers import BatchFeature
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import ImageProcessorItems, ImageSize
from vllm.multimodal.processing import BaseDummyInputsBuilder, BaseMultiModalProcessor, BaseProcessingInfo
from vllm.multimodal.processing.processor import PromptReplacement

from vllm_ascend.patch.worker.deepseek_v41_vision import image_grid, image_patches, image_span_size

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"


class DeepseekV41ProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self):
        return {"image": None}

    def get_num_image_tokens(self, *, image_width, image_height):
        config = self.get_hf_config().vision_config
        rows, columns = image_grid(image_width, image_height, config)
        return image_span_size(rows, columns, config.downsample_ratio)

    def get_image_size_with_most_features(self):
        config = self.get_hf_config().vision_config
        # Count start/end and a newline per row, including in profiling.
        best = (0, 0, 0, 0)
        for rows in range(1, (config.max_image_tokens - 2) // 2 + 1):
            columns = (config.max_image_tokens - 2) // rows - 1
            if config.max_wh_ratio is not None:
                columns = min(columns, math.floor(rows * config.max_wh_ratio))
            if columns > 0:
                best = max(best, (rows * (columns + 1) + 2, rows * columns, rows, columns))
        _, _, rows, columns = best
        if rows == 0:
            raise ValueError("Image token budget cannot fit a complete image span")
        cell = config.patch_size * config.downsample_ratio
        return ImageSize(width=columns * cell, height=rows * cell)

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        size = self.get_image_size_with_most_features()
        return {"image": self.get_num_image_tokens(image_width=size.width, image_height=size.height)}


class DeepseekV41DummyInputsBuilder(BaseDummyInputsBuilder[DeepseekV41ProcessingInfo]):
    def get_dummy_text(self, mm_counts):
        return IMAGE_PLACEHOLDER * mm_counts.get("image", 0)

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        size = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=size.width,
                height=size.height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            )
        }


class DeepseekV41MultiModalProcessor(BaseMultiModalProcessor[DeepseekV41ProcessingInfo]):
    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        if mm_kwargs:
            raise ValueError("V4.1 image preprocessing uses the checkpoint vision configuration")
        options = dict(tok_kwargs)
        options.setdefault("add_special_tokens", True)
        options["return_tensors"] = "pt"
        outputs = dict(self.info.get_tokenizer()(prompt, **options))
        images = mm_data.get("images", [])
        if images:
            config = self.info.get_hf_config().vision_config
            processed = [image_patches(image, config) for image in images]
            outputs["pixel_values"] = [item[0] for item in processed]
            outputs["image_grid_hw"] = torch.tensor([item[1:] for item in processed], dtype=torch.int64)
        return BatchFeature(outputs)

    def _hf_processor_applies_updates(self, prompt_text, mm_items, hf_processor_mm_kwargs, tokenization_kwargs):
        return False

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
            "image_grid_hw": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        }

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        image_token_id = self.info.get_hf_config().image_token_id

        def replacement(item_idx):
            images = mm_items.get_items("image", ImageProcessorItems)
            size = images.get_image_size(item_idx)
            count = self.info.get_num_image_tokens(image_width=size.width, image_height=size.height)
            return [image_token_id] * count

        return [PromptReplacement(modality="image", target=[image_token_id], replacement=replacement)]
