# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from vllm.multimodal.parse import MultiModalDataParser

from vllm_ascend.patch.worker.deepseek_v41_processor import (
    IMAGE_PLACEHOLDER,
    DeepseekV41DummyInputsBuilder,
    DeepseekV41MultiModalProcessor,
    DeepseekV41ProcessingInfo,
)


class TokenizerStub:
    def __call__(self, prompt, **kwargs):
        self.options = kwargs
        return {"input_ids": torch.tensor([[0, 129264, 129264, 1]])}


def processing_components():
    config = SimpleNamespace(
        image_token_id=129264,
        vision_config=SimpleNamespace(
            patch_size=2, downsample_ratio=2, max_image_tokens=32, min_pixels=4, max_wh_ratio=None
        ),
    )
    tokenizer = TokenizerStub()
    info = DeepseekV41ProcessingInfo(SimpleNamespace(get_hf_config=lambda: config, get_tokenizer=lambda: tokenizer))
    processor = DeepseekV41MultiModalProcessor.__new__(DeepseekV41MultiModalProcessor)
    processor.info = info
    return info, processor, tokenizer


def test_image_fields_keep_variable_patch_counts_and_cpu_grid():
    _, processor, tokenizer = processing_components()
    images = [Image.new("RGB", (8, 8), (255, 0, 0)), Image.new("RGB", (12, 4), (0, 255, 0))]
    outputs = processor._call_hf_processor(IMAGE_PLACEHOLDER * 2, {"images": images}, {}, {})
    assert [tuple(x.shape) for x in outputs["pixel_values"]] == [(16, 3, 2, 2), (12, 3, 2, 2)]
    assert outputs["image_grid_hw"].tolist() == [[4, 4], [2, 6]]
    assert all(x.dtype == torch.bfloat16 for x in outputs["pixel_values"])
    fields = processor._get_mm_fields_config(outputs, {})
    assert fields["image_grid_hw"].field.keep_on_cpu
    assert tokenizer.options == {"add_special_tokens": True, "return_tensors": "pt"}


def test_replacement_counts_include_newlines_start_and_end():
    _, processor, _ = processing_components()
    images = [Image.new("RGB", (8, 8)), Image.new("RGB", (12, 4))]
    items = MultiModalDataParser().parse_mm_data({"image": images})
    update = processor._get_prompt_updates(items, {}, {})[0]
    assert update.target == [129264]
    assert update.replacement(0) == [129264] * 8
    assert update.replacement(1) == [129264] * 6
    assert not processor._hf_processor_applies_updates("", items, {}, {})


def test_dummy_image_realizes_the_reported_maximum_span():
    info, _, _ = processing_components()
    dummy = DeepseekV41DummyInputsBuilder(info)
    images = dummy.get_dummy_mm_data(4096, {"image": 1}, {})["image"]
    assert len(images) == 1
    image = images[0]
    assert info.get_num_image_tokens(image_width=image.width, image_height=image.height) == 32
    assert info.get_mm_max_tokens_per_item(4096, {"image": 1}) == {"image": 32}
    assert dummy.get_dummy_text({"image": 2}) == IMAGE_PLACEHOLDER * 2


def test_processor_rejects_silent_preprocessing_overrides():
    _, processor, _ = processing_components()
    with pytest.raises(ValueError, match="checkpoint vision configuration"):
        processor._call_hf_processor("", {}, {"max_image_tokens": 10}, {})
