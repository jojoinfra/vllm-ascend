# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
from jinja2 import Environment


def render(messages, **kwargs):
    root = Path(__file__).resolve().parents[4]
    template = Environment().from_string((root / "examples/chat_templates/template_deepseek_v41.jinja").read_text())

    def reject(message):
        raise ValueError(message)

    return template.render(
        messages=messages,
        bos_token="<｜begin▁of▁sentence｜>",
        eos_token="<｜end▁of▁sentence｜>",
        add_generation_prompt=True,
        raise_exception=reject,
        **kwargs,
    )


def test_chat_image_blocks_match_official_delimiters():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "unused.png"}},
                {"type": "text", "text": "什么颜色？"},
            ],
        }
    ]
    assert render(messages, enable_thinking=False) == (
        "<｜begin▁of▁sentence｜><｜User｜><｜deepseek_image｜>\n\n什么颜色？<｜Assistant｜></think>"
    )


@pytest.mark.parametrize("effort,budget", [("low", 50), ("high", 75), ("max", 100), (17, 17)])
def test_thinking_uses_official_numeric_effort_prefix(effort, budget):
    assert render([{"role": "user", "content": "1+1？"}], reasoning_effort=effort) == (
        f"<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: {budget} "
        "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
        "<｜User｜>1+1？<｜Assistant｜><think>"
    )


def test_history_drops_old_thinking_and_merges_consecutive_user_messages():
    messages = [
        {"role": "user", "content": "甲"},
        {"role": "user", "content": "乙"},
        {"role": "assistant", "content": "丙", "reasoning_content": "旧思考"},
        {"role": "system", "content": "仅数字"},
    ]
    assert render(messages, enable_thinking=False) == (
        "<｜begin▁of▁sentence｜><｜User｜>甲\n\n乙<｜Assistant｜></think>丙<｜end▁of▁sentence｜>"
        "<｜System｜>仅数字<｜Assistant｜></think>"
    )


def test_invalid_effort_is_rejected_instead_of_silently_changing_prompt():
    with pytest.raises(ValueError, match="reasoning_effort"):
        render([{"role": "user", "content": "你好"}], reasoning_effort=101)
