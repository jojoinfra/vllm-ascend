# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from torch import nn
from vllm.config import get_current_vllm_config
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

from vllm_ascend.model_loader.deepseek_v41 import DeepseekV41ModelLoader
from vllm_ascend.models.deepseek_v41.graph_model import DeepseekV41Stage2ForCausalLM, PagedACLGraphWrapper
from vllm_ascend.models.deepseek_v41.model import AscendDeepseekV41ForCausalLM


@pytest.mark.parametrize(
    ("sequences", "spec", "eager", "expected"),
    [
        (1, None, True, AscendDeepseekV41ForCausalLM),
        (4, None, True, DeepseekV41Stage2ForCausalLM),
        (1, object(), True, DeepseekV41Stage2ForCausalLM),
        (1, None, False, DeepseekV41Stage2ForCausalLM),
    ],
)
def test_registered_class_initializes_the_selected_type_once(sequences, spec, eager, expected):
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=sequences),
        speculative_config=spec,
        model_config=SimpleNamespace(enforce_eager=eager),
    )
    with (
        patch.object(AscendDeepseekV41ForCausalLM, "__init__", autospec=True, return_value=None) as eager_init,
        patch.object(DeepseekV41Stage2ForCausalLM, "__init__", autospec=True, return_value=None) as paged_init,
    ):
        model = AscendDeepseekV41ForCausalLM(vllm_config=config, prefix="target")
    assert type(model) is expected
    assert isinstance(model, AscendDeepseekV41ForCausalLM)
    if expected is DeepseekV41Stage2ForCausalLM:
        paged_init.assert_called_once_with(model, vllm_config=config, prefix="target")
        eager_init.assert_not_called()
        assert expected.ascend_full_graph_wrapper_cls is PagedACLGraphWrapper
    else:
        eager_init.assert_called_once_with(model, vllm_config=config, prefix="target")
        paged_init.assert_not_called()
        assert not hasattr(expected, "ascend_full_graph_wrapper_cls")


def test_finalization_follows_generic_weight_loading(monkeypatch, tmp_path):
    config, model_config = object(), object()
    monkeypatch.setenv("VLLM_ASCEND_DSV41_RESULTS_DIR", str(tmp_path))
    calls = []

    class Paged(nn.Module):
        def finalize_after_load(self, *, receipt_dir):
            calls.append((get_current_vllm_config(), receipt_dir))
            self.new_child = nn.Identity()

    loader = DeepseekV41ModelLoader(load_config=None)
    for model in (nn.Identity(), Paged()):
        with patch.object(BaseModelLoader, "load_model", return_value=model) as generic:
            assert loader.load_model(config, model_config, "target") is model
            generic.assert_called_once_with(config, model_config, "target")
    assert calls == [(config, str(tmp_path))]
    assert not model.new_child.training
