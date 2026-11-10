# SPDX-License-Identifier: Apache-2.0
"""V4.1 text and DSpark blocks backed by the registered vLLM Ascend MoE runner."""

import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from vllm.config import get_current_vllm_config, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter

from vllm_ascend.quantization.configs import AscendModelSlimConfig
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod, AscendLinearMethod
from vllm_ascend.quantization.methods import AscendW4A8MXFPDSDynamicFusedMoEMethod, AscendW8A8MXFP8DynamicLinearMethod

from .expert_weights import install_expert_weights, make_shared_expert
from .native_ops import hash_route, initialize_native_ops, replace_native_leaves


class PackedExpertConfig(AscendModelSlimConfig):
    """Select the existing DS E4M3 activation × packed E2M1 expert scheme."""

    def get_quant_method(self, layer, prefix, tid2eid=None):
        if hasattr(layer, "moe_config"):
            return AscendFusedMoEMethod(AscendW4A8MXFPDSDynamicFusedMoEMethod(), layer.moe_config)
        return AscendLinearMethod(AscendW8A8MXFP8DynamicLinearMethod())


class TextRouter(FusedMoERouter):
    def __init__(self, gate):
        super().__init__()
        self.gate = gate
        self.capture = None

    @property
    def routing_method_type(self):
        return RoutingMethodType.Custom

    def set_capture_fn(self, capture_fn):
        self.capture = capture_fn

    def _select_experts(self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
        weights, ids = hash_route(
            router_logits,
            self.gate.bias,
            top_k=self.gate.topk,
            temperature=self.gate.gate_temp,
            route_scale=self.gate.route_scale,
        )
        if topk_indices_dtype is not None:
            ids = ids.to(topk_indices_dtype)
        if self.capture is not None:
            self.capture(ids)
        return weights, ids

    def route(self, hidden_states, gating_output, topk, renormalize):
        return self._select_experts(hidden_states, gating_output)


class NativeMoE(nn.Module):
    def __init__(self, source, prefix):
        super().__init__()
        initialize_native_ops()
        # The platform installs its factory override at runtime. Import after
        # registration to obtain the same public factory as other vLLM models.
        from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory

        from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner, AscendRoutedExperts

        self.gate = source.gate
        if self.gate.score_func != "sqrtsoftplus" or not self.gate.norm_topk_prob:
            raise ValueError("Expected V4.1 sqrtsoftplus with norm_topk_prob")
        self.dim = source.dim
        first = source.experts[source.experts_start_idx]
        n = len(source.experts)
        router = TextRouter(self.gate)
        config = copy.copy(get_current_vllm_config())
        config.quant_config = PackedExpertConfig({})
        with set_current_vllm_config(config), torch.device(self.gate.weight.device):
            shared = make_shared_expert(source.shared_experts, config.quant_config, prefix + ".shared_experts")
            self.runner = FusedMoEFactory(
                num_experts=n,
                top_k=self.gate.topk,
                hidden_size=source.dim,
                intermediate_size=first.w1.weight.shape[0],
                params_dtype=torch.bfloat16,
                router=router,
                custom_routing_function=router.route,
                runner_cls=AscendMoERunner,
                routed_experts_cls=AscendRoutedExperts,
                scoring_func="sqrtsoftplus",
                renormalize=True,
                routed_scaling_factor=1.0,
                swiglu_limit=first.swiglu_limit,
                ckpt_names=("w1", "w2", "w3"),
                quant_config=config.quant_config,
                shared_experts=shared,
                prefix=prefix,
                router_logits_dtype=torch.float32,
            )
            if not isinstance(self.runner, AscendMoERunner):
                raise TypeError(f"Expected the installed AscendMoERunner, got {type(self.runner)}")
            layer = self.runner.routed_experts
            self.receipt = install_expert_weights(layer, source)
        from vllm_ascend.ascend_config import get_ascend_config
        from vllm_ascend.utils import enable_sp, shared_expert_dp_enabled

        self.receipt.update(
            {
                "layer": prefix,
                "runner": type(self.runner).__qualname__,
                "routed_experts": type(layer).__qualname__,
                "quant_method": type(layer.quant_method).__qualname__,
                "scheme": AscendW4A8MXFPDSDynamicFusedMoEMethod.__name__,
                "routed_experts_total": n,
                "top_k": self.gate.topk,
                "route_scale": self.gate.route_scale,
                "gate_temperature": self.gate.gate_temp,
                "swiglu_limit": first.swiglu_limit,
                "shared_expert_mode": self.runner._get_shared_expert_parallel_mode().name,
                "shared_expert_dp_requested": bool(get_ascend_config().enable_shared_expert_dp),
                "shared_expert_dp_enabled": bool(shared_expert_dp_enabled()),
                "sequence_parallel_enabled": bool(enable_sp(config)),
                "tp_size": self.runner.moe_config.moe_parallel_config.tp_size,
                "ep_size": self.runner.moe_config.moe_parallel_config.ep_size,
            }
        )

    def forward(self, x, image_mask=None):
        flat = x.reshape(-1, self.dim)
        logits = F.linear(flat.float(), self.gate.weight.float())
        return self.runner(flat, logits).reshape_as(x)


def install_native_layers(model, *, prefix, receipt_dir=None):
    """Call once after the strict HF weights loader and before graph capture."""
    blocks = model.mtp if prefix == "mtp" else model.layers
    receipts = []
    # Free each old stand-alone expert list as its packed runner becomes ready.
    for i, block in enumerate(blocks):
        path = f"{prefix}.{i}.ffn"
        if not isinstance(block.ffn, NativeMoE):
            block.ffn = NativeMoE(block.ffn, path)
        receipts.append(block.ffn.receipt)
    sites = replace_native_leaves(model)
    if receipt_dir is not None:
        directory = Path(receipt_dir)
        directory.mkdir(parents=True, exist_ok=True)
        rank = model.placement.rank
        with (directory / f"native-{prefix}-rank-{rank}.json").open("w") as f:
            json.dump({"layers": receipts, "sites": sites}, f)
    return receipts
