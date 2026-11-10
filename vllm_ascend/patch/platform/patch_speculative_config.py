from contextlib import contextmanager
from copy import copy, deepcopy

from transformers import DeepseekV2Config, PretrainedConfig
from vllm.config.speculative import SpeculativeConfig

_orig_post_init = SpeculativeConfig.__post_init__
_orig_hf_config_override = SpeculativeConfig.hf_config_override

# Transformers 5.14 inherited a hidden_size % num_heads check from Llama in
# DeepseekV2Config. K3 MLA has independent projection/head dimensions (e.g.
# hidden_size=7168, num_heads=96), so that MHA constraint does not apply.
# strict stores unbound validators; patch that entry, not all config validation.
if hasattr(DeepseekV2Config, "__class_validators__"):
    _orig_validate_architecture = DeepseekV2Config.validate_architecture

    def _validate_dspark_architecture(config):
        if config.model_type != "k3_dspark":
            _orig_validate_architecture(config)

    DeepseekV2Config.__class_validators__ = [
        _validate_dspark_architecture if validator is _orig_validate_architecture else validator
        for validator in DeepseekV2Config.__class_validators__
    ]


def _normalize_legacy_qwen3_dspark_config(hf_config: PretrainedConfig) -> PretrainedConfig:
    if hf_config.model_type == "deepseek_v41":
        # Draft ModelConfig needs the text dimensions at the top level. Keep
        # the target's composite config and its expert counts unchanged.
        text = deepcopy(hf_config.get_text_config())
        text.architectures = ["DeepseekV41DSparkForCausalLM"]
        text.n_routed_experts = text.dspark_n_routed_experts
        text.num_experts_per_tok = text.dspark_num_experts_per_tok
        text.n_mtp_layers = text.num_nextn_predict_layers
        text.n_predict = text.dspark_block_size
        text.ptd_token_id = text.dspark_noise_token_id
        text.sample_from_anchor = True
        if hasattr(hf_config, "quantization_config"):
            text.quantization_config = deepcopy(hf_config.quantization_config)
        return text
    hf_config = _orig_hf_config_override(hf_config)
    architectures = hf_config.architectures or ()
    if hf_config.model_type == "qwen3" and "DSparkDraftModel" in architectures:
        dflash_config = hf_config.dflash_config
        hf_config.update(
            {
                "architectures": ["Qwen3DSparkModel"],
                "mask_token_id": dflash_config["mask_token_id"],
                "target_layer_ids": dflash_config["target_layer_ids"],
            }
        )
    return hf_config


@contextmanager
def _temporarily_disable_dspark_dcp(self: SpeculativeConfig):
    target_parallel_config = self.target_parallel_config
    if getattr(self, "method", None) != "dspark" or target_parallel_config.decode_context_parallel_size <= 1:
        yield
        return

    guard_parallel_config = copy(target_parallel_config)
    guard_parallel_config.decode_context_parallel_size = 1
    self.target_parallel_config = guard_parallel_config
    try:
        yield
    finally:
        self.target_parallel_config = target_parallel_config


def _dspark_post_init(self):
    # TODO: This block can be deleted after the upstream supports the overlay of mla dcp and dspark
    with _temporarily_disable_dspark_dcp(self):
        _orig_post_init(self)
    if self.use_dspark() and self.target_model_config.hf_config.model_type == "deepseek_v41":
        # vLLM 0.27.1 unconditionally selects the V4 architecture for native
        # DeepSeek DSpark. Restore the plugin architecture after that default.
        draft_hf = self.draft_model_config.hf_config
        draft_hf.model_type = "deepseek_v41_text"
        draft_hf.architectures = ["DeepseekV41DSparkForCausalLM"]
        self.update_arch_()
        if self.num_speculative_tokens != draft_hf.dspark_block_size:
            raise ValueError("V4.1 DSpark requires the checkpoint's exact dspark_block_size")
    if self.use_dspark():
        draft_model_config = getattr(self, "draft_model_config", None)
        draft_hf_config = getattr(draft_model_config, "hf_config", None)
        # deepseek v4 dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "dspark_noise_token_id", None)  # type: ignore
        # gqa backend dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "mask_token_id", None)  # type: ignore


SpeculativeConfig.hf_config_override = staticmethod(_normalize_legacy_qwen3_dspark_config)
SpeculativeConfig.__post_init__ = _dspark_post_init
