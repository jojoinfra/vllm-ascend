# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register the official nested V4.1 config without modifying transformers.

Defaults follow deepseek-ai/DeepSeek-V4.1-Flash-1017/config.json. This only
registers configuration; the V4 model must not be used as a V4.1 model alias.
"""

from transformers import AutoConfig, PretrainedConfig
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config


class DeepseekV41TextConfig(DeepseekV4Config):
    model_type = "deepseek_v41_text"

    def __init__(self, **kwargs):
        """Preserve official text, cache-ownership and draft defaults across round trips."""
        defaults = {
            "vocab_size": 129280,
            "hidden_size": 5120,
            "moe_intermediate_size": 2304,
            "num_hidden_layers": 40,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 1280,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "hidden_act": "silu",
            "swiglu_limit": 10.0,
            "rms_norm_eps": 1e-20,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "initializer_range": 0.02,
            "use_cache": True,
            "tie_word_embeddings": False,
            # Official ParallelHead stores/project weights in FP32. Without
            # this native vLLM setting the logits processor casts back to BF16.
            "head_dtype": "float32",
            "max_position_embeddings": 1048576,
            "rope_theta": 10000,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 16,
                "beta_fast": 32,
                "beta_slow": 1,
                "original_max_position_embeddings": 65536,
            },
            "n_routed_experts": 384,
            "n_shared_experts": 1,
            "num_hash_layers": 0,
            "num_experts_per_tok": 6,
            "scoring_func": "sqrtsoftplus",
            "topk_method": "noaux_tc",
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.5,
            "sliding_window": 128,
            "compress_ratios": [
                0,
                0,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
            ],
            "compress_rope_theta": 160000,
            "kv_source_layer_ids": [2, 8, 14, 20],
            "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
            "index_n_heads": 32,
            "index_head_dim": 128,
            "index_topk": 512,
            "candidate_source_layer_id": 20,
            "candidate_topk_blocks": 2048,
            "candidate_block_size": 8,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-06,
            "engram_layer_ids": [1, 14],
            "engram_num_embeddings": [384006168, 384016682],
            "engram_max_ngram_size": 4,
            "engram_vocab_size": 16000000,
            "engram_n_heads": 8,
            "engram_head_dim": 256,
            "engram_pad_token_id": 2,
            "engram_compressed_vocab_size": 99092,
            "num_nextn_predict_layers": 3,
            "dspark_block_size": 5,
            "dspark_noise_token_id": 128799,
            "dspark_target_layer_ids": [37, 38, 39],
            "dspark_markov_rank": 256,
            "dspark_n_routed_experts": 128,
            "dspark_num_experts_per_tok": 3,
        }
        if "rope_parameters" in kwargs and "rope_scaling" not in kwargs:
            defaults.pop("rope_scaling")
        defaults.update(kwargs)
        super().__init__(**defaults)


class DeepseekV41VisionConfig(PretrainedConfig):
    model_type = "deepseek_v41_vision"

    def __init__(self, **kwargs):
        defaults = {
            "num_hidden_layers": 32,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "intermediate_size": 2816,
            "patch_size": 14,
            "rope_theta": 10000,
            "downsample_ratio": 3,
            "max_image_tokens": 1024,
            "min_pixels": 295936,
            "max_wh_ratio": None,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class DeepseekV41Config(PretrainedConfig):
    model_type = "deepseek_v41"
    sub_configs = {"text_config": DeepseekV41TextConfig, "vision_config": DeepseekV41VisionConfig}

    def __init__(self, text_config=None, vision_config=None, image_token_id=129264, **kwargs):
        self.text_config = (
            text_config
            if isinstance(text_config, DeepseekV41TextConfig)
            else DeepseekV41TextConfig(**(text_config or {}))
        )
        self.vision_config = (
            vision_config
            if isinstance(vision_config, DeepseekV41VisionConfig)
            else DeepseekV41VisionConfig(**(vision_config or {}))
        )
        self.image_token_id = image_token_id
        # The common speculative proposer reads the language model's config
        # through the image_token_index convention used by multimodal targets.
        self.text_config.image_token_index = image_token_id
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("head_dtype", self.text_config.head_dtype)
        super().__init__(**kwargs)


AutoConfig.register(DeepseekV41TextConfig.model_type, DeepseekV41TextConfig)
AutoConfig.register(DeepseekV41VisionConfig.model_type, DeepseekV41VisionConfig)
AutoConfig.register(DeepseekV41Config.model_type, DeepseekV41Config)
