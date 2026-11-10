from transformers import PretrainedConfig


class DeepseekV41TextConfig(PretrainedConfig):
    model_type = "deepseek_v41_text"

    def __init__(self, max_position_embeddings=1048576, rope_scaling=None, rope_parameters=None,
                 rope_theta=10000, **kwargs):
        self.max_position_embeddings = max_position_embeddings
        self.rope_parameters = dict(rope_parameters or rope_scaling or {"rope_type": "default"})
        self.rope_parameters.setdefault("rope_theta", rope_theta)
        super().__init__(**kwargs)


class DeepseekV41Config(PretrainedConfig):
    """Expose the frozen text topology to vLLM; the custom loader owns quantization."""

    model_type = "deepseek_v41"
    sub_configs = {"text_config": DeepseekV41TextConfig}

    def __init__(self, text_config=None, vision_config=None, quantization_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = DeepseekV41TextConfig(**text_config)
        self.text_config = text_config if text_config is not None else DeepseekV41TextConfig()
        # Preserve the disk contract while keeping the generic FP8 re-packer out
        # of this model. The loader retains packed FP4 and the shared host tables.
        self.checkpoint_quantization = quantization_config or kwargs.pop("checkpoint_quantization", None)
        self.checkpoint_vision_config = vision_config or kwargs.pop("checkpoint_vision_config", None)
        super().__init__(**kwargs)
