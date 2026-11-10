"""Load the stage-one text implementation in each vLLM process."""

def register():
    from transformers import AutoConfig
    from vllm import ModelRegistry
    from vllm.model_executor.model_loader import register_model_loader
    from .config import DeepseekV41Config
    from .loader import Stage1Loader

    AutoConfig.register("deepseek_v41", DeepseekV41Config, exist_ok=True)
    ModelRegistry.register_model(
        "DeepseekV41ForCausalLM", "dsv41_stage1.model:DeepseekV41ForCausalLM"
    )
    register_model_loader("dsv41_stage1")(Stage1Loader)
