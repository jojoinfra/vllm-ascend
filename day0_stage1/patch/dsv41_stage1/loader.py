import json
from pathlib import Path

from vllm.model_executor.model_loader.base_loader import BaseModelLoader

from .weights import load_checkpoint


class Stage1Loader(BaseModelLoader):
    def download_model(self, model_config):
        if not (Path(model_config.model) / "model.safetensors.index.json").is_file():
            raise ValueError("Stage 1 requires the mounted, indexed official checkpoint")

    def load_weights(self, model, model_config):
        import os
        import torch

        self.download_model(model_config)
        q = model_config.hf_config.checkpoint_quantization
        if not q or (q.get("quant_method"), q.get("activation_scheme"), q.get("weight_block_size"),
                     q.get("scale_fmt"), q.get("expert_dtype")) != ("fp8", "dynamic", [32, 32], "ue8m0", "fp4"):
            raise ValueError("Unexpected V4.1 quantization geometry")
        run = Path(os.environ["DSV41_RUN_RESULTS"])
        ep = model.backbone.placement
        device = model.backbone.device
        load_checkpoint(model.backbone, model_config.model, ep_rank=ep.rank, ep_size=ep.size,
                        device=device, manifest=run / f"rank-{ep.rank}-load.json")
        props = dict(ep_rank=ep.rank, ep_size=ep.size, dp_rank=model.dp_rank,
                     tp_size=1, pp_size=1, pcp_size=1, device=torch.npu.current_device(),
                     cpu_affinity=sorted(os.sched_getaffinity(0)), model_layers=len(model.backbone.layers),
                     npu_name=torch.npu.get_device_name(device), expert_range=model.expert_range,
                     worker_pid=os.getpid(), memory_free_total=torch.npu.mem_get_info(device),
                     peak_torch_allocated=torch.npu.max_memory_allocated(device))
        with (run / f"rank-{ep.rank}-placement.json").open("x") as f:
            json.dump(props, f)
