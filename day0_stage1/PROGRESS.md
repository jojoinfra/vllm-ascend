# DeepSeek V4.1 A5 进展

阶段二已暂停，当前进行仓内发行包的阶段一八卡验证。[执行顺序](PLAN.md) · [标准入口与重放命令](patch/REPLAY.md)

| 日期 | 进展 | 对应材料 |
| --- | --- | --- |
| 2026-11-10 | 独立原型完成 eager 服务的 HTTP 和逐层 CPU 对照。保留首次数值分歧。 | [冻结检查器](validator/compare_traces.py) |
| 2026-11-10 | 按 DSV4 的既有入口把 V4.1 纳入仓内模型注册，添加模型专用的原始权重加载器、随包聊天模板和本地 Engram 逻辑。移除独立 day0 安装包。 | [模型注册](../vllm_ascend/models/__init__.py) · [权重加载](../vllm_ascend/model_loader/deepseek_v41.py) |
| 2026-11-10 | 指定镜像的 CPU 检查已覆盖 HF 文本配置、分片与单文件、离线缓存、损坏输入、逐 rank 权重形状和 n-gram 语义。标准 auto、hf 与 safetensors 格式均选用 V4.1 专用加载器。 | [自动选择](../vllm_ascend/platform.py) · [加载校验](../vllm_ascend/models/deepseek_v41/weights.py) |
| 2026-11-10 | 整仓 wheel 正在指定 A5 镜像中构建。新容器的普通 vllm serve 和八卡复验结果待归档。 | [重放步骤](patch/REPLAY.md) |
