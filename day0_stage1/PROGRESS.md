# DeepSeek V4.1 A5 进展

仓内发行包的阶段一八卡复验已完成。阶段二沿用本包和普通服务入口。[执行顺序](PLAN.md) · [标准入口与重放命令](patch/REPLAY.md)

| 日期 | 进展 | 对应材料 |
| --- | --- | --- |
| 2026-11-10 | 独立原型完成 eager 服务的 HTTP 和逐层 CPU 对照。保留首次数值分歧。 | [冻结检查器](validator/compare_traces.py) |
| 2026-11-10 | 按 DSV4 的既有入口把 V4.1 纳入仓内模型注册，添加模型专用的原始权重加载器、随包聊天模板和本地 Engram 逻辑。移除独立 day0 安装包。 | [模型注册](../vllm_ascend/models/__init__.py) · [权重加载](../vllm_ascend/model_loader/deepseek_v41.py) |
| 2026-11-10 | 指定镜像的 CPU 检查已覆盖 HF 文本配置、分片与单文件、离线缓存、损坏输入、逐 rank 权重形状和 n-gram 语义。标准 auto、hf 与 safetensors 格式均选用 V4.1 专用加载器。 | [自动选择](../vllm_ascend/platform.py) · [加载校验](../vllm_ascend/models/deepseek_v41/weights.py) |
| 2026-11-10 | 整仓 wheel 正在指定 A5 镜像中构建。新容器的普通 vllm serve 和八卡复验结果待归档。 | [重放步骤](patch/REPLAY.md) |
| 2026-11-10 | 新容器已安装包含仓内 Python 源码及指定镜像原有 A5 二进制的单一 wheel，并检查原生文件哈希。普通 `vllm serve` 已开始冷启动，取证开关关闭。标准源码构建仍在运行。 | [注册入口](../vllm_ascend/__init__.py) |
| 2026-11-10 | 初次仓内普通 CLI 已自动识别 V4.1 并选择仓内专用加载器，所有 rank 完成权重加载。服务就绪前，HCCL 首次集合通信报设备内核初始化错误；本轮未进入 HTTP 检查，正在独立复验通信。 | [普通 CLI](patch/infra/serve.sh) · [HCCL 探针](patch/infra/hccl_probe.py) |
| 2026-11-10 | 设备日志定位到 AICPU 缺失符号 `RunAicpuIndOpCommInit`。仅在进程内选用 HCCL AIV 后，八卡集合通信完成，普通 CLI 自动加载随包模板并完成四个真实 HTTP 请求，依次返回 ok、4、7、ok。已保存默认通信模式的失败记录。 | [复验说明](patch/REPLAY.md) · [服务检查器](validator/) |
| 2026-11-10 | 第二次独立冷启动完成模型、模板、HTTP、各 rank 权重账本和逐层输出采集；另在八卡重跑 Engram。固定请求的贪心 token 与冻结 HF CPU 对照逐步相同，第一处浮点差异在第 0 层 hidden，数值判定为 INCONCLUSIVE。新进程另行核验了安装包的模型和加载器注册。仓内阶段一归档，转入阶段二。 | [数值比较器](validator/compare_traces.py) · [Engram 检查器](validator/verify_engram.py) |
