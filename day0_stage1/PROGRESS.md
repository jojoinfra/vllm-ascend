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
| 2026-11-10 | 阶段二在单张 A5 上用真实主干与草稿权重核对 MoEHashTopk、RMSNorm、FP8 和 MXFP4 矩阵调用，并捕获后重放路由图。路由选择与独立计算一致，部分 FP8 矩阵输出有 BF16 舍入差异。此项覆盖单算子。 | [原生算子](../vllm_ascend/models/deepseek_v41/native_ops.py) · [上卡检查](../day0_stage2/validator/verify_native.py) |
| 2026-11-10 | 已加入按 vLLM 物理 KV 页保存请求历史的实现。交错请求、换序分块、拒绝后覆盖和释放后复用页均与 HF 全前缀哈希对齐，非法 slot 会被拒绝。 | [请求状态](../vllm_ascend/models/deepseek_v41/decode_state.py) · [回归用例](../tests/ut/models/test_deepseek_v41_paged_history.py) |
| 2026-11-10 | 三层 DSpark 草稿、顺序 Markov head 和 confidence head 已加载原始权重，复用了目标 embedding 和输出 head。单卡用冻结的目标层输出验证了完整五位提案，原生量化下提案 token 均与 HF CPU 对照一致，confidence 数值仍有差异。 | [草稿模型](../vllm_ascend/models/deepseek_v41/dspark.py) · [对照探针](../day0_stage2/validator/verify_dspark.py) |
| 2026-11-10 | 评测客户端已按冻结的 AISBench 配置完成 dry-run。检查了官方题目顺序、评分器源码和最终解析的 reader 范围、总并发、输出上限与单次尝试。正式评测等待组合服务。 | [评测启动器](../day0_stage2/validator/aisbench_gsm8k.py) |
| 2026-11-10 | 从仓内模型注册回调接入 DSpark 配置。指定镜像的真实 EngineArgs 为 V4.1 识别训练得到的草稿架构、目标层、噪声 token 和完整提案长度。 | [推理配置](../vllm_ascend/models/deepseek_v41/spec_config.py) |
| 2026-11-10 | 按实际物理页接入滑窗、压缩 KV、两级 indexer 和跨层共享。在单卡缩维六层用例上，真实 ACL 图捕获后三次重放均与逐请求 eager 输出一致。CPU 端另记录了填充宽度影响并列 top-k 的案例。 | [分页注意力](../vllm_ascend/models/deepseek_v41/graph_ops.py) · [硬件探针](../day0_stage2/validator/verify_paged_attention.py) |
| 2026-11-10 | 接入 AscendMoERunner 与原始 FP4 路由专家、FP8 共享专家。真实 A5 探针观察到路由、分组矩阵、重排及共享计算，捕获后的图在不同输入上完成回放。融合输出相对逐专家参考仍有差异。 | [融合执行](../vllm_ascend/models/deepseek_v41/fused_moe.py) · [专家权重](../vllm_ascend/models/deepseek_v41/expert_weights.py) |
| 2026-11-10 | 普通服务入口按调度配置选择单序列 eager 或分页模型。分页模型在权重加载后安装原生专家，并在指定镜像的 ACLGraphWrapper 完成真实 capture/replay 时更新 Engram 输入。 | [模型选择](../vllm_ascend/models/deepseek_v41/model.py) · [图执行](../vllm_ascend/models/deepseek_v41/graph_model.py) · [入口用例](../tests/ut/models/test_deepseek_v41_execution.py) |
| 2026-11-10 | V4.1 worker 已接入 DSpark 提案和目标模型拒绝采样，按请求提交被接受的目标行。单卡实测含真实接受、拒绝和污染无效行的检查。八卡组合仍待实测。 | [Worker 适配](../vllm_ascend/models/deepseek_v41/spec_runner.py) · [请求边界回归](../tests/ut/models/test_deepseek_v41_spec_runner.py) |
| 2026-11-10 | 同一阶段一实现的完整 A5 原生源码 wheel 已用普通 pip wheel 构建。安装后再次独立冷启动普通服务与取证服务，并重跑八卡 Engram。两次 HTTP 结果与此前一致，八份逐层张量文件逐字节相同。 | [仓内重放步骤](patch/REPLAY.md) · [数值探针](validator/compare_traces.py) |
| 2026-11-10 | 修正了同一批次中长短请求的 DSpark 上下文判断。接近长度上限的请求停止提案，其余请求继续；同一轮活跃或空闲的 DP worker 都只进入一次草稿计算。 | [调度回归](../tests/ut/models/test_deepseek_v41_spec_runner.py) |
| 2026-11-10 | 评测客户端已增加 HTTP 边界观测，记录每个题目 ID 的实际请求次数、完成状态、原始生成和同时在途数。用两条合成请求核验了转发及重叠测量，尚未开始正式 GSM8K。 | [HTTP 观测](../day0_stage2/validator/observe_gsm8k.py) |
| 2026-11-10 | 首次八卡阶段二候选已从普通 CLI 加载目标模型、独立的 DSpark 权重和所有 rank 的原生 MoE runner。启动 forward 中，ALL2ALLV 的 MX scale 张量与按 token 切分的通信尺寸不符，服务未进入 HTTP；已保留该轮及权重账本。 | [EP 分发](../vllm_ascend/ops/fused_moe/token_dispatcher.py) |
| 2026-11-10 | A5 单卡验证了动态 FP8 激活的 scale 算法。在含零值与量化边界的合成 BF16 块上，scale_alg=1 的所有 E4M3 码与所有非零组 scale 对齐 HF 参考，已按 V4.1 文本类型接入现有硬件能力选择。 | [量化选择](../vllm_ascend/quantization/utils.py) · [回归用例](../tests/ut/quantization/test_utils.py) |
| 2026-11-10 | DSpark 现在保留本步未排到的请求。槽位需要让给其他请求时保存已提交草稿 KV，恢复调度时按原请求还原，可继续验证或回绕。 | [请求状态与回归](../tests/ut/models/test_deepseek_v41_spec_runner.py) |
| 2026-11-10 | 专家字节布局回归已统一从发行包导入，覆盖跨 rank 范围、W1/W3 顺序、E2M1 原码与 UE8M0 边界。 | [权重回归](../tests/ut/models/test_deepseek_v41_expert_weights.py) |
| 2026-11-10 | 阶段二并发 HTTP 检查器复用阶段一冻结的问答检查，同时发起独立对话，核验不同请求 ID 与实际重叠。传输自测通过。 | [并发检查器](../day0_stage2/validator/check_concurrent_chat.py) |
| 2026-11-10 | 计划书已补充阶段二完整服务、无投机对照与 AISBench 的启动和取证方式。八卡首轮已完成较短的 target/draft 预热前向，较长预热在专家切分处失败。 | [阶段二计划](PLAN.md#阶段二的服务与评测) |
| 2026-11-10 | 八卡复现确认了首轮预热故障。主干与草稿曾共用按通信类型索引的分发器，后加载的草稿覆盖了专家数。现在每个 V4.1 runner 在调用实际 Ascend 算子时选用自己的通信配置。物理交换与原有 MX scale 维度通过八卡复验。 | [模型专用 runner](../vllm_ascend/models/deepseek_v41/fused_moe.py) · [八卡复现](../day0_stage2/validator/verify_moe_exchange.py) |
| 2026-11-10 | 分发器增加正常调用和异常退出的嵌套回归，已在指定镜像的安装包上执行。第二轮八卡通过完整权重加载和长预热，进入首个 FULL decode capture；worker 初始化草稿注意力元数据时缺少 attn_layer_names，尚无 HTTP 或正式评测结果。 | [通信上下文回归](../tests/ut/models/test_deepseek_v41_moe_communication.py) · [草稿接线](../vllm_ascend/models/deepseek_v41/spec_runner.py) |
| 2026-11-10 | 确定性服务探针现可从普通 completions 接口读取真实 prompt 和输出 token ID，保存完整响应，并逐 token 对照使用相同输入的无投机服务。独立传输测试覆盖一致和分歧两种结果。 | [确定性解码检查](../day0_stage2/validator/check_deterministic_decode.py) |
| 2026-11-10 | FULL 的 uniform decode 图现在通过核心 runner 的预热入口使用与捕获相同的索引长度。普通 profile 和短 prefill 继续按实际长度准备，预热异常也会恢复状态。指定镜像中六项 CPU 回归通过。 | [图准备与恢复](../vllm_ascend/models/deepseek_v41/graph_model.py) · [入口回归](../tests/ut/models/test_deepseek_v41_execution.py) |
| 2026-11-10 | 已修复第二轮八卡捕获前的草稿元数据错误。DSpark 公布目标模型注册的注意力名，核心据此选择实际 verifier 缓存组及 kernel block size；草稿的三层历史仍由各自缓存管理。指定镜像回归覆盖非零组号、缺组和重复组。 | [草稿元数据](../vllm_ascend/models/deepseek_v41/spec_runner.py) · [回归](../tests/ut/models/test_deepseek_v41_spec_runner.py) |
| 2026-11-10 | 前两轮组合的主干和草稿逐层账本均显示 TP1 下共享专家 DP 和序列并行未启用。已把两项的实际状态、共享专家本地数值、路由 EP 通信分别写入验收计划。 | [TP1 记录](PLAN.md#tp1-的共享专家与通信) |
