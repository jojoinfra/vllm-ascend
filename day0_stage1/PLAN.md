# DeepSeek V4.1 A5 仓内交付计划

阶段一已用仓内 vllm-ascend 发行包完成八卡复验。当前执行阶段二的开发、上卡和评测。阶段二的模型及推理改动继续随同一仓安装，使用普通服务入口。每个独立改动提交并推送到开发分支，实际执行情况写入 [PROGRESS.md](PROGRESS.md)。

| 顺序 | 交付内容 | 检查方式 |
| --- | --- | --- |
| 1 | 使用基线 DSV4 的模型注册入口登记 V4.1，通过 Ascend 权重加载器入口注册 V4.1 loader。模型、加载器、Engram 实现和缺省聊天模板随 vllm-ascend 发行。 | 安装发行包，在新进程中核对模型类、加载器、模板及 wheel 文件清单。检查 DSV4 注册和其他模型的默认加载路径。 |
| 2 | 从标准 `vllm serve <模型目录>` 启动。调用方提供普通硬件拓扑与 eager 配置，使用原始权重和 checkpoint tokenizer。 | 在指定镜像的新容器里运行八卡 DP8 / EP8 / TP1 / PP1，保存实际标准 CLI 和各 rank 的设备、专家、权重与 Engram 放置。 |
| 3 | 在这份仓内发行包上复验阶段一。 | 核对 HCCL、独立 Engram 输出、模型列表、模板 IDs、四个 HTTP 请求、prefill 末位 logits 与连续 decode。按原先的数值检查规则记录第一处分歧。 |
| 4 | 恢复阶段二，接入原生量化、融合路由与专家执行、decode 图回放、DSpark 提案验证及拒绝回滚。 | 在同一组合服务上复跑功能验证和已冻结的 AISBench GSM8K 验收，保存逐题结果与实际并发。 |

实现以冻结的官方 HF 模型语义、指定镜像以及基线 DSV4 注册接口为依据。其他分支已有的 DSV4.1 实现不在来源范围内。精确的环境、对照版本、权重哈希、样本和结果存放在私有验收记录中。[重放说明](patch/REPLAY.md)

## 阶段二的服务与评测

复验从同一仓的 wheel 和 `vllm serve` 启动。完整候选使用 eager prefill、`FULL_DECODE_ONLY`、原始权重及 `--speculative-config '{"method":"dspark","num_speculative_tokens":5}'`。冻结的验收记录提供上下文、页数、批宽、dtype、镜像和并行配置。对同一候选还会用关闭投机的确定性请求作对照。

先检查各 rank 的目标和草稿权重、原生 experts runner、实际通信方式与图捕获。服务就绪后重放阶段一四问，并同时发起独立请求。记录有实际请求时的 decode replay、已接收与被拒绝的草稿 token、拒绝后的请求状态以及共享专家的放置和通信。开关是否生效由运行路径判断。[并发功能检查](../day0_stage2/validator/check_concurrent_chat.py)

独立 AISBench 客户端先核验官方评分器和输入顺序，支持用 `--dry-run` 检查最终任务配置。正式请求记录题目 ID、HTTP 状态、模型输出和在途数，并与 AISBench 逐题评分对照。题量、并发数、分数门槛以及单次尝试预算均由冻结验收文件提供。[评测启动器](../day0_stage2/validator/aisbench_gsm8k.py) · [HTTP 观测器](../day0_stage2/validator/observe_gsm8k.py)
