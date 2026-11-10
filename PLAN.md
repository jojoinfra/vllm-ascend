# DeepSeek V4.1 Flash 适配计划

## 目标与约束

- 当前执行状态（2026-11-10，北京时间）：完整适配目标已由用户暂停；本次仅按用户最新指令提交并推送当前代码、计划和进展。
- 完整交付目标仍为完成适配和全部实机验收。本次提交是用户要求的阶段性快照，不代表阶段 5–9 验收完成。恢复目标后继续剩余阶段。
- 起点：vLLM Ascend `v0.27.1rc1`，SHA `3b31886237ed65c435e0965b10e8002ff5769766`。
- 仅修改 vllm-ascend；上游 vLLM `v0.27.1` 保持 SHA `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` 且工作树干净。不升级 transformers。
- 工作分支：`codex/dsv41-flash-0271rc1-mindie`。
- 本地源码：`/Users/jiangchenzhou/code/dsv41-mindie-20261109/vllm-ascend`。
- 远端任务：`jcz-dsv41-mindleagent`，实现目录 `/vllm-workspace/vllm-ascend`，运行目录 `/workspace`。
- 模型：`/home/admin/model`，DeepSeek-V4.1-Flash-1017 W8A8，真实权重验收。
- 默认复用指定 `v0.27.1rc1` 基线中的 DSV4、CANN、Ascend 自定义算子；必要适配采用插件侧注册、继承或精简补丁。
- 遵守用户提供的 AGENTS.md：新增功能和缺陷修复补充测试，环境变量集中定义，避免设备同步热路径，commit 带 Signed-off-by，push 到个人 fork。
- 不查询或使用历史任务文档中已有的 DSV4.1 实现、适配方案、补丁或实现细节。

## 最新来源限制（2026-11-10，最高优先级）

- 使用 MindIE Agent 的通用能力和知识库，包括通用算子接口与约束、硬件规则、调试、性能分析和实验管理。
- **禁止查询已有 DSV4.1 实现**，包括本地或远程其他分支、其他工作树、现有适配仓库/PR/补丁，以及知识库中的适配代码和实现方案。
- 不通过知识检索、其他工具或代理间接获得上述实现；不复制、移植或 cherry-pick 其他分支的适配内容。
- **DSV4.1 实现仅参考 Hugging Face 上 DeepSeek 官方实现**。当前指定入口为 `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017`；官方页面中的外链不自动成为获准实现来源。
- 指定基线中的 DSV4 代码、通用组件与 CANN/Ascend 算子资料仍可用于复用和核对；不得借此引入其他 DSV4.1 实现。
- MindIE 检索须聚焦通用技术问题；若结果包含已有 DSV4.1 适配实现，不展开或采纳该部分。
- 最新限制提出前已查看的历史分支片段、历史文档和 DSV4.1 知识条目导航，仅保留为审计事实，不作为后续设计、代码或验收依据。
- 先前授权的 `jcz-offical-dsv41` 仅可在恢复后用于黑盒评测对照，不查阅其 DSV4.1 源码或把它作为实现参考；最新来源限制优先。

## 验收标准

| 功能 | 验收要求 |
| --- | --- |
| 前缀缓存 | 真正命中，命中与无缓存结果一致；覆盖压缩边界、Engram 历史、跨层 KV/index 复用 |
| TP / EP | DP4TP4 与 DP2TP8；EP 使用 Ascend FusedMoE，并证明实际执行路径 |
| DSpark | 真实 mtp/Markov 权重、正确采样/拒绝处理，记录接受率 |
| 特性开关 | enable_dsa_cp、enable_flashcomm1、enable_shared_expert_dp、enable_cpu_binding，单开关与组合验证 |
| Vision | 真实 vision/aligner 权重、图片处理及图文请求，含准确性样例 |
| Decode full graph | 图捕获和实际回放证据，不能以 eager 路径冒充通过 |
| 长上下文 | 用户已明确：两种布局均支持全服务并发 16 × 150k，不只是单请求 |
| 性能 | 两种布局分别测试全服务 C1/C4/C16；8192 输入 / 1024 输出；报告 TPOT 均值/分位、TTFT、吞吐、成功数 |
| 精度 | aisbench；默认 GPQA Diamond 全量；temperature=1.0、top_p=1.0；用户通过线 ≥86% |
| 对照 | 恢复后必要时用 jcz-offical-dsv41 和同一 W8A8 权重做黑盒评测；禁止查阅或参考其已有 DSV4.1 实现，不将旧任务 GSM8K 当 GPQA 通过证据 |
| 交付 | 必要测试、格式检查；一个聚合的 signed-off commit，push 到 origin 用户 fork；中文 runbook 和证据 |

## 需固定的测试细节

- 150k 暂按 153600 个总上下文 token 实现（比十进制 150000 更严格）；输入预留生成空间，显式记录实际输入和生成长度。
- 并发均为整个服务的并发，不是每个 DP 副本的并发。长上下文用独立输入，避免共享前缀掩盖容量。
- 性能固定 8192/1024 token，可用 ignore_eos 保证长度；不能把这个 1024 输出限制用于 GPQA。
- 官方 README 推荐 max_tokens ≥256K，而本任务最大上下文为150k，不能同时满足。GPQA 在服务预算内采用足够长输出，记录截断率；固定思考模式、reasoning effort、提示编码、答案提取和随机种子；不通过缩短思考获取虚假性能/精度。
- 优先先测 GPQA 小样本排错，再做全量正式验收；固定阈值，不事后调低通过线。
- 16×150k 必须实际并行驻留和完成，记录抢占/重计算，不能仅以启动参数或排队成功宣称容量通过。

## 阶段

1. [x] 环境、权重、基线、官方模型说明预检；MindIE 激活与实际查询。
2. [x] 创建独立分支、工作树和持久计划/过程日志。
3. [x] 仅对照指定基线 DSV4 与 Hugging Face 官方 V4.1 实现，分析差异和可复用通用组件；制定最小实现。禁止参考已有 DSV4.1 适配。
4. [x] 插件内配置/模型/量化/Engram/注意力与权重加载适配；先基础文本真实权重请求。
5. [ ] TP/EP FusedMoE、DSpark、Async、缓存、开关、decode full graph 组合适配。
6. [ ] Vision encoder、图片处理、前端编码与图文真实权重验证。
7. [ ] DP4TP4 / DP2TP8 各自 16×150k 容量验证并修复。
8. [ ] 两布局 C1/C4/C16 性能与 aisbench GPQA ≥86% 验收。
9. [ ] 回归测试、文档、格式检查、signed-off commit、push；核实远端提交。

## 当前阶段检查点（2026-11-10）

- 已完成配置解析、静态KV/index归属规则、Engram词表压缩、分页history、ratio1/2 compressor、delayed-mHC和缓存池规划；50项适配UT通过，另有33项相关基线缓存回归通过。
- Engram完整tokenizer及24列哈希与固定官方实现一致，A3上eager和图回放通过。
- 上述属于阶段3/4基础组件验证，不代表模型服务、prefix cache、DSpark或decode full graph整体验收通过。
- Engram两张完整INT8/E8M0表已在EP16共同驻留，rank0占用约11.80GiB；查询、history原语和真实投影/gate均有NPU图验证。仍未接入真实scheduler/模型服务。
- 分页history已通过真实runner分配与KVCacheManager前缀命中测试；实际checkpoint compressor/mHC通过A3变化输入图回放。完整40层缓存拓扑的分组/池预算测试通过；仍不代表真实服务prefix/full-graph/150k容量验收。
- 第四轮已实现attention、RoPE、量化、index/candidate与跨层forward上下文；修复A3 exp2误差影响FP4中点舍入。实际层0/2/20的真实W8A8 attention prefill和变化位置图回放通过，4项NPU量化回归通过。
- 已注册模型骨架与indexed loader、接入Ascend FusedMoE；64项适配UT通过，随后新增量化配置回归，模型与现有ModelSlim回归合计54项通过。真实DP4TP4/EP16服务启动联调中，首个真实文本请求已完成（2+2→4，HTTP200），尚待更完整验证。
- DP4TP4 eager已实际完成256-token前缀命中及相同输入的输出/logprob一致性，不等长DP并发通过。FULL_DECODE_ONLY已捕获与回放，profiling确认实际MoE/graph执行。FP8图内标量移位干扰已通过最小反例和旧/新回归定位并修复；无观测v10两轮334token warm/hit/miss的token与selected logprob均与eager完全一致，第二轮在同DP四请求和不等长DP请求之后。69适配UT与5NPU量化测试通过；整模型结论仅限已测DP4TP4短文本。接下来继续DSpark、Vision及完整阶段5–9，未降低任何验收标准。

- 第九轮：独立 draft embedding/head 在加载后被通用 proposer 替换的问题已修复；原源码回归失败，修复后126CPU UT通过。v15真实三类请求接受233/340草稿token（68.53%），五个位置均有接受、无抢占。probabilistic、完整图/组合与长上下文仍待验证。
- DSpark使预填充块由128变112，warm/hit末块110/94行；真实权重独立mHC实验逐值重现最早pre-FFN差异，FP64 CPU对照确认FP32微小舍入差异被BF16量化放大。任意batch数值不变性未通过；v16用144总token预算（预留16、实际128）和无观测worker通过该334-token输入的prefix token/selectedLP逐值一致性，真实hit128；接受率233/340复现。相对非DSpark baseline的decodeLP仍差0.04084866，未宣称任意batch数值不变。未降低既定验收门槛。

## 执行和恢复规则

- 每个关键变更或实验后更新 `PROGRESS.md`，记录代码 SHA/diff、命令、日志、结果和下一步。
- 每次上下文压缩/恢复先读 `PROGRESS.md` 的 `in_flight_runs`，查询已有进程/任务，禁止重复启动。
- managed remote-dev/coordinator 后台任务必须立即登记 job_id/run_id、slot、endpoint、manifest，未登记前不等待。
- 实机运行前确认 iTASK 所有者、设备与端口；仅终止本任务自己启动且身份已核实的进程。
- 使用 iTASK 管理任务，不绕过所有权检查，不删除或重置用户现有目录。
- 默认不能派生子 agent；仅当前用户或适用技能的具体分派条件授权时才可使用。
- 失败保留证据；dummy 只用于早期调试，所有最终结论必须真实权重验证。
- 当前目标已暂停；本次提交和推送不恢复适配实验。后续恢复时继续未完成门槛，不以阶段性提交代替完整适配。

- 第十轮：206CPU回归通过；Vision完整263权重/32层的3张实图（含非纯色）与固定官方同NPU encoder/aligner逐元素一致，最大方图/宽图内存增量4.69/5.27GiB。v21真实DP4TP4 DSpark+Vision FULL_DECODE_ONLY捕获[6,12,24]且4DP实际回放，短prefix与DSpark eagerv16逐值一致，接受233/340（68.53%），4个单图/双图颜色及顺序样例通过。开启图片需0.96显存预算，本轮可用KV约0.65GiB，不能据此宣称长上下文容量。阶段5/6其余组合仍待完成。

- 第十二轮：已将剩余前缀差异追溯到早期历史token的BF16浮点投影批量舍入。新固定64行投影保留native TP/量化路径；270CPU回归、原始/修复paired NPU（原分组投影失败、修复2项通过）、真实3类投影36混合行案例和18变化图回放通过，完整format CI v5通过。49文件暂存。v28默认worker DP4TP4/EP16服务正在128总预算下复验任意prefill分块前缀；尚不宣称整模型门槛通过，也未commit/push。

- 第十三轮：v28无观测服务原334-token严格prefix通过；扩展12边界全部token IDs一致，257/513/1023/1024的selectedLP仍不同，其余8逐值一致。真实hit最高896、saltedmiss0、全DP抢占0；同DP4与不等长DP4均通过。新v29保持生产源码不变，用按逻辑位置的层边界摘要定位257-token最短反例，先验证observer外部结果与v28一致。阶段5仍未完成。

## 阶段性提交检查点（2026-11-10）

- 按用户最新要求，将当前适配代码与本计划、完整进展记录一起提交到 `codex/dsv41-flash-0271rc1-mindie` 并推送个人 fork。
- 前缀数值门槛仍未通过：257-token 反例中，warm 末块 33 tokens 走 All-to-All，hit 末块 17 tokens 走 MC2；相同输入在 dispatch 后首次出现 INT8/scale 差异。该诊断尚未转为生产修复。
- 阶段 5–9 保持未完成；GPQA、两布局长上下文并发容量和性能不作通过声明。
- 仓库根目录 `PLAN.md` 与 `PROGRESS.md` 是本次推送快照；其中历史实验引用的日志和 artifacts 位于任务目录，未随本次源码提交上传。

## 用户追加评测（2026-11-10）

- 用户要求本次代码、计划和进展提交推送完成后，运行GSM8K测试集前300题，全服务并发32，DP4TP4，开启decode full graph与DSpark。
- 这是独立授权的评测任务；完整适配目标仍暂停，不替代原定GPQA、长上下文容量或功能门槛。评测须使用本次提交的实现，记录实际服务参数、图回放、DSpark执行、准确率、成功数和截断。
