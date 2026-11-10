# DeepSeek V4.1 Flash 适配过程与恢复入口

## 当前状态

- 最新恢复入口：目标已暂停，本次只提交/推送当前代码与计划、进展。v31 CPU 分析25301已exit0：原始hidden/路由权重一致，All-to-All与MC2的dispatch后INT8有114个元素不同、scale有36个不同；首差已定位，尚未修复。代码保持诊断时实现。
- 最新验证检查点：270项CPU回归通过，上一轮完整format CI v5通过；真实权重组件、部分DP4TP4服务路径已有证据。12个prefix边界仍有257/513/1023/1024的selected logprob差异。完整GPQA、容量/性能和组合验收未完成。
- 当前执行状态（2026-11-10，北京时间）：用户已暂停完整适配目标，随后明确要求当前代码commit/push，并加入计划与进展。本次是阶段性提交，不恢复实验。
- 当前阶段：阶段5/6联调。无DSpark DP4TP4/EP16短文本、prefix与full graph已验证；DSpark greedy接受率68.53%，probabilistic65.53%，对齐prefill后prefix命中/未命中token+selectedLP完全一致，但相对非DSpark decode LP仍有差异。DSpark graph首次启动遇到草稿Dynamo序列化通信句柄，已按通用proposer的eager约定修复；v20越过加载/内存预热，因0.90预算下KV可用-3.03GiB退出，v21用0.96预算成功启动，4DP均实际图回放；prefix128hit及相对DSpark eager token/LP完全一致，4个单图/双图颜色顺序样例通过。206项CPU回归通过；Vision已注册processor/视觉权重/图片路由，保留完整3DSDPA运算shape并复用score缓冲后，3图（含非纯色）与官方NPU encoder/aligner逐元素一致，最大方图/宽图额外内存4.69/5.27GiB。v19缺少image_token_index别名已修复，v20实际加载越过该阶段。完整Vision、两布局容量/性能、GPQA、commit/push未完成。
- 分支：`codex/dsv41-flash-0271rc1-mindie`。
- 本地源码：`/Users/jiangchenzhou/code/dsv41-mindie-20261109/vllm-ascend`。
- 主计划：`/Users/jiangchenzhou/code/dsv41-mindie-20261109/PLAN.md`。
- 已执行CPU UT、真实权重attention及NPU图回放验证，并开始基础服务启动尝试；完整服务、性能/精度验收、commit和push均未完成。

## 本次阶段性提交（2026-11-10）

- 用户授权将当前代码、`PLAN.md` 和 `PROGRESS.md` 一起提交并推送个人 fork；完整适配目标保持暂停。
- 当前49个源码、测试及runbook文件，加仓库根目录的计划、进展及历史任务名的精确拼写检查例外，共52个文件。提交采用Conventional Commits并添加Signed-off-by。
- v31观测控制通过：warm/hit/miss三次响应的token IDs及selected logprobs逐值复现无观测v28，生产代码未加入诊断钩子。
- CPU分析按全部16个EP rank的实际路由、dummy mask、专家分组和行映射对齐。warm/miss的33真实tokens加18dummy共306条专家行走All-to-All；hit的17真实tokens加18dummy共210条专家行走MC2。
- 共同17个真实tokens的hidden和路由权重完全一致；dispatch后INT8在位置240/242/246/252共114个元素不同，scale在6个位置共36个值不同（最大3.725290298461914e-09）。MLP输出88774个BF16元素不同，combine后14577个元素不同；warm/miss各边界完全一致。故首差在dispatch量化阶段，尚未证明下游在统一量化输入后完全一致。
- 已排除本反例的router、shared experts及最终TP gather作为首差；没有修改生产代码来回避后端切换，也没有放宽前缀数值门槛。
- 验证证据：`logs/moe-backend-v31-v3.json`、`logs/diagnostic-v31-moe257.json`；七份相关日志/JSON已由retry80123成功取回并核对SHA。历史失败的inventory/v2分析保留，分别纠正了dummy计数和混合后端假设。
- 尚未完成：前缀跨batch完整一致性、DP2TP8整服务、DSpark采样/拒绝完整对照、开关与Vision组合、两布局16×153600实际并发驻留、8192/1024性能及全量198题GPQA≥86%。
- 提交前完整格式检查第一轮发现历史任务名称的既有拼写与Markdown样式问题；保留真实任务名，仅添加完整标识符匹配的拼写例外，修正Markdown并重新检查。检查记录为 `logs/format-ci-checkpoint-v1.log` 和 `logs/format-ci-checkpoint-v2.log`。
- 此文件保留历史记录，旧条目中的“未commit/push”“正在运行”等属于当时状态；当前状态以本节与最上方恢复入口为准。日志和artifacts引用指向本地任务目录，未随本次源码快照上传。

## 最新约束与来源边界（恢复必读）

- 必须使用 MindIE Agent 的通用能力和知识库，用于通用算子、硬件约束、调试、性能分析和实验管理。
- **禁止查询已有 DSV4.1 实现**：其他本地/远程分支、其他工作树、现有适配仓库/PR/补丁、历史任务实现材料，以及知识库中的适配代码和实现方案均不允许。
- **DSV4.1 实现只参考 Hugging Face 上 DeepSeek 官方实现**，当前指定仓库为 `deepseek-ai/DeepSeek-V4.1-Flash-1017`。不得沿外链扩大实现来源范围。
- 可继续研究指定 `v0.27.1rc1` 基线中的 DSV4 和通用组件，以及 CANN/Ascend 通用算子资料。
- 不通过 MindIE 检索或其他工具/代理绕过限制；检索意外命中已有 DSV4.1 适配内容时，不展开或采纳。
- 先前查看的历史分支内容与 DSV4.1 知识库导航不作为后续设计、代码或验收依据，不复制、移植或 cherry-pick。
- `jcz-offical-dsv41` 的旧授权仅保留黑盒评测对照用途；不得读取其 DSV4.1 源码作为实现参考。当前也不启动对照任务。

## 约束变更审计

- 最新限制提出前，曾读取历史分支的 diff/stat、部分源码和部署文档，并查询到 DSV4.1 知识库条目导航；相关知识条目正文未加载。这些是已发生的读取，不能表述为从未查看。
- 未从历史分支复制、移植或 cherry-pick 代码；本任务新工作树仍没有适配源码变更。
- 原计划中的“比较历史分支”“读取 DSV4.1 知识条目”步骤现已撤销。后续依据必须能追溯到获准的基线、Hugging Face 官方实现或通用技术资料。
- 2026-11-10 本次操作范围仅为修改本文件及 `PLAN.md`，不继续适配工作。

## DSpark 服务接入（第八轮，2026-11-10）

- v11 的 draft 继承了编译包装器调用方法，但自定义构造函数未初始化包装器。对子类应用 `support_torch_compile`，新增真实类构造与调用回归；隔离 LogitsProcessor 的无关测试依赖后，13 项相关测试通过。
- v12 在目标网络 All-to-All token dispatcher 报 split sizes 不符。基础 MoE runner 每构造一层都会替换进程级通信表，而目标网络是 384 experts/top-k 6，草稿网络是 128 experts/top-k 3，导致目标网络使用了草稿的直方图范围与专家分片。
- 两个模型分别保留构造完成时的通信对象，在对应 forward 作用域绑定，并通过 finally 恢复先前上下文。没有新增全局状态或修改通用 MoE/上游代码。新增嵌套及异常恢复测试。
- 全部适配 UT、既有 DSpark proposer UT 和新增视觉基础 UT 共 125 项通过，日志 `logs/dspark-comm-vision-ut-v1.log`。真实服务 v13 通过启动预热、health200 和三类文本请求；277 次 draft 提出 1385 个 token，仅接受 22 个（约 1.59%，全部在第一位置），不能据此认定 DSpark 正确。
- v13 前缀 warm/hit/miss 输出 token 均为 [25,954,1]，真实命中 128 token，但 selected logprob 不一致，且相对 eager-v4 最大差 0.1225235。严格 parity 检查 exit1，证据在 `logs/dspark-eager-parity-v13.json`。准备观察目标/草稿输入、位置、逐层输出和 Markov logits，定位首次偏差。

## Vision 基础组件（阶段 6 准备）

- 新增 `deepseek_v41_vision.py`：官方图片尺寸规划与归一化、二维 split-half RoPE、ViT、channel-major aligner 和图像分隔符布局。注意力复用通用 MMEncoderAttention（Ascend FIA）。尚未注册多模态 processor，也未接入主模型、图片专家路由或真实视觉权重。
- 10 项基础 UT 覆盖极端宽高比、完整 image span token 预算、patch 顺序、RoPE 轴分离、aligner 补零和行分隔符。首次 aligner 测试混入非连续布局 GELU 的 1.19e-7 舍入差异；改为直接检查 GELU 前布局，没有降低数值容差。
- 固定官方 vision/image_processor 已校验 SHA 后上传。7 种尺寸规划、真实像素转换和 3 种小型随机 BF16 ViT/aligner 输入与官方 CPU 参考逐值一致，日志 `logs/vision-official-cpu-v3.log`。v2 的 4D SDPA stub 与官方 3D SDPA 使用不同 CPU 内核，最大差 0.0078125；v3 将 CPU stub 对齐为官方输入维度，生产源码未改，精确断言未放宽。该结果不能代替 NPU FIA、真实权重或图文准确性验收。

## in_flight_runs

- TERMINAL exit0: evidence-only retry pull session `80123`; endpoint jcz-dsv41-mindleagent; all seven v31 log/JSON files pulled and SHA-verified locally. No inference or NPU experiment.

- TERMINAL exit1: initial iTASK API Service Unavailable; subsequent retry80123exit0 fetched all seven logs/JSON with SHA verification. evidence-only pull session `26273`; endpoint jcz-dsv41-mindleagent; SHA-checking seven v31 client/backend logs and JSON files into local logs/. No inference or NPU experiment.

- TERMINAL exit0: logical alignment passed; warm/miss exact; hit dispatch quant114/scale36 differences, downstreamMLP88774/combine14577 differences. CPU-only mixed-backend analysis session `25301`; endpoint jcz-dsv41-mindleagent; logs/moe-backend-v31-v3.log/json, artifacts/dspark-diagnostic-v31/moe257-backend-capsules-v3.pt. Warm33All2All, hit17MC2 padded20; verify active masks and group counts, establish MC2 logical identity via unique BF16 input within quantization half-step bound, reject ambiguous real rows. Service73729/PID172731 owns allNPUs.

- TERMINAL exit1: warm All2All inverse-permutation alignment passed306rows/maxquant-halfstep-ratio1.000003; hit uses MC2 metadata, so extend analyzer for mixed backends. CPU-only all16EP backend row reconstruction session `84125`; endpoint jcz-dsv41-mindleagent; logs/moe-backend-v31-v2.log/json, artifacts/dspark-diagnostic-v31/moe257-backend-capsules-v2.pt. Uses actual inverse local/global permutation, all-rank route counts including dummy, verifies expert counts and dequant half-step alignment. Service73729/PID172731 sole NPU owner; no new NPU work.

- TERMINAL exit1: inventory assumed real-only rows198, actual306 includes3 idleDPx6dummyx6routes; actual path All2All. CPU-only v31 backend inventory session `29799`; endpoint jcz-dsv41-mindleagent; logs/moe-backend-v31-inventory.log/json; reads all16 rank snapshots with mmap, verifies one dispatch/MLP/combine and aligned valid expert counts. No NPU workload; service73729/PID172731 owns all16NPUs.

- TERMINAL exit0: warm/hit/miss observer_exact=true, reproduces unobserved v28. v31 all16EP257tokenclient session `44826`; endpointjcz-dsv41-mindleagent,existing73729/PID172731ownsallNPUs. logs/diagnostic-v31-moe257.log/json; artifacts/dspark-diagnostic-v31/rank{0..15}/moe257-{warm,hit,miss}-target-*.pt. Replaysv28params,assertobserverresponseexact; capturesfirstlayerbackenddispatch/MLP/combineincludingidleEP. Nocompetingclients; healthwait<=1200.

- LAST OBSERVED RUNNING (pause does not certify shutdown): v31 eagerall16EPbackendobserver service session `73729`,PID172731; endpointjcz-dsv41-mindleagent,all16NPU/8127/13491. Sourcef6f3f495unchanged,DP4TP4prob128budget4096/image0/.90. Workerartifacts/validation/dspark_diagnostic_worker_v31.py wrapsfirstlayerMoEdispatch/MLP/combine onall16ranks,includesidlemetadataNoneforEPparticipation;weightsomittedfromsnapshots. loglogs/service-dp4tp4-dspark-v31.log,manifestartifacts/service-manifest-v31.json. v3037185terminal0; launchfreechecks.

- TERMINAL exit0: routeinputs/logits/topkids/topkweightsallexact,sharedall4TPexact; routedlocal5positions240/242/243/246/252diff14577valuesmax.001953125; joinedlocal==finalizedexact. CPU-only v30 correctedTPshardanalysis session `31575`; endpointjcz-dsv41-mindleagent,logs/moe-shards-v30-v2.log/json,artifacts/dspark-diagnostic-v30/moe257-joined-v2.pt. Deriveactualshardsizes,assertzero-paddedfullinputssplitexactly,trimjointoutputtorealrowsbeforecomparing. Prior13189terminal1; service37185soleNPUowner.

- TERMINAL exit1: validationassumednopadding,buthit17actuallypaddedto20before5x4split;warm33split9/8/8/8. Correctalignmentbeforecomparison. CPU-only v30 TPshardreassembly session `13189`; endpointjcz-dsv41-mindleagent,logs/moe-shards-v30.log/json,artifacts/dspark-diagnostic-v30/moe257-joined.pt. Verifyactualtensor_split33->9/8/8/8and17->5/4/4/4beforejoiningtopk/quantoutputs; cross-ranklogicalalignmentrequired. Service37185soleNPUowner,noinference.

- TERMINAL exit0: sharedoutputsallexact; routedbranchfirstdiffers5rowsbeforefinalMoE4rows; analyzeralsocomparesphysicalcache-slotmetadatawhichmustbeexcludedfromnumericalinterpretation. CPU-only v30 all4TPanalysis session `99562`; endpointjcz-dsv41-mindleagent,logs/diagnostic-v30-moe257-analysis.log,artifacts/dspark-diagnostic-v30/rank{0..3}/moe257-comparison.json. Client93917terminal0observerexact; service37185soleNPUowner.

- TERMINAL exit0: warm/hit/missallobserverexactv28; v30 257tokenfunctionalobserver client session `93917`; endpointjcz-dsv41-mindleagent,existingservice37185/PID156298soleNPUowner. logs/diagnostic-v30-moe257.log/json,artifacts/dspark-diagnostic-v30/rank{0..3}/moe257-{warm,hit,miss}-target-*.pt. Samev28requestparams,assertobserverresponseexact; awaitshealth<=1200. No otherinference.

- TERMINAL exit0 after verifiedPID156298 SIGTERM by39329; v30 eagerMoEfunctionalobserver service session `37185`,PID156298; endpointjcz-dsv41-mindleagent,all16NPU/8127/13491. Sourcef6f3f495unchanged,DP4TP4prob128budget4096/image0/.90. Workerartifacts/validation/dspark_diagnostic_worker_v30.py snapshotsDP0ranks0..3MoEfunctionalboundariesbefore/aftercall,actualgateweightFP32/runtimeflags. loglogs/service-dp4tp4-dspark-v30.log,manifestartifacts/service-manifest-v30.json. Gate55332terminal0,nootherservice,launcherfreechecks.

- TERMINAL exit0 diagnostic: native/fixed64bothlogitsandtopkFP32/BF16exactacross33/17; no gatebatchdifference;FP64maxabs1.197e-5failsoriginalboundratio2.09. NPU0 gateprojectiondiagnostic v3 session `55332`; endpointjcz-dsv41-mindleagent,logs/moe-gate-v29-v3.log/json. OriginalFP64boundretainedasboolean+maxratio; do not stopbeforepersistingnativeandfixed64measurements. Prior42384terminal1; no acceptanceclaim, productionunchanged.

- TERMINAL exit1 strictFP64boundassertionbeforewritingmeasurements; NPU0 correctedrealgateprobe session `42384`; endpointjcz-dsv41-mindleagent,logs/moe-gate-v29-v2.log/json. BiasconvertedFP32likeproductionparameterloader; otherwiseoriginalprobeunchanged. Prior11394terminal1,launchrechecks8packagesfree.

- TERMINAL exit1 testfixturebiasdtypeBF16expectedFP32beforeanyresults; NPU0 realgateprojectionprobe session `11394`; endpointjcz-dsv41-mindleagent,logs/moe-gate-v29.log/json. Usesactualv29FFNinputs33/17,checkpointgateFP32 andnativeAscendgating,comparesnative/fixed64logits,topkids/FP32/BF16weightsandCPUFP64bound. v2998361terminal0;launcherchecks8packagesfree. No modelservice.

- TERMINAL exit0: firstdifferenceinsideFusedMoEoutput; nativeexpertsusekwargsandbypasschildmodulehooks, rawFFNinput/outputcaptured. CPU-only v29 detailFFN257 digestanalysis session `80197`; endpointjcz-dsv41-mindleagent,logs/diagnostic-v29-ffn257-analysis.log,outputartifacts/dspark-diagnostic-v29/ffn257-comparison.json. Client7031terminal0;service98361soleNPUowner.

- TERMINAL exit0: all3observerresponseexactv28; v29 detailFFN257 observerclient session `7031`; endpoint jcz-dsv41-mindleagent,existing98361/PID138687ownsNPUs. Same257tokenprompt,raw_namesmodel.layers.0.ffn,logs/diagnostic-v29-ffn257.log/json; snapshotsartifacts/dspark-diagnostic-v29/ffn257-{warm,hit,miss}-target-*.pt. First2578terminal;assertsameunobservedv28response.

- TERMINAL exit0: warm/miss257positionsallexact; hit129common, firstdivergencelayer0FFNoutput4positions240/242/246/252withsameinput; downstreamlayer1attentionspreads17positions240..256. CPU-only v29 boundary257 digestanalysis session `37424`; endpoint jcz-dsv41-mindleagent,loglogs/diagnostic-v29-boundary257-analysis.log,outputartifacts/dspark-diagnostic-v29/boundary257-comparison.json. Client2578terminal0; existingservice98361ownsallNPUs. No inference/NPUwork.

- TERMINAL exit0: allwarm/hit/missIDs+selectedLPexactv28observercontrol; v29 shortestboundary257 observercontrol client session `2578`; endpoint jcz-dsv41-mindleagent,existingservice98361/PID138687ownsallNPUs. logs/diagnostic-v29-boundary257.log/json; snapshotsartifacts/dspark-diagnostic-v29/boundary257-{warm,hit,miss}-target-*.pt. Exactv28prompt/request params,assertobserveroutputsidentical; majorboundariesperrowSHA. Awaithealth<=1200s; no competinginference.

- TERMINAL exit0 after verifiedPID138687 SIGTERM by22959; v29 eager diagnostic service session `98361`,PID138687; endpoint jcz-dsv41-mindleagent,all16NPUs/ports8127,13491. Sourcef6f3f495unchanged,DP4TP4probabilistic128budget4096/image0/.90. Workerartifacts/validation/dspark_diagnostic_worker_v29.py rowdigestsandoptionalrawcaptures; CPUdigestsmokepassed. loglogs/service-dp4tp4-dspark-v29.log,manifestartifacts/service-manifest-v29.json. v2850791terminal0;launcherrechecks8packages/portsfree/upstreamclean. No seconddevicejob.

- TERMINAL exit1: all12IDs exact, selectedLP failed257/513/1023/1024; other8exact, truehitsupto896/miss0, zero preemption; sameDP4/unevenDP4exit0. v28 12boundary+DPconcurrencyvalidation session `71558`; endpoint jcz-dsv41-mindleagent,existingservice50791/PID120440ownsNPUs. logs/prefix-boundaries-v28.log/json plusbatch4-dspark-v28/uneven-dp-dspark-v28 logs/json. Lengths127..1025warm/hit/miss,strictIDs/LP,actualhitandmiss0;thensameDP4anduneven4DP;inspectpreemptmetrics. Prior98707terminal0,nootherclients.

- TERMINAL exit0,bothprefixandacceptancestagespass; v28 sequentialhealth/prefix/acceptance client session `98707`; endpoint jcz-dsv41-mindleagent,existingservice50791/PID120440 ownsNPUs. Helperrun_service_checks_v2.py,logs/dspark-v28-clients.json,prefix-service-dspark-v28.json,acceptance-dspark-v28.json. Comparewarm/hit/missstrictIDs/LPonlynewsource;nooldv23paritybecausemathchanged. No competinginference.

- TERMINAL exit0 after verifiedPID120440 SIGTERM by2123; v28 unobservedproductionservice session `50791`,PID120440,sourcef6f3f49574611b347ae3c5367140651c8e8b9de724936fff1038337746d53c61; endpoint jcz-dsv41-mindleagent,all16NPUs/ports8127,13491,logs/service-dp4tp4-dspark-v28.log,artifacts/service-manifest-v28.json. DP4TP4eagerprobabilistic,128budget/4096/image0/.90;newfixed64floatinglinears+existingFP32heads/mHC. 270CPU/pairedNPU/realweights/fullCIpass. No observerworker. Launcher8packages/portsfree/upstreamclean beforedeploy.

- TERMINAL exit0;36mixedrowcasesand18graphreplays/FP64checkswithproductionhelperspass; NPU0 realweightproductionlinearvalidation session `30471`; endpoint jcz-dsv41-mindleagent,logs/early-linear-production-v1.log/json,validate_early_linear_production_v1.py. Usesactualproductionapply_linear/grouped_output_projection forrealwq_a/Engramwkv/wo_a andsame12row/changedgraph/FP64protocol. Prior16689terminal;launcher8packagesfree.

- TERMINAL exit0 ALLHOOKSPASS; local fullformat CI v5 session `61073`; logs/format-ci-v5.log,49taskfilesstagedincludingfixedrowhelper/tests/Chineserunbook. Checkformatterchangesbeforefingerprintandv28launch; no commit.

- TERMINAL exit0;baseline1fail1pass/candidate2pass4.46s; NPU0 pairedlinear regression session `16689`; endpoint jcz-dsv41-mindleagent,logs/linear-row-stability-v1.log/-baseline.log/-candidate.log/-receipt.json. Originaldirectlinear/einsum injectedonlypytestmodule vsproductionfixed64helper. Test2components,10countsfirst/middle/lastand3changedgraphs; prior77491CPU270pass,launcher8packagesfree.

- TERMINAL exit0,270passed5.11s,sourcef70e21e0; source-syncCPU regression session `77491`; endpoint jcz-dsv41-mindleagent,logs/cpu-linear-row-stability-v3.log,artifacts/source-fingerprint-cpu-linear-row-stability-v3.json. Parameterlessnativeprojectiondelegation+13CPUcases,explicitNPUimportinST;samefloatinglinearfix. Prior20785terminal;no NPUprocesses.

- TERMINAL exit1,6fail263pass:DSparkCPUidentitystubsno.weight; fixedhelperdelegatesparameterlessnativepath; source-syncCPU regression session `20785`; endpoint jcz-dsv41-mindleagent,logs/cpu-linear-row-stability-v2.log,artifacts/source-fingerprint-cpu-linear-row-stability-v2.json. AddsDSparkproject_context/attention/main_proj useofsamefloatinglinearhelper;NPUregressionusesrealdimensions. Prior48274CPU269pass,8packagesfreechecked.

- TERMINAL exit0,269passed4.55s,source89c54bc4; source-syncCPU regression session `48274`; endpoint jcz-dsv41-mindleagent,logs/cpu-linear-row-stability-v1.log,artifacts/source-fingerprint-cpu-linear-row-stability-v1.json. Candidate addsfixed64nativefloatinglinear/groupedprojection helper,attention/indexer/Engramwkv/compressor routing,12CPU+2NPUtests. Prior88507terminal,launcher8packagesfree;DSparkoverridecallersstilltoaudit.

- TERMINAL exit0;36fixedrowmixedcasesand18changedgraphreplayspass,baselinesvary; NPU0 expandedfixedrowlinearvalidation session `88507`; endpoint jcz-dsv41-mindleagent,logs/early-linear-row-stability-v1.log/json,scriptvalidate_early_linear_rows_v1.py. Prior5973terminal/all8packagesfreechecked. Realwq_a/Engramwkv/wo_a;12rowcounts1..257,first/middle/lastprobe amongst mixedrows,nativevsfixed64;6/24row3changedgraphreplays percomponent. No productionmutation yet.

- TERMINAL exit0;all3nativecapture/FP64checkscomplete,pad64andtiles64removeobserveddifferences; NPU0 earlylinears realcaptureprobe session `5973`; endpoint jcz-dsv41-mindleagent,logs/early-linears-v27.log/json,scriptdiagnose_early_linears_v27.py. Launcherchecksall8packagesfreeafterv27terminal. Replayswq_a/Engramwkvfulloutputs exactlyagainst110/112capture,comparesnative/pad64/tiles64,wo_a rank0preprojectioninputs andsampledCPUFP64. Diagnostic-only,production14dbcb3dunchanged.

- TERMINAL exit0,metricsretained/PID101720SIGTERM; v27 stop session `26300`; exactmanifestPID101720/command,savefinalmetricsandSIGTERM; reconcile8671anddevices beforeisolatedNPUprobe. Evidencepull56855alsoinflight.

- TERMINAL exit0,observerexactv23; layer0wo_b andEngramwkv earliestboundaries; v27 CPU-onlytraceanalysis session `64558`; endpoint jcz-dsv41-mindleagent,logs/dspark-trace-analysis-v27.log,artifacts/dspark-diagnostic-v27/comparisons.json andearly-full-comparison.json. Requiresobserverexactv23,comparesfullfirstlayer+Engramatcommonlogical128..333andlayer1KV. Service8671stillownsNPUs.

- TERMINAL exit0,4requestscomplete; v27 serializeddiagnostic client session `49796`; endpoint jcz-dsv41-mindleagent,existingservice8671/PID101720 owns16NPUs; logs/dspark-diagnostic-v27-client.log/json,artifacts/dspark-diagnostic-v27. Same4v23prompts,temp0/max16. Waithealth600s thenobserver-controlagainstv23andfullearlypositionanalysis.

- TERMINAL nativeexit0 afterverifiedPID101720SIGTERM; v27 full-early-layer observer service session `8671`,PID101720; endpoint jcz-dsv41-mindleagent,all16NPUs/ports8127,13491; logs/service-dp4tp4-dspark-v27.log,artifacts/service-manifest-v27.json. Sameproduction14dbcb3d and v26config. Worker preservesdummyguard plusfull layer0submodules/layer1Engram/attnnorm andlayer0preprojection; useoriginal334prompt tofollowpositions227/234..239. Launcherchecks8packages/portsfree/upstreamclean.

- TERMINAL exit0;PID86597SIGTERM,metricsretained; v26 exactmanifestPID stop session `48053`; verifyPID86597 command/ports,savefinalmetrics thenSIGTERM. Reconcile93451beforev27.

- TERMINAL exit0;shortenedinputdoesNOTreproduceoriginalhiddenstates; v26 shorter240token probe session `33278`; endpoint jcz-dsv41-mindleagent, existingservice93451; logs/dspark-early240-v26.log,logs/dspark-early240-v26-client.json,artifacts/dspark-diagnostic-v26/early240-comparison.json. Twofreshsaltwarm/hitrequests240prompt/max1;checksfull layer1inputs224..239 against original334tokenrecords beforeusingselected232..239layer0/engramhooks. No productionchange orrestart.

- TERMINAL exit0;observerexactv23,7earlierlayer1input/KVdifferences227/234..239; v26 CPU-onlytraceanalysis session `59922`; endpoint jcz-dsv41-mindleagent; logs/dspark-trace-analysis-v26.log,artifacts/dspark-diagnostic-v26/comparisons.json. Firstrequires3prefixoutputs IDs/LPexactunobservedv23,thenlogicalKV,fullinputpositions,preoutputprojection/wo_binput. Service93451remainslive,nootherNPUwork.

- TERMINAL exit0,4requestscomplete; v26 serializeddiagnostic client session `40417`; endpoint jcz-dsv41-mindleagent, existingservice93451/PID86597ownsNPUs; logs/dspark-diagnostic-v26-client.log/json, artifacts/dspark-diagnostic-v26. Same4v23prompts,warm/hit/miss/count,temp0,max16;healthwait600s. Analyzev26onlyafter4complete;mustmatchunobservedv23outputs beforeinterpreting.

- TERMINAL nativeexit0 afterverifiedPID86597SIGTERM; v26 corrected observer service session `93451`,PID86597; endpoint jcz-dsv41-mindleagent,16NPUs/ports8127,13491, logs/service-dp4tp4-dspark-v26.log,artifacts/service-manifest-v26.json. Same14dbcb3d production,DP4TP4eager,128budget,image0,.90,probabilistic. Worker v26 skipsdummyandcapturespreprojection/wo_binput/logicalKV;CPUguardpassed. Launcherrequiresall8packages/portsfree/upstreamclean.

- TERMINAL firstcleanup27206exit1 during parent shutdown; followup41798exit0 after same-starttick/session revalidation and exact3workerSIGKILL; v25 exact-PID cleanup session `27206`; endpoint jcz-dsv41-mindleagent; artifacts/service-v25-cleanup.json. Service69768 and orphan workers70750/71028/71325 verified same session69719, exactstartticks and v25 log; portsclosed. TERM then boundedgrace/KILL onlythesePIDs; reconcile47828andnpu-smi beforev26.

- TERMINAL exit1 after warm request; observer dummy metadata AttributeError; v25 diagnostic client session `11842`; endpoint jcz-dsv41-mindleagent,onlyexistingservice47828ownsNPUs; logs/dspark-diagnostic-v25-client.log/json,artifacts/dspark-diagnostic-v25. Same4serializedv23prompts,warm/hit/miss/count,max16/temp0;waithealth600s. Nextanalyzev25mustcheckobserverresponseIDs/LPagainstunobservedv23, thencomparelogicalwindowKV and layer1projectiontensors atpositions332/333.

- TERMINAL service_exit-15/nativeexit1; verifiedparentSIGTERM + 3orphanworkersSIGKILL cleanup; v25 layer1attention diagnostic service session `47828`; endpoint jcz-dsv41-mindleagent,all16NPUs/ports8127,13491; logs/service-dp4tp4-dspark-v25.log,artifacts/service-manifest-v25.json. Sameproduction14dbcb3d,DP4TP4/EP16eager,128budget/4096/image0/.90/probabilistic. Workerartifact dspark_diagnostic_worker_v25.py expands layer1attn projections/fullinputs/logicallyalignedwindowKV andexcludesphysicalcachemodules fromtokenrowhooks. Priorv24service49654terminal0/client54114terminal0/analysis55226terminal0; launchchecks8packages/portsfree/upstreamclean.

- TERMINAL exit0;observerexactunobservedv23;firstlogicaldiff layer1attention (ignoreunalignedphysicalcachehook); CPU-only v24 trace analysis session `55226`; endpoint jcz-dsv41-mindleagent, logs/dspark-trace-analysis-v24.log,artifacts/dspark-diagnostic-v24/comparisons.json. Requiresobserverwarm/hit/miss IDs+LPexacttounobservedv23 beforelocatinglogicalposition332..335 firstdifferinghook; recordsactualheadruntimeFP32. Existingv24servicestillownsNPUs,noconcurrentinference.

- TERMINAL exit0,4serializedrequestscomplete; v24 serialized diagnostic client session `54114`; endpoint jcz-dsv41-mindleagent,existingservice49654onlyNPUowner, logs/dspark-diagnostic-v24-client.log/json, capturesartifacts/dspark-diagnostic-v24. Waithealth600s,readv23sameprefix+countprompt,warm/hit/miss/counttemperature0,max16,minposlast2/max12steps; mustcompareobserverresponseswithunobservedv23beforeinterpretingboundaries. No competingrequests.

- TERMINAL nativeexit0 afterverifiedPID54846SIGTERM; v24 health200,actualtarget/draftweight+processorallFP32; DP4TP4/EP16 eager DSparkdiagnostic service session `49654`,PID54846; endpoint jcz-dsv41-mindleagent,all16NPUs,ports8127/13491,logservice-dp4tp4-dspark-v24.log,manifestartifacts/service-manifest-v24.json. Sameproductionsource14dbcb3d asv23,128budget/4096/image0/.90,probabilisticdraft. Optionaltaskownedworker dspark_diagnostic_worker_v24.py capturesrank0alllayerattn/ffn/engram+norms/residuals,layer0submodules,actualheadweight/processor dtypes. No productionobservercode. Priorv23terminal0 andclient8747terminal1;launchverifies8packages/portsfree.

- TERMINAL exit1 aggregate(prefix1,acceptance0); v23 sequential health/prefix/acceptance client session `8747`; endpoint jcz-dsv41-mindleagent, existingservice47494ownsNPUs. Helperrun_service_checks_v2.py,logs/dspark-v23-clients.json;strictwarm/hit/missIDs+selectedLP then3acceptancesmokes. No v16comparison: FP32head/mHCchangedmath requiresnew same-sourceeagerbaseline. No competingrequests.

- TERMINAL nativeexit0 afterverifiedPID39353SIGTERM; prefixfailedselectedLP,acceptancesmokespassed; v23 DP4TP4/EP16 DSparkprobabilistic eager service session `47494`,PID39353,source14dbcb3dbea61a45ffa4bbcfcd249e0d9c49d8686c3ec6a71c9bc10db78f6bb1; endpoint jcz-dsv41-mindleagent,ownsall16NPUs,HTTP8127/RPC13491, logs/service-dp4tp4-dspark-v23.log,artifacts/service-manifest-v23.json. SourceFP32heads+padded64mHC+SP/CP/decodefixes,257CPU+pairedNPU2+realweight26pass,fullCIv4pass. max_model_len4096/tokenbudget128(actual112expected)/maxseq4/image0/gpuutil.90; intentionallyrevisitold110/94prefixdivergence without128alignmentworkaround. LaunchchecksNPU/portsfree/upstreamclean. Readmanifestbeforeclients.

- TERMINAL exit0 ALLHOOKSPASS; local fullformat CI v4 session `92454`; logs/format-ci-v4.log,46files stagedinclFP32head/draftCP/mHCcode/NPUST/Chineseupdates; precommit4.0.1/tasklocallintbinaries. No commit.

- TERMINAL exit0;26candidatecases1..2048bitinvariant/allFP64boundpass,oldcaptured94/110reproducedexact; NPU0 productionmHC realweight confirmation session `24273`; endpoint jcz-dsv41-mindleagent, logs/mhc-row-stability-real-v1.log/json, validate_mhc_real_weights.py. Oldsnapshotbaselinecaptured110/94mustreproduceexact;candidate26same/mixedcases1..2048 mustpreservecoeff/preFFNbitinvariance andoriginalFP64gate. Prior62204baseline2fail/candidate2pass,launcher8packagesfree.

- TERMINAL exit0;baseline2fail4.39s/candidate2pass4.58s; NPU0 paired mHC regression session `62204`; endpoint jcz-dsv41-mindleagent, logs/mhc-row-stability-v1.log, -baseline.log/-candidate.log/-receipt.json. Oldfunction immutable snapshotSHAe57cda11148d injectedonlyinto pytestmodule,expected2fail; productioncandidate ST2seeds,10rowcounts,first/middle/lastrow and3changedgraphs exact. Prior4554CPU257pass,launcher8packagesfree.

- TERMINAL exit0,257pass4.64s; CPU candidate mHC row-stability regression session `4554`; endpoint jcz-dsv41-mindleagent, logs/cpu-mhc-row-stability-v1.log/sourcefingerprint,46taskfiles sync. Candidate fixed64F.linear and64alignedvectorops,7additionalCPUshape/FP64cases and2newNPUSTcases pending. PriorTP8terminal0,8packagesfreechecked.

- TERMINAL exit0;all8ranks3layersprefill/decodegraphCPoff/onexact; TP8 HCCL realattentionCP component session `61476`; endpoint jcz-dsv41-mindleagent,exclusiveNPU0..7/reservedallpackages, logs/cp-tp8-v1.log and cp-tp8-v1-rank0..7.json. Same immutable helperas29509,priorTP4terminal0,launcher8packagesfree. Layers0/2/20,CPoff/on,realW8A8prefill9+3changeddecodegraphs; nofullserviceclaim.

- TERMINAL exit0;all4ranks3layersprefill/decodegraphCPoff/onexact; TP4 HCCL realattentionCP component session `29509`; endpoint jcz-dsv41-mindleagent,exclusiveNPU0..3/reservedallpackages, logs/cp-tp4-v1.log and cp-tp4-v1-rank0..3.json, validate_attention_cp_distributed.py. Prior67247terminal0,launcher8packagesfree. Layers0/2/20,realW8A8 TPshards,CPoff/on,prefill9andchangeddecode9/10/11NPUGraph,original.04/.04. Not nativefullmodelloader/service.

- TERMINAL exit0;12realweightcasesFP32/eager/graphexact,CPUFP64boundpass; NPU0 real-weight FP32 head regression session `67247`; endpoint jcz-dsv41-mindleagent, logs/fp32-heads-npu-v1.log/json, validate_fp32_heads_npu.py. Prior73460terminal0,8packagesfreechecked. NativeHFdtype/_apply_head usescompleteTP4target/draftweightshards; CPUFP64spread129channels,eager1/6/24rows,3changedNPUGraphreplays; componentonly,no distributed/serviceclaim.

- TERMINAL exit0;26paddedlinear64casesallbitinvariantthrough2048/allFP64gatepass; NPU0 mHC padded-linear diagnostic session `73460`; endpoint jcz-dsv41-mindleagent, logs/mhc-padded-linear-v1.log/json, script diagnose_mhc_padded_linear.py. Prior58926terminal0,88505CPUdeployterminal0/250pass. Fixed64F.linear withallvectorops paddedtomultiple64;13rowcounts through2048,mixedotherrows,sameFP64gate/capturedbaseline. No production substitution.

- TERMINAL exit0,250pass4.58s; CPU source-sync regression session `88505`; endpoint jcz-dsv41-mindleagent, deployed /vllm-workspace/vllm-ascend, logs/cpu-fp32-sp-cp-sync-v1.log/sourcefingerprint. 45taskfiles SHAverified,SP/CP/draftCP/FP32head candidate. PriorNPU58926terminal0,launchchecks8packagesfree/upstreamclean. No modelservice.

- TERMINAL exit0 diagnostic;native andbmm64 stillbatchdependent,inspectlinear64; NPU0 mHC projection-only fixed64 diagnostic session `58926`; endpoint jcz-dsv41-mindleagent, logs/mhc-fixed-gemm-v1.log/json, script diagnose_mhc_fixed_gemm.py. Prior30275terminal0, launcherrechecks8packagesfree. Compare native,looped64projection,bmm64projection;13rowcounts through2048,mixedotherrows,unchangedremainingmHCmath/FP64tolerance. No production substitution.

- TERMINAL exit0:54cases;fixed16/64 invariantacross9rowcounts/mixedothers,nativevaries,captured94/110reproducedexact,allFP64tolerancepass; NPU0 fixed-row mHC diagnostic session `30275`; endpoint jcz-dsv41-mindleagent, exclusive existing reservation/all8packagesfree verified21715 and launchrecheck; scripts/artifact diagnose_mhc_fixed_tiles.py, logs/mhc-fixed-tiles-v1.log/json. v22service85748terminal0/GPQA69825terminal1 andall3ownedPIDsabsent/portsclosed. Reproduces immutable v15warm110/hit94 preFFN; comparesunchangednative/fixed16/fixed64 at9rowcounts and mixedotherrows againstCPUFP64; no production substitution.

- TERMINAL reconciled from immutable logs (native handle expired); baseline2fail/candidate250pass5.16s; CPU-only FP32 head dtype paired regression session `4646`; endpoint jcz-dsv41-mindleagent, isolated cpu-validation-fp32-head-v1, logs/cpu-fp32-head-v1-baseline.log/logs/cpu-fp32-head-v1.log, artifacts/cpu-isolated-fp32-head-v1.json. Real v15 captured rawdraftlogits are BF16 (observerpreservesdtype); pinnedofficial ParallelHead FP32 andactual mtp.2.head.weight F32. Rootcause native LogitsProcessor head_dtype defaults modelBF16 andcasts evenFP32input/weight. Candidate sets native HFhead_dtype=float32 on text+composite configs and retains target head weights inFP32. Baseline2 native _apply_head tests expectedfail, candidate250suite. No live sync; after confirmedregression, preserve/abort oldsource fullGPQA and restart finalvalidation withfix rather thanaccept olddtype score.

- TERMINAL exit0,baseline4fail/candidate247pass4.88s; CPU-only DSpark+CP paired regression session `28388`; endpoint jcz-dsv41-mindleagent, isolated cpu-validation-draft-cp-v1, logs/cpu-draft-cp-v1-baseline.log/logs/cpu-draft-cp-v1.log, artifacts/cpu-isolated-draft-cp-v1.json. Audit found inherited CP fullhead sinks with draft override still partialqueryheads. Candidate gathers projected heads, partitions already-computed global5token draft blockkeys/requests, restores output token/head order. Four new TP4/8 x firstposition2/7 threaded CPU dense-oracle cases cover2requests+rejectedtailrewrites; deployedpreCPdraft control expected4fail, candidate247suite. No production writes or NPU execution; GPQA69825 continues.

- TERMINAL exit0 ALLHOOKSPASS; local full format CI v3 session `31184`; logs/format-ci-v3.log. All45taskfiles staged after MoESP/DSA-CP additions and Chinese runbook GPQA/SP checkpoint. Uses precommit4.0.1 and task-local verified lintbinaries. No commit/push/production sync; reconcile formatter changes before nextsourcefingerprint.

- TERMINAL exit0,243passed/4.62s; CPU-only DSA-CP query partition regression session `10126`; endpoint jcz-dsv41-mindleagent, isolated cpu-validation-cp-queries-v1, logs/cpu-cp-queries-v1.log, artifacts/cpu-isolated-cp-queries-v1.json. Candidate partitions projected full-head queries/index scores acrossTP; replicated cache writes, TPweightshards unchanged; gathers tokenoutputs and restoresheadshards. New18tensor-layout cases +6 real threaded CPU TP4/8 collectives vs independent TP1full-head paged reference, ratio0/1/2 and sharedKV/index/candidate, two requests incl idlepadded ranks. No production writes or NPUwork; fullGPQA69825 continues onv22.

- TERMINAL wrapperexit1/AISBenchexit-15 after intentionalSIGTERM;12/198 formalcompleted,no finalscore; old-headruninvalidated, abortreceipt artifacts/gpqa-full-v22-v3-abort.json; full198 GPQA Diamond session `69825`, AISBenchPID19248; endpoint jcz-dsv41-mindleagent, existing servicev22/PID505680/session85748 exclusively owns16NPUs. Helper artifacts/validation/run_gpqa_audited_v3.py, launchlog logs/gpqa-full-v22-v3-launch.log, benchlog logs/gpqa-full-v22-v3.log, receipt logs/gpqa-full-v22-v3-receipt.json, response metadata logs/gpqa-full-v22-v3-responses.jsonl, outputs eval-tooling/outputs/gpqa-full-v22-v3. Immutable fullv2 config copied to full_audited_v3 with ONLY hostport8127->8128 local byte-preserving proxy; unchanged145000output,temp1/top1/seed17/thinkingmax,batch16. Proxy retains finish/stop reasons and usage without changing response bytes. No competing inference/device clients. Read receipt/PID before any retry.

- TERMINAL exit0; baseline12fail/candidate219pass (2.62s), CPU-only SP partition paired regression session `45998`; endpoint jcz-dsv41-mindleagent, isolated /workspace/dsv41-mindie-20261109/cpu-validation-sp-partition-v1. Newcandidate DeepseekV41MoE shards full DP-local hidden+image_mask through native sequence_parallel_chunk when SPenabled, then sp_all_gather/truncate restores fullorder. Baseline deployedv22module only temporarily substituted insideisolatedcopy for12newregressioncases (mustfail), thenrestorecandidate/full219suite. logs/cpu-sp-partition-v1-baseline.log and cpu-sp-partition-v1.log,sourcefingerprint/artifactsmanifest. No productionwrites,NPUexecution,or SPfunctionalclaim yet.

- TERMINAL exit0,206passed/2.53s; CPU-only isolated post-lint v2 session `9778`; endpoint jcz-dsv41-mindleagent,/workspace/dsv41-mindie-20261109/cpu-validation-post-lint-v2; source2e6ec2a0d336353c244c9b0f429b8608a8cc0d4d81185179c8a6a11ae8250d3a,45files,206suite. logs/cpu-post-lint-v2.log,artifacts/cpu-isolated-post-lint-v2.json. Correctedoriginparse; v1preflightterminal,noCPU/NPUloadcompeting exceptsmallCPUregressionwhileGPQAruns. Liveproductionunchanged.

- TERMINAL exit1 beforepytest: import-origin output had normalplugin logprefix, correctcandidatepath confirmed but naive startswithassertfailed; helpernowcompareslastline. CPU-only isolated post-lint regression session `45149`; endpoint jcz-dsv41-mindleagent, repo /workspace/dsv41-mindie-20261109/cpu-validation-post-lint-v1 builtfromauthorizedbasegitarchive+45taskfiles; compiledextensionsread-onlysymlinks, importoriginverified. No deployment/source/devicemutation. logs/cpu-post-lint-v1.log,artifacts/cpu-isolated-post-lint-v1.json,artifacts/source-fingerprint-cpu-post-lint-v1.json. Same206suite. Source-sync helpers nowcollectstaged+unstaged+untracked(importantafterstaging); no accidentalzerosourcefuturelaunch.

- TERMINAL exit0 ALLHOOKSPASS; local fullformat CI v2 session `34360`; logs/format-ci-v2.log, all45files staged, precommit4.0.1, task-local verifiedDarwinbinariesgitleaks8.30.1/shellcheck0.11.0 (artifacts/lint-tools/receipt.json). Five lint-onlylocalfileschanged: sortedplatformimport,configdocstring,regeximportsloader/model, testhashreprreplacespickle. No runtime sync tolivev22;206CPUrecheckneededafterservice.

- TERMINAL wrapperexit1 due .json/.jsonl artifactglob; AISBench exit0,3/3correct, zero failures/preemptions; correctedGPQA3smoke client session `61939`, AISBenchPID12465; endpoint jcz-dsv41-mindleagent,servicev22/PID505680; logs/gpqa-smoke-v22-v2.log,logs/gpqa-smoke-v22-v2-receipt.json,eval-tooling/outputs/gpqa-smoke-v22-v2. New immutableconfigs retry=1 (=oneHTTPattempt),smokeSHAf5403f03...,fullSHAaf1ab78d...; outputbudget145000,temp1/top1/seed17/thinkingmaxunchanged. Prior64200terminal,NO competingrequests. Helpernowrequiresprediction/resultartifacts inadditiontoCLIexit; finalrowcount/qualitystillmustcheck.

- TERMINAL exit1: ruffsortedoneimport; forbidden re/pickle imports andconfigdocstringfixedlocally; missingmacOSgitleaks/shellcheckbeinginstalled. local fullformat CI session `93001`; taskworktree45changedfiles staged for staged-sensitive lintchecks, no commit. uv tool run --from pre-commit==4.0.1 bash format.sh ci, log logs/format-ci-v1.log; beforehash receipt artifacts/pre-format-v1.json. Inspect/reconcile automaticchanges; remotev22unchanged.

- TERMINAL wrapperexit0 but INVALID/NO PREDICTIONS; zero-attempt retry=0 skippedHTTP, warmupfailed immediately (AISBench base_api.py450 range(self.retry)); v22 AISBench GPQA Diamond first3smoke session `64200`, AISBench PID11219; endpoint jcz-dsv41-mindleagent, existingservice85748/PID505680 allNPUs; logs/gpqa-smoke-v22.log and logs/gpqa-smoke-v22-receipt.json; output eval-tooling/outputs/gpqa-smoke-v22. PinnedAISBench525ed177,configSHA8beab35...,145000out,temp1/top_p1/seed17/thinkingmax/retry0. 4707terminal0, no competingrequests. Do not duplicate; readreceipt/process on resume.

- TERMINAL exit0 dependenciescached; local lint environment preparation session `17538`; uv tool run --from pre-commit==4.0.1 pre-commit install-hooks in taskworktree. Onlyhookdependencycache setup; no git hook installation/sourceformat/commit.

- TERMINAL exit0 all3stagespassed; v22 health/strictprefix/graph-v16eagerparity/acceptance client session `4707`; endpoint jcz-dsv41-mindleagent,existingservice85748/PID505680 ownsallNPUs;receipt logs/dspark-v22-clients.json; waitshealth<=1200s andchecksprocessliveness. No concurrentrequests.

- TERMINAL nativeexit0 after verifiedPID505680 SIGTERM; oldBF16headinvalidatedforfinalprecision; v22 (health200; all4DP capturefinished2.09–2.10GiB,KV2.92GiB/preflight4.67x perDP, capacityunproved) DSpark probabilistic text FULL_DECODE_ONLY long-context service session `85748`, PID505680, sourcebda2123c1aaf9ee5cd4ac294360e11166d5af8b8c25d3c46ce414a80afaf9749; endpoint jcz-dsv41-mindleagent,all16NPUs,HTTP8127/RPC13491,DP4TP4/EP16,max_model_len153600,tokenbudget144,max_num_seqs4,capture6/12/24,image_limit0,gpu_memory_utilization0.90,cudagraphmetrics. v21service44816terminal0; read17580confirmedPIDabsent/portsclosed/8packagesno processes, launchrechecks. Log logs/service-dp4tp4-dspark-v22.log,manifest artifacts/service-manifest-v22.json. No seconddevicejob.

- TERMINAL exit0; all12boundary warm/hit/miss exacttokenIDs+selectedLP,hitup to896/miss0,sameDP4andunevenDP4passed; v21 prefix-boundary and DP clients session `1057`; endpoint jcz-dsv41-mindleagent, existingservice44816/PID468896 exclusivelyownsNPUs; 127/128/129,255/256/257,511/512/513,1023/1024/1025 token warm/hit/saltedmiss strictIDs+selectedLP,then sameDP4/unevenDP4. logs/prefix-boundaries-v21.log/json,batch4-dspark-graph-v21,uneven-dp-dspark-graph-v21. No competingclient.

- TERMINAL exit0; all4stages passed (prefix128hit,graph-v16eager token/LPexact,acceptance,4Visioncolors/order); v21 sequential HTTP client session `95449`; endpoint jcz-dsv41-mindleagent, logs/dspark-v21-clients.json beforehealthwait; existingservice44816onlydeviceowner. Strictprefix,graph-v16eagerparity,3acceptancecases,4single/twoimagecolors/orderchecks; logs prefix-service-dspark-v21,dspark-graph-parity-v21,acceptance-dspark-v21,vision-chat-v21.

- TERMINAL exit0 after verifiedPID468896 SIGTERM (helper70769exit0); v21 real-weight DSpark+Vision FULL_DECODE_ONLY service session `44816`, PID468896, sourcebda2123c1aaf9ee5cd4ac294360e11166d5af8b8c25d3c46ce414a80afaf9749; endpoint jcz-dsv41-mindleagent,all16NPUs,HTTP8127/RPC13491,DP4TP4/EP16/4096,144budget,capture6/12/24,image_limit2,gpu_memory_utilization0.96,cudagraphmetrics. Same runtimecode asv20 (runbook onlychanged); v20profile implies nonKV60.63GiB,0.96budget61.44GiB gives~0.81GiB KV beforegraph. Log logs/service-dp4tp4-dspark-v21.log,manifest artifacts/service-manifest-v21.json. v20terminal,8packagesfree confirmed86777; launch checksagain.

- TERMINAL exit1 before requests;receiptservice_exited_before_health; v20 sequential HTTP checks session `63255`; endpoint jcz-dsv41-mindleagent, logs/dspark-v20-clients.json receipt created before health wait. Existing v20service exclusively ownsNPUs. Oncehealthy runsstrictprefix,graph-v16eagerparity,3acceptancecases,4single/two-imagecolorcontrast/orderrequests; logs prefix-service-dspark-v20,dspark-graph-parity-v20,acceptance-dspark-v20,vision-chat-v20. No competing clients/profiling.

- TERMINAL exit1;weights+profiling passed alias/Dynamo stages,KVmemorybudget negative-3.03GiB at0.90util; nohealth/capture; real-weight DSpark+Vision FULL_DECODE_ONLY service v20 session `34415`, PID454090; endpoint jcz-dsv41-mindleagent, all16NPUs,HTTP8127/RPC13491,DP4TP4/EP16/4096,144budget,capture6/12/24,image_limit2,defaultworker,cudagraphmetrics enabled. Log logs/service-dp4tp4-dspark-v20.log,manifest artifacts/service-manifest-v20.json; source7fe0a7e1f7a879f6e4d75913db09dfe5af037c8123c718ab3a8a758ac826b82d. PreviousNPU12907terminal; launch checksNPU/ports/upstream.

- TERMINAL exit0;3images exactnativeofficial encoder+aligner;maxsquareincrement5033336320bytes/1.3356s,maxwide3x3063increment5663094784bytes/1.5304s,bothfinite; NPU0 production memory-bounded vision regression session `12907`; endpoint jcz-dsv41-mindleagent, logs/vision-production-sdpa-npu-v3.log/json and vision-production-sdpa-memory-v3.json. Source7fe0a7e1f7a879f6e4d75913db09dfe5af037c8123c718ab3a8a758ac826b82d,206CPU pass; no attention substitution;3images including nonsolidpattern, nativeofficial bit-exact comparison;maxsquare+maxwide memory probes.

- TERMINAL exit0,206passed; CPU full regression session `41053`; endpoint jcz-dsv41-mindleagent, logs/cpu-vision-memory-mm-alias-v3.log, artifacts/source-fingerprint-cpu-vision-memory-mm-alias-v3.json. Candidate implements in-place FP32 math-SDPA, image_token_index configalias and image-disabled text routing fastpath; tests added. Prior34476terminal, no modelservice.

- TERMINAL exit0;both nativeoraclecasesexact;peak6052282880bytes/increment5033336320bytes,1.3612s for93x93patches; NPU0 in-place FP32 softmax SDPA diagnostic session `34476`; endpoint jcz-dsv41-mindleagent, logs/vision-sdpa-inplace-npu-v1.log/json and vision-sdpa-inplace-memory-v1.json. Validation-only substitution matches upstream PyTorch2.10mathSDPA scaling/dtypes, reuses complete scorebuffer with softmaxout; same nativeofficial equality+maximum-square memory probe. Production unchanged; v19terminal,8packagesfree verified.

- TERMINAL exit0;198prompts113–2595tokens,145000output fits153600;bothmmengineconfigs load; CPU-only corrected GPQA prompt-budget/config parser preflight session `40624`; endpoint jcz-dsv41-mindleagent, logs/gpqa-configuration-preflight-v2.log, artifacts/gpqa-configuration-preflight-v2.json. Render text then encode tokenIDs explicitly, assertmin>40; reads/compares existing config without rewriting; validates mmengine Config load. No modelHTTP/NPU.

- TERMINAL exit0 but prompt-budget measurement INVALID:len(BatchEncoding)counted2fields, not tokens; config creation/body check valid, needs corrected budget; CPU-only GPQA config/prompt preflight session `97006`; endpoint jcz-dsv41-mindleagent, logs/gpqa-configuration-preflight-v1.log, artifacts/gpqa-configuration-preflight.json; pinnedAISBench isolatedvenv,198prompts budgetchecked,temperature1/top_p1,thinkingmax,output145000. Creates full/smoke configs, noHTTP requests/NPU work.

- TERMINAL exit1 before requests,service exited before health; v19 HTTP client session `88653`; endpoint jcz-dsv41-mindleagent, waits existing PID441630health<=900s then strictprefix, graph-v16eager parity,3acceptancecases sequentially; logs/dspark-v19-clients.json and prefix-service-dspark-v19,dspark-graph-parity-v19,acceptance-dspark-v19 logs/JSON. Service2600 ownsdevices, no competing clients.

- TERMINAL exit1; startup failed loading draft after target became multimodal:target text config lacks image_token_index alias; real-weight DSpark greedy FULL_DECODE_ONLY service v19 session `2600`, PID441630; endpoint jcz-dsv41-mindleagent, all16NPUs,HTTP8127/RPC13491,DP4TP4/EP16/4096,144budget/capture6,12,24,image_limit0,defaultworker. Log logs/service-dp4tp4-dspark-v19.log, manifest artifacts/service-manifest-v19.json; source f549d68ecda10ddcd0b01c5e0616f0c2e0f5dc0d9b27f2c9a949975f79b6ef24. All priorNPUjobs terminal, launch rechecks8packages/portsfree.

- TERMINAL exit0; both native-oraclecases exact;93x93patch memory peak12,073,346,560bytes(increment11,054,400,000),2.4165s; NPU0 production3DSDPA real-weight regression and maximum-square memory probe session `50170`; endpoint jcz-dsv41-mindleagent, logs/vision-production-sdpa-npu-v2.log/json and vision-production-sdpa-memory-v2.json. Source f549d68ecda10ddcd0b01c5e0616f0c2e0f5dc0d9b27f2c9a949975f79b6ef24,201CPU pass, previousdevicejobs terminal. Fixed nativeoracle threshold retained plus exact-value comparison; CPU drift retained in report.

- TERMINAL exit0; install and ais_bench --help pass; protected host versions unchanged; CPU-only AISBench install/help validation session `34998`; endpoint jcz-dsv41-mindleagent, logs/aisbench-install-v1.log, receipt artifacts/aisbench-install.json; installs only into eval-tooling/venv after successful dry-run, exact torch/torch-npu/transformers pinned. No evaluation request.

- TERMINAL exit0,201passed; CPU full adaptation/DSpark/modelrunner/chattemplate regression session `40999`; endpoint jcz-dsv41-mindleagent, logs/cpu-vision-sdpa-chat-v2.log, source manifest artifacts/source-fingerprint-cpu-vision-sdpa-chat-v2.json. Production Vision now official3DSDPA; new bidirectional/head-layout regression; allsource SHAverified during sync. NPU test80010terminal.

- TERMINAL exit0; resolution plan preserves torch/torch-npu/transformers; CPU-only AISBench isolated-venv dependency dry-run session `22619`; endpoint jcz-dsv41-mindleagent, logs/aisbench-install-dry-run-v1.log, report artifacts/aisbench-pip-plan.json. Venv eval-tooling/venv with inherited host packages, constraints pin exact installed torch/torch-npu/transformers. No package installation or evaluation requests yet.

- TERMINAL exit1 for head-chunk arm; full3DSDPAexactlymatches officialNPU vision+aligner on bothimages, headchunkrectangularNRMSE0.12947/0.05697 failed; NPU0 full-SDPA vs head-chunk-SDPA diagnostic session `80010`; endpoint jcz-dsv41-mindleagent, logs/vision-sdpa-heads-npu-v1.log/json, script artifacts/validation/validate_vision_head_sdpa_npu.py. Same real weights/square+rectangular images and fixed thresholds, official unmodified NPU oracle. Only validation substitution, production unchanged. 84895 reconciled terminal;8packagesfree checked.

- TERMINAL preflight receipt verified; AISBench source/download session handle lost at compaction, no matching process; source SHA525ed1779f50a8ac068b8638771f633a151e769f, GPQA Diamond198rows, ZIP SHA3ea7acd5c550280b428524b33ffe8f5998c9fcfda2250e0b97a12e9c9e78b40f. Endpoint jcz-dsv41-mindleagent, CPU-only; manifest artifacts/aisbench-preflight.json, log logs/aisbench-source-preflight-v1.log. Dependencies not installed; no evaluation requests. Recovery77883 verified all8NPU packages empty.

- TERMINAL failed numerical assertion (native handle lost; remote traceback and absent PID437216 reconciled); rectangular native-NPU visionNRMSE0.16362/aligner0.07387, fixed threshold unmet; isolated NPU0 Vision query-chunk SDPA experiment session `84895`, logs/vision-sdpa-chunk-npu-v1.log/json. Temporary attention substitution only in validation process, no productionchange; compare unchangedofficialNPU fullSDPA with same realweights and fixed2%/0.999 threshold. Prior14689terminal.

- TERMINAL exit1; rectangular still differs withofficialnativeNPU SDPA (visionNRMSE0.2285,aligner0.0979); official-native NPU Vision comparison session `14689`, logs/vision-official-native-npu-v1.log/json. Same realweights and twoimages; compare commonFIA to unmodified official ViT onNPU (SDPA), retaining CPUfailure measurements. Metrics accumulate CPUFP64 toavoid unreliablelargevectorFP32cosine. Prior54584/45775terminal; no service.

- TERMINAL exit0,56exactcases; CPU chat template vs pinnedofficial encoding session `13880`, logs/chat-template-official-v1.log;56 text/image/system/multi-turn/thinking-effort cases.

- TERMINAL exit0 diagnostic only; same rectangular failure reproduced; first >2% relative boundary isblock31(17%),teacher-forced sameinput block31error0.44%; finalCPU/NPUencoder34%,aligner20%; real-weight Vision boundary diagnostic NPU0 session `45775`, logs/vision-boundaries-npu-v1.log/json. Same source and square→rectangular sequence; per-block CPU/NPU captures and teacher-forced earliest >2% boundary. Diagnostic metrics only; original gate unchanged. v1 oracleterminal.

- TERMINAL exit1; square passes (visionNRMSE0.01317,aligner0.01045), rectangular fails visionNRMSE0.34443/cos0.94076; real-weight Vision/aligner NPU0 vs pinned-official CPU oracle session `54584`, logs/vision-real-weights-npu-v1.log/json. All8packagesfree checked before launch; exclusive iTASK reservation, no service. Two image sizes, full32-layer BF16 tower, common Ascend FIA; fixed acceptance cosine>=0.999 andnormalizedRMSE<=0.02 beforemeasure.

- TERMINAL exit0; actual one/two-image spans184/202 match expected placeholders; CPU MM processor integration v2 session `88759`, logs/vision-processor-integration-v2.log. Uses installed ProcessorInputs/TimingContext API; same production source.

- TERMINAL exit1 in probe API usage beforeprocessing; CPU actual MM registry/processor smoke session `41746`, logs/vision-processor-integration-v1.log. EngineArgs + real tokenizer + PIL image; no model/NPU inference.

- TERMINAL exit0,193passed; full CPU adaptation/common DSpark/model_runner regressions session `21155`, logs/cpu-vision-draft-graph-v1.log. Source fingerprint artifacts/source-fingerprint-cpu-vision-draft-graph-v1.json; v18 fully terminal and8NPU packages empty checked before sync. Candidate adds MM integration/router, dummy raw-token contract, and disables unsupported draft Dynamo compilation while retaining target graph.

- TERMINAL exit1 before requests; service exited before health; v18 HTTP client session `24565`, waits service PID419288 health then strict prefix, DSpark eager-v16 parity, and three acceptance requests; logs/dspark-v18-clients.json and stage logs. No competing device work.

- TERMINAL exit0 after verified SIGTERM419288 by38153; startup FAILED during draft Dynamo guard serialization (live ZMQ socket unpickleable); v18 DSpark greedy FULL_DECODE_ONLY service session `45874`, DP4TP4/EP16/4096,tokenbudget144,capture sizes6/12/24; logs/service-dp4tp4-dspark-v18.log,artifacts/service-manifest-v18.json. Launcher verifies8packages/portsfree andsource hashes; 40sourcefiles include two unreferenced processorfiles. v17terminal.

- TERMINAL exit0,4passed; CPU Vision processor UT session `8310`, logs/vision-processor-ut-v1.log. Only two unreferenced new processor/test files synced and SHA verified; v17 runtime imports unchanged, no NPU work.

- TERMINAL exit0 v17 probabilistic HTTP client session `29328`, healthdeadline600s then3cases attemperature1/top_p1/seeds41..43; logs/acceptance-dspark-prob-v17.log/json. Service92131ownsall16devices; no competingrequests. 249/380accepted(65.53%),perpos69/55/49/44/32,zero preemption; statistical distribution equivalence is not established by3smokes.

- TERMINAL exit0 after verified PID404439 SIGTERM by12279; v17 probabilistic DSpark eager service session `92131`, logs/service-dp4tp4-dspark-v17.log. Task-local launch_dspark_service.py verifies NPU/ports/upstream andsyncs+hashesall38sources before launch; only sourcechangefromv16 is runbookupdate. DP4TP4/EP16/4096,144budget/128scheduled,5drafttokens,draft_sample_method=probabilistic. Sourcefingerprint artifacts/source-fingerprint-dspark-service-v17.json; remote manifest artifacts/service-manifest-v17.json. v16terminal; do notduplicate.

- TERMINAL exit1 aggregate; v16 client session `22345`: strictprefixexit0(hit128/miss0,token+selectedLPexact), eager-v4compareexit1(firsttokenLPexact,decodeLPmaxabs0.04084866),3acceptancecasesexit0(233/340,perpos61/55/45/40/32,no preemption). All acceptance counts identical to observer-v15. logs/prefix-service-dspark-v16,dspark-eager-parity-v16,acceptance-dspark-v16 (.log/.json),dspark-v16-clients.json. Broader speculative/non-speculative numerical equivalence remains unresolved.

- TERMINAL exit0 after verifiedPID390033 SIGTERM bysession8487; unobserved v16 service session `12065`, logs/service-dp4tp4-dspark-v16.log. Defaultworker, sourcea74f4cb31de286b812555ff43292a34322dcd2cf31c393e244a4ada4e14744c7,DP4TP4/EP16/4096,5greedydrafts,maxbatched144/scheduled128. Prefixstrictpass and68.53%acceptance; nondraftbaselineLPstilldiffers. Reconcile terminal before newNPUrun.

- TERMINAL exit0; NPU0 batch-shape probe v2 session `59712`, logs/mhc-batch-shape-v2.log/json. CPU-before-FP64 reference valid, rawFP32maxabs1.9e-6..3.43e-5. Both warm110/hit94 first FFN normalized tensors reconstructed bit-exact using same embedding/attention andcheckpoint mHCweights, with no KV/MoE. Earliest captured divergence arises in batch-shape-dependent mHC FP32 arithmetic/BF16rounding; arbitrary-batch numerical invariance remains unresolved.

- TERMINAL exit0, captured warm110/hit94 pre-FFN tensors both reconstructed bit-exact without KV/MoE; isolated NPU0 probe session `65411`, logs/mhc-batch-shape-v1.log/json. Sixbatchsizes yield different FP32mHC reductions and BF16norm values. Caveat: FP64 reference conversion mistakenly ran onNPU first, warning downgraded tofloat; v1 raw_fp64 field is invalid as FP64 evidence. Correct CPU-before-double v2 follows.

- TERMINAL exit0, all3acceptancecases completed; v15 clients session `7370`:233/340drafttokensaccepted(68.53%),perpos[61,55,45,40,32],no preemption. Prefix LP failure unchanged. Layer0 FFN-normalized input atpos333 differs7BF16elements(max0.001953125), beforeexpert execution. Physical cache-page snapshots are not logically aligned and excluded from numerical boundary comparison. logs/dspark-diagnostic-v15-client,dspark-trace-analysis-v15,acceptance-dspark-v15 (.log/.json).

- TERMINAL exit0 after verified PID374345 SIGTERM by9569; v15 real-weight DSpark eager service session `9573`, logs/service-dp4tp4-dspark-v15.log; DP4TP4/EP16/4096,5greedy drafts,ports8127/13491. Startuplogs confirm distinct draft embedding/head retained;126UTpass. Source a74f4cb31de286b812555ff43292a34322dcd2cf31c393e244a4ada4e14744c7. Detailed rank0 observer captures layer0 submodules and Engram1.

- TERMINAL exit0,126passed; CPU full adaptation/common DSpark UT session `61177`, logs/dspark-vocab-sharing-after-ut.log. Updated load_weights sets own-vocab flags only after strict load; new real proposer-helper regression. v14 terminal;8 packages/ports free verified before source sync.

- TERMINAL exit1 in result summarizer; CPU old-source regression v2 session `10532` failed intended embedding identity assertion; trace analysis itself exit0. Summarizer hit non-JSON import warning and did not reach stop. logs/dspark-vocab-sharing-regression-before-v2.log,dspark-trace-analysis-v14.log. Recovery session `89100` skips non-JSON log lines and verifies PID359617 exact command before SIGTERM; no duplicate experiment.

- TERMINAL exit1, test setup lacked vllm_config after both unwanted substitutions occurred; CPU regression-before session `46133`, logs/dspark-vocab-sharing-regression-before.log. Test stub corrected before v2; this exit is not yet the intended identity assertion.

- TERMINAL exit0; v14 observer HTTP client session `54037`. Warm/hit/miss selected LP exactly reproduce v13 failure; count output1..6 in16tokens. logs/dspark-diagnostic-v14-client.log/json; control removed, rank0 tensors artifacts/dspark-diagnostic-v14/. Observer did not remove the numerical failure.

- TERMINAL exit0 after verified PID359617 SIGTERM by recovery89100; observer-only DSpark eager service v14 session `27967`, logs/service-dp4tp4-dspark-v14.log. Same v13 production digest d4904db3023fcfbd436ea65f9445eca56d7ebb1fea34ee72f892106eec26de58 and DP4TP4/EP16/4096. Prefix failure unchanged by observer. Earliest captured hit/warm difference at position333 is layer0 FFN(maxabs0.00390625), position332 at Engram1(maxabs0.0009765625); same warm/miss all recorded layers exactly equal. Source and startup logs prove independent draft vocab weights incorrectly replaced by target weights; fix and focused regression in progress.

- TERMINAL exit0 requestscompleted;22/1385drafttokensaccepted,onlyfirstposition;prefix128hits butlogprobmismatch; v13 HTTP client session `6743`: health200confirmed, then prefixwarm/hit/miss and3DSparkacceptancecases sequentially. logs/prefix-service-dspark-v13(.log/.json),acceptance-dspark-v13(.log/.json). No competing clients; service41268 ownsall16devices.

- TERMINAL exit0,7gridplans+pixelpatches+3BF16ViT/alignercasesexact; CPU Vision oracle v3 session `64620`, logs/vision-official-cpu-v3.log; same production,CPUattentionstubusesofficial3DSDPAranktoisolatekernelrounding; exactassertionsunchanged.

- TERMINAL exit1;7gridplans/pixels exact; tinyViT32/96differ,max0.0078125 with4Dvs3DSDPA; CPU pinned-official Vision oracle v2 session `52564`, logs/vision-official-cpu-v2.log; same comparisons,now officialvision+imageprocessorfilesuploadedandSHAprinted. NoNPU.

- TERMINAL exit1 beforecomparison; official vision.py was not yet uploaded; CPU pinned-official Vision oracle session `76533`, logs/vision-official-cpu-v1.log.7gridplans,exactpixelpreprocessing,3smallBF16ViT+alignercases; noNPU; v13servicealoneownsdevices.

- TERMINAL exit0 after verified PID332680 SIGTERM; real-weight DSpark eager service v13 session `41268`, logs/service-dp4tp4-dspark-v13.log. SameDP4TP4/EP16,4096,5draftgreedy,ports8127/13491; separate384expert target/128expert draft MoE communication contexts.125UTpass. Source d4904db3023fcfbd436ea65f9445eca56d7ebb1fea34ee72f892106eec26de58,38sourcefiles+receiptverified,upstreamclean,NPUportsfree. Requests complete, correctness unresolved.

- TERMINAL exit0,125passed; CPU all-adaptation/commonDSpark/Vision UT session `9973`, logs/dspark-comm-vision-ut-v1.log. Per-model communication context snapshot/restore added; v12terminal,no newservice. All38changed/newfiles syncedandSHAverified.

- TERMINAL exit1,9passed/1failed; exact layout check accidentally included strided GELU rounding (1.19e-7); now isolates pre-GELU layout; CPU-only Vision primitive UT session `51436`, logs/vision-components-ut-v1.log. Only2newunreferencedvisionfiles synced; livev12service source unchanged. Vision not registered or enabled yet.

- TERMINAL exit1; target MoE AlltoAll split count mismatch after draft replaces global communication config; real-weight DSpark eager service v12 session `20185`, logs/service-dp4tp4-dspark-v12.log; DP4TP4/EP16,4096,HTTP8127/RPC13491,5drafttokens/greedy. New subclass compile decorator;13targetedUT passed. Source digest e2604b99833c94329716d1d73ba36d6ae94b39bed7d3c046ac5a1e7ecd4b1569. Launch checks upstreamclean,NPUs/portsfree,allsourceSHA. Do not duplicate.

- TERMINAL exit0,13passed; CPU constructor/model UT v2 session `61373`, logs/dspark-constructor-ut-v2.log; mock unrelated logits processor to isolate real inherited compile call contract. No live service or NPU process.

- TERMINAL exit1,12passed/1failed; test setup missing logits-processor config context; CPU draft constructor/model regression session `72167`, logs/dspark-constructor-ut.log; adds same support_torch_compile decorator to V4.1 constructor and actual class-call regression. Priorv11service exited1; no newservice yet.

- TERMINAL exit1 after all16 loaded49.6317GiB/rank; missing inherited do_not_compile in draft `__call__`; real-weight DSpark eager service v11 session `91912`, DP4TP4/EP16,4096,HTTP8127/RPC13491,5drafttokens,greedy draft sampling, explicit draft indexed loader. Log logs/service-dp4tp4-dspark-v11.log; source fingerprint befa0dbc61c95eea0b3d1f427fbff3f088ded64d4c2f7cdc5e9acb43249d8caf. Launch script verifies36sourcehashes, upstreamcleanSHA, allNPU/portsfree. No duplicate service permitted.

- TERMINAL exit0,113passed; CPU DSpark/all-adaptation/common-proposer UT v2 session `19932`, logs/dspark-ut-v2.log; includes Eagle layer-number translation regression and circular key order.

- TERMINAL exit0,9casespass; DSpark attention NPU v3 session `56476`, logs/dspark-attention-npu-v3.log. Official attention structure retained; CPU oracle now shares actual NPU activation quantizer only, independently CPU GEMM plus FP64 channel checks; unchanged0.04 tolerance. Earlier seven INT8 half-boundary discrepancies documented.

- TERMINAL exit0; seven INT8 ties differ, sameinteger/scales CPU accumulation exactlymatches NPU; single NPU0 dynamic INT8 quantization boundary probe session `53101`; local logs/dspark-quant-probe-v1.log; saved captured q_norm input from83690, compare quantized integers/scales/FP64-checked accumulators. Priorprobe terminal.

- TERMINAL exit1 with complete boundary evidence; DSpark first-boundary probe session `83690`; logs/dspark-boundaries-v1.log, artifacts/dspark-boundaries-v1.pt; same stage0 realweights/seed412 as failingv2, capture Q/KV/selectedkeys/sparseoutput/projection boundaries. NPU0 only, priorv2terminal.

- TERMINAL exit1 stage0 maxabs0.326171875; DSpark attention NPU v2 session `1141`, logs/dspark-attention-npu-v2.log; same9-case real-weight official oracle and unchanged0.04 tolerances, key ordering corrected. v1terminal.

- TERMINAL exit1 stage0 maxabs0.3125; single NPU0 real MTP attention vs pinned HF DSparkAttention, session `16252`; logs/dspark-attention-npu-v1.log. All prior service/test terminal, NPU/ports free verified. Three draft layers, changing accepted context and draft tail; no whole-model claim.

- TERMINAL exit0, correct V4.1 draft architecture and128experts/3topk; CPU EngineArgs DSpark configuration probe session `8325`; local logs/dspark-config-v1.log. No service or device work launched.

- TERMINAL exit0, 112 passed; CPU-only DSpark + all V4.1 adaptation + common DSpark proposer UT session `14280`; logs/dspark-ut-v1.log. 36 source files verified local/remote. v10 service terminal; NPU/ports empty confirmed before these changes.

- TERMINAL exit0, both4-request tests passed and repeated prefix/logprob parity exact; v10 batch4 sameDP, unevenDP and repeated prefix/eager parity sequence session `72758`; logs/batch4-graph-v10, dp-concurrent-graph-v10, prefix-service-graph-v10-repeat, graph-eager-parity-v10-repeat (.log/.json). Same existing v10 service; no other requests/profiling.

- TERMINAL exit0, warm/hit/miss tokenIDs andselectedLPexactly equal eager-v4, hit256/miss0; uninstrumented v10 prefix+eager-parity client session `37539`: waits health up to120s then once runs warm/hit/miss at334tokens pinnedDP0, then strictselectedLP comparison to eager-v4. logs/prefix-service-graph-v10.log/json and graph-eager-parity-v10.log/json. No competing requests.

- TERMINAL exit0 after verified PID268072 SIGTERM; uninstrumented real-weight graph v10 service session `65505`, DP4TP4 EP16/4096/FULL_DECODE_ONLY capture[1,2,4], default NPUWorker (no observer), ports8127/13491. All previous runs terminal, processes/NPU/ports empty verified. Source fingerprint artifacts/source-fingerprint-service-v10.json SHA99e9c6a28ed50f9095be3f50461e696e1912bb105e3ecb346fdcf66b6d65a3f0; all33 files local/remote SHA match, upstream6e448d0 clean. Log logs/service-dp4tp4-graph-v10.log.

- TERMINAL exit0, 69passed; all adaptation CPU UT session `86658`: platformconfig + worker test_deepseek_v41_*.py, logs/v41-all-ut-scalar-fix.log. No model service running.

- TERMINAL exit0; original control failed481/512 maxabs0.109375, candidateall5passed; paired r7 session `1709`, changes pollution from torch.bitwise_right_shift/left_shift APIs to exact >>/<< syntax used by standalone failing repro; other conditionssame. logs/fp8-scalar-regression-before-r7.log, after-r7.log, pair-r7.json.

- TERMINAL exit0 both pass with torch.bitwise API; paired r6 session `72816`, exact scalar repro conditions: sourcecopy once, same1/128-row integer input, retain unrelated eager output, isolated right/left shift replays; inference_mode/internalTrue. Upload+SHAverification and pair in same connection. logs/fp8-scalar-regression-before-r6.log, after-r6.log, pair-r6.json.

- TERMINAL exit0 both passed, scenario still differed; paired r5 session `25758`: single connection first uploads+hash verifies inference_mode test, then original-round control and candidate5 tests sequentially. logs/fp8-scalar-regression-before-r5.log, after-r5.log, pair-r5.json. Prior r4 terminal.

- TERMINAL exit0 both pass with stale r3 test; paired r4 session `6470`, but test upload27087 failed on login-user503, so this run may still use r3 test without inference_mode. Do not count as new regression evidence; reconcile completion before retry.

- TERMINAL exit0 control/candidate passed; paired regression r3 session `66260` (27128 exited1 before launch on login-user503; confirmed no r3 logs/process before retry): now explicitly internal_format=True with native-value restore. Original saved round function control then clean-process all5 tests candidate. logs/fp8-scalar-regression-before-r3.log, after-r3.log, pair-r3.json.

- TERMINAL exit0 both tests passed with internal_format defaultFalse; paired regression session `10876`: original round_e4m3 loaded from our saved pre-fix source only in test process; then clean process full5-case NPU file on candidate. logs/fp8-scalar-regression-before-r2.log, after-r2.log, pair.json. SingleNPU0, earlier tests terminal.

- TERMINAL exit1 same test option setup bug; post-fix full NPU quantization test file session `92347` (5cases), math SHA256373c91517882b6a0d70eac51b5ea9e87581f74c8637872f54c2d09b98833f51b verified. logs/fp8-scalar-regression-after.log. Source fingerprint artifacts/source-fingerprint-scalar-fix.json (e09dfcf4e8689d5797b093341116c2a39a805235c3c7b16e5d6e2794a11226dd).

- TERMINAL exit1 test setup failed before numerical check (write-only NPU option cannot monkeypatch); pre-fix regression session `82607`, singleNPU0, only new test uploaded while remote attention_math stillold. logs/fp8-scalar-regression-before.log.

- TERMINAL exit1, reproduced scalar contamination; scalar-cache v6 session `71804`: singleNPU0 after v5 exit0; captureFP8 then eager bitwiseAND/right-shift with different scalars/rows before replay. logs/fp8-graph-scalar-v6.log.

- TERMINAL exit0 all12 exact; dummy+internal-format v5 session `92690`: singleNPU0 after v4exit0; reuse v3 with allow_internal_format=True. Log logs/fp8-graph-dummy-internal-v5.log.

- TERMINAL exit0 all24 exact; single-NPU0 internal-format v4 session `66499`: reuse minimal v1 unchanged, set allow_internal_format=True after imports (matching model_runner_v1 global). v9 terminal/processes/ports/NPU absence verified. Log logs/fp8-graph-internal-format-v4.log.

- TERMINAL exit0 v9 A/B/A client session `96017`, health200 checked; six graph/eager/graph warm/hit cases identical334-token baseline. logs/observed-graph-eager-v9.log/json. No concurrent requests.

- TERMINAL exit0 after verified PID237976 SIGTERM; v9 whole-model observer session `68913`, DP4TP4 EP16, 4096 FULL_DECODE_ONLY[1], QuantDiagnosticWorker. Prior v3 exited0, all NPU processes absent and ports8127/13491 closed verified. Records raw/quantized/written windowKV for all40layers plus v8 boundaries. Log logs/service-dp4tp4-diagnostic-v9.log; artifacts/graph-diagnostic-v9.

- TERMINAL exit0, all12 cases strict exact; dummy FP8 graph v3 session `69346`, single NPU0 after v2 exit0; inference_mode/zero input/invalid slots/shared graph pool then changing inputs/metadata and interleaved prefill. Log logs/fp8-graph-dummy-v3.log.

- TERMINAL exit0, all12 layer/step cases exact (246 per-layer checks); FP8 lifetime v2, session `35991`: ownership/task ID confirmed, all16 NPUs empty; single NPU0,1/40 layers with eager128-token prefill interleaved. Script SHA2564364a5c599a63ecc6de72af23f6f1eeb81dac218711d8cc115f88c7ffeacfbd0 verified remote. Log logs/fp8-graph-lifetime-v2.log.

- TERMINAL exit0, all24 strict cases passed; single-NPU0 FP8 graph reproducer; all16 modelworkers gone/portsclosed/NPU empty verified after v8. Tests pure quant and strided paged consumer, rows1/16/128 and4 changedinputs. Log logs/fp8-graph-reproducer-v1.log. Session `45016`.

- TERMINAL exit0 v8 observed A/B/A client: v8health200 confirmed, same six graph/eager/graph warm/hit cases; logs/observed-graph-eager-v8.log/json; captures artifacts/graph-diagnostic-v8; session `55974`. Do not submit competing requests.

- TERMINAL exit0 after verified PID213367 SIGTERM; diagnostic v8 full16NPU launch; priorv7 exit0 and allworker/port/NPU processes clear. Same DP4TP4 EP16 4096 FULL_DECODE_ONLY[1], DetailedDiagnosticWorker adds layer0 query/selectedKV/valid/sparse_output/wo_b_input to observer. Log logs/service-dp4tp4-diagnostic-v8.log; captures artifacts/graph-diagnostic-v8. Session `63229`.

- TERMINAL exit0, six cases completed and464 tensor snapshots captured; observed graph/eager/graph A/B/A client on v7 health200, six cases warm/hit; logs/observed-graph-eager-v7.log/json; captures artifacts/graph-diagnostic-v7/*.pt on16ranks. No concurrent requests. Client session `36234`.

- TERMINAL exit0 after verified PID190796 SIGTERM; diagnostic v7 launch: DP4TP4 EP16 real weights,4096/FULL_DECODE_ONLY capture[1], DiagnosticWorker in artifacts/validation (production source unchanged). Port8127/RPC13491. Priorv6 parent exit0/process list empty/NPU no processes and portsclosed confirmed. Log logs/service-dp4tp4-diagnostic-v7.log. Session `95730`.

- TERMINAL exit0 session `86009`: decode vs teacher-forced prefill diagnostic on existing v6; logs/decode-teacher-v6.log/json; two modes each warm+hit, same334-token input and fixed firstgeneratedtoken25.

- TERMINAL exit0, all16 analyse.done present; offline profile export: started with torch_npu.profiler.profiler.analyse(root, max_process_number=4), CPU-only parsing of completed16rank capture; log logs/profile-export-v6.log. Session `43144`. No new profiling capture.

- TERMINAL client_exit0, stop_profile receipt confirmed and handle missing; NPU profiling client session `51801`; existing v6 server owns all16 NPUs, calls start_profile/one real text request/stop_profile; `logs/profile-graph-v6.log`, receipt/response/metrics same prefix, traces under `profiles/service-graph-v6`. Do not start another profiler or submit competing benchmark.

- TERMINAL exit0, all4 correct: same-DP batch4 graph test session `79990`; existing v6 server; `logs/batch4-graph-v6.log/json`. Four18-token requests pinned DP0, otherDP idle.

- TERMINAL exit0, token parity passed but logprob delta remains: graph prefix test session `88629`, existing v6 server health200; `logs/prefix-service-graph-v6.log/json`, same334-token input as eager. Do not duplicate.

- TERMINAL exit0 after verified PID151255 SIGTERM; graph service v6 shell session `8171`, original iTASK all16 NPUs, HTTP8127/DP RPC13491; `logs/service-dp4tp4-graph-v6.log`; same v5 FULL_DECODE_ONLY [1,2,4]/4096/DP4TP4/EP16 options, now backend update capability fix. Profiler dir `profiles/service-graph-v6` (one completed capture; not currently profiling). All old processes/ports/NPUs empty immediately before launch. Source fingerprint `artifacts/source-fingerprint-service-v6.json`. Do not duplicate.

- TERMINAL exit0, **91 passed**: CPU regressions session `40687`: paged cache + full model_runner_v1 + ACLGraph unit files, log `logs/graph-backend-update-ut.log`. No live model service or NPU process after v5 failure.

- TERMINAL exit1, first request HTTP500: graph prefix/parity session `35014`, existing v5 server; `logs/prefix-service-graph-v5.log/json`. V5 health200 verified, capture completed; replay/output parity still under test.

- TERMINAL exit1 after first decode request: graph service v5 shell session `19968`; original iTASK all16 NPUs/8127/13491; `logs/service-dp4tp4-graph-v5.log`. Same DP4TP4/EP16 real weights/4096; remove enforce-eager, compilation mode0 + FULL_DECODE_ONLY capture sizes[1,2,4]; cudagraph metrics and torch NPU profiler configured (not started). Previous v4 exited0, processes/NPU/ports verified free. Do not duplicate.

- TERMINAL exit0, all4 correct: uneven DP smoke session `36713`; existing v4 service, requests pinned ranks0–3 with different prompt lengths; `logs/dp-concurrent-v4.log/json`.

- TERMINAL exit0, parity passed: prefix test session `29361` on existing v4 server; `logs/prefix-service-v4.log/json`. Official Chinese prompt >256 tokens, warm/hit/salted miss pinned DP0, token IDs + hit metrics. Do not duplicate.

- TERMINAL exit0, HTTP200 and correct answer `4`: HTTP smoke session `16378` targeting existing v4 server; log `logs/text-smoke-v4.log`, result `logs/text-smoke-v4.json`. Health200 confirmed; official encoder chat-mode 2+2 prompt, max32/temp0. Not a GPQA run.

- TERMINAL exit0, **67 passed**: CPU-only all adaptation UT session `27290`, log `/workspace/dsv41-mindie-20261109/logs/v41-all-ut-round5.log`; no NPU workload.

- TERMINAL exit0 after intentional SIGTERM: service-dp4tp4-v4 shell session `71889`, original iTASK/all16 NPUs, HTTP8127/DP RPC13491; log `/workspace/dsv41-mindie-20261109/logs/service-dp4tp4-v4.log`. History cache name fix, same DP4TP4/EP16 eager4096 command. Previous processes/ports/NPUs checked free. Do not duplicate.
- CPU model/cache binding session96816 TERMINAL exit0, 5 passed, `logs/model-cache-binding-ut.log`.

- TERMINAL exit1: service-dp4tp4-v3 shell session `91729`, original iTASK/all16 NPUs, HTTP127.0.0.1:8127/DP RPC13491; log `/workspace/dsv41-mindie-20261109/logs/service-dp4tp4-v3.log`. Same DP4TP4/EP16 eager4096 command as v2. Ports/processes/NPU use rechecked empty before launch. Do not duplicate this run.
- CPU regression session4626 completed exit0, 9 passed; log `logs/engram-contiguous-ut.log`.
- service-dp4tp4-v2/session91817 TERMINATED exit1 after **all16 ranks loaded real backbone weights**; log `/workspace/dsv41-mindie-20261109/logs/service-dp4tp4-v2.log`, each rank reported48.2738GB and EP24/384 experts. Warmup failed at HCCL all_gather because selecting one Engram layer from the hash tensor produced a strided view. Added `.contiguous()` at collective input and a regression. Previous ports8127/13491 and all16 NPUs must be rechecked before next launch.
- session87903 completed remote_exit0: V4.1 model UT plus existing ModelSlim regression, **54 passed**; log `logs/model-quant-config-ut-v2.log`.
- service-dp4tp4-v1/session96837 is TERMINAL exit1, before real weight loading. Cause: selector-only HF quantization metadata prevented sidecar loading. Fix and regression are under test. Do not treat its old PID71327 or ports8127/13491 as a live service without checking processes.
- All earlier round4 jobs are terminal: session74770 (64 UT pass); session68895 (11 UT, 4 NPU quantization cases, actual attention layers0/2/20 graph pass); config probe9497 passed; attention v1/v2/v3 failed then v4 passed. Diagnostics20618/77561/23437 and scalar probe47498 completed. All previous round3 and Engram jobs completed.
- No managed remote-dev/coordinator job exists. Native iTASK long-running actions use shell sessions plus process/log evidence. Before restarting, inspect existing handles, process ownership, ports and NPU use.

## 环境证据（2026-11-09，北京时间）

- iTASK：`jcz-dsv41-mindleagent`，任务 ID `d686c12e-f94f-43a1-8f6e-bbd760832a8e`，所有者 `jcz02615514`，Running。
- Pod：`itask-jcz-dsv41-mindleagent-jcz02615514-20261109231555-j6knf`。
- 节点 `10.30.12.22`，HostNetwork=true；128 CPU、1600Gi 主机内存、16 个 A3 逻辑设备（8 双 die 封装）。
- 每设备 HBM 65536 MiB，预检约占 2.9–3.2 GiB；npu-smi 未发现运行中的 NPU 进程。
- 镜像 `v0.27.1rc1-a3-openeuler-20261102112527_aarch64`。
- vllm-ascend `0.27.1rc1`，editable `/vllm-workspace/vllm-ascend`，SHA `3b31886237ed65c435e0965b10e8002ff5769766`，干净 detached HEAD。
- vllm `0.27.1+empty`，editable `/vllm-workspace/vllm`，SHA `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`，tag v0.27.1，干净 detached HEAD。
- torch `2.10.0+cpu`、torch-npu `2.10.0.post4`、transformers `5.14.1`；CANN `9.1.0`（V100R001C11SPC001B243）；npu-smi `25.5.1.1`。
- ais-bench distribution 和 ais_bench 命令未发现，需后续安装/定位；远端没有 rg、ss，使用 Python/其他现有命令进行等价检查。
- `/workspace` 存在；任务默认 cwd 为 `/a3_inference/itask/workdir/jcz02615514/jcz-afd1`，后续专用日志目录不可覆盖其中其他任务数据。

## 模型证据

- `/home/admin/model` 挂载 `DeepSeek-V4.1-Flash-1017 W8A8`，与 jcz-offical-dsv41 使用完全相同模型镜像。
- architectures=`DeepseekV41ForCausalLM`，model_type=`deepseek_v41`，nested text_config 与 vision_config；当前 vLLM 和 transformers 注册表均未发现 V4.1。
- 40 目标层、384 routed experts、每 token 6 experts；sqrtsoftplus 路由（Ascend FusedMoE 路由已有支持）。
- compress_ratios 为 0/2/1；KV source=[2,8,14,20]；index source=[2,8,14,20,24,28,32,36]；candidate source=20。
- 现有 DSV4 compressor Python backend 和自定义算子 README 只支持 ratio 4/128，不能直接处理 V4.1 ratio 2。
- Engram 位于层1、14；表的 INT8 weight shape 分别 [384006168,256]、[384016682,256]，F8_E8M0 scale shape 对应 [N,8]。
- engram_rotation_config：value_projection_rotated=true，value_basis=quarot_global，key_and_gate_basis=original，runtime_delta_rotation=false；optional/quarot.safetensors 存在，不能重复旋转。
- checkpoint 包含 vision/aligner/image 特殊 embedding、mtp.0/1/2 与 mtp.2.markov_head，未发现权重缺失；所有索引 keys 均能在指定 shard header 找到。
- tokenizer_config 没有 chat_template，模型目录没有 preprocessor_config；须按官方 encoding/image_processor 接入。
- DSpark block_size=5，target layers=[37,38,39]，3 draft 层、128 draft experts、每 token 3 experts。
- 量化配方含 quarot max_tp_size=4，需检查其实际含义及 TP8 运行时重分片，不直接推断 TP8 不可用。

### 权重容量（按索引引用的 safetensors headers 实际 data_offsets 汇总）

| 类别 | GiB |
| --- | ---: |
| Routed experts | 507.363 |
| MTP experts | 12.684 |
| Engram | 189.419 |
| 其他 | 16.712 |
| Vision / aligner | 0.904 |
| 总计 | 727.082 |

- 索引 metadata.total_size=577940550344 bytes，与实际 header 总和不一致；容量分析应使用真实 tensor headers，后续验证 loader 实际占用。
- 有约0.586GiB未被索引引用的 Engram tensor，不计入上述总量。
- 若 Engram 仅 TP4 分片，每 rank 约47.35GiB，再叠加 EP16 专家约32.5GiB，超出64GiB设备；需要跨更大组分片或主机卸载，且验证 Async/graph 的行为与性能。
- 不能由 aggregate HBM 大于权重体积推出16×150k一定可行；还需核算重复权重、KV/index/压缩状态、workspace与graph buffers。

## 对照任务

- `jcz-offical-dsv41`：用户先前授权必要时用于对照；最新限制下只可黑盒运行评测，不查询或参考已有 DSV4.1 实现。当前停止执行。
- 当前 Stopped，任务 ID `9e852394-81ae-4c12-b2d0-a1d2df869e56`。
- 镜像 `deepseek-v4.1-flash-a3-20261104105535_aarch64`，128CPU/1600Gi/16A3，绑定节点与目标任务相同。
- 本助手尚未启动该任务。以下状态为预检时记录；用户恢复后若需黑盒对照，再核实资源，避免与目标16设备占用冲突。

## MindIE 状态和恢复

- 用户最后发送纯文本 `/mindie-agent activate` 后，native UserPromptSubmit 授权成功。
- bridge.py activate 已返回 active；knowledge_cli.py describe 和 knowledge_query 已实际成功。不要再要求用户重复授权。
- runtime adapter：`/Users/jiangchenzhou/.config/mindie-agent/codex.json`；从此 JSON 读取 python/runtime_scripts，禁止记录或显示激活 token。
- 插件版本0.1.23；五个 hook 信任记录已保存。之前失败因为技能链接不匹配必须出现在消息开头的纯文本命令。
- 社区共享未配置、capture disabled；知识查询可正常使用。可选 onboarding 不应打断适配，不擅自开启共享或发送 Issue/PR。
- 先前查询曾命中 DSV4.1 相关知识条目导航，但尚未加载正文或应用。该读取计划已撤销，不再展开这些条目；后续只查询不含已有 DSV4.1 适配实现的通用知识。
- MindIE 会话授权不等于允许继续任务，也不扩大获准参考来源；本次用户已明确恢复适配。

## 参考来源和可复用代码

- 官方模型说明：<https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/README.md>
- 官方 config 与 inference/model.py、engram.py、image_processor.py 已读取。
- 官方 README 推荐 temperature=1.0/top_p=1.0、1M context、max_tokens≥256K；本任务容量不同，必须记录评测输出预算和截断。
- base 内 DSV4 实现位于 `vllm_ascend/models/deepseek_v4/{model,compressor,indexer,dspark}.py`，不是当前旧工作区的扁平文件结构。
- base 已有 DSV4 FusedMoE、sqrtsoftplus、DSpark proposer、DSA/flashcomm/shared-expert 配置；其中 enable_dsa_cp 会自动启用SP，并依赖索引器和EP/TP条件，开关非完全独立。
- 其他分支及历史适配已从可参考来源中排除；先前读取事实见“约束变更审计”，不得再检索或使用其内容。
- 原工作区 `/Users/jiangchenzhou/code/vllm-ascend` 在 `codex/inferrsi-dsv41-profiles` 有用户未提交文件，必须保留。
- origin 是 `https://github.com/jiangkuaixue123/vllm-ascend.git`，upstream 配为 `https://github.com/chopper0126/vllm-ascend.git`；push 只能到 origin。

## 恢复后的下一步

1. 重读最新来源限制及新工作树 AGENTS.md；仅比较指定基线 DSV4 与 Hugging Face 官方 V4.1 实现，不读取其他分支或已有适配。
2. 核实 A3 soc、精确端口与已运行进程，建立远端专用任务日志目录和源码分支。
3. 使用 MindIE 通用能力和通用知识，核实 CANN/Ascend 算子接口及 shape 约束；禁止查询知识库中的 DSV4.1 适配代码或实现方案。
4. 设计 Engram 内存布局、KV/index跨层共享、prefix/DSpark状态管理及多模态接入，不改上游vLLM。
5. 按 PLAN.md 阶段逐项实施，每次实验及时写入本文件与日志。

## 2026-11-10 恢复记录

- 当前目标授权恢复完整适配；只读取指定独立工作树和官方 HF 实现。
- 独立工作树 HEAD 仍为 3b31886237ed65c435e0965b10e8002ff5769766，工作树干净。
- itask describe 证实原 task ID 未变、Running、所有者 jcz02615514、16 个 A3 逻辑设备。

- 远端两仓库 SHA 与计划一致且初始干净；npu-smi 未发现 NPU 进程。通过原 iTASK 所有权检查；未启动服务。
- 已获取官方 HF 七个文件并固定 revision/sha256 到 `artifacts/official-hf/source-manifest.json`。resolve 下载曾不完整，改用同一官方 revision 的 raw 入口成功；未用外部实现来源。
- MindIE 查询限定 `vllm-ascend/architecture` 通用输入准备/异步/图契约；没有检索 DSV4.1 适配。结果提醒静态图缓冲地址和设备侧接受 token 更新；尚未作为实机已验证规则。
- 远端创建 `codex/dsv41-flash-0271rc1-mindie`，正在实现和测试嵌套配置注册及压缩 KV/index/cache 归属规划。尚无模型实现注册、真实权重推理或完成声明。

- 第一批变更：嵌套 config 注册（保留 W8A8 和 Engram 旋转元数据）、独立 KV/index/candidate cache 归属计划、无请求可变状态的 Engram 哈希模块。
- 从官方 `inference/engram.py` 在 CPU 生成小规模参考 fixture，覆盖两请求、图像边界、prefill+decode；fixture 生成脚本位于 `artifacts/validation/generate_engram_fixture.py`，fixture 已加入 UT。19 项 CPU UT 全通过，日志 `config-cache-engram-ut.log`。这不证明 scheduler prefix cache/DSpark 集成已完成。
- 真实 `/home/admin/model` 通过 vLLM `get_config(..., trust_remote_code=False)` 加载为 DeepseekV41Config/DeepseekV41TextConfig，40 层，quant_method=ascend。
- 本地新 Python 文件 ruff check / format 通过；尚未做完整 format.sh ci。

### 官方差异与下一步设计依据

- 官方 `inference/model.py:429`：ratio2 无重叠、无 APE，按通道对组内 token 做 softmax 加权，末尾不足一组须跨 chunk 保留；ratio1 仅投影+norm。不能直接套用基线 ratio4/128 compressor。
- 官方 `model.py:488,614`：压缩 latent 先生成 index K，再 RoPE/量化写 compressed KV。KV 源2/8/14/20同时拥有 index K；24/28/32/36仅有索引 query/权重，不创建新的 index K 缓存。
- 官方候选筛选在 index source20产生候选，但20本身及复用其top-k的21–23层不经过候选筛选；从index source24开始使用。缓存归属计划已修正并添加精确期望，不能把层21当成首个候选筛选层。
- 官方 `model.py:909,1186`：mHC 使用前一子层传来的 pre_mix；初始one-hot，attention产出的pre用于FFN，FFN产出的pre传给下一层；Engram先更新残差再计算mHC。基线 `DeepseekV4DecoderLayer` 的 fused hc_pre当前即时混合不等价，不能直接复用forward。
- DSpark读目标层37/38/39的attention输入均值，不是输出；后续对齐proposer、Markov head及拒绝回滚。此项尚未实现。
- Engram哈希采用显式 newest-first 压缩token窗口，不保存请求可变状态。下一步必须接入按请求/位置寻址的历史缓存，解决prefix hit、chunk边界、图像DEAD标志、异步输入和spec拒绝；当前原语测试不能替代这些集成验收。
- Engram大表仍需跨16rank分片或受验证卸载方案；不能只按TP4分片。下一步先检查实际W8A8 Engram scale/投影键和通用分布式lookup契约，再实现加载与通信，不读取其他V4.1实现。

### 已验证的边界

- 实际tokenizer长度129280，压缩词表99092；完整token map逐项与固定官方代码一致。
- 实际两张Engram表的bucket和为384006168 / 384016682，和模型配置一致。
- A3逻辑设备0（原iTASK独占16逻辑设备的保留内）完成完整24列哈希 eager、NPUGraph捕获与两次回放；第二次变更输入，结果仍与官方oracle一致。日志明确 `model_validation=false`。
- CANN提示内部格式/32 padding分配warning；没有错误，尚未做性能分析，不宣称吞吐或CPU同步开销已达标。
- 剩余阶段4–9均未完成；原验收规模、精度阈值和交付范围未改变。

- 最终源码8文件本地/远端SHA256一致，见 `artifacts/verified-source-state.json`；上游vLLM仍为指定SHA且干净。三份验证日志已取回本地 `logs/`。尚未commit/push。

## 2026-11-10 第二轮：Engram加载与分片

- 上轮分类为progress：已改变源码并取得21UT与A3图验证证据；本轮重新核实工作树与远端任务，没有遗留模型服务。
- 真实checkpoint量化说明明确 `layers.{1,14}.engram.embed.quant_type=INT8_SYM`；weight为I8，scale为F8_E8M0，[N,256]/[N,8]，每32通道一个scale。wkv/q_weight/k_weight均FLOAT/BF16；不是将I8字节解释成FP8。
- 实现 `ShardedEngramEmbedding`：分片slice加载、有限host staging、E8M0 scale按需反量化、EP-wide固定形状all_gather/reduce_scatter。未接入完整模型loader/forward，不宣称全表容量验证完成。
- MindIE仅检索通用architecture域的HCCL/DP/图padding契约；复核固定形状、idle rank共同参与和设备侧状态更新约束。
- 新增8项CPU UT均通过，包含INT8符号、非整除/空分片、loader不读取完整表和错误编码拒绝；日志 `embedding-ut.log`。

- 16rank真实行切片试验session80914退出0：两层eager、graph和变化输入回放均与CPU原生E8M0反量化精确一致，含idle rank15。开始全表加载测试session27934，约189GiB跨16设备，两表共同驻留验证；尚未完成不宣称容量通过。
- 新增分页历史读写原语及5项测试；与已有Engram UT合计9项通过（session13367）。覆盖非连续页、共享prefix整块、图像标记、spec拒绝覆盖、padding及块回收；尚未接入真实scheduler。

- EP16完整表验证完成：两张表共同驻留，rank0 torch已分配11.804696GiB（仅表与验证缓冲，不代表整模型），两层eager/graph/变化输入回放与CPU原生E8M0值精确一致。所测rank0行非零元素249–255/256，排除了全零样本假通过。`full_table_loaded=true`，`model_validation=false`。
- 分页历史NPU图连续4次回放通过：输入位置、slot和内容发生变化，覆盖跨物理页/图像标记/请求隔离/拒绝尾覆盖/padding；`scheduler_integration=false`。
- 新增Engram residual层，复用分片embedding、BF16 wkv和q/k gate；明确使用原始basis计算gate，checkpoint已旋转value不重复旋转。signed-permutation基变换UT与mask/正负gate UT通过。
- 全部新增UT共37项通过，ruff check/format与git diff --check通过。真实gate验证进行中，还未纳入完整模型。

- gate验证第一次session68073退出1：测试用float32 linspace生成约3.84亿行索引发生舍入越界（47行而非48行），并非模型实现报错。改为整数 `i*(n-1)//47` 后session3495重试，旧日志保留。

- gate验证v2/session3495失败：`aten::copysign.out`发生CPU fallback，且4/40960输出元素超出既定rtol=0.02/atol=0.02（最大失败点abs0.0390625）。已改用FP32符号位+where保留copysign含负零的语义；v3/session46549分开比较CPU/NPU投影与反旋转中间值，未放宽容差。

- gate v3发现CPU BF16 GEMM oracle不可靠：NPU投影与CPU FP32累加后BF16结果逐元素完全一致，CPU原生BF16与二者有33886/51200元素不同，最大abs0.0625；反旋转CPU/NPU最大误差仅1.43e-6。v4保留官方forward方程，用FP32累加的CPU线性层，并额外以64个跨全输出范围通道的FP64累加核验，不改容差；session74436。

### 第二轮最终验证与继续入口

- gate最终v4通过，两层均使用实际wkv/q/k和48条真实INT8/E8M0表行；按官方Engram.forward方程验证原始basis gate与已旋转value delta。NPU eager/图捕获/变化hidden回放通过，图像mask输出精确不变；第二次回放最大abs差层1为2.38e-7，层14为0。
- oracle修正依据：v3证明ARM CPU原生BF16 GEMM偏离FP32累加；v4显式FP32累加，并对64个均匀跨输出通道以FP64累加核验。层14 NPU投影仍有4个BF16舍入差（maxabs0.015625），最终测试输出通过既定rtol/atol，不推断整模型精度通过。
- `torch.copysign`替换为FP32符号位+where后最终日志无该CPU fallback；代码在gate路径保留负零语义。没有为诊断新增env变量或修改上游。
- 新实现文件：`patch/worker/deepseek_v41_embedding.py`（slice loader+EP lookup）、`deepseek_v41_engram.py`（新增paged history）、`deepseek_v41_engram_layer.py`（真实投影/gate）。测试和验证脚本见本地源码与 `artifacts/validation/`。
- 接下来必须进入模型装配，避免把组件测试当交付：给history建立由scheduler管理的cache spec/metadata并接入forward；实现ratio1/2 compressor和跨层KV/index读写；用V4基础FusedMoE/线性层构建V4.1 delayed-mHC层与loader，严格按官方target-layer输入收集DSpark状态。当前还没有DeepseekV41ForCausalLM运行实现注册。
- Engram大表已验证可跨16rank全量驻留。集成时每rank的token数量须由DP/图bucket共同padding，idle rank也参加collective；不能直接用各rank不同长度输入调用all_gather。

- 第二轮最终13个修改/新增文件本地与远端SHA256一致，见 `artifacts/verified-source-state-engram.json`；vLLM上游仍指定SHA且干净。成功及失败验证日志已取回本地 `logs/`，NPU终态见 `logs/post-engram-npu-state.txt`。尚未commit/push；完整目标保持active。

## 2026-11-10 第三轮：分页缓存、压缩器和延迟mHC

- 上轮知识库查询只证明查询可用，对适配目标属于no progress；本轮继续检查源代码和原iTASK，未重复启动旧实验。所有者、任务ID、上游SHA不变，开始时NPU无进程。
- 新增 `AscendTokenCacheSpec` / `AscendSlidingTokenCacheSpec`：scheduler拥有物理页，runner以单张tensor绑定并保留页stride，支持INT64历史、FP32压缩状态和BF16 latent/index。runner仅增加spec驱动的分配/reshape分支，未增加模型相关调度行为或改上游。
- `DeepseekV41TokenHistory.forward`读取真实forward_context的block_table/slot_mapping，使用vLLM永久null block 0处理padding；builder在设备上生成request_indices，复用固定buffer。保留旧显式history原语供已有验证，主路径使用分页模块。
- 新增ratio1/2 compressor：ratio2 FP32投影/门控，按通道softmax后先转换BF16再RMSNorm；状态按原始token位置分页，支持任意chunk和spec覆写。状态block_size=4，sliding_window=2，避免把每页扩成128-token状态。
- 新增delayed-mHC decoder组合层：attention使用上一个子层的pre_mix，FFN使用attention生成的pre_mix，返回FFN pre给下一层；combination按source→destination混合。后续由模型构造器注入真实attention和Ascend FusedMoE，当前不声称完整层已接实际attention/MoE权重。
- 新读取唯一官方来源同一固定HF revision的 `inference/kernel.py`，SHA256已加入source-manifest，用于核对Sinkhorn迭代及epsilon位置；未查其他V4.1实现。
- A3 session59152完成实际checkpoint层2/20 compressor和mHC检查：ratio2与提取官方Compressor类结果maxabs=0；ratio1 maxabs=4.768e-7（rtol/atol0.02）；两层mHC与独立FP64方程在rtol2e-5/atol2e-6内一致。NPU图变更输入多次回放通过；history覆盖页stride、跨页、request隔离、padding、拒绝覆写。日志 `paged-compressor-mhc-npu-v1.log`，明确model_validation=false。
- 全拓扑发现基线C4/C128分组硬编码不兼容，因此添加仅对新token-cache specs生效的分组/池规划：所有full缓存共用logical block IDs；每个SWA/state有独立回收组；不同组仅同page bytes可共用tensor池，组内layer绝不alias。真实40层拓扑为44 scheduler groups、9物理池；测试8GiB预算内每block总物理bytes=410624。此为规划单测，不是150k容量通过。
- 全拓扑测试v3失败为fixture漏speculative_config；v4失败为测试未走generate_scheduler_kv_cache_config的正式转换；v5补全真实调用链后10项通过，并验证全拓扑128-token prefix hit。旧失败日志保留，未改变通过标准或绕过注册检查。
- 全部适配UT 50 passed；runner缓存回归5 passed、prefix-cache/CP回归28 passed。ruff针对本轮新增/调整的小文件通过，git diff --check通过；尚未执行交付前完整format.sh ci。未启动完整模型服务，未提交/push；完整阶段4–9仍未完成。
- 下一步：实现窗口/压缩KV的RoPE与量化数学、index/candidate选择和跨层forward上下文，接真实attention、FusedMoE与loader并注册DeepseekV41ForCausalLM。官方target层37/38/39的DSpark输入必须在Engram后、attention前收集；模型装配后再进入完整服务验收。

- 本轮最终22个修改/新增文件本地/远端SHA256全部一致，见 `artifacts/verified-source-state-round3.json`；上游vLLM仍为指定SHA且工作树干净。所有成功/失败日志已取回 `logs/`，NPU无运行进程，见 `logs/post-round3-npu-state.txt`。
- 架构变更理由：runner原路径会把缓存按K/rope拆分，不能承载INT64 history或单张FP32状态；因此增加显式spec标志驱动的单tensor分配/reshape，而不按模型名分支。缓存分组补丁只对新token-cache specs生效，其余模型仍走原路径，并以已有prefix-cache/CP回归核验。上游推广可采用通用single-tensor layout标志与任意同尺寸页池规划；尚未提交外部PR。

## 2026-11-10 第四轮：attention 与模型装配

- 上一 goal turn 仅验证知识库可用，对适配目标分类为 no progress；本轮重新核实原 iTASK ID/owner/Running、上游 SHA/干净状态、NPU 全部空闲。
- 新增分页 attention/indexer 与每次 forward 独立的跨层上下文；实现官方 FP4/FP8 舍入、RoPE、64-key online softmax 与 sink、候选块和排序 top-k。测试日志 attention-ut-v1.log 为 8 passed，包含格式舍入、全 masked 行、候选来源语义、分 chunk 与全 prefill 对齐及 graph metadata 长度。
- 目前为有界 workspace 的 Torch 实现，尚未证明 NPU 性能或整模型 graph；没有将组件单测记作服务验收。接下来核对真实 W8A8 attention 权重与 NPU eager/graph，再接模型/loader/FusedMoE。

- attention NPU v1：layer0 真实权重 prefill/3次变化位置 graph 通过，maxabs0.01171875；layer2 官方 oracle cache 默认FP32与BF16 query冲突，测试壳未设置官方入口默认BF16。修复 oracle 构造时的默认dtype，保留v1日志，不改变实现或容差。

- v2 layer2 prefill 超出原容差（348/46080，maxabs0.091796875）。正在定位首个偏差：CPU oracle 两处原生BF16 einsum 与NPU比较，同时用FP64检验FP32累加，沿用之前已证实的ARM BF16 GEMM诊断，不调整容差。

- v3 FP64检查确认CPU einsum有最多0.0078125舍入差，但修正后主偏差仍存在，排除此项为主因。下一步直接比较Q、量化KV与稀疏索引顺序；不重复无辨别力的试验。

- 首偏差诊断v1发现窗口KV误差约0.00098，而从第一个完整压缩组开始KV最大差0.3125；因此优先隔离压缩latent与FP4量化，而非调整整层容差或直接归因softmax。

- 诊断v2确认压缩latent逐元素相同（maxabs0），相同post-RoPE输入的CPU/NPU FP4量化差0.3125，主因已收敛到量化路径。

- v3诊断明确A3 exp2的FP32误差影响精确二进制scale：应为0.15625却得到0.156249985，导致-5中点被舍入为-6；E4M3 scale自身也有一格跳变。实现改为指数位构造精确2的幂，并在原值域比较FP4中点，新增真实NPU格式/变化图回放回归。未放宽attention容差。
- 新增indexed checkpoint reader：在读取tensor前按本rank专家与显式架构prefix过滤；Engram权重/scale只通过safetensors slice读本rank行，支持分文件存储，不读取index外重复权重。尚待接整模型loader。

### 第四轮数值修复及模型启动

- FP4故障已修复，未改变attention容差。A3 exp2对2的整数幂产生一位FP32误差，既改变E4M3 scale的舍入，又把FP4中点推向相邻值；改为按指数位构造精确幂、用原值域中点比较。4项真实NPU回归通过，涵盖正常值/次正规值/中点及变化输入图回放。
- 真实层0、2、20 attention以固定HF官方Attention类为结构参照，保留真实W8A8动态量化/GEMM，prefill及3次变化位置decode graph均通过rtol=0.04/atol=0.04。ratio2 decode maxabs0.001953125，ratio1 decode maxabs0.0078125。CPU oracle的BF16 einsum用FP64核验FP32累加后处理；日志保留该差异，未宣称整模型或production loader通过。
- 新增 `deepseek_v41_model.py`：真实TP线性层、Ascend FusedMoE、delayed-mHC、Engram及DP共同padding；DSpark辅助状态采集点为Engram后、attention前的残差均值。当前图片路由仍明确拒绝image_mask，Vision/DSpark服务集成待后续完成，不能把此骨架作为最终交付。
- `DeepseekV41ModelLoader`以显式`--load-format deepseek_v41`接入默认加载生命周期，保留量化postprocess和EP专家过滤；大表走本rank行切片，普通tensor按index过滤后读取；普通参数缺失会报错。
- 真实EngineArgs DP4TP4配置识别成功；全部适配UT 64 passed。首个服务v1在16 worker模型构造时发现HF selector-only量化元数据阻止sidecar读取，已修复并新增回归；模型UT和现有ModelSlim回归合计54 passed（包含新增1项）。
- service v2正在尝试真实权重加载，具体状态见in_flight_runs。4096上下文/eager仅用于首次功能联调，最终两种布局16×153600、full graph、完整开关、Vision、DSpark、GPQA≥86%等门槛未改变。
- 当前32个修改/新增文件本地与远端SHA256一致，`artifacts/verified-source-state-round4.json`；13份本轮成功/失败日志已取回。未执行最终全量format.sh ci，未commit/push。

## 2026-11-10 第五轮：基础服务联调

- 上一goal turn仅验证知识库可用，对适配为no progress。本轮恢复已核实v3/session91729退出1且worker全部退出；原iTASK owner/ID不变，16NPU与8127/13491已空闲。
- v2真实文本主干权重已在16rank加载；修复Engram哈希切片不连续后，v3通过权重加载和内存预热，后于cache binding失败：`model.history`缺少数值层编号。共享history逻辑置于第0层之前，改注册名为`model.layers.0.history`，不修改绑定器或上游；新增真实bind_kv_cache回归。

## 2026-11-10 第六轮：整模型图数值排查

- 上一goal turn只查询知识库，对适配为no progress。本轮重查原iTASK owner/ID/Running、v6服务PID151255与health200，未重复启动。profile51801句柄消失；远端receipt确认client_exit0及stop_profile完成，真实24-token输入生成14token，结果1, 2, 3, 4, 5。16rank原始trace已生成，尚需导出分析。

- Teacher-forced对照：同一pos334以prefill计算时warm/hit logprob均-0.07889008；正常graph decode warm=-0.07889019、hit=-0.10020690。差异随请求历史变化，不能归因固定命中差异或直接放宽容差。准备临时all-rank观察worker，使用MindIE ACLGraphAddressTracker和图内定长copy定位首偏差；生产源码不加入调试代码。

## DSpark 接入（第七轮）

- 新增官方V4.1草稿网络：3层delayed mHC、128专家/3topk、分页非因果块注意力、独立MTP embedding/head及FP32 Markov/confidence。复用Ascend DSpark proposer和严格indexed loader。112项CPU测试、真实EngineArgs配置探测通过。
- 首次真实MTP attention对照失败（stage0 maxabs0.3125）。候选原因是短上下文前置masked key改变64-key在线softmax分块与BF16概率舍入；准备改为官方环形KV顺序、有效context和draft前置，保持容差0.04不变，再测。完整DSpark服务尚未验证。

- 首偏差probe：q_norm输出与全部14条selected KV完全相同；wq_b INT8投影输出33826/163840元素不同、maxabs0.0625。故短context顺序不是主因；正在独立检查CPU动态量化oracle与NPU算子的整数/scales契约。另通过runner代码发现Eagle辅助层编号+1契约，已在模型set_aux接口转回官方37/38/39并补UT。

- DSpark attentionv3九项真实权重对照通过（3层×3接受位置，rtol/atol0.04不变，maxabs0.0078125–0.0625）。测试共享NPU激活量化整数/scales，CPU GEMM仍独立并有FP64通道检验；该结论不包括独立量化数学验收。113CPU UT通过。准备真实DP4TP4 DSpark greedy服务，随后验证非零接受/拒绝与probabilistic。

## Lint and evaluation checkpoint (2026-11-10)

- All format.sh ci hooks pass in logs/format-ci-v2.log. 45 task files staged, no commit/push. Five lint-only edits: regex imports, deterministic repr test hashing, config docstring, import sorting. Official Darwin gitleaks/shellcheck binaries installed only under task artifacts, verified release SHA receipts.
- Isolated post-lint CPU regression: 206 passed. Local source2e6ec2a0d336353c244c9b0f429b8608a8cc0d4d81185179c8a6a11ae8250d3a; deployed v22 remainsbda2123c... and unchanged.
- v22 max_model_len153600 passed normal memory profile (2.92GiB KV, estimated4.67requests/DP), capture2.09-2.10GiB and real replay on4DP. Short334token prefix/eager numerical parity exact, temperature0 acceptance233/340. No long-context residency claim.
- GPQA retry=0 first attempt sent zero HTTP requests: AISBench uses range(self.retry). CLI exit0 had no predictions/results, explicitly invalid. New immutable v2 configs set retry=1 (one attempt); prompt/145000output/temp1/top1/seed17/thinkingmax unchanged. Warmup passed,3formal requests in progress; no score yet. REQUEST_TIME_OUT=None in pinned AISBench, so no client duration cap.

## Latest CPU and evaluation checkpoint

- Full GPQA v3 remains live on deployed sourcebda2123c, AISBenchPID19248/session69825 (start receipt artifacts/gpqa-full-v22-v3-start.json, immutable config SHA9588f93823199fff5c0e8041f3d887c4a11d59b2e506d08a37b00defd9b317a4); last checked9/198 formal completions,10includingwarmup, all audited finish_reason=stop, no failures or preemptions. This is progress, not accuracy acceptance.
- Local SP/DSA-CP source d1131e28 (before documentation-only update) passed243CPUtests. Full format CI v3 passed all hooks on45stagedfiles. No commit/push or live source sync.
- Prepared artifacts/validation/diagnose_mhc_fixed_tiles.py (syntaxchecked only, not uploaded/executed): compare unchanged native mHC coefficients at1/6/12/24/78/94/110/128/129 rows against fixed16/fixed64 coefficient tiling; retain captured94/110 first-divergence reproducer, same-row/mixed-row controls, FP64 CPU coefficient oracle and original2e-6/2e-7 component tolerance. Measures downstreamBF16 equality and3call timing. Production decoder unchanged; run only with free ownedNPU0 after currentservice stops.
- Prepared artifacts/validation/validate_attention_cp_distributed.py (syntaxchecked only): after service shutdown/free-NPU verification, real W8A8 layers0/2/20 with HCCL TP4 thenTP8, CPoff/on same sharded weights and changed-input graph replay; original .04/.04 attention tolerance. Not yet uploaded or executed.
- launch_dspark_service.py now aligns graph buckets to lcm(6,TP) when SP/DSA-CP/shared-expert-DP is requested, avoiding incompatible DSpark/TP padding; TP4[12,24],TP8/maxseq8[24,48]. Future service tagv23 remains unused.

## 恢复与 v25 诊断修正

- 上一 goal turn 只复核知识库可用，对适配目标为 no progress。本轮已重新核实原 iTASK ID、owner、Running 与独立工作树。
- v25 warm 响应与 v23 相同，随后 observer 在 execute_dummy_batch 中读取 None metadata 引发 AttributeError；这是诊断工具失败，不是新的生产模型证据。client11842已exit1，四份原始日志/manifest已SHA核验取回（pull29593exit0）。
- v26只修改任务诊断artifact：跳过无attention metadata的target forward，防御None window metadata，额外保存layer1输出投影前和wo_b输入。CPU AST控制流检查通过None/empty/inactive三条路径；未修改生产源码。

- v26完整4请求成功；分析59922exit0确认warm/hit/miss IDs及selectedLP逐值复现无观测v23。层1query末token虽同，但历史位置227,234..239的layer1输入已不同，逻辑window KV差异正好落在同7位置(maxabs0.125)。warm/miss全部334位置输入及332..335观测边界一致。输出投影前已分叉，因此没有执行仅针对wo_a的候选实验，也未修改生产代码。下一步复用服务做240token最小重现，并核实与原始长输入隐藏状态是否一致。

- 240token缩短实验33278exit0，但warm/hit的layer1输入均与原334token捕获不一致；短请求本身232..239各边界一致，故不能据此排除长请求的层0或Engram。保留失败缩减证据；下一轮v27仍用原334输入，并完整捕获前两层以定位227/234..239，不修改生产数学。

<!-- mindie-rsi-progress:start -->
## Goal

Optimize `dsv41-mindie-20261109` under verified measurement and ablation gates.

## Best Supported Candidate

- None yet.

## Active Hypotheses

- Establish first verified candidate passing Gates 1-3.

## Refuted Directions

- avoid dp4tp4-full-graph-v5-first-request when {"capture_sizes": [1, 2, 4], "dp": 4, "ep": 16, "graph": "FULL_DECODE_ONLY", "max_model_len": 4096, "soc": "Ascend910", "tp": 4}: refuted (run `unknown`)
- avoid dp4tp4-full-graph-v6-numerical-parity when {"capture_sizes": [1, 2, 4], "dp": 4, "ep": 16, "graph": "FULL_DECODE_ONLY", "max_model_len": 4096, "soc": "Ascend910", "tp": 4}: refuted (run `unknown`)
- avoid fp8-graph-isolated-repro-v1-v3 when {"model_weights": false, "npu_count": 1, "soc": "Ascend910"}: refuted (run `unknown`)
- avoid full-service-fp32-mhc-prefix-v23 when {"actual_prefill_tokens": 112, "dp": 4, "draft_method": "probabilistic", "ep": 16, "graph": false, "max_num_batched_tokens": 128, "temperature": 0, "tp": 4}: refuted (run `unknown`)
- avoid full-service-fixed-floating-boundaries-v28 when {"actual_prefill_tokens": 112, "dp": 4, "draft_method": "probabilistic", "ep": 16, "graph": false, "lengths": [127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025], "max_num_batched_tokens": 128, "tp": 4}: refuted (run `unknown`)

## Experiments To Complete

- `dspark-real-service-v13`: outcome=`None` (re-verify measurement/ablation)
<!-- mindie-rsi-progress:end -->

## 用户追加评测（2026-11-10）

- 用户要求本次代码、计划和进展提交推送完成后，运行GSM8K测试集前300题，全服务并发32，DP4TP4，开启decode full graph与DSpark。
- 这是独立授权的评测任务；完整适配目标仍暂停，不替代原定GPQA、长上下文容量或功能门槛。评测须使用本次提交的实现，记录实际服务参数、图回放、DSpark执行、准确率、成功数和截断。
