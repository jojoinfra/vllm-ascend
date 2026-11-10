# DeepSeek-V4.1-Flash-1017 开发验证记录

当前适配目标已暂停，本次按用户要求提交阶段性快照。以下命令和结果区分已验证路径与待验收能力；基础文本通过不代表完整模型功能、长上下文或精度已达标。完整记录见[计划](../../../../PLAN.md)与[进展](../../../../PROGRESS.md)。

## 环境与权重

- 基线：vLLM Ascend `v0.27.1rc1`（`3b31886237ed65c435e0965b10e8002ff5769766`）。
- vLLM：`v0.27.1`（`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`），无上游源码修改。
- 验证硬件：16 个 Ascend A3 逻辑设备，每个 64 GiB HBM。
- 软件：CANN 9.1.0、PyTorch 2.10.0、torch-npu 2.10.0.post4、transformers 5.14.1。
- 权重：DeepSeek-V4.1-Flash-1017 W8A8，本地目录包含索引、量化描述、Engram 表及 `optional/quarot.safetensors`。
- 架构与提示格式参考：[DeepSeek 官方 Hugging Face 仓库](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170)。

运行前核实设备归属、现有进程及 HTTP/DP RPC 端口。验证过程中只停止本次启动且身份已核实的服务进程。

## 基础文本启动

以下 DP4TP4/EP16 配置已使用真实权重完成基础文本、分块 prefill、短前缀命中和不等长 DP 并发验证：

```bash
cd /workspace
vllm serve /home/admin/model \
  --served-model-name deepseek-v41 \
  --host 127.0.0.1 --port 8127 \
  --load-format deepseek_v41 \
  --tensor-parallel-size 4 --data-parallel-size 4 \
  --data-parallel-rpc-port 13491 --enable-expert-parallel \
  --max-model-len 4096 --max-num-batched-tokens 128 --max-num-seqs 4 \
  --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":0}' --enforce-eager
```

`4096` 是首次联调范围，不能作为 `16 × 153600` 容量验收。每个 rank 的模型权重分配约 48.27 GiB，额外还需要 KV cache、计算 workspace 和图内存。

专用 loader 依据 checkpoint index 过滤普通参数和本 rank 专家，在读取 tensor 前完成筛选；Engram 表通过行切片分配到 EP16，避免每个 TP4 副本各放一份完整大表。选择器形式的 HF `quantization_config` 仍需要读取实际 ModelSlim sidecar。

## 文本请求与缓存检查

基础验证使用固定官方 revision 的 `encoding/encoding.py` 编码消息，再将 tokenizer 得到的 token IDs 交给 `/v1/completions`。插件另提供 `examples/chat_templates/template_deepseek_v41.jinja`，56 个文本/图片、多轮、system 和思考预算样例均与官方编码逐字一致。可通过 `--chat-template` 指定该文件并设置 `--chat-template-content-format openai`；目前仅支持文本和图片，未实现工具调用编码。

```python
from encoding import encode_messages
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/admin/model")
prompt = encode_messages(
    [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}],
    thinking_mode="chat",
)
token_ids = tokenizer.encode(prompt, add_special_tokens=False)
# POST /v1/completions:
# {"model": "deepseek-v41", "prompt": token_ids,
#  "temperature": 0, "max_tokens": 32, "seed": 0}
```

此处 `chat` 模式和短输出仅用于功能排错，不是 GPQA 的评测配置。

已有真实服务证据：

| 检查 | 实际结果 | 范围 |
| --- | --- | --- |
| 文本 smoke | 18 输入 token，返回 `4`，HTTP 200、正常 stop | DP4TP4 eager |
| 前缀缓存 | 334 输入 token，实际命中 256；新 salt 对照命中 0 | 同一 DP 副本，128-token prefill chunk |
| 输出一致性 | warm/hit/miss 的输出 token IDs 与所选 token logprobs 完全一致 | 上述短文本样例 |
| 不等长 DP 并发 | 4 副本分别输入 18/137/377/737 tokens，均正确返回“7天” | 覆盖不同 prefill/decode/idle 进度 |
| 适配及相关单测 | 257 项通过 | 配置、数学、加载、缓存、Engram、mHC、模型装配、DSpark proposer、视觉、SP/CP 分片、FP32 head |
| NPU 量化 | 5 项通过；新增移位干扰回归在原实现上失败 | FP4/FP8 独立 oracle、变化输入及 graph/eager 交错 |
| 图后端更新回归 | 91 项通过 | 分页缓存、完整 runner 和 ACLGraph 单测文件；与前述单测有重叠 |
| EP 算子执行 | 采集到 `MoeDistributeDispatchV2`、`MoeDistributeCombineV2`、`GroupedMatmulSwigluQuant` 和 `GroupedMatmul` | DP4TP4/EP16，真实权重请求 |
| 短文本 graph/eager | 两轮 warm/hit/miss 的 token IDs 和所选 logprob 完全一致，最大绝对差 0 | 无诊断钩子；第二轮在同 DP 四请求及不等长 DP 并发之后 |

前缀检查应固定 `X-data-parallel-rank`，比较 `vllm:prefix_cache_hits_total` 增量，并用新 `cache_salt` 构造相同输入的未命中对照。仅设置 `enable_prefix_caching` 或第二次请求更快均不能证明命中。

## 图模式排查状态

去掉 `--enforce-eager`，设置 `--compilation-config.mode=0`、`--compilation-config.cudagraph_mode=FULL_DECODE_ONLY` 和 `--compilation-config.cudagraph_capture_sizes='[1,2,4]'` 后，已完成捕获和实际回放。上述短前缀测试及同一 DP 的四请求测试均生成正确 token；真实请求采集记录了 `aclmdlRIExecuteAsync` 调用和 MoE 算子。

该路径曾出现部分 decode token logprob 随请求历史变化、但样例 token IDs 相同的问题。观测将偏差定位到 FP8 舍入输出；缓存写入与该输出一致。

在当前 A3/CANN 9.1 环境中，捕获后的 `>>` 标量移位会受到后续 eager 移位操作影响，使 FP8 使用固定 `2^-9` 舍入步长。最小反例和 NPU 回归均复现；用 `torch.bitwise_right_shift` 代替污染操作未触发该反例，因此复现须保留原运算符入口。量化实现改用等价整数整除提取指数、整数乘法构造精确二次幂，避免图内标量移位。旧实现回归失败 481/512 元素，修复后完整 5 项 NPU 量化测试通过。

无诊断钩子的真实 DP4TP4/EP16 服务已通过上述短文本对照：334-token 输入的 warm/hit/miss 均与 eager 基线的 token IDs 和所选 logprob 完全一致，实际前缀命中 256 tokens。同 DP 四请求、不等长四 DP 请求通过后再次对照，结果仍逐值一致。该结论只覆盖已测样例与配置，不能替代长上下文、DSpark、DP2TP8 或 GPQA 验收。

vLLM daemon worker 内可能无法自动解析 torch-npu 原始 profiling 数据；在采集停止后，可从独立进程调用 `torch_npu.profiler.profiler.analyse(capture_directory, max_process_number=4)` 离线导出。原始数据、解析完成标志和实际算子记录需要分别检查。

## DSpark 开发状态

草稿网络已接入官方三层 delayed mHC、非因果五 token 块注意力、独立 MTP embedding/head，以及 FP32 Markov/confidence head。目标侧辅助状态取官方层 37/38/39 的 Engram 后、attention 前残差均值，并在 Eagle 接口处转换层编号。草稿网络使用 128 个专家、每 token 选 3 个，目标网络保持 384/6；权重仍通过 indexed loader 分片读取。

当前组件证据：206 项适配及相关 CPU 测试通过；三层实际 MTP attention 在三种接受位置上，共 9 项 NPU 对照通过原定 `rtol=0.04, atol=0.04`。测试发现标量 CPU INT8 动态量化公式与 NPU 在 7 个半整数处不同，因此对照共用实际 NPU 激活量化整数及 scale，后续 CPU GEMM 独立计算并以 FP64 通道核验。此项只验证 attention 结构和后续计算，不应称为完全独立的量化 oracle。

真实 DP4TP4/EP16 eager 服务已完成数字序列、Python 函数和中文说明三类请求。修复加载后独立 embedding/head 被通用 proposer 替换的问题后，接受率由 22/1385（1.59%）提升到 233/340（68.53%），五个位置的接受数分别为 61/55/45/40/32，无抢占。加载器在严格完成权重加载后设置 own-vocab 标记，并通过实际 proposer 共享检查的回归验证。目标与草稿的专家数不同，模型 forward 作用域还须绑定各自的 MoE 通信对象。

早期前缀 warm/hit/miss 输出 token 相同，但 logprob 不一致。DSpark 预留 token 后，128 总预算实际形成 112-token prefill 块；334-token 样例的 warm/hit 末块分别为 110/94 行。逐层采集和独立真实权重实验将最早差异定位到 mHC 的 FP32 计算及 BF16 舍入：相同 embedding、attention 输出和权重，仅改变批大小即可逐值重建两份不同的 pre-FFN 输入。将总 token 预算设为 144（预留16、实际调度128）后，同一334-token样例的无观测 warm/hit/miss token IDs 和 selected logprobs 完全一致，实际命中128。但相对非DSpark基线的decode logprob仍差最多0.04084866，不能推广为任意batch不变性。

mHC 现已固定每次 FP32 投影为64行，并将后续向量运算补齐到64行倍数，末尾再恢复真实token数。原实现的两个NPU batch一致性回归均失败，修复后通过，覆盖两组随机权重、10种batch大小、首/中/末位置以及三次变化输入图回放。真实层0权重诊断中，26组相同/混合其他行、1至2048行的系数和pre-FFN结果逐值一致，并满足原有FP64误差界。该组件修复仍需整模型前缀和性能复验。

进一步按完整历史token对齐后发现，固定mHC仍不足以保证整模型前缀一致：相同输入的BF16 `wq_a`、Engram `wkv` 和分组输出投影，在110/112行时也会改变舍入。真实权重单算子重放已复现服务中的差异。现为浮点投影使用固定64行计算，保留原生linear的TP/权重处理，量化linear仍走原路径；覆盖attention/indexer、压缩器、Engram值/键投影与草稿对应投影。真实三类投影的12种行数、混合其他token及首/中/末位置验证均逐值一致，18次变化输入图回放通过，抽样CPU FP64误差界通过。270项CPU回归通过。移除观察钩子后的v28服务已通过原334-token样例，但12个扩展边界中的257/513/1023/1024仍有selected logprob差异，全部token IDs一致；不能宣称任意分块前缀门槛通过。v31观察器逐值复现最短257-token反例：warm末块33tokens走All-to-All，hit末块17tokens走MC2；按16个EP rank的逻辑token/专家对齐后，相同hidden及路由权重在dispatch后首次出现INT8/scale差异。该问题尚未修复，完整组合与性能仍待验收。

另外，原生`LogitsProcessor`默认使用模型BF16 dtype，会覆盖FP32草稿hidden/weight。配置现在显式指定`head_dtype=float32`，target和draft head权重均保留FP32，与官方`ParallelHead`一致。真实TP4权重分片的12项组件检查通过：eager与变化输入图回放输出为FP32，与官方NPU线性方程逐值一致，并经CPU FP64的129个分散词表通道核验。此前服务日志与接受率保留为历史诊断，新的整模型验收须使用修正后的源码。

概率采样在 temperature=1、top_p=1 的三项 smoke 中接受249/380草稿tokens（65.53%），各位置69/55/49/44/32，无抢占；这不构成采样分布等价证明。草稿网络遵循通用proposer的eager约定，禁用其Dynamo编译，目标网络的decode图独立捕获。DP4TP4 的 DSpark greedy 与 Vision 组合已在默认 worker 上通过 `FULL_DECODE_ONLY` 的 `[6,12,24]` 捕获和实际回放；334-token 前缀的输出 IDs/selected logprobs 与 DSpark eager 逐值一致，三类文本请求再次接受233/340草稿tokens。另测127/128/129、255/256/257、511/512/513、1023/1024/1025长度，12组 warm/hit/新salt均逐值一致，最大真实命中896tokens；同DP四请求、不等长四DP请求通过，全部DP抢占为0。该结论仍限128-token实际prefill块。

## Vision 与评测准备

图片processor、原始image token传递、Engram图像mask及独立视觉MoE选择偏置已接入。实际tokenizer/processor的一图和两图占位长度分别为184，以及184+202。真实263个vision/aligner权重张量已在A3验证：39×39、28×55和45×34patch网格（含非纯色图）使用官方3D SDPA后，encoder与aligner输出均与同设备未修改的官方实现逐元素一致。

FIA、按query和按head分块均会引入被最后视觉块放大的BF16差异，不能用已失败的分块路径代替。CPU与NPU整塔输出仍有差异；上述结果只证明同设备实现一致性。保留 [PyTorch 2.10 math-SDPA](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/transformers/attention.cpp) 的FP32缩放及完整矩阵形状、复用score缓冲后，93×93patch额外峰值由约10.30GiB降至4.69GiB；3×3063最大宽图额外峰值5.27GiB。三图生产代码回归仍逐元素一致。这是组件内存实测。真实整模型 DSpark+Vision+decode图服务又通过红图、蓝图、红蓝双图、蓝红双图四个准确性样例，均正常stop并正确识别颜色与顺序；代表性数据集精度仍未测。此组合在0.90显存预算下启动预估KV为-3.03GiB，0.96预算下成功启动，可用KV约0.65GiB，图捕获另用0.88GiB/rank，因此仅证明4096上下文组合，不能用作长文本容量证据。

AISBench固定源码 `525ed1779f50a8ac068b8638771f633a151e769f`，GPQA Diamond共198题。评测工具安装到独立venv，保留主环境torch/torch-npu/transformers版本。评测使用temperature=1、top_p=1、seed=17、thinking/max，145000输出token预算；198题编码后输入113–2595tokens，均在153600上下文内。前三题已实际完成并全部正确，输出616/6959/555tokens，均包含最终答案且未触及预算，服务没有抢占。这仅是流程检查，不代表全量精度达标。旧源码全量评测在12/198题完成后因上述head精度问题主动停止，响应审计与停止原因均已保存，不作为精度验收；修正后须按相同预算完整重跑。本地代理原样转发请求和响应，并记录finish_reason与usage以统计截断。

该AISBench版本的`retry=1`表示总共一次请求，`retry=0`不会发送请求。预测文件为`.jsonl`；必须核对实际记录数、成功数和评分文件，不能只凭CLI退出码或文件扩展名判断评测完成。

## SP 与 DSA-CP 开发状态

MoE入口已在开启原生SP时对hidden和图片mask使用相同的token分片，专家输出all-gather后恢复完整DP内顺序。12个TP4/TP8回归用例在原入口失败，修复后通过。DSA-CP继续保留TP线性权重，仅交换投影后的query heads；indexer与attention按token分片计算，分页写入保持现有归属，输出恢复原TP head分片。18项张量布局测试和6项真实CPU多线程collective测试覆盖TP4/TP8、ratio0/1/2、跨层KV/index/candidate复用、双请求及空闲补齐rank，与独立完整head参考相符。

包含上述改动的CPU回归合计257项通过。草稿attention的override也已接入完整query heads与全局五token块位置的分片，四个TP4/TP8、非对齐起点及拒绝尾部重写回归在原实现失败、修复后通过。真实W8A8层0/2/20的HCCL组件检查已覆盖TP4和TP8，全部rank的9-token预填充及三次变化输入decode图回放均与关闭CP逐值一致；这尚不代表原生完整模型加载、DP服务或其他开关组合通过。

## 剩余验收

以下项目尚未完成，不应据本记录推断支持：

- DP2TP8，包括该布局的实际 Ascend FusedMoE 执行路径证据。
- DSpark：采样/拒绝回滚完整对照、跨batch前缀数值一致性及长上下文组合；短文本greedy/probabilistic接受率与DP4TP4 greedy图已实测。
- Vision：更广泛的图文准确性与其他布局组合；DP4TP4真实权重四个单图/双图样例已通过。
- Decode full graph：扩展至 DP2TP8、长上下文及开关组合；短文本 DP4TP4 的非DSpark和DSpark greedy对照已通过。
- `enable_dsa_cp`、`enable_flashcomm1`、`enable_shared_expert_dp`、`enable_cpu_binding` 的单项与组合验证。
- 两种布局均实际并发驻留并完成 `16 × 153600` tokens，记录抢占/重计算。
- 两种布局的全服务 C1/C4/C16 性能，固定 8192 输入/1024 输出，报告 TPOT、TTFT、吞吐与成功数。
- aisbench GPQA Diamond 全量，temperature=1.0、top_p=1.0，正确率至少 86%；固定思考模式、随机种子和答案提取，记录输出预算及截断率。

本次阶段性快照按用户要求包含代码、测试、计划和进展，并进行格式检查及signed-off提交、推送；剩余功能和性能门槛保持未完成。
