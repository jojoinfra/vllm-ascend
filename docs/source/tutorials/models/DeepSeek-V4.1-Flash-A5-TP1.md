# DeepSeek-V4.1-Flash A5 TP1 部署验证

## 简介

本次在初始分支 `dsv41-a5-fp8-fp4` 的 CANNBot 优化镜像上验证
**TP1 / DP4 / EP4**，仍使用物理卡 **1、2、3、4**，不是单卡完整模型部署。
真实权重为 `/mnt/weight/DeepSeek-V4.1-Flash-1017`，容器内路径为
`/models/deepseek-v41`。仅验证 8 并发、8192 输入 token、256 输出 token。

## 支持特性与容量

| 项目 | 本次验证 |
| --- | --- |
| 权重 | 真实 FP8 dense / FP4 experts；未使用 dummy |
| EP | 四卡专家并行，每卡 96 个专家 |
| ACLGraph | FULL_DECODE_ONLY，捕获 `[1,2,4,8]`，日志确认实际 replay |
| 多模态 | 文本和图片请求均通过 |
| DSpark/MTP | 关闭，与优化后 TP4 对照保持自回归策略 |
| flashcomm1 / DSA CP | 关闭；TP1 无 TP 序列并行 |
| 上下文与调度 | max-model-len=65536，max-num-seqs=16，max-num-batched-tokens=512 |
| 实际工作负载 | c8，输入 8192（含模板），输出 256 |

每卡权重占用约 80.12 GiB。内存比例 0.93 时可用 KV 仅 2.83–2.84 GiB，
无法满足启动检查要求的 64K 上下文（需约 3.57 GiB），因此首轮部署失败。
显式设为 **0.94** 后 KV 达到 3.74–3.75 GiB，服务成功启动。
64K 是配置上限，本次没有做满 64K 请求测试；128K 的 KV 需求超出本配置预算，未测试。
历史 TP1 + DSpark 方案也曾因 KV 预算为负而启动失败，不应与本次自回归结果混用。

## 环境准备

沿用已验证的 A5 镜像和依赖，不升级：vLLM 0.27.1、PyTorch 2.10、
torch-npu 2.10.post4、CANN 9.2.0、CANNBot 0.6.0+g91e7c8f。
实现安装路径保持 `/opt/d4ac431/`。需要四张 96 GiB A5 和足够的主机内存；
Engram 表使用约 188 GiB 的 CPU mmap。

镜像：`jojo/vllm-ascend:dsv41-cannbot10x-graph16-v1`。
本次镜像 ID：
`sha256:e2b8af3808106cd45a7d2af0c9994f7cbbfe4504860fb2166046fc57245216d1`。
部署目录保留初始分支原样的 `serve.sh` 和 `reference/chat_template.jinja`。

## 部署与运行手册

在仓库根目录执行。启动器检查卡占用和容器重名，不会替换已有服务。
本次验证成功的容器名为 `jojo-cannbot10x-tp1-8k-v2-npu1-4`，已保留运行。
以下为重新部署命令，仅在这四张卡空闲时使用新容器名执行：

```bash
python3 examples/deepseek_v41_a5/cannbot/start_runtime.py \
  --image jojo/vllm-ascend:dsv41-cannbot10x-graph16-v1 \
  --name jojo-cannbot10x-tp1-8k-next-npu1-4 \
  --deployment /home/jojo/dsv41-cannbot-10x-20261111/deployment \
  --parallel-layout dp4 --decode-mode autoregressive \
  --gpu-memory-utilization 0.94 \
  --enable-hc --enable-cache --enable-cache-slots \
  --enable-fp8 --enable-fp4 --enable-index --enable-norm \
  --enable-rope --enable-residual
```

默认宿主机端口 8010，模型名 `deepseek-v4.1`。启动器最终直接执行保留的
`vllm serve` 脚本，并追加上述配置覆盖项；基础脚本不变。

## 功能验证

```bash
curl --noproxy '*' http://127.0.0.1:8010/v1/models
curl --noproxy '*' http://127.0.0.1:8010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1","messages":[{"role":"user","content":"只输出数字4"}],"max_tokens":32,"temperature":0,"chat_template_kwargs":{"thinking_mode":"chat"}}'
```

真实权重验证包括模型列表 HTTP 200、文本准确返回 `4`、图片计数准确返回 `5`。
启动成功本身不作为推理通过依据。

## 有限精度回归

GSM8K 固定前八题同时发出，**8/8** 数值答案匹配，随后图片请求通过。
这只是八题集成回归，不代表完整 GSM8K 或长上下文精度评测。
测试描述见 `tests/e2e/models/configs/DeepSeek-V4.1-Flash-A5-TP1.yaml`。
本次使用 `examples/deepseek_v41_a5/cannbot/validate_service.py --concurrency 8`，
数据集 SHA256：
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。

## 性能与分析

三组均使用相同八条固定输入，每条准确为 8192 token；每轮预热后计时，输出每条
256 token。独立 cache salt，所有有效轮次均无前缀缓存命中、无抢占。
初始 TP4 使用 DSpark5；优化后 TP4 和 TP1 均使用自回归。
TP1 额外改变了 DP 布局、图捕获档位、DSA CP 和内存比例，不能把差值全部归因于内核。

| 配置 | 平均输出吞吐 tok/s | 平均 TTFT 秒 | 平均 TPOT ms/token |
| --- | ---: | ---: | ---: |
| 初始 TP4 / DP1 / EP4 | 11.07 | 77.11 | 358.30 |
| CANNBot TP4 / DP1 / EP4 | 16.92 | 50.42 | 269.39 |
| CANNBot TP1 / DP4 / EP4（全部六轮） | 58.80 | 13.68 | 85.63 |

TP4 两组各三轮。TP1 原始三轮为 **50.04、50.71、77.66 tok/s**，
为排查波动又补测三轮，得到 **50.52、73.76、50.12 tok/s**，全部保留。
六轮均值 **58.80**，中位数 **50.61**，范围 **50.04–77.66 tok/s**。
相对优化后 TP4，均值为 **3.48 倍**、中位数为 **2.99 倍**；
相对初始 TP4，均值为 **5.31 倍**。该工作负载尚未达到 10 倍目标。

补测保留每个 DP engine 的原始 Prometheus 计数器，扣除单独的 64 token 预热输出，
按每请求 256 token 还原请求分布：

| 补测轮次 | DP0 / DP1 / DP2 / DP3 请求数 | 吞吐 tok/s |
| --- | --- | ---: |
| 1 | 2 / 1 / 2 / 3 | 50.52 |
| 2 | 2 / 2 / 2 / 2 | 73.76 |
| 3 | 1 / 2 / 2 / 3 | 50.12 |

波动与 DP 请求分布对应。原始前两轮日志也出现最多三条活跃请求以及 capture_size=4，
第三轮出现两条活跃请求的 capture_size=2。结合补测，负载分布不均是主要解释；
尚未单独控制 API 前端数量或调度算法，不能将具体前端机制认定为已证实根因。
不能把均匀分布下的单轮峰值当作稳定吞吐。
TPOT 按每请求 `(latency - TTFT) / 255` 计算，含其他请求 prefill 的干扰，不是独立 decode 内核耗时。

证据根目录：`/home/jojo/dsv41-cannbot-10x-20261111/`。
`tp1-8k-v1/service.log` 保存 0.93 启动失败；`tp1-8k-v2/` 保存成功服务日志、
`smoke.json`、`quality.json`、原始 `rounds/`、`distribution-rounds-v2/`
和复核比较的 `compare.py` / `comparison.json`。
TP4 对照位于 `long8k-v1/`。同一主机顺序测试，其他卡上的实验保持运行。
