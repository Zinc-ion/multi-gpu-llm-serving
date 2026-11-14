# M2 单 NUMA 实测记录

实验日期为 2026-11-14，代码为 `3c63dfd16eb76a120db220f25eed3d1ed44f4a54`；0.6B 默认 BF16 对齐使用 `e1945ee`，两者的模型计算代码相同。完整 B0、跨 NUMA、数值阈值和 SLO 不在本文的已验收结论中。

## 环境与复现

服务器 `186zincion`，Ubuntu 22.04.1、驱动 580.178.04。仅使用 NUMA 0 的 GPU0/1/2，CPU 和主存也绑定 NUMA 0；GPU3/4 有其他任务，没有运行这些卡或跨 NUMA 用例。共享服务器的其他任务仍可能造成系统噪声。

Python 3.11.16、PyTorch 2.9.1+cu128、Transformers 4.57.3、vLLM 0.15.0。`torch.cuda.nccl.version()` 为 2.27.5，NCCL INFO 构建版本为 `2.27.5+cuda12.9`。Python 和锁定依赖迁入独立目录 `/Data2/zincion/envs/m2`，在服务器重新安装 editable 项目并由 `uv sync --locked --offline --no-build-isolation --reinstall-package myvllm` 审计，没有升级包。`numactl` 来自解包到用户目录的 Ubuntu `2.0.14-3ubuntu2` 包。

原始数据根目录为 `/Data2/zincion/experiments/m2/20261114/`，下文文件名均相对此目录。`versions.txt`、`topology.txt`、`numa.txt` 保存环境；`commands.txt`、`campaign-events.jsonl` 和 bench 的 `*.command.json` 保存命令与代码版本。驱动脚本及配置也在该目录，不代替仓库模型、Scheduler、BlockManager 或客户端。

```bash
source /Data2/zincion/experiments/m2/20261114/env.sh
export M2_DATA=/Data2/zincion/experiments/m2/20261114
```

重跑应另选输出目录，GPU 应先空闲，本框架一次只启动一个实例。标准命令见 [GPU 验收复现](m2_experiments_zh.md)。vLLM 0.15.0 的 `platforms/interface.py::device_id_to_physical_device_id` 对可见设备元素执行 `int()`，相关导入路径不接受 UUID；本次设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，用数字编号限制设备，再用 PyTorch 返回的 UUID 核对 GPU0，见 `gpu-mapping.log`。

## 14B 加载和容量

模型和 tokenizer revision 均为 `40c069824f4251a91eefaf281ebe4c544efd3e18`。八个权重分片与官方 API 的 LFS SHA256 一致，见 `model-*.receipt.json`。除本地 snapshot 路径外，资源字段与 `configs/qwen3_14b.json` 相同。加载没有未加载参数或 `not found` 跳过，`compare_hf.py custom` 对 loader 返回集合、模型参数及 checkpoint 张量的覆盖断言均完成。

`custom-14b.log` 的 `capacity` 实测为：

| 项目 | 字节或数量 |
|---|---:|
| CUDA `mem_get_info` 总显存 | 50,864,390,144 B |
| 参数显存 | 29,536,614,400 B |
| RoPE cache | 838,860,800 B，即 800 MiB |
| warmup 峰值 allocated | 32,916,081,664 B |
| warmup 后 current allocated | 30,384,539,648 B |
| warmup 额外峰值 | 2,531,542,016 B，约 2.36 GiB |
| 分配 KV 前 free | 19,916,455,936 B |
| 每 block | 41,943,040 B，即 40 MiB |
| 实际 block / KV token | 367 / 93,952 |
| KV pool | 15,393,095,680 B，即 14.3359375 GiB |

原公式给出 `floor((free_before_kv × 0.9 − warmup_extra) / 41943040) = 367`，2048-block 配置上限未限制 profiling。每请求总长度先向上取整到 256-token block：

| 输入 token | 输出 128：block/请求 | KV 并发上界 | 输出 1024：block/请求 | KV 并发上界 |
|---|---:|---:|---:|---:|
| 1024 | 5 | 73 | 8 | 45 |
| 4096 | 17 | 21 | 20 | 18 |
| 16384 | 65 | 5 | 68 | 5 |

这是 KV 容量上界，不是满足延迟目标的稳定并发。`max_model_length=max_num_batched_tokens=17408`，warmup batch 为 1，最大请求需要 68 块，小于实测 367。保留原配置，不修改调度策略或 0.6B 基线。

```bash
uv run python tools/capacity_report.py \
  --log "$M2_DATA/custom-14b.log" --config "$M2_DATA/config-14b.json" \
  --output-tokens 128 --output /path/to/capacity-128.json
uv run python tools/capacity_report.py \
  --log "$M2_DATA/custom-14b.log" --config "$M2_DATA/config-14b.json" \
  --output-tokens 1024 --output /path/to/capacity-1024.json
```

## HF 对齐

固定七个用例，每项生成 32 token：三条 chat、1K/4K/16K 重复文本、16K 自然 qasper 文本。FP32 HF 固定 SDPA efficient 并显式展开 KV，BF16 固定 SDPA flash；两框架分进程顺序运行。

| 模型与 dtype | 独立 greedy 序列一致 | 同前缀 top1 一致 | 全词表 max abs err | 全词表 mean abs err |
|---|---:|---:|---:|---:|
| 0.6B FP32 | 7/7 | 224/224 | 0.065397 | 0.0015514 |
| 0.6B BF16，默认编译 | 5/7 | 222/224 | 1.96875 | 0.0622278 |
| 14B BF16，默认编译 | 7/7 | 223/224 | 1.375 | 0.0786557 |

mean 按被比较的 token 数加权；单项仍保留，例如 14B 4K 重复文本 mean abs err 为 0.175727。0.6B BF16 的 `chat-0` 在零起始第 18 个生成位置分歧，HF top1/top2 margin 为 0.25；自然 qasper 在第 8 个位置分歧，margin 为 0。尚未确定验收阈值。

默认编译的 BF16 RMSNorm/RoPE 在 GPU 探针中与 HF eager 有舍入差异。PyTorch 2.9.1 的 `torch/_inductor/config.py::emulate_precision_casts` 说明默认融合会省去低精度算子间的 downcast/upcast。显式开启 `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` 的诊断仍有两项 0.6B greedy 分歧，不能把整模型差异归因于单个算子，运行默认值未改。

显式 KV expansion 用普通函数替换 HF 的 GQA 判断。默认 Mock 的调用历史会保留每步 key 张量，导致长输入 OOM；`3c63dfd` 修复了这一参照工具问题，未修改模型计算、dtype 或 SDPA backend。

原始数据为 `alignment-0.6b-float32-fixed.json`、`alignment-0.6b-bfloat16.json`、`alignment-14b.json`，以及对应 `custom-*.pt`、逐项 HF `*.pt`。配置为 `config-0.6b-*.json`、`config-14b.json`。

## HTTP 和阶段计时

GPU0 上的 0.6B completions/chat 各生成 16 token，分别收到 16 条 choices 消息；usage 正确，流式拼接等于同一 greedy 请求的非流式文本。断开后真实引擎的 256 块全部归还，waiting/running 为空，见 `protocol/protocol-blocks.jsonl`。

真实 bench 八条请求均输出 32 token，每条 31 个 ITL 样本，无失败。按 request id 对齐，客户端 TTFT 减服务端首 chunk 时间为 1.17–1.49 ms。这是服务器 loopback 测量，不代表跨主机网络。

相同请求发给 vLLM 0.15.0，completions 的 16 token 只收到 15 条 choices 消息；八条 32-token bench 请求的 ITL 样本数为 `30,30,30,30,31,31,30,30`，usage 仍为 32。`vllm/v1/engine/output_processor.py::RequestOutputCollector` 允许合并消费者尚未取走的 DELTA 输出；chat 另有角色消息，不能凭一次数量相等判定逐 token 对应。因此这些 vLLM ITL 是 chunk 间隔，TPOT 仍按输出 token 数计算。原始 SSE、usage、detailed JSON 和分析均在 `protocol/`，bench 源码未改。

0.6B 使用 128 输入/32 输出、单并发、八请求，交替开关阶段计时。三次配对的 TPOT 中位数变化为 `+0.93%、−0.58%、+1.07%`，bench duration 变化为 `+0.84%、−0.71%、+0.90%`。见 `timing-overhead-0.6b/summary.json` 及 detailed 结果。小样本变化与噪声同量级，不能宣称开销为零或据此外推 14B 上界。forward event 仍包含逐层主机同步产生的空隙。

## 同 NUMA PCIe / NCCL

扫描 1/4/16/64/160/640/1024/2560 MiB，warmup 5 次、测量 20 次，主存 pinned 且绑定 NUMA 0。以下为 2560 MiB payload 的 event 中位数：

| 场景 | 有效带宽 GB/s |
|---|---:|
| GPU0/1/2 各自 H2D | 26.835 / 26.836 / 26.830 |
| GPU0/1/2 各自 D2H | 26.355 / 26.353 / 26.352 |
| 各卡同时 H2D+D2H，按较慢 stream 算每方向 | 约 22.68 |
| GPU0+1 同时 D2H，按较慢 stream 算每卡 | 约 26.35 |
| GPU0→主存→GPU1，串行 | 13.295 |
| GPU1→主存→GPU0，串行 | 13.296 |
| NCCL GPU0→1，sender rank | 21.525 |
| NCCL GPU1→0，sender rank | 21.479 |

活动卡负载下为 Gen4 x16。单向 DMA 为约 31.5 GB/s 理论值的 84–85%。双向/双卡表值尚未相加；aggregate 带宽、host wall 时间、p10/p90 和全部样本均在 `pcie/`。两跳按一份逻辑 payload 除以两跳 event 时长之和。

NCCL INFO 的数据通道为 `SHM/direct/direct`，同时出现的 `Using network Socket` 不能取代具体通道记录。NCCL 内部实现不等于自研串行两跳，不能把差异表述为已实现 M6 流水优化。跨 NUMA、GPU0↔3 和 GPU3/4 用例未运行。

## 请求列表与 B0 边界

14B tokenizer 重新物化 ShareGPT 1000 请求、qasper 197 请求、三档 random 各 100 请求。固定 revision、过滤策略、seed、逐请求输出预算和 SHA256 均在 `workloads/` 的 manifest。

同一八请求列表、seed 7、1 request/s 泊松参数重复运行，detailed 结果均保存 `start_times`；实际相对发送时间不完全相等，本次最大差约 0.95 ms，见 `arrival-check-0.6b/comparison.json`。这不是时间戳回放。

14B 的 1K/4K/16K 输入、128 输出、并发 1/4 小规模数据在 `b0-smoke/`，每格四请求、开启阶段计时。完整 B0 必须使用 `b0-full/` 的正式请求数与全部逐请求结果，不能把小规模分位数作为最终尾延迟或 goodput。数值阈值和暂定 SLO 仍须人工确认。
