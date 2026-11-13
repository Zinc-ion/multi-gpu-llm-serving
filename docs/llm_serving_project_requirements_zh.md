# 单机多卡 LLM 推理系统项目需求

## 1. 项目定位

### 项目名称

**无 NVLink/P2P 多卡环境下的 Prefill/Decode 分离推理与优化**

标题不使用基座仓库名称。项目说明中应如实写成“基于开源轻量推理框架进行二次开发”，并保留上游 Apache-2.0 许可证和版权信息。

### 主线

在没有 NVLink、GPU 间不支持 P2P 的消费级多卡服务器上实现 PD 分离推理，找出吃掉分离收益的瓶颈，并把收益拿回来。

项目必须形成闭环，所有模块都服务于这一条线：

1. **问题**：P 和 D 共置部署时，prefill 打断 decode，TPOT 尾延迟劣化。用自己的基线测出来，而不是引用论文结论。
2. **方案**：PD 分离，消除两阶段互相干扰。
3. **新瓶颈**：本机 GPU 间不支持 P2P，且分属两个 NUMA 域，KV 只能经锁页主存走两跳 PCIe；同时，固定的 P/D 配比无法适应负载变化。
4. **优化**：逐层流水传输、传输路径工程、FP8 KV cache、按阶段选择并行方式、弹性 P/D 调度。
5. **验证**：同硬件、同模型、同负载下，以 SLO 约束下的 goodput 为核心指标，与共置基线和 chunked prefill 基线对比。

与主线无关的工作不进入简历主线。

### 为什么做这个方向

- 主流 PD 分离工作默认有 NVLink 或 RDMA。DistServe 实测 KV 传输不到总延迟的 0.1%，但前提是节点内 NVLink 加 25Gbps 跨节点网络；在无 P2P 的 PCIe 拓扑下，这个结论不一定成立，需要实测。
- 优化手段集中在传输流水、内存路径和调度，不依赖算子开发经验。
- 自研框架可以修改的部分多，便于展示对推理系统核心机制（KV 生命周期、调度、并行、传输）的理解。这对实习面试有价值。

### 最终简历目标稿

以下内容是完成对应模块并通过验收后的目标稿。`XX` 必须用实际测量结果替换，未完成的功能不能提前写入简历。

**项目描述：**

在无 NVLink/P2P 的 4×RTX 4090 环境下，基于开源轻量推理框架二次开发 Prefill/Decode 分离推理系统。以共置连续批处理和 chunked prefill 为基线，定位跨 NUMA PCIe KV 传输和固定 P/D 配比两类瓶颈，在 SLO 约束下优化 goodput。

**个人工作：**

- 修复 KV block 生命周期、调度守恒和张量并行权重加载问题，建立真实 BlockManager/Scheduler 生命周期测试和单卡/多卡输出一致性测试。
- 实现读取 block table 的 paged prefill、chunked prefill，并修复 CUDA Graph decode 路径，建立可公平对比的强基线。
- 实现可独立启动的 Prefill/Decode 实例、请求协议、KV 传输与异常回收。
- 针对无 P2P 的 PCIe 拓扑实现逐层流水 KV 传输和传输路径优化，将不可隐藏的传输开销从 `XX ms` 降到 `XX ms`。
- 实现 FP8 KV cache（存储和传输），KV 传输量和显存占用降低 `XX%`，输出质量误差 `XX`。
- 按阶段选择并行方式，实现负载感知的弹性 P/D 调度，在 TTFT≤`XX`、TPOT≤`XX` 的 SLO 下将单卡 goodput 从 `XX` 提升到 `XX`（相对 chunked prefill 基线提升 `XX%`）。

**技术栈：**

Python、PyTorch、CUDA、Triton、NCCL、PyTorch Profiler、Nsight Systems。

## 2. 硬件和部署约束

- 服务器：5 张 NVIDIA GeForce RTX 4090，单卡可见显存 `49140 MiB`，驱动 `580.178.04`。
- GPU 拓扑：GPU0-2 位于一个 NUMA/PCIe 域，GPU3-4 位于另一个 NUMA/PCIe 域；跨域连接为 SYS；`nvidia-smi topo -p2p r/w` 显示 `CNS`，不能假设 GPU 间有直接 P2P 通道。
- 主部署：GPU0/1 为一侧、GPU3/4 为另一侧，共 4 张卡。P 和 D 的方向（哪一侧做 P）用实测传输基准决定。
- GPU2 作为弹性节点和对照卡。所有对比方案必须使用相同的 4 张卡。
- 模型和实验数据存放在 `/Data2/zincion`，避免占用根分区；详细硬件记录见 [server_hardware_2026-11-09.md](./server_hardware_2026-11-09.md)。

## 3. 模型与精度

- 主线模型：**Qwen3-14B（dense，GQA）**。
  - 与已实现的 Qwen3-0.6B 属于同一套架构，适配工作量小。仍需核对与 0.6B 的差异，例如输出头是否与 embedding 共享。
  - BF16 权重约 30 GB，单张 48 GB 卡可以放下。2P+2D 可以组成 4 个单卡实例，路由和弹性调度才有空间。
  - 每 token 的 BF16 KV 约 160 KiB；4K prompt 约 640 MiB。按每跳 20 GB/s 粗估，串行两跳传输约 67 ms。这个带宽是假设，以 M2 实测为准。
- 精度：BF16 为基线；量化只做 FP8 KV cache（M7）。
- 混合线性注意力模型（如 Qwen3.8-27B 的 Gated DeltaNet 结构）不进入主线，作为 M9 的可选扩展。

## 4. 基座已有能力与归属边界

上游仓库已经提供以下基础代码：

- Qwen3-0.6B 和 Llama-3.2-1B 的模型实现及权重加载入口。
- Prefill/Decode 推理流程、请求调度、KV block 管理和 CUDA Graph 路径（当前 CUDA Graph 路径不可用，见 M3）。
- Prefill FlashAttention、Decode Paged Attention 的 Triton 实现及独立 kernel benchmark。
- NCCL 进程组、进程 worker、张量并行线性层和简单 TPS 对照脚本。

这些内容可以作为项目基座和系统背景，但不能直接表述为本人的新增实现。个人贡献必须落在后续改造、缺陷修复、测试和可复现实测结果上。

## 5. 对比方案与评测口径

### 对比方案

所有方案使用相同的 4 张卡、同一模型和精度、同一负载：

| 编号 | 方案 | 作用 |
|---|---|---|
| B0 | 共置部署，连续批处理，prefill 优先（现有调度器） | 暴露 prefill 打断 decode 造成的 TPOT 尾延迟 |
| B1 | 共置部署，chunked prefill（Sarathi-Serve 的做法） | 强基线。不与它对比，PD 分离的收益没有说服力 |
| B2 | 朴素 PD：prefill 完成后整块同步传输 KV | 衡量传输优化的收益 |
| 参考 | 同硬件、同模型、同负载下的 vLLM 原生部署 | 工业实现参照，见下文 |

**vLLM 参照的规则：**

- vLLM 的数字必须测并保存原始数据。面试中“与 vLLM 相比如何”是高频问题，必须能回答。
- 是否写入简历按数据决定。绝对性能差距较大时，如实说明差距来源（kernel、Python 开销等），并强调优化收益是相对自身基线、同口径测得的。
- 是否把优化移植到 vLLM，判断标准是优化机制是否与框架本身的快慢无关，而不是差距大小。见 M9。

### 测量工具

- 自研框架提供 OpenAI 兼容的流式 HTTP 服务（见 M2），自研框架和 vLLM 用**同一个客户端**测量；每 token 的 chunk 和 usage 语义仍需按 M2 的要求实测核对。
- 性能：`vllm bench serve`。锁定的 vLLM 0.15.0 支持泊松/gamma 到达、并发上限和 `--goodput`，详细结果可保存实际请求发送时间；其数据集选项没有 `timed_trace`，不支持按原时间戳回放 trace。
- 质量：evalscope，通过 OpenAI API 接入。
- `vllm bench serve` 看不到框架内部的指标（如 KV 传输时间），这类指标由框架自己记录到日志，按请求 ID 与客户端结果对齐。

### 指标

- **核心指标：单卡 goodput**，即 SLO 达标率不低于 90% 时，每张 GPU 能承载的最大请求速率（DistServe 口径）。
- SLO 达标率：同时满足 TTFT 和 TPOT 目标的请求比例。
- 与 `vllm bench serve` 的输出对齐：请求吞吐、输出 token/s、总 token/s，以及 TTFT、TPOT、ITL、E2EL 的均值和 P50/P90/P99（`--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99`）。对外展示时按需选取。
- PD 专项（框架内部记录）：KV 传输大小、不可隐藏的传输时间、有效带宽占 PCIe 理论值的比例、第二个 token 延迟。
- 峰值显存、失败率、状态泄漏。
- SLO 的具体数值在 M2 测出 B0 的单请求延迟后确定，并在之后的所有实验中固定。

### 测试层次与数据

| 层次 | 要回答的问题 | 数据 | 发压方式 | 对应模块 |
|---|---|---|---|---|
| L1 受控实验 | KV 传输开销随 prompt 长度怎么变；TTFT 由哪几段组成 | 合成数据：`--dataset-name random`，输入长度扫 1K/4K/16K，输出长度固定，加 `--ignore-eos` | 单请求或固定并发 | M2、M5、M6 |
| L2 SLO / goodput（简历主要数字） | SLO 约束下每张卡能承载多少请求 | ShareGPT（对话，短输入，体现 PD 干扰）；LongBench（长输入，体现传输瓶颈） | `--request-rate` 扫速率，泊松到达，固定 `--seed`，`--goodput` 设 SLO | M3、M5–M8 |
| L3 负载变化 | 负载随时间变化时，弹性 P/D 是否有效 | 公开 trace（Azure LLM 推理 trace、Mooncake trace，格式需核实），或合成“对话为主”与“长输入为主”交替的分阶段负载 | 按时间戳回放（客户端待核实；vLLM 0.15.0 无 `timed_trace`） | M8 必需 |
| 吞吐模式 | 优化是否损失吞吐 | 合成数据，固定 `[输入, 输出, 并发, 请求数]` 用例 | 不限速率加固定并发 | 补充 |

- 静态泊松负载测不出弹性调度的价值：固定 P/D 配比只有在负载类型变化时才会一侧闲置，所以 M8 必须有 L3。
- 吞吐模式只作补充。它衡量的是吞吐，而 PD 分离的收益在延迟和 SLO 上，不能作为主线指标。
- 用例的并发和长度必须先按显存容量测算（见 M2），不能照搬其他场景的配置。
- 每次实验保存数据集版本、物化后的请求列表、seed 和实际发送时间。固定 seed 可复现随机到达计划，不能保证实际发送时刻完全相同；精确时间戳回放是 M8 前需要补齐或核实的能力。

### 质量与正确性

- **无损改动**（PD 传输、逐层流水、TP、CUDA Graph）：greedy 解码下 token 序列与改动前完全一致，logits 误差不超过预先固定的阈值。不需要任务数据集。
- **有损改动**（M7 FP8 KV cache）：用 evalscope 测任务精度，至少包括一个推理类数据集（如 math_500 或 GSM8K）和一个长上下文类数据集（LongBench 子集或大海捞针测试）。FP8 KV 的误差会随上下文变长而累积，只测短题看不出问题。
- 采样设置：正确性对比用 greedy；质量评测按数据集推荐设置，并在实验记录中固定。

### 实验纪律

- 每次只改变一个主要变量。
- 保留原始 JSONL/CSV、配置、trace、日志和统计脚本。
- 只有端到端数据改善时才写入简历。

## 6. 模块

### M0：基线、许可证和可复现入口（已完成）

- 固定上游基线 commit、模型 revision、dtype、tokenizer、采样参数、随机种子和测试输入；统一 JSON 配置。
- 见 [m0_m1_baseline_zh.md](./m0_m1_baseline_zh.md)。

### M1：调度和 KV Cache 正确性（已完成）

- 修复 block 引用计数、释放后的 hash 元数据和 block 生命周期问题；支持取消；请求长度在入口处约束。
- 用真实 BlockManager 和 Scheduler 测试 free/used block 守恒和请求守恒。

### M2：Qwen3-14B 路径与测量底座

- 增加 Qwen3-14B 配置并跑通单卡推理，与 Transformers 参考实现对比 greedy 输出和 logits 误差。
- **OpenAI 兼容流式服务**：实现 `/v1/completions` 和 `/v1/chat/completions`，支持流式返回，使 `vllm bench serve` 和 evalscope 能直接测量本框架。TTFT/ITL 依赖流式返回的 chunk 计时，接入后用单请求实测确认计时口径与 vLLM 一致。当前框架只有离线 `generate()`，服务层需要把同步 step 循环改为可持续接收请求的引擎循环。
- **容量测算与配置**：按单卡显存、权重大小和每 token KV 大小算出可容纳的 token 数，据此设计 L1/L2 用例的长度和并发。以 Qwen3-14B 粗估：单卡约剩 13 GiB 给 KV，BF16 约 160 KiB/token，约 8.5 万 token；4K+1K 请求约 17 并发，16K+1K 约 5 并发（以实测为准）。同时调大当前的 `max_model_length=128` 和 `max_num_batched_tokens=1024`；在 M3 实现 chunked prefill 之前，Scheduler 要求 `max_model_length ≤ max_num_batched_tokens`。
- 按阶段记录耗时：schedule、prepare、H2D、forward、sampling、D2H、postprocess，以及后续的状态传输。
- PCIe 微基准：分别测 D2H、H2D、双向同时、同 NUMA 与跨 NUMA，以及 NCCL send/recv 在无 P2P 下的实际带宽。
- 准备 L1/L2 的数据集和请求列表，固定数据集版本和随机种子。
- 产出 B0 的完整数据，并据此确定 SLO。

### M3：强基线（paged prefill、chunked prefill、CUDA Graph）

- 实现读取 block table 的 prefill：支持 `cu_seqlens_q != cu_seqlens_k`，并正确处理 RoPE 位置偏移。
- 在此基础上实现 chunked prefill（B1）。跨请求 prefix 复用可以作为附带能力，按需开启。
- 修复 CUDA Graph decode 路径（capture 时 `outputs` 的形状与模型返回的 hidden states 不匹配）。否则 TPOT 会被 Python 开销主导。
- 测同硬件下的 vLLM 参照数据。
- 验收：与 M2 的参考输出一致；产出 B1 数据。

### M4：张量并行正确性

- 修复 checkpoint loader 绕过并行层 `weight_loader` 的问题。
- 核对 Column/Row/QKV/Merged Linear 的权重、bias、reduce 和 shard 方向。
- 支持独立 GPU 映射，不把物理 GPU 编号、rank 和本地 rank 混为一谈。
- 验收：TP=2 无未加载参数；TP 输出与单卡参考 logits 一致；记录通信时间和峰值显存。
- 用途：为 M8 中“P 实例用 TP=2 降低 TTFT”提供可选配置。

### M5：Prefill/Decode 分离（朴素版，B2）

- 将 Prefill 和 Decode 拆成可独立启动、独立配置、独立生命周期的实例。
- 请求协议：请求 ID、token 进度、位置偏移、block 描述和采样元数据。
- KV 传输：先实现 prefill 完成后的整块同步传输，作为 B2。
- 使用实例独立的 rendezvous、共享内存和传输 buffer 名称，不复用固定的全局 `myvllm` 名称。
- 处理超时、取消、失败、重复传输、状态释放和 Decode 侧恢复失败。
- 验收：分离前后 token 结果一致；单请求、并发、取消和异常退出均无状态泄漏；产出 B2 数据，以及传输时间在端到端延迟中的占比。

### M6：KV 传输优化

- **逐层流水传输**：prefill 每算完一层，就在独立 CUDA stream 上异步发送该层 KV，与下一层计算重叠；decode 侧按层接收。短 prompt 仍用整块传输，切换阈值在 4090 上实测标定。参考 Splitwise。
- **传输路径工程**：锁页主存双缓冲；NUMA 绑定，决定跨 NUMA 那一跳放在哪一侧；合并连续 block，减少小拷贝；P、D 的 TP 度相同时按 rank 一对一传输分片，避免重新切分。参考 Splitwise、DistServe。
- 如果 M2/M5 实测发现传输不是瓶颈，就缩小本模块范围，主攻 M8。
- 验收：不可隐藏的传输时间、有效带宽、第二个 token 延迟、端到端 goodput 相对 B2 的变化。

### M7：FP8 KV cache

- KV cache 以 FP8 存储，attention kernel 读取时反量化；PD 传输直接传 FP8，传输量减半。
- 4090（Ada 架构）原生支持 FP8，不需要自己开发算子。
- 验收：与 BF16 对比 KV 显存、可容纳的 decode batch、传输时间、TTFT/TPOT、goodput，以及 NaN/Inf；按第 5 节“质量与正确性”用 evalscope 测推理类和长上下文类数据集的精度变化，可接受的精度下降阈值预先固定。参考 SpectrumKV、HACK 等 KV 传输压缩工作。

### M8：按阶段选择并行方式与弹性 P/D

- **按阶段选并行**：比较 P 用 TP=2、D 用 TP=1×2 等组合，按 goodput 选择。参考 DistServe 的并行策略搜索。
- **路由与负载均衡**：多个 D 实例之间按 KV 占用和请求数均衡，多个 P 实例之间按待处理 token 数均衡。参考 DeepSeek 推理系统的负载均衡。
- **弹性 P/D**：固定 2P+2D 时，负载变化会让一侧闲置。最小可行版是让 decode 实例以 chunked prefill 在本地处理部分 prefill（同时省掉 KV 传输），由路由器按队列长度和 TPOT 余量决定是否分流；GPU2 作为可切换角色的弹性节点。参考 TaiChi、负载感知 prefill 分流、FluidPD。
- 验收：在 ShareGPT、LongBench 和两者混合的负载下，goodput 相对 B1、B2 和固定配比方案的变化；弹性 P/D 必须用 L3 随时间变化的负载验证。

### M9：扩展（可选）

- 将 M6/M8 中与框架快慢无关的机制移植到 vLLM：KV connector 插件（利用 `save_kv_layer` / `wait_for_layer_load` 逐层钩子），加外部路由器。必须固定 vLLM commit。
- 混合线性注意力模型的 PD 状态传输：线性注意力状态是定长的，传输量不再随上下文长度线性增长，可作为对比实验。

### 贯穿各模块：Profiling、部署和文档

- 用 PyTorch Profiler 定位阶段热点，再用 Nsight Systems 验证 CPU/GPU 时间线、同步、通信和 kernel 空隙。
- 提供单卡、TP=2、2P+2D 的启动脚本和配置样例；记录 GPU 映射、NUMA 绑定、环境版本、模型路径和数据目录。
- 每个实验保存原始数据和结论；明确哪些代码来自上游、哪些是修复、哪些是新增实现。

## 7. 实施顺序

1. M2：先建测量底座、跑 PCIe 微基准和 B0。PCIe 实测带宽决定 M6 的空间。
2. M3：补齐强基线 B1，并测 vLLM 参照。
3. M5（需要 TP 时穿插 M4）：跑通 B2，测出干扰程度和传输占比。
4. 根据数据决定先做 M6、M7（传输），还是先做 M8（调度）。
5. M9 视进度和数据决定。

## 8. 简历事实边界

- 原仓库已有的分页注意力、基础调度器、基础 TP 代码和 kernel benchmark，只能写成项目基座或系统已有能力。
- B1（chunked prefill）是本项目对已有方法的实现，用作基线；不能写成本人提出的方法。
- 借鉴的方法（逐层传输、弹性 P/D 等）要写成“借鉴/实现”，不能写成原创。
- vLLM PD 分离的官方定位是控制 TTFT 和尾延迟，而不是提升吞吐。没有实测数据时，不写“PD 分离提升吞吐”。
- “实现 PD 分离”“支持 FP8 KV”“提升 XX%”只有在代码、测试、日志和对照数据齐全后才能写。
- 所有 `XX` 都必须来自同硬件、同模型、同精度、同负载的可复现实验。未完成的模块从简历中删去，改写成实际完成的内容。

## 9. 参考资料

**工业实践与工程实现**

- [DeepSeek-V3/R1 推理系统概览（Open Infra Week Day 6）](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)：PD 分离、P 和 D 采用不同并行方式、双 micro-batch 通信计算重叠，以及 P 侧、D 侧、专家三类负载均衡。
- [LMSYS：SGLang PD 分离与大规模 EP](https://lmsys.org/blog/2025-05-05-large-scale-ep/)：非阻塞后台 KV 传输、decode 侧预分配 KV 后通知 prefill，以及统一调度的三类问题。
- [vLLM Disaggregated Prefill 文档](https://docs.vllm.ai/en/latest/features/disagg_prefill.html)：现有 KV connector 列表；官方定位为控制延迟而非提升吞吐。
- [vLLM KV connector 接口 `base.py`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/base.py)：包含 `save_kv_layer` / `wait_for_layer_load` 逐层钩子，M9 移植的入口。

**PD 分离与基线**

- [DistServe（OSDI'24），arXiv 2401.09670](https://arxiv.org/abs/2401.09670)：goodput 口径、按阶段选择并行方式、低带宽集群上 P/D 的放置，以及传输开销分析。
- [Splitwise，arXiv 2311.18677](https://arxiv.org/abs/2311.18677)：逐层与整块 KV 传输的对比和切换阈值，以及 mixed pool。
- [Mooncake，arXiv 2407.00079](https://arxiv.org/abs/2407.00079)：以 KVCache 为中心的调度和过载时的提前拒绝。
- [Sarathi-Serve，arXiv 2403.02310](https://arxiv.org/abs/2403.02310)：chunked prefill，B1 基线来源。

**弹性 P/D**（2025 年后的论文，目前只读过摘要，引用前需核对全文）

- [TaiChi，arXiv 2508.01989](https://arxiv.org/abs/2508.01989)：PD 合并与分离的统一，按 SLO 松紧切换。
- [负载感知 prefill 分流，arXiv 2607.02043](https://arxiv.org/abs/2607.02043)：排队的 prefill 以 chunked 形式分流到 decode 节点。
- [FluidPD，arXiv 2610.06917](https://arxiv.org/abs/2610.06917)：原地弹性，不重新加载模型即可切换角色。

**KV 传输压缩**（2025 年后的论文，目前只读过摘要，引用前需核对全文）

- [SpectrumKV，arXiv 2606.08635](https://arxiv.org/abs/2606.08635)：按 token 重要性选择 FP16/INT8/INT4 传输。
- [HACK，arXiv 2502.03589](https://arxiv.org/abs/2502.03589)：在量化 KV 上直接计算，省去反量化。
- [ReKV，arXiv 2611.08770](https://arxiv.org/abs/2611.08770)：量化 KV 传输与后缀重计算重叠。
- [Lynx，arXiv 2607.01831](https://arxiv.org/abs/2607.01831)：渐进式推测量化传输。

**评测工具与数据**

- [`vllm bench serve` 文档](https://docs.vllm.ai/en/latest/cli/bench/serve.html)：数据集类型、到达过程、`--goodput`、百分位指标参数。以锁定的 0.15.0 源码为准，不把最新版文档中的能力视为本环境已支持。
- [evalscope math_500](https://evalscope.readthedocs.io/zh-cn/v1.5.1/benchmarks/math_500.html)：质量评测示例。
- [ShareGPT（Hugging Face）](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)：vLLM 和 DistServe 常用的对话负载。
- [LongBench](https://github.com/THUDM/LongBench)：长上下文负载和长上下文质量评测。
- [Azure Public Dataset](https://github.com/Azure/AzurePublicDataset)：含 Splitwise 团队发布的 LLM 推理 trace（格式需核实）。
- [Mooncake](https://github.com/kvcache-ai/Mooncake)：含 Mooncake 发布的请求 trace（格式需核实）。

**工具**

- [AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS)：其中 `llm-serving-auto-benchmark`（多框架同口径对比）和 `llm-torch-profiler-analysis`（trace 分析）可用于 M2、M3 和 profiling。
