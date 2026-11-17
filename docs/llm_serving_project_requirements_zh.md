# 基于 mini-vLLM 的单机多卡大模型推理框架开发

## 1. 项目定位

### 当前简历交付范围

2026-11-17 调整：围绕已有 TP2、P/D 分离、FP8 推理适配，完成 **实现梳理 → 必要修复 → 功能与正确性验证 → 支持范围整理 → 小规模端到端测量 → 简历与讲解材料**。目标是尽快形成可真实用于实习简历、能解释实现与证据的开发项目，控制 GPU 租用成本。

- 目标模型固定 **Qwen3-14B**。分别标记“已实现且验证通过”“已实现但待验证”“存在已知问题”“尚未支持”；tiny、旧模型和旧环境结果注明范围，不移作当前模型的验收。
- 先分别验收 BF16 TP1、BF16 TP2、bulk P/D 和 FP8 KV TP1。TP2 双卡、1P1D 双卡即可完成对应基本验收；TP2＋PD、FP8＋TP2、FP8＋PD 等组合只记录实际状态，不要求全部补齐。
- 当前 FP8 为 `float8_e4m3fn` KV 存储、运行时量化、attention 读取及 PD data/scale 传输，模型权重和 activation 仍为 BF16；没有接入预量化 FP8 权重或 FP8 GEMM。审计须核对权重格式、scale、布局、内核及来源，交付材料明确写“FP8 KV cache 适配”。
- 允许复用或移植 vLLM、SGLang 成熟实现，保留许可证、来源和实际适配说明。优先复用现有启动、正确性和测量工具，不开发完整实验控制器。
- 只修复直接影响验收的问题。结构性重做需报告工作量和未完成边界，优先交付其余成立的功能；不为凑齐功能清单无限扩展。
- 不要求发现新的优化点、逐层流水收益、战胜 vLLM、所有组合消融、最大 goodput、Math500 或重新搭建完整外部框架性能对比。逐层传输等已有代码保留，只有容易验证时才补充记录。仍需一段与正式测量分开的短 profile。
- 使用已提供的机器和环境，不额外租机或改变计费。需要开启 4×L40S 实测时通知用户；只有简历要写四卡部署时，才集中完成真实四卡验证和测量。

代码版本、配置、请求和原始结果每批保存一次即可，不增加反复 SHA、快照、标签或审批检查。主 agent 直接实施，不强制多 agent。当前范围优先于后文历史阶段与研究路线；保留已有修改，比赛项目不在本轮改动范围。真实状态和代码入口见 [项目说明与支持矩阵](inference_optimization_overview_zh.md)，执行清单见 [任务](../.codex/tasks/m8.md) 和 [验收](../.codex/tasks/m8_acceptance.md)。

### 项目名称

**基于 mini-vLLM 的单机多卡大模型推理框架开发**

基座为开源 MinivLLM，保留上游 Apache-2.0 许可证和版权信息。基础模型、TP 层、调度与 kernel 不直接归为个人原创。

### 最终交付物

1. 可运行代码、必要修复、BF16/FP8 KV 配置和复现命令。
2. 功能与组合支持矩阵、已知问题和未支持范围。
3. 正确性记录、端到端原始数据与配置、一段代表性短 profile。
4. 简洁 README／报告：架构、个人实现及借鉴来源、验证方法、性能与资源结果。
5. 三条可直接复制的简历项目描述；数字仅取真实结果，可写绝对性能和资源占用。
6. 三分钟项目讲解，以及紧贴实际代码、失败和限制的常见追问与回答。

## 2. 硬件和部署约束

- 当前已准备的租机目录为 `/Data2/zincion/ai-infra/l40s-20261117`，计划使用 4×L40S；最后一次准备记录为无卡模式，不能据此确认已开卡或设备拓扑。Python、模型及短请求配置已准备，开卡前不重复安装下载。
- FP8 KV TP1 使用一卡，TP2 或 1P1D 使用两卡；四卡部署只在实际需要时验收。不同卡数必须在表中明确列出，不能直接计算成同资源加速比。
- 旧服务器为 5 张可见显存 `49140 MiB` 的 RTX 4090，GPU0–2 与 GPU3–4 分属两个 NUMA 域，历史拓扑见 [硬件记录](server_hardware_2026-11-09.md)。这组设备号与 NUMA/P2P 结论不能套用到 L40S；现有生成器的 GPU0/1/3/4 假设须按现场映射适配。
- 服务器工作目录为 `/Data2/zincion/ai-infra`，新增模型缓存、实验数据、日志及临时文件均存放于此，避免占用根分区；详细硬件记录见 [server_hardware_2026-11-09.md](./server_hardware_2026-11-09.md)。

## 3. 模型与精度

- 主线模型：**Qwen3-14B（dense，GQA）**。
  - 沿用 `configs/qwen3_14b.json` 的固定模型/tokenizer revision；不要因某项仅通过 tiny 或 0.6B 就称 14B 通过。
  - 当前本地模型读取记录为 8 个 safetensors 分片、443 个 BF16 张量、29,536,614,400 字节权重；这是资产检查，不是 GPU 加载验收。
  - 原 M2 已有 14B BF16 TP1 加载及七项输入的 HF 对照，正式 logits 容差未定；最终代码和 L40S 上需复验，见 [M2 实测](m2_measurements_2026-11-14_zh.md)。
- 精度：BF16 为参考；当前量化实现仅为 FP8 KV cache（M7），不是预量化权重接入。分别记录模型权重字节、KV data＋scale、运行时总显存，不把预分配 KV 池或回退路径当成 FP8 加速。
- 其他模型架构不进入本轮，M9 仅保留为可选扩展。

## 4. 基座已有能力与归属边界

上游仓库已经提供以下基础代码：

- Qwen3-0.6B 和 Llama-3.2-1B 的模型实现及权重加载入口。
- Prefill/Decode 推理流程、请求调度、KV block 管理和 CUDA Graph 路径（上游问题及本项目修复、验证范围见 M3）。
- Prefill FlashAttention、Decode Paged Attention 的 Triton 实现及独立 kernel benchmark。
- NCCL 进程组、进程 worker、张量并行线性层和简单 TPS 对照脚本。

这些内容可以作为项目基座和系统背景，但不能直接表述为本人的新增实现。个人贡献必须落在后续改造、缺陷修复、测试和可复现实测结果上。

## 5. 本轮验收与测量口径

### 最小功能验收

| 路径 | 必要检查 | 边界 |
|---|---|---|
| BF16 TP1 | Qwen3-14B 加载、少量固定代表输入、greedy 及必要 logits | 与可信 HF/已有参考核对，作为其他路径的参考 |
| BF16 TP2 | 权重分片、rank/进程组织、collective，同输入对照可信 TP1 | 记录最大/平均 logits 误差、同前缀 top1、首个 token 分歧与容差依据；保留原 tiny 失败，不放宽门槛掩盖异常 |
| bulk P/D | 独立 worker、有效 KV 长度、分页边界、请求映射、恢复与接管、少量并发交错、结束/取消回收 | P 生成首 token，交接 prompt 的 KV；D 在位置 N 处理首 token 并生成第二个，不能重复或漏发 |
| FP8 KV TP1 | 权重/scale/布局/kernel 审计；少量固定输入的 finite、logits 和输出对 BF16 检查 | 同格式 PD 字节一致性与 BF16→FP8 有损差异分开；短样例不证明任务精度无损 |
| 组合 | TP2＋PD、FP8＋TP2、FP8＋PD 等支持矩阵 | 只标实际验证状态，不强求全部完成；TP2 P bulk 与 TP>1 D、TP2 pipeline 分开 |

旧 tiny 容差不能直接当作 14B 阈值。logits 在相同前缀比较，记录归约/舍入与 top1/top2 margin，定位异常；报告生成或 capture-only 不等于验收通过。若需要结构性重做，报告工作量及未完成边界，先交付其余成立路径。

### 集中小规模端到端测量

优先比较实际可运行的 BF16 TP1、BF16 TP2、FP8 KV TP1 和 BF16 P/D（可用双卡 1P1D）。失败或未通过正确性的路径标为失败/待验证，不用正常性能行掩盖。不同卡数与部署方式在结果中逐行标注。

- 从固定的约 4096 输入、128 输出开始；容量不足时明确调整并另存实际长度。并发 1 和一个可运行并发点（如 4）即可，不做参数扫描。先做有限预热，保存预热次数、正式请求数及实际生成 token 数。
- 各路径共用模型与 tokenizer 版本、实际 token 输入、输出预算、greedy/ignore-eos 策略和缓存设置。FP8 KV 是明确的比较变量；block 池大小与 eager/graph、whole/chunk 设置有差异时注明，不混写成单变量收益。
- 平均 TTFT 从客户端实际发出请求到首输出 token；平均 TPOT 按单请求 `(末 token 时间−首 token 时间)/(输出 token 数−1)`，单输出请求为不适用。只对可识别的输出事件计时，chunk 间隔不冒充逐 token ITL；失败计入请求分母，延迟均值说明仅含成功且可计时请求的数量。
- 输出吞吐为成功请求的输出 token 总数/正式批次墙钟时间；总 token 吞吐为成功请求的输入＋输出 token 总数/相同时间。记录批次起止和是否包含完成尾巴，排除预热，不把它们称为最大 goodput。
- 显存分别记录权重字节、KV data＋scale/实际 block 池和运行时显存。说明使用每进程 PyTorch allocated/reserved/peak，还是设备级 nvidia-smi 采样及其间隔；多个进程/卡的数值按口径汇总，采样最大值不称精确峰值。
- 每行记录请求数、成功数、失败数/原因、GPU 数和部署方式；PD 必要时记录客户端首二 token 间隔，内部 export/restore/handoff 的起止单独说明。
- 保留请求列表、配置、命令、环境和原始逐请求结果；优先复用现有服务、bench 与分析入口，不要求完整自动控制器或外部框架性能对比。短样本不包装成稳定尾延迟或普适结论。

### 代表性短 profile

在一个已跑通的代表配置上，另起一次短 PyTorch Profiler 或 Nsight 采集，保留 trace、配置、命令和可解释的时间区间。说明计算 kernel、通信/复制和主机或设备等待的大致分布；若该配置无 TP 通信或无 PD 传输则明确说明。多 rank 只采到部分进程时注明覆盖限制，不相加重叠区间制造百分比。profile 的耗时不混入正式性能结果，不要求逐层深挖或新增优化。

## 6. 历史模块路线（非本轮完整验收清单）

以下保留原阶段技术路线及其研究目标，不表示模块已全部验收。第 1、5 节覆盖这里的旧全矩阵、SLO/goodput、质量基准与弹性调度要求；本轮按最小功能交付，已有扩展代码保留。

### M0：基线、许可证和可复现入口（已完成）

- 固定上游基线 commit、模型 revision、dtype、tokenizer、采样参数、随机种子和测试输入；统一 JSON 配置。
- 见 [m0_m1_baseline_zh.md](./m0_m1_baseline_zh.md)。

### M1：调度和 KV Cache 正确性（已完成）

- 修复 block 引用计数、释放后的 hash 元数据和 block 生命周期问题；支持取消；请求长度在入口处约束。
- 用真实 BlockManager 和 Scheduler 测试 free/used block 守恒和请求守恒。

### M2：Qwen3-14B 路径与测量底座

- 增加 Qwen3-14B 配置并跑通单卡推理，与 Transformers 参考实现对比 greedy 输出和 logits 误差。
- **OpenAI 兼容流式服务**：实现 `/v1/completions` 和 `/v1/chat/completions`，支持流式返回，使 `vllm bench serve` 和 evalscope 能直接测量本框架。TTFT/ITL 依赖流式返回的 chunk 计时，接入后用单请求实测确认计时口径与 vLLM 一致。当前框架只有离线 `generate()`，服务层需要把同步 step 循环改为可持续接收请求的引擎循环。
- **容量测算与配置**：按单卡显存、权重大小和每 token KV 大小算出可容纳的 token 数，据此设计 L1/L2 用例的长度和并发。Qwen3-14B BF16 每 token KV 为 160 KiB；GPU0 在 M2 配置下实测得到 367 块、14.3359375 GiB KV、93,952 token，按 block 取整后的 4K+1K、16K+1K 请求容量上界分别为 18、5，并非稳定并发保证，见 [实测环境与推导](m2_measurements_2026-11-14_zh.md)。M2 配置已将 `max_model_length`、`max_num_batched_tokens` 均设为 17408；在 M3 实现 chunked prefill 之前，Scheduler 要求 `max_model_length ≤ max_num_batched_tokens`。
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

- KV cache 以 FP8 存储，attention kernel 读取时反量化；PD 传输直接传 FP8 数据及其 scale，实际字节数需包括 scale 和填充。
- 4090（Ada 架构）支持 FP8，但本项目的自定义 attention kernel 仍需实现相应读取与反量化。
- 原研究验收还计划比较 decode batch、传输时间、goodput，并以 evalscope 测推理类和长上下文任务精度；这些完整质量与性能矩阵不是本轮前提。本轮按第 5 节做代表输入、NaN/Inf、资源和端到端记录。参考 SpectrumKV、HACK 等 KV 传输压缩工作。

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

## 7. 本轮实施顺序

1. 审计当前实现和证据，先报告 TP2、PD、FP8 的真实状态；保留旧失败，更新支持矩阵。
2. 无卡阶段整理依赖、模型、最小配置、命令和 CPU 检查；准备好后通知用户开启 4×L40S，串行复用 GPU，不改变计费。
3. Qwen3-14B BF16 TP1 参考 → BF16 TP2 分歧定位 → 双卡 BF16 1P1D bulk → FP8 KV TP1；只做必要修复和相关测试。组合按实际情况记录，不要求全部补齐。
4. 在可运行且通过对应正确性检查的路径上集中测固定负载，另采一段短 profile；需要“四卡”简历表述时再补真实四卡部署记录。
5. 最终集成一次完整 CPU 回归与相关 GPU 验证，整理报告、复现入口、三条简历描述、三分钟讲解和问答。未验证或未解决的部分保持明确状态。

## 8. 简历事实边界

- 原仓库已有的分页注意力、基础调度器、基础 TP 代码和 kernel benchmark，只能写成项目基座或系统已有能力。
- B1（chunked prefill）是本项目对已有方法的实现，用作基线；不能写成本人提出的方法。
- 借鉴的方法（逐层传输、弹性 P/D 等）要写成“借鉴/实现”，不能写成原创。
- vLLM PD 分离的官方定位是控制 TTFT 和尾延迟，而不是提升吞吐。没有实测数据时，不写“PD 分离提升吞吐”。
- “实现 PD 分离”“支持 FP8 KV”“提升 XX%”只有在代码、测试、日志和对照数据齐全后才能写。
- 所有数字必须来自保留原始结果的可复现实验。提升比例需可比较的同资源对照；绝对性能可以使用，但须写清模型、GPU 数、部署、精度与负载。未完成能力和旧模型结果不能包装成当前模型成果。
- 不要求正加速比；可按实际多卡执行、KV 状态交接、FP8 KV 适配与验证交付。没有充分质量证据不写“无精度损失”，没有真实四卡记录不写“四卡部署已完成”。

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
