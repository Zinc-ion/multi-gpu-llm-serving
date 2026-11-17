# Qwen3-14B：L40S 最小验收与交付

本报告记录 2026-11-18 的实际运行。修复前推理源码基线为 `65906d2`，生产修复为 `dca7522`，包括 CUDA BF16 TP 的 Row Linear 部分积精度及 HTTP TP2 入口。本轮另增加短 profile 工具。**原 tiny TP2 严格失败已修复，Qwen3-14B 的独立 TP1/HF、TP2/TP1、TP1＋TP1 bulk PD 和 FP8 KV 完成固定输入检查；四卡 TP2＋PD 仍有数值分歧，通用 logits 门槛及更广质量结论未建立。**

项目定位、组合矩阵见 [项目说明](inference_optimization_overview_zh.md)，可复制的三条简历、三分钟讲解和问答见 [面试材料](project_interview_zh.md)。本轮交付开发原型、可复现实验与限制，不声称商用可用或优于成熟框架。

## 环境与统一配置

- 4×L40S，`nvidia-smi` 每卡报告 46,068 MiB；driver 610.43.02。GPU0/1 同 NUMA0、互连 PXB，GPU2/3 同 NUMA1、互连 PXB，两组间 SYS。P2P 读取状态均为 OK，这不是实测传输带宽。
- 开卡后 cgroup 配额为 52 CPU、502,511,173,632 字节内存。Python 3.11.16、PyTorch 2.9.1+cu128、Triton 3.5.1、Transformers 4.57.3。
- 模型/tokenizer：Qwen3-14B，revision `40c069824f4251a91eefaf281ebe4c544efd3e18`。8 个 BF16 safetensors 分片、443 个 checkpoint 张量；加载检查覆盖 323 个模型参数及所有 checkpoint 张量。
- 当前数据盘任务目录为 `/root/data/zincion/ai-infra/l40s-20261117`，模型、环境、缓存和原始结果均在数据盘；学校服务器的 `/Data2` 限制不适用于此租机。
- `block_size=256`、80 blocks、最多 4 序列、长度与 batch token 上限 4352、eager、whole prefill、禁用跨请求 prefix 复用。模型权重和 activation 为 BF16；FP8 KV 使用默认 `fp8_prefill_compute_dtype=float32`。

硬件原始记录、实际配置、输入和测量文件位于 [结果目录](results/l40s-20261118/README.md)。软件及模型沿用已有环境，没有重新下载模型或安装完整 vLLM 服务端。

## 正确性与数值边界

固定输入为两个短 chat，以及 255、256、257、4096 token 的位置/分页压力输入。255/256/257 检查一页前后边界；长输入是固定重复文本，不能代表多样化任务质量。表中 token 位置从 1 开始。

| 路径 | 实际检查 | 结果及限制 |
|---|---|---|
| BF16 TP1 对 HF | 六组独立 greedy，各最多 16 输出；HF BF16 SDPA flash；另按 custom 前缀 teacher-forced 检查 logits | 6/6 生成序列一致；同前缀 top1 为 95/96，最大绝对误差 5.53125。独立生成一致不等于 logits 逐位相同 |
| BF16 TP2 修复前对 TP1 | 相同六组输入，各 16 输出；实际双进程与 NCCL | 5/6 生成序列一致；chat-1 第 10 个输出由 3070 变为 3460，TP1 top1/top2 gap 为 0；最大同前缀绝对误差 9.390625。严格 greedy 检查失败 |
| BF16 TP2 修复后对 TP1 | 原输入、采样和输出预算，CUDA BF16 Row GEMM 保留 FP32 部分积与归约 | 6/6 序列一致，同前缀 top1 为 96/96；最大绝对误差 9.671875。输出检查通过，不声称所有 logits 误差都下降或已达到通用精度门槛 |
| BF16 TP1 P＋TP1 D bulk | 六组各 4 输出，独立 GPU；导出与恢复前有效 KV 原始字节比较 | 6/6 输出一致、有效 KV 不同字节数为 0；最大同前缀 logits 误差 0.908203125。4K 交接有效 KV 为 671,088,640 字节 |
| BF16 PD 生命周期 | 六组串行各 16 输出、三组并发交错、单输出、取消后再请求 | 对应输出与 TP1 一致；最后 open handles、used blocks、reserved transfer bytes 均为 0 |
| FP8 KV TP1 对 BF16 KV | 相同六组，各 16 输出；BF16 权重，运行时 E4M3FN KV | 6/6 输出一致，logits 全部 finite；最大同前缀绝对误差 8.796875。仅支持该小样本结论，不声称无精度损失 |

**判定口径。** BF16 greedy 保留严格 token 序列比较，修复没有修改输入、seed、容差或判定。14B 没有事先建立可靠的通用 `atol/rtol`，本次只报告误差和输出检查，不从结果反推一个容差来“通过”。tiny 原有 BF16 `rtol=.05, atol=.02` 仅用于该单元测试，不能移作 14B 阈值。FP8 为有损比较，finite、输出、同前缀误差分别记录；本次恰好输出一致也不能作为未来所有输入的要求或保证。

同前缀 logits 的口径也有区别：HF 工具对 custom 前缀重算一次 teacher-forced forward；TP1/TP2 与两种 KV 比较只计算首分歧预测位置及之前的行，分歧后的不同前缀不混算误差。HF 的独立生成和一次重算本身可以因执行形状而不同，因此“6/6 生成一致”和“95/96 top1”不矛盾。

### TP2 原严格失败

修复前在 L40S 重新运行原 seed、prompt、dtype、测试判定，原测试仍失败：HF/TP1 输出 `[32,33,41]`，TP2 输出 `[32,33,62]`。最后一步 TP1 的 token 41 和 62 均为 `0.34765625`，TP2 分别为 `0.34765625`、`0.349609375`。实际进程和 collective 完成，不是以 skip 或通信未执行冒充 TP2 验证。

独立重放确认第 0 层 `o_proj` 输入和权重分片精确一致。完整 BF16 GEMM 精确复现 TP1，分片 BF16 GEMM 后求和精确复现 TP2，连同全部 113/320 个差异元素及最大误差 `0.000244140625`。FP32 完整/分片 GEMM 的最大差为 `7.45e-9`，转回 BF16 后一致。将 Row 计算改为 FP32 的独立诊断也使 tiny 输出恢复一致。证据见 [重放记录](results/l40s-20261118/tp-row-replay-summary.json)。

生产修复只对 **CUDA、BF16、TP>1 的 Row Linear** 保留 FP32 部分积：`torch.mm` 仍使用 BF16 输入和权重、以 FP32 输出，经 FP32 all-reduce 后再转回 BF16，加 bias 的顺序保持不变。没有将整份权重临时转为 FP32，没有修改采样或其他 TP1/CPU/FP32 路径。代价是 Row 归约 payload 从 2 字节/元素变为 4 字节/元素；这是数值修复，不是通信加速。

修复后直接重跑原 `tests/test_tp_gpu.py`，FP32/BF16 **2 passed，71.52 秒**；BF16 TP1/TP2 均输出 `[32,33,41]`，对 HF 最大绝对误差均为 `0.0029296875`，满足原固定容差。随后 14B 六组 greedy 和 96 个同前缀 top1 一致。仍保留修复前失败，且不把一个边界的重放外推成所有层、所有输入都精确等价。

共置 HTTP 曾在构造时拒绝 `world_size=2`，与底层已有 TP2 engine 不一致。本轮先以新增公开接口回归复现，再将入口范围从 1 放开到 1/2；计算、调度和命令仍由同一个 owner 线程执行，PD 网关仍使用自己的 TP1 配置。修复后的真实 HTTP 测量另保存，不覆盖原启动失败日志。

### PD 接管契约与实际检查

长度 N 的 prompt 由 P 完成 prefill 并生成首 token。导出的有效 KV 覆盖 `[0,N)`，不含尚未作为输入的首 token。D 用自己的 block table 恢复，再在位置 N 输入首 token 并生成第二个。记录核对 request ID、completion 计数、prompt 长度与 client/callback token；首 token 已完成的请求不做无用交接。

`compare_pd.py capture --raw-kv` 对有效 KV 逐字节核对，覆盖尾块和整页；没有把未写入的池或填充区域当成有效 KV。独立 worker 的 source、PID、cache shape 与设备信息保存在原始 role metadata 中。额外实际请求检查并发交错、正常完成、Decode 取消及后续可继续运行。EOS 的确定性覆盖另由 tiny GPU 用例记录，不将 tiny 的 EOS 测试改写成 14B EOS 质量评测。

**四卡组合的已知问题。** TP2 P 占 GPU0/1，两个 TP1 D 分别占 GPU2/3，已实际执行串行、并发、单输出及取消后恢复，最终三项资源计数均为 0。但 Row 修复后六组串行仍只有 5/6 与 TP1 一致：`chat-1` 第 10 个输出由 3070 变为 3460；三组并发输入均一致。该组合的进一步根因未证明，不能把独立 TP2 的修复结论外推为 TP2＋PD 通过。其吞吐和延迟只作诊断，保留原始结果，不为凑齐组合扩大本轮范围。

## FP8 实现和显存口径

量化发生在运行时 KV 写入，非接入预量化权重。K/V 分别按 token/head 求 scale，采用 E4M3FN 数据与 FP32 scale；全零 head 使用 scale 1。Triton attention 根据实际缓存格式读取并反量化，默认 prefill 的 Q/K/V 中间计算为 FP32、dot 使用 IEEE 精度；没有 FP8 GEMM。可选 `query` 模式未列为本次通过项。

| 项目 | BF16 KV TP1 | FP8 KV TP1 | 口径 |
|---|---:|---:|---|
| 模型参数 | 29,536,614,400 B | 29,536,614,400 B | 实际模型参数，均 BF16 |
| KV data 每 block | 41,943,040 B | 20,971,520 B | 40 layers、8 KV heads、head_dim 128、256 tokens |
| scale 每 block | 0 | 655,360 B | FP32，每个 K/V token/head 独立 |
| 80 blocks 的 KV data＋scale | 3,355,443,200 B / 3.125 GiB | 1,730,150,400 B / 1.611328125 GiB | 固定预分配池，两者容量均为 20,480 token |
| 每个有效 token 的 KV data＋scale | 163,840 B | 84,480 B | 不含尾块填充及运行时其他内存 |

固定容量的 KV data＋scale 减少 **48.4375%**；这是分项存储结果，不是模型总显存减半或推理加速。RoPE 缓存每个实例另占 838,860,800 B。TP2 每个 rank 的 KV heads 减半，norm 等参数仍复制；不能简单用全模型参数字节除二代表所有运行时内存。

## 小规模端到端测量

统一使用实际 4096 token 的重复文本输入、128 输出、greedy、ignore-eos；每档并发 1 或 4，先预热 2 请求，再正式测量 8 请求。另对 BF16 TP1、修复后 TP2 的并发 4 各补独立 32 请求批次，每批仍先预热 2 次。输入在各路径相同、prefix 复用禁用；8 请求批次成功输入/输出总数为 32,768/1,024 token，32 请求批次为 131,072/4,096 token。两类批次不拼接或平均分位数。

TTFT 从请求函数实际发出 HTTP 请求到首个有 choices 的 SSE 输出，包含服务端排队；TPOT 为每个成功请求 `(最后输出时间−首输出时间)/(输出数−1)` 后求平均。输出吞吐为成功输出 token/正式批次墙钟，总吞吐为成功输入＋输出 token/同一墙钟；墙钟包含完整完成尾巴，排除预热，也包含客户端并发槽位等待。流式函数原样复用 vLLM 0.15.0，版权与许可证保留在结果资产中。

TTFT/TPOT 均记录 mean、p50、p95、p99。分位数采用排序后 `(n-1)q` 位置的线性插值，每请求 TPOT 使用 `tpot_s` 字段，单输出为不适用。各请求实际长度、成功/失败、原始时间与 SSE 间隔均在逐请求 JSON 中；输入长度由固定 tokenizer 回环确认，实际输出长度取服务端 usage。汇总提供 JSON 和 CSV，字段及长度来源见 [结果说明](results/l40s-20261118/README.md)。

| 部署 | 卡数 | 并发 | 请求/成功/失败 | 实际输入/输出 | 输出 token/s | 输入＋输出 token/s |
|---|---:|---:|---|---|---:|---:|
| BF16 TP1 | 1 | 1 | 8/8/0 | 4096/128 | 18.768 | 619.345 |
| BF16 TP1 | 1 | 4 | 8/8/0 | 4096/128 | 50.469 | 1665.472 |
| BF16 TP1 补测 | 1 | 4 | 32/32/0 | 4096/128 | 54.459 | 1797.140 |
| BF16 TP2 | 2 | 1 | 8/8/0 | 4096/128 | 14.731 | 486.112 |
| BF16 TP2 | 2 | 4 | 8/8/0 | 4096/128 | 46.054 | 1519.784 |
| BF16 TP2 补测 | 2 | 4 | 32/32/0 | 4096/128 | 47.515 | 1567.980 |
| FP8 KV TP1 | 1 | 1 | 8/8/0 | 4096/128 | 4.024 | 132.781 |
| FP8 KV TP1 | 1 | 4 | 8/8/0 | 4096/128 | 4.715 | 155.580 |
| BF16 1P1D | 2 | 1 | 8/8/0 | 4096/128 | 8.562 | 282.559 |
| BF16 1P1D | 2 | 4 | 8/8/0 | 4096/128 | 17.365 | 573.045 |

TTFT/TPOT 单位为 ms，下表每格依次为 **mean / p50 / p95 / p99**；每行两项计时样本数均等于成功请求数。

| 部署 | 并发 / 样本 | TTFT：mean / p50 / p95 / p99 | TPOT：mean / p50 / p95 / p99 |
|---|---|---|---|
| BF16 TP1 | 1 / 8 | 647.445 / 648.410 / 652.257 / 652.639 | 48.594 / 48.558 / 48.806 / 48.813 |
| BF16 TP1 | 4 / 8 | 1628.297 / 1629.047 / 2599.886 / 2604.922 | 67.041 / 67.079 / 79.458 / 80.879 |
| BF16 TP1 补测 | 4 / 32 | 1617.273 / 1610.993 / 2595.577 / 2601.151 | 61.285 / 61.821 / 68.840 / 69.819 |
| BF16 TP2 | 1 / 8 | 644.287 / 644.015 / 646.322 / 646.985 | 63.338 / 62.845 / 65.631 / 65.713 |
| BF16 TP2 | 4 / 8 | 1574.214 / 1573.924 / 2500.910 / 2501.890 | 75.124 / 75.123 / 82.516 / 82.703 |
| BF16 TP2 补测 | 4 / 32 | 1573.421 / 1572.710 / 2500.137 / 2504.354 | 72.450 / 73.868 / 79.960 / 84.204 |
| FP8 KV TP1 | 1 / 8 | 25417.131 / 25413.290 / 25439.907 / 25445.087 | 50.342 / 50.368 / 50.641 / 50.690 |
| FP8 KV TP1 | 4 / 8 | 63499.647 / 63504.005 / 101574.075 / 101579.023 | 355.103 / 355.071 / 655.197 / 655.660 |
| BF16 1P1D | 1 / 8 | 660.467 / 660.848 / 667.966 / 669.624 | 112.500 / 113.342 / 119.549 / 121.078 |
| BF16 1P1D | 4 / 8 | 2502.271 / 1626.906 / 7359.443 / 9539.725 | 209.538 / 208.923 / 239.571 / 240.934 |

**四卡诊断数据：该组合数值检查未通过，不列作已验收路径性能。** 修复后 TP2 P＋两个 TP1 D 的两批均为 8/8 HTTP 完成、0 失败、4096/128 实际长度；并发 1/4 的输出吞吐为 8.654/20.612 token/s，总吞吐为 285.570/680.199 token/s。完整 mean、p50、p95、p99 仍在 JSON/CSV 中保留。

8/32 请求的 p95/p99 是经验分位数，受极少数请求影响，不代表稳定尾延迟、SLA、最大 goodput 或容量上限；输入相同也不能当成多样化任务样本。正式 TP2 行来自修复后；四卡修复前后均只列诊断，不与其他结果混合。卡数不同，不计算同资源加速比。FP8 默认 FP32 prefill 在此负载上显著更慢，存储收益与延迟代价分别报告。

### 设备显存统计

| 部署 | 档位 | 每卡 device used 采样最大值 / MiB | 同时刻合计最大值 / MiB | 实际平均采样间隔 / ms |
|---|---|---|---:|---|
| BF16 TP1 | c1 / c4，各 n=8 | GPU0: 33,535 | 33,535 | 282.0 / 284.2 |
| BF16 TP2 | c1 / c4，各 n=8 | GPU0: 17,957, GPU1: 17,939 | 35,896 | 293.2 / 287.2 |
| FP8 KV TP1 | c1 / c4，各 n=8 | GPU0: 32,811 | 32,811 | 281.1 / 280.1 |
| BF16 1P1D | c1 / c4，各 n=8 | GPU0: 33,555, GPU1: 32,755 | 66,310 | 290.7 / 292.9 |
| 四卡 PD（诊断） | c1 / c4，各 n=8 | GPU0: 17,957, GPU1: 17,941, GPU2: 32,755, GPU3: 32,755 | 101,408 | 288.9 / 288.4 |
| BF16 TP1 补测 | c4，n=32 | GPU0: 33,535 | 33,535 | 280.3 |
| BF16 TP2 补测 | c4，n=32 | GPU0: 17,957, GPU1: 17,939 | 35,896 | 284.1 |

PD 首二输出 SSE 间隔的均值（每档 n=8）：

| 部署 | 并发 1 / ms | 并发 4 / ms |
|---|---:|---:|
| BF16 1P1D | 7416.774 | 14806.146 |
| 四卡 PD（诊断） | 7310.815 | 10795.583 |

设备值来自独立 `nvidia-smi` 采样，目标等待间隔 200 ms，实际平均间隔保存在每行原始摘要中；采样范围为服务启动完成后，该并发档的预热及正式请求。它是设备 used memory 的采样最大值，不是精确峰值，也不是 PyTorch allocated。多卡合计使用同一个样本时刻的设备和；单卡最大值相加不冒充同时峰值。启动 warmup 的 PyTorch peak/current 则保存在 capacity 原始日志，发生在 KV 分配前，不与设备运行值混用。

PD 首二输出的客户端 SSE 间隔保留在上表和 `performance-summary.json`，不能把跨进程绝对时钟直接相减作为传输时间。它包含 bulk 交接、排队及 Decode 激活。当前 bulk 经 D2H、主存/TCP、H2D；TP2 P 另经 Gloo 汇总完整 heads，没有 GPUDirect/RDMA 路径。

## 独立短 profile

在独立 TP2 BF16 eager 配置中，先运行一个未采集的同形请求，再采 **4096 输入、8 输出、并发 1**，覆盖两个 rank。每 rank 记录 1 次 prefill、7 次 decode，`compile_inner` 事件为 0；原始 Chrome trace 压缩随结果交付。profiler 会改变执行耗时，下面不与 HTTP 正式数据混算。

| rank | GPU 首尾跨度 / ms | GPU 活动区间并集 / ms | 无 GPU 活动区间 / ms | 计算 kernel 累计 / ms | elementwise 累计 / ms | NCCL 累计 / ms |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1475.023 | 1456.204 | 18.819 | 431.416 | 43.268 | 980.023 |
| 1 | 1472.091 | 794.151 | 677.940 | 431.789 | 42.548 | 318.363 |

每个 rank 有 656 个 NCCL kernel 事件：640 次 Row FP32 all-reduce、8 次 embedding BF16 all-reduce 和 8 次 logits gather。trace 中实际出现 `ncclDevKernel_AllReduce_Sum_f32_RING_LL`，确认修复后的归约 dtype 生效。名称分类是近似归属，未识别算子保留 other，明细在 `profile-summary-fixed.json`。

计算累计约 431 ms/卡；rank0 NCCL 时长明显高于 rank1，而 rank1 无 GPU 活动区间更多，说明执行中存在通信及等待不对称。NCCL kernel 包含等待对端的时间，不等于纯 PCIe 传输耗时；无活动区间可能包含主机调度和同步，不能全部算作可优化等待。不同类别可能重叠，不跨 rank 相加或据此宣称优化收益。本段没有采 PD worker/TCP 交接。

## 复现与数据位置

已有机器无需重新安装依赖或下载模型：

```bash
source /root/data/zincion/ai-infra/l40s-20261117/env.sh
cd "$PROJECT_ROOT/repo"
export ACCEPTANCE_ROOT="$PROJECT_ROOT/acceptance-ready"
export REPRO_ROOT="$PROJECT_ROOT/runs/repro-20261118"
mkdir -p "$REPRO_ROOT"

CUDA_VISIBLE_DEVICES=0 python tools/compare_hf.py custom \
  --config "$ACCEPTANCE_ROOT/configs/tp1-bf16.json" \
  --prompts "$ACCEPTANCE_ROOT/workloads/correctness.json" \
  --max-tokens 16 --output "$REPRO_ROOT/tp1-bf16.pt"
CUDA_VISIBLE_DEVICES=0 python tools/compare_hf.py reference \
  --config "$ACCEPTANCE_ROOT/configs/tp1-bf16.json" \
  --custom "$REPRO_ROOT/tp1-bf16.pt" --max-tokens 16 \
  --attention sdpa --sdpa-backend flash --output "$REPRO_ROOT/tp1-hf.json"
```

TP2 和 FP8 KV 分别将 custom 配置换为 `tp2-bf16.json`、`tp1-fp8-kv.json`。TP2 同格式对照使用 `tools/compare_runs.py`；BF16→FP8 的有损对比见保存的 CPU 比较脚本，不伪造 KV 格式相同以绕过工具的检查。

双卡 bulk 正确性采集（该工具会先运行并退出 reference，再启动独立 P/D；输出目录必须尚不存在）：

```bash
python tools/compare_pd.py capture \
  --engine-source "$PWD" --expected-sha "$(git rev-parse HEAD)" \
  --reference-config "$ACCEPTANCE_ROOT/configs/tp1-bf16.json" --reference-gpu 0 \
  --prefill-config "$ACCEPTANCE_ROOT/configs/pd-prefill-bf16.json" --prefill-gpu 0 \
  --decode-config "$ACCEPTANCE_ROOT/configs/pd-decode-bf16.json" --decode-gpu 1 \
  --prompts "$ACCEPTANCE_ROOT/workloads/correctness.json" \
  --max-tokens 4 --ignore-eos --transfer bulk --raw-kv \
  --max-transfer-bytes 3221225472 --output-dir "$REPRO_ROOT/pd-bulk" \
  --startup-timeout 240 --request-timeout 180 \
  --normal-exit-timeout 60 --term-timeout 10 --kill-timeout 10
```

共置 HTTP 服务一次只启动一种配置：

```bash
CUDA_VISIBLE_DEVICES=0 python -m myvllm.server \
  --config "$ACCEPTANCE_ROOT/configs/tp1-bf16.json" \
  --served-model-name Qwen3-14B --host 127.0.0.1 --port 18000
```

另一个 shell 运行正式小批次，并独立采设备显存：

```bash
python "$ACCEPTANCE_ROOT/client/measure_small_batch.py" \
  --config "$ACCEPTANCE_ROOT/configs/tp1-bf16.json" \
  --dataset "$ACCEPTANCE_ROOT/workloads/4k-128.jsonl" \
  --url http://127.0.0.1:18000 --model Qwen3-14B \
  --label bf16-tp1-c1 --gpu-ids 0 --concurrency 1 --requests 8 --warmups 2 \
  --output "$REPRO_ROOT/bf16-tp1-c1.json"
```

所有实际角色启动、GPU 映射、双卡和四卡网关参数见结果资产的 `run_small_acceptance.py` 与 `measurement-commands.json`，可以直接按对应命令手工启动。该脚本仅负责本次小批次的顺序启动、请求、采样与退出，不要求使用旧全矩阵控制器。

### 直接使用官方 vLLM benchmark

本轮租机只安装轻量 HTTP 客户端，未安装完整 vLLM。上面的正式数据使用未修改的 vLLM 请求函数与薄封装，不声称已运行完整官方 CLI 或赛事专用脚本。本地安装的 vLLM 0.15.0 已核对 CLI 与原始统计实现；使用其 `calculate_metrics` 对保存请求独立复算，结果见 `vllm-metrics-check.json`。

在已有 vLLM 0.15.0 客户端、可读取 tokenizer 并访问服务端的环境，可以直接测同一个 `/v1/completions`，服务端仍为 mini-vLLM。以下是替代入口模板，不与本轮已实测批次混写：

```bash
vllm bench serve \
  --backend openai --endpoint /v1/completions \
  --base-url http://127.0.0.1:18000 --model Qwen3-14B \
  --tokenizer "$PROJECT_ROOT/models/Qwen3-14B" \
  --dataset-name custom \
  --dataset-path "$ACCEPTANCE_ROOT/workloads/4k-128-32.jsonl" \
  --custom-output-len -1 --skip-chat-template --disable-shuffle \
  --num-prompts 32 --no-oversample --num-warmups 2 \
  --request-rate inf --max-concurrency 4 \
  --temperature 0 --ignore-eos --seed 7 \
  --percentile-metrics ttft,tpot --metric-percentiles 50,95,99 \
  --save-result --save-detailed --result-dir "$REPRO_ROOT" \
  --result-filename vllm-client-c4-n32.json
```

官方生成场景的 `total_token_throughput` 为 `(成功输入＋成功输出 token)/测量时长`，同时可保存输出吞吐和延迟分布；每卡显存仍需另行采样，PD 首二间隔可从详细 ITL 保留。比赛示例的专用脚本未包含在本项目，不能只凭命令认定其实现和口径完全相同。MiniCPM 的 4K/1K、16K/1K、并发 64 也不直接移作当前 14B 配置：本次 `max_model_length=4352`、`max_num_sequences=4`，超出者需另配并验证，不属于本轮小规模验收。

停止 HTTP/PD 服务后，单独采 TP2 profile：

```bash
CUDA_VISIBLE_DEVICES=0,1 python tools/profile_engine.py \
  --config "$ACCEPTANCE_ROOT/configs/tp2-bf16.json" \
  --prompts "$ACCEPTANCE_ROOT/workloads/correctness.json" \
  --case tokens-4096 --max-tokens 8 --output-dir "$REPRO_ROOT/profile-tp2"
python tools/summarize_profile.py "$REPRO_ROOT/profile-tp2/rank-0.json" \
  "$REPRO_ROOT/profile-tp2/rank-1.json" --output "$REPRO_ROOT/profile-summary.json"
```

采集器沿用真实 ModelRunner 的计算、命令和退出协议，只增加 profiler 生命周期与 prefill/decode 标注。每个 rank 分别预热后采集，保留原始 Chrome trace；没有将 rank0 的时间当成其他 rank 的时间。

## 贡献、验证与未完成边界

上游已提供模型骨架、TP 层、NCCL、基础 Flash/Paged Attention、调度和 CUDA Graph。项目改造包括严格分片加载及覆盖检查、执行和资源回收适配、数值修复、PD 协议和 KV 状态恢复、FP8 KV 与验证工具。本轮增加 Row 部分积精度和 HTTP TP2 范围修复，做真实 GPU 验证、数值重放、测量与材料整理，新增 profile 工具；不把上游或未修改的历史实现描述成新写的框架。

关键代码：`utils/loader.py`、`layers/linear.py`、`layers/embedding_head.py`、`engine/{llm_engine,model_runner}.py`、`pd/{protocol,kv,tp,core,worker,client,gateway}.py`、`layers/attention.py`。bulk 时序为 D reserve → P export → D restore → P acknowledge → D activate，资源回收覆盖正常关闭和本次的取消场景。

最终固定实现后的完整 CPU 回归：**995 passed、74 skipped、4 warnings，697.09 秒**，日志随结果保存为 `validation/cpu-regression.log.gz`。命令屏蔽 GPU，使用 `uv run --no-sync python -m pytest -q -rs -p no:cacheprovider`，并设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`PYTHONPATH="$PWD/src:$PWD/tests:$PWD"`。skip 包括 CUDA、显式 opt-in 和未提供的原始资产；不视作 GPU 通过。此前未提交候选时，旧实验资产测试因对照 HEAD 与修改后的生产源码不一致而失败；固定实现提交后本轮完整回归通过，没有删除该校验。

针对 CPU server/layer 的 58 项检查通过；原 GPU FP32/BF16 TP 两项 **2 passed**，tiny PD EOS **1 passed**。14B 数值、PD 生命周期、HTTP 小批次和两 rank profile 是另行实际 GPU 执行，不混入上述 CPU 计数。新 profile/统计工具另对真实 trace 和请求复算，不新增与实现细节重复的测试。

未完成范围：四卡 TP2＋PD 的 `chat-1` 分歧未解决；14B 通用 logits 容差与更广质量评测未建立；FP8 `query`、FP8＋TP2、FP8＋PD 的目标模型组合未补验证；TP>1 Decode 与 TP2 pipeline 尚未支持。没有最大负载扫描、成熟框架性能对比、稳定尾延迟、生产故障全矩阵或新的 kernel 优化结论。
