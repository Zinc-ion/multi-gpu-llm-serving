# M2 运行接口与正确性约束

M2 在上游 MinivLLM 的 eager 推理路径上增加 BF16 数值修复、greedy、多 EOS、Qwen3-14B 分派、HTTP 服务和测量工具。原有 attention kernel、调度策略、权重映射及 KV 分配算法继续来自上游及 M0/M1；本阶段不实现 paged/chunked prefill、CUDA Graph、TP 或 PD 分离。

CPU 协议与数值回归可以独立运行。14B、GPU 阶段计时、PCIe、B0 和跨框架计时的验收需要另行执行硬件实验；这些能力不能仅凭代码或 CPU 测试写成已实测通过。

## 数值路径

- RoPE 的 inverse frequencies 和 cos/sin cache 使用 FP32。旋转前将 cos/sin 转为 q/k 的 dtype，避免 BF16 attention 输出被提升成 FP32。
- RMSNorm 使用 FP32 计算方差及归一化，在乘权重前恢复输入 dtype；Qwen3 的五处 norm 都使用配置里的 `rms_norm_epsilon`。
- `temperature == 0` 使用 argmax；正温度继续使用原有 Gumbel 采样，同一批可以混合。负温度在 `SamplingParams` 构造时拒绝。
- `eos` 配置统一为列表，Scheduler 保存集合并做 O(1) membership 判断。两个 Qwen 配置使用固定 generation config 中的 `[151645, 151643]`；Llama 仅适配为 `[128009]`，保留原来的停止行为。

HF 算术参照固定为 Transformers 4.57.3：

| 行为 | 参照代码 |
|---|---|
| FP32 inverse frequencies | `transformers/modeling_rope_utils.py::_compute_default_rope_parameters` |
| FP32 cos/sin 与恢复 dtype | `transformers/models/qwen3/modeling_qwen3.py::Qwen3RotaryEmbedding.forward` |
| RMSNorm 精度与乘法顺序 | 同文件 `Qwen3RMSNorm.forward` |
| 长输入只返回指定位置 logits | 同文件 `Qwen3ForCausalLM.forward(logits_to_keep=...)` |

CPU 单元测试使用 `torch.compiler.set_stance("force_eager")` 隔离算术与编译器融合；另有需要 CUDA 的编译 RoPE 回归。算术单元测试通过不代表整模型 GPU logits 已对齐。

```bash
# 显式禁止使用 GPU。
CUDA_VISIBLE_DEVICES="" uv run python -m pytest -q
```

2026-11-13 的 CPU 验证记录：代码 `7380450`，PyTorch `2.9.1+cu128`、Transformers `4.57.3`，71 项通过、1 项 CUDA 测试跳过。原始输出保存于 `/home/zincion/ai_infra/experiments/m2-local-20261113/final-cpu-tests.log`，命令如上。覆盖真实 HTTP、tokenizer、Scheduler、BlockManager、OpenAI SDK 和锁定 vLLM 请求函数；HTTP 测试仅替换模型执行。这不是 GPU 计时或端到端模型质量的验收。

## 14B 配置与容量候选

`configs/qwen3_14b.json` 固定模型和 tokenizer revision：

```text
Qwen/Qwen3-14B
40c069824f4251a91eefaf281ebe4c544efd3e18
```

结构字段来自这个 revision 的 `config.json`，EOS 来自 `generation_config.json`。checkpoint index 记录 `29,536,614,400 B`、8 个分片、443 个张量，包含独立的 `lm_head.weight`；这些是文件元数据，不能代替实际加载验收。

资源字段是**尚未经过 14B GPU 实测的启动候选**：

| 字段 | 候选值 | 设计依据 |
|---|---:|---|
| `max_model_length` | 17408 | 覆盖 16384 输入加最多 1024 输出 |
| `max_num_batched_tokens` | 17408 | 当前 Scheduler 要求总长度上限不超过一次 prefill 预算 |
| `max_num_sequences` | 4 | 初始批次上限；不是实测稳定并发 |
| `max_cached_blocks` | 272 | `4 × 17408 / 256`，仅为配置上限 |
| `gpu_memory_utilization` | 0.9 | 沿用原运行参数，待日志及实验确定 |

运行时仍按原 profiling 语义取 `min(profiled_blocks, configured_blocks)`，实际块数可能小于 272。未实测前，不能据此宣称可同时容纳四个长请求。若启动时不足以容纳单个 `max_model_length`，应保存失败日志并调整候选配置后重测。

BF16 KV 每 token 的结构字节数为 `40 × 2 × 8 × 128 × 2 = 163840 B`。每层仍各有一个 FP32 RoPE cache，40960 个位置的结构大小合计为 800 MiB；没有为节省显存改变其结构。实际配置验收以启动日志里的 parameter/cache 字节数、warmup 峰值和实际 block 数为准。

`gpu_memory_utilization` 在本项目中作用于加载权重及 warmup 后的**剩余空闲显存**，还会扣除 warmup 峰值增量；它与 vLLM 对总显存的比例定义不同。跨框架必须比较实际 KV token 容量。

`tools/capacity_report.py` 对每个请求先按 block 大小向上取整，再计算 KV 并发上界。这个上界不代表满足延迟或 goodput 目标的稳定并发。

## HTTP 服务

```bash
uv run python -m myvllm.server \
  --config configs/qwen3_14b.json \
  --served-model-name Qwen3-14B \
  --host 127.0.0.1 --port 8000
```

只支持 `world_size == 1`。只有一个推理线程创建和调用 ModelRunner、Scheduler；HTTP 层通过线程安全队列提交新增、取消及关闭命令，通过事件循环的线程安全回调接收 token。空闲时阻塞等待命令。这样避免多个推理调用同时改写模块级 `utils/context.py`。

接口为 `POST /v1/completions` 和 `POST /v1/chat/completions`，支持 SSE 和普通 JSON 响应。普通 JSON chat 是为 EvalScope 1.5.1 的 OpenAI 适配器提供的：该版本 `GenerateConfig.stream` 默认为 `None`，适配器会调用 `client.chat.completions.create`；这属于源码及 SDK 契约核对，未执行 EvalScope 数据集评测。没有实现 `/v1/models`，bench 必须显式传 `--model`。

### 请求契约

- `model` 必须与服务名一致；completions 的 `prompt` 必须为字符串。
- chat 的 `messages` 支持 `system`、`user`、`assistant` 角色与字符串 content。使用 tokenizer 自带的 chat template、默认参数和 `add_generation_prompt=True`；Qwen3 默认允许 thinking。
- 支持 `temperature`、`max_tokens`、`ignore_eos`、`stream` 和 `stream_options.include_usage`。未显式传入时，temperature 为 1、max_tokens 为 64、ignore_eos 为 false。
- `repetition_penalty=1`、`n=1`、`best_of=1`、`top_p=1`、`top_k=-1`、零 frequency/presence penalty、`logprobs=null`、`stop=null`、`seed=null`、`echo=false` 只接受这些默认值。其余未知字段或非默认能力返回 400。
- 空 prompt、超长 prompt 等入队错误只拒绝该请求；其他请求继续执行。
- `x-request-id` 用于服务日志及开启计时后的 JSONL 对齐，缺失时由服务生成。

### token 与文本

每个生成 token 都产生且只产生一个带 choices 的 SSE 消息，包括没有可见文字的 token。最后一个 token 的同一条消息带 `finish_reason`，不额外发送带 choices 的结束消息。EOS 为 `stop`，输出或总长度上限为 `length`，从 Sequence 的现有状态推导。

增量反分词借鉴 Transformers 的 `TextStreamer`，缓冲尚未完成的词和 UTF-8 字符，在空格、换行或结束时输出稳定文本。中文或 emoji 可以暂时输出空 text，但 token 消息继续立即发送。拼接文本与使用 `skip_special_tokens=True, clean_up_tokenization_spaces=False` 整体 decode 的结果一致。SSE JSON 使用 ASCII 转义，以兼容锁定 bench 客户端逐网络块做 UTF-8 decode 的实现。

请求 `include_usage` 时，末尾再发送 choices 为空的 usage 消息，然后发送 `data: [DONE]`。usage 包括 prompt/completion/total token 数。流式和普通 JSON 连接断开都经命令队列取消，在两个 step 之间释放 block；不会中断已提交的 GPU kernel。

预期之外的推理线程异常会记录 traceback 并以非零状态结束进程，不吞错或自动重试。服务关闭会取消剩余请求、注销该引擎的 atexit handler 并释放资源。

## 阶段计时

配置增加可选的 `timing_path`，默认缺失，即关闭计时。先创建输出目录，再在仓库外的实验配置中设置 JSONL 路径；不修改 0.6B 的基线资源字段。

| 记录 | 字段与含义 |
|---|---|
| `admitted` | `queue_ms`：HTTP handler 收到请求到完成 tokenization 并进入 Scheduler |
| `step` | `is_prefill`、序列数、forward 前的 token 数、request id 列表 |
| CPU | `schedule_ms`、`prepare_ms`、`d2h_host_ms`、`postprocess_ms`、`step_ms`，使用 `perf_counter` |
| GPU | `h2d_ms`、`forward_ms`、`sampling_ms`、`d2h_ms`，使用 CUDA event |
| `chunk_sent` | 反分词、ASGI send 返回耗时，以及收到请求到首个 token chunk 写出的时长 |

H2D 包含输入/context 张量和 temperature 的提交区间。forward 是 GPU stream 上整个模型及 LM head 的时间跨度，**包含每层 prefill 主机同步造成的 GPU 空隙**，不是纯 kernel 忙碌时间。D2H event 围绕原来的阻塞 `.cpu()`；`d2h_host_ms` 还包含等待此前 GPU 工作和 `.tolist()` 的主机耗时。这些区间存在重叠，不能简单相加当成端到端延迟。

event 在既有 D2H 同步边界之后读取；若 D2H 结束 event 尚未就绪，该条记录延迟到下一步或既有退出同步后写出。只查询 event，不新增 step 同步。计时开启/关闭的端到端开销仍需同负载 GPU 实测，不能把未测的开销写成零。

## 已知限制

- 14B 加载完整性、整模型 HF 对齐、GPU 计时和 B0 尚需 GPU 验收；logits/token 阈值和 SLO 尚未确定。
- CUDA Graph 仍有 hidden-size/vocab-size 形状问题，保持 eager，留到 M3。
- prefill 不读取 block table，positions 每条序列从零开始；每层仍存在 `cu_seqlens.cpu()` 同步，留到 M3。跨请求 prefix 复用继续禁用。
- 每层 RoPE cache 和 `gpu_memory_utilization` 的原有结构/语义未调整。
- rendezvous 仍固定为 `tcp://localhost:12345`；不同 HTTP 端口不能让两个模型实例并存。多实例留到 M5。
- TP 权重加载及多卡一致性留到 M4；没有重构 loader。对比工具检查 loader 返回集合与模型参数及 checkpoint 张量的覆盖关系，拒绝未加载项。
- 原离线 `generate()` 的 prefill TPS 打印使用 postprocess 后的长度，包含刚采样的 token；M2 的 step JSONL 另用 forward 前长度，B0 使用客户端指标，不使用该打印作为测量结果。
- vLLM 0.15.0 没有 `timed_trace` 数据集；实际发送时间可以保存，精确时间戳回放仍待后续实现或版本核实。
