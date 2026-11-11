# M0/M1 基线与复现说明

本文档记录 M0/M1 阶段的可复现入口、状态不变量和审查边界。

## 固定基线

- 上游基线 commit：`00db3b477f33be1de91400483ae7d210c4d61868`
- Python：`3.11.16`（项目约束为 `>=3.11,<3.12`）
- 原 M0/M1 阶段记录的验证环境：PyTorch `2.9.1+cu128`、Transformers `4.57.3`
- 原 M0/M1 阶段记录的工作容器可见 1 张 CUDA GPU；五张 RTX 4090 的远端拓扑以 `server_hardware_2026-11-09.md` 为准
- 许可证：仓库根目录 `LICENSE` 保留 Apache-2.0
- 模型和实验数据目录：远端服务器使用 `/Data2/zincion`

M0/M1 改动以最小 diff 方式叠加在上游基线之上，保留上游注释、格式和函数结构；配置、KV 生命周期修复和新增测试属于项目改造。

模型入口配置位于：

- `configs/qwen3_0_6b.json`
- `configs/llama3_2_1b.json`

每个配置固定模型和 tokenizer 的 Hub commit SHA、dtype、随机种子、采样参数、测试 prompt、并行规模、block 大小和 KV cache 容量上限。运行时会先按显存 profiling 计算可用 block，再取配置上限；实际容量可能小于配置值。实际运行时可通过 `--config` 选择配置；配置加载器会拒绝未固定的 revision。

固定 seed 可复现同一软件/硬件环境中的随机数流，不代表不同 GPU、驱动或 kernel 版本之间逐位一致。

## 配置字段说明

以下说明迁移自上游入口配置字典的教学注释。JSON 不支持注释，因此保留在这里；不是本次新增的模型结构修正。

| 字段 | 原注释的修正或约束 | 当前值 |
| --- | --- | --- |
| Qwen `vocab_size` | `Fixed: was 151643, HF model uses 151936` | `151936` |
| Qwen `head_dim` | `Fixed: was 64, should be 128`；原括号中的 `hidden_size / num_heads` 不适用于此处，`1024 / 16` 为 64，不能据此推得 128；保留入口已有的显式维度 | `128` |
| Qwen `base` | `Fixed: was 10000, HF uses rope_theta=1000000` | `1000000` |
| Qwen `ffn_bias` | `Fixed: HF Qwen3 doesn't use MLP bias` | `false` |
| Qwen `eos` | `Fixed: should match tokenizer.eos_token_id` | `151645` |
| Qwen `max_position` / Llama `max_position_embeddings` | RoPE 可用位置上限应不小于请求总长度上限 `max_model_length` | `32768` |
| Llama `eos` | 应匹配 tokenizer；原注释另记 HF 的 EOS 列表为 `[128001, 128008, 128009]`。当前调度器仍只比较配置中的单个 EOS，没有新增多 EOS 支持 | `128009` |

`sampling.max_tokens` 是最多生成的 token 数；`max_model_length` 是 prompt 与 completion 的总长度上限，两者任一先到就停止。默认分别为 256 和 128，因此实际 completion 还受 prompt 长度约束。两个入口中原有的三行教学注释继续保留。

`max_num_batched_tokens` 同时用于调度与 warmup，不再保留拼错的 `max_num_batch_tokens` 或另设 `warmup_tokens`。`max_cached_blocks` 是 KV 容量上限：先取 profiling 与配置值的较小者，多卡再取所有 rank 的最小值；profiling 后仍需检查实际容量能容纳 `max_model_length`。

`model_type` 显式选择现有模型分支；`model_revision` 和 `tokenizer_revision` 固定下载版本；`cache_dir` 控制缓存位置。revision 的格式检查用于避免浮动版本，不计算文件 MD5 或新增文件校验。

原 Llama 配置里的 `# model params` 分组已由 JSON 字段承接。原 prompt 列表旁 `# * 15`、`#* 30` 是被注释掉的重复实验写法，随列表移入 JSON 而移除；当前只使用 JSON 中显式列出的三条 prompt，不做隐式重复。

## 最小复现

```bash
uv sync
uv run python -m pytest -q
uv run python main.py --config configs/qwen3_0_6b.json
```

运行 GPU demo 前，需要确认模型缓存目录 `/Data2/zincion/huggingface` 可写，或者在配置中改为当前机器的模型缓存目录。Llama 入口为：

```bash
uv run python main_llama32.py --config configs/llama3_2_1b.json
```

## M1 验收不变量

BlockManager 在任意稳定点必须满足：

1. `free_block_ids` 与 `used_block_ids` 不相交，二者并集覆盖全部 block。
2. 空闲 block 的 `ref_count == 0`、hash 为 `-1`。
3. 使用中的 block 的 `ref_count > 0`，释放到零时同时删除指向它的 hash 元数据。
4. Scheduler 分配前先调用 `can_allocate()` 检查所需 block 数量；`allocate()` 保留上游调用契约，不另加容量防御或回滚。

Scheduler 的请求状态必须满足：

- waiting、running 和当前 scheduled 集合不丢请求；KV 不足时优先抢占 running 队尾，保留队头请求继续推进。
- EOS、达到 token/长度上限、取消和抢占都会释放 block；不可满足的长度/容量在启动或请求入队前拒绝。
- `max_model_length` 在 Scheduler 初始化时必须同时不超过 batch token budget 和 KV 容量；没有单独上限的请求继承该值，避免抢占后超出 batch budget 而无进展。
- `cancel(seq_id)` 可取消 waiting/running 请求。
- 当前引擎是同步 step loop；取消在 scheduler step 之间生效，不会中断正在运行的 GPU kernel。

## 当前范围

M0/M1 只建立可复现基线和调度/KV 生命周期正确性。当前 prefill kernel 仍不消费 prefix-cache block table，因此活动请求之间不会共享 prefix block；哈希表只用于生命周期记录，并在释放时清理。Qwen3-14B、TP=2、2P+2D、FP8 KV cache 和端到端优化属于后续 M2-M9，不应在当前阶段写成已完成能力。

当前配置默认 `enforce_eager=true`。CUDA Graph 路径仍有上游遗留问题：capture 阶段 `outputs` 的形状为 `(max_bs, vocab_size)`，但 `self.model()` 返回宽度为 `hidden_size` 的 hidden states，赋值形状不匹配。因此本基线的 `enforce_eager=false` 路径不可用，留到 M3 修复；当前没有执行 GPU 验证。

## 人工学习/审核清单

- 阅读 `src/myvllm/config.py`、两个 JSON 配置和两个入口，确认配置字段确实覆盖运行所需参数。
- 阅读 `src/myvllm/engine/block_manager.py`，重点检查 append 跨 block、hash 删除和 ref count。
- 阅读 `src/myvllm/engine/scheduler.py`，重点检查 prefill 优先级、限额下的队列守恒、队尾抢占、取消和最大长度终态。
- 阅读 `tests/test_block_manager.py` 与 `tests/test_scheduler_lifecycle.py`，确认新增测试使用真实 BlockManager 验证 free/used/ref_count 守恒。原 `tests/test_scheduler.py` 保留，只适配构造参数。随机化测试使用 3 个 KV block、200 个种子，并断言确实发生抢占及每步请求队列归属守恒。
- 确认 GPU demo、模型下载和多卡行为尚未完成端到端验收。本次验证为 CPU 上的 28 项单元/生命周期测试；每批代码提交均独立通过测试。
