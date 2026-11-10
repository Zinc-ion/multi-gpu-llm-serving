# M0/M1 基线与复现说明

本文档记录 `dev` 分支在 M0/M1 阶段的可复现入口、状态不变量和审查边界。

## 固定基线

- 分支：`dev`
- 上游基线 commit：`00db3b477f33be1de91400483ae7d210c4d61868`
- Python：`3.11.16`（项目约束为 `>=3.11,<3.12`）
- 当前验证环境：PyTorch `2.9.1+cu128`、Transformers `4.57.3`
- 当前工作容器可见 1 张 CUDA GPU；五张 RTX 4090 的远端拓扑以 `server_hardware_2026-11-09.md` 为准
- 许可证：仓库根目录 `LICENSE` 保留 Apache-2.0
- 模型和实验数据目录：远端服务器使用 `/Data2/zincion`

本次工作开始前，工作区已经有未提交的 BlockManager、scheduler 测试和 `docs/`/BlockManager 测试改动。这些内容按项目本地工作处理并在当前实现中保留、扩充，没有提交或覆盖成上游代码。除这些本地改动外，项目基线来自上面的上游 commit；本次新增配置、生命周期修复、测试和文档均属于该基线之上的项目改造。

模型入口配置位于：

- `configs/qwen3_0_6b.json`
- `configs/llama3_2_1b.json`

每个配置固定模型和 tokenizer 的 Hub commit SHA、dtype、随机种子、采样参数、测试 prompt、并行规模、block 大小和 KV cache 容量上限。运行时会先按显存 profiling 计算可用 block，再取配置上限；实际容量可能小于配置值。实际运行时可通过 `--config` 选择配置；配置加载器会拒绝未固定的 revision。

固定 seed 可复现同一软件/硬件环境中的随机数流，不代表不同 GPU、驱动或 kernel 版本之间逐位一致。

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
4. 分配前先检查所需 block 数量，正常单线程路径不会留下半成品。

Scheduler 的请求状态必须满足：

- waiting、running 和当前 scheduled 集合不丢请求；KV 不足时优先抢占 running 队尾，保留队头请求继续推进。
- EOS、达到 token/长度上限、取消、抢占和不可满足的 KV 容量都会释放 block。
- `max_model_length` 在 Scheduler 初始化时必须同时不超过 batch token budget 和 KV 容量；没有单独上限的请求继承该值，避免抢占后超出 batch budget 而无进展。
- `cancel(seq_id)` 可取消 waiting/running 请求。
- 当前引擎是同步 step loop；取消在 scheduler step 之间生效，不会中断正在运行的 GPU kernel。

## 当前范围

M0/M1 只建立可复现基线和调度/KV 生命周期正确性。当前 prefill kernel 仍不消费 prefix-cache block table，因此活动请求之间不会共享 prefix block；哈希表只用于生命周期记录，并在释放时清理。Qwen3.8-27B、TP=2/4、2P+2D、FP8 和端到端优化属于后续 M2-M6，不应在当前阶段写成已完成能力。

当前配置默认 `enforce_eager=true`。CUDA Graph 路径仍有上游遗留问题：capture 阶段的输出缓冲区按 vocab size 分配，但模型前向返回 hidden states，尚未在 M0/M1 修复；M6 profiling/优化前需要单独处理。

## 人工学习/审核清单

- 阅读 `src/myvllm/config.py`、两个 JSON 配置和两个入口，确认配置字段确实覆盖运行所需参数。
- 阅读 `src/myvllm/engine/block_manager.py`，重点检查 append 跨 block、hash 删除和 ref count。
- 阅读 `src/myvllm/engine/scheduler.py`，重点检查 prefill 优先级、限额下的队列守恒、队尾抢占、取消和最大长度终态。
- 阅读 `tests/test_block_manager.py` 与 `tests/test_scheduler.py`，确认测试使用真实 BlockManager 验证 free/used/ref_count 守恒。
- 确认 GPU demo、模型下载和多卡行为尚未完成端到端验收；当前工作容器仅可见一张 GPU。
