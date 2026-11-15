# M3：paged prefill、chunked prefill 与 CUDA Graph

本文说明 M3 代码的接口和后续服务器验收方法。GPU 正确性、显存容量、B1 性能及 vLLM 完整参照数据均待实测，不能由 CPU 测试推定通过。M2 尚未完成的 B0、阈值和 SLO 验收仍独立保留。

## 调度与缓存契约

`enable_chunked_prefill` 默认关闭，保留 B0 的 prefill 优先调度；开启后使用 B1 调度：

- 先为 running 中的 decode 请求各分配一个 token，再将本步剩余的 `max_num_batched_tokens` 用于已有或新到达的 prefill。
- 每个 chunk 可在 block 内开始或结束。`num_cached_tokens` 表示已完成前向的 token 数，`num_scheduled_tokens` 表示本步实际处理的 token 数；两者由调度与 postprocess 推进。
- 未完成前缀的 chunk 不计算 completion logits、不采样、不触发服务层 token 回调；只有处理到当前完整前缀末尾才产生一个 token。因此 SSE choices 数仍等于 completion token 数。
- resident 请求数不超过 `min(max_num_sequences, max_num_batched_tokens)`，保证已有 decode 不会因新请求入驻而超出每步 token 预算。
- 入驻时沿用原有 BlockManager，预留整个已知前缀的 block，后续 chunk 填充对应位置；新增生成 token 仍按原有 append 逻辑申请 block。尚未实现按 chunk 渐进申请前缀 block。
- 取消和抢占仍在 step 之间生效。抢占释放全部 block 并清零 cached 进度，重算时保留包括已生成 token 在内的完整前缀。
- 跨请求 prefix 复用继续关闭。hash 元数据仍仅用于生命周期记录。

此调度借鉴 Sarathi-Serve 的 decode 优先、用剩余预算混入 prefill 的方法，作为 B1 基线，不是本项目提出的新算法，也不声称复现其全部策略。原论文：[Sarathi-Serve, OSDI 2024](https://arxiv.org/abs/2403.02310)。

## 模型路径

新增的 paged prefill kernel 从 block table 读取完整 KV 前缀，查询只来自本步 chunk。每个序列的 `K 长度 − Q 长度` 是 cached offset；因果掩码按这个绝对位置构造，支持 `cu_seqlens_q != cu_seqlens_k`、GQA 和不连续的物理 block。

Qwen3 和 Llama 的 RoPE positions 由 runner 在 CPU 组 batch 时计算一次并搬运，所有层复用，取消原先逐层读取 `cu_seqlens_q.cpu()` 的同步。常规 serving prefill 读取 paged KV；只有 KV 分配之前的 warmup 使用已有 packed FlashAttention，传入已知最大 Q 长度，避免逐层再取 CPU 长度。混合 batch 中的 decode 是 Q 长度为 1 的 suffix query；纯 decode 仍走原有 decode kernel。

LM head 只选取本步已完成前缀的最后一个 hidden state。`LLMEngine.step()` 的输入 token 数在 postprocess 前统计，避免把刚生成的 token 算进 prefill 输入。可选 JSONL 阶段计时仍记录实际处理的 token 数；`is_prefill=true` 包括混合 batch，不能把它解释成“全部是新 prompt token”。

## 配置与 CPU 验证

`configs/qwen3_14b.json` 保留 B0 配置，`configs/qwen3_14b_chunked.json` 是待测 B1 起点：相同模型、精度和总长度，开启 chunking 并将每步 token budget 设为 1024。该值是实验起点，未依据性能数据调优。chunking 开启时允许 `max_model_length > max_num_batched_tokens`，仍要求总长度能放进实际 KV 容量。

warmup 的单序列长度取 `min(max_model_length, max_num_batched_tokens)`，避免 budget 小于总长度时产生空 batch。chunking 的 warmup 与运行时 paged kernel 路径不同，且 decode batch 峰值可能高于 warmup；必须重新记录实际容量和峰值显存，不能复用 M2 的 block 数作为保证。

CPU 验证命令：

```bash
CUDA_VISIBLE_DEVICES='' uv run python -m pytest -q
```

CPU 测试保留真实 Scheduler、BlockManager、ModelRunner 输入准备、Qwen3/Llama 模型计算、tokenizer 和 HTTP 路径，GPU 数据搬运和 attention kernel 用 CPU 实现替代。覆盖分块、混合、抢占重算、取消和协议计数。tiny 模型 FP32 logits 使用 PyTorch 默认浮点比较规则，BF16 检查 greedy token 与有限值；这些是单元测试约束，不是待用户确认的 14B logits 验收阈值。

GPU kernel 测试带 CUDA skip 条件，比较打乱物理 block 后的 suffix attention 与 PyTorch SDPA，覆盖不同 block 大小和 FP32/BF16。尚未运行的 GPU 测试不能记作通过。

## CUDA Graph decode 的代码修复

`enforce_eager=false` 时只捕获纯 decode，混合 batch 和 prefill 仍 eager 执行。capture output 使用模型 dtype 的 `hidden_size` 宽度，LM head 和 sampling 保留在 graph 外。capture batch 不超过序列上限、每步 token 预算和 profiling 后的实际 block 数，包含边界 batch；replay 选择最小可容纳实际 batch 的 graph。

input、positions、slot mapping、context lengths 和 block tables 都使用固定地址。每步复制前清除 padding 和旧表列：padding 的 slot 为 `-1`，context length 为 0；decode kernel 对零长度行返回零，不访问 KV，也不除以零。warmup 在单独 stream 上完成，capture 后保存 graph pool，启动期间同步，replay 不新增主机同步。

CPU buffer 测试用设备执行替身验证真实 capture/replay 准备逻辑，不能据此宣称 CUDA Graph 可用。另用独立进程 `TRITON_INTERPRET=1` 在 CPU 执行真实 Triton kernel，验证 FP32 paged prefill、sentinel slot 和零长度 decode。解释模式不检验 GPU 代码生成、stream 行为或性能。

`tests/test_cudagraph_gpu.py` 保留真实 GPU 模型、attention、capture/replay，检查 whole/chunked 与 eager/graph 四种组合的 greedy token，并覆盖 batch 缩小及 padding。该测试尚待服务器执行。Graph 本身还有额外内存需求，现有 KV profiling 发生在 capture 前；14B 启用 graph 后必须重新检查峰值和容量，不能沿用 M2 eager 容量数字。

## 服务器验收顺序

所有新文件、缓存、日志及临时配置都放在 `/Data2/zincion/ai-infra` 下；先确认获准使用的 GPU 空闲，不影响其他进程。

1. 在保存的 M2 commit 上补齐独立 B0 验收，再运行 M3，避免用变更后的 kernel 反填 B0。
2. 运行 GPU 回归测试，并在同一固定 prompt 集上用 `tools/compare_hf.py custom` 分别捕获 M2、M3 whole-prefill 和 M3 chunked 输出。比较 greedy token，按预先确认的阈值核对 logits，不能为了通过事后放宽。
3. 用同一权重 revision 和 tokenizer 跑 Transformers 参照，复用 M2 的同前缀比较。CPU tiny 模型测试不能替代长上下文、14B BF16 验证。
4. 重新保存启动日志中的容量、RoPE、warmup 与峰值显存记录，验证 1K/4K/16K 输入和固定输出。B1 的 chunk budget 扫描应与其他配置变化分开。
5. 启动 B1 服务，复用 `tools/run_b0.py`（它只调用同一锁定 bench 客户端，可测任意兼容服务）和 `tools/analyze_bench.py`；保存到独立 B1 目录。输入文件、sha256、seed、输出长度、并发/到达速率、SLO 和 GPU/NUMA 绑定保持一致。
6. 测同硬件 vLLM 参照，固定 vLLM 0.15.0、模型/tokenizer revision、BF16、greedy、ignore-eos 和请求列表。对比实际 KV token 容量；不按两边不同含义的 memory-utilization 参数宣称容量相等。vLLM 的 ITL 可能是合并 chunk 的间隔，继续单独核对。

## 服务器命令模板

以下命令尚未在服务器执行。先用 M2 已保存的 0.6B FP32/BF16 配置和捕获文件执行同样流程，再运行 14B。例子使用已有的只读 Python 环境；在服务器仓库根目录执行，所有输出写入工作目录内的新目录。`M3_DATA` 每轮应另设值，防止覆盖上一轮结果。

```bash
export M3_WORKSPACE=/Data2/zincion/ai-infra
export M3_DATA="$M3_WORKSPACE/experiments/m3/acceptance-01"
export M3_M2_DATA="$M3_WORKSPACE/experiments/m2/20261114"
export M3_PYTHON=/Data2/zincion/envs/m2/bin/python
export TMPDIR="$M3_DATA/tmp"
export XDG_CACHE_HOME="$M3_DATA/cache"
export HF_HOME="$M3_DATA/cache/huggingface"
export HF_HUB_CACHE="$M3_WORKSPACE/huggingface"
export HF_DATASETS_CACHE="$M3_DATA/cache/datasets"
export PYTHONPYCACHEPREFIX="$M3_DATA/cache/pycache"
export TORCHINDUCTOR_CACHE_DIR="$M3_DATA/cache/torchinductor"
export TRITON_CACHE_DIR="$M3_DATA/cache/triton"
export CUDA_CACHE_PATH="$M3_DATA/cache/cuda"
export VLLM_CACHE_ROOT="$M3_DATA/cache/vllm"
export PYTHONPATH="$PWD/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$TMPDIR" "$M3_DATA"
git rev-parse HEAD > "$M3_DATA/code-commit.txt"
git diff > "$M3_DATA/worktree.diff"
nvidia-smi > "$M3_DATA/gpus.txt"
nvidia-smi topo -m > "$M3_DATA/topology.txt"
CUDA_VISIBLE_DEVICES='' "$M3_PYTHON" -c \
  'import sys,torch,transformers,vllm; print(sys.version,torch.__version__,transformers.__version__,vllm.__version__)' \
  > "$M3_DATA/versions.txt"

# 确认 GPU0 获准且空闲后再执行。不要停止其他人的进程。
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 \
  "$M3_PYTHON" -m pytest -q -ra \
  > "$M3_DATA/pytest-gpu.log" 2>&1
```

从已有 M2 14B 捕获文件恢复完全相同的 prompt token 列表，不重新采样或下载数据。下面的配置继承 M2 已记录的本地 snapshot 路径，只在实验目录更改执行模式；清除旧 timing 路径，避免将新数据写入 M2 文件。

```bash
"$M3_PYTHON" - <<'PY'
import json
import os
from pathlib import Path
import torch

old, out = Path(os.environ["M3_M2_DATA"]), Path(os.environ["M3_DATA"])
base = json.loads((old / "config-14b.json").read_text())
captured = torch.load(old / "custom-14b.pt", map_location="cpu", weights_only=False)
prompts = {
    "model_revision": base["model_revision"],
    "tokenizer_revision": base["tokenizer_revision"],
    "cases": [
        {k: v for k, v in case.items() if k not in ("logits", "token_ids", "num_prompt_tokens")}
        for case in captured["cases"]
    ],
}
(out / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2))
for chunked in (False, True):
    for eager in (True, False):
        config = dict(base)
        config.pop("timing_path", None)
        config["cache_dir"] = os.environ["HF_HUB_CACHE"]
        config["enable_chunked_prefill"] = chunked
        config["enforce_eager"] = eager
        config["max_num_batched_tokens"] = 1024 if chunked else base["max_num_batched_tokens"]
        name = ("chunked" if chunked else "whole") + ("-eager" if eager else "-graph")
        (out / f"{name}.json").write_text(json.dumps(config, indent=2))
PY

for m3_mode in whole-eager whole-graph chunked-eager chunked-graph; do
  CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 \
    "$M3_PYTHON" tools/compare_hf.py custom \
    --config "$M3_DATA/$m3_mode.json" --prompts "$M3_DATA/prompts.json" \
    --max-tokens 32 --output "$M3_DATA/$m3_mode.pt" \
    > "$M3_DATA/$m3_mode.log" 2>&1 || break
done

CUDA_VISIBLE_DEVICES='' "$M3_PYTHON" tools/compare_runs.py \
  --baseline "$M3_M2_DATA/custom-14b.pt" --candidate "$M3_DATA/whole-eager.pt" \
  --output "$M3_DATA/m2-vs-m3.json"
CUDA_VISIBLE_DEVICES='' "$M3_PYTHON" tools/compare_runs.py \
  --baseline "$M3_DATA/whole-eager.pt" --candidate "$M3_DATA/chunked-eager.pt" \
  --output "$M3_DATA/whole-vs-chunked.json"
CUDA_VISIBLE_DEVICES='' "$M3_PYTHON" tools/compare_runs.py \
  --baseline "$M3_DATA/chunked-eager.pt" --candidate "$M3_DATA/chunked-graph.pt" \
  --output "$M3_DATA/eager-vs-graph.json"
```

每次失败先保存日志并排查，不自动更换 dtype、attention backend 或 eager 模式。`compare_runs.py` 对模型/tokenizer revision、dtype、EOS 与输入 token 做匹配校验；逐例报告 token 首次分歧和 max/mean absolute logits error、top1 一致率。独立生成一旦分歧，只有预测首次分歧及之前的 logits 仍具有同前缀，脚本只统计这些位置；完整 teacher forcing 继续用 `compare_hf.py reference`。`logits_gate=null` 明确表示尚未批准的数值门槛，不输出自动“验收通过”。greedy 不一致时仍保存 JSON，并以非零状态退出。

通过前置正确性检查并重新测算容量后，在前台运行 B1 服务：

```bash
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 \
  "$M3_PYTHON" -m myvllm.server \
  --config "$M3_DATA/chunked-graph.json" --served-model-name Qwen3-14B --port 8000 \
  > "$M3_DATA/b1-server.log" 2>&1
```

在另一终端设置相同环境，用 M2 保存的请求文件运行 `tools/run_b0.py`，将 `--output-dir` 指向 `$M3_DATA/b1-l1` 或 `$M3_DATA/b1-l2`；完整 L1/L2 参数和 seed 规则见 [M2 测量复现](m2_experiments_zh.md)。同一次比较中固定 KV token 预算或如实记录容量差异，并保存服务日志和所有配置。不要为了填满 GPU 在同一卡并行跑四种模式。

完整 vLLM 参照使用同一个本地 snapshot 与请求文件。关闭自己前台运行的本项目实例后，才启动该服务；以下序列上限 1024 对应本项目 chunk budget 下的 resident 上限，显式关闭 prefix caching，避免重复的 random prompt 获得跨请求命中：

```bash
export M3_SNAPSHOT=$("$M3_PYTHON" -c \
  'import json,os; print(json.load(open(os.environ["M3_DATA"]+"/chunked-eager.json"))["model_name_or_path"])')
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 \
  "$M3_PYTHON" -m vllm.entrypoints.cli.main serve "$M3_SNAPSHOT" \
  --served-model-name Qwen3-14B --dtype bfloat16 --max-model-len 17408 \
  --max-num-batched-tokens 1024 --max-num-seqs 1024 \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --gpu-memory-utilization 0.9 --port 8001 \
  > "$M3_DATA/vllm-server.log" 2>&1
```

这些参数按锁定的 vLLM 0.15.0 `engine/arg_utils.py` 和 `config/scheduler.py` 核对，仍需服务器启动验收。使用相同 `tools/run_b0.py`、相同列表/seed/输出预算，显式传 `--base-url http://127.0.0.1:8001` 并另存 `$M3_DATA/vllm-l1`、`vllm-l2`。本项目的序列上限字段可高于 chunk budget，B1 的实际 resident 上限取两者较小；vLLM 配置直接要求 budget 不小于序列上限。两边 memory-utilization 的语义不同，0.9 相同不代表 KV 容量相同。`analyze_bench.py` 的逐 token ITL 等式只作为本项目协议检查，vLLM 合并 chunk 导致的不等式应单独记录，不能据此丢弃其 TPOT/usage 数据。
