# M4：张量并行加载与验证

M4 修复上游并行层和 checkpoint 加载，不将上游已有的张量并行层、NCCL 进程组和模型实现归为新增。CPU 两进程 Gloo 与 tiny Transformers checkpoint 的验证不能替代真实 CUDA/NCCL、Qwen3-14B 或服务器验收；通信时间、峰值显存与 14B 的 logits 门槛仍需服务器实测。

## 加载与分片契约

- checkpoint 的 Q/K/V 和 gate/up 分离张量分别调用目标并行层的 `weight_loader`；weight 和 bias 均适用。普通参数也调用自定义 loader，没有自定义 loader 时才按完整 shape 直接复制。
- Q 和 KV 头数均需为正且分别能被 TP 大小整除，Q 头数还必须能被 KV 头数整除；不支持 KV 头复制。Merged Linear 的每一部分、Column Linear 的输出维、Row Linear 的输入维都必须分别能被 TP 大小整除。构造时拒绝不符合条件的模型。
- embedding/head 按词表切分。词表大小不能整除 TP 时，尾部仅内部补零，rank 0 的 logits 截回原词表大小；实际 checkpoint shape 必须精确匹配原词表。Row Linear 的 bias 在 all-reduce 后添加一次。
- tied embedding/head 保持同一个 Parameter。checkpoint 可只包含共享权重的一份；若同时包含两份，内容必须一致。untied head 必须有自己的权重。模型设备转换后仍保持共享关系与参数 loader。
- 返回的 `loaded_params` 包含完成加载的模型参数名、共享参数别名和实际消费的 checkpoint 张量名，可供现有 `tools/compare_hf.py` 核对。缺失的模型参数或 packed 分片、重复 checkpoint 名、未知张量、shape 不符及不一致的共享权重均直接报错；不会截取部分权重继续运行。
- Qwen3 在启用 attention bias 时仍执行 Q/K norm；Qwen3/Llama 的 attention 输出投影跟随 attention bias。Llama attention 使用每个 rank 的局部 Q/KV 头数。

现有 runner 是单机进程组：它把 rank 作为 CUDA **可见设备的逻辑序号**，`CUDA_VISIBLE_DEVICES` 的顺序决定这些逻辑序号所对应的物理 GPU。例如在合适的 `CUDA_DEVICE_ORDER` 下使用 `CUDA_VISIBLE_DEVICES=4,3`，逻辑设备 0、1 对应所列的第一、第二张物理卡。启动前应使用 `nvidia-smi` 记录物理 GPU 的 UUID、PCI bus ID 和实际顺序；这不是多节点 local-rank 映射，也不新增设备列表配置接口。所有 rank 均加载 checkpoint，head 只在 rank 0 聚合完整 logits。

## CPU 验证

```bash
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES='' \
  uv run --no-sync python -m pytest -q
```

`tests/test_tp_checkpoint.py` 使用 Transformers 4.57.3 生成多文件 safetensors：Qwen3 和 Llama、tied 和 untied、attention bias、Llama MLP bias 均有覆盖。真实模型按 TP=1、TP=2 加载，同输入对照 HF 和两种 TP 的 logits，连续生成三个 greedy token，并核对每个 rank 的模型参数和 checkpoint 张量覆盖。CPU 仅将无法在 CPU 运行的 FlashAttention kernel 换为 PyTorch SDPA；模型、各并行层、loader 和 Gloo 通信均为真实实现。另有直接的 Column、Row、embedding/head、分片约束和错误 checkpoint 测试，以及 `cpu → meta` 参数转换测试。

测试前固定的 tiny 模型 logits 阈值为 FP32 `rtol=2e-4, atol=2e-5`、BF16 `rtol=5e-2, atol=2e-2`；这些是单元测试门槛，不是 14B GPU 验收阈值。Gloo 通信的进程墙钟时间只用于 CPU 诊断，不能作为 CUDA 通信时间或性能结论。

## 目标服务器复现模板

以下命令在服务器项目根目录执行，**尚未运行**。先固定经审核的候选 commit，并确认指定的两张 GPU 获准使用且空闲。所有新增文件、缓存和日志位于 `/Data2/zincion/ai-infra` 下；`M4_DATA` 每轮使用新目录。示例依赖 M2 已保存的配置及捕获文件；如果路径不同，应先定位同一模型 revision、精度和输入，不重新生成假数据。

```bash
export M4_WORKSPACE=/Data2/zincion/ai-infra
export M4_DATA="$M4_WORKSPACE/experiments/m4/acceptance-01"
export M4_M2_DATA="$M4_WORKSPACE/experiments/m2/20261114"
export M4_PYTHON=/Data2/zincion/envs/m2/bin/python
export TMPDIR="$M4_DATA/tmp"
export XDG_CACHE_HOME="$M4_DATA/cache"
export HF_HOME="$M4_DATA/cache/huggingface"
export HF_HUB_CACHE="$M4_WORKSPACE/huggingface"
export PYTHONPYCACHEPREFIX="$M4_DATA/cache/pycache"
export TORCHINDUCTOR_CACHE_DIR="$M4_DATA/cache/torchinductor"
export TRITON_CACHE_DIR="$M4_DATA/cache/triton"
export CUDA_CACHE_PATH="$M4_DATA/cache/cuda"
export PYTHONPATH="$PWD/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$TMPDIR" "$M4_DATA"
git rev-parse HEAD > "$M4_DATA/code-commit.txt"
git status --short > "$M4_DATA/worktree-status.txt"
git diff > "$M4_DATA/worktree.diff"
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used --format=csv > "$M4_DATA/gpus.csv"
nvidia-smi topo -m > "$M4_DATA/topology.txt"
CUDA_VISIBLE_DEVICES='' "$M4_PYTHON" -c \
  'import sys,torch,transformers; print(sys.version,torch.__version__,transformers.__version__)' \
  > "$M4_DATA/versions.txt"
```

使用获准的物理设备替换示例中的 `4,3`。运行时记录可见设备属性，并检查它与上面的 UUID/PCI 记录对应。以下双 GPU 测试使用真正的 CUDA kernel 和 NCCL；每个 rank 的 `TP measurement` 记录 all-reduce/gather 的 CUDA event 流耗时、调用次数和 PyTorch `peak_allocated_bytes`，原始结果也保存在 pytest 临时目录。event 区间包含等待其他 rank 等流依赖，不等于纯链路传输时间；PyTorch allocated 不包含所有驱动、NCCL 或缓存保留显存。实际完整 GPU 峰值还需结合 profiler/设备数据核对。

```bash
CUDA_VISIBLE_DEVICES=4,3 "$M4_PYTHON" -c \
  'import torch; [print(f"cuda:{i} uuid={p.uuid} pci={p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}") for i in range(torch.cuda.device_count()) for p in [torch.cuda.get_device_properties(i)]]' \
  > "$M4_DATA/visible-devices.txt"
CUDA_VISIBLE_DEVICES=4,3 "$M4_PYTHON" -m pytest -q -s -ra \
  --basetemp "$M4_DATA/pytest-gpu" tests/test_tp_gpu.py \
  > "$M4_DATA/tiny-gpu.log" 2>&1
```

复用 M2 捕获的 token 列表，生成只改变 `world_size` 的两份配置。TP=1 和 TP=2 顺序运行，避免占用同一设备的两个模型实例；用已有比较工具保留逐 token greedy 和同前缀 logits 数据。

```bash
"$M4_PYTHON" - <<'PY'
import json
import os
from pathlib import Path
import torch

old, out = Path(os.environ["M4_M2_DATA"]), Path(os.environ["M4_DATA"])
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
(out / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2) + "\n")
for world_size in (1, 2):
    config = dict(base)
    config.pop("timing_path", None)
    config["cache_dir"] = os.environ["HF_HUB_CACHE"]
    config["world_size"] = world_size
    config["enforce_eager"] = True
    (out / f"tp{world_size}.json").write_text(json.dumps(config, indent=2) + "\n")
PY

for m4_tp in 1 2; do
  CUDA_VISIBLE_DEVICES=4,3 "$M4_PYTHON" tools/compare_hf.py custom \
    --config "$M4_DATA/tp$m4_tp.json" --prompts "$M4_DATA/prompts.json" \
    --max-tokens 32 --output "$M4_DATA/tp$m4_tp.pt" \
    > "$M4_DATA/tp$m4_tp.log" 2>&1 || break
done
CUDA_VISIBLE_DEVICES='' "$M4_PYTHON" tools/compare_runs.py \
  --baseline "$M4_DATA/tp1.pt" --candidate "$M4_DATA/tp2.pt" \
  --output "$M4_DATA/tp1-vs-tp2.json"
```

`compare_runs.py` 比较 greedy、同前缀 logits 最大/平均误差和 top-1 一致率，但 `logits_gate=null` 表示 14B 门槛尚未确定；不能把报告生成当作最终验收。`compare_hf.py custom` 在 rank 0 额外核对完整加载覆盖；各 rank 的 loader 遇到未加载或未知张量均会报错。日志中的容量事件只来自 rank 0。

可在前述正确性运行之后另起一次 profile；profiling 有额外开销，不用它的耗时作为未插桩性能结果。先核对服务器 `nsys profile --help` 和 `nsys stats --help-reports` 是否支持以下选项与报告：

```bash
CUDA_VISIBLE_DEVICES=4,3 nsys profile --trace=cuda,nvtx --sample=none \
  --cuda-memory-usage=true --output="$M4_DATA/tp2-profile" \
  "$M4_PYTHON" tools/compare_hf.py custom \
  --config "$M4_DATA/tp2.json" --prompts "$M4_DATA/prompts.json" \
  --max-tokens 32 --output "$M4_DATA/tp2-profile.pt" \
  > "$M4_DATA/tp2-profile.log" 2>&1
nsys stats --report cuda_gpu_trace "$M4_DATA/tp2-profile.nsys-rep" \
  > "$M4_DATA/tp2-gpu-trace.txt" 2>&1
```

在 trace 中按进程、GPU、NCCL kernel 和实际前向区间提取通信 GPU 时间，并从内存轨迹记录对应进程/设备的显存峰值及统计口径；确认两个 worker 都被采集。通信 kernel 时间、主机等待和端到端延迟是不同指标，不能直接相加或互相替代。服务器没有完成上述测量、固定 14B 数值门槛和对应对照之前，M4 仍未完成全部验收。
