# M4：张量并行加载与验证

M4 修复上游并行层和 checkpoint 加载，不将上游已有的张量并行层、NCCL 进程组和模型实现归为新增。CPU 两进程 Gloo 与 tiny Transformers checkpoint 的验证不能替代真实 CUDA/NCCL、Qwen3-14B 或服务器验收；本页区分历史 CPU/tiny 结果与当前 L40S 结果，14B 通用 logits 门槛仍未建立。

2026-11-18 的当前结果：原 tiny BF16 greedy 失败已修复，原 GPU FP32/BF16 两项通过；14B 六组各 16 token 与 TP1 一致，同前缀 top1 为 96/96。下文的 2026-11-16 失败与诊断流程保留为历史证据，不能再视为修复后的当前失败；14B 通用 logits 门槛仍未建立。

## 加载与分片契约

- checkpoint 的 Q/K/V 和 gate/up 分离张量分别调用目标并行层的 `weight_loader`；weight 和 bias 均适用。普通参数也调用自定义 loader，没有自定义 loader 时才按完整 shape 直接复制。
- Q 和 KV 头数均需为正且分别能被 TP 大小整除，Q 头数还必须能被 KV 头数整除；不支持 KV 头复制。Merged Linear 的每一部分、Column Linear 的输出维、Row Linear 的输入维都必须分别能被 TP 大小整除。构造时拒绝不符合条件的模型。
- embedding/head 按词表切分。词表大小不能整除 TP 时，尾部仅内部补零，rank 0 的 logits 截回原词表大小；实际 checkpoint shape 必须精确匹配原词表。Row Linear 的 bias 在 all-reduce 后添加一次。
- 当前 CUDA BF16 TP>1 的 Row Linear 使用 BF16 输入/权重的 `torch.mm(..., out_dtype=torch.float32)` 保留部分积，经 FP32 all-reduce 后一次转回 BF16，再加 bias。CPU、TP1、FP32 路径保持原语义；此 API 在实际 PyTorch 2.9.1+cu128 验证，不增加低版本静默回退。归约 payload 比原 BF16 加倍，不能称通信加速。
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

上述 checkpoint 算术测试每步重算完整前缀，不代表 worker decode 已验证。`tests/test_tp_decode.py` 另由真实 Scheduler 分配、prefill 后 postprocess 首 token，再通过 `ModelRunner.write_shm/read_shm` 的实际 pickle 协议驱动 worker 的 `prepare_prefill/prepare_decode`，覆盖跨 block、取消和释放。Sequence 的 worker payload 现在携带 `block_size`，供 decode 计算最后一块的槽位；不增加 worker 不使用的调度字段。此 CPU 测试只替换 CUDA 数据搬运，不运行 GPU attention。

集成基线 `5a25360e173389de5d982e8440420441ba8c85fb` 已导入 M5 的实例 TCPStore、共享内存和启动失败清理，M4 不重复实现这些资源逻辑。`tests/test_tp_loader_cleanup.py` 从真实 tiny checkpoint 删除必需的 norm 权重，确认严格 loader 报错后进程组和 TCPStore 释放，并在同一进程成功重启；分别覆盖 engine 和 standalone runner。该测试使用真实模型、loader 和单进程 Gloo，替换 CUDA 搬运及 warmup/cache allocation，不等同于多 GPU 启动失败验证。

测试前固定的 tiny 模型 logits 阈值为 FP32 `rtol=2e-4, atol=2e-5`、BF16 `rtol=5e-2, atol=2e-2`；这些是单元测试门槛，不是 14B GPU 验收阈值。Gloo 通信的进程墙钟时间只用于 CPU 诊断，不能作为 CUDA 通信时间或性能结论。

GPU 入口使用 `tests/tp_engine_helpers.py` 创建真实 tokenizer/checkpoint，在独立子进程中构造 `LLMEngine`，由 Scheduler、`ModelRunner.run` 和 TP worker loop 执行首次 paged prefill、两步 decode 及跨 block。LM head hook 只在初始化完成后采集同前缀 logits；初始化仍执行原 warmup，其 cache-free FlashAttention 不进入数值对照。HF 参照使用 native `from_pretrained(dtype=...)`，保留 FP32 RoPE `inv_freq`，不对已初始化模型整体调用 `.to(bfloat16)` 量化该 buffer。seed、prompt 和上述阈值保持不变；每个 rank 的加载覆盖、实际 attention 路径、通信 event 和峰值 allocated 都保存在测试产物中。

`tests/test_tp_engine_gpu.py` 的 TP=1 入口于 2026-11-15 在本机 RTX 5080 Laptop GPU 实测通过：FP32/HF 最大绝对误差 `1.4901161e-7`，BF16/HF 为 `0.0029296875`，各自三个 greedy token 完全一致。`tests/test_tp_gpu.py` 复用相同 helper 对照 TP=1/TP=2/HF，但本机只有一张 GPU，两种 dtype 均明确 skip。单卡没有 TP collective，不能将其零次通信记录当作双卡 NCCL 测量；真实 TP=2、目标服务器和 14B 仍未验收。

2026-11-16 的目标服务器 tiny 复现中，原 M4 `c7bf6d331b42b1a7e078f6f728126218dc774d51` 的 BF16 TP2 第三个 greedy token 为 62，HF/TP1 为 41；三个同前缀 logits 事后复算在原 tiny 阈值内，严格 token 验收仍失败。`tools/diagnose_tp.py --help` 提供独立诊断入口：显式指定原源码、原 checkpoint、两份配置、已有 HF/TP 捕获及资产清单，在新目录逐 rank 保存加载权重、模块输入输出和真实 collective 前后张量；仅对相同前缀和匹配的分片语义做比较。它不修改原引擎，每步等待 worker 的完成标记；CPU 拷贝、等待和 hook 可能改变同步或编译边界，插桩结果须与原失败分开报告。控制器在启动 TP2 子进程前独占创建并持有通信共享内存，引擎附着并按原逻辑退出；进程组退出后，只回收本次创建且设备与 inode 仍匹配的段。清理期间记录中断并完成进程组及共享内存处理，所有期限须为有限正数。`--preflight-only` 仅核验输入和实际导入的源码，不启动 GPU 引擎；退出码 2 表示原始或插桩后的严格 greedy/原 logits 门槛失败，3 表示输入来源哈希未核验，4 表示分片权重不匹配，1 表示运行错误。

同日 GPU0/1 的原 M4 插桩结果为：TP1/TP2 的三个 token 分别仍为 `[32, 33, 41]` / `[32, 33, 62]`，三个 rank 的加载权重均与对应 checkpoint 分片一致。首个观察到的精确差异位于第 0 层 `self_attn.o_proj` 输出，113/320 个元素不同，最大绝对差为 `0.000244140625`；这不能确定根因。该轮使用的是共享内存强制清理修复前的脚本，作为历史结果保留。后续 `p3a-gpu-bf16-04` 使用固定候选 `fb51c70bb4d0061a5b5b901d70ac44a4f79d2501`对原 M4 BF16 用例复跑，TP1/TP2 lane 均正常退出（exit 0），输出仍分别为 `[32, 33, 41]` / `[32, 33, 62]`，无 TERM/KILL 或存活进程组，TP2 通信共享内存记录为 `already_released`。该次正常 GPU 运行未触发异常或强杀清理，相关异常路径由 CPU 验证；原 strict greedy 仍为 FAIL，继续阻塞 TP2 验收。

## L40S 原严格失败的重放与修复

新机器复现原输入、seed 和严格判定后仍得到 TP1 `[32,33,41]`、TP2 `[32,33,62]`；第三步 TP1 的 41/62 logits 均为 `0.34765625`。独立真实运行捕获确认首个 `layer0.o_proj` 的输入/权重分片精确一致，完整 BF16 GEMM 与分片 BF16 GEMM＋求和分别精确复现 TP1/TP2，全部 113/320 个差异及最大差 `0.000244140625` 相同。FP32 完整/分片结果转回 BF16 后一致；诊断干预也恢复原输出。

生产实现因此保留 FP32 部分积和归约，避免先舍入各个 shard。未修改测试 seed、输入、容差或 greedy 要求。原 `tests/test_tp_gpu.py` 两种 dtype 实际为 2 passed（71.52 秒），BF16 TP2 对 HF max abs `0.0029296875`；14B 原六组均 16 输出一致。但 14B 最大同前缀差仍达 `9.671875`，并非所有 logits 误差都更小，不建立事后容差或完整质量结论。

HTTP 原 `world_size==1` 检查在目标模型启动前拒绝 TP2。新增 `tests/test_server_tp_config.py` 先复现原失败，再开放共置入口到 TP1/TP2；实际 worker 组织与单 owner 线程不另起一套执行实现。原始失败、重放、修复及测量结果保存在租机数据盘任务目录的 `runs/acceptance-20261118`，其中 `tp2-fp32-partials.patch` 对应本节实现，修复前后文件分别保留。

## 历史学校服务器复现模板

以下命令沿用学校内网服务器环境，在服务器项目根目录执行，**尚未运行**。先固定经审核的候选 commit，并确认指定的两张 GPU 获准使用且空闲。该旧机器的新增文件、缓存和日志位于 `/Data2/zincion/ai-infra` 下；当前 L40S 租机的目录与入口见 [租机记录](server_hardware_l40s_2026-11-17.md)。`M4_DATA` 每轮使用新目录。示例依赖 M2 已保存的配置及捕获文件；如果路径不同，应先定位同一模型 revision、精度和输入，不重新生成假数据。

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
  --basetemp "$M4_DATA/pytest-gpu" tests/test_tp_engine_gpu.py tests/test_tp_gpu.py \
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
