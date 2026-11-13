# M2 GPU 验收与测量复现

以下命令用于有空闲 GPU 时手动执行，不表示已获得测量结果。模型和实验数据均放在 `/Data2/zincion`。先将对应代码放到服务器仓库目录，再在仓库根目录操作；以记录的 commit 和固定模型 revision 标识实验。

## 环境与原始记录

```bash
export UV_CACHE_DIR=/Data2/zincion/uv-cache
export UV_PYTHON_INSTALL_DIR=/Data2/zincion/python
uv python install 3.11.16
uv sync --locked --python 3.11.16
mkdir -p /Data2/zincion/experiments/m2
git rev-parse HEAD > /Data2/zincion/experiments/m2/code-commit.txt
uv run python -c "import sys; print(sys.version)"
uv run python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.version.cuda, torch.cuda.nccl.version())" \
  > /Data2/zincion/experiments/m2/versions.txt
nvidia-smi > /Data2/zincion/experiments/m2/nvidia-smi.txt
nvidia-smi topo -m > /Data2/zincion/experiments/m2/topology.txt
nvidia-smi topo -p2p r > /Data2/zincion/experiments/m2/p2p-read.txt
nvidia-smi topo -p2p w > /Data2/zincion/experiments/m2/p2p-write.txt
numactl --hardware > /Data2/zincion/experiments/m2/numa.txt
```

需要系统已有 `uv` 和 `numactl`。单独记录驱动、Python、框架、NCCL 和代码版本。锁文件中的 vLLM 为 0.15.0、Transformers 为 4.57.3；不要通过升级依赖掩盖验证失败。

## HF 输出及 logits 对齐

两个框架分进程顺序运行，避免同一卡同时装两份 14B。下例使用物理 GPU0；若改卡，保存实际的 `CUDA_VISIBLE_DEVICES` 与 NUMA 绑定。

```bash
uv run python tools/compare_hf.py prepare \
  --config configs/qwen3_14b.json \
  --output /Data2/zincion/experiments/m2/prompts-14b.json

CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 \
  uv run python tools/compare_hf.py custom \
  --config configs/qwen3_14b.json \
  --prompts /Data2/zincion/experiments/m2/prompts-14b.json \
  --max-tokens 32 --output /Data2/zincion/experiments/m2/custom-14b.pt \
  > /Data2/zincion/experiments/m2/custom-14b.log 2>&1

CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 \
  uv run python tools/compare_hf.py reference \
  --config configs/qwen3_14b.json \
  --custom /Data2/zincion/experiments/m2/custom-14b.pt \
  --max-tokens 32 --attention sdpa --sdpa-backend flash \
  --output /Data2/zincion/experiments/m2/alignment-14b.json \
  > /Data2/zincion/experiments/m2/reference-14b.log 2>&1
```

manifest 物化三条套 chat template 的配置 prompt，以及 1024/4096/16384 个 token 的长输入。custom 输出保存每个 step 的 logits，包括 prefill 最后一个位置和所有 decode 位置；参考输出逐 case 落盘，summary 的 `status=complete` 才表示所有 case 完成。

参考端同时测独立 greedy generation 和同前缀 teacher forcing；后者使用 `prompt + custom_tokens[:-1]`，只生成所需后缀 logits，避免长 prompt 的完整词表输出占满显存。记录首个分歧、该位置 HF top1/top2 margin、max/mean abs error、top1 一致率。

先使用 0.6B 做 FP32/BF16 对照时，应在仓库外复制配置，覆盖本地 checkpoint/cache 路径、dtype、长度预算及 KV 上限；保留固定 revision。使用本地 snapshot 路径加载 tokenizer，避免该 Transformers 版本离线时仍访问 Hub。FP32 的 Flash SDPA 不可用，可**显式**选用：

```bash
uv run python tools/compare_hf.py reference \
  --config /path/to/temporary-fp32-config.json \
  --custom /path/to/custom-fp32.pt --max-tokens 32 \
  --attention sdpa --sdpa-backend efficient --expand-kv \
  --output /path/to/alignment-fp32.json
```

`--expand-kv` 在对比进程内选择 HF 4.57.3 自带的 `repeat_kv` 分支，不修改安装包。原因是 PyTorch 2.9.1 的 efficient SDPA 不支持 `enable_gqa=True`；默认 FP32 SDPA 的 math 路径在 16 GB 设备上可能因长 attention 矩阵 OOM。backend、KV expansion 和 dtype 均记录在报告中，没有自动切换后端或精度。选择后端后的长输入实测仍属于 GPU 验收。

阈值必须在拿到上述数据后提出并由人工确认；工具只报告，不根据结果自动放宽验收阈值。

## 启动容量与服务

```bash
uv run python tools/capacity_report.py \
  --log /Data2/zincion/experiments/m2/custom-14b.log \
  --config configs/qwen3_14b.json --output-tokens 128 \
  --output /Data2/zincion/experiments/m2/capacity-14b.json
```

先核对参数加载完整性、RoPE cache、warmup 峰值、实际 block 数和三档长度的 block-rounded 并发上界，再选择服务及 bench 的并发。候选资源参数尚未实测，不把立项的 13 GiB/8.5 万 token 粗估作为容量结论。调整配置时使用仓库外的实验 JSON，并与原始数据一起保存。

开启计时的实验配置：

```bash
uv run python - <<'PY'
import json
from pathlib import Path
config = json.loads(Path("configs/qwen3_14b.json").read_text())
config["timing_path"] = "/Data2/zincion/experiments/m2/server-timing.jsonl"
Path("/Data2/zincion/experiments/m2/server-timed.json").write_text(json.dumps(config, indent=2))
PY

CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 \
  uv run python -m myvllm.server \
  --config /Data2/zincion/experiments/m2/server-timed.json \
  --served-model-name Qwen3-14B --port 8000 \
  > /Data2/zincion/experiments/m2/server.log 2>&1
```

该命令在前台运行；另一个终端启动客户端。启动第二个模型实例前先关闭前一个，HTTP 端口不同仍会冲突于固定 rendezvous。关闭服务也会在既有退出同步后刷新最后的 event 记录。

## 数据集与请求文件

固定数据集：

- ShareGPT：`anon8231489123/ShareGPT_Vicuna_unfiltered`，revision `192ab2185289094fc556ec8ce5ce1e8e587154ca`，文件 `ShareGPT_V3_unfiltered_cleaned_split.json`。
- LongBench：`THUDM/LongBench`，revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`，`data.zip` 中的 `qasper` 子集，prompt 为 context 与 input 拼接后套 chat template。

```bash
for m2_dataset in sharegpt longbench; do
  uv run python tools/prepare_workloads.py \
    --config configs/qwen3_14b.json --dataset "$m2_dataset" \
    --num-requests 100 --output-tokens 128 --seed 7 \
    --cache-dir /Data2/zincion/huggingface \
    --output "/Data2/zincion/experiments/m2/${m2_dataset}.jsonl"
done

for m2_length in 1024 4096 16384; do
  uv run python tools/prepare_workloads.py \
    --config configs/qwen3_14b.json --dataset random --input-length "$m2_length" \
    --num-requests 100 --output-tokens 128 --seed 7 \
    --cache-dir /Data2/zincion/huggingface \
    --output "/Data2/zincion/experiments/m2/random-${m2_length}.jsonl"
done
```

ShareGPT 取首个人类消息；两个真实数据集先套 template，再按完整输入加固定输出是否超过 `max_model_length` 过滤，不截断原样本。固定 seed 打乱后取样；不足请求数会报错。random 输入逐条验证 tokenizer round trip 后的准确长度。输出为 vLLM custom JSONL，旁边的 manifest 记录 revision、seed、过滤数、SHA256 和 bench 参数。

bench 使用 `--skip-chat-template --disable-shuffle`，防止对已经物化的请求二次套模板、再次改变顺序。换 tokenizer revision、template 或长度配置后必须重新物化。

## B0 与计时口径

客户端 tokenizer 应使用服务器缓存中对应 revision 的**本地 snapshot 目录**。下例固定到本次 14B revision；若 Hugging Face cache 路径不同，显式修改路径。

```bash
export M2_MODEL_SNAPSHOT=/Data2/zincion/huggingface/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18

# L1：示例并发为 1；其他固定并发先从实际容量报告选择。
for m2_length in 1024 4096 16384; do
  numactl --cpunodebind=0 --membind=0 uv run python tools/run_b0.py \
    --mode l1 --concurrency 1 --num-prompts 100 \
    --dataset "/Data2/zincion/experiments/m2/random-${m2_length}.jsonl" \
    --model Qwen3-14B --tokenizer "$M2_MODEL_SNAPSHOT" \
    --output-dir /Data2/zincion/experiments/m2/b0-l1
done

# L2：这些速率仅是起始扫描点，不是已测容量。
for m2_dataset in sharegpt longbench; do
  numactl --cpunodebind=0 --membind=0 uv run python tools/run_b0.py \
    --mode l2 --request-rates 1 2 4 --num-prompts 100 \
    --dataset "/Data2/zincion/experiments/m2/${m2_dataset}.jsonl" \
    --model Qwen3-14B --tokenizer "$M2_MODEL_SNAPSHOT" \
    --output-dir /Data2/zincion/experiments/m2/b0-l2
done
```

wrapper 使用未修改的 vLLM 0.15.0 CLI 模块，统一传 `--temperature 0 --ignore-eos --seed 7 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --save-result --save-detailed`。`python -m vllm.entrypoints.cli.main bench serve` 与 `vllm bench serve` 使用同一入口；前者也适用于移动过虚拟环境后 console-script shebang 失效的情况。command JSON 保存精确命令、代码版本、数据 manifest 和客户端 NUMA 信息。

wrapper 默认另发 8 次 warmup，可显式覆盖 `--num-warmups`。这只是固定的实验控制值；应从日志/trace 确认正式测量期间是否仍发生编译，不能把预热充分当成已验证事实。

例如检查 L1 单并发结果：

```bash
uv run python tools/analyze_bench.py \
  --result /Data2/zincion/experiments/m2/b0-l1/random-1024-l1-1.json \
  --server-timings /Data2/zincion/experiments/m2/server-timing.jsonl \
  --request-id-prefix random-1024-l1-1- \
  --output /Data2/zincion/experiments/m2/timing-check.json
```

逐请求核对 `len(itl) + 1 == output_tokens`，并按 request id 比较客户端 TTFT 与服务端“收到请求至首次 ASGI send 返回”。两者相差包含客户端/网络/HTTP 开销；不能预先认定为零。客户端的初始验证请求会复用第一条 ID，分析工具采用该 ID 最后的首 chunk 记录。

再用相同请求列表和 seed 运行一次，把输出放到另一个目录；`analyze_bench.py --compare-arrivals <第二次详细结果>` 比较实际相对发送时间。vLLM 0.15.0 的 `serve.py` 会保存 `start_times`（详细结果分支），其 `get_request` 采样 gamma 间隔并归一化累计时间；seed 控制随机计划，事件循环与并发排队仍影响实际发送时间。保存了实际时间不代表能按原时间戳回放；这个版本没有 `timed_trace`。

分别关闭/开启 `timing_path`，用同一单并发请求文件、相同 seed、预热和重复次数重测，保存每次 detailed JSON 后再比较开销。当前没有该 GPU 实测结果。

为核对 vLLM 服务的 chunk 口径，先停止本项目服务，再在可用卡上启动：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m vllm.entrypoints.cli.main serve \
  /path/to/pinned/Qwen3-0.6B/snapshot --served-model-name Qwen3-0.6B \
  --dtype bfloat16 --max-model-len 2048 --max-num-batched-tokens 2048 \
  --max-num-seqs 4 --gpu-memory-utilization 0.5 --enforce-eager --port 8001
```

使用相同的物化 0.6B 请求分别测两个服务，保存原始 SSE、usage、detailed JSON，检查是否有额外 role/finish choices 消息、合并 token 或缺失 usage。完整 vLLM 性能参照属于 M3；这里仅验计时契约。

保存一个固定请求的原始响应，例如当前运行的 vLLM 服务：

```bash
curl -N http://127.0.0.1:8001/v1/completions \
  -H 'Content-Type: application/json' -H 'x-request-id: protocol-probe' \
  --data-binary '{"model":"Qwen3-0.6B","prompt":"Explain 你好 and emoji 😀.","temperature":0,"max_tokens":16,"ignore_eos":true,"stream":true,"stream_options":{"include_usage":true}}' \
  > /Data2/zincion/experiments/m2/vllm-probe.sse

uv run python - <<'PY'
import json
from pathlib import Path
rows = [json.loads(line[6:]) for line in Path("/Data2/zincion/experiments/m2/vllm-probe.sse").read_text().splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"]
choices = [row for row in rows if row.get("choices")]
usage = [row["usage"] for row in rows if row.get("usage")][-1]
print({"choices_messages": len(choices), "completion_tokens": usage["completion_tokens"],
       "one_chunk_per_token": len(choices) == usage["completion_tokens"]})
PY
```

对本项目服务重复相同请求时，应以临时 0.6B 配置启动、指定同样的 served model name，改用其端口并保留另一份 SSE 文件；一次只运行正在核对的模型实例。

拿到 14B 单请求 TTFT/TPOT 后，提出有数据依据的暂定 SLO，经人工确认再统计。`analyze_bench.py` 可用 `--ttft-slo-ms` 和 `--tpot-slo-ms` 离线重算通过比例及 request goodput。失败请求保留在分母，核心容量口径仍是 SLO 达标率至少 90% 下的最大每卡请求率；一次负载下的 passing requests/s 不等于已经找到这个最大值。

## PCIe / NUMA / NCCL

执行前用新的 `topo -m` 和 `/sys/bus/pci/devices/<BDF>/numa_node` 核对 GPU 的 NUMA 节点。以下两个主存节点 0/1 来自硬件快照；若实际拓扑不同，调整节点列表。测试应在目标卡空闲且没有其他实验干扰时运行。

```bash
mkdir -p /Data2/zincion/experiments/m2/pcie
unset CUDA_VISIBLE_DEVICES
for m2_node in 0 1; do
  for m2_gpu in 0 1 2 3 4; do
    for m2_mode in h2d d2h bidirectional; do
      numactl --cpunodebind="$m2_node" --membind="$m2_node" \
        uv run python tools/pcie_benchmark.py --mode "$m2_mode" --gpus "$m2_gpu" \
        --output "/Data2/zincion/experiments/m2/pcie/${m2_mode}-gpu${m2_gpu}-host${m2_node}.jsonl"
    done
  done
  for m2_pair in "0 1" "3 4"; do
    read -r m2_a m2_b <<< "$m2_pair"
    numactl --cpunodebind="$m2_node" --membind="$m2_node" \
      uv run python tools/pcie_benchmark.py --mode dual-d2h --gpus "$m2_a" "$m2_b" \
      --output "/Data2/zincion/experiments/m2/pcie/dual-${m2_a}-${m2_b}-host${m2_node}.jsonl"
  done
  for m2_pair in "0 1" "1 0" "0 3" "3 0"; do
    read -r m2_a m2_b <<< "$m2_pair"
    numactl --cpunodebind="$m2_node" --membind="$m2_node" \
      uv run python tools/pcie_benchmark.py --mode twohop --gpus "$m2_a" "$m2_b" \
      --output "/Data2/zincion/experiments/m2/pcie/twohop-${m2_a}-${m2_b}-host${m2_node}.jsonl"
  done
  for m2_pair in "0 1" "0 3"; do
    read -r m2_a m2_b <<< "$m2_pair"
    for m2_sender in 0 1; do
      NCCL_DEBUG=INFO numactl --cpunodebind="$m2_node" --membind="$m2_node" \
        uv run python -m torch.distributed.run --standalone --nproc-per-node=2 \
        tools/pcie_benchmark.py --mode nccl --gpus "$m2_a" "$m2_b" --sender "$m2_sender" \
        --output "/Data2/zincion/experiments/m2/pcie/nccl-${m2_a}-${m2_b}-sender${m2_sender}-host${m2_node}.jsonl" \
        > "/Data2/zincion/experiments/m2/pcie/nccl-${m2_a}-${m2_b}-sender${m2_sender}-host${m2_node}.log" 2>&1
    done
  done
done
```

默认大小为 1/4/16/64/160/640/1024/2560 MiB，包含三档整体 KV 和 40 层拆分后的单层切片。每档先 warmup 5 次，再测 20 次，保存全部样本、中位数和 p10/p90；允许通过 CLI 显式覆盖并记录参数。

主存一律使用 pinned memory，并记录 buffer 的 NUMA mapping。双向拷贝使用不同的设备 buffer；两跳严格先完成 D2H 再提交 H2D。并发场景的 CUDA 指标是各 stream 时长的最大值，**不包含 host 提交偏移**，同时保存 host wall samples，不能把两者混为同一时间定义。NCCL 记录版本、rank、方向、INFO transport 日志；不要预先假定其选择 SHM 或某条网络。

脚本在持续 warmup 传输时采样实际 PCIe 代际和宽度。理论单向 Gen4 x16 为 `16 × 16 × 128/130 ÷ 8 ≈ 31.5 GB/s`；JSON 同时给出相对比例。两跳的 event 带宽以一份逻辑 payload 除以两跳 event 时长之和，排除 host handoff；`effective_wall_GB_s` 使用包含该开销的 wall 时间。不是把两份字节相加。并发双向另外给出 aggregate bandwidth。

当前没有 4090/NUMA/NCCL 微基准原始数据。应先完成这些实验，再决定 KV 传输优化的方向，不能用理论链路值替代实测。
