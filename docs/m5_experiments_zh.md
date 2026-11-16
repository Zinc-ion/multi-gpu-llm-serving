# M5 服务器实验准备

本文只提供尚未在目标服务器执行的配置生成与实验命令；服务器 14B 正确性、异常回收、容量、跨 GPU/NUMA 和 B2 数据均待验收。本机 tiny 同卡 P/D 与 HTTP 功能检查不能代替这些验收。不要使用 M5 候选重新填报旧候选的 B0/B1；它们各自在固定 commit 上独立完成。

## 生成前提

先确认获准 GPU 空闲，重新运行 `nvidia-smi topo -m`、`nvidia-smi topo -p2p r/w`、`numactl --hardware` 并核对 GPU 的 PCIe NUMA 节点。生成器假设 GPU0/1 在 NUMA 0，GPU3/4 在 NUMA 1；worker 的 `numactl --cpunodebind/--membind` 绑定到其 GPU 所在节点，网关和 bench 客户端绑定节点 0。若现场拓扑不同，不得沿用这组命令当成同/跨 NUMA 实验。每个 worker 使用 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 和单个物理 GPU 的 `CUDA_VISIBLE_DEVICES`，进程内是逻辑 `cuda:0`、rank 0、TP=1；确认设备编号与 `nvidia-smi`/UUID 对应关系。

选择已经固定的 M5 candidate commit 和 tag、其工作树、只读现有 Python 与固定 14B snapshot。tag 必须指向该 commit，运行脚本会检查工作树 HEAD 和 tag；生成的 manifest 与各角色 JSON 保存完整选择和配置。核对 HEAD 对应代码及工作树内容，无关修改不得混入实验。模型和 Python 可以只读使用工作目录外的现有文件，所有**新增**的配置、模型下载、缓存、临时文件、日志和结果必须在 `/Data2/zincion/ai-infra` 内。生成器会拒绝该根目录以外或根目录自身作为输出目录，并要求输出目录尚不存在。

复用 M2 已物化的五个 JSONL 和同名 `.manifest.json`：random 的 1024/4096/16384 token 输入、ShareGPT 和 LongBench。以下五个变量必须指向实际保留的文件，不自动重新生成。先核对 manifest 的模型/tokenizer revision、`source.input_length`、输出预算、样本数和 SHA256，及其与比较用 B0/B1 的对应关系。`run_b0.py` 会再校验 JSONL 的 SHA256，禁止超过物化数量取样。

下面以 `2p2d` 为例；单独生成目录并把 `--topology` 改为 `same-numa`（P GPU0、D GPU1）或 `cross-numa`（P GPU0、D GPU3）用于 1P1D 功能和传输实验。这些 1P1D 数据使用两张卡，不能直接与四卡的 B0/B1 goodput 作公平比较。`2p2d` 固定 P GPU0/1、D GPU3/4，四个实例各 TP=1，使用固定 pair 轮询；与 B0/B1 比较必须使用同样四张卡和相同模型、精度、负载、请求列表及 SLO，不能凭生成的配置宣称容量相同。

## 生成配置和脚本

在服务器执行前将以下路径变量设为实际只读文件及已批准的候选。`M5_OUT` 每次使用新目录；不运行 SSH 或远程命令的本轮不会产生这些结果。

```bash
export M5_ROOT=/Data2/zincion/ai-infra
export M5_BOOT="$M5_ROOT/experiments/m5/bootstrap"
export M5_OUT="$M5_ROOT/experiments/m5/2p2d-run-01"
export M5_REPO="$M5_ROOT/MinivLLM"             # 服务器上的实际固定候选工作树
export M5_PYTHON=/Data2/zincion/envs/m2/bin/python # 已有只读 Python 环境，按现场修正
export M5_SNAPSHOT="$M5_ROOT/huggingface/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18"
export M5_TAG=m5-candidate-XX                     # 替换为实际固定候选 tag
export M5_COMMIT=REPLACE_WITH_FULL_CANDIDATE_SHA
export M5_L1_1K=REPLACE_WITH_EXISTING_RANDOM_1024_JSONL
export M5_L1_4K=REPLACE_WITH_EXISTING_RANDOM_4096_JSONL
export M5_L1_16K=REPLACE_WITH_EXISTING_RANDOM_16384_JSONL
export M5_L2_SHAREGPT=REPLACE_WITH_EXISTING_SHAREGPT_JSONL
export M5_L2_LONGBENCH=REPLACE_WITH_EXISTING_LONGBENCH_JSONL

# 在执行任何 Python/uv 进程前设置新增缓存和临时文件位置。
export PYTHONPYCACHEPREFIX="$M5_BOOT/cache/pycache"
export UV_CACHE_DIR="$M5_BOOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$M5_BOOT/python"
export XDG_CACHE_HOME="$M5_BOOT/cache/xdg"
export HF_HOME="$M5_BOOT/cache/huggingface"
export HF_HUB_CACHE="$M5_BOOT/cache/hf-hub"
export HF_DATASETS_CACHE="$M5_BOOT/cache/datasets"
export TORCHINDUCTOR_CACHE_DIR="$M5_BOOT/cache/torchinductor"
export TRITON_CACHE_DIR="$M5_BOOT/cache/triton"
export CUDA_CACHE_PATH="$M5_BOOT/cache/cuda"
export VLLM_CACHE_ROOT="$M5_BOOT/cache/vllm"
export NUMBA_CACHE_DIR="$M5_BOOT/cache/numba"
export TMPDIR="$M5_BOOT/tmp"
mkdir -p "$PYTHONPYCACHEPREFIX" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" \
  "$XDG_CACHE_HOME" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" \
  "$VLLM_CACHE_ROOT" "$NUMBA_CACHE_DIR" "$TMPDIR"
cd "$M5_REPO"
export PYTHONPATH="$M5_REPO/src"
test "$(git rev-parse HEAD)" = "$M5_COMMIT"
test "$(git rev-parse "${M5_TAG}^{commit}")" = "$M5_COMMIT"

M5_ARGS=(--output-dir "$M5_OUT" --topology 2p2d \
  --candidate-commit "$M5_COMMIT" --candidate-tag "$M5_TAG" \
  --repo-dir "$M5_REPO" --python "$M5_PYTHON" --model-path "$M5_SNAPSHOT" \
  --l1-1k "$M5_L1_1K" --l1-4k "$M5_L1_4K" --l1-16k "$M5_L1_16K" \
  --l2-sharegpt "$M5_L2_SHAREGPT" --l2-longbench "$M5_L2_LONGBENCH")
"$M5_PYTHON" tools/prepare_pd.py "${M5_ARGS[@]}" --dry-run > "$M5_BOOT/plan.json"
"$M5_PYTHON" tools/prepare_pd.py "${M5_ARGS[@]}"
git status --short > "$M5_OUT/git-status.txt"
nvidia-smi > "$M5_OUT/nvidia-smi.txt"
nvidia-smi topo -m > "$M5_OUT/topology.txt"
numactl --hardware > "$M5_OUT/numa.txt"
```

生成的 `env.sh` 将相同类别的缓存和临时文件重新指向本次 `M5_OUT`；每个脚本在启动 Python 前 source 它，不修改 `HOME`。配置继承 `configs/qwen3_14b.json` 的 BF16 及固定 revision，配置缓存也位于 `M5_OUT`。默认 worker 的 `max_transfer_bytes=3 GiB`，网关的单 payload 上限也是 3 GiB、总逻辑中转预留 6 GiB；14B BF16 每 token 结构 KV 为 163840 B，16384 token 的整块 payload 为 2.5 GiB。这是可容纳该大小的配置界限，并非启动后的实测显存、进程 RSS 或可用并发。每个 worker 的 3 GiB 为该 worker **所有请求的总预留**，故不能据此假定多个 16K 请求能同时交接。生成器现在也可独立设置 worker 的聚合与单 payload 限额，以及网关的 payload 和 staging 预算；默认单 payload 仍为 3 GiB，字段、旧版本兼容和逻辑记账边界见 [M8 runtime](m8_runtime_zh.md#p-可选择-d)。更改这些限制须保留配置和对应测量，不能用理论上限代替容量日志。

## 前台启动、检查与关闭

分别在独立终端以相同 `M5_OUT` 启动本次生成的进程，先 D、后 P；下例为 2P2D。每条命令在自己的前台运行，重启时使用新的日志文件名，避免覆盖。

```bash
bash "$M5_OUT/decode-0.sh" > "$M5_OUT/decode-0-start-01.log" 2>&1
bash "$M5_OUT/decode-1.sh" > "$M5_OUT/decode-1-start-01.log" 2>&1
bash "$M5_OUT/prefill-0.sh" > "$M5_OUT/prefill-0-start-01.log" 2>&1
bash "$M5_OUT/prefill-1.sh" > "$M5_OUT/prefill-1-start-01.log" 2>&1
```

等待所有 worker 加载并监听后，在另一个终端执行实际网络检查，再启动网关：

```bash
bash "$M5_OUT/workers-ready.sh" > "$M5_OUT/workers-ready-01.json"
bash "$M5_OUT/gateway.sh" > "$M5_OUT/gateway-start-01.log" 2>&1
# 网关启动后另一个终端运行。
bash "$M5_OUT/ready.sh" > "$M5_OUT/ready-01.json"
```

`workers-ready.sh` 查询各 worker 的真实 `info` 并核对角色及 P 的固定 D 地址。`ready.sh` 另外检查网关 OpenAPI 端点存在；网关 startup 本身还检查模型规格、EOS、长度和 seed。它们证明连通性和启动检查通过，不证明模型数值或 KV 传输正确。失败时保存日志，不自动重试其他 pair。停止或重启某一进程时，在对应前台终端使用 Ctrl-C、等待该进程退出，再以该角色脚本启动新进程；不使用匹配进程名或进程组的批量 kill。脚本不生成后台 PID 管理器，也不操作其他作业。

## 正确性、资源与故障待验收

1. 先核对各启动日志的模型加载、GPU 映射、warmup 峰值、实际 KV block 与空闲显存，并记录 graph 模式；按实际 block 容量决定并发。任一实例无法装下配置的总长度时记录失败并另存新配置，不静默缩短请求。
2. 使用与 14B 共置参照捕获完全相同的 token 输入、greedy 参数、精度及预先批准的 logits 阈值，在不同物理 GPU 的 1P1D 上对照首 token、后续 token 和相同前缀 logits。分别覆盖 GPU0→1 和 GPU0→3，包含 1K/4K/16K、块边界、EOS、首 token 即结束及多个输出长度。现有本机 tiny GPU 对照和阈值不等于 14B 正式阈值。
3. 通过真实 HTTP stream/普通请求、客户端提前关闭、受控关闭本实验 P 或 D，检查存活侧 `PDClient.stats` 的 handle、block、reservation 和缓冲归零；重启相应角色并验证后续请求。超时、D 容量不足、恢复失败及流中断须保留各自状态码、错误事件和原始日志。跨 GPU 失败恢复不可用本机同卡结果代替。
4. 对每次请求保存 gateway、P、D 各自 JSONL，使用 `tools/analyze_pd.py --timings ... --output ...` 关联 session/transaction，检查实际 payload、export、restore、handoff、第二 token 与释放；进程间不作绝对时钟相减。`pd_gateway_handoff_done` 与客户端端到端时间不属于同一时钟起止定义，按运行说明分别解释。

可复用的捕获依据及当前缺口如下：

- `tools/compare_hf.py prepare` 输出包含模型/tokenizer revision 和 `cases` 的 JSON；每个 case 保存 `name`、`prompt`、`input_ids`。M2 已保存的输入可直接复用，不必重新下载或重新采样。
- `tools/compare_hf.py custom --config <配置> --prompts <输入JSON> --max-tokens <数量> --output <输出PT>` 在**共置** `LLMEngine` 的 LM head 上注册 hook，输出 `{"metadata": ..., "cases": [...]}`；各 case 的 `input_ids`、`token_ids` 和逐输出位置的二维 `logits` 可作为共置参照。`tools/compare_runs.py --baseline <PT> --candidate <PT> --output <JSON>` 接受这种结构，只比较首次 token 分歧及之前的同前缀 logits，报告 `logits_gate=null`，不会替用户决定正式阈值。
- `tests/pd_gpu_helpers.py::CaptureEngine` 及 `tests/test_pd_gpu.py::test_real_socket_pd_matches_colocated_gpu` 是现有 tiny P/D 测试 hook：通过测试进程注入真实引擎，分别保存共置、P、D 的 PT；其结构为 `requests[request_id].events` 中的 token/logits，以及 blocks、KV、step 信息，与 `compare_hf.py custom` 输出格式不同。该测试还把 P 首 token 和 D 后续事件拼接，在相同前缀比较 logits 和有效 KV。它的 tiny checkpoint、固定小 payload 上限和本机容差不能直接作为 14B 验收。
- `tools/compare_pd.py capture` 提供独立 reference、P、D 的 TP1 捕获，按 case 输出可直接交给 `compare_runs.py` 的 PT；使用方法及来源、进程和 KV 边界见 [M8 runtime](m8_runtime_zh.md#独立-pd-数值捕获)。原始 `input_ids` 是模型输入的权威来源；prepare 截断多字节字符时，`prompt` 可以是有损解码表示，捕获器记录 lossy/nonroundtrip 观测，不重新编码替换 ID。它也可显式选择 M5 的 bulk 接口；pipeline 要求引擎支持该接口。捕获 hook 会把 logits/KV 搬到 CPU，只适用于正确性诊断，不应开启后拿来衡量生产延迟。正式 14B logits 门槛和目标部署的验证仍须另行完成。

## L1/L2 与客户端分析

所有脚本在已就绪的服务上执行，结果目录是本次新 `M5_OUT/results` 下的独立 B2 路径；若相应子目录已存在则拒绝运行，复跑须生成新 `M5_OUT`。L1 每档默认全列表、并发 1、1 次；L2 使用完整物化列表、到达 seed `7..11`、每个速率 5 次、warmup 8 次，起始速率为 1/2/4 req/s；必须与对应对照组核对并固定这些值，速率只是扫描起点，不是已测容量。保存由 `run_b0.py` 输出的命令、manifest、详细结果与实际发送时间。

```bash
bash "$M5_OUT/bench-l1-1k.sh" > "$M5_OUT/bench-l1-1k.log" 2>&1
bash "$M5_OUT/bench-l1-4k.sh" > "$M5_OUT/bench-l1-4k.log" 2>&1
bash "$M5_OUT/bench-l1-16k.sh" > "$M5_OUT/bench-l1-16k.log" 2>&1
bash "$M5_OUT/bench-l2-sharegpt.sh" > "$M5_OUT/bench-l2-sharegpt.log" 2>&1
bash "$M5_OUT/bench-l2-longbench.sh" > "$M5_OUT/bench-l2-longbench.log" 2>&1
```

核对各 detailed JSON 的输出 token 数与 ITL 样本、失败及截断，再用 `tools/analyze_bench.py` 分析客户端 TTFT/ITL/TPOT/E2E；只有用户已基于独立 B0 数据确定 SLO 后才传入 SLO 参数。按相同重复、到达 seed、请求列表、warmup、速率及 GPU 数比较，报告失败请求、逐次分布和单卡 goodput；未找到满足 SLO 的最大速率时不能把单次吞吐写成该上界。网关只支持固定 pair，不提供 M8 负载感知。M2 完整 B0/SLO 与 M3 B1/服务器验收依然独立未完成。
