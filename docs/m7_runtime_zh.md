# M7：FP8 KV 存储、attention 与 PD 传输

本单元为引擎增加可选的 FP8 KV 存储、读取和 PD 传输。默认不启用；模型权重、activation 和 logits 的 dtype 仍由原有 `dtype` 配置决定。FP8 KV 是有损量化，不能把它与 BF16 KV 的输出差异当作无损传输误差。本机 tiny 模型的功能验证不能代替目标服务器、14B、长上下文质量或性能验收。

当前交付按 [最小验收](../.codex/tasks/m8_acceptance.md) 做少量 Qwen3-14B 代表输入、资源与端到端记录；下文完整任务质量工具属于保留的扩展，不要求 Math500/NIAH 全量或两种 prefill 计算模式全排列。

## 配置与布局

在模型 JSON 配置中设置：

```json
{
  "dtype": "bfloat16",
  "kv_cache_dtype": "float8_e4m3fn"
}
```

`kv_cache_dtype` 只接受 `auto` 和 `float8_e4m3fn`；省略时为 `auto`，KV dtype 跟随模型 dtype，并继续使用原来的计算路径。其他值在 runner 初始化时拒绝。

FP8 模式下，`ModelRunner.kv_cache` 的 dtype 为 `torch.float8_e4m3fn`，布局仍为 `[2, layers, blocks, block_size, local_kv_heads, head_dim]`，第一维依次为 K、V。`ModelRunner.kv_cache_scales` 是独立的 FP32 tensor，布局为 `[2, layers, blocks, block_size, local_kv_heads]`。每层 `Attention.k_scale_cache`、`Attention.v_scale_cache` 分别绑定对应的 K、V scale view；`auto` 模式下三个 scale 属性均为 `None`。

每个有效 token 的每个 K head、V head 分别计算 `amax = max(abs(values))` 和 `scale = max(amax / 448, float32_tiny)`；全零 head 的 scale 为 1。写入数据为 `clamp(values / scale, -448, 448)` 转换到 E4M3FN。scale 随对应 token 存储，之后写入其他 token 不改变已有 token 的 scale。池初次分配时 data 为零、scale 为 1；释放的 block 再分配时，新有效位置由写入覆盖，不承诺清零历史无效位置。

FP32 转 E4M3FN 时，先将向零舍入的 FP16 中间值按被丢弃的 FP32 位设为奇数，再执行 FP8 最近偶数舍入，避免 SM89 上两次舍入导致的边界错误。`amax` 归约与 clamp 显式传播 NaN；含 NaN 的 head 得到 NaN scale，含 Inf 而无 NaN 的 head 得到 Inf scale，其数据遵循相同的除法和转换规则。

Triton store 将对应 token/head 的数据及 scale 直接写入池；除法保留 FP32 subnormal。运行时 whole/chunked paged prefill 和 decode 均按 block table 读取 FP8 数据及相同物理位置的 scale，按需反量化。FP8 prefill 默认在 FP32 中做反量化及 attention 中间计算；独立 decode kernel 使用 FP32 中间值，输出仍使用 query 的 dtype。cache 分配前的模型 warmup 沿用原有的无 cache prefill 路径。不存在逐步整池反量化或整池复制。

可选配置 `fp8_prefill_compute_dtype` 只接受 `float32`（省略时的原路径）和 `query`。`query` 保留 Q 的 dtype；K/V 仍用原 token/head scale 在 FP32 反量化，再转成 Q dtype，softmax 和累加器使用 FP32，概率在与 V 相乘前转为 V dtype。这是有不同舍入的实验路径，不保证与 `float32` 结果一致；例如反量化值超出 FP16 范围时，转换可能产生 Inf 并进一步产生 NaN。该选项作用于整个 paged prefill batch，包含与 prefill 混排的 decode 行；独立 decode kernel 的计算未改变。单 token prompt 尾 chunk 若单独调度为 decode 则走独立 decode kernel，若与其他 prefill 混排则走 paged prefill。普通非 FP8 cache、量化与 data/scale 传输格式不变。同一 K/V 输入的存储字节不变，但完整模型后续层可能因前层舍入差异产生不同输入及 cache 内容，也可能改变 argmax 和输出质量。配置在 engine 入口校验，并在初始化时传给各 Attention 层。

此机制借鉴 Apache-2.0 许可的 vLLM v0.15.0，commit `f176443446f659dbab5315e056e605d8984fd976`，`vllm/v1/attention/ops/triton_unified_attention.py` 的 FP8 K/V 反量化后转换 Q dtype 路径。本项目保留自己的 token/head scale 和缓存布局，不使用该源码的 layer scale。

CUDA Graph 捕获和重放复用相同的 data/scale 池地址。slot 为 `-1` 的 padding 不写 cache，长度为零的 decode padding 不读取有效 KV；真实 batch 的 block 分配、释放和重用仍由原有 Scheduler 管理。

`ModelRunner.run()` 统一处于 PyTorch inference mode，覆盖 sampler 及模型执行。原先仅模型执行处于 inference mode，sampler 编译回退到 eager 时会原地修改 inference tensor 而失败；此修改不改变采样算法。

## 容量

FP8 每个物理 block 的数据和 scale 字节数分别为：

```text
data  = 2 * layers * block_size * local_kv_heads * head_dim
scale = 2 * layers * block_size * local_kv_heads * 4
total = data + scale
```

data 每元素 1 字节，scale 每元素 4 字节。`local_kv_heads` 是每个 TP rank 的本地 head 数；公式已包含 K 和 V。实际容量还受 block 尾部填充影响，不能把 KV 显存或传输量一律写成 BF16 的一半。

启动时 profiling 以 data 与 scale 的总字节计算 block 预算。rank 0 的 `capacity` JSON 记录 `model_dtype`、`kv_cache_dtype`、`data_block_bytes`、`scale_block_bytes` 和 `block_bytes`；`tools/capacity_report.py` 使用该实际 `block_bytes` 推导池容量：

```bash
PYTHONPATH="$PWD/src" uv run --no-sync python tools/capacity_report.py \
  --log /path/to/startup.log --config /path/to/config.json \
  --output /path/to/capacity.json
```

报告只描述 block 池容量，不是稳定并发或 goodput。profiling 在 CUDA Graph capture 之前进行，graph 和其他运行时内存仍须实测。

## FP8 PD 交接

M5 整块交接与 M6 逐层交接均可传输 FP8 KV；P、D 仍使用各自的 TP=1 引擎和物理 block table，经主存和 TCP 传递原始 data 与对应 scale，D 不重算 prompt 或重新量化。启动网关时核对模型、cache 格式和 FP8 prefill 计算模式；D 在整块和早期预留前核对版本、模型及计算模式、字节序、形状和字节数。不匹配会被拒绝。worker info 缺少模式字段时按旧版 `float32` 解释；同一 FP8 部署中 P、D 和本地 prefill 使用同一模式。

`auto` 的 model spec 和整块版本 2、早期版本 3 保持原格式。FP8 的 model spec 另有：

```json
{"kv_cache": {"dtype": "float8_e4m3fn", "scale_dtype": "float32", "scale_granularity": "token_head"}}
```

上述形状用于默认 `float32`，与旧版相同；`query` 在 `kv_cache` 中额外包含 `"fp8_prefill_compute_dtype": "query"`。这是参与模型身份匹配的计算策略标记，不改变 cache 存储布局、data 或 scale 字节。缺少该标记表示原 `float32` 策略；旧版的严格模型匹配也会拒绝带 `query` 标记的 metadata。

FP8 整块 metadata 为版本 4，早期 metadata 为版本 5，逐层完成后的 `commit_early` 校验对应版本 4。`kv` 的 `shape` 为 `[2, layers, blocks, block_size, local_kv_heads, head_dim]`，`scale_shape` 为去掉最后一维的形状；`dtype`、`scale_dtype`、`byte_order`、`data_nbytes`、`scale_nbytes` 和总 `nbytes` 均参与校验。每个 payload 先按 C 顺序放置所有 FP8 data 字节，再按 C 顺序放置所有原生字节序 FP32 scale 字节。两段紧邻，没有额外 wire padding；即使 `head_dim` 为奇数，也通过 uint8 字节视图复制 scale，不依赖未对齐的 FP32 主存视图。

令 prompt 有 N 个 token、block 大小为 S、块数 `B=ceil(N/S)`、层数 L、本地 KV head 数 H、每个 head 维数 D。整块的 data 字节为 `2*L*B*S*H*D`，scale 字节为 `2*L*B*S*H*4`；末块无效位置的 data 为 0、scale 为 1。逐层分帧只传有效位置，每帧有 C 个 token 时 data 字节为 `2*C*H*D`，scale 字节为 `2*C*H*4`；各层合计 `2*L*N*H*(D+4)` 字节。D 的早期预留会将末块无效位置初始化为 data 0、scale 1，即使该物理块曾被使用。有效 data 和 scale 均按原始字节恢复。

流水仍为 opt-in：worker 两侧设置 `--pipeline-frame-bytes`，每帧至少容纳 `2*H*(D+4)` 字节；网关 `pipeline_min_tokens` 默认 0，继续走整块路径。流水支持 P whole/eager、D eager 或 graph；M6 的客户端顺序及其他约束见 [M6 逐层交接](m6_runtime_zh.md)，其中 FP8 不支持的旧状态以本节为准。每端每事务两个帧大小的 pinned slot 同时容纳 data 和 scale。P 的完成 event 在两者 D2H 后记录，runner 在 sampling/postprocess 前 drain，确保源 block 在 EOS、取消或重用前读完。D 的非末帧 ACK 只表示复制已提交；最终 `copies_done=True` 等待 data 和 scale 的 H2D 完成，并在计算流建立依赖。

传输预留和帧限制均计入 data 与 scale；网关对整块计入尾部填充，对流水只计算有效 token。启用 timing 时，FP8 的 `payload_bytes` 是总字节，相关 worker、gateway、pipeline 记录及 `tools/analyze_pd.py` 还区分 `data_bytes`、`scale_bytes`；部分 sendall 的实际提交量仍遵循 M6 的 `partial_send_unknown` 语义。`tools/compare_runs.py` 将缺失 `kv_cache_dtype` 的旧配置视为 `auto`，拒绝把不同 cache 格式作为无损比较。

## 验证范围

在执行前固定的算子比较使用独立 PyTorch 量化/反量化公式：FP8 原始字节逐位相等，FP32 scale `rtol=1e-6, atol=0`，零 head 的 scale 精确为 1。attention 与反量化后的 FP32 参考比较：FP32 输出 `rtol=1e-4, atol=1e-5`，FP16 输出 `0.005/0.005`，BF16 输出 `0.02/0.02`。这些容差覆盖浮点归约和输出 dtype 的舍入，比较对象已经使用相同的 FP8 KV，不是 BF16 与 FP8 之间的质量门槛。

本机 RTX 5080 Laptop GPU 上的算子测试覆盖零 head、动态范围与 FP32 subnormal、非连续物理 block、不完整尾块、GQA、causal offset、跨 block、slot padding 和零长度 decode。随机初始化的 tiny Qwen3 checkpoint 使用 `scale=1.0`，由 `Attention.forward` 再除以 `sqrt(head_dim)`；真实引擎验证 FP32/BF16 模型 dtype 下的 FP8 cache、whole/chunked、eager/CUDA Graph、batch 缩小和资源回收。该小模型只验证同一 FP8 格式内的 finite、greedy 和同前缀 logits，不证明 BF16 与 FP8 无损。

以下命令在可使用 GPU 0 的环境中执行；测试现场生成 tiny checkpoint，`--basetemp` 应指向新的本地临时目录：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD/src" \
  uv run --no-sync python -m pytest -q -s -p no:cacheprovider \
  --basetemp=/path/to/new-test-dir \
  tests/test_fp8_storage.py tests/test_fp8_attention.py tests/test_fp8_model_gpu.py

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD/src" \
  uv run --no-sync python -m pytest -q -p no:cacheprovider \
  --basetemp=/path/to/new-cpu-test-dir
```

本机 RTX 5080 Laptop GPU 的 FP8 PD 测试使用 `scale=1.0` 的 tiny checkpoint 和独立的共置、P、D 进程；FP32/BF16 模型 dtype、D eager/graph 四组合分别比较共置、整块和逐层路径。greedy token 一致，同前缀 logits 使用 FP32 `rtol=1e-4, atol=1e-5`、BF16 `rtol=0.02, atol=0.02`，有效 FP8 data 与 FP32 scale 按字节精确比较。测试还覆盖不完整尾块、跨块、并发、EOS、源块重用、双槽背压下的取消、连接中断、超时和后续请求；这是同卡本地功能验证。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD/src" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --no-sync python -m pytest -q -p no:cacheprovider \
  --basetemp=/path/to/new-pd-test-dir tests/test_pd_fp8_gpu.py
```

FP8 相对于 BF16 的任务质量、显存与吞吐/延迟变化，以及目标 4090、14B 和长上下文表现均待后续固定工作负载和质量门槛后实测。

`query` 的独立算子参考、配置及 P/D 边界测试位于 `tests/test_fp8_prefill_compute.py`。CPU Triton interpreter 覆盖 FP32/FP16 query 和 BF16 的原 `float32` 路径；Triton 3.5.1 interpreter 的 BF16 `tl.dot` 结果异常，不能据此验收 BF16 query。SM89 离线编译只验证代码生成。下面的真实 GPU 测试须由主 agent 安排设备，并设置 `PROFILE_PYTHON` 为 serving 环境的 Python 绝对路径；测试包含算子、tiny whole/chunked、eager/graph 及 bulk/pipeline/local 路径，不代替 14B 或目标服务器验收：

```bash
CUDA_VISIBLE_DEVICES=0 RUN_FP8_PREFILL_GPU=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH="$PWD/src:$PWD/tests:$PWD" \
  "$PROFILE_PYTHON" -m pytest -q -s -p no:cacheprovider tests/test_fp8_prefill_compute.py
```

以下是同一 4K 输入和 chunk 1024 的单卡探索性 profile：`PROFILE_BASE` 为本地 14B 配置，`PROFILE_PROMPTS` 为 `compare_hf.py prepare` 物化的含 `tokens-4096` case 的 manifest，`PROFILE_DIR` 是服务器工作目录下尚不存在的目录。两个 case 的 token ID 完全相同，第一个预热实际 paged prefill，第二个供记录；每种模式单独启动一次，原始配置、step timing、结果和 Nsight trace 分开保存。运行前需核对模型路径、设备、空间与源码候选，并由主 agent 排程；完整 forward 事件不等于单一 attention kernel 时间。

```bash
export PROFILE_BASE PROFILE_PROMPTS PROFILE_DIR PROFILE_PYTHON
mkdir "$PROFILE_DIR"
mkdir -p "$PROFILE_DIR/tmp"
export TMPDIR="$PROFILE_DIR/tmp" XDG_CACHE_HOME="$PROFILE_DIR/cache/xdg" XDG_CONFIG_HOME="$PROFILE_DIR/cache/config"
export HF_HOME="$PROFILE_DIR/cache/huggingface" HF_HUB_CACHE="$PROFILE_DIR/cache/hf-hub"
export TRITON_CACHE_DIR="$PROFILE_DIR/cache/triton" TORCHINDUCTOR_CACHE_DIR="$PROFILE_DIR/cache/torchinductor"
export CUDA_CACHE_PATH="$PROFILE_DIR/cache/cuda" PYTHONPYCACHEPREFIX="$PROFILE_DIR/cache/pycache"
"$PROFILE_PYTHON" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["PROFILE_DIR"])
base = json.loads(Path(os.environ["PROFILE_BASE"]).read_text())
prompts = json.loads(Path(os.environ["PROFILE_PROMPTS"]).read_text())
case = next(c for c in prompts["cases"] if c["name"] == "tokens-4096")
assert len(case["input_ids"]) == 4096
prompts["cases"] = [{**case, "name": name} for name in ("warm-4k", "profile-4k")]
(out / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2) + "\n")
for mode in ("float32", "query"):
    config = {**base, "kv_cache_dtype": "float8_e4m3fn", "fp8_prefill_compute_dtype": mode,
              "enable_chunked_prefill": True, "max_num_batched_tokens": 1024,
              "timing_path": str(out / f"{mode}.timing.jsonl")}
    (out / f"{mode}.json").write_text(json.dumps(config, indent=2) + "\n")
PY
for mode in float32 query; do
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PWD/src" \
    nsys profile --trace=cuda,nvtx,osrt --force-overwrite=false -o "$PROFILE_DIR/$mode" \
    "$PROFILE_PYTHON" tools/compare_hf.py custom --config "$PROFILE_DIR/$mode.json" \
    --prompts "$PROFILE_DIR/prompts.json" --max-tokens 1 --output "$PROFILE_DIR/$mode.pt" \
    >"$PROFILE_DIR/$mode.log" 2>&1
done
```

## 可选扩展：FP8 任务质量准备与比较

`tools/prepare_fp8_eval.py` 生成一组固定的单实例 BF16 配置：baseline 使用 `kv_cache_dtype=auto`，candidate 的 KV cache 使用 `float8_e4m3fn`，其余服务配置相同。质量客户端独立使用 evalscope `1.5.1`，不修改 serving 或 vLLM 环境。以下流程是源码核对后的准备方案，尚未安装 evalscope 或执行真实数据集评测。

先在服务器工作目录中准备 Qwen3-14B 模型及 tokenizer 本地快照、可由 evalscope 加载的本地 MATH-500 数据和 NIAH 英中语料；记录模型和两份语料的来源及 revision。配置中的模型和 tokenizer revision 是使用者声明，当前工具没有独立来源或预期文件哈希证明本地快照属于该 revision。NIAH 目录须包含 `PaulGraham_Essays.txt` 和 `Journey_to_the_West.txt`。evalscope 1.5.1 的本地加载器会删除数学数据目录中的 `dataset_infos.json`，因此准备工具拒绝含该文件的语料目录，不改动语料。下面的路径和完整候选 commit 均由实际实验者填写：

```bash
QUALITY_ROOT=/Data2/zincion/ai-infra/experiments/m7-quality/run-01
REPO=/Data2/zincion/ai-infra/MinivLLM
SERVER_PYTHON=/Data2/zincion/ai-infra/envs/serving/bin/python
MODEL=/Data2/zincion/ai-infra/models/Qwen3-14B-snapshot
MATH=/Data2/zincion/ai-infra/datasets/MATH-500-snapshot
NEEDLE=/Data2/zincion/ai-infra/datasets/needle-haystack-snapshot

cd "$REPO"
"$SERVER_PYTHON" tools/prepare_fp8_eval.py \
  --output-dir "$QUALITY_ROOT" --repo-dir "$REPO" --server-python "$SERVER_PYTHON" \
  --model-path "$MODEL" --math-data "$MATH" --needle-data "$NEEDLE" \
  --math-revision <math-revision> --needle-revision <needle-revision> \
  --candidate-commit <full-40-character-commit>
```

输出目录必须是 `/Data2/zincion/ai-infra` 下的新目录。工具记录模型、tokenizer 和两份语料每个文件的 SHA-256、revision、服务与客户端配置及脚本的哈希；哈希模型权重会读取完整快照。模型 revision 标记为 `UNVERIFIED_USER_DECLARATION`：文件哈希用于检查两次运行使用的内容相同，不证明其官方来源。准备时和后续检查均将实际 `src/**/*.py` 的路径与内容对照候选 commit 的 Git blob，并固定两个质量工具的实际 SHA-256；`src` 本身及其内部目录不能是符号链接，模型资产文件的符号链接仍按目标文件内容哈希；不要求用户的 `AGENTS.md`、`.codex` 等其他文件为干净状态。服务通过 `-P` 和固定 `PYTHONPATH` 加载已检查的 `src`。使用非默认 `--workspace-root` 仅用于本地临时工作区模拟，服务器实验仍须使用默认根目录。可通过 `--math-gate`、`--needle-gate` 在准备时分别固定最大精度下降，单位均为 0–1 的比例；例如 `0.01` 表示一个百分点。比较使用逐样本正确数和原始十进制门槛精确判定，恰好等于门槛视为通过。省略时对应任务为 capture-only，`gate_passed=null`，不表示通过。门槛、配置和样本在评测前固定，不能在看到结果后修改门槛再作为验收。

默认是两任务相同的 `temperature=0`、`top_p=1`、`max_tokens=1024`，用于在本服务支持的采样契约下减少对比中的随机性；这不等同于数据集或模型推荐的完整采样方案，结果不能直接当成外部排行榜口径。NIAH 使用长度 1024、7680、14336 与深度 0%、50%、100% 的英中网格。长度上限加 1024 token 的 chat 预算和生成预算严格小于配置的 `max_model_length=17408`；准备后的 `materialize.sh` 还会用服务环境的 tokenizer 和实际 `tokenize_request` 对每个原生样本应用 chat template，逐条检查实际 prompt 加回复预算。输入不截断；命中 `max_tokens` 的回复数量会在比较结果中单独记录。

如需分别设置回复上限，可在准备命令中添加 `--math-max-tokens 8192 --needle-max-tokens 1024`；未指定的任务继承 `--max-tokens`。新计划记录各任务的实际上限；使用独立参数时，版本 2 计划生成每个任务独立的 evalscope `TaskConfig` 和 `baseline/<task>`、`candidate/<task>` 工作目录，并用各自上限检查每条实际输入；原生 prediction 的输入 token 数须与物化结果一致，输出 token 数不得超过该任务上限。物化和执行各任务时均使用计划中的同一个 seed，比较结果按任务记录输出 token 总数和停止原因。仍须先物化、冻结 baseline，再执行 candidate；设置预算本身不代表质量门槛或实验总预算已获批准。省略两个独立参数时保留原版本 1 计划及目录；旧计划缺少任务上限字段时使用其共享 `max_tokens`，已冻结的旧计划继续使用准备时哈希固定的原工具核验。

NIAH 明确使用 `judge_strategy=rule`，即 evalscope 的归一化精确匹配；这不同于该数据集默认的 LLM judge，两种 KV 格式始终使用同一规则，不把待测模型作为 judge。math 使用原生 `mean_acc`。两边的 `use_cache` 均为 null、prediction/review/work_dir 完全分开；evalscope 的 `use_cache` 会恢复预测和 review，不能指向另一格式的结果。固定源码中 `generation_config.retries` 是总尝试次数，因此设为 1；SDK 的 `model_args.max_retries=0` 禁止 SDK 重试。配置也显式包含 `generation_config.max_retries=0`，但源码只读取 `retries`，不能依靠前者控制外层重试。不发送请求级 seed；数据生成 seed 位于 TaskConfig。

每次准备固定一个新的 run ID，baseline 和 candidate 使用各自的服务模型别名作为 HTTP `model`；evalscope 的 `model_id` 仍为 `Qwen3-14B`，用于原生文件和报告。项目服务严格校验请求模型，因此 baseline 继续占用端口且 candidate 启动失败时，candidate 别名会被 baseline 拒绝。比较器同时核对原生 prediction 的模型和 HTTP 响应模型；这不能证明任意忽略请求模型或伪造响应的外部服务实际使用了哪个配置。

在隔离客户端中安装固定版本并记录依赖，然后物化样本、先跑 baseline、保留 baseline 冻结文件，再运行 candidate。以下启动命令仅供已安排服务器 GPU 的实验使用：

```bash
source "$QUALITY_ROOT/env.sh"
uv venv "$QUALITY_ROOT/client"
uv pip install --python "$QUALITY_ROOT/client/bin/python" 'evalscope==1.5.1'
uv pip freeze --python "$QUALITY_ROOT/client/bin/python" > "$QUALITY_ROOT/client-requirements.txt"
bash "$QUALITY_ROOT/materialize.sh"

# 在独立终端运行，并等待服务就绪
CUDA_VISIBLE_DEVICES=0 bash "$QUALITY_ROOT/server-baseline.sh"
# 在客户端终端运行；成功后自动生成 baseline-cases.json
bash "$QUALITY_ROOT/evaluate-baseline.sh"

# 停止 baseline 服务后，在独立终端启动 candidate 并等待就绪
CUDA_VISIBLE_DEVICES=0 bash "$QUALITY_ROOT/server-candidate.sh"
bash "$QUALITY_ROOT/evaluate-candidate.sh"
bash "$QUALITY_ROOT/compare.sh"
```

`materialized.json` 保留原生输入、目标、subset、index、消息、实际 token 数和客户端 Python/包版本；baseline、candidate 与比较阶段都检查客户端环境是否改变。`baseline-cases.json` 在 candidate 前固定同一清单和 baseline 原始文件哈希。服务启动前、评测后、冻结和比较时均检查上述源码与工具指纹；服务脚本使用 `exec`，没有在服务进程退出后运行额外检查。比较要求两个结果逐条匹配物化记录、预测与 review 配对、无失败及缺失，并核对 report 的命名指标、各 subset 和实际样本数；仅固定 seed 或共享语料不能证明请求相同。NIAH 的顶层 `Report.score` 仅对应第一个 context/depth 格子，工具按逐样本分数计算全网格 count-weighted accuracy，并保存每个格子的 baseline、candidate 和下降值。比较报告保留原生 report/prediction/review、配置、资产和输入指纹；有门槛且下降超过门槛时退出码非零。报告中的各任务门槛结果只表示数值比较；模型来源未经核验且 M7 其他验收项不在此工具范围，`formal_m7_acceptance` 为 null。

该工具不能仅凭日志独立证明外部服务的运行状态或服务器验收。客户端依赖冻结、实际数据格式、端到端 evalscope 对接、BF16/FP8 质量和目标硬件均须在目标环境进一步验证；预设结果、模拟输入或 capture-only 不构成 M7 质量通过。
