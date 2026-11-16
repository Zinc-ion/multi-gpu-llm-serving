# M7：FP8 KV 存储、attention 与 PD 传输

本单元为引擎增加可选的 FP8 KV 存储、读取和 PD 传输。默认不启用；模型权重、activation 和 logits 的 dtype 仍由原有 `dtype` 配置决定。FP8 KV 是有损量化，不能把它与 BF16 KV 的输出差异当作无损传输误差。本机 tiny 模型的功能验证不能代替目标服务器、14B、长上下文质量或性能验收。

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

Triton store 将对应 token/head 的数据及 scale 直接写入池；除法保留 FP32 subnormal。运行时 whole/chunked paged prefill 和 decode 均按 block table 读取 FP8 数据及相同物理位置的 scale，按需反量化。FP8 prefill 在 FP32 中做反量化及 attention 中间计算，decode 同样使用 FP32 中间值；输出仍使用 query 的 dtype。cache 分配前的模型 warmup 沿用原有的无 cache prefill 路径。不存在逐步整池反量化或整池复制。

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

M5 整块交接与 M6 逐层交接均可传输 FP8 KV；P、D 仍使用各自的 TP=1 引擎和物理 block table，经主存和 TCP 传递原始 data 与对应 scale，D 不重算 prompt 或重新量化。启动网关时核对模型与 cache 格式；D 在预留前核对版本、模型、字节序、形状和字节数。不匹配会被拒绝。

`auto` 的 model spec 和整块版本 2、早期版本 3 保持原格式。FP8 的 model spec 另有：

```json
{"kv_cache": {"dtype": "float8_e4m3fn", "scale_dtype": "float32", "scale_granularity": "token_head"}}
```

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
