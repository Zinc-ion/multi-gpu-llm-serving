# M5 运行说明

## 实例资源隔离

每个 `LLMEngine` 创建一个由系统分配端口的 PyTorch `TCPStore`，并在实例存活期间持有该 store。`127.0.0.1` 是各 rank 连接 store 的地址；当前 PyTorch 的服务端实测绑定通配地址 `::`，不能据此认为监听仅限 loopback。各 rank 使用同一实例的 store 初始化进程组；不同实例使用不同的 store。端口由 store 直接绑定，不存在先释放临时 socket 再绑定的窗口，也不创建 rendezvous 临时文件。

多 rank 实例另外由引擎创建一个系统分配名称的共享内存段，传给各 rank 附着。关闭时 runner 关闭自己的附着，引擎等待 worker 退出后关闭并 unlink 自己的共享内存；不会查找或删除固定名称 `myvllm`。`LLMEngine.exit()` 和 `ModelRunner.exit()` 可重复调用，显式关闭会注销该引擎的 atexit handler；现有服务和对比工具先 unregister 再关闭的调用仍可使用。关闭时计时文件写入失败仍会清理资源并传播错误。启动期间资源获取、runner、tokenizer 或 Scheduler 抛错时，引擎清理已创建的本实例资源并传播错误。

直接构造单 rank `ModelRunner(config, rank=0, event=[])` 时，runner 自行创建和释放 store；构造失败时也释放自己创建的进程组和 store，不销毁已有调用者的进程组。多 rank 的 store 与共享内存由 `LLMEngine` 管理。资源信息只在引擎内部传给 rank，不是用户配置字段。进程被强制终止时操作系统回收 TCP 监听端口；多 rank 的共享内存由 Python 的 resource tracker 在相关进程结束后清理，不能保证存活 worker 尚未退出时立即清理。

物理 GPU 由进程启动前的 `CUDA_VISIBLE_DEVICES` 限制；`ModelRunner` 的 rank 是可见设备中的逻辑编号，并通过 `torch.cuda.set_device(rank)` 选择设备。例如设置 `CUDA_VISIBLE_DEVICES=3` 时，实例内部 rank 0 使用可见的逻辑 `cuda:0`，即环境变量指定的设备 3。实验前应核对 CUDA 设备顺序与 `nvidia-smi` 或 GPU UUID 的对应关系。P 和 D 是各自的 TP=1 实例，各自内部 rank 均为 0。

CPU 测试使用真实引擎、进程组和共享内存，以 Gloo 及 CPU 执行替代 NCCL 和模型计算。本机 RTX 5080 Laptop GPU 上使用随机初始化的 tiny Qwen3 FP32 checkpoint，验证了同一 GPU 的两个真实 TP=1 实例并存、各生成两个 greedy token、关闭一个后另一个仍能生成并关闭：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m pytest -q -s \
  tests/test_instance_resources.py::test_real_gpu_instances_can_initialize_generate_and_close_independently
```

该测试不代表多卡 TP 正确性、服务器验收或 PD/KV 传输已经完成。

## PD 核心与交接协议

`myvllm.pd.PrefillCore` 和 `DecodeCore` 分别独占一个已创建的 TP=1 `LLMEngine` 的 Scheduler 和 KV pool；创建时要求 Scheduler 为空。核心不创建、关闭引擎，也不提供网络服务。所有操作须在各自引擎的唯一所有者线程中串行执行。`ModelRunner.kv_cache` 指向模型各层使用的同一 KV pool。

构造时显式提供 `max_open_requests` 和 `max_transfer_bytes`。每条连接对应一个 D handle，`DecodeCore.open(deadline=None)` 发放随机 `session_id`；P 用该 ID 调用 `PrefillCore.open(session_id, request_id, prompt_token_ids, params, deadline=None)`，再发放独立的随机 `transaction_id`。`request_id` 仅用于外部关联，允许不同连接重复。`deadline` 是各自进程中的 monotonic 绝对时间，不能跨进程直接比较；循环调用 `expire()` 或 `expire(now)` 返回新过期的 handle，客户端须在失败时关闭两侧连接，网络层按各自 owner 顺序执行清理。

典型调用顺序如下，其中 P 与 D 各在自己的所有者线程中运行；具体消息传输由下文的 TCP worker 实现：

1. D `open()`，将 `session_id` 交给 P；P `open()` 后按需调用 `step()`。返回的 `TokenEvent` 包含 handle、token、是否结束、结束原因及 prompt/completion 计数。P 首 token 先向客户端发送一次。EOS、输出数或总长度在首 token 即满足上限时，事件已标记结束且 P KV 已释放，无须交接。
2. 对未结束请求，P 的 `describe(handle)` 返回可 JSON 序列化的轻量元数据，此时不分配和读取 payload；D `reserve(handle, metadata)` 校验并预留容量。`CapacityError` 明确拒绝该请求，不自动重试。
3. P `export(handle)` 返回真实主存 `bytes`；D `restore(handle, payload)` 完成复制后才返回，随后 P `acknowledge(handle)` 释放 P block 和缓冲，D `activate(handle)` 开始解码。D `step()` 的事件只包含第二个及之后的 token。P/D 不共享 Sequence、seq_id 或物理 block。
4. 两侧 `cancel(handle)`、`close(handle)` 和 `shutdown()` 在 step/copy 边界执行，释放本侧资源；`shutdown()` 不关闭引擎。网络层在同步复制返回后、激活或下一次 decode 前处理已排队的取消并调用 `expire()`。正常完成、拒绝、异常、连接断开或超时后，客户端负责关闭两侧连接，各 worker 依次清理对应 handle。关闭后旧 handle 的交接/激活操作被拒绝；每条连接只能创建并使用其自己的 handle，不能把迟到消息当成新连接的 `open`。

同一连接内，相同的 P `open`、D `reserve`/`restore`/`activate`、P `export`/`acknowledge` 不会再次执行生成或释放；改变已有事务的 metadata 会被拒绝。重复 `restore` 仅检查 bytes 类型和长度，保留首次成功恢复的结果，不重新复制或校验内容。ACK 后不再保留 payload，迟到的 `export` 不会重新生成它。已取消或过期的 handle 不可重新激活；终态只保留到连接调用 `close`，关闭会清除 Sequence 和 metadata 引用。`stats()` 可观察打开的 handle、预留传输字节、free/used block；P 另报 held 请求和实际缓冲字节，D 另报预留 block、resident 数及上限。

元数据包括协议版本、session/transaction/request ID、P 本地 seq_id、完整 prompt 和首 token、原始 prompt 长度、cached token 数、下一个 position、completion 计数、采样及停止参数。模型名、固定 revision、tokenizer revision、模型结构、EOS 和规范 dtype 均须与 D 配置一致；当前模型 revision 必须显式提供。KV 标明 dtype、主机字节序、精确 shape 和字节数；D 在分配物理 block 前检查这些信息、请求长度及 payload 长度。

P 在首 token 采样后仅有 prompt 的 KV：`cached_tokens = next_position = N`，`completion_count = 1`。传输形状为 `[2, layers, ceil(N / block_size), block_size, kv_heads, head_dim]`，依据 cached 数取块，不依据首 token 加入后的 `seq.num_blocks`。仅复制有效位置，末块无效尾部为零。bytes 为本机原生字节序，不兼容的字节序在接收端拒绝。

P 的 D2H 和 D 的 H2D 使用同步复制；返回 bytes 或 restore 成功表示相应复制已完成。D 在验证 payload 后分配自己的 block，以原始 prompt 创建 Sequence、追加首 token，设 cached 为 N，并从位置 N 开始处理该 token；正常路径不重算 prompt。导出期间临时 host tensor 与输出 bytes 可同时存在，导入期间输入 bytes 与临时 host 存储可同时存在，数据大小可能达到预留字节的两倍；PyTorch 内部、调用者及网络层还可能持有额外副本，须另行计入并限制，不能把该上限直接当作进程峰值主存上限。

D 按 `ceil(min(prompt_length + max_tokens, max_model_length) / block_size)` 为每个已接纳请求预留完整逻辑 block 配额，同时限制 reserved/restored/active 请求数不超过 `min(max_num_sequences, max_num_batched_tokens)`。配额覆盖后续 decode 的物理分配，确保核心独占 Scheduler 时不会因普通资源压力抢占并重算 prompt。P 的 held 请求移出 Scheduler 但继续持有 block；若全部运行请求都已移出且 waiting 队首因这些 block 暂不能分配，P `step()` 返回空事件等待 ACK/取消，不触发原有无进展异常。

CPU 专项使用真实 `LLMEngine.step`、`Sequence`、Scheduler 和 BlockManager，覆盖块边界、跨块 decode、多请求、容量、取消、超时与重复消息。另用真实 tiny Qwen3 计算、CPU attention 替代 GPU kernel，对照 FP32/BF16 的 greedy token 和 logits。真实 GPU 上的 D2H/H2D、跨进程 TCP 传输、CUDA Graph 和代表性故障回收已在本地 tiny 模型上验证；服务器性能与 HTTP 入口仍待后续验证或实现。

## PD 采样随机流

P、D 必须分别在各自进程中独占一个引擎。模型权重、warmup 和 KV 初始化完成后，构造空引擎的 PD core 时仅设置一次采样种子：P 使用配置 `seed`（缺省为 0），D 使用 `seed ^ 0x80000000`；配置种子须为 0 到 `2**32 - 1` 的整数。`torch.manual_seed` 会重设进程内 PyTorch 的 CPU 和所有可见 CUDA 设备的全局随机状态，因此同进程其他随机操作也会影响该流。请求的 open、交接和后续 decode 不重设种子；不同请求按实际批处理和采样顺序共同推进实例流。

协议版本 2 的顶层 `sampling_seed` 记录 P 的实际采样种子，D 在预留资源前拒绝与自身实际采样种子相同的交接；协议版本 1 不兼容。P、D 的 `stats()` 分别提供本实例的 `sampling_seed`，便于核对独立配置。该策略使 P 和 D 使用不同的随机种子，不保证正温度采样与共置引擎逐 token 相同；随机流也受请求顺序、批次、设备和实现影响。

CPU 回归使用独立进程、真实 tiny Qwen3 和采样器，attention 使用 CPU 替身且采样强制 eager，验证初始化、并发及后续请求的随机流推进和种子碰撞回收。真实 GPU 测试另验证了正温度请求及 P/D 采样 RNG 状态不同；不承诺与共置结果逐 token 一致。

## TCP worker 与异步客户端

本单元新增 TP=1 的 P、D worker 和异步客户端。P、D 各自由一个线程拥有独立 `LLMEngine`，经普通主存和 TCP 传递 KV；本单元不提供 HTTP 网关或动态路由。worker 协议面向**可信的内部网关和 worker**，应部署在受控网络中；它没有认证、加密或完整性签名。session claim 用于约束请求生命周期，不是鉴权。

## 启动与配置

P 与 D 使用各自的配置、监听端口和独立进程，`world_size` 必须为 1。实际 GPU 由进程的 `CUDA_VISIBLE_DEVICES` 指定；单卡进程内部使用逻辑设备 0、rank 0。两个实例可以独立启动，P 启动时不要求 D 已在线；P 的固定 D 端点不可用时，该请求的 claim 失败。

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m myvllm.pd.worker \
  --role decode --config /path/to/decode.json --host 127.0.0.1 --port 9001 \
  --max-connections 16 --max-transfer-bytes 1073741824 --request-timeout 120

CUDA_VISIBLE_DEVICES=0 uv run python -m myvllm.pd.worker \
  --role prefill --config /path/to/prefill.json --host 127.0.0.1 --port 9000 \
  --decode-host 127.0.0.1 --decode-port 9001 \
  --max-connections 16 --max-transfer-bytes 1073741824 --request-timeout 120
```

`max-transfer-bytes` 的值需按实际 KV 大小和并发计算，上例只是说明启动参数，不是 14B 容量保证。`--max-header-bytes` 可选，默认为 1 MiB；本客户端发出和接收的 JSON header 也以 1 MiB 为上限。请求中的 `max_tokens` 可以大于剩余总长度，最终由 `max_model_length` 终结。

P、D 使用一致的模型、tokenizer revision、结构、dtype、EOS 和总长度配置。每个实例在 core 初始化时只设一次采样 RNG：P 的有效 seed 是配置的 32 位 base seed，D 为 `base XOR 0x80000000`。D 拒绝与自身有效 seed 相同的 P seed。固定多对实例应给各对使用不同 base seed，同一对 P/D 使用相同 base seed；非 greedy 的输出不承诺与共置实例逐 token 相同。

公开的 `PDWorker.start()` 与 `close()` 并发时，`close()` 等待当前启动过程结束，再关闭监听与 owner；返回后不会留下无 owner 的监听端口。关闭后仍可再次调用 `start()`。

## 客户端契约

`from myvllm.pd.client import PDClient, PDRemoteError`；端点是 `(host, port)`。`await PDClient.info(endpoint)` 通过独立连接返回：

- `role`、`model`（模型名和 revision、tokenizer revision、结构、dtype、EOS）、`max_model_length` 与有效 `sampling_seed`。
- `max_connections`、`max_transfer_bytes`、`max_header_bytes`、`request_timeout`、`max_socket_connections`。`max_completion_tokens` 仅是 `max_model_length - 1` 的推导值，不是请求 `max_tokens` 的额外上限。
- `decode_endpoint`：P 的固定 D 地址，JSON 返回两元素列表；D 为 `null`。

网关启动时应核对自身 tokenizer/模型配置及长度、各 worker 的 `model`、role 和有效 seed，并核对 P 的 `decode_endpoint` 指向所配置的 D。首 token 即结束的请求不会走 D reserve，不能只依赖 reserve 检查模型配置。

```python
from myvllm.pd.client import PDClient
from myvllm.sampling_parameters import SamplingParams

decode = await PDClient.open_d(decode_endpoint, lifetime=60)
prefill = None
try:
    prefill = await PDClient.open_p(
        prefill_endpoint, decode.session_id, request_id, prompt_token_ids,
        SamplingParams(temperature=0, max_tokens=32, ignore_eos=True),
        lifetime=60,
    )
    event = await prefill.next_token()
    # 发布 event["token"] 一次；finished 时无需传输。
    if not event["finished"]:
        metadata = await prefill.describe()
        await PDClient.handoff(prefill, decode, metadata)
        while True:
            event = await decode.next_token()
            # 发布 event["token"] 一次。
            if event["finished"]:
                break
finally:
    if prefill is not None:
        await prefill.close()
    await decode.close()
```

P 产生首 token，随后 D 从该 token 的位置开始 decode。交接元数据包含 D session、P transaction ID、请求 ID、prompt 与首 token、采样设置、已缓存 token 数和 KV 描述。P 输出的首 token 已计入 completion，但自身尚无 KV；传输的有效 KV 覆盖 prompt。D 先预留完整输出预算所需 block，再分配自己的 block table、复制 KV 并激活。最后不满 block 的填充不作为有效 token；P 和 D 的物理 block ID 不可互换。

`handoff()` 依次执行 D `reserve(metadata)`、P `export()`、D `restore(payload)`、P `acknowledge()` 和 D `activate()`。这些方法也可逐个调用；重复 reserve、restore、ack 和 activate 不会再次复制或释放，重复 restore 只检查字节数，不承担校验数据内容的职责。`next_token()` 返回 `token`、`finished`、`finish_reason`、`prompt_tokens` 和 `completion_tokens`。异常交接不会自动关闭另一侧：调用方必须在 `finally` 中关闭两个连接。

D 为每个连接创建 session；P 入队前向固定 D 原子 claim 此活跃 session，D 首次 reserve 再绑定 metadata 中的 transaction ID。已 claim、已取消、过期或已关闭的 session 不可再次 claim；D 重启后的旧 session 也不能重用。P claim 后如果入队或后续交接失败，由调用方关闭 D 或等待其生命周期超时。协议不重试，也不保存永久事务历史。

同一 `PDConnection` 的 RPC 串行执行。没有其他 RPC 在途时，`await connection.cancel()` 返回服务端取消结果；若已有操作在途，它立即抛 `RuntimeError`。要中断正在等待的 `next_token()` 或传输，直接 `await connection.close()`，并收集原任务的连接异常；关闭以 socket EOF 触发 owner 顺序清理，不等待一个正在执行的 GPU kernel 或 copy 被强行中断。连接空闲时的正常 `close()` 等待服务端关闭响应。异常、超时和中途关闭会 abort 本端发送缓冲；正常 socket 收尾有该连接 deadline，不会无限等待慢对端。

## 容量与生命周期

令每个 worker 的 `max_connections=N`、`max_transfer_bytes=B`、header 限额为 `H`：

- 事务连接最多 N 个；P 最多 `N+1` 个 socket，D 最多 `2N+1` 个 socket，容纳 N 个活跃 D session、最多 N 个并发 P claim 和一个状态连接。owner 命令队列上限为 `2 * max_socket_connections + 2`。这些是总量界限，不为恶意连接提供优先级或防护。
- core 的 P 或 D 已预留传输字节总和不超过 B，单个 payload 也不超过 B；消息头和声明的 body 长度在读取 body 前检查。每个事务连接同一时间最多有一个在途命令和 body；P 的导出结果、D 的输入及复制临时对象、socket 写缓冲可能同时占用多份数据，因此 B 不是整个进程的 RSS 上限。按最多 N 个事务、每个待处理 body 不超过 B、每个 socket 的 header 不超过 H，用户态传输存储为有限的 `O(N × B + (2N+1) × H)`，另加 owner 中一次同步 copy 的临时数据、固定流缓冲和操作系统 socket 缓冲。不要按 B 单独估算主存。
- 待发 token 队列按 `min(max_tokens, max_model_length - prompt_length)` 有界；D 不重复缓存 P 已发送的首 token。core 执行/copy 完成前，网络断开不会提前归还相关资源。关闭期间事务许可保持到 owner 清理及本端 socket 关闭或 abort 完成；错误、超时和 worker 关闭丢弃未发送的缓冲。
- `lifetime` 是从连接接受开始的单一请求生命周期，覆盖排队、prefill、claim、交接和 decode，不得超过 worker 的 `request_timeout`。客户端从连接前开始计算自己的 deadline，因此可能略早于服务端超时。超时在同步 GPU 操作之间处理，不强行中断已经开始的 kernel 或 copy。

`await PDClient.stats(endpoint)` 不改变请求生命周期，可检查 `open_handles`、KV block、传输预留、排队事件与 D resident 等计数。无效协议消息关闭该连接并执行清理；core 的非法重复 metadata 在拒绝时保持原有有效事务，经过网络连接时则按违规连接关闭和回收。用户可重复使用外部 request ID，session/transaction 才区分交接。

## 计时

配置可选 `timing_path`，复用已有 JSONL `write_record`；各进程使用独立文件。worker 记录 `pd_claim_done`、`pd_admitted`、`pd_first`、`pd_reserved`、`pd_export_done`、`pd_restore_done`、`pd_p_released`、`pd_second`、`pd_finished`、`pd_failed`、`pd_cancelled` 和 `pd_released`；记录包含可用的 request/session/transaction ID。

- `claim_ms` 从 P 开始连接固定 D 到 claim 响应及该控制连接收尾；`queue_ms` 从 P 接受事务连接到 P owner 完成入队，包含 claim 和排队。
- `export_ms` 围绕 P 同步导出，`restore_ms` 围绕 D 分配及同步恢复；完成时间在对应阻塞 copy 返回之后，`payload_bytes` 包含整块传输的末尾填充。
- `PDClient.handoff(..., timing_path=...)` 的 `pd_transfer_done.transfer_ms` 从 D reserve 响应之后开始，到 D restore 响应为止，包含 P export、socket、客户端转发和 D owner 排队/恢复；不含 D reserve、P acknowledge 或 D activate。P 首 token 和 D 第二 token 的 `elapsed_ms` 分别相对各自的连接接受时间，不可跨主机时钟直接相减。

正常 decode 没有为捕获 logits 增加同步。测试专用 engine hook 捕获 logits、有效 KV 和 block table，不能作为生产性能测量。

## 本地验证范围

```bash
CUDA_VISIBLE_DEVICES='' uv run python -m pytest -q tests/test_pd_transport.py
CUDA_VISIBLE_DEVICES='' uv run python -m pytest -q
CUDA_VISIBLE_DEVICES=0 uv run python -m pytest -q -s tests/test_pd_gpu.py \
  --basetemp=/path/to/new-evidence-directory
```

GPU 测试使用生成的 tiny Qwen3 checkpoint，顺序运行共置参照，再启动同一物理 GPU 的独立 P/D 进程，经真实 TCP 与主存交接。固定 FP32 `rtol=1e-4, atol=1e-5`、BF16 `rtol=0.02, atol=0.02`，检查 greedy token 和有效 KV 精确相等；仅在相同前缀上比较 logits。测试目录保存原始捕获和逐 case 数值报告。测试覆盖 eager/graph、whole/chunked prefill、块边界、并发、EOS/ignore、总长度、断连和受控子进程退出。

本地同卡小模型结果仅证明该环境下的功能和生命周期，不能代表跨 GPU 或跨 NUMA 的传输带宽、14B 数值/容量、正式 B2 或 goodput。目标服务器的独立 GPU、14B 和性能验收仍待后续执行。
