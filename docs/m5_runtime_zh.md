# M5 运行说明

## 实例资源隔离

每个 `LLMEngine` 创建一个监听 `127.0.0.1` 的 PyTorch `TCPStore`，由系统分配端口，并在实例存活期间持有该 store。各 rank 使用同一实例的 store 初始化进程组；不同实例使用不同的 store。端口由 store 直接绑定，不存在先释放临时 socket 再绑定的窗口，也不创建 rendezvous 临时文件。

多 rank 实例另外由引擎创建一个系统分配名称的共享内存段，传给各 rank 附着。关闭时 runner 关闭自己的附着，引擎等待 worker 退出后关闭并 unlink 自己的共享内存；不会查找或删除固定名称 `myvllm`。`LLMEngine.exit()` 和 `ModelRunner.exit()` 可重复调用，显式关闭会注销该引擎的 atexit handler；现有服务和对比工具先 unregister 再关闭的调用仍可使用。启动期间 runner、tokenizer 或 Scheduler 抛错时，引擎清理已创建的本实例资源并传播错误。

直接构造单 rank `ModelRunner(config, rank=0, event=[])` 时，runner 自行创建和释放 store；多 rank 的 store 与共享内存由 `LLMEngine` 管理。资源信息只在引擎内部传给 rank，不是用户配置字段。进程被强制终止时操作系统回收 TCP 监听端口；多 rank 的共享内存由 Python 的 resource tracker 在相关进程结束后清理，不能保证存活 worker 尚未退出时立即清理。

物理 GPU 由进程启动前的 `CUDA_VISIBLE_DEVICES` 限制；`ModelRunner` 的 rank 是可见设备中的逻辑编号，并通过 `torch.cuda.set_device(rank)` 选择设备。例如设置 `CUDA_VISIBLE_DEVICES=3` 时，实例内部 rank 0 使用可见的逻辑 `cuda:0`，即环境变量指定的设备 3。实验前应核对 CUDA 设备顺序与 `nvidia-smi` 或 GPU UUID 的对应关系。P 和 D 是各自的 TP=1 实例，各自内部 rank 均为 0。

CPU 测试使用真实引擎、进程组和共享内存，以 Gloo 及 CPU 执行替代 NCCL 和模型计算。本机 RTX 5080 Laptop GPU 上使用随机初始化的 tiny Qwen3 FP32 checkpoint，验证了同一 GPU 的两个真实 TP=1 实例并存、各生成两个 greedy token、关闭一个后另一个仍能生成并关闭：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m pytest -q -s \
  tests/test_instance_resources.py::test_real_gpu_instances_can_initialize_generate_and_close_independently
```

该测试不代表多卡 TP 正确性、服务器验收或 PD/KV 传输已经完成。

## PD 核心与交接协议

`myvllm.pd.PrefillCore` 和 `DecodeCore` 分别独占一个已创建的 TP=1 `LLMEngine` 的 Scheduler 和 KV pool；创建时要求 Scheduler 为空。核心不创建、关闭引擎，也不提供网络服务。所有操作须在各自引擎的唯一所有者线程中串行执行。`ModelRunner.kv_cache` 指向模型各层使用的同一 KV pool。

构造时显式提供 `max_open_requests` 和 `max_transfer_bytes`。每条连接对应一个 D handle，`DecodeCore.open(deadline=None)` 发放随机 `session_id`；P 用该 ID 调用 `PrefillCore.open(session_id, request_id, prompt_token_ids, params, deadline=None)`，再发放独立的随机 `transaction_id`。`request_id` 仅用于外部关联，允许不同连接重复。`deadline` 是各自进程中的 monotonic 绝对时间，不能跨进程直接比较；循环调用 `expire()` 或 `expire(now)` 返回新过期的 handle，网络层负责把取消传播到对端。

典型调用顺序如下，其中 P 与 D 各在自己的所有者线程中运行；具体消息传输由后续服务层实现：

1. D `open()`，将 `session_id` 交给 P；P `open()` 后按需调用 `step()`。返回的 `TokenEvent` 包含 handle、token、是否结束、结束原因及 prompt/completion 计数。P 首 token 先向客户端发送一次。EOS、输出数或总长度在首 token 即满足上限时，事件已标记结束且 P KV 已释放，无须交接。
2. 对未结束请求，P 的 `describe(handle)` 返回可 JSON 序列化的轻量元数据，此时不分配和读取 payload；D `reserve(handle, metadata)` 校验并预留容量。`CapacityError` 明确拒绝该请求，不自动重试。
3. P `export(handle)` 返回真实主存 `bytes`；D `restore(handle, payload)` 完成复制后才返回，随后 P `acknowledge(handle)` 释放 P block 和缓冲，D `activate(handle)` 开始解码。D `step()` 的事件只包含第二个及之后的 token。P/D 不共享 Sequence、seq_id 或物理 block。
4. 两侧 `cancel(handle)`、`close(handle)` 和 `shutdown()` 在 step/copy 边界执行，释放本侧资源；`shutdown()` 不关闭引擎。网络层在同步复制返回后、激活或下一次 decode 前处理已排队的取消并调用 `expire()`。正常完成、拒绝、异常、连接断开或超时后，网络层都要关闭两侧相应 handle 并传播必要的对端取消。关闭后旧 handle 的交接/激活操作被拒绝；每条连接只能创建并使用其自己的 handle，不能把迟到消息当成新连接的 `open`。

同一连接内，相同的 P `open`、D `reserve`/`restore`/`activate`、P `export`/`acknowledge` 不会再次执行生成或释放；改变已有事务的 metadata 会被拒绝。重复 `restore` 仅检查 bytes 类型和长度，保留首次成功恢复的结果，不重新复制或校验内容。ACK 后不再保留 payload，迟到的 `export` 不会重新生成它。已取消或过期的 handle 不可重新激活；终态只保留到连接调用 `close`，关闭会清除 Sequence 和 metadata 引用。`stats()` 可观察打开的 handle、预留传输字节、free/used block；P 另报 held 请求和实际缓冲字节，D 另报预留 block、resident 数及上限。

元数据包括协议版本、session/transaction/request ID、P 本地 seq_id、完整 prompt 和首 token、原始 prompt 长度、cached token 数、下一个 position、completion 计数、采样及停止参数。模型名、固定 revision、tokenizer revision、模型结构、EOS 和规范 dtype 均须与 D 配置一致；当前模型 revision 必须显式提供。KV 标明 dtype、主机字节序、精确 shape 和字节数；D 在分配物理 block 前检查这些信息、请求长度及 payload 长度。

P 在首 token 采样后仅有 prompt 的 KV：`cached_tokens = next_position = N`，`completion_count = 1`。传输形状为 `[2, layers, ceil(N / block_size), block_size, kv_heads, head_dim]`，依据 cached 数取块，不依据首 token 加入后的 `seq.num_blocks`。仅复制有效位置，末块无效尾部为零。bytes 为本机原生字节序，不兼容的字节序在接收端拒绝。

P 的 D2H 和 D 的 H2D 使用同步复制；返回 bytes 或 restore 成功表示相应复制已完成。D 在验证 payload 后分配自己的 block，以原始 prompt 创建 Sequence、追加首 token，设 cached 为 N，并从位置 N 开始处理该 token；正常路径不重算 prompt。导出期间临时 host tensor 与输出 bytes 可同时存在，导入期间输入 bytes 与临时 host 存储可同时存在，数据大小可能达到预留字节的两倍；PyTorch 内部、调用者及网络层还可能持有额外副本，须另行计入并限制，不能把该上限直接当作进程峰值主存上限。

D 按 `ceil(min(prompt_length + max_tokens, max_model_length) / block_size)` 为每个已接纳请求预留完整逻辑 block 配额，同时限制 reserved/restored/active 请求数不超过 `min(max_num_sequences, max_num_batched_tokens)`。配额覆盖后续 decode 的物理分配，确保核心独占 Scheduler 时不会因普通资源压力抢占并重算 prompt。P 的 held 请求移出 Scheduler 但继续持有 block；若全部运行请求都已移出且 waiting 队首因这些 block 暂不能分配，P `step()` 返回空事件等待 ACK/取消，不触发原有无进展异常。

CPU 专项使用真实 `LLMEngine.step`、`Sequence`、Scheduler 和 BlockManager，覆盖块边界、跨块 decode、多请求、容量、取消、超时与重复消息。另用真实 tiny Qwen3 计算、CPU attention 替代 GPU kernel，对照 FP32/BF16 的 greedy token 和 logits。真实 GPU 上的 D2H/H2D、跨进程传输、CUDA Graph、故障回收及服务器性能仍待后续集成验证；本核心尚无 HTTP 或网络入口。
