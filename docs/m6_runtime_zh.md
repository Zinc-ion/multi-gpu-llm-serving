# M6 逐层交接

M6 新增 opt-in 的 TP=1 逐层交接。P 在实际 decoder layer 返回、KV 写入后记录 event，独立 copy stream 等待该 event，将有效 KV 搬入两个 pinned host slot；独立 socket 线程把分片直接发给 D。D 按自己的物理 block table 恢复，并在全部 H2D 完成后确认。默认 M5 B2 整块路径及其公开调用保持不变。

当前支持 P whole/eager → D eager/graph。启用流水的 P 使用 graph 或 chunked prefill 时在配置边界拒绝，不自动切换。TP>1、rank 对应传输、不同 TP 度重分片和 FP8 KV 尚不支持。目标 4090 的阈值、跨卡/NUMA 带宽和 goodput 尚未测量；双槽背压可阻塞生产者，不能据此宣称优化收益。

## 配置与客户端

P、D 启动时都显式增加 `--pipeline-frame-bytes F`，默认为 0（禁用）。F 不超过 worker 的 `max-transfer-bytes`，且须至少装下一个 token 的 K 和 V，即 `2 × kv_heads × head_dim × dtype_bytes`。P 的 F 不得超过 D 的上限。两个角色仍使用各自的进程、监听地址与原有模型配置。数据连接复用 D 的监听端口，独立于各自的串行控制连接。

```python
decode = await PDClient.open_d(decode_endpoint, lifetime=60)
prefill = None
try:
    prefill = await PDClient.prepare_p(
        prefill_endpoint, decode.session_id, request_id, prompt_token_ids, params,
        lifetime=60,
    )
    await PDClient.start_pipeline(prefill, decode)
    first = await prefill.next_token()  # 首 token 仅发布一次
    if not first["finished"]:
        await PDClient.handoff_pipeline(prefill, decode, await prefill.describe())
        # 按 M5 的 next_token 循环继续读取 decode。
finally:
    if prefill is not None:
        await prefill.close()
    await decode.close()
```

`start_pipeline` 先做 D `reserve_early`，再建立数据通道并启动 P。`handoff_pipeline` 等最终 copy ACK，然后 D `commit_early`、P `acknowledge_early`、D `activate`。两侧的关闭仍由调用方 `finally` 负责。已知首 token 必为终态的请求使用原有 `open_p`，不准备流水。

## 早期事务

早期事务只接受 TP=1、P whole/eager prefill。P 启用 chunked prefill 或未启用 eager 时 `PrefillCore.prepare` 显式拒绝；D 使用的 decode eager/graph 模式不受此限制。首 token 已知必为终态（`max_tokens=1` 或总长度只允许一个输出）的请求继续走原有整块路径，不预留早期交接。

1. `PrefillCore.prepare` 验证并建立事务，生成协议版本 3 的早期元数据；此时不加入 Scheduler。它只包含 prompt，完成 token 数和缓存进度均为 0，KV 描述按完整 prompt 计算。`describe_early` 返回副本。
2. `DecodeCore.reserve_early` 验证会话、模型和采样规格，预留完整输出预算，并在 D 自己的 block table 中分配 prompt 目标块。P 的 queued/transfer 状态也按 prompt 块数计入配额；早期事务与整块事务同时存在时，共同受 P 的物理块总数约束。P 的 `reserved_prompt_blocks` 仅在存在早期事务时表示这项配额。
3. P 的 worker 只有在独立数据连接与已 claim、未过期的 D session 及相同 transaction 绑定、D 返回 ready 后，才能调用 core `ready/start`。core 的直接调用者须提供同样保证；core 本身不建立网络。
4. `restore_region` 使用 D 已分配的 block table 接收一层连续的逻辑 token 区间。层号从 0 开始递增，每层的区间从 0 连续到 prompt 长度；重复、跳跃、乱序或字节数错误使事务失败并释放 D 的预留。每帧的内存顺序是 K、V，各为 `[token, kv_head, head_dim]`，不携带 P 或 D 的物理 block ID。连续物理 block 合并复制，单帧至多 F 字节；尾块只传有效 token。core 的无 receiver 用法保留同步 host copy，worker 的流水连接挂接双槽 receiver。
5. P 首 token 非终态时，D `commit_early` 验证普通版本 2 最终元数据与早期描述一致。最终提交与分片可按任意先后顺序到达；只有两者齐全后 D 才进入 `restored`，可调用已有的 `activate`。已提交的首 token 尚无 KV，D 从该位置开始 decode。首 token 为 EOS 时取消 D 预留，不提交。
6. core 准备、预留、提交、P 释放和激活的重复调用在相同 handle 和相同内容下返回原有幂等结果；网络连接只允许一次 open/prepare/start。相同连接中的重复 reserve/commit/ack/activate 幂等；重复或乱序分片拒绝，不重试。`acknowledge_early` 只释放已离开 Scheduler 的 P 请求；worker 还核验最终 copy ACK 已成功。

## Copy、ACK 与回收

P 仅在实际包含流水请求的 batch 为 decoder layers 临时挂 hook，默认 B2 不挂逐层 hook。每层记录 compute event，copy stream 等待后发起非阻塞 D2H；socket 线程等待 D2H completion event 后才读 slot。每层不在 owner 上同步 copy stream；整个 batch 在 runner 返回前 drain 所有在途 D2H，先于 sampling/postprocess。因此首 token EOS 立即释放源块后，网络持有的只剩独立 host 数据。各分片 ACK 不释放 P 的源块。

D 的非末帧 ACK 为 `copies_done=False`，表示该帧已校验并提交恢复，H2D 可以仍在途。复用对应 pinned slot 前必须等待其 H2D event；末帧执行 `finish` 等完两个 slot，并在 D compute stream 建立 event 依赖，才返回 `copies_done=True`。CPU write、socket drain 及非末帧 ACK 均不表示 GPU copy 完成。

数据线程故障只标记该请求；控制 EOF、cancel、正常 close RPC 和 deadline 可在网络线程/事件循环直接 abort sender、唤醒满槽等待，不依赖正在 forward 的 owner 消费取消命令。owner 在 runner drain 后顺序回收 P；D 清理也等 H2D 后归还 block。数据连接未完成即断开时，即使 attach 命令还排在 owner 队列，也会追加有序清理并关闭所属 D 控制连接。迟到帧只能作用于原连接绑定的事务，不能恢复新 session。

## 边界与计时

令事务上限 N、帧上限 F、header 上限 H。每端每事务恰好两个 F 字节 host slot，GPU 上为 pinned，总上限 `2NF` 字节、`2N` 个 slot。P 直接从 slot 发送，无整帧 immutable 副本；一个数据连接最多一帧等待 ACK。D 当前 `readexactly` 的 frame bytes 与复制过程另占至多两个 F 字节的瞬时普通 host 存储；asyncio reader 的固定流缓冲及单次 transport read、至多 H 字节 header 另计。slot 限额不是进程 RSS 限额。

P 最多 N+1 个入站 socket，另有最多 N 个出站数据 socket；D 启用流水时最多 3N+1 个入站 socket，涵盖控制、claim、数据与状态。owner 队列维持 `2 × max_socket_connections + 2` 的上限。数据 socket 请求 `SO_SNDBUF/SO_RCVBUF=F`；操作系统可取整/加倍，开启计时后 `pd_pipeline_connected` 保留 P 实际报告的值。不可把 F 当作内核 socket 内存实测值。

stats 给出当前 staging/pinned bytes、slot/busy 数量及配置上限；关闭 sender/receiver 的 timing 保留已完成 frames/bytes、发送端峰值占用和 socket 成功提交量。`payload_bytes` 是已获 ACK 的有效 KV 字节；`socket_submitted_bytes` 只累计成功返回的整帧 sendall。若某次 sendall 抛错，该帧可能已部分提交，用 `partial_send_unknown=True` 标记；已知整帧总数是下界，不能报告成精确的实际接收量。

计时开启时另记 `pd_layer_ready`、`pd_d2h_done`、`pd_region_send_start/socket_done/sent`、`pd_h2d_done`。GPU copy 记录来自真实 CUDA timing event，值相对本端 transfer origin。`monotonic_ns` 仅用于同进程内观察发送与首 token 的先后，不跨 P/D 相减。`handoff_pipeline` 的 `pd_pipeline_done.residual_wait_ms` 是首 token 后开始等待最终 copy ACK 的暴露等待，区别于 B2 完整 `transfer_ms`；commit/ACK/activate 时间另由 gateway handoff 计时覆盖。layer-ready 到下一 layer-ready 包含 CPU 空隙，不能直接解释为 kernel 与 memcpy 重叠；重叠结论必须另外读取实际 profiler 时间线。

CPU 测试覆盖早期预算、真实 TCP、错误帧、双槽背压、断连/取消/超时及真实 runner/model 数值。CPU 测试不能证明 DMA 生命周期或收益；本机真实 CUDA 验证范围如下，跨卡和目标服务器性能仍待验证。

## 本机 GPU 验证

2026-11-16，在 RTX 5080 Laptop GPU 上用本地随机初始化的两层 tiny Qwen3 checkpoint，先运行独立共置参考进程，再运行同卡的独立 P/D 进程。P whole/eager → D eager/graph 的 FP32、BF16 四组合均通过：greedy token 与 B2、共置参考一致；同前缀 logits 按预先固定的 FP32 `rtol=1e-4, atol=1e-5`、BF16 `rtol=.02, atol=.02` 比较通过；有效 KV 与 B2 逐值相同。D 捕获记录只有 decode step，没有重算 prompt。各路径沿用相同的 M5 tiny fixture 配置，这些比较验证 PD 等价，未作 HF 对照。

测试覆盖 block 边界和部分尾块、并发请求、首 token EOS、单输出和总长度限制。F=256 字节，每事务每端两个 pinned slot 共 512 字节；在 postprocess 之前直接查询实际 copy stream，确认 D2H 已完成，并使后续请求复用源物理块。长度为 5 的非终态请求实际传输 FP32 1280 字节、BF16 640 字节，仅含有效 token。计时文件保留实际 D2H/H2D CUDA event 和首层发送先于首 token 的同进程时间戳。

真实 TCP relay 在 D 已恢复首帧、返回 `copies_done=False` 后暂停 ACK，验证 GPU 双槽背压期间正常 close、cancel、数据 EOF、D close 和 timeout 的回收；另强制终止本次测试创建的 P 进程，再启动替代 P。每种情况存活 worker 均能释放本侧资源并完成后续流水请求。强制退出不等于存活进程可跳过 DMA drain。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/src" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --no-sync python -m pytest -q -s tests/test_pd_pipeline_gpu.py
```

测试自行生成小 checkpoint，不下载模型；输出目录包含两端 `p.jsonl/d.jsonl`、捕获 tensor、源 copy 检查及数值/故障摘要。日志中的已确认 payload 字节与 sendall 部分提交未知的语义仍遵守上文。

同机原始 Torch Profiler Chrome JSON 同时保留 CPU/CUDA 活动、decoder layer forward 和 `LayerSender.on_layer` 区间。24-token、两层、FP32 诊断中有 54 个实际 kernel 和 13 次 memcpy，其中独立 stream 上 4 次 pinned D2H 各 1536 字节，共 6144 字节；D2H 时长总和 1.614 微秒，与任何实际 kernel 的时间交集为 0。原日志保留 profiler 的 external-init callback 警告；GPU event 数量及预期字节数从原始 trace 核对。这个 tiny 请求没有观测到 GPU kernel 与 D2H 重叠；layer 区间中的 CPU launch 空隙不算重叠，也不以 profile 时间推算带宽或收益。

另一个有界诊断形状使用 2048-token prompt、两层、hidden_size=32、16 个 Q/KV head、head_dim=128、FP32 和 `scale=1.0`，让 `Attention.forward` 自行除以 `sqrt(head_dim)`。F=32 MiB，每端两个 pinned slot 共 64 MiB；每层有效 KV 为 32 MiB。预热后的源物理块分成两个连续 run，实际发送四帧、总计 64 MiB；trace 含 48 个 kernel、17 次 memcpy，其中 8 次 KV D2H。第 0 层 D2H 与第 1 层实际 kernel 的交集并集为 1358.682 微秒，与全部实际 kernel 的交集并集为 1364.343 微秒。该单请求证明这个形状下存在真实重叠；两槽背压和网络 ACK 等待仍会暴露，不能据此推算 14B、跨卡/NUMA、4090 或 goodput 收益。
