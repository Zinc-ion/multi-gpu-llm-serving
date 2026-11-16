# M8：D 本地 prefill 与可选 D 端点

本节描述 PD worker 和可选负载路由；TP2 bulk 的目标 GPU 正确性及 M8 阶段验收尚未完成。固定 P/D 路径与协议版本 2、3、4、5 继续可用；完整交接及部署约束见 [M5](m5_runtime_zh.md)、[M6](m6_runtime_zh.md) 和 [M7](m7_runtime_zh.md)。

## TP 命令与生命周期

多 rank 引擎在 Linux 上用匿名 `memfd` 和共享映射保存至多 1 MiB 的单槽命令描述符；KV 数据不经过该槽。spawn 时通过标准库的 FD reduction 为每个 rank 传递句柄，不创建命名共享内存或命名信号量。每个 worker 的单向 pipe 只承载空的 ready 和 consumed 通知，rank 0 等所有 rank 消费旧槽后才能重用；worker 完成反序列化后、进入可能的 collective 前发送 consumed。超长描述符在写槽前拒绝，命令 ACK 和退出等待均有期限。消费确认和有限关闭借鉴 vLLM 0.15.0 的 `f176443446f659dbab5315e056e605d8984fd976`：`distributed/device_communicators/shm_broadcast.py:439–563` 与 `v1/executor/multiproc_executor.py:376–405`。

写槽前超限会抛出 `CommandSizeError`（`ValueError` 子类）；`LLMEngine.step()` 取消本轮已调度请求、释放其 block 和请求 ID 后原样抛出，未被该轮调度触碰的 waiting/running 请求保留。其他命令或执行错误不使用此回收分支。

worker 在进入 `ModelRunner` 前监视 spawn 提供的 rank 0 sentinel，使用 Linux `F_SETOWN`、`F_SETSIG` 和 `O_ASYNC`，父进程退出时由内核发送 `SIGKILL`；注册后还检查 sentinel 是否已经就绪，覆盖注册前父进程已退出的窗口。此信号针对父进程存活，不依赖创建 worker 的线程持续存活。依据为 CPython 3.11.16 的 `multiprocessing/spawn.py` 和 `popen_spawn_posix.py` 中 bootstrap pipe 的生命周期，以及 Linux v6.12 `fs/pipe.c`、`fs/fcntl.c` 中 pipe 异步通知和信号配置机制；v6.12 源码不是对目标服务器内核版本的声明，信号发送失败时该版本源码还有回退到 `SIGIO` 的路径。纯 Python 监视线程无法在持有 GIL 的 C 调用中保证运行；本机真实 checkpoint FIFO 和 CPU/Gloo 已覆盖这里的初始化阻塞。监视从 worker 目标函数开始，不覆盖此前的 spawn bootstrap；如果额外 fork 的进程继承并保留 sentinel 写端，父进程死亡通知会延迟，内核不可杀 I/O 也不能保证有限时间退出。

引擎退出时先记录实际 worker 的 PID 和退出码，再关闭已退出的 `Process`、pipe 与映射；结果分别保存在 `worker_pids` 和 `worker_exitcodes`。如果 worker 在 TERM/KILL 后仍存活，保留其进程对象和控制资源并报告清理错误。多次 `exit()` 安全；目标服务器 CUDA/NCCL、14B 及原 M4 BF16 TP2 严格 token 失败仍须独立验收。

## TP2 prefill bulk

PD worker 支持 `world_size=2` 的 prefill 和 `world_size=1` 的 decode，TP2 仅支持 bulk；TP2 early/pipeline、TP>1 decode 在入口拒绝。网关模型配置仍用 TP1 以描述全局架构和 tokenizer，load 模式可将同一个 P 配到多个允许的 D，并逐条检查采样 seed 不碰撞。TP2 P 的 KV head 数必须能被 2 整除；metadata 仍描述全部 head，D 按自己的 block table 恢复。P rank 按本地 cache 真实拷贝有效位置，尾块 data 补零、FP8 scale 补 1；独立 Gloo 组按 rank 收集原始 `uint8`，rank 0 依据全局 head 顺序合并 data 和 scale，不重新量化。该组只在 PD TP2 P 创建，默认 NCCL 组仍用于模型计算，组和命令均使用配置的 `tp_command_timeout_s`；1 MiB 槽只承载命令描述符。失败时不自动降级或重试。

启用 P `timing_path` 后，实际 `pd_export_done` 包含总 `payload_bytes`、`data_bytes`、`scale_bytes` 和 `tp_export`。后者的 `ranks` 分别记录按尾块填充的 `data_bytes`/`scale_bytes`、有效位置的 `valid_data_bytes`/`valid_scale_bytes` 和 `export_host_ms`；`gather_pack_host_ms` 包含 rank 0 等待 Gloo gather、合并和生成 bytes 的主机墙钟时间。这些值不是 PCIe CUDA event 时间，也不代表测得链路带宽。

内存预算可按单个填充后 payload `W`、其中 scale 字节 `S`、每 rank shard `A=W/2` 和 P 的聚合上限 `B=max_transfer_bytes` 估算。显式分配在 rank 0 包括本地 shard `A`、gather 列表 `W`、全局 pack `W`、不可变 payload bytes `W`、全局 scale 临时 `S` 和一个局部 scale 临时 `S/2`，另有至多 `B-W` 的其他已保留 payload；保守合计不超过 `B-W+A+3W+1.5S`，即不超过 `5B`。rank 1 的显式 shard 与 scale 临时至多 `A+S/2`。D 的一次 restore 同时持有 payload、`bytearray` 和 scale 临时，约 `2W+S`。这些只统计上述明确的 host 对象，不包括 Gloo、PyTorch allocator、TCP/asyncio 缓冲、网关 staging、模型和 GPU cache，也不是 RSS 实测或总 RSS 上限；并发逻辑预算仍由两端和网关分别执行。

CPU 测试通过真实两进程 Gloo、生产 runner 命令、core、worker、TCP 和网关检查 BF16/FP8 scale 的全头原始字节、乱序块表和尾块，以及 rank 退出后的有界失败和重启。CPU 模型替身不验证 CUDA D2H/NCCL、真实模型数值或性能。原 M4 BF16 TP2 strict token 失败保持 FAIL；目标服务器 14B、GPU 全头传输和冻结 logits gate 仍待验收。

## P 可选择 D

P 的 `decode_endpoint=(host, port)` 仍是默认目标。可通过构造参数 `allowed_decode_endpoints` 显式指定非空、无重复的端点列表；默认目标必须包含在列表中。省略列表时只允许默认目标。端点必须是非空 host 和 1–65535 的整数 port；D 不接受这两个 P 配置。命令行示例：

```bash
uv run --no-sync python -m myvllm.pd.worker \
  --role prefill --config /path/to/p.json --host 127.0.0.1 --port 9000 \
  --decode-host 127.0.0.1 --decode-port 9001 \
  --allowed-decode-endpoints '[["127.0.0.1",9001],["127.0.0.1",9002]]' \
  --max-connections 16 --max-transfer-bytes 1073741824 --request-timeout 120
```

`PDClient.open_p(...)` 和 `PDClient.prepare_p(...)` 可传 keyword 参数 `decode_endpoint=(host, port)`。省略参数时 wire header 保持原来的字段集合；指定后由 P 在连接前核对 allowlist，再向该 D claim。逐层发送的数据连接使用同一个选定端点。D 仍要求 session 存活、未被 claim，且在 reserve 时校验模型和采样种子；不会自动重试或迁移请求。allowlist 是启动时配置，不是网络鉴权。

`PDClient.info()` 在原有字段外返回：

- `allowed_decode_endpoints`：P 的允许列表；D 为 `null`。`decode_endpoint` 仍是 P 的默认目标。
- `enable_chunked_prefill`：实际 Scheduler 是否启用 chunked prefill；`local_prefill_enabled` 仅在 D 且已启用时为 true。
- `enforce_eager`、`world_size`；已有 `pipeline_frame_bytes` 为零时表示流水未启用。TP2 只用于 bulk prefill，decode 仍为 TP1。
- `max_transfer_bytes` 是该 worker 所有未完成交接的聚合逻辑预留上限；`max_payload_bytes` 是单个请求按 block 填充后的逻辑 payload 和 wire body 上限。两者也出现在事务 open 响应中。

JSON 中端点为 `[host, port]`。固定网关继续按固定 pair 工作；下述负载模式会核对配置 inventory 中每条允许的 P/D 路径。

worker 可通过 `--max-payload-bytes` 独立设置单请求上限，要求它为正整数且不超过 `--max-transfer-bytes`；省略时等于后者，保持旧配置的单包及聚合语义。bulk 和 pipeline 均在 P admission 与 D reserve 时检查完整、按 block 填充的 payload；pipeline 还要求每帧不超过 `pipeline_frame_bytes`，它自身不能大于 `max_payload_bytes`。因此显式配置较小的单包上限时，即使流水每帧都能放下，整次逻辑 payload 超限仍会被拒绝。

网关启动及 ready 按 worker 实际单包上限与网关 `max_payload_bytes` 比较；旧 worker 的 info/open 没有该字段时按其 `max_transfer_bytes` 作为真实 wire 上限。需先升级网关和 client：旧网关仍把 worker 聚合上限当作 wire 上限。bulk 网关先校验 metadata，再按其中的填充后 `kv.nbytes` 同步预留 staging，并在读取 export body 前核对响应长度；完成、失败和取消都释放对应预留。`max_staging_bytes` 是并发 bulk 交接的逻辑 payload 预算，pipeline 使用独立的有界帧缓冲；两者均不是 RSS 上限，host tensor、bytes、socket 缓冲和各进程副本也会占用内存。

## D 本地请求

D 配置 `enable_chunked_prefill=true` 后，可用原有 token 事件接口请求本地计算：

```python
local = await PDClient.open_local(
    decode_endpoint, request_id, prompt_token_ids, params, lifetime=60,
)
try:
    while True:
        event = await local.next_token()
        # 消费 event["token"] 一次；completion_tokens 从 1 开始。
        if event["finished"]:
            break
finally:
    await local.close()
```

本地请求校验与 P 相同的 prompt、采样和长度边界；D 未启用 chunked prefill 时直接拒绝。`lifetime`、取消、断连和 worker 停止与远端事务遵循相同的 owner 顺序清理。`next_token()` 可以在等待入调度器及部分 prefill chunk 期间调用；只有实际采样 token 才产生事件。首 token 以及后续 token 均由 D 采样；首 token 是 EOS、达到 `max_tokens=1` 或总长度上限时也正常完成。本地请求使用独立 session，没有 P claim、元数据、KV 导出/恢复或跨实例传输。

本地与远端 reserved、early-reserved、restored、active 请求共享 D 的 resident 和完整逻辑 block 预算，包括本地排队阶段。每个请求预留 `ceil(min(prompt_length + max_tokens, max_model_length) / block_size)`，在 Scheduler 入队或远端物理分配前拒绝超额请求。正常完成或取消等释放预算及物理块；本地的有界事件队列覆盖全部 completion token，远端队列只覆盖 P 首 token 之后的 token。chunked Scheduler 将本地 prefill 与已恢复远端 decode 放入同一 batch，远端 decode 从已恢复的位置继续，不重新计算其 prompt。

开启 D `timing_path` 后，本地请求的 `pd_admitted`、`pd_first`、`pd_second`、`pd_finished`、`pd_cancelled`、`pd_failed` 和 `pd_released` 等实际发生的记录包含 `route="local"` 与 `payload_bytes=0`；不产生伪造的交接或传输事件。这些 worker 记录使用本端时间，不能跨进程直接相减。

直接使用 `DecodeCore.expire()` 时，正常路径返回本轮新过期的 handle。接收端计时写入失败时，它先关闭本轮新过期的 handle，再抛出原始 `PipelineTimingError`；该异常是 `OSError` 子类，其 `__cause__` 保留底层文件 I/O 错误。异常路径不返回 handle 列表，调用方仍须关闭自己持有的 socket；core 不管理调用方的网络连接。尚未完成的 CUDA 复制仍须先完成，才能回收其使用的 block。

## owner 统计

`PDClient.stats()` 通过独立连接向 owner 查询，在原有字段上增加：

- `total_blocks`：物理块池大小；既有 `used_blocks` 和 `free_blocks` 是当前物理块数。
- P 的 `admitted_requests`：prepared、ready、queued、prefill、held、exported 请求数；`pending_prefill_tokens` 是尚未计算的 prompt token 数，部分 chunk 后按实际 cached 进度减少，held/exported 为零。`pending_prompt_blocks` 统计这些请求完整 prompt 所需的逻辑块；既有 `held_requests` 单独统计 held/exported。
- D 的 `reserved_blocks`、`resident_requests`、`resident_limit` 继续表示完整逻辑预算和使用量；`local_resident_requests` 与 `remote_resident_requests` 将当前占用 resident 配额的本地与远端请求分开。终态及尚未 reserve 的空 D session 不计入 resident。
- D 的 `decode_step_wall_ms`：最近至多 32 个合格 step 的最大 host wall 毫秒值。`decode_step_sample_count` 是窗口内样本数，`decode_step_sample_source` 为 `core.step_wall_decode_including_mixed`，`decode_step_sample_age_ms` 是该最大值样本在本 owner 的年龄。无样本时 wall 和年龄为 `null`、数量为 0。

只有实际产生 `completion_tokens > 1` 事件的 D step 进入该窗口，包括既有 decode 与本地 prefill 同批的 step；只有本地首 token 的 step 不计。计时包围实际 `core.step`，包含引擎已有的采样结果 D2H 完成，不为统计额外同步 GPU。窗口只按样本数有界，长时间无新样本时年龄会增长；调用方须自行设置过期和无样本策略。该值包含整批计算，但不包括两个 step 之间的 restore、控制、排队和网络等待，只能作为近期服务时间启发式，不能保证客户端 TPOT 或 SLO。当前计数与观测也不代替 owner 的实时 admission。

## 网关负载路由

不指定 `routing_mode` 时仍按配置中的独立固定 pair 轮转，保持原有启动和请求接口。显式设置 `"routing_mode": "load"` 后，`pairs` 中去重得到 P、D inventory；每个 P 的 allowlist 与 D inventory 的交集才是可选 edge。P 可以在多个 pair 中出现，P/D inventory 不能重叠。示例中的地址、容量和策略均需按部署填写：

```json
{
  "pairs": [
    {"prefill": ["127.0.0.1", 9100], "decode": ["127.0.0.1", 9200]},
    {"prefill": ["127.0.0.1", 9101], "decode": ["127.0.0.1", 9201]}
  ],
  "routing_mode": "load",
  "max_active_requests": 8,
  "max_payload_bytes": 3221225472,
  "max_staging_bytes": 6442450944,
  "max_request_bytes": 2097152,
  "request_timeout": 600,
  "poll_interval_s": 0.1,
  "poll_timeout_s": 1,
  "max_sample_age_s": 2,
  "elastic": {
    "enabled": true,
    "prefill_backlog_tokens": 1024,
    "tpot_budget_ms": 100,
    "tpot_margin_ms": 20,
    "step_sample_max_age_ms": 1000,
    "allow_idle_without_sample": false
  }
}
```

这些阈值仅用于说明配置格式，不是经测量或批准的 SLO。禁用本地分流可写 `"elastic": {"enabled": false}` 或省略该项。启用流水还需设置 `pipeline_min_tokens`，并在所有可达 edge 的两端启用匹配的 `pipeline_frame_bytes`；P 必须 whole/eager，D 可采用 eager 或 graph。

启动时网关逐个核对实际 worker 的角色、模型及 KV 格式、长度、TP、超时、连接数与协议限制，并对每条允许的 edge 检查 P/D 实际采样种子不能相等。load 模式不要求固定 pair 的 XOR 关系。worker 的 `info` 目前没有独立协议版本字段；上述检查依赖同一候选的 worker 实现，D 在实际 reserve 时仍校验 metadata 版本和格式，不能用此启动检查证明任意不同版本二进制兼容。`myvllm.pd.ready --workers-only` 对 load inventory 和所有允许的 edge 使用相同的 worker/edge 条件，但没有网关模型配置，不能替代网关自身的模型核对。

网关在每次选择前使用最近的 P、D owner 统计。统计请求有 `poll_timeout_s`，年龄从请求发出时开始保守计算，包含往返；查询失败会移除该样本，缺失或超过 `max_sample_age_s` 的 worker 暂不参与选路，没有合格路径时返回 503，不自动重试。P 按未计算 prompt token 和已接纳请求数排序；D 按完整输出预算的逻辑 block、resident 和连接上限筛选，再按 block 和 resident 排序。远端请求还检查两端现有传输预留加本请求按 block 填充的 KV payload 是否在各自上限内；单 token 终态路径不需要 KV 交接。

load 模式按实际 JSON/UTF-8 编码检查选路时已知的请求字段和控制帧长度下界，最终发送时再次检查所选 worker 的 `max_header_bytes`。实际 lifetime、P 序号和首 token 在选路时尚未确定，因此 header 余量极小时，最终帧仍可能超限，即使另一条 edge 有足够空间；此时不会自动迁移请求。部署时需为控制帧留出足够余量。

请求提交时同步预订 P token、D 逻辑 block/resident、两端传输字节及连接槽，且受网关活动请求数限制。每次成功采样只撤销该次轮询发出前已由 owner 接纳的对应 booking，之后的 booking 与样本相加；采样在途期间可能暂时重复计数，因此可能保守拒绝请求，后续样本会消除已覆盖的重复。外部客户端在采样后的新请求仍可能使估计落后；最终 block、连接和传输 admission 由 worker owner 决定，拒绝不会自动迁移。取消、失败或关闭时尚未消账的 booking 会释放。

本地分流还要求选定 P 的估计 backlog 达到 `prefill_backlog_tokens`，D 已启用 chunked prefill，且满足 D 的 block、resident 和连接预算。有新鲜 step 样本时，仅当 `decode_step_wall_ms + tpot_margin_ms <= tpot_budget_ms` 才可分流；无样本或 step 样本超龄时，仅在 `allow_idle_without_sample=true` 且估计 block/resident 均为零时允许。该策略不保证客户端 TPOT。决定记录包含 backlog、完整 block 需求、预留和连接估计、样本及 step 年龄、样本数量、idle 状态和策略参数；local 实际调用 D `open_local`，不创建 P 事务、不传输 KV，字节为零。

`tools/analyze_pd.py` 的 `outcome` 是网关生成结果，`frontend_outcome` 是网关前端观察到的完成或关闭；`worker_outcomes` 按提供的 P/D 记录分别归纳，缺少记录时为 unknown。它不凭网关记录推断客户端实际接收结果，`client_outcome` 为 null，应与独立客户端结果另行关联。local 和首 token 已结束的 bulk 请求实际传输字节为零；其余远端请求未观测到传输完成时保留未知。流水的 `payload_bytes` 是已确认字节；存在部分 sendall 未知、未确认的已提交字节或缺少发送端终态时，汇总的 `unknown_payload_requests` 包含该请求，不能把已确认字节当作完整传输量。

## 同候选四卡配置

`tools/prepare_pd.py --m8-comparison` 以 `--physical-gpus P0 P1 D0 D1` 明确指定同一组四张卡；GPU2 如使用，只能替换原四卡中的一张。除负载轮询、弹性策略和 `--chunk-tokens` 外，需显式提供 `--pipeline-min-tokens` 和 `--pipeline-frame-bytes`，以准备 bulk/pipeline 消融；这些值须先按目标硬件校准。

部署生成器可用 `--worker-max-connections`、`--worker-max-transfer-bytes`、`--worker-max-payload-bytes`、`--gateway-max-active-requests`、`--gateway-max-payload-bytes` 和 `--gateway-max-staging-bytes` 显式设定资源。默认分别为 4、3 GiB、3 GiB、8、3 GiB 和 6 GiB；生成器会检查 worker 单包不超过其聚合限额及网关单包限额，网关 staging 至少容纳一个网关最大 payload，所有 fixed/load/elastic 脚本及 manifest 使用同一组值。例如可分别设置 8、24 GiB、3 GiB、8、3 GiB、24 GiB；这只是逻辑预算配置，并不证明目标服务器的 GPU block、resident、RSS 或稳定并发容量。

`--m8-routing-transfer bulk` 是默认值，令 fixed/load/elastic 路由组均使用 bulk；选择 `pipeline` 时三组均使用相同流水阈值和帧上限。三组的 P 都为 whole/eager，D 都开启 chunked，使用相同 chunk budget 和 base config 中的 D eager/graph 模式；P、D 从 base 独立派生。生成的 `fixed-bulk-gateway.sh` 与 `fixed-pipeline-gateway.sh` 使用相同 fixed P/D worker 配置及各自网关计时文件，用于单独的机制比较。先为所选 D 模式完成实际 GPU 正确性验证，再执行任何性能比较。

同一输出目录另外生成四个 `whole-control-N` 与四个 `chunk-control-N` 单实例配置和启动脚本，逐卡独立配置、seed、端口和计时文件；各组以 manifest 中的四个 endpoint 组成一个目的地集合。manifest 同时记录物理 GPU 顺序、NUMA 映射、P/D 采样种子、策略与 chunk budget。各比较组复用相同物理端口和 GPU，按组依次启动；它们不是要同时启动的八个 GPU 实例。实际 GPU 空闲、NUMA 绑定、独立引擎资源及运行结果仍须在目标服务器核验。

显式追加 `--m8-tp2-bulk` 且使用 `--physical-gpus 0 1 3 4` 时，另生成 `tp2-bulk-*` 配置：一个 TP2 P 使用 GPU0/1，两个 TP1 D 使用 GPU3/4，D 的 chunk budget 与同一比较组一致；gateway 使用一个重复的 P endpoint 和两个 D endpoint，load 选路、禁用 elastic、仅 bulk。原 TP1 配置继续独立生成；请求计划、客户端、输出预算和四卡集合由同一比较实验指定。生成器仍要求为现有 M8 bulk/pipeline 消融显式提供 pipeline 参数，但 TP2 的 worker 和 gateway 配置不启用 pipeline。manifest 单独标识 TP2 配置、设备和 seed。

当前已完成真实 CPU worker/TCP/HTTP 的负载路由和资源回收测试；CPU 测试使用模型执行替身，不证明 CUDA Graph、DMA 或 14B 的正确性与性能。目标服务器 GPU、跨 NUMA、L3、TP2 和完整 M8 评测仍待验收。四卡 tiny smoke 入口为 `RUN_PD_TP2_GPU=1 python -m pytest -q -s tests/test_pd_tp_gpu.py`；运行前须由主 agent 调度 GPU0/1/3/4，且测试的 token 或 logits 差异会失败并保留报告。它不能取代原 M4 strict 测试及 14B gate。

## L2/L3 请求计划与公共流式客户端

`tools/prepare_trace.py`、`tools/replay_trace.py` 和 `tools/analyze_trace.py` 是新增的离线物化、公共 HTTP 回放和分析工具。它们读取 M2 已物化的 JSONL 和同名 `.manifest.json`，逐字节按 LF 分行，校验原文件 hash、行数和模型/tokenizer revision；已套模板的 prompt 与每行 `output_tokens` 原样保留，不再套 chat template。tokenizer 必须是本地文件，按 provenance 中 `tokenizer/` 下逐文件 SHA256 核对后，以 `local_files_only=True` 加载；输入 token 数通过该 tokenizer 的 `encode()` 实际计算。

例如，已确定本地路径和预算后，可用以下命令生成一份有限阶段计划。示例中的阶段定义只说明格式，速率、时长、并发和请求预算须按实际实验事先确定；它们不是已批准的服务器测量配置：

```json
[
  {"name": "chat-heavy", "duration_s": 120, "rate": 1, "chat_probability": 0.9},
  {"name": "long-heavy", "duration_s": 120, "rate": 1, "chat_probability": 0.1},
  {"name": "mixed", "duration_s": 120, "rate": 1, "chat_probability": 0.5}
]
```

```bash
uv run python tools/prepare_trace.py --mode l3 --kind formal \
  --chat "$M2_SHAREGPT_JSONL" --long "$M2_LONGBENCH_JSONL" \
  --tokenizer "$TOKENIZER_DIR" --tokenizer-provenance "$TOKENIZER_PROVENANCE_JSON" \
  --model-revision "$MODEL_REVISION" --tokenizer-revision "$TOKENIZER_REVISION" \
  --seed 7 --phases "$PHASES_JSON" --max-requests "$MAX_REQUESTS" --output "$PLAN_JSONL"
```

provenance JSON 的 `model_revision` 须匹配 tokenizer revision，`files` 按 `tokenizer/<相对路径>` 映射到带 `sha256` 的对象；该目录的文件集合及内容必须与记录一致。同一 `files` 中还须有每份使用的 `workloads/<JSONL文件名>` 和 `workloads/<JSONL文件名>.manifest.json`，两者的实际 hash 均须与记录一致。计划保存这些证据的绝对路径和 hash；回放与分析重新读取计划记录的来源 JSONL、manifest、provenance 和 tokenizer 文件，缺失或不匹配即拒绝。若使用证据副本，须在物化计划时显式传入副本路径，并在冻结后保持可读取。已有真实 provenance 没有 `scope` 字段时可继续使用；显式标为 `fixture` 的来源不能用于正式协议。

L2 使用 `--mode l2 --rate <每秒速率>`，并指定 `--count` 或 `--duration-s` 之一；有限窗口还须提供 `--max-requests`。只有一个来源时可只提供 `--chat` 或 `--long`；混合 L2 须同时指定两个来源及 `--chat-probability`。随机数生成器跨阶段使用同一 seed，输出绝对计划 offset、planned phase、唯一请求 ID、源文件/manifest hash、LF 行号与行 hash、实际输入 token 数和原输出预算。默认 ID 包含 `formal` 或 `warmup`；回放额外拒绝两份计划之间的 ID 交集。来源行用尽默认报错；显式传入 `--allow-recycle` 后，每个请求记录循环次数和使用序号。不同角色或路径引用相同 JSONL hash 时共享使用序号，不能绕过循环记录。

计划旁的 manifest 保存各阶段固定窗口、标称泊松速率、实际请求数和聊天比例，以及整份计划的输入 token 和最大输出 token 总数。有限窗口即使最后一次到达早于窗口结束，也保留原窗口时长。`--max-requests` 是物化上限，超过时不写计划，不会静默丢掉后续到达。warmup 用 `--kind warmup` 单独生成一份有界计划；正式计划用 `--kind formal`，两者及其 token 预算均可分别核对。

回放示例中的 endpoint 为 HTTP(S) 基础 URL，可以重复提供多个；一个计划按 ordinal 轮转目的地，所有目的地共用一个并发上限。`--warmup-plan` 或 `--no-warmup` 必须明确选择；warmup 有失败时正式请求标为未执行：

```bash
uv run python tools/replay_trace.py --plan "$PLAN_JSONL" --warmup-plan "$WARMUP_JSONL" \
  --endpoint "$ENDPOINT_1" --endpoint "$ENDPOINT_2" --model "$SERVED_MODEL" --ignore-eos true \
  --concurrency "$CONCURRENCY" --request-timeout-s "$REQUEST_TIMEOUT_S" \
  --batch-timeout-s "$BATCH_TIMEOUT_S" --max-raw-bytes "$MAX_RAW_BYTES" --gpu-count 4 \
  --output-dir "$RUN_DIR"
uv run python tools/analyze_trace.py --plan "$PLAN_JSONL" --run-dir "$RUN_DIR" --output "$ANALYSIS_JSON"
```

客户端使用绝对 monotonic deadline；等待并发槽导致的晚发不会平移后面的计划。单请求 timeout 从 dispatch 开始，整个 warmup 和正式回放共享有限 batch deadline，无隐式重试，也不跟随 HTTP 重定向；3xx 保留状态和响应体并记为失败。活动任务不超过全局并发上限。每条计划都有 success、failed 或 not_executed 结果；结果记录计划时间、dispatch、choice/finish/完成时间、HTTP 状态、usage 和错误。原始响应体以受控 ordinal 保存，已收到的 SSE 原字节、非 SSE 错误正文及失败时的 partial 会保留；HTTP chunk framing 由客户端库解码。同一次读取中先完成的合法事件会先记录，再处理后面的畸形字节或超限。`--max-raw-bytes` 是每个响应的硬存储上限：每次最多读 64 KiB、至多额外读取一个不保存的字节以区分恰好达到上限与超限；超限保存上限内前缀，先记录其中完成的事件，再将请求标为失败。结果中的 `raw_bytes` 是保存字节数，`received_body_bytes` 只计实际读取的 body 字节，`raw_truncated` 标明因超限截断，不代表未读取响应的完整长度。客户端禁用自动解压；若服务忽略 `Accept-Encoding: identity` 返回 gzip 等编码，则有界保留编码后的 body、记录 `content_encoding` 并失败，不把它解码成 SSE；此时 `.sse` 路径保存的是编码原字节。`run.json` 保存计划、工具、部署证据和 freeze hash、参数及 warmup/正式/合计预算；进程异常退出时可能只留下 `state=running` 和已落盘结果。batch deadline、warmup 失败或停止导致 `state=incomplete` 时回放 CLI 退出 1；分析 CLI 写出完整或 partial 报告后，对不完整批次或完整性校验问题退出 1。所有计划行均已执行但部分请求失败时 batch 可以完整结束，两个 CLI 可退出 0，失败仍留在分析分母。

流式解码支持拆分 UTF-8、CR/LF、注释、ping、多个 `data:` 行及一次读取中的多个 SSE 事件；同一次客户端读取中完成的事件使用同一接收时间。协议成功要求一致的 completion identity/object/model、单个 index 0 choice、唯一 `stop`/`length` finish、合法整数 usage、与本地 tokenizer 相同的 prompt token 数、正的 completion token 数、总数一致和 `[DONE]`。错误、截断、畸形、重复/冲突终态或缺字段均为失败；空文本和带 finish 的字段本身不证明事件是否包含 token。

默认没有事件/token 对应关系的证明，choice 接收时间仍保存，但首末输出时间和 TTFT/TPOT 为 `null`。可选 `--deployment`、`--protocol-contract` 仅在源码、部署配置和独立 probe 均已核验时启用规则。部署 JSON 的 `schema` 为 `m8-deployment-v1`，包含 `scope`、`endpoints`、`model`、`model_revision`、`tokenizer_revision`、`tokenizer_files_sha256`、`ignore_eos`、`generation_mode="standard_non_speculative"`、正整数 `stream_interval`，以及非空的 `source_files`、`config_files` 和 `runtime_provenance` 引用。文件引用格式为 `{"path": "...", "sha256": "...", "scope": "formal或fixture"}`，其中源码/配置引用还须有非空 `role`；相对路径相对引用它的 JSON 文件解析。tokenizer 文件集合 hash 是计划 manifest 中 `tokenizer.files` 对象按下述规范序列化后的 SHA256。

`runtime_provenance` 指向 `m8-runtime-provenance-v1` JSON，记录相同 `scope`、`endpoints`、`model`、`source_files`、`config_files`，非空 `observer` 和 `shared_implementation` 说明，以及 `processes` 列表。每项 process 包含所属 `endpoint`、正整数 `pid` 和非空 `command`；所有 endpoint 须被覆盖。`shared_implementation` 说明多副本共享源码与配置的实际依据，不能仅因版本号相同就认定运行源码相同。引用列表中的相对源码/配置路径仍相对部署 JSON 解析。

协议文件的 `schema` 为 `m8-protocol-v1`，以 `deployment_sha256` 绑定上述部署 JSON；`assertions` 精确包含布尔字段 `first_choice`、`last_choice`、`no_control_choices`、`usage_counts_special_tokens`。四项都允许为 false；它们是源码审阅与 probe 独立建立的陈述，false 表示不能据此启用相应计时规则，不使合法协议响应失败。`first_choice`/`last_choice` 分别证明首/末 choice 是对应输出事件，可只启用一端；两者均 false 时 formal 仍可计算全分母，成功请求的首末时间、TTFT/TPOT 为 `null`，SLO 为 indeterminate。仅当无控制 choice 且 usage 包含全部特殊 token 时才用 `choice 数 <= completion_tokens` 做必要一致性检查；该不等式从不证明事件含 token，违反时首末计时均为 `null`。`source_review` 引用 `m8-source-review-v1` JSON，记录相同 scope、deployment hash、assertions、`verified=true`、reviewer 及排序后的 `uncovered_cases`；`probe_manifest` 引用 `m8-protocol-probe-v1` JSON，也须记录相同 scope、deployment hash、assertions 和 `verified=true`。`cases` 必须包含 `ordinary`、`empty_first`、`empty_last`、`last_token_finish`、`single_token`、`eos`、`merged_choices`、`pure_terminal`、`error_and_missing`。每项包含相同 `scope`、布尔 `covered` 和 `samples`：覆盖时 samples 非空；确实无法构造的特殊 case 标 `covered=false`、`samples=[]` 并写非空 `reason`，独立 source review 须列出相应缺口。`ordinary` 须覆盖全部 endpoint，使用各自独立 raw 文件，且是通过 usage 和 DONE 校验的完整普通 SSE；其他 case 无须每个副本重复全部边界。

每个 sample 包含 `endpoint` 及 `request`、`raw_sse`、`independent_observation` 文件引用；endpoint 须属于部署。观测 JSON 的 `schema` 为 `m8-probe-observation-v1`，包含相同 `scope`、`case`、`endpoint`、`deployment_sha256`、`request_sha256`、`raw_sse_sha256`，非空 `observer`、`method` 及非空对象列表 `observations`。probe 请求使用与回放相同的受限 completions 字段；EOS probe 须设置 `ignore_eos=false`，其余 probe 须与回放一致。除错误/缺失场景外，raw 须包含该 model 的 completion SSE。普通样本不会自动生成首末事件 assertions；probe 和独立源码审阅需要真实对应部署，少数字符串或 `choice 数 <= usage` 不能代替证明。

`scope="fixture"` 仅用于有明确标记的测试夹具，分析结果标为 fixture；引用、runtime、源码审阅、probe case/观测及批准记录的 scope 必须与外层一致。`scope="formal"` 的协议文件另外要求 `approval` 引用。批准文件是 `m8-approval-v1` JSON，包含相同 `scope`、`status="approved"`、非空 `approver`、带时区的 `approved_at` 和 `subject_sha256`；后者是被批准 JSON **去掉 `approval` 字段后**的规范序列化 SHA256。规范序列化为 Python `json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"` 的 UTF-8 字节。批准时间不能晚于回放开始，工具校验文件与绑定 hash；批准者身份、实际运行进程源码和 probe 事实仍须由主 agent 独立核对，hash 本身不是电子签名。

正式 SLO 可用 `--freeze` 指定 `m8-slo-v1` JSON：`units="ms"`，`ttft_ms_by_prompt_tokens` 为递增的 `{"max_prompt_tokens": 整数或null, "ttft_ms": 数值}` 列表，最后一项上限为 `null`；另提供正数 `tpot_ms`、`single_token_policy="ttft_only"`、非空 `calibration_files` 引用和上述 `approval`。正式 freeze 的 `binding` 还须绑定模型与 tokenizer revision、`tokenizer_files_sha256`、`deployment_sha256`、`replay_tool_sha256`、`analyze_tool_sha256`、`concurrency`、`request_timeout_s`、`batch_timeout_s`、`max_raw_bytes`、`gpu_count`、`ignore_eos`。`plans` 每项提供 `plan_sha256`、`plan_manifest_sha256`、`warmup_plan_sha256`、`warmup_plan_manifest_sha256`；明确无 warmup 时后两项均为 `null`。回放和分析核对实际计划、部署、工具与设置，变更会拒绝正式判定。当前没有已批准的阈值或目标部署 probe，因此默认只能采集，不报告正式 SLO 或容量。

分析以计划全量为分母，检查结果中的缺失/重复 ID、计划 hash 和未完成 batch；按 planned phase 和固定窗口计算 offered/passing rate，跨窗口完成尾巴单列。`mode="formal"` 表示依据所给人工批准文件进行正式门槛计算；`formal_contract_checks_passed` 只表示自动检查的结构、绑定和批次完整性通过。结果另有 `evidence_authenticity="requires_primary_agent_verification"`、`stage_acceptance=null`，由主 agent 核验实际来源、原始 SSE 和批准的真实性，阶段验收不由脚本决定。正式 TTFT 从 planned deadline 起算，dispatch TTFT 只用于诊断。usage 已证明包含特殊 token 时，合法单 token 请求的 TPOT 为 `null`，冻结政策只检查可识别的 TTFT；多 token 仅有一个可识别输出事件时 TPOT 仍为 `null`、SLO 不可判定。多事件 TPOT 要求首末两个不同事件均有独立映射证明且 usage 包含全部特殊 token，取首末客户端事件时间差除以 `completion_tokens-1`。允许控制 choice 时仍可用独立已证明的两端和完整 usage 估计 TPOT，但不报告相邻 choice 的 chunk ITL；仅证明首端则只保留 TTFT。usage 未证明包含特殊 token 时，即使首末可识别也不能用该计数估计 TPOT，报告的 1 也不能证明实际仅输出一个 token，因此不能套用单 token TTFT-only 通过政策。chunk ITL 只在首末均可识别且无控制 choice 时报告为输出事件间隔；合并、缓存与网络带来的偏差目前未知，不能当作精确逐 token 时间。每次报告 P50/P90/P99 和样本数，失败及不可判定请求不从计划分母删除；单次 L3 不能称为稳态最大 goodput，五次 P99 的平均值也不是稳定 P99。

## 独立 PD 数值捕获

`tools/compare_pd.py` 供正确性诊断，reference 和 D 要求 `world_size=1`，bulk P 可为 TP1 或 TP2；TP2 P 的 `--prefill-gpu` 显式传两个不同的物理 GPU index（如 `0,1`）或两个 UUID，且不能使用 `--raw-kv`。TP2 的 rank 0 cache 快照仅包含局部 head，不作为全局 KV；全头字节通过上述交接测试单独核对。实际 engine 从指定 Git checkout 导入并核对完整 SHA，每个模型进程分别记录导入路径、被跟踪源码哈希、源码状态、实际 KV/scale 格式和输入及运行时配置。harness 的路径和哈希另记。三个实例必须使用相同模型、架构、精度、EOS 和总长度。各模型 step 显式在 `torch.inference_mode()` 下执行，采样保持 greedy。reference 每例完成后保存，且进程已退出后才启动 D、P。

每例保留原始 `prompt` 和 `input_ids`，模型输入始终以原 ID 为准（token IDs authoritative），不重新编码替换。三方实际 tokenizer 的身份、`encode(prompt)` 和 `decode(input_ids)` 观测必须一致。若 `encode(prompt)==input_ids`，正常接受；否则仅当 `decode(input_ids)==prompt` 时允许，适配 `compare_hf.py prepare` 先截断 ID、再解码文本的表示。截断多字节字符可生成 `�`，使文本无法重新编码回原 ID；各角色 `*.engine.json` 的 `tokenizer` 记录 `input_decodings`、两种关系的布尔值及 `lossy_nonroundtrip=true`。该标记表示文本相对原 ID 的非往返表示，不改变实际模型输入。编码和解码关系均不匹配时拒绝；这项边界不构成模型数值一致性结论。

以下变量均由运行者按实际环境显式提供；输出目录必须不存在，时间和容量参数不是正式实验预算。模型及 tokenizer 须已在配置指定的本地路径或缓存中，子进程使用离线模式。旧 M5 仅支持 bulk；选择 pipeline 时另外提供 `--pipeline-frame-bytes "$P4_FRAME_BYTES"`，P 必须 whole/eager，单输出或总长度只允许首 token 的 case 会被拒绝。

```bash
python tools/compare_pd.py capture \
  --engine-source "$P4_ENGINE_SOURCE" --expected-sha "$P4_ENGINE_SHA" \
  --reference-config "$P4_REFERENCE_CONFIG" --reference-gpu "$P4_REFERENCE_GPU" \
  --prefill-config "$P4_PREFILL_CONFIG" --prefill-gpu "$P4_PREFILL_GPU" \
  --decode-config "$P4_DECODE_CONFIG" --decode-gpu "$P4_DECODE_GPU" \
  --prompts "$P4_PROMPTS" --max-tokens "$P4_MAX_TOKENS" --transfer bulk \
  --max-transfer-bytes "$P4_MAX_TRANSFER_BYTES" --output-dir "$P4_NEW_OUTPUT" \
  --startup-timeout "$P4_STARTUP_TIMEOUT" --request-timeout "$P4_REQUEST_TIMEOUT" \
  --normal-exit-timeout "$P4_EXIT_TIMEOUT" --term-timeout "$P4_TERM_TIMEOUT" \
  --kill-timeout "$P4_KILL_TIMEOUT" --raw-kv
```

`--ignore-eos` 可按对照条件显式加入。`reference-00000.pt` 和 `candidate-00000.pt` 等分别含有一例、每 token 一行 logits，可直接传给 `tools/compare_runs.py`。`partial/` 每个 token 独立落盘；启用 raw KV 时，runner 返回后、postprocess 前保存有效位置，D 首次 decode 前保存恢复的 prompt。`kv-*.json` 对同格式 P→D 的 data/scale 原始字节逐字节比较，reference→P 的字节差异另作描述；首 token 终态没有 D 交接。不会比较尾块中无效位置。完成文件原子提交且禁止覆盖；未落盘数据和断电持久性不受保证。

按实际 GPU/NUMA 拓扑可分别添加 `--reference-numa N`、`--prefill-numa N`、`--decode-numa N`；指定的角色使用 `numactl --cpunodebind=N --membind=N` 启动，找不到 `numactl` 或绑定失败直接报错。各角色的 `*.engine.json` 记录模型执行线程的 `/proc/thread-self/status` CPU/内存允许掩码和 CPU affinity；内存允许掩码不是对当前内存策略或实际页位置的独立证明，目标服务器需结合启动结果与拓扑核验。

```bash
python tools/compare_pd.py analyze --baseline "$P4_NEW_OUTPUT/reference-00000.pt" \
  --candidate "$P4_NEW_OUTPUT/candidate-00000.pt" --output "$P4_NEW_REPORT"
# 已人工批准并冻结具体门槛时，另加 --gate-freeze "$P4_GATE_FREEZE"
```

无 freeze 是 capture-only，不产生门槛结论。freeze JSON 必须包含非空 `version`、`approved: true`、非空 `approval_declaration` 和有限非负 `atol`、`rtol`；记录文件 SHA 和声明，不验证声明者身份。分析只对首次 token 分歧及之前的同前缀位置逐元素检查 `abs(candidate-reference) <= atol + rtol*abs(reference)`，并要求有限值和所有 greedy token 一致；失败返回非零并保留报告。控制进程对各自创建的进程组设置有界正常退出、TERM 和 KILL 等待，保留退出码、信号、残留进程和可见产物；controller 被 SIGKILL 或主机故障时无法保证它继续清理子进程。最终候选的 GPU、14B 和正式门槛均须单独验收。
