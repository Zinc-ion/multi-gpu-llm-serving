# M8：D 本地 prefill 与可选 D 端点

本节描述 PD worker 和可选负载路由；TP2 PD 与 M8 阶段验收尚未完成。固定 P/D 路径与协议版本 2、3、4、5 继续可用；完整交接及部署约束见 [M5](m5_runtime_zh.md)、[M6](m6_runtime_zh.md) 和 [M7](m7_runtime_zh.md)。

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
- `enforce_eager`、`world_size`；已有 `pipeline_frame_bytes` 为零时表示流水未启用。当前 PD worker 仍限制 `world_size=1`。

JSON 中端点为 `[host, port]`。固定网关继续按固定 pair 工作；下述负载模式会核对配置 inventory 中每条允许的 P/D 路径。

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

`--m8-routing-transfer bulk` 是默认值，令 fixed/load/elastic 路由组均使用 bulk；选择 `pipeline` 时三组均使用相同流水阈值和帧上限。三组的 P 都为 whole/eager，D 都开启 chunked，使用相同 chunk budget 和 base config 中的 D eager/graph 模式；P、D 从 base 独立派生。生成的 `fixed-bulk-gateway.sh` 与 `fixed-pipeline-gateway.sh` 使用相同 fixed P/D worker 配置及各自网关计时文件，用于单独的机制比较。先为所选 D 模式完成实际 GPU 正确性验证，再执行任何性能比较。

同一输出目录另外生成四个 `whole-control-N` 与四个 `chunk-control-N` 单实例配置和启动脚本，逐卡独立配置、seed、端口和计时文件；各组以 manifest 中的四个 endpoint 组成一个目的地集合。manifest 同时记录物理 GPU 顺序、NUMA 映射、P/D 采样种子、策略与 chunk budget。各比较组复用相同物理端口和 GPU，按组依次启动；它们不是要同时启动的八个 GPU 实例。实际 GPU 空闲、NUMA 绑定、独立引擎资源及运行结果仍须在目标服务器核验。

当前已完成真实 CPU worker/TCP/HTTP 的负载路由和资源回收测试；CPU 测试使用模型执行替身，不证明 CUDA Graph、DMA 或 14B 的正确性与性能。目标服务器 GPU、跨 NUMA、L3、TP2 和完整 M8 评测仍待验收。
