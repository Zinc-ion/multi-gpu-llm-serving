# M8：D 本地 prefill 与可选 D 端点

本节描述 PD worker 的接口，不代表负载感知网关、动态路由、TP2 PD 或 M8 验收已经完成。固定 P/D 路径与协议版本 2、3、4、5 继续可用；完整交接及部署约束见 [M5](m5_runtime_zh.md)、[M6](m6_runtime_zh.md) 和 [M7](m7_runtime_zh.md)。

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

JSON 中端点为 `[host, port]`。网关仍需核对允许的每条 P/D 路径的模型格式、长度、特性与有效采样种子；这些字段不使固定网关自动支持交叉路由。

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

## owner 统计

`PDClient.stats()` 通过独立连接向 owner 查询，在原有字段上增加：

- `total_blocks`：物理块池大小；既有 `used_blocks` 和 `free_blocks` 是当前物理块数。
- P 的 `admitted_requests`：prepared、ready、queued、prefill、held、exported 请求数；`pending_prefill_tokens` 是尚未计算的 prompt token 数，部分 chunk 后按实际 cached 进度减少，held/exported 为零。既有 `held_requests` 单独统计 held/exported。
- D 的 `reserved_blocks`、`resident_requests`、`resident_limit` 继续表示完整逻辑预算和使用量；`local_resident_requests` 与 `remote_resident_requests` 将当前占用 resident 配额的本地与远端请求分开。终态及尚未 reserve 的空 D session 不计入 resident。
- D 的 `decode_step_wall_ms`：最近至多 32 个合格 step 的最大 host wall 毫秒值。`decode_step_sample_count` 是窗口内样本数，`decode_step_sample_source` 为 `core.step_wall_decode_including_mixed`，`decode_step_sample_age_ms` 是该最大值样本在本 owner 的年龄。无样本时 wall 和年龄为 `null`、数量为 0。

只有实际产生 `completion_tokens > 1` 事件的 D step 进入该窗口，包括既有 decode 与本地 prefill 同批的 step；只有本地首 token 的 step 不计。计时包围实际 `core.step`，包含引擎已有的采样结果 D2H 完成，不为统计额外同步 GPU。窗口只按样本数有界，长时间无新样本时年龄会增长；调用方须自行设置过期和无样本策略。该值包含整批计算，但不包括两个 step 之间的 restore、控制、排队和网络等待，只能作为近期服务时间启发式，不能保证客户端 TPOT 或 SLO。当前计数与观测也不代替 owner 的实时 admission。
