# M6 逐层交接

当前实现提供早期事务的 core 接口和 CPU 状态验证。worker、客户端和网关尚未接入该接口；独立数据连接、CUDA 异步拷贝及计算和传输重叠尚未实现或验证。默认 M5 整块传输仍使用原有接口。

## 早期事务

早期事务只接受 TP=1、P whole/eager prefill。P 启用 chunked prefill 或未启用 eager 时 `PrefillCore.prepare` 显式拒绝；D 使用的 decode eager/graph 模式不受此限制。首 token 已知必为终态（`max_tokens=1` 或总长度只允许一个输出）的请求继续走原有整块路径，不预留早期交接。

1. `PrefillCore.prepare` 验证并建立事务，生成协议版本 3 的早期元数据；此时不加入 Scheduler。它只包含 prompt，完成 token 数和缓存进度均为 0，KV 描述按完整 prompt 计算。`describe_early` 返回副本。
2. `DecodeCore.reserve_early` 验证会话、模型和采样规格，预留完整输出预算，并在 D 自己的 block table 中分配 prompt 目标块。P 的 queued/transfer 状态也按 prompt 块数计入配额；早期事务与整块事务同时存在时，共同受 P 的物理块总数约束。P 的 `reserved_prompt_blocks` 仅在存在早期事务时表示这项配额。
3. 调用方只有在数据通道与此 session 和 transaction 绑定且就绪后，才能调用 P `ready` 和 `start`。core 本身没有跨进程连接，不能独立证明通道已就绪；这一保证由后续网络接入负责。
4. `restore_region` 使用 D 已分配的 block table 接收一层连续的逻辑 token 区间。层号从 0 开始递增，每层的区间从 0 连续到 prompt 长度；重复、跳跃、乱序或字节数错误使事务失败并释放 D 的预留。每帧的内存顺序是 K、V，各为 `[token, kv_head, head_dim]`，不携带 P 或 D 的物理 block ID。当前该方法同步复制普通 host bytes，只供 core 阶段使用。
5. P 首 token 非终态时，D `commit_early` 验证普通版本 2 最终元数据与早期描述一致。最终提交与分片可按任意先后顺序到达；只有两者齐全后 D 才进入 `restored`，可调用已有的 `activate`。已提交的首 token 尚无 KV，D 从该位置开始 decode。首 token 为 EOS 时取消 D 预留，不提交。
6. 准备、预留、提交、P 释放和激活的重复调用在相同连接和相同内容下返回原有幂等结果；重复或乱序分片则拒绝，不进行重试。`acknowledge_early` 只释放已离开 Scheduler 的 P 请求，调用方须先确认 D 已完成恢复，不能用单个分片的网络 ACK 代替。

当前没有 GPU DMA，因此 CPU 测试中的取消和首 token EOS 不能证明异步拷贝期间的块生命周期。数据通道接入必须在 P 的 `postprocess` 可能释放源块前完成本 batch 的在途 D2H，并在 D 的 H2D 完成后才允许对应 pinned slot 重用及确认恢复。短 prompt 切换阈值、真实逐层重叠、TP rank 对应和目标服务器性能仍待实现或验证。
