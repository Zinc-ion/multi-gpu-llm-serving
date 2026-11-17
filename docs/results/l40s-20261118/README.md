# Qwen3-14B / L40S 原始结果

实验日期为 2026-11-18。配置、解释、支持范围及限制见 [实测报告](../../l40s_acceptance_2026-11-18_zh.md)。本目录不包含模型权重。

## 文件索引

| 内容 | 文件 |
|---|---|
| 实际硬件与拓扑 | `hardware.txt.gz` |
| 固定配置、输入及客户端 | `assets/configs/`、`assets/workloads/`、`assets/client/` |
| TP1 与 HF | `tp1-hf-reference.json` |
| TP2 修复前后对 TP1 | `tp1-vs-tp2.json`、`tp1-vs-tp2-fixed.json` |
| 首个 Row GEMM 边界重放 | `tp-row-replay-summary.json` |
| 生产数值及 HTTP 修复 | `tp2-fp32-partials.patch.gz`，对应 `dca7522` 的两个生产文件 |
| 双卡 bulk 有效 KV 与数值 | `pd-bf16-correctness-summary.json` |
| PD 生命周期及回收 | `bf16-pd-lifecycle.json`、`bf16-pd-after-http-stats.json` |
| 四卡组合的剩余分歧 | `bf16-four-pd-fixed-lifecycle.json`；`all_match_tp1=false` |
| BF16 KV 与 FP8 KV | `bf16-vs-fp8-kv.json` |
| 正式逐请求数据 | `bf16-tp1-c{1,4}.json`、`bf16-tp2-fixed-c{1,4}.json`、`fp8-kv-tp1-c{1,4}.json`、`bf16-pd-c{1,4}.json` |
| 并发 4、32 请求的补充批次 | `bf16-tp1-final32-c4.json`、`bf16-tp2-final32-c4.json` |
| 四卡诊断计时 | `bf16-four-pd-fixed-c{1,4}.json`；HTTP 成功不表示其数值检查通过 |
| 汇总及可直接打开的表格 | `performance-summary.json`、`performance-summary.csv` |
| vLLM 原版统计函数独立核对 | `vllm-metrics-check.json` |
| 设备显存原始采样 | `*-memory.jsonl.gz` |
| 启动时分项容量日志 | `capacity-summary.json` |
| 实际启动及测量命令 | `measurement-commands.json` |
| 原始双 rank profile | `profile/rank-{0,1}.json.gz`、`profile/capture.json` |
| profile 摘要 | `profile-summary-fixed.json` |
| 最终 CPU 与代表性 GPU 用例日志 | `validation/*.log.gz` |

保留的修复前四卡 `bf16-four-pd-*` 计时也仅为诊断，不与修复后数据合并。`measurement-progress.json` 的 `completed` 表示脚本完成；数值是否通过以对应比较和生命周期 JSON 为准。正确性工具中的 `capture_only`、`gate_passed=null` 表示未设置通用 logits 容差，不应改写为完整数值验收通过。

硬件文本、原始 patch 和测试日志压缩保存以保留原始空白及终端输出，使用 `gzip -dc 文件.gz` 查看。只复算性能表时保留已有 `capacity-summary.json`；只有提供原始启动 `.log` 时才重新提取容量，不用空结果覆盖它。

## 指标和长度

每批 2 个预热不计入正式样本。原批次每档 8 请求，补充批次每档 32 请求；它们分别统计，不拼接批次或平均分位数。全部请求的固定输入由同一 tokenizer 校验为 4096 token，实际输出数取服务端流式 usage，已完成请求均为 128 token。输入字段来自已校验的物化请求，不宣称它是客户端另行保存的服务端 input usage。合成输入为重复文本；重复请求不代表样本语义多样。

`results[].ttft`、`latency`、`tpot_s`、`itl` 的单位是秒。TPOT 使用保存的 `tpot_s=(latency-ttft)/(output_tokens-1)`，不是 dataclass 中未赋值的 `tpot` 字段。TTFT/TPOT 的 mean、p50、p95、p99 和计时样本数见汇总；分位数在排序后 `(n-1)q` 处线性插值，与本地 vLLM 0.15.0 的 NumPy 默认方法对照。8/32 请求的 p95/p99 都是经验分位数，不表示稳定尾延迟或 SLA。

输出吞吐为成功输出 token 数/正式批次墙钟；总 token 吞吐为成功输入＋输出 token 数/相同墙钟，排除预热，包含完成尾巴。单请求 TTFT/TPOT 从获得客户端并发槽位后的实际请求开始，不包含其等待槽位的时间；该等待包含在批次墙钟中。PD 首二输出 SSE 事件间隔单独保留，它包含交接、排队及 Decode 激活，不能当成纯 KV 传输时间。

显存为服务启动后、包含本档预热和请求的 `nvidia-smi` device used memory 采样最大值，单位 MiB。记录实际采样数和间隔，不称精确峰值或 PyTorch allocated。多卡总量使用同一时刻的采样和，每卡最大值分别保存。

## 复算与复现

只复算表格，无需 GPU、PyTorch 或 vLLM：

```bash
python3 docs/results/l40s-20261118/assets/collect_small_results.py \
  --root docs/results/l40s-20261118
python3 tools/summarize_profile.py \
  docs/results/l40s-20261118/profile/rank-0.json.gz \
  docs/results/l40s-20261118/profile/rank-1.json.gz \
  --output /tmp/minivllm-profile-summary.json
```

运行模型前将 `assets/` 复制到自己的任务目录，修改配置中的模型、cache 绝对路径，并使用已有环境；无需安装完整 vLLM 服务端。`run_small_acceptance.py` 使用 `PROJECT_ROOT`、`acceptance-ready/` 与 `runs/acceptance-20261118/`，必须显式选择场景，避免默认执行历史全列表或覆盖原始日志。两个补测场景为 `bf16-tp1-final32`、`bf16-tp2-final32`；运行前以 `tools/prepare_workloads.py --num-requests 32` 准备对应 `4k-128-32.jsonl`。单独启动服务与客户端的命令见实测报告。

HTTP 请求实现直接保存自 vLLM 0.15.0 的 `vllm/benchmarks/lib/endpoint_request_func.py`，未修改其代码，Apache-2.0 许可证在 `assets/client/VLLM_LICENSE`。外层脚本只提供固定请求、并发、预热和结果保存，不是完整 `vllm bench serve` CLI 运行；不能将它描述成赛事专用脚本的复现。已有完整 vLLM 客户端时可直接使用报告中的官方 CLI 模板。

## 较大捕获文件

完整 logits 捕获压缩包 `raw-logits.tar.gz` 为 133,955,102 字节，未放入 Git；执行机副本在仓库父目录的 `evidence/l40s-acceptance-20261118/raw-logits.tar.gz`，租机副本在 `/root/data/zincion/ai-infra/l40s-20261117/runs/acceptance-20261118/raw-logits.tar.gz`。这些较大文件不是 clone 后自动获得的资产。

完整有效 KV 原始快照保留在同一租机结果根目录的 `pd-bf16-correctness/`；本目录保存其逐案例字节比较摘要。profile 原始 trace 已压缩随 Git 提交，不只提供摘要。

运行时租机 HEAD 仍是 `65906d2`，生产文件叠加保存的修复 patch；它与提交 `dca7522` 的两个生产文件差异一致。部分旧 manifest/capture 中的 commit 因此仍为准备时或远端 HEAD，不表示忽略了实际生效的修复。源码基线和 patch 各记录一次即可，不需要反复生成快照或指纹。
