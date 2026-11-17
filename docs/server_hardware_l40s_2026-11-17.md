# L40S 租机：磁盘、环境与复现入口

无卡准备采集于 2026-11-17，GPU 实测于 2026-11-18。当前租机为 `instance-x3mho0nw`，实际配置为 4×L40S。目标模型正确性、性能与四卡部署结果见 [实测报告](l40s_acceptance_2026-11-18_zh.md)，本页保留磁盘和环境说明。

## 与学校内网服务器的区别

| 项目 | 学校内网服务器（历史环境） | 当前 L40S 租机 |
|---|---|---|
| 硬件记录 | [2026-11-09 快照](server_hardware_2026-11-09.md) | 本页 |
| 数据盘 | `/Data2`，历史快照约 7.0T | `/root/data`，XFS，容量 150 GiB |
| 新增文件范围 | `/Data2/zincion/ai-infra` | `/root/data/zincion/ai-infra` |
| 当前任务目录 | 不沿用旧实验结果作为新验收 | `/root/data/zincion/ai-infra/l40s-20261117` |
| GPU 与拓扑 | 5 张 RTX 4090，见旧快照 | 实测 4×L40S，GPU0/1 与 GPU2/3 分属两个 NUMA 域 |

2026-11-18 开卡后的实际记录：

| 项目 | 实测 |
|---|---|
| GPU 显存 | 每卡 `46068 MiB`（nvidia-smi）；CUDA runtime 报告总量另见原始 capacity 数据 |
| 驱动 / CUDA runtime | `610.43.02` / PyTorch `cu128` |
| GPU0/1 | PCI `52:00.0`、`56:00.0`，NUMA0，互连 PXB |
| GPU2/3 | PCI `D1:00.0`、`D5:00.0`，NUMA1，互连 PXB |
| 两组之间 | SYS；P2P 读取能力状态均为 OK，不是带宽测量 |
| CPU cgroup | `5200000 100000`，即 52 CPU 配额 |
| 内存 cgroup | `502511173632` 字节，约 468 GiB |
| 基础检查 | 四张卡的实际 CUDA 运算均通过 |

四卡部署使用 P 的 TP2 rank 占 GPU0/1、两个 TP1 D 分别占 GPU2/3；这些是当前租机的实际设备号。原始硬件记录以 `hardware.txt.gz` 随 [结果目录](results/l40s-20261118/README.md) 保存。

租机的根文件系统为 50 GiB 的 `overlay`。在租机中手动创建 `/Data2` 不会使它成为数据盘：迁移前 `findmnt -T /Data2/zincion/ai-infra/l40s-20261117` 返回 `/`，系统盘已用约 37 GiB，150 GiB 数据盘几乎为空。因此不能继续以旧目录名判断数据归属，也不需要扩容或重新下载模型。

当前任务将模型、Python 环境、依赖缓存、实验输入、日志及临时文件统一放在上述数据盘任务目录。旧 `/Data2/zincion/ai-infra/l40s-20261117` 仅保留指向新目录的符号链接，供已有 Python 可执行文件链接和脚本中的绝对路径继续工作；实际文件存储与新命令均使用数据盘目录，不保留系统盘上的完整副本。

迁移后实测：系统盘已用约 191 MiB（1%），数据盘已用约 37 GiB（25%）、剩余约 114 GiB。已核对复制前后文件类型、大小和链接目标，重新读取全部模型分片的张量元数据，并通过 tokenizer、配置、引擎/PD 模块导入及三个服务 CLI 检查；没有重装环境或重新下载模型。记录位于任务目录的 `runs/data-disk-*.json` 和 `runs/data-disk-*-help.txt`，不构成 GPU 正确性或性能验收。

旧实验文档中的 `/Data2` 示例及旧生成器的固定路径约束属于学校服务器环境，不是租机的全局写入限制。当前最小验收使用现有角色入口和数据盘中的配置，不要求运行这些旧实验生成器。

## 环境与模型

- Python `3.11.16`，PyTorch `2.9.1+cu128`，Triton `3.5.1`，Transformers `4.57.3`。
- BF16 Qwen3-14B 位于 `models/Qwen3-14B`；8 个 safetensors 分片、443 个 BF16 张量，权重索引记录 `29,536,614,400` 字节。资产读取通过不代表 GPU 推理通过。
- `env.sh` 将 Python、模型/依赖缓存、Triton/CUDA 缓存、临时文件和 Python bytecode 缓存设在任务目录中。复用已有环境，不重新安装整个 vLLM 或改动系统 Conda。
- 无卡阶段 cgroup 配额为 1 CPU、1 GiB 内存，开卡后配额如上表。迁移后的模块与 CLI 检查按进程串行执行；同时在父子进程导入 PyTorch 的检查曾触发 OOM，不能把该次退出记成通过。
- 原无卡 CPU 专项的 5 秒启动超时和 30 秒预算补充结果分别保存在 `runs/pytest-cpu*.log`，不能将补充运行写成原严格用例通过。

```bash
source /root/data/zincion/ai-infra/l40s-20261117/env.sh
cd "$PROJECT_ROOT/repo"
findmnt -T "$PROJECT_ROOT"
df -hT / "$PROJECT_ROOT"
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.device_count())'
```

`acceptance-ready/configs/` 包含 BF16 TP1、BF16 TP2、FP8 KV TP1 和 BF16 P/D 两端配置；总长度 4352，KV 上限 80 块，最多 4 序列，eager 模式。`acceptance-ready/workloads/correctness.json` 保存两个短 chat 与 255/256/257/4096 token 输入；`4k-128.jsonl` 及 manifest 保存 8 个实际 4096 token 输入、128 token 输出预算的请求，tokenizer 回环已检查。双卡 1P1D 与四卡 TP2 P＋两个 TP1 D 均已启动并运行实际请求，功能边界见实测报告。

`acceptance-ready/client/` 保存小批次测量入口、未修改的 vLLM 0.15.0 HTTP 请求函数及其 Apache-2.0 许可证；环境补充 `aiohttp==3.13.3`，未安装完整 vLLM 服务端。客户端保存预热及逐请求结果，汇总 TTFT/TPOT mean/p50/p95/p99、输出与输入＋输出吞吐、长度及计数；显存另以设备采样记录。TP1/TP2 并发 4 又分别使用 `4k-128-32.jsonl` 完成独立 32 请求补测，其他路径维持每档 8 请求。模拟 HTTP 协议检查与真实模型测量分别保存，后者才是本轮性能证据。

沿用用户开启的机器，按 [最小验收](../.codex/tasks/m8_acceptance.md) 串行验证。设备号和预算以现场检查为准；不复用学校服务器的 GPU0/1/3/4 与 NUMA 假设，不自行开卡、额外租机或更改计费。
