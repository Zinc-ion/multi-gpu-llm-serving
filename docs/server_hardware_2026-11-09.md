# 远端服务器硬件与系统快照

采集日期：2026-11-09

连接别名：`186zincion`。本文件只记录项目规划所需的硬件和系统信息，不包含 SSH 密钥或认证内容。

## GPU

- GPU：5 张 NVIDIA GeForce RTX 4090（AD102）
- 每张显存：`49140 MiB`，约 48 GiB；采集时每张已用 `0 MiB`
- 驱动：`580.178.04`
- NVIDIA-SMI 报告的 CUDA 兼容版本：`13.0`
- ECC：Disabled
- 功耗上限：`450 W`
- PCIe：最大 Gen4 x16；空闲采样时当前链路为 Gen1 x16
- MIG：未启用/不适用

PCIe/NUMA 拓扑：

```text
GPU0 GPU1 GPU2：同一 NUMA 域，彼此为 NODE
GPU3 GPU4：同一 NUMA 域，彼此为 NODE
GPU0-2 与 GPU3-4：跨 NUMA，连接类型为 SYS
```

`nvidia-smi topo -p2p r` 和 `nvidia-smi topo -p2p w` 对 GPU 间均显示 `CNS`（Chipset not supported）。因此不能假设 GPU 之间存在可用的直接 P2P 或 NVLink 通道，PD 状态传输必须实测实际路径和带宽。

## CPU、内存与磁盘

- 系统：Ubuntu `22.04.1 LTS`
- 内核：`5.15.0-191-generic`
- CPU：双路 AMD EPYC 9534，每路 64 核、2 线程/核；系统共 128 物理核/256 线程
- NUMA：2 个节点
- 内存：约 `1.0 TiB`
- `/dev/shm`：约 `504 GiB`
- 根分区：`394G`，可用约 `95G`
- `/Data1`：`7.0T`，可用约 `3.4T`
- `/Data2`：`7.0T`，可用约 `2.5T`
- `/Data2/zincion`：当前归属用户和用户组 `zincion`，目录可写，适合作为项目、模型和实验数据目录

## 对项目的直接影响

1. 第一版 2+2 PD 可优先尝试 `GPU0/1` 与 `GPU3/4`，使两侧实例都在各自的 NUMA/PCIe 域内；`GPU2` 暂作备用。P/D 方向需要用真实传输基准决定。
2. 5 卡 `3+2` 虽然符合硬件分组，但三卡 TP 是否满足目标模型的头数和权重切分约束，不能仅由拓扑判断。
3. 模型权重和 checkpoint 应放在 `/Data2/zincion` 下，避免消耗根分区；实验结果也放在该目录。
4. 服务器信息足以开始本地实现和测试调度、缓存、TP/PD 接口。远端 CUDA/Conda 环境只在需要 GPU 验证时再安装或启用。

## 采集命令

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,pci.bus_id --format=csv
nvidia-smi topo -m
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
uname -a
lscpu
free -h
df -hT
```
