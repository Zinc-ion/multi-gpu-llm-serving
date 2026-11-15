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
