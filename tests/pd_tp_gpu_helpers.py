"""GPU smoke helpers: each owned worker sets its physical visibility before CUDA use."""

import asyncio
import os


def reference_on_gpu(visible, config, cases, ready):
    os.environ['CUDA_VISIBLE_DEVICES'] = visible
    from pd_gpu_helpers import reference_process
    reference_process(config, cases, ready)


def worker_on_gpus(visible, role, config, ready, stop, decodes):
    os.environ['CUDA_VISIBLE_DEVICES'] = visible
    from myvllm.pd.worker import PDWorker
    from pd_gpu_helpers import CaptureEngine

    async def serve():
        kwargs = {}
        if role == 'prefill':
            kwargs.update(decode_endpoint=decodes[0], allowed_decode_endpoints=decodes)
        worker = PDWorker(role, config, max_connections=4, max_transfer_bytes=1000000,
                          request_timeout=120, engine_factory=CaptureEngine, **kwargs)
        endpoint = await worker.start('127.0.0.1', 0)
        try:
            ready.send(('ready', endpoint))
            ready.close()
            await asyncio.to_thread(stop.recv)
        finally:
            stop.close()
            await worker.close()
    asyncio.run(serve())
