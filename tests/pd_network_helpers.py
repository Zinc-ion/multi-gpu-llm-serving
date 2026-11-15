"""CPU execution stand-in and process entry for real PD socket tests."""

import asyncio
import time

from myvllm.pd.worker import PDWorker
from test_pd_core import CPUExecution, make_engine


class SlowCPUExecution(CPUExecution):
    def __init__(self, config, delay=0, forced_token=None):
        super().__init__(config, forced_token)
        self.delay = delay

    def call(self, method, sequences, is_prefill):
        if self.delay:
            time.sleep(self.delay)
        return super().call(method, sequences, is_prefill)


def cpu_engine(config):
    engine = make_engine(blocks=config.get("blocks", 12),
                         budget=config.get("budget", 32),
                         sequences=config.get("sequences", 4),
                         model_length=config.get("model_length", 32),
                         chunked=config.get("chunked", False),
                         forced_token=config.get("forced_token"))
    engine.model_runner = SlowCPUExecution(engine.config, config.get("delay", 0),
                                           config.get("forced_token"))
    engine.exit = lambda: None
    return engine


def make_worker(role, *, decode_endpoint=None, **config):
    return PDWorker(role, {"world_size": 1, **config}, max_connections=4,
                    max_transfer_bytes=1_000_000, request_timeout=5,
                    decode_endpoint=decode_endpoint, engine_factory=cpu_engine)


def cpu_worker_process(role, config, ready, stop, decode_endpoint, port=0):
    async def serve():
        worker = make_worker(role, decode_endpoint=decode_endpoint, **config)
        endpoint = await worker.start("127.0.0.1", port)
        ready.send(endpoint)
        ready.close()
        await asyncio.to_thread(stop.wait)
        await worker.close()
    asyncio.run(serve())
