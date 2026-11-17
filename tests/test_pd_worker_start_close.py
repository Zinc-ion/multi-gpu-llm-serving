import asyncio
import threading
import time

import pytest

from myvllm.pd.client import PDClient
from myvllm.pd.worker import PDWorker
from pd_network_helpers import cpu_engine


def test_close_during_start_waits_for_shutdown_and_worker_can_restart():
    loading = threading.Event()

    def delayed_engine(config):
        loading.set()
        time.sleep(.3)
        return cpu_engine(config)

    async def run():
        worker = PDWorker("decode", {"world_size": 1}, max_connections=2,
                          max_transfer_bytes=1_000_000, request_timeout=5,
                          engine_factory=delayed_engine)
        starting = asyncio.create_task(worker.start("127.0.0.1", 0))
        assert await asyncio.to_thread(loading.wait, 2)
        try:
            await worker.close()
            endpoint = await starting
            assert not worker.owner.thread.is_alive()
            assert not worker.server.is_serving()
            with pytest.raises((ConnectionError, OSError)):
                await PDClient.info(endpoint, timeout=.5)
            restarted = await worker.start("127.0.0.1", 0)
            assert (await PDClient.info(restarted))["role"] == "decode"
        finally:
            if not starting.done():
                starting.cancel()
                await asyncio.gather(starting, return_exceptions=True)
            await worker.close()
        assert not worker.owner.thread.is_alive()
        assert not worker.server.is_serving()

    asyncio.run(run())
