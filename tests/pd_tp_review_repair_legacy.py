"""Run unchanged M5 TCP workers for the fixed-gateway compatibility test."""

import asyncio
import json
from pathlib import Path
import sys

source = Path(sys.argv[1])
sys.path.insert(0, str(source / "tests"))
sys.path.insert(0, str(source / "src"))

from myvllm.pd.worker import PDWorker
from test_pd_server import cpu_gateway_engine, make_config


async def main():
    config, _ = make_config(Path(sys.argv[2]))
    options = dict(max_connections=4, max_transfer_bytes=1000000,
                   request_timeout=10, engine_factory=cpu_gateway_engine)
    workers = []
    try:
        decode = PDWorker("decode", config, **options)
        de = await decode.start("127.0.0.1", 0)
        workers.append(decode)
        prefill = PDWorker("prefill", config, decode_endpoint=de, **options)
        pe = await prefill.start("127.0.0.1", 0)
        workers.append(prefill)
        mismatch = PDWorker("prefill", {**config, "seed": 1}, decode_endpoint=de, **options)
        bad_pe = await mismatch.start("127.0.0.1", 0)
        workers.append(mismatch)
        print("READY=" + json.dumps({"config": config, "p": pe, "d": de, "bad_p": bad_pe}),
              flush=True)
        await asyncio.to_thread(sys.stdin.readline)
    finally:
        for worker in reversed(workers):
            await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
