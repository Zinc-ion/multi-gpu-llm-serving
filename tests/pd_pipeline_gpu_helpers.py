"""Diagnostic-only GPU captures for actual layer copies and source lifetimes."""

import asyncio
import os

import torch

from myvllm.pd.worker import PDWorker
from myvllm.pd.wire import DEFAULT_HEADER_LIMIT, read_body, read_header, write_frame
from pd_gpu_helpers import CaptureEngine


class PipelineCaptureEngine(CaptureEngine):
    def __init__(self, config):
        super().__init__(config)
        self.copy_checks = []
        postprocess = self.scheduler.postprocess

        def check_copies(seqs, outputs):
            transfers = getattr(self.model_runner, "layer_transfers", {})
            for seq in seqs:
                transfer = transfers.get(seq.seq_id)
                if transfer is not None:
                    drained = transfer.stream.query()
                    assert drained, "Source KV still has D2H readers before postprocess"
                    self.copy_checks.append({
                        "request_id": self.request_ids[seq.seq_id],
                        "blocks": list(seq.block_table), "d2h_drained": drained,
                        "pinned_slots": len(transfer.slots),
                        "slot_bytes": sum(slot.numel() for slot in transfer.slots),
                        "pinned": all(slot.is_pinned() for slot in transfer.slots),
                    })
            return postprocess(seqs, outputs)
        self.scheduler.postprocess = check_copies

    def exit(self):
        if hasattr(self, "copy_checks") and self.model_runner is not None:
            torch.save(self.copy_checks, self.capture_path + ".copies.pt")
        super().exit()


def pipeline_gpu_worker_process(role, config, ready, stop, decode_endpoint, progress_lock=None):
    if progress_lock is not None:
        from tqdm import tqdm
        tqdm.set_lock(progress_lock)

    async def serve():
        worker = PDWorker(role, config, max_connections=8, max_transfer_bytes=1_000_000,
                          request_timeout=120, decode_endpoint=decode_endpoint,
                          pipeline_frame_bytes=256, engine_factory=PipelineCaptureEngine)
        endpoint = await worker.start("127.0.0.1", 0)
        try:
            ready.send(("ready", endpoint, os.getpid()))
            ready.close()
            await asyncio.to_thread(stop.recv)
        finally:
            stop.close()
            await worker.close()
    asyncio.run(serve())


class PausedDataRelay:
    """Forward real D restores while withholding one ACK from the GPU producer."""

    def __init__(self, decode_endpoint):
        self.decode_endpoint = decode_endpoint
        self.pause = False
        self.received = asyncio.Event()
        self.release = asyncio.Event()
        self.tasks = set()
        self.writers = set()
        self.data_writers = ()
        self.acks = []

    async def start(self):
        self.server = await asyncio.start_server(self._accept, "127.0.0.1", 0)
        return self.server.sockets[0].getsockname()[:2]

    async def _accept(self, reader, writer):
        task = asyncio.current_task()
        self.tasks.add(task)
        upstream = None
        self.writers.add(writer)
        try:
            header, _ = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
            remote, upstream = await asyncio.open_connection(*self.decode_endpoint)
            self.writers.add(upstream)
            await write_frame(upstream, header)
            response, _ = await read_header(remote, DEFAULT_HEADER_LIMIT, 0)
            await write_frame(writer, response)
            if header["op"] == "pipeline":
                self.data_writers = (writer, upstream)
                while True:
                    region, size = await read_header(reader, DEFAULT_HEADER_LIMIT, 256)
                    payload = await read_body(reader, size)
                    await write_frame(upstream, region, payload)
                    ack, _ = await read_header(remote, DEFAULT_HEADER_LIMIT, 0)
                    assert ack["ok"]
                    if self.pause:
                        self.acks.append(ack["value"])
                        self.received.set()
                        await self.release.wait()
                    await write_frame(writer, ack)
                    if ack["value"]["copies_done"]:
                        break
        except (ConnectionError, OSError, asyncio.IncompleteReadError):
            pass
        finally:
            for connection in (writer, upstream):
                if connection is not None:
                    connection.close()
                    try:
                        await connection.wait_closed()
                    except (ConnectionError, OSError):
                        pass
                    self.writers.discard(connection)
            self.tasks.discard(task)

    async def close(self):
        self.release.set()
        self.server.close()
        await self.server.wait_closed()
        for writer in list(self.writers):
            writer.transport.abort()
        if self.tasks:
            await asyncio.gather(*list(self.tasks))
