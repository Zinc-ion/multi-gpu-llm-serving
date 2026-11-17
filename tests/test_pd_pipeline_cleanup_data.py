import asyncio
import errno

import pytest

from myvllm.pd.client import PDClient
from myvllm.pd.wire import read_body, read_header, write_frame
from myvllm.pd.worker import PDWorker
from myvllm.sampling_parameters import SamplingParams
from pd_pipeline_helpers import empty_pipeline, pipeline_engine


async def call(reader, writer, operation, **fields):
    await write_frame(writer, {"op": operation, **fields})
    header, size = await read_header(reader, 1_000_000, 1_000_000)
    assert not await read_body(reader, size)
    assert header["ok"] is True, header
    return header["value"]


async def recovered_pipeline(pe, de):
    dc = await PDClient.open_d(de, lifetime=8)
    pc = None
    try:
        pc = await PDClient.prepare_p(
            pe, dc.session_id, "recovered", [2, 3],
            SamplingParams(temperature=0, max_tokens=2, ignore_eos=True), lifetime=8)
        await PDClient.start_pipeline(pc, dc)
        first = await pc.next_token()
        assert not first["finished"]
        await PDClient.handoff_pipeline(pc, dc, await pc.describe())
        assert (await dc.next_token())["finished"]
    finally:
        if pc is not None:
            await pc.close()
        await dc.close()


def test_data_disconnect_timing_failure_closes_decode_control_socket(tmp_path):
    async def run():
        path = tmp_path / "decode.jsonl"
        config = {"world_size": 1, "enforce_eager": True, "layer_delay": .8}
        options = {"max_connections": 4, "max_transfer_bytes": 1_000_000,
                   "request_timeout": 10, "pipeline_frame_bytes": 64, "engine_factory": pipeline_engine}
        decode = PDWorker("decode", {**config, "timing_path": str(path)}, **options)
        de = await decode.start("127.0.0.1", 0)
        prefill = PDWorker("prefill", config, decode_endpoint=de, **options)
        pe = await prefill.start("127.0.0.1", 0)
        reader = writer = pc = close_pc = None
        try:
            reader, writer = await asyncio.open_connection(*de)
            opened = await call(reader, writer, "open_d", lifetime=8)
            pc = await PDClient.prepare_p(
                pe, opened["session_id"], "data-cleanup", [1, 2, 3, 4, 5],
                SamplingParams(temperature=0, max_tokens=4, ignore_eos=True), lifetime=8)
            await call(reader, writer, "reserve_early", metadata=await pc.describe_early())
            assert await pc.start()
            fault = path.with_name("fault-symlink")
            fault.symlink_to("/dev/full")
            fault.replace(path)
            close_pc = asyncio.create_task(pc.close())
            await asyncio.wait_for(decode.failed.wait(), 4)
            await asyncio.to_thread(decode.owner.thread.join, 3)
            assert not decode.owner.thread.is_alive()
            assert isinstance(decode.owner.error, OSError) and decode.owner.error.errno == errno.ENOSPC
            assert await asyncio.wait_for(reader.read(1), 2) == b""
            async with asyncio.timeout(2):
                while decode.sessions or decode.connections:
                    await asyncio.sleep(.01)
            stats = decode.owner.core.stats()
            assert stats["open_handles"] == stats["used_blocks"] == stats["reserved_blocks"] == 0
            assert stats["resident_requests"] == stats["reserved_transfer_bytes"] == 0
            await close_pc
            close_pc = None
            pc = None
            with pytest.raises(OSError) as error:
                await decode.close()
            assert error.value.errno == errno.ENOSPC
            path.unlink()
            assert await decode.start(*de) == de
            await recovered_pipeline(pe, de)
            await empty_pipeline(pe)
            await empty_pipeline(de)
        finally:
            if writer is not None:
                writer.transport.abort()
                await writer.wait_closed()
            if pc is not None:
                await pc.close()
            if close_pc is not None:
                await asyncio.gather(close_pc, return_exceptions=True)
            try:
                await prefill.close()
            finally:
                try:
                    await decode.close()
                except OSError as exc:
                    assert exc.errno == errno.ENOSPC

    asyncio.run(run())
