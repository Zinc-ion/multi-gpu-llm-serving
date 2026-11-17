import asyncio
import errno

import pytest

from myvllm.pd.client import PDClient, PDRemoteError
from myvllm.pd.worker import PDWorker
from myvllm.sampling_parameters import SamplingParams
from pd_pipeline_helpers import empty_pipeline, pipeline_engine, pipeline_request


@pytest.mark.parametrize("failed_role", ["prefill", "decode"])
def test_pipeline_timing_failure_closes_sender_receiver_and_core(tmp_path, failed_role):
    async def run():
        p_path = tmp_path / "p.jsonl"
        d_path = tmp_path / "d.jsonl"
        common = {"world_size": 1, "enforce_eager": True, "layer_delay": .12}
        opts = {"max_connections": 4, "max_transfer_bytes": 1_000_000, "request_timeout": 5,
                "pipeline_frame_bytes": 64, "engine_factory": pipeline_engine}
        d = PDWorker("decode", {**common, "timing_path": str(d_path)}, **opts)
        de = await d.start("127.0.0.1", 0)
        p = PDWorker("prefill", {**common, "timing_path": str(p_path)}, decode_endpoint=de, **opts)
        pe = await p.start("127.0.0.1", 0)
        pc = dc = None
        try:
            dc = await PDClient.open_d(de, lifetime=4)
            pc = await PDClient.prepare_p(
                pe, dc.session_id, "timing-failure", [1, 2, 3, 4, 5],
                SamplingParams(temperature=0, max_tokens=4, ignore_eos=True), lifetime=4)
            await PDClient.start_pipeline(pc, dc)
            sender = next(iter(p.owner.contexts.values())).sender
            receiver = next(iter(d.owner.contexts.values())).handle._receiver
            assert sender is not None and receiver is not None
            path = p_path if failed_role == "prefill" else d_path
            fault = path.with_name("timing-fault")
            fault.symlink_to("/dev/full")
            path.rename(path.with_name("before-failure.jsonl"))
            fault.replace(path)
            failed = p if failed_role == "prefill" else d
            with pytest.raises((OSError, ConnectionError, asyncio.IncompleteReadError, PDRemoteError)):
                await pc.next_token()
            await asyncio.wait_for(failed.failed.wait(), 3)
            await asyncio.to_thread(failed.owner.thread.join, 3)
            assert not failed.owner.thread.is_alive()
            assert isinstance(failed.owner.error, OSError) and failed.owner.error.errno == errno.ENOSPC
            stats = failed.owner.core.stats()
            assert stats["open_handles"] == stats["used_blocks"] == stats["reserved_transfer_bytes"] == 0
            assert not failed.owner.contexts
            if failed_role == "prefill":
                assert not sender.thread.is_alive() and not sender.slots
                assert not failed.owner.engine.model_runner.layer_transfers
            else:
                assert not receiver.slots
            await pc.close()
            pc = None
            await dc.close()
            dc = None
            with pytest.raises(OSError) as close_error:
                await failed.close()
            assert close_error.value.errno == errno.ENOSPC
            path.unlink()
            path.with_name("before-failure.jsonl").rename(path)
            assert await failed.start(*(pe if failed_role == "prefill" else de)) == (
                pe if failed_role == "prefill" else de)
            await empty_pipeline(pe)
            await empty_pipeline(de)
            assert len(await pipeline_request(pe, de, [2, 3])) == 6
            await empty_pipeline(pe)
            await empty_pipeline(de)
        finally:
            if pc is not None:
                await pc.close()
            if dc is not None:
                await dc.close()
            for worker in (p, d):
                try:
                    await worker.close()
                except OSError as exc:
                    assert exc.errno == errno.ENOSPC

    asyncio.run(run())


def test_pipeline_timing_failure_cleans_every_live_transaction(tmp_path):
    async def run():
        path = tmp_path / "p.jsonl"
        common = {"world_size": 1, "enforce_eager": True, "layer_delay": .1}
        opts = {"max_connections": 4, "max_transfer_bytes": 1_000_000, "request_timeout": 5,
                "pipeline_frame_bytes": 64, "engine_factory": pipeline_engine}
        d = PDWorker("decode", common, **opts)
        de = await d.start("127.0.0.1", 0)
        p = PDWorker("prefill", {**common, "timing_path": str(path)}, decode_endpoint=de, **opts)
        pe = await p.start("127.0.0.1", 0)
        connections = []
        try:
            for index in range(2):
                dc = await PDClient.open_d(de, lifetime=4)
                connections.append(dc)
                pc = await PDClient.prepare_p(
                    pe, dc.session_id, f"request-{index}", [1, 2, 3, 4, 5],
                    SamplingParams(temperature=0, max_tokens=4, ignore_eos=True), lifetime=4)
                connections.append(pc)
            await asyncio.gather(*(PDClient.start_pipeline(connections[i], connections[i - 1])
                                   for i in (1, 3)))
            senders = [ctx.sender for ctx in p.owner.contexts.values()]
            assert len(senders) == 2 and all(sender is not None for sender in senders)
            fault = path.with_name("timing-fault")
            fault.symlink_to("/dev/full")
            path.rename(path.with_name("before-failure.jsonl"))
            fault.replace(path)
            await connections[1].close()
            await asyncio.wait_for(p.failed.wait(), 3)
            await asyncio.to_thread(p.owner.thread.join, 3)
            assert not p.owner.thread.is_alive()
            stats = p.owner.core.stats()
            assert stats["open_handles"] == stats["used_blocks"] == stats["reserved_transfer_bytes"] == 0
            assert not p.owner.contexts and not p.owner.engine.model_runner.layer_transfers
            assert all(not sender.thread.is_alive() and not sender.slots for sender in senders)
        finally:
            await asyncio.gather(*(connection.close() for connection in connections), return_exceptions=True)
            try:
                await p.close()
            except OSError as exc:
                assert exc.errno == errno.ENOSPC
            await empty_pipeline(de)
            await d.close()

    asyncio.run(run())
