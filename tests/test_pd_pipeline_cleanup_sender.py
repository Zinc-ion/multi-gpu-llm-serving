import asyncio
import errno

import pytest

from myvllm.pd.client import PDClient, PDRemoteError
from myvllm.pd.pipeline import PipelineError, PipelineTimingError
from myvllm.pd.worker import PDWorker
from myvllm.sampling_parameters import SamplingParams
from pd_pipeline_helpers import empty_pipeline, pipeline_engine, pipeline_workers, prepare_pair
from test_pd_pipeline_cleanup_data import recovered_pipeline


@pytest.mark.parametrize("operation", ["wait_pipeline", "next_token"])
def test_transient_sender_timing_failure_preserves_error_and_cleans(tmp_path, operation):
    async def run():
        path = tmp_path / "prefill.jsonl"
        config = {"world_size": 1, "enforce_eager": True, "layer_delay": .8}
        options = {"max_connections": 4, "max_transfer_bytes": 1_000_000,
                   "request_timeout": 10, "pipeline_frame_bytes": 64, "engine_factory": pipeline_engine}
        decode = PDWorker("decode", config, **options)
        de = await decode.start("127.0.0.1", 0)
        prefill = PDWorker("prefill", {**config, "timing_path": str(path)}, decode_endpoint=de, **options)
        pe = await prefill.start("127.0.0.1", 0)
        pc = dc = wait = None
        try:
            dc = await PDClient.open_d(de, lifetime=8)
            pc = await PDClient.prepare_p(
                pe, dc.session_id, "transient-enospc", [1, 2, 3, 4, 5],
                SamplingParams(temperature=0, max_tokens=4, ignore_eos=True), lifetime=8)
            await PDClient.start_pipeline(pc, dc)
            sender = next(iter(prefill.owner.contexts.values())).sender
            fault = path.with_name("fault-symlink")
            restored = path.with_name("restored.jsonl")
            fault.symlink_to("/dev/full")
            restored.write_text("")
            fault.replace(path)
            wait = asyncio.create_task(getattr(pc, operation)())
            try:
                async with asyncio.timeout(4):
                    while sender.error is None:
                        await asyncio.sleep(.001)
            finally:
                restored.replace(path)
            assert isinstance(sender.error, PipelineTimingError)
            assert sender.error.errno == errno.ENOSPC
            assert isinstance(sender.error.__cause__, OSError) and sender.error.__cause__.errno == errno.ENOSPC
            with pytest.raises(PDRemoteError) as error:
                await asyncio.wait_for(wait, 4)
            assert error.value.code == "PipelineTimingError"
            await asyncio.wait_for(prefill.failed.wait(), 4)
            await asyncio.to_thread(prefill.owner.thread.join, 3)
            assert not prefill.owner.thread.is_alive()
            assert prefill.owner.error is sender.error
            assert not sender.thread.is_alive() and not sender.slots
            stats = prefill.owner.core.stats()
            assert stats["open_handles"] == stats["used_blocks"] == stats["reserved_transfer_bytes"] == 0
            assert not prefill.owner.contexts
            await pc.close()
            pc = None
            await dc.close()
            dc = None
            await empty_pipeline(de)
            with pytest.raises(PipelineTimingError) as shutdown_error:
                await prefill.close()
            assert shutdown_error.value is sender.error
            assert await prefill.start(*pe) == pe
            await recovered_pipeline(pe, de)
            await empty_pipeline(pe)
            await empty_pipeline(de)
        finally:
            if wait is not None:
                await asyncio.gather(wait, return_exceptions=True)
            if pc is not None:
                await pc.close()
            if dc is not None:
                await dc.close()
            try:
                try:
                    await prefill.close()
                except OSError as exc:
                    assert exc.errno == errno.ENOSPC
            finally:
                await decode.close()

    asyncio.run(run())


def test_sender_connect_timing_failure_reports_original_wire_error(tmp_path):
    async def run():
        path = tmp_path / "prefill.jsonl"
        config = {"world_size": 1, "enforce_eager": True}
        options = {"max_connections": 4, "max_transfer_bytes": 1_000_000,
                   "request_timeout": 10, "pipeline_frame_bytes": 64, "engine_factory": pipeline_engine}
        decode = PDWorker("decode", config, **options)
        de = await decode.start("127.0.0.1", 0)
        prefill = PDWorker("prefill", {**config, "timing_path": str(path)}, decode_endpoint=de, **options)
        pe = await prefill.start("127.0.0.1", 0)
        pc = dc = None
        try:
            pc, dc = await prepare_pair(pe, de, [1, 2, 3, 4, 5], lifetime=8)
            assert await dc.reserve_early(await pc.describe_early())
            fault = path.with_name("fault-symlink")
            restored = path.with_name("restored.jsonl")
            fault.symlink_to("/dev/full")
            restored.write_text("")
            fault.replace(path)
            try:
                with pytest.raises(PDRemoteError) as error:
                    await asyncio.wait_for(pc.start(), 4)
            finally:
                restored.replace(path)
            assert error.value.code == "PipelineTimingError"
            await asyncio.wait_for(prefill.failed.wait(), 4)
            await asyncio.to_thread(prefill.owner.thread.join, 3)
            assert isinstance(prefill.owner.error, PipelineTimingError)
            assert prefill.owner.error.errno == errno.ENOSPC
            assert prefill.owner.error.__cause__.errno == errno.ENOSPC
            assert prefill.owner.core.stats()["open_handles"] == 0
            assert not prefill.owner.contexts
        finally:
            if pc is not None:
                await pc.close()
            if dc is not None:
                await dc.close()
            try:
                try:
                    await prefill.close()
                except OSError as exc:
                    assert exc.errno == errno.ENOSPC
            finally:
                await decode.close()

    asyncio.run(run())


def test_normal_pipeline_cancel_keeps_pipeline_error_and_worker_alive():
    async def run():
        async with pipeline_workers(layer_delay=.1) as (worker, _, pe, de):
            pc, dc = await prepare_pair(pe, de, [1, 2, 3, 4, 5])
            try:
                await PDClient.start_pipeline(pc, dc)
                sender = next(iter(worker.owner.contexts.values())).sender
                try:
                    await pc.cancel()
                except (PDRemoteError, ConnectionError, OSError, asyncio.IncompleteReadError):
                    pass
                assert isinstance(sender.error, PipelineError)
                assert worker.owner.thread.is_alive() and worker.owner.error is None
            finally:
                await pc.close()
                await dc.close()
            await empty_pipeline(pe)
            await empty_pipeline(de)

    asyncio.run(run())


def test_pipeline_socket_failure_keeps_pipeline_error_and_worker_alive():
    async def run():
        async with pipeline_workers(layer_delay=.25) as (worker, decode, pe, de):
            pc, dc = await prepare_pair(pe, de, [1, 2, 3, 4, 5])
            try:
                await PDClient.start_pipeline(pc, dc)
                sender = next(iter(worker.owner.contexts.values())).sender
                await decode.close()
                async with asyncio.timeout(3):
                    while sender.error is None:
                        await asyncio.sleep(.005)
                assert isinstance(sender.error, PipelineError)
            finally:
                await pc.close()
                await dc.close()
            await empty_pipeline(pe)
            assert worker.owner.thread.is_alive() and worker.owner.error is None

    asyncio.run(run())
