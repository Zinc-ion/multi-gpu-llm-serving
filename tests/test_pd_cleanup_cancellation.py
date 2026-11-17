import asyncio
import errno
import threading

import pytest

from myvllm.pd.client import PDClient, PDRemoteError
from myvllm.pd.pipeline import PipelineTimingError
from myvllm.pd.worker import PDWorker
from myvllm.sampling_parameters import SamplingParams
from pd_pipeline_helpers import empty_pipeline, pipeline_engine, pipeline_request


@pytest.mark.parametrize("cancel_count", [1, 2, 4])
@pytest.mark.parametrize("cancel_target", ["close", "connection"])
def test_cancellation_waits_for_owner_cleanup_and_can_restart(tmp_path, cancel_count, cancel_target):
    async def run():
        model_go = threading.Event()
        cleanup_entered, cleanup_go = threading.Event(), threading.Event()

        def prefill_engine(config):
            engine = pipeline_engine(config)
            call = engine.model_runner.call

            def gated_call(*args, **kwargs):
                if not model_go.wait(10):
                    raise TimeoutError("model gate was not released")
                return call(*args, **kwargs)

            engine.model_runner.call = gated_call
            return engine

        path = tmp_path / "prefill.jsonl"
        config = {"world_size": 1, "enforce_eager": True}
        options = {"max_connections": 4, "max_transfer_bytes": 1_000_000,
                   "request_timeout": 20, "pipeline_frame_bytes": 64}
        decode = PDWorker("decode", config, engine_factory=pipeline_engine, **options)
        de = await decode.start("127.0.0.1", 0)
        prefill = PDWorker("prefill", {**config, "timing_path": str(path)},
                           decode_endpoint=de, engine_factory=prefill_engine, **options)
        pe = await prefill.start("127.0.0.1", 0)
        pc1 = pc2 = dc1 = dc2 = request = shutdown = sender = None
        try:
            params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
            dc1 = await PDClient.open_d(de, lifetime=15)
            pc1 = await PDClient.prepare_p(pe, dc1.session_id, "sender", [1, 2, 3, 4, 5],
                                           params, lifetime=15)
            dc2 = await PDClient.open_d(de, lifetime=15)
            pc2 = await PDClient.prepare_p(pe, dc2.session_id, "queued", [4, 5, 6],
                                           params, lifetime=15)
            await PDClient.start_pipeline(pc1, dc1)
            ctx1 = next(c for c in prefill.owner.contexts.values() if c.handle.request_id == "sender")
            ctx2 = next(c for c in prefill.owner.contexts.values() if c.handle.request_id == "queued")
            server_tasks = [task for task, writer in prefill.connections.items()
                            if writer is ctx1.writer or writer is ctx2.writer]
            assert len(server_tasks) == 2
            sender = ctx1.sender
            on_layer, close_sender = sender.on_layer, sender.close
            fault, restored = path.with_name("fault"), path.with_name("restored")
            fault.symlink_to("/dev/full")
            restored.write_text("")
            fault.replace(path)

            def on_layer_then_restore(layer):
                on_layer(layer)
                if isinstance(sender.error, PipelineTimingError) and restored.exists():
                    restored.replace(path)

            def gated_close():
                cleanup_entered.set()
                if not cleanup_go.wait(10):
                    raise TimeoutError("cleanup gate was not released")
                return close_sender()

            sender.on_layer = on_layer_then_restore
            sender.close = gated_close
            request = asyncio.create_task(pc2.describe_early())
            async with asyncio.timeout(3):
                while not prefill.owner.commands.qsize():
                    await asyncio.sleep(0)
            model_go.set()
            assert await asyncio.wait_for(asyncio.to_thread(cleanup_entered.wait, 8), 9)
            with pytest.raises(PDRemoteError) as error:
                await asyncio.wait_for(request, 4)
            assert error.value.code == "PipelineTimingError"
            assert prefill.owner.error is sender.error
            assert sender.error.errno == sender.error.__cause__.errno == errno.ENOSPC
            pc1._writer.transport.abort()
            await asyncio.wait_for(ctx1.failure_response.wait(), 4)
            await asyncio.wait_for(ctx2.failure_response.wait(), 4)
            assert not prefill.owner.cleanup_done.done()
            assert prefill.sessions == len(prefill.connections) == 2
            stats = prefill.owner.core.stats()
            assert stats["open_handles"] == 2 and stats["used_blocks"] > 0
            assert stats["reserved_transfer_bytes"] > 0

            if cancel_target == "close":
                shutdown = asyncio.create_task(prefill.close())
                for _ in range(6):
                    await asyncio.sleep(0)
                target = shutdown
            else:
                target = server_tasks[0]
            assert not target.done()
            for index in range(cancel_count):
                assert target.cancel()
                for _ in range(8):
                    await asyncio.sleep(0)
                print(f"target={cancel_target} cancellations={index + 1} sessions={prefill.sessions} "
                      f"connections={len(prefill.connections)} cleanup_done={prefill.owner.cleanup_done.done()}")
                assert not target.done()
                assert not prefill.owner.cleanup_done.done()
                assert prefill.sessions == len(prefill.connections) == 2
                assert all(not task.done() for task in server_tasks)

            cleanup_go.set()
            if shutdown is not None:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(shutdown, 5)
            results = await asyncio.wait_for(asyncio.gather(*server_tasks, return_exceptions=True), 5)
            if cancel_target == "connection":
                assert sum(isinstance(result, asyncio.CancelledError) for result in results) == 1
            assert prefill.owner.cleanup_done.done()
            stats = prefill.owner.core.stats()
            assert stats["open_handles"] == stats["used_blocks"] == stats["reserved_transfer_bytes"] == 0
            assert not prefill.owner.contexts and not prefill.connections and prefill.sessions == 0
            assert not sender.thread.is_alive() and not sender.slots
            with pytest.raises(PipelineTimingError) as error:
                await prefill.close()
            assert error.value is sender.error
            assert error.value.errno == error.value.__cause__.errno == errno.ENOSPC
            assert not prefill.owner.thread.is_alive()
            await asyncio.gather(*(conn.close() for conn in (pc1, pc2, dc1, dc2)))
            pc1 = pc2 = dc1 = dc2 = None
            assert await prefill.start(*pe) == pe
            assert len(await pipeline_request(pe, de, [2, 3], request_id="after-cancel")) == 6
            await empty_pipeline(pe)
            await empty_pipeline(de)
        finally:
            model_go.set()
            cleanup_go.set()
            if shutdown is not None:
                await asyncio.gather(shutdown, return_exceptions=True)
            if request is not None:
                await asyncio.gather(request, return_exceptions=True)
            for conn in (pc1, pc2, dc1, dc2):
                if conn is not None:
                    await conn.close()
            try:
                try:
                    await prefill.close()
                except PipelineTimingError as exc:
                    assert sender is not None and exc is sender.error
            finally:
                await decode.close()

    asyncio.run(run())
