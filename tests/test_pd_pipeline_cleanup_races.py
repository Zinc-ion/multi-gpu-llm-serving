import asyncio
import errno
import threading

import pytest

import myvllm.pd.pipeline as pipeline_module
import myvllm.pd.worker as worker_module
from myvllm.pd.client import PDClient, PDRemoteError
from myvllm.pd.pipeline import PipelineTimingError
from myvllm.pd.worker import PDWorker
from myvllm.sampling_parameters import SamplingParams
from pd_pipeline_helpers import empty_pipeline, pipeline_engine, pipeline_request
from test_pd_pipeline_cleanup_data import recovered_pipeline


async def wait_thread_event(event):
    assert await asyncio.wait_for(asyncio.to_thread(event.wait, 8), 9)


@pytest.mark.parametrize("operation", ["wait_pipeline", "next_token"])
@pytest.mark.parametrize("exit_first", [False, True])
def test_sender_failure_after_context_cleanup_reaches_control_client(tmp_path, operation, exit_first):
    async def run():
        model_go, fault_seen, resume_model = threading.Event(), threading.Event(), threading.Event()
        exit_entered, exit_go = threading.Event(), threading.Event()

        def prefill_engine(config):
            engine = pipeline_engine(config)
            call, exit_engine = engine.model_runner.call, engine.exit

            def gated_call(*args, **kwargs):
                if not model_go.wait(8):
                    raise TimeoutError("model gate was not released")
                return call(*args, **kwargs)

            def gated_exit():
                exit_entered.set()
                if not exit_go.wait(8):
                    raise TimeoutError("exit gate was not released")
                return exit_engine()

            engine.model_runner.call = gated_call
            engine.exit = gated_exit
            return engine

        path = tmp_path / "prefill.jsonl"
        config = {"world_size": 1, "enforce_eager": True}
        options = {"max_connections": 4, "max_transfer_bytes": 1_000_000,
                   "request_timeout": 15, "pipeline_frame_bytes": 64}
        decode = PDWorker("decode", config, engine_factory=pipeline_engine, **options)
        de = await decode.start("127.0.0.1", 0)
        prefill = PDWorker("prefill", {**config, "timing_path": str(path)},
                           decode_endpoint=de, engine_factory=prefill_engine, **options)
        pe = await prefill.start("127.0.0.1", 0)
        pc = dc = sender = None
        try:
            dc = await PDClient.open_d(de, lifetime=12)
            pc = await PDClient.prepare_p(
                pe, dc.session_id, "response-race", [1, 2, 3, 4, 5],
                SamplingParams(temperature=0, max_tokens=4, ignore_eos=True), lifetime=12)
            await PDClient.start_pipeline(pc, dc)
            ctx = next(iter(prefill.owner.contexts.values()))
            sender = ctx.sender
            on_layer = sender.on_layer

            def gated_on_layer(layer):
                on_layer(layer)
                if isinstance(sender.error, PipelineTimingError):
                    fault_seen.set()
                    if not resume_model.wait(8):
                        raise TimeoutError("fault gate was not released")

            sender.on_layer = gated_on_layer
            fault, restored = path.with_name("fault"), path.with_name("restored")
            fault.symlink_to("/dev/full")
            restored.write_text("")
            fault.replace(path)
            model_go.set()
            await wait_thread_event(fault_seen)
            restored.replace(path)
            assert isinstance(sender.error, PipelineTimingError)
            assert sender.error.errno == errno.ENOSPC
            assert isinstance(sender.error.__cause__, OSError)
            assert sender.error.__cause__.errno == errno.ENOSPC
            resume_model.set()
            await wait_thread_event(exit_entered)
            assert prefill.owner.error is sender.error
            assert ctx.sender is None and not prefill.owner.contexts
            if exit_first:
                exit_go.set()
                await asyncio.wait_for(asyncio.to_thread(prefill.owner.thread.join, 3), 4)
                assert not prefill.owner.thread.is_alive()
            else:
                assert prefill.owner.thread.is_alive()
            assert not prefill.failed.is_set()
            with pytest.raises(PDRemoteError) as error:
                await asyncio.wait_for(getattr(pc, operation)(), 3)
            assert error.value.code == "PipelineTimingError"
            assert "No space left" in str(error.value)
            exit_go.set()
            await asyncio.wait_for(prefill.failed.wait(), 4)
            await pc.close()
            pc = None
            await dc.close()
            dc = None
            await empty_pipeline(de)
            with pytest.raises(PipelineTimingError) as error:
                await asyncio.wait_for(prefill.close(), 4)
            assert error.value is sender.error
            stats = prefill.owner.core.stats()
            assert stats["open_handles"] == stats["used_blocks"] == stats["reserved_transfer_bytes"] == 0
            assert not prefill.connections and prefill.sessions == 0
            assert not sender.thread.is_alive() and not sender.slots
            assert await prefill.start(*pe) == pe
            await recovered_pipeline(pe, de)
            await empty_pipeline(pe)
            await empty_pipeline(de)
        finally:
            model_go.set()
            resume_model.set()
            exit_go.set()
            if pc is not None:
                await pc.close()
            if dc is not None:
                await dc.close()
            try:
                try:
                    await prefill.close()
                except PipelineTimingError as exc:
                    assert sender is not None and exc is sender.error
            finally:
                await decode.close()

    asyncio.run(run())


def test_cleanup_submission_competes_with_owner_failure(tmp_path):
    async def run():
        model_go, queue_entered, fault_seen = threading.Event(), threading.Event(), threading.Event()

        def prefill_engine(config):
            engine = pipeline_engine(config)
            call = engine.model_runner.call

            def gated_call(*args, **kwargs):
                if not model_go.wait(8):
                    raise TimeoutError("model gate was not released")
                return call(*args, **kwargs)

            engine.model_runner.call = gated_call
            return engine

        path = tmp_path / "prefill.jsonl"
        config = {"world_size": 1, "enforce_eager": True}
        options = {"max_connections": 4, "max_transfer_bytes": 1_000_000,
                   "request_timeout": 15, "pipeline_frame_bytes": 64}
        decode = PDWorker("decode", config, engine_factory=pipeline_engine, **options)
        de = await decode.start("127.0.0.1", 0)
        prefill = PDWorker("prefill", {**config, "timing_path": str(path)},
                           decode_endpoint=de, engine_factory=prefill_engine, **options)
        pe = await prefill.start("127.0.0.1", 0)
        pc1 = pc2 = dc1 = dc2 = sender = None
        captured = []
        try:
            params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
            dc2 = await PDClient.open_d(de, lifetime=12)
            pc2 = await PDClient.prepare_p(pe, dc2.session_id, "disconnect", [4, 5, 6], params, lifetime=12)
            dc1 = await PDClient.open_d(de, lifetime=12)
            pc1 = await PDClient.prepare_p(pe, dc1.session_id, "sender-failure", [1, 2, 3, 4, 5],
                                           params, lifetime=12)
            await PDClient.start_pipeline(pc1, dc1)
            ctx1 = next(c for c in prefill.owner.contexts.values() if c.handle.request_id == "sender-failure")
            ctx2 = next(c for c in prefill.owner.contexts.values() if c.handle.request_id == "disconnect")
            sender = ctx1.sender
            on_layer = sender.on_layer
            fault, restored = path.with_name("fault"), path.with_name("restored")
            fault.symlink_to("/dev/full")
            restored.write_text("")
            fault.replace(path)

            def on_layer_with_restore(layer):
                on_layer(layer)
                if isinstance(sender.error, PipelineTimingError) and restored.exists():
                    restored.replace(path)
                    fault_seen.set()

            sender.on_layer = on_layer_with_restore
            put = prefill.owner.commands.put_nowait

            def gated_put(item):
                if item[0] is ctx2 and item[1] == "cleanup":
                    captured.append(item)
                    queue_entered.set()
                    model_go.set()
                    if not fault_seen.wait(8):
                        raise TimeoutError("sender timing failure did not reach queue gate")
                return put(item)

            prefill.owner.commands.put_nowait = gated_put
            pc2._writer.transport.abort()
            await wait_thread_event(queue_entered)
            assert len(captured) == 1
            future = captured[0][-1]
            await asyncio.wait_for(asyncio.to_thread(prefill.owner.thread.join, 4), 5)
            assert isinstance(sender.error, PipelineTimingError) and sender.error.errno == errno.ENOSPC
            assert prefill.owner.error is sender.error
            assert not prefill.owner.thread.is_alive()
            assert future.done() and future.exception() is sender.error
            assert prefill.owner.commands.empty()
            assert not prefill.owner.contexts and prefill.owner.core.stats()["open_handles"] == 0
            with pytest.raises(PipelineTimingError) as error:
                await asyncio.wait_for(prefill.close(), 4)
            assert error.value is sender.error
            assert prefill.sessions == 0 and not prefill.connections
            await pc1.close()
            pc1 = None
            await pc2.close()
            pc2 = None
            await dc1.close()
            dc1 = None
            await dc2.close()
            dc2 = None
            await empty_pipeline(de)
        finally:
            model_go.set()
            fault_seen.set()
            await asyncio.to_thread(prefill.owner.thread.join, 4)
            if captured and not prefill.owner.thread.is_alive() and not captured[0][-1].done():
                captured[0][-1].set_result((True, b""))
            for connection in (pc1, pc2, dc1, dc2):
                if connection is not None:
                    await connection.close()
            try:
                try:
                    await prefill.close()
                except PipelineTimingError as exc:
                    assert sender is not None and exc is sender.error
            finally:
                await decode.close()

    asyncio.run(run())


def test_cleanup_queued_after_normal_shutdown_is_settled():
    async def run():
        model_entered, model_go = threading.Event(), threading.Event()

        def prefill_engine(config):
            engine = pipeline_engine(config)
            call = engine.model_runner.call

            def gated_call(*args, **kwargs):
                model_entered.set()
                if not model_go.wait(8):
                    raise TimeoutError("model gate was not released")
                return call(*args, **kwargs)

            engine.model_runner.call = gated_call
            return engine

        config = {"world_size": 1, "enforce_eager": True}
        options = {"max_connections": 4, "max_transfer_bytes": 1_000_000,
                   "request_timeout": 15, "pipeline_frame_bytes": 64}
        decode = PDWorker("decode", config, engine_factory=pipeline_engine, **options)
        de = await decode.start("127.0.0.1", 0)
        prefill = PDWorker("prefill", config, decode_endpoint=de, engine_factory=prefill_engine, **options)
        pe = await prefill.start("127.0.0.1", 0)
        pc1 = pc2 = dc1 = dc2 = None
        entered = threading.Event()
        captured = []
        try:
            params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
            dc2 = await PDClient.open_d(de, lifetime=12)
            pc2 = await PDClient.prepare_p(pe, dc2.session_id, "disconnect", [4, 5, 6], params, lifetime=12)
            dc1 = await PDClient.open_d(de, lifetime=12)
            pc1 = await PDClient.prepare_p(pe, dc1.session_id, "normal-shutdown", [1, 2, 3],
                                           params, lifetime=12)
            await PDClient.start_pipeline(pc1, dc1)
            await wait_thread_event(model_entered)
            ctx = next(c for c in prefill.owner.contexts.values() if c.handle.request_id == "disconnect")
            put = prefill.owner.commands.put_nowait

            def observed_put(item):
                result = put(item)
                if item[0] is ctx and item[1] == "cleanup":
                    captured.append(item)
                    entered.set()
                return result

            prefill.owner.commands.put_nowait = observed_put
            shutdown = prefill.owner.submit(None, "shutdown")
            pc2._writer.transport.abort()
            await wait_thread_event(entered)
            future = captured[0][-1]
            assert not shutdown.done() and not future.done()
            model_go.set()
            await asyncio.wait_for(asyncio.to_thread(prefill.owner.thread.join, 4), 5)
            assert not prefill.owner.thread.is_alive()
            assert shutdown.done() and shutdown.result() == (True, b"")
            assert future.done() and str(future.exception()) == "PD worker stopped"
            assert prefill.owner.commands.empty() and not prefill.owner.contexts
            assert prefill.owner.core.stats()["open_handles"] == 0
            await asyncio.wait_for(prefill.close(), 4)
            assert prefill.sessions == 0 and not prefill.connections
            for connection in (pc1, pc2, dc1, dc2):
                await connection.close()
            pc1 = pc2 = dc1 = dc2 = None
            await empty_pipeline(de)
        finally:
            model_go.set()
            for connection in (pc1, pc2, dc1, dc2):
                if connection is not None:
                    await connection.close()
            try:
                await prefill.close()
            finally:
                await decode.close()

    asyncio.run(run())


def test_delayed_failure_signal_does_not_poison_restarted_worker(tmp_path):
    async def run():
        model_go, fault_seen, resume_model = threading.Event(), threading.Event(), threading.Event()
        exit_entered, exit_go = threading.Event(), threading.Event()
        first_engine = True

        def prefill_engine(config):
            nonlocal first_engine
            engine = pipeline_engine(config)
            if not first_engine:
                return engine
            first_engine = False
            call, exit_engine = engine.model_runner.call, engine.exit

            def gated_call(*args, **kwargs):
                if not model_go.wait(8):
                    raise TimeoutError("model gate was not released")
                return call(*args, **kwargs)

            def gated_exit():
                exit_entered.set()
                if not exit_go.wait(8):
                    raise TimeoutError("exit gate was not released")
                return exit_engine()

            engine.model_runner.call = gated_call
            engine.exit = gated_exit
            return engine

        path = tmp_path / "prefill.jsonl"
        config = {"world_size": 1, "enforce_eager": True}
        options = {"max_connections": 4, "max_transfer_bytes": 1_000_000,
                   "request_timeout": 15, "pipeline_frame_bytes": 64}
        decode = PDWorker("decode", config, engine_factory=pipeline_engine, **options)
        de = await decode.start("127.0.0.1", 0)
        prefill = PDWorker("prefill", {**config, "timing_path": str(path)},
                           decode_endpoint=de, engine_factory=prefill_engine, **options)
        pe = await prefill.start("127.0.0.1", 0)
        pc = dc = sender = None
        try:
            dc = await PDClient.open_d(de, lifetime=12)
            pc = await PDClient.prepare_p(
                pe, dc.session_id, "restart-race", [1, 2, 3, 4, 5],
                SamplingParams(temperature=0, max_tokens=4, ignore_eos=True), lifetime=12)
            await PDClient.start_pipeline(pc, dc)
            ctx = next(iter(prefill.owner.contexts.values()))
            sender = ctx.sender
            on_layer = sender.on_layer

            def gated_on_layer(layer):
                on_layer(layer)
                if isinstance(sender.error, PipelineTimingError):
                    fault_seen.set()
                    if not resume_model.wait(8):
                        raise TimeoutError("fault gate was not released")

            sender.on_layer = gated_on_layer
            fault, restored = path.with_name("fault"), path.with_name("restored")
            fault.symlink_to("/dev/full")
            restored.write_text("")
            fault.replace(path)
            model_go.set()
            await wait_thread_event(fault_seen)
            restored.replace(path)
            assert sender.error.errno == errno.ENOSPC and sender.error.__cause__.errno == errno.ENOSPC
            resume_model.set()
            await wait_thread_event(exit_entered)
            old_owner, old_failed = prefill.owner, prefill.failed
            assert old_owner.error is sender.error and not old_owner.contexts
            await pc.close()
            pc = None
            if prefill.connections:
                await asyncio.wait_for(asyncio.gather(*list(prefill.connections)), 4)
            assert not prefill.connections and prefill.sessions == 0 and ctx.failure_response.is_set()
            exit_go.set()
            # Keep the event loop in this turn until the old owner enqueues its callback.
            old_owner.thread.join(5)
            assert not old_owner.thread.is_alive() and not old_failed.is_set()
            with pytest.raises(PipelineTimingError) as error:
                await prefill.close()
            assert error.value is sender.error and not old_failed.is_set()
            assert await prefill.start(*pe) == pe
            await asyncio.wait_for(old_failed.wait(), 2)
            assert prefill.failed is not old_failed and not prefill.failed.is_set()
            assert prefill.owner.thread.is_alive() and prefill.owner.error is None
            assert len(await pipeline_request(pe, de, [2, 3], request_id="healthy-after-restart")) == 6
            assert not prefill.failed.is_set()
            await dc.close()
            dc = None
            await empty_pipeline(pe)
            await empty_pipeline(de)
        finally:
            model_go.set()
            resume_model.set()
            exit_go.set()
            if pc is not None:
                await pc.close()
            if dc is not None:
                await dc.close()
            try:
                try:
                    await prefill.close()
                except PipelineTimingError as exc:
                    assert sender is not None and exc is sender.error
            finally:
                await decode.close()

    asyncio.run(run())


@pytest.mark.parametrize("later_error", ["enospc", "eisdir"])
def test_first_sender_timing_error_survives_later_worker_timing_failure(tmp_path, later_error):
    async def run():
        model_go, fault_seen, resume_model = threading.Event(), threading.Event(), threading.Event()

        def prefill_engine(config):
            engine = pipeline_engine(config)
            call = engine.model_runner.call

            def gated_call(*args, **kwargs):
                if not model_go.wait(8):
                    raise TimeoutError("model gate was not released")
                return call(*args, **kwargs)

            engine.model_runner.call = gated_call
            return engine

        path = tmp_path / "prefill.jsonl"
        config = {"world_size": 1, "enforce_eager": True}
        options = {"max_connections": 4, "max_transfer_bytes": 1_000_000,
                   "request_timeout": 15, "pipeline_frame_bytes": 64}
        decode = PDWorker("decode", config, engine_factory=pipeline_engine, **options)
        de = await decode.start("127.0.0.1", 0)
        prefill = PDWorker("prefill", {**config, "timing_path": str(path)},
                           decode_endpoint=de, engine_factory=prefill_engine, **options)
        pe = await prefill.start("127.0.0.1", 0)
        pc = dc = sender = request = None
        try:
            dc = await PDClient.open_d(de, lifetime=12)
            pc = await PDClient.prepare_p(
                pe, dc.session_id, "two-errors", [1, 2, 3, 4, 5],
                SamplingParams(temperature=0, max_tokens=4, ignore_eos=True), lifetime=12)
            await PDClient.start_pipeline(pc, dc)
            ctx = next(iter(prefill.owner.contexts.values()))
            sender = ctx.sender
            on_layer = sender.on_layer

            def gated_on_layer(layer):
                on_layer(layer)
                if isinstance(sender.error, PipelineTimingError):
                    fault_seen.set()
                    if not resume_model.wait(8):
                        raise TimeoutError("fault gate was not released")

            sender.on_layer = gated_on_layer
            fault = path.with_name("fault")
            fault.symlink_to("/dev/full")
            fault.replace(path)
            model_go.set()
            await wait_thread_event(fault_seen)
            first = sender.error
            assert isinstance(first, PipelineTimingError) and first.errno == errno.ENOSPC
            assert type(first.__cause__) is OSError and first.__cause__.errno == errno.ENOSPC
            await asyncio.wait_for(asyncio.to_thread(sender.thread.join, 3), 4)
            assert not sender.thread.is_alive()
            if later_error == "eisdir":
                path.unlink()
                path.mkdir()
            previous_response = ctx.response_done
            request = asyncio.create_task(pc.next_token())
            async with asyncio.timeout(3):
                while ctx.response_done is previous_response:
                    await asyncio.sleep(.001)
            resume_model.set()
            with pytest.raises(PDRemoteError) as error:
                await asyncio.wait_for(request, 4)
            assert error.value.code == "PipelineTimingError"
            await asyncio.wait_for(prefill.failed.wait(), 4)
            await asyncio.wait_for(asyncio.to_thread(prefill.owner.thread.join, 3), 4)
            assert prefill.owner.error is first and prefill.owner.error.__cause__.errno == errno.ENOSPC
            stats = prefill.owner.core.stats()
            assert stats["open_handles"] == stats["used_blocks"] == stats["reserved_transfer_bytes"] == 0
            assert not prefill.owner.contexts and not sender.slots
            await pc.close()
            pc = None
            await dc.close()
            dc = None
            await empty_pipeline(de)
            with pytest.raises(PipelineTimingError) as error:
                await asyncio.wait_for(prefill.close(), 4)
            assert error.value is first
        finally:
            model_go.set()
            resume_model.set()
            if request is not None:
                await asyncio.gather(request, return_exceptions=True)
            if pc is not None:
                await pc.close()
            if dc is not None:
                await dc.close()
            try:
                try:
                    await prefill.close()
                except OSError:
                    pass
            finally:
                await decode.close()

    asyncio.run(run())


def test_failed_owner_holds_session_permits_until_contexts_are_cleaned(tmp_path):
    async def run():
        model_go, close_entered, close_go = threading.Event(), threading.Event(), threading.Event()

        def prefill_engine(config):
            engine = pipeline_engine(config)
            call = engine.model_runner.call

            def gated_call(*args, **kwargs):
                if not model_go.wait(8):
                    raise TimeoutError("model gate was not released")
                return call(*args, **kwargs)

            engine.model_runner.call = gated_call
            return engine

        path = tmp_path / "prefill.jsonl"
        config = {"world_size": 1, "enforce_eager": True}
        options = {"max_connections": 4, "max_transfer_bytes": 1_000_000,
                   "request_timeout": 15, "pipeline_frame_bytes": 64}
        decode = PDWorker("decode", config, engine_factory=pipeline_engine, **options)
        de = await decode.start("127.0.0.1", 0)
        prefill = PDWorker("prefill", {**config, "timing_path": str(path)},
                           decode_endpoint=de, engine_factory=prefill_engine, **options)
        pe = await prefill.start("127.0.0.1", 0)
        pc1 = pc2 = dc1 = dc2 = request = sender = None
        try:
            params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
            dc1 = await PDClient.open_d(de, lifetime=12)
            pc1 = await PDClient.prepare_p(pe, dc1.session_id, "sender", [1, 2, 3, 4, 5],
                                           params, lifetime=12)
            dc2 = await PDClient.open_d(de, lifetime=12)
            pc2 = await PDClient.prepare_p(pe, dc2.session_id, "queued", [4, 5, 6], params, lifetime=12)
            await PDClient.start_pipeline(pc1, dc1)
            ctx1 = next(c for c in prefill.owner.contexts.values() if c.handle.request_id == "sender")
            ctx2 = next(c for c in prefill.owner.contexts.values() if c.handle.request_id == "queued")
            server1 = next(t for t, w in prefill.connections.items() if w is ctx1.writer)
            server2 = next(t for t, w in prefill.connections.items() if w is ctx2.writer)
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
                close_entered.set()
                if not close_go.wait(8):
                    raise TimeoutError("cleanup gate was not released")
                return close_sender()

            sender.on_layer = on_layer_then_restore
            sender.close = gated_close
            request = asyncio.create_task(pc2.describe_early())
            async with asyncio.timeout(3):
                while not prefill.owner.commands.qsize():
                    await asyncio.sleep(.001)
            model_go.set()
            await wait_thread_event(close_entered)
            with pytest.raises(PDRemoteError) as error:
                await asyncio.wait_for(request, 3)
            assert error.value.code == "PipelineTimingError"
            assert prefill.owner.error is sender.error and sender.error.errno == errno.ENOSPC
            pc1._writer.transport.abort()
            await asyncio.wait_for(ctx1.failure_response.wait(), 3)
            await asyncio.wait_for(ctx2.failure_response.wait(), 3)
            done, pending = await asyncio.wait((server1, server2), timeout=.2)
            assert not done and pending == {server1, server2}
            stats = prefill.owner.core.stats()
            assert prefill.sessions == 2 and len(prefill.connections) == 2
            assert stats["open_handles"] == 2 and stats["used_blocks"] == 2
            assert stats["reserved_transfer_bytes"] > 0
            assert ctx1.handle is not None and ctx2.handle is not None
            close_go.set()
            await asyncio.wait_for(asyncio.gather(server1, server2), 4)
            await asyncio.wait_for(prefill.failed.wait(), 4)
            stats = prefill.owner.core.stats()
            assert stats["open_handles"] == stats["used_blocks"] == stats["reserved_transfer_bytes"] == 0
            assert not prefill.owner.contexts and not prefill.connections and prefill.sessions == 0
            assert not sender.thread.is_alive() and not sender.slots
        finally:
            model_go.set()
            close_go.set()
            if request is not None:
                await asyncio.gather(request, return_exceptions=True)
            for connection in (pc1, pc2, dc1, dc2):
                if connection is not None:
                    await connection.close()
            try:
                try:
                    await prefill.close()
                except PipelineTimingError as exc:
                    assert sender is not None and exc is sender.error
            finally:
                await decode.close()

    asyncio.run(run())


def test_connect_sender_failure_after_owner_cleanup_reaches_client(tmp_path):
    async def run():
        exit_entered, exit_go = threading.Event(), threading.Event()

        def prefill_engine(config):
            engine = pipeline_engine(config)
            exit_engine = engine.exit

            def gated_exit():
                exit_entered.set()
                if not exit_go.wait(8):
                    raise TimeoutError("exit gate was not released")
                return exit_engine()

            engine.exit = gated_exit
            return engine

        path = tmp_path / "prefill.jsonl"
        config = {"world_size": 1, "enforce_eager": True}
        options = {"max_connections": 4, "max_transfer_bytes": 1_000_000,
                   "request_timeout": 15, "pipeline_frame_bytes": 64}
        decode = PDWorker("decode", config, engine_factory=pipeline_engine, **options)
        de = await decode.start("127.0.0.1", 0)
        prefill = PDWorker("prefill", {**config, "timing_path": str(path)},
                           decode_endpoint=de, engine_factory=prefill_engine, **options)
        pe = await prefill.start("127.0.0.1", 0)
        pc = dc = sender = None
        try:
            dc = await PDClient.open_d(de, lifetime=12)
            pc = await PDClient.prepare_p(
                pe, dc.session_id, "connect-race", [1, 2, 3, 4, 5],
                SamplingParams(temperature=0, max_tokens=4, ignore_eos=True), lifetime=12)
            assert await dc.reserve_early(await pc.describe_early())
            ctx = next(iter(prefill.owner.contexts.values()))
            original_await = prefill._await_owner
            original_submit = prefill.owner.submit
            connect_future = None

            def capture_connect(ctx, op, *args, **kwargs):
                nonlocal connect_future
                future = original_submit(ctx, op, *args, **kwargs)
                if op == "connect_pipeline":
                    connect_future = future
                return future

            async def await_after_cleanup(reader, future):
                result = await original_await(reader, future)
                if future is connect_future:
                    await wait_thread_event(exit_entered)
                    assert ctx.sender is None and not prefill.owner.contexts
                return result

            prefill._await_owner = await_after_cleanup
            prefill.owner.submit = capture_connect
            fault, restored = path.with_name("fault"), path.with_name("restored")
            fault.symlink_to("/dev/full")
            restored.write_text("")
            fault.replace(path)
            with pytest.raises(PDRemoteError) as error:
                await asyncio.wait_for(pc.start(), 4)
            restored.replace(path)
            assert error.value.code == "PipelineTimingError"
            assert isinstance(prefill.owner.error, PipelineTimingError)
            assert prefill.owner.error.errno == prefill.owner.error.__cause__.errno == errno.ENOSPC
            exit_go.set()
            await asyncio.wait_for(prefill.failed.wait(), 4)
            await asyncio.wait_for(asyncio.to_thread(prefill.owner.thread.join, 3), 4)
            assert prefill.owner.core.stats()["open_handles"] == 0
            assert not prefill.owner.contexts
        finally:
            exit_go.set()
            if pc is not None:
                await pc.close()
            if dc is not None:
                await dc.close()
            try:
                try:
                    await prefill.close()
                except PipelineTimingError as exc:
                    assert exc is prefill.owner.error
            finally:
                await decode.close()

    asyncio.run(run())


def test_owner_timing_error_precedes_later_sender_error(tmp_path):
    async def run():
        model_go, sender_at_write, sender_go = threading.Event(), threading.Event(), threading.Event()
        first_errors = []

        def prefill_engine(config):
            engine = pipeline_engine(config)
            call = engine.model_runner.call

            def gated_call(*args, **kwargs):
                if not model_go.wait(8):
                    raise TimeoutError("model gate was not released")
                return call(*args, **kwargs)

            engine.model_runner.call = gated_call
            return engine

        path = tmp_path / "prefill.jsonl"
        config = {"world_size": 1, "enforce_eager": True}
        options = {"max_connections": 4, "max_transfer_bytes": 1_000_000,
                   "request_timeout": 15, "pipeline_frame_bytes": 128}
        decode = PDWorker("decode", config, engine_factory=pipeline_engine, **options)
        de = await decode.start("127.0.0.1", 0)
        prefill = PDWorker("prefill", {**config, "timing_path": str(path)},
                           decode_endpoint=de, engine_factory=prefill_engine, **options)
        pe = await prefill.start("127.0.0.1", 0)
        owner_write, pipeline_write = worker_module.write_record, pipeline_module.write_record
        pc = dc = sender = None
        try:
            dc = await PDClient.open_d(de, lifetime=12)
            pc = await PDClient.prepare_p(
                pe, dc.session_id, "owner-first", [1, 2, 3],
                SamplingParams(temperature=0, max_tokens=4, ignore_eos=True), lifetime=12)
            await PDClient.start_pipeline(pc, dc)
            ctx = next(iter(prefill.owner.contexts.values()))
            sender = ctx.sender
            restored = path.with_name("restored")
            restored.write_text("")

            def gated_sender_write(target, record):
                if record["event"] == "pd_d2h_done" and record.get("role") == "prefill":
                    sender_at_write.set()
                    if not sender_go.wait(8):
                        raise TimeoutError("sender write gate was not released")
                return pipeline_write(target, record)

            def gated_owner_write(target, record):
                if record["event"] != "pd_first":
                    return owner_write(target, record)
                if not sender_at_write.wait(8):
                    raise TimeoutError("sender did not reach timing write")
                path.unlink()
                path.mkdir()
                try:
                    owner_write(target, record)
                except IsADirectoryError as exc:
                    first_errors.append(exc)
                    path.rmdir()
                    fault = path.with_name("fault")
                    fault.symlink_to("/dev/full")
                    fault.replace(path)
                    sender_go.set()
                    with sender.condition:
                        if not sender.condition.wait_for(lambda: sender.error is not None, timeout=8):
                            raise TimeoutError("sender failure did not arrive")
                    restored.replace(path)
                    raise
                raise AssertionError("expected real EISDIR")

            pipeline_module.write_record = gated_sender_write
            worker_module.write_record = gated_owner_write
            model_go.set()
            await asyncio.wait_for(prefill.failed.wait(), 6)
            await asyncio.wait_for(asyncio.to_thread(prefill.owner.thread.join, 4), 5)
            assert len(first_errors) == 1 and first_errors[0].errno == errno.EISDIR
            assert isinstance(sender.error, PipelineTimingError) and sender.error.errno == errno.ENOSPC
            assert sender.error.__cause__.errno == errno.ENOSPC
            assert prefill.owner.error is first_errors[0]
            stats = prefill.owner.core.stats()
            assert stats["open_handles"] == stats["used_blocks"] == stats["reserved_transfer_bytes"] == 0
            assert not prefill.owner.contexts and not sender.thread.is_alive() and not sender.slots
            with pytest.raises(IsADirectoryError) as error:
                await asyncio.wait_for(prefill.close(), 4)
            assert error.value is first_errors[0]
        finally:
            model_go.set()
            sender_go.set()
            if pc is not None:
                await pc.close()
            if dc is not None:
                await dc.close()
            try:
                try:
                    await prefill.close()
                except OSError:
                    pass
            finally:
                await decode.close()
                worker_module.write_record = owner_write
                pipeline_module.write_record = pipeline_write

    asyncio.run(run())
