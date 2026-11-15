"""Exercise GPU-test cleanup with real CPU processes and stop pipes."""

import multiprocessing as mp
import os
import time

import pytest

from test_pd_pipeline_gpu import _stop_workers


def _broken_worker(stop, ready):
    stop.close()
    ready.send(True)
    ready.close()
    time.sleep(120)


def _normal_worker(stop, ready):
    ready.send(True)
    ready.close()
    try:
        stop.recv()
    finally:
        stop.close()


def _nonzero_worker(stop, ready):
    ready.send(True)
    ready.close()
    stop.close()
    os._exit(17)


def _pipeline_worker_cpu(ready, stop):
    import pd_pipeline_gpu_helpers as pipeline
    from pd_pipeline_helpers import pipeline_engine

    # Keep the actual PDWorker, owner thread, socket and stop pipe; the existing
    # CPU engine replaces only the otherwise GPU-dependent model execution.
    pipeline.PipelineCaptureEngine = pipeline_engine
    pipeline.pipeline_gpu_worker_process("decode", {"world_size": 1}, ready, stop, None)


def _start(target):
    ctx = mp.get_context("spawn")
    stop_read, stop_write = ctx.Pipe(duplex=False)
    ready_read, ready_write = ctx.Pipe(duplex=False)
    process = ctx.Process(target=target, args=(stop_read, ready_write))
    process.start()
    stop_read.close()
    ready_write.close()
    try:
        assert ready_read.poll(30), (process.pid, process.exitcode)
        assert ready_read.recv()
    except BaseException:
        stop_write.close()
        if process.is_alive():
            process.terminate()
            process.join(5)
        raise
    finally:
        ready_read.close()
    return process, stop_write, None


def _reap(items):
    for process, stop, _ in items:
        stop.close()
        if process.is_alive():
            process.terminate()
            process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        assert not process.is_alive()


def test_cleanup_reports_multiple_failures_after_stopping_all_workers():
    items = []
    try:
        items.append(_start(_broken_worker))
        items.append(_start(_nonzero_worker))
        items.append(_start(_normal_worker))
        items[1][0].join(5)
        with pytest.raises(BaseExceptionGroup) as caught:
            _stop_workers(items)
        assert [type(exc) for exc in caught.value.exceptions] == [BrokenPipeError, AssertionError]
        assert [process.exitcode for process, _, _ in items[1:]] == [17, 0]
        assert all(not process.is_alive() and stop.closed for process, stop, _ in items)
    finally:
        _reap(items)


def test_cleanup_keeps_the_original_test_failure():
    items = []
    try:
        items.append(_start(_broken_worker))
        items.append(_start(_normal_worker))
        with pytest.raises(BaseExceptionGroup) as caught:
            try:
                raise ValueError("original test failure")
            finally:
                _stop_workers(items)
        assert [type(exc) for exc in caught.value.exceptions] == [ValueError, BrokenPipeError]
        assert str(caught.value.exceptions[0]) == "original test failure"
        assert items[1][0].exitcode == 0
        assert all(not process.is_alive() and stop.closed for process, stop, _ in items)
    finally:
        _reap(items)


def test_stop_pipe_eof_closes_actual_worker_and_owner_thread():
    ctx = mp.get_context("spawn")
    stop_read, stop_write = ctx.Pipe(duplex=False)
    ready_read, ready_write = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_pipeline_worker_cpu, args=(ready_write, stop_read))
    process.start()
    stop_read.close()
    ready_write.close()
    try:
        assert ready_read.poll(30), (process.pid, process.exitcode)
        message = ready_read.recv()
        assert message[0] == "ready"
        stop_write.close()
        process.join(5)
        assert not process.is_alive(), "PDWorker owner did not exit after stop EOF"
        assert process.exitcode != 0  # The unexpected EOF remains visible.
    finally:
        ready_read.close()
        _reap([(process, stop_write, None)])
