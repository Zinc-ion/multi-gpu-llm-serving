import asyncio
import json
import time

import pytest
import torch

from myvllm.pd.client import PDClient, PDRemoteError
from myvllm.pd.pipeline import contiguous_regions
from myvllm.pd.wire import DEFAULT_HEADER_LIMIT, read_body, read_header, write_frame
from myvllm.pd.worker import PDWorker
from myvllm.sampling_parameters import SamplingParams
from pd_pipeline_helpers import (
    empty_pipeline, pipeline_request, pipeline_worker, pipeline_workers, prepare_pair,
)
from test_pd_core import reference_tokens


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("length", [1, 3, 4, 5, 11])
def test_live_layer_frames_budget_kv_and_decode(dtype, length):
    async def run():
        async with pipeline_workers(dtype=dtype) as (pw, dw, pe, de):
            prompt = list(range(1, length + 1))
            p, d = await prepare_pair(pe, de, prompt, SamplingParams(
                temperature=0, max_tokens=6, ignore_eos=True, max_model_length=16))
            try:
                assert pw.owner.engine.model_runner.calls == []
                early = await p.describe_early()
                assert await d.reserve_early(early)
                assert not await d.reserve_early(early)
                assert pw.owner.engine.model_runner.calls == []
                stats = await PDClient.stats(de)
                assert stats["used_blocks"] == (length + 3) // 4
                assert stats["reserved_blocks"] == (min(length + 6, 16) + 3) // 4
                assert await p.start()
                first = await p.next_token()
                copied = await p.wait_pipeline()
                expected_bytes = 2 * 2 * length * 1 * 4 * getattr(torch, dtype).itemsize
                assert copied["payload_bytes"] == expected_bytes
                assert copied["frames"] >= 2
                assert not (await PDClient.stats(pe))["buffered_bytes"]
                stats = await PDClient.stats(de)
                assert stats["pipeline_slots"] == 2
                assert stats["pipeline_staging_bytes"] == 128
                assert stats["pipeline_pinned_bytes"] == 0
                assert stats["pipeline_staging_bytes"] <= stats["pipeline_staging_limit_bytes"]
                metadata = await p.describe()
                assert await d.commit_early(metadata)
                assert not await d.commit_early(metadata)
                ph = next(iter(pw.owner.core._handles.values()))
                dh = next(iter(dw.owner.core._handles.values()))
                for pos in range(length):
                    source = pw.owner.engine.model_runner.kv_cache[
                        :, :, ph.sequence.block_table[pos // 4], pos % 4]
                    target = dw.owner.engine.model_runner.kv_cache[
                        :, :, dh.sequence.block_table[pos // 4], pos % 4]
                    assert torch.equal(source, target)
                assert await p.acknowledge_early()
                assert not await p.acknowledge_early()
                assert await d.activate()
                tokens = [first["token"]]
                while True:
                    event = await d.next_token()
                    tokens.append(event["token"])
                    if event["finished"]:
                        break
                assert tokens == reference_tokens(prompt, dtype)
                calls = dw.owner.engine.model_runner.calls
                assert calls and calls[0][1] == length
                assert all(end == start + 1 and not prefill for _, start, end, _, prefill in calls)
            finally:
                await p.close()
                await d.close()
            await empty_pipeline(pe)
            await empty_pipeline(de)
    asyncio.run(run())


def test_contiguous_runs_split_frames_and_nonadjacent_physical_blocks():
    assert list(contiguous_regions([4, 5, 1, 2], 1, 15, 4, 6)) == [
        (1, 7, 17), (7, 8, 23), (8, 14, 4), (14, 15, 10)]


def test_first_layer_sent_before_first_token_and_concurrent_cleanup(tmp_path):
    path = tmp_path / "timing.jsonl"

    async def run():
        async with pipeline_workers(layer_delay=.02, timing_path=str(path)) as (_, _, pe, de):
            prompts = [[1, 2, 3, 4, 5], [6, 7, 8]]
            result = await asyncio.gather(*(
                pipeline_request(pe, de, prompt, request_id=str(i)) for i, prompt in enumerate(prompts)))
            assert result == [reference_tokens(prompt, "float32") for prompt in prompts]
            await empty_pipeline(pe)
            await empty_pipeline(de)
    asyncio.run(run())
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for request_id in ("0", "1"):
        p_events = [r for r in records if r.get("request_id") == request_id and r.get("role") == "prefill"]
        names = [r["event"] for r in p_events]
        assert names.index("pd_region_send_start") < names.index("pd_first")
        sent = [r for r in p_events if r["event"] == "pd_region_sent"]
        assert sent and sent[0]["layer"] == 0
        closed = next(r for r in p_events if r["event"] == "pd_pipeline_sender_closed")
        assert 0 < closed["peak_slots"] <= 2


def test_d_full_budget_rejected_before_p_computes_and_survivor():
    async def run():
        async with pipeline_workers(sequences=1) as (pw, _, pe, de):
            p1, d1 = await prepare_pair(pe, de, [1, 2, 3])
            p2, d2 = await prepare_pair(pe, de, [4, 5])
            try:
                await d1.reserve_early(await p1.describe_early())
                with pytest.raises(PDRemoteError, match="capacity"):
                    await PDClient.start_pipeline(p2, d2)
                assert pw.owner.engine.model_runner.calls == []
            finally:
                await p1.close()
                await d1.close()
                await p2.close()
                await d2.close()
            assert len(await pipeline_request(pe, de, [2, 3])) == 6
            await empty_pipeline(pe)
            await empty_pipeline(de)
    asyncio.run(run())


@pytest.mark.parametrize("fault", ["transaction", "duplicate", "gap", "length", "disconnect", "attach_eof"])
def test_data_fault_is_connection_scoped_and_reclaims(fault):
    async def run():
        async with pipeline_workers() as (_, _, pe, de):
            p, d = await prepare_pair(pe, de, [1, 2, 3, 4, 5])
            writer = None
            try:
                await d.reserve_early(await p.describe_early())
                reader, writer = await asyncio.open_connection(*de)
                await write_frame(writer, {
                    "op": "pipeline", "session_id": d.session_id,
                    "transaction_id": "wrong" if fault == "transaction" else p.transaction_id,
                    "frame_bytes": 64,
                })
                if fault == "attach_eof":
                    writer.transport.abort()
                else:
                    response, size = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
                    assert size == 0
                    if fault == "transaction":
                        assert response["ok"] is False
                    else:
                        assert response["ok"] is True
                        if fault != "disconnect":
                            payload = bytes(64)
                            header = {"op": "region", "layer": 0, "start": 0, "end": 2}
                            if fault == "duplicate":
                                await write_frame(writer, header, payload)
                                response, _ = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
                                assert response["ok"]
                            elif fault == "gap":
                                header.update(start=1, end=3)
                            else:
                                payload = payload[:-1]
                            await write_frame(writer, header, payload)
                            response, _ = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
                            assert response["ok"] is False
            finally:
                if writer:
                    writer.close()
                    await writer.wait_closed()
                if fault != "transaction":
                    await empty_pipeline(de)
                await p.close()
                await d.close()
            await empty_pipeline(pe)
            await empty_pipeline(de)
            assert len(await pipeline_request(pe, de, [4, 5])) == 6
    asyncio.run(run())


@pytest.mark.parametrize("interrupt", ["data_eof", "control_eof", "cancel", "close", "timeout"])
def test_two_slot_backpressure_wakes_without_owner_queue(interrupt):
    async def run():
        received = asyncio.Event()
        release = asyncio.Event()
        data_writers = []

        async def peer(reader, writer):
            try:
                header, _ = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
                await write_frame(writer, {"ok": True, "value": True})
                if header["op"] == "pipeline":
                    data_writers.append(writer)
                    _, size = await read_header(reader, DEFAULT_HEADER_LIMIT, 64)
                    await read_body(reader, size)
                    received.set()
                    await release.wait()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(peer, "127.0.0.1", 0)
        worker = pipeline_worker("prefill", server.sockets[0].getsockname()[:2])
        pe = await worker.start("127.0.0.1", 0)
        p = None
        try:
            p = await PDClient.prepare_p(
                pe, "live-session", "blocked", [1, 2, 3, 4, 5],
                SamplingParams(temperature=0, max_tokens=4, ignore_eos=True),
                lifetime=.6 if interrupt == "timeout" else 4)
            await p.start()
            await asyncio.wait_for(received.wait(), 1)
            sender = next(iter(worker.owner.contexts.values())).sender
            for _ in range(200):
                if sender.stats()["busy_slots"] == 2:
                    break
                await asyncio.sleep(.001)
            assert sender.stats()["busy_slots"] == sender.stats()["peak_slots"] == 2
            assert not sender.done.done()
            started = time.monotonic()
            if interrupt == "data_eof":
                data_writers[0].transport.abort()
                release.set()
            elif interrupt == "control_eof":
                p._writer.transport.abort()
            elif interrupt == "cancel":
                try:
                    await p.cancel()
                except (PDRemoteError, ConnectionError, OSError, asyncio.IncompleteReadError):
                    pass
            elif interrupt == "close":
                await p.close()
            if interrupt == "timeout":
                with pytest.raises((TimeoutError, ConnectionError, OSError, asyncio.IncompleteReadError)):
                    await p.next_token()
            await p.close()
            await empty_pipeline(pe)
            assert time.monotonic() - started < 2, str(sender.error)
            assert sender.error is not None
            assert not sender.thread.is_alive()
            assert worker.owner.thread.is_alive() and worker.owner.error is None
            bulk = await PDClient.open_p(pe, "next-session", "survivor", [2, 3],
                                         SamplingParams(max_tokens=1), lifetime=2)
            try:
                assert (await bulk.next_token())["finished"]
            finally:
                await bulk.close()
        finally:
            release.set()
            if p:
                await p.close()
            await worker.close()
            server.close()
            await server.wait_closed()
    asyncio.run(run())


@pytest.mark.parametrize("chunked,eager", [(True, True), (False, False)])
def test_optimized_worker_rejects_graph_or_chunked_p(chunked, eager):
    with pytest.raises(ValueError, match="whole eager"):
        PDWorker("prefill", {"world_size": 1, "enforce_eager": eager,
                            "enable_chunked_prefill": chunked},
                 max_connections=1, max_transfer_bytes=1000, request_timeout=5,
                 pipeline_frame_bytes=64, decode_endpoint=("127.0.0.1", 1))
