import asyncio
from contextlib import asynccontextmanager

import pytest
import torch

from myvllm.pd.client import PDClient
from myvllm.pd.wire import DEFAULT_HEADER_LIMIT, read_header, write_frame
from myvllm.sampling_parameters import SamplingParams
from pd_fp8_helpers import fp8_worker, valid_kv
from pd_pipeline_helpers import empty_pipeline


PARAMS = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)


@asynccontextmanager
async def workers():
    d = fp8_worker("decode")
    de = await d.start("127.0.0.1", 0)
    p = fp8_worker("prefill", de)
    pe = await p.start("127.0.0.1", 0)
    try:
        yield p, d, pe, de
    finally:
        await p.close()
        await d.close()


async def pair(pe, de, prompt, early=True):
    d = await PDClient.open_d(de, lifetime=4)
    try:
        method = PDClient.prepare_p if early else PDClient.open_p
        p = await method(pe, d.session_id, "fp8-wire", prompt, PARAMS, lifetime=4)
    except BaseException:
        await d.close()
        raise
    return p, d


async def survivor(pe, de):
    p, d = await pair(pe, de, [2, 3, 4, 5, 6])
    try:
        await PDClient.start_pipeline(p, d)
        first = await p.next_token()
        await PDClient.handoff_pipeline(p, d, await p.describe())
        tokens = [first["token"]]
        while True:
            event = await d.next_token()
            tokens.append(event["token"])
            if event["finished"]:
                break
        assert len(tokens) == 4
    finally:
        await p.close()
        await d.close()
    await empty_pipeline(pe)
    await empty_pipeline(de)


@pytest.mark.parametrize("length", [3, 4, 5, 11])
def test_real_fp8_pipeline_frames_and_exact_bytes(length, tmp_path):
    async def run():
        async with workers() as (pw, dw, pe, de):
            p, d = await pair(pe, de, list(range(1, length + 1)))
            try:
                early = await p.describe_early()
                assert early["version"] == 5
                await PDClient.start_pipeline(p, d)
                first = await p.next_token()
                copied = await p.wait_pipeline()
                assert copied["data_bytes"] == 2 * length * 6
                assert copied["scale_bytes"] == 2 * length * 8
                assert copied["payload_bytes"] == 2 * length * 14
                assert copied["frames"] >= 2
                meta = await p.describe()
                assert meta["version"] == 4
                assert await d.commit_early(meta)
                ph = next(iter(pw.owner.core._handles.values()))
                dh = next(iter(dw.owner.core._handles.values()))
                assert all(torch.equal(a, b) for a, b in zip(
                    valid_kv(pw.owner.engine, ph.sequence),
                    valid_kv(dw.owner.engine, dh.sequence)))
                await p.acknowledge_early()
                await d.activate()
                tokens = [first["token"]]
                while True:
                    event = await d.next_token()
                    tokens.append(event["token"])
                    if event["finished"]:
                        break
                assert len(tokens) == 4
            finally:
                await p.close()
                await d.close()
            await empty_pipeline(pe)
            await empty_pipeline(de)
    asyncio.run(run())


@pytest.mark.parametrize("fault", ["short", "duplicate", "gap", "eof", "cancel"])
def test_real_fp8_fault_reclaims_and_next_request_succeeds(fault):
    async def run():
        async with workers() as (_, _, pe, de):
            p, d = await pair(pe, de, [1, 2, 3, 4, 5])
            writer = None
            try:
                await d.reserve_early(await p.describe_early())
                if fault == "cancel":
                    assert await d.cancel()
                else:
                    reader, writer = await asyncio.open_connection(*de)
                    await write_frame(writer, {
                        "op": "pipeline", "session_id": d.session_id,
                        "transaction_id": p.transaction_id, "frame_bytes": 28,
                    })
                    response, _ = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
                    assert response["ok"]
                    if fault == "eof":
                        writer.transport.abort()
                    else:
                        header = {"op": "region", "layer": 0, "start": 0, "end": 2}
                        payload = bytes(28)
                        if fault == "duplicate":
                            await write_frame(writer, header, payload)
                            response, _ = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
                            assert response["ok"] and response["value"]["copies_done"] is False
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
                await p.close()
                await d.close()
            await empty_pipeline(pe)
            await empty_pipeline(de)
            await survivor(pe, de)
    asyncio.run(run())
