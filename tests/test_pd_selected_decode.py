import asyncio

import pytest
import torch

from myvllm.pd.client import PDClient, PDRemoteError
from myvllm.pd.worker import PDWorker
from myvllm.sampling_parameters import SamplingParams
from pd_fp8_helpers import fp8_engine, valid_kv
from pd_pipeline_helpers import empty_pipeline
from test_pd_core import reference_tokens


PARAMS = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)


def worker(role, *, default=None, allowed=None):
    return PDWorker(role, {'world_size': 1, 'enforce_eager': True},
                    max_connections=4, max_transfer_bytes=1_000_000, request_timeout=5,
                    decode_endpoint=default, allowed_decode_endpoints=allowed,
                    pipeline_frame_bytes=28, engine_factory=fp8_engine)


@pytest.mark.parametrize('pipeline', [False, True])
def test_selected_nondefault_decode_claim_and_fp8_data_scales(pipeline):
    async def run():
        d0, d1 = worker('decode'), worker('decode')
        de0, de1 = await d0.start('127.0.0.1', 0), await d1.start('127.0.0.1', 0)
        p = worker('prefill', default=de0, allowed=[de0, de1])
        pe = await p.start('127.0.0.1', 0)
        dc = pc = None
        try:
            info = await PDClient.info(pe)
            assert info['decode_endpoint'] == list(de0)
            assert info['allowed_decode_endpoints'] == [list(de0), list(de1)]
            dc = await PDClient.open_d(de1, lifetime=4)
            open_p = PDClient.prepare_p if pipeline else PDClient.open_p
            pc = await open_p(pe, dc.session_id, 'selected', [1, 2, 3, 4, 5], PARAMS,
                              lifetime=4, decode_endpoint=de1)
            if pipeline:
                await PDClient.start_pipeline(pc, dc)
                first = await pc.next_token()
                copied = await pc.wait_pipeline()
                assert copied['data_bytes'] > 0 and copied['scale_bytes'] > 0
                assert copied['payload_bytes'] == copied['data_bytes'] + copied['scale_bytes']
                assert await dc.commit_early(await pc.describe())
            else:
                first = await pc.next_token()
                metadata = await pc.describe()
                assert metadata['version'] == 4
                assert await dc.reserve(metadata)
                assert await dc.restore(await pc.export())
            ph = next(iter(p.owner.core._handles.values()))
            dh = next(iter(d1.owner.core._handles.values()))
            assert all(torch.equal(a, b) for a, b in zip(
                valid_kv(p.owner.engine, ph.sequence), valid_kv(d1.owner.engine, dh.sequence)))
            assert (await PDClient.stats(de0))['open_handles'] == 0
            if pipeline:
                assert await pc.acknowledge_early()
            else:
                assert await pc.acknowledge()
            assert await dc.activate()
            events = [first]
            while not events[-1]['finished']:
                events.append(await dc.next_token())
            assert [event['token'] for event in events] == reference_tokens([1, 2, 3, 4, 5], 'float32')[:4]
        finally:
            if pc:
                await pc.close()
            if dc:
                await dc.close()
            await empty_pipeline(pe)
            await empty_pipeline(de0)
            await empty_pipeline(de1)
            await p.close()
            await d0.close()
            await d1.close()
    asyncio.run(run())


def test_unallowed_endpoint_rejected_before_any_connection_and_valid_followup():
    async def run():
        connected = []
        async def forbidden(reader, writer):
            connected.append(True)
            writer.close()
            await writer.wait_closed()
        trap = await asyncio.start_server(forbidden, '127.0.0.1', 0)
        target = trap.sockets[0].getsockname()[:2]
        d = worker('decode')
        de = await d.start('127.0.0.1', 0)
        p = worker('prefill', default=de)
        pe = await p.start('127.0.0.1', 0)
        dc = None
        try:
            dc = await PDClient.open_d(de, lifetime=4)
            for operation in (PDClient.open_p, PDClient.prepare_p):
                with pytest.raises(PDRemoteError, match='not allowed'):
                    await operation(pe, dc.session_id, 'bad', [1], PARAMS,
                                    lifetime=4, decode_endpoint=target)
            assert not connected
            pc = await PDClient.open_p(pe, dc.session_id, 'good', [1], PARAMS, lifetime=4)
            try:
                assert not (await pc.next_token())['finished']
            finally:
                await pc.close()
                await dc.close()
                dc = None
            await empty_pipeline(pe)
            await empty_pipeline(de)
        finally:
            if dc:
                await dc.close()
            await p.close()
            await d.close()
            trap.close()
            await trap.wait_closed()
    asyncio.run(run())


@pytest.mark.parametrize('default,allowed', [
    (('localhost', 1), []),
    (('localhost', 1), [('localhost', 2)]),
    (('localhost', 1), [('localhost', 1), ('localhost', 1)]),
    (('localhost', 1), [('localhost', True)]),
    (('localhost', 1), ['localhost']),
    (('', 1), None),
    (('localhost', 0), None),
])
def test_invalid_allowlist_fails_before_start(default, allowed):
    with pytest.raises(ValueError):
        worker('prefill', default=default, allowed=allowed)


def test_decode_rejects_prefill_endpoint_options():
    with pytest.raises(ValueError):
        worker('decode', default=('localhost', 1))
    with pytest.raises(ValueError):
        worker('decode', allowed=[('localhost', 1)])


def test_client_sends_optional_target_only_when_requested():
    from myvllm.pd.wire import DEFAULT_HEADER_LIMIT, read_header, write_frame

    async def run():
        opens = []
        async def peer(reader, writer):
            try:
                header, size = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
                assert size == 0
                opens.append(header)
                await write_frame(writer, {'ok': True, 'value': {
                    'session_id': 's', 'transaction_id': 't', 'max_transfer_bytes': 1}})
                header, size = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
                assert header == {'op': 'close'} and size == 0
                await write_frame(writer, {'ok': True, 'value': True})
            finally:
                writer.close()
                await writer.wait_closed()
        server = await asyncio.start_server(peer, '127.0.0.1', 0)
        endpoint = server.sockets[0].getsockname()[:2]
        try:
            for operation in (PDClient.open_p, PDClient.prepare_p):
                default = await operation(endpoint, 's', 'r', [1], PARAMS, lifetime=3)
                await default.close()
                selected = await operation(endpoint, 's', 'r', [1], PARAMS, lifetime=3,
                                           decode_endpoint=('127.0.0.1', 1234))
                await selected.close()
            for default, selected in zip(opens[::2], opens[1::2]):
                assert 'decode_endpoint' not in default
                assert selected['decode_endpoint'] == ['127.0.0.1', 1234]
                assert set(selected) == set(default) | {'decode_endpoint'}
        finally:
            server.close()
            await server.wait_closed()
    asyncio.run(run())
