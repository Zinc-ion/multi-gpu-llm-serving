import asyncio
import json

import pytest

from myvllm.engine.sequence import Sequence
from myvllm.pd.client import PDClient, PDRemoteError
from myvllm.pd.worker import PDWorker
from myvllm.sampling_parameters import SamplingParams
from pd_network_helpers import cpu_engine
from test_pd_core import make_engine
from test_pd_transport import empty


def worker(role, *, target=None, **config):
    return PDWorker(role, {'world_size': 1, **config}, max_connections=4,
                    max_transfer_bytes=1_000_000, request_timeout=5,
                    decode_endpoint=target, engine_factory=cpu_engine)


async def collect(connection):
    events = []
    while not events or not events[-1]['finished']:
        events.append(await connection.next_token())
    return events


def reference(prompt, params):
    engine = make_engine(blocks=12, budget=32, model_length=32)
    seq = Sequence(prompt, 4, params)
    engine.scheduler.add_sequence(seq)
    while not engine.scheduler.is_finished():
        engine.step()
    return seq.completion_token_ids


def test_local_next_during_chunking_and_timing(tmp_path):
    async def run():
        path = tmp_path / 'local.jsonl'
        d = worker('decode', chunked=True, budget=2, delay=.04, timing_path=str(path))
        de = await d.start('127.0.0.1', 0)
        try:
            info = await PDClient.info(de)
            assert info['local_prefill_enabled'] and info['enable_chunked_prefill']
            assert info['world_size'] == 1 and info['decode_endpoint'] is None
            initial = await PDClient.stats(de)
            assert initial['decode_step_wall_ms'] is initial['decode_step_sample_age_ms'] is None
            assert initial['decode_step_sample_count'] == 0
            params = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
            local = await PDClient.open_local(de, 'one', [1, 2, 3, 4, 5, 6, 7, 8, 9], params, lifetime=4)
            try:
                pending = asyncio.create_task(local.next_token())
                await asyncio.sleep(.01)
                assert not pending.done()
                event = await pending
                assert event['finished'] and event['completion_tokens'] == 1
                assert event['token'] == reference([1, 2, 3, 4, 5, 6, 7, 8, 9], params)[0]
                stats = await PDClient.stats(de)
                assert stats['decode_step_sample_count'] == 0
                assert stats['queued_events'] == 0
            finally:
                await local.close()
            await empty(de)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            assert {'pd_admitted', 'pd_first', 'pd_finished', 'pd_released'} <= {r['event'] for r in records}
            assert all(r['route'] == 'local' and r['payload_bytes'] == 0 for r in records)
            assert not any('transfer' in r['event'] or 'restore' in r['event'] for r in records)
        finally:
            await d.close()
    asyncio.run(run())


def test_remote_decode_and_local_chunk_share_actual_batch():
    async def run():
        d = worker('decode', chunked=True, budget=3, sequences=3, delay=.05)
        de = await d.start('127.0.0.1', 0)
        p = worker('prefill', target=de, budget=32, sequences=3)
        pe = await p.start('127.0.0.1', 0)
        remote = prefill = local = None
        try:
            rparams = SamplingParams(temperature=0, max_tokens=12, ignore_eos=True)
            lparams = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
            rprompt, lprompt = [1, 2, 3], list(range(4, 18))
            remote = await PDClient.open_d(de, lifetime=4)
            prefill = await PDClient.open_p(pe, remote.session_id, 'remote', rprompt, rparams, lifetime=4)
            first = await prefill.next_token()
            await PDClient.handoff(prefill, remote, await prefill.describe())
            local = await PDClient.open_local(de, 'local', lprompt, lparams, lifetime=4)
            remote_events, local_events = await asyncio.gather(collect(remote), collect(local))
            assert [first['token']] + [e['token'] for e in remote_events] == reference(rprompt, rparams)
            assert [e['token'] for e in local_events] == reference(lprompt, lparams)
            calls = d.owner.engine.model_runner.calls
            seqids = {ctx.handle.request_id: ctx.handle.sequence.seq_id for ctx in d.owner.contexts.values()}
            rseq, lseq = seqids['remote'], seqids['local']
            mixed = [(a, b) for a, b in zip(calls, calls[1:])
                     if a[0] == rseq and b[0] == lseq and a[-1] and b[-1]]
            assert mixed and all(a[1] >= len(rprompt) for a, _ in mixed)
            assert any(b[1] < len(lprompt) for _, b in mixed)
            stats = await PDClient.stats(de)
            assert stats['decode_step_sample_count'] > 0
            assert stats['decode_step_wall_ms'] > 0 and stats['decode_step_sample_age_ms'] >= 0
            assert stats['decode_step_sample_source'] == 'core.step_wall_decode_including_mixed'
            assert stats['local_resident_requests'] == stats['remote_resident_requests'] == 0
        finally:
            for connection in (prefill, remote, local):
                if connection:
                    await connection.close()
            await empty(pe)
            await empty(de)
            await p.close()
            await d.close()
    asyncio.run(run())


@pytest.mark.parametrize('action', ['cancel', 'disconnect', 'timeout', 'shutdown'])
def test_local_failure_cleanup_and_next_request(action):
    async def run():
        d = worker('decode', chunked=True, budget=2, delay=.06)
        de = await d.start('127.0.0.1', 0)
        local = None
        try:
            params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
            local = await PDClient.open_local(de, 'interrupted', list(range(1, 15)), params,
                                              lifetime=.17 if action == 'timeout' else 4)
            if action == 'cancel':
                assert await local.cancel()
            elif action == 'disconnect':
                local._writer.transport.abort()
            elif action == 'timeout':
                with pytest.raises((TimeoutError, ConnectionError, OSError, asyncio.IncompleteReadError, PDRemoteError)):
                    await local.next_token()
            else:
                await d.close()
                assert not d.owner.thread.is_alive()
                de = await d.start('127.0.0.1', 0)
            await local.close()
            local = None
            await empty(de)
            survivor = await PDClient.open_local(de, 'survivor', [1, 2], params, lifetime=4)
            try:
                events = await collect(survivor)
                assert [e['token'] for e in events] == reference([1, 2], params)
            finally:
                await survivor.close()
            await empty(de)
        finally:
            if local:
                await local.close()
            await d.close()
    asyncio.run(run())


def test_worker_rejects_disabled_local_prefill():
    async def run():
        d = worker('decode')
        de = await d.start('127.0.0.1', 0)
        try:
            assert not (await PDClient.info(de))['local_prefill_enabled']
            with pytest.raises(PDRemoteError, match='chunked'):
                await PDClient.open_local(de, 'bad', [1], SamplingParams(max_tokens=1), lifetime=3)
            await empty(de)
        finally:
            await d.close()
    asyncio.run(run())


def test_local_event_queue_is_bounded_and_cancel_clears_it():
    async def run():
        d = worker('decode', chunked=True, budget=3)
        de = await d.start('127.0.0.1', 0)
        try:
            params = SamplingParams(temperature=0, max_tokens=5, ignore_eos=True)
            local = await PDClient.open_local(de, 'buffered', [1, 2], params, lifetime=4)
            try:
                for _ in range(200):
                    stats = await PDClient.stats(de)
                    if stats['queued_events'] == 5:
                        break
                    await asyncio.sleep(.005)
                assert stats['queued_events'] == 5
                assert stats['open_handles'] == 1 and stats['resident_requests'] == 0
                assert await local.cancel()
                assert (await PDClient.stats(de))['queued_events'] == 0
            finally:
                await local.close()
            await empty(de)
            survivor = await PDClient.open_local(de, 'survivor', [1], params, lifetime=4)
            try:
                assert len(await collect(survivor)) == 5
            finally:
                await survivor.close()
            await empty(de)
        finally:
            await d.close()
    asyncio.run(run())


def test_decode_step_samples_are_bounded_and_age_locally():
    async def run():
        d = worker('decode', chunked=True, budget=3)
        de = await d.start('127.0.0.1', 0)
        try:
            params = SamplingParams(temperature=0, max_tokens=22, ignore_eos=True)
            for index in range(2):
                local = await PDClient.open_local(de, str(index), [1], params, lifetime=4)
                try:
                    assert len(await collect(local)) == 22
                finally:
                    await local.close()
            stats = await PDClient.stats(de)
            assert stats['decode_step_sample_count'] == 32
            assert stats['decode_step_wall_ms'] > 0
            await asyncio.sleep(.01)
            older = await PDClient.stats(de)
            assert older['decode_step_sample_count'] == 32
            assert older['decode_step_sample_age_ms'] > stats['decode_step_sample_age_ms']
            await empty(de)
        finally:
            await d.close()
    asyncio.run(run())
