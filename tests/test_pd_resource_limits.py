import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import json
import subprocess
import struct

import pytest

from myvllm.pd.client import PDClient, PDRemoteError
from myvllm.pd.gateway import PDGateway
from myvllm.pd.ready import check
from myvllm.pd.worker import PDWorker
from myvllm.pd.wire import ProtocolError, read_header, write_frame
from myvllm.sampling_parameters import SamplingParams
from test_pd_deployment import arguments
from test_pd_load_deployment import comparison_args, value, worker_args
from test_pd_load_routing import http_gateway, layered_engine, load_engine, records, settings
from pd_pipeline_helpers import pipeline_request
from test_pd_server import body, cpu_gateway_engine, drained, make_config


@asynccontextmanager
async def resources(path, *, aggregate=24 * 1024**3, payload=3 * 1024**3,
                    pairs=1, delay=0, frame=0, engine_factory=cpu_gateway_engine):
    config, _ = make_config(path)
    config['enforce_eager'] = True
    workers, endpoints = [], []
    try:
        for index in range(pairs):
            options = dict(max_connections=8, max_transfer_bytes=aggregate,
                           max_payload_bytes=payload, request_timeout=6,
                           pipeline_frame_bytes=frame, engine_factory=engine_factory)
            node = {**config, 'seed': index, 'delay': delay}
            d = PDWorker('decode', node, **options)
            de = await d.start('127.0.0.1', 0)
            workers.append(d)
            p = PDWorker('prefill', node, decode_endpoint=de, **options)
            pe = await p.start('127.0.0.1', 0)
            workers.append(p)
            endpoints.append((pe, de))
        yield config, endpoints, workers
    finally:
        for worker in reversed(workers):
            await worker.close()


@pytest.mark.parametrize('routing', ['fixed', 'load'])
def test_real_24g_aggregate_3g_payload_startup_and_open(tmp_path, routing):
    async def run():
        async with resources(tmp_path) as (config, pairs, _):
            cfg = {**settings(pairs), 'max_payload_bytes': 3 * 1024**3,
                   'max_staging_bytes': 24 * 1024**3}
            cfg['pairs'] = [{'prefill': list(p), 'decode': list(d)} for p, d in pairs]
            if routing == 'fixed':
                cfg.pop('routing_mode')
            for pe, de in pairs:
                for endpoint in (pe, de):
                    info = await PDClient.info(endpoint)
                    assert info['max_transfer_bytes'] == 24 * 1024**3
                    assert info['max_payload_bytes'] == 3 * 1024**3
                reader, writer = await asyncio.open_connection(*de)
                await write_frame(writer, {'op': 'open_d', 'lifetime': 5})
                header, size = await read_header(reader, 1_000_000, 0)
                assert not size and header['value']['max_transfer_bytes'] == 24 * 1024**3
                assert header['value']['max_payload_bytes'] == 3 * 1024**3
                writer.close()
                await writer.wait_closed()
                decode = await PDClient.open_d(de, lifetime=5)
                reader, writer = await asyncio.open_connection(*pe)
                params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
                await write_frame(writer, {'op': 'open_p', 'lifetime': 5, 'session_id': decode.session_id,
                                           'request_id': 'raw-open', 'token_ids': [1, 2],
                                           'sampling': asdict(params)})
                header, size = await read_header(reader, 1_000_000, 0)
                assert not size and header['value']['max_transfer_bytes'] == 24 * 1024**3
                assert header['value']['max_payload_bytes'] == 3 * 1024**3
                writer.close()
                await writer.wait_closed()
                await decode.close()
            result = await check(cfg, None)
            assert result['pairs'][0]['prefill']['max_payload_bytes'] == 3 * 1024**3
            async with http_gateway(config, cfg, tmp_path) as (client, gateway, _):
                response = await client.post('/v1/completions', json=body(length=3))
                assert response.status_code == 200, response.text
                await drained(gateway, pairs)
    asyncio.run(run())


@asynccontextmanager
async def delayed_exports(endpoint, *, corrupt_metadata=False, legacy=False):
    """Forward real worker frames, applying TCP backpressure before export."""
    gate = asyncio.Event()
    arrivals = asyncio.Queue()
    tasks = set()

    async def forward(reader, writer):
        nonlocal corrupt_metadata
        task = asyncio.current_task()
        tasks.add(task)
        remote_writer = None
        try:
            remote_reader, remote_writer = await asyncio.open_connection(*endpoint)
            while True:
                header, size = await read_header(reader, 1_000_000, 1024 * 1024)
                data = await reader.readexactly(size)
                op = header['op']
                if header['op'] == 'export':
                    arrivals.put_nowait(True)
                    await gate.wait()
                await write_frame(remote_writer, header, data)
                header, size = await read_header(remote_reader, 1_000_000, 1024 * 1024)
                data = await remote_reader.readexactly(size)
                if op == 'describe' and corrupt_metadata:
                    header['value']['kv']['nbytes'] -= 1
                    corrupt_metadata = False
                if legacy and op in ('info', 'open_p'):
                    header['value'].pop('max_payload_bytes')
                await write_frame(writer, header, data)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            if remote_writer is not None:
                remote_writer.close()
                await remote_writer.wait_closed()
            writer.close()
            await writer.wait_closed()
            tasks.discard(task)

    server = await asyncio.start_server(forward, '127.0.0.1', 0)
    address = server.sockets[0].getsockname()[:2]
    try:
        yield address, arrivals, gate
    finally:
        gate.set()
        server.close()
        await server.wait_closed()
        if tasks:
            await asyncio.wait_for(asyncio.gather(*list(tasks), return_exceptions=True), 10)


def fixed_settings(pairs, payload, staging):
    return {'pairs': [{'prefill': list(p), 'decode': list(d)} for p, d in pairs],
            'max_active_requests': 8, 'max_payload_bytes': payload,
            'max_staging_bytes': staging, 'max_request_bytes': 4096, 'request_timeout': 5}


def test_concurrent_http_stages_actual_metadata_bytes(tmp_path):
    async def run():
        async with resources(tmp_path, aggregate=2048, payload=1024) as (config, pairs, _):
            async with delayed_exports(pairs[0][0]) as (proxy, arrivals, gate):
                cfg = fixed_settings([(proxy, pairs[0][1])], 1024, 1024)
                async with http_gateway(config, cfg, tmp_path) as (client, gateway, _):
                    requests = [asyncio.create_task(client.post('/v1/completions', json=body(length=3)))
                                for _ in range(2)]
                    try:
                        await asyncio.wait_for(arrivals.get(), 5)
                        await asyncio.wait_for(arrivals.get(), 5)
                        assert gateway.staging_bytes == 512
                    finally:
                        gate.set()
                    responses = await asyncio.gather(*requests)
                    assert [r.status_code for r in responses] == [200, 200]
                    await drained(gateway, pairs)
                    assert (await client.post('/v1/completions', json=body(length=2))).status_code == 200
                    await drained(gateway, pairs)
    asyncio.run(run())


def test_staging_overflow_and_disconnect_release_http_resources(tmp_path):
    async def run():
        async with resources(tmp_path, aggregate=2048, payload=1024) as (config, pairs, _):
            async with delayed_exports(pairs[0][0]) as (proxy, arrivals, gate):
                cfg = fixed_settings([(proxy, pairs[0][1])], 1024, 1024)
                async with http_gateway(config, cfg, tmp_path) as (client, gateway, port):
                    request = {**body(length=3), 'prompt': ' '.join(['hello'] * 9)}
                    first = asyncio.create_task(client.post('/v1/completions', json=request))
                    try:
                        await asyncio.wait_for(arrivals.get(), 5)
                        assert gateway.staging_bytes == 768
                        second = await client.post('/v1/completions', json=request)
                        assert second.status_code == 503, second.text
                        assert gateway.staging_bytes == 768
                    finally:
                        gate.set()
                    assert (await first).status_code == 200
                    await drained(gateway, pairs)
                    assert (await client.post('/v1/completions', json=body(length=2))).status_code == 200
                    await drained(gateway, pairs)
            # A fresh relay holds an actual in-flight transfer while the HTTP client disconnects.
            async with delayed_exports(pairs[0][0]) as (proxy, arrivals, gate):
                cfg = fixed_settings([(proxy, pairs[0][1])], 1024, 1024)
                async with http_gateway(config, cfg, tmp_path) as (client, gateway, port):
                    reader, writer = await asyncio.open_connection('127.0.0.1', port)
                    data = json.dumps(body(length=4)).encode()
                    writer.write(b'POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n'
                                 b'Content-Type: application/json\r\nContent-Length: '
                                 + str(len(data)).encode() + b'\r\n\r\n' + data)
                    await writer.drain()
                    await asyncio.wait_for(arrivals.get(), 5)
                    assert gateway.staging_bytes == 256
                    writer.transport.abort()
                    await writer.wait_closed()
                    async with asyncio.timeout(5):
                        while gateway.staging_bytes:
                            await asyncio.sleep(.01)
                    gate.set()
                    await drained(gateway, pairs)
                    assert (await client.post('/v1/completions', json=body(length=2))).status_code == 200
                    await drained(gateway, pairs)
    asyncio.run(run())


def test_worker_per_body_and_aggregate_limits_over_real_tcp(tmp_path):
    async def run():
        async with resources(tmp_path, aggregate=512, payload=256) as (_, pairs, _):
            pe, de = pairs[0]
            params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
            ds, ps = [], []
            try:
                for i in range(2):
                    d = await PDClient.open_d(de, lifetime=5)
                    ds.append(d)
                    p = await PDClient.open_p(pe, d.session_id, str(i), [1, 2], params, lifetime=5)
                    ps.append(p)
                stats = await PDClient.stats(pe)
                assert stats['reserved_transfer_bytes'] == 512
                third = await PDClient.open_d(de, lifetime=5)
                ds.append(third)
                with pytest.raises(PDRemoteError, match='P transfer capacity'):
                    await PDClient.open_p(pe, third.session_id, 'overflow', [1, 2], params, lifetime=5)
                assert (await PDClient.stats(pe))['reserved_transfer_bytes'] == 512
            finally:
                await asyncio.gather(*(p.close() for p in ps), *(d.close() for d in ds))
            assert (await PDClient.stats(pe))['reserved_transfer_bytes'] == 0
            d = await PDClient.open_d(de, lifetime=5)
            try:
                with pytest.raises(PDRemoteError, match='P transfer capacity'):
                    await PDClient.open_p(pe, d.session_id, 'too-large', [1, 2, 3, 4, 5],
                                          params, lifetime=5)
                assert (await PDClient.stats(pe))['reserved_transfer_bytes'] == 0
            finally:
                await d.close()
            d = await PDClient.open_d(de, lifetime=5)
            p = await PDClient.open_p(pe, d.session_id, 'later', [1, 2], params, lifetime=5)
            await p.close()
            await d.close()
    asyncio.run(run())


def test_decode_limits_apply_before_reservation_and_allocation(tmp_path):
    async def run():
        config, _ = make_config(tmp_path)
        workers = []
        try:
            d = PDWorker('decode', config, max_connections=8, max_transfer_bytes=512,
                         max_payload_bytes=256, request_timeout=6, engine_factory=cpu_gateway_engine)
            de = await d.start('127.0.0.1', 0)
            workers.append(d)
            p = PDWorker('prefill', config, max_connections=8, max_transfer_bytes=2048,
                         max_payload_bytes=1024, request_timeout=6, decode_endpoint=de,
                         engine_factory=cpu_gateway_engine)
            pe = await p.start('127.0.0.1', 0)
            workers.append(p)
            params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
            ds, ps = [], []
            try:
                for i, prompt in enumerate(([1, 2, 3, 4, 5], [1, 2], [1, 2], [1, 2])):
                    dc = await PDClient.open_d(de, lifetime=5)
                    ds.append(dc)
                    pc = await PDClient.open_p(pe, dc.session_id, str(i), prompt, params, lifetime=5)
                    ps.append(pc)
                    assert not (await pc.next_token())['finished']
                    metadata = await pc.describe()
                    if i in (1, 2):
                        assert await dc.reserve(metadata)
                    else:
                        with pytest.raises(PDRemoteError, match='D transfer capacity'):
                            await dc.reserve(metadata)
                    if i == 0:
                        stats = await PDClient.stats(de)
                        assert stats['reserved_transfer_bytes'] == stats['resident_requests'] == stats['used_blocks'] == 0
                stats = await PDClient.stats(de)
                assert stats['reserved_transfer_bytes'] == 512
                assert stats['used_blocks'] == 0 and stats['resident_requests'] == 2
            finally:
                await asyncio.gather(*(pc.close() for pc in ps), *(dc.close() for dc in ds))
            stats = await PDClient.stats(de)
            assert stats['reserved_transfer_bytes'] == stats['resident_requests'] == stats['used_blocks'] == 0
        finally:
            for worker in reversed(workers):
                await worker.close()
    asyncio.run(run())


def test_response_length_is_checked_before_export_body_read():
    async def run():
        peer_done = asyncio.Event()

        async def peer(reader, writer):
            try:
                header, _ = await read_header(reader, 1_000_000, 0)
                assert header['op'] == 'open_d'
                # Old open responses advertise aggregate as the actual wire limit.
                await write_frame(writer, {'ok': True, 'value': {'session_id': 'old',
                                  'transaction_id': None, 'max_transfer_bytes': 1024}})
                header, _ = await read_header(reader, 1_000_000, 0)
                assert header['op'] == 'export'
                response = json.dumps({'ok': True, 'value': None}).encode()
                writer.write(struct.pack('!IQ', len(response), 512) + response)
                await writer.drain()
                # No body follows; a client which reads first waits until its deadline.
                assert await reader.read() == b''
                peer_done.set()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(peer, '127.0.0.1', 0)
        try:
            endpoint = server.sockets[0].getsockname()[:2]
            connection = await PDClient.open_d(endpoint, lifetime=4)
            with pytest.raises(ProtocolError, match='Unexpected response body size'):
                await asyncio.wait_for(connection.export(expected_bytes=256), 1)
            await asyncio.wait_for(peer_done.wait(), 1)
        finally:
            server.close()
            await server.wait_closed()
    asyncio.run(run())


def test_advertised_wire_limit_rejects_client_body_before_sending(tmp_path):
    async def run():
        async with resources(tmp_path, aggregate=1024, payload=256) as (_, pairs, _):
            connection = await PDClient.open_d(pairs[0][1], lifetime=5)
            with pytest.raises(ProtocolError, match='Body exceeds transfer limit'):
                await connection.restore(bytes(257))
            async with asyncio.timeout(5):
                while (await PDClient.stats(pairs[0][1]))['open_handles']:
                    await asyncio.sleep(.01)
    asyncio.run(run())


def test_pipeline_request_payload_is_bounded_even_when_frames_fit(tmp_path):
    async def run():
        async with resources(tmp_path, aggregate=1024, payload=256, frame=64,
                             engine_factory=layered_engine) as (_, pairs, _):
            pe, de = pairs[0]
            params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
            dc = await PDClient.open_d(de, lifetime=5)
            try:
                with pytest.raises(PDRemoteError, match='P transfer capacity'):
                    await PDClient.prepare_p(pe, dc.session_id, 'too-large', [1, 2, 3, 4, 5],
                                             params, lifetime=5)
            finally:
                await dc.close()
            assert len(await pipeline_request(pe, de, [1, 2], params)) == 3
            assert (await PDClient.stats(pe))['reserved_transfer_bytes'] == 0
            assert (await PDClient.stats(de))['reserved_transfer_bytes'] == 0
    asyncio.run(run())


@pytest.mark.parametrize('routing', ['fixed', 'load'])
def test_legacy_worker_info_uses_aggregate_as_wire_limit(tmp_path, routing):
    async def run():
        async with resources(tmp_path, aggregate=1024, payload=1024) as (config, pairs, _):
            async with delayed_exports(pairs[0][0], legacy=True) as (proxy, _, gate):
                gate.set()
                cfg = {**settings([(proxy, pairs[0][1])]), 'max_payload_bytes': 1024,
                       'max_staging_bytes': 1024}
                cfg['pairs'] = [{'prefill': list(proxy), 'decode': list(pairs[0][1])}]
                if routing == 'fixed':
                    cfg.pop('routing_mode')
                info = await PDClient.info(proxy)
                assert info['max_transfer_bytes'] == 1024 and 'max_payload_bytes' not in info
                await check(cfg, None)
                async with http_gateway(config, cfg, tmp_path) as (client, gateway, _):
                    response = await client.post('/v1/completions', json=body(length=3))
                    assert response.status_code == 200, response.text
                    await drained(gateway, pairs)
                too_small = {**cfg, 'max_payload_bytes': 256}
                with pytest.raises(ValueError, match='worker body limit'):
                    await check(too_small, None)
                with pytest.raises(ValueError, match='worker body limit'):
                    await PDGateway(config, too_small).start()
    asyncio.run(run())


def test_load_routing_checks_each_edge_payload_before_admission(tmp_path):
    async def run():
        config, _ = make_config(tmp_path)
        workers, ds = [], []
        try:
            for index, payload in enumerate((256, 1024)):
                d = PDWorker('decode', {**config, 'seed': index}, max_connections=8,
                             max_transfer_bytes=2048, max_payload_bytes=payload,
                             request_timeout=6, engine_factory=load_engine)
                ds.append(await d.start('127.0.0.1', 0))
                workers.append(d)
            p = PDWorker('prefill', config, max_connections=8, max_transfer_bytes=2048,
                         max_payload_bytes=1024, request_timeout=6, decode_endpoint=ds[0],
                         allowed_decode_endpoints=ds, engine_factory=load_engine)
            pe = await p.start('127.0.0.1', 0)
            workers.append(p)
            pairs = [(pe, de) for de in ds]
            cfg = {**settings(pairs), 'max_payload_bytes': 1024, 'max_staging_bytes': 2048}
            async with http_gateway(config, cfg, tmp_path) as (client, gateway, _):
                request = {**body(length=3), 'prompt': ' '.join(['hello'] * 5)}
                response = await client.post('/v1/completions', json=request)
                assert response.status_code == 200, response.text
                assigned = records(tmp_path / 'gateway.jsonl', 'pd_gateway_assigned')[-1]
                assert tuple(assigned['decode']) == ds[1]
                assert assigned['decision']['estimated_remote_payload_bytes'] == 512
                request['prompt'] = ' '.join(['hello'] * 17)
                response = await client.post('/v1/completions', json=request)
                assert response.status_code == 503 and 'No fresh PD route' in response.text
                await drained(gateway, [(pe, ds[0]), (pe, ds[1])])
        finally:
            for worker in reversed(workers):
                await worker.close()
    asyncio.run(run())


def test_generated_resource_profile_starts_real_workers_and_gateways(tmp_path):
    cli = arguments() + comparison_args() + [
        '--worker-max-connections', '8', '--worker-max-transfer-bytes', str(24 * 1024**3),
        '--worker-max-payload-bytes', str(3 * 1024**3), '--gateway-max-active-requests', '8',
        '--gateway-max-payload-bytes', str(3 * 1024**3),
        '--gateway-max-staging-bytes', str(24 * 1024**3),
    ]
    result = subprocess.run(cli, capture_output=True, text=True, check=True)
    files = json.loads(result.stdout)['files']
    manifest = json.loads(files['manifest.json'])
    assert manifest['resources'] == {
        'worker_max_connections': 8, 'worker_max_transfer_bytes': 24 * 1024**3,
        'worker_max_payload_bytes': 3 * 1024**3, 'gateway_max_active_requests': 8,
        'gateway_max_payload_bytes': 3 * 1024**3, 'gateway_max_staging_bytes': 24 * 1024**3,
    }

    async def run():
        config, _ = make_config(tmp_path)
        for variant in ('', 'load-', 'elastic-'):
            workers, ds, ps = [], [], []
            try:
                for role in ('decode', 'prefill'):
                    for index in range(2):
                        name = f'{variant}{role}-{index}'
                        generated = json.loads(files[f'{name}.json'])
                        args = worker_args(files[f'{name}.sh'])
                        options = dict(
                            max_connections=int(value(args, '--max-connections')),
                            max_transfer_bytes=int(value(args, '--max-transfer-bytes')),
                            max_payload_bytes=int(value(args, '--max-payload-bytes')),
                            request_timeout=float(value(args, '--request-timeout')),
                            pipeline_frame_bytes=int(value(args, '--pipeline-frame-bytes')),
                            engine_factory=load_engine,
                        )
                        assert options['max_connections'] == 8
                        assert options['max_transfer_bytes'] == 24 * 1024**3
                        assert options['max_payload_bytes'] == 3 * 1024**3
                        node = {**config, 'seed': generated['seed'],
                                'chunked': generated['enable_chunked_prefill'],
                                'budget': 3 if generated['enable_chunked_prefill'] else 32,
                                'enforce_eager': generated['enforce_eager']}
                        if role == 'prefill':
                            options['decode_endpoint'] = ds[index]
                            if variant:
                                options['allowed_decode_endpoints'] = ds
                        worker = PDWorker(role, node, **options)
                        endpoint = await worker.start('127.0.0.1', 0)
                        workers.append(worker)
                        (ds if role == 'decode' else ps).append(endpoint)
                pairs = list(zip(ps, ds))
                gateway_names = (['gateway.json', 'fixed-bulk-gateway.json', 'fixed-pipeline-gateway.json']
                                 if not variant else [f'{variant}gateway.json'])
                for gateway_name in gateway_names:
                    cfg = json.loads(files[gateway_name])
                    cfg['pairs'] = [{'prefill': list(pe), 'decode': list(de)} for pe, de in pairs]
                    assert cfg['max_payload_bytes'] == 3 * 1024**3
                    assert cfg['max_staging_bytes'] == 24 * 1024**3
                    await check(cfg, None)
                    async with http_gateway(config, cfg, tmp_path) as (client, gateway, _):
                        response = await client.post('/v1/completions', json=body(length=2))
                        assert response.status_code == 200, (gateway_name, response.text)
                        await drained(gateway, pairs)
            finally:
                for worker in reversed(workers):
                    await worker.close()
    asyncio.run(run())


@pytest.mark.parametrize('options', [
    ['--worker-max-transfer-bytes', '256', '--worker-max-payload-bytes', '512'],
    ['--worker-max-payload-bytes', '1024', '--gateway-max-payload-bytes', '512'],
    ['--gateway-max-payload-bytes', '1024', '--gateway-max-staging-bytes', '512'],
    ['--worker-max-connections', '0'],
    ['--worker-max-transfer-bytes', '32768', '--worker-max-payload-bytes', '32768'],
])
def test_generated_inconsistent_resources_are_rejected(options):
    result = subprocess.run(arguments() + comparison_args() + options, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'resource' in result.stderr or 'payload' in result.stderr or 'pipeline frame' in result.stderr


def test_gateway_rejects_invalid_metadata_before_staging_or_export(tmp_path):
    async def run():
        async with resources(tmp_path, aggregate=2048, payload=1024) as (config, pairs, _):
            async with delayed_exports(pairs[0][0], corrupt_metadata=True) as (proxy, arrivals, gate):
                cfg = fixed_settings([(proxy, pairs[0][1])], 1024, 1024)
                async with http_gateway(config, cfg, tmp_path) as (client, gateway, _):
                    response = await client.post('/v1/completions', json=body(length=3))
                    assert response.status_code == 502 and 'Invalid KV' in response.text
                    assert gateway.staging_bytes == 0 and arrivals.empty()
                    await drained(gateway, pairs)
                    gate.set()
                    assert (await client.post('/v1/completions', json=body(length=2))).status_code == 200
                    await drained(gateway, pairs)
    asyncio.run(run())


def test_pipeline_decode_rejects_logical_payload_over_limit(tmp_path):
    async def run():
        config, _ = make_config(tmp_path)
        config['enforce_eager'] = True
        workers = []
        try:
            d = PDWorker('decode', config, max_connections=4, max_transfer_bytes=1024,
                         max_payload_bytes=256, request_timeout=6, pipeline_frame_bytes=64,
                         engine_factory=layered_engine)
            de = await d.start('127.0.0.1', 0)
            workers.append(d)
            p = PDWorker('prefill', config, max_connections=4, max_transfer_bytes=1024,
                         max_payload_bytes=1024, request_timeout=6, pipeline_frame_bytes=64,
                         decode_endpoint=de, engine_factory=layered_engine)
            pe = await p.start('127.0.0.1', 0)
            workers.append(p)
            params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
            dc = await PDClient.open_d(de, lifetime=5)
            pc = await PDClient.prepare_p(pe, dc.session_id, 'large', [1, 2, 3, 4, 5], params, lifetime=5)
            try:
                with pytest.raises(PDRemoteError, match='D transfer capacity'):
                    await dc.reserve_early(await pc.describe_early())
                stats = await PDClient.stats(de)
                assert stats['reserved_transfer_bytes'] == stats['resident_requests'] == stats['used_blocks'] == 0
            finally:
                await pc.close()
                await dc.close()
            assert len(await pipeline_request(pe, de, [1, 2], params)) == 3
        finally:
            for worker in reversed(workers):
                await worker.close()
    asyncio.run(run())


@pytest.mark.parametrize('payload', [0, -1, True, 1025])
def test_invalid_payload_configuration_is_rejected_at_owner_boundaries(payload):
    from myvllm.pd.core import DecodeCore
    from test_pd_core import make_engine

    with pytest.raises(ValueError, match='max_payload_bytes'):
        DecodeCore(make_engine(), 4, 1024, payload)
    with pytest.raises(ValueError, match='max_payload_bytes'):
        PDWorker('decode', {'world_size': 1}, max_connections=4, max_transfer_bytes=1024,
                 max_payload_bytes=payload, request_timeout=5)


def test_default_payload_and_pipeline_frame_configuration():
    from myvllm.pd.core import DecodeCore
    from test_pd_core import make_engine

    assert DecodeCore(make_engine(), 4, 1024).max_payload_bytes == 1024
    worker = PDWorker('decode', {'world_size': 1}, max_connections=4,
                      max_transfer_bytes=1024, request_timeout=5)
    assert worker.max_payload_bytes == 1024
    with pytest.raises(ValueError, match='pipeline_frame_bytes'):
        PDWorker('decode', {'world_size': 1}, max_connections=4, max_transfer_bytes=1024,
                 max_payload_bytes=256, request_timeout=5, pipeline_frame_bytes=257)
