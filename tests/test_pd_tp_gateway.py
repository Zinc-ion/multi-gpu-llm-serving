"""Reject unsupported TP combinations through real gateway TCP entry points."""

import asyncio

import pytest

from myvllm.pd.gateway import PDGateway, _model_spec
from myvllm.pd.ready import check
from myvllm.pd.wire import DEFAULT_HEADER_LIMIT, read_header, write_frame
from test_pd_load_routing import settings
from test_pd_server import make_config


@pytest.mark.parametrize('case', ['tp2-pipeline', 'tp2-advertised-pipeline', 'tp2-decode'])
def test_gateway_and_ready_reject_unsupported_remote_capabilities(tmp_path, case):
    async def run():
        config, _ = make_config(tmp_path)
        config['enforce_eager'] = True
        infos = {}
        async def serve(role, reader, writer):
            try:
                header, length = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
                assert header == {'op': 'info'} and length == 0
                await write_frame(writer, {'ok': True, 'value': infos[role]})
            finally:
                writer.close()
                await writer.wait_closed()
        servers = []
        try:
            for role in ('prefill', 'decode'):
                async def handler(reader, writer, role=role):
                    await serve(role, reader, writer)
                servers.append(await asyncio.start_server(handler, '127.0.0.1', 0))
            pe, de = [server.sockets[0].getsockname()[:2] for server in servers]
            common = {'model': _model_spec(config), 'max_model_length': config['max_model_length'],
                      'max_connections': 4, 'max_transfer_bytes': 1000000, 'max_payload_bytes': 1000000,
                      'max_header_bytes': 10000, 'request_timeout': 30, 'pipeline_frame_bytes': 0,
                      'enforce_eager': True, 'enable_chunked_prefill': False, 'local_prefill_enabled': False}
            infos['prefill'] = {**common, 'role': 'prefill', 'world_size': 2, 'sampling_seed': 0,
                                'decode_endpoint': de, 'allowed_decode_endpoints': [de]}
            infos['decode'] = {**common, 'role': 'decode', 'world_size': 1, 'sampling_seed': 1,
                               'decode_endpoint': None, 'allowed_decode_endpoints': None}
            cfg = settings([(pe, de)], pipeline=1 if case == 'tp2-pipeline' else 0)
            if case == 'tp2-advertised-pipeline':
                infos['prefill']['pipeline_frame_bytes'] = 100
            if case == 'tp2-decode':
                infos['decode']['world_size'] = 2
            gateway = PDGateway(config, cfg)
            try:
                with pytest.raises(ValueError):
                    await gateway.start()
                with pytest.raises(ValueError):
                    await check(cfg, None)
            finally:
                await gateway.close()
        finally:
            for server in servers:
                server.close()
                await server.wait_closed()
    asyncio.run(run())
