import asyncio
import json

import httpx
import pytest

from myvllm.pd.gateway import PDGateway
from myvllm.pd.worker import PDWorker
from myvllm.server import create_app
from pd_fp8_helpers import fp8_engine
from pd_pipeline_helpers import LayeredCPUExecution, empty_pipeline
from test_pd_server import body, make_config
from tools.analyze_pd import analyze


def engine_for_gateway(config):
    engine = fp8_engine(config)
    engine.config["model_name_or_path"] = config["model_name_or_path"]
    if config.get("kv_cache_dtype", "auto") == "auto":
        engine.config["kv_cache_dtype"] = "auto"
        engine.model_runner = LayeredCPUExecution(engine.config, 0, None)
    return engine


def settings(pe, de):
    return {"pairs": [{"prefill": pe, "decode": de}], "max_active_requests": 4,
            "max_payload_bytes": 1_000_000, "max_staging_bytes": 1_000_000,
            "max_request_bytes": 4096, "request_timeout": 4, "pipeline_min_tokens": 3}


@pytest.mark.parametrize("mismatch", [False, True])
def test_gateway_fp8_startup_and_actual_byte_accounting(tmp_path, mismatch):
    async def run():
        config, _ = make_config(tmp_path / "tokenizer")
        config.update(head_dim=3, kv_cache_dtype="float8_e4m3fn", enforce_eager=True,
                      timing_path=str(tmp_path / "timing.jsonl"))
        opts = {"max_connections": 4, "max_transfer_bytes": 1_000_000,
                "request_timeout": 5, "pipeline_frame_bytes": 28,
                "engine_factory": engine_for_gateway}
        dconfig = {**config, "kv_cache_dtype": "auto"} if mismatch else config
        dw = PDWorker("decode", dconfig, **opts)
        de = await dw.start("127.0.0.1", 0)
        pw = PDWorker("prefill", config, decode_endpoint=de, **opts)
        pe = await pw.start("127.0.0.1", 0)
        gateway = PDGateway(config, settings(pe, de))
        try:
            if mismatch:
                with pytest.raises(ValueError, match="Incompatible decode worker"):
                    await gateway.start()
                return
            app = create_app(config, "tiny", pd_worker=gateway)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                    for prompt in ("hello token2", "hello token2 token3 token4 token5"):
                        request = body(length=4)
                        request["prompt"] = prompt
                        response = await client.post("/v1/completions", json=request)
                        assert response.status_code == 200, response.text
                assert gateway.staging_bytes == gateway.claimed == 0
            await empty_pipeline(pe)
            await empty_pipeline(de)
        finally:
            await pw.close()
            await dw.close()
    asyncio.run(run())
    if mismatch:
        return
    path = tmp_path / "timing.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    handoffs = [r for r in records if r["event"] == "pd_gateway_handoff_done"]
    assert [(r["data_bytes"], r["scale_bytes"], r["payload_bytes"]) for r in handoffs] == [
        (48, 64, 112), (60, 80, 140)]
    groups = analyze([path])["transactions"]
    assert [(r["data_bytes"], r["scale_bytes"], r["payload_bytes"]) for r in groups] == [
        (48, 64, 112), (60, 80, 140)]
