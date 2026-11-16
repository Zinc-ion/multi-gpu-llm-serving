import asyncio
import importlib.util
import json
from pathlib import Path
import subprocess

import httpx
import pytest

from myvllm.pd.gateway import PDGateway
from myvllm.pd.worker import PDWorker
from myvllm.server import create_app
from pd_pipeline_helpers import empty_pipeline, pipeline_engine
from test_pd_deployment import arguments
from test_pd_server import body, make_config


def gateway_engine(config):
    engine = pipeline_engine(config)
    engine.config["model_name_or_path"] = config["model_name_or_path"]
    return engine


@pytest.mark.parametrize("terminal_eos", [False, True])
def test_gateway_threshold_bulk_pipeline_and_terminal_bytes(tmp_path, terminal_eos):
    async def run():
        config, _ = make_config(tmp_path / "tokenizer")
        config.update(enforce_eager=True, timing_path=str(tmp_path / "timing.jsonl"))
        if terminal_eos:
            config["forced_token"] = 62
        opts = {"max_connections": 4, "max_transfer_bytes": 1_000_000,
                "request_timeout": 5, "pipeline_frame_bytes": 64,
                "engine_factory": gateway_engine}
        dw = PDWorker("decode", config, **opts)
        de = await dw.start("127.0.0.1", 0)
        pw = PDWorker("prefill", config, decode_endpoint=de, **opts)
        pe = await pw.start("127.0.0.1", 0)
        settings = {"pairs": [{"prefill": pe, "decode": de}], "max_active_requests": 4,
                    "max_payload_bytes": 1_000_000, "max_staging_bytes": 1_000_000,
                    "max_request_bytes": 4096, "request_timeout": 4, "pipeline_min_tokens": 3}
        gateway = PDGateway(config, settings)
        app = create_app(config, "tiny", pd_worker=gateway)
        try:
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                    for length in (1, 4):
                        for stream in (False, True):
                            request = body(
                                length=length, stream=stream, ignore_eos=not terminal_eos,
                            )
                            request["prompt"] = "hello token2 token3 token4 token5"
                            response = await client.post("/v1/completions", json=request)
                            assert response.status_code == 200, response.text
                            if stream:
                                assert response.text.endswith("data: [DONE]\n\n")
                            else:
                                assert response.json()["usage"]["completion_tokens"] == (
                                    1 if terminal_eos else length)
                    response = await client.post("/v1/completions", json=body(length=3))
                    assert response.status_code == 200
                assert gateway.staging_bytes == 0
                assert gateway.claimed == 0 and not gateway.requests
            await empty_pipeline(pe)
            await empty_pipeline(de)
        finally:
            await pw.close()
            await dw.close()
    asyncio.run(run())
    path = tmp_path / "timing.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    opened = [r for r in records if r["event"] == "pd_gateway_open"]
    assert [r["transfer_mode"] for r in opened] == ["bulk", "bulk", "pipeline", "pipeline", "bulk"]
    spec = importlib.util.spec_from_file_location("pipeline_analysis", Path(__file__).parents[1] / "tools/analyze_pd.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    groups = module.analyze([path])["transactions"]
    pipelines = [g for g in groups if "pipeline" in g]
    assert len(pipelines) == 2
    assert all(g["payload_bytes"] > 0 and g["pipeline"]["acked_frames"] > 0 for g in pipelines)
    assert all(g["transfer_ms"] is None for g in pipelines)
    assert all(g["pipeline"]["socket_submitted_bytes_lower_bound"] >= g["payload_bytes"] for g in pipelines)
    if terminal_eos:
        assert all(g["handoff_ms"] is None and g["pipeline"]["residual_wait_ms"] is None for g in pipelines)


def test_pipeline_analysis_failure_counts_observed_frames_and_unknown_partial_send(tmp_path):
    ids = {"request_id": "request", "session_id": "s", "transaction_id": "t"}
    rows = [
        {"event": "pd_gateway_open", "gateway_id": "g", **ids, "elapsed_ms": 1},
        {"event": "pd_region_sent", **ids, "layer": 0, "payload_bytes": 64},
        {"event": "pd_pipeline_sender_closed", **ids, "frames": 1, "payload_bytes": 64,
         "socket_submitted_bytes": 128, "partial_send_unknown": True},
        {"event": "pd_gateway_failed", "gateway_id": "g", **ids, "elapsed_ms": 4},
    ]
    path = tmp_path / "timing.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    spec = importlib.util.spec_from_file_location("analysis", Path(__file__).parents[1] / "tools/analyze_pd.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    group = module.analyze([path])["transactions"][0]
    assert group["outcome"] == "failed" and group["payload_bytes"] == 64
    assert group["pipeline"] == {
        "acked_frames": 1, "acked_payload_bytes": 64, "socket_submitted_bytes_lower_bound": 128,
        "partial_send_unknown": True, "residual_wait_ms": None,
    }
    assert group["handoff_ms"] is None


def test_pipeline_deployment_opt_in_and_bounds():
    cmd = arguments("same-numa")
    result = subprocess.run(cmd + ["--pipeline-min-tokens", "1024", "--pipeline-frame-bytes", "1048576"],
                            cwd=Path(__file__).parents[1], capture_output=True, text=True, check=True)
    files = json.loads(result.stdout)["files"]
    assert json.loads(files["gateway.json"])["pipeline_min_tokens"] == 1024
    assert "--pipeline-frame-bytes 1048576" in files["prefill-0.sh"]
    assert "--pipeline-frame-bytes 1048576" in files["decode-0.sh"]
    assert json.loads(files["manifest.json"])["pipeline_frame_bytes"] == 1048576
    for extra in (["--pipeline-min-tokens", "1024"], ["--pipeline-min-tokens", "1024",
                                                     "--pipeline-frame-bytes", "4"]):
        result = subprocess.run(cmd + extra, cwd=Path(__file__).parents[1], capture_output=True, text=True)
        assert result.returncode != 0 and "pipeline" in result.stderr
