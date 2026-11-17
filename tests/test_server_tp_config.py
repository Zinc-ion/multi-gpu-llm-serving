"""The HTTP owner can pass a TP2 configuration to the existing TP engine."""

from myvllm.server import create_app


def test_colocated_http_accepts_tp2_config():
    app = create_app({"world_size": 2, "model_name_or_path": "tp2-checkpoint"})
    assert {"/v1/completions", "/v1/chat/completions"} <= app.openapi()["paths"].keys()
