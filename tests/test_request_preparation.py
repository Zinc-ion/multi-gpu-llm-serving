from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
from transformers import PreTrainedTokenizerFast

from test_server import service, tokenizer, stream_chunks


@pytest.mark.parametrize("chat", [False, True])
def test_real_request_tokenization_runs_outside_inference_owner(service, monkeypatch, chat):
    client, app, _, _ = service
    calls = []
    encode = PreTrainedTokenizerFast.encode
    template = PreTrainedTokenizerFast.apply_chat_template

    def observed_encode(self, *args, **kwargs):
        calls.append(("encode", threading.get_ident()))
        return encode(self, *args, **kwargs)

    def observed_template(self, *args, **kwargs):
        calls.append(("template", threading.get_ident()))
        return template(self, *args, **kwargs)

    monkeypatch.setattr(PreTrainedTokenizerFast, "encode", observed_encode)
    monkeypatch.setattr(PreTrainedTokenizerFast, "apply_chat_template", observed_template)
    body = {"model": "tiny", "max_tokens": 2}
    if chat:
        body["messages"] = [{"role": "user", "content": "hello"}]
    else:
        body["prompt"] = "hello"
    response = client.post("/v1/chat/completions" if chat else "/v1/completions", json=body)
    assert response.status_code == 200
    assert calls
    assert all(ident != app.state.worker.thread.ident for _, ident in calls)
    assert {name for name, _ in calls} == ({"encode", "template"} if chat else {"encode"})


def test_pending_tokenization_does_not_hold_up_an_existing_stream(service, monkeypatch):
    client, _, _, _ = service
    preparing = threading.Event()
    release = threading.Event()
    first_token = threading.Event()
    finished = threading.Event()
    encode = PreTrainedTokenizerFast.encode

    def delayed_encode(self, text, *args, **kwargs):
        if text == "delayed":
            preparing.set()
            release.wait(5)
        return encode(self, text, *args, **kwargs)

    monkeypatch.setattr(PreTrainedTokenizerFast, "encode", delayed_encode)

    def existing_stream():
        with client.stream("POST", "/v1/completions", json={
            "model": "tiny", "prompt": "hello", "stream": True,
            "max_tokens": 40, "ignore_eos": True,
        }) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if line.startswith("data: "):
                    first_token.set()
        finished.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        active = executor.submit(existing_stream)
        assert first_token.wait(3)
        pending = executor.submit(stream_chunks, client, "/v1/completions", {
            "model": "tiny", "prompt": "delayed", "stream": True, "max_tokens": 2,
        })
        try:
            assert preparing.wait(3)
            assert finished.wait(2), "request preprocessing stalled the inference loop"
        finally:
            release.set()
        active.result()
        assert pending.result()
