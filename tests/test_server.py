import json
from concurrent.futures import ThreadPoolExecutor
import socket
import threading
import time

import httpx
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
import torch
from transformers import PreTrainedTokenizerFast
import uvicorn

from myvllm.engine.llm_engine import LLMEngine
from myvllm.server import IncrementalDecoder, create_app


@pytest.fixture
def tokenizer(tmp_path):
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["你好 😀 café\n完成", "user assistant system hello world"],
        trainers.BpeTrainer(
            vocab_size=260, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["[UNK]", "[EOS]"],
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="[EOS]", unk_token="[UNK]",
        chat_template="{% for m in messages %}{{m['role']}}: {{m['content']}}\n{% endfor %}assistant: ",
    )
    tokenizer.save_pretrained(tmp_path)
    return tokenizer


@pytest.mark.parametrize("text", ["你好 😀 café\n完成", "👩‍💻中文𠮷", "a�b\nc", "no spaces"])
def test_incremental_decoding_matches_whole_sequence(tokenizer, text):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    decoder = IncrementalDecoder(tokenizer)
    result = "".join(decoder.push(token, i == len(tokens) - 1) for i, token in enumerate(tokens))
    assert result == tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)


@pytest.fixture
def service(tokenizer, monkeypatch, tmp_path):
    generated = tokenizer.encode("你好 😀 café\n完成", add_special_tokens=False)
    eos = tokenizer.eos_token_id

    class CPUModelRunner:
        """Only replace GPU execution; engine, scheduler and tokenizer are real."""
        def __init__(self, config, rank, event):
            self.timings = None
            self.owner = threading.get_ident()

        def call(self, method, *args):
            assert threading.get_ident() == self.owner
            if method == "exit":
                return
            assert method == "run"
            sequences, _ = args
            time.sleep(0.005)  # Model work gives the socket time to disconnect.
            return torch.tensor([
                generated[seq.num_completion_tokens] if seq.num_completion_tokens < len(generated) else eos
                for seq in sequences
            ])

    monkeypatch.setattr("myvllm.engine.llm_engine.ModelRunner", CPUModelRunner)
    config = {
        "model_name_or_path": str(tmp_path), "world_size": 1, "block_size": 4,
        "max_num_sequences": 2, "max_num_batched_tokens": 256,
        "max_model_length": 128, "max_cached_blocks": 64, "eos": [eos],
    }
    app = create_app(config, "tiny", engine_factory=LLMEngine)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10, trust_env=False) as client:
            yield client, app, tokenizer, generated
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        assert not thread.is_alive()


def stream_chunks(client, path, body):
    with client.stream("POST", path, json=body, headers={"x-request-id": "test-id"}) as response:
        assert response.status_code == 200
        assert response.headers["x-request-id"] == "test-id"
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    return [json.loads(line[6:]) for line in lines[:-1]]


@pytest.mark.parametrize("chat", [False, True])
def test_real_http_stream_counts_usage_and_text(service, chat):
    client, app, tokenizer, generated = service
    body = {
        "model": "tiny", "temperature": 0, "max_tokens": 100, "stream": True,
        "stream_options": {"include_usage": True},
    }
    if chat:
        body["messages"] = [{"role": "user", "content": "hello"}]
    else:
        body["prompt"] = "hello"
    chunks = stream_chunks(client, "/v1/chat/completions" if chat else "/v1/completions", body)
    choices = [chunk["choices"][0] for chunk in chunks if chunk["choices"]]
    usage = chunks[-1]["usage"]
    assert len(choices) == usage["completion_tokens"] == len(generated) + 1
    text = "".join(choice["delta"]["content"] if chat else choice["text"] for choice in choices)
    assert text == tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    assert choices[-1]["finish_reason"] == "stop"
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert not app.state.worker.engine.scheduler.block_manager.used_block_ids


def test_length_and_nonstream_response(service):
    client, _, _, _ = service
    response = client.post("/v1/chat/completions", json={
        "model": "tiny", "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0, "max_tokens": 2,
    })
    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "length"
    assert response.json()["usage"]["completion_tokens"] == 2


@pytest.mark.parametrize("extra", [
    {"model": "wrong"}, {"repetition_penalty": 1.2}, {"logprobs": 1},
    {"n": 2}, {"stop": ["x"]}, {"top_p": 0.5}, {"top_k": 2},
    {"temperature": -1}, {"max_tokens": 0}, {"unknown": True},
    {"prompt": "x" * 256},
])
def test_bad_request_does_not_break_other_requests(service, extra):
    client, _, _, _ = service
    body = {"model": "tiny", "prompt": "hello", "stream": True, "max_tokens": 1, **extra}
    assert client.post("/v1/completions", json=body).status_code == 400
    body = {"model": "tiny", "prompt": "hello", "stream": True, "max_tokens": 1}
    assert client.post("/v1/completions", json=body).status_code == 200


def test_disconnect_cancels_and_returns_all_blocks(service):
    client, app, _, _ = service
    with client.stream("POST", "/v1/completions", json={
        "model": "tiny", "prompt": "hello", "stream": True,
        "max_tokens": 100, "ignore_eos": True,
    }) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                break
    deadline = time.monotonic() + 3
    while app.state.worker.requests:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    manager = app.state.worker.engine.scheduler.block_manager
    assert not manager.used_block_ids
    assert len(manager.free_block_ids) == len(manager.blocks)
    assert all(block.ref_count == 0 and block.hash == -1 for block in manager.blocks)


def test_overlapping_requests_keep_separate_token_counts(service):
    client, app, _, _ = service

    def complete(length):
        return stream_chunks(client, "/v1/completions", {
            "model": "tiny", "prompt": "hello", "stream": True,
            "max_tokens": length, "ignore_eos": True,
            "stream_options": {"include_usage": True},
        })

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(complete, [5, 10]))
    assert [chunks[-1]["usage"]["completion_tokens"] for chunks in results] == [5, 10]
    assert not app.state.worker.engine.scheduler.block_manager.used_block_ids
