"""Single-owner inference thread and OpenAI completion endpoints."""

import argparse
import asyncio
import atexit
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import math
import os
import queue
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import uvicorn

from myvllm.config import load_config
from myvllm.engine.llm_engine import LLMEngine
from myvllm.sampling_parameters import SamplingParams
from myvllm.utils.timing import write_record


logger = logging.getLogger("myvllm.server")


class IncrementalDecoder:
    """Buffer unfinished words, following Transformers TextStreamer."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokens = []
        self.printed = 0

    def push(self, token, finished=False):
        self.tokens.append(token)
        text = self.tokenizer.decode(
            self.tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if finished or text.endswith("\n"):
            end = len(text)
        else:
            # Keep incomplete UTF-8 and unfinished words until they are stable.
            end = text.rfind(" ") + 1
        delta = text[self.printed:end]
        self.printed = end
        if finished or text.endswith("\n"):
            self.tokens.clear()
            self.printed = 0
        return delta


@dataclass
class PendingRequest:
    request_id: str
    body: dict
    chat: bool
    arrival: float
    loop: asyncio.AbstractEventLoop
    admitted: asyncio.Future
    tokens: asyncio.Queue


class EngineThread:
    def __init__(self, config, engine_factory):
        self.config = config
        self.engine_factory = engine_factory
        self.commands = queue.Queue()
        self.requests = {}
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self.run, name="inference")

    def start(self):
        self.thread.start()

    def close(self):
        self.commands.put(("shutdown", None))
        self.thread.join()

    def admit(self, pending):
        body = pending.body
        try:
            if pending.chat:
                prompt = self.engine.tokenizer.apply_chat_template(
                    body["messages"], tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = body["prompt"]
            params = SamplingParams(
                temperature=body.get("temperature", 1.0),
                max_tokens=body.get("max_tokens", 64),
                ignore_eos=body.get("ignore_eos", False),
                max_model_length=self.config["max_model_length"],
            )
            seq = self.engine.add_prompt(prompt, params)
        except ValueError as exc:
            pending.loop.call_soon_threadsafe(self.reject, pending, str(exc))
            return
        self.requests[seq.seq_id] = pending
        if self.config.get("timing_path"):
            self.engine.request_ids[seq.seq_id] = pending.request_id
        pending.loop.call_soon_threadsafe(self.accept, pending, seq)
        record = {
            "event": "admitted", "request_id": pending.request_id,
            "queue_ms": (time.perf_counter() - pending.arrival) * 1000,
            "prompt_tokens": seq.num_prompt_tokens,
        }
        logger.info(json.dumps(record))
        if self.config.get("timing_path"):
            write_record(self.config["timing_path"], record)

    def accept(self, pending, seq):
        if pending.admitted.cancelled():
            self.commands.put(("cancel", seq.seq_id))
        else:
            pending.admitted.set_result(seq)

    @staticmethod
    def reject(pending, message):
        if not pending.admitted.cancelled():
            pending.admitted.set_exception(ValueError(message))

    def publish(self, seq, token):
        pending = self.requests[seq.seq_id]
        reason = None
        if seq.is_finished:
            reason = "stop" if not seq.ignore_eos and token in self.engine.scheduler.eos else "length"
        pending.loop.call_soon_threadsafe(
            pending.tokens.put_nowait, (token, seq.is_finished, reason)
        )
        if seq.is_finished:
            del self.requests[seq.seq_id]
            if self.config.get("timing_path"):
                self.engine.request_ids.pop(seq.seq_id)

    def command(self, command, payload):
        if command == "add":
            self.admit(payload)
        elif command == "cancel":
            self.engine.cancel(payload)
            self.requests.pop(payload, None)
            if self.config.get("timing_path"):
                self.engine.request_ids.pop(payload, None)
        elif command == "shutdown":
            for seq_id in list(self.requests):
                self.engine.cancel(seq_id)
            self.requests.clear()
            return False
        return True

    def run(self):
        try:
            # All Scheduler/ModelRunner calls stay here: context.py is global.
            self.engine = self.engine_factory(self.config)
            self.ready.set()
            running = True
            while running:
                if self.engine.scheduler.is_finished():
                    running = self.command(*self.commands.get())
                while running:
                    try:
                        command = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    running = self.command(*command)
                if running and not self.engine.scheduler.is_finished():
                    self.engine.step(on_tokens=self.publish)
            atexit.unregister(self.engine.exit)
            self.engine.exit()
        except Exception:
            logger.exception("Inference thread failed")
            # An unexpected engine failure invalidates every outstanding stream.
            os._exit(1)


def validate_request(body, model, chat):
    if not isinstance(body, dict):
        raise ValueError("Request body must be an object")
    defaults = {
        "repetition_penalty": 1.0, "logprobs": None, "n": 1,
        "stop": None, "top_p": 1.0, "top_k": -1,
        "presence_penalty": 0, "frequency_penalty": 0,
        "seed": None, "echo": False, "best_of": 1,
    }
    allowed = {
        "model", "messages" if chat else "prompt", "max_tokens", "temperature",
        "ignore_eos", "stream", "stream_options",
    } | defaults.keys()
    for field, value in body.items():
        if field not in allowed:
            raise ValueError(f"Unsupported parameter: {field}")
        if field in defaults and value != defaults[field]:
            raise ValueError(f"Unsupported non-default parameter: {field}")
    if body.get("model") != model:
        raise ValueError(f"model must be {model}")
    if not isinstance(body.get("stream", False), bool):
        raise ValueError("stream must be boolean")
    if chat:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        for message in messages:
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ValueError("Each message must contain role and text content")
            if message["role"] not in ("system", "user", "assistant") or not isinstance(message["content"], str):
                raise ValueError("Unsupported message role or content")
    elif not isinstance(body.get("prompt"), str):
        raise ValueError("prompt must be text")
    max_tokens = body.get("max_tokens", 64)
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    temperature = body.get("temperature", 1.0)
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if type(body.get("ignore_eos", False)) is not bool:
        raise ValueError("ignore_eos must be boolean")
    options = body.get("stream_options")
    if options is None:
        options = {}
    if not isinstance(options, dict) or set(options) - {"include_usage"}:
        raise ValueError("Unsupported stream_options")
    if type(options.get("include_usage", False)) is not bool:
        raise ValueError("include_usage must be boolean")


def create_app(config, model_name=None, engine_factory=LLMEngine):
    if config["world_size"] != 1:
        raise ValueError("HTTP serving requires world_size == 1")
    model_name = model_name or config["model_name_or_path"]
    worker = EngineThread(config, engine_factory)

    @asynccontextmanager
    async def lifespan(app):
        worker.start()
        await asyncio.to_thread(worker.ready.wait)
        try:
            yield
        finally:
            await asyncio.to_thread(worker.close)

    app = FastAPI(lifespan=lifespan)
    app.state.worker = worker

    async def completion(request, chat):
        arrival = time.perf_counter()
        try:
            body = await request.json()
            validate_request(body, model_name, chat)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        pending = PendingRequest(
            request_id, body, chat,
            arrival, loop, loop.create_future(), asyncio.Queue(),
        )
        worker.commands.put(("add", pending))
        try:
            seq = await pending.admitted
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        async def events():
            decoder = IncrementalDecoder(worker.engine.tokenizer)
            completed = False
            first = True
            emitted_tokens = 0
            identifier = f"cmpl-{request_id}"
            created = int(time.time())
            try:
                while not completed:
                    token, completed, reason = await pending.tokens.get()
                    emitted_tokens += 1
                    decode_start = time.perf_counter()
                    text = decoder.push(token, completed)
                    decode_ms = (time.perf_counter() - decode_start) * 1000
                    choice = {"index": 0, "finish_reason": reason}
                    if chat:
                        choice["delta"] = {"content": text}
                        if first:
                            choice["delta"]["role"] = "assistant"
                    else:
                        choice.update(text=text, logprobs=None)
                    payload = {
                        "id": identifier, "object": "chat.completion.chunk" if chat else "text_completion",
                        "created": created, "model": model_name, "choices": [choice],
                    }
                    send_start = time.perf_counter()
                    # The locked bench parser decodes each network chunk as UTF-8.
                    yield "data: " + json.dumps(payload) + "\n\n"
                    sent = time.perf_counter()
                    if first or config.get("timing_path"):
                        record = {
                            "event": "chunk_sent", "request_id": request_id,
                            "first": first, "arrival_to_chunk_ms": (sent - arrival) * 1000,
                            "detokenize_ms": decode_ms, "send_ms": (sent - send_start) * 1000,
                            "completion_tokens": emitted_tokens,
                        }
                        logger.info(json.dumps(record))
                        if config.get("timing_path"):
                            write_record(config["timing_path"], record)
                    first = False
                if (body.get("stream_options") or {}).get("include_usage", False):
                    yield "data: " + json.dumps({
                        "id": identifier, "object": "chat.completion.chunk" if chat else "text_completion",
                        "created": created, "model": model_name, "choices": [],
                        "usage": {
                            "prompt_tokens": seq.num_prompt_tokens,
                            "completion_tokens": seq.num_completion_tokens,
                            "total_tokens": seq.num_tokens,
                        },
                    }) + "\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if not completed:
                    worker.commands.put(("cancel", seq.seq_id))

        if body.get("stream", False):
            return StreamingResponse(
                events(), media_type="text/event-stream",
                headers={"x-request-id": request_id, "cache-control": "no-cache"},
            )
        # EvalScope's fixed OpenAI adapter also uses ordinary completions.
        async def whole_response():
            decoder = IncrementalDecoder(worker.engine.tokenizer)
            parts = []
            completed = False
            try:
                while not completed:
                    token, completed, reason = await pending.tokens.get()
                    parts.append(decoder.push(token, completed))
                choice = {"index": 0, "finish_reason": reason}
                if chat:
                    choice["message"] = {"role": "assistant", "content": "".join(parts)}
                else:
                    choice.update(text="".join(parts), logprobs=None)
                yield json.dumps({
                    "id": f"cmpl-{request_id}", "object": "chat.completion" if chat else "text_completion",
                    "created": int(time.time()), "model": model_name, "choices": [choice],
                    "usage": {"prompt_tokens": seq.num_prompt_tokens,
                              "completion_tokens": seq.num_completion_tokens, "total_tokens": seq.num_tokens},
                })
            finally:
                if not completed:
                    worker.commands.put(("cancel", seq.seq_id))

        # A single JSON body also gets StreamingResponse's disconnect cancellation.
        return StreamingResponse(
            whole_response(), media_type="application/json", headers={"x-request-id": request_id}
        )

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await completion(request, False)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await completion(request, True)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--served-model-name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    logger.info(json.dumps({
        "event": "server_start", "arguments": vars(args), "config": config,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }))
    uvicorn.run(create_app(config, args.served_model_name), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
