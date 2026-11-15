"""OpenAI HTTP gateway for fixed pairs of independent PD TCP workers."""

import argparse
import asyncio
import json
import logging
import math
import time
import uuid
from types import SimpleNamespace

from transformers import AutoTokenizer
import uvicorn

from myvllm.config import load_config, resolve_dtype
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.pd.client import PDClient, PDRemoteError, PDTimingError
from myvllm.pd.http import GatewayError
from myvllm.pd.protocol import _ARCHITECTURE_FIELDS
from myvllm.pd.wire import ProtocolError
from myvllm.utils.timing import write_record


logger = logging.getLogger("myvllm.pd.gateway")


def _endpoint(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not isinstance(value[0], str) or not value[0]
            or type(value[1]) is not int or not 0 < value[1] < 65536):
        raise ValueError("Each endpoint must be [host, port]")
    return tuple(value)


def _model_spec(config):
    return {
        "name": config["model_name_or_path"],
        "revision": config["model_revision"],
        "tokenizer_revision": config["tokenizer_revision"],
        "type": config["model_type"],
        "dtype": str(resolve_dtype(config.get("dtype", "float32"))).removeprefix("torch."),
        "architecture": {key: config[key] for key in _ARCHITECTURE_FIELDS if key in config},
        "eos": sorted(set(config["eos"])),
    }


def _request_error(exc, phase):
    if isinstance(exc, GatewayError):
        return exc
    if isinstance(exc, PDTimingError):
        return GatewayError(500, f"PD timing write failed: {exc}")
    if isinstance(exc, TimeoutError) or isinstance(exc, PDRemoteError) and exc.code == "TimeoutError":
        return GatewayError(504, f"PD request timed out during {phase}")
    if isinstance(exc, PDRemoteError):
        if exc.code == "CapacityError" or "capacity" in str(exc).lower():
            return GatewayError(503, str(exc))
        if exc.code == "ValueError" and phase == "open_p":
            return GatewayError(400, str(exc))
        return GatewayError(502, f"PD {phase} failed: {exc}")
    if isinstance(exc, (ConnectionError, OSError, asyncio.IncompleteReadError)):
        return GatewayError(503, f"PD peer unavailable during {phase}: {exc}")
    if isinstance(exc, ProtocolError):
        return GatewayError(502, f"PD protocol error during {phase}: {exc}")
    return GatewayError(502, f"PD {phase} failed: {exc}")


class PDGateway:
    """Own bounded HTTP requests and their two serial network connections."""

    def __init__(self, config, settings):
        self.config = config
        pairs = settings["pairs"]
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("pairs must be a nonempty list")
        self.pairs = []
        for pair in pairs:
            if not isinstance(pair, dict) or set(pair) != {"prefill", "decode"}:
                raise ValueError("Each pair must contain prefill and decode")
            self.pairs.append((_endpoint(pair["prefill"]), _endpoint(pair["decode"])))
        if len(set(p for p, _ in self.pairs)) != len(self.pairs) or len(set(d for _, d in self.pairs)) != len(self.pairs):
            raise ValueError("Each fixed pair must use independent workers")
        for name in ("max_active_requests", "max_payload_bytes", "max_staging_bytes", "max_request_bytes"):
            value = settings[name]
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, value)
        if self.max_staging_bytes < self.max_payload_bytes:
            raise ValueError("max_staging_bytes must fit one max_payload_bytes")
        self.pipeline_min_tokens = settings.get("pipeline_min_tokens", 0)
        if type(self.pipeline_min_tokens) is not int or self.pipeline_min_tokens < 0:
            raise ValueError("pipeline_min_tokens must be a nonnegative integer")
        self.request_timeout = settings["request_timeout"]
        if (type(self.request_timeout) not in (int, float) or not math.isfinite(self.request_timeout)
                or self.request_timeout <= 0):
            raise ValueError("request_timeout must be a positive finite number")
        self.requests = {}
        self.claimed = 0
        self.staging_bytes = 0
        self.next_pair = 0
        self.closing = False
        self.engine = None

    async def start(self):
        if self.config["world_size"] != 1:
            raise ValueError("PD gateway requires world_size == 1")
        spec = _model_spec(self.config)
        seeds = set()
        for pe, de in self.pairs:
            pi, di = await asyncio.gather(PDClient.info(pe), PDClient.info(de))
            for info, role in ((pi, "prefill"), (di, "decode")):
                if (info["role"] != role or info["model"] != spec
                        or info["max_model_length"] != self.config["max_model_length"]):
                    raise ValueError(f"Incompatible {role} worker model or length")
                if info["request_timeout"] < self.request_timeout:
                    raise ValueError(f"{role} worker timeout is shorter than gateway timeout")
                if info["max_transfer_bytes"] > self.max_payload_bytes:
                    raise ValueError(f"{role} worker body limit exceeds gateway payload limit")
            if tuple(pi["decode_endpoint"] or ()) != de or di["decode_endpoint"] is not None:
                raise ValueError("Prefill worker's fixed decode endpoint does not match pair")
            if di["sampling_seed"] != pi["sampling_seed"] ^ 0x80000000 or pi["sampling_seed"] in seeds:
                raise ValueError("P/D seed mismatch or repeated pair seed")
            if self.pipeline_min_tokens and (
                    not pi.get("pipeline_frame_bytes") or
                    pi["pipeline_frame_bytes"] > di.get("pipeline_frame_bytes", 0)):
                raise ValueError("Pipeline requires enabled workers with a compatible frame limit")
            seeds.add(pi["sampling_seed"])
        tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained, self.config["model_name_or_path"],
            revision=self.config["tokenizer_revision"], cache_dir=self.config.get("cache_dir"),
        )
        self.engine = SimpleNamespace(tokenizer=tokenizer)

    def claim(self):
        if self.closing or self.claimed >= self.max_active_requests:
            return False
        self.claimed += 1
        return True

    def unclaim(self):
        self.claimed -= 1

    def submit(self, pending):
        if self.closing:
            raise GatewayError(503, "PD gateway is shutting down")
        if not pending.token_ids:
            raise ValueError("Prompt requires at least one token")
        if len(pending.token_ids) >= self.config["max_model_length"]:
            raise ValueError("Prompt must leave room for completion")
        seq = Sequence(pending.token_ids, self.config["block_size"], pending.params)
        limit = min(pending.params.max_tokens, self.config["max_model_length"] - len(pending.token_ids))
        pending.tokens = asyncio.Queue(maxsize=limit + 1)
        pair = self.pairs[self.next_pair]
        self.next_pair = (self.next_pair + 1) % len(self.pairs)
        entry = SimpleNamespace(
            seq=seq, pending=pending, pair=pair, gateway_id=uuid.uuid4().hex,
            session_id=None, transaction_id=None, frontend_done=False, remote_done=False,
            finishing=False, cancelled=False, task=None,
        )
        self._record(entry, "pd_gateway_assigned", prefill=list(pair[0]), decode=list(pair[1]))
        self.requests[seq.seq_id] = entry
        entry.task = asyncio.create_task(self._run(entry))
        entry.task.add_done_callback(lambda task: self._task_done(entry, task))
        return seq

    def _task_done(self, entry, task):
        # A task cancelled before its coroutine starts cannot run its finally.
        error = None if task.cancelled() else task.exception()
        if error is not None:
            logger.error("PD gateway request cleanup failed", exc_info=error)
        if not entry.remote_done:
            if task.cancelled():
                try:
                    self._record(entry, "pd_gateway_cancelled", phase="before_start")
                except GatewayError:
                    logger.exception("PD gateway cancellation timing failed")
            elif error is not None and not entry.frontend_done:
                entry.pending.tokens.put_nowait(_request_error(error, "cleanup"))
            entry.remote_done = True
            self._release_if_done(entry)

    def timing_ids(self, seq_id):
        entry = self.requests.get(seq_id)
        if entry is None:
            return {}
        return {"gateway_id": entry.gateway_id, "session_id": entry.session_id,
                "transaction_id": entry.transaction_id}

    def _record(self, entry, event, **fields):
        path = self.config.get("timing_path")
        if path:
            try:
                write_record(path, {
                    "event": event, "request_id": entry.pending.request_id,
                    "gateway_id": entry.gateway_id, "session_id": entry.session_id,
                    "transaction_id": entry.transaction_id,
                    "elapsed_ms": (time.perf_counter() - entry.pending.arrival) * 1000,
                    **fields,
                })
            except OSError as exc:
                raise GatewayError(500, f"PD timing write failed: {exc}") from exc

    def _release_if_done(self, entry):
        if entry.frontend_done and entry.remote_done:
            self.requests.pop(entry.seq.seq_id, None)
            self.unclaim()

    def finish(self, seq_id, completed):
        entry = self.requests.get(seq_id)
        if entry is None or entry.frontend_done:
            return
        entry.frontend_done = True
        try:
            self._record(entry, "pd_gateway_frontend_completed" if completed else "pd_gateway_frontend_closed")
        except GatewayError:
            logger.exception("PD gateway frontend timing failed")
        finally:
            if not completed and not entry.finishing and not entry.cancelled:
                entry.cancelled = True
                entry.task.cancel()
            self._release_if_done(entry)

    def _publish(self, entry, event):
        seq = entry.seq
        token = event["token"]
        if event["prompt_tokens"] != seq.num_prompt_tokens or event["completion_tokens"] != seq.num_completion_tokens + 1:
            raise GatewayError(502, "PD worker returned inconsistent token progress")
        seq.append_token(token)
        reason = None
        if not seq.ignore_eos and token in self.config["eos"]:
            reason = "stop"
        elif seq.num_completion_tokens >= seq.max_tokens or seq.num_tokens >= seq.max_model_length:
            reason = "length"
        if event["finished"] != (reason is not None) or event["finish_reason"] != reason:
            raise GatewayError(502, "PD worker returned inconsistent finish state")
        if reason is not None:
            seq.status = SequenceStatus.FINISHED
        if seq.num_completion_tokens in (1, 2):
            self._record(entry, "pd_gateway_first" if seq.num_completion_tokens == 1 else "pd_gateway_second")
        if not event["finished"]:
            entry.pending.tokens.put_nowait((token, False, reason))
        return event["finished"]

    async def _run(self, entry):
        prefill = decode = None
        staging = False
        cancelled = False
        error = None
        phase = "open_d"
        params = entry.pending.params
        limit = params.max_model_length or self.config["max_model_length"]
        pipeline = (self.pipeline_min_tokens > 0
                    and len(entry.pending.token_ids) >= self.pipeline_min_tokens
                    and params.max_tokens > 1 and len(entry.pending.token_ids) + 1 < limit)
        deadline = asyncio.get_running_loop().time() + self.request_timeout
        try:
            decode = await PDClient.open_d(entry.pair[1], lifetime=deadline - asyncio.get_running_loop().time())
            entry.session_id = decode.session_id
            phase = "open_p"
            open_prefill = PDClient.prepare_p if pipeline else PDClient.open_p
            prefill = await open_prefill(
                entry.pair[0], decode.session_id, entry.pending.request_id, entry.pending.token_ids,
                entry.pending.params, lifetime=deadline - asyncio.get_running_loop().time(),
            )
            entry.transaction_id = prefill.transaction_id
            self._record(entry, "pd_gateway_open", prefill=list(entry.pair[0]), decode=list(entry.pair[1]),
                         transfer_mode="pipeline" if pipeline else "bulk")
            if pipeline:
                phase = "prepare"
                await PDClient.start_pipeline(prefill, decode)
            phase = "first"
            event = await prefill.next_token()
            terminal = self._publish(entry, event)
            if not terminal:
                phase = "describe"
                metadata = await prefill.describe()
                arch = metadata["model"]["architecture"]
                nbytes = (2 * arch["num_layers"] * metadata["prompt_length"]
                          * arch["num_kv_heads"] * arch["head_dim"]
                          * resolve_dtype(metadata["model"]["dtype"]).itemsize) if pipeline else metadata["kv"]["nbytes"]
                if not pipeline:
                    if nbytes > self.max_payload_bytes or self.staging_bytes + self.max_payload_bytes > self.max_staging_bytes:
                        raise GatewayError(503, "PD gateway transfer capacity exhausted")
                    self.staging_bytes += self.max_payload_bytes
                    staging = True
                phase = "handoff"
                handoff_start = time.perf_counter()
                self._record(entry, "pd_gateway_handoff_start", payload_bytes=nbytes)
                handoff = PDClient.handoff_pipeline if pipeline else PDClient.handoff
                await handoff(prefill, decode, metadata, timing_path=self.config.get("timing_path"))
                self._record(entry, "pd_gateway_handoff_done", payload_bytes=nbytes,
                             handoff_ms=(time.perf_counter() - handoff_start) * 1000)
                if staging:
                    self.staging_bytes -= self.max_payload_bytes
                    staging = False
                phase = "decode"
                while not terminal:
                    event = await decode.next_token()
                    terminal = self._publish(entry, event)
            self._record(entry, "pd_gateway_completed", completion_tokens=entry.seq.num_completion_tokens)
        except asyncio.CancelledError:
            cancelled = True
            try:
                self._record(entry, "pd_gateway_cancelled", phase=phase)
            except GatewayError:
                logger.exception("PD gateway cancellation timing failed")
        except Exception as exc:
            error = _request_error(exc, phase)
            try:
                self._record(entry, "pd_gateway_failed", phase=phase, status=error.status, reason=str(error))
            except GatewayError:
                logger.exception("PD gateway failure timing failed")
        finally:
            entry.finishing = True
            results = await asyncio.gather(*(conn.close() for conn in (prefill, decode) if conn is not None),
                                           return_exceptions=True)
            if staging:
                self.staging_bytes -= self.max_payload_bytes
            try:
                self._record(entry, "pd_gateway_released", close_errors=[str(r) for r in results if isinstance(r, BaseException)],
                             staging_bytes=self.staging_bytes)
            except GatewayError as exc:
                logger.exception("PD gateway release timing failed")
                if error is None and not cancelled:
                    error = exc
            finally:
                if error is not None:
                    entry.pending.tokens.put_nowait(error)
                elif not cancelled:
                    entry.pending.tokens.put_nowait((event["token"], True, event["finish_reason"]))
                entry.remote_done = True
                self._release_if_done(entry)

    async def close(self):
        self.closing = True
        entries = list(self.requests.values())
        for entry in entries:
            if not entry.frontend_done and not entry.finishing and not entry.cancelled:
                entry.pending.tokens.put_nowait(GatewayError(503, "PD gateway is shutting down"))
                entry.cancelled = True
                entry.task.cancel()
        tasks = [entry.task for entry in entries]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def main():
    from myvllm.server import create_app

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gateway-config", required=True)
    parser.add_argument("--served-model-name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    with open(args.gateway_config, encoding="utf-8") as file:
        settings = json.load(file)
    gateway = PDGateway(config, settings)
    uvicorn.run(create_app(config, args.served_model_name, pd_worker=gateway), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
