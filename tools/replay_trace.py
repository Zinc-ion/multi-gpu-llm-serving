"""Replay a fixed arrival plan against public streaming completions endpoints."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import aiohttp

if __package__:
    from .prepare_trace import AutoTokenizer, file_sha256, json_bytes, load_plan, read_json, sha256
else:
    from prepare_trace import AutoTokenizer, file_sha256, json_bytes, load_plan, read_json, sha256


class ProtocolError(ValueError):
    pass


class RawLimitError(ValueError):
    pass


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def evidence_file(reference, base, scope):
    if reference.get("scope") != scope:
        raise ValueError("evidence reference scope does not match")
    path = Path(reference["path"])
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if file_sha256(path) != reference["sha256"]:
        raise ValueError(f"evidence hash mismatch: {path}")
    return path


def approval(document, base):
    path = evidence_file(document["approval"], base, document["scope"])
    record = read_json(path)
    subject = {key: value for key, value in document.items() if key != "approval"}
    if (record["schema"] != "m8-approval-v1" or record["scope"] != document["scope"]
            or record["status"] != "approved"
            or record["subject_sha256"] != sha256(json_bytes(subject))
            or not isinstance(record["approver"], str) or not record["approver"].strip()):
        raise ValueError("approval does not attest this document")
    when = datetime.fromisoformat(record["approved_at"].replace("Z", "+00:00"))
    if when.tzinfo is None or when > datetime.now(timezone.utc):
        raise ValueError("approval timestamp must be timezone-aware and no later than replay")
    return {"path": str(path), "sha256": file_sha256(path), "approver": record["approver"],
            "approved_at": record["approved_at"]}


def validate_probe_sample(name, sample, scope, base, model, ignore_eos, deployment_hash, tokenizer):
    request_path = evidence_file(sample["request"], base, scope)
    raw_path = evidence_file(sample["raw_sse"], base, scope)
    observation_path = evidence_file(sample["independent_observation"], base, scope)
    request = read_json(request_path)
    observation = read_json(observation_path)
    if (not isinstance(request, dict)
            or set(request) != {"model", "prompt", "n", "echo", "stream", "temperature", "stream_options",
                         "ignore_eos", "max_tokens"}
            or request.get("model") != model or not isinstance(request.get("prompt"), str)
            or type(request.get("n")) is not int or request["n"] != 1
            or request.get("echo") is not False or request.get("stream") is not True
            or type(request.get("temperature")) not in (int, float) or request["temperature"] != 0
            or request.get("stream_options") != {"include_usage": True}
            or type(request.get("ignore_eos")) is not bool or type(request.get("max_tokens")) is not int
            or (name != "eos" and request["ignore_eos"] != ignore_eos)
            or (name == "eos" and request["ignore_eos"] is not False)
            or request["max_tokens"] <= 0 or (name == "single_token" and request["max_tokens"] != 1)
            or not raw_path.stat().st_size or not observation_path.stat().st_size):
        raise ValueError(f"protocol probe evidence is invalid: {name}")
    if (observation["schema"] != "m8-probe-observation-v1" or observation["scope"] != scope
            or observation["case"] != name or observation["deployment_sha256"] != deployment_hash
            or observation["request_sha256"] != file_sha256(request_path)
            or observation["raw_sse_sha256"] != file_sha256(raw_path)
            or observation["endpoint"] != sample["endpoint"]
            or not isinstance(observation["observer"], str) or not observation["observer"].strip()
            or not isinstance(observation["method"], str) or not observation["method"].strip()
            or not isinstance(observation["observations"], list) or not observation["observations"]
            or any(not isinstance(item, dict) or not item for item in observation["observations"])):
        raise ValueError(f"protocol probe observation does not match: {name}")
    if name != "error_and_missing":
        decoder = SSEDecoder()
        try:
            events = list(decoder.feed(raw_path.read_bytes(), 0)) + decoder.finish()
            messages = [json.loads(data) for event, data, _ in events
                        if event == "message" and data != "[DONE]"]
        except (ProtocolError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"protocol probe raw SSE is invalid: {name}") from exc
        if not any(isinstance(message, dict) and message.get("object") == "text_completion"
                   and message.get("model") == model for message in messages):
            raise ValueError(f"protocol probe has no completion event: {name}")
        if name == "ordinary":
            stream = CompletionStream(len(tokenizer.encode(request["prompt"])), request["max_tokens"], model)
            try:
                for event, data, timestamp in events:
                    stream.event(event, data, timestamp)
                stream.finish()
            except ProtocolError as exc:
                raise ValueError("ordinary probe is not a complete successful SSE response") from exc
    return raw_path


def load_contract(path, deployment_path, endpoints, model, ignore_eos, manifest):
    if path is None:
        if deployment_path is not None:
            raise ValueError("deployment evidence requires a protocol contract")
        return None
    if deployment_path is None:
        raise ValueError("protocol contract requires deployment evidence")
    path, deployment_path = Path(path).resolve(), Path(deployment_path).resolve()
    contract, deployment = read_json(path), read_json(deployment_path)
    if contract["schema"] != "m8-protocol-v1" or contract["scope"] not in ("fixture", "formal"):
        raise ValueError("invalid protocol contract schema or scope")
    deployment_hash = file_sha256(deployment_path)
    expected = {"endpoints": endpoints, "model": model, "ignore_eos": ignore_eos,
                "model_revision": manifest["model_revision"],
                "tokenizer_revision": manifest["tokenizer"]["revision"],
                "tokenizer_files_sha256": sha256(json_bytes(manifest["tokenizer"]["files"]))}
    if (deployment["schema"] != "m8-deployment-v1" or deployment["scope"] != contract["scope"]
            or contract["deployment_sha256"] != deployment_hash
            or any(deployment[key] != value for key, value in expected.items())):
        raise ValueError("protocol deployment does not match replay")
    if deployment["generation_mode"] != "standard_non_speculative" or type(deployment["stream_interval"]) is not int or deployment["stream_interval"] < 1:
        raise ValueError("deployment generation mode or stream interval is not supported")
    scope = contract["scope"]
    if scope == "formal" and read_json(manifest["tokenizer"]["provenance_path"]).get("scope") not in (None, "formal"):
        raise ValueError("fixture input provenance cannot be used as formal evidence")
    for category in ("source_files", "config_files"):
        if not deployment[category]:
            raise ValueError(f"deployment requires {category}")
        for reference in deployment[category]:
            if not isinstance(reference.get("role"), str) or not reference["role"].strip():
                raise ValueError(f"deployment {category} requires a role")
            referenced_path = evidence_file(reference, deployment_path.parent, scope)
            if category == "config_files" and referenced_path.suffix == ".json":
                config = read_json(referenced_path)
                if (isinstance(config, dict) and
                        (config.get("scope", scope) != scope or (scope == "formal" and config.get("mode") == "fixture"))):
                    raise ValueError("deployment configuration scope does not match")
    runtime_path = evidence_file(deployment["runtime_provenance"], deployment_path.parent, scope)
    runtime = read_json(runtime_path)
    processes = runtime["processes"]
    if (runtime["schema"] != "m8-runtime-provenance-v1" or runtime["scope"] != scope
            or runtime["endpoints"] != endpoints or runtime["model"] != model
            or runtime["source_files"] != deployment["source_files"]
            or runtime["config_files"] != deployment["config_files"]
            or not isinstance(runtime["observer"], str) or not runtime["observer"].strip()
            or not isinstance(runtime["shared_implementation"], str) or not runtime["shared_implementation"].strip()
            or not isinstance(processes, list) or not processes
            or any(not isinstance(process, dict) or process.get("endpoint") not in endpoints
                   or type(process.get("pid")) is not int or process["pid"] <= 0
                   or not isinstance(process.get("command"), str) or not process["command"].strip()
                   for process in processes)
            or {process["endpoint"] for process in processes} != set(endpoints)):
        raise ValueError("runtime provenance does not match deployment")
    assertions = contract["assertions"]
    if (set(assertions) != {"first_choice", "last_choice", "no_control_choices", "usage_counts_special_tokens"}
            or any(type(value) is not bool for value in assertions.values())):
        raise ValueError("protocol assertions must be the four boolean choice/usage statements")
    review_path = evidence_file(contract["source_review"], path.parent, scope)
    review = read_json(review_path)
    if (review["schema"] != "m8-source-review-v1" or review["scope"] != contract["scope"]
            or review["deployment_sha256"] != deployment_hash
            or review["verified"] is not True or review["assertions"] != assertions
            or not isinstance(review["reviewer"], str) or not review["reviewer"].strip()):
        raise ValueError("source review does not establish this contract")
    probe_path = evidence_file(contract["probe_manifest"], path.parent, scope)
    probe = read_json(probe_path)
    required = {"ordinary", "empty_first", "empty_last", "last_token_finish", "single_token", "eos",
                "merged_choices", "pure_terminal", "error_and_missing"}
    if (probe["schema"] != "m8-protocol-probe-v1" or probe["scope"] != contract["scope"]
            or probe["deployment_sha256"] != deployment_hash
            or probe["verified"] is not True or probe["assertions"] != assertions
            or not required.issubset(probe["cases"])):
        raise ValueError("protocol probe is incomplete or mismatched")
    tokenizer = AutoTokenizer.from_pretrained(manifest["tokenizer"]["path"],
                                               revision=manifest["tokenizer"]["revision"], local_files_only=True)
    uncovered = []
    for name in required:
        case = probe["cases"][name]
        if case["scope"] != scope:
            raise ValueError(f"protocol probe case scope does not match: {name}")
        samples = case["samples"]
        if case["covered"] is False:
            if (name == "ordinary" or samples != [] or not isinstance(case.get("reason"), str)
                    or not case["reason"].strip()):
                raise ValueError(f"uncovered probe case needs a reason: {name}")
            uncovered.append(name)
            continue
        if (case["covered"] is not True or not isinstance(samples, list) or not samples
                or any(not isinstance(sample, dict) or sample.get("endpoint") not in endpoints for sample in samples)):
            raise ValueError(f"invalid protocol probe samples: {name}")
        if name == "ordinary" and {sample["endpoint"] for sample in samples} != set(endpoints):
            raise ValueError("ordinary protocol probe must cover every endpoint")
        raw_paths = [validate_probe_sample(name, sample, scope, probe_path.parent, model, ignore_eos,
                                           deployment_hash, tokenizer)
                     for sample in samples]
        if name == "ordinary":
            for index, raw_path in enumerate(raw_paths):
                if any(raw_path == other and samples[index]["endpoint"] != samples[other_index]["endpoint"]
                       for other_index, other in enumerate(raw_paths[:index])):
                    raise ValueError("ordinary probe endpoints require distinct raw files")
    if review["uncovered_cases"] != sorted(uncovered):
        raise ValueError("source review does not record the uncovered probe cases")
    approved = approval(contract, path.parent) if contract["scope"] == "formal" else None
    return {"scope": contract["scope"], "path": str(path), "sha256": file_sha256(path),
            "deployment_path": str(deployment_path), "deployment_sha256": deployment_hash,
            "assertions": assertions, "approval": approved}


def load_freeze(path):
    if path is None:
        return None
    path = Path(path).resolve()
    freeze = read_json(path)
    if freeze["schema"] != "m8-slo-v1" or freeze["scope"] not in ("fixture", "formal"):
        raise ValueError("invalid SLO freeze")
    if freeze["units"] != "ms" or freeze["single_token_policy"] != "ttft_only":
        raise ValueError("SLO units or single-token policy is unsupported")
    tpot = freeze["tpot_ms"]
    if type(tpot) not in (int, float) or not math.isfinite(tpot) or tpot <= 0:
        raise ValueError("tpot_ms must be positive and finite")
    rules = freeze["ttft_ms_by_prompt_tokens"]
    if not isinstance(rules, list) or not rules:
        raise ValueError("SLO freeze needs TTFT length rules")
    previous = 0
    for index, rule in enumerate(rules):
        if set(rule) != {"max_prompt_tokens", "ttft_ms"}:
            raise ValueError("invalid TTFT rule")
        bound, limit = rule["max_prompt_tokens"], rule["ttft_ms"]
        if bound is None:
            if index != len(rules) - 1:
                raise ValueError("unbounded TTFT rule must be last")
        elif type(bound) is not int or bound <= previous:
            raise ValueError("TTFT bounds must increase")
        else:
            previous = bound
        if type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0:
            raise ValueError("TTFT limits must be positive and finite")
    if rules[-1]["max_prompt_tokens"] is not None:
        raise ValueError("TTFT rules must cover all prompt lengths")
    if not freeze["calibration_files"]:
        raise ValueError("freeze requires calibration evidence")
    for reference in freeze["calibration_files"]:
        calibration_path = evidence_file(reference, path.parent, freeze["scope"])
        if calibration_path.suffix == ".json":
            calibration = read_json(calibration_path)
            if isinstance(calibration, dict) and calibration.get("scope", freeze["scope"]) != freeze["scope"]:
                raise ValueError("calibration evidence scope does not match")
    approved = approval(freeze, path.parent)
    binding = freeze["binding"] if freeze["scope"] == "formal" else None
    return {"scope": freeze["scope"], "path": str(path), "sha256": file_sha256(path), "binding": binding,
            "approval": approved, "ttft_ms_by_prompt_tokens": rules, "tpot_ms": tpot,
            "single_token_policy": freeze["single_token_policy"]}


def validate_freeze_binding(freeze, manifest, warmup_hash, settings, *,
                            plan_manifest_hash=None, warmup_manifest_hash=None):
    if freeze is None or freeze["scope"] != "formal":
        return
    required = {"deployment_sha256", "replay_tool_sha256", "analyze_tool_sha256", "concurrency",
                "request_timeout_s", "batch_timeout_s", "max_raw_bytes", "gpu_count", "ignore_eos"}
    if (not required.issubset(settings) or not plan_manifest_hash
            or (warmup_hash is None) != (warmup_manifest_hash is None)):
        raise ValueError("formal freeze requires deployment, tool, settings and plan manifest hashes")
    binding = freeze["binding"]
    expected = {
        "model_revision": manifest["model_revision"],
        "tokenizer_revision": manifest["tokenizer"]["revision"],
        "tokenizer_files_sha256": sha256(json_bytes(manifest["tokenizer"]["files"])),
        **settings,
    }
    if any(binding[key] != value or (type(value) in (bool, int) and type(binding[key]) is not type(value))
           for key, value in expected.items()):
        raise ValueError("formal freeze does not match the model, tokenizer or client settings")
    pair = {"plan_sha256": manifest["plan_sha256"], "plan_manifest_sha256": plan_manifest_hash,
            "warmup_plan_sha256": warmup_hash, "warmup_plan_manifest_sha256": warmup_manifest_hash}
    if pair not in binding["plans"]:
        raise ValueError("formal freeze does not include this plan and warmup pair")


class SSEDecoder:
    """Decode complete SSE events; all events completed by one read share its timestamp."""

    def __init__(self):
        self.line = bytearray()
        self.pending_cr = None
        self.data = []
        self.event = ""
        self.first_line = True

    def _end_line(self, timestamp):
        line = bytes(self.line).decode("utf-8")
        self.line.clear()
        if self.first_line:
            line = line.removeprefix("\ufeff")
            self.first_line = False
        if not line:
            event = (self.event or "message", "\n".join(self.data), timestamp) if self.data else None
            self.data = []
            self.event = ""
            return event
        if not line.startswith(":"):
            field, separator, value = line.partition(":")
            if separator and value.startswith(" "):
                value = value[1:]
            if field == "data":
                self.data.append(value)
            elif field == "event":
                self.event = value
        return None

    def feed(self, chunk, timestamp):
        for byte in chunk:
            if self.pending_cr is not None:
                event = self._end_line(timestamp if byte == 10 else self.pending_cr)
                self.pending_cr = None
                if event is not None:
                    yield event
                if byte == 10:
                    continue
            if byte == 13:
                self.pending_cr = timestamp
            elif byte == 10:
                event = self._end_line(timestamp)
                if event is not None:
                    yield event
            else:
                self.line.append(byte)

    def finish(self, *, truncated=False):
        events = []
        if self.pending_cr is not None:
            event = self._end_line(self.pending_cr)
            self.pending_cr = None
            if event is not None:
                events.append(event)
        if not truncated and (self.line or self.data):
            raise ProtocolError("incomplete SSE event at EOF")
        return events


class CompletionStream:
    def __init__(self, prompt_tokens, max_tokens, model):
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.model = model
        self.identity = None
        self.choice_times = []
        self.finish_reason = None
        self.finish_time_s = None
        self.usage = None
        self.done = False

    def event(self, event, data, timestamp):
        if event in ("ping", "keepalive"):
            return
        if self.done:
            raise ProtocolError("data after DONE")
        if event != "message":
            raise ProtocolError(f"unexpected SSE event type: {event}")
        if data == "[DONE]":
            self.done = True
            return
        try:
            obj = json.loads(data, object_pairs_hook=unique_object)
        except json.JSONDecodeError as exc:
            raise ProtocolError("invalid JSON event") from exc
        if not isinstance(obj, dict) or "error" in obj:
            raise ProtocolError("error or invalid completion event")
        identity = obj.get("id")
        if not isinstance(identity, str) or not identity or obj.get("object") != "text_completion":
            raise ProtocolError("invalid completion identity or object")
        if obj.get("model") != self.model:
            raise ProtocolError("completion model mismatch")
        if self.identity is None:
            self.identity = identity
        elif self.identity != identity:
            raise ProtocolError("completion identity changed")
        choices = obj.get("choices")
        if not isinstance(choices, list):
            raise ProtocolError("missing completion choices")
        if not choices:
            if self.usage is not None or self.finish_reason is None:
                raise ProtocolError("duplicate or premature usage")
            usage = obj.get("usage")
            if not isinstance(usage, dict) or any(type(usage.get(key)) is not int for key in (
                    "prompt_tokens", "completion_tokens", "total_tokens")):
                raise ProtocolError("invalid usage")
            if (usage["prompt_tokens"] != self.prompt_tokens or not 0 < usage["completion_tokens"] <= self.max_tokens
                    or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
                raise ProtocolError("inconsistent usage")
            self.usage = usage
            return
        if len(choices) != 1 or self.finish_reason is not None or self.usage is not None:
            raise ProtocolError("invalid or late completion choice")
        choice = choices[0]
        if (not isinstance(choice, dict) or type(choice.get("index")) is not int or choice["index"] != 0
                or not isinstance(choice.get("text"), str) or "finish_reason" not in choice
                or obj.get("usage") is not None):
            raise ProtocolError("invalid completion choice")
        reason = choice["finish_reason"]
        if reason is not None and reason not in ("stop", "length"):
            raise ProtocolError("invalid finish reason")
        self.choice_times.append(timestamp)
        if reason is not None:
            self.finish_reason = reason
            self.finish_time_s = timestamp

    def finish(self):
        if not self.done or self.finish_reason is None or self.usage is None or not self.choice_times:
            raise ProtocolError("missing finish, usage, choice or DONE")


def output_event_times(times, count, assertions):
    # M <= N is necessary only when every choice carries tokens and usage
    # includes every generated token; it never proves either endpoint mapping.
    if assertions["no_control_choices"] and assertions["usage_counts_special_tokens"] and len(times) > count:
        return None, None
    return (times[0] if assertions["first_choice"] else None,
            times[-1] if assertions["last_choice"] else None)


def endpoints_valid(endpoints):
    if not endpoints or len(set(endpoints)) != len(endpoints):
        raise ValueError("endpoints must be nonempty and unique")
    for endpoint in endpoints:
        url = urlsplit(endpoint)
        if url.scheme not in ("http", "https") or not url.hostname or url.path not in ("", "/") or url.query or url.fragment or url.username or url.password:
            raise ValueError("each endpoint must be an HTTP(S) base URL")


def run_metadata(path, metadata):
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(json_bytes(metadata))
    os.replace(temporary, path)


async def replay_rows(manifest, rows, directory, session, endpoints, model, ignore_eos,
                      concurrency, request_timeout, batch_deadline, max_raw_bytes, contract, skip_reason=None):
    directory.mkdir()
    raw_dir = directory / "raw"
    raw_dir.mkdir()
    results_path = directory / "results.jsonl"
    start = time.monotonic()
    live = set()
    counts = {"success": 0, "failed": 0, "not_executed": 0}
    stop_reason = skip_reason
    batch_expired = False

    with results_path.open("wb") as results:
        written = set()

        def write_result(result):
            written.add(result["ordinal"])
            counts[result["status"]] += 1
            results.write(json_bytes(result))
            results.flush()

        def initial(row):
            return {
                "ordinal": row["ordinal"], "request_id": row["request_id"], "phase": row["phase"],
                "plan_sha256": manifest["plan_sha256"], "planned_offset_s": row["offset_s"],
                "endpoint": endpoints[row["ordinal"] % len(endpoints)], "dispatch_offset_s": None,
                "queue_delay_s": None, "choice_times_s": [], "first_output_offset_s": None,
                "last_output_offset_s": None, "finish_offset_s": None, "completed_offset_s": None,
                "http_status": None, "finish_reason": None, "usage": None, "done": False,
                "response_id": None, "raw_path": None, "error_class": None, "error": None,
                "raw_bytes": 0, "received_body_bytes": 0, "raw_truncated": False, "content_encoding": None,
            }

        async def request(row):
            nonlocal batch_expired
            result = initial(row)
            raw_path = raw_dir / f"{row['ordinal']:08d}.sse"
            result["raw_path"] = str(raw_path.relative_to(directory))
            parser = SSEDecoder()
            stream = CompletionStream(row["prompt_tokens"], row["output_tokens"], model)
            body = {"model": model, "prompt": row["prompt"], "max_tokens": row["output_tokens"],
                    "temperature": 0, "n": 1, "echo": False, "ignore_eos": ignore_eos,
                    "stream": True, "stream_options": {"include_usage": True}}
            try:
                with raw_path.open("wb") as raw:
                    dispatch = time.monotonic()
                    result["dispatch_offset_s"] = dispatch - start
                    result["queue_delay_s"] = result["dispatch_offset_s"] - row["offset_s"]
                    async with asyncio.timeout_at(min(batch_deadline, dispatch + request_timeout)):
                        async with session.post(result["endpoint"].rstrip("/") + "/v1/completions", json=body,
                                                headers={"x-request-id": row["request_id"],
                                                         "accept-encoding": "identity"},
                                                allow_redirects=False, auto_decompress=False,
                                                timeout=aiohttp.ClientTimeout(total=None)) as response:
                            result["http_status"] = response.status
                            encoding = ",".join(response.headers.getall("Content-Encoding", ["identity"])).strip().lower()
                            result["content_encoding"] = encoding
                            is_sse = response.content_type == "text/event-stream"
                            parse_sse = response.status == 200 and is_sse and encoding == "identity"
                            while True:
                                remaining = max_raw_bytes - result["raw_bytes"]
                                # One extra byte distinguishes a body exactly at the cap
                                # from truncation. It is counted, but never stored.
                                chunk = await response.content.read(min(65536, remaining + 1))
                                received = time.monotonic() - start
                                if not chunk:
                                    break
                                result["received_body_bytes"] += len(chunk)
                                prefix = chunk[:remaining]
                                raw.write(prefix)
                                raw.flush()
                                result["raw_bytes"] += len(prefix)
                                result["raw_truncated"] = len(prefix) < len(chunk)
                                if parse_sse:
                                    for event, data, timestamp in parser.feed(prefix, received):
                                        stream.event(event, data, timestamp)
                                if result["raw_truncated"]:
                                    if parse_sse:
                                        for event, data, timestamp in parser.finish(truncated=True):
                                            stream.event(event, data, timestamp)
                                    raise RawLimitError("raw response exceeds max-raw-bytes; bounded prefix retained")
                            if encoding != "identity":
                                raise ProtocolError(f"unsupported Content-Encoding: {encoding}; raw stores encoded body")
                            if response.status != 200:
                                raise ProtocolError(f"HTTP status {response.status}")
                            if not is_sse:
                                raise ProtocolError("response content type is not text/event-stream")
                            for event, data, timestamp in parser.finish():
                                stream.event(event, data, timestamp)
                            stream.finish()
                result["status"] = "success"
            except asyncio.CancelledError:
                result.update(status="failed", error_class="CancelledError", error="batch stopped")
            except (aiohttp.ClientError, OSError, TimeoutError, UnicodeError, ProtocolError, RawLimitError) as exc:
                expired = isinstance(exc, TimeoutError) and time.monotonic() >= batch_deadline
                batch_expired = batch_expired or expired
                result.update(status="failed", error_class="BatchDeadline" if expired else type(exc).__name__,
                              error=str(exc))
            except Exception as exc:
                result.update(status="failed", error_class=type(exc).__name__, error=str(exc))
                raise
            finally:
                result.update(choice_times_s=stream.choice_times, finish_offset_s=stream.finish_time_s,
                              finish_reason=stream.finish_reason, usage=stream.usage, done=stream.done,
                              response_id=stream.identity, completed_offset_s=time.monotonic() - start)
                if result["status"] == "success" and contract is not None:
                    result["first_output_offset_s"], result["last_output_offset_s"] = output_event_times(
                        stream.choice_times, stream.usage["completion_tokens"], contract["assertions"])
                write_result(result)
            return result

        try:
            for row in (() if skip_reason else rows):
                now = time.monotonic()
                wait = start + row["offset_s"] - now
                while wait > 0 and now < batch_deadline:
                    await asyncio.sleep(min(wait, max(0, batch_deadline - now)))
                    now = time.monotonic()
                    wait = start + row["offset_s"] - now
                while len(live) >= concurrency and time.monotonic() < batch_deadline:
                    done, live = await asyncio.wait(live, timeout=max(0, batch_deadline - time.monotonic()),
                                                    return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                if time.monotonic() >= batch_deadline:
                    stop_reason = "batch_deadline"
                    break
                task = asyncio.create_task(request(row))
                live.add(task)
            if live:
                done, live = await asyncio.wait(live, timeout=max(0, batch_deadline - time.monotonic()))
                for task in done:
                    task.result()
                if live:
                    stop_reason = "batch_deadline"
            if batch_expired:
                stop_reason = "batch_deadline"
        except asyncio.CancelledError:
            stop_reason = "cancelled"
        finally:
            for task in live:
                if not task.done():
                    task.cancel()
            if live:
                await asyncio.gather(*live, return_exceptions=True)
            for row in rows:
                if row["ordinal"] in written:
                    continue
                result = initial(row)
                result.update(status="not_executed", error_class="NotExecuted", error=stop_reason)
                write_result(result)
    return {"plan_sha256": manifest["plan_sha256"], "results_path": str(results_path),
            "results_sha256": file_sha256(results_path), "counts": counts,
            "start_monotonic": start, "end_offset_s": time.monotonic() - start,
            "stop_reason": stop_reason}


async def replay(args):
    endpoints_valid(args.endpoint)
    for name in ("request_timeout_s", "batch_timeout_s"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if args.concurrency <= 0 or args.max_raw_bytes <= 0 or args.gpu_count <= 0:
        raise ValueError("concurrency, max-raw-bytes and gpu-count must be positive")
    manifest, rows = load_plan(args.plan)
    if manifest["kind"] != "formal":
        raise ValueError("main plan must have kind=formal")
    warm_manifest = warm_rows = None
    if args.warmup_plan:
        warm_manifest, warm_rows = load_plan(args.warmup_plan)
        if warm_manifest["kind"] != "warmup":
            raise ValueError("warmup plan must have kind=warmup")
        if (warm_manifest["model_revision"] != manifest["model_revision"]
                or warm_manifest["tokenizer"]["revision"] != manifest["tokenizer"]["revision"]
                or warm_manifest["tokenizer"]["files"] != manifest["tokenizer"]["files"]):
            raise ValueError("warmup and formal model/tokenizer differ")
        if {row["request_id"] for row in rows} & {row["request_id"] for row in warm_rows}:
            raise ValueError("warmup and formal request IDs must be disjoint")
    contract = load_contract(args.protocol_contract, args.deployment, args.endpoint, args.model,
                             args.ignore_eos == "true", manifest)
    if (contract and contract["scope"] == "formal" and warm_manifest
            and read_json(warm_manifest["tokenizer"]["provenance_path"]).get("scope") not in (None, "formal")):
        raise ValueError("fixture warmup provenance cannot be used as formal evidence")
    freeze = load_freeze(args.freeze)
    if freeze and freeze["scope"] == "formal" and (not contract or contract["scope"] != "formal"):
        raise ValueError("formal SLO requires formal protocol evidence")
    plan_manifest_hash = file_sha256(str(args.plan) + ".manifest.json")
    warmup_manifest_hash = file_sha256(str(args.warmup_plan) + ".manifest.json") if warm_manifest else None
    validate_freeze_binding(freeze, manifest, warm_manifest["plan_sha256"] if warm_manifest else None, {
        "deployment_sha256": contract["deployment_sha256"] if contract else None,
        "replay_tool_sha256": file_sha256(__file__),
        "analyze_tool_sha256": file_sha256(Path(__file__).with_name("analyze_trace.py")), "concurrency": args.concurrency,
        "request_timeout_s": args.request_timeout_s, "batch_timeout_s": args.batch_timeout_s,
        "max_raw_bytes": args.max_raw_bytes, "gpu_count": args.gpu_count, "ignore_eos": args.ignore_eos == "true",
    }, plan_manifest_hash=plan_manifest_hash, warmup_manifest_hash=warmup_manifest_hash)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started_wall = datetime.now(timezone.utc).isoformat()
    batch_start = time.monotonic()
    deadline = batch_start + args.batch_timeout_s
    metadata = {
        "schema": "m8-replay-v1", "state": "running", "started_at": started_wall,
        "plan_path": str(Path(args.plan).resolve()), "plan_sha256": manifest["plan_sha256"],
        "plan_manifest_sha256": plan_manifest_hash,
        "warmup_plan_path": str(Path(args.warmup_plan).resolve()) if args.warmup_plan else None,
        "warmup_plan_sha256": warm_manifest["plan_sha256"] if warm_manifest else None,
        "warmup_manifest_sha256": warmup_manifest_hash,
        "endpoints": args.endpoint, "model": args.model, "ignore_eos": args.ignore_eos == "true",
        "concurrency": args.concurrency, "request_timeout_s": args.request_timeout_s,
        "batch_timeout_s": args.batch_timeout_s, "max_raw_bytes": args.max_raw_bytes,
        "gpu_count": args.gpu_count, "protocol_contract": contract, "freeze": freeze,
        "tool_sha256": file_sha256(__file__),
        "budget": {"formal_requests": len(rows), "formal_input_tokens": manifest["input_tokens"],
                   "formal_max_output_tokens": manifest["max_output_tokens"],
                   "warmup_requests": len(warm_rows) if warm_rows is not None else 0,
                   "warmup_input_tokens": warm_manifest["input_tokens"] if warm_manifest else 0,
                   "warmup_max_output_tokens": warm_manifest["max_output_tokens"] if warm_manifest else 0,
                   "total_requests": len(rows) + (len(warm_rows) if warm_rows is not None else 0),
                   "total_input_tokens": manifest["input_tokens"] + (warm_manifest["input_tokens"] if warm_manifest else 0),
                   "total_max_output_tokens": manifest["max_output_tokens"] + (warm_manifest["max_output_tokens"] if warm_manifest else 0)},
    }
    run_metadata(output / "run.json", metadata)
    connector = aiohttp.TCPConnector(limit=args.concurrency, limit_per_host=args.concurrency)
    async with aiohttp.ClientSession(connector=connector, trust_env=False) as session:
        if warm_manifest:
            warm = await replay_rows(warm_manifest, warm_rows, output / "warmup", session, args.endpoint,
                                     args.model, args.ignore_eos == "true", args.concurrency,
                                     args.request_timeout_s, deadline, args.max_raw_bytes, contract)
            metadata["warmup"] = warm
            run_metadata(output / "run.json", metadata)
        else:
            warm = None
        skip_reason = "warmup_failed" if warm and (
            warm["counts"]["failed"] or warm["counts"]["not_executed"] or warm["stop_reason"]) else None
        result = await replay_rows(manifest, rows, output / "formal", session, args.endpoint,
                                   args.model, args.ignore_eos == "true", args.concurrency,
                                   args.request_timeout_s, deadline, args.max_raw_bytes, contract, skip_reason)
    metadata["formal"] = result
    metadata["state"] = ("complete" if not result["counts"]["not_executed"]
                         and not (warm and (warm["counts"]["failed"] or warm["counts"]["not_executed"]))
                         and result["stop_reason"] is None else "incomplete")
    metadata["elapsed_s"] = time.monotonic() - batch_start
    run_metadata(output / "run.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ignore-eos", choices=("true", "false"), required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--request-timeout-s", type=float, required=True)
    parser.add_argument("--batch-timeout-s", type=float, required=True)
    parser.add_argument("--max-raw-bytes", type=int, required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    warmup = parser.add_mutually_exclusive_group(required=True)
    warmup.add_argument("--warmup-plan")
    warmup.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--deployment")
    parser.add_argument("--protocol-contract")
    parser.add_argument("--freeze")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        result = asyncio.run(replay(args))
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["state"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
