import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from tools.prepare_trace import file_sha256, json_bytes, prepare, sha256


REVISION = "a" * 40


def make_plan(path, rows, *, kind="formal", rate=10000, seed=7, prefix="trace", tokenizer=None,
              provenance_scope="fixture"):
    path.mkdir(parents=True)
    tokenizer_path = path / "tokenizer"
    if tokenizer is None:
        backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.save_pretrained(tokenizer_path)
    provenance = path / "source-manifest.json"
    source = path / "m2.jsonl"
    source.write_bytes(b"".join(json_bytes(row) for row in rows))
    source_manifest = Path(str(source) + ".manifest.json")
    source_manifest.write_bytes(json_bytes({
        "sha256": file_sha256(source), "requests": len(rows), "model_revision": REVISION,
        "tokenizer_revision": REVISION, "max_model_length": 128, "source": {"dataset": "sharegpt"},
    }))
    files = {f"tokenizer/{item.name}": {"sha256": file_sha256(item)}
             for item in tokenizer_path.iterdir() if item.is_file()}
    files.update({f"workloads/{source.name}": {"sha256": file_sha256(source)},
                  f"workloads/{source_manifest.name}": {"sha256": file_sha256(source_manifest)}})
    provenance.write_bytes(json_bytes({"scope": provenance_scope, "model_revision": REVISION, "files": files}))
    plan = path / "plan.jsonl"
    args = argparse.Namespace(
        mode="l2", kind=kind, chat=str(source), long=None, tokenizer=str(tokenizer_path),
        tokenizer_provenance=str(provenance), model_revision=REVISION, tokenizer_revision=REVISION,
        seed=seed, id_prefix=prefix, rate=rate, count=len(rows), duration_s=None,
        chat_probability=None, phases=None, max_requests=None, allow_recycle=False, output=str(plan),
    )
    manifest = prepare(args)
    return plan, manifest, tokenizer


def ref(path, scope="fixture", role=None):
    result = {"path": str(path), "sha256": file_sha256(path), "scope": scope}
    if role is not None:
        result["role"] = role
    return result


def fixture_contract(path, manifest, endpoints, *, ignore_eos=True, scope="fixture"):
    path.mkdir(parents=True)
    config = path / "deployment-config.json"
    config.write_bytes(json_bytes({"scope": scope, "mode": "fixture" if scope == "fixture" else "synthetic-test"}))
    source_files = [ref(Path(__file__).resolve(), scope, "engine_source")]
    config_files = [ref(config, scope, "launch_config")]
    runtime = path / "runtime.json"
    runtime.write_bytes(json_bytes({
        "schema": "m8-runtime-provenance-v1", "scope": scope, "endpoints": endpoints, "model": "tiny",
        "source_files": source_files, "config_files": config_files, "observer": "synthetic test fixture",
        "shared_implementation": "synthetic fixtures share the same source and configuration",
        "processes": [{"endpoint": endpoint, "pid": os.getpid(), "command": "synthetic test server"}
                      for endpoint in endpoints],
    }))
    deployment_path = path / "deployment.json"
    deployment_path.write_bytes(json_bytes({
        "schema": "m8-deployment-v1", "scope": scope, "endpoints": endpoints, "model": "tiny", "ignore_eos": ignore_eos,
        "model_revision": manifest["model_revision"], "tokenizer_revision": manifest["tokenizer"]["revision"],
        "tokenizer_files_sha256": sha256(json_bytes(manifest["tokenizer"]["files"])),
        "generation_mode": "standard_non_speculative", "stream_interval": 1,
        "source_files": source_files, "config_files": config_files, "runtime_provenance": ref(runtime, scope),
    }))
    assertions = {"first_choice": True, "last_choice": True,
                  "no_control_choices": True, "usage_counts_special_tokens": True}
    source_review = path / "source-review.json"
    source_review.write_bytes(json_bytes({"schema": "m8-source-review-v1", "scope": scope, "verified": True,
                                         "deployment_sha256": file_sha256(deployment_path),
                                         "assertions": assertions, "reviewer": "test fixture", "uncovered_cases": []}))
    samples = {
        "ordinary": (["a", "b"], "length", 2),
        "empty_first": (["", "x"], "length", 2),
        "empty_last": (["x", ""], "length", 2),
        "last_token_finish": (["a", "b"], "length", 2),
        "single_token": ([""], "length", 1),
        "eos": ([""], "stop", 1),
        "merged_choices": (["a", "bc"], "length", 3),
        "pure_terminal": (["a", ""], "abort", 1),
        "error_and_missing": ([], None, 0),
    }
    cases = {}
    for name, (texts, finish, count) in samples.items():
        case_samples = []
        for endpoint_index, endpoint in enumerate(endpoints):
            stem = name if endpoint_index == 0 else f"{name}-{endpoint_index}"
            request = path / f"{stem}.request.json"
            request.write_bytes(json_bytes({"model": "tiny", "prompt": "hello", "n": 1, "echo": False,
                                           "stream": True, "temperature": 0, "stream_options": {"include_usage": True},
                                           "ignore_eos": False if name == "eos" else ignore_eos,
                                           "max_tokens": max(1, count)}))
            raw, observed = path / f"{stem}.sse", path / f"{stem}.observed.json"
            if texts:
                stream = b"".join(event(choice("synthetic", text, finish if index == len(texts) - 1 else None))
                                  for index, text in enumerate(texts))
                raw.write_bytes(stream + event(usage("synthetic", 1, count)) + b"data: [DONE]\n\n")
            else:
                raw.write_bytes(event({"error": {"message": "synthetic failure"}}))
            observed.write_bytes(json_bytes({"schema": "m8-probe-observation-v1", "scope": scope, "case": name,
                                             "deployment_sha256": file_sha256(deployment_path),
                                             "request_sha256": file_sha256(request), "raw_sse_sha256": file_sha256(raw),
                                             "endpoint": endpoint, "observer": "synthetic test fixture",
                                             "method": "synthetic event/token fixture",
                                             "observations": [{"fixture_token_count": count, "fixture_finish": finish}]}))
            case_samples.append({"endpoint": endpoint, "request": ref(request, scope), "raw_sse": ref(raw, scope),
                                 "independent_observation": ref(observed, scope)})
        cases[name] = {"scope": scope, "covered": True, "samples": case_samples}
    probe = path / "probe.json"
    probe.write_bytes(json_bytes({
        "schema": "m8-protocol-probe-v1", "scope": scope, "deployment_sha256": file_sha256(deployment_path),
        "verified": True, "assertions": assertions, "cases": cases,
    }))
    contract = path / "protocol.json"
    document = {"schema": "m8-protocol-v1", "scope": scope, "deployment_sha256": file_sha256(deployment_path),
                "assertions": assertions, "source_review": ref(source_review, scope),
                "probe_manifest": ref(probe, scope)}
    if scope == "formal":
        approved = path / "protocol-approval.json"
        approved.write_bytes(json_bytes({"schema": "m8-approval-v1", "scope": scope, "status": "approved",
                                        "subject_sha256": sha256(json_bytes(document)),
                                        "approver": "SYNTHETIC TEST APPROVAL",
                                        "approved_at": "2000-01-01T00:00:00+00:00"}))
        document["approval"] = ref(approved, scope)
    contract.write_bytes(json_bytes(document))
    return deployment_path, contract


def fixture_freeze(path, *, ttft_ms=100000, tpot_ms=100000, scope="fixture", binding=None):
    path.mkdir(parents=True)
    calibration = path / "calibration.json"
    calibration.write_bytes(json_bytes({"scope": scope, "note": "synthetic test calibration"}))
    document = {"schema": "m8-slo-v1", "scope": scope, "units": "ms",
                "ttft_ms_by_prompt_tokens": [{"max_prompt_tokens": 2, "ttft_ms": ttft_ms},
                                            {"max_prompt_tokens": None, "ttft_ms": ttft_ms}],
                "tpot_ms": tpot_ms, "single_token_policy": "ttft_only",
                "calibration_files": [ref(calibration, scope)]}
    if scope == "formal":
        document["binding"] = binding
    approval = path / "approval.json"
    approval.write_bytes(json_bytes({"schema": "m8-approval-v1", "scope": scope, "status": "approved",
                                     "subject_sha256": sha256(json_bytes(document)), "approver": "SYNTHETIC TEST APPROVAL",
                                     "approved_at": "2000-01-01T00:00:00+00:00"}))
    document["approval"] = ref(approval, scope)
    freeze = path / "freeze.json"
    freeze.write_bytes(json_bytes(document))
    return freeze


def replay_args(plan, output, endpoints, **overrides):
    values = dict(plan=str(plan), output_dir=str(output), endpoint=endpoints, model="tiny", ignore_eos="true",
                  concurrency=1, request_timeout_s=1.0, batch_timeout_s=3.0, max_raw_bytes=100000,
                  gpu_count=1, warmup_plan=None, no_warmup=True, protocol_contract=None, deployment=None,
                  freeze=None)
    values.update(overrides)
    return argparse.Namespace(**values)


def choice(identity, text, reason=None, *, index=0):
    return {"id": identity, "object": "text_completion", "model": "tiny",
            "choices": [{"index": index, "text": text, "finish_reason": reason}]}


def usage(identity, prompt_tokens, completion_tokens):
    return {"id": identity, "object": "text_completion", "model": "tiny", "choices": [],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens}}


def event(obj):
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"


@asynccontextmanager
async def tcp_server(respond, counter=None):
    tasks = set()
    counter = counter if counter is not None else {"active": 0, "maximum": 0, "requests": []}

    async def handle(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            lines = header.decode("latin1").split("\r\n")
            headers = {key.lower(): value.strip() for key, value in
                       (line.split(":", 1) for line in lines[1:] if ":" in line)}
            body = json.loads(await reader.readexactly(int(headers["content-length"])))
            counter["active"] += 1
            counter["maximum"] = max(counter["maximum"], counter["active"])
            counter["requests"].append({"body": body, "headers": headers})
            try:
                await respond(body, headers, writer)
            finally:
                counter["active"] -= 1
        except (ConnectionError, asyncio.IncompleteReadError, BrokenPipeError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, BrokenPipeError):
                pass
            tasks.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", counter
    finally:
        server.close()
        await server.wait_closed()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


async def send_response(writer, chunks, *, status=200, delay=0):
    header = (f"HTTP/1.1 {status} OK\r\nContent-Type: text/event-stream\r\n"
              "Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n").encode()
    writer.write(header)
    await writer.drain()
    for chunk in chunks:
        if delay:
            await asyncio.sleep(delay)
        writer.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
        await writer.drain()
    writer.write(b"0\r\n\r\n")
    await writer.drain()
