import argparse
import asyncio
import gzip
import json
from pathlib import Path

import pytest

from tools.prepare_trace import file_sha256, json_bytes, load_plan, prepare, sha256
from tools.replay_trace import replay, validate_freeze_binding
from trace_helpers import (REVISION, choice, event, fixture_contract, make_plan, ref, replay_args,
                           send_response, tcp_server, usage)


def result(directory, section="formal"):
    return json.loads((directory / section / "results.jsonl").read_bytes().split(b"\n")[0])


def test_formal_freeze_requires_deployment_and_both_client_tools(tmp_path):
    _, manifest, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 1}])
    settings = {"replay_tool_sha256": "1" * 64, "concurrency": 2, "request_timeout_s": 5,
                "batch_timeout_s": 30, "max_raw_bytes": 10000, "gpu_count": 4, "ignore_eos": True}
    binding = {"model_revision": manifest["model_revision"], "tokenizer_revision": manifest["tokenizer"]["revision"],
               "tokenizer_files_sha256": sha256(json_bytes(manifest["tokenizer"]["files"])), **settings,
               "deployment_sha256": "0" * 64, "analyze_tool_sha256": "0" * 64,
               "plans": [{"plan_sha256": manifest["plan_sha256"], "warmup_plan_sha256": None}]}
    with pytest.raises(ValueError, match="deployment|tool|settings"):
        validate_freeze_binding({"scope": "formal", "binding": binding}, manifest, None, settings)


def test_prepare_rejects_workload_outside_anchored_source_manifest(tmp_path):
    plan, _, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 1}])
    directory = plan.parent
    source = directory / "m2.jsonl"
    source_manifest = Path(str(source) + ".manifest.json")
    provenance = directory / "source-manifest.json"
    document = json.loads(provenance.read_bytes())
    document["files"].update({f"workloads/{source.name}": {"sha256": file_sha256(source)},
                              f"workloads/{source_manifest.name}": {"sha256": file_sha256(source_manifest)}})
    provenance.write_bytes(json_bytes(document))
    source.write_bytes(json_bytes({"prompt": "changed", "output_tokens": 1}))
    source_document = json.loads(source_manifest.read_bytes())
    source_document["sha256"] = file_sha256(source)
    source_manifest.write_bytes(json_bytes(source_document))
    args = argparse.Namespace(
        mode="l2", kind="formal", chat=str(source), long=None, tokenizer=str(directory / "tokenizer"),
        tokenizer_provenance=str(provenance), model_revision=REVISION, tokenizer_revision=REVISION,
        seed=7, id_prefix="trace", rate=1, count=1, duration_s=None, chat_probability=None,
        phases=None, max_requests=None, allow_recycle=False, output=str(directory / "changed-plan.jsonl"))
    with pytest.raises(ValueError, match="provenance|source"):
        prepare(args)


@pytest.mark.parametrize("item", ["source", "source_manifest", "tokenizer", "provenance"])
def test_load_plan_rechecks_actual_source_and_tokenizer_evidence(tmp_path, item):
    plan, _, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 1}])
    paths = {"source": plan.parent / "m2.jsonl",
             "source_manifest": plan.parent / "m2.jsonl.manifest.json",
             "tokenizer": plan.parent / "tokenizer" / "tokenizer.json",
             "provenance": plan.parent / "source-manifest.json"}
    path = paths[item]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        load_plan(plan)


def test_real_http_307_is_not_followed(tmp_path):
    async def run():
        plan, _, tokenizer = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 1}])

        async def destination(body, headers, writer):
            await send_response(writer, [event(choice("c", "", "length"))
                                        + event(usage("c", len(tokenizer.encode(body["prompt"])), 1))
                                        + b"data: [DONE]\n\n"])

        async with tcp_server(destination) as (target, target_counter):
            async def redirect(body, headers, writer):
                payload = b"redirect denied"
                writer.write((f"HTTP/1.1 307 Temporary Redirect\r\nLocation: {target}/v1/completions\r\n"
                              f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode() + payload)
                await writer.drain()

            async with tcp_server(redirect) as (endpoint, redirect_counter):
                output = tmp_path / "run"
                await replay(replay_args(plan, output, [endpoint]))
                row = result(output)
                assert len(redirect_counter["requests"]) == 1 and not target_counter["requests"]
                assert row["status"] == "failed" and row["http_status"] == 307
                assert (output / "formal" / row["raw_path"]).read_bytes() == b"redirect denied"

    asyncio.run(run())


def test_real_http_valid_choice_before_bad_utf8_keeps_choice_time(tmp_path):
    async def run():
        plan, _, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 2}])
        wire = event(choice("c", "first")) + b"data: \xff\n\n"

        async def respond(body, headers, writer):
            await send_response(writer, [wire])

        async with tcp_server(respond) as (endpoint, _):
            output = tmp_path / "run"
            await replay(replay_args(plan, output, [endpoint]))
            row = result(output)
            assert row["status"] == "failed" and row["error_class"] == "UnicodeDecodeError"
            assert len(row["choice_times_s"]) == 1
            assert (output / "formal" / row["raw_path"]).read_bytes() == wire

    asyncio.run(run())


def test_real_http_non_sse_body_is_preserved(tmp_path):
    async def run():
        plan, _, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 1}])
        payload = b'{"error":"backend failure"}'

        async def respond(body, headers, writer):
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                         + str(len(payload)).encode() + b"\r\nConnection: close\r\n\r\n" + payload)
            await writer.drain()

        async with tcp_server(respond) as (endpoint, _):
            output = tmp_path / "run"
            await replay(replay_args(plan, output, [endpoint]))
            row = result(output)
            assert row["status"] == "failed" and row["error_class"] == "ProtocolError"
            assert (output / "formal" / row["raw_path"]).read_bytes() == payload

    asyncio.run(run())


def test_warmup_and_formal_default_ids_are_distinct(tmp_path):
    plan, _, _ = make_plan(tmp_path / "formal", [{"prompt": "hello", "output_tokens": 1}])
    warmup, _, _ = make_plan(tmp_path / "warmup", [{"prompt": "hello", "output_tokens": 1}], kind="warmup")
    assert json.loads(plan.read_bytes())["request_id"] != json.loads(warmup.read_bytes())["request_id"]


def test_replay_rejects_warmup_formal_id_collision_before_sending(tmp_path):
    async def run():
        plan, _, tokenizer = make_plan(tmp_path / "formal", [{"prompt": "hello", "output_tokens": 1}])
        warmup, _, _ = make_plan(tmp_path / "warmup", [{"prompt": "hello", "output_tokens": 1}], kind="warmup")
        warm_row = json.loads(warmup.read_bytes())
        warm_row["request_id"] = json.loads(plan.read_bytes())["request_id"]
        warmup.write_bytes(json_bytes(warm_row))
        manifest_path = Path(str(warmup) + ".manifest.json")
        manifest = json.loads(manifest_path.read_bytes())
        manifest["plan_sha256"] = file_sha256(warmup)
        manifest_path.write_bytes(json_bytes(manifest))

        async def respond(body, headers, writer):
            await send_response(writer, [event(choice("c", "", "length"))
                                        + event(usage("c", len(tokenizer.encode(body["prompt"])), 1))
                                        + b"data: [DONE]\n\n"])

        async with tcp_server(respond) as (endpoint, counter):
            with pytest.raises(ValueError, match="request ID|request_id"):
                await replay(replay_args(plan, tmp_path / "run", [endpoint], warmup_plan=str(warmup)))
            assert not counter["requests"] and not (tmp_path / "run").exists()

    asyncio.run(run())


def test_fixture_evidence_cannot_be_promoted_by_outer_formal_scope(tmp_path):
    async def run():
        plan, manifest, tokenizer = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 1}])

        async def respond(body, headers, writer):
            await send_response(writer, [event(choice("c", "", "length"))
                                        + event(usage("c", len(tokenizer.encode(body["prompt"])), 1))
                                        + b"data: [DONE]\n\n"])

        async with tcp_server(respond) as (endpoint, _):
            deployment, contract = fixture_contract(tmp_path / "fixture", manifest, [endpoint])
            deployment_doc = json.loads(deployment.read_bytes())
            deployment_doc["scope"] = "formal"
            deployment.write_bytes(json_bytes(deployment_doc))
            document = json.loads(contract.read_bytes())
            document["scope"] = "formal"
            document["deployment_sha256"] = file_sha256(deployment)
            for key in ("source_review", "probe_manifest"):
                path = Path(document[key]["path"])
                nested = json.loads(path.read_bytes())
                nested["scope"] = "formal"
                nested["deployment_sha256"] = file_sha256(deployment)
                path.write_bytes(json_bytes(nested))
                document[key] = ref(path)
            approved = contract.parent / "formal-approval.json"
            approved.write_bytes(json_bytes({"schema": "m8-approval-v1", "status": "approved",
                                            "subject_sha256": sha256(json_bytes(document)),
                                            "approver": "fixture", "approved_at": "2000-01-01T00:00:00+00:00"}))
            document["approval"] = ref(approved)
            contract.write_bytes(json_bytes(document))
            with pytest.raises((ValueError, KeyError)):
                await replay(replay_args(plan, tmp_path / "run", [endpoint],
                                         deployment=str(deployment), protocol_contract=str(contract)))

    asyncio.run(run())


@pytest.mark.parametrize("boundary", ["tiny", "inside_event", "one_event", "two_events", "larger_prefix",
                                     "one_event_cr", "two_events_cr"])
def test_real_tcp_raw_hard_cap_keeps_completed_prefix_events_and_continues(tmp_path, boundary):
    async def run():
        plan, _, _ = make_plan(tmp_path / "inputs", [{"prompt": p, "output_tokens": 2}
                                                     for p in ("hello", "world")])
        first = event(choice("c", "first"))
        last = event(choice("c", "", "length"))
        if boundary.endswith("_cr"):
            first, last = first.replace(b"\n", b"\r"), last.replace(b"\n", b"\r")
        prefix = first + last
        payload = prefix + b": " + b"x" * 10000 + b"\n\n"
        caps = {"tiny": 1, "inside_event": len(first) - 1, "one_event": len(first),
                "two_events": len(prefix), "larger_prefix": 1024,
                "one_event_cr": len(first), "two_events_cr": len(prefix)}
        cap = caps[boundary]

        async def respond(body, headers, writer):
            wire = payload if body["prompt"] == "hello" else b"busy"
            await send_response(writer, [wire], status=200 if body["prompt"] == "hello" else 503)

        async with tcp_server(respond) as (endpoint, counter):
            output = tmp_path / "run"
            run_result = await replay(replay_args(plan, output, [endpoint], max_raw_bytes=cap))
            rows = [json.loads(line) for line in (output / "formal" / "results.jsonl").read_bytes().split(b"\n") if line]
            row = rows[0]
            raw = (output / "formal" / row["raw_path"]).read_bytes()
            assert raw == payload[:cap] and len(raw) == cap
            assert row["status"] == "failed" and row["error_class"] == "RawLimitError"
            assert row["raw_bytes"] == cap and row["received_body_bytes"] == cap + 1
            assert row["raw_truncated"] is True and row["content_encoding"] == "identity"
            assert len(row["choice_times_s"]) == (cap >= len(first)) + (cap >= len(prefix))
            if cap >= len(prefix):
                assert row["choice_times_s"][0] == row["choice_times_s"][1]
                assert row["finish_offset_s"] == row["choice_times_s"][-1]
            assert row["first_output_offset_s"] is row["last_output_offset_s"] is None
            assert len(rows) == len(counter["requests"]) == 2 and run_result["state"] == "complete"

    asyncio.run(run())


@pytest.mark.parametrize("cap,invalid_gzip", [(1, False), (1024, False), (10000, False), (1024, True)])
def test_real_tcp_unexpected_gzip_retains_bounded_wire_without_decoding(tmp_path, cap, invalid_gzip):
    async def run():
        plan, _, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 2}])
        payload = event(choice("c", "first")) + b": " + b"x" * (2 * 1024 * 1024) + b"\n\n"
        wire = b"not a gzip stream" if invalid_gzip else gzip.compress(payload, mtime=0)

        async def respond(body, headers, writer):
            assert headers["accept-encoding"] == "identity"
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Encoding: gzip\r\n"
                         b"Content-Length: " + str(len(wire)).encode() + b"\r\nConnection: close\r\n\r\n" + wire)
            await writer.drain()

        async with tcp_server(respond) as (endpoint, _):
            output = tmp_path / "run"
            await replay(replay_args(plan, output, [endpoint], max_raw_bytes=cap))
            row = result(output)
            raw = (output / "formal" / row["raw_path"]).read_bytes()
            assert raw == wire[:cap] and len(raw) <= cap
            assert row["status"] == "failed" and row["http_status"] == 200
            assert row["error_class"] == ("RawLimitError" if len(wire) > cap else "ProtocolError")
            assert row["raw_bytes"] == len(raw) and row["received_body_bytes"] == min(len(wire), cap + 1)
            assert row["raw_truncated"] == (len(wire) > cap) and row["content_encoding"] == "gzip"
            assert row["choice_times_s"] == [] and row["response_id"] is None
            assert row["first_output_offset_s"] is row["last_output_offset_s"] is None
            if len(wire) <= cap:
                assert "Content-Encoding" in row["error"]

    asyncio.run(run())


@pytest.mark.parametrize("status,content_type", [(200, "application/json"), (503, "text/event-stream")])
def test_real_tcp_non_sse_or_error_body_obeys_same_hard_cap(tmp_path, status, content_type):
    async def run():
        plan, _, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 2}])
        payload = event(choice("c", "first")) + b"x" * 10000

        async def respond(body, headers, writer):
            writer.write((f"HTTP/1.1 {status} OK\r\nContent-Type: {content_type}\r\n"
                          f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode() + payload)
            await writer.drain()

        async with tcp_server(respond) as (endpoint, _):
            output = tmp_path / "run"
            await replay(replay_args(plan, output, [endpoint], max_raw_bytes=1024))
            row = result(output)
            assert (output / "formal" / row["raw_path"]).read_bytes() == payload[:1024]
            assert row["raw_bytes"] == 1024 and row["raw_truncated"] is True
            assert row["status"] == "failed" and row["choice_times_s"] == []

    asyncio.run(run())


def test_real_tcp_response_exactly_at_raw_cap_is_complete(tmp_path):
    async def run():
        plan, _, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 1}])
        wire = event(choice("c", "", "length")) + event(usage("c", 1, 1)) + b"data: [DONE]\n\n"

        async def respond(body, headers, writer):
            await send_response(writer, [wire])

        async with tcp_server(respond) as (endpoint, _):
            output = tmp_path / "run"
            await replay(replay_args(plan, output, [endpoint], max_raw_bytes=len(wire)))
            row = result(output)
            assert row["status"] == "success" and row["done"]
            assert (output / "formal" / row["raw_path"]).read_bytes() == wire
            assert row["raw_bytes"] == row["received_body_bytes"] == len(wire)
            assert row["raw_truncated"] is False and row["content_encoding"] == "identity"

    asyncio.run(run())
