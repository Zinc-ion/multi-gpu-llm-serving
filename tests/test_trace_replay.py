import asyncio
import json
from pathlib import Path

import pytest

from test_pd_load_routing import elastic, http_gateway, nodes, records, settings
from test_pd_server import drained
from tools.analyze_pd import analyze
from tools.prepare_trace import file_sha256, json_bytes
from tools.replay_trace import replay
from trace_helpers import choice, event, fixture_contract, make_plan, replay_args, send_response, tcp_server, usage


def results(directory):
    return [json.loads(line) for line in (directory / "formal" / "results.jsonl").read_bytes().split(b"\n") if line]


def test_real_tcp_split_utf8_crlf_multidata_and_empty_choice(tmp_path):
    async def run():
        plan, manifest, tokenizer = make_plan(tmp_path / "inputs", [{"prompt": "hello world", "output_tokens": 2}])
        identity = "completion-1"
        first = choice(identity, "")
        first_json = json.dumps(first, ensure_ascii=False)
        split = first_json.index('"model"')
        first_event = (": keepalive\r\nevent: ping\r\ndata: {}\r\n\r\ndata: " + first_json[:split] + "\r\ndata: "
                       + first_json[split:] + "\r\n\r\n").encode()
        second_event = event(choice(identity, "😀", "length"))
        marker = second_event.index("😀".encode())
        last = event(usage(identity, len(tokenizer.encode("hello world")), 2)) + b"data: [DONE]\n\n"
        wire = first_event + second_event + last

        async def respond(body, headers, writer):
            assert body["n"] == 1 and body["echo"] is False and body["stream"] is True
            assert body["stream_options"] == {"include_usage": True}
            assert body["prompt"] == "hello world" and body["max_tokens"] == 2
            parts = [first_event[:-1], first_event[-1:] + second_event[:marker + 2], second_event[marker + 2:] + last]
            await send_response(writer, parts, delay=.01)

        async with tcp_server(respond) as (endpoint, _):
            deployment, contract = fixture_contract(tmp_path / "fixture", manifest, [endpoint])
            output = tmp_path / "run"
            result = await replay(replay_args(plan, output, [endpoint], deployment=str(deployment),
                                              protocol_contract=str(contract)))
            assert result["state"] == "complete" and result["formal"]["counts"]["success"] == 1
            row = results(output)[0]
            assert row["finish_reason"] == "length" and row["usage"]["completion_tokens"] == 2
            assert row["first_output_offset_s"] == row["choice_times_s"][0]
            assert row["last_output_offset_s"] == row["finish_offset_s"] == row["choice_times_s"][-1]
            assert (output / "formal" / row["raw_path"]).read_bytes() == wire

    asyncio.run(run())


def test_combined_events_share_timestamp_and_unverified_terminal_is_na(tmp_path):
    async def run():
        plan, manifest, tokenizer = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 3}])
        normal = (event(choice("c", "")) + event(choice("c", "x", "length"))
                  + event(usage("c", len(tokenizer.encode("hello")), 2)) + b"data: [DONE]\n\n")

        async def respond(body, headers, writer):
            await send_response(writer, [normal])

        async with tcp_server(respond) as (endpoint, _):
            deployment, contract = fixture_contract(tmp_path / "fixture", manifest, [endpoint])
            verified = tmp_path / "verified"
            await replay(replay_args(plan, verified, [endpoint], deployment=str(deployment),
                                     protocol_contract=str(contract)))
            row = results(verified)[0]
            assert len(row["choice_times_s"]) == 2 and row["choice_times_s"][0] == row["choice_times_s"][1]
            assert row["first_output_offset_s"] == row["last_output_offset_s"]
            capture = tmp_path / "capture"
            await replay(replay_args(plan, capture, [endpoint]))
            row = results(capture)[0]
            assert row["status"] == "success" and len(row["choice_times_s"]) == 2
            assert row["first_output_offset_s"] is None and row["last_output_offset_s"] is None

    asyncio.run(run())


def test_pure_terminal_ambiguity_and_inconsistent_event_count_have_no_timing(tmp_path):
    async def run():
        plan, manifest, tokenizer = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 2}])
        wire = (event(choice("c", "token")) + event(choice("c", "", "length"))
                + event(usage("c", len(tokenizer.encode("hello")), 1)) + b"data: [DONE]\n\n")

        async def respond(body, headers, writer):
            await send_response(writer, [wire])

        async with tcp_server(respond) as (endpoint, _):
            deployment, contract = fixture_contract(tmp_path / "fixture", manifest, [endpoint])
            for index, options in enumerate(({}, {"deployment": str(deployment), "protocol_contract": str(contract)})):
                output = tmp_path / f"run-{index}"
                await replay(replay_args(plan, output, [endpoint], **options))
                row = results(output)[0]
                assert row["status"] == "success" and len(row["choice_times_s"]) == 2
                assert row["usage"]["completion_tokens"] == 1
                assert row["first_output_offset_s"] is row["last_output_offset_s"] is None

    asyncio.run(run())


def test_request_id_cannot_choose_raw_file_path(tmp_path):
    async def run():
        plan, _, tokenizer = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 1}])
        row = json.loads(plan.read_bytes())
        row["request_id"] = "../../outside"
        plan.write_bytes(json_bytes(row))
        manifest_path = Path(str(plan) + ".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        manifest["plan_sha256"] = file_sha256(plan)
        manifest_path.write_bytes(json_bytes(manifest))

        async def respond(body, headers, writer):
            assert headers["x-request-id"] == "../../outside"
            await send_response(writer, [event(choice("c", "", "length"))
                                        + event(usage("c", len(tokenizer.encode(body["prompt"])), 1))
                                        + b"data: [DONE]\n\n"])

        async with tcp_server(respond) as (endpoint, _):
            output = tmp_path / "run"
            await replay(replay_args(plan, output, [endpoint]))
            row = results(output)[0]
            assert row["status"] == "success" and row["raw_path"] == "raw/00000000.sse"
            assert not (tmp_path / "outside").exists()

    asyncio.run(run())


@pytest.mark.parametrize("problem", ["missing_finish", "missing_usage", "missing_done", "error", "bad_index",
                                     "bad_object", "bad_id", "invalid_utf8", "duplicate_finish", "bad_usage",
                                     "prompt_mismatch", "truncated", "abort", "zero_completion", "http_error",
                                     "duplicate_json"])
def test_real_tcp_protocol_failures_are_not_success(tmp_path, problem):
    async def run():
        plan, _, tokenizer = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 3}])
        prompt_tokens = len(tokenizer.encode("hello"))
        c1, c2 = choice("c", "a"), choice("c", "b", "length")
        final_usage = usage("c", prompt_tokens, 2)
        messages = [event(c1), event(c2), event(final_usage), b"data: [DONE]\n\n"]
        status = 200
        if problem == "missing_finish":
            messages[1] = event(choice("c", "b"))
        elif problem == "missing_usage":
            messages.pop(2)
        elif problem == "missing_done":
            messages.pop()
        elif problem == "error":
            messages[1] = event({"error": {"message": "failed"}})
        elif problem == "bad_index":
            messages[0] = event(choice("c", "a", index=1))
        elif problem == "bad_object":
            messages[0] = event({**c1, "object": "chat.completion.chunk"})
        elif problem == "bad_id":
            messages[1] = event(choice("other", "b", "length"))
        elif problem == "invalid_utf8":
            messages[1] = b"data: \xff\n\n"
        elif problem == "duplicate_finish":
            messages.insert(2, event(choice("c", "c", "length")))
        elif problem == "bad_usage":
            final_usage["usage"]["total_tokens"] += 1
            messages[2] = event(final_usage)
        elif problem == "prompt_mismatch":
            messages[2] = event(usage("c", prompt_tokens + 1, 2))
        elif problem == "truncated":
            messages = [event(c1), b"data: {\"id\":\"c\""]
        elif problem == "abort":
            messages[1] = event(choice("c", "b", "abort"))
        elif problem == "zero_completion":
            messages[2] = event(usage("c", prompt_tokens, 0))
        elif problem == "http_error":
            status = 503
            messages = [b'{"error":"busy"}']
        elif problem == "duplicate_json":
            messages[0] = b'data: {"id":"c","id":"d","object":"text_completion","model":"tiny","choices":[]}\n\n'

        async def respond(body, headers, writer):
            await send_response(writer, [b"".join(messages)], status=status)

        async with tcp_server(respond) as (endpoint, _):
            output = tmp_path / "run"
            result = await replay(replay_args(plan, output, [endpoint]))
            row = results(output)[0]
            assert result["formal"]["counts"] == {"success": 0, "failed": 1, "not_executed": 0}
            assert row["status"] == "failed" and row["error_class"]
            assert (output / "formal" / row["raw_path"]).read_bytes()

    asyncio.run(run())


def test_global_concurrency_late_dispatch_two_endpoints_and_timeout_continues(tmp_path):
    async def run():
        prompts = ["hello timeout", "hello world", "world world", "hello hello"]
        plan, _, tokenizer = make_plan(tmp_path / "inputs", [{"prompt": p, "output_tokens": 1} for p in prompts])
        counter = {"active": 0, "maximum": 0, "requests": []}

        async def respond(body, headers, writer):
            await asyncio.sleep(.15 if body["prompt"] == "hello timeout" else .04)
            stream = (event(choice("c-" + headers["x-request-id"], "", "length"))
                      + event(usage("c-" + headers["x-request-id"], len(tokenizer.encode(body["prompt"])), 1))
                      + b"data: [DONE]\n\n")
            await send_response(writer, [stream])

        async with tcp_server(respond, counter) as (first, _), tcp_server(respond, counter) as (second, _):
            output = tmp_path / "run"
            result = await replay(replay_args(plan, output, [first, second], request_timeout_s=.1))
            rows = sorted(results(output), key=lambda row: row["ordinal"])
            assert [row["status"] for row in rows] == ["failed", "success", "success", "success"]
            assert rows[0]["error_class"] == "TimeoutError"
            assert [row["endpoint"] for row in rows] == [first, second, first, second]
            assert rows[1]["queue_delay_s"] >= .08 and rows[2]["queue_delay_s"] >= .11
            assert len({row["request_id"] for row in rows}) == 4
            assert all(row["raw_path"] == f"raw/{row['ordinal']:08d}.sse" for row in rows)
            assert result["formal"]["counts"] == {"success": 3, "failed": 1, "not_executed": 0}
            # The timed-out server may still be doing work after the client releases its slot.
            assert counter["maximum"] <= 2

    asyncio.run(run())


def test_two_real_endpoints_share_one_bounded_concurrency_cap(tmp_path):
    async def run():
        prompts = ["hello world", "world world", "hello hello", "hello world"]
        plan, _, tokenizer = make_plan(tmp_path / "inputs", [{"prompt": p, "output_tokens": 1} for p in prompts])
        counter = {"active": 0, "maximum": 0, "requests": []}

        async def respond(body, headers, writer):
            await asyncio.sleep(.04)
            identity = "c-" + headers["x-request-id"]
            stream = (event(choice(identity, "", "length"))
                      + event(usage(identity, len(tokenizer.encode(body["prompt"])), 1))
                      + b"data: [DONE]\n\n")
            await send_response(writer, [stream])

        async with tcp_server(respond, counter) as (first, _), tcp_server(respond, counter) as (second, _):
            output = tmp_path / "run"
            result = await replay(replay_args(plan, output, [first, second], concurrency=1))
            rows = sorted(results(output), key=lambda row: row["ordinal"])
            assert counter["maximum"] == 1 and len(counter["requests"]) == 4
            assert [row["endpoint"] for row in rows] == [first, second, first, second]
            assert all(row["status"] == "success" for row in rows)
            assert rows[1]["queue_delay_s"] >= .03 and rows[3]["queue_delay_s"] >= .1
            assert result["formal"]["counts"] == {"success": 4, "failed": 0, "not_executed": 0}

    asyncio.run(run())


def test_batch_deadline_retains_all_unexecuted_rows_and_partial_bytes(tmp_path):
    async def run():
        plan, _, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 2}] * 4)
        partial = event(choice("c", "first"))

        async def respond(body, headers, writer):
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                         b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
            writer.write(f"{len(partial):x}\r\n".encode() + partial + b"\r\n")
            await writer.drain()
            await asyncio.sleep(.3)

        async with tcp_server(respond) as (endpoint, _):
            output = tmp_path / "run"
            result = await replay(replay_args(plan, output, [endpoint], batch_timeout_s=.08))
            rows = sorted(results(output), key=lambda row: row["ordinal"])
            assert len(rows) == 4
            assert rows[0]["status"] == "failed" and all(row["status"] == "not_executed" for row in rows[1:])
            assert result["state"] == "incomplete" and result["formal"]["stop_reason"] == "batch_deadline"
            assert (output / "formal" / rows[0]["raw_path"]).read_bytes() == partial

    asyncio.run(run())


def test_real_pd_gateway_local_route_has_zero_transfer(tmp_path):
    async def run():
        node_path = tmp_path / "node"
        node_path.mkdir()
        async with nodes(node_path, count=1, d_options=[{"chunked": True, "budget": 3}]) as (
                config, tokenizer, ps, ds, _):
            pairs = list(zip(ps, ds))
            cfg = settings(pairs, poll_interval=.01, sample_age=5, elastic=elastic(backlog=0, idle=True))
            async with http_gateway(config, cfg, node_path) as (_, gateway, port):
                plan, _, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello token2", "output_tokens": 3}],
                                       tokenizer=tokenizer)
                output = tmp_path / "run"
                result = await replay(replay_args(plan, output, [f"http://127.0.0.1:{port}"]))
                assert result["formal"]["counts"]["success"] == 1
                row = results(output)[0]
                await drained(gateway, pairs)
                assigned = records(node_path / "gateway.jsonl", "pd_gateway_assigned")
                assert len(assigned) == 1 and assigned[0]["request_id"] == row["request_id"]
                assert assigned[0]["route"] == "local"
                worker = records(node_path / "d0.jsonl", "pd_admitted")
                assert len(worker) == 1 and worker[0]["route"] == "local" and worker[0]["payload_bytes"] == 0
                group = analyze([node_path / "gateway.jsonl"])["transactions"][0]
                assert group["route"] == "local" and group["payload_bytes"] == 0
                assert not records(node_path / "gateway.jsonl", "pd_gateway_handoff_start")

    asyncio.run(run())
