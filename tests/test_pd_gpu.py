import asyncio
import json
import multiprocessing as mp
import signal

import pytest
import torch

from myvllm.pd.client import PDClient
from myvllm.pd.wire import DEFAULT_HEADER_LIMIT, read_header, write_frame
from myvllm.sampling_parameters import SamplingParams
from pd_gpu_helpers import gpu_worker_process, reference_process
from pd_helpers import create_tiny_qwen_checkpoint


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def _params(max_tokens, ignore_eos=True, temperature=0, max_model_length=None):
    return {"temperature": temperature, "max_tokens": max_tokens,
            "ignore_eos": ignore_eos, "max_model_length": max_model_length}


def _cases():
    return [
        {"id": "prime", "prompt": [1], "sampling": _params(1)},
        {"id": "length-1", "prompt": [2], "sampling": _params(2)},
        {"id": "length-b-minus-1", "prompt": [1, 2, 3], "sampling": _params(4)},
        {"id": "length-b", "prompt": [3, 4, 5, 6], "sampling": _params(5)},
        {"id": "length-b-plus-1", "prompt": [1, 3, 5, 7, 9], "sampling": _params(9)},
        {"id": "length-cross-block", "prompt": [2, 3, 4, 5, 6, 7, 8], "sampling": _params(6)},
        {"id": "eos", "prompt": [5, 6], "sampling": _params(5, ignore_eos=False)},
        {"id": "eos-other", "prompt": [2], "sampling": _params(5, ignore_eos=False)},
        {"id": "total-limit", "prompt": [1, 2, 3, 4, 5], "sampling": _params(8, max_model_length=7)},
        {"id": "concurrent-a", "prompt": [3, 4, 5], "sampling": _params(4)},
        {"id": "concurrent-b", "prompt": [7, 8, 9, 10, 11], "sampling": _params(5)},
    ]


def _start_worker(ctx, role, config, decode_endpoint=None, progress_lock=None):
    parent, child = ctx.Pipe(duplex=False)
    stop_child, stop = ctx.Pipe(duplex=False)
    process = ctx.Process(target=gpu_worker_process,
                          args=(role, config, child, stop_child, decode_endpoint, progress_lock))
    process.start()
    child.close()
    stop_child.close()
    try:
        assert parent.poll(180), f"{role} did not start (exit={process.exitcode})"
        response = parent.recv()
        assert response[0] == "ready", response
    except BaseException:
        if process.is_alive():
            process.terminate()
            process.join(10)
        stop.close()
        raise
    finally:
        parent.close()
    return process, stop, response[1]


def _stop_worker(item):
    process, stop, _ = item
    if process.is_alive():
        stop.send("stop")
        process.join(90)
    if process.is_alive():
        process.terminate()
        process.join(10)
        pytest.fail("GPU worker did not exit")
    stop.close()
    assert process.exitcode == 0


async def _request(pe, de, case):
    d = await PDClient.open_d(de, lifetime=90)
    p = None
    try:
        p = await PDClient.open_p(pe, d.session_id, case["id"], case["prompt"],
                                  SamplingParams(**case["sampling"]), lifetime=90)
        event = await p.next_token()
        tokens = [event["token"]]
        if not event["finished"]:
            await PDClient.handoff(p, d, await p.describe())
            while True:
                event = await d.next_token()
                tokens.append(event["token"])
                if event["finished"]:
                    break
        return tokens, event["finish_reason"]
    finally:
        if p:
            await p.close()
        await d.close()


async def _exercise(pe, de, cases):
    pinfo, dinfo = await PDClient.info(pe), await PDClient.info(de)
    assert pinfo["model"] == dinfo["model"]
    assert pinfo["sampling_seed"] == 0
    assert dinfo["sampling_seed"] == 0x80000000
    sample = {"id": "positive-temperature", "prompt": [3, 4, 5],
              "sampling": _params(3, temperature=.8)}
    result = {sample["id"]: await _request(pe, de, sample)}
    for case in cases[:-2]:
        result[case["id"]] = await _request(pe, de, case)
    concurrent = await asyncio.gather(*(_request(pe, de, case) for case in cases[-2:]))
    result.update({case["id"]: value for case, value in zip(cases[-2:], concurrent)})
    for endpoint in (pe, de):
        stats = await PDClient.stats(endpoint)
        assert stats["open_handles"] == stats["used_blocks"] == stats["reserved_transfer_bytes"] == 0
    return result


@pytest.mark.parametrize("dtype,eager,chunked", [
    ("float32", True, False), ("bfloat16", True, True),
    ("float32", False, True), ("bfloat16", False, False),
])
def test_real_socket_pd_matches_colocated_gpu(dtype, eager, chunked, tmp_path):
    config = create_tiny_qwen_checkpoint(tmp_path / "checkpoint")
    config.update(dtype=dtype, max_num_sequences=2, max_num_batched_tokens=24,
                  max_model_length=24, max_cached_blocks=16, eos=[56, 60], seed=0)
    cases = _cases()
    ctx = mp.get_context("spawn")
    ref_path = tmp_path / "reference.pt"
    prefill_path = tmp_path / "prefill.pt"
    decode_path = tmp_path / "decode.pt"
    reference_config = {**config, "capture_role": "reference", "capture_path": str(ref_path)}
    parent, child = ctx.Pipe(duplex=False)
    ref = ctx.Process(target=reference_process, args=(reference_config, cases, child))
    ref.start()
    child.close()
    try:
        assert parent.poll(180), f"reference did not finish (exit={ref.exitcode})"
        response = parent.recv()
        assert response[0] == "ok", response
        ref.join(30)
        assert ref.exitcode == 0
    finally:
        parent.close()
        if ref.is_alive():
            ref.terminate()
            ref.join(10)

    p_config = {**config, "capture_role": "prefill", "capture_path": str(prefill_path),
                "enable_chunked_prefill": chunked,
                "max_num_batched_tokens": 4 if chunked else 24}
    d_config = {**config, "capture_role": "decode", "capture_path": str(decode_path),
                "enforce_eager": eager}
    d = _start_worker(ctx, "decode", d_config)
    p = None
    try:
        p = _start_worker(ctx, "prefill", p_config, d[2])
        actual = asyncio.run(_exercise(p[2], d[2], cases))
    finally:
        if p:
            _stop_worker(p)
        _stop_worker(d)

    reference = torch.load(ref_path, weights_only=False)
    prefill = torch.load(prefill_path, weights_only=False)
    decode = torch.load(decode_path, weights_only=False)
    assert all(capture["device"] == 0 and not capture["missing"]
               for capture in (reference, prefill, decode))
    assert decode["step_kinds"] and not any(decode["step_kinds"])
    sample_p = prefill["requests"]["positive-temperature"]["events"][0]["rng"]
    sample_d = decode["requests"]["positive-temperature"]["events"][0]["rng"]
    assert not torch.equal(sample_p, sample_d)
    assert actual["eos"][1] == "stop" and len(actual["eos"][0]) == 1
    assert actual["eos-other"][1] == "stop" and len(actual["eos-other"][0]) == 1
    assert {actual["eos"][0][0], actual["eos-other"][0][0]} == {56, 60}
    assert actual["length-1"][0][0] in config["eos"] and len(actual["length-1"][0]) == 2
    assert len(actual["total-limit"][0]) == 2
    assert all(prefill["requests"][case["id"]]["blocks"] != decode["requests"][case["id"]]["blocks"]
               for case in cases[-2:])
    report = {"dtype": dtype, "decode_eager": eager, "prefill_chunked": chunked,
              "rtol": 1e-4 if dtype == "float32" else .02,
              "atol": 1e-5 if dtype == "float32" else .02, "cases": {}}
    failures = []
    for case in cases:
        request_id = case["id"]
        expected = reference["requests"][request_id]["events"]
        events = prefill["requests"][request_id]["events"] + decode["requests"].get(request_id, {}).get("events", [])
        expected_tokens = [event["token"] for event in expected]
        observed_tokens = [event["token"] for event in events]
        tokens_equal = actual[request_id][0] == observed_tokens == expected_tokens
        if not tokens_equal:
            failures.append(f"{request_id}: greedy tokens differ")
        common = min(len(expected_tokens), len(observed_tokens))
        mismatch = next((i for i in range(common)
                         if expected_tokens[i] != observed_tokens[i]), None)
        if mismatch is None and len(expected_tokens) != len(observed_tokens):
            mismatch = common
        scored = min(common, mismatch + 1) if mismatch is not None else common
        assert scored > 0
        ref_logits = torch.stack([event["logits"] for event in expected[:scored]])
        got_logits = torch.stack([event["logits"] for event in events[:scored]])
        delta = (ref_logits - got_logits).abs()
        logits_equal = torch.allclose(got_logits, ref_logits, rtol=report["rtol"], atol=report["atol"])
        if not logits_equal:
            failures.append(f"{request_id}: logits exceed fixed tolerance")
        kv_equal = True
        if len(events) > 1:
            source = prefill["requests"][request_id]["kv"]
            destination = decode["requests"][request_id]["kv"]
            kv_equal = len(source) == len(destination) and all(
                torch.equal(a, b) for a, b in zip(source, destination)
            )
            if not kv_equal:
                failures.append(f"{request_id}: valid KV differs")
        report["cases"][request_id] = {"tokens": observed_tokens,
                                       "expected_tokens": expected_tokens,
                                       "first_divergence": mismatch,
                                       "same_prefix_logit_positions": scored,
                                       "max_abs_error": delta.max().item(),
                                       "mean_abs_error": delta.mean().item(),
                                       "logits_within_tolerance": logits_equal,
                                       "valid_kv_exact": kv_equal}
    (tmp_path / "numeric-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    assert not failures, failures


def test_gpu_killed_prefill_decode_survives_restart(tmp_path):
    config = create_tiny_qwen_checkpoint(tmp_path / "checkpoint")
    config.update(max_num_sequences=2, max_num_batched_tokens=16, max_model_length=16,
                  max_cached_blocks=12, eos=[62, 63], seed=0)
    ctx = mp.get_context("spawn")
    # Inductor's tqdm progress bar otherwise creates a multiprocessing RLock in
    # each worker; SIGKILL prevents that worker from unregistering its semaphore.
    progress_lock = ctx.RLock()
    p_config = {**config, "capture_role": "prefill", "capture_path": str(tmp_path / "killed-p.pt")}
    d_config = {**config, "capture_role": "decode", "capture_path": str(tmp_path / "survivor-d.pt")}
    d = _start_worker(ctx, "decode", d_config, progress_lock=progress_lock)
    p = None
    replacement = None
    try:
        p = _start_worker(ctx, "prefill", p_config, d[2], progress_lock)

        async def kill_and_recover():
            reader, writer = await asyncio.open_connection(*d[2])
            pc = None
            try:
                await write_frame(writer, {"op": "open_d", "lifetime": 90})
                response, size = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
                assert response["ok"] and size == 0
                pc = await PDClient.open_p(p[2], response["value"]["session_id"],
                                           "disconnect-d", [1, 2, 3],
                                           SamplingParams(temperature=0, max_tokens=4, ignore_eos=True),
                                           lifetime=90)
                await pc.next_token()
                metadata = await pc.describe()
                await write_frame(writer, {"op": "reserve", "metadata": metadata})
                response, size = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
                assert response["ok"] and size == 0
                await write_frame(writer, {"op": "restore"}, await pc.export())
                writer.close()
                await writer.wait_closed()
            finally:
                if pc:
                    await pc.close()
                writer.close()
                await writer.wait_closed()
            for _ in range(200):
                stats = await PDClient.stats(d[2])
                if stats["open_handles"] == stats["used_blocks"] == stats["reserved_blocks"] == 0:
                    break
                await asyncio.sleep(.01)
            else:
                pytest.fail(f"GPU decode resources did not drain: {stats}")
            case = {"id": "after-disconnect", "prompt": [2, 3], "sampling": _params(4)}
            tokens, reason = await _request(p[2], d[2], case)
            assert len(tokens) == 4 and reason == "length"

            dc = await PDClient.open_d(d[2], lifetime=90)
            pc = await PDClient.open_p(p[2], dc.session_id, "kill-p", [1, 2, 3],
                                       SamplingParams(temperature=0, max_tokens=4, ignore_eos=True),
                                       lifetime=90)
            try:
                assert not (await pc.next_token())["finished"]
                assert await dc.reserve(await pc.describe())
                p[0].kill()
                p[0].join(10)
                print(f"killed prefill pid={p[0].pid}, exitcode={p[0].exitcode}", flush=True)
                assert p[0].exitcode == -signal.SIGKILL
                with pytest.raises((ConnectionError, OSError, asyncio.IncompleteReadError)):
                    await pc.export()
            finally:
                await pc.close()
                await dc.close()
            stats = await PDClient.stats(d[2])
            assert stats["open_handles"] == stats["used_blocks"] == stats["reserved_blocks"] == 0

        asyncio.run(kill_and_recover())
        p_config["capture_path"] = str(tmp_path / "replacement-p.pt")
        replacement = _start_worker(ctx, "prefill", p_config, d[2], progress_lock)
        case = {"id": "after-restart", "prompt": [3, 4, 5], "sampling": _params(4)}
        tokens, reason = asyncio.run(_request(replacement[2], d[2], case))
        assert len(tokens) == 4 and reason == "length"
        stats = asyncio.run(PDClient.stats(d[2]))
        assert stats["open_handles"] == stats["used_blocks"] == stats["reserved_blocks"] == 0
    finally:
        if p:
            if p[0].is_alive():
                _stop_worker(p)
            else:
                p[1].close()
        if replacement:
            _stop_worker(replacement)
        _stop_worker(d)
        del progress_lock
