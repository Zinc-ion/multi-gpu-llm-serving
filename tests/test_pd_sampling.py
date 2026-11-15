from contextlib import contextmanager
from copy import deepcopy
import multiprocessing as mp
import traceback

import pytest
import torch

from myvllm.layers.sampler import SamplerLayer
from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.pd import DecodeCore, PrefillCore
from myvllm.sampling_parameters import SamplingParams
from test_paged_prefill import cpu_model_backend
from test_pd_core import assert_conserved, first_token, make_cores, make_engine, start
from test_pd_model_cpu import model_engine


def rng_state():
    return torch.get_rng_state().numpy().tobytes()


def sampling_worker(role, connection, directory, seed):
    torch.set_num_threads(1)
    patch = pytest.MonkeyPatch()
    backend = cpu_model_backend.__wrapped__(patch, directory)
    core = None
    try:
        next(backend)
        torch.manual_seed(0 if seed is None else seed)
        model = Qwen3ForCausalLM(
            vocab_size=64, hidden_size=8, num_heads=2, head_dim=4, num_kv_heads=1,
            intermediate_size=16, num_layers=2, max_position=32, block_size=4,
        ).eval()
        for param in model.parameters():
            torch.nn.init.zeros_(param)
        engine = model_engine(model, "float32")
        if seed is not None:
            engine.config["seed"] = seed
        before_core = rng_state()
        core_type = PrefillCore if role == "p" else DecodeCore
        core = core_type(engine, 8, 1_000_000)
        handles = {}
        connection.send(("ok", (before_core, core.stats()), rng_state()))
        with torch.inference_mode():
            while True:
                action, args = connection.recv()
                if action == "shutdown":
                    break
                try:
                    if action == "open":
                        handle = core.open(*args)
                        handles[handle.session_id] = handle
                        result = handle.session_id
                    elif action == "step":
                        result = [(event.handle.session_id, event.token, event.finished)
                                  for event in core.step()]
                    elif action == "stats":
                        result = core.stats()
                    else:
                        session_id, *remaining = args
                        result = getattr(core, action)(handles[session_id], *remaining)
                        if action == "close":
                            del handles[session_id]
                    connection.send(("ok", result, rng_state()))
                except Exception:
                    connection.send(("error", traceback.format_exc(), rng_state()))
    except BaseException:
        connection.send(("error", traceback.format_exc(), None))
    finally:
        if core is not None:
            core.shutdown()
        try:
            next(backend)
        except StopIteration:
            pass
        patch.undo()
        connection.close()


def receive(connection):
    assert connection.poll(60), "sampling worker did not respond"
    return connection.recv()


@contextmanager
def worker(role, directory, seed=None):
    directory.mkdir()
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=sampling_worker, args=(role, child, directory, seed))
    process.start()
    child.close()
    try:
        status, initial, state = receive(parent)
        assert status == "ok", initial
        yield parent, initial, state
    finally:
        if process.is_alive():
            parent.send(("shutdown", ()))
        process.join(15)
        if process.is_alive():
            process.terminate()
            process.join(15)
        parent.close()
        assert process.exitcode == 0


def call(connection, action, *args, error=None):
    connection.send((action, args))
    status, result, state = receive(connection)
    if error is None:
        assert status == "ok", result
    else:
        assert status == "error" and error in result
    return result, state


def unchanged(connection, state, action, *args, error=None):
    result, after = call(connection, action, *args, error=error)
    assert after == state
    return result


def reference_samples(seed, batches):
    states, tokens = [], []
    with torch.random.fork_rng(devices=[]), torch.compiler.set_stance("force_eager"):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        states.append(rng_state())
        sampler = SamplerLayer()
        for size in batches:
            tokens.append(sampler(torch.zeros(size, 64), torch.ones(size)).tolist())
            states.append(rng_state())
    return states, tokens


def transfer(p, d, session, p_state, d_state):
    metadata = unchanged(p, p_state, "describe", session)
    assert metadata["version"] == 2
    assert metadata["sampling_seed"] == 0
    assert unchanged(d, d_state, "reserve", session, metadata)
    payload = unchanged(p, p_state, "export", session)
    assert unchanged(d, d_state, "restore", session, payload)
    assert unchanged(p, p_state, "acknowledge", session)
    assert unchanged(p, p_state, "close", session)
    assert unchanged(d, d_state, "activate", session)


def test_independent_processes_keep_advancing_sampling_streams(tmp_path):
    p_states, p_tokens = reference_samples(0, [2, 1, 1])
    d_states, d_tokens = reference_samples(0x80000000, [2, 3, 3, 1, 1])
    params = SamplingParams(temperature=1, max_tokens=4, ignore_eos=True)
    with worker("p", tmp_path / "p") as (p, p_initial, p_state), \
            worker("d", tmp_path / "d") as (d, d_initial, d_state):
        assert p_initial[0] == d_initial[0]
        assert p_state != d_state
        assert p_state == p_states[0] and d_state == d_states[0]
        assert p_initial[1]["sampling_seed"] == 0
        assert d_initial[1]["sampling_seed"] == 0x80000000

        sessions = []
        for index in range(2):
            session = unchanged(d, d_state, "open")
            assert unchanged(p, p_state, "open", session, f"request-{index}", [1, 2, 3], params) == session
            sessions.append(session)
        events, p_state = call(p, "step")
        assert [event[0] for event in events] == sessions
        assert [event[1] for event in events] == p_tokens[0]
        assert p_state == p_states[1] and p_state != p_states[0]
        for session in sessions:
            transfer(p, d, session, p_state, d_state)
        events, d_state = call(d, "step")
        assert [event[0] for event in events] == sessions
        assert [event[1] for event in events] == d_tokens[0]
        assert d_state == d_states[1] and d_state != d_states[0]

        later = unchanged(d, d_state, "open")
        assert unchanged(p, p_state, "open", later, "later", [4, 5], params) == later
        events, p_state = call(p, "step")
        assert [event[1] for event in events] == p_tokens[1]
        assert p_state == p_states[2] and p_state != p_states[1]
        transfer(p, d, later, p_state, d_state)
        sessions.append(later)
        for index in (1, 2):
            events, d_state = call(d, "step")
            assert [event[0] for event in events] == sessions
            assert [event[1] for event in events] == d_tokens[index]
            assert d_state == d_states[index + 1] and d_state != d_states[index]
        for session in sessions[:2]:
            assert unchanged(d, d_state, "close", session)
        events, d_state = call(d, "step")
        assert [event[0] for event in events] == [later]
        assert [event[1] for event in events] == d_tokens[3]
        assert d_state == d_states[4] and d_state != d_states[3]
        assert unchanged(d, d_state, "close", later)

        last = unchanged(d, d_state, "open")
        last_params = SamplingParams(temperature=1, max_tokens=2, ignore_eos=True)
        assert unchanged(p, p_state, "open", last, "after-idle", [6], last_params) == last
        events, p_state = call(p, "step")
        assert [event[1] for event in events] == p_tokens[2]
        assert p_state == p_states[3] and p_state != p_states[2]
        transfer(p, d, last, p_state, d_state)
        events, d_state = call(d, "step")
        assert [event[1] for event in events] == d_tokens[4]
        assert events[0][2]
        assert d_state == d_states[5] and d_state != d_states[4]
        assert unchanged(d, d_state, "close", last)
        for connection, state in ((p, p_state), (d, d_state)):
            stats = unchanged(connection, state, "stats")
            assert stats["open_handles"] == stats["used_blocks"] == 0
            assert stats["reserved_transfer_bytes"] == 0


def test_equal_actual_seeds_rejected_before_reservation_and_d_can_continue(tmp_path):
    params = SamplingParams(temperature=1, max_tokens=2, ignore_eos=True)
    with worker("p", tmp_path / "collision-p", 0x80000000) as (p, _, p_state), \
            worker("d", tmp_path / "collision-d", 0) as (d, d_initial, d_state):
        assert d_initial[1]["sampling_seed"] == 0x80000000
        session = unchanged(d, d_state, "open")
        unchanged(p, p_state, "open", session, "collision", [1, 2], params)
        _, p_state = call(p, "step")
        metadata = unchanged(p, p_state, "describe", session)
        assert metadata["sampling_seed"] == 0x80000000
        unchanged(d, d_state, "reserve", session, metadata, error="sampling_seed")
        stats = unchanged(d, d_state, "stats")
        assert stats["reserved_blocks"] == stats["resident_requests"] == 0
        assert stats["reserved_transfer_bytes"] == stats["used_blocks"] == 0
        assert unchanged(p, p_state, "close", session)
        assert unchanged(d, d_state, "close", session)

        with worker("p", tmp_path / "survivor-p", 0) as (other_p, _, other_state):
            survivor = unchanged(d, d_state, "open")
            unchanged(other_p, other_state, "open", survivor, "survivor", [3, 4], params)
            _, other_state = call(other_p, "step")
            transfer(other_p, d, survivor, other_state, d_state)
            events, d_state = call(d, "step")
            assert len(events) == 1 and events[0][2]
            assert unchanged(d, d_state, "close", survivor)
        stats = unchanged(d, d_state, "stats")
        assert stats["open_handles"] == stats["reserved_blocks"] == 0
        assert stats["resident_requests"] == stats["used_blocks"] == 0


@pytest.mark.parametrize("invalid", [None, -1, 1 << 32, True, "0", "missing", "legacy-version"])
def test_invalid_sampling_seed_metadata_is_rejected(invalid):
    p, d = make_cores()
    ph, dh = start(p, d, [1, 2, 3])
    first_token(p, ph)
    metadata = deepcopy(p.describe(ph))
    if invalid == "missing":
        del metadata["sampling_seed"]
    elif invalid == "legacy-version":
        metadata["version"] = 1
    else:
        metadata["sampling_seed"] = invalid
    with pytest.raises(ValueError):
        d.reserve(dh, metadata)
    assert d.stats()["reserved_blocks"] == d.stats()["resident_requests"] == 0
    assert d.stats()["reserved_transfer_bytes"] == d.stats()["used_blocks"] == 0
    assert_conserved(p)
    assert_conserved(d)
    p.shutdown()
    d.shutdown()


@pytest.mark.parametrize("seed", [1, 0x7fffffff, 0x80000000, 0xffffffff])
@pytest.mark.parametrize("core_type,mask", [(PrefillCore, 0), (DecodeCore, 0x80000000)])
def test_pd_core_seed_derivation_across_32_bit_domain(seed, core_type, mask):
    engine = make_engine()
    engine.config["seed"] = seed
    core = core_type(engine, 8, 1_000_000)
    expected = torch.Generator(device="cpu").manual_seed(seed ^ mask).get_state()
    assert core.stats()["sampling_seed"] == seed ^ mask
    assert torch.equal(torch.get_rng_state(), expected)
    core.shutdown()


@pytest.mark.parametrize("seed", [-1, 1 << 32, True, 0.5])
@pytest.mark.parametrize("core_type", [PrefillCore, DecodeCore])
def test_pd_core_rejects_base_seed_outside_32_bit_integer_domain(seed, core_type):
    engine = make_engine()
    engine.config["seed"] = seed
    with pytest.raises(ValueError, match="seed"):
        core_type(engine, 8, 1_000_000)
