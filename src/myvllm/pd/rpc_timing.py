"""Opt-in same-host timestamps for the four pipeline handoff control RPCs."""
import os

from myvllm.utils.timing import write_record

_directory = os.environ.get("MYVLLM_RPC_TRACE_DIR")
TRACE_OPS = {"describe", "commit_early", "acknowledge_early", "activate"} if _directory else ()


def record_prefill_step(started_ns, finished_ns, batch):
    write_record(f"{_directory}/steps-{os.getpid()}.jsonl", {
        "started_ns": started_ns, "finished_ns": finished_ns, "batch": batch,
    })


def record_rpc(role, session_id, op, **timestamps):
    write_record(f"{_directory}/{role}-{os.getpid()}.jsonl", {
        "session_id": session_id, "op": op, **timestamps,
    })
