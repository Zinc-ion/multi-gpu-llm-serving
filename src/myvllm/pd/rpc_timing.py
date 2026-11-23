"""Opt-in same-host timestamps for the four pipeline handoff control RPCs."""
import os

from myvllm.utils.timing import write_record

_directory = os.environ.get("MYVLLM_RPC_TRACE_DIR")
TRACE_OPS = {"describe", "commit_early", "acknowledge_early", "activate"} if _directory else ()


def record_rpc(role, session_id, op, **timestamps):
    write_record(f"{_directory}/{role}-{os.getpid()}.jsonl", {
        "session_id": session_id, "op": op, **timestamps,
    })
