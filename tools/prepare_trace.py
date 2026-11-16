"""Materialize one reproducible L2/L3 arrival plan from pinned M2 workloads."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re

from transformers import AutoTokenizer


PLAN_SCHEMA = "m8-trace-plan-v1"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def positive_number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return float(value)


def tokenizer_files(directory, provenance_path, revision):
    directory = Path(directory).resolve()
    provenance_path = Path(provenance_path).resolve()
    provenance = read_json(provenance_path)
    if provenance["model_revision"] != revision or provenance.get("tokenizer_revision", revision) != revision:
        raise ValueError("tokenizer provenance revision does not match")
    expected = {name[len("tokenizer/"):]: row["sha256"]
                for name, row in provenance["files"].items() if name.startswith("tokenizer/")}
    actual = {str(path.relative_to(directory)): file_sha256(path)
              for path in sorted(directory.rglob("*")) if path.is_file()}
    if not expected or actual != expected:
        raise ValueError("tokenizer files do not match provenance hashes")
    return {
        "path": str(directory), "revision": revision, "files": actual,
        "provenance_path": str(provenance_path), "provenance_sha256": file_sha256(provenance_path),
    }


def read_source(path, model_revision, tokenizer_revision, provenance):
    path = Path(path).resolve()
    manifest_path = Path(str(path) + ".manifest.json")
    manifest_data = manifest_path.read_bytes()
    manifest = json.loads(manifest_data)
    content = path.read_bytes()
    source_entry = f"workloads/{path.name}"
    manifest_entry = f"{source_entry}.manifest.json"
    files = provenance["files"]
    if (source_entry not in files or manifest_entry not in files
            or sha256(content) != files[source_entry]["sha256"]
            or sha256(manifest_data) != files[manifest_entry]["sha256"]):
        raise ValueError(f"source provenance hash mismatch: {path}")
    if sha256(content) != manifest["sha256"]:
        raise ValueError(f"source hash mismatch: {path}")
    if manifest["model_revision"] != model_revision or manifest["tokenizer_revision"] != tokenizer_revision:
        raise ValueError(f"source revision mismatch: {path}")
    lines = content.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    if len(lines) != manifest["requests"] or not lines:
        raise ValueError(f"source LF row count mismatch: {path}")
    rows = []
    for number, line in enumerate(lines, 1):
        row = json.loads(line.decode("utf-8"))
        if (not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not row["prompt"]
                or type(row.get("output_tokens")) is not int or row["output_tokens"] <= 0):
            raise ValueError(f"invalid source row {number}: {path}")
        rows.append({"prompt": row["prompt"], "output_tokens": row["output_tokens"],
                     "source_line": number, "source_line_sha256": sha256(line)})
    metadata = {"path": str(path), "sha256": sha256(content), "requests": len(rows),
                "manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_data),
                "provenance_entry": source_entry, "dataset": manifest["source"]["dataset"],
                "max_model_length": manifest["max_model_length"]}
    return metadata, rows


def prepare(args):
    for name in ("model_revision", "tokenizer_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", getattr(args, name)):
            raise ValueError(f"{name} must be a full revision SHA")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be a 32-bit unsigned integer")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.id_prefix):
        raise ValueError("id-prefix must contain only letters, digits, dot, underscore or hyphen")
    if not args.chat and not args.long:
        raise ValueError("at least one --chat or --long source is required")
    if args.max_requests is not None and args.max_requests <= 0:
        raise ValueError("max-requests must be positive")
    tokenizer_info = tokenizer_files(args.tokenizer, args.tokenizer_provenance, args.tokenizer_revision)
    provenance = read_json(tokenizer_info["provenance_path"])
    tokenizer = AutoTokenizer.from_pretrained(str(Path(args.tokenizer).resolve()),
                                               revision=args.tokenizer_revision, local_files_only=True)
    sources = {}
    for kind in ("chat", "long"):
        path = getattr(args, kind)
        if path:
            sources[kind] = read_source(path, args.model_revision, args.tokenizer_revision, provenance)
    if args.mode == "l2":
        rate = positive_number(args.rate, "rate")
        if args.phases is not None:
            raise ValueError("phases apply only to L3")
        if args.chat_probability is None:
            if len(sources) != 1:
                raise ValueError("mixed L2 requires --chat-probability")
            probability = 1.0 if "chat" in sources else 0.0
        else:
            probability = args.chat_probability
        if args.count is not None:
            if args.count <= 0 or args.duration_s is not None:
                raise ValueError("count must be positive and exclusive with duration")
            duration = None
        else:
            duration = positive_number(args.duration_s, "duration-s")
            if args.max_requests is None:
                raise ValueError("finite-window L2 requires --max-requests")
        phases = [{"name": "l2", "duration_s": duration, "rate": rate,
                   "chat_probability": probability}]
    else:
        if (args.phases is None or args.count is not None or args.duration_s is not None
                or args.rate is not None or args.chat_probability is not None):
            raise ValueError("L3 requires --phases and no L2 count, duration or rate")
        if args.max_requests is None:
            raise ValueError("L3 requires --max-requests")
        phases = read_json(args.phases)
        if not isinstance(phases, list) or not phases:
            raise ValueError("phases must be a nonempty JSON array")
    names = set()
    for phase in phases:
        if not isinstance(phase, dict) or set(phase) != {"name", "duration_s", "rate", "chat_probability"}:
            raise ValueError("each phase must have name, duration_s, rate and chat_probability")
        if not isinstance(phase["name"], str) or not phase["name"] or phase["name"] in names:
            raise ValueError("phase names must be nonempty and unique")
        names.add(phase["name"])
        phase["rate"] = positive_number(phase["rate"], "phase rate")
        if args.mode == "l3" and phase["duration_s"] is None:
            raise ValueError("L3 phase duration must be finite")
        if phase["duration_s"] is not None:
            phase["duration_s"] = positive_number(phase["duration_s"], "phase duration")
        probability = phase["chat_probability"]
        if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("chat_probability must be between zero and one")
        if (probability > 0 and "chat" not in sources) or (probability < 1 and "long" not in sources):
            raise ValueError("each phase requires its possible source datasets")
    rng = random.Random(args.seed)
    cursors = Counter()
    role_counts = Counter()
    lengths = {}
    plan = []
    offset = 0.0
    phase_records = []
    for phase in phases:
        start = offset
        local = 0.0
        first = len(plan)
        while True:
            local += rng.expovariate(phase["rate"])
            if phase["duration_s"] is not None and local >= phase["duration_s"]:
                break
            if args.count is not None and len(plan) >= args.count:
                break
            if args.max_requests is not None and len(plan) >= args.max_requests:
                raise ValueError("generated arrivals exceed max-requests; no plan was written")
            kind = "chat" if rng.random() < phase["chat_probability"] else "long"
            metadata, rows = sources[kind]
            cursor = cursors[metadata["sha256"]]
            cycle, index = divmod(cursor, len(rows))
            if cycle and not args.allow_recycle:
                raise ValueError(f"{kind} source exhausted; --allow-recycle is required")
            row = rows[index]
            key = (kind, index)
            if key not in lengths:
                lengths[key] = len(tokenizer.encode(row["prompt"]))
            prompt_tokens = lengths[key]
            if not 0 < prompt_tokens <= metadata["max_model_length"] - row["output_tokens"]:
                raise ValueError(f"source {kind} line {row['source_line']} exceeds model length")
            ordinal = len(plan)
            plan.append({
                "ordinal": ordinal, "request_id": f"{args.id_prefix}-{args.kind}-{args.seed:08x}-{ordinal:08d}",
                "offset_s": start + local, "phase": phase["name"], "source": kind,
                "source_path": metadata["path"], "source_sha256": metadata["sha256"],
                "source_manifest_sha256": metadata["manifest_sha256"],
                "source_line": row["source_line"], "source_line_sha256": row["source_line_sha256"],
                "source_cycle": cycle, "source_occurrence": cursor + 1,
                "prompt": row["prompt"], "prompt_tokens": prompt_tokens, "output_tokens": row["output_tokens"],
            })
            cursors[metadata["sha256"]] += 1
            role_counts[kind] += 1
        duration = phase["duration_s"] if phase["duration_s"] is not None else plan[-1]["offset_s"] - start
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("generated plan duration must be positive and finite")
        offset = start + duration
        count = len(plan) - first
        phase_sources = Counter(row["source"] for row in plan[first:])
        phase_records.append({**phase, "duration_s": duration, "start_s": start, "end_s": offset,
                              "request_count": count, "realized_rate": count / duration,
                              "source_counts": {"chat": phase_sources["chat"], "long": phase_sources["long"]},
                              "realized_chat_fraction": phase_sources["chat"] / count if count else None})
    content = b"".join(json_bytes(row) for row in plan)
    output = Path(args.output).resolve()
    manifest_path = Path(str(output) + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise ValueError("plan or manifest already exists")
    manifest = {
        "schema": PLAN_SCHEMA, "kind": args.kind, "mode": args.mode, "seed": args.seed,
        "plan_path": str(output), "plan_sha256": sha256(content), "request_count": len(plan),
        "window_s": offset, "phases": phase_records, "source_counts": dict(role_counts),
        "sources": {kind: pair[0] for kind, pair in sources.items()},
        "model_revision": args.model_revision, "tokenizer": tokenizer_info,
        "allow_recycle": args.allow_recycle,
        "input_tokens": sum(row["prompt_tokens"] for row in plan),
        "max_output_tokens": sum(row["output_tokens"] for row in plan),
        "phase_spec_path": str(Path(args.phases).resolve()) if args.phases else None,
        "phase_spec_sha256": file_sha256(args.phases) if args.phases else None,
        "tool_sha256": file_sha256(__file__),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    manifest_path.write_bytes(json_bytes(manifest))
    return manifest


def load_plan(path):
    path = Path(path)
    data = path.read_bytes()
    manifest = read_json(str(path) + ".manifest.json")
    if manifest["schema"] != PLAN_SCHEMA or sha256(data) != manifest["plan_sha256"]:
        raise ValueError("plan schema or hash mismatch")
    tokenizer = manifest["tokenizer"]
    if file_sha256(tokenizer["provenance_path"]) != tokenizer["provenance_sha256"]:
        raise ValueError("source provenance hash mismatch")
    if tokenizer_files(tokenizer["path"], tokenizer["provenance_path"], tokenizer["revision"]) != tokenizer:
        raise ValueError("tokenizer evidence does not match plan")
    provenance = read_json(tokenizer["provenance_path"])
    source_rows = {}
    for kind, expected in manifest["sources"].items():
        actual, source_rows[kind] = read_source(expected["path"], manifest["model_revision"],
                                                tokenizer["revision"], provenance)
        if actual != expected:
            raise ValueError(f"source evidence does not match plan: {kind}")
    lines = data.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    rows = [json.loads(line) for line in lines]
    if len(rows) != manifest["request_count"]:
        raise ValueError("plan row count mismatch")
    if [row["ordinal"] for row in rows] != list(range(len(rows))):
        raise ValueError("invalid plan ordinals")
    if any(not isinstance(row.get("request_id"), str) or not row["request_id"]
           or type(row.get("prompt_tokens")) is not int or type(row.get("output_tokens")) is not int for row in rows):
        raise ValueError("invalid plan request ID or token budget")
    if len({row["request_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate plan request ID")
    if (sum(row["prompt_tokens"] for row in rows) != manifest["input_tokens"]
            or sum(row["output_tokens"] for row in rows) != manifest["max_output_tokens"]):
        raise ValueError("plan token budget mismatch")
    phases = {}
    phase_end = 0.0
    for phase in manifest["phases"]:
        if (phase["name"] in phases or phase["start_s"] != phase_end
                or type(phase["duration_s"]) not in (int, float) or not math.isfinite(phase["duration_s"])
                or phase["duration_s"] <= 0 or phase["end_s"] != phase_end + phase["duration_s"]):
            raise ValueError("invalid phase window")
        phases[phase["name"]] = phase
        phase_end = phase["end_s"]
    if not phases or phase_end != manifest["window_s"]:
        raise ValueError("invalid planned window")
    if any(type(source["requests"]) is not int or source["requests"] <= 0
           for source in manifest["sources"].values()):
        raise ValueError("invalid source request count")
    previous = 0.0
    occurrences = Counter()
    role_counts = Counter()
    phase_counts = Counter()
    for row in rows:
        if row["phase"] not in phases or row["source"] not in manifest["sources"]:
            raise ValueError("invalid plan phase or source")
        phase = phases[row["phase"]]
        source = manifest["sources"][row["source"]]
        offset = row["offset_s"]
        if (type(offset) not in (int, float) or not math.isfinite(offset) or offset < previous
                or not phase["start_s"] <= offset <= phase["end_s"]):
            raise ValueError("plan arrival/phase mismatch")
        occurrences[source["sha256"]] += 1
        occurrence = occurrences[source["sha256"]]
        role_counts[row["source"]] += 1
        if (not isinstance(row["request_id"], str) or not row["request_id"]
                or not isinstance(row["prompt"], str) or not row["prompt"]
                or type(row["prompt_tokens"]) is not int or row["prompt_tokens"] <= 0
                or type(row["output_tokens"]) is not int or row["output_tokens"] <= 0
                or row["prompt_tokens"] + row["output_tokens"] > source["max_model_length"]
                or row["source_path"] != source["path"] or row["source_sha256"] != source["sha256"]
                or row["source_manifest_sha256"] != source["manifest_sha256"]
                or not isinstance(row["source_line_sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["source_line_sha256"])
                or row["source_occurrence"] != occurrence
                or row["source_cycle"] != (occurrence - 1) // source["requests"]
                or row["source_line"] != (occurrence - 1) % source["requests"] + 1
                or (row["source_cycle"] and not manifest["allow_recycle"])):
            raise ValueError("invalid plan row or source provenance")
        original = source_rows[row["source"]][row["source_line"] - 1]
        if (row["prompt"] != original["prompt"] or row["output_tokens"] != original["output_tokens"]
                or row["source_line_sha256"] != original["source_line_sha256"]):
            raise ValueError("plan row does not match source evidence")
        phase_counts[row["phase"]] += 1
        previous = offset
    if any(phase["request_count"] != phase_counts[name] for name, phase in phases.items()):
        raise ValueError("phase request count mismatch")
    if dict(role_counts) != manifest["source_counts"]:
        raise ValueError("source occurrence count mismatch")
    return manifest, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("l2", "l3"), required=True)
    parser.add_argument("--kind", choices=("formal", "warmup"), required=True)
    parser.add_argument("--chat")
    parser.add_argument("--long")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-provenance", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--id-prefix", default="trace")
    parser.add_argument("--rate", type=float)
    parser.add_argument("--count", type=int)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--chat-probability", type=float)
    parser.add_argument("--phases", help="L3 JSON array of name/duration_s/rate/chat_probability")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--allow-recycle", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args), ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
