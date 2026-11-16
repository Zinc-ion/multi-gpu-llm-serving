"""OpenAI HTTP gateway for independent PD TCP workers."""

import argparse
import asyncio
from dataclasses import asdict
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
from myvllm.pd.client import PDClient, PDHeaderLimitError, PDRemoteError, PDTimingError
from myvllm.pd.http import GatewayError
from myvllm.pd.protocol import _ARCHITECTURE_FIELDS, cache_spec, make_early_metadata, make_metadata, payload_nbytes, token_byte_counts, validate_metadata
from myvllm.pd.wire import DEFAULT_HEADER_LIMIT, ProtocolError
from myvllm.utils.timing import write_record


logger = logging.getLogger("myvllm.pd.gateway")


def _endpoint(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not isinstance(value[0], str) or not value[0]
            or type(value[1]) is not int or not 0 < value[1] < 65536):
        raise ValueError("Each endpoint must be [host, port]")
    return tuple(value)


def _model_spec(config):
    spec = {
        "name": config["model_name_or_path"],
        "revision": config["model_revision"],
        "tokenizer_revision": config["tokenizer_revision"],
        "type": config["model_type"],
        "dtype": str(resolve_dtype(config.get("dtype", "float32"))).removeprefix("torch."),
        "architecture": {key: config[key] for key in _ARCHITECTURE_FIELDS if key in config},
        "eos": sorted(set(config["eos"])),
    }
    cache_format = cache_spec(config)
    if cache_format is not None:
        spec["kv_cache"] = cache_format
    return spec


def _worker_payload_limit(info):
    aggregate = info["max_transfer_bytes"]
    payload = info.get("max_payload_bytes", aggregate)
    if type(aggregate) is not int or type(payload) is not int or not 0 < payload <= aggregate:
        raise ValueError("Invalid worker transfer or payload limit")
    return payload


def _validate_worker_role(info, role):
    if (info["role"] != role or
            not (info["world_size"] == 1 or
                 role == "prefill" and info["world_size"] == 2 and not info["pipeline_frame_bytes"])):
        raise ValueError(f"Incompatible {role} worker role or world size")


def _validate_load_worker(info, role, request_timeout, max_payload_bytes):
    _validate_worker_role(info, role)
    if info["request_timeout"] < request_timeout:
        raise ValueError(f"{role} worker timeout is shorter than gateway timeout")
    if _worker_payload_limit(info) > max_payload_bytes:
        raise ValueError(f"{role} worker body limit exceeds gateway payload limit")
    if info["max_header_bytes"] <= 0 or info["max_connections"] <= 0:
        raise ValueError(f"{role} worker has invalid protocol limits")
    if role == "decode" and info["decode_endpoint"] is not None:
        raise ValueError("Decode worker has a fixed decode endpoint")


def _validate_load_edge(pi, di, pipeline_min_tokens):
    if pi["model"] != di["model"] or pi["max_model_length"] != di["max_model_length"]:
        raise ValueError("Incompatible model or length on allowed edge")
    if pi["sampling_seed"] == di["sampling_seed"]:
        raise ValueError("P/D sampling seed collision on allowed edge")
    if pipeline_min_tokens and (
            not pi["pipeline_frame_bytes"] or
            pi["pipeline_frame_bytes"] > di["pipeline_frame_bytes"] or
            not pi["enforce_eager"] or pi["enable_chunked_prefill"]):
        raise ValueError("Pipeline requires enabled workers with a compatible frame limit on allowed edge")


def _header_bytes(header):
    return len(json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _metadata_header_bounds(pending, info, limit, pipeline):
    length = len(pending.token_ids)
    sequence = SimpleNamespace(
        seq_id=0, token_ids=pending.token_ids + [0], num_prompt_tokens=length,
        num_cached_tokens=length, num_completion_tokens=1,
        temperature=pending.params.temperature, max_tokens=pending.params.max_tokens,
        ignore_eos=pending.params.ignore_eos, max_model_length=limit,
    )
    handle = SimpleNamespace(session_id="0" * 32, transaction_id="0" * 32,
                             request_id=pending.request_id, sequence=sequence)
    metadata = make_metadata(handle, info["model"], info["sampling_seed"])
    p_headers = [_header_bytes({"ok": True, "value": metadata})]
    d_headers = [_header_bytes({"op": "commit_early" if pipeline else "reserve", "metadata": metadata})]
    if pipeline:
        sequence.token_ids = pending.token_ids
        sequence.num_cached_tokens = sequence.num_completion_tokens = 0
        early = make_early_metadata(handle, info["model"], info["sampling_seed"])
        p_headers.append(_header_bytes({"ok": True, "value": early}))
        d_headers.append(_header_bytes({"op": "reserve_early", "metadata": early}))
    # P allocates the sequence id and first token later. Zero gives a lower bound;
    # the client checks each actual encoded request again at the worker limit.
    return max(p_headers), max(d_headers)


def _request_error(exc, phase):
    if isinstance(exc, GatewayError):
        return exc
    if isinstance(exc, PDTimingError):
        return GatewayError(500, f"PD timing write failed: {exc}")
    if isinstance(exc, PDHeaderLimitError):
        return GatewayError(503, f"PD control header exceeds worker limit during {phase}")
    if isinstance(exc, TimeoutError) or isinstance(exc, PDRemoteError) and exc.code == "TimeoutError":
        return GatewayError(504, f"PD request timed out during {phase}")
    if isinstance(exc, PDRemoteError):
        if exc.code == "CapacityError" or "capacity" in str(exc).lower():
            return GatewayError(503, str(exc))
        if exc.code == "ProtocolError" and "Control frame exceeds limit" in str(exc):
            return GatewayError(503, f"PD control header exceeds worker limit during {phase}")
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
        self.routing_mode = settings.get("routing_mode", "fixed")
        if self.routing_mode not in ("fixed", "load"):
            raise ValueError("routing_mode must be fixed or load")
        if self.routing_mode == "fixed" and (len(set(p for p, _ in self.pairs)) != len(self.pairs)
                                             or len(set(d for _, d in self.pairs)) != len(self.pairs)):
            raise ValueError("Each fixed pair must use independent workers")
        if self.routing_mode == "load":
            self.prefills = list(dict.fromkeys(p for p, _ in self.pairs))
            self.decodes = list(dict.fromkeys(d for _, d in self.pairs))
            if set(self.prefills) & set(self.decodes):
                raise ValueError("Prefill and decode inventories must be independent")
            for name in ("poll_interval_s", "poll_timeout_s", "max_sample_age_s"):
                value = settings[name]
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be a positive finite number")
                setattr(self, name, value)
            self.elastic = settings.get("elastic", {"enabled": False})
            if not isinstance(self.elastic, dict) or type(self.elastic.get("enabled")) is not bool:
                raise ValueError("elastic.enabled must be boolean")
            if self.elastic["enabled"]:
                backlog = self.elastic["prefill_backlog_tokens"]
                if type(backlog) is not int or backlog < 0:
                    raise ValueError("prefill_backlog_tokens must be a nonnegative integer")
                for name in ("tpot_budget_ms", "step_sample_max_age_ms"):
                    value = self.elastic[name]
                    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                        raise ValueError(f"{name} must be a positive finite number")
                margin = self.elastic["tpot_margin_ms"]
                if type(margin) not in (int, float) or not math.isfinite(margin) or margin < 0:
                    raise ValueError("tpot_margin_ms must be a nonnegative finite number")
                if type(self.elastic["allow_idle_without_sample"]) is not bool:
                    raise ValueError("allow_idle_without_sample must be boolean")
            self.infos = {}
            self.edges = {}
            self.samples = {}
            self.p_bookings = {p: {"tokens": 0, "requests": 0, "blocks": 0, "early_requests": 0,
                                   "transfer_bytes": 0} for p in self.prefills}
            self.d_bookings = {d: {"blocks": 0, "requests": 0, "transfer_bytes": 0} for d in self.decodes}
            self.connection_bookings = {endpoint: 0 for endpoint in self.prefills + self.decodes}
            self.poll_task = None
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
        if self.routing_mode == "load":
            await self._start_load()
            return
        spec = _model_spec(self.config)
        seeds = set()
        for pe, de in self.pairs:
            pi, di = await asyncio.gather(PDClient.info(pe), PDClient.info(de))
            for info, role in ((pi, "prefill"), (di, "decode")):
                _validate_worker_role(info, role)
                if (info["role"] != role or info["model"] != spec
                        or info["max_model_length"] != self.config["max_model_length"]):
                    raise ValueError(f"Incompatible {role} worker model or length")
                if info["request_timeout"] < self.request_timeout:
                    raise ValueError(f"{role} worker timeout is shorter than gateway timeout")
                if _worker_payload_limit(info) > self.max_payload_bytes:
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

    async def _start_load(self):
        spec = _model_spec(self.config)
        endpoints = self.prefills + self.decodes
        infos = await asyncio.gather(*(PDClient.info(endpoint, timeout=self.poll_timeout_s)
                                       for endpoint in endpoints))
        self.infos = dict(zip(endpoints, infos))
        for endpoint in endpoints:
            info = self.infos[endpoint]
            role = "prefill" if endpoint in self.p_bookings else "decode"
            _validate_load_worker(info, role, self.request_timeout, self.max_payload_bytes)
            if info["model"] != spec or info["max_model_length"] != self.config["max_model_length"]:
                raise ValueError(f"Incompatible {role} worker model or length")
        for pe in self.prefills:
            pi = self.infos[pe]
            allowed = {_endpoint(endpoint) for endpoint in pi["allowed_decode_endpoints"]}
            if _endpoint(pi["decode_endpoint"]) not in allowed:
                raise ValueError("Prefill worker's default decode endpoint is not allowed")
            self.edges[pe] = [de for de in self.decodes if de in allowed]
            if not self.edges[pe]:
                raise ValueError("Prefill worker has no allowed configured decode endpoint")
            for de in self.edges[pe]:
                di = self.infos[de]
                _validate_load_edge(pi, di, self.pipeline_min_tokens)
        if self.elastic["enabled"] and not any(self.infos[de]["local_prefill_enabled"]
                                               for targets in self.edges.values() for de in targets):
            raise ValueError("Elastic routing requires a reachable chunked decode worker")
        tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained, self.config["model_name_or_path"],
            revision=self.config["tokenizer_revision"], cache_dir=self.config.get("cache_dir"),
        )
        self.engine = SimpleNamespace(tokenizer=tokenizer)
        await self._poll_once()
        self.poll_task = asyncio.create_task(self._poll())

    async def _poll_once(self):
        async def sample(endpoint):
            role = "prefill" if endpoint in self.p_bookings else "decode"
            # Only admissions completed before this poll are covered by its snapshot.
            covered = [entry for entry in self.requests.values()
                       if ((role == "prefill" and entry.pair[0] == endpoint and entry.p_booked and entry.p_admitted)
                           or (role == "decode" and entry.pair[1] == endpoint and entry.d_booked and entry.d_admitted))]
            connections = [entry for entry in self.requests.values()
                           if ((role == "prefill" and entry.pair[0] == endpoint and entry.pc_booked and entry.p_admitted)
                               or (role == "decode" and entry.pair[1] == endpoint and entry.dc_booked and entry.d_opened))]
            started = time.monotonic()
            try:
                stats = await PDClient.stats(endpoint, timeout=self.poll_timeout_s)
            except (OSError, ConnectionError, TimeoutError, asyncio.IncompleteReadError,
                    PDRemoteError, ProtocolError):
                self.samples.pop(endpoint, None)
            else:
                self.samples[endpoint] = (stats, started)
                for entry in covered:
                    self._release_booking(entry, role)
                for entry in connections:
                    self._release_booking(entry, role + "_connection")

        await asyncio.gather(*(sample(endpoint) for endpoint in self.prefills + self.decodes))

    async def _poll(self):
        while True:
            await asyncio.sleep(self.poll_interval_s)
            await self._poll_once()

    def _sample(self, endpoint, now):
        sample = self.samples.get(endpoint)
        if sample is None or now - sample[1] > self.max_sample_age_s:
            return None
        return sample

    def _select_load(self, pending, pipeline):
        now = time.monotonic()
        limit = pending.params.max_model_length or self.config["max_model_length"]
        blocks = math.ceil(min(len(pending.token_ids) + pending.params.max_tokens, limit)
                           / self.config["block_size"])
        transfer_bytes = (payload_nbytes(self.infos[self.prefills[0]]["model"], len(pending.token_ids))
                          if pending.params.max_tokens > 1 and len(pending.token_ids) + 1 < limit else 0)
        prompt_blocks = math.ceil(len(pending.token_ids) / self.config["block_size"])
        sampling = asdict(pending.params)
        local_header = _header_bytes({
            "op": "open_local", "lifetime": .1, "request_id": pending.request_id,
            "token_ids": pending.token_ids, "sampling": sampling,
        })
        choices = []
        for pe in self.prefills:
            psample = self._sample(pe, now)
            if psample is None:
                continue
            pstats, ptime = psample
            pbooking = self.p_bookings[pe]
            pscore = (pstats["pending_prefill_tokens"] + pbooking["tokens"],
                      pstats["admitted_requests"] + pbooking["requests"])
            pblocks = pstats["pending_prompt_blocks"] + pbooking["blocks"]
            pcapacity = (not (pipeline or pstats["reserved_prompt_blocks"] or pbooking["early_requests"])
                         or pblocks + prompt_blocks <= pstats["total_blocks"])
            pconnections = pstats["open_handles"] + self.connection_bookings[pe]
            pbytes = pstats["reserved_transfer_bytes"] + pbooking["transfer_bytes"]
            pmeta, dmeta = (_metadata_header_bounds(pending, self.infos[pe], limit, pipeline)
                            if transfer_bytes else (0, 0))
            decodes = []
            for de in self.edges[pe]:
                dsample = self._sample(de, now)
                if dsample is None:
                    continue
                dstats, dtime = dsample
                dbooking = self.d_bookings[de]
                dscore = (dstats["reserved_blocks"] + dbooking["blocks"],
                          dstats["resident_requests"] + dbooking["requests"])
                dconnections = dstats["open_handles"] + self.connection_bookings[de]
                dbytes = dstats["reserved_transfer_bytes"] + dbooking["transfer_bytes"]
                if (dscore[0] + blocks > dstats["total_blocks"]
                        or dscore[1] + 1 > dstats["resident_limit"]
                        or dconnections + 1 > self.infos[de]["max_connections"]):
                    continue
                popen = _header_bytes({
                    "op": "prepare_p" if pipeline else "open_p", "lifetime": .1,
                    "session_id": "0" * 32, "request_id": pending.request_id,
                    "token_ids": pending.token_ids, "sampling": sampling, "decode_endpoint": de,
                })
                pheader = max(popen, pmeta)
                can_transfer = (pbytes + transfer_bytes <= self.infos[pe]["max_transfer_bytes"]
                                and dbytes + transfer_bytes <= self.infos[de]["max_transfer_bytes"]
                                and transfer_bytes <= self.infos[pe].get("max_payload_bytes",
                                                                          self.infos[pe]["max_transfer_bytes"])
                                and transfer_bytes <= self.infos[de].get("max_payload_bytes",
                                                                          self.infos[de]["max_transfer_bytes"])
                                and transfer_bytes <= self.max_payload_bytes
                                and pconnections + 1 <= self.infos[pe]["max_connections"]
                                and pcapacity
                                and pheader <= min(self.infos[pe]["max_header_bytes"], DEFAULT_HEADER_LIMIT)
                                and dmeta <= min(self.infos[de]["max_header_bytes"], DEFAULT_HEADER_LIMIT))
                decodes.append((de, dstats, dtime, dscore, can_transfer, dconnections, dbytes))
            if decodes:
                choices.append((pe, ptime, pscore, pblocks, pconnections, pbytes, decodes))
        for pe, ptime, pscore, pblocks, pconnections, pbytes, decodes in sorted(choices, key=lambda item: item[2]):
            eligible = []
            if self.elastic["enabled"] and pscore[0] >= self.elastic["prefill_backlog_tokens"]:
                for de, stats, received, score, _, dconnections, dbytes in decodes:
                    if not self.infos[de]["local_prefill_enabled"]:
                        continue
                    if local_header > min(self.infos[de]["max_header_bytes"], DEFAULT_HEADER_LIMIT):
                        continue
                    wall = stats["decode_step_wall_ms"]
                    age = stats["decode_step_sample_age_ms"]
                    if age is not None:
                        age += (now - received) * 1000
                    fresh = wall is not None and age is not None and age <= self.elastic["step_sample_max_age_ms"]
                    idle = self.elastic["allow_idle_without_sample"] and score == (0, 0)
                    if ((fresh and wall + self.elastic["tpot_margin_ms"] <= self.elastic["tpot_budget_ms"])
                            or (not fresh and idle)):
                        eligible.append((de, stats, received, score, age, dconnections, dbytes))
            if eligible:
                de, stats, dtime, dscore, age, dconnections, dbytes = min(eligible, key=lambda item: item[3])
                route = "local"
                break
            remote = [candidate for candidate in decodes if candidate[4]]
            if remote:
                de, stats, dtime, dscore, _, dconnections, dbytes = min(remote, key=lambda item: item[3])
                age = stats["decode_step_sample_age_ms"]
                if age is not None:
                    age += (now - dtime) * 1000
                route = "remote"
                break
        else:
            raise GatewayError(503, "No fresh PD route with estimated capacity")
        decision = {
            "prefill_pending_tokens": pscore[0], "prefill_requests": pscore[1],
            "prefill_prompt_blocks": pblocks, "request_prompt_blocks": prompt_blocks,
            "decode_reserved_blocks": dscore[0], "decode_resident_requests": dscore[1],
            "prefill_open_handles": pconnections, "decode_open_handles": dconnections,
            "prefill_connection_limit": self.infos[pe]["max_connections"],
            "decode_connection_limit": self.infos[de]["max_connections"],
            "prefill_reserved_transfer_bytes": pbytes, "decode_reserved_transfer_bytes": dbytes,
            "request_blocks": blocks, "estimated_remote_payload_bytes": transfer_bytes,
            "prefill_sample_age_ms": (now - ptime) * 1000,
            "decode_sample_age_ms": (now - dtime) * 1000,
            "decode_step_wall_ms": stats["decode_step_wall_ms"],
            "decode_step_sample_age_ms": age,
            "decode_step_sample_count": stats["decode_step_sample_count"],
            "decode_step_sample_source": stats["decode_step_sample_source"],
        }
        if self.elastic["enabled"]:
            decision.update(
                prefill_backlog_tokens=self.elastic["prefill_backlog_tokens"],
                tpot_budget_ms=self.elastic["tpot_budget_ms"],
                tpot_margin_ms=self.elastic["tpot_margin_ms"],
                step_sample_max_age_ms=self.elastic["step_sample_max_age_ms"],
                allow_idle_without_sample=self.elastic["allow_idle_without_sample"],
                decode_idle=dscore == (0, 0),
                step_sample_fresh=(stats["decode_step_wall_ms"] is not None and age is not None
                                   and age <= self.elastic["step_sample_max_age_ms"]),
            )
        return (pe, de), route, blocks, decision

    def _release_booking(self, entry, role):
        if role == "prefill" and entry.p_booked:
            booking = self.p_bookings[entry.pair[0]]
            booking["tokens"] -= len(entry.pending.token_ids)
            booking["requests"] -= 1
            booking["blocks"] -= entry.prompt_blocks
            booking["early_requests"] -= int(entry.pipeline)
            booking["transfer_bytes"] -= entry.transfer_bytes
            entry.p_booked = False
        elif role == "decode" and entry.d_booked:
            booking = self.d_bookings[entry.pair[1]]
            booking["blocks"] -= entry.blocks
            booking["requests"] -= 1
            booking["transfer_bytes"] -= entry.transfer_bytes
            entry.d_booked = False
        elif role == "prefill_connection" and entry.pc_booked:
            self.connection_bookings[entry.pair[0]] -= 1
            entry.pc_booked = False
        elif role == "decode_connection" and entry.dc_booked:
            self.connection_bookings[entry.pair[1]] -= 1
            entry.dc_booked = False

    def claim(self):
        if self.closing or self.claimed >= self.max_active_requests:
            return False
        self.claimed += 1
        return True

    def unclaim(self):
        self.claimed -= 1

    def record_admission_failure(self, request_id, arrival, *, status, reason, cancelled=False):
        pending = SimpleNamespace(request_id=request_id, arrival=arrival)
        entry = SimpleNamespace(pending=pending, gateway_id=uuid.uuid4().hex,
                                session_id=None, transaction_id=None)
        self._record(entry, "pd_gateway_cancelled" if cancelled else "pd_gateway_failed",
                     phase="admission", route="unassigned", status=status, reason=reason)

    def submit(self, pending):
        if self.closing:
            self.record_admission_failure(pending.request_id, pending.arrival, status=503,
                                          reason="PD gateway is shutting down")
            raise GatewayError(503, "PD gateway is shutting down")
        if not pending.token_ids:
            self.record_admission_failure(pending.request_id, pending.arrival, status=400,
                                          reason="Prompt requires at least one token")
            raise ValueError("Prompt requires at least one token")
        if len(pending.token_ids) >= self.config["max_model_length"]:
            self.record_admission_failure(pending.request_id, pending.arrival, status=400,
                                          reason="Prompt must leave room for completion")
            raise ValueError("Prompt must leave room for completion")
        seq = Sequence(pending.token_ids, self.config["block_size"], pending.params)
        limit = min(pending.params.max_tokens, self.config["max_model_length"] - len(pending.token_ids))
        pending.tokens = asyncio.Queue(maxsize=limit + 1)
        total_limit = pending.params.max_model_length or self.config["max_model_length"]
        pipeline = (self.pipeline_min_tokens > 0
                    and len(pending.token_ids) >= self.pipeline_min_tokens
                    and pending.params.max_tokens > 1 and len(pending.token_ids) + 1 < total_limit)
        gateway_id = uuid.uuid4().hex
        if self.routing_mode == "load":
            try:
                pair, route, blocks, decision = self._select_load(pending, pipeline)
            except GatewayError as exc:
                failed = SimpleNamespace(pending=pending, gateway_id=gateway_id,
                                         session_id=None, transaction_id=None)
                self._record(failed, "pd_gateway_failed", phase="routing", route="unassigned",
                             status=exc.status, reason=str(exc))
                raise
        else:
            pair = self.pairs[self.next_pair]
            self.next_pair = (self.next_pair + 1) % len(self.pairs)
            route, blocks, decision = "remote", 0, {}
        entry = SimpleNamespace(
            seq=seq, pending=pending, pair=pair, gateway_id=gateway_id,
            session_id=None, transaction_id=None, frontend_done=False, remote_done=False,
            finishing=False, cancelled=False, task=None, route=route, blocks=blocks,
            prompt_blocks=math.ceil(len(pending.token_ids) / self.config["block_size"]), pipeline=pipeline,
            transfer_bytes=decision.get("estimated_remote_payload_bytes", 0) if route == "remote" else 0,
            p_booked=False, d_booked=False, pc_booked=False, dc_booked=False,
            p_admitted=False, d_admitted=False, d_opened=False,
        )
        if self.routing_mode == "load":
            if route == "remote":
                self.p_bookings[pair[0]]["tokens"] += len(pending.token_ids)
                self.p_bookings[pair[0]]["requests"] += 1
                self.p_bookings[pair[0]]["blocks"] += entry.prompt_blocks
                self.p_bookings[pair[0]]["early_requests"] += int(entry.pipeline)
                self.p_bookings[pair[0]]["transfer_bytes"] += entry.transfer_bytes
                entry.p_booked = True
                self.connection_bookings[pair[0]] += 1
                entry.pc_booked = True
            self.d_bookings[pair[1]]["blocks"] += blocks
            self.d_bookings[pair[1]]["requests"] += 1
            self.d_bookings[pair[1]]["transfer_bytes"] += entry.transfer_bytes
            entry.d_booked = True
            self.connection_bookings[pair[1]] += 1
            entry.dc_booked = True
        try:
            fields = {"prefill": list(pair[0]) if route == "remote" else None, "decode": list(pair[1])}
            if self.routing_mode == "load":
                fields.update(route=route, considered_prefill=list(pair[0]), decision=decision)
            self._record(entry, "pd_gateway_assigned", **fields)
            self.requests[seq.seq_id] = entry
            entry.task = asyncio.create_task(self._run(entry))
            entry.task.add_done_callback(lambda task: self._task_done(entry, task))
        except BaseException:
            self.requests.pop(seq.seq_id, None)
            if self.routing_mode == "load":
                self._release_booking(entry, "prefill")
                self._release_booking(entry, "decode")
                self._release_booking(entry, "prefill_connection")
                self._release_booking(entry, "decode_connection")
            raise
        return seq

    def _task_done(self, entry, task):
        # A task cancelled before its coroutine starts cannot run its finally.
        if self.routing_mode == "load":
            self._release_booking(entry, "prefill")
            self._release_booking(entry, "decode")
            self._release_booking(entry, "prefill_connection")
            self._release_booking(entry, "decode_connection")
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
        staging = 0
        cancelled = False
        error = None
        phase = "open_d"
        pipeline = entry.pipeline
        deadline = asyncio.get_running_loop().time() + self.request_timeout

        def reserved():
            entry.d_admitted = True

        try:
            if entry.route == "local":
                phase = "open_local"
                decode = await PDClient.open_local(
                    entry.pair[1], entry.pending.request_id, entry.pending.token_ids, entry.pending.params,
                    lifetime=deadline - asyncio.get_running_loop().time(),
                    max_header_bytes=self.infos[entry.pair[1]]["max_header_bytes"],
                )
                entry.d_admitted = True
                entry.d_opened = True
                entry.session_id = decode.session_id
                self._record(entry, "pd_gateway_open", prefill=None, decode=list(entry.pair[1]),
                             route="local", payload_bytes=0)
                phase = "local"
                terminal = False
                while not terminal:
                    event = await decode.next_token()
                    terminal = self._publish(entry, event)
            else:
                dheader = ({"max_header_bytes": self.infos[entry.pair[1]]["max_header_bytes"]}
                           if self.routing_mode == "load" else {})
                decode = await PDClient.open_d(entry.pair[1], lifetime=deadline - asyncio.get_running_loop().time(),
                                               **dheader)
                entry.d_opened = True
                entry.session_id = decode.session_id
                phase = "open_p"
                open_prefill = PDClient.prepare_p if pipeline else PDClient.open_p
                selected = ({"decode_endpoint": entry.pair[1],
                             "max_header_bytes": self.infos[entry.pair[0]]["max_header_bytes"]}
                            if self.routing_mode == "load" else {})
                prefill = await open_prefill(
                    entry.pair[0], decode.session_id, entry.pending.request_id, entry.pending.token_ids,
                    entry.pending.params, lifetime=deadline - asyncio.get_running_loop().time(), **selected,
                )
                entry.p_admitted = True
                entry.transaction_id = prefill.transaction_id
                fields = {"route": "remote"} if self.routing_mode == "load" else {}
                self._record(entry, "pd_gateway_open", prefill=list(entry.pair[0]), decode=list(entry.pair[1]),
                             transfer_mode="pipeline" if pipeline else "bulk", **fields)
                if pipeline:
                    phase = "prepare"
                    notification = {"on_reserved": reserved} if self.routing_mode == "load" else {}
                    await PDClient.start_pipeline(prefill, decode, **notification)
                    entry.d_admitted = True
                phase = "first"
                event = await prefill.next_token()
                terminal = self._publish(entry, event)
                if terminal and self.routing_mode == "load":
                    self._release_booking(entry, "prefill")
            if entry.route == "remote" and not terminal:
                phase = "describe"
                metadata = await prefill.describe()
                validate_metadata(metadata, decode.session_id, _model_spec(self.config),
                                  self.config["max_model_length"])
                if (metadata["transaction_id"] != prefill.transaction_id
                        or metadata["request_id"] != entry.pending.request_id):
                    raise ProtocolError("PD metadata does not match the request")
                if pipeline:
                    data_per_token, scale_per_token = token_byte_counts(metadata["model"])
                    tokens = metadata["model"]["architecture"]["num_layers"] * metadata["prompt_length"]
                    data_bytes, scale_bytes = tokens * data_per_token, tokens * scale_per_token
                    nbytes = data_bytes + scale_bytes
                else:
                    nbytes = metadata["kv"]["nbytes"]
                    data_bytes = metadata["kv"].get("data_nbytes", nbytes)
                    scale_bytes = metadata["kv"].get("scale_nbytes", 0)
                components = {"data_bytes": data_bytes, "scale_bytes": scale_bytes} if scale_bytes else {}
                if not pipeline:
                    if nbytes > self.max_payload_bytes or self.staging_bytes + nbytes > self.max_staging_bytes:
                        raise GatewayError(503, "PD gateway transfer capacity exhausted")
                    self.staging_bytes += nbytes
                    staging = nbytes
                phase = "handoff"
                handoff_start = time.perf_counter()
                self._record(entry, "pd_gateway_handoff_start", payload_bytes=nbytes, **components)
                handoff = PDClient.handoff_pipeline if pipeline else PDClient.handoff
                notification = {"on_reserved": reserved} if self.routing_mode == "load" and not pipeline else {}
                await handoff(prefill, decode, metadata, timing_path=self.config.get("timing_path"),
                              **notification)
                entry.d_admitted = True
                if self.routing_mode == "load":
                    self._release_booking(entry, "prefill")
                self._record(entry, "pd_gateway_handoff_done", payload_bytes=nbytes,
                             handoff_ms=(time.perf_counter() - handoff_start) * 1000, **components)
                if staging:
                    self.staging_bytes -= staging
                    staging = 0
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
            if self.routing_mode == "load":
                self._release_booking(entry, "prefill")
                self._release_booking(entry, "decode")
                self._release_booking(entry, "prefill_connection")
                self._release_booking(entry, "decode_connection")
            if staging:
                self.staging_bytes -= staging
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
        if self.routing_mode == "load" and self.poll_task is not None:
            self.poll_task.cancel()
            await asyncio.gather(self.poll_task, return_exceptions=True)
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
