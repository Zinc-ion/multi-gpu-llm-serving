"""Check configured PD worker endpoints and the started HTTP gateway."""

import argparse
import asyncio
import json

import httpx

from myvllm.pd.client import PDClient
from myvllm.pd.gateway import PDGateway, _validate_load_edge, _validate_load_worker, _validate_worker_role, _worker_payload_limit
from myvllm.pd.wire import DEFAULT_HEADER_LIMIT


async def check(settings, base_url):
    pairs = []
    if settings.get("routing_mode", "fixed") not in ("fixed", "load"):
        raise ValueError("routing_mode must be fixed or load")
    if settings.get("routing_mode", "fixed") == "load":
        gateway = PDGateway({}, settings)
        prefills, decodes = gateway.prefills, gateway.decodes
        endpoints = prefills + decodes
        infos = await asyncio.gather(*(PDClient.info(endpoint, timeout=settings["poll_timeout_s"])
                                       for endpoint in endpoints))
        inventory = dict(zip(endpoints, infos))
        for endpoint in endpoints:
            role = "prefill" if endpoint in prefills else "decode"
            info = inventory[endpoint]
            _validate_load_worker(info, role, settings["request_timeout"], settings["max_payload_bytes"])
            if (info["model"] != infos[0]["model"]
                    or info["max_model_length"] != infos[0]["max_model_length"]):
                raise ValueError("Incompatible model or length in worker inventory")
        edges = []
        for pe in prefills:
            p = inventory[pe]
            allowed = {tuple(endpoint) for endpoint in p["allowed_decode_endpoints"]}
            reachable = [de for de in decodes if de in allowed]
            if p["role"] != "prefill" or not reachable or tuple(p["decode_endpoint"]) not in allowed:
                raise ValueError("Prefill worker role or allowlist does not match inventory")
            for de in reachable:
                d = inventory[de]
                _validate_load_edge(p, d, settings.get("pipeline_min_tokens", 0))
                edges.append({"prefill": list(pe), "decode": list(de)})
        if gateway.elastic["enabled"] and not any(
                inventory[tuple(edge["decode"])]["local_prefill_enabled"] for edge in edges):
            raise ValueError("Elastic routing requires a reachable chunked decode worker")
        pairs = [{"prefill": inventory[tuple(pair["prefill"])],
                  "decode": inventory[tuple(pair["decode"])]} for pair in settings["pairs"]]
    else:
        for pair in settings["pairs"]:
            p, d = await asyncio.gather(PDClient.info(tuple(pair["prefill"])),
                                        PDClient.info(tuple(pair["decode"])))
            if p["role"] != "prefill" or d["role"] != "decode" or p["decode_endpoint"] != pair["decode"]:
                raise ValueError("Worker role or fixed pair does not match configuration")
            for info, role in ((p, "prefill"), (d, "decode")):
                _validate_worker_role(info, role)
                if _worker_payload_limit(info) > settings["max_payload_bytes"]:
                    raise ValueError(f"{role} worker body limit exceeds gateway payload limit")
            pairs.append({"prefill": p, "decode": d})
    if base_url:
        async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
            response = await client.get(base_url.rstrip("/") + "/openapi.json")
            response.raise_for_status()
            paths = response.json()["paths"]
            if "/v1/completions" not in paths or "/v1/chat/completions" not in paths:
                raise ValueError("Gateway endpoints are missing")
    result = {"pairs": pairs, "http": "ready" if base_url else "not checked"}
    if settings.get("routing_mode", "fixed") == "load":
        result["allowed_edges"] = edges
        result["control_header_limits"] = [
            {**edge, "prefill_max_bytes": inventory[tuple(edge["prefill"])]["max_header_bytes"],
             "decode_max_bytes": inventory[tuple(edge["decode"])]["max_header_bytes"],
             "client_max_bytes": DEFAULT_HEADER_LIMIT}
            for edge in edges
        ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-config", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--workers-only", action="store_true")
    args = parser.parse_args()
    with open(args.gateway_config, encoding="utf-8") as file:
        settings = json.load(file)
    print(json.dumps(asyncio.run(check(settings, None if args.workers_only else args.base_url)), indent=2))


if __name__ == "__main__":
    main()
