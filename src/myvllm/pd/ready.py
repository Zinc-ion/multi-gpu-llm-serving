"""Check fixed PD worker endpoints and the started HTTP gateway."""

import argparse
import asyncio
import json

import httpx

from myvllm.pd.client import PDClient


async def check(settings, base_url):
    pairs = []
    for pair in settings["pairs"]:
        p, d = await asyncio.gather(PDClient.info(tuple(pair["prefill"])),
                                    PDClient.info(tuple(pair["decode"])))
        if p["role"] != "prefill" or d["role"] != "decode" or p["decode_endpoint"] != pair["decode"]:
            raise ValueError("Worker role or fixed pair does not match configuration")
        pairs.append({"prefill": p, "decode": d})
    if base_url:
        async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
            response = await client.get(base_url.rstrip("/") + "/openapi.json")
            response.raise_for_status()
            paths = response.json()["paths"]
            if "/v1/completions" not in paths or "/v1/chat/completions" not in paths:
                raise ValueError("Gateway endpoints are missing")
    return {"pairs": pairs, "http": "ready" if base_url else "not checked"}


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
