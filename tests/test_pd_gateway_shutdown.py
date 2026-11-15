import asyncio
import json

from test_pd_server import body, drained, service


def test_close_during_request_body_rejects_before_remote_admission(tmp_path):
    async def run():
        async with service(tmp_path, active=1) as (_, gateway, endpoints, _, logs, _, port):
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            data = json.dumps(body(length=2)).encode()
            headers = (
                b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
                b"Content-Type: application/json\r\nConnection: close\r\n"
                b"X-Request-ID: shutdown-upload\r\nContent-Length: "
                + str(len(data)).encode() + b"\r\n\r\n"
            )
            try:
                writer.write(headers + data[:1])
                await writer.drain()
                for _ in range(200):
                    if gateway.claimed:
                        break
                    await asyncio.sleep(.005)
                assert gateway.claimed == 1 and not gateway.requests

                await gateway.close()
                assert gateway.claimed == 1
                writer.write(data[1:])
                await writer.drain()
                response = await asyncio.wait_for(reader.read(), 10)
            finally:
                writer.close()
                await writer.wait_closed()

            await drained(gateway, endpoints)
            records = [json.loads(line) for line in logs.read_text().splitlines()] if logs.exists() else []
            assert response.startswith(b"HTTP/1.1 503"), response[:200]
            assert not any(record["event"] in ("pd_gateway_assigned", "pd_admitted", "pd_released")
                           for record in records)
            assert gateway.claimed == 0 and not gateway.requests

    asyncio.run(run())
