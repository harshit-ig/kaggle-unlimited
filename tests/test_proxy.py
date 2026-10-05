"""The proxy must survive a mid-stream cutover without dropping in-flight requests."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from kaggle_rotate.proxy import ProxyApp, Upstream, UpstreamRouter
from tests.helpers import serve, settle


def _fake_ollama(name: str, chunk_delay: float = 0.02):
    """Minimal Ollama surface: /api/ps for residency, /v1/chat/completions streaming."""

    async def ps(_):
        return JSONResponse({"models": [{"name": name, "size_vram": 1}]})

    async def tags(_):
        return JSONResponse({"models": [{"name": name}]})

    async def chat(request):
        payload = await request.json()
        if not payload.get("stream"):

            async def once():
                yield json.dumps({"model": name, "choices": [{"delta": {"content": "ok"}}]}) + "\n"

            return StreamingResponse(once(), media_type="application/x-ndjson")

        async def stream():
            for index in range(6):
                await asyncio.sleep(chunk_delay)
                yield (
                    json.dumps(
                        {"model": name, "choices": [{"delta": {"content": f"{name}:{index}"}}]}
                    )
                    + "\n"
                )

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    return Starlette(
        routes=[
            Route("/api/ps", ps),
            Route("/api/tags", tags),
            Route("/v1/chat/completions", chat, methods=["POST"]),
        ]
    )


async def _collect(client: httpx.AsyncClient, url: str) -> list[dict]:
    events = []
    async with client.stream("POST", url, json={"model": "m", "stream": True}) as response:
        assert response.status_code == 200
        assert response.headers["X-Kaggle-Rotate-Upstream"]
        async for line in response.aiter_lines():
            if line.strip():
                events.append(json.loads(line))
    return events


async def test_inflight_stream_finishes_on_retired_upstream():
    async with serve(_fake_ollama("A"), 0) as url_a, serve(_fake_ollama("B"), 0) as url_b:
        router = UpstreamRouter(drain_grace=5.0)
        router.set_active(Upstream(name="a", url=url_a, healthy=True))
        proxy = ProxyApp(router)
        try:
            transport = httpx.ASGITransport(app=proxy.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
                task = asyncio.create_task(_collect(client, "/v1/chat/completions"))
                await settle(0.05)

                # Cut over mid-stream, exactly as the pool would.
                router.add_standby(Upstream(name="b", url=url_b, healthy=True))
                previous = router.set_active(router.standby)
                assert previous is not None and previous.name == "a"

                old_events = await task
                assert len(old_events) == 6, "in-flight stream was truncated by the cutover"
                assert {e["model"] for e in old_events} == {"A"}

                new_events = await _collect(client, "/v1/chat/completions")
                assert {e["model"] for e in new_events} == {"B"}
        finally:
            await proxy.aclose()


async def test_proxy_503s_with_a_useful_body_when_nothing_is_active():
    router = UpstreamRouter()
    proxy = ProxyApp(router, connect_grace=0.3)
    try:
        transport = httpx.ASGITransport(app=proxy.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
            response = await client.get("/v1/models")
            assert response.status_code == 503
            body = response.json()
            assert body["error"]["type"] == "kaggle_rotate_unavailable"
            assert "no Kaggle session became available" in body["error"]["message"]
    finally:
        await proxy.aclose()


async def test_dead_active_falls_back_to_healthy_standby():
    async with serve(_fake_ollama("B"), 0) as url_b:
        router = UpstreamRouter(drain_grace=5.0)
        # port 1 is not listening: simulates a kernel whose tunnel just died
        router.set_active(Upstream(name="dead", url="http://127.0.0.1:1", healthy=True))
        router.add_standby(Upstream(name="b", url=url_b, healthy=True))
        proxy = ProxyApp(router, timeout=5.0)
        try:
            transport = httpx.ASGITransport(app=proxy.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
                events = await _collect(client, "/v1/chat/completions")
            assert {e["model"] for e in events} == {"B"}
            assert router.active.healthy is False
        finally:
            await proxy.aclose()


async def test_non_streaming_request_relays_json_unchanged():
    async with serve(_fake_ollama("A"), 0) as url_a:
        router = UpstreamRouter()
        router.set_active(Upstream(name="a", url=url_a, healthy=True))
        proxy = ProxyApp(router)
        try:
            transport = httpx.ASGITransport(app=proxy.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
                response = await client.post("/v1/chat/completions", json={"model": "m"})
            assert response.status_code == 200
            assert "A" in response.json()["model"]
        finally:
            await proxy.aclose()


def test_retired_upstreams_expire_after_the_drain_grace():
    router = UpstreamRouter(drain_grace=0.0)
    router.set_active(Upstream(name="a", url="http://a"))
    router.set_active(Upstream(name="b", url="http://b"))
    router.sweep()
    assert router.candidates() == [router.active]
    assert router.active is not None and router.active.name == "b"


@pytest.mark.parametrize("grace", [0.0, 30.0])
def test_status_shape_is_stable(grace: float):
    router = UpstreamRouter(drain_grace=grace)
    router.set_active(Upstream(name="a", url="http://a"))
    status = router.status()
    assert status["active"]["name"] == "a"
    assert status["standby"] is None
    assert status["retired"] == []


async def test_request_waits_for_a_session_instead_of_failing_immediately():
    """A client that calls mid-cutover should wait, not get a 503."""
    router = UpstreamRouter()
    proxy = ProxyApp(router, connect_grace=5.0)
    transport = httpx.ASGITransport(app=proxy.app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
            async with serve(_fake_ollama("B"), 0) as url_b:
                task = asyncio.create_task(
                    client.post("/v1/chat/completions", json={"model": "m", "stream": True})
                )
                await settle(0.2)
                router.set_active(Upstream(name="b", url=url_b, healthy=True))
                response = await asyncio.wait_for(task, timeout=10)
            assert response.status_code == 200
            assert response.headers["X-Kaggle-Rotate-Upstream"] == "b"
    finally:
        await proxy.aclose()


async def test_request_gives_up_after_the_connect_grace():
    router = UpstreamRouter()
    proxy = ProxyApp(router, connect_grace=0.4)
    transport = httpx.ASGITransport(app=proxy.app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
            response = await client.post("/v1/chat/completions", json={"model": "m"})
        assert response.status_code == 503
        assert "within 0s" in response.json()["error"]["message"]
    finally:
        await proxy.aclose()


async def test_responses_are_not_gzipped_for_clients_that_never_asked():
    """httpx advertises gzip by default; Cloudflare then compresses, and we relay the
    raw bytes, so a bare `curl` gets binary garbage. Force identity upstream."""

    async def inspect(request: Request):
        assert request.headers.get("accept-encoding") == "identity", request.headers.get(
            "accept-encoding"
        )
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/v1/models", inspect)])
    async with serve(app, 0) as url:
        router = UpstreamRouter()
        router.set_active(Upstream(name="a", url=url, healthy=True))
        proxy = ProxyApp(router)
        try:
            transport = httpx.ASGITransport(app=proxy.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
                response = await client.get("/v1/models")
            assert response.status_code == 200
            assert response.headers.get("content-encoding") is None
            assert response.json() == {"ok": True}
        finally:
            await proxy.aclose()


async def test_client_content_encoding_header_is_not_forwarded():
    seen: dict[str, str] = {}

    async def inspect(request: Request):
        seen["accept-encoding"] = request.headers.get("accept-encoding", "<none>")
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/v1/models", inspect)])
    async with serve(app, 0) as url:
        router = UpstreamRouter()
        router.set_active(Upstream(name="a", url=url, healthy=True))
        proxy = ProxyApp(router)
        try:
            transport = httpx.ASGITransport(app=proxy.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
                await client.get("/v1/models", headers={"Accept-Encoding": "gzip"})
            assert seen["accept-encoding"] == "identity"
        finally:
            await proxy.aclose()


async def test_cloudflare_524_becomes_an_actionable_error_not_html():
    """A long non-streamed generation dies at Cloudflare's ~100s edge timeout."""

    async def edge_timeout(_):
        return Response(
            "<html>error code: 524 ... A timeout occurred</html>",
            status_code=524,
            media_type="text/html",
        )

    app = Starlette(routes=[Route("/v1/chat/completions", edge_timeout, methods=["POST"])])
    async with serve(app, 0) as url:
        router = UpstreamRouter()
        router.set_active(Upstream(name="a", url=url, healthy=True))
        router.add_standby(Upstream(name="b", url=url, healthy=True))
        proxy = ProxyApp(router)
        try:
            transport = httpx.ASGITransport(app=proxy.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
                response = await client.post("/v1/chat/completions", json={"model": "m"})
            assert response.status_code == 504
            body = response.json()
            assert body["error"]["type"] == "kaggle_rotate_edge_timeout"
            assert '"stream": true' in body["error"]["message"]
            assert "error code: 524" not in response.text  # HTML replaced
        finally:
            await proxy.aclose()


async def test_524_is_not_retried_against_the_standby():
    """The model already burned GPU time; a retry would waste more quota."""
    calls: list[str] = []

    def make(name: str, status: int):
        async def handler(_):
            calls.append(name)
            return Response("boom", status_code=status)

        return Starlette(routes=[Route("/v1/models", handler)])

    async with serve(make("a", 524), 0) as url_a, serve(make("b", 200), 0) as url_b:
        router = UpstreamRouter()
        router.set_active(Upstream(name="a", url=url_a, healthy=True))
        router.add_standby(Upstream(name="b", url=url_b, healthy=True))
        proxy = ProxyApp(router)
        try:
            transport = httpx.ASGITransport(app=proxy.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
                response = await client.get("/v1/models")
            assert response.status_code == 504
            assert calls == ["a"], "a 524 must not be replayed on the standby"
        finally:
            await proxy.aclose()


async def test_502_is_still_retried_on_the_standby():
    """Unlike a 524, a 502 means the upstream never produced anything."""
    calls: list[str] = []

    def make(name: str, status: int):
        async def handler(_):
            calls.append(name)
            return Response("boom", status_code=status)

        return Starlette(routes=[Route("/v1/models", handler)])

    async with serve(make("a", 502), 0) as url_a, serve(make("b", 200), 0) as url_b:
        router = UpstreamRouter()
        router.set_active(Upstream(name="a", url=url_a, healthy=True))
        router.add_standby(Upstream(name="b", url=url_b, healthy=True))
        proxy = ProxyApp(router)
        try:
            transport = httpx.ASGITransport(app=proxy.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
                response = await client.get("/v1/models")
            assert response.status_code == 200
            assert calls == ["a", "b"]
        finally:
            await proxy.aclose()


async def test_preflight_is_answered_locally_and_never_forwarded():
    """Forwarding OPTIONS to the origin only invites an intermediary to reject it."""
    upstream_calls: list[str] = []

    async def watch(request: Request):
        upstream_calls.append(request.method)
        return JSONResponse({"ok": True})

    app = Starlette(
        routes=[Route("/v1/chat/completions", watch, methods=["GET", "POST", "OPTIONS"])]
    )
    async with serve(app, 0) as url:
        router = UpstreamRouter()
        router.set_active(Upstream(name="a", url=url, healthy=True))
        proxy = ProxyApp(router)
        try:
            transport = httpx.ASGITransport(app=proxy.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
                response = await client.options(
                    "/v1/chat/completions",
                    headers={
                        "Origin": "http://localhost:3000",
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "content-type",
                    },
                )
            assert response.status_code == 204
            assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
            assert "POST" in response.headers["access-control-allow-methods"]
            assert "content-type" in response.headers["access-control-allow-headers"]
            assert upstream_calls == [], "preflight leaked to the upstream"
        finally:
            await proxy.aclose()


async def test_preflight_succeeds_even_with_no_upstream_at_all():
    """A browser must not see a 503 before the pool has booted."""
    router = UpstreamRouter()
    proxy = ProxyApp(router, connect_grace=0.2)
    try:
        transport = httpx.ASGITransport(app=proxy.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
            response = await client.options(
                "/v1/chat/completions", headers={"Origin": "http://x.dev"}
            )
        assert response.status_code == 204
    finally:
        await proxy.aclose()


async def test_real_responses_echo_the_origin_for_browser_clients():
    async def echo(request: Request):
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/v1/models", echo)])
    async with serve(app, 0) as url:
        router = UpstreamRouter()
        router.set_active(Upstream(name="a", url=url, healthy=True))
        proxy = ProxyApp(router)
        try:
            transport = httpx.ASGITransport(app=proxy.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
                response = await client.get(
                    "/v1/models", headers={"Origin": "http://localhost:3000"}
                )
            assert response.status_code == 200
            assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
        finally:
            await proxy.aclose()


async def test_bodyless_response_does_not_stall_waiting_for_the_connection_to_close():
    """A 204 has no Content-Length and no chunked encoding.

    Relaying it as a stream blocks until the upstream closes, which behind a keep-alive
    proxy meant every CORS preflight took the full ~100s instead of 0.3s.
    """

    async def no_content(_):
        return Response(status_code=204)

    app = Starlette(routes=[Route("/v1/models", no_content, methods=["GET", "HEAD"])])
    async with serve(app, 0) as url:
        router = UpstreamRouter()
        router.set_active(Upstream(name="a", url=url, healthy=True))
        proxy = ProxyApp(router)
        try:
            transport = httpx.ASGITransport(app=proxy.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
                started = time.monotonic()
                response = await client.get("/v1/models")
                elapsed = time.monotonic() - started
            assert response.status_code == 204
            assert elapsed < 5, f"bodyless response stalled for {elapsed:.1f}s"
        finally:
            await proxy.aclose()


async def test_head_request_does_not_stall():
    async def no_content(_):
        return Response(status_code=200)

    app = Starlette(routes=[Route("/v1/models", no_content, methods=["GET", "HEAD"])])
    async with serve(app, 0) as url:
        router = UpstreamRouter()
        router.set_active(Upstream(name="a", url=url, healthy=True))
        proxy = ProxyApp(router)
        try:
            transport = httpx.ASGITransport(app=proxy.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
                started = time.monotonic()
                response = await client.head("/v1/models")
                elapsed = time.monotonic() - started
            assert response.status_code == 200
            assert elapsed < 5, f"HEAD stalled for {elapsed:.1f}s"
        finally:
            await proxy.aclose()


async def test_304_does_not_stall():
    async def not_modified(_):
        return Response(status_code=304, headers={"ETag": "abc"})

    app = Starlette(routes=[Route("/v1/models", not_modified)])
    async with serve(app, 0) as url:
        router = UpstreamRouter()
        router.set_active(Upstream(name="a", url=url, healthy=True))
        proxy = ProxyApp(router)
        try:
            transport = httpx.ASGITransport(app=proxy.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
                started = time.monotonic()
                response = await client.get("/v1/models")
                elapsed = time.monotonic() - started
            assert response.status_code == 304
            assert response.headers["etag"] == "abc"
            assert elapsed < 5, f"304 stalled for {elapsed:.1f}s"
        finally:
            await proxy.aclose()


async def test_undeclared_body_length_is_relayed_rather_than_stalled():
    """No Content-Length and not chunked: read it eagerly so it cannot hang."""

    async def closing(_):
        return StreamingResponse(iter([b"hello"]), media_type="text/plain")

    app = Starlette(routes=[Route("/v1/models", closing)])
    async with serve(app, 0) as url:
        router = UpstreamRouter()
        router.set_active(Upstream(name="a", url=url, healthy=True))
        proxy = ProxyApp(router)
        try:
            transport = httpx.ASGITransport(app=proxy.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
                started = time.monotonic()
                response = await client.get("/v1/models")
                elapsed = time.monotonic() - started
            assert response.text == "hello"
            assert elapsed < 5
        finally:
            await proxy.aclose()


async def test_browser_origin_is_not_leaked_to_the_upstream():
    """Cloudflare answers 403 to `Origin: chrome-extension://...`.

    A reverse proxy is the server, not a browser client, so the caller's Origin and
    Referer must not be forwarded upstream.
    """
    seen: dict[str, str] = {}

    async def inspect(request: Request):
        seen["origin"] = request.headers.get("origin", "<none>")
        seen["referer"] = request.headers.get("referer", "<none>")
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/v1/models", inspect)])
    async with serve(app, 0) as url:
        router = UpstreamRouter()
        router.set_active(Upstream(name="a", url=url, healthy=True))
        proxy = ProxyApp(router)
        try:
            transport = httpx.ASGITransport(app=proxy.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
                response = await client.get(
                    "/v1/models",
                    headers={
                        "Origin": "chrome-extension://lgohaagjodcckegcclifdkinkhgeldla",
                        "Referer": "chrome-extension://abc/popup.html",
                    },
                )
            assert response.status_code == 200
            assert seen["origin"] == "<none>", f"leaked Origin upstream: {seen}"
            assert seen["referer"] == "<none>", f"leaked Referer upstream: {seen}"
            # ...but the caller still gets its CORS headers back.
            assert response.headers["access-control-allow-origin"] == (
                "chrome-extension://lgohaagjodcckegcclifdkinkhgeldla"
            )
        finally:
            await proxy.aclose()


async def test_private_network_opt_in_is_present_for_chrome_extensions():
    router = UpstreamRouter()
    proxy = ProxyApp(router, connect_grace=0.2)
    try:
        transport = httpx.ASGITransport(app=proxy.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
            response = await client.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "chrome-extension://abc",
                    "Access-Control-Request-Private-Network": "true",
                },
            )
        assert response.status_code == 204
        assert response.headers["access-control-allow-private-network"] == "true"
    finally:
        await proxy.aclose()
