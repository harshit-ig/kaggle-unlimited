"""Client-facing reverse proxy: one stable URL in front of a rotating set of kernels.

Clients (Cline, aider, Continue, the OpenAI SDK) point at `http://127.0.0.1:8317/v1`
and never learn that the upstream changed. Rotation swaps the active upstream
atomically; a retired upstream is kept alive for a grace period so streams that were
already in flight finish on the session they started on.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

# Headers that describe a single hop and must not be forwarded.
log = logging.getLogger("kaggle_rotate.proxy")

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "accept-encoding",
}


@dataclass
class Upstream:
    name: str
    url: str
    token: str = ""
    model: str = ""
    account: str = ""
    kernel_ref: str = ""
    healthy: bool = False
    last_ok: float = 0.0
    last_error: str = ""
    retired_at: float | None = None
    sessions_in_flight: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def mark_retired(self) -> None:
        self.retired_at = time.time()

    def is_draining(self, grace: float) -> bool:
        return self.retired_at is not None and (time.time() - self.retired_at) < grace

    def is_stale(self, grace: float) -> bool:
        return self.retired_at is not None and (time.time() - self.retired_at) >= grace

    def to_public(self, grace: float = 0.0) -> dict[str, Any]:
        return {
            "name": self.name,
            "account": self.account,
            "kernel_ref": self.kernel_ref,
            "url": self.url,
            "model": self.model,
            "healthy": self.healthy,
            "last_ok": self.last_ok,
            "last_error": self.last_error,
            "retired": self.retired_at is not None,
            "draining": self.is_draining(grace) if grace else False,
            **self.meta,
        }


class UpstreamRouter:
    """Holds the active upstream plus the warm standby and any draining sessions."""

    def __init__(self, drain_grace: float = 30.0) -> None:
        self.drain_grace = drain_grace
        self.active: Upstream | None = None
        self.standby: Upstream | None = None
        self._retired: list[Upstream] = []

    # ------------------------------------------------------------- membership

    def add_standby(self, upstream: Upstream) -> None:
        if self.active is not None and self.active.name == upstream.name:
            return
        self.standby = upstream

    def set_active(self, upstream: Upstream) -> Upstream | None:
        previous = self.active
        if previous is not None and previous.name != upstream.name:
            previous.mark_retired()
            self._retired.append(previous)
        self.active = upstream
        if self.standby is not None and self.standby.name == upstream.name:
            self.standby = None
        return previous

    def drop(self, name: str) -> None:
        if self.active is not None and self.active.name == name:
            self.active = None
        if self.standby is not None and self.standby.name == name:
            self.standby = None
        self._retired = [u for u in self._retired if u.name != name]

    def promote_standby(self) -> Upstream | None:
        if self.standby is None:
            return None
        return self.set_active(self.standby)

    # ------------------------------------------------------------- selection

    def sweep(self) -> None:
        self._retired = [u for u in self._retired if not u.is_stale(self.drain_grace)]

    def candidates(self) -> list[Upstream]:
        """Ordered preference: active, then standby, then anything still draining."""
        self.sweep()
        ordered: list[Upstream] = []
        for upstream in (self.active, self.standby, *self._retired):
            if upstream is not None and upstream not in ordered:
                ordered.append(upstream)
        return ordered

    def pick(self, prefer: Upstream | None = None) -> Upstream | None:
        if prefer is not None:
            return prefer
        if self.active is not None:
            return self.active
        return self.standby

    def fallback(self, failed: Upstream) -> Upstream | None:
        for upstream in self.candidates():
            if upstream.name != failed.name and upstream.healthy:
                return upstream
        return None

    def status(self) -> dict[str, Any]:
        return {
            "active": self.active.to_public(self.drain_grace) if self.active else None,
            "standby": self.standby.to_public(self.drain_grace) if self.standby else None,
            "retired": [u.to_public(self.drain_grace) for u in self._retired],
        }


class ProxyApp:
    """Starlette app that forwards everything except its own control routes."""

    def __init__(
        self,
        router: UpstreamRouter,
        *,
        retry_on_standby: bool = True,
        connect_grace: float = 90.0,
        timeout: float = 3600.0,
        on_traffic: Callable[[], None] | None = None,
    ) -> None:
        self.router = router
        self.retry_on_standby = retry_on_standby
        self.connect_grace = connect_grace
        # Lets the pool notice that clients are (or are no longer) using the GPU.
        self.on_traffic = on_traffic
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=30.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            follow_redirects=False,
            # httpx advertises gzip by default, which makes Cloudflare compress every
            # response. We then relay raw bytes, so a client that never asked for gzip
            # would get gzip - fine for the OpenAI SDK, garbage for bare `curl`.
            headers={"Accept-Encoding": "identity"},
        )
        app = Starlette(
            routes=[
                Route(
                    "/{path:path}",
                    self.handle,
                    methods=["GET", "POST", "PUT", "DELETE", "HEAD", "PATCH", "OPTIONS"],
                )
            ]
        )
        self.app = app

    async def aclose(self) -> None:
        await self._client.aclose()

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _preflight(request: Request) -> Response:
        """204 with CORS headers, never touching the upstream."""
        origin = request.headers.get("origin", "*")
        requested = request.headers.get("access-control-request-headers")
        allow_headers = requested or ("Authorization,Content-Type,Accept,Origin,X-Requested-With")
        headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "GET,POST,PUT,PATCH,DELETE,HEAD,OPTIONS",
            "Access-Control-Allow-Headers": allow_headers,
            "Access-Control-Max-Age": "43200",
            "Access-Control-Allow-Credentials": "true",
            # Chrome requires this opt-in for an extension page to reach 127.0.0.1.
            "Access-Control-Allow-Private-Network": "true",
            "Vary": "Origin",
        }
        if origin != "*":
            # Credentials are allowed, so a wildcard would be invalid.
            headers.pop("Access-Control-Allow-Credentials")
        return Response(status_code=204, headers=headers)

    @staticmethod
    def _cors_headers(request: Request) -> dict[str, str]:
        """Echo the caller's Origin so browser clients can read the response."""
        origin = request.headers.get("origin")
        if not origin:
            return {}
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Expose-Headers": "*",
            "Access-Control-Allow-Private-Network": "true",
            "Vary": "Origin",
        }

    @staticmethod
    def _forward_headers(request: Request) -> dict[str, str]:
        headers: dict[str, str] = {}
        for key, value in request.headers.items():
            lowered = key.lower()
            if lowered in HOP_BY_HOP or lowered == "authorization":
                continue
            # Never leak the caller's browser identity to the origin. Cloudflare's edge
            # answers 403 to requests carrying `Origin: chrome-extension://...`, which
            # broke browser-based clients even though we were the actual server.
            if lowered in {"origin", "referer"}:
                continue
            headers[key] = value
        return headers

    async def handle(self, request: Request) -> Response:
        path = request.url.path
        if path == "/_rot/status":
            return JSONResponse(self.router.status())
        if path == "/_rot/health":
            active = self.router.active
            return JSONResponse({"ok": active is not None and active.healthy})

        # Answer CORS preflights here rather than forwarding them. Relaying OPTIONS to
        # the origin buys nothing and lets an intermediary reject it (Cloudflare's edge
        # returns 403 for some preflights), which breaks every browser-based client.
        if request.method == "OPTIONS":
            return self._preflight(request)

        upstream = await self._await_upstream()
        if upstream is not None and self.on_traffic is not None:
            self.on_traffic()
        if upstream is None:
            return JSONResponse(
                {
                    "error": {
                        "message": (
                            f"no Kaggle session became available within {self.connect_grace:.0f}s; "
                            "the pool is still booting one or every account is out of quota"
                        ),
                        "type": "kaggle_rotate_unavailable",
                    }
                },
                status_code=503,
            )

        body = await request.body()
        attempts = [upstream]
        if self.retry_on_standby:
            alt = self.router.fallback(upstream)
            if alt is not None:
                attempts.append(alt)

        last_error: str = "no attempt was made"
        for attempt in attempts:
            try:
                return await self._forward(request, attempt, body)
            except _Retryable as exc:
                # A connect failure, or an upstream that answered 502/503/504. Nothing
                # was delivered to the client, so replaying on the standby is safe.
                last_error = exc.reason
                attempt.last_error = last_error
                continue
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                attempt.healthy = False
                attempt.last_error = last_error
                attempt.sessions_in_flight = max(0, attempt.sessions_in_flight - 1)
                continue

        attempt_name = attempts[-1].name
        return JSONResponse(
            {
                "error": {
                    "message": f"upstream {attempt_name} unreachable: {last_error}",
                    "type": "kaggle_rotate_unreachable",
                }
            },
            status_code=502,
        )

    async def _await_upstream(self) -> Upstream | None:
        """Wait briefly for a session instead of failing during a cutover or boot."""
        deadline = time.monotonic() + self.connect_grace
        while True:
            upstream = self.router.pick()
            if upstream is not None:
                return upstream
            if time.monotonic() >= deadline:
                return None
            await asyncio.sleep(0.1)

    async def _forward(self, request: Request, upstream: Upstream, body: bytes) -> Response:
        target = f"{upstream.url}{request.url.path}"
        if request.url.query:
            target = f"{target}?{request.url.query}"
        headers = self._forward_headers(request)

        upstream.sessions_in_flight += 1
        try:
            stream_ctx = self._client.build_request(
                request.method,
                target,
                headers=headers,
                content=body or None,
            )
            response = await self._client.send(stream_ctx, stream=True)
        except httpx.HTTPError:
            upstream.sessions_in_flight = max(0, upstream.sessions_in_flight - 1)
            raise

        response_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() not in HOP_BY_HOP and key.lower() != "authorization"
        }
        response_headers["X-Kaggle-Rotate-Upstream"] = upstream.name
        for key, value in self._cors_headers(request).items():
            response_headers[key] = value

        status_code = response.status_code

        # Cloudflare quick tunnels abandon an origin response after ~100s, so a long
        # non-streamed generation comes back as a 524 wrapped in HTML. Retrying would
        # burn more GPU quota for a request the model already computed, so translate it
        # into something actionable instead.
        if status_code in (502, 503, 504) and self.retry_on_standby:
            await response.aclose()
            upstream.sessions_in_flight = max(0, upstream.sessions_in_flight - 1)
            raise _Retryable(f"upstream {upstream.name} returned {status_code}")

        if status_code == 524:
            await response.aread()
            await response.aclose()
            upstream.sessions_in_flight = max(0, upstream.sessions_in_flight - 1)
            log.warning(
                "upstream %s timed out at the Cloudflare edge (524): the request took "
                'over ~100s without completing. Use "stream": true, or lower num_predict.',
                upstream.name,
            )
            return JSONResponse(
                {
                    "error": {
                        "message": (
                            "the generation exceeded Cloudflare's ~100s edge timeout for "
                            "non-streamed responses (HTTP 524). The model itself is fine - "
                            'retry with "stream": true, or lower "num_predict".'
                        ),
                        "type": "kaggle_rotate_edge_timeout",
                    }
                },
                status_code=504,
            )

        async def body_iter():
            nonlocal upstream
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()
                upstream.sessions_in_flight = max(0, upstream.sessions_in_flight - 1)

        upstream.healthy = True
        upstream.last_ok = time.time()
        upstream.last_error = ""

        # A bodyless response (204/304/HEAD) carries neither Content-Length nor
        # Transfer-Encoding, so httpx's stream reader cannot tell where the body ends
        # and blocks until the upstream connection closes - which, behind a keep-alive
        # proxy, can be the full 100s idle timeout. Read those eagerly instead.
        if _is_bodyless(request.method, status_code) or not _has_delimited_body(response):
            payload = await response.aread()
            await response.aclose()
            upstream.sessions_in_flight = max(0, upstream.sessions_in_flight - 1)
            return Response(payload, status_code=status_code, headers=response_headers)

        return StreamingResponse(body_iter(), status_code=status_code, headers=response_headers)


# Responses that are defined to carry no body.
BODYLESS_STATUSES = frozenset({204, 205, 304})


def _is_bodyless(method: str, status: int) -> bool:
    return method == "HEAD" or status in BODYLESS_STATUSES or 100 <= status < 200


def _has_delimited_body(response: httpx.Response) -> bool:
    """True when the upstream told us how long the body is.

    Without Content-Length or chunked encoding the only end-of-body signal is the
    connection closing, which is exactly the case that stalls.
    """
    if "content-length" in response.headers:
        return True
    return "chunked" in response.headers.get("transfer-encoding", "").lower()


class _Retryable(Exception):
    """The upstream produced nothing usable, so the standby may serve this request.

    Anything after the first byte reaches the client is *not* retryable: replaying a
    partially delivered generation would corrupt the response and waste GPU quota.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
