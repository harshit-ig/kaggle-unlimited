"""Control plane the Kaggle kernels call back into.

A kernel publishes its model endpoint by POSTing to `/_rot/register`, then heartbeats
to `/_rot/heartbeat` and acts on whatever command comes back. This is how the local
pool learns a tunnel URL it did not have to scrape, and how it retires a session
without needing an API to kill a running notebook.

Every route except `/healthz` requires the per-launch bearer token, so the public
quick tunnel in front of this app is not an open registration endpoint.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


@dataclass
class RemoteSession:
    token: str
    account: str
    kernel_ref: str
    url: str
    model: str = ""
    gpu: str = ""
    started_at: float = 0.0
    registered_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_error: str = ""
    ollama_alive: bool = True
    command: str = "keepalive"
    command_reason: str = ""
    phase: str = "registered"

    def to_public(self) -> dict[str, Any]:
        return {
            "account": self.account,
            "kernel_ref": self.kernel_ref,
            "url": self.url,
            "model": self.model,
            "gpu": self.gpu,
            "started_at": self.started_at,
            "registered_at": self.registered_at,
            "last_seen": self.last_seen,
            "silent_seconds": round(time.time() - self.last_seen, 1),
            "ollama_alive": self.ollama_alive,
            "command": self.command,
            "command_reason": self.command_reason,
            "phase": self.phase,
            "last_error": self.last_error,
        }


class RelayState:
    """Token-keyed registry of live kernels plus their pending commands."""

    def __init__(self, driver_grace_seconds: float = 120.0) -> None:
        self._sessions: dict[str, RemoteSession] = {}
        self._lock = asyncio.Lock()
        # Separate from any kernel token: only the pool may assert it is driving.
        self.driver_token = secrets.token_urlsafe(32)
        # The relay can outlive the pool (an orphaned cloudflared child answers for a
        # while). A kernel that keeps getting replies from a driverless relay would
        # never notice it had been orphaned, so the pool pulses and we report back
        # whether it is still there.
        self.driver_grace_seconds = driver_grace_seconds
        self._last_pulse = time.time()

    async def pulse(self) -> None:
        async with self._lock:
            self._last_pulse = time.time()

    def driver_alive(self) -> bool:
        return (time.time() - self._last_pulse) < self.driver_grace_seconds

    def driver_silent_seconds(self) -> float:
        return round(time.time() - self._last_pulse, 1)

    async def register(self, token: str, payload: dict[str, Any]) -> RemoteSession:
        async with self._lock:
            session = RemoteSession(
                token=token,
                account=str(payload.get("account", "")),
                kernel_ref=str(payload.get("kernel_ref", "")),
                url=str(payload.get("url", "")).rstrip("/"),
                model=str(payload.get("model", "")),
                gpu=str(payload.get("gpu", "")),
                started_at=float(payload.get("started_at") or time.time()),
                last_seen=time.time(),
                phase="registered",
            )
            if not session.url:
                raise ValueError("register requires a url")
            self._sessions[token] = session
            return session

    async def heartbeat(self, token: str, payload: dict[str, Any]) -> RemoteSession | None:
        async with self._lock:
            session = self._sessions.get(token)
            if session is None:
                return None
            session.last_seen = time.time()
            if payload.get("url"):
                session.url = str(payload["url"]).rstrip("/")
            session.ollama_alive = bool(payload.get("ollama_alive", True))
            session.phase = "serving"
            return session

    async def final(self, token: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            session = self._sessions.get(token)
            if session is not None:
                session.command_reason = str(payload.get("stop_reason", ""))
                session.phase = "stopped"
                session.last_seen = time.time()

    async def get(self, token: str) -> RemoteSession | None:
        return self._sessions.get(token)

    async def command(self, token: str, action: str, reason: str = "") -> RemoteSession | None:
        async with self._lock:
            session = self._sessions.get(token)
            if session is None:
                return None
            session.command = action
            session.command_reason = reason
            return session

    def sessions(self) -> list[RemoteSession]:
        return sorted(self._sessions.values(), key=lambda s: s.registered_at)

    def by_account(self, account: str) -> list[RemoteSession]:
        return [s for s in self.sessions() if s.account == account]


def _bearer(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.query_params.get("token", "")


def build_relay_app(state: RelayState) -> Starlette:
    async def healthz(_: Request) -> JSONResponse:
        return JSONResponse({"ok": True, "sessions": len(state.sessions())})

    async def register(request: Request) -> JSONResponse:
        token = _bearer(request)
        if not token:
            return JSONResponse({"error": "missing token"}, status_code=401)
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        try:
            session = await state.register(token, payload)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, "session": session.to_public()})

    async def heartbeat(request: Request) -> JSONResponse:
        token = _bearer(request)
        if not token:
            return JSONResponse({"error": "missing token"}, status_code=401)
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        session = await state.heartbeat(token, payload)
        if session is None:
            return JSONResponse({"error": "unknown token"}, status_code=404)
        return JSONResponse(
            {
                "ok": True,
                "action": session.command,
                "reason": session.command_reason,
                "session": session.to_public(),
                "driver_alive": state.driver_alive(),
                "driver_silent_seconds": state.driver_silent_seconds(),
            }
        )

    async def final(request: Request) -> JSONResponse:
        token = _bearer(request)
        if not token:
            return JSONResponse({"error": "missing token"}, status_code=401)
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        await state.final(token, payload)
        return JSONResponse({"ok": True})

    async def command(request: Request) -> JSONResponse:
        token = _bearer(request)
        if not token:
            return JSONResponse({"error": "missing token"}, status_code=401)
        session = await state.get(token)
        if session is None:
            return JSONResponse({"error": "unknown token"}, status_code=404)
        return JSONResponse(
            {
                "action": session.command,
                "reason": session.command_reason,
                "session": session.to_public(),
            }
        )

    async def pulse(request: Request) -> JSONResponse:
        if _bearer(request) != state.driver_token:
            return JSONResponse({"error": "bad driver token"}, status_code=401)
        await state.pulse()
        return JSONResponse({"ok": True})

    async def sessions(request: Request) -> JSONResponse:
        if not _bearer(request):
            return JSONResponse({"error": "missing token"}, status_code=401)
        return JSONResponse({"sessions": [s.to_public() for s in state.sessions()]})

    async def retire(request: Request) -> JSONResponse:
        """Pool-side control: ask a session to exit."""
        token = _bearer(request)
        if not token:
            return JSONResponse({"error": "missing token"}, status_code=401)
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        action = str(payload.get("action", "shutdown"))
        reason = str(payload.get("reason", "retired by pool"))
        session = await state.command(token, action, reason)
        if session is None:
            return JSONResponse({"error": "unknown token"}, status_code=404)
        return JSONResponse({"ok": True, "session": session.to_public()})

    return Starlette(
        routes=[
            Route("/healthz", healthz),
            Route("/_rot/register", register, methods=["POST"]),
            Route("/_rot/pulse", pulse, methods=["POST"]),
            Route("/_rot/heartbeat", heartbeat, methods=["POST"]),
            Route("/_rot/final", final, methods=["POST"]),
            Route("/_rot/command", command, methods=["GET"]),
            Route("/_rot/sessions", sessions, methods=["GET"]),
            Route("/_rot/retire", retire, methods=["POST"]),
        ]
    )
