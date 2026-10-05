"""Executes the generated notebook's control cells against a real relay.

This is the contract that is easiest to break silently: the notebook registers,
heartbeats, and must exit when the pool retires it. Rather than assert on strings,
we run the actual generated code.
"""

from __future__ import annotations

import asyncio
import secrets
import sys
from pathlib import Path

from kaggle_rotate.notebook import PREAMBLE, REGISTER_CELL, SUPERVISOR_CELL
from kaggle_rotate.relay import RelayState, build_relay_app
from tests.helpers import serve, settle

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _render(template: str, **values) -> str:
    out = template
    for key, value in values.items():
        out = out.replace(f"@{key}@", str(value))
    return out


def _exec_cell(namespace: dict, source: str) -> None:
    exec(compile(source, "<generated-cell>", "exec"), namespace)  # noqa: S102


async def test_notebook_registers_serves_and_stops_on_command(tmp_path: Path):
    state = RelayState()
    token = secrets.token_urlsafe(16)
    values = dict(
        TOKEN=token,
        ACCOUNT="acct-a",
        KERNEL_REF="acct-a/kaggle-rotate-ollama",
        RELAY_URL="http://PLACEHOLDER",
        MODEL="qwen3.8-27b-uncensored-mtp",
        NUM_CTX=65536,
        DRAFT=2,
        STATUS_PATH=str(tmp_path / "kaggle-rotate"),
        MAX_RUNTIME=30,
        POLL_S=1,
        ORPHAN_GRACE=120,
    )

    async with serve(build_relay_app(state), 0) as relay_url:
        values["RELAY_URL"] = relay_url
        namespace: dict = {
            "PUBLIC_OLLAMA_URL": "http://127.0.0.1:11434",
            # In the real notebook PREWARM_CELL defines this; we skip that cell
            # because it needs a live GPU Ollama server.
            "MODEL_NAME": "qwen3.8-27b-uncensored-mtp",
        }

        _exec_cell(namespace, _render(PREAMBLE, **values))
        # The generated cells use blocking urllib, so they must not run on the
        # event loop that is also serving the relay.
        await asyncio.to_thread(_exec_cell, namespace, _render(REGISTER_CELL, **values))

        registered = await state.get(token)
        assert registered is not None, "notebook never registered"
        assert registered.account == "acct-a"
        assert registered.model == "qwen3.8-27b-uncensored-mtp"
        assert registered.url == "http://127.0.0.1:11434"
        assert (tmp_path / "kaggle-rotate" / "status.json").is_file()

        # The supervisor blocks in a while loop, so drive it off the event loop.
        supervisor = asyncio.create_task(
            asyncio.to_thread(_exec_cell, namespace, _render(SUPERVISOR_CELL, **values))
        )

        await settle(0.4)
        serving = await state.get(token)
        assert serving is not None and serving.phase == "serving"
        assert serving.last_seen > serving.registered_at

        await state.command(token, "shutdown", "rotated to acct-b")
        await asyncio.wait_for(supervisor, timeout=15)
        assert namespace["_stop_reason"] == "rotated to acct-b"

        status = __import__("json").loads((tmp_path / "kaggle-rotate" / "status.json").read_text())
        assert status["phase"] == "stopping"
        assert status["stop_reason"] == "rotated to acct-b"


async def test_relay_rejects_unknown_and_unauthenticated_callers():
    state = RelayState()
    async with serve(build_relay_app(state), 0) as base:
        import httpx

        async with httpx.AsyncClient(base_url=base) as client:
            assert (await client.get("/healthz")).status_code == 200
            assert (await client.post("/_rot/register", json={"url": "x"})).status_code == 401
            unknown = await client.post(
                "/_rot/heartbeat",
                json={},
                headers={"Authorization": "Bearer nope"},
            )
            assert unknown.status_code == 404
            assert (await client.get("/_rot/sessions")).status_code == 401


async def test_register_requires_a_url():
    state = RelayState()
    async with serve(build_relay_app(state), 0) as base:
        import httpx

        async with httpx.AsyncClient(base_url=base) as client:
            response = await client.post(
                "/_rot/register",
                json={"account": "a"},
                headers={"Authorization": "Bearer tok"},
            )
            assert response.status_code == 400
