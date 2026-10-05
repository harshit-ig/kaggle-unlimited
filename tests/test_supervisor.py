"""Executes the generated notebook's control cells against a real relay.

This is the contract that is easiest to break silently: the notebook registers,
heartbeats, and must exit when the pool retires it. Rather than assert on strings,
we run the actual generated code.
"""

from __future__ import annotations

import asyncio
import secrets
import sys
import threading
import time as time_module
from pathlib import Path

import pytest

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


class _FakeTime:
    """Boottime advances through suspend; monotonic does not. Mimics the kernel."""

    CLOCK_BOOTTIME = 7
    CLOCK_MONOTONIC = 1

    def __init__(self) -> None:
        self.boottime = 1000.0
        self.mono = 1000.0
        self.suspend = 6 * 3600

    def clock_gettime(self, clock_id):
        if clock_id == self.CLOCK_BOOTTIME:
            return self.boottime
        if clock_id == self.CLOCK_MONOTONIC:
            return self.mono
        raise AssertionError(f"unexpected clock {clock_id}")

    def monotonic(self):
        return self.mono

    def time(self):
        return time_module.time()

    def sleep(self, seconds):
        self.boottime += seconds
        self.mono += seconds

    def suspend_once(self):
        self.boottime += self.suspend  # only boottime moves


async def test_orphan_grace_survives_suspend(tmp_path: Path):
    """Regression: the orphan timer used CLOCK_MONOTONIC, which stops while the
    machine is suspended. After a suspend or hibernate the notebook thought only
    seconds had passed, never tripped the timer, and ran on to Kaggle's 12h cap
    instead of self-terminating. Suspend must count against the grace period.
    """
    token = secrets.token_urlsafe(16)
    values = dict(
        TOKEN=token,
        ACCOUNT="acct-a",
        KERNEL_REF="acct-a/kaggle-rotate-ollama",
        RELAY_URL="http://127.0.0.1:1",  # nothing there: every heartbeat fails
        MODEL="qwen3.8-27b-uncensored-mtp",
        NUM_CTX=65536,
        DRAFT=2,
        STATUS_PATH=str(tmp_path / "kaggle-rotate"),
        MAX_RUNTIME=999999,  # the hard deadline must NOT be what stops us
        POLL_S=1,
        ORPHAN_GRACE=60,
    )

    fake = _FakeTime()
    namespace: dict = {
        "PUBLIC_OLLAMA_URL": "http://127.0.0.1:11434",
        "MODEL_NAME": "qwen3.8-27b-uncensored-mtp",
        "SESSION_STARTED_AT": time_module.time() - 120,
        "RELAY_URL": values["RELAY_URL"],
        "_write_status": lambda *a, **k: {},
    }
    _exec_cell(namespace, _render(PREAMBLE, **values))
    namespace["_time"] = fake  # take over the clock for the supervisor loop

    calls = {"n": 0}

    def failing_post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            fake.suspend_once()  # machine resumes after a long sleep
        raise OSError("network down")

    namespace["_post"] = failing_post

    result: dict = {}

    def run():
        try:
            _exec_cell(namespace, _render(SUPERVISOR_CELL, **values))
        finally:
            result["done"] = True

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=8)

    assert result.get("done"), (
        "the supervisor loop never exited: it used CLOCK_MONOTONIC, which is frozen "
        "across suspend, so a hibernated machine leaves the kernel running"
    )
    assert namespace["_stop_reason"], "exited without recording a reason"
    assert "relay unreachable" in namespace["_stop_reason"], namespace["_stop_reason"]
    assert str(values["ORPHAN_GRACE"]) in namespace["_stop_reason"]


async def test_orphan_grace_fires_when_the_relay_is_simply_unreachable(tmp_path: Path):
    """The real safety net for a powered-off or hibernated machine.

    The notebook runs on Kaggle, so its own clock keeps advancing even while the
    operator's laptop is off. A monotonic timer is therefore enough here, and this
    test pins that behaviour down so it cannot silently regress.
    """
    values = dict(
        TOKEN="tok",
        ACCOUNT="acct-a",
        KERNEL_REF="acct-a/kaggle-rotate-ollama",
        RELAY_URL="http://127.0.0.1:1",
        MODEL="m",
        NUM_CTX=1024,
        DRAFT=2,
        STATUS_PATH=str(tmp_path / "st"),
        MAX_RUNTIME=999999,
        POLL_S=1,
        ORPHAN_GRACE=3,
    )
    fake = _FakeTime()
    namespace: dict = {
        "PUBLIC_OLLAMA_URL": "http://127.0.0.1:11434",
        "MODEL_NAME": "m",
        "SESSION_STARTED_AT": time_module.time() - 5,
        "RELAY_URL": values["RELAY_URL"],
        "_write_status": lambda *a, **k: {},
    }
    _exec_cell(namespace, _render(PREAMBLE, **values))
    namespace["_time"] = fake
    namespace["_post"] = lambda *a, **k: (_ for _ in ()).throw(OSError("relay gone"))

    done = threading.Event()

    def run():
        try:
            _exec_cell(namespace, _render(SUPERVISOR_CELL, **values))
        finally:
            done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=8)

    assert done.is_set(), "supervisor never gave up on an unreachable relay"
    assert "relay unreachable for 3s" in namespace["_stop_reason"], namespace["_stop_reason"]
    assert fake.boottime - fake.mono == 0, "no suspend here; both clocks should agree"


async def test_kernel_gives_up_when_the_relay_outlives_the_pool(tmp_path: Path):
    """The real orphan hazard: the pool dies but its cloudflared child survives, so the
    relay keeps answering. A kernel that only checks reachability never notices it was
    abandoned and runs on to Kaggle's 12h cap. It must act on driver_alive.
    """
    values = dict(
        TOKEN="tok",
        ACCOUNT="acct-a",
        KERNEL_REF="acct-a/kaggle-rotate-ollama",
        RELAY_URL="http://relay.invalid",
        MODEL="m",
        NUM_CTX=1024,
        DRAFT=2,
        STATUS_PATH=str(tmp_path / "st"),
        MAX_RUNTIME=999999,
        POLL_S=1,
        ORPHAN_GRACE=2,
    )

    fake = _FakeTime()
    namespace: dict = {
        "PUBLIC_OLLAMA_URL": "http://model",
        "MODEL_NAME": "m",
        "SESSION_STARTED_AT": time_module.time() - 5,
        "RELAY_URL": values["RELAY_URL"],
        "_write_status": lambda *a, **k: {},
    }
    _exec_cell(namespace, _render(PREAMBLE, **values))
    namespace["_time"] = fake

    # A reachable relay with no pool behind it: replies arrive, driver_alive is false.
    namespace["_post"] = lambda *a, **k: {
        "action": "keepalive",
        "driver_alive": False,
        "driver_silent_seconds": 600,
    }

    done = threading.Event()

    def run():
        try:
            _exec_cell(namespace, _render(SUPERVISOR_CELL, **values))
        finally:
            done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=10)

    assert done.is_set(), "kernel kept running against a driverless relay"
    assert "local pool disappeared" in namespace["_stop_reason"], namespace["_stop_reason"]


async def test_kernel_keeps_running_while_the_driver_is_alive(tmp_path: Path):
    """The other side of the same contract: a healthy pool must not be mistaken for a
    dead one, or every rotation would kill the session it just cut over to."""
    values = dict(
        TOKEN="tok",
        ACCOUNT="acct-a",
        KERNEL_REF="acct-a/kaggle-rotate-ollama",
        RELAY_URL="http://relay.invalid",
        MODEL="m",
        NUM_CTX=1024,
        DRAFT=2,
        STATUS_PATH=str(tmp_path / "st"),
        MAX_RUNTIME=999999,
        POLL_S=1,
        ORPHAN_GRACE=100000,  # would never trip in this test
    )
    fake = _FakeTime()
    namespace: dict = {
        "PUBLIC_OLLAMA_URL": "http://model",
        "MODEL_NAME": "m",
        "SESSION_STARTED_AT": time_module.time() - 5,
        "RELAY_URL": values["RELAY_URL"],
        "_write_status": lambda *a, **k: {},
    }
    _exec_cell(namespace, _render(PREAMBLE, **values))
    namespace["_time"] = fake

    polls = {"n": 0}

    def healthy_post(*a, **k):
        polls["n"] += 1
        if polls["n"] > 3:
            raise SystemExit("still running after 3 healthy heartbeats")
        return {"action": "keepalive", "driver_alive": True, "driver_silent_seconds": 0}

    namespace["_post"] = healthy_post

    try:
        await asyncio.to_thread(_exec_cell, namespace, _render(SUPERVISOR_CELL, **values))
    except SystemExit:
        pass
    else:  # pragma: no cover
        pytest.fail("the loop exited while the driver was alive")
    assert polls["n"] > 3


async def test_relay_reports_a_live_driver_when_pulsed():
    import httpx

    state = RelayState(driver_grace_seconds=30.0)
    async with serve(build_relay_app(state), 0) as base:
        async with httpx.AsyncClient(base_url=base) as client:
            # Wrong token cannot claim to be the driver.
            assert (
                await client.post("/_rot/pulse", headers={"Authorization": "Bearer nope"}, json={})
            ).status_code == 401
            assert state.driver_alive() is True  # fresh

            await state.register("t", {"account": "a", "url": "http://model"})
            beat = await client.post(
                "/_rot/heartbeat",
                headers={"Authorization": "Bearer t"},
                json={"url": "http://model"},
            )
            assert beat.json()["driver_alive"] is True
