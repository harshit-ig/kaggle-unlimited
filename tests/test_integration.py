"""End-to-end rotation against a stubbed `kaggle` CLI and stand-in kernels.

Nothing here touches Kaggle or Cloudflare. The stub CLI accepts pushes, the stand-in
kernels read their own launch credentials out of the generated notebook sidecar (which
is how the real thing learns its relay URL and token), and the assertions check that
client traffic actually moves from one account to the other.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import stat
import time
from contextlib import AsyncExitStack
from pathlib import Path

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from kaggle_rotate.accounts import Account, AccountStore, KaggleCLI
from kaggle_rotate.config import Config
from kaggle_rotate.pool import STATUS_POLL_TICKS, Pool
from kaggle_rotate.proxy import Upstream, UpstreamRouter
from kaggle_rotate.quota import Ledger
from kaggle_rotate.relay import RelayState, build_relay_app
from kaggle_rotate.state import PoolState
from tests.helpers import free_port, serve

STUB_KAGGLE = """#!/usr/bin/env python3
import os
import sys

args = sys.argv[1:]
if args[:2] == ["kernels", "list"]:
    print("no kernels")
elif args[:2] == ["kernels", "push"]:
    print("Kernel version 1 successfully pushed to the notebook viewer")
elif args[:2] == ["kernels", "status"]:
    print("Status: RUNNING")
elif args[:2] == ["kernels", "delete"]:
    print("deleted")
else:
    print("unsupported stub call: " + " ".join(args))
"""


@pytest.fixture
def stub_cli(tmp_path: Path) -> str:
    path = tmp_path / "stub-kaggle"
    path.write_text(STUB_KAGGLE)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _llama_server_app(model: str, served: list[str], delay: float = 0.0):
    """The OpenAI-compatible surface llama-server exposes.

    `/health` is the readiness gate: llama-server only starts answering it once the
    model is resident, so a 200 is the whole residency signal the pool needs.
    """

    async def health(_):
        return JSONResponse({"status": "ok"})

    async def models(_):
        return JSONResponse({"object": "list", "data": [{"id": model, "object": "model"}]})

    async def chat(_):
        served.append(model)

        async def body():
            if delay:
                await asyncio.sleep(delay)
            yield json.dumps({"model": model, "choices": [{"delta": {"content": "hi"}}]}) + "\n"

        return StreamingResponse(body(), media_type="application/x-ndjson")

    return Starlette(
        routes=[
            Route("/health", health),
            Route("/v1/models", models),
            Route("/v1/chat/completions", chat, methods=["POST"]),
        ]
    )


class FakeKernel:
    """Behaves like the tail of the generated notebook: register, serve, retire."""

    def __init__(self, pool: Pool, account: str, stack: AsyncExitStack, served: list[str]) -> None:
        self.pool = pool
        self.account = account
        self.stack = stack
        self.served = served
        self.spec = json.loads((self.launch_dir / "launch.json").read_text())
        self.heartbeat: asyncio.Task | None = None

    @property
    def launch_dir(self) -> Path:
        return self.pool.config.resolved_state_dir() / "launches" / self.account

    async def start(self) -> str:
        url = await self.stack.enter_async_context(
            serve(_llama_server_app(self.spec["model"], self.served), free_port())
        )
        await self.pool.relay_state.register(
            self.spec["token"],
            {
                "account": self.account,
                "kernel_ref": self.spec["kernel_ref"],
                "url": url,
                "model": self.spec["model"],
            },
        )
        self.heartbeat = asyncio.create_task(self._beat(url))
        return url

    async def _beat(self, url: str) -> None:
        while True:
            session = await self.pool.relay_state.heartbeat(
                self.spec["token"], {"url": url, "ollama_alive": True}
            )
            if session is not None and session.command == "shutdown":
                self.retired = True
                return
            await asyncio.sleep(0.05)

    retired = False


async def _wait_for(predicate, timeout: float = 30.0, interval: float = 0.05) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


async def test_pool_rotates_across_accounts_and_moves_traffic(tmp_path: Path, stub_cli: str):
    served: list[str] = []
    relay_state = RelayState()
    relay_port = free_port()

    config = Config()
    config.state_dir = tmp_path / "state"
    config.kaggle.command = stub_cli
    # Isolate credentials: never touch the user's real ~/.config/kaggle-rotate.
    config.kaggle.config_root = str(tmp_path / "creds")
    config.relay.expose = False
    config.relay.public_url = f"http://127.0.0.1:{relay_port}"
    config.rotation.tick_seconds = 0.05
    config.rotation.drain_grace_seconds = 0.0
    config.rotation.retire_timeout_seconds = 0.5
    config.proxy.drain_grace_seconds = 0.0
    config.rotation.boot_timeout_seconds = 30.0
    config.rotation.prewarm_lead_minutes = 60.0

    accounts = {
        "acct0": Account(slug="acct0", username="user0", kernel_slug="krotate"),
        "acct1": Account(slug="acct1", username="user1", kernel_slug="krotate"),
    }

    store = AccountStore(config)
    store.save(accounts)

    async with serve(build_relay_app(relay_state), relay_port):
        pool = Pool(
            config=config,
            accounts=accounts,
            cli=KaggleCLI(config, store),
            ledger=Ledger(config.resolved_state_dir() / "ledger.json"),
            relay_state=relay_state,
            router=UpstreamRouter(drain_grace=0.0),
            state=PoolState(config.resolved_state_dir() / "pool.json"),
        )
        pool.relay_public_url = config.relay.public_url

        async with AsyncExitStack() as stack:
            kernels: dict[str, FakeKernel] = {}
            watcher = asyncio.create_task(_watch_pushes(pool, accounts, kernels, stack, served))
            run = asyncio.create_task(pool.run_forever())
            try:
                assert await _wait_for(lambda: pool.router.active is not None), (
                    "no session ever became active"
                )
                first = pool.router.active.name
                assert pool.state.get(first).state == "active"
                assert pool.state.get(first).url

                first_spec = kernels[first].spec
                assert first_spec["relay_url"] == config.relay.public_url
                # The sidecar reports the model the notebook actually declares, so the
                # pool never advertises a name the kernel will not answer to.
                assert first_spec["model"] == "ternary-bonsai-2-27b-pq2"
                assert first_spec["token"]

                # Real HTTP through the proxy reaches the standing session.
                served.clear()
                header = await _chat(pool)
                assert header == first
                assert served == [first_spec["model"]]

                # Age the incumbent past its prewarm lead; a second account should boot.
                pool.state.update(first, started_at=time.time() - 11.5 * 3600)
                assert await _wait_for(lambda: len(kernels) == 2), "second account never launched"

                assert await _wait_for(
                    lambda: pool.router.active is not None and pool.router.active.name != first
                ), "traffic never moved to the second account"
                second = pool.router.active.name
                assert pool.state.get(second).state == "active"

                served.clear()
                assert await _chat(pool) == second
                assert served == [kernels[second].spec["model"]]

                # The old kernel was told to stand down and its quota slot closed.
                assert await _wait_for(lambda: pool.state.get(first).state == "stopped")
                assert kernels[first].retired is True
                assert pool.ledger.find_live(first) is None
                assert pool.ledger.used_hours(first) > 0
                assert pool.ledger.find_live(second) is not None
            finally:
                run.cancel()
                watcher.cancel()
                await pool.shutdown()


async def _watch_pushes(
    pool: Pool,
    accounts: dict[str, Account],
    kernels: dict[str, FakeKernel],
    stack: AsyncExitStack,
    served: list[str],
) -> None:
    """Stand in for Kaggle picking up our pushed notebook and starting it."""
    seen: set[str] = set()
    while True:
        for slug in accounts:
            launch_json = pool.config.resolved_state_dir() / "launches" / slug / "launch.json"
            if slug in seen or not launch_json.exists():
                continue
            seen.add(slug)
            kernels[slug] = FakeKernel(pool, slug, stack, served)
            await kernels[slug].start()
        await asyncio.sleep(0.02)


async def _chat(pool: Pool) -> str:
    transport = httpx.ASGITransport(app=pool.proxy.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        response = await client.post("/v1/chat/completions", json={"model": "m", "stream": True})
    assert response.status_code == 200, response.text
    assert '"content": "hi"' in response.text
    return response.headers["X-Kaggle-Rotate-Upstream"]


def test_stub_kaggle_is_executable(stub_cli: str):
    assert os.access(stub_cli, os.X_OK)


SLOW_KAGGLE = """#!/usr/bin/env python3
import os
import sys
import time

args = sys.argv[1:]
# The real Kaggle API can take tens of seconds to answer. A blocking call here used to
# freeze the proxy's event loop, so the API looked dead for the whole probe.
if args[:2] == ["kernels", "status"]:
    time.sleep(float(os.environ.get("STUB_STATUS_DELAY", "30")))
    print("Status: RUNNING")
elif args[:2] == ["kernels", "list"]:
    print("no kernels")
elif args[:2] == ["kernels", "push"]:
    print("pushed")
elif args[:2] == ["kernels", "delete"]:
    print("deleted")
else:
    print("unsupported stub call")
"""


@pytest.fixture
def slow_cli(tmp_path: Path) -> str:
    path = tmp_path / "slow-kaggle"
    path.write_text(SLOW_KAGGLE)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


async def test_a_slow_kaggle_probe_does_not_freeze_the_api(tmp_path: Path, slow_cli: str):
    """Regression: `kaggle kernels status` took 2 minutes and blocked the event loop,
    so the proxy stopped accepting connections and the API looked down."""
    served: list[str] = []
    relay_state = RelayState()
    relay_port = free_port()

    config = Config()
    config.state_dir = tmp_path / "state"
    config.kaggle.command = slow_cli
    config.kaggle.config_root = str(tmp_path / "creds")
    config.relay.expose = False
    config.relay.public_url = f"http://127.0.0.1:{relay_port}"
    config.rotation.tick_seconds = 0.05
    config.rotation.drain_grace_seconds = 0.0
    config.rotation.retire_timeout_seconds = 0.5
    config.proxy.drain_grace_seconds = 0.0
    config.rotation.boot_timeout_seconds = 30.0
    config.kaggle.cli_timeout_seconds = 120.0

    accounts = {"acct0": Account(slug="acct0", username="u0", kernel_slug="krotate")}
    store = AccountStore(config)
    store.save(accounts)

    async with serve(build_relay_app(relay_state), relay_port):
        pool = Pool(
            config=config,
            accounts=accounts,
            cli=KaggleCLI(config, store),
            ledger=Ledger(config.resolved_state_dir() / "ledger.json"),
            relay_state=relay_state,
            router=UpstreamRouter(drain_grace=0.0),
            state=PoolState(config.resolved_state_dir() / "pool.json"),
        )
        pool.relay_public_url = config.relay.public_url

        # An active session whose status probe hangs for 30s on every call.
        pool.state.update(
            "acct0", state="active", url="http://a", model="m", started_at=time.time()
        )
        pool.router.set_active(Upstream(name="acct0", url="http://a", model="m"))
        pool.router.active.healthy = True
        pool._last_traffic = time.time()
        # Stop the health probe from retiring the slot while we measure.
        pool.config.rotation.heartbeat_grace_seconds = 3600.0

        # The status probe only runs every STATUS_POLL_TICKS ticks; align so the tick
        # below actually hits it.
        pool._tick_count = STATUS_POLL_TICKS - 1
        started = time.monotonic()
        tick = asyncio.create_task(pool._tick())
        await asyncio.sleep(0.5)

        async with AsyncExitStack() as stack:
            url = await stack.enter_async_context(
                serve(_llama_server_app("m", served), free_port())
            )
            assert pool.router.active is not None
            pool.router.active.url = url
            pool.state.update("acct0", url=url)

            # The API must answer promptly *while* that probe is still running.
            transport = httpx.ASGITransport(app=pool.proxy.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
                t0 = time.monotonic()
                response = await client.post(
                    "/v1/chat/completions", json={"model": "m", "stream": True}
                )
                latency = time.monotonic() - t0

            assert response.status_code == 200, response.text
            assert latency < 5, f"proxy waited {latency:.1f}s behind the Kaggle probe"
            assert served == ["m"]

            # The whole point: the tick was still in a 30s probe, yet the API answered.
            assert not tick.done(), "the status probe should still be in flight"
            assert time.monotonic() - started < 10
            tick.cancel()
            with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(tick, timeout=5)
        await pool.shutdown()
