"""Rotation policy: when to prewarm, how to cut over, and how quota is booked."""

from __future__ import annotations

import asyncio
import time
from datetime import timedelta
from pathlib import Path

import pytest

from kaggle_rotate.accounts import Account
from kaggle_rotate.config import Config
from kaggle_rotate.notebook import LaunchSpec
from kaggle_rotate.pool import STATUS_POLL_TICKS, Pool
from kaggle_rotate.proxy import Upstream, UpstreamRouter
from kaggle_rotate.quota import Ledger, utcnow
from kaggle_rotate.relay import RelayState, RemoteSession
from kaggle_rotate.state import PoolState


class StubCLI:
    """Stands in for the kaggle CLI: no subprocesses, no network."""

    def __init__(self) -> None:
        self.pushed: list[str] = []
        self.deleted: list[str] = []
        self.delete_attempts: list[str] = []
        self.status_state = "COMPLETE"
        # None means "the quota API is unreachable", which is the fail-open path: the
        # pool must fall back to the local ledger rather than refusing to start.
        self.quota: float | None = None
        self.quota_calls = 0

    async def aquota_cached(self, account, cache, timeout=None):
        from kaggle_rotate.quota import QuotaSnapshot, QuotaUnavailable

        self.quota_calls += 1
        cached = cache.cached(account.slug)
        if cached is not None:
            return cached
        if self.quota is None:
            raise QuotaUnavailable("stub: quota unavailable")
        snapshot = QuotaSnapshot(
            account=account.slug,
            used_hours=30.0 - self.quota,
            remaining_hours=self.quota,
            total_hours=30.0,
        )
        cache.store(snapshot)
        return snapshot

    def status(self, account: Account):
        from kaggle_rotate.accounts import StatusResult

        return StatusResult(raw=self.status_state, state=self.status_state)

    async def adelete(self, account: Account, timeout=None) -> bool:
        self.deleted.append(account.slug)
        return True

    async def astatus(self, account: Account, timeout=None):
        from kaggle_rotate.accounts import StatusResult

        return StatusResult(raw=self.status_state, state=self.status_state)


def _accounts(n: int) -> dict[str, Account]:
    return {f"acct{i}": Account(slug=f"acct{i}", username=f"user{i}") for i in range(n)}


def _pool(tmp_path: Path, accounts: dict[str, Account], cli=None) -> Pool:
    config = Config()
    config.state_dir = tmp_path
    config.rotation.drain_grace_seconds = 0.0
    config.proxy.drain_grace_seconds = 0.0
    config.rotation.prewarm_lead_minutes = 60.0
    cli = cli or StubCLI()
    return Pool(
        config=config,
        accounts=accounts,
        cli=cli,
        ledger=Ledger(tmp_path / "ledger.json"),
        relay_state=RelayState(),
        router=UpstreamRouter(drain_grace=0.0),
        state=PoolState(tmp_path / "pool.json"),
    )


def test_fresh_session_does_not_prewarm(tmp_path: Path):
    pool = _pool(tmp_path, _accounts(2))
    pool.state.update("acct0", state="active", started_at=time.time(), url="http://a")
    assert pool._should_prewarm(pool.state.get("acct0")) is False


def test_session_near_its_cap_triggers_prewarm(tmp_path: Path):
    pool = _pool(tmp_path, _accounts(2))
    stale = time.time() - (11.5 * 3600)
    pool.state.update("acct0", state="active", started_at=stale, url="http://a")
    assert pool._should_prewarm(pool.state.get("acct0")) is True


def test_weekly_near_cap_triggers_prewarm(tmp_path: Path):
    pool = _pool(tmp_path, _accounts(2))
    spent = pool.ledger.open_session("acct0", "acct0/krotate", now=utcnow() - timedelta(hours=29))
    pool.ledger.close_session(spent, "done")
    pool.state.update("acct0", state="active", started_at=time.time(), url="http://a")
    # Only ~1h left this week, which is less than the boot reserve + safety margin.
    assert pool._should_prewarm(pool.state.get("acct0")) is True


def test_first_session_goes_to_the_account_with_most_quota(tmp_path: Path):
    accounts = _accounts(3)
    pool = _pool(tmp_path, accounts)
    spent = pool.ledger.open_session("acct2", "acct2/krotate", now=utcnow() - timedelta(hours=10))
    pool.ledger.close_session(spent, "done")

    chosen = asyncio.run(pool._pick_launch_target())
    assert chosen is not None
    assert chosen.slug in {"acct0", "acct1"}
    # acct2 has 20h left, the others 30h; tie broken by iteration order.
    assert chosen.slug == "acct0"


def test_exhausted_accounts_are_skipped(tmp_path: Path):
    pool = _pool(tmp_path, _accounts(2))
    for slug in ("acct0", "acct1"):
        spent = pool.ledger.open_session(
            slug, f"{slug}/krotate", now=utcnow() - timedelta(hours=31)
        )
        pool.ledger.close_session(spent, "done")
    assert asyncio.run(pool._pick_launch_target()) is None


async def test_concurrent_kernel_cap_is_respected(tmp_path: Path):
    pool = _pool(tmp_path, _accounts(3))
    pool.state.update("acct0", state="active", started_at=time.time())
    # max_active_sessions=1 allows exactly one extra warming kernel.
    assert pool._live_kernel_count() == 1
    pool.state.update("acct1", state="booting")
    assert pool._live_kernel_count() == 2
    assert await pool._pick_launch_target() is None


async def test_cutover_moves_traffic_to_the_warmed_account(tmp_path: Path):
    accounts = _accounts(2)
    pool = _pool(tmp_path, accounts, StubCLI())

    await pool.relay_state.register("old", {"account": "acct0", "url": "http://old", "model": "m"})
    await pool.relay_state.register("new", {"account": "acct1", "url": "http://new", "model": "m"})
    pool.state.update("acct0", state="ready", token="old", url="http://old", model="m")
    pool.state.update("acct1", state="ready", token="new", url="http://new", model="m")

    await pool._admit("acct0")
    assert pool.router.active is not None and pool.router.active.name == "acct0"

    # A second healthy session warms up but must NOT steal traffic yet.
    await pool._admit("acct1")
    assert pool.router.active.name == "acct0"
    assert pool.router.standby is not None and pool.router.standby.name == "acct1"

    # Once the incumbent nears its session cap, traffic flips and the old slot drains.
    pool.state.update("acct0", started_at=time.time() - 11.5 * 3600)
    await pool._admit("acct1")

    assert pool.router.active.name == "acct1"
    assert pool.router.standby is None
    assert pool.state.get("acct1").state == "active"

    await _drain(pool)
    assert pool.state.get("acct0").state == "stopped"


async def test_cutover_is_immediate_when_the_active_upstream_is_unhealthy(tmp_path: Path):
    pool = _pool(tmp_path, _accounts(2), StubCLI())
    await pool.relay_state.register("old", {"account": "acct0", "url": "http://old", "model": "m"})
    await pool.relay_state.register("new", {"account": "acct1", "url": "http://new", "model": "m"})
    pool.state.update("acct0", state="ready", token="old", url="http://old", model="m")
    pool.state.update("acct1", state="ready", token="new", url="http://new", model="m")

    await pool._admit("acct0")
    assert pool.router.active is not None
    pool.router.active.healthy = False

    await pool._admit("acct1")
    assert pool.router.active.name == "acct1"
    await _drain(pool)
    assert pool.state.get("acct0").state == "stopped"


async def _drain(pool: Pool) -> None:
    """Wait for background retire tasks spawned by a cutover."""
    for _ in range(50):
        if not pool._retire_tasks:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("retire tasks never finished")


async def test_retire_closes_the_ledger_and_tells_the_kernel_to_stop(tmp_path: Path):
    accounts = _accounts(1)
    cli = StubCLI()
    pool = _pool(tmp_path, accounts, cli)
    await pool.relay_state.register("tok", {"account": "acct0", "url": "http://old", "model": "m"})
    pool.state.update(
        "acct0",
        state="active",
        token="tok",
        kernel_ref="acct0/krotate",
        url="http://old",
        model="m",
        started_at=time.time(),
    )
    pool.router.set_active(Upstream(name="acct0", url="http://old", token="tok"))
    pool.ledger.open_session("acct0", "acct0/krotate", now=utcnow() - timedelta(hours=3))

    await pool._retire("acct0", reason="rotated to acct1", force=True)

    remote = await pool.relay_state.get("tok")
    assert remote is not None and remote.command == "shutdown"
    assert remote.command_reason == "rotated to acct1"
    assert pool.router.active is None
    assert pool.state.get("acct0").state == "stopped"
    assert pool.ledger.find_live("acct0") is None
    assert pool.ledger.used_hours("acct0") > 2.9


async def test_orphans_from_a_previous_run_are_retired_on_start(tmp_path: Path):
    pool = _pool(tmp_path, _accounts(1))
    pool.state.update("acct0", state="active", token="tok", kernel_ref="acct0/krotate", url="u")
    await pool.relay_state.register("tok", {"account": "acct0", "url": "http://old"})
    pool.ledger.open_session("acct0", "acct0/krotate")

    await pool._recover_orphans()

    assert pool.state.get("acct0").state == "stopped"
    assert pool.ledger.find_live("acct0") is None


def test_status_reports_budgets(tmp_path: Path):
    pool = _pool(tmp_path, _accounts(1))
    pool.ledger.open_session("acct0", "acct0/krotate", now=utcnow() - timedelta(hours=2))
    status = pool.status()
    slot = next(s for s in status["slots"] if s["account"] == "acct0")
    assert 1.9 < slot["session_remaining_hours"] < 10.1
    # The session is live, so the remaining budget ticks down between reads.
    assert slot["weekly_remaining_hours"] == pytest.approx(28.0, abs=0.01)
    assert status["proxy"]["base_url"].endswith("/v1")


async def test_coverage_gap_is_reported_once_per_distinct_reason(tmp_path: Path):
    pool = _pool(tmp_path, _accounts(1))
    pool.state.update("acct0", state="failed", attempts=1, last_attempt=time.time())

    assert await pool._pick_launch_target() is None
    first = pool._coverage_note
    assert "backing off" in first

    for _ in range(5):
        await pool._pick_launch_target()
    assert pool._coverage_note == first


def test_coverage_note_explains_quota_exhaustion(tmp_path: Path):
    pool = _pool(tmp_path, _accounts(2))
    for slug in ("acct0", "acct1"):
        spent = pool.ledger.open_session(slug, f"{slug}/k", now=utcnow() - timedelta(hours=31))
        pool.ledger.close_session(spent, "done")
    assert asyncio.run(pool._pick_launch_target()) is None
    assert pool._coverage_note.count("weekly quota exhausted") == 2


async def test_silent_session_is_retired_so_a_replacement_can_start(tmp_path: Path):
    pool = _pool(tmp_path, _accounts(2), StubCLI())
    pool.config.rotation.heartbeat_grace_seconds = 30.0
    pool.state.update(
        "acct0", state="active", token="tok", url="http://a", model="m", started_at=time.time()
    )
    pool.router.set_active(Upstream(name="acct0", url="http://a", token="tok", model="m"))
    pool.router.active.healthy = True

    # A relay session that stopped heartbeating long ago.
    remote = RemoteSession(
        token="tok", account="acct0", kernel_ref="acct0/krotate", url="http://a", model="m"
    )
    remote.last_seen = time.time() - 600
    pool.relay_state._sessions["tok"] = remote

    pool.router.active.healthy = False
    await pool._retire_dead_sessions()

    assert pool.state.get("acct0").state == "stopped"
    assert remote.command == "shutdown"


async def test_recently_silent_session_is_left_alone(tmp_path: Path):
    pool = _pool(tmp_path, _accounts(1), StubCLI())
    pool.config.rotation.heartbeat_grace_seconds = 3600.0
    pool.state.update("acct0", state="active", token="tok", url="http://a")
    pool.router.set_active(Upstream(name="acct0", url="http://a", token="tok"))
    pool.router.active.healthy = False

    remote = RemoteSession(token="tok", account="acct0", kernel_ref="acct0/krotate", url="http://a")
    pool.relay_state._sessions["tok"] = remote

    await pool._retire_dead_sessions()
    assert pool.state.get("acct0").state == "active"


async def test_traffic_through_the_proxy_resets_the_idle_timer(tmp_path: Path):
    import httpx

    pool = _pool(tmp_path, _accounts(1))
    pool.proxy.connect_grace = 0.2
    pool.config.rotation.idle_stop_minutes = 30
    pool._last_traffic -= 3600  # pretend nothing has been served for an hour

    pool.router.set_active(Upstream(name="acct0", url="http://127.0.0.1:1"))
    transport = httpx.ASGITransport(app=pool.proxy.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        # The upstream is dead, but a client still reached for it.
        await client.get("/v1/models")

    assert time.time() - pool._last_traffic < 5
    pool._maybe_idle_stop()  # must not schedule a retirement
    assert not pool._retire_tasks


async def test_idle_pool_retires_its_session(tmp_path: Path):
    pool = _pool(tmp_path, _accounts(1), StubCLI())
    pool.config.rotation.idle_stop_minutes = 0.001  # 60ms
    pool._last_traffic = time.time() - 3600
    pool.state.update("acct0", state="active", token="tok", url="http://a")
    await pool.relay_state.register("tok", {"account": "acct0", "url": "http://a"})

    pool._maybe_idle_stop()
    await _drain(pool)
    assert pool.state.get("acct0").state == "stopped"


class UnknownCLI(StubCLI):
    """Probes that time out, i.e. we learn nothing about the kernel."""

    def __init__(self, state: str = "UNKNOWN") -> None:
        super().__init__()
        self.state = state
        self.deleted: list[str] = []

    async def astatus(self, account, timeout=None):
        from kaggle_rotate.accounts import StatusResult

        return StatusResult(raw="status probe timed out after 60s", state=self.state)

    async def adelete(self, account, timeout=None) -> bool:
        self.deleted.append(account.slug)
        return True


def test_inconclusive_status_is_not_treated_as_stopped():
    """Regression: a timed-out probe used to read as 'stopped', so the pool forgot a
    kernel that was still burning GPU quota."""
    from kaggle_rotate.accounts import StatusResult

    unknown = StatusResult(raw="status probe timed out after 60s", state="UNKNOWN")
    assert unknown.known is False
    assert unknown.terminal is False
    # `not alive` is true, which is exactly the trap the old code fell into.
    assert not unknown.alive


async def test_retire_deletes_the_kernel(tmp_path: Path):
    """Deleting is the stop. Nothing about it is conditional on Kaggle answering."""
    accounts = _accounts(1)
    cli = UnknownCLI()
    pool = _pool(tmp_path, accounts, cli)

    await pool._retire("acct0", reason="test", force=True)

    assert cli.deleted == ["acct0"]
    assert pool.state.get("acct0").state == "stopped"


async def test_shutdown_deletes_without_waiting(tmp_path: Path):
    accounts = _accounts(1)
    cli = UnknownCLI()
    pool = _pool(tmp_path, accounts, cli)
    pool.config.rotation.drain_grace_seconds = 999.0  # must be bypassed on shutdown
    pool.state.update("acct0", state="active", token="t", url="u")

    await pool._retire("acct0", reason="pool shutdown", force=True, grace=0.0)

    assert cli.deleted == ["acct0"]
    assert pool.state.get("acct0").state == "stopped"


async def test_rotation_grace_keeps_the_old_session_up_briefly(tmp_path: Path):
    accounts = _accounts(2)
    cli = UnknownCLI()
    pool = _pool(tmp_path, accounts, cli)
    pool.config.rotation.drain_grace_seconds = 0.4
    pool.state.update("acct0", state="active", token="t", url="u")

    await pool._retire("acct0", reason="superseded", force=True)

    assert cli.deleted == ["acct0"]


async def test_dead_relay_tunnel_is_reported_once_and_loudly(tmp_path: Path):
    pool = _pool(tmp_path, _accounts(1))
    assert pool._relay_dead_logged is False

    dead = type("P", (), {"returncode": 1, "stop": None})()
    pool.relay_tunnel = type("T", (), {"process": dead, "url": "https://x"})()
    pool._check_relay_tunnel()
    assert pool._relay_dead_logged is True
    pool._check_relay_tunnel()  # must not spam every tick


async def test_live_relay_tunnel_clears_the_dead_flag(tmp_path: Path):
    pool = _pool(tmp_path, _accounts(1))
    pool._relay_dead_logged = True
    alive = type("P", (), {"returncode": None})()
    pool.relay_tunnel = type("T", (), {"process": alive})()
    pool._check_relay_tunnel()
    assert pool._relay_dead_logged is False


def test_orphan_grace_is_well_under_the_session_cap():
    from kaggle_rotate.config import Config

    config = Config()
    assert config.kernel.orphan_grace_seconds < config.rotation.session_limit_hours * 3600
    # It is the backstop for a lost pool, so it must be minutes, not hours.
    assert config.kernel.orphan_grace_seconds <= 600


def test_status_is_polled_often_while_booting_and_rarely_when_serving(tmp_path: Path):
    pool = _pool(tmp_path, _accounts(1))
    pool.state.update("acct0", state="active")
    served = []
    for tick in range(1, 3 * STATUS_POLL_TICKS):
        pool._tick_count = tick
        if pool._status_poll_due():
            served.append(tick)
    assert served == [STATUS_POLL_TICKS, 2 * STATUS_POLL_TICKS], (
        f"serving should poll every {STATUS_POLL_TICKS} ticks, got {served}"
    )

    pool.state.update("acct0", state="booting")
    due = []
    for tick in range(1, 13):
        pool._tick_count = tick
        if pool._status_poll_due():
            due.append(tick)
    assert due[:3] == [2, 4, 6], f"booting should poll often, got {due}"


async def test_boot_fails_fast_when_the_kernel_dies(tmp_path: Path):
    """A kernel that crashes during setup never registers; waiting out the boot timeout
    used to leave the user staring at a silent terminal for 90 minutes."""
    accounts = _accounts(1)
    cli = StubCLI()
    cli.status_state = "ERROR"
    pool = _pool(tmp_path, accounts, cli)
    spec = LaunchSpec(
        account="acct0",
        kernel_ref="acct0/krotate",
        relay_url="http://relay",
        token="tok",
        model="m",
        max_runtime_seconds=600,
        shutdown_poll_seconds=1,
    )
    pool.state.update("acct0", state="booting", token="tok", kernel_ref="acct0/krotate")

    with pytest.raises(RuntimeError, match="ERROR during boot"):
        await pool._await_ready("acct0", spec)


class FailingDeleteCLI(StubCLI):
    """A delete that Kaggle refuses, e.g. an expired session or a rate limit."""

    async def adelete(self, account, timeout=None) -> bool:
        self.delete_attempts.append(account.slug)
        return False


async def test_a_failed_delete_never_reports_stopped(tmp_path: Path):
    """Regression: adelete swallowed the exit code, so the pool logged
    'harshitig stopped' while the kernel kept billing GPU time."""
    accounts = _accounts(1)
    cli = FailingDeleteCLI()
    cli.delete_attempts = []
    pool = _pool(tmp_path, accounts, cli)
    pool.state.update("acct0", state="active", token="t", url="u", started_at=time.time())

    await pool._retire("acct0", reason="pool shutdown", force=True, grace=0.0)

    assert cli.delete_attempts == ["acct0"], "it must have tried"
    slot = pool.state.get("acct0")
    assert slot.state == "failed", "must not claim it stopped"
    assert "cleanup" in slot.detail, f"detail must point at the recovery path: {slot.detail}"


async def test_a_successful_delete_reports_stopped(tmp_path: Path):
    accounts = _accounts(1)
    cli = StubCLI()
    pool = _pool(tmp_path, accounts, cli)
    pool.state.update("acct0", state="active", token="t", url="u")

    await pool._retire("acct0", reason="pool shutdown", force=True, grace=0.0)

    assert cli.deleted == ["acct0"]
    assert pool.state.get("acct0").state == "stopped"
