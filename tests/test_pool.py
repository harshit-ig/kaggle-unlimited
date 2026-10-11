"""Rotation policy: when to prewarm, how to cut over, and how quota is booked."""

from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import timedelta
from pathlib import Path

import pytest
import uvicorn

from kaggle_rotate.accounts import Account
from kaggle_rotate.config import Config
from kaggle_rotate.notebook import LaunchSpec
from kaggle_rotate.pool import MIN_PROBE_FAILURES, STATUS_POLL_TICKS, Pool
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
    pool._probe_failures["acct0"] = MIN_PROBE_FAILURES
    await pool._retire_dead_sessions()

    assert pool.state.get("acct0").state == "stopped"
    assert remote.command == "shutdown"


async def test_one_failed_probe_never_retires_a_serving_session(tmp_path: Path):
    """A single request can trip the 20s health timeout on the way through a Cloudflare
    edge. That used to arm the retirement timer in the same tick, so a healthy session
    was torn down and re-launched on nothing more than one bad round trip."""
    pool = _pool(tmp_path, _accounts(2), StubCLI())
    pool.config.rotation.heartbeat_grace_seconds = 0.0
    pool.state.update(
        "acct0", state="active", token="tok", url="http://a", model="m", started_at=time.time()
    )
    pool.router.set_active(Upstream(name="acct0", url="http://a", token="tok", model="m"))
    pool.router.active.healthy = False
    remote = RemoteSession(token="tok", account="acct0", kernel_ref="acct0/krotate", url="http://a")
    remote.last_seen = time.time() - 10_000
    pool.relay_state._sessions["tok"] = remote

    pool._probe_failures["acct0"] = MIN_PROBE_FAILURES - 1
    await pool._retire_dead_sessions()

    assert pool.state.get("acct0").state == "active"


async def test_a_dead_generation_cannot_veto_a_healthy_session(tmp_path: Path):
    """Liveness is keyed by the slot's launch token. Keyed by account, the frozen
    timestamp of an earlier generation was attributed to the session now serving -- the
    "has not responded for 1170s" retirement of a working kernel."""
    pool = _pool(tmp_path, _accounts(2), StubCLI())
    pool.config.rotation.heartbeat_grace_seconds = 30.0
    pool.state.update(
        "acct0", state="active", token="live-token", url="http://a", model="m",
        started_at=time.time(),
    )
    pool.router.set_active(Upstream(name="acct0", url="http://a", token="live-token", model="m"))
    pool.router.active.healthy = False

    # An old generation of the same account, silent for a very long time.
    dead = RemoteSession(
        token="dead-token", account="acct0", kernel_ref="acct0/krotate", url="http://old",
        model="m",
    )
    dead.last_seen = time.time() - 4000
    pool.relay_state._sessions["dead-token"] = dead
    # The generation actually serving, heartbeating normally.
    live = RemoteSession(
        token="live-token", account="acct0", kernel_ref="acct0/krotate", url="http://a", model="m"
    )
    pool.relay_state._sessions["live-token"] = live

    pool._probe_failures["acct0"] = MIN_PROBE_FAILURES
    await pool._retire_dead_sessions()

    assert pool.state.get("acct0").state == "active", "a stale generation must not retire it"


async def test_a_session_with_no_relay_record_is_left_running(tmp_path: Path):
    """Silence needs a clock to measure against. Falling back to the session's start
    made the log report the session's whole age as its silence -- which is how a
    session up for 1170s was described as "not responded for 1170s"."""
    pool = _pool(tmp_path, _accounts(2), StubCLI())
    pool.config.rotation.heartbeat_grace_seconds = 30.0
    pool.state.update(
        "acct0", state="active", token="tok", url="http://a", started_at=time.time() - 4000
    )
    pool.router.set_active(Upstream(name="acct0", url="http://a", token="tok"))
    pool.router.active.healthy = False
    pool._probe_failures["acct0"] = MIN_PROBE_FAILURES

    await pool._retire_dead_sessions()

    assert pool.state.get("acct0").state == "active"


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


async def test_a_kaggle_quota_reading_is_persisted_for_status(tmp_path: Path):
    """`status` reads k-weekly out of pool.json and has no other route to Kaggle, so a
    reading that is never written there is invisible: it printed "no Kaggle reading yet"
    while `accounts` showed a live figure for the same account."""
    accounts = _accounts(1)
    cli = StubCLI()
    cli.quota = 17.5
    pool = _pool(tmp_path, accounts, cli)

    snapshot = await pool._kaggle_quota(accounts["acct0"])

    assert snapshot is not None
    slot = pool.state.get("acct0")
    assert slot.quota_source == "kaggle"
    assert slot.kaggle_weekly_remaining_hours == pytest.approx(17.5)
    assert slot.kaggle_weekly_used_hours == pytest.approx(12.5)

    # And it must survive a fresh process reading the file, not just live in memory.
    reloaded = PoolState(tmp_path / "pool.json").get("acct0")
    assert reloaded.kaggle_weekly_remaining_hours == pytest.approx(17.5)
    assert reloaded.quota_source == "kaggle"


async def test_an_unreadable_quota_leaves_the_source_on_the_ledger(tmp_path: Path):
    accounts = _accounts(1)
    cli = StubCLI()
    cli.quota = None
    pool = _pool(tmp_path, accounts, cli)

    assert await pool._kaggle_quota(accounts["acct0"]) is None
    assert pool.state.get("acct0").quota_source == "ledger"
    assert pool.state.get("acct0").kaggle_weekly_remaining_hours is None


async def test_an_unchanged_reading_does_not_rewrite_the_state_file(tmp_path: Path):
    """The cache is consulted every tick; persisting on every hit would rewrite
    pool.json every 10 seconds for a figure that moves once every 5 minutes."""
    accounts = _accounts(1)
    cli = StubCLI()
    cli.quota = 17.5
    pool = _pool(tmp_path, accounts, cli)

    await pool._kaggle_quota(accounts["acct0"])
    state_file = tmp_path / "pool.json"
    before = state_file.stat().st_mtime_ns

    await pool._kaggle_quota(accounts["acct0"])

    assert state_file.stat().st_mtime_ns == before, "a cache hit must not rewrite the file"


async def test_a_booting_session_counts_as_coverage(tmp_path: Path):
    """Regression: with no `active` slot the pool decided it had no coverage, tried to
    launch, and `_pick_launch_target` rejected the account it had just launched. Every
    start therefore logged "no session can start right now: X: already has a session"
    against its own healthy boot."""
    pool = _pool(tmp_path, _accounts(1))
    pool.state.update("acct0", state="booting", token="t")
    pool._coverage_note = "acct0: already has a session"

    await pool._ensure_coverage()

    assert pool._coverage_note == "acct0: already has a session", (
        "a boot in progress must not overwrite the coverage note"
    )


async def test_ensure_coverage_does_not_launch_over_a_boot_in_progress(tmp_path: Path):
    pool = _pool(tmp_path, _accounts(2))
    pool.state.update("acct0", state="booting", token="t")
    launched = []
    pool._launch_and_wait = lambda account: launched.append(account) or asyncio.sleep(0)

    await pool._ensure_coverage()

    assert launched == [], "must not start a second session while one is booting"


async def test_a_launch_in_flight_blocks_a_second_launch(tmp_path: Path):
    """`_boot_tasks` is the real in-flight registry. `create_task` does not run the
    coroutine, so for one tick the slot still reads "idle" and the status field alone
    would let a second push through for the same account."""
    pool = _pool(tmp_path, _accounts(1))
    pool.state.update("acct0", state="idle")

    async def _never_finishes():
        await asyncio.sleep(3600)

    task = asyncio.create_task(_never_finishes())
    pool._boot_tasks["acct0"] = task
    try:
        assert await pool._pick_launch_target(force_if_idle=True) is None
        assert "in flight" in pool._coverage_note
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_retire_cancels_the_boot_it_supersedes(tmp_path: Path):
    """A stopped session must stop being polled. The monitor used to keep announcing
    "still booting (N min)" for the rest of its own timeout, after the kernel was gone."""
    accounts = _accounts(1)
    cli = StubCLI()
    pool = _pool(tmp_path, accounts, cli)
    pool.state.update("acct0", state="booting", token="t")
    cancelled = asyncio.Event()

    async def _watcher():
        try:
            await asyncio.sleep(3600)
        finally:
            cancelled.set()

    task = asyncio.create_task(_watcher())
    await asyncio.sleep(0)
    pool._boot_tasks["acct0"] = task

    await pool._retire("acct0", reason="kernel error", force=True, grace=0.0)

    assert cancelled.is_set(), "the boot monitor must not outlive the session"
    assert task.done()
    assert "acct0" not in pool._boot_tasks
    assert pool.state.get("acct0").state == "stopped"


async def test_a_retire_from_inside_the_boot_never_cancels_its_own_caller(tmp_path: Path):
    """`_launch_and_wait` calls `_retire` when a boot fails. Cancelling the caller there
    would abort the retire between marking the slot and deleting the kernel -- the one
    moment the delete must not be skipped."""
    accounts = _accounts(1)
    cli = StubCLI()
    pool = _pool(tmp_path, accounts, cli)
    pool.state.update("acct0", state="booting", token="t")

    async def _failing_boot():
        pool._boot_tasks["acct0"] = asyncio.current_task()
        await pool._retire("acct0", reason="boot failed", force=True, grace=0.0)
        return "survived"

    assert await asyncio.create_task(_failing_boot()) == "survived"
    assert cli.deleted == ["acct0"], "the kernel must still have been deleted"


async def test_a_cancelled_boot_books_its_attempt_so_backoff_engages(tmp_path: Path):
    """A boot cancelled from outside never reaches its own error handler. Without an
    attempt booked here the account stayed immediately re-admissible and the next tick
    pushed another kernel -- the ~20 minute relaunch cycle in the logs."""
    accounts = _accounts(1)
    pool = _pool(tmp_path, accounts, StubCLI())
    pool.state.update("acct0", state="booting", token="t", attempts=0, last_attempt=0.0)

    async def _watcher():
        await asyncio.sleep(3600)

    task = asyncio.create_task(_watcher())
    await asyncio.sleep(0)
    pool._boot_tasks["acct0"] = task

    await pool._retire("acct0", reason="kernel error", force=True, grace=0.0)

    slot = pool.state.get("acct0")
    assert slot.attempts == 1
    assert slot.state == "stopped", "the account is now free..."
    assert await pool._pick_launch_target(force_if_idle=True) is None
    assert "backing off" in pool._coverage_note, "...but must not be immediately re-launched"


async def test_a_superseded_boot_monitor_gives_up(tmp_path: Path):
    """Last line of defence behind `_cancel_boot`: the loop is time-bounded only, so a
    monitor that survives must notice it has been replaced."""
    pool = _pool(tmp_path, _accounts(1))
    pool.state.update("acct0", state="booting", token="new-token")
    spec = LaunchSpec(
        account="acct0",
        kernel_ref="acct0/krotate",
        relay_url="http://relay",
        token="old-token",
        model="m",
        max_runtime_seconds=3600.0,
        shutdown_poll_seconds=20.0,
    )
    await pool._await_ready("acct0", spec)
    assert pool.state.get("acct0").state == "booting", "must not have touched the slot"


async def test_stopped_relay_generations_are_pruned(tmp_path: Path):
    """Nothing used to remove them, so one entry per launch generation accumulated for
    the life of the process, and any lookup by account had several generations to choose
    between -- one of them carrying a timestamp frozen at its own long-dead exit."""
    state = RelayState()
    for i in range(5):
        await state.register(
            f"tok{i}",
            {
                "account": "acct0",
                "kernel_ref": "acct0/krotate",
                "url": f"http://{i}",
                "model": "m",
            },
        )
        if i < 4:
            await state.final(f"tok{i}", {"stop_reason": "retired"})

    remaining = {s.token for s in state.sessions()}
    assert "tok4" in remaining, "the live generation must never be pruned"
    stopped = {s.token for s in state.sessions() if s.phase == "stopped"}
    assert stopped == {"tok3"}, f"at most the newest stopped generation is kept: {stopped}"


async def test_the_newest_stopped_generation_is_kept(tmp_path: Path):
    """Pruning must not empty the registry: a kernel that reported its own exit is still
    the most recent thing that happened and should stay visible."""
    state = RelayState()
    await state.register("tok0", {"account": "a", "url": "http://0", "model": "m"})
    await state.final("tok0", {"stop_reason": "done"})
    assert [s.token for s in state.sessions()] == ["tok0"]


async def test_an_unknown_heartbeat_is_reported_once(tmp_path: Path):
    """A kernel whose token the relay no longer holds heartbeats into a 404 forever.
    Silent, that looks exactly like a dead kernel from the outside."""
    state = RelayState()
    assert state.note_unknown_heartbeat("tok") is True
    assert state.note_unknown_heartbeat("tok") is False, "must not flood every 20s poll"

    await state.register("tok", {"account": "a", "url": "http://0", "model": "m"})
    assert state.note_unknown_heartbeat("tok") is True, "re-registering clears the warning"


async def test_the_boot_budget_includes_the_push(tmp_path: Path):
    """`kaggle kernels push` returns only once Kaggle has scheduled the kernel, which
    can take minutes. A deadline started after it came back was silently longer than
    `boot_timeout_seconds` and could outlast the rotation it was meant to protect."""
    pool = _pool(tmp_path, _accounts(1))
    pool.config.rotation.boot_timeout_seconds = 120.0
    pool.config.rotation.tick_seconds = 0.01
    slot = pool.state.get("acct0")
    slot.token = "tok"
    # The push already consumed the entire budget before this function was reached.
    slot.boot_started_at = time.time() - 500.0

    spec = LaunchSpec(
        account="acct0",
        kernel_ref="acct0/krotate",
        relay_url="http://relay",
        token="tok",
        model="m",
        max_runtime_seconds=3600.0,
        shutdown_poll_seconds=20.0,
    )
    with pytest.raises(TimeoutError) as caught:
        await pool._await_ready("acct0", spec)

    message = str(caught.value)
    assert "500s of it spent pushing" in message, f"must say where the time went: {message}"


async def test_a_boot_timeout_reports_measured_time_not_just_the_setting(tmp_path: Path):
    pool = _pool(tmp_path, _accounts(1))
    pool.config.rotation.boot_timeout_seconds = 0.0
    slot = pool.state.get("acct0")
    slot.token = "tok"
    slot.boot_started_at = time.time()

    spec = LaunchSpec(
        account="acct0",
        kernel_ref="acct0/krotate",
        relay_url="http://relay",
        token="tok",
        model="m",
        max_runtime_seconds=3600.0,
        shutdown_poll_seconds=20.0,
    )
    with pytest.raises(TimeoutError) as caught:
        await pool._await_ready("acct0", spec)

    assert "0s elapsed" in str(caught.value)


async def test_the_pool_servers_do_not_steal_the_process_signals(tmp_path: Path):
    """uvicorn installs its handlers in `serve()` through `capture_signals()`, not
    through `install_signal_handlers()`, which this subclass used to override. The
    override was dead code, so starting the servers replaced the CLI's own SIGINT
    handler and two shutdown drivers raced."""
    import signal

    from kaggle_rotate.pool import _Server

    before = signal.getsignal(signal.SIGINT)

    server = _Server(uvicorn.Config(app=None, host="127.0.0.1", port=0, log_level="warning"))
    with server.capture_signals():
        assert signal.getsignal(signal.SIGINT) is before, "the pool owns the signals"

    assert signal.getsignal(signal.SIGINT) is before


async def test_shutdown_never_raises_and_always_awaits_every_server(
    tmp_path: Path, monkeypatch
):
    """`wait_for` raises TimeoutError, which was not suppressed, so it escaped
    `shutdown()`: the pidfile was never cleared and the remaining servers were left
    unawaited for `asyncio.run` to destroy."""
    from kaggle_rotate import pool as pool_module

    monkeypatch.setattr(pool_module, "SERVER_GRACEFUL_SHUTDOWN_SECONDS", 0.05)
    accounts = _accounts(1)
    pool = _pool(tmp_path, accounts, StubCLI())
    calls: list[int] = []

    class _Stubborn:
        """A server that will not exit within the budget, like one with a wedged
        connection parked in the proxy's upstream wait."""

        def __init__(self, index: int) -> None:
            self.server = type("S", (), {"should_exit": False})()
            never = asyncio.create_task(asyncio.sleep(3600))

            async def _watch() -> None:
                calls.append(index)

            self.task = asyncio.gather(_watch(), never)

    pool.servers = [_Stubborn(0), _Stubborn(1)]  # type: ignore[assignment]

    await pool.shutdown()

    assert calls == [0, 1], "every server must be awaited even if the first one times out"
    assert pool.proxy._draining.is_set(), "the proxy must be told to stop waiting upstream"
    for server in pool.servers:  # type: ignore[attr-defined]
        server.task.cancel()


async def test_an_unknown_heartbeat_is_logged_once_not_every_poll(caplog):
    """The route logged on every poll after the first and skipped the first, which is the
    exact inverse of what it was written to do: the diagnostically useful first
    occurrence never appeared, and a 20s heartbeat poll produced a warning forever."""
    import logging

    from starlette.testclient import TestClient

    from kaggle_rotate.relay import build_relay_app

    state = RelayState()
    app = build_relay_app(state)
    TestClient(app)  # builds the lifespan the route runs inside

    with caplog.at_level(logging.WARNING, logger="kaggle_rotate.relay"):
        for _ in range(3):
            client = TestClient(app)
            response = client.post(
                "/_rot/heartbeat",
                headers={"Authorization": "Bearer ghost-token"},
                json={},
            )
            assert response.status_code == 404
            client.close()

    warnings = [r for r in caplog.records if "unknown token" in r.getMessage()]
    assert len(warnings) == 1, f"expected exactly one warning, got {len(warnings)}"


async def test_an_inconclusive_status_never_retires_a_serving_session(tmp_path: Path):
    """`kaggle kernels status` can hang or time out, and that comes back UNKNOWN -- which
    is neither alive nor terminal. Without a `known` check the ambiguity retired a
    healthy, serving kernel, and the failure was invisible because status errors are
    logged at debug."""
    from kaggle_rotate.accounts import StatusResult

    class _BlindCLI(StubCLI):
        async def astatus(self, account: Account, timeout=None):
            return StatusResult(raw="status probe timed out after 60s", state="UNKNOWN")

    accounts = _accounts(2)
    cli = _BlindCLI()
    pool = _pool(tmp_path, accounts, cli)
    pool.state.update(
        "acct0", state="active", token="tok", url="http://a", model="m", started_at=time.time()
    )

    await pool._refresh_status()

    assert pool.state.get("acct0").state == "active", "an unknown status is not a death"
    assert cli.deleted == [], "the kernel must not be deleted on a failed probe"


async def test_a_definitively_stopped_kernel_is_still_retired(tmp_path: Path):
    """The fix must not blunt the real signal: a COMPLETE kernel is known-and-terminal."""
    from kaggle_rotate.accounts import StatusResult

    class _CompletedCLI(StubCLI):
        async def astatus(self, account: Account, timeout=None):
            return StatusResult(raw="COMPLETE", state="COMPLETE")

    accounts = _accounts(2)
    cli = _CompletedCLI()
    pool = _pool(tmp_path, accounts, cli)
    pool.state.update(
        "acct0", state="active", token="tok", url="http://a", model="m", started_at=time.time()
    )

    await pool._refresh_status()

    assert pool.state.get("acct0").state == "stopped"
    assert cli.deleted == ["acct0"]
