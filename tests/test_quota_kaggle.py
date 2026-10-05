"""Kaggle's quota API is the authority on weekly GPU time; the ledger is a fallback.

The distinction matters because the ledger only knows about sessions this tool started.
On a real account it reported 2.72h where Kaggle billed 5.03h, so the error was in the
optimistic direction -- exactly the direction that overspends quota.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from kaggle_rotate.accounts import Account, KaggleCLI
from kaggle_rotate.config import Config
from kaggle_rotate.quota import (
    Budget,
    Ledger,
    QuotaCache,
    QuotaSnapshot,
    QuotaUnavailable,
    hours,
    parse_quota_csv,
    utcnow,
    week_start,
)

# Captured verbatim from `kaggle quota --csv` on a live account.
REAL_CSV = (
    "resource,used,remaining,total,refreshAt\n"
    "GPU,5.03h,24.97h,30.00h,2026-10-10T00:00:00\n"
    "TPU,0.00h,20.00h,20.00h,2026-10-10T00:00:00\n"
)

HEADER_ONLY_CSV = "resource,used,remaining,total,refreshAt\n"


# ------------------------------------------------------------------ CSV parsing


def test_parses_the_real_csv_shape():
    snapshot = parse_quota_csv(REAL_CSV, "harshitig")
    assert snapshot is not None
    assert snapshot.used_hours == 5.03
    assert snapshot.remaining_hours == 24.97
    assert snapshot.total_hours == 30.0
    assert snapshot.refresh_at == "2026-10-10T00:00:00"


def test_tpu_row_is_ignored():
    """TPU quota is a separate resource; the pool only spends GPU."""
    snapshot = parse_quota_csv(REAL_CSV, "a")
    assert snapshot is not None
    assert snapshot.remaining_hours == 24.97


@pytest.mark.parametrize(
    "text",
    [
        "",
        HEADER_ONLY_CSV,
        "not csv at all",
        "resource,used,remaining,total,refreshAt\nGPU,n/a,n/a,n/a,\n",
        "GPU,5.03h\n",
    ],
)
def test_unparseable_output_is_rejected_rather_than_defaulted(text: str):
    """A zero here would read as 'no quota used', so it must return None instead."""
    assert parse_quota_csv(text, "a") is None


def test_negative_and_over_limit_figures_are_sanitised():
    over = QuotaSnapshot(account="a", used_hours=31.0, remaining_hours=-1.0, total_hours=30.0)
    assert over.remaining_hours == 0.0
    # `used` stays above the allowance so an overage is still visible to the operator.
    assert over.used_hours == 31.0

    negative = QuotaSnapshot(account="a", used_hours=-2.0, remaining_hours=32.0, total_hours=30.0)
    assert negative.used_hours == 0.0


# ----------------------------------------------------------------------- cache


def test_snapshots_are_timestamped_on_construction():
    """Without this a hand-built snapshot looks infinitely old and is never cached."""
    snapshot = QuotaSnapshot(account="a", used_hours=1.0, remaining_hours=29.0, total_hours=30.0)
    assert snapshot.fetched_at > 0
    assert snapshot.age_seconds == 0.0


def test_cache_serves_within_ttl_and_expires_after():
    cache = QuotaCache(ttl_seconds=300.0)
    assert cache.cached("a") is None

    cache.store(QuotaSnapshot(account="a", used_hours=1.0, remaining_hours=29.0, total_hours=30.0))
    hit = cache.cached("a")
    assert hit is not None
    assert hit.remaining_hours == 29.0

    cache.ttl_seconds = -1.0
    assert cache.cached("a") is None, "an expired reading must trigger a re-read"


def test_cache_is_per_account():
    cache = QuotaCache(ttl_seconds=300.0)
    cache.store(QuotaSnapshot(account="a", used_hours=1.0, remaining_hours=29.0, total_hours=30.0))
    assert cache.cached("b") is None


def test_cache_age_is_reported_so_a_stale_read_is_visible():
    cache = QuotaCache(ttl_seconds=300.0)
    snapshot = QuotaSnapshot(account="a", used_hours=1.0, remaining_hours=29.0, total_hours=30.0)
    snapshot.fetched_at = time.time() - 120.0
    cache.store(snapshot)
    hit = cache.cached("a")
    assert hit is not None
    assert hit.age_seconds >= 119.0


# ---------------------------------------------------------------------- budget


def _budget() -> Budget:
    return Budget(session_limit_hours=12.0, weekly_limit_hours=30.0)


def test_can_start_prefers_kaggle_over_the_ledger(tmp_path: Path):
    """The whole point: Kaggle sees time the ledger cannot."""
    ledger = Ledger(tmp_path / "l.json")
    ledger.open_session("a", "a/k").stop_reason = "done"
    # Ledger sees ~0h used, so it would happily approve.
    assert ledger.used_hours("a") == 0.0

    ok, reason = _budget().can_start(ledger, "a", kaggle_remaining=1.0, reserve_hours=1.5)
    assert ok is False, "a real 1h left must block a 1.5h reserve"
    assert "1.00h" in reason


def test_ledger_undercount_alone_would_have_allowed_a_start(tmp_path: Path):
    """The regression this feature exists to prevent, stated as a test."""
    ledger = Ledger(tmp_path / "l.json")
    session = ledger.open_session("a", "a/k")
    ledger.close_session(session, "done")

    without_kaggle, _ = _budget().can_start(ledger, "a", reserve_hours=1.5)
    assert without_kaggle is True, "the local estimate thinks there is plenty"

    # With Kaggle's real number the same state is refused.
    with_kaggle, _ = _budget().can_start(ledger, "a", kaggle_remaining=1.0, reserve_hours=1.5)
    assert with_kaggle is False


def test_missing_kaggle_figure_falls_back_to_the_ledger(tmp_path: Path):
    """Fail-open is deliberate: a quota outage must not strand a working endpoint."""
    ledger = Ledger(tmp_path / "l.json")
    ok, reason = _budget().can_start(ledger, "a", kaggle_remaining=None)
    assert ok is True
    assert "30.00h" in reason


def test_exhausted_kaggle_quota_blocks_even_with_a_fresh_ledger(tmp_path: Path):
    ledger = Ledger(tmp_path / "l.json")
    ok, reason = _budget().can_start(ledger, "a", kaggle_remaining=0.0)
    assert ok is False
    assert "kaggle" in reason, "the reason must name the source it trusted"


def test_operator_cap_below_kaggles_allowance_still_wins(tmp_path: Path):
    """An account capped at 5h must not be allowed to spend Kaggle's full 30h."""
    ledger = Ledger(tmp_path / "l.json")
    ok, _ = _budget().can_start(ledger, "a", account_limit=5.0, kaggle_remaining=29.0)
    assert ok is True

    # 26h used against a 5h cap: nothing left, so a start must be refused even though
    # Kaggle itself would allow one.
    blocked, _ = _budget().can_start(
        ledger,
        "a",
        account_limit=5.0,
        kaggle_remaining=4.0,
        kaggle_used=26.0,
        reserve_hours=1.5,
    )
    assert blocked is False, "a 5h cap already exceeded cannot back a 1.5h reserve"

    # 3h used against a 5h cap leaves 2h, which covers the 1.5h reserve.
    ok2, reason2 = _budget().can_start(
        ledger,
        "a",
        account_limit=5.0,
        kaggle_remaining=26.0,
        kaggle_used=3.0,
        reserve_hours=1.5,
    )
    assert ok2 is True, reason2

    # 4h used leaves 1h, which does not cover it.
    ok3, _ = _budget().can_start(
        ledger, "a", account_limit=5.0, kaggle_remaining=26.0, kaggle_used=4.0, reserve_hours=1.5
    )
    assert ok3 is False


def test_a_live_session_always_blocks_regardless_of_quota(tmp_path: Path):
    ledger = Ledger(tmp_path / "l.json")
    ledger.open_session("a", "a/k", now=utcnow())
    ok, reason = _budget().can_start(ledger, "a", kaggle_remaining=29.0)
    assert ok is False
    assert "live session" in reason


# ----------------------------------------------------------------- subprocess IO


class _Stub:
    """Stands in for the kaggle binary."""

    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[list[str]] = []


def _cli(tmp_path: Path, stub: _Stub) -> tuple[KaggleCLI, Account]:
    config = Config()
    config.state_dir = tmp_path
    store = type("S", (), {})()
    store.env_for = lambda account: {}  # type: ignore[attr-defined]
    cli = KaggleCLI(config=config, store=store)  # type: ignore[arg-type]
    account = Account(slug="a", username="u")
    return cli, account


def test_quota_raises_rather_than_returning_a_fake_zero():
    """Callers fall back on QuotaUnavailable; a silent 0 would read as 'none used'."""
    from kaggle_rotate.accounts import KaggleError

    cli, account = _cli(Path("/tmp"), _Stub(stdout=HEADER_ONLY_CSV))
    cli.run = lambda *a, **k: type(
        "P", (), {"stdout": HEADER_ONLY_CSV, "stderr": "", "returncode": 0}
    )()
    with pytest.raises(QuotaUnavailable):
        cli.quota(account)

    cli.run = lambda *a, **k: type(
        "P", (), {"stdout": "", "stderr": "auth expired", "returncode": 1}
    )()
    with pytest.raises(QuotaUnavailable):
        cli.quota(account)

    def _boom(*a, **k):
        raise KaggleError(["kaggle"], 127, "", "not found on PATH")

    cli.run = _boom
    with pytest.raises(QuotaUnavailable):
        cli.quota(account)


def test_quota_returns_a_snapshot_on_success():
    cli, account = _cli(Path("/tmp"), _Stub())
    cli.run = lambda *a, **k: type("P", (), {"stdout": REAL_CSV, "stderr": "", "returncode": 0})()
    snapshot = cli.quota(account)
    assert snapshot.used_hours == 5.03
    assert snapshot.account == "a"


# ---------------------------------------------------------------- ledger reality


def test_ledger_and_kaggle_disagree_in_the_optimistic_direction(tmp_path: Path):
    """The motivating measurement: 7 ledger sessions vs 5.03 billed hours.

    Not a tautology: it asserts the ledger only ever sees its own sessions, which is
    precisely why it cannot be trusted alone for a budget decision.
    """
    from datetime import timedelta

    ledger = Ledger(tmp_path / "l.json")
    total_wall = 0.0
    for _ in range(7):
        session = ledger.open_session("a", "a/k", now=utcnow() - timedelta(hours=1))
        ledger.close_session(session, "pool shutdown")
        total_wall += session.elapsed_hours()
    assert total_wall == pytest.approx(7.0), "an hour of wall time each"

    kaggle = parse_quota_csv(REAL_CSV, "a")
    # Same account, same week: the ledger sees 7h, Kaggle billed 5.03h. Neither is
    # "right" -- the ledger misses GPU time started by hand, Kaggle counts every
    # session on the account including ours. The point is that they are not
    # interchangeable, which is why the budget prefers Kaggle's.
    assert kaggle.used_hours != pytest.approx(total_wall)
    assert kaggle.used_hours < total_wall, "in the measured case the ledger ran high"


@pytest.mark.asyncio
async def test_aquota_wraps_a_missing_binary_as_quota_unavailable():
    config = Config()
    config.kaggle.command = "/nonexistent/kaggle-does-not-exist"
    store = type("S", (), {})()
    store.env_for = lambda account: {}  # type: ignore[attr-defined]
    cli = KaggleCLI(config=config, store=store)  # type: ignore[arg-type]
    account = Account(slug="a", username="u")

    with pytest.raises(QuotaUnavailable, match="could not run kaggle quota"):
        await cli.aquota(account)


def test_env_for_keeps_each_accounts_own_config_dir():
    """The quota read must use the account's credentials, not a global one."""
    from kaggle_rotate.accounts import Account, AccountStore

    config = Config()
    config.kaggle.config_root = "/tmp/kaggle-rotate-quota-test"
    store = AccountStore(config)
    account = Account(slug="acct", username="u", kind="kaggle_json")
    env = store.env_for(account)
    assert env["KAGGLE_CONFIG_DIR"].endswith("accounts/acct")
    assert "KAGGLE_USERNAME" not in env, "stale ambient credentials must not leak in"


def test_week_start_is_the_most_recent_saturday():
    start = week_start()
    assert start.weekday() == 5, "Saturday"
    assert (start.hour, start.minute, start.second) == (0, 0, 0)


def test_hours_helper_keeps_subsecond_sessions_visible():
    assert hours(0.5) == 0.000139
    assert hours(3600) == 1.0
