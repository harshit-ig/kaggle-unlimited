"""The pool supervisor: launches kernels, rotates them, and hands out one stable endpoint.

Lifecycle per account slot:

    idle -> launching -> booting -> ready -> active -> draining -> stopped

Rotation is *prewarm-then-cutover*, never *kill-then-restart*: the next account is
booted while the current one is still serving, and only once the replacement reports
a healthy, model-resident endpoint does the router flip. Clients therefore keep one URL
and never observe an outage — at worst an in-flight stream rides the retiring session
to completion.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any

import httpx
import uvicorn
from starlette.applications import Starlette

from .accounts import Account, KaggleCLI
from .config import Config
from .notebook import HEALTH_PATH, LaunchSpec, read_served_model, write_kernel
from .proxy import ProxyApp, Upstream, UpstreamRouter
from .quota import Budget, Ledger, hours
from .relay import RelayState
from .state import PoolState, SlotState
from .tunnel import Tunnel, ensure_cloudflared, start_quick_tunnel, wait_reachable

log = logging.getLogger("kaggle_rotate.pool")

STATUS_POLL_TICKS = 60  # ~10 min at the default tick, when everything is serving
BOOT_POLL_TICKS = 2  # while a kernel is booting, so a crash is noticed fast


@dataclass
class RunningServer:
    server: uvicorn.Server
    task: asyncio.Task

    @property
    def port(self) -> int:
        return self.server.servers[0].sockets[0].getsockname()[1]


class _Server(uvicorn.Server):
    """uvicorn.Server without signal handling; the pool owns the process signals."""

    def install_signal_handlers(self) -> None:  # noqa: D102
        return


def _serve(app: Any, host: str, port: int) -> RunningServer:
    server = _Server(
        uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    )
    return RunningServer(server=server, task=asyncio.create_task(server.serve()))


class Pool:
    def __init__(
        self,
        config: Config,
        accounts: dict[str, Account],
        cli: KaggleCLI,
        ledger: Ledger,
        relay_state: RelayState,
        router: UpstreamRouter,
        state: PoolState,
    ) -> None:
        self.config = config
        self.accounts = accounts
        self.cli = cli
        self.ledger = ledger
        self.relay_state = relay_state
        self.router = router
        self.state = state
        self.budget = Budget(
            session_limit_hours=config.rotation.session_limit_hours,
            weekly_limit_hours=config.rotation.weekly_limit_hours,
        )
        self.proxy = ProxyApp(
            router,
            retry_on_standby=config.proxy.retry_on_standby,
            connect_grace=config.proxy.connect_grace_seconds,
            on_traffic=self.note_traffic,
        )
        self.relay_app = Starlette  # placeholder so linters keep the import meaningful
        self.servers: list[RunningServer] = []
        self.relay_tunnel: Tunnel | None = None
        self.relay_public_url = ""
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=15.0))
        self._tick_count = 0
        self._last_traffic = time.time()
        self._stopping = False
        self._boot_tasks: dict[str, asyncio.Task] = {}
        self._retire_tasks: set[asyncio.Task] = set()
        self._coverage_note = ""
        self._last_coverage_note = ""
        self._relay_dead_logged = False

    # ------------------------------------------------------------------- setup

    @property
    def relay_app_instance(self) -> Any:
        from .relay import build_relay_app

        return build_relay_app(self.relay_state)

    async def start(self) -> None:
        state_dir = self.config.resolved_state_dir()
        state_dir.mkdir(parents=True, exist_ok=True)

        self.servers.append(
            _serve(self.relay_app_instance, self.config.relay.host, self.config.relay.port)
        )
        self.servers.append(_serve(self.proxy.app, self.config.proxy.host, self.config.proxy.port))
        await self._await_ports()

        await self._expose_relay()
        self.ledger.reconcile(max_session_hours=self.config.rotation.session_limit_hours)
        await self._recover_orphans()
        log.info(
            "proxy listening on http://%s:%s/v1", self.config.proxy.host, self.config.proxy.port
        )
        if self.relay_public_url:
            log.info("relay published at %s", self.relay_public_url)

    async def _await_ports(self) -> None:
        for server in self.servers:
            deadline = time.monotonic() + 20
            while not server.server.started and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            if not server.server.started:
                raise RuntimeError("uvicorn server failed to start")

    async def _expose_relay(self) -> None:
        relay = self.config.relay
        if relay.public_url:
            self.relay_public_url = relay.public_url.rstrip("/")
            return
        if not relay.expose:
            raise RuntimeError(
                "relay.expose is false and relay.public_url is empty; Kaggle cannot call back"
            )
        binary = await ensure_cloudflared(
            self.config.resolved_state_dir() / "bin", self.config.kernel.cloudflared_path
        )
        target = f"http://{relay.host}:{relay.port}"
        self.relay_tunnel = await start_quick_tunnel(binary, target, relay.tunnel_start_timeout)
        self.relay_public_url = self.relay_tunnel.url
        if not await wait_reachable(f"{self.relay_public_url}/healthz", relay.tunnel_start_timeout):
            raise RuntimeError(
                f"relay tunnel {self.relay_public_url} is not reachable from the internet"
            )

    async def _recover_orphans(self) -> None:
        """A previous run may have left kernels running; retire them before we start."""
        for slot in self.state.orphans():
            log.warning("retiring orphaned session from a previous run: %s", slot.account)
            await self._retire(slot.account, reason="orphaned by a previous pool run", force=True)

    # -------------------------------------------------------------------- main

    async def run_forever(self) -> None:
        interval = self.config.rotation.tick_seconds
        while not self._stopping:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("pool tick failed; continuing")
            await asyncio.sleep(interval)

    async def _tick(self) -> None:
        self._tick_count += 1
        self.router.sweep()
        self._check_relay_tunnel()
        await self._pulse_relay()
        await self._sync_relay_sessions()
        await self._refresh_status()
        await self._probe_all()
        await self._retire_dead_sessions()

        if self._tick_count % STATUS_POLL_TICKS == 0:
            self._maybe_idle_stop()

        await self._ensure_coverage()

    # ------------------------------------------------------------- relay sync

    async def _sync_relay_sessions(self) -> None:
        """Promote any kernel that has registered and proven itself healthy."""
        for remote in self.relay_state.sessions():
            slot = self.state.get(remote.account)
            if slot.token != remote.token:
                # A registration from a slot we no longer own (stale launch).
                with contextlib.suppress(Exception):
                    await self.relay_state.command(remote.token, "shutdown", "stale registration")
                continue
            # A heartbeat proves the kernel is executing, so bill it from now even if
            # the status poll has not caught up yet.
            self._open_ledger_slot(slot)
            if slot.state in {"booting", "ready"} and remote.phase == "serving":
                if await self._probe(remote.url, model=remote.model):
                    self.state.update(
                        slot.account, state="ready", url=remote.url, model=remote.model
                    )
                    log.info("%s is serving at %s", slot.account, remote.url)
                    await self._admit(slot.account)

    # ------------------------------------------------------------- probing

    async def _probe(self, url: str, model: str = "") -> bool:
        """True when the endpoint is serving a resident model.

        llama-server only answers /health once the model is loaded, so a 200 already
        means "resident". The model name is deliberately not matched: llama-server
        reports the GGUF path as its id, not the MTP_MODEL alias the notebook declares.
        """
        if not url:
            return False
        try:
            response = await self._http.get(f"{url}{HEALTH_PATH}", timeout=20.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def _probe_all(self) -> None:
        for upstream in self.router.candidates():
            healthy = await self._probe(upstream.url, model=upstream.model)
            if healthy != upstream.healthy:
                upstream.healthy = healthy
                if healthy:
                    upstream.last_ok = time.time()
                    log.info("upstream %s healthy", upstream.name)
                else:
                    upstream.last_error = "health probe failed"
                    log.warning("upstream %s unhealthy", upstream.name)

    # ------------------------------------------------------------------ health

    async def _pulse_relay(self) -> None:
        """Tell the relay we are alive, so kernels can tell it from a zombie relay."""
        if not self.relay_public_url or self._stopping:
            return
        if not self.relay_public_url.startswith("http"):
            return  # nothing to point at
        try:
            await self._http.post(
                f"{self.relay_public_url}/_rot/pulse",
                headers={"Authorization": f"Bearer {self.relay_state.driver_token}"},
                timeout=10.0,
            )
        except httpx.HTTPError:
            # The tunnel may be mid-reconnect; the next tick tries again. The notebook
            # has its own orphan timer if this persists.
            log.debug("relay pulse failed", exc_info=True)

    def _check_relay_tunnel(self) -> None:
        """A dead relay tunnel is unrecoverable, so say so loudly and early.

        Kernels can only be told to shut down through this tunnel. When it dies they
        fall back to their own orphan timer, and every rotation decision silently
        becomes impossible - so this must never be a quiet condition.
        """
        tunnel = self.relay_tunnel
        if tunnel is None or self._stopping:
            return
        if tunnel.process.returncode is None:
            self._relay_dead_logged = False
            return
        if self._relay_dead_logged:
            return
        self._relay_dead_logged = True
        log.error(
            "relay tunnel died (exit %s). Kernels can no longer be told to shut down; "
            "they will self-terminate after their orphan grace period. Set "
            "relay.public_url to a stable named tunnel to avoid this.",
            tunnel.process.returncode,
        )

    async def _retire_dead_sessions(self) -> None:
        """Drop a session that stopped answering, so a replacement can take over."""
        grace = self.config.rotation.heartbeat_grace_seconds
        if grace <= 0:
            return
        now = time.time()
        remotes = {s.account: s for s in self.relay_state.sessions()}
        for slot in list(self.state.slots.values()):
            if slot.state not in {"active", "ready"}:
                continue
            upstream = next((u for u in self.router.candidates() if u.name == slot.account), None)
            if upstream is not None and upstream.healthy:
                continue
            remote = remotes.get(slot.account)
            last_signal = remote.last_seen if remote else slot.started_at
            if now - last_signal < grace:
                continue
            log.warning(
                "%s has not responded for %.0fs; retiring it",
                slot.account,
                now - last_signal,
            )
            await self._retire(slot.account, reason="session stopped responding", force=True)

    # ------------------------------------------------------------- status poll

    def _status_poll_due(self) -> bool:
        """Poll often while something is booting, rarely once everything is serving.

        A kernel that dies during setup never registers, so nothing else would notice
        for the whole boot timeout.
        """
        booting = any(slot.state in {"launching", "booting"} for slot in self.state.slots.values())
        every = BOOT_POLL_TICKS if booting else STATUS_POLL_TICKS
        return self._tick_count % every == 0

    async def _refresh_status(self) -> None:
        if not self._status_poll_due():
            return
        for slot in self.state.slots.values():
            if slot.state in {"idle", "stopped", "failed"}:
                continue
            account = self.accounts.get(slot.account)
            if account is None:
                continue
            try:
                status = await self.cli.astatus(account)
            except Exception as exc:
                log.debug("status check failed for %s: %s", slot.account, exc)
                continue
            if slot.state in {"launching", "booting"} and status.state in {"RUNNING", "STARTING"}:
                self._open_ledger_slot(slot)
            if status.terminal and slot.state not in {"stopped", "failed"}:
                log.warning("kernel %s finished with %s", slot.account, status.state)
                await self._retire(
                    slot.account, reason=f"kernel {status.state.lower()}", force=True
                )
            if not status.alive and not status.terminal and slot.state in {"active", "ready"}:
                await self._retire(slot.account, reason="kernel no longer running", force=True)

    def _open_ledger_slot(self, slot: SlotState) -> None:
        """Start billing a session the moment its kernel is actually running."""
        if self.ledger.find_live(slot.account) is not None:
            return
        self.ledger.open_session(slot.account, slot.kernel_ref)
        log.info("GPU session opened for %s (%s)", slot.account, slot.kernel_ref)

    # ------------------------------------------------------------- coverage

    async def _ensure_coverage(self) -> None:
        active_slot = self._active_slot()

        if active_slot is None:
            if self._pick_launch_target(force_if_idle=True) is None:
                self._log_coverage_gap()
            return

        if self._should_prewarm(active_slot):
            if self._pick_launch_target() is None:
                self._log_coverage_gap()

    def _log_coverage_gap(self) -> None:
        """Report an empty pool once per distinct reason, not once per tick."""
        note = self._coverage_note
        if note == self._last_coverage_note:
            return
        self._last_coverage_note = note
        log.warning("no session can start right now: %s", note)

    def _active_slot(self) -> SlotState | None:
        for slot in self.state.slots.values():
            if slot.state == "active":
                return slot
        return None

    def _should_prewarm(self, active: SlotState) -> bool:
        rotation = self.config.rotation
        account = self.accounts.get(active.account)
        if account is None:
            return False
        now = time.time()

        if active.started_at:
            session_left = hours(rotation.session_limit_hours * 3600 - (now - active.started_at))
            if session_left <= rotation.prewarm_lead_minutes / 60.0:
                return True

        weekly_left = self.budget.weekly_remaining_hours(
            self.ledger,
            active.account,
            account.effective_weekly_limit(rotation.weekly_limit_hours),
        )
        return weekly_left <= rotation.boot_reserve_hours + rotation.weekly_safety_minutes / 60.0

    def _live_kernel_count(self) -> int:
        return len(
            [
                s
                for s in self.state.slots.values()
                if s.state in {"launching", "booting", "ready", "active"}
            ]
        )

    def _pick_launch_target(self, force_if_idle: bool = False) -> Account | None:
        """Start a session on whichever account has the most weekly budget left."""
        rotation = self.config.rotation
        if self._live_kernel_count() >= rotation.max_active_sessions + 1:
            self._coverage_note = f"all {rotation.max_active_sessions + 1} kernel slots are busy"
            return None

        candidates: list[tuple[float, Account]] = []
        skipped: list[str] = []
        now = time.time()
        for account in self.accounts.values():
            if not account.enabled:
                continue
            slot = self.state.get(account.slug)
            if slot.state in {"launching", "booting", "ready", "active", "draining"}:
                skipped.append(f"{account.slug}: already has a session")
                continue
            # A failing account must not be retried on every tick.
            if slot.attempts and now - slot.last_attempt < self._launch_backoff(slot.attempts):
                # No countdown here: the exact wait is already logged once, at failure
                # time, and a moving value would defeat note de-duplication.
                skipped.append(f"{account.slug}: backing off after a failed launch")
                continue
            ok, reason = self.budget.can_start(
                self.ledger,
                account.slug,
                account_limit=account.effective_weekly_limit(rotation.weekly_limit_hours),
                reserve_hours=rotation.boot_reserve_hours if not force_if_idle else 0.0,
            )
            if not ok:
                skipped.append(f"{account.slug}: {reason}")
                continue
            remaining = self.budget.weekly_remaining_hours(
                self.ledger,
                account.slug,
                account.effective_weekly_limit(rotation.weekly_limit_hours),
            )
            candidates.append((remaining, account))

        self._coverage_note = "; ".join(skipped) or "no enabled accounts"
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        chosen = candidates[0][1]
        log.info(
            "starting a session on %s (%.1fh of weekly quota left)", chosen.slug, candidates[0][0]
        )
        self._boot_tasks[chosen.slug] = asyncio.create_task(self._launch_and_wait(chosen))
        return chosen

    # ------------------------------------------------------------------ launch

    async def _launch_and_wait(self, account: Account) -> None:
        self.state.update(account.slug, last_attempt=time.time())
        try:
            spec = await self._launch(account)
        except Exception as exc:
            log.error("launch failed for %s: %s", account.slug, exc)
            slot = self.state.get(account.slug)
            slot.attempts += 1
            self.state.update(account.slug, state="failed", detail=str(exc))
            self._close_ledger(account.slug, reason=f"launch failed: {exc}")
            log.info(
                "next attempt for %s in %.0fs",
                account.slug,
                self._launch_backoff(slot.attempts),
            )
            return
        try:
            await self._await_ready(account.slug, spec)
        except Exception as exc:
            log.error("boot failed for %s: %s", account.slug, exc)
            slot = self.state.get(account.slug)
            slot.attempts += 1
            self.state.update(account.slug, state="failed", detail=str(exc))
            await self._retire(account.slug, reason="boot failed", force=True)
            return
        finally:
            self._boot_tasks.pop(account.slug, None)

    def _launch_backoff(self, attempts: int) -> float:
        rotation = self.config.rotation
        return min(
            rotation.launch_retry_seconds * max(1, attempts), rotation.launch_retry_max_seconds
        )

    def _close_ledger(self, account_slug: str, reason: str) -> None:
        live = self.ledger.find_live(account_slug)
        if live is not None:
            self.ledger.close_session(live, reason)

    async def _launch(self, account: Account) -> LaunchSpec:
        kernel = self.config.kernel
        served = read_served_model(self.config.resolved_source_notebook())
        token = secrets.token_urlsafe(32)
        spec = LaunchSpec(
            account=account.slug,
            kernel_ref=account.ref,
            relay_url=self.relay_public_url,
            token=token,
            model=served.model,
            max_runtime_seconds=kernel.max_runtime_minutes * 60.0,
            shutdown_poll_seconds=kernel.shutdown_poll_seconds,
            orphan_grace_seconds=kernel.orphan_grace_seconds,
        )

        launch_dir = self.config.resolved_state_dir() / "launches" / account.slug
        write_kernel(self.config, spec, launch_dir)

        self.state.update(
            account.slug,
            state="launching",
            kernel_ref=account.ref,
            token=token,
            started_at=time.time(),
            url="",
            model=served.model,
            detail="pushing kernel",
        )
        log.info("pushing kernel for %s (%s)", account.slug, account.ref)

        timeout_s = int(kernel.max_runtime_minutes * 60) + 600
        argv = [
            self.config.kaggle.command,
            "kernels",
            "push",
            "-p",
            str(launch_dir),
            "--accelerator",
            kernel.accelerator,
            "-t",
            str(timeout_s),
        ]
        process = await asyncio.create_subprocess_exec(
            *argv,
            env=self.cli.store.env_for(account),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                stdout.decode("utf-8", "replace").strip() or "kaggle kernels push failed"
            )

        self.state.update(account.slug, state="booting", detail="waiting for tunnel + model")
        return spec

    async def _await_ready(self, account_slug: str, spec: LaunchSpec) -> None:
        rotation = self.config.rotation
        deadline = time.monotonic() + rotation.boot_timeout_seconds
        started = time.monotonic()
        last_progress = 0.0
        while time.monotonic() < deadline and not self._stopping:
            remote = await self.relay_state.get(spec.token)
            if remote is not None and remote.phase == "serving":
                if await self._probe(remote.url, model=spec.model):
                    self.state.update(
                        account_slug,
                        state="ready",
                        url=remote.url,
                        model=remote.model,
                        detail="ready",
                    )
                    log.info("%s ready at %s", account_slug, remote.url)
                    await self._admit(account_slug)
                    return

            # A kernel that crashed during setup will never register, so check whether
            # it is still alive rather than waiting out the whole boot timeout.
            if self._status_poll_due():
                account = self.accounts.get(account_slug)
                if account is not None:
                    status = await self.cli.astatus(account)
                    if status.known and status.failed:
                        raise RuntimeError(
                            f"kernel {account.ref} {status.state} during boot; open the "
                            "notebook on kaggle.com to read the cell output"
                        )

            # Booting takes 10-25 minutes (weights + VRAM). Without this the terminal
            # looks hung, and stage names point at the notebook on kaggle.com.
            if time.monotonic() - last_progress >= 60:
                last_progress = time.monotonic()
                minutes = (time.monotonic() - started) / 60
                stage = (
                    "tunnel is up, waiting for the model to land in VRAM"
                    if remote is not None
                    else "waiting for Kaggle to schedule the kernel, then pulling ~17GB"
                )
                log.info("%s still booting (%.0f min): %s", account_slug, minutes, stage)
            await asyncio.sleep(15)
        raise TimeoutError(
            f"{account_slug} did not become ready within {rotation.boot_timeout_seconds}s"
        )

    # ----------------------------------------------------------------- promote

    async def _admit(self, account_slug: str) -> None:
        """A session is healthy and model-resident; decide whether it takes traffic."""
        slot = self.state.get(account_slug)
        if slot.state != "ready" or not slot.url:
            return
        slot.attempts = 0  # it booted, so clear any launch backoff

        upstream = Upstream(
            name=account_slug,
            url=slot.url,
            token=slot.token,
            model=slot.model,
            account=account_slug,
            kernel_ref=slot.kernel_ref,
            healthy=True,
            meta={"started_at": slot.started_at},
        )

        if self.router.active is None:
            await self._cutover(account_slug, upstream, "no active session")
            return

        if self.router.active.name == account_slug:
            self.state.update(account_slug, state="active")
            return

        if self.router.standby is not None and self.router.standby.name != account_slug:
            await self._retire(account_slug, reason="standby already warm")
            return

        self.router.add_standby(upstream)
        self.state.update(account_slug, state="ready", detail="standby")
        log.info("%s warmed as standby", account_slug)

        active_slot = self._active_slot()
        if active_slot is None:
            return
        if not self.router.active.healthy or self._should_prewarm(active_slot):
            await self._cutover(
                account_slug,
                upstream,
                f"replacing {active_slot.account} ({'unhealthy' if not self.router.active.healthy else 'nearing its budget'})",
            )

    async def _cutover(self, account_slug: str, upstream: Upstream, reason: str) -> None:
        """Flip traffic to the new session; the old one drains in the background."""
        retired = self.router.set_active(upstream)
        self.state.update(account_slug, state="active", detail="active")
        if retired is None or retired.name == account_slug:
            log.info("cutover: %s is now serving client traffic", account_slug)
            return

        self.state.update(retired.name, state="draining", detail=f"superseded by {account_slug}")
        log.info("cutover: %s -> %s (%s)", retired.name, account_slug, reason)
        task = asyncio.create_task(
            self._retire(retired.name, reason=f"superseded by {account_slug}")
        )
        self._retire_tasks.add(task)
        task.add_done_callback(self._retire_tasks.discard)

    # ------------------------------------------------------------------ retire

    async def _retire(
        self,
        account_slug: str,
        reason: str,
        force: bool = False,
        grace: float | None = None,
    ) -> None:
        """Stop a session. Deleting the kernel is the stop; nothing else is reliable.

        The relay command is best-effort politeness (it lets the notebook tear down
        cleanly), but we never wait on it: if the relay is down or wedged, deleting
        still stops the GPU burn immediately.
        """
        slot = self.state.get(account_slug)
        if slot.state in {"stopped", "idle", "failed"} and not force:
            return

        self.state.update(account_slug, state="draining", detail=reason)
        self.router.drop(account_slug)

        if slot.token:
            with contextlib.suppress(Exception):
                await self.relay_state.command(slot.token, "shutdown", reason)

        # Rotation keeps the retiring session up briefly so in-flight streams finish.
        # Shutdown passes grace=0 and deletes straight away.
        wait = self.config.rotation.drain_grace_seconds if grace is None else grace
        if wait:
            await asyncio.sleep(wait)

        deleted = True
        account = self.accounts.get(account_slug)
        if account is not None:
            deleted = await self.cli.adelete(account)

        live = self.ledger.find_live(account_slug)
        if live is not None:
            self.ledger.close_session(live, reason)

        if not deleted:
            # Do not report a stop that did not happen: the kernel is still billing.
            self.state.update(
                account_slug,
                state="failed",
                url="",
                token="",
                detail=f"kernel delete FAILED ({reason}); run `kaggle-rotate cleanup`",
            )
            log.error(
                "%s: delete failed, the kernel may still be running and billing GPU time. "
                "Run `kaggle-rotate cleanup`.",
                account_slug,
            )
            return

        self.state.update(account_slug, state="stopped", url="", token="", detail=reason)
        log.info("%s stopped: %s", account_slug, reason)

    # -------------------------------------------------------------- idle stop

    def _maybe_idle_stop(self) -> None:
        limit = self.config.rotation.idle_stop_minutes
        if not limit:
            return
        if time.time() - self._last_traffic < limit * 60:
            return
        for slot in self.state.slots.values():
            if slot.state in {"active", "ready"}:
                log.info("idle for %s minutes; stopping to save quota", limit)
                task = asyncio.create_task(self._retire(slot.account, reason="idle timeout"))
                self._retire_tasks.add(task)
                task.add_done_callback(self._retire_tasks.discard)
                return

    def note_traffic(self) -> None:
        self._last_traffic = time.time()

    # --------------------------------------------------------------- shutdown

    async def shutdown(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        log.info("shutting down pool")

        for task in list(self._boot_tasks.values()):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        if self._retire_tasks:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    asyncio.gather(*self._retire_tasks, return_exceptions=True), timeout=90
                )

        for slot in list(self.state.slots.values()):
            if slot.state in {"active", "ready", "draining", "booting", "launching"}:
                with contextlib.suppress(Exception):
                    await self._retire(slot.account, reason="pool shutdown", force=True, grace=0.0)

        await self.proxy.aclose()
        await self._http.aclose()
        if self.relay_tunnel is not None:
            await self.relay_tunnel.stop()
        for server in self.servers:
            server.server.should_exit = True
        for server in self.servers:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(server.task, timeout=10)

    # ------------------------------------------------------------------ status

    def status(self) -> dict[str, Any]:
        now = time.time()
        rotation = self.config.rotation
        # Include ledger-only accounts so a restarted pool still shows the budget
        # it is working against.
        names = set(self.state.slots) | {s.account for s in self.ledger.sessions()}
        slots = []
        for name in sorted(names):
            slot = self.state.get(name)
            account = self.accounts.get(name)
            limit = (
                account.effective_weekly_limit(rotation.weekly_limit_hours)
                if account
                else rotation.weekly_limit_hours
            )
            entry = slot.to_public()
            live = self.ledger.find_live(slot.account)
            entry["weekly_used_hours"] = self.ledger.used_hours(slot.account)  # noqa: E501
            entry["weekly_limit_hours"] = limit
            entry["weekly_remaining_hours"] = max(0.0, limit - entry["weekly_used_hours"])
            entry["session_remaining_hours"] = (
                round(rotation.session_limit_hours - live.elapsed_hours(), 3)
                if live is not None
                else rotation.session_limit_hours
            )
            if slot.started_at:
                entry["session_age_hours"] = hours(now - slot.started_at)
            slots.append(entry)

        return {
            "relay": {
                "local": f"http://{self.config.relay.host}:{self.config.relay.port}",
                "public": self.relay_public_url,
            },
            "proxy": {
                "base_url": f"http://{self.config.proxy.host}:{self.config.proxy.port}/v1",
                "upstreams": self.router.status(),
            },
            "quota": {
                "week_started_at": self.ledger.usage(next(iter(self.accounts), "")).week_started_at,
                "total_used_hours": self.ledger.total_used_hours(),
            },
            "sessions": self.ledger.sessions()[-20:],
            "slots": slots,
            "remote_sessions": [s.to_public() for s in self.relay_state.sessions()],
        }
