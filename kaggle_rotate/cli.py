"""Command line interface."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import shutil
import signal
import sys
from pathlib import Path

from .accounts import Account, AccountStore, KaggleCLI, KaggleError, kernel_is_gone
from .config import Config, load_config
from .notebook import LaunchSpec, build_notebook, resolve_model_config
from .pool import Pool
from .proxy import UpstreamRouter
from .quota import Ledger, utcnow, week_start
from .relay import RelayState
from .state import PidFile, PoolState

log = logging.getLogger("kaggle_rotate")


def _setup_logging(verbose: bool, log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    # httpx logs every request at INFO, including signed GitHub download URLs and a
    # health probe every few seconds. That drowns the lines that actually matter.
    logging.getLogger("httpx").setLevel(logging.DEBUG if verbose else logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.DEBUG if verbose else logging.WARNING)


# --------------------------------------------------------------------------- init


def cmd_init(args: argparse.Namespace, config: Config) -> int:
    store = AccountStore(config)
    cli = KaggleCLI(config, store)
    accounts = store.load()

    state_dir = config.resolved_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    print(f"state directory: {state_dir}")

    source = config.resolved_source_notebook()
    if source.exists():
        model = resolve_model_config(config)
        print(f"source notebook: {source.name}")
        print(f"  model:        {model.model}")
        print(f"  base model:   {model.source_model}")
        print(f"  num_ctx:      {model.num_ctx}")
        print(f"  draft tokens: {model.draft_num_predict}")
    else:
        print(f"source notebook not found: {source}")

    credential_files = [Path(p).expanduser() for p in (args.credentials or [])]
    while True:
        account: Account | None
        if credential_files:
            account = _store_json(store, accounts, "", str(credential_files.pop(0)))
        elif args.yes:
            break
        else:
            path_text = input(
                "Path to a kaggle.json (blank when done, or 'token' to paste an API token): "
            ).strip()
            if not path_text:
                break
            if path_text.lower() == "token":
                slug = input("  account slug (e.g. work): ").strip()
                token = input("  KAGGLE_API_TOKEN: ").strip()
                if not slug or not token:
                    print("  skipped", file=sys.stderr)
                    continue
                account = _store_token(store, accounts, slug, token)
            else:
                slug = input("  account slug (blank to derive from username): ").strip()
                account = _store_json(store, accounts, slug, path_text)

        if account is None:
            continue
        if account.slug in accounts:
            print(f"  ! {account.slug} already registered; overwriting")
        accounts[account.slug] = account
        store.save(accounts)

        try:
            cli.verify(account)
            print(f"  ok {account.slug}: authenticated as {account.username}")
        except KaggleError as exc:
            print(f"  ! {account.slug}: verification failed: {exc}", file=sys.stderr)
            accounts.pop(account.slug, None)
            store.save(accounts)

    if not accounts:
        print("\nNo accounts configured. Nothing to rotate yet.", file=sys.stderr)
        return 1

    print("\nConfigured accounts:")
    for account in accounts.values():
        limit = account.weekly_limit_hours or config.rotation.weekly_limit_hours
        print(f"  {account.slug:<16} {account.username:<24} weekly budget {limit}h")
    print(f"\nNext: {sys.argv[0] or 'kaggle-rotate'} run")
    print(f"Clients should use: http://{config.proxy.host}:{config.proxy.port}/v1")
    return 0


def _store_json(store: AccountStore, accounts: dict, slug: str, path: str) -> Account | None:
    from .accounts import slugify

    src = Path(path).expanduser()
    try:
        data = json.loads(src.read_text())
        username = str(data.get("username", "")).strip()
        if not username or not data.get("key"):
            raise ValueError("not a valid kaggle.json (need 'username' and 'key')")
    except (OSError, ValueError) as exc:
        print(f"  ! cannot read {src}: {exc}", file=sys.stderr)
        return None

    final = slugify(slug or username)
    try:
        return store.install_credentials(final, "kaggle_json", src)
    except (OSError, ValueError) as exc:
        print(f"  ! {exc}", file=sys.stderr)
        return None


def _store_token(store: AccountStore, accounts: dict, slug: str, token: str) -> Account | None:
    from .accounts import slugify

    final = slugify(slug)
    directory = store.account_dir(final)
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / "access_token"
    probe.write_text(token + "\n")
    probe.chmod(0o600)
    try:
        return store.install_credentials(final, "access_token", probe)
    except (OSError, ValueError) as exc:
        print(f"  ! {exc}", file=sys.stderr)
        return None


# ------------------------------------------------------------------------ accounts


def cmd_add_account(args: argparse.Namespace, config: Config) -> int:
    store = AccountStore(config)
    cli = KaggleCLI(config, store)
    accounts = store.load()
    if args.kind == "token":
        if not args.token:
            print("--token is required", file=sys.stderr)
            return 2
        account = _store_token(store, accounts, args.slug or "account", args.token)
    else:
        if not args.credential:
            print("--credential is required", file=sys.stderr)
            return 2
        account = _store_json(store, accounts, args.slug or "", args.credential)
    if account is None:
        return 1
    accounts[account.slug] = account
    store.save(accounts)
    try:
        cli.verify(account)
    except KaggleError as exc:
        print(f"! verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"added {account.slug} ({account.username})")
    return 0


def cmd_accounts(args: argparse.Namespace, config: Config) -> int:
    store = AccountStore(config)
    ledger = Ledger(config.resolved_state_dir() / "ledger.json")
    accounts = store.load()
    if not accounts:
        print("no accounts configured; run `kaggle-rotate init`")
        return 1
    print(f"{'slug':<18}{'username':<24}{'kind':<14}{'used':>8}{'left':>8}  limit")
    for account in accounts.values():
        limit = account.effective_weekly_limit(config.rotation.weekly_limit_hours)
        used = ledger.used_hours(account.slug)
        left = max(0.0, limit - used)
        print(
            f"{account.slug:<18}{account.username:<24}{account.kind:<14}"
            f"{used:>7.2f}h{left:>7.2f}h  {limit:g}h"
        )
    print(
        f"\nquota week started {week_start(utcnow()).isoformat()} (Kaggle resets Saturday 00:00 UTC)"
    )
    return 0


# ------------------------------------------------------------------------- status


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def cmd_status(args: argparse.Namespace, config: Config) -> int:
    state_dir = config.resolved_state_dir()
    pool_state = PoolState(state_dir / "pool.json")
    ledger = Ledger(state_dir / "ledger.json")
    store = AccountStore(config)
    accounts = store.load()
    pidfile = PidFile(state_dir / "pool.pid")
    pid = pidfile.read()

    if args.json:
        print(
            json.dumps(
                {
                    "pid": pid,
                    "slots": [slot.to_public() for slot in pool_state.slots.values()],
                    "ledger": [s.to_public() for s in ledger.sessions()],
                    "relay_public_url": _read_json(state_dir / "relay.json").get("url", ""),
                },
                indent=2,
            )
        )
        return 0

    state = f"running (pid {pid})" if pid else "not running"
    print(f"pool: {state}")
    relay = _read_json(state_dir / "relay.json").get("url", "")
    if relay:
        print(f"relay (public): {relay}")
    print(f"proxy: http://{config.proxy.host}:{config.proxy.port}/v1")
    if not pool_state.slots:
        print("\nno sessions recorded")
        return 0

    print(
        f"\n{'account':<16}{'state':<12}{'session':>10}{'left':>9}{'weekly':>9}{'w-left':>9}  model"
    )
    for name, slot in sorted(pool_state.slots.items()):
        account = accounts.get(name)
        limit = (
            account.effective_weekly_limit(config.rotation.weekly_limit_hours)
            if account
            else config.rotation.weekly_limit_hours
        )
        used = ledger.used_hours(name)
        live = ledger.find_live(name)
        elapsed = live.elapsed_hours() if live else 0.0
        print(
            f"{name:<16}{slot.state:<12}"
            f"{elapsed:>9.2f}h{config.rotation.session_limit_hours - elapsed:>8.2f}h"
            f"{used:>8.2f}h{max(0.0, limit - used):>8.2f}h  {slot.model or '-'}"
        )
    return 0


def cmd_cleanup(args: argparse.Namespace, config: Config) -> int:
    """Delete every kernel this tool may have left running.

    The pool is not required to be alive. Deleting is the only dependable stop, so this
    does exactly that and nothing more.
    """
    store = AccountStore(config)
    cli = KaggleCLI(config, store)
    accounts = store.load()
    if not accounts:
        print("no accounts configured; run `kaggle-rotate init` first", file=sys.stderr)
        return 1

    failed: list[str] = []
    for account in accounts.values():
        label = f"{account.slug} ({account.ref})"
        if args.dry_run:
            print(f"  -- {label}: would delete")
            continue
        print(f"  .. {label}: deleting")
        try:
            deleted = cli.delete(account)
        except KaggleError as exc:
            print(f"  ! {label}: {exc}", file=sys.stderr)
            failed.append(label)
            continue
        if deleted:
            print(f"  ok {label}: deleted")
        else:
            # Kaggle answers 403 for a kernel that is already gone, so check before
            # telling the user their GPU is still being billed.
            try:
                after = cli.status(account)
            except KaggleError as exc:
                if kernel_is_gone(exc):
                    print(f"  ok {label}: already gone")
                else:
                    print(f"  ? {label}: delete failed, status unavailable: {exc}", file=sys.stderr)
                    failed.append(label)
                continue
            if not after.known or after.alive:
                print(f"  ! {label}: delete failed, kernel still {after.state}", file=sys.stderr)
                failed.append(label)
            else:
                print(f"  ok {label}: already gone (status {after.state})")

    if failed:
        print(f"\nfailed: {'; '.join(failed)}", file=sys.stderr)
        return 1
    if args.dry_run:
        print("\ndry run; nothing was changed")
    else:
        print("\nall managed kernels deleted")
    return 0


# ------------------------------------------------------------------------- render


def cmd_render(args: argparse.Namespace, config: Config) -> int:
    model = resolve_model_config(config)
    spec = LaunchSpec(
        account=args.account or "example",
        kernel_ref=f"{args.account or 'example'}/{config.kernel.kernel_slug}",
        relay_url=args.relay_url or "https://REPLACE-ME.trycloudflare.com",
        token="render-only-token",
        model=model.model,
        source_model=model.source_model,
        num_ctx=model.num_ctx,
        draft_num_predict=model.draft_num_predict,
        max_runtime_seconds=config.kernel.max_runtime_minutes * 60,
        shutdown_poll_seconds=config.kernel.shutdown_poll_seconds,
    )
    notebook = build_notebook(config, spec)
    out = Path(args.out) if args.out else config.resolved_state_dir() / "rendered.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"wrote {out} ({len(notebook['cells'])} cells)")
    return 0


# ------------------------------------------------------------------------- doctor


def cmd_doctor(args: argparse.Namespace, config: Config) -> int:
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        mark = "ok " if passed else "FAIL"
        print(f"[{mark}] {label}{(' - ' + detail) if detail else ''}")

    binary = shutil.which(config.kaggle.command) or config.kaggle.command
    check("kaggle CLI", shutil.which(config.kaggle.command) is not None, binary)
    check(
        "cloudflared",
        True,
        str(config.kernel.cloudflared_path)
        if config.kernel.cloudflared_path or shutil.which("cloudflared")
        else "not installed yet; will be downloaded into the state dir on first run",
    )

    import socket

    for label, host, port in (
        ("proxy port", config.proxy.host, config.proxy.port),
        ("relay port", config.relay.host, config.relay.port),
    ):
        sock = socket.socket()
        try:
            sock.bind((host, port))
            check(f"{label} {host}:{port} free", True)
        except OSError as exc:
            check(f"{label} {host}:{port} free", False, str(exc))
        finally:
            sock.close()

    source = config.resolved_source_notebook()
    check(f"source notebook {source.name}", source.exists(), str(source))
    if source.exists():
        model = resolve_model_config(config)
        check("model resolved", bool(model.model), model.model or "none")
        check("base model resolved", bool(model.source_model), model.source_model or "none")

    store = AccountStore(config)
    accounts = store.load()
    check("accounts configured", bool(accounts), f"{len(accounts)} account(s)")
    for account in accounts.values():
        directory = store.account_dir(account.slug)
        name = "kaggle.json" if account.kind == "kaggle_json" else "access_token"
        present = (directory / name).is_file()
        check(f"credentials for {account.slug}", present, str(directory / name))
        if present:
            with contextlib.suppress(KaggleError):
                KaggleCLI(config, store).verify(account)
    return 0 if ok else 1


# ----------------------------------------------------------------------------- run


async def _run_pool(config: Config) -> int:
    store = AccountStore(config)
    accounts = store.load()
    if not accounts:
        print("no accounts configured; run `kaggle-rotate init` first", file=sys.stderr)
        return 1

    cli = KaggleCLI(config, store)
    state_dir = config.resolved_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)

    relay_state = RelayState(driver_grace_seconds=config.relay.driver_grace_seconds)
    router = UpstreamRouter(drain_grace=config.proxy.drain_grace_seconds)
    pool = Pool(
        config=config,
        accounts=accounts,
        cli=cli,
        ledger=Ledger(state_dir / "ledger.json"),
        relay_state=relay_state,
        router=router,
        state=PoolState(state_dir / "pool.json"),
    )

    pidfile = PidFile(state_dir / "pool.pid")
    existing = pidfile.read()
    if existing:
        print(f"another pool is already running (pid {existing})", file=sys.stderr)
        return 1
    pidfile.write()

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    try:
        await pool.start()
        (state_dir / "relay.json").write_text(json.dumps({"url": pool.relay_public_url}) + "\n")
        print(f"\nclient base URL: http://{config.proxy.host}:{config.proxy.port}/v1")
        print(f"model name:      {resolve_model_config(config).model or '(unset)'}")
        print("press Ctrl-C to stop\n")
        waiter = asyncio.create_task(stop.wait())
        runner = asyncio.create_task(pool.run_forever())
        done, _ = await asyncio.wait({waiter, runner}, return_when=asyncio.FIRST_COMPLETED)
        for task in (waiter, runner):
            if task in done:
                task.cancel()
    finally:
        (state_dir / "relay.json").unlink(missing_ok=True)
        await pool.shutdown()
        pidfile.clear()
    return 0


def cmd_run(args: argparse.Namespace, config: Config) -> int:
    return asyncio.run(_run_pool(config))


# ---------------------------------------------------------------------------- stop


def cmd_stop(args: argparse.Namespace, config: Config) -> int:
    pidfile = PidFile(config.resolved_state_dir() / "pool.pid")
    pid = pidfile.read()
    if pid is None:
        print("no pool is running")
        return 1
    import os

    os.kill(pid, signal.SIGTERM)
    print(f"sent SIGTERM to pid {pid}")
    return 0


# -------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaggle-rotate",
        description="Run one notebook across several Kaggle accounts behind a single endpoint.",
    )
    parser.add_argument("-c", "--config", help="path to config.toml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="one-time account setup")
    p_init.add_argument("credentials", nargs="*", help="kaggle.json paths (optional)")
    p_init.add_argument("-y", "--yes", action="store_true", help="non-interactive")
    p_init.set_defaults(func=cmd_init)

    p_add = sub.add_parser("add-account", help="register another Kaggle account")
    p_add.add_argument("slug", nargs="?")
    p_add.add_argument("--credential", help="path to kaggle.json")
    p_add.add_argument("--token", help="raw KAGGLE_API_TOKEN")
    p_add.add_argument("--kind", choices=("kaggle_json", "access_token"), default="kaggle_json")
    p_add.set_defaults(func=cmd_add_account)

    p_accounts = sub.add_parser("accounts", help="show accounts and weekly quota")
    p_accounts.set_defaults(func=cmd_accounts)

    p_status = sub.add_parser("status", help="show pool state")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=cmd_status)

    p_render = sub.add_parser("render", help="write the generated Kaggle notebook")
    p_render.add_argument("--out")
    p_render.add_argument("--account")
    p_render.add_argument("--relay-url")
    p_render.set_defaults(func=cmd_render)

    p_clean = sub.add_parser("cleanup", help="stop any kernel left running by an earlier session")
    p_clean.add_argument("-n", "--dry-run", action="store_true", help="report only")
    p_clean.set_defaults(func=cmd_cleanup)

    p_doctor = sub.add_parser("doctor", help="check the local setup")
    p_doctor.set_defaults(func=cmd_doctor)

    p_run = sub.add_parser("run", help="start the pool (foreground)")
    p_run.set_defaults(func=cmd_run)

    p_stop = sub.add_parser("stop", help="stop a running pool")
    p_stop.set_defaults(func=cmd_stop)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    _setup_logging(args.verbose, config.resolved_state_dir() / "kaggle-rotate.log")
    try:
        return int(args.func(args, config) or 0)
    except KeyboardInterrupt:
        return 130
    except (KaggleError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
