"""Builds the Kaggle-side server notebook from the user's source notebook.

The source notebook's setup cells (PrismML llama.cpp build, GGUF download,
llama-server launch) are reused verbatim so the server inherits whatever the user
already tuned and proved works. The notebook is the single source of truth for the
model, the weights and the context window: there is no config surface for any of it.

Everything after the tunnel section is dropped and replaced with:

  * a Cloudflare quick tunnel for the model,
  * a readiness gate on llama-server,
  * registration with the local control relay,
  * a supervisor loop that heartbeats, honours remote shutdown, and self-terminates
    before Kaggle's hard session cap.

Placeholders use `@NAME@` markers rather than str.format() so the generated code can
contain braces freely.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .naming import slugify

# ---------------------------------------------------------------- source parsing

_ASSIGN_RE = r"""^\s*{name}\s*=\s*["']?(?P<value>[^"'\n#]+)["']?\s*$"""

# llama-server binds the loopback port the notebook's own `OLLAMA_URL` alias points at.
# The PrismML launcher reads PORT=8080; TUNNEL_CELL must expose the same port.
MODEL_BASE_URL = "http://127.0.0.1:8080"
HOST_HEADER = "localhost:8080"
HEALTH_PATH = "/health"

# Used only if the source notebook ever stops declaring these, so a build still
# renders and the failure names the notebook instead of raising.
FALLBACK_MODEL = "ternary-bonsai-2-27b-pq2"
FALLBACK_NUM_CTX = 262144


@dataclass
class ServedModel:
    """What the generated notebook serves, as declared by the source notebook."""

    model: str = FALLBACK_MODEL
    num_ctx: int = FALLBACK_NUM_CTX


def _assignment(source: str, name: str) -> str | None:
    match = re.search(_ASSIGN_RE.format(name=re.escape(name)), source, re.MULTILINE)
    return match.group("value").strip().strip("\"'") if match else None


def read_served_model(path: Path) -> ServedModel:
    """Read the served model name and context width out of the source notebook.

    The notebook owns these because it is what actually loads them: `MTP_MODEL` is the
    name llama-server is started under and `NUM_CTX` is the context the PrismML
    launcher was tuned for. The local pool needs the name too (it reports it from the
    relay and matches it in `doctor`), so it is parsed here rather than duplicated in
    config.
    """
    if not path.exists():
        return ServedModel()
    notebook = json.loads(path.read_text())
    blob = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))

    num_ctx = _assignment(blob, "NUM_CTX")
    try:
        parsed_ctx = int(str(num_ctx).strip())
    except (TypeError, ValueError):
        parsed_ctx = FALLBACK_NUM_CTX
    return ServedModel(
        model=_assignment(blob, "MTP_MODEL") or FALLBACK_MODEL,
        num_ctx=parsed_ctx,
    )


# ------------------------------------------------------------------- transforms

# Markdown headers whose section is rebuilt (or skipped) by kaggle-rotate. Dropping the
# header also drops the code cell that follows it, which is how the expensive demo
# cells get out of the boot path.
DROP_MARKDOWN_MARKERS = (
    "Raw llama.cpp benchmark",
    "API benchmark",
)

# Code cells after these are hard-cut: everything from here on is exposure/UI code,
# which kaggle-rotate replaces with its own tunnel and control-plane cells.
CUTOFF_CODE_MARKERS = ("cloudflared", "PUBLIC_OLLAMA_URL")

_SHELL_LINE = re.compile(r"^\s*!(?P<cmd>.+)$")


def _rewrite_shell_cell(source: str) -> str:
    """Turn `!cmd` notebook lines into subprocess calls.

    A leading `!` is a Python syntax error inside a notebook kernel, and naive
    interpolation would break on any command containing quotes, so the command is
    repr()'d as a single literal instead.
    """
    lines = source.splitlines()
    if not any(_SHELL_LINE.match(line) for line in lines):
        return source
    rewritten: list[str] = []
    for line in lines:
        match = _SHELL_LINE.match(line)
        if not match:
            rewritten.append(line)
            continue
        rewritten.append(f"sh({match.group('cmd').strip()!r})")
    return "\n".join(rewritten)


def extract_setup_cells(notebook: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep the source notebook's setup cells, dropping UI/demo code we replace."""
    kept: list[dict[str, Any]] = []
    dropping_code = False

    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            if any(marker in source for marker in DROP_MARKDOWN_MARKERS):
                dropping_code = True
                continue
            dropping_code = False
            kept.append(dict(cell))
            continue

        if dropping_code:
            continue
        if any(marker in source for marker in CUTOFF_CODE_MARKERS):
            break
        rewritten = dict(cell)
        source = _rewrite_shell_cell(source)
        rewritten["source"] = source.splitlines(keepends=True)
        kept.append(rewritten)

    # A header immediately before the cutoff would dangle without its section.
    while kept and kept[-1].get("cell_type") == "markdown":
        kept.pop()
    return kept


# ------------------------------------------------------------------ launch spec


@dataclass
class LaunchSpec:
    """Everything the notebook needs that is only known at launch time."""

    account: str
    kernel_ref: str
    relay_url: str
    token: str
    # The served model name, read out of the source notebook so the pool can report it
    # before any kernel registers.
    model: str
    max_runtime_seconds: float
    shutdown_poll_seconds: float
    orphan_grace_seconds: float = 900.0
    status_path: str = "/kaggle/working/kaggle-rotate"


def _fill(template: str, values: dict[str, Any]) -> str:
    out = template
    for key, value in values.items():
        out = out.replace(f"@{key}@", str(value))
    return out


# ------------------------------------------------------------------- cell bodies

PREAMBLE = '''\
import json as _json
import os as _os
import re as _re
import shutil as _shutil
import subprocess as _subprocess
import time as _time
import urllib.error as _urllib_error
import urllib.request as _urllib_request
from pathlib import Path as _Path

def _uptime():
    """Seconds since boot INCLUDING time spent suspended.

    CLOCK_MONOTONIC stops while the machine is suspended, so a monotonic orphan timer
    never fires after a suspend or hibernate and the kernel runs to its hard cap
    instead. CLOCK_BOOTTIME keeps counting.
    """
    try:
        return _time.clock_gettime(_time.CLOCK_BOOTTIME)
    except (AttributeError, ValueError, OSError):
        return _time.monotonic()

def sh(cmd, env=None):
    """Run a shell command the way a `!`-prefixed notebook line would."""
    print(f"$ {cmd}", flush=True)
    return _subprocess.run(
        cmd,
        shell=True,
        env=_os.environ if env is None else env,
        check=False,
    )

def _post(url, payload, timeout=20):
    request = _urllib_request.Request(
        url,
        data=_json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer @TOKEN@",
        },
        method="POST",
    )
    with _urllib_request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return _json.loads(body) if body.strip() else {}

def _get(url, timeout=20):
    request = _urllib_request.Request(
        url,
        headers={"Authorization": f"Bearer @TOKEN@"},
    )
    with _urllib_request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return _json.loads(body) if body.strip() else {}
'''

TUNNEL_CELL = """\
_MODEL_URL = "@MODEL_URL@"

if not _shutil.which("cloudflared"):
    sh(
        "curl -fsSL -o /tmp/cloudflared.deb "
        "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        "cloudflared-linux-amd64.deb && dpkg -i /tmp/cloudflared.deb",
        env=_os.environ,
    )

if not _shutil.which("cloudflared"):
    raise RuntimeError("cloudflared install failed")

import queue as _queue
import threading as _threading

def _pipe(process, sink):
    for line in iter(process.stdout.readline, ""):
        sink.put(line)

_tunnel_out = _queue.Queue()
_tunnel_process = _subprocess.Popen(
    [
        "cloudflared", "tunnel",
        "--no-autoupdate",
        "--protocol", "http2",
        "--edge-ip-version", "4",
        "--url", _MODEL_URL,
        "--http-host-header", "@HOST_HEADER@",
    ],
    stdout=_subprocess.PIPE,
    stderr=_subprocess.STDOUT,
    text=True,
    bufsize=1,
)
_threading.Thread(
    target=_pipe, args=(_tunnel_process, _tunnel_out), daemon=True
).start()

_deadline = _uptime() + 90
PUBLIC_OLLAMA_URL = None
while _uptime() < _deadline:
    if _tunnel_process.poll() is not None and _tunnel_out.empty():
        break
    try:
        line = _tunnel_out.get(timeout=1)
    except _queue.Empty:
        continue
    match = _re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", line)
    if match:
        PUBLIC_OLLAMA_URL = match.group(0)
        break
    print(line.rstrip(), flush=True)

if not PUBLIC_OLLAMA_URL:
    raise RuntimeError("Cloudflare quick tunnel did not produce a URL")

print("Public model endpoint:", PUBLIC_OLLAMA_URL + "/v1", flush=True)
"""

READY_CELL = """\
MODEL_NAME = "@MODEL@"
SERVER_LOG = "/tmp/bonsai-llama-server.log"

def _gpu_report():
    output = _subprocess.run(
        "nvidia-smi --query-gpu=index,name,memory.total,memory.used "
        "--format=csv,noheader",
        shell=True, capture_output=True, text=True, check=False,
    ).stdout.strip()
    gpus = [line for line in output.splitlines() if line.strip()]
    for line in gpus:
        print("  GPU:", line, flush=True)
    if not gpus:
        print("  WARNING: nvidia-smi reported no GPUs", flush=True)
    return gpus

# llama-server was launched by the source notebook and only serves /health once the
# model is resident, so this is the readiness gate the pool has been waiting on. It
# does NOT reload the model: the notebook already paid for the load, and a second load
# would double VRAM.
_gpus = _gpu_report()

_deadline = _uptime() + 120
while _uptime() < _deadline:
    try:
        with _urllib_request.urlopen(f"{_MODEL_URL}@HEALTH_PATH@", timeout=5) as _r:
            if _r.status == 200:
                break
    except Exception:
        pass
    _time.sleep(2)
else:
    _subprocess.run(f"tail -n 60 {SERVER_LOG} || true", shell=True, check=False)
    raise RuntimeError(
        f"llama-server never became healthy on {_MODEL_URL} "
        f"with {len(_gpus)} GPU(s); see {SERVER_LOG}"
    )

try:
    with _urllib_request.urlopen(f"{_MODEL_URL}/v1/models", timeout=15) as _r:
        _served = [_json.loads(_r.read().decode("utf-8"))]
    _ids = [e.get("id") for e in (_served[0].get("data") or [])]
    print("  llama-server models:", _ids or "(none reported)", flush=True)
except Exception as _exc:
    print(f"  could not list /v1/models: {_exc}", flush=True)

print(f"Model resident: {MODEL_NAME} on {len(_gpus)} GPU(s)", flush=True)
"""


REGISTER_CELL = """\
RELAY_URL = "@RELAY_URL@"
SESSION_STARTED_AT = _time.time()
STATUS_DIR = _Path("@STATUS_PATH@")
STATUS_DIR.mkdir(parents=True, exist_ok=True)

def _gpu_summary():
    try:
        output = _subprocess.run(
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
            shell=True, capture_output=True, text=True, check=False,
        ).stdout.strip()
    except Exception:
        output = ""
    return output

def _write_status(extra=None):
    status = {
        "account": "@ACCOUNT@",
        "kernel_ref": "@KERNEL_REF@",
        "url": PUBLIC_OLLAMA_URL,
        "model": MODEL_NAME,
        "gpu": _gpu_summary(),
        "started_at": SESSION_STARTED_AT,
        "updated_at": _time.time(),
        "pid": _os.getpid(),
    }
    status.update(extra or {})
    tmp = STATUS_DIR / "status.json.tmp"
    tmp.write_text(_json.dumps(status, indent=2))
    tmp.replace(STATUS_DIR / "status.json")
    return status

def _register():
    _post(
        f"{RELAY_URL}/_rot/register",
        {
            "token": "@TOKEN@",
            "account": "@ACCOUNT@",
            "kernel_ref": "@KERNEL_REF@",
            "url": PUBLIC_OLLAMA_URL,
            "model": MODEL_NAME,
            "gpu": _gpu_summary(),
            "started_at": SESSION_STARTED_AT,
        },
        timeout=30,
    )
    print("Registered with relay:", RELAY_URL, flush=True)

_write_status({"phase": "booting"})
_register()
"""

SUPERVISOR_CELL = """\
RELAY_URL = "@RELAY_URL@"
MAX_RUNTIME_S = @MAX_RUNTIME@
POLL_S = @POLL_S@
ORPHAN_GRACE_S = @ORPHAN_GRACE@

_stop_reason = None
_last_relay_ok = _uptime()
_driver_lost_at = None
_last_heartbeat = 0.0
_deadline = SESSION_STARTED_AT + MAX_RUNTIME_S

def _model_alive():
    try:
        _urllib_request.urlopen(f"{_MODEL_URL}@HEALTH_PATH@", timeout=5).close()
        return True
    except Exception:
        return False

while _stop_reason is None:
    if _time.time() >= _deadline:
        _stop_reason = "session self-deadline reached"
        break

    # One relay round trip per poll doubles as the keepalive that stops the
    # Cloudflare quick tunnel from being reaped, and as the retire channel.
    try:
        command = _post(
            f"{RELAY_URL}/_rot/heartbeat",
            {
                "token": "@TOKEN@",
                "account": "@ACCOUNT@",
                "url": PUBLIC_OLLAMA_URL,
                "model": MODEL_NAME,
                "model_alive": _model_alive(),
                "elapsed_s": _time.time() - SESSION_STARTED_AT,
                "started_at": SESSION_STARTED_AT,
            },
            timeout=20,
        )
        _last_heartbeat = _uptime()
        _last_relay_ok = _last_heartbeat
        _write_status({"phase": "serving", "model_alive": _model_alive()})
        action = command.get("action", "keepalive")
        if action == "shutdown":
            _stop_reason = command.get("reason", "shutdown requested by pool")
            break

        # The relay can outlive the pool (an orphaned cloudflared child keeps
        # answering). It is reachable but nobody is driving it, so nobody can rotate or
        # retire this session - treat that exactly like a lost relay.
        if not command.get("driver_alive", True):
            if _driver_lost_at is None:
                _driver_lost_at = _uptime()
                print(
                    "relay is answering but the local pool is gone "
                    f"(silent {command.get('driver_silent_seconds', '?')}s); "
                    f"self-terminating in {int(ORPHAN_GRACE_S)}s unless it returns",
                    flush=True,
                )
            if _uptime() - _driver_lost_at > ORPHAN_GRACE_S:
                _stop_reason = (
                    "local pool disappeared while the relay stayed up; "
                    "shutting down so the GPU is not billed unattended"
                )
                break
        else:
            _driver_lost_at = None
    except Exception as exc:
        print(f"relay heartbeat failed: {exc}", flush=True)
        _write_status({"phase": "serving", "relay_error": str(exc)})

    if _uptime() - _last_relay_ok > ORPHAN_GRACE_S:
        _stop_reason = f"relay unreachable for {int(ORPHAN_GRACE_S)}s; shutting down"
        break

    _time.sleep(POLL_S)

print("Stopping:", _stop_reason, flush=True)
_write_status({"phase": "stopping", "stop_reason": _stop_reason})
"""

TEARDOWN_CELL = """\
for _name, _process in (
    ("cloudflared", _tunnel_process),
    ("llama-server", globals().get("llama_server_process")),
):
    if _process is None:
        continue
    try:
        _process.terminate()
        _process.wait(timeout=20)
        print(f"stopped {_name}", flush=True)
    except Exception as exc:
        print(f"failed to stop {_name}: {exc}", flush=True)
        try:
            _process.kill()
        except Exception:
            pass

try:
    _post(
        f"{RELAY_URL}/_rot/final",
        {
            "token": "@TOKEN@",
            "account": "@ACCOUNT@",
            "stop_reason": _stop_reason,
            "ended_at": _time.time(),
        },
        timeout=15,
    )
except Exception as exc:
    print(f"final relay call failed (session ended anyway): {exc}", flush=True)

_write_status({"phase": "stopped", "stop_reason": _stop_reason})
print(f"Session for @ACCOUNT@ ended: {_stop_reason}", flush=True)
"""


def _code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": source.splitlines(True),
    }


def _markdown(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def _replace(template: str, **values: Any) -> str:
    return _fill(template, {k: str(v) for k, v in values.items()})


def build_notebook(config: Config, spec: LaunchSpec) -> dict[str, Any]:
    source_path = config.resolved_source_notebook()
    served = read_served_model(source_path)
    setup_cells: list[dict[str, Any]] = []
    if source_path.exists():
        setup_cells = extract_setup_cells(json.loads(source_path.read_text()))

    values = dict(
        TOKEN=spec.token,
        ACCOUNT=spec.account,
        KERNEL_REF=spec.kernel_ref,
        RELAY_URL=spec.relay_url,
        MODEL=served.model,
        MODEL_URL=MODEL_BASE_URL,
        HOST_HEADER=HOST_HEADER,
        HEALTH_PATH=HEALTH_PATH,
        STATUS_PATH=spec.status_path,
        MAX_RUNTIME=int(spec.max_runtime_seconds),
        POLL_S=int(spec.shutdown_poll_seconds),
        ORPHAN_GRACE=int(spec.orphan_grace_seconds),
    )

    header = (
        f"# Managed by kaggle-rotate (account `{spec.account}`)\n\n"
        f"Generated from `{source_path.name}`. Do not edit this kernel from the Kaggle UI: "
        "the local orchestrator re-renders it on every launch.\n\n"
        f"- relay: `{spec.relay_url}`\n"
        f"- model: `{served.model}`\n"
        f"- context: `{served.num_ctx}` (from the source notebook)\n"
        f"- runtime: PrismML `llama-server` on `{MODEL_BASE_URL}`\n"
        f"- self-terminate after: `{spec.max_runtime_seconds / 3600:.2f}h`\n"
    )

    cells: list[dict[str, Any]] = [_markdown(header), _code(_replace(PREAMBLE, **values))]
    cells += [
        {**cell, "source": _fill("".join(cell["source"]), values).splitlines(keepends=True)}
        for cell in setup_cells
    ]
    cells += [
        _markdown("## Expose the model (Cloudflare quick tunnel)"),
        _code(_replace(TUNNEL_CELL, **values)),
        _markdown("## Confirm llama-server is healthy before advertising readiness"),
        _code(_replace(READY_CELL, **values)),
        _markdown("## Register with the local control relay"),
        _code(_replace(REGISTER_CELL, **values)),
        _markdown("## Supervise until the pool retires this session"),
        _code(_replace(SUPERVISOR_CELL, **values)),
        _markdown("## Teardown"),
        _code(_replace(TEARDOWN_CELL, **values)),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def kernel_metadata(config: Config, spec: LaunchSpec) -> dict[str, Any]:
    """Build kernel-metadata.json.

    Kaggle derives the kernel slug from the title, and `id` must match it or the push
    is rejected. The title is therefore forced to agree with `kernel_slug`.
    """
    slug = config.kernel.kernel_slug
    title = config.kernel.title.strip()
    if not title:
        title = slug
    elif slugify(title) != slug:
        raise ValueError(
            f"kernel.title {title!r} slugifies to {slugify(title)!r}, but kernel.kernel_slug is "
            f"{slug!r}; Kaggle builds the kernel id from the title, so they must match"
        )

    return {
        "id": spec.kernel_ref,
        "title": title,
        "code_file": "server.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true" if config.kernel.is_private else "false",
        "enable_gpu": "true",
        "machine_shape": config.kernel.accelerator,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }


def write_kernel(config: Config, spec: LaunchSpec, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook(config, spec)
    # Record what was actually rendered, not what the caller believed: if the notebook's
    # declared model ever drifts from the pool's view, the sidecar shows the truth.
    rendered_model = read_served_model(config.resolved_source_notebook()).model
    (directory / "server.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    (directory / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata(config, spec), indent=2) + "\n"
    )
    # Sidecar describing the launch: the only place the per-launch token and relay URL
    # are recorded outside the notebook itself. Useful when a boot misbehaves.
    (directory / "launch.json").write_text(
        json.dumps(
            {
                "account": spec.account,
                "kernel_ref": spec.kernel_ref,
                "relay_url": spec.relay_url,
                "token": spec.token,
                "model": rendered_model,
                "max_runtime_seconds": spec.max_runtime_seconds,
                "status_dir": spec.status_path,
            },
            indent=2,
        )
        + "\n"
    )
    os.chmod(directory / "launch.json", 0o600)
    return directory / "server.ipynb"
