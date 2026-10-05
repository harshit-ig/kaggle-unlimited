"""Builds the Kaggle-side server notebook from the user's source notebook.

The source notebook's setup cells (Ollama install, model pull, MTP definition,
warm-up load) are reused so the server inherits whatever the user already tuned.
Everything after the tunnel section is dropped and replaced with:

  * a Cloudflare quick tunnel for the model,
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


@dataclass
class SourceModel:
    source_model: str = ""
    model: str = ""
    num_ctx: int = 65536
    draft_num_predict: int = 2


def _assignment(source: str, name: str) -> str | None:
    match = re.search(_ASSIGN_RE.format(name=re.escape(name)), source, re.MULTILINE)
    return match.group("value").strip().strip("\"'") if match else None


def load_source_config(path: Path) -> SourceModel:
    """Pull model/ctx/draft settings out of the user's notebook."""
    if not path.exists():
        return SourceModel()
    notebook = json.loads(path.read_text())
    blob = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))

    def as_int(value: str | None, fallback: int) -> int:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return fallback

    num_ctx = as_int(_assignment(blob, "NUM_CTX"), 65536)
    draft = as_int(_assignment(blob, "DRAFT_TOKENS"), 2)
    return SourceModel(
        source_model=_assignment(blob, "SOURCE_MODEL") or "",
        model=_assignment(blob, "MTP_MODEL") or "",
        num_ctx=num_ctx,
        draft_num_predict=draft,
    )


def resolve_model_config(config: Config) -> SourceModel:
    source = load_source_config(config.resolved_source_notebook())
    if config.kernel.source_model:
        source.source_model = config.kernel.source_model
    if config.kernel.model:
        source.model = config.kernel.model
    source.num_ctx = config.kernel.num_ctx or source.num_ctx
    source.draft_num_predict = config.kernel.draft_num_predict or source.draft_num_predict
    return source


# ------------------------------------------------------------------- transforms

# Markdown headers whose section is rebuilt (or skipped) by kaggle-rotate. Dropping the
# header also drops the code cell that follows it, which is how the expensive demo
# cells get out of the boot path.
DROP_MARKDOWN_MARKERS = (
    "Benchmark generation speed",
    "Load the base 27B model",
    "Optional: expose Ollama",
    "Install Cloudflare Tunnel",
    "Test the Ollama API",
    "Test through the public tunnel",
    "OpenAI-compatible client settings",
    "Interactive Ollama CLI",
)

# Code cells after these are hard-cut: everything from here on is exposure/UI code.
CUTOFF_CODE_MARKERS = ("cloudflared", "PUBLIC_OLLAMA_URL")

_SHELL_LINE = re.compile(r"^\s*!(?P<cmd>.+)$")
_BIN_VAR = "{OLLAMA_BIN}"


def _rewrite_shell_cell(source: str) -> str:
    """Turn `!cmd` notebook lines into subprocess calls.

    `!{OLLAMA_BIN} list` is a Python syntax error inside a notebook kernel, and naive
    interpolation would also break on any command containing quotes. Splitting the
    shell line on the OLLAMA_BIN marker and repr()ing each piece sidesteps both.
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
        command = match.group("cmd").strip()
        exprs: list[str] = []
        for index, part in enumerate(command.split(_BIN_VAR)):
            if index:
                exprs.append("OLLAMA_BIN")
            if part:
                exprs.append(repr(part))
        rewritten.append(f"sh({' + '.join(exprs) or repr('')})")
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
        rewritten["source"] = _rewrite_shell_cell(source).splitlines(keepends=True)
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
    model: str
    source_model: str
    num_ctx: int
    draft_num_predict: int
    max_runtime_seconds: float
    shutdown_poll_seconds: float
    orphan_grace_seconds: float = 900.0
    status_path: str = "/kaggle/working/kaggle-rotate"


def _fill(template: str, values: dict[str, str]) -> str:
    out = template
    for key, value in values.items():
        out = out.replace(f"@{key}@", value)
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
_OLLAMA_URL = "http://127.0.0.1:11434"

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
        "--url", _OLLAMA_URL,
        "--http-host-header", "localhost:11434",
    ],
    stdout=_subprocess.PIPE,
    stderr=_subprocess.STDOUT,
    text=True,
    bufsize=1,
)
_threading.Thread(
    target=_pipe, args=(_tunnel_process, _tunnel_out), daemon=True
).start()

_deadline = _time.monotonic() + 90
PUBLIC_OLLAMA_URL = None
while _time.monotonic() < _deadline:
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

PREWARM_CELL = """\
MODEL_NAME = "@MODEL@"
REQUESTED_NUM_CTX = @NUM_CTX@

def _gpu_report():
    output = _subprocess.run(
        "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader",
        shell=True, capture_output=True, text=True, check=False,
    ).stdout.strip()
    gpus = [line for line in output.splitlines() if line.strip()]
    for line in gpus:
        print("  GPU:", line, flush=True)
    if len(gpus) < 2:
        print(
            f"  WARNING: expected 2 GPUs for this model, found {len(gpus)}. A 27B Q4 "
            "needs ~17GB of weights plus KV cache, so a single T4 will not fit.",
            flush=True,
        )
    return gpus

def _ollama_json(path, payload=None, timeout=30):
    url = f"{_OLLAMA_URL}{path}"
    if payload is None:
        request = _urllib_request.Request(url)
    else:
        request = _urllib_request.Request(
            url,
            data=_json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    with _urllib_request.urlopen(request, timeout=timeout) as response:
        return _json.loads(response.read().decode("utf-8"))

def _model_loaded(name):
    try:
        running = _ollama_json("/api/ps")
    except Exception:
        return False
    return any(entry.get("name", "").startswith(name) for entry in running.get("models", []))

_gpus = _gpu_report()

# Try the configured context, then shrink. Ollama reports an out-of-memory load as a
# bare HTTP 500, which used to kill the session with no explanation while the pool sat
# waiting for a registration that could never arrive.
for _num_ctx in (REQUESTED_NUM_CTX, 32768, 16384, 8192):
    if _num_ctx < REQUESTED_NUM_CTX:
        print(f"Retrying with num_ctx={_num_ctx}", flush=True)
    try:
        _ollama_json(
            "/api/generate",
            {
                "model": MODEL_NAME,
                "prompt": "ready",
                "stream": False,
                "keep_alive": -1,
                "options": {
                    "draft_num_predict": @DRAFT@,
                    "num_ctx": _num_ctx,
                },
            },
            timeout=1800,
        )
    except Exception as exc:
        print(f"load failed at num_ctx={_num_ctx}: {exc}", flush=True)
        continue
    if _model_loaded(MODEL_NAME):
        print(f"Model resident: {MODEL_NAME} (num_ctx={_num_ctx})", flush=True)
        break
else:
    _subprocess.run("tail -n 40 /tmp/ollama-server.log || true", shell=True, check=False)
    raise RuntimeError(
        f"could not load {MODEL_NAME} into VRAM on {len(_gpus)} GPU(s)"
    )
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
_last_relay_ok = _time.monotonic()
_last_heartbeat = 0.0
_deadline = SESSION_STARTED_AT + MAX_RUNTIME_S

def _ollama_alive():
    try:
        _urllib_request.urlopen(f"{_OLLAMA_URL}/api/tags", timeout=5).close()
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
                "ollama_alive": _ollama_alive(),
                "elapsed_s": _time.time() - SESSION_STARTED_AT,
                "started_at": SESSION_STARTED_AT,
            },
            timeout=20,
        )
        _last_heartbeat = _time.monotonic()
        _last_relay_ok = _last_heartbeat
        _write_status({"phase": "serving", "ollama_alive": _ollama_alive()})
        action = command.get("action", "keepalive")
        if action == "shutdown":
            _stop_reason = command.get("reason", "shutdown requested by pool")
            break
    except Exception as exc:
        print(f"relay heartbeat failed: {exc}", flush=True)
        _write_status({"phase": "serving", "relay_error": str(exc)})

    if _time.monotonic() - _last_relay_ok > ORPHAN_GRACE_S:
        _stop_reason = f"relay unreachable for {int(ORPHAN_GRACE_S)}s; shutting down"
        break

    _time.sleep(POLL_S)

print("Stopping:", _stop_reason, flush=True)
_write_status({"phase": "stopping", "stop_reason": _stop_reason})
"""

TEARDOWN_CELL = """\
for _name, _process in (
    ("cloudflared", _tunnel_process),
    ("ollama", globals().get("ollama_server_process")),
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
    model = resolve_model_config(config)
    if spec.model:
        model.model = spec.model
    if spec.source_model:
        model.source_model = spec.source_model
    model.num_ctx = spec.num_ctx or model.num_ctx
    model.draft_num_predict = spec.draft_num_predict or model.draft_num_predict

    source_path = config.resolved_source_notebook()
    setup_cells: list[dict[str, Any]] = []
    if source_path.exists():
        setup_cells = extract_setup_cells(json.loads(source_path.read_text()))

    values = dict(
        TOKEN=spec.token,
        ACCOUNT=spec.account,
        KERNEL_REF=spec.kernel_ref,
        RELAY_URL=spec.relay_url,
        MODEL=model.model,
        NUM_CTX=model.num_ctx,
        DRAFT=model.draft_num_predict,
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
        f"- model: `{model.model}` (from `{model.source_model}`)\n"
        f"- context: `{model.num_ctx}`\n"
        f"- self-terminate after: `{spec.max_runtime_seconds / 3600:.2f}h`\n"
    )

    cells: list[dict[str, Any]] = [_markdown(header), _code(_replace(PREAMBLE, **values))]
    cells += setup_cells
    cells += [
        _markdown("## Expose the model (Cloudflare quick tunnel)"),
        _code(_replace(TUNNEL_CELL, **values)),
        _markdown("## Make sure the model is resident before advertising readiness"),
        _code(_replace(PREWARM_CELL, **values)),
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
                "model": spec.model,
                "source_model": spec.source_model,
                "num_ctx": spec.num_ctx,
                "draft_num_predict": spec.draft_num_predict,
                "max_runtime_seconds": spec.max_runtime_seconds,
                "status_dir": spec.status_path,
            },
            indent=2,
        )
        + "\n"
    )
    os.chmod(directory / "launch.json", 0o600)
    return directory / "server.ipynb"
