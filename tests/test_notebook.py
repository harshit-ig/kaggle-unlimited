"""The generated Kaggle notebook must be syntactically valid and fully rendered.

The build input is `bonsai2_27b_pq2_0_kaggle_llamacpp.ipynb`, which runs Ternary
Bonsai 2 through the PrismML llama.cpp fork. There is no Ollama anywhere in the stack,
so most of what these tests guard is that it cannot creep back in.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from kaggle_rotate.config import Config
from kaggle_rotate.naming import slugify
from kaggle_rotate.notebook import (
    FALLBACK_MODEL,
    FALLBACK_NUM_CTX,
    FALLBACK_SERVER_LOG,
    MODEL_BASE_URL,
    READY_CELL,
    LaunchSpec,
    _rewrite_shell_cell,
    build_notebook,
    kernel_metadata,
    read_served_model,
)

PROJECT = Path(__file__).resolve().parent.parent
SOURCE_NOTEBOOK = PROJECT / "bonsai2_27b_pq2_0_kaggle_llamacpp.ipynb"
PLACEHOLDER = re.compile(r"@[A-Z_]+@")

# Anything Ollama-specific that must never reach a Kaggle kernel: the model is served
# by llama-server, and these either target a port nothing listens on, hit endpoints
# llama.cpp does not implement, or name a binary that is never installed.
FORBIDDEN = (
    "11434",
    "/api/ps",
    "/api/tags",
    "/api/generate",
    "/api/create",
    "keep_alive",
    "OLLAMA_BIN",
    "ollama pull",
    "ollama_alive",
    "draft_num_predict",
    "hf.co/",
)


def _spec(**overrides) -> LaunchSpec:
    defaults = dict(
        account="acct",
        kernel_ref="acct/kaggle-rotate-llamacpp",
        relay_url="https://relay.example.trycloudflare.com",
        token="tok-123",
        model="ternary-bonsai-2-27b-pq2",
        max_runtime_seconds=39600.0,
        shutdown_poll_seconds=20.0,
    )
    return LaunchSpec(**{**defaults, **overrides})


def _config(source_notebook: Path | str = SOURCE_NOTEBOOK) -> Config:
    config = Config()
    config.kernel.source_notebook = str(source_notebook)
    return config


def _code_cells(notebook: dict) -> list[tuple[int, str]]:
    return [
        (index, "".join(cell["source"]))
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
    ]


def _blob(notebook: dict) -> str:
    return "\n".join(source for _, source in _code_cells(notebook))


def _raw_blob(notebook: dict) -> str:
    """Every cell including markdown, for placeholders and leaked config."""
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


# ── reading the served model off the notebook ────────────────────────────────────


def test_served_model_is_read_from_the_real_notebook():
    served = read_served_model(SOURCE_NOTEBOOK)
    assert served.model == "ternary-bonsai-2-27b-pq2"
    assert served.num_ctx == 262144
    assert served.server_log == "/tmp/bonsai-llama-server.log"


def test_served_model_falls_back_when_the_notebook_declares_nothing(tmp_path):
    served = read_served_model(tmp_path / "missing.ipynb")
    assert served.model == FALLBACK_MODEL
    assert served.num_ctx == FALLBACK_NUM_CTX
    assert served.server_log == FALLBACK_SERVER_LOG


def test_server_log_follows_the_notebook_not_the_previous_model(tmp_path):
    """A notebook serving a different model must not tail the old model's log."""
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": ['SERVER_LOG = "/tmp/qwen38-llama-server.log"\n'],
            }
        ]
    }
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(notebook))
    assert read_served_model(path).server_log == "/tmp/qwen38-llama-server.log"


def test_readiness_gate_tails_the_declared_log(tmp_path):
    """The boot-time failure message has to point at the log that was actually written."""
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": ['SERVER_LOG = "/tmp/other-server.log"\n'],
            }
        ]
    }
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(notebook))
    blob = _raw_blob(build_notebook(_config(path), _spec()))
    assert 'SERVER_LOG = "/tmp/other-server.log"' in blob
    assert "/tmp/bonsai-llama-server.log" not in blob


def test_served_model_ignores_an_unparseable_context(tmp_path):
    notebook = {"cells": [{"cell_type": "code", "source": ['MTP_MODEL = "m"\nNUM_CTX = "big"\n']}]}
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(notebook))
    served = read_served_model(path)
    assert served.model == "m"
    assert served.num_ctx == FALLBACK_NUM_CTX


def test_generated_notebook_reports_the_notebook_owned_context():
    notebook = build_notebook(_config(), _spec())
    assert "262144" in _raw_blob(notebook), "the header must show the real context width"


# ── the Ollama path must not come back ───────────────────────────────────────────


@pytest.mark.parametrize("needle", FORBIDDEN)
def test_no_ollama_residue_reaches_the_kernel(needle: str):
    blob = _raw_blob(build_notebook(_config(), _spec()))
    assert needle not in blob, f"Ollama residue {needle!r} is still in the generated notebook"


@pytest.mark.parametrize("needle", ("PUBLIC_OLLAMA_URL",))
def test_legacy_url_identifier_is_only_ever_a_name(needle: str):
    """`PUBLIC_OLLAMA_URL` survives as an identifier; it must never be a URL."""
    notebook = build_notebook(_config(), _spec())
    for index, source in _code_cells(notebook):
        for line in source.splitlines():
            if needle in line:
                assert "11434" not in line, f"cell {index} points {needle} at Ollama's port"


def test_source_notebook_has_no_ollama_cells():
    """The build input itself must be clean, not just the render."""
    notebook = json.loads(SOURCE_NOTEBOOK.read_text())
    blob = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for needle in ("OLLAMA_BIN", "11434", "/api/ps", "/api/tags"):
        assert needle not in blob, f"{needle} is still in {SOURCE_NOTEBOOK.name}"


def test_llama_server_wiring_is_present():
    blob = _blob(build_notebook(_config(), _spec()))
    assert f'_MODEL_URL = "{MODEL_BASE_URL}"' in blob, "tunnel must target llama-server"
    assert "localhost:8080" in blob, "cloudflared must forward the right Host header"
    assert blob.count("/health") >= 2, "readiness gate and supervisor both health-check"
    assert "/v1/models" in blob, "must confirm the OpenAI-compatible surface is up"
    assert "llama_server_process" in blob, "teardown must stop the server we started"


def test_tunnel_targets_the_port_the_notebook_serves():
    """A port mismatch between the tunnel and llama-server is silent: the tunnel comes
    up and every request 502s, so the two have to be asserted together."""
    notebook = json.loads(SOURCE_NOTEBOOK.read_text())
    blob = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert '"PORT": "8080"' in blob, "the notebook must still bind 8080"
    assert MODEL_BASE_URL.endswith(":8080")
    rendered = _blob(build_notebook(_config(), _spec()))
    assert MODEL_BASE_URL in rendered
    assert "localhost:8080" in rendered


# ── structural guarantees ────────────────────────────────────────────────────────


def test_generated_notebook_has_no_unrendered_placeholders():
    notebook = build_notebook(_config(), _spec())
    leftover = PLACEHOLDER.search(_raw_blob(notebook))
    assert leftover is None, f"unrendered {leftover.group(0) if leftover else ''}"
    assert "REPLACE-ME" not in json.dumps(notebook)


def test_every_code_cell_parses():
    for index, source in _code_cells(build_notebook(_config(), _spec())):
        try:
            ast.parse(source)
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"cell {index} does not parse: {exc}\n{source[:400]}")


def test_no_shell_escape_cells_survive():
    """`!cmd` is invalid Python inside a kernel and would abort the push."""
    for index, source in _code_cells(build_notebook(_config(), _spec())):
        for line in source.splitlines():
            assert not line.lstrip().startswith("!"), f"cell {index} kept a shell line: {line}"


def test_expensive_benchmark_cells_are_dropped():
    """A pool kernel must not spend its boot budget measuring throughput."""
    notebook = build_notebook(_config(), _spec())
    raw = _raw_blob(notebook)
    assert "Raw llama.cpp benchmark" not in raw, "benchmark section header survived"
    assert "API benchmark" not in raw, "API benchmark section header survived"
    # The `llama-bench` invocation itself (not the binary lookup in the config cell).
    assert '"-p", "512"' not in raw
    assert "LLAMA_BENCH_BIN,\n" not in raw


def test_setup_cells_the_boot_depends_on_are_kept():
    blob = _raw_blob(build_notebook(_config(), _spec()))
    for needed in (
        "PrismML-Eng/Bonsai-demo",  # clones the runtime
        "hf_hub_download",  # fetches the PQ2_0 GGUF
        "Ternary-Bonsai-2-27B-PQ2_0.gguf",  # the weights
        "start_llama_server.sh",  # launches the server
    ):
        assert needed in blob, f"boot path lost {needed}"


def test_control_plane_cells_are_present():
    blob = _blob(build_notebook(_config(), _spec()))
    for marker in ("/_rot/register", "/_rot/heartbeat", "/_rot/final", "cloudflared"):
        assert marker in blob, f"missing {marker}"


def _rendered_ready_cell() -> str:
    ready = READY_CELL.replace("@MODEL@", "m").replace("@MODEL_URL@", MODEL_BASE_URL)
    return ready.replace("@HEALTH_PATH@", "/health")


def test_readiness_gate_does_not_reload_the_model():
    """The source notebook already loaded the model; a second load would double VRAM
    and could OOM a T4 that was about to come up fine."""
    ready = _rendered_ready_cell()
    ast.parse(ready)
    assert "/v1/chat/completions" not in ready, "the gate must not issue a completion"
    assert "/v1/models" in ready, "listing models proves the API surface without generating"
    assert "poll()" not in ready, "the gate must not try to restart or manage the server"


def test_ready_cell_reports_the_server_log_on_failure():
    ready = _rendered_ready_cell()
    assert "tail -n 60" in ready, "a failed boot must point at the server log"
    assert "raise RuntimeError" in ready
    # The endpoint comes from the shared constant, not a hardcoded path in each cell.
    assert "@HEALTH_PATH@" not in ready


def test_supervisor_health_checks_llama_server_not_ollama():
    blob = _blob(build_notebook(_config(), _spec()))
    supervisor = blob.split("def _model_alive")[-1]
    assert "/health" in supervisor
    # `\b` matters: `PUBLIC_OLLAMA_URL` legitimately ends in `_OLLAMA_URL`.
    assert not re.search(r"\b_OLLAMA_URL\b", blob), "the Ollama base-URL alias is still referenced"


def test_relay_url_and_token_are_per_launch():
    first = build_notebook(_config(), _spec(token="alpha", relay_url="https://a.example"))
    second = build_notebook(_config(), _spec(token="beta", relay_url="https://b.example"))
    blob_one = json.dumps(first)
    blob_two = json.dumps(second)
    assert "alpha" in blob_one and "beta" not in blob_one
    assert "beta" in blob_two and "alpha" not in blob_two
    assert "https://a.example" in blob_one and "https://b.example" in blob_two


def test_model_comes_from_the_notebook_not_the_spec():
    """A stale LaunchSpec.model must not win: the notebook is the source of truth."""
    notebook = build_notebook(_config(), _spec(model="something-else-entirely"))
    blob = _blob(notebook)
    assert 'MODEL_NAME = "ternary-bonsai-2-27b-pq2"' in blob
    assert "something-else-entirely" not in blob


# ── shell rewriting ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("!nvidia-smi", "sh('nvidia-smi')"),
        ("!ps aux | grep llama-server", "sh('ps aux | grep llama-server')"),
        ('!echo "hi there"', "sh('echo \"hi there\"')"),
    ],
)
def test_shell_rewrite_handles_quotes(source: str, expected: str):
    assert _rewrite_shell_cell(source) == expected


def test_shell_rewrite_leaves_plain_python_alone():
    source = "import os\nprint(os.getcwd())"
    assert _rewrite_shell_cell(source) == source


# ── kernel metadata / sidecar ────────────────────────────────────────────────────


def test_kernel_metadata_targets_the_t4_and_stays_private():
    metadata = kernel_metadata(_config(), _spec())
    assert metadata["id"] == "acct/kaggle-rotate-llamacpp"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["enable_gpu"] == "true"
    assert metadata["is_private"] == "true"
    assert metadata["code_file"] == "server.ipynb"


def test_missing_source_notebook_does_not_crash_rendering(tmp_path):
    config = Config()
    config.kernel.source_notebook = str(tmp_path / "nope.ipynb")
    notebook = build_notebook(config, _spec())
    assert any("cloudflared" in source for _, source in _code_cells(notebook))


def test_kernel_title_must_agree_with_the_kernel_slug():
    """Kaggle derives the kernel id from the title; a mismatch breaks every push."""
    config = _config()
    config.kernel.title = "My Nice Llama Server"
    with pytest.raises(ValueError, match="slugifies"):
        kernel_metadata(config, _spec())

    config.kernel.title = ""
    assert kernel_metadata(config, _spec())["title"] == config.kernel.kernel_slug

    config.kernel.title = "Kaggle Rotate LlamaCPP"
    config.kernel.kernel_slug = "kaggle-rotate-llamacpp"
    assert kernel_metadata(config, _spec())["title"] == "Kaggle Rotate LlamaCPP"


def test_metadata_id_and_title_slug_always_agree():
    """Regression: `kernel_slug` used to be stored per account AND in config.

    `kernel-metadata.json`'s `id` came from the account copy while `title` came from
    config, so renaming only one produced a kernel whose id did not match the slug
    Kaggle derives from its title -- and the push was rejected with nothing pointing at
    the cause. The slug is now single-sourced through Config.kernel_ref().
    """
    config = _config()
    ref = config.kernel_ref("harshitig")
    metadata = kernel_metadata(config, _spec(kernel_ref=ref))
    assert metadata["id"] == "harshitig/kaggle-rotate-llamacpp"
    assert metadata["id"].split("/")[-1] == slugify(metadata["title"])


def test_heartbeat_field_matches_what_the_relay_reads():
    """Regression: the notebook was renamed to send `model_alive` while the relay still
    read `ollama_alive`, so every session was reported healthy no matter what the kernel
    said about its own server. The two sides are separate strings, so assert they match."""
    from kaggle_rotate import relay as relay_module

    supervisor = _blob(build_notebook(_config(), _spec())).split("def _model_alive")[-1]
    sent = set(re.findall(r'"(\w+_alive)":', supervisor))
    assert sent == {"model_alive"}, f"notebook sends {sent}"

    source = Path(relay_module.__file__).read_text()
    assert "session.model_alive = bool(" in source, "relay must read the field it receives"
    assert 'payload.get("model_alive"' in source, "relay must accept model_alive"
    # The legacy key stays readable so a kernel rendered before the switch is not
    # reported healthy just because we renamed the field.
    assert 'payload.get("ollama_alive"' in source, "legacy key should remain a fallback"


def test_account_no_longer_carries_its_own_kernel_slug():
    """Two sources of truth is the bug; assert the duplicate is gone, not just unused."""
    from kaggle_rotate.accounts import Account

    assert not hasattr(Account(slug="a", username="u"), "kernel_slug")


def test_stored_kernel_slug_in_accounts_json_is_ignored():
    """An existing accounts.json still carrying the old key must load, not explode."""
    from kaggle_rotate.accounts import AccountStore

    path = Path(__file__).parent / "_tmp_accounts.json"
    path.write_text(
        '{"a": {"slug": "a", "username": "u", "kind": "kaggle_json",'
        ' "kernel_slug": "kaggle-rotate-ollama"}}'
    )
    try:
        store = AccountStore(Config())
        store.registry_path = path
        loaded = store.load()
        assert loaded["a"].username == "u"
        assert not hasattr(loaded["a"], "kernel_slug")
    finally:
        path.unlink(missing_ok=True)


def test_launch_sidecar_records_the_credentials():
    from kaggle_rotate.notebook import write_kernel

    tmp = Path(__file__).parent / "_tmp_launch"
    try:
        write_kernel(_config(), _spec(token="secret-token"), tmp)
        sidecar = json.loads((tmp / "launch.json").read_text())
        assert sidecar["token"] == "secret-token"
        assert sidecar["relay_url"] == "https://relay.example.trycloudflare.com"
        assert sidecar["model"] == "ternary-bonsai-2-27b-pq2"
        assert oct((tmp / "launch.json").stat().st_mode)[-3:] == "600"
        assert (tmp / "server.ipynb").exists()
        assert (tmp / "kernel-metadata.json").exists()
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
