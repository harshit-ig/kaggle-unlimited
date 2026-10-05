"""The generated Kaggle notebook must be syntactically valid and fully rendered."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from kaggle_rotate.config import Config
from kaggle_rotate.notebook import (
    LaunchSpec,
    _rewrite_shell_cell,
    build_notebook,
    kernel_metadata,
    load_source_config,
    resolve_model_config,
)

PROJECT = Path(__file__).resolve().parent.parent
SOURCE_NOTEBOOK = PROJECT / "qwen38_27b_kaggle_ollama_mtp_clean_64k.ipynb"
PLACEHOLDER = re.compile(r"@[A-Z_]+@")


def _spec(**overrides) -> LaunchSpec:
    defaults = dict(
        account="acct",
        kernel_ref="acct/kaggle-rotate-ollama",
        relay_url="https://relay.example.trycloudflare.com",
        token="tok-123",
        model="qwen3.8-27b-uncensored-mtp",
        source_model="hf.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF:Q4_K_M",
        num_ctx=65536,
        draft_num_predict=2,
        max_runtime_seconds=39600.0,
        shutdown_poll_seconds=20.0,
    )
    return LaunchSpec(**{**defaults, **overrides})


def _config() -> Config:
    config = Config()
    config.kernel.source_notebook = str(SOURCE_NOTEBOOK)
    return config


def _code_cells(notebook: dict) -> list[tuple[int, str]]:
    return [
        (index, "".join(cell["source"]))
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
    ]


def test_source_config_is_read_from_the_real_notebook():
    source = load_source_config(SOURCE_NOTEBOOK)
    assert source.source_model == "hf.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF:Q4_K_M"
    assert source.model == "qwen3.8-27b-uncensored-mtp"
    assert source.num_ctx == 65536
    assert source.draft_num_predict == 2


def test_resolve_model_config_prefers_explicit_overrides():
    config = _config()
    config.kernel.num_ctx = 8192
    config.kernel.model = "custom-model"
    resolved = resolve_model_config(config)
    assert resolved.model == "custom-model"
    assert resolved.num_ctx == 8192
    assert resolved.source_model.startswith("hf.co/")


def test_generated_notebook_has_no_unrendered_placeholders():
    notebook = build_notebook(_config(), _spec())
    for index, source in _code_cells(notebook):
        leftover = PLACEHOLDER.search(source)
        assert leftover is None, f"cell {index} still has {leftover.group(0)}"
    assert "REPLACE-ME" not in json.dumps(notebook)


def test_every_code_cell_parses():
    notebook = build_notebook(_config(), _spec())
    for index, source in _code_cells(notebook):
        try:
            ast.parse(source)
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"cell {index} does not parse: {exc}\n{source[:400]}")


def test_no_shell_escape_cells_survive():
    """`!cmd` is invalid Python inside a kernel and would abort the push."""
    notebook = build_notebook(_config(), _spec())
    for index, source in _code_cells(notebook):
        for line in source.splitlines():
            assert not line.lstrip().startswith("!"), f"cell {index} kept a shell line: {line}"


def test_expensive_demo_cells_are_dropped():
    notebook = build_notebook(_config(), _spec())
    blob = "\n".join(source for _, source in _code_cells(notebook))
    # The 200-word benchmark and the raw-base-model load are replaced by a
    # residency check that targets the model clients actually call.
    assert "Explain in about 200 words" not in blob
    assert "/api/ps" in blob
    assert "keep_alive" in blob


def test_control_plane_cells_are_present():
    notebook = build_notebook(_config(), _spec())
    blob = "\n".join(source for _, source in _code_cells(notebook))
    for marker in ("/_rot/register", "/_rot/heartbeat", "/_rot/final", "cloudflared"):
        assert marker in blob, f"missing {marker}"


def test_relay_url_and_token_are_per_launch():
    first = build_notebook(_config(), _spec(token="alpha", relay_url="https://a.example"))
    second = build_notebook(_config(), _spec(token="beta", relay_url="https://b.example"))
    blob_one = json.dumps(first)
    blob_two = json.dumps(second)
    assert "alpha" in blob_one and "beta" not in blob_one
    assert "beta" in blob_two and "alpha" not in blob_two
    assert "https://a.example" in blob_one and "https://b.example" in blob_two


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("!nvidia-smi", "sh('nvidia-smi')"),
        ("!{OLLAMA_BIN} list", "sh(OLLAMA_BIN + ' list')"),
        ('!{OLLAMA_BIN} run "hf.co/A:B" "hi"', 'sh(OLLAMA_BIN + \' run "hf.co/A:B" "hi"\')'),
        ("!{OLLAMA_BIN}", "sh(OLLAMA_BIN)"),
    ],
)
def test_shell_rewrite_handles_quotes(source: str, expected: str):
    assert _rewrite_shell_cell(source) == expected


def test_shell_rewrite_leaves_plain_python_alone():
    source = "import os\nprint(os.getcwd())"
    assert _rewrite_shell_cell(source) == source


def test_kernel_metadata_requests_two_t4s_and_stays_private():
    config = _config()
    metadata = kernel_metadata(config, _spec())
    assert metadata["id"] == "acct/kaggle-rotate-ollama"
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
    config.kernel.title = "My Nice Ollama Server"
    with pytest.raises(ValueError, match="slugifies"):
        kernel_metadata(config, _spec())

    config.kernel.title = ""
    assert kernel_metadata(config, _spec())["title"] == config.kernel.kernel_slug

    config.kernel.title = "Kaggle Rotate Ollama"
    config.kernel.kernel_slug = "kaggle-rotate-ollama"
    assert kernel_metadata(config, _spec())["title"] == "Kaggle Rotate Ollama"


def test_launch_sidecar_records_the_credentials():
    from kaggle_rotate.notebook import write_kernel

    tmp = Path(__file__).parent / "_tmp_launch"
    try:
        write_kernel(_config(), _spec(token="secret-token"), tmp)
        sidecar = json.loads((tmp / "launch.json").read_text())
        assert sidecar["token"] == "secret-token"
        assert sidecar["relay_url"] == "https://relay.example.trycloudflare.com"
        assert sidecar["model"] == "qwen3.8-27b-uncensored-mtp"
        assert oct((tmp / "launch.json").stat().st_mode)[-3:] == "600"
        assert (tmp / "server.ipynb").exists()
        assert (tmp / "kernel-metadata.json").exists()
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


def test_prewarm_shrinks_the_context_instead_of_dying_on_oom():
    """Ollama reports an OOM load as a bare HTTP 500, which used to kill the session."""
    from kaggle_rotate.notebook import PREWARM_CELL

    src = PREWARM_CELL.replace("@MODEL@", "m").replace("@NUM_CTX@", "65536").replace("@DRAFT@", "2")
    ast.parse(src)
    assert "for _num_ctx in (REQUESTED_NUM_CTX, 32768, 16384, 8192):" in src
    assert "load failed at num_ctx" in src, "must catch the failure and try smaller"
    assert "_model_loaded(MODEL_NAME)" in src, "must verify residency, not assume it"


def test_prewarm_warns_when_kaggle_gives_fewer_than_two_gpus():
    from kaggle_rotate.notebook import PREWARM_CELL

    assert "expected 2 GPUs" in PREWARM_CELL
    assert "nvidia-smi" in PREWARM_CELL


def test_rendered_notebook_still_parses_after_the_prewarm_change():
    notebook = build_notebook(_config(), _spec())
    for index, source in _code_cells(notebook):
        try:
            ast.parse(source)
        except SyntaxError as exc:  # pragma: no cover
            pytest.fail(f"cell {index}: {exc}")
    blob = "\n".join(source for _, source in _code_cells(notebook))
    assert PLACEHOLDER.search(blob) is None
