"""`cleanup` is the recovery path; it must not report false successes."""

import argparse

from kaggle_rotate import cli as cli_module
from kaggle_rotate.accounts import Account, KaggleError
from kaggle_rotate.config import Config


class FakeCLI:
    def __init__(self, delete_ok: bool, gone: bool) -> None:
        self.delete_ok = delete_ok
        self.gone = gone
        self.deleted: list[str] = []

    def delete(self, account):
        self.deleted.append(account.slug)
        return self.delete_ok

    def status(self, account):
        if self.gone:
            raise KaggleError(
                ["kaggle", "kernels", "status"],
                1,
                "",
                "Cannot access kernel 'a/b' (Permission 'kernels.get' was denied).",
            )
        raise AssertionError("status should not be consulted when delete succeeds")


def _config() -> Config:
    """A real Config: cleanup derives the kernel ref from it, so a stub would not
    notice if the ref stopped agreeing with what a push would send."""
    config = Config()
    config.kernel.kernel_slug = "krotate"
    return config


def _run(fake, tmp_path, monkeypatch, capsys):
    config = _config()
    monkeypatch.setattr(cli_module, "KaggleCLI", lambda *a, **k: fake)
    monkeypatch.setattr(cli_module, "AccountStore", lambda *a, **k: _FakeStore(tmp_path))
    args = argparse.Namespace(dry_run=False)
    code = cli_module.cmd_cleanup(args, config)
    return code, capsys.readouterr().out


class _FakeStore:
    def __init__(self, tmp_path):
        self.root = tmp_path
        self.config_root = tmp_path

    def load(self):
        return {"a": Account(slug="a", username="u")}


def test_cleanup_reports_a_real_delete(monkeypatch, capsys, tmp_path):
    fake = FakeCLI(delete_ok=True, gone=False)
    code, out = _run(fake, tmp_path, monkeypatch, capsys)
    assert code == 0
    assert "deleted" in out
    assert fake.deleted == ["a"]


def test_cleanup_does_not_claim_success_when_the_kernel_is_gone(monkeypatch, capsys, tmp_path):
    """Kaggle 403s on an already-deleted kernel; that is a success, not a failure."""
    fake = FakeCLI(delete_ok=False, gone=True)
    code, out = _run(fake, tmp_path, monkeypatch, capsys)
    assert code == 0, "already gone must not be reported as a failure"
    assert "already gone" in out
    assert "ok a (a/krotate): deleted" not in out


def test_dry_run_changes_nothing(monkeypatch, capsys, tmp_path):
    fake = FakeCLI(delete_ok=True, gone=False)
    config = _config()
    monkeypatch.setattr(cli_module, "KaggleCLI", lambda *a, **k: fake)
    monkeypatch.setattr(cli_module, "AccountStore", lambda *a, **k: _FakeStore(tmp_path))
    cli_module.cmd_cleanup(argparse.Namespace(dry_run=True), config)
    out = capsys.readouterr().out
    assert fake.deleted == [], "dry run must not delete"
    assert "would delete" in out
