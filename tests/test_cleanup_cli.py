"""`cleanup` is the recovery path; it must not report false successes."""

import argparse

from kaggle_rotate import cli as cli_module
from kaggle_rotate.accounts import Account, KaggleError
from kaggle_rotate.config import Config

_REFUSED_403 = "403 Client Error: Forbidden for url: .../DeleteKernel"
_REFUSED_500 = "500 Server Error: Internal Server Error for url: .../DeleteKernel"


class FakeCLI:
    """Stands in for KaggleCLI. `delete` returns the refusal, matching the real
    signature; `lists_kernel` is the confirmation the old `status` probe could not be."""

    def __init__(self, refusal: KaggleError | None, still_listed: bool = False) -> None:
        self.refusal = refusal
        self.still_listed = still_listed
        self.deleted: list[str] = []
        self.listed = 0

    def delete(self, account):
        self.deleted.append(account.slug)
        return self.refusal

    def lists_kernel(self, account):
        self.listed += 1
        return self.still_listed


def _refusal(stderr: str, returncode: int = 1) -> KaggleError:
    return KaggleError(["kaggle", "kernels", "delete"], returncode, "", stderr)


def _gone() -> KaggleError:
    return _refusal("404 Not Found: the kernel does not exist")


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
    # Both streams: a refusal is reported on stderr, which is exactly the case these
    # tests exist to catch, so asserting on stdout alone would pass on the old bug.
    captured = capsys.readouterr()
    return code, captured.out + captured.err


class _FakeStore:
    def __init__(self, tmp_path):
        self.root = tmp_path
        self.config_root = tmp_path

    def load(self):
        return {"a": Account(slug="a", username="u")}

    def save(self, accounts):
        self.saved = accounts


def test_cleanup_reports_a_real_delete(monkeypatch, capsys, tmp_path):
    fake = FakeCLI(refusal=None)
    code, out = _run(fake, tmp_path, monkeypatch, capsys)
    assert code == 0
    assert "deleted" in out
    assert fake.deleted == ["a"]
    assert fake.listed == 0, "a confirmed delete needs no confirmation"


def test_a_definitive_not_found_is_a_success(monkeypatch, capsys, tmp_path):
    fake = FakeCLI(refusal=_gone())
    code, out = _run(fake, tmp_path, monkeypatch, capsys)
    assert code == 0
    assert "already gone" in out


def test_a_refused_delete_still_listed_is_reported_as_a_failure(monkeypatch, capsys, tmp_path):
    """The regression: a 403 on delete used to be re-asked as `kernels status`, which
    answers 403 for the same permission, and that was printed as "already gone" with
    exit 0 while the kernel kept billing. It must now fail loudly."""
    fake = FakeCLI(refusal=_refusal(_REFUSED_403), still_listed=True)
    code, out = _run(fake, tmp_path, monkeypatch, capsys)
    assert code == 1, "a kernel that still exists must not be reported as cleaned up"
    assert "still listed" in out
    assert "all managed kernels deleted" not in out
    assert fake.listed == 1, "the tie must be broken by a listing, not by `status`"


def test_a_500_delete_is_never_a_success(monkeypatch, capsys, tmp_path):
    fake = FakeCLI(refusal=_refusal(_REFUSED_500), still_listed=True)
    code, out = _run(fake, tmp_path, monkeypatch, capsys)
    assert code == 1
    assert "still listed" in out


def test_a_refused_delete_absent_from_the_listing_is_a_success(monkeypatch, capsys, tmp_path):
    """The genuine "already deleted" case Kaggle 403s: the listing settles it."""
    fake = FakeCLI(refusal=_refusal(_REFUSED_403), still_listed=False)
    code, out = _run(fake, tmp_path, monkeypatch, capsys)
    assert code == 0
    assert "already gone" in out
    assert fake.listed == 1


def test_an_unanswerable_confirmation_is_not_a_success(monkeypatch, capsys, tmp_path):
    """A listing that itself fails leaves the question open; open must not read as fine."""

    class _Unlistable(FakeCLI):
        def lists_kernel(self, account):
            raise KaggleError(["kaggle", "kernels", "list"], 1, "", "500 Server Error")

    fake = _Unlistable(refusal=_refusal(_REFUSED_403))
    code, out = _run(fake, tmp_path, monkeypatch, capsys)
    assert code == 1
    assert "could not confirm" in out
    assert "all managed kernels deleted" not in out


def test_dry_run_changes_nothing(monkeypatch, capsys, tmp_path):
    fake = FakeCLI(refusal=None)
    config = _config()
    monkeypatch.setattr(cli_module, "KaggleCLI", lambda *a, **k: fake)
    monkeypatch.setattr(cli_module, "AccountStore", lambda *a, **k: _FakeStore(tmp_path))
    cli_module.cmd_cleanup(argparse.Namespace(dry_run=True), config)
    out = capsys.readouterr().out
    assert fake.deleted == [], "dry run must not delete"
    assert "would delete" in out


def test_add_account_accepts_a_token_non_interactively(monkeypatch, capsys, tmp_path):
    """`--kind` only ever holds "kaggle_json" or "access_token", so testing
    `kind == "token"` could never be true: `--token` always fell through to
    "--credential is required" and there was no way to register a token account
    without the interactive `init` prompt."""
    from kaggle_rotate import cli as cli_module

    stored = {}

    class _CLI:
        def verify(self, account):
            return None

    monkeypatch.setattr(cli_module, "KaggleCLI", lambda *a, **k: _CLI())
    monkeypatch.setattr(cli_module, "AccountStore", lambda *a, **k: _FakeStore(tmp_path))
    monkeypatch.setattr(
        cli_module, "_store_token", lambda store, accounts, slug, token: stored.setdefault(
            "slug", slug
        ) and Account(slug=slug, username=slug)
    )

    args = argparse.Namespace(
        kind="kaggle_json", token="raw-token-value", slug="bob", credential=None
    )
    assert cli_module.cmd_add_account(args, _config()) == 0
    assert stored["slug"] == "bob"
