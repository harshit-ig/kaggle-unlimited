"""Delete must report the truth: a refused delete is not a deleted kernel."""

from kaggle_rotate.accounts import KaggleError, kernel_delete_refused, kernel_is_gone, parse_status


def _err(stdout: str = "", stderr: str = "", returncode: int = 1) -> KaggleError:
    return KaggleError(["kaggle", "kernels", "status", "x/y"], returncode, stdout, stderr)


def test_gone_kernel_is_recognised_from_status_errors():
    """A kernel Kaggle admits it cannot find is genuinely not running."""
    assert kernel_is_gone(_err(stderr="Not Found"))
    assert kernel_is_gone(_err(stdout="kernel not found"))
    assert kernel_is_gone(_err(stderr="404 The kernel does not exist"))
    assert kernel_is_gone(_err(stderr="The kernel does not exist"))


def test_a_403_is_never_proof_that_nothing_is_running():
    """The bug this guards: Kaggle 403s an already-deleted kernel with the same body it
    uses for wrong credentials, another owner's kernel and an unverified account. Reading
    that as "already gone" is how cleanup reported success over a kernel still billing."""
    denied = _err(stderr="Cannot access kernel 'a/b' (Permission 'kernels.get' was denied).")
    assert not kernel_is_gone(denied)
    assert kernel_delete_refused(denied), "must be routed to a listing, not to success"

    forbidden = _err(stderr="403 Client Error: Forbidden for url: .../DeleteKernel")
    assert not kernel_is_gone(forbidden)
    assert kernel_delete_refused(forbidden)


def test_real_failures_are_not_mistaken_for_gone():
    assert not kernel_is_gone(_err(stderr="500 Server Error: Internal Server Error"))
    assert not kernel_is_gone(_err(stderr="Your kernel title does not resolve to the specified id"))
    assert not kernel_is_gone(_err(stderr="Authentication required to call the Kaggle API."))


def test_a_command_that_never_reached_kaggle_proves_nothing():
    """124 is our own timeout, 127 a missing binary. Neither says anything about the
    kernel, and both contain wording that would otherwise read as an answer."""
    timed_out = _err(stderr="timed out after 300s", returncode=124)
    assert not kernel_is_gone(timed_out)
    assert not kernel_delete_refused(timed_out)

    missing = _err(stderr="kaggle not found on PATH", returncode=127)
    assert not kernel_is_gone(missing), "'not found' here is the binary, not the kernel"
    assert not kernel_delete_refused(missing)


def test_status_parsing_distinguishes_unknown_from_stopped():
    timed_out = parse_status("status probe timed out after 60s")
    assert timed_out.known is False, "a timeout must not read as 'stopped'"
    assert not timed_out.alive

    running = parse_status('a/b has status "KernelWorkerStatus.RUNNING"')
    assert running.known and running.alive and not running.terminal

    for text, expected in [
        ("KernelWorkerStatus.COMPLETE", "COMPLETE"),
        ("KernelWorkerStatus.ERROR", "ERROR"),
        ("KernelWorkerStatus.CANCELLED", "CANCELLED"),
    ]:
        result = parse_status(f'a/b has status "{text}"')
        assert result.state == expected, f"{text} parsed as {result.state}"
        assert result.terminal


class _StubAccount:
    def __init__(self, username="u"):
        self.username = username


def _cli(recorder, listing: str):
    """A KaggleCLI whose `run` records argv and returns `listing`, spawning nothing."""
    from kaggle_rotate.accounts import AccountStore, KaggleCLI

    class _Config:
        kernel = type("K", (), {"kernel_slug": "krotate"})()
        kaggle = type("G", (), {"command": "kaggle"})()

        @staticmethod
        def kernel_ref(username: str) -> str:
            return f"{username}/krotate"

    class _CLI(KaggleCLI):
        def run(self, account, *args, **kwargs):
            recorder.append(args)
            return type("P", (), {"returncode": 0, "stdout": listing, "stderr": ""})()

    return _CLI(config=_Config(), store=AccountStore.__new__(AccountStore))


def test_the_confirmation_listing_is_scoped_to_my_own_kernels():
    """A plain `kernels list` is every public kernel on the platform, so whether the
    ref appears depends on paging. "Absent" from that would prove nothing, which is the
    same failure as trusting a 403."""
    calls: list[tuple[str, ...]] = []
    cli = _cli(calls, listing="someoneelse/krotate\nu/krotate\n")

    assert cli.lists_kernel(_StubAccount()) is True
    assert "--mine" in calls[0], f"must be scoped to the account's own kernels: {calls[0]}"


def test_a_kernel_missing_from_my_own_listing_is_reported_absent():
    calls: list[tuple[str, ...]] = []
    cli = _cli(calls, listing="someoneelse/other-kernel\n")

    assert cli.lists_kernel(_StubAccount()) is False


def test_an_unanswered_listing_raises_rather_than_reading_as_absent():
    """Empty output from a *failed* command must not be read as 'the kernel is gone'."""
    from kaggle_rotate.accounts import AccountStore, KaggleCLI, KaggleError

    class _CLI(KaggleCLI):
        def run(self, account, *args, **kwargs):
            raise KaggleError(["kaggle", "kernels", "list"], 1, "", "500 Server Error")

    config = type("C", (), {"kernel_ref": staticmethod(lambda u: f"{u}/krotate")})()
    cli = _CLI(config=config, store=AccountStore.__new__(AccountStore))

    try:
        cli.lists_kernel(_StubAccount())
    except KaggleError:
        return
    raise AssertionError("a failed listing must propagate, not return False")
