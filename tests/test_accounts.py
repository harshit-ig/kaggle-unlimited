"""Delete must report the truth: Kaggle answers 403 for kernels that are already gone."""

from kaggle_rotate.accounts import KaggleError, kernel_is_gone, parse_status


def _err(stdout: str = "", stderr: str = "") -> KaggleError:
    return KaggleError(["kaggle", "kernels", "status", "x/y"], 1, stdout, stderr)


def test_gone_kernel_is_recognised_from_status_errors():
    """Kaggle reports a deleted kernel as an access error on `kernels status`."""
    assert kernel_is_gone(
        _err(stderr="Cannot access kernel 'a/b' (Permission 'kernels.get' was denied).")
    )
    assert kernel_is_gone(_err(stderr="Not Found"))
    assert kernel_is_gone(_err(stdout="kernel not found"))
    assert kernel_is_gone(_err(stderr="404 The kernel does not exist"))


def test_real_failures_are_not_mistaken_for_gone():
    assert not kernel_is_gone(_err(stderr="500 Server Error: Internal Server Error"))
    assert not kernel_is_gone(_err(stderr="Your kernel title does not resolve to the specified id"))
    assert not kernel_is_gone(_err(stderr="Authentication required to call the Kaggle API."))


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
