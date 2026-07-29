"""
Regression tests for silently-swallowed failures in calibration paths.

Written after _empirical_band_coverage spent its entire existence
returning FALLBACK_COVERAGE. backtest() returns a (DataFrame, summary)
TUPLE; the caller did result["pct_within_uncertainty_band"], which raises
TypeError on a tuple; a bare `except Exception: pass` swallowed it. The
function never raised, never logged, and always returned a plausible
constant - so nothing looked wrong, and the whole "calibrate sigma to the
model's measured track record" mechanism never ran.

The class of bug matters more than the instance: a fallback that is
indistinguishable from success. These tests pin the two properties that
would have caught it -

  1. when the upstream call works, the calibration uses the COMPUTED
     value and not the constant;
  2. when it genuinely fails, the fallback is used AND says so.

Everything here stubs its dependencies, so it runs offline and
deterministically - a test that needs the live NWS API would itself
become a silently-skipped check.

    python3 test_calibration_health.py
"""

import io
import contextlib

import paper_trading
import calibration_health


FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    if not condition:
        FAILURES.append(name)
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))


class _FakeFrame:
    """Stands in for the results DataFrame - only needs to be *something*
    occupying the first slot of the tuple."""


def _fake_backtest_tuple(*_a, **_k):
    """The REAL return shape: (results_df, summary). The original bug was
    treating this as if it were the summary dict alone."""
    return _FakeFrame(), {"pct_within_uncertainty_band": 0.53, "mae_f": 3.4}


def _fake_backtest_raises(*_a, **_k):
    raise RuntimeError("simulated network failure")


def _fake_backtest_dict(*_a, **_k):
    """The shape the buggy code assumed. If someone 'simplifies' the
    caller back to result[...] this stays green while the tuple test
    fails - which is the whole point of testing both."""
    return {"pct_within_uncertainty_band": 0.53}


def test_coverage_uses_computed_value():
    original = paper_trading.backtest
    try:
        paper_trading.backtest = _fake_backtest_tuple
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = paper_trading._empirical_band_coverage("KSEA")
        check(
            "coverage extracted from backtest's (df, summary) tuple",
            got == 0.53,
            f"got {got!r}, expected 0.53",
        )
        check(
            "computed coverage is NOT the fallback constant",
            got != paper_trading.FALLBACK_COVERAGE,
            f"got {got!r}, fallback is {paper_trading.FALLBACK_COVERAGE!r}",
        )
        check(
            "success path stays quiet",
            buf.getvalue().strip() == "",
            f"unexpected output: {buf.getvalue()!r}",
        )
    finally:
        paper_trading.backtest = original


def test_failure_falls_back_but_announces_it():
    original = paper_trading.backtest
    try:
        paper_trading.backtest = _fake_backtest_raises
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = paper_trading._empirical_band_coverage("KSEA")
        check(
            "genuine failure falls back rather than crashing",
            got == paper_trading.FALLBACK_COVERAGE,
            f"got {got!r}",
        )
        check(
            "the swallowed failure is reported, not silent",
            "simulated network failure" in buf.getvalue(),
            f"stdout was {buf.getvalue()!r}",
        )
    finally:
        paper_trading.backtest = original


def test_wrong_shape_would_be_caught():
    """If backtest's return shape ever changes back to a bare dict, the
    tuple unpacking fails - and that must surface as a reported fallback,
    never as a silent plausible number."""
    original = paper_trading.backtest
    try:
        paper_trading.backtest = _fake_backtest_dict
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = paper_trading._empirical_band_coverage("KSEA")
        check(
            "a shape mismatch reports instead of silently returning a constant",
            got == paper_trading.FALLBACK_COVERAGE and buf.getvalue().strip() != "",
            f"got {got!r}, stdout {buf.getvalue()!r}",
        )
    finally:
        paper_trading.backtest = original


def test_no_silent_swallows_in_repo():
    """Source-level guard against the pattern itself coming back
    anywhere, not just in this one function."""
    offenders = calibration_health.find_silent_exception_handlers()
    check(
        "no bare `except ...: pass` outside the reviewed allowlist",
        not offenders,
        "; ".join(f"{f}:{ln}" for f, ln in offenders),
    )


def test_health_report_flags_fallbacks():
    original = paper_trading.backtest
    try:
        paper_trading.backtest = _fake_backtest_raises
        with contextlib.redirect_stdout(io.StringIO()):
            report = calibration_health.check_calibration_sources("KSEA")
        band = next(r for r in report if r["name"] == "uncertainty_band_coverage")
        check(
            "health report marks a fallback as degraded, not ok",
            band["status"] == "fallback",
            f"status was {band['status']!r}",
        )

        paper_trading.backtest = _fake_backtest_tuple
        with contextlib.redirect_stdout(io.StringIO()):
            report = calibration_health.check_calibration_sources("KSEA")
        band = next(r for r in report if r["name"] == "uncertainty_band_coverage")
        check(
            "health report marks a real computed value as ok",
            band["status"] == "ok" and band["value"] == 0.53,
            f"got {band!r}",
        )
    finally:
        paper_trading.backtest = original


if __name__ == "__main__":
    print("calibration health / silent-fallback regression tests\n")
    for fn in (
        test_coverage_uses_computed_value,
        test_failure_falls_back_but_announces_it,
        test_wrong_shape_would_be_caught,
        test_no_silent_swallows_in_repo,
        test_health_report_flags_fallbacks,
    ):
        print(fn.__name__)
        fn()
        print()
    if FAILURES:
        raise SystemExit(f"FAILED: {len(FAILURES)} check(s): {', '.join(FAILURES)}")
    print("all checks passed")
