"""
Reports whether the model's calibration inputs are running on real
computed values or on fallback constants.

Motivation, concretely: paper_trading._empirical_band_coverage spent its
entire existence returning FALLBACK_COVERAGE because backtest() returns a
(DataFrame, summary) tuple and the caller indexed it like a dict. A bare
`except Exception: pass` swallowed the TypeError. Nothing raised, nothing
logged, and the value returned was a plausible constant - so the failure
was indistinguishable from success, and the whole "calibrate sigma to the
model's measured track record" mechanism silently never ran.

The general defect is a fallback you cannot tell apart from a real
result. This module makes that distinction explicit and loud:
check_calibration_sources() reports, per input, whether it is "ok"
(computed), "fallback" (degraded but running), or "error". Anything not
"ok" is printed prominently by build_dashboard on every refresh, so a
recurrence surfaces within the hour rather than in a month's worth of
quietly mis-scaled bet probabilities.

find_silent_exception_handlers() is the source-level counterpart: it
fails the test suite if `except ...: pass` reappears outside a small
reviewed allowlist of genuine try-the-next-candidate loops.

    python3 calibration_health.py          # human-readable report
"""

import ast
import os

# Handlers where swallowing really is the intent: iterate candidates, skip
# the ones that don't work, and the caller can see the outcome (a None
# station, a shorter list, a skipped day). Keyed by file -> set of the
# enclosing function names that are allowed to contain a silent handler.
SILENT_HANDLER_ALLOWLIST = {
    "weather_estimator.py": {
        "_fetch_upwind_df",      # try each upwind candidate; returns (None, None)
        "_fetch_role_df",        # same, per network role
        "backtest",              # skip a window that can't be estimated
    },
    "daily_performance.py": {
        "_kalshi_day_stats",             # per-bracket candle fetch
        "reconcile_peak_time_windows",   # skip a day whose history is gone
        "reconcile_stream_fallback_actuals",
        "finalize_pending_days",
    },
    "nws_climate.py": {
        "fetch_recent_cli_finals",  # per-product parse; absence is reported by the caller
    },
}

_REPO = os.path.dirname(os.path.abspath(__file__))


def _is_silent(handler):
    """A handler whose entire body discards the exception without
    reporting it - `pass`, or a bare continue/return with no logging."""
    body = [n for n in handler.body if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Constant)]
    if not body:
        return True
    if len(body) > 1:
        return False
    node = body[0]
    if isinstance(node, ast.Pass):
        return True
    if isinstance(node, ast.Continue):
        return True
    if isinstance(node, ast.Return) and (
        node.value is None or isinstance(node.value, ast.Constant)
        or isinstance(node.value, (ast.Dict, ast.List, ast.Tuple))
    ):
        return True
    return False


def find_silent_exception_handlers(repo=None):
    """Every `except ...:` whose body silently discards the error, outside
    the reviewed allowlist. Returns [(relative_path, lineno), ...]."""
    repo = repo or _REPO
    offenders = []
    for fname in sorted(os.listdir(repo)):
        if not fname.endswith(".py") or fname.startswith("test_"):
            continue
        if fname == os.path.basename(__file__):
            continue
        path = os.path.join(repo, fname)
        try:
            tree = ast.parse(open(path, encoding="utf-8").read(), filename=fname)
        except SyntaxError:
            continue
        allowed = SILENT_HANDLER_ALLOWLIST.get(fname, set())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.ExceptHandler) and _is_silent(sub):
                    if node.name not in allowed:
                        offenders.append((fname, sub.lineno))
    return sorted(set(offenders))


def check_calibration_sources(station_id="KSEA"):
    """
    Per calibration input: is it a real computed value, or a constant
    standing in for one?

    Returns a list of {name, status, value, expected_fallback, detail}
    where status is "ok" | "fallback" | "error".
    """
    import paper_trading

    results = []

    # 1. Empirical band coverage -> paper_trading's calibrated sigma ->
    #    every bracket probability -> every bet decision. This is the one
    #    that was silently broken.
    try:
        value = paper_trading._empirical_band_coverage(station_id)
        fallback = paper_trading.FALLBACK_COVERAGE
        on_fallback = value == fallback
        results.append({
            "name": "uncertainty_band_coverage",
            "status": "fallback" if on_fallback else "ok",
            "value": value,
            "expected_fallback": fallback,
            "detail": (
                "running on the FALLBACK constant - calibrated sigma is NOT "
                "tracking measured coverage"
                if on_fallback else
                "computed from a live backtest"
            ),
        })
    except Exception as e:
        results.append({
            "name": "uncertainty_band_coverage", "status": "error", "value": None,
            "expected_fallback": paper_trading.FALLBACK_COVERAGE, "detail": repr(e),
        })

    # 2. CLI settlement source. fetch_recent_cli_finals returns {} both
    #    when nothing is published yet and when parsing is broken - those
    #    look identical to callers, so surface the count.
    try:
        from nws_climate import fetch_recent_cli_finals
        finals = fetch_recent_cli_finals()
        results.append({
            "name": "cli_settlement_reports",
            "status": "ok" if finals else "fallback",
            "value": len(finals),
            "expected_fallback": 0,
            "detail": (
                f"{len(finals)} final CLI report(s) parsed"
                if finals else
                "NO CLI reports parsed - every actual will fall back to the "
                "observation stream; could be a genuine publishing gap OR a "
                "broken parser/station id"
            ),
        })
    except Exception as e:
        results.append({
            "name": "cli_settlement_reports", "status": "error", "value": None,
            "expected_fallback": 0, "detail": repr(e),
        })

    return results


def report(station_id="KSEA"):
    """Print any degraded calibration input. Returns True if all ok."""
    rows = check_calibration_sources(station_id)
    bad = [r for r in rows if r["status"] != "ok"]
    for r in rows:
        if r["status"] != "ok":
            print(
                f"CALIBRATION {r['status'].upper()}: {r['name']} = {r['value']!r} "
                f"- {r['detail']}"
            )
    return not bad


if __name__ == "__main__":
    print("silent exception handlers outside the allowlist:")
    off = find_silent_exception_handlers()
    print("  " + ("none" if not off else "; ".join(f"{f}:{l}" for f, l in off)))
    print("\ncalibration sources:")
    for r in check_calibration_sources():
        print(f"  [{r['status']:8}] {r['name']:26} = {r['value']!r:8} {r['detail']}")
