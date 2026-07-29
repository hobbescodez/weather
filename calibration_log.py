"""
Appends one row per dashboard refresh recording the model's current
high/low estimate alongside what's actually been observed so far, so
that after enough days accumulate we can see how the pre-peak/pre-dawn
"projected" estimate and the next-day forecast compared to what actually
settled - data-driven recalibration instead of eyeballing single days.

(PEAK_HEAT_FRACTION was tuned off just ~6 days by hand earlier in this
project - exactly the kind of small-sample noise this log exists to
replace with something sturdier over time.)

Wired into build_dashboard.py, which already runs on every scheduled
refresh - nothing extra needs to be scheduled. Run this file directly to
print a summary of what's accumulated so far:
    python3 calibration_log.py
"""

import json
import os

LOG_PATH = os.path.join(os.path.dirname(__file__), "calibration_log.jsonl")


def record_snapshot(extremes, est=None):
    """Append one row from an estimate_daily_extremes() result.

    estimated_high_time/estimated_low_time are included alongside the
    temp estimates - daily_performance.py's finalize_day() needs the
    model's predicted peak *time*, not just its predicted value, and
    these are the only place that's available (they're the sun-derived
    peak-heat-hour/sunrise times, not tied to when the actual high/low
    occurred - see weather_estimator.py's own comments on those fields).

    est: optionally, an estimate_temp() result logged alongside - the
    raw multi-station gradient network readings and the
    marine_push_index/offshore_flow_index derived from them (see
    weather_estimator.py's Calibration section), logged every day
    regardless of whether anything unusual happens. That's the point:
    once enough days accumulate (including at least one real marine-push
    and, ideally, one real offshore-flow/heat event), comparing days with
    large peak_temp_error_f against what these indices were reading
    beforehand is how those indices earn (or lose) a larger role in the
    estimate, instead of staying hand-picked constants forever. Omitted
    (all fields None) if est isn't passed - keeps this callable exactly
    as before for any other caller.
    """
    now = extremes["as_of"]
    row = {
        "logged_at": now.isoformat(),
        "date": now.date().isoformat(),
        "high_status": extremes["high_status"],
        "estimated_high_f": extremes["estimated_high_f"],
        "estimated_high_time": extremes["estimated_high_time"].isoformat(),
        # NWS's own hourly-forecast value for this same target time,
        # captured at this same checkpoint - only set while high_status ==
        # "projected" (see weather_estimator.estimate_daily_extremes). Lets
        # daily_performance.py compare model vs. NWS vs. actual for the
        # exact same moment, instead of just "do they agree today" with no
        # record of which one was actually closer.
        "nws_high_forecast_at_target_f": extremes.get("nws_high_forecast_at_target_f"),
        # estimated_high_f above is now the NWS-blended figure - what's
        # displayed and what paper_trading bets on. These two keep the
        # comparison alive: the in-house trend model's own unblended number,
        # and how much of the displayed value came from NWS. Without them a
        # backtest after the blend landed would only ever be able to score
        # the blend against itself.
        "trend_only_high_f": extremes.get("trend_only_high_f"),
        "high_nws_blend_weight": extremes.get("high_nws_blend_weight"),
        "observed_high_so_far_f": extremes["observed_high_so_far_f"],
        "observed_high_so_far_time": extremes["observed_high_so_far_time"].isoformat(),
        "low_status": extremes["low_status"],
        "estimated_low_f": extremes["estimated_low_f"],
        "estimated_low_time": extremes["estimated_low_time"].isoformat(),
        "nws_low_forecast_at_target_f": extremes.get("nws_low_forecast_at_target_f"),
        "trend_only_low_f": extremes.get("trend_only_low_f"),
        "low_nws_blend_weight": extremes.get("low_nws_blend_weight"),
        "observed_low_so_far_f": extremes["observed_low_so_far_f"],
        "observed_low_so_far_time": extremes["observed_low_so_far_time"].isoformat(),
        "tomorrow_high_f": extremes["tomorrow_high_f"],
        "tomorrow_high_source": extremes["tomorrow_high_source"],
        "tomorrow_low_f": extremes["tomorrow_low_f"],
        "tomorrow_low_source": extremes["tomorrow_low_source"],
        "pressure_gradient_station": est["pressure_gradient_station"] if est else None,
        "pressure_gradient_inhg": est["pressure_gradient_inhg"] if est else None,
        "pressure_gradient_trend_inhg_per_hr": est["pressure_gradient_trend_inhg_per_hr"] if est else None,
        "strait_station": est["strait_signal"]["station"] if est and est.get("strait_signal") else None,
        "strait_pressure_gradient_inhg": est["strait_signal"]["pressure_gradient_inhg"] if est and est.get("strait_signal") else None,
        "strait_pressure_gradient_trend_inhg_per_hr": est["strait_signal"]["pressure_gradient_trend_inhg_per_hr"] if est and est.get("strait_signal") else None,
        "interior_gap_station": est["interior_gap_signal"]["station"] if est and est.get("interior_gap_signal") else None,
        "interior_gap_pressure_gradient_inhg": est["interior_gap_signal"]["pressure_gradient_inhg"] if est and est.get("interior_gap_signal") else None,
        "interior_gap_pressure_gradient_trend_inhg_per_hr": est["interior_gap_signal"]["pressure_gradient_trend_inhg_per_hr"] if est and est.get("interior_gap_signal") else None,
        "interior_gap_temp_gradient_f": est["interior_gap_signal"]["temp_gradient_f"] if est and est.get("interior_gap_signal") else None,
        "interior_gap_temp_gradient_trend_f_per_hr": est["interior_gap_signal"]["temp_gradient_trend_f_per_hr"] if est and est.get("interior_gap_signal") else None,
        "marine_push_index": est["marine_push_index"] if est else None,
        "offshore_flow_index": est["offshore_flow_index"] if est else None,
        "uncertainty_note": est["uncertainty_note"] if est else None,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")


def get_last_prediction(date_str):
    """
    The model's final pre-peak/pre-dawn prediction for a given date
    ("YYYY-MM-DD"), for daily_performance.py's finalize_day(): the last
    snapshot logged while high_status was still "projected" (before that
    day's peak-heat hour had passed) and while low_status was still
    "today" (before dawn). Returns {"high": {"temp_f", "time"} or None,
    "low": {"temp_f", "time"} or None}.
    """
    rows = [r for r in _load_rows() if r["date"] == date_str]
    rows.sort(key=lambda r: r["logged_at"])

    high = None
    pre_peak = [r for r in rows if r["high_status"] == "projected"]
    if pre_peak:
        last = pre_peak[-1]
        high = {
            "temp_f": last["estimated_high_f"], "time": last["estimated_high_time"],
            "nws_forecast_f": last.get("nws_high_forecast_at_target_f"),
        }

    low = None
    pre_dawn = [r for r in rows if r["low_status"] == "today"]
    if pre_dawn:
        last = pre_dawn[-1]
        low = {
            "temp_f": last["estimated_low_f"], "time": last["estimated_low_time"],
            "nws_forecast_f": last.get("nws_low_forecast_at_target_f"),
        }

    return {"high": high, "low": low}


def _load_rows():
    if not os.path.exists(LOG_PATH):
        return []
    rows = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _stats(errors):
    if not errors:
        return None
    n = len(errors)
    bias = sum(errors) / n
    mae = sum(abs(e) for e in errors) / n
    return {"n": n, "bias_f": round(bias, 2), "mae_f": round(mae, 2)}


def summarize():
    """
    For each date with data, compares:
      - the last pre-peak "projected" high estimate against that date's
        final settled high (max observed_high_so_far_f seen that date)
      - the last pre-dawn "today"-status low estimate against that
        date's final settled low (min observed_low_so_far_f seen)
      - that date's tomorrow_high_f/tomorrow_low_f (the next-day forecast)
        against the FOLLOWING date's final settled high/low - the most
        useful one to calibrate, since it's the longest-horizon guess and
        the one closest to what a next-day Kalshi trade would lean on

    Returns per-day rows plus aggregate bias/MAE for each of the three
    tracks. Works fine with very few days logged - it's meant to grow.
    """
    rows = _load_rows()
    by_date = {}
    for r in rows:
        by_date.setdefault(r["date"], []).append(r)
    for day_rows in by_date.values():
        day_rows.sort(key=lambda r: r["logged_at"])

    dates = sorted(by_date.keys())
    days = []
    for i, date in enumerate(dates):
        day_rows = by_date[date]

        final_high = max(
            (r["observed_high_so_far_f"] for r in day_rows if r["observed_high_so_far_f"] is not None),
            default=None,
        )
        pre_peak = [r for r in day_rows if r["high_status"] == "projected"]
        last_high_projection = pre_peak[-1]["estimated_high_f"] if pre_peak else None

        final_low = min(
            (r["observed_low_so_far_f"] for r in day_rows if r["observed_low_so_far_f"] is not None),
            default=None,
        )
        pre_dawn = [r for r in day_rows if r["low_status"] == "today"]
        last_low_projection = pre_dawn[-1]["estimated_low_f"] if pre_dawn else None

        next_final_high = None
        next_final_low = None
        if i + 1 < len(dates):
            next_rows = by_date[dates[i + 1]]
            next_final_high = max(
                (r["observed_high_so_far_f"] for r in next_rows if r["observed_high_so_far_f"] is not None),
                default=None,
            )
            next_final_low = min(
                (r["observed_low_so_far_f"] for r in next_rows if r["observed_low_so_far_f"] is not None),
                default=None,
            )
        tomorrow_high_forecast = day_rows[-1]["tomorrow_high_f"]
        tomorrow_low_forecast = day_rows[-1]["tomorrow_low_f"]
        # Older rows predate this field - .get() so summarize() doesn't
        # break replaying an existing log.
        tomorrow_high_source = day_rows[-1].get("tomorrow_high_source")
        tomorrow_low_source = day_rows[-1].get("tomorrow_low_source")

        def err(pred, actual):
            return round(pred - actual, 2) if pred is not None and actual is not None else None

        days.append({
            "date": date,
            "same_day_high": {"projection": last_high_projection, "final": final_high,
                               "error": err(last_high_projection, final_high)},
            "same_day_low": {"projection": last_low_projection, "final": final_low,
                              "error": err(last_low_projection, final_low)},
            "next_day_high": {"projection": tomorrow_high_forecast, "final": next_final_high,
                               "error": err(tomorrow_high_forecast, next_final_high),
                               "source": tomorrow_high_source},
            "next_day_low": {"projection": tomorrow_low_forecast, "final": next_final_low,
                              "error": err(tomorrow_low_forecast, next_final_low),
                              "source": tomorrow_low_source},
        })

    return {
        "days": days,
        "same_day_high_stats": _stats([d["same_day_high"]["error"] for d in days if d["same_day_high"]["error"] is not None]),
        "same_day_low_stats": _stats([d["same_day_low"]["error"] for d in days if d["same_day_low"]["error"] is not None]),
        "next_day_high_stats": _stats([d["next_day_high"]["error"] for d in days if d["next_day_high"]["error"] is not None]),
        "next_day_low_stats": _stats([d["next_day_low"]["error"] for d in days if d["next_day_low"]["error"] is not None]),
    }


# build_dashboard.py's "tomorrow's high/low" confidence % used to be a
# hand-picked constant (75 for an NWS-forecast-grounded guess, 35-55 for
# the weaker persistence fallback - see weather_estimator.py's
# estimate_daily_extremes) with UI copy promising it "will switch to a
# measured number once enough days accumulate." Nothing ever computed that
# measured number - the promise was aspirational copy, not a real
# mechanism. next_day_confidence_pct below is that mechanism.
#
# Split by source (nws_forecast vs persistence_fallback) rather than one
# pooled MAE across both: the two paths have very different expected
# accuracy (that's the whole reason the UI distinguishes them), and
# persistence_fallback triggers rarely, so pooling would let mostly-
# nws_forecast days quietly stand in for a persistence-fallback day's own,
# probably worse, real track record.
MIN_NEXT_DAY_SAMPLES = 10  # modest but more than daily_performance.py's 5-day "low_sample" rollup threshold, since this feeds a headline UI number rather than an internal average

# First-pass MAE -> confidence-percent mapping - not a calibrated
# probability, just a monotonic "smaller error -> more confidence"
# translation, deliberately scaled so it lines up with the constants it's
# replacing (75 for the forecast path, 35-55 for persistence) rather than
# jumping to a wildly different number the day it switches on. Revisit
# this formula itself once real data shows whether it over- or
# understates confidence relative to how often next-day temps actually
# land close to the forecast - same "don't trust hand-picked constants
# forever" principle as PEAK_HEAT_FRACTION and the paper-trading sigma.
_CONFIDENCE_MAX_PCT = 90
_CONFIDENCE_MIN_PCT = 30
_CONFIDENCE_PCT_PER_DEGREE_MAE = 10


def _mae_to_confidence_pct(mae_f):
    pct = _CONFIDENCE_MAX_PCT - _CONFIDENCE_PCT_PER_DEGREE_MAE * mae_f
    return round(max(_CONFIDENCE_MIN_PCT, min(_CONFIDENCE_MAX_PCT, pct)))


def next_day_confidence_pct(source_high, source_low, min_samples=MIN_NEXT_DAY_SAMPLES):
    """
    Measured next-day confidence for each side, source-matched: only
    finalized days whose tomorrow_high_f/tomorrow_low_f came from the SAME
    source (source_high/source_low - typically today's own
    extremes["tomorrow_high_source"]/["tomorrow_low_source"]) count toward
    that side's sample, so a forecast-grounded day never gets padded out
    by persistence-fallback days' track record or vice versa.

    Returns {"high": pct_or_None, "low": pct_or_None, "n_high": n, "n_low":
    n}. A None pct means fewer than min_samples matching days exist yet -
    the caller should keep showing its own source-based placeholder
    (see build_dashboard.py's TOMORROW_HINTS) rather than a number this
    thin could support.
    """
    days = summarize()["days"]

    high_errors = [
        d["next_day_high"]["error"] for d in days
        if d["next_day_high"]["error"] is not None and d["next_day_high"]["source"] == source_high
    ]
    low_errors = [
        d["next_day_low"]["error"] for d in days
        if d["next_day_low"]["error"] is not None and d["next_day_low"]["source"] == source_low
    ]

    high_stats = _stats(high_errors)
    low_stats = _stats(low_errors)

    return {
        "high": _mae_to_confidence_pct(high_stats["mae_f"]) if high_stats and high_stats["n"] >= min_samples else None,
        "low": _mae_to_confidence_pct(low_stats["mae_f"]) if low_stats and low_stats["n"] >= min_samples else None,
        "n_high": high_stats["n"] if high_stats else 0,
        "n_low": low_stats["n"] if low_stats else 0,
    }


if __name__ == "__main__":
    summary = summarize()
    for d in summary["days"]:
        print(d["date"])
        for track in ("same_day_high", "same_day_low", "next_day_high", "next_day_low"):
            print(f"  {track}: {d[track]}")
    print()
    print("same_day_high_stats:", summary["same_day_high_stats"])
    print("same_day_low_stats: ", summary["same_day_low_stats"])
    print("next_day_high_stats:", summary["next_day_high_stats"])
    print("next_day_low_stats: ", summary["next_day_low_stats"])
