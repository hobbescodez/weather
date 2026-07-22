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
        "observed_high_so_far_f": extremes["observed_high_so_far_f"],
        "observed_high_so_far_time": extremes["observed_high_so_far_time"].isoformat(),
        "low_status": extremes["low_status"],
        "estimated_low_f": extremes["estimated_low_f"],
        "estimated_low_time": extremes["estimated_low_time"].isoformat(),
        "observed_low_so_far_f": extremes["observed_low_so_far_f"],
        "observed_low_so_far_time": extremes["observed_low_so_far_time"].isoformat(),
        "tomorrow_high_f": extremes["tomorrow_high_f"],
        "tomorrow_low_f": extremes["tomorrow_low_f"],
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
        high = {"temp_f": last["estimated_high_f"], "time": last["estimated_high_time"]}

    low = None
    pre_dawn = [r for r in rows if r["low_status"] == "today"]
    if pre_dawn:
        last = pre_dawn[-1]
        low = {"temp_f": last["estimated_low_f"], "time": last["estimated_low_time"]}

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

        def err(pred, actual):
            return round(pred - actual, 2) if pred is not None and actual is not None else None

        days.append({
            "date": date,
            "same_day_high": {"projection": last_high_projection, "final": final_high,
                               "error": err(last_high_projection, final_high)},
            "same_day_low": {"projection": last_low_projection, "final": final_low,
                              "error": err(last_low_projection, final_low)},
            "next_day_high": {"projection": tomorrow_high_forecast, "final": next_final_high,
                               "error": err(tomorrow_high_forecast, next_final_high)},
            "next_day_low": {"projection": tomorrow_low_forecast, "final": next_final_low,
                              "error": err(tomorrow_low_forecast, next_final_low)},
        })

    return {
        "days": days,
        "same_day_high_stats": _stats([d["same_day_high"]["error"] for d in days if d["same_day_high"]["error"] is not None]),
        "same_day_low_stats": _stats([d["same_day_low"]["error"] for d in days if d["same_day_low"]["error"] is not None]),
        "next_day_high_stats": _stats([d["next_day_high"]["error"] for d in days if d["next_day_high"]["error"] is not None]),
        "next_day_low_stats": _stats([d["next_day_low"]["error"] for d in days if d["next_day_low"]["error"] is not None]),
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
