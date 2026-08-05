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
from datetime import datetime

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
        "nws_high_forecast_daily_f": extremes.get("nws_high_forecast_daily_f"),
        "trend_only_high_f": extremes.get("trend_only_high_f"),
        "high_nws_blend_weight": extremes.get("high_nws_blend_weight"),
        "observed_high_so_far_f": extremes["observed_high_so_far_f"],
        "observed_high_so_far_time": extremes["observed_high_so_far_time"].isoformat(),
        "low_status": extremes["low_status"],
        "estimated_low_f": extremes["estimated_low_f"],
        "estimated_low_time": extremes["estimated_low_time"].isoformat(),
        "nws_low_forecast_at_target_f": extremes.get("nws_low_forecast_at_target_f"),
        "nws_low_forecast_daily_f": extremes.get("nws_low_forecast_daily_f"),
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


# ---------------------------------------------------------------------------
# Condition buckets
#
# One pooled accuracy figure says the same thing on a still, clear day as
# on one with a marine push coming in, which is precisely when the model
# is least like its own average. Every snapshot already records the
# condition indices that were live when the prediction was made, so the
# days can be grouped by them and scored separately.
#
# Thresholds are weather_estimator's own "elevated enough to call out"
# cutoffs, not new ones invented here, so a day is bucketed "active" by
# exactly the test that made the model flag it as active at the time.
# Imported lazily: this module is deliberately runnable and importable
# without pandas/requests/astral (see summarize()'s note on deriving local
# today from the log's own timestamps), and the fallback keeps it that way
# if that ever stops holding.
# ---------------------------------------------------------------------------

# The cross-station gradient trend has no equivalent published cutoff in
# weather_estimator - the persistence fallback's +/-0.015 inHg/hr is the
# nearest thing the codebase already commits to, so it is reused rather
# than a fresh number being chosen to make these buckets come out well.
CONDITION_PRESSURE_TREND_THRESHOLD = 0.015


_FALLBACK_INDEX_THRESHOLDS = (8.0, 8.0)
_index_thresholds_cache = None


def _index_thresholds():
    """weather_estimator's live thresholds, or a copy of them.

    The copy is the failure mode worth being loud about: if it silently
    stood in after those constants had been recalibrated, every bucket
    would be split at the old cutoffs while the model flagged conditions
    at the new ones, and the "similar past days" would stop being similar
    by any definition the model recognises. Reported once per process
    rather than per row - classify_conditions runs on every logged row.
    """
    global _index_thresholds_cache
    if _index_thresholds_cache is None:
        try:
            from weather_estimator import (
                MARINE_PUSH_INDEX_THRESHOLD, OFFSHORE_FLOW_INDEX_THRESHOLD)
            _index_thresholds_cache = (
                MARINE_PUSH_INDEX_THRESHOLD, OFFSHORE_FLOW_INDEX_THRESHOLD)
        except Exception as e:
            print(f"calibration_log: could not read weather_estimator's index "
                  f"thresholds ({e}); condition buckets fall back to "
                  f"{_FALLBACK_INDEX_THRESHOLDS}, which is only correct while "
                  f"those constants are unchanged")
            _index_thresholds_cache = _FALLBACK_INDEX_THRESHOLDS
    return _index_thresholds_cache


def classify_conditions(row):
    """Condition state of one logged snapshot, as {axis: state}.

    An axis whose index is missing from the row is omitted entirely rather
    than defaulted to "steady" - the early days of the log predate the
    station network, and calling those days steady would file genuinely
    unknown conditions under the calmest bucket and quietly inflate its
    apparent accuracy.
    """
    mpi_t, ofi_t = _index_thresholds()
    out = {}
    mpi = row.get("marine_push_index")
    if mpi is not None:
        out["marine_push"] = "active" if mpi > mpi_t else "steady"
    ofi = row.get("offshore_flow_index")
    if ofi is not None:
        out["offshore_flow"] = "rising" if ofi > ofi_t else "steady"
    pgt = row.get("pressure_gradient_trend_inhg_per_hr")
    if pgt is not None:
        if pgt < -CONDITION_PRESSURE_TREND_THRESHOLD:
            out["pressure_gradient"] = "falling"
        elif pgt > CONDITION_PRESSURE_TREND_THRESHOLD:
            out["pressure_gradient"] = "rising"
        else:
            out["pressure_gradient"] = "steady"
    return out


# Backoff ladder, most specific first. A bucket is used only if it has at
# least MIN_CONDITION_BUCKET_SAMPLES scored days behind it; otherwise the
# next level down is tried, and the pooled all-days figure is the floor.
#
# pressure_gradient is dropped first because it is the least independent
# of the three: compute_marine_push_index is literally a rescaled average
# of that same gradient trend, so the third axis mostly re-splits days the
# first axis has already split, spending sample size for very little new
# information.
#
# There is deliberately no single-axis rung between the pair and the
# pooled figure. On the log as it stands a marine_push-only bucket would
# hold 10 of the 13 scored high-side days - close enough to the pooled
# figure to be indistinguishable from it, while sounding more specific
# than it is.
CONDITION_BACKOFF_LEVELS = (
    ("marine_push", "offshore_flow", "pressure_gradient"),
    ("marine_push", "offshore_flow"),
    (),
)

# Five is the same "enough to say something, not enough to lean on"
# threshold daily_performance.py uses for its low_sample rollup flag. It
# is deliberately lower than MIN_NEXT_DAY_SAMPLES (10): that one gates a
# figure quoted with no qualifier attached, whereas this one is always
# displayed with its own sample count next to it, so the reader can see
# exactly how thin it is.
MIN_CONDITION_BUCKET_SAMPLES = 5

_CONDITION_LABELS = {
    ("marine_push", "active"): "marine push active",
    ("marine_push", "steady"): "marine push steady",
    ("offshore_flow", "rising"): "offshore flow rising",
    ("offshore_flow", "steady"): "offshore flow steady",
    ("pressure_gradient", "falling"): "pressure falling",
    ("pressure_gradient", "rising"): "pressure rising",
    ("pressure_gradient", "steady"): "pressure steady",
}


def describe_conditions(conditions, axes):
    """Human phrase for the subset of `conditions` on `axes`."""
    parts = [
        _CONDITION_LABELS.get((a, conditions[a]), f"{a} {conditions[a]}")
        for a in axes if a in conditions
    ]
    return ", ".join(parts)


def condition_confidence(side, conditions,
                         min_samples=MIN_CONDITION_BUCKET_SAMPLES,
                         summary=None):
    """
    Measured same-day accuracy for the finalized days whose conditions at
    prediction time matched `conditions`, walked down CONDITION_BACKOFF_LEVELS
    until a level has enough of them.

    side: "high" or "low". conditions: a classify_conditions() dict for
    the prediction being made now.

    Returns None when even the pooled all-days figure is too thin to
    quote. Otherwise a dict with:

        pct        confidence %, via the same _mae_to_confidence_pct
                   mapping the next-day figure uses, so the two numbers on
                   the page mean the same thing
        n          scored days behind it
        mae_f      those days' MAE
        axes       which condition axes it is bucketed on - () for pooled
        label      human phrase for the bucket, "" for pooled
        matched    True if bucketed on at least one axis, False if this is
                   the pooled fallback
        tried      [(axes, n)] for every level attempted, so the caller can
                   say why a more specific bucket was not used

    A bucket short of min_samples is never reported as a number. That is
    the whole point of the ladder: the alternative is a "confidence: 62%
    (based on 2 similar days)" that reads as a measurement and is a
    coin flip.
    """
    key = "same_day_high" if side == "high" else "same_day_low"
    if summary is None:
        summary = summarize()
    scored = [
        d[key] for d in summary["days"]
        if d[key]["error"] is not None and d[key].get("conditions") is not None
    ]

    tried = []
    for axes in CONDITION_BACKOFF_LEVELS:
        # A day only counts toward a bucket if it recorded every axis that
        # bucket is defined on - see classify_conditions on why a missing
        # index is not "steady".
        def matches(day_conditions):
            for a in axes:
                want = conditions.get(a)
                if want is None or day_conditions.get(a) != want:
                    return False
            return True

        errors = [d["error"] for d in scored if matches(d["conditions"])]
        tried.append((axes, len(errors)))
        if len(errors) >= min_samples:
            stats = _stats(errors)
            return {
                "pct": _mae_to_confidence_pct(stats["mae_f"]),
                "n": stats["n"],
                "mae_f": stats["mae_f"],
                "bias_f": stats["bias_f"],
                "axes": axes,
                "label": describe_conditions(conditions, axes),
                "matched": bool(axes),
                "tried": tried,
            }
    return None


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

    # A date's "final" high/low here is max/min of observed_high_so_far_f
    # across that date's rows - which is only the day's actual extreme once
    # the day is OVER. For the day still in progress it is whatever has
    # happened so far, and scoring against it produces nonsense: at 00:37 on
    # 2026-07-30 the partial high read 60.8F, making the previous day's
    # next-day forecast of 76 look like a +15.2F miss when the day had
    # barely started. That one bogus row moved next_day_high MAE from 1.54
    # to 3.06 - it more than doubled the headline error.
    #
    # So every date is tagged complete/incomplete, and an error is only
    # computed when the date being scored against is complete. Local
    # "today" comes from the log's own timestamps rather than a fresh
    # import, keeping this module free of a weather_estimator dependency.
    today_local = None
    if rows:
        try:
            tzinfo = datetime.fromisoformat(rows[-1]["logged_at"]).tzinfo
            today_local = datetime.now(tzinfo).date().isoformat()
        except Exception as e:
            print(f"calibration_log: could not derive local today ({e}); "
                  "in-progress days will be scored as if complete")

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
        # Conditions as they stood at the moment the prediction being
        # scored was made - the same row the projection itself comes from,
        # not the day's average or its end state. Bucketing by anything
        # else would be scoring the model against information it did not
        # have. See classify_conditions.
        high_conditions = classify_conditions(pre_peak[-1]) if pre_peak else None

        final_low = min(
            (r["observed_low_so_far_f"] for r in day_rows if r["observed_low_so_far_f"] is not None),
            default=None,
        )
        pre_dawn = [r for r in day_rows if r["low_status"] == "today"]
        last_low_projection = pre_dawn[-1]["estimated_low_f"] if pre_dawn else None
        low_conditions = classify_conditions(pre_dawn[-1]) if pre_dawn else None

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

        def err(pred, actual, scored_date):
            if pred is None or actual is None:
                return None
            if today_local is not None and scored_date >= today_local:
                return None  # that day isn't over; its "final" is partial
            return round(pred - actual, 2)

        next_date = dates[i + 1] if i + 1 < len(dates) else None

        days.append({
            "date": date,
            "complete": today_local is None or date < today_local,
            "same_day_high": {"projection": last_high_projection, "final": final_high,
                               "error": err(last_high_projection, final_high, date),
                               "conditions": high_conditions},
            "same_day_low": {"projection": last_low_projection, "final": final_low,
                              "error": err(last_low_projection, final_low, date),
                              "conditions": low_conditions},
            # Scored against the FOLLOWING date, so completeness is that
            # date's, not this one's.
            "next_day_high": {"projection": tomorrow_high_forecast, "final": next_final_high,
                               "error": err(tomorrow_high_forecast, next_final_high, next_date or date),
                               "target_date": next_date,
                               "source": tomorrow_high_source},
            "next_day_low": {"projection": tomorrow_low_forecast, "final": next_final_low,
                              "error": err(tomorrow_low_forecast, next_final_low, next_date or date),
                              "target_date": next_date,
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
