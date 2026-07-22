"""
Turns the dashboard's live snapshots into a running daily record: once a
station-day's actual high and low have both happened, finalize_day()
computes how the model's own pre-peak/pre-dawn prediction compared to
what actually settled (value AND timing), and how Kalshi's market
behaved that day (peak trading activity, what the market thought was
most likely right around the actual peak).

One record per station-day, appended to daily_performance.jsonl (same
append-only-JSONL approach as calibration_log.py - no new persistence
layer introduced). Re-running finalize_day() for an already-finalized
date is a no-op.

Sign convention: peak_temp_error_f and peak_time_error_minutes are both
(predicted - actual) - the same direction as weather_estimator.backtest()'s
error_f ("positive = estimator runs warm"). Positive peak_temp_error_f
means the model's predicted temp ran warm/high vs. what actually
happened; positive peak_time_error_minutes means the model predicted the
peak later than it actually occurred.

Run this file directly to finalize any completed days not yet in the
log:
    python3 daily_performance.py
"""

import json
import os
from datetime import datetime, time, timedelta

from weather_estimator import get_observation_history
from calibration_log import get_last_prediction
from kalshi import (
    HIGH_SERIES,
    LOW_SERIES,
    KALSHI_BASE,
    get_event_ticker_for_any_date,
    get_market_brackets,
    bracket_contains,
)
from paper_trading import resolve_paper_trade
import requests

LOG_PATH = os.path.join(os.path.dirname(__file__), "daily_performance.jsonl")

VOLUME_BUCKET_MINUTES = 15


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


def _already_finalized(station_id, date_str):
    return any(r["station"] == station_id.upper() and r["date"] == date_str for r in _load_rows())


def _peak_time_plausible(side, actual_time):
    """A cheap sanity check, not a fix: weather_estimator's own known
    data-completeness gap (a station query missing an entire morning,
    see kalshi.py's docstring) has previously produced a "high" or "low"
    that lands at an implausible hour - e.g. a summer afternoon high
    recorded at midnight because the real afternoon reading was simply
    missing that day. Rather than silently store that as a normal data
    point, flag it so the weekly/monthly views don't average it in
    uncritically."""
    hour = actual_time.hour
    if side == "high":
        return 9 <= hour <= 21
    return hour <= 11 or hour >= 21


def _observed_temp_at(df, target_time, tolerance_minutes=20):
    """Nearest observation to target_time, or None if the nearest one is
    further away than tolerance_minutes (rather than silently returning a
    stale reading from a data gap)."""
    if df.empty:
        return None
    deltas = (df["time"] - target_time).abs()
    idx = deltas.idxmin()
    if deltas.loc[idx] > timedelta(minutes=tolerance_minutes):
        return None
    return float(df.loc[idx, "temp_f"])


def _full_day_actuals(station_id, day):
    """Actual high/low + timing for a calendar date, from a clean
    midnight-to-midnight observation query - not reused from
    calibration_log's intraday "so far" snapshots, so it isn't affected
    by which moments happened to get logged that day."""
    tzinfo = get_observation_history(station_id, limit=5)["time"].iloc[0].tzinfo
    start = datetime.combine(day, time(0, 0), tzinfo=tzinfo)
    end = start + timedelta(days=1)
    df = get_observation_history(station_id, start=start, end=end)
    if df.empty:
        return None

    high_idx = df["temp_f"].idxmax()
    low_idx = df["temp_f"].idxmin()
    return {
        "df": df,
        "high_temp": float(df.loc[high_idx, "temp_f"]),
        "high_time": df.loc[high_idx, "time"],
        "low_temp": float(df.loc[low_idx, "temp_f"]),
        "low_time": df.loc[low_idx, "time"],
    }


def _bracket_representative_value(bracket):
    """A single representative temperature for a bracket, for comparing
    Kalshi's implied call against an actual numeric temperature. Tail
    brackets are unbounded on one side; nudging one degree in from the
    shared edge with their ranged neighbor (see kalshi.py's
    _bracket_contains fix) gives the value just barely inside that
    bracket's real range."""
    floor = bracket["floor_strike"]
    cap = bracket["cap_strike"]
    if floor is not None and cap is not None:
        return (floor + cap) / 2
    if floor is not None:
        return floor + 1
    if cap is not None:
        return cap - 1
    return None


def _fetch_minute_candles(series_ticker, ticker, start_ts, end_ts):
    r = requests.get(
        f"{KALSHI_BASE}/series/{series_ticker}/markets/{ticker}/candlesticks",
        params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": 1},
    )
    r.raise_for_status()
    return r.json()["candlesticks"]


def _implied_bracket_at(brackets, per_bracket_minutes, target_time, day_start_tzinfo):
    """Whichever bracket had the highest implied probability at the minute
    closest to target_time - each bracket read at its own nearest-in-time
    minute, then argmax by price across brackets."""
    if target_time is None:
        return None
    target_ts = target_time.timestamp()
    candidates = []
    for b in brackets:
        minutes = per_bracket_minutes.get(b["ticker"], [])
        if not minutes:
            continue
        ts, price = min(minutes, key=lambda mp: abs(mp[0] - target_ts))
        candidates.append((price, b, ts))
    if not candidates:
        return None
    price, bracket, ts = max(candidates, key=lambda c: c[0])
    return {
        "bracket": bracket,
        "bracket_label": bracket["label"],
        "value": _bracket_representative_value(bracket),
        "implied_probability": round(price, 4),
        "at_time": datetime.fromtimestamp(ts, tz=day_start_tzinfo).isoformat(),
    }


def _kalshi_day_stats(series_ticker, event_ticker, day_start, day_end, actual_peak_time, predicted_peak_time):
    """
    Peak 15-minute trading-volume window across the whole event that day,
    plus which bracket the market thought was most likely (a) right
    around the actual peak time, and (b) at the model's own prediction
    time - (b) is a cross-check of whether the model's prediction agreed
    with the market's at the moment the model made its call, independent
    of how the day actually turned out.

    Kalshi's candlestick endpoint only accepts period_interval 1 (minute)
    or 60 (hour) - nothing in between (confirmed directly against the
    API; 15/30 are rejected). 15-minute buckets are real per-minute data
    rolled up ourselves, not invented precision the API doesn't have.
    """
    brackets = get_market_brackets(event_ticker)
    start_ts = int(day_start.timestamp())
    end_ts = int(day_end.timestamp())
    bucket_seconds = VOLUME_BUCKET_MINUTES * 60

    volume_buckets = {}  # bucket_start_ts -> {"contracts": float, "dollars": float}
    per_bracket_minutes = {}  # ticker -> list of (minute_ts, close_price)

    for b in brackets:
        try:
            candles = _fetch_minute_candles(series_ticker, b["ticker"], start_ts, end_ts)
        except Exception:
            continue
        minute_prices = []
        for c in candles:
            ts = c["end_period_ts"]
            bucket_start = ts - (ts % bucket_seconds)
            bucket = volume_buckets.setdefault(bucket_start, {"contracts": 0.0, "dollars": 0.0})
            contracts = float(c["volume_fp"])
            price = c["price"].get("mean_dollars") or c["price"].get("close_dollars") or c["price"].get("previous_dollars")
            if contracts > 0 and price is not None:
                bucket["contracts"] += contracts
                bucket["dollars"] += contracts * float(price)
            if price is not None:
                minute_prices.append((ts, float(price)))
        per_bracket_minutes[b["ticker"]] = minute_prices

    peak_volume = None
    if volume_buckets:
        peak_ts = max(volume_buckets, key=lambda t: volume_buckets[t]["contracts"])
        peak_volume = {
            "contracts": round(volume_buckets[peak_ts]["contracts"], 2),
            "dollars": round(volume_buckets[peak_ts]["dollars"], 2),
            "bucket_start": datetime.fromtimestamp(peak_ts, tz=day_start.tzinfo).isoformat(),
        }

    implied_at_actual = _implied_bracket_at(brackets, per_bracket_minutes, actual_peak_time, day_start.tzinfo)
    implied_at_predicted = _implied_bracket_at(brackets, per_bracket_minutes, predicted_peak_time, day_start.tzinfo)

    return peak_volume, implied_at_actual, implied_at_predicted


def _finalize_side(station_id, day, side, actuals, prediction, series_ticker, tzinfo):
    """side: 'high' or 'low'. Builds one side's full record - temp/time
    errors always computed from real data; Kalshi fields are best-effort
    (None if that day's event can't be found or the API call fails,
    rather than blocking the whole finalization)."""
    actual_temp = actuals[f"{side}_temp"]
    actual_time = actuals[f"{side}_time"]
    df = actuals["df"]

    predicted_temp = prediction["temp_f"] if prediction else None
    predicted_time = datetime.fromisoformat(prediction["time"]) if prediction else None

    temp_1hr_before_actual = _observed_temp_at(df, actual_time - timedelta(hours=1))
    temp_1hr_before_predicted = (
        _observed_temp_at(df, predicted_time - timedelta(hours=1)) if predicted_time is not None else None
    )

    peak_temp_error_f = round(predicted_temp - actual_temp, 2) if predicted_temp is not None else None
    peak_time_error_minutes = (
        round((predicted_time - actual_time).total_seconds() / 60, 1) if predicted_time is not None else None
    )

    record = {
        "predicted_peak_time": predicted_time.isoformat() if predicted_time is not None else None,
        "predicted_peak_temp": predicted_temp,
        "temp_1hr_before_predicted_peak": temp_1hr_before_predicted,
        "actual_peak_time": actual_time.isoformat(),
        "actual_peak_temp": actual_temp,
        "temp_1hr_before_actual_peak": temp_1hr_before_actual,
        "peak_time_error_minutes": peak_time_error_minutes,
        "peak_temp_error_f": peak_temp_error_f,
        "data_quality_flag": (
            None if _peak_time_plausible(side, actual_time)
            else "peak time landed at an implausible hour - likely a station data gap, not a real physical peak"
        ),
        "kalshi_peak_volume_contracts": None,
        "kalshi_peak_volume_dollars_est": None,
        "kalshi_peak_volume_time": None,
        "kalshi_market_implied_bracket": None,
        "kalshi_market_implied_value": None,
        "kalshi_market_implied_probability": None,
        "kalshi_market_implied_at_time": None,
        # Market's implied call at the model's own prediction moment (not
        # at the actual peak) - lets the monthly rollup check how often
        # the model's predicted temp agreed with what the market thought
        # was most likely at that same moment, independent of outcome.
        "kalshi_implied_at_prediction_bracket": None,
        "kalshi_implied_at_prediction_value": None,
        "kalshi_implied_at_prediction_probability": None,
        "predicted_within_kalshi_implied_bracket": None,
        # Paper trade (simulated only, no real order): None if no bet was
        # placed that day/side (e.g. no edge cleared MIN_EDGE), otherwise
        # resolve_paper_trade's output merged in directly.
        "simulated_bucket_chosen": None,
        "simulated_entry_price": None,
        "simulated_stake": None,
        "model_implied_probability": None,
        "market_implied_probability": None,
        "outcome_bucket": None,
        "simulated_payout": None,
    }

    resolved_bet = resolve_paper_trade(day.isoformat(), side, actual_temp)
    if resolved_bet is not None:
        record["simulated_bucket_chosen"] = resolved_bet["simulated_bucket_chosen"]
        record["simulated_entry_price"] = resolved_bet["simulated_entry_price"]
        record["simulated_stake"] = resolved_bet["simulated_stake"]
        record["model_implied_probability"] = resolved_bet["model_implied_probability"]
        record["market_implied_probability"] = resolved_bet["market_implied_probability"]
        record["outcome_bucket"] = resolved_bet["outcome_bucket"]
        record["simulated_payout"] = resolved_bet["simulated_payout"]

    try:
        event_ticker = get_event_ticker_for_any_date(series_ticker, day)
        if event_ticker is not None:
            day_start = datetime.combine(day, time(0, 0), tzinfo=tzinfo)
            day_end = day_start + timedelta(days=1)
            peak_volume, implied_actual, implied_predicted = _kalshi_day_stats(
                series_ticker, event_ticker, day_start, day_end, actual_time, predicted_time
            )
            if peak_volume is not None:
                record["kalshi_peak_volume_contracts"] = peak_volume["contracts"]
                record["kalshi_peak_volume_dollars_est"] = peak_volume["dollars"]
                record["kalshi_peak_volume_time"] = peak_volume["bucket_start"]
            if implied_actual is not None:
                record["kalshi_market_implied_bracket"] = implied_actual["bracket_label"]
                record["kalshi_market_implied_value"] = implied_actual["value"]
                record["kalshi_market_implied_probability"] = implied_actual["implied_probability"]
                record["kalshi_market_implied_at_time"] = implied_actual["at_time"]
            if implied_predicted is not None:
                record["kalshi_implied_at_prediction_bracket"] = implied_predicted["bracket_label"]
                record["kalshi_implied_at_prediction_value"] = implied_predicted["value"]
                record["kalshi_implied_at_prediction_probability"] = implied_predicted["implied_probability"]
                if predicted_temp is not None:
                    # Did the model's own predicted temp fall inside the
                    # SAME bracket the market considered most likely, at
                    # the moment the model made its call? Not a label
                    # comparison - a real floor/cap containment check.
                    record["predicted_within_kalshi_implied_bracket"] = bracket_contains(
                        implied_predicted["bracket"], predicted_temp
                    )
    except Exception as e:
        record["kalshi_error"] = str(e)

    return record


def finalize_day(station_id, day):
    """
    Build and append one performance record for a station-day, if that
    day is actually over and hasn't already been finalized. Returns the
    record, or None if skipped (day not over yet, or already finalized).
    """
    date_str = day.isoformat()
    if _already_finalized(station_id, date_str):
        return None

    sample = get_observation_history(station_id, limit=5)
    tzinfo = sample["time"].iloc[0].tzinfo
    today_local = datetime.now(tzinfo).date()
    if day >= today_local:
        return None  # day isn't over yet

    actuals_high = _full_day_actuals(station_id, day)
    if actuals_high is None:
        return None  # no observation data at all for that date - can't finalize

    prediction = get_last_prediction(date_str)

    high_record = _finalize_side(station_id, day, "high", actuals_high, prediction["high"], HIGH_SERIES, tzinfo)
    low_record = _finalize_side(station_id, day, "low", actuals_high, prediction["low"], LOW_SERIES, tzinfo)

    full_record = {
        "station": station_id.upper(),
        "date": date_str,
        "finalized_at": datetime.now(tzinfo).isoformat(),
        "high": high_record,
        "low": low_record,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(full_record) + "\n")
    return full_record


def finalize_pending_days(station_id, lookback_days=7):
    """Catches up on any completed-but-not-yet-finalized days in the
    trailing window - safe to call on every dashboard refresh."""
    sample = get_observation_history(station_id, limit=5)
    tzinfo = sample["time"].iloc[0].tzinfo
    today_local = datetime.now(tzinfo).date()

    finalized = []
    for i in range(1, lookback_days + 1):
        day = today_local - timedelta(days=i)
        try:
            record = finalize_day(station_id, day)
        except Exception as e:
            print(f"daily_performance: failed to finalize {station_id} {day}: {e}")
            continue
        if record is not None:
            finalized.append(record)
    return finalized


def _load_records(station_id):
    return [r for r in _load_rows() if r["station"] == station_id.upper()]


def weekly_table(station_id, days=7):
    """
    Trailing N-day rolling window (not a fixed calendar week) of finalized
    records, most recent first. days_with_data lets the caller show "3 of
    7 days" instead of silently presenting a partial week as if it were
    complete.
    """
    sample = get_observation_history(station_id, limit=5)
    tzinfo = sample["time"].iloc[0].tzinfo
    today_local = datetime.now(tzinfo).date()

    records_by_date = {r["date"]: r for r in _load_records(station_id)}
    rows = []
    for i in range(1, days + 1):
        d = (today_local - timedelta(days=i)).isoformat()
        rows.append(records_by_date.get(d, {"date": d, "high": None, "low": None}))

    finalized_rows = [r for r in rows if r.get("high") is not None or r.get("low") is not None]
    return {
        "days_requested": days,
        "days_with_data": len(finalized_rows),
        "rows": rows,
        "paper_trading": _paper_trading_stats(finalized_rows),
    }


def _mean(xs):
    return round(sum(xs) / len(xs), 2) if xs else None


def _mean_abs(xs):
    return round(sum(abs(x) for x in xs) / len(xs), 2) if xs else None


def _paper_trading_stats(records):
    """
    Aggregates simulated bets across a set of finalized-day records (both
    sides). total_pnl/win_rate answer "would this have made money"; the
    model-vs-market comparison answers the actual question this feature
    exists to test - on the days the model's and Kalshi's probabilities
    disagreed most, which one ended up closer to the real outcome?
    """
    bets = []
    for r in records:
        for side in ("high", "low"):
            rec = r.get(side)
            if rec is None or rec.get("simulated_payout") is None:
                continue
            bets.append({
                "date": r["date"], "side": side,
                "payout": rec["simulated_payout"],
                "hit": rec["hit"],
                "model_p": rec["model_implied_probability"],
                "market_p": rec["market_implied_probability"],
            })

    if not bets:
        return {"n_bets": 0, "total_pnl": None, "win_rate": None, "model_vs_market": None}

    total_pnl = round(sum(b["payout"] for b in bets), 2)
    win_rate = round(sum(1 for b in bets if b["hit"]) / len(bets), 3)

    for b in bets:
        truth = 1.0 if b["hit"] else 0.0
        b["model_dist"] = abs(b["model_p"] - truth)
        b["market_dist"] = abs(b["market_p"] - truth)
        b["disagreement"] = abs(b["model_p"] - b["market_p"])

    # The more-disagreeing half of bets - whoever's probability ended up
    # closer to the actual outcome on THOSE days "won" that comparison.
    ranked = sorted(bets, key=lambda b: b["disagreement"], reverse=True)
    top_n = ranked[:max(1, len(ranked) // 2)]
    model_closer = sum(1 for b in top_n if b["model_dist"] < b["market_dist"])
    market_closer = sum(1 for b in top_n if b["market_dist"] < b["model_dist"])

    return {
        "n_bets": len(bets),
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "model_vs_market": {
            "n_high_disagreement_bets": len(top_n),
            "model_closer_count": model_closer,
            "market_closer_count": market_closer,
        },
    }


def _side_month_stats(records, side):
    """Reuses backtest()'s mae_f/bias_f naming AND sign convention
    (weather_estimator.py) - see this module's docstring."""
    temp_errors = [r[side]["peak_temp_error_f"] for r in records if r[side]["peak_temp_error_f"] is not None]
    time_errors = [r[side]["peak_time_error_minutes"] for r in records if r[side]["peak_time_error_minutes"] is not None]
    volumes = [r[side]["kalshi_peak_volume_contracts"] for r in records if r[side]["kalshi_peak_volume_contracts"] is not None]
    hits = [r[side]["predicted_within_kalshi_implied_bracket"] for r in records if r[side]["predicted_within_kalshi_implied_bracket"] is not None]

    return {
        "n_temp_error_samples": len(temp_errors),
        "bias_f": _mean(temp_errors),
        "mae_f": _mean_abs(temp_errors),
        "n_time_error_samples": len(time_errors),
        "time_bias_minutes": _mean(time_errors),
        "time_mae_minutes": _mean_abs(time_errors),
        "mean_kalshi_peak_volume_contracts": _mean(volumes),
        "n_hit_samples": len(hits),
        "model_in_kalshi_bracket_rate": round(sum(hits) / len(hits), 3) if hits else None,
    }


def monthly_rollup(station_id, year, month):
    """Calendar-month averages (not rolling 30 days) for high and low
    separately. low_sample flags months with fewer than 5 finalized days
    so a thin average isn't shown with the same confidence as a full
    month's worth."""
    prefix = f"{year:04d}-{month:02d}"
    records = [r for r in _load_records(station_id) if r["date"].startswith(prefix)]

    return {
        "year": year,
        "month": month,
        "days_with_data": len(records),
        "low_sample": len(records) < 5,
        "high": _side_month_stats(records, "high"),
        "low": _side_month_stats(records, "low"),
        "paper_trading": _paper_trading_stats(records),
    }


if __name__ == "__main__":
    finalized = finalize_pending_days("KSEA")
    if not finalized:
        print("Nothing new to finalize.")
    for r in finalized:
        print(json.dumps(r, indent=2))
