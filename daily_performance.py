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

"Actual" source: actual_peak_temp prefers NWS's official CLI climate
report (nws_climate.py) - the same source Kalshi itself settles
against - falling back to the raw observation stream only when CLI
isn't retained/published yet for that date. actual_peak_temp_source
("cli" or "stream") tags which one won, and
actual_peak_temp_stream_f/actual_peak_temp_cli_f keep BOTH values
around regardless of which was picked, so accuracy stats never
silently mix methodologies and a stream-fallback day can be identified
and later upgraded (see reconcile_stream_fallback_actuals below) once
CLI becomes available. This also drives what paper_trading.py's bets
get resolved against, for the same reason.

Run this file directly to finalize any completed days not yet in the
log:
    python3 daily_performance.py
"""

import json
import os
from datetime import date, datetime, time, timedelta

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
from nws_climate import fetch_recent_cli_finals
from paper_trading import resolve_paper_trade, LEAD_TIME_HINTS
import requests

LOG_PATH = os.path.join(os.path.dirname(__file__), "daily_performance.jsonl")

VOLUME_BUCKET_MINUTES = 15

# Reused everywhere a rollup needs to flag "too few observations to mean
# anything" rather than present a number with false confidence - monthly_
# rollup's own low_sample flag, and now each paper-trading lead time's too.
LOW_SAMPLE_THRESHOLD = 5


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


def _finalize_side(station_id, day, side, actuals, prediction, series_ticker, tzinfo, cli):
    """side: 'high' or 'low'. Builds one side's full record - temp/time
    errors always computed from real data; Kalshi fields are best-effort
    (None if that day's event can't be found or the API call fails,
    rather than blocking the whole finalization).

    cli: that day's fetch_recent_cli_finals() entry, or None if CLI
    hasn't got a final report for this date retained. actual_time
    always comes from the observation stream regardless of temp
    source - CLI never publishes a time-of-day for its max/min (see
    nws_climate.py) - only the temperature VALUE prefers CLI."""
    stream_temp = actuals[f"{side}_temp"]
    actual_time = actuals[f"{side}_time"]
    df = actuals["df"]

    cli_temp = cli.get(f"{side}_f") if cli else None
    if cli_temp is not None:
        actual_temp = cli_temp
        actual_temp_source = "cli"
    else:
        actual_temp = stream_temp
        actual_temp_source = "stream"

    predicted_temp = prediction["temp_f"] if prediction else None
    predicted_time = datetime.fromisoformat(prediction["time"]) if prediction else None
    nws_forecast_temp = prediction.get("nws_forecast_f") if prediction else None

    temp_1hr_before_actual = _observed_temp_at(df, actual_time - timedelta(hours=1))
    temp_1hr_before_predicted = (
        _observed_temp_at(df, predicted_time - timedelta(hours=1)) if predicted_time is not None else None
    )

    peak_temp_error_f = round(predicted_temp - actual_temp, 2) if predicted_temp is not None else None
    peak_time_error_minutes = (
        round((predicted_time - actual_time).total_seconds() / 60, 1) if predicted_time is not None else None
    )
    # Same sign convention as peak_temp_error_f (predicted/forecast minus
    # actual) so the two are directly comparable - see module docstring
    # and calibration_log.py's nws_high/low_forecast_at_target_f (captured
    # at the SAME checkpoint as predicted_temp, for the same target time).
    nws_forecast_error_f = round(nws_forecast_temp - actual_temp, 2) if nws_forecast_temp is not None else None

    record = {
        "predicted_peak_time": predicted_time.isoformat() if predicted_time is not None else None,
        "predicted_peak_temp": predicted_temp,
        "temp_1hr_before_predicted_peak": temp_1hr_before_predicted,
        "nws_forecast_temp": nws_forecast_temp,
        "nws_forecast_error_f": nws_forecast_error_f,
        "actual_peak_time": actual_time.isoformat(),
        "actual_peak_temp": actual_temp,
        "actual_peak_temp_source": actual_temp_source,
        "actual_peak_temp_stream_f": stream_temp,
        "actual_peak_temp_cli_f": cli_temp,
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
        # Paper trades (simulated only, no real order), kept SEPARATE per
        # lead_time_hint - "1hr" and "2hr" are two independent experimental
        # legs being compared, never averaged/combined (see
        # paper_trading.py's module docstring). None for a given lead time
        # means no bet was placed that day/side/lead-time (e.g. no edge
        # cleared MIN_EDGE at that particular moment); otherwise
        # resolve_paper_trade's output for that lead time merged in
        # directly.
        "paper_trades": {lt: None for lt in LEAD_TIME_HINTS},
    }

    resolved_bets = resolve_paper_trade(day.isoformat(), side, actual_temp)
    for lead_time_hint, resolved_bet in resolved_bets.items():
        if resolved_bet is None:
            continue
        record["paper_trades"][lead_time_hint] = {
            "simulated_bucket_chosen": resolved_bet["simulated_bucket_chosen"],
            "simulated_entry_price": resolved_bet["simulated_entry_price"],
            "simulated_stake": resolved_bet["simulated_stake"],
            "model_implied_probability": resolved_bet["model_implied_probability"],
            "market_implied_probability": resolved_bet["market_implied_probability"],
            "edge_at_entry": resolved_bet.get("edge_at_entry"),
            "outcome_bucket": resolved_bet["outcome_bucket"],
            "hit": resolved_bet["hit"],
            "simulated_payout": resolved_bet["simulated_payout"],
            "within_uncertainty_band": resolved_bet.get("within_uncertainty_band"),
        }

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
    now_local = datetime.now(tzinfo)
    today_local = now_local.date()
    if day >= today_local:
        return None  # day isn't over yet
    if day == today_local - timedelta(days=1) and now_local.time() < time(2, 0):
        # NWS's FINAL CLI report for "yesterday" is typically published
        # ~1:15-1:30am local (see nws_climate.py) - waiting until 2am
        # before finalizing the very next day avoids an avoidable
        # stream-fallback in the narrow window right after midnight when
        # it just hasn't landed yet. Days further back are unaffected -
        # CLI would already be out for them one way or another by now.
        return None

    actuals_high = _full_day_actuals(station_id, day)
    if actuals_high is None:
        return None  # no observation data at all for that date - can't finalize

    prediction = get_last_prediction(date_str)
    cli = fetch_recent_cli_finals().get(day)

    high_record = _finalize_side(station_id, day, "high", actuals_high, prediction["high"], HIGH_SERIES, tzinfo, cli)
    low_record = _finalize_side(station_id, day, "low", actuals_high, prediction["low"], LOW_SERIES, tzinfo, cli)

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


def reconcile_stream_fallback_actuals(station_id, lookback_days=5):
    """
    Revisits recently-finalized days whose actual_peak_temp_source is
    "stream" (or predates that field, from before this module tracked
    it - same thing, since the stream was the only source that existed
    then) and upgrades them to CLI if a final report has since become
    available for that date. This is the one deliberate exception to
    this log's otherwise-append-only pattern (see finalize_day): a
    stream-fallback record is a best-effort placeholder, not a
    permanent decision, because paper_trading.py's bet resolution
    needs to reflect CLI - the real settlement source - once it
    exists, even at the cost of re-writing an already-written row (see
    paper_trading.py's module docstring on why accuracy there matters
    more than immediacy). Re-finalizing a side rebuilds its whole
    record, so this also naturally corrects the accuracy-tracking
    fields (peak_temp_error_f etc.) for that side, not just the paper
    trade.

    Cheap to call on every refresh: once a day's actual has converged
    to "cli" (or the retention window has passed with no final report
    ever appearing - the two look identical from here, so both are
    simply retried next time at near-zero cost), it's never touched
    again.

    Returns the list of (date_str, side) pairs actually changed.
    """
    cli_by_date = fetch_recent_cli_finals()
    if not cli_by_date:
        return []

    rows = _load_rows()
    sample = get_observation_history(station_id, limit=5)
    tzinfo = sample["time"].iloc[0].tzinfo
    today_local = datetime.now(tzinfo).date()

    changed = []
    for row in rows:
        if row["station"] != station_id.upper():
            continue
        day = date.fromisoformat(row["date"])
        if (today_local - day).days > lookback_days:
            continue
        cli = cli_by_date.get(day)
        if cli is None:
            continue

        prediction = None
        for side, series_ticker in (("high", HIGH_SERIES), ("low", LOW_SERIES)):
            side_record = row.get(side)
            if side_record is None or side_record.get("actual_peak_temp_source", "stream") != "stream":
                continue
            if cli.get(f"{side}_f") is None:
                continue  # this specific side is itself "MM" in the CLI report - nothing to upgrade to

            actuals = _full_day_actuals(station_id, day)
            if actuals is None:
                continue
            if prediction is None:
                prediction = get_last_prediction(row["date"])
            row[side] = _finalize_side(
                station_id, day, side, actuals, prediction[side], series_ticker, tzinfo, cli
            )
            changed.append((row["date"], side))

    if changed:
        with open(LOG_PATH, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    return changed


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
    Aggregates simulated bets across a set of finalized-day records, kept
    SEPARATE per lead_time_hint ("1hr" vs "2hr") - never averaged or
    combined into one number. That separation is the entire point of
    running both in parallel (see paper_trading.py's module docstring):
    whether betting closer to peak or further out actually performs
    better is meant to be answered by comparing these two blocks against
    each other, not by blending them into a single stat that erases the
    comparison. Returns {"1hr": {...}, "2hr": {...}}, each shaped like
    _paper_trading_stats_for_lead_time's return.
    """
    return {lt: _paper_trading_stats_for_lead_time(records, lt) for lt in LEAD_TIME_HINTS}


def _paper_trading_stats_for_lead_time(records, lead_time_hint):
    """
    total_pnl/win_rate answer "would this have made money"; avg_edge_at_
    entry is the raw mispricing the bet was based on; pct_within_
    uncertainty_band checks whether the model's stated confidence band,
    specifically at THIS lead time, was honestly calibrated (a band
    evaluated 1hr before peak isn't necessarily as well-calibrated as one
    evaluated 2hr before - that's exactly the kind of thing this
    comparison exists to surface); model_vs_market answers the actual
    question the wider feature exists to test - on the days the model's
    and Kalshi's probabilities disagreed most, which one ended up closer
    to the real outcome? low_sample flags fewer than LOW_SAMPLE_THRESHOLD
    bets, same guard as monthly_rollup's own low_sample - the caller
    should show "insufficient data" rather than a percentage this thin.
    """
    bets = []
    for r in records:
        for side in ("high", "low"):
            rec = r.get(side)
            if rec is None:
                continue
            trade = (rec.get("paper_trades") or {}).get(lead_time_hint)
            if trade is None or trade.get("simulated_payout") is None:
                continue
            bets.append({
                "date": r["date"], "side": side,
                "payout": trade["simulated_payout"],
                "hit": trade["hit"],
                "model_p": trade["model_implied_probability"],
                "market_p": trade["market_implied_probability"],
                "edge_at_entry": trade.get("edge_at_entry"),
                "within_band": trade.get("within_uncertainty_band"),
            })

    if not bets:
        return {
            "n_bets": 0, "total_pnl": None, "win_rate": None,
            "avg_edge_at_entry": None, "pct_within_uncertainty_band": None,
            "model_vs_market": None, "low_sample": True,
        }

    total_pnl = round(sum(b["payout"] for b in bets), 2)
    win_rate = round(sum(1 for b in bets if b["hit"]) / len(bets), 3)
    avg_edge_at_entry = _mean([b["edge_at_entry"] for b in bets if b["edge_at_entry"] is not None])

    band_samples = [b["within_band"] for b in bets if b["within_band"] is not None]
    pct_within_uncertainty_band = round(sum(band_samples) / len(band_samples), 3) if band_samples else None

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
        "avg_edge_at_entry": avg_edge_at_entry,
        "pct_within_uncertainty_band": pct_within_uncertainty_band,
        "model_vs_market": {
            "n_high_disagreement_bets": len(top_n),
            "model_closer_count": model_closer,
            "market_closer_count": market_closer,
        },
        "low_sample": len(bets) < LOW_SAMPLE_THRESHOLD,
    }


def _side_month_stats(records, side):
    """Reuses backtest()'s mae_f/bias_f naming AND sign convention
    (weather_estimator.py) - see this module's docstring."""
    temp_errors = [r[side]["peak_temp_error_f"] for r in records if r[side]["peak_temp_error_f"] is not None]
    time_errors = [r[side]["peak_time_error_minutes"] for r in records if r[side]["peak_time_error_minutes"] is not None]
    volumes = [r[side]["kalshi_peak_volume_contracts"] for r in records if r[side]["kalshi_peak_volume_contracts"] is not None]
    hits = [r[side]["predicted_within_kalshi_implied_bracket"] for r in records if r[side]["predicted_within_kalshi_implied_bracket"] is not None]
    # Older records predate this field - .get() so this replays fine
    # against an existing log that has rows from before it was added.
    nws_errors = [r[side].get("nws_forecast_error_f") for r in records if r[side].get("nws_forecast_error_f") is not None]

    return {
        "n_temp_error_samples": len(temp_errors),
        "bias_f": _mean(temp_errors),
        "mae_f": _mean_abs(temp_errors),
        "n_nws_error_samples": len(nws_errors),
        "nws_bias_f": _mean(nws_errors),
        "nws_mae_f": _mean_abs(nws_errors),
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
