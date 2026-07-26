"""
Paper-trades the model's Kalshi picks - simulated bets only. Tests
whether the model's own probability estimate would have beaten Kalshi's
market price, before ever considering a real order. No order-placement
API call is made anywhere in this module.

Lead-time comparison: every bet is placed under one of two independent
LEAD_TIME_HINTS - "1hr" (close to peak, sharper prediction but nearer the
model's own low-confidence turning point and a more efficiently-priced
market) or "2hr" (further out, less certain prediction but a market
that's had less time to price in whatever the model sees). Which one
actually performs better isn't resolvable by reasoning - it's why both
run in parallel rather than picking one: daily_performance.py's
_paper_trading_stats keeps every rollup split by lead_time_hint, never
averaged/combined, so the calibration log itself can answer the question
once enough bets accumulate for each.

Timing: "1hr" piggybacks on peak_alerts.py's existing "1hr before
predicted peak" scheduling (same one-shot Routine fire) rather than
building a second mechanism for it - see build_dashboard.py's
ALERT_SCHEDULE_NEEDED integration and the hourly Routine's prompt, which
creates one one-shot trigger per side that runs both peak_alerts.py
check and this module's place command at the same moment. "2hr" gets its
own, simpler lock-once schedule (get_or_lock_2hr_targets /
PAPER_TRADE_2HR_SCHEDULE_NEEDED below) - it's not gated by trend
confidence the way peak_alerts.py's text-alert window is, since this is
a data-gathering experiment, not a user-facing message; a fixed
peak-minus-2-hours instant is all the comparison needs.

Storage: a bet is placed intraday (near peak time) but can only be
resolved once the actual peak is known, which isn't until
daily_performance.py's finalize_day() runs - well after the bet exists.
paper_trades_pending.json bridges that gap (a small, transient staging
file, not a second permanent store): place_paper_trade() records the bet
there (now keyed by lead_time_hint under each side, so "1hr" and "2hr"
coexist without colliding), and finalize_day() calls resolve_paper_trade()
to compute the outcome for BOTH lead times and folds the resolved fields
directly into the SAME daily_performance.jsonl record it already writes -
per the spec, no separate tracking table.

Settlement value: resolve_paper_trade() is handed whatever
daily_performance.py decided "actual_temp" is for that side (NWS's CLI
report if available at finalize time, else the observation stream -
see daily_performance.py's module docstring and nws_climate.py). A bet
finalized before CLI was retained/published gets resolved off the
stream value initially, then daily_performance.reconcile_stream_
fallback_actuals() re-calls this same function with the CLI value once
it becomes available and overwrites the stored resolution - resolve_
paper_trade() itself is pure (bet parameters in, outcome out, no
side effects), so nothing special is needed here to support that; the
reconciliation lives entirely on daily_performance.py's side.

Probability model: the model's own uncertainty band is not a calibrated
confidence interval as-is - weather_estimator.backtest()'s own
pct_within_uncertainty_band shows real coverage well below what the band
nominally targets (~0.36 actual vs. a ~0.9 aim, as of this writing).
Treating the raw band as if it were already an honest interval would
make the model look more confident than its track record supports, so a
Normal distribution's width is solved to match that empirical coverage
rate instead, and bucket probabilities are priced off THAT calibrated
distribution. The band bounds at entry are stored on the bet too, so
resolution can also check whether the actual temp landed inside them -
per lead time, since a stated band evaluated 1hr vs 2hr before peak isn't
necessarily equally well-calibrated at both.

Edge threshold: MIN_EDGE below is a deliberate, adjustable choice (not a
derived constant) - "don't force a bet without a real edge," per the
spec, needs some numeric cutoff, and 15 percentage points was picked as
a conservative bar given the model's still-modest calibration. Both lead
times apply this same threshold independently - it's fine for one to
clear it and not the other on a given day.

Edge logging: MIN_EDGE being conservative means most days place zero
bets, which starves daily_performance.py's win-rate stats but also
throws away the one thing that could tell us whether 15 points is
actually well-calibrated - how big the model/Kalshi disagreement was on
the days that DIDN'T clear it. So every time place_paper_trade evaluates
a market (whether or not the edge clears MIN_EDGE), the best edge found
is appended to EDGE_LOG_PATH - deliberately every evaluation, not just
"close" ones, since deciding what counts as "close" is the open question
here and pre-filtering on a second guessed cutoff would just hide the
same problem one level down. `python3 paper_trading.py edge-stats`
summarizes that log's distribution.

Second strategy, LOW MARKET ONLY: place_unconditional_low_bet() bets
every single day on whichever bracket the model's own point estimate
falls into, with no edge threshold at all - the opposite decision rule
from place_paper_trade's MIN_EDGE gate. It answers a different
question than the edge-gated bets do: does simply following the
model's own call every day beat (or lose to) only betting when a real
mispricing is detected? Every bet - both strategies - carries a
trigger_type field ("edge" or "unconditional") precisely so the two
populations are never pooled into one win-rate/P&L number (same
never-combine principle as LEAD_TIME_HINTS). High-side markets don't
get an unconditional mode - only low, per the spec this was built to;
nothing here stops adding one for high later, but nothing calls it
today. It fires once per day, timed off the SAME ~1hr-before-predicted-
low checkpoint window peak_alerts.py already locks in for the low
side's text alert and the existing edge-gated "1hr" bet - no separate
scheduling exists for it, and it's just as idempotent (day_pending's
"low_unconditional" key, checked before placing).

Real-money guardrails (documented, NOT implemented - this module never
places a real order):
  - hard daily cap on number of real bets and total dollar exposure
  - idempotency: a real order must never fire twice for the same
    date + market
  - a kill switch (env var, default OFF) checked before every real
    order call
  - flipping that switch on is a manual, explicit human decision -
    never auto-enabled based on paper-trading performance looking good

CLI:
    python3 paper_trading.py place high          # defaults to 1hr
    python3 paper_trading.py place high 2hr
    python3 paper_trading.py place low
    python3 paper_trading.py place low 2hr
    python3 paper_trading.py place-unconditional-low  # low only, no edge gate
    python3 paper_trading.py lock-2hr             # lock in today's 2hr targets
    python3 paper_trading.py status
    python3 paper_trading.py edge-stats           # distribution of every edge found, bet or not
"""

import json
import math
import os
from datetime import datetime, timedelta

from weather_estimator import estimate_daily_extremes, estimate_temp, backtest
from kalshi import HIGH_SERIES, LOW_SERIES, get_market_for_date, bracket_contains

STATION = "KSEA"
PENDING_PATH = os.path.join(os.path.dirname(__file__), "paper_trades_pending.json")
SCHEDULE_2HR_PATH = os.path.join(os.path.dirname(__file__), "paper_trading_2hr_schedule.json")
EDGE_LOG_PATH = os.path.join(os.path.dirname(__file__), "paper_trading_edge_log.jsonl")

LEAD_TIME_HINTS = ("1hr", "2hr")
LEAD_2HR_HOURS = 2

STAKE = 0.50
MIN_EDGE = 0.15
FALLBACK_COVERAGE = 0.40  # used only if a fresh backtest can't be computed


def _normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _normal_ppf(p, tol=1e-6):
    """Inverse of _normal_cdf via bisection - avoids adding a scipy
    dependency for this one lookup."""
    lo, hi = -8.0, 8.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def _empirical_band_coverage(station_id):
    """Real fraction of outcomes the model's stated band actually
    covers (from a fresh backtest), used to calibrate the probability
    distribution below to the model's real track record rather than
    its nominal target."""
    try:
        result = backtest(station_id, hours_ahead=3, window_obs=8, lookback_days=5)
        coverage = result["pct_within_uncertainty_band"]
        if 0 < coverage < 1:
            return coverage
    except Exception:
        pass
    return FALLBACK_COVERAGE


def _calibrated_sigma(half_band, coverage):
    """Sigma such that Normal(mean, sigma) puts `coverage` fraction of
    its mass within +/- half_band - the band width recalibrated to the
    model's actual historical accuracy."""
    z = _normal_ppf((1 + coverage) / 2)
    if z <= 0:
        z = 0.01
    return half_band / z


def _bucket_probability(mean, sigma, floor, cap):
    lo = -math.inf if floor is None else floor
    hi = math.inf if cap is None else cap
    p_hi = 1.0 if hi == math.inf else _normal_cdf((hi - mean) / sigma)
    p_lo = 0.0 if lo == -math.inf else _normal_cdf((lo - mean) / sigma)
    return max(0.0, p_hi - p_lo)


def _load_pending():
    if not os.path.exists(PENDING_PATH):
        return {}
    with open(PENDING_PATH) as f:
        return json.load(f)


def _save_pending(pending):
    with open(PENDING_PATH, "w") as f:
        json.dump(pending, f, indent=2)


def _log_edge_evaluation(record):
    with open(EDGE_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _load_edge_log():
    if not os.path.exists(EDGE_LOG_PATH):
        return []
    records = []
    with open(EDGE_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def edge_log_stats():
    """
    Summarizes the edge log so MIN_EDGE can be judged against real data
    instead of a guess: how often the model finds an edge at all, how big
    those edges typically are, and specifically how close the near-misses
    (edge found but below MIN_EDGE) got - the population MIN_EDGE would
    need to move to catch.
    """
    records = _load_edge_log()
    if not records:
        return {"count": 0}

    edges = [r["edge"] for r in records]
    no_bet = [r for r in records if not r["bet_placed"]]
    no_bet_edges = [r["edge"] for r in no_bet]

    buckets = [(-math.inf, 0.0), (0.0, 0.05), (0.05, 0.10), (0.10, 0.13), (0.13, MIN_EDGE), (MIN_EDGE, math.inf)]
    histogram = {}
    for lo, hi in buckets:
        label = f"{lo if lo != -math.inf else '<0'}-{hi if hi != math.inf else '+'}"
        histogram[label] = sum(1 for e in edges if lo <= e < hi or (hi == math.inf and e >= lo))

    closest_near_misses = sorted(no_bet, key=lambda r: r["edge"], reverse=True)[:10]

    return {
        "count": len(records),
        "bets_placed": len(records) - len(no_bet),
        "no_bet": len(no_bet),
        "min_edge_threshold": MIN_EDGE,
        "edge_min": round(min(edges), 4),
        "edge_max": round(max(edges), 4),
        "edge_mean": round(sum(edges) / len(edges), 4),
        "no_bet_edge_mean": round(sum(no_bet_edges) / len(no_bet_edges), 4) if no_bet_edges else None,
        "histogram": histogram,
        "closest_near_misses": [
            {
                "date": r["date"], "side": r["side"], "lead_time_hint": r["lead_time_hint"],
                "edge": r["edge"], "bucket_label": r["bucket_label"],
            }
            for r in closest_near_misses
        ],
    }


def place_paper_trade(station_id, side, lead_time_hint="1hr"):
    """
    Compares the model's own calibrated probability against Kalshi's
    current price for every bucket in today's market, and - only if the
    biggest mispricing clears MIN_EDGE - records a simulated $STAKE bet
    on the most-underpriced bucket. Returns the bet dict, or None if no
    bet was placed (no edge, market unavailable, or already placed today
    for this side + lead_time_hint).

    lead_time_hint doesn't change how the bet is computed - hours_ahead is
    still derived from however far the CURRENT moment actually is from the
    predicted peak, whatever that happens to be when this is called. It
    only tags which experimental leg the resulting bet belongs to, so
    calling this ~1hr before peak vs. ~2hr before peak (see this module's
    two independent schedules) naturally produces a "1hr" or "2hr" bet
    without any separate branch here.
    """
    if lead_time_hint not in LEAD_TIME_HINTS:
        raise ValueError(f"lead_time_hint must be one of {LEAD_TIME_HINTS}, got {lead_time_hint!r}")

    extremes = estimate_daily_extremes(station_id)
    now = extremes["as_of"]
    date_str = now.date().isoformat()

    pending = _load_pending()
    day_pending = pending.get(date_str, {})
    side_pending = day_pending.get(side, {})
    if lead_time_hint in side_pending:
        return None

    if side == "high":
        point = extremes["estimated_high_f"]
        peak_time = extremes["estimated_high_time"]
        series = HIGH_SERIES
    else:
        point = extremes["estimated_low_f"]
        peak_time = extremes["estimated_low_time"]
        series = LOW_SERIES

    hours_ahead = max((peak_time - now).total_seconds() / 3600, 0.25)
    est = estimate_temp(station_id, hours_ahead=hours_ahead)
    lo, hi = est["estimated_range_f"]
    half_band = (hi - lo) / 2

    coverage = _empirical_band_coverage(station_id)
    sigma = _calibrated_sigma(half_band, coverage)

    market = get_market_for_date(series, now.date())
    if market is None:
        return None

    best = None
    for b in market["brackets"]:
        if b["last_price"] is None:
            continue
        model_p = _bucket_probability(point, sigma, b["floor_strike"], b["cap_strike"])
        edge = model_p - b["last_price"]
        if best is None or edge > best["edge"]:
            best = {"bracket": b, "model_p": model_p, "market_p": b["last_price"], "edge": edge}

    if best is not None:
        _log_edge_evaluation({
            "date": date_str,
            "side": side,
            "lead_time_hint": lead_time_hint,
            "evaluated_at": now.isoformat(),
            "hours_ahead": round(hours_ahead, 2),
            "bucket_label": best["bracket"]["label"],
            "model_implied_probability": round(best["model_p"], 4),
            "market_implied_probability": round(best["market_p"], 4),
            "edge": round(best["edge"], 4),
            "min_edge_threshold": MIN_EDGE,
            "bet_placed": best["edge"] >= MIN_EDGE,
        })

    if best is None or best["edge"] < MIN_EDGE:
        return None  # no meaningful edge today - don't force a bet

    bet = {
        "trigger_type": "edge",
        "lead_time_hint": lead_time_hint,
        "simulated_bucket_chosen": best["bracket"]["label"],
        "simulated_bucket_floor": best["bracket"]["floor_strike"],
        "simulated_bucket_cap": best["bracket"]["cap_strike"],
        "simulated_entry_price": best["market_p"],
        "simulated_stake": STAKE,
        "model_implied_probability": round(best["model_p"], 4),
        "market_implied_probability": round(best["market_p"], 4),
        "edge_at_entry": round(best["edge"], 4),
        "estimated_range_low_f": lo,
        "estimated_range_high_f": hi,
        "hours_ahead_at_entry": round(hours_ahead, 2),
        "placed_at": now.isoformat(),
        # Full bucket list at entry, so resolve_paper_trade can report
        # which bucket the actual temp landed in even if it wasn't the
        # one chosen - a wrong bet is still informative.
        "all_buckets": [
            {"label": b["label"], "floor_strike": b["floor_strike"], "cap_strike": b["cap_strike"]}
            for b in market["brackets"]
        ],
    }
    side_pending[lead_time_hint] = bet
    day_pending[side] = side_pending
    pending[date_str] = day_pending
    _save_pending(pending)
    return bet


def place_unconditional_low_bet(station_id=STATION):
    """
    The low market's second, non-edge-gated strategy (see module
    docstring): once per day, unconditionally bets $STAKE on whichever
    Kalshi bracket the model's own point estimate (estimated_low_f)
    falls into at call time - regardless of whether that bracket's
    price disagrees with the model at all. No MIN_EDGE check here; that
    is the entire point of this strategy.

    Stored under pending[date_str]["low_unconditional"] - a sibling key
    to pending[date_str]["low"], never nested inside it, so it can
    never collide with the edge-gated bets' own lead_time_hint keys
    there.

    Returns the bet dict, or None if already placed today, today's low
    market can't be found, or the specific bracket the point estimate
    falls into has no live price to simulate an entry against (same
    "can't bet what isn't priced" rule place_paper_trade already
    follows - unconditional means "no edge required," not "bet blind
    against an unknown price").
    """
    extremes = estimate_daily_extremes(station_id)
    now = extremes["as_of"]
    date_str = now.date().isoformat()

    pending = _load_pending()
    day_pending = pending.get(date_str, {})
    if "low_unconditional" in day_pending:
        return None

    point = extremes["estimated_low_f"]
    peak_time = extremes["estimated_low_time"]

    hours_ahead = max((peak_time - now).total_seconds() / 3600, 0.25)
    est = estimate_temp(station_id, hours_ahead=hours_ahead)
    lo, hi = est["estimated_range_f"]
    half_band = (hi - lo) / 2

    coverage = _empirical_band_coverage(station_id)
    sigma = _calibrated_sigma(half_band, coverage)

    market = get_market_for_date(LOW_SERIES, now.date())
    if market is None:
        return None

    chosen = next((b for b in market["brackets"] if bracket_contains(b, point)), None)
    if chosen is None or chosen["last_price"] is None:
        return None  # nothing to simulate an entry price against

    model_p = _bucket_probability(point, sigma, chosen["floor_strike"], chosen["cap_strike"])
    market_p = chosen["last_price"]

    bet = {
        "trigger_type": "unconditional",
        "lead_time_hint": "1hr",
        "simulated_bucket_chosen": chosen["label"],
        "simulated_bucket_floor": chosen["floor_strike"],
        "simulated_bucket_cap": chosen["cap_strike"],
        "simulated_entry_price": market_p,
        "simulated_stake": STAKE,
        "model_implied_probability": round(model_p, 4),
        "market_implied_probability": round(market_p, 4),
        "edge_at_entry": round(model_p - market_p, 4),
        "estimated_range_low_f": lo,
        "estimated_range_high_f": hi,
        "hours_ahead_at_entry": round(hours_ahead, 2),
        "placed_at": now.isoformat(),
        "all_buckets": [
            {"label": b["label"], "floor_strike": b["floor_strike"], "cap_strike": b["cap_strike"]}
            for b in market["brackets"]
        ],
    }
    day_pending["low_unconditional"] = bet
    pending[date_str] = day_pending
    _save_pending(pending)
    return bet


def _resolve_bet(bet, actual_temp):
    """Shared per-bet resolution math - which bracket actual_temp
    landed in, whether that matches the bet's chosen bracket, and the
    resulting simulated payout - used identically by resolve_paper_
    trade (both lead times) and resolve_unconditional_low_bet, since
    the payout arithmetic doesn't care which strategy chose the
    bracket, only what was bet and what happened."""
    outcome_bucket = None
    for b in bet.get("all_buckets", []):
        if bracket_contains(b, actual_temp):
            outcome_bucket = b["label"]
            break

    hit = outcome_bucket == bet["simulated_bucket_chosen"]
    stake = bet["simulated_stake"]
    price = bet["simulated_entry_price"]
    payout = stake * (1 - price) / price if hit and price > 0 else (0.0 if price <= 0 else -stake)

    r = {k: v for k, v in bet.items() if k != "all_buckets"}
    r["outcome_bucket"] = outcome_bucket
    r["hit"] = hit
    r["simulated_payout"] = round(payout, 4)
    range_low = bet.get("estimated_range_low_f")
    range_high = bet.get("estimated_range_high_f")
    r["within_uncertainty_band"] = (
        range_low <= actual_temp <= range_high if range_low is not None and range_high is not None else None
    )
    return r


def resolve_paper_trade(date_str, side, actual_temp):
    """
    Called from daily_performance.py's finalize_day() once the actual
    peak is known. Resolves BOTH lead times independently - never
    averaged/combined, since keeping them separate is the entire point of
    running the comparison (see module docstring). Returns
    {lead_time_hint: resolved_bet_or_None for each of LEAD_TIME_HINTS} -
    always both keys, so the caller doesn't need to guess which lead
    times might be missing.
    """
    pending = _load_pending()
    day_bets = pending.get(date_str, {}).get(side, {})

    resolved = {}
    for lead_time_hint in LEAD_TIME_HINTS:
        bet = day_bets.get(lead_time_hint)
        resolved[lead_time_hint] = _resolve_bet(bet, actual_temp) if bet is not None else None

    return resolved


def resolve_unconditional_low_bet(date_str, actual_temp):
    """
    Resolves the unconditional low bet (see place_unconditional_low_bet)
    the same way resolve_paper_trade resolves everything else - against
    whatever actual_temp the caller hands in (daily_performance.py's
    CLI-preferred value, stream as fallback - see its module docstring).
    Pure, like resolve_paper_trade, so daily_performance.reconcile_
    stream_fallback_actuals can re-call this with a corrected
    actual_temp exactly the same way it already does for the edge-gated
    bets - no special-casing needed there. Returns the resolved bet
    dict, or None if no unconditional bet was placed that day.
    """
    pending = _load_pending()
    bet = pending.get(date_str, {}).get("low_unconditional")
    return _resolve_bet(bet, actual_temp) if bet is not None else None


def _load_2hr_schedule():
    if not os.path.exists(SCHEDULE_2HR_PATH):
        return {}
    with open(SCHEDULE_2HR_PATH) as f:
        return json.load(f)


def _save_2hr_schedule(state):
    with open(SCHEDULE_2HR_PATH, "w") as f:
        json.dump(state, f, indent=2)


def get_or_lock_2hr_targets(station_id=STATION):
    """
    Locks a fire time of predicted_peak_time - LEAD_2HR_HOURS, once per
    side per date, so the "2hr" leg fires at a fixed point rather than
    chasing the model's peak-time estimate as it drifts later in the day -
    same lock-once principle as peak_alerts.py's own targets, just for a
    plain instant rather than a confidence-gated window: this is a
    data-gathering experiment, not a user-facing alert, so there's no need
    for that same gating.

    Each side is filed under the calendar date its OWN prediction targets
    (predicted_peak_time.date()), not the date lock-in happens to run on -
    same "low" flips to forecasting tonight's dawn handling as
    peak_alerts.get_or_lock_daily_targets; filing it under the lock date
    instead would double-lock it from the next day's own morning.

    Returns {"newly_locked": [(date_str, side), ...], "state": {...}} -
    newly_locked entries are what the caller needs to schedule a one-shot
    fire for (see module docstring); "state" holds only the touched
    dates' records.
    """
    extremes = estimate_daily_extremes(station_id)
    now = extremes["as_of"]

    state = _load_2hr_schedule()
    newly_locked = []
    touched_dates = set()

    for side, time_key in [("high", "estimated_high_time"), ("low", "estimated_low_time")]:
        predicted_peak_time = extremes[time_key]
        target_date_str = predicted_peak_time.date().isoformat()
        touched_dates.add(target_date_str)
        day_state = state.setdefault(target_date_str, {})
        if side in day_state:
            continue

        target_time = predicted_peak_time - timedelta(hours=LEAD_2HR_HOURS)
        day_state[side] = {
            "predicted_peak_time_at_lock": predicted_peak_time.isoformat(),
            "target_time": target_time.isoformat(),
            "locked_at": now.isoformat(),
            "skipped_missed_window": target_time <= now,
        }
        newly_locked.append((target_date_str, side))

    _save_2hr_schedule(state)
    return {
        "newly_locked": newly_locked,
        "state": {d: state[d] for d in sorted(touched_dates)},
    }


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "place":
        side = sys.argv[2] if len(sys.argv) > 2 else "high"
        lead_time_hint = sys.argv[3] if len(sys.argv) > 3 else "1hr"
        bet = place_paper_trade(STATION, side, lead_time_hint)
        print(json.dumps(bet, indent=2) if bet else json.dumps({"placed": False}))
    elif cmd == "place-unconditional-low":
        bet = place_unconditional_low_bet(STATION)
        print(json.dumps(bet, indent=2) if bet else json.dumps({"placed": False}))
    elif cmd == "lock-2hr":
        result = get_or_lock_2hr_targets(STATION)
        print(json.dumps(result, indent=2))
        for date_str, side in result["newly_locked"]:
            side_state = result["state"][date_str][side]
            if not side_state["skipped_missed_window"]:
                print(f"PAPER_TRADE_2HR_SCHEDULE_NEEDED side={side} date={date_str} target_time={side_state['target_time']}")
    elif cmd == "status":
        print(json.dumps(_load_pending(), indent=2))
    elif cmd == "edge-stats":
        print(json.dumps(edge_log_stats(), indent=2))
    else:
        print(f"Unknown command: {cmd}. Use place <high|low> [1hr|2hr] | place-unconditional-low | lock-2hr | status | edge-stats.")
