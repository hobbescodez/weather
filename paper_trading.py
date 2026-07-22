"""
Paper-trades the model's Kalshi picks - simulated bets only. Tests
whether the model's own probability estimate would have beaten Kalshi's
market price, before ever considering a real order. No order-placement
API call is made anywhere in this module.

Timing: piggybacks on peak_alerts.py's existing "1hr before predicted
peak" scheduling (same one-shot Routine fire) rather than building a
second scheduling mechanism - see build_dashboard.py's
ALERT_SCHEDULE_NEEDED integration and the hourly Routine's prompt, which
creates one one-shot trigger per side that runs both peak_alerts.py
check and this module's place command at the same moment. That's also
the "close to the actual peak" placement the feature spec asked for:
thin overnight trading (e.g. ~5am for the low) is exactly when a real
model edge is least likely to already be priced in.

Storage: a bet is placed intraday (near peak time) but can only be
resolved once the actual peak is known, which isn't until
daily_performance.py's finalize_day() runs - well after the bet exists.
paper_trades_pending.json bridges that gap (a small, transient staging
file, not a second permanent store): place_paper_trade() records the bet
there, and finalize_day() calls resolve_paper_trade() to compute the
outcome and folds the resolved fields directly into the SAME
daily_performance.jsonl record it already writes - per the spec, no
separate tracking table.

Probability model: the model's own uncertainty band is not a calibrated
confidence interval as-is - weather_estimator.backtest()'s own
pct_within_uncertainty_band shows real coverage well below what the band
nominally targets (~0.36 actual vs. a ~0.9 aim, as of this writing).
Treating the raw band as if it were already an honest interval would
make the model look more confident than its track record supports, so a
Normal distribution's width is solved to match that empirical coverage
rate instead, and bucket probabilities are priced off THAT calibrated
distribution.

Edge threshold: MIN_EDGE below is a deliberate, adjustable choice (not a
derived constant) - "don't force a bet without a real edge," per the
spec, needs some numeric cutoff, and 15 percentage points was picked as
a conservative bar given the model's still-modest calibration.

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
    python3 paper_trading.py place high
    python3 paper_trading.py place low
    python3 paper_trading.py status
"""

import json
import math
import os
from datetime import datetime

from weather_estimator import estimate_daily_extremes, estimate_temp, backtest
from kalshi import HIGH_SERIES, LOW_SERIES, get_market_for_date, bracket_contains

PENDING_PATH = os.path.join(os.path.dirname(__file__), "paper_trades_pending.json")

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


def place_paper_trade(station_id, side):
    """
    Compares the model's own calibrated probability against Kalshi's
    current price for every bucket in today's market, and - only if the
    biggest mispricing clears MIN_EDGE - records a simulated $STAKE bet
    on the most-underpriced bucket. Returns the bet dict, or None if no
    bet was placed (no edge, market unavailable, or already placed today
    for this side).
    """
    extremes = estimate_daily_extremes(station_id)
    now = extremes["as_of"]
    date_str = now.date().isoformat()

    pending = _load_pending()
    day_pending = pending.get(date_str, {})
    if side in day_pending:
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

    if best is None or best["edge"] < MIN_EDGE:
        return None  # no meaningful edge today - don't force a bet

    bet = {
        "simulated_bucket_chosen": best["bracket"]["label"],
        "simulated_bucket_floor": best["bracket"]["floor_strike"],
        "simulated_bucket_cap": best["bracket"]["cap_strike"],
        "simulated_entry_price": best["market_p"],
        "simulated_stake": STAKE,
        "model_implied_probability": round(best["model_p"], 4),
        "market_implied_probability": round(best["market_p"], 4),
        "edge_at_entry": round(best["edge"], 4),
        "placed_at": now.isoformat(),
        # Full bucket list at entry, so resolve_paper_trade can report
        # which bucket the actual temp landed in even if it wasn't the
        # one chosen - a wrong bet is still informative.
        "all_buckets": [
            {"label": b["label"], "floor_strike": b["floor_strike"], "cap_strike": b["cap_strike"]}
            for b in market["brackets"]
        ],
    }
    day_pending[side] = bet
    pending[date_str] = day_pending
    _save_pending(pending)
    return bet


def resolve_paper_trade(date_str, side, actual_temp):
    """
    Called from daily_performance.py's finalize_day() once the actual
    peak is known. Returns the bet dict augmented with outcome_bucket,
    hit, and simulated_payout - or None if no bet was placed that
    date/side.
    """
    pending = _load_pending()
    bet = pending.get(date_str, {}).get(side)
    if bet is None:
        return None

    outcome_bucket = None
    for b in bet.get("all_buckets", []):
        if bracket_contains(b, actual_temp):
            outcome_bucket = b["label"]
            break

    hit = outcome_bucket == bet["simulated_bucket_chosen"]
    stake = bet["simulated_stake"]
    price = bet["simulated_entry_price"]
    payout = stake * (1 - price) / price if hit and price > 0 else (0.0 if price <= 0 else -stake)

    resolved = {k: v for k, v in bet.items() if k != "all_buckets"}
    resolved["outcome_bucket"] = outcome_bucket
    resolved["hit"] = hit
    resolved["simulated_payout"] = round(payout, 4)
    return resolved


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "place":
        side = sys.argv[2] if len(sys.argv) > 2 else "high"
        bet = place_paper_trade("KSEA", side)
        print(json.dumps(bet, indent=2) if bet else json.dumps({"placed": False}))
    elif cmd == "status":
        print(json.dumps(_load_pending(), indent=2))
    else:
        print(f"Unknown command: {cmd}. Use place <high|low> | status.")
