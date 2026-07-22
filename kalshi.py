"""
Pulls live public market data from Kalshi for Seattle daily high/low
temperature markets, for comparison against weather_estimator's own
predictions.

Read-only: this only hits Kalshi's public market-data endpoints (prices,
brackets, resolution rules), which don't require authentication. No order
placement here - trades stay manual until the model's calibration has a
track record.

Settlement source confirmed directly from a live market's rules_secondary
field: the NWS Climatological Report (Daily) for Seattle-Tacoma (product
CLI, issued by site SEA under WFO SEW) - not the raw ASOS observation feed
weather_estimator.py itself uses, and not Weather Underground. Spot-checked
against 7 days of the actual CLI reports: highs matched within about 1F with
no consistent bias; lows matched too, except one day with a multi-hour
morning gap in our own observation data (see weather_estimator's known
data-completeness caveat - not fixed as of this module's writing).

pip install requests
"""

import time
from datetime import datetime, timezone

import requests

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

HIGH_SERIES = "KXHIGHTSEA"
LOW_SERIES = "KXLOWTSEA"


def get_event_ticker_for_date(series_ticker, for_date):
    """Find the open event ticker for a specific date in a Kalshi series,
    e.g. KXHIGHTSEA-26JUL21. Returns None if that date has no open event
    (already settled, or too far in the future to be listed yet)."""
    r = requests.get(f"{KALSHI_BASE}/events", params={"series_ticker": series_ticker, "status": "open"})
    r.raise_for_status()
    events = r.json()["events"]

    suffix = for_date.strftime("%y%b%d").upper()  # e.g. "26JUL21"
    for e in events:
        if e["event_ticker"].endswith(suffix):
            return e["event_ticker"]
    return None


def bracket_contains(bracket, value):
    """Whether a temperature value falls inside a bracket's strike range.
    Kalshi settles on the officially reported whole-degree temperature,
    while a model estimate is usually a continuous decimal (e.g. 91.40) -
    rounding first avoids it falling in the crack between adjacent integer
    brackets like "90 to 91" and "92 to 93", where neither would match."""
    value = round(value)
    floor = bracket["floor_strike"]
    cap = bracket["cap_strike"]
    if floor is not None and cap is not None:
        return floor <= value <= cap
    if floor is not None:
        # "X or above" tail bracket - its floor_strike reuses the same
        # number as the adjacent ranged bracket's cap_strike (e.g. "64 or
        # above" has floor=63, same as "62 to 63"'s cap=63), so it has to
        # be strictly greater than or both brackets would match on 63.
        return value > floor
    if cap is not None:
        # Same idea in reverse for "X or below" (e.g. "87 or below" has
        # cap=88, same as "88 to 89"'s floor=88).
        return value < cap
    return False


def get_event_ticker_for_any_date(series_ticker, for_date):
    """Same as get_event_ticker_for_date, but finds the event regardless of
    whether it's still open or has already settled - needed for looking up
    a past date's market after the fact (e.g. daily_performance.py, which
    only finalizes a day once it's over, by which point that day's event
    has typically already settled)."""
    r = requests.get(f"{KALSHI_BASE}/events", params={"series_ticker": series_ticker, "limit": 200})
    r.raise_for_status()
    events = r.json()["events"]

    suffix = for_date.strftime("%y%b%d").upper()
    for e in events:
        if e["event_ticker"].endswith(suffix):
            return e["event_ticker"]
    return None


def get_market_brackets(event_ticker):
    """
    Return every bracket market in an event, sorted low to high, with its
    current price (yes_bid/yes_ask/last_price as 0-1 probabilities - Kalshi
    prices are in dollars where $1.00 = 100% probability of settling Yes).
    """
    r = requests.get(f"{KALSHI_BASE}/markets", params={"event_ticker": event_ticker})
    r.raise_for_status()
    markets = r.json()["markets"]

    def sort_key(m):
        # "T88" (87 or below) style tail markets have no floor_strike - sort
        # them before/after the ranged brackets using cap_strike instead
        if m.get("floor_strike") is not None:
            return m["floor_strike"]
        return m.get("cap_strike", 0) - 0.5

    def to_float(d):
        return float(d) if d else None

    rows = []
    for m in sorted(markets, key=sort_key):
        rows.append({
            "ticker": m["ticker"],
            "label": m.get("yes_sub_title"),
            "floor_strike": m.get("floor_strike"),
            "cap_strike": m.get("cap_strike"),
            "yes_bid": to_float(m.get("yes_bid_dollars")),
            "yes_ask": to_float(m.get("yes_ask_dollars")),
            "last_price": to_float(m.get("last_price_dollars")),
            "volume": to_float(m.get("volume_fp")),
        })
    return rows


def get_event_hourly_volume(series_ticker, brackets, hours=24):
    """
    Hourly dollar volume traded across an entire event, by summing every
    bracket's own hourly candlesticks (Kalshi already tracks this exactly
    per market - no need to snapshot it ourselves over time). Each
    candlestick gives contracts traded that hour and the mean trade price;
    contracts * mean_price estimates the actual dollars exchanged (Kalshi's
    own site figure, like a "$100k+ Vol" total, is this same idea summed
    over an event's full lifetime rather than just one hour).

    An hour with zero trades has no "price" data at all (checked directly
    against the API), so those contribute $0/0 contracts rather than being
    read as a real trade at a real price. They're still included as an
    hour with zero volume, though - every hour in the window gets a slot,
    even a quiet one, so the chart's time axis stays evenly spaced instead
    of silently skipping gaps.
    """
    end_ts = int(time.time())
    start_ts = end_ts - hours * 3600
    buckets = {}  # end_period_ts -> {"contracts": float, "dollars": float}

    for b in brackets:
        r = requests.get(
            f"{KALSHI_BASE}/series/{series_ticker}/markets/{b['ticker']}/candlesticks",
            params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": 60},
        )
        r.raise_for_status()
        for c in r.json()["candlesticks"]:
            bucket = buckets.setdefault(c["end_period_ts"], {"contracts": 0.0, "dollars": 0.0})
            contracts = float(c["volume_fp"])
            if contracts <= 0:
                continue
            mean_price = c["price"].get("mean_dollars")
            if mean_price is None:
                continue
            bucket["contracts"] += contracts
            bucket["dollars"] += contracts * float(mean_price)

    # Fill in any hour boundary in the window with no candlestick at all
    # (e.g. a bracket with no trading history yet) so every hour still
    # gets a zero-volume slot instead of vanishing from the timeline.
    first_hour = (start_ts // 3600 + 1) * 3600
    last_hour = (end_ts // 3600) * 3600
    for ts in range(first_hour, last_hour + 1, 3600):
        buckets.setdefault(ts, {"contracts": 0.0, "dollars": 0.0})

    hourly = [
        {
            "hour_end": datetime.fromtimestamp(ts, tz=timezone.utc),
            "contracts": round(v["contracts"], 2),
            "dollars": round(v["dollars"], 2),
        }
        for ts, v in sorted(buckets.items())
    ]
    return {
        "hourly": hourly,
        "total_dollars": round(sum(h["dollars"] for h in hourly), 2),
        "total_contracts": round(sum(h["contracts"] for h in hourly), 2),
    }


def get_market_for_date(series_ticker, for_date):
    """Event ticker + bracket list for a given date, or None if that date
    isn't an open event on this series."""
    event_ticker = get_event_ticker_for_date(series_ticker, for_date)
    if event_ticker is None:
        return None
    return {
        "event_ticker": event_ticker,
        "brackets": get_market_brackets(event_ticker),
    }


if __name__ == "__main__":
    from datetime import date
    today = date.today()

    for series, label in [(HIGH_SERIES, "HIGH"), (LOW_SERIES, "LOW")]:
        market = get_market_for_date(series, today)
        print(f"--- {label} ({series}) for {today} ---")
        if market is None:
            print("no open event for today")
            continue
        print(market["event_ticker"])
        for b in market["brackets"]:
            print(f"  {b['label']:>15}  last={b['last_price']}  bid={b['yes_bid']}  ask={b['yes_ask']}")
