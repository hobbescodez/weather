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
