"""
Fetches and parses NWS's official daily Climatological Report (CLI
product, issued by WFO Seattle = SEW) - the actual settlement source
Kalshi's KXHIGHTSEA/KXLOWTSEA markets use (confirmed via a live
market's rules_secondary field - see kalshi.py's own docstring). Used
as the primary source of "actual" high/low throughout
daily_performance.py and paper_trading.py, in place of the raw ASOS
observation stream those previously relied on exclusively - spot
checks found the stream running ~2-3F warm on the high side against
CLI's own reported values on the days compared.

Two products get issued per calendar day: a PRELIMINARY report around
5pm ("VALID TODAY AS OF 0500 PM LOCAL TIME") covering only part of the
day - its MAXIMUM can still be beaten later that evening - and a FINAL
report in the small hours of the next day covering the complete
previous day. Only the FINAL report is treated as authoritative here;
the preliminary is deliberately skipped even though it's usually
available sooner, since a partial-day summary isn't what Kalshi
actually settles against.

Retention: api.weather.gov's /products/types/CLI/locations/{loc}
endpoint returned roughly the last 5 calendar days of reports (10
entries, 2/day) as observed while building this - not a documented
guarantee, so callers must treat "not found" as "not available (either
too old, or not published yet)" and fall back to another actual-value
source rather than assuming a fixed retention window.

The report never gives a time-of-day for the max/min (always "MM" -
missing - in the OBSERVED TIME column on every report checked, for
this WFO); callers needing a peak TIME still have to fall back to the
observation stream for that. This module only supplies the temperature
VALUE.
"""

import re
from datetime import datetime

import requests

NWS_API_BASE = "https://api.weather.gov"
CLIMATE_PRODUCT_CODE = "CLI"
CLIMATE_LOCATION_ID = "SEW"  # WFO Seattle - distinct from the ASOS station id (KSEA) used elsewhere
USER_AGENT = "(ksea-weather-dashboard, github.com/meganfinnrigney-eng/weather)"

_HEADERS = {"User-Agent": USER_AGENT}

_SUMMARY_DATE_RE = re.compile(r"CLIMATE SUMMARY FOR ([A-Z]+ \d{1,2} \d{4})")
_PRELIMINARY_RE = re.compile(r"VALID (?:TODAY|YESTERDAY) AS OF")
_MAX_RE = re.compile(r"^\s*MAXIMUM\s+(-?\d+|MM)", re.MULTILINE)
_MIN_RE = re.compile(r"^\s*MINIMUM\s+(-?\d+|MM)", re.MULTILINE)


def _fetch_recent_product_ids():
    r = requests.get(
        f"{NWS_API_BASE}/products/types/{CLIMATE_PRODUCT_CODE}/locations/{CLIMATE_LOCATION_ID}",
        headers=_HEADERS,
    )
    r.raise_for_status()
    return [item["id"] for item in r.json().get("@graph", [])]


def _fetch_product_text(product_id):
    r = requests.get(f"{NWS_API_BASE}/products/{product_id}", headers=_HEADERS)
    r.raise_for_status()
    return r.json()["productText"]


def _parse_cli_text(text):
    """None if this doesn't look like a CLI report at all (unexpected
    format - fetch failure text, a schema change, etc.); otherwise a
    dict with covers_date/is_final/high_f/low_f. Either temp field can
    be None on a real match if NWS itself reported that side "MM" -
    that's a genuine "missing" from NWS, not a parse failure."""
    date_match = _SUMMARY_DATE_RE.search(text)
    if not date_match:
        return None
    covers_date = datetime.strptime(date_match.group(1), "%B %d %Y").date()
    is_final = not _PRELIMINARY_RE.search(text)

    max_match = _MAX_RE.search(text)
    min_match = _MIN_RE.search(text)
    high_f = float(max_match.group(1)) if max_match and max_match.group(1) != "MM" else None
    low_f = float(min_match.group(1)) if min_match and min_match.group(1) != "MM" else None

    return {"covers_date": covers_date, "is_final": is_final, "high_f": high_f, "low_f": low_f}


def fetch_recent_cli_finals():
    """
    Fetches the whole currently-retained window of CLI reports in one
    pass, parses each, and returns {date: {"high_f":..., "low_f":...,
    "product_id": str}} for every FINAL report found - preliminary
    reports are parsed too (to correctly detect is_final) but never
    included in the returned mapping.

    Callers checking several dates in one pass (daily_performance.py's
    reconciliation loop) should call this ONCE and look up by date from
    the result, rather than each date re-fetching and re-parsing the
    same handful of reports. Returns {} (not raising) on any fetch
    failure - CLI is a fallback-augmented source everywhere it's used,
    never a hard dependency.
    """
    try:
        product_ids = _fetch_recent_product_ids()
    except Exception:
        return {}

    by_date = {}
    for product_id in product_ids:
        try:
            text = _fetch_product_text(product_id)
        except Exception:
            continue
        parsed = _parse_cli_text(text)
        if parsed is None or not parsed["is_final"]:
            continue
        by_date[parsed["covers_date"]] = {
            "high_f": parsed["high_f"],
            "low_f": parsed["low_f"],
            "product_id": product_id,
        }
    return by_date


def get_cli_final_actuals_for_date(day):
    """Single-date convenience wrapper around fetch_recent_cli_finals -
    prefer the bulk function directly when checking multiple dates."""
    return fetch_recent_cli_finals().get(day)


if __name__ == "__main__":
    import sys
    from datetime import date

    if len(sys.argv) > 1:
        target = date.fromisoformat(sys.argv[1])
        print(get_cli_final_actuals_for_date(target))
    else:
        by_date = fetch_recent_cli_finals()
        for d in sorted(by_date):
            print(d, by_date[d])
