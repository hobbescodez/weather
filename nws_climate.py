"""
Fetches and parses NWS's official daily Climatological Report - the
CLISEA product for Seattle-Tacoma International Airport, the actual
settlement source
Kalshi's KXHIGHTSEA/KXLOWTSEA markets use (confirmed via a live
market's rules_secondary field - see kalshi.py's own docstring). Used
as the primary source of "actual" high/low throughout
daily_performance.py and paper_trading.py, alongside the raw ASOS
observation stream those previously relied on exclusively. The two
agree closely once the right product is being read - CLISEA and the
KSEA stream matched to within ~0.5F on every day checked. An earlier
version of this module read CLISEW instead (the Seattle WFO office at
Sand Point, not the airport) and the resulting 2-4F disagreements were
initially mistaken for the observation stream being biased; see
CLIMATE_LOCATION_ID.

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
from datetime import datetime, time, timedelta

import requests

NWS_API_BASE = "https://api.weather.gov"
CLIMATE_PRODUCT_CODE = "CLI"

# MUST be SEA (product CLISEA, "THE SEATTLE-TACOMA WA AIRPORT CLIMATE
# SUMMARY") - the airport, which is the station KSEA reports from and the
# one Kalshi settles against.
#
# This was originally SEW, which is wrong and cost a week of bad
# "actual" values. The trap: /products/types/CLI/locations lists SEW
# with the human label "Seattle/Tacoma, WA" while listing SEA with no
# label at all, so SEW looks like the obvious choice. But SEW is the
# Seattle WFO's own office site (Sand Point, on Lake Washington,
# ~10 miles from the airport and a different microclimate) and its
# product says "THE SEATTLE WA WFO CLIMATE SUMMARY". The two disagree by
# 2-4F in BOTH directions depending on the airmass - SEW ran 2-3F cooler
# than the airport on hot offshore days and 2-4F warmer on marine days -
# which is exactly the sort of error that looks like model bias rather
# than a data-source bug.
#
# _EXPECTED_SITE_RE below hard-fails any report that isn't the airport,
# so this can't silently drift again.
CLIMATE_LOCATION_ID = "SEA"
USER_AGENT = "(ksea-weather-dashboard, github.com/hobbescodez/weather)"

_HEADERS = {"User-Agent": USER_AGENT}

_EXPECTED_SITE_RE = re.compile(r"SEATTLE-TACOMA\s+WA\s+AIRPORT", re.IGNORECASE)
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
    # Station identity is checked before anything else is trusted. Every
    # CLI product parses identically, so without this a report for a
    # different site reads as perfectly valid data for the wrong place -
    # which is precisely what happened with SEW (see CLIMATE_LOCATION_ID).
    if not _EXPECTED_SITE_RE.search(text):
        return None

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


_FINALS_CACHE = None

# When the FINAL report for a finished day actually shows up, measured over
# the API's whole retention window (every product, parsed, lag from the local
# midnight that ended the covered day):
#
#   07-22 +1.5h   07-23 +1.5h   07-24 +1.5h
#   07-25 +1.5h   07-26 +1.4h   07-27 +1.4h
#   07-21 +9.4h  <- the one genuinely late first issue
#
# So the schedule is tight: normally ~01:25 local, six of the last seven
# within six minutes of each other. That tightness is what makes "hasn't
# published yet" a misleading thing to say at 9am - by then it is not the
# normal wait, it is an outlier, and the two cases should not read the same.
# GRACE is set past the observed spread but well short of the +9.4h outlier.
CLI_FINAL_TYPICAL_LAG_HOURS = 1.5
CLI_FINAL_GRACE_HOURS = 4.0


def cli_final_lag_hours(day, now):
    """Hours since the local midnight that ended `day` - i.e. how long the
    final report for that day has been possible to publish."""
    end_of_day = datetime.combine(day, time(0, 0), tzinfo=now.tzinfo) + timedelta(days=1)
    return (now - end_of_day).total_seconds() / 3600.0


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

    Memoised for the life of the process. A single build_dashboard run now
    asks for this from three places (daily_performance's reconcilers,
    calibration_health, and weather_estimator's "yesterday"), each of which
    would otherwise re-fetch and re-parse the same dozen-odd products over
    HTTP. Published CLI reports don't change within one run, so there is
    nothing to gain by re-reading them. Only successful non-empty results
    are cached: an empty result can mean a transient fetch failure, and
    pinning that for the rest of the run would turn one bad request into a
    whole refresh with no settlement data.
    """
    global _FINALS_CACHE
    if _FINALS_CACHE is not None:
        return _FINALS_CACHE

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
    if by_date:
        _FINALS_CACHE = by_date
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
