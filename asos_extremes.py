"""
The station's own un-quantised daily extremes, from the METAR remarks.

Why this exists: the 5-minute observation feed reports whole degrees
Celsius, so the coldest reading it can show for a morning whose true low
was 14.4C is "14" - which converts to 57.20F and looks like a precise
answer to hundredths. It isn't. It is 1.8F-granular, and it is what made
2026-07-29's low ambiguous across three Kalshi brackets when the actual
answer was never in doubt.

ASOS already publishes the un-quantised number. Every six hours the METAR
remarks carry the max and min computed from the 1-minute data:

    KSEA 291153Z ... RMK AO2 T01560106 10161 20144 56004
                                       ^^^^^ ^^^^^
                                       6h max 6h min
                                       16.1C  14.4C

20144 = group 2 (minimum), sign 0 (positive), 144 = 14.4C = 57.92F, which
settles to 58 - not the 57 the 5-minute feed implies, and exactly where
the market was priced. The tenths are real here: they come off the same
1-minute record NWS uses to write the CLI report, so this is the closest
thing to the settlement value available before CLI publishes.

Group meanings (FMH-1, and the 24h group at local midnight):

    1sTTT      6-hour maximum, s=1 means negative
    2sTTT      6-hour minimum
    4sTTTsTTT  24-hour max then min, issued in the 00 LST report

The 6-hour groups are issued at the synoptic hours 00/06/12/18Z, each
covering the SIX HOURS ENDING at issuance. At KSEA that puts the 12Z
report over 23:00-05:00 local - which brackets a typical dawn minimum -
and the 00Z report over 11:00-17:00 local, which brackets a typical
afternoon maximum. So both of a day's extremes are usually covered, but
neither window lines up with the local calendar day, and the 12Z window
in particular straddles midnight.

That misalignment is why this module never simply believes a group. It
uses the group to REFINE the stream's own extreme: the stream says when
the extreme happened and roughly what it was, the group says precisely
what, and the two are only combined when they agree to within the
stream reading's own quantisation slack. If they disagree by more than
that, the extreme probably fell outside the group's window (or on the
other side of midnight) and the stream value stands, unrefined.
"""

import re
from datetime import datetime, time, timedelta

# 6-hour groups cover the six hours ending at issuance.
SIX_HOUR_WINDOW = timedelta(hours=6)
# How far the group may sit from the stream's own extreme and still be
# believed as the same event. The stream value is a whole-C reading, so
# the truth is within +/-0.5C = +/-0.9F of it; anything further apart is a
# different extreme, not a more precise view of this one.
AGREEMENT_TOLERANCE_F = 0.95

_GROUP_6H = re.compile(r'(?<!\S)([12])([01])(\d{3})(?!\S)')
_GROUP_24H = re.compile(r'(?<!\S)4([01])(\d{3})([01])(\d{3})(?!\S)')


def _celsius(sign, digits):
    value = int(digits) / 10.0
    return -value if sign == "1" else value


def _to_f(c):
    return c * 9.0 / 5.0 + 32.0


def parse_remark_extremes(raw_metar, observed_at):
    """
    Every max/min group in one METAR's remarks.

    Returns a list of {"side", "temp_f", "window_start", "window_end"}.
    observed_at is the report's own timestamp, tz-aware, and anchors the
    windows - the groups carry no times of their own.
    """
    if not raw_metar or "RMK" not in raw_metar:
        return []
    rmk = raw_metar.split("RMK", 1)[1]
    out = []

    for m in _GROUP_24H.finditer(rmk):
        # Issued in the midnight-LST report and covering the calendar day
        # that just ended.
        end = observed_at
        start = end - timedelta(hours=24)
        out.append({"side": "high", "temp_f": _to_f(_celsius(m.group(1), m.group(2))),
                    "window_start": start, "window_end": end, "span_hours": 24})
        out.append({"side": "low", "temp_f": _to_f(_celsius(m.group(3), m.group(4))),
                    "window_start": start, "window_end": end, "span_hours": 24})

    consumed = {m.span() for m in _GROUP_24H.finditer(rmk)}
    for m in _GROUP_6H.finditer(rmk):
        # A 4-group contains digit runs that also match the 6-hour pattern;
        # skip anything sitting inside one already parsed above.
        if any(s <= m.start() and m.end() <= e for s, e in consumed):
            continue
        side = "high" if m.group(1) == "1" else "low"
        out.append({"side": side, "temp_f": _to_f(_celsius(m.group(2), m.group(3))),
                    "window_start": observed_at - SIX_HOUR_WINDOW,
                    "window_end": observed_at, "span_hours": 6})
    return out


def collect_remark_extremes(times, raw_messages):
    """All groups across a series of observations, newest last."""
    found = []
    for t, raw in zip(times, raw_messages):
        if raw:
            found.extend(parse_remark_extremes(raw, t))
    return found


def refine_extreme(side, stream_temp_f, stream_time, candidates):
    """
    The precise value of an extreme the stream only saw quantised.

    Picks the candidate group whose window contains the stream's extreme
    and whose value agrees with it to within AGREEMENT_TOLERANCE_F,
    preferring the narrowest window when several qualify (a 6-hour group
    localises the event better than a 24-hour one).

    Returns (temp_f, source) where source is "asos_remark_1min" when a
    group was used and "observation_stream" when none applied - callers
    should treat the two differently, because only the first is precise
    enough to settle a bracket on its own.
    """
    if stream_temp_f is None or stream_time is None:
        return stream_temp_f, "observation_stream"

    usable = [
        c for c in candidates
        if c["side"] == side
        and c["window_start"] <= stream_time <= c["window_end"]
        and abs(c["temp_f"] - stream_temp_f) <= AGREEMENT_TOLERANCE_F
    ]
    if not usable:
        return stream_temp_f, "observation_stream"

    # Narrowest window first; among equals, the one closest to the stream
    # value - both are the same physical extreme, so this just avoids
    # depending on report order.
    best = min(usable, key=lambda c: (c["span_hours"], abs(c["temp_f"] - stream_temp_f)))
    return round(best["temp_f"], 2), "asos_remark_1min"


if __name__ == "__main__":
    import requests
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("America/Los_Angeles")
    r = requests.get(
        "https://api.weather.gov/stations/KSEA/observations",
        params={"limit": 300},
        headers={"User-Agent": "(ksea-dashboard, meganfinn.rigney@gmail.com)"},
        timeout=60,
    )
    feats = sorted((f["properties"] for f in r.json()["features"]),
                   key=lambda p: p["timestamp"])
    times = [datetime.fromisoformat(p["timestamp"]).astimezone(tz) for p in feats]
    raws = [p.get("rawMessage") for p in feats]
    for c in collect_remark_extremes(times, raws):
        print(f"{c['window_start']:%m-%d %H:%M} .. {c['window_end']:%m-%d %H:%M}  "
              f"{c['side']:4} {c['temp_f']:6.2f}F -> {round(c['temp_f'])}")
