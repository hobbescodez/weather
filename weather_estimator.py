"""
Weather temperature estimator.

Pulls recent observations from api.weather.gov, fits a trend, damps that
trend using actual sunrise/sunset for the station (via astral), and
estimates temperature N hours ahead. Includes a backtest harness that
checks past estimates against what the station actually recorded.

pip install requests pandas numpy astral
"""

import math
import warnings

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time
from zoneinfo import ZoneInfo
from astral import LocationInfo
from astral.sun import sun

PST = ZoneInfo("America/Los_Angeles")
UTC = ZoneInfo("UTC")


# ---------------------------------------------------------------------------
# Station metadata / location
# ---------------------------------------------------------------------------

def get_station_location(station_id):
    """Return (lat, lon, name) for a station."""
    url = f"https://api.weather.gov/stations/{station_id.upper()}"
    r = requests.get(url, headers={"Accept": "application/geo+json"})
    r.raise_for_status()
    data = r.json()
    lon, lat = data["geometry"]["coordinates"]
    name = data["properties"].get("name", station_id.upper())
    return lat, lon, name


def get_sun_times(lat, lon, for_date, tz=PST):
    """Return (sunrise_hour, sunset_hour) as decimal hours in local time for a given date."""
    loc = LocationInfo(latitude=lat, longitude=lon, timezone=str(tz))
    s = sun(loc.observer, date=for_date, tzinfo=tz)
    sunrise = s["sunrise"]
    sunset = s["sunset"]
    return (
        sunrise.hour + sunrise.minute / 60,
        sunset.hour + sunset.minute / 60,
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

CLOUD_AMOUNT_TO_FRACTION = {
    "SKC": 0.0, "CLR": 0.0,
    "FEW": 0.1875,
    "SCT": 0.4375,
    "BKN": 0.75,
    "OVC": 1.0,
    "VV": 1.0,
}


def _cloud_fraction(cloud_layers):
    if not cloud_layers:
        return None
    fractions = [CLOUD_AMOUNT_TO_FRACTION.get(l.get("amount"), None) for l in cloud_layers]
    fractions = [f for f in fractions if f is not None]
    return max(fractions) if fractions else None


def get_observation_history(station_id, limit=8, start=None, end=None):
    """
    Pull observations for a station, oldest to newest.
    If start/end (timezone-aware datetimes) are given, use those instead of `limit`
    so this can also be used to pull a historical window for backtesting.
    """
    url = f"https://api.weather.gov/stations/{station_id.upper()}/observations"

    if start is not None and end is not None:
        # The API's `end` bound is exclusive in practice - a request with
        # end=<some observation's exact timestamp> silently drops that
        # observation. Callers reasonably expect start/end to be inclusive
        # (e.g. "today's observations" using end=<the latest reading's own
        # timestamp>), so nudge it forward slightly before sending.
        end = end + timedelta(minutes=1)

        # The API returns at most 500 observations per request, newest first,
        # and exposes a `pagination.next` cursor URL for older pages. A busy
        # station (e.g. KSEA reports every few minutes) can blow past 500
        # observations well within a multi-day window, so page backwards
        # until we've covered the requested start time.
        page_url = url
        page_params = {
            "start": start.astimezone(UTC).isoformat(),
            "end": end.astimezone(UTC).isoformat(),
            "limit": 500,
        }
        features = []
        for _ in range(20):  # safety cap: 10,000 observations
            r = requests.get(page_url, headers={"Accept": "application/geo+json"}, params=page_params)
            r.raise_for_status()
            data = r.json()
            page_features = data["features"]
            features.extend(page_features)

            oldest_ts = datetime.fromisoformat(page_features[-1]["properties"]["timestamp"].replace("Z", "+00:00")) if page_features else None
            next_url = data.get("pagination", {}).get("next")
            if not next_url or oldest_ts is None or oldest_ts <= start.astimezone(UTC):
                break
            page_url = next_url
            page_params = {"limit": 500}  # cursor URL carries the rest of the query
        else:
            warnings.warn(
                f"Stopped paging observations for {station_id.upper()} after 10,000 records "
                f"without reaching {start}; the window may still be truncated."
            )
    else:
        params = {"limit": limit}
        r = requests.get(url, headers={"Accept": "application/geo+json"}, params=params)
        r.raise_for_status()
        features = r.json()["features"]

    rows = []
    for f in features:
        props = f["properties"]
        temp_c = props["temperature"]["value"]
        dewpoint_c = props["dewpoint"]["value"]
        if temp_c is None:
            continue

        wind_kmh = props.get("windSpeed", {}).get("value")
        pressure_pa = props.get("barometricPressure", {}).get("value")
        rh = props.get("relativeHumidity", {}).get("value")
        cloud_layers = props.get("cloudLayers", [])

        rows.append({
            "time": datetime.fromisoformat(props["timestamp"].replace("Z", "+00:00")).astimezone(PST),
            "temp_f": temp_c * 9 / 5 + 32,
            "dewpoint_f": dewpoint_c * 9 / 5 + 32 if dewpoint_c is not None else None,
            "wind_mph": wind_kmh * 0.621371 if wind_kmh is not None else None,
            "pressure_inhg": pressure_pa / 3386.39 if pressure_pa is not None else None,
            "relative_humidity": rh,
            "cloud_fraction": _cloud_fraction(cloud_layers),
        })

    if start is not None and end is not None:
        # `start` above only controls when to STOP paginating (older pages
        # keep getting fetched until one crosses it) - it never actually
        # trimmed the collected rows to the requested window. A single
        # page already covers 500 observations (~1.7 days at KSEA's
        # reporting cadence), so any query for a window shorter than that
        # - e.g. "today so far" checked a few hours after midnight -
        # silently kept observations from days earlier in the same
        # DataFrame, which then quietly won the min()/max() if they
        # happened to be more extreme. Confirmed directly: a request for
        # today's midnight-to-now window came back containing readings
        # from two days prior.
        rows = [r for r in rows if start <= r["time"] <= end]

    if not rows:
        raise ValueError("No valid temperature readings for the requested window.")
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Forward-looking signals
#
# Everything above only ever looks backward: a trend fit on the station's own
# last several observations. That's blind to anything that hasn't reached the
# station yet - an approaching front, a marine push - which is exactly what
# went wrong during the Jul 20 event (predictions off by -5 to -9.6F because
# the model had no way to see the regime change coming). These two functions
# pull in signals that *can* see it coming: a real forecast model, and an
# upwind pressure gradient that tends to lead marine intrusions by 1-3 hours.
# ---------------------------------------------------------------------------

def get_hourly_forecast(lat, lon, hours=12):
    """
    Fetch the NWS gridpoint hourly forecast (HRRR-model-based) for this
    location, for roughly the next `hours` hours (the endpoint actually
    returns about a week's worth - 156 periods observed - so `hours` just
    controls how much of that we keep). Unlike the trend/damping model
    above, this has real atmospheric dynamics behind it - fronts, marine
    pushes, large-scale flow - that a straight line fit through the last
    8 observations has no way to represent. Returns a DataFrame of
    (time, forecast_temp_f).
    """
    points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
    r = requests.get(points_url, headers={"Accept": "application/geo+json"})
    r.raise_for_status()
    forecast_url = r.json()["properties"]["forecastHourly"]

    r2 = requests.get(forecast_url, headers={"Accept": "application/geo+json"})
    r2.raise_for_status()
    periods = r2.json()["properties"]["periods"]

    rows = [
        {"time": datetime.fromisoformat(p["startTime"]), "forecast_temp_f": p["temperature"]}
        for p in periods[:hours]
    ]
    return pd.DataFrame(rows)


def _nws_forecast_temp_at(forecast_df, target_time):
    """The forecast's nearest-hour temp to target_time, or None if the nearest
    hour is more than an hour away (forecast doesn't cover that time)."""
    if forecast_df is None or forecast_df.empty:
        return None
    deltas = (forecast_df["time"] - target_time).abs()
    idx = deltas.idxmin()
    if deltas[idx] > timedelta(hours=1):
        return None
    return forecast_df.loc[idx, "forecast_temp_f"]


# KHQM (Hoquiam) reports reliably (checked: 8/8 recent observations with
# pressure data). KUIL (Quillayute) is the other coastal option upwind of the
# Chehalis Gap but was returning zero recent observations when checked, so it
# sits second in line as a fallback rather than the default.
UPWIND_STATION_CANDIDATES = ["KHQM", "KUIL"]


def _fetch_upwind_df(start, end, candidates=None):
    """Try each candidate upwind station in order; return (df, station_id) for
    the first with usable data in this window, or (None, None) if all fail."""
    for station in (candidates or UPWIND_STATION_CANDIDATES):
        try:
            return get_observation_history(station, start=start, end=end), station
        except Exception:
            continue
    return None, None


def _merge_and_gradient(df_local, df_other, value_col, tolerance_minutes=20):
    """
    Aligns two stations' observations by nearest timestamp within tolerance
    (same pattern used by backtest to score a projection against the closest
    real observation) and computes (local - other) for value_col plus its
    trend via np.polyfit - the same trend-fitting approach used everywhere
    else in this file. Shared by get_pressure_gradient (coastal) and
    get_station_network_signals (strait/interior_gap) so pressure and
    temperature gradients for any station pair go through one code path.

    Returns (gradient_now, gradient_trend_per_hr), or (None, None) if fewer
    than 2 aligned readings exist.
    """
    merged = pd.merge_asof(
        df_local[["time", value_col]].sort_values("time"),
        df_other[["time", value_col]].sort_values("time"),
        on="time", direction="nearest", tolerance=timedelta(minutes=tolerance_minutes),
        suffixes=("_local", "_other"),
    )
    merged = merged.dropna(subset=[f"{value_col}_local", f"{value_col}_other"])
    if len(merged) < 2:
        return None, None

    merged["gradient"] = merged[f"{value_col}_local"] - merged[f"{value_col}_other"]
    t0 = merged["time"].iloc[0]
    elapsed = (merged["time"] - t0).dt.total_seconds() / 3600
    gradient_now = merged["gradient"].iloc[-1]
    if merged["gradient"].nunique() > 1:
        gradient_trend = np.polyfit(elapsed, merged["gradient"], 1)[0]
    else:
        gradient_trend = 0.0

    return round(gradient_now, 4), round(gradient_trend, 4)


def get_pressure_gradient(df_local, df_upwind=None, upwind_station_id=None):
    """
    Marine-air intrusions into Puget Sound are driven by a pressure gradient
    between the coast and the interior: when the interior (KSEA) heats up
    faster than the coast, its pressure falls relative to the coast, and that
    widening differential pulls cool marine air inland through gaps like the
    Chehalis Gap - reaching KSEA anywhere from about 1 to 3 hours later.

    Checked empirically against the Jul 20 event: the local-minus-upwind
    (KSEA - KHQM) gradient flattened and briefly went negative in the hour
    before the cooling dip, then recovered as temperatures resumed climbing.
    So a *shrinking or negative-trending* gradient is the onshore-push
    signature here - not a widening one.

    df_upwind can be pre-fetched and passed in (e.g. by backtest, which pulls
    the whole window once rather than re-fetching per rolling window). If
    omitted, fetches a matching window around df_local's own span.

    Returns (gradient_inhg, gradient_trend_inhg_per_hr, station_used, note).
    note is None on success; otherwise a short explanation of why no gradient
    was computed, so callers can fall back to no adjustment instead of
    crashing when the upwind station is unavailable.
    """
    if df_upwind is None:
        start = df_local["time"].iloc[0] - timedelta(minutes=20)
        end = df_local["time"].iloc[-1] + timedelta(minutes=20)
        candidates = [upwind_station_id] if upwind_station_id else None
        df_upwind, used_station = _fetch_upwind_df(start, end, candidates)
        if df_upwind is None:
            return None, None, None, "no upwind station data available; no gradient adjustment applied"
    else:
        used_station = upwind_station_id or "prefetched"

    gradient_now, gradient_trend = _merge_and_gradient(df_local, df_upwind, "pressure_inhg")
    if gradient_now is None:
        return None, None, used_station, "not enough aligned upwind readings; no gradient adjustment applied"

    return gradient_now, gradient_trend, used_station, None


# ---------------------------------------------------------------------------
# Multi-station gradient network (strait / interior_gap roles)
#
# The coastal role above (get_pressure_gradient) only sees one directional
# pattern: marine air pushing in from the coast. Real Seattle-area swings
# also include the opposite pattern - warm, dry offshore/gap flow spilling
# from the interior through the Cascade passes, the actual mechanism behind
# most real heat events (not just gradual solar heating), which a single
# coastal station pair has zero visibility into. This section adds two more
# roles so the two patterns can be told apart instead of both just widening
# a single generic "something's changing" uncertainty flag.
# ---------------------------------------------------------------------------

# Each role's candidates are tried in order; the first with fresh, usable
# data wins (see _fetch_role_df). Checked directly against live data before
# picking these:
#   - strait: KCLM (Port Angeles) is the textbook Strait of Juan de Fuca
#     station, but was returning a 4-hour-stale feed when checked (only 4 of
#     the last 8 requested observations, latest ~4h old) - a real reporting
#     gap, not just infrequent cadence. KORS (Orcas Island/Eastsound) sits
#     first instead: fresh (~25 min old) and 8/8 with pressure, still well
#     within the San Juans/strait convergence zone. KCLM stays listed as a
#     fallback in case its feed recovers.
#   - interior_gap: KELN (Ellensburg) and KYKM (Yakima) were both fresh
#     (8/8 with pressure, ~5 min cadence) when checked - either is a fine
#     "east of the Cascades" station for the classic Seattle-to-Yakima/
#     Ellensburg gap-wind differential; KELN goes first arbitrarily.
STATION_NETWORK = {
    "strait": ["KORS", "KCLM"],
    "interior_gap": ["KELN", "KYKM"],
}

# A station that's technically reachable but hasn't reported in hours (like
# KCLM when checked - see above) is functionally unavailable for a *live*
# gradient, not a source of real signal - using its stale reading as if it
# were current would silently misrepresent "right now." Treated the same as
# an unreachable station: skip to the next candidate.
STALE_OBS_TOLERANCE_MINUTES = 90


def _fetch_role_df(role, start, end, candidates=None):
    """
    Try each candidate station for a role (strait/interior_gap) in order,
    skipping any that error out OR whose latest reading is stale (see
    STALE_OBS_TOLERANCE_MINUTES) relative to `end`. Same
    try-in-order-and-degrade shape as _fetch_upwind_df, generalized across
    roles and with the added freshness check.

    Returns (df, station_id) for the first usable candidate, or (None, None)
    if every candidate fails or is stale.
    """
    for station in (candidates or STATION_NETWORK[role]):
        try:
            df = get_observation_history(station, start=start, end=end)
        except Exception:
            continue
        staleness_minutes = (end - df["time"].iloc[-1]).total_seconds() / 60
        if staleness_minutes > STALE_OBS_TOLERANCE_MINUTES:
            continue
        return df, station
    return None, None


def get_station_network_signals(df_local, df_strait=None, df_interior_gap=None):
    """
    Computes the strait and interior_gap gradient signals (coastal is
    get_pressure_gradient, kept separate/unchanged above). For each role:
      - pressure gradient (local - role station) and its trend, same as the
        coastal signal.
      - temp gradient and its trend, computed ONLY for interior_gap - a
        KSEA-vs-coastal temp differential isn't the physically meaningful
        one here (see module notes on offshore flow); a KSEA-vs-interior one
        is, since a big Yakima/Ellensburg-vs-Seattle temp gap building up
        often precedes a gap-wind event reaching the coast.

    df_strait/df_interior_gap: pass a pre-fetched DataFrame to reuse it
    (e.g. backtest, which fetches each role once for the whole window rather
    than per rolling slice); pass False to mean "already tried for this
    whole run, not available - don't retry" (also for backtest, so hundreds
    of rolling-window calls don't each re-attempt and fail against the same
    down station); leave as None (the default) to fetch fresh right now -
    the live single-call path.

    Never raises: a role with no usable candidate just gets None fields and
    a note explaining why, so a missing/stale station degrades gracefully
    instead of blocking the whole estimate.

    Returns {"strait": {...}, "interior_gap": {...}}.
    """
    start = df_local["time"].iloc[0] - timedelta(minutes=20)
    end = df_local["time"].iloc[-1] + timedelta(minutes=20)

    signals = {}
    for role, prefetched in (("strait", df_strait), ("interior_gap", df_interior_gap)):
        if prefetched is False:
            df_other, station = None, None
        elif prefetched is None:
            df_other, station = _fetch_role_df(role, start, end)
        else:
            df_other, station = prefetched, "prefetched"

        if df_other is None:
            signals[role] = {
                "station": None,
                "pressure_gradient_inhg": None,
                "pressure_gradient_trend_inhg_per_hr": None,
                "temp_gradient_f": None,
                "temp_gradient_trend_f_per_hr": None,
                "note": f"no usable {role} station data available (unreachable or stale); signal not computed",
            }
            continue

        p_now, p_trend = _merge_and_gradient(df_local, df_other, "pressure_inhg")
        t_now = t_trend = None
        if role == "interior_gap":
            t_now, t_trend = _merge_and_gradient(df_local, df_other, "temp_f")

        signals[role] = {
            "station": station,
            "pressure_gradient_inhg": p_now,
            "pressure_gradient_trend_inhg_per_hr": p_trend,
            "temp_gradient_f": t_now,
            "temp_gradient_trend_f_per_hr": t_trend,
            "note": None if p_now is not None else "not enough aligned readings; gradient not computed",
        }
    return signals


# ---------------------------------------------------------------------------
# Derived composite indices
#
# First-pass formulas, not hand-tuned-and-final - see calibration_log.py's
# per-day logging of these raw values plus daily_performance.py's error
# tracking. Once enough days accumulate (including at least one real
# marine-push and, ideally, one real offshore-flow/heat event), these
# weights/thresholds should get refit from whether large peak_temp_error_f
# days actually showed an unusual reading here beforehand - not trusted as
# hand-picked constants indefinitely. Scaled arbitrarily so a "typical"
# gradient-trend event (per the coastal Jul 20 case study, about -0.01
# inHg/hr) lands around 10 on the index - purely to make the dashboard
# number legible, not a calibrated unit.
# ---------------------------------------------------------------------------

_PRESSURE_TREND_TO_INDEX_SCALE = 1000  # inHg/hr -> index points
_TEMP_TREND_TO_INDEX_SCALE = 10  # F/hr -> index points


def compute_marine_push_index(coastal_gradient_trend, strait_signal):
    """
    Positive/increasing = marine air more likely pushing in. Built from the
    same "shrinking/negative-trending gradient precedes onshore push"
    relationship get_pressure_gradient already established for the coastal
    station (see its docstring re: the Jul 20 event); the strait station's
    own KSEA-relative pressure gradient trend is averaged in as a second,
    independent read on the same push when available - a convergence-zone
    station feels the same onshore surge, just from a different angle.

    Returns None only if neither station's gradient trend is available.
    """
    trends = [t for t in (
        coastal_gradient_trend,
        strait_signal.get("pressure_gradient_trend_inhg_per_hr") if strait_signal else None,
    ) if t is not None]
    if not trends:
        return None
    avg_trend = sum(trends) / len(trends)
    return round(-avg_trend * _PRESSURE_TREND_TO_INDEX_SCALE, 1)


def compute_offshore_flow_index(interior_gap_signal):
    """
    Positive/increasing = offshore/gap flow strengthening - conditions favor
    rapid warming and dropping humidity, the classic Seattle heat-event
    mechanism (see module notes above). This is a hypothesis, not yet
    validated against a real event: this signal has no history to backtest
    against (nothing was logging it until now - see calibration_log.py). The
    working theory this first pass encodes: the interior's pressure gradient
    versus KSEA deepening, AND the interior warming faster than KSEA (its
    temp gradient trending more negative), both plausibly precede offshore
    flow reaching the coast - so both contribute.

    Returns None if neither trend is available.
    """
    if interior_gap_signal is None:
        return None
    p_trend = interior_gap_signal.get("pressure_gradient_trend_inhg_per_hr")
    t_trend = interior_gap_signal.get("temp_gradient_trend_f_per_hr")
    if p_trend is None and t_trend is None:
        return None

    contribution = 0.0
    if p_trend is not None:
        contribution += -p_trend * _PRESSURE_TREND_TO_INDEX_SCALE
    if t_trend is not None:
        contribution += -t_trend * _TEMP_TREND_TO_INDEX_SCALE
    return round(contribution, 1)


# ---------------------------------------------------------------------------
# Damping based on real sunrise/sunset
# ---------------------------------------------------------------------------

# Fraction of the sunrise-to-sunset span at which the day's peak heat
# typically falls. Checked against a week of real KSEA data (excluding one
# day whose "max" landed at midnight - a calendar-day-boundary artifact from
# a day that never really warmed, not informative about peak timing): 6
# valid days averaged 0.708 (range 0.656-0.735), consistently well past the
# previous hardcoded 0.65 - which was landing estimated peak-heat hour
# 30-45 minutes earlier than where the actual daily max was showing up.
PEAK_HEAT_FRACTION = 0.70

# Floor and ramp of diurnal_damping_factor. The floor is the multiplier
# applied to the raw trend when the projection window touches an
# inflection point (sunrise or peak-heat hour); the ramp is how many
# hours of distance from that point it takes to fade back to trusting
# the raw trend fully.
#
# Baseline floor/ramp for diurnal_damping_factor, governing ordinary
# estimate_temp calls at arbitrary target times. Left at the original
# 0.35: a backtest sweep found its bias on those calls was already near
# zero (-0.27F at 3h lead), so raising it bought nothing and cost MAE
# (3.58 -> 3.76F at 3h, 7.39 -> 8.03F at 6h). Damping hard here is
# correct - when the window sits ON a turning point the measured slope
# is about to reverse.
DIURNAL_DAMPING_FLOOR = 0.35
DIURNAL_DAMPING_RAMP_HOURS = 6.0

# Separate, looser floor used ONLY when projecting the daily high, where
# the target time IS the peak-heat hour. That case is structurally
# different and was structurally broken: because dist_target is 0 by
# construction, closest_approach is 0 on every such call and the factor
# was pinned at exactly the baseline 0.35 - not a worst case but the only
# case - keeping ~35% of the trend's remaining rise, less again after
# cloud/wind damping. Measured against CLISEA actuals that compressed the
# predicted daily range by 3.1F at 3h lead and 4.6F at 4h.
#
# The physical difference justifying a separate constant: a window
# sitting on a turning point has an unreliable slope (baseline case,
# damp hard), whereas a window mid-rise aiming at the peak has a
# perfectly well-measured slope and only needs discounting for the
# curve flattening as it approaches the top. Those want different
# numbers, and conflating them is what produced the compression.
#
# 0.70 is the mean-MAE minimum of a 0.35-1.00 sweep by
# backtest_peak_projection over 6 days. Bias and MAE both improve at
# every lead time tested; past ~0.8 bias keeps shrinking but MAE at 4h
# climbs sharply, so this is not simply "as high as possible".
#
# What it does NOT fix: compression at 6h+ lead, which stays near -6F at
# 6h and -11.5F at 8h at any floor, and barely moves when
# TREND_HORIZON_HOURS goes from 6 to 12. Six-plus hours before the peak
# it is still early morning and the local trend carries no information
# about the afternoon high. That is a real limit of a local-trend model,
# not a mistuned constant. Closing it means leaning harder on the NWS
# blend at long lead, which cannot be backtested here (no historical
# forecasts - see estimate_from_df) and so has not been changed on a
# guess.
PEAK_PROJECTION_DAMPING_FLOOR = 0.70


def _hour_to_datetime(base_date, hour_decimal, tzinfo):
    """Convert a decimal hour (e.g. 15.65) on a given date into an aware datetime."""
    h = int(hour_decimal) % 24
    m = int(round((hour_decimal - int(hour_decimal)) * 60))
    if m == 60:
        m = 0
        h = (h + 1) % 24
    return datetime.combine(base_date, time(h, m), tzinfo=tzinfo)


def diurnal_damping_factor(current_time, hours_ahead, lat, lon,
                           floor=None, ramp=None):
    """
    Multiplier applied to the raw trend. Damps hardest when the window
    crosses near actual sunrise (bottom of curve) or a few hours after
    sunrise-to-sunset midpoint (rough proxy for peak heating), using real
    sun times for that date instead of hardcoded hours.

    floor/ramp default to DIURNAL_DAMPING_FLOOR / DIURNAL_DAMPING_RAMP_HOURS
    and exist so backtest_peak_projection can sweep them without
    monkey-patching module state.

    Note the interaction that made the floor matter so much: when this is
    called to project the DAILY HIGH, target_hour is the peak-heat hour by
    construction, so dist_target is 0 and closest_approach is 0 on every
    such call - the factor is pinned at exactly `floor` regardless of lead
    time or anything else. The floor is therefore not a worst case for
    peak projection; it is the only case. See DIURNAL_DAMPING_FLOOR.
    """
    if floor is None:
        floor = DIURNAL_DAMPING_FLOOR
    if ramp is None:
        ramp = DIURNAL_DAMPING_RAMP_HOURS
    sunrise_h, sunset_h = get_sun_times(lat, lon, current_time.date())
    peak_h = sunrise_h + (sunset_h - sunrise_h) * PEAK_HEAT_FRACTION

    hour = current_time.hour + current_time.minute / 60
    target_hour = (hour + hours_ahead) % 24

    def circular_dist(a, b):
        diff = abs(a - b) % 24
        return min(diff, 24 - diff)

    inflection_points = [sunrise_h, peak_h]
    dist_now = min(circular_dist(hour, p) for p in inflection_points)
    dist_target = min(circular_dist(target_hour, p) for p in inflection_points)

    closest_approach = min(dist_now, dist_target)
    damping = min(1.0, floor + closest_approach / ramp)
    return damping


def _expected_trend_sign(current_time, lat, lon):
    """
    +1 during the sunrise-to-peak stretch of the day (temperature should be
    climbing), -1 the rest of the time - peak through the next sunrise
    (temperature should be falling toward the overnight low, then flat/rising
    again right at dawn, which the sunrise boundary already accounts for).
    """
    sunrise_h, sunset_h = get_sun_times(lat, lon, current_time.date())
    peak_h = sunrise_h + (sunset_h - sunrise_h) * PEAK_HEAT_FRACTION
    hour = current_time.hour + current_time.minute / 60
    return 1 if sunrise_h <= hour < peak_h else -1


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def _cloud_wind_damping(df):
    """
    Clear skies + calm air let temperature swing further (fast radiative
    heating/cooling). Overcast skies and/or wind mix the near-surface air
    and suppress the swing. Returns a multiplier (0.5-1.0) applied on top
    of the diurnal damping.
    """
    latest = df.iloc[-1]
    cloud = latest.get("cloud_fraction")
    wind = latest.get("wind_mph")

    factor = 1.0
    if pd.notna(cloud):
        # overcast (1.0) suppresses swing by up to 35%; clear (0.0) suppresses none
        factor *= (1 - 0.35 * cloud)
    if pd.notna(wind):
        # mixing effect, saturates - 20mph wind roughly halves the swing
        factor *= 1 / (1 + wind / 20)

    return max(0.5, factor)


# First-pass thresholds (see the indices' own docstrings re: calibration) -
# an index above this is "elevated enough to call out," not a calibrated
# cutoff.
MARINE_PUSH_INDEX_THRESHOLD = 8.0
OFFSHORE_FLOW_INDEX_THRESHOLD = 8.0


# Weight on the fitted slope's own standard error when it enters the
# uncertainty band. 1.0 = propagate it at face value (plain OLS error
# propagation); 0.0 disables the term, which is how the before/after
# backtest for this change was run. Not a fudge factor to tune away a
# coverage shortfall - if the band still under-covers at 1.0, that is the
# model being genuinely overconfident, not this constant being wrong.
TREND_SE_WEIGHT = 1.0


def _slope_standard_error(x, y, slope, intercept):
    """Standard error of the OLS slope: s / sqrt(Sxx), with s the residual
    standard deviation.

    This is the piece the band was missing. The trend is fit on 8
    observations that are themselves quantised to whole degrees Celsius
    (+/-0.9F - see observation_precision), and that noise propagates
    straight into the slope: on a representative window, +/-0.9F of
    reading noise alone moves the fitted slope by sd ~0.97 F/hr against a
    slope of -4.11 F/hr. Multiplied out over the projection horizon that
    is degrees of temperature uncertainty the band never accounted for.

    Returns None when the fit is too short or degenerate to have a
    meaningful standard error.
    """
    n = len(x)
    if n < 3:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sxx = float(((x - x.mean()) ** 2).sum())
    if sxx <= 0:
        return None
    resid = y - (slope * x + intercept)
    s2 = float((resid ** 2).sum()) / (n - 2)
    if s2 < 0:
        return None
    return float(np.sqrt(s2 / sxx))


def _pressure_trend_and_uncertainty(df, elapsed_hours, hours_ahead, marine_push_index=None, offshore_flow_index=None, trend_uncertainty_f=None):
    """
    Falling pressure signals a front/unsettled system may be approaching,
    but not which direction temp will move - so this widens the uncertainty
    band around the point estimate rather than shifting it.

    marine_push_index/offshore_flow_index (see compute_marine_push_index/
    compute_offshore_flow_index), when elevated, widen uncertainty further -
    each independently, since they're distinguishable physical patterns
    (marine push vs. offshore/gap flow), not the same "something's changing"
    flag counted twice. This replaces the old single coastal-gradient-only
    widening: marine_push_index already folds that same coastal reading in
    (plus the strait station when available), so this is a superset, not an
    addition on top of it.

    Returns (pressure_trend_inhg_per_hr, uncertainty_f, uncertainty_note).
    uncertainty_note is a short explanation of what's driving uncertainty
    beyond the horizon-only baseline (or None if nothing beyond baseline
    applied), so the dashboard doesn't just show a wider band with no
    explanation of which pattern is implicated.
    """
    base_uncertainty = 1.0 + 0.3 * hours_ahead  # baseline grows with horizon
    notes = []

    if df["pressure_inhg"].notna().sum() < 2:
        pressure_trend, uncertainty = None, base_uncertainty
    else:
        valid = df["pressure_inhg"].notna()
        p_slope = np.polyfit(elapsed_hours[valid], df["pressure_inhg"][valid], 1)[0]

        # a drop of ~0.03 inHg/hr or faster is a reasonably brisk pressure fall
        if p_slope < 0:
            uncertainty = base_uncertainty + min(2.0, abs(p_slope) * 40)
            notes.append("local pressure falling")
        else:
            uncertainty = base_uncertainty
        pressure_trend = round(p_slope, 4)

    if marine_push_index is not None and marine_push_index > MARINE_PUSH_INDEX_THRESHOLD:
        uncertainty += min(1.5, marine_push_index / 20)
        notes.append("marine push signal rising")

    if offshore_flow_index is not None and offshore_flow_index > OFFSHORE_FLOW_INDEX_THRESHOLD:
        uncertainty += min(1.5, offshore_flow_index / 20)
        notes.append("offshore flow signal rising")

    # The heuristic terms above are all "some pattern suggests conditions
    # are less predictable". The slope's standard error is a different
    # kind of thing - a statistical property of the fit itself - so it
    # combines in quadrature as an independent source rather than being
    # added on like another flag.
    if trend_uncertainty_f:
        widened = math.sqrt(uncertainty ** 2 + trend_uncertainty_f ** 2)
        if widened - uncertainty >= 0.3:
            notes.append("noisy local trend fit")
        uncertainty = widened

    uncertainty_note = "; ".join(notes) if notes else None
    return pressure_trend, round(uncertainty, 1), uncertainty_note


# A short local trend (fit over the last handful of observations) is only a
# credible predictor a few hours out. Projections further ahead than this
# still use the real hours_ahead for figuring out *where* in the diurnal
# cycle the target time falls, but the trend's contribution to the magnitude
# of the change is capped at this many hours - otherwise a small slope
# measured over the last 20 minutes gets multiplied out to an absurd swing
# over a 12+ hour projection (e.g. projecting to the next sunrise).
TREND_HORIZON_HOURS = 6


def estimate_from_df(
    df, hours_ahead, lat, lon,
    use_nws_forecast=True, nws_blend_mode="divergence",
    use_gradient=True, df_upwind=None, df_strait=None, df_interior_gap=None,
    damping_floor=None,
):
    """
    Core estimation logic, given a dataframe of observations. Reused by both
    the live estimator and the backtest.

    use_nws_forecast: blend the trend/damping estimate with the NWS hourly
        gridpoint forecast at target_time (see get_hourly_forecast). The
        backtest disables this - forecastHourly only exposes the forecast as
        issued right now, so there's no historical "what did the forecast say
        3 hours before this point" to test against; blending it into a
        backtest would either silently score today's forecast against past
        observations (meaningless) or require fabricating a proxy forecast
        history (more misleading than admitting the gap).
    nws_blend_mode: "fixed" always blends 50/50. "divergence" blends 50/50
        normally but shifts to 80% forecast / 20% trend when the two disagree
        by more than 3F, on the theory that a big gap means the trend is
        missing something (a front, a marine push) the forecast model's real
        atmospheric dynamics can see.
    use_gradient: master switch for the whole cross-station network
        (coastal + strait + interior_gap) - each role still degrades
        independently if its own candidates are unavailable (see
        get_station_network_signals); this only gates whether any of it is
        attempted at all.
    df_upwind / df_strait / df_interior_gap: see get_pressure_gradient /
        get_station_network_signals. Let a caller (backtest) pre-fetch each
        role's data once instead of re-fetching per rolling window.
    damping_floor: override for DIURNAL_DAMPING_FLOOR, so
        backtest_peak_projection can sweep it. None uses the module value.
    """
    latest = df.iloc[-1]
    now = latest["time"]

    t0 = df["time"].iloc[0]
    elapsed_hours = (df["time"] - t0).dt.total_seconds() / 3600
    slope, intercept = np.polyfit(elapsed_hours, df["temp_f"], 1)

    spread_adjustment = 0
    if df["dewpoint_f"].notna().sum() >= 2:
        spread = df["temp_f"] - df["dewpoint_f"]
        spread_slope = np.polyfit(elapsed_hours, spread, 1)[0]
        if slope < 0 and spread_slope < 0:
            spread_adjustment = min(0.5, abs(spread_slope) * hours_ahead * 0.1)

    diurnal_damping = diurnal_damping_factor(now, hours_ahead, lat, lon, floor=damping_floor)
    sky_wind_damping = _cloud_wind_damping(df)
    combined_damping = diurnal_damping * sky_wind_damping

    # diurnal_damping_factor only weighs proximity to an inflection point, so
    # it fades back toward 1.0 (trusting the raw trend) the further past
    # peak/sunrise you get - even if that raw trend (fit on the last several
    # minutes) is still pointed the "wrong" way for the time of day, e.g.
    # still reading warming at 6pm, well after peak-heat hour. That's a much
    # weaker signal than a trend already pointed the expected direction, so
    # discount it further here rather than let the proximity fade-out alone
    # decide how much to trust it.
    if slope != 0 and np.sign(slope) != _expected_trend_sign(now, lat, lon):
        combined_damping *= 0.4

    gradient_now = gradient_trend = gradient_station = gradient_note = None
    network_signals = {"strait": None, "interior_gap": None}
    marine_push_index = offshore_flow_index = None
    if use_gradient:
        gradient_now, gradient_trend, gradient_station, gradient_note = get_pressure_gradient(
            df, df_upwind=df_upwind
        )
        network_signals = get_station_network_signals(
            df, df_strait=df_strait, df_interior_gap=df_interior_gap
        )
        marine_push_index = compute_marine_push_index(gradient_trend, network_signals["strait"])
        offshore_flow_index = compute_offshore_flow_index(network_signals["interior_gap"])

    # Propagate the slope's own uncertainty through exactly the same path
    # the slope itself takes: capped at TREND_HORIZON_HOURS and scaled by
    # the same damping, because that is what multiplies the slope into a
    # temperature change.
    slope_se = _slope_standard_error(elapsed_hours, df["temp_f"], slope, intercept)
    trend_uncertainty_f = (
        slope_se * min(hours_ahead, TREND_HORIZON_HOURS) * combined_damping * TREND_SE_WEIGHT
        if slope_se is not None else None
    )

    pressure_trend, uncertainty_f, uncertainty_note = _pressure_trend_and_uncertainty(
        df, elapsed_hours, hours_ahead,
        marine_push_index=marine_push_index, offshore_flow_index=offshore_flow_index,
        trend_uncertainty_f=trend_uncertainty_f,
    )

    raw_change = slope * min(hours_ahead, TREND_HORIZON_HOURS)
    damped_change = raw_change * combined_damping + spread_adjustment * np.sign(raw_change) * -1

    trend_estimate = latest["temp_f"] + damped_change
    target_time = now + timedelta(hours=hours_ahead)

    estimated_temp = trend_estimate
    nws_forecast_temp = None
    blend_weight_used = None
    nws_note = "disabled for this call" if not use_nws_forecast else None
    if use_nws_forecast:
        try:
            forecast_df = get_hourly_forecast(lat, lon)
            nws_forecast_temp = _nws_forecast_temp_at(forecast_df, target_time)
        except Exception:
            nws_forecast_temp = None
        if nws_forecast_temp is None:
            nws_note = "NWS hourly forecast unavailable or target_time out of its range; using trend estimate only"
        else:
            if nws_blend_mode == "fixed":
                w = 0.5
            else:
                w = 0.2 if abs(trend_estimate - nws_forecast_temp) > 3 else 0.5
            blend_weight_used = w
            estimated_temp = w * trend_estimate + (1 - w) * nws_forecast_temp

    return {
        "as_of": now,
        "target_time": target_time,
        "current_temp_f": round(latest["temp_f"], 1),
        "trend_estimate_f": round(trend_estimate, 1),
        "estimated_temp_f": round(estimated_temp, 1),
        "estimated_range_f": (
            round(estimated_temp - uncertainty_f, 1),
            round(estimated_temp + uncertainty_f, 1),
        ),
        "raw_trend_f_per_hr": round(slope, 2),
        # (c) - how well-determined that slope actually is, and what it
        # contributed to the band. Displayed rather than buried, because a
        # trend of -4.1 F/hr +/-1.0 is a different claim from -4.1 +/-0.1.
        "trend_slope_se_f_per_hr": round(slope_se, 2) if slope_se is not None else None,
        "trend_uncertainty_f": round(trend_uncertainty_f, 2) if trend_uncertainty_f else None,
        "diurnal_damping": round(diurnal_damping, 2),
        "sky_wind_damping": round(sky_wind_damping, 2),
        "cloud_fraction": latest.get("cloud_fraction") if pd.notna(latest.get("cloud_fraction")) else None,
        "wind_mph": round(latest["wind_mph"], 1) if pd.notna(latest.get("wind_mph")) else None,
        "pressure_trend_inhg_per_hr": pressure_trend,
        "pressure_gradient_inhg": gradient_now,
        "pressure_gradient_trend_inhg_per_hr": gradient_trend,
        "pressure_gradient_station": gradient_station,
        "pressure_gradient_note": gradient_note,
        "strait_signal": network_signals["strait"],
        "interior_gap_signal": network_signals["interior_gap"],
        "marine_push_index": marine_push_index,
        "offshore_flow_index": offshore_flow_index,
        "uncertainty_note": uncertainty_note,
        "nws_forecast_temp_f": round(float(nws_forecast_temp), 1) if nws_forecast_temp is not None else None,
        "blend_weight_used": blend_weight_used,
        "nws_forecast_note": nws_note,
        "n_observations": len(df),
    }


def estimate_temp(
    station_id, hours_ahead=3, obs_limit=8,
    use_nws_forecast=True, nws_blend_mode="divergence", use_gradient=True,
):
    lat, lon, name = get_station_location(station_id)
    df = get_observation_history(station_id, limit=obs_limit)
    result = estimate_from_df(
        df, hours_ahead, lat, lon,
        use_nws_forecast=use_nws_forecast, nws_blend_mode=nws_blend_mode, use_gradient=use_gradient,
    )

    # Reconcile the headline number against the same peak/trough curve shown
    # elsewhere on the dashboard (see estimate_day_curve), so this can't show
    # a "trend by X" that overshoots the stated peak or undershoots the
    # stated trough - the raw_trend/damping/confidence figures above still
    # explain *why*, they just no longer double as the displayed number.
    curve = estimate_day_curve(station_id, obs_limit=obs_limit)
    reconciled_temp = curve["at"](result["target_time"])
    half_band = (result["estimated_range_f"][1] - result["estimated_range_f"][0]) / 2

    # result["trend_estimate_f"] is already the pure pre-NWS-blend trend value
    # from estimate_from_df; add the blended-but-not-yet-reconciled value too
    # so all three stages (trend only -> +NWS blend -> +peak/trough
    # reconciliation) stay inspectable.
    result["pre_reconciliation_temp_f"] = result["estimated_temp_f"]
    result["estimated_temp_f"] = round(reconciled_temp, 1)
    result["estimated_range_f"] = (
        round(reconciled_temp - half_band, 1),
        round(reconciled_temp + half_band, 1),
    )
    result["station"] = station_id.upper()
    result["station_name"] = name
    return result


def estimate_daily_extremes(station_id, obs_limit=8):
    """
    Estimate today's high and the next overnight low.

    An extreme that has already happened today is just the observed value -
    no model needed. For one still ahead:

    - High: project with the same damped-trend model as estimate_temp, but
      capped at TREND_HORIZON_HOURS out. Peak-heat hour can be many hours
      away (e.g. checking at 6am), and this model's short local trend isn't
      a credible predictor that far out, so this reports a near-term "at
      least this warm" floor rather than pretending to see all the way to
      peak.
    - Low (only when today's low has already happened, so we're forecasting
      the *next* one, which may be many hours away across sunset): trend
      extrapolation has the same problem, and additionally the recent local
      slope is often still warming at that point, which would extrapolate
      into a "low" warmer than the current temperature. Prefers the NWS
      forecast's temp at the (sun-derived) estimated low time; falls back to
      a radiative-cooling heuristic if that's unavailable - on a clear, calm
      night the overnight minimum tends toward the dewpoint (further cooling
      slows as air nears saturation), clouds/wind suppress that drop. Reuses
      the same cloud/wind damping factor as the short-term model, just aimed
      at a different physical effect.
    """
    lat, lon, name = get_station_location(station_id)
    df = get_observation_history(station_id, limit=obs_limit)
    now = df["time"].iloc[-1]
    today = now.date()

    midnight = datetime.combine(today, time(0, 0), tzinfo=now.tzinfo)
    today_obs = get_observation_history(station_id, start=midnight, end=now)
    observed_high = today_obs["temp_f"].max()
    observed_low = today_obs["temp_f"].min()
    observed_high_time = today_obs.loc[today_obs["temp_f"].idxmax(), "time"]
    observed_low_time = today_obs.loc[today_obs["temp_f"].idxmin(), "time"]

    yesterday = today - timedelta(days=1)
    yesterday_start = datetime.combine(yesterday, time(0, 0), tzinfo=now.tzinfo)
    try:
        yesterday_obs = get_observation_history(station_id, start=yesterday_start, end=midnight)
        yesterday_high = yesterday_obs["temp_f"].max()
        yesterday_low = yesterday_obs["temp_f"].min()
        yesterday_high_time = yesterday_obs.loc[yesterday_obs["temp_f"].idxmax(), "time"]
        yesterday_low_time = yesterday_obs.loc[yesterday_obs["temp_f"].idxmin(), "time"]
    except ValueError:
        yesterday_high = yesterday_low = yesterday_high_time = yesterday_low_time = None

    sunrise_today, sunset_today = get_sun_times(lat, lon, today)
    peak_today = sunrise_today + (sunset_today - sunrise_today) * PEAK_HEAT_FRACTION
    hour = now.hour + now.minute / 60

    high_time = _hour_to_datetime(today, peak_today, now.tzinfo)

    if hour < peak_today:
        horizon = min(peak_today - hour, TREND_HORIZON_HOURS)
        # PEAK_PROJECTION_DAMPING_FLOOR, not the baseline: this call aims
        # at the peak-heat hour itself, the case diurnal_damping_factor
        # pins to the floor by construction. See that constant.
        peak_est = estimate_from_df(
            df, horizon, lat, lon, damping_floor=PEAK_PROJECTION_DAMPING_FLOOR
        )
        estimated_high = max(observed_high, peak_est["estimated_temp_f"])
        high_status = "projected"  # today's peak-heat hour hasn't happened yet
    else:
        estimated_high = observed_high
        high_status = "observed"  # today's peak-heat hour has passed

    if hour < sunrise_today:
        hours_to_low = sunrise_today - hour
        low_est = estimate_from_df(df, hours_to_low, lat, lon)
        estimated_low = min(observed_low, low_est["estimated_temp_f"])
        low_status = "today"  # still before dawn; today's low is imminent
        low_time = _hour_to_datetime(today, sunrise_today, now.tzinfo)
        low_source = "trend_model"  # imminent (a few hours out at most) - the short-term trend model is fine here
    else:
        low_status = "tonight"  # today's low already happened; forecasting the next one
        tomorrow = today + timedelta(days=1)
        sunrise_tomorrow, _ = get_sun_times(lat, lon, tomorrow)
        low_time = _hour_to_datetime(tomorrow, sunrise_tomorrow, now.tzinfo)

        # Prefer the actual NWS forecast's temp at the sun-computed low_time
        # (same reasoning as tomorrow's high: real atmospheric dynamics beat
        # a heuristic). Keeping low_time itself sun-derived - not the
        # forecast's own argmin - matches "estimated hottest/coldest time of
        # day, based on the sun" elsewhere on the dashboard; only the value
        # at that time changes source.
        try:
            forecast_df = get_hourly_forecast(lat, lon, hours=24)
            nws_low = _nws_forecast_temp_at(forecast_df, low_time)
            if nws_low is None:
                raise ValueError("forecast doesn't cover the estimated low time")
            estimated_low = float(nws_low)
            low_source = "nws_forecast"
        except Exception:
            # Forecast unavailable - fall back to the radiative-cooling
            # heuristic: on a clear, calm night the overnight minimum tends
            # toward the dewpoint (further cooling slows as air nears
            # saturation); clouds/wind suppress that drop. Reuses the same
            # cloud/wind damping factor as the short-term model, just aimed
            # at a different physical effect.
            latest = df.iloc[-1]
            current_temp = latest["temp_f"]
            current_dewpoint = latest.get("dewpoint_f")
            if pd.notna(current_dewpoint):
                sky_wind = _cloud_wind_damping(df)  # 0.5 (cloudy/windy) .. 1.0 (clear/calm)
                cooling_fraction = 0.3 + 0.5 * (sky_wind - 0.5) / 0.5
                gap = max(0.0, current_temp - current_dewpoint)
                estimated_low = current_temp - gap * cooling_fraction
            else:
                estimated_low = current_temp
            low_source = "dewpoint_fallback"

    # NWS's own hourly-forecast value at THIS model's still-pending
    # extreme's target time (today's peak-heat hour, or dawn if the low
    # hasn't happened yet) - captured here, at the same checkpoint
    # calibration_log.py logs this model's own last pre-peak/pre-dawn
    # prediction, so the two can be compared against the same eventual
    # actual later (see daily_performance.py's nws_forecast_error_f).
    # Only meaningful while that extreme is still pending: once it's
    # already observed, there's no forecast-vs-actual comparison left to
    # make for that checkpoint. NWS's forecastHourly only covers the
    # future, so this can't be reconstructed retroactively - it has to be
    # captured live, right here, or the comparison is lost for that day.
    nws_high_forecast_at_target_f = None
    nws_low_forecast_at_target_f = None
    if high_status == "projected" or low_status == "today":
        try:
            pending_forecast_df = get_hourly_forecast(lat, lon, hours=24)
            if high_status == "projected":
                v = _nws_forecast_temp_at(pending_forecast_df, high_time)
                nws_high_forecast_at_target_f = round(float(v), 1) if v is not None else None
            if low_status == "today":
                v = _nws_forecast_temp_at(pending_forecast_df, low_time)
                nws_low_forecast_at_target_f = round(float(v), 1) if v is not None else None
        except Exception:
            pass  # NWS forecast unavailable - leave both None, not a hard failure

    # Tomorrow's high and low: prefer the actual NWS gridpoint forecast
    # (HRRR-based, real atmospheric dynamics - it can see a heat event
    # building that nothing in this station's own recent data would show
    # any sign of). Fetch far enough out (48h) to cover all of tomorrow
    # regardless of what time "now" is, then take the max/min forecasted
    # temp within tomorrow's calendar date. Both come from the same fetch,
    # so they succeed or fall back together.
    tomorrow = today + timedelta(days=1)
    tomorrow_high_time = None
    tomorrow_low_time = None
    tomorrow_source = "nws_forecast"
    tomorrow_low_source = "nws_forecast"
    try:
        forecast_df = get_hourly_forecast(lat, lon, hours=48)
        tomorrow_forecast = forecast_df[forecast_df["time"].dt.date == tomorrow]
        if tomorrow_forecast.empty:
            raise ValueError("forecast didn't include tomorrow's date")
        peak_row = tomorrow_forecast.loc[tomorrow_forecast["forecast_temp_f"].idxmax()]
        tomorrow_high = float(peak_row["forecast_temp_f"])
        tomorrow_high_time = peak_row["time"]
        tomorrow_confidence = 75  # grounded in an actual forecast model, not a guess - but still next-day

        trough_row = tomorrow_forecast.loc[tomorrow_forecast["forecast_temp_f"].idxmin()]
        tomorrow_low = float(trough_row["forecast_temp_f"])
        tomorrow_low_time = trough_row["time"]
        tomorrow_low_confidence = 75
    except Exception:
        # Forecast unavailable - fall back to the old persistence guess
        # (today's/yesterday's high or low, nudged by the current pressure
        # trend) rather than crash. Confidence is capped low here on
        # purpose: a short local trend genuinely can't see a day ahead on
        # its own.
        #
        # Anchor on today's high/low only once each is actually resolved
        # (high_status == "observed" / low_status == "tonight"); until then
        # "estimated_high"/"estimated_low" are deliberately conservative
        # near-term values (see above), not stand-ins for the day's
        # eventual extreme, and using them here would make tomorrow's guess
        # track that same not-yet-settled number - e.g. checked at 6am,
        # it would anchor on a not-yet-warmed-up ~60s reading instead of a
        # real high. Yesterday's actual high/low is a much better baseline
        # then.
        tomorrow_source = "persistence_fallback"
        tomorrow_low_source = "persistence_fallback"
        if high_status == "observed":
            persistence_high = estimated_high
        elif yesterday_high is not None:
            persistence_high = yesterday_high
        else:
            persistence_high = estimated_high

        if low_status == "tonight":
            persistence_low = estimated_low
        elif yesterday_low is not None:
            persistence_low = yesterday_low
        else:
            persistence_low = estimated_low

        t0 = df["time"].iloc[0]
        elapsed_hours_all = (df["time"] - t0).dt.total_seconds() / 3600
        pressure_trend, _, _ = _pressure_trend_and_uncertainty(df, elapsed_hours_all, 24)
        if pressure_trend is None:
            pressure_adj, tomorrow_confidence = 0.0, 45
        elif pressure_trend < -0.015:
            pressure_adj, tomorrow_confidence = -2.0, 35  # falling pressure: system/front likely changing things
        elif pressure_trend > 0.015:
            pressure_adj, tomorrow_confidence = 1.0, 55  # rising pressure: current pattern more likely to hold
        else:
            pressure_adj, tomorrow_confidence = 0.0, 50
        tomorrow_low_confidence = tomorrow_confidence
        tomorrow_high = persistence_high + pressure_adj
        tomorrow_low = persistence_low + pressure_adj

    return {
        "as_of": now,
        "station": station_id.upper(),
        "station_name": name,
        "estimated_high_f": round(estimated_high, 1),
        "estimated_high_time": high_time,  # theoretical peak-heat hour for today, from sun position - not tied to when the observed high actually occurred
        "estimated_low_time": low_time,  # theoretical sunrise (today's or tomorrow's) - not tied to when the observed low actually occurred
        "high_status": high_status,
        "nws_high_forecast_at_target_f": nws_high_forecast_at_target_f,  # NWS's own forecast for high_time, only while high_status == "projected"
        "estimated_low_f": round(estimated_low, 1),
        "low_status": low_status,
        "nws_low_forecast_at_target_f": nws_low_forecast_at_target_f,  # NWS's own forecast for low_time, only while low_status == "today"
        "low_source": low_source,  # "trend_model", "nws_forecast", or "dewpoint_fallback"
        "observed_high_so_far_f": round(observed_high, 2),
        "observed_high_so_far_time": observed_high_time,  # when that actual high was recorded
        "observed_low_so_far_f": round(observed_low, 2),
        "observed_low_so_far_time": observed_low_time,  # when that actual low was recorded
        "tomorrow_high_f": round(tomorrow_high, 1),
        "tomorrow_high_time": tomorrow_high_time,  # only set when tomorrow_source == "nws_forecast"
        "tomorrow_high_confidence_pct": tomorrow_confidence,
        "tomorrow_high_source": tomorrow_source,  # "nws_forecast" or "persistence_fallback"
        "tomorrow_low_f": round(tomorrow_low, 1),
        "tomorrow_low_time": tomorrow_low_time,  # only set when tomorrow_low_source == "nws_forecast"
        "tomorrow_low_confidence_pct": tomorrow_low_confidence,
        "tomorrow_low_source": tomorrow_low_source,  # "nws_forecast" or "persistence_fallback"
        "yesterday_high_f": round(yesterday_high, 2) if yesterday_high is not None else None,
        "yesterday_high_time": yesterday_high_time,
        "yesterday_low_f": round(yesterday_low, 2) if yesterday_low is not None else None,
        "yesterday_low_time": yesterday_low_time,
    }


def estimate_day_curve(station_id, obs_limit=8):
    """
    A single curve for "what will the temperature be at any future time
    today/tonight," so that estimate_temp's "trend by X" projection and
    estimate_daily_extremes' peak/trough figures can't contradict each other
    the way they used to: previously a "trend by X" 3-hour projection just
    kept extrapolating the current damped slope past the point where the
    separately-computed peak-hour estimate said the curve should already be
    turning over (e.g. showing 93F at 5:20pm when the peak was estimated at
    90F at 3:33pm).

    Why this isn't just "sweep estimate_from_df across many hours_ahead and
    take the max": that formula's damping only ever shrinks the *magnitude*
    of the current trend, never flips its *sign* - swept out far enough it
    just plateaus (verified: swept to 103.9F by 6h out and stayed there
    through 2am), it never comes back down on its own. So instead this reuses
    the peak/trough *values* that estimate_daily_extremes already computes
    (which do properly turn over, since they're anchored to sunrise/peak-hour
    and the dewpoint-cooling heuristic rather than a straight-line trend) as
    fixed points, and interpolates between them with a half-cosine ease -
    the same slow-near-the-extremes, faster-in-between shape a real diurnal
    cycle has. That interpolation can never exceed the higher of two
    consecutive anchors or undershoot the lower one, so the "can't overshoot
    the peak/undershoot the trough" property this bug needs falls out of the
    curve's shape rather than needing a separate clamp bolted on.

    Returns a dict with the anchor points used and an `at(target_time)`
    function to evaluate the curve at any future time within its span
    (holds flat at the last anchor's value beyond it, rather than
    extrapolating blindly).
    """
    extremes = estimate_daily_extremes(station_id, obs_limit=obs_limit)
    now = extremes["as_of"]
    lat, lon, name = get_station_location(station_id)
    df = get_observation_history(station_id, limit=obs_limit)
    current_temp = df["temp_f"].iloc[-1]

    anchors = [(now, current_temp)]
    future_anchors = []
    if extremes["high_status"] == "projected":  # peak hasn't happened yet - it's a real future anchor
        future_anchors.append((extremes["estimated_high_time"], extremes["estimated_high_f"]))
    future_anchors.append((extremes["estimated_low_time"], extremes["estimated_low_f"]))  # always still ahead
    future_anchors.sort(key=lambda pair: pair[0])
    anchors.extend(future_anchors)

    def at(target_time):
        if target_time <= anchors[0][0]:
            return anchors[0][1]
        for (t_a, v_a), (t_b, v_b) in zip(anchors, anchors[1:]):
            if t_a <= target_time <= t_b:
                span_seconds = (t_b - t_a).total_seconds()
                frac = 0.5 if span_seconds == 0 else (target_time - t_a).total_seconds() / span_seconds
                eased = (1 - math.cos(frac * math.pi)) / 2  # slow at the ends, fast in the middle
                return v_a + (v_b - v_a) * eased
        return anchors[-1][1]  # beyond the last anchor: hold flat rather than guess

    return {
        "as_of": now,
        "station": station_id.upper(),
        "station_name": name,
        "anchors": anchors,
        "at": at,
    }


def average_midday_growth(station_id, lookback_days=5, tolerance_minutes=30):
    """
    Pull `lookback_days` of historical observations and compute the average
    temperature change from 12pm->2pm and 2pm->4pm across those days - a
    simple empirical check on how much the model's peak-heat window actually
    tends to warm, independent of the trend/damping machinery elsewhere in
    this file.

    A day is only included if it has an observation within
    `tolerance_minutes` of all three of noon, 2pm, and 4pm local time.
    """
    end = datetime.now(PST)
    start = end - timedelta(days=lookback_days)
    df = get_observation_history(station_id, start=start, end=end)
    df = df.copy()
    df["date"] = df["time"].dt.date

    def closest_at(day_df, target_hour):
        day = day_df["time"].iloc[0].date()
        tzinfo = day_df["time"].iloc[0].tzinfo
        target = datetime.combine(day, time(target_hour, 0), tzinfo=tzinfo)
        deltas = (day_df["time"] - target).abs()
        idx = deltas.idxmin()
        if deltas[idx] > timedelta(minutes=tolerance_minutes):
            return None
        return day_df.loc[idx]

    growth_12_2, growth_2_4, per_day = [], [], []

    for day, day_df in df.groupby("date"):
        noon = closest_at(day_df, 12)
        two = closest_at(day_df, 14)
        four = closest_at(day_df, 16)
        if noon is None or two is None or four is None:
            continue

        g1 = two["temp_f"] - noon["temp_f"]
        g2 = four["temp_f"] - two["temp_f"]
        growth_12_2.append(g1)
        growth_2_4.append(g2)
        per_day.append({
            "date": str(day),
            "noon_temp_f": round(noon["temp_f"], 1),
            "2pm_temp_f": round(two["temp_f"], 1),
            "4pm_temp_f": round(four["temp_f"], 1),
            "growth_12_to_2_f": round(g1, 2),
            "growth_2_to_4_f": round(g2, 2),
        })

    if not growth_12_2:
        raise ValueError("No days in this window had readings within tolerance of noon, 2pm, and 4pm.")

    return {
        "station": station_id.upper(),
        "n_days": len(growth_12_2),
        "avg_growth_12pm_to_2pm_f": round(sum(growth_12_2) / len(growth_12_2), 2),
        "avg_growth_2pm_to_4pm_f": round(sum(growth_2_4) / len(growth_2_4), 2),
        "per_day": per_day,
    }


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def backtest(station_id, hours_ahead=3, window_obs=8, lookback_days=5):
    """
    Pull `lookback_days` of historical observations for a station, then slide
    through them: at each point t (once we have `window_obs` prior readings),
    estimate the temp at t + hours_ahead using only data up to t, and compare
    against the actual observation closest to t + hours_ahead.

    The cross-station gradient network (get_pressure_gradient/
    get_station_network_signals - coastal, strait, interior_gap) IS
    exercised here - real historical data exists for all three roles too,
    each fetched once up front rather than re-fetched per rolling window.
    The NWS hourly forecast blend is NOT: forecastHourly only exposes the
    forecast as issued right now, so there's no way to ask what it would
    have said 3 hours before each historical point. Blending it in here
    would either silently score today's forecast against past observations
    (meaningless) or require fabricating a proxy forecast history - worse
    than just noting the gap. See the live estimate for how much that
    signal actually moves the point estimate.

    Returns a DataFrame of individual predictions plus a summary dict of
    error metrics (MAE, bias, RMSE).
    """
    lat, lon, name = get_station_location(station_id)

    end = datetime.now(PST)
    start = end - timedelta(days=lookback_days)
    full_df = get_observation_history(station_id, start=start, end=end)

    if len(full_df) < window_obs + 2:
        raise ValueError("Not enough historical observations in this window to backtest.")

    upwind_df, upwind_station = _fetch_upwind_df(start - timedelta(minutes=20), end + timedelta(minutes=20))
    use_gradient = upwind_df is not None

    # strait/interior_gap prefetch mirrors the coastal one above: fetch once
    # for the whole window rather than per rolling slice. Gated on the same
    # use_gradient (coastal availability) rather than tested independently -
    # if the cross-station network is down entirely for this station's
    # region, there's little value scoring the other two roles alone, and
    # this keeps the "prefetch once, reuse for every window" performance
    # property simple rather than adding three independent master switches.
    strait_df = interior_df = None
    strait_station = interior_station = None
    if use_gradient:
        strait_fetch, strait_station = _fetch_role_df(
            "strait", start - timedelta(minutes=20), end + timedelta(minutes=20)
        )
        interior_fetch, interior_station = _fetch_role_df(
            "interior_gap", start - timedelta(minutes=20), end + timedelta(minutes=20)
        )
        # False (not None) tells get_station_network_signals "already tried
        # for this whole run, don't retry" - see its docstring - so hundreds
        # of rolling-window calls don't each re-attempt and fail identically
        # against the same down station.
        strait_df = strait_fetch if strait_fetch is not None else False
        interior_df = interior_fetch if interior_fetch is not None else False

    records = []
    for i in range(window_obs, len(full_df)):
        train_df = full_df.iloc[i - window_obs:i].reset_index(drop=True)
        as_of_time = train_df["time"].iloc[-1]
        target_time = as_of_time + timedelta(hours=hours_ahead)

        # find the actual observation closest to target_time, within a 45 min tolerance
        future = full_df[full_df["time"] > as_of_time].copy()
        if future.empty:
            continue
        future["delta"] = (future["time"] - target_time).abs()
        closest = future.loc[future["delta"].idxmin()]
        if closest["delta"] > timedelta(minutes=45):
            continue  # no observation close enough to target_time to score against

        try:
            est = estimate_from_df(
                train_df, hours_ahead, lat, lon,
                use_nws_forecast=False,
                use_gradient=use_gradient, df_upwind=upwind_df,
                df_strait=strait_df, df_interior_gap=interior_df,
            )
        except Exception:
            continue

        error = est["estimated_temp_f"] - closest["temp_f"]
        lo, hi = est["estimated_range_f"]
        within_band = lo <= closest["temp_f"] <= hi
        records.append({
            "as_of": as_of_time,
            "target_time": target_time,
            "estimated_temp_f": est["estimated_temp_f"],
            "actual_temp_f": round(closest["temp_f"], 1),
            "error_f": round(error, 2),
            "within_band": within_band,
            "raw_trend_f_per_hr": est["raw_trend_f_per_hr"],
            "diurnal_damping": est["diurnal_damping"],
            "sky_wind_damping": est["sky_wind_damping"],
            "cloud_fraction": est["cloud_fraction"],
            "wind_mph": est["wind_mph"],
            "pressure_gradient_trend_inhg_per_hr": est["pressure_gradient_trend_inhg_per_hr"],
            "marine_push_index": est["marine_push_index"],
            "offshore_flow_index": est["offshore_flow_index"],
        })

    results_df = pd.DataFrame(records)
    if results_df.empty:
        raise ValueError("No scoreable predictions in this window (try a longer lookback).")

    summary = {
        "station": station_id.upper(),
        "n_predictions": len(results_df),
        "mae_f": round(results_df["error_f"].abs().mean(), 2),
        "bias_f": round(results_df["error_f"].mean(), 2),  # positive = estimator runs warm
        "rmse_f": round(np.sqrt((results_df["error_f"] ** 2).mean()), 2),
        "max_abs_error_f": round(results_df["error_f"].abs().max(), 2),
        # if this is far below ~0.9, the uncertainty band is too narrow (overconfident);
        # if it's near 1.0 with a huge band, it's too wide to be useful
        "pct_within_uncertainty_band": round(results_df["within_band"].mean(), 2),
        "gradient_signal_used": use_gradient,
        "gradient_upwind_station": upwind_station,
        "strait_station": strait_station,
        "interior_gap_station": interior_station,
        "mean_marine_push_index": (
            round(results_df["marine_push_index"].dropna().mean(), 2)
            if results_df["marine_push_index"].notna().any() else None
        ),
        "mean_offshore_flow_index": (
            round(results_df["offshore_flow_index"].dropna().mean(), 2)
            if results_df["offshore_flow_index"].notna().any() else None
        ),
        "nws_forecast_note": (
            "not backtested - forecastHourly has no historical issue-time data; "
            "this signal only affects the point estimate, not the uncertainty band, "
            "so it's checked live instead (see estimate_temp)"
        ),
    }

    return results_df, summary


if __name__ == "__main__":
    station = "KSEA"

    print("--- Live estimate ---")
    print(estimate_temp(station, hours_ahead=3))

    print("\n--- Backtest (last 5 days) ---")
    results, summary = backtest(station, hours_ahead=3, lookback_days=5)
    print(summary)
    print(results.tail(10))


def backtest_peak_projection(station_id="KSEA", actuals_by_date=None,
                             leads=(1, 2, 3, 4, 6, 8), floors=(PEAK_PROJECTION_DAMPING_FLOOR,),
                             lookback_days=7, obs_limit=8):
    """
    Backtests the DAILY-HIGH projection specifically, which backtest()
    above does not cover - that one scores estimate_temp at a fixed
    hours_ahead, whereas the number the dashboard and the Kalshi markets
    actually care about is estimate_daily_extremes' projected high, whose
    target time is always the peak-heat hour.

    That distinction is the whole point: because the target IS the
    inflection point, diurnal_damping_factor's closest_approach is always
    0 on these calls and the damping is pinned at DIURNAL_DAMPING_FLOOR.
    Sweeping `floors` here is what fits that constant against real
    outcomes instead of guessing it.

    For each day with a known actual and each lead time, reconstructs the
    observation window as it stood that many hours before the peak and
    re-runs the same projection estimate_daily_extremes would have made,
    including the `max(observed_so_far, projection)` floor.

    use_nws_forecast is off, exactly as in backtest() and for the same
    reason: forecastHourly only exposes the forecast as issued now, so
    there is no historical forecast to blend. That means these numbers
    isolate the trend/damping component - live estimates blend toward
    NWS and will land closer to it than this shows.

    actuals_by_date: {date -> actual_high_f}. Pass CLI values
    (nws_climate.fetch_recent_cli_finals) rather than stream maxima when
    the question is settlement accuracy.

    Returns {floor: {lead: {"n","bias","mae","mean_predicted"}}}.
    """
    lat, lon, _ = get_station_location(station_id)
    end = datetime.now(PST)
    start = end - timedelta(days=lookback_days)
    full_df = get_observation_history(station_id, start=start, end=end)
    if full_df.empty:
        raise ValueError("no observations in the backtest window")

    pad = timedelta(minutes=20)
    upwind_df, _ = _fetch_upwind_df(start - pad, end + pad)
    use_gradient = upwind_df is not None
    strait_df = interior_df = None
    if use_gradient:
        strait_df, _ = _fetch_role_df("strait", start - pad, end + pad)
        interior_df, _ = _fetch_role_df("interior_gap", start - pad, end + pad)

    results = {}
    for floor in floors:
        per_lead = {}
        for lead in leads:
            errs, preds = [], []
            for day, actual in sorted((actuals_by_date or {}).items()):
                sunrise_h, sunset_h = get_sun_times(lat, lon, day)
                peak_h = sunrise_h + (sunset_h - sunrise_h) * PEAK_HEAT_FRACTION
                peak_time = _hour_to_datetime(day, peak_h, full_df["time"].iloc[0].tzinfo)
                as_of = peak_time - timedelta(hours=lead)

                window = full_df[full_df["time"] <= as_of].tail(obs_limit)
                if len(window) < 3:
                    continue
                midnight = datetime.combine(day, time(0, 0), tzinfo=as_of.tzinfo)
                so_far = full_df[(full_df["time"] >= midnight) & (full_df["time"] <= as_of)]
                observed_high = float(so_far["temp_f"].max()) if not so_far.empty else -math.inf

                horizon = min(lead, TREND_HORIZON_HOURS)
                est = estimate_from_df(
                    window, horizon, lat, lon,
                    use_nws_forecast=False, use_gradient=use_gradient,
                    df_upwind=upwind_df, df_strait=strait_df, df_interior_gap=interior_df,
                    damping_floor=floor,
                )
                predicted = max(observed_high, est["estimated_temp_f"])
                errs.append(predicted - actual)
                preds.append(predicted)
            per_lead[lead] = {
                "n": len(errs),
                "bias": round(sum(errs) / len(errs), 2) if errs else None,
                "mae": round(sum(abs(e) for e in errs) / len(errs), 2) if errs else None,
                "mean_predicted": round(sum(preds) / len(preds), 2) if preds else None,
            }
        results[floor] = per_lead
    return results
