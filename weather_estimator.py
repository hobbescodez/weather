"""
Weather temperature estimator.

Pulls recent observations from api.weather.gov, fits a trend, damps that
trend using actual sunrise/sunset for the station (via astral), and
estimates temperature N hours ahead. Includes a backtest harness that
checks past estimates against what the station actually recorded.

pip install requests pandas numpy astral
"""

import warnings

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
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

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid temperature readings for the requested window.")
    return df


# ---------------------------------------------------------------------------
# Damping based on real sunrise/sunset
# ---------------------------------------------------------------------------

def diurnal_damping_factor(current_time, hours_ahead, lat, lon):
    """
    Multiplier applied to the raw trend. Damps hardest when the window
    crosses near actual sunrise (bottom of curve) or a few hours after
    sunrise-to-sunset midpoint (rough proxy for peak heating), using real
    sun times for that date instead of hardcoded hours.
    """
    sunrise_h, sunset_h = get_sun_times(lat, lon, current_time.date())
    # rough peak-heat proxy: ~2/3 of the way between sunrise and sunset
    peak_h = sunrise_h + (sunset_h - sunrise_h) * 0.65

    hour = current_time.hour + current_time.minute / 60
    target_hour = (hour + hours_ahead) % 24

    def circular_dist(a, b):
        diff = abs(a - b) % 24
        return min(diff, 24 - diff)

    inflection_points = [sunrise_h, peak_h]
    dist_now = min(circular_dist(hour, p) for p in inflection_points)
    dist_target = min(circular_dist(target_hour, p) for p in inflection_points)

    closest_approach = min(dist_now, dist_target)
    damping = min(1.0, 0.35 + closest_approach / 6)
    return damping


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


def _pressure_trend_and_uncertainty(df, elapsed_hours, hours_ahead):
    """
    Falling pressure signals a front/unsettled system may be approaching,
    but not which direction temp will move - so this widens the uncertainty
    band around the point estimate rather than shifting it.
    Returns (pressure_trend_inhg_per_hr, uncertainty_f).
    """
    base_uncertainty = 1.0 + 0.3 * hours_ahead  # baseline grows with horizon

    if df["pressure_inhg"].notna().sum() < 2:
        return None, round(base_uncertainty, 1)

    valid = df["pressure_inhg"].notna()
    p_slope = np.polyfit(elapsed_hours[valid], df["pressure_inhg"][valid], 1)[0]

    # a drop of ~0.03 inHg/hr or faster is a reasonably brisk pressure fall
    if p_slope < 0:
        uncertainty = base_uncertainty + min(2.0, abs(p_slope) * 40)
    else:
        uncertainty = base_uncertainty

    return round(p_slope, 4), round(uncertainty, 1)


def estimate_from_df(df, hours_ahead, lat, lon):
    """Core estimation logic, given a dataframe of observations. Reused by both
    the live estimator and the backtest."""
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

    diurnal_damping = diurnal_damping_factor(now, hours_ahead, lat, lon)
    sky_wind_damping = _cloud_wind_damping(df)
    combined_damping = diurnal_damping * sky_wind_damping

    pressure_trend, uncertainty_f = _pressure_trend_and_uncertainty(df, elapsed_hours, hours_ahead)

    raw_change = slope * hours_ahead
    damped_change = raw_change * combined_damping + spread_adjustment * np.sign(raw_change) * -1

    estimated_temp = latest["temp_f"] + damped_change
    target_time = now + timedelta(hours=hours_ahead)

    return {
        "as_of": now,
        "target_time": target_time,
        "current_temp_f": round(latest["temp_f"], 1),
        "estimated_temp_f": round(estimated_temp, 1),
        "estimated_range_f": (
            round(estimated_temp - uncertainty_f, 1),
            round(estimated_temp + uncertainty_f, 1),
        ),
        "raw_trend_f_per_hr": round(slope, 2),
        "diurnal_damping": round(diurnal_damping, 2),
        "sky_wind_damping": round(sky_wind_damping, 2),
        "cloud_fraction": latest.get("cloud_fraction"),
        "wind_mph": round(latest["wind_mph"], 1) if pd.notna(latest.get("wind_mph")) else None,
        "pressure_trend_inhg_per_hr": pressure_trend,
        "n_observations": len(df),
    }


def estimate_temp(station_id, hours_ahead=3, obs_limit=8):
    lat, lon, name = get_station_location(station_id)
    df = get_observation_history(station_id, limit=obs_limit)
    result = estimate_from_df(df, hours_ahead, lat, lon)
    result["station"] = station_id.upper()
    result["station_name"] = name
    return result


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def backtest(station_id, hours_ahead=3, window_obs=8, lookback_days=5):
    """
    Pull `lookback_days` of historical observations for a station, then slide
    through them: at each point t (once we have `window_obs` prior readings),
    estimate the temp at t + hours_ahead using only data up to t, and compare
    against the actual observation closest to t + hours_ahead.

    Returns a DataFrame of individual predictions plus a summary dict of
    error metrics (MAE, bias, RMSE).
    """
    lat, lon, name = get_station_location(station_id)

    end = datetime.now(PST)
    start = end - timedelta(days=lookback_days)
    full_df = get_observation_history(station_id, start=start, end=end)

    if len(full_df) < window_obs + 2:
        raise ValueError("Not enough historical observations in this window to backtest.")

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
            est = estimate_from_df(train_df, hours_ahead, lat, lon)
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
