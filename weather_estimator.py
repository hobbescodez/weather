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

def _hour_to_datetime(base_date, hour_decimal, tzinfo):
    """Convert a decimal hour (e.g. 15.65) on a given date into an aware datetime."""
    h = int(hour_decimal) % 24
    m = int(round((hour_decimal - int(hour_decimal)) * 60))
    if m == 60:
        m = 0
        h = (h + 1) % 24
    return datetime.combine(base_date, time(h, m), tzinfo=tzinfo)


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


# A short local trend (fit over the last handful of observations) is only a
# credible predictor a few hours out. Projections further ahead than this
# still use the real hours_ahead for figuring out *where* in the diurnal
# cycle the target time falls, but the trend's contribution to the magnitude
# of the change is capped at this many hours - otherwise a small slope
# measured over the last 20 minutes gets multiplied out to an absurd swing
# over a 12+ hour projection (e.g. projecting to the next sunrise).
TREND_HORIZON_HOURS = 6


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

    raw_change = slope * min(hours_ahead, TREND_HORIZON_HOURS)
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
      into a "low" warmer than the current temperature. Instead this uses a
      standard radiative-cooling heuristic: on a clear, calm night the
      overnight minimum tends toward the dewpoint (further cooling slows as
      air nears saturation); clouds/wind suppress that drop. Reuses the same
      cloud/wind damping factor as the short-term model, just aimed at a
      different physical effect.
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

    sunrise_today, sunset_today = get_sun_times(lat, lon, today)
    peak_today = sunrise_today + (sunset_today - sunrise_today) * 0.65
    hour = now.hour + now.minute / 60

    high_time = _hour_to_datetime(today, peak_today, now.tzinfo)

    if hour < peak_today:
        horizon = min(peak_today - hour, TREND_HORIZON_HOURS)
        peak_est = estimate_from_df(df, horizon, lat, lon)
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
    else:
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
        low_status = "tonight"  # today's low already happened; forecasting the next one
        tomorrow = today + timedelta(days=1)
        sunrise_tomorrow, _ = get_sun_times(lat, lon, tomorrow)
        low_time = _hour_to_datetime(tomorrow, sunrise_tomorrow, now.tzinfo)

    # Tomorrow's high: there's no real forecast model behind this - just a
    # persistence guess (assume tomorrow's peak looks like today's) nudged by
    # the current pressure trend, which is the only signal this station-only
    # tool has about a system change coming. Confidence is capped low and
    # explicitly separate from today's sun-grounded numbers above, since a
    # short local trend genuinely can't see a day ahead.
    t0 = df["time"].iloc[0]
    elapsed_hours_all = (df["time"] - t0).dt.total_seconds() / 3600
    pressure_trend, _ = _pressure_trend_and_uncertainty(df, elapsed_hours_all, 24)
    if pressure_trend is None:
        pressure_adj, tomorrow_confidence = 0.0, 45
    elif pressure_trend < -0.015:
        pressure_adj, tomorrow_confidence = -2.0, 35  # falling pressure: system/front likely changing things
    elif pressure_trend > 0.015:
        pressure_adj, tomorrow_confidence = 1.0, 55  # rising pressure: current pattern more likely to hold
    else:
        pressure_adj, tomorrow_confidence = 0.0, 50
    tomorrow_high = estimated_high + pressure_adj

    return {
        "as_of": now,
        "station": station_id.upper(),
        "station_name": name,
        "estimated_high_f": round(estimated_high, 1),
        "estimated_high_time": high_time,  # theoretical peak-heat hour for today, from sun position - not tied to when the observed high actually occurred
        "estimated_low_time": low_time,  # theoretical sunrise (today's or tomorrow's) - not tied to when the observed low actually occurred
        "high_status": high_status,
        "estimated_low_f": round(estimated_low, 1),
        "low_status": low_status,
        "observed_high_so_far_f": round(observed_high, 2),
        "observed_high_so_far_time": observed_high_time,  # when that actual high was recorded
        "observed_low_so_far_f": round(observed_low, 2),
        "observed_low_so_far_time": observed_low_time,  # when that actual low was recorded
        "tomorrow_high_f": round(tomorrow_high, 1),
        "tomorrow_high_confidence_pct": tomorrow_confidence,
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
