"""
Renders weather_estimator's live KSEA estimate into a static HTML dashboard.

Run standalone to regenerate dashboard.html in this directory:
    python3 build_dashboard.py
"""

import json
from datetime import timedelta

from weather_estimator import (
    estimate_temp,
    estimate_daily_extremes,
    get_station_location,
    get_observation_history,
    get_sun_times,
)

STATION = "KSEA"
HOURS_AHEAD = 3
SPARKLINE_HOURS = 6


def _fmt_time(ts):
    return ts.strftime("%-I:%M %p").lower()


def _fmt_day_time(ts):
    return ts.strftime("%a %-I:%M %p").lower()


def build_sparkline_svg(times, temps, est_time, est_temp, width=640, height=160):
    pad_x, pad_top, pad_bottom = 8, 16, 28
    all_temps = temps + [est_temp]
    lo, hi = min(all_temps), max(all_temps)
    span = max(hi - lo, 1)
    lo -= span * 0.15
    hi += span * 0.15
    span = hi - lo

    t0 = times[0]
    t_end = est_time
    total_seconds = (t_end - t0).total_seconds()

    def xy(t, temp):
        x = pad_x + (width - 2 * pad_x) * ((t - t0).total_seconds() / total_seconds)
        y = pad_top + (height - pad_top - pad_bottom) * (1 - (temp - lo) / span)
        return x, y

    pts = [xy(t, v) for t, v in zip(times, temps)]
    est_pt = xy(est_time, est_temp)

    line_path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area_path = (
        line_path
        + f" L {pts[-1][0]:.1f},{height - pad_bottom} L {pts[0][0]:.1f},{height - pad_bottom} Z"
    )
    proj_path = f"M {pts[-1][0]:.1f},{pts[-1][1]:.1f} L {est_pt[0]:.1f},{est_pt[1]:.1f}"

    now_x = pts[-1][0]

    return f"""
<svg viewBox="0 0 {width} {height}" class="sparkline" preserveAspectRatio="none" role="img" aria-label="Temperature trend, last {SPARKLINE_HOURS} hours and projected estimate">
  <path d="{area_path}" class="spark-area" />
  <path d="{line_path}" class="spark-line" />
  <path d="{proj_path}" class="spark-proj" />
  <circle cx="{now_x:.1f}" cy="{pts[-1][1]:.1f}" r="3.5" class="spark-now-dot" />
  <circle cx="{est_pt[0]:.1f}" cy="{est_pt[1]:.1f}" r="4.5" class="spark-est-dot" />
</svg>
""".strip()


def pressure_chip(trend):
    if trend is None:
        return ("no data", "chip-neutral")
    if trend < -0.015:
        return ("falling", "chip-warn")
    if trend > 0.015:
        return ("rising", "chip-good")
    return ("steady", "chip-neutral")


HIGH_CAPTIONS = {
    "observed": "today's high so far",
    "projected": "projected for today's peak-heat hour",
}
LOW_CAPTIONS = {
    "today": "today's overnight low, almost here",
    "tonight": "expected low tonight",
}


def sky_condition(cloud_fraction, is_day):
    """(condition label, sky class) from cloud cover and day/night, for the
    background gradient and condition text - both drawn from the same
    observation the rest of the page already uses."""
    c = cloud_fraction if cloud_fraction is not None else 0.0
    if c <= 0.15:
        return ("Sunny" if is_day else "Clear", "day-clear" if is_day else "night-clear")
    if c <= 0.5:
        return ("Mostly Sunny" if is_day else "Mostly Clear", "day-clear" if is_day else "night-clear")
    if c <= 0.85:
        return ("Partly Cloudy", "day-cloudy" if is_day else "night-cloudy")
    return ("Cloudy", "day-cloudy" if is_day else "night-cloudy")


def main():
    est = estimate_temp(STATION, hours_ahead=HOURS_AHEAD)
    extremes = estimate_daily_extremes(STATION)
    lat, lon, _ = get_station_location(STATION)

    now = est["as_of"]
    window_start = now - timedelta(hours=SPARKLINE_HOURS)
    hist = get_observation_history(STATION, start=window_start, end=now)

    times = list(hist["time"])
    temps = list(hist["temp_f"])

    svg = build_sparkline_svg(times, temps, est["target_time"], est["estimated_temp_f"])

    lo, hi = est["estimated_range_f"]
    band_width_f = hi - lo

    p_label, p_class = pressure_chip(est["pressure_trend_inhg_per_hr"])

    cloud_pct = est["cloud_fraction"]
    cloud_label = f"{round(cloud_pct * 100)}%" if cloud_pct is not None else "—"

    confidence_pct = round(est["diurnal_damping"] * est["sky_wind_damping"] * 100)

    sunrise_h, sunset_h = get_sun_times(lat, lon, now.date())
    now_h = now.hour + now.minute / 60
    is_day = sunrise_h <= now_h < sunset_h
    condition_text, sky_class = sky_condition(cloud_pct, is_day)

    obs_json_url = f"https://api.weather.gov/stations/{STATION}/observations"
    obhistory_url = f"https://forecast.weather.gov/data/obhistory/{STATION}.html"
    forecast_url = f"https://forecast.weather.gov/MapClick.php?lat={lat:.4f}&lon={lon:.4f}"
    timeseries_url = f"https://www.weather.gov/wrh/timeseries?site={STATION}"

    ctx = {
        "station_name": est["station_name"],
        "station_id": est["station"],
        "as_of_time": _fmt_time(now),
        "as_of_date": now.strftime("%A, %B %-d"),
        "current_temp": f"{temps[-1]:.2f}",  # raw observed value, not rounded like est['current_temp_f']
        "target_time": _fmt_day_time(est["target_time"]),
        "estimated_temp": f"{est['estimated_temp_f']:.2f}",
        "range_low": f"{lo:.2f}",
        "range_high": f"{hi:.2f}",
        "band_width": f"{band_width_f:.2f}",
        "hours_ahead": HOURS_AHEAD,
        "trend_per_hr": f"{est['raw_trend_f_per_hr']:+.2f}",
        "wind_mph": f"{est['wind_mph']:.2f}" if est["wind_mph"] is not None else "—",
        "cloud_label": cloud_label,
        "pressure_label": p_label,
        "pressure_class": p_class,
        "confidence_pct": confidence_pct,
        "n_observations": est["n_observations"],
        "sparkline_svg": svg,
        "sparkline_hours": SPARKLINE_HOURS,
        "data_json": json.dumps(est, default=str, indent=2),
        "daily_high": f"{extremes['estimated_high_f']:.2f}",
        "daily_high_caption": HIGH_CAPTIONS[extremes["high_status"]],
        "est_peak_time": _fmt_time(extremes["estimated_high_time"]),
        "daily_low": f"{extremes['estimated_low_f']:.2f}",
        "daily_low_caption": LOW_CAPTIONS[extremes["low_status"]],
        "est_trough_time": _fmt_day_time(extremes["estimated_low_time"]),
        "observed_high": f"{extremes['observed_high_so_far_f']:.2f}",
        "observed_high_time": _fmt_time(extremes["observed_high_so_far_time"]),
        "observed_low": f"{extremes['observed_low_so_far_f']:.2f}",
        "observed_low_time": _fmt_time(extremes["observed_low_so_far_time"]),
        "tomorrow_high": f"{extremes['tomorrow_high_f']:.2f}",
        "tomorrow_confidence_pct": extremes["tomorrow_high_confidence_pct"],
        "yesterday_high": f"{extremes['yesterday_high_f']:.2f}" if extremes["yesterday_high_f"] is not None else "—",
        "yesterday_high_time": _fmt_time(extremes["yesterday_high_time"]) if extremes["yesterday_high_time"] is not None else "—",
        "yesterday_low": f"{extremes['yesterday_low_f']:.2f}" if extremes["yesterday_low_f"] is not None else "—",
        "yesterday_low_time": _fmt_time(extremes["yesterday_low_time"]) if extremes["yesterday_low_time"] is not None else "—",
        "sky_class": sky_class,
        "condition_text": condition_text,
        "obs_json_url": obs_json_url,
        "obhistory_url": obhistory_url,
        "forecast_url": forecast_url,
        "timeseries_url": timeseries_url,
    }

    with open("dashboard_template.html", "r") as f:
        template = f.read()

    for key, value in ctx.items():
        template = template.replace("{{" + key + "}}", str(value))

    out_path = "/tmp/claude-0/-home-user-weather/805f9339-6e9f-5b7a-a9f0-f3408e81362c/scratchpad/ksea_dashboard.html"
    with open(out_path, "w") as f:
        f.write(template)

    print(f"Wrote {out_path}")
    print(json.dumps({k: v for k, v in ctx.items() if k not in ("sparkline_svg", "data_json")}, indent=2))


if __name__ == "__main__":
    main()
