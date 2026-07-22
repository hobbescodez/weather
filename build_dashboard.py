"""
Renders weather_estimator's live KSEA estimate into a static HTML dashboard.

Run standalone to regenerate dashboard.html in this directory:
    python3 build_dashboard.py
"""

import json
from datetime import date, timedelta

from weather_estimator import (
    estimate_temp,
    estimate_daily_extremes,
    get_station_location,
    get_observation_history,
    get_sun_times,
)
from kalshi import HIGH_SERIES, LOW_SERIES, get_market_for_date, get_event_hourly_volume
from calibration_log import record_snapshot

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


def build_volume_bars_svg(hourly, tzinfo, width=640, height=190):
    """Bar chart of a Kalshi event's contracts traded per hour, in the same
    visual language as the temperature sparkline. Every bar gets its own
    tick + hour label and a printed dollar estimate - this renders as a
    static image in most places it's viewed, so a hover-only <title>
    tooltip alone isn't a reliable way to read the numbers."""
    if not hourly:
        return '<div class="hint">No trades in this window yet.</div>'

    pad_x, pad_top, pad_bottom = 4, 34, 42
    # Contracts, not dollars - this is the same unit Kalshi's own "Volume"
    # figure uses, so the chart matches what you'd see on their site.
    values = [h["contracts"] for h in hourly]
    max_val = max(max(values), 1.0)
    n = len(hourly)
    gap = 3
    bar_w = max((width - 2 * pad_x - gap * (n - 1)) / n, 1)
    baseline_y = height - pad_bottom

    parts = []
    for i, h in enumerate(hourly):
        bar_h = (baseline_y - pad_top) * (h["contracts"] / max_val)
        x = pad_x + i * (bar_w + gap)
        y = baseline_y - bar_h
        cx = x + bar_w / 2
        hour_label = h["hour_end"].astimezone(tzinfo).strftime("%-I%p").lower()
        dollar_label = f"${h['dollars']/1000:.1f}k" if h["dollars"] >= 1000 else f"${h['dollars']:,.0f}"

        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{max(bar_h, 1):.1f}" '
            f'class="volume-bar"><title>{hour_label}: {h["contracts"]:,.0f} contracts (≈${h["dollars"]:,.0f})</title></rect>'
            f'<line x1="{cx:.1f}" y1="{baseline_y:.1f}" x2="{cx:.1f}" y2="{baseline_y + 4:.1f}" class="volume-tick" />'
            f'<text x="{cx:.1f}" y="{baseline_y + 7:.1f}" class="volume-hour-label" '
            f'transform="rotate(-60 {cx:.1f} {baseline_y + 7:.1f})">{hour_label}</text>'
            f'<text x="{cx:.1f}" y="{pad_top - 5:.1f}" class="volume-dollar-label" '
            f'transform="rotate(-60 {cx:.1f} {pad_top - 5:.1f})">{dollar_label}</text>'
        )

    return f"""
<svg viewBox="0 0 {width} {height}" class="volume-chart" preserveAspectRatio="none" role="img" aria-label="Contracts and estimated dollars traded per hour">
  {"".join(parts)}
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


def build_cloud_icon_svg(cloud_pct):
    """A simple cloud outline that fills bottom-up by cloud_pct, like a
    gauge - so cloud cover reads as a shape at a glance, not just a lone
    percentage number."""
    pct = cloud_pct if cloud_pct is not None else 0
    frac = max(0.0, min(1.0, pct / 100))
    fill_h = 24 * frac
    fill_y = 24 - fill_h
    cloud_path = "M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"
    return f"""<svg viewBox="0 0 24 24" width="30" height="30" class="cloud-icon" role="img" aria-label="{pct:.0f}% cloud cover">
  <path d="{cloud_path}" class="cloud-icon-outline" />
  <clipPath id="cloud-fill-clip"><rect x="0" y="{fill_y:.2f}" width="24" height="{fill_h:.2f}" /></clipPath>
  <path d="{cloud_path}" class="cloud-icon-fill" clip-path="url(#cloud-fill-clip)" />
</svg>""".strip()


def _bracket_contains(bracket, value):
    # Kalshi settles on the officially reported whole-degree temperature,
    # while our own estimate is a continuous decimal (e.g. 91.40) - rounding
    # first avoids it falling in the crack between adjacent integer
    # brackets like "90 to 91" and "92 to 93", where neither would match.
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


THIN_VOLUME_THRESHOLD = 5  # contracts traded - below this, last_price is easy to be stale/unreliable


def build_kalshi_rows(brackets, our_estimate, estimate_label):
    """
    HTML rows for one Kalshi bracket market, highlighting whichever bracket
    our own point estimate currently falls into - a quick visual check of
    whether the model and the market agree, without computing a full
    probability distribution (that's a deliberate next step, not this one).

    Also prints our_estimate itself right above the brackets, so the number
    driving the highlight is legible next to the market's own pricing - not
    just implied by which row lit up.

    last_price is the most recent trade, not "percent of people betting" -
    it's the market's implied probability (yes/no contracts settle at $1/$0,
    so price ~= probability under normal arbitrage). Bid/ask and volume are
    shown alongside it since a lightly-traded bracket's last_price can be
    stale; low-volume rows are dimmed as a caution, not hidden.
    """
    if not brackets:
        return '<div class="hint">Market unavailable.</div>'

    rows = [f'<div class="kalshi-estimate">{estimate_label}: <strong>{our_estimate:.2f}°F</strong></div>']
    for b in brackets:
        pct = round(b["last_price"] * 100) if b["last_price"] is not None else None
        pct_label = f"{pct}%" if pct is not None else "—"

        bid = round(b["yes_bid"] * 100) if b["yes_bid"] is not None else None
        ask = round(b["yes_ask"] * 100) if b["yes_ask"] is not None else None
        spread_label = f"bid {bid}¢ / ask {ask}¢" if bid is not None and ask is not None else "no quote"

        is_thin = b["volume"] is None or b["volume"] < THIN_VOLUME_THRESHOLD
        thin_class = " kalshi-row-thin" if is_thin else ""
        thin_flag = ' <span class="kalshi-thin-flag">thin</span>' if is_thin else ""

        is_match = _bracket_contains(b, our_estimate)
        match_class = " kalshi-row-match" if is_match else ""
        rows.append(
            f'<div class="kalshi-row{match_class}{thin_class}">'
            f'<span class="kalshi-label">{b["label"]}{thin_flag}</span>'
            f'<span class="kalshi-meta">'
            f'<span class="kalshi-pct">{pct_label}</span>'
            f'<span class="kalshi-spread">{spread_label}</span>'
            f"</span>"
            f"</div>"
        )
    return "\n".join(rows)


HIGH_CAPTIONS = {
    "observed": "today's high so far",
    "projected": "projected for today's peak-heat hour",
}
TOMORROW_HINTS = {
    "nws_forecast": "From the NWS hourly forecast (HRRR model) - real atmospheric dynamics, not this tool's own trend/persistence guess. The reference confidence % is a placeholder, not a backtested figure - calibration_log.py is tracking real next-day accuracy and this will switch to a measured number once enough days accumulate.",
    "persistence_fallback": "NWS forecast unavailable - falling back to a persistence guess (today's/yesterday's high, nudged by pressure trend). The reference confidence % is a placeholder reflecting that this fallback is weaker than the forecast path, not a backtested figure - a single station's short trend can't see a full day ahead.",
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

    record_snapshot(extremes)

    now = est["as_of"]

    # Kalshi's KSEA markets are dated by Seattle's own calendar day, not
    # the system clock's - this container runs on UTC, which is already
    # into the next day while it's still evening in Seattle (UTC-7/8).
    # date.today() here would silently fetch tomorrow's just-opened, still
    # nearly-empty event as if it were "today's" actively-trading one.
    today = now.date()
    tomorrow = today + timedelta(days=1)
    try:
        kalshi_high = get_market_for_date(HIGH_SERIES, today)
    except Exception as e:
        print(f"Kalshi high market fetch failed: {e}")
        kalshi_high = None
    try:
        kalshi_low = get_market_for_date(LOW_SERIES, today)
    except Exception as e:
        print(f"Kalshi low market fetch failed: {e}")
        kalshi_low = None

    try:
        high_volume = get_event_hourly_volume(HIGH_SERIES, kalshi_high["brackets"], hours=24) if kalshi_high else None
    except Exception as e:
        print(f"Kalshi high volume fetch failed: {e}")
        high_volume = None
    try:
        low_volume = get_event_hourly_volume(LOW_SERIES, kalshi_low["brackets"], hours=24) if kalshi_low else None
    except Exception as e:
        print(f"Kalshi low volume fetch failed: {e}")
        low_volume = None
    try:
        kalshi_tomorrow_high = get_market_for_date(HIGH_SERIES, tomorrow)
    except Exception as e:
        print(f"Kalshi tomorrow high market fetch failed: {e}")
        kalshi_tomorrow_high = None
    try:
        kalshi_tomorrow_low = get_market_for_date(LOW_SERIES, tomorrow)
    except Exception as e:
        print(f"Kalshi tomorrow low market fetch failed: {e}")
        kalshi_tomorrow_low = None

    try:
        tomorrow_high_volume = get_event_hourly_volume(HIGH_SERIES, kalshi_tomorrow_high["brackets"], hours=24) if kalshi_tomorrow_high else None
    except Exception as e:
        print(f"Kalshi tomorrow high volume fetch failed: {e}")
        tomorrow_high_volume = None
    try:
        tomorrow_low_volume = get_event_hourly_volume(LOW_SERIES, kalshi_tomorrow_low["brackets"], hours=24) if kalshi_tomorrow_low else None
    except Exception as e:
        print(f"Kalshi tomorrow low volume fetch failed: {e}")
        tomorrow_low_volume = None

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
    cloud_icon_svg = build_cloud_icon_svg(cloud_pct * 100 if cloud_pct is not None else None)

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
        "cloud_icon_svg": cloud_icon_svg,
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
        "tomorrow_meta": (
            f"~{_fmt_time(extremes['tomorrow_high_time'])} · reference confidence {extremes['tomorrow_high_confidence_pct']}%*"
            if extremes["tomorrow_high_time"] is not None
            else f"reference confidence {extremes['tomorrow_high_confidence_pct']}%*"
        ),
        "tomorrow_hint": TOMORROW_HINTS[extremes["tomorrow_high_source"]],
        "tomorrow_low": f"{extremes['tomorrow_low_f']:.2f}",
        "tomorrow_low_confidence_pct": extremes["tomorrow_low_confidence_pct"],
        "tomorrow_low_meta": (
            f"~{_fmt_time(extremes['tomorrow_low_time'])} · reference confidence {extremes['tomorrow_low_confidence_pct']}%*"
            if extremes["tomorrow_low_time"] is not None
            else f"reference confidence {extremes['tomorrow_low_confidence_pct']}%*"
        ),
        "tomorrow_low_hint": TOMORROW_HINTS[extremes["tomorrow_low_source"]],
        "yesterday_high": f"{extremes['yesterday_high_f']:.2f}" if extremes["yesterday_high_f"] is not None else "—",
        "yesterday_high_time": _fmt_time(extremes["yesterday_high_time"]) if extremes["yesterday_high_time"] is not None else "—",
        "yesterday_low": f"{extremes['yesterday_low_f']:.2f}" if extremes["yesterday_low_f"] is not None else "—",
        "yesterday_low_time": _fmt_time(extremes["yesterday_low_time"]) if extremes["yesterday_low_time"] is not None else "—",
        "kalshi_high_ticker": kalshi_high["event_ticker"] if kalshi_high else "no open market",
        "kalshi_high_rows": build_kalshi_rows(
            kalshi_high["brackets"], extremes["estimated_high_f"],
            "Observed high" if extremes["high_status"] == "observed" else "Estimated high",
        ) if kalshi_high else '<div class="hint">Market unavailable.</div>',
        "kalshi_low_ticker": kalshi_low["event_ticker"] if kalshi_low else "no open market",
        # Today's Kalshi low market settles on TODAY's calendar-day low. Once
        # that's already happened (low_status == "tonight"), estimated_low_f
        # has moved on to forecasting the *next* night's low instead (a
        # different, tomorrow-dated quantity) - so the observed value is the
        # correct one to compare against today's market, not the forecast.
        "kalshi_low_rows": build_kalshi_rows(
            kalshi_low["brackets"],
            extremes["observed_low_so_far_f"] if extremes["low_status"] == "tonight" else extremes["estimated_low_f"],
            "Observed low" if extremes["low_status"] == "tonight" else "Estimated low",
        ) if kalshi_low else '<div class="hint">Market unavailable.</div>',
        "kalshi_tomorrow_high_ticker": kalshi_tomorrow_high["event_ticker"] if kalshi_tomorrow_high else "no open market",
        "kalshi_tomorrow_high_rows": build_kalshi_rows(kalshi_tomorrow_high["brackets"], extremes["tomorrow_high_f"], "Estimated high") if kalshi_tomorrow_high else '<div class="hint">Market not open yet.</div>',
        "kalshi_tomorrow_low_ticker": kalshi_tomorrow_low["event_ticker"] if kalshi_tomorrow_low else "no open market",
        "kalshi_tomorrow_low_rows": build_kalshi_rows(kalshi_tomorrow_low["brackets"], extremes["tomorrow_low_f"], "Estimated low") if kalshi_tomorrow_low else '<div class="hint">Market not open yet.</div>',
        "high_volume_total": f"{high_volume['total_contracts']:,.0f}" if high_volume else "—",
        "high_volume_dollars_est": f"≈${high_volume['total_dollars']:,.0f} est." if high_volume else "—",
        "high_volume_svg": build_volume_bars_svg(high_volume["hourly"], now.tzinfo) if high_volume else '<div class="hint">Volume unavailable.</div>',
        "low_volume_total": f"{low_volume['total_contracts']:,.0f}" if low_volume else "—",
        "low_volume_dollars_est": f"≈${low_volume['total_dollars']:,.0f} est." if low_volume else "—",
        "low_volume_svg": build_volume_bars_svg(low_volume["hourly"], now.tzinfo) if low_volume else '<div class="hint">Volume unavailable.</div>',
        "tomorrow_high_volume_total": f"{tomorrow_high_volume['total_contracts']:,.0f}" if tomorrow_high_volume else "—",
        "tomorrow_high_volume_dollars_est": f"≈${tomorrow_high_volume['total_dollars']:,.0f} est." if tomorrow_high_volume else "—",
        "tomorrow_high_volume_svg": build_volume_bars_svg(tomorrow_high_volume["hourly"], now.tzinfo) if tomorrow_high_volume else '<div class="hint">Volume unavailable.</div>',
        "tomorrow_low_volume_total": f"{tomorrow_low_volume['total_contracts']:,.0f}" if tomorrow_low_volume else "—",
        "tomorrow_low_volume_dollars_est": f"≈${tomorrow_low_volume['total_dollars']:,.0f} est." if tomorrow_low_volume else "—",
        "tomorrow_low_volume_svg": build_volume_bars_svg(tomorrow_low_volume["hourly"], now.tzinfo) if tomorrow_low_volume else '<div class="hint">Volume unavailable.</div>',
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
