"""
Renders weather_estimator's live KSEA estimate into a static HTML dashboard.

Output path: defaults to docs/index.html (relative to this file's
directory) so a plain `python3 build_dashboard.py` produces something
GitHub Pages can serve directly from this repo's docs/ folder - no
separate publish step required. Override with the DASHBOARD_OUTPUT_PATH
env var for any other destination (e.g. the Claude Code session's own
scratch directory, when publishing to the claude.ai Artifact instead).

Run standalone to regenerate the dashboard:
    python3 build_dashboard.py
    DASHBOARD_OUTPUT_PATH=/some/other/path.html python3 build_dashboard.py
"""

import json
import os
from datetime import date, datetime, timedelta
from html.parser import HTMLParser

import pandas as pd

from weather_estimator import (
    estimate_temp,
    estimate_daily_extremes,
    get_station_location,
    get_observation_history,
    get_sun_times,
    get_hourly_forecast,
    MARINE_PUSH_INDEX_THRESHOLD,
    OFFSHORE_FLOW_INDEX_THRESHOLD,
)
from kalshi import HIGH_SERIES, LOW_SERIES, get_market_for_date, get_event_hourly_volume, bracket_contains
from observation_precision import (
    format_reading, precision_note, is_whole_celsius,
    format_headline_reading, headline_precision_note, settlement_band,
)
from calibration_log import (
    record_snapshot, next_day_confidence_pct, summarize, MIN_NEXT_DAY_SAMPLES,
    classify_conditions, condition_confidence, describe_conditions,
    MIN_CONDITION_BUCKET_SAMPLES,
)
from daily_performance import (
    finalize_pending_days,
    reconcile_stream_fallback_actuals,
    reconcile_peak_time_windows,
    weekly_table,
    monthly_rollup,
    LOW_SAMPLE_THRESHOLD,
)
from peak_alerts import get_or_lock_daily_targets
from paper_trading import get_or_lock_2hr_targets, LEAD_TIME_HINTS, MIN_EXIT_GAIN

STATION = "KSEA"
HOURS_AHEAD = 3
SPARKLINE_HOURS = 6

# Second location: Seattle proper (Fremont/Aurora area), ~15 miles from
# Sea-Tac and a genuinely different microclimate (closer to Lake Union/
# marine moderation - Sea-Tac runs warmer in summer). This is a much
# lighter-weight panel than the KSEA one on purpose - see module docstring
# note in the Fremont section below for what's deliberately NOT built here.
FREMONT_LAT, FREMONT_LON = 47.6511, -122.3547  # Fremont Bridge / Aurora Ave N area
FREMONT_OBS_STATION = "KBFI"  # Boeing Field - nearest full ASOS station with real ground-truth obs (still ~5-6mi off)


def _fmt_time(ts):
    return ts.strftime("%-I:%M %p").lower()


def _fmt_day_time(ts):
    return ts.strftime("%a %-I:%M %p").lower()


def build_sparkline_svg(times, temps, est_time, est_temp, width=640, height=160):
    # pad_top leaves room for the estimate label sitting above its dot -
    # that dot marks the projection time, which is easy to lose track of
    # against the plain trend line, so it gets a halo + outline + a printed
    # value instead of just a small solid circle (see spark-est-* CSS).
    pad_x, pad_top, pad_bottom = 8, 30, 28
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
    est_label_y = max(est_pt[1] - 14, 12)

    return f"""
<svg viewBox="0 0 {width} {height}" class="sparkline" role="img" aria-label="Temperature trend, last {SPARKLINE_HOURS} hours and projected estimate">
  <path d="{area_path}" class="spark-area" />
  <path d="{line_path}" class="spark-line" />
  <path d="{proj_path}" class="spark-proj" />
  <line x1="{est_pt[0]:.1f}" y1="{est_pt[1]:.1f}" x2="{est_pt[0]:.1f}" y2="{height - pad_bottom:.1f}" class="spark-est-guide" />
  <circle cx="{now_x:.1f}" cy="{pts[-1][1]:.1f}" r="3.5" class="spark-now-dot" />
  <circle cx="{est_pt[0]:.1f}" cy="{est_pt[1]:.1f}" r="9" class="spark-est-dot-halo" />
  <circle cx="{est_pt[0]:.1f}" cy="{est_pt[1]:.1f}" r="5.5" class="spark-est-dot" />
  <text x="{est_pt[0]:.1f}" y="{est_label_y:.1f}" text-anchor="end" class="spark-est-label">{est_temp:.0f}°</text>
</svg>
""".strip()


# Which observation columns the metric switcher offers, in tab order.
# Every one of these is already being pulled by get_observation_history
# for the trend fit and the gradient network - none of this adds a
# request. (key, tab label, unit, decimals, aria noun)
METRIC_TABS = [
    ("temp_f", "Temp", "°F", 1, "temperature"),
    ("wind_mph", "Wind", " mph", 0, "wind speed"),
    ("pressure_inhg", "Pressure", " inHg", 2, "barometric pressure"),
    ("dewpoint_f", "Dew pt", "°F", 1, "dew point"),
    ("cloud_pct", "Cloud", "%", 0, "cloud cover"),
]


def build_metric_chart_svg(times, values, unit, decimals, aria, width=640, height=150):
    """One metric's recent history, in the same visual language as the
    hero sparkline - same area+line+now-dot, same classes.

    Deliberately NOT build_sparkline_svg with arguments bolted on: that
    one's whole job is the projection (the dashed segment, the haloed
    estimate dot, its printed label, the guide line down to the axis), and
    none of those exist for wind or pressure, which have history only.
    Threading "no projection" through it would have left half its body
    behind a conditional for no gain.

    Gaps matter here in a way they don't for temperature: cloud_fraction
    in particular is absent from plenty of observations, so the series is
    drawn as separate polylines split on missing values rather than one
    path that would draw a straight line across a hole it has no data for.
    """
    pts_all = [(t, v) for t, v in zip(times, values)]
    present = [(t, v) for t, v in pts_all if v is not None and not pd.isna(v)]
    if len(present) < 2:
        return (
            '<div class="hint" style="padding:18px 0;">'
            f"No {aria} readings in this window.</div>"
        )

    pad_x, pad_top, pad_bottom = 8, 22, 26
    vals = [v for _, v in present]
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1e-6)
    lo -= span * 0.18
    hi += span * 0.18
    span = hi - lo

    t0, t1 = pts_all[0][0], pts_all[-1][0]
    total = max((t1 - t0).total_seconds(), 1)

    def xy(t, v):
        x = pad_x + (width - 2 * pad_x) * ((t - t0).total_seconds() / total)
        y = pad_top + (height - pad_top - pad_bottom) * (1 - (v - lo) / span)
        return x, y

    # Split into runs of consecutive present values.
    runs, cur = [], []
    for t, v in pts_all:
        if v is None or pd.isna(v):
            if len(cur) > 1:
                runs.append(cur)
            cur = []
        else:
            cur.append(xy(t, v))
    if len(cur) > 1:
        runs.append(cur)

    lines = "".join(
        '<path d="M ' + " L ".join(f"{x:.1f},{y:.1f}" for x, y in r) + '" class="spark-line" />'
        for r in runs
    )
    # Every run gets its area, not just the longest. Shading one run and
    # leaving the others as bare lines doesn't read as "this run is the
    # important one" - it reads as the fill having failed to render, which
    # is exactly how cloud cover (the gappiest series) looked.
    area = "".join(
        '<path d="M ' + " L ".join(f"{x:.1f},{y:.1f}" for x, y in r)
        + f' L {r[-1][0]:.1f},{height - pad_bottom} L {r[0][0]:.1f},{height - pad_bottom} Z"'
        ' class="spark-area" />'
        for r in runs
    )

    last_x, last_y = xy(*present[-1])
    last_v = present[-1][1]
    # Clear the dot's own radius plus the stroke, then keep the text inside
    # the top pad. 12 put the baseline within a few px of the line itself,
    # so on a series that ends climbing (wind, most afternoons) the label
    # sat on top of the data.
    label_y = min(max(last_y - 16, 13), height - pad_bottom - 4)
    return f"""
<svg viewBox="0 0 {width} {height}" class="sparkline" role="img" aria-label="{aria}, last {SPARKLINE_HOURS} hours">
  {area}
  {lines}
  <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4" class="spark-now-dot" />
  <text x="{last_x:.1f}" y="{label_y:.1f}" text-anchor="end" class="spark-metric-label">{last_v:.{decimals}f}{unit}</text>
</svg>
""".strip()


def build_metric_switcher_html(hist):
    """Tabbed metric charts, CSS-only.

    No JavaScript: this page is a statically generated file with none, and
    adding a script purely to toggle visibility would be the first script
    on it. Hidden radio inputs plus :checked sibling selectors do the same
    job, keep the tabs keyboard-operable and screen-reader-labelled for
    free, and survive the Artifact CSP without a thought.
    """
    df = hist.copy()
    # cloud_fraction is 0-1; everything else is already in display units.
    df["cloud_pct"] = df["cloud_fraction"] * 100
    times = list(df["time"])

    inputs, tabs, panels = [], [], []
    for i, (key, label, unit, decimals, aria) in enumerate(METRIC_TABS):
        checked = " checked" if i == 0 else ""
        inputs.append(
            f'<input type="radio" name="metric" id="metric-{key}" class="metric-radio"{checked}>'
        )
        tabs.append(f'<label for="metric-{key}" class="metric-tab">{label}</label>')
        chart = build_metric_chart_svg(
            times, list(df[key]) if key in df else [], unit, decimals, aria
        )
        panels.append(f'<div class="metric-panel">{chart}</div>')

    return (
        '<div class="metric-switcher">'
        + "".join(inputs)
        # <nav>, not <div>, and that is load-bearing: the panel selectors
        # use :nth-of-type, which counts among same-tag siblings, so a div
        # tab strip would shift every panel's index by one.
        + '<nav class="metric-tabs">' + "".join(tabs) + "</nav>"
        + "".join(panels)
        + "</div>"
    )


def build_volume_bars_svg(hourly, tzinfo, width=640, height=190):
    """Bar chart of a Kalshi event's contracts traded per hour, in the same
    visual language as the temperature sparkline. Every bar gets its own
    tick + hour label and a printed dollar estimate - this renders as a
    static image in most places it's viewed, so a hover-only <title>
    tooltip alone isn't a reliable way to read the numbers. The single
    busiest hour (peak trading volume) is picked out in the accent color
    with bold labels - otherwise it's just the tallest bar among many
    similar ones, easy to skim past."""
    if not hourly:
        return '<div class="hint">No trades in this window yet.</div>'

    pad_x, pad_top, pad_bottom = 4, 34, 42
    # Contracts, not dollars - this is the same unit Kalshi's own "Volume"
    # figure uses, so the chart matches what you'd see on their site.
    values = [h["contracts"] for h in hourly]
    max_val = max(max(values), 1.0)
    peak_idx = values.index(max(values))
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

        is_peak = i == peak_idx and h["contracts"] > 0
        bar_class = "volume-bar-peak" if is_peak else "volume-bar"
        tick_class = "volume-tick-peak" if is_peak else "volume-tick"
        hour_class = "volume-hour-label-peak" if is_peak else "volume-hour-label"
        dollar_class = "volume-dollar-label-peak" if is_peak else "volume-dollar-label"
        title_prefix = "Peak hour - " if is_peak else ""

        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{max(bar_h, 1):.1f}" '
            f'class="{bar_class}"><title>{title_prefix}{hour_label}: {h["contracts"]:,.0f} contracts (≈${h["dollars"]:,.0f})</title></rect>'
            f'<line x1="{cx:.1f}" y1="{baseline_y:.1f}" x2="{cx:.1f}" y2="{baseline_y + 4:.1f}" class="{tick_class}" />'
            f'<text x="{cx:.1f}" y="{baseline_y + 7:.1f}" class="{hour_class}" '
            f'transform="rotate(-60 {cx:.1f} {baseline_y + 7:.1f})">{hour_label}</text>'
            f'<text x="{cx:.1f}" y="{pad_top - 5:.1f}" class="{dollar_class}" '
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


def index_chip(value, threshold, rising_label):
    """Same shape as pressure_chip, for marine_push_index/offshore_flow_index
    - "rising" (above threshold, chip-warn - matches _pressure_trend_and_
    uncertainty's own threshold for widening uncertainty) means the pattern
    is elevated enough to call out; "quiet" (below -threshold) means the
    opposite pattern; otherwise "steady". None means the underlying station
    was unavailable, not that the signal read zero."""
    if value is None:
        return ("no data", "chip-neutral")
    if value > threshold:
        return (rising_label, "chip-warn")
    if value < -threshold:
        return ("quiet", "chip-good")
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


def build_hero_icon_svg(cloud_fraction, is_day):
    """The big animated weather glyph at the top of the page.

    Drawn inline rather than pulled from an icon set. Two reasons, both
    hard constraints rather than preference: the published Artifact runs
    under a CSP that blocks every external host, so a CDN-hosted set
    cannot load at all, and vendoring one would mean carrying a
    third-party licence and its files in a repo whose whole output is a
    single self-contained HTML file.

    Picks its parts from the same cloud_fraction/is_day the background
    gradient and condition text already use (see sky_condition), so the
    icon can never disagree with the words next to it. Animation is pure
    CSS on the classes below - a static generated page has no JS to drive
    anything, and transform/opacity keyframes composite on the GPU
    without triggering layout.
    """
    c = cloud_fraction if cloud_fraction is not None else 0.0
    parts = []

    if is_day:
        # Sun. Rays are a single dasharray circle rather than 8 <line>s -
        # same picture, one element to rotate.
        parts.append(
            '<g class="hero-sun">'
            '<circle cx="34" cy="34" r="12" class="hero-sun-core" />'
            '<circle cx="34" cy="34" r="20" class="hero-sun-rays" />'
            "</g>"
        )
    else:
        # Crescent via an offset mask, so it stays one shape at any size.
        parts.append(
            '<defs><mask id="hero-moon-mask">'
            '<rect width="100" height="100" fill="#fff" />'
            '<circle cx="42" cy="26" r="14" fill="#000" />'
            "</mask></defs>"
            '<g class="hero-moon"><circle cx="34" cy="34" r="15" '
            'mask="url(#hero-moon-mask)" class="hero-moon-body" /></g>'
        )

    # Cloud only appears once there is meaningfully some, and the second
    # (front) puff only for genuinely overcast skies - so the glyph tracks
    # the same three bands sky_condition labels.
    if c > 0.15:
        parts.append(
            '<g class="hero-cloud hero-cloud-back">'
            '<path d="M30 66h34a11 11 0 0 0 0-22 15 15 0 0 0-28-5 10 10 0 0 0-6 27z" />'
            "</g>"
        )
    if c > 0.5:
        parts.append(
            '<g class="hero-cloud hero-cloud-front">'
            '<path d="M22 76h40a10 10 0 0 0 0-20 13 13 0 0 0-25-4 9 9 0 0 0-15 24z" />'
            "</g>"
        )

    # Crop the viewBox to whatever was actually drawn. The parts are laid
    # out on a fixed 100x100 grid so the sun and the clouds keep their
    # relative positions, but a clear-sky glyph only occupies the top-left
    # of that grid - shipping the full square left ~40% of the element as
    # empty space below the sun, which the layout then dutifully reserved
    # and which read as the icon floating too high in the card.
    if c > 0.5:
        view_box = "6 6 74 74"      # sun/moon + both puffs
    elif c > 0.15:
        view_box = "6 6 72 72"      # sun/moon + back puff
    elif is_day:
        view_box = "8 8 52 52"      # sun alone: rays reach r=24.5 from (34,34)
    else:
        view_box = "16 16 36 36"    # moon alone: body is only r=15

    label = sky_condition(cloud_fraction, is_day)[0]
    return (
        f'<svg viewBox="{view_box}" class="hero-icon" role="img" aria-label="{label}">'
        + "".join(parts)
        + "</svg>"
    )


def _fmt_dt_short(iso_str):
    if iso_str is None:
        return "—"
    return datetime.fromisoformat(iso_str).strftime("%-I:%M%p").lower()


def _fmt_num(x, digits=1, sign=False):
    if x is None:
        return "—"
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def _format_time_delta(s):
    """Timing error, flagged when it lands inside the plateau the extreme
    actually occupied - the model can't be 'wrong' by less than the
    measurement's own resolution (see observation_precision)."""
    err = s.get("peak_time_error_minutes")
    if err is None:
        return "—"
    txt = f"{_fmt_num(err, 0, sign=True)} min"
    if s.get("peak_time_within_window"):
        w = s.get("actual_peak_time_window_minutes") or 0
        return f'<span title="inside the {w:.0f}-minute plateau the extreme occupied">{txt} \u2713</span>'
    return txt


def next_day_projection_map(summary):
    """target_date -> {"high": projection, "low": projection}.

    calibration_log already logs tomorrow_high_f/tomorrow_low_f on every
    refresh and summarize() already pairs each with the date it was aimed
    at, so the night-before number needs no new logging - it only needs
    re-keying from "the day the forecast was made" to "the day it was
    about", which is what the weekly table is indexed by.
    """
    out = {}
    for d in summary.get("days", []):
        for side in ("high", "low"):
            nd = d.get(f"next_day_{side}") or {}
            target = nd.get("target_date")
            if target and nd.get("projection") is not None:
                out.setdefault(target, {})[side] = nd["projection"]
    return out


def build_weekly_performance_table(rows, side, night_before=None):
    """One row per trailing day for a single side (high/low), all spec'd
    fields - wrapped in a horizontally-scrolling container by the
    template since there are too many columns for a phone-width card.

    Two of the columns exist to make the same day readable across lead
    times and against the official outcome:

    "Night before" is the projection made the previous evening, so the
    same-day estimate sitting one column to its left can be compared
    against a genuinely longer-lead call for the identical date.

    There is deliberately no separate "settled" column. finalize_day
    already adopts the CLI value as actual_peak_temp the moment CLI
    publishes, so on any finalized row the two are the same number by
    construction and a second column would only ever restate the first.
    What does vary is the *provenance*, so "Actual" is marked instead on
    the rows where it is not the official settled figure - which is the
    case a reader actually needs flagged, since only the CLI integer
    settles a contract and the stream can sit most of a degree away from
    it (2026-08-08's low: 57.2 streamed, 58 settled).
    """
    night_before = night_before or {}
    header = (
        "<tr><th>Date</th><th>Predicted</th><th>Night before</th>"
        "<th>1h before pred.</th><th>NWS 1h before</th>"
        "<th>Actual</th><th>1h before actual</th>"
        "<th>Temp Δ</th><th>Time Δ</th>"
        "<th>Kalshi peak vol.</th><th>Kalshi implied</th></tr>"
    )
    body = []
    flagged = []  # (date_label, reason) - rendered as a persistent caption below
    # the table, since the inline ⚠'s hover-only title tooltip is easy to
    # miss (doesn't work on mobile taps or in a static PNG export at all).
    for r in rows:
        s = r.get(side)
        date_label = r["date"][5:]  # MM-DD is plenty given the 7-day window
        if not s:
            body.append(f'<tr><td>{date_label}</td><td colspan="10" class="perf-nodata">no data</td></tr>')
            continue

        if s.get("data_quality_flag"):
            flag = f' <span class="perf-flag" title="{s["data_quality_flag"]}">⚠</span>'
            flagged.append((date_label, s["data_quality_flag"]))
        else:
            flag = ""
        predicted = (
            f"{_fmt_dt_short(s['predicted_peak_time'])} · {_fmt_num(s['predicted_peak_temp'])}°"
            if s["predicted_peak_temp"] is not None else "—"
        )
        actual = f"{_fmt_dt_short(s['actual_peak_time'])} · {_fmt_num(s['actual_peak_temp'])}°"
        kalshi_vol = (
            f"{s['kalshi_peak_volume_contracts']:,.0f} <span class=\"perf-subtle\">@ {_fmt_dt_short(s['kalshi_peak_volume_time'])}</span>"
            if s["kalshi_peak_volume_contracts"] is not None else "—"
        )
        kalshi_implied = (
            f"{s['kalshi_market_implied_value']:.0f}° <span class=\"perf-subtle\">({s['kalshi_market_implied_bracket']})</span>"
            if s["kalshi_market_implied_value"] is not None else "—"
        )
        nws_1hr_before = (
            f"{_fmt_num(s['nws_forecast_temp'])}°" if s.get("nws_forecast_temp") is not None else "—"
        )
        # Night-before projection for *this* date, plus its own error against
        # the settled value - the error is the whole point of the column, and
        # recomputing it here keeps it consistent with the settled number in
        # the next cell rather than with whatever "Actual" fell back to.
        nb_val = (night_before.get(r["date"]) or {}).get(side)
        settled_val = s.get("actual_peak_temp_cli_f")
        if nb_val is None:
            night_before_cell = "—"
        elif settled_val is None:
            night_before_cell = f"{_fmt_num(nb_val)}°"
        else:
            nb_err = nb_val - settled_val
            cls = "perf-err-good" if abs(nb_err) <= 1.5 else "perf-err-bad"
            night_before_cell = (
                f"{_fmt_num(nb_val)}° "
                f'<span class="perf-subtle {cls}">({nb_err:+.1f})</span>'
            )
        # Mark the actual only when it is NOT the settled CLI figure, so the
        # unmarked majority reads as "official" and the eye goes to the rows
        # where the number could still move.
        if settled_val is None:
            actual += ' <span class="perf-subtle" title="No CLI report yet - this is the observation stream, which can differ from the settled value">(unsettled)</span>'
        body.append(
            "<tr>"
            f"<td>{date_label}{flag}</td>"
            f"<td>{predicted}</td>"
            f"<td>{night_before_cell}</td>"
            f"<td>{_fmt_num(s['temp_1hr_before_predicted_peak'])}°</td>"
            f"<td>{nws_1hr_before}</td>"
            f"<td>{actual}</td>"
            f"<td>{_fmt_num(s['temp_1hr_before_actual_peak'])}°</td>"
            f"<td>{_fmt_num(s['peak_temp_error_f'], 2, sign=True) if s['peak_temp_error_f'] is not None else '—'}</td>"
            f"<td>{_format_time_delta(s)}</td>"
            f"<td>{kalshi_vol}</td>"
            f"<td>{kalshi_implied}</td>"
            "</tr>"
        )
    table_html = f'<table class="perf-table"><thead>{header}</thead><tbody>{"".join(body)}</tbody></table>'
    if not flagged:
        return table_html

    caption_lines = "".join(
        f'<div>⚠ {date_label}: {reason}</div>' for date_label, reason in flagged
    )
    return table_html + f'<div class="perf-flag-caption">{caption_lines}</div>'


def build_monthly_stats_rows(stats):
    has_anything = (
        stats["n_temp_error_samples"] > 0
        or stats.get("n_nws_error_samples", 0) > 0
        or stats["mean_kalshi_peak_volume_contracts"] is not None
        or stats["n_hit_samples"] > 0
    )
    if not has_anything:
        return '<div class="hint">No finalized days yet this month.</div>'

    def row(label, value):
        return f'<div class="range-row"><span class="range-label">{label}</span><span class="range-values">{value}</span></div>'

    rows = []
    if stats["n_temp_error_samples"] > 0:
        rows.append(row("Temp bias (signed)", f"{_fmt_num(stats['bias_f'], 2, sign=True)}°F ({stats['n_temp_error_samples']} days)"))
        rows.append(row("Temp MAE", f"{_fmt_num(stats['mae_f'], 2)}°F"))
    if stats.get("n_nws_error_samples", 0) > 0:
        rows.append(row("NWS bias (signed)", f"{_fmt_num(stats['nws_bias_f'], 2, sign=True)}°F ({stats['n_nws_error_samples']} days)"))
        rows.append(row("NWS MAE", f"{_fmt_num(stats['nws_mae_f'], 2)}°F"))
    if stats["n_temp_error_samples"] > 0 and stats.get("n_nws_error_samples", 0) > 0:
        diff = round(stats["mae_f"] - stats["nws_mae_f"], 2)
        if diff < 0:
            comparison = f"model's MAE is {abs(diff):.2f}°F lower (better) than NWS's this month"
        elif diff > 0:
            comparison = f"model's MAE is {diff:.2f}°F higher (worse) than NWS's this month"
        else:
            comparison = "model and NWS MAE are tied this month"
        rows.append(f'<div class="hint">{comparison}.</div>')
    if stats["time_bias_minutes"] is not None:
        rows.append(row("Peak-time bias (signed)", f"{_fmt_num(stats['time_bias_minutes'], 0, sign=True)} min"))
        rows.append(row("Peak-time MAE", f"{_fmt_num(stats['time_mae_minutes'], 0)} min"))
    if stats["mean_kalshi_peak_volume_contracts"] is not None:
        rows.append(row("Mean Kalshi peak volume", f"{stats['mean_kalshi_peak_volume_contracts']:,.0f} contracts"))
    if stats["model_in_kalshi_bracket_rate"] is not None:
        # Named for the exact bracket it tests. The weekly table's "Kalshi
        # implied" column is the market's favourite at the *actual peak*,
        # this is its favourite at the moment the model committed - they
        # disagree on real days, so a reader comparing the two by eye gets
        # a mismatch that looks like a bug in one of them. It is also
        # model-vs-market agreement, not market accuracy: whether the
        # settled value landed in the market's bracket is a different
        # question with a much higher rate.
        rows.append(row(
            "Model agreed w/ Kalshi favourite <span class=\"perf-subtle\">(at prediction time)</span>",
            f"{stats['model_in_kalshi_bracket_rate'] * 100:.0f}% ({stats['n_hit_samples']} days)"))
    if stats["n_temp_error_samples"] == 0:
        rows.append('<div class="hint">No model predictions logged yet this month - calibration_log.py only started capturing pre-peak predictions recently.</div>')
    return "\n".join(rows)


LEAD_TIME_LABELS = {"1hr": "1hr before peak", "2hr": "2hr before peak"}
LOW_STRATEGY_KEYS = ("edge", "unconditional")
LOW_STRATEGY_LABELS = {
    "edge": "Low - edge-triggered (1hr)",
    "unconditional": "Low - unconditional (1hr)",
}

STRATEGY_LABELS = {
    "same_day_edge": "Same-day · edge-triggered <span class=\"perf-subtle\">(1hr leg)</span>",
    "same_day_unconditional": "Same-day · unconditional <span class=\"perf-subtle\">(high + low)</span>",
    "advance_cashed_out": "Next-day advance · cashed out early",
    "advance_held": "Next-day advance · held to settlement",
}


def build_cashed_out_block(stats, label):
    """
    The cash-out population's own renderer. It deliberately shows no win
    rate and no "P&L" line: nothing here was ever scored against the
    weather, so a win rate would be undefined and putting its realized
    gains under the same heading as settlement P&L is exactly the
    collapse this feature exists to avoid. What it shows instead is what
    an early exit actually is - how many, how much realized, how long
    held, and how far above entry the exit bid was.
    """
    def row(l, v):
        return f'<div class="range-row"><span class="range-label">{l}</span><span class="range-values">{v}</span></div>'

    header = f'<div class="module-label" style="margin-top: 16px;">{label}</div>'
    if stats["n_positions"] == 0:
        return header + (
            '<div class="hint">No position has been cashed out early yet. A position is only '
            f'exited when the bid clears the entry ask by at least {MIN_EXIT_GAIN * 100:.0f} '
            'cents; otherwise it is carried into settlement and counted in the block below.</div>'
        )

    rows = [row("Positions exited early", f"{stats['n_positions']}")]
    if stats["total_realized_gain"] is not None:
        sign = "+" if stats["total_realized_gain"] >= 0 else ""
        rows.append(row("Realized on exit", f"{sign}${stats['total_realized_gain']:.2f}"))
    if stats["avg_exit_gain_per_contract"] is not None:
        rows.append(row("Avg. exit above entry", f"{stats['avg_exit_gain_per_contract'] * 100:.1f} cents"))
    if stats["avg_hours_held"] is not None:
        rows.append(row("Avg. holding time", f"{stats['avg_hours_held']:.1f} h"))

    block = header + "\n".join(rows)
    if stats["low_sample"]:
        block += (
            f'<div class="hint">Only {stats["n_positions"]} early exit(s) so far - too few to read '
            f'as a rate or a strategy result (needs at least {LOW_SAMPLE_THRESHOLD}). Shown as a '
            'running count, not a performance claim.</div>'
        )
    block += (
        '<div class="hint">Bought at the ask, sold at the bid - the spread is paid on both legs, '
        'never a mid-price fill. These gains are NOT added to any settlement P&amp;L below: '
        '"the market re-priced in our favour" and "the forecast was right" are different claims.</div>'
    )
    return block


def build_strategy_comparison(stats_by_key):
    """All four strategies, one block each, never summed. The three
    settlement-resolved populations share build_paper_trading_rows'
    renderer; the cash-out population gets its own because its numbers
    mean something different (see build_cashed_out_block)."""
    settled_keys = ("same_day_edge", "same_day_unconditional", "advance_held")
    parts = [
        build_paper_trading_rows(
            {k: stats_by_key[k] for k in settled_keys[:2]},
            keys=settled_keys[:2], labels=STRATEGY_LABELS,
        ),
        build_cashed_out_block(stats_by_key["advance_cashed_out"], STRATEGY_LABELS["advance_cashed_out"]),
        # Wrapped rather than relying on build_paper_trading_rows' own
        # spacing: that function only margins blocks after the first, and
        # this is a single-key call that is nonetheless the third block on
        # the page.
        '<div style="margin-top: 16px;">'
        + build_paper_trading_rows(
            {"advance_held": stats_by_key["advance_held"]},
            keys=("advance_held",), labels=STRATEGY_LABELS,
        )
        + "</div>",
    ]
    return "\n".join(parts)


def build_paper_trading_rows(stats_by_key, keys=LEAD_TIME_HINTS, labels=LEAD_TIME_LABELS):
    """
    stats_by_key: a dict of {key: {...}} shaped like daily_performance.
    _aggregate_paper_trading_bets's return, for each of `keys`. Rendered
    as independent blocks - NEVER averaged/combined into one set of
    numbers, since the whole point of running multiple populations in
    parallel is comparing them against each other (see paper_trading.py's
    module docstring); collapsing them here would quietly erase the
    comparison the feature exists to make. Used both for the existing
    lead-time split (keys=LEAD_TIME_HINTS, the default) and for the low
    market's edge-vs-unconditional strategy split (keys=LOW_STRATEGY_KEYS)
    - same rendering rules either way. A block with fewer than
    LOW_SAMPLE_THRESHOLD bets shows "insufficient data" instead of a
    percentage/comparison that thin would misrepresent as meaningful -
    same guard monthly_rollup already uses for low_sample.
    """
    def row(label, value):
        return f'<div class="range-row"><span class="range-label">{label}</span><span class="range-values">{value}</span></div>'

    blocks = []
    for i, key in enumerate(keys):
        stats = stats_by_key[key]
        label_style = "" if i == 0 else " style=\"margin-top: 16px;\""
        header = f'<div class="module-label"{label_style}>{labels[key]}</div>'

        if stats["n_bets"] == 0:
            blocks.append(header + '<div class="hint">No simulated bets placed yet in this window.</div>')
            continue

        if stats["low_sample"]:
            blocks.append(
                header
                + f'<div class="hint">Insufficient data - only {stats["n_bets"]} bet(s) so far '
                f'(need at least {LOW_SAMPLE_THRESHOLD} to treat win rate/band coverage as meaningful).</div>'
            )
            continue

        rows = [
            row("Simulated P&L", f"{'+' if stats['total_pnl'] >= 0 else ''}${stats['total_pnl']:.2f}"),
            row("Win rate", f"{stats['win_rate'] * 100:.0f}% ({stats['n_bets']} bets)"),
        ]
        if stats["avg_edge_at_entry"] is not None:
            rows.append(row("Avg. edge at entry", f"{stats['avg_edge_at_entry'] * 100:.0f} pts"))
        if stats["pct_within_uncertainty_band"] is not None:
            rows.append(row("Within stated uncertainty band", f"{stats['pct_within_uncertainty_band'] * 100:.0f}%"))

        block_html = header + "\n".join(rows)
        mvm = stats["model_vs_market"]
        if mvm is not None:
            block_html += (
                f'<div class="hint">On the {mvm["n_high_disagreement_bets"]} bet(s) where the model and Kalshi '
                f'disagreed most: model was closer to the actual outcome {mvm["model_closer_count"]} time(s), '
                f'market was closer {mvm["market_closer_count"]} time(s).</div>'
            )
        blocks.append(block_html)

    return "\n".join(blocks)




THIN_VOLUME_THRESHOLD = 5  # contracts traded - below this, last_price is easy to be stale/unreliable


def build_kalshi_rows(brackets, our_estimate, estimate_label, observed_source=None):
    """
    HTML rows for one Kalshi bracket market, highlighting whichever bracket
    our own point estimate currently falls into - a quick visual check of
    whether the model and the market agree, without computing a full
    probability distribution (that's a deliberate next step, not this one).

    Also prints our_estimate itself right above the brackets, so the number
    driving the highlight is legible next to the market's own pricing - not
    just implied by which row lit up.

    observed_source distinguishes an OBSERVED extreme's provenance. The
    5-minute feed reports whole degrees Celsius, so its extremes are
    1.8F-granular however many decimals the conversion prints - 57.20F is
    exactly 14.0C, and printing it to hundredths claims a resolution the
    sensor never had. But the fix for that is upstream, not here: ASOS
    publishes its own un-quantised extreme in the METAR remarks (see
    asos_extremes), so once the right field is read the ambiguity mostly
    evaporates. On 2026-07-29 the feed said 57.20F, spanning three brackets;
    the station's own 6-hourly minimum said 14.4C = 57.92F, which settles to
    58 - one bracket, and the one the market was actually priced at.

    So exactly one bracket is ever highlighted. An earlier version lit every
    bracket a quantised reading could reach, which was honest about the feed
    but useless to act on - "it might be any of these three" is not an
    answer. When only the quantised value is available the header still says
    so in a caption, but the pick itself stays single.

    last_price is the most recent trade, not "percent of people betting" -
    it's the market's implied probability (yes/no contracts settle at $1/$0,
    so price ~= probability under normal arbitrage). Bid/ask and volume are
    shown alongside it since a lightly-traded bracket's last_price can be
    stale; low-volume rows are dimmed as a caution, not hidden.
    """
    if not brackets:
        return '<div class="hint">Market unavailable.</div>'

    if observed_source == "asos_remark_1min":
        head = f'{estimate_label}: <strong>{our_estimate:.1f}°F</strong>'
        sub = ('<div class="kalshi-band-note">station\'s own 1-minute figure '
               'from the METAR remarks - not the rounded 5-minute feed</div>')
    elif observed_source == "observation_stream":
        head = f'{estimate_label}: <strong>{format_reading(our_estimate, unit="°F")}</strong>'
        note = precision_note(our_estimate)
        sub = f'<div class="kalshi-band-note">{note}</div>' if note else ""
    else:
        head = f'{estimate_label}: <strong>{our_estimate:.2f}°F</strong>'
        sub = ""

    rows = [f'<div class="kalshi-estimate">{head}</div>{sub}']
    for b in brackets:
        pct = round(b["last_price"] * 100) if b["last_price"] is not None else None
        pct_label = f"{pct}%" if pct is not None else "—"

        bid = round(b["yes_bid"] * 100) if b["yes_bid"] is not None else None
        ask = round(b["yes_ask"] * 100) if b["yes_ask"] is not None else None
        spread_label = f"bid {bid}¢ / ask {ask}¢" if bid is not None and ask is not None else "no quote"

        is_thin = b["volume"] is None or b["volume"] < THIN_VOLUME_THRESHOLD
        thin_class = " kalshi-row-thin" if is_thin else ""
        thin_flag = ' <span class="kalshi-thin-flag">thin</span>' if is_thin else ""

        is_match = bracket_contains(b, our_estimate)
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


def format_settled_reading(temp_f, source):
    """A finished day's extreme, rendered according to whether it is
    actually finished.

    format_reading's "~60-62" range is the honest rendering of a live
    observation-stream value: 60.80F is a quantised 16.0C standing in for a
    settlement number that hasn't been published, so the +/-0.9F is real.
    But once the CLI report lands, the settlement number IS published -
    it's the integer Kalshi pays out on and daily_performance scores
    against - and dressing it back up as a range would be reporting
    uncertainty that has already been resolved.
    """
    if temp_f is None:
        return "—"
    if source == "cli_final":
        return f"{temp_f:.0f}"
    return format_reading(temp_f, unit="")


def yesterday_source_note(source, lag_hours=None, overdue=False):
    """Which of the two readings above is on screen, in one short line.

    "hasn't published yet" is true whether it's 3am or 3pm, which makes it
    useless exactly when it matters. The final report normally lands ~1:25am
    local, six of the last seven within six minutes of each other (see
    nws_climate.CLI_FINAL_TYPICAL_LAG_HOURS), so a wait of hours past that is
    an outlier worth naming rather than the same sentence in a different
    light.
    """
    if source == "cli_final":
        return "final NWS climate report (CLI) - the value Kalshi settled on"
    if overdue and lag_hours is not None:
        return (
            f"from the observation stream - NWS's final report is about "
            f"{lag_hours:.0f}h overdue (it normally lands ~1:25 am). Unusual, "
            f"not unheard of; the value can still move when it publishes."
        )
    return (
        "from the observation stream; NWS's final report normally lands "
        "~1:25 am and isn't out yet"
    )


def trend_significance_note(slope, slope_se):
    """A note for when the raw local trend is indistinguishable from no
    trend at all.

    "+1.42°/hr ± 2.20°/hr" is not a warming trend - the error bar spans
    zero, so the sign isn't even established, and the last few observations
    are consistent with the temperature going nowhere. Shown on its own the
    +1.42 reads as a claim the fit does not support, and it looked like a
    bug when it was the model correctly reporting a noisy window. The
    downstream damping already discounts this (that's what the confidence %
    is doing), so this is a display gap rather than a modelling one.
    """
    if slope is None or slope_se is None or slope_se <= 0:
        return ""
    if abs(slope) > slope_se:
        return ""
    return "not statistically significant right now - the error bar spans zero, so treat this as flat"
TOMORROW_SOURCE_BLURB = {
    "nws_forecast": "From the NWS hourly forecast (HRRR model) - real atmospheric dynamics, not this tool's own trend/persistence guess.",
    "persistence_fallback": "NWS forecast unavailable - falling back to a persistence guess (today's/yesterday's high, nudged by pressure trend).",
}


def tomorrow_hint(source, measured_pct, n_samples, min_samples=MIN_NEXT_DAY_SAMPLES):
    """
    measured_pct/n_samples: see calibration_log.next_day_confidence_pct.
    The confidence % shown next to "tomorrow's high/low" used to be a
    hand-picked constant with copy promising it "will switch to a measured
    number once enough days accumulate" - nothing ever computed that
    number, so the promise was aspirational copy, not a real feature. This
    reports the actual state instead: a measured figure once
    next_day_confidence_pct has enough source-matched samples, otherwise
    an honest placeholder that says how many of the needed samples exist
    so far.
    """
    base = TOMORROW_SOURCE_BLURB[source]
    if measured_pct is not None:
        return (
            f"{base} Reference confidence is measured from the last {n_samples} finalized "
            f"next-day predictions using this same source (calibration_log.py) - not a placeholder."
        )
    return (
        f"{base} The reference confidence % is still a placeholder, not a backtested figure - "
        f"only {n_samples} of {min_samples} finalized next-day predictions using this source are "
        f"logged so far; this will switch to a measured number once enough accumulate."
    )
def build_condition_confidence_html(same_day_confidence):
    """
    The "how well has this model done on days that looked like today"
    panel, one row per side.

    Deliberately shows the sample count and the bucket it came from in the
    same breath as the number. "82%" alone invites the reader to treat all
    such numbers alike, when one may rest on seven matched days and another
    on a pooled thirteen because no specific bucket was populated enough
    yet - and the difference between those two claims is the entire point
    of bucketing. When a more specific bucket was tried and rejected, that
    is stated too, with the count it fell short at, so a reader can see the
    specific figure coming rather than wonder whether it exists.
    """
    rows = []
    for side, arrow, cls in (("high", "↑", "hilo-arrow-high"),
                             ("low", "↓", "hilo-arrow-low")):
        conf = same_day_confidence.get(side)
        if conf is None:
            rows.append(
                f'<div class="hilo-detail"><span class="hilo-detail-arrow {cls}">{arrow}</span>'
                f'<span class="hilo-value">—</span>'
                f'<span class="hilo-detail-caption">not enough finalized days yet</span></div>'
            )
            continue
        if conf["matched"]:
            basis = f'based on {conf["n"]} days with {conf["label"]}'
        else:
            specific = [
                (axes, n) for axes, n in conf["tried"] if axes and n
            ]
            shortfall = (
                f' &middot; no specific match yet ({specific[0][1]} of '
                f'{MIN_CONDITION_BUCKET_SAMPLES} needed)'
                if specific else ""
            )
            basis = f'based on all {conf["n"]} finalized days{shortfall}'
        rows.append(
            f'<div class="hilo-detail"><span class="hilo-detail-arrow {cls}">{arrow}</span>'
            f'<span class="hilo-value">{conf["pct"]}%</span>'
            f'<span class="hilo-detail-caption">{basis}</span>'
            f'<span class="hilo-detail-caption hero-precision">MAE {conf["mae_f"]}°F, '
            f'bias {conf["bias_f"]:+.2f}°F</span></div>'
        )
    return (
        '<div class="module-label" style="margin-top: 16px;">Confidence, '
        'from similar past days</div>'
        f'<div class="hilo-detail-row">{"".join(rows)}</div>'
        '<div class="hint">Measured from finalized days whose marine-push, '
        'offshore-flow and pressure-gradient state at prediction time matched '
        "today's, not one average across every day regardless of conditions "
        '(calibration_log.py). Falls back to the all-days figure until a '
        f'specific combination has {MIN_CONDITION_BUCKET_SAMPLES} matching days '
        'behind it - the sample count above always says which one you are '
        'looking at.</div>'
    )


LOW_CAPTIONS = {
    "today": "today's overnight low, almost here",
    "tonight": "expected low tonight",
}


def build_next_day_table(summary, days=8):
    """
    How the *next-day* forecast has actually done - the panel that was
    missing. calibration_log has logged tomorrow_high_f/tomorrow_low_f on
    every refresh since the beginning, and summarize() has been scoring
    them against the following day's settled extreme, but none of it was
    ever rendered, so the only next-day number on the page was a
    confidence % with nothing behind it you could check.

    Rows the summary marks as unscored (error None) are still listed, with
    the error blank - that is the in-progress day, and showing it as a row
    with no verdict is more honest than hiding the prediction until it can
    be graded.
    """
    scored = [
        d for d in summary["days"]
        if d["next_day_high"]["projection"] is not None
        or d["next_day_low"]["projection"] is not None
    ][-days:]
    if not scored:
        return '<div class="hint">No next-day forecasts logged yet.</div>'

    def cell(v, fmt="{:.1f}"):
        return fmt.format(v) if isinstance(v, (int, float)) else "—"

    def err_cell(v):
        if not isinstance(v, (int, float)):
            return '<span class="perf-pending">pending</span>'
        cls = "perf-err-good" if abs(v) <= 1.5 else "perf-err-bad"
        return f'<span class="{cls}">{v:+.1f}</span>'

    rows = [
        "<table class='perf-table'><thead><tr>"
        "<th>Forecast for</th><th>High est.</th><th>Actual</th><th>Err</th>"
        "<th>Low est.</th><th>Actual</th><th>Err</th></tr></thead><tbody>"
    ]
    for d in reversed(scored):
        nh, nl = d["next_day_high"], d["next_day_low"]
        # The newest row has no following date logged yet - its target is
        # simply the day after it. Deriving it beats printing a dash on the
        # one row a reader is most likely to be looking for.
        target = nh.get("target_date")
        if not target:
            target = (date.fromisoformat(d["date"]) + timedelta(days=1)).isoformat()
        rows.append(
            f"<tr><td>{target[5:]}</td>"
            f"<td>{cell(nh['projection'])}</td><td>{cell(nh['final'])}</td><td>{err_cell(nh['error'])}</td>"
            f"<td>{cell(nl['projection'])}</td><td>{cell(nl['final'])}</td><td>{err_cell(nl['error'])}</td></tr>"
        )
    rows.append("</tbody></table>")

    def stat_line(label, st):
        if not st or not st.get("n"):
            return f"{label}: not enough scored days yet"
        return (f"{label}: MAE {st['mae_f']:.2f}°F, bias {st['bias_f']:+.2f}°F "
                f"over {st['n']} day{'s' if st['n'] != 1 else ''}")

    rows.append(
        '<div class="hint">'
        + stat_line("High", summary.get("next_day_high_stats"))
        + " · " + stat_line("Low", summary.get("next_day_low_stats"))
        + ". Bias is signed forecast minus actual, so positive means the "
        "forecast ran warm. Days still in progress show <em>pending</em> - "
        "they are excluded from the averages, because scoring against a "
        "partial day\'s high is what made these numbers look twice as bad "
        "as they are.</div>"
    )
    return "\n".join(rows)


def blend_source_note(weight, trend_only_f, nws_f):
    """Where today's still-pending estimate is actually coming from.

    The mix moves through the day (see weather_estimator's
    NWS_BLEND_START_HOURS), so a static "trend model" label would be wrong
    most of the time. Showing both inputs alongside the weight also keeps
    the comparison the blend was going to hide: the in-house model's own
    number is still on screen even when it isn't the one being used.
    """
    if weight is None:
        return ""
    parts = []
    if trend_only_f is not None:
        parts.append(f"local trend {trend_only_f:.1f}°")
    if nws_f is not None:
        parts.append(f"NWS {nws_f:.1f}°")
    detail = f" ({' · '.join(parts)})" if parts else ""
    if weight <= 0:
        return f"local trend model only - close enough in that it beats NWS here{detail}"
    if weight >= 1:
        return f"NWS hourly forecast - too far out for the local trend to be worth anything{detail}"
    return f"{round((1 - weight) * 100)}% local trend / {round(weight * 100)}% NWS forecast{detail}"


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


# Elements that never get a closing tag, so they must not move the nesting
# depth while we walk the document.
_VOID_TAGS = frozenset(
    "area base br col embed hr img input link meta param source track wbr".split()
)


class _ContentChildren(HTMLParser):
    """Collect the direct children of <div class="content">.

    Only the top level matters: those are the grid items. For each one we
    record its classes and whether it contains a .perf-scroll anywhere
    inside, since that is what the CSS keys full width off.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_content = False
        self.depth = 0
        self.children = []
        self._cur = None

    def handle_startendtag(self, tag, attrs):
        # <line ... /> and friends inside the SVGs: self-closing, so they
        # neither open a child nor change depth.
        if self.in_content and self._cur is not None:
            if "perf-scroll" in (dict(attrs).get("class") or "").split():
                self._cur["perf"] = True

    def handle_starttag(self, tag, attrs):
        if tag in _VOID_TAGS:
            return
        classes = (dict(attrs).get("class") or "").split()
        if not self.in_content:
            if tag == "div" and "content" in classes:
                self.in_content = True
                self.depth = 0
            return
        if self.depth == 0:
            self._cur = {"classes": classes, "perf": False}
            self.children.append(self._cur)
        elif self._cur is not None and "perf-scroll" in classes:
            self._cur["perf"] = True
        self.depth += 1

    def handle_endtag(self, tag):
        if not self.in_content or tag in _VOID_TAGS:
            return
        self.depth -= 1
        if self.depth <= 0:
            self._cur = None
            if self.depth < 0:
                self.in_content = False


def check_panel_layout(html):
    """Warn when the desktop grid would leave an empty cell beside a panel.

    Above 860px the panels flow in DOM order into two equal columns, and a
    full-width panel always starts a fresh row. So every run of half-width
    panels bounded by full-width ones has to be an even count - an odd run
    leaves its last panel alone in column 1 with a visible hole beside it,
    which is what put a 673px gap next to the tomorrow's-low volume chart.

    CSS has no selector for "last item in a row", so the invariant cannot
    live in the stylesheet and is checked here instead. This warns rather
    than raising: an empty grid cell is cosmetic, and the hourly refresh
    publishing a slightly gapped dashboard beats it publishing nothing.
    """
    parser = _ContentChildren()
    parser.feed(html)
    if not parser.children:
        print("LAYOUT_WARNING: could not find .content children to check")
        return

    def is_wide(child):
        cls = child["classes"]
        return (
            "wide" in cls
            or "place" in cls
            or "hero" in cls
            or "footer" in cls
            or child["perf"]  # .glass:has(.perf-scroll)
        )

    runs, run = [], 0
    for child in parser.children:
        if is_wide(child):
            runs.append(run)
            run = 0
        else:
            run += 1
    runs.append(run)

    odd = [n for n in runs if n % 2]
    if odd:
        print(
            f"LAYOUT_WARNING: {len(odd)} run(s) of half-width panels have an odd "
            f"count {odd} - the last panel in each will sit alone in column 1 "
            f"with an empty cell beside it. Mark that panel .wide (it must be "
            f"the LAST of the run; widening an earlier one just moves the hole)."
        )
    else:
        n_half = sum(runs)
        print(
            f"Panel layout OK: {len(parser.children)} grid items, {n_half} "
            f"half-width in even runs {[n for n in runs if n]}, no orphan cells."
        )


def main():
    est = estimate_temp(STATION, hours_ahead=HOURS_AHEAD)
    extremes = estimate_daily_extremes(STATION)
    lat, lon, _ = get_station_location(STATION)

    record_snapshot(extremes, est=est)

    now = est["as_of"]

    try:
        finalize_pending_days(STATION, lookback_days=7)
    except Exception as e:
        print(f"daily_performance: finalize_pending_days failed: {e}")

    try:
        reconcile_stream_fallback_actuals(STATION, lookback_days=5)
    except Exception as e:
        print(f"daily_performance: reconcile_stream_fallback_actuals failed: {e}")

    try:
        reconcile_peak_time_windows(STATION, lookback_days=10)
    except Exception as e:
        print(f"daily_performance: reconcile_peak_time_windows failed: {e}")

    # Loudly surface any calibration input that is running on a fallback
    # constant instead of a real computed value. The band-coverage bug was
    # invisible for its entire life precisely because a fallback looked
    # identical to success; this makes the difference legible on every
    # refresh. Never fatal - a degraded input still produces a page.
    try:
        import calibration_health
        if calibration_health.report(STATION):
            print("calibration: all sources computed (no fallbacks in use)")
    except Exception as e:
        print(f"calibration_health: check failed: {e}")

    try:
        lock_result = get_or_lock_daily_targets(STATION)
        for date_str, side in lock_result["newly_locked"]:
            side_state = lock_result["state"][date_str][side]
            if not side_state["skipped_missed_window"]:
                for checkpoint in side_state["checkpoints"]:
                    print(f"ALERT_SCHEDULE_NEEDED side={side} date={date_str} target_alert_time={checkpoint}")
    except Exception as e:
        print(f"peak_alerts: get_or_lock_daily_targets failed: {e}")

    try:
        lock_2hr_result = get_or_lock_2hr_targets(STATION)
        for date_str, side in lock_2hr_result["newly_locked"]:
            side_state = lock_2hr_result["state"][date_str][side]
            if not side_state["skipped_missed_window"]:
                print(f"PAPER_TRADE_2HR_SCHEDULE_NEEDED side={side} date={date_str} target_time={side_state['target_time']}")
    except Exception as e:
        print(f"paper_trading: get_or_lock_2hr_targets failed: {e}")

    weekly_perf = weekly_table(STATION, days=7)
    monthly_perf = monthly_rollup(STATION, now.year, now.month)

    # One summarize() for both the next-day table and the weekly tables'
    # night-before column. Degrading to an empty map rather than raising
    # keeps a calibration-log problem from taking down the whole page: the
    # weekly tables still render, just with "—" in that one column.
    try:
        _calib_summary = summarize()
    except Exception as e:
        print(f"calibration_log: summarize() failed, night-before column empty: {e}")
        _calib_summary = {"days": []}
    _night_before = next_day_projection_map(_calib_summary)

    # Kalshi's KSEA markets are dated by Seattle's own calendar day, not
    # the system clock's - this container runs on UTC, which is already
    # into the next day while it's still evening in Seattle (UTC-7/8).
    # date.today() here would silently fetch tomorrow's just-opened, still
    # nearly-empty event as if it were "today's" actively-trading one.
    today = now.date()
    tomorrow = today + timedelta(days=1)
    # get_market_for_date itself returns None for a genuine "no open event
    # for this date yet" - a normal, expected state, especially for
    # tomorrow's markets before Kalshi opens them. A raised exception here
    # is a different thing entirely (a transient network hiccup or Kalshi
    # rate-limiting - both observed in practice), so it's tracked
    # separately rather than collapsed into the same "no market" None the
    # UI would otherwise show identically for both cases.
    kalshi_high_error = None
    try:
        kalshi_high = get_market_for_date(HIGH_SERIES, today)
    except Exception as e:
        print(f"Kalshi high market fetch failed: {e}")
        kalshi_high = None
        kalshi_high_error = str(e)
    kalshi_low_error = None
    try:
        kalshi_low = get_market_for_date(LOW_SERIES, today)
    except Exception as e:
        print(f"Kalshi low market fetch failed: {e}")
        kalshi_low = None
        kalshi_low_error = str(e)

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
    kalshi_tomorrow_high_error = None
    try:
        kalshi_tomorrow_high = get_market_for_date(HIGH_SERIES, tomorrow)
    except Exception as e:
        print(f"Kalshi tomorrow high market fetch failed: {e}")
        kalshi_tomorrow_high = None
        kalshi_tomorrow_high_error = str(e)
    kalshi_tomorrow_low_error = None
    try:
        kalshi_tomorrow_low = get_market_for_date(LOW_SERIES, tomorrow)
    except Exception as e:
        print(f"Kalshi tomorrow low market fetch failed: {e}")
        kalshi_tomorrow_low = None
        kalshi_tomorrow_low_error = str(e)

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

    try:
        measured_conf = next_day_confidence_pct(extremes["tomorrow_high_source"], extremes["tomorrow_low_source"])
    except Exception as e:
        print(f"calibration_log: next_day_confidence_pct failed: {e}")
        measured_conf = {"high": None, "low": None, "n_high": 0, "n_low": 0}

    tomorrow_high_confidence_pct = (
        measured_conf["high"] if measured_conf["high"] is not None else extremes["tomorrow_high_confidence_pct"]
    )
    tomorrow_low_confidence_pct = (
        measured_conf["low"] if measured_conf["low"] is not None else extremes["tomorrow_low_confidence_pct"]
    )

    # Same-day confidence, measured against the finalized days whose
    # conditions at prediction time looked like right now's rather than
    # against every day pooled together. `est` already carries exactly the
    # index fields classify_conditions reads, so the live state is
    # classified by the identical function and thresholds that classified
    # the historical rows - if those two ever drifted apart the buckets
    # would be comparing a day to days it isn't actually like.
    today_conditions = {}
    same_day_confidence = {"high": None, "low": None}
    try:
        today_conditions = classify_conditions(est)
        _summary = summarize()
        for _side in ("high", "low"):
            same_day_confidence[_side] = condition_confidence(
                _side, today_conditions, summary=_summary)
    except Exception as e:
        # Never fatal: the panel degrades to its explanatory note. Loud
        # rather than silent for the same reason the NWS capture is - a
        # quietly missing confidence figure looks identical to one that is
        # legitimately still accumulating samples.
        print(f"calibration_log: condition_confidence failed: {e}")

    same_day_confidence_html = build_condition_confidence_html(same_day_confidence)

    window_start = now - timedelta(hours=SPARKLINE_HOURS)
    hist = get_observation_history(STATION, start=window_start, end=now)

    times = list(hist["time"])
    temps = list(hist["temp_f"])

    svg = build_sparkline_svg(times, temps, est["target_time"], est["estimated_temp_f"])

    today_max_gap_minutes = extremes.get("today_max_gap_minutes")

    lo, hi = est["estimated_range_f"]
    band_width_f = hi - lo

    p_label, p_class = pressure_chip(est["pressure_trend_inhg_per_hr"])

    marine_push_label, marine_push_class = index_chip(est["marine_push_index"], MARINE_PUSH_INDEX_THRESHOLD, "rising")
    offshore_flow_label, offshore_flow_class = index_chip(est["offshore_flow_index"], OFFSHORE_FLOW_INDEX_THRESHOLD, "rising")
    strait_signal = est.get("strait_signal") or {}
    interior_gap_signal = est.get("interior_gap_signal") or {}
    coastal_gradient_station = est.get("pressure_gradient_station")
    if coastal_gradient_station and strait_signal.get("station"):
        marine_push_meta = f"via {coastal_gradient_station} + {strait_signal['station']}"
    elif coastal_gradient_station:
        marine_push_meta = f"via {coastal_gradient_station}"
    elif strait_signal.get("station"):
        marine_push_meta = f"via {strait_signal['station']}"
    else:
        marine_push_meta = "no station data available"
    offshore_flow_meta = f"via {interior_gap_signal['station']}" if interior_gap_signal.get("station") else "no interior station available"
    uncertainty_note_html = (
        f'<div class="hint" style="margin-top: 6px;">Widened: {est["uncertainty_note"]}.</div>'
        if est.get("uncertainty_note") else ""
    )

    cloud_pct = est["cloud_fraction"]
    cloud_label = f"{round(cloud_pct * 100)}%" if cloud_pct is not None else "—"
    cloud_icon_svg = build_cloud_icon_svg(cloud_pct * 100 if cloud_pct is not None else None)

    confidence_pct = round(est["diurnal_damping"] * est["sky_wind_damping"] * 100)

    sunrise_h, sunset_h = get_sun_times(lat, lon, now.date())
    now_h = now.hour + now.minute / 60
    is_day = sunrise_h <= now_h < sunset_h
    condition_text, sky_class = sky_condition(cloud_pct, is_day)

    # Second location: Seattle proper (Fremont/Aurora) - see FREMONT_LAT/LON
    # note above. Deliberately simple: real ground-truth conditions from
    # the nearest full ASOS station (KBFI), plus NWS's own gridpoint
    # forecast for Fremont's actual coordinates - no trend/gradient-network/
    # backtesting machinery, and never blended into the KSEA numbers above.
    try:
        fremont_obs = get_observation_history(FREMONT_OBS_STATION, limit=1)
        fremont_latest = fremont_obs.iloc[-1]
        fremont_current_temp = fremont_latest["temp_f"]
        fremont_wind_mph = fremont_latest["wind_mph"]
        fremont_cloud_fraction = fremont_latest["cloud_fraction"]
        fremont_obs_time = fremont_latest["time"]
    except Exception as e:
        print(f"Fremont/{FREMONT_OBS_STATION} observation fetch failed: {e}")
        fremont_current_temp = fremont_wind_mph = fremont_cloud_fraction = fremont_obs_time = None

    try:
        fremont_forecast_df = get_hourly_forecast(FREMONT_LAT, FREMONT_LON, hours=24)
        fremont_today_forecast = fremont_forecast_df[fremont_forecast_df["time"].dt.date == now.date()]
        if fremont_today_forecast.empty:
            raise ValueError("forecast didn't include today's date")
        f_high_row = fremont_today_forecast.loc[fremont_today_forecast["forecast_temp_f"].idxmax()]
        f_low_row = fremont_today_forecast.loc[fremont_today_forecast["forecast_temp_f"].idxmin()]
        fremont_forecast_high = float(f_high_row["forecast_temp_f"])
        fremont_forecast_high_time = f_high_row["time"]
        fremont_forecast_low = float(f_low_row["forecast_temp_f"])
        fremont_forecast_low_time = f_low_row["time"]
    except Exception as e:
        print(f"Fremont NWS gridpoint forecast fetch failed: {e}")
        fremont_forecast_high = fremont_forecast_high_time = None
        fremont_forecast_low = fremont_forecast_low_time = None

    fremont_condition_text = (
        sky_condition(fremont_cloud_fraction, is_day)[0] if fremont_cloud_fraction is not None else "—"
    )

    obs_json_url = f"https://api.weather.gov/stations/{STATION}/observations"
    obhistory_url = f"https://forecast.weather.gov/data/obhistory/{STATION}.html"
    forecast_url = f"https://forecast.weather.gov/MapClick.php?lat={lat:.4f}&lon={lon:.4f}"
    timeseries_url = f"https://www.weather.gov/wrh/timeseries?site={STATION}"

    ctx = {
        "station_name": est["station_name"],
        "station_id": est["station"],
        "as_of_time": _fmt_time(now),
        "as_of_date": now.strftime("%A, %B %-d"),
        # Rendered at the reading's real resolution: 69.80 is exactly 21.0C,
        # a whole-degree-C report carrying +/-0.9F, so two decimals were
        # inventing precision the sensor never reported.
        # Hero stays a single whole number - the range treatment belongs on
        # the retrospective high/low, where the exact value decides a
        # bracket. Two decimals would be worse still: 66.20F is exactly
        # 19.0C, so those digits are a unit-conversion artefact.
        "current_temp": format_headline_reading(temps[-1]),
        "current_temp_precision_note": headline_precision_note(temps[-1]) or "",
        "target_time": _fmt_day_time(est["target_time"]),
        "estimated_temp": f"{est['estimated_temp_f']:.2f}",
        "range_low": f"{lo:.2f}",
        "range_high": f"{hi:.2f}",
        "band_width": f"{band_width_f:.2f}",
        "hours_ahead": HOURS_AHEAD,
        "trend_per_hr": f"{est['raw_trend_f_per_hr']:+.2f}",
        # (c) how well-determined that slope is. A trend of -4.1 ±0.2 and one
        # of -4.1 ±2.8 are very different claims; only one of them was ever
        # shown before.
        "trend_slope_se": (
            f"±{est['trend_slope_se_f_per_hr']:.2f}"
            if est.get("trend_slope_se_f_per_hr") is not None else "—"
        ),
        "trend_uncertainty_contrib": (
            f"widens the band by {est['trend_uncertainty_f']:.1f}°F"
            if est.get("trend_uncertainty_f") else "not material at this horizon"
        ),
        "trend_significance_note": trend_significance_note(
            est.get("raw_trend_f_per_hr"), est.get("trend_slope_se_f_per_hr")
        ),
        "wind_mph": f"{est['wind_mph']:.2f}" if est["wind_mph"] is not None else "—",
        "cloud_label": cloud_label,
        "cloud_icon_svg": cloud_icon_svg,
        "pressure_label": p_label,
        "pressure_class": p_class,
        "marine_push_label": marine_push_label,
        "marine_push_class": marine_push_class,
        "marine_push_meta": marine_push_meta,
        "offshore_flow_label": offshore_flow_label,
        "offshore_flow_class": offshore_flow_class,
        "offshore_flow_meta": offshore_flow_meta,
        "uncertainty_note_html": uncertainty_note_html,
        "confidence_pct": confidence_pct,
        "n_observations": est["n_observations"],
        "sparkline_svg": svg,
        "sparkline_hours": SPARKLINE_HOURS,
        "metric_switcher_html": build_metric_switcher_html(hist),
        "data_json": json.dumps(est, default=str, indent=2),
        # A model estimate is not a station reading, so format_reading's
        # quantisation range never applied here - it only ever fired when an
        # estimate happened to land on a whole degree Celsius, which made an
        # 83.00 estimate print as "83.0" and a 59.00 one as "≈58-60" for no
        # reason but arithmetic coincidence. The estimate's real uncertainty
        # is the uncertainty band, shown separately.
        "daily_high": format_headline_reading(extremes["estimated_high_f"]),
        "daily_high_caption": HIGH_CAPTIONS[extremes["high_status"]],
        "daily_high_source_note": blend_source_note(
            extremes.get("high_nws_blend_weight"),
            extremes.get("trend_only_high_f"),
            # The daily extremum is what the blend consumed, so it is what
            # the caption must cite - quoting the at-target sample here would
            # print a number that doesn't reconcile with the value above it.
            extremes.get("nws_high_forecast_daily_f")
            or extremes.get("nws_high_forecast_at_target_f"),
        ),
        "est_peak_time": _fmt_time(extremes["estimated_high_time"]),
        "daily_low": format_headline_reading(extremes["estimated_low_f"]),
        "daily_low_caption": LOW_CAPTIONS[extremes["low_status"]],
        "daily_low_source_note": blend_source_note(
            extremes.get("low_nws_blend_weight"),
            extremes.get("trend_only_low_f"),
            extremes.get("nws_low_forecast_daily_f")
            or extremes.get("nws_low_forecast_at_target_f"),
        ),
        "est_trough_time": _fmt_day_time(extremes["estimated_low_time"]),
        # One number, not a range. The +/-0.9F of whole-degree-Celsius slack
        # is real and still stated - it moves to the note underneath, exactly
        # as the hero already does it. A range here forced the reader to pick
        # between three answers on the one line that should just say what the
        # station recorded; the midpoint is the value that minimises expected
        # error, so it is the one to show.
        #
        # This is display only. Everything that has to be *correct* about the
        # reading's resolution still goes through the precise path:
        # settlement_band gates whether a bet may be resolved, and the Kalshi
        # bracket highlight reads observed_*_source, which is
        # "asos_remark_1min" only when the station's own un-quantised 1-minute
        # extreme is available (see asos_extremes).
        "observed_high": format_headline_reading(extremes["observed_high_so_far_f"]),
        # The bracket highlight is read against these, not against the hero,
        # so the "why is this a range" explanation belongs here.
        "observed_high_precision_note": precision_note(extremes["observed_high_so_far_f"]) or "",
        "observed_high_time": _fmt_time(extremes["observed_high_so_far_time"]),
        "observed_low": format_headline_reading(extremes["observed_low_so_far_f"]),
        "observed_low_precision_note": precision_note(extremes["observed_low_so_far_f"]) or "",
        "observed_low_time": _fmt_time(extremes["observed_low_so_far_time"]),
        "same_day_confidence_html": same_day_confidence_html,
        "tomorrow_high": f"{extremes['tomorrow_high_f']:.2f}",
        "tomorrow_confidence_pct": tomorrow_high_confidence_pct,
        "tomorrow_meta": (
            f"~{_fmt_time(extremes['tomorrow_high_time'])} · reference confidence {tomorrow_high_confidence_pct}%*"
            if extremes["tomorrow_high_time"] is not None
            else f"reference confidence {tomorrow_high_confidence_pct}%*"
        ),
        "tomorrow_hint": tomorrow_hint(extremes["tomorrow_high_source"], measured_conf["high"], measured_conf["n_high"]),
        "tomorrow_low": f"{extremes['tomorrow_low_f']:.2f}",
        "tomorrow_low_confidence_pct": tomorrow_low_confidence_pct,
        "tomorrow_low_meta": (
            f"~{_fmt_time(extremes['tomorrow_low_time'])} · reference confidence {tomorrow_low_confidence_pct}%*"
            if extremes["tomorrow_low_time"] is not None
            else f"reference confidence {tomorrow_low_confidence_pct}%*"
        ),
        "tomorrow_low_hint": tomorrow_hint(extremes["tomorrow_low_source"], measured_conf["low"], measured_conf["n_low"]),
        # Yesterday is settled, not estimated. Once its CLI report is out,
        # that whole degree is the value of record - render it plainly, with
        # no quantisation range and no +/-0.9F caption, because there is no
        # longer anything uncertain to describe. Only the pre-CLI window
        # still gets the stream treatment.
        "yesterday_high": format_settled_reading(
            extremes["yesterday_high_f"], extremes.get("yesterday_source")
        ),
        "yesterday_high_time": _fmt_time(extremes["yesterday_high_time"]) if extremes["yesterday_high_time"] is not None else "—",
        "yesterday_low": format_settled_reading(
            extremes["yesterday_low_f"], extremes.get("yesterday_source")
        ),
        "yesterday_low_time": _fmt_time(extremes["yesterday_low_time"]) if extremes["yesterday_low_time"] is not None else "—",
        "yesterday_source_note": yesterday_source_note(
            extremes.get("yesterday_source"),
            extremes.get("yesterday_cli_lag_hours"),
            extremes.get("yesterday_cli_overdue", False),
        ),
        "kalshi_high_ticker": kalshi_high["event_ticker"] if kalshi_high else "no open market",
        "kalshi_high_rows": build_kalshi_rows(
            kalshi_high["brackets"], extremes["estimated_high_f"],
            "Observed high" if extremes["high_status"] == "observed" else "Estimated high",
            # Provenance only applies to an already-observed extreme; a
            # forecast's error is its own, not the sensor's.
            observed_source=(
                extremes.get("observed_high_source")
                if extremes["high_status"] == "observed" else None
            ),
        ) if kalshi_high else (
            '<div class="hint">Temporarily unable to reach Kalshi for today\'s high market - try refreshing shortly.</div>'
            if kalshi_high_error else '<div class="hint">No open market for today\'s high yet.</div>'
        ),
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
            observed_source=(
                extremes.get("observed_low_source")
                if extremes["low_status"] == "tonight" else None
            ),
        ) if kalshi_low else (
            '<div class="hint">Temporarily unable to reach Kalshi for today\'s low market - try refreshing shortly.</div>'
            if kalshi_low_error else '<div class="hint">No open market for today\'s low yet.</div>'
        ),
        "kalshi_tomorrow_high_ticker": kalshi_tomorrow_high["event_ticker"] if kalshi_tomorrow_high else "no open market",
        "kalshi_tomorrow_high_rows": build_kalshi_rows(kalshi_tomorrow_high["brackets"], extremes["tomorrow_high_f"], "Estimated high") if kalshi_tomorrow_high else (
            '<div class="hint">Temporarily unable to reach Kalshi for tomorrow\'s high market - try refreshing shortly.</div>'
            if kalshi_tomorrow_high_error else '<div class="hint">Market not open yet.</div>'
        ),
        "kalshi_tomorrow_low_ticker": kalshi_tomorrow_low["event_ticker"] if kalshi_tomorrow_low else "no open market",
        "kalshi_tomorrow_low_rows": build_kalshi_rows(kalshi_tomorrow_low["brackets"], extremes["tomorrow_low_f"], "Estimated low") if kalshi_tomorrow_low else (
            '<div class="hint">Temporarily unable to reach Kalshi for tomorrow\'s low market - try refreshing shortly.</div>'
            if kalshi_tomorrow_low_error else '<div class="hint">Market not open yet.</div>'
        ),
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
        "weekly_days_with_data": weekly_perf["days_with_data"],
        "weekly_days_requested": weekly_perf["days_requested"],
        "next_day_table": build_next_day_table(_calib_summary),
        "weekly_high_table": build_weekly_performance_table(
            weekly_perf["rows"], "high", _night_before),
        "weekly_low_table": build_weekly_performance_table(
            weekly_perf["rows"], "low", _night_before),
        "monthly_label": now.strftime("%B %Y"),
        "monthly_low_sample_note": (
            f'<div class="hint">Only {monthly_perf["days_with_data"]} day(s) finalized so far this month - treat these as low-sample, not a stable average.</div>'
            if monthly_perf["low_sample"] else ""
        ),
        "monthly_high_stats": build_monthly_stats_rows(monthly_perf["high"]),
        "monthly_low_stats": build_monthly_stats_rows(monthly_perf["low"]),
        "weekly_strategy_comparison": build_strategy_comparison(weekly_perf["strategy_comparison"]),
        "monthly_strategy_comparison": build_strategy_comparison(monthly_perf["strategy_comparison"]),
        "weekly_paper_trading": build_paper_trading_rows(weekly_perf["paper_trading"]),
        "monthly_paper_trading": build_paper_trading_rows(monthly_perf["paper_trading"]),
        "weekly_low_strategy": build_paper_trading_rows(
            weekly_perf["low_strategy_comparison"], LOW_STRATEGY_KEYS, LOW_STRATEGY_LABELS
        ),
        "monthly_low_strategy": build_paper_trading_rows(
            monthly_perf["low_strategy_comparison"], LOW_STRATEGY_KEYS, LOW_STRATEGY_LABELS
        ),
        "fremont_current_temp": f"{fremont_current_temp:.1f}" if fremont_current_temp is not None else "—",
        "fremont_condition_text": fremont_condition_text,
        "fremont_obs_time": _fmt_time(fremont_obs_time) if fremont_obs_time is not None else "—",
        "fremont_wind_mph": f"{fremont_wind_mph:.1f}" if fremont_wind_mph is not None else "—",
        "fremont_forecast_high": f"{fremont_forecast_high:.1f}" if fremont_forecast_high is not None else "—",
        "fremont_forecast_high_time": _fmt_time(fremont_forecast_high_time) if fremont_forecast_high_time is not None else "—",
        "fremont_forecast_low": f"{fremont_forecast_low:.1f}" if fremont_forecast_low is not None else "—",
        "fremont_forecast_low_time": _fmt_time(fremont_forecast_low_time) if fremont_forecast_low_time is not None else "—",
        "sky_class": sky_class,
        "condition_text": condition_text,
        "hero_icon_svg": build_hero_icon_svg(cloud_pct, is_day),
        "obs_json_url": obs_json_url,
        "obhistory_url": obhistory_url,
        "forecast_url": forecast_url,
        "timeseries_url": timeseries_url,
    }

    with open("dashboard_template.html", "r") as f:
        template = f.read()

    for key, value in ctx.items():
        template = template.replace("{{" + key + "}}", str(value))

    check_panel_layout(template)

    default_out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "index.html")
    out_path = os.environ.get("DASHBOARD_OUTPUT_PATH", default_out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(template)

    print(f"Wrote {out_path}")
    print(json.dumps({k: v for k, v in ctx.items() if k not in ("sparkline_svg", "data_json")}, indent=2))


if __name__ == "__main__":
    main()
