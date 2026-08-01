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
from calibration_log import record_snapshot, next_day_confidence_pct, summarize, MIN_NEXT_DAY_SAMPLES
from daily_performance import (
    finalize_pending_days,
    reconcile_stream_fallback_actuals,
    reconcile_peak_time_windows,
    weekly_table,
    monthly_rollup,
    LOW_SAMPLE_THRESHOLD,
)
from peak_alerts import get_or_lock_daily_targets
from paper_trading import get_or_lock_2hr_targets, LEAD_TIME_HINTS

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
<svg viewBox="0 0 {width} {height}" class="sparkline" preserveAspectRatio="none" role="img" aria-label="Temperature trend, last {SPARKLINE_HOURS} hours and projected estimate">
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


def build_weekly_performance_table(rows, side):
    """One row per trailing day for a single side (high/low), all spec'd
    fields - wrapped in a horizontally-scrolling container by the
    template since there are too many columns for a phone-width card."""
    header = (
        "<tr><th>Date</th><th>Predicted</th><th>1h before pred.</th><th>NWS 1h before</th>"
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
            body.append(f'<tr><td>{date_label}</td><td colspan="9" class="perf-nodata">no data</td></tr>')
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
        body.append(
            "<tr>"
            f"<td>{date_label}{flag}</td>"
            f"<td>{predicted}</td>"
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
        rows.append(row("Model in Kalshi's top bracket", f"{stats['model_in_kalshi_bracket_rate'] * 100:.0f}% ({stats['n_hit_samples']} days)"))
    if stats["n_temp_error_samples"] == 0:
        rows.append('<div class="hint">No model predictions logged yet this month - calibration_log.py only started capturing pre-peak predictions recently.</div>')
    return "\n".join(rows)


LEAD_TIME_LABELS = {"1hr": "1hr before peak", "2hr": "2hr before peak"}
LOW_STRATEGY_KEYS = ("edge", "unconditional")
LOW_STRATEGY_LABELS = {
    "edge": "Low - edge-triggered (1hr)",
    "unconditional": "Low - unconditional (1hr)",
}


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
        "next_day_table": build_next_day_table(summarize()),
        "weekly_high_table": build_weekly_performance_table(weekly_perf["rows"], "high"),
        "weekly_low_table": build_weekly_performance_table(weekly_perf["rows"], "low"),
        "monthly_label": now.strftime("%B %Y"),
        "monthly_low_sample_note": (
            f'<div class="hint">Only {monthly_perf["days_with_data"]} day(s) finalized so far this month - treat these as low-sample, not a stable average.</div>'
            if monthly_perf["low_sample"] else ""
        ),
        "monthly_high_stats": build_monthly_stats_rows(monthly_perf["high"]),
        "monthly_low_stats": build_monthly_stats_rows(monthly_perf["low"]),
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
        "obs_json_url": obs_json_url,
        "obhistory_url": obhistory_url,
        "forecast_url": forecast_url,
        "timeseries_url": timeseries_url,
    }

    with open("dashboard_template.html", "r") as f:
        template = f.read()

    for key, value in ctx.items():
        template = template.replace("{{" + key + "}}", str(value))

    default_out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "index.html")
    out_path = os.environ.get("DASHBOARD_OUTPUT_PATH", default_out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(template)

    print(f"Wrote {out_path}")
    print(json.dumps({k: v for k, v in ctx.items() if k not in ("sparkline_svg", "data_json")}, indent=2))


if __name__ == "__main__":
    main()
