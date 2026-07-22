"""
Sends a text near the model's estimated daily high/low, so there's a
heads-up close to the actual turning point rather than a generic
morning forecast.

Delivery: Twilio's SMS REST API (https://api.twilio.com). Two earlier
approaches were tried and ruled out first:
  - Raw SMTP (smtplib, ports 587/465) does NOT work from this sandbox -
    confirmed by direct connection tests: only HTTPS egress is proxied.
  - Resend's email API (HTTPS, so it *can* reach this sandbox) was tried
    next, sending to the phone's carrier email-to-SMS gateway - but
    Resend's unverified/free tier only allows sending to the account's
    own registered email address, not to arbitrary third-party
    recipients like a carrier gateway, without first verifying a domain.
Twilio sends directly to the phone number over SMS - no email gateway,
no domain-verification requirement, small per-message cost. (As of this
writing, sending is still blocked on Twilio's own side pending Trust
Hub / A2P compliance registration on the account - the account owner
needs to finish that in the Twilio console; nothing in this module can
complete it. Once that clears, sends will start working with no code
changes needed here.)

Trigger logic (see get_or_lock_daily_targets / check_and_send):
  - A fixed "1 hour before peak" offset can land exactly when the
    model's own trend-confidence is lowest (right at a turning point),
    which is the worst time to text a number that's likely to move. So
    instead of a single target instant, each extreme locks a WINDOW:
      window_start  = predicted_peak_time - LEAD_HOURS (1 hour) - the
                       earliest we'd ever send
      hard_cutoff   = predicted_peak_time - CUTOFF_MINUTES (20 min) -
                       the latest we'd wait before sending regardless
    Within that window, the text goes out at the first checkpoint where
    trend_confidence (the same diurnal_damping * sky_wind_damping the
    dashboard shows) clears CONFIDENCE_THRESHOLD_PCT, OR at hard_cutoff
    if confidence never clears the bar that day - whichever comes
    first. That way a low-confidence guess doesn't go out too early,
    and the text still always goes out by hard_cutoff even on a day
    the model stays unsure the whole window.
  - The window is locked once per day per extreme from whatever the
    model's estimate is at lock time - it does NOT keep re-chasing the
    target if the model's peak-time estimate drifts later in the day.
    That avoids both double-sends and never firing because the target
    kept moving.
  - Checking the window requires re-evaluating confidence at more than
    one instant, so lock-in also produces a handful of checkpoint
    timestamps spanning the window (10 minutes apart); each one asks
    the same question ("send now?") and the first "yes" wins - see
    CHECK_INTERVAL_MINUTES. Re-checking is free: check_and_send is
    idempotent via the "sent" flag, so a checkpoint firing after the
    text already went out is a silent no-op.
  - The actual send re-checks the model's CURRENT estimate at fire time
    (not the value cached at lock time) - the window is locked, the
    *content* is always fresh.
  - "Already sent today for this extreme" is persisted to
    peak_alerts_state.json (mutable per-day state, not an append log -
    unlike calibration_log.jsonl/daily_performance.jsonl, this needs to
    flip a "sent" flag after firing), so a process restart or a
    duplicate check within the tolerance window can't double-send.

Scheduling the actual one-shot fires (i.e. "wake up and call
check_and_send at exactly this timestamp") isn't something this plain
Python module can do by itself - only the agent session can create a
Routine. See build_dashboard.py's integration and the hourly Routine's
prompt for how the lock-in step signals that new one-shot fires need to
be scheduled (one per checkpoint - the marker format is unchanged from
the single-fire version, just emitted multiple times per side).

Config (never commit real values - see .env.example):
    ALERT_PHONE_NUMBER    10-digit phone number to text, digits only
    TWILIO_ACCOUNT_SID    from the Twilio console
    TWILIO_AUTH_TOKEN     from the Twilio console
    TWILIO_FROM_NUMBER    the Twilio phone number sending the text
                          (E.164, e.g. +15551234567)

CLI:
    python3 peak_alerts.py lock            # lock in today's targets
    python3 peak_alerts.py check high       # fire the high-alert check now
    python3 peak_alerts.py check low
    python3 peak_alerts.py status           # show today's state
"""

import json
import os
import sys
from datetime import datetime, timedelta

import requests

from weather_estimator import estimate_daily_extremes, estimate_temp, get_observation_history

STATE_PATH = os.path.join(os.path.dirname(__file__), "peak_alerts_state.json")
STATION = "KSEA"
LEAD_HOURS = 1
CUTOFF_MINUTES = 20
CHECK_INTERVAL_MINUTES = 10
CONFIDENCE_THRESHOLD_PCT = 58
TOLERANCE_MINUTES = 5


def _load_env():
    """Tiny .env loader (KEY=VALUE lines) - no external dependency."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()

PHONE_NUMBER = os.environ.get("ALERT_PHONE_NUMBER", "")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER", "")


def _load_state():
    if not os.path.exists(STATE_PATH):
        return {}
    with open(STATE_PATH) as f:
        return json.load(f)


def _save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _checkpoints_between(window_start, hard_cutoff, now):
    """Timestamps CHECK_INTERVAL_MINUTES apart spanning [window_start,
    hard_cutoff], excluding any that are already in the past (no point
    scheduling a fire for a moment that's already gone, e.g. when
    lock-in itself runs a little late)."""
    checkpoints = []
    t = window_start
    while t < hard_cutoff:
        checkpoints.append(t)
        t += timedelta(minutes=CHECK_INTERVAL_MINUTES)
    checkpoints.append(hard_cutoff)
    return [t for t in checkpoints if t > now]


def get_or_lock_daily_targets(station_id=STATION):
    """
    Once per extreme, lock in an alert WINDOW - window_start (predicted
    peak time - LEAD_HOURS) through hard_cutoff (predicted peak time -
    CUTOFF_MINUTES) - using whatever the model's estimate is right now.
    Each side is filed under the calendar date its OWN prediction is
    actually for (predicted_peak_time.date()), not the date lock-in
    happens to run on - this matters for "low", which flips to
    forecasting the *next* day's dawn (status "tonight") once today's
    own low has passed. Filing it under the date it's locked would
    double-lock it again from that next day's own morning, when the
    model naturally starts a fresh "today" prediction for the same
    physical low.

    Returns {"newly_locked": [(date_str, side), ...], "state": {...}} -
    newly_locked entries are what the caller needs to schedule one-shot
    fires for, one per checkpoint in that side's state entry (see module
    docstring); "state" holds only the touched dates' records, formatted
    for the CLI to print.
    """
    extremes = estimate_daily_extremes(station_id)
    now = extremes["as_of"]

    state = _load_state()
    newly_locked = []
    touched_dates = set()

    for side, time_key in [("high", "estimated_high_time"), ("low", "estimated_low_time")]:
        predicted_peak_time = extremes[time_key]
        target_date_str = predicted_peak_time.date().isoformat()
        touched_dates.add(target_date_str)
        day_state = state.setdefault(target_date_str, {})
        if side in day_state:
            continue

        window_start = predicted_peak_time - timedelta(hours=LEAD_HOURS)
        hard_cutoff = predicted_peak_time - timedelta(minutes=CUTOFF_MINUTES)
        day_state[side] = {
            "predicted_peak_time_at_lock": predicted_peak_time.isoformat(),
            "window_start": window_start.isoformat(),
            "hard_cutoff": hard_cutoff.isoformat(),
            "checkpoints": [t.isoformat() for t in _checkpoints_between(window_start, hard_cutoff, now)],
            "locked_at": now.isoformat(),
            "sent": False,
            "sent_at": None,
            "message": None,
            "confidence_at_send": None,
            # If the whole window is already behind us by the moment we
            # locked in (e.g. lock-in ran late), there's nothing
            # meaningful left to schedule - don't invent a past fire time.
            "skipped_missed_window": hard_cutoff <= now,
        }
        newly_locked.append((target_date_str, side))

    _save_state(state)
    return {
        "newly_locked": newly_locked,
        "state": {d: state[d] for d in sorted(touched_dates)},
    }


def _trend_confidence_pct(est):
    """Same formula build_dashboard.py uses for the dashboard's own
    "confidence" figure - proximity to sunrise/peak turning points and
    current cloud/wind - so the text and the dashboard never disagree
    about how sure the model is."""
    return round(est["diurnal_damping"] * est["sky_wind_damping"] * 100)


def _format_message(station_id, side, extremes, est, trend_confidence_pct):
    now = extremes["as_of"]

    if side == "high":
        peak_temp, peak_time = extremes["estimated_high_f"], extremes["estimated_high_time"]
        label = "high"
    else:
        peak_temp, peak_time = extremes["estimated_low_f"], extremes["estimated_low_time"]
        label = "low"

    lo, hi = est["estimated_range_f"]

    return (
        f"{station_id}: Est. {label} {peak_temp:.0f}°F ~{peak_time.strftime('%-I:%M%p').lower()} "
        f"(currently {est['current_temp_f']:.0f}°F, {now.strftime('%-I:%M%p').lower()}).\n"
        f"Trend confidence: {trend_confidence_pct}%. Band: {lo:.0f}-{hi:.0f}°F."
    )


def send_text(message):
    """POSTs to Twilio's SMS REST API - see module docstring for why
    this isn't smtplib or an email-to-SMS gateway."""
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_FROM_NUMBER:
        raise RuntimeError("TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_FROM_NUMBER not set - fill in .env before sending")
    if not PHONE_NUMBER:
        raise RuntimeError("ALERT_PHONE_NUMBER not set in .env")

    to_number = f"+1{PHONE_NUMBER}" if not PHONE_NUMBER.startswith("+") else PHONE_NUMBER
    r = requests.post(
        f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json",
        auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
        data={"To": to_number, "From": TWILIO_FROM_NUMBER, "Body": message},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def _legacy_or(side_state, key, fallback_key="target_alert_time"):
    """peak_alerts_state.json may still hold entries locked before the
    window/confidence-gating refactor (a single target_alert_time
    instead of window_start/hard_cutoff). Fall back to that single
    instant for both bounds so an in-flight legacy entry keeps behaving
    exactly like it did before - fire once, right at that instant -
    rather than erroring out or silently changing behavior mid-flight."""
    return side_state.get(key, side_state.get(fallback_key))


def check_and_send(station_id=STATION, side="high", tolerance_minutes=TOLERANCE_MINUTES):
    """
    Re-checks the model's CURRENT estimate (not the value cached at lock
    time) and sends the text once we're inside the locked alert window
    AND either trend_confidence clears CONFIDENCE_THRESHOLD_PCT or
    we've reached hard_cutoff - whichever comes first - and haven't
    already sent for this extreme. Safe to call repeatedly (e.g. once
    per checkpoint): every call after the first "yes" is a no-op because
    of the "sent" flag.

    Looks up the locked record by scanning yesterday/today/tomorrow
    rather than assuming an exact date-key match, and picks whichever
    candidate's hard_cutoff is closest to now - robust to "low"
    sometimes being filed under the next calendar date relative to when
    it was locked (see get_or_lock_daily_targets).
    """
    extremes = estimate_daily_extremes(station_id)
    now = extremes["as_of"]

    state = _load_state()
    candidates = []
    for offset in (-1, 0, 1):
        d = (now.date() + timedelta(days=offset)).isoformat()
        side_state = state.get(d, {}).get(side)
        if side_state is not None:
            candidates.append((d, side_state))

    if not candidates:
        return {"sent": False, "reason": "no locked target found nearby - run 'lock' first"}

    date_str, side_state = min(
        candidates,
        key=lambda ds: abs((now - datetime.fromisoformat(_legacy_or(ds[1], "hard_cutoff"))).total_seconds()),
    )

    if side_state["sent"]:
        return {"sent": False, "reason": "already sent for this extreme"}
    if side_state.get("skipped_missed_window"):
        return {"sent": False, "reason": "alert window was already past at lock time"}

    window_start = datetime.fromisoformat(_legacy_or(side_state, "window_start"))
    hard_cutoff = datetime.fromisoformat(_legacy_or(side_state, "hard_cutoff"))

    if now < window_start - timedelta(minutes=tolerance_minutes):
        early_minutes = (window_start - now).total_seconds() / 60
        return {"sent": False, "reason": f"before alert window opens ({early_minutes:.0f} min early)"}

    est = estimate_temp(station_id, hours_ahead=3)
    trend_confidence_pct = _trend_confidence_pct(est)
    past_cutoff = now >= hard_cutoff - timedelta(minutes=tolerance_minutes)
    confident_enough = trend_confidence_pct >= CONFIDENCE_THRESHOLD_PCT

    if not (confident_enough or past_cutoff):
        cutoff_minutes = (hard_cutoff - now).total_seconds() / 60
        return {
            "sent": False,
            "reason": (
                f"trend confidence {trend_confidence_pct}% below {CONFIDENCE_THRESHOLD_PCT}% threshold "
                f"and {cutoff_minutes:.0f} min before hard cutoff - waiting for next checkpoint"
            ),
        }

    message = _format_message(station_id, side, extremes, est, trend_confidence_pct)
    send_text(message)

    side_state["sent"] = True
    side_state["sent_at"] = now.isoformat()
    side_state["message"] = message
    side_state["confidence_at_send"] = trend_confidence_pct
    state[date_str][side] = side_state
    _save_state(state)
    return {"sent": True, "message": message, "trend_confidence_pct": trend_confidence_pct}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "lock":
        result = get_or_lock_daily_targets(STATION)
        print(json.dumps(result, indent=2))
        for date_str, side in result["newly_locked"]:
            side_state = result["state"][date_str][side]
            if not side_state["skipped_missed_window"]:
                for checkpoint in side_state["checkpoints"]:
                    print(f"ALERT_SCHEDULE_NEEDED side={side} date={date_str} target_alert_time={checkpoint}")
    elif cmd == "check":
        side = sys.argv[2] if len(sys.argv) > 2 else "high"
        print(json.dumps(check_and_send(STATION, side), indent=2))
    elif cmd == "status":
        state = _load_state()
        # Station-local date, not the container's system clock (which
        # runs UTC and can already be a day ahead of Seattle evenings -
        # same class of bug fixed earlier in build_dashboard.py). Shows
        # yesterday/today/tomorrow since "low" can be filed a day ahead
        # of when it was locked.
        today = estimate_daily_extremes(STATION)["as_of"].date()
        window = {
            (today + timedelta(days=offset)).isoformat(): state[(today + timedelta(days=offset)).isoformat()]
            for offset in (-1, 0, 1)
            if (today + timedelta(days=offset)).isoformat() in state
        }
        print(json.dumps(window, indent=2))
    else:
        print(f"Unknown command: {cmd}. Use lock | check <high|low> | status.")
