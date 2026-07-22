"""
Sends a text ~1 hour before the model's estimated daily high/low, so
there's a heads-up close to the actual turning point rather than a
generic morning forecast.

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
no domain-verification requirement, small per-message cost.

Trigger logic (see get_or_lock_daily_targets / check_and_send):
  - Once per day per extreme (high, low), the target alert time
    (predicted_peak_time - 1 hour) is locked in from whatever the
    model's estimate is at the moment of locking - it does NOT keep
    re-chasing the target if the model's peak-time estimate drifts
    later in the day. That avoids both double-sends and never firing
    because the target kept moving.
  - The actual send re-checks the model's CURRENT estimate at fire time
    (not the value cached at lock time) - the target *time* is locked,
    the *content* is always fresh.
  - "Already sent today for this extreme" is persisted to
    peak_alerts_state.json (mutable per-day state, not an append log -
    unlike calibration_log.jsonl/daily_performance.jsonl, this needs to
    flip a "sent" flag after firing), so a process restart or a
    duplicate check within the tolerance window can't double-send.

Scheduling the actual one-shot fire (i.e. "wake up and call check_and_send
at exactly this timestamp") isn't something this plain Python module can
do by itself - only the agent session can create a Routine. See
build_dashboard.py's integration and the hourly Routine's prompt for how
the lock-in step signals that a new one-shot fire needs to be scheduled.

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


def get_or_lock_daily_targets(station_id=STATION):
    """
    Once per extreme, lock in target_alert_time = predicted peak time -
    1 hour, using whatever the model's estimate is right now. Each side
    is filed under the calendar date its OWN prediction is actually for
    (predicted_peak_time.date()), not the date lock-in happens to run on -
    this matters for "low", which flips to forecasting the *next* day's
    dawn (status "tonight") once today's own low has passed. Filing it
    under the date it's locked would double-lock it again from that next
    day's own morning, when the model naturally starts a fresh "today"
    prediction for the same physical low.

    Returns {"newly_locked": [(date_str, side), ...], "state": {...}} -
    newly_locked entries are what the caller needs to schedule a one-shot
    fire for (see module docstring); "state" holds only the touched
    dates' records, formatted for the CLI to print.
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

        target_alert_time = predicted_peak_time - timedelta(hours=LEAD_HOURS)
        day_state[side] = {
            "predicted_peak_time_at_lock": predicted_peak_time.isoformat(),
            "target_alert_time": target_alert_time.isoformat(),
            "locked_at": now.isoformat(),
            "sent": False,
            "sent_at": None,
            "message": None,
            # If the lead-time window is already behind us by the moment
            # we locked in (e.g. lock-in ran late), there's nothing
            # meaningful left to schedule - don't invent a past fire time.
            "skipped_missed_window": target_alert_time <= now,
        }
        newly_locked.append((target_date_str, side))

    _save_state(state)
    return {
        "newly_locked": newly_locked,
        "state": {d: state[d] for d in sorted(touched_dates)},
    }


def _format_message(station_id, side, extremes):
    est = estimate_temp(station_id, hours_ahead=3)
    now = extremes["as_of"]

    if side == "high":
        peak_temp, peak_time = extremes["estimated_high_f"], extremes["estimated_high_time"]
        label = "high"
    else:
        peak_temp, peak_time = extremes["estimated_low_f"], extremes["estimated_low_time"]
        label = "low"

    lo, hi = est["estimated_range_f"]
    damping_pct = round(est["diurnal_damping"] * est["sky_wind_damping"] * 100)

    return (
        f"{station_id}: Est. {label} {peak_temp:.0f}°F ~{peak_time.strftime('%-I:%M%p').lower()} "
        f"(currently {est['current_temp_f']:.0f}°F, {now.strftime('%-I:%M%p').lower()}).\n"
        f"Trend: {est['raw_trend_f_per_hr']:+.1f}°/hr, damped {damping_pct}%. Band: {lo:.0f}-{hi:.0f}°F."
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


def check_and_send(station_id=STATION, side="high", tolerance_minutes=TOLERANCE_MINUTES):
    """
    Re-checks the model's CURRENT estimate (not the value cached at lock
    time) and sends the text if we're within tolerance_minutes of the
    locked target and haven't already sent for this extreme.

    Looks up the locked record by scanning yesterday/today/tomorrow
    rather than assuming an exact date-key match, and picks whichever
    candidate's target_alert_time is closest to now - robust to "low"
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
        key=lambda ds: abs((now - datetime.fromisoformat(ds[1]["target_alert_time"])).total_seconds()),
    )

    if side_state["sent"]:
        return {"sent": False, "reason": "already sent for this extreme"}
    if side_state.get("skipped_missed_window"):
        return {"sent": False, "reason": "target window was already past at lock time"}

    target_time = datetime.fromisoformat(side_state["target_alert_time"])
    delta_minutes = abs((now - target_time).total_seconds()) / 60
    if delta_minutes > tolerance_minutes:
        return {"sent": False, "reason": f"outside tolerance window ({delta_minutes:.1f} min from target)"}

    message = _format_message(station_id, side, extremes)
    send_text(message)

    side_state["sent"] = True
    side_state["sent_at"] = now.isoformat()
    side_state["message"] = message
    state[date_str][side] = side_state
    _save_state(state)
    return {"sent": True, "message": message}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "lock":
        result = get_or_lock_daily_targets(STATION)
        print(json.dumps(result, indent=2))
        for date_str, side in result["newly_locked"]:
            side_state = result["state"][date_str][side]
            if not side_state["skipped_missed_window"]:
                print(f"ALERT_SCHEDULE_NEEDED side={side} date={date_str} target_alert_time={side_state['target_alert_time']}")
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
