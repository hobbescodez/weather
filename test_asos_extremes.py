"""
Guards the METAR remark parser, which is now the source of the observed
daily extremes that Kalshi brackets are read against.

Worth guarding tightly because a silent failure here is invisible: the
refinement falls back to the 5-minute stream, which still produces a
plausible-looking number - just a quantised one, off by up to 0.9F and
capable of pointing at the wrong bracket. That is exactly the failure this
module was written to end, so it must not come back by regression.

Offline: every METAR here is a real KSEA observation captured 2026-07-28/29.

    python3 test_asos_extremes.py
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from asos_extremes import (
    parse_remark_extremes,
    collect_remark_extremes,
    refine_extreme,
    AGREEMENT_TOLERANCE_F,
)

TZ = ZoneInfo("America/Los_Angeles")
passed = failed = 0


def check(label, got, want):
    global passed, failed
    if got == want:
        passed += 1
        print(f"  [PASS] {label} - {got!r}")
    else:
        failed += 1
        print(f"  [FAIL] {label} - got {got!r}, want {want!r}")


# The real 12Z report whose 6-hour minimum settled 2026-07-29's low market.
METAR_12Z = ("KSEA 291153Z 01005KT 10SM FEW035 SCT045 16/11 A3005 "
             "RMK AO2 SLP176 T01560106 10161 20144 56004 $")
T_12Z = datetime(2026, 7, 29, 4, 53, tzinfo=TZ)


def test_parses_six_hour_groups():
    print("\ntest_parses_six_hour_groups")
    got = parse_remark_extremes(METAR_12Z, T_12Z)
    lows = [c for c in got if c["side"] == "low"]
    highs = [c for c in got if c["side"] == "high"]
    check("20144 -> 14.4C = 57.92F", round(lows[0]["temp_f"], 2), 57.92)
    check("10161 -> 16.1C = 60.98F", round(highs[0]["temp_f"], 2), 60.98)
    check("window ends at the report time", lows[0]["window_end"], T_12Z)
    check("window is six hours", lows[0]["window_start"], T_12Z - timedelta(hours=6))


def test_ignores_non_temperature_groups():
    print("\ntest_ignores_non_temperature_groups")
    # 56004 is a pressure tendency and T01560106 the instantaneous temp -
    # neither is a max/min, and a loose regex would happily read both.
    got = parse_remark_extremes(METAR_12Z, T_12Z)
    check("exactly one max and one min parsed", len(got), 2)
    # SLP176 must not be mistaken for anything; A3005 likewise.
    check("no bogus values", sorted(round(c["temp_f"], 1) for c in got), [57.9, 61.0])


def test_no_remarks_is_not_a_crash():
    print("\ntest_no_remarks_is_not_a_crash")
    check("empty METAR", parse_remark_extremes("", T_12Z), [])
    check("None METAR", parse_remark_extremes(None, T_12Z), [])
    check("METAR with no RMK", parse_remark_extremes("KSEA 291153Z 16/11 A3005", T_12Z), [])


def test_refines_the_quantised_stream_value():
    print("\ntest_refines_the_quantised_stream_value")
    cands = parse_remark_extremes(METAR_12Z, T_12Z)
    stream_time = datetime(2026, 7, 29, 0, 25, tzinfo=TZ)
    value, source = refine_extreme("low", 57.20, stream_time, cands)
    check("uses the 1-minute figure", value, 57.92)
    check("reports its provenance", source, "asos_remark_1min")
    # This is the whole point: the quantised value and the real one settle
    # into DIFFERENT Kalshi brackets.
    check("quantised value would settle 57", round(57.20), 57)
    check("refined value settles 58", round(value), 58)


def test_declines_when_the_extreme_is_outside_the_window():
    print("\ntest_declines_when_the_extreme_is_outside_the_window")
    cands = parse_remark_extremes(METAR_12Z, T_12Z)
    # An afternoon low can't be described by a report covering 23:00-05:00.
    afternoon = datetime(2026, 7, 29, 14, 5, tzinfo=TZ)
    value, source = refine_extreme("low", 57.20, afternoon, cands)
    check("keeps the stream value", value, 57.20)
    check("says so", source, "observation_stream")


def test_declines_when_the_values_disagree():
    print("\ntest_declines_when_the_values_disagree")
    cands = parse_remark_extremes(METAR_12Z, T_12Z)
    stream_time = datetime(2026, 7, 29, 0, 25, tzinfo=TZ)
    # A stream low far from the group's minimum means they are not the same
    # event - the group must not overwrite it.
    far = 57.92 - (AGREEMENT_TOLERANCE_F + 1.0)
    value, source = refine_extreme("low", far, stream_time, cands)
    check("keeps the stream value", value, far)
    check("says so", source, "observation_stream")


def test_prefers_the_narrower_window():
    print("\ntest_prefers_the_narrower_window")
    # A 24-hour group and a 6-hour group both covering the same extreme:
    # the 6-hour one localises it better and should win.
    daily = ("KSEA 290700Z 00000KT 10SM CLR 15/10 A3005 "
             "RMK AO2 T01500100 401940144")
    t_daily = datetime(2026, 7, 29, 0, 0, tzinfo=TZ)
    cands = parse_remark_extremes(METAR_12Z, T_12Z) + parse_remark_extremes(daily, t_daily)
    spans = sorted({c["span_hours"] for c in cands})
    check("both window sizes present", spans, [6, 24])
    stream_time = datetime(2026, 7, 29, 0, 25, tzinfo=TZ)
    # Only the 6-hour window contains 00:25 on the 29th (the 24-hour one
    # ends at midnight), so this also checks window filtering.
    value, source = refine_extreme("low", 57.20, stream_time, cands)
    check("picks the 6-hour figure", value, 57.92)
    check("provenance", source, "asos_remark_1min")


def test_collect_walks_a_series():
    print("\ntest_collect_walks_a_series")
    times = [T_12Z - timedelta(minutes=5), T_12Z, T_12Z + timedelta(minutes=5)]
    raws = [None, METAR_12Z, "KSEA 291158Z 16/11 A3005"]
    got = collect_remark_extremes(times, raws)
    check("only the report with groups contributes", len(got), 2)


for fn in (
    test_parses_six_hour_groups,
    test_ignores_non_temperature_groups,
    test_no_remarks_is_not_a_crash,
    test_refines_the_quantised_stream_value,
    test_declines_when_the_extreme_is_outside_the_window,
    test_declines_when_the_values_disagree,
    test_prefers_the_narrower_window,
    test_collect_walks_a_series,
):
    fn()

print(f"\n{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
