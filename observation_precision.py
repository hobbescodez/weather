"""
How precise an ASOS temperature reading actually is, and what that means
for deciding which Kalshi bracket a day landed in.

The problem this exists to stop: a reading like 60.80F looks precise to
two decimals and is nothing of the sort. It is exactly 16.0C. 91% of
KSEA's observations report a WHOLE degree Celsius - only the :53 hourly
METARs carry tenths - so that 60.80 is a rounded 16C, and the real
temperature was anywhere in [15.5, 16.5)C = [59.9, 61.7]F. Nearly two
degrees F of slack, presented as if it were hundredths.

That matters because Kalshi settles on the whole-degree integer in NWS's
CLI report (see kalshi.bracket_contains), and 1.8F of slack routinely
straddles a bracket boundary. On 2026-07-28 the observed low read 60.80,
which rounds to 61 and highlighted "61 or above" - while the market held
86% on "59 to 60". The market was right; the observation was quantised.

Two distinct error sources, and they behave differently:

  1. Quantisation, symmetric. A whole-C report of 16 means the true
     instantaneous value was within +/-0.5C = +/-0.9F. A tenths-C report
     is good to about +/-0.05C = +/-0.09F.

  2. Sampling, ONE-SIDED and different per side. ASOS derives the
     official daily max/min from continuous 1-minute data; we only see
     discrete observations. A discrete sample of a continuous MINIMUM can
     only miss the dip, never invent one below it - so the observed low
     is an upper bound on the true low. Symmetrically the observed high
     is a lower bound on the true high.

Measured against 5 days of CLI actuals, stream minus CLI:

    low   n=5  mean +0.26F  range -0.02 .. +1.08   (always >= CLI)
    high  n=5  mean -0.16F  range -0.80 .. +0.40   (mixed)

The low side is cleanly one-sided, exactly as the sampling argument
predicts. The high side comes out mixed only because symmetric
quantisation there is large enough to swamp the one-sided sampling
effect - the error MAGNITUDE is comparable (max |diff| 1.08 low, 0.80
high). So this is not a low-side quirk; it applies to both.

Deliberately NOT done here: correcting the stream value by the mean
offset. A +0.26F "fix" would be a point estimate dressed up as a
measurement - the same false precision in a new coat - and it would be
fitted to five days. Representing the honest interval and refusing to
decide when the interval spans a boundary is the correction.
"""

# +/-0.5C, the half-width of a whole-degree-Celsius report.
QUANTISED_HALF_WIDTH_F = 0.9
# +/-0.05C, for the :53 METARs that carry tenths.
TENTHS_HALF_WIDTH_F = 0.1
# One-sided allowance for a discrete sample missing a continuous
# extreme, applied only in the direction sampling can actually err:
# downward for a low, upward for a high.
#
# It has to scale with how far apart the observations are, because that
# is exactly what determines how much of the curve is invisible. The
# first version of this used a flat 1.1F - the largest disagreement then
# on record (2026-07-26 low, +1.08F, on 5-minute data) - and it failed
# validation on 2026-07-22, where the sub-hourly feed was missing
# entirely, only 26 hourly observations existed for the whole day, and
# the stream low missed CLI by 1.92F. A day sampled every 5 minutes and
# a day sampled every 60 are not equally trustworthy and can't share a
# constant.
SAMPLING_ALLOWANCE_BASE_F = 1.2      # dense sampling, gaps up to DENSE_GAP_MINUTES
SAMPLING_ALLOWANCE_PER_MIN_F = 0.035  # widening per extra minute of gap
DENSE_GAP_MINUTES = 10.0
DEFAULT_GAP_MINUTES = 60.0  # assume the worst when coverage is unknown

_C_EPSILON = 0.02


def celsius_of(temp_f):
    return (temp_f - 32.0) * 5.0 / 9.0


def is_whole_celsius(temp_f):
    """Whether this reading came in as a whole degree Celsius, and so
    carries the full +/-0.5C of quantisation slack."""
    c = celsius_of(temp_f)
    return abs(c - round(c)) < _C_EPSILON


def reading_half_width_f(temp_f):
    """Symmetric uncertainty on a single reading, from quantisation
    alone."""
    return QUANTISED_HALF_WIDTH_F if is_whole_celsius(temp_f) else TENTHS_HALF_WIDTH_F


def sampling_allowance_f(max_gap_minutes=None):
    """How much of a continuous extreme a discrete sample can hide,
    given the widest gap between that day's observations."""
    gap = DEFAULT_GAP_MINUTES if max_gap_minutes is None else max_gap_minutes
    extra = max(0.0, gap - DENSE_GAP_MINUTES)
    return SAMPLING_ALLOWANCE_BASE_F + SAMPLING_ALLOWANCE_PER_MIN_F * extra


def settlement_band(side, stream_temp_f, max_gap_minutes=None):
    """
    Range the true CLI settlement temperature plausibly lies in, given
    only an observation-stream extreme.

    Combines symmetric quantisation with the gap-scaled one-sided
    sampling allowance, in whichever direction sampling can err for this
    side: the observed low can only be too high, the observed high only
    too low. Returns (lo, hi) in F.

    max_gap_minutes is the widest interval between that day's
    observations. Omit it and the band assumes hourly-only coverage,
    which is the conservative reading - better to decline to settle than
    to settle on a day whose coverage we never checked.

    Validated against every available side-day of CLI actuals, including
    the sparse-coverage day that broke the flat-constant version.
    Deliberately generous: this gates whether we are willing to settle a
    bet, so a band that is too wide costs a delay while one that is too
    narrow costs a wrong resolution.
    """
    if stream_temp_f is None:
        return None
    q = reading_half_width_f(stream_temp_f)
    s = sampling_allowance_f(max_gap_minutes)
    if side == "low":
        return (stream_temp_f - q - s, stream_temp_f + q)
    return (stream_temp_f - q, stream_temp_f + q + s)


def format_reading(temp_f, decimals=1, unit="°F"):
    """Render a reading at its real resolution.

    A whole-C reading becomes an honest whole-degree-F range (60.80 ->
    "~60-62°F") rather than a fake two-decimal point value. A tenths
    reading is genuinely good to about a tenth and is shown as one.
    """
    if temp_f is None:
        return "—"
    if not is_whole_celsius(temp_f):
        return f"{temp_f:.{decimals}f}{unit}"
    lo = round(temp_f - QUANTISED_HALF_WIDTH_F)
    hi = round(temp_f + QUANTISED_HALF_WIDTH_F)
    if lo == hi:
        return f"{lo}{unit}"
    return f"≈{lo}–{hi}{unit}"


def precision_note(temp_f):
    """Short human explanation, or None when the reading is precise."""
    if temp_f is None or not is_whole_celsius(temp_f):
        return None
    return (
        f"station reported {celsius_of(temp_f):.0f}°C (whole degree), "
        f"so this is ±{QUANTISED_HALF_WIDTH_F:.1f}°F"
    )
