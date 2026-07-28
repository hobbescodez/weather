# Pre-tail-fix paper-trading ledger (archived 2026-07-27)

These files are the complete paper-trading record from before the
bracket-selection fix. They are kept for auditability and are **not**
read by any code — `daily_performance.py`'s win-rate, P&L, and
strategy-comparison stats see only the live files, which were reset to
empty at the same time.

## Why it was retired

Selection ranked brackets on `model_probability − market_price`, using
each bracket's raw probability under the calibrated Normal. An
open-ended `"X or above"` / `"X or below"` bracket integrates from its
strike to infinity; an interior bracket covers one or two degrees. So
the tails carried 0.2–0.6 of probability mass against a market price of
$0.01 and won the max-edge comparison by construction.

The record shows the fingerprint exactly:

| | |
|---|---|
| Bets | 15 |
| On open-ended tail brackets | **15 of 15** |
| Entered at the $0.01 price floor | 11 of 15 |
| Wins | 1 |
| Net | +$42.50 on $7.50 staked |

The single win (2026-07-22 high, `"91° or below"` at $0.01) landed
exactly on its inclusive boundary — CLI reported 91.0 while the
observation stream reported 93.92. Under the stream value it would have
lost and the ledger would read −$7.50. The headline P&L is one boundary
case, not evidence of edge.

The same rule also had the two lead-time legs bet opposite tails of the
same market an hour apart on 2026-07-24 (high: `"81° or below"` vs
`"90° or above"`; low: `"53° or below"` vs `"62° or above"`) with the
underlying forecast essentially unchanged.

## Files

| File | Contents |
|---|---|
| `resolved_bets.pre-tailfix.jsonl` | 16 resolved bets, one per line, with the date/side/lead-time and the actual peak they settled against |
| `paper_trades_pending.pre-tailfix.json` | The staging file as it stood at reset |
| `paper_trading_edge_log.pre-tailfix.jsonl` | Every edge evaluation made under the old rule, bet or not |

`resolved_bets` carries 16 rows to the ledger's 15 — the extra is the
2026-07-27 unconditional low bet, which was still unresolved at reset.

## What changed

See `paper_trading.py`'s module docstring, sections "Tail brackets" and
"Lead-time direction conflicts". In short: selection now ranks on
`_credible_probability`, which scores every bracket over at most the
market's typical interior width and clips that to within
`CREDIBLE_SIGMA_SPAN` of the point estimate; and `_direction_conflict`
blocks a leg that jumps two or more brackets away from the other leg's
pick without a matching move in the point estimate.

Replaying the 2026-07-26 low under both rules: the old rule bet
`"62° or above"` at an edge of +0.57 and lost; the new rule scores that
same bracket at +0.09, below `MIN_EDGE`, and places no bet.
