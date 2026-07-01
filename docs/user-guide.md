# User Guide — Finding Better Entries and Maximizing Reward vs. Risk

This guide explains how to use every feature of the Trade Setup Planner together as a
repeatable workflow: **where to enter, where to exit, and how much to risk** — while
staying on the right side of reward vs. risk.

> Educational decision-support tool only. Nothing here is financial advice or a trade
> recommendation. Market data may be delayed or incomplete.

---

## The big picture

The app started as an **exit/risk engine** (regime-aware ATR stops) and now also provides
**entry-timing context**, so you can reason about a complete trade plan:

```
entry trigger  →  stop  →  target  →  reward:risk  →  position size
```

The workflow below moves from a broad watchlist down to a single, well-justified trade
plan.

---

## Step 1 — Set your context (sidebar)

| Input | What it does | How to choose |
|---|---|---|
| **Tickers** | Watchlist to analyze (commas or new lines). | Paste the names you're tracking. |
| **Trading strategy** | Sets stop width *and* tunes entry thresholds (pullback bands, target distance, RSI reset band, volume/relative-strength expectations). | Match it to how long you actually hold: `Day`, `Swing`, `Trend`, `Position`. |
| **Direction** | Long or short. Every entry signal and the stop flip accordingly. | Pick the side you intend to trade. |
| **Relative-strength benchmark** | Index your relative strength is measured against. | `SPY` for broad names; `QQQ` for tech-heavy names; `IWM` for small caps. |
| Indicator settings / overrides | ATR window, regime lookback, Bollinger window, VIX overlay, regime/multiplier overrides. | Leave at defaults unless you have a specific reason. |

Click **Analyze Setups** to run the whole watchlist.

---

## Step 2 — Triage with the Setup Scanner

The **Setup Scanner** ranks your entire watchlist by setup quality so you spend attention
on the best opportunities first.

- **Rank by Quality Score** to surface the strongest names overall.
- **Rank by Reward:Risk** to find the most asymmetric opportunities.
- **Rank by Signal Win % / Avg R** to lean on historical behavior of that signal type.
- **Filter to A/B quality grades**, set the **Timing** filter to `Ready` when you want
  pullback-ready entries (skip `Stretched` chases), and set a **Min Reward:Risk** (e.g.
  `2.0`) to hide weak ideas.

Only the names that survive triage are worth a deeper look. The top row is highlighted in
the caption beneath the table.

---

## Step 3 — Drill into a candidate (Entry Panel)

Select a ticker and read its two badges: **Setup Quality (0-100 + A/B/C grade)** for how
strong the name is, and **Entry Timing (Ready / Fair / Stretched)** for whether now is a
good moment. Two component tables show *why*, factor by factor (each factor's 0-100
sub-score and its weight). The strongest, best-timed setups generally line up like this
(long example; shorts mirror it):

- **Trend alignment** — price stacked with MA50/MA200 in your direction. *(Quality)*
- **Relative strength** — `Leader` or `Improving Laggard` versus the benchmark. *(Quality)*
- **Cost basis / volume** — price above VWAP (`Clean`), and for breakout strategies a
  **volume surge** confirming the move. *(Quality)*
- **Location** — `At Support` or `Near`. A shallow pullback toward rising support beats
  chasing an `Extended` move. Location is measured in **ATR units** from MA50. *(Timing)*
- **Trigger (RSI)** — `Resetting Up`: momentum turning back your way rather than an
  exhausted chase at overbought. *(Timing)*

Because the two reads are independent, a strong name can score **Quality A** while its
**Timing is Stretched** (extended, overbought) — that means wait for a pullback rather than
chase. The panel shows a warning when this happens. Factors with missing data are excluded
from their axis (weights renormalize), so a missing series lowers confidence rather than
unfairly tanking the read.

---

## Step 4 — Build the trade plan

In the Entry Panel:

- **Planned entry** — set the price you'd actually act on. The stop *distance* is
  volatility-based and stays fixed; the **stop price, risk %, and share count update** to
  the planned entry.
- **Target method** — `Structure` uses the nearest recent swing high/low; `ATR multiple`
  uses a fixed multiple of ATR.
- **Reward:Risk** — favor setups offering at least your strategy's target R (≈2R for
  trend/position). If R:R drops below `1`, the app warns you: wait for a better entry or a
  closer level of support to lean on.

---

## Step 5 — Sanity-check with the signal replay (backtest)

The Entry Panel replays this ticker's historical entry triggers with the same ATR stop and
an ATR-multiple target over the strategy's holding window, then reports:

- **Win rate** — wins / (wins + losses); open trades excluded.
- **Average R (closed)** — mean realized R across closed trades.
- **Expectancy** — average R across all signals (open trades marked to market).
- **W / L / Open** counts, and the markers on the chart are colored by outcome.

Prefer setups with **positive expectancy**. This is a look-ahead-safe what-if for
calibration only — it ignores slippage, intrabar gaps, and overlapping positions, so treat
it as rough context, not a track record.

---

## Step 6 — Size the position and manage the exit

- Enable **Calculate position size** and set your account size and **risk per trade (%)**.
  The app converts your fixed risk into a share count using the stop distance, so a
  stop-out only costs your planned percentage of the account. An optional capital cap
  limits position value.
- The **Stop Price** and **Risk % to Stop** define your exit *before* you enter. Plan the
  exit first; the entry only earns its place if the reward justifies that risk. The chart's
  stop-price line shows how that exit would have tracked price historically.

---

## Quick checklist to maximize reward vs. risk

1. Trade **with** the trend and a **leader** in relative strength.
2. Enter **near support on a resetting trigger** — not extended, not chasing.
3. Require an **asymmetric Reward:Risk** (ideally ≥ 2R) before committing.
4. Confirm **positive historical expectancy** for that setup type.
5. **Size by a fixed risk %**, and let the predefined ATR stop manage the downside.

A grade-A setup with a 3R target, leadership relative strength, and positive expectancy —
entered near support and sized to a 1% risk budget — is the kind of asymmetric opportunity
this workflow is designed to surface.

---

## Field reference (quick)

- **Quality Score / Quality Grade** — 0-100 name-strength score (trend, relative strength, cost basis, volume) and its A/B/C grade.
- **Timing** — entry-timing read (Ready / Fair / Stretched) from location vs. support and the RSI trigger.
- **Location** — distance from MA50 in ATR units (At Support / Near / Neutral / Extended).
- **Target / Reward:Risk** — estimated target and reward-to-risk in R.
- **Signal Win % / Signal Avg R** — historical hit-rate and average R from the signal replay.
- **Stop Price / Stop Distance / Risk % to Stop** — the predefined exit and its size.
- **Trend Strength / Long-Term Trend / VWAP Strength** — % vs. MA50, MA200, and VWAP50.

See `docs/entry-panel-design.md` for the full design and the per-strategy configuration.
