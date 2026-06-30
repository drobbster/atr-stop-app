# Trade Setup Planner

Streamlit app that plans a full stock/ETF trade — **entry, stop, target, reward:risk, and
position size** — and scores how good each setup is. (Formerly the *ATR Stop Calculator*;
the regime-aware ATR stop is now the risk leg of a broader entry/exit planner.)

The app is an educational decision-support tool. It helps answer:

- How good is this setup right now (0-100 Entry Score and A/B/C grade)?
- Which names on my watchlist deserve attention first (Setup Scanner)?
- Where would the entry, stop, and target sit, and is the reward:risk asymmetric?
- How volatile is this ticker, and what volatility regime is it in?
- How far can the position move against me before the stop is hit, and how many shares fit my risk budget?
- Is price above or below its intermediate and long-term moving averages?

## Functionality

### Setup Scanner

When you analyze more than one ticker, the **Setup Scanner** ranks the whole watchlist so
you can triage the best opportunities first. Sort by **Entry Score** (highest-quality
setups), **Reward:Risk** (most asymmetric), or **Signal Win % / Avg R**, and filter to
**A/B grades** with a **minimum Reward:Risk** to hide weak ideas. Results also include the
per-ticker chart/detail and the full Entry/Exit Plan in separate tabs, plus a CSV export.

### Regime-Aware ATR Stops

For each ticker, the app downloads daily Yahoo Finance price history and calculates:

- Average True Range (ATR)
- ATR regime
- Bollinger Band width regime
- Optional VIX macro volatility regime
- Combined volatility regime
- Strategy-specific ATR multiplier
- Stop distance
- Stop price
- Risk % to stop

The default ATR multiplier depends on both the selected trading horizon and volatility regime.
Shorter-term strategies use tighter stops, while longer-term strategies use wider stops.

### Trend Context

The app also calculates simple trend context:

- `MA50`: 50-day moving average
- `MA200`: 200-day moving average
- `VWAP50`: 50-day volume-weighted average price
- `Trend Strength`: current close vs. 50-day moving average
- `Long-Term Trend`: current close vs. 200-day moving average
- `VWAP Strength`: current close vs. 50-day volume-weighted average price

Positive trend values mean the latest close is above the moving average. Negative values mean
it is below the moving average. Positive VWAP strength means price is above the recent
volume-weighted cost basis.

Use these metrics as context:

- Price above both `MA50` and `VWAP50` suggests the simple trend and recent volume-weighted cost
  basis are confirming each other.
- Price above `MA50` but below `VWAP50` can mean the trend looks constructive, but recent
  high-volume buyers may still be underwater.
- Very high positive values can mean strength, but they can also mean the move is extended.
- Compare these metrics with ATR and risk % to stop before deciding whether the stop distance
  gives the trade enough room.

### Entry Setup (Entry Panel)

While the stop logic answers *where to exit*, the Entry Panel adds context for *when to
enter*, so a full plan reads as `entry → stop → target → reward:risk → size`. For the
selected ticker it shows:

- **Entry Score (0-100)** and **grade (A/B/C)**: a composite, explainable score that
  weights direction-aware sub-scores — trend alignment, location vs. support (in ATR),
  cost basis, the RSI trigger, relative volume, and relative strength vs. a benchmark.
  Each factor scores 0-100 and is weighted per strategy; factors with missing data are
  excluded and the rest are renormalized. The grade is derived from the score
  (A ≥ 75, B ≥ 55, else C).
- **Trend alignment**: whether `Close`, `MA50`, and `MA200` are stacked in the trade's
  direction.
- **Location state**: distance from `MA50` measured in ATR units, classified as
  `At Support` / `Near` / `Neutral` / `Extended`. A shallow pullback toward rising
  support scores better than chasing an extended move.
- **RSI trigger**: a Wilder RSI(14) state that favors a pullback resetting back up (for
  longs) rather than a chase at overbought.
- **Relative volume** and **relative strength** vs. the selected benchmark (`SPY` by
  default) as confirmation and leadership signals.
- **Planned entry**: model the plan at any entry price. The stop *distance* stays fixed
  (it is volatility-based), while the stop price, risk %, and shares update.
- **Target & Reward:Risk**: a target from the nearest recent swing level (structure) or
  a fixed ATR multiple, and the resulting reward:risk estimate.
- **Signal replay (backtest)**: every historical entry trigger is walked forward with
  the same ATR stop and an ATR-multiple target to estimate win rate, average R, and
  expectancy. This is a look-ahead-safe what-if for calibration only — it ignores
  slippage, intrabar gaps, and overlapping positions, so treat it as rough context.

All entry thresholds and weights are tuned per trading strategy. This is educational
setup context, not a trade recommendation.

### Chart

The selected ticker chart shows:

- Close price
- MA50
- MA200
- Stop price
- Optional entry-trigger markers (where trend, location, and the RSI trigger all fired
  for the selected direction)

The chart tooltip shows the selected date's close, MA50, MA200, stop price, and risk % to stop.

### Optional Position Sizing

When enabled, position sizing estimates:

- Dollars at risk
- Risk-based share count
- Optional capital-cap share count
- Final shares
- Position value

Position sizing uses the stop distance and your selected account/risk inputs. It does not make
trade recommendations.

## Key Field Definitions

- `Stop Distance`: dollar distance between entry price and stop price.
- `Stop Price`: calculated stop level for the selected direction.
- `Risk % to Stop`: how far the position can move against you before the stop is hit.
- `Trend Strength`: percent above or below the 50-day moving average.
- `Long-Term Trend`: percent above or below the 200-day moving average.
- `VWAP Strength`: percent above or below the 50-day volume-weighted average price.
- `Entry Score`: composite 0-100 entry-quality score from the weighted Entry Panel factors.
- `Setup Grade`: A/B/C grade derived from the Entry Score.
- `Location`: distance from `MA50` in ATR units, bucketed (At Support / Near / Neutral / Extended).
- `Target`: estimated target price (nearest swing level or ATR multiple).
- `Reward:Risk`: distance to target divided by stop distance, expressed in R.
- `Signal Win %` / `Signal Avg R`: historical hit-rate and average R from replaying the
  ticker's entry triggers.

## Disclaimer

This app is for educational use only. Market data may be delayed, incomplete, or unavailable
depending on Yahoo Finance availability. Nothing in the app is financial advice or a trade
recommendation.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Deploy with Streamlit Community Cloud:

1. Sign in with GitHub.
2. Select this repository.
3. Choose the default branch.
4. Set the main file path to `app.py`.
5. Click deploy.
