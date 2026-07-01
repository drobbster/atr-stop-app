# Entry Panel — Design Spec (Phase 1 & Phase 2)

## 0. Purpose & scope

The app (now the **Trade Setup Planner**, originally the ATR Stop Calculator) began as
an **exit/risk engine**: given a ticker you already hold (or plan to hold), it answers
"how volatile is this, what regime is it in, where does the stop go, and how many shares
fit my risk budget."

This spec designs an **Entry Panel** that completes the trade-plan loop:

```
entry trigger  →  stop  →  target  →  reward:risk  →  position size
```

It is delivered in two phases that reuse the existing data pipeline
(`calculate_volatility_indicators` → `generate_stop_for_ticker`) and the existing
Streamlit layout. Nothing here changes the stop math; it adds entry context around it.

- **Phase 1 — Trend & Location state + Reward:Risk trade plan.** Almost entirely
  derived from columns the app already computes. Establishes the panel and the
  planned-entry workflow.
- **Phase 2 — Momentum/volume trigger + Relative Strength + quality tag.** Adds the
  "is it firing now, and is this worth entering at all?" signals, plus a coarse A/B/C
  setup grade.

Both phases preserve the educational, non-advice framing. The panel presents **setup
quality context**, never "buy now" calls, and keeps the same explainable,
component-by-component presentation already used by the "Regime Details" table.

> **Out of scope (deferred):** composite numeric entry score, backtest hit-rate / R
> statistics, intraday/day-trade entries, multi-ticker scanner ranking, alerts, and
> earnings-date awareness. These are noted in §7 as the roadmap beyond Phase 2.

---

## 1. Current-state reference (what we build on)

The values below already exist and are reused as-is.

`calculate_volatility_indicators(ticker, ...) -> (df, summary)` produces a daily
DataFrame with at least these columns:

- Price: `Open`, `High`, `Low`, `Close`, `Volume`
- Trend: `MA50`, `MA200`, `Trend Strength`, `Long-Term Trend`, `VWAP50`, `VWAP Strength`
- Volatility: `TR`, `ATR`, `ATR_Ratio`, `ATR_Regime`, `BB_Width`, `BB_Ratio`,
  `BB_Regime`, `VIX_Regime`, `Regime_Score`, `Volatility_Regime`

The `summary` dict carries the latest-bar scalars: `entry_price` (latest close),
`atr`, `ma50`, `ma200`, `vwap50`, `trend_strength`, `long_term_trend`,
`vwap_strength`, `volatility_regime`, `regime_score`, `date`, etc.

`calculate_best_stop(...)` returns `Stop Distance`, `Stop Price`, `Risk % to Stop`,
`ATR Multiplier`, etc. `calculate_position_size(...)` converts a stop distance into
shares. `download_price_history("^VIX", period="1y")` already shows the pattern for
pulling a secondary series (reused for `SPY` in Phase 2).

Strategy buckets are `day`, `swing`, `trend`, `position` (labels in `STRATEGY_LABELS`,
multipliers in `ATR_MULTIPLIERS`).

---

## 2. Phase 1 — Trend & Location state + Reward:Risk trade plan

### 2.1 New derived columns (computed on the existing `df`)

These are added inside `calculate_volatility_indicators` (or a dedicated
`add_entry_features(df)` helper called right after it) so both the latest-bar summary
and the historical chart can use them. All are pure functions of existing columns —
**no new network calls in Phase 1**.

| Column | Formula | Notes |
|---|---|---|
| `Dist_MA50_ATR` | `(Close - MA50) / ATR` | Location vs. intermediate trend, in ATR units. Sign = above/below; magnitude = how extended. |
| `Dist_VWAP50_ATR` | `(Close - VWAP50) / ATR` | Location vs. recent volume-weighted cost basis, in ATR units. |
| `Dist_MA200_ATR` | `(Close - MA200) / ATR` | Location vs. long-term trend, in ATR units. |
| `Trend_Alignment` | see §2.2 | Categorical: `Aligned Long` / `Aligned Short` / `Mixed`. |
| `Location_State` | see §2.3 | Categorical per direction: `At Support` / `Near` / `Neutral` / `Extended`. |
| `CostBasis_Flag` | see §2.4 | `Clean` / `Overhead Supply` / `N/A`. |

Guard every division: if `ATR`, `MA50`, `MA200`, or `VWAP50` is `NaN` or `<= 0`, the
derived value is `NaN` and the corresponding state degrades to a neutral label
(`Mixed` / `Neutral` / `N/A`). `VWAP50` may be `NaN` when `Volume` is unavailable
(already handled upstream), so `Dist_VWAP50_ATR` and `CostBasis_Flag` must tolerate it.

### 2.2 Trend alignment

```text
if Close > MA50 > MA200:        Trend_Alignment = "Aligned Long"
elif Close < MA50 < MA200:      Trend_Alignment = "Aligned Short"
else:                           Trend_Alignment = "Mixed"
```

If `MA200` is `NaN` (insufficient history), fall back to the `Close` vs `MA50`
relationship only and mark alignment as `Mixed` unless `Close`/`MA50` agree, in which
case label `Aligned Long*` / `Aligned Short*` (the asterisk surfaced in a tooltip as
"long-term trend unconfirmed"). The app already requires ≥ ~205 bars, so this is an
edge case, not the norm.

### 2.3 Location state (direction-aware)

Location is interpreted relative to the **selected `direction`**. For a long, "good"
means a shallow pullback toward rising support; for a short, the mirror image above
resistance. Using `d = Dist_MA50_ATR`:

```text
# direction == "long"
d <= 0.5    →  "At Support"     (pullback into MA50, prime long-entry zone)
0.5 < d<=1.5 → "Near"
1.5 < d<=3.0 → "Neutral"
d > 3.0     →  "Extended"        (chasing; poor entry location)
d < -0.5    →  "Below Support"   (trend not confirming a long)

# direction == "short": flip the sign of d before applying the same bands
```

Thresholds (`0.5 / 1.5 / 3.0`) are **strategy-tunable** (see §4). They live in a single
config dict so day vs. position can use tighter/wider bands.

### 2.4 Cost-basis flag

```text
above_ma50  = Close > MA50
below_vwap  = Close < VWAP50
if VWAP50 is NaN:                       CostBasis_Flag = "N/A"
elif above_ma50 and below_vwap:         CostBasis_Flag = "Overhead Supply"
else:                                   CostBasis_Flag = "Clean"
```

This encodes the nuance already documented in the README: price above MA50 but below
VWAP50 means recent high-volume buyers are underwater.

### 2.5 Planned-entry workflow

Today every metric is pinned to the latest close. Phase 1 lets the user model an entry
at a chosen price for the **selected** ticker in the detail view (not the whole batch).

New UI input in the Entry Panel:

- **Planned Entry Price** — `st.number_input`, default = `summary["entry_price"]`
  (latest close), `min_value = 0.0`, `step` sized to the ticker (e.g.
  `max(0.01, round(price * 0.001, 2))`). A "Reset to last close" affordance restores
  the default.

When the planned entry differs from the latest close, recompute the trade plan at that
price **without re-fetching data**:

- `stop_distance` is unchanged in dollars (ATR × multiplier is a volatility measure,
  independent of where you enter) — reuse `selected_result["Stop Distance"]`.
- `stop_price = planned_entry - stop_distance` (long) / `+ stop_distance` (short).
- `risk_pct = stop_distance / planned_entry * 100`.
- Position sizing, if enabled, is recomputed via `calculate_position_size(...)` using
  `entry_price = planned_entry`.

> Rationale: stop *distance* is a property of volatility; only the *price* and
> *risk %* shift with the entry. This keeps Phase 1 consistent with the existing
> stop engine and avoids a second data fetch.

### 2.6 Target & Reward:Risk

Add a `compute_target(...)` helper. Two user-selectable target methods, defaulting to
the structure method:

1. **Structure (default):** nearest swing extreme from recent history.
   - Long target = highest `High` over the last `N` bars *excluding the last bar*
     (`rolling(N).max().shift(1)` style), where `N` is strategy-tuned (§4, e.g. swing
     `N=20`, trend `N=55`). If that level is below the planned entry (already broken
     out), fall back to method 2.
   - Short target = lowest `Low` over the last `N` bars, mirror logic.
2. **ATR multiple:** `target = entry ± (reward_atr_mult × ATR)`, with `reward_atr_mult`
   defaulting per strategy (e.g. swing 3.0, trend 5.0). Always available as a fallback.

Then:

```text
risk_distance   = |entry - stop_price|        # = stop_distance
reward_distance = |target - entry|
reward_to_risk  = reward_distance / risk_distance      # guard risk_distance > 0
risk_in_R       = 1.0                                    # by definition, 1R = stop distance
reward_in_R     = reward_to_risk
```

Display `R:R` as e.g. `2.8R` and the target price/level. If the structure target
produces `R:R < 1`, surface a caution ("limited upside to nearest resistance") rather
than hiding it — that *is* useful entry information.

### 2.7 Phase 1 function-level changes (signatures)

No existing signatures are broken; new helpers are additive.

```python
def add_entry_features(df: pd.DataFrame) -> pd.DataFrame: ...
    # adds Dist_*_ATR, Trend_Alignment, Location_State, CostBasis_Flag

def compute_target(
    entry_price: float,
    atr_value: float,
    direction: str,
    strategy_type: str,
    df: pd.DataFrame,
    method: str = "structure",   # "structure" | "atr"
) -> dict: ...
    # returns {"target_price", "target_method", "reward_distance"}

def build_trade_plan(
    planned_entry: float,
    stop_distance: float,
    direction: str,
    target_price: float,
) -> dict: ...
    # returns {"Planned Entry", "Stop Price", "Stop Distance",
    #          "Risk %", "Target", "Reward:Risk", "Reward in R"}
```

`add_entry_features` is called once inside `calculate_volatility_indicators` (so the
chart history carries the columns) and the latest-bar values are copied into `summary`
(`dist_ma50_atr`, `trend_alignment`, `location_state`, `cost_basis_flag`, ...).

---

## 3. Phase 2 — Momentum/volume trigger + Relative Strength + quality tag

### 3.1 New derived columns

| Column | Formula | Notes |
|---|---|---|
| `RSI` | Wilder RSI(14) — see §3.2 | Reuse the existing Wilder smoothing pattern (`ewm(alpha=1/n, adjust=False)`). |
| `RSI_State` | banding of `RSI` — see §3.3 | `Oversold` / `Resetting Up` / `Neutral` / `Overbought`. |
| `RelVolume` | `Volume / Volume.rolling(20).mean()` | `NaN` when `Volume` missing. |
| `Vol_Confirm` | `RelVolume >= vol_confirm_mult` | Boolean-ish flag; mult strategy-tuned (default 1.5). |
| `RS_Ratio` | `Close / SPY_Close` (aligned, ffilled) | Relative-strength line vs. benchmark. |
| `RS_Slope` | sign/slope of `RS_Ratio` over ~21 bars | Rising = leadership, falling = laggard. |
| `RS_Return_Rel` | `ret_ticker_63 - ret_spy_63` | 3-month relative return (pct points). |

### 3.2 RSI(14) — Wilder

Mirror the ATR smoothing already in the codebase for consistency:

```python
delta = df["Close"].diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
df["RSI"] = 100 - (100 / (1 + rs))
```

### 3.3 RSI state (direction-aware trigger)

For a long, the high-quality trigger is a pullback that *resets and turns back up*,
not a chase at overbought:

```text
RSI <= 30                         →  "Oversold"
30 < RSI <= 50 and RSI rising      →  "Resetting Up"   (prime long trigger w/ At Support)
50 < RSI < 70                      →  "Neutral"
RSI >= 70                          →  "Overbought"     (entry caution for new longs)
```

"rising" = `RSI > RSI.shift(1)`. Mirror the bands for shorts
(`>=70 Overbought = short reset zone`, etc.). Thresholds are strategy-tunable.

### 3.4 Relative strength vs. benchmark

Add `download_price_history("SPY", period="1y")` using the **exact same cached helper
and join pattern** already used for `^VIX`:

```python
spy = download_price_history("SPY", period="1y")
spy_close = spy[["Close"]].rename(columns={"Close": "SPY_Close"})
df = df.join(spy_close, how="left")
df["SPY_Close"] = df["SPY_Close"].ffill()
df["RS_Ratio"] = df["Close"] / df["SPY_Close"]
df["RS_Slope"] = df["RS_Ratio"].diff(21)        # 1-month change in RS line
# strategy-tuned relative-return lookback (ENTRY_CONFIG["rs_lookback"]), in pct points
n = ENTRY_CONFIG[strategy_type]["rs_lookback"]
df["RS_Return_Rel"] = (
    df["Close"].pct_change(n) - df["SPY_Close"].pct_change(n)
) * 100
```

Leadership read (latest bar):

```text
RS_Return_Rel > 0 and RS_Slope > 0   →  "Leader"
RS_Return_Rel > 0 and RS_Slope <= 0  →  "Cooling Leader"
RS_Return_Rel <= 0 and RS_Slope > 0  →  "Improving Laggard"
else                                 →  "Laggard"
```

> True percentile ranking across a universe is deferred (needs a scanner). The
> ratio/slope read above is self-contained and needs only one extra series.
> If the `SPY` fetch fails, RS degrades to `N/A` and is dropped from the quality tag
> (weights renormalize, §3.5) — the panel still renders.

The benchmark symbol should be a small config constant (`BENCHMARK = "SPY"`) so it can
later become user-selectable (e.g. `QQQ` for tech-heavy names).

### 3.5 Quality tag (A / B / C) — coarse, explainable

Phase 2 introduces a **categorical grade**, not a numeric score (the numeric composite
is deferred). The grade is the count of satisfied, direction-aware conditions mapped to
a letter. Each condition is a clear pass/neutral/fail already surfaced in the
components table:

| Condition (long; mirror for short) | Pass when |
|---|---|
| Trend aligned | `Trend_Alignment == "Aligned Long"` |
| Good location | `Location_State in {"At Support", "Near"}` |
| Clean cost basis | `CostBasis_Flag == "Clean"` |
| Trigger firing | `RSI_State in {"Oversold", "Resetting Up"}` |
| Volume confirms | `Vol_Confirm` is true *(gating only when `ENTRY_CONFIG[strategy]["vol_required"]`; otherwise neutral/confirming)* |
| RS leadership | RS read `in {"Leader", "Improving Laggard"}` |

Grade mapping (over the conditions that are *applicable* and *not N/A*):

```text
pass_ratio = passed / applicable
pass_ratio >= 0.80   →  "A"   (high-quality setup)
0.55 <= pass_ratio<0.80 → "B" (constructive, some caveats)
pass_ratio < 0.55    →  "C"   (low-quality / wait)
```

N/A conditions (e.g. RS unavailable, VWAP missing) are excluded from `applicable` so
the denominator renormalizes — a missing data series lowers confidence, not the grade
unfairly. The headline shows the grade **plus** the human-readable reason string built
from the passing/failing components (e.g. `B · Aligned Long · Near support · RSI
resetting · Laggard RS`).

### 3.6 Phase 2 function-level changes

```python
def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame: ...   # RSI, RelVolume, states
def add_relative_strength(df: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame: ...
def grade_setup(summary: dict, strategy_type: str, direction: str) -> dict: ...
    # returns {"grade", "pass_ratio", "reasons": [...], "components": [...]}
```

`add_momentum_features` and `add_relative_strength` are called right after
`add_entry_features`; latest-bar values are folded into `summary`. `grade_setup`
consumes the enriched `summary`.

---

## 4. Strategy-tuned thresholds (single config block)

All tunables live in one place (parallel to `ATR_MULTIPLIERS`) so behavior is auditable
and per-strategy:

```python
ENTRY_CONFIG = {
    "day":      {"support": 0.30, "near": 1.0, "ext": 2.0, "target_n": 10,
                 "reward_atr": 2.0, "rsi_lo": 40, "rsi_hi": 60,
                 "vol_mult": 2.0, "vol_required": True,  "rs_lookback": 21},
    "swing":    {"support": 0.50, "near": 1.5, "ext": 3.0, "target_n": 20,
                 "reward_atr": 3.0, "rsi_lo": 35, "rsi_hi": 55,
                 "vol_mult": 1.5, "vol_required": False, "rs_lookback": 42},
    "trend":    {"support": 0.75, "near": 2.0, "ext": 4.0, "target_n": 55,
                 "reward_atr": 5.0, "rsi_lo": 40, "rsi_hi": 60,
                 "vol_mult": 1.3, "vol_required": True,  "rs_lookback": 63},
    "position": {"support": 1.00, "near": 3.0, "ext": 5.0, "target_n": 120,
                 "reward_atr": 6.0, "rsi_lo": 40, "rsi_hi": 60,
                 "vol_mult": 1.2, "vol_required": False, "rs_lookback": 126},
}
```

**Field reference**

| Field | Meaning | Used by |
|---|---|---|
| `support` / `near` / `ext` | Location bands in ATR units of distance from MA50 (direction-aware) | §2.3 `Location_State` |
| `target_n` | Lookback (bars) for the structure swing-extreme target | §2.6 `compute_target` |
| `reward_atr` | Fallback ATR-multiple target when no usable structure level | §2.6 `compute_target` |
| `rsi_lo` / `rsi_hi` | "Reset" band for the RSI trigger | §3.3 `RSI_State` |
| `vol_mult` | Relative-volume threshold for `Vol_Confirm` | §3.1 / §3.5 |
| `vol_required` | Whether volume *gates* the grade (breakout strategies) vs. merely confirms | §3.5 `grade_setup` |
| `rs_lookback` | Relative-return window (bars) vs. benchmark | §3.4 `add_relative_strength` |

**Rationale for these defaults**

- **Bands scale with the strategy's own stop width.** Stops are already horizon-scaled
  via `ATR_MULTIPLIERS` (Normal regime: day ≈1.25 ATR, swing 2.0, trend 2.5, position
  3.0). Location bands widen in the same proportion, and `ext` sits near/just past the
  stop multiple so "Extended" means price has run roughly a full stop's worth past
  support. A position trade (3-ATR stop) shouldn't flag a 2-ATR pullback as chasing.
- **Targets give an escalating, sensible R:R.** Using the `reward_atr` fallback against
  the Normal-regime stop: day 2.0/1.25 ≈ **1.6R**, swing 3.0/2.0 ≈ **1.5R**,
  trend 5.0/2.5 ≈ **2.0R**, position 6.0/3.0 ≈ **2.0R**. Shorter horizons accept ~1.5R;
  longer horizons demand ~2R to justify holding through more noise. The structure
  target overrides this whenever a real swing level is closer/farther.
- **`target_n` matches typical visibility/hold:** day 10, swing 20, trend 55 (classic
  Donchian breakout level), position 120 (~6 months).
- **RSI reset band:** swing uses the tightest classic-pullback window (35–55); others
  use 40–60. Below `rsi_lo` = oversold; above `rsi_hi` = caution for new longs.
- **Volume:** required and strong for breakout-style entries (day 2.0×, trend 1.3×);
  lower and merely confirming for pullback-style entries (swing 1.5×, position 1.2×).
- **`rs_lookback` matches the holding horizon** (day 21 → position 126 bars) so a
  short-horizon setup isn't graded on multi-month leadership.

(These are calibrated starting points, not sacred — they are exactly the knobs the
deferred backtest phase, §7, would tune against historical hit-rate / R.)

Per-strategy entry archetype emphasis (drives which conditions are "required" vs.
"neutral" in §3.5):

- **day:** VWAP/short-MA reclaim + strong volume; tight location bands. *(daily-data
  approximation until intraday lands in a later phase.)*
- **swing:** pullback to rising MA50/VWAP + RSI reset; volume helpful, not required.
- **trend:** Donchian-ish breakout (location `Near`/`Extended` of a fresh N-day high) +
  RS leadership; volume confirmation weighted higher.
- **position:** MA200 reclaim / long-base; RS and long-term trend weighted higher,
  location bands widest.

---

## 5. Entry Panel UX

Rendered in the selected-ticker detail view, beside/above the existing stop & risk
metric columns. Top-to-bottom:

1. **Headline verdict** — grade chip + reason string, e.g.
   `B  ·  Aligned Long · Near MA50 support · RSI resetting · Laggard RS`.
   Color the chip green/amber/grey for A/B/C. Phase 1 ships this without the letter
   grade (just the state string); Phase 2 adds the grade.
2. **Trade-plan row** (`st.columns`): `Planned Entry` (input) | `Stop` | `Target` |
   `R:R` | `Risk %` | `Shares` (when sizing enabled). Mirrors the existing
   `st.metric` styling.
3. **Setup components table** — one row per factor with raw value + Pass/Neutral/Fail,
   styled exactly like the current "Regime Details" dataframe so it reads as
   transparent context, not a black box. Columns: `Factor`, `Value`, `Read`.
4. **Chart markers** — overlay entry-trigger markers on the existing Altair chart by
   adding a layer of `mark_point` where the historical bar satisfied the trigger
   (reuses the historical `df` that already powers the stop-price line). A toggle hides
   them.
5. **Method controls** — small `selectbox`es for target method (`structure`/`atr`) and
   (Phase 2) benchmark, plus the planned-entry reset.

The strategy selector in the sidebar now also tunes entry thresholds via `ENTRY_CONFIG`
(§4), so it meaningfully shapes *entries*, not just stop width. Keep all existing
disclaimers; add one line to the panel: "Setup context for education only — not a
trade recommendation."

### 5.1 Text wireframe

```
┌─ Entry Panel ───────────────────────────────────────────────┐
│  [ B ]  Aligned Long · Near MA50 support · RSI resetting · Laggard RS │
│                                                              │
│  Planned Entry   Stop      Target    R:R     Risk %   Shares │
│  [ 121.40 ]      112.10    138.0     2.8R    7.6%     63     │
│                                                              │
│  Factor            Value              Read                   │
│  Trend alignment   Close>MA50>MA200   Pass                   │
│  Location (ATR)    +0.9 ATR vs MA50   Pass (Near)            │
│  Cost basis        above VWAP50       Pass (Clean)           │
│  Trigger (RSI)     44, rising         Pass (Resetting Up)    │
│  Rel volume        1.1x avg           Neutral                │
│  Rel strength      -3.2% vs SPY       Fail (Laggard)         │
│                                                              │
│  Target method: [structure ▾]   Benchmark: [SPY ▾]  ↺ reset  │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. Edge cases, data, performance, testing

**Data / robustness**
- Missing `Volume` → `VWAP50`, `RelVolume`, `Vol_Confirm`, `CostBasis_Flag` degrade to
  `N/A`; excluded from grading denominator.
- `SPY` fetch failure → RS read `N/A`; panel still renders; grade renormalizes.
- Short history (`MA200` NaN) → alignment falls back per §2.2; structure target falls
  back to ATR-multiple when the rolling window is incomplete.
- All divisions guarded against zero/NaN (ATR, MA, VWAP, avg_loss, risk_distance).
- Planned entry of `0`/blank → revert to latest close and warn inline.

**Performance / caching**
- Phase 1 adds zero network calls. Phase 2 adds exactly one cached series (`SPY`) via
  the existing `@st.cache_data(ttl=900)` `download_price_history`, fetched once and
  joined like `^VIX`. Negligible added latency.
- All new columns are vectorized pandas ops on the already-loaded `df`.

**Backward compatibility**
- No existing function signature changes; all additions are new helpers and new
  optional UI. The batch results table, CSV export, and stop chart are unchanged
  (new columns may optionally be appended to the CSV behind the same code path).

**Testing**
- Unit tests for: `add_entry_features` (alignment/location/cost-basis truth tables incl.
  NaN inputs), `compute_target` (structure vs. atr, breakout fallback, long/short),
  `build_trade_plan` (R:R math, risk% at non-close entries), RSI(14) against a known
  reference series, `add_relative_strength` (SPY-missing path), and `grade_setup`
  (renormalization when conditions are N/A, A/B/C boundaries).
- Manual smoke test in Streamlit across: a clean uptrend leader, an extended/chasing
  name, a downtrend, a no-volume ticker, and an `SPY`-fetch-failure simulation.

---

## 7. Roadmap beyond Phase 2

> **Status:** items 1–3 have since shipped, and the single Entry Score was later **split
> into two independent reads** (Setup Quality + Entry Timing) — see the §10 decision record.
> A look-ahead-safe **signal replay** reports win-rate / average-R / expectancy per ticker,
> and the **Setup Scanner** ranks a whole watchlist. An RSI-divergence factor was prototyped
> and then **rejected** after backtesting — see the §9 decision record. Items 4–5 remain deferred.

1. ~~**Composite numeric Entry Score**~~ — *shipped, then superseded.* Originally a single
   weighted 0–100 score (`grade_setup` + `ENTRY_WEIGHTS`); now split into a Setup Quality
   score/grade and an Entry Timing state (§10).
2. ~~**Backtest markers & stats**~~ — *shipped.* The signal replay walks every historical
   trigger forward with the same ATR stop and an ATR-multiple target and reports win
   rate, average R, and expectancy.
3. ~~**Scanner mode**~~ — *shipped.* The Setup Scanner ranks a watchlist by Quality Score /
   Reward:Risk with quality-grade, timing, and Reward:Risk filters.
4. **Intraday entries** — opening-range / intraday VWAP for the `day` bucket (new data
   cadence).
5. **Alerts & earnings-date awareness** — "notify on pullback to MA50 within 0.5 ATR";
   flag event risk near entry.

---

## 8. Build order (summary)

- **Phase 1:** `add_entry_features` → planned-entry input → `compute_target` /
  `build_trade_plan` → Entry Panel scaffold (headline state + trade-plan row +
  components table) → chart unchanged. Completes the entry↔exit loop on existing data.
- **Phase 2:** `add_momentum_features` (RSI/volume) → `add_relative_strength` (SPY) →
  `grade_setup` (A/B/C) → wire grade + RS/RSI/volume rows into the panel + trigger
  chart markers + `ENTRY_CONFIG` strategy tuning.

---

## 9. RSI divergence — evaluated and rejected (decision record)

> **Status: not in the app.** RSI divergence (price highs not confirmed by RSI) was built
> as a swing-pivot factor, trialed first as a weighted scoring modifier and then as an
> informational read, and finally **removed entirely** after backtesting showed no edge.
> This section is kept as a decision record so the idea isn't naively re-added.

### 9.1 The idea

The `Trigger (RSI)` factor (§3.3) reads RSI's *absolute zone* but is blind to whether
momentum is **confirming** price. A **bearish divergence** is a higher price high on a
*lower* RSI high (the mirror, **bullish divergence**, is a lower low on a higher RSI low).
It is a staple of chart commentary as a topping/bottoming caution.

### 9.2 How it was measured

A swing-pivot implementation (fractal pivot highs/lows, RSI compared at the last two
*confirmed* pivots, look-ahead-safe) was evaluated **onset-based**: forward outcomes were
measured from the moment each `Confirmed` / `Bearish Divergence` state first appears, which
avoids the autocorrelation that inflates naive per-bar tests. Universe: 35 liquid
stocks/ETFs, 5–8y daily. Candidate pivot windows of 2–21 bars were tested per horizon.

### 9.3 Findings — no usable edge

- **Forward return** (Confirmed − Bearish; positive = divergence precedes weaker returns):
  **negative in 10 of 12** strategy/window cells — bearish-divergence onsets were on
  average followed by *higher* returns than confirmed higher-highs.
- **Forward drawdown** (the "tighten the stop" rationale): bearish-divergence onsets had
  **shallower** max drawdowns than confirmed onsets in **every** strategy/window — opposite
  of a risk caution.
- **Overbought-gated steelman** (divergence + RSI ≥ 65): sample collapses (n≈2–66), no
  coherent edge.
- **Regime cut** (risk-off bars, SPY < MA200, 8y): only a weak, inconsistent hint of
  relative-return underperformance at swing/position horizons (small n); the drawdown gap
  stayed negative in every regime, so still not a stop signal.

The pivot window barely changed any of this — the weakness was the signal, not the
parameter. Divergences are routinely overrun in trends.

### 9.4 Decision

Removed from the codebase and all user-facing surfaces: no scoring factor, no informational
read, no Scanner column, no `ENTRY_CONFIG` pivot knobs, not in the signal replay, and no
mention in the in-app Help, README, or user guide. This design doc is the only remaining
record, kept so the idea isn't naively re-added.

### 9.5 Caveats / if revisited

The test window (2018–2026) was predominantly bullish and the basket is current survivors,
so the study under-samples genuine major tops where divergence is theorized to work best.
If revisited, do it *only* behind the deferred backtest harness (§7), test a *specific
conditional* setup (e.g. divergence + extended location + overbought) rather than a blanket
read, and require it to clear a pre-registered edge threshold before it goes anywhere near
the score — do **not** reinstate it as an always-on indicator.

---

## 10. Single Entry Score → two-axis read (decision record)

> **Status: shipped.** The single blended Entry Score was replaced with two independent
> reads: **Setup Quality** and **Entry Timing**. `grade_setup` returns both.

### 10.1 Why

The composite Entry Score blended two questions a trader keeps separate: *is this a strong
name?* and *is now a good moment to enter?* Because the timing factors (location, RSI
trigger) are a minority of the per-strategy weight (~28% for `trend`), a strong-but-extended
leader (e.g. MU: Aligned Long, Leader, Clean, volume — but `Location: Extended` and
`Trigger: Overbought`) still cleared the A threshold. The single green "A" read as "enter
now" even though two factors were explicitly failing. This is a semantics problem the
one-number UI amplified, not a bug in the factor math.

### 10.2 The split

`grade_setup` partitions the same six sub-scores (unchanged) into two groups and computes a
weighted average within each (renormalized over applicable factors):

- **Setup Quality** — `trend`, `rs`, `cost_basis`, `volume` → 0–100 score + A/B/C grade
  (A ≥ 75, B ≥ 55, else C).
- **Entry Timing** — `location`, `trigger` → internal 0–1 fraction mapped to a state:
  **Ready** (≥ 0.66), **Fair** (≥ 0.33), **Stretched** (< 0.33).

The return dict is `{quality_grade, quality_score, timing_state, timing_score,
quality_reason, timing_reason, components}`; each component row carries a `Group`
("Quality"/"Timing") tag so the panel can render two tables.

### 10.3 Surfacing

- **Entry Panel:** two badges (Quality grade + score, Timing state) and two component
  tables. A `st.warning` fires when Quality is A/B but Timing is Stretched ("strong name,
  poor entry — wait for a pullback").
- **Scanner:** `Quality Grade`, `Quality Score`, and `Timing` columns; sort defaults to
  Quality Score; filters for quality grade and timing (pick `Ready` to isolate
  pullback-ready entries).

### 10.4 Deliberately not done

No hard veto or grade cap when a name is Extended/Overbought — that would suppress
legitimately strong trend names on an unvalidated rule (same discipline that led to
rejecting RSI divergence, §9). Timing is surfaced alongside Quality, not used to gate it.
