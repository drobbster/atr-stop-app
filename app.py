import math
from typing import Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf


# =========================
# Configuration
# =========================

ATR_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "day": {"Low": 1.0, "Normal": 1.25, "High": 1.5},
    "swing": {"Low": 1.5, "Normal": 2.0, "High": 2.5},
    "trend": {"Low": 2.0, "Normal": 2.5, "High": 3.5},
    "position": {"Low": 2.5, "Normal": 3.0, "High": 4.0},
}

STRATEGY_LABELS = {
    "day": "Day",
    "swing": "Swing",
    "trend": "Trend",
    "position": "Position",
}

# Default benchmark for relative-strength comparison.
BENCHMARK = "SPY"

# Per-strategy entry tunables. Location bands are in ATR units of distance from MA50;
# target_n is the structure-target lookback; reward_atr is the fallback ATR-multiple
# target; rsi_lo/rsi_hi define the RSI reset band; vol_mult is the relative-volume
# confirmation threshold; vol_required gates the grade for breakout strategies;
# rs_lookback is the relative-return window (bars). See docs/entry-panel-design.md.
ENTRY_CONFIG: Dict[str, Dict[str, float]] = {
    "day": {"support": 0.30, "near": 1.0, "ext": 2.0, "target_n": 10,
            "reward_atr": 2.0, "rsi_lo": 40, "rsi_hi": 60,
            "vol_mult": 2.0, "vol_required": True, "rs_lookback": 21},
    "swing": {"support": 0.50, "near": 1.5, "ext": 3.0, "target_n": 20,
              "reward_atr": 3.0, "rsi_lo": 35, "rsi_hi": 55,
              "vol_mult": 1.5, "vol_required": False, "rs_lookback": 42},
    "trend": {"support": 0.75, "near": 2.0, "ext": 4.0, "target_n": 55,
              "reward_atr": 5.0, "rsi_lo": 40, "rsi_hi": 60,
              "vol_mult": 1.3, "vol_required": True, "rs_lookback": 63},
    "position": {"support": 1.00, "near": 3.0, "ext": 5.0, "target_n": 120,
                 "reward_atr": 6.0, "rsi_lo": 40, "rsi_hi": 60,
                 "vol_mult": 1.2, "vol_required": False, "rs_lookback": 126},
}

# Per-strategy weights for the composite 0-100 Entry Score. Each factor contributes a
# 0..1 sub-score; the weighted average over applicable factors is scaled to 0-100.
# Weights reflect each strategy's archetype (e.g. trend leans on alignment + relative
# strength; day/swing lean on the trigger and location).
ENTRY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "day": {"trend": 1.0, "location": 2.0, "cost_basis": 1.0, "trigger": 2.0, "volume": 2.0, "rs": 1.0},
    "swing": {"trend": 1.5, "location": 2.0, "cost_basis": 1.0, "trigger": 2.0, "volume": 1.0, "rs": 1.5},
    "trend": {"trend": 2.0, "location": 1.5, "cost_basis": 1.0, "trigger": 1.0, "volume": 1.5, "rs": 2.0},
    "position": {"trend": 2.0, "location": 1.0, "cost_basis": 1.0, "trigger": 1.0, "volume": 0.5, "rs": 2.0},
}

ETF_KEYWORDS = [
    "ETF",
    "Fund",
    "Trust",
    "Index",
    "iShares",
    "SPDR",
    "Vanguard",
    "Invesco",
    "ProShares",
    "Direxion",
]

MIN_HISTORY_BUFFER = 5
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]


# =========================
# Helpers
# =========================

def clean_tickers(raw: str) -> List[str]:
    tickers = []
    for item in raw.replace("\n", ",").split(","):
        ticker = item.strip().upper()
        if ticker:
            tickers.append(ticker)
    return list(dict.fromkeys(tickers))


def classify_ticker_type(ticker: str, info: Optional[dict] = None) -> str:
    if info:
        quote_type = str(info.get("quoteType", "")).upper()
        long_name = str(info.get("longName", ""))
        short_name = str(info.get("shortName", ""))

        if quote_type in {"ETF", "MUTUALFUND"}:
            return "ETF"

        combined_name = f"{long_name} {short_name}"
        if any(word.lower() in combined_name.lower() for word in ETF_KEYWORDS):
            return "ETF"

    return "Stock"


def true_range(df: pd.DataFrame) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - df["Close"].shift(1)).abs()
    low_prev_close = (df["Low"] - df["Close"].shift(1)).abs()

    return pd.concat(
        [high_low, high_prev_close, low_prev_close],
        axis=1,
    ).max(axis=1)


def wilder_atr(df: pd.DataFrame, atr_window: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / atr_window, adjust=False).mean()


def classify_regime_from_ratio(ratio: float) -> str:
    if pd.isna(ratio):
        return "Normal"
    if ratio < 0.75:
        return "Low"
    if ratio > 1.50:
        return "High"
    return "Normal"


def regime_score(label: str) -> int:
    return {"Low": 0, "Normal": 1, "High": 2}.get(str(label), 1)


def score_to_regime(score: float) -> str:
    if score <= 0.75:
        return "Low"
    if score > 1.50:
        return "High"
    return "Normal"


def wilder_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_entry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add direction/strategy-independent entry context columns."""
    atr_safe = df["ATR"].where(df["ATR"] > 0)

    df["Dist_MA50_ATR"] = (df["Close"] - df["MA50"]) / atr_safe
    df["Dist_MA200_ATR"] = (df["Close"] - df["MA200"]) / atr_safe
    df["Dist_VWAP50_ATR"] = (df["Close"] - df["VWAP50"]) / atr_safe

    close, ma50, ma200 = df["Close"], df["MA50"], df["MA200"]
    has_ma200 = ma200.notna()
    df["Trend_Alignment"] = np.select(
        [
            has_ma200 & (close > ma50) & (ma50 > ma200),
            has_ma200 & (close < ma50) & (ma50 < ma200),
            ~has_ma200 & (close > ma50),
            ~has_ma200 & (close < ma50),
        ],
        ["Aligned Long", "Aligned Short", "Aligned Long*", "Aligned Short*"],
        default="Mixed",
    )

    vwap = df["VWAP50"]
    df["CostBasis_Flag"] = np.select(
        [vwap.isna(), (close > ma50) & (close < vwap)],
        ["N/A", "Overhead Supply"],
        default="Clean",
    )

    df["RSI"] = wilder_rsi(df["Close"])

    if "Volume" in df.columns:
        avg_volume = df["Volume"].rolling(20).mean()
        df["RelVolume"] = df["Volume"] / avg_volume.replace(0, np.nan)
    else:
        df["RelVolume"] = np.nan

    return df


def add_relative_strength(df: pd.DataFrame, benchmark: str = BENCHMARK) -> pd.DataFrame:
    """Join a benchmark series and compute the relative-strength line and slope."""
    bench = download_price_history(benchmark, period="2y")
    if not bench.empty:
        bench_close = bench[["Close"]].rename(columns={"Close": "BENCH_Close"})
        df = df.join(bench_close, how="left")
        df["BENCH_Close"] = df["BENCH_Close"].ffill()
        df["RS_Ratio"] = df["Close"] / df["BENCH_Close"].replace(0, np.nan)
        df["RS_Slope"] = df["RS_Ratio"].diff(21)
    else:
        df["BENCH_Close"] = np.nan
        df["RS_Ratio"] = np.nan
        df["RS_Slope"] = np.nan
    return df


def _classify_location_series(dist: pd.Series, direction: str, cfg: Dict[str, float]) -> np.ndarray:
    d = dist if direction == "long" else -dist
    return np.select(
        [d.isna(), d < -cfg["support"], d <= cfg["support"], d <= cfg["near"], d <= cfg["ext"]],
        ["Neutral", "Below Support", "At Support", "Near", "Neutral"],
        default="Extended",
    )


def _classify_rsi_series(rsi: pd.Series, direction: str, cfg: Dict[str, float]) -> np.ndarray:
    lo, hi = cfg["rsi_lo"], cfg["rsi_hi"]
    if direction == "long":
        rising = rsi > rsi.shift(1)
        return np.select(
            [rsi.isna(), rsi >= 70, rsi < lo, (rsi >= lo) & (rsi <= hi) & rising],
            ["Neutral", "Overbought", "Oversold", "Resetting Up"],
            default="Neutral",
        )
    falling = rsi < rsi.shift(1)
    return np.select(
        [rsi.isna(), rsi <= 30, rsi > 70, (rsi >= (100 - hi)) & (rsi <= (100 - lo)) & falling],
        ["Neutral", "Oversold", "Overbought", "Resetting Down"],
        default="Neutral",
    )


def apply_strategy_entry_features(
    df: pd.DataFrame, strategy_type: str, direction: str
) -> pd.DataFrame:
    """Add direction/strategy-dependent entry columns to a (copied) DataFrame."""
    cfg = ENTRY_CONFIG[strategy_type]
    direction = direction.lower()

    df["Location_State"] = _classify_location_series(df["Dist_MA50_ATR"], direction, cfg)
    df["RSI_State"] = _classify_rsi_series(df["RSI"], direction, cfg)

    if "BENCH_Close" in df.columns and df["BENCH_Close"].notna().any():
        n = int(cfg["rs_lookback"])
        df["RS_Return_Rel"] = (
            df["Close"].pct_change(n) - df["BENCH_Close"].pct_change(n)
        ) * 100
    else:
        df["RS_Return_Rel"] = np.nan

    rr, slope = df["RS_Return_Rel"], df.get("RS_Slope", pd.Series(np.nan, index=df.index))
    df["RS_Read"] = np.select(
        [
            rr.isna(),
            (rr > 0) & (slope > 0),
            (rr > 0) & (slope <= 0),
            (rr <= 0) & (slope > 0),
        ],
        ["N/A", "Leader", "Cooling Leader", "Improving Laggard"],
        default="Laggard",
    )

    df["Vol_Confirm"] = df["RelVolume"] >= cfg["vol_mult"]

    aligned_set = (
        ["Aligned Long", "Aligned Long*"]
        if direction == "long"
        else ["Aligned Short", "Aligned Short*"]
    )
    rsi_fire = (
        ["Oversold", "Resetting Up"]
        if direction == "long"
        else ["Overbought", "Resetting Down"]
    )
    df["Entry_Trigger"] = (
        df["Trend_Alignment"].isin(aligned_set)
        & df["Location_State"].isin(["At Support", "Near"])
        & df["RSI_State"].isin(rsi_fire)
    )

    return df


def compute_target(
    entry_price: float,
    atr_value: float,
    direction: str,
    strategy_type: str,
    df: pd.DataFrame,
    method: str = "structure",
) -> dict:
    """Estimate a target price from recent structure or an ATR multiple."""
    cfg = ENTRY_CONFIG[strategy_type]
    direction = direction.lower()
    n = int(cfg["target_n"])
    reward_atr = cfg["reward_atr"]

    atr_target = (
        entry_price + reward_atr * atr_value
        if direction == "long"
        else entry_price - reward_atr * atr_value
    )

    target_price = atr_target
    target_method = "atr"

    if method == "structure" and len(df) > n:
        window = df.iloc[-(n + 1):-1]
        if direction == "long":
            level = window["High"].max()
            if not pd.isna(level) and level > entry_price:
                target_price, target_method = float(level), "structure"
        else:
            level = window["Low"].min()
            if not pd.isna(level) and level < entry_price:
                target_price, target_method = float(level), "structure"

    return {
        "target_price": float(target_price),
        "target_method": target_method,
        "reward_distance": abs(float(target_price) - entry_price),
    }


def build_trade_plan(
    planned_entry: float,
    stop_distance: float,
    direction: str,
    target_price: float,
) -> dict:
    """Recompute the stop/target/reward:risk plan at a chosen entry price."""
    direction = direction.lower()
    stop_price = (
        planned_entry - stop_distance
        if direction == "long"
        else planned_entry + stop_distance
    )
    risk_distance = abs(stop_distance)
    reward_distance = abs(target_price - planned_entry)
    reward_to_risk = reward_distance / risk_distance if risk_distance > 0 else np.nan
    risk_pct = (stop_distance / planned_entry) * 100 if planned_entry > 0 else np.nan

    return {
        "Planned Entry": round(planned_entry, 2),
        "Stop Price": round(stop_price, 2),
        "Stop Distance": round(stop_distance, 2),
        "Target": round(target_price, 2),
        "Risk %": round(risk_pct, 2) if not pd.isna(risk_pct) else np.nan,
        "Reward:Risk": round(reward_to_risk, 2) if not pd.isna(reward_to_risk) else np.nan,
    }


def _trend_subscore(ta: str, is_long: bool) -> float:
    exact = "Aligned Long" if is_long else "Aligned Short"
    soft = "Aligned Long*" if is_long else "Aligned Short*"
    if ta == exact:
        return 1.0
    if ta == soft:
        return 0.6
    if ta == "Mixed":
        return 0.3
    return 0.0


def _location_subscore(loc: str) -> float:
    return {
        "At Support": 1.0,
        "Near": 0.8,
        "Neutral": 0.4,
        "Below Support": 0.2,
        "Extended": 0.1,
    }.get(loc, 0.3)


def _trigger_subscore(rsi_state: str, is_long: bool) -> float:
    if is_long:
        return {"Resetting Up": 1.0, "Oversold": 0.7, "Neutral": 0.4, "Overbought": 0.1}.get(
            rsi_state, 0.4
        )
    return {"Resetting Down": 1.0, "Overbought": 0.7, "Neutral": 0.4, "Oversold": 0.1}.get(
        rsi_state, 0.4
    )


def _rs_subscore(rs_read: str) -> Optional[float]:
    if rs_read in ("N/A", "nan", "None"):
        return None
    return {"Leader": 1.0, "Improving Laggard": 0.7, "Cooling Leader": 0.5, "Laggard": 0.2}.get(
        rs_read, 0.4
    )


def grade_setup(summary: dict, strategy_type: str, direction: str) -> dict:
    """Compute the composite 0-100 Entry Score, its A/B/C grade, and components."""
    cfg = ENTRY_CONFIG[strategy_type]
    weights = ENTRY_WEIGHTS[strategy_type]
    is_long = direction.lower() == "long"
    benchmark = summary.get("benchmark", BENCHMARK)

    ta = str(summary.get("trend_alignment"))
    loc = str(summary.get("location_state"))
    cb = str(summary.get("cost_basis_flag"))
    rsi_state = str(summary.get("rsi_state"))
    rs_read = str(summary.get("rs_read"))
    vol_confirm = bool(summary.get("vol_confirm"))

    aligned_ok = ta in (
        ("Aligned Long", "Aligned Long*") if is_long else ("Aligned Short", "Aligned Short*")
    )
    loc_ok = loc in ("At Support", "Near")
    trigger_ok = rsi_state in (
        ("Oversold", "Resetting Up") if is_long else ("Overbought", "Resetting Down")
    )
    rs_ok = rs_read in ("Leader", "Improving Laggard")

    dist = summary.get("dist_ma50_atr")
    dist_str = f"{dist:+.1f} ATR vs MA50" if dist is not None and not pd.isna(dist) else "n/a"
    rsi_val = summary.get("rsi")
    rsi_str = f"{rsi_val:.0f}" if rsi_val is not None and not pd.isna(rsi_val) else "n/a"
    relv = summary.get("rel_volume")
    relv_str = f"{relv:.1f}x avg" if relv is not None and not pd.isna(relv) else "n/a"
    rsret = summary.get("rs_return_rel")
    rsret_str = (
        f"{rsret:+.1f}% vs {benchmark}" if rsret is not None and not pd.isna(rsret) else ""
    )

    # Component sub-scores (None => excluded from the score and renormalized)
    trend_sub = _trend_subscore(ta, is_long)
    loc_sub = _location_subscore(loc)
    cost_sub = None if cb == "N/A" else (1.0 if cb == "Clean" else 0.3)
    trig_sub = _trigger_subscore(rsi_state, is_long)
    if relv is None or pd.isna(relv):
        vol_sub = None
    else:
        vol_sub = float(np.clip(relv / cfg["vol_mult"], 0.0, 1.0))
    rs_sub = _rs_subscore(rs_read)

    components = []

    def add(factor, value, read, weight, sub):
        components.append(
            {
                "Factor": factor,
                "Value": value,
                "Read": read,
                "Weight": weight,
                "Score": round(sub * 100) if sub is not None else None,
                "_w": weight,
                "_sub": sub,
            }
        )

    add("Trend alignment", ta, "Pass" if aligned_ok else "Fail", weights["trend"], trend_sub)
    add("Location", f"{loc} ({dist_str})", "Pass" if loc_ok else "Fail", weights["location"], loc_sub)
    if cost_sub is None:
        add("Cost basis", "N/A", "N/A", weights["cost_basis"], None)
    else:
        add("Cost basis", cb, "Pass" if cb == "Clean" else "Fail", weights["cost_basis"], cost_sub)
    add("Trigger (RSI)", f"{rsi_state} ({rsi_str})", "Pass" if trigger_ok else "Fail", weights["trigger"], trig_sub)
    if vol_sub is None:
        add("Rel volume", "N/A", "N/A", weights["volume"], None)
    else:
        vol_read = "Pass" if vol_confirm else ("Neutral" if not cfg["vol_required"] else "Fail")
        add("Rel volume", relv_str, vol_read, weights["volume"], vol_sub)
    if rs_sub is None:
        add("Rel strength", "N/A", "N/A", weights["rs"], None)
    else:
        rs_value = f"{rs_read} ({rsret_str})" if rsret_str else rs_read
        add("Rel strength", rs_value, "Pass" if rs_ok else "Fail", weights["rs"], rs_sub)

    scored = [c for c in components if c["_sub"] is not None]
    total_w = sum(c["_w"] for c in scored)
    score = (
        round(100.0 * sum(c["_w"] * c["_sub"] for c in scored) / total_w, 1)
        if total_w > 0
        else np.nan
    )

    if not pd.isna(score) and score >= 75:
        grade = "A"
    elif not pd.isna(score) and score >= 55:
        grade = "B"
    else:
        grade = "C"

    reason_parts = [ta, f"{loc} location", f"RSI {rsi_state}"]
    if rs_read not in ("N/A", "nan", "None"):
        reason_parts.append(f"{rs_read} RS")
    reason = " · ".join(reason_parts)

    return {
        "grade": grade,
        "score": score,
        "reason": reason,
        "components": [
            {k: v for k, v in c.items() if not k.startswith("_")} for c in components
        ],
    }


def backtest_triggers(
    df: pd.DataFrame, strategy_type: str, direction: str
) -> Tuple[dict, pd.DataFrame]:
    """Replay historical entry triggers to estimate hit-rate and average R.

    For each historical bar where ``Entry_Trigger`` fired, simulate a trade entered at
    that bar's close with the same ATR stop distance used by the stop engine and an
    ATR-multiple target. Walk forward up to the strategy's holding window and record
    whether the target (win) or stop (loss) was hit first. Bars where neither is hit by
    the end of the window are marked ``Open`` and excluded from the win rate. This is a
    look-ahead-safe what-if (the signal uses only data up to the entry bar), not a
    trading simulation: it ignores slippage, gaps mid-bar, and overlapping positions.
    """
    df = df.copy()
    df["Trigger_Outcome"] = ""
    df["Trigger_R"] = np.nan

    empty_stats = {
        "signals": 0, "wins": 0, "losses": 0, "open": 0,
        "win_rate": np.nan, "avg_r": np.nan, "expectancy": np.nan,
        "hold_bars": int(ENTRY_CONFIG[strategy_type]["target_n"]),
        "reward_atr": ENTRY_CONFIG[strategy_type]["reward_atr"],
    }
    if "Entry_Trigger" not in df.columns or "Stop Distance" not in df.columns:
        return empty_stats, df

    cfg = ENTRY_CONFIG[strategy_type]
    reward_atr = cfg["reward_atr"]
    hold = int(cfg["target_n"])
    is_long = direction.lower() == "long"

    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    atr = df["ATR"].to_numpy(dtype=float)
    stopd = df["Stop Distance"].to_numpy(dtype=float)
    trig = df["Entry_Trigger"].fillna(False).to_numpy(dtype=bool)
    n = len(df)

    outcomes = np.array([""] * n, dtype=object)
    realized = np.full(n, np.nan)

    for i in range(n):
        if not trig[i]:
            continue
        risk, a, entry = stopd[i], atr[i], close[i]
        if not (np.isfinite(risk) and risk > 0 and np.isfinite(a) and a > 0):
            continue
        reward = reward_atr * a
        if is_long:
            target, stop = entry + reward, entry - risk
        else:
            target, stop = entry - reward, entry + risk

        end = min(i + hold, n - 1)
        outcome, r_val = None, None
        for j in range(i + 1, end + 1):
            if is_long:
                hit_stop, hit_tgt = low[j] <= stop, high[j] >= target
            else:
                hit_stop, hit_tgt = high[j] >= stop, low[j] <= target
            if hit_stop:  # conservative: if both hit in one bar, assume stop first
                outcome, r_val = "Loss", -1.0
                break
            if hit_tgt:
                outcome, r_val = "Win", reward / risk
                break
        if outcome is None:
            outcome = "Open"
            r_val = ((close[end] - entry) if is_long else (entry - close[end])) / risk

        outcomes[i], realized[i] = outcome, r_val

    df["Trigger_Outcome"] = outcomes
    df["Trigger_R"] = realized

    signal_mask = outcomes != ""
    all_r = realized[signal_mask]
    wins = int((outcomes == "Win").sum())
    losses = int((outcomes == "Loss").sum())
    opens = int((outcomes == "Open").sum())
    closed = wins + losses
    closed_r = realized[(outcomes == "Win") | (outcomes == "Loss")]

    stats = {
        "signals": int(signal_mask.sum()),
        "wins": wins,
        "losses": losses,
        "open": opens,
        "win_rate": (wins / closed * 100) if closed else np.nan,
        "avg_r": float(np.nanmean(closed_r)) if closed_r.size else np.nan,
        "expectancy": float(np.nanmean(all_r)) if all_r.size else np.nan,
        "hold_bars": hold,
        "reward_atr": reward_atr,
    }
    return stats, df


def close_mobile_sidebar() -> None:
    components.html(
        """
        <script>
        (() => {
            let clicked = false;

            const closeSidebar = () => {
                if (clicked) {
                    return true;
                }

                const parentDoc = window.parent.document;
                const isMobile = window.parent.innerWidth <= 768;
                if (!isMobile) {
                    return false;
                }

                const selectors = [
                    'button[aria-label="Close sidebar"]',
                    'button[title="Close sidebar"]',
                    'button[aria-label="Collapse sidebar"]',
                    'button[title="Collapse sidebar"]',
                    '[data-testid="stSidebarCollapseButton"] button',
                    '[data-testid="stSidebarCollapseButton"]',
                    '[data-testid="stSidebar"] button'
                ];

                for (const selector of selectors) {
                    const control = parentDoc.querySelector(selector);
                    if (control) {
                        const rect = control.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            clicked = true;
                            control.click();
                            return true;
                        }
                    }
                }

                return false;
            };

            setTimeout(closeSidebar, 150);
            setTimeout(closeSidebar, 400);
            setTimeout(closeSidebar, 900);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


@st.cache_data(ttl=900)
def download_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    periods = list(dict.fromkeys([period, "2y", "5y"]))

    for candidate_period in periods:
        df = yf.download(
            ticker,
            period=candidate_period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df.empty:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        missing_price_columns = [col for col in PRICE_COLUMNS if col not in df.columns]
        if missing_price_columns:
            continue

        available_columns = PRICE_COLUMNS + [col for col in ["Volume"] if col in df.columns]
        cleaned = df[available_columns].dropna(subset=PRICE_COLUMNS)
        if not cleaned.empty:
            return cleaned

    return pd.DataFrame()


@st.cache_data(ttl=3600)
def download_ticker_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


def calculate_volatility_indicators(
    ticker: str,
    atr_window: int = 14,
    regime_window: int = 50,
    bb_window: int = 20,
    use_vix: bool = True,
    benchmark: str = BENCHMARK,
) -> Tuple[pd.DataFrame, dict]:
    # Fetch ~2 years so the 200-day MA has a full 200-bar lead-in before the most
    # recent ~1 year, letting both MA50 and MA200 render across the whole chart view.
    df = download_price_history(ticker, period="2y")
    min_required_rows = max(atr_window, regime_window, bb_window, 200) + MIN_HISTORY_BUFFER

    if df.empty or len(df) < min_required_rows:
        raise ValueError(
            f"Not enough price history found for {ticker}. "
            f"Need at least {min_required_rows} usable daily bars; got {len(df)}."
        )

    df = df.copy()

    # Trend context
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["Trend Strength"] = ((df["Close"] - df["MA50"]) / df["MA50"]) * 100
    df["Long-Term Trend"] = ((df["Close"] - df["MA200"]) / df["MA200"]) * 100
    if "Volume" in df.columns:
        rolling_volume = df["Volume"].rolling(50).sum()
        rolling_price_volume = (df["Close"] * df["Volume"]).rolling(50).sum()
        df["VWAP50"] = rolling_price_volume / rolling_volume.replace(0, np.nan)
        df["VWAP Strength"] = ((df["Close"] - df["VWAP50"]) / df["VWAP50"]) * 100
    else:
        df["VWAP50"] = np.nan
        df["VWAP Strength"] = np.nan

    # ATR regime
    df["TR"] = true_range(df)
    df["ATR"] = wilder_atr(df, atr_window=atr_window)
    df["ATR_Mean"] = df["ATR"].rolling(regime_window).mean()
    df["ATR_Ratio"] = df["ATR"] / df["ATR_Mean"]
    df["ATR_Regime"] = df["ATR_Ratio"].apply(classify_regime_from_ratio)

    # Bollinger Band Width regime
    df["BB_MA"] = df["Close"].rolling(bb_window).mean()
    df["BB_STD"] = df["Close"].rolling(bb_window).std()
    df["BB_Upper"] = df["BB_MA"] + 2 * df["BB_STD"]
    df["BB_Lower"] = df["BB_MA"] - 2 * df["BB_STD"]
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_MA"]
    df["BB_Width_Mean"] = df["BB_Width"].rolling(regime_window).mean()
    df["BB_Ratio"] = df["BB_Width"] / df["BB_Width_Mean"]
    df["BB_Regime"] = df["BB_Ratio"].apply(classify_regime_from_ratio)

    # VIX macro regime
    if use_vix:
        vix = download_price_history("^VIX", period="2y")
        if not vix.empty:
            vix_close = vix[["Close"]].rename(columns={"Close": "VIX"})
            df = df.join(vix_close, how="left")
            df["VIX"] = df["VIX"].ffill()
            df["VIX_Mean"] = df["VIX"].rolling(regime_window).mean()
            df["VIX_Ratio"] = df["VIX"] / df["VIX_Mean"]
            df["VIX_Regime"] = df["VIX_Ratio"].apply(classify_regime_from_ratio)
        else:
            df["VIX"] = np.nan
            df["VIX_Ratio"] = np.nan
            df["VIX_Regime"] = "Normal"
    else:
        df["VIX"] = np.nan
        df["VIX_Ratio"] = np.nan
        df["VIX_Regime"] = "Normal"

    # Combined regime
    df["Regime_Score"] = (
        df["ATR_Regime"].map(regime_score)
        + df["BB_Regime"].map(regime_score)
        + df["VIX_Regime"].map(regime_score)
    ) / 3.0

    df["Volatility_Regime"] = df["Regime_Score"].apply(score_to_regime)

    # Entry context (direction/strategy-independent) and relative strength
    df = add_entry_features(df)
    df = add_relative_strength(df, benchmark=benchmark)

    latest = df.dropna(subset=["Close", "ATR", "Volatility_Regime"]).iloc[-1]

    summary = {
        "ticker": ticker,
        "benchmark": benchmark,
        "entry_price": float(latest["Close"]),
        "atr": float(latest["ATR"]),
        "ma50": float(latest["MA50"]) if not pd.isna(latest["MA50"]) else np.nan,
        "ma200": float(latest["MA200"]) if not pd.isna(latest["MA200"]) else np.nan,
        "vwap50": float(latest["VWAP50"]) if not pd.isna(latest["VWAP50"]) else np.nan,
        "trend_strength": float(latest["Trend Strength"])
        if not pd.isna(latest["Trend Strength"])
        else np.nan,
        "long_term_trend": float(latest["Long-Term Trend"])
        if not pd.isna(latest["Long-Term Trend"])
        else np.nan,
        "vwap_strength": float(latest["VWAP Strength"])
        if not pd.isna(latest["VWAP Strength"])
        else np.nan,
        "atr_ratio": float(latest["ATR_Ratio"]) if not pd.isna(latest["ATR_Ratio"]) else np.nan,
        "atr_regime": str(latest["ATR_Regime"]),
        "bb_ratio": float(latest["BB_Ratio"]) if not pd.isna(latest["BB_Ratio"]) else np.nan,
        "bb_regime": str(latest["BB_Regime"]),
        "vix": float(latest["VIX"]) if "VIX" in latest and not pd.isna(latest["VIX"]) else np.nan,
        "vix_ratio": float(latest["VIX_Ratio"]) if "VIX_Ratio" in latest and not pd.isna(latest["VIX_Ratio"]) else np.nan,
        "vix_regime": str(latest["VIX_Regime"]),
        "volatility_regime": str(latest["Volatility_Regime"]),
        "regime_score": float(latest["Regime_Score"]),
        "date": latest.name,
    }

    return df, summary


def calculate_best_stop(
    entry_price: float,
    atr_value: float,
    volatility_regime: str,
    strategy_type: str,
    ticker: str,
    ticker_type: str,
    direction: str = "long",
    custom_multiplier: Optional[float] = None,
) -> dict:
    strategy_type = strategy_type.lower()
    direction = direction.lower()

    if strategy_type not in ATR_MULTIPLIERS:
        raise ValueError(f"Unsupported strategy type: {strategy_type}")

    if volatility_regime not in {"Low", "Normal", "High"}:
        volatility_regime = "Normal"

    multiplier = (
        float(custom_multiplier)
        if custom_multiplier is not None
        else ATR_MULTIPLIERS[strategy_type][volatility_regime]
    )

    stop_distance = atr_value * multiplier

    if direction == "long":
        stop_price = entry_price - stop_distance
    elif direction == "short":
        stop_price = entry_price + stop_distance
    else:
        raise ValueError("direction must be 'long' or 'short'.")

    return {
        "Ticker": ticker,
        "Type": ticker_type,
        "Direction": direction.capitalize(),
        "Strategy": STRATEGY_LABELS.get(strategy_type, strategy_type.capitalize()),
        "Regime": volatility_regime,
        "Entry Price": round(entry_price, 2),
        "ATR": round(atr_value, 2),
        "ATR Multiplier": round(multiplier, 2),
        "Stop Distance": round(stop_distance, 2),
        "Stop Price": round(stop_price, 2),
        "Risk % to Stop": round((stop_distance / entry_price) * 100, 2) if entry_price else np.nan,
    }


def build_stop_price_history(
    df: pd.DataFrame,
    strategy_type: str,
    direction: str,
    override_regime: Optional[str] = None,
    custom_multiplier: Optional[float] = None,
) -> pd.DataFrame:
    chart_df = df.copy()

    if custom_multiplier is not None:
        chart_df["ATR Multiplier"] = float(custom_multiplier)
    elif override_regime:
        chart_df["ATR Multiplier"] = ATR_MULTIPLIERS[strategy_type][override_regime]
    else:
        chart_df["ATR Multiplier"] = (
            chart_df["Volatility_Regime"]
            .map(ATR_MULTIPLIERS[strategy_type])
            .fillna(ATR_MULTIPLIERS[strategy_type]["Normal"])
        )

    chart_df["Stop Distance"] = chart_df["ATR"] * chart_df["ATR Multiplier"]
    if direction.lower() == "short":
        chart_df["Stop Price"] = chart_df["Close"] + chart_df["Stop Distance"]
    else:
        chart_df["Stop Price"] = chart_df["Close"] - chart_df["Stop Distance"]

    return chart_df


def calculate_position_size(
    account_size: float,
    risk_pct: float,
    stop_distance: float,
    entry_price: float,
    max_position_pct: Optional[float] = None,
) -> dict:
    risk_dollars = account_size * (risk_pct / 100)

    if stop_distance <= 0 or entry_price <= 0:
        return {
            "Risk $": np.nan,
            "Risk-Based Shares": np.nan,
            "Capital Cap Shares": np.nan,
            "Final Shares": np.nan,
            "Position Value": np.nan,
        }

    risk_based_shares = math.floor(risk_dollars / stop_distance)

    if max_position_pct is not None and max_position_pct > 0:
        max_position_value = account_size * (max_position_pct / 100)
        capital_cap_shares = math.floor(max_position_value / entry_price)
        final_shares = min(risk_based_shares, capital_cap_shares)
    else:
        capital_cap_shares = np.nan
        final_shares = risk_based_shares

    return {
        "Risk $": round(risk_dollars, 2),
        "Risk-Based Shares": risk_based_shares,
        "Capital Cap Shares": capital_cap_shares,
        "Final Shares": final_shares,
        "Position Value": round(final_shares * entry_price, 2),
    }


def generate_stop_for_ticker(
    ticker: str,
    strategy_type: str,
    direction: str,
    atr_window: int,
    regime_window: int,
    bb_window: int,
    use_vix: bool,
    override_regime: Optional[str] = None,
    custom_multiplier: Optional[float] = None,
    benchmark: str = BENCHMARK,
) -> Tuple[dict, pd.DataFrame, dict]:
    df, vol_summary = calculate_volatility_indicators(
        ticker=ticker,
        atr_window=atr_window,
        regime_window=regime_window,
        bb_window=bb_window,
        use_vix=use_vix,
        benchmark=benchmark,
    )

    info = download_ticker_info(ticker)
    ticker_type = classify_ticker_type(ticker, info)

    regime = override_regime if override_regime else vol_summary["volatility_regime"]
    df = build_stop_price_history(
        df=df,
        strategy_type=strategy_type,
        direction=direction,
        override_regime=override_regime,
        custom_multiplier=custom_multiplier,
    )
    df = apply_strategy_entry_features(df, strategy_type, direction)
    backtest_stats, df = backtest_triggers(df, strategy_type, direction)

    stop = calculate_best_stop(
        entry_price=vol_summary["entry_price"],
        atr_value=vol_summary["atr"],
        volatility_regime=regime,
        strategy_type=strategy_type,
        ticker=ticker,
        ticker_type=ticker_type,
        direction=direction,
        custom_multiplier=custom_multiplier,
    )

    stop.update(
        {
            "Data Date": pd.to_datetime(vol_summary["date"]).strftime("%Y-%m-%d"),
            "MA50": round(vol_summary["ma50"], 2) if not pd.isna(vol_summary["ma50"]) else np.nan,
            "MA200": round(vol_summary["ma200"], 2) if not pd.isna(vol_summary["ma200"]) else np.nan,
            "VWAP50": round(vol_summary["vwap50"], 2) if not pd.isna(vol_summary["vwap50"]) else np.nan,
            "Trend Strength": round(vol_summary["trend_strength"], 2)
            if not pd.isna(vol_summary["trend_strength"])
            else np.nan,
            "Long-Term Trend": round(vol_summary["long_term_trend"], 2)
            if not pd.isna(vol_summary["long_term_trend"])
            else np.nan,
            "VWAP Strength": round(vol_summary["vwap_strength"], 2)
            if not pd.isna(vol_summary["vwap_strength"])
            else np.nan,
            "ATR Regime": vol_summary["atr_regime"],
            "BB Regime": vol_summary["bb_regime"],
            "VIX Regime": vol_summary["vix_regime"],
            "Regime Score": round(vol_summary["regime_score"], 2),
        }
    )

    # Fold latest-bar entry context into the summary and grade the setup
    entry_latest = df.dropna(subset=["Close"]).iloc[-1]

    def _num(value):
        return float(value) if value is not None and not pd.isna(value) else np.nan

    vol_summary.update(
        {
            "strategy_type": strategy_type,
            "direction": direction,
            "dist_ma50_atr": _num(entry_latest.get("Dist_MA50_ATR")),
            "dist_vwap50_atr": _num(entry_latest.get("Dist_VWAP50_ATR")),
            "trend_alignment": str(entry_latest.get("Trend_Alignment")),
            "location_state": str(entry_latest.get("Location_State")),
            "cost_basis_flag": str(entry_latest.get("CostBasis_Flag")),
            "rsi": _num(entry_latest.get("RSI")),
            "rsi_state": str(entry_latest.get("RSI_State")),
            "rel_volume": _num(entry_latest.get("RelVolume")),
            "vol_confirm": bool(entry_latest.get("Vol_Confirm"))
            if not pd.isna(entry_latest.get("Vol_Confirm"))
            else False,
            "rs_return_rel": _num(entry_latest.get("RS_Return_Rel")),
            "rs_read": str(entry_latest.get("RS_Read")),
        }
    )

    vol_summary["grade"] = grade_setup(vol_summary, strategy_type, direction)
    vol_summary["backtest"] = backtest_stats

    # Default trade plan at the latest close (structure target) for the scanner/results
    default_target = compute_target(
        vol_summary["entry_price"], vol_summary["atr"], direction, strategy_type, df, "structure"
    )
    default_plan = build_trade_plan(
        vol_summary["entry_price"],
        float(stop["Stop Distance"]),
        direction,
        default_target["target_price"],
    )

    stop.update(
        {
            "Entry Score": vol_summary["grade"]["score"],
            "Setup Grade": vol_summary["grade"]["grade"],
            "Trend Alignment": vol_summary["trend_alignment"],
            "Location": vol_summary["location_state"],
            "RSI": round(vol_summary["rsi"], 1) if not pd.isna(vol_summary["rsi"]) else np.nan,
            "RSI State": vol_summary["rsi_state"],
            "Rel Strength": vol_summary["rs_read"],
            "Target": default_plan["Target"],
            "Reward:Risk": default_plan["Reward:Risk"],
            "Signal Win %": round(backtest_stats["win_rate"], 1)
            if not pd.isna(backtest_stats["win_rate"])
            else np.nan,
            "Signal Avg R": round(backtest_stats["avg_r"], 2)
            if not pd.isna(backtest_stats["avg_r"])
            else np.nan,
        }
    )

    return stop, df, vol_summary


def render_entry_panel(
    selected: str,
    hist: pd.DataFrame,
    summary: dict,
    selected_result: pd.Series,
) -> None:
    """Render the Entry Panel: grade headline, trade plan, and component table."""
    strategy_type_panel = summary.get("strategy_type", "trend")
    direction_panel = summary.get("direction", "long")
    grade = summary.get("grade") or grade_setup(
        summary, strategy_type_panel, direction_panel
    )

    st.subheader(f"{selected} Entry Setup")
    st.caption(
        "Completes the trade plan with entry trigger, location, target, and reward:risk. "
        "Setup context for education only — not a trade recommendation."
    )

    grade_letter = grade["grade"]
    score = grade.get("score", np.nan)
    score_str = f"{score:.0f}/100" if not pd.isna(score) else "n/a"
    grade_colors = {"A": "#1a7f37", "B": "#9a6700", "C": "#6e7781"}
    color = grade_colors.get(grade_letter, "#6e7781")
    st.markdown(
        f"<div style='font-size:1.05rem;margin-bottom:0.5rem'>"
        f"<span style='background:{color};color:white;padding:2px 12px;border-radius:6px;"
        f"font-weight:700;margin-right:10px'>{grade_letter}</span>"
        f"<span style='font-weight:700;margin-right:10px'>Entry Score {score_str}</span>"
        f"<span style='color:#444'>{grade['reason']}</span></div>",
        unsafe_allow_html=True,
    )

    ctrl = st.columns(2)
    default_entry = float(summary["entry_price"])
    step = max(0.01, round(default_entry * 0.001, 2))
    planned_entry = ctrl[0].number_input(
        "Planned entry price",
        min_value=0.0,
        value=default_entry,
        step=step,
        key=f"planned_entry_{selected}",
        help="Model the plan at a chosen entry. Defaults to the latest close.",
    )
    target_method = ctrl[1].selectbox(
        "Target method",
        options=["structure", "atr"],
        format_func=lambda m: "Structure (swing level)" if m == "structure" else "ATR multiple",
        key=f"target_method_{selected}",
        help="Structure uses the nearest recent swing high/low; ATR uses a fixed ATR multiple.",
    )

    if planned_entry <= 0:
        planned_entry = default_entry
        st.warning("Planned entry must be positive; using the latest close instead.")

    atr_value = float(summary["atr"])
    stop_distance = float(selected_result["Stop Distance"])
    tgt = compute_target(
        planned_entry, atr_value, direction_panel, strategy_type_panel, hist, method=target_method
    )
    plan = build_trade_plan(planned_entry, stop_distance, direction_panel, tgt["target_price"])

    shares = None
    if enable_position_sizing and account_size and risk_pct:
        sizing = calculate_position_size(
            account_size=float(account_size),
            risk_pct=float(risk_pct),
            stop_distance=stop_distance,
            entry_price=planned_entry,
            max_position_pct=max_position_pct,
        )
        shares = sizing["Final Shares"]

    plan_cols = st.columns(6 if shares is not None else 5)
    plan_cols[0].metric("Planned Entry", f"${plan['Planned Entry']:.2f}")
    plan_cols[1].metric(
        "Stop",
        f"${plan['Stop Price']:.2f}",
        help="Stop price at the planned entry (stop distance is volatility-based and fixed).",
    )
    plan_cols[2].metric(
        "Target",
        f"${plan['Target']:.2f}",
        help=f"{tgt['target_method'].capitalize()} target.",
    )
    rr = plan["Reward:Risk"]
    plan_cols[3].metric(
        "Reward:Risk",
        f"{rr:.2f}R" if not pd.isna(rr) else "N/A",
        help="Distance to target divided by stop distance.",
    )
    risk_pct_val = plan["Risk %"]
    plan_cols[4].metric(
        "Risk %",
        f"{risk_pct_val:.2f}%" if not pd.isna(risk_pct_val) else "N/A",
        help="How far price can move against the planned entry before the stop is hit.",
    )
    if shares is not None:
        plan_cols[5].metric(
            "Shares",
            f"{int(shares):,}" if not pd.isna(shares) else "N/A",
            help="Risk-based share count at the planned entry.",
        )

    if not pd.isna(rr) and rr < 1:
        st.warning(
            "Reward:Risk is below 1R — limited upside to the nearest target relative to "
            "stop risk. Consider a different entry or target."
        )

    st.caption("Setup components (Score is each factor's 0-100 sub-score; Weight is its share of the Entry Score)")
    comp_df = pd.DataFrame(grade["components"])
    st.dataframe(comp_df, width="stretch", hide_index=True)

    backtest = summary.get("backtest")
    if backtest and backtest.get("signals", 0) > 0:
        st.caption(
            f"{selected} historical entry-trigger replay — entered at each trigger's close "
            f"with the same ATR stop and a {backtest['reward_atr']:.1f}×ATR target, held up "
            f"to {backtest['hold_bars']} bars. Look-ahead-safe what-if, not a trading "
            f"simulation (ignores slippage, gaps, and overlapping positions)."
        )
        bt_cols = st.columns(5)
        bt_cols[0].metric(
            "Signals",
            f"{backtest['signals']}",
            help="Historical bars where trend, location, and the RSI trigger all fired.",
        )
        bt_cols[1].metric(
            "Win rate",
            f"{backtest['win_rate']:.0f}%" if not pd.isna(backtest["win_rate"]) else "N/A",
            help="Wins / (wins + losses). Open trades that never hit target or stop are excluded.",
        )
        bt_cols[2].metric(
            "Avg R (closed)",
            f"{backtest['avg_r']:+.2f}R" if not pd.isna(backtest["avg_r"]) else "N/A",
            help="Average realized R across closed trades (win = +reward:risk, loss = -1R).",
        )
        bt_cols[3].metric(
            "Expectancy",
            f"{backtest['expectancy']:+.2f}R" if not pd.isna(backtest["expectancy"]) else "N/A",
            help="Average R across all signals, including open trades marked to market.",
        )
        bt_cols[4].metric(
            "W / L / Open",
            f"{backtest['wins']} / {backtest['losses']} / {backtest['open']}",
            help="Counts of winning, losing, and still-open simulated trades.",
        )
    else:
        st.caption("No historical entry triggers in the available window to backtest.")

    with st.expander("How the Entry Panel works", expanded=False):
        st.markdown(
            """
            The Entry Panel adds **entry timing** context on top of the existing stop/risk engine,
            so a full trade plan reads as `entry → stop → target → reward:risk → size`.

            - **Entry Score (0-100)** is a weighted blend of direction-aware sub-scores: trend
              alignment, location vs. support (in ATR), cost basis, the RSI trigger, relative
              volume, and relative strength vs. the benchmark. Each factor scores 0-100 and is
              weighted per strategy (e.g. trend leans on alignment and relative strength). Factors
              with missing data are excluded and the rest are renormalized. The **A/B/C grade** is
              derived from the score (A ≥ 75, B ≥ 55, else C).
            - **Location** measures distance from MA50 in ATR units — a shallow pullback toward
              rising support scores better than chasing an extended move.
            - **Trigger (RSI)** favors a pullback that is resetting back up (for longs) rather than
              a chase at overbought.
            - **Planned entry** lets you model the plan at any price; the stop *distance* stays
              fixed (it is volatility-based), while the stop price, risk %, and shares update.
            - **Target** is the nearest recent swing level (structure) or a fixed ATR multiple,
              and drives the reward:risk estimate.
            - **Signal replay** walks every historical trigger forward with the same ATR stop and
              an ATR-multiple target to estimate win rate, average R, and expectancy. It is a
              look-ahead-safe what-if for calibration only — it ignores slippage, intrabar gaps,
              and overlapping positions, so treat it as rough context, not a track record.

            All thresholds and weights are tuned per strategy. This is educational setup context,
            not a trade recommendation.
            """
        )


def render_help_reference() -> None:
    """Consolidated guides and reference tables, kept out of the working views."""
    st.subheader("How this calculator works")
    st.markdown(
        """
        This app estimates a stop-loss price by multiplying a ticker's Average True Range (ATR)
        by a strategy-specific multiplier. Wider stops are used for longer-horizon strategies
        and higher-volatility regimes.

        The volatility regime combines three signals: the ticker's ATR ratio, Bollinger Band
        width ratio, and an optional VIX macro overlay. Trend Strength compares the close with
        the 50-day moving average, and Long-Term Trend compares it with the 200-day moving average.
        The chart compares close, MA50, MA200, and stop price so you can see trend context and
        how much room the stop gives the trade.
        """
    )

    st.subheader("ATR multiplier table")
    st.caption(
        "The app uses this table unless you choose a custom ATR multiplier. "
        "Higher multipliers create wider stops."
    )
    multiplier_table = pd.DataFrame(ATR_MULTIPLIERS).T
    multiplier_table.index = [STRATEGY_LABELS[idx] for idx in multiplier_table.index]
    st.dataframe(multiplier_table, width="stretch")

    st.subheader("What does Risk % to Stop mean?")
    st.markdown(
        """
        **Risk % to Stop** answers: _How far can this position move against me before my stop
        is hit?_

        For a long trade, it is the percentage drop from entry to stop. For a short trade,
        it is the percentage rise from entry to stop. A larger value means the position has
        more room to move before the stop is hit, but each share carries more price risk.

        Example: if entry is `$100` and the stop is `$92`, the stop is `$8` away, so
        Risk % to Stop is `8%`. With position sizing enabled, the app uses that stop distance
        to estimate how many shares fit your selected risk budget.
        """
    )

    st.subheader("How to use the trend and VWAP metrics")
    st.markdown(
        """
        **Trend Strength** shows whether price is above or below the 50-day moving average.
        **Long-Term Trend** does the same against the 200-day moving average. Positive values
        confirm price is above those trend lines; very large positive values can also mean the
        move is extended.

        **VWAP Strength** compares price with the 50-day volume-weighted average price, which
        is a rough view of recent volume-weighted cost basis. If price is above both MA50 and
        VWAP50, trend and recent buyer cost basis are confirming each other. If price is above
        MA50 but below VWAP50, the simple trend may look constructive, but recent high-volume
        buyers may still be underwater, which can create overhead supply.

        Use these as context with ATR and Risk % to Stop: a strong trend with a reasonable
        stop distance is usually cleaner than a strong-looking trend that is very extended or
        sitting below its volume-weighted cost basis.
        """
    )

    with st.expander(
        "Full user guide: find better entries and maximize reward vs. risk", expanded=False
    ):
        st.markdown(
            """
            This app helps with the full trade plan — **where to enter, where to exit, and how much
            to risk** — while keeping you on the right side of reward vs. risk. Use it as a repeatable
            workflow:

            #### 1. Set your context (sidebar)
            - **Tickers:** paste a watchlist (commas or new lines).
            - **Trading strategy:** `Day` / `Swing` / `Trend` / `Position`. This sets stop width *and*
              tunes the entry thresholds (pullback bands, target distance, RSI reset band, volume and
              relative-strength expectations). Match it to how long you actually hold.
            - **Direction:** long or short. Every entry signal flips accordingly.
            - **Benchmark:** the index your relative-strength is measured against (`SPY` by default;
              use `QQQ` for tech-heavy names).
            - Leave the indicator/override defaults unless you have a reason to change them, then click
              **Calculate Stops**.

            #### 2. Triage with the Setup Scanner
            The scanner ranks your whole watchlist by setup quality. Start at the top:
            - Rank by **Entry Score** to find the highest-quality setups, or by **Reward:Risk** to find
              the most asymmetric ones.
            - Filter to **A/B grades** and set a **Min Reward:Risk** (e.g. 2.0) to hide weak ideas.
            - Only the names that survive triage are worth a closer look.

            #### 3. Drill into a candidate (Entry Plan)
            Pick a ticker and read its **Entry Score (0-100) and grade**. The component table shows
            *why*, factor by factor. The strongest setups usually line up like this:
            - **Trend alignment:** price stacked with MA50/MA200 in your direction.
            - **Location:** `At Support` or `Near` — a shallow pullback toward rising support beats
              chasing an `Extended` move (location is measured in ATR units).
            - **Trigger (RSI):** `Resetting Up` (longs) — momentum turning back your way, not an
              exhausted chase at overbought.
            - **Relative strength:** `Leader` or `Improving Laggard` vs. the benchmark.
            - **Cost basis / volume:** price above VWAP (clean) and, for breakout strategies, a volume
              surge confirming the move.

            #### 4. Build the trade plan
            - Set a **Planned entry** (the price you'd actually buy/short). The stop *distance* is
              volatility-based and stays fixed; the stop price, risk %, and shares update.
            - Choose a **target method** and read **Reward:Risk**. Favor setups offering at least your
              strategy's target R (≈2R for trend/position). If R:R is below 1, the app warns you —
              wait for a better entry or a closer support to lean on.

            #### 5. Sanity-check with the signal replay
            The backtest replays this ticker's historical triggers and reports **win rate, average R,
            and expectancy**. Prefer setups with **positive expectancy**. Treat it as rough calibration
            context — it ignores slippage, gaps, and overlapping trades, and the past is not a promise.

            #### 6. Size and manage the risk
            - Enable **position sizing** to convert your fixed risk-per-trade into a share count, so a
              stop-out only costs your planned percentage of the account.
            - The **stop price** and **Risk % to Stop** define your exit before you ever enter. Plan the
              exit first; the entry only earns its place if the reward justifies that risk.

            #### Quick checklist to maximize reward vs. risk
            1. Trade *with* the trend and a leader in relative strength.
            2. Enter near support on a resetting trigger — not extended, not chasing.
            3. Require an asymmetric Reward:Risk (ideally ≥ 2R) before committing.
            4. Confirm positive historical expectancy for that setup type.
            5. Size by a fixed risk %, and let the predefined ATR stop manage the downside.

            Everything here is educational decision support, not financial advice or a trade
            recommendation.
            """
        )


def render_scanner(result_df: pd.DataFrame) -> None:
    """Render the watchlist Setup Scanner plus the full results table and CSV export."""
    st.subheader("Setup Scanner")
    st.caption(
        "Ranks your watchlist by setup quality so you can focus on the best reward-vs-risk "
        "ideas first. Educational ranking only, not trade advice."
    )

    scan_controls = st.columns([2, 1, 1])
    sort_options = {
        "Entry Score": ("Entry Score", False),
        "Reward:Risk": ("Reward:Risk", False),
        "Signal Win %": ("Signal Win %", False),
        "Signal Avg R": ("Signal Avg R", False),
        "Risk % to Stop (tightest)": ("Risk % to Stop", True),
    }
    sort_choice = scan_controls[0].selectbox(
        "Rank by",
        options=list(sort_options.keys()),
        index=0,
        help="Choose the metric used to rank your watchlist.",
    )
    grade_filter = scan_controls[1].multiselect(
        "Grades",
        options=["A", "B", "C"],
        default=["A", "B", "C"],
        help="Show only setups with these grades.",
    )
    min_rr = scan_controls[2].number_input(
        "Min Reward:Risk",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.5,
        help="Hide setups whose default Reward:Risk is below this value. 0 disables the filter.",
    )

    sort_col, ascending = sort_options[sort_choice]
    scan_df = result_df.copy()
    if "Setup Grade" in scan_df.columns and grade_filter:
        scan_df = scan_df[scan_df["Setup Grade"].isin(grade_filter)]
    if min_rr > 0 and "Reward:Risk" in scan_df.columns:
        scan_df = scan_df[scan_df["Reward:Risk"].fillna(0) >= min_rr]

    if scan_df.empty:
        st.info("No setups match the current scanner filters.")
    else:
        if sort_col in scan_df.columns:
            scan_df = scan_df.sort_values(sort_col, ascending=ascending, na_position="last")
        scanner_cols = [
            c
            for c in [
                "Ticker", "Type", "Direction", "Setup Grade", "Entry Score",
                "Reward:Risk", "Risk % to Stop", "Trend Alignment", "Location",
                "RSI State", "Rel Strength", "Signal Win %", "Signal Avg R", "Regime",
            ]
            if c in scan_df.columns
        ]
        scan_df = scan_df[scanner_cols].reset_index(drop=True)
        scan_df.insert(0, "Rank", range(1, len(scan_df) + 1))
        st.dataframe(scan_df, width="stretch", hide_index=True)
        top = scan_df.iloc[0]
        st.caption(
            f"Top setup by {sort_choice}: **{top['Ticker']}** "
            f"(Grade {top.get('Setup Grade', 'n/a')}, "
            f"Entry Score {top.get('Entry Score', 'n/a')}, "
            f"Reward:Risk {top.get('Reward:Risk', 'n/a')})."
        )

    with st.expander("Full results table (all columns) and CSV export", expanded=False):
        st.dataframe(result_df, width="stretch")
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="atr_stop_results.csv",
            mime="text/csv",
        )


def render_ticker_detail(
    result_df: pd.DataFrame,
    available_tickers: List[str],
    direction: str,
) -> Tuple[str, pd.DataFrame, dict, pd.Series]:
    """Render the per-ticker chart and grouped Stop/Trend/Regime metrics."""
    selected = st.selectbox(
        "Ticker",
        options=available_tickers,
        key="selected_ticker",
        help="Switch between the tickers from the latest calculation without rerunning the data fetch.",
    )

    hist = st.session_state.history_by_ticker[selected].copy()
    summary = st.session_state.summaries_by_ticker[selected]
    selected_result = result_df[result_df["Ticker"] == selected].iloc[0]

    st.subheader(f"{selected} price, moving averages, and stop")
    st.caption(
        "MA50 and MA200 give trend context; the stop-price line is recalculated for each "
        "historical day from that day's close, ATR, direction, and settings."
    )
    show_triggers = st.checkbox(
        "Show entry-trigger markers",
        value=True,
        key=f"show_triggers_{selected}",
        help=(
            "Marks historical bars where trend alignment, location, and the RSI trigger all "
            "fired for the selected direction. Context only, not trade signals."
        ),
    )
    chart_df = hist[["Close", "MA50", "MA200", "Stop Price", "Stop Distance"]].dropna(
        subset=["Close", "Stop Price"]
    )
    # Start the visible window where the longest MA first has a value so MA50 and MA200
    # both span the entire chart instead of starting partway in and looking cut off.
    if chart_df["MA200"].notna().any():
        chart_df = chart_df.loc[chart_df["MA200"].first_valid_index():]
    elif chart_df["MA50"].notna().any():
        chart_df = chart_df.loc[chart_df["MA50"].first_valid_index():]
    visible_start = chart_df.index.min() if not chart_df.empty else None
    chart_df["Risk % to Stop"] = (chart_df["Stop Distance"] / chart_df["Close"]) * 100
    chart_df["Tooltip Close"] = chart_df["Close"]
    chart_df["Tooltip MA50"] = chart_df["MA50"]
    chart_df["Tooltip MA200"] = chart_df["MA200"]
    chart_df["Tooltip Stop Price"] = chart_df["Stop Price"]
    chart_data = (
        chart_df.reset_index(names="Date")
        .melt(
            id_vars=[
                "Date",
                "Tooltip Close",
                "Tooltip MA50",
                "Tooltip MA200",
                "Tooltip Stop Price",
                "Risk % to Stop",
            ],
            value_vars=["Close", "MA50", "MA200", "Stop Price"],
            var_name="Series",
            value_name="Price",
        )
        .dropna(subset=["Price"])
    )
    price_chart = (
        alt.Chart(chart_data)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Price:Q", title="Price"),
            color=alt.Color("Series:N", title="Series"),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip("Tooltip Close:Q", title="Close", format=",.2f"),
                alt.Tooltip("Tooltip MA50:Q", title="MA50", format=",.2f"),
                alt.Tooltip("Tooltip MA200:Q", title="MA200", format=",.2f"),
                alt.Tooltip("Tooltip Stop Price:Q", title="Stop Price", format=",.2f"),
                alt.Tooltip("Risk % to Stop:Q", title="Risk % to Stop", format=".2f"),
            ],
        )
        .properties(height=360)
    )

    chart_layers = price_chart
    if show_triggers and "Entry_Trigger" in hist.columns:
        has_outcome = "Trigger_Outcome" in hist.columns
        marker_cols = ["Close"] + (["Trigger_Outcome", "Trigger_R"] if has_outcome else [])
        trigger_mask = hist["Entry_Trigger"].fillna(False)
        if visible_start is not None:
            trigger_mask = trigger_mask & (hist.index >= visible_start)
        trigger_points = (
            hist[trigger_mask]
            .reset_index(names="Date")[["Date"] + marker_cols]
            .dropna(subset=["Close"])
        )
        if not trigger_points.empty:
            panel_direction = summary.get("direction", direction)
            marker_shape = "triangle-up" if panel_direction == "long" else "triangle-down"
            tooltip = [
                alt.Tooltip("Date:T", title="Trigger Date"),
                alt.Tooltip("Close:Q", title="Close", format=",.2f"),
            ]
            if has_outcome:
                trigger_points["Trigger_Outcome"] = trigger_points["Trigger_Outcome"].replace(
                    "", "Open"
                )
                tooltip += [
                    alt.Tooltip("Trigger_Outcome:N", title="Outcome"),
                    alt.Tooltip("Trigger_R:Q", title="Realized R", format="+.2f"),
                ]
                color_enc = alt.Color(
                    "Trigger_Outcome:N",
                    title="Trigger outcome",
                    scale=alt.Scale(
                        domain=["Win", "Loss", "Open"],
                        range=["#1a7f37", "#c1121f", "#6e7781"],
                    ),
                )
            else:
                color_enc = alt.value("#1a7f37")

            marker_layer = (
                alt.Chart(trigger_points)
                .mark_point(shape=marker_shape, size=90, filled=True, opacity=0.85)
                .encode(
                    x=alt.X("Date:T"),
                    y=alt.Y("Close:Q"),
                    color=color_enc,
                    tooltip=tooltip,
                )
            )
            chart_layers = alt.layer(price_chart, marker_layer)

    st.altair_chart(chart_layers, width="stretch")

    st.subheader("Stop & risk")
    risk_cols = st.columns(3)
    risk_cols[0].metric(
        "Stop Price",
        f"${selected_result['Stop Price']:.2f}",
        help="Calculated stop price for the selected ticker.",
    )
    risk_cols[1].metric(
        "Stop Distance",
        f"${selected_result['Stop Distance']:.2f}",
        help="Dollar distance between entry price and stop price.",
    )
    risk_cols[2].metric(
        "Risk % to Stop",
        f"{selected_result['Risk % to Stop']:.2f}%",
        help="How far this position can move against you before the stop is hit.",
    )

    st.subheader("Trend context")
    trend_cols = st.columns(3)
    trend_cols[0].metric(
        "Trend Strength",
        f"{summary['trend_strength']:.2f}%" if not pd.isna(summary["trend_strength"]) else "N/A",
        help="Current close compared with the 50-day moving average.",
    )
    trend_cols[1].metric(
        "Long-Term Trend",
        f"{summary['long_term_trend']:.2f}%" if not pd.isna(summary["long_term_trend"]) else "N/A",
        help="Current close compared with the 200-day moving average.",
    )
    trend_cols[2].metric(
        "VWAP Strength",
        f"{summary['vwap_strength']:.2f}%" if not pd.isna(summary["vwap_strength"]) else "N/A",
        help=(
            "Current close compared with the 50-day volume-weighted average price. "
            "Positive means price is above the recent volume-weighted cost basis."
        ),
    )

    st.subheader("Volatility regime")
    detail_cols = st.columns(4)
    detail_cols[0].metric(
        "Latest Close",
        f"${summary['entry_price']:.2f}",
        help="Most recent close from Yahoo Finance daily data.",
    )
    detail_cols[1].metric(
        "ATR",
        f"${summary['atr']:.2f}",
        help="Average True Range, a dollar estimate of recent daily price movement.",
    )
    detail_cols[2].metric(
        "Combined Regime",
        summary["volatility_regime"],
        help="Low, Normal, or High volatility classification from the combined regime score.",
    )
    detail_cols[3].metric(
        "Regime Score",
        f"{summary['regime_score']:.2f}",
        help="Average of ATR, Bollinger width, and VIX regime scores. Higher means more volatility.",
    )
    st.caption(
        "Ratios compare the current reading with its recent average. Low is below 0.75, "
        "High is above 1.50, and Normal is between those levels."
    )
    regime_detail = pd.DataFrame(
        [
            {
                "Signal": "ATR Ratio",
                "Value": round(summary["atr_ratio"], 3) if not pd.isna(summary["atr_ratio"]) else np.nan,
                "Regime": summary["atr_regime"],
            },
            {
                "Signal": "Bollinger Width Ratio",
                "Value": round(summary["bb_ratio"], 3) if not pd.isna(summary["bb_ratio"]) else np.nan,
                "Regime": summary["bb_regime"],
            },
            {
                "Signal": "VIX Ratio",
                "Value": round(summary["vix_ratio"], 3) if not pd.isna(summary["vix_ratio"]) else np.nan,
                "Regime": summary["vix_regime"],
            },
        ]
    )
    st.dataframe(regime_detail, width="stretch")

    return selected, hist, summary, selected_result


# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="ATR Stop Calculator",
    page_icon="",
    layout="wide",
)

st.title("ATR Stop Calculator")
st.caption("Regime-aware ATR stop-loss calculator for stocks and ETFs.")
st.markdown(
    "Enter a watchlist in the sidebar and click **Calculate Stops**. "
    "Results open in tabs: **Scanner** to triage, **Ticker Detail** for the chart and stop, "
    "**Entry Plan** for the deeper setup, and **Help & Reference** for the guide."
)

with st.sidebar:
    st.header("Inputs")

    raw_tickers = st.text_area(
        "Tickers",
        value="NVDA, VRT, SOXX, VGT",
        help="Enter one or more tickers separated by commas or line breaks.",
    )

    strategy_type = st.selectbox(
        "Trading strategy",
        options=["day", "swing", "trend", "position"],
        index=2,
        format_func=lambda x: STRATEGY_LABELS[x],
        help=(
            "Controls the default ATR multiplier. Shorter-term trades use tighter stops; "
            "longer-term trades use wider stops."
        ),
    )

    direction = st.radio(
        "Direction",
        options=["long", "short"],
        horizontal=True,
        help="Long stops are below the entry price. Short stops are above the entry price.",
    )

    st.divider()

    st.subheader("Indicator settings")
    st.caption("These settings control how volatility is measured from daily price history.")
    atr_window = st.number_input(
        "ATR window",
        min_value=5,
        max_value=100,
        value=14,
        step=1,
        help="Number of daily bars used for Average True Range. Higher values smooth the ATR.",
    )
    regime_window = st.number_input(
        "Regime lookback window",
        min_value=20,
        max_value=252,
        value=50,
        step=5,
        help="Lookback used to compare current ATR and Bollinger width against their recent averages.",
    )
    bb_window = st.number_input(
        "Bollinger window",
        min_value=10,
        max_value=100,
        value=20,
        step=1,
        help="Window used to calculate Bollinger Band width as a second volatility signal.",
    )
    use_vix = st.checkbox(
        "Use VIX macro overlay",
        value=True,
        help="Includes VIX volatility regime as a broad market risk signal.",
    )
    benchmark = st.selectbox(
        "Relative-strength benchmark",
        options=["SPY", "QQQ", "IWM", "DIA"],
        index=0,
        help="Benchmark used to measure the ticker's relative strength for entry context.",
    )

    st.divider()

    st.subheader("Overrides")
    regime_override_choice = st.selectbox(
        "Volatility regime override",
        options=["Auto", "Low", "Normal", "High"],
        index=0,
        help="Use Auto for the calculated regime, or force a Low/Normal/High regime manually.",
    )
    override_regime = None if regime_override_choice == "Auto" else regime_override_choice

    use_custom_multiplier = st.checkbox(
        "Use custom ATR multiplier",
        value=False,
        help="Override the strategy/regime table with one multiplier for every ticker.",
    )
    custom_multiplier = None
    if use_custom_multiplier:
        custom_multiplier = st.number_input(
            "Custom ATR multiplier",
            min_value=0.25,
            max_value=10.0,
            value=2.5,
            step=0.25,
            help="Stop distance equals ATR multiplied by this value.",
        )

    st.divider()

    st.subheader("Optional position sizing")
    enable_position_sizing = st.checkbox(
        "Calculate position size",
        value=False,
        help="Estimate shares from account size, risk per trade, and stop distance.",
    )
    account_size = risk_pct = max_position_pct = None

    if enable_position_sizing:
        account_size = st.number_input(
            "Account size ($)",
            min_value=100.0,
            value=100000.0,
            step=1000.0,
            help="Total account value used to estimate dollars at risk.",
        )
        risk_pct = st.number_input(
            "Risk per trade (%)",
            min_value=0.05,
            max_value=10.0,
            value=1.0,
            step=0.05,
            help="Percent of account value to risk if the stop is hit.",
        )
        max_position_pct = st.number_input(
            "Max position size (% of account)",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=1.0,
            help="Set to 0 to disable capital cap.",
        )
        if max_position_pct == 0:
            max_position_pct = None

    run_button = st.button("Calculate Stops", type="primary")


for key, initial_value in {
    "results": [],
    "history_by_ticker": {},
    "summaries_by_ticker": {},
    "errors": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = initial_value


if run_button:
    tickers = clean_tickers(raw_tickers)

    if not tickers:
        st.error("Please enter at least one ticker.")
        st.stop()

    results = []
    history_by_ticker = {}
    summaries_by_ticker = {}
    errors = []

    with st.spinner("Fetching market data and calculating stops..."):
        for ticker in tickers:
            try:
                stop, hist, summary = generate_stop_for_ticker(
                    ticker=ticker,
                    strategy_type=strategy_type,
                    direction=direction,
                    atr_window=int(atr_window),
                    regime_window=int(regime_window),
                    bb_window=int(bb_window),
                    use_vix=bool(use_vix),
                    override_regime=override_regime,
                    custom_multiplier=custom_multiplier,
                    benchmark=str(benchmark),
                )

                if enable_position_sizing:
                    sizing = calculate_position_size(
                        account_size=float(account_size),
                        risk_pct=float(risk_pct),
                        stop_distance=float(stop["Stop Distance"]),
                        entry_price=float(stop["Entry Price"]),
                        max_position_pct=max_position_pct,
                    )
                    stop.update(sizing)

                results.append(stop)
                history_by_ticker[ticker] = hist
                summaries_by_ticker[ticker] = summary

            except Exception as exc:
                errors.append({"Ticker": ticker, "Error": str(exc)})

    st.session_state.results = results
    st.session_state.history_by_ticker = history_by_ticker
    st.session_state.summaries_by_ticker = summaries_by_ticker
    st.session_state.errors = errors
    close_mobile_sidebar()

if st.session_state.results:
    result_df = pd.DataFrame(st.session_state.results)
    available_tickers = list(st.session_state.history_by_ticker.keys())
    if st.session_state.get("selected_ticker") not in available_tickers:
        st.session_state.selected_ticker = available_tickers[0]

    multiple = len(available_tickers) > 1

    # With several tickers, lead with the Scanner for triage. With a single ticker the
    # scanner is redundant, so land on Ticker Detail and push the scanner to the end.
    if multiple:
        scanner_tab, detail_tab, entry_tab, help_tab = st.tabs(
            ["Scanner", "Ticker Detail", "Entry Plan", "Help & Reference"]
        )
    else:
        detail_tab, entry_tab, scanner_tab, help_tab = st.tabs(
            ["Ticker Detail", "Entry Plan", "Scanner", "Help & Reference"]
        )

    # Render Ticker Detail first so the selected ticker and its context are available to
    # the Entry Plan tab regardless of tab display order.
    with detail_tab:
        selected, hist, summary, selected_result = render_ticker_detail(
            result_df, available_tickers, direction
        )

    with scanner_tab:
        render_scanner(result_df)

    with entry_tab:
        render_entry_panel(selected, hist, summary, selected_result)

    with help_tab:
        render_help_reference()

if st.session_state.errors:
    st.subheader("Errors")
    st.caption("These tickers could not be calculated from the available Yahoo Finance data.")
    st.dataframe(pd.DataFrame(st.session_state.errors), width="stretch")

if not run_button and not st.session_state.results:
    start_tab, help_tab = st.tabs(["Get started", "Help & Reference"])
    with start_tab:
        st.info("Enter tickers in the sidebar and click **Calculate Stops** to begin.")
        st.markdown(
            "Once you calculate, results open in tabs:\n\n"
            "- **Scanner** — rank your watchlist by setup quality and triage the best ideas.\n"
            "- **Ticker Detail** — the price/MA/stop chart plus grouped stop, trend, and regime metrics.\n"
            "- **Entry Plan** — the deeper entry score, target, reward:risk, and signal replay.\n"
            "- **Help & Reference** — the full guide, the ATR multiplier table, and metric definitions."
        )
    with help_tab:
        render_help_reference()

st.caption(
    "Educational tool only. Market data may be delayed or incomplete depending on Yahoo Finance availability."
)
