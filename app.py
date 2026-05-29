import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
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


@st.cache_data(ttl=900)
def download_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[col for col in required if col in df.columns]].dropna()

    return df


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
) -> Tuple[pd.DataFrame, dict]:
    df = download_price_history(ticker, period="1y")

    if df.empty or len(df) < max(atr_window, regime_window, bb_window) + 5:
        raise ValueError(f"Not enough data found for {ticker}.")

    df = df.copy()

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
        vix = download_price_history("^VIX", period="1y")
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

    latest = df.dropna(subset=["Close", "ATR", "Volatility_Regime"]).iloc[-1]

    summary = {
        "ticker": ticker,
        "entry_price": float(latest["Close"]),
        "atr": float(latest["ATR"]),
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
) -> Tuple[dict, pd.DataFrame, dict]:
    df, vol_summary = calculate_volatility_indicators(
        ticker=ticker,
        atr_window=atr_window,
        regime_window=regime_window,
        bb_window=bb_window,
        use_vix=use_vix,
    )

    info = download_ticker_info(ticker)
    ticker_type = classify_ticker_type(ticker, info)

    regime = override_regime if override_regime else vol_summary["volatility_regime"]

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
            "ATR Regime": vol_summary["atr_regime"],
            "BB Regime": vol_summary["bb_regime"],
            "VIX Regime": vol_summary["vix_regime"],
            "Regime Score": round(vol_summary["regime_score"], 2),
        }
    )

    return stop, df, vol_summary


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
    )

    direction = st.radio("Direction", options=["long", "short"], horizontal=True)

    st.divider()

    st.subheader("Indicator settings")
    atr_window = st.number_input("ATR window", min_value=5, max_value=100, value=14, step=1)
    regime_window = st.number_input("Regime lookback window", min_value=20, max_value=252, value=50, step=5)
    bb_window = st.number_input("Bollinger window", min_value=10, max_value=100, value=20, step=1)
    use_vix = st.checkbox("Use VIX macro overlay", value=True)

    st.divider()

    st.subheader("Overrides")
    regime_override_choice = st.selectbox(
        "Volatility regime override",
        options=["Auto", "Low", "Normal", "High"],
        index=0,
    )
    override_regime = None if regime_override_choice == "Auto" else regime_override_choice

    use_custom_multiplier = st.checkbox("Use custom ATR multiplier", value=False)
    custom_multiplier = None
    if use_custom_multiplier:
        custom_multiplier = st.number_input(
            "Custom ATR multiplier",
            min_value=0.25,
            max_value=10.0,
            value=2.5,
            step=0.25,
        )

    st.divider()

    st.subheader("Optional position sizing")
    enable_position_sizing = st.checkbox("Calculate position size", value=False)
    account_size = risk_pct = max_position_pct = None

    if enable_position_sizing:
        account_size = st.number_input("Account size ($)", min_value=100.0, value=100000.0, step=1000.0)
        risk_pct = st.number_input("Risk per trade (%)", min_value=0.05, max_value=10.0, value=1.0, step=0.05)
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


st.subheader("ATR Multiplier Table")

multiplier_table = pd.DataFrame(ATR_MULTIPLIERS).T
multiplier_table.index = [STRATEGY_LABELS[idx] for idx in multiplier_table.index]
st.dataframe(multiplier_table, use_container_width=True)


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

    if results:
        st.subheader("Stop Results")
        result_df = pd.DataFrame(results)
        st.dataframe(result_df, use_container_width=True)

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="atr_stop_results.csv",
            mime="text/csv",
        )

        selected = st.selectbox("View chart/details for ticker", options=list(history_by_ticker.keys()))

        hist = history_by_ticker[selected].copy()
        summary = summaries_by_ticker[selected]

        st.subheader(f"{selected} Price and ATR")
        chart_df = hist[["Close", "ATR"]].dropna()
        st.line_chart(chart_df)

        detail_cols = st.columns(4)
        detail_cols[0].metric("Latest Close", f"${summary['entry_price']:.2f}")
        detail_cols[1].metric("ATR", f"${summary['atr']:.2f}")
        detail_cols[2].metric("Combined Regime", summary["volatility_regime"])
        detail_cols[3].metric("Regime Score", f"{summary['regime_score']:.2f}")

        st.subheader(f"{selected} Regime Details")
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
        st.dataframe(regime_detail, use_container_width=True)

    if errors:
        st.subheader("Errors")
        st.dataframe(pd.DataFrame(errors), use_container_width=True)

else:
    st.info("Enter tickers and click **Calculate Stops**.")

st.caption(
    "Educational tool only. Market data may be delayed or incomplete depending on Yahoo Finance availability."
)
