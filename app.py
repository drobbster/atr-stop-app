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
) -> Tuple[pd.DataFrame, dict]:
    df = download_price_history(ticker, period="1y")
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
        "ma50": float(latest["MA50"]) if not pd.isna(latest["MA50"]) else np.nan,
        "ma200": float(latest["MA200"]) if not pd.isna(latest["MA200"]) else np.nan,
        "trend_strength": float(latest["Trend Strength"])
        if not pd.isna(latest["Trend Strength"])
        else np.nan,
        "long_term_trend": float(latest["Long-Term Trend"])
        if not pd.isna(latest["Long-Term Trend"])
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
    df = build_stop_price_history(
        df=df,
        strategy_type=strategy_type,
        direction=direction,
        override_regime=override_regime,
        custom_multiplier=custom_multiplier,
    )

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
            "Trend Strength": round(vol_summary["trend_strength"], 2)
            if not pd.isna(vol_summary["trend_strength"])
            else np.nan,
            "Long-Term Trend": round(vol_summary["long_term_trend"], 2)
            if not pd.isna(vol_summary["long_term_trend"])
            else np.nan,
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

with st.expander("How this calculator works", expanded=False):
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


st.subheader("ATR Multiplier Table")
st.caption(
    "The app uses this table unless you choose a custom ATR multiplier. "
    "Higher multipliers create wider stops."
)

multiplier_table = pd.DataFrame(ATR_MULTIPLIERS).T
multiplier_table.index = [STRATEGY_LABELS[idx] for idx in multiplier_table.index]
st.dataframe(multiplier_table, width="stretch")


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
    st.subheader("Stop Results")
    st.caption(
        "Entry Price is the latest close. Stop Distance is ATR times the selected multiplier. "
        "Stop Price is Entry Price minus Stop Distance for longs, or plus Stop Distance for shorts. "
        "Risk % to Stop answers: how far can this position move against me before my stop is hit? "
        "Trend Strength and Long-Term Trend show percent above or below the 50-day and 200-day "
        "moving averages."
    )
    with st.expander("What does Risk % to Stop mean?", expanded=False):
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
    result_df = pd.DataFrame(st.session_state.results)
    st.dataframe(result_df, width="stretch")

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name="atr_stop_results.csv",
        mime="text/csv",
    )

    available_tickers = list(st.session_state.history_by_ticker.keys())
    if st.session_state.get("selected_ticker") not in available_tickers:
        st.session_state.selected_ticker = available_tickers[0]

    selected = st.selectbox(
        "View chart/details for ticker",
        options=available_tickers,
        key="selected_ticker",
        help="Switch between the tickers from the latest calculation without rerunning the data fetch.",
    )

    hist = st.session_state.history_by_ticker[selected].copy()
    summary = st.session_state.summaries_by_ticker[selected]
    selected_result = result_df[result_df["Ticker"] == selected].iloc[0]

    st.subheader(f"{selected} Close, Moving Averages, and Stop Price")
    st.caption(
        "MA50 and MA200 show intermediate and long-term trend context. The stop-price line is "
        "recalculated for each historical day using that day's close, ATR, direction, and selected "
        "multiplier/regime settings."
    )
    chart_df = hist[["Close", "MA50", "MA200", "Stop Price", "Stop Distance"]].dropna(
        subset=["Close", "Stop Price"]
    )
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
    st.altair_chart(price_chart, width="stretch")

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

    trend_cols = st.columns(2)
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

    st.subheader(f"{selected} Regime Details")
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

if st.session_state.errors:
    st.subheader("Errors")
    st.caption("These tickers could not be calculated from the available Yahoo Finance data.")
    st.dataframe(pd.DataFrame(st.session_state.errors), width="stretch")

if not run_button and not st.session_state.results:
    st.info("Enter tickers and click **Calculate Stops**.")

st.caption(
    "Educational tool only. Market data may be delayed or incomplete depending on Yahoo Finance availability."
)
