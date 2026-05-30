# ATR Stop Calculator

Streamlit app for calculating regime-aware ATR stop-loss levels for stocks and ETFs.

The app is designed as an educational risk-management tool. It helps answer:

- How volatile is this ticker right now?
- What volatility regime is it in?
- Where would an ATR-based stop sit?
- How far can the position move against me before the stop is hit?
- Is price above or below its intermediate and long-term moving averages?

## Functionality

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
- `Trend Strength`: current close vs. 50-day moving average
- `Long-Term Trend`: current close vs. 200-day moving average

Positive trend values mean the latest close is above the moving average. Negative values mean
it is below the moving average.

### Chart

The selected ticker chart shows:

- Close price
- MA50
- MA200
- Stop price

Tooltips are disabled to avoid sticky mobile browser overlays.

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
