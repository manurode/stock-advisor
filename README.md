# 📈 Stock Advisor

AI-powered stock investment advisor for US stocks. Combines technical indicators, news sentiment analysis (NLP), and machine learning (XGBoost) to predict next-day stock direction with continuous backtesting.

> ⚠️ **Not financial advice.** This is a data analysis tool for educational purposes. Stock market prediction is inherently uncertain.

## Features

- **📊 Technical Analysis**: 34 indicators (RSI, MACD, Bollinger Bands, moving averages, ATR)
- **📰 News Sentiment**: VADER sentiment analysis on financial news (Bloomberg, Barrons, Motley Fool)
- **🤖 ML Predictions**: XGBoost model combining technical + sentiment features
- **🔬 Backtesting**: Walk-forward validation with accuracy, precision, recall metrics
- **📈 Dashboard**: Interactive Streamlit web UI for monitoring
- **💾 Data Caching**: Parquet/JSON caching to avoid API rate limits

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (first time)
python -m src.cli train

# Get today's predictions
python -m src.cli predict

# Check sentiment
python -m src.cli sentiment

# Run backtest for a stock
python -m src.cli backtest AAPL

# Launch dashboard
python -m src.cli dashboard
```

## Architecture

```
stock-advisor/
├── src/
│   ├── collectors/         # Data ingestion (yfinance)
│   │   ├── price.py        # OHLCV price data
│   │   └── news.py         # News articles
│   ├── features/           # Feature engineering
│   │   ├── technical.py    # 34 technical indicators
│   │   └── sentiment.py    # VADER sentiment
│   ├── models/             # ML pipeline
│   │   └── train.py        # XGBoost training & prediction
│   ├── backtest/           # Validation
│   │   └── engine.py       # Walk-forward backtesting
│   ├── dashboard/          # Web UI
│   │   └── app.py          # Streamlit dashboard
│   └── cli.py              # Command-line interface
├── spikes/                 # Feasibility experiments
├── tests/
└── data/                   # Cached data (gitignored)
```

## Data Sources

- **yfinance**: Free stock data from Yahoo Finance (no API key needed)
- **VADER**: Rule-based sentiment analysis (no API calls, runs locally)

## Tracked Stocks

Default watchlist (configurable in `src/config.py`):
`AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM, BAC, XOM, CVX, WMT, KO`

## Model Performance

Based on spike experiments (May 2026):
- Technical-only model: ~50% accuracy (no better than baseline)
- Sentiment pipeline: Working and differentiated across stocks
- **Combined model**: Expected improvement (in production validation)

## License

MIT — Use freely, invest wisely.
