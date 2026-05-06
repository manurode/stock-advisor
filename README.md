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

Training results with 40 features (34 technical + 6 market sentiment) on 3 years of data:

| Ticker | Accuracy | Baseline | Δ | F1 |
|--------|----------|----------|---|----|
| AAPL | 55.0% | 50.5% | +4.5% | 55.4% |
| JPM | 54.1% | 52.3% | +1.8% | 51.4% |
| NVDA | 48.6% | 52.3% | -3.6% | 46.7% |
| AMZN | 46.8% | 56.8% | -9.9% | 40.4% |
| MSFT | 45.9% | 52.3% | -6.3% | 58.3% |
| META | 45.0% | 50.5% | -5.4% | 45.0% |
| TSLA | 45.0% | 50.5% | -5.4% | 40.8% |
| GOOGL | 43.2% | 50.5% | -7.2% | 47.1% |

**Key findings:**
- AAPL (+4.5%) and JPM (+1.8%) beat baseline — mature companies show more predictable patterns
- High-growth tech (TSLA, NVDA, META) underperform — driven by narratives/news, not captured in training
- The real edge comes from **news sentiment at prediction time** (VADER blending)
- Use `python -m src.cli track` daily to accumulate accuracy data

## License

MIT — Use freely, invest wisely.
