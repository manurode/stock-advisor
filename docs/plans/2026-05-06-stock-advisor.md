# Stock Advisor — AI-Powered Investment Assistant

> **For Hermes:** Use writing-plans + subagent-driven-development skill to implement task-by-task.

**Goal:** Build an AI-powered stock investment advisor that combines technical indicators, news sentiment analysis, and ML predictions to assess whether a stock is a good buy/sell at any given moment, with daily backtesting to measure reliability.

**Architecture:** Modular Python pipeline: data collectors (yfinance) → feature engineering (technical + NLP) → XGBoost/LSTM model → backtesting engine → Streamlit dashboard. All orchestrated via CLI with daily paper trading tracking.

**Tech Stack:** Python 3.13, yfinance, pandas, numpy, scikit-learn, XGBoost, VADER/nltk, Streamlit, pytest

**Spikes (completed):** `/spikes/001-news-data` (VALIDATED), `/spikes/002-sentiment-correlation` (VALIDATED), `/spikes/003-baseline-model` (INVALIDATED — technical alone insufficient, needs sentiment)

---

## Project Structure

```
stock-advisor/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration (tickers, API keys, paths)
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── price.py           # yfinance price data
│   │   └── news.py            # yfinance news data
│   ├── features/
│   │   ├── __init__.py
│   │   ├── technical.py       # Technical indicators (RSI, MACD, etc.)
│   │   └── sentiment.py       # VADER sentiment analysis
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py           # Model training + hyperparameter tuning
│   │   └── predict.py         # Daily prediction generation
│   ├── backtest/
│   │   ├── __init__.py
│   │   └── engine.py          # Walk-forward backtesting
│   └── dashboard/
│       ├── __init__.py
│       └── app.py             # Streamlit dashboard
├── tests/
│   ├── __init__.py
│   ├── test_collectors.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_backtest.py
├── data/                      # Cached data (gitignored)
│   ├── prices/
│   └── news/
├── predictions/               # Daily prediction logs (gitignored)
├── spikes/                    # Spike experiment results
│   ├── 001-news-data/
│   ├── 002-sentiment-correlation/
│   └── 003-baseline-model/
├── requirements.txt
├── Makefile
├── .gitignore
└── README.md
```

---

## Implementation Phases

### Phase 1: Project Setup & Infrastructure

#### Task 1.1: Initialize project structure
**Objective:** Create all directories and base files

**Files:**
- Create: `src/__init__.py`, `src/collectors/__init__.py`, `src/features/__init__.py`, `src/models/__init__.py`, `src/backtest/__init__.py`, `src/dashboard/__init__.py`, `tests/__init__.py`

**Step 1: Create directory structure**
```bash
mkdir -p src/{collectors,features,models,backtest,dashboard} tests data/{prices,news} predictions
touch src/__init__.py src/collectors/__init__.py src/features/__init__.py src/models/__init__.py src/backtest/__init__.py src/dashboard/__init__.py tests/__init__.py
```

**Step 2: Commit**
```bash
git add -A && git commit -m "chore: initialize project structure"
```

#### Task 1.2: Create requirements.txt
**Objective:** Lock all project dependencies

**File:** `requirements.txt`

```txt
yfinance>=1.3.0
pandas>=3.0.0
numpy>=2.4.0
scikit-learn>=1.8.0
xgboost>=3.2.0
nltk>=3.9.0
vaderSentiment>=3.3.0
streamlit>=1.40.0
pytest>=8.0.0
```

#### Task 1.3: Create .gitignore
**File:** `.gitignore`
```gitignore
__pycache__/
*.pyc
.venv/
.env
data/prices/
data/news/
predictions/
.streamlit/
*.egg-info/
dist/
build/
```

#### Task 1.4: Create config module
**File:** `src/config.py`
```python
"""Central configuration for stock-advisor."""
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = ROOT / "data"
PRICES_DIR = DATA_DIR / "prices"
NEWS_DIR = DATA_DIR / "news"
PREDICTIONS_DIR = ROOT / "predictions"

# Default tickers to track
DEFAULT_TICKERS = ["AAPL", "TSLA", "MSFT", "AMZN", "GOOGL", "NVDA", "META", "JPM"]

# Model parameters
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

# Backtesting
TRAIN_SPLIT = 0.8
LOOKBACK_YEARS = 3

# Ensure data dirs exist
PRICES_DIR.mkdir(parents=True, exist_ok=True)
NEWS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
```

### Phase 2: Data Collection

#### Task 2.1: Price data collector
**Objective:** Fetch OHLCV data from yfinance with caching

**File:** `src/collectors/price.py`
- `fetch_prices(ticker, period="3y")` → DataFrame with OHLCV
- `get_current_price(ticker)` → float
- Cache to `data/prices/{ticker}.parquet`

#### Task 2.2: News data collector
**Objective:** Fetch news articles from yfinance

**File:** `src/collectors/news.py`
- `fetch_news(ticker, count=20)` → list of article dicts
- `get_recent_news(ticker, hours=48)` → filtered list
- `normalize_article(article)` → standardized dict with title, summary, source, timestamp, type

### Phase 3: Feature Engineering

#### Task 3.1: Technical indicators
**Objective:** Compute all technical indicators from OHLCV data

**File:** `src/features/technical.py`
- `compute_all_features(df)` → DataFrame with 30+ features
- Individual functions: `rsi()`, `macd()`, `bollinger()`, `atr()`, `sma_crossovers()`

#### Task 3.2: Sentiment features
**Objective:** Compute VADER sentiment from news articles

**File:** `src/features/sentiment.py`
- `analyze_sentiment(articles)` → dict with compound, pos, neg, neu, count
- `daily_sentiment(ticker, date)` → sentiment dict for a specific day

### Phase 4: ML Model

#### Task 4.1: Model training
**Objective:** Train XGBoost model with combined technical + sentiment features

**File:** `src/models/train.py`
- `prepare_dataset(ticker, period)` → X, y with all features
- `train_model(X_train, y_train, params)` → trained model
- `hyperparameter_tuning(X, y)` → best params via GridSearchCV
- `save_model(model, ticker)` → pickle to `models/{ticker}.pkl`

#### Task 4.2: Daily prediction
**Objective:** Generate daily predictions for all tracked tickers

**File:** `src/models/predict.py`
- `predict_today(ticker)` → {prediction, probability, sentiment, indicators}
- `predict_all(tickers)` → list of predictions
- `save_prediction(ticker, prediction)` → append to `predictions/{date}.json`

### Phase 5: Backtesting

#### Task 5.1: Backtesting engine
**Objective:** Walk-forward validation with performance metrics

**File:** `src/backtest/engine.py`
- `walk_forward_test(ticker, model, start_date, end_date)` → results dict
- `evaluate_predictions(predictions, actuals)` → accuracy, precision, recall, F1
- `trading_simulation(predictions, prices, capital=10000)` → P&L, Sharpe, max drawdown
- `compare_baseline(results)` → vs buy-and-hold, vs always-up

### Phase 6: Dashboard

#### Task 6.1: Streamlit dashboard
**Objective:** Interactive web dashboard for monitoring

**File:** `src/dashboard/app.py`
- **Tab 1 — Today:** Current prices, sentiment, predictions for all tickers
- **Tab 2 — History:** Prediction accuracy over time, charts
- **Tab 3 — Backtest:** Backtesting results, performance metrics
- **Tab 4 — Model:** Feature importance, model health

### Phase 7: CLI + Daily Tracking

#### Task 7.1: CLI tool
**Objective:** Command-line interface for daily operations

**File:** `src/cli.py`
- `python -m src.cli predict` → today's predictions
- `python -m src.cli backtest AAPL` → backtest single stock
- `python -m src.cli dashboard` → launch Streamlit
- `python -m src.cli track` → check yesterday's predictions vs reality

---

## Principles
- **DRY:** Extract reusable functions (all indicators in one module)
- **YAGNI:** Only build features validated by spikes
- **TDD:** Every module has tests before implementation
- **Frequent commits:** After each completed task
- **Caching:** Parquet files for price data, JSON for news to avoid rate limits

## Edge Cases to Handle
- Market closed (weekends/holidays) → skip prediction, warn
- Missing news data → fall back to technical-only prediction
- API rate limits → exponential backoff
- NaN values in indicators (warmup period) → forward-fill
- Ticker not found → graceful error, skip
- Empty news response → log warning, use cached sentiment
