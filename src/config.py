"""Central configuration for Stock Advisor."""
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = ROOT / "data"
PRICES_DIR = DATA_DIR / "prices"
NEWS_DIR = DATA_DIR / "news"
PREDICTIONS_DIR = ROOT / "predictions"
MODELS_DIR = ROOT / "models"

# Default US stocks to track (diversified: tech, finance, energy, consumer)
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",  # Tech
    "TSLA",                                               # Auto/Tech
    "JPM", "BAC",                                         # Finance
    "XOM", "CVX",                                         # Energy
    "WMT", "KO",                                          # Consumer
]

# XGBoost model parameters (to be tuned per stock)
DEFAULT_MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
}

# Backtesting
TRAIN_SPLIT = 0.8
LOOKBACK_YEARS = 3
MIN_TRAINING_DAYS = 252  # At least 1 year

# News
NEWS_LOOKBACK_HOURS = 48
MAX_NEWS_ARTICLES = 20

# Sentiment thresholds
BULLISH_THRESHOLD = 0.10
BEARISH_THRESHOLD = -0.10

# Ensure data directories exist
PRICES_DIR.mkdir(parents=True, exist_ok=True)
NEWS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
