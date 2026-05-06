"""Price data collector using yfinance."""
import yfinance as yf
import pandas as pd
from pathlib import Path
from src.config import PRICES_DIR, LOOKBACK_YEARS


def fetch_prices(ticker: str, period: str = f"{LOOKBACK_YEARS}y") -> pd.DataFrame:
    """Fetch OHLCV price history for a ticker.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        period: yfinance period string (e.g., '3y', '1y', '6mo')
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'")
    
    return df


def get_current_price(ticker: str) -> float:
    """Get the most recent closing price for a ticker."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose', 0.0)


def get_price_summary(ticker: str) -> dict:
    """Get a quick price summary for a ticker."""
    prices = fetch_prices(ticker, period="5d")
    
    if len(prices) < 2:
        return {"ticker": ticker, "error": "Insufficient data"}
    
    current = float(prices['Close'].iloc[-1])
    prev_close = float(prices['Close'].iloc[-2])
    change = current - prev_close
    change_pct = (change / prev_close) * 100
    
    return {
        "ticker": ticker,
        "price": round(current, 2),
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "high_5d": round(float(prices['High'].max()), 2),
        "low_5d": round(float(prices['Low'].min()), 2),
        "volume_avg": int(prices['Volume'].mean()),
    }


def save_prices(ticker: str, df: pd.DataFrame) -> Path:
    """Cache price data to parquet."""
    path = PRICES_DIR / f"{ticker}.parquet"
    df.to_parquet(path)
    return path


def load_prices(ticker: str) -> pd.DataFrame | None:
    """Load cached price data if available."""
    path = PRICES_DIR / f"{ticker}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def fetch_and_cache(ticker: str, period: str = f"{LOOKBACK_YEARS}y") -> pd.DataFrame:
    """Fetch prices and cache them. Returns fresh data."""
    df = fetch_prices(ticker, period)
    save_prices(ticker, df)
    return df
