"""Technical indicators feature engineering."""
import pandas as pd
import numpy as np


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — overbought/oversold indicator.
    
    RSI > 70 = overbought, RSI < 30 = oversold
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, pd.Series]:
    """MACD: Moving Average Convergence Divergence.
    
    Bullish signal: MACD line crosses above signal line
    Bearish signal: MACD line crosses below signal line
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_histogram': histogram,
    }


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> dict[str, pd.Series]:
    """Bollinger Bands — volatility indicator.
    
    %B > 1: price above upper band (overbought)
    %B < 0: price below lower band (oversold)
    """
    sma = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    upper = sma + (rolling_std * num_std)
    lower = sma - (rolling_std * num_std)
    bandwidth = (upper - lower) / sma
    pct_b = (series - lower) / (upper - lower)
    
    return {
        'bb_upper': upper,
        'bb_lower': lower,
        'bb_width': bandwidth,
        'bb_pct_b': pct_b,
    }


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range — volatility measure."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(alpha=1/period, adjust=False).mean()


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators from OHLCV DataFrame.
    
    Args:
        df: DataFrame with columns Open, High, Low, Close, Volume
    
    Returns:
        DataFrame with 30+ technical indicator features + target column
    """
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    features = pd.DataFrame(index=df.index)
    
    # === Price Returns ===
    features['returns_1d'] = close.pct_change(1)
    features['returns_5d'] = close.pct_change(5)
    features['returns_10d'] = close.pct_change(10)
    features['returns_21d'] = close.pct_change(21)
    
    # === Simple Moving Averages ===
    for window in [5, 10, 20, 50, 200]:
        features[f'sma_{window}'] = close.rolling(window).mean()
    
    # === Price Relative to MAs ===
    features['close_to_sma5'] = close / features['sma_5'] - 1
    features['close_to_sma20'] = close / features['sma_20'] - 1
    features['close_to_sma50'] = close / features['sma_50'] - 1
    
    # === MA Crossovers ===
    features['sma5_cross_sma20'] = (features['sma_5'] - features['sma_20']) / close
    features['sma20_cross_sma50'] = (features['sma_20'] - features['sma_50']) / close
    
    # === RSI ===
    features['rsi_7'] = rsi(close, 7)
    features['rsi_14'] = rsi(close, 14)
    
    # === MACD ===
    macd_data = macd(close)
    features['macd'] = macd_data['macd']
    features['macd_signal'] = macd_data['macd_signal']
    features['macd_histogram'] = macd_data['macd_histogram']
    features['macd_histogram_pct'] = macd_data['macd_histogram'] / close
    
    # === Bollinger Bands ===
    bb = bollinger_bands(close)
    features['bb_pct_b'] = bb['bb_pct_b']
    features['bb_width'] = bb['bb_width']
    
    # === Volatility ===
    features['volatility_5d'] = features['returns_1d'].rolling(5).std()
    features['volatility_21d'] = features['returns_1d'].rolling(21).std()
    
    # === ATR ===
    features['atr_14'] = atr(high, low, close, 14)
    features['atr_pct'] = features['atr_14'] / close
    
    # === Volume ===
    features['volume_ratio'] = volume / volume.rolling(20).mean()
    features['volume_change'] = volume.pct_change()
    
    # === Price Position ===
    features['daily_range'] = (high - low) / close
    features['gap'] = (df['Open'] - close.shift(1)) / close.shift(1)
    
    # === High/Low Relative Position ===
    features['high_5d'] = close / high.rolling(5).max() - 1
    features['low_5d'] = close / low.rolling(5).min() - 1
    features['high_21d'] = close / high.rolling(21).max() - 1
    features['low_21d'] = close / low.rolling(21).min() - 1
    
    # === Target: Next-day direction (1=up, 0=down) ===
    features['target'] = (close.shift(-1) > close).astype(int)
    
    return features
