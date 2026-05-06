"""Walk-forward backtesting engine."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.models.train import prepare_dataset, train_model, evaluate_model
from src.config import TRAIN_SPLIT


def walk_forward_test(
    ticker: str,
    period: str = "3y",
    retrain_frequency: str = "3mo",
) -> dict:
    """Walk-forward backtesting with periodic retraining.
    
    Simulates real-world usage: train on past data, predict next period,
    then retrain including that period, and repeat.
    
    Args:
        ticker: Stock symbol
        period: Total data period to use
        retrain_frequency: How often to retrain ('1mo', '3mo', '6mo')
    
    Returns:
        dict with detailed backtest results
    """
    X, y = prepare_dataset(ticker, period)
    
    # Initial train/test split
    split_idx = int(len(X) * TRAIN_SPLIT)
    
    results = []
    predictions = []
    actuals = []
    dates = []
    
    for i in range(split_idx, len(X)):
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]
        X_test = X.iloc[i:i+1]
        y_test = y.iloc[i:i+1]
        
        try:
            model = train_model(X_train, y_train)
            pred = int(model.predict(X_test)[0])
            prob = float(model.predict_proba(X_test)[0][1])
            actual = int(y_test.iloc[0])
            
            predictions.append(pred)
            actuals.append(actual)
            dates.append(X_test.index[0])
            
        except Exception as e:
            continue
    
    if not predictions:
        return {'ticker': ticker, 'error': 'No predictions generated'}
    
    # Calculate metrics
    correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
    accuracy = correct / len(predictions)
    
    pred_series = pd.Series(predictions, index=dates)
    actual_series = pd.Series(actuals, index=dates)
    
    # Baseline
    baseline_up = actual_series.mean()
    baseline = max(baseline_up, 1 - baseline_up)
    
    # Running accuracy
    correct_series = (pred_series == actual_series)
    running_acc = correct_series.expanding().mean()
    
    # Directional analysis
    up_mask = pred_series == 1
    down_mask = pred_series == 0
    
    up_acc = correct_series[up_mask].mean() if up_mask.sum() > 0 else 0
    down_acc = correct_series[down_mask].mean() if down_mask.sum() > 0 else 0
    
    return {
        'ticker': ticker,
        'predictions': len(predictions),
        'accuracy': round(accuracy, 4),
        'baseline': round(baseline, 4),
        'improvement': round(accuracy - baseline, 4),
        'up_accuracy': round(float(up_acc), 4),
        'down_accuracy': round(float(down_acc), 4),
        'up_predictions': int(up_mask.sum()),
        'down_predictions': int(down_mask.sum()),
        'running_accuracy_final': round(float(running_acc.iloc[-1]), 4),
        'running_accuracy_max': round(float(running_acc.max()), 4),
        'running_accuracy_min': round(float(running_acc.min()), 4),
        'test_start': str(dates[0].date()) if dates else None,
        'test_end': str(dates[-1].date()) if dates else None,
    }


def trading_simulation(
    predictions: list[int],
    prices: pd.Series,
    initial_capital: float = 10000.0,
) -> dict:
    """Simulate trading following model predictions.
    
    Strategy: Buy when model predicts UP, sell/hold cash when predicts DOWN.
    
    Args:
        predictions: List of 0 (down) or 1 (up) predictions
        prices: Series of closing prices (aligned with predictions)
        initial_capital: Starting capital
    
    Returns:
        dict with P&L, returns, Sharpe ratio, max drawdown
    """
    capital = initial_capital
    position = 0  # Shares held
    equity_curve = [capital]
    trades = []
    
    daily_returns = prices.pct_change().dropna()
    
    for i in range(len(predictions)):
        price = float(prices.iloc[i])
        
        if predictions[i] == 1 and position == 0:
            # Buy
            position = capital / price
            capital = 0
            trades.append({'day': i, 'action': 'BUY', 'price': price})
        elif predictions[i] == 0 and position > 0:
            # Sell
            capital = position * price
            position = 0
            trades.append({'day': i, 'action': 'SELL', 'price': price})
        
        # Track equity
        equity = capital + (position * price)
        equity_curve.append(equity)
    
    # Final sell if still holding
    if position > 0:
        capital = position * float(prices.iloc[-1])
        equity_curve[-1] = capital
    
    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()
    
    total_return = (equity_curve[-1] / initial_capital - 1) * 100
    
    # Sharpe ratio (annualized, assuming 252 trading days)
    if len(returns) > 0 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Max drawdown
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak
    max_drawdown = float(drawdown.min() * 100)
    
    # Buy & hold comparison
    buy_hold_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    
    return {
        'initial_capital': initial_capital,
        'final_equity': round(float(equity_curve[-1]), 2),
        'total_return_pct': round(total_return, 2),
        'sharpe_ratio': round(sharpe, 4),
        'max_drawdown_pct': round(max_drawdown, 2),
        'num_trades': len(trades),
        'buy_hold_return_pct': round(float(buy_hold_return), 2),
    }
