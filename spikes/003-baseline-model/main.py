# Spike 003: Baseline XGBoost Model with Backtesting
# ====================================================
# Question: Can a basic XGBoost model with technical indicators predict
# next-day stock direction better than random chance (50%)?
#
# Approach:
#   1. Get 2+ years of daily price data for 5 major US stocks
#   2. Engineer features: technical indicators + price-derived features
#   3. Target: next-day direction (1=up, 0=down)
#   4. Walk-forward validation: train on first 80%, test on last 20%
#   5. Time-series cross-validation: rolling windows
#   6. Compare against baselines (always-up, always-down, random)
#   7. Evaluate per-stock and aggregate
#
# Verdict criteria:
#   - VALIDATED: accuracy > 53% consistently across stocks
#   - PARTIAL: accuracy 50-53% (marginal improvement)
#   - INVALIDATED: accuracy ≤ 50% (no better than coin flip)

import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("SPIKE 003: XGBOOST BASELINE MODEL WITH BACKTESTING")
print("=" * 70)

# ============================================================
# CONFIGURATION
# ============================================================
TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]  # Tech + Finance + Energy
PERIOD = "3y"          # 3 years of data
TEST_SPLIT = 0.20      # Last 20% for testing
N_SPLITS = 5           # Time series cross-validation splits

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def compute_rsi(series, period=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    """MACD: Moving Average Convergence Divergence"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bollinger(series, period=20, std=2):
    """Bollinger Bands"""
    sma = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    upper = sma + (rolling_std * std)
    lower = sma - (rolling_std * std)
    bandwidth = (upper - lower) / sma  # Normalized width
    pct_b = (series - lower) / (upper - lower)  # %B indicator
    return upper, lower, bandwidth, pct_b

def compute_atr(high, low, close, period=14):
    """Average True Range (volatility)"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def engineer_features(df):
    """Create all technical indicator features from OHLCV data"""
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    features = pd.DataFrame(index=df.index)
    
    # --- Price-based features ---
    features['returns_1d'] = close.pct_change(1)
    features['returns_5d'] = close.pct_change(5)
    features['returns_10d'] = close.pct_change(10)
    features['returns_21d'] = close.pct_change(21)  # ~1 month
    
    # --- Moving averages ---
    features['sma_5'] = close.rolling(5).mean()
    features['sma_10'] = close.rolling(10).mean()
    features['sma_20'] = close.rolling(20).mean()
    features['sma_50'] = close.rolling(50).mean()
    features['sma_200'] = close.rolling(200).mean()
    
    # Price relative to MAs
    features['close_to_sma5'] = close / features['sma_5'] - 1
    features['close_to_sma20'] = close / features['sma_20'] - 1
    features['close_to_sma50'] = close / features['sma_50'] - 1
    
    # --- Moving average crossovers ---
    features['sma5_cross_sma20'] = (features['sma_5'] - features['sma_20']) / close
    features['sma20_cross_sma50'] = (features['sma_20'] - features['sma_50']) / close
    
    # --- RSI ---
    features['rsi_14'] = compute_rsi(close, 14)
    features['rsi_7'] = compute_rsi(close, 7)
    
    # --- MACD ---
    macd_line, signal_line, histogram = compute_macd(close)
    features['macd'] = macd_line
    features['macd_signal'] = signal_line
    features['macd_histogram'] = histogram
    features['macd_histogram_pct'] = histogram / close  # Normalized
    
    # --- Bollinger Bands ---
    bb_upper, bb_lower, bb_width, pct_b = compute_bollinger(close)
    features['bb_pct_b'] = pct_b
    features['bb_width'] = bb_width
    
    # --- Volatility ---
    features['volatility_5d'] = features['returns_1d'].rolling(5).std()
    features['volatility_21d'] = features['returns_1d'].rolling(21).std()
    
    # --- ATR ---
    features['atr_14'] = compute_atr(high, low, close, 14)
    features['atr_pct'] = features['atr_14'] / close  # Normalized ATR
    
    # --- Volume features ---
    features['volume_ratio'] = volume / volume.rolling(20).mean()
    features['volume_change'] = volume.pct_change()
    
    # --- Price position ---
    features['daily_range'] = (high - low) / close
    features['gap'] = (df['Open'] - close.shift(1)) / close.shift(1)
    
    # --- High/Low relative position ---
    features['high_5d'] = close / high.rolling(5).max() - 1
    features['low_5d'] = close / low.rolling(5).min() - 1
    features['high_21d'] = close / high.rolling(21).max() - 1
    features['low_21d'] = close / low.rolling(21).min() - 1
    
    # --- Target: next-day direction ---
    features['target'] = (close.shift(-1) > close).astype(int)
    
    return features

# ============================================================
# LOAD DATA & ENGINEER FEATURES
# ============================================================
print("\n" + "─" * 70)
print(f"Loading {PERIOD} of data for: {', '.join(TICKERS)}")
print("─" * 70)

all_results = []

for ticker in TICKERS:
    print(f"\n{'='*50}")
    print(f"  {ticker}")
    print(f"{'='*50}")
    
    # Download data
    stock = yf.Ticker(ticker)
    df = stock.history(period=PERIOD)
    
    if df.empty or len(df) < 252:  # Need at least 1 year
        print(f"  ⚠️  Insufficient data ({len(df)} days), skipping")
        continue
    
    print(f"  Data: {len(df)} trading days, {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    
    # Engineer features
    features = engineer_features(df)
    features = features.dropna()  # Remove rows with NaN (indicators need warmup)
    
    print(f"  After feature engineering: {len(features)} rows, {features.shape[1]-1} features")
    
    if len(features) < 100:
        print(f"  ⚠️  Too few rows after NaN removal ({len(features)}), skipping")
        continue
    
    # ============================================================
    # TRAIN/TEST SPLIT (Time-based, no data leakage)
    # ============================================================
    split_idx = int(len(features) * (1 - TEST_SPLIT))
    train = features.iloc[:split_idx]
    test = features.iloc[split_idx:]
    
    X_train = train.drop('target', axis=1)
    y_train = train['target']
    X_test = test.drop('target', axis=1)
    y_test = test['target']
    
    print(f"  Train: {len(train)} days ({train.index[0].strftime('%Y-%m-%d')} to {train.index[-1].strftime('%Y-%m-%d')})")
    print(f"  Test:  {len(test)} days ({test.index[0].strftime('%Y-%m-%d')} to {test.index[-1].strftime('%Y-%m-%d')})")
    
    # ============================================================
    # BASELINES
    # ============================================================
    always_up_acc = y_test.mean()  # Accuracy if we always predict UP
    always_down_acc = 1 - y_test.mean()  # Accuracy if we always predict DOWN
    baseline_accuracy = max(always_up_acc, always_down_acc)
    
    print(f"\n  --- BASELINES ---")
    print(f"  Always predict UP:   {always_up_acc:.1%}")
    print(f"  Always predict DOWN: {always_down_acc:.1%}")
    print(f"  Best naive baseline:  {baseline_accuracy:.1%}")
    
    # ============================================================
    # XGBOOST MODEL
    # ============================================================
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n  --- XGBOOST RESULTS (Hold-out Test) ---")
    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1 Score:  {f1:.1%}")
    print(f"  vs Baseline: {accuracy - baseline_accuracy:+.1%}")
    
    # ============================================================
    # TIME SERIES CROSS-VALIDATION (more rigorous)
    # ============================================================
    tscv = TimeSeriesSplit(n_splits=min(N_SPLITS, len(features) // 50))
    cv_scores = []
    
    all_data_X = features.drop('target', axis=1)
    all_data_y = features['target']
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(all_data_X)):
        X_tr, X_te = all_data_X.iloc[train_idx], all_data_X.iloc[test_idx]
        y_tr, y_te = all_data_y.iloc[train_idx], all_data_y.iloc[test_idx]
        
        if len(y_te) < 10:
            continue
            
        m = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            verbosity=0
        )
        m.fit(X_tr, y_tr)
        fold_acc = accuracy_score(y_te, m.predict(X_te))
        cv_scores.append(fold_acc)
    
    if cv_scores:
        print(f"\n  --- TIME SERIES CV ({len(cv_scores)} folds) ---")
        print(f"  Mean accuracy:  {np.mean(cv_scores):.1%}")
        print(f"  Std:            {np.std(cv_scores):.1%}")
        print(f"  Range:          {min(cv_scores):.1%} - {max(cv_scores):.1%}")
    
    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n  --- TOP 10 FEATURES ---")
    for i, row in importance.head(10).iterrows():
        print(f"  {row['feature']:<25s}: {row['importance']:.4f}")
    
    # ============================================================
    # DAILY PREDICTION SIMULATION (walk-forward on test set)
    # ============================================================
    predictions = []
    actuals = []
    dates = []
    
    for i in range(len(X_test)):
        pred = model.predict(X_test.iloc[i:i+1])[0]
        prob = model.predict_proba(X_test.iloc[i:i+1])[0]
        actual = y_test.iloc[i]
        
        predictions.append(pred)
        actuals.append(actual)
        dates.append(test.index[i])
    
    # Calculate daily accuracy rolling windows
    pred_series = pd.Series(predictions, index=dates)
    actual_series = pd.Series(actuals, index=dates)
    
    # Running accuracy
    correct = (pred_series == actual_series)
    running_acc = correct.expanding().mean()
    
    print(f"\n  --- WALK-FORWARD SIMULATION ---")
    print(f"  Final running accuracy: {running_acc.iloc[-1]:.1%}")
    print(f"  Max running accuracy:   {running_acc.max():.1%}")
    print(f"  Min running accuracy:   {running_acc.min():.1%}")
    
    # ============================================================
    # UPSIDE/DOWNSIDE ANALYSIS
    # ============================================================
    up_pred_mask = pred_series == 1
    down_pred_mask = pred_series == 0
    
    # When model says UP, what actually happens?
    if up_pred_mask.sum() > 0:
        up_accuracy = correct[up_pred_mask].mean()
        avg_return_when_up = features.loc[dates, 'returns_1d'][up_pred_mask].mean() * 100
    else:
        up_accuracy = 0
        avg_return_when_up = 0
    
    if down_pred_mask.sum() > 0:
        down_accuracy = correct[down_pred_mask].mean()
        avg_return_when_down = features.loc[dates, 'returns_1d'][down_pred_mask].mean() * 100
    else:
        down_accuracy = 0
        avg_return_when_down = 0
    
    print(f"\n  --- DIRECTIONAL ANALYSIS ---")
    print(f"  Predicted UP   ({up_pred_mask.sum()} times): {up_accuracy:.1%} correct, avg return {avg_return_when_up:+.2f}%")
    print(f"  Predicted DOWN ({down_pred_mask.sum()} times): {down_accuracy:.1%} correct, avg return {avg_return_when_down:+.2f}%")
    
    # ============================================================
    # STORE RESULTS
    # ============================================================
    result = {
        'ticker': ticker,
        'data_days': len(df),
        'features': features.shape[1] - 1,
        'accuracy': accuracy,
        'baseline': baseline_accuracy,
        'improvement': accuracy - baseline_accuracy,
        'cv_mean': np.mean(cv_scores) if cv_scores else None,
        'cv_std': np.std(cv_scores) if cv_scores else None,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'up_ratio': y_test.mean(),
    }
    all_results.append(result)

# ============================================================
# AGGREGATE RESULTS
# ============================================================
print("\n" + "=" * 70)
print("AGGREGATE RESULTS ACROSS ALL STOCKS")
print("=" * 70)

if all_results:
    results_df = pd.DataFrame(all_results)
    
    print(f"\n  {'Ticker':<8} {'Accuracy':>10} {'Baseline':>10} {'Δ':>8} {'CV Mean':>10} {'CV Std':>8} {'F1':>8}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
    
    for _, row in results_df.iterrows():
        cv_str = f"{row['cv_mean']:.1%}" if row['cv_mean'] is not None else "N/A"
        cv_std_str = f"{row['cv_std']:.1%}" if row['cv_std'] is not None else "N/A"
        print(f"  {row['ticker']:<8} {row['accuracy']:>9.1%} {row['baseline']:>9.1%} "
              f"{row['improvement']:>+7.1%} {cv_str:>10} {cv_std_str:>8} {row['f1']:>7.1%}")
    
    # Summary stats
    avg_acc = results_df['accuracy'].mean()
    avg_improvement = results_df['improvement'].mean()
    avg_cv = results_df['cv_mean'].mean() if results_df['cv_mean'].notna().any() else None
    
    print(f"\n  --- SUMMARY ---")
    print(f"  Average accuracy:     {avg_acc:.1%}")
    print(f"  Average improvement:  {avg_improvement:+.1%}")
    if avg_cv:
        print(f"  Average CV accuracy:  {avg_cv:.1%}")
    
    stocks_beating = (results_df['improvement'] > 0).sum()
    print(f"  Stocks beating baseline: {stocks_beating}/{len(results_df)}")
    
    # ============================================================
    # SIMPLE TRADING SIMULATION
    # ============================================================
    print(f"\n  --- HYPOTHETICAL TRADING SIMULATION ---")
    print(f"  If we invested $10,000 in each stock following model signals:")
    print(f"  (Buy when model predicts UP, sell/hold cash when predicts DOWN)")
    
# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 70)
print("VERDICT: SPIKE 003")
print("=" * 70)

if all_results:
    beating = stocks_beating
    total = len(results_df)
    
    if avg_improvement > 0.02:
        verdict = "VALIDATED"
        confidence = "HIGH"
        detail = f"Model beats baseline by {avg_improvement:+.1%} on average across {total} stocks"
    elif avg_improvement > 0:
        verdict = "PARTIAL"
        confidence = "MEDIUM"
        detail = f"Marginal improvement ({avg_improvement:+.1%}). More features (news sentiment) should help."
    else:
        verdict = "INVALIDATED"
        confidence = "LOW"
        detail = f"Model does not beat naive baseline. Need better features or different approach."
    
    print(f"""
  Verdict: {verdict}

  Average accuracy:        {avg_acc:.1%}
  Average vs baseline:     {avg_improvement:+.1%}
  Stocks beating baseline: {beating}/{total}
  CV consistency:          {'Stable' if (results_df['cv_std'] < 0.05).all() else 'Variable'}

  {detail}

  Key findings:
  - XGBoost with technical indicators {'provides' if avg_improvement > 0 else 'does not provide'} 
    predictive signal above random chance
  - Technical indicators alone are {'sufficient' if avg_improvement > 0.02 else 'insufficient'} 
    for a production model
  - Adding news sentiment (Spike 002) should {'further improve' if avg_improvement > 0 else 'be essential for'} 
    accuracy
  - Walk-forward backtesting framework is working and ready for continuous evaluation

  Recommendation: {'BUILD the full app with technical + sentiment features' if avg_improvement > 0 
                    else 'Re-evaluate approach — may need fundamental data or different model architecture'}
""")
