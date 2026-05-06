"""Model training and prediction using XGBoost with sentiment features."""
import pickle
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import (
    MODELS_DIR, DEFAULT_MODEL_PARAMS, TRAIN_SPLIT, MIN_TRAINING_DAYS,
    BULLISH_THRESHOLD, BEARISH_THRESHOLD,
)
from src.collectors.price import fetch_prices
from src.features.technical import compute_all_features


# ─── Market Sentiment Features (available historically) ───

def _add_market_features(features: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add market-wide sentiment proxy features (VIX, SPY) to the feature set."""
    try:
        vix = fetch_prices("^VIX", period="3y")
        spy = fetch_prices("SPY", period="3y")
        
        if vix.empty or spy.empty:
            return features
        
        def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            return df
        
        vix = _normalize_index(vix)
        spy = _normalize_index(spy)
        
        date_index = pd.to_datetime(features.index).tz_localize(None).normalize()
        
        vix_returns = vix['Close'].pct_change()
        vix_change_5d = vix['Close'].pct_change(5)
        spy_returns = spy['Close'].pct_change()
        spy_change_5d = spy['Close'].pct_change(5)
        
        features['market_vix_return'] = vix_returns.reindex(date_index).values
        features['market_vix_5d'] = vix_change_5d.reindex(date_index).values
        features['market_spy_return'] = spy_returns.reindex(date_index).values
        features['market_spy_5d'] = spy_change_5d.reindex(date_index).values
        
        vix_level = vix['Close'].reindex(date_index)
        features['market_vix_level'] = (vix_level.values / vix_level.rolling(50).mean().values) - 1
        
        spy_up = (spy_returns > 0).astype(int)
        features['market_breadth_5d'] = spy_up.rolling(5).sum().reindex(date_index).values / 5
        
    except Exception:
        pass
    
    return features


# ─── Dataset Preparation ───

def prepare_dataset(
    ticker: str,
    period: str = "3y",
    include_market: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Fetch price data and engineer features for a ticker."""
    df = fetch_prices(ticker, period)
    features = compute_all_features(df)
    
    if include_market:
        features = _add_market_features(features, ticker)
    
    features = features.dropna()
    
    if len(features) < MIN_TRAINING_DAYS:
        raise ValueError(
            f"Insufficient data for {ticker}: {len(features)} days "
            f"(need at least {MIN_TRAINING_DAYS})"
        )
    
    target = features['target']
    X = features.drop('target', axis=1)
    
    return X, target


# ─── Training ───

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier."""
    if params is None:
        params = DEFAULT_MODEL_PARAMS.copy()
    
    model = xgb.XGBClassifier(**params, verbosity=0)
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate a trained model on test data."""
    y_pred = model.predict(X_test)
    
    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'baseline': float(max(y_test.mean(), 1 - y_test.mean())),
        'samples': len(y_test),
    }


# ─── Hyperparameter Tuning ───

def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 25,
) -> dict:
    """Optimize XGBoost hyperparameters using RandomizedSearchCV with TimeSeriesSplit.
    
    Args:
        X_train: Training features
        y_train: Training targets
        n_iter: Number of parameter combinations to try
    
    Returns:
        dict with best_params, best_score, and search details
    """
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.3, 0.5],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [1.0, 2.0, 5.0],
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
    )
    
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring='accuracy',
        random_state=42,
        verbose=0,
        n_jobs=1,
    )
    
    search.fit(X_train, y_train)
    
    return {
        'best_params': search.best_params_,
        'best_score': float(search.best_score_),
        'n_iter': n_iter,
    }


def train_and_evaluate(
    ticker: str,
    period: str = "3y",
    tune: bool = False,
    n_iter: int = 25,
) -> dict:
    """Full training pipeline: fetch, train/test split, train, evaluate.
    
    Args:
        ticker: Stock symbol
        period: Data period
        tune: If True, run hyperparameter tuning before final training
        n_iter: Number of tuning iterations (only if tune=True)
    """
    X, y = prepare_dataset(ticker, period)
    
    # Time-respecting train/test split
    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Optional hyperparameter tuning
    tuning_result = None
    if tune:
        tuning_result = tune_hyperparameters(X_train, y_train, n_iter=n_iter)
        params = tuning_result['best_params']
    else:
        params = DEFAULT_MODEL_PARAMS.copy()
    
    model = train_model(X_train, y_train, params)
    metrics = evaluate_model(model, X_test, y_test)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)
    
    return {
        'ticker': ticker,
        'model': model,
        'metrics': metrics,
        'features': X_train.columns.tolist(),
        'n_features': len(X_train.columns),
        'train_days': len(X_train),
        'test_days': len(X_test),
        'top_features': importance.head(10).to_dict('records'),
        'params': params,
        'tuning': tuning_result,
    }


# ─── Persistence ───

def save_model(model: xgb.XGBClassifier, ticker: str) -> Path:
    """Save a trained model to disk."""
    path = MODELS_DIR / f"{ticker}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    return path


def save_params(params: dict, ticker: str) -> Path:
    """Save best hyperparameters to JSON."""
    path = MODELS_DIR / f"{ticker}_params.json"
    path.write_text(json.dumps(params, indent=2))
    return path


def load_model(ticker: str) -> xgb.XGBClassifier | None:
    """Load a trained model from disk."""
    path = MODELS_DIR / f"{ticker}.pkl"
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_params(ticker: str) -> dict | None:
    """Load saved hyperparameters."""
    path = MODELS_DIR / f"{ticker}_params.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ─── Prediction ───

def _get_latest_features(ticker: str) -> pd.DataFrame:
    """Get feature vector for the most recent trading day."""
    df = fetch_prices(ticker, period="6mo")
    features = compute_all_features(df)
    features = _add_market_features(features, ticker)
    
    latest_rows = features.drop('target', axis=1).iloc[-5:]
    latest_rows = latest_rows.ffill().bfill()
    latest = latest_rows.iloc[-1:]
    
    if latest.isna().any().any():
        latest = latest.fillna(0)
    
    return latest


def predict_next_day(
    ticker: str,
    model: xgb.XGBClassifier | None = None,
) -> dict:
    """Predict next-day direction using technical + market features."""
    if model is None:
        model = load_model(ticker)
    
    if model is None:
        return {'ticker': ticker, 'error': 'No trained model found'}
    
    latest = _get_latest_features(ticker)
    
    # Align features with model's expected columns
    expected_cols = model.get_booster().feature_names
    if set(expected_cols) != set(latest.columns):
        common = [c for c in expected_cols if c in latest.columns]
        if len(common) < len(expected_cols) * 0.5:
            return {'ticker': ticker, 'error': 'Feature mismatch — retrain model'}
        latest = latest[common]
    
    prediction = int(model.predict(latest)[0])
    probabilities = model.predict_proba(latest)[0]
    
    return {
        'ticker': ticker,
        'prediction': 'UP' if prediction == 1 else 'DOWN',
        'confidence': round(float(max(probabilities)), 4),
        'prob_up': round(float(probabilities[1]), 4),
        'prob_down': round(float(probabilities[0]), 4),
    }


def predict_with_sentiment(
    ticker: str,
    model: xgb.XGBClassifier | None = None,
    sentiment: dict | None = None,
    sentiment_weight: float = 0.30,
) -> dict:
    """Predict next-day direction combining ML model + news sentiment.
    
    The final prediction blends:
      - ML model probability (technical + market features)
      - Current news sentiment (VADER compound score)
    
    Args:
        ticker: Stock symbol
        model: Pre-trained XGBoost model
        sentiment: Sentiment dict from features.sentiment.get_ticker_sentiment()
        sentiment_weight: How much weight to give sentiment (0-1)
    """
    # Get base ML prediction
    ml_result = predict_next_day(ticker, model)
    
    if 'error' in ml_result:
        return ml_result
    
    # Get sentiment if not provided
    if sentiment is None:
        try:
            from src.features.sentiment import get_ticker_sentiment
            sentiment = get_ticker_sentiment(ticker)
        except Exception:
            sentiment = {'compound': 0.0, 'label': 'NO_DATA', 'article_count': 0}
    
    compound = sentiment.get('compound', 0.0)
    ml_prob_up = ml_result['prob_up']
    sentiment_adjustment = compound * sentiment_weight
    combined_prob_up = max(0.0, min(1.0, ml_prob_up + sentiment_adjustment))
    
    # Final prediction
    if combined_prob_up >= 0.50:
        combined_prediction = 'UP'
    elif combined_prob_up <= 0.45:
        combined_prediction = 'DOWN'
    else:
        combined_prediction = ml_result['prediction']
    
    # Determine signal agreement
    ml_direction = ml_result['prediction']
    sent_direction = 'UP' if compound > BULLISH_THRESHOLD else ('DOWN' if compound < BEARISH_THRESHOLD else 'NEUTRAL')
    
    if ml_direction == sent_direction:
        signal = 'ALIGNED ✅'
    elif sent_direction == 'NEUTRAL':
        signal = 'ML_ONLY 🤖'
    else:
        signal = 'CONFLICT ⚠️'
    
    return {
        'ticker': ticker,
        'prediction': combined_prediction,
        'confidence': round(max(combined_prob_up, 1 - combined_prob_up), 4),
        'prob_up': round(combined_prob_up, 4),
        'prob_down': round(1 - combined_prob_up, 4),
        'ml_prediction': ml_result['prediction'],
        'ml_prob_up': ml_result['prob_up'],
        'sentiment_label': sentiment.get('label', 'N/A'),
        'sentiment_compound': compound,
        'sentiment_articles': sentiment.get('article_count', 0),
        'signal': signal,
        'method': 'combined',
    }
