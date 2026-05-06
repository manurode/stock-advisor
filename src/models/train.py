"""Model training and prediction using XGBoost."""
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import MODELS_DIR, DEFAULT_MODEL_PARAMS, TRAIN_SPLIT, MIN_TRAINING_DAYS
from src.collectors.price import fetch_prices
from src.features.technical import compute_all_features


def prepare_dataset(ticker: str, period: str = "3y") -> tuple[pd.DataFrame, pd.Series]:
    """Fetch price data and engineer features for a ticker.
    
    Returns:
        X (features DataFrame), y (target Series)
    """
    df = fetch_prices(ticker, period)
    features = compute_all_features(df)
    features = features.dropna()
    
    if len(features) < MIN_TRAINING_DAYS:
        raise ValueError(
            f"Insufficient data for {ticker}: {len(features)} days "
            f"(need at least {MIN_TRAINING_DAYS})"
        )
    
    target = features['target']
    X = features.drop('target', axis=1)
    
    return X, target


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier.
    
    Args:
        X_train: Feature DataFrame
        y_train: Target Series (0 or 1)
        params: XGBoost parameters (uses defaults if None)
    
    Returns:
        Trained XGBClassifier
    """
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


def train_and_evaluate(ticker: str, period: str = "3y") -> dict:
    """Full training pipeline: fetch, train/test split, train, evaluate.
    
    Returns:
        dict with model, metrics, feature importance, and data
    """
    X, y = prepare_dataset(ticker, period)
    
    # Time-respecting train/test split
    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    model = train_model(X_train, y_train)
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
    }


def save_model(model: xgb.XGBClassifier, ticker: str) -> Path:
    """Save a trained model to disk."""
    path = MODELS_DIR / f"{ticker}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    return path


def load_model(ticker: str) -> xgb.XGBClassifier | None:
    """Load a trained model from disk."""
    path = MODELS_DIR / f"{ticker}.pkl"
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def predict_next_day(
    ticker: str,
    model: xgb.XGBClassifier | None = None,
) -> dict:
    """Predict next-day direction for a ticker.
    
    Args:
        ticker: Stock symbol
        model: Pre-trained model (loads from disk if None)
    
    Returns:
        dict with prediction, probability, and current indicators
    """
    if model is None:
        model = load_model(ticker)
    
    if model is None:
        return {'ticker': ticker, 'error': 'No trained model found'}
    
    # Get fresh data and compute features for the latest day
    df = fetch_prices(ticker, period="6mo")
    features = compute_all_features(df)
    latest = features.drop('target', axis=1).iloc[-1:]
    
    if latest.isna().any().any():
        return {'ticker': ticker, 'error': 'Missing features for latest day'}
    
    prediction = int(model.predict(latest)[0])
    probabilities = model.predict_proba(latest)[0]
    
    return {
        'ticker': ticker,
        'prediction': 'UP' if prediction == 1 else 'DOWN',
        'confidence': round(float(max(probabilities)), 4),
        'prob_up': round(float(probabilities[1]), 4),
        'prob_down': round(float(probabilities[0]), 4),
    }
