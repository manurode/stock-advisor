"""LSTM-style neural network model using scikit-learn MLPClassifier.

Since TensorFlow/PyTorch are not available in this environment, we use
MLPClassifier with sequence-aware feature engineering (lagged features
from previous days) to capture temporal patterns similar to an LSTM.

The key difference from XGBoost:
- XGBoost: each day is independent, learns non-linear feature interactions
- LSTM/MLP+lagged: explicitly models time dependency by feeding the model
  features from the last N days, letting it learn temporal patterns.
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import MODELS_DIR, TRAIN_SPLIT, MIN_TRAINING_DAYS
from src.models.train import prepare_dataset


def create_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    lookback: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Create time-series sequences from tabular data.
    
    Reshapes (n_samples, n_features) into (n_samples - lookback, lookback * n_features).
    Each sample now contains the last `lookback` days of features, allowing
    the MLP to learn temporal patterns.
    
    Args:
        X: Feature DataFrame (n_samples × n_features)
        y: Target Series
        lookback: Number of past days to include in each sample
    
    Returns:
        X_seq: (n_samples - lookback) × (lookback * n_features) array
        y_seq: aligned target array
    """
    n_samples, n_features = X.shape
    
    if n_samples <= lookback:
        raise ValueError(f"Need more than {lookback} samples, got {n_samples}")
    
    X_seq = np.zeros((n_samples - lookback, lookback * n_features))
    
    for i in range(lookback, n_samples):
        window = X.iloc[i - lookback:i].values
        X_seq[i - lookback] = window.flatten()
    
    y_seq = y.iloc[lookback:].values
    
    return X_seq, y_seq


def train_lstm_model(
    ticker: str,
    period: str = "3y",
    lookback: int = 10,
    hidden_layers: tuple = (128, 64, 32),
) -> dict:
    """Train an MLP-based sequential model (LSTM alternative).
    
    Args:
        ticker: Stock symbol
        period: Data period
        lookback: Days of history to feed the model
        hidden_layers: MLP hidden layer sizes
    
    Returns:
        dict with model, scaler, metrics, and metadata
    """
    X, y = prepare_dataset(ticker, period)
    
    # Create sequences
    X_seq, y_seq = create_sequences(X, y, lookback=lookback)
    
    # Train/test split (time-respecting)
    split_idx = int(len(X_seq) * TRAIN_SPLIT)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    # Scale features (important for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train MLP
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        alpha=0.001,          # L2 regularization
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=False,
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'baseline': float(max(y_test.mean(), 1 - y_test.mean())),
        'samples': len(y_test),
        'iterations': model.n_iter_,
        'final_loss': float(model.loss_curve_[-1]) if model.loss_curve_ else None,
    }
    
    return {
        'ticker': ticker,
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'lookback': lookback,
        'hidden_layers': hidden_layers,
        'input_dim': X_seq.shape[1],
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features_original': X.shape[1],
    }


def save_lstm(result: dict, ticker: str) -> tuple[Path, Path]:
    """Save LSTM model and scaler to disk."""
    model_path = MODELS_DIR / f"{ticker}_lstm.pkl"
    scaler_path = MODELS_DIR / f"{ticker}_lstm_scaler.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(result['model'], f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(result['scaler'], f)
    
    return model_path, scaler_path


def load_lstm(ticker: str) -> tuple[MLPClassifier | None, StandardScaler | None]:
    """Load LSTM model and scaler from disk."""
    model_path = MODELS_DIR / f"{ticker}_lstm.pkl"
    scaler_path = MODELS_DIR / f"{ticker}_lstm_scaler.pkl"
    
    model = None
    scaler = None
    
    if model_path.exists():
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    return model, scaler


def predict_lstm(
    ticker: str,
    model: MLPClassifier | None = None,
    scaler: StandardScaler | None = None,
    lookback: int = 10,
) -> dict:
    """Predict next-day direction using the LSTM-style model.
    
    Args:
        ticker: Stock symbol
        model: Pre-trained MLPClassifier
        scaler: Fitted StandardScaler
        lookback: Must match training lookback
    
    Returns:
        Prediction dict
    """
    if model is None or scaler is None:
        model, scaler = load_lstm(ticker)
    
    if model is None or scaler is None:
        return {'ticker': ticker, 'error': 'No LSTM model found'}
    
    # Get recent features (need lookback+5 rows for safety)
    X, y = prepare_dataset(ticker, period="6mo")
    
    if len(X) < lookback:
        return {'ticker': ticker, 'error': f'Need {lookback} days, got {len(X)}'}
    
    # Take the last `lookback` rows and flatten
    recent = X.iloc[-lookback:]
    X_input = recent.values.flatten().reshape(1, -1)
    X_scaled = scaler.transform(X_input)
    
    prediction = int(model.predict(X_scaled)[0])
    probabilities = model.predict_proba(X_scaled)[0]
    
    return {
        'ticker': ticker,
        'prediction': 'UP' if prediction == 1 else 'DOWN',
        'confidence': round(float(max(probabilities)), 4),
        'prob_up': round(float(probabilities[1]), 4),
        'prob_down': round(float(probabilities[0]), 4),
        'model_type': 'LSTM (MLP+sequential)',
    }


def compare_models(ticker: str) -> dict:
    """Train both XGBoost and LSTM models and compare performance.
    
    Returns comparison dict with metrics for both models.
    """
    from src.models.train import train_and_evaluate as train_xgb
    
    print(f"Comparing models for {ticker}...")
    
    # XGBoost
    print("  Training XGBoost...")
    xgb_result = train_xgb(ticker)
    
    # LSTM/MLP
    print("  Training LSTM/MLP...")
    lstm_result = train_lstm_model(ticker)
    
    comparison = {
        'ticker': ticker,
        'xgboost': {
            'accuracy': xgb_result['metrics']['accuracy'],
            'f1': xgb_result['metrics']['f1'],
            'improvement': xgb_result['metrics']['accuracy'] - xgb_result['metrics']['baseline'],
            'features': xgb_result['n_features'],
        },
        'lstm': {
            'accuracy': lstm_result['metrics']['accuracy'],
            'f1': lstm_result['metrics']['f1'],
            'improvement': lstm_result['metrics']['accuracy'] - lstm_result['metrics']['baseline'],
            'lookback': lstm_result['lookback'],
            'iterations': lstm_result['metrics']['iterations'],
        },
    }
    
    # Determine winner
    if comparison['xgboost']['accuracy'] > comparison['lstm']['accuracy']:
        comparison['winner'] = 'XGBoost'
    elif comparison['lstm']['accuracy'] > comparison['xgboost']['accuracy']:
        comparison['winner'] = 'LSTM/MLP'
    else:
        comparison['winner'] = 'TIE'
    
    print(f"  XGBoost: {xgb_result['metrics']['accuracy']:.1%}  |  LSTM: {lstm_result['metrics']['accuracy']:.1%}")
    print(f"  Winner: {comparison['winner']}")
    
    return comparison
