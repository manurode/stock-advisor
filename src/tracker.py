"""Daily prediction tracking and paper trading system.

Every day the system:
1. Generates predictions for all tickers → saves to predictions/{date}.json
2. The next day, fetches actual price changes
3. Compares predictions vs reality
4. Maintains a running accuracy log → predictions/accuracy.json
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

from src.config import PREDICTIONS_DIR, DEFAULT_TICKERS
from src.models.train import predict_with_sentiment
from src.features.sentiment import get_ticker_sentiment
from src.collectors.price import fetch_prices, get_current_price


def save_daily_predictions(
    tickers: list[str] | None = None,
    date: str | None = None,
) -> Path:
    """Generate and save predictions for today.
    
    Args:
        tickers: List of tickers (default: all configured)
        date: Date string YYYY-MM-DD (default: today)
    
    Returns:
        Path to the saved prediction file
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    predictions = []
    
    for ticker in tickers:
        try:
            sentiment = get_ticker_sentiment(ticker)
            pred = predict_with_sentiment(ticker, sentiment=sentiment)
            
            # Get current price
            try:
                price = get_current_price(ticker)
            except Exception:
                price = None
            
            predictions.append({
                'ticker': ticker,
                'date': date,
                'prediction': pred.get('prediction', 'ERROR'),
                'confidence': pred.get('confidence', 0),
                'prob_up': pred.get('prob_up', 0),
                'method': pred.get('method', 'unknown'),
                'ml_direction': pred.get('ml_prediction', ''),
                'sentiment_label': pred.get('sentiment_label', ''),
                'sentiment_compound': pred.get('sentiment_compound', 0),
                'signal': pred.get('signal', ''),
                'price_at_prediction': price,
                'error': pred.get('error'),
            })
        except Exception as e:
            predictions.append({
                'ticker': ticker,
                'date': date,
                'prediction': 'ERROR',
                'error': str(e),
            })
    
    # Save
    path = PREDICTIONS_DIR / f"{date}.json"
    path.write_text(json.dumps(predictions, indent=2, ensure_ascii=False, default=str))
    
    return path


def evaluate_predictions(date: str) -> dict:
    """Compare predictions from a given date with actual next-day outcomes.
    
    Args:
        date: Date of the predictions (YYYY-MM-DD)
    
    Returns:
        dict with per-ticker results and aggregate accuracy
    """
    pred_path = PREDICTIONS_DIR / f"{date}.json"
    
    if not pred_path.exists():
        return {'error': f'No predictions found for {date}'}
    
    predictions = json.loads(pred_path.read_text())
    
    results = []
    correct = 0
    total = 0
    errors = 0
    
    for pred in predictions:
        ticker = pred['ticker']
        
        if pred.get('error'):
            errors += 1
            results.append({**pred, 'actual': 'ERROR', 'match': False})
            continue
        
        try:
            # Get price data around the prediction date
            df = fetch_prices(ticker, period="5d")
            
            # Find the prediction date and next trading day
            pred_date = datetime.strptime(date, '%Y-%m-%d')
            df_dates = df.index.tz_localize(None).normalize()
            
            pred_idx = None
            for i, d in enumerate(df_dates):
                if d.date() == pred_date.date():
                    pred_idx = i
                    break
            
            if pred_idx is None or pred_idx >= len(df) - 1:
                results.append({**pred, 'actual': 'NO_DATA', 'match': None})
                continue
            
            # Compare
            price_before = float(df['Close'].iloc[pred_idx])
            price_after = float(df['Close'].iloc[pred_idx + 1])
            actual_direction = 'UP' if price_after > price_before else 'DOWN'
            actual_change_pct = round((price_after / price_before - 1) * 100, 2)
            
            predicted_direction = pred['prediction']
            matched = (predicted_direction == actual_direction)
            
            if matched:
                correct += 1
            total += 1
            
            results.append({
                **pred,
                'actual': actual_direction,
                'actual_change_pct': actual_change_pct,
                'price_before': round(price_before, 2),
                'price_after': round(price_after, 2),
                'match': matched,
            })
        except Exception as e:
            errors += 1
            results.append({**pred, 'actual': 'ERROR', 'match': False, 'eval_error': str(e)})
    
    accuracy = correct / total if total > 0 else 0
    
    summary = {
        'date': date,
        'total_predictions': len(predictions),
        'evaluated': total,
        'correct': correct,
        'errors': errors,
        'accuracy': round(accuracy, 4),
        'results': results,
    }
    
    # Save evaluation
    eval_path = PREDICTIONS_DIR / f"{date}_eval.json"
    eval_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    
    return summary


def update_accuracy_log(date: str) -> dict:
    """Evaluate predictions and update the running accuracy log."""
    eval_result = evaluate_predictions(date)
    
    if 'error' in eval_result:
        return eval_result
    
    # Load or create accuracy log
    log_path = PREDICTIONS_DIR / "accuracy_log.json"
    
    if log_path.exists():
        log = json.loads(log_path.read_text())
    else:
        log = {
            'created': datetime.now().isoformat(),
            'entries': [],
            'summary': {
                'total_predictions': 0,
                'total_correct': 0,
                'overall_accuracy': 0,
                'by_ticker': {},
            }
        }
    
    # Add entry
    entry = {
        'date': date,
        'accuracy': eval_result['accuracy'],
        'correct': eval_result['correct'],
        'evaluated': eval_result['evaluated'],
        'errors': eval_result['errors'],
    }
    log['entries'].append(entry)
    
    # Update summary
    log['summary']['total_predictions'] += eval_result['evaluated']
    log['summary']['total_correct'] += eval_result['correct']
    log['summary']['overall_accuracy'] = round(
        log['summary']['total_correct'] / log['summary']['total_predictions'], 4
    ) if log['summary']['total_predictions'] > 0 else 0
    
    # Per-ticker tracking
    for r in eval_result['results']:
        ticker = r['ticker']
        if ticker not in log['summary']['by_ticker']:
            log['summary']['by_ticker'][ticker] = {'correct': 0, 'total': 0}
        
        if r.get('match') is not None:
            log['summary']['by_ticker'][ticker]['total'] += 1
            if r['match']:
                log['summary']['by_ticker'][ticker]['correct'] += 1
    
    # Calculate per-ticker accuracy
    for ticker, stats in log['summary']['by_ticker'].items():
        if stats['total'] > 0:
            stats['accuracy'] = round(stats['correct'] / stats['total'], 4)
    
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False, default=str))
    
    return {'date': date, 'accuracy': eval_result['accuracy'], 'log_entries': len(log['entries'])}


def get_accuracy_summary() -> dict:
    """Get the current accuracy summary from the log."""
    log_path = PREDICTIONS_DIR / "accuracy_log.json"
    
    if not log_path.exists():
        return {'error': 'No accuracy log yet. Run predict + track first.'}
    
    log = json.loads(log_path.read_text())
    return log['summary']


def print_tracker_report():
    """Print a comprehensive tracking report."""
    summary = get_accuracy_summary()
    
    if 'error' in summary:
        print(summary['error'])
        return
    
    print(f"\n{'='*60}")
    print(f"📊 PAPER TRADING ACCURACY REPORT")
    print(f"{'='*60}\n")
    
    print(f"  Overall: {summary['overall_accuracy']:.1%} "
          f"({summary['total_correct']}/{summary['total_predictions']} correct)\n")
    
    # By ticker
    if summary['by_ticker']:
        print(f"  {'Ticker':<8} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
        print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*8}")
        
        sorted_tickers = sorted(
            summary['by_ticker'].items(),
            key=lambda x: x[1].get('accuracy', 0),
            reverse=True,
        )
        
        for ticker, stats in sorted_tickers:
            acc = stats.get('accuracy', 0)
            corr = stats.get('correct', 0)
            tot = stats.get('total', 0)
            bar = '█' * int(acc * 20) + '░' * (20 - int(acc * 20))
            print(f"  {ticker:<8} {acc:>9.1%} {corr:>9}/{tot:<7} {bar}")
