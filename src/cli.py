"""CLI tool for Stock Advisor operations.

Usage:
    python -m src.cli predict      # Today's predictions
    python -m src.cli backtest AAPL  # Backtest a single stock
    python -m src.cli train        # Train models for all tickers
    python -m src.cli dashboard    # Launch Streamlit dashboard
    python -m src.cli sentiment    # Show sentiment for all tickers
"""
import sys
import json
import argparse
from datetime import datetime

from src.config import DEFAULT_TICKERS
from src.collectors.price import get_price_summary, fetch_and_cache
from src.collectors.news import fetch_news, save_news
from src.features.sentiment import get_ticker_sentiment, batch_sentiment
from src.models.train import train_and_evaluate, save_model, save_params, predict_next_day, predict_with_sentiment
from src.backtest.engine import walk_forward_test


def cmd_train(args):
    """Train models for specified tickers."""
    tickers = args.tickers or DEFAULT_TICKERS
    do_tune = args.tune
    
    print(f"Training models for {len(tickers)} tickers...")
    if do_tune:
        print(f"🔧 Hyperparameter tuning ENABLED (n_iter={args.n_iter})")
    print("=" * 60)
    
    for ticker in tickers:
        try:
            print(f"\n📈 {ticker}:")
            result = train_and_evaluate(ticker, tune=do_tune, n_iter=args.n_iter)
            save_model(result['model'], ticker)
            save_params(result['params'], ticker)
            
            metrics = result['metrics']
            improvement = metrics['accuracy'] - metrics['baseline']
            
            print(f"   Accuracy:  {metrics['accuracy']:.1%}")
            print(f"   Baseline:  {metrics['baseline']:.1%}")
            print(f"   Δ:         {improvement:+.1%}")
            print(f"   F1:        {metrics['f1']:.1%}")
            print(f"   Features:  {result['n_features']}")
            print(f"   Data:      {result['train_days']} train / {result['test_days']} test days")
            
            if result['tuning']:
                print(f"   Tuned CV:  {result['tuning']['best_score']:.1%}")
                print(f"   Best params: {result['tuning']['best_params']}")
            
            if result['top_features']:
                print(f"   Top feature: {result['top_features'][0]['feature']}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ Training complete. Models saved to models/")


def cmd_predict(args):
    """Generate predictions for specified tickers."""
    tickers = args.tickers or DEFAULT_TICKERS
    
    print(f"\n{'='*60}")
    print(f"📊 STOCK ADVISOR — Predictions for {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*60}\n")
    
    predictions = []
    for ticker in tickers:
        try:
            # Get price summary
            price_info = get_price_summary(ticker)
            
            # Get sentiment
            sentiment = get_ticker_sentiment(ticker)
            
            # Get ML prediction with sentiment blending
            ml_pred = predict_with_sentiment(ticker, sentiment=sentiment)
            
            row = {
                'ticker': ticker,
                'price': price_info.get('price', 0),
                'change_pct': price_info.get('change_pct', 0),
                'sentiment': sentiment.get('label', 'N/A'),
                'compound': sentiment.get('compound', 0),
                'articles': sentiment.get('article_count', 0),
                'ml_prediction': ml_pred.get('prediction', 'N/A'),
                'confidence': ml_pred.get('confidence', 0),
            }
            predictions.append(row)
            
            # Display
            arrow = "📈" if row['change_pct'] >= 0 else "📉"
            sent_emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}.get(row['sentiment'], '⚪')
            
            print(f"  {row['ticker']:<8s} {arrow} ${row['price']:>8.2f} ({row['change_pct']:>+6.2f}%)  "
                  f"{sent_emoji} {row['sentiment']:<8s} ({row['compound']:>+7.3f})  "
                  f"🤖 {row['ml_prediction']:<6s} ({row['confidence']:.0%})")
            
        except Exception as e:
            print(f"  {ticker:<8s} ❌ Error: {str(e)[:50]}")
    
    # Save predictions
    today = datetime.now().strftime('%Y-%m-%d')
    path = __import__('src.config', fromlist=['PREDICTIONS_DIR']).PREDICTIONS_DIR / f"{today}.json"
    path.write_text(json.dumps(predictions, indent=2, default=str))
    print(f"\n  Predictions saved to {path}")
    
    return predictions


def cmd_sentiment(args):
    """Show sentiment analysis for tickers."""
    tickers = args.tickers or DEFAULT_TICKERS[:5]
    
    print(f"\n{'='*70}")
    print(f"📰 SENTIMENT ANALYSIS — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}\n")
    
    for ticker in tickers:
        try:
            sentiment = get_ticker_sentiment(ticker)
            print(f"  {ticker}:")
            print(f"    Label:     {sentiment['label']}")
            print(f"    Compound:  {sentiment['compound']:+.3f}")
            print(f"    Articles:  {sentiment['article_count']} "
                  f"(😊{sentiment['positive']} 😞{sentiment['negative']} 😐{sentiment['neutral']})")
            if sentiment.get('latest_headline'):
                print(f"    Headline:  {sentiment['latest_headline'][:100]}")
            print()
        except Exception as e:
            print(f"  {ticker}: ❌ {e}\n")


def cmd_backtest(args):
    """Run backtest for a ticker."""
    ticker = args.ticker.upper()
    
    print(f"\n{'='*60}")
    print(f"📊 BACKTEST: {ticker}")
    print(f"{'='*60}\n")
    print("Running walk-forward validation (this may take a minute)...")
    
    try:
        result = walk_forward_test(ticker)
        
        if 'error' in result:
            print(f"  ❌ {result['error']}")
            return
        
        print(f"\n  Results ({result['predictions']} trading days):")
        print(f"  {'─'*40}")
        print(f"  Accuracy:     {result['accuracy']:.1%}")
        print(f"  Baseline:     {result['baseline']:.1%}")
        print(f"  Improvement:  {result['improvement']:+.1%}")
        print(f"  UP accuracy:  {result['up_accuracy']:.1%} ({result['up_predictions']} predictions)")
        print(f"  DOWN acc:     {result['down_accuracy']:.1%} ({result['down_predictions']} predictions)")
        print(f"  Period:       {result['test_start']} → {result['test_end']}")
        
    except Exception as e:
        print(f"  ❌ Backtest failed: {e}")


def cmd_dashboard(args):
    """Launch Streamlit dashboard."""
    import subprocess
    import os
    
    dashboard_path = os.path.join(os.path.dirname(__file__), 'dashboard', 'app.py')
    print(f"🚀 Launching dashboard at http://localhost:8501")
    subprocess.run(['streamlit', 'run', dashboard_path])


def main():
    parser = argparse.ArgumentParser(
        description='Stock Advisor CLI — AI-powered investment analysis'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # predict
    pred_parser = subparsers.add_parser('predict', help='Generate today\'s predictions')
    pred_parser.add_argument('--tickers', nargs='*', help='Tickers to predict (default: all)')
    
    # sentiment
    sent_parser = subparsers.add_parser('sentiment', help='Show sentiment analysis')
    sent_parser.add_argument('--tickers', nargs='*', help='Tickers to analyze')
    
    # train
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--tickers', nargs='*', help='Tickers to train (default: all)')
    train_parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    train_parser.add_argument('--n_iter', type=int, default=25, help='Tuning iterations (default: 25)')
    
    # backtest
    back_parser = subparsers.add_parser('backtest', help='Backtest a ticker')
    back_parser.add_argument('ticker', help='Ticker to backtest')
    
    # dashboard
    subparsers.add_parser('dashboard', help='Launch Streamlit dashboard')
    
    args = parser.parse_args()
    
    if args.command == 'predict':
        cmd_predict(args)
    elif args.command == 'sentiment':
        cmd_sentiment(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'backtest':
        cmd_backtest(args)
    elif args.command == 'dashboard':
        cmd_dashboard(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
