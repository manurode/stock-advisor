"""Sentiment analysis using VADER."""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.collectors.news import fetch_news, get_recent_news
from src.config import BULLISH_THRESHOLD, BEARISH_THRESHOLD

# Singleton analyzer — initialized once
_analyzer = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    """Get or create the VADER sentiment analyzer (lazy init)."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def analyze_text(text: str) -> dict[str, float]:
    """Analyze sentiment of a single text string.
    
    Returns:
        dict with 'compound', 'pos', 'neg', 'neu' scores
    """
    analyzer = _get_analyzer()
    return analyzer.polarity_scores(str(text))


def analyze_articles(articles: list[dict]) -> dict:
    """Analyze sentiment across multiple news articles.
    
    Args:
        articles: List of normalized article dicts with 'text' field
    
    Returns:
        dict with aggregate sentiment metrics
    """
    if not articles:
        return {
            'compound': 0.0,
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 0.0,
            'article_count': 0,
            'scores': [],
            'label': 'NO_DATA',
        }
    
    scores = []
    for article in articles:
        text = article.get('text', '')
        if text.strip():
            result = analyze_text(text)
            scores.append(result['compound'])
    
    if not scores:
        return {
            'compound': 0.0,
            'positive': 0.0, 'negative': 0.0, 'neutral': 0.0,
            'article_count': 0, 'scores': [], 'label': 'NO_DATA',
        }
    
    import numpy as np
    avg_compound = float(np.mean(scores))
    pos_count = sum(1 for s in scores if s > 0.05)
    neg_count = sum(1 for s in scores if s < -0.05)
    neu_count = sum(1 for s in scores if -0.05 <= s <= 0.05)
    
    # Determine label
    if avg_compound > BULLISH_THRESHOLD:
        label = 'BULLISH'
    elif avg_compound < BEARISH_THRESHOLD:
        label = 'BEARISH'
    else:
        label = 'NEUTRAL'
    
    return {
        'compound': round(avg_compound, 4),
        'positive': pos_count,
        'negative': neg_count,
        'neutral': neu_count,
        'article_count': len(scores),
        'scores': [round(s, 4) for s in scores],
        'label': label,
    }


def get_ticker_sentiment(ticker: str, recent_only: bool = True) -> dict:
    """Get sentiment analysis for a ticker's current news.
    
    Args:
        ticker: Stock symbol
        recent_only: If True, only use news from last 48 hours
    
    Returns:
        dict with sentiment metrics + article metadata
    """
    if recent_only:
        articles = get_recent_news(ticker)
        # Fallback to all recent if none in 48h window
        if not articles:
            articles = fetch_news(ticker, count=10)
    else:
        articles = fetch_news(ticker)
    
    sentiment = analyze_articles(articles)
    
    # Add context about the articles analyzed
    sentiment['ticker'] = ticker
    sentiment['articles_total'] = len(articles)
    if articles:
        sentiment['latest_headline'] = articles[0].get('title', '')
        sentiment['latest_source'] = articles[0].get('source', '')
    
    return sentiment


def batch_sentiment(tickers: list[str]) -> dict[str, dict]:
    """Get sentiment for multiple tickers at once."""
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = get_ticker_sentiment(ticker)
        except Exception as e:
            results[ticker] = {'ticker': ticker, 'label': 'ERROR', 'error': str(e)}
    return results
