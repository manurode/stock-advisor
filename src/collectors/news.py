"""News data collector using yfinance."""
import yfinance as yf
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from src.config import NEWS_DIR, NEWS_LOOKBACK_HOURS, MAX_NEWS_ARTICLES


def fetch_news(ticker: str, count: int = MAX_NEWS_ARTICLES) -> list[dict]:
    """Fetch recent news articles for a ticker.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        count: Maximum number of articles to fetch
    
    Returns:
        List of normalized article dicts
    """
    stock = yf.Ticker(ticker)
    
    try:
        raw_news = stock.news
        if not raw_news:
            raw_news = stock.get_news(count=count)
    except Exception:
        raw_news = []
    
    if not raw_news:
        return []
    
    articles = [normalize_article(a) for a in raw_news[:count]]
    return [a for a in articles if a is not None]


def normalize_article(raw: dict) -> dict | None:
    """Normalize a yfinance news article to a standard format.
    
    The yfinance API sometimes nests data under 'content', sometimes
    at the top level. This handles both cases.
    """
    # Handle nested content structure
    content = raw.get('content', {})
    if not isinstance(content, dict):
        content = {}
    
    # Title: try top-level, then content, then fallback
    title = raw.get('title') or content.get('title', '')
    
    # Summary
    summary = raw.get('summary') or content.get('summary', '')
    
    # Source
    publisher = raw.get('publisher', '')
    if isinstance(publisher, dict):
        publisher = publisher.get('displayName', publisher.get('name', ''))
    elif not publisher:
        publisher = content.get('provider', {}).get('displayName', 'Unknown')
        if isinstance(publisher, dict):
            publisher = publisher.get('displayName', 'Unknown')
    
    # Timestamp
    pub_time = raw.get('providerPublishTime') or content.get('pubDate')
    if isinstance(pub_time, (int, float)) and pub_time > 0:
        pub_dt = datetime.fromtimestamp(pub_time, tz=timezone.utc)
        pub_iso = pub_dt.isoformat()
    elif isinstance(pub_time, str):
        pub_iso = pub_time
        try:
            pub_dt = datetime.fromisoformat(pub_time.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None
    else:
        return None  # No valid timestamp
    
    # Content type
    content_type = raw.get('contentType', content.get('contentType', 'STORY'))
    
    # Skip videos — less useful for text sentiment
    # (Keep them for now, sentiment pipeline filters later)
    
    if not title or len(title.strip()) < 5:
        return None
    
    return {
        'title': str(title).strip(),
        'summary': str(summary).strip() if summary else '',
        'source': str(publisher),
        'timestamp': pub_iso,
        'datetime': pub_dt,
        'type': str(content_type),
        'text': f"{title}. {summary}".strip() if summary else str(title).strip(),
    }


def get_recent_news(ticker: str, hours: int = NEWS_LOOKBACK_HOURS) -> list[dict]:
    """Get news articles from the last N hours."""
    articles = fetch_news(ticker)
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
    
    recent = [
        a for a in articles
        if a['datetime'] > cutoff
    ]
    
    return recent


def save_news(ticker: str, articles: list[dict]) -> Path:
    """Cache news articles to JSON."""
    today = datetime.now().strftime('%Y-%m-%d')
    path = NEWS_DIR / f"{ticker}_{today}.json"
    
    # Convert datetime objects to strings for JSON
    serializable = []
    for a in articles:
        a_copy = {k: v for k, v in a.items() if k != 'datetime'}
        serializable.append(a_copy)
    
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False))
    return path


def load_news(ticker: str, date: str | None = None) -> list[dict]:
    """Load cached news for a ticker on a specific date."""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    path = NEWS_DIR / f"{ticker}_{date}.json"
    if path.exists():
        return json.loads(path.read_text())
    return []
