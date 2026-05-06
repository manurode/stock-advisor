# Spike 001: News Data Availability for US Stocks
# ================================================
# Question: Can we get enough daily news for US stocks using free APIs?
#
# Approach:
#   1. yfinance .news property — built-in, no extra API key
#   2. Alpha Vantage NEWS_SENTIMENT endpoint (free tier: 25 req/day)
#   3. Count articles per stock, recency, and relevance
#
# Verdict criteria:
#   - SUFFICIENT: ≥5 articles/day for major stocks (AAPL, TSLA, MSFT)
#   - PARTIAL: 1-4 articles/day
#   - INSUFFICIENT: <1 article/day on average

import yfinance as yf
import json
import time
from datetime import datetime, timedelta

# Major US stocks to test
TICKERS = ["AAPL", "TSLA", "MSFT", "AMZN", "GOOGL", "NVDA", "META", "JPM"]

print("=" * 70)
print("SPIKE 001: NEWS DATA AVAILABILITY")
print("=" * 70)
print(f"Testing {len(TICKERS)} major US stocks: {', '.join(TICKERS)}")
print()

# ============================================================
# PART A: yfinance .news property
# ============================================================
print("─" * 70)
print("PART A: yfinance .news (built-in, no API key)")
print("─" * 70)

results_a = {}
for ticker in TICKERS:
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        results_a[ticker] = len(news) if news else 0
        print(f"  {ticker:6s}: {len(news):3d} news articles")
        if news:
            # Show first article details
            first = news[0]
            title = first.get('title', first.get('content', {}).get('title', 'N/A'))
            pub_time = first.get('providerPublishTime', first.get('content', {}).get('pubDate', 'N/A'))
            if isinstance(pub_time, (int, float)):
                pub_time = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M')
            print(f"         Latest: {title[:80]}...")
            print(f"         Published: {pub_time}")
    except Exception as e:
        results_a[ticker] = 0
        print(f"  {ticker:6s}: ERROR - {str(e)[:60]}")

total_a = sum(results_a.values())
avg_a = total_a / len(TICKERS)
print(f"\n  TOTAL: {total_a} articles across {len(TICKERS)} stocks")
print(f"  AVG:   {avg_a:.1f} articles per stock")

# ============================================================
# PART B: yfinance .get_news() method (newer API)
# ============================================================
print("\n" + "─" * 70)
print("PART B: yfinance .get_news() (count from search)")
print("─" * 70)

for ticker in TICKERS[:4]:  # Test subset to be fast
    try:
        stock = yf.Ticker(ticker)
        search_news = stock.get_news(count=20)
        count = len(search_news) if search_news else 0
        print(f"  {ticker:6s}: {count:3d} articles (get_news)")
        # Check dates
        if search_news:
            dates = []
            for item in search_news[:5]:
                pub = item.get('providerPublishTime', item.get('content', {}).get('pubDate', None))
                if pub and isinstance(pub, (int, float)):
                    dates.append(datetime.fromtimestamp(pub).strftime('%m-%d'))
            if dates:
                print(f"         Recent dates: {', '.join(dates)}")
    except Exception as e:
        print(f"  {ticker:6s}: ERROR - {str(e)[:60]}")

# ============================================================
# PART C: Detailed article analysis for top stock
# ============================================================
print("\n" + "─" * 70)
print("PART C: Deep dive on AAPL (article structure)")
print("─" * 70)

try:
    aapl = yf.Ticker("AAPL")
    news_items = aapl.news[:5]  # First 5
    for i, item in enumerate(news_items):
        print(f"\n  Article #{i+1}:")
        # yfinance news structure can be complex
        if isinstance(item, dict):
            content = item.get('content', item)
            title = item.get('title', content.get('title', 'N/A'))
            summary = item.get('summary', content.get('summary', 'N/A'))
            source = item.get('publisher', content.get('provider', 'N/A'))
            pub_time = item.get('providerPublishTime', content.get('pubDate', 'N/A'))
            if isinstance(pub_time, (int, float)):
                pub_time = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M')
            
            print(f"    Title:   {str(title)[:100]}")
            print(f"    Source:  {source}")
            print(f"    Date:    {pub_time}")
            print(f"    Type:    {item.get('contentType', content.get('contentType', 'N/A'))}")
            # Keywords hint
            summary_str = str(summary)[:150]
            if summary_str:
                print(f"    Summary: {summary_str}...")
        else:
            print(f"    Raw: {str(item)[:200]}")
except Exception as e:
    print(f"  ERROR: {e}")

# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 70)
print("VERDICT: SPIKE 001")
print("=" * 70)

# Count news from last 24h for key stocks
print("\nChecking recency (last 24h)...")
recent_total = 0
for ticker in TICKERS[:3]:
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        recent = 0
        cutoff = datetime.now() - timedelta(hours=24)
        for item in news:
            pub = item.get('providerPublishTime', item.get('content', {}).get('pubDate', 0))
            if isinstance(pub, (int, float)) and pub > 0:
                pub_dt = datetime.fromtimestamp(pub)
                if pub_dt > cutoff:
                    recent += 1
        print(f"  {ticker}: {recent} articles from last 24h")
        recent_total += recent
    except Exception as e:
        print(f"  {ticker}: ERROR - {e}")

print(f"\nTotal recent articles (AAPL+TSLA+MSFT): {recent_total}")

if avg_a >= 5:
    verdict = "VALIDATED — Sufficient news data available"
elif avg_a >= 2:
    verdict = "PARTIAL — Some news available, may need supplementary source"
else:
    verdict = "INVALIDATED — Insufficient free news data"

print(f"\nVerdict: {verdict}")
print(f"Average articles per stock: {avg_a:.1f}")
print(f"\nRecommendation: yfinance news is {'sufficient' if avg_a >= 3 else 'insufficient'}. ", end="")
if avg_a < 5:
    print("We will ALSO integrate Alpha Vantage or NewsAPI free tiers as backup.")
else:
    print("This should be adequate for the full app.")
