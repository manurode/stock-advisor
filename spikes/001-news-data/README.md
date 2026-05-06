# Spike 001: News Data Availability for US Stocks

## Question
Can we get sufficient daily news for US stocks using free APIs?

## What we tested
- **yfinance `.news` property**: 10 articles per stock (AAPL, TSLA, MSFT, AMZN, GOOGL, NVDA, META, JPM) — consistently 10 for all
- **yfinance `.get_news(count=20)`**: Returns up to 20 articles per stock
- **Article quality**: Full titles, summaries, timestamps, sources (Bloomberg, Barrons, Motley Fool, Yahoo Finance)

## What worked
- ✅ 10-20 articles per stock consistently
- ✅ Rich metadata (title, summary, source, timestamp, content type)
- ✅ Legitimate financial sources (Bloomberg, Barrons, Motley Fool)
- ✅ No API key required
- ✅ Articles include sector-wide news (useful for context)

## What didn't
- ⚠️ Not all articles are ticker-specific — some are sector/industry news
- ⚠️ Some older articles mixed in (but majority are recent)

## Surprises
- yfinance news is much more generous than expected — 10 articles per stock with no rate limiting issues
- Article timestamps sometimes vary in format (ISO strings vs unix timestamps)

## Verdict: VALIDATED

**yfinance news is sufficient for the full app.** We can get 10-20 articles per stock with rich metadata, from legitimate financial sources, without any API key.

### Recommendation for the real build
- Use yfinance as primary news source
- Filter to last 48h articles only
- Normalize timestamps (handle both string and int formats)
- Consider adding Alpha Vantage NEWS_SENTIMENT as backup (25 free req/day)
- Filter out VIDEO type, keep STORY type for NLP analysis
