"""Streamlit dashboard for Stock Advisor."""
import streamlit as st
import pandas as pd
from datetime import datetime

from src.config import DEFAULT_TICKERS
from src.collectors.price import get_price_summary, fetch_prices
from src.collectors.news import fetch_news
from src.features.sentiment import get_ticker_sentiment, batch_sentiment
from src.models.train import predict_next_day, load_model


st.set_page_config(
    page_title="Stock Advisor",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Stock Advisor — AI Investment Assistant")
st.caption("Technical indicators + News sentiment + ML predictions")

# Sidebar: Ticker selection
st.sidebar.header("Configuration")
selected_tickers = st.sidebar.multiselect(
    "Tickers to analyze",
    DEFAULT_TICKERS,
    default=["AAPL", "TSLA", "MSFT", "NVDA"],
)

refresh = st.sidebar.button("🔄 Refresh Data")

# ============================================================
# TAB 1: Today's Overview
# ============================================================
tab1, tab2, tab3 = st.tabs(["📊 Today", "📰 News & Sentiment", "📈 History"])

with tab1:
    st.header("Today's Market Snapshot")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    
    if not selected_tickers:
        st.info("Select tickers in the sidebar to begin.")
    else:
        rows = []
        for ticker in selected_tickers:
            try:
                summary = get_price_summary(ticker)
                sentiment = get_ticker_sentiment(ticker)
                
                rows.append({
                    'Ticker': ticker,
                    'Price': f"${summary.get('price', 'N/A'):,.2f}",
                    'Change': f"{summary.get('change_pct', 0):+.2f}%",
                    'Sentiment': sentiment.get('label', 'N/A'),
                    'Compound': f"{sentiment.get('compound', 0):+.3f}",
                    'Articles': sentiment.get('article_count', 0),
                })
            except Exception as e:
                rows.append({
                    'Ticker': ticker,
                    'Price': 'Error',
                    'Change': '-',
                    'Sentiment': 'ERROR',
                    'Compound': '-',
                    'Articles': 0,
                })
        
        df = pd.DataFrame(rows)
        
        # Color-code sentiment
        def color_sentiment(val):
            if val == 'BULLISH':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'BEARISH':
                return 'background-color: #f8d7da; color: #721c24'
            return ''
        
        styled = df.style.applymap(color_sentiment, subset=['Sentiment'])
        st.dataframe(styled, use_container_width=True, hide_index=True)
        
        # Predictions section
        st.subheader("🤖 ML Predictions")
        st.caption("Predictions from trained XGBoost models (if available)")
        
        pred_rows = []
        for ticker in selected_tickers[:5]:  # Limit to avoid slow loading
            try:
                pred = predict_next_day(ticker)
                if 'error' not in pred:
                    pred_rows.append({
                        'Ticker': ticker,
                        'Prediction': pred['prediction'],
                        'Confidence': f"{pred['confidence']:.1%}",
                        'P(Up)': f"{pred['prob_up']:.1%}",
                    })
                else:
                    pred_rows.append({
                        'Ticker': ticker,
                        'Prediction': '⚠️ No model',
                        'Confidence': '-',
                        'P(Up)': '-',
                    })
            except Exception:
                pred_rows.append({
                    'Ticker': ticker,
                    'Prediction': '⚠️ Error',
                    'Confidence': '-',
                    'P(Up)': '-',
                })
        
        if pred_rows:
            pred_df = pd.DataFrame(pred_rows)
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
        else:
            st.info("No trained models found. Train models first with `python -m src.cli train`.")

# ============================================================
# TAB 2: News & Sentiment
# ============================================================
with tab2:
    st.header("News & Sentiment Analysis")
    
    focus_ticker = st.selectbox("Select ticker for detailed news", selected_tickers)
    
    if focus_ticker:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            sentiment = get_ticker_sentiment(focus_ticker)
            st.metric("Sentiment Label", sentiment.get('label', 'N/A'))
            st.metric("Compound Score", f"{sentiment.get('compound', 0):+.3f}")
            st.metric("Articles Analyzed", sentiment.get('article_count', 0))
            
            pos = sentiment.get('positive', 0)
            neg = sentiment.get('negative', 0)
            neu = sentiment.get('neutral', 0)
            total = pos + neg + neu or 1
            st.write(f"😊 Positive: {pos} ({pos/total*100:.0f}%)")
            st.write(f"😞 Negative: {neg} ({neg/total*100:.0f}%)")
            st.write(f"😐 Neutral:  {neu} ({neu/total*100:.0f}%)")
        
        with col2:
            articles = fetch_news(focus_ticker, count=10)
            for a in articles:
                source = a.get('source', 'Unknown')
                title = a.get('title', 'No title')
                ts = a.get('timestamp', '')[:10]
                
                with st.expander(f"[{ts}] {title[:100]} — {source}"):
                    summary = a.get('summary', 'No summary available')
                    st.write(summary)
                    st.caption(f"Source: {source} | Type: {a.get('type', 'N/A')}")

# ============================================================
# TAB 3: History & Backtesting
# ============================================================
with tab3:
    st.header("Historical Performance")
    
    backtest_ticker = st.selectbox(
        "Select ticker for backtesting",
        selected_tickers,
        key="backtest_ticker",
    )
    
    if backtest_ticker and st.button("Run Backtest"):
        with st.spinner(f"Running walk-forward backtest for {backtest_ticker}..."):
            try:
                from src.backtest.engine import walk_forward_test
                result = walk_forward_test(backtest_ticker)
                
                if 'error' not in result:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{result['accuracy']:.1%}")
                    col2.metric("Baseline", f"{result['baseline']:.1%}")
                    col3.metric("Improvement", f"{result['improvement']:+.1%}")
                    col4.metric("Predictions", result['predictions'])
                    
                    st.caption(f"Test period: {result['test_start']} → {result['test_end']}")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("UP Prediction Accuracy", f"{result['up_accuracy']:.1%}")
                    col2.metric("DOWN Prediction Accuracy", f"{result['down_accuracy']:.1%}")
                else:
                    st.error(result['error'])
            except Exception as e:
                st.error(f"Backtest failed: {e}")


# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Stock Advisor v1.0 — Use at your own risk. Not financial advice.")
