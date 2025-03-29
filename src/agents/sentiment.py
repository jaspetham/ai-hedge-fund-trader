from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
import pandas as pd
import numpy as np
import json
from tools.api import get_insider_trades, get_company_news, get_prices
from datetime import datetime, timedelta


##### Sentiment Agent #####
def sentiment_agent(state: AgentState):
    """Analyzes market sentiment and generates trading signals for multiple tickers."""
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    portfolio = data.get("portfolio")
    current_prices = data.get("current_prices")

    sentiment_analysis = {}

    for ticker in tickers:
        progress.update_status("sentiment_agent", ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(ticker=ticker, end_date=end_date, limit=1000)

        progress.update_status("sentiment_agent", ticker, "Analyzing trading patterns")
        transaction_shares = pd.Series([getattr(t, "transaction_shares", 0) for t in insider_trades]).dropna()
        insider_signals = np.where(transaction_shares < 0, "bearish", "bullish").tolist()

        progress.update_status("sentiment_agent", ticker, "Fetching company news")
        company_news = get_company_news(ticker, end_date, limit=50)

        sentiment = pd.Series([getattr(n, "sentiment", "neutral") for n in company_news]).dropna()
        news_signals = np.where(sentiment == "negative", "bearish",
                              np.where(sentiment == "positive", "bullish", "neutral")).tolist()

        progress.update_status("sentiment_agent", ticker, "Fetching price history")
        price_history = get_prices(ticker, start_date=(datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d"), end_date=end_date)

        progress.update_status("sentiment_agent", ticker, "Analyzing volatility")
        volatility_score = 0
        if price_history:
            prices = [p["close"] for p in price_history if "close" in p]
            if len(prices) >= 90:
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns) * np.sqrt(252)
                volatility_score = 2 if volatility < 0.20 else 1 if volatility < 0.30 else 0
            else:
                volatility_score = 0
        else:
            volatility_score = 0

        progress.update_status("sentiment_agent", ticker, "Combining signals")
        insider_weight = 0.3
        news_weight = 0.6
        volatility_weight = 0.1

        bullish_signals = (
            insider_signals.count("bullish") * insider_weight +
            news_signals.count("bullish") * news_weight +
            (volatility_score > 0) * volatility_weight
        )
        bearish_signals = (
            insider_signals.count("bearish") * insider_weight +
            news_signals.count("bearish") * news_weight +
            (volatility_score == 0) * volatility_weight
        )

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        total_weighted_signals = len(insider_signals) * insider_weight + len(news_signals) * news_weight + volatility_weight
        confidence = 0
        if total_weighted_signals > 0:
            confidence = round(max(bullish_signals, bearish_signals) / total_weighted_signals, 2) * 100
        reasoning = (f"Weighted Bullish signals: {bullish_signals:.1f}, Weighted Bearish signals: {bearish_signals:.1f}, "
                     f"Volatility Score: {volatility_score}/2 (low volatility strengthens sentiment)")

        sentiment_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status("sentiment_agent", ticker, "Done")

    message = HumanMessage(content=json.dumps(sentiment_analysis), name="sentiment_agent")

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(sentiment_analysis, "Sentiment Analysis Agent")

    state["data"]["analyst_signals"]["sentiment_agent"] = sentiment_analysis

    return {"messages": [message], "data": data}