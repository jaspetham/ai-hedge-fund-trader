from graph.state import AgentState, show_agent_reasoning
from tools.api import (
    get_financial_metrics,
    get_market_cap,
    search_line_items,
    get_insider_trades,
    get_company_news,
    get_prices,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm
import statistics
import numpy as np


class StanleyDruckenmillerSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def stanley_druckenmiller_agent(state: AgentState):
    """
    Analyzes stocks using Stanley Druckenmiller's investing principles:
      - Seeking asymmetric risk-reward opportunities
      - Emphasizing growth, momentum, and sentiment
      - Willing to be aggressive if conditions are favorable
      - Focus on preserving capital by avoiding high-risk, low-reward bets

    Returns a bullish/bearish/neutral signal with confidence and reasoning.
    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    druck_analysis = {}

    for ticker in tickers:
        progress.update_status("stanley_druckenmiller_agent", ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5)

        progress.update_status("stanley_druckenmiller_agent", ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "earnings_per_share",
                "net_income",
                "operating_income",
                "gross_margin",
                "operating_margin",
                "free_cash_flow",
                "capital_expenditure",
                "cash_and_equivalents",
                "total_debt",
                "shareholders_equity",
                "outstanding_shares",
                "ebit",
                "ebitda",
            ],
            end_date,
            period="annual",
            limit=5,
        )

        progress.update_status("stanley_druckenmiller_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date)

        progress.update_status("stanley_druckenmiller_agent", ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(ticker, end_date, start_date=None, limit=50)
        if not insider_trades:
            progress.update_status("stanley_druckenmiller_agent", ticker, "No insider trades available")

        progress.update_status("stanley_druckenmiller_agent", ticker, "Fetching company news")
        company_news = get_company_news(ticker, end_date, start_date=None, limit=50)
        if not company_news:
            progress.update_status("stanley_druckenmiller_agent", ticker, "No company news available")

        progress.update_status("stanley_druckenmiller_agent", ticker, "Fetching recent price data")
        prices = get_prices(ticker, start_date=start_date, end_date=end_date)

        progress.update_status("stanley_druckenmiller_agent", ticker, "Analyzing growth & momentum")
        growth_momentum_analysis = analyze_growth_and_momentum(financial_line_items, prices)

        progress.update_status("stanley_druckenmiller_agent", ticker, "Analyzing sentiment")
        sentiment_analysis = analyze_sentiment(company_news)

        progress.update_status("stanley_druckenmiller_agent", ticker, "Analyzing insider activity")
        insider_activity = analyze_insider_activity(insider_trades)

        progress.update_status("stanley_druckenmiller_agent", ticker, "Analyzing risk-reward")
        risk_reward_analysis = analyze_risk_reward(financial_line_items, market_cap, prices)

        progress.update_status("stanley_druckenmiller_agent", ticker, "Performing Druckenmiller-style valuation")
        valuation_analysis = analyze_druckenmiller_valuation(financial_line_items, market_cap)

        # Combine partial scores with weights typical for Druckenmiller:
        #   35% Growth/Momentum, 20% Risk/Reward, 20% Valuation,
        #   15% Sentiment, 10% Insider Activity = 100%
        total_score = (
            growth_momentum_analysis["score"] * 0.35
            + risk_reward_analysis["score"] * 0.20
            + valuation_analysis["score"] * 0.20
            + sentiment_analysis["score"] * 0.15
            + insider_activity["score"] * 0.10
        )

        max_possible_score = 10

        # Simple bullish/neutral/bearish signal
        if total_score >= 7.5:
            signal = "bullish"
        elif total_score <= 4.5:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "growth_momentum_analysis": growth_momentum_analysis,
            "sentiment_analysis": sentiment_analysis,
            "insider_activity": insider_activity,
            "risk_reward_analysis": risk_reward_analysis,
            "valuation_analysis": valuation_analysis,
        }

        progress.update_status("stanley_druckenmiller_agent", ticker, "Generating Stanley Druckenmiller analysis")
        druck_output = generate_druckenmiller_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        druck_analysis[ticker] = {
            "signal": druck_output.signal,
            "confidence": druck_output.confidence,
            "reasoning": druck_output.reasoning,
        }

        progress.update_status("stanley_druckenmiller_agent", ticker, "Done")

    message = HumanMessage(content=json.dumps(druck_analysis), name="stanley_druckenmiller_agent")

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(druck_analysis, "Stanley Druckenmiller Agent")

    state["data"]["analyst_signals"]["stanley_druckenmiller_agent"] = druck_analysis
    return {"messages": [message], "data": state["data"]}


def analyze_growth_and_momentum(financial_line_items: list, prices: list) -> dict:
    """
    Evaluate:
      - Revenue Growth (YoY)
      - EPS Growth (YoY)
      - Price Momentum
      - Volatility (new)
    """
    if not financial_line_items or len(financial_line_items) < 2:
        return {"score": 0, "details": "Insufficient financial data for growth analysis"}

    details = []
    raw_score = 0  # Max 12 points (3 each for 4 metrics), normalized to 10

    # 1. Revenue Growth
    revenues = [getattr(fi, "revenue", None) for fi in financial_line_items if getattr(fi, "revenue", None) is not None]
    if len(revenues) >= 2:
        latest_rev, older_rev = revenues[0], revenues[-1]
        if older_rev > 0:
            rev_growth = (latest_rev - older_rev) / older_rev
            if rev_growth > 0.30:
                raw_score += 3
                details.append(f"Strong revenue growth: {rev_growth:.1%}")
            elif rev_growth > 0.15:
                raw_score += 2
                details.append(f"Moderate revenue growth: {rev_growth:.1%}")
            elif rev_growth > 0.05:
                raw_score += 1
                details.append(f"Slight revenue growth: {rev_growth:.1%}")
            else:
                details.append(f"Minimal/negative revenue growth: {rev_growth:.1%}")
        else:
            details.append("Older revenue is zero/negative")
    else:
        details.append("Insufficient revenue data")

    # 2. EPS Growth
    eps_values = [getattr(fi, "earnings_per_share", None) for fi in financial_line_items if getattr(fi, "earnings_per_share", None) is not None]
    if len(eps_values) >= 2:
        latest_eps, older_eps = eps_values[0], eps_values[-1]
        if abs(older_eps) > 1e-9:
            eps_growth = (latest_eps - older_eps) / abs(older_eps)
            if eps_growth > 0.30:
                raw_score += 3
                details.append(f"Strong EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.15:
                raw_score += 2
                details.append(f"Moderate EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.05:
                raw_score += 1
                details.append(f"Slight EPS growth: {eps_growth:.1%}")
            else:
                details.append(f"Minimal/negative EPS growth: {eps_growth:.1%}")
        else:
            details.append("Older EPS near zero")
    else:
        details.append("Insufficient EPS data")

    # 3. Price Momentum
    if prices and len(prices) > 30:
        close_prices = [getattr(p, "close", 0) for p in prices]
        if len(close_prices) >= 2 and close_prices[0] > 0:
            pct_change = (close_prices[-1] - close_prices[0]) / close_prices[0]
            if pct_change > 0.50:
                raw_score += 3
                details.append(f"Very strong momentum: {pct_change:.1%}")
            elif pct_change > 0.20:
                raw_score += 2
                details.append(f"Moderate momentum: {pct_change:.1%}")
            elif pct_change > 0:
                raw_score += 1
                details.append(f"Slight momentum: {pct_change:.1%}")
            else:
                details.append(f"Negative momentum: {pct_change:.1%}")
        else:
            details.append("Invalid price data for momentum")
    else:
        details.append("Insufficient price data")

    # 4. Volatility (Druckenmiller favors stability in momentum)
    if prices and len(prices) > 30:
        closes = [getattr(p, "close", 0) for p in prices]
        if len(closes) >= 30:
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns) * np.sqrt(252)
            if volatility < 0.20:
                raw_score += 3
                details.append(f"Low volatility: {volatility:.2%}")
            elif volatility < 0.30:
                raw_score += 2
                details.append(f"Moderate volatility: {volatility:.2%}")
            elif volatility < 0.40:
                raw_score += 1
                details.append(f"High volatility: {volatility:.2%}")
            else:
                details.append(f"Very high volatility: {volatility:.2%}")
        else:
            details.append("Insufficient data for volatility")
    else:
        details.append("No price data for volatility")

    final_score = min(10, (raw_score / 12) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_insider_activity(insider_trades: list) -> dict:
    """
    Simple insider-trade analysis:
      - If there's heavy insider buying, we nudge the score up.
      - If there's mostly selling, we reduce it.
      - Otherwise, neutral.
    """
    score = 5  # Default neutral
    details = []

    if not insider_trades:
        details.append("No insider trades; neutral")
        return {"score": score, "details": "; ".join(details)}

    buys, sells = 0, 0
    for trade in insider_trades:
        shares = getattr(trade, "transaction_shares", None)
        if shares is not None:
            if shares > 0:
                buys += 1
            elif shares < 0:
                sells += 1

    total = buys + sells
    if total == 0:
        details.append("No transactions; neutral")
        return {"score": score, "details": "; ".join(details)}

    buy_ratio = buys / total
    if buy_ratio > 0.7:
        score = 8
        details.append(f"Heavy buying: {buys} buys vs {sells} sells")
    elif buy_ratio > 0.4:
        score = 6
        details.append(f"Moderate buying: {buys} buys vs {sells} sells")
    else:
        score = 4
        details.append(f"Mostly selling: {buys} buys vs {sells} sells")

    return {"score": score, "details": "; ".join(details)}


def analyze_sentiment(news_items: list) -> dict:
    """
    Basic news sentiment: negative keyword check vs. overall volume.
    """
    if not news_items:
        return {"score": 5, "details": "No news; neutral"}

    negative_keywords = ["lawsuit", "fraud", "negative", "downturn", "decline", "investigation", "recall"]
    negative_count = 0
    for news in news_items:
        title = getattr(news, "title", "") or ""
        if any(word in title.lower() for word in negative_keywords):
            negative_count += 1

    details = []
    if negative_count > len(news_items) * 0.3:
        score = 3
        details.append(f"High negative headlines: {negative_count}/{len(news_items)}")
    elif negative_count > 0:
        score = 6
        details.append(f"Some negative headlines: {negative_count}/{len(news_items)}")
    else:
        score = 8
        details.append("Mostly positive/neutral headlines")

    return {"score": score, "details": "; ".join(details)}


def analyze_risk_reward(financial_line_items: list, market_cap: float | None, prices: list) -> dict:
    """
    Assesses risk via:
      - Debt-to-Equity
      - Price Volatility
    Aims for strong upside with contained downside.
    """
    if not financial_line_items or not prices:
        return {"score": 0, "details": "Insufficient data"}

    details = []
    raw_score = 0  # Max 6 points (3 each)

    # 1. Debt-to-Equity
    debt_values = [getattr(fi, "total_debt", 0) for fi in financial_line_items]
    equity_values = [getattr(fi, "shareholders_equity", None) for fi in financial_line_items if getattr(fi, "shareholders_equity", None) is not None]
    if debt_values and equity_values and len(debt_values) == len(equity_values):
        recent_debt = debt_values[0]
        recent_equity = equity_values[0] if equity_values[0] else 1e-9
        de_ratio = recent_debt / recent_equity
        if de_ratio < 0.3:
            raw_score += 3
            details.append(f"Low debt-to-equity: {de_ratio:.2f}")
        elif de_ratio < 0.7:
            raw_score += 2
            details.append(f"Moderate debt-to-equity: {de_ratio:.2f}")
        elif de_ratio < 1.5:
            raw_score += 1
            details.append(f"Somewhat high debt-to-equity: {de_ratio:.2f}")
        else:
            details.append(f"High debt-to-equity: {de_ratio:.2f}")
    else:
        details.append("No debt/equity data")

    # 2. Volatility (Druckenmiller avoids excessive risk)
    if len(prices) > 30:
        closes = [getattr(p, "close", 0) for p in prices]
        if len(closes) >= 30:
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns) * np.sqrt(252)
            if volatility < 0.20:
                raw_score += 3
                details.append(f"Low volatility: {volatility:.2%}")
            elif volatility < 0.30:
                raw_score += 2
                details.append(f"Moderate volatility: {volatility:.2%}")
            elif volatility < 0.40:
                raw_score += 1
                details.append(f"High volatility: {volatility:.2%}")
            else:
                details.append(f"Very high volatility: {volatility:.2%}")
        else:
            details.append("Insufficient price data for volatility")
    else:
        details.append("No price data for volatility")

    final_score = min(10, (raw_score / 6) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_druckenmiller_valuation(financial_line_items: list, market_cap: float | None) -> dict:
    """
    Druckenmiller is willing to pay up for growth, but still checks:
      - P/E
      - P/FCF
      - EV/EBIT
      - EV/EBITDA
    Each can yield up to 2 points => max 8 raw points => scale to 0–10.
    """
    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "Insufficient data"}

    details = []
    raw_score = 0  # Max 8 points (2 each for 4 metrics)

    recent_debt = getattr(financial_line_items[0], "total_debt", 0)
    recent_cash = getattr(financial_line_items[0], "cash_and_equivalents", 0)
    enterprise_value = market_cap + recent_debt - recent_cash

    # 1. P/E
    recent_net_income = getattr(financial_line_items[0], "net_income", None)
    if recent_net_income and recent_net_income > 0:
        pe = market_cap / recent_net_income
        if pe < 15:
            raw_score += 2
            details.append(f"Attractive P/E: {pe:.2f}")
        elif pe < 25:
            raw_score += 1
            details.append(f"Fair P/E: {pe:.2f}")
        else:
            details.append(f"High P/E: {pe:.2f}")
    else:
        details.append("No net income for P/E")

    # 2. P/FCF
    recent_fcf = getattr(financial_line_items[0], "free_cash_flow", None)
    if recent_fcf and recent_fcf > 0:
        pfcf = market_cap / recent_fcf
        if pfcf < 15:
            raw_score += 2
            details.append(f"Attractive P/FCF: {pfcf:.2f}")
        elif pfcf < 25:
            raw_score += 1
            details.append(f"Fair P/FCF: {pfcf:.2f}")
        else:
            details.append(f"High P/FCF: {pfcf:.2f}")
    else:
        details.append("No positive free cash flow for P/FCF calculation")

    # 3. EV/EBIT
    recent_ebit = getattr(financial_line_items[0], "ebit", None)
    if enterprise_value > 0 and recent_ebit and recent_ebit > 0:
        ev_ebit = enterprise_value / recent_ebit
        if ev_ebit < 15:
            raw_score += 2
            details.append(f"Attractive EV/EBIT: {ev_ebit:.2f}")
        elif ev_ebit < 25:
            raw_score += 1
            details.append(f"Fair EV/EBIT: {ev_ebit:.2f}")
        else:
            details.append(f"High EV/EBIT: {ev_ebit:.2f}")
    else:
        details.append("No EV/EBIT data")

    # 4. EV/EBITDA
    recent_ebitda = getattr(financial_line_items[0], "ebitda", None)
    if enterprise_value > 0 and recent_ebitda and recent_ebitda > 0:
        ev_ebitda = enterprise_value / recent_ebitda
        if ev_ebitda < 10:
            raw_score += 2
            details.append(f"Attractive EV/EBITDA: {ev_ebitda:.2f}")
        elif ev_ebitda < 18:
            raw_score += 1
            details.append(f"Fair EV/EBITDA: {ev_ebitda:.2f}")
        else:
            details.append(f"High EV/EBITDA: {ev_ebitda:.2f}")
    else:
        details.append("No EV/EBITDA data")

    final_score = min(10, (raw_score / 8) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def generate_druckenmiller_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> StanleyDruckenmillerSignal:
    """
    Generates a JSON signal in the style of Stanley Druckenmiller.
    """
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Stanley Druckenmiller AI agent, making investment decisions using his principles:

            1. Seek asymmetric risk-reward opportunities (large upside, limited downside).
            2. Emphasize growth, momentum, and market sentiment.
            3. Preserve capital by avoiding major drawdowns.
            4. Willing to pay higher valuations for true growth leaders.
            5. Be aggressive when conviction is high.
            6. Cut losses quickly if the thesis changes.

            Rules:
            - Reward companies showing strong revenue/earnings growth and positive stock momentum.
            - Evaluate sentiment and insider activity as supportive or contradictory signals.
            - Watch out for high leverage or extreme volatility that threatens capital.
            - Output a JSON object with signal, confidence, and a reasoning string.
            """,
            ),
            (
                "human",
                """Based on the following analysis, create a Druckenmiller-style investment signal.

            Analysis Data for {ticker}:
            {analysis_data}

            Return the trading signal in this JSON format:
            {{
              "signal": "bullish/bearish/neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def create_default_signal():
        return StanleyDruckenmillerSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral"
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=StanleyDruckenmillerSignal,
        agent_name="stanley_druckenmiller_agent",
        default_factory=create_default_signal,
    )