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
import numpy as np


class CathieWoodSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def cathie_wood_agent(state: AgentState):
    """
    Analyzes stocks using Cathie Wood's investing principles and LLM reasoning.
    1. Prioritizes companies with breakthrough technologies or business models
    2. Focuses on industries with rapid adoption curves and massive TAM (Total Addressable Market).
    3. Invests mostly in AI, robotics, genomic sequencing, fintech, and blockchain.
    4. Willing to endure short-term volatility for long-term gains.
    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    cw_analysis = {}

    for ticker in tickers:
        progress.update_status("cathie_wood_agent", ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5)

        progress.update_status("cathie_wood_agent", ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "gross_margin",
                "operating_margin",
                "debt_to_equity",
                "free_cash_flow",
                "total_assets",
                "total_liabilities",
                "dividends_and_other_cash_distributions",
                "outstanding_shares",
                "research_and_development",
                "capital_expenditure",
                "operating_expense",
            ],
            end_date,
            period="annual",
            limit=5,
        )

        progress.update_status("cathie_wood_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date)

        progress.update_status("cathie_wood_agent", ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(ticker, end_date, start_date=None, limit=50)
        if not insider_trades:
            progress.update_status("cathie_wood_agent", ticker, "No insider trades available")

        progress.update_status("cathie_wood_agent", ticker, "Fetching company news")
        company_news = get_company_news(ticker, end_date, start_date=None, limit=50)
        if not company_news:
            progress.update_status("cathie_wood_agent", ticker, "No company news available")

        progress.update_status("cathie_wood_agent", ticker, "Fetching recent price data")
        prices = get_prices(ticker, start_date=start_date, end_date=end_date)

        progress.update_status("cathie_wood_agent", ticker, "Analyzing disruptive potential")
        disruptive_analysis = analyze_disruptive_potential(metrics, financial_line_items)

        progress.update_status("cathie_wood_agent", ticker, "Analyzing innovation-driven growth")
        innovation_analysis = analyze_innovation_growth(metrics, financial_line_items, prices)

        progress.update_status("cathie_wood_agent", ticker, "Analyzing valuation")
        valuation_analysis = analyze_cathie_wood_valuation(financial_line_items, market_cap, prices)

        progress.update_status("cathie_wood_agent", ticker, "Analyzing insider activity")
        insider_analysis = analyze_insider_activity(insider_trades)

        progress.update_status("cathie_wood_agent", ticker, "Analyzing sentiment")
        sentiment_analysis = analyze_sentiment(company_news)

        # Aggregate scoring with weights reflecting Cathie Wood's priorities
        # Disruptive Potential (35%), Innovation Growth (35%), Valuation (20%), Insider (5%), Sentiment (5%)
        total_score = (
            disruptive_analysis["score"] * 0.35
            + innovation_analysis["score"] * 0.35
            + valuation_analysis["score"] * 0.20
            + insider_analysis["score"] * 0.05
            + sentiment_analysis["score"] * 0.05
        )
        max_possible_score = 10  # Normalized to 0-10 scale

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
            "disruptive_analysis": disruptive_analysis,
            "innovation_analysis": innovation_analysis,
            "valuation_analysis": valuation_analysis,
            "insider_analysis": insider_analysis,
            "sentiment_analysis": sentiment_analysis,
        }

        progress.update_status("cathie_wood_agent", ticker, "Generating Cathie Wood analysis")
        cw_output = generate_cathie_wood_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        cw_analysis[ticker] = {
            "signal": cw_output.signal,
            "confidence": cw_output.confidence,
            "reasoning": cw_output.reasoning,
        }

        progress.update_status("cathie_wood_agent", ticker, "Done")

    message = HumanMessage(content=json.dumps(cw_analysis), name="cathie_wood_agent")

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(cw_analysis, "Cathie Wood Agent")

    state["data"]["analyst_signals"]["cathie_wood_agent"] = cw_analysis
    return {"messages": [message], "data": state["data"]}


def analyze_disruptive_potential(metrics: list, financial_line_items: list) -> dict:
    """
     Analyze whether the company has disruptive products, technology, or business model.
    Evaluates multiple dimensions of disruptive potential:
    1. Revenue Growth Acceleration - indicates market adoption
    2. R&D Intensity - shows innovation investment
    3. Gross Margin Trends - suggests pricing power and scalability
    4. Operating Leverage - demonstrates business model efficiency
    5. Market Share Dynamics - indicates competitive position
    """
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data to analyze disruptive potential"}

    # 1. Revenue Growth Acceleration
    revenues = [getattr(item, "revenue", None) for item in financial_line_items]
    valid_revenues = [r for r in revenues if r is not None]
    if len(valid_revenues) >= 3:
        growth_rates = []
        for i in range(len(valid_revenues) - 1):
            if valid_revenues[i] > 0:
                rate = (valid_revenues[i + 1] - valid_revenues[i]) / valid_revenues[i]
                growth_rates.append(rate)
        if len(growth_rates) >= 2 and growth_rates[-1] > growth_rates[0]:
            score += 3
            details.append(f"Revenue growth accelerating: {growth_rates[-1]:.1%} vs {growth_rates[0]:.1%}")
        latest_growth = growth_rates[-1] if growth_rates else 0
        if latest_growth > 0.5:
            score += 3
            details.append(f"Exceptional growth: {latest_growth:.1%}")
        elif latest_growth > 0.2:
            score += 2
            details.append(f"Strong growth: {latest_growth:.1%}")
    else:
        details.append("Insufficient revenue data.")

    # 2. R&D Intensity
    rd_expenses = [getattr(item, "research_and_development", None) for item in financial_line_items]
    valid_rd = [r for r in rd_expenses if r is not None]
    if valid_rd and valid_revenues:
        rd_intensity = valid_rd[0] / valid_revenues[0] if valid_revenues[0] > 0 else 0  # Latest period
        if rd_intensity > 0.15:
            score += 3
            details.append(f"High R&D intensity: {rd_intensity:.1%}")
        elif rd_intensity > 0.08:
            score += 2
            details.append(f"Moderate R&D intensity: {rd_intensity:.1%}")
    else:
        details.append("No R&D data.")

    # 3. Gross Margin Trends
    gross_margins = [getattr(item, "gross_margin", None) for item in financial_line_items]
    valid_margins = [m for m in gross_margins if m is not None]
    if len(valid_margins) >= 2:
        trend = valid_margins[0] - valid_margins[-1]  # Latest - Oldest
        if trend > 0.05:
            score += 2
            details.append(f"Expanding gross margins: +{trend:.1%}")
        if valid_margins[0] > 0.50:
            score += 2
            details.append(f"High gross margin: {valid_margins[0]:.1%}")
    else:
        details.append("Insufficient gross margin data.")

    # 4. Operating Leverage
    op_expenses = [getattr(item, "operating_expense", None) for item in financial_line_items]
    valid_op_ex = [e for e in op_expenses if e is not None]
    if len(valid_revenues) >= 2 and len(valid_op_ex) >= 2:
        rev_growth = (valid_revenues[0] - valid_revenues[-1]) / valid_revenues[-1] if valid_revenues[-1] > 0 else 0
        opex_growth = (valid_op_ex[0] - valid_op_ex[-1]) / valid_op_ex[-1] if valid_op_ex[-1] > 0 else 0
        if rev_growth > opex_growth:
            score += 2
            details.append("Positive operating leverage.")
    else:
        details.append("Insufficient data for operating leverage.")

    final_score = min(10, score * (10 / 12))  # Scale to 0-10, max raw score 12
    return {"score": final_score, "details": "; ".join(details)}


def analyze_innovation_growth(metrics: list, financial_line_items: list, prices: list) -> dict:
    """
    Evaluate innovation and growth potential:
    1. R&D Investment Trends
    2. Free Cash Flow Generation
    3. Operating Efficiency
    4. Capital Allocation
    5. Price Volatility (Cathie tolerates volatility for growth)
    """
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data to analyze innovation-driven growth"}

    # 1. R&D Investment Trends
    rd_expenses = [getattr(item, "research_and_development", None) for item in financial_line_items]
    revenues = [getattr(item, "revenue", None) for item in financial_line_items]
    valid_rd = [r for r in rd_expenses if r is not None]
    valid_rev = [r for r in revenues if r is not None]
    if len(valid_rd) >= 2 and len(valid_rev) >= 2:
        rd_growth = (valid_rd[0] - valid_rd[-1]) / valid_rd[-1] if valid_rd[-1] > 0 else 0
        if rd_growth > 0.5:
            score += 3
            details.append(f"Strong R&D growth: {rd_growth:.1%}")
        elif rd_growth > 0.2:
            score += 2
            details.append(f"Moderate R&D growth: {rd_growth:.1%}")
    else:
        details.append("Insufficient R&D data.")

    # 2. Free Cash Flow Generation
    fcf_vals = [getattr(item, "free_cash_flow", None) for item in financial_line_items]
    valid_fcf = [f for f in fcf_vals if f is not None]
    if valid_fcf:
        positive_fcf = sum(1 for f in valid_fcf if f > 0)
        if positive_fcf >= (len(valid_fcf) // 2 + 1):
            score += 3
            details.append(f"Positive FCF in {positive_fcf}/{len(valid_fcf)} periods.")
    else:
        details.append("No FCF data.")

    # 3. Operating Efficiency
    op_margins = [getattr(item, "operating_margin", None) for item in financial_line_items]
    valid_op_margins = [m for m in op_margins if m is not None]
    if len(valid_op_margins) >= 2 and valid_op_margins[0] > 0.15:
        score += 3
        details.append(f"Strong operating margin: {valid_op_margins[0]:.1%}")
    elif valid_op_margins:
        score += 1
        details.append(f"Operating margin: {valid_op_margins[0]:.1%}")
    else:
        details.append("No operating margin data.")

    # 4. Capital Allocation
    capex = [getattr(item, "capital_expenditure", None) for item in financial_line_items]
    valid_capex = [c for c in capex if c is not None]
    if len(valid_capex) >= 2 and valid_rev:
        capex_intensity = abs(valid_capex[0]) / valid_rev[0] if valid_rev[0] > 0 else 0
        if capex_intensity > 0.10:
            score += 3
            details.append(f"High CAPEX intensity: {capex_intensity:.1%}")
    else:
        details.append("Insufficient CAPEX data.")

    # 5. Price Volatility (Cathie tolerates volatility for growth)
    if prices and len(prices) >= 90:
        close_prices = [getattr(p, "close", 0) for p in prices]
        if len(close_prices) >= 90:
            returns = np.diff(close_prices) / close_prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)
            if volatility > 0.40:
                score += 2
                details.append(f"High price volatility: {volatility:.2%} (acceptable for growth)")
            elif volatility > 0.20:
                score += 1
                details.append(f"Moderate price volatility: {volatility:.2%}")
            else:
                details.append(f"Low price volatility: {volatility:.2%}")
    else:
        details.append("Insufficient price data for volatility analysis")

    final_score = min(10, score * (10 / 14))  # Scale to 0-10, max raw score 14
    return {"score": final_score, "details": "; ".join(details)}


def analyze_cathie_wood_valuation(financial_line_items: list, market_cap: float, prices: list) -> dict:
    """
    Valuation with a growth focus:
    - High growth rate for disruptive companies
    - Price volatility tolerance
    """
    if not financial_line_items or market_cap is None or not prices:
        return {"score": 0, "details": "Insufficient data for valuation"}

    latest = financial_line_items[0]  # Most recent
    fcf = getattr(latest, "free_cash_flow", 0)

    growth_rate = 0.20  # High growth assumption
    discount_rate = 0.15
    terminal_multiple = 25
    projection_years = 5

    if fcf <= 0:
        return {"score": 0, "details": f"No positive FCF; FCF = {fcf}"}

    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv

    terminal_value = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) / ((1 + discount_rate) ** projection_years)
    intrinsic_value = present_value + terminal_value
    margin_of_safety = (intrinsic_value - market_cap) / market_cap if market_cap > 0 else -1

    score = 0
    details = [f"Intrinsic value: ${intrinsic_value:,.2f}", f"Market cap: ${market_cap:,.2f}", f"Margin of safety: {margin_of_safety:.2%}"]
    if margin_of_safety > 0.5:
        score += 6
    elif margin_of_safety > 0.2:
        score += 3

    # Volatility Tolerance (Cathie accepts high volatility)
    if prices and len(prices) >= 90:
        close_prices = [getattr(p, "close", 0) for p in prices]
        if len(close_prices) >= 90:
            returns = np.diff(close_prices) / close_prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)
            details.append(f"Volatility: {volatility:.2%}")
            if volatility > 0.40:
                score += 3
                details.append(f"High volatility: {volatility:.2%} (acceptable for growth)")
            elif volatility > 0.20:
                score += 2
                details.append(f"Moderate volatility: {volatility:.2%}")
            else:
                score += 1
                details.append(f"Low volatility: {volatility:.2%}")
    else:
        details.append("Insufficient price data.")

    final_score = min(10, score * (10 / 9))  # Scale to 0-10, max raw score 9
    return {"score": final_score, "details": "; ".join(details)}


def analyze_insider_activity(insider_trades: list) -> dict:
    """
    Insider buying signals confidence in innovation; selling is less critical.
    """
    score = 5  # Neutral default
    details = []

    if not insider_trades:
        details.append("No insider trades data.")
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
        details.append("No buy/sell transactions.")
        return {"score": score, "details": "; ".join(details)}

    buy_ratio = buys / total
    if buy_ratio > 0.7:
        score = 8
        details.append(f"Strong insider buying: {buys} buys vs. {sells} sells.")
    elif buy_ratio < 0.3:
        score = 4
        details.append(f"More insider selling: {buys} buys vs. {sells} sells.")
    else:
        details.append(f"Balanced insider activity: {buys} buys vs. {sells} sells.")

    return {"score": score, "details": "; ".join(details)}


def analyze_sentiment(news_items: list) -> dict:
    """
    Positive sentiment aligns with disruptive potential; negative news is less of a concern.
    """
    score = 5  # Neutral default
    details = []

    if not news_items:
        details.append("No news data.")
        return {"score": score, "details": "; ".join(details)}

    positive_keywords = ["innovation", "breakthrough", "growth", "adoption", "technology"]
    negative_keywords = ["lawsuit", "decline", "investigation"]
    positive_count = sum(1 for news in news_items if any(word in (getattr(news, "title", "") or "").lower() for word in positive_keywords))
    negative_count = sum(1 for news in news_items if any(word in (getattr(news, "title", "") or "").lower() for word in negative_keywords))

    if positive_count > negative_count and positive_count >= len(news_items) * 0.3:
        score = 8
        details.append(f"Positive news sentiment: {positive_count} positive vs. {negative_count} negative.")
    elif negative_count > positive_count:
        score = 4
        details.append(f"More negative news: {positive_count} positive vs. {negative_count} negative.")
    else:
        details.append(f"Neutral news sentiment: {positive_count} positive vs. {negative_count} negative.")

    return {"score": score, "details": "; ".join(details)}


def generate_cathie_wood_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> CathieWoodSignal:
    """
    Generates investment decisions in the style of Cathie Wood.
    """
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Cathie Wood AI agent, making investment decisions using her principles:\n\n"
            "1. Seek companies leveraging disruptive innovation.\n"
            "2. Emphasize exponential growth potential, large TAM.\n"
            "3. Focus on technology, healthcare, or other future-facing sectors.\n"
            "4. Consider multi-year time horizons for potential breakthroughs.\n"
            "5. Accept higher volatility in pursuit of high returns.\n"
            "6. Evaluate management's vision and ability to invest in R&D.\n\n"
            "Rules:\n"
            "- Identify disruptive or breakthrough technology.\n"
            "- Evaluate strong potential for multi-year revenue growth.\n"
            "- Check if the company can scale effectively in a large market.\n"
            "- Use a growth-biased valuation approach.\n"
            "- Provide a data-driven recommendation (bullish, bearish, or neutral)."""
        ),
        (
            "human",
            """Based on the following analysis, create a Cathie Wood-style investment signal.\n\n"
            "Analysis Data for {ticker}:\n"
            "{analysis_data}\n\n"
            "Return the trading signal in this JSON format:\n"
            "{{\n  \"signal\": \"bullish/bearish/neutral\",\n  \"confidence\": float (0-100),\n  \"reasoning\": \"string\"\n}}"""
        )
    ])

    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker
    })

    def create_default_cathie_wood_signal():
        return CathieWoodSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral"
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=CathieWoodSignal,
        agent_name="cathie_wood_agent",
        default_factory=create_default_cathie_wood_signal,
    )