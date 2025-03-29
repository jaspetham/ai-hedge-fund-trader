from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
import json
from tools.api import get_financial_metrics, get_prices
from datetime import datetime, timedelta
import numpy as np


##### Fundamental Agent #####
def fundamentals_agent(state: AgentState):
    """Analyzes fundamental data and generates trading signals for multiple tickers."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    fundamental_analysis = {}

    for ticker in tickers:
        progress.update_status("fundamentals_agent", ticker, "Fetching financial metrics")
        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
            limit=10,
        )

        if not financial_metrics:
            progress.update_status("fundamentals_agent", ticker, "Failed: No financial metrics found")
            continue

        metrics = financial_metrics[0]

        progress.update_status("fundamentals_agent", ticker, "Fetching price history")
        price_history = get_prices(ticker, start_date=(datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d"), end_date=end_date)

        # Initialize scoring and reasoning
        scores = {}
        reasoning = {}

        progress.update_status("fundamentals_agent", ticker, "Analyzing profitability")
        # 1. Profitability Analysis
        profitability_score, profitability_details = analyze_profitability(metrics)
        scores["profitability"] = profitability_score
        reasoning["profitability_signal"] = {
            "signal": "bullish" if profitability_score >= 7 else "bearish" if profitability_score <= 3 else "neutral",
            "details": profitability_details,
        }

        progress.update_status("fundamentals_agent", ticker, "Analyzing growth")
        # 2. Growth Analysis
        growth_score, growth_details = analyze_growth(metrics)
        scores["growth"] = growth_score
        reasoning["growth_signal"] = {
            "signal": "bullish" if growth_score >= 7 else "bearish" if growth_score <= 3 else "neutral",
            "details": growth_details,
        }

        progress.update_status("fundamentals_agent", ticker, "Analyzing financial health")
        # 3. Financial Health
        health_score, health_details = analyze_financial_health(metrics, price_history)
        scores["financial_health"] = health_score
        reasoning["financial_health_signal"] = {
            "signal": "bullish" if health_score >= 7 else "bearish" if health_score <= 3 else "neutral",
            "details": health_details,
        }

        progress.update_status("fundamentals_agent", ticker, "Analyzing valuation ratios")
        # 4. Valuation Ratios
        valuation_score, valuation_details = analyze_valuation_ratios(metrics)
        scores["valuation"] = valuation_score
        reasoning["price_ratios_signal"] = {
            "signal": "bullish" if valuation_score >= 7 else "bearish" if valuation_score <= 3 else "neutral",
            "details": valuation_details,
        }

        progress.update_status("fundamentals_agent", ticker, "Calculating final signal")
        # Calculate total score (weighted average)
        total_score = (
            scores["profitability"] * 0.3 +
            scores["growth"] * 0.3 +
            scores["financial_health"] * 0.2 +
            scores["valuation"] * 0.2
        )

        if total_score >= 7:
            overall_signal = "bullish"
        elif total_score <= 3:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        confidence = round((abs(total_score - 5) / 5) * 100, 2)  # Centered at 5 (neutral)

        fundamental_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": min(confidence, 100),
            "reasoning": reasoning,
        }

        progress.update_status("fundamentals_agent", ticker, "Done")

    message = HumanMessage(
        content=json.dumps(fundamental_analysis),
        name="fundamentals_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(fundamental_analysis, "Fundamental Analysis Agent")

    state["data"]["analyst_signals"]["fundamentals_agent"] = fundamental_analysis

    return {
        "messages": [message],
        "data": data,
    }


def analyze_profitability(metrics) -> tuple[float, str]:
    """Analyze profitability metrics."""
    return_on_equity = getattr(metrics, "return_on_equity", None)
    net_margin = getattr(metrics, "net_margin", None)
    operating_margin = getattr(metrics, "operating_margin", None)

    score = 0
    if return_on_equity:
        if return_on_equity > 0.20:
            score += 4
        elif return_on_equity > 0.15:
            score += 3
        elif return_on_equity > 0.10:
            score += 2
        elif return_on_equity > 0:
            score += 1
    if net_margin:
        if net_margin > 0.25:
            score += 3
        elif net_margin > 0.20:
            score += 2
        elif net_margin > 0.10:
            score += 1
    if operating_margin:
        if operating_margin > 0.20:
            score += 3
        elif operating_margin > 0.15:
            score += 2
        elif operating_margin > 0.10:
            score += 1

    final_score = min(10, score * 10 / 10)  # Max raw score 10, normalized to 10
    details = (
        f"ROE: {return_on_equity:.2%}" if return_on_equity else "ROE: N/A") + ", " + (
        f"Net Margin: {net_margin:.2%}" if net_margin else "Net Margin: N/A") + ", " + (
        f"Op Margin: {operating_margin:.2%}" if operating_margin else "Op Margin: N/A"
    )
    return final_score, details


def analyze_growth(metrics) -> tuple[float, str]:
    """Analyze growth metrics."""
    revenue_growth = getattr(metrics, "revenue_growth", None)
    earnings_growth = getattr(metrics, "earnings_growth", None)
    book_value_growth = getattr(metrics, "book_value_growth", None)

    score = 0
    if revenue_growth:
        if revenue_growth > 0.20:
            score += 4
        elif revenue_growth > 0.15:
            score += 3
        elif revenue_growth > 0.10:
            score += 2
        elif revenue_growth > 0:
            score += 1
    if earnings_growth:
        if earnings_growth > 0.20:
            score += 3
        elif earnings_growth > 0.15:
            score += 2
        elif earnings_growth > 0.10:
            score += 1
    if book_value_growth:
        if book_value_growth > 0.15:
            score += 3
        elif book_value_growth > 0.10:
            score += 2
        elif book_value_growth > 0:
            score += 1

    final_score = min(10, score * 10 / 10)  # Max raw score 10, normalized to 10
    details = (
        f"Revenue Growth: {revenue_growth:.2%}" if revenue_growth else "Revenue Growth: N/A") + ", " + (
        f"Earnings Growth: {earnings_growth:.2%}" if earnings_growth else "Earnings Growth: N/A"
    )
    return final_score, details


def analyze_financial_health(metrics, price_history: list) -> tuple[float, str]:
    """Analyze financial health with volatility."""
    current_ratio = getattr(metrics, "current_ratio", None)
    debt_to_equity = getattr(metrics, "debt_to_equity", None)
    free_cash_flow_per_share = getattr(metrics, "free_cash_flow_per_share", None)
    earnings_per_share = getattr(metrics, "earnings_per_share", None)

    score = 0
    if current_ratio:
        if current_ratio > 2.0:
            score += 3
        elif current_ratio > 1.5:
            score += 2
        elif current_ratio > 1.0:
            score += 1
    if debt_to_equity:
        if debt_to_equity < 0.3:
            score += 3
        elif debt_to_equity < 0.5:
            score += 2
        elif debt_to_equity < 1.0:
            score += 1
    if free_cash_flow_per_share and earnings_per_share:
        fcf_eps_ratio = free_cash_flow_per_share / earnings_per_share if earnings_per_share else 0
        if fcf_eps_ratio > 1.0:
            score += 3
        elif fcf_eps_ratio > 0.8:
            score += 2
        elif fcf_eps_ratio > 0.5:
            score += 1

    # Volatility (low volatility strengthens financial health)
    if price_history and len(price_history) >= 252:
        prices = [p["close"] for p in price_history if "close" in p]
        if len(prices) >= 252:
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)
            if volatility < 0.20:
                score += 3
            elif volatility < 0.30:
                score += 2
            elif volatility < 0.40:
                score += 1

    final_score = min(10, score * 10 / 12)  # Max raw score 12, normalized to 10
    details = (
        f"Current Ratio: {current_ratio:.2f}" if current_ratio else "Current Ratio: N/A") + ", " + (
        f"D/E: {debt_to_equity:.2f}" if debt_to_equity else "D/E: N/A"
    )
    return final_score, details


def analyze_valuation_ratios(metrics) -> tuple[float, str]:
    """Analyze valuation ratios."""
    pe_ratio = getattr(metrics, "price_to_earnings_ratio", None)
    pb_ratio = getattr(metrics, "price_to_book_ratio", None)
    ps_ratio = getattr(metrics, "price_to_sales_ratio", None)

    score = 0
    if pe_ratio:
        if pe_ratio < 15:
            score += 4
        elif pe_ratio < 20:
            score += 3
        elif pe_ratio < 25:
            score += 2
        elif pe_ratio < 30:
            score += 1
    if pb_ratio:
        if pb_ratio < 2:
            score += 3
        elif pb_ratio < 3:
            score += 2
        elif pb_ratio < 4:
            score += 1
    if ps_ratio:
        if ps_ratio < 3:
            score += 3
        elif ps_ratio < 5:
            score += 2
        elif ps_ratio < 7:
            score += 1

    final_score = min(10, score * 10 / 10)  # Max raw score 10, normalized to 10
    details = (
        f"P/E: {pe_ratio:.2f}" if pe_ratio else "P/E: N/A") + ", " + (
        f"P/B: {pb_ratio:.2f}" if pb_ratio else "P/B: N/A") + ", " + (
        f"P/S: {ps_ratio:.2f}" if ps_ratio else "P/S: N/A"
    )
    return final_score, details