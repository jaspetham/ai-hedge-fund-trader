from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
import json
from tools.api import get_financial_metrics, get_market_cap, search_line_items, get_prices
from datetime import datetime, timedelta
import numpy as np


##### Valuation Agent #####
def valuation_agent(state: AgentState):
    """Performs detailed valuation analysis using multiple methodologies for multiple tickers."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    portfolio = data["portfolio"]
    current_prices = data["current_prices"]

    valuation_analysis = {}

    for ticker in tickers:
        progress.update_status("valuation_agent", ticker, "Fetching financial data")
        financial_metrics = get_financial_metrics(ticker=ticker, end_date=end_date, period="ttm")

        if not financial_metrics:
            progress.update_status("valuation_agent", ticker, "Failed: No financial metrics found")
            continue
        metrics = financial_metrics[0]

        progress.update_status("valuation_agent", ticker, "Gathering line items")
        financial_line_items = search_line_items(
            ticker=ticker,
            line_items=[
                "free_cash_flow",
                "net_income",
                "depreciation_and_amortization",
                "capital_expenditure",
                "working_capital",
            ],
            end_date=end_date,
            period="ttm",
            limit=2,
        )

        if len(financial_line_items) < 2:
            progress.update_status("valuation_agent", ticker, "Failed: Insufficient financial line items")
            continue

        current_financial_line_item = financial_line_items[0]
        previous_financial_line_item = financial_line_items[1]

        progress.update_status("valuation_agent", ticker, "Fetching price history")
        price_history = get_prices(ticker, start_date=(datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d"), end_date=end_date)

        progress.update_status("valuation_agent", ticker, "Calculating owner earnings")
        working_capital_change = calculate_working_capital_change(
            getattr(current_financial_line_item, "working_capital", 0),
            getattr(previous_financial_line_item, "working_capital", 0)
        )
        growth_rate = getattr(metrics, "earnings_growth", None)
        if growth_rate is None:
            growth_rate = 0.05  # Explicitly set default if None
        owner_earnings_result = calculate_owner_earnings_value(
            net_income=getattr(current_financial_line_item, "net_income", 0),
            depreciation=getattr(current_financial_line_item, "depreciation_and_amortization", 0),
            capex=getattr(current_financial_line_item, "capital_expenditure", 0),
            working_capital_change=working_capital_change,
            growth_rate=growth_rate,
            required_return=0.15,
            margin_of_safety=0.25,
            price_history=price_history
        )

        progress.update_status("valuation_agent", ticker, "Calculating DCF value")
        dcf_result = calculate_intrinsic_value(
            free_cash_flow=getattr(current_financial_line_item, "free_cash_flow", 0),
            growth_rate=growth_rate,
            discount_rate=0.10,
            terminal_growth_rate=0.03,
            num_years=5,
            price_history=price_history
        )

        progress.update_status("valuation_agent", ticker, "Comparing to market value")
        market_cap = get_market_cap(ticker=ticker, end_date=end_date)
        current_price = current_prices.get(ticker)

        total_score = owner_earnings_result["score"] + dcf_result["score"]
        max_score = 10
        raw_max = owner_earnings_result["max_score"] + dcf_result["max_score"]
        normalized_score = (total_score / raw_max) * 10 if raw_max > 0 else 0

        dcf_value = dcf_result["value"]
        owner_earnings_value = owner_earnings_result["value"]
        market_value = market_cap if market_cap else (current_price * getattr(current_financial_line_item, "outstanding_shares", 1))
        dcf_gap = (dcf_value - market_value) / market_value if market_value else 0
        owner_earnings_gap = (owner_earnings_value - market_value) / market_value if market_value else 0
        valuation_gap = (dcf_gap + owner_earnings_gap) / 2

        if normalized_score >= 7 and valuation_gap > 0.15:
            signal = "bullish"
        elif normalized_score <= 3 or valuation_gap < -0.15:
            signal = "bearish"
        else:
            signal = "neutral"

        reasoning = {
            "dcf_analysis": {
                "signal": ("bullish" if dcf_gap > 0.15 else "bearish" if dcf_gap < -0.15 else "neutral"),
                "details": f"Intrinsic Value: ${dcf_value:,.2f}, Market Value: ${market_value:,.2f}, Gap: {dcf_gap:.1%}, Score: {dcf_result['score']}/5",
            },
            "owner_earnings_analysis": {
                "signal": ("bullish" if owner_earnings_gap > 0.15 else "bearish" if owner_earnings_gap < -0.15 else "neutral"),
                "details": f"Owner Earnings Value: ${owner_earnings_value:,.2f}, Market Value: ${market_value:,.2f}, Gap: {owner_earnings_gap:.1%}, Score: {owner_earnings_result['score']}/5",
            }
        }

        confidence = round(abs(valuation_gap) * normalized_score / 10, 2) * 100
        valuation_analysis[ticker] = {
            "signal": signal,
            "confidence": min(confidence, 100),
            "reasoning": reasoning,
        }

        progress.update_status("valuation_agent", ticker, "Done")

    message = HumanMessage(content=json.dumps(valuation_analysis), name="valuation_agent")

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(valuation_analysis, "Valuation Analysis Agent")

    state["data"]["analyst_signals"]["valuation_agent"] = valuation_analysis
    return {"messages": [message], "data": data}


def calculate_owner_earnings_value(
    net_income: float,
    depreciation: float,
    capex: float,
    working_capital_change: float,
    growth_rate: float = 0.05,
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    num_years: int = 5,
    price_history: list = None
) -> dict:
    """
    Calculates the intrinsic value using Buffett's Owner Earnings method.

    Owner Earnings = Net Income + Depreciation/Amortization - Capital Expenditures - Working Capital Changes

    Args:
        net_income: Annual net income
        depreciation: Annual depreciation and amortization
        capex: Annual capital expenditures
        working_capital_change: Annual change in working capital
        growth_rate: Expected growth rate
        required_return: Required rate of return (Buffett typically uses 15%)
        margin_of_safety: Margin of safety to apply to final value
        num_years: Number of years to project
        price_history: List of historical prices for volatility analysis

    Returns:
        dict: {"value": float, "score": int, "max_score": int}
    """
    if not all([isinstance(x, (int, float)) for x in [net_income, depreciation, capex, working_capital_change]]):
        return {"value": 0, "score": 0, "max_score": 5}

    owner_earnings = net_income + depreciation - capex - working_capital_change
    if owner_earnings <= 0:
        return {"value": 0, "score": 0, "max_score": 5}

    future_values = []
    for year in range(1, num_years + 1):
        future_value = owner_earnings * (1 + growth_rate) ** year
        discounted_value = future_value / (1 + required_return) ** year
        future_values.append(discounted_value)

    terminal_growth = min(growth_rate, 0.03)
    terminal_value = (future_values[-1] * (1 + terminal_growth)) / (required_return - terminal_growth)
    terminal_value_discounted = terminal_value / (1 + required_return) ** num_years

    intrinsic_value = sum(future_values) + terminal_value_discounted
    value_with_safety_margin = intrinsic_value * (1 - margin_of_safety)

    volatility_score = 0
    if price_history and len(price_history) >= 252:
        prices = [p["close"] for p in price_history if "close" in p]
        if len(prices) > 1:
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)
            volatility_score = 2 if volatility < 0.20 else 1 if volatility < 0.30 else 0

    score = 3 + volatility_score  # 3 for valid calculation, plus volatility adjustment
    return {"value": value_with_safety_margin, "score": score, "max_score": 5}


def calculate_intrinsic_value(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
    num_years: int = 5,
    price_history: list = None
) -> dict:
    """
    Computes the discounted cash flow (DCF) for a given company based on the current free cash flow.
    """
    if not isinstance(free_cash_flow, (int, float)) or free_cash_flow <= 0:
        return {"value": 0, "score": 0, "max_score": 5}
    if not isinstance(growth_rate, (int, float)):
        growth_rate = 0.05  # Fallback if growth_rate is invalid

    cash_flows = [free_cash_flow * (1 + growth_rate) ** i for i in range(num_years)]
    present_values = [cf / (1 + discount_rate) ** (i + 1) for i, cf in enumerate(cash_flows)]

    terminal_value = cash_flows[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    terminal_present_value = terminal_value / (1 + discount_rate) ** num_years

    dcf_value = sum(present_values) + terminal_present_value

    volatility_score = 0
    if price_history and len(price_history) >= 252:
        prices = [p["close"] for p in price_history if "close" in p]
        if len(prices) > 1:
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)
            volatility_score = 2 if volatility < 0.20 else 1 if volatility < 0.30 else 0

    score = 3 + volatility_score
    return {"value": dcf_value, "score": score, "max_score": 5}


def calculate_working_capital_change(
    current_working_capital: float,
    previous_working_capital: float,
) -> float:
    """
    Calculate the absolute change in working capital between two periods.
    A positive change means more capital is tied up in working capital (cash outflow).
    A negative change means less capital is tied up (cash inflow).

    Args:
        current_working_capital: Current period's working capital
        previous_working_capital: Previous period's working capital

    Returns:
        float: Change in working capital (current - previous)
    """
    return current_working_capital - previous_working_capital