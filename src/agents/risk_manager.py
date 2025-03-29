from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
from tools.api import get_prices, prices_to_df, get_latest_price
import json
import numpy as np


def risk_management_agent(state: AgentState):
    """Controls position sizing based on real-world risk factors."""
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    tickers = data["tickers"]
    current_prices = state["data"].get("current_prices", {})

    risk_analysis = {}

    for ticker in tickers:
        progress.update_status("risk_management_agent", ticker, "Analyzing price data")
        prices = get_prices(ticker, data["start_date"], data["end_date"])
        if not prices:
            progress.update_status("risk_management_agent", ticker, "Failed: No price data found")
            continue

        prices_df = prices_to_df(prices)
        current_price = current_prices.get(ticker, get_latest_price(ticker))

        # Use numpy for volatility calculation
        if len(prices) >= 30:
            closes = [p.close for p in prices[-30:]]  # Fixed: Use attribute, not .get()
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.1
        else:
            volatility = 0.1

        progress.update_status("risk_management_agent", ticker, "Calculating position limits")
        position_values = sum(
            pos["long"] * current_prices.get(t, get_latest_price(t)) - pos["short"] * current_prices.get(t, get_latest_price(t))
            for t, pos in portfolio["positions"].items()
        )
        total_portfolio_value = portfolio["cash"] + position_values

        # Dynamic position limit with volatility adjustment
        base_limit = 0.20
        volatility_adjustment = max(0.5, 1 - volatility)  # Cap at 50% reduction for high volatility
        position_limit = total_portfolio_value * (base_limit * volatility_adjustment)

        pos = portfolio["positions"].get(ticker, {"long": 0, "short": 0})
        current_position_value = pos["long"] * current_price - pos["short"] * current_price

        remaining_position_limit = max(0, position_limit - current_position_value)
        max_position_size = min(remaining_position_limit, portfolio["cash"] if pos["long"] > 0 else float('inf'))

        risk_analysis[ticker] = {
            "remaining_position_limit": float(max_position_size),
            "current_price": float(current_price),
            "volatility": float(volatility),
            "reasoning": {
                "portfolio_value": float(total_portfolio_value),
                "current_position": float(current_position_value),
                "position_limit": float(position_limit),
                "remaining_limit": float(remaining_position_limit),
                "available_cash": float(portfolio["cash"]),
                "volatility_adjustment": float(volatility_adjustment),
            },
        }
        progress.update_status("risk_management_agent", ticker, "Done")

    message = HumanMessage(content=json.dumps(risk_analysis), name="risk_management_agent")
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(risk_analysis, "Risk Management Agent")

    state["data"]["analyst_signals"]["risk_management_agent"] = risk_analysis
    return {"messages": state["messages"] + [message], "data": state["data"]}