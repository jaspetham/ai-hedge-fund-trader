from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
from tools.api import get_prices, prices_to_df, get_latest_price
import json
import statistics

def risk_management_agent(state: AgentState):
    """Controls position sizing based on real-world risk factors."""
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    tickers = data["tickers"]

    risk_analysis = {}
    current_prices = state["data"].get("current_prices", {})  # From main.py

    for ticker in tickers:
        progress.update_status("risk_management_agent", ticker, "Analyzing price data")

        prices = get_prices(ticker, data["start_date"], data["end_date"])
        if not prices:
            progress.update_status("risk_management_agent", ticker, "Failed: No price data found")
            continue

        prices_df = prices_to_df(prices)
        current_price = current_prices.get(ticker, get_latest_price(ticker))
        volatility = statistics.stdev([p.close for p in prices[-30:]]) / current_price if len(prices) >= 30 else 0.1

        progress.update_status("risk_management_agent", ticker, "Calculating position limits")

        # Total portfolio value (cash + positions)
        position_values = sum(
            pos["long"] * current_prices.get(t, get_latest_price(t)) - pos["short"] * current_prices.get(t, get_latest_price(t))
            for t, pos in portfolio["positions"].items()
        )
        total_portfolio_value = portfolio["cash"] + position_values

        # Dynamic position limit: 20% base, adjusted by volatility
        position_limit = total_portfolio_value * (0.20 / (1 + volatility))
        pos = portfolio["positions"][ticker]
        current_position_value = pos["long"] * current_price - pos["short"] * current_price

        # Remaining limit and cash check
        remaining_position_limit = max(0, position_limit - current_position_value)
        max_position_size = min(remaining_position_limit, portfolio["cash"] if pos["long"] else float('inf'))  # No cash limit for shorts

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
            },
        }
        progress.update_status("risk_management_agent", ticker, "Done")

    message = HumanMessage(content=json.dumps(risk_analysis), name="risk_management_agent")
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(risk_analysis, "Risk Management Agent")

    state["data"]["analyst_signals"]["risk_management_agent"] = risk_analysis
    return {"messages": state["messages"] + [message], "data": state["data"]}