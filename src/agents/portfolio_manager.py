import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from graph.state import AgentState, show_agent_reasoning
from pydantic import BaseModel, Field
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm
from tools.api import get_prices
from datetime import datetime, timedelta
import numpy as np


class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(description="Number of shares to trade")
    confidence: float = Field(description="Confidence in the decision, between 0.0 and 100.0")
    reasoning: str = Field(description="Reasoning for the decision")
    take_profit: float = Field(description="Take-profit price", default=0.0)
    stop_loss: float = Field(description="Stop-loss price", default=0.0)


class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, PortfolioDecision] = Field(description="Dictionary of ticker to trading decisions")


def portfolio_management_agent(state: AgentState):
    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]
    end_date = state["data"]["end_date"]

    progress.update_status("portfolio_management_agent", None, "Analyzing signals")

    position_limits = {}
    current_prices = state["data"].get("current_prices", {})
    max_shares = {}
    signals_by_ticker = {}
    volatilities = {}

    for ticker in tickers:
        progress.update_status("portfolio_management_agent", ticker, "Processing analyst signals")
        risk_data = analyst_signals.get("risk_management_agent", {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0)
        current_prices[ticker] = risk_data.get("current_price", current_prices.get(ticker, 0))

        if current_prices[ticker] > 0:
            max_shares[ticker] = int(position_limits[ticker] / current_prices[ticker])
        else:
            max_shares[ticker] = 0

        progress.update_status("portfolio_management_agent", ticker, "Fetching price history")
        price_history = get_prices(ticker, start_date=(datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d"), end_date=end_date)
        volatility = 0.1  # Default
        if price_history and len(price_history) >= 90:
            prices = [p.get("close", 0) for p in price_history]
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)
        volatilities[ticker] = volatility

        ticker_signals = {}
        for agent, signals in analyst_signals.items():
            if agent != "risk_management_agent" and ticker in signals:
                ticker_signals[agent] = {"signal": signals[ticker]["signal"], "confidence": signals[ticker]["confidence"]}
        signals_by_ticker[ticker] = ticker_signals

    progress.update_status("portfolio_management_agent", None, "Making trading decisions")
    result = generate_trading_decision(
        tickers=tickers,
        signals_by_ticker=signals_by_ticker,
        current_prices=current_prices,
        max_shares=max_shares,
        portfolio=portfolio,
        volatilities=volatilities,
        model_name=state["metadata"]["model_name"],
        model_provider=state["metadata"]["model_provider"],
    )

    message = HumanMessage(
        content=json.dumps({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}),
        name="portfolio_management",
    )
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}, "Portfolio Management Agent")

    progress.update_status("portfolio_management_agent", None, "Done")
    return {"messages": state["messages"] + [message], "data": state["data"]}


def generate_trading_decision(
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    current_prices: dict[str, float],
    max_shares: dict[str, int],
    portfolio: dict[str, float],
    volatilities: dict[str, float],
    model_name: str,
    model_provider: str,
) -> PortfolioManagerOutput:
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a portfolio manager making final trading decisions based on multiple tickers.

                Trading Rules:
                - For long positions:
                  * Buy only if cash available (max_buy_shares = cash / price)
                  * Sell only if holding long shares
                  * Sell quantity ≤ current long shares
                  * Buy quantity ≤ max_shares * (0.5 if volatility > 0.30 else 1.0)
                - For short positions:
                  * Short only if margin allows (50% of position value required, max_short_shares = (cash - margin) / (price * 0.5))
                  * Cover only if holding short shares
                  * Cover quantity ≤ current short shares
                - Volatility Adjustment:
                  * High volatility (>30%): Reduce position size by 50%
                  * Low volatility (<20%): Full position allowed
                - Set dynamic take-profit (TP) and stop-loss (SL) based on signal confidence:
                  * Confidence > 80%: TP = price * 1.15, SL = price * 0.97 (buy); TP = price * 0.85, SL = price * 1.03 (short)
                  * Confidence 50-80%: TP = price * 1.10, SL = price * 0.95 (buy); TP = price * 0.90, SL = price * 1.05 (short)
                  * Confidence < 50%: TP = price * 1.05, SL = price * 0.90 (buy); TP = price * 0.95, SL = price * 1.10 (short)
                - For existing positions:
                  * Adjust TP/SL if signal changes significantly (±20% confidence)
                  * Scale in (buy/short more) if signal strengthens (+20% confidence)
                  * Hold if TP/SL not hit and signal stable (±10%)
                - Actions: "buy", "sell", "short", "cover", "hold"

                Inputs:
                - signals_by_ticker: ticker → signals
                - max_shares: max shares allowed (before volatility adjustment)
                - portfolio_cash: current cash
                - portfolio_positions: current positions (long/short, TP/SL)
                - current_prices: real-time prices
                - margin_requirement: current margin
                - volatilities: annualized volatility per ticker

                Compute TP and SL as numeric values (e.g., 123.45), NOT expressions (e.g., 100 * 1.15). Round to 2 decimal places.

                Example Output:
                {{
                  "decisions": {{
                    "NVDA": {{
                      "action": "hold",
                      "quantity": 0,
                      "confidence": 70.0,
                      "reasoning": "Bearish signal, already short",
                      "take_profit": 92.92,
                      "stop_loss": 103.85
                    }},
                    "GOOG": {{
                      "action": "sell",
                      "quantity": 116,
                      "confidence": 80.0,
                      "reasoning": "Bearish signal, selling long position",
                      "take_profit": 0.00,
                      "stop_loss": 0.00
                    }}
                  }}
                }}
                """,
            ),
            (
                "human",
                """Make trading decisions for each ticker based on the analysis.

                Signals by Ticker:
                {signals_by_ticker}

                Current Prices:
                {current_prices}

                Maximum Shares Allowed:
                {max_shares}

                Portfolio Cash: {portfolio_cash}
                Current Positions: {portfolio_positions}
                Current Margin Requirement: {margin_requirement}
                Volatilities: {volatilities}

                Output in JSON format as shown in the example.
                """,
            ),
        ]
    )

    # Precompute constraints to assist LLM
    cash = portfolio.get("cash", 0)
    margin = portfolio.get("margin_requirement", 0)
    positions = portfolio.get("positions", {})
    constraints = {}
    for ticker in tickers:
        price = current_prices.get(ticker, 0.0)
        volatility = volatilities.get(ticker, 0.1)
        pos = positions.get(ticker, {"long": 0, "short": 0, "long_tp": 0.0, "long_sl": 0.0, "short_tp": 0.0, "short_sl": 0.0})
        max_buy = int(cash / price) if price > 0 else 0
        max_short = int((cash - margin) / (price * 0.5)) if price > 0 else 0
        constraints[ticker] = {
            "max_buy_shares": max_buy,
            "max_short_shares": max_short,
            "current_long": pos["long"],
            "current_short": pos["short"],
            "existing_tp": pos["long_tp"] if pos["long"] > 0 else pos["short_tp"],
            "existing_sl": pos["long_sl"] if pos["long"] > 0 else pos["short_sl"]
        }

    prompt = template.invoke(
        {
            "signals_by_ticker": json.dumps(signals_by_ticker, indent=2),
            "current_prices": json.dumps(current_prices, indent=2),
            "max_shares": json.dumps(max_shares, indent=2),
            "portfolio_cash": f"{cash:.2f}",
            "portfolio_positions": json.dumps(positions, indent=2),
            "margin_requirement": f"{margin:.2f}",
            "volatilities": json.dumps(volatilities, indent=2),
        }
    )

    def create_default_portfolio_output():
        return PortfolioManagerOutput(
            decisions={
                ticker: PortfolioDecision(action="hold", quantity=0, confidence=0.0, reasoning="Error, defaulting to hold", take_profit=0.0, stop_loss=0.0)
                for ticker in tickers
            }
        )

    output = call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=PortfolioManagerOutput,
        agent_name="portfolio_management_agent",
        default_factory=create_default_portfolio_output
    )

    # Post-process to enforce rules and fix TP/SL
    if output and output.decisions:
        for ticker, decision in output.decisions.items():
            price = current_prices.get(ticker, 0.0)
            volatility = volatilities.get(ticker, 0.1)
            pos = positions.get(ticker, {"long": 0, "short": 0})
            max_allowed = max_shares.get(ticker, 0) * (0.5 if volatility > 0.30 else 1.0)
            constraint = constraints.get(ticker, {})

            # Enforce quantity constraints
            if decision.action == "buy":
                decision.quantity = min(decision.quantity, int(max_allowed), constraint["max_buy_shares"])
            elif decision.action == "sell":
                decision.quantity = min(decision.quantity, pos["long"])
            elif decision.action == "short":
                decision.quantity = min(decision.quantity, int(max_allowed), constraint["max_short_shares"])
            elif decision.action == "cover":
                decision.quantity = min(decision.quantity, pos["short"])

            # Fix TP/SL if invalid or missing
            if price > 0 and (decision.take_profit == 0.0 or decision.stop_loss == 0.0):
                confidence = decision.confidence
                if confidence > 80:
                    tp_factor, sl_factor = (1.15, 0.97) if decision.action == "buy" else (0.85, 1.03)
                elif 50 <= confidence <= 80:
                    tp_factor, sl_factor = (1.10, 0.95) if decision.action == "buy" else (0.90, 1.05)
                else:
                    tp_factor, sl_factor = (1.05, 0.90) if decision.action == "buy" else (0.95, 1.10)

                if decision.action in ["buy", "short"] or (decision.action == "hold" and (pos["long"] > 0 or pos["short"] > 0)):
                    decision.take_profit = round(price * tp_factor, 2)
                    decision.stop_loss = round(price * sl_factor, 2)
                decision.reasoning += f" (Volatility: {volatility:.2%}, TP/SL adjusted post-LLM)"

    return output