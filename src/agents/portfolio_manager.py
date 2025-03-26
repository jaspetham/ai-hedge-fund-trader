import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from graph.state import AgentState, show_agent_reasoning
from pydantic import BaseModel, Field
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm

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

    progress.update_status("portfolio_management_agent", None, "Analyzing signals")

    position_limits = {}
    current_prices = state["data"].get("current_prices", {})
    max_shares = {}
    signals_by_ticker = {}
    for ticker in tickers:
        progress.update_status("portfolio_management_agent", ticker, "Processing analyst signals")
        risk_data = analyst_signals.get("risk_management_agent", {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0)
        current_prices[ticker] = risk_data.get("current_price", current_prices.get(ticker, 0))

        if current_prices[ticker] > 0:
            max_shares[ticker] = int(position_limits[ticker] / current_prices[ticker])
        else:
            max_shares[ticker] = 0

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
                  * Buy only if cash available
                  * Sell only if holding long shares
                  * Sell quantity ≤ current long shares
                  * Buy quantity ≤ max_shares
                - For short positions:
                  * Short only if margin allows (50% of position value required)
                  * Cover only if holding short shares
                  * Cover quantity ≤ current short shares
                - Set dynamic take-profit (TP) and stop-loss (SL) based on signal confidence:
                  * Confidence > 80%: TP +15%, SL -3%
                  * Confidence 50-80%: TP +10%, SL -5%
                  * Confidence < 50%: TP +5%, SL -10%
                - For existing positions:
                  * Adjust TP/SL if signal changes significantly (±20% confidence)
                  * Scale in (buy/short more) if signal strengthens (+20% confidence)
                  * Hold if TP/SL not hit and signal stable (±10%)
                - Actions: "buy", "sell", "short", "cover", "hold"

                Inputs:
                - signals_by_ticker: ticker → signals
                - max_shares: max shares allowed
                - portfolio_cash: current cash
                - portfolio_positions: current positions (long/short, TP/SL)
                - current_prices: real-time prices
                - margin_requirement: current margin

                Compute TP and SL as numeric values (e.g., 123.45), not expressions (e.g., 100 * 1.15).
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

                Output in JSON:
                {{
                  "decisions": {{
                    "TICKER1": {{
                      "action": "buy/sell/short/cover/hold",
                      "quantity": integer,
                      "confidence": float (0-100),
                      "reasoning": "string",
                      "take_profit": float,
                      "stop_loss": float
                    }},
                    ...
                  }}
                }}
                """,
            ),
        ]
    )

    prompt = template.invoke(
        {
            "signals_by_ticker": json.dumps(signals_by_ticker, indent=2),
            "current_prices": json.dumps(current_prices, indent=2),
            "max_shares": json.dumps(max_shares, indent=2),
            "portfolio_cash": f"{portfolio.get('cash', 0):.2f}",
            "portfolio_positions": json.dumps(portfolio.get('positions', {}), indent=2),
            "margin_requirement": f"{portfolio.get('margin_requirement', 0):.2f}",
        }
    )

    def create_default_portfolio_output():
        return PortfolioManagerOutput(
            decisions={
                ticker: PortfolioDecision(action="hold", quantity=0, confidence=0.0, reasoning="Error, defaulting to hold", take_profit=0.0, stop_loss=0.0)
                for ticker in tickers
            }
        )

    # Post-process LLM output to ensure TP/SL are floats
    output = call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=PortfolioManagerOutput,
        agent_name="portfolio_management_agent",
        default_factory=create_default_portfolio_output
    )

    # Ensure TP/SL are computed correctly if LLM fails to follow rules
    for ticker, decision in output.decisions.items():
        price = current_prices.get(ticker, 0.0)
        if price > 0 and decision.action in ["buy", "short"]:
            confidence = decision.confidence
            if confidence > 80:
                tp_factor, sl_factor = (1.15, 0.97) if decision.action == "buy" else (0.85, 1.03)
            elif 50 <= confidence <= 80:
                tp_factor, sl_factor = (1.10, 0.95) if decision.action == "buy" else (0.90, 1.05)
            else:
                tp_factor, sl_factor = (1.05, 0.90) if decision.action == "buy" else (0.95, 1.10)
            decision.take_profit = round(price * tp_factor, 2)
            decision.stop_loss = round(price * sl_factor, 2)

    return output