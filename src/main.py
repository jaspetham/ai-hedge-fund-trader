import sys
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import Fore, Back, Style, init
import questionary
from agents.ben_graham import ben_graham_agent
from agents.bill_ackman import bill_ackman_agent
from agents.fundamentals import fundamentals_agent
from agents.portfolio_manager import portfolio_management_agent
from agents.technicals import technical_analyst_agent
from agents.risk_manager import risk_management_agent
from agents.sentiment import sentiment_agent
from agents.warren_buffett import warren_buffett_agent
from agents.stanley_druckenmiller import stanley_druckenmiller_agent
from graph.state import AgentState
from agents.valuation import valuation_agent
from utils.display import print_trading_output
from utils.analysts import ANALYST_ORDER, get_analyst_nodes
from utils.progress import progress
from llm.models import LLM_ORDER, get_model_info
import argparse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tabulate import tabulate
from utils.visualize import save_graph_as_png
import json
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import os
from tools.api import get_prices

# Load environment variables
load_dotenv()
init(autoreset=True)

# Alpaca Trading Client
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
alpaca_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

def parse_hedge_fund_response(response):
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return None

def get_current_price(ticker: str):
    """Fetch the latest price for a ticker."""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - relativedelta(days=5)).strftime("%Y-%m-%d")
    price_data = get_prices(ticker, start_date=start_date, end_date=end_date)
    return price_data[-1].close if price_data else 100.0  # Fallback for simulation

def place_alpaca_order(ticker: str, qty: float, action: str, portfolio: dict):
    """Place an order with Alpaca and update the portfolio."""
    current_price = get_current_price(ticker)
    if current_price == 100.0:
        print(f"Warning: Using placeholder price $100 for {ticker} as no data was fetched.")

    order_cost = qty * current_price
    position = portfolio["positions"].get(ticker, {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0})
    cash = portfolio["cash"]

    try:
        if action == "buy":
            if cash >= order_cost:
                order_data = MarketOrderRequest(symbol=ticker, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
                order = alpaca_client.submit_order(order_data)
                portfolio["cash"] -= order_cost
                position["long"] += qty
                position["long_cost_basis"] = ((position["long_cost_basis"] * (position["long"] - qty)) + order_cost) / position["long"] if position["long"] else 0.0
                print(f"Buy order placed: {order.id} for {qty} shares of {ticker} at ${current_price:.2f}")
            else:
                print(f"Insufficient cash to buy {qty} shares of {ticker}. Required: ${order_cost:.2f}, Available: ${cash:.2f}")

        elif action == "sell":
            if position["long"] >= qty:
                order_data = MarketOrderRequest(symbol=ticker, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC)
                order = alpaca_client.submit_order(order_data)
                portfolio["cash"] += order_cost
                realized_gain = (current_price - position["long_cost_basis"]) * qty
                portfolio["realized_gains"][ticker]["long"] += realized_gain
                position["long"] -= qty
                print(f"Sell order placed: {order.id} for {qty} shares of {ticker} at ${current_price:.2f}, Gain: ${realized_gain:.2f}")
            else:
                print(f"Insufficient long position to sell {qty} shares of {ticker}. Available: {position['long']}")

        elif action == "short":
            order_data = MarketOrderRequest(symbol=ticker, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC)
            order = alpaca_client.submit_order(order_data)
            position["short"] += qty
            position["short_cost_basis"] = ((position["short_cost_basis"] * (position["short"] - qty)) + order_cost) / position["short"] if position["short"] else 0.0
            portfolio["cash"] += order_cost
            print(f"Short order placed: {order.id} for {qty} shares of {ticker} at ${current_price:.2f}")

        elif action == "cover":
            if position["short"] >= qty:
                order_data = MarketOrderRequest(symbol=ticker, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
                order = alpaca_client.submit_order(order_data)
                portfolio["cash"] -= order_cost
                realized_gain = (position["short_cost_basis"] - current_price) * qty
                portfolio["realized_gains"][ticker]["short"] += realized_gain
                position["short"] -= qty
                print(f"Cover order placed: {order.id} for {qty} shares of {ticker} at ${current_price:.2f}, Gain: ${realized_gain:.2f}")
            else:
                print(f"Insufficient short position to cover {qty} shares of {ticker}. Available: {position['short']}")

        else:
            print(f"Invalid action for {ticker}: {action}")

        portfolio["positions"][ticker] = position

    except Exception as e:
        print(f"Error placing order for {ticker}: {e}")

def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4o",
    model_provider: str = "OpenAI",
):
    progress.start()
    try:
        if selected_analysts:
            workflow = create_workflow(selected_analysts)
            agent = workflow.compile()
        else:
            agent = app

        final_state = agent.invoke(
            {
                "messages": [HumanMessage(content="Make trading decisions based on the provided data.")],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": model_provider,
                },
            }
        )
        return {
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
        }
    finally:
        progress.stop()

def start(state: AgentState):
    return state

def create_workflow(selected_analysts=None):
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)
    analyst_nodes = get_analyst_nodes()
    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_management_agent", portfolio_management_agent)
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")
    workflow.add_edge("risk_management_agent", "portfolio_management_agent")
    workflow.add_edge("portfolio_management_agent", END)
    workflow.set_entry_point("start_node")
    return workflow

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hedge fund trading system")
    parser.add_argument("--initial-cash", type=float, default=100000.0, help="Initial cash position")
    parser.add_argument("--margin-requirement", type=float, default=0.0, help="Initial margin requirement")
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated list of stock ticker symbols")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--runtime", type=float, default=24.0, help="Runtime in hours (default: 24)")
    parser.add_argument("--interval", type=int, default=300, help="Interval between runs in seconds (default: 300)")
    parser.add_argument("--show-reasoning", action="store_true", help="Show reasoning from each agent")
    parser.add_argument("--show-agent-graph", action="store_true", help="Show the agent graph")
    args = parser.parse_args()

    tickers = [ticker.strip() for ticker in args.tickers.split(",")]
    selected_analysts = None
    choices = questionary.checkbox(
        "Select your AI analysts.",
        choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
        instruction="\n\nInstructions: \n1. Press Space to select/unselect analysts.\n2. Press 'a' to select/unselect all.\n3. Press Enter when done to run the hedge fund.\n",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style([("checkbox-selected", "fg:green"), ("selected", "fg:green noinherit"), ("pointer", "noinherit"), ("highlighted", "noinherit")])
    ).ask()

    if not choices:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)
    else:
        selected_analysts = choices
        print(f"\nSelected analysts: {', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in choices)}\n")

    model_choice = questionary.select(
        "Select your LLM model:",
        choices=[questionary.Choice(display, value=value) for display, value, _ in LLM_ORDER],
        style=questionary.Style([("selected", "fg:green bold"), ("pointer", "fg:green bold"), ("highlighted", "fg:green"), ("answer", "fg:green bold")])
    ).ask()

    if not model_choice:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)
    else:
        model_info = get_model_info(model_choice)
        model_provider = model_info.provider.value if model_info else "Unknown"
        print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")

    workflow = create_workflow(selected_analysts)
    app = workflow.compile()

    if args.show_agent_graph:
        file_path = "" if selected_analysts is None else "".join(selected_analyst + "_" for selected_analyst in selected_analysts) + "graph.png"
        save_graph_as_png(app, file_path)

    if args.start_date:
        datetime.strptime(args.start_date, "%Y-%m-%d")
    if args.end_date:
        datetime.strptime(args.end_date, "%Y-%m-%d")

    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(months=3)).strftime("%Y-%m-%d") if not args.start_date else args.start_date

    portfolio = {
        "cash": args.initial_cash,
        "margin_requirement": args.margin_requirement,
        "positions": {ticker: {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0} for ticker in tickers},
        "realized_gains": {ticker: {"long": 0.0, "short": 0.0} for ticker in tickers}
    }

    # Continuous trading loop
    runtime_seconds = args.runtime * 3600  # Convert hours to seconds
    interval_seconds = args.interval
    start_time = time.time()

    print(f"Starting continuous trading for {args.runtime} hours, checking every {interval_seconds} seconds...")
    try:
        while time.time() - start_time < runtime_seconds:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running trading cycle...")

            # Update end_date to current time for fresh data
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(months=3)).strftime("%Y-%m-%d")

            result = run_hedge_fund(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                portfolio=portfolio,
                show_reasoning=args.show_reasoning,
                selected_analysts=selected_analysts,
                model_name=model_choice,
                model_provider=model_provider,
            )

            print_trading_output(result)

            # Execute trades and update portfolio
            if result and result["decisions"]:
                for ticker, decision in result["decisions"].items():
                    if isinstance(decision, dict) and "action" in decision and "quantity" in decision:
                        action = decision["action"]
                        qty = decision["quantity"]
                        if action in ["buy", "sell", "short", "cover"] and qty > 0:
                            place_alpaca_order(ticker, qty, action, portfolio)
                        elif action == "hold":
                            print(f"Holding position for {ticker}")
                        else:
                            print(f"Invalid action for {ticker}: {action}")
                    else:
                        print(f"Invalid decision format for {ticker}: {decision}")

            # Print portfolio summary with unrealized gains/losses
            print("\nPortfolio Status:")
            print(f"Cash: ${portfolio['cash']:.2f}")
            for ticker, pos in portfolio["positions"].items():
                current_price = get_current_price(ticker)
                long_value = pos["long"] * current_price
                short_value = pos["short"] * current_price
                unrealized_long = long_value - (pos["long"] * pos["long_cost_basis"]) if pos["long"] else 0.0
                unrealized_short = (pos["short_cost_basis"] * pos["short"]) - short_value if pos["short"] else 0.0
                print(f"{ticker}: Long {pos['long']} (Cost: ${pos['long_cost_basis']:.2f}, Unrealized: ${unrealized_long:.2f}), "
                      f"Short {pos['short']} (Cost: ${pos['short_cost_basis']:.2f}, Unrealized: ${unrealized_short:.2f})")
            for ticker, gains in portfolio["realized_gains"].items():
                print(f"{ticker} Realized Gains: Long ${gains['long']:.2f}, Short ${gains['short']:.2f}")

            # Wait for the next interval
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")

    print("\nTrading session ended.")
    print("Final Portfolio:")
    print(f"Cash: ${portfolio['cash']:.2f}")
    for ticker, pos in portfolio["positions"].items():
        current_price = get_current_price(ticker)
        long_value = pos["long"] * current_price
        short_value = pos["short"] * current_price
        unrealized_long = long_value - (pos["long"] * pos["long_cost_basis"]) if pos["long"] else 0.0
        unrealized_short = (pos["short_cost_basis"] * pos["short"]) - short_value if pos["short"] else 0.0
        print(f"{ticker}: Long {pos['long']} (Cost: ${pos['long_cost_basis']:.2f}, Unrealized: ${unrealized_long:.2f}), "
              f"Short {pos['short']} (Cost: ${pos['short_cost_basis']:.2f}, Unrealized: ${unrealized_short:.2f})")
    for ticker, gains in portfolio["realized_gains"].items():
        print(f"{ticker} Realized Gains: Long ${gains['long']:.2f}, Short ${gains['short']:.2f}")