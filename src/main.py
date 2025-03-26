import sys
import time
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import Fore, Style, init
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
from dateutil.relativedelta import relativedelta
from tabulate import tabulate
from utils.visualize import save_graph_as_png
import json
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import os
from tools.api import get_prices, get_latest_price

load_dotenv()
init(autoreset=True)

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
alpaca_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

def parse_hedge_fund_response(response):
    try:
        return json.loads(response)
    except Exception as e:
        print(f"Error parsing response: {e}\nResponse: {repr(response)}")
        return None

def place_alpaca_order(ticker: str, qty: float, action: str, tp: float, sl: float, portfolio: dict):
    current_price = get_latest_price(ticker)
    order_cost = qty * current_price
    position = portfolio["positions"].get(ticker, {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0, "long_tp": 0.0, "long_sl": 0.0, "short_tp": 0.0, "short_sl": 0.0})
    cash = portfolio["cash"]

    try:
        if action == "buy":
            if cash >= order_cost:
                order_data = MarketOrderRequest(
                    symbol=ticker, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC,
                    take_profit={"limit_price": tp}, stop_loss={"stop_price": sl}
                )
                order = alpaca_client.submit_order(order_data)
                portfolio["cash"] -= order_cost
                position["long"] += qty
                position["long_cost_basis"] = ((position["long_cost_basis"] * (position["long"] - qty)) + order_cost) / position["long"] if position["long"] else 0.0
                position["long_tp"] = tp
                position["long_sl"] = sl
                print(f"Buy order placed: {order.id} for {qty} shares of {ticker} at ${current_price:.2f}, TP ${tp:.2f}, SL ${sl:.2f}")
        elif action == "sell":
            if position["long"] >= qty:
                order_data = MarketOrderRequest(symbol=ticker, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC)
                order = alpaca_client.submit_order(order_data)
                portfolio["cash"] += order_cost
                realized_gain = (current_price - position["long_cost_basis"]) * qty
                portfolio["realized_gains"][ticker]["long"] += realized_gain
                position["long"] -= qty
                if position["long"] == 0:
                    position["long_tp"] = position["long_sl"] = 0.0
                print(f"Sell order placed: {order.id} for {qty} shares of {ticker} at ${current_price:.2f}, Gain: ${realized_gain:.2f}")
        elif action == "short":
            order_data = MarketOrderRequest(
                symbol=ticker, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
                take_profit={"limit_price": tp}, stop_loss={"stop_price": sl}
            )
            order = alpaca_client.submit_order(order_data)
            position["short"] += qty
            position["short_cost_basis"] = ((position["short_cost_basis"] * (position["short"] - qty)) + order_cost) / position["short"] if position["short"] else 0.0
            position["short_tp"] = tp
            position["short_sl"] = sl
            portfolio["cash"] += order_cost
            print(f"Short order placed: {order.id} for {qty} shares of {ticker} at ${current_price:.2f}, TP ${tp:.2f}, SL ${sl:.2f}")
        elif action == "cover":
            if position["short"] >= qty:
                order_data = MarketOrderRequest(symbol=ticker, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
                order = alpaca_client.submit_order(order_data)
                portfolio["cash"] -= order_cost
                realized_gain = (position["short_cost_basis"] - current_price) * qty
                portfolio["realized_gains"][ticker]["short"] += realized_gain
                position["short"] -= qty
                if position["short"] == 0:
                    position["short_tp"] = position["short_sl"] = 0.0
                print(f"Cover order placed: {order.id} for {qty} shares of {ticker} at ${current_price:.2f}, Gain: ${realized_gain:.2f}")
        else:
            return

        portfolio["positions"][ticker] = position
    except Exception as e:
        print(f"Error placing order for {ticker}: {e}")

def run_hedge_fund(tickers, start_date, end_date, portfolio, show_reasoning, selected_analysts, model_name, model_provider, current_prices):
    progress.start()
    try:
        workflow = create_workflow(selected_analysts)
        agent = workflow.compile()
        final_state = agent.invoke(
            {
                "messages": [HumanMessage(content="Make trading decisions based on the provided data.")],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                    "current_prices": current_prices,
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

def create_workflow(selected_analysts):
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)
    analyst_nodes = get_analyst_nodes()
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
    parser.add_argument("--initial-cash", type=float, default=100000.0)
    parser.add_argument("--margin-requirement", type=float, default=0.0)
    parser.add_argument("--tickers", type=str, required=True)
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date", type=str)
    parser.add_argument("--runtime", type=float, default=24.0)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--show-reasoning", action="store_true")
    parser.add_argument("--show-agent-graph", action="store_true")
    args = parser.parse_args()

    tickers = [ticker.strip() for ticker in args.tickers.split(",")]
    selected_analysts = questionary.checkbox(
        "Select your AI analysts.",
        choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
    ).ask() or sys.exit("Interrupted")

    print(f"\nSelected analysts: {', '.join(Fore.GREEN + c.title().replace('_', ' ') + Style.RESET_ALL for c in selected_analysts)}")

    model_choice = questionary.select(
        "Select your LLM model:",
        choices=[questionary.Choice(display, value=value) for display, value, _ in LLM_ORDER],
    ).ask() or sys.exit("Interrupted")

    model_info = get_model_info(model_choice)
    model_provider = model_info.provider.value if model_info else "Unknown"
    print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN}{model_choice}{Style.RESET_ALL}")

    workflow = create_workflow(selected_analysts)
    if args.show_agent_graph:
        save_graph_as_png(workflow.compile(), "graph.png")

    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    start_date = args.start_date or (datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(months=3)).strftime("%Y-%m-%d")

    portfolio = {
        "cash": args.initial_cash,
        "margin_requirement": args.margin_requirement,
        "positions": {ticker: {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0, "long_tp": 0.0, "long_sl": 0.0, "short_tp": 0.0, "short_sl": 0.0} for ticker in tickers},
        "realized_gains": {ticker: {"long": 0.0, "short": 0.0} for ticker in tickers}
    }

    runtime_seconds = args.runtime * 3600
    interval_seconds = args.interval
    start_time = time.time()

    print(f"Starting trading for {args.runtime} hours, checking every {interval_seconds} seconds...")
    try:
        while time.time() - start_time < runtime_seconds:
            # Use Eastern Time (ET) for market hours check
            et_timezone = pytz.timezone("America/New_York")
            now = datetime.now(et_timezone)
            if now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour >= 16:
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Market closed. Waiting...")
                time.sleep(60)
                continue

            print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Running trading cycle...")
            end_date = now.strftime("%Y-%m-%d")
            start_date = (datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(days=1)).strftime("%Y-%m-%d")  # Changed to 1 day for runtime=1
            current_prices = {}
            for ticker in tickers:
                prices = get_prices(ticker, start_date=start_date, end_date=end_date)  # Use full range
                if prices:
                    current_prices[ticker] = prices[-1].close  # Latest close
                else:
                    current_prices[ticker] = get_latest_price(ticker)  # Fallback
                    print(f"Using latest price for {ticker} as no historical data found")

            # Check TP/SL manually
            for ticker, pos in portfolio["positions"].items():
                price = current_prices[ticker]
                if pos["long"] > 0 and (price >= pos["long_tp"] or price <= pos["long_sl"]):
                    place_alpaca_order(ticker, pos["long"], "sell", 0.0, 0.0, portfolio)
                elif pos["short"] > 0 and (price <= pos["short_tp"] or price >= pos["short_sl"]):
                    place_alpaca_order(ticker, pos["short"], "cover", 0.0, 0.0, portfolio)

            result = run_hedge_fund(
                tickers, start_date, end_date, portfolio, args.show_reasoning, selected_analysts, model_choice, model_provider, current_prices
            )
            print_trading_output(result)

            if result and result["decisions"]:
                for ticker, decision in result["decisions"].items():
                    if isinstance(decision, dict) and "action" in decision and "quantity" in decision:
                        action = decision["action"]
                        qty = decision["quantity"]
                        tp = decision.get("take_profit", 0.0)
                        sl = decision.get("stop_loss", 0.0)
                        if action in ["buy", "short"] and qty > 0:
                            place_alpaca_order(ticker, qty, action, tp, sl, portfolio)
                        elif action in ["sell", "cover"] and qty > 0:
                            place_alpaca_order(ticker, qty, action, 0.0, 0.0, portfolio)
                        elif action == "hold":
                            print(f"Holding position for {ticker}")

            print("\nPortfolio Status:")
            print(f"Cash: ${portfolio['cash']:.2f}")
            for ticker, pos in portfolio["positions"].items():
                price = current_prices[ticker]
                long_value = pos["long"] * price
                short_value = pos["short"] * price
                unrealized_long = long_value - (pos["long"] * pos["long_cost_basis"]) if pos["long"] else 0.0
                unrealized_short = (pos["short_cost_basis"] * pos["short"]) - short_value if pos["short"] else 0.0
                print(f"{ticker}: Long {pos['long']} (Cost: ${pos['long_cost_basis']:.2f}, TP: ${pos['long_tp']:.2f}, SL: ${pos['long_sl']:.2f}, Unrealized: ${unrealized_long:.2f}), "
                      f"Short {pos['short']} (Cost: ${pos['short_cost_basis']:.2f}, TP: ${pos['short_tp']:.2f}, SL: ${pos['short_sl']:.2f}, Unrealized: ${unrealized_short:.2f})")
            for ticker, gains in portfolio["realized_gains"].items():
                print(f"{ticker} Realized Gains: Long ${gains['long']:.2f}, Short ${gains['short']:.2f}")

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    print("\nFinal Portfolio:")
    current_prices = {ticker: get_latest_price(ticker) for ticker in tickers}
    now = datetime.now(pytz.timezone("America/New_York"))
    print(f"Time (ET): {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Cash: ${portfolio['cash']:.2f}")
    for ticker, pos in portfolio["positions"].items():
        price = current_prices[ticker]
        long_value = pos["long"] * price
        short_value = pos["short"] * price
        unrealized_long = long_value - (pos["long"] * pos["long_cost_basis"]) if pos["long"] else 0.0
        unrealized_short = (pos["short_cost_basis"] * pos["short"]) - short_value if pos["short"] else 0.0
        print(f"{ticker}: Long {pos['long']} (Cost: ${pos['long_cost_basis']:.2f}, TP: ${pos['long_tp']:.2f}, SL: ${pos['long_sl']:.2f}, Unrealized: ${unrealized_long:.2f}), "
              f"Short {pos['short']} (Cost: ${pos['short_cost_basis']:.2f}, TP: ${pos['short_tp']:.2f}, SL: ${pos['short_sl']:.2f}, Unrealized: ${unrealized_short:.2f})")
    for ticker, gains in portfolio["realized_gains"].items():
        print(f"{ticker} Realized Gains: Long ${gains['long']:.2f}, Short ${gains['short']:.2f}")