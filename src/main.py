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
from agents.cathie_wood import cathie_wood_agent
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
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
import os
import requests
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

def cancel_single_order(order_id, api_key, secret_key, paper=True):
    base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
    url = f"{base_url}/v2/orders/{order_id}"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key
    }
    response = requests.delete(url, headers=headers)
    if response.status_code == 204:
        print(f"Order {order_id} canceled successfully")
    else:
        print(f"Failed to cancel order {order_id}: {response.text}")

def is_market_open(now):
    et_tz = pytz.timezone("America/New_York")
    now = now.astimezone(et_tz)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    is_open = market_open <= now <= market_close and now.weekday() < 5
    return is_open

def get_available_qty(ticker, trading_client, current_price):
    account = trading_client.get_account()
    buying_power = float(account.buying_power)
    positions = {p.symbol: p for p in trading_client.get_all_positions()}
    position = positions.get(ticker)
    existing_qty = int(position.qty) if position else 0
    held_for_orders = sum(int(order.qty) for order in trading_client.get_orders(GetOrdersRequest(status="open", symbol=ticker)))
    max_new_qty = int(buying_power // current_price)
    available_qty = max_new_qty - held_for_orders
    print(f"Available qty for {ticker}: {available_qty} (Buying Power: ${buying_power:.2f}, Held: {held_for_orders}, Existing: {existing_qty})")
    return available_qty

def submit_order(ticker, decision, current_price, trading_client):
    action = decision["action"].lower()
    requested_qty = decision["quantity"]
    tp = decision.get("take_profit")
    sl = decision.get("stop_loss")

    side = OrderSide.BUY if action == "buy" else OrderSide.SELL
    now = datetime.now(pytz.timezone("America/New_York"))
    extended_hours = not is_market_open(now)

    request = GetOrdersRequest(status="open", symbol=ticker)
    open_orders = trading_client.get_orders(request)
    for order in open_orders:
        print(f"Canceling existing order {order.id} for {ticker}")
        cancel_single_order(order.id, ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

    available_qty = get_available_qty(ticker, trading_client, current_price)
    qty = min(requested_qty, available_qty) if action == "buy" else requested_qty
    if qty <= 0:
        print(f"Insufficient buying power for {ticker}: Requested {requested_qty}, Adjusted to {qty}")
        return None

    order_data = MarketOrderRequest(
        symbol=ticker,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.GTC,
        extended_hours=extended_hours
    )

    if action in ["buy", "short"] and qty > 0 and is_market_open(now) and tp and sl:
        print(f"Submitting bracket order for {ticker} (TP: {tp}, SL: {sl})")
        order_data.order_class = "bracket"
        order_data.take_profit = TakeProfitRequest(limit_price=str(tp))
        order_data.stop_loss = StopLossRequest(stop_price=str(sl))
    else:
        print(f"Submitting plain market order for {ticker} (extended hours: {extended_hours})")

    try:
        response = trading_client.submit_order(order_data)
        return response
    except Exception as e:
        print(f"Failed to submit bracket order for {ticker}: {e}")
        order_data.order_class = None
        order_data.take_profit = None
        order_data.stop_loss = None
        try:
            response = trading_client.submit_order(order_data)
            print(f"Fallback market order submitted for {ticker}")
            return response
        except Exception as e2:
            print(f"Failed to submit fallback order for {ticker}: {e2}")
            return None

def place_order_with_alpaca(ticker, decision, current_price, portfolio, trading_client):
    action = decision["action"].lower()
    qty = decision["quantity"]
    tp = float(decision.get("take_profit", 0.0))
    sl = float(decision.get("stop_loss", 0.0))
    position = portfolio["positions"].get(ticker, {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0, "long_tp": 0.0, "long_sl": 0.0, "short_tp": 0.0, "short_sl": 0.0})

    print(f"Placing order for {ticker} - Action: {action}, Qty: {qty}, Current Price: ${current_price:.2f}")

    if action == "buy":
        if tp <= current_price or tp == 0: tp = current_price * 1.10
        if sl >= current_price or sl == 0: sl = current_price * 0.95
    elif action == "short":
        if tp >= current_price or tp == 0: tp = current_price * 0.90
        if sl <= current_price or sl == 0: sl = current_price * 1.05

    order_response = submit_order(ticker, {"action": action, "quantity": qty, "take_profit": tp, "stop_loss": sl}, current_price, trading_client)

    if order_response and order_response.status in [OrderStatus.PENDING_NEW, OrderStatus.ACCEPTED, OrderStatus.NEW]:
        executed_qty = int(order_response.qty)
        order_cost = current_price * executed_qty
        account = trading_client.get_account()
        buying_power = float(account.buying_power)
        if action == "buy" and buying_power >= order_cost:
            position["long"] += executed_qty
            position["long_cost_basis"] = ((position["long_cost_basis"] * (position["long"] - executed_qty)) + order_cost) / position["long"] if position["long"] else order_cost
            position["long_tp"] = tp
            position["long_sl"] = sl
            portfolio["cash"] -= order_cost  # Adjusts equity
            print(f"Buy order placed for {ticker}: {executed_qty} shares at ~${current_price:.2f} (pending)")
        elif action == "sell" and position["long"] >= executed_qty:
            realized_gain = (current_price - position["long_cost_basis"]) * executed_qty
            portfolio["cash"] += order_cost
            portfolio["realized_gains"][ticker]["long"] += realized_gain
            position["long"] -= executed_qty
            if position["long"] == 0:
                position["long_tp"] = position["long_sl"] = 0.0
            print(f"Sell order placed for {ticker}: {executed_qty} shares at ~${current_price:.2f}, Gain: ${realized_gain:.2f}")
        elif action == "short":
            position["short"] += executed_qty
            position["short_cost_basis"] = ((position["short_cost_basis"] * (position["short"] - executed_qty)) + order_cost) / position["short"] if position["short"] else order_cost
            position["short_tp"] = tp
            position["short_sl"] = sl
            portfolio["cash"] += order_cost
            print(f"Short order placed for {ticker}: {executed_qty} shares at ~${current_price:.2f} (pending)")
        elif action == "cover" and position["short"] >= executed_qty:
            realized_gain = (position["short_cost_basis"] - current_price) * executed_qty
            portfolio["cash"] -= order_cost
            portfolio["realized_gains"][ticker]["short"] += realized_gain
            position["short"] -= executed_qty
            if position["short"] == 0:
                position["short_tp"] = position["short_sl"] = 0.0
            print(f"Cover order placed for {ticker}: {executed_qty} shares at ~${current_price:.2f}, Gain: ${realized_gain:.2f}")
        portfolio["positions"][ticker] = position
    else:
        print(f"Order submission failed for {ticker}: {order_response}")

def run_hedge_fund(tickers, start_date, end_date, portfolio, show_reasoning, selected_analysts, model_name, model_provider, current_prices):
    print(f"Running hedge fund with analysts: {selected_analysts}")
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

def initialize_portfolio(tickers, trading_client):
    account = trading_client.get_account()
    cash = float(account.equity)
    positions = {p.symbol: p for p in trading_client.get_all_positions()}
    portfolio = {
        "cash": cash,
        "margin_requirement": 0.0,
        "positions": {},
        "realized_gains": {ticker: {"long": 0.0, "short": 0.0} for ticker in tickers}
    }
    for ticker in tickers:
        pos = positions.get(ticker)
        if pos:
            qty = int(pos.qty)
            cost_basis = float(pos.avg_entry_price)
            portfolio["positions"][ticker] = {
                "long": qty if qty > 0 else 0,
                "short": abs(qty) if qty < 0 else 0,
                "long_cost_basis": cost_basis if qty > 0 else 0.0,
                "short_cost_basis": cost_basis if qty < 0 else 0.0,
                "long_tp": 0.0,
                "long_sl": 0.0,
                "short_tp": 0.0,
                "short_sl": 0.0
            }
        else:
            portfolio["positions"][ticker] = {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0, "long_tp": 0.0, "long_sl": 0.0, "short_tp": 0.0, "short_sl": 0.0}
    print(f"Initialized portfolio from Alpaca: Cash ${cash:.2f}, Positions: {portfolio['positions']}")
    return portfolio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hedge fund trading system")
    parser.add_argument("--initial-cash", type=float, default=None)
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

    portfolio = initialize_portfolio(tickers, alpaca_client)

    runtime_seconds = args.runtime * 3600
    interval_seconds = args.interval
    start_time = time.time()

    print(f"Starting trading for {args.runtime} hours, checking every {interval_seconds} seconds...")
    try:
        while time.time() - start_time < runtime_seconds:
            et_timezone = pytz.timezone("America/New_York")
            now = datetime.now(et_timezone)
            print(f"Current time (ET): {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            # Uncomment to enforce market hours
            # if not is_market_open(now):
            #     print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Market closed. Waiting...")
            #     time.sleep(60)
            #     continue

            print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Running trading cycle...")
            end_date = now.strftime("%Y-%m-%d")
            start_date = (datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(days=1)).strftime("%Y-%m-%d")
            current_prices = {}
            for ticker in tickers:
                current_prices[ticker] = get_latest_price(ticker)
                print(f"Real-time price for {ticker}: ${current_prices[ticker]:.2f}")

            for ticker, pos in portfolio["positions"].items():
                price = current_prices[ticker]
                if pos["long"] > 0 and (price >= pos["long_tp"] or price <= pos["long_sl"]):
                    place_order_with_alpaca(ticker, {"action": "sell", "quantity": pos["long"]}, price, portfolio, alpaca_client)
                elif pos["short"] > 0 and (price <= pos["short_tp"] or price >= pos["short_sl"]):
                    place_order_with_alpaca(ticker, {"action": "cover", "quantity": pos["short"]}, price, portfolio, alpaca_client)

            result = run_hedge_fund(
                tickers, start_date, end_date, portfolio, args.show_reasoning, selected_analysts, model_choice, model_provider, current_prices
            )
            print_trading_output(result)

            if result and result["decisions"]:
                for ticker, decision in result["decisions"].items():
                    if isinstance(decision, dict) and "action" in decision and "quantity" in decision:
                        action = decision["action"]
                        qty = decision["quantity"]
                        current_price = current_prices[ticker]
                        if action in ["buy", "short", "sell", "cover"] and qty > 0:
                            place_order_with_alpaca(ticker, decision, current_price, portfolio, alpaca_client)
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
    print(f"Time (ET): {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
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