from graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from tools.api import get_financial_metrics, get_market_cap, search_line_items, get_prices
from utils.llm import call_llm
from utils.progress import progress
import numpy as np


class WarrenBuffettSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def warren_buffett_agent(state: AgentState):
    """Analyzes stocks using Buffett's principles and LLM reasoning."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    portfolio = data["portfolio"]
    current_prices = data["current_prices"]

    analysis_data = {}
    buffett_analysis = {}

    for ticker in tickers:
        progress.update_status("warren_buffett_agent", ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=5)

        progress.update_status("warren_buffett_agent", ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "capital_expenditure",
                "depreciation_and_amortization",
                "net_income",
                "outstanding_shares",
                "total_assets",
                "total_liabilities",
                "dividends_and_other_cash_distributions",
                "issuance_or_purchase_of_equity_shares",
            ],
            end_date,
        )

        progress.update_status("warren_buffett_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date)

        progress.update_status("warren_buffett_agent", ticker, "Fetching price history")
        price_history = get_prices(ticker, start_date=(datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d"), end_date=end_date)

        progress.update_status("warren_buffett_agent", ticker, "Analyzing fundamentals")
        fundamental_analysis = analyze_fundamentals(metrics)

        progress.update_status("warren_buffett_agent", ticker, "Analyzing consistency")
        consistency_analysis = analyze_consistency(financial_line_items)

        progress.update_status("warren_buffett_agent", ticker, "Analyzing moat")
        moat_analysis = analyze_moat(metrics)

        progress.update_status("warren_buffett_agent", ticker, "Analyzing management quality")
        mgmt_analysis = analyze_management_quality(financial_line_items)

        progress.update_status("warren_buffett_agent", ticker, "Calculating intrinsic value")
        intrinsic_value_analysis = calculate_intrinsic_value(financial_line_items, price_history)

        # Normalize total score to 0-10 scale
        total_score = (
            fundamental_analysis["score"] +
            consistency_analysis["score"] +
            moat_analysis["score"] +
            mgmt_analysis["score"] +
            intrinsic_value_analysis["score"]
        )
        max_possible_score = 10  # Normalized to 10

        # Adjust score based on individual max scores
        raw_max = 7 + 3 + 3 + 2 + 3  # Original max scores: fundamentals (7), consistency (3), moat (3), mgmt (2), intrinsic (3)
        normalized_score = (total_score / raw_max) * 10 if raw_max > 0 else 0

        margin_of_safety = None
        intrinsic_value = intrinsic_value_analysis["intrinsic_value"]
        current_price = current_prices.get(ticker)
        if intrinsic_value and current_price and financial_line_items and financial_line_items[0].outstanding_shares:
            market_value = current_price * financial_line_items[0].outstanding_shares
            margin_of_safety = (intrinsic_value - market_value) / market_value

        # Portfolio context
        position = portfolio["positions"].get(ticker, {"long": 0, "short": 0})
        position_context = {
            "long_shares": position["long"],
            "short_shares": position["short"],
            "short_cost_basis": position["short_cost_basis"],
            "current_price": current_price
        }

        if normalized_score >= 7 and margin_of_safety and margin_of_safety >= 0.3:
            signal = "bullish"
        elif normalized_score <= 3 or (margin_of_safety is not None and margin_of_safety < -0.3):
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": normalized_score,
            "max_score": max_possible_score,
            "fundamental_analysis": fundamental_analysis,
            "consistency_analysis": consistency_analysis,
            "moat_analysis": moat_analysis,
            "management_analysis": mgmt_analysis,
            "intrinsic_value_analysis": intrinsic_value_analysis,
            "market_cap": market_cap,
            "margin_of_safety": margin_of_safety,
            "position_context": position_context
        }

        progress.update_status("warren_buffett_agent", ticker, "Generating Warren Buffett analysis")
        buffett_output = generate_buffett_output(
            ticker=ticker,
            analysis_data=analysis_data,
            portfolio_cash=portfolio["cash"],
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        buffett_analysis[ticker] = {
            "signal": buffett_output.signal,
            "confidence": buffett_output.confidence,
            "reasoning": buffett_output.reasoning,
        }

        progress.update_status("warren_buffett_agent", ticker, "Done")

    message = HumanMessage(content=json.dumps(buffett_analysis), name="warren_buffett_agent")

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(buffett_analysis, "Warren Buffett Agent")

    state["data"]["analyst_signals"]["warren_buffett_agent"] = buffett_analysis

    return {"messages": [message], "data": state["data"]}


def analyze_fundamentals(metrics: list) -> dict[str, any]:
    """Analyze company fundamentals based on Buffett's criteria."""
    if not metrics:
        return {"score": 0, "max_score": 7, "details": "Insufficient fundamental data"}

    latest_metrics = metrics[0]
    score = 0
    reasoning = []

    roe = getattr(latest_metrics, "return_on_equity", None)
    if roe and roe > 0.15:
        score += 2
        reasoning.append(f"Strong ROE of {roe:.1%}")
    elif roe is not None:
        reasoning.append(f"Weak ROE of {roe:.1%}")
    else:
        reasoning.append("ROE data not available")

    debt_to_equity = getattr(latest_metrics, "debt_to_equity", None)
    if debt_to_equity and debt_to_equity < 0.5:
        score += 2
        reasoning.append("Conservative debt levels")
    elif debt_to_equity is not None:
        reasoning.append(f"High debt to equity ratio of {debt_to_equity:.1f}")
    else:
        reasoning.append("Debt to equity data not available")

    operating_margin = getattr(latest_metrics, "operating_margin", None)
    if operating_margin and operating_margin > 0.15:
        score += 2
        reasoning.append("Strong operating margins")
    elif operating_margin is not None:
        reasoning.append(f"Weak operating margin of {operating_margin:.1%}")
    else:
        reasoning.append("Operating margin data not available")

    current_ratio = getattr(latest_metrics, "current_ratio", None)
    if current_ratio and current_ratio > 1.5:
        score += 1
        reasoning.append("Good liquidity position")
    elif current_ratio is not None:
        reasoning.append(f"Weak liquidity with current ratio of {current_ratio:.1f}")
    else:
        reasoning.append("Current ratio data not available")

    return {"score": score, "max_score": 7, "details": "; ".join(reasoning), "metrics": latest_metrics.model_dump()}


def analyze_consistency(financial_line_items: list) -> dict[str, any]:
    """Analyze earnings consistency and growth."""
    if len(financial_line_items) < 4:
        return {"score": 0, "max_score": 3, "details": "Insufficient historical data"}

    score = 0
    reasoning = []

    earnings_values = [getattr(item, "net_income", None) for item in financial_line_items if getattr(item, "net_income", None) is not None]
    if len(earnings_values) >= 4:
        earnings_growth = all(earnings_values[i] > earnings_values[i + 1] for i in range(len(earnings_values) - 1))
        if earnings_growth:
            score += 3
            reasoning.append("Consistent earnings growth over past periods")
        else:
            reasoning.append("Inconsistent earnings growth pattern")

        if len(earnings_values) >= 2 and earnings_values[-1] != 0:
            growth_rate = (earnings_values[0] - earnings_values[-1]) / abs(earnings_values[-1])
            reasoning.append(f"Total earnings growth of {growth_rate:.1%} over past {len(earnings_values)} periods")
    else:
        reasoning.append("Insufficient earnings data for trend analysis")

    return {"score": score, "max_score": 3, "details": "; ".join(reasoning)}


def analyze_moat(metrics: list) -> dict[str, any]:
    """
    Evaluate whether the company likely has a durable competitive advantage (moat).
    For simplicity, we look at stability of ROE/operating margins over multiple periods
    or high margin over the last few years. Higher stability => higher moat score.
    """
    if not metrics or len(metrics) < 3:
        return {"score": 0, "max_score": 3, "details": "Insufficient data for moat analysis"}

    reasoning = []
    moat_score = 0
    historical_roes = []
    historical_margins = []

    for m in metrics:
        roe = getattr(m, "return_on_equity", None)
        margin = getattr(m, "operating_margin", None)
        if roe is not None:
            historical_roes.append(roe)
        if margin is not None:
            historical_margins.append(margin)

    if len(historical_roes) >= 3:
        stable_roe = all(r > 0.15 for r in historical_roes)
        if stable_roe:
            moat_score += 1
            reasoning.append("Stable ROE above 15% across periods (suggests moat)")
        else:
            reasoning.append("ROE not consistently above 15%")

    if len(historical_margins) >= 3:
        stable_margin = all(m > 0.15 for m in historical_margins)
        if stable_margin:
            moat_score += 1
            reasoning.append("Stable operating margins above 15% (moat indicator)")
        else:
            reasoning.append("Operating margin not consistently above 15%")

    if moat_score == 2:
        moat_score += 1
        reasoning.append("Both ROE and margin stability indicate a solid moat")

    return {"score": moat_score, "max_score": 3, "details": "; ".join(reasoning)}


def analyze_management_quality(financial_line_items: list) -> dict[str, any]:
    """
    Checks for share dilution or consistent buybacks, and some dividend track record.
    A simplified approach:
      - if there's net share repurchase or stable share count, it suggests management
        might be shareholder-friendly.
      - if there's a big new issuance, it might be a negative sign (dilution).
    """
    if not financial_line_items:
        return {"score": 0, "max_score": 2, "details": "Insufficient data for management analysis"}

    reasoning = []
    mgmt_score = 0
    latest = financial_line_items[0]

    issuance = getattr(latest, "issuance_or_purchase_of_equity_shares", None)
    if issuance and issuance < 0:
        mgmt_score += 1
        reasoning.append("Company has been repurchasing shares (shareholder-friendly)")
    elif issuance and issuance > 0:
        reasoning.append("Recent common stock issuance (potential dilution)")
    else:
        reasoning.append("No significant new stock issuance detected")

    dividends = getattr(latest, "dividends_and_other_cash_distributions", None)
    if dividends and dividends < 0:
        mgmt_score += 1
        reasoning.append("Company has a track record of paying dividends")
    else:
        reasoning.append("No or minimal dividends paid")

    return {"score": mgmt_score, "max_score": 2, "details": "; ".join(reasoning)}


def calculate_owner_earnings(financial_line_items: list) -> dict[str, any]:
    """Calculate owner earnings (Buffett's preferred measure of true earnings power).
    Owner Earnings = Net Income + Depreciation - Maintenance CapEx"""
    if not financial_line_items or len(financial_line_items) < 1:
        return {"owner_earnings": None, "details": ["Insufficient data for owner earnings calculation"]}

    latest = financial_line_items[0]
    net_income = getattr(latest, "net_income", None)
    depreciation = getattr(latest, "depreciation_and_amortization", None)
    capex = getattr(latest, "capital_expenditure", None)

    if not all([net_income, depreciation, capex]):
        return {"owner_earnings": None, "details": ["Missing components for owner earnings calculation"]}

    maintenance_capex = capex * 0.75
    owner_earnings = net_income + depreciation - maintenance_capex

    return {
        "owner_earnings": owner_earnings,
        "components": {"net_income": net_income, "depreciation": depreciation, "maintenance_capex": maintenance_capex},
        "details": ["Owner earnings calculated successfully"]
    }


def calculate_intrinsic_value(financial_line_items: list, price_history: list) -> dict[str, any]:
    """Calculate intrinsic value using DCF with owner earnings, adjusted for volatility."""
    if not financial_line_items or not price_history:
        return {"intrinsic_value": None, "score": 0, "max_score": 3, "details": ["Insufficient data for valuation"]}

    earnings_data = calculate_owner_earnings(financial_line_items)
    if not earnings_data["owner_earnings"]:
        return {"intrinsic_value": None, "score": 0, "max_score": 3, "details": earnings_data["details"]}

    owner_earnings = earnings_data["owner_earnings"]
    latest = financial_line_items[0]
    shares_outstanding = getattr(latest, "outstanding_shares", None)

    if not shares_outstanding:
        return {"intrinsic_value": None, "score": 0, "max_score": 3, "details": ["Missing shares outstanding data"]}

    # Volatility analysis (Buffett prefers stable businesses)
    prices = [p["close"] for p in price_history if "close" in p]
    volatility_score = 0
    reasoning = []
    if len(prices) >= 252:  # ~1 year of daily data
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        if volatility < 0.20:
            volatility_score = 2
            reasoning.append(f"Low price volatility ({volatility:.2%}) supports stability")
        elif volatility < 0.30:
            volatility_score = 1
            reasoning.append(f"Moderate volatility ({volatility:.2%})")
        else:
            reasoning.append(f"High volatility ({volatility:.2%}) suggests instability")
    else:
        reasoning.append("Insufficient price data for volatility analysis")

    # DCF assumptions
    growth_rate = 0.05
    discount_rate = 0.09
    terminal_multiple = 12
    projection_years = 10

    future_value = 0
    for year in range(1, projection_years + 1):
        future_earnings = owner_earnings * (1 + growth_rate) ** year
        present_value = future_earnings / (1 + discount_rate) ** year
        future_value += present_value

    terminal_value = (owner_earnings * (1 + growth_rate) ** projection_years * terminal_multiple) / ((1 + discount_rate) ** projection_years)
    intrinsic_value = future_value + terminal_value

    # Add score for successful calculation
    score = volatility_score + 1  # 1 for valid intrinsic value, plus volatility
    reasoning.append("Intrinsic value calculated using DCF model with owner earnings")

    return {
        "intrinsic_value": intrinsic_value,
        "owner_earnings": owner_earnings,
        "score": score,
        "max_score": 3,
        "assumptions": {
            "growth_rate": growth_rate,
            "discount_rate": discount_rate,
            "terminal_multiple": terminal_multiple,
            "projection_years": projection_years
        },
        "details": "; ".join(reasoning)
    }


def generate_buffett_output(
    ticker: str,
    analysis_data: dict[str, any],
    portfolio_cash: float,
    model_name: str,
    model_provider: str,
) -> WarrenBuffettSignal:
    """Get investment decision from LLM with Buffett's principles"""
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Warren Buffett AI agent. Decide on investment signals based on Warren Buffett’s principles:
                - Circle of Competence: Only invest in businesses you understand
                - Margin of Safety (> 30%): Buy at a significant discount to intrinsic value
                - Economic Moat: Look for durable competitive advantages
                - Quality Management: Seek conservative, shareholder-oriented teams
                - Financial Strength: Favor low debt, strong returns on equity
                - Long-term Horizon: Invest in businesses, not just stocks
                - Sell only if fundamentals deteriorate or valuation far exceeds intrinsic value

                Follow these guidelines strictly.
                """,
            ),
            (
                "human",
                """Based on the following data, create the investment signal as Warren Buffett would:

                Analysis Data for {ticker}:
                {analysis_data}
                Portfolio Cash (Equity): ${portfolio_cash}

                Return the trading signal in the following JSON format exactly:
                {{
                  "signal": "bullish" | "bearish" | "neutral",
                  "confidence": float between 0 and 100,
                  "reasoning": "string"
                }}
                """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker, "portfolio_cash": portfolio_cash})

    def create_default_warren_buffett_signal():
        return WarrenBuffettSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=WarrenBuffettSignal,
        agent_name="warren_buffett_agent",
        default_factory=create_default_warren_buffett_signal,
    )