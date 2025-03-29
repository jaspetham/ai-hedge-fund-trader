from graph.state import AgentState, show_agent_reasoning
from tools.api import (
    get_financial_metrics,
    get_market_cap,
    search_line_items,
    get_insider_trades,
    get_company_news,
    get_prices,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm
import numpy as np


class CharlieMungerSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def charlie_munger_agent(state: AgentState):
    """
    Analyzes stocks using Charlie Munger's investing principles and mental models.
    Focuses on moat strength, management quality, predictability, and valuation.
    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    munger_analysis = {}

    for ticker in tickers:
        progress.update_status("charlie_munger_agent", ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=10)  # Munger looks at longer periods

        progress.update_status("charlie_munger_agent", ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "net_income",
                "operating_income",
                "return_on_invested_capital",
                "gross_margin",
                "operating_margin",
                "free_cash_flow",
                "capital_expenditure",
                "cash_and_equivalents",
                "total_debt",
                "shareholders_equity",
                "outstanding_shares",
                "research_and_development",
                "goodwill_and_intangible_assets",
            ],
            end_date,
            period="annual",
            limit=10  # Munger examines long-term trends
        )

        progress.update_status("charlie_munger_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date)

        progress.update_status("charlie_munger_agent", ticker, "Fetching insider trades")
        # Munger values management with skin in the game
        insider_trades = get_insider_trades(
            ticker,
            end_date,
            # Look back 2 years for insider trading patterns
            start_date=None,
            limit=100
        )

        progress.update_status("charlie_munger_agent", ticker, "Fetching company news")
        # Munger avoids businesses with frequent negative press
        company_news = get_company_news(
            ticker,
            end_date,
            # Look back 1 year for news
            start_date=None,
            limit=50
        )

        progress.update_status("charlie_munger_agent", ticker, "Fetching recent price data")
        prices = get_prices(ticker, start_date=start_date, end_date=end_date)

        progress.update_status("charlie_munger_agent", ticker, "Analyzing moat strength")
        moat_analysis = analyze_moat_strength(metrics, financial_line_items)

        progress.update_status("charlie_munger_agent", ticker, "Analyzing management quality")
        management_analysis = analyze_management_quality(financial_line_items, insider_trades)

        progress.update_status("charlie_munger_agent", ticker, "Analyzing business predictability")
        predictability_analysis = analyze_predictability(financial_line_items, prices)

        progress.update_status("charlie_munger_agent", ticker, "Calculating Munger-style valuation")
        valuation_analysis = calculate_munger_valuation(financial_line_items, market_cap, prices)

        progress.update_status("charlie_munger_agent", ticker, "Analyzing sentiment")
        sentiment_analysis = analyze_sentiment(company_news)

        # Combine partial scores with Munger's weighting preferences
        # Munger weights quality and predictability higher than current valuation
        # Adjusted to include sentiment (small weight)
        total_score = (
            moat_analysis["score"] * 0.30
            + management_analysis["score"] * 0.25
            + predictability_analysis["score"] * 0.25
            + valuation_analysis["score"] * 0.15
            + sentiment_analysis["score"] * 0.05
        )

        max_possible_score = 10  # Scale to 0-10

        # Generate a simple buy/hold/sell signal
        if total_score >= 7.5:  # Munger has very high standards
            signal = "bullish"
        elif total_score <= 4.5:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "moat_analysis": moat_analysis,
            "management_analysis": management_analysis,
            "predictability_analysis": predictability_analysis,
            "valuation_analysis": valuation_analysis,
            # Include some qualitative assessment from news
            "sentiment_analysis": sentiment_analysis
        }

        progress.update_status("charlie_munger_agent", ticker, "Generating Charlie Munger analysis")
        munger_output = generate_munger_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        munger_analysis[ticker] = {
            "signal": munger_output.signal,
            "confidence": munger_output.confidence,
            "reasoning": munger_output.reasoning
        }

        progress.update_status("charlie_munger_agent", ticker, "Done")

    # Wrap results in a single message for the chain
    message = HumanMessage(
        content=json.dumps(munger_analysis),
        name="charlie_munger_agent"
    )

    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(munger_analysis, "Charlie Munger Agent")

    # Add signals to the overall state
    state["data"]["analyst_signals"]["charlie_munger_agent"] = munger_analysis

    return {
        "messages": [message],
        "data": state["data"]
    }


def analyze_moat_strength(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze the business's competitive advantage using Munger's approach:
    - Consistent high returns on capital (ROIC)
    - Pricing power (stable/improving gross margins)
    - Low capital requirements
    - Network effects and intangible assets (R&D investments, goodwill)
    """
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze moat strength"
        }

    # 1. Return on Invested Capital (ROIC) analysis - Munger's favorite metric
    roic_values = [getattr(item, "return_on_invested_capital", None) for item in financial_line_items]
    valid_roic = [r for r in roic_values if r is not None]

    if valid_roic:
        # Check if ROIC consistently above 15% (Munger's threshold)
        high_roic_count = sum(1 for r in valid_roic if r > 0.15)
        if high_roic_count >= len(valid_roic) * 0.8:  # 80% of periods show high ROIC
            score += 3
            details.append(f"Excellent ROIC: >15% in {high_roic_count}/{len(valid_roic)} periods")
        elif high_roic_count >= len(valid_roic) * 0.5:  # 50% of periods
            score += 2
            details.append(f"Good ROIC: >15% in {high_roic_count}/{len(valid_roic)} periods")
        elif high_roic_count > 0:
            score += 1
            details.append(f"Mixed ROIC: >15% in only {high_roic_count}/{len(valid_roic)} periods")
        else:
            details.append("Poor ROIC: Never exceeds 15% threshold")
    else:
        details.append("No ROIC data available")

    # 2. Pricing power - check gross margin stability and trends
    gross_margins = [getattr(item, "gross_margin", None) for item in financial_line_items]
    valid_margins = [m for m in gross_margins if m is not None]

    if valid_margins and len(valid_margins) >= 3:
        # Munger likes stable or improving gross margins
        margin_trend = sum(1 for i in range(1, len(valid_margins)) if valid_margins[i] >= valid_margins[i-1])
        if margin_trend >= len(valid_margins) * 0.7:  # Improving in 70% of periods
            score += 2
            details.append("Strong pricing power: Gross margins consistently improving")
        elif sum(valid_margins) / len(valid_margins) > 0.3:  # Average margin > 30%
            score += 1
            details.append(f"Good pricing power: Average gross margin {sum(valid_margins)/len(valid_margins):.1%}")
        else:
            details.append("Limited pricing power: Low or declining gross margins")
    else:
        details.append("Insufficient gross margin data")

    # 3. Capital intensity - Munger prefers low capex businesses
    if len(financial_line_items) >= 3:
        capex_to_revenue = []
        for item in financial_line_items:
            capex = getattr(item, "capital_expenditure", None)
            revenue = getattr(item, "revenue", None)
            if capex is not None and revenue is not None and revenue > 0:
                # Note: capital_expenditure is typically negative in financial statements
                capex_ratio = abs(capex) / revenue
                capex_to_revenue.append(capex_ratio)

        if capex_to_revenue:
            avg_capex_ratio = sum(capex_to_revenue) / len(capex_to_revenue)
            if avg_capex_ratio < 0.05:  # Less than 5% of revenue
                score += 2
                details.append(f"Low capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
            elif avg_capex_ratio < 0.10:  # Less than 10% of revenue
                score += 1
            else:
                details.append(f"High capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
        else:
            details.append("No capital expenditure data available")
    else:
        details.append("Insufficient data for capital intensity analysis")

    # 4. Intangible assets - Munger values R&D and intellectual property
    r_and_d = [getattr(item, "research_and_development", None) for item in financial_line_items]
    goodwill_and_intangible_assets = [getattr(item, "goodwill_and_intangible_assets", None) for item in financial_line_items]

    valid_r_and_d = [r for r in r_and_d if r is not None]
    valid_goodwill = [g for g in goodwill_and_intangible_assets if g is not None]

    if valid_r_and_d and sum(valid_r_and_d) > 0:  # If company is investing in R&D
        score += 1
        details.append("Invests in R&D, building intellectual property")

    if valid_goodwill and len(valid_goodwill) > 0:
        score += 1
        details.append("Significant goodwill/intangible assets, suggesting brand value or IP")

    # Scale score to 0-10 range
    final_score = min(10, score * 10 / 9)  # Max possible raw score is 9

    return {
        "score": final_score,
        "details": "; ".join(details)
    }


def analyze_management_quality(financial_line_items: list, insider_trades: list) -> dict:
    """
    Evaluate management quality using Munger's criteria:
    - Capital allocation wisdom
    - Insider ownership and transactions
    - Cash management efficiency
    - Candor and transparency
    - Long-term focus
    """
    score = 0
    details = []

    if not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze management quality"
        }

    # 1. Capital allocation - Check FCF to net income ratio
    # Munger values companies that convert earnings to cash
    fcf_values = [getattr(item, "free_cash_flow", None) for item in financial_line_items]
    net_income_values = [getattr(item, "net_income", None) for item in financial_line_items]

    valid_fcf = [f for f in fcf_values if f is not None]
    valid_net_income = [n for n in net_income_values if n is not None]

    if valid_fcf and valid_net_income and len(valid_fcf) == len(valid_net_income):
        # Calculate FCF to Net Income ratio for each period
        fcf_to_ni_ratios = [f / n if n > 0 else 0 for f, n in zip(valid_fcf, valid_net_income)]

        if fcf_to_ni_ratios:
            avg_ratio = sum(fcf_to_ni_ratios) / len(fcf_to_ni_ratios)
            if avg_ratio > 1.1:  # FCF > net income suggests good accounting
                score += 3
                details.append(f"Excellent cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            elif avg_ratio > 0.9:  # FCF roughly equals net income
                score += 2
                details.append(f"Good cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            elif avg_ratio > 0.7:  # FCF somewhat lower than net income
                score += 1
                details.append(f"Moderate cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            else:
                details.append(f"Poor cash conversion: FCF/NI ratio of only {avg_ratio:.2f}")
        else:
            details.append("Could not calculate FCF to Net Income ratios")
    else:
        details.append("Missing FCF or Net Income data")

    # 2. Debt management - Munger is cautious about debt
    debt_values = [getattr(item, "total_debt", None) for item in financial_line_items]
    equity_values = [getattr(item, "shareholders_equity", None) for item in financial_line_items]

    valid_debt = [d for d in debt_values if d is not None]
    valid_equity = [e for e in equity_values if e is not None]

    if valid_debt and valid_equity and len(valid_debt) == len(valid_equity):
        # Calculate D/E ratio for most recent period
        recent_de_ratio = valid_debt[0] / valid_equity[0] if valid_equity[0] > 0 else float('inf')

        if recent_de_ratio < 0.3:  # Very low debt
            score += 3
            details.append(f"Conservative debt management: D/E ratio of {recent_de_ratio:.2f}")
        elif recent_de_ratio < 0.7:  # Moderate debt
            score += 2
            details.append(f"Prudent debt management: D/E ratio of {recent_de_ratio:.2f}")
        elif recent_de_ratio < 1.5:  # Higher but still reasonable debt
            score += 1
            details.append(f"Moderate debt level: D/E ratio of {recent_de_ratio:.2f}")
        else:
            details.append(f"High debt level: D/E ratio of {recent_de_ratio:.2f}")
    else:
        details.append("Missing debt or equity data")

    # 3. Cash management efficiency - Munger values appropriate cash levels
    cash_values = [getattr(item, "cash_and_equivalents", None) for item in financial_line_items]
    revenue_values = [getattr(item, "revenue", None) for item in financial_line_items]

    valid_cash = [c for c in cash_values if c is not None]
    valid_revenue = [r for r in revenue_values if r is not None]

    if valid_cash and valid_revenue and len(valid_cash) > 0 and len(valid_revenue) > 0:
        # Calculate cash to revenue ratio (Munger likes 10-20% for most businesses)
        cash_to_revenue = valid_cash[0] / valid_revenue[0] if valid_revenue[0] > 0 else 0

        if 0.1 <= cash_to_revenue <= 0.25:
            # Goldilocks zone - not too much, not too little
            score += 2
            details.append(f"Prudent cash management: Cash/Revenue ratio of {cash_to_revenue:.2f}")
        elif 0.05 <= cash_to_revenue < 0.1 or 0.25 < cash_to_revenue <= 0.4:
            # Reasonable but not ideal
            score += 1
            details.append(f"Acceptable cash position: Cash/Revenue ratio of {cash_to_revenue:.2f}")
        elif cash_to_revenue > 0.4:
            # Too much cash - potentially inefficient capital allocation
            details.append(f"Excess cash reserves: Cash/Revenue ratio of {cash_to_revenue:.2f}")
        else:
            # Too little cash - potentially risky
            details.append(f"Low cash reserves: Cash/Revenue ratio of {cash_to_revenue:.2f}")
    else:
        details.append("Insufficient cash or revenue data")

    # 4. Insider activity - Munger values skin in the game
    if insider_trades and len(insider_trades) > 0:
        # Count buys vs. sells
        buys = sum(1 for trade in insider_trades if getattr(trade, 'transaction_shares', 0) > 0)
        sells = sum(1 for trade in insider_trades if getattr(trade, 'transaction_shares', 0) < 0)

        # Calculate the buy ratio
        total_trades = buys + sells
        if total_trades > 0:
            buy_ratio = buys / total_trades
            if buy_ratio > 0.7:  # Strong insider buying
                score += 2
                details.append(f"Strong insider buying: {buys}/{total_trades} transactions are purchases")
            elif buy_ratio > 0.4:  # Balanced insider activity
                score += 1
                details.append(f"Balanced insider trading: {buys}/{total_trades} transactions are purchases")
            elif buy_ratio < 0.1 and sells > 5:  # Heavy selling
                score -= 1  # Penalty for excessive selling
                details.append(f"Concerning insider selling: {sells}/{total_trades} transactions are sales")
            else:
                details.append(f"Mixed insider activity: {buys}/{total_trades} transactions are purchases")
        else:
            details.append("No recorded insider transactions")
    else:
        details.append("No insider trading data available")

    # 5. Consistency in share count - Munger prefers stable/decreasing shares
    share_counts = [getattr(item, "outstanding_shares", None) for item in financial_line_items]
    valid_shares = [s for s in share_counts if s is not None]

    if valid_shares and len(valid_shares) >= 3:
        if valid_shares[0] < valid_shares[-1] * 0.95:  # 5%+ reduction in shares
            score += 2
            details.append("Shareholder-friendly: Reducing share count over time")
        elif valid_shares[0] < valid_shares[-1] * 1.05:  # Stable share count
            score += 1
            details.append("Stable share count: Limited dilution")
        elif valid_shares[0] > valid_shares[-1] * 1.2:  # >20% dilution
            score -= 1  # Penalty for excessive dilution
            details.append("Concerning dilution: Share count increased significantly")
        else:
            details.append("Moderate share count increase over time")
    else:
        details.append("Insufficient share count data")

    # Scale score to 0-10 range
    # Maximum possible raw score would be 12 (3+3+2+2+2)
    final_score = max(0, min(10, score * 10 / 12))

    return {
        "score": final_score,
        "details": "; ".join(details)
    }


def analyze_predictability(financial_line_items: list, prices: list) -> dict:
    """
    Assess the predictability of the business - Munger strongly prefers businesses
    whose future operations and cashflows are relatively easy to predict.
    """
    score = 0
    details = []

    if not financial_line_items or len(financial_line_items) < 5:
        return {
            "score": 0,
            "details": "Insufficient data to analyze business predictability (need 5+ years)"
        }

    # 1. Revenue stability and growth
    revenues = [getattr(item, "revenue", None) for item in financial_line_items]
    valid_revenues = [r for r in revenues if r is not None]

    if valid_revenues and len(valid_revenues) >= 5:
        # Calculate year-over-year growth rates
        growth_rates = [(valid_revenues[i] / valid_revenues[i+1] - 1) for i in range(len(valid_revenues)-1) if valid_revenues[i+1] > 0]

        if growth_rates:
            avg_growth = sum(growth_rates) / len(growth_rates)
            growth_volatility = sum(abs(r - avg_growth) for r in growth_rates) / len(growth_rates)

            if avg_growth > 0.05 and growth_volatility < 0.1:
                # Steady, consistent growth (Munger loves this)
                score += 3
                details.append(f"Highly predictable revenue: {avg_growth:.1%} avg growth with low volatility")
            elif avg_growth > 0 and growth_volatility < 0.2:
                # Positive but somewhat volatile growth
                score += 2
                details.append(f"Moderately predictable revenue: {avg_growth:.1%} avg growth with some volatility")
            elif avg_growth > 0:
                # Growing but unpredictable
                score += 1
                details.append(f"Growing but less predictable revenue: {avg_growth:.1%} avg growth with high volatility")
            else:
                details.append(f"Declining or highly unpredictable revenue: {avg_growth:.1%} avg growth")
    else:
        details.append("Insufficient revenue history for predictability analysis")

    # 2. Operating income stability
    op_income = [getattr(item, "operating_income", None) for item in financial_line_items]
    valid_op_income = [o for o in op_income if o is not None]

    if valid_op_income and len(valid_op_income) >= 5:
        # Count positive operating income periods
        positive_periods = sum(1 for income in valid_op_income if income > 0)

        if positive_periods == len(valid_op_income):
            # Consistently profitable operations
            score += 3
            details.append("Highly predictable operations: Operating income positive in all periods")
        elif positive_periods >= len(valid_op_income) * 0.8:
            # Mostly profitable operations
            score += 2
            details.append(f"Predictable operations: Operating income positive in {positive_periods}/{len(valid_op_income)} periods")
        elif positive_periods >= len(valid_op_income) * 0.6:
            # Somewhat profitable operations
            score += 1
            details.append(f"Somewhat predictable operations: Operating income positive in {positive_periods}/{len(valid_op_income)} periods")
        else:
            details.append(f"Unpredictable operations: Operating income positive in only {positive_periods}/{len(valid_op_income)} periods")
    else:
        details.append("Insufficient operating income history")

    # 3. Margin consistency - Munger values stable margins
    op_margins = [getattr(item, "operating_margin", None) for item in financial_line_items]
    valid_margins = [m for m in op_margins if m is not None]

    if valid_margins and len(valid_margins) >= 5:
        # Calculate margin volatility
        avg_margin = sum(valid_margins) / len(valid_margins)
        margin_volatility = sum(abs(m - avg_margin) for m in valid_margins) / len(valid_margins)

        if margin_volatility < 0.03:  # Very stable margins
            score += 2
            details.append(f"Highly predictable margins: {avg_margin:.1%} avg with minimal volatility")
        elif margin_volatility < 0.07:  # Moderately stable margins
            score += 1
            details.append(f"Moderately predictable margins: {avg_margin:.1%} avg with some volatility")
        else:
            details.append(f"Unpredictable margins: {avg_margin:.1%} avg with high volatility ({margin_volatility:.1%})")
    else:
        details.append("Insufficient margin history")

    # 4. Cash generation reliability
    fcf_values = [getattr(item, "free_cash_flow", None) for item in financial_line_items]
    valid_fcf = [f for f in fcf_values if f is not None]

    if valid_fcf and len(valid_fcf) >= 5:
        # Count positive FCF periods
        positive_fcf_periods = sum(1 for fcf in valid_fcf if fcf > 0)

        if positive_fcf_periods == len(valid_fcf):
            # Consistently positive FCF
            score += 2
            details.append("Highly predictable cash generation: Positive FCF in all periods")
        elif positive_fcf_periods >= len(valid_fcf) * 0.8:
            # Mostly positive FCF
            score += 1
            details.append(f"Predictable cash generation: Positive FCF in {positive_fcf_periods}/{len(valid_fcf)} periods")
        else:
            details.append(f"Unpredictable cash generation: Positive FCF in only {positive_fcf_periods}/{len(valid_fcf)} periods")
    else:
        details.append("Insufficient free cash flow history")

    # 5. Price Volatility (Munger prefers predictable businesses)
    if prices and len(prices) >= 90:
        close_prices = [getattr(p, "close", 0) for p in prices]
        if len(close_prices) >= 90:
            returns = np.diff(close_prices) / close_prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)
            if volatility < 0.20:
                score += 2
                details.append(f"Low price volatility: {volatility:.2%}")
            elif volatility < 0.30:
                score += 1
                details.append(f"Moderate price volatility: {volatility:.2%}")
            else:
                details.append(f"High price volatility: {volatility:.2%}")
    else:
        details.append("Insufficient price data for volatility analysis")

    # Scale score to 0-10 range
    final_score = min(10, score * 10 / 12)  # Max possible raw score is 12

    return {
        "score": final_score,
        "details": "; ".join(details)
    }


def calculate_munger_valuation(financial_line_items: list, market_cap: float, prices: list) -> dict:
    """
    Calculate intrinsic value using Munger's approach:
    - Focus on owner earnings (approximated by FCF)
    - Simple multiple on normalized earnings
    - Prefer paying a fair price for a wonderful business
    """
    score = 0
    details = []

    if not financial_line_items or market_cap is None or not prices:
        return {
            "score": 0,
            "details": "Insufficient data to perform valuation"
        }

    # Get FCF values (Munger's preferred "owner earnings" metric)
    fcf_values = [getattr(item, "free_cash_flow", None) for item in financial_line_items]
    valid_fcf = [f for f in fcf_values if f is not None]

    if not valid_fcf or len(valid_fcf) < 3:
        return {
            "score": 0,
            "details": "Insufficient free cash flow data for valuation"
        }

    # 1. Normalize earnings by taking average of last 3-5 years
    # (Munger prefers to normalize earnings to avoid over/under-valuation based on cyclical factors)
    normalized_fcf = sum(valid_fcf[:min(5, len(valid_fcf))]) / min(5, len(valid_fcf))

    if normalized_fcf <= 0:
        return {
            "score": 0,
            "details": f"Negative or zero normalized FCF ({normalized_fcf}), cannot value",
            "intrinsic_value": None
        }

    # 2. Calculate FCF yield (inverse of P/FCF multiple)
    fcf_yield = normalized_fcf / market_cap if market_cap > 0 else 0

    # 3. Apply Munger's FCF multiple based on business quality
    # Munger would pay higher multiples for wonderful businesses
    # Let's use a sliding scale where higher FCF yields are more attractive
    if fcf_yield > 0.08:  # >8% FCF yield (P/FCF < 12.5x)
        score += 4
        details.append(f"Excellent value: {fcf_yield:.1%} FCF yield")
    elif fcf_yield > 0.05:  # >5% FCF yield (P/FCF < 20x)
        score += 3
        details.append(f"Good value: {fcf_yield:.1%} FCF yield")
    elif fcf_yield > 0.03:  # >3% FCF yield (P/FCF < 33x)
        score += 1
        details.append(f"Fair value: {fcf_yield:.1%} FCF yield")
    else:
        details.append(f"Expensive: Only {fcf_yield:.1%} FCF yield")

    # 4. Calculate simple intrinsic value range
    # Munger tends to use straightforward valuations, avoiding complex DCF models
    conservative_value = normalized_fcf * 10  # 10x FCF = 10% yield
    reasonable_value = normalized_fcf * 15    # 15x FCF ≈ 6.7% yield
    optimistic_value = normalized_fcf * 20    # 20x FCF = 5% yield

    # 5. Calculate margins of safety
    current_to_reasonable = (reasonable_value - market_cap) / market_cap if market_cap > 0 else -1

    if current_to_reasonable > 0.3:  # >30% upside
        score += 3
        details.append(f"Large margin of safety: {current_to_reasonable:.1%} upside to reasonable value")
    elif current_to_reasonable > 0.1:  # >10% upside
        score += 2
        details.append(f"Moderate margin of safety: {current_to_reasonable:.1%} upside to reasonable value")
    elif current_to_reasonable > -0.1:  # Within 10% of reasonable value
        score += 1
        details.append(f"Fair price: Within 10% of reasonable value ({current_to_reasonable:.1%})")
    else:
        details.append(f"Expensive: {-current_to_reasonable:.1%} premium to reasonable value")

    # 6. Check earnings trajectory for additional context
    # Munger likes growing owner earnings
    if len(valid_fcf) >= 3:
        recent_avg = sum(valid_fcf[:3]) / 3
        older_avg = sum(valid_fcf[-3:]) / 3 if len(valid_fcf) >= 6 else valid_fcf[-1]

        if recent_avg > older_avg * 1.2:  # >20% growth in FCF
            score += 3
            details.append("Growing FCF trend adds to intrinsic value")
        elif recent_avg > older_avg:
            score += 2
            details.append("Stable to growing FCF supports valuation")
        else:
            details.append("Declining FCF trend is concerning")

    # 7. Price Volatility (Munger prefers stability)
    if prices and len(prices) >= 90:
        close_prices = [getattr(p, "close", 0) for p in prices]
        if len(close_prices) >= 90:
            returns = np.diff(close_prices) / close_prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)
            if volatility < 0.20:
                score += 2
                details.append(f"Low price volatility: {volatility:.2%}")
            elif volatility < 0.30:
                score += 1
                details.append(f"Moderate price volatility: {volatility:.2%}")
            else:
                details.append(f"High price volatility: {volatility:.2%}")
    else:
        details.append("Insufficient price data for volatility analysis")

    # Scale score to 0-10 range
    final_score = min(10, score * 10 / 13)  # Max raw score is 13 (4+3+3+3)

    return {
        "score": final_score,
        "details": "; ".join(details),
        "intrinsic_value_range": {
            "conservative": conservative_value,
            "reasonable": reasonable_value,
            "optimistic": optimistic_value
        },
        "fcf_yield": fcf_yield,
        "normalized_fcf": normalized_fcf
    }


def analyze_sentiment(news_items: list) -> dict:
    """
    Simple qualitative analysis of recent news adapted as a scored metric.
    Munger pays attention to significant news but doesn't overreact to short-term stories.
    """
    score = 5  # Neutral default
    details = []

    if not news_items or len(news_items) == 0:
        details.append("No news data available")
        return {"score": score, "details": "; ".join(details)}

    negative_keywords = ["lawsuit", "fraud", "bankruptcy", "investigation", "decline"]
    negative_count = sum(1 for news in news_items if any(word in (getattr(news, "title", "") or "").lower() for word in negative_keywords))

    if negative_count > len(news_items) * 0.3:
        score = 3
        details.append(f"Significant negative news: {negative_count}/{len(news_items)} items.")
    elif negative_count > 0:
        score = 4
        details.append(f"Some negative news: {negative_count}/{len(news_items)} items.")
    else:
        score = 6
        details.append("No significant negative news.")

    return {"score": score, "details": "; ".join(details)}


def generate_munger_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> CharlieMungerSignal:
    """
    Generates investment decisions in the style of Charlie Munger.
    """
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Charlie Munger AI agent, making investment decisions using his principles:

            1. Focus on the quality and predictability of the business.
            2. Rely on mental models from multiple disciplines to analyze investments.
            3. Look for strong, durable competitive advantages (moats).
            4. Emphasize long-term thinking and patience.
            5. Value management integrity and competence.
            6. Prioritize businesses with high returns on invested capital.
            7. Pay a fair price for wonderful businesses.
            8. Never overpay, always demand a margin of safety.
            9. Avoid complexity and businesses you don't understand.
            10. "Invert, always invert" - focus on avoiding stupidity rather than seeking brilliance.

            Rules:
            - Praise businesses with predictable, consistent operations and cash flows.
            - Value businesses with high ROIC and pricing power.
            - Prefer simple businesses with understandable economics.
            - Admire management with skin in the game and shareholder-friendly capital allocation.
            - Focus on long-term economics rather than short-term metrics.
            - Be skeptical of businesses with rapidly changing dynamics or excessive share dilution.
            - Avoid excessive leverage or financial engineering.
            - Provide a rational, data-driven recommendation (bullish, bearish, or neutral)."""
        ),
        (
            "human",
            """Based on the following analysis, create a Munger-style investment signal.

            Analysis Data for {ticker}:
            {analysis_data}

            Return the trading signal in this JSON format:
            {{
              "signal": "bullish/bearish/neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """
        )
    ])

    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker
    })

    def create_default_charlie_munger_signal():
        return CharlieMungerSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral"
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=CharlieMungerSignal,
        agent_name="charlie_munger_agent",
        default_factory=create_default_charlie_munger_signal,
    )