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


class BillAckmanSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def bill_ackman_agent(state: AgentState):
    """
    Analyzes stocks using Bill Ackman's investing principles:
    1. High-quality businesses with durable competitive advantages (moats).
    2. Consistent free cash flow and growth potential.
    3. Strong financial discipline (reasonable leverage, efficient capital allocation).
    4. Valuation with a margin of safety.
    5. High-conviction, long-term investments, sometimes with an activist angle.
    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    ackman_analysis = {}

    for ticker in tickers:
        progress.update_status("bill_ackman_agent", ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5)

        progress.update_status("bill_ackman_agent", ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "operating_margin",
                "debt_to_equity",
                "free_cash_flow",
                "total_assets",
                "total_liabilities",
                "dividends_and_other_cash_distributions",
                "outstanding_shares",
            ],
            end_date,
            period="annual",
            limit=5,
        )

        progress.update_status("bill_ackman_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date)

        progress.update_status("bill_ackman_agent", ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(ticker, end_date, start_date=None, limit=50)
        if not insider_trades:
            progress.update_status("bill_ackman_agent", ticker, "No insider trades available")

        progress.update_status("bill_ackman_agent", ticker, "Fetching company news")
        company_news = get_company_news(ticker, end_date, start_date=None, limit=50)
        if not company_news:
            progress.update_status("bill_ackman_agent", ticker, "No company news available")

        progress.update_status("bill_ackman_agent", ticker, "Fetching recent price data")
        prices = get_prices(ticker, start_date=start_date, end_date=end_date)

        progress.update_status("bill_ackman_agent", ticker, "Analyzing business quality")
        quality_analysis = analyze_business_quality(metrics, financial_line_items)

        progress.update_status("bill_ackman_agent", ticker, "Analyzing financial discipline")
        balance_sheet_analysis = analyze_financial_discipline(financial_line_items)

        progress.update_status("bill_ackman_agent", ticker, "Analyzing valuation")
        valuation_analysis = analyze_valuation(financial_line_items, market_cap)

        progress.update_status("bill_ackman_agent", ticker, "Analyzing insider activity")
        insider_analysis = analyze_insider_activity(insider_trades)

        progress.update_status("bill_ackman_agent", ticker, "Analyzing sentiment")
        sentiment_analysis = analyze_sentiment(company_news)

        # Aggregate scoring with weights reflecting Ackman's priorities
        # Quality (40%), Financial Discipline (25%), Valuation (20%), Insider (10%), Sentiment (5%)
        total_score = (
            quality_analysis["score"] * 0.40
            + balance_sheet_analysis["score"] * 0.25
            + valuation_analysis["score"] * 0.20
            + insider_analysis["score"] * 0.10
            + sentiment_analysis["score"] * 0.05
        )
        max_possible_score = 10  # Normalized to 0-10 scale

        # Map total_score to signal
        if total_score >= 7.5:
            signal = "bullish"
        elif total_score <= 4.5:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "quality_analysis": quality_analysis,
            "balance_sheet_analysis": balance_sheet_analysis,
            "valuation_analysis": valuation_analysis,
            "insider_analysis": insider_analysis,
            "sentiment_analysis": sentiment_analysis,
        }

        progress.update_status("bill_ackman_agent", ticker, "Generating Bill Ackman analysis")
        ackman_output = generate_ackman_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        ackman_analysis[ticker] = {
            "signal": ackman_output.signal,
            "confidence": ackman_output.confidence,
            "reasoning": ackman_output.reasoning,
        }

        progress.update_status("bill_ackman_agent", ticker, "Done")

    # Wrap results in a single message
    message = HumanMessage(content=json.dumps(ackman_analysis), name="bill_ackman_agent")

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(ackman_analysis, "Bill Ackman Agent")

    state["data"]["analyst_signals"]["bill_ackman_agent"] = ackman_analysis
    return {"messages": [message], "data": state["data"]}


def analyze_business_quality(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze business quality:
    - Revenue growth
    - Operating margin consistency
    - Free cash flow consistency
    - Return on Equity (ROE)
    """
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data to analyze business quality"}

    # 1. Revenue Growth
    revenues = [getattr(item, "revenue", None) for item in financial_line_items if hasattr(item, "revenue")]
    valid_revenues = [r for r in revenues if r is not None]
    if len(valid_revenues) >= 2:
        initial, final = valid_revenues[-1], valid_revenues[0]  # Oldest to newest
        if initial > 0 and final > initial:
            growth_rate = (final - initial) / initial
            if growth_rate > 0.5:
                score += 4
                details.append(f"Strong revenue growth: {growth_rate:.1%}")
            elif growth_rate > 0.2:
                score += 2
                details.append(f"Moderate revenue growth: {growth_rate:.1%}")
            else:
                details.append(f"Low revenue growth: {growth_rate:.1%}")
        else:
            details.append("Revenue flat or declined.")
    else:
        details.append("Insufficient revenue data.")

    # 2. Operating Margin Consistency
    op_margins = [getattr(item, "operating_margin", None) for item in financial_line_items if hasattr(item, "operating_margin")]
    valid_margins = [m for m in op_margins if m is not None]
    if valid_margins:
        above_15 = sum(1 for m in valid_margins if m > 0.15)
        if above_15 >= (len(valid_margins) // 2 + 1):
            score += 3
            details.append(f"Operating margin >15% in {above_15}/{len(valid_margins)} periods.")
        else:
            details.append(f"Operating margin inconsistent or below 15%.")
    else:
        details.append("No operating margin data.")

    # 3. Free Cash Flow Consistency
    fcf_vals = [getattr(item, "free_cash_flow", None) for item in financial_line_items if hasattr(item, "free_cash_flow")]
    valid_fcf = [f for f in fcf_vals if f is not None]
    if valid_fcf:
        positive_fcf = sum(1 for f in valid_fcf if f > 0)
        if positive_fcf >= (len(valid_fcf) // 2 + 1):
            score += 2
            details.append(f"Positive FCF in {positive_fcf}/{len(valid_fcf)} periods.")
        else:
            details.append(f"FCF not consistently positive.")
    else:
        details.append("No FCF data.")

    # 4. ROE from Latest Metrics
    if metrics and hasattr(metrics[0], "return_on_equity") and metrics[0].return_on_equity is not None:
        roe = metrics[0].return_on_equity
        if roe > 0.15:
            score += 3
            details.append(f"High ROE: {roe:.1%}")
        elif roe > 0.10:
            score += 1
            details.append(f"Moderate ROE: {roe:.1%}")
        else:
            details.append(f"Low ROE: {roe:.1%}")
    else:
        details.append("ROE data unavailable.")

    final_score = min(10, score * (10 / 12))  # Scale to 0-10
    return {"score": final_score, "details": "; ".join(details)}


def analyze_financial_discipline(financial_line_items: list) -> dict:
    """
    Evaluate financial discipline:
    - Debt-to-equity trend
    - Capital returns (dividends, buybacks)
    """
    score = 0
    details = []

    if not financial_line_items:
        return {"score": 0, "details": "Insufficient data to analyze financial discipline"}

    # 1. Debt-to-Equity Trend
    dte_vals = [getattr(item, "debt_to_equity", None) for item in financial_line_items if hasattr(item, "debt_to_equity")]
    valid_dte = [d for d in dte_vals if d is not None]
    if valid_dte:
        below_one = sum(1 for d in valid_dte if d < 1.0)
        if below_one >= (len(valid_dte) // 2 + 1):
            score += 4
            details.append(f"D/E < 1.0 in {below_one}/{len(valid_dte)} periods.")
        else:
            details.append(f"D/E >= 1.0 in many periods.")
    else:
        # Fallback to liabilities-to-assets
        liab_to_assets = []
        for item in financial_line_items:
            liabilities = getattr(item, "total_liabilities", 0)
            assets = getattr(item, "total_assets", 0)
            if assets > 0:
                liab_to_assets.append(liabilities / assets)
        if liab_to_assets:
            below_50 = sum(1 for r in liab_to_assets if r < 0.5)
            if below_50 >= (len(liab_to_assets) // 2 + 1):
                score += 4
                details.append(f"Liabilities-to-assets < 50% in {below_50}/{len(liab_to_assets)} periods.")
            else:
                details.append(f"Liabilities-to-assets >= 50% in many periods.")
        else:
            details.append("No leverage data available.")

    # 2. Capital Returns (Dividends or Buybacks)
    dividends = [getattr(item, "dividends_and_other_cash_distributions", None) for item in financial_line_items if hasattr(item, "dividends_and_other_cash_distributions")]
    valid_dividends = [d for d in dividends if d is not None]
    if valid_dividends:
        paying_dividends = sum(1 for d in valid_dividends if d < 0)
        if paying_dividends >= (len(valid_dividends) // 2 + 1):
            score += 3
            details.append(f"Dividends paid in {paying_dividends}/{len(valid_dividends)} periods.")
        else:
            details.append(f"Dividends not consistent.")

    shares = [getattr(item, "outstanding_shares", None) for item in financial_line_items if hasattr(item, "outstanding_shares")]
    valid_shares = [s for s in shares if s is not None]
    if len(valid_shares) >= 2 and valid_shares[0] > valid_shares[-1]:
        score += 3
        details.append("Share count reduced, indicating buybacks.")
    else:
        details.append("No significant share reduction.")

    final_score = min(10, score * (10 / 10))  # Scale to 0-10
    return {"score": final_score, "details": "; ".join(details)}


def analyze_valuation(financial_line_items: list, market_cap: float) -> dict:
    """
Ackman invests in companies trading at a discount to intrinsic value.
    We can do a simplified DCF or an FCF-based approach.
    This function currently uses the latest free cash flow only,
    but you could expand it to use an average or multi-year FCF approach.
    """
    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "Insufficient data for valuation"}

    latest = financial_line_items[0]  # Most recent
    fcf = getattr(latest, "free_cash_flow", 0)

    growth_rate = 0.06
    discount_rate = 0.10
    terminal_multiple = 15
    projection_years = 5

    if fcf <= 0:
        return {"score": 0, "details": f"No positive FCF for valuation; FCF = {fcf}"}

    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv

    terminal_value = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) / ((1 + discount_rate) ** projection_years)
    intrinsic_value = present_value + terminal_value
    margin_of_safety = (intrinsic_value - market_cap) / market_cap if market_cap > 0 else -1

    score = 0
    if margin_of_safety > 0.3:
        score = 8
        details = [f"Intrinsic value: ${intrinsic_value:,.2f}", f"Market cap: ${market_cap:,.2f}", f"Margin of safety: {margin_of_safety:.2%}"]
    elif margin_of_safety > 0.1:
        score = 4
        details = [f"Intrinsic value: ${intrinsic_value:,.2f}", f"Market cap: ${market_cap:,.2f}", f"Margin of safety: {margin_of_safety:.2%}"]
    else:
        details = [f"Intrinsic value: ${intrinsic_value:,.2f}", f"Market cap: ${market_cap:,.2f}", f"Low or negative margin of safety: {margin_of_safety:.2%}"]

    return {"score": score, "details": "; ".join(details)}


def analyze_insider_activity(insider_trades: list) -> dict:
    """
    Ackman might see insider buying as a positive signal, heavy selling as a concern.
    """
    score = 5  # Neutral default
    details = []

    if not insider_trades:
        details.append("No insider trades data available.")
        return {"score": score, "details": "; ".join(details)}

    buys, sells = 0, 0
    for trade in insider_trades:
        if trade.transaction_shares is not None:
            if trade.transaction_shares > 0:
                buys += 1
            elif trade.transaction_shares < 0:
                sells += 1

    total = buys + sells
    if total == 0:
        details.append("No buy/sell transactions found.")
        return {"score": score, "details": "; ".join(details)}

    buy_ratio = buys / total
    if buy_ratio > 0.7:
        score = 8
        details.append(f"Strong insider buying: {buys} buys vs. {sells} sells.")
    elif buy_ratio < 0.3:
        score = 3
        details.append(f"Heavy insider selling: {buys} buys vs. {sells} sells.")
    else:
        details.append(f"Balanced insider activity: {buys} buys vs. {sells} sells.")

    return {"score": score, "details": "; ".join(details)}


def analyze_sentiment(news_items: list) -> dict:
    """
    Positive sentiment supports conviction; negative news could signal risk.
    """
    score = 5  # Neutral default
    details = []

    if not news_items:
        details.append("No news data available.")
        return {"score": score, "details": "; ".join(details)}

    negative_keywords = ["lawsuit", "fraud", "investigation", "decline", "restructuring"]
    negative_count = sum(1 for news in news_items if any(word in (news.title or "").lower() for word in negative_keywords))

    if negative_count > len(news_items) * 0.3:
        score = 3
        details.append(f"Significant negative news: {negative_count}/{len(news_items)}.")
    elif negative_count > 0:
        score = 4
        details.append(f"Some negative news: {negative_count}/{len(news_items)}.")
    else:
        score = 7
        details.append("Mostly positive or neutral news.")

    return {"score": score, "details": "; ".join(details)}


def generate_ackman_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> BillAckmanSignal:
    """
    Generates investment decisions in the style of Bill Ackman.
    """
    template = ChatPromptTemplate.from_messages([
        (
             "system",
            """You are a Bill Ackman AI agent, making investment decisions using his principles:

            1. Seek high-quality businesses with durable competitive advantages (moats).
            2. Prioritize consistent free cash flow and growth potential.
            3. Advocate for strong financial discipline (reasonable leverage, efficient capital allocation).
            4. Valuation matters: target intrinsic value and margin of safety.
            5. Invest with high conviction in a concentrated portfolio for the long term.
            6. Potential activist approach if management or operational improvements can unlock value.

            Rules:
            - Evaluate brand strength, market position, or other moats.
            - Check free cash flow generation, stable or growing earnings.
            - Analyze balance sheet health (reasonable debt, good ROE).
            - Buy at a discount to intrinsic value; higher discount => stronger conviction.
            - Engage if management is suboptimal or if there's a path for strategic improvements.
            - Provide a rational, data-driven recommendation (bullish, bearish, or neutral)."""
        ),
        (
            "human",
            """Based on the following analysis, create an Ackman-style investment signal.

            Analysis Data for {ticker}:
            {analysis_data}

            Return the trading signal in this JSON format:
            {{
              "signal": "bullish" or "bearish" or "neutral",
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

    def create_default_bill_ackman_signal():
        return BillAckmanSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral"
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=BillAckmanSignal,
        agent_name="bill_ackman_agent",
        default_factory=create_default_bill_ackman_signal,
    )