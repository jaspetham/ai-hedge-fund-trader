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
import math


class BenGrahamSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def ben_graham_agent(state: AgentState):
    """
    Analyzes stocks using Benjamin Graham's classic value-investing principles:
    1. Earnings stability over multiple years.
    2. Solid financial strength (low debt, adequate liquidity).
    3. Discount to intrinsic value (e.g., Graham Number or net-net).
    4. Adequate margin of safety.
    Incorporates additional data (insider trades, news, prices) conservatively.
    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    graham_analysis = {}

    for ticker in tickers:
        progress.update_status("ben_graham_agent", ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=10)

        progress.update_status("ben_graham_agent", ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "earnings_per_share",
                "revenue",
                "net_income",
                "book_value_per_share",
                "total_assets",
                "total_liabilities",
                "current_assets",
                "current_liabilities",
                "dividends_and_other_cash_distributions",
                "outstanding_shares",
            ],
            end_date,
            period="annual",
            limit=10,
        )

        progress.update_status("ben_graham_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date)

        # Additional data from Stanley's approach, used conservatively
        progress.update_status("ben_graham_agent", ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(ticker, end_date, start_date=None, limit=50)
        if not insider_trades:
            progress.update_status("ben_graham_agent", ticker, "No insider trades available")

        progress.update_status("ben_graham_agent", ticker, "Fetching company news")
        company_news = get_company_news(ticker, end_date, start_date=None, limit=50)
        if not company_news:
            progress.update_status("ben_graham_agent", ticker, "No company news available")

        progress.update_status("ben_graham_agent", ticker, "Fetching recent price data")
        prices = get_prices(ticker, start_date=start_date, end_date=end_date)

        # Perform sub-analyses
        progress.update_status("ben_graham_agent", ticker, "Analyzing earnings stability")
        earnings_analysis = analyze_earnings_stability(financial_line_items)

        progress.update_status("ben_graham_agent", ticker, "Analyzing financial strength")
        strength_analysis = analyze_financial_strength(financial_line_items)

        progress.update_status("ben_graham_agent", ticker, "Analyzing Graham valuation")
        valuation_analysis = analyze_valuation_graham(financial_line_items, market_cap, prices)

        progress.update_status("ben_graham_agent", ticker, "Analyzing insider activity")
        insider_analysis = analyze_insider_activity(insider_trades)

        progress.update_status("ben_graham_agent", ticker, "Analyzing sentiment")
        sentiment_analysis = analyze_sentiment(company_news)

        # Aggregate scoring with weights reflecting Graham's priorities
        # Earnings (35%), Financial Strength (35%), Valuation (20%), Insider (5%), Sentiment (5%)
        total_score = (
            earnings_analysis["score"] * 0.35
            + strength_analysis["score"] * 0.35
            + valuation_analysis["score"] * 0.20
            + insider_analysis["score"] * 0.05
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
            "earnings_analysis": earnings_analysis,
            "strength_analysis": strength_analysis,
            "valuation_analysis": valuation_analysis,
            "insider_analysis": insider_analysis,
            "sentiment_analysis": sentiment_analysis,
        }

        progress.update_status("ben_graham_agent", ticker, "Generating Ben Graham analysis")
        graham_output = generate_graham_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        graham_analysis[ticker] = {
            "signal": graham_output.signal,
            "confidence": graham_output.confidence,
            "reasoning": graham_output.reasoning,
        }

        progress.update_status("ben_graham_agent", ticker, "Done")

    # Wrap results in a single message
    message = HumanMessage(content=json.dumps(graham_analysis), name="ben_graham_agent")

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(graham_analysis, "Ben Graham Agent")

    state["data"]["analyst_signals"]["ben_graham_agent"] = graham_analysis
    return {"messages": [message], "data": state["data"]}


def analyze_earnings_stability(financial_line_items: list) -> dict:
    """
    Graham wants at least several years of consistently positive earnings (ideally 5+).
    Checks:
    1. Number of years with positive EPS.
    2. Growth in EPS from first to last period.
    """
    score = 0
    details = []

    if not financial_line_items:
        return {"score": score, "details": "Insufficient data for earnings stability analysis"}

    eps_vals = [getattr(item, "earnings_per_share", None) for item in financial_line_items if hasattr(item, "earnings_per_share")]

    if len(eps_vals) < 2:
        details.append("Not enough multi-year EPS data.")
        return {"score": score, "details": "; ".join(details)}

    # 1. Consistently positive EPS
    positive_eps_years = sum(1 for e in eps_vals if e is not None and e > 0)
    total_eps_years = len([e for e in eps_vals if e is not None])
    if total_eps_years == 0:
        details.append("No valid EPS data available.")
        return {"score": score, "details": "; ".join(details)}

    if positive_eps_years == total_eps_years:
        score += 5
        details.append(f"EPS positive in all {total_eps_years} periods.")
    elif positive_eps_years >= (total_eps_years * 0.8):
        score += 3
        details.append(f"EPS positive in {positive_eps_years}/{total_eps_years} periods.")
    else:
        details.append(f"EPS negative in {total_eps_years - positive_eps_years}/{total_eps_years} periods.")

    # 2. EPS growth
    if eps_vals[-1] is not None and eps_vals[0] is not None and eps_vals[-1] > eps_vals[0]:
        score += 2
        details.append("EPS grew from earliest to latest period.")
    else:
        details.append("EPS did not grow from earliest to latest period or data missing.")

    final_score = min(10, score * (10 / 7))  # Scale to 0-10
    return {"score": final_score, "details": "; ".join(details)}


def analyze_financial_strength(financial_line_items: list) -> dict:
    """
    Graham checks liquidity (current ratio >= 2), manageable debt,
    and dividend record (preferably some history of dividends).
    """
    score = 0
    details = []

    if not financial_line_items:
        return {"score": score, "details": "No data for financial strength analysis"}

    latest_item = financial_line_items[0]  # Most recent
    total_assets = getattr(latest_item, "total_assets", 0)
    total_liabilities = getattr(latest_item, "total_liabilities", 0)
    current_assets = getattr(latest_item, "current_assets", 0)
    current_liabilities = getattr(latest_item, "current_liabilities", 0)

    # 1. Current ratio
    if current_liabilities > 0:
        current_ratio = current_assets / current_liabilities
        if current_ratio >= 2.0:
            score += 4
            details.append(f"Current ratio = {current_ratio:.2f} (>=2.0: solid).")
        elif current_ratio >= 1.5:
            score += 2
            details.append(f"Current ratio = {current_ratio:.2f} (moderately strong).")
        else:
            details.append(f"Current ratio = {current_ratio:.2f} (<1.5: weaker liquidity).")
    else:
        details.append("Cannot compute current ratio (missing or zero current_liabilities).")

    # 2. Debt vs. Assets
    if total_assets > 0:
        debt_ratio = total_liabilities / total_assets
        if debt_ratio < 0.5:
            score += 4
            details.append(f"Debt ratio = {debt_ratio:.2f}, under 0.50 (conservative).")
        elif debt_ratio < 0.8:
            score += 2
            details.append(f"Debt ratio = {debt_ratio:.2f}, somewhat high but acceptable.")
        else:
            details.append(f"Debt ratio = {debt_ratio:.2f}, high by Graham standards.")
    else:
        details.append("Cannot compute debt ratio (missing total_assets).")

    # 3. Dividend track record
    div_periods = [getattr(item, "dividends_and_other_cash_distributions", None) for item in financial_line_items if hasattr(item, "dividends_and_other_cash_distributions")]
    if div_periods:
        div_paid_years = sum(1 for d in div_periods if d is not None and d < 0)  # Negative indicates payout
        total_div_periods = len([d for d in div_periods if d is not None])
        if total_div_periods > 0 and div_paid_years >= (total_div_periods // 2 + 1):
            score += 2
            details.append(f"Dividends paid in {div_paid_years}/{total_div_periods} years.")
        else:
            details.append(f"Dividends paid in only {div_paid_years}/{total_div_periods} years or insufficient data.")
    else:
        details.append("No dividend data available.")

    final_score = min(10, score * (10 / 10))  # Scale to 0-10
    return {"score": final_score, "details": "; ".join(details)}


def analyze_valuation_graham(financial_line_items: list, market_cap: float, prices: list) -> dict:
    """
    Core Graham approach to valuation:
    1. Net-Net Check: (Current Assets - Total Liabilities) vs. Market Cap
    2. Graham Number: sqrt(22.5 * EPS * Book Value per Share)
    3. Compare current price to Graham Number for margin of safety
    4. Check price stability (low volatility preferred)
    """
    if not financial_line_items or not market_cap or market_cap <= 0 or not prices:
        return {"score": 0, "details": "Insufficient data for valuation"}

    latest = financial_line_items[0]  # Most recent
    current_assets = getattr(latest, "current_assets", 0)
    total_liabilities = getattr(latest, "total_liabilities", 0)
    book_value_ps = getattr(latest, "book_value_per_share", 0)
    eps = getattr(latest, "earnings_per_share", 0)
    shares_outstanding = getattr(latest, "outstanding_shares", 0)

    details = []
    score = 0

    # 1. Net-Net Check
    net_current_asset_value = current_assets - total_liabilities
    if net_current_asset_value > 0 and shares_outstanding > 0:
        ncav_ps = net_current_asset_value / shares_outstanding
        price_per_share = market_cap / shares_outstanding
        details.append(f"NCAV Per Share = {ncav_ps:.2f}, Price Per Share = {price_per_share:.2f}")
        if ncav_ps > price_per_share:
            score += 4
            details.append("Net-Net: NCAV exceeds price (strong value).")
        elif ncav_ps >= (price_per_share * 0.67):
            score += 2
            details.append("NCAV >= 2/3 of price (moderate value).")
        else:
            details.append("NCAV below 2/3 of price.")
    else:
        details.append("Insufficient data for net-net valuation.")

    # 2. Graham Number
    graham_number = None
    if eps > 0 and book_value_ps > 0:
        graham_number = math.sqrt(22.5 * eps * book_value_ps)
        details.append(f"Graham Number = {graham_number:.2f}")
    else:
        details.append("Cannot compute Graham Number (EPS or BVPS missing/<=0).")

    # 3. Margin of Safety
    if graham_number and shares_outstanding > 0:
        current_price = market_cap / shares_outstanding
        if current_price > 0:
            margin_of_safety = (graham_number - current_price) / current_price
            details.append(f"Margin of Safety = {margin_of_safety:.2%}")
            if margin_of_safety > 0.5:
                score += 4
                details.append("Price well below Graham Number (>=50% margin).")
            elif margin_of_safety > 0.2:
                score += 2
                details.append("Some margin of safety (>20%).")
            else:
                details.append("Low or negative margin of safety.")
    else:
        details.append("Cannot compute margin of safety.")

    # 4. Price Stability (Graham prefers low volatility)
    if len(prices) > 10:
        close_prices = [p.close for p in prices if p.close is not None]
        if len(close_prices) > 10:
            daily_returns = [(close_prices[i] - close_prices[i-1]) / close_prices[i-1] for i in range(1, len(close_prices)) if close_prices[i-1] > 0]
            if daily_returns:
                import statistics
                volatility = statistics.pstdev(daily_returns)
                if volatility < 0.02:
                    score += 2
                    details.append(f"Low volatility: {volatility:.2%}")
                elif volatility < 0.04:
                    score += 1
                    details.append(f"Moderate volatility: {volatility:.2%}")
                else:
                    details.append(f"High volatility: {volatility:.2%}")
            else:
                details.append("Insufficient returns data for volatility.")
        else:
            details.append("Not enough price data for volatility.")
    else:
        details.append("Insufficient price history.")

    final_score = min(10, score * (10 / 10))  # Scale to 0-10
    return {"score": final_score, "details": "; ".join(details)}


def analyze_insider_activity(insider_trades: list) -> dict:
    """
    Graham would view heavy insider selling skeptically, buying as neutral/positive.
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
        score = 7
        details.append(f"Mostly insider buying: {buys} buys vs. {sells} sells.")
    elif buy_ratio < 0.3:
        score = 3
        details.append(f"Heavy insider selling: {buys} buys vs. {sells} sells.")
    else:
        details.append(f"Balanced insider activity: {buys} buys vs. {sells} sells.")

    return {"score": score, "details": "; ".join(details)}


def analyze_sentiment(news_items: list) -> dict:
    """
    Graham would penalize negative news (e.g., lawsuits) but not overemphasize sentiment.
    """
    score = 5  # Neutral default
    details = []

    if not news_items:
        details.append("No news data available.")
        return {"score": score, "details": "; ".join(details)}

    negative_keywords = ["lawsuit", "fraud", "investigation", "decline", "bankruptcy"]
    negative_count = sum(1 for news in news_items if any(word in (news.title or "").lower() for word in negative_keywords))

    if negative_count > len(news_items) * 0.3:
        score = 3
        details.append(f"Significant negative news: {negative_count}/{len(news_items)}.")
    elif negative_count > 0:
        score = 4
        details.append(f"Some negative news: {negative_count}/{len(news_items)}.")
    else:
        score = 6
        details.append("No significant negative news.")

    return {"score": score, "details": "; ".join(details)}


def generate_graham_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> BenGrahamSignal:
    """
    Generates an investment decision in the style of Benjamin Graham:
    - Value emphasis, margin of safety, net-nets, conservative balance sheet, stable earnings.
    """
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Benjamin Graham AI agent, making investment decisions using his principles:
            1. Insist on a margin of safety by buying below intrinsic value (e.g., Graham Number, net-net).
            2. Emphasize financial strength (low leverage, ample current assets).
            3. Prefer stable earnings over multiple years.
            4. Consider dividend record for extra safety.
            5. View insider selling skeptically, negative news as a warning.
            6. Avoid speculative assumptions; focus on proven metrics.

            Return a rational recommendation: bullish, bearish, or neutral, with a confidence level (0-100) and concise reasoning.
            """
        ),
        (
            "human",
            """Based on the following analysis, create a Graham-style investment signal:

            Analysis Data for {ticker}:
            {analysis_data}

            Return JSON in this format:
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

    def create_default_ben_graham_signal():
        return BenGrahamSignal(signal="neutral", confidence=0.0, reasoning="Error in generating analysis; defaulting to neutral.")

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=BenGrahamSignal,
        agent_name="ben_graham_agent",
        default_factory=create_default_ben_graham_signal,
    )