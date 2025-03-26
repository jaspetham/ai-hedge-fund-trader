import os
from dotenv import load_dotenv
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data import NewsClient, NewsRequest
from yfinance import Ticker
from data.cache import get_cache
from data.models import (
    Price,
    PriceResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    LineItem,
    LineItemResponse,
    NewsItem,
)

# Global cache instance
_cache = get_cache()

# Alpaca API credentials
load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env file")
alpaca_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
news_client = NewsClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch historical price data from Alpaca."""
    # Adjust start_date to ensure enough data (e.g., 30 days for momentum)
    adjusted_start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    if cached_data := _cache.get_prices(ticker):
        filtered_data = [Price(**price) for price in cached_data if adjusted_start_date <= price["time"] <= end_date]
        if filtered_data:
            return filtered_data

    request_params = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=adjusted_start_date,
        end=end_date
    )
    bars = alpaca_client.get_stock_bars(request_params).df
    if bars.empty:
        print(f"No historical data for {ticker} from {adjusted_start_date} to {end_date}")
        return []

    bars = bars.reset_index()
    prices = [
        Price(
            open=row["open"],
            close=row["close"],
            high=row["high"],
            low=row["low"],
            volume=int(row["volume"]),
            time=pd.Timestamp(row["timestamp"]).strftime("%Y-%m-%d")
        ) for _, row in bars.iterrows()
    ]
    # Add latest price if end_date is today
    if end_date == pd.Timestamp.now().strftime("%Y-%m-%d"):
        latest = get_latest_price(ticker)
        prices.append(Price(
            open=latest, close=latest, high=latest, low=latest,  # Simplified
            volume=0,  # Volume unavailable in latest quote
            time=end_date
        ))
    _cache.set_prices(ticker, [p.model_dump() for p in prices])
    return prices

def get_latest_price(ticker: str) -> float:
    """Fetch the latest real-time price (midpoint of bid/ask)."""
    request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
    quote = alpaca_client.get_stock_latest_quote(request)
    if ticker in quote and quote[ticker].bid_price and quote[ticker].ask_price:
        return (quote[ticker].bid_price + quote[ticker].ask_price) / 2
    return 100.0  # Fallback for simulation

def get_market_cap(ticker: str, end_date: str | None = None) -> float:
    yf_ticker = Ticker(ticker)
    if not end_date:
        return float(yf_ticker.info.get("marketCap", 0.0))
    prices = yf_ticker.history(start=end_date, end=end_date)
    if prices.empty:
        return 0.0
    shares = yf_ticker.info.get("sharesOutstanding", 0)
    return float(prices["Close"].iloc[0] * shares if shares else 0.0)

def get_financial_metrics(ticker: str, end_date: str, period: str = "annual", limit: int = 5) -> list[FinancialMetrics]:
    """Fetch financial metrics from yfinance with improved mapping."""
    yf_ticker = Ticker(ticker)
    financials = yf_ticker.financials if period == "annual" else yf_ticker.quarterly_financials
    if financials.empty:
        return []

    metrics = []
    currency = yf_ticker.info.get("currency", "USD")  # Default to USD if unavailable
    for date in financials.columns[:limit]:
        row = financials[date]
        metrics.append(FinancialMetrics(
            ticker=ticker,
            report_period=date.strftime("%Y-%m-%d"),  # Use date as report_period
            period=period,  # "annual" or "quarterly" from function arg
            currency=currency,
            net_income=row.get("Net Income", None),
            revenue=row.get("Total Revenue", None),
            operating_income=row.get("Operating Income", None),
            earnings_per_share=row.get("Diluted EPS", None),
            market_cap=get_market_cap(ticker),  # Add market cap for completeness
        ))
    return metrics

def search_line_items(ticker: str, line_items: list[str], end_date: str, period: str = "annual", limit: int = 5) -> list[LineItem]:
    """Fetch financial line items from yfinance with broader mapping."""
    yf_ticker = Ticker(ticker)
    financials = yf_ticker.financials if period == "annual" else yf_ticker.quarterly_financials
    balance_sheet = yf_ticker.balance_sheet if period == "annual" else yf_ticker.quarterly_balance_sheet
    cashflow = yf_ticker.cashflow if period == "annual" else yf_ticker.quarterly_cashflow

    if financials.empty:
        return []

    items = []
    currency = yf_ticker.info.get("currency", "USD")  # Default to USD if unavailable
    for date in financials.columns[:limit]:
        fin_row = financials[date]
        bal_row = balance_sheet.get(date, pd.Series())
        cash_row = cashflow.get(date, pd.Series())
        item = LineItem(
            ticker=ticker,
            report_period=date.strftime("%Y-%m-%d"),  # Use date as report_period
            period=period,  # "annual" or "quarterly" from function arg
            currency=currency
        )
        for li in line_items:
            if li == "revenue": item.revenue = fin_row.get("Total Revenue", None)
            elif li == "earnings_per_share": item.earnings_per_share = fin_row.get("Diluted EPS", None)
            elif li == "net_income": item.net_income = fin_row.get("Net Income", None)
            elif li == "operating_income": item.operating_income = fin_row.get("Operating Income", None)
            elif li == "gross_margin": item.gross_margin = fin_row.get("Gross Profit", None)
            elif li == "operating_margin": item.operating_margin = fin_row.get("Operating Margin", None)
            elif li == "free_cash_flow": item.free_cash_flow = cash_row.get("Free Cash Flow", None)
            elif li == "capital_expenditure": item.capital_expenditure = cash_row.get("Capital Expenditures", None)
            elif li == "cash_and_equivalents": item.cash_and_equivalents = bal_row.get("Cash", None)
            elif li == "total_debt": item.total_debt = bal_row.get("Total Debt", None)
            elif li == "shareholders_equity": item.shareholders_equity = bal_row.get("Total Stockholder Equity", None)
            elif li == "outstanding_shares": item.outstanding_shares = yf_ticker.info.get("sharesOutstanding", None)
            elif li == "ebit": item.ebit = fin_row.get("EBIT", None)
            elif li == "ebitda": item.ebitda = fin_row.get("EBITDA", None)
        items.append(item)
    return items

def get_insider_trades(ticker: str, end_date: str, start_date: str | None = None, limit: int = 50) -> list:
    return []  # Still unimplemented

def get_company_news(ticker: str, end_date: str, start_date: str | None = None, limit: int = 50) -> list[NewsItem]:
    """Fetch news from Alpaca News API."""
    if not start_date:
        start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    request = NewsRequest(
        symbols=ticker,
        start=start_date,
        end=end_date,
        limit=limit
    )
    response = news_client.get_news(request)
    news = response.data['news']
    return [NewsItem(
        ticker=ticker,
        title=n.headline,
        summary=n.summary,
        date=n.created_at.strftime("%Y-%m-%d"),
        author=n.author,
        source=n.source,
        url=n.url,
        sentiment=None  # Add sentiment analysis if desired
    ) for n in news]

def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    return pd.DataFrame([p.model_dump() for p in prices])