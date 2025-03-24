import os
import sys
from dotenv import load_dotenv
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from yfinance import Ticker
from data.cache import get_cache
from data.models import (
    Price,
    PriceResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    LineItem,
    LineItemResponse,
)

# Global cache instance
load_dotenv()
_cache = get_cache()

# Alpaca API credentials (add to .env)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
alpaca_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from Alpaca."""
    if cached_data := _cache.get_prices(ticker):
        filtered_data = [Price(**price) for price in cached_data if start_date <= price["time"] <= end_date]
        if filtered_data:
            return filtered_data

    request_params = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date
    )
    bars = alpaca_client.get_stock_bars(request_params).df
    if bars.empty:
        return []

    # Ensure the index is reset and timestamp is accessible
    bars = bars.reset_index()
    # If 'timestamp' column exists, use it; otherwise, assume single-symbol index
    if "timestamp" in bars.columns:
        time_col = "timestamp"
    else:
        time_col = bars.index.name  # Fallback, though reset_index should handle this

    prices = [
        Price(
            open=row["open"],
            close=row["close"],
            high=row["high"],
            low=row["low"],
            volume=int(row["volume"]),
            time=pd.Timestamp(row[time_col]).strftime("%Y-%m-%d")
        ) for _, row in bars.iterrows()
    ]
    _cache.set_prices(ticker, [p.model_dump() for p in prices])
    return prices

def get_financial_metrics(ticker: str, end_date: str, period: str = "ttm", limit: int = 10) -> list[FinancialMetrics]:
    """Fetch financial metrics from yfinance."""
    if cached_data := _cache.get_financial_metrics(ticker):
        filtered_data = [FinancialMetrics(**metric) for metric in cached_data if metric["report_period"] <= end_date]
        filtered_data.sort(key=lambda x: x.report_period, reverse=True)
        if filtered_data:
            return filtered_data[:limit]

    yf_ticker = Ticker(ticker)
    info = yf_ticker.info
    financials = yf_ticker.financials
    balance_sheet = yf_ticker.balance_sheet
    cash_flow = yf_ticker.cashflow

    metrics_list = []
    for i in range(min(limit, len(financials.columns))):
        date = financials.columns[i].strftime("%Y-%m-%d")
        if date > end_date:
            continue
        metrics = FinancialMetrics(
            ticker=ticker,
            report_period=date,
            period="annual" if period != "ttm" else "ttm",
            currency="USD",
            market_cap=info.get("marketCap"),
            enterprise_value=info.get("enterpriseValue"),
            price_to_earnings_ratio=info.get("trailingPE"),
            price_to_book_ratio=info.get("priceToBook"),
            price_to_sales_ratio=info.get("priceToSalesTTM"),
            enterprise_value_to_ebitda_ratio=info.get("enterpriseToEbitda"),
            free_cash_flow_yield=info.get("freeCashflow") / info.get("marketCap") if info.get("freeCashflow") and info.get("marketCap") else None,
            peg_ratio=info.get("pegRatio"),
            gross_margin=info.get("grossMargins"),
            operating_margin=info.get("operatingMargins"),
            net_margin=info.get("profitMargins"),
            return_on_equity=info.get("returnOnEquity"),
            return_on_assets=info.get("returnOnAssets"),
            current_ratio=info.get("currentRatio"),
            quick_ratio=info.get("quickRatio"),
            debt_to_equity=info.get("debtToEquity"),
            earnings_per_share=info.get("trailingEps"),
            book_value_per_share=info.get("bookValue"),
            free_cash_flow=cash_flow.loc["Free Cash Flow", financials.columns[i]] if "Free Cash Flow" in cash_flow.index else None,
            total_debt=balance_sheet.loc["Total Debt", financials.columns[i]] if "Total Debt" in balance_sheet.index else None,
            shareholders_equity=balance_sheet.loc["Total Stockholder Equity", financials.columns[i]] if "Total Stockholder Equity" in balance_sheet.index else None,
            # Add more mappings as needed from yfinance
        )
        metrics_list.append(metrics)

    if not metrics_list:
        return []

    _cache.set_financial_metrics(ticker, [m.model_dump() for m in metrics_list])
    return metrics_list

def search_line_items(ticker: str, line_items: list[str], end_date: str, period: str = "ttm", limit: int = 10) -> list[LineItem]:
    """Fetch line items from yfinance."""
    yf_ticker = Ticker(ticker)
    financials = yf_ticker.financials
    balance_sheet = yf_ticker.balance_sheet
    cash_flow = yf_ticker.cashflow

    items = []
    for i in range(min(limit, len(financials.columns))):
        date = financials.columns[i].strftime("%Y-%m-%d")
        if date > end_date:
            continue
        item_data = {"ticker": ticker, "report_period": date, "period": period, "currency": "USD"}
        for line in line_items:
            if line in financials.index:
                item_data[line] = financials.loc[line, financials.columns[i]]
            elif line in balance_sheet.index:
                item_data[line] = balance_sheet.loc[line, financials.columns[i]]
            elif line in cash_flow.index:
                item_data[line] = cash_flow.loc[line, financials.columns[i]]
        items.append(LineItem(**item_data))
    return items[:limit]

# Simplified for now (not supported by yfinance or Alpaca directly)
def get_insider_trades(ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000) -> list:
    return []  # Placeholder; add a source like Finnhub if needed

def get_company_news(ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000) -> list:
    return []  # Placeholder; use Alpaca’s news API if desired

def get_market_cap(ticker: str, end_date: str) -> float | None:
    metrics = get_financial_metrics(ticker, end_date)
    return metrics[0].market_cap if metrics else None

def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df

def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)