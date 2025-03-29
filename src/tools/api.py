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
    FinancialMetrics,
    LineItem,
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
    adjusted_start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")  # Extended to 90 days
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
    if end_date == pd.Timestamp.now().strftime("%Y-%m-%d"):
        latest = get_latest_price(ticker)
        prices.append(Price(
            open=latest, close=latest, high=latest, low=latest,
            volume=0,
            time=end_date
        ))
    _cache.set_prices(ticker, [p.model_dump() for p in prices])
    print(f"Fetched {len(prices)} price records for {ticker}")
    return prices

def get_latest_price(ticker: str) -> float:
    request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
    quote = alpaca_client.get_stock_latest_quote(request)
    if ticker in quote and quote[ticker].bid_price and quote[ticker].ask_price:
        return (quote[ticker].bid_price + quote[ticker].ask_price) / 2
    return 100.0  # Fallback

def get_market_cap(ticker: str, end_date: str | None = None) -> float:
    if not end_date:
        latest_price = get_latest_price(ticker)
        shares = Ticker(ticker).info.get("sharesOutstanding", 0)
        return float(latest_price * shares if shares else 0.0)

    prices = get_prices(ticker, start_date=end_date, end_date=end_date)
    if not prices:
        print(f"No price data for {ticker} on {end_date} to calculate market cap")
        return 0.0
    latest_close = prices[-1].close
    shares = Ticker(ticker).info.get("sharesOutstanding", 0)
    return float(latest_close * shares if shares else 0.0)

def get_financial_metrics(ticker: str, end_date: str, period: str = "annual", limit: int = 10) -> list[FinancialMetrics]:
    try:
        yf_ticker = Ticker(ticker)
        financials = yf_ticker.financials if period == "annual" else yf_ticker.quarterly_financials
        balance_sheet = yf_ticker.balance_sheet if period == "annual" else yf_ticker.quarterly_balance_sheet

        if financials.empty or balance_sheet.empty:
            print(f"No financial data for {ticker} - trying alternative sources")
            # Add fallback data sources here if available
            return []

        metrics = []
        currency = yf_ticker.info.get("currency", "USD")
        shares = yf_ticker.info.get("sharesOutstanding", 0)
        for date in financials.columns[:limit]:
            fin_row = financials[date]
            bal_row = balance_sheet.get(date, pd.Series())
            metrics.append(FinancialMetrics(
                ticker=ticker,
                report_period=date.strftime("%Y-%m-%d"),
                period=period,
                currency=currency,
                net_income=fin_row.get("Net Income", None),
                revenue=fin_row.get("Total Revenue", None),
                operating_income=fin_row.get("Operating Income", None),
                earnings_per_share=fin_row.get("Diluted EPS", None),
                market_cap=get_market_cap(ticker, end_date),
                total_assets=bal_row.get("Total Assets", None),
                current_assets=bal_row.get("Total Current Assets", None),
                current_liabilities=bal_row.get("Total Current Liabilities", None),
                total_debt=bal_row.get("Total Debt", None),
                shareholders_equity=bal_row.get("Total Stockholder Equity", None),
                outstanding_shares=shares if shares else None,
            ))
        print(f"Fetched {len(metrics)} financial metrics for {ticker}")
        return metrics
    except Exception as e:
        print(f"Error fetching financial data for {ticker}: {str(e)}")
        return []

def search_line_items(ticker: str, line_items: list[str], end_date: str, period: str = "annual", limit: int = 10) -> list[LineItem]:
    yf_ticker = Ticker(ticker)
    financials = yf_ticker.financials if period == "annual" else yf_ticker.quarterly_financials
    balance_sheet = yf_ticker.balance_sheet if period == "annual" else yf_ticker.quarterly_balance_sheet
    cashflow = yf_ticker.cashflow if period == "annual" else yf_ticker.quarterly_cashflow

    if financials.empty or balance_sheet.empty:
        print(f"No financial data for {ticker} in search_line_items")
        return []

    items = []
    currency = yf_ticker.info.get("currency", "USD")
    shares = yf_ticker.info.get("sharesOutstanding", 0)
    for date in financials.columns[:limit]:
        fin_row = financials[date]
        bal_row = balance_sheet.get(date, pd.Series())
        cash_row = cashflow.get(date, pd.Series())
        item = LineItem(
            ticker=ticker,
            report_period=date.strftime("%Y-%m-%d"),
            period=period,
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
            elif li == "outstanding_shares": item.outstanding_shares = shares if shares else None
            elif li == "ebit": item.ebit = fin_row.get("EBIT", None)
            elif li == "ebitda": item.ebitda = fin_row.get("EBITDA", None)
            elif li == "current_assets": item.current_assets = bal_row.get("Total Current Assets", None)
            elif li == "current_liabilities": item.current_liabilities = bal_row.get("Total Current Liabilities", None)
            elif li == "total_assets": item.total_assets = bal_row.get("Total Assets", None)
            elif li == "total_liabilities": item.total_liabilities = bal_row.get("Total Liabilities", None)  # Added
        items.append(item)
    print(f"Fetched {len(items)} line items for {ticker}")
    return items

def get_insider_trades(ticker: str, end_date: str, start_date: str | None = None, limit: int = 50) -> list:
    try:
        yf_ticker = Ticker(ticker)
        insider_data = yf_ticker.insider_transactions

        if insider_data is None:
            print(f"No insider trade data structure available for {ticker}")
            return []

        # Handle different yfinance data formats
        if isinstance(insider_data, pd.Series):
            insider_data = insider_data.to_frame().T
        elif not isinstance(insider_data, pd.DataFrame):
            print(f"Unexpected insider data format for {ticker}")
            return []

        if insider_data.empty:
            print(f"No insider trade records for {ticker}")
            return []

        # Try to find transaction date column
        date_col = None
        possible_date_cols = ['Date', 'date', 'Transaction Date', 'trade_date', 'startDate']
        for col in possible_date_cols + list(insider_data.columns):
            if str(col).lower() in [c.lower() for c in insider_data.columns]:
                date_col = col
                break

        if not date_col:
            print(f"No identifiable date column in insider data for {ticker}. Available columns: {list(insider_data.columns)}")
            return []

        # Prepare date range
        if not start_date:
            start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")

        trades = []
        for _, row in insider_data.iterrows():
            try:
                # Handle various date formats
                date_value = row[date_col]
                if pd.isna(date_value):
                    continue

                trade_date = pd.to_datetime(date_value).strftime("%Y-%m-%d")
                if not trade_date:
                    continue

                if start_date <= trade_date <= end_date and len(trades) < limit:
                    trades.append({
                        "ticker": ticker,
                        "date": trade_date,
                        "insider_name": str(row.get('Insider Name', row.get('insider_name', 'Unknown'))),
                        "transaction_type": str(row.get('Transaction', row.get('transaction_type', 'Unknown'))),
                        "shares": int(row.get('Shares', row.get('shares', 0))),
                        "value": float(row.get('Value', row.get('value', 0.0)))
                    })
            except Exception as e:
                print(f"Skipping malformed insider trade row for {ticker}: {e}")
                continue

        return trades

    except Exception as e:
        print(f"Error processing insider trades for {ticker}: {str(e)}")
        return []

def get_dividend_history(ticker: str, end_date: str, start_date: str | None = None, limit: int = 10) -> list:
    yf_ticker = Ticker(ticker)
    dividends = yf_ticker.dividends
    if dividends.empty:
        print(f"No dividend data for {ticker}")
        return []

    if not start_date:
        start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=365 * 5)).strftime("%Y-%m-%d")

    div_history = []
    for date, amount in dividends.items():
        div_date = date.strftime("%Y-%m-%d")
        if start_date <= div_date <= end_date and len(div_history) < limit:
            div_history.append({
                "ticker": ticker,
                "date": div_date,
                "amount": float(amount)
            })
    return div_history

def get_company_news(ticker: str, end_date: str, start_date: str | None = None, limit: int = 50) -> list[NewsItem]:
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
        sentiment=None
    ) for n in news]

def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    return pd.DataFrame([p.model_dump() for p in prices])