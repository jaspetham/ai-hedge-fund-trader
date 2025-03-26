from pydantic import BaseModel, ConfigDict


class Price(BaseModel):
    open: float
    close: float
    high: float
    low: float
    volume: int
    time: str


class PriceResponse(BaseModel):
    ticker: str
    prices: list[Price]


class FinancialMetrics(BaseModel):
    ticker: str
    report_period: str  # e.g., "2024-09-30"
    period: str  # "annual" or "quarterly"
    currency: str  # e.g., "USD"
    market_cap: float | None = None
    enterprise_value: float | None = None
    price_to_earnings_ratio: float | None = None
    price_to_book_ratio: float | None = None
    price_to_sales_ratio: float | None = None
    enterprise_value_to_ebitda_ratio: float | None = None
    enterprise_value_to_revenue_ratio: float | None = None
    free_cash_flow_yield: float | None = None
    peg_ratio: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    return_on_equity: float | None = None
    return_on_assets: float | None = None
    return_on_invested_capital: float | None = None
    asset_turnover: float | None = None
    inventory_turnover: float | None = None
    receivables_turnover: float | None = None
    days_sales_outstanding: float | None = None
    operating_cycle: float | None = None
    working_capital_turnover: float | None = None
    current_ratio: float | None = None
    quick_ratio: float | None = None
    cash_ratio: float | None = None
    operating_cash_flow_ratio: float | None = None
    debt_to_equity: float | None = None
    debt_to_assets: float | None = None
    interest_coverage: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    book_value_growth: float | None = None
    earnings_per_share_growth: float | None = None
    free_cash_flow_growth: float | None = None
    operating_income_growth: float | None = None
    ebitda_growth: float | None = None
    payout_ratio: float | None = None
    earnings_per_share: float | None = None
    book_value_per_share: float | None = None
    free_cash_flow_per_share: float | None = None
    net_income: float | None = None  # Added for consistency with api.py
    revenue: float | None = None  # Added for consistency with api.py
    operating_income: float | None = None  # Added for consistency with api.py


class FinancialMetricsResponse(BaseModel):
    financial_metrics: list[FinancialMetrics]


class LineItem(BaseModel):
    ticker: str
    report_period: str  # e.g., "2024-09-30"
    period: str  # "annual" or "quarterly"
    currency: str  # e.g., "USD"
    revenue: float | None = None
    earnings_per_share: float | None = None
    net_income: float | None = None
    operating_income: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    free_cash_flow: float | None = None
    capital_expenditure: float | None = None
    cash_and_equivalents: float | None = None
    total_debt: float | None = None
    shareholders_equity: float | None = None
    outstanding_shares: float | None = None
    ebit: float | None = None
    ebitda: float | None = None

    model_config = ConfigDict(extra="allow")  # Allow extra fields for flexibility


class LineItemResponse(BaseModel):
    search_results: list[LineItem]


class InsiderTrade(BaseModel):
    ticker: str
    issuer: str | None = None
    name: str | None = None
    title: str | None = None
    is_board_director: bool | None = None
    transaction_date: str | None = None
    transaction_shares: float | None = None
    transaction_price_per_share: float | None = None
    transaction_value: float | None = None
    shares_owned_before_transaction: float | None = None
    shares_owned_after_transaction: float | None = None
    security_title: str | None = None
    filing_date: str


class InsiderTradeResponse(BaseModel):
    insider_trades: list[InsiderTrade]


class NewsItem(BaseModel):  # Renamed from CompanyNews to match api.py
    ticker: str | None = None  # Added for consistency
    title: str | None = None
    summary: str | None = None  # Replaced author/source/url with summary
    date: str | None = None
    author: str | None = None  # Optional fields from your version
    source: str | None = None
    url: str | None = None
    sentiment: str | None = None


class NewsItemResponse(BaseModel):  # Added to match pattern
    news: list[NewsItem]


class Position(BaseModel):
    cash: float = 0.0
    shares: int = 0
    ticker: str


class Portfolio(BaseModel):
    positions: dict[str, Position]  # ticker -> Position mapping
    total_cash: float = 0.0


class AnalystSignal(BaseModel):
    signal: str | None = None
    confidence: float | None = None
    reasoning: dict | str | None = None
    max_position_size: float | None = None  # For risk management signals


class TickerAnalysis(BaseModel):
    ticker: str
    analyst_signals: dict[str, AnalystSignal]  # agent_name -> signal mapping


class AgentStateData(BaseModel):
    tickers: list[str]
    portfolio: Portfolio
    start_date: str
    end_date: str
    ticker_analyses: dict[str, TickerAnalysis]  # ticker -> analysis mapping


class AgentStateMetadata(BaseModel):
    show_reasoning: bool = False
    model_config = ConfigDict(extra="allow")  # Updated syntax