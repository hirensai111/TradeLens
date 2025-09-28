// Types that match the actual backend JSON structure

// Price data from AAPL_price_data.json
export interface OHLCVData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PriceDataResponse {
  ticker: string;
  last_updated: string;
  data_period: {
    start_date: string;
    end_date: string;
    total_days: number;
  };
  ohlcv: OHLCVData[];
}

// Company data from AAPL_company.json
export interface CompanyInfo {
  name: string;
  sector: string;
  industry: string;
  country: string;
  website: string;
  description: string;
  employees: number | null;
  exchange: string;
  currency: string;
}

export interface BusinessMetrics {
  market_cap: number;
  enterprise_value: number;
  pe_ratio: number;
  price_to_book: number;
  dividend_yield: number;
  profit_margin: number;
  return_on_equity: number;
  return_on_assets: number;
}

export interface FinancialHighlights {
  revenue: number | null;
  gross_profit: number | null;
  operating_margin: number | null;
  ebitda: number;
  total_cash: number | null;
  total_debt: number | null;
  free_cash_flow: number | null;
  revenue_growth: number;
}

export interface CompanyDataResponse {
  ticker: string;
  last_updated: string;
  company_info: CompanyInfo;
  business_metrics: BusinessMetrics;
  financial_highlights: FinancialHighlights;
}

// Events data from AAPL_events.json
export interface EventData {
  date: string;
  type: string;
  description: string;
  sentiment: string;
  confidence: number;
  impact: string;
  price_change_pct: number;
  open_price: number;
  close_price: number;
  volume: number;
  news_count: number;
  sentiment_score: number;
  analysis_method: string;
  analysis_phase: string;
  key_phrases: string;
  sentiment_overall: string;
  sentiment_financial: string;
  sentiment_confidence: number;
  sentiment_relevance: number;
}

export interface EventsDataResponse {
  ticker: string;
  last_updated: string;
  events: EventData[];
}

// Summary data from AAPL_summary.json
export interface KeyMetrics {
  current_price: number;
  price_change_1d: number;
  price_change_1d_pct: number;
}

export interface TradingSignals {
  overall_signal: string;
  signal_strength: string;
  signals: any[];
}

export interface PerformanceMetrics {
  total_return_pct: number;
  annualized_return_pct: number;
  max_drawdown_pct: number;
  sharpe_ratio: number;
}

export interface RiskMetrics {
  volatility_annualized_pct: number;
  beta: number | null;
  var_95_pct: number;
  skewness: number;
  kurtosis: number;
}

export interface SummaryDataResponse {
  ticker: string;
  last_updated: string;
  current_price: number;
  key_metrics: KeyMetrics;
  trading_signals: TradingSignals;
  performance_metrics: PerformanceMetrics;
  risk_metrics: RiskMetrics;
  analysis_summary: any; // This seems to be empty in the actual data
}

// Combined analysis result
export interface BackendAnalysisResult {
  price_data: PriceDataResponse;
  company_data: CompanyDataResponse;
  events_data: EventsDataResponse;
  summary_data: SummaryDataResponse;
}

// Market Status interface
export interface MarketStatus {
  is_open: boolean;
  status: 'OPEN' | 'CLOSED' | 'CLOSED_WEEKEND' | 'CLOSED_HOLIDAY' | 'PREMARKET' | 'AFTERHOURS';
  status_message: string;
  current_time_et: string;
  current_time_local: string;
  market_open: string;
  market_close: string;
  premarket_open: string;
  afterhours_close: string;
  is_weekday: boolean;
  is_holiday: boolean;
  is_premarket: boolean;
  is_afterhours: boolean;
  next_market_open: string | null;
  day_name: string;
}

// API Response wrapper
export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: string;
  ticker?: string;
  analysis_time_seconds?: number;
  market_status?: MarketStatus;
}

export interface APIError {
  message: string;
  status?: number;
  code?: string;
}

// Simplified interfaces for frontend use
export interface StockData {
  ticker: string;
  current_price: number;
  price_change_1d?: number;
  price_change_1d_pct?: number;
  volume?: number;
  market_cap?: number;
  pe_ratio?: number;
  eps?: number;
  beta?: number;
  annualized_return_pct?: number;
  sharpe_ratio?: number;
  open?: number;
  high?: number;
  low?: number;
  previous_close?: number;
  week_52_high?: number;
  week_52_low?: number;
}

export interface TechnicalIndicators {
  // These might not be available in current backend, so making them optional
  rsi?: {
    value: number;
    signal: 'overbought' | 'oversold' | 'neutral';
    period: number;
  };
  macd?: {
    macd_line: number;
    signal_line: number;
    histogram: number;
    signal: 'bullish' | 'bearish' | 'neutral';
  };
  moving_averages?: {
    sma_20: number;
    sma_50: number;
    sma_200?: number;
  };
  sma_20?: number;
  ema_12?: number;
  macd_signal?: number;
}

export interface AnalysisResult {
  stock_data: StockData;
  company_info: CompanyInfo;
  technical_indicators: TechnicalIndicators;
  events: EventData[];
  price_history: OHLCVData[];
  market_status?: MarketStatus;
  recommendation?: {
    action: string;
    confidence: number;
    reasoning: string;
  };
  sentiment?: {
    overall: string;
    score: number;
  };
  analysis_summary?: {
    overall_sentiment: 'bullish' | 'bearish' | 'neutral';
    recommendation?: 'buy' | 'sell' | 'hold';
    confidence_score?: number;
    key_insights?: string[];
    risk_factors?: string[];
  };
}