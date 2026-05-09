export interface StockData {
  ticker: string;
  company_name: string;
  current_price: number;
  price_change: number;
  price_change_percent: number;
  volume: number;
  market_cap: number;
  pe_ratio: number;
  dividend_yield: number;
  fifty_two_week_high: number;
  fifty_two_week_low: number;
  beta: number;
  annualized_return_pct: number;
  sharpe_ratio: number;
  last_updated: string;
}

export interface EventData {
  event_type: string;
  timestamp: string;
  price: number;
  confidence_score: number;
  description: string;
  impact: 'positive' | 'negative' | 'neutral';
  volume?: number;
  technical_indicator?: string;
}

export interface CompanyInfo {
  ticker: string;
  company_name: string;
  sector: string;
  industry: string;
  description: string;
  headquarters: string;
  employees: number;
  website: string;
  founded: string;
}

export interface TechnicalIndicators {
  rsi: {
    value: number;
    signal: 'overbought' | 'oversold' | 'neutral';
    period: number;
  };
  macd: {
    macd_line: number;
    signal_line: number;
    histogram: number;
    signal: 'bullish' | 'bearish' | 'neutral';
  };
  moving_averages: {
    sma_20: number;
    sma_50: number;
    sma_200: number;
    ema_12: number;
    ema_26: number;
  };
  bollinger_bands: {
    upper_band: number;
    middle_band: number;
    lower_band: number;
    signal: 'overbought' | 'oversold' | 'neutral';
  };
  support_resistance: {
    support_levels: number[];
    resistance_levels: number[];
  };
}

export interface PriceHistory {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  adjusted_close: number;
}

export interface AnalysisResult {
  stock_data: StockData;
  company_info: CompanyInfo;
  technical_indicators: TechnicalIndicators;
  events: EventData[];
  price_history: PriceHistory[];
  analysis_summary: {
    overall_sentiment: 'bullish' | 'bearish' | 'neutral';
    recommendation: 'buy' | 'sell' | 'hold';
    confidence_score: number;
    key_insights: string[];
    risk_factors: string[];
  };
}

export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: string;
}

export interface APIError {
  message: string;
  status?: number;
  code?: string;
}

export interface LoadingState {
  isLoading: boolean;
  error: APIError | null;
}

export interface ComparisonData {
  primary_ticker: string;
  comparison_ticker: string;
  metrics: {
    price_performance: {
      primary_change: number;
      comparison_change: number;
      relative_performance: number;
    };
    valuation: {
      primary_pe: number;
      comparison_pe: number;
      primary_pb: number;
      comparison_pb: number;
    };
    financial_strength: {
      primary_debt_to_equity: number;
      comparison_debt_to_equity: number;
      primary_current_ratio: number;
      comparison_current_ratio: number;
    };
  };
  recommendation: {
    winner: string;
    reasons: string[];
    confidence: number;
  };
}

export interface WatchlistItem {
  ticker: string;
  company_name: string;
  current_price: number;
  price_change: number;
  price_change_percent: number;
  alert_conditions?: {
    price_above?: number;
    price_below?: number;
    volume_spike?: boolean;
  };
  added_date: string;
}

export interface UserPreferences {
  default_time_range: '1D' | '1W' | '1M' | '3M' | '6M' | '1Y' | '5Y';
  default_indicators: string[];
  alert_preferences: {
    email_notifications: boolean;
    push_notifications: boolean;
    price_movement_threshold: number;
  };
  theme: 'light' | 'dark' | 'auto';
}