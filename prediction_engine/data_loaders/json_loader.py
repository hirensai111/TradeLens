#!/usr/bin/env python3
"""
JSON Data Loader - Load stock analysis data from JSON files
Replaces Excel loader with JSON files from stock_analyzer cache
"""
import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class JSONDataLoader:
    """
    JSON data loader that processes data from stock_analyzer cache
    Provides the same interface as ExcelDataLoader for compatibility
    """

    def __init__(self, cache_dir: str = None):
        """
        Initialize JSON data loader.
        FIXED: Use unified cache directory from config if not specified.
        """
        # Import config to get unified cache path
        try:
            from core.config.config import config
            self.cache_dir = Path(cache_dir) if cache_dir else config.CACHE_DIR
        except ImportError:
            # Fallback to default if config not available
            self.cache_dir = Path(cache_dir) if cache_dir else Path("../cache")

        self.logger = self._setup_logging()

        # Data storage
        self.raw_data = None
        self.technical_data = None
        self.sentiment_data = None
        self.performance_data = None
        self.summary_data = None
        self.company_info = None
        self.calculated_volatility = 25.0

    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

    def load_all_data(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """Load ALL data for a ticker from JSON files"""
        self.logger.info(f"Loading data for {ticker} from JSON files")

        processed_data = {}

        try:
            # 1. Load company info
            company_file = self.cache_dir / f"company_info_{ticker}.json"
            if company_file.exists():
                with open(company_file, 'r') as f:
                    company_data = json.load(f)
                processed_data['company_info'] = pd.DataFrame([company_data])
                self.company_info = processed_data['company_info']
                self.logger.info(f"Loaded company info for {ticker}")

            # 2. Load historical price data
            hist_file = self.cache_dir / f"historical_{ticker}_5y.json"
            if hist_file.exists():
                with open(hist_file, 'r') as f:
                    hist_data = json.load(f)

                # Convert to DataFrame
                df = pd.DataFrame(hist_data)

                # Handle Date column - check if it exists
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                elif 'index' in df.columns:
                    # Handle case where index was reset and saved as 'index'
                    df['Date'] = pd.to_datetime(df['index'])
                    df.set_index('Date', inplace=True)
                    df = df.drop('index', axis=1, errors='ignore')
                else:
                    # No date column - generate dates assuming daily data going back 5 years
                    self.logger.warning(f"No Date column found in {ticker} data, generating dates")
                    end_date = pd.Timestamp.now()
                    dates = pd.date_range(end=end_date, periods=len(df), freq='B')  # Business days
                    df.index = dates
                    df.index.name = 'Date'

                df = df.sort_index()

                # This serves as both raw_data and basis for technical_data
                processed_data['raw_data'] = df
                self.raw_data = df

                # Calculate technical indicators
                processed_data['technical_data'] = self._calculate_technical_indicators(df)
                self.technical_data = processed_data['technical_data']

                # Calculate performance metrics
                processed_data['performance_data'] = self._calculate_performance_metrics(df)
                self.performance_data = processed_data['performance_data']

                # Calculate summary data
                processed_data['summary_data'] = self._calculate_summary_data(df)
                self.summary_data = processed_data['summary_data']

                self.logger.info(f"Loaded historical data with {len(df)} rows")

            self.logger.info(f"Successfully processed {len(processed_data)} data sources")
            return processed_data

        except Exception as e:
            self.logger.error(f"Failed to load JSON files: {e}")
            raise

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators from OHLCV data"""
        tech_df = df.copy()

        # Simple Moving Averages
        tech_df['SMA_20'] = tech_df['Close'].rolling(window=20).mean()
        tech_df['SMA_50'] = tech_df['Close'].rolling(window=50).mean()
        tech_df['SMA_200'] = tech_df['Close'].rolling(window=200).mean()

        # RSI
        tech_df['RSI'] = self._calculate_rsi(tech_df['Close'])

        # MACD
        macd_data = self._calculate_macd(tech_df['Close'])
        tech_df['MACD'] = macd_data['macd']
        tech_df['MACD_Signal'] = macd_data['signal']

        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(tech_df['Close'])
        tech_df['BB_Upper'] = bb_data['upper']
        tech_df['BB_Lower'] = bb_data['lower']

        # ATR (Average True Range)
        tech_df['ATR'] = self._calculate_atr(tech_df)

        # Calculate volatility
        if len(tech_df) > 10:
            daily_returns = tech_df['Close'].pct_change().dropna()
            if len(daily_returns) > 5:
                daily_volatility = daily_returns.std()
                annualized_volatility = daily_volatility * np.sqrt(252)
                volatility_percentage = annualized_volatility * 100

                # Sanity check
                if 5 <= volatility_percentage <= 100:
                    self.calculated_volatility = volatility_percentage
                    self.logger.info(f"Calculated volatility: {volatility_percentage:.2f}%")
                else:
                    self.calculated_volatility = 25.0
                    self.logger.warning(f"Volatility {volatility_percentage:.2f}% seems unrealistic, using default")

        return tech_df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal=9) -> Dict:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return {'macd': macd, 'signal': signal_line}

    def _calculate_bollinger_bands(self, prices: pd.Series, period=20, std_dev=2) -> Dict:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return {'upper': upper, 'lower': lower}

    def _calculate_atr(self, df: pd.DataFrame, period=14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        return atr

    def _calculate_performance_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate performance metrics from historical data"""
        performance_data = {}

        current_price = df['Close'].iloc[-1]

        # Calculate returns for different periods
        periods = {
            '1_day': 1,
            '1_week': 5,
            '1_month': 21,
            '3_month': 63,
            '6_month': 126,
            '1_year': 252
        }

        for period_name, days in periods.items():
            if len(df) > days:
                past_price = df['Close'].iloc[-days - 1]
                return_pct = ((current_price - past_price) / past_price) * 100
                performance_data[f'performance_{period_name}'] = return_pct

                # Create aliases for 1-month return
                if period_name == '1_month':
                    performance_data['performance_return_1_month'] = return_pct
                    performance_data['return_1_month'] = return_pct
                    performance_data['monthly_return'] = return_pct
                    performance_data['30d_return'] = return_pct

        # Calculate volatility
        if len(df) > 252:
            returns = df['Close'].pct_change().dropna()
            annual_volatility = returns.std() * np.sqrt(252) * 100
            performance_data['annual_volatility'] = annual_volatility

            # Sharpe ratio (assuming risk-free rate of 4%)
            risk_free_rate = 4.0
            if 'performance_1_year' in performance_data:
                sharpe_ratio = (performance_data['performance_1_year'] - risk_free_rate) / annual_volatility
                performance_data['sharpe_ratio'] = sharpe_ratio

            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            performance_data['maximum_drawdown'] = max_drawdown

        self.logger.info(f"Calculated {len(performance_data)} performance metrics")
        return pd.DataFrame([performance_data]) if performance_data else pd.DataFrame()

    def _calculate_summary_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate summary statistics"""
        summary = {}

        if len(df) > 0:
            summary['current_price'] = df['Close'].iloc[-1]
            summary['high_52w'] = df['High'].tail(252).max() if len(df) >= 252 else df['High'].max()
            summary['low_52w'] = df['Low'].tail(252).min() if len(df) >= 252 else df['Low'].min()
            summary['avg_volume'] = df['Volume'].mean()
            summary['total_records'] = len(df)
            summary['date_range_start'] = df.index.min().strftime('%Y-%m-%d')
            summary['date_range_end'] = df.index.max().strftime('%Y-%m-%d')

        return pd.DataFrame([summary]) if summary else pd.DataFrame()

    def get_enhanced_features_for_date(self, target_date: datetime, lookback_days: int = 30) -> Dict:
        """Enhanced feature extraction - compatible with prediction engine"""
        target_date = pd.to_datetime(target_date)

        # Handle timezone compatibility
        if self.raw_data is not None and not self.raw_data.empty:
            try:
                if hasattr(self.raw_data.index, 'tz'):
                    if self.raw_data.index.tz is not None and target_date.tz is None:
                        target_date = target_date.tz_localize('UTC').tz_convert(self.raw_data.index.tz)
                    elif self.raw_data.index.tz is None and target_date.tz is not None:
                        target_date = target_date.tz_localize(None)
            except Exception:
                # If timezone handling fails, just remove timezone from target_date
                if hasattr(target_date, 'tz') and target_date.tz is not None:
                    target_date = target_date.replace(tzinfo=None)

        features = {}

        # 1. Raw data features - use latest available data
        if self.raw_data is not None and not self.raw_data.empty:
            try:
                # Just use the last available row
                latest_data = self.raw_data.iloc[-1]

                features.update({
                    'close': float(latest_data.get('Close', 0)),
                    'volume': float(latest_data.get('Volume', 0)),
                    'open': float(latest_data.get('Open', 0)),
                    'high': float(latest_data.get('High', 0)),
                    'low': float(latest_data.get('Low', 0))
                })
            except Exception as e:
                self.logger.warning(f"Error extracting raw data features: {e}")

        # 2. Technical indicators
        if self.technical_data is not None and not self.technical_data.empty:
            try:
                # Just use the last available row
                latest_tech = self.technical_data.iloc[-1]

                tech_indicators = ['SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'MACD',
                                'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR']

                for indicator in tech_indicators:
                    if indicator in latest_tech.index and pd.notna(latest_tech[indicator]):
                        features[f'tech_{indicator.lower()}'] = float(latest_tech[indicator])
            except Exception as e:
                self.logger.warning(f"Error extracting technical features: {e}")

        # 3. Company fundamentals
        if self.company_info is not None and not self.company_info.empty:
            try:
                company_row = self.company_info.iloc[0]
                fundamental_fields = ['marketCap', 'trailingPE', 'forwardPE', 'dividendYield',
                                    'beta', 'priceToBook', 'profitMargins', 'debtToEquity']

                for field in fundamental_fields:
                    if field in company_row.index and pd.notna(company_row[field]):
                        value = company_row[field]
                        if isinstance(value, (int, float)):
                            features[f'fundamental_{field.lower()}'] = float(value)
            except Exception as e:
                self.logger.warning(f"Error extracting company features: {e}")

        # 4. Performance features
        if self.performance_data is not None and not self.performance_data.empty:
            try:
                perf_row = self.performance_data.iloc[0]
                for col in self.performance_data.columns:
                    if pd.notna(perf_row[col]) and isinstance(perf_row[col], (int, float)):
                        features[col] = float(perf_row[col])
            except Exception as e:
                self.logger.warning(f"Error extracting performance features: {e}")

        # 5. Summary features
        if self.summary_data is not None and not self.summary_data.empty:
            try:
                summary_row = self.summary_data.iloc[0]
                for col in self.summary_data.columns:
                    if pd.notna(summary_row[col]) and isinstance(summary_row[col], (int, float)):
                        features[f'summary_{col}'] = float(summary_row[col])
            except Exception as e:
                self.logger.warning(f"Error extracting summary features: {e}")

        # 6. Time-based features
        features.update({
            'day_of_week': target_date.dayofweek,
            'month': target_date.month,
            'quarter': (target_date.month - 1) // 3 + 1,
            'year': target_date.year,
            'day_of_month': target_date.day
        })

        self.logger.info(f"Extracted {len(features)} features for date {target_date.date()}")
        return features

    def get_proper_volatility(self) -> float:
        """Get volatility with fallback"""
        if hasattr(self, 'calculated_volatility'):
            if 5 <= self.calculated_volatility <= 100:
                return self.calculated_volatility

        # Try from performance data
        if self.performance_data is not None and not self.performance_data.empty:
            perf_row = self.performance_data.iloc[0]
            if 'annual_volatility' in perf_row.index and pd.notna(perf_row['annual_volatility']):
                vol = float(perf_row['annual_volatility'])
                if 5 <= vol <= 100:
                    return vol

        self.logger.warning("Using default volatility of 25%")
        return 25.0

    def get_ticker_symbol(self) -> str:
        """Extract ticker symbol"""
        if self.company_info is not None and not self.company_info.empty:
            if 'symbol' in self.company_info.columns:
                return str(self.company_info.iloc[0]['symbol'])
        return "UNKNOWN"


# Test usage
if __name__ == "__main__":
    # FIXED: Use unified cache from config (no need to specify cache_dir)
    loader = JSONDataLoader()
    all_data = loader.load_all_data("AAPL")

    print(f"\nLoaded data sources: {list(all_data.keys())}")
    print(f"Ticker: {loader.get_ticker_symbol()}")
    print(f"Volatility: {loader.get_proper_volatility():.2f}%")

    # Test feature extraction
    features = loader.get_enhanced_features_for_date(datetime.now())
    print(f"\nExtracted {len(features)} features")
    print(f"Sample features: {list(features.keys())[:10]}")
