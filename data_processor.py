"""
Core data processing module for Stock Analyzer.
Handles stock data collection, technical indicator calculations, and data quality validation.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional, List
import warnings
import time

from config import config
from utils import get_logger, ProgressTracker, retry_on_failure, cache_manager, data_formatter
from validators import data_quality_validator, ValidationError
from alpha_vantage_client import alpha_vantage_client

# Suppress yfinance warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class StockDataProcessor:
    """Main class for processing stock data and calculating indicators."""
    
    def __init__(self):
        self.logger = get_logger("StockDataProcessor")
        self.cache = cache_manager
        self.alpha_vantage = alpha_vantage_client
        
    def process_stock(self, ticker: str) -> Dict[str, Any]:
        """
        Complete stock processing pipeline.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing all processed data and analysis
        """
        self.logger.log_function_call("process_stock", ticker)
        
        # Initialize progress tracker
        progress = ProgressTracker(6, f"Processing {ticker}")
        
        try:
            # Step 1: Fetch historical data
            progress.update("Fetching historical data")
            raw_data = self._fetch_historical_data(ticker)
            
            # Step 2: Validate data quality
            progress.update("Validating data quality")
            quality_report = self._validate_data_quality(raw_data, ticker)
            
            # Step 3: Calculate technical indicators
            progress.update("Calculating technical indicators")
            technical_data = self._calculate_technical_indicators(raw_data)
            
            # Step 4: Calculate summary statistics
            progress.update("Calculating summary statistics")
            summary_stats = self._calculate_summary_statistics(technical_data, ticker)
            
            # Step 5: Get company information
            progress.update("Fetching company information")
            company_info = self._get_company_info(ticker)
            
            # Step 6: Compile final results
            progress.update("Compiling results")
            result = self._compile_results(
                ticker=ticker,
                raw_data=raw_data,
                technical_data=technical_data,
                summary_stats=summary_stats,
                company_info=company_info,
                quality_report=quality_report
            )
            
            progress.finish(f"Successfully processed {ticker}")
            self.logger.log_success(f"Stock processing completed for {ticker}")
            
            return result
            
        except Exception as e:
            self.logger.log_failure(f"Failed to process stock {ticker}: {str(e)}")
            raise ValidationError(f"Stock processing failed: {str(e)}")
    
    def _fetch_historical_data(self, ticker: str) -> pd.DataFrame:
        """
        Fetch historical stock data using multiple sources.
        Tries Yahoo Finance first, then Alpha Vantage as fallback.
        """
        cache_key = f"historical_{ticker}_{config.DATA_PERIOD_YEARS}y"
        
        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data:
            self.logger.debug(f"Using cached data for {ticker}")
            # Convert cached data back to DataFrame with proper index
            df = pd.DataFrame(cached_data)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            return df
        
        # Try data sources in order of priority
        data_sources = []
        
        # Always try Yahoo Finance first
        data_sources.append(('yfinance', self._fetch_from_yfinance))
        
        # Add Alpha Vantage if configured
        if self.alpha_vantage.is_configured():
            data_sources.append(('alpha_vantage', self._fetch_from_alpha_vantage))
        
        # Try each data source
        for source_name, fetch_method in data_sources:
            try:
                self.logger.info(f"Attempting to fetch data from {source_name} for {ticker}")
                hist_data = fetch_method(ticker)
                
                if hist_data is not None and not hist_data.empty:
                    self.logger.info(f"Successfully fetched data from {source_name} for {ticker}")
                    
                    # Cache the data
                    cache_data = hist_data.copy()
                    cache_data.reset_index(inplace=True)
                    self.cache.set(cache_key, cache_data.to_dict('records'))
                    
                    return hist_data
                    
            except Exception as e:
                self.logger.warning(f"Failed to fetch from {source_name}: {str(e)}")
                continue
        
        # If all sources fail, generate sample data for development
        if config.USE_SAMPLE_DATA or config.DEBUG_MODE:
            self.logger.warning(f"All data sources failed for {ticker}, generating sample data")
            return self._generate_sample_data(ticker)
        
        raise ValidationError(f"Failed to fetch data for {ticker} from any source")
    
    @retry_on_failure(max_retries=3, delay=2.0)
    def _fetch_from_yfinance(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance."""
        try:
            stock = yf.Ticker(ticker)
            
            # Try with period parameter
            hist_data = stock.history(period="5y", interval="1d")
            
            if hist_data.empty:
                # Try with date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365 * config.DATA_PERIOD_YEARS)
                hist_data = stock.history(start=start_date, end=end_date, interval="1d")
            
            if hist_data.empty:
                return None
            
            # Clean the data
            hist_data = self._clean_historical_data(hist_data)
            
            self.logger.debug(f"Retrieved {len(hist_data)} days of data from Yahoo Finance")
            return hist_data
            
        except Exception as e:
            self.logger.error(f"Yahoo Finance error: {str(e)}")
            raise
    
    def _fetch_from_alpha_vantage(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage."""
        try:
            # Use Alpha Vantage client
            hist_data = self.alpha_vantage.fetch_daily_data(ticker, outputsize="full")
            
            if hist_data.empty:
                return None
            
            # Clean the data
            hist_data = self._clean_historical_data(hist_data)
            
            self.logger.debug(f"Retrieved {len(hist_data)} days of data from Alpha Vantage")
            return hist_data
            
        except Exception as e:
            self.logger.error(f"Alpha Vantage error: {str(e)}")
            raise
    
    def _generate_sample_data(self, ticker: str) -> pd.DataFrame:
        """Generate sample data for development when all data sources fail."""
        self.logger.info(f"Generating sample data for {ticker}")
        
        # Generate 5 years of sample data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 5)
        
        # Create date range (business days only)
        date_range = pd.bdate_range(start=start_date, end=end_date)
        
        # Generate realistic stock data
        np.random.seed(42)  # For reproducible results
        
        # More realistic starting prices (as of 2020)
        price_map = {
            'AAPL': 91.0,    # Split-adjusted price from 2020
            'MSFT': 200.0,   # Microsoft in 2020
            'GOOGL': 1500.0, # Google in 2020
            'AMZN': 3200.0,  # Amazon in 2020
            'TSLA': 500.0,   # Tesla in 2020 (pre-split)
            'META': 220.0,   # Facebook/Meta in 2020
            'NVDA': 120.0,   # NVIDIA in 2020 (pre-split)
            'JPM': 100.0,    # JP Morgan in 2020
            'V': 190.0,      # Visa in 2020
            'JNJ': 145.0     # Johnson & Johnson in 2020
        }
        
        base_price = price_map.get(ticker, 100.0)
        
        # Different volatility for different stocks
        volatility_map = {
            'AAPL': 0.018,   # ~1.8% daily volatility
            'MSFT': 0.016,   # ~1.6% daily volatility
            'GOOGL': 0.019,  # ~1.9% daily volatility
            'AMZN': 0.022,   # ~2.2% daily volatility
            'TSLA': 0.035,   # ~3.5% daily volatility (more volatile)
            'META': 0.025,   # ~2.5% daily volatility
            'NVDA': 0.028,   # ~2.8% daily volatility
            'JPM': 0.020,    # ~2.0% daily volatility
            'V': 0.015,      # ~1.5% daily volatility
            'JNJ': 0.012     # ~1.2% daily volatility (less volatile)
        }
        
        stock_volatility = volatility_map.get(ticker, 0.02)
        
        # Realistic annual return expectations
        annual_return_map = {
            'AAPL': 0.25,   # ~25% annual return
            'MSFT': 0.22,   # ~22% annual return
            'GOOGL': 0.18,  # ~18% annual return
            'AMZN': 0.15,   # ~15% annual return
            'TSLA': 0.35,   # ~35% annual return (high growth)
            'META': 0.20,   # ~20% annual return
            'NVDA': 0.40,   # ~40% annual return (AI boom)
            'JPM': 0.12,    # ~12% annual return
            'V': 0.15,      # ~15% annual return
            'JNJ': 0.08     # ~8% annual return (stable dividend stock)
        }
        
        annual_return = annual_return_map.get(ticker, 0.10)
        daily_drift = annual_return / 252  # Convert to daily
        
        # Generate price movements with drift and volatility
        returns = np.random.normal(daily_drift, stock_volatility, len(date_range))
        
        # Calculate prices
        prices = [base_price]
        for return_rate in returns[1:]:
            new_price = prices[-1] * (1 + return_rate)
            prices.append(max(new_price, 1.0))  # Ensure price doesn't go below $1
        
        # Expected current prices (July 2024) - for validation
        expected_current_prices = {
            'AAPL': 230.0,
            'MSFT': 450.0,
            'GOOGL': 180.0,  # Post-split
            'AMZN': 190.0,   # Post-split
            'TSLA': 250.0,
            'META': 500.0,
            'NVDA': 125.0,   # Post-split
            'JPM': 200.0,
            'V': 280.0,
            'JNJ': 150.0
        }
        
        # Adjust final price to be more realistic if we have an expected price
        if ticker in expected_current_prices:
            target_price = expected_current_prices[ticker]
            current_price = prices[-1]
            adjustment_factor = target_price / current_price
            
            # Gradually adjust prices over the last year
            adjustment_days = min(252, len(prices))
            for i in range(adjustment_days):
                weight = (i + 1) / adjustment_days
                idx = len(prices) - adjustment_days + i
                prices[idx] = prices[idx] * (1 + (adjustment_factor - 1) * weight)
        
        # Create OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(date_range, prices)):
            # Generate realistic OHLC from close price
            daily_volatility = stock_volatility * 0.5  # Intraday volatility is typically lower
            
            # Generate high and low
            high = close * (1 + np.random.uniform(0, daily_volatility))
            low = close * (1 - np.random.uniform(0, daily_volatility))
            
            # Generate open (influenced by previous close)
            if i > 0:
                gap = np.random.normal(0, stock_volatility * 0.3)  # Overnight gaps
                open_price = prices[i-1] * (1 + gap)
            else:
                open_price = close * (1 + np.random.uniform(-daily_volatility, daily_volatility))
            
            # Ensure OHLC logic
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Generate volume (with some patterns)
            base_volume = {
                'AAPL': 80000000,
                'MSFT': 25000000,
                'GOOGL': 20000000,
                'AMZN': 40000000,
                'TSLA': 100000000,
                'META': 15000000,
                'NVDA': 300000000,
                'JPM': 10000000,
                'V': 7000000,
                'JNJ': 6000000
            }.get(ticker, 10000000)
            
            # Add some randomness to volume
            volume = int(base_volume * np.random.uniform(0.7, 1.3))
            
            data.append({
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close, 2),
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=date_range)
        df.index.name = 'Date'
        
        return df
    
    def _clean_historical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize historical data."""
        try:
            # Remove any rows with all NaN values
            df = df.dropna(how='all')
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Ensure proper data types
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
            
            # Remove rows with invalid prices (negative or zero)
            df = df[(df['Close'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Open'] > 0)]
            
            # Sort by date
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to clean historical data: {str(e)}")
            raise
    
    def _validate_data_quality(self, df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """Validate data quality using our validator."""
        try:
            is_valid, warnings, quality_report = data_quality_validator.validate_dataframe(df, ticker)
            
            if not is_valid:
                critical_issues = [w for w in warnings if 'insufficient' in w.lower() or 'missing required' in w.lower()]
                if critical_issues:
                    raise ValidationError(f"Critical data quality issues: {critical_issues}")
            
            # Log warnings if any
            if warnings:
                for warning in warnings:
                    self.logger.warning(f"Data quality warning for {ticker}: {warning}")
            
            return quality_report
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            raise
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for the stock data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with original data plus technical indicators
        """
        try:
            # Create a copy to avoid modifying original
            data = df.copy()
            
            # Moving Averages
            data['SMA_20'] = self._calculate_sma(data['Close'], 20)
            data['SMA_50'] = self._calculate_sma(data['Close'], 50)
            data['SMA_200'] = self._calculate_sma(data['Close'], 200)
            data['EMA_12'] = self._calculate_ema(data['Close'], 12)
            data['EMA_26'] = self._calculate_ema(data['Close'], 26)
            
            # MACD
            data['MACD'], data['MACD_Signal'], data['MACD_Histogram'] = self._calculate_macd(data['Close'])
            
            # RSI
            data['RSI'] = self._calculate_rsi(data['Close'])
            
            # Bollinger Bands
            data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = self._calculate_bollinger_bands(data['Close'])
            
            # Volume indicators
            data['Volume_SMA'] = self._calculate_sma(data['Volume'], 20)
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
            
            # Price patterns
            data['Daily_Return'] = data['Close'].pct_change()
            data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
            
            # Support and Resistance levels
            data['Support'], data['Resistance'] = self._calculate_support_resistance(data)
            
            # ATR (Average True Range)
            data['ATR'] = self._calculate_atr(data)
            
            # Stochastic Oscillator
            data['Stoch_K'], data['Stoch_D'] = self._calculate_stochastic(data)
            
            # OBV (On-Balance Volume)
            data['OBV'] = self._calculate_obv(data)
            
            # Drop any rows with NaN in critical columns
            critical_columns = ['Close', 'SMA_20', 'RSI']
            data = data.dropna(subset=critical_columns)
            
            self.logger.debug(f"Calculated {len(data.columns) - len(df.columns)} technical indicators")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to calculate technical indicators: {str(e)}")
            raise
    
    def _calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return series.rolling(window=period, min_periods=1).mean()
    
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period, adjust=False, min_periods=1).mean()
    
    def _calculate_macd(self, close_prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_12 = self._calculate_ema(close_prices, 12)
        ema_26 = self._calculate_ema(close_prices, 26)
        
        macd_line = ema_12 - ema_26
        signal_line = self._calculate_ema(macd_line, 9)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_rsi(self, close_prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_bands(self, close_prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle_band = self._calculate_sma(close_prices, period)
        std = close_prices.rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    def _calculate_support_resistance(self, df: pd.DataFrame, lookback: int = 120) -> Tuple[pd.Series, pd.Series]:
        """Calculate dynamic support and resistance levels."""
        support = df['Low'].rolling(window=lookback, min_periods=20).min()
        resistance = df['High'].rolling(window=lookback, min_periods=20).max()
        
        return support, resistance
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = df['Low'].rolling(window=period).min()
        highest_high = df['High'].rolling(window=period).max()
        
        k_percent = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
        k_percent_smooth = k_percent.rolling(window=smooth_k).mean()
        d_percent = k_percent_smooth.rolling(window=smooth_d).mean()
        
        return k_percent_smooth, d_percent
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        return pd.Series(obv, index=df.index)
    
    def _calculate_summary_statistics(self, df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics."""
        try:
            # Basic price statistics
            current_price = df['Close'].iloc[-1]
            price_change_1d = df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) > 1 else 0
            price_change_1d_pct = (price_change_1d / df['Close'].iloc[-2] * 100) if len(df) > 1 else 0
            
            # Period returns
            returns = {
                '1_day': self._calculate_period_return(df, 1),
                '1_week': self._calculate_period_return(df, 5),
                '1_month': self._calculate_period_return(df, 21),
                '3_months': self._calculate_period_return(df, 63),
                '6_months': self._calculate_period_return(df, 126),
                '1_year': self._calculate_period_return(df, 252),
                '3_years': self._calculate_period_return(df, 756),
                '5_years': self._calculate_period_return(df, 1260),
            }
            
            # Risk metrics
            daily_returns = df['Daily_Return'].dropna()
            volatility_annual = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
            max_drawdown = self._calculate_max_drawdown(df['Close'])
            
            # Technical indicator signals
            latest_data = df.iloc[-1]
            signals = self._generate_trading_signals(df)
            
            # Volume analysis
            avg_volume = df['Volume'].mean()
            volume_trend = 'High' if latest_data['Volume'] > avg_volume * 1.5 else 'Normal' if latest_data['Volume'] > avg_volume * 0.5 else 'Low'
            
            # Price levels
            all_time_high = df['Close'].max()
            all_time_low = df['Close'].min()
            fifty_two_week_high = df['Close'].iloc[-252:].max() if len(df) >= 252 else all_time_high
            fifty_two_week_low = df['Close'].iloc[-252:].min() if len(df) >= 252 else all_time_low
            
            summary = {
                'ticker': ticker,
                'current_price': round(current_price, 2),
                'price_change_1d': round(price_change_1d, 2),
                'price_change_1d_pct': round(price_change_1d_pct, 2),
                'volume': int(latest_data['Volume']),
                'avg_volume': int(avg_volume),
                'volume_trend': volume_trend,
                'returns': returns,
                'volatility_annual': round(volatility_annual, 4),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown': round(max_drawdown, 4),
                'all_time_high': round(all_time_high, 2),
                'all_time_low': round(all_time_low, 2),
                '52_week_high': round(fifty_two_week_high, 2),
                '52_week_low': round(fifty_two_week_low, 2),
                'technical_indicators': {
                    'rsi': round(latest_data['RSI'], 2) if not pd.isna(latest_data['RSI']) else None,
                    'macd': round(latest_data['MACD'], 4) if not pd.isna(latest_data['MACD']) else None,
                    'macd_signal': round(latest_data['MACD_Signal'], 4) if not pd.isna(latest_data['MACD_Signal']) else None,
                    'sma_20': round(latest_data['SMA_20'], 2) if not pd.isna(latest_data['SMA_20']) else None,
                    'sma_50': round(latest_data['SMA_50'], 2) if not pd.isna(latest_data['SMA_50']) else None,
                    'sma_200': round(latest_data['SMA_200'], 2) if not pd.isna(latest_data['SMA_200']) else None,
                    'bb_upper': round(latest_data['BB_Upper'], 2) if not pd.isna(latest_data['BB_Upper']) else None,
                    'bb_lower': round(latest_data['BB_Lower'], 2) if not pd.isna(latest_data['BB_Lower']) else None,
                    'atr': round(latest_data['ATR'], 2) if not pd.isna(latest_data['ATR']) else None,
                    'stoch_k': round(latest_data['Stoch_K'], 2) if not pd.isna(latest_data['Stoch_K']) else None,
                    'stoch_d': round(latest_data['Stoch_D'], 2) if not pd.isna(latest_data['Stoch_D']) else None,
                },
                'signals': signals,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_points': len(df),
                'date_range': {
                    'start': df.index[0].strftime('%Y-%m-%d'),
                    'end': df.index[-1].strftime('%Y-%m-%d')
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to calculate summary statistics: {str(e)}")
            raise
    
    def _calculate_period_return(self, df: pd.DataFrame, days: int) -> Optional[float]:
        """Calculate return over specified period."""
        if len(df) < days + 1:
            return None
        
        start_price = df['Close'].iloc[-days-1]
        end_price = df['Close'].iloc[-1]
        
        if start_price == 0:
            return None
            
        return round((end_price - start_price) / start_price * 100, 2)
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        
        if excess_returns.std() == 0:
            return 0.0
            
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return abs(drawdown.min())
    
    def _generate_trading_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on technical indicators."""
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        signals = {
            'overall': 'Neutral',
            'trend': 'Neutral',
            'momentum': 'Neutral',
            'volume': 'Normal',
            'signals': []
        }
        
        # Trend signals
        if latest['Close'] > latest['SMA_200']:
            signals['trend'] = 'Bullish'
            signals['signals'].append('Price above 200-day SMA (Bullish)')
        elif latest['Close'] < latest['SMA_200']:
            signals['trend'] = 'Bearish'
            signals['signals'].append('Price below 200-day SMA (Bearish)')
        
        # Moving average crossovers
        if latest['SMA_20'] > latest['SMA_50'] and prev['SMA_20'] <= prev['SMA_50']:
            signals['signals'].append('Golden Cross: 20-day SMA crossed above 50-day SMA (Bullish)')
        elif latest['SMA_20'] < latest['SMA_50'] and prev['SMA_20'] >= prev['SMA_50']:
            signals['signals'].append('Death Cross: 20-day SMA crossed below 50-day SMA (Bearish)')
        
        # RSI signals
        if not pd.isna(latest['RSI']):
            if latest['RSI'] > 70:
                signals['momentum'] = 'Overbought'
                signals['signals'].append(f'RSI at {latest["RSI"]:.1f} - Overbought condition')
            elif latest['RSI'] < 30:
                signals['momentum'] = 'Oversold'
                signals['signals'].append(f'RSI at {latest["RSI"]:.1f} - Oversold condition')
        
        # MACD signals
        if not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_Signal']):
            if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
                signals['signals'].append('MACD crossed above signal line (Bullish)')
            elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
                signals['signals'].append('MACD crossed below signal line (Bearish)')
        
        # Bollinger Bands signals
        if not pd.isna(latest['BB_Upper']) and not pd.isna(latest['BB_Lower']):
            if latest['Close'] > latest['BB_Upper']:
                signals['signals'].append('Price above upper Bollinger Band (Potential reversal)')
            elif latest['Close'] < latest['BB_Lower']:
                signals['signals'].append('Price below lower Bollinger Band (Potential bounce)')
        
        # Volume signals
        if latest['Volume_Ratio'] > 1.5:
            signals['volume'] = 'High'
            signals['signals'].append('Volume 50% above average')
        elif latest['Volume_Ratio'] < 0.5:
            signals['volume'] = 'Low'
            signals['signals'].append('Volume 50% below average')
        
        # Determine overall signal
        bullish_count = sum(1 for s in signals['signals'] if 'Bullish' in s or 'bounce' in s)
        bearish_count = sum(1 for s in signals['signals'] if 'Bearish' in s or 'reversal' in s)
        
        if bullish_count > bearish_count + 1:
            signals['overall'] = 'Bullish'
        elif bearish_count > bullish_count + 1:
            signals['overall'] = 'Bearish'
        else:
            signals['overall'] = 'Neutral'
        
        return signals
    
    def _get_company_info(self, ticker: str) -> Dict[str, Any]:
        """Get company information from multiple sources."""
        cache_key = f"company_info_{ticker}"
        
        # Check cache first
        cached_info = self.cache.get(cache_key)
        if cached_info:
            self.logger.debug(f"Using cached company info for {ticker}")
            return cached_info
        
        company_info = {}
        
        # Try Yahoo Finance first
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if info and 'symbol' in info:
                company_info = {
                    'symbol': info.get('symbol', ticker),
                    'shortName': info.get('shortName', f'{ticker} Corporation'),
                    'longName': info.get('longName', info.get('shortName', f'{ticker} Corporation')),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'website': info.get('website', ''),
                    'description': info.get('longBusinessSummary', '')[:500] + '...' if info.get('longBusinessSummary') else '',
                    'employees': info.get('fullTimeEmployees', 0),
                    'country': info.get('country', 'Unknown'),
                    'currency': info.get('currency', 'USD'),
                    'exchange': info.get('exchange', 'Unknown'),
                    'marketCap': info.get('marketCap', 0),
                    'enterpriseValue': info.get('enterpriseValue', 0),
                    'trailingPE': info.get('trailingPE', 0),
                    'forwardPE': info.get('forwardPE', 0),
                    'dividendYield': info.get('dividendYield', 0),
                    'beta': info.get('beta', 0),
                    'priceToBook': info.get('priceToBook', 0),
                    'profitMargins': info.get('profitMargins', 0),
                    'grossMargins': info.get('grossMargins', 0),
                    'revenueGrowth': info.get('revenueGrowth', 0),
                    'earningsGrowth': info.get('earningsGrowth', 0),
                    'currentRatio': info.get('currentRatio', 0),
                    'debtToEquity': info.get('debtToEquity', 0),
                    'returnOnEquity': info.get('returnOnEquity', 0),
                    'returnOnAssets': info.get('returnOnAssets', 0),
                    'revenue': info.get('totalRevenue', 0),
                    'grossProfit': info.get('grossProfit', 0),
                    'ebitda': info.get('ebitda', 0),
                    'netIncome': info.get('netIncomeToCommon', 0),
                    'sharesOutstanding': info.get('sharesOutstanding', 0),
                    'floatShares': info.get('floatShares', 0),
                }
                
                self.logger.debug(f"Retrieved company info from Yahoo Finance for {ticker}")
        
        except Exception as e:
            self.logger.warning(f"Failed to get company info from Yahoo Finance: {str(e)}")
        
        # Try Alpha Vantage as fallback
        if not company_info and self.alpha_vantage.is_configured():
            try:
                av_info = self.alpha_vantage.fetch_company_overview(ticker)
                if av_info:
                    company_info = av_info
                    self.logger.debug(f"Retrieved company info from Alpha Vantage for {ticker}")
            except Exception as e:
                self.logger.warning(f"Failed to get company info from Alpha Vantage: {str(e)}")
        
        # Use default values if no data available
        if not company_info:
            company_info = {
                'symbol': ticker,
                'shortName': f'{ticker} Corporation',
                'longName': f'{ticker} Corporation',
                'sector': 'Unknown',
                'industry': 'Unknown',
                'website': '',
                'description': f'No company information available for {ticker}',
                'employees': 0,
                'country': 'Unknown',
                'currency': 'USD',
                'exchange': 'Unknown',
            }
        
        # Cache the result
        self.cache.set(cache_key, company_info)
        
        return company_info
    
    def _compile_results(self, **kwargs) -> Dict[str, Any]:
        """Compile all analysis results into final format."""
        try:
            # Extract arguments
            ticker = kwargs['ticker']
            raw_data = kwargs['raw_data']
            technical_data = kwargs['technical_data']
            summary_stats = kwargs['summary_stats']
            company_info = kwargs['company_info']
            quality_report = kwargs['quality_report']
            
            # Format the data for output
            formatted_data = data_formatter.format_dataframe(
                technical_data,
                columns_to_round={
                    'Open': 2, 'High': 2, 'Low': 2, 'Close': 2,
                    'SMA_20': 2, 'SMA_50': 2, 'SMA_200': 2,
                    'RSI': 1, 'MACD': 4, 'MACD_Signal': 4,
                    'BB_Upper': 2, 'BB_Lower': 2, 'ATR': 2,
                    'Support': 2, 'Resistance': 2,
                    'Daily_Return': 4, 'Volatility': 4
                }
            )
            
            # Compile final results
            results = {
                'ticker': ticker,
                'company_info': company_info,
                'summary_statistics': summary_stats,
                'data_quality': quality_report,
                'historical_data': formatted_data.to_dict('records'),
                'raw_data': raw_data,  # Keep raw data for further analysis
                'technical_data': technical_data,  # Keep full technical data
                'metadata': {
                    'processed_at': datetime.now().isoformat(),
                    'data_points': len(technical_data),
                    'indicators_calculated': list(technical_data.columns),
                    'data_source': 'Multiple (Yahoo Finance, Alpha Vantage)',
                    'cache_status': 'cached' if self.cache.get(f"historical_{ticker}_{config.DATA_PERIOD_YEARS}y") else 'fresh'
                }
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to compile results: {str(e)}")
            raise

    def check_data_freshness(self, ticker: str) -> Dict[str, Any]:
        """
        Check the freshness of cached data for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with freshness information including status, age, and whether update is needed
        """
        try:
            cache_key = f"historical_{ticker}_{config.DATA_PERIOD_YEARS}y"
            cache_path = self.cache._get_cache_path(cache_key)

            if not cache_path.exists():
                return {
                    'status': 'missing',
                    'data_exists': False,
                    'needs_update': True,
                    'cache_age_hours': None,
                    'last_updated': None,
                    'reason': 'No cached data found'
                }

            # Get cache file modification time
            cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
            current_time = datetime.now()
            cache_age = current_time - cache_time
            cache_age_hours = cache_age.total_seconds() / 3600

            # Determine if data needs update based on freshness threshold
            needs_update = cache_age_hours > config.CACHE_FRESHNESS_HOURS

            # Determine status
            if cache_age_hours < 1:
                status = 'fresh'
            elif cache_age_hours < config.CACHE_FRESHNESS_HOURS:
                status = 'current'
            else:
                status = 'stale'

            return {
                'status': status,
                'data_exists': True,
                'needs_update': needs_update,
                'cache_age_hours': round(cache_age_hours, 2),
                'last_updated': cache_time.isoformat(),
                'reason': f'Data is {cache_age_hours:.1f} hours old'
            }

        except Exception as e:
            self.logger.error(f"Error checking data freshness for {ticker}: {e}")
            return {
                'status': 'error',
                'data_exists': False,
                'needs_update': True,
                'cache_age_hours': None,
                'last_updated': None,
                'reason': f'Error checking cache: {str(e)}'
            }

    def needs_incremental_update(self, ticker: str) -> bool:
        """
        Determine if ticker needs incremental update based on cache age.

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if incremental update is needed, False otherwise
        """
        freshness_info = self.check_data_freshness(ticker)

        # If no data exists, full update is needed (not incremental)
        if not freshness_info.get('data_exists', False):
            return False

        # If data is older than threshold, incremental update is appropriate
        cache_age_hours = freshness_info.get('cache_age_hours', 0)
        return (
            cache_age_hours > config.INCREMENTAL_UPDATE_THRESHOLD_HOURS and
            cache_age_hours < (config.CACHE_FRESHNESS_HOURS * 2)  # Not too old for incremental
        )

    def get_cached_data(self, ticker: str) -> Dict[str, Any]:
        """
        Retrieve cached data bundle for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Cached data bundle or empty dict if not available
        """
        try:
            # Try to get cached historical data
            cache_key = f"historical_{ticker}_{config.DATA_PERIOD_YEARS}y"
            cached_historical = self.cache.get(cache_key)

            if not cached_historical:
                raise ValueError(f"No cached historical data for {ticker}")

            # Convert cached data back to DataFrame
            raw_data = pd.DataFrame(cached_historical)
            if 'Date' in raw_data.columns:
                raw_data['Date'] = pd.to_datetime(raw_data['Date'])
                raw_data.set_index('Date', inplace=True)

            # Try to get cached company info
            company_cache_key = f"company_info_{ticker}"
            company_info = self.cache.get(company_cache_key) or {}

            # Recalculate technical indicators and other analysis
            # (since these are derived from historical data)
            technical_data = self._calculate_technical_indicators(raw_data)
            summary_stats = self._calculate_summary_statistics(technical_data, ticker)
            quality_report = self._validate_data_quality(raw_data, ticker)

            # Compile results
            result = self._compile_results(
                ticker=ticker,
                raw_data=raw_data,
                technical_data=technical_data,
                summary_stats=summary_stats,
                company_info=company_info,
                quality_report=quality_report
            )

            self.logger.info(f"Retrieved cached data for {ticker}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to retrieve cached data for {ticker}: {e}")
            raise

    def _enhanced_fetch_historical_data(self, ticker: str, incremental: bool = False) -> pd.DataFrame:
        """
        Enhanced fetch method that supports incremental updates.

        Args:
            ticker: Stock ticker symbol
            incremental: Whether to perform incremental update

        Returns:
            DataFrame with historical data
        """
        if not incremental:
            return self._fetch_historical_data(ticker)

        try:
            # Get existing cached data
            cache_key = f"historical_{ticker}_{config.DATA_PERIOD_YEARS}y"
            cached_data = self.cache.get(cache_key)

            if not cached_data:
                # No cached data, perform full fetch
                return self._fetch_historical_data(ticker)

            # Convert to DataFrame
            cached_df = pd.DataFrame(cached_data)
            if 'Date' in cached_df.columns:
                cached_df['Date'] = pd.to_datetime(cached_df['Date'])
                cached_df.set_index('Date', inplace=True)

            # Get the latest date in cache
            latest_date = cached_df.index.max()

            # Fetch new data from the day after latest cached date
            start_date = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')

            # Only fetch if there are new days to get
            if start_date <= end_date:
                self.logger.info(f"Performing incremental update for {ticker} from {start_date}")

                # Fetch incremental data
                new_data = self._fetch_data_range(ticker, start_date, end_date)

                if not new_data.empty:
                    # Combine with cached data
                    combined_data = pd.concat([cached_df, new_data]).drop_duplicates()
                    combined_data.sort_index(inplace=True)

                    # Update cache
                    self._cache_data(ticker, combined_data)

                    return combined_data
                else:
                    self.logger.info(f"No new data available for {ticker}")
                    return cached_df
            else:
                return cached_df

        except Exception as e:
            self.logger.warning(f"Incremental update failed for {ticker}, falling back to full fetch: {e}")
            return self._fetch_historical_data(ticker)

    def _fetch_data_range(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch data for a specific date range.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with data for the specified range
        """
        try:
            # Use yfinance to fetch data for the range
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)

            if data.empty:
                return pd.DataFrame()

            # Clean and validate the data
            data = self._clean_historical_data(data)
            return data

        except Exception as e:
            self.logger.error(f"Failed to fetch data range for {ticker} ({start_date} to {end_date}): {e}")
            return pd.DataFrame()

    def _cache_data(self, ticker: str, data: pd.DataFrame):
        """
        Cache historical data for a ticker.

        Args:
            ticker: Stock ticker symbol
            data: DataFrame to cache
        """
        try:
            cache_key = f"historical_{ticker}_{config.DATA_PERIOD_YEARS}y"

            # Reset index to make Date a column for JSON serialization
            data_to_cache = data.copy()
            data_to_cache.reset_index(inplace=True)

            # Convert DataFrame to JSON-serializable format
            cache_data = data_to_cache.to_dict('records')

            self.cache.set(cache_key, cache_data)
            self.logger.debug(f"Cached historical data for {ticker}")

        except Exception as e:
            self.logger.error(f"Failed to cache data for {ticker}: {e}")


# Convenience functions for external use
def process_stock(ticker: str) -> Dict[str, Any]:
    """Process a single stock ticker."""
    processor = StockDataProcessor()
    return processor.process_stock(ticker)


def process_multiple_stocks(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """Process multiple stock tickers."""
    processor = StockDataProcessor()
    results = {}
    
    for ticker in tickers:
        try:
            results[ticker] = processor.process_stock(ticker)
        except Exception as e:
            results[ticker] = {
                'error': str(e),
                'status': 'failed'
            }
    
    return results


if __name__ == "__main__":
    # Test the processor
    import json
    
    test_ticker = "AAPL"
    print(f"Testing StockDataProcessor with {test_ticker}")
    print("=" * 50)
    
    try:
        processor = StockDataProcessor()
        result = processor.process_stock(test_ticker)
        
        # Print summary
        print(f"\n✓ Successfully processed {test_ticker}")
        print(f"  Company: {result['company_info']['longName']}")
        print(f"  Current Price: ${result['summary_statistics']['current_price']}")
        print(f"  52-Week Range: ${result['summary_statistics']['52_week_low']} - ${result['summary_statistics']['52_week_high']}")
        print(f"  1-Year Return: {result['summary_statistics']['returns']['1_year']}%")
        print(f"  Data Points: {result['metadata']['data_points']}")
        print(f"  Overall Signal: {result['summary_statistics']['signals']['overall']}")
        
        # Save sample output
        with open('sample_output.json', 'w') as f:
            # Create a serializable version
            output = {
                'ticker': result['ticker'],
                'company_info': result['company_info'],
                'summary_statistics': result['summary_statistics'],
                'data_quality': result['data_quality'],
                'metadata': result['metadata'],
                'sample_data': result['historical_data'][:5]  # Just first 5 rows
            }
            json.dump(output, f, indent=2, default=str)
            print(f"\n✓ Sample output saved to sample_output.json")
        
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()