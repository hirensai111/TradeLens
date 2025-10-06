"""
Alpha Vantage data source integration for Stock Analyzer.
Provides fallback data source when Yahoo Finance is unavailable.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import time
import json

from core.config.config import config
from core.utils.utils import get_logger, retry_on_failure
from core.validators.validators import ValidationError


class AlphaVantageClient:
    """Client for fetching stock data from Alpha Vantage API."""
    
    def __init__(self):
        self.logger = get_logger("AlphaVantageClient")
        self.api_key = config.ALPHA_VANTAGE_API_KEY
        self.base_url = config.ALPHA_VANTAGE_BASE_URL
        self.rate_limiter = RateLimiter(
            calls_per_minute=config.ALPHA_VANTAGE_RATE_LIMIT,
            daily_limit=config.ALPHA_VANTAGE_DAILY_LIMIT
        )
        
        if not self.api_key:
            self.logger.warning("Alpha Vantage API key not configured")
    
    def is_configured(self) -> bool:
        """Check if Alpha Vantage is properly configured."""
        return bool(self.api_key and self.api_key != "your-alpha-vantage-key-here")
    
    @retry_on_failure(max_retries=2, delay=2.0)
    def fetch_daily_data(self, ticker: str, outputsize: str = "full") -> pd.DataFrame:
        """
        Fetch daily historical data from Alpha Vantage.
        
        Args:
            ticker: Stock ticker symbol
            outputsize: 'compact' for last 100 days, 'full' for up to 20 years
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.is_configured():
            raise ValidationError("Alpha Vantage API key not configured")
        
        # Check rate limits
        if not self.rate_limiter.can_make_request():
            wait_time = self.rate_limiter.get_wait_time()
            self.logger.warning(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        
        # Prepare API request
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': ticker,
            'apikey': self.api_key,
            'outputsize': outputsize,
            'datatype': 'json'
        }
        
        try:
            self.logger.debug(f"Fetching data from Alpha Vantage for {ticker}")
            response = requests.get(
                self.base_url,
                params=params,
                timeout=config.ALPHA_VANTAGE_TIMEOUT
            )
            
            # Record the API call
            self.rate_limiter.record_request()
            
            # Check response status
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                raise ValidationError(f"Alpha Vantage error: {data['Error Message']}")
            
            if "Note" in data:
                # API call frequency limit
                self.logger.warning(f"Alpha Vantage note: {data['Note']}")
                raise ValidationError("Alpha Vantage API call frequency exceeded")
            
            if "Time Series (Daily)" not in data:
                raise ValidationError("Invalid response format from Alpha Vantage")
            
            # Convert to DataFrame
            time_series = data["Time Series (Daily)"]
            df = self._convert_to_dataframe(time_series)
            
            # Get only the last 5 years of data
            five_years_ago = datetime.now() - timedelta(days=365 * config.DATA_PERIOD_YEARS)
            df = df[df.index >= five_years_ago]
            
            self.logger.info(f"Successfully fetched {len(df)} days of data for {ticker} from Alpha Vantage")
            return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error fetching data from Alpha Vantage: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error fetching data from Alpha Vantage: {e}")
            raise
    
    def _convert_to_dataframe(self, time_series: Dict[str, Dict[str, str]]) -> pd.DataFrame:
        """Convert Alpha Vantage time series data to DataFrame."""
        data = []
        
        for date_str, values in time_series.items():
            try:
                data.append({
                    'Date': pd.to_datetime(date_str),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Adjusted Close': float(values['5. adjusted close']),
                    'Volume': int(values['6. volume']),
                    'Dividend': float(values.get('7. dividend amount', 0)),
                    'Split': float(values.get('8. split coefficient', 1))
                })
            except (KeyError, ValueError) as e:
                self.logger.warning(f"Skipping invalid data point for date {date_str}: {e}")
                continue
        
        if not data:
            raise ValidationError("No valid data points found in Alpha Vantage response")
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # Use adjusted close as the main close price
        df['Close'] = df['Adjusted Close']
        
        # Keep only the columns we need (matching yfinance format)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        return df
    
    def fetch_company_overview(self, ticker: str) -> Dict[str, Any]:
        """Fetch company information from Alpha Vantage."""
        if not self.is_configured():
            return {}
        
        params = {
            'function': 'OVERVIEW',
            'symbol': ticker,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(
                self.base_url,
                params=params,
                timeout=config.ALPHA_VANTAGE_TIMEOUT
            )
            
            response.raise_for_status()
            data = response.json()
            
            if not data or "Symbol" not in data:
                return {}
            
            # Convert to format matching yfinance
            return {
                'symbol': data.get('Symbol', ticker),
                'shortName': data.get('Name', f'{ticker} Corporation'),
                'longName': data.get('Name', f'{ticker} Corporation'),
                'sector': data.get('Sector', 'Unknown'),
                'industry': data.get('Industry', 'Unknown'),
                'exchange': data.get('Exchange', 'Unknown'),
                'currency': data.get('Currency', 'USD'),
                'marketCap': float(data.get('MarketCapitalization', 0)),
                'employees': int(data.get('FullTimeEmployees', 0)) if data.get('FullTimeEmployees') else 0,
                'website': data.get('Website', ''),
                'description': data.get('Description', '')[:500] + '...' if data.get('Description') else ''
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to fetch company overview from Alpha Vantage: {e}")
            return {}


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int = 5, daily_limit: int = 25):
        self.calls_per_minute = calls_per_minute
        self.daily_limit = daily_limit
        self.call_times = []
        self.daily_calls = 0
        self.last_reset = datetime.now().date()
        self.logger = get_logger("RateLimiter")
    
    def can_make_request(self) -> bool:
        """Check if we can make a request without exceeding rate limits."""
        self._reset_if_new_day()
        
        # Check daily limit
        if self.daily_calls >= self.daily_limit:
            self.logger.warning(f"Daily limit of {self.daily_limit} calls reached")
            return False
        
        # Check per-minute limit
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Remove old calls
        self.call_times = [t for t in self.call_times if t > minute_ago]
        
        return len(self.call_times) < self.calls_per_minute
    
    def record_request(self):
        """Record that a request was made."""
        self._reset_if_new_day()
        self.call_times.append(datetime.now())
        self.daily_calls += 1
        self.logger.debug(f"API call recorded. Daily calls: {self.daily_calls}/{self.daily_limit}")
    
    def get_wait_time(self) -> float:
        """Get the time to wait before the next request can be made."""
        if self.daily_calls >= self.daily_limit:
            # Wait until tomorrow
            tomorrow = datetime.now().replace(hour=0, minute=0, second=0) + timedelta(days=1)
            return (tomorrow - datetime.now()).total_seconds()
        
        if len(self.call_times) >= self.calls_per_minute:
            # Wait until the oldest call is more than a minute old
            oldest_call = min(self.call_times)
            wait_until = oldest_call + timedelta(minutes=1)
            return max(0, (wait_until - datetime.now()).total_seconds())
        
        return 0
    
    def _reset_if_new_day(self):
        """Reset daily counter if it's a new day."""
        today = datetime.now().date()
        if today > self.last_reset:
            self.daily_calls = 0
            self.last_reset = today
            self.logger.debug("Daily rate limit counter reset")


# Global client instance
alpha_vantage_client = AlphaVantageClient()


def test_alpha_vantage():
    """Test Alpha Vantage integration."""
    print("Testing Alpha Vantage Integration")
    print("=" * 50)
    
    client = AlphaVantageClient()
    
    if not client.is_configured():
        print("[ERROR] Alpha Vantage API key not configured")
        print("Please add your API key to the .env file:")
        print("ALPHA_VANTAGE_API_KEY=your-key-here")
        return
    
    try:
        # Test with Apple stock
        ticker = "AAPL"
        print(f"\nFetching data for {ticker}...")
        
        # Fetch daily data
        df = client.fetch_daily_data(ticker, outputsize="compact")
        print(f"[OK] Retrieved {len(df)} days of data")
        print(f"[OK] Date range: {df.index[0]} to {df.index[-1]}")
        print(f"[OK] Latest close price: ${df['Close'].iloc[-1]:.2f}")
        
        # Fetch company info
        print(f"\nFetching company information...")
        info = client.fetch_company_overview(ticker)
        if info:
            print(f"[OK] Company: {info.get('longName', 'Unknown')}")
            print(f"[OK] Sector: {info.get('sector', 'Unknown')}")
            print(f"[OK] Industry: {info.get('industry', 'Unknown')}")
        
        print("\n[OK] Alpha Vantage test completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_alpha_vantage()