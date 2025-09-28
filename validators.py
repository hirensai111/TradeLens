"""
Validation module for Stock Analyzer.
Handles input validation, data quality checks, and API response validation.
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any, Optional, Union
from config import config
from utils import get_logger

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

class InputValidator:
    """Validates user inputs and ticker symbols."""
    
    def __init__(self):
        self.logger = get_logger("InputValidator")
        self.valid_ticker_pattern = re.compile(r'^[\^]?[A-Z0-9\-]{1,8}(\.[A-Z]{1,2})?$')
        
    def validate_ticker(self, ticker: str) -> str:
        """
        Validate and standardize ticker symbol.
        
        Args:
            ticker: Raw ticker input
            
        Returns:
            Standardized ticker symbol
            
        Raises:
            ValidationError: If ticker is invalid
        """
        if not ticker:
            raise ValidationError("Ticker symbol cannot be empty")
        
        # Convert to uppercase
        ticker = ticker.upper().strip()
        
        # Check pattern
        if not self.valid_ticker_pattern.match(ticker):
            raise ValidationError(
                f"Invalid ticker format: '{ticker}'. "
                f"Must be 1-8 characters (letters, numbers, hyphens), optionally starting with ^, "
                f"and optionally followed by a dot and 1-2 letters."
            )
        
        self.logger.debug(f"Validated ticker: {ticker}")
        return ticker
    
    def validate_date_range(self, start_date: Optional[str], end_date: Optional[str]) -> Tuple[datetime, datetime]:
        """
        Validate date range for historical data.
        
        Args:
            start_date: Start date string (YYYY-MM-DD) or None
            end_date: End date string (YYYY-MM-DD) or None
            
        Returns:
            Tuple of (start_datetime, end_datetime)
        """
        # Default to 5 years if not specified
        if not end_date:
            end_dt = datetime.now()
        else:
            try:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                raise ValidationError(f"Invalid end date format: {end_date}. Use YYYY-MM-DD")
        
        if not start_date:
            start_dt = end_dt - timedelta(days=365 * config.DATA_PERIOD_YEARS)
        else:
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            except ValueError:
                raise ValidationError(f"Invalid start date format: {start_date}. Use YYYY-MM-DD")
        
        # Validate range
        if start_dt >= end_dt:
            raise ValidationError("Start date must be before end date")
        
        if (end_dt - start_dt).days > 365 * 10:
            raise ValidationError("Date range cannot exceed 10 years")
        
        if end_dt > datetime.now():
            raise ValidationError("End date cannot be in the future")
        
        return start_dt, end_dt
    
    def validate_output_path(self, path: str) -> str:
        """
        Validate output file path.
        
        Args:
            path: File path
            
        Returns:
            Validated path
        """
        import os
        
        if not path:
            raise ValidationError("Output path cannot be empty")
        
        # Check if directory exists
        directory = os.path.dirname(path) or '.'
        if not os.path.exists(directory):
            raise ValidationError(f"Directory does not exist: {directory}")
        
        # Check write permissions
        if not os.access(directory, os.W_OK):
            raise ValidationError(f"No write permission for directory: {directory}")
        
        # Add .xlsx extension if missing
        if not path.endswith('.xlsx'):
            path += '.xlsx'
        
        return path

class DataQualityValidator:
    """Validates data quality for stock data."""
    
    def __init__(self):
        self.logger = get_logger("DataQualityValidator")
        
    def validate_dataframe(self, df: pd.DataFrame, ticker: str) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Comprehensive data quality validation.
        
        Args:
            df: DataFrame to validate
            ticker: Stock ticker for context
            
        Returns:
            Tuple of (is_valid, warnings, quality_report)
        """
        warnings = []
        quality_report = {
            'ticker': ticker,
            'total_rows': len(df),
            'date_range': f"{df.index[0]} to {df.index[-1]}" if len(df) > 0 else "N/A",
            'missing_data': {},
            'data_issues': [],
            'completeness_score': 100.0
        }
        
        # Check if DataFrame is empty
        if df.empty:
            warnings.append("DataFrame is empty")
            quality_report['completeness_score'] = 0.0
            return False, warnings, quality_report
        
        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            warnings.append(f"Missing required columns: {missing_columns}")
            quality_report['data_issues'].append(f"Missing columns: {missing_columns}")
            return False, warnings, quality_report
        
        # Check data types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                warnings.append(f"Column {col} is not numeric")
                quality_report['data_issues'].append(f"Non-numeric column: {col}")
        
        # Check for missing values
        missing_counts = df[required_columns].isnull().sum()
        total_missing = missing_counts.sum()
        if total_missing > 0:
            quality_report['missing_data'] = missing_counts.to_dict()
            missing_pct = (total_missing / (len(df) * len(required_columns))) * 100
            warnings.append(f"Missing values detected: {missing_pct:.1f}% of data")
            quality_report['completeness_score'] -= missing_pct
        
        # Check for negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    warnings.append(f"Negative values in {col}: {negative_count} rows")
                    quality_report['data_issues'].append(f"Negative {col} prices")
        
        # Check for zero prices
        for col in price_columns:
            if col in df.columns:
                zero_count = (df[col] == 0).sum()
                if zero_count > 0:
                    warnings.append(f"Zero values in {col}: {zero_count} rows")
                    quality_report['data_issues'].append(f"Zero {col} prices")
        
        # Check OHLC logic
        ohlc_issues = self._check_ohlc_logic(df)
        if ohlc_issues:
            warnings.extend(ohlc_issues)
            quality_report['data_issues'].extend(ohlc_issues)
        
        # Check date gaps
        date_gaps = self._check_date_gaps(df)
        if date_gaps:
            warnings.append(f"Found {len(date_gaps)} significant date gaps")
            quality_report['date_gaps'] = len(date_gaps)
        
        # Check data sufficiency
        min_required_days = config.MIN_DATA_POINTS  # Use config value instead of hardcoded
        if len(df) < min_required_days:
            warnings.append(f"Insufficient data: {len(df)} days (minimum {min_required_days} required)")
            quality_report['completeness_score'] *= (len(df) / min_required_days)
        
        # Determine overall validity
        is_valid = len([w for w in warnings if 'insufficient' in w.lower() or 'missing required' in w.lower()]) == 0
        
        # Cap completeness score at 0
        quality_report['completeness_score'] = max(0, quality_report['completeness_score'])
        
        return is_valid, warnings, quality_report
    
    def _check_ohlc_logic(self, df: pd.DataFrame) -> List[str]:
        """Check OHLC logical consistency."""
        issues = []
        
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # High should be >= Low
            invalid_hl = df['High'] < df['Low']
            if invalid_hl.any():
                issues.append(f"High < Low in {invalid_hl.sum()} rows")
            
            # High should be >= Open and Close
            invalid_ho = df['High'] < df['Open']
            invalid_hc = df['High'] < df['Close']
            if invalid_ho.any() or invalid_hc.any():
                issues.append(f"High < Open/Close in {invalid_ho.sum() + invalid_hc.sum()} rows")
            
            # Low should be <= Open and Close
            invalid_lo = df['Low'] > df['Open']
            invalid_lc = df['Low'] > df['Close']
            if invalid_lo.any() or invalid_lc.any():
                issues.append(f"Low > Open/Close in {invalid_lo.sum() + invalid_lc.sum()} rows")
        
        return issues
    
    def _check_date_gaps(self, df: pd.DataFrame) -> List[Tuple[str, str, int]]:
        """Check for significant gaps in dates."""
        gaps = []
        
        # Convert index to series for proper date operations
        dates = pd.Series(df.index)
        date_diffs = dates.diff().dt.days
        
        # Look for gaps > 5 days (accounting for weekends)
        significant_gaps = date_diffs[date_diffs > 5]
        
        # Fix: Check if there are any significant gaps before iterating
        if len(significant_gaps) > 0:
            for idx in significant_gaps.index:
                if idx > 0:  # Skip the first index (which would be NaN from diff())
                    gap_start = dates.iloc[idx - 1].strftime('%Y-%m-%d')
                    gap_end = dates.iloc[idx].strftime('%Y-%m-%d')
                    gap_days = int(date_diffs.iloc[idx])
                    gaps.append((gap_start, gap_end, gap_days))
        
        return gaps

class APIResponseValidator:
    """Validates API responses."""
    
    def __init__(self):
        self.logger = get_logger("APIResponseValidator")
    
    def validate_yfinance_response(self, data: pd.DataFrame, ticker: str) -> bool:
        """
        Validate Yahoo Finance API response.
        
        Args:
            data: Response data
            ticker: Expected ticker
            
        Returns:
            True if valid, False otherwise
        """
        if data is None or data.empty:
            self.logger.warning(f"Empty response for {ticker}")
            return False
        
        # Check if it's a DataFrame
        if not isinstance(data, pd.DataFrame):
            self.logger.warning(f"Response is not a DataFrame for {ticker}")
            return False
        
        # Check minimum required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            self.logger.warning(f"Missing required columns in response for {ticker}")
            return False
        
        return True
    
    def validate_alpha_vantage_response(self, data: Union[pd.DataFrame, Dict], ticker: str) -> bool:
        """
        Validate Alpha Vantage API response.
        
        Args:
            data: Response data (DataFrame or dict)
            ticker: Expected ticker
            
        Returns:
            True if valid, False otherwise
        """
        if data is None:
            self.logger.warning(f"None response for {ticker}")
            return False
        
        # If it's a dict, check for error messages
        if isinstance(data, dict):
            if "Error Message" in data:
                self.logger.warning(f"Alpha Vantage error: {data['Error Message']}")
                return False
            if "Note" in data:
                self.logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                return False
        
        # If it's a DataFrame, validate like Yahoo Finance
        if isinstance(data, pd.DataFrame):
            return self.validate_yfinance_response(data, ticker)
        
        return False
    
    def validate_openai_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate OpenAI API response.
        
        Args:
            response: API response
            
        Returns:
            True if valid, False otherwise
        """
        if not response:
            self.logger.warning("Empty OpenAI response")
            return False
        
        # Check for required fields
        if 'choices' not in response or not response['choices']:
            self.logger.warning("No choices in OpenAI response")
            return False
        
        if 'message' not in response['choices'][0]:
            self.logger.warning("No message in OpenAI response")
            return False
        
        return True

class ConfigValidator:
    """Validates configuration settings."""
    
    def __init__(self):
        self.logger = get_logger("ConfigValidator")
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """
        Validate all configuration settings.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check API keys
        if config.OPENAI_API_KEY:
            if config.OPENAI_API_KEY == "your-openai-api-key-here":
                errors.append("OpenAI API key not properly configured (still using placeholder)")
        else:
            # OpenAI is optional for basic functionality
            self.logger.info("OpenAI API key not configured - AI insights will be unavailable")
        
        # Check Alpha Vantage (optional)
        if config.ALPHA_VANTAGE_API_KEY:
            if config.ALPHA_VANTAGE_API_KEY == "your-alpha-vantage-key-here":
                errors.append("Alpha Vantage API key not properly configured (still using placeholder)")
        
        # Check numeric settings
        if config.DATA_PERIOD_YEARS < 1 or config.DATA_PERIOD_YEARS > 10:
            errors.append("DATA_PERIOD_YEARS must be between 1 and 10")
        
        if config.MAX_RETRIES < 1:
            errors.append("MAX_RETRIES must be at least 1")
        
        if config.REQUEST_TIMEOUT < 5:
            errors.append("REQUEST_TIMEOUT must be at least 5 seconds")
        
        # Check paths - use the actual config attributes
        import os
        if not os.path.exists(config.OUTPUT_DIR):
            try:
                config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create output directory: {str(e)}")
        
        if not os.path.exists(config.LOGS_DIR):
            try:
                config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create log directory: {str(e)}")
        
        # Check technical indicator settings
        if not all(isinstance(p, int) and p > 0 for p in config.MA_PERIODS):
            errors.append("MA_PERIODS must contain positive integers")
        
        if config.RSI_PERIOD < 2:
            errors.append("RSI_PERIOD must be at least 2")
        
        # Only return critical errors that prevent operation
        critical_errors = [e for e in errors if "placeholder" not in e.lower()]
        
        return len(critical_errors) == 0, errors


# Create singleton instances
input_validator = InputValidator()
data_quality_validator = DataQualityValidator()
api_response_validator = APIResponseValidator()
config_validator = ConfigValidator()

def validate_ticker(ticker: str) -> str:
    """Validate and standardize ticker symbol."""
    return input_validator.validate_ticker(ticker)

def validate_data_quality(df: pd.DataFrame, ticker: str) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Validate data quality."""
    return data_quality_validator.validate_dataframe(df, ticker)

def test_validators():
    """Test validation functions."""
    print("Testing Validators")
    print("=" * 50)
    
    # Ensure test directories exist
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.LOGS_DIR.mkdir(exist_ok=True)
    
    # Test ticker validation
    print("\n1. Testing ticker validation:")
    test_tickers = ["AAPL", "MSFT", "GOOGL", "123", "TOOLONG", "aapl", ""]
    for ticker in test_tickers:
        try:
            validated = validate_ticker(ticker)
            print(f"✓ '{ticker}' -> '{validated}'")
        except ValidationError as e:
            print(f"✗ '{ticker}' -> Error: {str(e)}")
    
    # Test date validation
    print("\n2. Testing date validation:")
    test_dates = [
        (None, None),
        ("2020-01-01", "2023-12-31"),
        ("2023-12-31", "2020-01-01"),
        ("invalid", "2023-12-31")
    ]
    for start, end in test_dates:
        try:
            start_dt, end_dt = input_validator.validate_date_range(start, end)
            print(f"✓ ({start}, {end}) -> Valid range")
        except ValidationError as e:
            print(f"✗ ({start}, {end}) -> Error: {str(e)}")
    
    # Test data quality validation
    print("\n3. Testing data quality validation:")
    
    # Create test DataFrame
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    test_df = pd.DataFrame({
        'Open': np.random.uniform(100, 200, len(dates)),
        'High': np.random.uniform(100, 200, len(dates)),
        'Low': np.random.uniform(100, 200, len(dates)),
        'Close': np.random.uniform(100, 200, len(dates)),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    is_valid, warnings, report = validate_data_quality(test_df, "TEST")
    print(f"✓ Valid data: {is_valid}")
    print(f"✓ Completeness score: {report['completeness_score']:.1f}%")
    if warnings:
        print(f"⚠ Warnings: {warnings}")
    
    # Test configuration validation
    print("\n4. Testing configuration validation:")
    is_valid, errors = config_validator.validate_config()
    if is_valid:
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    
    print("\n✅ Validator tests completed!")

if __name__ == "__main__":
    test_validators()