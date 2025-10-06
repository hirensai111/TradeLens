"""
Utility functions and helpers for Stock Analyzer.
Provides logging, formatting, progress tracking, and common utilities.
"""

import os
import sys
import logging
import colorlog
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
import json
import time
from functools import wraps
import pandas as pd

from core.config.config import config


class ColoredFormatter(colorlog.ColoredFormatter):
    """Custom colored formatter for better log readability."""
    
    def __init__(self):
        super().__init__(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt=config.LOG_DATE_FORMAT,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )


class StockAnalyzerLogger:
    """Centralized logging system for the Stock Analyzer."""
    
    def __init__(self, name: str = "StockAnalyzer"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.LOG_LEVEL))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up console and file handlers."""
        # Console handler with colors
        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(ColoredFormatter())
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = config.LOGS_DIR / config.get_log_filename()
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            config.LOG_FORMAT,
            datefmt=config.LOG_DATE_FORMAT
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, **kwargs)
    
    def log_function_call(self, func_name: str, ticker: str, **kwargs):
        """Log function calls with context."""
        self.info(f"[CALL] {func_name} called for {ticker}", **kwargs)
    
    def log_success(self, message: str, **kwargs):
        """Log success message with special formatting."""
        self.info(f"[OK] {message}", **kwargs)
    
    def log_failure(self, message: str, **kwargs):
        """Log failure message with special formatting."""
        self.error(f"[FAIL] {message}", **kwargs)
    
    def log_progress(self, message: str, **kwargs):
        """Log progress message."""
        self.info(f"⏳ {message}", **kwargs)


class ProgressTracker:
    """Track and display progress for long-running operations."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.logger = get_logger("ProgressTracker")
    
    def update(self, step_description: str = ""):
        """Update progress by one step."""
        self.current_step += 1
        percentage = (self.current_step / self.total_steps) * 100
        
        # Calculate elapsed and estimated time
        elapsed = time.time() - self.start_time
        if self.current_step > 0:
            estimated_total = elapsed * (self.total_steps / self.current_step)
            remaining = estimated_total - elapsed
        else:
            remaining = 0
        
        # Create progress bar
        bar_length = 30
        filled = int(bar_length * percentage / 100)
        bar = "#" * filled + "-" * (bar_length - filled)
        
        # Format time
        elapsed_str = format_duration(elapsed)
        remaining_str = format_duration(remaining)
        
        # Log progress
        progress_msg = f"{self.description}: [{bar}] {percentage:5.1f}% ({self.current_step}/{self.total_steps})"
        if step_description:
            progress_msg += f" - {step_description}"
        progress_msg += f" | Elapsed: {elapsed_str} | Remaining: {remaining_str}"
        
        self.logger.info(progress_msg)
    
    def finish(self, success_message: str = ""):
        """Mark progress as complete."""
        total_time = time.time() - self.start_time
        time_str = format_duration(total_time)
        
        if success_message:
            self.logger.info(f"[OK] {success_message} (completed in {time_str})")
        else:
            self.logger.info(f"[OK] {self.description} completed in {time_str}")


class DataFormatter:
    """Format data for display and export."""
    
    @staticmethod
    def format_currency(value: float, currency: str = "USD") -> str:
        """Format currency values."""
        if currency == "USD":
            return f"${value:,.2f}"
        else:
            return f"{value:,.2f} {currency}"
    
    @staticmethod
    def format_percentage(value: float, decimal_places: int = 2) -> str:
        """Format percentage values."""
        return f"{value:.{decimal_places}f}%"
    
    @staticmethod
    def format_large_number(value: float) -> str:
        """Format large numbers with K, M, B suffixes."""
        if abs(value) >= 1e12:
            return f"{value/1e12:.2f}T"
        elif abs(value) >= 1e9:
            return f"{value/1e9:.2f}B"
        elif abs(value) >= 1e6:
            return f"{value/1e6:.2f}M"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.2f}K"
        else:
            return f"{value:.2f}"
    
    @staticmethod
    def format_date(date_value: Union[datetime, pd.Timestamp, str]) -> str:
        """Format dates consistently."""
        if isinstance(date_value, str):
            return date_value
        elif isinstance(date_value, (datetime, pd.Timestamp)):
            return date_value.strftime("%Y-%m-%d")
        else:
            return str(date_value)
    
    @staticmethod
    def format_market_cap(market_cap: float) -> str:
        """Format market capitalization."""
        if market_cap <= 0:
            return "N/A"
        
        if market_cap >= 200e9:
            return f"{market_cap/1e9:.1f}B (Large Cap)"
        elif market_cap >= 10e9:
            return f"{market_cap/1e9:.1f}B (Mid Cap)"
        elif market_cap >= 2e9:
            return f"{market_cap/1e9:.1f}B (Small Cap)"
        else:
            return f"{market_cap/1e6:.1f}M (Micro Cap)"
        
    @staticmethod
    def format_dataframe(df: pd.DataFrame, 
                        columns_to_round: Optional[Dict[str, int]] = None,
                        date_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Format DataFrame for display/export.
        
        Args:
            df: DataFrame to format
            columns_to_round: Dict mapping column names to decimal places
            date_columns: List of columns to format as dates
            
        Returns:
            Formatted DataFrame (copy)
        """
        formatted_df = df.copy()
        
        # Round numeric columns
        if columns_to_round:
            for col, decimals in columns_to_round.items():
                if col in formatted_df.columns:
                    formatted_df[col] = formatted_df[col].round(decimals)
        
        # Format date columns
        if date_columns:
            for col in date_columns:
                if col in formatted_df.columns:
                    formatted_df[col] = pd.to_datetime(formatted_df[col]).dt.strftime('%Y-%m-%d')
        
        return formatted_df
    
    @staticmethod
    def format_signal_strength(value: str) -> str:
        """Format signal strength with emoji indicators."""
        signal_map = {
            'Bullish': '🟢 Bullish',
            'Bearish': '🔴 Bearish',
            'Neutral': '🟡 Neutral',
            'Overbought': '[WARNING] Overbought',
            'Oversold': '[WARNING] Oversold'
        }
        return signal_map.get(value, value)


class FileManager:
    """Handle file operations and management."""
    
    @staticmethod
    def ensure_directory_exists(directory: Path) -> bool:
        """Ensure directory exists, create if it doesn't."""
        try:
            directory.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger = get_logger("FileManager")
            logger.error(f"Failed to create directory {directory}: {e}")
            return False
    
    @staticmethod
    def save_json(data: Dict[str, Any], filepath: Path) -> bool:
        """Save data as JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger = get_logger("FileManager")
            logger.error(f"Failed to save JSON to {filepath}: {e}")
            return False
    
    @staticmethod
    def load_json(filepath: Path) -> Optional[Dict[str, Any]]:
        """Load data from JSON file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger = get_logger("FileManager")
            logger.error(f"Failed to load JSON from {filepath}: {e}")
            return None
    
    @staticmethod
    def get_file_size(filepath: Path) -> str:
        """Get human-readable file size."""
        try:
            size = filepath.stat().st_size
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} TB"
        except:
            return "Unknown"
    
    @staticmethod
    def clean_old_files(directory: Path, days_old: int = 30) -> int:
        """Clean files older than specified days."""
        try:
            count = 0
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            for file_path in directory.glob("*"):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_path.unlink()
                        count += 1
            
            return count
        except Exception as e:
            logger = get_logger("FileManager")
            logger.error(f"Failed to clean old files: {e}")
            return 0


class CacheManager:
    """Simple caching system for API responses."""
    
    def __init__(self, cache_dir: Path = config.CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_duration = timedelta(hours=config.CACHE_DURATION_HOURS)
        self.logger = get_logger("CacheManager")
        
        # Ensure cache directory exists
        FileManager.ensure_directory_exists(self.cache_dir)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a given key."""
        # Sanitize key to be filesystem-safe
        safe_key = key.replace('/', '_').replace('\\', '_').replace(':', '_')
        return self.cache_dir / f"{safe_key}.json"
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data if valid."""
        if not config.CACHE_ENABLED:
            return None
        
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            # Check if cache is still valid
            cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if datetime.now() - cache_time > self.cache_duration:
                cache_path.unlink()  # Remove expired cache
                return None
            
            # Load cached data
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            self.logger.debug(f"Cache hit for {key}")
            return data
        
        except Exception as e:
            self.logger.error(f"Failed to load cache for {key}: {e}")
            return None
    
    def set(self, key: str, data: Any) -> bool:
        """Cache data."""
        if not config.CACHE_ENABLED:
            return False

        try:
            cache_path = self._get_cache_path(key)

            # Convert pandas objects to serializable format
            if isinstance(data, pd.DataFrame):
                # FIXED: Preserve index as a column to prevent data loss
                # Reset index converts the Date index to a 'Date' column
                data_copy = data.copy()
                if isinstance(data_copy.index, pd.DatetimeIndex):
                    # Preserve the index name if it exists
                    index_name = data_copy.index.name or 'Date'
                    data_copy = data_copy.reset_index()
                    if 'index' in data_copy.columns and index_name != 'index':
                        data_copy = data_copy.rename(columns={'index': index_name})
                data = data_copy.to_dict('records')
            elif isinstance(data, pd.Series):
                data = data.to_dict()

            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            self.logger.debug(f"Cached data for {key}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cache data for {key}: {e}")
            return False
    
    def clear(self, key: Optional[str] = None) -> bool:
        """Clear cache (specific key or all)."""
        try:
            if key:
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    cache_path.unlink()
            else:
                # Clear all cache files
                for cache_file in self.cache_dir.glob("*.json"):
                    cache_file.unlink()
            
            self.logger.info(f"Cleared cache for {key if key else 'all keys'}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False


def get_logger(name: str) -> StockAnalyzerLogger:
    """Get a logger instance."""
    return StockAnalyzerLogger(name)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry function calls on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger("RetryDecorator")
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Function {func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    
                    wait_time = delay * (backoff ** attempt)
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.1f}s: {e}")
                    time.sleep(wait_time)
            
            return None
        return wrapper
    return decorator


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, return default if division by zero."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values."""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """Validate that DataFrame has required columns."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    return len(missing_columns) == 0, missing_columns


def print_banner(title: str, width: int = 60):
    """Print a formatted banner."""
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)


def print_summary_table(data: Dict[str, Any], title: str = "Summary"):
    """Print a formatted summary table."""
    print(f"\n{title}")
    print("-" * len(title))
    
    max_key_length = max(len(str(key)) for key in data.keys())
    
    for key, value in data.items():
        if isinstance(value, float):
            if abs(value) > 1000:
                value_str = DataFormatter.format_large_number(value)
            else:
                value_str = f"{value:.2f}"
        else:
            value_str = str(value)
        
        print(f"{key:<{max_key_length}} : {value_str}")


def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging."""
    return {
        'python_version': sys.version,
        'platform': sys.platform,
        'working_directory': os.getcwd(),
        'config_debug_mode': config.DEBUG_MODE,
        'log_level': config.LOG_LEVEL,
        'cache_enabled': config.CACHE_ENABLED,
        'timestamp': datetime.now().isoformat()
    }
    
def sanitize_filename(filename: str) -> str:
    """Sanitize filename to be filesystem-safe."""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


def get_market_hours_status() -> Dict[str, Any]:
    """
    Get comprehensive market hours status including timezone handling and holidays.

    Returns:
        Dict with market status, times, and detailed information
    """
    try:
        import pytz
        # US Eastern timezone (handles DST automatically)
        eastern = pytz.timezone('US/Eastern')
        now_et = datetime.now(eastern)
    except ImportError:
        # Fallback to local time if pytz not available
        now_et = datetime.now()

    # Check if it's a weekday (Monday=0, Sunday=6)
    is_weekday = now_et.weekday() < 5

    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    # Pre-market: 4:00 AM - 9:30 AM ET
    premarket_open = now_et.replace(hour=4, minute=0, second=0, microsecond=0)

    # After-hours: 4:00 PM - 8:00 PM ET
    afterhours_close = now_et.replace(hour=20, minute=0, second=0, microsecond=0)

    # Check if current time is within trading periods
    is_market_hours = market_open <= now_et <= market_close
    is_premarket = premarket_open <= now_et < market_open
    is_afterhours = market_close < now_et <= afterhours_close

    # Check for market holidays (basic implementation)
    is_market_holiday = _is_market_holiday(now_et)

    # Determine market status
    if not is_weekday:
        status = "CLOSED_WEEKEND"
        status_message = "Market closed (Weekend)"
    elif is_market_holiday:
        status = "CLOSED_HOLIDAY"
        status_message = "Market closed (Holiday)"
    elif is_market_hours:
        status = "OPEN"
        status_message = "Market open"
    elif is_premarket:
        status = "PREMARKET"
        status_message = "Pre-market trading"
    elif is_afterhours:
        status = "AFTERHOURS"
        status_message = "After-hours trading"
    else:
        status = "CLOSED"
        status_message = "Market closed"

    # Calculate next market open
    next_open = _get_next_market_open(now_et)

    return {
        'is_open': status == "OPEN",
        'status': status,
        'status_message': status_message,
        'current_time_et': now_et.strftime('%Y-%m-%d %H:%M:%S %Z'),
        'current_time_local': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'market_open': market_open.strftime('%H:%M ET'),
        'market_close': market_close.strftime('%H:%M ET'),
        'premarket_open': premarket_open.strftime('%H:%M ET'),
        'afterhours_close': afterhours_close.strftime('%H:%M ET'),
        'is_weekday': is_weekday,
        'is_holiday': is_market_holiday,
        'is_premarket': is_premarket,
        'is_afterhours': is_afterhours,
        'next_market_open': next_open.strftime('%Y-%m-%d %H:%M ET') if next_open else None,
        'day_name': now_et.strftime('%A')
    }


def _is_market_holiday(date_et: datetime) -> bool:
    """
    Check if the given date is a US market holiday.

    Args:
        date_et: Datetime in Eastern timezone

    Returns:
        True if it's a market holiday
    """
    # Basic implementation of major US market holidays
    year = date_et.year
    month = date_et.month
    day = date_et.day

    # Fixed date holidays
    holidays = [
        (1, 1),   # New Year's Day
        (7, 4),   # Independence Day
        (12, 25), # Christmas Day
    ]

    if (month, day) in holidays:
        return True

    # Martin Luther King Jr. Day (3rd Monday in January)
    if month == 1 and _is_nth_weekday(date_et, 1, 3):
        return True

    # Presidents' Day (3rd Monday in February)
    if month == 2 and _is_nth_weekday(date_et, 1, 3):
        return True

    # Good Friday (Friday before Easter) - complex calculation, skip for now

    # Memorial Day (last Monday in May)
    if month == 5 and _is_last_weekday(date_et, 1):
        return True

    # Labor Day (1st Monday in September)
    if month == 9 and _is_nth_weekday(date_et, 1, 1):
        return True

    # Thanksgiving (4th Thursday in November)
    if month == 11 and _is_nth_weekday(date_et, 4, 4):
        return True

    return False


def _is_nth_weekday(date_et: datetime, weekday: int, n: int) -> bool:
    """Check if date is the nth occurrence of weekday in the month."""
    if date_et.weekday() != weekday:
        return False

    # Count how many of this weekday have occurred this month
    first_of_month = date_et.replace(day=1)
    days_to_first_weekday = (weekday - first_of_month.weekday()) % 7
    first_occurrence = first_of_month.day + days_to_first_weekday

    occurrence_number = ((date_et.day - first_occurrence) // 7) + 1
    return occurrence_number == n


def _is_last_weekday(date_et: datetime, weekday: int) -> bool:
    """Check if date is the last occurrence of weekday in the month."""
    if date_et.weekday() != weekday:
        return False

    # Check if adding 7 days would go to next month
    next_week = date_et + timedelta(days=7)
    return next_week.month != date_et.month


def _get_next_market_open(current_time_et: datetime) -> datetime:
    """
    Calculate the next market opening time.

    Args:
        current_time_et: Current time in Eastern timezone

    Returns:
        Next market open datetime
    """
    # Start with next day
    next_day = current_time_et + timedelta(days=1)

    # Look for next weekday (Monday-Friday)
    while next_day.weekday() >= 5 or _is_market_holiday(next_day):
        next_day += timedelta(days=1)

    # Set to 9:30 AM ET
    return next_day.replace(hour=9, minute=30, second=0, microsecond=0)


def format_technical_indicator(indicator: str, value: Optional[float]) -> str:
    """Format technical indicator with appropriate precision."""
    if value is None:
        return "N/A"
    
    precision_map = {
        'RSI': 1,
        'MACD': 4,
        'ATR': 2,
        'SMA': 2,
        'EMA': 2,
        'BB': 2,
        'Stoch': 1
    }
    
    # Find matching precision
    precision = 2  # default
    for key, prec in precision_map.items():
        if key in indicator.upper():
            precision = prec
            break
    
    return f"{value:.{precision}f}"


def create_summary_report(results: Dict[str, Any]) -> str:
    """Create a text summary report from analysis results."""
    report = []
    
    # Header
    report.append("=" * 70)
    report.append(f"Stock Analysis Report - {results.get('ticker', 'Unknown')}")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    
    # Company Info
    if 'company_info' in results:
        info = results['company_info']
        report.append(f"\nCompany: {info.get('longName', 'N/A')}")
        report.append(f"Sector: {info.get('sector', 'N/A')}")
        report.append(f"Industry: {info.get('industry', 'N/A')}")
    
    # Price Summary
    if 'summary_statistics' in results:
        stats = results['summary_statistics']
        report.append(f"\nCurrent Price: ${stats.get('current_price', 0):.2f}")
        report.append(f"Day Change: {stats.get('price_change_1d_pct', 0):.2f}%")
        report.append(f"52-Week Range: ${stats.get('52_week_low', 0):.2f} - ${stats.get('52_week_high', 0):.2f}")
        
        # Returns
        if 'returns' in stats:
            report.append("\nReturns:")
            for period, return_val in stats['returns'].items():
                if return_val is not None:
                    report.append(f"  {period}: {return_val:.2f}%")
        
        # Technical Indicators
        if 'technical_indicators' in stats:
            report.append("\nTechnical Indicators:")
            for indicator, value in stats['technical_indicators'].items():
                if value is not None:
                    formatted_value = format_technical_indicator(indicator, value)
                    report.append(f"  {indicator}: {formatted_value}")
        
        # Signals
        if 'signals' in stats:
            signals = stats['signals']
            report.append(f"\nOverall Signal: {signals.get('overall', 'N/A')}")
            if 'signals' in signals and signals['signals']:
                report.append("Key Signals:")
                for signal in signals['signals'][:5]:  # Top 5 signals
                    report.append(f"  • {signal}")
    
    report.append("\n" + "=" * 70)
    
    return "\n".join(report)


# Global instances
logger = get_logger("Utils")
cache_manager = CacheManager()
file_manager = FileManager()
data_formatter = DataFormatter()


def test_utils():
    """Test utility functions."""
    print_banner("Testing Stock Analyzer Utils")
    
    # Test logger
    test_logger = get_logger("TestUtils")
    test_logger.info("Testing logger functionality")
    test_logger.log_success("Logger test passed")
    
    # Test formatter
    print("\nTesting Data Formatter:")
    print(f"Currency: {data_formatter.format_currency(1234.567)}")
    print(f"Percentage: {data_formatter.format_percentage(12.345)}")
    print(f"Large Number: {data_formatter.format_large_number(1234567890)}")
    print(f"Market Cap: {data_formatter.format_market_cap(50e9)}")
    
    # Test progress tracker
    print("\nTesting Progress Tracker:")
    progress = ProgressTracker(3, "Test Process")
    progress.update("Step 1")
    time.sleep(0.5)
    progress.update("Step 2")
    time.sleep(0.5)
    progress.update("Step 3")
    progress.finish("All tests completed")
    
    # Test utilities
    print("\nTesting Utilities:")
    print(f"Duration: {format_duration(125.5)}")
    print(f"Percentage Change: {calculate_percentage_change(100, 150):.1f}%")
    
    # Test summary table
    test_data = {
        'Stock': 'AAPL',
        'Current Price': 150.25,
        'Market Cap': 2.5e12,
        'Change': 2.3
    }
    print_summary_table(test_data, "Stock Summary")
    
    print("\n[OK] All utils tests completed!")


if __name__ == "__main__":
    test_utils()