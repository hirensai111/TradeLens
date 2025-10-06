"""
Configuration settings for Stock Analyzer Phase 1
Contains all constants, settings, and configuration parameters.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import warnings

# Load environment variables from .env file (gracefully handle missing file)
try:
    load_dotenv()
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")
    print("Using default environment settings...")

class Config:
    """Main configuration class for the Stock Analyzer application."""
    
    # ========================
    # APPLICATION SETTINGS
    # ========================
    APP_NAME = "Stock Analyzer Phase 1"
    VERSION = "1.0.0"
    AUTHOR = "Hiren Sai Vellanki"
    
    # ========================
    # DIRECTORIES & PATHS
    # ========================
    # FIXED: Use project root instead of module directory to unify all cache locations
    PROJECT_ROOT = Path(__file__).parent.parent.parent  # d:\tradelens\
    BASE_DIR = PROJECT_ROOT

    # All paths relative to project root for consistency across modules
    CACHE_DIR = PROJECT_ROOT / "cache"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    LOGS_DIR = PROJECT_ROOT / "logs"
    VIZ_DATA_DIR = PROJECT_ROOT / "output" / "viz_data"

    # Ensure directories exist
    CACHE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    VIZ_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========================
    # DATA COLLECTION SETTINGS
    # ========================
    
    # Historical data period (5 years as specified in requirements)
    DATA_PERIOD_YEARS = 5
    DATA_START_DATE = (datetime.now() - timedelta(days=365 * DATA_PERIOD_YEARS)).strftime('%Y-%m-%d')
    DATA_END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    # Data source priority (will try in order)
    DATA_SOURCES = ['yfinance', 'alpha_vantage', 'twelve_data']
    PRIMARY_DATA_SOURCE = 'yfinance'
    
    # Yahoo Finance settings
    YFINANCE_TIMEOUT = 30  # seconds
    YFINANCE_RETRY_ATTEMPTS = 3
    YFINANCE_RETRY_DELAY = 2  # seconds between retries
    
    # Alpha Vantage settings
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    ALPHA_VANTAGE_BASE_URL = 'https://www.alphavantage.co/query'
    ALPHA_VANTAGE_TIMEOUT = 30  # seconds
    ALPHA_VANTAGE_MAX_RETRIES = 2
    ALPHA_VANTAGE_RATE_LIMIT = 5  # requests per minute for free tier
    ALPHA_VANTAGE_DAILY_LIMIT = 25  # daily API calls for free tier

    
    # Twelve Data settings (future implementation)
    TWELVE_DATA_API_KEY = os.getenv('TWELVE_DATA_API_KEY')
    TWELVE_DATA_BASE_URL = 'https://api.twelvedata.com'
    TWELVE_DATA_RATE_LIMIT = 8  # requests per minute for free tier
    
    # Data validation thresholds
    MIN_DATA_POINTS = 100  # Minimum number of trading days required
    MAX_PRICE_CHANGE_THRESHOLD = 0.5  # 50% daily change threshold for outlier detection
    
    # ========================
    # TECHNICAL INDICATORS SETTINGS
    # ========================
    
    # Moving Averages periods
    MA_PERIODS = [5, 10, 20, 50, 200]
    
    # RSI settings
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    # MACD settings
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # Bollinger Bands settings
    BB_PERIOD = 20
    BB_STD_DEV = 2
    
    # Volatility calculation period
    VOLATILITY_PERIOD = 30  # days
    
    # ========================
    # EXCEL FORMATTING SETTINGS
    # ========================
    
    # Excel file naming
    EXCEL_FILE_PREFIX = "StockAnalysis"
    EXCEL_DATE_FORMAT = "%Y-%m-%d"
    
    # Sheet names
    SHEET_NAMES = {
        'raw_data': 'Raw Data',
        'technical_analysis': 'Technical Analysis', 
        'summary': 'Summary Dashboard',
        'charts': 'Charts & Visualizations',
        'quality_report': 'Data Quality Report'
    }
    
    # Excel styling
    EXCEL_COLORS = {
        'header_bg': 'D9E1F2',      # Light blue
        'positive': '00B04F',        # Green
        'negative': 'FF0000',        # Red
        'neutral': 'FFC000',         # Yellow/Orange
        'border': '366092'           # Dark blue
    }
    
    # Column widths for different data types
    EXCEL_COLUMN_WIDTHS = {
        'date': 12,
        'price': 10,
        'volume': 15,
        'percentage': 12,
        'indicator': 10,
        'text': 20
    }
    
    # ========================
    # LOGGING SETTINGS
    # ========================
    
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    # Log file settings
    LOG_FILE_MAX_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_FILE_BACKUP_COUNT = 5
    
    # ========================
    # API SETTINGS
    # ========================
    
    # ChatGPT API settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = 'gpt-3.5-turbo'
    OPENAI_MAX_TOKENS = 2000
    OPENAI_TEMPERATURE = 0.1
    
    # ========================
    # PERFORMANCE SETTINGS
    # ========================
    
    # Caching
    CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'True').lower() == 'true'
    CACHE_DURATION_HOURS = 24  # Cache data for 24 hours
    
    # Processing limits
    MAX_CONCURRENT_REQUESTS = 5
    REQUEST_TIMEOUT = 30  # seconds
    
    # Rate limiting
    RATE_LIMIT_CALLS = {}  # Track API calls for rate limiting
    
    # ========================
    # ERROR HANDLING SETTINGS
    # ========================
    
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
    RETRY_DELAY = 2  # seconds
    BACKOFF_MULTIPLIER = 2  # Exponential backoff
    
    # ========================
    # VALIDATION SETTINGS
    # ========================
    
    # Ticker validation
    VALID_TICKER_PATTERN = r'^[A-Z]{1,5}$'  # 1-5 uppercase letters
    COMMON_EXCHANGES = ['NASDAQ', 'NYSE', 'AMEX']
    
    # Data quality thresholds
    MIN_DATA_COMPLETENESS = 0.95  # 95% completeness required
    MAX_CONSECUTIVE_MISSING_DAYS = 5
    
    # ========================
    # OUTPUT SETTINGS
    # ========================

    # File formats to generate
    GENERATE_XLSX = True
    GENERATE_CSV = False  # For Phase 1, focus on Excel
    GENERATE_JSON = True  # Enable JSON for API phase

    # JSON export settings
    JSON_EXPORT_ENABLED = True

    # Excel chart settings
    CHART_WIDTH = 15
    CHART_HEIGHT = 10
    CHART_TITLE_FONT_SIZE = 14

    # ========================
    # API SETTINGS
    # ========================

    # Flask API configuration
    API_HOST = os.getenv('API_HOST', '127.0.0.1')
    API_PORT = int(os.getenv('API_PORT', '5000'))
    API_DEBUG = os.getenv('API_DEBUG', 'False').lower() == 'true'

    # CORS configuration
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')

    # Data freshness thresholds
    CACHE_FRESHNESS_HOURS = int(os.getenv('CACHE_FRESHNESS_HOURS', '24'))
    INCREMENTAL_UPDATE_THRESHOLD_HOURS = int(os.getenv('INCREMENTAL_UPDATE_THRESHOLD_HOURS', '24'))

    # API response settings
    API_TIMEOUT_SECONDS = int(os.getenv('API_TIMEOUT_SECONDS', '300'))
    MAX_CONCURRENT_ANALYSIS = int(os.getenv('MAX_CONCURRENT_ANALYSIS', '3'))

    # ========================
    # TIER SYSTEM SETTINGS
    # ========================

    # Deployment tier ('free' or 'premium')
    TIER = os.getenv('TIER', 'premium').lower()

    # Free tier stock restrictions
    FREE_TIER_STOCKS = [
        'AAPL', 'ABNB', 'ADDYY', 'AMD', 'AMZN', 'APLT', 'BABA', 'COF',
        'DVLT', 'GOOG', 'GOOGL', 'INTC', 'META', 'MSFT', 'NFLX', 'NKE',
        'NVDA', 'PLTR', 'RDDT', 'TSLA', 'WOLF', 'ZYBT'
    ]

    # Override free tier stocks via environment variable
    @classmethod
    def get_allowed_stocks(cls):
        """Get list of allowed stocks based on tier and environment"""
        if cls.TIER == 'premium':
            return []  # Empty list means no restrictions for premium

        # Check for environment override
        env_stocks = os.getenv('ALLOWED_STOCKS')
        if env_stocks:
            return [stock.strip().upper() for stock in env_stocks.split(',')]

        return cls.FREE_TIER_STOCKS

    @classmethod
    def is_premium_tier(cls):
        """Check if running in premium tier"""
        return cls.TIER == 'premium'

    @classmethod
    def is_free_tier(cls):
        """Check if running in free tier"""
        return cls.TIER == 'free'
    
    # ========================
    # DEVELOPMENT SETTINGS
    # ========================
    
    DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'
    VERBOSE_LOGGING = DEBUG_MODE
    
    # Test data settings (for development)
    TEST_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    USE_SAMPLE_DATA = os.getenv('USE_SAMPLE_DATA', 'False').lower() == 'true'
    
    # ========================
    # FALLBACK SETTINGS
    # ========================

    # Sample data generation when APIs fail
    SAMPLE_DATA_ENABLED = os.getenv('USE_SAMPLE_DATA', 'False').lower() == 'true'
    SAMPLE_DATA_STOCKS = {
        'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology'},
        'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Technology'},
        'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology'},
        'AMZN': {'name': 'Amazon.com Inc.', 'sector': 'Consumer Discretionary'},
        'TSLA': {'name': 'Tesla Inc.', 'sector': 'Consumer Discretionary'},
        'META': {'name': 'Meta Platforms Inc.', 'sector': 'Technology'},
        'NVDA': {'name': 'NVIDIA Corporation', 'sector': 'Technology'},
        'JPM': {'name': 'JPMorgan Chase & Co.', 'sector': 'Financial'},
        'V': {'name': 'Visa Inc.', 'sector': 'Financial'},
        'JNJ': {'name': 'Johnson & Johnson', 'sector': 'Healthcare'}
    }

    
    @classmethod
    def get_output_filename(cls, ticker: str) -> str:
        """Generate standardized output filename for a given ticker."""
        timestamp = datetime.now().strftime(cls.EXCEL_DATE_FORMAT)
        return f"{cls.EXCEL_FILE_PREFIX}_{ticker}_{timestamp}.xlsx"
    
    @classmethod
    def get_log_filename(cls) -> str:
        """Generate log filename with current date."""
        date_str = datetime.now().strftime('%Y-%m-%d')
        return f"stock_analyzer_{date_str}.log"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present and valid."""
        try:
            # Check if output directory is writable
            test_file = cls.OUTPUT_DIR / "test_write.tmp"
            test_file.touch()
            test_file.unlink()
            
            # Validate data period
            if cls.DATA_PERIOD_YEARS <= 0:
                raise ValueError("DATA_PERIOD_YEARS must be positive")
            
            # Validate technical indicator settings
            if cls.RSI_PERIOD <= 0:
                raise ValueError("RSI_PERIOD must be positive")
            
            if len(cls.MA_PERIODS) == 0:
                raise ValueError("MA_PERIODS cannot be empty")
            
            # Check API keys and provide helpful messages
            api_warnings = []
            
            if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY == 'your-openai-api-key-here':
                api_warnings.append("[WARNING] OpenAI API key not configured - AI insights will be unavailable")

            if not cls.ALPHA_VANTAGE_API_KEY or cls.ALPHA_VANTAGE_API_KEY == 'your-alpha-vantage-key-here':
                api_warnings.append("[WARNING] Alpha Vantage API key not configured - using Yahoo Finance only")
            
            # Only show warnings if not using sample data
            if api_warnings and not cls.SAMPLE_DATA_ENABLED:
                print("\nConfiguration Warnings:")
                for warning in api_warnings:
                    print(f"  {warning}")
                print("  [INFO] To use sample data, set USE_SAMPLE_DATA=True in .env file")
                print()
            
            # Check if at least one data source is available
            if not cls.SAMPLE_DATA_ENABLED:
                # Yahoo Finance doesn't need API key, so we're good
                pass
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Configuration validation failed: {e}")
            return False
    
    @classmethod
    def get_api_status(cls) -> dict:
        """Get the status of all configured APIs."""
        return {
            'openai': {
                'configured': bool(cls.OPENAI_API_KEY and cls.OPENAI_API_KEY != 'your-openai-api-key-here'),
                'name': 'OpenAI (ChatGPT)',
                'purpose': 'AI-powered insights'
            },
            'alpha_vantage': {
                'configured': bool(cls.ALPHA_VANTAGE_API_KEY and cls.ALPHA_VANTAGE_API_KEY != 'your-alpha-vantage-key-here'),
                'name': 'Alpha Vantage',
                'purpose': 'Backup stock data source'
            },
            'yahoo_finance': {
                'configured': True,  # Always available
                'name': 'Yahoo Finance',
                'purpose': 'Primary stock data source'
            }
        }
    
    @classmethod
    def print_config_summary(cls):
        """Print a summary of current configuration (useful for debugging)."""
        print(f"\n{cls.APP_NAME} v{cls.VERSION}")
        print("=" * 50)
        print(f"Data Period: {cls.DATA_PERIOD_YEARS} years ({cls.DATA_START_DATE} to {cls.DATA_END_DATE})")
        print(f"Output Directory: {cls.OUTPUT_DIR}")
        print(f"Primary Data Source: {cls.PRIMARY_DATA_SOURCE}")
        print(f"Data Source Priority: {', '.join(cls.DATA_SOURCES)}")
        print(f"Moving Averages: {cls.MA_PERIODS}")
        print(f"RSI Period: {cls.RSI_PERIOD}")
        print(f"Cache Enabled: {cls.CACHE_ENABLED}")
        print(f"Debug Mode: {cls.DEBUG_MODE}")
        
        # API Status
        print("\nAPI Configuration Status:")
        print(f"  OpenAI API: {'[OK] Configured' if cls.OPENAI_API_KEY else '[X] Not configured'}")
        print(f"  Alpha Vantage API: {'[OK] Configured' if cls.ALPHA_VANTAGE_API_KEY else '[X] Not configured'}")
        print(f"  Twelve Data API: {'[OK] Configured' if cls.TWELVE_DATA_API_KEY else '[X] Not configured'}")
        print("=" * 50)


# Create a global config instance
config = Config()

# Validate configuration on import
if not config.validate_config():
    raise RuntimeError("Configuration validation failed. Please check your settings.")