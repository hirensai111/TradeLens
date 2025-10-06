# api_config.py
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class APIConfig:
    """Configuration for all news APIs"""
    
    def __init__(self):
        # Load .env file
        load_dotenv()
        
        # API Keys - Updated to match your .env file
        self.keys = {
            'polygon': os.getenv('POLYGON_API_KEY', ''),
            'alphavantage': os.getenv('ALPHA_VANTAGE_API_KEY', ''),  # Fixed: Added underscore
            'newsapi': os.getenv('NEWSAPI_API_KEY', ''),
        }
        
        # API Configuration
        self.api_settings = {
            'polygon': {
                'enabled': bool(self.keys['polygon']),
                'base_url': 'https://api.polygon.io',
                'rate_limit': 5,  # per minute
                'rate_window': 60,  # seconds
            },
            'alphavantage': {
                'enabled': bool(self.keys['alphavantage']),
                'base_url': 'https://www.alphavantage.co',
                'rate_limit': 5,  # per minute
                'rate_window': 60,
            },
            'newsapi': {
                'enabled': bool(self.keys['newsapi']),
                'base_url': 'https://newsapi.org/v2',
                'rate_limit': 100,  # per day
                'rate_window': 86400,  # 24 hours
            },
            'yfinance': {
                'enabled': True,
                'rate_limit': None,  # No official limit
            }
        }
        
        # General settings
        self.general = {
            'min_articles_required': 5,
            'max_apis_to_try': 3,
            'cache_enabled': True,
            'cache_duration_hours': 24,
            'historical_lookback_days': 3,
        }

api_config = APIConfig()