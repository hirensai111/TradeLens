# api_adapters/base_adapter.py
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional

class BaseAPIAdapter(ABC):
    """Base class for all news API adapters"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.name = self.__class__.__name__
        
    @abstractmethod
    def get_news(self, ticker: str, date: datetime, 
                 lookback_days: int = 3) -> List[Dict[str, Any]]:
        """Fetch news for a ticker around a specific date"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the API is accessible"""
        pass
    
    def normalize_article(self, raw_article: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize article to common format"""
        return {
            'title': '',
            'summary': '',
            'url': '',
            'source': '',
            'published_date': None,
            'ticker': '',
            'relevance_score': 0.5,
            'sentiment': None,
            'api_source': self.name,
            'raw_data': raw_article
        }