# api_adapters/polygon_adapter.py
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from .base_adapter import BaseAPIAdapter
import logging
import time

class PolygonAdapter(BaseAPIAdapter):
    """Adapter for Polygon.io API"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.polygon.io"
        self.logger = logging.getLogger(__name__)
        self.last_request_time = 0
        self.rate_limit_delay = 12  # 5 requests per minute = 12 seconds between requests
        
    def get_news(self, ticker: str, date: datetime, 
                 lookback_days: int = 3) -> List[Dict[str, Any]]:
        """Fetch news for a ticker around a specific date"""
        if not self.api_key:
            self.logger.error("Polygon API key not provided")
            return []
            
        # Rate limiting
        self._rate_limit()
        
        # Calculate date range
        end_date = date + timedelta(days=1)
        start_date = date - timedelta(days=lookback_days)
        
        # Format dates for API
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Build URL
        url = f"{self.base_url}/v2/reference/news"
        params = {
            'ticker': ticker.upper(),
            'published_utc.gte': start_str,
            'published_utc.lte': end_str,
            'sort': 'published_utc',
            'limit': 50,
            'apiKey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = data.get('results', [])
            
            self.logger.info(f"Polygon returned {len(results)} articles for {ticker}")
            
            # Convert to our format
            articles = []
            for item in results:
                article = self.normalize_article(item)
                
                article['title'] = item.get('title', '')
                article['summary'] = item.get('description', '')[:500] if item.get('description') else ''
                article['url'] = item.get('article_url', '')
                article['source'] = item.get('publisher', {}).get('name', 'Unknown')
                article['ticker'] = ticker.upper()
                
                # Parse date
                pub_date = item.get('published_utc')
                if pub_date:
                    try:
                        article['published_date'] = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    except:
                        article['published_date'] = datetime.strptime(pub_date, '%Y-%m-%dT%H:%M:%S.%fZ')
                
                # Add relevance score if available
                if 'tickers' in item:
                    for t in item['tickers']:
                        if t == ticker.upper():
                            article['relevance_score'] = 0.9
                            break
                
                articles.append(article)
            
            return articles
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching from Polygon: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Test Polygon API connection"""
        if not self.api_key:
            return False
            
        url = f"{self.base_url}/v2/reference/news"
        params = {'limit': 1, 'apiKey': self.api_key}
        
        try:
            response = requests.get(url, params=params, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()