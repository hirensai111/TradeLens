# api_adapters/newsapi_adapter.py
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from .base_adapter import BaseAPIAdapter
import logging

class NewsAPIAdapter(BaseAPIAdapter):
    """Adapter for NewsAPI.org"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://newsapi.org/v2"
        self.logger = logging.getLogger(__name__)
        
    def get_news(self, ticker: str, date: datetime, 
                 lookback_days: int = 3) -> List[Dict[str, Any]]:
        """Fetch news for a ticker around a specific date"""
        if not self.api_key:
            self.logger.error("NewsAPI key not provided")
            return []
        
        # NewsAPI free tier only allows up to 1 month old articles
        max_date = datetime.now() - timedelta(days=30)
        if date < max_date:
            self.logger.warning(f"NewsAPI free tier only supports articles from last 30 days. Requested date {date.date()} is too old.")
            return []
        
        # Calculate date range
        from_date = (date - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        to_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Build query - search for company name and ticker
        company_names = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google Alphabet',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'META': 'Meta Facebook',
            'NVDA': 'Nvidia',
        }
        
        company_name = company_names.get(ticker.upper(), ticker)
        query = f'"{ticker}" OR "{company_name}" stock'
        
        # Build URL
        url = f"{self.base_url}/everything"
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 50,
            'apiKey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'ok':
                self.logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
            
            articles_data = data.get('articles', [])
            self.logger.info(f"NewsAPI returned {len(articles_data)} articles for {ticker}")
            
            # Convert to our format
            articles = []
            for item in articles_data:
                # Skip removed articles
                if '[Removed]' in item.get('title', '') or not item.get('title'):
                    continue
                    
                article = self.normalize_article(item)
                
                article['title'] = item.get('title', '')
                article['summary'] = item.get('description', '')
                article['url'] = item.get('url', '')
                article['source'] = item.get('source', {}).get('name', 'Unknown')
                article['ticker'] = ticker.upper()
                
                # Parse date
                pub_date = item.get('publishedAt')
                if pub_date:
                    try:
                        article['published_date'] = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    except:
                        continue
                
                # Simple relevance check
                title_lower = article['title'].lower()
                summary_lower = article['summary'].lower() if article['summary'] else ''
                
                if ticker.lower() in title_lower or company_name.lower() in title_lower or \
                   ticker.lower() in summary_lower or company_name.lower() in summary_lower:
                    article['relevance_score'] = 0.8
                else:
                    article['relevance_score'] = 0.5
                
                articles.append(article)
            
            return articles
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching from NewsAPI: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Test NewsAPI connection"""
        if not self.api_key:
            return False
            
        url = f"{self.base_url}/everything"
        params = {
            'q': 'test',
            'pageSize': 1,
            'apiKey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            return data.get('status') == 'ok'
        except:
            return False