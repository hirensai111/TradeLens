# api_adapters/yfinance_adapter.py
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from .base_adapter import BaseAPIAdapter
import logging

class YFinanceAdapter(BaseAPIAdapter):
    """Adapter for Yahoo Finance (yfinance) news"""
    
    def __init__(self):
        super().__init__(api_key=None)  # No API key needed
        self.logger = logging.getLogger(__name__)
        
    def get_news(self, ticker: str, date: datetime, 
                 lookback_days: int = 7) -> List[Dict[str, Any]]:
        """
        Fetch recent news for a ticker.
        Note: yfinance only provides recent news (last 1-3 days typically)
        """
        try:
            # Create ticker object
            stock = yf.Ticker(ticker.upper())
            
            # Get news
            raw_news = stock.news
            
            self.logger.info(f"yfinance returned {len(raw_news) if raw_news else 0} articles for {ticker}")
            
            if not raw_news:
                return []
            
            # Check how old our request date is
            days_from_today = (datetime.now().date() - date.date()).days
            self.logger.info(f"Requested date is {days_from_today} days from today")
            
            # Convert to our format
            articles = []
            for item in raw_news:
                article = self.normalize_article(item)
                
                # Fill in the normalized fields
                article['title'] = item.get('title', '')
                article['summary'] = item.get('summary', '')
                article['url'] = item.get('link', '')
                article['source'] = item.get('publisher', 'Yahoo Finance')
                article['ticker'] = ticker.upper()
                
                # Parse published date
                pub_timestamp = item.get('providerPublishTime')
                
                if pub_timestamp:
                    article['published_date'] = datetime.fromtimestamp(pub_timestamp)
                    self.logger.debug(f"Article has timestamp: {article['published_date']}")
                else:
                    # If no timestamp, assume it's recent (today)
                    # This is common with yfinance
                    article['published_date'] = datetime.now()
                    self.logger.debug(f"No timestamp, assuming today for: {article['title'][:50]}...")
                
                # For YFinance, if we're looking for recent news (within last 3 days), include it
                if days_from_today <= 3:
                    articles.append(article)
                    self.logger.debug(f"Including recent article: {article['title'][:50]}...")
            
            self.logger.info(f"Returning {len(articles)} articles after filtering")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching news from yfinance: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Test yfinance connection"""
        try:
            ticker = yf.Ticker("AAPL")
            info = ticker.info
            return 'symbol' in info
        except:
            return False