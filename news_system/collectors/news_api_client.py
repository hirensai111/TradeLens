# news_api_client.py
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import defaultdict

from news_system.collectors.api_config import api_config
from news_system.collectors.adapters import YFinanceAdapter, PolygonAdapter, NewsAPIAdapter

class NewsAPIClient:
    """Main client that orchestrates multiple news API adapters"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = api_config
        
        # Initialize adapters
        self.adapters = {}
        self._initialize_adapters()
        
        # Statistics
        self.stats = defaultdict(int)
        
    def _initialize_adapters(self):
        """Initialize all available API adapters"""
        
        # YFinance (no key needed)
        if self.config.api_settings['yfinance']['enabled']:
            self.adapters['yfinance'] = YFinanceAdapter()
            self.logger.info("Initialized YFinance adapter")
        
        # Polygon
        if self.config.api_settings['polygon']['enabled'] and self.config.keys['polygon']:
            self.adapters['polygon'] = PolygonAdapter(self.config.keys['polygon'])
            self.logger.info("Initialized Polygon adapter")
        
        # NewsAPI
        if self.config.api_settings['newsapi']['enabled'] and self.config.keys['newsapi']:
            self.adapters['newsapi'] = NewsAPIAdapter(self.config.keys['newsapi'])
            self.logger.info("Initialized NewsAPI adapter")
        
        # Alpha Vantage (implement later)
        # if self.config.api_settings['alphavantage']['enabled'] and self.config.keys['alphavantage']:
        #     self.adapters['alphavantage'] = AlphaVantageAdapter(self.config.keys['alphavantage'])
    
    def get_news(self, ticker: str, date: datetime, 
                 lookback_days: int = 3,
                 min_articles: int = 5,
                 sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get news for a ticker around a specific date.
        Tries multiple APIs until minimum article count is reached.
        """
        all_articles = []
        used_sources = []
        
        # Determine which adapters to use based on date
        days_from_now = (datetime.now().date() - date.date()).days
        
        if days_from_now <= 1:
            # For very recent news, prioritize real-time sources
            priority_order = ['yfinance', 'polygon', 'newsapi']
        elif days_from_now <= 30:
            # For recent but not today, use APIs with good recent coverage
            priority_order = ['polygon', 'newsapi', 'yfinance']
        else:
            # For historical news, only use APIs that support it
            priority_order = ['polygon']  # NewsAPI free tier only goes back 30 days
        
        # Filter by requested sources if specified
        if sources:
            priority_order = [s for s in priority_order if s in sources]
        
        # Try each adapter in priority order
        for adapter_name in priority_order:
            if adapter_name not in self.adapters:
                continue
            
            try:
                self.logger.info(f"Trying {adapter_name} for {ticker} on {date.date()}")
                adapter = self.adapters[adapter_name]
                
                # Fetch articles
                articles = adapter.get_news(ticker, date, lookback_days)
                
                if articles:
                    all_articles.extend(articles)
                    used_sources.append(adapter_name)
                    self.stats[f'{adapter_name}_success'] += 1
                    self.logger.info(f"Got {len(articles)} articles from {adapter_name}")
                    
                    # Check if we have enough articles
                    if len(all_articles) >= min_articles:
                        break
                else:
                    self.stats[f'{adapter_name}_empty'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error with {adapter_name}: {e}")
                self.stats[f'{adapter_name}_error'] += 1
        
        # Remove duplicates
        unique_articles = self._deduplicate_articles(all_articles)
        
        self.logger.info(f"Total unique articles: {len(unique_articles)} from sources: {used_sources}")
        return unique_articles
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on title similarity"""
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            title = article.get('title', '').lower().strip()
            
            # Simple deduplication - could be enhanced with fuzzy matching
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
        
        return unique_articles
    
    def get_sentiment_analysis(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment of articles (placeholder for integration with your sentiment analyzer)
        """
        # This is where you'd integrate your existing sentiment analyzer
        # For now, return a simple summary
        return {
            'total_articles': len(articles),
            'sources': list(set(a.get('api_source', 'unknown') for a in articles)),
            'date_range': self._get_date_range(articles),
            'needs_sentiment_analysis': True
        }
    
    def _get_date_range(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get the date range of articles"""
        if not articles:
            return {'earliest': None, 'latest': None}
        
        dates = [a['published_date'] for a in articles if a.get('published_date')]
        if not dates:
            return {'earliest': None, 'latest': None}
        
        return {
            'earliest': min(dates),
            'latest': max(dates)
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Get usage statistics"""
        return dict(self.stats)

# Create global instance
news_client = NewsAPIClient()