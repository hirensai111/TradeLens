"""
NewsAPI collector for Phase 3 News Intelligence Engine.

This module collects news articles from NewsAPI.org, one of the most comprehensive
news aggregation services. It supports multiple endpoints and search queries.
"""

import os
import re
import requests
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Iterator
from urllib.parse import urlencode

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from news_system.collectors.base_collector import BaseCollector, CollectionConfig
from news_system.database import NewsArticle

# Configure logging
logger = logging.getLogger(__name__)

class NewsAPICollector(BaseCollector):
    """NewsAPI.org collector for gathering news articles."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 config: Optional[CollectionConfig] = None,
                 search_queries: Optional[List[str]] = None,
                 sources: Optional[List[str]] = None,
                 categories: Optional[List[str]] = None):
        """
        Initialize NewsAPI collector.
        
        Args:
            api_key: NewsAPI.org API key (or set NEWSAPI_KEY env var)
            config: Collection configuration
            search_queries: List of search terms/queries
            sources: List of news source IDs (e.g., 'bbc-news', 'cnn')
            categories: List of categories ('business', 'technology', etc.)
        """
        # Set API key first
        self.api_key = api_key or os.getenv('NEWSAPI_KEY')
        
        # Set default timeout
        self.timeout_seconds = getattr(config, 'timeout_seconds', 30) if config else 30
        
        # Initialize basic attributes before calling super()
        self.base_url = "https://newsapi.org/v2"
        self.http_session = requests.Session()
        
        # Configure HTTP session
        if self.api_key:
            self.http_session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'User-Agent': 'NewsIntelligence/1.0'
            })
        else:
            # Use demo key for testing
            self.http_session.headers.update({
                'User-Agent': 'NewsIntelligence/1.0 (Demo Mode)'
            })
            logger.warning("No API key provided. Using demo mode with limited functionality.")
        
        # Default search configuration
        self.search_queries = search_queries or [
            'stock market',
            'financial news',
            'earnings report',
            'IPO',
            'merger acquisition',
            'Federal Reserve',
            'cryptocurrency bitcoin',
            'economic indicators',
            'tech stocks FAANG',
            'market volatility'
        ]
        
        self.sources = sources or [
            'bloomberg',
            'reuters',
            'financial-times',
            'wall-street-journal',
            'cnbc',
            'marketwatch',
            'business-insider',
            'forbes',
            'techcrunch',
            'the-verge'
        ]
        
        self.categories = categories or ['business', 'technology']
        
        # Stock symbol patterns for extraction
        self.stock_pattern = re.compile(r'\b[A-Z]{1,5}\b(?:\s+(?:stock|shares|ticker))?', re.IGNORECASE)
        self.common_stock_symbols = {
            'AAPL', 'GOOGL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA',
            'JPM', 'JNJ', 'WMT', 'PG', 'UNH', 'DIS', 'HD', 'MA', 'PYPL',
            'ADBE', 'CRM', 'NFLX', 'INTC', 'AMD', 'QCOM', 'TXN', 'AVGO',
            'ORCL', 'IBM', 'CSCO', 'ACN', 'INTU', 'NOW', 'SHOP', 'ZM'
        }
        
        # Initialize parent class
        super().__init__("NewsAPI", config)
    
    def test_connection(self) -> Dict[str, Any]:
        """Test NewsAPI connection."""
        try:
            # Use a simple request to test connection
            response = self.http_session.get(
                f"{self.base_url}/top-headlines",
                params={'country': 'us', 'pageSize': 1, 'apiKey': self.api_key} if self.api_key else {'country': 'us', 'pageSize': 1},
                timeout=10
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'status_code': response.status_code,
                    'rate_limit_remaining': response.headers.get('X-RateLimit-Remaining', 'Unknown'),
                    'api_key_valid': bool(self.api_key)
                }
            else:
                return {
                    'success': False,
                    'status_code': response.status_code,
                    'error': f'HTTP {response.status_code}',
                    'api_key_valid': bool(self.api_key)
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'api_key_valid': bool(self.api_key)
            }
    
    def collect_articles(self) -> Iterator[NewsArticle]:
        """Collect articles from NewsAPI using multiple strategies."""
        
        if not self.api_key:
            logger.warning("No API key available. Cannot collect articles.")
            return
        
        # Strategy 1: Search for specific queries
        for query in self.search_queries:
            try:
                yield from self._collect_by_search(query)
            except Exception as e:
                logger.error(f"Failed to collect articles for query '{query}': {e}")
        
        # Strategy 2: Collect from specific sources
        for source in self.sources:
            try:
                yield from self._collect_by_source(source)
            except Exception as e:
                logger.error(f"Failed to collect articles from source '{source}': {e}")
        
        # Strategy 3: Collect by category
        for category in self.categories:
            try:
                yield from self._collect_by_category(category)
            except Exception as e:
                logger.error(f"Failed to collect articles for category '{category}': {e}")
    
    def _collect_by_search(self, query: str, max_articles: int = 20) -> Iterator[NewsArticle]:
        """Collect articles using the everything endpoint with search query."""
        
        params = {
            'q': query,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': min(max_articles, 100),  # API max is 100
            'from': (datetime.now() - timedelta(days=1)).isoformat(),
            'to': datetime.now().isoformat(),
            'apiKey': self.api_key
        }
        
        url = f"{self.base_url}/everything"
        
        try:
            response = self.http_session.get(url, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'ok':
                logger.error(f"NewsAPI error for query '{query}': {data.get('message')}")
                return
            
            articles = data.get('articles', [])
            logger.info(f"Found {len(articles)} articles for query: {query}")
            
            for article_data in articles:
                article = self._parse_article(article_data, query_context=query)
                if article:
                    yield article
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for query '{query}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error collecting articles for query '{query}': {e}")
    
    def _collect_by_source(self, source: str, max_articles: int = 15) -> Iterator[NewsArticle]:
        """Collect articles from a specific news source."""
        
        params = {
            'sources': source,
            'language': 'en',
            'pageSize': min(max_articles, 100),
            'apiKey': self.api_key
        }
        
        url = f"{self.base_url}/top-headlines"
        
        try:
            response = self.http_session.get(url, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'ok':
                logger.error(f"NewsAPI error for source '{source}': {data.get('message')}")
                return
            
            articles = data.get('articles', [])
            logger.info(f"Found {len(articles)} articles from source: {source}")
            
            for article_data in articles:
                article = self._parse_article(article_data, source_context=source)
                if article:
                    yield article
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for source '{source}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error collecting articles from source '{source}': {e}")
    
    def _collect_by_category(self, category: str, max_articles: int = 15) -> Iterator[NewsArticle]:
        """Collect top headlines from a specific category."""
        
        params = {
            'category': category,
            'language': 'en',
            'country': 'us',
            'pageSize': min(max_articles, 100),
            'apiKey': self.api_key
        }
        
        url = f"{self.base_url}/top-headlines"
        
        try:
            response = self.http_session.get(url, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'ok':
                logger.error(f"NewsAPI error for category '{category}': {data.get('message')}")
                return
            
            articles = data.get('articles', [])
            logger.info(f"Found {len(articles)} articles in category: {category}")
            
            for article_data in articles:
                article = self._parse_article(article_data, category_context=category)
                if article:
                    yield article
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for category '{category}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error collecting articles from category '{category}': {e}")
    
    def _parse_article(self, article_data: Dict[str, Any], 
                      query_context: str = None,
                      source_context: str = None,
                      category_context: str = None) -> Optional[NewsArticle]:
        """Parse NewsAPI article data into NewsArticle object."""
        
        try:
            # Extract basic fields
            title = article_data.get('title', '').strip()
            description = article_data.get('description', '').strip()
            content = article_data.get('content', '').strip()
            url = article_data.get('url', '').strip()
            
            if not title or not url:
                return None
            
            # Combine description and content
            full_content = f"{description}\n\n{content}".strip()
            min_length = getattr(self.config, 'min_article_length', 100)
            if not full_content or len(full_content) < min_length:
                return None
            
            # Parse published date
            published_at = None
            if article_data.get('publishedAt'):
                try:
                    published_at = datetime.fromisoformat(
                        article_data['publishedAt'].replace('Z', '+00:00')
                    )
                    # Convert to naive datetime for consistency with database
                    published_at = published_at.replace(tzinfo=None)
                except ValueError:
                    logger.warning(f"Could not parse date: {article_data['publishedAt']}")
            
            # Extract source information
            source_info = article_data.get('source', {})
            source_name = source_info.get('name', self.source_name)
            
            # Extract author
            author = article_data.get('author')
            if author:
                author = author.strip()
                # Clean up common author format issues
                if author.lower().startswith('by '):
                    author = author[3:]
            
            # Extract stock symbols from title and content
            stock_symbols = self._extract_stock_symbols(f"{title} {full_content}")
            
            # Generate keywords based on context
            keywords = self._extract_keywords(title, full_content, query_context, category_context)
            
            # Determine event type based on content
            event_type = self._classify_event_type(title, full_content)
            
            return NewsArticle(
                title=title,
                content=full_content,
                summary=description,
                source=source_name,
                author=author,
                url=url,
                published_at=published_at,
                keywords=keywords,
                stock_symbols=stock_symbols,
                event_type=event_type
            )
            
        except Exception as e:
            logger.error(f"Failed to parse article: {e}")
            return None
    
    def _extract_stock_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text content."""
        if not text:
            return []
        
        # Find potential stock symbols
        potential_symbols = set()
        
        # Look for explicit mentions like "AAPL stock", "GOOGL shares"
        explicit_pattern = re.compile(r'\b([A-Z]{1,5})\s+(?:stock|shares|ticker|inc|corp)\b', re.IGNORECASE)
        for match in explicit_pattern.finditer(text):
            potential_symbols.add(match.group(1).upper())
        
        # Look for symbols in common formats
        symbol_contexts = [
            r'\$([A-Z]{1,5})\b',  # $AAPL
            r'\b([A-Z]{1,5}):\s*[A-Z]',  # AAPL: NASDAQ
            r'\(([A-Z]{1,5})\)',  # (AAPL)
        ]
        
        for pattern in symbol_contexts:
            for match in re.finditer(pattern, text):
                potential_symbols.add(match.group(1).upper())
        
        # Filter against known stock symbols
        valid_symbols = []
        for symbol in potential_symbols:
            if symbol in self.common_stock_symbols:
                valid_symbols.append(symbol)
        
        return list(set(valid_symbols))
    
    def _extract_keywords(self, title: str, content: str, 
                         query_context: str = None, 
                         category_context: str = None) -> List[str]:
        """Extract relevant keywords from article content."""
        keywords = set()
        
        # Add context-based keywords
        if query_context:
            keywords.update(query_context.lower().split())
        
        if category_context:
            keywords.add(category_context.lower())
        
        # Financial keywords
        financial_terms = [
            'earnings', 'revenue', 'profit', 'loss', 'growth', 'decline',
            'market', 'trading', 'investor', 'analyst', 'forecast',
            'ipo', 'merger', 'acquisition', 'dividend', 'buyback',
            'fed', 'interest rate', 'inflation', 'recession', 'bull market', 'bear market'
        ]
        
        text_lower = f"{title} {content}".lower()
        for term in financial_terms:
            if term in text_lower:
                keywords.add(term)
        
        # Limit keywords
        return list(keywords)[:10]
    
    def _classify_event_type(self, title: str, content: str) -> Optional[str]:
        """Classify the type of news event."""
        text_lower = f"{title} {content}".lower()
        
        # Breaking news indicators
        breaking_indicators = ['breaking', 'urgent', 'just in', 'developing']
        if any(indicator in text_lower for indicator in breaking_indicators):
            return 'breaking'
        
        # Earnings indicators
        earnings_indicators = ['earnings', 'quarterly results', 'q1', 'q2', 'q3', 'q4']
        if any(indicator in text_lower for indicator in earnings_indicators):
            return 'earnings'
        
        # M&A indicators
        ma_indicators = ['merger', 'acquisition', 'buyout', 'takeover']
        if any(indicator in text_lower for indicator in ma_indicators):
            return 'merger_acquisition'
        
        # IPO indicators
        if 'ipo' in text_lower or 'initial public offering' in text_lower:
            return 'ipo'
        
        # Fed/Central bank
        fed_indicators = ['federal reserve', 'fed', 'central bank', 'interest rate']
        if any(indicator in text_lower for indicator in fed_indicators):
            return 'monetary_policy'
        
        return 'general'
    
    def get_source_info(self) -> Dict[str, Any]:
        """Get NewsAPI source information."""
        return {
            'url': 'https://newsapi.org',
            'type': 'api',
            'reliability': 0.8,
            'description': 'News aggregation API service',
            'api_key_available': bool(self.api_key)
        }
    
    def get_api_status(self) -> Dict[str, Any]:
        """Check NewsAPI service status and usage."""
        try:
            # Use a simple request to check status
            params = {'country': 'us', 'pageSize': 1}
            if self.api_key:
                params['apiKey'] = self.api_key
            
            response = self.http_session.get(
                f"{self.base_url}/top-headlines",
                params=params,
                timeout=10
            )
            
            return {
                'status': 'ok' if response.status_code == 200 else 'error',
                'status_code': response.status_code,
                'rate_limit_remaining': response.headers.get('X-RateLimit-Remaining'),
                'rate_limit_reset': response.headers.get('X-RateLimit-Reset'),
                'response_time': response.elapsed.total_seconds(),
                'api_key_available': bool(self.api_key)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'api_key_available': bool(self.api_key)
            }

# Example usage and testing
if __name__ == "__main__":
    # Test NewsAPI collector
    collector = NewsAPICollector()
    
    print("=== NewsAPI Collector Test ===")
    print(f"API Key Available: {bool(collector.api_key)}")
    
    # Test API status
    status = collector.get_api_status()
    print(f"API Status: {status}")
    
    # Test connection
    connection_test = collector.test_connection()
    print(f"Connection test: {connection_test}")
    
    if connection_test['success']:
        # Run a small collection test
        print("\nRunning test collection...")
        result = collector.run_collection()
        
        print(f"Collection result:")
        print(f"  Success: {result.success}")
        print(f"  Articles collected: {result.articles_collected}")
        print(f"  Articles processed: {result.articles_processed}")
        print(f"  Articles deduplicated: {result.articles_deduplicated}")
        print(f"  Collection time: {result.collection_time:.2f}s")
        
        if result.errors:
            print(f"  Errors: {result.errors}")
        
        # Get collection stats
        stats = collector.get_collection_stats()
        print(f"Collection stats: {stats}")
    
    else:
        print("NewsAPI connection failed. Check your API key in .env file")
        print("Make sure you have: NEWSAPI_KEY=your_api_key_here")