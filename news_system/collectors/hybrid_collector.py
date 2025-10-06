"""
Hybrid NewsAPI + Polygon collector that automatically switches to Polygon
when NewsAPI hits rate limits and supports dynamic ticker search.
"""

import os
import re
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Iterator

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from news_system.collectors.base_collector import BaseCollector, CollectionConfig
from news_system.database import NewsArticle

logger = logging.getLogger(__name__)

class HybridNewsCollector(BaseCollector):
    """Hybrid collector that uses NewsAPI with Polygon fallback and dynamic ticker search."""
    
    def __init__(self, config: Optional[CollectionConfig] = None):
        """Initialize hybrid collector."""
        # API keys from environment
        self.newsapi_key = os.getenv('NEWSAPI_KEY')
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        
        # Set timeout
        self.timeout_seconds = getattr(config, 'timeout_seconds', 30) if config else 30
        
        # API endpoints
        self.newsapi_base = "https://newsapi.org/v2"
        self.polygon_base = "https://api.polygon.io/v2/reference/news"
        
        # Initialize sessions
        self.newsapi_session = requests.Session()
        self.polygon_session = requests.Session()
        
        # Configure NewsAPI session
        if self.newsapi_key:
            self.newsapi_session.headers.update({
                'Authorization': f'Bearer {self.newsapi_key}',
                'User-Agent': 'NewsIntelligence/1.0'
            })
        
        # Configure Polygon session
        self.polygon_session.headers.update({
            'User-Agent': 'NewsIntelligence/1.0 (Polygon)'
        })
        
        # Rate limiting
        self.newsapi_last_request = 0
        self.polygon_last_request = 0
        self.newsapi_rate_limit = 1.0  # 1 second
        self.polygon_rate_limit = 12.0  # 12 seconds (5 requests/minute for free tier)
        
        # Status tracking
        self.newsapi_available = bool(self.newsapi_key)
        self.polygon_available = bool(self.polygon_key)
        self.newsapi_rate_limited = False
        
        # Dynamic search configuration
        self.search_ticker = None
        self.search_keywords = []
        
        # Collection targets
        self.financial_keywords = [
            'earnings', 'stock market', 'IPO', 'merger', 'acquisition',
            'Federal Reserve', 'inflation', 'recession', 'bull market',
            'bear market', 'cryptocurrency', 'bitcoin', 'economic indicators'
        ]
        
        self.major_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']
        
        # Stock symbols for filtering
        self.common_stock_symbols = {
            'AAPL', 'GOOGL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA',
            'JPM', 'JNJ', 'WMT', 'PG', 'UNH', 'DIS', 'HD', 'MA', 'PYPL',
            'ADBE', 'CRM', 'NFLX', 'INTC', 'AMD', 'QCOM', 'TXN', 'AVGO'
        }
        
        # Initialize parent class AFTER setting attributes
        super().__init__("HybridNews", config)
    
    def set_search_ticker(self, ticker: str):
        """Set the ticker to search for in fresh collection."""
        self.search_ticker = ticker.upper()
        self.search_keywords = [
            ticker.upper(), 
            f"{ticker.upper()} stock", 
            f"{ticker.upper()} earnings", 
            f"{ticker.upper()} news"
        ]
        logger.info(f"Hybrid collector configured to search for: {ticker.upper()}")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test both API connections."""
        results = {
            'newsapi': {'available': self.newsapi_available},
            'polygon': {'available': self.polygon_available}
        }
        
        # Debug: Print API key status
        logger.debug(f"NewsAPI key present: {bool(self.newsapi_key)}")
        logger.debug(f"Polygon key present: {bool(self.polygon_key)}")
        
        # Test NewsAPI
        if self.newsapi_key:
            try:
                response = self.newsapi_session.get(
                    f"{self.newsapi_base}/top-headlines",
                    params={'country': 'us', 'pageSize': 1, 'apiKey': self.newsapi_key},
                    timeout=10
                )
                results['newsapi'].update({
                    'status': 'ok' if response.status_code == 200 else 'rate_limited' if response.status_code == 429 else 'error',
                    'status_code': response.status_code,
                    'rate_limit_remaining': response.headers.get('X-RateLimit-Remaining')
                })
                self.newsapi_rate_limited = (response.status_code == 429)
            except Exception as e:
                results['newsapi']['error'] = str(e)
        
        # Test Polygon
        if self.polygon_key:
            try:
                response = self.polygon_session.get(
                    self.polygon_base,
                    params={'apikey': self.polygon_key, 'limit': 1},
                    timeout=10
                )
                logger.debug(f"Polygon response status: {response.status_code}")
                
                results['polygon'].update({
                    'status': 'ok' if response.status_code == 200 else 'error',
                    'status_code': response.status_code
                })
            except Exception as e:
                logger.debug(f"Polygon error: {e}")
                results['polygon']['error'] = str(e)
        else:
            results['polygon']['error'] = 'No API key found'
        
        return results
    
    def collect_articles(self) -> Iterator[NewsArticle]:
        """Collect articles using hybrid approach with dynamic ticker search."""
        
        # Check API status first
        connection_status = self.test_connection()
        
        newsapi_working = (connection_status.get('newsapi', {}).get('status') == 'ok')
        polygon_working = (connection_status.get('polygon', {}).get('status') == 'ok')
        
        logger.info(f"API Status - NewsAPI: {newsapi_working}, Polygon: {polygon_working}")
        if self.search_ticker:
            logger.info(f"Searching specifically for: {self.search_ticker}")
        
        articles_collected = 0
        
        # Strategy 1: Try NewsAPI first (if available and not rate limited)
        if newsapi_working and not self.newsapi_rate_limited:
            logger.info("Attempting NewsAPI collection...")
            try:
                for article in self._collect_from_newsapi():
                    yield article
                    articles_collected += 1
                    if articles_collected >= 20:  # Limit to avoid rate limiting
                        break
            except Exception as e:
                logger.warning(f"NewsAPI collection failed: {e}")
                self.newsapi_rate_limited = True
        
        # Strategy 2: Use Polygon (if available)
        if polygon_working and (not newsapi_working or self.newsapi_rate_limited or articles_collected < 10):
            logger.info("Using Polygon collection...")
            try:
                for article in self._collect_from_polygon():
                    yield article
                    articles_collected += 1
                    if articles_collected >= 50:  # Polygon has higher limits
                        break
            except Exception as e:
                logger.error(f"Polygon collection failed: {e}")
        
        # Strategy 3: If NewsAPI is rate limited but we have the key, try a minimal collection
        if not newsapi_working and self.newsapi_key and not polygon_working:
            logger.info("Both APIs unavailable, trying minimal NewsAPI collection...")
            try:
                for article in self._collect_minimal_newsapi():
                    yield article
                    articles_collected += 1
                    if articles_collected >= 5:
                        break
            except Exception as e:
                logger.error(f"Minimal NewsAPI collection failed: {e}")
        
        logger.info(f"Total articles collected: {articles_collected}")
        
        # If we got nothing, that's still a valid result
        if articles_collected == 0:
            logger.warning("No articles collected from any source")
    
    def _collect_from_newsapi(self) -> Iterator[NewsArticle]:
        """Collect from NewsAPI with rate limiting and dynamic ticker search."""
        
        # Priority 1: Search for specific ticker if set
        search_queries = []
        if self.search_ticker:
            search_queries.extend([
                self.search_ticker,
                f"{self.search_ticker} stock",
                f"{self.search_ticker} earnings",
                f"{self.search_ticker} news"
            ])
            logger.info(f"NewsAPI: Searching for ticker-specific queries: {search_queries}")
        
        # Priority 2: General financial queries
        search_queries.extend(['earnings report', 'stock market', 'financial news'])
        
        for query in search_queries:
            try:
                self._rate_limit_newsapi()
                
                params = {
                    'q': query,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 10,
                    'from': (datetime.now() - timedelta(hours=6)).isoformat(),
                    'apiKey': self.newsapi_key
                }
                
                response = self.newsapi_session.get(
                    f"{self.newsapi_base}/everything",
                    params=params,
                    timeout=self.timeout_seconds
                )
                
                if response.status_code == 429:
                    logger.warning("NewsAPI rate limited, switching to Polygon")
                    self.newsapi_rate_limited = True
                    break
                
                response.raise_for_status()
                data = response.json()
                
                if data.get('status') == 'ok':
                    articles = data.get('articles', [])
                    logger.info(f"NewsAPI: Found {len(articles)} articles for '{query}'")
                    
                    for article_data in articles:
                        article = self._parse_newsapi_article(article_data, query)
                        if article:
                            yield article
                
            except Exception as e:
                logger.error(f"NewsAPI query '{query}' failed: {e}")
                continue
    
    def _collect_from_polygon(self) -> Iterator[NewsArticle]:
        """Collect from Polygon with rate limiting and dynamic ticker search."""
        
        # Strategy 1: Search for specific ticker if set
        if self.search_ticker:
            try:
                self._rate_limit_polygon()
                
                params = {
                    'apikey': self.polygon_key,
                    'ticker': self.search_ticker,
                    'limit': 15,
                    'published_utc.gte': (datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%d'),
                    'order': 'desc'
                }
                
                logger.info(f"Polygon: Searching for ticker-specific articles: {self.search_ticker}")
                
                response = self.polygon_session.get(
                    self.polygon_base,
                    params=params,
                    timeout=self.timeout_seconds
                )
                
                response.raise_for_status()
                data = response.json()
                
                if data.get('status') == 'OK':
                    articles = data.get('results', [])
                    logger.info(f"Polygon: Found {len(articles)} articles for {self.search_ticker}")
                    
                    for article_data in articles:
                        article = self._parse_polygon_article(article_data, self.search_ticker)
                        if article:
                            yield article
            
            except Exception as e:
                logger.error(f"Polygon search for {self.search_ticker} failed: {e}")
        
        # Strategy 2: General financial news
        try:
            self._rate_limit_polygon()
            
            params = {
                'apikey': self.polygon_key,
                'limit': 10,
                'published_utc.gte': (datetime.now() - timedelta(hours=12)).strftime('%Y-%m-%d'),
                'order': 'desc'
            }
            
            response = self.polygon_session.get(
                self.polygon_base,
                params=params,
                timeout=self.timeout_seconds
            )
            
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'OK':
                articles = data.get('results', [])
                logger.info(f"Polygon: Found {len(articles)} general articles")
                
                for article_data in articles:
                    article = self._parse_polygon_article(article_data)
                    if article:
                        yield article
        
        except Exception as e:
            logger.error(f"Polygon general collection failed: {e}")
        
        # Strategy 3: Only search major tickers if no specific ticker set
        if not self.search_ticker:
            for ticker in self.major_tickers[:2]:  # Reduced to 2 to avoid rate limits
                try:
                    self._rate_limit_polygon()
                    
                    params = {
                        'apikey': self.polygon_key,
                        'ticker': ticker,
                        'limit': 3,
                        'published_utc.gte': (datetime.now() - timedelta(hours=12)).strftime('%Y-%m-%d'),
                        'order': 'desc'
                    }
                    
                    response = self.polygon_session.get(
                        self.polygon_base,
                        params=params,
                        timeout=self.timeout_seconds
                    )
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    if data.get('status') == 'OK':
                        articles = data.get('results', [])
                        logger.info(f"Polygon: Found {len(articles)} articles for {ticker}")
                        
                        for article_data in articles:
                            article = self._parse_polygon_article(article_data, ticker)
                            if article:
                                yield article
                
                except Exception as e:
                    logger.error(f"Polygon ticker collection for {ticker} failed: {e}")
                    continue
    
    def _collect_minimal_newsapi(self) -> Iterator[NewsArticle]:
        """Minimal NewsAPI collection when rate limited."""
        try:
            self._rate_limit_newsapi()
            
            # Use specific ticker if set, otherwise use general query
            search_query = self.search_ticker if self.search_ticker else 'stock market'
            
            params = {
                'q': search_query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 5,
                'from': (datetime.now() - timedelta(hours=2)).isoformat(),
                'apiKey': self.newsapi_key
            }
            
            response = self.newsapi_session.get(
                f"{self.newsapi_base}/everything",
                params=params,
                timeout=self.timeout_seconds
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'ok':
                    articles = data.get('articles', [])
                    logger.info(f"Minimal NewsAPI: Found {len(articles)} articles for '{search_query}'")
                    
                    for article_data in articles:
                        article = self._parse_newsapi_article(article_data, search_query)
                        if article:
                            yield article
        except Exception as e:
            logger.error(f"Minimal NewsAPI collection failed: {e}")
    
    def _rate_limit_newsapi(self):
        """Rate limit NewsAPI requests."""
        current_time = time.time()
        time_since_last = current_time - self.newsapi_last_request
        if time_since_last < self.newsapi_rate_limit:
            time.sleep(self.newsapi_rate_limit - time_since_last)
        self.newsapi_last_request = time.time()
    
    def _rate_limit_polygon(self):
        """Rate limit Polygon requests."""
        current_time = time.time()
        time_since_last = current_time - self.polygon_last_request
        if time_since_last < self.polygon_rate_limit:
            time.sleep(self.polygon_rate_limit - time_since_last)
        self.polygon_last_request = time.time()
    
    def _parse_newsapi_article(self, article_data: Dict[str, Any], query: str) -> Optional[NewsArticle]:
        """Parse NewsAPI article."""
        try:
            title = article_data.get('title', '').strip()
            description = article_data.get('description', '').strip()
            content = article_data.get('content', '').strip()
            url = article_data.get('url', '').strip()
            
            if not title or not url:
                return None
            
            full_content = f"{description}\n\n{content}".strip()
            if len(full_content) < 100:
                return None
            
            # Parse date
            published_at = None
            if article_data.get('publishedAt'):
                try:
                    published_at = datetime.fromisoformat(
                        article_data['publishedAt'].replace('Z', '+00:00')
                    ).replace(tzinfo=None)
                except ValueError:
                    pass
            
            # Extract source
            source_info = article_data.get('source', {})
            source_name = f"{source_info.get('name', 'NewsAPI')} (NewsAPI)"
            
            # Extract stock symbols
            stock_symbols = self._extract_stock_symbols(f"{title} {full_content}")
            
            # If we're searching for a specific ticker, ensure it's included
            if self.search_ticker and self.search_ticker not in stock_symbols:
                # Check if the ticker is mentioned in the content
                if self.search_ticker.lower() in f"{title} {full_content}".lower():
                    stock_symbols.append(self.search_ticker)
            
            # Generate keywords
            keywords = [query.lower()]
            keywords.extend(self._extract_financial_keywords(f"{title} {full_content}"))
            
            return NewsArticle(
                title=title,
                content=full_content,
                summary=description,
                source=source_name,
                author=article_data.get('author'),
                url=url,
                published_at=published_at or datetime.now(),
                keywords=keywords[:10],
                stock_symbols=stock_symbols,
                event_type=self._classify_event_type(title, full_content)
            )
            
        except Exception as e:
            logger.error(f"Failed to parse NewsAPI article: {e}")
            return None
    
    def _parse_polygon_article(self, article_data: Dict[str, Any], ticker: str = None) -> Optional[NewsArticle]:
        """Parse Polygon article."""
        try:
            title = article_data.get('title', '').strip()
            description = article_data.get('description', '').strip()
            url = article_data.get('article_url', '').strip()
            
            if not title or not url or len(description) < 100:
                return None
            
            # Parse date
            published_at = None
            if article_data.get('published_utc'):
                try:
                    published_at = datetime.fromisoformat(
                        article_data['published_utc'].replace('Z', '+00:00')
                    ).replace(tzinfo=None)
                except ValueError:
                    pass
            
            # Extract publisher
            publisher = article_data.get('publisher', {})
            source_name = f"{publisher.get('name', 'Polygon')} (Polygon)"
            
            # Extract stock symbols
            stock_symbols = self._extract_stock_symbols(f"{title} {description}")
            if ticker and ticker not in stock_symbols:
                stock_symbols.append(ticker)
            
            # If we're searching for a specific ticker, ensure it's included
            if self.search_ticker and self.search_ticker not in stock_symbols:
                if self.search_ticker.lower() in f"{title} {description}".lower():
                    stock_symbols.append(self.search_ticker)
            
            # Generate keywords
            keywords = []
            if ticker:
                keywords.append(ticker.lower())
            if self.search_ticker:
                keywords.append(self.search_ticker.lower())
            keywords.extend(self._extract_financial_keywords(f"{title} {description}"))
            
            return NewsArticle(
                title=title,
                content=description,
                summary=description,
                source=source_name,
                author=article_data.get('author'),
                url=url,
                published_at=published_at or datetime.now(),
                keywords=keywords[:10],
                stock_symbols=stock_symbols,
                event_type=self._classify_event_type(title, description)
            )
            
        except Exception as e:
            logger.error(f"Failed to parse Polygon article: {e}")
            return None
    
    def _extract_stock_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text."""
        if not text:
            return []
        
        symbols = set()
        
        # Pattern for explicit stock mentions
        patterns = [
            r'\$([A-Z]{1,5})\b',  # $AAPL
            r'\b([A-Z]{1,5})\s+(?:stock|shares|ticker)\b',  # AAPL stock
            r'\(([A-Z]{1,5})\)',  # (AAPL)
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                symbol = match.upper()
                if symbol in self.common_stock_symbols:
                    symbols.add(symbol)
        
        return list(symbols)
    
    def _extract_financial_keywords(self, text: str) -> List[str]:
        """Extract financial keywords from text."""
        keywords = []
        text_lower = text.lower()
        
        for keyword in self.financial_keywords:
            if keyword.lower() in text_lower:
                keywords.append(keyword.lower())
        
        return keywords
    
    def _classify_event_type(self, title: str, content: str) -> str:
        """Classify event type."""
        text_lower = f"{title} {content}".lower()
        
        if any(word in text_lower for word in ['earnings', 'quarterly', 'q1', 'q2', 'q3', 'q4']):
            return 'earnings'
        elif any(word in text_lower for word in ['merger', 'acquisition', 'buyout']):
            return 'merger_acquisition'
        elif 'ipo' in text_lower:
            return 'ipo'
        elif any(word in text_lower for word in ['fed', 'federal reserve', 'interest rate']):
            return 'monetary_policy'
        elif any(word in text_lower for word in ['breaking', 'urgent', 'just in']):
            return 'breaking'
        else:
            return 'general'
    
    def get_source_info(self) -> Dict[str, Any]:
        """Get source information."""
        return {
            'name': 'Hybrid News Collector',
            'type': 'hybrid_api',
            'newsapi_available': self.newsapi_available,
            'polygon_available': self.polygon_available,
            'search_ticker': self.search_ticker,
            'reliability': 0.85,
            'description': 'Hybrid collector using NewsAPI with Polygon fallback and dynamic ticker search'
        }

# Test the hybrid collector
if __name__ == "__main__":
    collector = HybridNewsCollector()
    
    print("=== Hybrid News Collector Test ===")
    print(f"NewsAPI Available: {collector.newsapi_available}")
    print(f"Polygon Available: {collector.polygon_available}")
    
    # Test with specific ticker
    test_ticker = "NVDA"
    collector.set_search_ticker(test_ticker)
    print(f"Search ticker set to: {test_ticker}")
    
    # Test connections
    connection_test = collector.test_connection()
    print(f"Connection Test: {connection_test}")
    
    # Run collection
    print(f"\nRunning hybrid collection for {test_ticker}...")
    result = collector.run_collection()
    
    print(f"Collection Results:")
    print(f"  Success: {result.success}")
    print(f"  Articles collected: {result.articles_collected}")
    print(f"  Processing time: {result.collection_time:.2f}s")
    
    if result.errors:
        print(f"  Errors: {result.errors}")