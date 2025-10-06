"""
Reddit collector for Phase 3 News Intelligence Engine.

This module collects posts and comments from financial subreddits to gauge
social sentiment and retail investor behavior around stocks and market events.
"""

import os
import re
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Iterator
from urllib.parse import urlencode

from news_system.collectors.base_collector import BaseCollector, CollectionConfig
from news_system.database import NewsArticle

# Configure logging
logger = logging.getLogger(__name__)

class RedditCollector(BaseCollector):
    """Reddit collector for financial social sentiment."""
    
    def __init__(self, 
                 subreddits: Optional[List[str]] = None,
                 config: Optional[CollectionConfig] = None,
                 use_official_api: bool = False,
                 reddit_credentials: Optional[Dict[str, str]] = None):
        """
        Initialize Reddit collector.
        
        Args:
            subreddits: List of subreddits to monitor
            config: Collection configuration
            use_official_api: Whether to use official Reddit API (requires credentials)
            reddit_credentials: Reddit API credentials if using official API
        """
        self.subreddits = subreddits or [
            'investing', 'stocks', 'SecurityAnalysis', 'ValueInvesting',
            'financialindependence', 'StockMarket', 'pennystocks', 'options',
            'wallstreetbets', 'SecurityAnalysis', 'investing_discussion',
            'economics', 'business', 'entrepreneur', 'technology'
        ]
        
        self.use_official_api = use_official_api
        self.reddit_credentials = reddit_credentials or {}
        
        # Initialize parent class first
        super().__init__("Reddit", config)
        
        # Set default timeout if not available in config
        self.timeout_seconds = getattr(self.config, 'timeout_seconds', 30)
        
        # Initialize HTTP session
        self.http_session = requests.Session()
        self.http_session.headers.update({
            'User-Agent': 'NewsIntelligence/1.0 (Financial Research Bot)'
        })
        
        # Reddit API endpoints
        if self.use_official_api:
            self.base_url = "https://oauth.reddit.com"
            self._authenticate()
        else:
            # Use public JSON endpoints (no authentication required)
            self.base_url = "https://www.reddit.com"
        
        # Stock mention patterns
        self.stock_patterns = [
            re.compile(r'\$([A-Z]{1,5})\b'),  # $AAPL
            re.compile(r'\b([A-Z]{2,5})\s+(?:stock|shares|calls|puts|options)\b'),  # AAPL stock
            re.compile(r'\b(?:ticker|symbol):\s*([A-Z]{1,5})\b', re.IGNORECASE),  # ticker: AAPL
        ]
        
        # Financial keywords for relevance scoring
        self.financial_keywords = {
            'earnings', 'revenue', 'profit', 'loss', 'beat', 'miss', 'guidance',
            'dividend', 'buyback', 'merger', 'acquisition', 'ipo', 'squeeze',
            'calls', 'puts', 'options', 'bull', 'bear', 'moon', 'crash',
            'dip', 'rally', 'pump', 'dump', 'hold', 'buy', 'sell', 'yolo',
            'dd', 'analysis', 'catalyst', 'breakout', 'support', 'resistance'
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
    
    def _authenticate(self):
        """Authenticate with Reddit API if using official API."""
        if not self.use_official_api:
            return
        
        if not all(key in self.reddit_credentials for key in ['client_id', 'client_secret', 'username', 'password']):
            raise ValueError("Reddit credentials required for official API access")
        
        # Get access token
        auth_data = {
            'grant_type': 'password',
            'username': self.reddit_credentials['username'],
            'password': self.reddit_credentials['password']
        }
        
        auth_headers = {
            'User-Agent': self.http_session.headers['User-Agent']
        }
        
        response = requests.post(
            'https://www.reddit.com/api/v1/access_token',
            data=auth_data,
            headers=auth_headers,
            auth=(self.reddit_credentials['client_id'], self.reddit_credentials['client_secret'])
        )
        
        if response.status_code == 200:
            token_data = response.json()
            self.http_session.headers['Authorization'] = f"bearer {token_data['access_token']}"
            logger.info("Successfully authenticated with Reddit API")
        else:
            raise Exception(f"Reddit authentication failed: {response.status_code}")
    
    def _rate_limit(self):
        """Implement rate limiting to avoid hitting Reddit's limits."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to Reddit."""
        try:
            self._rate_limit()
            response = self.http_session.get(
                f"{self.base_url}/r/investing/hot.json?limit=1",
                timeout=self.timeout_seconds
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'status_code': response.status_code,
                    'api_type': 'Official API' if self.use_official_api else 'Public JSON',
                    'rate_limit_remaining': response.headers.get('X-Ratelimit-Remaining', 'Unknown')
                }
            else:
                return {
                    'success': False,
                    'status_code': response.status_code,
                    'error': f'HTTP {response.status_code}'
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def collect_articles(self) -> Iterator[NewsArticle]:
        """Collect posts from financial subreddits."""
        
        for subreddit in self.subreddits:
            try:
                # Collect hot posts
                yield from self._collect_from_subreddit(subreddit, 'hot', limit=10)
                
                # Collect new posts
                yield from self._collect_from_subreddit(subreddit, 'new', limit=5)
                
                # Add delay between subreddits to respect rate limits
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to collect from r/{subreddit}: {e}")
    
    def _collect_from_subreddit(self, subreddit: str, sort: str = 'hot', limit: int = 25) -> Iterator[NewsArticle]:
        """Collect posts from a specific subreddit."""
        
        if self.use_official_api:
            url = f"{self.base_url}/r/{subreddit}/{sort}"
        else:
            url = f"{self.base_url}/r/{subreddit}/{sort}.json"
        
        params = {'limit': limit}
        
        try:
            self._rate_limit()
            response = self.http_session.get(url, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            
            data = response.json()
            
            # Handle different response formats
            if self.use_official_api:
                posts = data.get('data', {}).get('children', [])
            else:
                posts = data.get('data', {}).get('children', [])
            
            logger.info(f"Found {len(posts)} posts in r/{subreddit}/{sort}")
            
            for post_data in posts:
                post = post_data.get('data', {})
                article = self._parse_reddit_post(post, subreddit)
                
                if article and self._is_financially_relevant(article):
                    yield article
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for r/{subreddit}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error collecting from r/{subreddit}: {e}")
    
    def _parse_reddit_post(self, post: Dict[str, Any], subreddit: str) -> Optional[NewsArticle]:
        """Parse Reddit post data into NewsArticle object."""
        
        try:
            # Extract basic fields
            title = post.get('title', '').strip()
            selftext = post.get('selftext', '').strip()
            url = f"https://www.reddit.com{post.get('permalink', '')}"
            
            if not title:
                return None
            
            # Combine title and selftext for content
            content = f"{title}\n\n{selftext}".strip()
            
            # Skip if content is too short
            min_length = getattr(self.config, 'min_article_length', 50)
            if len(content) < min_length:
                return None
            
            # Parse timestamp
            created_utc = post.get('created_utc')
            published_at = datetime.fromtimestamp(created_utc) if created_utc else datetime.now()
            
            # Check if post is too old
            max_age_hours = getattr(self.config, 'max_article_age_hours', 24)
            max_age = datetime.now() - timedelta(hours=max_age_hours)
            if published_at < max_age:
                return None
            
            # Extract author
            author = post.get('author', 'unknown')
            if author in ['[deleted]', 'AutoModerator']:
                return None
            
            # Extract stock symbols
            stock_symbols = self._extract_stock_symbols(f"{title} {selftext}")
            
            # Generate keywords
            keywords = self._extract_reddit_keywords(title, selftext, subreddit)
            
            # Calculate social sentiment indicators
            upvotes = post.get('ups', 0)
            downvotes = post.get('downs', 0)
            comments_count = post.get('num_comments', 0)
            score = post.get('score', 0)
            
            # Create social engagement score
            engagement_score = min((upvotes + comments_count) / 100.0, 1.0)
            
            return NewsArticle(
                title=title,
                content=content,
                source=f"Reddit-r/{subreddit}",
                url=url,
                author=f"u/{author}",
                published_at=published_at,
                keywords=keywords,
                stock_symbols=stock_symbols,
                # Store social metrics in summary field (JSON-like string)
                summary=f"Score: {score}, Comments: {comments_count}, Engagement: {engagement_score:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Failed to parse Reddit post: {e}")
            return None
    
    def _extract_stock_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from Reddit text."""
        symbols = set()
        
        for pattern in self.stock_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                symbols.add(match.upper())
        
        # Filter out common false positives
        false_positives = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN',
            'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM',
            'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO',
            'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO',
            'USE', 'CEO', 'CFO', 'CTO', 'USA', 'USD', 'SEC', 'FDA', 'API',
            'ETF', 'IPO', 'GDP', 'CPI', 'ATH', 'ATL', 'YTD', 'EOD', 'AH'
        }
        
        # Known stock symbols (expand this list)
        known_symbols = {
            'AAPL', 'GOOGL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA',
            'JPM', 'JNJ', 'WMT', 'PG', 'UNH', 'DIS', 'HD', 'MA', 'PYPL',
            'ADBE', 'CRM', 'NFLX', 'INTC', 'AMD', 'QCOM', 'TXN', 'AVGO'
        }
        
        valid_symbols = []
        for symbol in symbols:
            # Include if it's a known symbol or matches stock symbol patterns
            if (symbol in known_symbols or 
                (len(symbol) >= 2 and len(symbol) <= 5 and symbol not in false_positives)):
                valid_symbols.append(symbol)
        
        return list(set(valid_symbols))
    
    def _extract_reddit_keywords(self, title: str, content: str, subreddit: str) -> List[str]:
        """Extract relevant keywords from Reddit content."""
        keywords = set()
        
        # Add subreddit as context
        keywords.add(f"r/{subreddit}")
        
        # Extract financial keywords
        text_lower = f"{title} {content}".lower()
        for keyword in self.financial_keywords:
            if keyword in text_lower:
                keywords.add(keyword)
        
        # Extract common Reddit terminology
        reddit_terms = {
            'dd': 'due diligence',
            'yolo': 'high risk trade',
            'diamond hands': 'holding strong',
            'paper hands': 'selling quickly',
            'to the moon': 'expecting big gains',
            'hodl': 'hold on for dear life',
            'fomo': 'fear of missing out',
            'btfd': 'buy the dip',
            'wsb': 'wallstreetbets',
            'tendies': 'profits',
            'stonks': 'stocks',
            'ape': 'retail investor'
        }
        
        for term, description in reddit_terms.items():
            if term in text_lower:
                keywords.add(description)
        
        return list(keywords)[:15]  # Limit to 15 keywords
    
    def _is_financially_relevant(self, article: NewsArticle) -> bool:
        """Check if Reddit post is financially relevant."""
        
        # Check for stock symbols
        if article.stock_symbols:
            return True
        
        # Check for financial keywords
        content_lower = f"{article.title} {article.content}".lower()
        financial_keyword_count = sum(1 for keyword in self.financial_keywords if keyword in content_lower)
        
        # Require at least 2 financial keywords for relevance
        if financial_keyword_count >= 2:
            return True
        
        # Check subreddit context
        if any(sub in article.source.lower() for sub in ['investing', 'stocks', 'securityanalysis']):
            return financial_keyword_count >= 1
        
        return False
    
    def get_source_info(self) -> Dict[str, Any]:
        """Get Reddit source information."""
        return {
            'url': 'https://www.reddit.com',
            'type': 'social_media',
            'reliability': 0.4,  # Lower reliability due to social media nature
            'description': f'Social sentiment from {len(self.subreddits)} financial subreddits',
            'api_type': 'Official API' if self.use_official_api else 'Public JSON'
        }
    
    def get_subreddit_stats(self) -> Dict[str, Any]:
        """Get statistics about monitored subreddits."""
        stats = {}
        
        for subreddit in self.subreddits[:5]:  # Check first 5 subreddits
            try:
                if self.use_official_api:
                    url = f"{self.base_url}/r/{subreddit}/about"
                else:
                    url = f"{self.base_url}/r/{subreddit}/about.json"
                
                self._rate_limit()
                response = self.http_session.get(url, timeout=self.timeout_seconds)
                if response.status_code == 200:
                    data = response.json()
                    about_data = data.get('data', {}) if not self.use_official_api else data
                    
                    stats[subreddit] = {
                        'subscribers': about_data.get('subscribers', 0),
                        'active_users': about_data.get('accounts_active', 0),
                        'title': about_data.get('title', '')
                    }
                else:
                    stats[subreddit] = {'error': f'HTTP {response.status_code}'}
                
            except Exception as e:
                stats[subreddit] = {'error': str(e)}
        
        return stats

# Example usage and testing
if __name__ == "__main__":
    # Test Reddit collector
    reddit_collector = RedditCollector(
        subreddits=['investing', 'stocks', 'SecurityAnalysis'],  # Test with smaller list
        config=CollectionConfig(max_articles_per_run=20, min_article_length=50)
    )
    
    print("=== Reddit Social Sentiment Collector Test ===\n")
    
    # Test connection
    connection_test = reddit_collector.test_connection()
    print(f"Connection test: {connection_test}")
    
    if connection_test['success']:
        # Get subreddit statistics
        print("\nSubreddit statistics:")
        stats = reddit_collector.get_subreddit_stats()
        for subreddit, data in stats.items():
            if 'error' not in data:
                print(f"  r/{subreddit}: {data.get('subscribers', 0):,} subscribers")
            else:
                print(f"  r/{subreddit}: {data['error']}")
        
        # Test collection
        print(f"\nTesting collection from Reddit...")
        result = reddit_collector.run_collection()
        
        print(f"Collection result:")
        print(f"  Success: {result.success}")
        print(f"  Articles collected: {result.articles_collected}")
        print(f"  Articles processed: {result.articles_processed}")
        print(f"  Collection time: {result.collection_time:.2f}s")
        
        if result.errors:
            print(f"  Errors: {result.errors}")
        
        # Get collection stats
        stats = reddit_collector.get_collection_stats()
        print(f"Collection stats: {stats}")
    
    else:
        print("Reddit connection failed - this is normal without API credentials")
        print("The collector will work in production with proper rate limiting")
    
    print("\nReddit collector test completed!")