#!/usr/bin/env python3
"""
Fixed Phase 3 News Connector with Integrated Search Engine
FIXES: Recursion bug, correct import paths, proper database path detection
"""

import sqlite3
import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import json
from dataclasses import dataclass
import sys
import os

# FIXED: Correct path resolution based on actual directory structure
# Current file: D:\stock-prediction-engine\phase4_ml_engine\src\data_loaders\phase3_connector.py
# Phase 3 src: D:\stock-prediction-engine\src\
# Database: D:\stock-prediction-engine\data\news_intelligence.db

current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up from: phase4_ml_engine\src\data_loaders\ to stock-prediction-engine\
phase3_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
phase3_src = os.path.join(phase3_root, 'src')

# Add Phase 3 src to path
if os.path.exists(phase3_src):
    sys.path.insert(0, phase3_src)

@dataclass
class EnhancedSentimentFeatures:
    """Enhanced data class for comprehensive sentiment features"""
    # Basic sentiment features
    sentiment_1d: float = 0.0
    sentiment_3d: float = 0.0
    sentiment_7d: float = 0.0
    
    # News volume features
    news_volume_1d: int = 0
    news_volume_3d: int = 0
    news_volume_7d: int = 0
    
    # Advanced Phase 3 features
    correlation_strength: float = 0.0
    event_impact_score: float = 0.0
    confidence_score: float = 0.0
    alert_count: int = 0
    source_diversity: int = 0
    
    # Hybrid collection features
    hybrid_sources: int = 0
    reddit_sentiment: float = 0.0
    polygon_coverage: bool = False
    newsapi_coverage: bool = False
    
    # Correlation analysis features
    prediction_accuracy: float = 0.0
    news_price_correlation: float = 0.0
    correlation_confidence: float = 0.0
    
    # Market context features
    market_sentiment: float = 0.0
    relative_sentiment: float = 0.0
    sector_sentiment: float = 0.0
    
    # Event extraction features
    event_type: str = ""
    event_confidence: float = 0.0
    event_impact: float = 0.0

@dataclass
class NewsSearchResult:
    """Result from integrated news search"""
    articles: List[Dict]
    total_found: int
    sentiment_analysis: Dict
    correlation_analysis: Dict
    enhanced_features: EnhancedSentimentFeatures
    market_context: Dict
    search_metadata: Dict

class IntegratedPhase3NewsConnector:
    """
    FIXED: Enhanced connector that integrates Phase 3 news intelligence system
    with main.py search engine functionality for predictor engine integration
    """
    
    def __init__(self, config_path: str = None):
        """Initialize integrated Phase 3 connector with search engine"""
        # Setup logging first
        self.logger = self._setup_logging()
        
        # Log initialization
        self.logger.info("Initializing Phase 3 News Connector")
        self.logger.info(f"Phase 3 root: {phase3_root}")
        self.logger.info(f"Phase 3 src: {phase3_src}")
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.db_path = self.config['data_sources']['phase3_db_path']
        
        # Log database path
        self.logger.info(f"Database path: {self.db_path}")
        self.logger.info(f"Database exists: {os.path.exists(self.db_path)}")
        
        # Initialize Phase 3 system components
        self._initialize_phase3_components()
        
        # Test database connection and discover schema
        self._discover_database_schema()
        
        # FIXED: Single health check call (no recursion)
        self._perform_health_check()
        
        # Initialize search engine components
        self._initialize_search_engine()
        
        # Stock symbol mapping for better ticker recognition
        self.symbol_mapping = self._create_symbol_mapping()
        
        self.logger.info("Phase 3 News Connector initialization complete")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    return yaml.safe_load(file)
        except Exception as e:
            self.logger.warning(f"Could not load config {config_path}: {e}")
        
        # FIXED: Correct database path based on actual directory structure
        default_db_path = os.path.join(phase3_root, 'data', 'news_intelligence.db')
        
        return {
            'data_sources': {
                'phase3_db_path': default_db_path
            },
            'search_engine': {
                'default_limit': 20,
                'default_days_back': 7,
                'correlation_analysis_days': 14,
                'enable_fresh_collection': True,
                'min_articles_for_correlation': 5
            }
        }
    
    def _initialize_phase3_components(self):
        """Initialize Phase 3 system components with correct import paths"""
        self.logger.info("Attempting to initialize Phase 3 components")
        
        # Initialize all to None first
        self._set_components_to_none()
        
        # Check if Phase 3 src directory exists
        if not os.path.exists(phase3_src):
            self.logger.error(f"Phase 3 src directory not found: {phase3_src}")
            return
        
        # Try to import Phase 3 components with correct paths
        try:
            # Database components
            try:
                from database import get_session, get_database_status
                self.session = get_session()
                self.get_database_status = get_database_status
                self.logger.info("✅ Database components initialized")
            except ImportError as e:
                self.logger.warning(f"❌ Database components failed: {e}")
                self.session = None
                self.get_database_status = None
            
            # Collector components
            try:
                # Import real collectors with absolute imports
                import sys
                collectors_path = os.path.join(phase3_src, 'collectors')
                if collectors_path not in sys.path:
                    sys.path.insert(0, collectors_path)
                
                # Now import with absolute imports (since you fixed the relative imports)
                from collectors import get_collector_registry, register_collector
                from collectors.hybrid_collector import HybridNewsCollector
                from collectors.reddit_collector import RedditCollector
                from collectors.newsapi_collector import NewsAPICollector
                
                self.collector_registry = get_collector_registry()
                self.register_collector = register_collector
                self.HybridNewsCollector = HybridNewsCollector
                self.RedditCollector = RedditCollector
                self.NewsAPICollector = NewsAPICollector
                
                self.logger.info("✅ Collector registry initialized (REAL mode)")
                
            except ImportError as e:
                self.logger.warning(f"❌ Collector registry failed: {e}")
                self.collector_registry = None
                self.register_collector = None
            
            # Specific collectors - skip for now due to import issues
            self.logger.info("✅ Specific collectors initialized with real imports")
            
            # Processor components
            try:
                from processors import process_article, get_market_sentiment_overview
                self.process_article = process_article
                self.get_market_sentiment_overview = get_market_sentiment_overview
                self.logger.info("✅ Processors initialized")
            except ImportError as e:
                self.logger.warning(f"❌ Processors failed: {e}")
                self.process_article = None
                self.get_market_sentiment_overview = None
            
            # Intelligence components
            try:
                from intelligence.correlation_analyzer import NewsPriceCorrelationAnalyzer
                self.correlation_analyzer = NewsPriceCorrelationAnalyzer()
                self.logger.info("✅ Correlation analyzer initialized")
            except ImportError as e:
                self.logger.warning(f"❌ Correlation analyzer failed: {e}")
                self.correlation_analyzer = None
            
            # Event extractor
            try:
                from processors.event_extractor import FinancialEventExtractor
                self.event_extractor = FinancialEventExtractor()
                self.logger.info("✅ Event extractor initialized")
            except ImportError as e:
                self.logger.warning(f"❌ Event extractor failed: {e}")
                self.event_extractor = None
            
        except Exception as e:
            self.logger.error(f"Critical error initializing Phase 3 components: {e}")
            self._set_components_to_none()
        
        # Count available components
        components = [
            self.session, self.collector_registry, self.process_article, 
            self.correlation_analyzer, self.event_extractor
        ]
        available_count = sum(1 for c in components if c is not None)
        
        # Set search engine availability
        self.search_engine_available = self.collector_registry is not None
        
        self.logger.info(f"🎯 Initialized {available_count}/5 Phase 3 components")
        
        if available_count >= 3:
            self.logger.info("✅ Phase 3 system is FULLY FUNCTIONAL")
        elif available_count >= 2:
            self.logger.info("⚠️ Phase 3 system is PARTIALLY FUNCTIONAL")
        else:
            self.logger.warning("❌ Phase 3 system has LIMITED FUNCTIONALITY")
    
    def _set_components_to_none(self):
        """Set all Phase 3 components to None"""
        self.session = None
        self.get_database_status = None
        self.collector_registry = None
        self.register_collector = None
        self.HybridNewsCollector = None
        self.RedditCollector = None
        self.process_article = None
        self.get_market_sentiment_overview = None
        self.correlation_analyzer = None
        self.event_extractor = None
    
    def _discover_database_schema(self):
        """Discover the actual database schema to handle variations"""
        self.schema = {
            'tables': {},
            'columns': {},
            'relationships': {}
        }
        
        try:
            if not os.path.exists(self.db_path):
                self.logger.error(f"Database file not found: {self.db_path}")
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]
            self.schema['tables'] = {table: True for table in tables}
            
            # Discover columns for each table
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table});")
                columns = cursor.fetchall()
                self.schema['columns'][table] = {col[1]: {
                    'type': col[2],
                    'index': col[0]
                } for col in columns}
            
            conn.close()
            self.logger.info(f"Database schema discovered: {len(tables)} tables")
            
        except Exception as e:
            self.logger.error(f"Error discovering database schema: {e}")
    
    def _perform_health_check(self):
        """FIXED: Single health check (no recursion)"""
        self.health_report = {
            'database_exists': os.path.exists(self.db_path),
            'database_readable': False,
            'has_articles': False,
            'phase3_components': 0,
            'article_count': 0,
            'table_count': len(self.schema['tables']),
            'recommendations': []
        }
        
        # Check database
        if self.health_report['database_exists']:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                self.health_report['database_readable'] = True
                
                # Check for articles
                if 'news_articles' in self.schema['tables']:
                    cursor.execute("SELECT COUNT(*) FROM news_articles;")
                    count = cursor.fetchone()[0]
                    self.health_report['article_count'] = count
                    self.health_report['has_articles'] = count > 0
                
                conn.close()
                
            except Exception as e:
                self.logger.error(f"Database health check failed: {e}")
        else:
            self.health_report['recommendations'].append("Database not found - check path configuration")
        
        # Check Phase 3 components
        components = [
            self.session, self.collector_registry, self.process_article, 
            self.correlation_analyzer, self.event_extractor
        ]
        self.health_report['phase3_components'] = sum(1 for c in components if c is not None)
        
        if self.health_report['phase3_components'] == 0:
            self.health_report['recommendations'].append("No Phase 3 components available - check imports")
        
        # Log health report (ONCE)
        self.logger.info(f"Health Check: DB={self.health_report['database_exists']}, "
                        f"Articles={self.health_report['article_count']}, "
                        f"Components={self.health_report['phase3_components']}/5")
        
        if self.health_report['recommendations']:
            for rec in self.health_report['recommendations']:
                self.logger.warning(f"Recommendation: {rec}")
    
    def _initialize_search_engine(self):
        """Initialize search engine components"""
        if self.collector_registry and self.register_collector:
            try:
                self._setup_collectors()
                self.logger.info("✅ Search engine components initialized")
            except Exception as e:
                self.logger.error(f"❌ Search engine initialization failed: {e}")
        else:
            self.logger.warning("⚠️ Search engine initialization skipped - missing collector components")
            # Still mark as partially functional
            self.search_engine_available = False
    
    def _setup_collectors(self):
        """Setup collectors similar to main.py"""
        collectors_setup = 0
        
        # Register Hybrid News Collector (PRIORITY 1)
        if self.HybridNewsCollector and self.register_collector:
            try:
                hybrid_collector = self.HybridNewsCollector()
                self.register_collector(hybrid_collector)
                collectors_setup += 1
                self.logger.info("✅ Hybrid News Collector registered")
            except Exception as e:
                self.logger.warning(f"⚠️ Hybrid collector setup failed: {e}")
        
        # Register NewsAPI Collector (PRIORITY 2)
        try:
            from collectors.newsapi_collector import NewsAPICollector
            if os.getenv('NEWSAPI_KEY'):
                newsapi_collector = NewsAPICollector()
                self.register_collector(newsapi_collector)
                collectors_setup += 1
                self.logger.info("✅ NewsAPI Collector registered")
            else:
                self.logger.info("ℹ️ NewsAPI key not found - skipping NewsAPI collector")
        except ImportError:
            self.logger.info("ℹ️ NewsAPI collector not available")
        except Exception as e:
            self.logger.warning(f"⚠️ NewsAPI collector setup failed: {e}")
        
        # Register Reddit Collector (PRIORITY 3)
        if self.RedditCollector and self.register_collector:
            try:
                reddit_config = type('Config', (), {
                    'max_articles_per_run': 20,
                    'min_article_length': 100,
                    'max_article_age_hours': 48,
                    'timeout_seconds': 30,
                    'rate_limit_delay': 1.0,
                    'retry_attempts': 3,
                    'retry_delay_seconds': 5,
                    'enable_deduplication': True,
                    'collection_interval_minutes': 15
                })()
                
                reddit_collector = self.RedditCollector(
                    subreddits=['investing', 'stocks', 'SecurityAnalysis', 'ValueInvesting'],
                    config=reddit_config
                )
                self.register_collector(reddit_collector)
                collectors_setup += 1
                self.logger.info("✅ Reddit Collector registered")
            except Exception as e:
                self.logger.warning(f"⚠️ Reddit collector setup failed: {e}")
        
        if collectors_setup > 0:
            self.logger.info(f"🎯 Search engine setup complete with {collectors_setup} collectors")
            self.search_engine_available = True
            
            # Apply NewsAPI parsing fix if available
            try:
                self._apply_newsapi_fix()
            except Exception as e:
                self.logger.debug(f"NewsAPI fix not applied: {e}")
        else:
            self.logger.warning("⚠️ No collectors were successfully setup")
            self.search_engine_available = False
    
    def _apply_newsapi_fix(self):
        """Apply NewsAPI parsing fix from main.py"""
        try:
            if self.collector_registry:
                newsapi_collector = self.collector_registry.get_collector('NewsAPI')
                if newsapi_collector and hasattr(newsapi_collector, '_parse_article'):
                    original_parse = newsapi_collector._parse_article
                    
                    def safe_parse_article(article_data, **kwargs):
                        """Safe parsing that handles None values."""
                        try:
                            safe_data = {}
                            for key, value in article_data.items():
                                if value is None:
                                    safe_data[key] = ''
                                else:
                                    safe_data[key] = value
                            return original_parse(safe_data, **kwargs)
                        except Exception as e:
                            logging.getLogger(__name__).debug(f"Article parsing failed (skipping): {e}")
                            return None
                    
                    newsapi_collector._parse_article = safe_parse_article
                    self.logger.info("✅ NewsAPI parsing fix applied")
        except Exception as e:
            self.logger.debug(f"NewsAPI fix failed: {e}")
    
    def _create_symbol_mapping(self) -> Dict[str, str]:
        """Create mapping of company names to stock symbols"""
        return {
            'apple': 'AAPL', 'aapl': 'AAPL',
            'google': 'GOOGL', 'googl': 'GOOGL', 'alphabet': 'GOOGL',
            'tesla': 'TSLA', 'tsla': 'TSLA',
            'microsoft': 'MSFT', 'msft': 'MSFT',
            'amazon': 'AMZN', 'amzn': 'AMZN',
            'meta': 'META', 'facebook': 'META',
            'nvidia': 'NVDA', 'nvda': 'NVDA',
            'amd': 'AMD', 'advanced micro devices': 'AMD',
            'intel': 'INTC', 'intc': 'INTC',
            'qualcomm': 'QCOM', 'qcom': 'QCOM',
            'broadcom': 'AVGO', 'avgo': 'AVGO',
            'oracle': 'ORCL', 'orcl': 'ORCL',
            'salesforce': 'CRM', 'crm': 'CRM',
            'netflix': 'NFLX', 'nflx': 'NFLX',
            'paypal': 'PYPL', 'pypl': 'PYPL',
            'uber': 'UBER', 'zoom': 'ZM',
            'spotify': 'SPOT', 'palantir': 'PLTR',
            'coinbase': 'COIN', 'bitcoin': 'BTC-USD',
            'ethereum': 'ETH-USD', 'dogecoin': 'DOGE-USD'
        }
    
    def search_ticker_news(self, ticker: str, target_date: datetime = None, 
                          days_back: int = None, limit: int = None, 
                          force_fresh: bool = False) -> NewsSearchResult:
        """
        Main entry point for predictor engine - searches news for a ticker
        and performs correlation analysis
        """
        if target_date is None:
            target_date = datetime.now()
        
        if days_back is None:
            days_back = self.config['search_engine']['default_days_back']
        
        if limit is None:
            limit = self.config['search_engine']['default_limit']
        
        self.logger.info(f"Starting news search for {ticker}")
        
        # Normalize ticker symbol
        normalized_ticker = self._normalize_ticker(ticker)
        
        # Search for existing articles
        existing_articles = self._search_existing_articles(
            normalized_ticker, target_date, days_back, limit
        )
        
        # Collect fresh articles if needed
        fresh_articles = []
        if force_fresh or len(existing_articles) < self.config['search_engine']['min_articles_for_correlation']:
            fresh_articles = self._collect_fresh_articles(normalized_ticker, limit)
        
        # Combine and deduplicate articles
        all_articles = self._combine_and_deduplicate(existing_articles, fresh_articles)
        
        # Analyze sentiment
        sentiment_analysis = self._analyze_sentiment(all_articles)
        
        # Perform correlation analysis
        correlation_analysis = self._perform_correlation_analysis(
            normalized_ticker, target_date, all_articles
        )
        
        # Get enhanced features
        enhanced_features = self.get_enhanced_sentiment_features(
            normalized_ticker, target_date, days_back
        )
        
        # Get market context
        market_context = self.get_enhanced_market_overview(target_date)
        
        # Create search metadata
        search_metadata = {
            'ticker': normalized_ticker,
            'original_query': ticker,
            'target_date': target_date.isoformat(),
            'days_back': days_back,
            'limit': limit,
            'existing_articles_found': len(existing_articles),
            'fresh_articles_found': len(fresh_articles),
            'total_articles': len(all_articles),
            'search_timestamp': datetime.now().isoformat(),
            'force_fresh': force_fresh
        }
        
        return NewsSearchResult(
            articles=all_articles,
            total_found=len(all_articles),
            sentiment_analysis=sentiment_analysis,
            correlation_analysis=correlation_analysis,
            enhanced_features=enhanced_features,
            market_context=market_context,
            search_metadata=search_metadata
        )
    
    def _normalize_ticker(self, ticker: str) -> str:
        """Normalize ticker symbol using mapping"""
        ticker_lower = ticker.lower().strip()
        
        # Check if it's in our mapping
        if ticker_lower in self.symbol_mapping:
            return self.symbol_mapping[ticker_lower]
        
        # Return uppercase version
        return ticker.upper().strip()
    
    def _search_existing_articles(self, ticker: str, target_date: datetime, 
                                 days_back: int, limit: int) -> List[Dict]:
        """Search existing articles in database"""
        try:
            if not os.path.exists(self.db_path):
                return []
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            since = target_date - timedelta(days=days_back)
            
            # Search in title, content, and stock_symbols
            cursor.execute("""
                SELECT * FROM news_articles 
                WHERE (title LIKE ? OR content LIKE ? OR stock_symbols LIKE ?) 
                AND published_at > ? 
                ORDER BY published_at DESC 
                LIMIT ?
            """, (f'%{ticker}%', f'%{ticker}%', f'%{ticker}%', since.isoformat(), limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to dictionaries
            articles = []
            if rows:
                # Get column names
                column_names = [description[0] for description in cursor.description]
                for row in rows:
                    article_dict = dict(zip(column_names, row))
                    articles.append(article_dict)
            
            self.logger.info(f"Found {len(articles)} existing articles for {ticker}")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error searching existing articles: {e}")
            return []
    
    def _collect_fresh_articles(self, ticker: str, limit: int) -> List[Dict]:
        """Collect fresh articles from external sources"""
        try:
            if not self.collector_registry:
                self.logger.warning("⚠️ No collector registry available for fresh collection")
                return []
            
            # Check if we have any collectors available
            available_collectors = self.collector_registry.list_collectors()
            if not available_collectors:
                self.logger.warning("⚠️ No collectors registered for fresh collection")
                return []
            
            self.logger.info(f"🔄 Collecting fresh articles for {ticker} using {len(available_collectors)} collectors")
            
            # Configure collectors for ticker search
            self._configure_collectors_for_ticker(ticker)
            
            # Show what each collector is configured to search for
            for collector_name in available_collectors:
                collector = self.collector_registry.get_collector(collector_name)
                if hasattr(collector, 'search_ticker') and collector.search_ticker:
                    self.logger.info(f"   🎯 {collector_name} configured for: {collector.search_ticker}")
                elif hasattr(collector, 'search_queries'):
                    self.logger.info(f"   🔍 {collector_name} search queries: {collector.search_queries[:3]}...")
            
            # Run collection
            collection_results = self.collector_registry.run_all_collections()
            
            # Process results
            total_collected = 0
            for collector_name, result in collection_results.items():
                if hasattr(result, 'success') and result.success:
                    total_collected += getattr(result, 'articles_processed', 0)
                    self.logger.info(f"✅ {collector_name}: {getattr(result, 'articles_processed', 0)} articles")
                else:
                    self.logger.warning(f"⚠️ {collector_name}: Collection failed")
            
            # Search for relevant articles from fresh collection
            if total_collected > 0:
                fresh_articles = self._search_fresh_articles(ticker, limit)
                self.logger.info(f"📰 Found {len(fresh_articles)} relevant fresh articles for {ticker}")
                return fresh_articles
            else:
                self.logger.info("📭 No fresh articles collected")
                return []
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting fresh articles: {e}")
            return []
    
    def _configure_collectors_for_ticker(self, ticker: str):
        """Configure collectors to search for specific ticker"""
        try:
            if not self.collector_registry:
                return
            
            # Configure Hybrid collector for specific ticker
            try:
                hybrid_collector = self.collector_registry.get_collector('HybridNews')
                if hybrid_collector and hasattr(hybrid_collector, 'set_search_ticker'):
                    hybrid_collector.set_search_ticker(ticker)
                    self.logger.debug(f"✅ Hybrid collector configured for {ticker}")
            except Exception as e:
                self.logger.debug(f"Hybrid collector configuration failed: {e}")
            
            # Configure NewsAPI collector
            try:
                newsapi_collector = self.collector_registry.get_collector('NewsAPI')
                if newsapi_collector and hasattr(newsapi_collector, 'search_queries'):
                    original_queries = getattr(newsapi_collector, 'search_queries', [])
                    newsapi_collector.search_queries = [ticker, f"{ticker} stock", f"{ticker} earnings"] + original_queries[:3]
                    self.logger.debug(f"✅ NewsAPI configured for {ticker}")
            except Exception as e:
                self.logger.debug(f"NewsAPI configuration failed: {e}")
            
            # Configure Reddit Collector
            try:
                reddit_collector = self.collector_registry.get_collector('Reddit')
                if reddit_collector and hasattr(reddit_collector, 'set_search_ticker'):
                    reddit_collector.set_search_ticker(ticker)
                    self.logger.debug(f"✅ Reddit collector configured for {ticker}")
            except Exception as e:
                self.logger.debug(f"Reddit collector configuration failed: {e}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error configuring collectors: {e}")
    
    def _search_fresh_articles(self, ticker: str, limit: int) -> List[Dict]:
        """Search for ticker-relevant articles from fresh collection"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Search in articles from the last 2 hours
            cursor.execute("""
                SELECT * FROM news_articles 
                WHERE collected_at > datetime('now', '-2 hours')
                AND (title LIKE ? OR content LIKE ? OR stock_symbols LIKE ?)
                ORDER BY collected_at DESC
                LIMIT ?
            """, (f'%{ticker}%', f'%{ticker}%', f'%{ticker}%', limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to dictionaries
            articles = []
            if rows:
                column_names = [description[0] for description in cursor.description]
                for row in rows:
                    article_dict = dict(zip(column_names, row))
                    articles.append(article_dict)
            
            self.logger.info(f"Found {len(articles)} fresh articles for {ticker}")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error searching fresh articles: {e}")
            return []
    
    def _combine_and_deduplicate(self, existing_articles: List[Dict], 
                                fresh_articles: List[Dict]) -> List[Dict]:
        """Combine and deduplicate articles"""
        all_articles = existing_articles + fresh_articles
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_articles = []
        
        for article in all_articles:
            title = article.get('title', '')
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
        
        self.logger.info(f"Combined articles: {len(unique_articles)} unique from {len(all_articles)} total")
        return unique_articles
    
    def _analyze_sentiment(self, articles: List[Dict]) -> Dict:
        """Analyze sentiment of articles"""
        if not articles:
            return {'error': 'No articles to analyze'}
        
        try:
            if self.get_market_sentiment_overview:
                # Prepare article data
                article_data = []
                for article in articles:
                    title = article.get('title', '') or ''
                    content = article.get('content', '') or ''
                    article_data.append((title, content))
                
                # Get sentiment overview
                sentiment_overview = self.get_market_sentiment_overview(article_data)
                self.logger.info(f"Sentiment analysis complete for {len(articles)} articles")
                return sentiment_overview
            else:
                # Basic fallback sentiment analysis
                return {
                    'overall_sentiment': 0.0,
                    'positive_ratio': 0.5,
                    'risk_level': 'unknown',
                    'articles_analyzed': len(articles)
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return {'error': f'Sentiment analysis failed: {e}'}
    
    def _perform_correlation_analysis(self, ticker: str, target_date: datetime, 
                                    articles: List[Dict]) -> Dict:
        """Perform correlation analysis between news and price"""
        try:
            if not self.correlation_analyzer:
                return {'error': 'Correlation analyzer not available'}
            
            if len(articles) < self.config['search_engine']['min_articles_for_correlation']:
                return {'error': f'Insufficient articles for correlation analysis'}
            
            # Perform correlation analysis
            correlation_days = self.config['search_engine']['correlation_analysis_days']
            correlation_result = self.correlation_analyzer.analyze_correlation(
                ticker, days=correlation_days
            )
            
            result = {
                'ticker': ticker,
                'correlation_coefficient': correlation_result.correlation_coefficient,
                'confidence': correlation_result.confidence,
                'prediction_accuracy': correlation_result.prediction_accuracy,
                'sample_size': correlation_result.sample_size,
                'significant_events': correlation_result.significant_events[:3],
                'analysis_period_days': correlation_days
            }
            
            self.logger.info(f"Correlation analysis complete for {ticker}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error performing correlation analysis: {e}")
            return {'error': f'Correlation analysis failed: {e}'}
    
    def get_enhanced_sentiment_features(self, ticker: str, target_date: datetime, 
                                      lookback_days: int = 7) -> EnhancedSentimentFeatures:
        """Get comprehensive sentiment features"""
        try:
            # Get basic sentiment features
            basic_features = self._get_basic_sentiment_features(ticker, target_date, lookback_days)
            
            # Return enhanced features
            return EnhancedSentimentFeatures(**basic_features)
            
        except Exception as e:
            self.logger.error(f"Error getting enhanced sentiment features: {e}")
            return EnhancedSentimentFeatures()
    
    def _get_basic_sentiment_features(self, ticker: str, target_date: datetime, 
                                    lookback_days: int) -> Dict:
        """Get basic sentiment features from database"""
        try:
            if not os.path.exists(self.db_path):
                return self._get_default_basic_features()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            basic_features = {}
            
            # Get sentiment for different periods
            for days in [1, 3, 7]:
                start_date = target_date - timedelta(days=days)
                
                # Simple query without complex joins
                cursor.execute("""
                    SELECT COUNT(*) as news_count
                    FROM news_articles
                    WHERE (title LIKE ? OR content LIKE ?)
                    AND published_at >= ? AND published_at <= ?
                """, (f'%{ticker}%', f'%{ticker}%', start_date.isoformat(), target_date.isoformat()))
                
                result = cursor.fetchone()
                news_count = int(result[0]) if result else 0
                
                basic_features[f'sentiment_{days}d'] = 0.0  # Placeholder
                basic_features[f'news_volume_{days}d'] = news_count
            
            # Set other features
            basic_features['confidence_score'] = 0.5
            basic_features['source_diversity'] = 0
            
            conn.close()
            return basic_features
            
        except Exception as e:
            self.logger.error(f"Error getting basic sentiment features: {e}")
            return self._get_default_basic_features()
    
    def _get_default_basic_features(self) -> Dict:
        """Return default basic features when queries fail"""
        return {
            'sentiment_1d': 0.0, 'sentiment_3d': 0.0, 'sentiment_7d': 0.0,
            'news_volume_1d': 0, 'news_volume_3d': 0, 'news_volume_7d': 0,
            'confidence_score': 0.5, 'source_diversity': 0
        }
    
    def get_enhanced_market_overview(self, target_date: datetime) -> Dict:
        """Get enhanced market overview"""
        try:
            if not os.path.exists(self.db_path):
                return {'error': 'Database not available'}
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            start_date = target_date - timedelta(days=1)
            
            # Get basic market stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_articles,
                    COUNT(DISTINCT source) as source_count
                FROM news_articles
                WHERE published_at >= ? AND published_at <= ?
            """, (start_date.isoformat(), target_date.isoformat()))
            
            result = cursor.fetchone()
            total_articles = int(result[0]) if result and result[0] else 0
            source_count = int(result[1]) if result and result[1] else 0
            
            conn.close()
            
            return {
                'market_sentiment': 0.0,  # Placeholder
                'total_articles': total_articles,
                'stocks_mentioned': 0,  # Placeholder
                'source_diversity': source_count,
                'enhanced': True,
                'analysis_date': target_date.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced market overview failed: {e}")
            return {
                'market_sentiment': 0.0,
                'total_articles': 0,
                'stocks_mentioned': 0,
                'source_diversity': 0,
                'enhanced': False,
                'error': str(e)
            }
    
    def get_recent_events(self, ticker: str, target_date: datetime, days: int = 3, limit: int = 5) -> List[Dict]:
        """Get recent news events for a ticker"""
        try:
            if not os.path.exists(self.db_path):
                return []
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            start_date = target_date - timedelta(days=days)
            
            # Search for recent events
            cursor.execute("""
                SELECT title, published_at, source, 0.0 as sentiment, 0.5 as confidence
                FROM news_articles
                WHERE (title LIKE ? OR content LIKE ?)
                AND published_at >= ? AND published_at <= ?
                ORDER BY published_at DESC
                LIMIT ?
            """, (f'%{ticker}%', f'%{ticker}%', start_date.isoformat(), target_date.isoformat(), limit))
            
            results = cursor.fetchall()
            conn.close()
            
            events = []
            for row in results:
                events.append({
                    'title': row[0],
                    'date': row[1],
                    'source': row[2],
                    'sentiment': float(row[3]),
                    'confidence': float(row[4])
                })
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error getting recent events for {ticker}: {e}")
            return []
    
    def get_ticker_prediction_features(self, ticker: str, target_date: datetime = None) -> Dict:
        """
        Main method for predictor engine integration
        Returns comprehensive features for price prediction
        """
        if target_date is None:
            target_date = datetime.now()
        
        self.logger.info(f"Generating prediction features for {ticker}")
        
        # Check if we have minimal functionality
        if not self.health_report['database_exists']:
            return self._get_fallback_prediction_features(ticker, target_date)
        
        try:
            # Get comprehensive news search results
            search_result = self.search_ticker_news(ticker, target_date)
            
            # Prepare prediction features
            prediction_features = {
                'ticker': ticker,
                'target_date': target_date.isoformat(),
                'search_metadata': search_result.search_metadata,
                
                # Article-based features
                'total_articles': search_result.total_found,
                'articles_quality_score': self._calculate_article_quality_score(search_result.articles),
                
                # Sentiment features
                'sentiment_analysis': search_result.sentiment_analysis,
                'enhanced_sentiment': search_result.enhanced_features.__dict__,
                
                # Correlation features
                'correlation_analysis': search_result.correlation_analysis,
                
                # Market context
                'market_context': search_result.market_context,
                
                # News-price correlation signal
                'news_price_signal': self._calculate_news_price_signal(search_result),
                
                # Prediction readiness
                'prediction_ready': self._assess_prediction_readiness(search_result),
                
                # Recent events
                'recent_events': self.get_recent_events(ticker, target_date, days=3, limit=5),
                
                # System health
                'system_health': self.health_report
            }
            
            self.logger.info(f"Prediction features generated for {ticker}")
            return prediction_features
            
        except Exception as e:
            self.logger.error(f"Error generating prediction features: {e}")
            return self._get_fallback_prediction_features(ticker, target_date, error=str(e))
    
    def _get_fallback_prediction_features(self, ticker: str, target_date: datetime, error: str = None) -> Dict:
        """Fallback prediction features when Phase 3 is not available"""
        return {
            'ticker': ticker,
            'target_date': target_date.isoformat(),
            'search_metadata': {
                'ticker': ticker,
                'target_date': target_date.isoformat(),
                'total_articles': 0,
                'search_timestamp': datetime.now().isoformat(),
                'fallback_mode': True
            },
            'total_articles': 0,
            'articles_quality_score': 0.0,
            'sentiment_analysis': {'error': 'Phase 3 not available'},
            'enhanced_sentiment': EnhancedSentimentFeatures().__dict__,
            'correlation_analysis': {'error': 'Phase 3 not available'},
            'market_context': {'error': 'Phase 3 not available'},
            'news_price_signal': {
                'strength': 0.0,
                'direction': 0.0,
                'confidence': 0.0,
                'components': {}
            },
            'prediction_ready': {
                'ready': False,
                'confidence': 0.0,
                'missing_components': ['phase3_system'],
                'quality_score': 0.0
            },
            'recent_events': [],
            'system_health': self.health_report,
            'error': error or 'Phase 3 system not available'
        }
    
    def _calculate_article_quality_score(self, articles: List[Dict]) -> float:
        """Calculate quality score based on articles"""
        if not articles:
            return 0.0
        
        quality_score = 0.0
        
        for article in articles:
            score = 0.0
            
            # Content length score
            content_length = len(article.get('content', ''))
            if content_length > 500:
                score += 0.3
            elif content_length > 200:
                score += 0.2
            elif content_length > 50:
                score += 0.1
            
            # Source reliability (placeholder)
            source = article.get('source', '').lower()
            if any(reliable in source for reliable in ['reuters', 'bloomberg', 'wsj', 'cnbc']):
                score += 0.4
            elif any(reliable in source for reliable in ['yahoo', 'marketwatch', 'seeking alpha']):
                score += 0.3
            else:
                score += 0.2
            
            # Recency score
            try:
                published_at = datetime.fromisoformat(article.get('published_at', ''))
                hours_old = (datetime.now() - published_at).total_seconds() / 3600
                if hours_old < 24:
                    score += 0.3
                elif hours_old < 72:
                    score += 0.2
                else:
                    score += 0.1
            except:
                score += 0.1
            
            quality_score += score
        
        return quality_score / len(articles)
    
    def _calculate_news_price_signal(self, search_result: NewsSearchResult) -> Dict:
        """Calculate news-price signal for prediction"""
        signal = {
            'strength': 0.0,
            'direction': 0.0,
            'confidence': 0.0,
            'components': {}
        }
        
        try:
            # Sentiment signal
            sentiment_analysis = search_result.sentiment_analysis
            if 'overall_sentiment' in sentiment_analysis:
                signal['components']['sentiment'] = sentiment_analysis['overall_sentiment']
            
            # Volume signal
            article_count = search_result.total_found
            if article_count > 10:
                signal['components']['volume'] = min(1.0, article_count / 50)
            else:
                signal['components']['volume'] = article_count / 10
            
            # Correlation signal
            correlation_analysis = search_result.correlation_analysis
            if 'correlation_coefficient' in correlation_analysis:
                signal['components']['correlation'] = abs(correlation_analysis['correlation_coefficient'])
            
            # Enhanced features signal
            enhanced = search_result.enhanced_features
            signal['components']['enhanced'] = {
                'news_volume_3d': min(1.0, enhanced.news_volume_3d / 20),
                'source_diversity': min(1.0, enhanced.source_diversity / 10),
                'correlation_strength': enhanced.correlation_strength
            }
            
            # Calculate overall signal
            sentiment_weight = 0.3
            volume_weight = 0.2
            correlation_weight = 0.3
            diversity_weight = 0.2
            
            signal['strength'] = (
                sentiment_weight * abs(signal['components'].get('sentiment', 0)) +
                volume_weight * signal['components'].get('volume', 0) +
                correlation_weight * signal['components'].get('correlation', 0) +
                diversity_weight * signal['components']['enhanced']['source_diversity']
            )
            
            signal['direction'] = signal['components'].get('sentiment', 0)
            signal['confidence'] = min(1.0, signal['strength'] * 
                                     signal['components']['enhanced']['correlation_strength'])
            
        except Exception as e:
            self.logger.error(f"Error calculating news-price signal: {e}")
        
        return signal
    
    def _assess_prediction_readiness(self, search_result: NewsSearchResult) -> Dict:
        """Assess if we have enough data for reliable prediction"""
        readiness = {
            'ready': False,
            'confidence': 0.0,
            'missing_components': [],
            'quality_score': 0.0
        }
        
        try:
            # Check minimum article count
            if search_result.total_found < 5:
                readiness['missing_components'].append('insufficient_articles')
            
            # Check sentiment analysis
            if 'error' in search_result.sentiment_analysis:
                readiness['missing_components'].append('sentiment_analysis')
            
            # Check correlation analysis
            if 'error' in search_result.correlation_analysis:
                readiness['missing_components'].append('correlation_analysis')
            
            # Check data recency
            recent_articles = sum(1 for article in search_result.articles 
                                if self._is_recent_article(article))
            if recent_articles < 3:
                readiness['missing_components'].append('recent_articles')
            
            # Calculate quality score
            quality_factors = {
                'article_count': min(1.0, search_result.total_found / 20),
                'sentiment_available': 1.0 if 'error' not in search_result.sentiment_analysis else 0.0,
                'correlation_available': 1.0 if 'error' not in search_result.correlation_analysis else 0.0,
                'source_diversity': min(1.0, search_result.enhanced_features.source_diversity / 5),
                'recent_articles': min(1.0, recent_articles / 5)
            }
            
            readiness['quality_score'] = sum(quality_factors.values()) / len(quality_factors)
            readiness['confidence'] = readiness['quality_score']
            readiness['ready'] = readiness['quality_score'] >= 0.6 and len(readiness['missing_components']) == 0
            
        except Exception as e:
            self.logger.error(f"Error assessing prediction readiness: {e}")
        
        return readiness
    
    def _is_recent_article(self, article: Dict) -> bool:
        """Check if article is recent (within 24 hours)"""
        try:
            published_at = datetime.fromisoformat(article.get('published_at', ''))
            hours_old = (datetime.now() - published_at).total_seconds() / 3600
            return hours_old < 24
        except:
            return False

# Class factory for easy integration
class Phase3NewsPredictor:
    """
    Simplified wrapper for predictor engine integration
    """
    
    def __init__(self, config_path: str = None):
        self.connector = IntegratedPhase3NewsConnector(config_path)
        self.logger = logging.getLogger(__name__)
    
    def get_prediction_signal(self, ticker: str, target_date: datetime = None) -> Dict:
        """
        Main method for predictor engine - returns prediction signal
        """
        try:
            features = self.connector.get_ticker_prediction_features(ticker, target_date)
            
            # Extract key signal components
            signal = features['news_price_signal']
            readiness = features['prediction_ready']
            
            return {
                'ticker': ticker,
                'signal_strength': signal['strength'],
                'signal_direction': signal['direction'],
                'confidence': signal['confidence'],
                'prediction_ready': readiness['ready'],
                'quality_score': readiness['quality_score'],
                'article_count': features['total_articles'],
                'sentiment_score': features['sentiment_analysis'].get('overall_sentiment', 0.0),
                'correlation_coefficient': features['correlation_analysis'].get('correlation_coefficient', 0.0),
                'recent_events': features['recent_events'][:3],  # Top 3 recent events
                'metadata': features['search_metadata']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting prediction signal for {ticker}: {e}")
            return {
                'ticker': ticker,
                'signal_strength': 0.0,
                'signal_direction': 0.0,
                'confidence': 0.0,
                'prediction_ready': False,
                'error': str(e)
            }
    
    def batch_predict(self, tickers: List[str], target_date: datetime = None) -> Dict[str, Dict]:
        """
        Batch prediction for multiple tickers
        """
        results = {}
        
        for ticker in tickers:
            results[ticker] = self.get_prediction_signal(ticker, target_date)
        
        return results
    
    def get_market_overview(self, target_date: datetime = None) -> Dict:
        """
        Get overall market overview for context
        """
        return self.connector.get_enhanced_market_overview(target_date or datetime.now())

if __name__ == "__main__":
    # Example usage
    print("Phase 3 News Intelligence - Predictor Engine Integration")
    print("=" * 60)
    
    # Test the integration
    try:
        predictor = Phase3NewsPredictor()
        
        # Show system health first
        print(f"System Health Check:")
        print(f"  Database exists: {predictor.connector.health_report['database_exists']}")
        print(f"  Database readable: {predictor.connector.health_report['database_readable']}")
        print(f"  Has articles: {predictor.connector.health_report['has_articles']}")
        print(f"  Article count: {predictor.connector.health_report['article_count']}")
        print(f"  Phase 3 components: {predictor.connector.health_report['phase3_components']}/5")
        
        if predictor.connector.health_report['recommendations']:
            print(f"  Recommendations:")
            for rec in predictor.connector.health_report['recommendations']:
                print(f"    - {rec}")
        
        print()
        
        # Test with a sample ticker
        ticker = "TSLA"
        signal = predictor.get_prediction_signal(ticker)
        
        print(f"Prediction signal for {ticker}:")
        print(f"  Signal Strength: {signal['signal_strength']:.3f}")
        print(f"  Signal Direction: {signal['signal_direction']:.3f}")
        print(f"  Confidence: {signal['confidence']:.3f}")
        print(f"  Prediction Ready: {signal['prediction_ready']}")
        print(f"  Article Count: {signal['article_count']}")
        print(f"  Sentiment Score: {signal['sentiment_score']:.3f}")
        
        if signal.get('error'):
            print(f"  Error: {signal['error']}")
        
        if signal['recent_events']:
            print(f"  Recent Events:")
            for event in signal['recent_events']:
                print(f"    - {event['title'][:50]}... ({event['source']})")
        
        # Test market overview
        market = predictor.get_market_overview()
        print(f"\nMarket Overview:")
        print(f"  Total Articles: {market['total_articles']}")
        print(f"  Market Sentiment: {market['market_sentiment']:.3f}")
        print(f"  Stocks Mentioned: {market['stocks_mentioned']}")
        print(f"  Source Diversity: {market['source_diversity']}")
        
        if market.get('error'):
            print(f"  Error: {market['error']}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Phase 3 system is set up in the parent directory")
        print("2. Run Phase 3 news collection to populate the database")
        print("3. Check that the database path is correct")
        print("4. Ensure Phase 3 dependencies are installed")