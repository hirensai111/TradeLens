"""
Base collector framework for Phase 3 News Intelligence Engine.

This module provides the abstract base class and common functionality
for all news collectors (NewsAPI, Reddit, web scrapers, etc.).

FIXED: Added database schema validation and graceful column handling
"""

import time
import logging
import hashlib
import sqlite3
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Iterator, Set
from dataclasses import dataclass, field

from news_system.database import get_session, NewsArticle, NewsSource, CollectionMetric

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class CollectionConfig:
    """Configuration for news collection."""
    max_articles_per_run: int = 100
    collection_interval_minutes: int = 15
    enable_deduplication: bool = True
    min_article_length: int = 100
    max_article_age_hours: int = 72
    retry_attempts: int = 3
    retry_delay_seconds: int = 5
    rate_limit_delay: float = 1.0  # seconds between requests
    timeout_seconds: int = 30

@dataclass
class CollectionResult:
    """Result of a collection operation."""
    source_name: str
    success: bool
    articles_collected: int = 0
    articles_processed: int = 0
    articles_deduplicated: int = 0
    errors: List[str] = field(default_factory=list)
    collection_time: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

class DatabaseSchemaManager:
    """Handles database schema validation and column management."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or self._get_default_db_path()
        self._table_columns: Dict[str, Set[str]] = {}
        self._schema_checked = False
    
    def _get_default_db_path(self) -> str:
        """Get the default database path."""
        # Try to get from session first
        try:
            session = get_session()
            if hasattr(session, 'db') and hasattr(session.db, 'db_path'):
                return session.db.db_path
        except:
            pass
        
        # Fallback to standard path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, '..', 'data', 'news_intelligence.db')
    
    def _discover_table_schema(self, table_name: str) -> Set[str]:
        """Discover the columns in a database table."""
        if not os.path.exists(self.db_path):
            logger.warning(f"Database not found: {self.db_path}")
            return set()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = {col[1] for col in cursor.fetchall()}
            
            conn.close()
            
            self._table_columns[table_name] = columns
            logger.debug(f"Discovered columns for {table_name}: {columns}")
            return columns
            
        except Exception as e:
            logger.error(f"Error discovering schema for {table_name}: {e}")
            return set()
    
    def get_table_columns(self, table_name: str) -> Set[str]:
        """Get the columns for a table, discovering if necessary."""
        if table_name not in self._table_columns:
            self._discover_table_schema(table_name)
        
        return self._table_columns.get(table_name, set())
    
    def column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table."""
        columns = self.get_table_columns(table_name)
        return column_name in columns
    
    def add_missing_columns(self, table_name: str, required_columns: Dict[str, str]):
        """Add missing columns to a table."""
        if not os.path.exists(self.db_path):
            logger.error(f"Database not found: {self.db_path}")
            return
        
        existing_columns = self.get_table_columns(table_name)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for column_name, column_def in required_columns.items():
                if column_name not in existing_columns:
                    try:
                        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def};")
                        logger.info(f"[OK] Added missing column: {table_name}.{column_name}")
                        
                        # Update our cache
                        if table_name in self._table_columns:
                            self._table_columns[table_name].add(column_name)
                        
                    except Exception as e:
                        logger.warning(f"[WARNING] Could not add column {column_name}: {e}")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error adding missing columns: {e}")
    
    def filter_article_data(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter article data to only include existing columns."""
        news_columns = self.get_table_columns('news_articles')
        
        if not news_columns:
            # If we can't determine columns, return all data and let DB handle it
            return article_data
        
        filtered_data = {}
        for key, value in article_data.items():
            if key in news_columns:
                filtered_data[key] = value
            else:
                logger.debug(f"Skipping column '{key}' - not in database schema")
        
        return filtered_data

class BaseCollector(ABC):
    """Abstract base class for all news collectors."""
    
    def __init__(self, source_name: str, config: CollectionConfig = None):
        self.source_name = source_name
        self.config = config or CollectionConfig()
        self.session = get_session()
        self.schema_manager = DatabaseSchemaManager()
        self._ensure_database_schema()
        self._ensure_source_exists()
    
    def _ensure_database_schema(self):
        """Ensure the database has all required columns."""
        required_columns = {
            'content_hash': 'TEXT',
            'collected_at': 'DATETIME DEFAULT CURRENT_TIMESTAMP',
            'stock_symbols': 'TEXT',
            'sentiment_score': 'REAL DEFAULT 0.0',
            'confidence_score': 'REAL DEFAULT 0.0',
            'processed': 'BOOLEAN DEFAULT FALSE'
        }
        
        try:
            self.schema_manager.add_missing_columns('news_articles', required_columns)
        except Exception as e:
            logger.warning(f"Could not ensure database schema: {e}")
    
    @abstractmethod
    def collect_articles(self) -> Iterator[NewsArticle]:
        """
        Collect articles from the news source.
        
        Yields:
            NewsArticle: Individual news articles
        """
        pass
    
    @abstractmethod
    def get_source_info(self) -> Dict[str, Any]:
        """
        Get information about the news source.
        
        Returns:
            Dict containing source metadata
        """
        pass
    
    def _ensure_source_exists(self):
        """Ensure the news source exists in the database."""
        try:
            sources = self.session.get_active_sources()
            source_names = [s.name for s in sources]
            
            if self.source_name not in source_names:
                source_info = self.get_source_info()
                source = NewsSource(
                    name=self.source_name,
                    url=source_info.get('url'),
                    source_type=source_info.get('type', 'unknown'),
                    reliability_score=source_info.get('reliability', 0.5),
                    active=True
                )
                
                source_id = self.session.create_source(source)
                logger.info(f"Created new source: {self.source_name} (ID: {source_id})")
        except Exception as e:
            logger.error(f"Error ensuring source exists: {e}")
    
    def run_collection(self) -> CollectionResult:
        """
        Run a complete collection cycle.
        
        Returns:
            CollectionResult: Results of the collection operation
        """
        result = CollectionResult(
            source_name=self.source_name,
            success=False,
            start_time=datetime.now()
        )
        
        collection_start = time.time()
        
        try:
            logger.info(f"Starting collection from {self.source_name}")
            
            # Collect articles
            articles_collected = []
            for article in self.collect_articles():
                if len(articles_collected) >= self.config.max_articles_per_run:
                    logger.info(f"Reached max articles limit: {self.config.max_articles_per_run}")
                    break
                
                # Validate article
                if self._validate_article(article):
                    articles_collected.append(article)
                else:
                    logger.debug(f"Skipped invalid article: {article.title[:50]}...")
                
                # Rate limiting
                time.sleep(self.config.rate_limit_delay)
            
            result.articles_collected = len(articles_collected)
            logger.info(f"Collected {result.articles_collected} articles from {self.source_name}")
            
            # Process and store articles
            for article in articles_collected:
                try:
                    if self._should_process_article(article):
                        # FIXED: Safe article processing with schema validation
                        if self._process_article_safely(article):
                            result.articles_processed += 1
                        else:
                            result.errors.append(f"Failed to store article: {article.title[:50]}...")
                    else:
                        result.articles_deduplicated += 1
                        logger.debug(f"Deduplicated article: {article.title[:50]}...")
                
                except Exception as e:
                    error_msg = f"Failed to process article '{article.title[:50]}...': {e}"
                    result.errors.append(error_msg)
                    logger.error(error_msg)
            
            # Update source collection timestamp
            try:
                sources = self.session.get_active_sources()
                source = next((s for s in sources if s.name == self.source_name), None)
                if source:
                    self.session.update_source_collection_time(source.id)
            except Exception as e:
                logger.warning(f"Could not update source collection time: {e}")
            
            result.success = True
            result.end_time = datetime.now()
            result.collection_time = time.time() - collection_start
            
            logger.info(f"Collection completed: {result.articles_processed} processed, "
                       f"{result.articles_deduplicated} deduplicated, "
                       f"{len(result.errors)} errors")
        
        except Exception as e:
            error_msg = f"Collection failed for {self.source_name}: {e}"
            result.errors.append(error_msg)
            result.end_time = datetime.now()
            result.collection_time = time.time() - collection_start
            logger.error(error_msg)
        
        finally:
            # Record collection metrics
            self._record_collection_metrics(result)
        
        return result
    
    def _process_article_safely(self, article: NewsArticle) -> bool:
        """
        FIXED: Safely process and store an article with schema validation.
        
        Args:
            article: The article to process
            
        Returns:
            bool: True if successfully stored
        """
        try:
            # Prepare article data
            article_dict = self._prepare_article_for_storage(article)
            
            # Filter data based on actual database schema
            safe_article_dict = self.schema_manager.filter_article_data(article_dict)
            
            # Store the article
            article_id = self.session.create_article(NewsArticle(**safe_article_dict))
            
            logger.debug(f"Stored article ID {article_id}: {article.title[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to safely process article '{article.title[:50]}...': {e}")
            return False
    
    def _validate_article(self, article: NewsArticle) -> bool:
        """
        Validate an article meets quality criteria.
        
        Args:
            article: The article to validate
            
        Returns:
            bool: True if article is valid
        """
        # Check required fields
        if not article.title or not article.content or not article.url:
            return False
        
        # Check minimum content length
        if len(article.content) < self.config.min_article_length:
            return False
        
        # Check article age
        if article.published_at:
            max_age = datetime.now() - timedelta(hours=self.config.max_article_age_hours)
            # Ensure both datetimes are naive for comparison
            published_at = article.published_at
            if published_at.tzinfo is not None:
                published_at = published_at.replace(tzinfo=None)
            if published_at < max_age:
                return False
        
        return True
    
    def _should_process_article(self, article: NewsArticle) -> bool:
        """
        Determine if an article should be processed (deduplication check).
        
        Args:
            article: The article to check
            
        Returns:
            bool: True if article should be processed
        """
        if not self.config.enable_deduplication:
            return True
        
        try:
            # Check for duplicate URL
            existing_articles = self.session.db.execute_query(
                "SELECT id FROM news_articles WHERE url = ?",
                (article.url,)
            )
            
            if existing_articles:
                return False
            
            # Check for duplicate content hash (only if column exists)
            if self.schema_manager.column_exists('news_articles', 'content_hash'):
                content_hash = self._generate_content_hash(article)
                existing_by_hash = self.session.db.execute_query(
                    "SELECT id FROM news_articles WHERE content_hash = ?",
                    (content_hash,)
                )
                
                return len(existing_by_hash) == 0
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking for duplicate article: {e}")
            return True  # Process anyway if deduplication fails
    
    def _generate_content_hash(self, article: NewsArticle) -> str:
        """
        Generate a hash of the article content for deduplication.
        
        Args:
            article: The article to hash
            
        Returns:
            str: Content hash
        """
        # Combine title and first 500 chars of content
        content_to_hash = f"{article.title or ''}{(article.content or '')[:500]}"
        return hashlib.md5(content_to_hash.encode('utf-8')).hexdigest()
    
    def _prepare_article_for_storage(self, article: NewsArticle) -> Dict[str, Any]:
        """
        FIXED: Prepare article for database storage with schema validation.
        
        Args:
            article: The article to prepare
            
        Returns:
            Dict: Article data ready for storage
        """
        # Base article data
        article_dict = {
            'title': article.title or '',
            'content': article.content or '',
            'summary': getattr(article, 'summary', None) or '',
            'source': self.source_name,
            'author': getattr(article, 'author', None) or '',
            'url': article.url or '',
            'published_at': article.published_at,
            'sentiment_score': getattr(article, 'sentiment_score', None) or 0.0,
            'sentiment_label': getattr(article, 'sentiment_label', None) or '',
            'keywords': getattr(article, 'keywords', None) or '',
            'stock_symbols': getattr(article, 'stock_symbols', None) or '',
            'event_type': getattr(article, 'event_type', None) or '',
            'impact_score': getattr(article, 'impact_score', None) or 0.0,
            'processed': False
        }
        
        # Add optional columns only if they exist in database
        optional_columns = {
            'content_hash': self._generate_content_hash(article),
            'collected_at': datetime.now(),
            'confidence_score': getattr(article, 'confidence_score', None) or 0.0
        }
        
        for column_name, value in optional_columns.items():
            if self.schema_manager.column_exists('news_articles', column_name):
                article_dict[column_name] = value
            else:
                logger.debug(f"Skipping column '{column_name}' - not in database schema")
        
        return article_dict
    
    def _record_collection_metrics(self, result: CollectionResult):
        """
        Record collection metrics in the database.
        
        Args:
            result: Collection result to record
        """
        try:
            metric = CollectionMetric(
                source_name=self.source_name,
                articles_collected=result.articles_processed,
                errors_count=len(result.errors),
                collection_time=result.collection_time
            )
            
            self.session.create_collection_metric(metric)
            logger.debug(f"Recorded collection metrics for {self.source_name}")
            
        except Exception as e:
            logger.error(f"Failed to record collection metrics: {e}")
    
    def get_collection_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Get collection statistics for the past N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dict: Collection statistics
        """
        try:
            since = datetime.now() - timedelta(days=days)
            metrics = self.session.get_collection_metrics(since=since)
            
            source_metrics = [m for m in metrics if m.source_name == self.source_name]
            
            if not source_metrics:
                return {
                    'source_name': self.source_name,
                    'total_collections': 0,
                    'total_articles': 0,
                    'total_errors': 0,
                    'avg_collection_time': 0.0,
                    'success_rate': 0.0,
                    'articles_per_collection': 0.0,
                    'period_days': days
                }
            
            total_collections = len(source_metrics)
            total_articles = sum(m.articles_collected for m in source_metrics)
            total_errors = sum(m.errors_count for m in source_metrics)
            avg_collection_time = sum(m.collection_time for m in source_metrics) / total_collections
            success_rate = (total_collections - len([m for m in source_metrics if m.errors_count > 0])) / total_collections
            articles_per_collection = total_articles / total_collections if total_collections > 0 else 0
            
            return {
                'source_name': self.source_name,
                'total_collections': total_collections,
                'total_articles': total_articles,
                'total_errors': total_errors,
                'avg_collection_time': round(avg_collection_time, 2),
                'success_rate': round(success_rate * 100, 2),
                'articles_per_collection': round(articles_per_collection, 1),
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                'source_name': self.source_name,
                'error': str(e),
                'period_days': days
            }
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to the news source.
        
        Returns:
            Dict: Connection test results
        """
        start_time = time.time()
        
        try:
            # Try to collect just one article to test connectivity
            articles_iter = self.collect_articles()
            test_article = next(articles_iter, None)
            
            connection_time = time.time() - start_time
            
            return {
                'source_name': self.source_name,
                'success': test_article is not None,
                'connection_time': round(connection_time, 3),
                'test_article_found': test_article is not None,
                'error': None
            }
            
        except Exception as e:
            connection_time = time.time() - start_time
            return {
                'source_name': self.source_name,
                'success': False,
                'connection_time': round(connection_time, 3),
                'test_article_found': False,
                'error': str(e)
            }

class CollectorRegistry:
    """Registry for managing multiple news collectors."""
    
    def __init__(self):
        self.collectors: Dict[str, BaseCollector] = {}
    
    def register(self, collector: BaseCollector):
        """Register a collector."""
        self.collectors[collector.source_name] = collector
        logger.info(f"Registered collector: {collector.source_name}")
    
    def unregister(self, source_name: str):
        """Unregister a collector."""
        if source_name in self.collectors:
            del self.collectors[source_name]
            logger.info(f"Unregistered collector: {source_name}")
    
    def get_collector(self, source_name: str) -> Optional[BaseCollector]:
        """Get a collector by source name."""
        return self.collectors.get(source_name)
    
    def list_collectors(self) -> List[str]:
        """Get list of registered collector names."""
        return list(self.collectors.keys())
    
    def run_all_collections(self) -> Dict[str, CollectionResult]:
        """Run collection for all registered collectors."""
        results = {}
        
        for source_name, collector in self.collectors.items():
            try:
                logger.info(f"Running collection for {source_name}")
                results[source_name] = collector.run_collection()
            except Exception as e:
                logger.error(f"Collection failed for {source_name}: {e}")
                results[source_name] = CollectionResult(
                    source_name=source_name,
                    success=False,
                    errors=[str(e)]
                )
        
        return results
    
    def test_all_connections(self) -> Dict[str, Dict[str, Any]]:
        """Test connections for all registered collectors."""
        results = {}
        
        for source_name, collector in self.collectors.items():
            results[source_name] = collector.test_connection()
        
        return results
    
    def get_all_stats(self, days: int = 7) -> Dict[str, Dict[str, Any]]:
        """Get collection statistics for all collectors."""
        stats = {}
        
        for source_name, collector in self.collectors.items():
            stats[source_name] = collector.get_collection_stats(days)
        
        return stats

# Global collector registry
_collector_registry = CollectorRegistry()

def get_collector_registry() -> CollectorRegistry:
    """Get the global collector registry."""
    return _collector_registry

def register_collector(collector: BaseCollector):
    """Register a collector with the global registry."""
    _collector_registry.register(collector)

def run_all_collections() -> Dict[str, CollectionResult]:
    """Run collection for all registered collectors."""
    return _collector_registry.run_all_collections()

# Example usage and testing
if __name__ == "__main__":
    # This would be implemented by concrete collectors
    # Here's an example of how it would be used:
    
    class TestCollector(BaseCollector):
        """Test collector for demonstration."""
        
        def collect_articles(self) -> Iterator[NewsArticle]:
            """Collect test articles."""
            for i in range(3):
                yield NewsArticle(
                    title=f"Test Article {i+1}",
                    content=f"This is test content for article {i+1}. " * 20,
                    source=self.source_name,
                    url=f"https://example.com/article/{i+1}",
                    published_at=datetime.now() - timedelta(hours=i),
                    author=f"Test Author {i+1}"
                )
        
        def get_source_info(self) -> Dict[str, Any]:
            """Get test source info."""
            return {
                'url': 'https://example.com',
                'type': 'test',
                'reliability': 0.9
            }
    
    # Test the base collector
    print("Testing Fixed Base Collector...")
    print("=" * 50)
    
    try:
        test_collector = TestCollector("Test Source")
        
        # Test schema manager
        print(f"Database path: {test_collector.schema_manager.db_path}")
        print(f"Database exists: {os.path.exists(test_collector.schema_manager.db_path)}")
        
        # Test connection
        connection_test = test_collector.test_connection()
        print(f"Connection test: {connection_test}")
        
        # Test collection
        result = test_collector.run_collection()
        print(f"Collection result: Success={result.success}, "
              f"Collected={result.articles_collected}, "
              f"Processed={result.articles_processed}, "
              f"Errors={len(result.errors)}")
        
        if result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"  - {error}")
        
        # Test stats
        stats = test_collector.get_collection_stats()
        print(f"Collection stats: {stats}")
        
        print("[OK] Base collector test completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Base collector test failed: {e}")
        import traceback
        traceback.print_exc()