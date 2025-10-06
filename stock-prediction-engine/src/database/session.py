"""
Database session management and ORM-like operations for Phase 3 News Intelligence Engine.

This module provides high-level database operations, session management,
and data models for easy interaction with the news intelligence database.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager

from .connection import get_database, DatabaseConnection

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """News article data model."""
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    id: Optional[int] = None
    summary: Optional[str] = None
    author: Optional[str] = None
    collected_at: Optional[datetime] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    keywords: Optional[List[str]] = None
    stock_symbols: Optional[List[str]] = None
    event_type: Optional[str] = None
    impact_score: Optional[float] = None
    processed: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class NewsSource:
    """News source data model."""
    name: str
    source_type: str
    id: Optional[int] = None
    url: Optional[str] = None
    reliability_score: float = 0.5
    active: bool = True
    last_collected: Optional[datetime] = None
    created_at: Optional[datetime] = None

@dataclass
class SentimentAnalysis:
    """Sentiment analysis result data model."""
    article_id: int
    sentiment_score: float
    sentiment_label: str
    confidence: float
    analyzer_version: str
    id: Optional[int] = None
    created_at: Optional[datetime] = None

@dataclass
class StockMention:
    """Stock mention data model."""
    article_id: int
    stock_symbol: str
    mention_count: int = 1
    id: Optional[int] = None
    context: Optional[str] = None
    created_at: Optional[datetime] = None

@dataclass
class Alert:
    """Alert data model."""
    alert_type: str
    title: str
    message: str
    priority: str = 'medium'
    id: Optional[int] = None
    stock_symbol: Optional[str] = None
    triggered_by: Optional[int] = None
    sent: bool = False
    created_at: Optional[datetime] = None

@dataclass
class CollectionMetric:
    """Collection metrics data model."""
    source_name: str
    articles_collected: int = 0
    errors_count: int = 0
    collection_time: float = 0.0
    id: Optional[int] = None
    created_at: Optional[datetime] = None

class DatabaseSession:
    """High-level database session manager with ORM-like operations."""
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        self.db = db or get_database()
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        try:
            yield self
            # Note: For SQLite, transactions are auto-committed by execute_* methods
            # For PostgreSQL, you might want to implement explicit transaction control
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            raise
    
    # NewsArticle operations
    def create_article(self, article: NewsArticle) -> int:
        """Create a new news article."""
        query = """
            INSERT INTO news_articles 
            (title, content, summary, source, author, url, published_at, 
             sentiment_score, sentiment_label, keywords, stock_symbols, 
             event_type, impact_score, processed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            article.title,
            article.content,
            article.summary,
            article.source,
            article.author,
            article.url,
            article.published_at,
            article.sentiment_score,
            article.sentiment_label,
            json.dumps(article.keywords) if article.keywords else None,
            json.dumps(article.stock_symbols) if article.stock_symbols else None,
            article.event_type,
            article.impact_score,
            article.processed
        )
        
        try:
            article_id = self.db.execute_insert(query, params)
            logger.info(f"Created article with ID: {article_id}")
            return article_id
        except Exception as e:
            logger.error(f"Failed to create article: {e}")
            raise
    
    def get_article(self, article_id: int) -> Optional[NewsArticle]:
        """Get a news article by ID."""
        query = "SELECT * FROM news_articles WHERE id = ?"
        results = self.db.execute_query(query, (article_id,))
        
        if results:
            return self._row_to_article(results[0])
        return None
    
    def get_articles(self, limit: int = 100, offset: int = 0, 
                    source: Optional[str] = None,
                    since: Optional[datetime] = None) -> List[NewsArticle]:
        """Get multiple news articles with optional filtering."""
        query = "SELECT * FROM news_articles WHERE 1=1"
        params = []
        
        if source:
            query += " AND source = ?"
            params.append(source)
        
        if since:
            query += " AND published_at >= ?"
            params.append(since)
        
        query += " ORDER BY published_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        results = self.db.execute_query(query, tuple(params))
        return [self._row_to_article(row) for row in results]
    
    def update_article(self, article_id: int, **kwargs) -> bool:
        """Update an article with provided fields."""
        if not kwargs:
            return False
        
        # Add updated_at timestamp
        kwargs['updated_at'] = datetime.now()
        
        # Handle JSON fields - convert lists to JSON strings
        if 'keywords' in kwargs and kwargs['keywords'] is not None:
            if isinstance(kwargs['keywords'], list):
                kwargs['keywords'] = json.dumps(kwargs['keywords'])
        if 'stock_symbols' in kwargs and kwargs['stock_symbols'] is not None:
            if isinstance(kwargs['stock_symbols'], list):
                kwargs['stock_symbols'] = json.dumps(kwargs['stock_symbols'])
        
        set_clause = ", ".join([f"{key} = ?" for key in kwargs.keys()])
        query = f"UPDATE news_articles SET {set_clause} WHERE id = ?"
        params = list(kwargs.values()) + [article_id]
        
        rows_affected = self.db.execute_update(query, tuple(params))
        logger.info(f"Updated article {article_id}, rows affected: {rows_affected}")
        return rows_affected > 0
    
    def delete_article(self, article_id: int) -> bool:
        """Delete a news article."""
        query = "DELETE FROM news_articles WHERE id = ?"
        rows_affected = self.db.execute_update(query, (article_id,))
        logger.info(f"Deleted article {article_id}, rows affected: {rows_affected}")
        return rows_affected > 0
    
    # NewsSource operations
    def create_source(self, source: NewsSource) -> int:
        """Create a new news source."""
        query = """
            INSERT INTO news_sources (name, url, source_type, reliability_score, active)
            VALUES (?, ?, ?, ?, ?)
        """
        params = (source.name, source.url, source.source_type, 
                 source.reliability_score, source.active)
        
        return self.db.execute_insert(query, params)
    
    def get_active_sources(self) -> List[NewsSource]:
        """Get all active news sources."""
        query = "SELECT * FROM news_sources WHERE active = ? ORDER BY reliability_score DESC"
        results = self.db.execute_query(query, (True,))
        return [self._row_to_source(row) for row in results]
    
    def update_source_collection_time(self, source_id: int):
        """Update the last collection time for a source."""
        query = "UPDATE news_sources SET last_collected = ? WHERE id = ?"
        self.db.execute_update(query, (datetime.now(), source_id))
    
    # Sentiment Analysis operations
    def create_sentiment_analysis(self, sentiment: SentimentAnalysis) -> int:
        """Create a sentiment analysis record."""
        query = """
            INSERT INTO sentiment_analysis 
            (article_id, sentiment_score, sentiment_label, confidence, analyzer_version)
            VALUES (?, ?, ?, ?, ?)
        """
        params = (sentiment.article_id, sentiment.sentiment_score,
                 sentiment.sentiment_label, sentiment.confidence,
                 sentiment.analyzer_version)
        
        return self.db.execute_insert(query, params)
    
    def get_sentiment_by_article(self, article_id: int) -> Optional[SentimentAnalysis]:
        """Get sentiment analysis for a specific article."""
        query = "SELECT * FROM sentiment_analysis WHERE article_id = ?"
        results = self.db.execute_query(query, (article_id,))
        
        if results:
            return self._row_to_sentiment(results[0])
        return None
    
    # Stock Mention operations
    def create_stock_mention(self, mention: StockMention) -> int:
        """Create a stock mention record."""
        query = """
            INSERT INTO stock_mentions (article_id, stock_symbol, mention_count, context)
            VALUES (?, ?, ?, ?)
        """
        params = (mention.article_id, mention.stock_symbol,
                 mention.mention_count, mention.context)
        
        return self.db.execute_insert(query, params)
    
    def get_stock_mentions(self, stock_symbol: str, since: Optional[datetime] = None) -> List[StockMention]:
        """Get all mentions for a specific stock."""
        query = "SELECT * FROM stock_mentions WHERE stock_symbol = ?"
        params = [stock_symbol]
        
        if since:
            query += " AND created_at >= ?"
            params.append(since)
        
        query += " ORDER BY created_at DESC"
        results = self.db.execute_query(query, tuple(params))
        return [self._row_to_stock_mention(row) for row in results]
    
    # Alert operations
    def create_alert(self, alert: Alert) -> int:
        """Create a new alert."""
        query = """
            INSERT INTO alerts (alert_type, title, message, priority, stock_symbol, triggered_by)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (alert.alert_type, alert.title, alert.message,
                 alert.priority, alert.stock_symbol, alert.triggered_by)
        
        return self.db.execute_insert(query, params)
    
    def get_unsent_alerts(self) -> List[Alert]:
        """Get all unsent alerts."""
        query = "SELECT * FROM alerts WHERE sent = ? ORDER BY created_at ASC"
        results = self.db.execute_query(query, (False,))
        return [self._row_to_alert(row) for row in results]
    
    def mark_alert_sent(self, alert_id: int):
        """Mark an alert as sent."""
        query = "UPDATE alerts SET sent = ? WHERE id = ?"
        self.db.execute_update(query, (True, alert_id))
    
    # Collection Metrics operations
    def create_collection_metric(self, metric: CollectionMetric) -> int:
        """Create a collection metric record."""
        query = """
            INSERT INTO collection_metrics 
            (source_name, articles_collected, errors_count, collection_time)
            VALUES (?, ?, ?, ?)
        """
        params = (metric.source_name, metric.articles_collected,
                 metric.errors_count, metric.collection_time)
        
        return self.db.execute_insert(query, params)
    
    def get_collection_metrics(self, since: Optional[datetime] = None) -> List[CollectionMetric]:
        """Get collection metrics."""
        query = "SELECT * FROM collection_metrics"
        params = []
        
        if since:
            query += " WHERE created_at >= ?"
            params.append(since)
        
        query += " ORDER BY created_at DESC"
        results = self.db.execute_query(query, tuple(params))
        return [self._row_to_collection_metric(row) for row in results]
    
    # Analytics and reporting methods
    def get_article_count_by_source(self, since: Optional[datetime] = None) -> Dict[str, int]:
        """Get article count grouped by source."""
        query = "SELECT source, COUNT(*) as count FROM news_articles"
        params = []
        
        if since:
            query += " WHERE published_at >= ?"
            params.append(since)
        
        query += " GROUP BY source ORDER BY count DESC"
        results = self.db.execute_query(query, tuple(params))
        
        return {row['source']: row['count'] for row in results}
    
    def get_sentiment_distribution(self, since: Optional[datetime] = None) -> Dict[str, int]:
        """Get sentiment distribution."""
        query = """
            SELECT sentiment_label, COUNT(*) as count 
            FROM news_articles 
            WHERE sentiment_label IS NOT NULL
        """
        params = []
        
        if since:
            query += " AND published_at >= ?"
            params.append(since)
        
        query += " GROUP BY sentiment_label"
        results = self.db.execute_query(query, tuple(params))
        
        return {row['sentiment_label']: row['count'] for row in results}
    
    def get_top_mentioned_stocks(self, limit: int = 10, 
                                since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get most mentioned stocks."""
        query = """
            SELECT stock_symbol, COUNT(*) as mentions, AVG(mention_count) as avg_mentions
            FROM stock_mentions
        """
        params = []
        
        if since:
            query += " WHERE created_at >= ?"
            params.append(since)
        
        query += " GROUP BY stock_symbol ORDER BY mentions DESC LIMIT ?"
        params.append(limit)
        
        results = self.db.execute_query(query, tuple(params))
        return [dict(row) for row in results]
    
    def get_recent_breaking_news(self, hours: int = 24) -> List[NewsArticle]:
        """Get recent breaking news articles."""
        since = datetime.now() - timedelta(hours=hours)
        query = """
            SELECT * FROM news_articles 
            WHERE published_at >= ? 
            AND (event_type = 'breaking' OR impact_score > 0.7)
            ORDER BY published_at DESC
        """
        results = self.db.execute_query(query, (since,))
        return [self._row_to_article(row) for row in results]
    
    # Helper methods for row to object conversion
    def _row_to_article(self, row) -> NewsArticle:
        """Convert database row to NewsArticle object."""
        return NewsArticle(
            id=row['id'],
            title=row['title'],
            content=row['content'],
            summary=row['summary'],
            source=row['source'],
            author=row['author'],
            url=row['url'],
            published_at=row['published_at'],
            collected_at=row['collected_at'],
            sentiment_score=row['sentiment_score'],
            sentiment_label=row['sentiment_label'],
            keywords=json.loads(row['keywords']) if row['keywords'] else None,
            stock_symbols=json.loads(row['stock_symbols']) if row['stock_symbols'] else None,
            event_type=row['event_type'],
            impact_score=row['impact_score'],
            processed=row['processed'],
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )
    
    def _row_to_source(self, row) -> NewsSource:
        """Convert database row to NewsSource object."""
        return NewsSource(
            id=row['id'],
            name=row['name'],
            url=row['url'],
            source_type=row['source_type'],
            reliability_score=row['reliability_score'],
            active=row['active'],
            last_collected=row['last_collected'],
            created_at=row['created_at']
        )
    
    def _row_to_sentiment(self, row) -> SentimentAnalysis:
        """Convert database row to SentimentAnalysis object."""
        return SentimentAnalysis(
            id=row['id'],
            article_id=row['article_id'],
            sentiment_score=row['sentiment_score'],
            sentiment_label=row['sentiment_label'],
            confidence=row['confidence'],
            analyzer_version=row['analyzer_version'],
            created_at=row['created_at']
        )
    
    def _row_to_stock_mention(self, row) -> StockMention:
        """Convert database row to StockMention object."""
        return StockMention(
            id=row['id'],
            article_id=row['article_id'],
            stock_symbol=row['stock_symbol'],
            mention_count=row['mention_count'],
            context=row['context'],
            created_at=row['created_at']
        )
    
    def _row_to_alert(self, row) -> Alert:
        """Convert database row to Alert object."""
        return Alert(
            id=row['id'],
            alert_type=row['alert_type'],
            title=row['title'],
            message=row['message'],
            priority=row['priority'],
            stock_symbol=row['stock_symbol'],
            triggered_by=row['triggered_by'],
            sent=row['sent'],
            created_at=row['created_at']
        )
    
    def _row_to_collection_metric(self, row) -> CollectionMetric:
        """Convert database row to CollectionMetric object."""
        return CollectionMetric(
            id=row['id'],
            source_name=row['source_name'],
            articles_collected=row['articles_collected'],
            errors_count=row['errors_count'],
            collection_time=row['collection_time'],
            created_at=row['created_at']
        )

# Global session instance
_session_instance = None

def get_session() -> DatabaseSession:
    """Get singleton database session instance."""
    global _session_instance
    if _session_instance is None:
        _session_instance = DatabaseSession()
    return _session_instance

# Example usage and testing
if __name__ == "__main__":
    # Test database session
    session = get_session()
    
    # Test creating a news source
    source = NewsSource(
        name="Test API Source",
        url="https://api.example.com/news",
        source_type="api",
        reliability_score=0.8
    )
    
    source_id = session.create_source(source)
    print(f"Created source with ID: {source_id}")
    
    # Test creating a news article
    article = NewsArticle(
        title="Test Article: Market Update",
        content="This is a test article about market conditions...",
        source="Test API Source",
        url="https://example.com/article/123",
        published_at=datetime.now(),
        keywords=["market", "stocks", "update"],
        stock_symbols=["AAPL", "GOOGL"],
        sentiment_score=0.3,
        sentiment_label="positive"
    )
    
    article_id = session.create_article(article)
    print(f"Created article with ID: {article_id}")
    
    # Test retrieving the article
    retrieved_article = session.get_article(article_id)
    print(f"Retrieved article: {retrieved_article.title}")
    
    # Test analytics
    article_counts = session.get_article_count_by_source()
    print(f"Article counts by source: {article_counts}")
    
    sentiment_dist = session.get_sentiment_distribution()
    print(f"Sentiment distribution: {sentiment_dist}")
    
    print("Session tests completed successfully!")