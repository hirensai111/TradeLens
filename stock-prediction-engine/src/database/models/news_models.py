"""
Stock Prediction Engine - Phase 3: News Intelligence Engine
Database Models for News Intelligence System

This module defines all database models for storing news articles, sentiment analysis,
events, correlations, and intelligence data.
"""

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, Boolean, 
    ForeignKey, Index, UniqueConstraint, JSON, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum
import json
import hashlib


# Create base class for all models
Base = declarative_base()


class SentimentType(Enum):
    """Sentiment classification types"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class EventType(Enum):
    """News event types"""
    EARNINGS = "earnings"
    ANALYST = "analyst"
    PRODUCT = "product"
    CORPORATE = "corporate"
    REGULATORY = "regulatory"
    LEADERSHIP = "leadership"
    LEGAL = "legal"
    ECONOMIC = "economic"
    SOCIAL = "social"
    TECHNICAL = "technical"
    OTHER = "other"


class ImpactLevel(Enum):
    """Impact level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NewsSource(Base):
    """
    News sources with reliability and configuration data
    """
    __tablename__ = "news_sources"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    url = Column(String(500), nullable=True)
    source_type = Column(String(50), nullable=False)  # api, web_scraping, social_media
    tier = Column(String(20), nullable=False)  # tier_1, tier_2, tier_3, social_media
    reliability_weight = Column(Float, default=0.5)
    confidence_score = Column(Float, default=0.6)
    
    # Rate limiting and scraping config
    rate_limit_per_hour = Column(Integer, default=100)
    last_scraped = Column(DateTime, nullable=True)
    scraping_enabled = Column(Boolean, default=True)
    scraping_config = Column(JSON, nullable=True)  # Store scraping parameters
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    articles = relationship("NewsArticle", back_populates="source")
    
    def __repr__(self):
        return f"<NewsSource(name='{self.name}', tier='{self.tier}', weight={self.reliability_weight})>"
    
    def can_scrape_now(self) -> bool:
        """Check if source can be scraped based on rate limits"""
        if not self.scraping_enabled or not self.last_scraped:
            return True
        
        time_since_last = datetime.utcnow() - self.last_scraped
        min_interval = timedelta(hours=1) / self.rate_limit_per_hour
        return time_since_last >= min_interval


class NewsArticle(Base):
    """
    Core news articles with comprehensive metadata
    """
    __tablename__ = "news_articles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Article identification
    url = Column(String(1000), nullable=False, index=True)
    url_hash = Column(String(64), unique=True, nullable=False, index=True)  # For deduplication
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    
    # Stock association
    ticker = Column(String(10), nullable=False, index=True)
    tickers_mentioned = Column(JSON, nullable=True)  # List of all tickers mentioned
    
    # Source information
    source_id = Column(Integer, ForeignKey("news_sources.id"), nullable=False)
    author = Column(String(200), nullable=True)
    
    # Temporal data
    published_date = Column(DateTime, nullable=False, index=True)
    scraped_date = Column(DateTime, default=func.now())
    
    # Content classification
    category = Column(String(50), nullable=True, index=True)  # financial, political, etc.
    subcategory = Column(String(50), nullable=True)  # earnings, merger, etc.
    keywords = Column(JSON, nullable=True)  # Extracted keywords
    entities = Column(JSON, nullable=True)  # Named entities (companies, people, places)
    
    # Quality and relevance metrics
    relevance_score = Column(Float, nullable=True)  # How relevant to the ticker (0-1)
    quality_score = Column(Float, nullable=True)  # Content quality score (0-1)
    readability_score = Column(Float, nullable=True)  # Text readability
    word_count = Column(Integer, nullable=True)
    
    # Sentiment analysis results
    sentiment_score = Column(Float, nullable=True)  # -1 to 1
    sentiment_type = Column(SQLEnum(SentimentType), nullable=True)
    sentiment_confidence = Column(Float, nullable=True)  # 0 to 1
    emotional_indicators = Column(JSON, nullable=True)  # fear, greed, optimism, etc.
    
    # Impact assessment
    impact_score = Column(Float, nullable=True)  # Predicted impact on stock (0-1)
    impact_level = Column(SQLEnum(ImpactLevel), nullable=True)
    impact_timeframe = Column(String(20), nullable=True)  # immediate, short_term, long_term
    
    # Processing flags
    is_processed = Column(Boolean, default=False)
    is_breaking_news = Column(Boolean, default=False)
    is_duplicate = Column(Boolean, default=False)
    processing_errors = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    source = relationship("NewsSource", back_populates="articles")
    events = relationship("NewsEvent", back_populates="article")
    correlations = relationship("NewsCorrelation", back_populates="article")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_ticker_date', 'ticker', 'published_date'),
        Index('idx_sentiment_impact', 'sentiment_score', 'impact_score'),
        Index('idx_category_date', 'category', 'published_date'),
        Index('idx_source_date', 'source_id', 'published_date'),
        UniqueConstraint('url_hash', name='uq_url_hash'),
    )
    
    def __repr__(self):
        return f"<NewsArticle(ticker='{self.ticker}', title='{self.title[:50]}...', sentiment={self.sentiment_score})>"
    
    @classmethod
    def create_url_hash(cls, url: str) -> str:
        """Create hash for URL deduplication"""
        return hashlib.sha256(url.encode()).hexdigest()
    
    def calculate_age_hours(self) -> float:
        """Calculate article age in hours"""
        if not self.published_date:
            return 0.0
        return (datetime.utcnow() - self.published_date).total_seconds() / 3600
    
    def is_fresh(self, hours: int = 24) -> bool:
        """Check if article is considered fresh"""
        return self.calculate_age_hours() <= hours
    
    def get_sentiment_label(self) -> str:
        """Get human-readable sentiment label"""
        if self.sentiment_type:
            return self.sentiment_type.value.replace('_', ' ').title()
        return "Unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert article to dictionary for API responses"""
        return {
            'id': self.id,
            'url': self.url,
            'title': self.title,
            'ticker': self.ticker,
            'published_date': self.published_date.isoformat() if self.published_date else None,
            'sentiment_score': self.sentiment_score,
            'sentiment_type': self.sentiment_type.value if self.sentiment_type else None,
            'impact_score': self.impact_score,
            'relevance_score': self.relevance_score,
            'category': self.category,
            'is_breaking_news': self.is_breaking_news,
            'source_name': self.source.name if self.source else None
        }


class NewsEvent(Base):
    """
    Extracted events from news articles
    """
    __tablename__ = "news_events"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Event identification
    article_id = Column(Integer, ForeignKey("news_articles.id"), nullable=False)
    ticker = Column(String(10), nullable=False, index=True)
    
    # Event details
    event_type = Column(SQLEnum(EventType), nullable=False, index=True)
    event_description = Column(Text, nullable=False)
    event_date = Column(DateTime, nullable=True)  # When the event occurred/will occur
    
    # Impact assessment
    impact_level = Column(SQLEnum(ImpactLevel), nullable=False)
    impact_score = Column(Float, nullable=True)  # Quantified impact (0-1)
    confidence = Column(Float, nullable=True)  # Confidence in event extraction (0-1)
    
    # Event metadata
    key_figures = Column(JSON, nullable=True)  # Important numbers, dates, people
    related_entities = Column(JSON, nullable=True)  # Companies, people, locations involved
    event_keywords = Column(JSON, nullable=True)  # Keywords that triggered event detection
    
    # Correlation data
    historical_correlation = Column(Float, nullable=True)  # Historical impact of similar events
    price_impact_predicted = Column(Float, nullable=True)  # Predicted price impact %
    
    # Metadata
    extracted_at = Column(DateTime, default=func.now())
    
    # Relationships
    article = relationship("NewsArticle", back_populates="events")
    
    # Indexes
    __table_args__ = (
        Index('idx_ticker_event_type', 'ticker', 'event_type'),
        Index('idx_event_date', 'event_date'),
        Index('idx_impact_level', 'impact_level'),
    )
    
    def __repr__(self):
        return f"<NewsEvent(ticker='{self.ticker}', type='{self.event_type.value}', impact='{self.impact_level.value}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            'id': self.id,
            'ticker': self.ticker,
            'event_type': self.event_type.value,
            'description': self.event_description,
            'impact_level': self.impact_level.value,
            'impact_score': self.impact_score,
            'confidence': self.confidence,
            'event_date': self.event_date.isoformat() if self.event_date else None,
            'extracted_at': self.extracted_at.isoformat()
        }


class NewsCorrelation(Base):
    """
    Correlation between news and stock price movements
    """
    __tablename__ = "news_correlations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Article and stock data
    article_id = Column(Integer, ForeignKey("news_articles.id"), nullable=False)
    ticker = Column(String(10), nullable=False, index=True)
    
    # Price movement data
    price_before = Column(Float, nullable=True)  # Price before news
    price_after_1h = Column(Float, nullable=True)  # Price 1 hour after
    price_after_1d = Column(Float, nullable=True)  # Price 1 day after
    price_after_1w = Column(Float, nullable=True)  # Price 1 week after
    
    # Calculated changes
    change_1h = Column(Float, nullable=True)  # % change in 1 hour
    change_1d = Column(Float, nullable=True)  # % change in 1 day
    change_1w = Column(Float, nullable=True)  # % change in 1 week
    
    # Volume data
    volume_before = Column(Float, nullable=True)
    volume_after_1h = Column(Float, nullable=True)
    volume_after_1d = Column(Float, nullable=True)
    volume_change_1d = Column(Float, nullable=True)  # % volume change
    
    # Correlation metrics
    correlation_score = Column(Float, nullable=True)  # Overall correlation (-1 to 1)
    statistical_significance = Column(Float, nullable=True)  # P-value
    lag_minutes = Column(Integer, nullable=True)  # Time lag for price reaction
    
    # Market context
    market_trend = Column(String(20), nullable=True)  # bullish, bearish, sideways
    sector_performance = Column(Float, nullable=True)  # Sector performance same period
    market_volatility = Column(Float, nullable=True)  # VIX or similar
    
    # Analysis metadata
    analysis_date = Column(DateTime, default=func.now())
    analysis_version = Column(String(10), default="1.0")
    
    # Relationships
    article = relationship("NewsArticle", back_populates="correlations")
    
    # Indexes
    __table_args__ = (
        Index('idx_ticker_correlation', 'ticker', 'correlation_score'),
        Index('idx_analysis_date', 'analysis_date'),
    )
    
    def __repr__(self):
        return f"<NewsCorrelation(ticker='{self.ticker}', correlation={self.correlation_score:.3f}, change_1d={self.change_1d:.2f}%)>"
    
    def calculate_correlation_strength(self) -> str:
        """Get correlation strength description"""
        if self.correlation_score is None:
            return "Unknown"
        
        abs_corr = abs(self.correlation_score)
        if abs_corr >= 0.7:
            return "Strong"
        elif abs_corr >= 0.4:
            return "Moderate"
        elif abs_corr >= 0.2:
            return "Weak"
        else:
            return "Very Weak"


class SentimentTrend(Base):
    """
    Aggregated sentiment trends over time for tickers
    """
    __tablename__ = "sentiment_trends"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Ticker and time data
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    time_period = Column(String(10), nullable=False)  # hourly, daily, weekly
    
    # Aggregated sentiment metrics
    avg_sentiment = Column(Float, nullable=True)  # Average sentiment (-1 to 1)
    sentiment_volatility = Column(Float, nullable=True)  # Sentiment volatility
    sentiment_momentum = Column(Float, nullable=True)  # Trend direction
    
    # Article counts by sentiment
    articles_total = Column(Integer, default=0)
    articles_positive = Column(Integer, default=0)
    articles_negative = Column(Integer, default=0)
    articles_neutral = Column(Integer, default=0)
    
    # Weighted metrics (by source reliability)
    weighted_sentiment = Column(Float, nullable=True)
    weighted_impact = Column(Float, nullable=True)
    
    # Breaking news indicators
    breaking_news_count = Column(Integer, default=0)
    high_impact_events = Column(Integer, default=0)
    
    # Metadata
    calculated_at = Column(DateTime, default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_ticker_date_period', 'ticker', 'date', 'time_period'),
        UniqueConstraint('ticker', 'date', 'time_period', name='uq_ticker_date_period'),
    )
    
    def __repr__(self):
        return f"<SentimentTrend(ticker='{self.ticker}', date='{self.date.date()}', sentiment={self.avg_sentiment:.3f})>"


class AlertLog(Base):
    """
    Log of alerts generated by the system
    """
    __tablename__ = "alert_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Alert details
    ticker = Column(String(10), nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)  # sentiment_spike, breaking_news, etc.
    severity = Column(String(20), nullable=False)  # low, medium, high, critical
    
    # Alert content
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    
    # Related data
    article_id = Column(Integer, ForeignKey("news_articles.id"), nullable=True)
    trigger_value = Column(Float, nullable=True)  # Value that triggered alert
    threshold_value = Column(Float, nullable=True)  # Threshold that was exceeded
    
    # Alert metadata
    created_at = Column(DateTime, default=func.now())
    sent_at = Column(DateTime, nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    
    # Delivery status
    email_sent = Column(Boolean, default=False)
    slack_sent = Column(Boolean, default=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_ticker_severity', 'ticker', 'severity'),
        Index('idx_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<AlertLog(ticker='{self.ticker}', type='{self.alert_type}', severity='{self.severity}')>"


class SystemMetrics(Base):
    """
    System performance and health metrics
    """
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Timestamp
    timestamp = Column(DateTime, default=func.now(), index=True)
    
    # Scraping metrics
    articles_scraped_hour = Column(Integer, default=0)
    articles_processed_hour = Column(Integer, default=0)
    api_calls_hour = Column(Integer, default=0)
    scraping_errors_hour = Column(Integer, default=0)
    
    # Processing metrics
    sentiment_analysis_latency_ms = Column(Float, nullable=True)
    event_extraction_latency_ms = Column(Float, nullable=True)
    correlation_analysis_latency_ms = Column(Float, nullable=True)
    
    # System health
    memory_usage_mb = Column(Float, nullable=True)
    cpu_usage_percent = Column(Float, nullable=True)
    disk_usage_percent = Column(Float, nullable=True)
    active_connections = Column(Integer, nullable=True)
    
    # Data quality metrics
    duplicate_articles_percent = Column(Float, nullable=True)
    low_quality_articles_percent = Column(Float, nullable=True)
    sentiment_confidence_avg = Column(Float, nullable=True)
    
    def __repr__(self):
        return f"<SystemMetrics(timestamp='{self.timestamp}', articles_scraped={self.articles_scraped_hour})>"


# Database utility functions
def create_all_tables(engine):
    """Create all tables in the database"""
    Base.metadata.create_all(engine)


def drop_all_tables(engine):
    """Drop all tables from the database"""
    Base.metadata.drop_all(engine)


def get_table_names():
    """Get list of all table names"""
    return [table.name for table in Base.metadata.tables.values()]


# Query helper functions
class NewsQueries:
    """Helper class for common database queries"""
    
    @staticmethod
    def get_recent_articles(session: Session, ticker: str, hours: int = 24) -> List[NewsArticle]:
        """Get recent articles for a ticker"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return session.query(NewsArticle).filter(
            NewsArticle.ticker == ticker,
            NewsArticle.published_date >= cutoff_time
        ).order_by(NewsArticle.published_date.desc()).all()
    
    @staticmethod
    def get_sentiment_summary(session: Session, ticker: str, days: int = 7) -> Dict[str, Any]:
        """Get sentiment summary for a ticker"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        articles = session.query(NewsArticle).filter(
            NewsArticle.ticker == ticker,
            NewsArticle.published_date >= cutoff_time,
            NewsArticle.sentiment_score.isnot(None)
        ).all()
        
        if not articles:
            return {'total': 0, 'avg_sentiment': None}
        
        sentiments = [a.sentiment_score for a in articles if a.sentiment_score is not None]
        
        return {
            'total': len(articles),
            'avg_sentiment': sum(sentiments) / len(sentiments) if sentiments else None,
            'positive': len([s for s in sentiments if s > 0.1]),
            'negative': len([s for s in sentiments if s < -0.1]),
            'neutral': len([s for s in sentiments if -0.1 <= s <= 0.1])
        }
    
    @staticmethod
    def get_high_impact_events(session: Session, ticker: str, days: int = 7) -> List[NewsEvent]:
        """Get high impact events for a ticker"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        return session.query(NewsEvent).filter(
            NewsEvent.ticker == ticker,
            NewsEvent.extracted_at >= cutoff_time,
            NewsEvent.impact_level.in_([ImpactLevel.HIGH, ImpactLevel.CRITICAL])
        ).order_by(NewsEvent.extracted_at.desc()).all()
    
    @staticmethod
    def get_breaking_news(session: Session, hours: int = 2) -> List[NewsArticle]:
        """Get recent breaking news across all tickers"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return session.query(NewsArticle).filter(
            NewsArticle.is_breaking_news == True,
            NewsArticle.published_date >= cutoff_time
        ).order_by(NewsArticle.published_date.desc()).all()


if __name__ == "__main__":
    # Test model creation
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Create test database
    engine = create_engine("sqlite:///test_news_intelligence.db", echo=True)
    create_all_tables(engine)
    
    # Create session
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    # Test model creation
    source = NewsSource(
        name="Reuters",
        url="https://reuters.com",
        source_type="web_scraping",
        tier="tier_1",
        reliability_weight=1.0,
        confidence_score=0.95
    )
    
    session.add(source)
    session.commit()
    
    article = NewsArticle(
        url="https://reuters.com/test-article",
        url_hash=NewsArticle.create_url_hash("https://reuters.com/test-article"),
        title="Test Article About AAPL",
        content="This is a test article about Apple Inc.",
        ticker="AAPL",
        source_id=source.id,
        published_date=datetime.utcnow(),
        sentiment_score=0.75,
        sentiment_type=SentimentType.POSITIVE,
        impact_score=0.6
    )
    
    session.add(article)
    session.commit()
    
    print(f"Created article: {article}")
    print(f"Article sentiment: {article.get_sentiment_label()}")
    
    # Test queries
    recent_articles = NewsQueries.get_recent_articles(session, "AAPL")
    print(f"Recent AAPL articles: {len(recent_articles)}")
    
    sentiment_summary = NewsQueries.get_sentiment_summary(session, "AAPL")
    print(f"Sentiment summary: {sentiment_summary}")
    
    session.close()
    print("Database models test completed successfully!")