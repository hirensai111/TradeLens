"""
Database connection manager for Phase 3 News Intelligence Engine.

This module handles database connections, table creation, and basic database operations.
Supports both SQLite (development) and PostgreSQL (production).
"""

import os
import sqlite3
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    db_type: str = "sqlite"  # "sqlite" or "postgresql"
    host: str = "localhost"
    port: int = 5432
    database: str = "news_intelligence"
    username: str = ""
    password: str = ""
    sqlite_path: str = "data/news_intelligence.db"

class DatabaseConnection:
    """Manages database connections and operations."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or self._load_config()
        self.connection = None
        self._initialize_database()
    
    def _load_config(self) -> DatabaseConfig:
        """Load database configuration from environment variables."""
        return DatabaseConfig(
            db_type=os.getenv("DB_TYPE", "sqlite"),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "news_intelligence"),
            username=os.getenv("DB_USER", ""),
            password=os.getenv("DB_PASSWORD", ""),
            sqlite_path=os.getenv("SQLITE_PATH", "data/news_intelligence.db")
        )
    
    def _initialize_database(self):
        """Initialize database and create tables if they don't exist."""
        try:
            self.connect()
            self.create_tables()
            logger.info(f"Database initialized successfully ({self.config.db_type})")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def connect(self):
        """Establish database connection."""
        if self.config.db_type == "sqlite":
            self._connect_sqlite()
        elif self.config.db_type == "postgresql":
            self._connect_postgresql()
        else:
            raise ValueError(f"Unsupported database type: {self.config.db_type}")
    
    def _connect_sqlite(self):
        """Connect to SQLite database."""
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.config.sqlite_path), exist_ok=True)
        
        self.connection = sqlite3.connect(
            self.config.sqlite_path,
            check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row  # Enable dict-like access
        logger.info(f"Connected to SQLite database: {self.config.sqlite_path}")
    
    def _connect_postgresql(self):
        """Connect to PostgreSQL database."""
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 is required for PostgreSQL connections")
        
        self.connection = psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.username,
            password=self.config.password,
            cursor_factory=RealDictCursor
        )
        logger.info(f"Connected to PostgreSQL database: {self.config.host}:{self.config.port}")
    
    def create_tables(self):
        """Create all necessary tables for the news intelligence system."""
        tables = self._get_table_schemas()
        
        with self.get_cursor() as cursor:
            for table_name, schema in tables.items():
                cursor.execute(schema)
                logger.info(f"Created/verified table: {table_name}")
            
            self.connection.commit()
    
    def _get_table_schemas(self) -> Dict[str, str]:
        """Get SQL schemas for all tables."""
        if self.config.db_type == "sqlite":
            return self._get_sqlite_schemas()
        else:
            return self._get_postgresql_schemas()
    
    def _get_sqlite_schemas(self) -> Dict[str, str]:
        """SQLite table schemas."""
        return {
            "news_articles": """
                CREATE TABLE IF NOT EXISTS news_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT,
                    summary TEXT,
                    source TEXT NOT NULL,
                    author TEXT,
                    url TEXT UNIQUE,
                    published_at TIMESTAMP,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    keywords TEXT,  -- JSON array of keywords
                    stock_symbols TEXT,  -- JSON array of mentioned stocks
                    event_type TEXT,
                    impact_score REAL,
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "news_sources": """
                CREATE TABLE IF NOT EXISTS news_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    url TEXT,
                    source_type TEXT,  -- 'rss', 'api', 'scraper'
                    reliability_score REAL DEFAULT 0.5,
                    active BOOLEAN DEFAULT TRUE,
                    last_collected TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "sentiment_analysis": """
                CREATE TABLE IF NOT EXISTS sentiment_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id INTEGER,
                    sentiment_score REAL NOT NULL,
                    sentiment_label TEXT NOT NULL,
                    confidence REAL,
                    analyzer_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (article_id) REFERENCES news_articles (id)
                )
            """,
            "stock_mentions": """
                CREATE TABLE IF NOT EXISTS stock_mentions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id INTEGER,
                    stock_symbol TEXT NOT NULL,
                    mention_count INTEGER DEFAULT 1,
                    context TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (article_id) REFERENCES news_articles (id)
                )
            """,
            "alerts": """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT,
                    priority TEXT DEFAULT 'medium',
                    stock_symbol TEXT,
                    triggered_by INTEGER,  -- article_id
                    sent BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (triggered_by) REFERENCES news_articles (id)
                )
            """,
            "collection_metrics": """
                CREATE TABLE IF NOT EXISTS collection_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL,
                    articles_collected INTEGER DEFAULT 0,
                    errors_count INTEGER DEFAULT 0,
                    collection_time REAL,  -- seconds
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        }
    
    def _get_postgresql_schemas(self) -> Dict[str, str]:
        """PostgreSQL table schemas."""
        return {
            "news_articles": """
                CREATE TABLE IF NOT EXISTS news_articles (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT,
                    summary TEXT,
                    source TEXT NOT NULL,
                    author TEXT,
                    url TEXT UNIQUE,
                    published_at TIMESTAMP,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    keywords JSONB,  -- JSON array of keywords
                    stock_symbols JSONB,  -- JSON array of mentioned stocks
                    event_type TEXT,
                    impact_score REAL,
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "news_sources": """
                CREATE TABLE IF NOT EXISTS news_sources (
                    id SERIAL PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    url TEXT,
                    source_type TEXT,  -- 'rss', 'api', 'scraper'
                    reliability_score REAL DEFAULT 0.5,
                    active BOOLEAN DEFAULT TRUE,
                    last_collected TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "sentiment_analysis": """
                CREATE TABLE IF NOT EXISTS sentiment_analysis (
                    id SERIAL PRIMARY KEY,
                    article_id INTEGER REFERENCES news_articles(id),
                    sentiment_score REAL NOT NULL,
                    sentiment_label TEXT NOT NULL,
                    confidence REAL,
                    analyzer_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "stock_mentions": """
                CREATE TABLE IF NOT EXISTS stock_mentions (
                    id SERIAL PRIMARY KEY,
                    article_id INTEGER REFERENCES news_articles(id),
                    stock_symbol TEXT NOT NULL,
                    mention_count INTEGER DEFAULT 1,
                    context TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "alerts": """
                CREATE TABLE IF NOT EXISTS alerts (
                    id SERIAL PRIMARY KEY,
                    alert_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT,
                    priority TEXT DEFAULT 'medium',
                    stock_symbol TEXT,
                    triggered_by INTEGER REFERENCES news_articles(id),
                    sent BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "collection_metrics": """
                CREATE TABLE IF NOT EXISTS collection_metrics (
                    id SERIAL PRIMARY KEY,
                    source_name TEXT NOT NULL,
                    articles_collected INTEGER DEFAULT 0,
                    errors_count INTEGER DEFAULT 0,
                    collection_time REAL,  -- seconds
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        }
    
    @contextmanager
    def get_cursor(self):
        """Context manager for database cursor."""
        cursor = self.connection.cursor()
        try:
            yield cursor
        finally:
            cursor.close()
    
    def execute_query(self, query: str, params: tuple = None) -> Any:
        """Execute a query and return results."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params or ())
            return cursor.fetchall()
    
    def execute_insert(self, query: str, params: tuple = None) -> int:
        """Execute an insert query and return the inserted ID."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params or ())
            self.connection.commit()
            
            if self.config.db_type == "sqlite":
                return cursor.lastrowid
            else:
                return cursor.fetchone()[0]
    
    def execute_update(self, query: str, params: tuple = None) -> int:
        """Execute an update query and return affected rows."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params or ())
            self.connection.commit()
            return cursor.rowcount
    
    def health_check(self) -> Dict[str, Any]:
        """Check database health and return status."""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
            
            return {
                "status": "healthy",
                "db_type": self.config.db_type,
                "timestamp": datetime.now().isoformat(),
                "connection_active": result is not None
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "db_type": self.config.db_type,
                "timestamp": datetime.now().isoformat(),
                "connection_active": False
            }
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

# Global database instance
_db_instance = None

def get_database() -> DatabaseConnection:
    """Get singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseConnection()
    return _db_instance

def close_database():
    """Close global database instance."""
    global _db_instance
    if _db_instance:
        _db_instance.close()
        _db_instance = None

# Example usage and testing
if __name__ == "__main__":
    # Test database connection
    db = DatabaseConnection()
    
    # Test health check
    health = db.health_check()
    print(f"Database health: {health}")
    
    # Test basic operations
    try:
        # Insert a test news source
        source_id = db.execute_insert(
            "INSERT INTO news_sources (name, url, source_type) VALUES (?, ?, ?)",
            ("Test Source", "https://example.com", "api")
        )
        print(f"Inserted news source with ID: {source_id}")
        
        # Query news sources
        sources = db.execute_query("SELECT * FROM news_sources")
        print(f"News sources: {sources}")
        
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        db.close()