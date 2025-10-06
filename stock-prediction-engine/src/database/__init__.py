"""
Database package for Phase 3 News Intelligence Engine.

This package provides database connectivity, session management, and migrations
for the news intelligence system.
"""

from .connection import DatabaseConnection, DatabaseConfig, get_database, close_database
from .session import (
    DatabaseSession, 
    NewsArticle, 
    NewsSource, 
    SentimentAnalysis, 
    StockMention, 
    Alert, 
    CollectionMetric,
    get_session
)
from .migrations import MigrationManager, migrate_database

__version__ = "1.0.0"
__all__ = [
    # Connection management
    "DatabaseConnection",
    "DatabaseConfig", 
    "get_database",
    "close_database",
    
    # Session management
    "DatabaseSession",
    "get_session",
    
    # Data models
    "NewsArticle",
    "NewsSource", 
    "SentimentAnalysis",
    "StockMention",
    "Alert",
    "CollectionMetric",
    
    # Migrations
    "MigrationManager",
    "migrate_database"
]

# Package-level convenience functions
def initialize_database(config=None):
    """Initialize database with tables and run migrations."""
    from .connection import DatabaseConnection
    from .migrations import MigrationManager
    
    # Initialize database connection
    db = DatabaseConnection(config)
    
    # Run migrations
    migration_manager = MigrationManager(db)
    migration_manager.migrate_up()
    
    return db

def get_database_status():
    """Get comprehensive database status including migrations."""
    from .migrations import migrate_database
    
    # Get migration status
    migration_status = migrate_database("status")
    
    # Get database health
    db = get_database()
    health_status = db.health_check()
    
    return {
        "health": health_status,
        "migrations": migration_status,
        "package_version": __version__
    }

# Initialize logging for the package
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())