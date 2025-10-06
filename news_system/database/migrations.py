"""
Database migrations manager for Phase 3 News Intelligence Engine.

This module handles database schema versioning, migrations, and upgrades.
Ensures database schema stays in sync across different environments.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Callable
from dataclasses import dataclass

from .connection import get_database, DatabaseConnection

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class Migration:
    """Database migration definition."""
    version: str
    description: str
    up_sql: str
    down_sql: str
    applied_at: datetime = None

class MigrationManager:
    """Manages database schema migrations."""
    
    def __init__(self, db: DatabaseConnection = None):
        self.db = db or get_database()
        self._ensure_migrations_table()
        self._migrations = self._get_migrations()
    
    def _ensure_migrations_table(self):
        """Create migrations tracking table if it doesn't exist."""
        if self.db.config.db_type == "sqlite":
            schema = """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        else:  # PostgreSQL
            schema = """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        
        with self.db.get_cursor() as cursor:
            cursor.execute(schema)
            self.db.connection.commit()
    
    def _get_migrations(self) -> List[Migration]:
        """Define all database migrations in order."""
        return [
            Migration(
                version="001",
                description="Add indexes for performance optimization",
                up_sql="""
                    CREATE INDEX IF NOT EXISTS idx_news_articles_published_at ON news_articles(published_at);
                    CREATE INDEX IF NOT EXISTS idx_news_articles_source ON news_articles(source);
                    CREATE INDEX IF NOT EXISTS idx_news_articles_sentiment ON news_articles(sentiment_label);
                    CREATE INDEX IF NOT EXISTS idx_news_articles_processed ON news_articles(processed);
                    CREATE INDEX IF NOT EXISTS idx_stock_mentions_symbol ON stock_mentions(stock_symbol);
                    CREATE INDEX IF NOT EXISTS idx_stock_mentions_article ON stock_mentions(article_id);
                    CREATE INDEX IF NOT EXISTS idx_alerts_sent ON alerts(sent);
                    CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type);
                    CREATE INDEX IF NOT EXISTS idx_sentiment_analysis_article ON sentiment_analysis(article_id);
                """,
                down_sql="""
                    DROP INDEX IF EXISTS idx_news_articles_published_at;
                    DROP INDEX IF EXISTS idx_news_articles_source;
                    DROP INDEX IF EXISTS idx_news_articles_sentiment;
                    DROP INDEX IF EXISTS idx_news_articles_processed;
                    DROP INDEX IF EXISTS idx_stock_mentions_symbol;
                    DROP INDEX IF EXISTS idx_stock_mentions_article;
                    DROP INDEX IF EXISTS idx_alerts_sent;
                    DROP INDEX IF EXISTS idx_alerts_type;
                    DROP INDEX IF EXISTS idx_sentiment_analysis_article;
                """
            ),
            Migration(
                version="002",
                description="Add article duplicate detection fields",
                up_sql="""
                    ALTER TABLE news_articles ADD COLUMN content_hash TEXT;
                    ALTER TABLE news_articles ADD COLUMN duplicate_of INTEGER;
                    CREATE INDEX IF NOT EXISTS idx_news_articles_content_hash ON news_articles(content_hash);
                """ if self.db.config.db_type == "sqlite" else """
                    ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS content_hash TEXT;
                    ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS duplicate_of INTEGER;
                    CREATE INDEX IF NOT EXISTS idx_news_articles_content_hash ON news_articles(content_hash);
                """,
                down_sql="""
                    -- SQLite doesn't support DROP COLUMN, so we'd need to recreate the table
                    -- For now, we'll just remove the index
                    DROP INDEX IF EXISTS idx_news_articles_content_hash;
                """
            ),
            Migration(
                version="003",
                description="Add news source reliability tracking",
                up_sql="""
                    ALTER TABLE news_sources ADD COLUMN articles_count INTEGER DEFAULT 0;
                    ALTER TABLE news_sources ADD COLUMN error_rate REAL DEFAULT 0.0;
                    ALTER TABLE news_sources ADD COLUMN last_error TEXT;
                    ALTER TABLE news_sources ADD COLUMN last_error_at TIMESTAMP;
                """ if self.db.config.db_type == "sqlite" else """
                    ALTER TABLE news_sources ADD COLUMN IF NOT EXISTS articles_count INTEGER DEFAULT 0;
                    ALTER TABLE news_sources ADD COLUMN IF NOT EXISTS error_rate REAL DEFAULT 0.0;
                    ALTER TABLE news_sources ADD COLUMN IF NOT EXISTS last_error TEXT;
                    ALTER TABLE news_sources ADD COLUMN IF NOT EXISTS last_error_at TIMESTAMP;
                """,
                down_sql="""
                    -- SQLite doesn't support DROP COLUMN easily
                    -- For PostgreSQL:
                    -- ALTER TABLE news_sources DROP COLUMN IF EXISTS articles_count;
                    -- ALTER TABLE news_sources DROP COLUMN IF EXISTS error_rate;
                    -- ALTER TABLE news_sources DROP COLUMN IF EXISTS last_error;
                    -- ALTER TABLE news_sources DROP COLUMN IF EXISTS last_error_at;
                """
            ),
            Migration(
                version="004",
                description="Add article categorization and tagging",
                up_sql="""
                    CREATE TABLE IF NOT EXISTS article_categories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE TABLE IF NOT EXISTS article_tags (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_id INTEGER NOT NULL,
                        tag TEXT NOT NULL,
                        confidence REAL DEFAULT 1.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (article_id) REFERENCES news_articles (id)
                    );
                    
                    ALTER TABLE news_articles ADD COLUMN category_id INTEGER;
                    CREATE INDEX IF NOT EXISTS idx_article_tags_article ON article_tags(article_id);
                    CREATE INDEX IF NOT EXISTS idx_article_tags_tag ON article_tags(tag);
                """ if self.db.config.db_type == "sqlite" else """
                    CREATE TABLE IF NOT EXISTS article_categories (
                        id SERIAL PRIMARY KEY,
                        name TEXT UNIQUE NOT NULL,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE TABLE IF NOT EXISTS article_tags (
                        id SERIAL PRIMARY KEY,
                        article_id INTEGER NOT NULL REFERENCES news_articles(id),
                        tag TEXT NOT NULL,
                        confidence REAL DEFAULT 1.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS category_id INTEGER;
                    CREATE INDEX IF NOT EXISTS idx_article_tags_article ON article_tags(article_id);
                    CREATE INDEX IF NOT EXISTS idx_article_tags_tag ON article_tags(tag);
                """,
                down_sql="""
                    DROP TABLE IF EXISTS article_tags;
                    DROP TABLE IF EXISTS article_categories;
                    DROP INDEX IF EXISTS idx_article_tags_article;
                    DROP INDEX IF EXISTS idx_article_tags_tag;
                """
            ),
            Migration(
                version="005",
                description="Add real-time processing queue",
                up_sql="""
                    CREATE TABLE IF NOT EXISTS processing_queue (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_id INTEGER NOT NULL,
                        processing_type TEXT NOT NULL,
                        priority INTEGER DEFAULT 5,
                        status TEXT DEFAULT 'pending',
                        attempts INTEGER DEFAULT 0,
                        max_attempts INTEGER DEFAULT 3,
                        error_message TEXT,
                        scheduled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (article_id) REFERENCES news_articles (id)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_processing_queue_status ON processing_queue(status);
                    CREATE INDEX IF NOT EXISTS idx_processing_queue_priority ON processing_queue(priority);
                    CREATE INDEX IF NOT EXISTS idx_processing_queue_scheduled ON processing_queue(scheduled_at);
                """ if self.db.config.db_type == "sqlite" else """
                    CREATE TABLE IF NOT EXISTS processing_queue (
                        id SERIAL PRIMARY KEY,
                        article_id INTEGER NOT NULL REFERENCES news_articles(id),
                        processing_type TEXT NOT NULL,
                        priority INTEGER DEFAULT 5,
                        status TEXT DEFAULT 'pending',
                        attempts INTEGER DEFAULT 0,
                        max_attempts INTEGER DEFAULT 3,
                        error_message TEXT,
                        scheduled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_processing_queue_status ON processing_queue(status);
                    CREATE INDEX IF NOT EXISTS idx_processing_queue_priority ON processing_queue(priority);
                    CREATE INDEX IF NOT EXISTS idx_processing_queue_scheduled ON processing_queue(scheduled_at);
                """,
                down_sql="""
                    DROP TABLE IF EXISTS processing_queue;
                    DROP INDEX IF EXISTS idx_processing_queue_status;
                    DROP INDEX IF EXISTS idx_processing_queue_priority;
                    DROP INDEX IF EXISTS idx_processing_queue_scheduled;
                """
            )
        ]
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions."""
        query = "SELECT version FROM schema_migrations ORDER BY version"
        results = self.db.execute_query(query)
        return [row['version'] for row in results]
    
    def get_pending_migrations(self) -> List[Migration]:
        """Get list of pending migrations."""
        applied = set(self.get_applied_migrations())
        return [m for m in self._migrations if m.version not in applied]
    
    def apply_migration(self, migration: Migration) -> bool:
        """Apply a single migration."""
        try:
            logger.info(f"Applying migration {migration.version}: {migration.description}")
            
            # Execute migration SQL
            for sql_statement in migration.up_sql.strip().split(';'):
                sql_statement = sql_statement.strip()
                if sql_statement:
                    with self.db.get_cursor() as cursor:
                        cursor.execute(sql_statement)
            
            # Record migration as applied
            insert_query = """
                INSERT INTO schema_migrations (version, description, applied_at)
                VALUES (?, ?, ?)
            """
            self.db.execute_insert(insert_query, (
                migration.version,
                migration.description,
                datetime.now()
            ))
            
            logger.info(f"Successfully applied migration {migration.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply migration {migration.version}: {e}")
            raise
    
    def rollback_migration(self, migration: Migration) -> bool:
        """Rollback a single migration."""
        try:
            logger.info(f"Rolling back migration {migration.version}: {migration.description}")
            
            # Execute rollback SQL
            for sql_statement in migration.down_sql.strip().split(';'):
                sql_statement = sql_statement.strip()
                if sql_statement:
                    with self.db.get_cursor() as cursor:
                        cursor.execute(sql_statement)
            
            # Remove migration record
            delete_query = "DELETE FROM schema_migrations WHERE version = ?"
            self.db.execute_update(delete_query, (migration.version,))
            
            logger.info(f"Successfully rolled back migration {migration.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback migration {migration.version}: {e}")
            raise
    
    def migrate_up(self, target_version: str = None) -> int:
        """Apply all pending migrations up to target version."""
        pending = self.get_pending_migrations()
        
        if target_version:
            # Filter to only migrations up to target version
            pending = [m for m in pending if m.version <= target_version]
        
        applied_count = 0
        for migration in pending:
            self.apply_migration(migration)
            applied_count += 1
        
        logger.info(f"Applied {applied_count} migrations")
        return applied_count
    
    def migrate_down(self, target_version: str) -> int:
        """Rollback migrations down to target version."""
        applied = self.get_applied_migrations()
        
        # Find migrations to rollback (in reverse order)
        to_rollback = []
        for version in reversed(applied):
            if version > target_version:
                migration = next((m for m in self._migrations if m.version == version), None)
                if migration:
                    to_rollback.append(migration)
        
        rollback_count = 0
        for migration in to_rollback:
            self.rollback_migration(migration)
            rollback_count += 1
        
        logger.info(f"Rolled back {rollback_count} migrations")
        return rollback_count
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        applied = self.get_applied_migrations()
        pending = self.get_pending_migrations()
        
        return {
            "database_type": self.db.config.db_type,
            "total_migrations": len(self._migrations),
            "applied_count": len(applied),
            "pending_count": len(pending),
            "applied_versions": applied,
            "pending_versions": [m.version for m in pending],
            "latest_version": self._migrations[-1].version if self._migrations else None,
            "current_version": applied[-1] if applied else None
        }
    
    def reset_database(self) -> bool:
        """Reset database by rolling back all migrations."""
        try:
            applied = self.get_applied_migrations()
            for version in reversed(applied):
                migration = next((m for m in self._migrations if m.version == version), None)
                if migration:
                    self.rollback_migration(migration)
            
            logger.info("Database reset completed")
            return True
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            raise

def migrate_database(action: str = "up", target_version: str = None) -> Dict[str, Any]:
    """Convenience function to run migrations."""
    manager = MigrationManager()
    
    if action == "up":
        count = manager.migrate_up(target_version)
        return {"action": "migrate_up", "migrations_applied": count}
    
    elif action == "down":
        if not target_version:
            raise ValueError("Target version required for rollback")
        count = manager.migrate_down(target_version)
        return {"action": "migrate_down", "migrations_rolled_back": count}
    
    elif action == "status":
        return manager.get_migration_status()
    
    elif action == "reset":
        manager.reset_database()
        return {"action": "reset", "status": "completed"}
    
    else:
        raise ValueError(f"Unknown migration action: {action}")

# CLI interface for migrations
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.database.migrations <action> [target_version]")
        print("Actions: up, down, status, reset")
        sys.exit(1)
    
    action = sys.argv[1]
    target_version = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        result = migrate_database(action, target_version)
        print(f"Migration result: {result}")
        
        # Show current status
        status = migrate_database("status")
        print(f"\nCurrent status:")
        print(f"  Database: {status['database_type']}")
        print(f"  Current version: {status['current_version']}")
        print(f"  Latest version: {status['latest_version']}")
        print(f"  Applied: {status['applied_count']}/{status['total_migrations']}")
        
        if status['pending_count'] > 0:
            print(f"  Pending migrations: {status['pending_versions']}")
        
    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)