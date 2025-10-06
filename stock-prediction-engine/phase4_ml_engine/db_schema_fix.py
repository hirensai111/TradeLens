#!/usr/bin/env python3
"""
🔧 Database Schema Fix Script
Ensures your existing database has all required columns for automation integration
"""

import os
import sqlite3
import logging
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)

class DatabaseSchemaMigrator:
    """Migrate existing database to support automation features"""
    
    def __init__(self, db_path: str = "indian_trading_bot.db"):
        self.db_path = db_path
        self.backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def backup_database(self):
        """Create backup of existing database"""
        if os.path.exists(self.db_path):
            try:
                shutil.copy2(self.db_path, self.backup_path)
                print(f"✅ Database backed up to: {self.backup_path}")
                return True
            except Exception as e:
                print(f"❌ Backup failed: {e}")
                return False
        else:
            print("ℹ️ No existing database found - will create new one")
            return True
    
    def check_current_schema(self):
        """Check current database schema"""
        if not os.path.exists(self.db_path):
            print("ℹ️ Database doesn't exist yet")
            return {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check signals table
            cursor.execute("PRAGMA table_info(signals)")
            signals_columns = cursor.fetchall()
            signals_column_names = [col[1] for col in signals_columns]
            
            # Check other tables
            tables_info = {}
            for table in ['signals', 'trades', 'performance', 'patterns']:
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                table_exists = cursor.fetchone() is not None
                
                if table_exists:
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = cursor.fetchall()
                    tables_info[table] = [col[1] for col in columns]
                else:
                    tables_info[table] = []
            
            conn.close()
            
            print(f"📊 Current Database Schema:")
            for table, columns in tables_info.items():
                if columns:
                    print(f"   {table}: {len(columns)} columns")
                else:
                    print(f"   {table}: ❌ Missing")
            
            return tables_info
            
        except Exception as e:
            print(f"❌ Error checking schema: {e}")
            return {}
    
    def migrate_schema(self):
        """Migrate database schema to support automation"""
        print("\n🔄 Starting Database Schema Migration...")
        
        # Create backup first
        if not self.backup_database():
            print("❌ Cannot proceed without backup")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            # Enable WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode = WAL")
            cursor.execute("PRAGMA synchronous = NORMAL")
            
            # 1. Check and update signals table
            print("\n🔧 Migrating signals table...")
            
            # Get current columns
            cursor.execute("PRAGMA table_info(signals)")
            existing_columns = [col[1] for col in cursor.fetchall()]
            
            # Required automation columns
            automation_columns = {
                'processed_by_automation': 'BOOLEAN DEFAULT 0',
                'automation_timestamp': 'DATETIME',
                'source': 'TEXT DEFAULT "signal_generator"'
            }
            
            # Add missing columns
            for col_name, col_def in automation_columns.items():
                if col_name not in existing_columns:
                    try:
                        cursor.execute(f"ALTER TABLE signals ADD COLUMN {col_name} {col_def}")
                        print(f"   ✅ Added column: {col_name}")
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" not in str(e).lower():
                            print(f"   ❌ Error adding {col_name}: {e}")
                else:
                    print(f"   ℹ️ Column {col_name} already exists")
            
            # 2. Create missing tables if needed
            print("\n🔧 Checking other tables...")
            
            # Create trades table if missing
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
            if not cursor.fetchone():
                cursor.execute("""
                    CREATE TABLE trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_id INTEGER,
                        timestamp DATETIME NOT NULL,
                        ticker TEXT NOT NULL,
                        action TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        price REAL NOT NULL,
                        commission REAL DEFAULT 0,
                        trade_type TEXT,
                        notes TEXT,
                        FOREIGN KEY (signal_id) REFERENCES signals (id)
                    )
                """)
                print("   ✅ Created trades table")
            
            # Create performance table if missing
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='performance'")
            if not cursor.fetchone():
                cursor.execute("""
                    CREATE TABLE performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE NOT NULL UNIQUE,
                        signals_sent INTEGER DEFAULT 0,
                        high_confidence_signals INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        losing_trades INTEGER DEFAULT 0,
                        total_pnl REAL DEFAULT 0,
                        best_trade REAL DEFAULT 0,
                        worst_trade REAL DEFAULT 0,
                        daily_summary TEXT
                    )
                """)
                print("   ✅ Created performance table")
            
            # Create patterns table if missing
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patterns'")
            if not cursor.fetchone():
                cursor.execute("""
                    CREATE TABLE patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_name TEXT NOT NULL UNIQUE,
                        occurrences INTEGER DEFAULT 0,
                        successful INTEGER DEFAULT 0,
                        average_return REAL DEFAULT 0,
                        last_seen DATETIME
                    )
                """)
                print("   ✅ Created patterns table")
            
            # 3. Create indices for better performance
            print("\n🔧 Creating performance indices...")
            
            indices = [
                ("idx_signals_processed", "signals", "processed_by_automation, status"),
                ("idx_signals_timestamp", "signals", "timestamp DESC"),
                ("idx_signals_ticker", "signals", "ticker"),
            ]
            
            for index_name, table, columns in indices:
                try:
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({columns})")
                    print(f"   ✅ Created index: {index_name}")
                except sqlite3.OperationalError as e:
                    if "already exists" not in str(e).lower():
                        print(f"   ⚠️ Index creation warning for {index_name}: {e}")
            
            # Commit all changes
            conn.commit()
            print("\n✅ Database schema migration completed successfully!")
            
            return True
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return False
        
        finally:
            try:
                conn.close()
            except:
                pass
    
    def verify_schema(self):
        """Verify that schema migration was successful"""
        print("\n🔍 Verifying Schema Migration...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check signals table has automation columns
            cursor.execute("PRAGMA table_info(signals)")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            required_columns = ['processed_by_automation', 'automation_timestamp', 'source']
            missing_columns = [col for col in required_columns if col not in column_names]
            
            if missing_columns:
                print(f"❌ Missing columns: {missing_columns}")
                return False
            
            # Check required tables exist
            required_tables = ['signals', 'trades', 'performance', 'patterns']
            for table in required_tables:
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                if not cursor.fetchone():
                    print(f"❌ Missing table: {table}")
                    return False
            
            # Check indices exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indices = [row[0] for row in cursor.fetchall()]
            
            required_indices = ['idx_signals_processed', 'idx_signals_timestamp', 'idx_signals_ticker']
            missing_indices = [idx for idx in required_indices if idx not in indices]
            
            if missing_indices:
                print(f"⚠️ Missing indices (performance may be affected): {missing_indices}")
            
            conn.close()
            
            print("✅ Schema verification passed!")
            print("🎉 Database is ready for automation integration!")
            
            return True
            
        except Exception as e:
            print(f"❌ Schema verification failed: {e}")
            return False
    
    def run_full_migration(self):
        """Run complete database migration process"""
        print("🔧 Starting Complete Database Migration Process...")
        print("=" * 60)
        
        # Step 1: Check current schema
        print("Step 1: Analyzing current database schema...")
        current_schema = self.check_current_schema()
        
        # Step 2: Create backup
        print(f"\nStep 2: Creating backup...")
        if not self.backup_database():
            print("❌ Cannot proceed without backup!")
            return False
        
        # Step 3: Run migration
        print(f"\nStep 3: Migrating database schema...")
        if not self.migrate_schema():
            print("❌ Migration failed!")
            return False
        
        # Step 4: Verify migration
        print(f"\nStep 4: Verifying migration...")
        if not self.verify_schema():
            print("❌ Migration verification failed!")
            return False
        
        print("\n" + "=" * 60)
        print("🎉 DATABASE MIGRATION COMPLETED SUCCESSFULLY!")
        print("✅ Your database now supports automation integration")
        print(f"💾 Backup saved at: {self.backup_path}")
        print("🚀 You can now run the Indian Trading Bot safely")
        
        return True


def main():
    """Main function to run database migration"""
    print("🔧 Indian Trading Bot Database Migration Tool")
    print("=" * 50)
    
    import argparse
    parser = argparse.ArgumentParser(description='Migrate Indian Trading Bot database for automation support')
    parser.add_argument('--db-path', default='indian_trading_bot.db', help='Path to database file')
    parser.add_argument('--check-only', action='store_true', help='Only check current schema without migration')
    
    args = parser.parse_args()
    
    migrator = DatabaseSchemaMigrator(args.db_path)
    
    if args.check_only:
        print("📊 Checking current database schema only...")
        current_schema = migrator.check_current_schema()
        
        # Check if migration is needed
        if os.path.exists(args.db_path):
            verification_result = migrator.verify_schema()
            if verification_result:
                print("\n✅ Database schema is up-to-date!")
                print("🚀 No migration needed - bot can run safely")
            else:
                print("\n⚠️ Database schema needs migration")
                print("💡 Run without --check-only flag to migrate")
        else:
            print("\n📝 No database found - will be created on first bot run")
    else:
        print("🔄 Running full database migration...")
        success = migrator.run_full_migration()
        
        if success:
            print("\n🎯 Next Steps:")
            print("1. ✅ Run your integration tests")
            print("2. ✅ Start the Indian Trading Bot")
            print("3. ✅ Test automation bot integration")
        else:
            print("\n❌ Migration failed!")
            print("💡 Check the error messages above and try again")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())