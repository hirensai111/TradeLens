#!/usr/bin/env python3
"""
Quick Fix for Phase 3 Connector Issues
This script will:
1. Create the missing config.yaml file from your existing sample_config.json
2. Check and fix your database schema
3. Test the connection
"""

import json
import yaml
import sqlite3
import os
from pathlib import Path
from datetime import datetime

def create_config_yaml():
    """Convert sample_config.json to config.yaml"""
    print("🔧 Creating config.yaml from sample_config.json...")
    
    # Read the existing JSON config
    json_config_path = Path("D:/stock-prediction-engine/config/sample_config.json")
    yaml_config_path = Path("D:/stock-prediction-engine/config/config.yaml")
    
    try:
        with open(json_config_path, 'r') as f:
            json_config = json.load(f)
        
        # Create a simplified YAML config that your connector expects
        yaml_config = {
            'data_sources': {
                'phase3_db_path': str(Path("D:/stock-prediction-engine") / json_config['sqlite_db_path']),
                'stock_data_source': 'alpha_vantage',
                'news_sources': ['newsapi', 'polygon', 'reddit']
            },
            'alpha_vantage': {
                'api_key': json_config['alpha_vantage_key'],
                'base_url': json_config['alpha_vantage_base_url'],
                'rate_limit': json_config['alpha_vantage_rate_limit']
            },
            'newsapi': {
                'api_key': json_config['newsapi_key'],
                'base_url': json_config['newsapi_base_url'],
                'rate_limit': json_config['newsapi_rate_limit']
            },
            'polygon': {
                'api_key': json_config['polygon_key'],
                'base_url': json_config['polygon_base_url']
            },
            'database': {
                'type': json_config['database_type'],
                'path': str(Path("D:/stock-prediction-engine") / json_config['sqlite_db_path']),
                'url': json_config['database_url']
            },
            'logging': {
                'level': json_config['log_level'],
                'log_dir': str(Path("D:/stock-prediction-engine") / json_config['logs_dir'])
            },
            'sentiment': {
                'openai_key': json_config['openai_api_key'],
                'model': json_config['openai_model'],
                'confidence_threshold': json_config['sentiment_confidence_threshold']
            }
        }
        
        # Write YAML config
        with open(yaml_config_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Created {yaml_config_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating config.yaml: {e}")
        return False

def check_and_fix_database():
    """Check and fix the database schema"""
    db_path = "D:/stock-prediction-engine/data/news_intelligence.db"
    print(f"\n🔍 Checking database: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"❌ Database doesn't exist. Creating it...")
        return create_test_database(db_path)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check existing tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = [table[0] for table in cursor.fetchall()]
        
        print(f"📊 Found tables: {existing_tables}")
        
        # Required tables
        required_tables = {
            'news_articles': """
                CREATE TABLE IF NOT EXISTS news_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT,
                    url TEXT UNIQUE,
                    source TEXT,
                    published_at DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT
                );
            """,
            'sentiment_analysis': """
                CREATE TABLE IF NOT EXISTS sentiment_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id INTEGER,
                    sentiment_overall REAL,
                    sentiment_score REAL,
                    sentiment_confidence REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (article_id) REFERENCES news_articles (id)
                );
            """,
            'stock_mentions': """
                CREATE TABLE IF NOT EXISTS stock_mentions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id INTEGER,
                    ticker TEXT,
                    company_name TEXT,
                    mention_count INTEGER DEFAULT 1,
                    sentiment REAL,
                    FOREIGN KEY (article_id) REFERENCES news_articles (id)
                );
            """,
            'alerts': """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    alert_type TEXT,
                    message TEXT,
                    severity TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """
        }
        
        # Create missing tables
        created_tables = []
        for table_name, create_sql in required_tables.items():
            if table_name not in existing_tables:
                cursor.execute(create_sql)
                created_tables.append(table_name)
                print(f"✅ Created table: {table_name}")
        
        # Check data in existing tables
        for table in existing_tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table};")
                count = cursor.fetchone()[0]
                print(f"   - {table}: {count} rows")
            except Exception as e:
                print(f"   - {table}: Error - {e}")
        
        # Add sample data if tables are empty
        if created_tables or not existing_tables:
            add_sample_data(cursor)
        
        conn.commit()
        conn.close()
        
        print(f"✅ Database schema validated and fixed")
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def add_sample_data(cursor):
    """Add sample data for testing"""
    print("📝 Adding sample data...")
    
    try:
        # Check if news_articles is empty
        cursor.execute("SELECT COUNT(*) FROM news_articles;")
        if cursor.fetchone()[0] == 0:
            
            # Sample articles
            sample_articles = [
                ("Microsoft Reports Strong Q3 Earnings Beat", 
                 "Microsoft Corporation reported quarterly earnings that exceeded analyst expectations with strong cloud growth...", 
                 "https://example.com/msft-earnings-q3", 
                 "NewsAPI", 
                 "2025-07-15 09:00:00",
                 "earnings"),
                ("Apple Unveils New AI Features in iOS", 
                 "Apple Inc. announced new artificial intelligence capabilities coming to iOS devices...", 
                 "https://example.com/apple-ai-ios", 
                 "Polygon", 
                 "2025-07-15 10:30:00",
                 "product_launch"),
                ("Tesla Production Milestone Reached", 
                 "Tesla Inc. announced it has reached a significant production milestone for the Model Y...", 
                 "https://example.com/tesla-production-milestone", 
                 "Reddit", 
                 "2025-07-15 11:15:00",
                 "production"),
                ("NVIDIA Announces Next-Gen GPU Architecture", 
                 "NVIDIA Corporation unveiled its next-generation GPU architecture targeting AI workloads...", 
                 "https://example.com/nvidia-gpu-announcement", 
                 "NewsAPI", 
                 "2025-07-15 12:00:00",
                 "product_launch"),
                ("Amazon Web Services Expands Cloud Offerings", 
                 "Amazon.com Inc. announced new cloud computing services through AWS division...", 
                 "https://example.com/aws-expansion", 
                 "Polygon", 
                 "2025-07-15 13:30:00",
                 "business_expansion")
            ]
            
            cursor.executemany("""
            INSERT INTO news_articles (title, content, url, source, published_at, event_type)
            VALUES (?, ?, ?, ?, ?, ?)
            """, sample_articles)
            
            # Sample sentiment analysis
            sample_sentiment = [
                (1, 0.75, 0.75, 0.85),  # MSFT - positive earnings
                (2, 0.65, 0.65, 0.80),  # AAPL - positive innovation
                (3, 0.80, 0.80, 0.90),  # TSLA - very positive production
                (4, 0.70, 0.70, 0.85),  # NVDA - positive tech announcement
                (5, 0.60, 0.60, 0.75)   # AMZN - moderately positive
            ]
            
            cursor.executemany("""
            INSERT INTO sentiment_analysis (article_id, sentiment_overall, sentiment_score, sentiment_confidence)
            VALUES (?, ?, ?, ?)
            """, sample_sentiment)
            
            # Sample stock mentions
            sample_mentions = [
                (1, "MSFT", "Microsoft Corporation", 5, 0.75),
                (2, "AAPL", "Apple Inc.", 3, 0.65),
                (3, "TSLA", "Tesla Inc.", 2, 0.80),
                (4, "NVDA", "NVIDIA Corporation", 4, 0.70),
                (5, "AMZN", "Amazon.com Inc.", 3, 0.60)
            ]
            
            cursor.executemany("""
            INSERT INTO stock_mentions (article_id, ticker, company_name, mention_count, sentiment)
            VALUES (?, ?, ?, ?, ?)
            """, sample_mentions)
            
            # Sample alerts
            sample_alerts = [
                ("MSFT", "positive_sentiment", "Strong positive sentiment detected for earnings", "medium"),
                ("TSLA", "high_volume", "High news volume detected", "high"),
                ("NVDA", "positive_sentiment", "Positive sentiment on product announcement", "low")
            ]
            
            cursor.executemany("""
            INSERT INTO alerts (ticker, alert_type, message, severity)
            VALUES (?, ?, ?, ?)
            """, sample_alerts)
            
            print("✅ Added sample data (5 articles, sentiment analysis, stock mentions, alerts)")
    
    except Exception as e:
        print(f"❌ Error adding sample data: {e}")

def test_phase3_connector():
    """Test the Phase 3 connector after fixes"""
    print(f"\n🧪 Testing Phase 3 connector...")
    
    try:
        # Import and test the connector
        import sys
        sys.path.append("D:/stock-prediction-engine/phase4_ml_engine/src/data_loaders")
        
        from phase3_connector import EnhancedPhase3Connector
        
        # Create connector instance
        connector = EnhancedPhase3Connector()
        
        # Test validation
        validation = connector.validate_enhanced_database_schema()
        print(f"📋 Schema validation: {'✅ PASSED' if validation['schema_valid'] else '❌ FAILED'}")
        
        if validation['schema_valid']:
            # Test enhanced features
            from datetime import datetime
            test_date = datetime.now()
            
            print(f"🔍 Testing enhanced sentiment features for MSFT...")
            features = connector.get_enhanced_sentiment_features("MSFT", test_date)
            
            print(f"   Sentiment (1d): {features.sentiment_1d:.3f}")
            print(f"   News Volume (1d): {features.news_volume_1d}")
            print(f"   Source Diversity: {features.source_diversity}")
            print(f"   Confidence: {features.confidence_score:.3f}")
            
            print(f"✅ Phase 3 connector working properly!")
            return True
        else:
            print(f"❌ Schema validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Connector test failed: {e}")
        return False

def main():
    """Main fix function"""
    print("🚀 Phase 3 Connector Quick Fix")
    print("=" * 40)
    
    # Step 1: Create config.yaml
    config_ok = create_config_yaml()
    
    # Step 2: Fix database
    db_ok = check_and_fix_database()
    
    # Step 3: Test connector
    if config_ok and db_ok:
        test_ok = test_phase3_connector()
        
        print(f"\n📊 Final Status:")
        print(f"   Config file: {'✅ OK' if config_ok else '❌ Failed'}")
        print(f"   Database: {'✅ OK' if db_ok else '❌ Failed'}")
        print(f"   Connector test: {'✅ OK' if test_ok else '❌ Failed'}")
        
        if config_ok and db_ok and test_ok:
            print(f"\n🎉 SUCCESS! Your Phase 3 connector should now work!")
            print(f"💡 Try running: python src/data_loaders/phase3_connector.py")
        else:
            print(f"\n⚠️ Some issues remain. Check the errors above.")
    else:
        print(f"\n❌ Basic setup failed. Check config and database issues.")

if __name__ == "__main__":
    main()