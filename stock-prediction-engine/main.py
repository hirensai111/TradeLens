"""
News Intelligence Engine - Main Interface
Interactive keyword search with hybrid collection capabilities

This script provides a user-friendly interface to search for news articles
using keywords through the Phase 3 News Intelligence Engine.
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import get_session, get_database_status
from src.collectors import get_collector_registry, register_collector
from src.collectors.hybrid_collector import HybridNewsCollector
from src.collectors.reddit_collector import RedditCollector
from src.processors import process_article, get_market_sentiment_overview
from src.processors.event_extractor import FinancialEventExtractor
from src.intelligence.correlation_analyzer import NewsPriceCorrelationAnalyzer

class NewsSearchEngine:
    """Main news search engine interface."""
    
    def __init__(self):
        self.session = get_session()
        self.registry = get_collector_registry()
        self.setup_collectors()
        self.correlation_analyzer = NewsPriceCorrelationAnalyzer()
    
    def setup_collectors(self):
        """Initialize and register all collectors."""
        print("🔧 Setting up News Intelligence Engine...")
        
        # Register Hybrid News Collector
        try:
            hybrid_collector = HybridNewsCollector()
            register_collector(hybrid_collector)
            print("✅ Hybrid News Collector (NewsAPI + Polygon) registered")
        except Exception as e:
            print(f"⚠️ Hybrid collector setup warning: {e}")
        
        # Register Reddit Collector
        try:
            reddit_config = type('Config', (), {
                'max_articles_per_run': 20,
                'min_article_length': 100,
                'max_article_age_hours': 48,
                'timeout_seconds': 30,
                'rate_limit_delay': 1.0,  # Add missing rate_limit_delay
                'retry_attempts': 3,
                'retry_delay_seconds': 5,
                'enable_deduplication': True,
                'collection_interval_minutes': 15
            })()
            
            reddit_collector = RedditCollector(
                subreddits=['investing', 'stocks', 'SecurityAnalysis', 'ValueInvesting'],
                config=reddit_config
            )
            register_collector(reddit_collector)
            print("✅ Reddit Collector registered")
        except Exception as e:
            print(f"⚠️ Reddit collector setup warning: {e}")
        
        collectors = self.registry.list_collectors()
        print(f"📊 Total collectors available: {len(collectors)}")
        
        # Show collector health
        if collectors:
            print("🔍 Testing collector connections...")
            for collector_name in collectors:
                try:
                    collector = self.registry.get_collector(collector_name)
                    if hasattr(collector, 'test_connection'):
                        test_result = collector.test_connection()
                        if test_result.get('success', False):
                            print(f"   ✅ {collector_name}: Connection OK")
                        else:
                            print(f"   ⚠️ {collector_name}: Connection issues")
                except Exception as e:
                    print(f"   ❌ {collector_name}: Test failed")
        
        # Fix NewsAPI parsing issue
        newsapi_collector = self.registry.get_collector('NewsAPI')
        if newsapi_collector:
            print("🔧 Applying NewsAPI parsing fix...")
            self._patch_newsapi_parsing(newsapi_collector)
    
    def _patch_newsapi_parsing(self, collector):
        """Patch NewsAPI collector to handle None values properly."""
        original_parse = collector._parse_article
        
        def safe_parse_article(article_data, **kwargs):
            """Safe parsing that handles None values."""
            try:
                # Pre-process article_data to handle None values
                safe_data = {}
                for key, value in article_data.items():
                    if value is None:
                        safe_data[key] = ''
                    else:
                        safe_data[key] = value
                
                return original_parse(safe_data, **kwargs)
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(f"Article parsing failed (skipping): {e}")
                return None
        
        collector._parse_article = safe_parse_article
        print("   ✅ NewsAPI parsing fix applied")
    
    def search_existing_articles(self, keyword: str, limit: int = 10, days_back: int = 7) -> List[Dict]:
        """Search existing articles in the database."""
        print(f"🔍 Searching existing articles for '{keyword}'...")
        
        since = datetime.now() - timedelta(days=days_back)
        
        # Search in title, content, and stock_symbols
        articles = self.session.db.execute_query(
            """
            SELECT * FROM news_articles 
            WHERE (title LIKE ? OR content LIKE ? OR stock_symbols LIKE ?) 
            AND published_at > ? 
            AND processed = 1
            ORDER BY published_at DESC 
            LIMIT ?
            """,
            (f'%{keyword}%', f'%{keyword}%', f'%{keyword}%', since, limit)
        )
        
        if articles:
            print(f"📰 Found {len(articles)} existing articles")
            return [self.session._row_to_article(row) for row in articles]
        else:
            print("📭 No existing articles found")
            return []
    
    def collect_fresh_articles(self, keyword: str, limit: int = 10) -> List[Dict]:
        """Collect fresh articles from external sources."""
        print(f"🌐 Collecting fresh articles for '{keyword}'...")
        
        fresh_articles = []
        
        # Get available collectors
        collectors = self.registry.list_collectors()
        
        if not collectors:
            print("⚠️ No collectors available for fresh collection")
            return fresh_articles
        
        # Modify collectors to search for specific keyword
        print(f"🔧 Configuring collectors to search for '{keyword}'...")
        
        # Update NewsAPI collector to search for keyword
        newsapi_collector = self.registry.get_collector('NewsAPI')
        if newsapi_collector:
            # Add keyword to search queries
            original_queries = newsapi_collector.search_queries.copy()
            newsapi_collector.search_queries = [keyword] + original_queries[:3]  # Prioritize keyword
            print(f"   📰 NewsAPI: Added '{keyword}' to search queries")
        
        # Update Hybrid collector
        hybrid_collector = self.registry.get_collector('HybridNews')
        if hybrid_collector:
            # Set focused search query
            hybrid_collector._search_keyword = keyword
            print(f"   🔀 HybridNews: Set focused search for '{keyword}'")
        
        # Temporarily disable deduplication to ensure fresh collection
        original_dedup_settings = {}
        for collector_name in collectors:
            collector = self.registry.get_collector(collector_name)
            if collector and hasattr(collector, 'config'):
                original_dedup_settings[collector_name] = collector.config.enable_deduplication
                collector.config.enable_deduplication = False
        
        print(f"   🔄 Temporarily disabled deduplication for fresh collection")
        
        # Clear recent articles from database to allow fresh collection
        try:
            # Remove articles from the last 30 minutes to allow fresh collection
            deleted_count = self.session.db.execute_update(
                "DELETE FROM news_articles WHERE collected_at > datetime('now', '-30 minutes')",
                ()
            )
            print(f"   🗑️ Cleared {deleted_count} recent articles to allow fresh collection")
        except Exception as e:
            print(f"   ⚠️ Could not clear recent articles: {e}")
            
        # Alternative approach: Use INSERT OR IGNORE for fresh collection
        print(f"   🔄 Using INSERT OR IGNORE for fresh collection handling")
        
        # Run collection from all sources
        print(f"📡 Running keyword-focused collection from {len(collectors)} sources...")
        collection_results = self.registry.run_all_collections()
        
        total_collected = 0
        total_processed = 0
        
        for collector_name, result in collection_results.items():
            if result.success:
                total_collected += result.articles_collected
                total_processed += result.articles_processed
                print(f"   ✅ {collector_name}: {result.articles_processed} articles processed")
                if result.errors:
                    print(f"      ⚠️ With {len(result.errors)} parsing warnings")
            else:
                print(f"   ❌ {collector_name}: Failed - {result.errors[:1]}")  # Show first error only
        
        # Restore original search queries and deduplication settings
        if newsapi_collector:
            newsapi_collector.search_queries = original_queries
            print(f"   📰 NewsAPI: Restored original search queries")
        
        if hybrid_collector:
            hybrid_collector._search_keyword = None
            print(f"   🔀 HybridNews: Cleared focused search")
        
        # Restore deduplication settings
        for collector_name, original_setting in original_dedup_settings.items():
            collector = self.registry.get_collector(collector_name)
            if collector and hasattr(collector, 'config'):
                collector.config.enable_deduplication = original_setting
        
        print(f"   🔄 Restored deduplication settings")
        
        print(f"📊 Total fresh collection: {total_processed} articles from {len(collectors)} sources")
        
        # Now search for keyword-relevant articles from the fresh collection
        if total_collected > 0:  # Change from total_processed to total_collected
            print(f"🔍 Searching for '{keyword}' in {total_collected} collected articles...")
            
            # First, search in articles from the last 2 hours (broader time range)
            recent_articles = self.session.db.execute_query(
                """
                SELECT * FROM news_articles 
                WHERE collected_at > datetime('now', '-2 hours')
                AND (title LIKE ? OR content LIKE ? OR stock_symbols LIKE ?)
                ORDER BY collected_at DESC
                LIMIT ?
                """,
                (f'%{keyword}%', f'%{keyword}%', f'%{keyword}%', limit * 2)
            )
            
            if recent_articles:
                # Convert to NewsArticle objects and filter for relevance
                relevant_articles = []
                for row in recent_articles:
                    article = self.session._row_to_article(row)
                    
                    # Check relevance score
                    title_matches = keyword.lower() in article.title.lower()
                    content_matches = keyword.lower() in article.content.lower()
                    stock_matches = False
                    
                    # Check if it's a stock symbol search
                    if hasattr(article, 'stock_symbols') and article.stock_symbols:
                        stock_matches = keyword.upper() in str(article.stock_symbols).upper()
                    
                    if title_matches or content_matches or stock_matches:
                        relevant_articles.append(article)
                
                fresh_articles.extend(relevant_articles[:limit])
                print(f"   ✅ Found {len(relevant_articles)} relevant articles")
            else:
                print(f"   📭 No articles matching '{keyword}' found in recent collection")
                
                # Search in all articles if no recent matches
                print(f"   🔍 Searching in all articles for '{keyword}'...")
                all_articles = self.session.db.execute_query(
                    """
                    SELECT * FROM news_articles 
                    WHERE (title LIKE ? OR content LIKE ? OR stock_symbols LIKE ?)
                    ORDER BY published_at DESC
                    LIMIT ?
                    """,
                    (f'%{keyword}%', f'%{keyword}%', f'%{keyword}%', limit)
                )
                
                if all_articles:
                    relevant_articles = []
                    for row in all_articles:
                        article = self.session._row_to_article(row)
                        
                        # Check relevance score
                        title_matches = keyword.lower() in article.title.lower()
                        content_matches = keyword.lower() in article.content.lower()
                        stock_matches = False
                        
                        if hasattr(article, 'stock_symbols') and article.stock_symbols:
                            stock_matches = keyword.upper() in str(article.stock_symbols).upper()
                        
                        if title_matches or content_matches or stock_matches:
                            relevant_articles.append(article)
                    
                    fresh_articles.extend(relevant_articles[:limit])
                    print(f"   ✅ Found {len(relevant_articles)} relevant articles in database")
                else:
                    print(f"   📭 No articles matching '{keyword}' found in database")
                    
                    # Show what we have in the database for debugging
                    sample_articles = self.session.db.execute_query(
                        """
                        SELECT title, source, published_at FROM news_articles 
                        ORDER BY published_at DESC
                        LIMIT 10
                        """
                    )
                    
                    if sample_articles:
                        print(f"   📋 Sample articles in database (showing 10):")
                        for row in sample_articles:
                            title_snippet = row['title'][:50] + "..." if len(row['title']) > 50 else row['title']
                            print(f"      • {title_snippet} ({row['source']}, {row['published_at']})")
                    else:
                        print(f"   📭 No articles found in database at all")
        else:
            print(f"   📭 No articles were collected")
        
        return fresh_articles
    
    def analyze_sentiment(self, articles: List) -> Dict:
        """Analyze sentiment of found articles."""
        if not articles:
            return {'error': 'No articles to analyze'}
        
        print(f"📊 Analyzing sentiment for {len(articles)} articles...")
        
        # Prepare article data
        article_data = []
        for article in articles:
            title = getattr(article, 'title', '') or ''
            content = getattr(article, 'content', '') or ''
            article_data.append((title, content))
        
        # Get sentiment overview
        sentiment_overview = get_market_sentiment_overview(article_data)
        
        return sentiment_overview
    
    def get_correlation_analysis(self, keyword: str, symbol: str = None) -> Dict:
        """Get correlation analysis if keyword relates to a stock symbol."""
        if not symbol:
            # Try to extract symbol from keyword
            common_symbols = {
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
            
            symbol = common_symbols.get(keyword.lower())
        
        if not symbol:
            return {'error': 'No stock symbol identified'}
        
        print(f"📈 Analyzing correlation for {symbol}...")
        
        try:
            correlation_result = self.correlation_analyzer.analyze_correlation(symbol, days=14)
            
            return {
                'symbol': symbol,
                'correlation': correlation_result.correlation_coefficient,
                'confidence': correlation_result.confidence,
                'prediction_accuracy': correlation_result.prediction_accuracy,
                'sample_size': correlation_result.sample_size,
                'significant_events': correlation_result.significant_events[:3]  # Top 3
            }
        except Exception as e:
            return {'error': f'Correlation analysis failed: {e}'}
    
    def display_articles(self, articles: List, max_display: int = 5):
        """Display article information in a formatted way."""
        if not articles:
            print("📭 No articles to display")
            return
        
        print(f"\n📰 ARTICLES FOUND ({len(articles)} total, showing top {min(max_display, len(articles))})")
        print("=" * 80)
        
        for i, article in enumerate(articles[:max_display], 1):
            title = getattr(article, 'title', 'No title') or 'No title'
            source = getattr(article, 'source', 'Unknown') or 'Unknown'
            published_at = getattr(article, 'published_at', 'Unknown') or 'Unknown'
            sentiment = getattr(article, 'sentiment', 'Unknown') or 'Unknown'
            
            print(f"\n{i}. {title}")
            print(f"   📍 Source: {source}")
            print(f"   📅 Published: {published_at}")
            print(f"   😊 Sentiment: {sentiment}")
            
            # Show snippet of content
            content = getattr(article, 'content', '') or ''
            if content:
                snippet = content[:200] + "..." if len(content) > 200 else content
                print(f"   📝 Content: {snippet}")
            
            print("-" * 40)
    
    def search(self, keyword: str, fresh: bool = False, limit: int = 10, 
               days_back: int = 7, symbol: str = None):
        """Main search function."""
        print(f"\n🚀 NEWS INTELLIGENCE ENGINE - SEARCH")
        print("=" * 60)
        print(f"🔍 Keyword: '{keyword}'")
        print(f"📅 Search scope: {'Fresh collection' if fresh else f'Last {days_back} days'}")
        print(f"📊 Limit: {limit} articles")
        print("=" * 60)
        
        all_articles = []
        
        # Search existing articles first (unless fresh-only mode)
        if not fresh:
            existing_articles = self.search_existing_articles(keyword, limit, days_back)
            all_articles.extend(existing_articles)
        
        # Auto-collect fresh articles if no existing articles found or if requested
        should_collect_fresh = fresh or len(all_articles) < 3
        
        if should_collect_fresh:
            if len(all_articles) < 3 and not fresh:
                print(f"💡 Found only {len(all_articles)} existing articles. Collecting fresh ones...")
            
            fresh_articles = self.collect_fresh_articles(keyword, limit)
            all_articles.extend(fresh_articles)
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title = getattr(article, 'title', '') or ''
            if title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
        
        # Display results
        self.display_articles(unique_articles, limit)
        
        # Analyze sentiment
        if unique_articles:
            sentiment_analysis = self.analyze_sentiment(unique_articles)
            
            print(f"\n📊 SENTIMENT ANALYSIS")
            print("=" * 40)
            if 'error' not in sentiment_analysis:
                print(f"Overall Sentiment: {sentiment_analysis.get('overall_sentiment', 0):.3f}")
                print(f"Risk Level: {sentiment_analysis.get('risk_level', 'unknown')}")
                print(f"Positive Ratio: {sentiment_analysis.get('positive_ratio', 0):.1%}")
                print(f"Articles Analyzed: {len(unique_articles)}")
            else:
                print(f"Error: {sentiment_analysis['error']}")
            
            # Correlation analysis
            correlation_analysis = self.get_correlation_analysis(keyword, symbol)
            
            print(f"\n📈 CORRELATION ANALYSIS")
            print("=" * 40)
            if 'error' not in correlation_analysis:
                print(f"Stock Symbol: {correlation_analysis['symbol']}")
                print(f"News-Price Correlation: {correlation_analysis['correlation']:.3f}")
                print(f"Confidence: {correlation_analysis['confidence']:.3f}")
                print(f"Prediction Accuracy: {correlation_analysis['prediction_accuracy']:.3f}")
                
                if correlation_analysis['significant_events']:
                    print(f"Top Events:")
                    for event in correlation_analysis['significant_events']:
                        print(f"  • {event['title'][:50]}... ({event['price_change']:+.2f}%)")
            else:
                print(f"Note: {correlation_analysis['error']}")
        
        return unique_articles

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="News Intelligence Engine - Keyword Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Tesla earnings"
  python main.py "Apple iPhone" --fresh --limit 15
  python main.py "NVDA" --symbol NVDA --days 14
  python main.py "Federal Reserve" --fresh --limit 20
  python main.py --interactive
        """
    )
    
    parser.add_argument(
        'keyword',
        nargs='?',  # Make keyword optional
        help='Search keyword or phrase (e.g., "Tesla", "Apple earnings", "Federal Reserve")'
    )
    
    parser.add_argument(
        '--fresh', '-f',
        action='store_true',
        help='Collect fresh articles from external sources'
    )
    
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=10,
        help='Maximum number of articles to display (default: 10)'
    )
    
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=7,
        help='Number of days back to search existing articles (default: 7)'
    )
    
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        help='Stock symbol for correlation analysis (e.g., AAPL, GOOGL)'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start interactive search mode'
    )
    
    args = parser.parse_args()
    
    # If no keyword and not interactive, show help and start interactive mode
    if not args.keyword and not args.interactive:
        print("\n🚀 NEWS INTELLIGENCE ENGINE")
        print("=" * 50)
        print("Welcome! You can search for news articles using keywords.")
        print("\nQuick start options:")
        print("1. python main.py \"Tesla\" - Search for Tesla articles")
        print("2. python main.py \"Apple earnings\" --fresh - Get fresh Apple earnings news")
        print("3. python main.py --interactive - Start interactive mode")
        print("\nStarting interactive mode...")
        args.interactive = True
    
    # Check system health
    try:
        db_status = get_database_status()
        if db_status['health']['status'] != 'healthy':
            print("⚠️ Database health check failed. Please run the test script first.")
            return
    except Exception as e:
        print(f"❌ System health check failed: {e}")
        return
    
    # Initialize search engine
    try:
        engine = NewsSearchEngine()
    except Exception as e:
        print(f"❌ Failed to initialize search engine: {e}")
        return
    
    # Interactive mode
    if args.interactive:
        print("\n🎯 INTERACTIVE NEWS SEARCH MODE")
        print("Type 'quit' to exit, 'help' for commands")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n🔍 Enter keyword to search: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("""
Available commands:
  <keyword>        - Search for articles (auto-collects fresh if few results)
  fresh <keyword>  - Force collect fresh articles
  stock <symbol>   - Search with stock symbol (e.g., stock AMD)
  quit/exit/q      - Exit interactive mode
  help             - Show this help
  
Examples:
  AMD              - Search for AMD articles
  fresh Tesla      - Get fresh Tesla articles
  stock NVDA       - Search NVDA with correlation analysis
                    """)
                    continue
                
                if not user_input:
                    continue
                
                # Check for different modes
                fresh_mode = user_input.startswith('fresh ')
                stock_mode = user_input.startswith('stock ')
                
                if fresh_mode:
                    keyword = user_input[6:].strip()
                    symbol = None
                elif stock_mode:
                    keyword = user_input[6:].strip()
                    symbol = keyword.upper()  # Use keyword as symbol
                else:
                    keyword = user_input
                    symbol = args.symbol
                
                # Run search
                engine.search(
                    keyword=keyword,
                    fresh=fresh_mode,
                    limit=args.limit,
                    days_back=args.days,
                    symbol=symbol
                )
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Search error: {e}")
    
    # Single search mode
    elif args.keyword:  # Only run single search if keyword is provided
        try:
            engine.search(
                keyword=args.keyword,
                fresh=args.fresh,
                limit=args.limit,
                days_back=args.days,
                symbol=args.symbol
            )
        except Exception as e:
            print(f"❌ Search failed: {e}")
    
    else:
        # This shouldn't happen with the logic above, but just in case
        parser.print_help()

if __name__ == "__main__":
    main()