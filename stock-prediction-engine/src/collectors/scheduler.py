"""
Collection scheduler for Phase 3 News Intelligence Engine.

This module provides automated scheduling for news collection, processing,
and alert generation. Supports continuous monitoring and real-time updates.
"""

import time
import logging
import threading
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..database import get_session
from ..collectors import get_collector_registry, run_all_collections
from ..processors import process_article, get_market_sentiment_overview
from ..processors.event_extractor import FinancialEventExtractor
from ..monitoring.alert_system import NewsIntelligenceAlertSystem

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ScheduleConfig:
    """Configuration for scheduled operations."""
    collection_interval_minutes: int = 15
    processing_interval_minutes: int = 5
    alert_check_interval_minutes: int = 2
    cleanup_interval_hours: int = 24
    max_article_age_days: int = 30
    enable_parallel_processing: bool = True
    max_workers: int = 4
    
@dataclass
class SchedulerStats:
    """Statistics for scheduler operations."""
    started_at: datetime = field(default_factory=datetime.now)
    total_collections: int = 0
    total_articles_collected: int = 0
    total_articles_processed: int = 0
    total_alerts_generated: int = 0
    last_collection: Optional[datetime] = None
    last_processing: Optional[datetime] = None
    last_alert_check: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    uptime_seconds: float = 0.0

class NewsIntelligenceScheduler:
    """Automated scheduler for news intelligence operations."""
    
    def __init__(self, config: ScheduleConfig = None, alert_system: NewsIntelligenceAlertSystem = None):
        self.config = config or ScheduleConfig()
        self.alert_system = alert_system or NewsIntelligenceAlertSystem()
        self.session = get_session()
        self.event_extractor = FinancialEventExtractor()
        
        # Scheduler state
        self.running = False
        self.scheduler_thread = None
        self.stats = SchedulerStats()
        
        # Thread pool for parallel processing
        if self.config.enable_parallel_processing:
            self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        else:
            self.executor = None
        
        # Set up scheduled jobs
        self._setup_schedule()
    
    def _setup_schedule(self):
        """Set up scheduled jobs."""
        
        # News collection
        schedule.every(self.config.collection_interval_minutes).minutes.do(
            self._scheduled_collection
        )
        
        # Article processing
        schedule.every(self.config.processing_interval_minutes).minutes.do(
            self._scheduled_processing
        )
        
        # Alert checking
        schedule.every(self.config.alert_check_interval_minutes).minutes.do(
            self._scheduled_alert_check
        )
        
        # Database cleanup
        schedule.every(self.config.cleanup_interval_hours).hours.do(
            self._scheduled_cleanup
        )
        
        # Stats update
        schedule.every(1).minutes.do(self._update_stats)
        
        logger.info("Scheduled jobs configured:")
        logger.info(f"  Collection: every {self.config.collection_interval_minutes} minutes")
        logger.info(f"  Processing: every {self.config.processing_interval_minutes} minutes") 
        logger.info(f"  Alerts: every {self.config.alert_check_interval_minutes} minutes")
        logger.info(f"  Cleanup: every {self.config.cleanup_interval_hours} hours")
    
    def start(self):
        """Start the scheduler."""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self.stats.started_at = datetime.now()
        
        logger.info("🚀 Starting News Intelligence Scheduler...")
        
        # Start scheduler in separate thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("✅ Scheduler started successfully")
    
    def stop(self):
        """Stop the scheduler."""
        if not self.running:
            logger.warning("Scheduler is not running")
            return
        
        self.running = False
        logger.info("🛑 Stopping News Intelligence Scheduler...")
        
        # Wait for scheduler thread to finish
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=10)
        
        # Shutdown thread pool
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("✅ Scheduler stopped successfully")
    
    def _run_scheduler(self):
        """Main scheduler loop."""
        logger.info("📅 Scheduler loop started")
        
        # Run initial collection and processing
        try:
            logger.info("🔄 Running initial collection and processing...")
            self._scheduled_collection()
            time.sleep(5)  # Brief delay
            self._scheduled_processing()
            self._scheduled_alert_check()
        except Exception as e:
            logger.error(f"Initial setup failed: {e}")
        
        # Main scheduling loop
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                self.stats.errors.append(f"{datetime.now()}: {str(e)}")
                time.sleep(60)  # Wait longer on error
        
        logger.info("📅 Scheduler loop ended")
    
    def _scheduled_collection(self):
        """Scheduled news collection."""
        logger.info("📰 Starting scheduled news collection...")
        
        try:
            start_time = time.time()
            
            # Run collection from all registered collectors
            results = run_all_collections()
            
            # Update statistics
            total_collected = sum(r.articles_processed for r in results.values() if hasattr(r, 'articles_processed'))
            
            self.stats.total_collections += 1
            self.stats.total_articles_collected += total_collected
            self.stats.last_collection = datetime.now()
            
            collection_time = time.time() - start_time
            
            logger.info(f"✅ Collection completed: {total_collected} articles in {collection_time:.2f}s")
            
            # Log individual collector results
            for source_name, result in results.items():
                if hasattr(result, 'success') and result.success:
                    logger.info(f"  {source_name}: {getattr(result, 'articles_processed', 0)} articles")
                else:
                    logger.warning(f"  {source_name}: Collection failed")
            
        except Exception as e:
            logger.error(f"Scheduled collection failed: {e}")
            self.stats.errors.append(f"Collection error: {str(e)}")
    
    def _scheduled_processing(self):
        """Scheduled article processing."""
        logger.info("🧠 Starting scheduled article processing...")
        
        try:
            start_time = time.time()
            
            # Get unprocessed articles
            unprocessed_articles = self.session.db.execute_query(
                "SELECT * FROM news_articles WHERE processed = ? ORDER BY collected_at DESC LIMIT ?",
                (False, 50)  # Process up to 50 articles at a time
            )
            
            if not unprocessed_articles:
                logger.info("📝 No articles to process")
                return
            
            articles = [self.session._row_to_article(row) for row in unprocessed_articles]
            logger.info(f"📝 Processing {len(articles)} articles...")
            
            processed_count = 0
            
            if self.config.enable_parallel_processing and self.executor:
                # Parallel processing
                futures = []
                for article in articles:
                    future = self.executor.submit(self._process_single_article, article)
                    futures.append(future)
                
                # Wait for completion
                for future in as_completed(futures):
                    try:
                        if future.result():
                            processed_count += 1
                    except Exception as e:
                        logger.error(f"Parallel processing error: {e}")
            
            else:
                # Sequential processing
                for article in articles:
                    try:
                        if self._process_single_article(article):
                            processed_count += 1
                    except Exception as e:
                        logger.error(f"Processing error for article {article.id}: {e}")
            
            # Update statistics
            self.stats.total_articles_processed += processed_count
            self.stats.last_processing = datetime.now()
            
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Processing completed: {processed_count}/{len(articles)} articles in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Scheduled processing failed: {e}")
            self.stats.errors.append(f"Processing error: {str(e)}")
    
    def _process_single_article(self, article) -> bool:
        """Process a single article."""
        try:
            # Run text processing and sentiment analysis
            result = process_article(article.title, article.content)
            
            # Extract events
            event_result = self.event_extractor.extract_events(article.content, article.title)
            
            # Determine primary event type
            primary_event = None
            if event_result.events:
                primary_event = max(event_result.events, key=lambda x: x.confidence)
            
            # Update article in database
            self.session.update_article(
                article.id,
                sentiment_score=result['sentiment_analysis']['sentiment_score'],
                sentiment_label=result['sentiment_analysis']['sentiment_label'],
                keywords=result['text_analysis']['keywords'],
                stock_symbols=result['text_analysis']['stock_symbols'],
                event_type=primary_event.event_type.value if primary_event else None,
                impact_score=primary_event.confidence if primary_event else None,
                processed=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process article {article.id}: {e}")
            return False
    
    def _scheduled_alert_check(self):
        """Scheduled alert checking."""
        logger.info("🚨 Starting scheduled alert check...")
        
        try:
            start_time = time.time()
            
            # Get recently processed articles
            since = datetime.now() - timedelta(minutes=self.config.alert_check_interval_minutes * 2)
            recent_articles = self.session.db.execute_query(
                "SELECT * FROM news_articles WHERE processed = ? AND updated_at > ? ORDER BY updated_at DESC",
                (True, since)
            )
            
            if not recent_articles:
                logger.info("📝 No recent articles for alert checking")
                return
            
            articles = [self.session._row_to_article(row) for row in recent_articles]
            logger.info(f"🔍 Checking {len(articles)} articles for alerts...")
            
            all_alerts = []
            
            for article in articles:
                try:
                    alerts = self.alert_system.process_article(article)
                    all_alerts.extend(alerts)
                except Exception as e:
                    logger.error(f"Alert processing error for article {article.id}: {e}")
            
            if all_alerts:
                logger.info(f"🔔 Generated {len(all_alerts)} alerts")
                
                # Deliver alerts
                self.alert_system.deliver_alerts(all_alerts)
                
                # Update statistics
                self.stats.total_alerts_generated += len(all_alerts)
                
                # Log alert summary
                priority_counts = {}
                for alert in all_alerts:
                    priority = alert.priority.value
                    priority_counts[priority] = priority_counts.get(priority, 0) + 1
                
                logger.info(f"📊 Alert priorities: {priority_counts}")
            
            else:
                logger.info("📝 No alerts generated")
            
            self.stats.last_alert_check = datetime.now()
            
            check_time = time.time() - start_time
            logger.info(f"✅ Alert check completed in {check_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Scheduled alert check failed: {e}")
            self.stats.errors.append(f"Alert check error: {str(e)}")
    
    def _scheduled_cleanup(self):
        """Scheduled database cleanup."""
        logger.info("🧹 Starting scheduled database cleanup...")
        
        try:
            start_time = time.time()
            
            # Delete old articles
            cutoff_date = datetime.now() - timedelta(days=self.config.max_article_age_days)
            
            deleted_articles = self.session.db.execute_update(
                "DELETE FROM news_articles WHERE published_at < ?",
                (cutoff_date,)
            )
            
            # Delete old collection metrics
            deleted_metrics = self.session.db.execute_update(
                "DELETE FROM collection_metrics WHERE created_at < ?",
                (cutoff_date,)
            )
            
            # Delete old alerts
            alert_cutoff = datetime.now() - timedelta(days=7)  # Keep alerts for 7 days
            deleted_alerts = self.session.db.execute_update(
                "DELETE FROM alerts WHERE created_at < ?",
                (alert_cutoff,)
            )
            
            cleanup_time = time.time() - start_time
            
            logger.info(f"✅ Cleanup completed in {cleanup_time:.2f}s:")
            logger.info(f"  Deleted {deleted_articles} old articles")
            logger.info(f"  Deleted {deleted_metrics} old metrics")
            logger.info(f"  Deleted {deleted_alerts} old alerts")
            
        except Exception as e:
            logger.error(f"Scheduled cleanup failed: {e}")
            self.stats.errors.append(f"Cleanup error: {str(e)}")
    
    def _update_stats(self):
        """Update scheduler statistics."""
        if self.stats.started_at:
            self.stats.uptime_seconds = (datetime.now() - self.stats.started_at).total_seconds()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            'running': self.running,
            'uptime_hours': self.stats.uptime_seconds / 3600,
            'total_collections': self.stats.total_collections,
            'total_articles_collected': self.stats.total_articles_collected,
            'total_articles_processed': self.stats.total_articles_processed,
            'total_alerts_generated': self.stats.total_alerts_generated,
            'last_collection': self.stats.last_collection.isoformat() if self.stats.last_collection else None,
            'last_processing': self.stats.last_processing.isoformat() if self.stats.last_processing else None,
            'last_alert_check': self.stats.last_alert_check.isoformat() if self.stats.last_alert_check else None,
            'error_count': len(self.stats.errors),
            'recent_errors': self.stats.errors[-5:] if self.stats.errors else []
        }
    
    def force_collection(self):
        """Force immediate collection."""
        logger.info("🔄 Forcing immediate collection...")
        self._scheduled_collection()
    
    def force_processing(self):
        """Force immediate processing."""
        logger.info("🔄 Forcing immediate processing...")
        self._scheduled_processing()
    
    def force_alert_check(self):
        """Force immediate alert check."""
        logger.info("🔄 Forcing immediate alert check...")
        self._scheduled_alert_check()
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get current market overview from recent articles."""
        try:
            # Get articles from last 24 hours
            since = datetime.now() - timedelta(hours=24)
            recent_articles = self.session.db.execute_query(
                "SELECT * FROM news_articles WHERE processed = ? AND published_at > ? ORDER BY published_at DESC LIMIT 100",
                (True, since)
            )
            
            if not recent_articles:
                return {'error': 'No recent articles available'}
            
            articles = [self.session._row_to_article(row) for row in recent_articles]
            article_data = [(a.title, a.content) for a in articles]
            
            # Get market sentiment overview
            sentiment_overview = get_market_sentiment_overview(article_data)
            
            # Add additional metrics
            sentiment_overview['articles_analyzed'] = len(articles)
            sentiment_overview['time_period'] = '24 hours'
            sentiment_overview['last_updated'] = datetime.now().isoformat()
            
            # Get top stocks mentioned
            stock_mentions = self.session.get_top_mentioned_stocks(limit=10, since=since)
            sentiment_overview['top_stocks'] = stock_mentions
            
            # Get event distribution
            event_articles = self.session.db.execute_query(
                "SELECT event_type, COUNT(*) as count FROM news_articles WHERE event_type IS NOT NULL AND published_at > ? GROUP BY event_type",
                (since,)
            )
            
            event_distribution = {row['event_type']: row['count'] for row in event_articles}
            sentiment_overview['event_distribution'] = event_distribution
            
            return sentiment_overview
            
        except Exception as e:
            logger.error(f"Failed to get market overview: {e}")
            return {'error': str(e)}

def create_default_scheduler(email_config=None) -> NewsIntelligenceScheduler:
    """Create scheduler with default configuration."""
    
    # Create alert system
    alert_system = NewsIntelligenceAlertSystem(email_config)
    
    # Create scheduler config
    config = ScheduleConfig(
        collection_interval_minutes=15,  # Collect every 15 minutes
        processing_interval_minutes=5,   # Process every 5 minutes
        alert_check_interval_minutes=2,  # Check alerts every 2 minutes
        cleanup_interval_hours=24,       # Cleanup daily
        max_article_age_days=30,         # Keep articles for 30 days
        enable_parallel_processing=True,
        max_workers=4
    )
    
    return NewsIntelligenceScheduler(config, alert_system)

def start_monitoring_service(email_config=None):
    """Start the complete monitoring service."""
    
    print("🚀 Starting News Intelligence Monitoring Service...")
    
    # Create and start scheduler
    scheduler = create_default_scheduler(email_config)
    scheduler.start()
    
    try:
        print("✅ Service started successfully!")
        print("📊 Monitoring dashboard:")
        print(f"  Collection: Every {scheduler.config.collection_interval_minutes} minutes")
        print(f"  Processing: Every {scheduler.config.processing_interval_minutes} minutes")
        print(f"  Alerts: Every {scheduler.config.alert_check_interval_minutes} minutes")
        print("\nPress Ctrl+C to stop the service...")
        
        # Keep running until interrupted
        while scheduler.running:
            time.sleep(10)
            
            # Print periodic status updates
            status = scheduler.get_status()
            if status['total_collections'] > 0:
                print(f"📈 Status: {status['total_collections']} collections, "
                      f"{status['total_articles_collected']} articles, "
                      f"{status['total_alerts_generated']} alerts")
                
                # Show market overview every hour
                if int(status['uptime_hours']) > 0 and int(status['uptime_hours']) % 1 == 0:
                    overview = scheduler.get_market_overview()
                    if 'error' not in overview:
                        print(f"💰 Market Sentiment: {overview.get('overall_sentiment', 0):.3f} "
                              f"(Risk: {overview.get('risk_level', 'unknown')})")
            
            time.sleep(50)  # Wait 1 minute total
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping service...")
        scheduler.stop()
        print("✅ Service stopped successfully!")

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Test scheduler
    print("=== News Intelligence Scheduler Test ===\n")
    
    # Create test scheduler with shorter intervals
    test_config = ScheduleConfig(
        collection_interval_minutes=2,  # Every 2 minutes for testing
        processing_interval_minutes=1,  # Every 1 minute for testing
        alert_check_interval_minutes=1, # Every 1 minute for testing
        cleanup_interval_hours=1,       # Every hour for testing
        enable_parallel_processing=True,
        max_workers=2
    )
    
    scheduler = NewsIntelligenceScheduler(test_config)
    
    print("📅 Starting test scheduler...")
    scheduler.start()
    
    try:
        # Run for 5 minutes for testing
        print("⏱️ Running for 5 minutes (test mode)...")
        time.sleep(300)  # 5 minutes
        
        # Show final status
        status = scheduler.get_status()
        print(f"\n📊 Test Results:")
        print(f"  Collections: {status['total_collections']}")
        print(f"  Articles Collected: {status['total_articles_collected']}")
        print(f"  Articles Processed: {status['total_articles_processed']}")
        print(f"  Alerts Generated: {status['total_alerts_generated']}")
        print(f"  Uptime: {status['uptime_hours']:.2f} hours")
        print(f"  Errors: {status['error_count']}")
        
        # Get market overview
        overview = scheduler.get_market_overview()
        if 'error' not in overview:
            print(f"\n💰 Market Overview:")
            print(f"  Sentiment: {overview.get('overall_sentiment', 0):.3f}")
            print(f"  Risk Level: {overview.get('risk_level', 'unknown')}")
            print(f"  Articles: {overview.get('articles_analyzed', 0)}")
    
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    
    finally:
        scheduler.stop()
        print("✅ Test scheduler stopped")
    
    print("\nScheduler test completed!")
    
    # Option to start full monitoring service
    if len(sys.argv) > 1 and sys.argv[1] == '--monitor':
        start_monitoring_service()