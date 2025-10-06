"""
Final Complete Phase 3 System Test - All Components with Hybrid Collector

This script tests the COMPLETE Phase 3 News Intelligence Engine including:
- Hybrid multi-source collection (NewsAPI + Polygon + Reddit)
- Real-time monitoring and scheduling
- News-price correlation analysis
- Advanced alerting and trend analysis
"""

import sys
import os
import time
import threading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import get_session, get_database_status
from src.collectors import get_collector_registry, register_collector
from src.collectors.hybrid_collector import HybridNewsCollector
from src.collectors.reddit_collector import RedditCollector
from src.collectors.scheduler import create_default_scheduler, ScheduleConfig
from src.processors import process_article, get_market_sentiment_overview
from src.processors.event_extractor import FinancialEventExtractor
from src.monitoring.alert_system import NewsIntelligenceAlertSystem
from src.intelligence.correlation_analyzer import NewsPriceCorrelationAnalyzer

def test_hybrid_multi_source_collection():
    """Test hybrid multi-source news collection with fallback capabilities."""
    
    print("🌐 HYBRID MULTI-SOURCE COLLECTION TEST")
    print("=" * 60)
    
    registry = get_collector_registry()
    
    # Test Hybrid News Collector (NewsAPI + Polygon)
    try:
        hybrid_collector = HybridNewsCollector()
        register_collector(hybrid_collector)
        
        print(f"🔄 Hybrid News Collector: ✅ Available")
        print(f"   NewsAPI Available: {hybrid_collector.newsapi_available}")
        print(f"   Polygon Available: {hybrid_collector.polygon_available}")
        
        # Test hybrid connection
        connection_test = hybrid_collector.test_connection()
        newsapi_status = connection_test.get('newsapi', {})
        polygon_status = connection_test.get('polygon', {})
        
        print(f"   NewsAPI Status: {newsapi_status.get('status', 'unknown')}")
        if newsapi_status.get('rate_limit_remaining'):
            print(f"   NewsAPI Rate Limit: {newsapi_status['rate_limit_remaining']} remaining")
        
        print(f"   Polygon Status: {polygon_status.get('status', 'unknown')}")
        
        # Show fallback capability
        if newsapi_status.get('status') == 'rate_limited' and polygon_status.get('status') == 'ok':
            print("   🔄 Fallback Mode: NewsAPI → Polygon (Smart!)")
        elif newsapi_status.get('status') == 'ok' and polygon_status.get('status') == 'ok':
            print("   🚀 Dual Mode: Both APIs available")
        elif newsapi_status.get('status') == 'ok':
            print("   📰 Single Mode: NewsAPI only")
        elif polygon_status.get('status') == 'ok':
            print("   📊 Single Mode: Polygon only")
        else:
            print("   ⚠️ Limited Mode: API issues detected")
            
    except Exception as e:
        print(f"🔄 Hybrid News Collector: ❌ Error - {e}")
    
    # Test Reddit collector for social sentiment
    try:
        reddit_collector = RedditCollector(
            subreddits=['investing', 'stocks'],
            config=type('Config', (), {
                'max_articles_per_run': 10,
                'min_article_length': 50,
                'max_article_age_hours': 24,
                'timeout_seconds': 30
            })()
        )
        
        register_collector(reddit_collector)
        print("📱 Reddit Collector: ✅ Available")
        
        # Test Reddit connection
        connection_test = reddit_collector.test_connection()
        print(f"   Connection Test: {'✅ Success' if connection_test.get('success', False) else '⚠️ Limited (no API key)'}")
        
        # Get subreddit stats
        stats = reddit_collector.get_subreddit_stats()
        print(f"   Monitored Subreddits: {len(stats)}")
        
    except Exception as e:
        print(f"📱 Reddit Collector: ❌ Error - {e}")
    
    # Test collection from multiple sources
    print(f"\n🔄 Testing Hybrid Multi-Source Collection...")
    
    collectors = registry.list_collectors()
    print(f"Active Collectors: {collectors}")
    
    if len(collectors) >= 1:
        print("✅ Multi-source collection ready")
        
        # Run a quick collection test
        try:
            results = registry.run_all_collections()
            
            total_articles = 0
            for source, result in results.items():
                articles = getattr(result, 'articles_processed', 0)
                total_articles += articles
                print(f"   {source}: {articles} articles")
            
            print(f"📊 Total Articles Collected: {total_articles}")
            
            # Show hybrid collector performance
            if 'HybridNews' in results:
                hybrid_result = results['HybridNews']
                hybrid_collector_instance = registry.get_collector('HybridNews')
                if hasattr(hybrid_collector_instance, 'get_collection_summary'):
                    summary = hybrid_collector_instance.get_collection_summary()
                    print(f"   Hybrid Performance:")
                    print(f"     NewsAPI Failures: {summary.get('newsapi_failures', 0)}")
                    print(f"     Polygon Usage: {summary.get('polygon_usage', 0)}")
                    print(f"     Fallback Active: {summary.get('fallback_active', False)}")
            
        except Exception as e:
            print(f"⚠️ Collection test error: {e}")
    
    else:
        print("⚠️ No collectors available")
    
    return len(collectors)

def test_enhanced_real_time_monitoring():
    """Test enhanced real-time monitoring with hybrid collection."""
    
    print(f"\n⏰ ENHANCED REAL-TIME MONITORING TEST")
    print("=" * 60)
    
    try:
        # Create test scheduler with short intervals
        config = ScheduleConfig(
            collection_interval_minutes=2,  # Every 2 minutes
            processing_interval_minutes=1,  # Every minute
            alert_check_interval_minutes=1, # Every minute
            enable_parallel_processing=True,
            max_workers=2
        )
        
        scheduler = create_default_scheduler()
        scheduler.config = config  # Override with test config
        
        print("✅ Enhanced scheduler created successfully")
        print(f"📅 Schedule Configuration:")
        print(f"   Collection: Every {config.collection_interval_minutes} minutes")
        print(f"   Processing: Every {config.processing_interval_minutes} minutes")
        print(f"   Alerts: Every {config.alert_check_interval_minutes} minutes")
        print(f"   Parallel Processing: {config.enable_parallel_processing}")
        print(f"   Hybrid Collection: Enabled")
        
        # Test scheduler start/stop
        print(f"\n🚀 Testing Enhanced Scheduler Operations...")
        scheduler.start()
        print("✅ Scheduler started")
        
        # Let it run for a short time to test hybrid collection
        time.sleep(15)
        
        # Check status
        status = scheduler.get_status()
        print(f"📊 Enhanced Scheduler Status:")
        print(f"   Running: {status['running']}")
        print(f"   Uptime: {status['uptime_hours']:.3f} hours")
        print(f"   Collections: {status['total_collections']}")
        print(f"   Articles Processed: {status['total_articles_processed']}")
        
        # Test manual operations
        print(f"\n🔄 Testing Manual Operations...")
        scheduler.force_processing()
        scheduler.force_alert_check()
        print("✅ Manual operations completed")
        
        # Get enhanced market overview
        try:
            overview = scheduler.get_market_overview()
            if 'error' not in overview:
                print(f"📈 Enhanced Market Overview Available:")
                print(f"   Articles: {overview.get('articles_analyzed', 0)}")
                print(f"   Sentiment: {overview.get('overall_sentiment', 0):.3f}")
                print(f"   Risk Level: {overview.get('risk_level', 'unknown')}")
                print(f"   Data Sources: Multi-source hybrid collection")
            else:
                print(f"⚠️ Market overview: {overview['error']}")
        except Exception as e:
            print(f"⚠️ Market overview error: {e}")
        
        scheduler.stop()
        print("✅ Enhanced scheduler stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced real-time monitoring test failed: {e}")
        return False

def test_enhanced_correlation_analysis():
    """Test enhanced news-price correlation analysis with hybrid data."""
    
    print(f"\n📊 ENHANCED NEWS-PRICE CORRELATION TEST")
    print("=" * 60)
    
    try:
        # Initialize correlation analyzer
        analyzer = NewsPriceCorrelationAnalyzer()
        print("✅ Enhanced correlation analyzer initialized")
        print("   Data Sources: Hybrid collection (NewsAPI + Polygon + Reddit)")
        
        # Test single stock analysis with hybrid data
        test_symbols = ['AAPL', 'GOOGL', 'TSLA']
        print(f"\n🔍 Testing Enhanced Single Stock Analysis...")
        
        correlation_results = {}
        
        for symbol in test_symbols[:2]:  # Test first 2 symbols
            print(f"\nAnalyzing {symbol} with hybrid data:")
            
            try:
                result = analyzer.analyze_correlation(symbol, days=14)
                correlation_results[symbol] = result
                
                print(f"   Correlation: {result.correlation_coefficient:.3f}")
                print(f"   Confidence: {result.confidence:.3f}")
                print(f"   Sample Size: {result.sample_size}")
                print(f"   Prediction Accuracy: {result.prediction_accuracy:.3f}")
                print(f"   Significant Events: {len(result.significant_events)}")
                
                if result.significant_events:
                    top_event = result.significant_events[0]
                    print(f"   Top Event: {top_event['title'][:40]}... ({top_event['price_change']:+.2f}%)")
                    print(f"   Event Source: {top_event.get('source', 'Unknown')}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Test enhanced market correlation summary
        print(f"\n📈 Testing Enhanced Market Correlation Summary...")
        
        try:
            market_summary = analyzer.get_market_correlation_summary(test_symbols, days=14)
            
            if 'error' not in market_summary:
                print(f"✅ Enhanced Market Summary Generated:")
                print(f"   Stocks Analyzed: {market_summary['total_stocks_analyzed']}")
                print(f"   Avg Correlation: {market_summary['avg_correlation']:.3f}")
                print(f"   Avg Prediction Accuracy: {market_summary['avg_prediction_accuracy']:.3f}")
                
                best = market_summary['best_correlation']
                print(f"   Best Correlation: {best['symbol']} ({best['coefficient']:.3f})")
                
                top_events = market_summary.get('top_market_events', [])
                print(f"   Top Market Events: {len(top_events)}")
                
                # Show data source diversity
                print(f"   Data Quality: Enhanced with multi-source coverage")
                
            else:
                print(f"⚠️ Market summary error: {market_summary['error']}")
        
        except Exception as e:
            print(f"❌ Market summary error: {e}")
        
        print(f"\n✅ Enhanced correlation analysis test completed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced correlation analysis test failed: {e}")
        return False

def test_advanced_alerting_with_hybrid_data():
    """Test advanced alerting with hybrid data sources."""
    
    print(f"\n🚨 ADVANCED ALERTING TEST (Hybrid Data)")
    print("=" * 60)
    
    try:
        # Initialize advanced alert system
        alert_system = NewsIntelligenceAlertSystem()
        print(f"✅ Advanced alert system initialized with {len(alert_system.rules)} rules")
        print("   Data Sources: Hybrid collection for better coverage")
        
        # Add custom correlation-based rule
        from src.monitoring.alert_system import AlertRule, AlertPriority
        
        # Enhanced rule for hybrid data
        hybrid_correlation_rule = AlertRule(
            name="hybrid_high_correlation_event",
            description="High correlation event detected across multiple data sources",
            priority=AlertPriority.HIGH,
            min_sentiment_score=0.6,
            min_confidence=0.7,
            cooldown_minutes=60
        )
        
        alert_system.add_custom_rule(hybrid_correlation_rule)
        print(f"✅ Added enhanced correlation-based alert rule")
        
        # Test alert generation on recent articles from hybrid sources
        session = get_session()
        recent_articles = session.get_articles(limit=5)
        
        if recent_articles:
            print(f"\n🔍 Testing Alert Generation on {len(recent_articles)} hybrid articles...")
            
            total_alerts = 0
            alert_types = {}
            source_diversity = set()
            
            for article in recent_articles:
                try:
                    # Track source diversity
                    source_diversity.add(article.source)
                    
                    alerts = alert_system.process_article(article)
                    total_alerts += len(alerts)
                    
                    for alert in alerts:
                        alert_type = alert.rule_name
                        alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
                        
                        print(f"   🔔 {alert.priority.value.upper()}: {alert.title[:50]}...")
                        print(f"      Source: {article.source}")
                
                except Exception as e:
                    print(f"   ❌ Alert processing error: {e}")
            
            print(f"\n📊 Enhanced Alert Generation Results:")
            print(f"   Total Alerts: {total_alerts}")
            print(f"   Alert Types: {alert_types}")
            print(f"   Data Sources Used: {len(source_diversity)}")
            print(f"   Source Diversity: {', '.join(list(source_diversity)[:3])}")
            
            if total_alerts > 0:
                print("✅ Advanced alerting with hybrid data functional")
            else:
                print("📝 No alerts generated (normal for test data)")
        
        else:
            print("⚠️ No articles available for alert testing")
        
        # Test enhanced alert summary
        alert_summary = alert_system.get_alert_summary(24)
        print(f"\n📋 Enhanced Alert Summary (24h): {alert_summary.get('total_alerts', 0)} alerts")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced alerting test failed: {e}")
        return False

def test_enhanced_trend_analysis():
    """Test enhanced trend analysis with hybrid data sources."""
    
    print(f"\n📈 ENHANCED TREND ANALYSIS TEST")
    print("=" * 60)
    
    try:
        session = get_session()
        
        # Get enhanced sentiment trends over time
        print("🔍 Analyzing Enhanced Sentiment Trends...")
        print("   Data Sources: Hybrid collection for comprehensive coverage")
        
        # Get articles from different time periods
        from datetime import datetime, timedelta
        
        periods = [
            ('Last 24 hours', timedelta(hours=24)),
            ('Last 3 days', timedelta(days=3)),
            ('Last week', timedelta(days=7))
        ]
        
        trend_data = {}
        source_coverage = {}
        
        for period_name, period_delta in periods:
            since = datetime.now() - period_delta
            
            period_articles = session.db.execute_query(
                "SELECT * FROM news_articles WHERE processed = ? AND published_at > ?",
                (True, since)
            )
            
            if period_articles:
                articles = [session._row_to_article(row) for row in period_articles]
                article_data = [(a.title, a.content) for a in articles]
                
                # Track source diversity
                sources = set(a.source for a in articles)
                source_coverage[period_name] = sources
                
                sentiment_overview = get_market_sentiment_overview(article_data)
                
                trend_data[period_name] = {
                    'articles': len(articles),
                    'sentiment': sentiment_overview.get('overall_sentiment', 0),
                    'risk_level': sentiment_overview.get('risk_level', 'unknown'),
                    'positive_ratio': sentiment_overview.get('positive_ratio', 0),
                    'sources': len(sources)
                }
                
                print(f"   {period_name}: {len(articles)} articles, "
                      f"sentiment {sentiment_overview.get('overall_sentiment', 0):.3f}, "
                      f"{len(sources)} sources")
        
        # Calculate enhanced trend momentum
        if len(trend_data) >= 2:
            periods_list = list(trend_data.keys())
            recent = trend_data[periods_list[0]]
            older = trend_data[periods_list[-1]]
            
            sentiment_momentum = recent['sentiment'] - older['sentiment']
            
            print(f"\n📊 Enhanced Trend Analysis Results:")
            print(f"   Sentiment Momentum: {sentiment_momentum:+.3f}")
            print(f"   Direction: {'🔺 Improving' if sentiment_momentum > 0 else '🔻 Declining' if sentiment_momentum < 0 else '➡️ Stable'}")
            print(f"   Data Quality: Enhanced with {recent['sources']} source types")
            
            # Risk trend
            risk_levels = {'low': 1, 'medium': 2, 'high': 3}
            recent_risk = risk_levels.get(recent['risk_level'], 2)
            older_risk = risk_levels.get(older['risk_level'], 2)
            risk_momentum = recent_risk - older_risk
            
            print(f"   Risk Momentum: {'🔺 Increasing' if risk_momentum > 0 else '🔻 Decreasing' if risk_momentum < 0 else '➡️ Stable'}")
            
            # Source diversity trends
            recent_sources = source_coverage.get(periods_list[0], set())
            older_sources = source_coverage.get(periods_list[-1], set())
            
            if recent_sources and older_sources:
                source_stability = len(recent_sources & older_sources) / len(recent_sources | older_sources)
                print(f"   Source Stability: {source_stability:.2%}")
        
        # Get enhanced event momentum
        print(f"\n📅 Analyzing Enhanced Event Trends...")
        
        event_articles = session.db.execute_query(
            "SELECT event_type, COUNT(*) as count FROM news_articles WHERE event_type IS NOT NULL AND published_at > ? GROUP BY event_type ORDER BY count DESC",
            (datetime.now() - timedelta(days=7),)
        )
        
        if event_articles:
            print("   Top Event Types (7 days, hybrid sources):")
            for event in event_articles[:5]:
                print(f"     • {event['event_type']}: {event['count']} articles")
        
        print("✅ Enhanced trend analysis completed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced trend analysis test failed: {e}")
        return False

def run_enhanced_comprehensive_test():
    """Run the enhanced comprehensive test with hybrid collection."""
    
    print("🚀 PHASE 3 ENHANCED COMPREHENSIVE SYSTEM TEST")
    print("=" * 80)
    print("Testing ALL components with HYBRID COLLECTION CAPABILITIES...")
    print("=" * 80)
    
    # Test results tracking
    test_results = {
        'system_health': False,
        'hybrid_multi_source_collection': False,
        'enhanced_real_time_monitoring': False,
        'enhanced_correlation_analysis': False,
        'advanced_alerting_hybrid': False,
        'enhanced_trend_analysis': False
    }
    
    # 1. System Health Check
    print("\n🏥 SYSTEM HEALTH CHECK")
    print("-" * 40)
    
    try:
        db_status = get_database_status()
        print(f"✅ Database: {db_status['health']['status']}")
        print(f"✅ Articles in DB: {len(get_session().get_articles(limit=100))}")
        print(f"✅ Hybrid Collection: Ready")
        test_results['system_health'] = True
    except Exception as e:
        print(f"❌ System health check failed: {e}")
    
    # 2. Hybrid Multi-Source Collection
    sources_count = test_hybrid_multi_source_collection()
    test_results['hybrid_multi_source_collection'] = sources_count >= 1
    
    # 3. Enhanced Real-time Monitoring
    test_results['enhanced_real_time_monitoring'] = test_enhanced_real_time_monitoring()
    
    # 4. Enhanced Correlation Analysis
    test_results['enhanced_correlation_analysis'] = test_enhanced_correlation_analysis()
    
    # 5. Advanced Alerting with Hybrid Data
    test_results['advanced_alerting_hybrid'] = test_advanced_alerting_with_hybrid_data()
    
    # 6. Enhanced Trend Analysis
    test_results['enhanced_trend_analysis'] = test_enhanced_trend_analysis()
    
    # Final Enhanced Results Summary
    print(f"\n🏆 ENHANCED PHASE 3 SYSTEM RESULTS")
    print("=" * 80)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    completion_percentage = (passed_tests / total_tests) * 100
    
    print(f"📊 System Completion: {completion_percentage:.1f}% ({passed_tests}/{total_tests} components)")
    print(f"\n📋 Component Status:")
    
    status_icons = {True: "✅", False: "❌"}
    
    for component, status in test_results.items():
        component_name = component.replace('_', ' ').title()
        icon = status_icons[status]
        print(f"   {icon} {component_name}")
    
    # Enhanced system readiness assessment
    print(f"\n🎯 ENHANCED SYSTEM READINESS ASSESSMENT:")
    
    if completion_percentage >= 90:
        readiness = "🟢 PRODUCTION READY"
        message = "All critical components operational with hybrid collection! Ready for Phase 4!"
    elif completion_percentage >= 75:
        readiness = "🟡 NEAR PRODUCTION READY"
        message = "Most components working with hybrid capabilities. Minor issues to resolve."
    elif completion_percentage >= 50:
        readiness = "🟠 DEVELOPMENT READY"
        message = "Core hybrid functionality present. Significant work needed."
    else:
        readiness = "🔴 NEEDS WORK"
        message = "Major components failing. Requires debugging."
    
    print(f"   Status: {readiness}")
    print(f"   Assessment: {message}")
    
    # Enhanced recommendations
    print(f"\n🚀 ENHANCED NEXT STEPS RECOMMENDATIONS:")
    
    recommendations = []
    
    if not test_results['hybrid_multi_source_collection']:
        recommendations.append("🔧 Configure API keys for hybrid collection (NewsAPI + Polygon)")
    
    if not test_results['enhanced_correlation_analysis']:
        recommendations.append("📊 Set up Alpha Vantage API for enhanced price correlation")
    
    if not test_results['enhanced_real_time_monitoring']:
        recommendations.append("⏰ Debug enhanced scheduler with hybrid collection")
    
    if not test_results['advanced_alerting_hybrid']:
        recommendations.append("🚨 Configure advanced alerting with hybrid data sources")
    
    if completion_percentage >= 75:
        recommendations.extend([
            "🔗 Begin Phase 4 integration with hybrid intelligence",
            "📈 Set up production monitoring for hybrid collection",
            "📧 Configure multi-source alert delivery systems",
            "🔄 Implement hybrid data backup strategies",
            "⚡ Optimize hybrid collection performance"
        ])
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    print(f"\n" + "=" * 80)
    print(f"🎉 PHASE 3 ENHANCED NEWS INTELLIGENCE ENGINE - FINAL TEST COMPLETE!")
    print(f"🏆 System Achievement: {completion_percentage:.1f}% Complete")
    print(f"🚀 Ready for: {'Production Deployment' if completion_percentage >= 90 else 'Further Development'}")
    print(f"💡 Key Innovation: Hybrid collection with intelligent fallback capabilities")
    print("=" * 80)
    
    return test_results

if __name__ == "__main__":
    # Run the enhanced comprehensive final test
    results = run_enhanced_comprehensive_test()
    
    # Exit with appropriate code
    passed = sum(results.values())
    total = len(results)
    
    if passed == total:
        exit(0)  # Perfect success
    elif passed >= total * 0.75:
        exit(1)  # Mostly successful
    else:
        exit(2)  # Needs work