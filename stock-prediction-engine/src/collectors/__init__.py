"""
News collectors package for Phase 3 News Intelligence Engine.

This package provides news collection functionality from various sources
including APIs, web scraping, and social media platforms.
"""

from base_collector import (
    BaseCollector, 
    CollectionConfig, 
    CollectionResult,
    CollectorRegistry,
    get_collector_registry,
    register_collector,
    run_all_collections
)
from newsapi_collector import NewsAPICollector

__version__ = "1.0.0"
__all__ = [
    # Base collector framework
    "BaseCollector",
    "CollectionConfig", 
    "CollectionResult",
    "CollectorRegistry",
    "get_collector_registry",
    "register_collector",
    "run_all_collections",
    
    # Specific collectors
    "NewsAPICollector"
]

# Auto-register available collectors
def initialize_collectors():
    """Initialize and register available collectors."""
    import os
    
    collectors_registered = []
    
    # Register NewsAPI collector if API key is available
    if os.getenv('NEWSAPI_KEY'):
        try:
            newsapi_collector = NewsAPICollector()
            register_collector(newsapi_collector)
            collectors_registered.append('NewsAPI')
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to initialize NewsAPI collector: {e}")
    
    return collectors_registered

def get_available_collectors():
    """Get list of available collector types."""
    return [
        {
            'name': 'NewsAPI',
            'class': 'NewsAPICollector',
            'description': 'Professional news aggregation API service',
            'requirements': ['NEWSAPI_KEY environment variable'],
            'reliability': 'High',
            'cost': 'Free tier available'
        }
        # More collectors will be added here
    ]

def setup_default_collection_config():
    """Set up default collection configuration."""
    return CollectionConfig(
        max_articles_per_run=100,
        collection_interval_minutes=15,
        enable_deduplication=True,
        min_article_length=200,
        max_article_age_hours=24,
        retry_attempts=3,
        retry_delay_seconds=5,
        rate_limit_delay=1.0,
        timeout_seconds=30
    )

# Package-level convenience functions
def quick_collect_news(sources=None, max_articles=50):
    """
    Quick news collection from available sources.
    
    Args:
        sources: List of source names to collect from (None for all)
        max_articles: Maximum articles per source
        
    Returns:
        Dict: Collection results by source
    """
    registry = get_collector_registry()
    
    # Initialize collectors if not already done
    if not registry.list_collectors():
        initialize_collectors()
    
    results = {}
    collectors_to_run = sources or registry.list_collectors()
    
    for source_name in collectors_to_run:
        collector = registry.get_collector(source_name)
        if collector:
            # Temporarily adjust max articles
            original_max = collector.config.max_articles_per_run
            collector.config.max_articles_per_run = max_articles
            
            try:
                results[source_name] = collector.run_collection()
            finally:
                # Restore original setting
                collector.config.max_articles_per_run = original_max
        else:
            results[source_name] = {
                'error': f'Collector {source_name} not available'
            }
    
    return results

def health_check_all_collectors():
    """
    Perform health check on all registered collectors.
    
    Returns:
        Dict: Health check results by source
    """
    registry = get_collector_registry()
    return registry.test_all_connections()

# Initialize logging for the package
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Auto-initialize collectors when package is imported
try:
    registered = initialize_collectors()
    if registered:
        logger = logging.getLogger(__name__)
        logger.info(f"Initialized collectors: {', '.join(registered)}")
except Exception as e:
    # Silently handle initialization errors - they'll be logged by individual collectors
    pass