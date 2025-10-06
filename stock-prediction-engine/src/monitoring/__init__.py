"""
Monitoring package for Phase 3 News Intelligence Engine.

This package provides real-time monitoring, alert generation, and system health tracking
for the news intelligence system.
"""

from .alert_system import (
    NewsIntelligenceAlertSystem,
    AlertRule,
    AlertPriority,
    AlertChannel,
    GeneratedAlert
)

__version__ = "1.0.0"
__all__ = [
    "NewsIntelligenceAlertSystem",
    "AlertRule", 
    "AlertPriority",
    "AlertChannel",
    "GeneratedAlert"
]

# Package-level convenience functions
def create_default_alert_system(email_config=None):
    """Create alert system with default configuration."""
    return NewsIntelligenceAlertSystem(email_config)

def monitor_market_news(check_interval_minutes=5, email_config=None):
    """
    Start monitoring market news with default settings.
    
    Args:
        check_interval_minutes: How often to check for new articles
        email_config: Email configuration for alerts
    """
    alert_system = NewsIntelligenceAlertSystem(email_config)
    alert_system.monitor_new_articles(check_interval_minutes)

def get_system_health():
    """Get overall system health status."""
    from ..database import get_database
    from ..collectors import get_collector_registry
    
    db = get_database()
    registry = get_collector_registry()
    
    # Database health
    db_health = db.health_check()
    
    # Collector health
    collector_health = registry.test_all_connections()
    
    # Alert system health
    alert_system = NewsIntelligenceAlertSystem()
    alert_summary = alert_system.get_alert_summary(24)
    
    return {
        'database': db_health,
        'collectors': collector_health,
        'alerts_24h': alert_summary,
        'timestamp': db_health.get('timestamp'),
        'overall_status': 'healthy' if db_health.get('status') == 'healthy' else 'unhealthy'
    }

# Initialize logging for the package
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())