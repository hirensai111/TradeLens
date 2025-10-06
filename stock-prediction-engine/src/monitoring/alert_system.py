"""
Real-time monitoring and alert system for Phase 3 News Intelligence Engine.

This module generates intelligent alerts based on news sentiment, events, and market impact.
Supports multiple notification channels and customizable alert rules.
"""

import logging
import smtplib
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from ..database import get_session, NewsArticle, Alert
from ..processors.sentiment_analyzer import FinancialSentimentAnalyzer
from ..processors.event_extractor import FinancialEventExtractor, EventType, EventSeverity

# Configure logging
logger = logging.getLogger(__name__)

class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertChannel(Enum):
    """Alert delivery channels."""
    DATABASE = "database"
    EMAIL = "email"
    CONSOLE = "console"
    WEBHOOK = "webhook"
    FILE = "file"

@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    description: str
    priority: AlertPriority
    enabled: bool = True
    
    # Conditions
    event_types: List[EventType] = field(default_factory=list)
    event_severities: List[EventSeverity] = field(default_factory=list)
    min_sentiment_score: Optional[float] = None
    max_sentiment_score: Optional[float] = None
    min_confidence: float = 0.0
    min_urgency: float = 0.0
    
    # Stock filtering
    stock_symbols: List[str] = field(default_factory=list)
    exclude_symbols: List[str] = field(default_factory=list)
    
    # Timing
    cooldown_minutes: int = 60  # Prevent spam
    last_triggered: Optional[datetime] = None
    
    # Custom conditions
    custom_condition: Optional[Callable] = None

@dataclass
class GeneratedAlert:
    """A generated alert ready for delivery."""
    rule_name: str
    priority: AlertPriority
    title: str
    message: str
    article_id: int
    stock_symbols: List[str]
    event_types: List[str]
    sentiment_score: float
    confidence: float
    created_at: datetime = field(default_factory=datetime.now)
    channels: List[AlertChannel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class NewsIntelligenceAlertSystem:
    """Comprehensive alert system for news intelligence."""
    
    def __init__(self, email_config: Dict[str, str] = None):
        self.session = get_session()
        self.sentiment_analyzer = FinancialSentimentAnalyzer()
        self.event_extractor = FinancialEventExtractor()
        
        # Email configuration
        self.email_config = email_config or {}
        
        # Alert rules
        self.rules = self._initialize_default_rules()
        
        # Alert history for cooldown tracking
        self.alert_history: Dict[str, datetime] = {}
    
    def _initialize_default_rules(self) -> List[AlertRule]:
        """Initialize default alert rules for financial news."""
        return [
            # Critical Market Events
            AlertRule(
                name="fed_monetary_policy",
                description="Federal Reserve monetary policy decisions",
                priority=AlertPriority.CRITICAL,
                event_types=[EventType.MONETARY_POLICY],
                event_severities=[EventSeverity.CRITICAL, EventSeverity.HIGH],
                min_confidence=0.6,
                cooldown_minutes=30
            ),
            
            AlertRule(
                name="major_mergers",
                description="Major merger and acquisition announcements",
                priority=AlertPriority.HIGH,
                event_types=[EventType.MERGER_ACQUISITION],
                event_severities=[EventSeverity.HIGH, EventSeverity.CRITICAL],
                min_confidence=0.7,
                cooldown_minutes=60
            ),
            
            AlertRule(
                name="earnings_surprises",
                description="Significant earnings beats or misses",
                priority=AlertPriority.HIGH,
                event_types=[EventType.EARNINGS],
                min_sentiment_score=0.6,  # Strong positive
                min_confidence=0.6,
                cooldown_minutes=120
            ),
            
            AlertRule(
                name="earnings_disappointments",
                description="Significant earnings disappointments",
                priority=AlertPriority.HIGH,
                event_types=[EventType.EARNINGS],
                max_sentiment_score=-0.5,  # Strong negative
                min_confidence=0.6,
                cooldown_minutes=120
            ),
            
            AlertRule(
                name="breaking_news",
                description="Breaking financial news",
                priority=AlertPriority.HIGH,
                event_types=[EventType.BREAKING_NEWS],
                min_urgency=0.6,
                min_confidence=0.5,
                cooldown_minutes=15
            ),
            
            AlertRule(
                name="large_stock_movements",
                description="Significant positive sentiment on major stocks",
                priority=AlertPriority.MEDIUM,
                stock_symbols=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META'],
                min_sentiment_score=0.7,
                min_confidence=0.6,
                cooldown_minutes=180
            ),
            
            AlertRule(
                name="regulatory_actions",
                description="Major regulatory actions affecting markets",
                priority=AlertPriority.HIGH,
                event_types=[EventType.REGULATORY],
                event_severities=[EventSeverity.HIGH, EventSeverity.CRITICAL],
                min_confidence=0.6,
                cooldown_minutes=90
            ),
            
            AlertRule(
                name="ipo_announcements",
                description="IPO announcements from significant companies",
                priority=AlertPriority.MEDIUM,
                event_types=[EventType.IPO],
                event_severities=[EventSeverity.MEDIUM, EventSeverity.HIGH],
                min_confidence=0.7,
                cooldown_minutes=240
            ),
            
            AlertRule(
                name="bankruptcy_alerts",
                description="Bankruptcy filings or concerns",
                priority=AlertPriority.CRITICAL,
                event_types=[EventType.BANKRUPTCY],
                min_confidence=0.5,
                cooldown_minutes=60
            ),
            
            AlertRule(
                name="market_sentiment_extreme",
                description="Extreme market sentiment shifts",
                priority=AlertPriority.MEDIUM,
                min_sentiment_score=0.8,  # Very positive
                min_confidence=0.7,
                min_urgency=0.5,
                cooldown_minutes=120
            )
        ]
    
    def add_custom_rule(self, rule: AlertRule):
        """Add a custom alert rule."""
        self.rules.append(rule)
        logger.info(f"Added custom alert rule: {rule.name}")
    
    def process_article(self, article: NewsArticle) -> List[GeneratedAlert]:
        """
        Process an article and generate alerts based on rules.
        
        Args:
            article: News article to analyze
            
        Returns:
            List[GeneratedAlert]: Generated alerts
        """
        if not article.processed:
            logger.warning(f"Article {article.id} not fully processed. Running analysis...")
            # Analyze if not already done
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(article.content, article.title)
            event_result = self.event_extractor.extract_events(article.content, article.title)
        else:
            # Use existing analysis
            sentiment_result = type('SentimentResult', (), {
                'sentiment_score': article.sentiment_score or 0.0,
                'sentiment_label': article.sentiment_label or 'neutral',
                'confidence': 0.5  # Default confidence
            })()
            
            # Mock event result from stored data
            event_result = type('EventResult', (), {
                'events': [],
                'events_found': 0
            })()
            
            if article.event_type:
                mock_event = type('Event', (), {
                    'event_type': type('EventType', (), {'value': article.event_type})(),
                    'severity': type('EventSeverity', (), {'value': 'medium'})(),
                    'confidence': article.impact_score or 0.5
                })()
                event_result.events = [mock_event]
                event_result.events_found = 1
        
        alerts = []
        
        # Check each rule
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            # Check cooldown
            if self._is_rule_in_cooldown(rule):
                continue
            
            # Evaluate rule conditions
            if self._evaluate_rule(rule, article, sentiment_result, event_result):
                alert = self._generate_alert(rule, article, sentiment_result, event_result)
                alerts.append(alert)
                
                # Update cooldown
                self.alert_history[rule.name] = datetime.now()
                
                logger.info(f"Generated {alert.priority.value} alert: {alert.title}")
        
        return alerts
    
    def _is_rule_in_cooldown(self, rule: AlertRule) -> bool:
        """Check if rule is in cooldown period."""
        if rule.name not in self.alert_history:
            return False
        
        last_triggered = self.alert_history[rule.name]
        cooldown_end = last_triggered + timedelta(minutes=rule.cooldown_minutes)
        
        return datetime.now() < cooldown_end
    
    def _evaluate_rule(self, rule: AlertRule, article: NewsArticle, 
                      sentiment_result: Any, event_result: Any) -> bool:
        """Evaluate if a rule's conditions are met."""
        
        # Check event types
        if rule.event_types and event_result.events_found > 0:
            event_types_found = [e.event_type for e in event_result.events]
            if not any(et in rule.event_types for et in event_types_found):
                return False
        elif rule.event_types and event_result.events_found == 0:
            return False
        
        # Check event severities
        if rule.event_severities and event_result.events_found > 0:
            severities_found = [e.severity for e in event_result.events]
            if not any(sev in rule.event_severities for sev in severities_found):
                return False
        
        # Check sentiment conditions
        sentiment_score = sentiment_result.sentiment_score
        
        if rule.min_sentiment_score is not None and sentiment_score < rule.min_sentiment_score:
            return False
        
        if rule.max_sentiment_score is not None and sentiment_score > rule.max_sentiment_score:
            return False
        
        # Check confidence
        confidence = getattr(sentiment_result, 'confidence', 0.5)
        if confidence < rule.min_confidence:
            return False
        
        # Check urgency (if available from text analysis)
        urgency = 0.0  # Would need to get from text processor
        if urgency < rule.min_urgency:
            # For now, skip urgency check if not available
            pass
        
        # Check stock symbols
        if rule.stock_symbols:
            article_symbols = article.stock_symbols or []
            if isinstance(article_symbols, str):
                article_symbols = json.loads(article_symbols)
            
            if not any(symbol in article_symbols for symbol in rule.stock_symbols):
                return False
        
        # Check excluded symbols
        if rule.exclude_symbols:
            article_symbols = article.stock_symbols or []
            if isinstance(article_symbols, str):
                article_symbols = json.loads(article_symbols)
            
            if any(symbol in article_symbols for symbol in rule.exclude_symbols):
                return False
        
        # Check custom condition
        if rule.custom_condition:
            try:
                if not rule.custom_condition(article, sentiment_result, event_result):
                    return False
            except Exception as e:
                logger.error(f"Error evaluating custom condition for rule {rule.name}: {e}")
                return False
        
        return True
    
    def _generate_alert(self, rule: AlertRule, article: NewsArticle,
                       sentiment_result: Any, event_result: Any) -> GeneratedAlert:
        """Generate an alert based on rule and analysis results."""
        
        # Extract stock symbols
        stock_symbols = article.stock_symbols or []
        if isinstance(stock_symbols, str):
            stock_symbols = json.loads(stock_symbols)
        
        # Extract event types
        event_types = []
        if event_result.events_found > 0:
            event_types = [e.event_type.value for e in event_result.events]
        
        # Generate title and message
        title = self._generate_alert_title(rule, article, event_types, stock_symbols)
        message = self._generate_alert_message(rule, article, sentiment_result, event_result)
        
        return GeneratedAlert(
            rule_name=rule.name,
            priority=rule.priority,
            title=title,
            message=message,
            article_id=article.id,
            stock_symbols=stock_symbols,
            event_types=event_types,
            sentiment_score=sentiment_result.sentiment_score,
            confidence=getattr(sentiment_result, 'confidence', 0.5),
            channels=[AlertChannel.DATABASE, AlertChannel.CONSOLE],  # Default channels
            metadata={
                'source': article.source,
                'published_at': article.published_at.isoformat() if article.published_at else None,
                'url': article.url
            }
        )
    
    def _generate_alert_title(self, rule: AlertRule, article: NewsArticle,
                             event_types: List[str], stock_symbols: List[str]) -> str:
        """Generate alert title."""
        
        # Priority indicator
        priority_emoji = {
            AlertPriority.LOW: "📊",
            AlertPriority.MEDIUM: "⚠️",
            AlertPriority.HIGH: "🚨",
            AlertPriority.CRITICAL: "🔥"
        }
        
        emoji = priority_emoji.get(rule.priority, "📰")
        
        # Event type indicator
        event_emoji = {
            "earnings": "📈",
            "merger_acquisition": "🤝",
            "monetary_policy": "🏛️",
            "breaking_news": "⚡",
            "regulatory": "⚖️",
            "ipo": "🆕",
            "bankruptcy": "💥"
        }
        
        # Build title
        title_parts = [emoji]
        
        if event_types:
            event_emoji_str = "".join([event_emoji.get(et, "") for et in event_types[:2]])
            if event_emoji_str:
                title_parts.append(event_emoji_str)
        
        if stock_symbols:
            symbols_str = ", ".join(stock_symbols[:3])
            if len(stock_symbols) > 3:
                symbols_str += f" +{len(stock_symbols)-3} more"
            title_parts.append(f"[{symbols_str}]")
        
        title_parts.append(article.title[:50] + ("..." if len(article.title) > 50 else ""))
        
        return " ".join(title_parts)
    
    def _generate_alert_message(self, rule: AlertRule, article: NewsArticle,
                               sentiment_result: Any, event_result: Any) -> str:
        """Generate detailed alert message."""
        
        lines = [
            f"Alert: {rule.description}",
            f"Priority: {rule.priority.value.upper()}",
            "",
            f"Article: {article.title}",
            f"Source: {article.source}",
            f"Published: {article.published_at.strftime('%Y-%m-%d %H:%M') if article.published_at else 'Unknown'}",
            "",
            f"Sentiment: {sentiment_result.sentiment_score:.3f} ({getattr(sentiment_result, 'sentiment_label', 'unknown')})",
            f"Confidence: {getattr(sentiment_result, 'confidence', 0.5):.3f}",
        ]
        
        if event_result.events_found > 0:
            lines.append("")
            lines.append("Events detected:")
            for event in event_result.events[:3]:  # Show top 3 events
                lines.append(f"  • {event.event_type.value} (confidence: {event.confidence:.3f})")
        
        stock_symbols = article.stock_symbols or []
        if isinstance(stock_symbols, str):
            stock_symbols = json.loads(stock_symbols)
        
        if stock_symbols:
            lines.append(f"Stocks: {', '.join(stock_symbols)}")
        
        lines.append("")
        lines.append(f"URL: {article.url}")
        
        return "\n".join(lines)
    
    def deliver_alerts(self, alerts: List[GeneratedAlert]):
        """Deliver alerts through configured channels."""
        
        for alert in alerts:
            try:
                # Always store in database
                self._store_alert_in_database(alert)
                
                # Console output
                if AlertChannel.CONSOLE in alert.channels:
                    self._send_console_alert(alert)
                
                # Email delivery
                if AlertChannel.EMAIL in alert.channels and self.email_config:
                    self._send_email_alert(alert)
                
                # File logging
                if AlertChannel.FILE in alert.channels:
                    self._log_alert_to_file(alert)
                
            except Exception as e:
                logger.error(f"Failed to deliver alert: {e}")
    
    def _store_alert_in_database(self, alert: GeneratedAlert):
        """Store alert in database."""
        db_alert = Alert(
            alert_type=alert.rule_name,
            title=alert.title,
            message=alert.message,
            priority=alert.priority.value,
            stock_symbol=", ".join(alert.stock_symbols) if alert.stock_symbols else None,
            triggered_by=alert.article_id
        )
        
        self.session.create_alert(db_alert)
    
    def _send_console_alert(self, alert: GeneratedAlert):
        """Send alert to console."""
        print("\n" + "="*60)
        print(f"ALERT: {alert.title}")
        print("="*60)
        print(alert.message)
        print("="*60 + "\n")
    
    def _send_email_alert(self, alert: GeneratedAlert):
        """Send alert via email."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = self.email_config['to_email']
            msg['Subject'] = f"News Intelligence Alert: {alert.title}"
            
            msg.attach(MIMEText(alert.message, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.sendmail(msg['From'], msg['To'], msg.as_string())
            server.quit()
            
            logger.info(f"Email alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _log_alert_to_file(self, alert: GeneratedAlert):
        """Log alert to file."""
        try:
            with open('alerts.log', 'a') as f:
                f.write(f"{datetime.now().isoformat()} - {alert.priority.value.upper()} - {alert.title}\n")
                f.write(f"{alert.message}\n")
                f.write("-" * 80 + "\n")
        except Exception as e:
            logger.error(f"Failed to log alert to file: {e}")
    
    def monitor_new_articles(self, check_interval_minutes: int = 5):
        """Monitor for new articles and generate alerts."""
        logger.info("Starting news monitoring...")
        
        # Get timestamp of last check
        last_check = datetime.now() - timedelta(minutes=check_interval_minutes)
        
        # Get new articles since last check
        new_articles = self.session.db.execute_query(
            "SELECT * FROM news_articles WHERE collected_at > ? ORDER BY collected_at DESC",
            (last_check,)
        )
        
        if new_articles:
            logger.info(f"Found {len(new_articles)} new articles to analyze")
            
            all_alerts = []
            
            for article_row in new_articles:
                article = self.session._row_to_article(article_row)
                alerts = self.process_article(article)
                all_alerts.extend(alerts)
            
            if all_alerts:
                logger.info(f"Generated {len(all_alerts)} alerts")
                self.deliver_alerts(all_alerts)
            else:
                logger.info("No alerts generated")
        else:
            logger.info("No new articles found")
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of alerts generated in the last N hours."""
        since = datetime.now() - timedelta(hours=hours)
        
        alerts = self.session.db.execute_query(
            "SELECT * FROM alerts WHERE created_at > ?",
            (since,)
        )
        
        if not alerts:
            return {'total_alerts': 0, 'period_hours': hours}
        
        # Analyze alerts
        priority_counts = {}
        type_counts = {}
        
        for alert in alerts:
            priority = alert['priority']
            alert_type = alert['alert_type']
            
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        
        return {
            'total_alerts': len(alerts),
            'period_hours': hours,
            'by_priority': priority_counts,
            'by_type': type_counts,
            'latest_alert': alerts[0]['created_at'] if alerts else None
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the alert system
    alert_system = NewsIntelligenceAlertSystem()
    
    print("=== News Intelligence Alert System Test ===\n")
    
    # Get recent articles from database
    session = get_session()
    articles = session.get_articles(limit=5)
    
    if articles:
        print(f"Testing alert generation with {len(articles)} articles...\n")
        
        all_alerts = []
        
        for article in articles:
            print(f"Processing: {article.title[:60]}...")
            alerts = alert_system.process_article(article)
            
            if alerts:
                print(f"  Generated {len(alerts)} alerts")
                all_alerts.extend(alerts)
            else:
                print(f"  No alerts generated")
        
        print(f"\nTotal alerts generated: {len(all_alerts)}")
        
        if all_alerts:
            print("\nDelivering alerts...")
            alert_system.deliver_alerts(all_alerts)
            
            # Show alert summary
            summary = alert_system.get_alert_summary(1)  # Last hour
            print(f"\nAlert Summary: {summary}")
    
    else:
        print("No articles found in database. Run news collection first!")
    
    print("\nAlert system test completed!")