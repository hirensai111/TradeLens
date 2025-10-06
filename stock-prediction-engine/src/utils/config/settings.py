"""
Stock Prediction Engine - Phase 3: News Intelligence Engine
Core Configuration and Settings Management

This module handles all configuration for the comprehensive news intelligence system,
including API keys, database connections, scraping parameters, and system settings.
"""

import os
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
from enum import Enum
import json
from pathlib import Path


class EnvironmentType(str, Enum):
    """Environment types for the application"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseType(str, Enum):
    """Supported database types"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


class Settings(BaseSettings):
    """
    Comprehensive configuration settings for the News Intelligence Engine
    
    This class manages all configuration parameters including:
    - API credentials and endpoints
    - Database connections
    - Scraping parameters and rate limits
    - AI/ML model settings
    - Monitoring and alerting
    - System performance tuning
    """
    
    # ================================
    # CORE APPLICATION SETTINGS
    # ================================
    
    app_name: str = Field(default="Stock Prediction Engine - Phase 3", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: EnvironmentType = Field(default=EnvironmentType.DEVELOPMENT, description="Runtime environment")
    debug: bool = Field(default=True, description="Debug mode flag")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    
    # Base directories
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    logs_dir: Path = Field(default_factory=lambda: Path("logs"))
    config_dir: Path = Field(default_factory=lambda: Path("config"))
    
    # ================================
    # DATABASE CONFIGURATION
    # ================================
    
    # Database settings
    database_type: DatabaseType = Field(default=DatabaseType.SQLITE, description="Database type")
    database_url: Optional[str] = Field(default=None, description="Complete database URL")
    
    # SQLite settings
    sqlite_db_path: str = Field(default="data/news_intelligence.db", description="SQLite database file path")
    
    # PostgreSQL settings
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_database: str = Field(default="news_intelligence", description="PostgreSQL database name")
    postgres_username: str = Field(default="postgres", description="PostgreSQL username")
    postgres_password: str = Field(default="", description="PostgreSQL password")
    
    # Database pool settings
    db_pool_size: int = Field(default=10, description="Database connection pool size")
    db_max_overflow: int = Field(default=20, description="Database max overflow connections")
    db_pool_timeout: int = Field(default=30, description="Database pool timeout seconds")
    
    # ================================
    # NEWS API CONFIGURATIONS
    # ================================
    
    # NewsAPI settings
    newsapi_key: Optional[str] = Field(default=None, description="NewsAPI key")
    newsapi_base_url: str = Field(default="https://newsapi.org/v2/", description="NewsAPI base URL")
    newsapi_rate_limit: int = Field(default=1000, description="NewsAPI daily rate limit")
    
    # Alpha Vantage settings
    alpha_vantage_key: Optional[str] = Field(default=None, description="Alpha Vantage API key")
    alpha_vantage_base_url: str = Field(default="https://www.alphavantage.co/query", description="Alpha Vantage base URL")
    alpha_vantage_rate_limit: int = Field(default=5, description="Alpha Vantage requests per minute")
    
    # Polygon settings
    polygon_key: Optional[str] = Field(default=None, description="Polygon.io API key")
    polygon_base_url: str = Field(default="https://api.polygon.io", description="Polygon base URL")
    
    # Yahoo Finance settings (unofficial API)
    yahoo_finance_rate_limit: int = Field(default=100, description="Yahoo Finance requests per hour")
    
    # ================================
    # SOCIAL MEDIA API CONFIGURATIONS
    # ================================
    
    # Reddit API settings
    reddit_client_id: Optional[str] = Field(default=None, description="Reddit client ID")
    reddit_client_secret: Optional[str] = Field(default=None, description="Reddit client secret")
    reddit_user_agent: str = Field(default="NewsIntelligenceBot/1.0", description="Reddit user agent")
    reddit_rate_limit: int = Field(default=60, description="Reddit requests per minute")
    
    # Twitter/X API settings
    twitter_bearer_token: Optional[str] = Field(default=None, description="Twitter Bearer token")
    twitter_api_key: Optional[str] = Field(default=None, description="Twitter API key")
    twitter_api_secret: Optional[str] = Field(default=None, description="Twitter API secret")
    twitter_access_token: Optional[str] = Field(default=None, description="Twitter access token")
    twitter_access_secret: Optional[str] = Field(default=None, description="Twitter access secret")
    
    # ================================
    # AI/ML API CONFIGURATIONS
    # ================================
    
    # OpenAI settings
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI model for sentiment analysis")
    openai_max_tokens: int = Field(default=1000, description="OpenAI max tokens per request")
    openai_temperature: float = Field(default=0.3, description="OpenAI temperature for consistency")
    openai_rate_limit: int = Field(default=60, description="OpenAI requests per minute")
    
    # Hugging Face settings
    huggingface_api_key: Optional[str] = Field(default=None, description="Hugging Face API key")
    huggingface_model: str = Field(default="cardiffnlp/twitter-roberta-base-sentiment", description="HF sentiment model")
    
    # ================================
    # WEB SCRAPING CONFIGURATIONS
    # ================================
    
    # General scraping settings
    scraping_delay_min: float = Field(default=1.0, description="Minimum delay between requests (seconds)")
    scraping_delay_max: float = Field(default=3.0, description="Maximum delay between requests (seconds)")
    scraping_timeout: int = Field(default=30, description="Request timeout in seconds")
    scraping_retries: int = Field(default=3, description="Number of retries for failed requests")
    scraping_concurrent_limit: int = Field(default=5, description="Maximum concurrent scraping requests")
    
    # User agent rotation
    use_rotating_user_agents: bool = Field(default=True, description="Enable user agent rotation")
    custom_user_agents: List[str] = Field(default_factory=list, description="Custom user agents list")
    
    # Proxy settings
    use_proxies: bool = Field(default=False, description="Enable proxy rotation")
    proxy_list: List[str] = Field(default_factory=list, description="List of proxy servers")
    
    # Browser automation settings
    headless_browser: bool = Field(default=True, description="Run browser in headless mode")
    browser_timeout: int = Field(default=60, description="Browser operation timeout")
    
    # ================================
    # NEWS SOURCE CONFIGURATIONS
    # ================================
    
    # News sources with reliability weights
    news_sources_config: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "tier_1": {
                "sources": ["Reuters", "Bloomberg", "Wall Street Journal", "SEC.gov", "FDA.gov"],
                "weight": 1.0,
                "confidence": 0.95,
                "priority": 1
            },
            "tier_2": {
                "sources": ["CNBC", "MarketWatch", "Yahoo Finance", "Seeking Alpha", "Financial Times"],
                "weight": 0.8,
                "confidence": 0.85,
                "priority": 2
            },
            "tier_3": {
                "sources": ["TechCrunch", "Motley Fool", "Benzinga", "VentureBeat"],
                "weight": 0.6,
                "confidence": 0.75,
                "priority": 3
            },
            "social_media": {
                "sources": ["Reddit", "Twitter", "Social Forums"],
                "weight": 0.3,
                "confidence": 0.60,
                "priority": 4
            }
        },
        description="News sources configuration with reliability scoring"
    )
    
    # News categories configuration
    news_categories_config: Dict[str, Dict[str, List[str]]] = Field(
        default_factory=lambda: {
            "financial": {
                "earnings": ["earnings", "revenue", "profit", "EPS", "guidance", "quarterly"],
                "analyst": ["upgrade", "downgrade", "price target", "rating", "recommendation"],
                "metrics": ["cash flow", "debt", "dividend", "margins", "ROI", "EBITDA"]
            },
            "political": {
                "regulatory": ["SEC", "FDA", "FTC", "antitrust", "investigation", "compliance"],
                "policy": ["tax", "trade", "regulation", "policy", "tariff", "sanctions"],
                "elections": ["election", "political", "government", "legislation", "congress"]
            },
            "products": {
                "launches": ["launch", "release", "new product", "unveil", "announce"],
                "issues": ["recall", "defect", "problem", "lawsuit", "safety", "quality"],
                "innovation": ["patent", "R&D", "breakthrough", "technology", "innovation"]
            },
            "events": {
                "corporate": ["merger", "acquisition", "partnership", "deal", "joint venture"],
                "leadership": ["CEO", "executive", "board", "appointment", "resignation"],
                "legal": ["lawsuit", "court", "settlement", "ruling", "litigation"]
            },
            "broader": {
                "supply_chain": ["supply", "shortage", "raw materials", "shipping", "logistics"],
                "economic": ["interest rates", "inflation", "GDP", "currency", "recession"],
                "social": ["consumer", "social media", "cultural", "trend", "sentiment"]
            }
        },
        description="News categorization keywords and patterns"
    )
    
    # ================================
    # SENTIMENT ANALYSIS SETTINGS
    # ================================
    
    # Sentiment analysis parameters
    sentiment_confidence_threshold: float = Field(default=0.7, description="Minimum confidence for sentiment scores")
    sentiment_batch_size: int = Field(default=10, description="Batch size for sentiment analysis")
    sentiment_cache_ttl: int = Field(default=3600, description="Sentiment cache TTL in seconds")
    
    # Sentiment scoring weights
    sentiment_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "very_negative": -1.0,
            "negative": -0.5,
            "neutral": 0.0,
            "positive": 0.5,
            "very_positive": 1.0
        },
        description="Sentiment scoring weights"
    )
    
    # ================================
    # REAL-TIME MONITORING SETTINGS
    # ================================
    
    # Real-time monitoring parameters
    real_time_monitoring_enabled: bool = Field(default=True, description="Enable real-time monitoring")
    monitoring_interval_seconds: int = Field(default=300, description="Monitoring check interval")
    alert_threshold_high: float = Field(default=0.8, description="High priority alert threshold")
    alert_threshold_medium: float = Field(default=0.6, description="Medium priority alert threshold")
    
    # News freshness settings
    news_freshness_hours: int = Field(default=24, description="Consider news fresh for X hours")
    breaking_news_keywords: List[str] = Field(
        default_factory=lambda: ["breaking", "urgent", "alert", "just in", "developing"],
        description="Keywords indicating breaking news"
    )
    
    # ================================
    # CACHING AND PERFORMANCE
    # ================================
    
    # Redis cache settings
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_ttl_default: int = Field(default=3600, description="Default Redis TTL")
    
    # Performance settings
    max_workers: int = Field(default=10, description="Maximum worker threads")
    chunk_size: int = Field(default=100, description="Processing chunk size")
    memory_limit_mb: int = Field(default=1024, description="Memory limit in MB")
    
    # ================================
    # NOTIFICATION SETTINGS
    # ================================
    
    # Email notifications
    smtp_host: Optional[str] = Field(default=None, description="SMTP host")
    smtp_port: int = Field(default=587, description="SMTP port")
    smtp_username: Optional[str] = Field(default=None, description="SMTP username")
    smtp_password: Optional[str] = Field(default=None, description="SMTP password")
    notification_emails: List[str] = Field(default_factory=list, description="Notification email addresses")
    
    # Slack notifications
    slack_webhook_url: Optional[str] = Field(default=None, description="Slack webhook URL")
    slack_channel: str = Field(default="#alerts", description="Slack channel for alerts")
    
    # ================================
    # SYSTEM MONITORING
    # ================================
    
    # Health check settings
    health_check_enabled: bool = Field(default=True, description="Enable health checks")
    health_check_interval: int = Field(default=60, description="Health check interval in seconds")
    
    # Metrics and monitoring
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    prometheus_port: int = Field(default=9090, description="Prometheus metrics port")
    
    # ================================
    # PHASE 4 INTEGRATION SETTINGS
    # ================================
    
    # Export settings for Phase 4
    export_format: str = Field(default="json", description="Export format for Phase 4 integration")
    export_interval_minutes: int = Field(default=15, description="Export interval for Phase 4")
    export_path: str = Field(default="data/exports/", description="Export path for Phase 4 data")
    
    # ML feature extraction settings
    feature_extraction_enabled: bool = Field(default=True, description="Enable ML feature extraction")
    feature_window_days: int = Field(default=30, description="Feature extraction window in days")
    
    # Pydantic v2 configuration
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_assignment=True,
        extra="ignore"
    )
    
    @field_validator("database_url")
    @classmethod
    def build_database_url(cls, v, info):
        """Build database URL based on database type and settings"""
        if v:
            return v
        
        # Get values from info.data (field values)
        values = info.data
        db_type = values.get("database_type")
        
        if db_type == DatabaseType.SQLITE:
            sqlite_path = values.get("sqlite_db_path", "data/news_intelligence.db")
            return f"sqlite:///{sqlite_path}"
        
        elif db_type == DatabaseType.POSTGRESQL:
            host = values.get("postgres_host", "localhost")
            port = values.get("postgres_port", 5432)
            database = values.get("postgres_database", "news_intelligence")
            username = values.get("postgres_username", "postgres")
            password = values.get("postgres_password", "")
            return f"postgresql://{username}:{password}@{host}:{port}/{database}"
        
        return v
    
    @field_validator("data_dir", "logs_dir", "config_dir")
    @classmethod
    def create_directories(cls, v):
        """Ensure directories exist"""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    def get_api_config(self, api_name: str) -> Dict[str, Any]:
        """Get configuration for specific API"""
        api_configs = {
            "newsapi": {
                "key": self.newsapi_key,
                "base_url": self.newsapi_base_url,
                "rate_limit": self.newsapi_rate_limit
            },
            "alpha_vantage": {
                "key": self.alpha_vantage_key,
                "base_url": self.alpha_vantage_base_url,
                "rate_limit": self.alpha_vantage_rate_limit
            },
            "openai": {
                "key": self.openai_api_key,
                "model": self.openai_model,
                "max_tokens": self.openai_max_tokens,
                "temperature": self.openai_temperature,
                "rate_limit": self.openai_rate_limit
            },
            "reddit": {
                "client_id": self.reddit_client_id,
                "client_secret": self.reddit_client_secret,
                "user_agent": self.reddit_user_agent,
                "rate_limit": self.reddit_rate_limit
            }
        }
        return api_configs.get(api_name, {})
    
    def get_scraping_config(self) -> Dict[str, Any]:
        """Get web scraping configuration"""
        return {
            "delay_range": (self.scraping_delay_min, self.scraping_delay_max),
            "timeout": self.scraping_timeout,
            "retries": self.scraping_retries,
            "concurrent_limit": self.scraping_concurrent_limit,
            "use_rotating_agents": self.use_rotating_user_agents,
            "custom_agents": self.custom_user_agents,
            "use_proxies": self.use_proxies,
            "proxy_list": self.proxy_list,
            "headless": self.headless_browser
        }
    
    def get_source_weight(self, source: str) -> float:
        """Get reliability weight for a news source"""
        for tier_config in self.news_sources_config.values():
            if source in tier_config["sources"]:
                return tier_config["weight"]
        return 0.5  # Default weight for unknown sources
    
    def get_source_confidence(self, source: str) -> float:
        """Get confidence level for a news source"""
        for tier_config in self.news_sources_config.values():
            if source in tier_config["sources"]:
                return tier_config["confidence"]
        return 0.6  # Default confidence for unknown sources
    
    def is_breaking_news(self, title: str, content: str = "") -> bool:
        """Check if news qualifies as breaking news"""
        text = f"{title} {content}".lower()
        return any(keyword in text for keyword in self.breaking_news_keywords)
    
    def export_config(self, file_path: str):
        """Export current configuration to file"""
        config_dict = self.model_dump()
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def load_config_from_file(self, file_path: str):
        """Load configuration from file"""
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        
        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings


def reload_settings():
    """Reload settings from environment"""
    global settings
    settings = Settings()
    return settings


# Configuration validation
def validate_configuration():
    """Validate that all required configuration is present"""
    required_api_keys = []
    missing_keys = []
    
    # Check for required API keys based on enabled features
    if settings.newsapi_key is None:
        missing_keys.append("NEWSAPI_KEY")
    
    if settings.openai_api_key is None:
        missing_keys.append("OPENAI_API_KEY")
    
    if missing_keys:
        print(f"Warning: Missing API keys: {', '.join(missing_keys)}")
        print("Some features may not work properly without these keys.")
    
    # Validate database configuration
    try:
        # Test database URL construction
        db_url = settings.database_url
        print(f"Database URL configured: {db_url}")
    except Exception as e:
        print(f"Database configuration error: {e}")
    
    return len(missing_keys) == 0


if __name__ == "__main__":
    # Test configuration loading
    print(f"App: {settings.app_name} v{settings.app_version}")
    print(f"Environment: {settings.environment}")
    print(f"Database URL: {settings.database_url}")
    print(f"Debug mode: {settings.debug}")
    
    # Validate configuration
    is_valid = validate_configuration()
    print(f"Configuration valid: {is_valid}")
    
    # Export sample configuration
    settings.export_config("config/sample_config.json")