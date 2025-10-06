#!/usr/bin/env python3
"""
Feature Engineering Pipeline Improvements and Fixes
Addresses import issues, adds missing features, and enhances robustness
"""

import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Fix import paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'data_loaders'))

@dataclass
class MLFeatures:
    """Complete feature set for ML models"""
    # Price features
    price_features: Dict[str, float]
    
    # Technical indicator features
    technical_features: Dict[str, float]
    
    # Sentiment features
    sentiment_features: Dict[str, float]
    
    # Time-based features
    time_features: Dict[str, float]
    
    # Derived features
    derived_features: Dict[str, float]
    
    # NEW: Fundamental features (from enhanced Excel loader)
    fundamental_features: Dict[str, float]
    
    # NEW: Quality features (data reliability)
    quality_features: Dict[str, float]
    
    # Target variable (for training)
    target: Optional[float] = None
    
    # Metadata
    date: Optional[datetime] = None
    ticker: Optional[str] = None

class FeatureEngineer:
    """
    Enhanced Feature Engineering Pipeline with:
    1. Fixed import issues
    2. All Excel sheets integration
    3. Improved Phase 3 connector integration
    4. Enhanced derived features
    5. Better error handling
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize enhanced feature engineer"""
        # Setup logging FIRST before anything else
        self.logger = self._setup_logging()
        
        # NOW we can load config (which uses self.logger)
        self.config = self._load_config(config_path)
        
        # Initialize data loaders with improved error handling
        self.excel_loader = self._initialize_excel_loader()
        self.news_connector = self._initialize_news_connector()
        
        # Load Excel data once
        self.excel_data = self._load_excel_data()
        
        # Feature scaling parameters (for normalization)
        self.feature_scalers = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration with comprehensive fallback"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.logger.info(f"Loaded config from {config_path}")
                return config
        except Exception as e:
            self.logger.warning(f"Could not load config {config_path}: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Enhanced default configuration"""
        return {
            'data_sources': {
                'excel_path': 'D:/stock-prediction-engine/phase4_ml_engine/data/MSFT_analysis_report_20250715_141735.xlsx',
                'phase3_db_path': 'D:/stock-prediction-engine/data/news_intelligence.db'
            },
            'features': {
                'technical_indicators': [
                    'SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_Signal', 
                    'BB_Upper', 'BB_Lower', 'ATR', 'Volume'
                ],
                'sentiment_features': [
                    'sentiment_1d', 'sentiment_3d', 'sentiment_7d', 'news_volume_1d',
                    'correlation_strength', 'event_impact_score', 'confidence_score'
                ],
                'time_features': [
                    'day_of_week', 'month', 'quarter', 'is_earnings_season',
                    'is_month_end', 'is_quarter_end'
                ],
                'quality_thresholds': {
                    'min_data_completeness': 0.95,
                    'min_confidence_score': 0.7,
                    'max_missing_features': 0.1
                }
            },
            'ml_settings': {
                'target_offset_days': 1,
                'lookback_days': 30,
                'min_training_samples': 100,
                'feature_selection_threshold': 0.01
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup enhanced logging"""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/feature_engineering.log', mode='a')
            ]
        )
        return logging.getLogger(__name__)
    
    def _initialize_excel_loader(self):
        """Initialize Excel loader with proper error handling"""
        try:
            # Try to import the enhanced Excel loader
            from excel_loader import EnhancedExcelDataLoader
            self.logger.info("Using EnhancedExcelDataLoader")
            return EnhancedExcelDataLoader()
        except ImportError:
            try:
                # Fallback to standard Excel loader
                from excel_loader import ExcelDataLoader
                self.logger.info("Using standard ExcelDataLoader")
                return ExcelDataLoader()
            except ImportError as e:
                self.logger.error(f"Could not import Excel loader: {e}")
                self.logger.info("Creating mock Excel loader")
                return self._create_mock_excel_loader()
    
    def _initialize_news_connector(self):
        """Initialize news connector with proper error handling"""
        try:
            # Try to import the enhanced Phase 3 connector
            from phase3_connector import EnhancedPhase3Connector
            self.logger.info("Using EnhancedPhase3Connector")
            return EnhancedPhase3Connector()
        except ImportError:
            try:
                # Fallback to standard connector
                from phase3_connector import Phase3NewsConnector
                self.logger.info("Using Phase3NewsConnector")
                return Phase3NewsConnector()
            except ImportError as e:
                self.logger.error(f"Could not import Phase 3 connector: {e}")
                self.logger.info("Creating mock news connector")
                return self._create_mock_news_connector()
    
    def _create_mock_excel_loader(self):
        """Create mock Excel loader for testing"""
        class MockExcelLoader:
            def load_all_data(self):
                # Create mock data for testing
                dates = pd.date_range(start='2024-01-01', end='2025-07-14', freq='D')
                mock_data = {
                    'raw_data': pd.DataFrame({
                        'Date': dates,
                        'Open': np.random.uniform(400, 500, len(dates)),
                        'High': np.random.uniform(450, 550, len(dates)),
                        'Low': np.random.uniform(350, 450, len(dates)),
                        'Close': np.random.uniform(400, 500, len(dates)),
                        'Volume': np.random.randint(1000000, 50000000, len(dates)),
                        'Daily Change %': np.random.uniform(-5, 5, len(dates))
                    }),
                    'technical_data': pd.DataFrame({
                        'Date': dates[-60:],  # Recent data only
                        'Close': np.random.uniform(400, 500, 60),
                        'SMA_20': np.random.uniform(450, 480, 60),
                        'SMA_50': np.random.uniform(440, 470, 60),
                        'RSI': np.random.uniform(30, 70, 60),
                        'MACD': np.random.uniform(-2, 2, 60),
                        'MACD_Signal': np.random.uniform(-2, 2, 60)
                    })
                }
                return mock_data
        
        return MockExcelLoader()
    
    def _create_mock_news_connector(self):
        """Create mock news connector for testing"""
        class MockNewsConnector:
            def get_enhanced_sentiment_features(self, ticker, target_date, lookback_days=7):
                # Mock sentiment features
                from types import SimpleNamespace
                return SimpleNamespace(
                    sentiment_1d=np.random.uniform(-0.5, 0.5),
                    sentiment_3d=np.random.uniform(-0.5, 0.5),
                    sentiment_7d=np.random.uniform(-0.5, 0.5),
                    news_volume_1d=np.random.randint(0, 10),
                    news_volume_3d=np.random.randint(0, 20),
                    news_volume_7d=np.random.randint(0, 30),
                    correlation_strength=np.random.uniform(0, 1),
                    event_impact_score=np.random.uniform(0, 1),
                    confidence_score=np.random.uniform(0.5, 1.0),
                    alert_count=np.random.randint(0, 5),
                    source_diversity=np.random.randint(1, 5)
                )
            
            def get_enhanced_market_overview(self, target_date):
                return {
                    'market_sentiment': np.random.uniform(-0.3, 0.3),
                    'total_articles': np.random.randint(50, 200),
                    'source_diversity': np.random.randint(3, 8),
                    'stocks_mentioned': np.random.randint(20, 100)
                }
        
        return MockNewsConnector()
    
    def _load_excel_data(self) -> Dict:
        """Load Excel data with comprehensive error handling"""
        try:
            if hasattr(self.excel_loader, 'load_all_data'):
                excel_data = self.excel_loader.load_all_data()
                self.logger.info(f"Successfully loaded {len(excel_data)} Excel data sources")
                return excel_data
            else:
                self.logger.error("Excel loader does not have load_all_data method")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load Excel data: {e}")
            return {}
    
    def create_enhanced_features_for_date(self, ticker: str, target_date: datetime, 
                                        include_target: bool = False, target_offset: int = 1) -> MLFeatures:
        """
        Create enhanced feature set including ALL Excel sheets and Phase 3 data
        """
        target_date = pd.to_datetime(target_date)
        
        # 1. Price Features (enhanced)
        price_features = self._extract_enhanced_price_features(ticker, target_date)
        
        # 2. Technical Indicator Features (enhanced)
        technical_features = self._extract_enhanced_technical_features(ticker, target_date)
        
        # 3. Sentiment Features (enhanced with Phase 3)
        sentiment_features = self._extract_enhanced_sentiment_features(ticker, target_date)
        
        # 4. Time-based Features (enhanced)
        time_features = self._extract_enhanced_time_features(target_date)
        
        # 5. NEW: Fundamental Features (from Summary, Company Info)
        fundamental_features = self.extract_fundamental_features(ticker, target_date)
        
        # 6. NEW: Quality Features (from Data Quality sheet)
        quality_features = self._extract_quality_features(ticker, target_date)
        
        # 7. Enhanced Derived Features
        derived_features = self._create_enhanced_derived_features(
            price_features, technical_features, sentiment_features, 
            time_features, fundamental_features, quality_features
        )
        
        # 8. Target Variable (if requested)
        target = None
        if include_target:
            target = self._extract_target_variable(ticker, target_date, target_offset)
        
        return MLFeatures(
            price_features=price_features,
            technical_features=technical_features,
            sentiment_features=sentiment_features,
            time_features=time_features,
            fundamental_features=fundamental_features,
            quality_features=quality_features,
            derived_features=derived_features,
            target=target,
            date=target_date,
            ticker=ticker
        )
    
    def _extract_enhanced_price_features(self, ticker: str, target_date: datetime) -> Dict[str, float]:
        """Enhanced price feature extraction with more indicators"""
        features = {}
        
        try:
            if 'raw_data' in self.excel_data and self.excel_data['raw_data'] is not None:
                raw_data = self.excel_data['raw_data']
                date_mask = raw_data['Date'] <= target_date
                recent_data = raw_data[date_mask].tail(50)  # Increased lookback
                
                if not recent_data.empty:
                    latest = recent_data.iloc[-1]
                    
                    # Enhanced OHLCV features
                    features.update({
                        'open': float(latest['Open']),
                        'high': float(latest['High']),
                        'low': float(latest['Low']),
                        'close': float(latest['Close']),
                        'volume': float(latest['Volume']),
                        'daily_change_pct': float(latest['Daily Change %'])
                    })
                    
                    # Enhanced price action features
                    hl_range = latest['High'] - latest['Low']
                    features.update({
                        'price_range': float(hl_range),
                        'price_range_pct': float(hl_range / latest['Close']) if latest['Close'] > 0 else 0,
                        'body_size': float(abs(latest['Close'] - latest['Open'])),
                        'body_size_pct': float(abs(latest['Close'] - latest['Open']) / hl_range) if hl_range > 0 else 0,
                        'upper_shadow': float(latest['High'] - max(latest['Open'], latest['Close'])),
                        'lower_shadow': float(min(latest['Open'], latest['Close']) - latest['Low']),
                        'body_position': float((latest['Close'] - latest['Low']) / hl_range) if hl_range > 0 else 0.5,
                        'is_doji': float(abs(latest['Close'] - latest['Open']) / hl_range < 0.1) if hl_range > 0 else 0,
                        'is_hammer': float(latest['Low'] < min(latest['Open'], latest['Close']) - 2 * abs(latest['Close'] - latest['Open'])),
                        'is_shooting_star': float(latest['High'] > max(latest['Open'], latest['Close']) + 2 * abs(latest['Close'] - latest['Open']))
                    })
                    
                    # Enhanced rolling statistics
                    for window in [5, 10, 20]:
                        if len(recent_data) >= window:
                            window_data = recent_data.tail(window)
                            features.update({
                                f'close_sma_{window}': float(window_data['Close'].mean()),
                                f'close_std_{window}': float(window_data['Close'].std()),
                                f'volume_sma_{window}': float(window_data['Volume'].mean()),
                                f'volume_std_{window}': float(window_data['Volume'].std()),
                                f'volatility_{window}d': float(window_data['Daily Change %'].std()),
                                f'avg_range_{window}d': float((window_data['High'] - window_data['Low']).mean()),
                                f'high_low_ratio_{window}d': float((window_data['High'] / window_data['Low']).mean()) if (window_data['Low'] > 0).all() else 1.0
                            })
                    
                    # Enhanced momentum features
                    for lag in [1, 2, 3, 5, 10]:
                        if len(recent_data) > lag:
                            prev_close = recent_data.iloc[-(lag+1)]['Close']
                            features[f'price_momentum_{lag}d'] = float((latest['Close'] - prev_close) / prev_close)
                            
                            prev_volume = recent_data.iloc[-(lag+1)]['Volume']
                            features[f'volume_momentum_{lag}d'] = float((latest['Volume'] - prev_volume) / prev_volume) if prev_volume > 0 else 0
                    
                    # Price channel features
                    if len(recent_data) >= 20:
                        high_20 = recent_data['High'].tail(20).max()
                        low_20 = recent_data['Low'].tail(20).min()
                        features.update({
                            'price_channel_position': float((latest['Close'] - low_20) / (high_20 - low_20)) if high_20 > low_20 else 0.5,
                            'near_high_20': float(latest['Close'] > high_20 * 0.98),
                            'near_low_20': float(latest['Close'] < low_20 * 1.02)
                        })
                    
                    # Volume profile features
                    if len(recent_data) >= 10:
                        avg_volume = recent_data['Volume'].mean()
                        features.update({
                            'volume_ratio': float(latest['Volume'] / avg_volume) if avg_volume > 0 else 1.0,
                            'high_volume_day': float(latest['Volume'] > avg_volume * 1.5),
                            'low_volume_day': float(latest['Volume'] < avg_volume * 0.5),
                            'volume_trend_slope': float(np.polyfit(range(len(recent_data)), recent_data['Volume'], 1)[0]) if len(recent_data) > 1 else 0.0
                        })
        
        except Exception as e:
            self.logger.warning(f"Error extracting enhanced price features: {e}")
        
        return features
    
    def _extract_enhanced_technical_features(self, ticker: str, target_date: datetime) -> Dict[str, float]:
        """Enhanced technical feature extraction with additional indicators"""
        features = {}
        
        try:
            if 'technical_data' in self.excel_data and self.excel_data['technical_data'] is not None:
                tech_data = self.excel_data['technical_data']
                date_mask = tech_data['Date'] <= target_date
                recent_tech = tech_data[date_mask].tail(50)  # More historical data
                
                if not recent_tech.empty:
                    latest_tech = recent_tech.iloc[-1]
                    
                    # Standard technical indicators
                    technical_indicators = self.config['features']['technical_indicators']
                    for indicator in technical_indicators:
                        if indicator in latest_tech.index and pd.notna(latest_tech[indicator]):
                            features[f'tech_{indicator.lower()}'] = float(latest_tech[indicator])
                    
                    # Enhanced RSI features
                    if 'RSI' in latest_tech.index and pd.notna(latest_tech['RSI']):
                        rsi = float(latest_tech['RSI'])
                        features.update({
                            'rsi_overbought': float(rsi > 70),
                            'rsi_oversold': float(rsi < 30),
                            'rsi_neutral': float(30 <= rsi <= 70),
                            'rsi_extreme_overbought': float(rsi > 80),
                            'rsi_extreme_oversold': float(rsi < 20),
                            'rsi_momentum': float(rsi - 50) / 50  # Normalized RSI momentum
                        })
                        
                        # RSI trend
                        if len(recent_tech) >= 5 and 'RSI' in recent_tech.columns:
                            rsi_series = recent_tech['RSI'].tail(5)
                            if not rsi_series.isna().all():
                                features['rsi_trend'] = float(np.polyfit(range(len(rsi_series)), rsi_series.fillna(50), 1)[0])
                    
                    # Enhanced MACD features
                    if 'MACD' in latest_tech.index and 'MACD_Signal' in latest_tech.index:
                        if pd.notna(latest_tech['MACD']) and pd.notna(latest_tech['MACD_Signal']):
                            macd = float(latest_tech['MACD'])
                            signal = float(latest_tech['MACD_Signal'])
                            histogram = macd - signal
                            
                            features.update({
                                'macd_histogram': histogram,
                                'macd_bullish': float(macd > signal),
                                'macd_bearish': float(macd < signal),
                                'macd_strong_bullish': float(histogram > 0.5),
                                'macd_strong_bearish': float(histogram < -0.5),
                                'macd_divergence_strength': abs(histogram) / max(abs(macd), abs(signal), 0.01)
                            })
                    
                    # Enhanced Bollinger Bands features
                    if all(col in latest_tech.index for col in ['BB_Upper', 'BB_Lower', 'Close']):
                        if all(pd.notna(latest_tech[col]) for col in ['BB_Upper', 'BB_Lower', 'Close']):
                            bb_upper = float(latest_tech['BB_Upper'])
                            bb_lower = float(latest_tech['BB_Lower'])
                            close = float(latest_tech['Close'])
                            bb_middle = (bb_upper + bb_lower) / 2
                            bb_width = bb_upper - bb_lower
                            
                            features.update({
                                'bb_position': (close - bb_lower) / bb_width if bb_width > 0 else 0.5,
                                'bb_squeeze': bb_width / bb_middle if bb_middle > 0 else 0,
                                'bb_upper_breach': float(close > bb_upper),
                                'bb_lower_breach': float(close < bb_lower),
                                'bb_middle_distance': (close - bb_middle) / bb_middle if bb_middle > 0 else 0,
                                'bb_expansion': float(bb_width > bb_middle * 0.1) if bb_middle > 0 else 0
                            })
                    
                    # Enhanced moving average features
                    sma_cols = ['SMA_20', 'SMA_50', 'SMA_200']
                    sma_values = {}
                    for sma in sma_cols:
                        if sma in latest_tech.index and pd.notna(latest_tech[sma]):
                            sma_values[sma] = float(latest_tech[sma])
                    
                    if 'Close' in latest_tech.index and pd.notna(latest_tech['Close']):
                        close = float(latest_tech['Close'])
                        for sma, value in sma_values.items():
                            features[f'price_vs_{sma.lower()}'] = (close - value) / value if value > 0 else 0
                    
                    # Moving average convergence/divergence
                    if len(sma_values) >= 2:
                        if 'SMA_20' in sma_values and 'SMA_50' in sma_values:
                            features['sma20_vs_sma50'] = (sma_values['SMA_20'] - sma_values['SMA_50']) / sma_values['SMA_50'] if sma_values['SMA_50'] > 0 else 0
                        if 'SMA_50' in sma_values and 'SMA_200' in sma_values:
                            features['sma50_vs_sma200'] = (sma_values['SMA_50'] - sma_values['SMA_200']) / sma_values['SMA_200'] if sma_values['SMA_200'] > 0 else 0
                    
                    # Market regime detection
                    if len(sma_values) >= 3:
                        sma_20 = sma_values.get('SMA_20', 0)
                        sma_50 = sma_values.get('SMA_50', 0)
                        sma_200 = sma_values.get('SMA_200', 0)
                        
                        features.update({
                            'bullish_alignment': float(sma_20 > sma_50 > sma_200),
                            'bearish_alignment': float(sma_20 < sma_50 < sma_200),
                            'mixed_signals': float(not (sma_20 > sma_50 > sma_200 or sma_20 < sma_50 < sma_200))
                        })
                    
                    # ATR-based features
                    if 'ATR' in latest_tech.index and pd.notna(latest_tech['ATR']):
                        atr = float(latest_tech['ATR'])
                        if 'Close' in latest_tech.index and pd.notna(latest_tech['Close']):
                            close = float(latest_tech['Close'])
                            features.update({
                                'atr_ratio': atr / close if close > 0 else 0,
                                'high_volatility': float(atr / close > 0.02) if close > 0 else 0,
                                'low_volatility': float(atr / close < 0.005) if close > 0 else 0
                            })
        
        except Exception as e:
            self.logger.warning(f"Error extracting enhanced technical features: {e}")
        
        return features
    
    def _extract_enhanced_sentiment_features(self, ticker: str, target_date: datetime) -> Dict[str, float]:
        """Enhanced sentiment feature extraction with Phase 3 integration"""
        features = {}
        
        try:
            # Get sentiment from Phase 3 connector
            if hasattr(self.news_connector, 'get_enhanced_sentiment_features'):
                sentiment_data = self.news_connector.get_enhanced_sentiment_features(ticker, target_date, lookback_days=7)
            else:
                sentiment_data = self.news_connector.get_enhanced_sentiment_features(ticker, target_date, lookback_days=7)
            
            # Basic sentiment features
            features.update({
                'sentiment_1d': float(sentiment_data.sentiment_1d),
                'sentiment_3d': float(sentiment_data.sentiment_3d),
                'sentiment_7d': float(sentiment_data.sentiment_7d),
                'news_volume_1d': float(sentiment_data.news_volume_1d),
                'news_volume_3d': float(sentiment_data.news_volume_3d),
                'news_volume_7d': float(sentiment_data.news_volume_7d),
                'correlation_strength': float(sentiment_data.correlation_strength),
                'event_impact_score': float(sentiment_data.event_impact_score),
                'confidence_score': float(sentiment_data.confidence_score),
                'alert_count': float(sentiment_data.alert_count),
                'source_diversity': float(sentiment_data.source_diversity)
            })
            
            # Enhanced derived sentiment features
            # Sentiment momentum and acceleration
            sentiment_values = [sentiment_data.sentiment_1d, sentiment_data.sentiment_3d, sentiment_data.sentiment_7d]
            if len(sentiment_values) >= 2:
                features['sentiment_momentum_1d'] = sentiment_values[0] - sentiment_values[1]
                features['sentiment_momentum_3d'] = sentiment_values[1] - sentiment_values[2]
                
                # Sentiment acceleration
                if len(sentiment_values) >= 3:
                    momentum_1d = sentiment_values[0] - sentiment_values[1]
                    momentum_3d = sentiment_values[1] - sentiment_values[2]
                    features['sentiment_acceleration'] = momentum_1d - momentum_3d
            
            # News volume analysis
            news_volumes = [sentiment_data.news_volume_1d, sentiment_data.news_volume_3d, sentiment_data.news_volume_7d]
            if news_volumes[2] > 0:
                features['news_volume_trend'] = (news_volumes[0] - news_volumes[2]) / news_volumes[2]
            else:
                features['news_volume_trend'] = 0.0
            
            # News quality metrics
            features.update({
                'sentiment_strength': abs(sentiment_data.sentiment_1d) * sentiment_data.confidence_score,
                'news_intensity': sentiment_data.news_volume_1d * sentiment_data.source_diversity,
                'high_confidence_sentiment': float(sentiment_data.confidence_score > 0.8),
                'low_confidence_sentiment': float(sentiment_data.confidence_score < 0.5),
                'news_spike': float(sentiment_data.news_volume_1d > sentiment_data.news_volume_7d * 2),
                'sentiment_extreme': float(abs(sentiment_data.sentiment_1d) > 0.8)
            })
            
            # Get market context
            if hasattr(self.news_connector, 'get_enhanced_market_overview'):
                market_data = self.news_connector.get_enhanced_market_overview(target_date)
                features.update({
                    'market_sentiment': float(market_data.get('market_sentiment', 0)),
                    'market_news_volume': float(market_data.get('total_articles', 0)),
                    'market_source_diversity': float(market_data.get('source_diversity', 0))
                })
                
                # Relative sentiment analysis
                market_sentiment = market_data.get('market_sentiment', 0)
                if market_sentiment != 0:
                    features['relative_sentiment'] = sentiment_data.sentiment_1d - market_sentiment
                    features['sentiment_leadership'] = float(abs(sentiment_data.sentiment_1d) > abs(market_sentiment))
                else:
                    features['relative_sentiment'] = sentiment_data.sentiment_1d
                    features['sentiment_leadership'] = 0.0
            
            # Sentiment consistency analysis
            sentiment_consistency = 0
            if all(s > 0 for s in sentiment_values):
                sentiment_consistency = 1.0  # Consistently bullish
            elif all(s < 0 for s in sentiment_values):
                sentiment_consistency = -1.0  # Consistently bearish
            else:
                sentiment_consistency = 0.0  # Mixed signals
            
            features.update({
                'sentiment_consistency': sentiment_consistency,
                'sentiment_reversal': float(sentiment_values[0] * sentiment_values[1] < 0),  # Sign change
                'sentiment_volatility': np.std(sentiment_values) if len(sentiment_values) > 1 else 0.0
            })
        
        except Exception as e:
            self.logger.warning(f"Error extracting enhanced sentiment features: {e}")
            # Set default values
            default_sentiment_features = {
                'sentiment_1d': 0.0, 'sentiment_3d': 0.0, 'sentiment_7d': 0.0,
                'news_volume_1d': 0.0, 'news_volume_3d': 0.0, 'news_volume_7d': 0.0,
                'correlation_strength': 0.5, 'event_impact_score': 0.0, 'confidence_score': 0.5,
                'alert_count': 0.0, 'source_diversity': 0.0, 'sentiment_momentum_1d': 0.0,
                'sentiment_momentum_3d': 0.0, 'sentiment_acceleration': 0.0, 'news_volume_trend': 0.0,
                'sentiment_strength': 0.0, 'news_intensity': 0.0, 'high_confidence_sentiment': 0.0,
                'low_confidence_sentiment': 1.0, 'news_spike': 0.0, 'sentiment_extreme': 0.0,
                'market_sentiment': 0.0, 'market_news_volume': 0.0, 'market_source_diversity': 0.0,
                'relative_sentiment': 0.0, 'sentiment_leadership': 0.0, 'sentiment_consistency': 0.0,
                'sentiment_reversal': 0.0, 'sentiment_volatility': 0.0
            }
            features.update(default_sentiment_features)
        
        return features
    
    def _extract_enhanced_time_features(self, target_date: datetime) -> Dict[str, float]:
        """Enhanced time-based feature extraction"""
        features = {}
        
        # Basic time features
        features.update({
            'day_of_week': float(target_date.weekday()),
            'month': float(target_date.month),
            'quarter': float((target_date.month - 1) // 3 + 1),
            'day_of_month': float(target_date.day),
            'day_of_year': float(target_date.timetuple().tm_yday),
            'week_of_year': float(target_date.isocalendar()[1]),
            'year': float(target_date.year)
        })
        
        # Enhanced cyclical encoding
        features.update({
            'dow_sin': float(np.sin(2 * np.pi * target_date.weekday() / 7)),
            'dow_cos': float(np.cos(2 * np.pi * target_date.weekday() / 7)),
            'month_sin': float(np.sin(2 * np.pi * (target_date.month - 1) / 12)),
            'month_cos': float(np.cos(2 * np.pi * (target_date.month - 1) / 12)),
            'day_sin': float(np.sin(2 * np.pi * target_date.day / 31)),
            'day_cos': float(np.cos(2 * np.pi * target_date.day / 31))
        })
        
        # Enhanced market calendar features
        features.update({
            'is_monday': float(target_date.weekday() == 0),
            'is_tuesday': float(target_date.weekday() == 1),
            'is_wednesday': float(target_date.weekday() == 2),
            'is_thursday': float(target_date.weekday() == 3),
            'is_friday': float(target_date.weekday() == 4),
            'is_weekend': float(target_date.weekday() >= 5),
            'is_month_start': float(target_date.day <= 5),
            'is_month_mid': float(10 <= target_date.day <= 20),
            'is_month_end': float(target_date.day >= 26),
            'is_quarter_start': float(target_date.month in [1, 4, 7, 10] and target_date.day <= 10),
            'is_quarter_end': float(target_date.month in [3, 6, 9, 12] and target_date.day >= 20),
            'is_year_start': float(target_date.month == 1 and target_date.day <= 10),
            'is_year_end': float(target_date.month == 12 and target_date.day >= 20)
        })
        
        # Enhanced earnings and market event features
        earnings_months = [1, 2, 4, 5, 7, 8, 10, 11]
        features.update({
            'is_earnings_season': float(target_date.month in earnings_months),
            'earnings_intensity': float(target_date.month in [1, 4, 7, 10]),  # Peak earnings months
            'summer_period': float(target_date.month in [6, 7, 8]),  # Summer trading
            'year_end_rally': float(target_date.month in [11, 12]),  # Year-end effects
            'january_effect': float(target_date.month == 1),
            'sell_in_may': float(target_date.month == 5),
            'september_effect': float(target_date.month == 9),  # Historically weak month
            'october_effect': float(target_date.month == 10)   # Historically volatile
        })
        
        # Enhanced holiday and market closure effects
        features.update({
            'near_new_year': float(target_date.month == 1 and target_date.day <= 3),
            'near_july_4th': float(target_date.month == 7 and abs(target_date.day - 4) <= 1),
            'near_thanksgiving': float(target_date.month == 11 and 22 <= target_date.day <= 29),
            'near_christmas': float(target_date.month == 12 and 20 <= target_date.day <= 31),
            'holiday_week': float(
                (target_date.month == 1 and target_date.day <= 7) or
                (target_date.month == 7 and abs(target_date.day - 4) <= 3) or
                (target_date.month == 11 and 22 <= target_date.day <= 29) or
                (target_date.month == 12 and target_date.day >= 20)
            )
        })
        
        # Market microstructure features
        features.update({
            'triple_witching': float(
                target_date.month in [3, 6, 9, 12] and 
                15 <= target_date.day <= 21 and 
                target_date.weekday() == 4  # Third Friday
            ),
            'opex_week': float(
                target_date.day >= 15 and target_date.day <= 21 and 
                target_date.weekday() == 4  # Third Friday of month
            ),
            'fomc_week': float(
                # Approximate FOMC meeting schedule (8 times per year)
                target_date.month in [1, 3, 5, 6, 7, 9, 11, 12] and 
                target_date.day >= 15 and target_date.day <= 21
            )
        })
        
        return features
    
    def extract_fundamental_features(self, ticker: str, target_date: datetime) -> Dict[str, float]:
        """Extract fundamental features from Summary and Company Info sheets"""
        features = {}
        
        try:
            # From Summary sheet
            features.update(self._extract_sheet_features('summary_data', 'summary', process_strings=True))
            
            # From Company Info sheet
            features.update(self._extract_sheet_features('company_info', 'fundamental'))
            
            # From Performance Metrics sheet
            features.update(self._extract_sheet_features('performance_data', 'performance'))
            
        except Exception as e:
            self.logger.warning(f"Error extracting fundamental features for {ticker}: {e}")
        
        return features

    def _extract_sheet_features(self, sheet_key: str, prefix: str, process_strings: bool = False) -> Dict[str, float]:
        """Helper method to extract features from a specific sheet"""
        features = {}
        
        if sheet_key not in self.excel_data or self.excel_data[sheet_key] is None:
            return features
        
        sheet_data = self.excel_data[sheet_key]
        if sheet_data.empty:
            return features
        
        try:
            # Use first row of data
            data_row = sheet_data.iloc[0]
            
            for col in sheet_data.columns:
                if pd.notna(data_row[col]):
                    feature_name = f'{prefix}_{col}'
                    
                    try:
                        # Handle different data types
                        if isinstance(data_row[col], (int, float)):
                            features[feature_name] = float(data_row[col])
                        elif isinstance(data_row[col], str) and process_strings:
                            # Clean and convert string values
                            cleaned_value = self._clean_string_value(data_row[col])
                            if cleaned_value is not None:
                                features[feature_name] = cleaned_value
                    except (ValueError, TypeError) as e:
                        self.logger.debug(f"Could not convert {col} value '{data_row[col]}' to float: {e}")
                        continue
        
        except Exception as e:
            self.logger.warning(f"Error processing sheet {sheet_key}: {e}")
        
        return features

    def _clean_string_value(self, value: str) -> float:
        """Clean string values and convert to float"""
        if not isinstance(value, str):
            return None
        
        # Remove common formatting characters
        clean_val = value.strip()
        
        # Handle percentage values
        is_percentage = clean_val.endswith('%')
        if is_percentage:
            clean_val = clean_val[:-1]
        
        # Remove currency symbols and commas
        clean_val = clean_val.replace('$', '').replace(',', '').replace(' ', '')
        
        # Handle negative values in parentheses
        if clean_val.startswith('(') and clean_val.endswith(')'):
            clean_val = '-' + clean_val[1:-1]
        
        # Validate that we have a numeric string
        if not clean_val.replace('.', '').replace('-', '').isdigit():
            return None
        
        try:
            result = float(clean_val)
            # Convert percentage to decimal
            if is_percentage:
                result = result / 100
            return result
        except ValueError:
            return None
    
    def _extract_quality_features(self, ticker: str, target_date: datetime) -> Dict[str, float]:
        """NEW: Extract data quality features"""
        features = {}
        
        try:
            # From Data Quality sheet
            if 'data_quality' in self.excel_data and self.excel_data['data_quality'] is not None:
                quality_data = self.excel_data['data_quality']
                if not quality_data.empty:
                    quality_row = quality_data.iloc[0]
                    for col in quality_data.columns:
                        if pd.notna(quality_row[col]) and isinstance(quality_row[col], (int, float)):
                            features[f'quality_{col}'] = float(quality_row[col])
            
            # From Metadata sheet
            if 'metadata' in self.excel_data and self.excel_data['metadata'] is not None:
                metadata = self.excel_data['metadata']
                if not metadata.empty:
                    metadata_row = metadata.iloc[0]
                    # Extract version info, analysis dates, etc.
                    if 'analysis_tool' in metadata_row.index:
                        features['analysis_tool_version'] = 1.0 if 'v1.0' in str(metadata_row['analysis_tool']) else 0.0
            
            # Calculate data freshness
            if 'raw_data' in self.excel_data and self.excel_data['raw_data'] is not None:
                raw_data = self.excel_data['raw_data']
                if not raw_data.empty:
                    latest_data_date = raw_data['Date'].max()
                    days_old = (target_date - latest_data_date).days
                    features.update({
                        'data_freshness_days': float(days_old),
                        'data_is_fresh': float(days_old <= 1),
                        'data_is_stale': float(days_old > 7)
                    })
        
        except Exception as e:
            self.logger.warning(f"Error extracting quality features: {e}")
        
        return features
    
    def _create_enhanced_derived_features(self, price_features: Dict, technical_features: Dict, 
                                        sentiment_features: Dict, time_features: Dict,
                                        fundamental_features: Dict, quality_features: Dict) -> Dict[str, float]:
        """Enhanced derived features with more sophisticated combinations"""
        derived = {}
        
        try:
            # Enhanced Price-Technical combinations
            if 'close' in price_features and 'tech_rsi' in technical_features:
                price_momentum = price_features.get('price_momentum_1d', 0)
                rsi = technical_features['tech_rsi']
                
                # RSI divergence detection
                if rsi > 70 and price_momentum < -0.01:
                    derived['rsi_bearish_divergence'] = 1.0
                elif rsi < 30 and price_momentum > 0.01:
                    derived['rsi_bullish_divergence'] = 1.0
                else:
                    derived['rsi_bearish_divergence'] = 0.0
                    derived['rsi_bullish_divergence'] = 0.0
                
                # RSI momentum alignment
                rsi_momentum = (rsi - 50) / 50  # Normalize RSI to -1 to 1
                derived['rsi_price_momentum_alignment'] = float(
                    (rsi_momentum > 0 and price_momentum > 0) or 
                    (rsi_momentum < 0 and price_momentum < 0)
                )
            
            # Enhanced Volume-Price combinations
            if 'volume_ratio' in price_features and 'daily_change_pct' in price_features:
                volume_ratio = price_features['volume_ratio']
                price_change = price_features['daily_change_pct']
                
                derived.update({
                    'volume_price_confirmation': volume_ratio * abs(price_change),
                    'volume_breakout': float(volume_ratio > 2.0 and abs(price_change) > 2.0),
                    'low_volume_drift': float(volume_ratio < 0.5 and abs(price_change) < 0.5),
                    'volume_surge_up': float(volume_ratio > 1.5 and price_change > 1.0),
                    'volume_surge_down': float(volume_ratio > 1.5 and price_change < -1.0)
                })
            
            # Enhanced Sentiment-Technical combinations
            if 'sentiment_1d' in sentiment_features and 'tech_rsi' in technical_features:
                sentiment = sentiment_features['sentiment_1d']
                rsi = technical_features['tech_rsi']
                
                # Advanced sentiment-technical alignment
                derived.update({
                    'sentiment_technical_alignment': float(
                        (sentiment > 0.1 and rsi > 55) or 
                        (sentiment < -0.1 and rsi < 45)
                    ),
                    'sentiment_technical_divergence': float(
                        (sentiment > 0.1 and rsi < 30) or 
                        (sentiment < -0.1 and rsi > 70)
                    ),
                    'sentiment_rsi_momentum': sentiment * (rsi - 50) / 50,
                    'extreme_sentiment_oversold': float(sentiment < -0.5 and rsi < 30),
                    'extreme_sentiment_overbought': float(sentiment > 0.5 and rsi > 70)
                })
            
            # Multi-timeframe sentiment analysis
            sentiments = [sentiment_features.get(f'sentiment_{period}', 0) for period in ['1d', '3d', '7d']]
            sentiment_trend = np.polyfit(range(len(sentiments)), sentiments, 1)[0] if len(sentiments) > 1 else 0
            
            derived.update({
                'sentiment_trend_slope': float(sentiment_trend),
                'sentiment_improving': float(sentiment_trend > 0.1),
                'sentiment_deteriorating': float(sentiment_trend < -0.1),
                'sentiment_stability': float(np.std(sentiments)) if len(sentiments) > 1 else 0,
                'sentiment_range': float(max(sentiments) - min(sentiments))
            })
            
            # News volume and price volatility correlation
            if 'news_volume_1d' in sentiment_features and 'volatility_5d' in price_features:
                news_vol = sentiment_features['news_volume_1d']
                price_vol = price_features['volatility_5d']
                
                derived.update({
                    'news_volatility_ratio': news_vol / max(price_vol, 0.1),
                    'high_news_low_vol': float(news_vol > 5 and price_vol < 1.0),
                    'news_volatility_spike': float(news_vol > 10 and price_vol > 3.0),
                    'quiet_news_period': float(news_vol < 2 and price_vol < 0.5)
                })
            
            # Enhanced time-based market effects
            if 'is_friday' in time_features and 'sentiment_1d' in sentiment_features:
                # Weekend effect and sentiment interaction
                derived.update({
                    'friday_sentiment_factor': sentiment_features['sentiment_1d'] * (1.2 if time_features['is_friday'] else 1.0),
                    'monday_sentiment_factor': sentiment_features['sentiment_1d'] * (1.1 if time_features.get('is_monday', 0) else 1.0),
                    'earnings_season_sentiment': sentiment_features['sentiment_1d'] * (1.3 if time_features.get('is_earnings_season', 0) else 1.0)
                })
            
            # Advanced market regime detection
            if all(f'tech_{sma}' in technical_features for sma in ['sma_20', 'sma_50', 'sma_200']):
                sma20 = technical_features['tech_sma_20']
                sma50 = technical_features['tech_sma_50']
                sma200 = technical_features['tech_sma_200']
                
                derived.update({
                    'strong_uptrend': float(sma20 > sma50 > sma200 and sma20 > sma200 * 1.05),
                    'strong_downtrend': float(sma20 < sma50 < sma200 and sma20 < sma200 * 0.95),
                    'trend_strength': abs(sma20 - sma200) / sma200 if sma200 > 0 else 0,
                    'ma_convergence': float(abs(sma20 - sma50) / sma50 < 0.01) if sma50 > 0 else 0,
                    'golden_cross': float(sma50 > sma200 and sma20 > sma50),
                    'death_cross': float(sma50 < sma200 and sma20 < sma50)
                })
            
            # Quality-adjusted confidence features
            if 'confidence_score' in sentiment_features and 'quality_data_completeness' in quality_features:
                confidence = sentiment_features['confidence_score']
                quality = quality_features['quality_data_completeness'] / 100.0  # Convert percentage
                
                derived.update({
                    'quality_adjusted_confidence': confidence * quality,
                    'high_quality_high_confidence': float(confidence > 0.8 and quality > 0.95),
                    'reliability_score': (confidence + quality) / 2.0
                })
            
            # Fundamental-Technical combinations
            if 'fundamental_market_cap' in fundamental_features and 'volume_ratio' in price_features:
                # Large cap vs small cap volume sensitivity
                market_cap = fundamental_features['fundamental_market_cap']
                volume_ratio = price_features['volume_ratio']
                
                # Rough market cap classification (this would need real market cap data)
                large_cap_threshold = 10_000_000_000  # $10B
                is_large_cap = float(market_cap > large_cap_threshold)
                
                derived.update({
                    'large_cap_volume_sensitivity': is_large_cap * volume_ratio,
                    'small_cap_volume_sensitivity': (1 - is_large_cap) * volume_ratio,
                    'market_cap_volume_interaction': market_cap * volume_ratio / 1_000_000_000  # Scale down
                })
            
            # Performance-based features
            if 'performance_sharpe_ratio' in fundamental_features:
                sharpe = fundamental_features['performance_sharpe_ratio']
                
                derived.update({
                    'high_sharpe_momentum': float(sharpe > 1.0) * price_features.get('price_momentum_1d', 0),
                    'risk_adjusted_return_signal': sharpe * sentiment_features.get('sentiment_1d', 0),
                    'quality_performance_score': sharpe * quality_features.get('quality_data_completeness', 95) / 100.0
                })
            
            # Comprehensive market stress indicator
            volatility = price_features.get('volatility_5d', 1.0)
            volume_ratio = price_features.get('volume_ratio', 1.0)
            sentiment_vol = sentiment_features.get('sentiment_volatility', 0.0)
            
            derived['market_stress_indicator'] = float(
                (volatility > 3.0) + 
                (volume_ratio > 2.0) + 
                (sentiment_vol > 0.5) + 
                (abs(sentiment_features.get('sentiment_1d', 0)) > 0.7)
            ) / 4.0
            
            # Momentum convergence indicator
            price_momentum = price_features.get('price_momentum_1d', 0)
            sentiment_momentum = sentiment_features.get('sentiment_momentum_1d', 0)
            
            derived.update({
                'momentum_convergence': float(
                    (price_momentum > 0 and sentiment_momentum > 0) or 
                    (price_momentum < 0 and sentiment_momentum < 0)
                ),
                'momentum_divergence': float(
                    (price_momentum > 0 and sentiment_momentum < 0) or 
                    (price_momentum < 0 and sentiment_momentum > 0)
                ),
                'combined_momentum_strength': abs(price_momentum) + abs(sentiment_momentum)
            })
        
        except Exception as e:
            self.logger.warning(f"Error creating enhanced derived features: {e}")
        
        return derived
    
    def _extract_target_variable(self, ticker: str, target_date: datetime, offset_days: int = 1) -> Optional[float]:
        """Extract target variable with enhanced logic"""
        target_date = pd.to_datetime(target_date)
        
        try:
            if 'raw_data' in self.excel_data and self.excel_data['raw_data'] is not None:
                raw_data = self.excel_data['raw_data']
                
                # Get current price
                current_mask = raw_data['Date'] <= target_date
                current_data = raw_data[current_mask]
                
                if current_data.empty:
                    return None
                
                current_price = current_data.iloc[-1]['Close']
                
                # Get future price with flexible offset
                future_dates = []
                for i in range(offset_days, offset_days + 5):  # Try up to 5 days ahead
                    future_date = target_date + timedelta(days=i)
                    future_mask = raw_data['Date'] >= future_date
                    future_data = raw_data[future_mask]
                    
                    if not future_data.empty:
                        future_price = future_data.iloc[0]['Close']
                        return float((future_price - current_price) / current_price)
                
                return None
        
        except Exception as e:
            self.logger.warning(f"Error extracting target variable: {e}")
        
        return None
    
    def create_comprehensive_training_dataset(self, ticker: str, start_date: datetime, end_date: datetime, 
                                           target_offset: int = 1, min_samples: int = 50) -> Tuple[pd.DataFrame, pd.Series]:
        """Create comprehensive training dataset with enhanced features"""
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        self.logger.info(f"Creating comprehensive training dataset for {ticker}")
        self.logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        
        all_features = []
        all_targets = []
        all_dates = []
        
        # Generate business days only
        business_days = pd.bdate_range(start_date, end_date)
        
        for current_date in business_days:
            try:
                # Create enhanced features for this date
                ml_features = self.create_enhanced_features_for_date(
                    ticker, current_date, include_target=True, target_offset=target_offset
                )
                
                if ml_features.target is not None:
                    # Flatten all features into single dictionary
                    feature_dict = {}
                    feature_dict.update(ml_features.price_features)
                    feature_dict.update(ml_features.technical_features)
                    feature_dict.update(ml_features.sentiment_features)
                    feature_dict.update(ml_features.time_features)
                    feature_dict.update(ml_features.fundamental_features)
                    feature_dict.update(ml_features.quality_features)
                    feature_dict.update(ml_features.derived_features)
                    
                    all_features.append(feature_dict)
                    all_targets.append(ml_features.target)
                    all_dates.append(current_date)
                
            except Exception as e:
                self.logger.warning(f"Error processing date {current_date.date()}: {e}")
                continue
        
        if len(all_features) < min_samples:
            raise ValueError(f"Insufficient samples: {len(all_features)} < {min_samples}")
        
        # Convert to DataFrames
        features_df = pd.DataFrame(all_features, index=all_dates)
        targets = pd.Series(all_targets, index=all_dates, name='target')
        
        # Enhanced data cleaning
        features_df = self._clean_features_dataframe(features_df)
        
        self.logger.info(f"Created comprehensive training dataset:")
        self.logger.info(f"  Samples: {len(features_df)}")
        self.logger.info(f"  Features: {len(features_df.columns)}")
        self.logger.info(f"  Target range: {targets.min():.4f} to {targets.max():.4f}")
        self.logger.info(f"  Feature categories: Price({len([c for c in features_df.columns if c.startswith('price') or c in ['open', 'high', 'low', 'close', 'volume']])}), Technical({len([c for c in features_df.columns if c.startswith('tech_')])}), Sentiment({len([c for c in features_df.columns if c.startswith('sentiment')])})")
        
        return features_df, targets
    
    def _clean_features_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced feature dataframe cleaning"""
        # Fill missing values with appropriate defaults
        df = df.fillna(0)
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], 0)
        
        # Remove constant features
        constant_features = [col for col in df.columns if df[col].std() == 0]
        if constant_features:
            self.logger.info(f"Removing {len(constant_features)} constant features")
            df = df.drop(columns=constant_features)
        
        # Remove highly correlated features (optional)
        correlation_threshold = 0.95
        corr_matrix = df.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        high_corr_features = []
        for column in upper_triangle.columns:
            if any(upper_triangle[column] > correlation_threshold):
                high_corr_features.append(column)
        
        if high_corr_features:
            self.logger.info(f"Removing {len(high_corr_features)} highly correlated features")
            df = df.drop(columns=high_corr_features)
        
        return df
    
    def get_feature_importance_analysis(self, features_df: pd.DataFrame, targets: pd.Series) -> Dict:
        """Analyze feature importance using multiple methods"""
        try:
            from sklearn.feature_selection import mutual_info_regression
            from sklearn.ensemble import RandomForestRegressor
            
            # Mutual information
            mi_scores = mutual_info_regression(features_df, targets)
            mi_importance = dict(zip(features_df.columns, mi_scores))
            
            # Random forest feature importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(features_df, targets)
            rf_importance = dict(zip(features_df.columns, rf.feature_importances_))
            
            # Correlation with target
            corr_importance = features_df.corrwith(targets).abs().to_dict()
            
            return {
                'mutual_info': mi_importance,
                'random_forest': rf_importance,
                'correlation': corr_importance
            }
            
        except ImportError:
            self.logger.warning("sklearn not available for feature importance analysis")
            return {}
    
    def validate_enhanced_features(self, features: MLFeatures) -> Dict[str, any]:
        """Enhanced feature validation"""
        validation = {
            'valid': True,
            'warnings': [],
            'feature_counts': {
                'price': len(features.price_features),
                'technical': len(features.technical_features),
                'sentiment': len(features.sentiment_features),
                'time': len(features.time_features),
                'fundamental': len(features.fundamental_features),
                'quality': len(features.quality_features),
                'derived': len(features.derived_features)
            },
            'quality_score': 0.0,
            'completeness_score': 0.0
        }
        
        # Enhanced validation criteria
        min_requirements = {
            'price': 10, 'technical': 5, 'sentiment': 8, 'time': 5, 'fundamental': 2, 'quality': 1, 'derived': 5
        }
        
        quality_score = 0.0
        total_categories = len(min_requirements)
        
        for category, min_count in min_requirements.items():
            actual_count = validation['feature_counts'][category]
            category_score = min(actual_count / min_count, 1.0)
            quality_score += category_score / total_categories
            
            if actual_count < min_count:
                validation['warnings'].append(f"Low {category} feature count: {actual_count} < {min_count}")
        
        # Check for invalid values across all feature categories
        all_features = [features.price_features, features.technical_features, 
                       features.sentiment_features, features.time_features,
                       features.fundamental_features, features.quality_features, features.derived_features]
        
        total_features = sum(len(f) for f in all_features)
        invalid_count = 0
        
        for feature_dict in all_features:
            for value in feature_dict.values():
                if not np.isfinite(value):
                    invalid_count += 1
        
        if invalid_count > 0:
            validation['warnings'].append(f"Found {invalid_count} invalid feature values out of {total_features}")
            validation['valid'] = False
        
        # Calculate completeness score
        expected_total_features = sum(min_requirements.values())
        actual_total_features = sum(validation['feature_counts'].values())
        validation['completeness_score'] = min(actual_total_features / expected_total_features, 1.0)
        
        validation['quality_score'] = quality_score
        
        # Additional quality checks
        if features.target is not None and not np.isfinite(features.target):
            validation['warnings'].append("Target variable is invalid")
            validation['valid'] = False
        
        # Check confidence scores
        confidence = features.sentiment_features.get('confidence_score', 0.5)
        if confidence < 0.3:
            validation['warnings'].append(f"Low sentiment confidence: {confidence:.2f}")
        
        # Check data freshness
        data_age = features.quality_features.get('data_freshness_days', 0)
        if data_age > 7:
            validation['warnings'].append(f"Stale data: {data_age} days old")
        
        return validation


def test_enhanced_pipeline():
    """Test the enhanced feature engineering pipeline"""
    print("🚀 Testing Enhanced Feature Engineering Pipeline...")
    print("=" * 70)
    
    try:
        engineer = FeatureEngineer()
        
        # Test single date feature extraction
        test_date = datetime.now() - timedelta(days=5)
        test_ticker = "MSFT"
        
        print(f"\n🔍 Testing enhanced feature extraction for {test_ticker} on {test_date.date()}...")
        
        features = engineer.create_enhanced_features_for_date(test_ticker, test_date, include_target=True)
        
        print(f"✅ Enhanced feature extraction successful!")
        print(f"   📊 Feature Breakdown:")
        print(f"      Price features: {len(features.price_features)}")
        print(f"      Technical features: {len(features.technical_features)}")
        print(f"      Sentiment features: {len(features.sentiment_features)}")
        print(f"      Time features: {len(features.time_features)}")
        print(f"      Fundamental features: {len(features.fundamental_features)}")
        print(f"      Quality features: {len(features.quality_features)}")
        print(f"      Derived features: {len(features.derived_features)}")
        
        total_features = (len(features.price_features) + len(features.technical_features) + 
                         len(features.sentiment_features) + len(features.time_features) +
                         len(features.fundamental_features) + len(features.quality_features) + 
                         len(features.derived_features))
        
        print(f"      🎯 Total features: {total_features}")
        print(f"      🎯 Target: {features.target:.4f}" if features.target else "      🎯 Target: None")
        
        # Show sample features from each category
        print(f"\n📋 Sample Features by Category:")
        
        categories = [
            ("Price", features.price_features),
            ("Technical", features.technical_features),
            ("Sentiment", features.sentiment_features),
            ("Time", features.time_features),
            ("Fundamental", features.fundamental_features),
            ("Quality", features.quality_features),
            ("Derived", features.derived_features)
        ]
        
        for cat_name, cat_features in categories:
            if cat_features:
                sample_items = list(cat_features.items())[:3]
                print(f"   📈 {cat_name}: {sample_items}")
        
        # Enhanced validation
        validation = engineer.validate_enhanced_features(features)
        print(f"\n✅ Enhanced Feature Validation:")
        print(f"   Valid: {validation['valid']}")
        print(f"   Quality Score: {validation['quality_score']:.2f}")
        print(f"   Completeness Score: {validation['completeness_score']:.2f}")
        if validation['warnings']:
            print(f"   ⚠️ Warnings: {validation['warnings']}")
        
        # Test training dataset creation (small sample)
        print(f"\n📈 Testing enhanced training dataset creation...")
        start_date = test_date - timedelta(days=20)
        end_date = test_date
        
        X, y = engineer.create_comprehensive_training_dataset(test_ticker, start_date, end_date, min_samples=5)
        
        print(f"✅ Enhanced training dataset created:")
        print(f"   📊 Samples: {len(X)}")
        print(f"   📊 Features: {len(X.columns)}")
        print(f"   📊 Target range: {y.min():.4f} to {y.max():.4f}")
        print(f"   📊 Target std: {y.std():.4f}")
        
        # Feature importance analysis
        print(f"\n🔍 Feature Importance Analysis...")
        try:
            importance_analysis = engineer.get_feature_importance_analysis(X, y)
            if importance_analysis:
                # Show top 5 features by correlation
                if 'correlation' in importance_analysis:
                    corr_sorted = sorted(importance_analysis['correlation'].items(), 
                                       key=lambda x: abs(x[1]), reverse=True)
                    print(f"   📊 Top 5 features by correlation:")
                    for name, corr in corr_sorted[:5]:
                        print(f"      {name}: {corr:.3f}")
            else:
                print(f"   ⚠️ Feature importance analysis not available (sklearn not installed)")
        except Exception as e:
            print(f"   ⚠️ Feature importance analysis failed: {e}")
        
        # Data quality summary
        print(f"\n📊 Data Quality Summary:")
        print(f"   Missing values: {X.isnull().sum().sum()}")
        print(f"   Infinite values: {np.isinf(X).sum().sum()}")
        print(f"   Constant features: {(X.std() == 0).sum()}")
        print(f"   Feature correlation range: {X.corr().abs().values.max():.3f}")
        
        print(f"\n🎉 Enhanced feature engineering pipeline test completed!")
        print(f"✅ Successfully created {total_features} features from ALL data sources!")
        print(f"📈 Ready for advanced ML model training!")
        
    except Exception as e:
        print(f"\n❌ Error testing enhanced pipeline: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function"""
    test_enhanced_pipeline()


if __name__ == "__main__":
    main()