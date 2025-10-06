#!/usr/bin/env python3
"""
[IN] Indian Stock Market Trading Signal Bot
Monitors NSE stocks and sends intelligent trade signals via Telegram
Uses sophisticated options analysis adapted for Indian markets with Zerodha Kite Connect API
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import time
import logging
from datetime import datetime, timedelta, time as dt_time
import pytz
import asyncio
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
import warnings
import sqlite3
import shutil

logger = logging.getLogger(__name__)

# Import your existing analyzer
from options_analyzer.analyzers.options_analyzer import ZerodhaEnhancedOptionsAnalyzer

# Import new Indian market components with Zerodha API
from options_analyzer.brokers.zerodha_api_client import ZerodhaAPIClient  # Changed from GrowwAPIClient
from options_analyzer.indian_market.ind_data_processor import IndianMarketProcessor
from api.telegram.telegram_bot import TelegramSignalBot
from options_analyzer.indian_market.ind_trade_logger import IndianTradeLogger

warnings.filterwarnings('ignore')

# Load environment variables from project root
dotenv_path = project_root / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Configure logging with rotation
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            'indian_trading_bot.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BotConfig:
    """Bot configuration"""
    # API Keys - Updated for Zerodha
    ZERODHA_API_KEY: str = None
    ZERODHA_API_SECRET: str = None
    ZERODHA_ACCESS_TOKEN: str = None
    TELEGRAM_BOT_TOKEN: str = None
    TELEGRAM_CHAT_ID: str = None
    CLAUDE_API_KEY: str = None
    
    # Market settings
    INDIAN_WATCHLIST: List[str] = None
    SCAN_INTERVAL_MINUTES: int = 5
    ACCOUNT_SIZE: float = 100000  # ₹1 lakh default
    RISK_TOLERANCE: str = 'medium'
    
    # Trading hours (IST)
    MARKET_OPEN: dt_time = dt_time(9, 15)
    MARKET_CLOSE: dt_time = dt_time(15, 30)
    PRE_MARKET_START: dt_time = dt_time(9, 0)
    POST_MARKET_END: dt_time = dt_time(16, 0)
    
    # Thresholds and Constants
    MIN_CONFIDENCE_SCORE: float = 0.30  # 30% minimum
    MEDIUM_CONFIDENCE_THRESHOLD: float = 0.70  # 70% for medium confidence
    HIGH_CONFIDENCE_THRESHOLD: float = 0.80  # 80% for high confidence
    MAX_SIGNALS_PER_DAY: int = 20
    MAX_POSITIONS_OPEN: int = 5

    # Quality Score Thresholds
    MINIMUM_QUALITY_SCORE: float = 0.60
    MIN_VOLUME: int = 50000
    MAX_SPREAD_PERCENT: float = 0.10  # 10% max spread

    # Account Limits
    MIN_ACCOUNT_SIZE: float = 10000  # Minimum ₹10,000
    MAX_ACCOUNT_SIZE: float = 100000000  # Maximum ₹10 crore

    # Scan Interval Limits
    MIN_SCAN_INTERVAL: int = 1  # 1 minute minimum
    MAX_SCAN_INTERVAL: int = 60  # 60 minutes maximum

    # Rate Limiting
    ZERODHA_CALLS_PER_SECOND: int = 3
    ZERODHA_HISTORICAL_CALLS_PER_SECOND: int = 2
    
class DatabaseSchemaManager:
    """Handles database schema migration for automation integration"""
    
    def __init__(self, db_path: str = "indian_trading_bot.db"):
        self.db_path = db_path
    
    def ensure_automation_schema(self) -> bool:
        """Ensure database has automation-required columns with proper resource management"""

        if not os.path.exists(self.db_path):
            logger.info("Database doesn't exist yet - will be created with correct schema")
            return True

        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()

            # Check if automation columns exist
            cursor.execute("PRAGMA table_info(signals)")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]

            # Required automation columns
            automation_columns = {
                'processed_by_automation': 'BOOLEAN DEFAULT 0',
                'automation_timestamp': 'DATETIME',
                'source': 'TEXT DEFAULT "signal_generator"'
            }

            migration_needed = False

            # Add missing columns
            for col_name, col_def in automation_columns.items():
                if col_name not in column_names:
                    try:
                        logger.info(f"Adding missing column: {col_name}")
                        cursor.execute(f"ALTER TABLE signals ADD COLUMN {col_name} {col_def}")
                        migration_needed = True
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" not in str(e).lower():
                            logger.error(f"Error adding column {col_name}: {e}")
                            return False

            if migration_needed:
                conn.commit()
                logger.info("[OK] Database schema updated for automation integration")
            else:
                logger.info("[OK] Database schema is already up-to-date")

            return True

        except sqlite3.Error as e:
            logger.error(f"SQLite error during schema check: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.exception(f"Unexpected error in database schema check: {e}")
            return False
        finally:
            if conn:
                try:
                    conn.close()
                except Exception as e:
                    logger.error(f"Error closing database connection: {e}")


class SmartSignalFilter:
    """Intelligent signal filtering to prevent spam and ensure quality"""
    
    def __init__(self):
        self.sent_signals = {}  # Track sent signals
        self.signal_history = []  # Keep history
        self.cooldown_periods = {
            'same_symbol_same_strategy': 60,  # 1 hour
            'same_symbol_different_strategy': 30,  # 30 minutes  
            'similar_strikes': 45,  # 45 minutes
            'same_direction': 20   # 20 minutes for same market direction
        }
        
    def should_send_signal(self, ticker: str, analysis_result: Dict) -> Tuple[bool, str]:
        """Determine if signal should be sent with reasoning"""
        
        current_time = datetime.now()
        trade_rec = analysis_result.get('trade_recommendation', {})
        
        # Extract signal characteristics
        strategy = trade_rec.get('strategy', '')
        confidence = trade_rec.get('confidence', 0)
        option_legs = trade_rec.get('option_legs', [])
        
        # Get strikes for comparison
        strikes = [leg.get('strike', 0) for leg in option_legs]
        avg_strike = sum(strikes) / len(strikes) if strikes else 0
        
        signal_key = f"{ticker}_{strategy}"
        
        # 1. Check for duplicate signals (exact same)
        if signal_key in self.sent_signals:
            last_sent = self.sent_signals[signal_key]['timestamp']
            time_diff = (current_time - last_sent).total_seconds() / 60
            
            if time_diff < self.cooldown_periods['same_symbol_same_strategy']:
                return False, f"Same strategy sent {time_diff:.0f}min ago (cooldown: {self.cooldown_periods['same_symbol_same_strategy']}min)"
        
        # 2. Check for similar signals on same symbol
        recent_signals = [s for s in self.signal_history 
                         if s['ticker'] == ticker and 
                         (current_time - s['timestamp']).total_seconds() < 1800]  # 30 min
        
        if recent_signals:
            # Check for similar strikes (within 100 points for indices, 50 for stocks)
            strike_tolerance = 100 if ticker in ['NIFTY', 'BANKNIFTY'] else 50
            
            for recent in recent_signals:
                recent_avg_strike = recent.get('avg_strike', 0)
                if abs(avg_strike - recent_avg_strike) < strike_tolerance:
                    time_diff = (current_time - recent['timestamp']).total_seconds() / 60
                    if time_diff < self.cooldown_periods['similar_strikes']:
                        return False, f"Similar strike signal sent {time_diff:.0f}min ago"
        
        # 3. Quality thresholds based on market conditions
        min_confidence = self._get_dynamic_confidence_threshold(ticker, current_time)
        
        if confidence < min_confidence:
            return False, f"Confidence {confidence:.0%} below dynamic threshold {min_confidence:.0%}"
        
        # 4. Check market direction spam (too many bullish/bearish signals)
        market_direction = self._determine_market_direction(strategy)
        if market_direction:
            same_direction_count = sum(1 for s in self.signal_history[-10:] 
                                     if self._determine_market_direction(s.get('strategy', '')) == market_direction)
            
            if same_direction_count >= 3:
                return False, f"Too many {market_direction} signals recently ({same_direction_count}/10)"
        
        # 5. Check for strategy diversity (avoid over-concentration)
        recent_strategies = [s.get('strategy', '') for s in self.signal_history[-5:]]
        strategy_count = recent_strategies.count(strategy)
        
        if strategy_count >= 2:
            return False, f"Strategy {strategy} used {strategy_count} times in last 5 signals"
        
        # 6. Time-based quality enhancement
        quality_score = self._calculate_signal_quality(analysis_result, current_time)
        
        if quality_score < 0.60:  # LOWERED from 0.70 to 0.60
            return False, f"Signal quality score {quality_score:.2f} below threshold 0.60"
        
        # 7. Market volatility check
        if not self._is_good_market_condition(ticker, analysis_result):
            return False, "Poor market conditions for trading"
        
        return True, "Signal approved - meets all quality criteria"
    
    def record_signal(self, ticker: str, analysis_result: Dict):
        """Record sent signal for tracking"""
        
        current_time = datetime.now()
        trade_rec = analysis_result.get('trade_recommendation', {})
        
        strategy = trade_rec.get('strategy', '')
        confidence = trade_rec.get('confidence', 0)
        option_legs = trade_rec.get('option_legs', [])
        strikes = [leg.get('strike', 0) for leg in option_legs]
        avg_strike = sum(strikes) / len(strikes) if strikes else 0
        
        signal_key = f"{ticker}_{strategy}"
        
        # Update sent signals tracking
        self.sent_signals[signal_key] = {
            'timestamp': current_time,
            'confidence': confidence,
            'strikes': strikes
        }
        
        # Add to history
        signal_record = {
            'ticker': ticker,
            'strategy': strategy,
            'confidence': confidence,
            'avg_strike': avg_strike,
            'timestamp': current_time,
            'trading_style': analysis_result.get('trade_type', 'SWING')
        }
        
        self.signal_history.append(signal_record)
        
        # Keep only last 50 signals in memory
        if len(self.signal_history) > 50:
            self.signal_history = self.signal_history[-50:]
        
        # Clean up old sent_signals (older than 4 hours)
        cutoff_time = current_time - timedelta(hours=4)
        self.sent_signals = {k: v for k, v in self.sent_signals.items() 
                           if v['timestamp'] > cutoff_time}
    
    def _get_dynamic_confidence_threshold(self, ticker: str, current_time: datetime) -> float:
        """Dynamic confidence threshold based on market conditions"""
        
        base_threshold = 0.60  # LOWERED from 0.65 to 0.60
        
        # Higher threshold during volatile times
        market_hour = current_time.hour
        
        # Opening hour (9-10 AM) - higher threshold due to volatility
        if 9 <= market_hour < 10:
            base_threshold += 0.05  # REDUCED from 0.10
        
        # Lunch time (12-2 PM) - lower liquidity, higher threshold
        elif 12 <= market_hour < 14:
            base_threshold += 0.03  # REDUCED from 0.05
        
        # Power hour (2:30-3:30 PM) - high volatility, higher threshold
        elif market_hour >= 14 and current_time.minute >= 30:
            base_threshold += 0.05  # REDUCED from 0.10
        
        # Index vs stock adjustment
        if ticker in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
            base_threshold -= 0.05  # Slightly lower for liquid indices
        
        return min(0.75, base_threshold)  # REDUCED cap from 0.85 to 0.75
    
    def _determine_market_direction(self, strategy: str) -> Optional[str]:
        """Determine market direction from strategy"""
        
        bullish_strategies = [
            'BULLISH_CALL', 'BULL_CALL_SPREAD', 'BULLISH_PUT',
            'INTRADAY_LONG_ATM_CALL'
        ]
        
        bearish_strategies = [
            'BEARISH_PUT', 'BEAR_PUT_SPREAD', 'BEAR_CALL_SPREAD',
            'INTRADAY_LONG_ATM_PUT'
        ]
        
        if any(bull in strategy for bull in bullish_strategies):
            return 'bullish'
        elif any(bear in strategy for bear in bearish_strategies):
            return 'bearish'
        
        return None  # Neutral strategies
    
    def _calculate_signal_quality(self, analysis_result: Dict, current_time: datetime) -> float:
        """Calculate overall signal quality score - ENHANCED FOR NEW ANALYZERS"""
        
        score = 0.0
        debug_info = []
        
        # Extract from enhanced structure
        trade_rec = analysis_result.get('trade_recommendation', {})
        
        # Confidence component (30% weight) - REDUCED to make room for enhanced features
        confidence = trade_rec.get('confidence', 0)
        confidence_score = confidence * 0.3
        score += confidence_score
        debug_info.append(f"Confidence: {confidence:.1%} -> {confidence_score:.2f}")
        
        # Enhanced Technical Analysis (35% weight) - MAJOR ENHANCEMENT
        technical = analysis_result.get('technical_analysis', {})
        technical_score = 0
        if technical:
            # Market bias and confidence (15% weight)
            market_bias = technical.get('market_bias', 'NEUTRAL')
            tech_confidence = technical.get('confidence_score', 0)
            
            bias_score = 0.10 if market_bias in ['BULLISH', 'BEARISH'] else 0.03
            conf_score = tech_confidence * 0.05
            
            # NEW: Today's context for intraday (10% weight)
            today_context = technical.get('today_context', {})
            intraday_score = 0
            if today_context:
                gap_type = today_context.get('gap_type', 'NONE')
                intraday_momentum = today_context.get('intraday_momentum', 'NEUTRAL')
                volume_ratio = today_context.get('volume_ratio', 1.0)
                
                # Strong gap with momentum continuation
                if gap_type != 'NONE' and intraday_momentum != 'NEUTRAL':
                    intraday_score += 0.05
                
                # High volume confirmation
                if volume_ratio > 1.5:
                    intraday_score += 0.03
                
                # Wide range day (high volatility)
                high_low_range = today_context.get('high_low_range_percent', 0)
                if high_low_range > 2:
                    intraday_score += 0.02
            
            # NEW: Entry signal quality (10% weight)
            entry_signal = technical.get('entry_signal', {})
            entry_score = 0
            if entry_signal:
                signal_type = entry_signal.get('signal_type', 'HOLD')
                signal_strength = entry_signal.get('strength', 0)
                
                # Strong directional signals get higher scores
                if signal_type in ['BUY', 'SELL']:
                    entry_score = signal_strength * 0.10
                elif signal_type == 'WAIT':
                    entry_score = 0.02  # Some credit for clear wait signal
            
            technical_score = bias_score + conf_score + intraday_score + entry_score
            score += technical_score
            
        debug_info.append(f"Technical: {technical_score:.2f} (bias: {bias_score:.2f}, conf: {conf_score:.2f}, intraday: {intraday_score:.2f}, entry: {entry_score:.2f})")
        
        # Options analysis quality (20% weight) - ENHANCED
        options_score = 0
        options_analysis = analysis_result.get('options_analysis', {})
        if options_analysis:
            current_iv = options_analysis.get('current_iv', 0.20)
            volatility = options_analysis.get('volatility', 25)
            put_call_ratio = options_analysis.get('put_call_ratio', 1.0)
            
            # Prefer extreme IV levels (8% weight)
            if current_iv < 0.15:
                iv_score = 0.08  # Low IV = cheap options
            elif current_iv > 0.35:
                iv_score = 0.06  # High IV = premium selling opportunities
            elif current_iv > 0.25:
                iv_score = 0.04  # Moderate high IV
            else:
                iv_score = 0.02  # Normal IV
            
            # Volatility regime scoring (8% weight)
            if 15 <= volatility <= 25:
                vol_score = 0.08  # Sweet spot for options
            elif volatility > 30:
                vol_score = 0.06  # High vol = good for straddles
            elif volatility < 12:
                vol_score = 0.04  # Low vol = breakout potential
            else:
                vol_score = 0.02
            
            # Put-call ratio insight (4% weight)
            pcr_score = 0
            if 0.7 <= put_call_ratio <= 1.3:
                pcr_score = 0.04  # Balanced market
            elif put_call_ratio > 1.5:
                pcr_score = 0.03  # Bearish extreme
            elif put_call_ratio < 0.5:
                pcr_score = 0.03  # Bullish extreme
            
            options_score = iv_score + vol_score + pcr_score
            score += options_score
            
        debug_info.append(f"Options: {options_score:.2f} (IV: {iv_score:.2f}, Vol: {vol_score:.2f}, PCR: {pcr_score:.2f})")
        
        # Zerodha integration quality (10% weight) - ENHANCED
        zerodha_integration = analysis_result.get('zerodha_integration', {})
        integration_score = 0
        if zerodha_integration:
            # Live data availability (5% weight)
            if zerodha_integration.get('live_data_available', False):
                integration_score += 0.05
            
            # Technical analysis availability (3% weight)
            if zerodha_integration.get('technical_analysis_available', False):
                integration_score += 0.03
            
            # Account connection status (2% weight)
            if zerodha_integration.get('account_connected', False):
                integration_score += 0.02
            
            score += integration_score
            
        debug_info.append(f"Zerodha: {integration_score:.2f}")
        
        # NEW: Time-based quality enhancement (5% weight)
        time_score = 0
        current_hour = current_time.hour
        current_minute = current_time.minute
        
        # Optimal trading hours get bonus
        if 9 <= current_hour <= 15:  # Market hours
            if current_hour == 9 and current_minute >= 15:  # Opening session
                time_score = 0.03  # High volatility period
            elif 10 <= current_hour <= 14:  # Mid-session
                time_score = 0.05  # Best liquidity period
            elif current_hour == 15 and current_minute <= 15:  # Power hour
                time_score = 0.04  # High volatility again
            else:
                time_score = 0.02  # Market hours but not optimal
        
        score += time_score
        debug_info.append(f"Time: {time_score:.2f}")
        
        # Final score calculation with quality bonuses
        final_score = min(1.0, score)
        
        # BONUS: Exceptional setups get small bonus (max 0.05)
        bonus = 0
        if technical and today_context:
            # Perfect gap continuation setup
            gap_type = today_context.get('gap_type', 'NONE')
            intraday_momentum = today_context.get('intraday_momentum', 'NEUTRAL')
            signal_type = entry_signal.get('signal_type', 'HOLD')
            
            if ((gap_type == 'GAP_UP' and intraday_momentum == 'BULLISH' and signal_type == 'BUY') or
                (gap_type == 'GAP_DOWN' and intraday_momentum == 'BEARISH' and signal_type == 'SELL')):
                bonus = 0.05
                debug_info.append(f"Bonus: {bonus:.2f} (perfect gap setup)")
            
            # High confidence + strong technical alignment
            elif (confidence > 0.8 and tech_confidence > 0.8 and 
                market_bias in ['BULLISH', 'BEARISH'] and signal_type in ['BUY', 'SELL']):
                bonus = 0.03
                debug_info.append(f"Bonus: {bonus:.2f} (high confidence alignment)")
        
        final_score = min(1.0, final_score + bonus)
        
        # Enhanced logging with breakdown
        logger.info(f"[TARGET] Quality Score: {final_score:.3f} | {' | '.join(debug_info)}")
        
        # Additional context logging for debugging
        if final_score >= 0.8:
            logger.info(f"🌟 EXCEPTIONAL setup: {final_score:.3f}")
        elif final_score >= 0.7:
            logger.info(f"[OK] STRONG setup: {final_score:.3f}")
        elif final_score >= 0.6:
            logger.info(f"👍 GOOD setup: {final_score:.3f}")
        else:
            logger.info(f"[WARNING] WEAK setup: {final_score:.3f}")
        
        return final_score
    
    def _is_good_market_condition(self, ticker: str, analysis_result: Dict) -> bool:
        """Check if market conditions are good for trading"""
        
        market_data = analysis_result.get('market_data', {})
        
        # Check for sufficient volume (RELAXED)
        volume = market_data.get('volume', 0)
        if volume < 50000:  # LOWERED from 100000 to 50000
            return False
        
        # Check for reasonable spread (RELAXED)
        bid = market_data.get('bid', 0)
        ask = market_data.get('ask', 0)
        
        if bid > 0 and ask > 0:
            spread = (ask - bid) / ask
            if spread > 0.10:  # INCREASED from 0.05 to 0.10 (more lenient)
                return False
        
        # Check market hours
        current_time = datetime.now()
        if current_time.hour < 9 or current_time.hour >= 15:
            return False
        
        return True
    
    def get_signal_stats(self) -> Dict:
        """Get statistics about sent signals"""
        
        if not self.signal_history:
            return {'total_signals': 0}
        
        recent_signals = [s for s in self.signal_history 
                         if (datetime.now() - s['timestamp']).total_seconds() < 86400]  # Last 24 hours
        
        strategies = [s['strategy'] for s in recent_signals]
        tickers = [s['ticker'] for s in recent_signals]
        
        return {
            'total_signals': len(self.signal_history),
            'signals_today': len(recent_signals),
            'avg_confidence': sum(s['confidence'] for s in recent_signals) / len(recent_signals) if recent_signals else 0,
            'most_common_strategy': max(set(strategies), key=strategies.count) if strategies else 'None',
            'most_active_ticker': max(set(tickers), key=tickers.count) if tickers else 'None',
            'signal_frequency': len(recent_signals) / 24 if recent_signals else 0  # per hour
        }



class RateLimiter:
    """Rate limiter for API calls to avoid hitting Zerodha limits"""

    def __init__(self, max_calls: int, time_window: int = 1):
        """
        Initialize rate limiter

        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds (default: 1 second)
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        async with self._lock:
            import time as time_module
            now = time_module.time()

            # Remove calls outside time window
            self.calls = [call_time for call_time in self.calls
                         if call_time > now - self.time_window]

            # If at limit, wait
            if len(self.calls) >= self.max_calls:
                sleep_time = self.time_window - (now - self.calls[0])
                if sleep_time > 0:
                    logger.debug(f"Rate limit reached, sleeping {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)

            self.calls.append(time_module.time())


class IndianTradingBot:
    """Main Indian Trading Bot orchestrator with Zerodha Kite Connect API integration"""
    
    def __init__(self, config: BotConfig = None):
        """FIXED Initialize the trading bot with proper database handling"""
        
        # Load configuration
        self.config = config or self._load_config()
        
        # Initialize timezone
        self.timezone = pytz.timezone('Asia/Kolkata')
        
        self.signal_filter = SmartSignalFilter()
        
        # Initialize components
        logger.info("[ROCKET] Initializing Indian Trading Bot components with Zerodha Kite Connect API...")
        
        try:
            # CRITICAL FIX 1: Ensure database schema BEFORE initializing trade logger
            schema_manager = DatabaseSchemaManager()
            if not schema_manager.ensure_automation_schema():
                raise Exception("Database schema migration failed - cannot continue")
            
            # Core components - Updated for Zerodha
            self.zerodha_client = ZerodhaAPIClient(
                api_key=self.config.ZERODHA_API_KEY,
                access_token=self.config.ZERODHA_ACCESS_TOKEN
            )
            
            # UPDATED: Use the Enhanced Indian Options Trade Generator
            self.options_analyzer = ZerodhaEnhancedOptionsAnalyzer(
                zerodha_client=self.zerodha_client,
                claude_api_key=self.config.CLAUDE_API_KEY
            )
            
            self.market_processor = IndianMarketProcessor()
            
            self.telegram_bot = TelegramSignalBot(
                bot_token=self.config.TELEGRAM_BOT_TOKEN,
                chat_id=self.config.TELEGRAM_CHAT_ID
            )
            
            # CRITICAL FIX 2: Initialize trade logger AFTER schema is ready
            self.trade_logger = IndianTradeLogger()
            
            # Test automation integration
            try:
                automation_test = self.trade_logger.test_automation_integration()
                if automation_test.get('status') == 'success':
                    logger.info("[OK] Automation integration verified")
                else:
                    logger.warning(f"[WARNING] Automation integration issue: {automation_test.get('message')}")
            except Exception as e:
                logger.error(f"Automation integration test failed: {e}")
            
            # Thread-safe state management
            import threading
            self._state_lock = threading.Lock()
            self._scan_lock = threading.Lock()

            self.daily_signals_sent = 0
            self.active_positions = []
            self.last_scan_time = None
            self.market_status = 'CLOSED'

            # Rate limiters for API calls
            self.rate_limiter = RateLimiter(
                max_calls=self.config.ZERODHA_CALLS_PER_SECOND,
                time_window=1
            )
            self.historical_rate_limiter = RateLimiter(
                max_calls=self.config.ZERODHA_HISTORICAL_CALLS_PER_SECOND,
                time_window=1
            )
            
            # Performance tracking - FIXED: Include all required keys
            self.performance_stats = {
                'signals_sent_today': 0,
                'high_confidence_signals': 0,
                'scan_cycles_completed': 0,
                'errors_today': 0,
                'last_error': None,
                'intraday_signals': 0,
                'swing_signals': 0,
                'api_issues': 0,
                'analyzer_version': 'Enhanced v4.0',
                # NEW: Automation integration stats
                'automation_signals_sent': 0,
                'automation_processing_rate': 0
            }
            
            logger.info("[OK] All components initialized successfully with Zerodha Kite Connect API")
            logger.info("[OK] Database automation integration ready")
            
        except Exception as e:
            logger.error(f"[ERROR] Initialization failed: {e}")
            raise
    
    def _load_config(self) -> BotConfig:
        """Load configuration from environment variables with validation"""

        # Load and validate watchlist
        default_watchlist = 'NIFTY,RELIANCE,HDFCBANK,TCS,INFY,BAJFINANCE,MARUTI,HINDUNILVR,HCLTECH,MPHASIS,BHARTIARTL'
        watchlist = os.getenv('INDIAN_WATCHLIST', default_watchlist).split(',')

        # Validate and parse ACCOUNT_SIZE
        try:
            account_size = float(os.getenv('ACCOUNT_SIZE', '100000'))
            if account_size < BotConfig.MIN_ACCOUNT_SIZE:
                logger.error(f"ACCOUNT_SIZE too small: ₹{account_size}. Using minimum ₹{BotConfig.MIN_ACCOUNT_SIZE}")
                account_size = BotConfig.MIN_ACCOUNT_SIZE
            elif account_size > BotConfig.MAX_ACCOUNT_SIZE:
                logger.error(f"ACCOUNT_SIZE too large: ₹{account_size}. Using maximum ₹{BotConfig.MAX_ACCOUNT_SIZE}")
                account_size = BotConfig.MAX_ACCOUNT_SIZE
        except ValueError:
            logger.error("Invalid ACCOUNT_SIZE in environment, using default ₹100,000")
            account_size = 100000

        # Validate SCAN_INTERVAL_MINUTES
        try:
            scan_interval = int(os.getenv('SCAN_INTERVAL_MINUTES', '5'))
            if scan_interval < BotConfig.MIN_SCAN_INTERVAL:
                logger.error(f"SCAN_INTERVAL_MINUTES must be at least {BotConfig.MIN_SCAN_INTERVAL}. Using minimum")
                scan_interval = BotConfig.MIN_SCAN_INTERVAL
            elif scan_interval > BotConfig.MAX_SCAN_INTERVAL:
                logger.error(f"SCAN_INTERVAL_MINUTES too large. Using maximum {BotConfig.MAX_SCAN_INTERVAL} minutes")
                scan_interval = BotConfig.MAX_SCAN_INTERVAL
        except ValueError:
            logger.error("Invalid SCAN_INTERVAL_MINUTES, using default 5 minutes")
            scan_interval = 5

        # Validate risk tolerance
        risk_tolerance = os.getenv('RISK_TOLERANCE', 'medium').lower()
        if risk_tolerance not in ['low', 'medium', 'high']:
            logger.error(f"Invalid RISK_TOLERANCE: {risk_tolerance}. Using 'medium'")
            risk_tolerance = 'medium'

        return BotConfig(
            ZERODHA_API_KEY=os.getenv('ZERODHA_API_KEY'),
            ZERODHA_API_SECRET=os.getenv('ZERODHA_API_SECRET'),
            ZERODHA_ACCESS_TOKEN=os.getenv('ZERODHA_ACCESS_TOKEN'),
            TELEGRAM_BOT_TOKEN=os.getenv('TELEGRAM_BOT_TOKEN'),
            TELEGRAM_CHAT_ID=os.getenv('TELEGRAM_CHAT_ID'),
            CLAUDE_API_KEY=os.getenv('CLAUDE_API_KEY'),
            INDIAN_WATCHLIST=[ticker.strip() for ticker in watchlist if ticker.strip()],
            SCAN_INTERVAL_MINUTES=scan_interval,
            ACCOUNT_SIZE=account_size,
            RISK_TOLERANCE=risk_tolerance
        )
    
    def start_bot(self):
        """Start the automated Indian trading bot"""
        
        try:
            logger.info("[IN] Starting Indian Trading Signal Bot with Zerodha Kite Connect API...")
            logger.info(f"📋 Monitoring {len(self.config.INDIAN_WATCHLIST)} stocks")
            logger.info(f"⏰ Scan interval: {self.config.SCAN_INTERVAL_MINUTES} minutes")
            logger.info(f"[MONEY] Account size: ₹{self.config.ACCOUNT_SIZE:,.0f}")
            
            # Display startup information
            self._display_startup_info()
            
            # Send startup notification
            self.telegram_bot.send_message(
                "[ROCKET] Indian Trading Bot Started with Zerodha Kite Connect API!\n"
                f"[CHART] Monitoring: {', '.join(self.config.INDIAN_WATCHLIST[:5])}...\n"
                f"⏰ Scan interval: {self.config.SCAN_INTERVAL_MINUTES} min\n"
                f"[BULB] Min confidence: {self.config.MIN_CONFIDENCE_SCORE:.0%}\n"
                "Ready for market open! [BELL]"
            )
            
            # Setup scheduler with IST timezone
            scheduler = BlockingScheduler(
                timezone=self.timezone,
                executors={
                    'default': ThreadPoolExecutor(20)
                }
            )
            
            # Pre-market preparation (9:00 AM)
            scheduler.add_job(
                func=self._async_job_wrapper(self.pre_market_analysis),
                trigger="cron",
                hour=9, minute=0,
                day_of_week='mon-fri',
                id='pre_market'
            )
            
            # ===== UPDATED: 5-MINUTE MARKET SCANNING =====
            # Generate scan times every 5 minutes: 9:15, 9:20, 9:25, 9:30, etc.
            scan_times = []
            for hour in range(9, 16):  # 9 AM to 3 PM
                start_minute = 30 if hour == 9 else 0  # Start at 9:30 for first hour
                end_minute = 30 if hour == 15 else 60  # End at 15:30 for last hour
                
                for minute in range(start_minute, end_minute, 5):  # Every 5 minutes
                    if hour == 15 and minute > 30:  # Don't go past 3:30 PM
                        break
                    scan_times.append((hour, minute))
            
            # FIXED: Add each scan time as a separate cron job with async wrapper
            for hour, minute in scan_times:
                scheduler.add_job(
                    func=self._async_job_wrapper(self.market_scan_cycle),  # [OK] FIXED: Use async wrapper
                    trigger="cron",
                    hour=hour, 
                    minute=minute,
                    day_of_week='mon-fri',
                    id=f'market_scan_{hour}_{minute:02d}'
                )
            
            logger.info(f"📅 Scheduled {len(scan_times)} market scans every 5 minutes")
            
            # Post-market summary (3:35 PM)
            scheduler.add_job(
                func=self._async_job_wrapper(self.post_market_summary),
                trigger="cron",
                hour=15, minute=35,
                day_of_week='mon-fri',
                id='post_market'
            )
            
            # Heartbeat check every 5 minutes
            scheduler.add_job(
                func=self._heartbeat_check,
                trigger="interval",
                minutes=5,
                id='heartbeat'
            )
            
            # Start scheduler
            logger.info("📅 Scheduler configured, starting...")
            scheduler.start()
            
        except KeyboardInterrupt:
            logger.info("🛑 Bot shutdown requested...")
            self._cleanup()
            scheduler.shutdown()
            
        except Exception as e:
            logger.error(f"[ERROR] Fatal error: {e}", exc_info=True)
            self.telegram_bot.send_error_alert(f"Bot crashed: {str(e)}")
            raise
    
    async def pre_market_analysis(self):
        """Enhanced pre-market preparation and analysis with technical insights (9:00 AM)"""
        
        logger.info("🌅 Starting enhanced pre-market analysis...")
        
        try:
            # Reset daily counters
            self.daily_signals_sent = 0
            self.performance_stats['signals_sent_today'] = 0
            self.performance_stats['high_confidence_signals'] = 0
            self.performance_stats['errors_today'] = 0
            
            # Check market status using Zerodha API
            market_status = self._check_market_status()
            if market_status == 'HOLIDAY':
                self.telegram_bot.send_message("🏖️ Market holiday today. Bot will rest.")
                return
            
            # Get global market sentiment
            global_sentiment = self._analyze_global_markets()
            
            # Check for major events
            events_today = self._check_market_events()
            
            # **NEW: Enhanced Gap Analysis using Technical Analyzer**
            enhanced_gap_analysis = await self._get_enhanced_gap_analysis()
            
            # **NEW: Pre-market Technical Screening**
            pre_market_opportunities = await self._scan_pre_market_opportunities()
            
            # Prepare enhanced pre-market summary
            summary_parts = [
                "🌅 ENHANCED PRE-MARKET BRIEFING",
                f"📅 {datetime.now(self.timezone).strftime('%d %B %Y, %A')}",
                "",
                f"🌍 Global Markets: {global_sentiment['summary']}",
                f"• US Markets: {global_sentiment['us_close']}",
                f"• Asian Markets: {global_sentiment['asia_status']}",
                f"• SGX Nifty: {global_sentiment['sgx_nifty']}",
                ""
            ]
            
            # **NEW: Enhanced Gap Analysis Section**
            if enhanced_gap_analysis['gaps_detected']:
                summary_parts.extend([
                    "[CHART] GAP ANALYSIS & STRATEGY:",
                    ""
                ])
                
                for gap_info in enhanced_gap_analysis['gap_setups'][:4]:  # Top 4 gap setups
                    gap_emoji = "[UP]" if gap_info['gap_type'] == 'GAP_UP' else "[DOWN]"
                    strategy_emoji = "[TARGET]" if gap_info['confidence'] > 0.7 else "[WARNING]"
                    
                    summary_parts.extend([
                        f"{gap_emoji} {gap_info['ticker']} - {gap_info['gap_type'].replace('_', ' ')}",
                        f"   Gap: {gap_info['gap_percent']:+.1f}% | Momentum: {gap_info['intraday_momentum']}",
                        f"   {strategy_emoji} Strategy: {gap_info['recommended_strategy']}",
                        f"   Targets: ₹{gap_info['gap_fill_target']:.0f} (fill) | ₹{gap_info['gap_extension_target']:.0f} (extension)",
                        ""
                    ])
                
                # Gap trading summary
                total_gap_ups = len([g for g in enhanced_gap_analysis['gap_setups'] if g['gap_type'] == 'GAP_UP'])
                total_gap_downs = len([g for g in enhanced_gap_analysis['gap_setups'] if g['gap_type'] == 'GAP_DOWN'])
                
                summary_parts.extend([
                    f"[UP] Total Gap Ups: {total_gap_ups} | [DOWN] Gap Downs: {total_gap_downs}",
                    f"[TARGET] High Confidence Setups: {len([g for g in enhanced_gap_analysis['gap_setups'] if g['confidence'] > 0.7])}",
                    ""
                ])
            else:
                summary_parts.extend([
                    "[CHART] GAP ANALYSIS:",
                    "• No significant gaps detected",
                    "• Normal opening expected",
                    ""
                ])
            
            # **NEW: Key Technical Levels with Enhanced Analysis**
            technical_levels = enhanced_gap_analysis.get('technical_levels', {})
            
            summary_parts.extend([
                "[CHART] KEY TECHNICAL LEVELS:",
                f"• Nifty: Support ₹{technical_levels.get('nifty_support', 24500):.0f} | Resistance ₹{technical_levels.get('nifty_resistance', 25000):.0f}",
                f"• Bank Nifty: Support ₹{technical_levels.get('banknifty_support', 56000):.0f} | Resistance ₹{technical_levels.get('banknifty_resistance', 57000):.0f}",
                ""
            ])
            
            # **NEW: Add trend context**
            if technical_levels.get('nifty_trend'):
                nifty_trend = technical_levels['nifty_trend']
                trend_emoji = "🐂" if nifty_trend == 'BULLISH' else "🐻" if nifty_trend == 'BEARISH' else "🦆"
                summary_parts.append(f"• Nifty Trend: {trend_emoji} {nifty_trend} (Strength: {technical_levels.get('nifty_trend_strength', 0.5):.1f})")
            
            if technical_levels.get('banknifty_trend'):
                bn_trend = technical_levels['banknifty_trend']
                trend_emoji = "🐂" if bn_trend == 'BULLISH' else "🐻" if bn_trend == 'BEARISH' else "🦆"
                summary_parts.append(f"• Bank Nifty Trend: {trend_emoji} {bn_trend} (Strength: {technical_levels.get('banknifty_trend_strength', 0.5):.1f})")
            
            summary_parts.append("")
            
            # **NEW: Pre-market Opportunities Section**
            if pre_market_opportunities['high_confidence_setups']:
                summary_parts.extend([
                    "[TARGET] PRE-MARKET OPPORTUNITIES:",
                    ""
                ])
                
                for opp in pre_market_opportunities['high_confidence_setups'][:3]:  # Top 3
                    confidence_stars = "[STAR][STAR][STAR][STAR][STAR]" if opp['confidence'] > 0.8 else "[STAR][STAR][STAR][STAR]" if opp['confidence'] > 0.7 else "[STAR][STAR][STAR]"
                    
                    summary_parts.extend([
                        f"• {opp['ticker']}: {opp['bias']} bias ({opp['confidence']:.0%}) {confidence_stars}",
                        f"  Entry: {opp['entry_condition'][:60]}",
                        f"  Strategy: {opp['recommended_strategy'].replace('_', ' ')}",
                        ""
                    ])
            
            # **NEW: Market Regime Analysis**
            market_regime = enhanced_gap_analysis.get('market_regime', {})
            if market_regime:
                volatility_regime = market_regime.get('volatility_regime', 'NORMAL')
                session_expectation = market_regime.get('session_expectation', 'NORMAL')
                
                regime_emoji = "🔥" if volatility_regime == 'HIGH' else "😴" if volatility_regime == 'LOW' else "[CHART]"
                
                summary_parts.extend([
                    "🌡️ MARKET REGIME:",
                    f"• Volatility: {regime_emoji} {volatility_regime}",
                    f"• Session Expectation: {session_expectation}",
                    f"• Recommended Style: {market_regime.get('recommended_trading_style', 'BALANCED')}",
                    ""
                ])
            
            # Add events
            if events_today:
                summary_parts.append("📢 Events Today: " + ", ".join(events_today))
                summary_parts.append("")
            
            # **NEW: Enhanced Bot Configuration with Gap Strategy Info**
            summary_parts.extend([
                f"⚙️ BOT CONFIGURATION:",
                f"• API: Zerodha Kite Connect (Enhanced v4.0)",
                f"• Watchlist: {len(self.config.INDIAN_WATCHLIST)} stocks + gap analysis",
                f"• Min Confidence: {self.config.MIN_CONFIDENCE_SCORE:.0%} (session-adaptive)",
                f"• Risk Level: {self.config.RISK_TOLERANCE}",
                f"• Gap Strategy: {'Enabled' if enhanced_gap_analysis['gaps_detected'] else 'Monitoring'}",
                ""
            ])
            
            # **NEW: Today's Focus Areas**
            focus_areas = []
            
            if enhanced_gap_analysis['gaps_detected']:
                gap_count = len(enhanced_gap_analysis['gap_setups'])
                focus_areas.append(f"Gap trading ({gap_count} setups)")
            
            if pre_market_opportunities['high_confidence_setups']:
                opp_count = len(pre_market_opportunities['high_confidence_setups'])
                focus_areas.append(f"High-confidence setups ({opp_count} found)")
            
            if market_regime.get('volatility_regime') == 'HIGH':
                focus_areas.append("Volatility plays")
            
            if technical_levels.get('nifty_trend') in ['BULLISH', 'BEARISH']:
                focus_areas.append(f"Trend following ({technical_levels['nifty_trend'].lower()})")
            
            if focus_areas:
                summary_parts.extend([
                    "[TARGET] TODAY'S FOCUS AREAS:",
                    "• " + " | ".join(focus_areas),
                    ""
                ])
            
            summary_parts.extend([
                "[BELL] Market opens in 15 minutes!",
                "Enhanced bot ready for gap analysis & technical scanning."
            ])
            
            # Send enhanced pre-market summary
            self.telegram_bot.send_message("\n".join(summary_parts))
            
            # **NEW: Send detailed gap strategy alerts if significant gaps**
            if enhanced_gap_analysis['gaps_detected']:
                await self._send_gap_strategy_alerts(enhanced_gap_analysis['gap_setups'])
            
            # Log enhanced pre-market prep completion
            logger.info("[OK] Enhanced pre-market analysis completed")
            logger.info(f"[CHART] Gaps detected: {len(enhanced_gap_analysis.get('gap_setups', []))}")
            logger.info(f"[TARGET] High-confidence opportunities: {len(pre_market_opportunities.get('high_confidence_setups', []))}")
            
        except Exception as e:
            logger.error(f"Enhanced pre-market analysis error: {e}", exc_info=True)
            self.telegram_bot.send_error_alert(f"Enhanced pre-market prep failed: {str(e)}")

    async def _get_enhanced_gap_analysis(self) -> Dict:
        """Get enhanced gap analysis using technical analyzer"""
        
        try:
            gap_setups = []
            technical_levels = {}
            gaps_detected = False
            
            # Analyze key indices and stocks for gaps
            key_symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'HDFCBANK']
            
            for symbol in key_symbols:
                try:
                    # Get market data
                    market_data = await self.market_data_provider.fetch_live_data(symbol)
                    current_price = market_data['current_price']
                    
                    # Use quick intraday analysis for gap detection
                    gap_analysis = await self.technical_analyzer.quick_intraday_analysis(
                        symbol, current_price, market_data
                    )
                    
                    # Extract gap information
                    gap_info = gap_analysis.get('gap_analysis', {})
                    if gap_info.get('gap_type', 'NONE') != 'NONE':
                        gaps_detected = True
                        
                        # Get momentum analysis
                        momentum_info = gap_analysis.get('momentum_immediate', {})
                        
                        gap_setup = {
                            'ticker': symbol,
                            'gap_type': gap_info['gap_type'],
                            'gap_percent': gap_info['gap_percent'],
                            'gap_significance': gap_info['gap_significance'],
                            'today_open': gap_info['today_open'],
                            'yesterday_close': gap_info['yesterday_close'],
                            'intraday_momentum': momentum_info.get('direction', 'NEUTRAL'),
                            'confidence': gap_analysis['confidence'],
                            'recommended_strategy': self._get_gap_strategy(gap_info, momentum_info),
                            'gap_fill_target': gap_info['yesterday_close'],
                            'gap_extension_target': gap_info['today_open'] + (gap_info['today_open'] - gap_info['yesterday_close']) * 0.5,
                            'recommended_action': gap_analysis['recommended_action']
                        }
                        
                        gap_setups.append(gap_setup)
                    
                    # Store technical levels for indices
                    if symbol in ['NIFTY', 'BANKNIFTY']:
                        # Try to get technical analysis for trend info
                        try:
                            tech_analysis = await self.technical_analyzer.analyze_symbol_for_options(
                                symbol, current_price, market_data, 'intraday'
                            )
                            
                            support_resistance = tech_analysis.get('support_resistance', {})
                            trend_analysis = tech_analysis.get('trend_analysis', {})
                            
                            if symbol == 'NIFTY':
                                technical_levels.update({
                                    'nifty_support': support_resistance.get('nearest_support', current_price * 0.98),
                                    'nifty_resistance': support_resistance.get('nearest_resistance', current_price * 1.02),
                                    'nifty_trend': tech_analysis.get('market_bias', 'NEUTRAL'),
                                    'nifty_trend_strength': trend_analysis.get('trend_strength', 0.5)
                                })
                            else:  # BANKNIFTY
                                technical_levels.update({
                                    'banknifty_support': support_resistance.get('nearest_support', current_price * 0.98),
                                    'banknifty_resistance': support_resistance.get('nearest_resistance', current_price * 1.02),
                                    'banknifty_trend': tech_analysis.get('market_bias', 'NEUTRAL'),
                                    'banknifty_trend_strength': trend_analysis.get('trend_strength', 0.5)
                                })
                        except Exception as tech_error:
                            logger.warning(f"Technical analysis failed for {symbol}: {tech_error}")
                            # Use fallback levels
                            if symbol == 'NIFTY':
                                technical_levels.update({
                                    'nifty_support': current_price * 0.98,
                                    'nifty_resistance': current_price * 1.02
                                })
                            else:
                                technical_levels.update({
                                    'banknifty_support': current_price * 0.98,
                                    'banknifty_resistance': current_price * 1.02
                                })
                    
                except Exception as symbol_error:
                    logger.error(f"Gap analysis failed for {symbol}: {symbol_error}")
                    continue
            
            # Sort gap setups by significance and confidence
            gap_setups.sort(key=lambda x: (x['confidence'], abs(x['gap_percent'])), reverse=True)
            
            # Determine market regime
            market_regime = self._determine_market_regime(gap_setups, technical_levels)
            
            return {
                'gaps_detected': gaps_detected,
                'gap_setups': gap_setups,
                'technical_levels': technical_levels,
                'market_regime': market_regime
            }
            
        except Exception as e:
            logger.error(f"Enhanced gap analysis error: {e}")
            return {
                'gaps_detected': False,
                'gap_setups': [],
                'technical_levels': {},
                'market_regime': {}
            }

    async def _scan_pre_market_opportunities(self) -> Dict:
        """Scan for high-confidence pre-market opportunities"""
        
        try:
            high_confidence_setups = []
            
            # Quick scan of watchlist for opportunities
            for ticker in self.config.INDIAN_WATCHLIST[:8]:  # Limit to top 8 for speed
                try:
                    market_data = await self.market_data_provider.fetch_live_data(ticker)
                    current_price = market_data['current_price']
                    
                    # Quick analysis
                    quick_result = await self.technical_analyzer.quick_intraday_analysis(
                        ticker, current_price, market_data
                    )
                    
                    if quick_result['confidence'] > 0.7:  # High confidence only
                        setup = {
                            'ticker': ticker,
                            'bias': quick_result['immediate_bias'],
                            'confidence': quick_result['confidence'],
                            'entry_condition': quick_result.get('gap_analysis', {}).get('reason', 'Technical setup'),
                            'recommended_strategy': self._map_bias_to_strategy(quick_result['immediate_bias']),
                            'current_price': current_price
                        }
                        
                        high_confidence_setups.append(setup)
                        
                except Exception as ticker_error:
                    logger.warning(f"Pre-market scan failed for {ticker}: {ticker_error}")
                    continue
            
            # Sort by confidence
            high_confidence_setups.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                'high_confidence_setups': high_confidence_setups
            }
            
        except Exception as e:
            logger.error(f"Pre-market opportunities scan error: {e}")
            return {'high_confidence_setups': []}

    def _get_gap_strategy(self, gap_info: Dict, momentum_info: Dict) -> str:
        """Determine optimal gap trading strategy"""
        
        gap_type = gap_info.get('gap_type', 'NONE')
        gap_significance = gap_info.get('gap_significance', 'LOW')
        momentum_direction = momentum_info.get('direction', 'NEUTRAL')
        
        if gap_type == 'GAP_UP':
            if momentum_direction == 'BULLISH':
                return 'GAP_UP_CONTINUATION' if gap_significance == 'HIGH' else 'BULLISH_CALL'
            elif momentum_direction == 'BEARISH':
                return 'GAP_UP_REVERSAL'
            else:
                return 'GAP_FILL_PLAY'
        
        elif gap_type == 'GAP_DOWN':
            if momentum_direction == 'BEARISH':
                return 'GAP_DOWN_CONTINUATION' if gap_significance == 'HIGH' else 'BEARISH_PUT'
            elif momentum_direction == 'BULLISH':
                return 'GAP_DOWN_REVERSAL'
            else:
                return 'GAP_FILL_PLAY'
        
        return 'NORMAL_STRATEGY'

    def _determine_market_regime(self, gap_setups: List[Dict], technical_levels: Dict) -> Dict:
        """Determine overall market regime for the day"""
        
        # Count significant gaps
        significant_gaps = len([g for g in gap_setups if g['gap_significance'] in ['HIGH', 'MODERATE']])
        
        # Determine volatility regime
        if significant_gaps >= 3:
            volatility_regime = 'HIGH'
            session_expectation = 'VOLATILE_OPENING'
            recommended_style = 'GAP_TRADING'
        elif significant_gaps >= 1:
            volatility_regime = 'MODERATE'
            session_expectation = 'SELECTIVE_GAPS'
            recommended_style = 'MIXED'
        else:
            volatility_regime = 'LOW'
            session_expectation = 'NORMAL_SESSION'
            recommended_style = 'TREND_FOLLOWING'
        
        return {
            'volatility_regime': volatility_regime,
            'session_expectation': session_expectation,
            'recommended_trading_style': recommended_style,
            'gap_count': len(gap_setups),
            'significant_gap_count': significant_gaps
        }

    def _map_bias_to_strategy(self, bias: str) -> str:
        """Map immediate bias to trading strategy"""
        
        bias_mapping = {
            'STRONG_BULLISH': 'INTRADAY_LONG_CALL',
            'STRONG_BEARISH': 'INTRADAY_LONG_PUT',
            'MOMENTUM_BULLISH': 'BULLISH_CALL',
            'MOMENTUM_BEARISH': 'BEARISH_PUT',
            'REVERSAL_SETUP': 'CONTRARIAN_PLAY'
        }
        
        return bias_mapping.get(bias, 'WAIT_AND_WATCH')

    async def _send_gap_strategy_alerts(self, gap_setups: List[Dict]):
        """Send detailed gap strategy alerts for significant gaps"""
        
        significant_gaps = [g for g in gap_setups if g['gap_significance'] in ['HIGH', 'MODERATE'] and g['confidence'] > 0.7]
        
        if not significant_gaps:
            return
        
        alert_parts = [
            "[TARGET] GAP TRADING ALERTS",
            "High-confidence gap setups detected:",
            ""
        ]
        
        for gap in significant_gaps[:3]:  # Top 3 only
            strategy_emoji = "[ROCKET]" if 'CONTINUATION' in gap['recommended_strategy'] else "🔄" if 'REVERSAL' in gap['recommended_strategy'] else "[CHART]"
            
            alert_parts.extend([
                f"{strategy_emoji} {gap['ticker']} - {gap['gap_type'].replace('_', ' ')}",
                f"   Gap: {gap['gap_percent']:+.1f}% | Confidence: {gap['confidence']:.0%}",
                f"   Strategy: {gap['recommended_strategy'].replace('_', ' ')}",
                f"   Entry: {gap['recommended_action'][:50]}",
                f"   Targets: Fill ₹{gap['gap_fill_target']:.0f} | Extend ₹{gap['gap_extension_target']:.0f}",
                ""
            ])
        
        alert_parts.append("[WARNING] Execute with proper risk management!")
        
        # Send as separate message for gap alerts
        self.telegram_bot.send_message("\n".join(alert_parts))
    
    async def market_scan_cycle(self):
        """Main market scanning cycle - runs every 5 minutes during market hours (ASYNC VERSION)"""
        
        scan_start = datetime.now(self.timezone)
        logger.info(f"🔍 Starting market scan at {scan_start.strftime('%H:%M:%S')}")
        
        try:
            # Check if market is open
            if not self._is_market_open():
                logger.info("Market is closed, skipping scan")
                return
            
            # Update market status
            self.market_status = 'OPEN'
            
            # Track performance
            signals_found = 0
            errors = []
            scan_results = []
            
            # Scan each stock in watchlist
            for ticker in self.config.INDIAN_WATCHLIST:
                try:
                    logger.info(f"Analyzing {ticker}...")
                    
                    # PROPER ASYNC: Use await instead of asyncio.run()
                    analysis_result = await self._analyze_stock(ticker)
                    
                    if analysis_result and self._should_send_signal(analysis_result, ticker):
                        # Format and send signal
                        signal_sent = self._process_and_send_signal(ticker, analysis_result)
                        
                        if signal_sent:
                            signals_found += 1
                            scan_results.append({
                                'ticker': ticker,
                                'strategy': analysis_result['trade_recommendation']['strategy'],
                                'confidence': analysis_result['trade_recommendation']['confidence'],
                                'trading_style': analysis_result.get('trade_type', 'SWING')
                            })
                            
                            # Log the signal
                            self.trade_logger.log_signal(ticker, analysis_result, source='automated_scan')
                    
                    # Rate limiting - avoid overwhelming APIs
                    await asyncio.sleep(1)  # Use async sleep
                    
                except Exception as e:
                    error_msg = f"Error analyzing {ticker}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    errors.append(error_msg)
                    continue
            
            # Update performance stats
            self.performance_stats['scan_cycles_completed'] += 1
            self.performance_stats['signals_sent_today'] += signals_found
            
            # Send scan summary
            scan_duration = (datetime.now(self.timezone) - scan_start).total_seconds()
            self._send_scan_summary(signals_found, errors, scan_duration, scan_results)
            
            logger.info(f"[OK] Scan completed: {signals_found} signals found in {scan_duration:.1f}s")
            
        except Exception as e:
            logger.error(f"Market scan cycle error: {e}", exc_info=True)
            self.telegram_bot.send_error_alert(f"Scan cycle failed: {str(e)}")
            self.performance_stats['errors_today'] += 1
            self.performance_stats['last_error'] = str(e)

    # If you use this async version, you'll also need to update your scheduler job registration:
    def _async_job_wrapper(self, coro_func):
        """Wrapper to run async functions in scheduler with proper locking"""
        def wrapper():
            # Prevent concurrent scans
            if not self._scan_lock.acquire(blocking=False):
                logger.warning("Previous scan still running, skipping this cycle")
                return

            try:
                # Use a dedicated event loop for this job
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(coro_func())
                finally:
                    loop.close()
            finally:
                self._scan_lock.release()
        return wrapper
    
    async def _analyze_stock(self, ticker: str) -> Optional[Dict]:
        """Enhanced analysis with technical integration"""
        
        try:
            logger.info(f"Running enhanced analysis for {ticker}...")
            
            # Use the enhanced method from your technical analyzer
            analysis_result = await self.options_analyzer.analyze_trade(
                symbol=ticker,
                trading_style='intraday' if self._is_market_open() else 'swing',
                prediction_days=1 if self._is_market_open() else 14,
                risk_tolerance=self.config.RISK_TOLERANCE,
                capital=self.config.ACCOUNT_SIZE,
                execute_trades=False
            )
            
            if analysis_result and not analysis_result.get('error'):
                logger.info(f"[OK] Enhanced analysis completed for {ticker}")
                return analysis_result
            else:
                error_msg = analysis_result.get('message', 'Unknown error') if analysis_result else 'No result'
                logger.warning(f"Analysis failed for {ticker}: {error_msg}")
                return None
                
        except Exception as e:
            logger.error(f"Analysis error for {ticker}: {e}", exc_info=True)
            return None
    
    def _get_zerodha_symbol(self, symbol: str) -> str:
        """Convert bot symbol to Zerodha format"""
        symbol_mapping = {
            # Indices
            'NIFTY': 'NIFTY 50',
            # Your Existing Stocks
            'RELIANCE': 'RELIANCE',
            'HDFCBANK': 'HDFCBANK',
            'TCS': 'TCS',
            'INFY': 'INFY',
            'BAJFINANCE': 'BAJFINANCE',
            'MARUTI': 'MARUTI',
            
            # ADD THESE NEW MAPPINGS:
            'HINDUNILVR': 'HINDUNILVR',
            'HCLTECH': 'HCLTECH',
            'MPHASIS': 'MPHASIS',
            'BHARTIARTL': 'BHARTIARTL'
        }
        return symbol_mapping.get(symbol, symbol)

    def _should_send_signal(self, analysis_result: Dict, ticker: str) -> bool:
        """Enhanced signal validation with smart filtering and technical integration"""
        
        # Basic checks first
        if 'trade_recommendation' not in analysis_result:
            logger.debug(f"No trade recommendation for {ticker}")
            return False
        
        trade_rec = analysis_result['trade_recommendation']
        confidence = trade_rec.get('confidence', 0)
        
        # Check for errors
        if analysis_result.get('error', False):
            logger.debug(f"Analysis error for {ticker}")
            return False
        
        # Daily limits
        if self.daily_signals_sent >= self.config.MAX_SIGNALS_PER_DAY:
            logger.warning(f"Daily signal limit reached: {self.daily_signals_sent}/{self.config.MAX_SIGNALS_PER_DAY}")
            return False
        
        # Position limits
        if len(self.active_positions) >= self.config.MAX_POSITIONS_OPEN:
            logger.info(f"Maximum open positions reached: {len(self.active_positions)}/{self.config.MAX_POSITIONS_OPEN}")
            return False
        
        # **FIX 1: Initialize iv_score with default value**
        iv_score = 1.0  # Default neutral IV score
        
        # **NEW: IV Score Calculation (was missing)**
        market_data = analysis_result.get('market_data', {})
        zerodha_integration = analysis_result.get('zerodha_integration', {})
        
        # Calculate IV score if data is available
        try:
            if market_data:
                current_iv = market_data.get('implied_volatility', 0)
                historical_iv = market_data.get('historical_volatility', 0)
                iv_percentile = market_data.get('iv_percentile', 50)
                
                if current_iv > 0 and historical_iv > 0:
                    # IV score based on current vs historical IV
                    iv_score = current_iv / historical_iv
                    # Cap between 0.5 and 2.0 for reasonable range
                    iv_score = max(0.5, min(iv_score, 2.0))
                    logger.debug(f"IV Score for {ticker}: {iv_score:.2f} (Current: {current_iv:.1%}, Historical: {historical_iv:.1%})")
                elif iv_percentile > 0:
                    # Fallback to IV percentile
                    iv_score = 0.5 + (iv_percentile / 100)  # Range: 0.5 to 1.5
                    logger.debug(f"IV Score from percentile for {ticker}: {iv_score:.2f} (Percentile: {iv_percentile})")
        except Exception as e:
            logger.warning(f"IV score calculation failed for {ticker}: {e}")
            iv_score = 1.0  # Safe fallback
        
        # **NEW: Enhanced Confidence Blending**
        technical_analysis = analysis_result.get('technical_analysis', {})
        tech_confidence = technical_analysis.get('confidence_score', 0)
        
        # Blend options and technical confidence (60% options, 40% technical)
        if tech_confidence > 0:
            blended_confidence = (confidence * 0.6) + (tech_confidence * 0.4)
            logger.debug(f"Confidence blend for {ticker}: Options {confidence:.1%} + Technical {tech_confidence:.1%} = {blended_confidence:.1%}")
        else:
            blended_confidence = confidence
            logger.debug(f"Using options confidence only for {ticker}: {confidence:.1%}")
        
        # **NEW: Session-Aware Confidence Thresholds**
        current_time = datetime.now()
        base_threshold = self.config.MIN_CONFIDENCE_SCORE
        
        # Dynamic threshold based on market session
        if current_time.hour == 9:  # Opening session
            session_threshold = base_threshold + 0.05  # Higher threshold for volatile opening
            session_context = "opening_session"
        elif 10 <= current_time.hour <= 14:  # Mid-session
            session_threshold = base_threshold - 0.02  # Slightly lower for stable mid-session
            session_context = "mid_session"
        elif current_time.hour >= 14:  # Power hour
            session_threshold = base_threshold + 0.03  # Higher threshold for volatile power hour
            session_context = "power_hour"
        else:
            session_threshold = base_threshold
            session_context = "standard_session"
        
        # **FIX 2: Add IV Score Validation**
        # Apply IV score filter - avoid extremely high or low IV
        iv_filter_passed = True
        if iv_score > 1.8:  # Very high IV - risky
            logger.debug(f"High IV warning for {ticker}: {iv_score:.2f}")
            session_threshold += 0.05  # Require higher confidence
        elif iv_score < 0.6:  # Very low IV - less premium
            logger.debug(f"Low IV warning for {ticker}: {iv_score:.2f}")
            session_threshold += 0.03  # Require slightly higher confidence
        
        # **NEW: Today's Context Evaluation**
        today_context = technical_analysis.get('today_context', {})
        gap_bonus = 0
        momentum_bonus = 0
        volume_bonus = 0  # Initialize volume_bonus
        
        if today_context:
            gap_type = today_context.get('gap_type', 'NONE')
            intraday_momentum = today_context.get('intraday_momentum', 'NEUTRAL')
            volume_ratio = today_context.get('volume_ratio', 1.0)
            
            # **Gap Strategy Bonus**
            if gap_type != 'NONE':
                gap_percent = abs(today_context.get('gap_percent', 0))
                
                # Strong gaps get confidence bonus
                if gap_percent > 2.0:  # Strong gap
                    gap_bonus = 0.08
                elif gap_percent > 1.0:  # Moderate gap
                    gap_bonus = 0.05
                elif gap_percent > 0.5:  # Weak gap
                    gap_bonus = 0.02
                
                logger.debug(f"Gap bonus for {ticker}: {gap_type} {gap_percent:.1f}% = +{gap_bonus:.2f}")
            
            # **Momentum Alignment Bonus**
            entry_signal = technical_analysis.get('entry_signal', {})
            signal_type = entry_signal.get('signal_type', 'HOLD')
            
            # Check for gap continuation setups (highest priority)
            if ((gap_type == 'GAP_UP' and intraday_momentum == 'BULLISH' and signal_type == 'BUY') or
                (gap_type == 'GAP_DOWN' and intraday_momentum == 'BEARISH' and signal_type == 'SELL')):
                momentum_bonus = 0.10
                logger.debug(f"Perfect gap continuation setup for {ticker}: +{momentum_bonus:.2f}")
            
            # Strong intraday momentum with technical confirmation
            elif (intraday_momentum == 'BULLISH' and signal_type == 'BUY'):
                momentum_bonus = 0.05
                logger.debug(f"Bullish momentum alignment for {ticker}: +{momentum_bonus:.2f}")
            elif (intraday_momentum == 'BEARISH' and signal_type == 'SELL'):
                momentum_bonus = 0.05
                logger.debug(f"Bearish momentum alignment for {ticker}: +{momentum_bonus:.2f}")
            
            # **Volume Confirmation Bonus**
            if volume_ratio > 2.0:  # Very high volume
                volume_bonus = 0.03
            elif volume_ratio > 1.5:  # High volume
                volume_bonus = 0.02
            elif volume_ratio > 1.2:  # Above average volume
                volume_bonus = 0.01
            
            if volume_bonus > 0:
                logger.debug(f"Volume bonus for {ticker}: {volume_ratio:.1f}x = +{volume_bonus:.2f}")
            
            # Apply bonuses to blended confidence
            enhanced_confidence = blended_confidence + gap_bonus + momentum_bonus + volume_bonus
            enhanced_confidence = min(0.95, enhanced_confidence)  # Cap at 95%
        else:
            enhanced_confidence = blended_confidence
            logger.debug(f"No today's context available for {ticker}")
        
        # **NEW: Technical Signal Quality Check**
        entry_signal = technical_analysis.get('entry_signal', {})
        signal_strength = entry_signal.get('strength', 0)
        signal_type = entry_signal.get('signal_type', 'HOLD')
        
        # Require minimum signal strength for directional trades
        if signal_type in ['BUY', 'SELL']:
            min_signal_strength = 0.6  # 60% minimum for directional signals
            if signal_strength < min_signal_strength:
                logger.info(f"Signal strength too low for {ticker}: {signal_type} at {signal_strength:.1%} (min: {min_signal_strength:.1%})")
                return False
        
        # **NEW: Market Bias Alignment Check**
        market_bias = technical_analysis.get('market_bias', 'NEUTRAL')
        strategy = trade_rec.get('strategy', '')
        
        # Check for strategy-bias alignment
        strategy_direction = self._get_strategy_direction(strategy)
        bias_aligned = self._check_bias_alignment(market_bias, strategy_direction)
        
        if not bias_aligned:
            # Allow with higher confidence threshold
            session_threshold += 0.05
            logger.debug(f"Strategy-bias misalignment for {ticker}: {strategy_direction} vs {market_bias}, raising threshold to {session_threshold:.1%}")
        
        # **Main Confidence Check**
        if enhanced_confidence < session_threshold:
            logger.info(f"Enhanced confidence too low for {ticker}: {enhanced_confidence:.1%} < {session_threshold:.1%} ({session_context})")
            return False
        
        # **NEW: Enhanced Zerodha Integration Check**
        live_data_available = zerodha_integration.get('live_data_available', False)
        
        # Prefer signals with live data, but don't block fallback data
        if not live_data_available:
            logger.warning(f"Using fallback data for {ticker} - consider as lower priority")
            # Could add small confidence penalty here if desired
        
        # **FIX 3: Add error handling for smart filtering**
        try:
            # **Smart Filtering (Your Existing Logic)**
            should_send, reason = self.signal_filter.should_send_signal(ticker, analysis_result)
            
            if not should_send:
                logger.info(f"Smart filter blocked {ticker}: {reason}")
                return False
        except Exception as e:
            logger.error(f"Smart filter error for {ticker}: {e}")
            # Continue without smart filter if it fails
        
        # **NEW: Special Gap Strategy Validation**
        if today_context and gap_type != 'NONE':
            try:
                gap_validation = self._validate_gap_strategy(today_context, entry_signal, ticker)
                if not gap_validation['valid']:
                    logger.info(f"Gap strategy validation failed for {ticker}: {gap_validation['reason']}")
                    return False
            except Exception as e:
                logger.warning(f"Gap strategy validation error for {ticker}: {e}")
        
        # **NEW: Risk-Reward Validation**
        risk_reward_ratio = entry_signal.get('risk_reward_ratio', 0)
        if risk_reward_ratio > 0 and risk_reward_ratio < 1.2:  # Below 1.2:1 ratio
            logger.info(f"Poor risk-reward ratio for {ticker}: {risk_reward_ratio:.1f}:1")
            return False
        
        # **Enhanced Approval Logging**
        approval_details = {
            'ticker': ticker,
            'options_confidence': f"{confidence:.1%}",
            'technical_confidence': f"{tech_confidence:.1%}",
            'enhanced_confidence': f"{enhanced_confidence:.1%}",
            'session_threshold': f"{session_threshold:.1%}",
            'session_context': session_context,
            'iv_score': f"{iv_score:.2f}",
            'gap_bonus': f"+{gap_bonus:.2f}" if gap_bonus > 0 else "0",
            'momentum_bonus': f"+{momentum_bonus:.2f}" if momentum_bonus > 0 else "0",
            'volume_bonus': f"+{volume_bonus:.2f}" if volume_bonus > 0 else "0",
            'signal_type': signal_type,
            'signal_strength': f"{signal_strength:.1%}",
            'market_bias': market_bias,
            'strategy_direction': strategy_direction,
            'bias_aligned': bias_aligned,
            'live_data': live_data_available
        }
        
        logger.info(f"[OK] Signal APPROVED for {ticker}: {approval_details}")
        
        return True

    def _get_strategy_direction(self, strategy: str) -> str:
        """Determine strategy direction from strategy name"""
        
        bullish_keywords = ['BULLISH', 'BULL', 'CALL', 'LONG_CALL', 'INTRADAY_LONG_CALL']
        bearish_keywords = ['BEARISH', 'BEAR', 'PUT', 'LONG_PUT', 'INTRADAY_LONG_PUT']
        
        strategy_upper = strategy.upper()
        
        if any(keyword in strategy_upper for keyword in bullish_keywords):
            return 'BULLISH'
        elif any(keyword in strategy_upper for keyword in bearish_keywords):
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def _check_bias_alignment(self, market_bias: str, strategy_direction: str) -> bool:
        """Check if market bias aligns with strategy direction"""
        
        if market_bias == 'NEUTRAL' or strategy_direction == 'NEUTRAL':
            return True  # Neutral is always aligned
        
        return market_bias == strategy_direction

    def _validate_gap_strategy(self, today_context: Dict, entry_signal: Dict, ticker: str) -> Dict:
        """Validate gap trading strategy for enhanced accuracy"""
        
        gap_type = today_context.get('gap_type', 'NONE')
        gap_percent = abs(today_context.get('gap_percent', 0))
        intraday_momentum = today_context.get('intraday_momentum', 'NEUTRAL')
        signal_type = entry_signal.get('signal_type', 'HOLD')
        
        # Gap too small for meaningful trade
        if gap_percent < 0.3:
            return {'valid': False, 'reason': f'Gap too small: {gap_percent:.1f}%'}
        
        # Gap continuation validation
        if gap_type == 'GAP_UP':
            if signal_type == 'BUY' and intraday_momentum == 'BULLISH':
                return {'valid': True, 'reason': 'Valid gap up continuation'}
            elif signal_type == 'SELL' and intraday_momentum == 'BEARISH':
                return {'valid': True, 'reason': 'Valid gap up reversal'}
            else:
                return {'valid': False, 'reason': 'Gap up signal mismatch'}
        
        elif gap_type == 'GAP_DOWN':
            if signal_type == 'SELL' and intraday_momentum == 'BEARISH':
                return {'valid': True, 'reason': 'Valid gap down continuation'}
            elif signal_type == 'BUY' and intraday_momentum == 'BULLISH':
                return {'valid': True, 'reason': 'Valid gap down reversal'}
            else:
                return {'valid': False, 'reason': 'Gap down signal mismatch'}
        
        return {'valid': True, 'reason': 'No gap strategy applied'}
    
    def _process_and_send_signal(self, ticker: str, analysis_result: Dict) -> bool:
        """ENHANCED signal processing with better automation integration and validation"""

        try:
            # VALIDATE analysis_result structure
            if not analysis_result or not isinstance(analysis_result, dict):
                logger.error(f"Invalid analysis_result for {ticker}: {type(analysis_result)}")
                return False

            if 'trade_recommendation' not in analysis_result:
                logger.error(f"Missing trade_recommendation in analysis for {ticker}")
                return False

            trade_rec = analysis_result['trade_recommendation']

            if not isinstance(trade_rec, dict):
                logger.error(f"Invalid trade_recommendation type for {ticker}: {type(trade_rec)}")
                return False

            confidence = trade_rec.get('confidence', 0)

            if confidence <= 0 or confidence > 1:
                logger.error(f"Invalid confidence value for {ticker}: {confidence}")
                return False
            
            # Determine signal strength
            if confidence >= self.config.HIGH_CONFIDENCE_THRESHOLD:
                signal_strength = "🚨 STRONG SIGNAL"
                stars = "[STAR][STAR][STAR][STAR][STAR]"
                self.performance_stats['high_confidence_signals'] += 1
            elif confidence >= 0.70:
                signal_strength = "[BELL] MEDIUM SIGNAL"
                stars = "[STAR][STAR][STAR][STAR]"
            else:
                signal_strength = "[WARNING] WEAK SIGNAL"
                stars = "[STAR][STAR][STAR]"
            
            # Track trading style
            trading_style = analysis_result.get('trade_type', 'SWING')
            if trading_style == 'INTRADAY':
                self.performance_stats['intraday_signals'] += 1
            else:
                self.performance_stats['swing_signals'] += 1
            
            # Format and send the signal
            signal_message = self._format_signal_message(
                ticker, signal_strength, stars, trade_rec, analysis_result
            )
            
            # Send the signal
            message_sent = self.telegram_bot.send_trade_signal(signal_message)
            
            if message_sent:
                # Thread-safe state updates
                with self._state_lock:
                    # **NEW: Record the signal**
                    self.signal_filter.record_signal(ticker, analysis_result)

                    # Update counters
                    self.daily_signals_sent += 1

                    # Add to active positions tracking
                    position_info = {
                        'ticker': ticker,
                        'strategy': trade_rec.get('strategy', 'Unknown'),
                        'confidence': confidence,
                        'entry_time': datetime.now(self.timezone),
                        'trading_style': trading_style,
                        'option_legs': trade_rec.get('option_legs', []),
                        'expected_hold_days': self._get_avg_hold_time(trade_rec.get('strategy', '')),
                        'analysis_result': analysis_result
                    }

                    self.active_positions.append(position_info)

                    # Update performance metrics
                    self.performance_stats['signals_sent_today'] += 1

                # Log successful signal (outside lock)
                logger.info(f"[OK] Signal sent for {ticker}: {trade_rec.get('strategy')} ({confidence:.0%})")
                
                # CRITICAL FIX 3: Save signal for main database
                try:
                    signal_id = self.trade_logger.log_signal(ticker, analysis_result, source='signal_generator')
                    logger.info(f"[CHART] Signal logged to database (ID: {signal_id})")
                except Exception as log_error:
                    logger.error(f"Database logging error: {log_error}")
                    # Don't fail the main process if logging fails
                
                # CRITICAL FIX 4: Save signal for automation bot with better error handling
                try:
                    automation_saved = self._save_signal_for_automation(ticker, analysis_result)
                    if automation_saved:
                        logger.info(f"🤖 Signal queued for automation bot: {ticker}")
                        self.performance_stats['automation_signals_sent'] += 1
                    else:
                        logger.warning(f"[WARNING] Failed to queue signal for automation: {ticker}")
                except Exception as automation_error:
                    logger.error(f"Automation integration error for {ticker}: {automation_error}")
                    # Don't fail the main signal processing if automation fails
                
                return True
            else:
                logger.warning(f"Failed to send signal for {ticker}")
                return False
            
        except Exception as e:
            logger.error(f"Error processing signal for {ticker}: {e}", exc_info=True)
            self.performance_stats['errors_today'] += 1
            return False
    
    async def execute_trade(self, ticker: str) -> bool:
        """Execute actual trades using the enhanced analyzer"""
        
        try:
            logger.info(f"Executing trade for {ticker}...")
            
            # Analyze with execution enabled
            result = await self.options_analyzer.analyze_trade(
                symbol=ticker,
                trading_style='intraday' if self._is_market_open() else 'swing',
                capital=self.config.ACCOUNT_SIZE,
                execute_trades=True  # Enable actual execution
            )
            
            if result and not result.get('error'):
                execution_results = result.get('execution_results', [])
                successful_orders = [r for r in execution_results if r.get('status') == 'success']
                
                if successful_orders:
                    self.telegram_bot.send_message(
                        f"[ROCKET] TRADE EXECUTED - {ticker}\n"
                        f"Orders placed: {len(successful_orders)}\n"
                        f"Strategy: {result['trade_recommendation']['strategy']}"
                    )
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return False

    async def monitor_positions(self):
        """Monitor positions using enhanced analyzer"""
        
        try:
            position_status = await self.options_analyzer.monitor_positions()
            
            if not position_status.get('error'):
                positions = position_status.get('positions', {})
                net_positions = positions.get('net_positions', [])
                
                if net_positions:
                    summary = f"📍 {len(net_positions)} Active Positions:\n"
                    for pos in net_positions[:3]:
                        summary += f"• {pos['tradingsymbol']}: ₹{pos['pnl']:+.0f} P&L\n"
                    
                    self.telegram_bot.send_message(summary, silent=True)
        
        except Exception as e:
            logger.error(f"Position monitoring error: {e}")
        
        
    def _format_signal_message(self, ticker: str, signal_strength: str, stars: str,
                     trade_rec: Dict, analysis_result: Dict) -> str:
        """Enhanced signal formatting with complete trade setup details and technical analysis"""
        
        # Extract from new structure
        market_data = analysis_result.get('market_data', {})
        zerodha_integration = analysis_result.get('zerodha_integration', {})
        portfolio_impact = analysis_result.get('portfolio_impact', {})
        technical_analysis = analysis_result.get('technical_analysis', {})
        
        current_price = market_data.get('current_price', 0)
        trading_style = analysis_result.get('trade_type', 'SWING')
        
        lines = [
            f"{signal_strength} - {ticker} ({trading_style})",
            f"[CHART] Strategy: {trade_rec.get('strategy', 'Unknown').replace('_', ' ')}",
            f"💪 Confidence: {trade_rec.get('confidence', 0):.0%} {stars}",
            f"[MONEY] Current Price: ₹{current_price:.2f}",
            ""
        ]
        
        # NEW: Today's Context for Intraday Trading
        if technical_analysis and trading_style == 'INTRADAY':
            today_context = technical_analysis.get('today_context', {})
            if today_context:
                gap_type = today_context.get('gap_type', 'NONE')
                gap_percent = today_context.get('gap_percent', 0)
                intraday_momentum = today_context.get('intraday_momentum', 'NEUTRAL')
                intraday_change = today_context.get('intraday_change_percent', 0)
                session_phase = today_context.get('session_phase', 'MID_SESSION')
                volume_ratio = today_context.get('volume_ratio', 1.0)
                
                # Add today's action summary
                gap_emoji = "[UP]" if gap_type == 'GAP_UP' else "[DOWN]" if gap_type == 'GAP_DOWN' else "➡️"
                momentum_emoji = "[ROCKET]" if intraday_momentum == 'BULLISH' else "[DOWN]" if intraday_momentum == 'BEARISH' else "🔄"
                volume_emoji = "🔥" if volume_ratio > 1.5 else "[CHART]" if volume_ratio > 1.2 else "[DOWN]"
                
                lines.extend([
                    "🌅 TODAY'S ACTION:",
                    f"• Opening: {gap_emoji} {gap_type.replace('_', ' ')} ({gap_percent:+.1f}%)" if gap_type != 'NONE' else "• Opening: ➡️ Normal open",
                    f"• Momentum: {momentum_emoji} {intraday_momentum} ({intraday_change:+.1f}%)",
                    f"• Volume: {volume_emoji} {volume_ratio:.1f}x avg volume",
                    f"• Session: {session_phase.replace('_', ' ')}",
                    ""
                ])
                
                # Add gap strategy context if applicable
                if gap_type != 'NONE':
                    gap_fill_target = today_context.get('gap_fill_target', current_price)
                    gap_extension_target = today_context.get('gap_extension_target', current_price)
                    
                    lines.extend([
                        "[TARGET] GAP STRATEGY LEVELS:",
                        f"• Gap Fill Target: ₹{gap_fill_target:.0f}",
                        f"• Gap Extension Target: ₹{gap_extension_target:.0f}",
                        ""
                    ])
        
        # ENHANCED: Add technical analysis summary at the top
        if technical_analysis:
            market_bias = technical_analysis.get('market_bias', 'NEUTRAL')
            confidence_score = technical_analysis.get('confidence_score', 0)
            entry_signal = technical_analysis.get('entry_signal', {})
            
            bias_emoji = (
                "[ROCKET]" if market_bias == 'STRONG_BULLISH' else
                "🐂" if market_bias in ['BULLISH', 'LEAN_BULLISH'] else
                "[DOWN]" if market_bias == 'STRONG_BEARISH' else
                "🐻" if market_bias in ['BEARISH', 'LEAN_BEARISH'] else
                "🦆"  # NEUTRAL
            )
            
            lines.extend([
                "[UP] TECHNICAL SUMMARY:",
                f"• Market Bias: {bias_emoji} {market_bias} ({confidence_score:.0%} confidence)",
                f"• Entry Signal: {entry_signal.get('signal_type', 'HOLD')} - {entry_signal.get('reason', 'No clear setup')[:50]}",
                f"• Support: ₹{technical_analysis.get('support_resistance', {}).get('nearest_support', current_price * 0.95):.0f}",
                f"• Resistance: ₹{technical_analysis.get('support_resistance', {}).get('nearest_resistance', current_price * 1.05):.0f}",
                ""
            ])
            
            # NEW: Enhanced entry condition with session timing
            entry_condition = entry_signal.get('entry_condition')
            if entry_condition and entry_condition != 'Wait for confirmation':
                # Add session-specific timing advice
                current_time = datetime.now()
                if trading_style == 'INTRADAY':
                    if current_time.hour == 9:
                        timing_advice = " (Opening volatility - be quick)"
                    elif 10 <= current_time.hour <= 14:
                        timing_advice = " (Good liquidity window)"
                    elif current_time.hour >= 14:
                        timing_advice = " (Power hour - high risk/reward)"
                    else:
                        timing_advice = ""
                        
                    lines.append(f"[TARGET] Entry Condition: {entry_condition}{timing_advice}")
                else:
                    lines.append(f"[TARGET] Entry Condition: {entry_condition}")
                lines.append("")
        
        lines.append("[TARGET] COMPLETE TRADE SETUP:")
        
        # Get option legs from the analysis result
        option_legs = trade_rec.get('option_legs', [])
        
        total_premium = 0
        total_margin = 0
        all_strikes = []
        
        if option_legs:
            
            
            for i, leg in enumerate(option_legs, 1):
                action = leg.get('action', 'BUY')
                option_type = leg.get('option_type', 'call').upper()
                strike = leg.get('strike', 0)
                contracts = leg.get('contracts', 0)
                lot_size = leg.get('lot_size', 1)  # Default to 1 if not found
                quantity = contracts * lot_size
                premium = leg.get('theoretical_price', 0)
                tradingsymbol = leg.get('tradingsymbol', f"{ticker}28OCT{int(strike)}{option_type[0]}E")
                
                # NEW: Add liquidity and edge scores if available
                liquidity_score = leg.get('liquidity_score', 0)
                edge_score = leg.get('edge_score', 0)
                
                all_strikes.append(strike)
                
                # Calculate individual cost
                individual_cost = premium * quantity
                total_premium += individual_cost if action == 'BUY' else -individual_cost
                
                # Add leg details with enhanced info
                action_emoji = "🟢" if action == "BUY" else "🔴"
                quality_emoji = "[GEM]" if liquidity_score > 0.7 else "[STAR]" if liquidity_score > 0.4 else "[WARNING]"
                
                lines.append(
                    f"Leg {i}: {action_emoji} {action} {leg.get('contracts', 0)} lots ({quantity} shares) {option_type} @ ₹{strike} {quality_emoji}"
                )
                lines.append(
                    f"       Symbol: {tradingsymbol}"
                )
                lines.append(
                    f"       Premium: ₹{premium:.2f} x {quantity} = ₹{individual_cost:.0f}"
                )
                
                # Add quality metrics if available
                if liquidity_score > 0 or edge_score > 0:
                    lines.append(
                        f"       Quality: Liquidity {liquidity_score:.1f}/1.0, Edge {edge_score:.1f}/1.0"
                    )
                
                lines.append("")
            
            # Add totals with enhanced risk metrics
            lines.extend([
                "💵 TRADE FINANCIALS:",
                f"• Total Premium: ₹{abs(total_premium):,.0f}",
                f"• Max Risk: ₹{abs(total_premium):,.0f}" if total_premium > 0 else f"• Credit Received: ₹{abs(total_premium):,.0f}",
            ])
            
            # NEW: Add margin efficiency if available
            zerodha_exec = trade_rec.get('zerodha_execution', {})
            if zerodha_exec:
                estimated_margin = zerodha_exec.get('estimated_margin', 0)
                if estimated_margin > 0:
                    margin_efficiency = (abs(total_premium) / estimated_margin) * 100
                    lines.append(f"• Margin Efficiency: {margin_efficiency:.0f}%")
            
            lines.append("")
            
            # Strategy-specific profit/loss levels (keeping your existing logic but enhanced)
            strategy_name = trade_rec.get('strategy', '')
            
            if 'STRADDLE' in strategy_name or 'STRANGLE' in strategy_name:
                if 'LONG' in strategy_name:
                    if len(option_legs) >= 2:
                        if 'STRADDLE' in strategy_name:
                            center_strike = all_strikes[0]
                            total_cost = abs(total_premium)
                            total_quantity = option_legs[0].get('total_quantity', 1)
                            cost_per_share = total_cost / total_quantity
                            
                            breakeven_upper = center_strike + cost_per_share
                            breakeven_lower = center_strike - cost_per_share
                            move_required = (cost_per_share / current_price) * 100
                            
                            # NEW: Add probability assessment
                            prob_assessment = "High" if move_required < 3 else "Medium" if move_required < 5 else "Low"
                            
                        else:  # STRANGLE
                            upper_strike = max(all_strikes)
                            lower_strike = min(all_strikes)
                            total_cost = abs(total_premium)
                            total_quantity = option_legs[0].get('total_quantity', 1)
                            cost_per_share = total_cost / total_quantity
                            
                            breakeven_upper = upper_strike + cost_per_share
                            breakeven_lower = lower_strike - cost_per_share
                            
                            move_up_required = ((breakeven_upper - current_price) / current_price) * 100
                            move_down_required = ((current_price - breakeven_lower) / current_price) * 100
                            move_required = min(abs(move_up_required), abs(move_down_required))
                            
                            prob_assessment = "High" if move_required < 4 else "Medium" if move_required < 6 else "Low"
                        
                        lines.extend([
                            "[UP] PROFIT/LOSS LEVELS:",
                            f"• Breakeven Upper: ₹{breakeven_upper:.0f}",
                            f"• Breakeven Lower: ₹{breakeven_lower:.0f}",
                            f"• Max Loss: ₹{total_cost:.0f} (between strikes)",
                            f"• Profit if move > {move_required:.1f}% ({prob_assessment} probability)",
                            ""
                        ])
                else:  # SHORT STRADDLE/STRANGLE
                    center_strike = sum(all_strikes) / len(all_strikes)
                    credit_received = abs(total_premium)
                    total_quantity = option_legs[0].get('total_quantity', 1)
                    credit_per_share = credit_received / total_quantity
                    
                    profit_zone_upper = center_strike + credit_per_share
                    profit_zone_lower = center_strike - credit_per_share
                    
                    # NEW: Add range probability
                    range_size = ((profit_zone_upper - profit_zone_lower) / current_price) * 100
                    range_prob = "High" if range_size > 8 else "Medium" if range_size > 5 else "Low"
                    
                    lines.extend([
                        "[UP] PROFIT/LOSS LEVELS:",
                        f"• Max Profit: ₹{credit_received:.0f} (if stays between strikes)",
                        f"• Profit Zone: ₹{profit_zone_lower:.0f} to ₹{profit_zone_upper:.0f} ({range_size:.1f}% range)",
                        f"• Range Hold Probability: {range_prob}",
                        f"• Break-even Points: ₹{profit_zone_lower:.0f} and ₹{profit_zone_upper:.0f}",
                        ""
                    ])
            
            elif 'SPREAD' in strategy_name:
                if len(all_strikes) >= 2:
                    strikes = sorted(all_strikes)
                    spread_width = strikes[-1] - strikes[0]
                    total_quantity = option_legs[0].get('total_quantity', 1)
                    
                    if 'CALL' in strategy_name:
                        max_profit = (spread_width * total_quantity) - abs(total_premium)
                        breakeven = strikes[0] + (abs(total_premium) / total_quantity)
                        
                        # NEW: Add success probability
                        distance_to_target = ((strikes[-1] - current_price) / current_price) * 100
                        success_prob = "High" if distance_to_target < 3 else "Medium" if distance_to_target < 6 else "Low"
                        
                        lines.extend([
                            "[UP] PROFIT/LOSS LEVELS:",
                            f"• Max Profit: ₹{max_profit:.0f} (if above ₹{strikes[-1]}) - {success_prob} probability",
                            f"• Max Loss: ₹{abs(total_premium):.0f} (if below ₹{strikes[0]})",
                            f"• Breakeven: ₹{breakeven:.0f}",
                            f"• Spread Width: ₹{spread_width}",
                            ""
                        ])
                    else:  # PUT SPREAD
                        max_profit = (spread_width * total_quantity) - abs(total_premium)
                        breakeven = strikes[-1] - (abs(total_premium) / total_quantity)
                        
                        distance_to_target = ((current_price - strikes[0]) / current_price) * 100
                        success_prob = "High" if distance_to_target < 3 else "Medium" if distance_to_target < 6 else "Low"
                        
                        lines.extend([
                            "[UP] PROFIT/LOSS LEVELS:",
                            f"• Max Profit: ₹{max_profit:.0f} (if below ₹{strikes[0]}) - {success_prob} probability",
                            f"• Max Loss: ₹{abs(total_premium):.0f} (if above ₹{strikes[-1]})",
                            f"• Breakeven: ₹{breakeven:.0f}",
                            f"• Spread Width: ₹{spread_width}",
                            ""
                        ])
            
            elif len(option_legs) == 1:
                # Single option with enhanced probability analysis
                leg = option_legs[0]
                strike = leg.get('strike', 0)
                action = leg.get('action', 'BUY')
                option_type = leg.get('option_type', 'call')
                cost = abs(total_premium)
                
                if action == 'BUY':
                    if option_type.lower() == 'call':
                        breakeven = strike + (cost / leg.get('total_quantity', 1))
                        distance_to_breakeven = ((breakeven - current_price) / current_price) * 100
                        
                        lines.extend([
                            "[UP] PROFIT/LOSS LEVELS:",
                            f"• Breakeven: ₹{breakeven:.0f} ({distance_to_breakeven:+.1f}% from current)",
                            f"• Max Loss: ₹{cost:.0f} (premium paid)",
                            f"• Profit if above: ₹{breakeven:.0f}",
                            f"• Unlimited upside potential",
                            ""
                        ])
                    else:  # put
                        breakeven = strike - (cost / leg.get('total_quantity', 1))
                        distance_to_breakeven = ((current_price - breakeven) / current_price) * 100
                        
                        lines.extend([
                            "[UP] PROFIT/LOSS LEVELS:",
                            f"• Breakeven: ₹{breakeven:.0f} ({distance_to_breakeven:+.1f}% from current)",
                            f"• Max Loss: ₹{cost:.0f} (premium paid)",
                            f"• Profit if below: ₹{breakeven:.0f}",
                            f"• Max Profit: ₹{(strike * leg.get('total_quantity', 1)) - cost:.0f} (if stock goes to zero)",
                            ""
                        ])
        
        # ENHANCED: Smart entry/exit rules using technical analysis with session timing
        lines.extend([
            "⚡ ENTRY/EXIT RULES:",
        ])
        
        # Use technical analysis for smarter entry/exit rules
        if technical_analysis and technical_analysis.get('entry_signal'):
            entry_signal = technical_analysis['entry_signal']
            exit_rules = trade_rec.get('exit_rules', {})
            
            # Get technical levels
            support = technical_analysis.get('support_resistance', {}).get('nearest_support', current_price * 0.95)
            resistance = technical_analysis.get('support_resistance', {}).get('nearest_resistance', current_price * 1.05)
            
            # NEW: Session-aware entry/exit timing
            current_time = datetime.now()
            
            if trading_style == 'INTRADAY':
                profit_targets = exit_rules.get('profit_targets', [f"10% profit (₹{abs(total_premium) * 1.10:.0f})"])
                stop_losses = exit_rules.get('stop_losses', [f"15% loss (₹{abs(total_premium) * 0.85:.0f})"])
                
                # Add session-specific advice
                if current_time.hour == 9:
                    session_advice = "Opening session - Quick moves possible"
                elif 10 <= current_time.hour <= 14:
                    session_advice = "Mid-session - Steady trends"
                elif current_time.hour >= 14:
                    session_advice = "Power hour - High volatility"
                else:
                    session_advice = "Standard session"
                
                lines.extend([
                    f"• Entry: {entry_signal.get('reason', 'Current level looks good')[:60]}",
                    f"• Session: {session_advice}",
                    f"• Target: {profit_targets[0] if profit_targets else '10% profit'}",
                    f"• Stop Loss: {stop_losses[0] if stop_losses else '15% loss'}",
                    f"• Technical Stop: Below ₹{support:.0f} support",
                    f"• Time Stop: Exit by 2:45 PM",
                    f"• Risk-Reward: {entry_signal.get('risk_reward_ratio', 1.5):.1f}:1"
                ])
            else:
                profit_targets = exit_rules.get('profit_targets', [f"15% profit (₹{abs(total_premium) * 1.15:.0f})"])
                stop_losses = exit_rules.get('stop_losses', [f"20% loss (₹{abs(total_premium) * 0.80:.0f})"])
                
                lines.extend([
                    f"• Entry: {entry_signal.get('reason', 'Current level looks good')[:60]}",
                    f"• Target 1: {profit_targets[0] if profit_targets else '15% profit'}",
                    f"• Target 2: Resistance at ₹{resistance:.0f}",
                    f"• Stop Loss: {stop_losses[0] if stop_losses else '20% loss'}",
                    f"• Technical Stop: Below ₹{support:.0f} support",
                    f"• Time Frame: Hold for 7-14 days max"
                ])
        else:
            # Fallback logic (keeping your existing code)
            if trading_style == 'INTRADAY':
                lines.extend([
                    f"• Entry: Current price ₹{current_price:.2f} - Good entry level",
                    f"• Quick Target: 8-12% profit (₹{abs(total_premium) * 1.08:.0f} - ₹{abs(total_premium) * 1.12:.0f})",
                    f"• Maximum Target: 15% if major move (₹{abs(total_premium) * 1.15:.0f})",
                    f"• Stop Loss: 15% loss (₹{abs(total_premium) * 0.85:.0f})",
                    f"• Time Stop: Review at 1:30 PM, Exit by 2:45 PM"
                ])
            else:
                lines.extend([
                    f"• Entry: Current price ₹{current_price:.2f} - Monitor for confirmation",
                    f"• Target 1: 15% profit (₹{abs(total_premium) * 1.15:.0f})",
                    f"• Target 2: 25% profit (₹{abs(total_premium) * 1.25:.0f})",
                    f"• Stop Loss: 20% loss (₹{abs(total_premium) * 0.80:.0f})",
                    f"• Time Frame: Hold for 7-14 days max"
                ])
        
        lines.append("")
        
        # ENHANCED: Better technical analysis section using actual data
        if technical_analysis:
            lines.append("[UP] DETAILED TECHNICAL ANALYSIS:")
            
            # Trend analysis with intraday vs daily alignment
            trend_analysis = technical_analysis.get('trend_analysis', {})
            if trend_analysis:
                daily_trend = trend_analysis.get('daily_trend', 'UNKNOWN')
                intraday_trend = trend_analysis.get('intraday_trend', 'UNKNOWN')
                trend_strength = trend_analysis.get('trend_strength', 0)
                alignment = trend_analysis.get('intraday_vs_daily_alignment', 'NEUTRAL')
                
                lines.append(f"• Daily Trend: {daily_trend.replace('_', ' ')} (Strength: {trend_strength:.1f})")
                if trading_style == 'INTRADAY':
                    lines.append(f"• Intraday Trend: {intraday_trend} (Alignment: {alignment.replace('_', ' ')})")
            
            # Momentum analysis with intraday RSI
            momentum_analysis = technical_analysis.get('momentum_analysis', {})
            if momentum_analysis:
                momentum_direction = momentum_analysis.get('direction', 'NEUTRAL')
                rsi = momentum_analysis.get('rsi', 50)
                intraday_rsi = momentum_analysis.get('intraday_rsi', 50)
                
                lines.append(f"• Momentum: {momentum_direction} (Daily RSI: {rsi:.0f})")
                if trading_style == 'INTRADAY' and intraday_rsi != rsi:
                    lines.append(f"• Intraday RSI: {intraday_rsi:.0f}")
            
            # Pattern signals
            pattern_signals = technical_analysis.get('pattern_signals', {})
            if pattern_signals and pattern_signals.get('detected_patterns'):
                strongest_pattern = pattern_signals.get('strongest_pattern')
                if strongest_pattern:
                    consolidation = pattern_signals.get('consolidation', False)
                    lines.append(f"• Pattern: {strongest_pattern.replace('_', ' ')}")
                    if consolidation:
                        lines.append("• Market: Consolidating (breakout pending)")
            
            # Support/Resistance with distances
            support_resistance = technical_analysis.get('support_resistance', {})
            if support_resistance:
                current_level = support_resistance.get('current_level', 'UNKNOWN')
                level_strength = support_resistance.get('level_strength', 0)
                support_distance = support_resistance.get('support_distance_pct', 0)
                resistance_distance = support_resistance.get('resistance_distance_pct', 0)
                
                lines.append(f"• Position: {current_level.replace('_', ' ')} (Strength: {level_strength:.1f})")
                lines.append(f"• Support Distance: {support_distance:.1f}% | Resistance: {resistance_distance:.1f}%")
            
            lines.append("")
        
        # Portfolio Greeks (keeping your existing code)
        if portfolio_impact and portfolio_impact.get('greeks'):
            greeks = portfolio_impact['greeks']
            lines.extend([
                "[CHART] PORTFOLIO GREEKS:",
                f"• Delta: {greeks.get('delta', 0):.2f} (directional risk)",
                f"• Gamma: {greeks.get('gamma', 0):.3f} (acceleration risk)",
                f"• Theta: ₹{greeks.get('theta', 0):.0f}/day (time decay)",
                f"• Vega: ₹{greeks.get('vega', 0):.0f} (volatility risk)",
                ""
            ])
        
        # Data source and execution info (enhanced)
        data_source = zerodha_integration.get('market_data_source', 'unknown')
        technical_available = zerodha_integration.get('technical_analysis_available', False)
        zerodha_exec = trade_rec.get('zerodha_execution', {})
        
        lines.extend([
            "[WRENCH] EXECUTION INFO:",
            f"[SIGNAL] Data: {'Live Zerodha [OK]' if data_source == 'zerodha_live' else 'Fallback [WARNING]'}",
            f"[UP] Technical: {'Enhanced [OK]' if technical_available else 'Basic [WARNING]'}",
        ])
        
        if zerodha_exec:
            estimated_margin = zerodha_exec.get('estimated_margin', 0)
            execution_ready = zerodha_exec.get('execution_ready', False)
            total_lots = zerodha_exec.get('total_lots', 0)
            
            lines.extend([
                f"[MONEY] Estimated Margin: ₹{estimated_margin:,.0f}",
                f"📦 Total Lots: {total_lots}",
                f"[OK] Execution Ready: {'Yes' if execution_ready else 'No'}"
            ])
        
        # Enhanced risk warning with session-specific advice
        if trading_style == 'INTRADAY':
            current_time = datetime.now()
            if current_time.hour >= 14:
                time_warning = "⏰ LATE SESSION - Extra caution advised"
            elif current_time.hour == 9:
                time_warning = "🌅 OPENING SESSION - High volatility expected"
            else:
                time_warning = "⏰ MID SESSION - Normal volatility"
            
            lines.extend([
                "",
                "[WARNING] INTRADAY WARNING:",
                f"• {time_warning}",
                "• Monitor positions closely",
                "• Auto square-off at 2:45 PM",
                "• Time decay accelerates rapidly",
                "• Keep stop-losses tight"
            ])
        
        message = "\n".join(lines)
        
        # Escape Telegram markdown special characters (simplified)
        message = message.replace('_', '\\_').replace('*', '\\*').replace('`', '\\`')
        
        return message
    
    def _save_signal_for_automation(self, ticker: str, analysis_result: Dict):
        """Save signal for automation bot to pick up - NEW METHOD"""
        
        try:
            trade_rec = analysis_result['trade_recommendation']
            
            # Determine direction from strategy
            strategy = trade_rec.get('strategy', '')
            direction = 'bullish' if any(word in strategy.upper() for word in ['BULLISH', 'CALL', 'BUY']) else 'bearish'
            
            signal_data = {
                'ticker': ticker,
                'direction': direction,
                'confidence': trade_rec.get('confidence', 0),
                'strategy': strategy,
                'current_price': analysis_result.get('market_data', {}).get('current_price', 0),
                'timestamp': datetime.now().isoformat(),
                'full_analysis': analysis_result
            }
            
            # Save to database for automation bot
            automation_signal_id = self.trade_logger.save_automation_signal(signal_data)
            
            logger.info(f"💾 Signal saved for automation: {ticker} {direction} (ID: {automation_signal_id})")
            
            return automation_signal_id > 0
            
        except Exception as e:
            logger.error(f"Error saving signal for automation: {e}")
            return False
        
    def _get_automation_integration_stats(self) -> Dict:
        """Get automation integration statistics"""
        
        try:
            automation_stats = self.trade_logger.get_automation_stats()
            
            return {
                'signals_for_automation': automation_stats.get('total_signals', 0),
                'automation_processed': automation_stats.get('processed_signals', 0),
                'automation_pending': automation_stats.get('pending_signals', 0),
                'processing_rate': automation_stats.get('processing_rate', 0)
            }
        except Exception as e:
            logger.error(f"Error getting automation stats: {e}")
            return {
                'signals_for_automation': 0,
                'automation_processed': 0,
                'automation_pending': 0,
                'processing_rate': 0
            }

    
    def post_market_summary(self):
        """Post-market analysis and daily summary (3:35 PM)"""
        
        logger.info("🌇 Starting post-market summary...")
        
        try:
            # Get market closing data
            closing_data = self._get_market_closing_data()
            
            # Get daily statistics
            daily_stats = self.trade_logger.get_daily_summary()
            
            # Analyze top movers
            top_movers = self._analyze_top_movers()
            
            # Build summary message
            summary_lines = [
                "🌇 MARKET CLOSE SUMMARY",
                f"📅 {datetime.now(self.timezone).strftime('%d %B %Y')}",
                "",
                "[CHART] Market Performance:",
                f"• Nifty 50: {closing_data['nifty_close']} ({closing_data['nifty_change']:+.2f}%)",
                f"• Bank Nifty: {closing_data['banknifty_close']} ({closing_data['banknifty_change']:+.2f}%)",
                f"• India VIX: {closing_data['vix_close']:.2f} ({closing_data['vix_change']:+.2f}%)",
                "",
                "[UP] Bot Performance Today:",
                f"• Signals Sent: {self.performance_stats['signals_sent_today']}",
                f"• High Confidence: {self.performance_stats['high_confidence_signals']}",
                f"• Scan Cycles: {self.performance_stats['scan_cycles_completed']}",
                f"• Errors: {self.performance_stats['errors_today']}",
                ""
            ]
            
            # Add top signals of the day
            if daily_stats.get('top_signals'):
                summary_lines.extend([
                    "[TARGET] Top Signals Today:"
                ])
                for signal in daily_stats['top_signals'][:3]:
                    summary_lines.append(
                        f"• {signal['ticker']} - {signal['strategy']} ({signal['confidence']:.0%})"
                    )
                summary_lines.append("")
            
            # Add market movers
            if top_movers['gainers']:
                summary_lines.append("[UP] Top Gainers: " + ", ".join(
                    [f"{m['ticker']} ({m['change']:+.1f}%)" for m in top_movers['gainers'][:3]]
                ))
            
            if top_movers['losers']:
                summary_lines.append("[DOWN] Top Losers: " + ", ".join(
                    [f"{m['ticker']} ({m['change']:.1f}%)" for m in top_movers['losers'][:3]]
                ))
            
            # Add sector performance
            sector_perf = self._analyze_sector_performance()
            if sector_perf:
                summary_lines.extend([
                    "",
                    "🏭 Sector Performance:",
                    f"• Best: {sector_perf['best_sector']} ({sector_perf['best_change']:+.1f}%)",
                    f"• Worst: {sector_perf['worst_sector']} ({sector_perf['worst_change']:.1f}%)"
                ])
            
            # Add learning insights
            learning_insights = daily_stats.get('learning_insights', {})
            if learning_insights:
                summary_lines.extend([
                    "",
                    "[BULB] Today's Insights:",
                    f"• Most successful pattern: {learning_insights.get('best_pattern', 'N/A')}",
                    f"• Optimal hold time: {learning_insights.get('optimal_hold_days', 'N/A')} days",
                    f"• Best strategy: {learning_insights.get('best_strategy', 'N/A')}"
                ])
            
            # Add tomorrow's outlook
            tomorrow_outlook = self._generate_tomorrow_outlook()
            summary_lines.extend([
                "",
                "🔮 Tomorrow's Outlook:",
                tomorrow_outlook,
                "",
                "🤖 Bot performance: All systems operational",
                f"⏰ Next scan: Tomorrow {self.config.PRE_MARKET_START.strftime('%H:%M')} IST",
                "",
                "Good night! 🌙"
            ])
            
            # Send summary
            self.telegram_bot.send_message("\n".join(summary_lines))
            
            # Save daily report
            self._save_daily_report(daily_stats, closing_data)
            
            # Reset market status
            self.market_status = 'CLOSED'
            
            logger.info("[OK] Post-market summary completed")
            
        except Exception as e:
            logger.error(f"Post-market summary error: {e}", exc_info=True)
            self.telegram_bot.send_error_alert(f"Post-market summary failed: {str(e)}")
   
    async def manual_analysis(self, ticker: str) -> Optional[Dict]:
        """Manually analyze a specific ticker on demand - SENDS FULL SIGNAL MESSAGE"""
        
        logger.info(f"🔍 Manual analysis requested for {ticker}")
        
        try:
            # Validate ticker exists in Zerodha
            zerodha_symbol = self._get_zerodha_symbol(ticker)
            instrument_info = self.zerodha_client.get_instrument_info(zerodha_symbol)
            if not instrument_info:
                logger.warning(f"{ticker} not found in Zerodha instruments")
            
            # Run analysis with await since _analyze_stock is async
            analysis_result = await self._analyze_stock(ticker)
            
            if analysis_result:
                trade_rec = analysis_result.get('trade_recommendation', {})
                confidence = trade_rec.get('confidence', 0)
                
                # Send detailed analysis regardless of threshold
                if self._should_send_signal(analysis_result, ticker):
                    # If signal meets threshold, send via normal process
                    self._process_and_send_signal(ticker, analysis_result)
                    
                    # Send additional manual analysis note
                    self.telegram_bot.send_message(
                        f"📋 Manual Analysis Complete for {ticker}\n"
                        f"[OK] Signal met all quality thresholds and was sent above.\n"
                        f"[TARGET] Confidence: {confidence:.1%}"
                    )
                else:
                    # FIXED: Send the COMPLETE signal message even if below threshold
                    
                    # Determine signal strength (same logic as _process_and_send_signal)
                    if confidence >= self.config.HIGH_CONFIDENCE_THRESHOLD:
                        signal_strength = "🚨 STRONG SIGNAL (Manual)"
                        stars = "[STAR][STAR][STAR][STAR][STAR]"
                    elif confidence >= 0.70:
                        signal_strength = "[BELL] MEDIUM SIGNAL (Manual)"
                        stars = "[STAR][STAR][STAR][STAR]"
                    else:
                        signal_strength = "[WARNING] WEAK SIGNAL (Manual)"
                        stars = "[STAR][STAR][STAR]"
                    
                    # Generate the COMPLETE signal message using the same formatter
                    full_signal_message = self._format_signal_message(
                        ticker, signal_strength, stars, trade_rec, analysis_result
                    )
                    
                    # Add manual analysis header
                    manual_header = [
                        "🔍 MANUAL ANALYSIS RESULT",
                        f"[WARNING] Below Bot Threshold ({self.config.MIN_CONFIDENCE_SCORE:.0%}) - Manual Override",
                        "=" * 50,
                        ""
                    ]
                    
                    complete_message = "\n".join(manual_header) + full_signal_message
                    
                    # Add footer explaining why it was below threshold
                    footer = [
                        "",
                        "=" * 50,
                        "📋 MANUAL ANALYSIS NOTES:",
                        f"• Bot Threshold: {self.config.MIN_CONFIDENCE_SCORE:.0%}",
                        f"• Signal Confidence: {confidence:.0%}",
                        f"• Status: Below automatic threshold",
                        f"• Action: Manual review recommended",
                        "",
                        "[BULB] This signal would NOT be sent automatically.",
                        "Use your discretion for manual execution."
                    ]
                    
                    complete_message += "\n".join(footer)
                    
                    # Send the complete message
                    self.telegram_bot.send_trade_signal(complete_message)
                    
                    logger.info(f"📋 Manual analysis complete for {ticker} - Full signal sent (below threshold)")
                
                return analysis_result
            else:
                self.telegram_bot.send_message(f"[ERROR] Could not analyze {ticker}")
                return None
                
        except Exception as e:
            logger.error(f"Manual analysis error: {e}", exc_info=True)
            self.telegram_bot.send_error_alert(f"Manual analysis failed: {str(e)}")
            return None
   
   # === Helper Methods ===
   
    def _is_market_open(self) -> bool:
       """Check if Indian markets are currently open"""
       
       now = datetime.now(self.timezone)
       
       # Check if weekday (Monday = 0, Sunday = 6)
       if now.weekday() >= 5:  # Saturday or Sunday
           return False
       
       # Check if market holiday
       if self._is_market_holiday(now.date()):
           return False
       
       # Check market hours
       current_time = now.time()
       return self.config.MARKET_OPEN <= current_time <= self.config.MARKET_CLOSE
   
    def _is_market_holiday(self, date) -> bool:
       """Check if given date is an NSE holiday"""
       
       # This should ideally check against NSE holiday calendar
       # For now, simplified implementation
       holidays_2025 = [
           '2025-01-26',  # Republic Day
           '2025-03-17',  # Holi
           '2025-04-14',  # Ambedkar Jayanti
           '2025-04-18',  # Good Friday
           '2025-05-01',  # Maharashtra Day
           '2025-08-15',  # Independence Day
           '2025-10-02',  # Gandhi Jayanti
           '2025-10-24',  # Dussehra
           '2025-11-12',  # Diwali
           '2025-11-26',  # Guru Nanak Jayanti
       ]
       
       return str(date) in holidays_2025
   
    def _check_market_status(self) -> str:
       """Check current market status using Zerodha API"""
       
       # Use Zerodha's market status check
       market_status = self.zerodha_client.get_market_status()
       
       if market_status.get('status') == 'open':
           return 'OPEN'
       elif market_status.get('reason') == 'weekend':
           return 'WEEKEND'
       elif market_status.get('reason') == 'holiday':
           return 'HOLIDAY'
       elif market_status.get('reason') == 'pre-market':
           return 'PRE_MARKET'
       else:
           return 'CLOSED'
   
    def _analyze_global_markets(self) -> Dict:
       """Analyze global market sentiment"""
       
       # This would fetch real data from APIs
       # Simplified for now
       return {
           'summary': 'Mixed sentiment',
           'us_close': 'S&P 500 +0.3%, Nasdaq +0.5%',
           'asia_status': 'Nikkei +0.8%, Hang Seng -0.2%',
           'sgx_nifty': '24,850 (+0.2%)',
           'dollar_index': '104.5',
           'crude_oil': '$78.5',
           'sentiment_score': 0.6
       }
   
    def _check_market_events(self) -> List[str]:
       """Check for important market events today"""
       
       events = []
       today = datetime.now(self.timezone).date()
       
       # Check for RBI policy
       if today.day in [6, 7, 8] and today.month in [2, 4, 6, 8, 10, 12]:
           events.append("RBI Policy")
       
       # Check for monthly expiry (last Thursday)
       if today.weekday() == 3:  # Thursday
           # Check if last Thursday of month
           next_week = today + timedelta(days=7)
           if next_week.month != today.month:
               events.append("Monthly Expiry")
       
       # Check for results season
       if today.month in [1, 4, 7, 10] and 10 <= today.day <= 25:
           events.append("Earnings Season")
       
       return events
   
    async def _analyze_overnight_gaps(self) -> Dict:
        """Enhanced overnight gap analysis using technical analyzer with actionable insights"""
        
        try:
            gaps = {
                'gap_ups': [],
                'gap_downs': [],
                'nifty_support': 24500,
                'nifty_resistance': 25000,
                'banknifty_support': 56000,
                'banknifty_resistance': 57000,
                # NEW: Enhanced gap analysis data
                'significant_gaps': [],
                'gap_strategies': [],
                'market_sentiment': 'NEUTRAL',
                'gap_quality_score': 0.0
            }
            
            gap_analysis_results = []
            total_gap_score = 0
            significant_gap_count = 0
            
            # **Enhanced gap analysis with technical insights**
            for ticker in self.config.INDIAN_WATCHLIST[:8]:  # Expand to top 8 stocks
                try:
                    zerodha_symbol = self._get_zerodha_symbol(ticker)
                    
                    # Get historical data for gap calculation
                    data = self.zerodha_client.get_historical_data(zerodha_symbol, 'day', 3)  # Get 3 days for better context
                    
                    if isinstance(data, pd.DataFrame) and len(data) >= 2:
                        yesterday_close = data.iloc[-2]['close']
                        today_open = data.iloc[-1]['open']
                        current_price = data.iloc[-1]['close']  # Use latest available price
                        
                        # Calculate gap
                        gap_pct = ((today_open - yesterday_close) / yesterday_close) * 100
                        
                        # **NEW: Use technical analyzer for enhanced gap analysis**
                        if abs(gap_pct) > 0.3:  # Only analyze meaningful gaps
                            try:
                                # Get market data for technical analysis
                                market_data = await self.market_data_provider.fetch_live_data(ticker)
                                
                                # Use quick intraday analysis for gap insights
                                gap_technical_analysis = await self.technical_analyzer.quick_intraday_analysis(
                                    ticker, current_price, market_data
                                )
                                
                                # Extract enhanced gap information
                                gap_info = gap_technical_analysis.get('gap_analysis', {})
                                momentum_info = gap_technical_analysis.get('momentum_immediate', {})
                                
                                # Create enhanced gap entry
                                enhanced_gap = {
                                    'ticker': ticker,
                                    'gap_percent': gap_pct,
                                    'gap_type': gap_info.get('gap_type', 'GAP_UP' if gap_pct > 0 else 'GAP_DOWN'),
                                    'gap_significance': gap_info.get('gap_significance', self._classify_gap_significance(abs(gap_pct))),
                                    'yesterday_close': yesterday_close,
                                    'today_open': today_open,
                                    'current_price': current_price,
                                    'momentum_direction': momentum_info.get('direction', 'NEUTRAL'),
                                    'volume_factor': momentum_info.get('volume_factor', 1.0),
                                    'confidence': gap_technical_analysis.get('confidence', 0.5),
                                    'recommended_action': gap_technical_analysis.get('recommended_action', 'WAIT'),
                                    # NEW: Gap strategy analysis
                                    'gap_fill_probability': self._calculate_gap_fill_probability(gap_pct, ticker, momentum_info),
                                    'continuation_probability': self._calculate_continuation_probability(gap_pct, momentum_info),
                                    'optimal_strategy': self._determine_gap_strategy(gap_pct, momentum_info, gap_technical_analysis),
                                    'risk_level': self._assess_gap_risk(gap_pct, momentum_info),
                                    'targets': {
                                        'gap_fill': yesterday_close,
                                        'extension_25': today_open + (today_open - yesterday_close) * 0.25,
                                        'extension_50': today_open + (today_open - yesterday_close) * 0.50,
                                        'extension_100': today_open + (today_open - yesterday_close) * 1.0
                                    }
                                }
                                
                                gap_analysis_results.append(enhanced_gap)
                                
                                # Calculate gap quality score
                                gap_score = self._calculate_gap_quality_score(enhanced_gap)
                                total_gap_score += gap_score
                                
                                # Classify as significant if high quality
                                if gap_score > 0.6 or abs(gap_pct) > 2.0:
                                    significant_gap_count += 1
                                    gaps['significant_gaps'].append(enhanced_gap)
                                
                                # **Traditional gap classification for backward compatibility**
                                if gap_pct > 1.0:
                                    confidence_indicator = "[TARGET]" if enhanced_gap['confidence'] > 0.7 else "[WARNING]" if enhanced_gap['confidence'] > 0.5 else "[?]"
                                    gaps['gap_ups'].append(f"{ticker} ({gap_pct:+.1f}%) {confidence_indicator}")
                                elif gap_pct < -1.0:
                                    confidence_indicator = "[TARGET]" if enhanced_gap['confidence'] > 0.7 else "[WARNING]" if enhanced_gap['confidence'] > 0.5 else "[?]"
                                    gaps['gap_downs'].append(f"{ticker} ({gap_pct:.1f}%) {confidence_indicator}")
                                
                            except Exception as tech_error:
                                logger.warning(f"Technical gap analysis failed for {ticker}: {tech_error}")
                                
                                # Fallback to basic gap classification
                                if gap_pct > 1.0:
                                    gaps['gap_ups'].append(f"{ticker} ({gap_pct:+.1f}%)")
                                elif gap_pct < -1.0:
                                    gaps['gap_downs'].append(f"{ticker} ({gap_pct:.1f}%)")
                    
                except Exception as ticker_error:
                    logger.warning(f"Gap analysis failed for {ticker}: {ticker_error}")
                    continue
            
            # **NEW: Enhanced support/resistance using technical analysis**
            await self._get_enhanced_support_resistance(gaps)
            
            # **NEW: Generate gap trading strategies**
            gaps['gap_strategies'] = self._generate_gap_strategies(gap_analysis_results)
            
            # **NEW: Determine overall market sentiment from gaps**
            gaps['market_sentiment'] = self._determine_gap_sentiment(gap_analysis_results)
            
            # **NEW: Calculate overall gap quality score**
            gaps['gap_quality_score'] = total_gap_score / max(len(gap_analysis_results), 1)
            
            # **NEW: Add gap statistics**
            gaps['gap_statistics'] = {
                'total_gaps_analyzed': len(gap_analysis_results),
                'significant_gaps': significant_gap_count,
                'avg_gap_size': sum(abs(g['gap_percent']) for g in gap_analysis_results) / max(len(gap_analysis_results), 1),
                'max_gap_up': max([g['gap_percent'] for g in gap_analysis_results if g['gap_percent'] > 0], default=0),
                'max_gap_down': min([g['gap_percent'] for g in gap_analysis_results if g['gap_percent'] < 0], default=0),
                'high_confidence_gaps': len([g for g in gap_analysis_results if g['confidence'] > 0.7])
            }
            
            # **NEW: Gap trading recommendations**
            gaps['trading_recommendations'] = self._generate_gap_trading_recommendations(gap_analysis_results)
            
            logger.info(f"[OK] Enhanced gap analysis completed: {len(gap_analysis_results)} gaps analyzed, {significant_gap_count} significant")
            
            return gaps
            
        except Exception as e:
            logger.error(f"Enhanced gap analysis error: {e}", exc_info=True)
            return {
                'gap_ups': [],
                'gap_downs': [],
                'nifty_support': 24500,
                'nifty_resistance': 25000,
                'banknifty_support': 56000,
                'banknifty_resistance': 57000,
                'significant_gaps': [],
                'gap_strategies': [],
                'market_sentiment': 'NEUTRAL',
                'gap_quality_score': 0.0,
                'gap_statistics': {},
                'trading_recommendations': []
            }

    async def _get_enhanced_support_resistance(self, gaps: Dict):
        """Get enhanced support/resistance levels using technical analysis"""
        
        try:
            # Analyze NIFTY for enhanced levels
            nifty_market_data = await self.market_data_provider.fetch_live_data('NIFTY')
            nifty_price = nifty_market_data['current_price']
            
            try:
                nifty_technical = await self.technical_analyzer.analyze_symbol_for_options(
                    'NIFTY', nifty_price, nifty_market_data, 'intraday'
                )
                
                support_resistance = nifty_technical.get('support_resistance', {})
                if support_resistance:
                    gaps['nifty_support'] = support_resistance.get('nearest_support', nifty_price * 0.98)
                    gaps['nifty_resistance'] = support_resistance.get('nearest_resistance', nifty_price * 1.02)
                    gaps['nifty_trend'] = nifty_technical.get('market_bias', 'NEUTRAL')
                    gaps['nifty_trend_strength'] = nifty_technical.get('trend_analysis', {}).get('trend_strength', 0.5)
            except:
                gaps['nifty_support'] = nifty_price * 0.98
                gaps['nifty_resistance'] = nifty_price * 1.02
            
            # Analyze BANKNIFTY for enhanced levels
            bn_market_data = await self.market_data_provider.fetch_live_data('BANKNIFTY')
            bn_price = bn_market_data['current_price']
            
            try:
                bn_technical = await self.technical_analyzer.analyze_symbol_for_options(
                    'BANKNIFTY', bn_price, bn_market_data, 'intraday'
                )
                
                support_resistance = bn_technical.get('support_resistance', {})
                if support_resistance:
                    gaps['banknifty_support'] = support_resistance.get('nearest_support', bn_price * 0.98)
                    gaps['banknifty_resistance'] = support_resistance.get('nearest_resistance', bn_price * 1.02)
                    gaps['banknifty_trend'] = bn_technical.get('market_bias', 'NEUTRAL')
                    gaps['banknifty_trend_strength'] = bn_technical.get('trend_analysis', {}).get('trend_strength', 0.5)
            except:
                gaps['banknifty_support'] = bn_price * 0.98
                gaps['banknifty_resistance'] = bn_price * 1.02
                
        except Exception as e:
            logger.warning(f"Enhanced support/resistance calculation failed: {e}")

    def _classify_gap_significance(self, gap_percent: float) -> str:
        """Classify gap significance based on percentage"""
        
        if gap_percent > 3.0:
            return 'VERY_HIGH'
        elif gap_percent > 2.0:
            return 'HIGH'
        elif gap_percent > 1.0:
            return 'MODERATE'
        elif gap_percent > 0.5:
            return 'LOW'
        else:
            return 'MINIMAL'

    def _calculate_gap_fill_probability(self, gap_percent: float, ticker: str, momentum_info: Dict) -> float:
        """Calculate probability of gap fill based on historical patterns and momentum"""
        
        base_probability = 0.7  # Base 70% gap fill probability
        
        # Adjust based on gap size
        if abs(gap_percent) > 3.0:
            base_probability -= 0.2  # Large gaps less likely to fill quickly
        elif abs(gap_percent) > 2.0:
            base_probability -= 0.1
        elif abs(gap_percent) < 1.0:
            base_probability += 0.1  # Small gaps more likely to fill
        
        # Adjust based on momentum
        momentum_direction = momentum_info.get('direction', 'NEUTRAL')
        if gap_percent > 0 and momentum_direction == 'BEARISH':
            base_probability += 0.15  # Gap up with bearish momentum = higher fill probability
        elif gap_percent < 0 and momentum_direction == 'BULLISH':
            base_probability += 0.15  # Gap down with bullish momentum = higher fill probability
        elif gap_percent > 0 and momentum_direction == 'BULLISH':
            base_probability -= 0.1  # Gap up with bullish momentum = lower fill probability
        elif gap_percent < 0 and momentum_direction == 'BEARISH':
            base_probability -= 0.1  # Gap down with bearish momentum = lower fill probability
        
        # Adjust based on volume
        volume_factor = momentum_info.get('volume_factor', 1.0)
        if volume_factor > 1.5:
            base_probability += 0.05  # High volume supports the move
        
        # Index vs stock adjustment
        if ticker in ['NIFTY', 'BANKNIFTY']:
            base_probability += 0.05  # Indices more likely to fill gaps
        
        return max(0.1, min(0.95, base_probability))

    def _calculate_continuation_probability(self, gap_percent: float, momentum_info: Dict) -> float:
        """Calculate probability of gap continuation"""
        
        base_probability = 0.4  # Base 40% continuation probability
        
        # Adjust based on gap size
        if abs(gap_percent) > 2.0:
            base_probability += 0.2  # Large gaps more likely to continue
        elif abs(gap_percent) > 1.0:
            base_probability += 0.1
        
        # Adjust based on momentum alignment
        momentum_direction = momentum_info.get('direction', 'NEUTRAL')
        if gap_percent > 0 and momentum_direction == 'BULLISH':
            base_probability += 0.25  # Gap up with bullish momentum
        elif gap_percent < 0 and momentum_direction == 'BEARISH':
            base_probability += 0.25  # Gap down with bearish momentum
        
        # Volume confirmation
        volume_factor = momentum_info.get('volume_factor', 1.0)
        if volume_factor > 2.0:
            base_probability += 0.15  # Very high volume supports continuation
        elif volume_factor > 1.5:
            base_probability += 0.1
        
        return max(0.1, min(0.9, base_probability))

    def _determine_gap_strategy(self, gap_percent: float, momentum_info: Dict, technical_analysis: Dict) -> str:
        """Determine optimal gap trading strategy"""
        
        momentum_direction = momentum_info.get('direction', 'NEUTRAL')
        confidence = technical_analysis.get('confidence', 0.5)
        
        if abs(gap_percent) < 0.5:
            return 'NO_GAP_STRATEGY'
        
        if gap_percent > 0:  # Gap up
            if momentum_direction == 'BULLISH' and confidence > 0.7:
                return 'GAP_UP_CONTINUATION_AGGRESSIVE'
            elif momentum_direction == 'BULLISH':
                return 'GAP_UP_CONTINUATION_CONSERVATIVE'
            elif momentum_direction == 'BEARISH':
                return 'GAP_UP_REVERSAL'
            else:
                return 'GAP_UP_RANGE_PLAY'
        
        else:  # Gap down
            if momentum_direction == 'BEARISH' and confidence > 0.7:
                return 'GAP_DOWN_CONTINUATION_AGGRESSIVE'
            elif momentum_direction == 'BEARISH':
                return 'GAP_DOWN_CONTINUATION_CONSERVATIVE'
            elif momentum_direction == 'BULLISH':
                return 'GAP_DOWN_REVERSAL'
            else:
                return 'GAP_DOWN_RANGE_PLAY'

    def _assess_gap_risk(self, gap_percent: float, momentum_info: Dict) -> str:
        """Assess risk level of gap trading"""
        
        risk_score = 0
        
        # Gap size risk
        if abs(gap_percent) > 3.0:
            risk_score += 3
        elif abs(gap_percent) > 2.0:
            risk_score += 2
        elif abs(gap_percent) > 1.0:
            risk_score += 1
        
        # Momentum alignment risk
        momentum_direction = momentum_info.get('direction', 'NEUTRAL')
        if momentum_direction == 'NEUTRAL':
            risk_score += 2  # Uncertain momentum = higher risk
        
        # Volume risk
        volume_factor = momentum_info.get('volume_factor', 1.0)
        if volume_factor < 0.8:
            risk_score += 1  # Low volume = higher risk
        
        if risk_score >= 4:
            return 'HIGH'
        elif risk_score >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _calculate_gap_quality_score(self, gap_data: Dict) -> float:
        """Calculate overall quality score for gap trading opportunity"""
        
        score = 0.0
        
        # Gap significance (30% weight)
        gap_percent = abs(gap_data['gap_percent'])
        if gap_percent > 2.0:
            score += 0.3
        elif gap_percent > 1.0:
            score += 0.2
        elif gap_percent > 0.5:
            score += 0.1
        
        # Momentum alignment (25% weight)
        momentum_direction = gap_data['momentum_direction']
        gap_direction = 'BULLISH' if gap_data['gap_percent'] > 0 else 'BEARISH'
        
        if momentum_direction == gap_direction:
            score += 0.25  # Perfect alignment
        elif momentum_direction == 'NEUTRAL':
            score += 0.1   # Neutral momentum
        # No score for opposing momentum
        
        # Technical confidence (20% weight)
        confidence = gap_data['confidence']
        score += confidence * 0.2
        
        # Volume confirmation (15% weight)
        volume_factor = gap_data['volume_factor']
        if volume_factor > 2.0:
            score += 0.15
        elif volume_factor > 1.5:
            score += 0.1
        elif volume_factor > 1.2:
            score += 0.05
        
        # Risk assessment (10% weight)
        risk_level = gap_data['risk_level']
        if risk_level == 'LOW':
            score += 0.1
        elif risk_level == 'MEDIUM':
            score += 0.05
        # No score for high risk
        
        return min(1.0, score)

    def _generate_gap_strategies(self, gap_results: List[Dict]) -> List[Dict]:
        """Generate specific gap trading strategies"""
        
        strategies = []
        
        for gap in gap_results:
            if gap['gap_quality_score'] > 0.6 or abs(gap['gap_percent']) > 2.0:
                strategy = {
                    'ticker': gap['ticker'],
                    'strategy_type': gap['optimal_strategy'],
                    'entry_level': gap['current_price'],
                    'targets': gap['targets'],
                    'stop_loss': self._calculate_gap_stop_loss(gap),
                    'risk_reward_ratio': self._calculate_gap_risk_reward(gap),
                    'confidence': gap['confidence'],
                    'expected_hold_time': self._estimate_gap_hold_time(gap),
                    'notes': self._generate_gap_strategy_notes(gap)
                }
                strategies.append(strategy)
        
        return strategies

    def _determine_gap_sentiment(self, gap_results: List[Dict]) -> str:
        """Determine overall market sentiment from gap analysis"""
        
        if not gap_results:
            return 'NEUTRAL'
        
        bullish_count = len([g for g in gap_results if g['gap_percent'] > 1.0])
        bearish_count = len([g for g in gap_results if g['gap_percent'] < -1.0])
        
        total_gap_bias = sum(g['gap_percent'] for g in gap_results)
        
        if bullish_count > bearish_count * 2 and total_gap_bias > 2.0:
            return 'STRONGLY_BULLISH'
        elif bullish_count > bearish_count and total_gap_bias > 1.0:
            return 'BULLISH'
        elif bearish_count > bullish_count * 2 and total_gap_bias < -2.0:
            return 'STRONGLY_BEARISH'
        elif bearish_count > bullish_count and total_gap_bias < -1.0:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def _generate_gap_trading_recommendations(self, gap_results: List[Dict]) -> List[str]:
        """Generate actionable gap trading recommendations"""
        
        recommendations = []
        
        high_quality_gaps = [g for g in gap_results if g.get('gap_quality_score', 0) > 0.7]
        
        if not high_quality_gaps:
            recommendations.append("No high-quality gap setups detected")
            return recommendations
        
        for gap in high_quality_gaps[:3]:  # Top 3 recommendations
            if 'CONTINUATION' in gap['optimal_strategy']:
                recommendations.append(
                    f"{gap['ticker']}: Trade gap continuation - "
                    f"Target ₹{gap['targets']['extension_25']:.0f} with SL at ₹{gap['yesterday_close']:.0f}"
                )
            elif 'REVERSAL' in gap['optimal_strategy']:
                recommendations.append(
                    f"{gap['ticker']}: Trade gap reversal - "
                    f"Target ₹{gap['targets']['gap_fill']:.0f} with SL at ₹{gap['targets']['extension_25']:.0f}"
                )
            else:
                recommendations.append(
                    f"{gap['ticker']}: Monitor for {gap['optimal_strategy'].lower().replace('_', ' ')}"
                )
        
        return recommendations

    def _calculate_gap_stop_loss(self, gap_data: Dict) -> float:
        """Calculate appropriate stop loss for gap trade"""
        
        if 'CONTINUATION' in gap_data['optimal_strategy']:
            return gap_data['yesterday_close']  # Stop at gap fill level
        elif 'REVERSAL' in gap_data['optimal_strategy']:
            return gap_data['targets']['extension_25']  # Stop at 25% extension
        else:
            return gap_data['current_price'] * 0.98  # 2% stop loss

    def _calculate_gap_risk_reward(self, gap_data: Dict) -> float:
        """Calculate risk-reward ratio for gap trade"""
        
        entry = gap_data['current_price']
        stop_loss = self._calculate_gap_stop_loss(gap_data)
        
        if 'CONTINUATION' in gap_data['optimal_strategy']:
            target = gap_data['targets']['extension_25']
        else:
            target = gap_data['targets']['gap_fill']
        
        risk = abs(entry - stop_loss)
        reward = abs(target - entry)
        
        return reward / risk if risk > 0 else 0

    def _estimate_gap_hold_time(self, gap_data: Dict) -> str:
        """Estimate optimal hold time for gap trade"""
        
        if abs(gap_data['gap_percent']) > 2.0:
            return "2-4 hours"  # Large gaps may take longer to resolve
        elif gap_data['momentum_direction'] != 'NEUTRAL':
            return "30-90 minutes"  # Strong momentum trades
        else:
            return "1-2 hours"  # Standard gap trades

    def _generate_gap_strategy_notes(self, gap_data: Dict) -> str:
        """Generate strategy-specific notes for gap trade"""
        
        notes = []
        
        if gap_data['volume_factor'] > 2.0:
            notes.append("High volume confirms gap")
        
        if gap_data['confidence'] > 0.8:
            notes.append("High technical confidence")
        
        if gap_data['risk_level'] == 'HIGH':
            notes.append("Use smaller position size")
        
        if gap_data['gap_fill_probability'] > 0.8:
            notes.append("High gap fill probability")
        
        return " | ".join(notes) if notes else "Standard gap trade"
   
    def _calculate_indian_margin(self, trade_rec: Dict, current_price: float, lot_size: int) -> float:
       """Calculate margin requirements for Indian markets using Zerodha margin calculator"""
       
       try:
           # Format positions for Zerodha margin calculator
           positions = []
           
           for leg in trade_rec.get('options_legs', []):
               positions.append({
                   'exchange': 'NFO',
                   'tradingsymbol': leg.get('ticker', trade_rec.get('ticker')),
                   'transaction_type': leg['action'],
                   'quantity': lot_size,
                   'product': 'MIS',  # Intraday
                   'order_type': 'LIMIT',
                   'price': leg.get('theoretical_price', 0)
               })
           
           # Get margin from Zerodha
           try:
               margin_data = self.zerodha_client.get_order_margins(positions)
               if margin_data:
                   return margin_data.get('total', 0)
           except:
               pass
           
           # Fallback calculation
           strategy = trade_rec.get('primary_strategy', '')
           legs = trade_rec.get('options_legs', [])
           
           if not legs:
               return 0
           
           # For option buying
           if all(leg['action'] == 'BUY' for leg in legs):
               # Premium only
               return sum(leg['theoretical_price'] * lot_size for leg in legs)
           
           # For spreads
           elif 'SPREAD' in strategy:
               # Margin = difference in strikes × lot size
               strikes = [leg['strike'] for leg in legs]
               if len(strikes) >= 2:
                   spread_width = abs(max(strikes) - min(strikes))
                   return spread_width * lot_size
           
           # For naked selling (simplified)
           else:
               # Approximate 15% of notional
               return current_price * lot_size * 0.15
                   
       except Exception as e:
           logger.error(f"Margin calculation error: {e}")
           # Return simplified calculation
           return current_price * lot_size * 0.15
   
    def _get_portfolio_state(self) -> Dict:
       """Get current portfolio state for risk management from Zerodha"""
       
       try:
           # Get positions from Zerodha
           positions = self.zerodha_client.get_positions()
           holdings = self.zerodha_client.get_holdings()
           
           # Aggregate Greeks from active positions
           portfolio = {
               'total_delta': 0,
               'total_gamma': 0,
               'total_theta': 0,
               'total_vega': 0,
               'positions': self.active_positions,
               'zerodha_positions': positions,
               'zerodha_holdings': holdings,
               'total_value': self.config.ACCOUNT_SIZE
           }
           
           # Calculate Greeks if available in positions
           net_positions = positions.get('net', [])
           for pos in net_positions:
               portfolio['total_delta'] += pos.get('delta', 0) * pos.get('quantity', 0)
               portfolio['total_gamma'] += pos.get('gamma', 0) * pos.get('quantity', 0)
               portfolio['total_theta'] += pos.get('theta', 0) * pos.get('quantity', 0)
               portfolio['total_vega'] += pos.get('vega', 0) * pos.get('quantity', 0)
           
           return portfolio
           
       except Exception as e:
           logger.error(f"Portfolio state error: {e}")
           return {
               'total_delta': 0,
               'total_gamma': 0,
               'total_theta': 0,
               'total_vega': 0,
               'positions': self.active_positions,
               'total_value': self.config.ACCOUNT_SIZE
           }
   
    def _send_scan_summary(self, signals_found: int, errors: List[str], 
                          duration: float, scan_results: List[Dict]):
        """Enhanced scan summary with filtering stats"""
        
        if signals_found > 0 or errors or len(errors) > 2:  # Only send if significant activity
            
            # Get signal filter stats
            filter_stats = self.signal_filter.get_signal_stats()
            
            summary_parts = [
                f"🔍 Smart Scan Complete ({datetime.now(self.timezone).strftime('%H:%M')})",
                f"⏱️ Duration: {duration:.1f}s",
                f"[CHART] Signals Sent: {signals_found}",
                f"[TARGET] Avg Confidence: {filter_stats.get('avg_confidence', 0):.0%}",
            ]
            
            # Add filtering effectiveness
            signals_today = filter_stats.get('signals_today', 0)
            if signals_today > 0:
                summary_parts.append(f"[UP] Today's Signals: {signals_today} (Quality filtered)")
                
                most_common = filter_stats.get('most_common_strategy', 'None')
                if most_common != 'None':
                    summary_parts.append(f"🔥 Top Strategy: {most_common.replace('_', ' ')}")
        
        if signals_found > 0 or errors:
            summary_parts = [
                f"🔍 Enhanced Scan Complete ({datetime.now(self.timezone).strftime('%H:%M')})",
                f"⏱️ Duration: {duration:.1f}s",
                f"[CHART] Total Signals: {signals_found}",
            ]
            
            # Add enhanced metrics
            if signals_found > 0:
                intraday_count = self.performance_stats.get('intraday_signals', 0)
                swing_count = self.performance_stats.get('swing_signals', 0)
                high_conf_count = self.performance_stats.get('high_confidence_signals', 0)
                
                summary_parts.extend([
                    f"[UP] Intraday: {intraday_count} | Swing: {swing_count}",
                    f"[STAR] High Confidence: {high_conf_count}"
                ])
            
            if scan_results:
                summary_parts.append("\n[TARGET] Signals Found:")
                for result in scan_results[:5]:  # Top 5
                    # Get trading style from result
                    trading_style = result.get('trading_style', 'SWING')
                    style_emoji = "⚡" if trading_style == 'INTRADAY' else "[CHART]"
                    
                    summary_parts.append(
                        f"• {style_emoji} {result['ticker']}: {result['strategy']} "
                        f"({result['confidence']:.0%})"
                    )
            
            # Add daily performance summary
            daily_total = self.performance_stats.get('signals_sent_today', 0)
            scan_cycles = self.performance_stats.get('scan_cycles_completed', 0)
            
            summary_parts.extend([
                f"\n📋 Today's Stats:",
                f"• Total Signals: {daily_total}/{self.config.MAX_SIGNALS_PER_DAY}",
                f"• Active Positions: {len(self.active_positions)}/{self.config.MAX_POSITIONS_OPEN}",
                f"• Scan Cycles: {scan_cycles}"
            ])
            
            if errors:
                summary_parts.append(f"\n[WARNING] Errors: {len(errors)}")
                for error in errors[:3]:  # First 3 errors
                    summary_parts.append(f"• {error}")
            
            # Add analyzer version info (only if significant signals)
            if signals_found > 0 or len(errors) > 2:
                self.telegram_bot.send_message("\n".join(summary_parts), silent=True)
   
    def _get_market_closing_data(self) -> Dict:
       """Get market closing data from Zerodha"""
       
       try:
           # Get index data using Zerodha
           quotes = self.zerodha_client.get_live_quotes(['NIFTY 50', 'NIFTY BANK'])
           
           return {
               'nifty_close': quotes.get('NIFTY 50', {}).get('price', 24800),
               'nifty_change': quotes.get('NIFTY 50', {}).get('change_percent', 0),
               'banknifty_close': quotes.get('NIFTY BANK', {}).get('price', 56000),
               'banknifty_change': quotes.get('NIFTY BANK', {}).get('change_percent', 0),
               'vix_close': 15.0,  # Would fetch actual VIX
               'vix_change': 0.5
           }
       except:
           return {
               'nifty_close': 24800,
               'nifty_change': 0,
               'banknifty_close': 56000,
               'banknifty_change': 0,
               'vix_close': 15.0,
               'vix_change': 0
           }
   
    def _analyze_top_movers(self) -> Dict:
       """Analyze top gainers and losers using Zerodha data"""
       
       movers = {'gainers': [], 'losers': []}
       
       try:
           # Check all watchlist stocks
           zerodha_symbols = [self._get_zerodha_symbol(ticker) for ticker in self.config.INDIAN_WATCHLIST]
           quotes = self.zerodha_client.get_live_quotes(zerodha_symbols)
           
           changes = []
           for ticker, data in quotes.items():
               change = data.get('change_percent', 0)
               changes.append({'ticker': ticker, 'change': change})
           
           # Sort by change
           changes.sort(key=lambda x: x['change'], reverse=True)
           
           # Get top 3 gainers and losers
           movers['gainers'] = [c for c in changes if c['change'] > 0][:3]
           movers['losers'] = [c for c in changes if c['change'] < 0][:3]
           
       except Exception as e:
           logger.error(f"Top movers analysis error: {e}")
       
       return movers
   
    def _analyze_sector_performance(self) -> Dict:
       """Analyze sector performance"""
       
       # Simplified sector mapping
       sectors = {
            'IT': ['TCS', 'INFY'],
            'Banking': ['HDFCBANK'],
            'Energy': ['RELIANCE'],
            'NBFC': ['BAJFINANCE'],
            'Auto': ['MARUTI'],
            'Indices': ['NIFTY 50']
        }
       
       sector_performance = {}
       
       try:
           for sector, stocks in sectors.items():
               quotes = self.zerodha_client.get_live_quotes(stocks)
               
               changes = []
               for stock, data in quotes.items():
                   changes.append(data.get('change_percent', 0))
               
               if changes:
                   sector_performance[sector] = sum(changes) / len(changes)
           
           if sector_performance:
               best_sector = max(sector_performance.items(), key=lambda x: x[1])
               worst_sector = min(sector_performance.items(), key=lambda x: x[1])
               
               return {
                   'best_sector': best_sector[0],
                   'best_change': best_sector[1],
                   'worst_sector': worst_sector[0],
                   'worst_change': worst_sector[1]
               }
       except:
           pass
       
       return {}
   
    def _generate_tomorrow_outlook(self) -> str:
       """Generate outlook for next trading day"""
       
       # Check global cues
       global_sentiment = self._analyze_global_markets()
       sentiment_score = global_sentiment.get('sentiment_score', 0.5)
       
       if sentiment_score > 0.7:
           return "Positive global cues, expect gap-up opening"
       elif sentiment_score < 0.3:
           return "Negative global sentiment, cautious approach recommended"
       else:
           return "Mixed global cues, stock-specific opportunities likely"
   
    def _save_daily_report(self, daily_stats: Dict, closing_data: Dict):
       """Save daily trading report"""
       
       try:
           report = {
               'date': datetime.now(self.timezone).strftime('%Y-%m-%d'),
               'market_data': closing_data,
               'bot_performance': self.performance_stats,
               'signals_detail': daily_stats,
               'active_positions': len(self.active_positions),
               'api_provider': 'Zerodha Kite Connect'
           }
           
           filename = f"daily_report_{report['date']}.json"
           with open(filename, 'w') as f:
               json.dump(report, f, indent=2)
           
           logger.info(f"Daily report saved: {filename}")
           
       except Exception as e:
           logger.error(f"Failed to save daily report: {e}")
   
    def _get_pattern_success_rate(self, pattern: str) -> float:
       """Get historical success rate for pattern"""
       
       # This would query historical performance
       # Simplified for now
       pattern_success = {
           'ascending_triangle': 0.72,
           'descending_triangle': 0.68,
           'cup_and_handle': 0.78,
           'head_and_shoulders': 0.70,
           'double_top': 0.69,
           'double_bottom': 0.71,
           'bull_flag': 0.75,
           'bear_flag': 0.73
       }
       
       return pattern_success.get(pattern, 0.65)
   
    def _get_avg_hold_time(self, strategy: str) -> int:
       """Get average hold time for strategy"""
       
       # This would be calculated from historical data
       strategy_hold_times = {
           'BULLISH_CALL': 20,
           'BEARISH_PUT': 18,
           'BULLISH_CALL_SPREAD': 25,
           'BEARISH_PUT_SPREAD': 23,
           'IRON_CONDOR': 20,
           'STRADDLE': 12,
           'STRANGLE': 15,
           'CALENDAR_SPREAD': 22
       }
       
       return strategy_hold_times.get(strategy, 20)
   
    def _heartbeat_check(self):
        """Periodic health check"""
        
        try:
            # Synchronize market status with actual market hours
            expected_status = 'OPEN' if self._is_market_open() else 'CLOSED'
            
            if self.market_status != expected_status:
                logger.info(f"Updating market status from {self.market_status} to {expected_status}")
                self.market_status = expected_status
                
                # Send status update notification
                if expected_status == 'OPEN':
                    self.telegram_bot.send_message("[BELL] Market opened - Bot status synchronized", silent=True)
                else:
                    self.telegram_bot.send_message("🔕 Market closed - Bot status synchronized", silent=True)
            
            # Check for stuck positions
            stuck_positions = []
            for position in self.active_positions:
                hold_time = (datetime.now(self.timezone) - position['entry_time']).days
                if hold_time > 30:
                    logger.warning(f"Position {position['ticker']} held for {hold_time} days")
                    stuck_positions.append({
                        'ticker': position['ticker'],
                        'strategy': position['strategy'],
                        'hold_days': hold_time,
                        'confidence': position.get('confidence', 0)
                    })
            
            # Alert about stuck positions
            if stuck_positions:
                alert_msg = "[WARNING] Long-held positions detected:\n"
                for pos in stuck_positions[:3]:  # Show max 3
                    alert_msg += f"• {pos['ticker']}: {pos['hold_days']} days ({pos['strategy']})\n"
                
                if len(stuck_positions) > 3:
                    alert_msg += f"...and {len(stuck_positions) - 3} more"
                
                self.telegram_bot.send_message(alert_msg, silent=True)
            
            # Check API connectivity
            self._check_api_health()
            
            # Check system resources if needed
            self._check_system_health()
            
            # Log heartbeat with status
            current_time = datetime.now(self.timezone).strftime('%H:%M:%S')
            logger.debug(f"💓 Heartbeat - Bot alive | Status: {self.market_status} | Time: {current_time}")
            
        except Exception as e:
            logger.error(f"Heartbeat check error: {e}", exc_info=True)
            # Send critical error alert
            try:
                self.telegram_bot.send_error_alert(f"Heartbeat check failed: {str(e)}")
            except:
                pass  # Don't let telegram errors crash the heartbeat

    def _check_api_health(self):
        """Check API connectivity and health"""
        
        try:
            # Test Zerodha API connection
            test_result = self.zerodha_client.get_market_status()
            if not test_result:
                logger.warning("Zerodha API connectivity issue detected")
                self.performance_stats['api_issues'] = self.performance_stats.get('api_issues', 0) + 1
        
        except Exception as e:
            logger.warning(f"API health check failed: {e}")
            self.performance_stats['api_issues'] = self.performance_stats.get('api_issues', 0) + 1

    def _check_system_health(self):
        """Check system resources and performance"""
        
        try:
            import psutil
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                logger.warning(f"High memory usage: {memory.percent:.1f}%")
            
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.percent > 95:
                logger.warning(f"Low disk space: {disk.percent:.1f}% used")
                
        except ImportError:
            # psutil not available, skip system checks
            pass
        except Exception as e:
            logger.debug(f"System health check error: {e}")
   
    def _cleanup(self):
       """Cleanup on shutdown"""
       
       logger.info("Performing cleanup...")
       
       # Save state
       state = {
           'shutdown_time': datetime.now(self.timezone).isoformat(),
           'active_positions': self.active_positions,
           'performance_stats': self.performance_stats,
           'api_provider': 'Zerodha Kite Connect'
       }
       
       try:
           with open('bot_state.json', 'w') as f:
               json.dump(state, f, indent=2)
           logger.info("State saved successfully")
       except Exception as e:
           logger.error(f"Failed to save state: {e}")
       
       # Send shutdown notification
       try:
           self.telegram_bot.send_message(
               "🛑 Bot shutting down\n"
               f"Active positions: {len(self.active_positions)}\n"
               "State saved successfully."
           )
       except:
           pass
   
    def _display_startup_info(self):
       """Display startup information"""
       
       info = f"""
╔══════════════════════════════════════════════════════╗
║         [IN] INDIAN TRADING BOT v2.0 [IN]              ║
║        Powered by Zerodha Kite Connect API           ║
╠══════════════════════════════════════════════════════╣
║ Watchlist: {len(self.config.INDIAN_WATCHLIST)} stocks                              ║
║ Account: ₹{self.config.ACCOUNT_SIZE:,.0f}                            ║
║ Risk Level: {self.config.RISK_TOLERANCE}                            ║
║ Min Confidence: {self.config.MIN_CONFIDENCE_SCORE:.0%}                      ║
║ Scan Interval: {self.config.SCAN_INTERVAL_MINUTES} minutes                    ║
╠══════════════════════════════════════════════════════╣
║ Market Hours: 9:15 AM - 3:30 PM IST                 ║
║ Pre-market: 9:00 AM | Post-market: 3:35 PM          ║
╚══════════════════════════════════════════════════════╝
       """
       print(info)


def main():
   """Main entry point"""
   
   print("[ROCKET] Starting Indian Trading Bot with Zerodha Kite Connect API...")
   print("=" * 60)
   
   try:
       # Check for required environment variables
       required_vars = []
       
       # Check for Zerodha authentication (need either access token OR api key+secret)
       if not os.getenv('ZERODHA_ACCESS_TOKEN'):
           if not (os.getenv('ZERODHA_API_KEY') and os.getenv('ZERODHA_API_SECRET')):
               print("[ERROR] Missing Zerodha authentication!")
               print("Please provide either:")
               print("  - ZERODHA_ACCESS_TOKEN")
               print("  - ZERODHA_API_KEY and ZERODHA_API_SECRET")
               print("\n[BULB] Run 'python get_access_token.py' to generate access token")
               return
       
       # Check other required vars
       required_vars.extend([
           'TELEGRAM_BOT_TOKEN',
           'TELEGRAM_CHAT_ID'
       ])
       
       missing_vars = [var for var in required_vars if not os.getenv(var)]
       
       if missing_vars:
           print(f"[ERROR] Missing required environment variables: {missing_vars}")
           print("Please set these in your .env file")
           return
       
       # Check if API key is provided (needed for initialization)
       if not os.getenv('ZERODHA_API_KEY'):
           print("[ERROR] ZERODHA_API_KEY is required even with access token")
           print("Please set ZERODHA_API_KEY in your .env file")
           return
       
       # Manual mode check
       if len(sys.argv) > 1:
           # Manual analysis mode
           ticker = sys.argv[1].upper()
           print(f"\n[CHART] Manual analysis mode for {ticker}")
           
           bot = IndianTradingBot()
           result = asyncio.run(bot.manual_analysis(ticker))
           
           if result:
               print(f"[OK] Analysis complete for {ticker}")
           else:
               print(f"[ERROR] Analysis failed for {ticker}")
       
       else:
           # Automated mode
           print("[WRENCH] Initializing bot components...")
           bot = IndianTradingBot()
           
           print("[CHART] Testing Zerodha API connection...")
           # Test API connection
           market_status = bot.zerodha_client.get_market_status()
           if market_status:
               print(f"[OK] Zerodha API connected - Market status: {market_status.get('status', 'Unknown')}")
           else:
               print("[WARNING] Zerodha API connection issue detected")
           
           print("[ROCKET] Starting automated trading bot...")
           bot.start_bot()
   
   except Exception as e:
       print(f"\n[ERROR] Fatal error: {e}")
       logger.error(f"Bot failed to start: {e}", exc_info=True)
       
       # Additional help for common errors
       if "access_token" in str(e).lower():
           print("\n[BULB] Troubleshooting:")
           print("1. Check if your access token is valid and not expired")
           print("2. Run 'python get_access_token.py' to generate a new token")
           print("3. Ensure ZERODHA_ACCESS_TOKEN is set in your .env file")
       elif "api_key" in str(e).lower():
           print("\n[BULB] Troubleshooting:")
           print("1. Verify your ZERODHA_API_KEY in .env file")
           print("2. Check if your Zerodha Connect app is approved")
           print("3. Ensure you're using the correct API key from Zerodha Developer Console")
       
       sys.exit(1)


if __name__ == "__main__":
   main()