#!/usr/bin/env python3
"""
[ROCKET] Automated Indian Intraday Options Bot v2.0 - ENHANCED VERSION
Fully automated execution system for intraday options trading with intelligent risk management
Now with dynamic capital management, intelligent profit optimization, and multi-leg strategies
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import asyncio
import time
import logging
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import warnings
import sqlite3
import hashlib
import pickle
from contextlib import contextmanager
from collections import defaultdict, Counter
import pytz

# Import your existing components
from options_analyzer.brokers.zerodha_api_client import ZerodhaAPIClient
from options_analyzer.analyzers.options_analyzer import (
    ZerodhaEnhancedOptionsAnalyzer,
    ZerodhaMarketDataProvider,
    ZerodhaOptionsChainProvider,
    ZerodhaOrderManager,
    ZerodhaRiskManager,
    OptionsLeg
)
from options_analyzer.brokers.zerodha_technical_analyzer import ZerodhaTechnicalAnalyzer
from api.telegram.telegram_bot import TelegramSignalBot
from options_analyzer.indian_market.ind_trade_logger import IndianTradeLogger

warnings.filterwarnings('ignore')

# Load environment variables from project root
from dotenv import load_dotenv
dotenv_path = project_root / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automated_options_bot_v2.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ================== ENUMS & DATA CLASSES ==================

class TradingState(Enum):
    """Trading bot states"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    ENTERING = "entering" 
    MONITORING = "monitoring"
    OPTIMIZING = "optimizing"  # NEW: For profit optimization
    EXITING = "exiting"
    RECOVERY = "recovery"
    STOPPED = "stopped"

class ExitReason(Enum):
    """Reasons for position exit"""
    PROFIT_TARGET = "profit_target"
    PROFIT_OPTIMIZATION = "profit_optimization"  # NEW
    TECHNICAL_REVERSAL = "technical_reversal"
    SUPPORT_BREAK = "support_break"
    RESISTANCE_HIT = "resistance_hit"
    TIME_STOP = "time_stop"
    EMERGENCY_STOP = "emergency_stop"
    PATTERN_FAILURE = "pattern_failure"
    MANUAL_OVERRIDE = "manual_override"
    TRAILING_STOP = "trailing_stop"  # NEW

class CapitalTier(Enum):
    """Capital tier for strategy selection"""
    TIER_1 = 1  # Under ₹30,000 - Basic directional
    TIER_2 = 2  # ₹30,000 - ₹1,00,000 - Multi-leg buying only
    TIER_3 = 3  # Above ₹1,00,000 - Full arsenal including selling

class StrategyType(Enum):
    """Options strategy types"""
    # Tier 1 Strategies
    BUY_CALL = "buy_call"
    BUY_PUT = "buy_put"
    
    # Tier 2 Strategies (buying only)
    LONG_STRADDLE = "long_straddle"
    LONG_STRANGLE = "long_strangle"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    
    # Tier 3 Strategies (selling allowed)
    IRON_CONDOR = "iron_condor"
    CREDIT_SPREAD = "credit_spread"
    RATIO_SPREAD = "ratio_spread"
    CALENDAR_SPREAD = "calendar_spread"
    SHORT_STRADDLE = "short_straddle"
    SHORT_STRANGLE = "short_strangle"

@dataclass
class TradingSignal:
    """Enhanced signal structure from main bot"""
    ticker: str
    direction: str  # 'bullish', 'bearish', 'neutral'
    confidence: float
    strategy: str
    current_price: float
    timestamp: datetime
    source: str = 'main_bot'
    technical_confirmation: bool = False
    market_data: Dict = field(default_factory=dict)
    suggested_strategy_type: Optional[StrategyType] = None  # NEW

@dataclass 
class ActivePosition:
    """Enhanced active position tracking"""
    signal: TradingSignal
    option_legs: List[OptionsLeg]
    strategy_type: StrategyType  # NEW
    entry_time: datetime
    entry_price: float
    entry_premium: float
    current_pnl: float = 0.0
    max_profit: float = 0.0
    max_loss: float = 0.0
    order_ids: List[str] = field(default_factory=list)  # NEW: Multiple order IDs for multi-leg
    
    # Technical levels at entry
    entry_support: float = 0.0
    entry_resistance: float = 0.0
    
    # Risk management
    stop_loss: float = 0.0
    profit_target: float = 0.0
    trailing_stop: float = 0.0
    
    # NEW: Dynamic profit management
    profit_checkpoints_hit: List[float] = field(default_factory=list)
    last_profit_analysis: datetime = field(default_factory=datetime.now)
    profit_locked: float = 0.0  # Amount of profit locked in
    
    # Monitoring
    last_technical_check: datetime = field(default_factory=datetime.now)
    recovery_mode: bool = False
    exit_signals: List[str] = field(default_factory=list)
    
    # NEW: Position lifecycle tracking
    position_state: str = "active"  # active, optimizing, exiting
    re_entry_eligible: bool = False
    original_thesis: Dict = field(default_factory=dict)

@dataclass
class DailyPerformance:
    """Enhanced daily performance tracking"""
    date: str
    starting_capital: float = 5000.0
    current_capital: float = 5000.0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    trades_executed: int = 0
    trades_profitable: int = 0
    trades_loss: int = 0
    
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    
    # Risk tier tracking
    current_risk_tier: int = 1
    position_size_multiplier: float = 1.0
    
    signals_received: int = 0
    signals_acted_upon: int = 0
    signals_filtered: int = 0
    
    # NEW: Enhanced metrics
    capital_tier: CapitalTier = CapitalTier.TIER_1
    daily_profit_target: float = 0.08  # 8% minimum
    aggressive_profit_target: float = 0.15  # 15% aggressive
    profit_target_hit: bool = False
    max_profit_reached: float = 0.0
    
    # NEW: Learning metrics
    successful_patterns: List[str] = field(default_factory=list)
    failed_patterns: List[str] = field(default_factory=list)
    best_performing_strategy: Optional[StrategyType] = None

@dataclass
class TradeLifecycle:
    """NEW: Complete trade lifecycle documentation"""
    signal_id: int
    ticker: str
    strategy_type: StrategyType
    
    # Pre-trade
    signal_data: Dict = field(default_factory=dict)
    market_conditions: Dict = field(default_factory=dict)
    technical_setup: Dict = field(default_factory=dict)
    strategy_reasoning: str = ""
    
    # During trade
    entry_timestamp: datetime = field(default_factory=datetime.now)
    entry_prices: List[float] = field(default_factory=list)
    position_updates: List[Dict] = field(default_factory=list)
    decision_points: List[Dict] = field(default_factory=list)
    
    # Post-trade
    exit_timestamp: Optional[datetime] = None
    exit_prices: List[float] = field(default_factory=list)
    final_pnl: float = 0.0
    exit_reason: Optional[ExitReason] = None
    
    # Analysis
    what_worked: List[str] = field(default_factory=list)
    what_failed: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    
    # Metrics
    max_profit_reached: float = 0.0
    max_drawdown: float = 0.0
    time_in_position: Optional[timedelta] = None
    technical_accuracy: float = 0.0

# ================== ENHANCED COMPONENTS ==================

class DynamicCapitalManager:
    """NEW: Dynamic capital and tier management"""
    
    def __init__(self, zerodha_client: ZerodhaAPIClient):
        self.zerodha = zerodha_client
        self.current_tier = CapitalTier.TIER_1
        self.available_capital = 0.0
        self.last_update = None
        
    async def update_capital_status(self) -> Dict:
        """Fetch real-time capital and determine tier"""
        try:
            margins = self.zerodha.get_margins()
            
            if margins:
                # Get available cash from equity segment
                equity_margin = margins.get('equity', {})
                available_cash = equity_margin.get('available', {}).get('cash', 0)
                
                # Account for existing positions
                used_margin = equity_margin.get('used', {}).get('total', 0)
                net_available = available_cash - used_margin
                
                # Determine capital tier
                if net_available < 30000:
                    self.current_tier = CapitalTier.TIER_1
                elif net_available < 100000:
                    self.current_tier = CapitalTier.TIER_2
                else:
                    self.current_tier = CapitalTier.TIER_3
                
                self.available_capital = net_available
                self.last_update = datetime.now()
                
                return {
                    'available_capital': net_available,
                    'tier': self.current_tier,
                    'tier_name': self._get_tier_name(),
                    'allowed_strategies': self.get_allowed_strategies(),
                    'used_margin': used_margin,
                    'total_cash': available_cash
                }
            
            return {
                'available_capital': 0,
                'tier': CapitalTier.TIER_1,
                'error': 'Could not fetch margins'
            }
            
        except Exception as e:
            logger.error(f"Error updating capital status: {e}")
            return {
                'available_capital': self.available_capital,
                'tier': self.current_tier,
                'error': str(e)
            }
    
    def get_allowed_strategies(self) -> List[StrategyType]:
        """Get allowed strategies based on capital tier"""
        if self.current_tier == CapitalTier.TIER_1:
            return [StrategyType.BUY_CALL, StrategyType.BUY_PUT]
        elif self.current_tier == CapitalTier.TIER_2:
            return [
                StrategyType.BUY_CALL, StrategyType.BUY_PUT,
                StrategyType.LONG_STRADDLE, StrategyType.LONG_STRANGLE,
                StrategyType.BULL_CALL_SPREAD, StrategyType.BEAR_PUT_SPREAD
            ]
        else:  # TIER_3
            return list(StrategyType)  # All strategies allowed
    
    def _get_tier_name(self) -> str:
        """Get human-readable tier name"""
        tier_names = {
            CapitalTier.TIER_1: "Basic Directional (< ₹30K)",
            CapitalTier.TIER_2: "Multi-Leg Buying (₹30K-₹1L)",
            CapitalTier.TIER_3: "Full Arsenal (> ₹1L)"
        }
        return tier_names.get(self.current_tier, "Unknown")
    
    def can_execute_strategy(self, strategy_type: StrategyType) -> bool:
        """Check if strategy can be executed with current capital"""
        allowed = self.get_allowed_strategies()
        return strategy_type in allowed

class IntelligentRiskManager:
    """NEW: Enhanced risk management with position-based monitoring"""
    
    def __init__(self, technical_analyzer: ZerodhaTechnicalAnalyzer):
        self.technical_analyzer = technical_analyzer
        self.loss_thresholds = {
            5: 'light_analysis',   # 5% loss - light touch
            10: 'detailed_analysis', # 10% loss - detailed review
            15: 'emergency_analysis' # 15% loss - emergency mode
        }
        
    async def analyze_position_health(self, position: ActivePosition, 
                                     current_price: float, market_data: Dict) -> Dict:
        """Analyze position health at various loss thresholds"""
        
        # Calculate current loss percentage
        position_value = sum(leg.theoretical_price * leg.contracts * leg.lot_size 
                           for leg in position.option_legs)
        loss_percent = abs(min(0, position.current_pnl / position_value * 100))
        
        # Determine analysis depth
        analysis_type = 'normal'
        for threshold, analysis in self.loss_thresholds.items():
            if loss_percent >= threshold:
                analysis_type = analysis
        
        # Perform technical analysis
        tech_analysis = await self.technical_analyzer.analyze_symbol_for_options(
            position.signal.ticker, current_price, market_data, 'intraday'
        )
        
        # Analyze based on loss level
        if analysis_type == 'light_analysis':
            return self._light_position_analysis(position, tech_analysis, loss_percent)
        elif analysis_type == 'detailed_analysis':
            return self._detailed_position_analysis(position, tech_analysis, loss_percent)
        elif analysis_type == 'emergency_analysis':
            return self._emergency_position_analysis(position, tech_analysis, loss_percent)
        else:
            return self._normal_position_analysis(position, tech_analysis)
    
    def _light_position_analysis(self, position: ActivePosition, 
                                tech_analysis: Dict, loss_percent: float) -> Dict:
        """Light analysis at 5% loss"""
        
        # Check if it's just a temporary dip
        momentum = tech_analysis.get('momentum_analysis', {})
        trend = tech_analysis.get('trend_analysis', {})
        
        hold_position = (
            momentum.get('oversold', False) or  # Oversold bounce expected
            trend.get('daily_trend') in ['UPTREND', 'STRONG_UPTREND']  # Still in uptrend
        )
        
        return {
            'action': 'HOLD' if hold_position else 'MONITOR_CLOSELY',
            'reason': f"5% loss - {'temporary dip expected' if hold_position else 'monitoring'}",
            'confidence': 0.7 if hold_position else 0.5,
            'suggested_stop': position.entry_premium * 0.90,  # Widen stop to 10%
            'analysis_depth': 'light'
        }
    
    def _detailed_position_analysis(self, position: ActivePosition, 
                                   tech_analysis: Dict, loss_percent: float) -> Dict:
        """Detailed analysis at 10% loss"""
        
        # More thorough analysis
        entry_signal = tech_analysis.get('entry_signal', {})
        sr_levels = tech_analysis.get('support_resistance', {})
        
        # Check if original thesis still valid
        thesis_valid = self._check_thesis_validity(position, tech_analysis)
        
        # Check for reversal signs
        reversal_signs = (
            entry_signal.get('signal_type') == 'SELL' and position.signal.direction == 'bullish' or
            entry_signal.get('signal_type') == 'BUY' and position.signal.direction == 'bearish'
        )
        
        if thesis_valid and not reversal_signs:
            action = 'HOLD_WITH_TIGHT_STOP'
            confidence = 0.6
        else:
            action = 'EXIT_ON_BOUNCE'
            confidence = 0.4
        
        return {
            'action': action,
            'reason': f"10% loss - {'thesis still valid' if thesis_valid else 'thesis broken'}",
            'confidence': confidence,
            'suggested_stop': position.entry_premium * 0.85,
            'exit_target': sr_levels.get('nearest_resistance') if position.signal.direction == 'bearish' else sr_levels.get('nearest_support'),
            'analysis_depth': 'detailed',
            'thesis_valid': thesis_valid,
            'reversal_detected': reversal_signs
        }
    
    def _emergency_position_analysis(self, position: ActivePosition, 
                                    tech_analysis: Dict, loss_percent: float) -> Dict:
        """Emergency analysis at 15%+ loss"""
        
        # Capital preservation mode
        return {
            'action': 'EXIT_IMMEDIATELY',
            'reason': f"{loss_percent:.1f}% loss - emergency capital preservation",
            'confidence': 0.9,
            'suggested_stop': 'MARKET_ORDER',
            'analysis_depth': 'emergency',
            'preserve_capital': True
        }
    
    def _normal_position_analysis(self, position: ActivePosition, tech_analysis: Dict) -> Dict:
        """Normal position analysis when not in loss"""
        return {
            'action': 'CONTINUE',
            'reason': 'Position healthy',
            'confidence': 0.8,
            'analysis_depth': 'normal'
        }
    
    def _check_thesis_validity(self, position: ActivePosition, tech_analysis: Dict) -> bool:
        """Check if original trading thesis is still valid"""
        
        original_direction = position.signal.direction
        current_bias = tech_analysis.get('market_bias', 'NEUTRAL')
        
        # Map bias to direction
        if original_direction == 'bullish':
            return 'BULLISH' in current_bias or 'LEAN_BULLISH' in current_bias
        elif original_direction == 'bearish':
            return 'BEARISH' in current_bias or 'LEAN_BEARISH' in current_bias
        
        return False

class DynamicProfitOptimizer:
    """NEW: Dynamic profit optimization system"""
    
    def __init__(self):
        self.profit_checkpoints = [0.03, 0.06, 0.09, 0.12, 0.15]  # 3%, 6%, 9%, 12%, 15%
        self.daily_target_range = (0.08, 0.15)  # 8-15% daily target
        
    async def analyze_profit_continuation(self, position: ActivePosition, 
                                         profit_percent: float,
                                         tech_analysis: Dict) -> Dict:
        """Analyze whether to continue or exit at profit checkpoint"""
        
        # Determine which checkpoint we're at
        current_checkpoint = self._get_current_checkpoint(profit_percent)
        
        # Analyze continuation signals
        momentum = tech_analysis.get('momentum_analysis', {})
        trend = tech_analysis.get('trend_analysis', {})
        volume = tech_analysis.get('volume_analysis', {})
        
        # Continuation indicators
        continuation_score = 0
        exit_score = 0
        
        # Check RSI
        rsi = momentum.get('rsi', 50)
        if 45 <= rsi <= 75:
            continuation_score += 1  # Healthy RSI range
        elif rsi > 80:
            exit_score += 2  # Overbought
        elif rsi < 20:
            exit_score += 2  # Oversold (for puts)
        
        # Check volume
        if volume.get('trend') in ['SURGING', 'INCREASING']:
            continuation_score += 1
        elif volume.get('trend') == 'DECLINING':
            exit_score += 1
        
        # Check trend strength
        if trend.get('trend_strength', 0) > 0.6:
            continuation_score += 1
        
        # Check for higher highs/lower lows
        if self._check_momentum_continuation(position, tech_analysis):
            continuation_score += 2
        else:
            exit_score += 1
        
        # Time of day factor
        current_hour = datetime.now().hour
        if current_hour >= 14:  # After 2 PM
            exit_score += 1  # Prefer taking profits late in day
        
        # Decision based on checkpoint
        if current_checkpoint >= 0.15:  # At 15%+
            # In bonus territory - use trailing stop
            action = 'TRAIL_STOP'
            trail_percent = 0.03  # Trail by 3%
            reason = f"At {profit_percent:.1f}% profit - bonus territory, trailing stop"
        elif current_checkpoint >= 0.12:  # At 12%
            if continuation_score > exit_score:
                action = 'CONTINUE_WITH_TIGHT_STOP'
                reason = f"At {profit_percent:.1f}% - momentum continues"
            else:
                action = 'TAKE_PROFIT'
                reason = f"At {profit_percent:.1f}% - approaching daily target"
        elif current_checkpoint >= 0.09:  # At 9%
            if continuation_score > exit_score + 1:  # Need stronger continuation
                action = 'CONTINUE'
                reason = f"At {profit_percent:.1f}% - strong continuation signals"
            else:
                action = 'PARTIAL_EXIT'
                reason = f"At {profit_percent:.1f}% - secure partial profits"
        else:  # Below 9%
            action = 'CONTINUE'
            reason = f"At {profit_percent:.1f}% - below daily target"
        
        return {
            'action': action,
            'reason': reason,
            'profit_percent': profit_percent,
            'checkpoint': current_checkpoint,
            'continuation_score': continuation_score,
            'exit_score': exit_score,
            'trailing_stop_percent': 0.03 if action == 'TRAIL_STOP' else None,
            'suggested_stop': self._calculate_profit_stop(position, profit_percent, action)
        }
    
    def _get_current_checkpoint(self, profit_percent: float) -> float:
        """Get current profit checkpoint"""
        profit_decimal = profit_percent / 100
        for checkpoint in reversed(self.profit_checkpoints):
            if profit_decimal >= checkpoint:
                return checkpoint
        return 0
    
    def _check_momentum_continuation(self, position: ActivePosition, 
                                    tech_analysis: Dict) -> bool:
        """Check if momentum is continuing in profit direction"""
        
        intraday_trend = tech_analysis.get('trend_analysis', {}).get('intraday_trend')
        
        if position.signal.direction == 'bullish':
            return intraday_trend == 'UP'
        elif position.signal.direction == 'bearish':
            return intraday_trend == 'DOWN'
        
        return False
    
    def _calculate_profit_stop(self, position: ActivePosition, 
                              profit_percent: float, action: str) -> float:
        """Calculate stop loss to protect profits"""
        
        current_price = position.entry_premium * (1 + profit_percent / 100)
        
        if action == 'TRAIL_STOP':
            # Trail by 3% from current
            return current_price * 0.97
        elif action == 'CONTINUE_WITH_TIGHT_STOP':
            # Tight stop at 2% below current
            return current_price * 0.98
        elif 'CONTINUE' in action:
            # Normal stop at 5% below current
            return current_price * 0.95
        else:
            # Keep original stop
            return position.stop_loss

class MultiLegStrategyExecutor:
    """Complete multi-leg option strategies executor with full Tier 1-3 implementation"""
    
    def __init__(self, options_chain_provider: ZerodhaOptionsChainProvider,
                 order_manager: ZerodhaOrderManager):
        self.options_provider = options_chain_provider
        self.order_manager = order_manager
        
        # Lot size mapping for different instruments
        self.lot_sizes = {
            'NIFTY': 75,           # Updated lot sizes
            'BANKNIFTY': 15,
            'FINNIFTY': 40,
            'MIDCPNIFTY': 75,
            'RELIANCE': 500,
            'HDFCBANK': 550,
            'TCS': 175,
            'INFY': 400,
            'BAJFINANCE': 750,
            'MARUTI': 50,
            'ITC': 3200,
            'SBIN': 3000,
            'HINDUNILVR': 300,
            'BHARTIARTL': 1400
        }
        
        # Strike intervals for different instruments
        self.strike_intervals = {
            'NIFTY': 50,
            'BANKNIFTY': 100,
            'FINNIFTY': 50,
            'MIDCPNIFTY': 25,
            'RELIANCE': 25,
            'HDFCBANK': 25,
            'TCS': 50,
            'INFY': 25,
            'BAJFINANCE': 25,
            'MARUTI': 100,
            'ITC': 10,
            'SBIN': 10,
            'HINDUNILVR': 50,
            'BHARTIARTL': 25
        }
        
    def get_lot_size(self, symbol: str) -> int:
        """Get lot size for a symbol"""
        # Extract base symbol from option symbol if needed
        base_symbol = symbol.split('24')[0] if '24' in symbol else symbol.split('25')[0] if '25' in symbol else symbol
        return self.lot_sizes.get(base_symbol.upper(), 1)
    
    def get_strike_interval(self, symbol: str) -> int:
        """Get strike interval for a symbol"""
        base_symbol = symbol.split('24')[0] if '24' in symbol else symbol.split('25')[0] if '25' in symbol else symbol
        return self.strike_intervals.get(base_symbol.upper(), 50)
        
    async def create_strategy_legs(self, signal: TradingSignal, 
                              strategy_type: StrategyType,
                              available_capital: float,
                              tech_analysis: Dict,
                              capital_tier: CapitalTier = CapitalTier.TIER_1) -> List[OptionsLeg]:
        """Create option legs for the selected strategy with tier enforcement"""
        
        current_price = signal.current_price
        
        # Add comprehensive debug logging
        logger.info(f"🔍 Creating strategy legs:")
        logger.info(f"   Signal: {signal.ticker} {signal.direction}")
        logger.info(f"   Strategy type: {strategy_type}")
        logger.info(f"   Capital tier: {capital_tier}")
        logger.info(f"   Available capital: ₹{available_capital:,.0f}")
        logger.info(f"   Current price: ₹{current_price:.2f}")
        
        # Get option chain
        logger.info(f"🔄 Fetching option chain for {signal.ticker}...")
        option_chain = await self.options_provider.fetch_option_chain(signal.ticker)
        
        if not option_chain:
            logger.error(f"[ERROR] No option chain available for {signal.ticker}")
            return []
        
        logger.info(f"[OK] Option chain fetched: {len(option_chain.get('calls', []))} calls, {len(option_chain.get('puts', []))} puts")
        
        # Enforce capital tier restrictions
        logger.info(f"🔍 Validating strategy {strategy_type} for {capital_tier}...")
        if not self._validate_strategy_for_tier(strategy_type, capital_tier):
            logger.warning(f"Strategy {strategy_type} not allowed for {capital_tier}")
            return []
        
        logger.info(f"[OK] Strategy validation passed")
        
        # Create legs based on strategy type
        logger.info(f"🔄 Creating legs for strategy: {strategy_type}")
        
        if strategy_type == StrategyType.BUY_CALL:
            logger.info(f"[UP] Creating single call option")
            legs = await self._create_single_call(signal, option_chain, available_capital)
            
        elif strategy_type == StrategyType.BUY_PUT:
            logger.info(f"[DOWN] Creating single put option")
            legs = await self._create_single_put(signal, option_chain, available_capital)
            
        elif strategy_type == StrategyType.LONG_STRADDLE:
            logger.info(f"[TARGET] Creating long straddle")
            legs = await self._create_long_straddle(signal, option_chain, available_capital)
            
        elif strategy_type == StrategyType.LONG_STRANGLE:
            logger.info(f"[TARGET] Creating long strangle")
            legs = await self._create_long_strangle(signal, option_chain, available_capital)
            
        elif strategy_type == StrategyType.BULL_CALL_SPREAD:
            logger.info(f"[UP] Creating bull call spread")
            legs = await self._create_bull_call_spread(signal, option_chain, available_capital, capital_tier)
            
        elif strategy_type == StrategyType.BEAR_PUT_SPREAD:
            logger.info(f"[DOWN] Creating bear put spread")
            legs = await self._create_bear_put_spread(signal, option_chain, available_capital, capital_tier)
        
        # Tier 3 only strategies (selling allowed)
        elif strategy_type == StrategyType.IRON_CONDOR and capital_tier == CapitalTier.TIER_3:
            logger.info(f"🏗️ Creating iron condor")
            legs = await self._create_iron_condor(signal, option_chain, available_capital)
            
        elif strategy_type == StrategyType.CREDIT_SPREAD and capital_tier == CapitalTier.TIER_3:
            logger.info(f"[MONEY] Creating credit spread")
            legs = await self._create_credit_spread(signal, option_chain, available_capital)
            
        elif strategy_type == StrategyType.RATIO_SPREAD and capital_tier == CapitalTier.TIER_3:
            logger.info(f"⚖️ Creating ratio spread")
            legs = await self._create_ratio_spread(signal, option_chain, available_capital)
            
        elif strategy_type == StrategyType.CALENDAR_SPREAD and capital_tier == CapitalTier.TIER_3:
            logger.info(f"📅 Creating calendar spread")
            legs = await self._create_calendar_spread(signal, option_chain, available_capital)
            
        elif strategy_type == StrategyType.SHORT_STRADDLE and capital_tier == CapitalTier.TIER_3:
            logger.info(f"[DOWN] Creating short straddle")
            legs = await self._create_short_straddle(signal, option_chain, available_capital)
            
        elif strategy_type == StrategyType.SHORT_STRANGLE and capital_tier == CapitalTier.TIER_3:
            logger.info(f"[DOWN] Creating short strangle")
            legs = await self._create_short_strangle(signal, option_chain, available_capital)
            
        else:
            logger.warning(f"Strategy {strategy_type} not implemented or not allowed for tier {capital_tier}")
            return []
        
        # Log results
        logger.info(f"[TARGET] Strategy leg creation completed:")
        logger.info(f"   Created {len(legs)} legs")
        
        for i, leg in enumerate(legs):
            logger.info(f"   Leg {i+1}: {leg.action} {leg.contracts}x {leg.tradingsymbol} @ ₹{leg.theoretical_price:.2f}")
        
        if not legs:
            logger.warning(f"[ERROR] No strategy legs created - check individual strategy method")
        
        return legs

    def _validate_strategy_for_tier(self, strategy_type: StrategyType, capital_tier: CapitalTier) -> bool:
        """Validate if strategy is allowed for the capital tier"""
        
        # Define allowed strategies for each tier
        tier_1_strategies = [StrategyType.BUY_CALL, StrategyType.BUY_PUT]
        tier_2_strategies = tier_1_strategies + [
            StrategyType.LONG_STRADDLE, StrategyType.LONG_STRANGLE,
            StrategyType.BULL_CALL_SPREAD, StrategyType.BEAR_PUT_SPREAD
        ]
        tier_3_strategies = list(StrategyType)  # All strategies
        
        # 🔍 Enhanced DEBUG with multiple comparison methods
        logger.info(f"🔍 Enhanced validation debug:")
        logger.info(f"   Strategy to check: {strategy_type}")
        logger.info(f"   Strategy type: {type(strategy_type)}")
        logger.info(f"   Strategy repr: {repr(strategy_type)}")
        logger.info(f"   Strategy value: {strategy_type.value}")
        logger.info(f"   Strategy name: {strategy_type.name}")
        
        logger.info(f"   Tier 1 strategies: {tier_1_strategies}")
        logger.info(f"   BUY_PUT reference: {StrategyType.BUY_PUT}")
        logger.info(f"   BUY_PUT repr: {repr(StrategyType.BUY_PUT)}")
        
        # Multiple comparison methods for debugging
        direct_comparison = strategy_type == StrategyType.BUY_PUT
        in_list_comparison = strategy_type in tier_1_strategies
        value_comparison = strategy_type.value == StrategyType.BUY_PUT.value
        name_comparison = strategy_type.name == StrategyType.BUY_PUT.name
        
        logger.info(f"   Direct equality (==): {direct_comparison}")
        logger.info(f"   In list check: {in_list_comparison}")
        logger.info(f"   Value comparison: {value_comparison}")
        logger.info(f"   Name comparison: {name_comparison}")
        
        # ID comparison for debugging enum identity issues
        logger.info(f"   Strategy ID: {id(strategy_type)}")
        logger.info(f"   BUY_PUT ID: {id(StrategyType.BUY_PUT)}")
        logger.info(f"   Same object? {strategy_type is StrategyType.BUY_PUT}")
        
        # Get the appropriate allowed strategies for the tier
        if capital_tier == CapitalTier.TIER_1:
            allowed_strategies = tier_1_strategies
            tier_name = "TIER_1"
        elif capital_tier == CapitalTier.TIER_2:
            allowed_strategies = tier_2_strategies
            tier_name = "TIER_2"
        else:  # TIER_3
            allowed_strategies = tier_3_strategies
            tier_name = "TIER_3"
        
        # Primary validation using direct enum comparison
        result_direct = strategy_type in allowed_strategies
        
        # Fallback validation using enum values (in case of enum identity issues)
        allowed_values = [s.value for s in allowed_strategies]
        result_value = strategy_type.value in allowed_values
        
        # Fallback validation using enum names
        allowed_names = [s.name for s in allowed_strategies]
        result_name = strategy_type.name in allowed_names
        
        logger.info(f"   {tier_name} allowed strategies: {allowed_strategies}")
        logger.info(f"   {tier_name} allowed values: {allowed_values}")
        logger.info(f"   {tier_name} allowed names: {allowed_names}")
        logger.info(f"   Direct validation result: {result_direct}")
        logger.info(f"   Value validation result: {result_value}")
        logger.info(f"   Name validation result: {result_name}")
        
        # Use the most reliable validation method
        # If any validation method returns True, consider it valid
        final_result = result_direct or result_value or result_name
        
        if not result_direct and (result_value or result_name):
            logger.warning(f"[WARNING]  Enum identity issue detected! Using fallback validation.")
            logger.warning(f"   Direct comparison failed but value/name comparison succeeded.")
        
        logger.info(f"   Final validation result: {final_result}")
        return final_result

    async def _create_single_call(self, signal: TradingSignal, 
                                 option_chain: Dict, available_capital: float) -> List[OptionsLeg]:
        """Create single call option leg"""
        
        current_price = signal.current_price
        strike_interval = self.get_strike_interval(signal.ticker)
        lot_size = self.get_lot_size(signal.ticker)
        
        # Find ATM or slightly OTM call based on signal strength
        if signal.confidence > 0.8:
            target_moneyness = 1.0  # ATM for high confidence
        elif signal.confidence > 0.6:
            target_moneyness = 1.01  # Slightly OTM for medium confidence
        else:
            target_moneyness = 1.02  # More OTM for lower confidence
        
        target_strike = round(current_price * target_moneyness / strike_interval) * strike_interval
        
        # Get call option data
        calls = option_chain.get('calls', [])
        selected_call = None
        
        # Find exact strike or closest available
        for call in calls:
            if call['strike'] == target_strike:
                selected_call = call
                break
        
        # If exact strike not found, find closest
        if not selected_call and calls:
            selected_call = min(calls, key=lambda x: abs(x['strike'] - target_strike))
        
        if not selected_call:
            logger.error(f"No suitable call options found for {signal.ticker}")
            return []
        
        # Calculate position size
        premium = selected_call['lastPrice']
        max_contracts = int(available_capital * 0.9 / (premium * lot_size))
        contracts = max(1, min(max_contracts, 10))  # Between 1-10 lots
        
        return [OptionsLeg(
            action='BUY',
            option_type='call',
            strike=selected_call['strike'],
            expiry=option_chain['expiry'],
            contracts=contracts,
            lot_size=lot_size,
            max_premium=premium * 1.02,  # 2% slippage
            min_premium=premium * 0.98,
            theoretical_price=premium,
            tradingsymbol=selected_call.get('tradingsymbol', ''),
            exchange='NFO'
        )]

    async def _create_single_put(self, signal: TradingSignal, 
                                option_chain: Dict, available_capital: float) -> List[OptionsLeg]:
        """Create single put option leg"""
        
        current_price = signal.current_price
        strike_interval = self.get_strike_interval(signal.ticker)
        lot_size = self.get_lot_size(signal.ticker)
        
        # Find ATM or slightly OTM put based on signal strength
        if signal.confidence > 0.8:
            target_moneyness = 1.0  # ATM for high confidence
        elif signal.confidence > 0.6:
            target_moneyness = 0.99  # Slightly OTM for medium confidence
        else:
            target_moneyness = 0.98  # More OTM for lower confidence
        
        target_strike = round(current_price * target_moneyness / strike_interval) * strike_interval
        
        # Get put option data
        puts = option_chain.get('puts', [])
        selected_put = None
        
        # Find exact strike or closest available
        for put in puts:
            if put['strike'] == target_strike:
                selected_put = put
                break
        
        # If exact strike not found, find closest
        if not selected_put and puts:
            selected_put = min(puts, key=lambda x: abs(x['strike'] - target_strike))
        
        if not selected_put:
            logger.error(f"No suitable put options found for {signal.ticker}")
            return []
        
        # Calculate position size
        premium = selected_put['lastPrice']
        max_contracts = int(available_capital * 0.9 / (premium * lot_size))
        contracts = max(1, min(max_contracts, 10))  # Between 1-10 lots
        
        return [OptionsLeg(
            action='BUY',
            option_type='put',
            strike=selected_put['strike'],
            expiry=option_chain['expiry'],
            contracts=contracts,
            lot_size=lot_size,
            max_premium=premium * 1.02,  # 2% slippage
            min_premium=premium * 0.98,
            theoretical_price=premium,
            tradingsymbol=selected_put.get('tradingsymbol', ''),
            exchange='NFO'
        )]

    async def _create_long_straddle(self, signal: TradingSignal, 
                                   option_chain: Dict, available_capital: float) -> List[OptionsLeg]:
        """Create long straddle (buy ATM call + put)"""
        
        current_price = signal.current_price
        strike_interval = self.get_strike_interval(signal.ticker)
        lot_size = self.get_lot_size(signal.ticker)
        
        # Find ATM strike
        atm_strike = round(current_price / strike_interval) * strike_interval
        
        # Get both call and put at same strike
        calls = option_chain.get('calls', [])
        puts = option_chain.get('puts', [])
        
        selected_call = None
        selected_put = None
        
        for call in calls:
            if call['strike'] == atm_strike:
                selected_call = call
                break
        
        for put in puts:
            if put['strike'] == atm_strike:
                selected_put = put
                break
        
        if not selected_call or not selected_put:
            logger.error(f"ATM options not found for straddle at strike {atm_strike}")
            return []
        
        # Calculate position size for both legs
        total_premium = selected_call['lastPrice'] + selected_put['lastPrice']
        max_contracts = int(available_capital * 0.9 / (total_premium * lot_size))
        contracts = max(1, min(max_contracts, 5))  # Conservative for straddles
        
        return [
            OptionsLeg(
                action='BUY',
                option_type='call',
                strike=atm_strike,
                expiry=option_chain['expiry'],
                contracts=contracts,
                lot_size=lot_size,
                max_premium=selected_call['lastPrice'] * 1.02,
                min_premium=selected_call['lastPrice'] * 0.98,
                theoretical_price=selected_call['lastPrice'],
                tradingsymbol=selected_call.get('tradingsymbol', ''),
                exchange='NFO'
            ),
            OptionsLeg(
                action='BUY',
                option_type='put',
                strike=atm_strike,
                expiry=option_chain['expiry'],
                contracts=contracts,
                lot_size=lot_size,
                max_premium=selected_put['lastPrice'] * 1.02,
                min_premium=selected_put['lastPrice'] * 0.98,
                theoretical_price=selected_put['lastPrice'],
                tradingsymbol=selected_put.get('tradingsymbol', ''),
                exchange='NFO'
            )
        ]

    async def _create_long_strangle(self, signal: TradingSignal, 
                                   option_chain: Dict, available_capital: float) -> List[OptionsLeg]:
        """Create long strangle (buy OTM call + OTM put) - COMPLETE IMPLEMENTATION"""
        
        current_price = signal.current_price
        strike_interval = self.get_strike_interval(signal.ticker)
        lot_size = self.get_lot_size(signal.ticker)
        
        # Calculate OTM strikes based on volatility expectation
        # For high volatility expectation, use wider strikes
        volatility_adjustment = 1.0
        if hasattr(signal, 'market_data') and signal.market_data:
            implied_vol = signal.market_data.get('implied_volatility', 0.2)
            if implied_vol > 0.3:  # High volatility
                volatility_adjustment = 1.2
            elif implied_vol < 0.15:  # Low volatility
                volatility_adjustment = 0.8
        
        # OTM strikes - typically 2-5% away from current price
        call_distance = 0.03 * volatility_adjustment  # 3% OTM call (adjustable)
        put_distance = 0.03 * volatility_adjustment   # 3% OTM put (adjustable)
        
        call_strike = round(current_price * (1 + call_distance) / strike_interval) * strike_interval
        put_strike = round(current_price * (1 - put_distance) / strike_interval) * strike_interval
        
        logger.info(f"Long Strangle: Call @ {call_strike}, Put @ {put_strike} (Spot: {current_price})")
        
        # Find the OTM options
        calls = option_chain.get('calls', [])
        puts = option_chain.get('puts', [])
        
        selected_call = None
        selected_put = None
        
        # Find call option (prefer exact strike, then closest higher strike)
        call_candidates = [c for c in calls if c['strike'] >= call_strike]
        if call_candidates:
            selected_call = min(call_candidates, key=lambda x: abs(x['strike'] - call_strike))
        elif calls:  # Fallback to any available call
            selected_call = min(calls, key=lambda x: abs(x['strike'] - call_strike))
        
        # Find put option (prefer exact strike, then closest lower strike)
        put_candidates = [p for p in puts if p['strike'] <= put_strike]
        if put_candidates:
            selected_put = min(put_candidates, key=lambda x: abs(x['strike'] - put_strike))
        elif puts:  # Fallback to any available put
            selected_put = min(puts, key=lambda x: abs(x['strike'] - put_strike))
        
        if not selected_call or not selected_put:
            logger.error(f"Suitable OTM options not found for strangle")
            logger.info(f"Available calls: {len(calls)}, Available puts: {len(puts)}")
            return []
        
        # Validate the strangle structure
        actual_call_strike = selected_call['strike']
        actual_put_strike = selected_put['strike']
        
        if actual_call_strike <= actual_put_strike:
            logger.warning(f"Invalid strangle: Call strike {actual_call_strike} <= Put strike {actual_put_strike}")
            # Try to fix by finding a higher call strike
            higher_calls = [c for c in calls if c['strike'] > actual_put_strike]
            if higher_calls:
                selected_call = min(higher_calls, key=lambda x: x['strike'])
                logger.info(f"Adjusted call strike to {selected_call['strike']}")
            else:
                return []
        
        # Calculate position size
        total_premium = selected_call['lastPrice'] + selected_put['lastPrice']
        max_contracts = int(available_capital * 0.9 / (total_premium * lot_size))
        contracts = max(1, min(max_contracts, 5))  # Conservative for strangles
        
        logger.info(f"Strangle sizing: {contracts} lots, Total premium: ₹{total_premium:.2f}")
        
        return [
            OptionsLeg(
                action='BUY',
                option_type='call',
                strike=selected_call['strike'],
                expiry=option_chain['expiry'],
                contracts=contracts,
                lot_size=lot_size,
                max_premium=selected_call['lastPrice'] * 1.02,
                min_premium=selected_call['lastPrice'] * 0.98,
                theoretical_price=selected_call['lastPrice'],
                tradingsymbol=selected_call.get('tradingsymbol', ''),
                exchange='NFO'
            ),
            OptionsLeg(
                action='BUY',
                option_type='put',
                strike=selected_put['strike'],
                expiry=option_chain['expiry'],
                contracts=contracts,
                lot_size=lot_size,
                max_premium=selected_put['lastPrice'] * 1.02,
                min_premium=selected_put['lastPrice'] * 0.98,
                theoretical_price=selected_put['lastPrice'],
                tradingsymbol=selected_put.get('tradingsymbol', ''),
                exchange='NFO'
            )
        ]

    async def _create_bull_call_spread(self, signal: TradingSignal, 
                                      option_chain: Dict, available_capital: float,
                                      capital_tier: CapitalTier = CapitalTier.TIER_2) -> List[OptionsLeg]:
        """Create bull call spread - Tier 2 (buy only) vs Tier 3 (buy+sell)"""
        
        current_price = signal.current_price
        strike_interval = self.get_strike_interval(signal.ticker)
        lot_size = self.get_lot_size(signal.ticker)
        
        # Calculate strikes
        atm_strike = round(current_price / strike_interval) * strike_interval
        otm_strike = atm_strike + (2 * strike_interval)  # 2 strikes away
        
        calls = option_chain.get('calls', [])
        
        # Find both legs
        buy_call = None
        sell_call = None
        
        for call in calls:
            if call['strike'] == atm_strike:
                buy_call = call
            elif call['strike'] == otm_strike:
                sell_call = call
        
        if not buy_call or not sell_call:
            logger.error(f"Bull call spread legs not found: ATM={atm_strike}, OTM={otm_strike}")
            return []
        
        if capital_tier == CapitalTier.TIER_2:
            # TIER 2: BUY BOTH CALLS (no selling allowed)
            logger.info("Tier 2 Bull Call Spread: Buying both calls (no selling)")
            
            total_premium = buy_call['lastPrice'] + sell_call['lastPrice']
            max_contracts = int(available_capital * 0.9 / (total_premium * lot_size))
            contracts = max(1, min(max_contracts, 5))
            
            return [
                OptionsLeg(
                    action='BUY',
                    option_type='call',
                    strike=atm_strike,
                    expiry=option_chain['expiry'],
                    contracts=contracts,
                    lot_size=lot_size,
                    max_premium=buy_call['lastPrice'] * 1.02,
                    min_premium=buy_call['lastPrice'] * 0.98,
                    theoretical_price=buy_call['lastPrice'],
                    tradingsymbol=buy_call.get('tradingsymbol', ''),
                    exchange='NFO'
                ),
                OptionsLeg(
                    action='BUY',  # BUY instead of SELL for Tier 2
                    option_type='call',
                    strike=otm_strike,
                    expiry=option_chain['expiry'],
                    contracts=contracts,
                    lot_size=lot_size,
                    max_premium=sell_call['lastPrice'] * 1.02,
                    min_premium=sell_call['lastPrice'] * 0.98,
                    theoretical_price=sell_call['lastPrice'],
                    tradingsymbol=sell_call.get('tradingsymbol', ''),
                    exchange='NFO'
                )
            ]
        
        else:  # TIER_3
            # TIER 3: TRADITIONAL BULL CALL SPREAD (buy lower, sell higher)
            logger.info("Tier 3 Bull Call Spread: Buy lower strike, sell higher strike")
            
            net_premium = buy_call['lastPrice'] - sell_call['lastPrice']  # Net debit
            max_contracts = int(available_capital * 0.9 / (net_premium * lot_size))
            contracts = max(1, min(max_contracts, 10))
            
            return [
                OptionsLeg(
                    action='BUY',
                    option_type='call',
                    strike=atm_strike,
                    expiry=option_chain['expiry'],
                    contracts=contracts,
                    lot_size=lot_size,
                    max_premium=buy_call['lastPrice'] * 1.02,
                    min_premium=buy_call['lastPrice'] * 0.98,
                    theoretical_price=buy_call['lastPrice'],
                    tradingsymbol=buy_call.get('tradingsymbol', ''),
                    exchange='NFO'
                ),
                OptionsLeg(
                    action='SELL',  # SELL for Tier 3
                    option_type='call',
                    strike=otm_strike,
                    expiry=option_chain['expiry'],
                    contracts=contracts,
                    lot_size=lot_size,
                    max_premium=sell_call['lastPrice'] * 1.02,
                    min_premium=sell_call['lastPrice'] * 0.98,
                    theoretical_price=sell_call['lastPrice'],
                    tradingsymbol=sell_call.get('tradingsymbol', ''),
                    exchange='NFO'
                )
            ]

    async def _create_bear_put_spread(self, signal: TradingSignal, 
                                     option_chain: Dict, available_capital: float,
                                     capital_tier: CapitalTier = CapitalTier.TIER_2) -> List[OptionsLeg]:
        """Create bear put spread - Tier 2 (buy only) vs Tier 3 (buy+sell)"""
        
        current_price = signal.current_price
        strike_interval = self.get_strike_interval(signal.ticker)
        lot_size = self.get_lot_size(signal.ticker)
        
        # Calculate strikes
        atm_strike = round(current_price / strike_interval) * strike_interval
        otm_strike = atm_strike - (2 * strike_interval)  # 2 strikes below
        
        puts = option_chain.get('puts', [])
        
        # Find both legs
        buy_put = None
        sell_put = None
        
        for put in puts:
            if put['strike'] == atm_strike:
                buy_put = put
            elif put['strike'] == otm_strike:
                sell_put = put
        
        if not buy_put or not sell_put:
            logger.error(f"Bear put spread legs not found: ATM={atm_strike}, OTM={otm_strike}")
            return []
        
        if capital_tier == CapitalTier.TIER_2:
            # TIER 2: BUY BOTH PUTS (no selling allowed)
            logger.info("Tier 2 Bear Put Spread: Buying both puts (no selling)")
            
            total_premium = buy_put['lastPrice'] + sell_put['lastPrice']
            max_contracts = int(available_capital * 0.9 / (total_premium * lot_size))
            contracts = max(1, min(max_contracts, 5))
            
            return [
                OptionsLeg(
                    action='BUY',
                    option_type='put',
                    strike=atm_strike,
                    expiry=option_chain['expiry'],
                    contracts=contracts,
                    lot_size=lot_size,
                    max_premium=buy_put['lastPrice'] * 1.02,
                    min_premium=buy_put['lastPrice'] * 0.98,
                    theoretical_price=buy_put['lastPrice'],
                    tradingsymbol=buy_put.get('tradingsymbol', ''),
                    exchange='NFO'
                ),
                OptionsLeg(
                    action='BUY',  # BUY instead of SELL for Tier 2
                    option_type='put',
                    strike=otm_strike,
                    expiry=option_chain['expiry'],
                    contracts=contracts,
                    lot_size=lot_size,
                    max_premium=sell_put['lastPrice'] * 1.02,
                    min_premium=sell_put['lastPrice'] * 0.98,
                    theoretical_price=sell_put['lastPrice'],
                    tradingsymbol=sell_put.get('tradingsymbol', ''),
                    exchange='NFO'
                )
            ]
        
        else:  # TIER_3
            # TIER 3: TRADITIONAL BEAR PUT SPREAD (buy higher, sell lower)
            logger.info("Tier 3 Bear Put Spread: Buy higher strike, sell lower strike")
            
            net_premium = buy_put['lastPrice'] - sell_put['lastPrice']  # Net debit
            max_contracts = int(available_capital * 0.9 / (net_premium * lot_size))
            contracts = max(1, min(max_contracts, 10))
            
            return [
                OptionsLeg(
                    action='BUY',
                    option_type='put',
                    strike=atm_strike,
                    expiry=option_chain['expiry'],
                    contracts=contracts,
                    lot_size=lot_size,
                    max_premium=buy_put['lastPrice'] * 1.02,
                    min_premium=buy_put['lastPrice'] * 0.98,
                    theoretical_price=buy_put['lastPrice'],
                    tradingsymbol=buy_put.get('tradingsymbol', ''),
                    exchange='NFO'
                ),
                OptionsLeg(
                    action='SELL',  # SELL for Tier 3
                    option_type='put',
                    strike=otm_strike,
                    expiry=option_chain['expiry'],
                    contracts=contracts,
                    lot_size=lot_size,
                    max_premium=sell_put['lastPrice'] * 1.02,
                    min_premium=sell_put['lastPrice'] * 0.98,
                    theoretical_price=sell_put['lastPrice'],
                    tradingsymbol=sell_put.get('tradingsymbol', ''),
                    exchange='NFO'
                )
            ]

    # ==== TIER 3 ONLY STRATEGIES (SELLING ALLOWED) ====

    async def _create_iron_condor(self, signal: TradingSignal, 
                                 option_chain: Dict, available_capital: float) -> List[OptionsLeg]:
        """Create iron condor (Tier 3 only) - Sell ATM straddle + Buy protective wings"""
        
        current_price = signal.current_price
        strike_interval = self.get_strike_interval(signal.ticker)
        lot_size = self.get_lot_size(signal.ticker)
        
        # Iron Condor strikes
        atm_strike = round(current_price / strike_interval) * strike_interval
        
        # Wing distances (wider for iron condor than butterfly)
        wing_distance = 3 * strike_interval  # 3 strikes away
        
        lower_put_buy = atm_strike - wing_distance
        upper_put_sell = atm_strike
        lower_call_sell = atm_strike  
        upper_call_buy = atm_strike + wing_distance
        
        calls = option_chain.get('calls', [])
        puts = option_chain.get('puts', [])
        
        # Find all four legs
        legs_data = {}
        for call in calls:
            if call['strike'] == atm_strike:
                legs_data['sell_call'] = call
            elif call['strike'] == upper_call_buy:
                legs_data['buy_call'] = call
        
        for put in puts:
            if put['strike'] == atm_strike:
                legs_data['sell_put'] = put
            elif put['strike'] == lower_put_buy:
                legs_data['buy_put'] = put
        
        if len(legs_data) != 4:
            logger.error(f"Iron Condor: Missing legs, found {len(legs_data)}/4")
            return []
        
        # Calculate net credit
        total_credit = legs_data['sell_call']['lastPrice'] + legs_data['sell_put']['lastPrice']
        total_debit = legs_data['buy_call']['lastPrice'] + legs_data['buy_put']['lastPrice']
        net_credit = total_credit - total_debit
        
        # Position sizing based on max loss (wing distance - net credit)
        max_loss_per_lot = (wing_distance - net_credit) * lot_size
        max_contracts = int(available_capital * 0.1 / max_loss_per_lot) if max_loss_per_lot > 0 else 1
        contracts = max(1, min(max_contracts, 3))  # Very conservative for iron condor
        
        logger.info(f"Iron Condor: Net credit ₹{net_credit:.2f}, Max loss ₹{max_loss_per_lot:.2f}")
        
        return [
            # Buy lower put (protection)
            OptionsLeg(
                action='BUY',
                option_type='put',
                strike=lower_put_buy,
                expiry=option_chain['expiry'],
                contracts=contracts,
                lot_size=lot_size,
                max_premium=legs_data['buy_put']['lastPrice'] * 1.02,
                min_premium=legs_data['buy_put']['lastPrice'] * 0.98,
                theoretical_price=legs_data['buy_put']['lastPrice'],
                tradingsymbol=legs_data['buy_put'].get('tradingsymbol', ''),
                exchange='NFO'
            ),
            # Sell ATM put (income)
            OptionsLeg(
                action='SELL',
                option_type='put',
                strike=atm_strike,
                expiry=option_chain['expiry'],
                contracts=contracts,
                lot_size=lot_size,
                max_premium=legs_data['sell_put']['lastPrice'] * 1.02,
                min_premium=legs_data['sell_put']['lastPrice'] * 0.98,
                theoretical_price=legs_data['sell_put']['lastPrice'],
                tradingsymbol=legs_data['sell_put'].get('tradingsymbol', ''),
                exchange='NFO'
            ),
            # Sell ATM call (income)
            OptionsLeg(
                action='SELL',
                option_type='call',
                strike=atm_strike,
                expiry=option_chain['expiry'],
                contracts=contracts,
                lot_size=lot_size,
                max_premium=legs_data['sell_call']['lastPrice'] * 1.02,
                min_premium=legs_data['sell_call']['lastPrice'] * 0.98,
                theoretical_price=legs_data['sell_call']['lastPrice'],
                tradingsymbol=legs_data['sell_call'].get('tradingsymbol', ''),
                exchange='NFO'
            ),
            # Buy upper call (protection)
            OptionsLeg(
                action='BUY',
                option_type='call',
                strike=upper_call_buy,
                expiry=option_chain['expiry'],
                contracts=contracts,
                lot_size=lot_size,
                max_premium=legs_data['buy_call']['lastPrice'] * 1.02,
                min_premium=legs_data['buy_call']['lastPrice'] * 0.98,
                theoretical_price=legs_data['buy_call']['lastPrice'],
                tradingsymbol=legs_data['buy_call'].get('tradingsymbol', ''),
                exchange='NFO'
            )
        ]

    async def _create_credit_spread(self, signal: TradingSignal, 
                                   option_chain: Dict, available_capital: float) -> List[OptionsLeg]:
        """Create credit spread (Tier 3 only) - Bull put or bear call spread for income"""
        
        if signal.direction == 'bullish':
            return await self._create_bull_put_spread(signal, option_chain, available_capital)
        elif signal.direction == 'bearish':
            return await self._create_bear_call_spread(signal, option_chain, available_capital)
        else:
            # Neutral - create both bull put and bear call for iron condor-like structure
            return await self._create_iron_condor(signal, option_chain, available_capital)

    async def _create_bull_put_spread(self, signal: TradingSignal, 
                                     option_chain: Dict, available_capital: float) -> List[OptionsLeg]:
        """Create bull put spread (Tier 3) - Sell higher strike put, buy lower strike put"""
        
        current_price = signal.current_price
        strike_interval = self.get_strike_interval(signal.ticker)
        lot_size = self.get_lot_size(signal.ticker)
        
        # For bull put spread: sell put closer to ATM, buy put further OTM
        sell_strike = round(current_price * 0.97 / strike_interval) * strike_interval  # 3% OTM
        buy_strike = sell_strike - (2 * strike_interval)  # 2 strikes lower
        
        puts = option_chain.get('puts', [])
        
        sell_put = None
        buy_put = None
        
        for put in puts:
            if put['strike'] == sell_strike:
                sell_put = put
            elif put['strike'] == buy_strike:
                buy_put = put
        
        if not sell_put or not buy_put:
            logger.error(f"Bull put spread legs not found: Sell={sell_strike}, Buy={buy_strike}")
            return []
        
        # Calculate net credit and position size
        net_credit = sell_put['lastPrice'] - buy_put['lastPrice']
        max_loss = (sell_strike - buy_strike) - net_credit
        max_contracts = int(available_capital * 0.15 / (max_loss * lot_size)) if max_loss > 0 else 1
        contracts = max(1, min(max_contracts, 5))
        
        return [
            OptionsLeg(
                action='SELL',
                option_type='put',
                strike=sell_strike,
                expiry=option_chain['expiry'],
                contracts=contracts,
                lot_size=lot_size,
                max_premium=sell_put['lastPrice'] * 1.02,
                min_premium=sell_put['lastPrice'] * 0.98,
                theoretical_price=sell_put['lastPrice'],
                tradingsymbol=sell_put.get('tradingsymbol', ''),
                exchange='NFO'
            ),
            OptionsLeg(
                action='BUY',
                option_type='put',
                strike=buy_strike,
                expiry=option_chain['expiry'],
                contracts=contracts,
                lot_size=lot_size,
                max_premium=buy_put['lastPrice'] * 1.02,
                min_premium=buy_put['lastPrice'] * 0.98,
                theoretical_price=buy_put['lastPrice'],
                tradingsymbol=buy_put.get('tradingsymbol', ''),
                exchange='NFO'
            )
        ]

    async def _create_bear_call_spread(self, signal: TradingSignal, 
                                      option_chain: Dict, available_capital: float) -> List[OptionsLeg]:
        """Create bear call spread (Tier 3) - Sell lower strike call, buy higher strike call"""
        
        current_price = signal.current_price
        strike_interval = self.get_strike_interval(signal.ticker)
        lot_size = self.get_lot_size(signal.ticker)
        
        # For bear call spread: sell call closer to ATM, buy call further OTM
        sell_strike = round(current_price * 1.03 / strike_interval) * strike_interval  # 3% OTM
        buy_strike = sell_strike + (2 * strike_interval)  # 2 strikes higher
        
        calls = option_chain.get('calls', [])
        
        sell_call = None
        buy_call = None
        
        for call in calls:
            if call['strike'] == sell_strike:
                sell_call = call
            elif call['strike'] == buy_strike:
                buy_call = call
        
        if not sell_call or not buy_call:
            logger.error(f"Bear call spread legs not found: Sell={sell_strike}, Buy={buy_strike}")
            return []
        
        # Calculate net credit and position size
        net_credit = sell_call['lastPrice'] - buy_call['lastPrice']
        max_loss = (buy_strike - sell_strike) - net_credit
        max_contracts = int(available_capital * 0.15 / (max_loss * lot_size)) if max_loss > 0 else 1
        contracts = max(1, min(max_contracts, 5))
        
        return [
            OptionsLeg(
                action='SELL',
                option_type='call',
                strike=sell_strike,
                expiry=option_chain['expiry'],
                contracts=contracts,
                lot_size=lot_size,
                max_premium=sell_call['lastPrice'] * 1.02,
                min_premium=sell_call['lastPrice'] * 0.98,
                theoretical_price=sell_call['lastPrice'],
                tradingsymbol=sell_call.get('tradingsymbol', ''),
                exchange='NFO'
            ),
            OptionsLeg(
                action='BUY',
                option_type='call',
                strike=buy_strike,
                expiry=option_chain['expiry'],
                contracts=contracts,
                lot_size=lot_size,
                max_premium=buy_call['lastPrice'] * 1.02,
                min_premium=buy_call['lastPrice'] * 0.98,
                theoretical_price=buy_call['lastPrice'],
                tradingsymbol=buy_call.get('tradingsymbol', ''),
                exchange='NFO'
            )
        ]

    async def _create_ratio_spread(self, signal: TradingSignal, 
                                  option_chain: Dict, available_capital: float) -> List[OptionsLeg]:
        """Create ratio spread (Tier 3) - Buy 1, Sell 2 for income with limited risk"""
        
        current_price = signal.current_price
        strike_interval = self.get_strike_interval(signal.ticker)
        lot_size = self.get_lot_size(signal.ticker)
        
        if signal.direction == 'bullish':
            # Bull call ratio spread: Buy 1 ATM call, Sell 2 OTM calls
            buy_strike = round(current_price / strike_interval) * strike_interval
            sell_strike = buy_strike + (2 * strike_interval)
            option_type = 'call'
        else:
            # Bear put ratio spread: Buy 1 ATM put, Sell 2 OTM puts  
            buy_strike = round(current_price / strike_interval) * strike_interval
            sell_strike = buy_strike - (2 * strike_interval)
            option_type = 'put'
        
        options = option_chain.get('calls' if option_type == 'call' else 'puts', [])
        
        buy_option = None
        sell_option = None
        
        for option in options:
            if option['strike'] == buy_strike:
                buy_option = option
            elif option['strike'] == sell_strike:
                sell_option = option
        
        if not buy_option or not sell_option:
            logger.error(f"Ratio spread legs not found")
            return []
        
        # Calculate net credit (2 * sell premium - 1 * buy premium)
        net_credit = (2 * sell_option['lastPrice']) - buy_option['lastPrice']
        
        # Conservative position sizing for ratio spreads
        max_contracts = int(available_capital * 0.05 / (buy_option['lastPrice'] * lot_size))
        contracts = max(1, min(max_contracts, 2))  # Very conservative
        
        return [
            OptionsLeg(
                action='BUY',
                option_type=option_type,
                strike=buy_strike,
                expiry=option_chain['expiry'],
                contracts=contracts,
                lot_size=lot_size,
                max_premium=buy_option['lastPrice'] * 1.02,
                min_premium=buy_option['lastPrice'] * 0.98,
                theoretical_price=buy_option['lastPrice'],
                tradingsymbol=buy_option.get('tradingsymbol', ''),
                exchange='NFO'
            ),
            OptionsLeg(
                action='SELL',
                option_type=option_type,
                strike=sell_strike,
                expiry=option_chain['expiry'],
                contracts=contracts * 2,  # Sell 2x contracts
                lot_size=lot_size,
                max_premium=sell_option['lastPrice'] * 1.02,
                min_premium=sell_option['lastPrice'] * 0.98,
                theoretical_price=sell_option['lastPrice'],
                tradingsymbol=sell_option.get('tradingsymbol', ''),
                exchange='NFO'
            )
        ]

    async def _create_calendar_spread(self, signal: TradingSignal, 
                                     option_chain: Dict, available_capital: float) -> List[OptionsLeg]:
        """Create calendar spread (Tier 3) - Different expiries, same strike"""
        
        # Note: This requires multiple expiry option chains
        # For simplicity, we'll create a proxy using different strikes
        logger.warning("Calendar spread requires multiple expiries - creating diagonal spread instead")
        
        current_price = signal.current_price
        strike_interval = self.get_strike_interval(signal.ticker)
        lot_size = self.get_lot_size(signal.ticker)
        
        # Diagonal spread: Buy ATM current month, Sell OTM current month
        buy_strike = round(current_price / strike_interval) * strike_interval
        
        if signal.direction == 'bullish':
            sell_strike = buy_strike + strike_interval
            option_type = 'call'
        else:
            sell_strike = buy_strike - strike_interval
            option_type = 'put'
        
        options = option_chain.get('calls' if option_type == 'call' else 'puts', [])
        
        buy_option = None
        sell_option = None
        
        for option in options:
            if option['strike'] == buy_strike:
                buy_option = option
            elif option['strike'] == sell_strike:
                sell_option = option
        
        if not buy_option or not sell_option:
            logger.error(f"Calendar/Diagonal spread legs not found")
            return []
        
        net_premium = buy_option['lastPrice'] - sell_option['lastPrice']
        max_contracts = int(available_capital * 0.1 / (abs(net_premium) * lot_size))
        contracts = max(1, min(max_contracts, 3))
        
        return [
            OptionsLeg(
                action='BUY',
                option_type=option_type,
                strike=buy_strike,
                expiry=option_chain['expiry'],
                contracts=contracts,
                lot_size=lot_size,
                max_premium=buy_option['lastPrice'] * 1.02,
                min_premium=buy_option['lastPrice'] * 0.98,
                theoretical_price=buy_option['lastPrice'],
                tradingsymbol=buy_option.get('tradingsymbol', ''),
                exchange='NFO'
            ),
            OptionsLeg(
                action='SELL',
                option_type=option_type,
                strike=sell_strike,
                expiry=option_chain['expiry'],
                contracts=contracts,
                lot_size=lot_size,
                max_premium=sell_option['lastPrice'] * 1.02,
                min_premium=sell_option['lastPrice'] * 0.98,
                theoretical_price=sell_option['lastPrice'],
                tradingsymbol=sell_option.get('tradingsymbol', ''),
                exchange='NFO'
            )
        ]

    async def _create_short_straddle(self, signal: TradingSignal, 
                                    option_chain: Dict, available_capital: float) -> List[OptionsLeg]:
        """Create short straddle (Tier 3) - Sell ATM call + put for income"""
        
        current_price = signal.current_price
        strike_interval = self.get_strike_interval(signal.ticker)
        lot_size = self.get_lot_size(signal.ticker)
        
        # Find ATM strike
        atm_strike = round(current_price / strike_interval) * strike_interval
        
        calls = option_chain.get('calls', [])
        puts = option_chain.get('puts', [])
        
        sell_call = None
        sell_put = None
        
        for call in calls:
            if call['strike'] == atm_strike:
                sell_call = call
                break
        
        for put in puts:
            if put['strike'] == atm_strike:
                sell_put = put
                break
        
        if not sell_call or not sell_put:
            logger.error(f"Short straddle: ATM options not found at {atm_strike}")
            return []
        
        # Very conservative sizing for unlimited risk strategy
        total_credit = sell_call['lastPrice'] + sell_put['lastPrice']
        max_contracts = int(available_capital * 0.02 / (current_price * lot_size))  # 2% of capital
        contracts = max(1, min(max_contracts, 1))  # Maximum 1 lot for safety
        
        logger.warning(f"Short Straddle: UNLIMITED RISK strategy with {contracts} contracts")
        
        return [
            OptionsLeg(
                action='SELL',
                option_type='call',
                strike=atm_strike,
                expiry=option_chain['expiry'],
                contracts=contracts,
                lot_size=lot_size,
                max_premium=sell_call['lastPrice'] * 1.02,
                min_premium=sell_call['lastPrice'] * 0.98,
                theoretical_price=sell_call['lastPrice'],
                tradingsymbol=sell_call.get('tradingsymbol', ''),
                exchange='NFO'
            ),
            OptionsLeg(
                action='SELL',
                option_type='put',
                strike=atm_strike,
                expiry=option_chain['expiry'],
                contracts=contracts,
                lot_size=lot_size,
                max_premium=sell_put['lastPrice'] * 1.02,
                min_premium=sell_put['lastPrice'] * 0.98,
                theoretical_price=sell_put['lastPrice'],
                tradingsymbol=sell_put.get('tradingsymbol', ''),
                exchange='NFO'
            )
        ]

    async def _create_short_strangle(self, signal: TradingSignal, 
                                    option_chain: Dict, available_capital: float) -> List[OptionsLeg]:
        """Create short strangle (Tier 3) - Sell OTM call + put for income"""
        
        current_price = signal.current_price
        strike_interval = self.get_strike_interval(signal.ticker)
        lot_size = self.get_lot_size(signal.ticker)
        
        # OTM strikes for short strangle
        call_strike = round(current_price * 1.05 / strike_interval) * strike_interval  # 5% OTM
        put_strike = round(current_price * 0.95 / strike_interval) * strike_interval   # 5% OTM
        
        calls = option_chain.get('calls', [])
        puts = option_chain.get('puts', [])
        
        sell_call = None
        sell_put = None
        
        for call in calls:
            if call['strike'] == call_strike:
                sell_call = call
                break
        
        for put in puts:
            if put['strike'] == put_strike:
                sell_put = put
                break
        
        if not sell_call or not sell_put:
            logger.error(f"Short strangle: OTM options not found")
            return []
        
        # Very conservative sizing for unlimited risk strategy
        total_credit = sell_call['lastPrice'] + sell_put['lastPrice']
        max_contracts = int(available_capital * 0.03 / (current_price * lot_size))  # 3% of capital
        contracts = max(1, min(max_contracts, 2))  # Maximum 2 lots
        
        logger.warning(f"Short Strangle: UNLIMITED RISK strategy with {contracts} contracts")
        
        return [
            OptionsLeg(
                action='SELL',
                option_type='call',
                strike=call_strike,
                expiry=option_chain['expiry'],
                contracts=contracts,
                lot_size=lot_size,
                max_premium=sell_call['lastPrice'] * 1.02,
                min_premium=sell_call['lastPrice'] * 0.98,
                theoretical_price=sell_call['lastPrice'],
                tradingsymbol=sell_call.get('tradingsymbol', ''),
                exchange='NFO'
            ),
            OptionsLeg(
                action='SELL',
                option_type='put',
                strike=put_strike,
                expiry=option_chain['expiry'],
                contracts=contracts,
                lot_size=lot_size,
                max_premium=sell_put['lastPrice'] * 1.02,
                min_premium=sell_put['lastPrice'] * 0.98,
                theoretical_price=sell_put['lastPrice'],
                tradingsymbol=sell_put.get('tradingsymbol', ''),
                exchange='NFO'
            )
        ]

    # ==== UTILITY METHODS ====

    def validate_legs(self, legs: List[OptionsLeg], strategy_type: StrategyType) -> Dict:
        """Validate created option legs for consistency"""
        
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'summary': {}
        }
        
        if not legs:
            validation_result['valid'] = False
            validation_result['errors'].append("No legs created")
            return validation_result
        
        # Basic validations
        for i, leg in enumerate(legs):
            if leg.contracts <= 0:
                validation_result['errors'].append(f"Leg {i+1}: Invalid contract count {leg.contracts}")
                validation_result['valid'] = False
            
            if leg.theoretical_price <= 0:
                validation_result['errors'].append(f"Leg {i+1}: Invalid premium {leg.theoretical_price}")
                validation_result['valid'] = False
            
            if not leg.tradingsymbol:
                validation_result['warnings'].append(f"Leg {i+1}: Missing trading symbol")
        
        # Strategy-specific validations
        if strategy_type in [StrategyType.LONG_STRADDLE, StrategyType.SHORT_STRADDLE]:
            if len(legs) != 2:
                validation_result['errors'].append(f"Straddle should have 2 legs, found {len(legs)}")
                validation_result['valid'] = False
            elif legs[0].strike != legs[1].strike:
                validation_result['errors'].append("Straddle legs should have same strike")
                validation_result['valid'] = False
        
        elif strategy_type in [StrategyType.LONG_STRANGLE, StrategyType.SHORT_STRANGLE]:
            if len(legs) != 2:
                validation_result['errors'].append(f"Strangle should have 2 legs, found {len(legs)}")
                validation_result['valid'] = False
            elif legs[0].strike == legs[1].strike:
                validation_result['warnings'].append("Strangle legs have same strike (effectively a straddle)")
        
        elif strategy_type == StrategyType.IRON_CONDOR:
            if len(legs) != 4:
                validation_result['errors'].append(f"Iron Condor should have 4 legs, found {len(legs)}")
                validation_result['valid'] = False
        
        # Calculate summary
        total_cost = sum(leg.theoretical_price * leg.contracts * leg.lot_size 
                        for leg in legs if leg.action == 'BUY')
        total_credit = sum(leg.theoretical_price * leg.contracts * leg.lot_size 
                          for leg in legs if leg.action == 'SELL')
        
        validation_result['summary'] = {
            'total_legs': len(legs),
            'buy_legs': sum(1 for leg in legs if leg.action == 'BUY'),
            'sell_legs': sum(1 for leg in legs if leg.action == 'SELL'),
            'total_cost': total_cost,
            'total_credit': total_credit,
            'net_cost': total_cost - total_credit,
            'total_contracts': sum(leg.contracts for leg in legs)
        }
        
        return validation_result

    def get_strategy_description(self, strategy_type: StrategyType) -> Dict:
        """Get detailed description of strategy characteristics"""
        
        descriptions = {
            StrategyType.BUY_CALL: {
                'name': 'Long Call',
                'market_view': 'Bullish',
                'max_risk': 'Premium paid',
                'max_reward': 'Unlimited',
                'breakeven': 'Strike + Premium',
                'ideal_conditions': 'Rising prices, increasing volatility',
                'tier_requirement': 'Tier 1+'
            },
            StrategyType.BUY_PUT: {
                'name': 'Long Put',
                'market_view': 'Bearish',
                'max_risk': 'Premium paid',
                'max_reward': 'Strike - Premium',
                'breakeven': 'Strike - Premium',
                'ideal_conditions': 'Falling prices, increasing volatility',
                'tier_requirement': 'Tier 1+'
            },
            StrategyType.LONG_STRADDLE: {
                'name': 'Long Straddle',
                'market_view': 'Neutral (expecting volatility)',
                'max_risk': 'Total premium paid',
                'max_reward': 'Unlimited',
                'breakeven': 'Strike ± Total Premium',
                'ideal_conditions': 'High volatility, directional movement',
                'tier_requirement': 'Tier 2+'
            },
            StrategyType.LONG_STRANGLE: {
                'name': 'Long Strangle',
                'market_view': 'Neutral (expecting volatility)',
                'max_risk': 'Total premium paid',
                'max_reward': 'Unlimited',
                'breakeven': 'Call Strike + Premium, Put Strike - Premium',
                'ideal_conditions': 'High volatility, wide price movements',
                'tier_requirement': 'Tier 2+'
            },
            StrategyType.BULL_CALL_SPREAD: {
                'name': 'Bull Call Spread',
                'market_view': 'Moderately Bullish',
                'max_risk': 'Net premium paid (Tier 2) or Net debit (Tier 3)',
                'max_reward': 'Strike difference - Net debit',
                'breakeven': 'Lower strike + Net debit',
                'ideal_conditions': 'Moderate upward movement',
                'tier_requirement': 'Tier 2+ (Tier 2: Buy both, Tier 3: Buy/Sell)'
            },
            StrategyType.BEAR_PUT_SPREAD: {
                'name': 'Bear Put Spread',
                'market_view': 'Moderately Bearish',
                'max_risk': 'Net premium paid (Tier 2) or Net debit (Tier 3)',
                'max_reward': 'Strike difference - Net debit',
                'breakeven': 'Higher strike - Net debit',
                'ideal_conditions': 'Moderate downward movement',
                'tier_requirement': 'Tier 2+ (Tier 2: Buy both, Tier 3: Buy/Sell)'
            },
            StrategyType.IRON_CONDOR: {
                'name': 'Iron Condor',
                'market_view': 'Neutral (range-bound)',
                'max_risk': 'Wing width - Net credit',
                'max_reward': 'Net credit received',
                'breakeven': 'Short strikes ± Net credit',
                'ideal_conditions': 'Low volatility, sideways movement',
                'tier_requirement': 'Tier 3 only'
            },
            StrategyType.SHORT_STRADDLE: {
                'name': 'Short Straddle',
                'market_view': 'Neutral (expecting low volatility)',
                'max_risk': 'UNLIMITED',
                'max_reward': 'Net credit received',
                'breakeven': 'Strike ± Net credit',
                'ideal_conditions': 'Very low volatility, minimal movement',
                'tier_requirement': 'Tier 3 only - HIGH RISK'
            },
            StrategyType.SHORT_STRANGLE: {
                'name': 'Short Strangle',
                'market_view': 'Neutral (expecting low volatility)',
                'max_risk': 'UNLIMITED',
                'max_reward': 'Net credit received',
                'breakeven': 'Strikes ± Net credit',
                'ideal_conditions': 'Low volatility, range-bound movement',
                'tier_requirement': 'Tier 3 only - HIGH RISK'
            }
        }
        
        return descriptions.get(strategy_type, {
            'name': str(strategy_type),
            'market_view': 'Unknown',
            'max_risk': 'Unknown',
            'max_reward': 'Unknown',
            'tier_requirement': 'Unknown'
        })

    async def get_strategy_analytics(self, legs: List[OptionsLeg], 
                                   current_price: float) -> Dict:
        """Calculate comprehensive strategy analytics"""
        
        if not legs:
            return {}
        
        # Calculate key metrics
        total_cost = sum(leg.theoretical_price * leg.contracts * leg.lot_size 
                        for leg in legs if leg.action == 'BUY')
        total_credit = sum(leg.theoretical_price * leg.contracts * leg.lot_size 
                          for leg in legs if leg.action == 'SELL')
        net_cost = total_cost - total_credit
        
        # Find breakeven points (simplified calculation)
        strikes = sorted(set(leg.strike for leg in legs))
        min_strike = min(strikes) if strikes else current_price
        max_strike = max(strikes) if strikes else current_price
        
        # Calculate Greeks (simplified)
        total_delta = sum(getattr(leg.greeks, 'delta', 0) * leg.contracts * leg.lot_size 
                         * (1 if leg.action == 'BUY' else -1) for leg in legs)
        total_gamma = sum(getattr(leg.greeks, 'gamma', 0) * leg.contracts * leg.lot_size 
                         * (1 if leg.action == 'BUY' else -1) for leg in legs)
        total_theta = sum(getattr(leg.greeks, 'theta', 0) * leg.contracts * leg.lot_size 
                         * (1 if leg.action == 'BUY' else -1) for leg in legs)
        total_vega = sum(getattr(leg.greeks, 'vega', 0) * leg.contracts * leg.lot_size 
                        * (1 if leg.action == 'BUY' else -1) for leg in legs)
        
        # Risk analysis
        has_unlimited_risk = any(leg.action == 'SELL' for leg in legs) and \
                           not self._has_protective_legs(legs)
        
        return {
            'cost_analysis': {
                'total_investment': total_cost,
                'total_credit': total_credit,
                'net_cost': net_cost,
                'cost_per_point': net_cost / max(1, current_price * 0.01),  # Cost per 1% move
            },
            'risk_metrics': {
                'max_risk': 'Unlimited' if has_unlimited_risk else f"₹{net_cost:,.0f}",
                'has_unlimited_risk': has_unlimited_risk,
                'protective_legs': self._has_protective_legs(legs),
                'risk_reward_ratio': abs(net_cost / max(1, total_credit)) if total_credit > 0 else 'N/A'
            },
            'greeks_summary': {
                'total_delta': total_delta,
                'total_gamma': total_gamma,
                'total_theta': total_theta,
                'total_vega': total_vega,
                'delta_neutral': abs(total_delta) < 0.1,
                'gamma_risk': abs(total_gamma) > 1.0,
                'theta_positive': total_theta > 0
            },
            'position_details': {
                'total_legs': len(legs),
                'total_contracts': sum(leg.contracts for leg in legs),
                'strike_range': f"₹{min_strike} - ₹{max_strike}",
                'current_moneyness': current_price / ((min_strike + max_strike) / 2),
                'expiry': legs[0].expiry if legs else None
            },
            'liquidity_score': {
                'average_liquidity': sum(leg.liquidity_score for leg in legs) / len(legs),
                'min_liquidity': min(leg.liquidity_score for leg in legs),
                'liquidity_warning': any(leg.liquidity_score < 0.3 for leg in legs)
            }
        }

    def _has_protective_legs(self, legs: List[OptionsLeg]) -> bool:
        """Check if strategy has protective legs for sold options"""
        
        sell_legs = [leg for leg in legs if leg.action == 'SELL']
        buy_legs = [leg for leg in legs if leg.action == 'BUY']
        
        if not sell_legs:
            return True  # No selling, so protected
        
        # Check if each sold option has protection
        for sell_leg in sell_legs:
            has_protection = False
            
            for buy_leg in buy_legs:
                if buy_leg.option_type == sell_leg.option_type:
                    # For calls: protection is higher strike
                    # For puts: protection is lower strike
                    if sell_leg.option_type == 'call' and buy_leg.strike > sell_leg.strike:
                        has_protection = True
                        break
                    elif sell_leg.option_type == 'put' and buy_leg.strike < sell_leg.strike:
                        has_protection = True
                        break
            
            if not has_protection:
                return False
        
        return True

    def get_tier_allowed_strategies(self, capital_tier: CapitalTier) -> List[StrategyType]:
        """Get list of strategies allowed for a capital tier"""
        
        if capital_tier == CapitalTier.TIER_1:
            return [
                StrategyType.BUY_CALL,
                StrategyType.BUY_PUT
            ]
        
        elif capital_tier == CapitalTier.TIER_2:
            return [
                StrategyType.BUY_CALL,
                StrategyType.BUY_PUT,
                StrategyType.LONG_STRADDLE,
                StrategyType.LONG_STRANGLE,
                StrategyType.BULL_CALL_SPREAD,  # Buy both legs
                StrategyType.BEAR_PUT_SPREAD    # Buy both legs
            ]
        
        else:  # TIER_3
            return list(StrategyType)  # All strategies allowed

    def estimate_margin_requirement(self, legs: List[OptionsLeg], 
                                  underlying_price: float) -> Dict:
        """Estimate margin requirements for the strategy"""
        
        margin_info = {
            'total_margin': 0.0,
            'span_margin': 0.0,
            'exposure_margin': 0.0,
            'premium_received': 0.0,
            'premium_paid': 0.0,
            'net_margin': 0.0,
            'margin_breakdown': []
        }
        
        premium_paid = 0.0
        premium_received = 0.0
        total_span = 0.0
        total_exposure = 0.0
        
        for leg in legs:
            leg_value = leg.theoretical_price * leg.contracts * leg.lot_size
            
            if leg.action == 'BUY':
                premium_paid += leg_value
                # No additional margin for buying options
                margin_info['margin_breakdown'].append({
                    'leg': f"{leg.action} {leg.contracts}x {leg.option_type.upper()} {leg.strike}",
                    'premium': leg_value,
                    'margin': 0,
                    'type': 'Premium Payment'
                })
            
            else:  # SELL
                premium_received += leg_value
                
                # Calculate SPAN and Exposure margin for selling
                contract_value = leg.strike * leg.contracts * leg.lot_size
                
                # SPAN margin (varies by moneyness and instrument type)
                moneyness = leg.strike / underlying_price
                
                if abs(moneyness - 1.0) < 0.05:  # ATM
                    span_rate = 0.05  # 5% for ATM
                elif abs(moneyness - 1.0) < 0.10:  # Near ATM
                    span_rate = 0.04  # 4% for near ATM
                else:  # OTM
                    span_rate = 0.03  # 3% for OTM
                
                span_margin = contract_value * span_rate
                exposure_margin = contract_value * 0.025  # 2.5% exposure
                
                total_span += span_margin
                total_exposure += exposure_margin
                
                margin_info['margin_breakdown'].append({
                    'leg': f"{leg.action} {leg.contracts}x {leg.option_type.upper()} {leg.strike}",
                    'premium': -leg_value,  # Negative for received
                    'margin': span_margin + exposure_margin,
                    'type': 'SPAN + Exposure'
                })
        
        # Calculate net margin requirement
        gross_margin = total_span + total_exposure
        net_margin = max(0, gross_margin - premium_received + premium_paid)
        
        margin_info.update({
            'total_margin': gross_margin,
            'span_margin': total_span,
            'exposure_margin': total_exposure,
            'premium_received': premium_received,
            'premium_paid': premium_paid,
            'net_margin': net_margin
        })
        
        return margin_info

    async def execute_strategy(self, legs: List[OptionsLeg], 
                             dry_run: bool = True) -> Dict:
        """Execute all legs of the strategy"""
        
        execution_results = {
            'strategy_executed': True,
            'total_legs': len(legs),
            'successful_legs': 0,
            'failed_legs': 0,
            'leg_results': [],
            'total_cost': 0.0,
            'total_credit': 0.0,
            'execution_summary': {}
        }
        
        if not legs:
            execution_results['strategy_executed'] = False
            execution_results['error'] = "No legs to execute"
            return execution_results
        
        logger.info(f"Executing {len(legs)} leg strategy (Dry run: {dry_run})")
        
        for i, leg in enumerate(legs):
            try:
                logger.info(f"Executing leg {i+1}/{len(legs)}: {leg.action} {leg.contracts}x {leg.tradingsymbol}")
                
                # Execute individual leg
                result = await self.order_manager.place_options_order(leg, dry_run=dry_run)
                
                execution_results['leg_results'].append({
                    'leg_number': i + 1,
                    'leg_details': f"{leg.action} {leg.contracts}x {leg.tradingsymbol}",
                    'result': result,
                    'success': result.get('status') == 'success'
                })
                
                if result.get('status') == 'success':
                    execution_results['successful_legs'] += 1
                    
                    # Track costs and credits
                    leg_value = leg.theoretical_price * leg.contracts * leg.lot_size
                    if leg.action == 'BUY':
                        execution_results['total_cost'] += leg_value
                    else:
                        execution_results['total_credit'] += leg_value
                    
                    logger.info(f"[OK] Leg {i+1} executed successfully: {result.get('order_id', 'N/A')}")
                else:
                    execution_results['failed_legs'] += 1
                    logger.error(f"[ERROR] Leg {i+1} failed: {result.get('message', 'Unknown error')}")
                
                # Small delay between leg executions
                if not dry_run and i < len(legs) - 1:
                    await asyncio.sleep(0.5)
                
            except Exception as e:
                execution_results['failed_legs'] += 1
                execution_results['leg_results'].append({
                    'leg_number': i + 1,
                    'leg_details': f"{leg.action} {leg.contracts}x {leg.tradingsymbol}",
                    'result': {'status': 'error', 'message': str(e)},
                    'success': False
                })
                logger.error(f"[ERROR] Exception executing leg {i+1}: {e}")
        
        # Calculate execution summary
        success_rate = execution_results['successful_legs'] / len(legs) * 100
        net_cost = execution_results['total_cost'] - execution_results['total_credit']
        
        execution_results['execution_summary'] = {
            'success_rate': success_rate,
            'all_legs_successful': execution_results['successful_legs'] == len(legs),
            'partial_execution': 0 < execution_results['successful_legs'] < len(legs),
            'net_cost': net_cost,
            'execution_type': 'Credit Strategy' if net_cost < 0 else 'Debit Strategy',
            'dry_run': dry_run
        }
        
        # Update strategy execution status
        if execution_results['successful_legs'] == len(legs):
            logger.info(f"[OK] Strategy fully executed: {execution_results['successful_legs']}/{len(legs)} legs")
        elif execution_results['successful_legs'] > 0:
            logger.warning(f"[WARNING] Partial execution: {execution_results['successful_legs']}/{len(legs)} legs")
            execution_results['strategy_executed'] = False
        else:
            logger.error(f"[ERROR] Strategy execution failed: 0/{len(legs)} legs executed")
            execution_results['strategy_executed'] = False
        
        return execution_results

    def get_strategy_monitoring_params(self, strategy_type: StrategyType, 
                                     legs: List[OptionsLeg]) -> Dict:
        """Get monitoring parameters specific to the strategy"""
        
        monitoring_params = {
            'profit_targets': [],
            'stop_losses': [],
            'time_decay_sensitivity': 'medium',
            'delta_hedge_threshold': 0.2,
            'gamma_risk_level': 'medium',
            'key_monitoring_points': [],
            'exit_conditions': []
        }
        
        # Strategy-specific monitoring
        if strategy_type in [StrategyType.BUY_CALL, StrategyType.BUY_PUT]:
            monitoring_params.update({
                'profit_targets': ['25%', '50%', '100%'],
                'stop_losses': ['50% of premium'],
                'time_decay_sensitivity': 'high',
                'key_monitoring_points': ['Breakeven level', 'Time to expiry'],
                'exit_conditions': ['50% loss', '100% gain', '1 week to expiry']
            })
        
        elif strategy_type in [StrategyType.LONG_STRADDLE, StrategyType.LONG_STRANGLE]:
            monitoring_params.update({
                'profit_targets': ['20%', '40%', '60%'],
                'stop_losses': ['50% of premium', 'Volatility collapse'],
                'time_decay_sensitivity': 'very_high',
                'key_monitoring_points': ['Both breakeven levels', 'Implied volatility'],
                'exit_conditions': ['50% loss', '60% gain', 'IV drop below 15%']
            })
        
        elif strategy_type == StrategyType.IRON_CONDOR:
            monitoring_params.update({
                'profit_targets': ['25%', '50%', '75%'],
                'stop_losses': ['2x credit received', 'Breakout of wings'],
                'time_decay_sensitivity': 'low',
                'gamma_risk_level': 'high',
                'key_monitoring_points': ['Short strike levels', 'Wing protection'],
                'exit_conditions': ['75% profit', '200% loss', 'Price near wings']
            })
        
        elif strategy_type in [StrategyType.SHORT_STRADDLE, StrategyType.SHORT_STRANGLE]:
            monitoring_params.update({
                'profit_targets': ['25%', '50%', '75%'],
                'stop_losses': ['2x credit received', 'Immediate on breakout'],
                'time_decay_sensitivity': 'beneficial',
                'gamma_risk_level': 'very_high',
                'key_monitoring_points': ['Breakeven levels', 'Volatility expansion'],
                'exit_conditions': ['75% profit', '200% loss', 'Volatility spike']
            })
        
        # Add position-specific parameters
        if legs:
            strikes = [leg.strike for leg in legs]
            monitoring_params['position_specifics'] = {
                'strike_range': f"₹{min(strikes)} - ₹{max(strikes)}",
                'total_contracts': sum(leg.contracts for leg in legs),
                'expiry_date': legs[0].expiry,
                'net_position_type': 'Credit' if sum(leg.theoretical_price * leg.contracts * leg.lot_size * (-1 if leg.action == 'SELL' else 1) for leg in legs) < 0 else 'Debit'
            }
        
        return monitoring_params
    
class LearningSystem:
    """Advanced learning and pattern recognition system for trading optimization"""
    
    def __init__(self, trade_logger: IndianTradeLogger):
        self.trade_logger = trade_logger
        self.db_path = trade_logger.db_path.replace('.db', '_learning.db')
        self.timezone = trade_logger.timezone
        
        # Thread lock for learning operations
        self._learning_lock = threading.Lock()
        
        # In-memory caches for performance
        self.pattern_cache = {}
        self.strategy_performance_cache = {}
        self.last_cache_update = None
        
        # Learning parameters
        self.min_trades_for_pattern = 3
        self.confidence_threshold = 0.6
        self.pattern_decay_factor = 0.95  # Older patterns get less weight
        
        # Initialize learning database
        self._init_learning_database()
        
        logger.info("LearningSystem initialized with advanced pattern recognition")
    
    @contextmanager
    def _get_learning_db_connection(self):
        """Context manager for learning database connections"""
        conn = None
        try:
            with self._learning_lock:
                conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
                conn.execute("PRAGMA busy_timeout = 30000")
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                conn.row_factory = sqlite3.Row
                yield conn
                if conn.in_transaction:
                    conn.commit()
        except Exception as e:
            logger.error(f"Learning database error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    def _init_learning_database(self):
        """Initialize learning-specific database tables"""
        
        try:
            with self._get_learning_db_connection() as conn:
                cursor = conn.cursor()
                
                # Trade lifecycles table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trade_lifecycles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        lifecycle_id TEXT UNIQUE NOT NULL,
                        signal_id INTEGER,
                        ticker TEXT NOT NULL,
                        strategy_type TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        
                        -- Pre-trade data
                        signal_confidence REAL,
                        market_conditions TEXT,
                        technical_setup TEXT,
                        strategy_reasoning TEXT,
                        entry_expectations TEXT,
                        
                        -- During trade data
                        entry_timestamp DATETIME,
                        entry_prices TEXT,
                        position_updates TEXT,
                        decision_points TEXT,
                        
                        -- Post-trade data
                        exit_timestamp DATETIME,
                        exit_prices TEXT,
                        final_pnl REAL,
                        exit_reason TEXT,
                        time_in_position INTEGER,
                        
                        -- Analysis data
                        what_worked TEXT,
                        what_failed TEXT,
                        lessons_learned TEXT,
                        max_profit_reached REAL,
                        max_drawdown REAL,
                        technical_accuracy REAL,
                        
                        -- Meta data
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Pattern recognition table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS learned_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_hash TEXT UNIQUE NOT NULL,
                        pattern_name TEXT NOT NULL,
                        pattern_data TEXT NOT NULL,
                        
                        -- Performance metrics
                        total_occurrences INTEGER DEFAULT 0,
                        successful_occurrences INTEGER DEFAULT 0,
                        total_pnl REAL DEFAULT 0,
                        avg_pnl REAL DEFAULT 0,
                        max_pnl REAL DEFAULT 0,
                        min_pnl REAL DEFAULT 0,
                        avg_time_to_profit INTEGER DEFAULT 0,
                        
                        -- Context data
                        market_conditions TEXT,
                        time_patterns TEXT,
                        volatility_ranges TEXT,
                        
                        -- Learning metadata
                        confidence_score REAL DEFAULT 0,
                        last_seen DATETIME,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Strategy performance tracking
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_type TEXT NOT NULL,
                        market_condition TEXT NOT NULL,
                        time_period TEXT NOT NULL,
                        
                        -- Performance metrics
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        total_pnl REAL DEFAULT 0,
                        avg_pnl_per_trade REAL DEFAULT 0,
                        win_rate REAL DEFAULT 0,
                        profit_factor REAL DEFAULT 0,
                        avg_time_in_position INTEGER DEFAULT 0,
                        
                        -- Risk metrics
                        max_drawdown REAL DEFAULT 0,
                        sharpe_ratio REAL DEFAULT 0,
                        volatility REAL DEFAULT 0,
                        
                        -- Context
                        capital_tier TEXT,
                        volatility_regime TEXT,
                        
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                        
                        UNIQUE(strategy_type, market_condition, time_period, capital_tier, volatility_regime)
                    )
                """)
                
                # Market regime detection
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS market_regimes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE NOT NULL UNIQUE,
                        regime_type TEXT NOT NULL,
                        volatility_level TEXT NOT NULL,
                        trend_strength REAL,
                        market_sentiment TEXT,
                        sector_rotation TEXT,
                        regime_confidence REAL,
                        
                        -- Performance in this regime
                        trades_executed INTEGER DEFAULT 0,
                        regime_pnl REAL DEFAULT 0,
                        best_strategies TEXT,
                        
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Decision tracking for continuous improvement
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS decision_tracking (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        lifecycle_id TEXT NOT NULL,
                        decision_timestamp DATETIME NOT NULL,
                        decision_type TEXT NOT NULL,
                        decision_data TEXT NOT NULL,
                        
                        -- Context at decision time
                        current_pnl REAL,
                        time_in_position INTEGER,
                        market_conditions TEXT,
                        technical_signals TEXT,
                        
                        -- Outcome tracking
                        decision_outcome TEXT,
                        outcome_pnl REAL,
                        was_optimal BOOLEAN,
                        
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        
                        FOREIGN KEY (lifecycle_id) REFERENCES trade_lifecycles (lifecycle_id)
                    )
                """)
                
                # Learning insights cache
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS learning_insights (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        insight_type TEXT NOT NULL,
                        insight_data TEXT NOT NULL,
                        confidence_score REAL,
                        applicable_conditions TEXT,
                        
                        -- Validation
                        times_applied INTEGER DEFAULT 0,
                        times_successful INTEGER DEFAULT 0,
                        avg_improvement REAL DEFAULT 0,
                        
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_applied DATETIME,
                        
                        UNIQUE(insight_type, insight_data)
                    )
                """)
                
                # Create indices for performance
                indices = [
                    "CREATE INDEX IF NOT EXISTS idx_lifecycles_ticker ON trade_lifecycles(ticker)",
                    "CREATE INDEX IF NOT EXISTS idx_lifecycles_strategy ON trade_lifecycles(strategy_type)",
                    "CREATE INDEX IF NOT EXISTS idx_lifecycles_timestamp ON trade_lifecycles(entry_timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_patterns_name ON learned_patterns(pattern_name)",
                    "CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON learned_patterns(confidence_score DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_strategy_perf ON strategy_performance(strategy_type, market_condition)",
                    "CREATE INDEX IF NOT EXISTS idx_decisions_lifecycle ON decision_tracking(lifecycle_id)",
                    "CREATE INDEX IF NOT EXISTS idx_regimes_date ON market_regimes(date DESC)"
                ]
                
                for index_sql in indices:
                    cursor.execute(index_sql)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error initializing learning database: {e}")
            raise
    
    def start_trade_lifecycle(self, signal: 'TradingSignal', strategy_type: 'StrategyType', 
                            market_conditions: Dict, technical_setup: Dict,
                            strategy_reasoning: str = "") -> int:
        """Start a new trade lifecycle for learning"""
        
        try:
            # Generate unique lifecycle ID
            lifecycle_id = f"{signal.ticker}_{strategy_type.value}_{int(datetime.now().timestamp())}"
            
            with self._get_learning_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO trade_lifecycles (
                        lifecycle_id, ticker, strategy_type, direction,
                        signal_confidence, market_conditions, technical_setup,
                        strategy_reasoning, entry_expectations, entry_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    lifecycle_id,
                    signal.ticker,
                    strategy_type.value,
                    signal.direction,
                    signal.confidence,
                    json.dumps(market_conditions, default=str),
                    json.dumps(technical_setup, default=str),
                    strategy_reasoning,
                    json.dumps({
                        'expected_direction': signal.direction,
                        'confidence': signal.confidence,
                        'current_price': signal.current_price,
                        'timestamp': signal.timestamp.isoformat()
                    }, default=str),
                    datetime.now(self.timezone).isoformat()
                ))
                
                learning_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Started trade lifecycle: {lifecycle_id} (ID: {learning_id})")
                return learning_id
                
        except Exception as e:
            logger.error(f"Error starting trade lifecycle: {e}")
            return -1
    
    def update_trade_lifecycle(self, lifecycle_id: int, stage: str, data: Dict):
        """Update trade lifecycle with decision points and progress"""
        
        try:
            with self._get_learning_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get current lifecycle data
                cursor.execute("""
                    SELECT lifecycle_id, position_updates, decision_points 
                    FROM trade_lifecycles WHERE id = ?
                """, (lifecycle_id,))
                
                result = cursor.fetchone()
                if not result:
                    logger.warning(f"Lifecycle {lifecycle_id} not found")
                    return
                
                current_lifecycle_id = result['lifecycle_id']
                
                # Parse existing data
                try:
                    position_updates = json.loads(result['position_updates'] or '[]')
                    decision_points = json.loads(result['decision_points'] or '[]')
                except json.JSONDecodeError:
                    position_updates = []
                    decision_points = []
                
                # Update based on stage
                if stage == 'entry':
                    # Update entry data
                    cursor.execute("""
                        UPDATE trade_lifecycles 
                        SET entry_prices = ?, entry_timestamp = ?, updated_at = ?
                        WHERE id = ?
                    """, (
                        json.dumps(data.get('entry_prices', []), default=str),
                        data.get('entry_timestamp', datetime.now(self.timezone).isoformat()),
                        datetime.now(self.timezone).isoformat(),
                        lifecycle_id
                    ))
                
                elif stage == 'position_update':
                    # Add position update
                    position_updates.append({
                        'timestamp': datetime.now(self.timezone).isoformat(),
                        'current_pnl': data.get('current_pnl', 0),
                        'unrealized_pnl': data.get('unrealized_pnl', 0),
                        'market_price': data.get('market_price', 0),
                        'time_in_position': data.get('time_in_position', 0)
                    })
                    
                    cursor.execute("""
                        UPDATE trade_lifecycles 
                        SET position_updates = ?, updated_at = ?
                        WHERE id = ?
                    """, (
                        json.dumps(position_updates, default=str),
                        datetime.now(self.timezone).isoformat(),
                        lifecycle_id
                    ))
                
                elif stage == 'decision_point':
                    # Add decision point
                    decision_point = {
                        'timestamp': datetime.now(self.timezone).isoformat(),
                        'decision_type': data.get('decision', 'unknown'),
                        'reason': data.get('reason', ''),
                        'confidence': data.get('confidence', 0),
                        'current_pnl': data.get('current_pnl', 0),
                        'market_conditions': data.get('market_conditions', {}),
                        'technical_signals': data.get('technical_signals', {})
                    }
                    
                    decision_points.append(decision_point)
                    
                    cursor.execute("""
                        UPDATE trade_lifecycles 
                        SET decision_points = ?, updated_at = ?
                        WHERE id = ?
                    """, (
                        json.dumps(decision_points, default=str),
                        datetime.now(self.timezone).isoformat(),
                        lifecycle_id
                    ))
                    
                    # Also log to decision tracking table
                    cursor.execute("""
                        INSERT INTO decision_tracking (
                            lifecycle_id, decision_timestamp, decision_type, decision_data,
                            current_pnl, market_conditions, technical_signals
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        current_lifecycle_id,
                        decision_point['timestamp'],
                        decision_point['decision_type'],
                        json.dumps(data, default=str),
                        data.get('current_pnl', 0),
                        json.dumps(data.get('market_conditions', {}), default=str),
                        json.dumps(data.get('technical_signals', {}), default=str)
                    ))
                
                elif stage == 'exit':
                    # Update exit data
                    time_in_position = self._calculate_time_in_position(lifecycle_id)
                    
                    cursor.execute("""
                        UPDATE trade_lifecycles 
                        SET exit_timestamp = ?, exit_prices = ?, final_pnl = ?,
                            exit_reason = ?, time_in_position = ?, updated_at = ?
                        WHERE id = ?
                    """, (
                        data.get('exit_timestamp', datetime.now(self.timezone).isoformat()),
                        json.dumps(data.get('exit_prices', []), default=str),
                        data.get('final_pnl', 0),
                        data.get('exit_reason', ''),
                        time_in_position,
                        datetime.now(self.timezone).isoformat(),
                        lifecycle_id
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating trade lifecycle: {e}")
    
    def analyze_completed_trade(self, lifecycle_id: int) -> Dict:
        """Analyze completed trade and extract learning insights"""
        
        try:
            with self._get_learning_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get complete lifecycle data
                cursor.execute("""
                    SELECT * FROM trade_lifecycles WHERE id = ?
                """, (lifecycle_id,))
                
                lifecycle = cursor.fetchone()
                if not lifecycle:
                    return {'error': 'Lifecycle not found'}
                
                # Parse JSON data
                try:
                    market_conditions = json.loads(lifecycle['market_conditions'] or '{}')
                    technical_setup = json.loads(lifecycle['technical_setup'] or '{}')
                    position_updates = json.loads(lifecycle['position_updates'] or '[]')
                    decision_points = json.loads(lifecycle['decision_points'] or '[]')
                except json.JSONDecodeError:
                    market_conditions = {}
                    technical_setup = {}
                    position_updates = []
                    decision_points = []
                
                # Analyze trade performance
                analysis = self._analyze_trade_performance(lifecycle, position_updates, decision_points)
                
                # Extract patterns
                patterns = self._extract_trade_patterns(lifecycle, market_conditions, technical_setup)
                
                # Update pattern database
                for pattern in patterns:
                    self._update_pattern_performance(pattern, lifecycle['final_pnl'] > 0, lifecycle['final_pnl'])
                
                # Update strategy performance
                self._update_strategy_performance(lifecycle, market_conditions)
                
                # Generate lessons learned
                lessons = self._generate_lessons_learned(analysis, patterns, lifecycle)
                
                # Update lifecycle with analysis
                cursor.execute("""
                    UPDATE trade_lifecycles 
                    SET what_worked = ?, what_failed = ?, lessons_learned = ?,
                        max_profit_reached = ?, max_drawdown = ?, technical_accuracy = ?
                    WHERE id = ?
                """, (
                    json.dumps(analysis.get('what_worked', []), default=str),
                    json.dumps(analysis.get('what_failed', []), default=str),
                    json.dumps(lessons, default=str),
                    analysis.get('max_profit', 0),
                    analysis.get('max_drawdown', 0),
                    analysis.get('technical_accuracy', 0),
                    lifecycle_id
                ))
                
                conn.commit()
                
                return {
                    'analysis_completed': True,
                    'performance_analysis': analysis,
                    'patterns_identified': len(patterns),
                    'lessons_learned': lessons,
                    'trade_profitable': lifecycle['final_pnl'] > 0,
                    'final_pnl': lifecycle['final_pnl']
                }
                
        except Exception as e:
            logger.error(f"Error analyzing completed trade: {e}")
            return {'error': str(e)}
    
    def get_learning_insights(self) -> Dict:
        """Get comprehensive learning insights and recommendations"""
        
        try:
            # Update cache if needed
            self._update_learning_cache()
            
            with self._get_learning_db_connection() as conn:
                cursor = conn.cursor()
                
                insights = {
                    'total_trades_analyzed': 0,
                    'overall_performance': {},
                    'strategy_performance': {},
                    'pattern_success_rates': {},
                    'market_regime_performance': {},
                    'best_strategy': None,
                    'recommendations': [],
                    'learning_confidence': 0.0
                }
                
                # Overall performance metrics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN final_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        AVG(final_pnl) as avg_pnl,
                        SUM(final_pnl) as total_pnl,
                        AVG(time_in_position) as avg_time_in_position,
                        MAX(final_pnl) as best_trade,
                        MIN(final_pnl) as worst_trade
                    FROM trade_lifecycles 
                    WHERE final_pnl IS NOT NULL
                """)
                
                overall = cursor.fetchone()
                if overall and overall['total_trades'] > 0:
                    insights['total_trades_analyzed'] = overall['total_trades']
                    insights['overall_performance'] = {
                        'win_rate': overall['winning_trades'] / overall['total_trades'],
                        'avg_pnl': overall['avg_pnl'],
                        'total_pnl': overall['total_pnl'],
                        'avg_time_in_position_minutes': overall['avg_time_in_position'],
                        'best_trade': overall['best_trade'],
                        'worst_trade': overall['worst_trade'],
                        'profit_factor': self._calculate_profit_factor(cursor)
                    }
                
                # Strategy performance
                cursor.execute("""
                    SELECT 
                        strategy_type,
                        COUNT(*) as trades,
                        SUM(CASE WHEN final_pnl > 0 THEN 1 ELSE 0 END) as wins,
                        AVG(final_pnl) as avg_pnl,
                        SUM(final_pnl) as total_pnl
                    FROM trade_lifecycles 
                    WHERE final_pnl IS NOT NULL
                    GROUP BY strategy_type
                    ORDER BY avg_pnl DESC
                """)
                
                strategy_results = cursor.fetchall()
                for row in strategy_results:
                    if row['trades'] >= self.min_trades_for_pattern:
                        insights['strategy_performance'][row['strategy_type']] = {
                            'total_trades': row['trades'],
                            'win_rate': row['wins'] / row['trades'],
                            'avg_pnl': row['avg_pnl'],
                            'total_pnl': row['total_pnl']
                        }
                
                # Best strategy identification
                if insights['strategy_performance']:
                    best_strategy = max(
                        insights['strategy_performance'].items(),
                        key=lambda x: x[1]['avg_pnl'] if x[1]['total_trades'] >= 3 else -999999
                    )
                    insights['best_strategy'] = best_strategy[0]
                
                # Pattern success rates
                cursor.execute("""
                    SELECT 
                        pattern_name,
                        total_occurrences,
                        successful_occurrences,
                        avg_pnl,
                        confidence_score
                    FROM learned_patterns
                    WHERE total_occurrences >= ?
                    ORDER BY confidence_score DESC
                    LIMIT 10
                """, (self.min_trades_for_pattern,))
                
                pattern_results = cursor.fetchall()
                for row in pattern_results:
                    insights['pattern_success_rates'][row['pattern_name']] = {
                        'total': row['total_occurrences'],
                        'success': row['successful_occurrences'],
                        'success_rate': row['successful_occurrences'] / row['total_occurrences'],
                        'avg_pnl': row['avg_pnl'],
                        'confidence': row['confidence_score']
                    }
                
                # Market regime performance
                insights['market_regime_performance'] = self._get_regime_performance(cursor)
                
                # Generate recommendations
                insights['recommendations'] = self._generate_learning_recommendations(insights)
                
                # Calculate learning confidence
                insights['learning_confidence'] = self._calculate_learning_confidence(insights)
                
                return insights
                
        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return {
                'total_trades_analyzed': 0,
                'error': str(e)
            }
    
    def get_strategy_recommendation(self, signal: 'TradingSignal', market_conditions: Dict, 
                                    available_strategies: List['StrategyType']) -> Dict:
        """Get AI-powered strategy recommendation based on learning"""
        
        try:
            with self._get_learning_db_connection() as conn:
                cursor = conn.cursor()
                
                # Analyze current market conditions
                current_regime = self._detect_market_regime(market_conditions)
                
                # Get strategy performance in similar conditions
                strategy_scores = {}
                
                for strategy in available_strategies:
                    cursor.execute("""
                        SELECT 
                            AVG(final_pnl) as avg_pnl,
                            COUNT(*) as sample_size,
                            SUM(CASE WHEN final_pnl > 0 THEN 1 ELSE 0 END) as wins
                        FROM trade_lifecycles tl
                        JOIN market_regimes mr ON DATE(tl.entry_timestamp) = mr.date
                        WHERE tl.strategy_type = ? 
                        AND tl.direction = ?
                        AND mr.regime_type = ?
                        AND tl.final_pnl IS NOT NULL
                    """, (strategy.value, signal.direction, current_regime.get('type', 'UNKNOWN')))
                    
                    result = cursor.fetchone()
                    
                    if result and result['sample_size'] >= 2:
                        win_rate = result['wins'] / result['sample_size']
                        confidence = min(result['sample_size'] / 10.0, 1.0)  # More samples = more confidence
                        
                        # Score based on avg PnL, win rate, and confidence
                        score = (result['avg_pnl'] * 0.4 + 
                                win_rate * 100 * 0.4 + 
                                confidence * 20 * 0.2)
                        
                        strategy_scores[strategy] = {
                            'score': score,
                            'avg_pnl': result['avg_pnl'],
                            'win_rate': win_rate,
                            'sample_size': result['sample_size'],
                            'confidence': confidence
                        }
                
                # Pattern-based adjustment
                pattern_adjustments = self._get_pattern_based_adjustments(signal, market_conditions)
                
                # Apply pattern adjustments
                for strategy, adjustment in pattern_adjustments.items():
                    if strategy in strategy_scores:
                        strategy_scores[strategy]['score'] += adjustment
                        strategy_scores[strategy]['pattern_adjustment'] = adjustment
                
                # Find best strategy
                if strategy_scores:
                    best_strategy = max(strategy_scores.items(), key=lambda x: x[1]['score'])
                    
                    return {
                        'recommended_strategy': best_strategy[0],
                        'confidence': best_strategy[1]['confidence'],
                        'expected_performance': {
                            'avg_pnl': best_strategy[1]['avg_pnl'],
                            'win_rate': best_strategy[1]['win_rate'],
                            'sample_size': best_strategy[1]['sample_size']
                        },
                        'all_strategy_scores': {k.value: v for k, v in strategy_scores.items()},
                        'market_regime': current_regime,
                        'reasoning': self._generate_strategy_reasoning(best_strategy[0], best_strategy[1], current_regime)
                    }
                
                # Fallback to default if no learning data
                return {
                    'recommended_strategy': available_strategies[0] if available_strategies else None,
                    'confidence': 0.1,
                    'reasoning': 'Insufficient learning data - using default strategy',
                    'market_regime': current_regime
                }
                
        except Exception as e:
            logger.error(f"Error getting strategy recommendation: {e}")
            return {
                'recommended_strategy': None,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def should_exit_position(self, lifecycle_id: int, current_pnl: float, 
                            time_in_position: int, market_conditions: Dict) -> Dict:
        """AI-powered exit decision based on learned patterns"""
        
        try:
            with self._get_learning_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get current trade details
                cursor.execute("""
                    SELECT * FROM trade_lifecycles WHERE id = ?
                """, (lifecycle_id,))
                
                lifecycle = cursor.fetchone()
                if not lifecycle:
                    return {'should_exit': False, 'reason': 'Lifecycle not found'}
                
                # Analyze similar historical situations
                cursor.execute("""
                    SELECT 
                        final_pnl,
                        time_in_position as final_time,
                        exit_reason,
                        max_profit_reached,
                        max_drawdown
                    FROM trade_lifecycles
                    WHERE strategy_type = ?
                    AND direction = ?
                    AND ABS(signal_confidence - ?) < 0.2
                    AND final_pnl IS NOT NULL
                """, (lifecycle['strategy_type'], lifecycle['direction'], lifecycle['signal_confidence']))
                
                similar_trades = cursor.fetchall()
                
                if len(similar_trades) < 3:
                    return {
                        'should_exit': False,
                        'confidence': 0.1,
                        'reason': 'Insufficient historical data for AI decision'
                    }
                
                # Analyze patterns
                exit_analysis = self._analyze_exit_patterns(
                    similar_trades, current_pnl, time_in_position, market_conditions
                )
                
                return exit_analysis
                
        except Exception as e:
            logger.error(f"Error in AI exit decision: {e}")
            return {'should_exit': False, 'error': str(e)}
    
    # =================== HELPER METHODS ===================
    
    def _calculate_time_in_position(self, lifecycle_id: int) -> int:
        """Calculate time in position in minutes"""
        
        try:
            with self._get_learning_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT entry_timestamp, exit_timestamp 
                    FROM trade_lifecycles WHERE id = ?
                """, (lifecycle_id,))
                
                result = cursor.fetchone()
                if result and result['entry_timestamp'] and result['exit_timestamp']:
                    entry = datetime.fromisoformat(result['entry_timestamp'])
                    exit_time = datetime.fromisoformat(result['exit_timestamp'])
                    return int((exit_time - entry).total_seconds() / 60)
                
        except Exception as e:
            logger.error(f"Error calculating time in position: {e}")
        
        return 0
    
    def _analyze_trade_performance(self, lifecycle: sqlite3.Row, 
                                    position_updates: List, decision_points: List) -> Dict:
        """Analyze individual trade performance"""
        
        analysis = {
            'what_worked': [],
            'what_failed': [],
            'max_profit': 0,
            'max_drawdown': 0,
            'technical_accuracy': 0,
            'decision_quality': 0
        }
        
        try:
            # Analyze P&L progression
            if position_updates:
                pnls = [update.get('current_pnl', 0) for update in position_updates]
                analysis['max_profit'] = max(pnls) if pnls else 0
                analysis['max_drawdown'] = min(pnls) if pnls else 0
                
                # Check if we gave back profits
                if analysis['max_profit'] > 100 and lifecycle['final_pnl'] < analysis['max_profit'] * 0.5:
                    analysis['what_failed'].append('Gave back significant profits')
                elif lifecycle['final_pnl'] > 0 and lifecycle['final_pnl'] >= analysis['max_profit'] * 0.8:
                    analysis['what_worked'].append('Captured most of available profit')
            
            # Analyze decision quality
            if decision_points:
                good_decisions = 0
                total_decisions = len(decision_points)
                
                for decision in decision_points:
                    decision_type = decision.get('decision_type', '')
                    pnl_at_decision = decision.get('current_pnl', 0)
                    
                    # Simple heuristic: holding during profits was good, exiting during losses was good
                    if decision_type == 'CONTINUE' and pnl_at_decision > 0:
                        good_decisions += 1
                    elif decision_type in ['EXIT', 'TAKE_PROFIT'] and pnl_at_decision < 0:
                        good_decisions += 1
                
                analysis['decision_quality'] = good_decisions / total_decisions if total_decisions > 0 else 0
            
            # Technical accuracy assessment
            expected_direction = lifecycle['direction']
            if lifecycle['final_pnl'] > 0:
                analysis['technical_accuracy'] = 1.0
                analysis['what_worked'].append(f'Correct {expected_direction} direction call')
            else:
                analysis['technical_accuracy'] = 0.0
                analysis['what_failed'].append(f'Incorrect {expected_direction} direction call')
            
            # Time-based analysis
            if lifecycle['time_in_position']:
                if lifecycle['time_in_position'] < 30 and lifecycle['final_pnl'] > 0:
                    analysis['what_worked'].append('Quick profitable exit')
                elif lifecycle['time_in_position'] > 180 and lifecycle['final_pnl'] < 0:
                    analysis['what_failed'].append('Held losing position too long')
            
        except Exception as e:
            logger.error(f"Error analyzing trade performance: {e}")
        
        return analysis
    
    def _extract_trade_patterns(self, lifecycle: sqlite3.Row, 
                                market_conditions: Dict, technical_setup: Dict) -> List[Dict]:
        """Extract recognizable patterns from trade data"""
        
        patterns = []
        
        try:
            # Market condition patterns
            volatility = market_conditions.get('volatility', 'NORMAL')
            trend = technical_setup.get('trend', 'NEUTRAL')
            time_of_day = datetime.fromisoformat(lifecycle['entry_timestamp']).hour
            
            # Pattern 1: Time-based pattern
            if 9 <= time_of_day <= 11:
                time_pattern = 'MORNING_SESSION'
            elif 11 <= time_of_day <= 14:
                time_pattern = 'MIDDAY_SESSION'
            else:
                time_pattern = 'AFTERNOON_SESSION'
            
            patterns.append({
                'name': f"{lifecycle['strategy_type']}_{time_pattern}",
                'type': 'TIME_BASED',
                'data': {
                    'strategy': lifecycle['strategy_type'],
                    'time_session': time_pattern,
                    'hour': time_of_day
                }
            })
            
            # Pattern 2: Volatility-strategy pattern
            patterns.append({
                'name': f"{lifecycle['strategy_type']}_VOL_{volatility}",
                'type': 'VOLATILITY_BASED',
                'data': {
                    'strategy': lifecycle['strategy_type'],
                    'volatility_regime': volatility,
                    'direction': lifecycle['direction']
                }
            })
            
            # Pattern 3: Trend-direction pattern
            patterns.append({
                'name': f"{trend}_{lifecycle['direction']}_ALIGNMENT",
                'type': 'TREND_ALIGNMENT',
                'data': {
                    'trend': trend,
                    'direction': lifecycle['direction'],
                    'strategy': lifecycle['strategy_type']
                }
            })
            
            # Pattern 4: Confidence-outcome pattern
            confidence_bucket = 'HIGH' if lifecycle['signal_confidence'] > 0.8 else 'MEDIUM' if lifecycle['signal_confidence'] > 0.6 else 'LOW'
            patterns.append({
                'name': f"{confidence_bucket}_CONFIDENCE_{lifecycle['strategy_type']}",
                'type': 'CONFIDENCE_BASED',
                'data': {
                    'confidence_level': confidence_bucket,
                    'strategy': lifecycle['strategy_type'],
                    'actual_confidence': lifecycle['signal_confidence']
                }
            })
            
        except Exception as e:
            logger.error(f"Error extracting patterns: {e}")
        
        return patterns
    
    def _update_pattern_performance(self, pattern: Dict, success: bool, pnl: float):
        """Update pattern performance in database"""
        
        try:
            pattern_hash = hashlib.md5(
                json.dumps(pattern['data'], sort_keys=True).encode()
            ).hexdigest()
            
            with self._get_learning_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check if pattern exists
                cursor.execute("""
                    SELECT * FROM learned_patterns WHERE pattern_hash = ?
                """, (pattern_hash,))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing pattern
                    new_occurrences = existing['total_occurrences'] + 1
                    new_successful = existing['successful_occurrences'] + (1 if success else 0)
                    new_total_pnl = existing['total_pnl'] + pnl
                    new_avg_pnl = new_total_pnl / new_occurrences
                    
                    # Update max/min PnL
                    new_max_pnl = max(existing['max_pnl'], pnl)
                    new_min_pnl = min(existing['min_pnl'], pnl)
                    
                    # Calculate confidence score
                    success_rate = new_successful / new_occurrences
                    sample_confidence = min(new_occurrences / 10.0, 1.0)
                    confidence_score = success_rate * sample_confidence
                    
                    cursor.execute("""
                        UPDATE learned_patterns 
                        SET total_occurrences = ?, successful_occurrences = ?,
                            total_pnl = ?, avg_pnl = ?, max_pnl = ?, min_pnl = ?,
                            confidence_score = ?, last_seen = ?, updated_at = ?
                        WHERE pattern_hash = ?
                    """, (
                        new_occurrences, new_successful, new_total_pnl, new_avg_pnl,
                        new_max_pnl, new_min_pnl, confidence_score,
                        datetime.now(self.timezone).isoformat(),
                        datetime.now(self.timezone).isoformat(),
                        pattern_hash
                    ))
                    
                else:
                    # Insert new pattern
                    cursor.execute("""
                        INSERT INTO learned_patterns (
                            pattern_hash, pattern_name, pattern_data,
                            total_occurrences, successful_occurrences,
                            total_pnl, avg_pnl, max_pnl, min_pnl,
                            confidence_score, last_seen
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pattern_hash, pattern['name'],
                        json.dumps(pattern['data'], default=str),
                        1, 1 if success else 0,
                        pnl, pnl, pnl, pnl,
                        0.1,  # Low initial confidence
                        datetime.now(self.timezone).isoformat()
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating pattern performance: {e}")
    
    def _update_strategy_performance(self, lifecycle: sqlite3.Row, market_conditions: Dict):
        """Update strategy performance metrics"""
        
        try:
            with self._get_learning_db_connection() as conn:
                cursor = conn.cursor()
                
                # Determine market condition and time period
                market_condition = self._classify_market_condition(market_conditions)
                time_period = self._get_time_period(lifecycle['entry_timestamp'])
                volatility_regime = market_conditions.get('volatility', 'NORMAL')
                
                # Try to get existing record
                cursor.execute("""
                    SELECT * FROM strategy_performance 
                    WHERE strategy_type = ? AND market_condition = ? 
                    AND time_period = ? AND volatility_regime = ?
                """, (lifecycle['strategy_type'], market_condition, time_period, volatility_regime))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing record
                    new_total = existing['total_trades'] + 1
                    new_winning = existing['winning_trades'] + (1 if lifecycle['final_pnl'] > 0 else 0)
                    new_total_pnl = existing['total_pnl'] + lifecycle['final_pnl']
                    new_avg_pnl = new_total_pnl / new_total
                    new_win_rate = new_winning / new_total
                    
                    cursor.execute("""
                        UPDATE strategy_performance 
                        SET total_trades = ?, winning_trades = ?, total_pnl = ?,
                            avg_pnl_per_trade = ?, win_rate = ?, last_updated = ?
                        WHERE id = ?
                    """, (
                        new_total, new_winning, new_total_pnl,
                        new_avg_pnl, new_win_rate,
                        datetime.now(self.timezone).isoformat(),
                        existing['id']
                    ))
                else:
                    # Insert new record
                    cursor.execute("""
                        INSERT INTO strategy_performance (
                            strategy_type, market_condition, time_period,
                            total_trades, winning_trades, total_pnl,
                            avg_pnl_per_trade, win_rate, volatility_regime
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        lifecycle['strategy_type'], market_condition, time_period,
                        1, 1 if lifecycle['final_pnl'] > 0 else 0, lifecycle['final_pnl'],
                        lifecycle['final_pnl'], 1.0 if lifecycle['final_pnl'] > 0 else 0.0,
                        volatility_regime
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")
    
    def _generate_lessons_learned(self, analysis: Dict, patterns: List, lifecycle: sqlite3.Row) -> List[str]:
        """Generate actionable lessons from trade analysis"""
        
        lessons = []
        
        try:
            # Performance-based lessons
            if lifecycle['final_pnl'] > 0:
                lessons.append(f"[OK] {lifecycle['strategy_type']} profitable in {lifecycle['direction']} direction")
                
                if analysis['max_profit'] > lifecycle['final_pnl'] * 1.5:
                    lessons.append("[WARNING] Consider tighter profit-taking rules to avoid giving back gains")
                else:
                    lessons.append("[OK] Good profit capture - minimal profit give-back")
            else:
                lessons.append(f"[ERROR] {lifecycle['strategy_type']} unprofitable in {lifecycle['direction']} direction")
                
                if analysis['max_profit'] > 0:
                    lessons.append("[BULB] Position had profitable moments - improve exit timing")
                else:
                    lessons.append("[BULB] Review entry criteria - position never profitable")
            
            # Time-based lessons
            if lifecycle['time_in_position']:
                if lifecycle['time_in_position'] > 120:  # More than 2 hours
                    if lifecycle['final_pnl'] > 0:
                        lessons.append("⏰ Patience paid off - long hold was profitable")
                    else:
                        lessons.append("⏰ Avoid holding losing positions beyond 2 hours")
                elif lifecycle['time_in_position'] < 30:  # Less than 30 minutes
                    if lifecycle['final_pnl'] > 0:
                        lessons.append("[ROCKET] Quick profitable exit - good momentum recognition")
                    else:
                        lessons.append("⚡ Too quick to exit - consider wider initial stops")
            
            # Pattern-based lessons
            for pattern in patterns:
                if pattern['type'] == 'TIME_BASED':
                    session = pattern['data']['time_session']
                    if lifecycle['final_pnl'] > 0:
                        lessons.append(f"📅 {session} favorable for {lifecycle['strategy_type']}")
                    else:
                        lessons.append(f"📅 Avoid {lifecycle['strategy_type']} during {session}")
            
            # Decision quality lessons
            if analysis.get('decision_quality', 0) < 0.5:
                lessons.append("[TARGET] Improve decision-making process - review exit criteria")
            elif analysis.get('decision_quality', 0) > 0.8:
                lessons.append("[TARGET] Excellent decision-making - maintain current process")
            
        except Exception as e:
            logger.error(f"Error generating lessons: {e}")
        
        return lessons
    
    def _update_learning_cache(self):
        """Update in-memory learning cache for performance"""
        
        if (self.last_cache_update and 
            datetime.now() - self.last_cache_update < timedelta(minutes=5)):
            return  # Cache is still fresh
        
        try:
            with self._get_learning_db_connection() as conn:
                # Update pattern cache
                pattern_df = pd.read_sql_query("""
                    SELECT pattern_name, total_occurrences, successful_occurrences,
                            avg_pnl, confidence_score
                    FROM learned_patterns
                    WHERE total_occurrences >= ?
                """, conn, params=(self.min_trades_for_pattern,))
                
                self.pattern_cache = pattern_df.to_dict('records')
                
                # Update strategy performance cache
                strategy_df = pd.read_sql_query("""
                    SELECT strategy_type, market_condition, avg_pnl_per_trade,
                            win_rate, total_trades
                    FROM strategy_performance
                    WHERE total_trades >= ?
                """, conn, params=(self.min_trades_for_pattern,))
                
                self.strategy_performance_cache = strategy_df.to_dict('records')
                
                self.last_cache_update = datetime.now()
                
        except Exception as e:
            logger.error(f"Error updating learning cache: {e}")
    
    def _detect_market_regime(self, market_conditions: Dict) -> Dict:
        """Detect current market regime"""
        
        # Simple regime detection - can be enhanced with more sophisticated ML
        volatility = market_conditions.get('volatility', 15)
        trend_strength = market_conditions.get('trend_strength', 0.5)
        
        if volatility > 25:
            regime_type = 'HIGH_VOLATILITY'
        elif volatility < 10:
            regime_type = 'LOW_VOLATILITY'
        else:
            regime_type = 'NORMAL_VOLATILITY'
        
        if trend_strength > 0.7:
            market_sentiment = 'STRONG_TREND'
        elif trend_strength > 0.3:
            market_sentiment = 'WEAK_TREND'
        else:
            market_sentiment = 'SIDEWAYS'
        
        return {
            'type': regime_type,
            'sentiment': market_sentiment,
            'volatility_level': volatility,
            'trend_strength': trend_strength
        }
    
    def _get_pattern_based_adjustments(self, signal: 'TradingSignal', 
                                        market_conditions: Dict) -> Dict:
        """Get strategy score adjustments based on learned patterns"""
        
        adjustments = defaultdict(float)
        
        try:
            # Check time-based patterns
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 11:
                time_session = 'MORNING_SESSION'
            elif 11 <= current_hour <= 14:
                time_session = 'MIDDAY_SESSION'
            else:
                time_session = 'AFTERNOON_SESSION'
            
            # Look for matching patterns in cache
            for pattern in self.pattern_cache:
                pattern_name = pattern['pattern_name']
                
                if time_session in pattern_name:
                    # Extract strategy from pattern name
                    for strategy_name in ['BUY_CALL', 'BUY_PUT', 'LONG_STRADDLE', 
                                        'LONG_STRANGLE', 'BULL_CALL_SPREAD', 'BEAR_PUT_SPREAD']:
                        if strategy_name in pattern_name:
                            # Positive adjustment for high-performing patterns
                            if pattern['avg_pnl'] > 50 and pattern['confidence_score'] > 0.6:
                                adjustments[strategy_name] += 10
                            elif pattern['avg_pnl'] < -20:
                                adjustments[strategy_name] -= 10
            
        except Exception as e:
            logger.error(f"Error getting pattern adjustments: {e}")
        
        return dict(adjustments)
    
    def _generate_learning_recommendations(self, insights: Dict) -> List[str]:
        """Generate actionable recommendations based on learning insights"""
        
        recommendations = []
        
        try:
            # Overall performance recommendations
            overall_perf = insights.get('overall_performance', {})
            win_rate = overall_perf.get('win_rate', 0)
            
            if win_rate > 0.7:
                recommendations.append("[TARGET] Excellent win rate! Consider increasing position sizes")
            elif win_rate < 0.4:
                recommendations.append("[WARNING] Low win rate detected. Review entry criteria and risk management")
            
            # Strategy-specific recommendations
            strategy_perf = insights.get('strategy_performance', {})
            if strategy_perf:
                best_strategies = sorted(
                    strategy_perf.items(),
                    key=lambda x: x[1]['avg_pnl'],
                    reverse=True
                )[:2]
                
                if best_strategies:
                    recommendations.append(
                        f"[STAR] Best performing strategies: {', '.join([s[0] for s in best_strategies])}"
                    )
                
                # Find underperforming strategies
                poor_strategies = [
                    strategy for strategy, perf in strategy_perf.items()
                    if perf['avg_pnl'] < -20 and perf['total_trades'] >= 3
                ]
                
                if poor_strategies:
                    recommendations.append(
                        f"[ERROR] Avoid these strategies: {', '.join(poor_strategies)}"
                    )
            
            # Pattern-based recommendations
            pattern_perf = insights.get('pattern_success_rates', {})
            if pattern_perf:
                best_patterns = [
                    pattern for pattern, perf in pattern_perf.items()
                    if perf['success_rate'] > 0.7 and perf['total'] >= 3
                ]
                
                if best_patterns:
                    recommendations.append(
                        f"🎨 Focus on these patterns: {', '.join(best_patterns[:3])}"
                    )
            
            # Learning confidence recommendations
            confidence = insights.get('learning_confidence', 0)
            if confidence < 0.3:
                recommendations.append("📚 Insufficient data for reliable AI recommendations - continue gathering data")
            elif confidence > 0.8:
                recommendations.append("🧠 High learning confidence - AI recommendations are reliable")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _calculate_learning_confidence(self, insights: Dict) -> float:
        """Calculate overall learning system confidence"""
        
        try:
            total_trades = insights.get('total_trades_analyzed', 0)
            
            # Base confidence on number of trades analyzed
            base_confidence = min(total_trades / 50.0, 1.0)  # Max confidence at 50+ trades
            
            # Adjust for strategy diversity
            strategy_count = len(insights.get('strategy_performance', {}))
            strategy_bonus = min(strategy_count / 5.0, 0.2)  # Max 20% bonus for 5+ strategies
            
            # Adjust for pattern recognition
            pattern_count = len(insights.get('pattern_success_rates', {}))
            pattern_bonus = min(pattern_count / 10.0, 0.2)  # Max 20% bonus for 10+ patterns
            
            total_confidence = base_confidence + strategy_bonus + pattern_bonus
            
            return min(total_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating learning confidence: {e}")
            return 0.0
    
    # Additional helper methods for completeness
    
    def _calculate_profit_factor(self, cursor) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        try:
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN final_pnl > 0 THEN final_pnl ELSE 0 END) as gross_profit,
                    SUM(CASE WHEN final_pnl < 0 THEN ABS(final_pnl) ELSE 0 END) as gross_loss
                FROM trade_lifecycles
                WHERE final_pnl IS NOT NULL
            """)
            
            result = cursor.fetchone()
            if result and result['gross_loss'] > 0:
                return result['gross_profit'] / result['gross_loss']
        except:
            pass
        return 0.0
    
    def _get_regime_performance(self, cursor) -> Dict:
        """Get performance by market regime"""
        try:
            cursor.execute("""
                SELECT 
                    mr.regime_type,
                    COUNT(tl.id) as trades,
                    AVG(tl.final_pnl) as avg_pnl,
                    SUM(CASE WHEN tl.final_pnl > 0 THEN 1 ELSE 0 END) as wins
                FROM market_regimes mr
                JOIN trade_lifecycles tl ON DATE(tl.entry_timestamp) = mr.date
                WHERE tl.final_pnl IS NOT NULL
                GROUP BY mr.regime_type
            """)
            
            results = cursor.fetchall()
            return {
                row['regime_type']: {
                    'trades': row['trades'],
                    'avg_pnl': row['avg_pnl'],
                    'win_rate': row['wins'] / row['trades'] if row['trades'] > 0 else 0
                }
                for row in results
            }
        except:
            return {}
    
    def _classify_market_condition(self, market_conditions: Dict) -> str:
        """Classify market condition for strategy performance tracking"""
        volatility = market_conditions.get('volatility', 15)
        
        if volatility > 25:
            return 'HIGH_VOLATILITY'
        elif volatility < 10:
            return 'LOW_VOLATILITY'
        else:
            return 'NORMAL_VOLATILITY'
    
    def _get_time_period(self, timestamp_str: str) -> str:
        """Get time period for strategy performance tracking"""
        try:
            dt = datetime.fromisoformat(timestamp_str)
            hour = dt.hour
            
            if 9 <= hour < 12:
                return 'MORNING'
            elif 12 <= hour < 15:
                return 'AFTERNOON'
            else:
                return 'LATE'
        except:
            return 'UNKNOWN'
    
    def _analyze_exit_patterns(self, similar_trades: List, current_pnl: float, 
                                time_in_position: int, market_conditions: Dict) -> Dict:
        """Analyze exit patterns from similar historical trades"""
        
        try:
            # Analyze trades that were in similar PnL situation
            similar_pnl_trades = [
                trade for trade in similar_trades
                if abs(current_pnl - (trade['max_profit_reached'] or 0)) < abs(current_pnl * 0.3)
            ]
            
            if len(similar_pnl_trades) < 2:
                return {
                    'should_exit': False,
                    'confidence': 0.1,
                    'reason': 'Insufficient similar historical situations'
                }
            
            # Count how many similar situations ended profitably
            profitable_outcomes = sum(1 for trade in similar_pnl_trades if trade['final_pnl'] > current_pnl)
            exit_success_rate = profitable_outcomes / len(similar_pnl_trades)
            
            # Consider time factor
            long_time_trades = [t for t in similar_pnl_trades if t['final_time'] > time_in_position + 30]
            if long_time_trades:
                extended_success_rate = sum(1 for t in long_time_trades if t['final_pnl'] > current_pnl) / len(long_time_trades)
            else:
                extended_success_rate = 0.5
            
            # Decision logic
            if current_pnl > 0:
                # In profit - should we take it or hold?
                if exit_success_rate < 0.3:  # Most similar situations ended worse
                    return {
                        'should_exit': True,
                        'confidence': 0.8,
                        'reason': f'Historical data shows {exit_success_rate:.1%} success rate for holding in similar situations'
                    }
                elif extended_success_rate > 0.7:  # Holding longer usually worked
                    return {
                        'should_exit': False,
                        'confidence': 0.7,
                        'reason': f'Historical data shows {extended_success_rate:.1%} success rate for holding longer'
                    }
            else:
                # In loss - cut losses or wait for recovery?
                recovery_rate = sum(1 for t in similar_pnl_trades if t['final_pnl'] > 0) / len(similar_pnl_trades)
                
                if recovery_rate < 0.2:  # Rarely recovered
                    return {
                        'should_exit': True,
                        'confidence': 0.9,
                        'reason': f'Only {recovery_rate:.1%} of similar losing positions recovered'
                    }
            
            return {
                'should_exit': False,
                'confidence': 0.5,
                'reason': 'Historical data is mixed - continue monitoring'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing exit patterns: {e}")
            return {'should_exit': False, 'error': str(e)}
    
    def _generate_strategy_reasoning(self, strategy: 'StrategyType', 
                                    performance_data: Dict, market_regime: Dict) -> str:
        """Generate human-readable reasoning for strategy recommendation"""
        
        try:
            reasoning = f"Recommended {strategy.value} based on: "
            
            reasons = []
            
            if performance_data['avg_pnl'] > 50:
                reasons.append(f"strong historical performance (avg ₹{performance_data['avg_pnl']:.0f})")
            
            if performance_data['win_rate'] > 0.7:
                reasons.append(f"high win rate ({performance_data['win_rate']:.1%})")
            
            if performance_data['sample_size'] >= 5:
                reasons.append(f"reliable data ({performance_data['sample_size']} trades)")
            
            if market_regime.get('type') == 'HIGH_VOLATILITY' and 'STRADDLE' in strategy.value:
                reasons.append("high volatility favors straddle strategies")
            
            if market_regime.get('sentiment') == 'STRONG_TREND' and any(x in strategy.value for x in ['CALL', 'PUT']):
                reasons.append("strong trend supports directional strategies")
            
            if performance_data.get('pattern_adjustment', 0) > 0:
                reasons.append("favorable pattern recognition")
            
            if reasons:
                reasoning += ", ".join(reasons)
            else:
                reasoning += "balanced risk-reward profile for current conditions"
            
            return reasoning
            
        except Exception as e:
            logger.error(f"Error generating strategy reasoning: {e}")
            return "Recommended based on available data"
               

    # ================== CONFIGURATION MANAGEMENT ==================

class BotConfiguration:
        """Configuration manager for the automation bot"""
        
        def __init__(self):
            self.config = {
                # Capital tiers
                'tier_1_threshold': 30000,   # Under ₹30K
                'tier_2_threshold': 100000,  # ₹30K - ₹1L
                
                # Risk management
                'max_daily_loss': 5000,      # ₹5K max daily loss
                'max_position_size': 0.90,   # 90% of capital max
                'emergency_stop_loss': 0.15, # 15% emergency stop
                
                # Profit targets
                'daily_profit_min': 0.08,    # 8% minimum daily target
                'daily_profit_aggressive': 0.15, # 15% aggressive target
                'profit_checkpoints': [0.03, 0.06, 0.09, 0.12, 0.15],
                
                # Time settings
                'market_open': '09:00',
                'market_close': '15:30',
                'square_off_time': '15:15',
                'no_new_trades_after': '14:00',
                
                # Monitoring intervals
                'monitoring_interval': 30,    # seconds
                'technical_check_interval': 30, # seconds
                'pnl_update_interval': 15,    # seconds
                
                # Strategy parameters
                'min_signal_confidence': 0.60,
                'high_confidence_threshold': 0.75,
                'technical_confirmation_required': True,
                
                # Risk tier adjustments
                'tier_confidence_requirements': {
                    1: 0.60,  # Normal
                    2: 0.70,  # Caution  
                    3: 0.80,  # Danger
                    4: 0.90   # Recovery (very high bar)
                },
                
                'tier_position_multipliers': {
                    1: 1.0,    # Normal position size
                    2: 0.75,   # Reduced position size
                    3: 0.50,   # Half position size
                    4: 0.25    # Quarter position size
                }
            }
        
        def get(self, key: str, default=None):
            """Get configuration value"""
            return self.config.get(key, default)
        
        def update(self, updates: Dict):
            """Update configuration"""
            self.config.update(updates)
            logger.info(f"Configuration updated: {list(updates.keys())}")
        
        def get_tier_from_capital(self, capital: float) -> CapitalTier:
            """Determine capital tier from available capital"""
            if capital < self.config['tier_1_threshold']:
                return CapitalTier.TIER_1
            elif capital < self.config['tier_2_threshold']:
                return CapitalTier.TIER_2
            else:
                return CapitalTier.TIER_3
        
        def is_market_hours(self) -> bool:
            """Check if currently in market hours"""
            from datetime import datetime, time
            
            now = datetime.now().time()
            market_open = datetime.strptime(self.config['market_open'], '%H:%M').time()
            market_close = datetime.strptime(self.config['market_close'], '%H:%M').time()
            
            return market_open <= now <= market_close
        
        def time_until_square_off(self) -> int:
            """Get minutes until square off time"""
            from datetime import datetime, timedelta
            
            now = datetime.now()
            square_off_time = datetime.strptime(self.config['square_off_time'], '%H:%M').time()
            square_off_datetime = datetime.combine(now.date(), square_off_time)
            
            if square_off_datetime > now:
                return int((square_off_datetime - now).total_seconds() / 60)
            return 0

# ================== ENHANCED SIGNAL PROCESSING ==================

class EnhancedSignalProcessor:
        """Enhanced signal processing with filtering and validation"""
        
        def __init__(self, config: BotConfiguration):
            self.config = config
            self.signal_history = []
            self.duplicate_filter_window = 300  # 5 minutes
            
        async def process_signal(self, signal: TradingSignal, 
                            technical_analysis: Dict = None) -> Dict:
            """Process and validate incoming signal"""
            
            processing_result = {
                'signal': signal,
                'approved': False,
                'rejection_reason': None,
                'confidence_adjustment': 0.0,
                'technical_confirmation': False,
                'recommended_action': 'REJECT'
            }
            
            try:
                # 1. Basic validation
                if not self._basic_signal_validation(signal):
                    processing_result['rejection_reason'] = 'Basic validation failed'
                    return processing_result
                
                # 2. Duplicate detection
                if self._is_duplicate_signal(signal):
                    processing_result['rejection_reason'] = 'Duplicate signal detected'
                    return processing_result
                
                # 3. Market timing check
                if not self._check_market_timing(signal):
                    processing_result['rejection_reason'] = 'Outside valid trading hours'
                    return processing_result
                
                # 4. Technical confirmation
                if technical_analysis:
                    tech_confirmation = self._check_technical_confirmation(
                        signal, technical_analysis
                    )
                    processing_result['technical_confirmation'] = tech_confirmation['confirmed']
                    processing_result['confidence_adjustment'] += tech_confirmation['confidence_delta']
                
                # 5. Confidence adjustment
                final_confidence = signal.confidence + processing_result['confidence_adjustment']
                
                # 6. Final approval decision
                min_confidence = self.config.get('min_signal_confidence', 0.60)
                if final_confidence >= min_confidence:
                    processing_result['approved'] = True
                    processing_result['recommended_action'] = 'EXECUTE'
                else:
                    processing_result['rejection_reason'] = f'Confidence {final_confidence:.1%} below threshold {min_confidence:.1%}'
                
                # Store signal for duplicate detection
                self.signal_history.append({
                    'signal': signal,
                    'timestamp': datetime.now(),
                    'processed': True
                })
                
                # Clean old signals
                self._cleanup_signal_history()
                
                return processing_result
                
            except Exception as e:
                logger.error(f"Error processing signal: {e}")
                processing_result['rejection_reason'] = f'Processing error: {str(e)}'
                return processing_result
        
        def _basic_signal_validation(self, signal: TradingSignal) -> bool:
            """Basic signal validation"""
            
            # Check required fields
            if not signal.ticker or not signal.direction or not signal.strategy:
                return False
            
            # Check confidence range
            if not (0.0 <= signal.confidence <= 1.0):
                return False
            
            # Check direction validity
            if signal.direction not in ['bullish', 'bearish', 'neutral']:
                return False
            
            # Check if ticker is supported
            supported_tickers = [
                'NIFTY', 'NIFTY 50', 'BANKNIFTY', 'NIFTY BANK', 'FINNIFTY',
                'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK',
                'SBIN', 'KOTAKBANK', 'BAJFINANCE', 'BHARTIARTL', 'ASIANPAINT',
                'ITC', 'LT', 'MARUTI'
            ]
            
            if signal.ticker.upper() not in supported_tickers:
                logger.warning(f"Unsupported ticker: {signal.ticker}")
                return False
            
            return True
        
        def _is_duplicate_signal(self, signal: TradingSignal) -> bool:
            """Check for duplicate signals"""
            
            current_time = datetime.now()
            
            for historical_signal in self.signal_history:
                hist_signal = historical_signal['signal']
                hist_time = historical_signal['timestamp']
                
                # Check if within duplicate window
                if (current_time - hist_time).total_seconds() > self.duplicate_filter_window:
                    continue
                
                # Check for similarity
                if (hist_signal.ticker == signal.ticker and
                    hist_signal.direction == signal.direction and
                    abs(hist_signal.confidence - signal.confidence) < 0.1):
                    return True
            
            return False
        
        def _check_market_timing(self, signal: TradingSignal) -> bool:
            """Check if signal timing is valid"""
            
            # Skip timing check for test signals
            if signal.source == 'test_injection':
                return True
            
            # Check market hours
            if not self.config.is_market_hours():
                return False
            
            # Check if too late for new positions
            time_until_square_off = self.config.time_until_square_off()
            if time_until_square_off < 75:  # Less than 1.25 hours left
                return False
            
            return True
        
        def _check_technical_confirmation(self, signal: TradingSignal, 
                                        technical_analysis: Dict) -> Dict:
            """Check technical analysis confirmation"""
            
            confirmation_result = {
                'confirmed': False,
                'confidence_delta': 0.0,
                'reasons': []
            }
            
            market_bias = technical_analysis.get('market_bias', 'NEUTRAL')
            entry_signal = technical_analysis.get('entry_signal', {})
            confidence_score = technical_analysis.get('confidence_score', 0.5)
            
            # Direction alignment check
            signal_bullish = signal.direction.lower() == 'bullish'
            signal_bearish = signal.direction.lower() == 'bearish'
            
            bias_bullish = 'BULLISH' in market_bias
            bias_bearish = 'BEARISH' in market_bias
            
            if (signal_bullish and bias_bullish) or (signal_bearish and bias_bearish):
                confirmation_result['confirmed'] = True
                confirmation_result['confidence_delta'] += 0.1
                confirmation_result['reasons'].append('Direction alignment')
            
            # Entry signal confirmation
            signal_type = entry_signal.get('signal_type', 'HOLD')
            if ((signal_bullish and signal_type == 'BUY') or 
                (signal_bearish and signal_type == 'SELL')):
                confirmation_result['confidence_delta'] += 0.05
                confirmation_result['reasons'].append('Entry signal confirmation')
            
            # Technical confidence boost
            if confidence_score > 0.7:
                confirmation_result['confidence_delta'] += 0.05
                confirmation_result['reasons'].append('High technical confidence')
            
            # Final confirmation decision
            if confirmation_result['confidence_delta'] >= 0.1:
                confirmation_result['confirmed'] = True
            
            return confirmation_result
        
        def _cleanup_signal_history(self):
            """Clean up old signals from history"""
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(seconds=self.duplicate_filter_window * 2)
            
            self.signal_history = [
                s for s in self.signal_history 
                if s['timestamp'] > cutoff_time
            ]

    # ================== POSITION MONITORING SYSTEM ==================

class PositionMonitor:
        """Advanced position monitoring with real-time updates"""
        
        def __init__(self, zerodha_client, config: BotConfiguration):
            self.zerodha = zerodha_client
            self.config = config
            self.monitoring_active = False
            self.position_cache = {}
            self.last_update = None
        
        async def start_monitoring(self, position: ActivePosition):
            """Start monitoring a position"""
            
            logger.info(f"🔍 Starting position monitoring for {position.signal.ticker}")
            
            self.monitoring_active = True
            position_id = f"{position.signal.ticker}_{position.entry_time.isoformat()}"
            
            # Store position
            self.position_cache[position_id] = {
                'position': position,
                'last_pnl': 0.0,
                'max_profit': 0.0,
                'max_loss': 0.0,
                'pnl_history': [],
                'alerts_sent': []
            }
            
            # Start monitoring loop
            while self.monitoring_active and position_id in self.position_cache:
                try:
                    await self._update_position_data(position_id)
                    await self._check_exit_conditions(position_id)
                    await asyncio.sleep(self.config.get('pnl_update_interval', 15))
                    
                except Exception as e:
                    logger.error(f"Error in position monitoring: {e}")
                    await asyncio.sleep(30)  # Wait longer on error
        
        async def _update_position_data(self, position_id: str):
            """Update position P&L and metrics"""
            
            if position_id not in self.position_cache:
                return
            
            cache_entry = self.position_cache[position_id]
            position = cache_entry['position']
            
            try:
                # Get current positions from Zerodha
                positions = self.zerodha.get_positions()
                
                # Find our position
                our_position = None
                for pos in positions.get('net', []) + positions.get('day', []):
                    if pos['tradingsymbol'] in [leg.tradingsymbol for leg in position.option_legs]:
                        our_position = pos
                        break
                
                if our_position:
                    # Update P&L
                    current_pnl = float(our_position.get('pnl', 0))
                    position.current_pnl = current_pnl
                    
                    # Update tracking metrics
                    cache_entry['last_pnl'] = current_pnl
                    cache_entry['max_profit'] = max(cache_entry['max_profit'], current_pnl)
                    cache_entry['max_loss'] = min(cache_entry['max_loss'], current_pnl)
                    
                    # Store P&L history
                    cache_entry['pnl_history'].append({
                        'timestamp': datetime.now(),
                        'pnl': current_pnl,
                        'price': float(our_position.get('last_price', 0))
                    })
                    
                    # Limit history size
                    if len(cache_entry['pnl_history']) > 100:
                        cache_entry['pnl_history'] = cache_entry['pnl_history'][-50:]
                    
                    # Update position object
                    position.max_profit = cache_entry['max_profit']
                    position.max_loss = cache_entry['max_loss']
                    
                    self.last_update = datetime.now()
                    
                    logger.debug(f"Position update: {position.signal.ticker} P&L: ₹{current_pnl:.0f}")
                    
                else:
                    logger.warning(f"Position not found in Zerodha positions: {position.signal.ticker}")
                    
            except Exception as e:
                logger.error(f"Error updating position data: {e}")
        
        async def _check_exit_conditions(self, position_id: str):
            """Check if position should be exited"""
            
            if position_id not in self.position_cache:
                return
            
            cache_entry = self.position_cache[position_id]
            position = cache_entry['position']
            current_pnl = cache_entry['last_pnl']
            
            # Calculate profit percentage
            investment = position.entry_premium * sum(leg.contracts * leg.lot_size for leg in position.option_legs)
            profit_percent = (current_pnl / investment * 100) if investment > 0 else 0
            
            # Check various exit conditions
            exit_signals = []
            
            # 1. Profit targets
            if profit_percent >= 15:
                exit_signals.append('PROFIT_TARGET_15')
            elif profit_percent >= 10:
                exit_signals.append('PROFIT_TARGET_10')
            
            # 2. Stop losses
            if profit_percent <= -15:
                exit_signals.append('STOP_LOSS_15')
            elif profit_percent <= -10:
                exit_signals.append('STOP_LOSS_10')
            
            # 3. Time-based exits
            time_until_square_off = self.config.time_until_square_off()
            if time_until_square_off <= 15:
                exit_signals.append('TIME_STOP_IMMINENT')
            elif time_until_square_off <= 30:
                exit_signals.append('TIME_STOP_WARNING')
            
            # 4. Trailing stop
            if cache_entry['max_profit'] > 100:  # Only if we've seen decent profit
                trailing_threshold = cache_entry['max_profit'] * 0.5  # 50% of max profit
                if current_pnl <= trailing_threshold:
                    exit_signals.append('TRAILING_STOP')
            
            # Send alerts for new exit signals
            for signal in exit_signals:
                if signal not in cache_entry['alerts_sent']:
                    await self._send_exit_alert(position, signal, profit_percent)
                    cache_entry['alerts_sent'].append(signal)
            
            # Auto-exit on critical conditions
            critical_exits = ['STOP_LOSS_15', 'TIME_STOP_IMMINENT', 'TRAILING_STOP']
            if any(signal in exit_signals for signal in critical_exits):
                logger.info(f"🚨 Critical exit condition for {position.signal.ticker}: {exit_signals}")
                # Here you would trigger the actual exit
                # For now, just log it
        
        async def _send_exit_alert(self, position: ActivePosition, exit_signal: str, profit_percent: float):
            """Send exit alert"""
            
            alert_messages = {
                'PROFIT_TARGET_15': f"🎉 15% profit target hit!",
                'PROFIT_TARGET_10': f"[MONEY] 10% profit checkpoint",
                'STOP_LOSS_15': f"🛑 15% stop loss triggered",
                'STOP_LOSS_10': f"[WARNING] 10% loss checkpoint", 
                'TIME_STOP_IMMINENT': f"⏰ Square-off time approaching",
                'TIME_STOP_WARNING': f"⏰ 30 minutes to square-off",
                'TRAILING_STOP': f"[DOWN] Trailing stop triggered"
            }
            
            message = alert_messages.get(exit_signal, f"Exit signal: {exit_signal}")
            
            logger.info(f"📢 Alert: {position.signal.ticker} - {message} (P&L: {profit_percent:+.1f}%)")
            
            # Here you would send to Telegram if available
            # await self.telegram_bot.send_message(f"{message}\n{position.signal.ticker}: {profit_percent:+.1f}%")
        
        def stop_monitoring(self, position_id: str = None):
            """Stop monitoring position(s)"""
            
            if position_id:
                if position_id in self.position_cache:
                    del self.position_cache[position_id]
                    logger.info(f"Stopped monitoring position: {position_id}")
            else:
                self.monitoring_active = False
                self.position_cache.clear()
                logger.info("Stopped all position monitoring")
        
        def get_monitoring_status(self) -> Dict:
            """Get current monitoring status"""
            
            return {
                'monitoring_active': self.monitoring_active,
                'positions_tracked': len(self.position_cache),
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'positions': {
                    pid: {
                        'ticker': cache['position'].signal.ticker,
                        'current_pnl': cache['last_pnl'],
                        'max_profit': cache['max_profit'],
                        'max_loss': cache['max_loss'],
                        'alerts_sent': len(cache['alerts_sent'])
                    }
                    for pid, cache in self.position_cache.items()
                }
            }

# ================== PERFORMANCE ANALYTICS ==================

class PerformanceAnalytics:
        """Advanced performance analytics and reporting"""
        
        def __init__(self):
            self.daily_metrics = {}
            self.trade_history = []
            self.pattern_performance = {}
            
        def record_trade(self, trade_data: Dict):
            """Record completed trade"""
            
            trade_data['timestamp'] = datetime.now().isoformat()
            self.trade_history.append(trade_data)
            
            # Update daily metrics
            today = datetime.now().strftime('%Y-%m-%d')
            if today not in self.daily_metrics:
                self.daily_metrics[today] = {
                    'trades': 0,
                    'profitable': 0,
                    'total_pnl': 0.0,
                    'total_investment': 0.0,
                    'max_profit': 0.0,
                    'max_loss': 0.0,
                    'avg_hold_time': 0.0
                }
            
            metrics = self.daily_metrics[today]
            metrics['trades'] += 1
            
            pnl = trade_data.get('pnl', 0)
            if pnl > 0:
                metrics['profitable'] += 1
            
            metrics['total_pnl'] += pnl
            metrics['total_investment'] += trade_data.get('investment', 0)
            metrics['max_profit'] = max(metrics['max_profit'], pnl)
            metrics['max_loss'] = min(metrics['max_loss'], pnl)
            
            # Update pattern performance
            strategy = trade_data.get('strategy', 'UNKNOWN')
            if strategy not in self.pattern_performance:
                self.pattern_performance[strategy] = {
                    'total': 0,
                    'profitable': 0,
                    'total_pnl': 0.0,
                    'avg_pnl': 0.0,
                    'win_rate': 0.0
                }
            
            pattern = self.pattern_performance[strategy]
            pattern['total'] += 1
            if pnl > 0:
                pattern['profitable'] += 1
            pattern['total_pnl'] += pnl
            pattern['avg_pnl'] = pattern['total_pnl'] / pattern['total']
            pattern['win_rate'] = pattern['profitable'] / pattern['total']
        
        def get_daily_summary(self, date: str = None) -> Dict:
            """Get daily performance summary"""
            
            if not date:
                date = datetime.now().strftime('%Y-%m-%d')
            
            metrics = self.daily_metrics.get(date, {})
            
            if not metrics:
                return {
                    'date': date,
                    'no_data': True
                }
            
            return {
                'date': date,
                'trades_executed': metrics['trades'],
                'profitable_trades': metrics['profitable'],
                'loss_trades': metrics['trades'] - metrics['profitable'],
                'win_rate': metrics['profitable'] / metrics['trades'] if metrics['trades'] > 0 else 0,
                'total_pnl': metrics['total_pnl'],
                'total_investment': metrics['total_investment'],
                'roi_percent': (metrics['total_pnl'] / metrics['total_investment'] * 100) if metrics['total_investment'] > 0 else 0,
                'best_trade': metrics['max_profit'],
                'worst_trade': metrics['max_loss'],
                'avg_pnl_per_trade': metrics['total_pnl'] / metrics['trades'] if metrics['trades'] > 0 else 0
            }
        
        def get_strategy_performance(self) -> Dict:
            """Get performance by strategy"""
            
            # Sort strategies by win rate
            sorted_strategies = sorted(
                self.pattern_performance.items(),
                key=lambda x: x[1]['win_rate'],
                reverse=True
            )
            
            return {
                'strategy_rankings': sorted_strategies,
                'best_strategy': sorted_strategies[0] if sorted_strategies else None,
                'total_strategies_used': len(self.pattern_performance)
            }
        
        def get_weekly_performance(self) -> Dict:
            """Get weekly performance analysis"""
            
            # Get last 7 days
            weekly_data = []
            total_pnl = 0
            total_trades = 0
            
            for i in range(7):
                date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                daily_data = self.get_daily_summary(date)
                weekly_data.append(daily_data)
                
                if not daily_data.get('no_data'):
                    total_pnl += daily_data['total_pnl']
                    total_trades += daily_data['trades_executed']
            
            return {
                'weekly_pnl': total_pnl,
                'weekly_trades': total_trades,
                'daily_breakdown': weekly_data,
                'avg_daily_pnl': total_pnl / 7,
                'avg_daily_trades': total_trades / 7
            }

    # ================== MAIN BOT EXTENSION METHODS ==================

    # Add these methods to your main AutomatedIntradayOptionsBot class:
    
class AutomatedIntradayOptionsBot:
    """
    Main automated intraday options trading bot with v2.0 enhancements
    Integrates all components for intelligent automated trading
    """
    
    def __init__(self, zerodha_client: ZerodhaAPIClient):
        """Initialize the automated trading bot - FIXED VERSION"""
        
        # Core dependencies
        self.zerodha = zerodha_client
        
        # [OK] FIXED: Initialize core components with correct parameters
        self.market_data_provider = ZerodhaMarketDataProvider(zerodha_client)
        self.options_chain_provider = ZerodhaOptionsChainProvider(zerodha_client)
        
        # [OK] CRITICAL FIX: Pass zerodha_client directly to options analyzer
        # The analyzer expects a ZerodhaAPIClient, not a ZerodhaMarketDataProvider
        self.options_analyzer = ZerodhaEnhancedOptionsAnalyzer(zerodha_client)
        
        self.order_manager = ZerodhaOrderManager(zerodha_client)
        self.risk_manager = ZerodhaRiskManager(zerodha_client)
        self.technical_analyzer = ZerodhaTechnicalAnalyzer(zerodha_client)
        self.trade_logger = IndianTradeLogger()
        
        # Initialize Telegram bot (optional)
        try:
            # Get credentials from environment variables
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if bot_token and chat_id:
                self.telegram_bot = TelegramSignalBot(bot_token, chat_id)
                logger.info("[OK] Telegram bot initialized successfully")
            else:
                logger.warning("[WARNING] Telegram credentials not found - running without notifications")
                self.telegram_bot = None
        except Exception as e:
            logger.warning(f"Telegram bot initialization failed: {e}")
            self.telegram_bot = None
        
        # Bot state management
        self.state = TradingState.IDLE
        self.monitoring_active = False
        self.monitoring_task = None
        self.signal_queue = asyncio.Queue()
        self.session_start_time = datetime.now()

        # CRITICAL: Add thread-safe locks for position management
        self.position_lock = asyncio.Lock()
        self.signal_processing_lock = asyncio.Lock()
        self.order_execution_lock = asyncio.Lock()

        # Emergency stop tracking
        self.emergency_stop_triggered = False
        self.consecutive_losses = 0
        self.recovery_mode = False

        # Position tracking
        self.active_position: Optional[ActivePosition] = None
        self.position_history: List[ActivePosition] = []
        
        # Performance tracking
        self.daily_performance = DailyPerformance(
            date=datetime.now().strftime('%Y-%m-%d'),
            starting_capital=5000.0  # Will be updated from actual account
        )
        
        # Signal tracking
        self.signals_received_today = 0
        self.signals_processed_today = 0
        self.last_signal_time = None
        
        # Risk management
        self.emergency_stop_triggered = False
        self.recovery_mode = False
        self.consecutive_losses = 0
        
        # Threading for signal interception
        self.signal_thread = None
        self.signal_interceptor = None
        
        logger.info("[OK] AutomatedIntradayOptionsBot initialized with fixed integration")

    def initialize_v2_components(self):
        """Initialize v2.0 specific components - FIXED VERSION"""
        
        # Configuration
        self.bot_config = BotConfiguration()
        
        # Enhanced signal processing
        self.signal_processor = EnhancedSignalProcessor(self.bot_config)
        
        # Position monitoring
        self.position_monitor = PositionMonitor(self.zerodha, self.bot_config)
        
        # Performance analytics
        self.performance_analytics = PerformanceAnalytics()
        
        # Dynamic capital manager
        self.capital_manager = DynamicCapitalManager(self.zerodha)
        
        # Intelligent risk manager
        self.intelligent_risk_manager = IntelligentRiskManager(self.technical_analyzer)
        
        # Profit optimizer
        self.profit_optimizer = DynamicProfitOptimizer()
        
        # Multi-leg executor
        self.strategy_executor = MultiLegStrategyExecutor(
            self.options_chain_provider, self.order_manager
        )
        
        # Learning system
        self.learning_system = LearningSystem(self.trade_logger)
        
        logger.info("[OK] v2.0 components initialized successfully")

    async def enhanced_signal_processing(self, signal: TradingSignal) -> Dict:
        """Enhanced signal processing with v2.0 features - FIXED VERSION"""
        
        # 1. Process signal through enhanced processor
        processing_result = await self.signal_processor.process_signal(signal)
        
        if not processing_result['approved']:
            logger.info(f"[ERROR] Signal rejected: {processing_result['rejection_reason']}")
            return {'signal_processed': False, 'processing_result': processing_result}
        
        # 2. Get technical analysis with error handling
        try:
            # [OK] FIXED: Use the correct method from your technical analyzer
            technical_analysis = await self.technical_analyzer.analyze_symbol_for_options(
                signal.ticker, 
                signal.current_price, 
                signal.market_data, 
                'intraday'  # or signal.trading_style if available
            )
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            # Create fallback technical analysis
            technical_analysis = {
                'market_bias': 'NEUTRAL',
                'confidence_score': 0.5,
                'entry_signal': {
                    'signal_type': 'HOLD',
                    'strength': 0.3,
                    'reason': 'Technical analysis unavailable'
                }
            }
        
        # 3. Update capital status
        capital_status = await self.capital_manager.update_capital_status()
        
        # 4. Determine strategy based on capital tier
        allowed_strategies = self.capital_manager.get_allowed_strategies()
        
        # 5. Get strategy recommendation from analyzer
        try:
            # [OK] FIXED: Use the options analyzer's analyze_best_options method
            options_analysis = await self.options_analyzer.analyze_best_options(
                signal.ticker, 
                signal.market_data, 
                technical_analysis
            )
            
            recommended_strategy = options_analysis.get('recommended_strategy', 'LONG_CALL')
            
        except Exception as e:
            logger.error(f"Options analysis failed: {e}")
            # Fallback to simple strategy
            recommended_strategy = 'LONG_CALL' if signal.direction == 'bullish' else 'LONG_PUT'
        
        # 6. Create strategy legs using the multi-leg executor
        try:
            # Convert string strategy to StrategyType enum
            from automated_options_bot import StrategyType  # Import from your main file
            
            strategy_mapping = {
                'LONG_CALL': StrategyType.BUY_CALL,
                'LONG_PUT': StrategyType.BUY_PUT,
                'BULL_CALL_SPREAD': StrategyType.BULL_CALL_SPREAD,
                'BEAR_PUT_SPREAD': StrategyType.BEAR_PUT_SPREAD,
                'LONG_STRADDLE': StrategyType.LONG_STRADDLE,
                'LONG_STRANGLE': StrategyType.LONG_STRANGLE
            }
            
            strategy_type = strategy_mapping.get(recommended_strategy, StrategyType.BUY_CALL)
            
            strategy_legs = await self.strategy_executor.create_strategy_legs(
                signal, 
                strategy_type,
                capital_status['available_capital'],
                technical_analysis,
                capital_status.get('tier', CapitalTier.TIER_1)
            )
            
            logger.info(f"[TARGET] Strategy legs created: {len(strategy_legs)} legs")
            for i, leg in enumerate(strategy_legs):
                logger.info(f"  Leg {i+1}: {leg.action} {leg.contracts}x {leg.tradingsymbol}")
            
        except Exception as e:
            logger.error(f"Strategy leg creation failed: {e}")
            strategy_legs = []
        
        # 7. Risk check - FIXED: Handle None position properly
        if self.active_position:
            risk_analysis = await self.intelligent_risk_manager.analyze_position_health(
                self.active_position, signal.current_price, signal.market_data
            )
        else:
            # No active position, so no risk analysis needed for new signal
            risk_analysis = {
                'action': 'CONTINUE',
                'reason': 'No active position - proceeding with new signal',
                'confidence': 0.8,
                'analysis_depth': 'none'
            }
        
        # 8. Prepare execution results (dry run for now)
        execution_results = []
        if risk_analysis.get('action') != 'EXIT_IMMEDIATELY':
            for leg in strategy_legs:
                try:
                    # Dry run execution
                    result = await self.order_manager.place_options_order(leg, dry_run=True)
                    execution_results.append(result)
                    logger.info(f"📋 Simulated order: {leg.action} {leg.contracts}x {leg.tradingsymbol} - {result.get('status', 'unknown')}")
                except Exception as e:
                    logger.error(f"Order simulation failed: {e}")
                    execution_results.append({'status': 'error', 'message': str(e)})
        else:
            logger.warning(f"[WARNING] Risk analysis suggests EXIT_IMMEDIATELY - skipping execution")
        
        return {
            'signal_processed': True,
            'processing_result': processing_result,
            'capital_status': capital_status,
            'strategy_legs': len(strategy_legs),
            'actual_strategy_legs': strategy_legs,  # [OK] Added actual legs for execution
            'risk_analysis': risk_analysis,
            'execution_results': execution_results,
            'technical_analysis_available': technical_analysis is not None,
            'technical_analysis': technical_analysis,  # [OK] Added for execution method
            'recommended_strategy': recommended_strategy,
            'strategy_type': strategy_type if 'strategy_type' in locals() else StrategyType.BUY_CALL  # [OK] Added strategy type
        }
    
    async def start_automated_trading_v2(self, test_mode: bool = False):
        """Start enhanced automated trading with v2.0 features"""
        
        logger.info("[ROCKET] Starting Enhanced Automated Trading v2.0")
        
        try:
            # 1. Initialize v2.0 components
            self.initialize_v2_components()
            
            # 2. Update starting capital from account
            await self._update_starting_capital()
            
            # 3. Validate market hours (skip in test mode)
            if not test_mode and not self.bot_config.is_market_hours():
                logger.error("[ERROR] Market is closed - cannot start trading")
                return False
            
            # 4. Validate API connection
            if not await self._validate_api_connection():
                logger.error("[ERROR] API connection validation failed")
                return False
            
            # 5. Send startup notification
            await self._send_startup_notification(test_mode)
            
            # 6. Start signal interception
            self._start_signal_interception()
            
            # 7. Start main monitoring loop
            self.state = TradingState.ANALYZING
            self.monitoring_active = True
            
            self.monitoring_task = asyncio.create_task(
                self._main_monitoring_loop(test_mode)
            )
            
            logger.info("[OK] Enhanced automated trading started successfully")
            
            # Wait for monitoring task
            await self.monitoring_task
            
        except asyncio.CancelledError:
            logger.info("🛑 Trading cancelled by user")
        except Exception as e:
            logger.error(f"[ERROR] Error in automated trading: {e}")
            await self.emergency_shutdown_v2()
        finally:
            await self._cleanup_resources()
    
    async def _main_monitoring_loop(self, test_mode: bool = False):
        """Main monitoring and trading loop"""
        
        logger.info("🔄 Starting main monitoring loop")
        
        while self.monitoring_active:
            try:
                # 1. Check market hours (skip in test mode)
                if not test_mode and not self.bot_config.is_market_hours():
                    logger.info("📅 Market closed - stopping monitoring")
                    break
                
                # 2. Check emergency conditions
                if await self._check_emergency_conditions():
                    logger.critical("🚨 Emergency conditions detected")
                    await self.emergency_shutdown_v2()
                    break
                
                # 3. Update daily performance
                await self._update_daily_performance()
                
                # 4. Process pending signals
                await self._process_signal_queue()
                
                # 5. Monitor active position
                if self.active_position:
                    await self._monitor_active_position()
                
                # 6. Check for square-off time
                if not test_mode:
                    time_until_square_off = self.bot_config.time_until_square_off()
                    if time_until_square_off <= 15:  # 15 minutes to close
                        logger.info("⏰ Square-off time approaching")
                        if self.active_position:
                            await self.execute_enhanced_exit(
                                self.active_position, ExitReason.TIME_STOP
                            )
                        break
                
                # 7. Learning and optimization
                if datetime.now().minute % 5 == 0:  # Every 5 minutes
                    await self._periodic_learning_update()
                
                # 8. Sleep before next iteration
                await asyncio.sleep(self.bot_config.get('monitoring_interval', 30))
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _process_signal_queue(self):
        """Process signals from the queue"""
        
        try:
            # Process up to 3 signals per iteration
            for _ in range(3):
                if self.signal_queue.empty():
                    break
                
                signal = await asyncio.wait_for(
                    self.signal_queue.get(), timeout=1.0
                )
                
                await self._handle_incoming_signal(signal)
                
        except asyncio.TimeoutError:
            pass  # No signals in queue
        except Exception as e:
            logger.error(f"Error processing signal queue: {e}")
    
    async def _handle_incoming_signal(self, signal: TradingSignal):
        """Handle incoming trading signal"""
        
        self.signals_received_today += 1
        self.last_signal_time = datetime.now()
        
        logger.info(f"[SIGNAL] Received signal: {signal.ticker} {signal.direction} ({signal.confidence:.1%})")
        
        try:
            # 1. Check if we can process signals
            if not self._can_process_signals():
                logger.info(f"⏸️ Signal blocked: {self._get_block_reason()}")
                return
            
            logger.info("🔄 Step 1: Signal processing checks passed")
            
            # 2. Process signal through enhanced processor
            logger.info("🔄 Step 2: Starting enhanced signal processing...")
            processing_result = await self.enhanced_signal_processing(signal)
            logger.info(f"[OK] Step 2 completed: signal_processed = {processing_result.get('signal_processed', False)}")
            
            if not processing_result.get('signal_processed'):
                logger.info(f"[ERROR] Signal processing failed: {processing_result}")
                return
            
            logger.info("🔄 Step 3: Checking if signal approved...")
            approved = processing_result.get('processing_result', {}).get('approved', False)
            logger.info(f"[OK] Step 3: Signal approved = {approved}")
            
            # 3. Execute if approved
            if approved:
                logger.info("🔄 Step 4: Starting signal execution...")
                await self._execute_signal(signal, processing_result)
                logger.info("[OK] Step 4: Signal execution completed")
                self.signals_processed_today += 1
            else:
                rejection_reason = processing_result.get('processing_result', {}).get('rejection_reason', 'Unknown')
                logger.info(f"🚫 Signal rejected: {rejection_reason}")
        
        except Exception as e:
            logger.error(f"Error handling signal: {e}", exc_info=True)  # [OK] This will show full traceback
            
            # Additional debug info
            logger.error(f"🔍 Debug info:")
            logger.error(f"   Signal: {signal.ticker} {signal.direction}")
            logger.error(f"   Active position exists: {self.active_position is not None}")
            logger.error(f"   Bot state: {self.state}")
            logger.error(f"   Monitoring active: {self.monitoring_active}")
    
    def _can_process_signals(self) -> bool:
        """Check if we can process new signals"""
        
        # Block if position is active (single position focus)
        if self.active_position:
            return False
        
        # Block if in emergency or recovery mode
        if self.emergency_stop_triggered or self.recovery_mode:
            return False
        
        # Block if daily loss limit exceeded
        if self.daily_performance.total_pnl < -self.bot_config.get('max_daily_loss', 5000):
            return False
        
        # Block if too close to market close
        if self.bot_config.time_until_square_off() < 75:  # Less than 1.25 hours
            return False
        
        return True
    
    def _get_block_reason(self) -> str:
        """Get reason why signals are blocked"""
        
        if self.active_position:
            return "Active position exists"
        if self.emergency_stop_triggered:
            return "Emergency stop active"
        if self.recovery_mode:
            return "Recovery mode active"
        if self.daily_performance.total_pnl < -self.bot_config.get('max_daily_loss', 5000):
            return "Daily loss limit exceeded"
        if self.bot_config.time_until_square_off() < 75:
            return "Too close to market close"
        
        return "Unknown reason"

    async def _verify_order_fill(self, order_id: str, timeout: int = 10) -> Dict:
        """CRITICAL: Verify that order is actually filled"""

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                order_status = self.zerodha.get_order_history(order_id)

                if order_status:
                    status = order_status[-1].get('status', 'UNKNOWN')

                    if status == 'COMPLETE':
                        return {
                            'filled': True,
                            'status': status,
                            'filled_quantity': order_status[-1].get('filled_quantity', 0),
                            'average_price': order_status[-1].get('average_price', 0)
                        }
                    elif status in ['REJECTED', 'CANCELLED']:
                        return {
                            'filled': False,
                            'status': status,
                            'reason': order_status[-1].get('status_message', 'Unknown')
                        }

                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error verifying order {order_id}: {e}")

        # Timeout - order status unclear
        return {
            'filled': False,
            'status': 'TIMEOUT',
            'reason': f'Could not confirm fill within {timeout} seconds'
        }

    async def _emergency_exit_legs(self, filled_legs: List[OptionsLeg]):
        """CRITICAL: Emergency exit for partially filled positions"""

        logger.critical("🚨 EMERGENCY EXIT: Squaring off filled legs")

        for leg in filled_legs:
            try:
                # Square off immediately at market price
                quantity = leg.contracts * leg.lot_size
                transaction_type = 'SELL' if leg.action == 'BUY' else 'BUY'

                exit_order = self.zerodha.place_order(
                    tradingsymbol=leg.tradingsymbol,
                    exchange='NFO',
                    transaction_type=transaction_type,
                    quantity=quantity,
                    product='MIS',
                    order_type='MARKET',
                    validity='DAY'
                )

                logger.info(f"[OK] Emergency exit order placed: {exit_order}")

            except Exception as e:
                logger.critical(f"[ERROR] FAILED TO EXIT LEG {leg.tradingsymbol}: {e}")
                # Send urgent alert via Telegram
                if self.telegram_bot:
                    await self.telegram_bot.send_message(
                        f"🚨 CRITICAL: Failed to exit {leg.tradingsymbol} - MANUAL INTERVENTION REQUIRED"
                    )

    def _get_actual_fill_price(self, leg: OptionsLeg, order_result: Dict) -> float:
        """Get actual fill price from order result"""

        try:
            order_id = order_result.get('order_id')
            order_history = self.zerodha.get_order_history(order_id)

            if order_history:
                return float(order_history[-1].get('average_price', leg.theoretical_price))
        except:
            pass

        return leg.theoretical_price  # Fallback to theoretical

    async def _execute_signal(self, signal: TradingSignal, processing_result: Dict):
        """Execute approved signal - FIXED VERSION"""
        
        logger.info(f"[TARGET] Executing signal: {signal.ticker} {signal.direction}")
        
        try:
            # Debug the processing result first
            logger.info(f"[CHART] Processing result keys: {list(processing_result.keys())}")
            logger.info(f"[CHART] Strategy legs count: {processing_result.get('strategy_legs', 0)}")
            
            # 1. Change state
            self.state = TradingState.ENTERING
            
            # 2. Check if we have valid strategy legs
            strategy_legs_count = processing_result.get('strategy_legs', 0)
            if strategy_legs_count == 0:
                logger.error(f"[ERROR] No strategy legs created for {signal.ticker} - cannot execute")
                self.state = TradingState.IDLE
                return
            
            # 3. Get the actual strategy legs (this is missing in your current code!)
            # You need to get this from your strategy executor, not from processing_result
            capital_status = processing_result.get('capital_status', {})
            
            # Create strategy legs properly
            strategy_type = StrategyType.BUY_PUT if signal.direction == 'bearish' else StrategyType.BUY_CALL
            
            try:
                # Get technical analysis 
                tech_analysis = processing_result.get('technical_analysis', {})
                
                # Create actual strategy legs
                strategy_legs = await self.strategy_executor.create_strategy_legs(
                    signal,
                    strategy_type,
                    capital_status.get('available_capital', 9062),
                    tech_analysis,
                    capital_status.get('tier', CapitalTier.TIER_1)
                )
                
                logger.info(f"[TARGET] Created {len(strategy_legs)} strategy legs")
                
                if not strategy_legs:
                    logger.error(f"[ERROR] Strategy leg creation failed for {signal.ticker}")
                    self.state = TradingState.IDLE
                    return
                
            except Exception as e:
                logger.error(f"[ERROR] Error creating strategy legs: {e}")
                self.state = TradingState.IDLE
                return
            
            # 4. Start trade lifecycle
            lifecycle_id = self.learning_system.start_trade_lifecycle(
                signal,
                strategy_type,
                capital_status,
                tech_analysis
            )
            
            # 5. Execute orders for all legs
            execution_results = []
            successful_orders = []
            
            for i, leg in enumerate(strategy_legs):
                try:
                    logger.info(f"🔄 Executing leg {i+1}/{len(strategy_legs)}: {leg.action} {leg.contracts}x {leg.tradingsymbol}")
                    
                    # Execute the order (dry run for now)
                    result = await self.order_manager.place_options_order(leg, dry_run=True)
                    execution_results.append(result)
                    
                    if result.get('status') == 'success':
                        successful_orders.append(result)
                        logger.info(f"[OK] Leg {i+1} executed: {result.get('order_id', 'DRY_RUN')}")
                    else:
                        logger.error(f"[ERROR] Leg {i+1} failed: {result.get('message', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"[ERROR] Error executing leg {i+1}: {e}")
                    execution_results.append({'status': 'error', 'message': str(e)})
            
            # 6. Create active position if we have successful orders
            if successful_orders:
                # Calculate entry premium
                total_premium = sum(leg.theoretical_price * leg.contracts * leg.lot_size 
                                for leg in strategy_legs if leg.action == 'BUY')
                
                self.active_position = ActivePosition(
                    signal=signal,
                    option_legs=strategy_legs,  # [OK] Now we have actual legs!
                    strategy_type=strategy_type,
                    entry_time=datetime.now(),
                    entry_price=signal.current_price,
                    entry_premium=total_premium,
                    order_ids=[r.get('order_id') for r in successful_orders]
                )
                
                # Add lifecycle ID
                self.active_position.lifecycle_id = lifecycle_id
                
                # 7. Start position monitoring
                self.state = TradingState.MONITORING
                await self.enhanced_position_management(self.active_position)
                
                # 8. Send entry notification
                await self._send_entry_notification(self.active_position)
                
                logger.info(f"[OK] Position entered: {signal.ticker} with {len(strategy_legs)} legs")
                
            else:
                logger.error(f"[ERROR] All order executions failed for {signal.ticker}")
                self.state = TradingState.IDLE
        
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            self.state = TradingState.IDLE
    
    async def _monitor_active_position(self):
        """Monitor the active position"""
        
        if not self.active_position:
            return
        
        try:
            # 1. Update position P&L
            await self._update_position_pnl()
            
            # 2. Check exit conditions
            exit_analysis = await self._analyze_exit_conditions()
            
            # 3. Execute exit if required
            if exit_analysis.get('should_exit'):
                await self.execute_enhanced_exit(
                    self.active_position, 
                    exit_analysis.get('exit_reason', ExitReason.MANUAL_OVERRIDE)
                )
        
        except Exception as e:
            logger.error(f"Error monitoring position: {e}")
    
    async def _update_position_pnl(self):
        """Update current position P&L"""
        
        if not self.active_position:
            return
        
        try:
            # Get current positions from Zerodha
            positions = await self._get_current_positions()
            
            # Calculate total P&L for our position
            total_pnl = 0.0
            
            for leg in self.active_position.option_legs:
                for pos in positions.get('net', []) + positions.get('day', []):
                    if pos.get('tradingsymbol') == leg.tradingsymbol:
                        total_pnl += float(pos.get('pnl', 0))
                        break
            
            # Update position
            self.active_position.current_pnl = total_pnl
            self.active_position.max_profit = max(
                self.active_position.max_profit, total_pnl
            )
            self.active_position.max_loss = min(
                self.active_position.max_loss, total_pnl
            )
            
            # Update daily performance
            self.daily_performance.unrealized_pnl = total_pnl
            
        except Exception as e:
            logger.error(f"Error updating position P&L: {e}")
    
    async def _analyze_exit_conditions(self) -> Dict:
        """Analyze if position should be exited"""
        
        if not self.active_position:
            return {'should_exit': False}
        
        try:
            # Calculate profit percentage
            investment = (self.active_position.entry_premium * 
                         sum(leg.contracts * leg.lot_size for leg in self.active_position.option_legs))
            
            if investment <= 0:
                return {'should_exit': False}
            
            profit_percent = (self.active_position.current_pnl / investment) * 100
            
            # Check profit targets
            if profit_percent >= 15:
                return {
                    'should_exit': True,
                    'exit_reason': ExitReason.PROFIT_TARGET,
                    'reason': f'Profit target hit: {profit_percent:.1f}%'
                }
            
            # Check stop losses
            if profit_percent <= -15:
                return {
                    'should_exit': True,
                    'exit_reason': ExitReason.EMERGENCY_STOP,
                    'reason': f'Stop loss hit: {profit_percent:.1f}%'
                }
            
            # Check time-based exit
            time_until_square_off = self.bot_config.time_until_square_off()
            if time_until_square_off <= 30:
                return {
                    'should_exit': True,
                    'exit_reason': ExitReason.TIME_STOP,
                    'reason': 'Square-off time approaching'
                }
            
            # Use intelligent analysis for other conditions
            market_data = await self.market_data_provider.fetch_live_data(
                self.active_position.signal.ticker
            )
            
            health_analysis = await self.intelligent_risk_manager.analyze_position_health(
                self.active_position, 
                market_data.get('current_price', 0), 
                market_data
            )
            
            if health_analysis.get('action') == 'EXIT_IMMEDIATELY':
                return {
                    'should_exit': True,
                    'exit_reason': ExitReason.TECHNICAL_REVERSAL,
                    'reason': health_analysis.get('reason', 'Technical exit signal')
                }
            
            return {'should_exit': False}
        
        except Exception as e:
            logger.error(f"Error analyzing exit conditions: {e}")
            return {'should_exit': False}
    
    async def inject_test_signal(self, ticker: str, direction: str, confidence: float = 0.75) -> bool:
        """Inject test signal for development/testing"""
        
        try:
            # Get current price
            market_data = await self.market_data_provider.fetch_live_data(ticker)
            current_price = market_data.get('current_price', 24500)
            
            # Create test signal
            test_signal = TradingSignal(
                ticker=ticker,
                direction=direction,
                confidence=confidence,
                strategy='TEST_INJECTION',
                current_price=current_price,
                timestamp=datetime.now(),
                source='test_injection',
                market_data=market_data
            )
            
            # Add to signal queue
            await self.signal_queue.put(test_signal)
            
            logger.info(f"🧪 Test signal injected: {ticker} {direction} ({confidence:.1%})")
            return True
            
        except Exception as e:
            logger.error(f"Error injecting test signal: {e}")
            return False
    
    async def manual_exit(self, reason: str = "manual_override") -> bool:
        """Manually exit current position"""
        
        try:
            if not self.active_position:
                logger.warning("No active position to exit")
                return False
            
            # Map reason to ExitReason enum
            exit_reason = ExitReason.MANUAL_OVERRIDE
            if reason == "emergency":
                exit_reason = ExitReason.EMERGENCY_STOP
            elif reason == "profit":
                exit_reason = ExitReason.PROFIT_TARGET
            
            # Execute exit
            result = await self.execute_enhanced_exit(self.active_position, exit_reason)
            
            return result.get('exit_completed', False)
            
        except Exception as e:
            logger.error(f"Error in manual exit: {e}")
            return False
    
    async def manual_analysis(self, ticker: str) -> bool:
        """Perform manual analysis on a ticker"""
        
        try:
            logger.info(f"🔍 Starting manual analysis for {ticker}")
            
            # 1. Get market data
            market_data = await self.market_data_provider.fetch_live_data(ticker)
            
            # 2. Get technical analysis
            tech_analysis = await self.technical_analyzer.analyze_symbol_for_options(
                ticker, market_data.get('current_price', 0), market_data, 'intraday'
            )
            
            # 3. Get options chain
            option_chain = await self.options_chain_provider.fetch_option_chain(ticker)
            
            # 4. Analyze best options
            options_analysis = await self.options_analyzer.analyze_best_options(
                ticker, market_data, tech_analysis
            )
            
            # 5. Log results
            logger.info(f"[CHART] Manual Analysis Results for {ticker}:")
            logger.info(f"Current Price: ₹{market_data.get('current_price', 0):.2f}")
            logger.info(f"Market Bias: {tech_analysis.get('market_bias', 'NEUTRAL')}")
            logger.info(f"Confidence: {tech_analysis.get('confidence_score', 0):.1%}")
            logger.info(f"Best Strategy: {options_analysis.get('recommended_strategy', 'None')}")
            
            # 6. Send to Telegram if available
            if self.telegram_bot:
                message = f"🔍 <b>Manual Analysis: {ticker}</b>\n\n"
                message += f"[MONEY] Price: ₹{market_data.get('current_price', 0):.2f}\n"
                message += f"[UP] Bias: {tech_analysis.get('market_bias', 'NEUTRAL')}\n"
                message += f"[TARGET] Confidence: {tech_analysis.get('confidence_score', 0):.1%}\n"
                message += f"⚡ Strategy: {options_analysis.get('recommended_strategy', 'None')}"
                
                await self.telegram_bot.send_message(message, parse_mode='HTML')
            
            return True
            
        except Exception as e:
            logger.error(f"Error in manual analysis: {e}")
            return False
    
    def get_current_status(self) -> Dict:
        """Get current bot status"""
        
        try:
            # Calculate win rate
            total_trades = self.daily_performance.trades_profitable + self.daily_performance.trades_loss
            win_rate = (self.daily_performance.trades_profitable / max(1, total_trades)) * 100
            
            # Calculate ROI
            roi_percent = 0
            if self.daily_performance.starting_capital > 0:
                roi_percent = (self.daily_performance.total_pnl / self.daily_performance.starting_capital) * 100
            
            return {
                'timestamp': datetime.now().isoformat(),
                'state': self.state.value,
                'monitoring_active': self.monitoring_active,
                'session_duration': str(datetime.now() - self.session_start_time).split('.')[0],
                
                # Position status
                'active_position': bool(self.active_position),
                'position_ticker': self.active_position.signal.ticker if self.active_position else None,
                'position_pnl': self.active_position.current_pnl if self.active_position else 0,
                'position_duration': str(datetime.now() - self.active_position.entry_time).split('.')[0] if self.active_position else None,
                
                # Daily performance
                'daily_pnl': self.daily_performance.total_pnl,
                'daily_roi_percent': roi_percent,
                'trades_today': total_trades,
                'win_rate_percent': win_rate,
                'starting_capital': self.daily_performance.starting_capital,
                'current_capital': self.daily_performance.current_capital,
                
                # Signal statistics
                'signals_received_today': self.signals_received_today,
                'signals_processed_today': self.signals_processed_today,
                'signal_conversion_rate': (self.signals_processed_today / max(1, self.signals_received_today)) * 100,
                'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
                
                # Risk status
                'emergency_stop': self.emergency_stop_triggered,
                'recovery_mode': self.recovery_mode,
                'consecutive_losses': self.consecutive_losses,
                
                # Market timing
                'market_open': self.bot_config.is_market_hours() if hasattr(self, 'bot_config') else True,
                'time_until_square_off': self.bot_config.time_until_square_off() if hasattr(self, 'bot_config') else 0,
                
                # System health
                'api_connected': True,  # Will be checked in real implementation
                'telegram_active': self.telegram_bot is not None,
                'components_initialized': hasattr(self, 'bot_config'),
            }
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'state': 'ERROR'
            }
    
    def _start_signal_interception(self):
        """Start signal interception from main bot"""
        
        # This would connect to your main bot's signal system
        # For now, it's a placeholder for the integration
        logger.info("[SIGNAL] Signal interception started")
        
        # In real implementation, this would:
        # 1. Connect to main bot's signal broadcast
        # 2. Start listening thread
        # 3. Forward signals to our queue
    
    async def _update_starting_capital(self):
        """Update starting capital from account balance"""
        
        try:
            if hasattr(self, 'capital_manager'):
                capital_status = await self.capital_manager.update_capital_status()
                self.daily_performance.starting_capital = capital_status.get('available_capital', 5000)
                self.daily_performance.current_capital = capital_status.get('available_capital', 5000)
                
                logger.info(f"[MONEY] Starting capital updated: ₹{self.daily_performance.starting_capital:,.0f}")
        except Exception as e:
            logger.error(f"Error updating starting capital: {e}")
    
    async def _validate_api_connection(self) -> bool:
        """Validate API connections"""
        
        try:
            # Test Zerodha connection
            profile = self.zerodha.get_profile()
            if not profile:
                return False
            
            # Test market data
            market_data = await self.market_data_provider.fetch_live_data('NIFTY')
            if not market_data:
                return False
            
            logger.info("[OK] API connections validated")
            return True
            
        except Exception as e:
            logger.error(f"API validation failed: {e}")
            return False
    
    async def _send_startup_notification(self, test_mode: bool):
        """Send startup notification"""
        
        try:
            if self.telegram_bot:
                mode = "🧪 TEST MODE" if test_mode else "[ROCKET] LIVE TRADING"
                message = f"{mode}\n\n"
                message += f"[MONEY] Capital: ₹{self.daily_performance.starting_capital:,.0f}\n"
                message += f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}\n"
                message += f"⏰ Started: {datetime.now().strftime('%H:%M:%S')}\n"
                message += f"[TARGET] Target: 8-15% daily returns"
                
                await self.telegram_bot.send_message(message)
                
        except Exception as e:
            logger.error(f"Error sending startup notification: {e}")
    
    async def _send_entry_notification(self, position: ActivePosition):
        """Send position entry notification"""
        
        try:
            if self.telegram_bot:
                message = f"[UP] <b>POSITION ENTERED</b>\n\n"
                message += f"[CHART] <b>{position.signal.ticker}</b> {position.signal.direction.upper()}\n"
                message += f"[MONEY] Entry: ₹{position.entry_price:.2f}\n"
                message += f"[TARGET] Strategy: {position.strategy_type.value if hasattr(position, 'strategy_type') else 'Unknown'}\n"
                message += f"⏰ Time: {position.entry_time.strftime('%H:%M:%S')}\n"
                message += f"🔥 Confidence: {position.signal.confidence:.1%}"
                
                await self.telegram_bot.send_message(message, parse_mode='HTML')
                
        except Exception as e:
            logger.error(f"Error sending entry notification: {e}")
    
    async def _update_daily_performance(self):
        """Update daily performance metrics"""
        
        try:
            # Update current capital
            if hasattr(self, 'capital_manager'):
                capital_status = await self.capital_manager.update_capital_status()
                self.daily_performance.current_capital = capital_status.get('available_capital', 0)
            
            # Calculate total P&L
            realized_pnl = sum(pos.current_pnl for pos in self.position_history if pos.current_pnl)
            unrealized_pnl = self.active_position.current_pnl if self.active_position else 0
            
            self.daily_performance.realized_pnl = realized_pnl
            self.daily_performance.unrealized_pnl = unrealized_pnl
            self.daily_performance.total_pnl = realized_pnl + unrealized_pnl
            
            # Update drawdown
            current_drawdown = min(0, self.daily_performance.total_pnl)
            self.daily_performance.current_drawdown = current_drawdown
            self.daily_performance.max_drawdown = min(
                self.daily_performance.max_drawdown, current_drawdown
            )
            
        except Exception as e:
            logger.error(f"Error updating daily performance: {e}")
    
    async def _check_emergency_conditions(self) -> bool:
        """Check for emergency stop conditions"""
        
        try:
            # Check daily loss limit
            max_daily_loss = self.bot_config.get('max_daily_loss', 5000)
            if self.daily_performance.total_pnl < -max_daily_loss:
                logger.critical(f"🚨 Daily loss limit exceeded: ₹{self.daily_performance.total_pnl:+.0f}")
                return True
            
            # Check consecutive losses
            if self.consecutive_losses >= 5:
                logger.critical(f"🚨 Too many consecutive losses: {self.consecutive_losses}")
                return True
            
            # Check API connection
            if not await self._validate_api_connection():
                logger.critical("🚨 API connection lost")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking emergency conditions: {e}")
            return True  # Err on side of caution
    
    async def _periodic_learning_update(self):
        """Periodic learning and optimization updates"""
        
        try:
            if hasattr(self, 'learning_system'):
                # Update learning insights
                insights = self.learning_system.get_learning_insights()
                
                # Log insights periodically
                if insights.get('total_trades_analyzed', 0) > 0:
                    logger.info(f"📚 Learning update: {insights.get('total_trades_analyzed')} trades analyzed")
                    
                    # Update best performing strategy
                    best_strategy = insights.get('best_strategy')
                    if best_strategy:
                        self.daily_performance.best_performing_strategy = best_strategy
            
        except Exception as e:
            logger.error(f"Error in periodic learning update: {e}")
    
    async def _cleanup_resources(self):
        """Cleanup resources on shutdown"""
        
        try:
            # Stop monitoring
            self.monitoring_active = False
            
            # Cancel monitoring task
            if self.monitoring_task and not self.monitoring_task.done():
                self.monitoring_task.cancel()
            
            # Stop signal interception
            if self.signal_thread and self.signal_thread.is_alive():
                # Signal thread cleanup would go here
                pass
            
            # Stop position monitoring
            if hasattr(self, 'position_monitor'):
                self.position_monitor.stop_monitoring()
            
            # Save final performance data
            self._save_enhanced_performance_data()
            
            logger.info("[OK] Resources cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")
    
    def initialize_v2_components(self):
        """Initialize v2.0 specific components"""
        
        # Configuration
        self.bot_config = BotConfiguration()
        
        # Enhanced signal processing
        self.signal_processor = EnhancedSignalProcessor(self.bot_config)
        
        # Position monitoring
        self.position_monitor = PositionMonitor(self.zerodha, self.bot_config)
        
        # Performance analytics
        self.performance_analytics = PerformanceAnalytics()
        
        # Dynamic capital manager
        self.capital_manager = DynamicCapitalManager(self.zerodha)
        
        # Intelligent risk manager
        self.intelligent_risk_manager = IntelligentRiskManager(self.technical_analyzer)
        
        # Profit optimizer
        self.profit_optimizer = DynamicProfitOptimizer()
        
        # Multi-leg executor
        self.strategy_executor = MultiLegStrategyExecutor(
            self.options_chain_provider, self.order_manager
        )
        
        # Learning system
        self.learning_system = LearningSystem(self.trade_logger)
        
        logger.info("[OK] v2.0 components initialized successfully")

    async def enhanced_signal_processing(self, signal: TradingSignal) -> Dict:
        """Enhanced signal processing with v2.0 features - FIXED VERSION"""
        
        # 1. Process signal through enhanced processor
        processing_result = await self.signal_processor.process_signal(signal)
        
        if not processing_result['approved']:
            logger.info(f"[ERROR] Signal rejected: {processing_result['rejection_reason']}")
            return {'signal_processed': False, 'processing_result': processing_result}
        
        # 2. Get technical analysis with error handling
        try:
            # [OK] FIXED: Use the correct method from your technical analyzer
            technical_analysis = await self.technical_analyzer.analyze_symbol_for_options(
                signal.ticker, 
                signal.current_price, 
                signal.market_data, 
                'intraday'  # or signal.trading_style if available
            )
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            # Create fallback technical analysis
            technical_analysis = {
                'market_bias': 'NEUTRAL',
                'confidence_score': 0.5,
                'entry_signal': {
                    'signal_type': 'HOLD',
                    'strength': 0.3,
                    'reason': 'Technical analysis unavailable'
                }
            }
        
        # 3. Update capital status
        capital_status = await self.capital_manager.update_capital_status()
        
        # 4. Determine strategy based on capital tier
        allowed_strategies = self.capital_manager.get_allowed_strategies()
        
        # 5. Get strategy recommendation from analyzer
        try:
            # [OK] FIXED: Use the options analyzer's analyze_best_options method
            options_analysis = await self.options_analyzer.analyze_best_options(
                signal.ticker, 
                signal.market_data, 
                technical_analysis
            )
            
            recommended_strategy = options_analysis.get('recommended_strategy', 'LONG_CALL')
            
        except Exception as e:
            logger.error(f"Options analysis failed: {e}")
            # Fallback to simple strategy
            recommended_strategy = 'LONG_CALL' if signal.direction == 'bullish' else 'LONG_PUT'
        
        # 6. Create strategy legs using the multi-leg executor
        try:
            # Convert string strategy to StrategyType enum
            from automated_options_bot import StrategyType  # Import from your main file
            
            strategy_mapping = {
                'LONG_CALL': StrategyType.BUY_CALL,
                'LONG_PUT': StrategyType.BUY_PUT,
                'BULL_CALL_SPREAD': StrategyType.BULL_CALL_SPREAD,
                'BEAR_PUT_SPREAD': StrategyType.BEAR_PUT_SPREAD,
                'LONG_STRADDLE': StrategyType.LONG_STRADDLE,
                'LONG_STRANGLE': StrategyType.LONG_STRANGLE
            }
            
            strategy_type = strategy_mapping.get(recommended_strategy, StrategyType.BUY_CALL)
            
            strategy_legs = await self.strategy_executor.create_strategy_legs(
                signal, 
                strategy_type,
                capital_status['available_capital'],
                technical_analysis,
                capital_status.get('tier', CapitalTier.TIER_1)
            )
            
            logger.info(f"[TARGET] Strategy legs created: {len(strategy_legs)} legs")
            for i, leg in enumerate(strategy_legs):
                logger.info(f"  Leg {i+1}: {leg.action} {leg.contracts}x {leg.tradingsymbol}")
            
        except Exception as e:
            logger.error(f"Strategy leg creation failed: {e}")
            strategy_legs = []
        
        # 7. Risk check - FIXED: Handle None position properly
        if self.active_position:
            risk_analysis = await self.intelligent_risk_manager.analyze_position_health(
                self.active_position, signal.current_price, signal.market_data
            )
        else:
            # No active position, so no risk analysis needed for new signal
            risk_analysis = {
                'action': 'CONTINUE',
                'reason': 'No active position - proceeding with new signal',
                'confidence': 0.8,
                'analysis_depth': 'none'
            }
        
        # 8. Prepare execution results (dry run for now)
        execution_results = []
        if risk_analysis.get('action') != 'EXIT_IMMEDIATELY':
            for leg in strategy_legs:
                try:
                    # Dry run execution
                    result = await self.order_manager.place_options_order(leg, dry_run=True)
                    execution_results.append(result)
                    logger.info(f"📋 Simulated order: {leg.action} {leg.contracts}x {leg.tradingsymbol} - {result.get('status', 'unknown')}")
                except Exception as e:
                    logger.error(f"Order simulation failed: {e}")
                    execution_results.append({'status': 'error', 'message': str(e)})
        else:
            logger.warning(f"[WARNING] Risk analysis suggests EXIT_IMMEDIATELY - skipping execution")
        
        return {
            'signal_processed': True,
            'processing_result': processing_result,
            'capital_status': capital_status,
            'strategy_legs': len(strategy_legs),
            'actual_strategy_legs': strategy_legs,  # [OK] Added actual legs for execution
            'risk_analysis': risk_analysis,
            'execution_results': execution_results,
            'technical_analysis_available': technical_analysis is not None,
            'technical_analysis': technical_analysis,  # [OK] Added for execution method
            'recommended_strategy': recommended_strategy,
            'strategy_type': strategy_type if 'strategy_type' in locals() else StrategyType.BUY_CALL  # [OK] Added strategy type
        }

    async def enhanced_position_management(self, position: ActivePosition) -> Dict:
        """Enhanced position management with v2.0 features"""
        
        # 1. Start position monitoring
        await self.position_monitor.start_monitoring(position)
        
        # 2. Get current position health
        market_data = await self.market_data_provider.fetch_live_data(position.signal.ticker)
        current_price = market_data['current_price']
        
        # 3. Analyze position health
        health_analysis = await self.intelligent_risk_manager.analyze_position_health(
            position, current_price, market_data
        )
        
        # 4. Check for profit optimization opportunities
        if position.current_pnl > 0:
            investment = position.entry_premium * sum(leg.contracts * leg.lot_size for leg in position.option_legs)
            profit_percent = (position.current_pnl / investment * 100) if investment > 0 else 0
            
            # Get technical analysis for profit optimization
            try:
                tech_analysis = await self.technical_analyzer.analyze_symbol_for_options(
                    position.signal.ticker, current_price, market_data, 'intraday'
                )
                
                profit_analysis = await self.profit_optimizer.analyze_profit_continuation(
                    position, profit_percent, tech_analysis
                )
                
                # Execute profit optimization action
                if profit_analysis['action'] == 'TAKE_PROFIT':
                    logger.info(f"[MONEY] Taking profit: {profit_analysis['reason']}")
                    # Here you would execute the exit
                    
                elif profit_analysis['action'] == 'TRAIL_STOP':
                    # Update trailing stop
                    new_stop = profit_analysis['suggested_stop']
                    position.trailing_stop = new_stop
                    logger.info(f"[UP] Trailing stop updated to ₹{new_stop:.2f}")
                    
            except Exception as e:
                logger.error(f"Profit optimization failed: {e}")
                profit_analysis = {'action': 'CONTINUE', 'reason': 'Technical analysis unavailable'}
        else:
            profit_analysis = {'action': 'MONITOR', 'reason': 'Position not in profit'}
        
        # 5. Record decision in learning system
        lifecycle_id = getattr(position, 'lifecycle_id', None)
        if lifecycle_id:
            self.learning_system.update_trade_lifecycle(
                lifecycle_id, 
                'decision_point',
                {
                    'decision': health_analysis.get('action', 'CONTINUE'),
                    'reason': health_analysis.get('reason', 'Position monitoring'),
                    'confidence': health_analysis.get('confidence', 0.5),
                    'profit_action': profit_analysis['action'],
                    'current_pnl': position.current_pnl
                }
            )
        
        return {
            'position_health': health_analysis,
            'profit_optimization': profit_analysis,
            'monitoring_active': True,
            'current_pnl': position.current_pnl,
            'max_profit': position.max_profit,
            'max_loss': position.max_loss
        }

    async def execute_enhanced_exit(self, position: ActivePosition, exit_reason: ExitReason) -> Dict:
        """Enhanced exit execution with v2.0 features"""
        
        logger.info(f"🚪 Executing enhanced exit for {position.signal.ticker}: {exit_reason.value}")
        
        try:
            # 1. Stop position monitoring
            position_id = f"{position.signal.ticker}_{position.entry_time.isoformat()}"
            self.position_monitor.stop_monitoring(position_id)
            
            # 2. Execute exit for all legs
            exit_results = []
            total_pnl = 0
            
            for leg in position.option_legs:
                try:
                    # Calculate quantity to exit
                    total_quantity = leg.contracts * leg.lot_size
                    
                    # Execute square off
                    result = await self.order_manager.square_off_position(
                        leg.tradingsymbol, 
                        total_quantity if leg.action == 'BUY' else -total_quantity
                    )
                    
                    exit_results.append({
                        'leg': leg.tradingsymbol,
                        'quantity': total_quantity,
                        'result': result
                    })
                    
                    if result.get('status') == 'success':
                        logger.info(f"[OK] Exited {leg.tradingsymbol}: {total_quantity} lots")
                    else:
                        logger.error(f"[ERROR] Exit failed for {leg.tradingsymbol}: {result.get('message')}")
                    
                except Exception as e:
                    logger.error(f"Error exiting leg {leg.tradingsymbol}: {e}")
                    exit_results.append({
                        'leg': leg.tradingsymbol,
                        'error': str(e)
                    })
            
            # 3. Calculate final P&L
            final_pnl = position.current_pnl
            trade_duration = datetime.now() - position.entry_time
            
            # 4. Update performance analytics
            trade_data = {
                'ticker': position.signal.ticker,
                'strategy': position.strategy_type.value if hasattr(position, 'strategy_type') else 'UNKNOWN',
                'direction': position.signal.direction,
                'entry_time': position.entry_time.isoformat(),
                'exit_time': datetime.now().isoformat(),
                'duration_minutes': trade_duration.total_seconds() / 60,
                'entry_premium': position.entry_premium,
                'pnl': final_pnl,
                'max_profit': position.max_profit,
                'max_loss': position.max_loss,
                'exit_reason': exit_reason.value,
                'investment': position.entry_premium * sum(leg.contracts * leg.lot_size for leg in position.option_legs),
                'roi_percent': (final_pnl / (position.entry_premium * sum(leg.contracts * leg.lot_size for leg in position.option_legs)) * 100) if position.entry_premium > 0 else 0
            }
            
            self.performance_analytics.record_trade(trade_data)
            
            # 5. Complete learning lifecycle
            lifecycle_id = getattr(position, 'lifecycle_id', None)
            if lifecycle_id:
                self.learning_system.update_trade_lifecycle(
                    lifecycle_id,
                    'exit',
                    {
                        'exit_prices': [leg.market_price or leg.theoretical_price for leg in position.option_legs],
                        'final_pnl': final_pnl,
                        'exit_reason': exit_reason
                    }
                )
                
                # Analyze completed trade
                learning_insights = self.learning_system.analyze_completed_trade(lifecycle_id)
                logger.info(f"📚 Learning insights: {learning_insights}")
            
            # 6. Update daily performance
            self.daily_performance.total_pnl += final_pnl
            self.daily_performance.realized_pnl += final_pnl
            
            if final_pnl > 0:
                self.daily_performance.trades_profitable += 1
            else:
                self.daily_performance.trades_loss += 1
            
            # 7. Check if daily target achieved
            daily_target_check = self._check_daily_target_status()
            
            # 8. Send notifications
            await self._send_exit_notification(position, exit_reason, final_pnl, trade_duration)
            
            logger.info(f"[OK] Enhanced exit completed: P&L ₹{final_pnl:+.0f}, Duration: {trade_duration}")
            
            return {
                'exit_completed': True,
                'final_pnl': final_pnl,
                'trade_duration': str(trade_duration).split('.')[0],
                'exit_results': exit_results,
                'daily_target_status': daily_target_check,
                'learning_insights': learning_insights if lifecycle_id else None,
                'trade_summary': trade_data
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced exit execution: {e}")
            return {
                'exit_completed': False,
                'error': str(e),
                'position_ticker': position.signal.ticker
            }

    def _check_daily_target_status(self) -> Dict:
        """Check daily target achievement status"""
        
        current_pnl = self.daily_performance.total_pnl
        starting_capital = self.daily_performance.starting_capital
        
        # Calculate percentages
        current_roi = (current_pnl / starting_capital * 100) if starting_capital > 0 else 0
        min_target = self.bot_config.get('daily_profit_min', 0.08) * 100  # 8%
        aggressive_target = self.bot_config.get('daily_profit_aggressive', 0.15) * 100  # 15%
        
        status = {
            'current_pnl': current_pnl,
            'current_roi_percent': current_roi,
            'min_target_percent': min_target,
            'aggressive_target_percent': aggressive_target,
            'min_target_achieved': current_roi >= min_target,
            'aggressive_target_achieved': current_roi >= aggressive_target,
            'remaining_to_min_target': max(0, (min_target * starting_capital / 100) - current_pnl),
            'remaining_to_aggressive_target': max(0, (aggressive_target * starting_capital / 100) - current_pnl)
        }
        
        # Determine status message
        if status['aggressive_target_achieved']:
            status['status_message'] = "🎉 Aggressive daily target achieved!"
            status['recommendation'] = "Consider reducing position sizes or stopping for the day"
        elif status['min_target_achieved']:
            status['status_message'] = "[OK] Minimum daily target achieved"
            status['recommendation'] = "Continue trading with conservative approach"
        elif current_roi > 0:
            status['status_message'] = "[UP] Positive P&L, working towards target"
            status['recommendation'] = "Continue normal trading operations"
        else:
            loss_percent = abs(current_roi)
            if loss_percent > 5:
                status['status_message'] = "🚨 Significant daily loss"
                status['recommendation'] = "Review strategy and consider reducing risk"
            else:
                status['status_message'] = "[CHART] Minor daily loss"
                status['recommendation'] = "Continue with cautious approach"
        
        return status

    async def _send_exit_notification(self, position: ActivePosition, exit_reason: ExitReason, 
                                    final_pnl: float, duration: timedelta):
        """Send exit notification via Telegram"""
        
        try:
            if not self.telegram_bot:
                return
            
            # Determine emoji based on P&L
            if final_pnl > 100:
                emoji = "🎉"
            elif final_pnl > 0:
                emoji = "[MONEY]"
            elif final_pnl > -100:
                emoji = "😐"
            else:
                emoji = "😞"
            
            # Format duration
            duration_str = str(duration).split('.')[0]  # Remove microseconds
            
            # Calculate ROI
            investment = position.entry_premium * sum(leg.contracts * leg.lot_size for leg in position.option_legs)
            roi_percent = (final_pnl / investment * 100) if investment > 0 else 0
            
            # Build message
            message = f"{emoji} <b>POSITION CLOSED</b>\n\n"
            message += f"[CHART] <b>{position.signal.ticker}</b> {position.signal.direction.upper()}\n"
            message += f"[MONEY] P&L: <b>₹{final_pnl:+,.0f}</b> ({roi_percent:+.1f}%)\n"
            message += f"⏰ Duration: {duration_str}\n"
            message += f"🚪 Exit: {exit_reason.value.replace('_', ' ').title()}\n\n"
            
            # Add daily summary
            daily_status = self._check_daily_target_status()
            message += f"[UP] Daily P&L: ₹{daily_status['current_pnl']:+,.0f} ({daily_status['current_roi_percent']:+.1f}%)\n"
            message += f"[TARGET] Target: {daily_status['status_message']}"
            
            await self.telegram_bot.send_message(message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error sending exit notification: {e}")

    async def get_enhanced_status(self) -> Dict:
        """Get comprehensive bot status with v2.0 features"""
        
        try:
            # Basic status
            basic_status = self.get_current_status()
            
            # Capital status
            capital_status = await self.capital_manager.update_capital_status()
            
            # Performance analytics
            daily_summary = self.performance_analytics.get_daily_summary()
            strategy_performance = self.performance_analytics.get_strategy_performance()
            
            # Position monitoring status
            monitoring_status = self.position_monitor.get_monitoring_status()
            
            # Learning insights
            learning_insights = self.learning_system.get_learning_insights()
            
            # Risk tier status
            current_loss = abs(min(0, self.daily_performance.total_pnl))
            risk_tier = self._determine_current_risk_tier(current_loss)
            
            # Market timing
            time_until_square_off = self.bot_config.time_until_square_off()
            is_market_hours = self.bot_config.is_market_hours()
            
            enhanced_status = {
                **basic_status,
                
                'v2_features': {
                    'capital_management': capital_status,
                    'risk_tier': {
                        'current_tier': risk_tier,
                        'tier_description': self._get_risk_tier_description(risk_tier),
                        'position_multiplier': self.bot_config.get('tier_position_multipliers', {}).get(risk_tier, 1.0),
                        'confidence_requirement': self.bot_config.get('tier_confidence_requirements', {}).get(risk_tier, 0.60)
                    },
                    'performance_analytics': {
                        'daily_summary': daily_summary,
                        'strategy_performance': strategy_performance,
                        'learning_insights': learning_insights
                    },
                    'position_monitoring': monitoring_status,
                    'market_timing': {
                        'is_market_hours': is_market_hours,
                        'time_until_square_off': time_until_square_off,
                        'trading_allowed': is_market_hours and time_until_square_off > 60
                    }
                },
                
                'system_health': {
                    'all_components_active': all([
                        hasattr(self, 'capital_manager'),
                        hasattr(self, 'intelligent_risk_manager'),
                        hasattr(self, 'profit_optimizer'),
                        hasattr(self, 'learning_system')
                    ]),
                    'last_update': datetime.now().isoformat(),
                    'version': 'v2.0-enhanced'
                }
            }
            
            return enhanced_status
            
        except Exception as e:
            logger.error(f"Error getting enhanced status: {e}")
            return {
                'error': str(e),
                'basic_status': self.get_current_status(),
                'version': 'v2.0-enhanced'
            }

    def _determine_current_risk_tier(self, current_loss: float) -> int:
        """Determine current risk tier based on losses"""
        
        if current_loss < 150:
            return 1  # Normal
        elif current_loss < 300:
            return 2  # Caution
        elif current_loss < 500:
            return 3  # Danger
        else:
            return 4  # Recovery

    def _get_risk_tier_description(self, tier: int) -> str:
        """Get description for risk tier"""
        
        descriptions = {
            1: "Normal - Full trading capacity",
            2: "Caution - Reduced position sizes",
            3: "Danger - Conservative trading only",
            4: "Recovery - Emergency protocols active"
        }
        
        return descriptions.get(tier, "Unknown tier")

    # ================== TESTING AND UTILITIES ==================

    async def run_enhanced_test_suite(self):
        """Run comprehensive test suite for v2.0 features"""
        
        logger.info("🧪 Running enhanced test suite for v2.0 features...")
        
        test_results = {
            'component_tests': {},
            'integration_tests': {},
            'system_tests': {},
            'overall_success': True
        }
        
        try:
            # 1. Component tests
            test_results['component_tests']['capital_manager'] = await self._test_capital_manager()
            test_results['component_tests']['risk_manager'] = await self._test_risk_manager()
            test_results['component_tests']['profit_optimizer'] = await self._test_profit_optimizer()
            test_results['component_tests']['learning_system'] = self._test_learning_system()
            
            # 2. Integration tests
            test_results['integration_tests']['signal_processing'] = await self._test_signal_processing_integration()
            test_results['integration_tests']['position_management'] = await self._test_position_management_integration()
            
            # 3. System tests
            test_results['system_tests']['full_workflow'] = await self._test_full_workflow()
            
            # Check overall success
            all_tests = []
            for category in test_results.values():
                if isinstance(category, dict):
                    all_tests.extend(category.values())
            
            test_results['overall_success'] = all(
                test.get('passed', False) for test in all_tests if isinstance(test, dict)
            )
            
            if test_results['overall_success']:
                logger.info("[OK] All v2.0 tests passed!")
            else:
                logger.warning("[WARNING] Some v2.0 tests failed - check results")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error running test suite: {e}")
            test_results['error'] = str(e)
            test_results['overall_success'] = False
            return test_results

    async def _test_capital_manager(self) -> Dict:
        """Test capital manager functionality"""
        try:
            capital_status = await self.capital_manager.update_capital_status()
            
            return {
                'passed': 'available_capital' in capital_status and 'tier' in capital_status,
                'capital_status': capital_status,
                'test_name': 'Capital Manager'
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'test_name': 'Capital Manager'
            }

    async def _test_risk_manager(self) -> Dict:
        """Test intelligent risk manager"""
        try:
            # Create dummy position for testing
            test_signal = TradingSignal(
                ticker='NIFTY',
                direction='bullish',
                confidence=0.75,
                strategy='TEST',
                current_price=24500,
                timestamp=datetime.now()
            )
            
            # Mock position
            test_position = ActivePosition(
                signal=test_signal,
                option_legs=[],
                entry_time=datetime.now(),
                entry_price=24500,
                entry_premium=100
            )
            
            # Test analysis
            analysis = await self.intelligent_risk_manager.analyze_position_health(
                test_position, 24500, {'current_price': 24500}
            )
            
            return {
                'passed': 'action' in analysis,
                'analysis_result': analysis,
                'test_name': 'Risk Manager'
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'test_name': 'Risk Manager'
            }

    async def _test_profit_optimizer(self) -> Dict:
        """Test profit optimizer"""
        try:
            # Mock position and technical analysis
            test_position = ActivePosition(
                signal=TradingSignal('NIFTY', 'bullish', 0.75, 'TEST', 24500, datetime.now()),
                option_legs=[],
                entry_time=datetime.now(),
                entry_price=24500,
                entry_premium=100
            )
            
            mock_tech_analysis = {
                'momentum_analysis': {'rsi': 55, 'direction': 'BULLISH'},
                'trend_analysis': {'trend_strength': 0.7},
                'volume_analysis': {'trend': 'INCREASING'}
            }
            
            # Test profit analysis
            analysis = await self.profit_optimizer.analyze_profit_continuation(
                test_position, 8.0, mock_tech_analysis  # 8% profit
            )
            
            return {
                'passed': 'action' in analysis and 'reason' in analysis,
                'analysis_result': analysis,
                'test_name': 'Profit Optimizer'
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'test_name': 'Profit Optimizer'
            }

    def _test_learning_system(self) -> Dict:
        """Test learning system"""
        try:
            # Test lifecycle creation
            lifecycle_id = self.learning_system.start_trade_lifecycle(
                TradingSignal('NIFTY', 'bullish', 0.75, 'TEST', 24500, datetime.now()),
                StrategyType.BUY_CALL,
                {'volatility': 15, 'trend': 'UP'},
                {'rsi': 55, 'support': 24400}
            )
            
            # Test lifecycle update
            self.learning_system.update_trade_lifecycle(
                lifecycle_id,
                'decision_point',
                {'decision': 'CONTINUE', 'reason': 'Test update'}
            )
            
            return {
                'passed': lifecycle_id > 0,
                'lifecycle_id': lifecycle_id,
                'test_name': 'Learning System'
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'test_name': 'Learning System'
            }

    async def _test_signal_processing_integration(self) -> Dict:
        """Test signal processing integration"""
        try:
            # Create test signal
            test_signal = TradingSignal(
                ticker='NIFTY',
                direction='bullish',
                confidence=0.75,
                strategy='TEST_SIGNAL',
                current_price=24500,
                timestamp=datetime.now(),
                source='test_injection'
            )
            
            # Process signal
            result = await self.enhanced_signal_processing(test_signal)
            
            return {
                'passed': 'signal_processed' in result,
                'processing_result': result,
                'test_name': 'Signal Processing Integration'
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'test_name': 'Signal Processing Integration'
            }

    async def _test_position_management_integration(self) -> Dict:
        """Test position management integration"""
        try:
            # This is a mock test since we don't want to create real positions
            test_result = {
                'position_monitoring_available': hasattr(self, 'position_monitor'),
                'risk_manager_available': hasattr(self, 'intelligent_risk_manager'),
                'profit_optimizer_available': hasattr(self, 'profit_optimizer')
            }
            
            return {
                'passed': all(test_result.values()),
                'component_availability': test_result,
                'test_name': 'Position Management Integration'
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'test_name': 'Position Management Integration'
            }

    async def _test_full_workflow(self) -> Dict:
        """Test full workflow without executing trades"""
        try:
            # Get system status
            status = await self.get_enhanced_status()
            
            # Check all components are initialized
            components_ok = status.get('system_health', {}).get('all_components_active', False)
            
            return {
                'passed': components_ok and not status.get('error'),
                'system_status': status,
                'test_name': 'Full Workflow'
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'test_name': 'Full Workflow'
            }

    # ================== EMERGENCY PROCEDURES ==================

    async def emergency_shutdown_v2(self):
        """Enhanced emergency shutdown for v2.0"""
        
        logger.critical("🚨 INITIATING ENHANCED EMERGENCY SHUTDOWN v2.0")
        
        try:
            # 1. Stop all monitoring
            if hasattr(self, 'position_monitor'):
                self.position_monitor.stop_monitoring()
                logger.info("[OK] Position monitoring stopped")
            
            # 2. Stop signal processing
            self.monitoring_active = False
            if self.monitoring_task:
                self.monitoring_task.cancel()
            
            if hasattr(self, 'signal_interceptor'):
                self.signal_interceptor.stop()
                logger.info("[OK] Signal interceptor stopped")
            
            # 3. Emergency exit all positions
            if self.active_position:
                await self.execute_enhanced_exit(self.active_position, ExitReason.EMERGENCY_STOP)
                logger.info("[OK] Emergency position exit completed")
            
            # 4. Save all data
            self._save_enhanced_performance_data()
            
            # 5. Send final notifications
            if self.telegram_bot:
                final_status = await self.get_enhanced_status()
                await self._send_emergency_shutdown_notification(final_status)
            
            # 6. Generate shutdown report
            shutdown_report = self._generate_shutdown_report()
            
            logger.critical("[OK] ENHANCED EMERGENCY SHUTDOWN COMPLETED")
            
            return {
                'shutdown_completed': True,
                'emergency_exits_performed': 1 if self.active_position else 0,
                'data_saved': True,
                'shutdown_report': shutdown_report
            }
            
        except Exception as e:
            logger.critical(f"[ERROR] EMERGENCY SHUTDOWN FAILED: {e}")
            return {
                'shutdown_completed': False,
                'error': str(e),
                'critical_failure': True
            }

    def _save_enhanced_performance_data(self):
        """Save enhanced performance data"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save performance analytics
            daily_summary = self.performance_analytics.get_daily_summary()
            weekly_performance = self.performance_analytics.get_weekly_performance()
            strategy_performance = self.performance_analytics.get_strategy_performance()
            
            performance_data = {
                'timestamp': timestamp,
                'daily_summary': daily_summary,
                'weekly_performance': weekly_performance,
                'strategy_performance': strategy_performance,
                'learning_insights': self.learning_system.get_learning_insights() if hasattr(self, 'learning_system') else {}
            }
            
            filename = f"enhanced_performance_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(performance_data, f, indent=2, default=str)
            
            logger.info(f"💾 Enhanced performance data saved: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving enhanced performance data: {e}")

    def _generate_shutdown_report(self) -> Dict:
        """Generate comprehensive shutdown report"""
        
        return {
            'shutdown_time': datetime.now().isoformat(),
            'session_duration': str(datetime.now() - getattr(self, 'session_start_time', datetime.now())),
            'daily_performance': {
                'total_pnl': self.daily_performance.total_pnl,
                'trades_executed': self.daily_performance.trades_executed,
                'win_rate': (self.daily_performance.trades_profitable / max(1, self.daily_performance.trades_executed)) * 100,
                'signals_received': self.daily_performance.signals_received,
                'signals_acted_upon': self.daily_performance.signals_acted_upon
            },
            'system_health': {
                'v2_components_active': all([
                    hasattr(self, 'capital_manager'),
                    hasattr(self, 'intelligent_risk_manager'),
                    hasattr(self, 'profit_optimizer'),
                    hasattr(self, 'learning_system')
                ]),
                'monitoring_active': getattr(self, 'monitoring_active', False),
                'active_positions': 1 if self.active_position else 0
            },
            'recommendations': self._generate_session_recommendations()
        }

    def _generate_session_recommendations(self) -> List[str]:
        """Generate recommendations based on session performance"""
        
        recommendations = []
        
        # Performance-based recommendations
        if self.daily_performance.total_pnl > 0:
            recommendations.append("[OK] Profitable session - review successful strategies")
        else:
            recommendations.append("[CHART] Review losing trades for learning opportunities")
        
        # Win rate recommendations
        win_rate = (self.daily_performance.trades_profitable / max(1, self.daily_performance.trades_executed)) * 100
        if win_rate < 50:
            recommendations.append("[TARGET] Focus on improving entry criteria and signal quality")
        elif win_rate > 70:
            recommendations.append("🎉 Excellent win rate - consider scaling up position sizes")
        
        # Signal conversion recommendations
        conversion_rate = (self.daily_performance.signals_acted_upon / max(1, self.daily_performance.signals_received)) * 100
        if conversion_rate < 30:
            recommendations.append("🔍 Review signal filtering - may be too restrictive")
        elif conversion_rate > 80:
            recommendations.append("[WARNING] High signal conversion rate - ensure quality over quantity")
       
        # Capital tier recommendations
        if hasattr(self.daily_performance, 'capital_tier'):
            if self.daily_performance.capital_tier == CapitalTier.TIER_1:
                if self.daily_performance.total_pnl > self.daily_performance.starting_capital * 0.05:
                    recommendations.append("[MONEY] Consider upgrading capital for multi-leg strategies")
            elif self.daily_performance.capital_tier == CapitalTier.TIER_2:
                if win_rate > 60:
                    recommendations.append("[UP] Good performance - ready for advanced strategies")
            else:  # TIER_3
                if conversion_rate < 40:
                    recommendations.append("[TARGET] Utilize full strategy arsenal - low conversion rate")
        
        # Risk tier recommendations
        current_risk_tier = getattr(self.daily_performance, 'current_risk_tier', 1)
        if current_risk_tier > 2:
            recommendations.append("🛡️ High risk tier active - focus on capital preservation")
        elif current_risk_tier == 1 and win_rate > 65:
            recommendations.append("[ROCKET] Low risk environment - consider increasing position sizes")
        
        # Daily target recommendations
        if hasattr(self.daily_performance, 'daily_profit_target'):
            target_achievement = (self.daily_performance.total_pnl / self.daily_performance.daily_profit_target) * 100
            if target_achievement > 100:
                recommendations.append("[TARGET] Daily target achieved - consider profit protection")
            elif target_achievement > 150:
                recommendations.append("🏆 Exceptional performance - implement trailing stops")
            elif target_achievement < 25 and self.daily_performance.trades_executed > 3:
                recommendations.append("[DOWN] Below target pace - review strategy effectiveness")
        
        # Strategy-specific recommendations
        if hasattr(self, 'learning_system'):
            learning_insights = self.learning_system.get_learning_insights()
            best_strategy = learning_insights.get('best_strategy')
            if best_strategy:
                recommendations.append(f"[STAR] Best performing strategy: {best_strategy.value}")
            
            # Pattern-based recommendations
            if learning_insights.get('pattern_success_rates'):
                best_pattern = max(learning_insights['pattern_success_rates'].items(), 
                                    key=lambda x: x[1]['success'] / x[1]['total'] if x[1]['total'] > 0 else 0)
                if best_pattern[1]['total'] > 2:
                    success_rate = best_pattern[1]['success'] / best_pattern[1]['total']
                    if success_rate > 0.7:
                        recommendations.append(f"🎨 Focus on {best_pattern[0]} pattern (success rate: {success_rate*100:.1f}%)")
        
        # Time-based recommendations
        current_time = datetime.now()
        if current_time.hour >= 14:  # After 2 PM
            if self.daily_performance.total_pnl > 0:
                recommendations.append("⏰ Late session profits - consider early close")
            else:
                recommendations.append("⏰ Late session - avoid overtrading")
        
        # Market condition recommendations
        if hasattr(self, 'performance_analytics'):
            recent_volatility = getattr(self.performance_analytics, 'recent_volatility', 'NORMAL')
            if recent_volatility == 'HIGH':
                recommendations.append("🌊 High volatility - use wider stops and smaller sizes")
            elif recent_volatility == 'LOW':
                recommendations.append("😴 Low volatility - consider breakout strategies")
        
        return recommendations
    
    # ================== SIGNAL INTERCEPTION METHODS ==================
    
    def setup_signal_interception(self, main_bot_instance, integration_method: str = "direct_callback", **kwargs):
        """
        Enhanced setup for signal interception with multiple integration methods
        
        Args:
            main_bot_instance: Reference to your main trading bot
            integration_method: Method of integration ('direct_callback', 'method_polling', 
                            'shared_queue', 'file_watcher', 'database_watcher')
            **kwargs: Method-specific configuration parameters
        """
        
        try:
            logger.info(f"[WRENCH] Setting up signal interception with method: {integration_method}")
            
            # Store main bot reference
            self.main_bot = main_bot_instance
            self.integration_method = integration_method
            self.integration_config = kwargs
            
            # Initialize signal interception components
            self._initialize_signal_components()
            
            # Setup method-specific integration
            setup_success = self._setup_integration_method(integration_method, **kwargs)
            
            if not setup_success:
                logger.error(f"[ERROR] Failed to setup integration method: {integration_method}")
                return False
            
            # Start signal listener thread with enhanced functionality
            self.signal_thread = threading.Thread(
                target=self._enhanced_signal_listener_thread,
                name="SignalListenerThread",
                daemon=True
            )
            self.signal_thread.start()
            
            # Initialize signal monitoring
            self._start_signal_monitoring()
            
            # Test the connection
            connection_test = self._test_signal_connection()
            
            if connection_test['success']:
                logger.info(f"[OK] Signal interception setup complete - {integration_method}")
                logger.info(f"🔗 Connection test: {connection_test['message']}")
                
                # Send setup notification
                if hasattr(self, 'telegram_bot') and self.telegram_bot:
                    asyncio.create_task(self._send_setup_notification(integration_method, connection_test))
                
                return True
            else:
                logger.warning(f"[WARNING] Signal interception setup with warnings: {connection_test['message']}")
                return True  # Still proceed but with warnings
                
        except Exception as e:
            logger.error(f"[ERROR] Error setting up signal interception: {e}", exc_info=True)
            return False

    def _initialize_signal_components(self):
        """Initialize components needed for signal interception"""
        
        # Signal processing queues
        self.signal_buffer = queue.Queue(maxsize=50)
        self.priority_signal_buffer = queue.Queue(maxsize=10)
        
        # Threading controls
        self.stop_signal_listener = threading.Event()
        self.signal_listener_active = False
        
        # Signal tracking
        self.signal_history = []
        self.signal_errors = []
        self.last_signal_check = datetime.now()
        
        # Statistics
        self.signal_stats = {
            'total_received': 0,
            'total_processed': 0,
            'total_errors': 0,
            'average_latency': 0.0,
            'sources': {},
            'start_time': datetime.now()
        }
        
        # Configuration
        self.max_signal_age = 300  # 5 minutes
        self.signal_check_interval = 1.0  # 1 second
        self.max_retries = 3
        
        logger.info("[WRENCH] Signal interception components initialized")

    def _setup_integration_method(self, method: str, **kwargs) -> bool:
        """Setup specific integration method"""
        
        try:
            if method == "direct_callback":
                return self._setup_direct_callback(**kwargs)
            elif method == "method_polling":
                return self._setup_method_polling(**kwargs)
            elif method == "shared_queue":
                return self._setup_shared_queue(**kwargs)
            elif method == "file_watcher":
                return self._setup_file_watcher(**kwargs)
            elif method == "database_watcher":
                return self._setup_database_watcher(**kwargs)
            elif method == "websocket":
                return self._setup_websocket(**kwargs)
            else:
                logger.error(f"[ERROR] Unknown integration method: {method}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting up integration method {method}: {e}")
            return False

    def _setup_direct_callback(self, **kwargs) -> bool:
        """Setup direct callback integration - main bot calls automation bot directly"""
        
        try:
            # Register callback with main bot
            if hasattr(self.main_bot, 'register_signal_callback'):
                callback_success = self.main_bot.register_signal_callback(
                    self.receive_signal_from_main_bot
                )
                
                if callback_success:
                    logger.info("[OK] Direct callback registered with main bot")
                    self.signal_retrieval_method = 'callback'
                    return True
                else:
                    logger.error("[ERROR] Failed to register callback with main bot")
                    return False
            else:
                logger.error("[ERROR] Main bot does not support callback registration")
                return False
                
        except Exception as e:
            logger.error(f"Error setting up direct callback: {e}")
            return False

    def _setup_method_polling(self, method_name: str = "get_latest_signals", 
                            poll_interval: float = 2.0, **kwargs) -> bool:
        """Setup method polling - regularly call main bot method for signals"""
        
        try:
            # Check if main bot has the required method
            if not hasattr(self.main_bot, method_name):
                logger.error(f"[ERROR] Main bot does not have method: {method_name}")
                return False
            
            self.signal_retrieval_method = 'polling'
            self.polling_method_name = method_name
            self.polling_interval = poll_interval
            self.last_poll_time = datetime.now()
            
            logger.info(f"[OK] Method polling setup: {method_name} every {poll_interval}s")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up method polling: {e}")
            return False

    def _setup_shared_queue(self, queue_instance=None, queue_name: str = "signal_queue", **kwargs) -> bool:
        """Setup shared queue integration - main bot and automation bot share a queue"""
        
        try:
            if queue_instance:
                self.shared_signal_queue = queue_instance
            elif hasattr(self.main_bot, queue_name):
                self.shared_signal_queue = getattr(self.main_bot, queue_name)
            else:
                logger.error(f"[ERROR] Shared queue not found: {queue_name}")
                return False
            
            self.signal_retrieval_method = 'shared_queue'
            self.shared_queue_timeout = kwargs.get('timeout', 1.0)
            
            logger.info(f"[OK] Shared queue integration setup: {queue_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up shared queue: {e}")
            return False

    def _setup_file_watcher(self, signal_file: str = "signals.json", **kwargs) -> bool:
        """Setup file watcher - monitor file for new signals"""
        
        try:
            self.signal_file_path = Path(signal_file)
            self.signal_retrieval_method = 'file_watcher'
            self.last_file_check = 0
            self.file_check_interval = kwargs.get('check_interval', 2.0)
            
            # Create file if it doesn't exist
            if not self.signal_file_path.exists():
                self.signal_file_path.write_text('[]')
                logger.info(f"📁 Created signal file: {signal_file}")
            
            logger.info(f"[OK] File watcher setup: {signal_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up file watcher: {e}")
            return False

    def _setup_database_watcher(self, db_path: str = "signals.db", 
                            table_name: str = "signals", **kwargs) -> bool:
        """Setup database watcher - monitor database table for new signals"""
        
        try:
            self.signal_db_path = db_path
            self.signal_table_name = table_name
            self.signal_retrieval_method = 'database_watcher'
            self.last_signal_id = 0
            self.db_check_interval = kwargs.get('check_interval', 3.0)
            
            # Test database connection
            with sqlite3.connect(self.signal_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        strategy TEXT,
                        current_price REAL,
                        signal_data TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        processed BOOLEAN DEFAULT 0
                    )
                """)
                conn.commit()
            
            logger.info(f"[OK] Database watcher setup: {db_path}/{table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up database watcher: {e}")
            return False

    def _setup_websocket(self, websocket_url: str, **kwargs) -> bool:
        """Setup WebSocket integration - receive signals via WebSocket"""
        
        try:
            # This would require additional WebSocket libraries
            # For now, log that it's not implemented
            logger.warning("[WARNING] WebSocket integration not implemented yet")
            return False
            
        except Exception as e:
            logger.error(f"Error setting up WebSocket: {e}")
            return False

    def _enhanced_signal_listener_thread(self):
        """Enhanced signal listener thread with multiple integration support"""
        
        self.signal_listener_active = True
        logger.info("🎧 Enhanced signal listener thread started")
        
        try:
            while self.monitoring_active and not self.stop_signal_listener.is_set():
                try:
                    # Get signals based on integration method
                    signals = self._retrieve_signals()
                    
                    # Process retrieved signals
                    if signals:
                        for signal_data in signals:
                            self._process_retrieved_signal(signal_data)
                    
                    # Update statistics
                    self._update_listener_stats()
                    
                    # Sleep based on integration method
                    sleep_interval = self._get_listener_sleep_interval()
                    time.sleep(sleep_interval)
                    
                except Exception as e:
                    logger.error(f"Error in signal listener iteration: {e}")
                    self.signal_stats['total_errors'] += 1
                    time.sleep(5)  # Wait longer on error
                    
        except Exception as e:
            logger.error(f"Signal listener thread error: {e}", exc_info=True)
        finally:
            self.signal_listener_active = False
            logger.info("🛑 Signal listener thread stopped")

    def _retrieve_signals(self) -> List[Dict]:
        """Retrieve signals based on configured integration method"""
        
        signals = []
        
        try:
            if self.signal_retrieval_method == 'polling':
                signals = self._retrieve_signals_by_polling()
            elif self.signal_retrieval_method == 'shared_queue':
                signals = self._retrieve_signals_from_shared_queue()
            elif self.signal_retrieval_method == 'file_watcher':
                signals = self._retrieve_signals_from_file()
            elif self.signal_retrieval_method == 'database_watcher':
                signals = self._retrieve_signals_from_database()
            elif self.signal_retrieval_method == 'callback':
                # Callback method doesn't need active retrieval
                signals = []
            else:
                logger.warning(f"[WARNING] Unknown signal retrieval method: {self.signal_retrieval_method}")
            
        except Exception as e:
            logger.error(f"Error retrieving signals: {e}")
            
        return signals

    def _retrieve_signals_by_polling(self) -> List[Dict]:
        """Retrieve signals by polling main bot method"""
        
        signals = []
        
        try:
            # Check if it's time to poll
            now = datetime.now()
            if (now - self.last_poll_time).total_seconds() < self.polling_interval:
                return signals
            
            # Call main bot method
            method = getattr(self.main_bot, self.polling_method_name)
            result = method()
            
            if result:
                if isinstance(result, list):
                    signals = result
                elif isinstance(result, dict):
                    signals = [result]
                else:
                    logger.warning(f"[WARNING] Unexpected polling result type: {type(result)}")
            
            self.last_poll_time = now
            
        except Exception as e:
            logger.error(f"Error in signal polling: {e}")
        
        return signals

    def _retrieve_signals_from_shared_queue(self) -> List[Dict]:
        """Retrieve signals from shared queue"""
        
        signals = []
        
        try:
            # Get signals from shared queue (non-blocking)
            while True:
                try:
                    signal_data = self.shared_signal_queue.get(
                        block=False, 
                        timeout=self.shared_queue_timeout
                    )
                    signals.append(signal_data)
                    
                    # Mark task as done if queue supports it
                    if hasattr(self.shared_signal_queue, 'task_done'):
                        self.shared_signal_queue.task_done()
                        
                except queue.Empty:
                    break
                    
        except Exception as e:
            logger.error(f"Error retrieving from shared queue: {e}")
        
        return signals

    def _retrieve_signals_from_file(self) -> List[Dict]:
        """Retrieve signals from file watcher"""
        
        signals = []
        
        try:
            # Check if file was modified
            if not self.signal_file_path.exists():
                return signals
            
            file_mtime = self.signal_file_path.stat().st_mtime
            if file_mtime <= self.last_file_check:
                return signals
            
            # Read and parse file
            file_content = self.signal_file_path.read_text()
            if file_content.strip():
                file_signals = json.loads(file_content)
                
                # Filter for new signals only
                for signal in file_signals:
                    signal_time = signal.get('timestamp', '')
                    if signal_time:
                        try:
                            signal_dt = datetime.fromisoformat(signal_time)
                            if signal_dt.timestamp() > self.last_file_check:
                                signals.append(signal)
                        except:
                            # If timestamp parsing fails, include signal
                            signals.append(signal)
            
            self.last_file_check = file_mtime
            
        except Exception as e:
            logger.error(f"Error reading signal file: {e}")
        
        return signals

    def _retrieve_signals_from_database(self) -> List[Dict]:
        """Retrieve signals from database watcher"""
        
        signals = []
        
        try:
            with sqlite3.connect(self.signal_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get new signals
                cursor.execute(f"""
                    SELECT * FROM {self.signal_table_name} 
                    WHERE id > ? AND processed = 0 
                    ORDER BY id
                """, (self.last_signal_id,))
                
                rows = cursor.fetchall()
                
                for row in rows:
                    signal_data = {
                        'id': row['id'],
                        'ticker': row['ticker'],
                        'direction': row['direction'],
                        'confidence': row['confidence'],
                        'strategy': row['strategy'],
                        'current_price': row['current_price'],
                        'timestamp': row['timestamp']
                    }
                    
                    # Parse additional signal data if available
                    if row['signal_data']:
                        try:
                            additional_data = json.loads(row['signal_data'])
                            signal_data.update(additional_data)
                        except:
                            pass
                    
                    signals.append(signal_data)
                    self.last_signal_id = max(self.last_signal_id, row['id'])
                
                # Mark signals as processed
                if signals:
                    signal_ids = [s['id'] for s in signals]
                    placeholders = ','.join('?' * len(signal_ids))
                    cursor.execute(f"""
                        UPDATE {self.signal_table_name} 
                        SET processed = 1 
                        WHERE id IN ({placeholders})
                    """, signal_ids)
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error retrieving from database: {e}")
        
        return signals

    def _process_retrieved_signal(self, signal_data: Dict):
        """Process a retrieved signal and add to async queue"""
        
        try:
            # Add timestamp if missing
            if 'timestamp' not in signal_data:
                signal_data['timestamp'] = datetime.now().isoformat()
            
            # Add to async queue for processing
            asyncio.run_coroutine_threadsafe(
                self.receive_signal_from_main_bot(signal_data),
                asyncio.get_event_loop()
            )
            
            self.signal_stats['total_received'] += 1
            
        except Exception as e:
            logger.error(f"Error processing retrieved signal: {e}")
            self.signal_stats['total_errors'] += 1

    def _get_listener_sleep_interval(self) -> float:
        """Get appropriate sleep interval based on integration method"""
        
        intervals = {
            'polling': self.polling_interval,
            'shared_queue': 0.5,
            'file_watcher': self.file_check_interval,
            'database_watcher': self.db_check_interval,
            'callback': 2.0,  # Minimal sleep for callback method
            'websocket': 0.1
        }
        
        return intervals.get(self.signal_retrieval_method, 1.0)

    def _update_listener_stats(self):
        """Update listener statistics"""
        
        try:
            # Update running statistics
            current_time = datetime.now()
            
            # Calculate average latency (simplified)
            if hasattr(self, 'last_signal_check'):
                time_diff = (current_time - self.last_signal_check).total_seconds()
                self.signal_stats['average_latency'] = (
                    self.signal_stats['average_latency'] * 0.9 + time_diff * 0.1
                )
            
            self.last_signal_check = current_time
            
        except Exception as e:
            logger.debug(f"Error updating listener stats: {e}")

    def _test_signal_connection(self) -> Dict:
        """Test the signal connection"""
        
        try:
            if self.signal_retrieval_method == 'polling':
                # Test if we can call the polling method
                method = getattr(self.main_bot, self.polling_method_name)
                result = method()
                return {
                    'success': True,
                    'message': f"Polling method {self.polling_method_name} accessible"
                }
                
            elif self.signal_retrieval_method == 'callback':
                return {
                    'success': True,
                    'message': "Callback integration ready"
                }
                
            elif self.signal_retrieval_method == 'shared_queue':
                return {
                    'success': True,
                    'message': f"Shared queue accessible, size: {self.shared_signal_queue.qsize()}"
                }
                
            elif self.signal_retrieval_method == 'file_watcher':
                return {
                    'success': self.signal_file_path.exists(),
                    'message': f"Signal file {'accessible' if self.signal_file_path.exists() else 'not found'}"
                }
                
            elif self.signal_retrieval_method == 'database_watcher':
                with sqlite3.connect(self.signal_db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT COUNT(*) FROM {self.signal_table_name}")
                    count = cursor.fetchone()[0]
                    return {
                        'success': True,
                        'message': f"Database accessible, {count} total signals"
                    }
            else:
                return {
                    'success': False,
                    'message': f"Unknown integration method: {self.signal_retrieval_method}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"Connection test failed: {str(e)}"
            }

    def _start_signal_monitoring(self):
        """Start signal monitoring and health checks"""
        
        # This could include additional monitoring threads, health checks, etc.
        logger.info("[CHART] Signal monitoring started")

    async def _send_setup_notification(self, integration_method: str, connection_test: Dict):
        """Send setup notification via Telegram"""
        
        try:
            if self.telegram_bot:
                message = f"🔗 <b>Signal Integration Setup</b>\n\n"
                message += f"[SIGNAL] Method: {integration_method}\n"
                message += f"[OK] Status: {connection_test['message']}\n"
                message += f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}"
                
                await self.telegram_bot.send_message(message, parse_mode='HTML')
                
        except Exception as e:
            logger.debug(f"Failed to send setup notification: {e}")

    def stop_signal_interception(self):
        """Stop signal interception"""
        
        try:
            logger.info("🛑 Stopping signal interception...")
            
            # Set stop event
            if hasattr(self, 'stop_signal_listener'):
                self.stop_signal_listener.set()
            
            # Wait for thread to finish
            if self.signal_thread and self.signal_thread.is_alive():
                self.signal_thread.join(timeout=5.0)
                if self.signal_thread.is_alive():
                    logger.warning("[WARNING] Signal thread did not stop gracefully")
            
            # Cleanup resources
            self._cleanup_signal_resources()
            
            logger.info("[OK] Signal interception stopped")
            
        except Exception as e:
            logger.error(f"Error stopping signal interception: {e}")

    def _cleanup_signal_resources(self):
        """Cleanup signal interception resources"""
        
        try:
            # Clear queues
            if hasattr(self, 'signal_buffer'):
                while not self.signal_buffer.empty():
                    try:
                        self.signal_buffer.get_nowait()
                    except:
                        break
            
            # Reset state
            self.signal_listener_active = False
            self.main_bot = None
            
            logger.debug("🧹 Signal resources cleaned up")
            
        except Exception as e:
            logger.debug(f"Error cleaning up signal resources: {e}")

    def get_signal_interception_status(self) -> Dict:
        """Get comprehensive signal interception status"""
        
        try:
            return {
                'integration_method': getattr(self, 'signal_retrieval_method', 'None'),
                'listener_active': getattr(self, 'signal_listener_active', False),
                'thread_alive': self.signal_thread.is_alive() if self.signal_thread else False,
                'main_bot_connected': self.main_bot is not None,
                'statistics': self.signal_stats,
                'configuration': {
                    'method': self.integration_method,
                    'config': getattr(self, 'integration_config', {})
                },
                'last_check': self.last_signal_check.isoformat() if hasattr(self, 'last_signal_check') else None
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def receive_signal_from_main_bot(self, signal_data: Dict) -> Dict:
        """
        Enhanced method to receive signal from main bot with validation and error handling
        
        Args:
            signal_data: Dictionary containing signal information
            
        Returns:
            Dict: Status of signal reception and processing
        """
        
        reception_result = {
            'received': False,
            'processed': False,
            'signal_id': None,
            'error': None,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # 1. Validate incoming signal data
            validation_result = self._validate_signal_data(signal_data)
            if not validation_result['valid']:
                reception_result['error'] = f"Signal validation failed: {validation_result['errors']}"
                logger.warning(f"[ERROR] Invalid signal data: {validation_result['errors']}")
                return reception_result
            
            # 2. Check if bot can receive signals
            if not self._can_receive_signals():
                reception_result['error'] = f"Bot cannot receive signals: {self._get_reception_block_reason()}"
                logger.info(f"⏸️ Signal reception blocked: {reception_result['error']}")
                return reception_result
            
            # 3. Create enhanced TradingSignal object
            signal = self._create_enhanced_trading_signal(signal_data)
            reception_result['signal_id'] = f"{signal.ticker}_{signal.timestamp.timestamp()}"
            
            # 4. Check for duplicate signals
            if self._is_duplicate_signal(signal):
                reception_result['error'] = "Duplicate signal detected within time window"
                logger.info(f"🔄 Duplicate signal filtered: {signal.ticker} {signal.direction}")
                return reception_result
            
            # 5. Add signal to queue with priority handling
            queue_result = await self._add_signal_to_queue(signal)
            if not queue_result['success']:
                reception_result['error'] = f"Queue error: {queue_result['error']}"
                logger.error(f"[ERROR] Failed to queue signal: {queue_result['error']}")
                return reception_result
            
            # 6. Update statistics
            self._update_signal_statistics(signal)
            
            # 7. Log successful reception
            logger.info(f"[SIGNAL] Signal received: {signal.ticker} {signal.direction} "
                    f"({signal.confidence:.1%}) from {signal.source}")
            
            # 8. Send acknowledgment notification if configured
            if hasattr(self, 'telegram_bot') and self.telegram_bot:
                await self._send_signal_received_notification(signal)
            
            reception_result['received'] = True
            reception_result['processed'] = True
            
            return reception_result
            
        except Exception as e:
            error_msg = f"Error receiving signal from main bot: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            reception_result['error'] = error_msg
            
            # Try to save failed signal for debugging
            try:
                self._save_failed_signal(signal_data, str(e))
            except:
                pass
            
            return reception_result

    def _validate_signal_data(self, signal_data: Dict) -> Dict:
        """Validate incoming signal data structure and content"""
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Required fields
        required_fields = ['ticker', 'direction', 'confidence', 'strategy', 'current_price']
        
        for field in required_fields:
            if field not in signal_data:
                validation_result['errors'].append(f"Missing required field: {field}")
                validation_result['valid'] = False
            elif signal_data[field] is None:
                validation_result['errors'].append(f"Field {field} cannot be None")
                validation_result['valid'] = False
        
        if not validation_result['valid']:
            return validation_result
        
        # Field-specific validations
        ticker = signal_data.get('ticker', '').upper()
        if not ticker or len(ticker) < 2:
            validation_result['errors'].append("Invalid ticker symbol")
            validation_result['valid'] = False
        
        direction = signal_data.get('direction', '').lower()
        if direction not in ['bullish', 'bearish', 'neutral']:
            validation_result['errors'].append(f"Invalid direction: {direction}")
            validation_result['valid'] = False
        
        confidence = signal_data.get('confidence', 0)
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            validation_result['errors'].append(f"Invalid confidence: {confidence} (must be 0.0-1.0)")
            validation_result['valid'] = False
        
        current_price = signal_data.get('current_price', 0)
        if not isinstance(current_price, (int, float)) or current_price <= 0:
            validation_result['errors'].append(f"Invalid current_price: {current_price}")
            validation_result['valid'] = False
        
        # Warnings for missing optional fields
        optional_fields = ['market_data', 'source', 'technical_analysis']
        for field in optional_fields:
            if field not in signal_data:
                validation_result['warnings'].append(f"Optional field missing: {field}")
        
        # Log warnings
        if validation_result['warnings']:
            logger.debug(f"Signal validation warnings: {validation_result['warnings']}")
        
        return validation_result

    def _can_receive_signals(self) -> bool:
        """Check if bot is in state to receive signals"""
        
        # Check if monitoring is active
        if not getattr(self, 'monitoring_active', False):
            return False
        
        # Check if in emergency state
        if getattr(self, 'emergency_stop_triggered', False):
            return False
        
        # Check signal queue capacity
        if hasattr(self, 'signal_queue'):
            try:
                queue_size = self.signal_queue.qsize()
                if queue_size >= 10:  # Queue getting full
                    return False
            except:
                pass
        
        # Check market hours (if configured)
        if hasattr(self, 'bot_config') and self.bot_config:
            if not self.bot_config.is_market_hours():
                return False
        
        return True

    def _get_reception_block_reason(self) -> str:
        """Get reason why signal reception is blocked"""
        
        if not getattr(self, 'monitoring_active', False):
            return "Bot monitoring not active"
        
        if getattr(self, 'emergency_stop_triggered', False):
            return "Emergency stop activated"
        
        if hasattr(self, 'signal_queue'):
            try:
                if self.signal_queue.qsize() >= 10:
                    return "Signal queue full"
            except:
                pass
        
        if hasattr(self, 'bot_config') and self.bot_config:
            if not self.bot_config.is_market_hours():
                return "Outside market hours"
        
        return "Unknown reason"

    def _create_enhanced_trading_signal(self, signal_data: Dict) -> TradingSignal:
        """Create enhanced TradingSignal object with additional processing"""
        
        # Extract and clean data
        ticker = signal_data.get('ticker', '').upper()
        direction = signal_data.get('direction', '').lower()
        confidence = float(signal_data.get('confidence', 0.0))
        strategy = signal_data.get('strategy', 'UNKNOWN')
        current_price = float(signal_data.get('current_price', 0.0))
        source = signal_data.get('source', 'main_bot')
        
        # Enhanced market data
        market_data = signal_data.get('market_data', {})
        if not market_data:
            market_data = {
                'timestamp': datetime.now().isoformat(),
                'source': source
            }
        
        # Add signal metadata
        market_data.update({
            'reception_time': datetime.now().isoformat(),
            'signal_source': source,
            'validation_passed': True
        })
        
        # Create signal with enhanced timestamp
        signal_timestamp = signal_data.get('timestamp')
        if signal_timestamp:
            if isinstance(signal_timestamp, str):
                try:
                    signal_timestamp = datetime.fromisoformat(signal_timestamp.replace('Z', '+00:00'))
                except:
                    signal_timestamp = datetime.now()
            elif not isinstance(signal_timestamp, datetime):
                signal_timestamp = datetime.now()
        else:
            signal_timestamp = datetime.now()
        
        return TradingSignal(
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            strategy=strategy,
            current_price=current_price,
            timestamp=signal_timestamp,
            source=source,
            market_data=market_data,
            technical_confirmation=signal_data.get('technical_confirmation', False)
        )

    def _is_duplicate_signal(self, signal: TradingSignal) -> bool:
        """Check if signal is duplicate within time window"""
        
        if not hasattr(self, 'recent_signals_cache'):
            self.recent_signals_cache = []
        
        current_time = datetime.now()
        duplicate_window = 300  # 5 minutes
        
        # Clean old signals from cache
        self.recent_signals_cache = [
            s for s in self.recent_signals_cache 
            if (current_time - s['timestamp']).total_seconds() < duplicate_window
        ]
        
        # Check for duplicates
        for recent_signal in self.recent_signals_cache:
            if (recent_signal['ticker'] == signal.ticker and
                recent_signal['direction'] == signal.direction and
                abs(recent_signal['confidence'] - signal.confidence) < 0.05):
                return True
        
        # Add current signal to cache
        self.recent_signals_cache.append({
            'ticker': signal.ticker,
            'direction': signal.direction,
            'confidence': signal.confidence,
            'timestamp': current_time
        })
        
        return False

    async def _add_signal_to_queue(self, signal: TradingSignal) -> Dict:
        """Add signal to queue with priority handling"""
        
        try:
            # Check queue capacity
            if hasattr(self, 'signal_queue'):
                current_size = self.signal_queue.qsize()
                max_size = 20  # Configure as needed
                
                if current_size >= max_size:
                    # Queue full - remove oldest signal if new one has higher confidence
                    try:
                        # This is a simplified approach - you might want more sophisticated priority logic
                        if signal.confidence > 0.8:  # High confidence signal
                            logger.warning(f"[WARNING] Queue full, making room for high confidence signal")
                            # In a real implementation, you'd remove the lowest confidence signal
                        else:
                            return {
                                'success': False,
                                'error': f'Signal queue full ({current_size}/{max_size})'
                            }
                    except:
                        return {
                            'success': False,
                            'error': 'Queue full and unable to make room'
                        }
            
            # Add signal to queue
            await self.signal_queue.put(signal)
            
            return {
                'success': True,
                'queue_size': self.signal_queue.qsize()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _update_signal_statistics(self, signal: TradingSignal):
        """Update signal reception statistics"""
        
        # Update counters
        self.signals_received_today += 1
        self.last_signal_time = datetime.now()
        
        # Update signal source statistics
        if not hasattr(self, 'signal_source_stats'):
            self.signal_source_stats = {}
        
        source = signal.source
        if source not in self.signal_source_stats:
            self.signal_source_stats[source] = {
                'count': 0,
                'last_received': None,
                'avg_confidence': 0.0,
                'directions': {'bullish': 0, 'bearish': 0, 'neutral': 0}
            }
        
        stats = self.signal_source_stats[source]
        stats['count'] += 1
        stats['last_received'] = datetime.now()
        stats['avg_confidence'] = (stats['avg_confidence'] * (stats['count'] - 1) + signal.confidence) / stats['count']
        stats['directions'][signal.direction] = stats['directions'].get(signal.direction, 0) + 1

    async def _send_signal_received_notification(self, signal: TradingSignal):
        """Send notification about received signal (optional)"""
        
        try:
            # Only notify for high confidence signals to avoid spam
            if signal.confidence >= 0.8 and self.telegram_bot:
                message = f"[SIGNAL] <b>Signal Received</b>\n\n"
                message += f"[CHART] {signal.ticker} {signal.direction.upper()}\n"
                message += f"[TARGET] Confidence: {signal.confidence:.1%}\n"
                message += f"[MONEY] Price: ₹{signal.current_price:.2f}\n"
                message += f"⚡ Source: {signal.source}\n"
                message += f"⏰ Time: {signal.timestamp.strftime('%H:%M:%S')}"
                
                await self.telegram_bot.send_message(message, parse_mode='HTML')
                
        except Exception as e:
            logger.debug(f"Failed to send signal notification: {e}")

    def _save_failed_signal(self, signal_data: Dict, error: str):
        """Save failed signal for debugging"""
        
        try:
            failed_signal = {
                'timestamp': datetime.now().isoformat(),
                'signal_data': signal_data,
                'error': error,
                'bot_state': getattr(self, 'state', 'unknown').value if hasattr(getattr(self, 'state', None), 'value') else str(getattr(self, 'state', 'unknown'))
            }
            
            # Save to file for debugging
            failed_signals_file = 'logs/failed_signals.json'
            os.makedirs('logs', exist_ok=True)
            
            failed_signals = []
            if os.path.exists(failed_signals_file):
                try:
                    with open(failed_signals_file, 'r') as f:
                        failed_signals = json.load(f)
                except:
                    failed_signals = []
            
            failed_signals.append(failed_signal)
            
            # Keep only last 100 failed signals
            failed_signals = failed_signals[-100:]
            
            with open(failed_signals_file, 'w') as f:
                json.dump(failed_signals, f, indent=2, default=str)
                
            logger.debug(f"Failed signal saved to {failed_signals_file}")
            
        except Exception as e:
            logger.debug(f"Failed to save failed signal: {e}")

    def get_signal_reception_stats(self) -> Dict:
        """Get comprehensive signal reception statistics"""
        
        try:
            return {
                'total_signals_received': self.signals_received_today,
                'total_signals_processed': self.signals_processed_today,
                'signal_conversion_rate': (self.signals_processed_today / max(1, self.signals_received_today)) * 100,
                'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
                'source_statistics': getattr(self, 'signal_source_stats', {}),
                'queue_status': {
                    'current_size': self.signal_queue.qsize() if hasattr(self, 'signal_queue') else 0,
                    'queue_available': self._can_receive_signals(),
                    'block_reason': self._get_reception_block_reason() if not self._can_receive_signals() else None
                },
                'session_start': self.session_start_time.isoformat(),
                'monitoring_active': getattr(self, 'monitoring_active', False)
            }
            
        except Exception as e:
            logger.error(f"Error getting signal reception stats: {e}")
            return {'error': str(e)}
    
    # ================== INTEGRATION HELPER METHODS ==================
    
    async def _get_current_positions(self) -> Dict:
        """Get current positions for risk calculation"""
        try:
            positions = self.zerodha.get_positions()
            return positions if positions else {'net': [], 'day': []}
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {'net': [], 'day': []}
    
    async def _execute_option_order(self, leg: OptionsLeg, dry_run: bool = False) -> Dict:
        """Execute option order for a leg"""
        
        try:
            if dry_run:
                logger.info(f"🧪 DRY RUN: Would execute {leg.action} {leg.contracts} lots of {leg.tradingsymbol}")
                return {
                    'status': 'success',
                    'order_id': f'DRY_RUN_{datetime.now().timestamp()}',
                    'message': 'Dry run execution'
                }
            
            # Real order execution
            order_params = {
                'tradingsymbol': leg.tradingsymbol,
                'exchange': leg.exchange,
                'transaction_type': leg.action,
                'quantity': leg.contracts * leg.lot_size,
                'product': 'MIS',  # Intraday
                'order_type': 'MARKET',
                'validity': 'DAY'
            }
            
            order_id = self.zerodha.place_order(**order_params)
            
            if order_id:
                return {
                    'status': 'success',
                    'order_id': order_id,
                    'message': 'Order placed successfully'
                }
            else:
                return {
                    'status': 'failed',
                    'message': 'Order placement failed'
                }
                
        except Exception as e:
            logger.error(f"Error executing option order: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def _square_off_option_position(self, tradingsymbol: str, quantity: int) -> Dict:
        """Square off option position"""
        
        try:
            # Determine transaction type for square off
            transaction_type = 'SELL' if quantity > 0 else 'BUY'
            square_off_quantity = abs(quantity)
            
            order_params = {
                'tradingsymbol': tradingsymbol,
                'exchange': 'NFO',
                'transaction_type': transaction_type,
                'quantity': square_off_quantity,
                'product': 'MIS',
                'order_type': 'MARKET',
                'validity': 'DAY'
            }
            
            order_id = self.zerodha.place_order(**order_params)
            
            if order_id:
                return {
                    'status': 'success',
                    'order_id': order_id,
                    'message': f'Square off order placed: {tradingsymbol}'
                }
            else:
                return {
                    'status': 'failed',
                    'message': 'Square off order failed'
                }
                
        except Exception as e:
            logger.error(f"Error in square off: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    # ================== TESTING AND DEBUG METHODS ==================
    
    async def run_system_diagnostics(self) -> Dict:
        """Run comprehensive system diagnostics"""
        
        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': True,
            'component_status': {},
            'api_status': {},
            'performance_metrics': {},
            'recommendations': []
        }
        
        try:
            # Test core components
            diagnostics['component_status'] = {
                'zerodha_client': bool(self.zerodha),
                'market_data_provider': bool(self.market_data_provider),
                'options_analyzer': bool(self.options_analyzer),
                'order_manager': bool(self.order_manager),
                'technical_analyzer': bool(self.technical_analyzer),
                'telegram_bot': bool(self.telegram_bot)
            }
            
            # Test v2.0 components
            if hasattr(self, 'bot_config'):
                diagnostics['component_status'].update({
                    'capital_manager': bool(getattr(self, 'capital_manager', None)),
                    'risk_manager': bool(getattr(self, 'intelligent_risk_manager', None)),
                    'profit_optimizer': bool(getattr(self, 'profit_optimizer', None)),
                    'learning_system': bool(getattr(self, 'learning_system', None)),
                    'position_monitor': bool(getattr(self, 'position_monitor', None))
                })
            
            # Test API connections
            try:
                profile = self.zerodha.get_profile()
                diagnostics['api_status']['zerodha'] = bool(profile)
            except:
                diagnostics['api_status']['zerodha'] = False
            
            try:
                test_data = await self.market_data_provider.fetch_live_data('NIFTY')
                diagnostics['api_status']['market_data'] = bool(test_data)
            except:
                diagnostics['api_status']['market_data'] = False
            
            # Performance metrics
            diagnostics['performance_metrics'] = {
                'session_duration': str(datetime.now() - self.session_start_time).split('.')[0],
                'signals_received': self.signals_received_today,
                'signals_processed': self.signals_processed_today,
                'active_position': bool(self.active_position),
                'daily_pnl': self.daily_performance.total_pnl,
                'monitoring_active': self.monitoring_active
            }
            
            # Check overall health
            all_core_components = all(diagnostics['component_status'].values())
            all_apis_working = all(diagnostics['api_status'].values())
            
            diagnostics['overall_health'] = all_core_components and all_apis_working
            
            # Generate recommendations
            if not all_core_components:
                diagnostics['recommendations'].append("Some components failed to initialize")
            
            if not all_apis_working:
                diagnostics['recommendations'].append("API connection issues detected")
            
            if self.signals_received_today == 0:
                diagnostics['recommendations'].append("No signals received today - check signal source")
            
            if diagnostics['overall_health']:
                diagnostics['recommendations'].append("All systems operational")
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Error running diagnostics: {e}")
            diagnostics['overall_health'] = False
            diagnostics['error'] = str(e)
            return diagnostics
    
    async def simulate_trading_session(self, duration_minutes: int = 60) -> Dict:
        """Simulate a trading session for testing"""
        
        logger.info(f"🎭 Starting trading simulation for {duration_minutes} minutes")
        
        simulation_results = {
            'start_time': datetime.now().isoformat(),
            'duration_minutes': duration_minutes,
            'signals_injected': 0,
            'trades_executed': 0,
            'final_pnl': 0.0,
            'events': []
        }
        
        try:
            # Start automated trading in test mode
            trading_task = asyncio.create_task(
                self.start_automated_trading_v2(test_mode=True)
            )
            
            # Wait for initialization
            await asyncio.sleep(5)
            
            # Inject test signals periodically
            test_signals = [
                ('NIFTY', 'bullish', 0.75),
                ('RELIANCE', 'bearish', 0.80),
                ('BAJFINANCE', 'bullish', 0.70),
                ('NIFTY', 'bearish', 0.85),
            ]
            
            signal_interval = duration_minutes / len(test_signals)
            
            for i, (ticker, direction, confidence) in enumerate(test_signals):
                # Wait before next signal
                if i > 0:
                    await asyncio.sleep(signal_interval * 60)
                
                # Inject signal
                success = await self.inject_test_signal(ticker, direction, confidence)
                if success:
                    simulation_results['signals_injected'] += 1
                    simulation_results['events'].append({
                        'time': datetime.now().isoformat(),
                        'event': 'signal_injected',
                        'ticker': ticker,
                        'direction': direction,
                        'confidence': confidence
                    })
                
                # Check if we should break early
                if not self.monitoring_active:
                    break
            
            # Wait for simulation to complete
            remaining_time = duration_minutes * 60 - (len(test_signals) * signal_interval * 60)
            if remaining_time > 0:
                await asyncio.sleep(remaining_time)
            
            # Get final results
            final_status = self.get_current_status()
            simulation_results['trades_executed'] = final_status.get('trades_today', 0)
            simulation_results['final_pnl'] = final_status.get('daily_pnl', 0.0)
            
            # Stop trading
            trading_task.cancel()
            
            simulation_results['end_time'] = datetime.now().isoformat()
            simulation_results['success'] = True
            
            logger.info(f"🎭 Simulation completed: {simulation_results['trades_executed']} trades, ₹{simulation_results['final_pnl']:+.0f} P&L")
            
            return simulation_results
            
        except Exception as e:
            logger.error(f"Error in trading simulation: {e}")
            simulation_results['success'] = False
            simulation_results['error'] = str(e)
            return simulation_results
    
    # ================== PERFORMANCE MONITORING ==================
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        
        try:
            # Calculate metrics
            total_trades = self.daily_performance.trades_profitable + self.daily_performance.trades_loss
            win_rate = (self.daily_performance.trades_profitable / max(1, total_trades)) * 100
            
            roi_percent = 0
            if self.daily_performance.starting_capital > 0:
                roi_percent = (self.daily_performance.total_pnl / self.daily_performance.starting_capital) * 100
            
            # Performance summary
            summary = {
                'date': self.daily_performance.date,
                'session_duration': str(datetime.now() - self.session_start_time).split('.')[0],
                
                # Capital metrics
                'starting_capital': self.daily_performance.starting_capital,
                'current_capital': self.daily_performance.current_capital,
                'total_pnl': self.daily_performance.total_pnl,
                'realized_pnl': self.daily_performance.realized_pnl,
                'unrealized_pnl': self.daily_performance.unrealized_pnl,
                'roi_percent': roi_percent,
                
                # Trading metrics
                'total_trades': total_trades,
                'profitable_trades': self.daily_performance.trades_profitable,
                'loss_trades': self.daily_performance.trades_loss,
                'win_rate_percent': win_rate,
                
                # Risk metrics
                'max_drawdown': self.daily_performance.max_drawdown,
                'current_drawdown': self.daily_performance.current_drawdown,
                'consecutive_losses': self.consecutive_losses,
                
                # Signal metrics
                'signals_received': self.signals_received_today,
                'signals_processed': self.signals_processed_today,
                'signal_conversion_rate': (self.signals_processed_today / max(1, self.signals_received_today)) * 100,
                
                # Current status
                'active_position': bool(self.active_position),
                'monitoring_active': self.monitoring_active,
                'emergency_stop': self.emergency_stop_triggered,
                'recovery_mode': self.recovery_mode,
                
                # Performance rating
                'performance_rating': self._calculate_performance_rating(roi_percent, win_rate),
                
                # Recommendations
                'recommendations': self._generate_performance_recommendations(roi_percent, win_rate, total_trades)
            }
            
            # Add v2.0 specific metrics if available
            if hasattr(self, 'learning_system'):
                learning_insights = self.learning_system.get_learning_insights()
                summary['learning_metrics'] = {
                    'trades_analyzed': learning_insights.get('total_trades_analyzed', 0),
                    'best_strategy': str(learning_insights.get('best_strategy', 'None')),
                    'pattern_success_rates': len(learning_insights.get('pattern_success_rates', {})),
                    'recommendations_count': len(learning_insights.get('recommendations', []))
                }
            
            if hasattr(self, 'capital_manager'):
                summary['capital_tier'] = {
                    'current_tier': str(self.capital_manager.current_tier),
                    'tier_name': self.capital_manager._get_tier_name(),
                    'allowed_strategies': len(self.capital_manager.get_allowed_strategies())
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_performance_rating(self, roi_percent: float, win_rate: float) -> str:
        """Calculate performance rating"""
        
        if roi_percent >= 15 and win_rate >= 70:
            return "🏆 EXCELLENT"
        elif roi_percent >= 10 and win_rate >= 60:
            return "[TARGET] VERY GOOD"
        elif roi_percent >= 5 and win_rate >= 50:
            return "[OK] GOOD"
        elif roi_percent >= 0 and win_rate >= 40:
            return "[CHART] AVERAGE"
        elif roi_percent >= -5:
            return "[WARNING] BELOW AVERAGE"
        else:
            return "🚨 POOR"
    
    def _generate_performance_recommendations(self, roi_percent: float, win_rate: float, total_trades: int) -> List[str]:
        """Generate performance-based recommendations"""
        
        recommendations = []
        
        # ROI-based recommendations
        if roi_percent >= 15:
            recommendations.append("🎉 Excellent returns! Consider taking some profits off the table")
        elif roi_percent >= 8:
            recommendations.append("[OK] Good performance, maintain current strategy")
        elif roi_percent >= 0:
            recommendations.append("[CHART] Positive returns, room for improvement")
        else:
            recommendations.append("🚨 Negative returns, review strategy and risk management")
        
        # Win rate recommendations
        if win_rate >= 70:
            recommendations.append("[TARGET] Excellent win rate, consider scaling position sizes")
        elif win_rate >= 50:
            recommendations.append("[OK] Good win rate, maintain current approach")
        elif win_rate >= 40:
            recommendations.append("[WARNING] Below average win rate, review entry criteria")
        else:
            recommendations.append("🚨 Poor win rate, major strategy review needed")
        
        # Trade frequency recommendations
        if total_trades < 3:
            recommendations.append("[UP] Low trade frequency, consider more opportunities")
        elif total_trades > 10:
            recommendations.append("[WARNING] High trade frequency, ensure quality over quantity")
        
        # Risk-based recommendations
        if self.consecutive_losses >= 3:
            recommendations.append("🛑 Multiple consecutive losses, consider reducing position sizes")
        
        if abs(self.daily_performance.max_drawdown) > 1000:
            recommendations.append("[DOWN] Significant drawdown detected, review risk management")
        
        return recommendations

# ================== AUTOMATION COMMAND INTERFACE ==================

class AutomationCommandInterface:
   """Command interface for automation bot control"""
   
   def __init__(self, bot_instance):
       self.bot = bot_instance
       
   async def inject_test_signal(self, ticker: str, direction: str, confidence: float = 0.75) -> Dict:
       """Inject test signal for development/testing"""
       
       try:
           success = await self.bot.inject_test_signal(ticker, direction, confidence)
           
           return {
               'success': success,
               'message': f"Test signal {'injected' if success else 'failed'} for {ticker}",
               'ticker': ticker,
               'direction': direction,
               'confidence': confidence
           }
           
       except Exception as e:
           return {
               'success': False,
               'message': f"Test signal injection failed: {str(e)}",
               'error': str(e)
           }
   
   async def manual_exit_position(self, reason: str = "manual_override") -> Dict:
       """Manually exit current position"""
       
       try:
           if self.bot.active_position:
               result = await self.bot.manual_exit(reason)
               
               return {
                   'success': result,
                   'message': f"Position {'exited' if result else 'exit failed'}",
                   'reason': reason,
                   'position_ticker': self.bot.active_position.signal.ticker if self.bot.active_position else None
               }
           else:
               return {
                   'success': False,
                   'message': "No active position to exit"
               }
               
       except Exception as e:
           return {
               'success': False,
               'message': f"Manual exit failed: {str(e)}",
               'error': str(e)
           }
   
   async def get_enhanced_status(self) -> Dict:
       """Get comprehensive bot status"""
       
       try:
           return await self.bot.get_enhanced_status()
       except Exception as e:
           return {
               'error': str(e),
               'basic_status': self.bot.get_current_status() if hasattr(self.bot, 'get_current_status') else {},
               'timestamp': datetime.now().isoformat()
           }
   
   async def update_configuration(self, config_updates: Dict) -> Dict:
       """Update bot configuration dynamically"""
       
       try:
           if hasattr(self.bot, 'bot_config'):
               self.bot.bot_config.update(config_updates)
               
               return {
                   'success': True,
                   'message': f"Configuration updated: {list(config_updates.keys())}",
                   'updated_keys': list(config_updates.keys())
               }
           else:
               return {
                   'success': False,
                   'message': "Bot configuration not available"
               }
               
       except Exception as e:
           return {
               'success': False,
               'message': f"Configuration update failed: {str(e)}",
               'error': str(e)
           }
   
   async def force_capital_update(self) -> Dict:
       """Force update of capital status and tier"""
       
       try:
           if hasattr(self.bot, 'capital_manager'):
               capital_status = await self.bot.capital_manager.update_capital_status()
               
               return {
                   'success': True,
                   'message': "Capital status updated",
                   'capital_status': capital_status
               }
           else:
               return {
                   'success': False,
                   'message': "Capital manager not available"
               }
               
       except Exception as e:
           return {
               'success': False,
               'message': f"Capital update failed: {str(e)}",
               'error': str(e)
           }
   
   async def run_test_suite(self) -> Dict:
       """Run comprehensive test suite"""
       
       try:
           if hasattr(self.bot, 'run_enhanced_test_suite'):
               return await self.bot.run_enhanced_test_suite()
           else:
               return {
                   'success': False,
                   'message': "Enhanced test suite not available"
               }
               
       except Exception as e:
           return {
               'success': False,
               'message': f"Test suite failed: {str(e)}",
               'error': str(e)
           }

# ================== ENHANCED MAIN EXECUTION ==================

async def main_v2():
   """Enhanced main function for v2.0 automated bot"""
   
   print("[ROCKET] Automated Indian Intraday Options Bot v2.0 - ENHANCED")
   print("=" * 70)
   print("🔥 Features: Dynamic Capital Management | Intelligent Risk | Profit Optimization")
   print("🧠 Learning System | Multi-Leg Strategies | Real-time Monitoring")
   print("=" * 70)
   
   try:
       # Initialize Zerodha client
       zerodha_client = ZerodhaAPIClient()
       
       if not zerodha_client.access_token:
           print("[ERROR] Zerodha access token not found!")
           print("Please ensure ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN are set")
           return
       
       # Initialize enhanced automated bot
       bot = AutomatedIntradayOptionsBot(zerodha_client)
       
       # Initialize command interface
       command_interface = AutomationCommandInterface(bot)
       
       # Check command line arguments
       if len(sys.argv) > 1:
           command = sys.argv[1].lower()
           
           if command == 'test':
               print("\n🧪 Running in Test Mode...")
               
               # Start bot in background
               bot_task = asyncio.create_task(bot.start_automated_trading_v2(test_mode=True))
               
               # Wait for initialization
               await asyncio.sleep(5)
               
               # Run test suite
               print("🔬 Running enhanced test suite...")
               test_results = await command_interface.run_test_suite()
               
               if test_results.get('overall_success'):
                   print("[OK] All tests passed!")
               else:
                   print("[WARNING] Some tests failed - check results")
               
               # Inject test signals
               test_signals = [
                   ('NIFTY', 'bullish', 0.75),
                   ('RELIANCE', 'bearish', 0.80),
                   ('BAJFINANCE', 'bullish', 0.85),
               ]
               
               for ticker, direction, confidence in test_signals:
                   print(f"🧪 Testing signal: {ticker} {direction} ({confidence:.0%})")
                   result = await command_interface.inject_test_signal(ticker, direction, confidence)
                   
                   if result['success']:
                       print(f"[OK] {result['message']}")
                       await asyncio.sleep(45)  # Wait between signals
                   else:
                       print(f"[ERROR] {result['message']}")
               
               # Let it run for testing period
               print("⏳ Running test session for 10 minutes...")
               await asyncio.sleep(600)  # 10 minutes
               
               # Get final status
               final_status = await command_interface.get_enhanced_status()
               print("\n[CHART] Final Test Status:")
               print(f"Signals Processed: {final_status.get('signals_sent_today', 0)}")
               print(f"Current State: {final_status.get('state', 'unknown')}")
               
               # Manual exit if position exists
               if final_status.get('active_position'):
                   exit_result = await command_interface.manual_exit_position("test_completion")
                   print(f"🚪 {exit_result['message']}")
               
               # Stop the bot
               bot_task.cancel()
               
           elif command == 'status':
               print("\n[CHART] Getting Enhanced Status...")
               
               # Initialize bot components only
               bot.initialize_v2_components()
               
               # Get status
               status = await command_interface.get_enhanced_status()
               
               print("\n" + "="*50)
               print("[CHART] ENHANCED BOT STATUS")
               print("="*50)
               print(f"State: {status.get('state', 'Unknown')}")
               print(f"Capital Tier: {status.get('v2_features', {}).get('capital_management', {}).get('tier_name', 'Unknown')}")
               print(f"Available Capital: ₹{status.get('v2_features', {}).get('capital_management', {}).get('available_capital', 0):,.0f}")
               print(f"Active Position: {status.get('active_position', False)}")
               print(f"Daily P&L: ₹{status.get('daily_pnl', 0):+,.0f}")
               print(f"System Health: {'[OK] All Components Active' if status.get('system_health', {}).get('all_components_active') else '[WARNING] Issues Detected'}")
               
               # Print v2.0 specific features
               v2_features = status.get('v2_features', {})
               if v2_features:
                   print(f"\n🔥 v2.0 FEATURES STATUS:")
                   print(f"Risk Tier: {v2_features.get('risk_tier', {}).get('current_tier', 1)}")
                   print(f"Position Monitoring: {'Active' if v2_features.get('position_monitoring', {}).get('monitoring_active') else 'Inactive'}")
                   print(f"Learning System: {v2_features.get('performance_analytics', {}).get('learning_insights', {}).get('total_trades_analyzed', 0)} trades analyzed")
               
           elif command == 'manual':
               if len(sys.argv) < 3:
                   print("[ERROR] Usage: python bot.py manual <ticker>")
                   return
               
               ticker = sys.argv[2].upper()
               print(f"\n🔍 Manual Analysis Mode for {ticker}")
               
               # Initialize bot components
               bot.initialize_v2_components()
               
               # Run enhanced manual analysis
               result = await bot.manual_analysis(ticker)
               
               if result:
                   print(f"[OK] Enhanced analysis complete for {ticker}")
               else:
                   print(f"[ERROR] Enhanced analysis failed for {ticker}")
           
           elif command == 'config':
               print("\n⚙️ Configuration Management Mode")
               
               if len(sys.argv) < 4:
                   print("Usage: python bot.py config <key> <value>")
                   print("Example: python bot.py config min_confidence 0.70")
                   return
               
               key = sys.argv[2]
               value = float(sys.argv[3]) if '.' in sys.argv[3] else int(sys.argv[3]) if sys.argv[3].isdigit() else sys.argv[3]
               
               # Initialize bot components
               bot.initialize_v2_components()
               
               # Update configuration
               result = await command_interface.update_configuration({key: value})
               print(f"⚙️ {result['message']}")
           
           else:
               print(f"[ERROR] Unknown command: {command}")
               print("Available commands: test, status, manual <ticker>, config <key> <value>")
       
       else:
           # Normal production mode
           print("\n[ROCKET] Production Mode - Starting Enhanced Automated Trading...")
           
           # Test API connection
           market_status = zerodha_client.get_market_status()
           if market_status:
               print(f"[OK] Zerodha API connected - Market: {market_status.get('status', 'Unknown')}")
           else:
               print("[WARNING] Zerodha API connection issue detected")
           
           print("[WRENCH] Initializing enhanced v2.0 components...")
           
           # Start enhanced automated trading
           await bot.start_automated_trading_v2()
   
   except KeyboardInterrupt:
       print("\n🛑 Shutdown requested by user")
   except Exception as e:
       print(f"\n[ERROR] Fatal error: {e}")
       logger.error("Enhanced bot failed to start", exc_info=True)
       
       # Enhanced error help
       if "access_token" in str(e).lower():
           print("\n[BULB] Troubleshooting:")
           print("1. Check if your access token is valid and not expired")
           print("2. Run 'python get_access_token.py' to generate a new token")
           print("3. Ensure ZERODHA_ACCESS_TOKEN is set in your .env file")
       elif "capital" in str(e).lower():
           print("\n[BULB] Capital Management Issue:")
           print("1. Ensure sufficient account balance for trading")
           print("2. Check margin requirements for selected strategies")
           print("3. Verify account permissions for options trading")
       
       sys.exit(1)

# ================== ENHANCED UTILITY FUNCTIONS ==================

def setup_enhanced_logging():
   """Setup enhanced logging for v2.0"""
   
   # Create logs directory if it doesn't exist
   os.makedirs('logs', exist_ok=True)
   
   # Configure enhanced logging
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('logs/automated_options_bot_v2.log'),
           logging.FileHandler('logs/latest.log'),
           logging.StreamHandler(sys.stdout)
       ]
   )
   
   # Add specific loggers for different components
   loggers = {
       'capital_manager': logging.getLogger('capital_manager'),
       'risk_manager': logging.getLogger('risk_manager'),
       'profit_optimizer': logging.getLogger('profit_optimizer'),
       'learning_system': logging.getLogger('learning_system'),
       'position_monitor': logging.getLogger('position_monitor')
   }
   
   # Set specific log levels if needed
   for name, logger_instance in loggers.items():
       logger_instance.setLevel(logging.INFO)

def print_startup_banner():
   """Print enhanced startup banner"""
   
   banner = """
╔══════════════════════════════════════════════════════════════╗
║          [ROCKET] AUTOMATED OPTIONS BOT v2.0 ENHANCED [ROCKET]          ║
║                     ⚡ NEXT GENERATION ⚡                    ║
╠══════════════════════════════════════════════════════════════╣
║  🧠 INTELLIGENCE: Dynamic Learning & Pattern Recognition    ║
║  [MONEY] CAPITAL MGMT: 3-Tier Adaptive Strategy Selection        ║
║  [TARGET] RISK CONTROL: Intelligent Multi-Level Protection        ║
║  [UP] OPTIMIZATION: Real-time Profit Enhancement              ║
║  🔄 MONITORING: Continuous Performance Analytics            ║
║  🎨 STRATEGIES: Single → Multi-leg → Advanced Arsenal       ║
╠══════════════════════════════════════════════════════════════╣
║  [GEM] TIER 1: <₹30K  → Basic Directional Strategies          ║
║  [GEM] TIER 2: ₹30K-₹1L → Multi-leg Buying Strategies         ║
║  [GEM] TIER 3: >₹1L → Full Advanced Strategy Arsenal           ║
╠══════════════════════════════════════════════════════════════╣
║  [TARGET] TARGET: 8-15% Daily Returns with Capital Protection     ║
║  🛡️ SAFETY: Emergency Protocols & Recovery Modes           ║
║  [CHART] LEARNING: Every Trade Improves Future Performance       ║
╚══════════════════════════════════════════════════════════════╝
   """
   
   print(banner)

if __name__ == "__main__":
   # Setup enhanced logging
   setup_enhanced_logging()
   
   # Print startup banner
   print_startup_banner()
   
   # Run enhanced main function
   asyncio.run(main_v2())
        
    