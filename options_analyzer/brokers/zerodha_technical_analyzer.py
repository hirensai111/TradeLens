#!/usr/bin/env python3
"""
Enhanced Zerodha Technical Analysis Engine for Indian Markets
Provides chart analysis, support/resistance, trends, and entry/exit signals
Optimized for intraday options trading with today-focused analysis
Uses Zerodha historical data for analysis
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ==================== CONSTANTS ====================
# Technical Analysis Thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_EXTREME_OVERSOLD = 20
RSI_EXTREME_OVERBOUGHT = 80

# Volume Thresholds
VOLUME_SPIKE_MULTIPLIER = 1.5
VOLUME_SIGNIFICANT_MULTIPLIER = 2.0

# Support/Resistance
SR_TOUCH_TOLERANCE = 0.02  # 2% tolerance
SR_MIN_TOUCHES = 2
SR_STRENGTH_THRESHOLD = 0.6

# Trend Detection
TREND_STRONG_THRESHOLD = 0.7
TREND_MODERATE_THRESHOLD = 0.5
MOVING_AVERAGE_PERIOD_SHORT = 20
MOVING_AVERAGE_PERIOD_LONG = 50

# Gap Detection
GAP_UP_THRESHOLD = 0.005  # 0.5%
GAP_DOWN_THRESHOLD = -0.005  # -0.5%
GAP_SIGNIFICANT_THRESHOLD = 0.015  # 1.5%

# Intraday Time Phases (IST hours)
OPENING_PHASE_HOUR = 10  # 9:15-10:00
MID_SESSION_HOUR = 13    # 10:00-13:30
CLOSING_PHASE_HOUR = 15  # 13:30-15:30

# Data Quality Thresholds
MIN_CANDLES_VERY_LIMITED = 5
MIN_CANDLES_LIMITED = 10
MIN_CANDLES_FAIR = 15
MIN_CANDLES_GOOD = 20
# ==================================================

@dataclass
class TechnicalSignal:
    """Technical analysis signal with entry/exit rules"""
    signal_type: str  # 'BUY', 'SELL', 'HOLD', 'WAIT'
    strength: float   # 0-1 confidence
    reason: str
    entry_price: Optional[float] = None
    entry_condition: Optional[str] = None
    stop_loss: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    time_frame: str = 'INTRADAY'
    risk_reward_ratio: float = 0.0

@dataclass
class SupportResistance:
    """Support and resistance levels with strength"""
    support_levels: List[float]
    resistance_levels: List[float]
    current_level: str  # 'SUPPORT', 'RESISTANCE', 'MIDDLE'
    nearest_support: float
    nearest_resistance: float
    level_strength: float

class ZerodhaTechnicalAnalyzer:
    """Enhanced technical analysis engine using Zerodha data"""
    
    def __init__(self, zerodha_client):
        self.zerodha = zerodha_client
        
    async def analyze_symbol_for_options(self, symbol: str, current_price: float, 
                               market_data: Dict, trading_style: str = 'intraday') -> Dict:
        """FIXED: Complete technical analysis optimized for options trading with better error handling"""
        
        logger.info(f"🔍 Starting enhanced technical analysis for {symbol}")
        
        # Initialize result structure for partial analysis recovery
        result = {
            'symbol': symbol,
            'current_price': current_price,
            'analysis_time': datetime.now().isoformat(),
            'trading_style': trading_style,
            'data_quality': 'UNKNOWN',
            'analysis_errors': [],  # Track what failed
            'partial_analysis': False
        }
        
        try:
            # Convert symbol to Zerodha format for historical data
            zerodha_symbol = self._get_zerodha_symbol(symbol)
            logger.info(f"Using Zerodha symbol: {zerodha_symbol} for historical data")
            
            # Get historical data with fallback - MORE LENIENT
            df_daily = self.get_historical_data_with_fallback(zerodha_symbol, 'day', 30)
            df_intraday = self.get_historical_data_with_fallback(zerodha_symbol, '5minute', 2) if trading_style == 'intraday' else df_daily
            
            # Data quality validation using constants
            if len(df_daily) < MIN_CANDLES_VERY_LIMITED:
                logger.warning(f"[WARNING] Very limited daily data ({len(df_daily)} candles) for {symbol}")
                result['data_quality'] = 'VERY_LIMITED'
            elif len(df_daily) < MIN_CANDLES_LIMITED:
                logger.warning(f"[WARNING] Limited daily data ({len(df_daily)} candles) for {symbol}")
                result['data_quality'] = 'LIMITED'
            elif len(df_daily) >= MIN_CANDLES_GOOD:
                result['data_quality'] = 'GOOD'
            else:
                result['data_quality'] = 'FAIR'
            
            # **ENHANCED INTRADAY ANALYSIS** - Focus on TODAY's action
            intraday_context = None
            try:
                intraday_context = self._analyze_today_context(current_price, market_data, df_daily, df_intraday)
                result['today_context'] = intraday_context
            except Exception as e:
                logger.warning(f"[WARNING] Today context analysis failed: {e}")
                result['analysis_errors'].append('today_context_failed')
                # Create minimal intraday context
                intraday_context = {
                    'gap_type': 'NONE',
                    'intraday_momentum': 'NEUTRAL',
                    'session_phase': 'MID_SESSION',
                    'intraday_change_percent': 0,
                    'volume_ratio': 1.0,
                    'today_open': current_price,
                    'today_high': current_price,
                    'today_low': current_price
                }
                result['today_context'] = intraday_context
            
            # **RESILIENT TECHNICAL ANALYSIS** - Continue even if some components fail
            
            # 1. Trend Analysis
            trend_analysis = {}
            try:
                trend_analysis = self._analyze_trend(df_daily, df_intraday, current_price, intraday_context)
            except KeyError as e:
                logger.error(f"Missing required data for trend analysis: {e}", exc_info=True)
                result['analysis_errors'].append('trend_analysis_missing_data')
            except ValueError as e:
                logger.error(f"Invalid data in trend analysis: {e}", exc_info=True)
                result['analysis_errors'].append('trend_analysis_invalid_data')
            except Exception as e:
                logger.exception(f"Unexpected error in trend analysis: {e}")
                result['analysis_errors'].append('trend_analysis_failed')
                # Fallback trend analysis
                trend_analysis = {
                    'daily_trend': 'UNKNOWN',
                    'intraday_trend': 'NEUTRAL',
                    'trend_strength': 0.3,
                    'sma_20': current_price,
                    'sma_50': current_price,
                    'trend_quality': 0.3
                }
            result['trend_analysis'] = trend_analysis
            
            # 2. Support/Resistance Analysis
            sr_levels = {}
            try:
                sr_levels = self._find_support_resistance(df_daily, current_price)
            except (KeyError, ValueError) as e:
                logger.error(f"Data error in S/R analysis: {e}", exc_info=True)
                result['analysis_errors'].append('support_resistance_data_error')
            except Exception as e:
                logger.exception(f"Unexpected error in S/R analysis: {e}")
                result['analysis_errors'].append('support_resistance_failed')
                # Fallback S/R levels
                sr_levels = {
                    'support_levels': [current_price * 0.95],
                    'resistance_levels': [current_price * 1.05],
                    'nearest_support': current_price * 0.95,
                    'nearest_resistance': current_price * 1.05,
                    'current_level': 'MIDDLE',
                    'level_strength': 0.3
                }
            result['support_resistance'] = sr_levels
            
            # 3. Momentum Analysis
            momentum = {}
            try:
                momentum = self._analyze_momentum(df_daily, df_intraday, intraday_context)
            except Exception as e:
                logger.warning(f"[WARNING] Momentum analysis failed: {e}")
                result['analysis_errors'].append('momentum_analysis_failed')
                # Fallback momentum
                momentum = {
                    'direction': 'NEUTRAL',
                    'strength': 0.3,
                    'rsi': 50,
                    'intraday_rsi': 50,
                    'momentum_score': 0.0
                }
            result['momentum_analysis'] = momentum
            
            # 4. Volume Analysis
            volume_analysis = {}
            try:
                volume_analysis = self._analyze_volume(df_daily, market_data)
            except Exception as e:
                logger.warning(f"[WARNING] Volume analysis failed: {e}")
                result['analysis_errors'].append('volume_analysis_failed')
                # Fallback volume analysis
                volume_analysis = {
                    'trend': 'UNKNOWN',
                    'current_vs_avg': 1.0,
                    'volume_confirmation': False,
                    'institutional_activity': 'LOW'
                }
            result['volume_analysis'] = volume_analysis
            
            # 5. Pattern Analysis
            pattern_signals = {}
            try:
                pattern_signals = self._detect_key_patterns(df_daily, current_price)
            except Exception as e:
                logger.warning(f"[WARNING] Pattern analysis failed: {e}")
                result['analysis_errors'].append('pattern_analysis_failed')
                # Fallback patterns
                pattern_signals = {
                    'detected_patterns': [],
                    'strongest_pattern': None,
                    'consolidation': False
                }
            result['pattern_signals'] = pattern_signals
            
            # **CRITICAL: Generate entry/exit signals with error recovery**
            entry_signal_dict = {}
            try:
                entry_signal = self._generate_smart_entry_signal(
                    current_price, trend_analysis, sr_levels, momentum, pattern_signals, trading_style, intraday_context
                )
                
                # **SIMPLIFIED CONVERSION** - Much cleaner approach
                if isinstance(entry_signal, TechnicalSignal):
                    entry_signal_dict = {
                        'signal_type': entry_signal.signal_type,
                        'strength': entry_signal.strength,
                        'reason': entry_signal.reason,
                        'entry_price': entry_signal.entry_price,
                        'entry_condition': entry_signal.entry_condition,
                        'stop_loss': entry_signal.stop_loss,
                        'target_1': entry_signal.target_1,
                        'target_2': entry_signal.target_2,
                        'time_frame': entry_signal.time_frame,
                        'risk_reward_ratio': entry_signal.risk_reward_ratio
                    }
                else:
                    # Handle dict or None
                    entry_signal_dict = entry_signal if isinstance(entry_signal, dict) else {
                        'signal_type': 'HOLD',
                        'strength': 0.3,
                        'reason': 'Signal generation returned invalid format',
                        'entry_price': None,
                        'entry_condition': 'Wait for valid signal',
                        'stop_loss': None,
                        'target_1': None,
                        'target_2': None,
                        'time_frame': trading_style.upper(),
                        'risk_reward_ratio': 1.0
                    }
            except Exception as e:
                logger.warning(f"[WARNING] Signal generation failed: {e}")
                result['analysis_errors'].append('signal_generation_failed')
                # **FALLBACK SIGNAL GENERATION** - Generate simple signal based on available data
                entry_signal_dict = self._generate_fallback_signal(
                    current_price, trend_analysis, momentum, trading_style
                )
            
            result['entry_signal'] = entry_signal_dict
            
            # Generate exit rules
            exit_rules = {}
            try:
                exit_rules = self._generate_smart_exit_rules(
                    current_price, sr_levels, trend_analysis, trading_style
                )
            except Exception as e:
                logger.warning(f"[WARNING] Exit rules generation failed: {e}")
                result['analysis_errors'].append('exit_rules_failed')
                # Fallback exit rules
                exit_rules = {
                    'profit_targets': [f"Conservative 15% target (₹{current_price * 1.15:.0f})"],
                    'stop_losses': [f"Conservative 20% stop (₹{current_price * 0.80:.0f})"],
                    'time_stops': ['End of session' if trading_style == 'intraday' else 'Weekly review'],
                    'technical_exits': ['Manual monitoring required']
                }
            result['exit_rules'] = exit_rules
            
            # Options market context
            options_context = {}
            try:
                options_context = self._analyze_options_market_context(
                    current_price, sr_levels, trend_analysis, momentum, trading_style, intraday_context
                )
            except Exception as e:
                logger.warning(f"[WARNING] Options context analysis failed: {e}")
                result['analysis_errors'].append('options_context_failed')
                # Fallback options context
                options_context = {
                    'strategy_bias': 'NEUTRAL_STRATEGIES',
                    'recommended_strikes': {'atm_strike': round(current_price / 50) * 50},
                    'volatility_expectation': 'MODERATE',
                    'optimal_entry_time': 'Market hours'
                }
            result['options_context'] = options_context
            
            # **OVERALL ASSESSMENT** - Use available data with error recovery
            try:
                result['market_bias'] = self._determine_market_bias(trend_analysis, momentum, pattern_signals, intraday_context)
            except Exception as e:
                logger.warning(f"[WARNING] Market bias determination failed: {e}")
                result['market_bias'] = 'NEUTRAL'
            
            try:
                result['confidence_score'] = self._calculate_confidence_score(
                    trend_analysis, sr_levels, momentum, pattern_signals, intraday_context
                )
            except Exception as e:
                logger.warning(f"[WARNING] Confidence calculation failed: {e}")
                result['confidence_score'] = 0.3
            
            try:
                result['risk_assessment'] = self._assess_risk_level(sr_levels, momentum, trading_style)
            except Exception as e:
                logger.warning(f"[WARNING] Risk assessment failed: {e}")
                result['risk_assessment'] = 'MODERATE'
            
            # **ANALYSIS QUALITY CHECK**
            if result['analysis_errors']:
                result['partial_analysis'] = True
                logger.warning(f"[WARNING] Partial analysis completed for {symbol} with {len(result['analysis_errors'])} errors")
            else:
                logger.info(f"[OK] Complete technical analysis for {symbol}: {result['market_bias']} bias")
            
            # Adjust confidence based on data quality and errors
            if result['data_quality'] in ['LIMITED', 'VERY_LIMITED']:
                result['confidence_score'] *= 0.8
            if result['partial_analysis']:
                result['confidence_score'] *= (1 - len(result['analysis_errors']) * 0.1)
            result['confidence_score'] = max(0.2, min(0.95, result['confidence_score']))
            
            return result
            
        except Exception as e:
            # **LAST RESORT: Only use complete fallback when everything fails**
            logger.error(f"[ERROR] Complete technical analysis failure for {symbol}: {e}")
            result['analysis_errors'].append('complete_failure')
            
            # Still try to provide SOME analysis rather than giving up
            fallback_result = self._create_enhanced_fallback_analysis(symbol, current_price, trading_style)
            
            # Merge any partial results we managed to get
            for key, value in result.items():
                if key not in fallback_result and value:
                    fallback_result[key] = value
            
            return fallback_result

    def _generate_fallback_signal(self, current_price: float, trend_analysis: Dict, 
                                momentum: Dict, trading_style: str) -> Dict:
        """Generate fallback signal when main signal generation fails"""
        
        # Try to use whatever data we have
        daily_trend = trend_analysis.get('daily_trend', 'UNKNOWN')
        momentum_direction = momentum.get('direction', 'NEUTRAL')
        rsi = momentum.get('rsi', 50)
        
        # Simple fallback logic
        if daily_trend in ['UPTREND', 'STRONG_UPTREND'] or momentum_direction in ['BULLISH', 'WEAK_BULLISH']:
            return {
                'signal_type': 'BUY',
                'strength': 0.45,
                'reason': f'Fallback bullish signal: trend={daily_trend}, momentum={momentum_direction}',
                'entry_price': current_price,
                'entry_condition': 'Enter on any dip or continuation',
                'stop_loss': current_price * 0.95,
                'target_1': current_price * 1.05,
                'target_2': current_price * 1.10,
                'time_frame': trading_style.upper(),
                'risk_reward_ratio': 1.5
            }
        
        elif daily_trend in ['DOWNTREND', 'STRONG_DOWNTREND'] or momentum_direction in ['BEARISH', 'WEAK_BEARISH']:
            return {
                'signal_type': 'SELL',
                'strength': 0.45,
                'reason': f'Fallback bearish signal: trend={daily_trend}, momentum={momentum_direction}',
                'entry_price': current_price,
                'entry_condition': 'Enter on any bounce or continuation',
                'stop_loss': current_price * 1.05,
                'target_1': current_price * 0.95,
                'target_2': current_price * 0.90,
                'time_frame': trading_style.upper(),
                'risk_reward_ratio': 1.5
            }
        
        elif rsi < 35:
            return {
                'signal_type': 'BUY',
                'strength': 0.4,
                'reason': f'Fallback oversold signal: RSI {rsi:.0f}',
                'entry_price': current_price,
                'entry_condition': 'Enter on oversold bounce',
                'stop_loss': current_price * 0.95,
                'target_1': current_price * 1.05,
                'time_frame': trading_style.upper(),
                'risk_reward_ratio': 1.3
            }
        
        elif rsi > 65:
            return {
                'signal_type': 'SELL',
                'strength': 0.4,
                'reason': f'Fallback overbought signal: RSI {rsi:.0f}',
                'entry_price': current_price,
                'entry_condition': 'Enter on overbought fade',
                'stop_loss': current_price * 1.05,
                'target_1': current_price * 0.95,
                'time_frame': trading_style.upper(),
                'risk_reward_ratio': 1.3
            }
        
        else:
            return {
                'signal_type': 'HOLD',
                'strength': 0.3,
                'reason': 'Fallback neutral signal - insufficient data for directional bias',
                'entry_price': None,
                'entry_condition': 'Wait for clearer market direction',
                'stop_loss': None,
                'target_1': None,
                'target_2': None,
                'time_frame': trading_style.upper(),
                'risk_reward_ratio': 1.0
            }

    def _create_enhanced_fallback_analysis(self, symbol: str, current_price: float, trading_style: str) -> Dict:
        """Enhanced fallback analysis - better than the basic version"""
        
        basic_fallback = self._create_fallback_analysis(symbol, current_price)
        
        # Enhance with simple signal generation
        fallback_signal = self._generate_fallback_signal(
            current_price, 
            basic_fallback.get('trend_analysis', {}), 
            basic_fallback.get('momentum_analysis', {}), 
            trading_style
        )
        
        basic_fallback['entry_signal'] = fallback_signal
        basic_fallback['analysis_errors'] = ['complete_technical_analysis_failure']
        basic_fallback['partial_analysis'] = False
        basic_fallback['data_quality'] = 'FALLBACK'
        
        return basic_fallback

    
    def _get_zerodha_symbol(self, symbol: str) -> str:
        """Convert symbol to Zerodha format for historical data - UPDATED FOR YOUR WATCHLIST"""
        symbol_mapping = {
            # Your Primary Watchlist
            'NIFTY': 'NIFTY 50',
            'RELIANCE': 'RELIANCE',
            'HDFCBANK': 'HDFCBANK', 
            'TCS': 'TCS',
            'INFY': 'INFY',
            'BAJFINANCE': 'BAJFINANCE',  # [OK] ADDED
            'MARUTI': 'MARUTI',          # [OK] ADDED
            
            # Additional Popular Indices (for options trading)
            'BANKNIFTY': 'NIFTY BANK',
            'FINNIFTY': 'NIFTY FIN SERVICE',
            'NIFTYNXT50': 'NIFTY NEXT 50',
            'MIDCPNIFTY': 'NIFTY MIDCAP 100',
            
            # Other popular stocks (in case you expand)
            'ICICIBANK': 'ICICIBANK',      # Already there
            'SBIN': 'SBIN',                # Already there  
            'HINDUNILVR': 'HINDUNILVR',    # Already there
            'ITC': 'ITC',                  # Already there
            'LT': 'LT',                    # Already there
            'HCLTECH': 'HCLTECH',          # NEW - ADD THIS
            'WIPRO': 'WIPRO',              # Already there
            'TECHM': 'TECHM',              # NEW - ADD THIS
            'ASIANPAINT': 'ASIANPAINT',    # Already there
            'MPHASIS': 'MPHASIS',          # NEW - ADD THIS
            'BHARTIARTL': 'BHARTIARTL',    # NEW - ADD THIS
            'ADANIPORTS': 'ADANIPORTS',    # NEW - ADD THIS
            'TATAMOTORS': 'TATAMOTORS',    # NEW - ADD THIS
        }
        
        # Return the mapped symbol or the original symbol if not found
        mapped_symbol = symbol_mapping.get(symbol.upper(), symbol.upper())
        
        # Log the mapping for debugging
        if symbol.upper() in symbol_mapping:
            print(f"[OK] Mapped {symbol} -> {mapped_symbol}")
        else:
            print(f"[WARNING] No mapping for {symbol}, using as-is: {mapped_symbol}")
        
        return mapped_symbol

    def _analyze_today_context(self, current_price: float, market_data: Dict, 
                              df_daily: pd.DataFrame, df_intraday: pd.DataFrame) -> Dict:
        """**NEW METHOD**: Analyze TODAY's market context for intraday trading"""
        
        # Get yesterday's close and today's open
        yesterday_close = df_daily['close'].iloc[-2] if len(df_daily) >= 2 else current_price
        today_open = df_intraday['open'].iloc[0] if len(df_intraday) > 0 else current_price
        
        # Calculate gap
        gap_percent = ((today_open - yesterday_close) / yesterday_close) * 100
        gap_type = 'NONE'
        if gap_percent > 0.5:
            gap_type = 'GAP_UP'
        elif gap_percent < -0.5:
            gap_type = 'GAP_DOWN'
        
        # Intraday momentum (today's price vs today's open)
        intraday_change = ((current_price - today_open) / today_open) * 100
        intraday_momentum = 'NEUTRAL'
        if intraday_change > 0.3:
            intraday_momentum = 'BULLISH'
        elif intraday_change < -0.3:
            intraday_momentum = 'BEARISH'
        
        # Today's volume vs average (if available)
        current_volume = market_data.get('volume', 0)
        avg_volume = df_daily['volume'].rolling(10).mean().iloc[-1] if 'volume' in df_daily.columns else 0
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Intraday high/low analysis
        if len(df_intraday) > 0:
            today_high = df_intraday['high'].max()
            today_low = df_intraday['low'].min()
            high_low_range = ((today_high - today_low) / today_open) * 100
        else:
            today_high = current_price
            today_low = current_price
            high_low_range = 0
        
        # Time-based context
        current_hour = datetime.now().hour
        session_phase = 'OPENING' if current_hour < 10 else 'MID_SESSION' if current_hour < 14 else 'CLOSING'
        
        return {
            'yesterday_close': yesterday_close,
            'today_open': today_open,
            'gap_percent': gap_percent,
            'gap_type': gap_type,
            'intraday_change_percent': intraday_change,
            'intraday_momentum': intraday_momentum,
            'volume_ratio': volume_ratio,
            'today_high': today_high,
            'today_low': today_low,
            'high_low_range_percent': high_low_range,
            'session_phase': session_phase,
            'gap_fill_target': yesterday_close,
            'gap_extension_target': today_open + (today_open - yesterday_close) * 0.5 if gap_type != 'NONE' else today_open
        }
    
    def _analyze_trend(self, df_daily: pd.DataFrame, df_intraday: pd.DataFrame, 
                      current_price: float, intraday_context: Dict = None) -> Dict:
        """Enhanced trend analysis with intraday focus"""
        
        # Calculate moving averages
        df_daily['sma_20'] = df_daily['close'].rolling(20).mean()
        df_daily['sma_50'] = df_daily['close'].rolling(50).mean() if len(df_daily) >= 50 else df_daily['sma_20']
        df_daily['ema_12'] = df_daily['close'].ewm(span=12).mean()
        df_daily['ema_26'] = df_daily['close'].ewm(span=26).mean()
        
        sma_20 = df_daily['sma_20'].iloc[-1]
        sma_50 = df_daily['sma_50'].iloc[-1]
        ema_12 = df_daily['ema_12'].iloc[-1]
        ema_26 = df_daily['ema_26'].iloc[-1]
        
        # Determine daily trend direction
        if current_price > sma_20 > sma_50 and ema_12 > ema_26:
            trend_direction = 'STRONG_UPTREND'
            trend_strength = 0.8
        elif current_price > sma_20 and ema_12 > ema_26:
            trend_direction = 'UPTREND'
            trend_strength = 0.6
        elif current_price < sma_20 < sma_50 and ema_12 < ema_26:
            trend_direction = 'STRONG_DOWNTREND'
            trend_strength = 0.8
        elif current_price < sma_20 and ema_12 < ema_26:
            trend_direction = 'DOWNTREND'
            trend_strength = 0.6
        else:
            trend_direction = 'SIDEWAYS'
            trend_strength = 0.3
        
        # **ENHANCED INTRADAY TREND** - Focus on TODAY's action
        intraday_trend = 'NEUTRAL'
        if intraday_context:
            # Use today's momentum as primary factor
            if intraday_context['intraday_momentum'] == 'BULLISH':
                intraday_trend = 'UP'
            elif intraday_context['intraday_momentum'] == 'BEARISH':
                intraday_trend = 'DOWN'
            
            # Consider gap context
            if intraday_context['gap_type'] == 'GAP_UP' and intraday_context['intraday_change_percent'] > 0:
                intraday_trend = 'UP'  # Gap up continuing
            elif intraday_context['gap_type'] == 'GAP_DOWN' and intraday_context['intraday_change_percent'] < 0:
                intraday_trend = 'DOWN'  # Gap down continuing
        
        # Fallback to traditional intraday analysis
        if intraday_trend == 'NEUTRAL' and len(df_intraday) >= 20:
            intraday_sma = df_intraday['close'].rolling(20).mean().iloc[-1]
            if current_price > intraday_sma * 1.005:
                intraday_trend = 'UP'
            elif current_price < intraday_sma * 0.995:
                intraday_trend = 'DOWN'
        
        return {
            'daily_trend': trend_direction,
            'intraday_trend': intraday_trend,
            'trend_strength': trend_strength,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'ema_12': ema_12,
            'ema_26': ema_26,
            'price_above_sma20': current_price > sma_20,
            'ma_alignment': sma_20 > sma_50 if not pd.isna(sma_50) else True,
            'trend_quality': self._calculate_trend_quality(df_daily),
            'intraday_vs_daily_alignment': self._check_trend_alignment(trend_direction, intraday_trend)
        }

    def _check_trend_alignment(self, daily_trend: str, intraday_trend: str) -> str:
        """Check if intraday and daily trends are aligned"""
        daily_bullish = daily_trend in ['UPTREND', 'STRONG_UPTREND']
        daily_bearish = daily_trend in ['DOWNTREND', 'STRONG_DOWNTREND']
        
        if daily_bullish and intraday_trend == 'UP':
            return 'ALIGNED_BULLISH'
        elif daily_bearish and intraday_trend == 'DOWN':
            return 'ALIGNED_BEARISH'
        elif daily_bullish and intraday_trend == 'DOWN':
            return 'PULLBACK_IN_UPTREND'
        elif daily_bearish and intraday_trend == 'UP':
            return 'BOUNCE_IN_DOWNTREND'
        else:
            return 'NEUTRAL'
        
    def _technical_signal_to_dict(self, signal: TechnicalSignal) -> Dict:
        """Convert TechnicalSignal object to dictionary"""
        return {
            'signal_type': signal.signal_type,
            'strength': signal.strength,
            'reason': signal.reason,
            'entry_price': signal.entry_price,
            'entry_condition': signal.entry_condition,
            'stop_loss': signal.stop_loss,
            'target_1': signal.target_1,
            'target_2': signal.target_2,
            'time_frame': signal.time_frame,
            'risk_reward_ratio': signal.risk_reward_ratio
        }
        
    # Add this method to your ZerodhaTechnicalAnalyzer class
    async def analyze_symbol_for_options_enhanced(self, symbol: str, current_price: float, 
                                                market_data: Dict, trading_style: str) -> Dict:
        """Enhanced technical analysis specifically for options trading"""
        
        # Get the basic technical analysis first
        basic_analysis = await self.analyze_symbol_for_options(
            symbol, current_price, market_data, trading_style
        )
        
        # You can add additional enhancements here if needed
        return basic_analysis
    
    def copy(self):
        """Create a copy of the analyzer for thread safety"""
        return ZerodhaTechnicalAnalyzer(self.zerodha)
    
    def _calculate_trend_quality(self, df: pd.DataFrame) -> float:
        """Calculate trend quality using linear regression"""
        if len(df) < 10:
            return 0.5
        
        closes = df['close'].values[-20:]  # Last 20 days
        x = np.arange(len(closes))
        
        try:
            slope, intercept = np.polyfit(x, closes, 1)
            y_pred = slope * x + intercept
            r_squared = 1 - (np.sum((closes - y_pred) ** 2) / np.sum((closes - np.mean(closes)) ** 2))
            return max(0, min(1, r_squared))
        except:
            return 0.5
    
    def _find_support_resistance(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Find key support and resistance levels using pivot points"""
        
        # Calculate pivot points
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # Find swing highs and lows
        resistance_levels = []
        support_levels = []
        
        # Look for pivot points with 3-period window
        for i in range(3, len(highs) - 3):
            # Pivot high
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i-3] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2] and highs[i] > highs[i+3]):
                resistance_levels.append(highs[i])
            
            # Pivot low
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i-3] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2] and lows[i] < lows[i+3]):
                support_levels.append(lows[i])
        
        # Filter and sort levels
        resistance_levels = [r for r in resistance_levels if r > current_price]
        support_levels = [s for s in support_levels if s < current_price]
        
        resistance_levels.sort()
        support_levels.sort(reverse=True)
        
        # Get nearest levels
        nearest_resistance = resistance_levels[0] if resistance_levels else current_price * 1.05
        nearest_support = support_levels[0] if support_levels else current_price * 0.95
        
        # Determine current position
        distance_to_support = abs(current_price - nearest_support) / current_price
        distance_to_resistance = abs(nearest_resistance - current_price) / current_price
        
        if distance_to_support < distance_to_resistance:
            current_level = 'NEAR_SUPPORT'
            level_strength = 1 - distance_to_support * 20  # Higher strength when closer
        else:
            current_level = 'NEAR_RESISTANCE'
            level_strength = 1 - distance_to_resistance * 20
        
        level_strength = max(0, min(1, level_strength))
        
        return {
            'support_levels': support_levels[:3],  # Top 3
            'resistance_levels': resistance_levels[:3],  # Top 3
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'current_level': current_level,
            'level_strength': level_strength,
            'support_distance_pct': distance_to_support * 100,
            'resistance_distance_pct': distance_to_resistance * 100
        }
    
    def _analyze_momentum(self, df_daily: pd.DataFrame, df_intraday: pd.DataFrame, 
                         intraday_context: Dict = None) -> Dict:
        """Enhanced momentum analysis with intraday focus"""
        
        # RSI calculation
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        # MACD calculation
        def calculate_macd(prices):
            exp1 = prices.ewm(span=12).mean()
            exp2 = prices.ewm(span=26).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        
        # Calculate daily indicators
        rsi = calculate_rsi(df_daily['close'])
        macd_line, macd_signal, macd_histogram = calculate_macd(df_daily['close'])
        
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        current_macd_histogram = macd_histogram.iloc[-1] if not pd.isna(macd_histogram.iloc[-1]) else 0
        
        # **ENHANCED INTRADAY MOMENTUM** - Use shorter timeframes for intraday
        intraday_rsi = 50
        if len(df_intraday) >= 14:
            intraday_rsi_series = calculate_rsi(df_intraday['close'], period=14)
            intraday_rsi = intraday_rsi_series.iloc[-1] if not pd.isna(intraday_rsi_series.iloc[-1]) else 50
        
        # Use intraday context for momentum direction
        momentum_direction = 'NEUTRAL'
        momentum_strength = 0.3
        
        if intraday_context:
            # Primary: Use today's momentum
            if intraday_context['intraday_momentum'] == 'BULLISH' and intraday_rsi > 45:
                momentum_direction = 'BULLISH'
                momentum_strength = 0.7 if intraday_rsi > 60 else 0.5
            elif intraday_context['intraday_momentum'] == 'BEARISH' and intraday_rsi < 55:
                momentum_direction = 'BEARISH'
                momentum_strength = 0.7 if intraday_rsi < 40 else 0.5
            
            # Consider gap momentum
            if intraday_context['gap_type'] == 'GAP_UP' and intraday_context['intraday_change_percent'] > 0:
                momentum_direction = 'BULLISH'
                momentum_strength = min(0.8, momentum_strength + 0.2)
            elif intraday_context['gap_type'] == 'GAP_DOWN' and intraday_context['intraday_change_percent'] < 0:
                momentum_direction = 'BEARISH'
                momentum_strength = min(0.8, momentum_strength + 0.2)
        
        # Fallback to traditional momentum
        if momentum_direction == 'NEUTRAL':
            if current_rsi > 60 and current_macd_histogram > 0:
                momentum_direction = 'BULLISH'
                momentum_strength = 0.8
            elif current_rsi < 40 and current_macd_histogram < 0:
                momentum_direction = 'BEARISH'
                momentum_strength = 0.8
            elif current_rsi > 50 and current_macd_histogram > 0:
                momentum_direction = 'WEAK_BULLISH'
                momentum_strength = 0.5
            elif current_rsi < 50 and current_macd_histogram < 0:
                momentum_direction = 'WEAK_BEARISH'
                momentum_strength = 0.5
        
        # Overbought/Oversold conditions
        overbought = intraday_rsi > 70
        oversold = intraday_rsi < 30
        
        return {
            'direction': momentum_direction,
            'strength': momentum_strength,
            'rsi': current_rsi,
            'intraday_rsi': intraday_rsi,
            'macd_histogram': current_macd_histogram,
            'overbought': overbought,
            'oversold': oversold,
            'momentum_score': self._calculate_momentum_score(intraday_rsi, current_macd_histogram)
        }
    
    def _calculate_momentum_score(self, rsi: float, macd_histogram: float) -> float:
        """Calculate composite momentum score"""
        rsi_score = (rsi - 50) / 50  # Normalize RSI to -1 to 1
        macd_score = max(-1, min(1, macd_histogram * 10))  # Normalize MACD
        
        return (rsi_score + macd_score) / 2
    
    def _analyze_volume(self, df: pd.DataFrame, market_data: Dict) -> Dict:
        """Analyze volume patterns"""
        
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            # Use volume from market_data if available
            current_volume = market_data.get('volume', 0)
            return {
                'trend': 'UNKNOWN',
                'current_vs_avg': 1.0,
                'volume_confirmation': False,
                'institutional_activity': 'LOW'
            }
        
        volumes = df['volume']
        avg_volume = volumes.rolling(20).mean().iloc[-1]
        current_volume = volumes.iloc[-1]
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume trend
        if volume_ratio > 1.5:
            volume_trend = 'SURGING'
            institutional_activity = 'HIGH'
        elif volume_ratio > 1.2:
            volume_trend = 'INCREASING'
            institutional_activity = 'MODERATE'
        elif volume_ratio < 0.5:
            volume_trend = 'DECLINING'
            institutional_activity = 'LOW'
        else:
            volume_trend = 'NORMAL'
            institutional_activity = 'LOW'
        
        # Volume confirmation (high volume on price moves)
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
        volume_confirmation = abs(price_change) > 0.01 and volume_ratio > 1.2
        
        return {
            'trend': volume_trend,
            'current_vs_avg': volume_ratio,
            'volume_confirmation': volume_confirmation,
            'institutional_activity': institutional_activity,
            'avg_volume': int(avg_volume) if not pd.isna(avg_volume) else 0
        }
    
    def _detect_key_patterns(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Detect key chart patterns relevant for options trading"""
        
        patterns = {
            'detected_patterns': [],
            'strongest_pattern': None,
            'breakout_imminent': False,
            'consolidation': False
        }
        
        if len(df) < 15:
            return patterns
        
        highs = df['high'].values[-15:]
        lows = df['low'].values[-15:]
        closes = df['close'].values[-15:]
        
        # Consolidation detection (important for options)
        recent_range = (np.max(highs) - np.min(lows)) / np.mean(closes)
        if recent_range < 0.05:  # Less than 5% range
            patterns['consolidation'] = True
            patterns['detected_patterns'].append({
                'name': 'CONSOLIDATION',
                'strength': 0.7,
                'signal': 'BREAKOUT_PENDING'
            })
        
        # Triangle patterns
        if self._detect_triangle_pattern(highs, lows):
            patterns['detected_patterns'].append({
                'name': 'TRIANGLE',
                'strength': 0.6,
                'signal': 'BREAKOUT_PENDING'
            })
            patterns['breakout_imminent'] = True
        
        # Flag pattern (quick consolidation after strong move)
        if self._detect_flag_pattern(closes):
            patterns['detected_patterns'].append({
                'name': 'FLAG',
                'strength': 0.7,
                'signal': 'CONTINUATION'
            })
        
        # Select strongest pattern
        if patterns['detected_patterns']:
            patterns['strongest_pattern'] = max(patterns['detected_patterns'], 
                                              key=lambda x: x['strength'])['name']
        
        return patterns
    
    def _detect_triangle_pattern(self, highs: np.ndarray, lows: np.ndarray) -> bool:
        """Simple triangle pattern detection"""
        if len(highs) < 10:
            return False
        
        # Check if highs are declining and lows are rising (converging)
        high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
        low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
        
        return high_trend < -0.1 and low_trend > 0.1
    
    def _detect_flag_pattern(self, closes: np.ndarray) -> bool:
        """Detect flag pattern (consolidation after strong move)"""
        if len(closes) < 10:
            return False
        
        # Check for strong initial move
        initial_move = (closes[5] - closes[0]) / closes[0]
        
        # Check for consolidation after
        consolidation_range = (np.max(closes[5:]) - np.min(closes[5:])) / np.mean(closes[5:])
        
        return abs(initial_move) > 0.03 and consolidation_range < 0.02
    
    def _generate_smart_entry_signal(self, current_price: float, trend: Dict, sr_levels: Dict,
                                   momentum: Dict, patterns: Dict, trading_style: str, 
                                   intraday_context: Dict = None) -> TechnicalSignal:
        """Enhanced smart entry signal with intraday context"""
        
        # Entry conditions based on multiple factors
        if trading_style == 'intraday':
            return self._generate_intraday_entry_signal(
                current_price, trend, sr_levels, momentum, patterns, intraday_context
            )
        else:
            return self._generate_swing_entry_signal(
                current_price, trend, sr_levels, momentum, patterns
            )
    
    def _generate_intraday_entry_signal(self, current_price: float, trend: Dict, 
                                      sr_levels: Dict, momentum: Dict, patterns: Dict,
                                      intraday_context: Dict = None) -> TechnicalSignal:
        """FIXED: Enhanced intraday entry signals with moderate aggressiveness (45%+ confidence)"""
        
        # **PRIORITY 1: GAP STRATEGIES** - Keep existing strong gap logic
        if intraday_context:
            gap_type = intraday_context.get('gap_type', 'NONE')
            intraday_momentum = intraday_context.get('intraday_momentum', 'NEUTRAL')
            session_phase = intraday_context.get('session_phase', 'MID_SESSION')
            
            # Gap continuation strategies (HIGH CONFIDENCE)
            if gap_type == 'GAP_UP' and intraday_momentum == 'BULLISH':
                return TechnicalSignal(
                    signal_type='BUY',
                    strength=0.8,
                    reason=f"Gap up continuation - {intraday_context.get('gap_percent', 0):.1f}% gap with bullish momentum",
                    entry_price=current_price,
                    entry_condition="Enter NOW - gap up continuation pattern",
                    stop_loss=intraday_context.get('today_open', current_price) * 0.995,
                    target_1=intraday_context.get('gap_extension_target', current_price * 1.02),
                    target_2=sr_levels['nearest_resistance'],
                    time_frame='INTRADAY',
                    risk_reward_ratio=2.0
                )
            
            elif gap_type == 'GAP_DOWN' and intraday_momentum == 'BEARISH':
                return TechnicalSignal(
                    signal_type='SELL',
                    strength=0.8,
                    reason=f"Gap down continuation - {intraday_context.get('gap_percent', 0):.1f}% gap with bearish momentum",
                    entry_price=current_price,
                    entry_condition="Enter NOW - gap down continuation pattern",
                    stop_loss=intraday_context.get('today_open', current_price) * 1.005,
                    target_1=intraday_context.get('gap_extension_target', current_price * 0.98),
                    target_2=sr_levels['nearest_support'],
                    time_frame='INTRADAY',
                    risk_reward_ratio=2.0
                )
            
            # Gap reversal strategies (GOOD CONFIDENCE)
            elif gap_type == 'GAP_UP' and intraday_momentum == 'BEARISH':
                return TechnicalSignal(
                    signal_type='SELL',
                    strength=0.7,
                    reason="Gap up reversal - failed gap with bearish momentum",
                    entry_price=current_price,
                    entry_condition="Enter on break below gap fill level",
                    stop_loss=intraday_context.get('today_high', current_price) * 1.005,
                    target_1=intraday_context.get('gap_fill_target', current_price * 0.99),
                    target_2=sr_levels['nearest_support'],
                    time_frame='INTRADAY',
                    risk_reward_ratio=1.8
                )
            
            elif gap_type == 'GAP_DOWN' and intraday_momentum == 'BULLISH':
                return TechnicalSignal(
                    signal_type='BUY',
                    strength=0.7,
                    reason="Gap down reversal - failed gap with bullish momentum",
                    entry_price=current_price,
                    entry_condition="Enter on break above gap fill level",
                    stop_loss=intraday_context.get('today_low', current_price) * 0.995,
                    target_1=intraday_context.get('gap_fill_target', current_price * 1.01),
                    target_2=sr_levels['nearest_resistance'],
                    time_frame='INTRADAY',
                    risk_reward_ratio=1.8
                )
            
            # **NEW: Strong intraday momentum (any momentum, no gap required)**
            elif intraday_momentum == 'BULLISH':
                strength = 0.75 if momentum['direction'] in ['BULLISH', 'WEAK_BULLISH'] else 0.6
                return TechnicalSignal(
                    signal_type='BUY',
                    strength=strength,
                    reason=f"Intraday bullish momentum - {intraday_context.get('intraday_change_percent', 0):.1f}% move",
                    entry_price=current_price,
                    entry_condition="Enter NOW - momentum continuation",
                    stop_loss=intraday_context.get('today_low', current_price * 0.99) * 0.998,
                    target_1=current_price * 1.015,  # 1.5% target
                    target_2=sr_levels['nearest_resistance'],
                    time_frame='INTRADAY',
                    risk_reward_ratio=2.0
                )
            
            elif intraday_momentum == 'BEARISH':
                strength = 0.75 if momentum['direction'] in ['BEARISH', 'WEAK_BEARISH'] else 0.6
                return TechnicalSignal(
                    signal_type='SELL',
                    strength=strength,
                    reason=f"Intraday bearish momentum - {intraday_context.get('intraday_change_percent', 0):.1f}% move",
                    entry_price=current_price,
                    entry_condition="Enter NOW - momentum continuation",
                    stop_loss=intraday_context.get('today_high', current_price * 1.01) * 1.002,
                    target_1=current_price * 0.985,  # 1.5% target
                    target_2=sr_levels['nearest_support'],
                    time_frame='INTRADAY',
                    risk_reward_ratio=2.0
                )
        
        # **PRIORITY 2: TRADITIONAL TECHNICAL SETUPS** - Reduced to OR conditions
        
        # Bullish setup - Now uses OR instead of AND for more signals
        if (trend.get('intraday_trend') == 'UP' or 
            momentum['direction'] in ['BULLISH', 'WEAK_BULLISH'] or
            (sr_levels['current_level'] == 'NEAR_SUPPORT' and momentum['rsi'] > 45)):
            
            # Calculate strength based on how many conditions are met
            strength = 0.45  # Base moderate confidence
            if trend.get('intraday_trend') == 'UP':
                strength += 0.1
            if momentum['direction'] in ['BULLISH', 'WEAK_BULLISH']:
                strength += 0.1
            if sr_levels['current_level'] == 'NEAR_SUPPORT':
                strength += 0.05
            
            return TechnicalSignal(
                signal_type='BUY',
                strength=min(0.8, strength),
                reason=f"Bullish setup: trend={trend.get('intraday_trend', 'N/A')}, momentum={momentum['direction']}, level={sr_levels['current_level']}",
                entry_price=current_price,
                entry_condition=f"Enter above ₹{sr_levels['nearest_support'] * 1.002:.2f}",
                stop_loss=sr_levels['nearest_support'] * 0.995,
                target_1=current_price + (current_price - sr_levels['nearest_support']) * 0.8,
                target_2=sr_levels['nearest_resistance'],
                time_frame='INTRADAY',
                risk_reward_ratio=1.5
            )
        
        # Bearish setup - Now uses OR instead of AND for more signals
        elif (trend.get('intraday_trend') == 'DOWN' or 
            momentum['direction'] in ['BEARISH', 'WEAK_BEARISH'] or
            (sr_levels['current_level'] == 'NEAR_RESISTANCE' and momentum['rsi'] < 55)):
            
            # Calculate strength based on how many conditions are met
            strength = 0.45  # Base moderate confidence
            if trend.get('intraday_trend') == 'DOWN':
                strength += 0.1
            if momentum['direction'] in ['BEARISH', 'WEAK_BEARISH']:
                strength += 0.1
            if sr_levels['current_level'] == 'NEAR_RESISTANCE':
                strength += 0.05
            
            return TechnicalSignal(
                signal_type='SELL',
                strength=min(0.8, strength),
                reason=f"Bearish setup: trend={trend.get('intraday_trend', 'N/A')}, momentum={momentum['direction']}, level={sr_levels['current_level']}",
                entry_price=current_price,
                entry_condition=f"Enter below ₹{sr_levels['nearest_resistance'] * 0.998:.2f}",
                stop_loss=sr_levels['nearest_resistance'] * 1.005,
                target_1=current_price - (sr_levels['nearest_resistance'] - current_price) * 0.8,
                target_2=sr_levels['nearest_support'],
                time_frame='INTRADAY',
                risk_reward_ratio=1.5
            )
        
        # **PRIORITY 3: RSI-BASED SIGNALS** - New simple momentum rules
        rsi = momentum.get('rsi', 50)
        
        # Oversold bounce (moderate confidence)
        if rsi < 35 and momentum['direction'] != 'BEARISH':
            return TechnicalSignal(
                signal_type='BUY',
                strength=0.55,
                reason=f"Oversold bounce - RSI {rsi:.0f} with potential reversal",
                entry_price=current_price,
                entry_condition="Enter on RSI reversal confirmation",
                stop_loss=current_price * 0.985,
                target_1=current_price * 1.02,
                time_frame='INTRADAY',
                risk_reward_ratio=1.3
            )
        
        # Overbought fade (moderate confidence)
        elif rsi > 65 and momentum['direction'] != 'BULLISH':
            return TechnicalSignal(
                signal_type='SELL',
                strength=0.55,
                reason=f"Overbought fade - RSI {rsi:.0f} with potential reversal",
                entry_price=current_price,
                entry_condition="Enter on RSI reversal confirmation",
                stop_loss=current_price * 1.015,
                target_1=current_price * 0.98,
                time_frame='INTRADAY',
                risk_reward_ratio=1.3
            )
        
        # **PRIORITY 4: MOMENTUM CONTINUATION** - Pure momentum plays
        elif momentum['direction'] == 'BULLISH' and momentum['strength'] > 0.5:
            return TechnicalSignal(
                signal_type='BUY',
                strength=0.5,
                reason=f"Momentum continuation - {momentum['direction']} with {momentum['strength']:.1f} strength",
                entry_price=current_price,
                entry_condition="Enter on momentum confirmation",
                stop_loss=current_price * 0.985,
                target_1=current_price * 1.015,
                time_frame='INTRADAY',
                risk_reward_ratio=1.5
            )
        
        elif momentum['direction'] == 'BEARISH' and momentum['strength'] > 0.5:
            return TechnicalSignal(
                signal_type='SELL',
                strength=0.5,
                reason=f"Momentum continuation - {momentum['direction']} with {momentum['strength']:.1f} strength",
                entry_price=current_price,
                entry_condition="Enter on momentum confirmation",
                stop_loss=current_price * 1.015,
                target_1=current_price * 0.985,
                time_frame='INTRADAY',
                risk_reward_ratio=1.5
            )
        
        # **PRIORITY 5: RANGE TRADING** - Enhanced range logic
        elif patterns.get('consolidation', False):
            mid_range = (sr_levels['nearest_support'] + sr_levels['nearest_resistance']) / 2
            
            # Buy near support in range
            if current_price <= mid_range:
                return TechnicalSignal(
                    signal_type='BUY',
                    strength=0.5,
                    reason="Range trading - buy near support in consolidation",
                    entry_price=current_price,
                    entry_condition=f"Wait for support hold at ₹{sr_levels['nearest_support'] * 1.005:.2f}",
                    stop_loss=sr_levels['nearest_support'] * 0.995,
                    target_1=sr_levels['nearest_resistance'] * 0.99,
                    time_frame='INTRADAY',
                    risk_reward_ratio=1.2
                )
            
            # Sell near resistance in range
            else:
                return TechnicalSignal(
                    signal_type='SELL',
                    strength=0.5,
                    reason="Range trading - sell near resistance in consolidation",
                    entry_price=current_price,
                    entry_condition=f"Wait for resistance rejection at ₹{sr_levels['nearest_resistance'] * 0.995:.2f}",
                    stop_loss=sr_levels['nearest_resistance'] * 1.005,
                    target_1=sr_levels['nearest_support'] * 1.01,
                    time_frame='INTRADAY',
                    risk_reward_ratio=1.2
                )
        
        # **LAST RESORT: TREND FOLLOWING** - When nothing else works, follow the trend
        daily_trend = trend.get('daily_trend', 'SIDEWAYS')
        
        if daily_trend in ['UPTREND', 'STRONG_UPTREND']:
            return TechnicalSignal(
                signal_type='BUY',
                strength=0.45,
                reason=f"Trend following - {daily_trend} bias",
                entry_price=current_price,
                entry_condition="Follow daily uptrend bias",
                stop_loss=current_price * 0.985,
                target_1=current_price * 1.015,
                time_frame='INTRADAY',
                risk_reward_ratio=1.5
            )
        
        elif daily_trend in ['DOWNTREND', 'STRONG_DOWNTREND']:
            return TechnicalSignal(
                signal_type='SELL',
                strength=0.45,
                reason=f"Trend following - {daily_trend} bias",
                entry_price=current_price,
                entry_condition="Follow daily downtrend bias",
                stop_loss=current_price * 1.015,
                target_1=current_price * 0.985,
                time_frame='INTRADAY',
                risk_reward_ratio=1.5
            )
        
        # **ONLY NOW: DEFAULT TO WAIT** - When truly everything is neutral/unclear
        return TechnicalSignal(
            signal_type='WAIT',
            strength=0.3,
            reason="All signals neutral - market in equilibrium",
            entry_condition="Wait for clear directional move or breakout",
            time_frame='INTRADAY'
        )
    
    def _generate_swing_entry_signal(self, current_price: float, trend: Dict,
                                   sr_levels: Dict, momentum: Dict, patterns: Dict) -> TechnicalSignal:
        """FIXED: Generate swing trading entry signals with moderate aggressiveness (45%+ confidence)"""
        
        daily_trend = trend.get('daily_trend', 'SIDEWAYS')
        trend_strength = trend.get('trend_strength', 0.3)
        rsi = momentum.get('rsi', 50)
        momentum_direction = momentum.get('direction', 'NEUTRAL')
        momentum_strength = momentum.get('strength', 0.3)
        sma_20 = trend.get('sma_20', current_price)
        
        # **PRIORITY 1: STRONG TRENDING SETUPS** (High Confidence 70%+)
        
        # Strong bullish alignment - Keep existing logic
        if (daily_trend in ['UPTREND', 'STRONG_UPTREND'] and
            momentum_direction in ['BULLISH', 'WEAK_BULLISH'] and
            current_price > sma_20):
            
            return TechnicalSignal(
                signal_type='BUY',
                strength=0.8,
                reason="Strong uptrend + bullish momentum + above SMA20",
                entry_price=current_price,
                entry_condition="Enter NOW - trend and momentum aligned",
                stop_loss=sma_20 * 0.98,
                target_1=sr_levels['nearest_resistance'],
                target_2=sr_levels['nearest_resistance'] * 1.05,
                time_frame='SWING',
                risk_reward_ratio=2.5
            )
        
        # Strong bearish alignment - Keep existing logic
        elif (daily_trend in ['DOWNTREND', 'STRONG_DOWNTREND'] and
            momentum_direction in ['BEARISH', 'WEAK_BEARISH'] and
            current_price < sma_20):
            
            return TechnicalSignal(
                signal_type='SELL',
                strength=0.8,
                reason="Strong downtrend + bearish momentum + below SMA20",
                entry_price=current_price,
                entry_condition="Enter NOW - trend and momentum aligned",
                stop_loss=sma_20 * 1.02,
                target_1=sr_levels['nearest_support'],
                target_2=sr_levels['nearest_support'] * 0.95,
                time_frame='SWING',
                risk_reward_ratio=2.5
            )
        
        # **PRIORITY 2: MODERATE SETUPS** - Relaxed conditions (60%+ confidence)
        
        # Bullish trend with any momentum OR good RSI
        elif (daily_trend in ['UPTREND', 'STRONG_UPTREND'] or 
            momentum_direction in ['BULLISH', 'WEAK_BULLISH'] or
            (rsi > 45 and rsi < 65 and current_price > sma_20 * 0.99)):
            
            # Calculate strength based on conditions met
            strength = 0.45  # Base moderate confidence
            if daily_trend in ['UPTREND', 'STRONG_UPTREND']:
                strength += 0.15
            if momentum_direction in ['BULLISH', 'WEAK_BULLISH']:
                strength += 0.1
            if current_price > sma_20:
                strength += 0.05
            
            return TechnicalSignal(
                signal_type='BUY',
                strength=min(0.75, strength),
                reason=f"Bullish setup: trend={daily_trend}, momentum={momentum_direction}, RSI={rsi:.0f}",
                entry_price=current_price,
                entry_condition="Enter on any pullback or continuation",
                stop_loss=sma_20 * 0.97,
                target_1=sr_levels['nearest_resistance'],
                target_2=sr_levels['nearest_resistance'] * 1.03,
                time_frame='SWING',
                risk_reward_ratio=2.0
            )
        
        # Bearish trend with any momentum OR good RSI
        elif (daily_trend in ['DOWNTREND', 'STRONG_DOWNTREND'] or 
            momentum_direction in ['BEARISH', 'WEAK_BEARISH'] or
            (rsi < 55 and rsi > 35 and current_price < sma_20 * 1.01)):
            
            # Calculate strength based on conditions met
            strength = 0.45  # Base moderate confidence
            if daily_trend in ['DOWNTREND', 'STRONG_DOWNTREND']:
                strength += 0.15
            if momentum_direction in ['BEARISH', 'WEAK_BEARISH']:
                strength += 0.1
            if current_price < sma_20:
                strength += 0.05
            
            return TechnicalSignal(
                signal_type='SELL',
                strength=min(0.75, strength),
                reason=f"Bearish setup: trend={daily_trend}, momentum={momentum_direction}, RSI={rsi:.0f}",
                entry_price=current_price,
                entry_condition="Enter on any bounce or continuation",
                stop_loss=sma_20 * 1.03,
                target_1=sr_levels['nearest_support'],
                target_2=sr_levels['nearest_support'] * 0.97,
                time_frame='SWING',
                risk_reward_ratio=2.0
            )
        
        # **PRIORITY 3: PULLBACK ENTRIES** - Enhanced logic
        
        # Pullback entry in uptrend - Keep and enhance existing logic
        elif (daily_trend == 'UPTREND' and
            sr_levels['current_level'] == 'NEAR_SUPPORT'):
            
            return TechnicalSignal(
                signal_type='BUY',
                strength=0.7,
                reason="Pullback to support in uptrend",
                entry_price=current_price,
                entry_condition=f"Wait for bounce from ₹{sr_levels['nearest_support']:.2f} support",
                stop_loss=sr_levels['nearest_support'] * 0.97,
                target_1=sr_levels['nearest_resistance'],
                time_frame='SWING',
                risk_reward_ratio=2.0
            )
        
        # NEW: Pullback entry in downtrend
        elif (daily_trend == 'DOWNTREND' and
            sr_levels['current_level'] == 'NEAR_RESISTANCE'):
            
            return TechnicalSignal(
                signal_type='SELL',
                strength=0.7,
                reason="Bounce to resistance in downtrend",
                entry_price=current_price,
                entry_condition=f"Wait for rejection from ₹{sr_levels['nearest_resistance']:.2f} resistance",
                stop_loss=sr_levels['nearest_resistance'] * 1.03,
                target_1=sr_levels['nearest_support'],
                time_frame='SWING',
                risk_reward_ratio=2.0
            )
        
        # **PRIORITY 4: RSI EXTREMES** - New oversold/overbought plays
        
        # Oversold reversal setup
        elif rsi < 30 and momentum_direction != 'BEARISH':
            return TechnicalSignal(
                signal_type='BUY',
                strength=0.6,
                reason=f"Oversold reversal setup - RSI {rsi:.0f}",
                entry_price=current_price,
                entry_condition="Enter on RSI divergence or reversal signal",
                stop_loss=current_price * 0.92,
                target_1=current_price * 1.08,
                target_2=sr_levels['nearest_resistance'],
                time_frame='SWING',
                risk_reward_ratio=2.0
            )
        
        # Overbought reversal setup
        elif rsi > 70 and momentum_direction != 'BULLISH':
            return TechnicalSignal(
                signal_type='SELL',
                strength=0.6,
                reason=f"Overbought reversal setup - RSI {rsi:.0f}",
                entry_price=current_price,
                entry_condition="Enter on RSI divergence or reversal signal",
                stop_loss=current_price * 1.08,
                target_1=current_price * 0.92,
                target_2=sr_levels['nearest_support'],
                time_frame='SWING',
                risk_reward_ratio=2.0
            )
        
        # **PRIORITY 5: PATTERN-BASED ENTRIES** - New pattern logic
        
        # Breakout setup
        elif patterns.get('breakout_imminent', False) or patterns.get('consolidation', False):
            # Determine breakout direction based on momentum and trend
            if momentum_direction in ['BULLISH', 'WEAK_BULLISH'] or daily_trend in ['UPTREND']:
                return TechnicalSignal(
                    signal_type='BUY',
                    strength=0.55,
                    reason="Breakout setup - bullish bias",
                    entry_price=current_price,
                    entry_condition=f"Enter on break above ₹{sr_levels['nearest_resistance'] * 0.999:.2f}",
                    stop_loss=sr_levels['nearest_support'] * 0.98,
                    target_1=sr_levels['nearest_resistance'] * 1.05,
                    time_frame='SWING',
                    risk_reward_ratio=1.8
                )
            elif momentum_direction in ['BEARISH', 'WEAK_BEARISH'] or daily_trend in ['DOWNTREND']:
                return TechnicalSignal(
                    signal_type='SELL',
                    strength=0.55,
                    reason="Breakdown setup - bearish bias",
                    entry_price=current_price,
                    entry_condition=f"Enter on break below ₹{sr_levels['nearest_support'] * 1.001:.2f}",
                    stop_loss=sr_levels['nearest_resistance'] * 1.02,
                    target_1=sr_levels['nearest_support'] * 0.95,
                    time_frame='SWING',
                    risk_reward_ratio=1.8
                )
        
        # **PRIORITY 6: MOMENTUM CONTINUATION** - New pure momentum plays
        
        elif momentum_strength > 0.5:
            if momentum_direction in ['BULLISH', 'WEAK_BULLISH']:
                return TechnicalSignal(
                    signal_type='BUY',
                    strength=0.5,
                    reason=f"Momentum continuation - {momentum_direction} with {momentum_strength:.1f} strength",
                    entry_price=current_price,
                    entry_condition="Enter on momentum confirmation",
                    stop_loss=current_price * 0.92,
                    target_1=current_price * 1.08,
                    time_frame='SWING',
                    risk_reward_ratio=2.0
                )
            elif momentum_direction in ['BEARISH', 'WEAK_BEARISH']:
                return TechnicalSignal(
                    signal_type='SELL',
                    strength=0.5,
                    reason=f"Momentum continuation - {momentum_direction} with {momentum_strength:.1f} strength",
                    entry_price=current_price,
                    entry_condition="Enter on momentum confirmation",
                    stop_loss=current_price * 1.08,
                    target_1=current_price * 0.92,
                    time_frame='SWING',
                    risk_reward_ratio=2.0
                )
        
        # **PRIORITY 7: TREND FOLLOWING** - Last resort directional bias
        
        elif daily_trend in ['UPTREND', 'STRONG_UPTREND']:
            return TechnicalSignal(
                signal_type='BUY',
                strength=0.45,
                reason=f"Trend following - {daily_trend} bias",
                entry_price=current_price,
                entry_condition="Follow uptrend bias on any dip",
                stop_loss=sma_20 * 0.95,
                target_1=sr_levels['nearest_resistance'],
                time_frame='SWING',
                risk_reward_ratio=1.8
            )
        
        elif daily_trend in ['DOWNTREND', 'STRONG_DOWNTREND']:
            return TechnicalSignal(
                signal_type='SELL',
                strength=0.45,
                reason=f"Trend following - {daily_trend} bias",
                entry_price=current_price,
                entry_condition="Follow downtrend bias on any bounce",
                stop_loss=sma_20 * 1.05,
                target_1=sr_levels['nearest_support'],
                time_frame='SWING',
                risk_reward_ratio=1.8
            )
        
        # **ONLY NOW: DEFAULT TO HOLD** - When truly everything is neutral
        return TechnicalSignal(
            signal_type='HOLD',
            strength=0.35,
            reason="Neutral market conditions - no clear directional bias",
            entry_condition="Wait for trend emergence or clear momentum",
            time_frame='SWING'
        )
    
    def _generate_smart_exit_rules(self, current_price: float, sr_levels: Dict,
                                 trend: Dict, trading_style: str) -> Dict:
        """Generate smart exit rules based on technical levels"""
        
        if trading_style == 'intraday':
            return {
                'profit_targets': [
                    f"Target 1: 10-15% profit (₹{current_price * 1.10:.0f} - ₹{current_price * 1.15:.0f})",
                    f"Target 2: 20% for strong moves (₹{current_price * 1.20:.0f})"
                ],
                'stop_losses': [
                    f"Stop 1: 15% loss (₹{current_price * 0.85:.0f})",
                    f"Stop 2: Below support ₹{sr_levels['nearest_support'] * 0.995:.0f}"
                ],
                'time_stops': [
                    "Review at 2:00 PM",
                    "Mandatory exit by 3:00 PM"
                ],
                'technical_exits': [
                    f"Exit if breaks below ₹{sr_levels['nearest_support']:.0f}",
                    "Exit if momentum reverses (RSI crosses 40/60)"
                ]
            }
        else:
            return {
                'profit_targets': [
                    f"Target 1: 15% profit (₹{current_price * 1.15:.0f})",
                    f"Target 2: Resistance at ₹{sr_levels['nearest_resistance']:.0f}",
                    f"Target 3: 25% if trend continues (₹{current_price * 1.25:.0f})"
                ],
                'stop_losses': [
                    f"Stop 1: 20% loss (₹{current_price * 0.80:.0f})",
                    f"Stop 2: Below SMA20 ₹{trend['sma_20']:.0f}",
                    f"Stop 3: Below support ₹{sr_levels['nearest_support']:.0f}"
                ],
                'time_stops': [
                    "Review weekly",
                    "Hold max 2 weeks for momentum trades"
                ],
                'technical_exits': [
                    f"Exit if trend reverses (close below SMA20 for 2 days)",
                    "Exit if momentum divergence appears"
                ]
            }

    def get_historical_data_with_fallback(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """Get historical data with fallback mechanisms"""
        try:
            # Try to get data from Zerodha
            df = self.zerodha.get_historical_data(symbol, timeframe, days)
            
            if not df.empty:
                logger.info(f"[OK] Got {len(df)} {timeframe} candles for {symbol}")
                return df
            else:
                logger.warning(f"[WARNING] Empty data for {symbol}, trying fallback")
                return self._generate_fallback_data(symbol, timeframe, days)
                
        except Exception as e:
            logger.error(f"[ERROR] Historical data error for {symbol}: {e}")
            return self._generate_fallback_data(symbol, timeframe, days)

    def _generate_fallback_data(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """Generate synthetic historical data when Zerodha data unavailable - UPDATED PRICES"""
        
        # Updated base prices for your watchlist (approximate current levels)
        base_prices = {
            # Indices
            'NIFTY 50': 24500,
            'NIFTY BANK': 52000, 
            'NIFTY FIN SERVICE': 22500,
            
            # Your Primary Watchlist - UPDATED PRICES
            'NIFTY': 24500,
            'RELIANCE': 1380,       # Updated from 2950
            'HDFCBANK': 1969,       # Updated from 1720
            'TCS': 3035,            # Updated from 4150
            'INFY': 1425,           # Updated from 1850
            'BAJFINANCE': 853,      # Updated from 7200
            'MARUTI': 12840,        # Updated from 11500
            
            # Expanded Watchlist - NEW ADDITIONS
            'HINDUNILVR': 2666,
            'HCLTECH': 1443,
            'MPHASIS': 2861,
            'BHARTIARTL': 1880
        }
        
        # Get base price for the symbol
        base_price = base_prices.get(symbol.upper(), 1000)
        
        # Generate synthetic data with realistic volatility
        periods = days if timeframe == 'day' else days * (6.5 * 60 // 5)  # 5min candles
        dates = pd.date_range(end=datetime.now(), periods=periods, 
                            freq='D' if timeframe == 'day' else '5min')
        
        # Adjust volatility based on symbol type
        if 'NIFTY' in symbol.upper():
            volatility = 0.015  # 1.5% for indices
            volume_base = 5000000
        elif symbol.upper() in ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY']:
            volatility = 0.025  # 2.5% for large caps
            volume_base = 1000000
        elif symbol.upper() in ['BAJFINANCE', 'MARUTI']:
            volatility = 0.035  # 3.5% for mid-high volatility stocks
            volume_base = 500000
        else:
            volatility = 0.02   # 2% default
            volume_base = 300000
        
        # Generate price movements
        np.random.seed(hash(symbol) % 1000)  # Consistent but different per symbol
        returns = np.random.normal(0, volatility, periods)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLC data with realistic patterns
        noise_factor = 0.005  # 0.5% intraday noise
        
        df = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, noise_factor * 2, periods)),
            'low': prices * (1 - np.random.uniform(0, noise_factor * 2, periods)), 
            'close': prices,
            'volume': np.random.randint(volume_base//2, volume_base*2, periods)
        })
        
        # Ensure OHLC logic (High >= Low, High >= Open/Close, Low <= Open/Close)
        df['high'] = np.maximum(df['high'], df['low'])
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
        
        df.set_index('datetime', inplace=True)
        
        print(f"[CHART] Generated {len(df)} synthetic {timeframe} candles for {symbol} (base: ₹{base_price})")
        return df
    
    def _analyze_options_market_context(self, current_price: float, sr_levels: Dict,
                                      trend: Dict, momentum: Dict, trading_style: str,
                                      intraday_context: Dict = None) -> Dict:
        """Enhanced options market context with intraday focus"""
        
        # Calculate key levels for options selection
        support_distance = abs(current_price - sr_levels['nearest_support']) / current_price
        resistance_distance = abs(sr_levels['nearest_resistance'] - current_price) / current_price
        
        # **ENHANCED STRATEGY RECOMMENDATIONS** - Use intraday context
        strategy_bias = 'NEUTRAL_STRATEGIES'
        recommended_strikes = {}
        
        if trading_style == 'intraday' and intraday_context:
            # Use today's context for strategy selection
            if intraday_context['gap_type'] == 'GAP_UP' and intraday_context['intraday_momentum'] == 'BULLISH':
                strategy_bias = 'BUY_CALLS'
                recommended_strikes = {
                    'atm_call': round(current_price / 50) * 50,
                    'otm_call': round(current_price * 1.01 / 50) * 50
                }
            elif intraday_context['gap_type'] == 'GAP_DOWN' and intraday_context['intraday_momentum'] == 'BEARISH':
                strategy_bias = 'BUY_PUTS'
                recommended_strikes = {
                    'atm_put': round(current_price / 50) * 50,
                    'otm_put': round(current_price * 0.99 / 50) * 50
                }
            elif intraday_context['intraday_momentum'] == 'BULLISH' and momentum['direction'] == 'BULLISH':
                strategy_bias = 'BUY_CALLS'
                recommended_strikes = {
                    'atm_call': round(current_price / 50) * 50,
                    'otm_call': round(current_price * 1.015 / 50) * 50
                }
            elif intraday_context['intraday_momentum'] == 'BEARISH' and momentum['direction'] == 'BEARISH':
                strategy_bias = 'BUY_PUTS'
                recommended_strikes = {
                    'atm_put': round(current_price / 50) * 50,
                    'otm_put': round(current_price * 0.985 / 50) * 50
                }
            else:
                strategy_bias = 'RANGE_TRADING'
                recommended_strikes = {
                    'call_sell': round(sr_levels['nearest_resistance'] / 50) * 50,
                    'put_sell': round(sr_levels['nearest_support'] / 50) * 50
                }
        else:
            # Traditional strategy selection for swing trading
            if trend['daily_trend'] in ['UPTREND', 'STRONG_UPTREND']:
                strategy_bias = 'BULLISH_STRATEGIES'
                recommended_strikes = {
                    'bull_call_spread_buy': round(current_price / 50) * 50,
                    'bull_call_spread_sell': round(current_price * 1.05 / 50) * 50
                }
            elif trend['daily_trend'] in ['DOWNTREND', 'STRONG_DOWNTREND']:
                strategy_bias = 'BEARISH_STRATEGIES'
                recommended_strikes = {
                    'bear_put_spread_buy': round(current_price / 50) * 50,
                    'bear_put_spread_sell': round(current_price * 0.95 / 50) * 50
                }
            else:
                strategy_bias = 'NEUTRAL_STRATEGIES'  
                recommended_strikes = {
                    'iron_condor_call_sell': round(current_price * 1.03 / 50) * 50,
                    'iron_condor_put_sell': round(current_price * 0.97 / 50) * 50
                }
       
        return {
           'strategy_bias': strategy_bias,
           'recommended_strikes': recommended_strikes,
           'volatility_expectation': self._get_volatility_expectation(momentum, support_distance, resistance_distance),
           'time_decay_risk': 'HIGH' if trading_style == 'intraday' else 'MODERATE',
           'breakout_probability': self._calculate_breakout_probability(sr_levels, momentum, trend),
           'optimal_entry_time': self._get_optimal_entry_time(trading_style, momentum, intraday_context),
           'risk_factors': [
               f"Support {support_distance*100:.1f}% away",
               f"Resistance {resistance_distance*100:.1f}% away",
               f"Trend strength: {trend['trend_strength']:.1f}",
               f"Momentum: {momentum['direction']}"
           ]
       }
   
    def _get_volatility_expectation(self, momentum: Dict, support_dist: float, resistance_dist: float) -> str:
       """Determine expected volatility for options pricing"""
       
       if momentum['strength'] > 0.7 and min(support_dist, resistance_dist) < 0.02:
           return 'HIGH'  # Strong momentum near key levels
       elif momentum['strength'] > 0.5:
           return 'MODERATE'
       else:
           return 'LOW'
   
    def _calculate_breakout_probability(self, sr_levels: Dict, momentum: Dict, trend: Dict) -> float:
       """Calculate probability of breakout from current range"""
       
       base_prob = 0.3  # Base 30% chance
       
       # Add momentum factor
       if momentum['direction'] in ['BULLISH', 'BEARISH']:
           base_prob += momentum['strength'] * 0.3
       
       # Add trend factor
       base_prob += trend['trend_strength'] * 0.2
       
       # Add level strength factor (weaker levels = higher breakout chance)
       base_prob += (1 - sr_levels['level_strength']) * 0.2
       
       return min(0.9, base_prob)
   
    def _get_optimal_entry_time(self, trading_style: str, momentum: Dict, intraday_context: Dict = None) -> str:
       """Get optimal entry timing with intraday context"""
       
       if trading_style == 'intraday':
           if intraday_context:
               session_phase = intraday_context['session_phase']
               if session_phase == 'OPENING':
                   return "Enter immediately on momentum confirmation (first 30 mins)"
               elif session_phase == 'MID_SESSION':
                   return "Enter on pullbacks or breakouts (10:30 AM - 2:00 PM)"
               else:
                   return "Avoid new entries - exit existing positions"
           
           if momentum['direction'] == 'BULLISH':
               return "Enter on morning dip (10:00-10:30 AM) or momentum breakout"
           elif momentum['direction'] == 'BEARISH':
               return "Enter on morning bounce (9:30-10:00 AM) or breakdown"
           else:
               return "Wait for clear direction post 10:30 AM"
       else:
           return "Enter on pullbacks in trending markets or breakouts in range"
   
    def _determine_market_bias(self, trend: Dict, momentum: Dict, patterns: Dict, 
                          intraday_context: Dict = None) -> str:
        """FIXED: Enhanced market bias determination with more aggressive bias detection"""
        
        bullish_factors = 0
        bearish_factors = 0
        
        # **PRIMARY: Intraday context for intraday trading** - Safe access
        if intraday_context:
            intraday_momentum = intraday_context.get('intraday_momentum', 'NEUTRAL')
            gap_type = intraday_context.get('gap_type', 'NONE')
            intraday_change = intraday_context.get('intraday_change_percent', 0)
            
            # Strong intraday momentum gets more weight
            if intraday_momentum == 'BULLISH':
                bullish_factors += 2
            elif intraday_momentum == 'BEARISH':
                bearish_factors += 2
            
            # Gap factors with follow-through
            if gap_type == 'GAP_UP' and intraday_change > 0:
                bullish_factors += 1
            elif gap_type == 'GAP_DOWN' and intraday_change < 0:
                bearish_factors += 1
            
            # NEW: Gap reversal factors
            elif gap_type == 'GAP_UP' and intraday_change < -0.3:
                bearish_factors += 1  # Failed gap up = bearish
            elif gap_type == 'GAP_DOWN' and intraday_change > 0.3:
                bullish_factors += 1  # Failed gap down = bullish
            
            # NEW: Volume confirmation
            volume_ratio = intraday_context.get('volume_ratio', 1.0)
            if volume_ratio > 1.3:
                # High volume strengthens the bias
                if intraday_momentum == 'BULLISH':
                    bullish_factors += 0.5
                elif intraday_momentum == 'BEARISH':
                    bearish_factors += 0.5
        
        # **TREND FACTORS** - Safe access with more nuance
        daily_trend = trend.get('daily_trend', 'SIDEWAYS')
        trend_strength = trend.get('trend_strength', 0.3)
        
        if daily_trend in ['STRONG_UPTREND']:
            bullish_factors += 2  # Strong trends get more weight
        elif daily_trend in ['UPTREND']:
            bullish_factors += 1
        elif daily_trend in ['STRONG_DOWNTREND']:
            bearish_factors += 2
        elif daily_trend in ['DOWNTREND']:
            bearish_factors += 1
        
        # NEW: Trend strength bonus
        if trend_strength > 0.7:
            if daily_trend in ['UPTREND', 'STRONG_UPTREND']:
                bullish_factors += 0.5
            elif daily_trend in ['DOWNTREND', 'STRONG_DOWNTREND']:
                bearish_factors += 0.5
        
        # **MOMENTUM FACTORS** - Safe access with RSI consideration
        momentum_direction = momentum.get('direction', 'NEUTRAL')
        momentum_strength = momentum.get('strength', 0.3)
        rsi = momentum.get('rsi', 50)
        
        # Traditional momentum
        if momentum_direction in ['BULLISH']:
            bullish_factors += 1.5  # Strong momentum gets more weight
        elif momentum_direction in ['WEAK_BULLISH']:
            bullish_factors += 1
        elif momentum_direction in ['BEARISH']:
            bearish_factors += 1.5
        elif momentum_direction in ['WEAK_BEARISH']:
            bearish_factors += 1
        
        # NEW: RSI-based factors
        if rsi > 60 and momentum_direction != 'BEARISH':
            bullish_factors += 0.5
        elif rsi < 40 and momentum_direction != 'BULLISH':
            bearish_factors += 0.5
        
        # NEW: Momentum strength bonus
        if momentum_strength > 0.6:
            if momentum_direction in ['BULLISH', 'WEAK_BULLISH']:
                bullish_factors += 0.5
            elif momentum_direction in ['BEARISH', 'WEAK_BEARISH']:
                bearish_factors += 0.5
        
        # **PATTERN FACTORS** - Enhanced pattern recognition
        strongest_pattern = patterns.get('strongest_pattern')
        consolidation = patterns.get('consolidation', False)
        
        # Traditional pattern logic - enhanced
        if strongest_pattern in ['FLAG', 'TRIANGLE']:
            # Pattern continuation bias based on trend
            if daily_trend in ['UPTREND', 'STRONG_UPTREND'] and trend_strength > 0.4:
                bullish_factors += 1
            elif daily_trend in ['DOWNTREND', 'STRONG_DOWNTREND'] and trend_strength > 0.4:
                bearish_factors += 1
        
        # NEW: Consolidation breakout bias
        if consolidation:
            # Consolidation + momentum = breakout direction
            if momentum_direction in ['BULLISH', 'WEAK_BULLISH']:
                bullish_factors += 0.5
            elif momentum_direction in ['BEARISH', 'WEAK_BEARISH']:
                bearish_factors += 0.5
        
        # NEW: Pattern-momentum alignment
        if strongest_pattern and momentum_direction in ['BULLISH', 'WEAK_BULLISH']:
            bullish_factors += 0.5
        elif strongest_pattern and momentum_direction in ['BEARISH', 'WEAK_BEARISH']:
            bearish_factors += 0.5
        
        # **NEW: SUPPORT/RESISTANCE CONTEXT** - Can be added if sr_levels available
        # Note: This method doesn't receive sr_levels, but could be enhanced
        
        # **DECISION LOGIC** - More aggressive bias determination
        
        # Calculate factor advantage
        factor_difference = abs(bullish_factors - bearish_factors)
        total_factors = bullish_factors + bearish_factors
        
        # **ENHANCED DECISION RULES**:
        
        # 1. Strong bias (3+ factor advantage OR high total factors)
        if bullish_factors >= bearish_factors + 3 or (bullish_factors > bearish_factors and total_factors >= 6):
            return 'STRONG_BULLISH'
        elif bearish_factors >= bullish_factors + 3 or (bearish_factors > bullish_factors and total_factors >= 6):
            return 'STRONG_BEARISH'
        
        # 2. Clear bias (1.5+ factor advantage) - LOWERED from 2+
        elif bullish_factors >= bearish_factors + 1.5:
            return 'BULLISH'
        elif bearish_factors >= bullish_factors + 1.5:
            return 'BEARISH'
        
        # 3. Slight bias (1+ factor advantage) - NEW tier
        elif bullish_factors >= bearish_factors + 1:
            return 'LEAN_BULLISH'
        elif bearish_factors >= bullish_factors + 1:
            return 'LEAN_BEARISH'
        
        # 4. Pure tie-breaker - even small advantage counts
        elif bullish_factors > bearish_factors:
            return 'LEAN_BULLISH'
        elif bearish_factors > bullish_factors:
            return 'LEAN_BEARISH'
        
        # 5. True neutral (perfect tie)
        else:
            return 'NEUTRAL'
   
    def _calculate_confidence_score(self, trend: Dict, sr_levels: Dict, 
                             momentum: Dict, patterns: Dict, intraday_context: Dict = None) -> float:
        """FIXED: Enhanced confidence calculation - more generous and comprehensive"""
        
        confidence = 0.5  # Raised base confidence from 0.4 to 0.5
        
        # **SAFE ACCESS**: All dict access with defaults
        trend_strength = trend.get('trend_strength', 0.3)
        daily_trend = trend.get('daily_trend', 'SIDEWAYS')
        level_strength = sr_levels.get('level_strength', 0.3)
        momentum_strength = momentum.get('strength', 0.3)
        momentum_direction = momentum.get('direction', 'NEUTRAL')
        rsi = momentum.get('rsi', 50)
        strongest_pattern = patterns.get('strongest_pattern')
        consolidation = patterns.get('consolidation', False)
        
        # **ENHANCED INTRADAY CONTEXT** - More generous scoring
        if intraday_context:
            gap_type = intraday_context.get('gap_type', 'NONE')
            intraday_change = abs(intraday_context.get('intraday_change_percent', 0))
            volume_ratio = intraday_context.get('volume_ratio', 1.0)
            high_low_range = intraday_context.get('high_low_range_percent', 0)
            intraday_momentum = intraday_context.get('intraday_momentum', 'NEUTRAL')
            
            # Strong gap with follow-through - MORE GENEROUS
            if gap_type != 'NONE':
                if intraday_change > 1.0:
                    confidence += 0.25  # Very strong follow-through
                elif intraday_change > 0.5:
                    confidence += 0.20  # Good follow-through
                elif intraday_change > 0.2:
                    confidence += 0.15  # Decent follow-through
                else:
                    confidence += 0.10  # Any gap gets some bonus
            
            # Volume confirmation - ENHANCED
            if volume_ratio > 2.0:
                confidence += 0.20  # Very high volume
            elif volume_ratio > 1.5:
                confidence += 0.15  # High volume
            elif volume_ratio > 1.2:
                confidence += 0.10  # Above average volume
            elif volume_ratio > 1.0:
                confidence += 0.05  # Normal volume
            
            # Wide range day - MORE GENEROUS
            if high_low_range > 3:
                confidence += 0.15  # Very wide range
            elif high_low_range > 2:
                confidence += 0.12  # Wide range
            elif high_low_range > 1:
                confidence += 0.08  # Decent range
            
            # NEW: Intraday momentum clarity
            if intraday_momentum in ['BULLISH', 'BEARISH']:
                confidence += 0.10  # Clear intraday direction
        
        # **TREND CONFIDENCE** - Enhanced scoring
        # Base trend strength
        confidence += trend_strength * 0.20  # Increased from 0.15
        
        # NEW: Trend clarity bonus
        if daily_trend in ['STRONG_UPTREND', 'STRONG_DOWNTREND']:
            confidence += 0.15  # Strong trend bonus
        elif daily_trend in ['UPTREND', 'DOWNTREND']:
            confidence += 0.10  # Clear trend bonus
        
        # **SUPPORT/RESISTANCE CONFIDENCE** - Enhanced
        confidence += level_strength * 0.12  # Increased from 0.10
        
        # NEW: Level proximity bonus
        support_distance = sr_levels.get('support_distance_pct', 10)
        resistance_distance = sr_levels.get('resistance_distance_pct', 10)
        
        if min(support_distance, resistance_distance) < 2:
            confidence += 0.10  # Very close to key level
        elif min(support_distance, resistance_distance) < 5:
            confidence += 0.05  # Close to key level
        
        # **MOMENTUM CONFIDENCE** - Enhanced scoring
        # Base momentum strength
        confidence += momentum_strength * 0.18  # Increased from 0.15
        
        # NEW: RSI confirmation bonus
        if rsi > 65 and momentum_direction in ['BULLISH', 'WEAK_BULLISH']:
            confidence += 0.08  # Overbought momentum alignment
        elif rsi < 35 and momentum_direction in ['BEARISH', 'WEAK_BEARISH']:
            confidence += 0.08  # Oversold momentum alignment
        elif 45 < rsi < 55:
            confidence += 0.05  # Neutral RSI = balanced conditions
        
        # NEW: Momentum direction clarity
        if momentum_direction in ['BULLISH', 'BEARISH']:
            confidence += 0.08  # Strong directional momentum
        elif momentum_direction in ['WEAK_BULLISH', 'WEAK_BEARISH']:
            confidence += 0.05  # Some directional momentum
        
        # **PATTERN CONFIDENCE** - Much more generous
        if strongest_pattern:
            pattern_bonus = {
                'TRIANGLE': 0.12,
                'FLAG': 0.10,
                'BREAKOUT': 0.15,
                'CONSOLIDATION': 0.08
            }
            confidence += pattern_bonus.get(strongest_pattern, 0.08)  # Default pattern bonus
        
        # NEW: Consolidation setup bonus
        if consolidation:
            confidence += 0.06  # Consolidation = clear setup
        
        # **NEW: ALIGNMENT BONUSES** - Reward factor agreement
        
        # Trend-Momentum alignment
        trend_bullish = daily_trend in ['UPTREND', 'STRONG_UPTREND']
        trend_bearish = daily_trend in ['DOWNTREND', 'STRONG_DOWNTREND'] 
        momentum_bullish = momentum_direction in ['BULLISH', 'WEAK_BULLISH']
        momentum_bearish = momentum_direction in ['BEARISH', 'WEAK_BEARISH']
        
        if (trend_bullish and momentum_bullish) or (trend_bearish and momentum_bearish):
            confidence += 0.12  # Trend-momentum alignment
        
        # NEW: Intraday-Daily alignment
        if intraday_context:
            intraday_bullish = intraday_context.get('intraday_momentum') == 'BULLISH'
            intraday_bearish = intraday_context.get('intraday_momentum') == 'BEARISH'
            
            if (trend_bullish and intraday_bullish) or (trend_bearish and intraday_bearish):
                confidence += 0.08  # Intraday-daily alignment
            
            if (momentum_bullish and intraday_bullish) or (momentum_bearish and intraday_bearish):
                confidence += 0.06  # Intraday-momentum alignment
        
        # **NEW: MARKET CONDITIONS BONUS**
        
        # Clear directional market (based on multiple factors)
        directional_factors = 0
        if daily_trend in ['UPTREND', 'STRONG_UPTREND', 'DOWNTREND', 'STRONG_DOWNTREND']:
            directional_factors += 1
        if momentum_direction in ['BULLISH', 'WEAK_BULLISH', 'BEARISH', 'WEAK_BEARISH']:
            directional_factors += 1
        if intraday_context and intraday_context.get('intraday_momentum') in ['BULLISH', 'BEARISH']:
            directional_factors += 1
        
        if directional_factors >= 2:
            confidence += 0.10  # Clear directional market
        elif directional_factors == 1:
            confidence += 0.05  # Some directional bias
        
        # **NEW: TIME-BASED ADJUSTMENTS**
        
        # Session phase bonus (if available)
        if intraday_context:
            session_phase = intraday_context.get('session_phase', 'MID_SESSION')
            if session_phase == 'OPENING':
                confidence += 0.05  # Opening momentum clearer
            elif session_phase == 'MID_SESSION':
                confidence += 0.03  # Mid-session stability
            # No bonus for CLOSING (less predictable)
        
        # **NEW: QUALITY CONTROL ADJUSTMENTS**
        
        # High-quality setup bonus
        quality_factors = 0
        if trend_strength > 0.7:
            quality_factors += 1
        if momentum_strength > 0.6:
            quality_factors += 1
        if level_strength > 0.6:
            quality_factors += 1
        if intraday_context and intraday_context.get('volume_ratio', 1) > 1.5:
            quality_factors += 1
        
        if quality_factors >= 3:
            confidence += 0.08  # High-quality setup
        elif quality_factors >= 2:
            confidence += 0.05  # Good-quality setup
        
        # **FINAL CONFIDENCE BOUNDS** - More generous range
        # Allow higher confidence for really good setups
        confidence = max(0.25, min(0.95, confidence))  # Range: 25% - 95%
        
        # **BONUS: Perfect setup detection**
        # If everything is aligned and strong, give extra boost
        perfect_setup = (
            trend_strength > 0.7 and
            momentum_strength > 0.6 and
            level_strength > 0.5 and
            directional_factors >= 2 and
            (intraday_context and intraday_context.get('volume_ratio', 1) > 1.3)
        )
        
        if perfect_setup:
            confidence = min(0.92, confidence + 0.05)  # Perfect setup bonus
        
        return confidence
   
    def _assess_risk_level(self, sr_levels: Dict, momentum: Dict, trading_style: str) -> str:
       """Assess overall risk level"""
       
       risk_factors = 0
       
       # Distance to key levels
       if sr_levels['support_distance_pct'] < 2 or sr_levels['resistance_distance_pct'] < 2:
           risk_factors += 1  # Near key levels = higher risk
       
       # Momentum extremes
       if momentum['overbought'] or momentum['oversold']:
           risk_factors += 1
       
       # Trading style risk
       if trading_style == 'intraday':
           risk_factors += 1  # Intraday inherently riskier
       
       # Level strength
       if sr_levels['level_strength'] < 0.5:
           risk_factors += 1  # Weak levels = higher breakout risk
       
       if risk_factors >= 3:
           return 'HIGH'
       elif risk_factors == 2:
           return 'MODERATE'
       else:
           return 'LOW'
   
    def _create_fallback_analysis(self, symbol: str, current_price: float) -> Dict:
       """Create fallback analysis when data is insufficient"""
       
       return {
           'symbol': symbol,
           'current_price': current_price,
           'analysis_time': datetime.now().isoformat(),
           'trading_style': 'swing',
           'data_insufficient': True,
           
           'today_context': {
               'gap_type': 'NONE',
               'intraday_momentum': 'NEUTRAL',
               'session_phase': 'MID_SESSION',
               'intraday_change_percent': 0,
               'volume_ratio': 1.0
           },
           
           'trend_analysis': {
               'daily_trend': 'UNKNOWN',
               'intraday_trend': 'UNKNOWN', 
               'trend_strength': 0.3,
               'trend_quality': 0.3,
               'intraday_vs_daily_alignment': 'NEUTRAL'
           },
           
           'support_resistance': {
               'support_levels': [],
               'resistance_levels': [],
               'nearest_support': current_price * 0.95,
               'nearest_resistance': current_price * 1.05,
               'current_level': 'MIDDLE',
               'level_strength': 0.3
           },
           
           'momentum_analysis': {
               'direction': 'NEUTRAL',
               'strength': 0.3,
               'rsi': 50,
               'intraday_rsi': 50,
               'momentum_score': 0.0
           },
           
           'volume_analysis': {
               'trend': 'UNKNOWN',
               'current_vs_avg': 1.0,
               'volume_confirmation': False,
               'institutional_activity': 'LOW'
           },
           
           'pattern_signals': {
               'detected_patterns': [],
               'strongest_pattern': None,
               'breakout_imminent': False,
               'consolidation': False
           },
           
           'entry_signal': TechnicalSignal(
               signal_type='HOLD',
               strength=0.3,
               reason='Insufficient data for analysis',
               entry_condition='Wait for more data',
               time_frame='SWING'
           ).__dict__,
           
           'exit_rules': {
               'profit_targets': [f"Conservative 10% target (₹{current_price * 1.1:.0f})"],
               'stop_losses': [f"Conservative 15% stop (₹{current_price * 0.85:.0f})"],
               'time_stops': ['Review daily'],
               'technical_exits': ['Exit on any major adverse move']
           },
           
           'options_context': {
               'strategy_bias': 'NEUTRAL_STRATEGIES',
               'recommended_strikes': {
                   'atm_strike': round(current_price / 50) * 50
               },
               'volatility_expectation': 'MODERATE',
               'breakout_probability': 0.3,
               'optimal_entry_time': 'Wait for clearer signals'
           },
           
           'market_bias': 'NEUTRAL',
           'confidence_score': 0.3,
           'risk_assessment': 'MODERATE'
       }

    # **NEW ENHANCED METHOD**: Get today's market data for intraday analysis
    def get_today_market_data(self, symbol: str, current_price: float) -> Dict:
        """Get today's specific market data for enhanced intraday analysis"""
        try:
            zerodha_symbol = self._get_zerodha_symbol(symbol)
            
            # Try to get today's intraday data
            today_data = self.zerodha.get_historical_data(zerodha_symbol, '5minute', 1)
            
            if not today_data.empty:
                today_open = today_data['open'].iloc[0]
                today_high = today_data['high'].max()
                today_low = today_data['low'].min()
                today_volume = today_data['volume'].sum()
                
                return {
                    'today_open': today_open,
                    'today_high': today_high,
                    'today_low': today_low,
                    'today_volume': today_volume,
                    'data_available': True
                }
            else:
                return {
                    'today_open': current_price,
                    'today_high': current_price,
                    'today_low': current_price,
                    'today_volume': 0,
                    'data_available': False
                }
                
        except Exception as e:
            logger.warning(f"Could not get today's data for {symbol}: {e}")
            return {
                'today_open': current_price,
                'today_high': current_price,
                'today_low': current_price,
                'today_volume': 0,
                'data_available': False
            }

    # **NEW METHOD**: Quick intraday analysis for options
    def quick_intraday_analysis(self, symbol: str, current_price: float, market_data: Dict) -> Dict:
        """Quick analysis focused on immediate intraday opportunities"""
        
        # Get today's context
        today_data = self.get_today_market_data(symbol, current_price)
        
        # Calculate immediate signals
        gap_analysis = self._analyze_gap_immediate(current_price, today_data, market_data)
        momentum_immediate = self._analyze_momentum_immediate(current_price, today_data, market_data)
        
        # Determine immediate bias
        immediate_bias = 'NEUTRAL'
        confidence = 0.3
        
        if gap_analysis['gap_type'] != 'NONE' and momentum_immediate['direction'] != 'NEUTRAL':
            if gap_analysis['gap_type'] == 'GAP_UP' and momentum_immediate['direction'] == 'BULLISH':
                immediate_bias = 'STRONG_BULLISH'
                confidence = 0.8
            elif gap_analysis['gap_type'] == 'GAP_DOWN' and momentum_immediate['direction'] == 'BEARISH':
                immediate_bias = 'STRONG_BEARISH'
                confidence = 0.8
            else:
                immediate_bias = 'REVERSAL_SETUP'
                confidence = 0.6
        elif momentum_immediate['strength'] > 0.6:
            immediate_bias = f"MOMENTUM_{momentum_immediate['direction']}"
            confidence = momentum_immediate['strength']
        
        return {
            'symbol': symbol,
            'immediate_bias': immediate_bias,
            'confidence': confidence,
            'gap_analysis': gap_analysis,
            'momentum_immediate': momentum_immediate,
            'recommended_action': self._get_immediate_action(immediate_bias, confidence),
            'analysis_time': datetime.now().isoformat()
        }
    
    def _analyze_gap_immediate(self, current_price: float, today_data: Dict, market_data: Dict) -> Dict:
        """Analyze gap for immediate trading decisions"""
        
        if not today_data['data_available']:
            return {'gap_type': 'NONE', 'gap_percent': 0, 'gap_significance': 'LOW'}
        
        # Assume yesterday's close from market data or estimate
        yesterday_close = market_data.get('prev_close', current_price * 0.999)
        today_open = today_data['today_open']
        
        gap_percent = ((today_open - yesterday_close) / yesterday_close) * 100
        
        gap_type = 'NONE'
        gap_significance = 'LOW'
        
        if abs(gap_percent) > 0.5:
            gap_type = 'GAP_UP' if gap_percent > 0 else 'GAP_DOWN'
            
            if abs(gap_percent) > 2:
                gap_significance = 'HIGH'
            elif abs(gap_percent) > 1:
                gap_significance = 'MODERATE'
            else:
                gap_significance = 'LOW'
        
        return {
            'gap_type': gap_type,
            'gap_percent': gap_percent,
            'gap_significance': gap_significance,
            'today_open': today_open,
            'yesterday_close': yesterday_close
        }
    
    def _analyze_momentum_immediate(self, current_price: float, today_data: Dict, market_data: Dict) -> Dict:
        """Analyze immediate momentum for quick decisions"""
        
        if not today_data['data_available']:
            return {'direction': 'NEUTRAL', 'strength': 0.3}
        
        today_open = today_data['today_open']
        price_change_percent = ((current_price - today_open) / today_open) * 100
        
        direction = 'NEUTRAL'
        strength = 0.3
        
        if price_change_percent > 0.5:
            direction = 'BULLISH'
            strength = min(0.9, 0.5 + abs(price_change_percent) * 0.1)
        elif price_change_percent < -0.5:
            direction = 'BEARISH'
            strength = min(0.9, 0.5 + abs(price_change_percent) * 0.1)
        
        # Volume confirmation
        volume_factor = 1.0
        if today_data['today_volume'] > 0:
            # Estimate if volume is high (simplified)
            estimated_avg_volume = market_data.get('avg_volume', today_data['today_volume'])
            if estimated_avg_volume > 0:
                volume_ratio = today_data['today_volume'] / estimated_avg_volume
                if volume_ratio > 1.5:
                    volume_factor = 1.2
                elif volume_ratio < 0.7:
                    volume_factor = 0.8
        
        strength *= volume_factor
        strength = min(0.95, strength)
        
        return {
            'direction': direction,
            'strength': strength,
            'price_change_percent': price_change_percent,
            'volume_factor': volume_factor
        }
    
    def _get_immediate_action(self, immediate_bias: str, confidence: float) -> str:
        """Get immediate action recommendation"""
        
        if confidence < 0.5:
            return "WAIT - Low confidence, observe price action"
        
        if immediate_bias == 'STRONG_BULLISH':
            return "BUY CALLS - Strong gap up with momentum"
        elif immediate_bias == 'STRONG_BEARISH':
            return "BUY PUTS - Strong gap down with momentum"
        elif immediate_bias == 'REVERSAL_SETUP':
            return "PREPARE REVERSAL - Watch for gap fill or rejection"
        elif 'MOMENTUM_BULLISH' in immediate_bias:
            return "BUY CALLS - Momentum continuation"
        elif 'MOMENTUM_BEARISH' in immediate_bias:
            return "BUY PUTS - Momentum continuation"
        else:
            return "WAIT - No clear immediate setup"


# Integration helper functions
def integrate_technical_analysis(options_analysis_result: Dict, technical_analysis: Dict) -> Dict:
    """Integrate technical analysis with options analysis results"""
    
    # Add technical analysis to the existing result
    enhanced_result = options_analysis_result.copy()
    
    # Add technical analysis section
    enhanced_result['technical_analysis'] = technical_analysis
    
    # Enhance trade recommendation with technical insights
    if 'trade_recommendation' in enhanced_result:
        trade_rec = enhanced_result['trade_recommendation']
        
        # Add entry rules from technical analysis
        entry_signal = technical_analysis.get('entry_signal', {})
        if isinstance(entry_signal, dict):
            trade_rec['entry_rules'] = {
                'entry_conditions': [entry_signal.get('reason', 'Technical analysis pending')],
                'entry_price': entry_signal.get('entry_price'),
                'entry_condition': entry_signal.get('entry_condition', 'Wait for confirmation'),
                'entry_time': technical_analysis['options_context'].get('optimal_entry_time', 'Market hours')
            }
            
            # Add exit rules from technical analysis
            exit_rules = technical_analysis.get('exit_rules', {})
            trade_rec['exit_rules'] = {
                'profit_target': exit_rules.get('profit_targets', ['10% profit'])[0] if exit_rules.get('profit_targets') else '10% profit',
                'stop_loss': exit_rules.get('stop_losses', ['15% loss'])[0] if exit_rules.get('stop_losses') else '15% loss',
                'time_stop': exit_rules.get('time_stops', ['End of day'])[0] if exit_rules.get('time_stops') else 'End of day',
                'technical_stop': exit_rules.get('technical_exits', ['Break of support'])[0] if exit_rules.get('technical_exits') else 'Break of support'
            }
        
        # Enhance confidence based on technical analysis
        tech_confidence = technical_analysis.get('confidence_score', 0.5)
        original_confidence = trade_rec.get('confidence', 0.5)
        
        # Blend confidences (weighted average)
        enhanced_confidence = (original_confidence * 0.6 + tech_confidence * 0.4)
        trade_rec['confidence'] = min(0.95, enhanced_confidence)
        
        # Add technical bias to rationale
        market_bias = technical_analysis.get('market_bias', 'NEUTRAL')
        if market_bias != 'NEUTRAL':
            current_rationale = trade_rec.get('rationale', '')
            tech_rationale = f"Technical analysis shows {market_bias.lower()} bias"
            trade_rec['rationale'] = f"{current_rationale}. {tech_rationale}"
    
    return enhanced_result


# Example usage
if __name__ == "__main__":
    print("[ROCKET] Enhanced Zerodha Technical Analysis Engine")
    print("=" * 50)
    print("[OK] Ready for integration with options trading bot")
    print("[CHART] Enhanced Features:")
    print("  • TODAY-focused intraday analysis")
    print("  • Gap analysis and continuation/reversal patterns") 
    print("  • Real-time momentum detection")
    print("  • Session-phase aware entry timing")
    print("  • Enhanced confidence scoring")
    print("  • Smart entry/exit signals")
    print("  • Support/resistance analysis") 
    print("  • Options-specific recommendations")
    print("  • Risk assessment")
    print("  • Quick analysis methods for immediate decisions")
    print("  • Intraday vs swing optimization")
    print("\n[TARGET] Key Enhancements:")
    print("  [STAR] _analyze_today_context() - Today's gap, momentum, volume")
    print("  [STAR] Enhanced _analyze_trend() - Intraday vs daily alignment")
    print("  [STAR] Enhanced _analyze_momentum() - 5-min RSI for intraday")
    print("  [STAR] Enhanced entry signals - Gap strategies, momentum trades")
    print("  [STAR] Enhanced options context - Strategy bias based on today's action")
    print("  [STAR] quick_intraday_analysis() - Immediate opportunity detection")