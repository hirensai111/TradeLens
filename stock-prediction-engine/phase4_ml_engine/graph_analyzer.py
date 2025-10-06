#!/usr/bin/env python3
"""
Enhanced Graph Analyzer - 30-Day Pattern Analysis Engine
Integrates candlestick patterns, advanced momentum detection, and ML-ready features
"""

import yfinance as yf
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class PatternType(Enum):
    """Enumeration of pattern types for better organization"""
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    RECTANGLE = "rectangle"
    CUP_AND_HANDLE = "cup_and_handle"
    FLAG = "flag_pattern"
    WEDGE = "wedge_pattern"
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"

class CandlestickPattern(Enum):
    """Enumeration of candlestick patterns"""
    HAMMER = "hammer"
    DOJI = "doji"
    SHOOTING_STAR = "shooting_star"
    HANGING_MAN = "hanging_man"
    ENGULFING = "engulfing_pattern"
    PIERCING_LINE = "piercing_line"
    DARK_CLOUD = "dark_cloud_cover"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    HARAMI = "harami"
    MARUBOZU = "marubozu"

@dataclass
class PatternResult:
    """Data class for pattern detection results"""
    name: str
    detected: bool
    strength: float
    reliability: float
    days_forming: int
    completion_prob: float
    signal_type: str
    additional_info: Dict = None

class GraphAnalyzer:
    """
    Enhanced 30-day candlestick pattern analyzer with improved accuracy and features
    """
    
    def __init__(self, use_cache: bool = True):
        self.cache = {} if use_cache else None
        self.pattern_thresholds = {
            'triangle_min_days': 10,
            'breakout_volume_multiplier': 1.5,
            'momentum_acceleration_threshold': 0.7,
            'support_resistance_touch_min': 2,
            'pattern_min_strength': 0.3,
            'volume_spike_threshold': 2.0
        }
        
        # Enhanced pattern weights for scoring
        self.pattern_weights = {
            'chart_patterns': 0.3,
            'candlestick_patterns': 0.25,
            'momentum_signals': 0.25,
            'volume_signals': 0.2
        }
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types"""
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        else:
            return obj
    
    
    def analyze_ticker(self, ticker: str, days: int = 30, include_extended_analysis: bool = True) -> Dict:
        """
        Enhanced main analysis function with comprehensive pattern detection
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            days: Analysis period (default 30)
            include_extended_analysis: Include advanced metrics (default True)
            
        Returns:
            Comprehensive dictionary with enhanced pattern analysis
        """
        print(f"📊 Analyzing 30-day patterns for {ticker}...")
        
        try:
            # Step 1: Get 30-day data with validation
            raw_data = self._get_30day_data(ticker, days)
            if raw_data is None or len(raw_data) < 10:
                return self._create_fallback_analysis(ticker, "Insufficient data")
            
            # Step 2: Enhanced preprocessing
            processed_data = self._preprocess_data(raw_data)
            
            # Step 3: Comprehensive pattern detection
            patterns = self._detect_patterns(processed_data)
            
            # Step 4: Enhanced candlestick analysis
            candlestick_analysis = self._detect_candlestick_patterns(processed_data)
            
            # Step 5: Advanced momentum analysis
            momentum = self._analyze_momentum_context(processed_data)
            
            # Step 6: Volume profile analysis
            volume_analysis = self._analyze_volume_patterns(processed_data)
            
            # Step 7: Dynamic support/resistance levels
            levels = self._identify_key_levels(processed_data)
            
            # Step 8: Breakout and trend analysis
            breakout = self._analyze_breakout_potential(processed_data, patterns, volume_analysis, levels)
            
            # Step 9: Extended analysis (if requested)
            extended_metrics = {}
            if include_extended_analysis:
                extended_metrics = self._calculate_extended_metrics(processed_data, patterns, candlestick_analysis)
            
            # Step 10: Compile comprehensive analysis with risk metrics
            analysis = self._compile_analysis(
                ticker, patterns, candlestick_analysis, momentum, 
                volume_analysis, levels, breakout, processed_data, extended_metrics
            )
            
            # Step 11: Add AI/ML ready features
            analysis['ml_features'] = self._extract_ml_features(analysis, processed_data)
            
            print(f"✅ Pattern analysis complete for {ticker}")
            return analysis
            
        except Exception as e:
            print(f"❌ Graph analysis error for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_analysis(ticker, str(e))
    
    def _get_30day_data(self, ticker: str, days: int) -> Optional[pd.DataFrame]:
        """Enhanced data retrieval with better error handling and validation"""
        try:
            # Check cache first
            if self.cache is not None:
                cache_key = f"{ticker}_{days}_{datetime.now().strftime('%Y%m%d')}"
                if cache_key in self.cache:
                    return self.cache[cache_key]
            
            # Fetch data with retry logic
            stock = yf.Ticker(ticker)
            
            # Try different periods if initial fetch fails
            for period in [f"{days + 10}d", "2mo", "3mo"]:
                try:
                    data = stock.history(period=period, interval="1d")
                    if not data.empty:
                        break
                except:
                    continue
            
            if data.empty:
                print(f"   ⚠️ No data returned for {ticker}")
                return None
            
            # Validate data quality
            data = data.dropna()
            
            # Keep only last 'days' trading days
            data = data.tail(days)
            
            # Additional validation
            if len(data) < 10:
                print(f"   ⚠️ Insufficient data points: {len(data)}")
                return None
            
            # Cache the data
            if self.cache is not None:
                self.cache[cache_key] = data
            
            print(f"   📈 Retrieved {len(data)} days of data for {ticker}")
            return data
            
        except Exception as e:
            print(f"   ❌ Data retrieval failed for {ticker}: {e}")
            return None
    
    def _preprocess_data(self, data: pd.DataFrame) -> Dict:
        """Enhanced preprocessing with additional technical indicators"""
        
        processed = {
            'raw_data': data,
            'dates': data.index.tolist(),
            'opens': data['Open'].values,
            'highs': data['High'].values,
            'lows': data['Low'].values,
            'closes': data['Close'].values,
            'volumes': data['Volume'].values,
            'length': len(data)
        }
        
        # Calculate derived metrics
        processed['daily_ranges'] = processed['highs'] - processed['lows']
        processed['body_sizes'] = np.abs(processed['closes'] - processed['opens'])
        processed['upper_wicks'] = processed['highs'] - np.maximum(processed['opens'], processed['closes'])
        processed['lower_wicks'] = np.minimum(processed['opens'], processed['closes']) - processed['lows']
        
        # Daily returns and changes
        processed['daily_returns'] = np.concatenate([[0], np.diff(processed['closes']) / processed['closes'][:-1]])
        processed['daily_changes'] = np.concatenate([[0], np.diff(processed['closes'])])
        
        # Enhanced volume metrics
        processed['volume_sma_5'] = self._simple_moving_average(processed['volumes'], 5)
        processed['volume_sma_10'] = self._simple_moving_average(processed['volumes'], 10)
        processed['volume_sma_20'] = self._simple_moving_average(processed['volumes'], 20)
        processed['volume_ratio'] = processed['volumes'] / processed['volume_sma_20']
        
        # Price moving averages
        processed['sma_5'] = self._simple_moving_average(processed['closes'], 5)
        processed['sma_10'] = self._simple_moving_average(processed['closes'], 10)
        processed['sma_20'] = self._simple_moving_average(processed['closes'], 20)
        processed['ema_9'] = self._exponential_moving_average(processed['closes'], 9)
        processed['ema_21'] = self._exponential_moving_average(processed['closes'], 21)
        
        # RSI calculation
        processed['rsi'] = self._calculate_rsi(processed['closes'], 14)
        
        # Bollinger Bands
        processed['bb_upper'], processed['bb_middle'], processed['bb_lower'] = self._calculate_bollinger_bands(
            processed['closes'], 20, 2
        )
        
        # ATR (Average True Range)
        processed['atr'] = self._calculate_atr(processed['highs'], processed['lows'], processed['closes'], 14)
        
        # Gap analysis
        processed['gaps'] = self._identify_gaps(processed)
        
        # Trend strength
        processed['trend_strength'] = self._calculate_trend_strength(processed['closes'])
        
        print(f"   🔧 Preprocessed {processed['length']} days of data with enhanced indicators")
        return processed
    
    def _exponential_moving_average(self, values: np.ndarray, period: int) -> np.ndarray:
        """Calculate exponential moving average"""
        if len(values) < period:
            return np.full(len(values), np.nan)
        
        ema = np.full(len(values), np.nan)
        ema[period-1] = np.mean(values[:period])
        
        multiplier = 2 / (period + 1)
        for i in range(period, len(values)):
            ema[i] = (values[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)
        
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi = np.zeros_like(prices)
        rsi[:period] = 50.0
        rsi[period] = 100 - 100 / (1 + rs)
        
        for i in range(period + 1, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 100
            rsi[i] = 100 - 100 / (1 + rs)
        
        return rsi
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        middle = self._simple_moving_average(prices, period)
        std = np.array([np.std(prices[max(0, i-period+1):i+1]) if i >= period-1 else np.nan for i in range(len(prices))])
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Average True Range"""
        if len(highs) < 2:
            return np.full(len(highs), 0.0)
        
        tr = np.zeros(len(highs))
        tr[0] = highs[0] - lows[0]
        
        for i in range(1, len(highs)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr[i] = max(hl, hc, lc)
        
        atr = self._simple_moving_average(tr, period)
        return atr
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate overall trend strength using linear regression"""
        if len(prices) < 5:
            return 0.0
        
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Normalize slope
        normalized_slope = slope / np.mean(prices) * 100
        
        # Combine slope and R-squared for trend strength
        trend_strength = normalized_slope * r_squared
        
        return trend_strength
    
    def _detect_candlestick_patterns(self, data: Dict) -> Dict:
        """Enhanced candlestick pattern detection with all patterns"""
        
        candlestick_analysis = {
            'detected_patterns': [],
            'total_patterns_found': 0,
            'strongest_pattern': None,
            'reversal_signals': [],
            'continuation_signals': [],
            'pattern_clusters': []
        }
        
        opens = data['opens']
        highs = data['highs']
        lows = data['lows']
        closes = data['closes']
        volumes = data['volumes']
        length = len(closes)
        
        if length < 3:
            return candlestick_analysis
        
        # Analyze last 10 days for candlestick patterns
        lookback = min(10, length - 2)
        
        for i in range(length - lookback, length):
            if i < 2:  # Need at least 2 previous days for context
                continue
            
            # Single candle patterns
            patterns_to_check = [
                self._detect_hammer(opens, highs, lows, closes, i),
                self._detect_doji(opens, highs, lows, closes, i),
                self._detect_shooting_star(opens, highs, lows, closes, i),
                self._detect_hanging_man(opens, highs, lows, closes, i),
                self._detect_marubozu(opens, highs, lows, closes, i)
            ]
            
            # Two candle patterns
            if i >= 1:
                patterns_to_check.extend([
                    self._detect_engulfing_pattern(opens, highs, lows, closes, i-1, i),
                    self._detect_piercing_line(opens, highs, lows, closes, i-1, i),
                    self._detect_dark_cloud_cover(opens, highs, lows, closes, i-1, i),
                    self._detect_harami(opens, highs, lows, closes, i-1, i)
                ])
            
            # Three candle patterns
            if i >= 2:
                patterns_to_check.extend([
                    self._detect_morning_star(opens, highs, lows, closes, i-2, i-1, i),
                    self._detect_evening_star(opens, highs, lows, closes, i-2, i-1, i),
                    self._detect_three_white_soldiers(opens, highs, lows, closes, i-2, i-1, i),
                    self._detect_three_black_crows(opens, highs, lows, closes, i-2, i-1, i)
                ])
            
            # Add detected patterns
            for pattern in patterns_to_check:
                if pattern['detected']:
                    candlestick_analysis['detected_patterns'].append(pattern)
                    if 'reversal' in pattern['signal_type']:
                        candlestick_analysis['reversal_signals'].append(pattern)
                    elif 'continuation' in pattern['signal_type']:
                        candlestick_analysis['continuation_signals'].append(pattern)
        
        # Detect pattern clusters (multiple patterns at same time)
        candlestick_analysis['pattern_clusters'] = self._identify_pattern_clusters(
            candlestick_analysis['detected_patterns']
        )
        
        # Find strongest pattern
        if candlestick_analysis['detected_patterns']:
            strongest = max(candlestick_analysis['detected_patterns'], key=lambda x: x['strength'])
            candlestick_analysis['strongest_pattern'] = strongest['name']
        
        candlestick_analysis['total_patterns_found'] = len(candlestick_analysis['detected_patterns'])
        
        return candlestick_analysis
    
    def _detect_hammer(self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, i: int) -> Dict:
        """Detect Hammer candlestick pattern (bullish reversal)"""
        pattern = {
            'name': 'hammer',
            'detected': False,
            'strength': 0.0,
            'reliability': 0.0,
            'days_forming': 1,
            'completion_prob': 0.0,
            'signal_type': 'bullish_reversal',
            'day_index': i
        }
        
        if i < 1:
            return pattern
        
        # Current candle metrics
        open_price = opens[i]
        high_price = highs[i]
        low_price = lows[i]
        close_price = closes[i]
        
        # Calculate candle components
        body_size = abs(close_price - open_price)
        upper_wick = high_price - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low_price
        total_range = high_price - low_price
        
        if total_range == 0:
            return pattern
        
        # Hammer criteria
        body_ratio = body_size / total_range
        lower_wick_ratio = lower_wick / total_range
        upper_wick_ratio = upper_wick / total_range
        
        if (body_ratio <= 0.3 and  # Small body
            lower_wick_ratio >= 0.6 and  # Long lower wick
            upper_wick_ratio <= 0.1):  # Small upper wick
            
            # Check for downward trend context
            trend_context = 0.0
            if i >= 4:
                recent_trend = (closes[i-1] - closes[i-4]) / closes[i-4]
                if recent_trend < -0.02:  # Downtrend
                    trend_context = min(1.0, abs(recent_trend) * 10)
            
            if trend_context > 0:
                pattern['detected'] = True
                pattern['strength'] = min(1.0, 0.4 + lower_wick_ratio * 0.3 + trend_context * 0.3)
                pattern['reliability'] = pattern['strength'] * 0.8
                pattern['completion_prob'] = pattern['strength'] * 0.7
        
        return pattern
    
    def _detect_doji(self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, i: int) -> Dict:
        """Detect Doji candlestick pattern (indecision/reversal)"""
        pattern = {
            'name': 'doji',
            'detected': False,
            'strength': 0.0,
            'reliability': 0.0,
            'days_forming': 1,
            'completion_prob': 0.0,
            'signal_type': 'reversal_indecision',
            'day_index': i
        }
        
        open_price = opens[i]
        high_price = highs[i]
        low_price = lows[i]
        close_price = closes[i]
        
        body_size = abs(close_price - open_price)
        total_range = high_price - low_price
        
        if total_range == 0:
            return pattern
        
        # Doji criteria
        body_ratio = body_size / total_range
        
        if body_ratio <= 0.05:
            pattern['detected'] = True
            
            # Calculate wick symmetry
            upper_wick = high_price - max(open_price, close_price)
            lower_wick = min(open_price, close_price) - low_price
            
            if total_range > 0:
                wick_symmetry = 1.0 - abs(upper_wick - lower_wick) / total_range
            else:
                wick_symmetry = 1.0
            
            # Strength based on how perfect the doji is
            pattern['strength'] = min(1.0, 0.5 + (1 - body_ratio * 20) * 0.3 + wick_symmetry * 0.2)
            pattern['reliability'] = pattern['strength'] * 0.6
            pattern['completion_prob'] = pattern['strength'] * 0.5
        
        return pattern
    
    def _detect_shooting_star(self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, i: int) -> Dict:
        """Detect Shooting Star candlestick pattern (bearish reversal)"""
        pattern = {
            'name': 'shooting_star',
            'detected': False,
            'strength': 0.0,
            'reliability': 0.0,
            'days_forming': 1,
            'completion_prob': 0.0,
            'signal_type': 'bearish_reversal',
            'day_index': i
        }
        
        if i < 1:
            return pattern
        
        open_price = opens[i]
        high_price = highs[i]
        low_price = lows[i]
        close_price = closes[i]
        
        body_size = abs(close_price - open_price)
        upper_wick = high_price - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low_price
        total_range = high_price - low_price
        
        if total_range == 0:
            return pattern
        
        # Shooting Star criteria
        body_ratio = body_size / total_range
        upper_wick_ratio = upper_wick / total_range
        lower_wick_ratio = lower_wick / total_range
        
        if (body_ratio <= 0.3 and
            upper_wick_ratio >= 0.6 and
            lower_wick_ratio <= 0.1):
            
            # Check for upward trend context
            trend_context = 0.0
            if i >= 4:
                recent_trend = (closes[i-1] - closes[i-4]) / closes[i-4]
                if recent_trend > 0.02:  # Uptrend
                    trend_context = min(1.0, recent_trend * 10)
            
            if trend_context > 0:
                pattern['detected'] = True
                pattern['strength'] = min(1.0, 0.4 + upper_wick_ratio * 0.3 + trend_context * 0.3)
                pattern['reliability'] = pattern['strength'] * 0.8
                pattern['completion_prob'] = pattern['strength'] * 0.7
        
        return pattern
    
    def _detect_hanging_man(self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, i: int) -> Dict:
        """Detect Hanging Man candlestick pattern (bearish reversal)"""
        pattern = {
            'name': 'hanging_man',
            'detected': False,
            'strength': 0.0,
            'reliability': 0.0,
            'days_forming': 1,
            'completion_prob': 0.0,
            'signal_type': 'bearish_reversal',
            'day_index': i
        }
        
        if i < 1:
            return pattern
        
        open_price = opens[i]
        high_price = highs[i]
        low_price = lows[i]
        close_price = closes[i]
        
        body_size = abs(close_price - open_price)
        upper_wick = high_price - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low_price
        total_range = high_price - low_price
        
        if total_range == 0:
            return pattern
        
        # Same shape as hammer but in uptrend
        body_ratio = body_size / total_range
        lower_wick_ratio = lower_wick / total_range
        upper_wick_ratio = upper_wick / total_range
        
        if (body_ratio <= 0.3 and
            lower_wick_ratio >= 0.6 and
            upper_wick_ratio <= 0.1):
            
            # Check for upward trend context (key difference from hammer)
            trend_context = 0.0
            if i >= 4:
                recent_trend = (closes[i-1] - closes[i-4]) / closes[i-4]
                if recent_trend > 0.02:  # Uptrend required
                    trend_context = min(1.0, recent_trend * 10)
                    
                    pattern['detected'] = True
                    pattern['strength'] = min(1.0, 0.4 + lower_wick_ratio * 0.3 + trend_context * 0.3)
                    pattern['reliability'] = pattern['strength'] * 0.7
                    pattern['completion_prob'] = pattern['strength'] * 0.6
        
        return pattern
    
    def _detect_marubozu(self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, i: int) -> Dict:
        """Detect Marubozu pattern (strong continuation)"""
        pattern = {
            'name': 'marubozu',
            'detected': False,
            'strength': 0.0,
            'reliability': 0.0,
            'days_forming': 1,
            'completion_prob': 0.0,
            'signal_type': 'continuation',
            'marubozu_type': None,
            'day_index': i
        }
        
        open_price = opens[i]
        high_price = highs[i]
        low_price = lows[i]
        close_price = closes[i]
        
        body_size = abs(close_price - open_price)
        total_range = high_price - low_price
        
        if total_range == 0:
            return pattern
        
        # Marubozu has almost no wicks
        upper_wick = high_price - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low_price
        
        wick_ratio = (upper_wick + lower_wick) / total_range
        body_ratio = body_size / total_range
        
        if wick_ratio < 0.05 and body_ratio > 0.95:
            pattern['detected'] = True
            pattern['marubozu_type'] = 'bullish' if close_price > open_price else 'bearish'
            pattern['signal_type'] = f"{pattern['marubozu_type']}_continuation"
            pattern['strength'] = min(1.0, 0.7 + body_ratio * 0.3)
            pattern['reliability'] = pattern['strength'] * 0.9  # Very reliable pattern
            pattern['completion_prob'] = pattern['strength'] * 0.85
        
        return pattern
    
    def _detect_engulfing_pattern(self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, i1: int, i2: int) -> Dict:
        """Detect Bullish/Bearish Engulfing pattern (2-day reversal)"""
        pattern = {
            'name': 'engulfing_pattern',
            'detected': False,
            'strength': 0.0,
            'reliability': 0.0,
            'days_forming': 2,
            'completion_prob': 0.0,
            'signal_type': 'reversal',
            'engulfing_type': None,
            'day_index': i2
        }
        
        # Day 1 and Day 2 data
        open1, high1, low1, close1 = opens[i1], highs[i1], lows[i1], closes[i1]
        open2, high2, low2, close2 = opens[i2], highs[i2], lows[i2], closes[i2]
        
        body1 = abs(close1 - open1)
        body2 = abs(close2 - open2)
        
        # Bullish Engulfing
        if (close1 < open1 and  # Day 1: Red
            close2 > open2 and  # Day 2: Green
            open2 <= close1 and  # Day 2 opens at or below day 1 close
            close2 >= open1):    # Day 2 closes at or above day 1 open
            
            pattern['detected'] = True
            pattern['engulfing_type'] = 'bullish'
            pattern['signal_type'] = 'bullish_reversal'
            
            # Strength based on size ratio and completeness
            size_ratio = body2 / body1 if body1 > 0 else 2.0
            engulf_completeness = min(1.0, (close2 - open1) / (open1 - close1)) if open1 > close1 else 1.0
            
            pattern['strength'] = min(1.0, 0.5 + min(0.3, (size_ratio - 1) * 0.15) + engulf_completeness * 0.2)
            pattern['reliability'] = pattern['strength'] * 0.85
            pattern['completion_prob'] = pattern['strength'] * 0.8
        
        # Bearish Engulfing
        elif (close1 > open1 and  # Day 1: Green
              close2 < open2 and  # Day 2: Red
              open2 >= close1 and  # Day 2 opens at or above day 1 close
              close2 <= open1):    # Day 2 closes at or below day 1 open
            
            pattern['detected'] = True
            pattern['engulfing_type'] = 'bearish'
            pattern['signal_type'] = 'bearish_reversal'
            
            size_ratio = body2 / body1 if body1 > 0 else 2.0
            engulf_completeness = min(1.0, (open1 - close2) / (close1 - open1)) if close1 > open1 else 1.0
            
            pattern['strength'] = min(1.0, 0.5 + min(0.3, (size_ratio - 1) * 0.15) + engulf_completeness * 0.2)
            pattern['reliability'] = pattern['strength'] * 0.85
            pattern['completion_prob'] = pattern['strength'] * 0.8
        
        return pattern
    
    def _detect_piercing_line(self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, i1: int, i2: int) -> Dict:
        """Detect Piercing Line pattern (bullish reversal)"""
        pattern = {
            'name': 'piercing_line',
            'detected': False,
            'strength': 0.0,
            'reliability': 0.0,
            'days_forming': 2,
            'completion_prob': 0.0,
            'signal_type': 'bullish_reversal',
            'day_index': i2
        }
        
        open1, high1, low1, close1 = opens[i1], highs[i1], lows[i1], closes[i1]
        open2, high2, low2, close2 = opens[i2], highs[i2], lows[i2], closes[i2]
        
        midpoint1 = (open1 + close1) / 2
        
        if (close1 < open1 and      # Day 1: Red
            close2 > open2 and      # Day 2: Green
            open2 < low1 and        # Day 2 gaps down
            close2 > midpoint1 and  # Day 2 closes above midpoint
            close2 < open1):        # But below day 1 open
            
            pattern['detected'] = True
            
            # Strength based on penetration depth
            penetration = (close2 - midpoint1) / (open1 - close1) if open1 > close1 else 0
            pattern['strength'] = min(1.0, 0.5 + penetration * 0.5)
            pattern['reliability'] = pattern['strength'] * 0.8
            pattern['completion_prob'] = pattern['strength'] * 0.75
        
        return pattern
    
    def _detect_dark_cloud_cover(self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, i1: int, i2: int) -> Dict:
        """Detect Dark Cloud Cover pattern (bearish reversal)"""
        pattern = {
            'name': 'dark_cloud_cover',
            'detected': False,
            'strength': 0.0,
            'reliability': 0.0,
            'days_forming': 2,
            'completion_prob': 0.0,
            'signal_type': 'bearish_reversal',
            'day_index': i2
        }
        
        open1, high1, low1, close1 = opens[i1], highs[i1], lows[i1], closes[i1]
        open2, high2, low2, close2 = opens[i2], highs[i2], lows[i2], closes[i2]
        
        midpoint1 = (open1 + close1) / 2
        
        if (close1 > open1 and      # Day 1: Green
            close2 < open2 and      # Day 2: Red
            open2 > high1 and       # Day 2 gaps up
            close2 < midpoint1 and  # Day 2 closes below midpoint
            close2 > open1):        # But above day 1 open
            
            pattern['detected'] = True
            
            # Strength based on penetration
            penetration = (midpoint1 - close2) / (close1 - open1) if close1 > open1 else 0
            pattern['strength'] = min(1.0, 0.5 + penetration * 0.5)
            pattern['reliability'] = pattern['strength'] * 0.8
            pattern['completion_prob'] = pattern['strength'] * 0.75
        
        return pattern
    
    def _detect_harami(self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, i1: int, i2: int) -> Dict:
        """Detect Harami pattern (potential reversal)"""
        pattern = {
            'name': 'harami',
            'detected': False,
            'strength': 0.0,
            'reliability': 0.0,
            'days_forming': 2,
            'completion_prob': 0.0,
            'signal_type': 'reversal',
            'harami_type': None,
            'day_index': i2
        }
        
        open1, high1, low1, close1 = opens[i1], highs[i1], lows[i1], closes[i1]
        open2, high2, low2, close2 = opens[i2], highs[i2], lows[i2], closes[i2]
        
        # Day 2 body should be completely inside day 1 body
        day1_body_high = max(open1, close1)
        day1_body_low = min(open1, close1)
        day2_body_high = max(open2, close2)
        day2_body_low = min(open2, close2)
        
        if (day2_body_high < day1_body_high and 
            day2_body_low > day1_body_low and
            abs(close1 - open1) > abs(close2 - open2) * 2):  # Day 1 body significantly larger
            
            pattern['detected'] = True
            
            # Bullish harami: Day 1 red, Day 2 green
            if close1 < open1 and close2 > open2:
                pattern['harami_type'] = 'bullish'
                pattern['signal_type'] = 'bullish_reversal'
            # Bearish harami: Day 1 green, Day 2 red
            elif close1 > open1 and close2 < open2:
                pattern['harami_type'] = 'bearish'
                pattern['signal_type'] = 'bearish_reversal'
            
            # Strength based on size ratio
            size_ratio = abs(close2 - open2) / abs(close1 - open1) if abs(close1 - open1) > 0 else 0
            pattern['strength'] = min(1.0, 0.6 + (1 - size_ratio) * 0.4)
            pattern['reliability'] = pattern['strength'] * 0.7  # Needs confirmation
            pattern['completion_prob'] = pattern['strength'] * 0.65
        
        return pattern
    
    def _detect_morning_star(self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, i1: int, i2: int, i3: int) -> Dict:
        """Detect Morning Star pattern (3-day bullish reversal)"""
        pattern = {
            'name': 'morning_star',
            'detected': False,
            'strength': 0.0,
            'reliability': 0.0,
            'days_forming': 3,
            'completion_prob': 0.0,
            'signal_type': 'bullish_reversal',
            'day_index': i3
        }
        
        open1, high1, low1, close1 = opens[i1], highs[i1], lows[i1], closes[i1]
        open2, high2, low2, close2 = opens[i2], highs[i2], lows[i2], closes[i2]
        open3, high3, low3, close3 = opens[i3], highs[i3], lows[i3], closes[i3]
        
        body1 = abs(close1 - open1)
        body2 = abs(close2 - open2)
        body3 = abs(close3 - open3)
        
        # Morning star criteria
        if (close1 < open1 and              # Day 1: Large red
            body1 > body2 * 2 and           # Day 1 significantly larger than day 2
            max(open2, close2) < close1 and # Day 2: Gaps down
            body2 < body1 * 0.3 and         # Day 2: Small body (star)
            close3 > open3 and              # Day 3: Green
            close3 > (open1 + close1) / 2 and # Day 3: Closes above day 1 midpoint
            open3 > max(open2, close2)):   # Day 3: Gaps up from star
            
            pattern['detected'] = True
            
            # Strength calculation
            gap_down = (close1 - max(open2, close2)) / close1
            gap_up = (open3 - max(open2, close2)) / open3
            penetration = (close3 - (open1 + close1) / 2) / (open1 - close1) if open1 > close1 else 0
            
            pattern['strength'] = min(1.0, 0.5 + gap_down * 10 + gap_up * 10 + penetration * 0.3)
            pattern['reliability'] = pattern['strength'] * 0.9  # Very reliable
            pattern['completion_prob'] = pattern['strength'] * 0.85
        
        return pattern
    
    def _detect_evening_star(self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, i1: int, i2: int, i3: int) -> Dict:
        """Detect Evening Star pattern (3-day bearish reversal)"""
        pattern = {
            'name': 'evening_star',
            'detected': False,
            'strength': 0.0,
            'reliability': 0.0,
            'days_forming': 3,
            'completion_prob': 0.0,
            'signal_type': 'bearish_reversal',
            'day_index': i3
        }
        
        open1, high1, low1, close1 = opens[i1], highs[i1], lows[i1], closes[i1]
        open2, high2, low2, close2 = opens[i2], highs[i2], lows[i2], closes[i2]
        open3, high3, low3, close3 = opens[i3], highs[i3], lows[i3], closes[i3]
        
        body1 = abs(close1 - open1)
        body2 = abs(close2 - open2)
        body3 = abs(close3 - open3)
        
        # Evening star criteria
        if (close1 > open1 and              # Day 1: Large green
            body1 > body2 * 2 and           # Day 1 significantly larger
            min(open2, close2) > close1 and # Day 2: Gaps up
            body2 < body1 * 0.3 and         # Day 2: Small body (star)
            close3 < open3 and              # Day 3: Red
            close3 < (open1 + close1) / 2 and # Day 3: Closes below day 1 midpoint
            open3 < min(open2, close2)):   # Day 3: Gaps down from star
            
            pattern['detected'] = True
            
            # Strength calculation
            gap_up = (min(open2, close2) - close1) / close1
            gap_down = (min(open2, close2) - open3) / min(open2, close2)
            penetration = ((open1 + close1) / 2 - close3) / (close1 - open1) if close1 > open1 else 0
            
            pattern['strength'] = min(1.0, 0.5 + gap_up * 10 + gap_down * 10 + penetration * 0.3)
            pattern['reliability'] = pattern['strength'] * 0.9
            pattern['completion_prob'] = pattern['strength'] * 0.85
        
        return pattern
    
    def _detect_three_white_soldiers(self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, i1: int, i2: int, i3: int) -> Dict:
        """Detect Three White Soldiers pattern (3-day bullish continuation)"""
        pattern = {
            'name': 'three_white_soldiers',
            'detected': False,
            'strength': 0.0,
            'reliability': 0.0,
            'days_forming': 3,
            'completion_prob': 0.0,
            'signal_type': 'bullish_continuation',
            'day_index': i3
        }
        
        # Three consecutive green candles
        if (closes[i1] > opens[i1] and      # Day 1: Green
            closes[i2] > opens[i2] and      # Day 2: Green
            closes[i3] > opens[i3] and      # Day 3: Green
            closes[i2] > closes[i1] and     # Progressive higher closes
            closes[i3] > closes[i2] and
            opens[i2] > opens[i1] and       # Progressive higher opens
            opens[i3] > opens[i2] and
            opens[i2] < closes[i1] and      # Each opens within previous body
            opens[i3] < closes[i2]):
            
            pattern['detected'] = True
            
            # Strength based on consistency and size
            avg_body = (abs(closes[i1] - opens[i1]) + abs(closes[i2] - opens[i2]) + abs(closes[i3] - opens[i3])) / 3
            progression_rate = ((closes[i3] - closes[i1]) / closes[i1])
            
            pattern['strength'] = min(1.0, 0.6 + progression_rate * 5)
            pattern['reliability'] = pattern['strength'] * 0.85
            pattern['completion_prob'] = pattern['strength'] * 0.8
        
        return pattern
    
    def _detect_three_black_crows(self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, i1: int, i2: int, i3: int) -> Dict:
        """Detect Three Black Crows pattern (3-day bearish continuation)"""
        pattern = {
            'name': 'three_black_crows',
            'detected': False,
            'strength': 0.0,
            'reliability': 0.0,
            'days_forming': 3,
            'completion_prob': 0.0,
            'signal_type': 'bearish_continuation',
            'day_index': i3
        }
        
        # Three consecutive red candles
        if (closes[i1] < opens[i1] and      # Day 1: Red
            closes[i2] < opens[i2] and      # Day 2: Red
            closes[i3] < opens[i3] and      # Day 3: Red
            closes[i2] < closes[i1] and     # Progressive lower closes
            closes[i3] < closes[i2] and
            opens[i2] < opens[i1] and       # Progressive lower opens
            opens[i3] < opens[i2] and
            opens[i2] > closes[i1] and      # Each opens within previous body
            opens[i3] > closes[i2]):
            
            pattern['detected'] = True
            
            # Strength calculation
            decline_rate = ((closes[i1] - closes[i3]) / closes[i1])
            
            pattern['strength'] = min(1.0, 0.6 + decline_rate * 5)
            pattern['reliability'] = pattern['strength'] * 0.85
            pattern['completion_prob'] = pattern['strength'] * 0.8
        
        return pattern
    
    def _identify_pattern_clusters(self, patterns: List[Dict]) -> List[Dict]:
        """Identify clusters of patterns occurring at same time"""
        if not patterns:
            return []
        
        clusters = []
        grouped = {}
        
        # Group patterns by day index
        for pattern in patterns:
            day_idx = pattern.get('day_index', -1)
            if day_idx not in grouped:
                grouped[day_idx] = []
            grouped[day_idx].append(pattern)
        
        # Create clusters
        for day_idx, day_patterns in grouped.items():
            if len(day_patterns) > 1:
                cluster_strength = np.mean([p['strength'] for p in day_patterns])
                cluster_types = [p['signal_type'] for p in day_patterns]
                
                # Determine dominant signal
                bullish_count = sum(1 for t in cluster_types if 'bullish' in t)
                bearish_count = sum(1 for t in cluster_types if 'bearish' in t)
                
                dominant_signal = 'bullish' if bullish_count > bearish_count else 'bearish' if bearish_count > bullish_count else 'mixed'
                
                clusters.append({
                    'day_index': day_idx,
                    'pattern_count': len(day_patterns),
                    'patterns': [p['name'] for p in day_patterns],
                    'cluster_strength': cluster_strength,
                    'dominant_signal': dominant_signal
                })
        
        return clusters
    
    def _simple_moving_average(self, values: np.ndarray, period: int) -> np.ndarray:
        """Calculate simple moving average with proper edge handling"""
        if len(values) < period:
            return np.full(len(values), np.nan)
        
        sma = np.full(len(values), np.nan)
        for i in range(period - 1, len(values)):
            sma[i] = np.mean(values[i - period + 1:i + 1])
        
        return sma
    
    def _identify_gaps(self, processed: Dict) -> List[Dict]:
        """Enhanced gap identification with classification"""
        gaps = []
        
        for i in range(1, len(processed['opens'])):
            prev_close = processed['closes'][i-1]
            curr_open = processed['opens'][i]
            
            gap_size = (curr_open - prev_close) / prev_close
            
            if abs(gap_size) > 0.005:  # 0.5% gap threshold
                gap_type = 'gap_up' if gap_size > 0 else 'gap_down'
                
                # Classify gap magnitude
                if abs(gap_size) > 0.05:
                    magnitude = 'major'
                elif abs(gap_size) > 0.02:
                    magnitude = 'significant'
                else:
                    magnitude = 'minor'
                
                # Check if gap was filled
                filled = False
                if gap_type == 'gap_up':
                    filled = processed['lows'][i] <= prev_close
                else:
                    filled = processed['highs'][i] >= prev_close
                
                gaps.append({
                    'day': i,
                    'date': processed['dates'][i],
                    'type': gap_type,
                    'magnitude': magnitude,
                    'size_pct': abs(gap_size) * 100,
                    'prev_close': prev_close,
                    'curr_open': curr_open,
                    'filled': filled
                })
        
        return gaps
    
    def _detect_patterns(self, data: Dict) -> Dict:
        """Enhanced pattern detection with additional patterns"""
        patterns = {
            'primary_pattern': None,
            'pattern_reliability': 0.0,
            'days_in_formation': 0,
            'completion_probability': 0.0,
            'detected_patterns': []
        }
        
        closes = data['closes']
        highs = data['highs']
        lows = data['lows']
        volumes = data['volumes']
        length = len(closes)
        
        if length < 15:
            return patterns
        
        # Detect all patterns
        pattern_detectors = [
            self._detect_ascending_triangle(highs, lows, closes),
            self._detect_descending_triangle(highs, lows, closes),
            self._detect_rectangle_pattern(highs, lows, closes),
            self._detect_cup_and_handle(closes),
            self._detect_flag_patterns(closes, volumes),
            self._detect_wedge_pattern(highs, lows, closes),
            self._detect_head_and_shoulders(highs, lows, closes),
            self._detect_double_top_bottom(highs, lows, closes)
        ]
        
        # Filter valid patterns - ensure all have required fields
        for pattern in pattern_detectors:
            # Ensure pattern has all required fields
            if 'strength' not in pattern:
                pattern['strength'] = pattern.get('reliability', 0.0)  # Use reliability as fallback
            
            if pattern['detected'] and pattern['strength'] >= self.pattern_thresholds['pattern_min_strength']:
                patterns['detected_patterns'].append(pattern)
        
        # Select primary pattern
        if patterns['detected_patterns']:
            primary = max(patterns['detected_patterns'], key=lambda x: x['reliability'])
            patterns['primary_pattern'] = primary['name']
            patterns['pattern_reliability'] = primary['reliability']
            patterns['days_in_formation'] = primary['days_forming']
            patterns['completion_probability'] = primary['completion_prob']
        
        print(f"   🔍 Detected {len(patterns['detected_patterns'])} chart patterns")
        return patterns
    
    def _detect_ascending_triangle(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Dict:
        """Detect ascending triangle pattern"""
        pattern = {
            'name': 'ascending_triangle',
            'detected': False,
            'strength': 0.0,
            'reliability': 0.0,
            'days_forming': 0,
            'completion_prob': 0.0,
            'signal_type': 'bullish',
            'resistance_level': 0.0
        }
        
        if len(closes) < self.pattern_thresholds['triangle_min_days']:
            return pattern
        
        # Look for horizontal resistance and rising support
        lookback = min(20, len(closes))
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        
        # Check for horizontal resistance
        resistance_level = np.mean(recent_highs[-5:])
        resistance_touches = sum(1 for h in recent_highs if abs(h - resistance_level) / resistance_level < 0.01)
        
        if resistance_touches >= self.pattern_thresholds['support_resistance_touch_min']:
            # Check for rising support
            days = np.arange(len(recent_lows))
            support_slope = np.polyfit(days, recent_lows, 1)[0]
            
            if support_slope > 0:  # Rising support
                pattern['detected'] = True
                pattern['resistance_level'] = resistance_level
                pattern['days_forming'] = lookback
                
                # Calculate strength based on touches and slope
                pattern['strength'] = min(1.0, 0.3 + (resistance_touches / 5) * 0.4 + (support_slope * 100) * 0.3)
                pattern['reliability'] = pattern['strength'] * 0.8
                pattern['completion_prob'] = pattern['reliability'] * 0.7
        
        return pattern

    def _detect_descending_triangle(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Dict:
        """Detect descending triangle pattern"""
        pattern = {
            'name': 'descending_triangle',
            'detected': False,
            'strength': 0.0,
            'reliability': 0.0,
            'days_forming': 0,
            'completion_prob': 0.0,
            'signal_type': 'bearish',
            'support_level': 0.0
        }
        
        if len(closes) < self.pattern_thresholds['triangle_min_days']:
            return pattern
        
        lookback = min(20, len(closes))
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        
        # Check for horizontal support
        support_level = np.mean(recent_lows[-5:])
        support_touches = sum(1 for l in recent_lows if abs(l - support_level) / support_level < 0.01)
        
        if support_touches >= self.pattern_thresholds['support_resistance_touch_min']:
            # Check for descending resistance
            days = np.arange(len(recent_highs))
            resistance_slope = np.polyfit(days, recent_highs, 1)[0]
            
            if resistance_slope < 0:  # Descending resistance
                pattern['detected'] = True
                pattern['support_level'] = support_level
                pattern['days_forming'] = lookback
                
                pattern['strength'] = min(1.0, 0.3 + (support_touches / 5) * 0.4 + (abs(resistance_slope) * 100) * 0.3)
                pattern['reliability'] = pattern['strength'] * 0.8
                pattern['completion_prob'] = pattern['reliability'] * 0.7
        
        return pattern

    def _detect_rectangle_pattern(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Dict:
        """Detect rectangle/channel pattern"""
        pattern = {
            'name': 'rectangle',
            'detected': False,
            'strength': 0.0,
            'reliability': 0.0,
            'days_forming': 0,
            'completion_prob': 0.0,
            'signal_type': 'neutral',
            'channel_width': 0.0
        }
        
        if len(closes) < self.pattern_thresholds['triangle_min_days']:
            return pattern
        
        lookback = min(20, len(closes))
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        
        # Check for horizontal resistance and support
        resistance_level = np.mean(recent_highs)
        support_level = np.mean(recent_lows)
        
        # Count touches
        resistance_touches = sum(1 for h in recent_highs if abs(h - resistance_level) / resistance_level < 0.02)
        support_touches = sum(1 for l in recent_lows if abs(l - support_level) / support_level < 0.02)
        
        if (resistance_touches >= self.pattern_thresholds['support_resistance_touch_min'] and 
            support_touches >= self.pattern_thresholds['support_resistance_touch_min']):
            
            # Check if levels are relatively parallel
            channel_width = (resistance_level - support_level) / support_level
            
            if 0.02 < channel_width < 0.15:  # Reasonable channel width
                pattern['detected'] = True
                pattern['channel_width'] = channel_width
                pattern['days_forming'] = lookback
                
                # Determine signal based on previous trend
                if closes[-1] > closes[0]:
                    pattern['signal_type'] = 'bullish_continuation'
                elif closes[-1] < closes[0]:
                    pattern['signal_type'] = 'bearish_continuation'
                
                touch_score = (resistance_touches + support_touches) / 10
                pattern['strength'] = min(1.0, 0.4 + touch_score * 0.6)
                pattern['reliability'] = pattern['strength'] * 0.75
                pattern['completion_prob'] = pattern['reliability'] * 0.65
        
        return pattern

    def _detect_cup_and_handle(self, closes: np.ndarray) -> Dict:
        """Detect cup and handle pattern"""
        pattern = {
            'name': 'cup_and_handle',
            'detected': False,
            'strength': 0.0,
            'reliability': 0.0,
            'days_forming': 0,
            'completion_prob': 0.0,
            'signal_type': 'bullish',
            'cup_depth': 0.0
        }
        
        if len(closes) < 20:
            return pattern
        
        # Look for U-shaped pattern
        lookback = min(30, len(closes))
        recent_closes = closes[-lookback:]
        
        # Find potential cup
        peak1_idx = np.argmax(recent_closes[:10])
        trough_idx = np.argmin(recent_closes[peak1_idx:peak1_idx+15]) + peak1_idx if peak1_idx < len(recent_closes)-15 else -1
        
        if trough_idx > peak1_idx and trough_idx < len(recent_closes) - 5:
            peak2_idx = np.argmax(recent_closes[trough_idx:]) + trough_idx
            
            if peak2_idx > trough_idx:
                # Check if peaks are at similar levels
                peak1_price = recent_closes[peak1_idx]
                peak2_price = recent_closes[peak2_idx]
                trough_price = recent_closes[trough_idx]
                
                peak_diff = abs(peak1_price - peak2_price) / peak1_price
                cup_depth = (peak1_price - trough_price) / peak1_price
                
                if peak_diff < 0.05 and 0.1 < cup_depth < 0.35:
                    # Look for handle (small pullback after second peak)
                    if peak2_idx < len(recent_closes) - 3:
                        handle_low = np.min(recent_closes[peak2_idx:])
                        handle_depth = (peak2_price - handle_low) / peak2_price
                        
                        if 0.05 < handle_depth < 0.15:
                            pattern['detected'] = True
                            pattern['cup_depth'] = cup_depth
                            pattern['days_forming'] = peak2_idx - peak1_idx
                            
                            pattern['strength'] = min(1.0, 0.5 + (1 - peak_diff) * 0.3 + (0.2 - abs(cup_depth - 0.2)) * 2)
                            pattern['reliability'] = pattern['strength'] * 0.85
                            pattern['completion_prob'] = pattern['reliability'] * 0.8
        
        return pattern

    def _detect_flag_patterns(self, closes: np.ndarray, volumes: np.ndarray) -> Dict:
        """Detect bull/bear flag patterns"""
        pattern = {
            'name': 'flag_pattern',
            'detected': False,
            'strength': 0.0,
            'reliability': 0.0,
            'days_forming': 0,
            'completion_prob': 0.0,
            'signal_type': 'continuation',
            'flag_type': None
        }
        
        if len(closes) < 15:
            return pattern
        
        # Look for strong move (pole) followed by consolidation (flag)
        for i in range(5, len(closes) - 5):
            # Check for pole
            pole_start = i - 5
            pole_end = i
            pole_move = (closes[pole_end] - closes[pole_start]) / closes[pole_start]
            
            # Check for significant move with volume
            avg_volume_pole = np.mean(volumes[pole_start:pole_end])
            avg_volume_before = np.mean(volumes[max(0, pole_start-5):pole_start]) if pole_start > 0 else avg_volume_pole
            
            if abs(pole_move) > 0.05 and avg_volume_pole > avg_volume_before * 1.3:
                # Check for flag (consolidation)
                flag_closes = closes[pole_end:pole_end+5] if pole_end+5 <= len(closes) else closes[pole_end:]
                
                if len(flag_closes) >= 3:
                    flag_range = (np.max(flag_closes) - np.min(flag_closes)) / np.mean(flag_closes)
                    
                    if flag_range < 0.03:  # Tight consolidation
                        pattern['detected'] = True
                        pattern['flag_type'] = 'bull_flag' if pole_move > 0 else 'bear_flag'
                        pattern['signal_type'] = 'bullish_continuation' if pole_move > 0 else 'bearish_continuation'
                        pattern['days_forming'] = len(flag_closes)
                        
                        pattern['strength'] = min(1.0, 0.5 + abs(pole_move) * 2 + (1 - flag_range * 10) * 0.3)
                        pattern['reliability'] = pattern['strength'] * 0.8
                        pattern['completion_prob'] = pattern['reliability'] * 0.75
                        break
        
        return pattern
    
    def _detect_wedge_pattern(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Dict:
        """Detect rising/falling wedge patterns"""
        pattern = {
            'name': 'wedge_pattern',
            'detected': False,
            'strength': 0.0,
            'reliability': 0.0,
            'days_forming': 0,
            'completion_prob': 0.0,
            'wedge_type': None,
            'breakout_direction': None
        }
        
        if len(closes) < 15:
            return pattern
        
        # Analyze last 20 days
        lookback = min(20, len(closes))
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        
        # Calculate trendlines
        days = np.arange(len(recent_highs))
        high_slope = np.polyfit(days, recent_highs, 1)[0]
        low_slope = np.polyfit(days, recent_lows, 1)[0]
        
        # Both slopes positive = rising wedge (bearish)
        # Both slopes negative = falling wedge (bullish)
        if high_slope > 0 and low_slope > 0 and high_slope < low_slope * 1.5:
            pattern['detected'] = True
            pattern['wedge_type'] = 'rising_wedge'
            pattern['breakout_direction'] = 'bearish'
            pattern['days_forming'] = lookback
            
            # Converging lines increase reliability
            convergence_rate = (low_slope - high_slope) / high_slope if high_slope != 0 else 0
            pattern['reliability'] = min(0.8, 0.5 + abs(convergence_rate) * 2)
            pattern['completion_prob'] = pattern['reliability'] * 0.7
            
        elif high_slope < 0 and low_slope < 0 and abs(high_slope) < abs(low_slope) * 1.5:
            pattern['detected'] = True
            pattern['wedge_type'] = 'falling_wedge'
            pattern['breakout_direction'] = 'bullish'
            pattern['days_forming'] = lookback
            
            convergence_rate = (high_slope - low_slope) / low_slope if low_slope != 0 else 0
            pattern['reliability'] = min(0.8, 0.5 + abs(convergence_rate) * 2)
            pattern['completion_prob'] = pattern['reliability'] * 0.7
        
        return pattern
    
    def _detect_head_and_shoulders(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Dict:
        """Detect head and shoulders pattern"""
        pattern = {
            'name': 'head_and_shoulders',
            'detected': False,
            'strength': 0.0,
            'reliability': 0.0,
            'days_forming': 0,
            'completion_prob': 0.0,
            'pattern_type': None,
            'neckline': 0.0
        }
        
        if len(closes) < 20:
            return pattern
        
        # Look for pattern in last 25 days
        lookback = min(25, len(closes))
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        
        # Find peaks (potential shoulders and head)
        peaks = []
        for i in range(2, len(recent_highs) - 2):
            if (recent_highs[i] > recent_highs[i-1] and 
                recent_highs[i] > recent_highs[i-2] and
                recent_highs[i] > recent_highs[i+1] and 
                recent_highs[i] > recent_highs[i+2]):
                peaks.append((i, recent_highs[i]))
        
        # Need at least 3 peaks
        if len(peaks) >= 3:
            # Sort by height
            sorted_peaks = sorted(peaks, key=lambda x: x[1], reverse=True)
            
            # Check if highest peak is in middle (head)
            head_idx = sorted_peaks[0][0]
            
            # Find potential shoulders
            left_shoulder = None
            right_shoulder = None
            
            for idx, height in peaks:
                if idx < head_idx - 3 and left_shoulder is None:
                    left_shoulder = (idx, height)
                elif idx > head_idx + 3 and right_shoulder is None:
                    right_shoulder = (idx, height)
            
            if left_shoulder and right_shoulder:
                # Check symmetry
                shoulder_height_diff = abs(left_shoulder[1] - right_shoulder[1]) / min(left_shoulder[1], right_shoulder[1])
                
                if shoulder_height_diff < 0.1:  # Shoulders within 10% of each other
                    pattern['detected'] = True
                    pattern['pattern_type'] = 'regular'
                    pattern['days_forming'] = right_shoulder[0] - left_shoulder[0]
                    
                    # Calculate neckline
                    left_trough = np.min(recent_lows[left_shoulder[0]:head_idx])
                    right_trough = np.min(recent_lows[head_idx:right_shoulder[0]])
                    pattern['neckline'] = (left_trough + right_trough) / 2
                    
                    # Reliability based on symmetry and formation
                    pattern['reliability'] = min(0.9, 0.6 + (1 - shoulder_height_diff) * 0.3)
                    pattern['completion_prob'] = pattern['reliability'] * 0.8
        
        return pattern
    
    def _detect_double_top_bottom(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Dict:
        """Detect double top/bottom patterns"""
        pattern = {
            'name': 'double_pattern',
            'detected': False,
            'strength': 0.0,
            'reliability': 0.0,
            'days_forming': 0,
            'completion_prob': 0.0,
            'pattern_type': None,
            'resistance_support_level': 0.0
        }
        
        if len(closes) < 15:
            return pattern
        
        # Look for double top
        recent_highs = highs[-20:]
        peaks = []
        
        for i in range(2, len(recent_highs) - 2):
            if (recent_highs[i] >= recent_highs[i-1] and 
                recent_highs[i] >= recent_highs[i-2] and
                recent_highs[i] >= recent_highs[i+1] and 
                recent_highs[i] >= recent_highs[i+2]):
                peaks.append((i, recent_highs[i]))
        
        if len(peaks) >= 2:
            # Check if two peaks are at similar levels
            for i in range(len(peaks) - 1):
                peak1 = peaks[i]
                peak2 = peaks[i + 1]
                
                height_diff = abs(peak1[1] - peak2[1]) / peak1[1]
                
                if height_diff < 0.03 and peak2[0] - peak1[0] >= 5:  # Similar height, adequate separation
                    pattern['detected'] = True
                    pattern['pattern_type'] = 'double_top'
                    pattern['name'] = 'double_top'
                    pattern['resistance_support_level'] = (peak1[1] + peak2[1]) / 2
                    pattern['days_forming'] = peak2[0] - peak1[0]
                    pattern['reliability'] = min(0.85, 0.6 + (1 - height_diff * 10) * 0.25)
                    pattern['completion_prob'] = pattern['reliability'] * 0.75
                    break
        
        # If no double top, check for double bottom
        if not pattern['detected']:
            recent_lows = lows[-20:]
            troughs = []
            
            for i in range(2, len(recent_lows) - 2):
                if (recent_lows[i] <= recent_lows[i-1] and 
                    recent_lows[i] <= recent_lows[i-2] and
                    recent_lows[i] <= recent_lows[i+1] and 
                    recent_lows[i] <= recent_lows[i+2]):
                    troughs.append((i, recent_lows[i]))
            
            if len(troughs) >= 2:
                for i in range(len(troughs) - 1):
                    trough1 = troughs[i]
                    trough2 = troughs[i + 1]
                    
                    depth_diff = abs(trough1[1] - trough2[1]) / trough1[1]
                    
                    if depth_diff < 0.03 and trough2[0] - trough1[0] >= 5:
                        pattern['detected'] = True
                        pattern['pattern_type'] = 'double_bottom'
                        pattern['name'] = 'double_bottom'
                        pattern['resistance_support_level'] = (trough1[1] + trough2[1]) / 2
                        pattern['days_forming'] = trough2[0] - trough1[0]
                        pattern['reliability'] = min(0.85, 0.6 + (1 - depth_diff * 10) * 0.25)
                        pattern['completion_prob'] = pattern['reliability'] * 0.75
                        break
        
        return pattern
    
    def _analyze_momentum_context(self, data: Dict) -> Dict:
        """Enhanced momentum analysis with multiple timeframes"""
        momentum = {
            'momentum_acceleration': 0.0,
            'current_vs_30day_avg': 0.0,
            'acceleration_trend': 'neutral',
            'momentum_persistence': 0.0,
            'velocity_profile': {},
            'momentum_divergence': None,
            'momentum_quality': 0.0
        }
        
        closes = data['closes']
        volumes = data['volumes']
        length = len(closes)
        
        if length < 10:
            return momentum
        
        # Multi-timeframe velocity analysis
        velocity_periods = [3, 5, 7, 10, 20]
        velocities = {}
        
        for period in velocity_periods:
            if length >= period + 1:
                velocity = (closes[-1] - closes[-period-1]) / closes[-period-1]
                velocities[f'{period}_day'] = velocity
        
        momentum['velocity_profile'] = velocities
        
        # Calculate acceleration
        if length >= 15:
            # First half velocity
            mid_point = length // 2
            first_half_velocity = (closes[mid_point] - closes[0]) / closes[0] / mid_point
            
            # Second half velocity
            second_half_velocity = (closes[-1] - closes[mid_point]) / closes[mid_point] / (length - mid_point)
            
            # Acceleration is change in velocity
            acceleration = second_half_velocity - first_half_velocity
            
            # Normalize to 0-1 scale
            momentum['momentum_acceleration'] = min(1.0, max(0.0, 0.5 + acceleration * 50))
        
        # Current vs 30-day average momentum
        if length >= 5:
            daily_changes = np.diff(closes) / closes[:-1]
            avg_daily_change = np.mean(daily_changes)
            recent_avg_change = np.mean(daily_changes[-3:]) if len(daily_changes) >= 3 else avg_daily_change
            
            momentum['current_vs_30day_avg'] = recent_avg_change / avg_daily_change if avg_daily_change != 0 else 1.0
        
        # Trend determination with volume confirmation
        if acceleration > 0.02:
            # Check if volume confirms acceleration
            recent_avg_volume = np.mean(volumes[-5:]) if length >= 5 else np.mean(volumes)
            if recent_avg_volume > np.mean(volumes) * 1.2:
                momentum['acceleration_trend'] = 'accelerating_confirmed'
            else:
                momentum['acceleration_trend'] = 'accelerating'
        elif acceleration < -0.02:
            momentum['acceleration_trend'] = 'decelerating'
        else:
            momentum['acceleration_trend'] = 'stable'
        
        # Momentum persistence
        momentum['momentum_persistence'] = self._calculate_momentum_persistence(closes)
        
        # Check for momentum divergence with price
        momentum['momentum_divergence'] = self._check_momentum_divergence(data)
        
        # Calculate momentum quality (consistency + strength)
        if velocities:
            velocity_consistency = 1 - np.std(list(velocities.values())) / (abs(np.mean(list(velocities.values()))) + 0.001)
            momentum['momentum_quality'] = min(1.0, velocity_consistency * momentum['momentum_acceleration'])
        
        print(f"   ⚡ Momentum acceleration: {momentum['momentum_acceleration']:.3f} ({momentum['acceleration_trend']})")
        return momentum
    
    def _check_momentum_divergence(self, data: Dict) -> Optional[str]:
        """Check for divergence between price and momentum indicators"""
        if len(data['closes']) < 20:
            return None
        
        closes = data['closes']
        rsi = data['rsi']
        
        # Check last 10 days for divergence
        price_trend = (closes[-1] - closes[-10]) / closes[-10]
        rsi_trend = (rsi[-1] - rsi[-10]) / 50  # Normalize RSI trend
        
        # Bullish divergence: price down, RSI up
        if price_trend < -0.02 and rsi_trend > 0.05:
            return 'bullish_divergence'
        # Bearish divergence: price up, RSI down
        elif price_trend > 0.02 and rsi_trend < -0.05:
            return 'bearish_divergence'
        
        return None
    
    def _calculate_momentum_persistence(self, closes: np.ndarray) -> float:
        """Enhanced momentum persistence calculation"""
        if len(closes) < 5:
            return 0.0
        
        # Calculate streaks of up/down days
        changes = np.diff(closes)
        
        current_streak = 1
        max_streak = 1
        
        for i in range(1, len(changes)):
            if (changes[i] > 0 and changes[i-1] > 0) or (changes[i] < 0 and changes[i-1] < 0):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        
        # Also consider the magnitude of moves
        avg_change = np.mean(np.abs(changes))
        recent_change = np.mean(np.abs(changes[-3:])) if len(changes) >= 3 else avg_change
        
        magnitude_factor = min(1.0, recent_change / (avg_change + 0.0001))
        
        # Combine streak and magnitude
        persistence = min(1.0, (max_streak / 7) * 0.7 + magnitude_factor * 0.3)
        
        return persistence
    
    def _analyze_volume_patterns(self, data: Dict) -> Dict:
        """Enhanced volume analysis with profile detection"""
        volume_analysis = {
            'volume_regime': 'normal',
            '30day_avg_volume': 0,
            'recent_volume_trend': 'stable',
            'volume_breakout_signal': False,
            'institutional_activity': 'low',
            'volume_price_relationship': 'neutral',
            'volume_profile': {},
            'unusual_volume_days': []
        }
        
        volumes = data['volumes']
        closes = data['closes']
        length = len(volumes)
        
        if length < 10:
            return volume_analysis
        
        # Basic volume metrics
        volume_analysis['30day_avg_volume'] = int(np.mean(volumes))
        recent_avg_volume = np.mean(volumes[-5:]) if length >= 5 else np.mean(volumes)
        
        # Volume trend analysis with multiple thresholds
        volume_ratio = recent_avg_volume / np.mean(volumes)
        
        if volume_ratio > 1.5:
            volume_analysis['recent_volume_trend'] = 'surging'
            volume_analysis['volume_breakout_signal'] = True
        elif volume_ratio > 1.2:
            volume_analysis['recent_volume_trend'] = 'increasing'
        elif volume_ratio < 0.5:
            volume_analysis['recent_volume_trend'] = 'drying_up'
        elif volume_ratio < 0.8:
            volume_analysis['recent_volume_trend'] = 'decreasing'
        
        # Enhanced volume regime classification
        volume_cv = np.std(volumes) / np.mean(volumes)  # Coefficient of variation
        volume_spikes = np.sum(volumes > np.mean(volumes) * 2)
        
        if volume_spikes >= 3:
            volume_analysis['volume_regime'] = 'high_volatility'
        elif volume_cv > 0.8:
            volume_analysis['volume_regime'] = 'volatile'
        elif recent_avg_volume > np.mean(volumes) * 1.3:
            volume_analysis['volume_regime'] = 'accumulation'
        elif recent_avg_volume < np.mean(volumes) * 0.7:
            volume_analysis['volume_regime'] = 'distribution'
        
        # Institutional activity detection
        large_volume_days = []
        for i in range(length):
            if volumes[i] > np.mean(volumes) * 2:
                price_change = (closes[i] - closes[i-1]) / closes[i-1] if i > 0 else 0
                large_volume_days.append({
                    'day': i,
                    'volume_ratio': volumes[i] / np.mean(volumes),
                    'price_change': price_change
                })
        
        volume_analysis['unusual_volume_days'] = large_volume_days[-5:]  # Last 5 unusual days
        
        # Determine institutional activity level
        if len(large_volume_days) >= 3:
            avg_spike_size = np.mean([d['volume_ratio'] for d in large_volume_days])
            if avg_spike_size > 3:
                volume_analysis['institutional_activity'] = 'very_high'
            elif avg_spike_size > 2:
                volume_analysis['institutional_activity'] = 'high'
            else:
                volume_analysis['institutional_activity'] = 'moderate'
        
        # Volume-price relationship analysis
        if length > 5:
            # Calculate correlation
            price_changes = np.diff(closes) / closes[:-1]
            volume_changes = np.diff(volumes) / volumes[:-1]
            
            if len(price_changes) > 0 and len(volume_changes) > 0:
                correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
                
                recent_price_change = (closes[-1] - closes[-5]) / closes[-5]
                recent_volume_change = (recent_avg_volume - np.mean(volumes[:-5])) / np.mean(volumes[:-5])
                
                # Classify relationship
                if correlation > 0.3:
                    if recent_price_change > 0.02 and recent_volume_change > 0.2:
                        volume_analysis['volume_price_relationship'] = 'bullish_confirmation'
                    elif recent_price_change < -0.02 and recent_volume_change > 0.2:
                        volume_analysis['volume_price_relationship'] = 'bearish_distribution'
                elif correlation < -0.3:
                    if recent_price_change > 0.02 and recent_volume_change < -0.1:
                        volume_analysis['volume_price_relationship'] = 'bullish_divergence_warning'
                    elif recent_price_change < -0.02 and recent_volume_change < -0.1:
                        volume_analysis['volume_price_relationship'] = 'bearish_exhaustion'
        
        # Volume profile
        volume_analysis['volume_profile'] = {
            'avg_volume': int(np.mean(volumes)),
            'median_volume': int(np.median(volumes)),
            'volume_volatility': volume_cv,
            'high_volume_threshold': int(np.mean(volumes) * 2),
            'low_volume_threshold': int(np.mean(volumes) * 0.5)
        }
        
        return volume_analysis
    
    def _identify_key_levels(self, data: Dict) -> Dict:
        """Enhanced support/resistance identification with clustering"""
        highs = data['highs']
        lows = data['lows']
        closes = data['closes']
        volumes = data['volumes']
        current_price = closes[-1]
        
        levels = {
            'key_support_levels': [],
            'key_resistance_levels': [],
            'current_level_strength': 0.0,
            'nearest_support': current_price * 0.95,  # Default 5% below
            'nearest_resistance': current_price * 1.05,  # Default 5% above
            'price_zones': [],
            'level_quality': 0.0
        }
        
        highs = data['highs']
        lows = data['lows']
        closes = data['closes']
        volumes = data['volumes']
        current_price = closes[-1]
        
        # Find all potential levels
        all_levels = []
        
        # Method 1: Swing highs and lows
        swing_highs = self._find_swing_points(highs, 'high')
        swing_lows = self._find_swing_points(lows, 'low')
        
        # Method 2: High volume price levels
        if len(volumes) > 0 and np.mean(volumes) > 0:
            for i in range(len(closes)):
                if i < len(volumes) and volumes[i] > np.mean(volumes) * 1.5:
                    all_levels.append({
                        'price': closes[i],
                        'type': 'volume',
                        'strength': volumes[i] / np.mean(volumes)
                    })
        
        # Method 3: Gap levels
        for gap in data.get('gaps', []):
            if gap.get('prev_close', 0) > 0 and gap.get('size_pct', 0) > 0:
                all_levels.append({
                    'price': gap['prev_close'],
                    'type': 'gap',
                    'strength': gap['size_pct'] / 2
                })
        
        # Cluster nearby levels
        resistance_clusters = []
        support_clusters = []
        
        for level in swing_highs:
            cluster = {'price': level, 'touches': 1, 'strength': 0.5}
            
            # Count touches within 1%
            for high in highs:
                if abs(high - level) / level < 0.01:
                    cluster['touches'] += 1
            
            cluster['strength'] = min(1.0, cluster['touches'] / 5)
            resistance_clusters.append(cluster)
        
        for level in swing_lows:
            cluster = {'price': level, 'touches': 1, 'strength': 0.5}
            
            for low in lows:
                if abs(low - level) / level < 0.01:
                    cluster['touches'] += 1
            
            cluster['strength'] = min(1.0, cluster['touches'] / 5)
            support_clusters.append(cluster)
        
        # Sort by strength and select top levels
        resistance_clusters.sort(key=lambda x: x['strength'], reverse=True)
        support_clusters.sort(key=lambda x: x['strength'], reverse=True)
        
        levels['key_resistance_levels'] = [r['price'] for r in resistance_clusters[:3]]
        levels['key_support_levels'] = [s['price'] for s in support_clusters[:3]]
        
        # Find nearest levels
        # Find nearest levels
        if resistance_clusters:
            above_price = [r['price'] for r in resistance_clusters if r['price'] > current_price]
            if above_price:
                levels['nearest_resistance'] = min(above_price)
            else:
                # Use highest resistance level if none above current price
                if resistance_clusters:
                    levels['nearest_resistance'] = max(r['price'] for r in resistance_clusters)
                else:
                    levels['nearest_resistance'] = current_price * 1.05
        
        if support_clusters:
            below_price = [s['price'] for s in support_clusters if s['price'] < current_price]
            if below_price:
                levels['nearest_support'] = max(below_price)
            else:
                # Use lowest support level if none below current price
                if support_clusters:
                    levels['nearest_support'] = min(s['price'] for s in support_clusters)
                else:
                    levels['nearest_support'] = current_price * 0.95
        
        # Calculate current level strength
        if resistance_clusters and support_clusters:
            # Get top clusters safely
            resistance_touches = [r['touches'] for r in resistance_clusters[:2] if 'touches' in r]
            support_touches = [s['touches'] for s in support_clusters[:2] if 'touches' in s]
            
            all_touches = resistance_touches + support_touches
            if all_touches:
                avg_touches = np.mean(all_touches)
                levels['current_level_strength'] = max(0.0, min(1.0, avg_touches / 5))
            else:
                levels['current_level_strength'] = 0.2
        else:
            levels['current_level_strength'] = 0.1
        
        # Define price zones
        if levels['nearest_support'] and levels['nearest_resistance']:
            zone_size = (levels['nearest_resistance'] - levels['nearest_support']) / levels['nearest_support']
            levels['price_zones'] = [{
                'type': 'current_range',
                'lower': levels['nearest_support'],
                'upper': levels['nearest_resistance'],
                'size_pct': zone_size * 100
            }]
        
        # Calculate level quality based on clarity and strength
        if resistance_clusters and support_clusters:
            clarity = 1.0
            # Only calculate clarity if we have multiple clusters
            if len(resistance_clusters) > 1:
                for i in range(min(2, len(resistance_clusters) - 1)):
                    if i + 1 < len(resistance_clusters) and resistance_clusters[i]['price'] > 0:
                        diff = abs(resistance_clusters[i]['price'] - resistance_clusters[i+1]['price']) / resistance_clusters[i]['price']
                        clarity *= (1 - min(0.5, 1 - diff * 20))
            
            # Ensure level_quality is between 0 and 1
            raw_quality = levels['current_level_strength'] * clarity
            levels['level_quality'] = max(0.0, min(1.0, raw_quality))
        else:
            # Set default quality when no clusters found
            levels['level_quality'] = 0.3
        
        # Final validation - ensure all values are valid
        if levels['nearest_support'] <= 0:
            levels['nearest_support'] = current_price * 0.95
            
        if levels['nearest_resistance'] <= 0:
            levels['nearest_resistance'] = current_price * 1.05
            
        # Ensure resistance > support
        if levels['nearest_resistance'] <= levels['nearest_support']:
            levels['nearest_resistance'] = levels['nearest_support'] * 1.1
        
        # Ensure reasonable distance from current price
        if abs(levels['nearest_resistance'] - current_price) / current_price < 0.01:
            levels['nearest_resistance'] = current_price * 1.05
            
        if abs(current_price - levels['nearest_support']) / current_price < 0.01:
            levels['nearest_support'] = current_price * 0.95
        
        # Ensure current_level_strength is between 0 and 1
        levels['current_level_strength'] = max(0.0, min(1.0, levels['current_level_strength']))
        
        # Ensure level_quality is between 0 and 1
        levels['level_quality'] = max(0.0, min(1.0, levels['level_quality']))
        
        return levels
    
    def _find_swing_points(self, prices: np.ndarray, point_type: str) -> List[float]:
        """Enhanced swing point detection with variable window"""
        swing_points = []
        
        # Handle insufficient data
        if len(prices) < 7:
            return swing_points
        
        # Use multiple window sizes for better detection
        for window in [2, 3, 5]:
            if len(prices) <= window * 2:
                continue
            for i in range(window, len(prices) - window):
                is_swing = True
                
                if point_type == 'high':
                    # Check if it's a local maximum
                    for j in range(1, window + 1):
                        if prices[i] < prices[i-j] or prices[i] < prices[i+j]:
                            is_swing = False
                            break
                else:  # low
                    # Check if it's a local minimum
                    for j in range(1, window + 1):
                        if prices[i] > prices[i-j] or prices[i] > prices[i+j]:
                            is_swing = False
                            break
                
                if is_swing:
                    swing_points.append(prices[i])
        
        # Remove duplicates and sort
        swing_points = list(set(swing_points))
        swing_points.sort(reverse=(point_type == 'high'))
        
        return swing_points[:10]  # Return top 10 levels
    
    def _analyze_breakout_potential(self, data: Dict, patterns: Dict, volume_analysis: Dict, levels: Dict) -> Dict:
        """Enhanced breakout analysis with multiple confirmation factors"""
        breakout = {
            'breakout_detected': False,
            'breakout_direction': 'neutral',
            'breakout_strength': 0.0,
            'volume_confirmation': False,
            'breakout_target': 0.0,
            'probability_of_follow_through': 0.0,
            'breakout_type': None,
            'confirmation_factors': []
        }
        
        closes = data['closes']
        highs = data['highs']
        lows = data['lows']
        volumes = data['volumes']
        current_price = closes[-1]
        
        # Get recent price action
        if len(closes) < 5:
            return breakout
        
        recent_high = np.max(highs[-5:])
        recent_low = np.min(lows[-5:])
        price_range = recent_high - recent_low
        
        # Check for breakout conditions
        resistance_levels = levels['key_resistance_levels']
        support_levels = levels['key_support_levels']
        
        # Bullish breakout detection
        if resistance_levels and levels['nearest_resistance']:
            nearest_resistance = levels['nearest_resistance']
            
            # Check if price is breaking resistance
            if current_price > nearest_resistance * 0.995:
                breakout['breakout_detected'] = True
                breakout['breakout_direction'] = 'bullish'
                
                # Determine breakout type
                if current_price > nearest_resistance * 1.02:
                    breakout['breakout_type'] = 'strong_breakout'
                else:
                    breakout['breakout_type'] = 'testing_resistance'
                
                # Calculate strength based on multiple factors
                strength_factors = []
                
                # Factor 1: Volume confirmation
                recent_volume = np.mean(volumes[-3:])
                if recent_volume > volume_analysis['30day_avg_volume'] * 1.5:
                    breakout['volume_confirmation'] = True
                    strength_factors.append(0.25)
                    breakout['confirmation_factors'].append('volume_surge')
                
                # Factor 2: Pattern confirmation
                bullish_patterns = ['ascending_triangle', 'cup_and_handle', 'bull_flag', 
                                  'falling_wedge', 'double_bottom']
                if patterns['primary_pattern'] in bullish_patterns:
                    strength_factors.append(0.25)
                    breakout['confirmation_factors'].append(f'pattern_{patterns["primary_pattern"]}')
                
                # Factor 3: Momentum confirmation
                if (closes[-1] - closes[-5]) / closes[-5] > 0.02:
                    strength_factors.append(0.15)
                    breakout['confirmation_factors'].append('momentum_positive')
                
                # Factor 4: Gap confirmation
                if data['gaps'] and data['gaps'][-1]['type'] == 'gap_up' and not data['gaps'][-1]['filled']:
                    strength_factors.append(0.15)
                    breakout['confirmation_factors'].append('gap_up_unfilled')
                
                # Factor 5: Multiple timeframe confirmation
                if all(closes[-1] > sma for sma in [data['sma_5'][-1], data['sma_10'][-1], data['sma_20'][-1]] 
                      if not np.isnan(sma)):
                    strength_factors.append(0.1)
                    breakout['confirmation_factors'].append('above_all_mas')
                
                # Factor 6: RSI confirmation
                if data['rsi'][-1] > 50 and data['rsi'][-1] < 70:
                    strength_factors.append(0.1)
                    breakout['confirmation_factors'].append('rsi_bullish_zone')
                
                breakout['breakout_strength'] = min(1.0, sum(strength_factors))
                
                # Calculate target
                if len(resistance_levels) > 1:
                    next_resistance = min([r for r in resistance_levels if r > nearest_resistance], 
                                        default=nearest_resistance * 1.1)
                    breakout['breakout_target'] = next_resistance
                else:
                    # Use ATR-based target
                    atr = data['atr'][-1] if not np.isnan(data['atr'][-1]) else price_range * 0.02
                    breakout['breakout_target'] = current_price + (atr * 2)
        
        # Bearish breakdown detection
        elif support_levels and levels['nearest_support']:
            nearest_support = levels['nearest_support']
            
            if current_price < nearest_support * 1.005:
                breakout['breakout_detected'] = True
                breakout['breakout_direction'] = 'bearish'
                
                if current_price < nearest_support * 0.98:
                    breakout['breakout_type'] = 'strong_breakdown'
                else:
                    breakout['breakout_type'] = 'testing_support'
                
                # Similar strength calculation for bearish
                strength_factors = []
                
                recent_volume = np.mean(volumes[-3:])
                if recent_volume > volume_analysis['30day_avg_volume'] * 1.5:
                    breakout['volume_confirmation'] = True
                    strength_factors.append(0.25)
                    breakout['confirmation_factors'].append('volume_surge')
                
                bearish_patterns = ['descending_triangle', 'bear_flag', 'rising_wedge', 
                                  'head_and_shoulders', 'double_top']
                if patterns['primary_pattern'] in bearish_patterns:
                    strength_factors.append(0.25)
                    breakout['confirmation_factors'].append(f'pattern_{patterns["primary_pattern"]}')
                
                if (closes[-5] - closes[-1]) / closes[-5] > 0.02:
                    strength_factors.append(0.15)
                    breakout['confirmation_factors'].append('momentum_negative')
                
                if all(closes[-1] < sma for sma in [data['sma_5'][-1], data['sma_10'][-1], data['sma_20'][-1]] 
                      if not np.isnan(sma)):
                    strength_factors.append(0.1)
                    breakout['confirmation_factors'].append('below_all_mas')
                
                breakout['breakout_strength'] = min(1.0, sum(strength_factors))
                
                # Bearish target
                if len(support_levels) > 1:
                    next_support = max([s for s in support_levels if s < nearest_support], 
                                     default=nearest_support * 0.9)
                    breakout['breakout_target'] = next_support
                else:
                    atr = data['atr'][-1] if not np.isnan(data['atr'][-1]) else price_range * 0.02
                    breakout['breakout_target'] = current_price - (atr * 2)
        
        # Calculate follow-through probability
        if breakout['breakout_detected']:
            base_probability = 0.35
            
            # Adjust based on strength
            base_probability += breakout['breakout_strength'] * 0.35
            
            # Adjust based on pattern reliability
            if patterns['pattern_reliability'] > 0:
                base_probability += patterns['pattern_reliability'] * 0.15
            
            # Adjust based on volume
            if breakout['volume_confirmation']:
                base_probability += 0.1
            
            # Adjust based on number of confirmation factors
            base_probability += len(breakout['confirmation_factors']) * 0.02
            
            breakout['probability_of_follow_through'] = min(0.95, base_probability)
        
        return breakout
    
    def _calculate_extended_metrics(self, data: Dict, patterns: Dict, candlestick_analysis: Dict) -> Dict:
        """Calculate additional advanced metrics"""
        extended = {
            'volatility_analysis': self._analyze_volatility(data),
            'trend_quality': self._assess_trend_quality(data),
            'pattern_confluence': self._check_pattern_confluence(patterns, candlestick_analysis),
            'risk_metrics': self._calculate_risk_metrics(data)
        }
        
        return extended
    
    def _analyze_volatility(self, data: Dict) -> Dict:
        """Analyze volatility patterns and regime"""
        closes = data['closes']
        atr = data['atr']
        
        volatility = {
            'current_volatility': 0.0,
            'volatility_trend': 'stable',
            'volatility_regime': 'normal',
            'volatility_percentile': 0.0
        }
        
        if len(closes) < 10:
            return volatility
        
        # Calculate historical volatility
        returns = np.diff(closes) / closes[:-1]
        hist_vol = np.std(returns) * np.sqrt(252)  # Annualized
        
        volatility['current_volatility'] = hist_vol
        
        # Volatility trend
        recent_vol = np.std(returns[-5:]) if len(returns) >= 5 else hist_vol
        older_vol = np.std(returns[:-5]) if len(returns) > 5 else hist_vol
        
        if recent_vol > older_vol * 1.2:
            volatility['volatility_trend'] = 'increasing'
        elif recent_vol < older_vol * 0.8:
            volatility['volatility_trend'] = 'decreasing'
        
        # Volatility regime
        if hist_vol < 0.15:
            volatility['volatility_regime'] = 'low'
        elif hist_vol > 0.35:
            volatility['volatility_regime'] = 'high'
        elif hist_vol > 0.25:
            volatility['volatility_regime'] = 'elevated'
        
        # Percentile ranking
        all_vols = [np.std(returns[max(0, i-20):i]) for i in range(20, len(returns)+1)]
        if all_vols:
            current_vol_rank = sum(1 for v in all_vols if v < recent_vol) / len(all_vols)
            volatility['volatility_percentile'] = current_vol_rank
        
        return volatility
    
    def _assess_trend_quality(self, data: Dict) -> Dict:
        """Assess the quality and sustainability of current trend"""
        closes = data['closes']
        volumes = data['volumes']
        
        quality = {
            'trend_strength': 0.0,
            'trend_consistency': 0.0,
            'trend_sustainability': 0.0,
            'trend_phase': 'undefined'
        }
        
        if len(closes) < 20:
            return quality
        
        # Linear regression for trend
        x = np.arange(len(closes))
        slope, intercept = np.polyfit(x, closes, 1)
        
        # R-squared for consistency
        y_pred = slope * x + intercept
        ss_res = np.sum((closes - y_pred) ** 2)
        ss_tot = np.sum((closes - np.mean(closes)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        quality['trend_consistency'] = max(0, r_squared)
        
        # Normalized trend strength
        normalized_slope = slope / np.mean(closes) * 100
        quality['trend_strength'] = min(1.0, abs(normalized_slope) / 10)
        
        # Trend sustainability (based on volume and momentum)
        if slope > 0:  # Uptrend
            volume_confirmation = np.corrcoef(closes, volumes)[0, 1] > 0.2
            momentum_slowing = data['daily_returns'][-1] < np.mean(data['daily_returns'][:-5])
            
            sustainability = 0.5
            if volume_confirmation:
                sustainability += 0.25
            if not momentum_slowing:
                sustainability += 0.25
            
            quality['trend_sustainability'] = sustainability
            quality['trend_phase'] = 'uptrend'
            
        elif slope < 0:  # Downtrend
            quality['trend_sustainability'] = 0.5
            quality['trend_phase'] = 'downtrend'
        else:
            quality['trend_phase'] = 'sideways'
        
        return quality
    
    def _check_pattern_confluence(self, patterns: Dict, candlestick_analysis: Dict) -> Dict:
        """Check for confluence between different pattern types"""
        confluence = {
            'confluence_score': 0.0,
            'aligned_patterns': [],
            'conflicting_patterns': [],
            'dominant_signal': 'neutral'
        }
        
        # Collect all pattern signals
        pattern_signals = []
        
        # Chart patterns
        if patterns['primary_pattern']:
            signal = 'bullish' if patterns['primary_pattern'] in ['ascending_triangle', 'cup_and_handle', 
                                                                   'double_bottom', 'falling_wedge'] else 'bearish'
            pattern_signals.append({
                'type': 'chart',
                'pattern': patterns['primary_pattern'],
                'signal': signal,
                'strength': patterns['pattern_reliability']
            })
        
        # Candlestick patterns
        if candlestick_analysis['strongest_pattern']:
            for pattern in candlestick_analysis['detected_patterns']:
                if pattern['strength'] > 0.5:
                    signal = 'bullish' if 'bullish' in pattern['signal_type'] else 'bearish' if 'bearish' in pattern['signal_type'] else 'neutral'
                    pattern_signals.append({
                        'type': 'candlestick',
                        'pattern': pattern['name'],
                        'signal': signal,
                        'strength': pattern['strength']
                    })
        
        # Calculate confluence
        if pattern_signals:
            bullish_strength = sum(p['strength'] for p in pattern_signals if p['signal'] == 'bullish')
            bearish_strength = sum(p['strength'] for p in pattern_signals if p['signal'] == 'bearish')
            
            total_strength = bullish_strength + bearish_strength
            if total_strength > 0:
                confluence['confluence_score'] = abs(bullish_strength - bearish_strength) / total_strength
                
                if bullish_strength > bearish_strength * 1.5:
                    confluence['dominant_signal'] = 'bullish'
                elif bearish_strength > bullish_strength * 1.5:
                    confluence['dominant_signal'] = 'bearish'
                
                # Identify aligned patterns
                dominant = 'bullish' if bullish_strength > bearish_strength else 'bearish'
                confluence['aligned_patterns'] = [p['pattern'] for p in pattern_signals if p['signal'] == dominant]
                confluence['conflicting_patterns'] = [p['pattern'] for p in pattern_signals if p['signal'] != dominant and p['signal'] != 'neutral']
        
        return confluence
    
    def _calculate_risk_metrics(self, data: Dict) -> Dict:
        """Calculate risk metrics for position sizing and stop losses"""
        closes = data['closes']
        highs = data['highs']
        lows = data['lows']
        atr = data['atr']
        
        risk_metrics = {
            'atr_stop_loss': 0.0,
            'support_stop_loss': 0.0,
            'risk_reward_ratio': 0.0,
            'max_drawdown_30d': 0.0,
            'recovery_time': 0
        }
        
        if len(closes) < 5:
            return risk_metrics
        
        current_price = closes[-1]
        
        # ATR-based stop loss (2x ATR)
        if not np.isnan(atr[-1]):
            risk_metrics['atr_stop_loss'] = current_price - (2 * atr[-1])
        
        # Support-based stop loss
        recent_support = np.min(lows[-10:]) if len(lows) >= 10 else np.min(lows)
        risk_metrics['support_stop_loss'] = recent_support * 0.98
        
        # Maximum drawdown calculation
        peak = np.maximum.accumulate(closes)
        drawdown = (closes - peak) / peak
        risk_metrics['max_drawdown_30d'] = np.min(drawdown)
        
        # Recovery time from last drawdown
        if np.min(drawdown) < -0.05:  # Significant drawdown
            drawdown_idx = np.argmin(drawdown)
            if drawdown_idx < len(closes) - 1:
                recovery_days = 0
                for i in range(drawdown_idx + 1, len(closes)):
                    recovery_days += 1
                    if closes[i] >= peak[drawdown_idx]:
                        break
                risk_metrics['recovery_time'] = recovery_days
        
        return risk_metrics
    
    def _compile_analysis(self, ticker: str, patterns: Dict, candlestick_analysis: Dict, momentum: Dict, 
                         volume_analysis: Dict, levels: Dict, breakout: Dict, data: Dict, extended_metrics: Dict) -> Dict:
        """Compile comprehensive analysis results with all enhancements"""
        
        current_price = data['closes'][-1]
        
        analysis = {
            # Metadata
            'ticker': ticker,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_period': '30_days',
            'data_points': data['length'],
            
            # Enhanced Pattern Analysis
            'pattern_detected': {
                'primary_pattern': patterns['primary_pattern'],
                'pattern_reliability': patterns['pattern_reliability'],
                'days_in_formation': patterns['days_in_formation'],
                'completion_probability': patterns['completion_probability'],
                'all_patterns': [p['name'] for p in patterns['detected_patterns']],
                'pattern_count': len(patterns['detected_patterns'])
            },
            
            # Comprehensive Candlestick Analysis
            'candlestick_analysis': {
                'total_patterns_found': candlestick_analysis['total_patterns_found'],
                'strongest_pattern': candlestick_analysis['strongest_pattern'],
                'reversal_signals_count': len(candlestick_analysis['reversal_signals']),
                'continuation_signals_count': len(candlestick_analysis['continuation_signals']),
                'pattern_clusters': candlestick_analysis['pattern_clusters'],
                'recent_signals': [
                    {
                        'pattern': signal['name'],
                        'signal_type': signal['signal_type'],
                        'strength': signal['strength'],
                        'day_index': signal['day_index']
                    }
                    for signal in candlestick_analysis['detected_patterns'][-5:]  # Last 5 signals
                ]
            },
            
            # Enhanced Momentum Analysis
            'momentum_analysis': {
                'momentum_acceleration': momentum['momentum_acceleration'],
                'current_vs_30day_avg': momentum['current_vs_30day_avg'],
                'acceleration_trend': momentum['acceleration_trend'],
                'momentum_persistence': momentum['momentum_persistence'],
                'velocity_profile': momentum['velocity_profile'],
                'momentum_divergence': momentum['momentum_divergence'],
                'momentum_quality': momentum['momentum_quality']
            },
            
            # Enhanced Breakout Analysis
            'breakout_analysis': {
                'breakout_detected': breakout['breakout_detected'],
                'breakout_direction': breakout['breakout_direction'],
                'breakout_strength': breakout['breakout_strength'],
                'volume_confirmation': breakout['volume_confirmation'],
                'breakout_target': breakout['breakout_target'],
                'probability_of_follow_through': breakout['probability_of_follow_through'],
                'breakout_type': breakout['breakout_type'],
                'confirmation_factors': breakout['confirmation_factors']
            },
            
            # Enhanced Support/Resistance Analysis
            'support_resistance': {
                'key_support_levels': levels['key_support_levels'],
                'key_resistance_levels': levels['key_resistance_levels'],
                'current_level_strength': levels['current_level_strength'],
                'nearest_support': levels['nearest_support'],
                'nearest_resistance': levels['nearest_resistance'],
                'price_zones': levels['price_zones'],
                'level_quality': levels['level_quality']
            },
            
            # Enhanced Volume Analysis
            'volume_analysis': {
                'volume_regime': volume_analysis['volume_regime'],
                '30day_avg_volume': volume_analysis['30day_avg_volume'],
                'recent_volume_trend': volume_analysis['recent_volume_trend'],
                'volume_breakout_signal': volume_analysis['volume_breakout_signal'],
                'institutional_activity': volume_analysis['institutional_activity'],
                'volume_price_relationship': volume_analysis['volume_price_relationship'],
                'volume_profile': volume_analysis['volume_profile'],
                'unusual_volume_days': volume_analysis['unusual_volume_days']
            },
            
            # Current Market State with Technical Indicators
            'current_state': {
                'current_price': current_price,
                'daily_range': data['highs'][-1] - data['lows'][-1],
                'recent_gaps': len([g for g in data['gaps'] if g['day'] >= len(data['closes']) - 5]),
                'price_vs_sma20': (current_price - data['sma_20'][-1]) / data['sma_20'][-1] if len(data['sma_20']) > 0 and not np.isnan(data['sma_20'][-1]) else 0,
                'rsi': data['rsi'][-1] if not np.isnan(data['rsi'][-1]) else 50,
                'bb_position': (current_price - data['bb_lower'][-1]) / (data['bb_upper'][-1] - data['bb_lower'][-1]) if not np.isnan(data['bb_upper'][-1]) and data['bb_upper'][-1] != data['bb_lower'][-1] else 0.5,
                'trend_strength': data['trend_strength']
            },
            
            # Integration Signals (FOR PREDICTION ENGINE)
            'integration_signals': {
                'momentum_boost_factor': self._calculate_momentum_boost(momentum, patterns, candlestick_analysis),
                'technical_boost_factor': self._calculate_technical_boost(breakout, levels, extended_metrics),
                'confidence_boost_factor': self._calculate_confidence_boost(patterns, candlestick_analysis, volume_analysis, breakout),
                'regime_override_signal': self._get_regime_override(patterns, breakout, momentum, candlestick_analysis),
                'risk_adjustment_factor': self._calculate_risk_adjustment(extended_metrics)
            },
            
            # Extended Metrics
            'extended_analysis': extended_metrics,
            
            # Quality Metrics
            'analysis_quality': {
                'data_completeness': 1.0 if data['length'] >= 25 else data['length'] / 25,
                'pattern_confidence': patterns['pattern_reliability'],
                'volume_data_quality': 1.0 if np.mean(data['volumes']) > 1000 else 0.7,
                'overall_quality': self._calculate_overall_quality(data, patterns, volume_analysis)
            }
        }
        
        analysis = self._convert_numpy_types(analysis)
        return analysis
    
    def _calculate_momentum_boost(self, momentum: Dict, patterns: Dict, candlestick_analysis: Dict) -> float:
        """Enhanced momentum boost calculation with candlestick integration"""
        boost = 1.0  # Base factor
        
        # Strong momentum acceleration
        if momentum['momentum_acceleration'] > 0.7:
            boost += 0.25
        elif momentum['momentum_acceleration'] > 0.5:
            boost += 0.15
        
        # Trend acceleration
        if momentum['acceleration_trend'] == 'accelerating_confirmed':
            boost += 0.25
        elif momentum['acceleration_trend'] == 'accelerating':
            boost += 0.15
        elif momentum['acceleration_trend'] == 'decelerating':
            boost -= 0.15
        
        # Momentum quality
        if momentum['momentum_quality'] > 0.7:
            boost += 0.1
        
        # Pattern support
        bullish_patterns = ['ascending_triangle', 'bull_flag', 'cup_and_handle', 'falling_wedge', 'double_bottom']
        bearish_patterns = ['descending_triangle', 'bear_flag', 'rising_wedge', 'head_and_shoulders', 'double_top']
        
        if patterns['primary_pattern'] in bullish_patterns:
            boost += 0.15
        elif patterns['primary_pattern'] in bearish_patterns:
            boost -= 0.1
        
        # Candlestick pattern boost
        if candlestick_analysis['strongest_pattern']:
            # Count recent bullish vs bearish signals
            recent_candles = [p for p in candlestick_analysis['detected_patterns'] if p['day_index'] >= len(candlestick_analysis['detected_patterns']) - 5]
            
            bullish_strength = sum(p['strength'] for p in recent_candles if 'bullish' in p['signal_type'])
            bearish_strength = sum(p['strength'] for p in recent_candles if 'bearish' in p['signal_type'])
            
            if bullish_strength > bearish_strength * 1.5:
                boost += 0.2
            elif bearish_strength > bullish_strength * 1.5:
                boost -= 0.15
        
        # Momentum divergence adjustment
        if momentum['momentum_divergence'] == 'bullish_divergence':
            boost += 0.1
        elif momentum['momentum_divergence'] == 'bearish_divergence':
            boost -= 0.1
        
        return max(0.5, min(2.0, boost))
    
    def _calculate_technical_boost(self, breakout: Dict, levels: Dict, extended_metrics: Dict) -> float:
        """Enhanced technical analysis boost factor"""
        boost = 1.0
        
        # Breakout detection with strength weighting
        if breakout['breakout_detected']:
            boost += breakout['breakout_strength'] * 0.4
            
            # Extra boost for strong breakouts
            if breakout['breakout_type'] in ['strong_breakout', 'strong_breakdown']:
                boost += 0.1
        
        # Level strength and quality
        if levels['current_level_strength'] > 0.7:
            boost += 0.15
        
        if levels['level_quality'] > 0.7:
            boost += 0.1
        
        # Trend quality from extended metrics
        if 'trend_quality' in extended_metrics:
            trend_quality = extended_metrics['trend_quality']
            if trend_quality['trend_consistency'] > 0.7:
                boost += 0.15
            if trend_quality['trend_sustainability'] > 0.7:
                boost += 0.1
        
        return max(0.5, min(2.0, boost))
    
    def _calculate_confidence_boost(self, patterns: Dict, candlestick_analysis: Dict, 
                                  volume_analysis: Dict, breakout: Dict) -> float:
        """Enhanced confidence boost calculation"""
        boost = 0.0  # Additional confidence points
        
        # Pattern reliability with threshold
        if patterns['pattern_reliability'] > 0.6:
            boost += patterns['pattern_reliability'] * 0.1
        
        # Candlestick pattern clusters
        if candlestick_analysis['pattern_clusters']:
            strongest_cluster = max(candlestick_analysis['pattern_clusters'], 
                                  key=lambda x: x['cluster_strength'], 
                                  default=None)
            if strongest_cluster and strongest_cluster['cluster_strength'] > 0.7:
                boost += 0.05
        
        # Volume confirmation hierarchy
        if volume_analysis['volume_breakout_signal']:
            boost += 0.05
        
        if volume_analysis['institutional_activity'] == 'very_high':
            boost += 0.1
        elif volume_analysis['institutional_activity'] == 'high':
            boost += 0.08
        elif volume_analysis['institutional_activity'] == 'moderate':
            boost += 0.05
        
        # Breakout confirmation with multiple factors
        if breakout['breakout_detected'] and breakout['volume_confirmation']:
            boost += 0.1
            
            # Extra boost for multiple confirmation factors
            if len(breakout['confirmation_factors']) >= 4:
                boost += 0.05
        
        # Volume-price relationship confirmation
        if volume_analysis['volume_price_relationship'] in ['bullish_confirmation', 'bearish_distribution']:
            boost += 0.05
        
        return min(0.35, boost)  # Cap at 35% confidence boost
    
    def _get_regime_override(self, patterns: Dict, breakout: Dict, momentum: Dict, 
                           candlestick_analysis: Dict) -> Optional[str]:
        """Enhanced regime override detection"""
        
        # Strong breakout with multiple confirmations
        if (breakout['breakout_detected'] and 
            breakout['breakout_strength'] > 0.7 and 
            breakout['volume_confirmation'] and
            len(breakout['confirmation_factors']) >= 3):
            return 'breakout_confirmed'
        
        # Momentum acceleration with pattern support
        if (momentum['momentum_acceleration'] > 0.8 and 
            momentum['acceleration_trend'] in ['accelerating', 'accelerating_confirmed'] and
            patterns['pattern_reliability'] > 0.6):
            return 'momentum_acceleration'
        
        # Pattern completion with candlestick confirmation
        if (patterns['primary_pattern'] and 
            patterns['pattern_reliability'] > 0.8 and 
            patterns['completion_probability'] > 0.7):
            
            # Check if candlesticks align
            if candlestick_analysis['strongest_pattern']:
                pattern_signal = 'bullish' if patterns['primary_pattern'] in ['ascending_triangle', 'cup_and_handle', 'double_bottom'] else 'bearish'
                candle_signal = 'bullish' if 'bullish' in candlestick_analysis['strongest_pattern'] else 'bearish'
                
                if pattern_signal == candle_signal:
                    return 'pattern_completion_confirmed'
            
            return 'pattern_completion'
        
        # Candlestick cluster override
        if candlestick_analysis['pattern_clusters']:
            strongest_cluster = max(candlestick_analysis['pattern_clusters'], 
                                  key=lambda x: x['cluster_strength'], 
                                  default=None)
            if (strongest_cluster and 
                strongest_cluster['cluster_strength'] > 0.8 and 
                strongest_cluster['pattern_count'] >= 3):
                return f"candlestick_cluster_{strongest_cluster['dominant_signal']}"
        
        return None
    
    def _calculate_risk_adjustment(self, extended_metrics: Dict) -> float:
        """Calculate risk adjustment factor based on volatility and drawdown"""
        if 'risk_metrics' not in extended_metrics:
            return 1.0
        
        risk_metrics = extended_metrics['risk_metrics']
        volatility_analysis = extended_metrics.get('volatility_analysis', {})
        
        adjustment = 1.0
        
        # Adjust for high volatility
        if volatility_analysis.get('volatility_regime') == 'high':
            adjustment *= 0.85
        elif volatility_analysis.get('volatility_regime') == 'elevated':
            adjustment *= 0.95
        
        # Adjust for recent drawdown
        max_drawdown = risk_metrics.get('max_drawdown_30d', 0)
        if max_drawdown < -0.15:
            adjustment *= 0.8
        elif max_drawdown < -0.1:
            adjustment *= 0.9
        
        return adjustment
    
    def _calculate_overall_quality(self, data: Dict, patterns: Dict, volume_analysis: Dict) -> float:
        """Enhanced overall analysis quality score"""
        quality_factors = []
        
        # Data quality and completeness
        quality_factors.append(min(1.0, data['length'] / 30))
        
        # Pattern detection quality
        if patterns['detected_patterns']:
            avg_pattern_strength = np.mean([p['reliability'] for p in patterns['detected_patterns']])
            quality_factors.append(avg_pattern_strength)
        else:
            quality_factors.append(0.3)  # Base quality if no patterns
        
        # Volume data quality
        avg_volume = np.mean(data['volumes'])
        if avg_volume > 100000:
            quality_factors.append(1.0)
        elif avg_volume > 10000:
            quality_factors.append(0.8)
        else:
            quality_factors.append(avg_volume / 10000)
        
        # Data consistency (gaps and missing data)
        gap_penalty = min(0.3, len(data['gaps']) * 0.05)
        quality_factors.append(max(0.5, 1.0 - gap_penalty))
        
        # Price data quality (no extreme outliers)
        returns = data['daily_returns']
        outliers = np.sum(np.abs(returns) > 0.2)  # 20% daily moves
        outlier_penalty = min(0.3, outliers * 0.1)
        quality_factors.append(max(0.6, 1.0 - outlier_penalty))
        
        return np.mean(quality_factors)
    
    def _extract_ml_features(self, analysis: Dict, processed_data: Dict) -> Dict:
        """Extract machine learning ready features from analysis"""
        features = {
            # Price features
            'price_momentum_3d': analysis['momentum_analysis']['velocity_profile'].get('3_day', 0),
            'price_momentum_7d': analysis['momentum_analysis']['velocity_profile'].get('7_day', 0),
            'price_momentum_30d': analysis['momentum_analysis']['velocity_profile'].get('30_day', 0),
            'momentum_acceleration': analysis['momentum_analysis']['momentum_acceleration'],
            
            # Pattern features
            'pattern_reliability': analysis['pattern_detected']['pattern_reliability'],
            'pattern_completion_prob': analysis['pattern_detected']['completion_probability'],
            'candlestick_reversal_count': analysis['candlestick_analysis']['reversal_signals_count'],
            'candlestick_continuation_count': analysis['candlestick_analysis']['continuation_signals_count'],
            
            # Volume features
            'volume_regime_encoded': 1 if analysis['volume_analysis']['volume_regime'] == 'accumulation' else -1 if analysis['volume_analysis']['volume_regime'] == 'distribution' else 0,
            'institutional_activity_level': 2 if analysis['volume_analysis']['institutional_activity'] == 'very_high' else 1 if analysis['volume_analysis']['institutional_activity'] == 'high' else 0,
            
            # Breakout features
            'breakout_detected': 1 if analysis['breakout_analysis']['breakout_detected'] else 0,
            'breakout_strength': analysis['breakout_analysis']['breakout_strength'],
            'breakout_confirmation_count': len(analysis['breakout_analysis']['confirmation_factors']),
            
            # Technical features
            'rsi_value': analysis['current_state']['rsi'],
            'bb_position': analysis['current_state']['bb_position'],
            'price_vs_sma20': analysis['current_state']['price_vs_sma20'],
            
            # Risk features
            'volatility_regime_risk': 2 if analysis['extended_analysis']['volatility_analysis']['volatility_regime'] == 'high' else 1 if analysis['extended_analysis']['volatility_analysis']['volatility_regime'] == 'elevated' else 0,
            'max_drawdown_30d': analysis['extended_analysis']['risk_metrics']['max_drawdown_30d'],
            
            # Composite features
            'momentum_pattern_alignment': 1 if (analysis['momentum_analysis']['acceleration_trend'] == 'accelerating' and analysis['pattern_detected']['primary_pattern'] in ['ascending_triangle', 'cup_and_handle']) else -1 if (analysis['momentum_analysis']['acceleration_trend'] == 'decelerating' and analysis['pattern_detected']['primary_pattern'] in ['descending_triangle', 'head_and_shoulders']) else 0,
            
            # Integration scores
            'momentum_boost': analysis['integration_signals']['momentum_boost_factor'],
            'technical_boost': analysis['integration_signals']['technical_boost_factor'],
            'confidence_boost': analysis['integration_signals']['confidence_boost_factor']
        }
        
        return features
    
    def _create_fallback_analysis(self, ticker: str, error_msg: str) -> Dict:
        """Enhanced fallback analysis with complete structure"""
        return {
            'ticker': ticker,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_period': '30_days',
            'error': error_msg,
            'data_points': 0,
            
            'pattern_detected': {
                'primary_pattern': None,
                'pattern_reliability': 0.0,
                'days_in_formation': 0,
                'completion_probability': 0.0,
                'all_patterns': [],
                'pattern_count': 0
            },
            
            'candlestick_analysis': {
                'total_patterns_found': 0,
                'strongest_pattern': None,
                'reversal_signals_count': 0,
                'continuation_signals_count': 0,
                'pattern_clusters': [],
                'recent_signals': []
            },
            
            'momentum_analysis': {
                'momentum_acceleration': 0.5,
                'current_vs_30day_avg': 1.0,
                'acceleration_trend': 'neutral',
                'momentum_persistence': 0.0,
                'velocity_profile': {},
                'momentum_divergence': None,
                'momentum_quality': 0.0
            },
            
            'breakout_analysis': {
                'breakout_detected': False,
                'breakout_direction': 'neutral',
                'breakout_strength': 0.0,
                'volume_confirmation': False,
                'breakout_target': 0.0,
                'probability_of_follow_through': 0.0,
                'breakout_type': None,
                'confirmation_factors': []
            },
            
            'support_resistance': {
                'key_support_levels': [],
                'key_resistance_levels': [],
                'current_level_strength': 0.0,
                'nearest_support': 0.0,
                'nearest_resistance': 0.0,
                'price_zones': [],
                'level_quality': 0.0
            },
            
            'volume_analysis': {
                'volume_regime': 'unknown',
                '30day_avg_volume': 0,
                'recent_volume_trend': 'unknown',
                'volume_breakout_signal': False,
                'institutional_activity': 'unknown',
                'volume_price_relationship': 'unknown',
                'volume_profile': {},
                'unusual_volume_days': []
            },
            
            'current_state': {
                'current_price': 0.0,
                'daily_range': 0.0,
                'recent_gaps': 0,
                'price_vs_sma20': 0.0,
                'rsi': 50.0,
                'bb_position': 0.5,
                'trend_strength': 0.0
            },
            
            'integration_signals': {
                'momentum_boost_factor': 1.0,
                'technical_boost_factor': 1.0,
                'confidence_boost_factor': 0.0,
                'regime_override_signal': None,
                'risk_adjustment_factor': 1.0
            },
            
            'extended_analysis': {
                'volatility_analysis': {
                    'current_volatility': 0.0,
                    'volatility_trend': 'stable',
                    'volatility_regime': 'normal',
                    'volatility_percentile': 0.5
                },
                'trend_quality': {
                    'trend_strength': 0.0,
                    'trend_consistency': 0.0,
                    'trend_sustainability': 0.0,
                    'trend_phase': 'undefined'
                },
                'pattern_confluence': {
                    'confluence_score': 0.0,
                    'aligned_patterns': [],
                    'conflicting_patterns': [],
                    'dominant_signal': 'neutral'
                },
                'risk_metrics': {
                    'atr_stop_loss': 0.0,
                    'support_stop_loss': 0.0,
                    'risk_reward_ratio': 0.0,
                    'max_drawdown_30d': 0.0,
                    'recovery_time': 0
                }
            },
            
            'ml_features': {},
            
            'analysis_quality': {
                'data_completeness': 0.0,
                'pattern_confidence': 0.0,
                'volume_data_quality': 0.0,
                'overall_quality': 0.0
            }
        }


# Helper functions for integration
def integrate_graph_analysis_with_prediction(graph_analysis: Dict, existing_analysis: Dict) -> Dict:
    """
    Enhanced integration helper function to merge graph analysis with existing prediction engine results
    """
    
    # Extract integration signals
    momentum_boost = graph_analysis['integration_signals']['momentum_boost_factor']
    technical_boost = graph_analysis['integration_signals']['technical_boost_factor']
    confidence_boost = graph_analysis['integration_signals']['confidence_boost_factor']
    regime_override = graph_analysis['integration_signals']['regime_override_signal']
    risk_adjustment = graph_analysis['integration_signals']['risk_adjustment_factor']
    
    # Deep copy to avoid modifying original
    enhanced_analysis = existing_analysis.copy()
    
    # Apply boosts to existing analysis
    if 'composite_scores' in enhanced_analysis:
        # Apply momentum boost
        original_momentum = enhanced_analysis['composite_scores'].get('momentum_component', 0)
        enhanced_analysis['composite_scores']['momentum_component'] = original_momentum * momentum_boost
        
        # Apply technical boost
        original_technical = enhanced_analysis['composite_scores'].get('technical_component', 0)
        enhanced_analysis['composite_scores']['technical_component'] = original_technical * technical_boost
        
        # Update overall signal with risk adjustment
        components = []
        weights = []
        
        if 'momentum_component' in enhanced_analysis['composite_scores']:
            components.append(enhanced_analysis['composite_scores']['momentum_component'])
            weights.append(0.4)
        
        if 'technical_component' in enhanced_analysis['composite_scores']:
            components.append(enhanced_analysis['composite_scores']['technical_component'])
            weights.append(0.3)
        
        if 'sentiment_component' in enhanced_analysis['composite_scores']:
            components.append(enhanced_analysis['composite_scores']['sentiment_component'])
            weights.append(0.3)
        
        if components:
            weighted_signal = sum(c * w for c, w in zip(components, weights)) / sum(weights)
            enhanced_analysis['composite_scores']['overall_signal'] = max(-1, min(1, weighted_signal * risk_adjustment))
    
    # Boost confidence
    if 'enhanced_confidence' in enhanced_analysis:
        enhanced_analysis['enhanced_confidence'] = min(0.95, 
            enhanced_analysis['enhanced_confidence'] + confidence_boost
        )
    
    # Add graph analysis insights
    enhanced_analysis['graph_insights'] = {
        'primary_pattern': graph_analysis['pattern_detected']['primary_pattern'],
        'candlestick_signals': graph_analysis['candlestick_analysis']['recent_signals'],
        'momentum_state': graph_analysis['momentum_analysis']['acceleration_trend'],
        'breakout_status': graph_analysis['breakout_analysis']['breakout_detected'],
        'volume_regime': graph_analysis['volume_analysis']['volume_regime'],
        'key_levels': {
            'support': graph_analysis['support_resistance']['nearest_support'],
            'resistance': graph_analysis['support_resistance']['nearest_resistance']
        }
    }
    
    # Override regime if strong signal
    if regime_override and 'regime_analysis' in enhanced_analysis:
        enhanced_analysis['regime_analysis']['graph_override'] = regime_override
        enhanced_analysis['regime_analysis']['graph_override_applied'] = True
        
        # Adjust regime confidence based on override type
        if 'breakout_confirmed' in regime_override:
            enhanced_analysis['regime_analysis']['regime_confidence'] = max(
                enhanced_analysis['regime_analysis'].get('regime_confidence', 0.5),
                0.8
            )
        elif 'momentum_acceleration' in regime_override:
            enhanced_analysis['regime_analysis']['regime_confidence'] = max(
                enhanced_analysis['regime_analysis'].get('regime_confidence', 0.5),
                0.75
            )
    
    # Add ML features for advanced models
    if 'ml_features' in graph_analysis:
        enhanced_analysis['ml_ready_features'] = graph_analysis['ml_features']
    
    # Add risk metrics
    if 'extended_analysis' in graph_analysis and 'risk_metrics' in graph_analysis['extended_analysis']:
        enhanced_analysis['risk_analysis'] = graph_analysis['extended_analysis']['risk_metrics']
    
    return enhanced_analysis


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Enhanced Graph Analyzer - Advanced Pattern Analysis Engine")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = GraphAnalyzer(use_cache=True)
    
    # Test tickers
    test_tickers = ["AMD"]
    
    for ticker in test_tickers:
        print(f"\n📊 Testing enhanced analysis for {ticker}...")
        print("-" * 40)
        
        try:
            # Run analysis
            analysis = analyzer.analyze_ticker(ticker, days=30, include_extended_analysis=True)
            
            # Display comprehensive results
            print(f"\n✅ Analysis Results for {ticker}:")
            print(f"Data Points: {analysis['data_points']}")
            print(f"Analysis Quality: {analysis['analysis_quality']['overall_quality']:.1%}")
            
            # Chart Patterns
            print(f"\n📈 Chart Patterns:")
            print(f"Primary Pattern: {analysis['pattern_detected']['primary_pattern']}")
            print(f"Pattern Reliability: {analysis['pattern_detected']['pattern_reliability']:.3f}")
            print(f"Total Patterns Found: {analysis['pattern_detected']['pattern_count']}")
            
            # Candlestick Patterns
            print(f"\n🕯️ Candlestick Analysis:")
            print(f"Total Candlestick Patterns: {analysis['candlestick_analysis']['total_patterns_found']}")
            print(f"Strongest Pattern: {analysis['candlestick_analysis']['strongest_pattern']}")
            print(f"Reversal Signals: {analysis['candlestick_analysis']['reversal_signals_count']}")
            print(f"Continuation Signals: {analysis['candlestick_analysis']['continuation_signals_count']}")
            
            if analysis['candlestick_analysis']['pattern_clusters']:
                print(f"Pattern Clusters Detected: {len(analysis['candlestick_analysis']['pattern_clusters'])}")
            
            # Momentum Analysis
            print(f"\n⚡ Momentum Analysis:")
            print(f"Momentum Acceleration: {analysis['momentum_analysis']['momentum_acceleration']:.3f}")
            print(f"Acceleration Trend: {analysis['momentum_analysis']['acceleration_trend']}")
            print(f"Momentum Quality: {analysis['momentum_analysis']['momentum_quality']:.3f}")
            
            if analysis['momentum_analysis']['momentum_divergence']:
                print(f"⚠️ Divergence Detected: {analysis['momentum_analysis']['momentum_divergence']}")
            
            # Breakout Analysis
            print(f"\n🚀 Breakout Analysis:")
            print(f"Breakout Detected: {analysis['breakout_analysis']['breakout_detected']}")
            if analysis['breakout_analysis']['breakout_detected']:
                print(f"Direction: {analysis['breakout_analysis']['breakout_direction']}")
                print(f"Strength: {analysis['breakout_analysis']['breakout_strength']:.3f}")
                print(f"Type: {analysis['breakout_analysis']['breakout_type']}")
                print(f"Target: ${analysis['breakout_analysis']['breakout_target']:.2f}")
                print(f"Follow-through Probability: {analysis['breakout_analysis']['probability_of_follow_through']:.1%}")
                print(f"Confirmation Factors: {', '.join(analysis['breakout_analysis']['confirmation_factors'])}")
            
            # Volume Analysis
            print(f"\n📊 Volume Analysis:")
            print(f"Volume Regime: {analysis['volume_analysis']['volume_regime']}")
            print(f"Recent Trend: {analysis['volume_analysis']['recent_volume_trend']}")
            print(f"Institutional Activity: {analysis['volume_analysis']['institutional_activity']}")
            print(f"Volume-Price Relationship: {analysis['volume_analysis']['volume_price_relationship']}")
            
            # Support/Resistance
            print(f"\n📏 Key Levels:")
            if analysis['support_resistance']['key_support_levels']:
                print(f"Support: ${analysis['support_resistance']['key_support_levels'][0]:.2f}")
            if analysis['support_resistance']['key_resistance_levels']:
                print(f"Resistance: ${analysis['support_resistance']['key_resistance_levels'][0]:.2f}")
            print(f"Level Quality: {analysis['support_resistance']['level_quality']:.3f}")
            
            # Current State
            print(f"\n📍 Current State:")
            print(f"Price: ${analysis['current_state']['current_price']:.2f}")
            print(f"RSI: {analysis['current_state']['rsi']:.1f}")
            print(f"BB Position: {analysis['current_state']['bb_position']:.2f}")
            print(f"Trend Strength: {analysis['current_state']['trend_strength']:.3f}")
            
            # Extended Analysis
            if 'extended_analysis' in analysis:
                print(f"\n🔬 Extended Analysis:")
                vol_analysis = analysis['extended_analysis']['volatility_analysis']
                print(f"Volatility Regime: {vol_analysis['volatility_regime']}")
                print(f"Volatility Trend: {vol_analysis['volatility_trend']}")
                
                trend_quality = analysis['extended_analysis']['trend_quality']
                print(f"Trend Phase: {trend_quality['trend_phase']}")
                print(f"Trend Consistency: {trend_quality['trend_consistency']:.3f}")
                
                confluence = analysis['extended_analysis']['pattern_confluence']
                print(f"Pattern Confluence Score: {confluence['confluence_score']:.3f}")
                print(f"Dominant Signal: {confluence['dominant_signal']}")
            
            # Integration Signals
            print(f"\n🔗 Integration Signals:")
            print(f"Momentum Boost: {analysis['integration_signals']['momentum_boost_factor']:.2f}x")
            print(f"Technical Boost: {analysis['integration_signals']['technical_boost_factor']:.2f}x")
            print(f"Confidence Boost: +{analysis['integration_signals']['confidence_boost_factor']*100:.1f}%")
            print(f"Risk Adjustment: {analysis['integration_signals']['risk_adjustment_factor']:.2f}x")
            
            if analysis['integration_signals']['regime_override_signal']:
                print(f"⚡ Regime Override: {analysis['integration_signals']['regime_override_signal']}")
            
            # ML Features Sample
            if 'ml_features' in analysis:
                print(f"\n🤖 ML Features (Sample):")
                ml_features = analysis['ml_features']
                print(f"Price Momentum 7d: {ml_features['price_momentum_7d']:.3f}")
                print(f"Breakout Strength: {ml_features['breakout_strength']:.3f}")
                print(f"Pattern Reliability: {ml_features['pattern_reliability']:.3f}")
            
        except Exception as e:
            print(f"❌ Analysis failed for {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n\n🎯 Enhanced Graph Analyzer ready for integration!")
    print("💡 Key improvements:")
    print("  • Complete candlestick pattern detection (13 patterns)")
    print("  • Enhanced momentum analysis with divergence detection")
    print("  • Multi-factor breakout confirmation")
    print("  • Advanced volume profiling")
    print("  • ML-ready feature extraction")
    print("  • Risk metrics and volatility analysis")
    print("  • Pattern confluence scoring")
    print("  • Technical indicator integration (RSI, Bollinger Bands, ATR)")
    print("\n📚 Usage: analyzer.analyze_ticker(ticker, days=30, include_extended_analysis=True)")