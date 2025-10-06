#!/usr/bin/env python3
"""
AI Stock Prediction Engine - FIXED VERSION
Handles multi-day predictions, options analysis, Greeks calculation, and trading recommendations
with corrected mathematics and consistent predictions
"""

import os
import sys
import json
import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize_scalar
import math
import re

# Add paths for your existing modules
sys.path.append('src')
sys.path.append('src/data_loaders')

class OptionsAnalyzer:
    """Dedicated class for options analysis and pricing"""
    
    @staticmethod
    def black_scholes_call(S, K, T, r, sigma):
        """Black-Scholes call option pricing"""
        if T <= 0:
            return max(S - K, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        return call_price
    
    @staticmethod
    def black_scholes_put(S, K, T, r, sigma):
        """Black-Scholes put option pricing"""
        if T <= 0:
            return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        return put_price
    
    @staticmethod
    def calculate_greeks(S, K, T, r, sigma, option_type='call'):
        """Calculate option Greeks"""
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == 'call':
            delta = stats.norm.cdf(d1)
        else:
            delta = stats.norm.cdf(d1) - 1
        
        # Gamma (same for calls and puts)
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        if option_type == 'call':
            theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                    r * K * np.exp(-r * T) * stats.norm.cdf(d2)) / 365
        else:
            theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                    r * K * np.exp(-r * T) * stats.norm.cdf(-d2)) / 365
        
        # Vega (same for calls and puts)
        vega = S * stats.norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    @staticmethod
    def implied_volatility(option_price, S, K, T, r, option_type='call'):
        """Calculate implied volatility using Brent's method"""
        if T <= 0:
            return 0
        
        def objective(sigma):
            if option_type == 'call':
                theoretical_price = OptionsAnalyzer.black_scholes_call(S, K, T, r, sigma)
            else:
                theoretical_price = OptionsAnalyzer.black_scholes_put(S, K, T, r, sigma)
            return abs(theoretical_price - option_price)
        
        try:
            result = minimize_scalar(objective, bounds=(0.01, 5.0), method='bounded')
            return result.x
        except:
            return 0.3  # Default volatility if calculation fails
        
from collections import deque  # Add this import at the top

class PredictionErrorTracker:
    """Enhanced error tracker with MAE learning and ticker-specific calibration"""
    
    def __init__(self):
        self.error_log = deque(maxlen=100)  # Keep last 100 predictions
        self.ticker_errors = {}
        
        # NEW: MAE-specific tracking
        self.mae_history = {}  # Track MAE performance by ticker
        self.volatility_mae_profiles = {}  # MAE performance by volatility tier
        
    def validate_graph_analysis(self, graph_analysis):
        """Validate that graph_analysis contains all required components"""
        print("🔍 Validating graph analysis structure...")
        
        try:
            # Check if graph_analysis is a dictionary
            if not isinstance(graph_analysis, dict):
                print("   [ERROR] Graph analysis must be a dictionary")
                return False
            
            # Required top-level keys
            required_keys = [
                'pattern_detected',
                'breakout_analysis', 
                'momentum_analysis',
                'candlestick_analysis',
                'support_resistance'
            ]
            
            # Check for required top-level keys
            missing_keys = []
            for key in required_keys:
                if key not in graph_analysis:
                    missing_keys.append(key)
            
            if missing_keys:
                print(f"   [ERROR] Missing required keys: {missing_keys}")
                return False
            
            # Validate pattern_detected structure
            pattern_detected = graph_analysis['pattern_detected']
            if not isinstance(pattern_detected, dict):
                print("   [ERROR] pattern_detected must be a dictionary")
                return False
            
            pattern_required = ['primary_pattern', 'pattern_reliability']
            for key in pattern_required:
                if key not in pattern_detected:
                    print(f"   [ERROR] pattern_detected missing key: {key}")
                    return False
            
            # Validate pattern_reliability is between 0 and 1
            reliability = pattern_detected.get('pattern_reliability', 0)
            if not isinstance(reliability, (int, float)) or not (0 <= reliability <= 1):
                print("   [ERROR] pattern_reliability must be a number between 0 and 1")
                return False
            
            # Validate breakout_analysis structure
            breakout_analysis = graph_analysis['breakout_analysis']
            if not isinstance(breakout_analysis, dict):
                print("   [ERROR] breakout_analysis must be a dictionary")
                return False
            
            breakout_required = ['breakout_detected', 'breakout_direction', 'breakout_strength']
            for key in breakout_required:
                if key not in breakout_analysis:
                    print(f"   [ERROR] breakout_analysis missing key: {key}")
                    return False
            
            # Validate breakout_detected is boolean
            if not isinstance(breakout_analysis['breakout_detected'], bool):
                print("   [ERROR] breakout_detected must be a boolean")
                return False
            
            # Validate breakout_direction
            valid_directions = ['bullish', 'bearish', 'neutral']
            if breakout_analysis['breakout_direction'] not in valid_directions:
                print(f"   [ERROR] breakout_direction must be one of: {valid_directions}")
                return False
            
            # Validate breakout_strength is between 0 and 1
            strength = breakout_analysis.get('breakout_strength', 0)
            if not isinstance(strength, (int, float)) or not (0 <= strength <= 1):
                print("   [ERROR] breakout_strength must be a number between 0 and 1")
                return False
            
            # Validate momentum_analysis structure
            momentum_analysis = graph_analysis['momentum_analysis']
            if not isinstance(momentum_analysis, dict):
                print("   [ERROR] momentum_analysis must be a dictionary")
                return False
            
            if 'momentum_acceleration' not in momentum_analysis:
                print("   [ERROR] momentum_analysis missing momentum_acceleration")
                return False
            
            # Validate momentum_acceleration is between 0 and 1
            acceleration = momentum_analysis.get('momentum_acceleration', 0.5)
            if not isinstance(acceleration, (int, float)) or not (0 <= acceleration <= 1):
                print("   [ERROR] momentum_acceleration must be a number between 0 and 1")
                return False
            
            # Validate candlestick_analysis structure
            candlestick_analysis = graph_analysis['candlestick_analysis']
            if not isinstance(candlestick_analysis, dict):
                print("   [ERROR] candlestick_analysis must be a dictionary")
                return False
            
            candlestick_required = ['total_patterns_found', 'strongest_pattern']
            for key in candlestick_required:
                if key not in candlestick_analysis:
                    print(f"   [ERROR] candlestick_analysis missing key: {key}")
                    return False
            
            # Validate total_patterns_found is a non-negative integer
            total_patterns = candlestick_analysis.get('total_patterns_found', 0)
            if not isinstance(total_patterns, int) or total_patterns < 0:
                print("   [ERROR] total_patterns_found must be a non-negative integer")
                return False
            
            # Validate support_resistance structure
            support_resistance = graph_analysis['support_resistance']
            if not isinstance(support_resistance, dict):
                print("   [ERROR] support_resistance must be a dictionary")
                return False
            
            sr_required = ['nearest_support', 'nearest_resistance', 'level_quality']
            for key in sr_required:
                if key not in support_resistance:
                    print(f"   [ERROR] support_resistance missing key: {key}")
                    return False
            
            # Validate support/resistance levels are positive numbers
            nearest_support = support_resistance.get('nearest_support', 0)
            nearest_resistance = support_resistance.get('nearest_resistance', 0)
            
            if not isinstance(nearest_support, (int, float)) or nearest_support <= 0:
                print("   [ERROR] nearest_support must be a positive number")
                return False
            
            if not isinstance(nearest_resistance, (int, float)) or nearest_resistance <= 0:
                print("   [ERROR] nearest_resistance must be a positive number")
                return False
            
            if nearest_support >= nearest_resistance:
                print("   [ERROR] nearest_support must be less than nearest_resistance")
                return False
            
            # Validate level_quality is between 0 and 1
            level_quality = support_resistance.get('level_quality', 0)
            if not isinstance(level_quality, (int, float)) or not (0 <= level_quality <= 1):
                print("   [ERROR] level_quality must be a number between 0 and 1")
                return False
            
            print("   [OK] Graph analysis validation passed")
            return True
            
        except Exception as e:
            print(f"   [ERROR] Graph analysis validation error: {e}")
            return False
        
    def generate_default_graph_analysis(self, ticker, excel_data, market_data):
        """Generate basic graph analysis if none provided (emergency fallback)"""
        print(f"[WRENCH] Generating default graph analysis for {ticker}...")
        
        try:
            current_price = market_data['current_price']
            volatility = excel_data.get('volatility', 25.0)
            thirty_day_return = excel_data.get('performance_return_1_month', 0.0)
            rsi = excel_data.get('current_rsi', 50.0)
            sma_20 = excel_data.get('sma_20', current_price)
            sma_50 = excel_data.get('sma_50', current_price)
            
            # Determine primary pattern based on available data
            primary_pattern = 'None'
            pattern_reliability = 0.0
            
            # Simple pattern detection based on price vs moving averages
            if current_price > sma_20 > sma_50 and thirty_day_return > 10:
                if rsi < 70:  # Not overbought
                    primary_pattern = 'ascending_triangle'
                    pattern_reliability = 0.6
                else:
                    primary_pattern = 'bull_flag'
                    pattern_reliability = 0.5
            elif current_price < sma_20 < sma_50 and thirty_day_return < -10:
                if rsi > 30:  # Not oversold
                    primary_pattern = 'descending_triangle'
                    pattern_reliability = 0.6
                else:
                    primary_pattern = 'bear_flag'
                    pattern_reliability = 0.5
            elif abs(thirty_day_return) < 5 and 40 < rsi < 60:
                primary_pattern = 'sideways_consolidation'
                pattern_reliability = 0.4
            
            # Determine breakout based on momentum and volatility
            breakout_detected = False
            breakout_direction = 'neutral'
            breakout_strength = 0.0
            
            if abs(thirty_day_return) > 15 and volatility > 30:
                breakout_detected = True
                breakout_direction = 'bullish' if thirty_day_return > 0 else 'bearish'
                breakout_strength = min(0.8, abs(thirty_day_return) / 25)
                print(f"   [ROCKET] Default breakout detected: {breakout_direction} (strength: {breakout_strength:.2f})")
            
            # Calculate momentum acceleration based on recent performance
            momentum_acceleration = 0.5  # Neutral baseline
            
            if thirty_day_return > 20:
                momentum_acceleration = min(0.9, 0.5 + (thirty_day_return / 100))
            elif thirty_day_return > 10:
                momentum_acceleration = min(0.8, 0.5 + (thirty_day_return / 150))
            elif thirty_day_return < -20:
                momentum_acceleration = max(0.1, 0.5 + (thirty_day_return / 100))
            elif thirty_day_return < -10:
                momentum_acceleration = max(0.2, 0.5 + (thirty_day_return / 150))
            
            # Estimate candlestick patterns based on volatility and momentum
            total_patterns_found = 0
            strongest_pattern = 'None'
            
            if volatility > 40:
                total_patterns_found = 3  # High volatility = more patterns
                if thirty_day_return > 10:
                    strongest_pattern = 'hammer'
                elif thirty_day_return < -10:
                    strongest_pattern = 'shooting_star'
                else:
                    strongest_pattern = 'doji'
            elif volatility > 25:
                total_patterns_found = 2  # Medium volatility
                strongest_pattern = 'spinning_top' if abs(thirty_day_return) < 5 else 'marubozu'
            else:
                total_patterns_found = 1  # Low volatility
                strongest_pattern = 'small_body'
            
            # Calculate support and resistance levels
            # Use simple percentage-based levels if no better data available
            nearest_support = current_price * 0.97  # 3% below current
            nearest_resistance = current_price * 1.03  # 3% above current
            
            # Adjust based on volatility
            volatility_adjustment = volatility / 100 / 4  # Scale volatility impact
            nearest_support = current_price * (1 - (0.03 + volatility_adjustment))
            nearest_resistance = current_price * (1 + (0.03 + volatility_adjustment))
            
            # Level quality based on how much data we have
            level_quality = 0.3  # Low quality since this is generated
            
            # Boost quality if we have moving averages
            if sma_20 != current_price and sma_50 != current_price:
                level_quality = 0.5
                # Use SMA levels as support/resistance if they're reasonable
                if abs(sma_20 - current_price) / current_price < 0.1:  # Within 10%
                    if sma_20 < current_price:
                        nearest_support = sma_20
                    else:
                        nearest_resistance = sma_20
                    level_quality = 0.6
            
            # Create the default graph analysis structure
            default_graph_analysis = {
                'pattern_detected': {
                    'primary_pattern': primary_pattern,
                    'pattern_reliability': pattern_reliability,
                    'detection_method': 'default_generated'
                },
                'breakout_analysis': {
                    'breakout_detected': breakout_detected,
                    'breakout_direction': breakout_direction,
                    'breakout_strength': breakout_strength,
                    'detection_method': 'momentum_based'
                },
                'momentum_analysis': {
                    'momentum_acceleration': momentum_acceleration,
                    'calculation_method': 'thirty_day_return_based'
                },
                'candlestick_analysis': {
                    'total_patterns_found': total_patterns_found,
                    'strongest_pattern': strongest_pattern,
                    'pattern_clusters': [],  # Empty for default
                    'detection_method': 'volatility_estimated'
                },
                'support_resistance': {
                    'nearest_support': nearest_support,
                    'nearest_resistance': nearest_resistance,
                    'level_quality': level_quality,
                    'calculation_method': 'percentage_based'
                },
                'integration_signals': {
                    'momentum_boost_factor': 1.0 + (abs(thirty_day_return) / 100 * 0.1),
                    'technical_boost_factor': 1.0,
                    'confidence_boost_factor': 0.0,  # No boost for default analysis
                    'regime_override_signal': None
                },
                'metadata': {
                    'generated_default': True,
                    'generation_timestamp': datetime.now().isoformat(),
                    'source_data': ['excel_data', 'market_data'],
                    'quality_level': 'basic'
                }
            }
            
            print(f"   [OK] Default graph analysis generated:")
            print(f"   [CHART] Pattern: {primary_pattern} (reliability: {pattern_reliability:.1%})")
            print(f"   [ROCKET] Breakout: {breakout_direction} ({'detected' if breakout_detected else 'none'})")
            print(f"   ⚡ Momentum: {momentum_acceleration:.3f}")
            print(f"   [UP] Support: ${nearest_support:.2f} | Resistance: ${nearest_resistance:.2f}")
            print(f"   [WARNING]  Quality: BASIC (generated from limited data)")
            
            return default_graph_analysis
            
        except Exception as e:
            print(f"   [ERROR] Error generating default graph analysis: {e}")
            
            # Ultimate fallback - minimal structure
            minimal_graph_analysis = {
                'pattern_detected': {
                    'primary_pattern': 'None',
                    'pattern_reliability': 0.0
                },
                'breakout_analysis': {
                    'breakout_detected': False,
                    'breakout_direction': 'neutral',
                    'breakout_strength': 0.0
                },
                'momentum_analysis': {
                    'momentum_acceleration': 0.5
                },
                'candlestick_analysis': {
                    'total_patterns_found': 0,
                    'strongest_pattern': 'None',
                    'pattern_clusters': []
                },
                'support_resistance': {
                    'nearest_support': market_data.get('current_price', 100) * 0.97,
                    'nearest_resistance': market_data.get('current_price', 100) * 1.03,
                    'level_quality': 0.1
                },
                'integration_signals': {
                    'momentum_boost_factor': 1.0,
                    'technical_boost_factor': 1.0,
                    'confidence_boost_factor': 0.0,
                    'regime_override_signal': None
                },
                'metadata': {
                    'generated_default': True,
                    'generation_timestamp': datetime.now().isoformat(),
                    'quality_level': 'minimal'
                }
            }
            
            print(f"   [WARNING]  Using minimal fallback graph analysis")
            return minimal_graph_analysis
        
    def log_prediction_error(self, ticker, predicted_price, actual_price, prediction_date, volatility, predicted_range=None):
        """Enhanced error logging with MAE tracking"""
        error_pct = abs(predicted_price - actual_price) / actual_price * 100
        
        error_record = {
            'ticker': ticker,
            'date': prediction_date,
            'predicted': predicted_price,
            'actual': actual_price,
            'error_pct': error_pct,
            'volatility': volatility,
            'direction_correct': (predicted_price > actual_price) == (predicted_price > actual_price)
        }
        
        # NEW: Add MAE tracking if range provided
        if predicted_range:
            predicted_low, predicted_high = predicted_range
            actual_in_range = predicted_low <= actual_price <= predicted_high
            range_width_pct = ((predicted_high - predicted_low) / predicted_price) * 100
            
            error_record.update({
                'predicted_low': predicted_low,
                'predicted_high': predicted_high,
                'actual_in_range': actual_in_range,
                'range_width_pct': range_width_pct,
                'mae_performance': range_width_pct / 2  # Approximate MAE
            })
        
        self.error_log.append(error_record)
        
        # Store by ticker
        if ticker not in self.ticker_errors:
            self.ticker_errors[ticker] = []
        self.ticker_errors[ticker].append(error_record)
        
        # NEW: Store MAE history
        if predicted_range:
            if ticker not in self.mae_history:
                self.mae_history[ticker] = []
            
            mae_record = {
                'date': prediction_date,
                'predicted_mae': range_width_pct / 2,
                'actual_in_range': actual_in_range,
                'volatility': volatility,
                'range_success': actual_in_range
            }
            self.mae_history[ticker].append(mae_record)
        
    def get_calibration_factor(self, ticker, volatility):
        """Enhanced calibration factor with MAE learning"""
        # Base calibration from traditional error tracking
        base_factor = self._get_traditional_calibration_factor(ticker, volatility)
        
        # NEW: MAE-based calibration adjustment
        mae_adjustment = self._get_mae_calibration_adjustment(ticker, volatility)
        
        # Combine both factors
        combined_factor = base_factor * mae_adjustment
        
        # Bounds checking
        final_factor = max(0.7, min(1.5, combined_factor))
        
        return final_factor
    
    def _get_traditional_calibration_factor(self, ticker, volatility):
        """Original calibration logic"""
        if ticker not in self.ticker_errors or len(self.ticker_errors[ticker]) < 3:
            return 1.0
        
        recent_errors = self.ticker_errors[ticker][-10:]  # Last 10 predictions
        avg_error = np.mean([e['error_pct'] for e in recent_errors])
        
        # Adjust calibration based on systematic over/under-estimation
        if avg_error > 3.0:  # Consistently overestimating ranges
            return 0.85
        elif avg_error < 1.0:  # Consistently underestimating
            return 1.15
        
        return 1.0
    
    def _get_mae_calibration_adjustment(self, ticker, volatility):
        """NEW: MAE-based calibration adjustment"""
        if ticker not in self.mae_history or len(self.mae_history[ticker]) < 3:
            return 1.0
        
        recent_mae_records = self.mae_history[ticker][-10:]  # Last 10 predictions
        
        # Calculate MAE success rate
        range_success_rate = np.mean([r['range_success'] for r in recent_mae_records])
        
        # Calculate average predicted vs required MAE
        avg_predicted_mae = np.mean([r['predicted_mae'] for r in recent_mae_records])
        
        # Determine volatility tier for comparison
        volatility_tier = self._get_volatility_tier(volatility)
        target_mae = self._get_target_mae_for_tier(volatility_tier)
        
        # MAE calibration logic
        if range_success_rate < 0.7:  # Less than 70% success rate
            if avg_predicted_mae > target_mae:
                # Ranges too wide but still failing - need better calibration
                return 0.9  # Slightly tighter ranges
            else:
                # Ranges too narrow - widen them
                return 1.1
        elif range_success_rate > 0.9:  # Very high success rate
            if avg_predicted_mae > target_mae * 1.3:
                # Ranges much wider than needed - tighten significantly
                return 0.8
            elif avg_predicted_mae > target_mae * 1.1:
                # Ranges somewhat wide - tighten moderately
                return 0.95
        
        return 1.0  # No adjustment needed
    
    def _get_volatility_tier(self, volatility):
        """Helper to determine volatility tier"""
        if volatility < 20:
            return 'low'
        elif volatility < 35:
            return 'medium'
        elif volatility < 50:
            return 'high'
        else:
            return 'extreme'
    
    def _get_target_mae_for_tier(self, tier):
        """Helper to get target MAE for volatility tier"""
        targets = {
            'low': 1.5,
            'medium': 2.2,
            'high': 3.0,
            'extreme': 4.0
        }
        return targets.get(tier, 2.5)
    
    def get_mae_performance_summary(self, ticker=None):
        """Get MAE performance summary for analysis"""
        if ticker:
            if ticker not in self.mae_history:
                return None
            records = self.mae_history[ticker]
        else:
            records = []
            for ticker_records in self.mae_history.values():
                records.extend(ticker_records)
        
        if not records:
            return None
        
        recent_records = records[-20:]  # Last 20 predictions
        
        summary = {
            'total_predictions': len(recent_records),
            'range_success_rate': np.mean([r['range_success'] for r in recent_records]),
            'avg_predicted_mae': np.mean([r['predicted_mae'] for r in recent_records]),
            'avg_volatility': np.mean([r['volatility'] for r in recent_records]),
            'by_volatility_tier': {}
        }
        
        # Break down by volatility tier
        for tier in ['low', 'medium', 'high', 'extreme']:
            tier_records = [r for r in recent_records if self._get_volatility_tier(r['volatility']) == tier]
            if tier_records:
                summary['by_volatility_tier'][tier] = {
                    'count': len(tier_records),
                    'success_rate': np.mean([r['range_success'] for r in tier_records]),
                    'avg_mae': np.mean([r['predicted_mae'] for r in tier_records]),
                    'target_mae': self._get_target_mae_for_tier(tier)
                }
        
        return summary
    
    def should_apply_emergency_tightening(self, ticker, volatility):
        """Determine if emergency MAE tightening is needed"""
        if ticker not in self.mae_history or len(self.mae_history[ticker]) < 5:
            return False
        
        recent_records = self.mae_history[ticker][-5:]  # Last 5 predictions
        recent_success_rate = np.mean([r['range_success'] for r in recent_records])
        recent_avg_mae = np.mean([r['predicted_mae'] for r in recent_records])
        
        volatility_tier = self._get_volatility_tier(volatility)
        target_mae = self._get_target_mae_for_tier(volatility_tier)
        
        # Emergency tightening conditions
        conditions = [
            recent_success_rate < 0.6,  # Very low success rate
            recent_avg_mae > target_mae * 1.5,  # MAE much higher than target
            len([r for r in recent_records if r['predicted_mae'] > target_mae * 2]) >= 3  # Multiple very wide predictions
        ]
        
        return any(conditions)
    
    def get_ticker_mae_recommendation(self, ticker, volatility):
        """Get specific MAE recommendations for a ticker"""
        if ticker not in self.mae_history:
            return "No historical data available"
        
        mae_summary = self.get_mae_performance_summary(ticker)
        if not mae_summary:
            return "Insufficient data for recommendation"
        
        success_rate = mae_summary['range_success_rate']
        avg_mae = mae_summary['avg_predicted_mae']
        volatility_tier = self._get_volatility_tier(volatility)
        target_mae = self._get_target_mae_for_tier(volatility_tier)
        
        if success_rate < 0.7:
            if avg_mae > target_mae * 1.2:
                return f"TIGHTEN RANGES: Success rate {success_rate:.1%}, MAE {avg_mae:.1f}% vs target {target_mae:.1f}%"
            else:
                return f"WIDEN RANGES: Success rate {success_rate:.1%} too low despite narrow ranges"
        elif success_rate > 0.9 and avg_mae > target_mae * 1.3:
            return f"OPTIMIZE RANGES: High success rate {success_rate:.1%} but ranges too wide {avg_mae:.1f}%"
        else:
            return f"RANGES OK: Success rate {success_rate:.1%}, MAE {avg_mae:.1f}% vs target {target_mae:.1f}%"
        
class MultiSourceNewsAnalyzer:
    """Multi-source news analyzer for comprehensive sentiment analysis"""
    
    def __init__(self, api_keys=None):
        self.api_keys = api_keys or {}
        self.source_weights = {
            'phase3': 0.35,        # Your existing Phase 3 system
            'alpha_vantage': 0.25, # Alpha Vantage news
            'finnhub': 0.20,       # Finnhub news
            'newsapi': 0.15,       # NewsAPI
            'fallback': 0.05       # Fallback/synthetic data
        }
        
    def get_alpha_vantage_news(self, ticker):
        """Get news from Alpha Vantage"""
        if not self.api_keys.get('alpha_vantage'):
            print("   [WARNING] Alpha Vantage API key not available")
            return {'sentiment': 0.0, 'articles': 0, 'confidence': 0.0}
        
        try:
            import requests
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'apikey': self.api_keys['alpha_vantage'],
                'limit': 50
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'feed' in data and len(data['feed']) > 0:
                sentiment_scores = []
                relevance_scores = []
                
                for article in data['feed'][:20]:  # Process top 20 articles
                    for ticker_sentiment in article.get('ticker_sentiment', []):
                        if ticker_sentiment['ticker'] == ticker:
                            sentiment_scores.append(float(ticker_sentiment['ticker_sentiment_score']))
                            relevance_scores.append(float(ticker_sentiment['relevance_score']))
                
                if sentiment_scores:
                    # Weight by relevance
                    weighted_sentiment = sum(s * r for s, r in zip(sentiment_scores, relevance_scores))
                    total_relevance = sum(relevance_scores)
                    
                    final_sentiment = weighted_sentiment / total_relevance if total_relevance > 0 else 0
                    confidence = min(0.9, len(sentiment_scores) / 20)  # Confidence based on article count
                    
                    print(f"   📰 Alpha Vantage: {len(sentiment_scores)} articles, sentiment: {final_sentiment:+.3f}")
                    return {
                        'sentiment': final_sentiment,
                        'articles': len(sentiment_scores),
                        'confidence': confidence,
                        'source': 'alpha_vantage'
                    }
            
            return {'sentiment': 0.0, 'articles': 0, 'confidence': 0.0}
            
        except Exception as e:
            print(f"   [ERROR] Alpha Vantage news error: {e}")
            return {'sentiment': 0.0, 'articles': 0, 'confidence': 0.0}
    
    def get_finnhub_news(self, ticker):
        """Get news from Finnhub"""
        if not self.api_keys.get('finnhub'):
            print("   [WARNING] Finnhub API key not available")
            return {'sentiment': 0.0, 'articles': 0, 'confidence': 0.0}
        
        try:
            import requests
            from datetime import datetime, timedelta
            
            # Try to import TextBlob, but provide fallback
            try:
                from textblob import TextBlob
            except ImportError:
                print("   [WARNING] TextBlob not available, using basic sentiment")
                # Simple fallback sentiment analysis
                def basic_sentiment(text):
                    positive_words = ['good', 'great', 'excellent', 'positive', 'strong', 'growth', 'profit', 'gain']
                    negative_words = ['bad', 'poor', 'weak', 'negative', 'loss', 'decline', 'drop', 'fall']
                    
                    text_lower = text.lower()
                    pos_count = sum(1 for word in positive_words if word in text_lower)
                    neg_count = sum(1 for word in negative_words if word in text_lower)
                    
                    if pos_count > neg_count:
                        return type('obj', (object,), {'sentiment': type('obj', (object,), {'polarity': 0.3})})()
                    elif neg_count > pos_count:
                        return type('obj', (object,), {'sentiment': type('obj', (object,), {'polarity': -0.3})})()
                    else:
                        return type('obj', (object,), {'sentiment': type('obj', (object,), {'polarity': 0.0})})()
                
                TextBlob = lambda text: basic_sentiment(text)
            
            # Get company news from last 7 days
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            url = f"https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': ticker,
                'from': from_date,
                'to': to_date,
                'token': self.api_keys['finnhub']
            }
            
            response = requests.get(url, params=params, timeout=30)
            news_data = response.json()
            
            if isinstance(news_data, list) and len(news_data) > 0:
                sentiment_scores = []
                
                for article in news_data[:15]:  # Process top 15 articles
                    # Combine headline and summary for sentiment
                    text = f"{article.get('headline', '')} {article.get('summary', '')}"
                    if text.strip():
                        blob = TextBlob(text)
                        sentiment_scores.append(blob.sentiment.polarity)
                
                if sentiment_scores:
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                    confidence = min(0.8, len(sentiment_scores) / 15)
                    
                    print(f"   📰 Finnhub: {len(sentiment_scores)} articles, sentiment: {avg_sentiment:+.3f}")
                    return {
                        'sentiment': avg_sentiment,
                        'articles': len(sentiment_scores),
                        'confidence': confidence,
                        'source': 'finnhub'
                    }
            
            return {'sentiment': 0.0, 'articles': 0, 'confidence': 0.0}
            
        except Exception as e:
            print(f"   [ERROR] Finnhub news error: {e}")
            return {'sentiment': 0.0, 'articles': 0, 'confidence': 0.0}
    
    def get_newsapi_sentiment(self, ticker):
        """Get general news sentiment from NewsAPI"""
        if not self.api_keys.get('newsapi'):
            print("   [WARNING] NewsAPI key not available")
            return {'sentiment': 0.0, 'articles': 0, 'confidence': 0.0}
        
        try:
            import requests
            
            # Try to import TextBlob with fallback
            try:
                from textblob import TextBlob
            except ImportError:
                # Simple fallback sentiment
                def basic_sentiment(text):
                    positive_words = ['good', 'great', 'excellent', 'positive', 'strong', 'growth', 'profit']
                    negative_words = ['bad', 'poor', 'weak', 'negative', 'loss', 'decline', 'drop']
                    
                    text_lower = text.lower()
                    pos_count = sum(1 for word in positive_words if word in text_lower)
                    neg_count = sum(1 for word in negative_words if word in text_lower)
                    
                    if pos_count > neg_count:
                        return type('obj', (object,), {'sentiment': type('obj', (object,), {'polarity': 0.3})})()
                    elif neg_count > pos_count:
                        return type('obj', (object,), {'sentiment': type('obj', (object,), {'polarity': -0.3})})()
                    else:
                        return type('obj', (object,), {'sentiment': type('obj', (object,), {'polarity': 0.0})})()
                
                TextBlob = lambda text: basic_sentiment(text)
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': ticker,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20,
                'apiKey': self.api_keys['newsapi']
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if data.get('status') == 'ok' and data.get('articles'):
                sentiment_scores = []
                
                for article in data['articles'][:15]:
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                    if text.strip():
                        blob = TextBlob(text)
                        sentiment_scores.append(blob.sentiment.polarity)
                
                if sentiment_scores:
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                    confidence = min(0.7, len(sentiment_scores) / 15)
                    
                    print(f"   📰 NewsAPI: {len(sentiment_scores)} articles, sentiment: {avg_sentiment:+.3f}")
                    return {
                        'sentiment': avg_sentiment,
                        'articles': len(sentiment_scores),
                        'confidence': confidence,
                        'source': 'newsapi'
                    }
            
            return {'sentiment': 0.0, 'articles': 0, 'confidence': 0.0}
            
        except Exception as e:
            print(f"   [ERROR] NewsAPI error: {e}")
            return {'sentiment': 0.0, 'articles': 0, 'confidence': 0.0}
    
    def aggregate_multi_source_sentiment(self, ticker, phase3_data=None):
        """Aggregate sentiment from multiple sources"""
        print(f"   🔄 Aggregating multi-source news for {ticker}...")
        
        sources = {}
        
        # Get Phase 3 data (your existing system)
        if phase3_data:
            sources['phase3'] = {
                'sentiment': phase3_data.get('sentiment_1d', 0.0),
                'articles': phase3_data.get('news_volume_1d', 0),
                'confidence': phase3_data.get('confidence_score', 0.5)
            }
        
        # Get data from other sources
        sources['alpha_vantage'] = self.get_alpha_vantage_news(ticker)
        sources['finnhub'] = self.get_finnhub_news(ticker)
        sources['newsapi'] = self.get_newsapi_sentiment(ticker)
        
        # Calculate weighted aggregation
        total_weight = 0
        weighted_sentiment = 0
        total_articles = 0
        active_sources = 0
        
        for source_name, source_data in sources.items():
            if source_data and source_data.get('articles', 0) > 0:
                weight = self.source_weights.get(source_name, 0.05)
                confidence = source_data.get('confidence', 0.5)
                
                # Adjust weight by confidence
                adjusted_weight = weight * (0.5 + confidence * 0.5)
                
                weighted_sentiment += source_data['sentiment'] * adjusted_weight
                total_weight += adjusted_weight
                total_articles += source_data['articles']
                active_sources += 1
        
        # Calculate final aggregated sentiment
        if total_weight > 0:
            final_sentiment = weighted_sentiment / total_weight
        else:
            final_sentiment = 0.0
        
        # Calculate aggregate confidence
        if active_sources > 0:
            base_confidence = 0.4 + (active_sources / 4 * 0.4)  # More sources = higher confidence
            article_bonus = min(0.2, total_articles / 50)  # More articles = higher confidence
            aggregate_confidence = min(0.95, base_confidence + article_bonus)
        else:
            aggregate_confidence = 0.3
        
        print(f"   [OK] Multi-source aggregation complete:")
        print(f"   [CHART] Active sources: {active_sources}, Total articles: {total_articles}")
        print(f"   [UP] Final sentiment: {final_sentiment:+.3f}, Confidence: {aggregate_confidence:.3f}")
        
        return {
            'aggregated_sentiment': final_sentiment,
            'total_articles': total_articles,
            'active_sources': active_sources,
            'aggregate_confidence': aggregate_confidence,
            'source_breakdown': sources,
            'multi_source_enabled': True
        }
        
class MarketRegimeDetector:
    """Detect market regimes for adaptive weight adjustment"""
    
    def __init__(self):
        self.regime_thresholds = {
            'volatility': {'low': 20, 'high': 50},
            'trend_strength': {'weak': 0.1, 'strong': 0.3},
            'news_volume': {'low': 5, 'high': 15}
        }
    
    def detect_market_regime(self, excel_data, market_data, news_data, graph_analysis):
        """Detect current market regime based on multiple factors INCLUDING GRAPH ANALYSIS"""
        print("🔍 Detecting market regime for adaptive weights with GRAPH ANALYSIS...")
        
        # Factor 1: Volatility regime
        volatility = excel_data.get('volatility', 25.0)
        if volatility > self.regime_thresholds['volatility']['high']:
            volatility_regime = 'high'
        elif volatility < self.regime_thresholds['volatility']['low']:
            volatility_regime = 'low'
        else:
            volatility_regime = 'medium'
        
        # Factor 2: Trend strength
        thirty_day_return = excel_data.get('performance_return_1_month', 0.0)
        trend_strength = abs(thirty_day_return / 100)
        
        if trend_strength > self.regime_thresholds['trend_strength']['strong']:
            trend_regime = 'strong'
        elif trend_strength < self.regime_thresholds['trend_strength']['weak']:
            trend_regime = 'weak'
        else:
            trend_regime = 'moderate'
        
        # Factor 3: News volume regime
        news_volume = news_data.get('total_articles', news_data.get('news_volume_1d', 0))
        if news_volume > self.regime_thresholds['news_volume']['high']:
            news_regime = 'high'
        elif news_volume < self.regime_thresholds['news_volume']['low']:
            news_regime = 'low'
        else:
            news_regime = 'medium'
        
        # Factor 4: Technical alignment (ENHANCED WITH GRAPH ANALYSIS)
        current_price = market_data.get('current_price', 0)
        sma_20 = excel_data.get('sma_20', current_price)
        sma_50 = excel_data.get('sma_50', current_price)
        
        # Base technical regime
        if current_price > sma_20 > sma_50:
            base_technical_regime = 'bullish_aligned'
        elif current_price < sma_20 < sma_50:
            base_technical_regime = 'bearish_aligned'
        else:
            base_technical_regime = 'mixed'
        
        # NEW: GRAPH ANALYSIS TECHNICAL ENHANCEMENT
        graph_technical_regime = base_technical_regime
        graph_technical_strength = 0.0
        graph_override_applied = False
        
        # Check for graph analysis data in news_data (passed from mathematical analysis)
        if news_data.get('graph_analysis_available', False):
            try:
                print("   [UP] Integrating graph analysis into regime detection...")
                
                # Extract graph analysis signals
                primary_pattern = news_data.get('primary_pattern')
                breakout_detected = news_data.get('breakout_detected', False)
                momentum_acceleration = news_data.get('momentum_acceleration', 0.5)
                
                # Pattern-based technical regime enhancement
                bullish_patterns = ['ascending_triangle', 'cup_and_handle', 'bull_flag', 'falling_wedge', 'double_bottom']
                bearish_patterns = ['descending_triangle', 'bear_flag', 'rising_wedge', 'head_and_shoulders', 'double_top']
                
                pattern_technical_boost = 0.0
                if primary_pattern in bullish_patterns:
                    pattern_technical_boost = 0.3  # Boost toward bullish
                    if base_technical_regime == 'mixed':
                        graph_technical_regime = 'pattern_bullish'
                        graph_override_applied = True
                        print(f"   [UP] Pattern override: {primary_pattern} → pattern_bullish")
                elif primary_pattern in bearish_patterns:
                    pattern_technical_boost = -0.3  # Boost toward bearish
                    if base_technical_regime == 'mixed':
                        graph_technical_regime = 'pattern_bearish'
                        graph_override_applied = True
                        print(f"   [DOWN] Pattern override: {primary_pattern} → pattern_bearish")
                
                # Breakout-based regime enhancement (STRONGEST SIGNAL)
                if breakout_detected:
                    if base_technical_regime == 'mixed' or pattern_technical_boost == 0:
                        graph_technical_regime = 'breakout_regime'
                        graph_override_applied = True
                        print(f"   [ROCKET] BREAKOUT REGIME OVERRIDE applied")
                    else:
                        # Strengthen existing regime
                        graph_technical_strength += 0.4
                        print(f"   [ROCKET] Breakout strengthens {graph_technical_regime} regime")
                
                # Momentum acceleration enhancement
                if momentum_acceleration > 0.75:  # Very strong momentum
                    momentum_boost = 0.2
                    graph_technical_strength += momentum_boost
                    print(f"   ⚡ Strong momentum acceleration boosts regime strength: +{momentum_boost:.1f}")
                elif momentum_acceleration < 0.25:  # Very weak momentum
                    momentum_penalty = -0.2
                    graph_technical_strength += momentum_penalty
                    print(f"   ⚡ Weak momentum acceleration reduces regime strength: {momentum_penalty:.1f}")
                
                # Calculate overall graph technical strength
                graph_technical_strength += abs(pattern_technical_boost)
                
            except Exception as graph_error:
                print(f"   [WARNING] Graph analysis integration failed: {graph_error}")
                graph_technical_regime = base_technical_regime
        
        technical_regime = graph_technical_regime
        
        # Factor 5: Multi-source news availability
        multi_source_enabled = news_data.get('multi_source_enabled', False)
        active_sources = news_data.get('active_sources', 1)
        
        if multi_source_enabled and active_sources >= 3:
            news_quality_regime = 'rich'
        elif multi_source_enabled and active_sources >= 2:
            news_quality_regime = 'moderate'
        else:
            news_quality_regime = 'limited'
        
        # NEW: Factor 6: Graph Analysis Pattern Regime
        graph_pattern_regime = 'none'
        if news_data.get('graph_analysis_available', False):
            primary_pattern = news_data.get('primary_pattern')
            breakout_detected = news_data.get('breakout_detected', False)
            
            if breakout_detected:
                graph_pattern_regime = 'breakout_active'
            elif primary_pattern and primary_pattern != 'None':
                if primary_pattern in ['ascending_triangle', 'cup_and_handle', 'bull_flag', 'falling_wedge', 'double_bottom']:
                    graph_pattern_regime = 'bullish_patterns'
                elif primary_pattern in ['descending_triangle', 'bear_flag', 'rising_wedge', 'head_and_shoulders', 'double_top']:
                    graph_pattern_regime = 'bearish_patterns'
                else:
                    graph_pattern_regime = 'neutral_patterns'
            else:
                graph_pattern_regime = 'no_clear_patterns'
        
        # Determine overall regime (ENHANCED WITH GRAPH ANALYSIS)
        regime_factors = {
            'volatility': volatility_regime,
            'trend_strength': trend_regime,
            'news_volume': news_regime,
            'technical_alignment': technical_regime,
            'news_quality': news_quality_regime,
            'graph_pattern_regime': graph_pattern_regime,  # NEW
            'graph_technical_strength': graph_technical_strength,  # NEW
            'graph_override_applied': graph_override_applied  # NEW
        }
        
        # ENHANCED: Main regime classification with graph analysis priority
        main_regime = 'mixed_signals'  # Default
        
        # PRIORITY 1: Graph Analysis Breakout Override (HIGHEST PRIORITY)
        if news_data.get('breakout_detected', False) and graph_technical_strength > 0.6:
            main_regime = 'technical_breakout'
            print(f"   [ROCKET] BREAKOUT REGIME: Technical breakout detected with strength {graph_technical_strength:.1f}")
        
        # PRIORITY 2: Strong Graph Pattern + Technical Alignment
        elif graph_override_applied and graph_technical_strength > 0.4:
            if graph_technical_regime in ['pattern_bullish', 'pattern_bearish']:
                main_regime = 'pattern_trending'
                print(f"   [UP] PATTERN REGIME: {graph_technical_regime} with strength {graph_technical_strength:.1f}")
            else:
                main_regime = 'graph_technical_trending'
        
        # PRIORITY 3: Original regime logic (enhanced)
        elif volatility_regime == 'high' and trend_regime == 'strong':
            # Check if graph analysis supports or contradicts
            if graph_technical_strength > 0.3:
                main_regime = 'high_vol_trending_confirmed'  # Graph confirms trend
            else:
                main_regime = 'high_vol_trending'
        
        elif volatility_regime == 'low' and trend_regime == 'weak':
            # Check for graph patterns in low volatility
            if graph_pattern_regime in ['bullish_patterns', 'bearish_patterns']:
                main_regime = 'low_vol_pattern_forming'  # Patterns forming in calm market
            else:
                main_regime = 'low_vol_sideways'
        
        elif news_regime == 'high' and news_quality_regime == 'rich':
            # Check if graph analysis aligns with news
            if news_data.get('graph_analysis_integrated', False) and graph_technical_strength > 0.2:
                main_regime = 'news_driven_confirmed'  # Technical patterns support news
            else:
                main_regime = 'news_driven'
        
        elif technical_regime in ['bullish_aligned', 'bearish_aligned', 'pattern_bullish', 'pattern_bearish']:
            if graph_technical_strength > 0.3:
                main_regime = 'technical_trending_strong'  # Graph analysis strengthens technical signals
            else:
                main_regime = 'technical_trending'
        
        # PRIORITY 4: Graph-specific regimes
        elif graph_pattern_regime == 'breakout_active':
            main_regime = 'technical_breakout'
        elif graph_pattern_regime in ['bullish_patterns', 'bearish_patterns'] and graph_technical_strength > 0.25:
            main_regime = 'pattern_formation'
        else:
            # Check for any graph enhancement of mixed signals
            if graph_technical_strength > 0.2:
                main_regime = 'mixed_signals_with_patterns'
            else:
                main_regime = 'mixed_signals'
        
        # Calculate ENHANCED regime confidence with graph analysis
        base_confidence = self.calculate_regime_confidence(regime_factors)
        
        # Graph analysis confidence boost
        graph_confidence_boost = 0.0
        if news_data.get('graph_analysis_available', False):
            # Strong patterns increase confidence
            if graph_technical_strength > 0.5:
                graph_confidence_boost += 0.15  # Strong patterns boost confidence
            elif graph_technical_strength > 0.3:
                graph_confidence_boost += 0.10  # Moderate patterns boost confidence
            elif graph_technical_strength > 0.1:
                graph_confidence_boost += 0.05  # Weak patterns minimal boost
            
            # Breakout detection major confidence boost
            if news_data.get('breakout_detected', False):
                graph_confidence_boost += 0.12
                print(f"   [ROCKET] Breakout detection confidence boost: +0.12")
            
            # Pattern override confidence boost
            if graph_override_applied:
                graph_confidence_boost += 0.08
                print(f"   [UP] Pattern override confidence boost: +0.08")
        
        enhanced_confidence = min(0.95, base_confidence + graph_confidence_boost)
        
        regime_data = {
            'main_regime': main_regime,
            'factors': regime_factors,
            'volatility_level': volatility,
            'trend_strength': trend_strength,
            'news_volume': news_volume,
            'active_sources': active_sources,
            'regime_confidence': enhanced_confidence,
            
            # NEW: Graph analysis specific data
            'graph_analysis_applied': news_data.get('graph_analysis_available', False),
            'graph_technical_strength': graph_technical_strength,
            'graph_pattern_regime': graph_pattern_regime,
            'graph_override_applied': graph_override_applied,
            'graph_confidence_boost': graph_confidence_boost,
            'base_confidence': base_confidence,
            'enhanced_confidence': enhanced_confidence,
            
            # Enhanced regime metadata
            'regime_priority': 'graph_breakout' if 'breakout' in main_regime else 'graph_pattern' if 'pattern' in main_regime else 'traditional',
            'technical_regime_enhanced': technical_regime != base_technical_regime,
            'graph_integration_successful': news_data.get('graph_analysis_available', False) and graph_technical_strength > 0
        }
        
        print(f"   [CHART] Market regime detected: {main_regime}")
        print(f"   [CHART] Regime factors: Vol={volatility_regime}, Trend={trend_regime}, News={news_regime}, Tech={technical_regime}")
        print(f"   [CHART] Enhanced confidence: {enhanced_confidence:.3f} (base: {base_confidence:.3f}, graph boost: +{graph_confidence_boost:.3f})")
        if news_data.get('graph_analysis_available', False):
            print(f"   [UP] Graph integration: Pattern={graph_pattern_regime}, Strength={graph_technical_strength:.2f}, Override={graph_override_applied}")
        
        return regime_data
    
    def calculate_regime_confidence(self, regime_factors):
        """Calculate confidence in regime detection"""
        # Simple confidence based on factor alignment
        clear_signals = 0
        total_factors = len(regime_factors)
        
        # Check for clear signals
        if regime_factors['volatility'] in ['high', 'low']:
            clear_signals += 1
        if regime_factors['trend_strength'] in ['strong', 'weak']:
            clear_signals += 1
        if regime_factors['news_volume'] in ['high', 'low']:
            clear_signals += 1
        if regime_factors['technical_alignment'] in ['bullish_aligned', 'bearish_aligned']:
            clear_signals += 1
        if regime_factors['news_quality'] in ['rich', 'limited']:
            clear_signals += 1
        
        return clear_signals / total_factors

    def get_adaptive_weights(self, regime_data, prediction_days):
        """Get adaptive weights based on detected regime"""
        main_regime = regime_data['main_regime']
        confidence = regime_data['regime_confidence']
        
        print(f"   [TARGET] Calculating adaptive weights for {main_regime} regime...")
        
        # Base weights (default)
        base_weights = {
            "momentum": 0.30,
            "technical": 0.35,
            "news": 0.30,
            "volatility": 0.05
        }
        
        # Regime-specific weight adjustments
        if main_regime == 'high_vol_trending':
            # High volatility + strong trend = favor momentum, reduce technical
            weights = {
                "momentum": 0.45,
                "technical": 0.25,
                "news": 0.25,
                "volatility": 0.05
            }
            
        elif main_regime == 'low_vol_sideways':
            # Low volatility + weak trend = favor technical analysis
            weights = {
                "momentum": 0.20,
                "technical": 0.50,
                "news": 0.25,
                "volatility": 0.05
            }
            
        elif main_regime == 'news_driven':
            # High news volume + rich sources = favor news
            weights = {
                "momentum": 0.25,
                "technical": 0.25,
                "news": 0.45,
                "volatility": 0.05
            }
            
        elif main_regime == 'technical_trending':
            # Strong technical alignment = favor technical + momentum
            weights = {
                "momentum": 0.35,
                "technical": 0.40,
                "news": 0.20,
                "volatility": 0.05
            }
            
        else:  # mixed_signals
            # Use balanced approach
            weights = base_weights.copy()
        
        # Apply confidence-based blending
        final_weights = {}
        for key in weights:
            # High confidence = use regime weights, low confidence = blend with base
            final_weights[key] = weights[key] * confidence + base_weights[key] * (1 - confidence)
        
        # Ensure weights sum to 1.0
        total_weight = sum(final_weights.values())
        for key in final_weights:
            final_weights[key] /= total_weight
        
        # Time-based adjustments
        if prediction_days <= 5:
            # Short-term: boost news and technical, reduce momentum
            final_weights["news"] *= 1.1
            final_weights["technical"] *= 1.05
            final_weights["momentum"] *= 0.9
        elif prediction_days > 20:
            # Long-term: boost momentum, reduce news
            final_weights["momentum"] *= 1.1
            final_weights["news"] *= 0.9
            final_weights["technical"] *= 0.95
        
        # Renormalize after time adjustments
        total_weight = sum(final_weights.values())
        for key in final_weights:
            final_weights[key] /= total_weight
        
        print(f"   [OK] Adaptive weights: Mom={final_weights['momentum']:.3f}, Tech={final_weights['technical']:.3f}, News={final_weights['news']:.3f}")
        
        return final_weights
    
       
class StockPredictionEngine:
    """Enhanced engine for stock prediction and options analysis - FIXED VERSION"""
    
    def __init__(self):
        self.alpha_key = None
        self.claude_key = None
        self.options_analyzer = OptionsAnalyzer()
        self.load_api_keys()
        self.error_tracker = PredictionErrorTracker()
    
    def load_api_keys(self):
        """Load API keys from .env file"""
        try:
            with open('.env', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('ALPHA_VANTAGE_API_KEY='):
                        self.alpha_key = line.split('=', 1)[1].strip()
                    elif line.startswith('CLAUDE_API_KEY='):
                        self.claude_key = line.split('=', 1)[1].strip()
        except FileNotFoundError:
            print("[ERROR] .env file not found. Please create one with your API keys.")
    
    def load_excel_historical_data(self, ticker, excel_file_path=None):
        """Load historical data from user-selected Excel file"""
        print(f"[CHART] Loading Excel analysis for {ticker}...")
        
        if excel_file_path is None:
            print("   [WARNING] No Excel file provided - using fallback data")
            return {
                'performance_return_1_month': 6.19,
                'volatility': 25.0,
                'sector': 'Technology',
                'avg_daily_change': 0.5,
                'excel_recommendation': 'Hold',
                'excel_risk_level': 'Moderate',
                'recent_high': 180.0,
                'recent_low': 150.0,
                'sma_20': 165.0,
                'sma_50': 160.0,
                'sma_200': 155.0,
                'current_rsi': 55.0
            }
        
        try:
            # Import the fixed Excel loader
            from excel_loader import ExcelDataLoader
            
            loader = ExcelDataLoader()
            loader.excel_path = excel_file_path
            
            print(f"   📂 Loading from: {excel_file_path}")
            
            # Load all data using the fixed loader
            all_excel_data = loader.load_all_data()
            
            if all_excel_data:
                historical_analysis = {}
                
                # Get the PROPER VOLATILITY from Technical Analysis
                proper_volatility = loader.get_proper_volatility()
                historical_analysis['volatility'] = proper_volatility
                historical_analysis['calculated_volatility'] = proper_volatility
                historical_analysis['annualized_volatility'] = proper_volatility
                print(f"   [OK] USING PROPER VOLATILITY: {proper_volatility:.2f}%")
                
                # Extract comprehensive data
                if loader.raw_data is not None and not loader.raw_data.empty:
                    raw_data = loader.raw_data
                    print(f"   [UP] Historical data: {len(raw_data)} days")
                    
                    # Process daily changes
                    if 'Daily Change %' in raw_data.columns:
                        try:
                            daily_changes = pd.to_numeric(raw_data['Daily Change %'], errors='coerce').dropna()
                            if len(daily_changes) > 0:
                                historical_analysis['avg_daily_change'] = float(daily_changes.mean())
                                historical_analysis['max_gain'] = float(daily_changes.max())
                                historical_analysis['max_loss'] = float(daily_changes.min())
                                historical_analysis['positive_days_pct'] = float((daily_changes > 0).mean() * 100)
                        except Exception as e:
                            print(f"   [WARNING] Error processing daily changes: {e}")
                    
                    # Support/resistance levels
                    try:
                        if 'High' in raw_data.columns and 'Low' in raw_data.columns:
                            highs = pd.to_numeric(raw_data['High'], errors='coerce').dropna()
                            lows = pd.to_numeric(raw_data['Low'], errors='coerce').dropna()
                            if len(highs) > 0 and len(lows) > 0:
                                historical_analysis['recent_high'] = float(highs.tail(60).max())
                                historical_analysis['recent_low'] = float(lows.tail(60).min())
                    except Exception as e:
                        print(f"   [WARNING] Error processing high/low: {e}")
                
                # Technical indicators
                if loader.technical_data is not None and not loader.technical_data.empty:
                    try:
                        tech_data = loader.technical_data
                        latest_tech = tech_data.iloc[-1]
                        
                        tech_indicators = ['SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_Signal', 'ATR']
                        for indicator in tech_indicators:
                            if indicator in latest_tech.index:
                                try:
                                    value = pd.to_numeric(latest_tech[indicator], errors='coerce')
                                    if pd.notna(value):
                                        key = f'current_{indicator.lower()}' if indicator == 'RSI' else indicator.lower()
                                        historical_analysis[key] = float(value)
                                except:
                                    pass
                    except Exception as e:
                        print(f"   [WARNING] Error processing technical data: {e}")
                
                # Summary data
                if loader.summary_data is not None and not loader.summary_data.empty:
                    try:
                        summary = loader.summary_data.iloc[0]
                        
                        # Text fields
                        text_fields = ['overall_signal', 'trend', 'momentum', 'recommendation', 'risk_level']
                        for field in text_fields:
                            if field in summary.index and pd.notna(summary[field]):
                                value = str(summary[field])
                                if not ('%' in value and len(value) > 10):  # Skip corrupted data
                                    historical_analysis[f'excel_{field}'] = value
                    except Exception as e:
                        print(f"   [WARNING] Error processing summary: {e}")
                
                # Get 30-day performance
                try:
                    thirty_day_performance = loader.get_30d_performance()
                    historical_analysis['performance_return_1_month'] = thirty_day_performance
                    historical_analysis['30d_return'] = thirty_day_performance
                    historical_analysis['recent_30d_return'] = thirty_day_performance
                    print(f"   [OK] 30-day performance = {thirty_day_performance:.2f}%")
                except Exception as e:
                    print(f"   [WARNING] Error getting 30-day performance: {e}")
                    historical_analysis['performance_return_1_month'] = 0.0
                
                # Company info
                if loader.company_info is not None and not loader.company_info.empty:
                    try:
                        company = loader.company_info.iloc[0]
                        if 'sector' in company.index and pd.notna(company['sector']):
                            historical_analysis['sector'] = str(company['sector'])
                        if 'industry' in company.index and pd.notna(company['industry']):
                            historical_analysis['industry'] = str(company['industry'])
                    except Exception as e:
                        print(f"   [WARNING] Error processing company info: {e}")
                
                # Set defaults for missing values
                defaults = {
                    'avg_daily_change': 0.0, 'sector': 'Technology',
                    'excel_recommendation': 'Hold', 'excel_risk_level': 'Moderate',
                    'recent_high': 180.0, 'recent_low': 150.0, 'sma_20': 165.0,
                    'sma_50': 160.0, 'sma_200': 155.0, 'current_rsi': 55.0
                }
                
                for key, default_value in defaults.items():
                    if key not in historical_analysis:
                        historical_analysis[key] = default_value
                
                print(f"[OK] Excel analysis complete: {len(historical_analysis)} metrics extracted")
                print(f"[CHART] FINAL VOLATILITY: {historical_analysis['volatility']:.2f}%")
                
                return historical_analysis
                
        except Exception as e:
            print(f"[ERROR] Excel loading error: {e}")
            return {
                'performance_return_1_month': 0.0, 'volatility': 25.0, 'sector': 'Technology',
                'avg_daily_change': 0.0, 'excel_recommendation': 'Hold', 'excel_risk_level': 'Moderate'
            }

    def analyze_article_sentiment(self, article_content, ticker):
        """Analyze article for specific momentum indicators"""
        positive_indicators = [
            'beat expectations', 'record revenue', 'strong growth', 'exceeded',
            'breakthrough', 'partnership', 'expansion', 'bullish', 'upgrade',
            'raised guidance', 'positive outlook', 'strong demand', 'all-time high',
            'surpassed', 'accelerating', 'robust', 'outperform', 'momentum'
        ]
        
        negative_indicators = [
            'missed expectations', 'revenue decline', 'weak growth', 'below',
            'concerns', 'layoffs', 'contraction', 'bearish', 'downgrade',
            'lowered guidance', 'negative outlook', 'weak demand', 'disappointing',
            'slowing', 'struggling', 'underperform', 'cut', 'warning'
        ]
        
        content_lower = article_content.lower()
        
        positive_count = sum(1 for phrase in positive_indicators if phrase in content_lower)
        negative_count = sum(1 for phrase in negative_indicators if phrase in content_lower)
        
        net_score = positive_count - negative_count
        momentum_score = max(-1, min(1, net_score / 5))
        
        return {
            'momentum_score': momentum_score,
            'positive_indicators': positive_count,
            'negative_indicators': negative_count,
            'momentum_direction': 'Bullish' if momentum_score > 0.2 else 'Bearish' if momentum_score < -0.2 else 'Neutral'
        }

    def detect_major_catalysts(self, article_content):
        """Detect if article contains major market-moving catalysts"""
        major_catalysts = {
            'earnings_beat': ['beat earnings', 'exceeded eps', 'earnings surprise', 'beat expectations', 'topped estimates'],
            'guidance_raise': ['raised guidance', 'increased outlook', 'upgraded forecast', 'raised full-year', 'boosted outlook'],
            'major_contract': ['billion dollar', 'major contract', 'landmark deal', 'mega deal', 'significant contract'],
            'fda_approval': ['fda approval', 'regulatory approval', 'cleared by fda', 'approved by fda'],
            'acquisition': ['acquiring', 'acquisition', 'merger', 'buyout', 'to acquire'],
            'breakthrough': ['breakthrough', 'revolutionary', 'game-changing', 'paradigm shift', 'disrupting']
        }
        
        content_lower = article_content.lower()
        detected_catalysts = []
        
        for catalyst_type, phrases in major_catalysts.items():
            for phrase in phrases:
                if phrase in content_lower:
                    detected_catalysts.append(catalyst_type)
                    break
        
        return {
            'has_major_catalyst': len(detected_catalysts) > 0,
            'catalyst_types': detected_catalysts,
            'catalyst_count': len(detected_catalysts),
            'impact_multiplier': 1.0 + (len(detected_catalysts) * 0.3)
        }

    def get_phase3_news_intelligence(self, ticker, graph_analysis=None):
        
        """Enhanced news intelligence with multi-source integration AND Graph Analysis"""
        print(f"📰 Gathering MULTI-SOURCE news intelligence with GRAPH ANALYSIS for {ticker}...")
        
        # Initialize multi-source analyzer
        api_keys = {
            'alpha_vantage': self.alpha_key,
            'finnhub': getattr(self, 'finnhub_key', None),
            'newsapi': getattr(self, 'newsapi_key', None)
        }
        
        multi_analyzer = MultiSourceNewsAnalyzer(api_keys)
        
        # Step 1: Get Phase 3 data (your existing system)
        phase3_data = None
        try:
            from phase3_connector import Phase3NewsPredictor
            predictor = Phase3NewsPredictor()
            
            # Get comprehensive prediction signal with fresh search
            prediction_signal = predictor.get_prediction_signal(ticker)
            
            phase3_data = {
                'sentiment_1d': float(prediction_signal.get('sentiment_score', 0.0)),
                'sentiment_7d': float(prediction_signal.get('sentiment_score', 0.0)),
                'news_volume_1d': int(prediction_signal.get('article_count', 0)),
                'news_volume_7d': int(prediction_signal.get('article_count', 0)),
                'confidence_score': float(prediction_signal.get('confidence', 0.5)),
                'source_diversity': 3 if prediction_signal.get('article_count', 0) > 10 else 2 if prediction_signal.get('article_count', 0) > 5 else 1,
                'event_impact_score': float(prediction_signal.get('signal_strength', 0.0)),
                'recent_events': prediction_signal.get('recent_events', []),
                'fresh_search_performed': True,
                'fresh_articles_found': prediction_signal.get('article_count', 0),
                'prediction_ready': prediction_signal.get('prediction_ready', False)
            }
            
            print(f"[OK] Phase 3 data: {phase3_data['fresh_articles_found']} articles, {phase3_data['sentiment_1d']:+.3f} sentiment")
            
        except Exception as e:
            print(f"[WARNING] Phase 3 system failed: {e}")
            phase3_data = {
                'sentiment_1d': 0.0, 'sentiment_7d': 0.0, 'news_volume_1d': 0,
                'confidence_score': 0.3, 'fresh_search_performed': False,
                'fresh_articles_found': 0, 'prediction_ready': False
            }
        
        # Step 2: Get multi-source aggregation
        try:
            multi_source_results = multi_analyzer.aggregate_multi_source_sentiment(ticker, phase3_data)
            
            # Step 3: Blend Phase 3 with multi-source results
            if multi_source_results['active_sources'] > 1:
                # Use multi-source aggregation
                enhanced_sentiment = multi_source_results['aggregated_sentiment']
                enhanced_confidence = multi_source_results['aggregate_confidence']
                
                print(f"   🔄 Multi-source blending applied:")
                print(f"   [CHART] Enhanced sentiment: {enhanced_sentiment:+.3f} (from {multi_source_results['active_sources']} sources)")
                print(f"   [CHART] Enhanced confidence: {enhanced_confidence:.3f}")
                
            else:
                # Fall back to Phase 3 only
                enhanced_sentiment = phase3_data['sentiment_1d']
                enhanced_confidence = phase3_data['confidence_score']
                print(f"   [WARNING] Limited sources available, using Phase 3 primary")
            
            # NEW STEP 3.5: GRAPH ANALYSIS INTEGRATION
            graph_enhanced_sentiment = enhanced_sentiment
            graph_confidence_boost = 0.0
            graph_momentum_signal = 0.0
            graph_technical_alignment = 'neutral'
            
            if graph_analysis:
                print(f"   [UP] Integrating Graph Analysis insights...")
                
                try:
                    # Extract graph analysis components
                    primary_pattern = graph_analysis.get('pattern_detected', {}).get('primary_pattern')
                    pattern_reliability = graph_analysis.get('pattern_detected', {}).get('pattern_reliability', 0.0)
                    candlestick_patterns = graph_analysis.get('candlestick_analysis', {}).get('strongest_pattern')
                    breakout_detected = graph_analysis.get('breakout_analysis', {}).get('breakout_detected', False)
                    breakout_strength = graph_analysis.get('breakout_analysis', {}).get('breakout_strength', 0.0)
                    momentum_acceleration = graph_analysis.get('momentum_analysis', {}).get('momentum_acceleration', 0.5)
                    
                    # Calculate graph-based sentiment adjustment
                    graph_sentiment_adjustment = 0.0
                    
                    # Pattern-based sentiment adjustment
                    bullish_patterns = ['ascending_triangle', 'cup_and_handle', 'bull_flag', 'falling_wedge', 'double_bottom']
                    bearish_patterns = ['descending_triangle', 'bear_flag', 'rising_wedge', 'head_and_shoulders', 'double_top']
                    
                    if primary_pattern in bullish_patterns:
                        pattern_boost = pattern_reliability * 0.3  # Up to +0.3 for strong bullish patterns
                        graph_sentiment_adjustment += pattern_boost
                        graph_technical_alignment = 'bullish'
                        print(f"   [UP] Bullish pattern boost: +{pattern_boost:.3f} from {primary_pattern}")
                    elif primary_pattern in bearish_patterns:
                        pattern_penalty = pattern_reliability * -0.3  # Up to -0.3 for strong bearish patterns
                        graph_sentiment_adjustment += pattern_penalty
                        graph_technical_alignment = 'bearish'
                        print(f"   [DOWN] Bearish pattern penalty: {pattern_penalty:.3f} from {primary_pattern}")
                    
                    # Candlestick pattern sentiment adjustment
                    if candlestick_patterns:
                        reversal_signals = graph_analysis.get('candlestick_analysis', {}).get('reversal_signals_count', 0)
                        continuation_signals = graph_analysis.get('candlestick_analysis', {}).get('continuation_signals_count', 0)
                        
                        # Get pattern clusters for stronger signal
                        pattern_clusters = graph_analysis.get('candlestick_analysis', {}).get('pattern_clusters', [])
                        if pattern_clusters:
                            strongest_cluster = max(pattern_clusters, key=lambda x: x.get('cluster_strength', 0))
                            cluster_signal = strongest_cluster.get('dominant_signal', 'neutral')
                            cluster_strength = strongest_cluster.get('cluster_strength', 0)
                            
                            if cluster_signal == 'bullish' and cluster_strength > 0.6:
                                candlestick_boost = cluster_strength * 0.2
                                graph_sentiment_adjustment += candlestick_boost
                                print(f"   🕯️ Bullish candlestick cluster: +{candlestick_boost:.3f}")
                            elif cluster_signal == 'bearish' and cluster_strength > 0.6:
                                candlestick_penalty = cluster_strength * -0.2
                                graph_sentiment_adjustment += candlestick_penalty
                                print(f"   🕯️ Bearish candlestick cluster: {candlestick_penalty:.3f}")
                    
                    # Breakout-based sentiment adjustment (strongest signal)
                    if breakout_detected and breakout_strength > 0.5:
                        breakout_direction = graph_analysis.get('breakout_analysis', {}).get('breakout_direction', 'neutral')
                        if breakout_direction == 'bullish':
                            breakout_boost = breakout_strength * 0.4  # Strong breakout can add up to +0.4
                            graph_sentiment_adjustment += breakout_boost
                            graph_technical_alignment = 'strong_bullish'
                            print(f"   [ROCKET] BREAKOUT boost: +{breakout_boost:.3f} (strength: {breakout_strength:.3f})")
                        elif breakout_direction == 'bearish':
                            breakout_penalty = breakout_strength * -0.4
                            graph_sentiment_adjustment += breakout_penalty
                            graph_technical_alignment = 'strong_bearish'
                            print(f"   [DOWN] BREAKDOWN penalty: {breakout_penalty:.3f} (strength: {breakout_strength:.3f})")
                    
                    # Momentum acceleration adjustment
                    if momentum_acceleration > 0.7:  # Strong positive momentum
                        momentum_boost = (momentum_acceleration - 0.5) * 0.3  # Scale from 0.5 baseline
                        graph_sentiment_adjustment += momentum_boost
                        graph_momentum_signal = momentum_boost
                        print(f"   ⚡ Momentum acceleration: +{momentum_boost:.3f}")
                    elif momentum_acceleration < 0.3:  # Strong negative momentum
                        momentum_penalty = (0.5 - momentum_acceleration) * -0.3
                        graph_sentiment_adjustment += momentum_penalty
                        graph_momentum_signal = momentum_penalty
                        print(f"   ⚡ Momentum deceleration: {momentum_penalty:.3f}")
                    
                    # Apply graph sentiment adjustment with dampening for extreme values
                    max_adjustment = 0.5  # Cap total adjustment at ±0.5
                    graph_sentiment_adjustment = max(-max_adjustment, min(max_adjustment, graph_sentiment_adjustment))
                    
                    # Blend with existing sentiment using weighted approach
                    if abs(graph_sentiment_adjustment) > 0.05:  # Only apply if meaningful
                        # Strong graph signals get more weight
                        graph_weight = min(0.4, abs(graph_sentiment_adjustment) * 0.8)
                        news_weight = 1.0 - graph_weight
                        
                        graph_enhanced_sentiment = (enhanced_sentiment * news_weight + 
                                                (enhanced_sentiment + graph_sentiment_adjustment) * graph_weight)
                        
                        print(f"   [CHART] Graph-enhanced sentiment: {enhanced_sentiment:+.3f} → {graph_enhanced_sentiment:+.3f}")
                        print(f"   [CHART] Graph weight: {graph_weight:.3f}, News weight: {news_weight:.3f}")
                    
                    # Calculate confidence boost from graph analysis
                    if pattern_reliability > 0.6:
                        graph_confidence_boost += 0.05  # Pattern confidence boost
                    if breakout_detected and breakout_strength > 0.6:
                        graph_confidence_boost += 0.08  # Breakout confidence boost
                    if len(graph_analysis.get('candlestick_analysis', {}).get('pattern_clusters', [])) > 0:
                        graph_confidence_boost += 0.03  # Candlestick cluster boost
                    
                    # Cap confidence boost
                    graph_confidence_boost = min(0.15, graph_confidence_boost)
                    enhanced_confidence = min(0.95, enhanced_confidence + graph_confidence_boost)
                    
                    print(f"   [TARGET] Graph confidence boost: +{graph_confidence_boost:.3f} → {enhanced_confidence:.3f}")
                    
                except Exception as graph_error:
                    print(f"   [WARNING] Graph analysis integration failed: {graph_error}")
                    # Continue with original sentiment if graph integration fails
                    pass
            
            # Step 4: Create enhanced news data structure WITH GRAPH INSIGHTS
            enhanced_news_data = {
                'sentiment_1d': graph_enhanced_sentiment,  # Now includes graph adjustments
                'sentiment_7d': phase3_data.get('sentiment_7d', graph_enhanced_sentiment),
                'news_volume_1d': multi_source_results['total_articles'],
                'news_volume_7d': multi_source_results['total_articles'],
                'confidence_score': enhanced_confidence,  # Now includes graph confidence boost
                'source_diversity': multi_source_results['active_sources'],
                'event_impact_score': phase3_data.get('event_impact_score', abs(graph_enhanced_sentiment)),
                'recent_events': phase3_data.get('recent_events', []),
                
                # Multi-source specific data
                'multi_source_enabled': True,
                'active_sources': multi_source_results['active_sources'],
                'total_articles': multi_source_results['total_articles'],
                'source_breakdown': multi_source_results['source_breakdown'],
                'aggregated_sentiment': enhanced_sentiment,  # Original before graph adjustment
                'aggregate_confidence': multi_source_results['aggregate_confidence'],
                
                # Phase 3 specific data
                'fresh_search_performed': phase3_data.get('fresh_search_performed', False),
                'fresh_articles_found': phase3_data.get('fresh_articles_found', 0),
                'prediction_ready': phase3_data.get('prediction_ready', False),
                'phase3_sentiment': phase3_data.get('sentiment_1d', 0.0),
                
                # Enhanced metrics (with graph integration)
                'signal_strength': abs(graph_enhanced_sentiment) * enhanced_confidence,
                'signal_direction': graph_enhanced_sentiment,
                'quality_score': enhanced_confidence,
                'news_price_correlation': phase3_data.get('news_price_correlation', 0.0),
                
                # NEW: Graph Analysis Integration Results
                'graph_analysis_integrated': graph_analysis is not None,
                'graph_sentiment_adjustment': graph_enhanced_sentiment - enhanced_sentiment if graph_analysis else 0.0,
                'graph_confidence_boost': graph_confidence_boost,
                'graph_momentum_signal': graph_momentum_signal,
                'graph_technical_alignment': graph_technical_alignment,
                'graph_insights': {
                    'primary_pattern': graph_analysis.get('pattern_detected', {}).get('primary_pattern') if graph_analysis else None,
                    'pattern_reliability': graph_analysis.get('pattern_detected', {}).get('pattern_reliability', 0.0) if graph_analysis else 0.0,
                    'breakout_detected': graph_analysis.get('breakout_analysis', {}).get('breakout_detected', False) if graph_analysis else False,
                    'breakout_strength': graph_analysis.get('breakout_analysis', {}).get('breakout_strength', 0.0) if graph_analysis else 0.0,
                    'momentum_acceleration': graph_analysis.get('momentum_analysis', {}).get('momentum_acceleration', 0.5) if graph_analysis else 0.5,
                    'candlestick_signals': graph_analysis.get('candlestick_analysis', {}).get('total_patterns_found', 0) if graph_analysis else 0,
                    'support_resistance_quality': graph_analysis.get('support_resistance', {}).get('level_quality', 0.0) if graph_analysis else 0.0
                }
            }
            
            print(f"[OK] Multi-source news intelligence with GRAPH ANALYSIS complete:")
            print(f"   [CHART] Sources: {multi_source_results['active_sources']}, Articles: {multi_source_results['total_articles']}")
            print(f"   [UP] Final sentiment: {graph_enhanced_sentiment:+.3f} (news: {enhanced_sentiment:+.3f})")
            print(f"   [TARGET] Final confidence: {enhanced_confidence:.3f}")
            if graph_analysis:
                print(f"   [UP] Graph integration: [OK] Applied")
                print(f"   [CHART] Technical alignment: {graph_technical_alignment}")
                print(f"   [CHART] Pattern: {graph_analysis.get('pattern_detected', {}).get('primary_pattern', 'None')}")
                if graph_analysis.get('breakout_analysis', {}).get('breakout_detected', False):
                    print(f"   [ROCKET] Breakout detected: {graph_analysis.get('breakout_analysis', {}).get('breakout_direction', 'unknown')}")
            
            return enhanced_news_data
            
        except Exception as e:
            print(f"[ERROR] Multi-source aggregation failed: {e}")
            
            # Ultimate fallback
            return {
                'sentiment_1d': phase3_data.get('sentiment_1d', 0.0),
                'sentiment_7d': phase3_data.get('sentiment_7d', 0.0),
                'news_volume_1d': phase3_data.get('news_volume_1d', 0),
                'news_volume_7d': phase3_data.get('news_volume_7d', 0),
                'confidence_score': phase3_data.get('confidence_score', 0.5),
                'source_diversity': 1,
                'event_impact_score': phase3_data.get('event_impact_score', 0.0),
                'recent_events': phase3_data.get('recent_events', []),
                'multi_source_enabled': False,
                'active_sources': 1,
                'fresh_search_performed': phase3_data.get('fresh_search_performed', False),
                'fresh_articles_found': phase3_data.get('fresh_articles_found', 0),
                'prediction_ready': phase3_data.get('prediction_ready', False),
                'graph_analysis_integrated': False,
                'graph_sentiment_adjustment': 0.0,
                'graph_confidence_boost': 0.0,
                'graph_technical_alignment': 'neutral'
            }
            
    def load_api_keys(self):
        """Load API keys from .env file - Enhanced for multi-source"""
        try:
            with open('.env', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('ALPHA_VANTAGE_API_KEY='):
                        self.alpha_key = line.split('=', 1)[1].strip()
                    elif line.startswith('CLAUDE_API_KEY='):
                        self.claude_key = line.split('=', 1)[1].strip()
                    elif line.startswith('FINNHUB_API_KEY='):  # NEW
                        self.finnhub_key = line.split('=', 1)[1].strip()
                    elif line.startswith('NEWSAPI_KEY='):  # NEW
                        self.newsapi_key = line.split('=', 1)[1].strip()
        except FileNotFoundError:
            print("[ERROR] .env file not found. Please create one with your API keys.")

    def enhance_news_with_custom_articles(self, news_data, custom_articles):
        """Enhanced news enhancement with multi-source integration and catalyst detection"""
        if not custom_articles:
            return news_data
        
        print(f"🔄 Enhancing MULTI-SOURCE news analysis with {len(custom_articles)} custom articles...")
        
        total_impact = 0
        has_major_catalyst = False
        catalyst_count = 0
        catalyst_types = []
        
        # Enhanced custom article analysis
        for article in custom_articles:
            momentum_analysis = article.get('momentum_analysis', {'momentum_score': article['sentiment_score']})
            catalyst_analysis = article.get('catalyst_analysis', {'has_major_catalyst': False, 'impact_multiplier': 1.0})
            
            # Enhanced sentiment combination with catalyst boost
            base_sentiment = (article['sentiment_score'] * 0.4 + 
                            momentum_analysis['momentum_score'] * 0.6)
            
            # Apply catalyst multiplier
            catalyst_multiplier = catalyst_analysis.get('impact_multiplier', 1.0)
            combined_sentiment = base_sentiment * catalyst_multiplier
            
            # Enhanced recency weighting
            hours_old = (datetime.now() - datetime.fromisoformat(article['timestamp'])).total_seconds() / 3600
            if hours_old < 6:
                recency_weight = 1.2  # Breaking news boost
            elif hours_old < 24:
                recency_weight = 1.0
            elif hours_old < 48:
                recency_weight = 0.8
            else:
                recency_weight = 0.6
            
            final_impact = combined_sentiment * recency_weight
            total_impact += final_impact
            
            # Track catalysts
            if catalyst_analysis.get('has_major_catalyst', False):
                has_major_catalyst = True
                catalyst_count += catalyst_analysis.get('catalyst_count', 1)
                catalyst_types.extend(catalyst_analysis.get('catalyst_types', []))
        
        avg_impact = total_impact / len(custom_articles) if custom_articles else 0
        
        # Enhanced blending logic for multi-source integration
        base_sentiment = news_data.get('sentiment_1d', 0)
        
        # Check if multi-source is enabled
        if news_data.get('multi_source_enabled', False):
            # Multi-source blending
            active_sources = news_data.get('active_sources', 1)
            total_articles = news_data.get('total_articles', 0)
            
            # Adaptive weighting based on data richness
            if total_articles > 20 and active_sources >= 3:
                # Rich multi-source data - be more conservative with custom articles
                multi_source_weight = 0.7
                custom_weight = 0.3
                print(f"   🔄 Rich multi-source blending: {active_sources} sources, {total_articles} articles")
            elif total_articles > 10 and active_sources >= 2:
                # Moderate multi-source data
                multi_source_weight = 0.6
                custom_weight = 0.4
                print(f"   🔄 Moderate multi-source blending: {active_sources} sources, {total_articles} articles")
            else:
                # Limited multi-source data - give more weight to custom articles
                multi_source_weight = 0.5
                custom_weight = 0.5
                print(f"   🔄 Limited multi-source blending: {active_sources} sources, {total_articles} articles")
            
            # Major catalyst override
            if has_major_catalyst:
                custom_weight = min(0.6, custom_weight + 0.2)  # Boost for major catalysts
                multi_source_weight = 1.0 - custom_weight
                print(f"   🚨 Major catalyst boost: custom weight increased to {custom_weight}")
            
            enhanced_sentiment = (base_sentiment * multi_source_weight + avg_impact * custom_weight)
            
        elif news_data.get('fresh_search_performed', False):
            # Fresh search blending (fallback when multi-source fails)
            fresh_weight = 0.6
            custom_weight = 0.4 if has_major_catalyst else 0.3
            enhanced_sentiment = (base_sentiment * fresh_weight + avg_impact * custom_weight)
            print(f"   🔄 Fresh search blending (multi-source unavailable)")
            
        else:
            # Original Phase 3 only blending
            custom_weight = 0.8 if has_major_catalyst else 0.7
            phase3_weight = 1.0 - custom_weight
            enhanced_sentiment = (base_sentiment * phase3_weight + avg_impact * custom_weight)
            print(f"   🔄 Phase 3 only blending (fallback mode)")
        
        # Create enhanced news data with multi-source awareness
        enhanced_news_data = news_data.copy()
        enhanced_news_data['sentiment_1d'] = enhanced_sentiment
        enhanced_news_data['custom_impact_score'] = avg_impact
        enhanced_news_data['momentum_direction'] = 'Bullish' if avg_impact > 0.2 else 'Bearish' if avg_impact < -0.2 else 'Neutral'
        enhanced_news_data['has_major_catalyst'] = has_major_catalyst
        enhanced_news_data['total_catalyst_count'] = catalyst_count
        enhanced_news_data['custom_articles_count'] = len(custom_articles)
        enhanced_news_data['custom_sentiment_impact'] = avg_impact
        
        # Enhanced catalyst tracking
        enhanced_news_data['catalyst_types'] = list(set(catalyst_types))
        enhanced_news_data['catalyst_impact_level'] = 'high' if catalyst_count > 2 else 'moderate' if catalyst_count > 0 else 'none'
        
        # Update article counts with multi-source awareness
        base_articles = enhanced_news_data.get('total_articles', enhanced_news_data.get('news_volume_1d', 0))
        enhanced_news_data['news_volume_1d'] = base_articles + len(custom_articles)
        enhanced_news_data['total_articles'] = base_articles + len(custom_articles)
        
        # Enhanced confidence calculation with multi-source consideration
        base_confidence = enhanced_news_data.get('confidence_score', 0.5)
        
        # Multi-source confidence boost
        if enhanced_news_data.get('multi_source_enabled', False):
            multi_source_boost = min(0.1, enhanced_news_data.get('active_sources', 1) / 40)
            article_boost = min(0.05, enhanced_news_data.get('total_articles', 0) / 100)
            custom_boost = min(0.05, len(custom_articles) / 20)
            
            enhanced_confidence = min(0.95, base_confidence + multi_source_boost + article_boost + custom_boost)
            print(f"   [TARGET] Multi-source confidence boost: +{multi_source_boost + article_boost + custom_boost:.3f}")
        else:
            # Standard confidence boost
            data_richness_bonus = min(0.1, enhanced_news_data.get('fresh_articles_found', 0) / 50)
            enhanced_confidence = min(0.95, base_confidence + data_richness_bonus)
        
        enhanced_news_data['confidence_score'] = enhanced_confidence
        
        # Enhanced recent events with multi-source context
        if 'recent_events' not in enhanced_news_data:
            enhanced_news_data['recent_events'] = []
        
        # Add custom articles to recent events with enhanced tagging
        for article in custom_articles[-3:]:
            catalyst_info = ""
            if article.get('catalyst_analysis', {}).get('has_major_catalyst', False):
                catalyst_types_str = ", ".join(article.get('catalyst_analysis', {}).get('catalyst_types', []))
                catalyst_info = f" [CATALYST: {catalyst_types_str}]"
            
            enhanced_news_data['recent_events'].append({
                'title': f"[CUSTOM]{catalyst_info} {article['title']}",
                'sentiment': article['sentiment_score'],
                'confidence': article['confidence'],
                'date': article['timestamp'][:10],
                'has_catalyst': article.get('catalyst_analysis', {}).get('has_major_catalyst', False),
                'catalyst_types': article.get('catalyst_analysis', {}).get('catalyst_types', []),
                'source': 'custom_user_input'
            })
        
        # Enhanced signal strength calculation
        if enhanced_news_data.get('multi_source_enabled', False):
            # Multi-source signal strength
            source_diversity_factor = min(1.5, enhanced_news_data.get('active_sources', 1) / 3)
            article_volume_factor = min(1.3, enhanced_news_data.get('total_articles', 0) / 30)
            catalyst_factor = 1.5 if has_major_catalyst else 1.0
            
            enhanced_signal_strength = abs(enhanced_sentiment) * enhanced_confidence * source_diversity_factor * article_volume_factor * catalyst_factor
            enhanced_news_data['signal_strength'] = min(1.0, enhanced_signal_strength)
            enhanced_news_data['signal_direction'] = enhanced_sentiment
            enhanced_news_data['quality_score'] = enhanced_confidence * source_diversity_factor
            
            print(f"   [TARGET] Multi-source signal strength: {enhanced_signal_strength:.3f}")
        
        # Final logging with multi-source details
        print(f"   [OK] Multi-source news enhancement complete:")
        print(f"   [CHART] Enhanced sentiment: {enhanced_sentiment:+.3f} (base: {base_sentiment:+.3f})")
        print(f"   [CHART] Custom impact: {avg_impact:+.2f}")
        print(f"   📰 Total articles: {enhanced_news_data.get('total_articles', 0)}")
        if enhanced_news_data.get('multi_source_enabled', False):
            print(f"   🔄 Multi-source: {enhanced_news_data.get('active_sources', 0)} sources active")
        print(f"   [TARGET] Final confidence: {enhanced_confidence:.3f}")
        if has_major_catalyst:
            print(f"   🚨 Major catalysts: {catalyst_count} detected ({', '.join(set(catalyst_types))})")
        
        return enhanced_news_data

    def get_realtime_market_data(self, ticker, api_key):
        """Get real-time data from Alpha Vantage API"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {'function': 'GLOBAL_QUOTE', 'symbol': ticker, 'apikey': api_key}
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if "Global Quote" in data:
                quote = data["Global Quote"]
                return {
                    'current_price': float(quote['05. price']),
                    'open': float(quote['02. open']),
                    'high': float(quote['03. high']),
                    'low': float(quote['04. low']),
                    'volume': int(quote['06. volume']),
                    'change': float(quote['09. change']),
                    'change_percent': float(quote['10. change percent'].replace('%', '')),
                    'previous_close': float(quote['08. previous close']),
                    'current_rsi': 50.0,
                    'av_news_sentiment': 0.0,
                    'av_news_count': 0,
                    'data_source': 'alpha_vantage'
                }
            else:
                print(f"   [WARNING] API issue: {data.get('Note', data.get('Error Message', 'Unknown error'))}")
                return None
        except Exception as e:
            print(f"   [ERROR] API failed: {e}")
            return None

    def generate_multi_day_price_paths(self, current_price, volatility, prediction_days, num_paths=None, drift=None, overall_signal=0):
        """Enhanced price path generation with VOLATILITY-AWARE simulation counts and improved accuracy"""
        print(f"🎲 Generating VOLATILITY-AWARE price paths for {prediction_days} days...")
        
        # DYNAMIC SIMULATION COUNT based on volatility
        if num_paths is None:
            if volatility < 20:
                num_paths = 1500
                vol_tier = "Low"
            elif volatility < 35:
                num_paths = 2500
                vol_tier = "Medium"
            elif volatility < 50:
                num_paths = 4000
                vol_tier = "High"
            else:
                num_paths = 6000
                vol_tier = "Extreme"
            
            print(f"   [CHART] Volatility {volatility:.1f}% → {vol_tier} tier → {num_paths:,} simulations")
        
        # Use signal to adjust drift more intelligently
        max_annual_vol = 100.0
        realistic_annual_vol = min(volatility, max_annual_vol)
        annual_vol_decimal = realistic_annual_vol / 100
        daily_vol = annual_vol_decimal / math.sqrt(252)
        
        # ENHANCED: Momentum-aware drift calculation
        if drift is None:
            daily_drift = 0.0
        else:
            annual_drift_decimal = drift / 100
            base_daily_drift = annual_drift_decimal / 252
            
            # NEW: Enhanced signal-aware drift adjustment with stronger momentum capture
            if prediction_days <= 5:
                # For short-term, apply stronger momentum-based adjustments
                if overall_signal > 0.15:  # Strong bullish - AMPLIFY
                    signal_drift_adjustment = 0.0008  # Increased from 0.0003
                    drift_multiplier = 1.2  # AMPLIFY bullish drift instead of reducing
                    print(f"   [ROCKET] Strong bullish amplification: +{signal_drift_adjustment:.6f} drift, {drift_multiplier:.1f}x multiplier")
                elif overall_signal > 0.05:  # Moderate bullish
                    signal_drift_adjustment = 0.0004
                    drift_multiplier = 0.9  # Slightly reduce
                elif overall_signal < -0.15:  # Strong bearish - AMPLIFY
                    signal_drift_adjustment = -0.0008  # Increased from -0.0005
                    drift_multiplier = 1.1  # AMPLIFY bearish drift
                    print(f"   [DOWN] Strong bearish amplification: {signal_drift_adjustment:.6f} drift, {drift_multiplier:.1f}x multiplier")
                elif overall_signal < -0.05:  # Moderate bearish
                    signal_drift_adjustment = -0.0004
                    drift_multiplier = 0.6  # Reduce bullish drift more
                else:  # Neutral/weak signals
                    signal_drift_adjustment = 0
                    drift_multiplier = 0.7  # Slightly reduce for weak signals
            else:
                # For longer-term, use enhanced but gentler adjustments
                if overall_signal > 0.2:  # Strong bullish long-term
                    signal_drift_adjustment = overall_signal * 0.0003  # Increased from 0.0001
                    drift_multiplier = 1.1
                elif overall_signal < -0.2:  # Strong bearish long-term
                    signal_drift_adjustment = overall_signal * 0.0003
                    drift_multiplier = 1.05
                else:
                    signal_drift_adjustment = overall_signal * 0.0002  # Increased from 0.0001
                    drift_multiplier = 0.9
            
            daily_drift = (base_daily_drift * drift_multiplier) + signal_drift_adjustment
        
        print(f"   [CHART] Enhanced daily drift: {daily_drift:.6f} (signal: {overall_signal:+.3f})")
        
        # VOLATILITY-AWARE daily movement limits
        if volatility < 20:
            # Low volatility stocks - tighter limits
            base_limit_short = min(0.06, realistic_annual_vol / 100 / 10)  # More conservative
            base_limit_long = min(0.08, realistic_annual_vol / 100 / 8)
            vol_adjustment = 1.0
        elif volatility < 35:
            # Medium volatility stocks - standard limits  
            base_limit_short = min(0.08, realistic_annual_vol / 100 / 8)
            base_limit_long = min(0.12, realistic_annual_vol / 100 / 6)
            vol_adjustment = 1.1
        elif volatility < 50:
            # High volatility stocks - wider limits
            base_limit_short = min(0.10, realistic_annual_vol / 100 / 7)  # More generous
            base_limit_long = min(0.15, realistic_annual_vol / 100 / 5)
            vol_adjustment = 1.2
        else:
            # Extreme volatility stocks - very wide limits
            base_limit_short = min(0.12, realistic_annual_vol / 100 / 6)  # Much more generous
            base_limit_long = min(0.18, realistic_annual_vol / 100 / 4)
            vol_adjustment = 1.3
        
        print(f"   [CHART] Volatility tier limits: Short={base_limit_short:.3f}, Long={base_limit_long:.3f}, Adj={vol_adjustment:.1f}x")
        
        # Generate paths with enhanced mean reversion and momentum awareness
        price_paths = []
        convergence_check_interval = max(500, num_paths // 4)  # Check convergence periodically
        
        for path_num in range(num_paths):
            path = [current_price]
            price = current_price
            
            for day in range(prediction_days):
                # Enhanced mean reversion with momentum consideration
                distance_from_start_pct = (price - current_price) / current_price
                
                # NEW: Volatility-aware mean reversion strength
                if volatility < 20:
                    base_mean_reversion = 0.18  # Stronger for low-vol stocks
                elif volatility < 35:
                    base_mean_reversion = 0.15  # Standard
                elif volatility < 50:
                    base_mean_reversion = 0.12  # Weaker for high-vol stocks
                else:
                    base_mean_reversion = 0.08  # Much weaker for extreme-vol stocks
                
                # Momentum-aware mean reversion adjustment
                if prediction_days <= 5:
                    if overall_signal > 0.1:  # Bullish - reduce mean reversion
                        mean_reversion_strength = base_mean_reversion * 0.6
                    elif overall_signal < -0.1:  # Bearish - increase mean reversion
                        mean_reversion_strength = base_mean_reversion * 1.3
                    else:
                        mean_reversion_strength = base_mean_reversion
                else:
                    mean_reversion_strength = base_mean_reversion * 0.8
                
                # Apply progressive mean reversion
                if abs(distance_from_start_pct) > 0.15:  # Far from start
                    mean_reversion = -mean_reversion_strength * 1.2 * distance_from_start_pct
                elif abs(distance_from_start_pct) > 0.10:
                    mean_reversion = -mean_reversion_strength * distance_from_start_pct
                elif abs(distance_from_start_pct) > 0.05:
                    mean_reversion = -mean_reversion_strength * 0.5 * distance_from_start_pct
                else:
                    mean_reversion = 0
                
                # Enhanced signal-influenced momentum component
                if prediction_days <= 5:
                    signal_momentum = overall_signal * 0.0012 * vol_adjustment  # Scale with volatility
                else:
                    signal_momentum = overall_signal * 0.0008 * vol_adjustment
                
                # NEW: Add momentum persistence (trending behavior)
                if day > 0:
                    previous_change = (price - path[-2]) / path[-2] if len(path) > 1 else 0
                    # Add small persistence for strong signals, scaled by volatility
                    if abs(overall_signal) > 0.15:
                        momentum_persistence = previous_change * 0.15 * (overall_signal / abs(overall_signal)) * (vol_adjustment * 0.5)
                    else:
                        momentum_persistence = 0
                else:
                    momentum_persistence = 0
                
                random_shock = np.random.normal(0, daily_vol)
                total_daily_change = daily_drift + mean_reversion + signal_momentum + momentum_persistence + random_shock
                
                # VOLATILITY-AWARE dynamic daily limits with momentum consideration
                if prediction_days <= 5:
                    max_daily_move = base_limit_short
                    # Allow larger moves for strong signals, scaled by volatility tier
                    if abs(overall_signal) > 0.2:
                        max_daily_move *= (1.3 * vol_adjustment)  # Volatility-scaled amplification
                    else:
                        max_daily_move *= vol_adjustment
                else:
                    max_daily_move = base_limit_long
                    if abs(overall_signal) > 0.2:
                        max_daily_move *= (1.1 * vol_adjustment)
                    else:
                        max_daily_move *= vol_adjustment
                
                total_daily_change = max(-max_daily_move, min(max_daily_move, total_daily_change))
                
                new_price = price * (1 + total_daily_change)
                new_price = max(0.01, new_price)
                
                price = new_price
                path.append(price)
            
            price_paths.append(path)
            
            # CONVERGENCE CHECK for high simulation counts
            if num_paths >= 3000 and (path_num + 1) % convergence_check_interval == 0:
                if path_num >= 1500:  # Only check after minimum paths
                    temp_paths = np.array(price_paths)
                    temp_final_prices = temp_paths[:, -1]
                    temp_mean = np.mean(temp_final_prices)
                    temp_std = np.std(temp_final_prices)
                    
                    # Check if we have enough paths for this volatility level
                    std_error = temp_std / math.sqrt(path_num + 1)
                    convergence_ratio = std_error / temp_mean
                    
                    # Volatility-based convergence thresholds
                    if volatility < 20:
                        convergence_threshold = 0.005  # 0.5%
                    elif volatility < 35:
                        convergence_threshold = 0.008  # 0.8%
                    elif volatility < 50:
                        convergence_threshold = 0.012  # 1.2%
                    else:
                        convergence_threshold = 0.015  # 1.5%
                    
                    if convergence_ratio < convergence_threshold:
                        print(f"   [OK] Early convergence achieved at {path_num + 1:,} paths (threshold: {convergence_threshold:.1%})")
                        break
        
        # Convert to numpy array
        price_paths = np.array(price_paths)
        
        # Log path generation summary with enhanced metrics
        final_prices = price_paths[:, -1]
        avg_final_price = np.mean(final_prices)
        expected_return = (avg_final_price - current_price) / current_price * 100
        std_final_prices = np.std(final_prices)
        
        # Calculate quality metrics
        std_error = std_final_prices / math.sqrt(len(price_paths))
        confidence_interval_width = (std_error / avg_final_price) * 100 * 1.96  # 95% CI width as %
        
        print(f"   [OK] VOLATILITY-AWARE paths generated: {len(price_paths):,} paths")
        print(f"   [CHART] Expected return: {expected_return:+.2f}%")
        print(f"   [CHART] Path stats: Mean=${avg_final_price:.2f}, Range=${np.min(final_prices):.2f}-${np.max(final_prices):.2f}")
        print(f"   [TARGET] Precision: ±{confidence_interval_width:.2f}% (95% CI width)")
        print(f"   [UP] Standard error: {std_error:.2f} ({std_error/avg_final_price*100:.3f}%)")
        
        return price_paths
    
    def calculate_daily_stats_from_paths(self, price_paths, current_price, prediction_days, volatility):
        """Calculate daily statistics from price paths"""
        print("[CHART] Calculating daily statistics from price paths...")
        
        realistic_annual_vol = min(volatility, 80.0)
        daily_stats = []
        
        for day in range(1, prediction_days + 1):
            day_prices = price_paths[:, day]
            mean_price = np.mean(day_prices)
            
            # MAE TUNING: Stricter target MAE < 2.5% with volatility-based adjustment
            if realistic_annual_vol > 70:
                mae_target = 0.035  # 3.5%
            elif realistic_annual_vol > 50:
                mae_target = 0.030  # 3.0%
            elif realistic_annual_vol > 30:
                mae_target = 0.025  # 2.5%
            else:
                mae_target = 0.020  # 2.0%
                
            mae_quality_threshold = mae_target
            
            # Time-based expansion (but more controlled)
            time_expansion = 1.0 + (day - 1) * 0.05  # 5% per day
            adjusted_mae_target = mae_target * time_expansion
            
            # Calculate MAE-controlled range
            mae_range = mean_price * adjusted_mae_target
            
            # Confidence adjustment (tighter control)
            confidence_factor = 0.85
            confidence_adjustment = 1.0 + (1.0 - confidence_factor) * 0.2
            
            final_range = mae_range * confidence_adjustment
            
            # Calculate realistic high/low
            realistic_low = mean_price - final_range
            realistic_high = mean_price + final_range
            
            # Statistical bounds from Monte Carlo (for validation)
            percentile_15 = np.percentile(day_prices, 15)
            percentile_85 = np.percentile(day_prices, 85)
            
            # Use the most restrictive (conservative) bounds
            final_low = min(realistic_low, percentile_15)  # Taking the LOWER of the two
            final_high = max(realistic_high, percentile_85)  # Taking the HIGHER of the two
            
            # Validate MAE target achievement
            actual_mae_pct = (final_high - final_low) / (2 * mean_price)
            
            
            daily_stats.append({
                'day': day,
                'mean_price': mean_price,
                'median_price': np.median(day_prices),
                'std_price': np.std(day_prices),
                'min_price': final_low,
                'max_price': final_high,
                'percentile_5': np.percentile(day_prices, 5),
                'percentile_25': np.percentile(day_prices, 25),
                'percentile_75': np.percentile(day_prices, 75),
                'percentile_95': np.percentile(day_prices, 95),
                'prob_above_current': np.mean(day_prices > current_price),
                'expected_return': (mean_price - current_price) / current_price,
                # MAE quality metrics
                'mae_percent': actual_mae_pct * 100,
                'mae_target': mae_quality_threshold * 100,
                'mae_quality': 'Excellent' if actual_mae_pct <= 0.015 else 'Good' if actual_mae_pct <= 0.02 else 'Acceptable' if actual_mae_pct <= 0.025 else 'Wide',
                'range_width_pct': ((final_high - final_low) / mean_price) * 100,
                'prediction_confidence': confidence_factor,
                'volatility_adjusted_target': mae_quality_threshold * 100
            })
        
        print(f"   [OK] Daily statistics calculated for {len(daily_stats)} days")
        return daily_stats

    # IMPROVED PREDICTION ENGINE WITH BETTER SIGNAL HANDLING

    def perform_mathematical_analysis(self, excel_data, market_data, custom_articles, prediction_days, price_paths, graph_analysis):
        """IMPROVED mathematical analysis with MAE-ENFORCED range calculation, VOLATILITY-AWARE confidence integration, AND GRAPH ANALYSIS"""
        print("🔢 Performing IMPROVED mathematical analysis with MAE ENFORCEMENT and GRAPH ANALYSIS...")
        
        # ADD THIS NEW CALL AT THE BEGINNING:
        calibrated_signal, calibration_results = self.enhanced_directional_bias_calibration(
            excel_data, market_data, custom_articles
        )
        
        # Initialize analysis results
        analysis_results = {}
        analysis_results['bias_calibration'] = calibration_results  # Store the calibration results
        
        # Extract key data
        volatility = excel_data.get('volatility', 25.0)
        current_price = market_data['current_price']
        ticker = market_data.get('symbol', 'UNKNOWN')  # NEW: Get ticker for MAE tracking
        thirty_day_return = excel_data.get('performance_return_1_month', 0.0)
        
        # IMPROVEMENT 1: Better momentum analysis with time decay
        annualized_return = ((1 + thirty_day_return/100) ** 12 - 1) * 100
        momentum_score = thirty_day_return / 100 / 12  # Monthly to monthly decimal
        
        # Apply time decay for momentum - older returns have less predictive power
        momentum_decay_factor = 0.8 if prediction_days <= 5 else 0.9 if prediction_days <= 10 else 1.0
        adjusted_momentum = momentum_score * momentum_decay_factor
        
        scaled_momentum = max(-0.1, min(0.1, adjusted_momentum))  # Cap at ±10%
        
        # NEW: GRAPH ANALYSIS MOMENTUM ENHANCEMENT
        graph_momentum_boost = 0.0
        graph_momentum_acceleration = 0.5  # Default neutral
        
        if graph_analysis:
            print(f"   [UP] Integrating Graph Analysis momentum insights...")
            
            # Extract momentum acceleration from graph analysis
            graph_momentum_acceleration = graph_analysis.get('momentum_analysis', {}).get('momentum_acceleration', 0.5)
            
            # Strong momentum acceleration override
            if graph_momentum_acceleration > 0.75:  # Very strong momentum
                graph_momentum_boost = (graph_momentum_acceleration - 0.5) * 0.08  # Up to +0.02 boost
                print(f"   [ROCKET] Strong graph momentum boost: +{graph_momentum_boost:.4f} from acceleration {graph_momentum_acceleration:.3f}")
            elif graph_momentum_acceleration < 0.25:  # Very weak/negative momentum
                graph_momentum_boost = (graph_momentum_acceleration - 0.5) * 0.08  # Up to -0.02 penalty
                print(f"   [DOWN] Graph momentum penalty: {graph_momentum_boost:.4f} from acceleration {graph_momentum_acceleration:.3f}")
            
            # Apply to scaled momentum
            scaled_momentum = max(-0.12, min(0.12, scaled_momentum + graph_momentum_boost))
        
        # Store momentum metrics
        analysis_results['momentum_metrics'] = {
            'thirty_day_return': thirty_day_return,
            'annualized_return': annualized_return,
            'momentum_score': momentum_score,
            'adjusted_momentum': adjusted_momentum,
            'momentum_decay_factor': momentum_decay_factor,
            'scaled_momentum': scaled_momentum,
            'momentum_direction': 'Bullish' if adjusted_momentum > 0.001 else 'Bearish' if adjusted_momentum < -0.001 else 'Neutral',
            # NEW: Graph momentum integration
            'graph_momentum_boost': graph_momentum_boost,
            'graph_momentum_acceleration': graph_momentum_acceleration,
            'graph_momentum_applied': graph_analysis is not None
        }
        
        # IMPROVEMENT 2: Enhanced technical signals with more weight for short-term + GRAPH ANALYSIS
        daily_volatility = volatility / math.sqrt(252)
        rsi = excel_data.get('current_rsi', 50.0)
        sma_20 = excel_data.get('sma_20', current_price)
        sma_50 = excel_data.get('sma_50', current_price)
        
        # More sophisticated technical signals
        rsi_signal = 0.5 if 30 <= rsi <= 70 else (0.8 if rsi < 30 else 0.2)
        trend_signal = 0.7 if current_price > sma_20 > sma_50 else (0.3 if current_price < sma_20 < sma_50 else 0.5)
        
        # Short-term trend acceleration (price vs SMA_20)
        short_term_trend = (current_price - sma_20) / sma_20 if sma_20 > 0 else 0
        trend_acceleration = 0.6 if short_term_trend > 0.02 else 0.4 if short_term_trend < -0.02 else 0.5
        
        # NEW: GRAPH ANALYSIS TECHNICAL ENHANCEMENT
        graph_technical_boost = 0.0
        graph_pattern_signal = 0.5  # Default neutral
        graph_breakout_signal = 0.0
        
        if graph_analysis:
            print(f"   [CHART] Integrating Graph Analysis technical insights...")
            
            try:
                # 1. Chart Pattern Analysis
                primary_pattern = graph_analysis.get('pattern_detected', {}).get('primary_pattern')
                pattern_reliability = graph_analysis.get('pattern_detected', {}).get('pattern_reliability', 0.0)
                
                bullish_patterns = ['ascending_triangle', 'cup_and_handle', 'bull_flag', 'falling_wedge', 'double_bottom']
                bearish_patterns = ['descending_triangle', 'bear_flag', 'rising_wedge', 'head_and_shoulders', 'double_top']
                
                if primary_pattern in bullish_patterns and pattern_reliability > 0.5:
                    pattern_boost = pattern_reliability * 0.3  # Up to +0.3 for very reliable patterns
                    graph_pattern_signal = 0.5 + pattern_boost
                    print(f"   [UP] Bullish pattern boost: +{pattern_boost:.3f} from {primary_pattern} (reliability: {pattern_reliability:.3f})")
                elif primary_pattern in bearish_patterns and pattern_reliability > 0.5:
                    pattern_penalty = pattern_reliability * 0.3
                    graph_pattern_signal = 0.5 - pattern_penalty
                    print(f"   [DOWN] Bearish pattern penalty: -{pattern_penalty:.3f} from {primary_pattern} (reliability: {pattern_reliability:.3f})")
                
                # 2. Breakout Analysis (STRONG SIGNAL)
                breakout_detected = graph_analysis.get('breakout_analysis', {}).get('breakout_detected', False)
                breakout_strength = graph_analysis.get('breakout_analysis', {}).get('breakout_strength', 0.0)
                breakout_direction = graph_analysis.get('breakout_analysis', {}).get('breakout_direction', 'neutral')
                
                if breakout_detected and breakout_strength > 0.6:
                    if breakout_direction == 'bullish':
                        graph_breakout_signal = breakout_strength * 0.4  # Very strong signal
                        print(f"   [ROCKET] BREAKOUT detected: +{graph_breakout_signal:.3f} bullish signal (strength: {breakout_strength:.3f})")
                    elif breakout_direction == 'bearish':
                        graph_breakout_signal = -breakout_strength * 0.4
                        print(f"   [DOWN] BREAKDOWN detected: {graph_breakout_signal:.3f} bearish signal (strength: {breakout_strength:.3f})")
                
                # 3. Support/Resistance Level Quality
                level_quality = graph_analysis.get('support_resistance', {}).get('level_quality', 0.0)
                nearest_support = graph_analysis.get('support_resistance', {}).get('nearest_support', 0.0)
                nearest_resistance = graph_analysis.get('support_resistance', {}).get('nearest_resistance', 0.0)
                
                # Adjust technical signals based on proximity to key levels
                if nearest_support and nearest_resistance and level_quality > 0.6:
                    support_distance = (current_price - nearest_support) / current_price
                    resistance_distance = (nearest_resistance - current_price) / current_price
                    
                    if support_distance < 0.02:  # Very close to support (bullish)
                        level_boost = level_quality * 0.1
                        graph_technical_boost += level_boost
                        print(f"   [CHART] Near support level boost: +{level_boost:.3f}")
                    elif resistance_distance < 0.02:  # Very close to resistance (bearish)
                        level_penalty = level_quality * 0.1
                        graph_technical_boost -= level_penalty
                        print(f"   [CHART] Near resistance level penalty: -{level_penalty:.3f}")
                
                # 4. Candlestick Pattern Clusters
                pattern_clusters = graph_analysis.get('candlestick_analysis', {}).get('pattern_clusters', [])
                if pattern_clusters:
                    strongest_cluster = max(pattern_clusters, key=lambda x: x.get('cluster_strength', 0))
                    cluster_signal = strongest_cluster.get('dominant_signal', 'neutral')
                    cluster_strength = strongest_cluster.get('cluster_strength', 0)
                    
                    if cluster_strength > 0.7:
                        if cluster_signal == 'bullish':
                            candlestick_boost = cluster_strength * 0.15
                            graph_technical_boost += candlestick_boost
                            print(f"   🕯️ Strong bullish candlestick cluster: +{candlestick_boost:.3f}")
                        elif cluster_signal == 'bearish':
                            candlestick_penalty = cluster_strength * 0.15
                            graph_technical_boost -= candlestick_penalty
                            print(f"   🕯️ Strong bearish candlestick cluster: -{candlestick_penalty:.3f}")
                
            except Exception as graph_tech_error:
                print(f"   [WARNING] Graph technical analysis integration failed: {graph_tech_error}")
                graph_technical_boost = 0.0
        
        analysis_results['technical_signals'] = {
            'rsi_value': rsi,
            'rsi_signal': rsi_signal,
            'trend_signal': trend_signal,
            'trend_acceleration': trend_acceleration,
            'short_term_trend': short_term_trend,
            'price_vs_sma20': short_term_trend * 100,
            # NEW: Graph technical integration
            'graph_technical_boost': graph_technical_boost,
            'graph_pattern_signal': graph_pattern_signal,
            'graph_breakout_signal': graph_breakout_signal,
            'graph_technical_applied': graph_analysis is not None
        }
        
        # IMPROVEMENT 3: News impact with catalyst detection
        news_sentiment = 0
        catalyst_boost = 0
        total_news_articles = 0  # NEW: Track total news articles for confidence
        
        if custom_articles:
            total_sentiment = sum(article.get('sentiment_score', 0) for article in custom_articles)
            news_sentiment = total_sentiment / len(custom_articles)
            total_news_articles = len(custom_articles)  # NEW: Count custom articles
            
            # Check for major catalysts that could override technical signals
            for article in custom_articles:
                if article.get('catalyst_analysis', {}).get('has_major_catalyst', False):
                    catalyst_boost += 0.1 * article.get('sentiment_score', 0)
        
        # NEW: Add news data from stored context for total count
        if hasattr(self, '_last_news_data') and self._last_news_data:
            phase3_articles = self._last_news_data.get('total_articles', 0)
            fresh_articles = self._last_news_data.get('fresh_articles_found', 0)
            total_news_articles += max(phase3_articles, fresh_articles)  # Don't double count
            
            # NEW: Extract graph-enhanced news sentiment if available
            if self._last_news_data.get('graph_analysis_integrated', False):
                graph_sentiment_adjustment = self._last_news_data.get('graph_sentiment_adjustment', 0.0)
                if abs(graph_sentiment_adjustment) > 0.02:
                    print(f"   📰 Graph-enhanced news sentiment detected: {graph_sentiment_adjustment:+.3f}")
                    # This adjustment is already in the news sentiment, so just log it
        
        news_impact_score = (news_sentiment * 0.3) + catalyst_boost
        
        analysis_results['news_analysis'] = {
            'average_sentiment': news_sentiment,
            'catalyst_boost': catalyst_boost,
            'impact_score': news_impact_score,
            'has_catalysts': catalyst_boost > 0,
            'total_news_articles': total_news_articles,  # NEW: For confidence calculation
        }
        
        # NEW: Dynamic weight adjustment based on market regime (ENHANCED WITH GRAPH ANALYSIS)
        regime_detector = MarketRegimeDetector()
        
        # Prepare news data for regime detection
        news_data_for_regime = {
            'total_articles': len(custom_articles) if custom_articles else 0,
            'news_volume_1d': len(custom_articles) if custom_articles else 0,
            'multi_source_enabled': False,
            'active_sources': 1
        }
        
        # Try to extract news data from stored context if available
        if hasattr(self, '_last_news_data') and self._last_news_data:
            news_data_for_regime = self._last_news_data
        
        # NEW: Add graph analysis data to regime detection
        if graph_analysis:
            news_data_for_regime['graph_analysis_available'] = True
            news_data_for_regime['primary_pattern'] = graph_analysis.get('pattern_detected', {}).get('primary_pattern')
            news_data_for_regime['breakout_detected'] = graph_analysis.get('breakout_analysis', {}).get('breakout_detected', False)
            news_data_for_regime['momentum_acceleration'] = graph_analysis.get('momentum_analysis', {}).get('momentum_acceleration', 0.5)
        
        # Detect market regime
        regime_data = regime_detector.detect_market_regime(excel_data, market_data, news_data_for_regime, graph_analysis)
        
        # Get adaptive weights instead of fixed weights
        adaptive_weights = regime_detector.get_adaptive_weights(regime_data, prediction_days)
        
        # NEW: GRAPH ANALYSIS WEIGHT ADJUSTMENTS
        if graph_analysis:
            print(f"   [TARGET] Applying graph analysis weight adjustments...")
            
            # Extract integration signals from graph analysis
            integration_signals = graph_analysis.get('integration_signals', {})
            momentum_boost_factor = integration_signals.get('momentum_boost_factor', 1.0)
            technical_boost_factor = integration_signals.get('technical_boost_factor', 1.0)
            confidence_boost_factor = integration_signals.get('confidence_boost_factor', 0.0)
            regime_override_signal = integration_signals.get('regime_override_signal')
            
            # Apply boost factors to weights
            original_momentum_weight = adaptive_weights['momentum']
            original_technical_weight = adaptive_weights['technical']
            
            adaptive_weights['momentum'] *= momentum_boost_factor
            adaptive_weights['technical'] *= technical_boost_factor
            
            # REGIME OVERRIDE LOGIC
            if regime_override_signal:
                print(f"   🚨 REGIME OVERRIDE DETECTED: {regime_override_signal}")
                
                if 'breakout_confirmed' in regime_override_signal:
                    # Strong breakout - heavily favor technical analysis
                    adaptive_weights['technical'] = min(0.6, adaptive_weights['technical'] * 1.5)
                    adaptive_weights['momentum'] = min(0.3, adaptive_weights['momentum'] * 1.2)
                    adaptive_weights['news'] *= 0.7
                    print(f"   [ROCKET] Breakout regime override applied")
                
                elif 'momentum_acceleration' in regime_override_signal:
                    # Strong momentum - heavily favor momentum component
                    adaptive_weights['momentum'] = min(0.6, adaptive_weights['momentum'] * 1.4)
                    adaptive_weights['technical'] *= 0.9
                    adaptive_weights['news'] *= 0.8
                    print(f"   ⚡ Momentum acceleration regime override applied")
                
                elif 'pattern_completion' in regime_override_signal:
                    # Pattern completion - boost technical and momentum
                    adaptive_weights['technical'] = min(0.5, adaptive_weights['technical'] * 1.3)
                    adaptive_weights['momentum'] = min(0.4, adaptive_weights['momentum'] * 1.2)
                    adaptive_weights['news'] *= 0.8
                    print(f"   [UP] Pattern completion regime override applied")
            
            print(f"   [CHART] Graph boost factors: Momentum={momentum_boost_factor:.2f}x, Technical={technical_boost_factor:.2f}x")
            print(f"   [CHART] Weight changes: Mom {original_momentum_weight:.3f}→{adaptive_weights['momentum']:.3f}, Tech {original_technical_weight:.3f}→{adaptive_weights['technical']:.3f}")
        
        # Use adaptive weights
        momentum_weight = adaptive_weights['momentum']
        technical_weight = adaptive_weights['technical']
        news_weight = adaptive_weights['news']
        volatility_weight = adaptive_weights['volatility']
        
        # ENHANCEMENT: Apply momentum amplification on top of adaptive weights
        momentum_amplification_applied = False
        if prediction_days <= 5:
            if thirty_day_return > 15:  # Strong positive momentum
                momentum_boost = 0.15  # Boost momentum weight
                momentum_weight = min(0.55, momentum_weight + momentum_boost)
                technical_weight *= 0.85  # Slightly reduce technical
                momentum_amplification_applied = True
                print(f"   [ROCKET] Momentum amplification: +{momentum_boost:.2f} weight boost for {thirty_day_return:.1f}% return")
            elif thirty_day_return < -15:  # Strong negative momentum
                momentum_boost = 0.10  # Smaller boost for bearish
                momentum_weight = min(0.45, momentum_weight + momentum_boost)
                technical_weight *= 0.90
                momentum_amplification_applied = True
                print(f"   [DOWN] Bearish momentum amplification: +{momentum_boost:.2f} weight boost for {thirty_day_return:.1f}% return")
        
        # Renormalize weights after amplification
        total_weight = momentum_weight + technical_weight + news_weight + volatility_weight
        momentum_weight /= total_weight
        technical_weight /= total_weight
        news_weight /= total_weight
        volatility_weight /= total_weight
        
        # CRITICAL: Respect bearish technical signals more for short-term
        technical_signal_multiplier = 1.2 if trend_signal < 0.5 and prediction_days <= 5 else 1.0
        
        print(f"   [TARGET] FINAL WEIGHTS ({prediction_days}d, {regime_data['main_regime']}): Mom={momentum_weight:.3f}, Tech={technical_weight:.3f}, News={news_weight:.3f}")
        
        # IMPROVEMENT 5: Better signal calculation with context-aware adjustment + GRAPH INTEGRATION
        momentum_component = scaled_momentum * momentum_weight
        
        # Enhanced technical component with acceleration + GRAPH ANALYSIS
        technical_base = (rsi_signal + trend_signal + trend_acceleration - 1.5)  # Center around 0
        
        # NEW: Add graph technical signals
        if graph_analysis:
            # Integrate graph pattern signal
            pattern_adjustment = (graph_pattern_signal - 0.5) * 0.3  # Convert to ±0.15 range
            technical_base += pattern_adjustment
            
            # Add breakout signal (strongest)
            technical_base += graph_breakout_signal
            
            # Add general technical boost
            technical_base += graph_technical_boost
            
            print(f"   [CHART] Graph technical integration: pattern_adj={pattern_adjustment:+.3f}, breakout={graph_breakout_signal:+.3f}, boost={graph_technical_boost:+.3f}")
        
        technical_component = technical_base * technical_weight * technical_signal_multiplier
        
        news_component = news_impact_score * news_weight
        volatility_component = -abs(daily_volatility - 0.02) * volatility_weight
        
        raw_signal = momentum_component + technical_component + news_component + volatility_component
        
        # NEW: Context-aware signal adjustment (REPLACES old conservative dampening)
        if thirty_day_return > 20 and raw_signal > 0.1:  # Strong momentum amplification
            signal_override_protection = 1.15  # AMPLIFY instead of dampen
            bearish_protection_applied = False
            print(f"   [ROCKET] Strong momentum amplification: {thirty_day_return:.1f}% 30d return")
        elif thirty_day_return < -10 and raw_signal < -0.1:  # Strong downtrend amplification
            signal_override_protection = 1.1   # Amplify bearish too
            bearish_protection_applied = False
            print(f"   [DOWN] Strong downtrend amplification: {thirty_day_return:.1f}% 30d return")
        elif abs(raw_signal) < 0.05:  # Weak signals
            signal_override_protection = 0.9   # Slight dampening
            bearish_protection_applied = False
            print(f"   [WRENCH] Weak signal dampening applied")
        else:
            signal_override_protection = 1.0   # No dampening for normal signals
            bearish_protection_applied = False
        
        # Apply context-aware adjustment
        adjusted_signal = raw_signal * signal_override_protection
        
        # CHANGE: Use the calibrated signal instead of raw calculation when appropriate
        if abs(calibrated_signal) > abs(adjusted_signal) and calibration_results['strong_bullish_override']:
            print(f"   [WRENCH] Using calibrated signal override: {calibrated_signal:+.3f} vs adjusted: {adjusted_signal:+.3f}")
            overall_signal = calibrated_signal
        else:
            overall_signal = max(-1, min(1, adjusted_signal))
        
        # NEW: FINAL GRAPH ANALYSIS SIGNAL ADJUSTMENT
        if graph_analysis:
            # Check for very strong graph signals that should override
            breakout_strength = graph_analysis.get('breakout_analysis', {}).get('breakout_strength', 0.0)
            pattern_reliability = graph_analysis.get('pattern_detected', {}).get('pattern_reliability', 0.0)
            
            # Strong breakout override
            if (graph_analysis.get('breakout_analysis', {}).get('breakout_detected', False) and 
                breakout_strength > 0.8 and abs(graph_breakout_signal) > 0.25):
                
                breakout_direction = graph_analysis.get('breakout_analysis', {}).get('breakout_direction', 'neutral')
                if breakout_direction in ['bullish', 'bearish']:
                    # Blend with existing signal but give breakout significant weight
                    breakout_signal = 0.6 if breakout_direction == 'bullish' else -0.6
                    graph_override_weight = 0.3  # 30% weight to breakout signal
                    
                    original_signal = overall_signal
                    overall_signal = overall_signal * (1 - graph_override_weight) + breakout_signal * graph_override_weight
                    overall_signal = max(-1, min(1, overall_signal))
                    
                    print(f"   [ROCKET] STRONG BREAKOUT OVERRIDE: {original_signal:+.3f} → {overall_signal:+.3f} (breakout strength: {breakout_strength:.3f})")
        
        # IMPROVEMENT 7: MOMENTUM-AWARE drift adjustment based on signal strength and timeframe
        if abs(overall_signal) > 0.4:  # Very strong signal
            drift_confidence = 0.9  # Increased from 0.8
        elif abs(overall_signal) > 0.3:  # Strong signal
            drift_confidence = 0.8
        elif abs(overall_signal) > 0.1:  # Moderate signal
            drift_confidence = 0.6
        else:  # Weak signal
            drift_confidence = 0.4
        
        # COMPLETELY REWRITTEN: Momentum-aware drift reduction system
        momentum_strength = abs(thirty_day_return)
        
        if prediction_days <= 5:
            # Short-term predictions with momentum consideration
            
            if thirty_day_return > 25 and overall_signal > 0.2:  # EXCEPTIONAL bullish (like AMD)
                drift_reduction = 1.05  # BOOST instead of reduce!
                momentum_override = "exceptional_bullish"
                print(f"   [ROCKET][ROCKET] EXCEPTIONAL bullish momentum override: +5% BOOST (was reduction)")
                
            elif thirty_day_return > 20 and overall_signal > 0.15:  # Very strong bullish
                drift_reduction = 1.0  # NO reduction at all
                momentum_override = "very_strong_bullish"
                print(f"   [ROCKET] Very strong bullish: NO drift reduction applied")
                
            elif thirty_day_return > 15 and overall_signal > 0.1:  # Strong bullish
                drift_reduction = 0.98  # Minimal 2% reduction
                momentum_override = "strong_bullish"
                print(f"   [ROCKET] Strong bullish: Minimal 2% drift reduction")
                
            elif thirty_day_return > 10 and overall_signal > 0.05:  # Moderate bullish
                drift_reduction = 0.95  # Small 5% reduction
                momentum_override = "moderate_bullish"
                
            elif overall_signal > 0.1:  # General bullish without strong momentum
                drift_reduction = 0.92  # Reduced from 0.9 to 0.92
                momentum_override = "general_bullish"
                
            elif overall_signal < -0.3:  # Very bearish - more aggressive reduction
                drift_reduction = 0.6  # Reduced from 0.7
                momentum_override = "very_bearish"
                print(f"   [DOWN] Very bearish signal: Strong drift reduction applied")
                
            elif overall_signal < -0.2:  # Bearish
                drift_reduction = 0.75  # Increased from 0.7
                momentum_override = "bearish"
                
            elif overall_signal < -0.1:  # Moderate bearish
                drift_reduction = 0.88  # Increased from 0.85
                momentum_override = "moderate_bearish"
                
            else:  # Neutral/weak signals
                drift_reduction = 0.9  # Keep same
                momentum_override = "neutral"
                
        else:  # Longer-term predictions (>5 days)
            if thirty_day_return > 20 and overall_signal > 0.2:  # Strong momentum long-term
                drift_reduction = 1.0  # No reduction for strong momentum
                momentum_override = "strong_longterm_bullish"
                print(f"   [ROCKET] Strong long-term bullish: NO drift reduction")
            elif overall_signal > 0.15:
                drift_reduction = 0.98  # Minimal reduction
                momentum_override = "moderate_longterm_bullish"
            elif overall_signal > 0.05:
                drift_reduction = 0.95  # Keep existing
                momentum_override = "general_longterm_bullish"
            else:
                drift_reduction = 0.95  # Keep existing
                momentum_override = "longterm_neutral"
        
        # Store momentum override information
        momentum_override_applied = momentum_override not in ["neutral", "longterm_neutral"]
        
        analysis_results['composite_scores'] = {
            'overall_signal': overall_signal,
            'calibrated_signal_used': abs(calibrated_signal) > abs(adjusted_signal) and calibration_results['strong_bullish_override'],
            'momentum_component': momentum_component,
            'technical_component': technical_component,
            'news_component': news_component,
            'volatility_component': volatility_component,
            'raw_signal': raw_signal,
            'signal_override_protection': signal_override_protection,
            'drift_confidence': drift_confidence,
            'drift_reduction': drift_reduction,
            'signal_strength': abs(overall_signal),
            'signal_direction': 'Bullish' if overall_signal > 0.1 else 'Bearish' if overall_signal < -0.1 else 'Neutral',
            'prediction_timeframe': 'short' if prediction_days <= 5 else 'medium' if prediction_days <= 20 else 'long',
            'bearish_protection_applied': bearish_protection_applied,
            'context_aware_adjustment': True,  # NEW FLAG
            'momentum_amplification_applied': thirty_day_return > 20 and raw_signal > 0.1,  # NEW FLAG
            'regime_adaptive_weights': True,  # NEW FLAG
            
            # NEW: Momentum override tracking
            'momentum_override': momentum_override,
            'momentum_override_applied': momentum_override_applied,
            'drift_boost_applied': drift_reduction > 1.0,  # NEW FLAG for boosts
            'momentum_strength_pct': momentum_strength,
            
            # NEW: Graph analysis integration tracking
            'graph_analysis_integrated': graph_analysis is not None,
            'graph_signal_adjustment': overall_signal - adjusted_signal if graph_analysis else 0.0,
            'graph_regime_override_applied': bool(graph_analysis and graph_analysis.get('integration_signals', {}).get('regime_override_signal')),
            'graph_boost_factors_applied': graph_analysis is not None
        }
        
        # REPLACE STATISTICAL ANALYSIS SECTION WITH MAE-ENFORCED VERSION:
        if len(price_paths) > 0:
            # NEW: Use MAE-enforced daily statistics calculation
            daily_stats = self.calculate_daily_stats_from_paths_enhanced(
                price_paths, current_price, prediction_days, volatility, ticker
            )
            
            # Extract final day statistics (MAE-enforced)
            final_day_stats = daily_stats[-1] if daily_stats else {}
            mae_enforced_final_price = final_day_stats.get('mean_price', np.mean(price_paths[:, -1]))
            
            # Calculate expected return using MAE-enforced price
            raw_expected_return = (mae_enforced_final_price - current_price) / current_price * 100
            
            # ENHANCED: Momentum-aware confidence adjustment
            if thirty_day_return > 25 and overall_signal > 0.2:
                # Exceptional momentum - use higher confidence factor
                confidence_factor = (drift_confidence * 0.9 + 0.1)  # Higher base confidence
                print(f"   [ROCKET][ROCKET] Exceptional momentum confidence boost applied")
            elif thirty_day_return > 15 and overall_signal > 0.1:
                # Strong momentum - standard high confidence
                confidence_factor = (drift_confidence * 0.85 + 0.15)
            else:
                # Standard confidence factor
                confidence_factor = (drift_confidence * 0.8 + 0.2)
            
            # Apply momentum-aware adjustments
            adjusted_expected_return = raw_expected_return * drift_reduction * confidence_factor
            
            # IMPROVED: More generous bearish caps, less restrictive for bullish
            if overall_signal < -0.3 and adjusted_expected_return > 0:  # Very bearish threshold increased
                adjusted_expected_return = min(adjusted_expected_return, 1.0)  # Tighter cap for very bearish
            elif overall_signal < -0.2 and adjusted_expected_return > 0:  # Bearish threshold increased
                adjusted_expected_return = min(adjusted_expected_return, 2.0)  # Reduced from 1.5% to 2.0%
            elif overall_signal < -0.1 and adjusted_expected_return > 0:  # Moderate bearish
                adjusted_expected_return = min(adjusted_expected_return, 4.0)  # Increased from 3.0% to 4.0%
            # NO CAPS for bullish signals - let momentum run!
            
            # NEW: Extract MAE performance metrics
            mae_performance_summary = {
                'final_day_mae': final_day_stats.get('mae_percent', 0),
                'mae_target': final_day_stats.get('mae_target', 0),
                'mae_target_met': final_day_stats.get('mae_target_met', False),
                'volatility_tier': final_day_stats.get('volatility_tier', 'unknown'),
                'avg_mae_all_days': np.mean([day['mae_percent'] for day in daily_stats]) if daily_stats else 0,
                'mae_success_rate': np.mean([day['mae_target_met'] for day in daily_stats]) if daily_stats else 0,
                'avg_range_tightening': np.mean([day['range_tightening_applied'] for day in daily_stats]) if daily_stats else 0,
                'mae_enforcement_applied': True
            }
            
            analysis_results['statistical_metrics'] = {
                'expected_final_price': current_price * (1 + adjusted_expected_return/100),
                'raw_expected_return_pct': raw_expected_return,
                'expected_return_pct': adjusted_expected_return,
                'drift_adjustment_applied': abs(adjusted_expected_return - raw_expected_return) > 0.1,
                'probability_of_gain': final_day_stats.get('prob_above_current', np.mean(price_paths[:, -1] > current_price)),
                'value_at_risk_5pct': final_day_stats.get('percentile_5', np.percentile(price_paths[:, -1], 5)),
                'value_at_risk_95pct': final_day_stats.get('percentile_95', np.percentile(price_paths[:, -1], 95)),
                
                # Enhanced tracking
                'confidence_factor': confidence_factor,
                'momentum_boost_applied': drift_reduction > 1.0,
                'bearish_cap_applied': (overall_signal < -0.1 and adjusted_expected_return > 0 and 
                                    adjusted_expected_return < raw_expected_return * drift_reduction * confidence_factor),
                
                # NEW: MAE-specific metrics
                'mae_enforced_price_used': True,
                'mae_performance': mae_performance_summary,
                'daily_stats_with_mae': daily_stats  # Full daily statistics with MAE data
            }
        
        analysis_results['volatility_metrics'] = {
            'annualized_volatility': volatility,
            'daily_volatility': daily_volatility,
            'volatility_category': 'High' if volatility > 50 else 'Moderate' if volatility > 25 else 'Low',
        }
        
        # NEW: Calculate VOLATILITY-AWARE CONFIDENCE here
        # Get Monte Carlo path count (from our new dynamic system)
        monte_carlo_paths = len(price_paths) if len(price_paths) > 0 else 1000
        
        # Get news confidence (from stored context or default)
        news_confidence = 0.5  # Default
        if hasattr(self, '_last_news_data') and self._last_news_data:
            news_confidence = self._last_news_data.get('confidence_score', 0.5)
            
            # NEW: Boost confidence if graph analysis was integrated in news
            if self._last_news_data.get('graph_analysis_integrated', False):
                graph_confidence_boost = self._last_news_data.get('graph_confidence_boost', 0.0)
                news_confidence = min(0.95, news_confidence + graph_confidence_boost)
                print(f"   [TARGET] Graph-enhanced news confidence: +{graph_confidence_boost:.3f}")
        
        # Calculate enhanced confidence with our new volatility-aware method
        enhanced_confidence, confidence_results = self.enhanced_confidence_integration(
            signal_strength=abs(overall_signal),
            news_confidence=news_confidence,
            drift_factor=annualized_return,  # Use annualized return as drift factor
            volatility=volatility,
            prediction_days=prediction_days,
            monte_carlo_paths=monte_carlo_paths,  # NEW: Pass dynamic path count
            total_news_articles=total_news_articles  # NEW: Pass total news count
        )
        
        # NEW: Apply graph analysis confidence boost if available
        if graph_analysis:
            integration_signals = graph_analysis.get('integration_signals', {})
            graph_confidence_boost = integration_signals.get('confidence_boost_factor', 0.0)
            
            if graph_confidence_boost > 0:
                original_confidence = enhanced_confidence
                enhanced_confidence = min(0.95, enhanced_confidence + graph_confidence_boost)
                print(f"   [UP] Graph analysis confidence boost: {original_confidence:.3f} → {enhanced_confidence:.3f} (+{graph_confidence_boost:.3f})")
        
        # Store enhanced confidence results
        analysis_results['enhanced_confidence'] = enhanced_confidence
        analysis_results['confidence_breakdown'] = confidence_results
        analysis_results['volatility_aware_confidence'] = True  # NEW FLAG
        
        # Store regime analysis results
        analysis_results['regime_analysis'] = regime_data
        analysis_results['adaptive_weights'] = adaptive_weights
        analysis_results['final_weights'] = {
            'momentum': momentum_weight,
            'technical': technical_weight,
            'news': news_weight,
            'volatility': volatility_weight
        }
        
        # NEW: Store Monte Carlo metadata for transparency
        analysis_results['monte_carlo_metadata'] = {
            'path_count': monte_carlo_paths,
            'volatility_tier': confidence_results.get('volatility_tier', 'Unknown'),
            'data_quality_bonus': confidence_results.get('data_quality_bonus', 0),
            'volatility_penalty': confidence_results.get('volatility_penalty', 0)
        }
        
        # NEW: Store comprehensive graph analysis integration results
        if graph_analysis:
            analysis_results['graph_integration_results'] = {
                'graph_analysis_applied': True,
                'momentum_boost_applied': graph_momentum_boost != 0.0,
                'technical_boost_applied': graph_technical_boost != 0.0,
                'breakout_override_applied': abs(graph_breakout_signal) > 0.1,
                'pattern_signal_applied': abs(graph_pattern_signal - 0.5) > 0.1,
                'regime_override_applied': bool(graph_analysis.get('integration_signals', {}).get('regime_override_signal')),
                'confidence_boost_applied': graph_analysis.get('integration_signals', {}).get('confidence_boost_factor', 0.0) > 0,
                'integration_summary': {
                    'primary_pattern': graph_analysis.get('pattern_detected', {}).get('primary_pattern'),
                    'pattern_reliability': graph_analysis.get('pattern_detected', {}).get('pattern_reliability', 0.0),
                    'breakout_detected': graph_analysis.get('breakout_analysis', {}).get('breakout_detected', False),
                    'breakout_strength': graph_analysis.get('breakout_analysis', {}).get('breakout_strength', 0.0),
                    'momentum_acceleration': graph_analysis.get('momentum_analysis', {}).get('momentum_acceleration', 0.5),
                    'candlestick_patterns': graph_analysis.get('candlestick_analysis', {}).get('total_patterns_found', 0),
                    'support_resistance_quality': graph_analysis.get('support_resistance', {}).get('level_quality', 0.0)
                }
            }
        else:
            analysis_results['graph_integration_results'] = {
                'graph_analysis_applied': False,
                'integration_summary': 'No graph analysis provided'
            }
        
        # Enhanced logging with MAE information AND GRAPH ANALYSIS
        mae_info = analysis_results['statistical_metrics']['mae_performance']
        print(f"   [OK] MAE-ENFORCED analysis complete with VOLATILITY-AWARE CONFIDENCE and GRAPH ANALYSIS")
        print(f"   [CHART] Market regime: {regime_data['main_regime']} (confidence: {regime_data['regime_confidence']:.3f})")
        print(f"   [CHART] Overall signal: {overall_signal:+.3f} (raw: {raw_signal:+.3f})")
        print(f"   [CHART] Expected return: {analysis_results['statistical_metrics']['expected_return_pct']:+.2f}%")
        print(f"   [CHART] Drift reduction: {drift_reduction:.3f} (override: {momentum_override})")
        print(f"   [CHART] Drift confidence: {drift_confidence:.1%}")
        print(f"   [TARGET] VOLATILITY-AWARE Confidence: {enhanced_confidence:.1%} ({confidence_results.get('volatility_tier', 'Unknown')} volatility tier)")
        print(f"   [TARGET] MAE Performance: {mae_info['final_day_mae']:.1f}% vs target {mae_info['mae_target']:.1f}% ({'[OK]' if mae_info['mae_target_met'] else '[ERROR]'})")
        print(f"   [TARGET] MAE Success Rate: {mae_info['mae_success_rate']:.1%} across {len(daily_stats) if 'daily_stats' in locals() else 0} days")
        
        if graph_analysis:
            print(f"   [UP] Graph Analysis Integration: [OK] APPLIED")
            print(f"   [CHART] Graph momentum boost: {graph_momentum_boost:+.4f}")
            print(f"   [CHART] Graph technical boost: {graph_technical_boost:+.3f}")
            print(f"   [CHART] Graph breakout signal: {graph_breakout_signal:+.3f}")
            if graph_analysis.get('integration_signals', {}).get('regime_override_signal'):
                print(f"   🚨 Graph regime override: {graph_analysis.get('integration_signals', {}).get('regime_override_signal')}")
        
        return analysis_results
    
    def set_news_data_for_regime(self, news_data):
        """Store news data for regime detection"""
        self._last_news_data = news_data
    
    def enhanced_directional_bias_calibration(self, excel_data, market_data, custom_articles):
        """ENHANCED directional bias calibration with stronger momentum amplification"""
        print("[TARGET] Applying enhanced directional bias calibration...")
        
        # Base calculations
        thirty_day_return = excel_data.get('performance_return_1_month', 0.0)
        recent_5d_return = self.calculate_recent_momentum(excel_data)
        momentum_score = thirty_day_return / 100 / 12  # Convert to monthly decimal
        
        # Enhanced news momentum with stronger catalyst detection
        news_momentum_score = 0
        total_catalyst_score = 0
        
        if custom_articles:
            for article in custom_articles:
                # Enhanced catalyst detection
                catalyst_analysis = self.detect_enhanced_financial_catalysts(
                    article.get('content', article.get('title', '')),
                    market_data.get('symbol', 'STOCK')
                )
                
                # Enhanced sentiment analysis
                sentiment_analysis = self.analyze_enhanced_sentiment_momentum(
                    article.get('content', article.get('title', '')), 
                    market_data.get('symbol', 'STOCK')
                )
                
                # Combine sentiment with catalyst boost
                article_impact = (
                    sentiment_analysis['momentum_score'] * 0.6 + 
                    article.get('sentiment_score', 0) * 0.4
                ) * (1 + catalyst_analysis['catalyst_score'])
                
                news_momentum_score += article_impact
                total_catalyst_score += catalyst_analysis['catalyst_score']
        
        if custom_articles:
            news_momentum_score /= len(custom_articles)
        
        # ENHANCED BIAS CALIBRATION RULES with stronger amplification
        base_signal = 0
        
        # Rule 1: STRONGER bullish override for clearly bullish stocks
        if thirty_day_return > 25:  # Very strong momentum
            print(f"   [ROCKET][ROCKET] VERY strong bullish override: {thirty_day_return:.1f}% 30d return")
            base_signal = max(0.2, min(0.4, momentum_score * 10))  # Even stronger signal
        elif thirty_day_return > 15:  # Strong momentum (lowered threshold)
            print(f"   [ROCKET] Strong bullish override triggered: {thirty_day_return:.1f}% 30d return")
            base_signal = max(0.1, min(0.25, momentum_score * 7))  # Increased from 5 to 7
        elif thirty_day_return > 8:  # Moderate momentum (lowered from 10)
            print(f"   [UP] Moderate bullish override: {thirty_day_return:.1f}% 30d return")
            base_signal = max(0.05, min(0.15, momentum_score * 5))
        
        # Rule 2: Enhanced moderate bullish nudge (lowered threshold)
        elif thirty_day_return > 3 and momentum_score > 0.003:  # Lowered from 5 and 0.005
            print(f"   [UP] Moderate bullish nudge: {thirty_day_return:.1f}% 30d return")
            base_signal = max(0.03, momentum_score * 4)  # Increased from 3 to 4
        
        # NEW Rule 2.5: Handle strong bearish momentum
        elif thirty_day_return < -15:  # Strong bearish momentum
            print(f"   [DOWN] Strong bearish override: {thirty_day_return:.1f}% 30d return")
            base_signal = min(-0.1, max(-0.25, momentum_score * 6))  # Amplify bearish signals
        elif thirty_day_return < -8:  # Moderate bearish momentum
            print(f"   [DOWN] Moderate bearish override: {thirty_day_return:.1f}% 30d return")
            base_signal = min(-0.05, max(-0.15, momentum_score * 4))
        
        # Rule 3: Enhanced recent momentum boost (lowered threshold)
        if recent_5d_return > 1.5:  # Lowered from 2
            recent_boost = min(0.15, recent_5d_return / 100 * 3)  # Increased from 2 to 3
            base_signal += recent_boost
            print(f"   ⚡ Recent momentum boost: +{recent_boost:.3f} from {recent_5d_return:.1f}% 5d return")
        elif recent_5d_return < -1.5:  # NEW: Recent bearish momentum
            recent_boost = max(-0.15, recent_5d_return / 100 * 3)
            base_signal += recent_boost
            print(f"   ⚡ Recent bearish momentum: {recent_boost:.3f} from {recent_5d_return:.1f}% 5d return")
        
        # Rule 4: Enhanced catalyst boost (lowered threshold)
        if total_catalyst_score > 0.6:  # Lowered from 0.8
            catalyst_boost = min(0.2, total_catalyst_score * 0.25)  # Increased from 0.15 and 0.2
            base_signal += catalyst_boost
            print(f"   🎆 Strong catalyst boost: +{catalyst_boost:.3f} (score: {total_catalyst_score:.2f})")
        elif total_catalyst_score > 0.3:  # NEW: Moderate catalyst boost
            catalyst_boost = min(0.1, total_catalyst_score * 0.2)
            base_signal += catalyst_boost
            print(f"   🎆 Moderate catalyst boost: +{catalyst_boost:.3f} (score: {total_catalyst_score:.2f})")
        
        # NEW Rule 5: Technical momentum alignment
        current_price = market_data.get('current_price', 0)
        sma_20 = excel_data.get('sma_20', current_price)
        if current_price > 0 and sma_20 > 0:
            price_above_sma = (current_price - sma_20) / sma_20
            if price_above_sma > 0.05 and thirty_day_return > 5:  # Price well above SMA + positive momentum
                technical_boost = min(0.1, price_above_sma * 0.5)
                base_signal += technical_boost
                print(f"   [CHART] Technical momentum alignment: +{technical_boost:.3f} (price {price_above_sma*100:+.1f}% above SMA20)")
        
        # Enhanced signal calculation with stronger weighting
        enhanced_signal = (
            base_signal + 
            0.3 * (recent_5d_return / 100) +  # Increased from 0.2 to 0.3
            0.15 * news_momentum_score  # Increased from 0.1 to 0.15
        )
        
        # Apply final calibration with wider range
        calibrated_signal = max(-1.0, min(1.0, enhanced_signal))  # Increased from -0.8/0.8 to -1.0/1.0
        
        # NEW: Additional boost for very strong signals
        if abs(calibrated_signal) > 0.5:
            signal_boost = min(0.2, abs(calibrated_signal) * 0.3)
            calibrated_signal += signal_boost if calibrated_signal > 0 else -signal_boost
            calibrated_signal = max(-1.0, min(1.0, calibrated_signal))
            print(f"   [ROCKET] Strong signal boost applied: +{signal_boost:.3f}")
        
        calibration_results = {
            'base_signal': base_signal,
            'recent_5d_return_pct': recent_5d_return,
            'news_momentum_score': news_momentum_score,
            'total_catalyst_score': total_catalyst_score,
            'enhanced_signal': enhanced_signal,
            'calibrated_signal': calibrated_signal,
            'strong_bullish_override': thirty_day_return > 15 and momentum_score > 0.01,  # Lowered from 10
            'very_strong_bullish_override': thirty_day_return > 25,  # NEW
            'strong_bearish_override': thirty_day_return < -15,  # NEW
            'catalyst_boost_applied': total_catalyst_score > 0.6,  # Lowered from 0.8
            'technical_alignment_boost': current_price > sma_20 * 1.05 and thirty_day_return > 5,  # NEW
            'signal_amplification_applied': abs(calibrated_signal) > 0.5  # NEW
        }
        
        print(f"   [OK] Enhanced calibrated signal: {calibrated_signal:+.3f} (enhanced: {enhanced_signal:+.3f})")
        return calibrated_signal, calibration_results
    
    # =============================================================================
    # METHOD 2: Enhanced Drift Scaling (REPLACE EXISTING)
    # =============================================================================

    def enhanced_drift_scaling(self, base_drift, momentum_score, volatility, news_momentum, recent_performance):
        """IMPROVED drift scaling with stronger momentum capture for high-momentum stocks"""
        print("🎲 Applying IMPROVED enhanced drift scaling...")
        
        # Normalize volatility impact (0-1 scale)
        normalized_volatility = min(volatility / 100, 1.0)  # Cap at 100%
        volatility_impact = normalized_volatility * 0.3  # Up to 30% boost
        
        # ENHANCED: More aggressive momentum amplification with tier system
        momentum_amplification = 1.0
        if momentum_score > 0.01:  # Bullish momentum
            if momentum_score > 0.05:  # VERY strong momentum (>60% annualized)
                momentum_amplification = 1.0 + min(6.0, abs(momentum_score) * 30)  # BOOSTED: from 20 to 30
                print(f"   [ROCKET][ROCKET] VERY strong bullish momentum: {abs(momentum_score):.3f} → {momentum_amplification:.1f}x")
            elif momentum_score > 0.02:  # Strong momentum (>24% annualized)
                momentum_amplification = 1.0 + min(5.0, abs(momentum_score) * 25)  # BOOSTED: from 20 to 25
                print(f"   [ROCKET] Strong bullish momentum: {abs(momentum_score):.3f} → {momentum_amplification:.1f}x")
            else:  # Moderate momentum
                momentum_amplification = 1.0 + min(4.0, abs(momentum_score) * 22)  # BOOSTED: from 20 to 22
        elif momentum_score < -0.01:  # Bearish momentum
            if momentum_score < -0.05:  # VERY strong bearish
                momentum_amplification = 1.0 + min(4.5, abs(momentum_score) * 20)  # Increased from 16 to 20
                print(f"   [DOWN][DOWN] VERY strong bearish momentum: {abs(momentum_score):.3f} → {momentum_amplification:.1f}x")
            elif momentum_score < -0.02:  # Strong bearish
                momentum_amplification = 1.0 + min(3.5, abs(momentum_score) * 18)  # Increased from 16 to 18
                print(f"   [DOWN] Strong bearish momentum: {abs(momentum_score):.3f} → {momentum_amplification:.1f}x")
            else:  # Moderate bearish
                momentum_amplification = 1.0 + min(3.0, abs(momentum_score) * 16)  # Keep existing
        
        # ENHANCED: More generous news amplification with momentum interaction
        base_news_amplification = 1.0 + min(1.2, abs(news_momentum) * 4)
        
        # Boost news impact if it aligns with momentum
        if (momentum_score > 0.01 and news_momentum > 0.1) or (momentum_score < -0.01 and news_momentum < -0.1):
            news_momentum_synergy = 1.2  # 20% synergy bonus
            news_amplification = base_news_amplification * news_momentum_synergy
            print(f"   📰💪 News-momentum synergy bonus: {news_momentum_synergy}x")
        else:
            news_amplification = base_news_amplification
        
        # ENHANCED: More aggressive performance amplification with momentum tiers
        performance_amplification = 1.0
        if abs(recent_performance) > 20:  # Exceptional performance (like AMD's 26.92%)
            performance_amplification = 1.0 + min(2.0, abs(recent_performance) / 100 * 12)  # BOOSTED: from 8 to 12
            print(f"   [TARGET][TARGET] Exceptional performance boost: {recent_performance:.1f}% → {performance_amplification:.1f}x")
        elif abs(recent_performance) > 10:  # Strong performance
            performance_amplification = 1.0 + min(1.8, abs(recent_performance) / 100 * 10)  # BOOSTED: from 8 to 10
            print(f"   [TARGET] Strong performance boost: {recent_performance:.1f}% → {performance_amplification:.1f}x")
        elif abs(recent_performance) > 5:  # Moderate performance
            performance_amplification = 1.0 + min(1.5, abs(recent_performance) / 100 * 8)  # Keep existing
        
        # IMPROVED: Dynamic weighting based on momentum strength
        if abs(momentum_score) > 0.03:  # Very high momentum stocks
            momentum_weight = 0.6   # Increased from 0.5
            news_weight = 0.25      # Reduced from 0.3
            performance_weight = 0.15  # Reduced from 0.2
            print(f"   ⚖️ Very high momentum weighting: Mom=60%, News=25%, Perf=15%")
        elif abs(momentum_score) > 0.015:  # High momentum stocks (like AMD)
            momentum_weight = 0.55  # Increased from 0.5
            news_weight = 0.27      # Reduced from 0.3
            performance_weight = 0.18  # Reduced from 0.2
            print(f"   ⚖️ High momentum weighting: Mom=55%, News=27%, Perf=18%")
        else:  # Standard/low momentum stocks
            momentum_weight = 0.5   # Keep original
            news_weight = 0.3
            performance_weight = 0.2
        
        # Calculate enhanced drift with improved weighting
        drift_multiplier = (
            momentum_amplification * momentum_weight + 
            news_amplification * news_weight + 
            performance_amplification * performance_weight
        )
        
        # ENHANCED: Apply volatility boost and momentum-specific adjustment
        volatility_boost = 1 + volatility_impact
        
        # Additional momentum-volatility interaction
        if abs(momentum_score) > 0.02 and volatility > 30:
            # High momentum + high volatility = extra amplification (momentum can overcome volatility)
            momentum_vol_synergy = 1.1
            print(f"   🔥 High momentum + high volatility synergy: {momentum_vol_synergy}x")
        else:
            momentum_vol_synergy = 1.0
        
        enhanced_drift = base_drift * drift_multiplier * volatility_boost * momentum_vol_synergy
        
        # IMPROVED: More generous safety caps with momentum consideration
        if abs(momentum_score) > 0.03:  # Very high momentum
            max_drift = min(120, volatility * 2.5)  # More generous cap
        elif abs(momentum_score) > 0.015:  # High momentum (like AMD)
            max_drift = min(100, volatility * 2.2)  # Increased from 2.0 to 2.2
        else:
            max_drift = min(80, volatility * 2.0)   # Keep original
        
        enhanced_drift = max(-max_drift, min(max_drift, enhanced_drift))
        
        # Check if we hit the cap (might indicate need for adjustment)
        hit_cap = abs(enhanced_drift) >= max_drift * 0.95
        if hit_cap:
            print(f"   [WARNING] Drift near/at cap: {enhanced_drift:.1f}% (cap: ±{max_drift:.1f}%)")
        
        drift_scaling_results = {
            'base_drift': base_drift,
            'momentum_amplification': momentum_amplification,
            'news_amplification': news_amplification,
            'performance_amplification': performance_amplification,
            'volatility_impact': volatility_impact,
            'momentum_vol_synergy': momentum_vol_synergy,
            'drift_multiplier': drift_multiplier,
            'enhanced_drift': enhanced_drift,
            'max_drift_cap': max_drift,
            'weighting_system': {
                'momentum_weight': momentum_weight,
                'news_weight': news_weight,
                'performance_weight': performance_weight
            },
            'momentum_tier': 'very_high' if abs(momentum_score) > 0.03 else 'high' if abs(momentum_score) > 0.015 else 'standard',
            'hit_drift_cap': hit_cap,
            'synergy_bonuses': {
                'news_momentum_synergy': news_momentum_synergy if 'news_momentum_synergy' in locals() else 1.0,
                'momentum_vol_synergy': momentum_vol_synergy
            }
        }
        
        print(f"   [OK] IMPROVED drift: {enhanced_drift:.2f}% (base: {base_drift:.2f}%, multiplier: {drift_multiplier:.2f}x)")
        print(f"   [CHART] Momentum tier: {drift_scaling_results['momentum_tier']}, Amplification: {momentum_amplification:.1f}x")
        
        return enhanced_drift, drift_scaling_results

    # =============================================================================
    # METHOD 3: Enhanced Range Calibration (REPLACE EXISTING)
    # =============================================================================

    def get_volatility_mae_targets(self, volatility, is_tech_stock=False):
        """Get dynamic MAE targets based on volatility tier"""
        
        # Base MAE targets by volatility tier
        if volatility < 20:
            base_target = 0.015  # 1.5% for low volatility
            tier = 'low'
        elif volatility < 35:
            base_target = 0.022  # 2.2% for medium volatility
            tier = 'medium'
        elif volatility < 50:
            base_target = 0.030  # 3.0% for high volatility
            tier = 'high'
        else:
            base_target = 0.040  # 4.0% for extreme volatility
            tier = 'extreme'
        
        # Tech stock adjustment (slightly more lenient due to higher inherent volatility)
        if is_tech_stock and volatility > 30:
            tech_adjustment = 0.003  # +0.3% for volatile tech stocks
            base_target += tech_adjustment
        
        return {
            'target': base_target,
            'tier': tier,
            'is_tech_adjusted': is_tech_stock and volatility > 30
        }

    def enforce_mae_target(self, initial_range_pct, mean_price, target_mae, prediction_day):
        """Enforce MAE target through range compression and progressive tightening"""
        
        # Calculate current MAE from initial range
        current_mae = initial_range_pct  # Range percentage is equivalent to MAE
        
        # Progressive tightening by day (ranges should get tighter over time for accuracy)
        day_compression_factors = {
            1: 1.0,    # No compression for day 1
            2: 0.95,   # 5% tighter for day 2
            3: 0.90,   # 10% tighter for day 3
            4: 0.87,   # 13% tighter for day 4
            5: 0.85,   # 15% tighter for day 5
        }
        
        # Get day compression (default to 0.80 for days > 5)
        day_compression = day_compression_factors.get(prediction_day, 0.80)
        
        # Calculate compression factor to meet MAE target
        mae_compression = target_mae / current_mae if current_mae > target_mae else 1.0
        
        # Combine both compression factors
        total_compression = mae_compression * day_compression
        
        # Apply compression to range
        compressed_range_pct = initial_range_pct * total_compression
        
        # Validate final MAE
        final_mae = compressed_range_pct
        
        compression_results = {
            'initial_mae': current_mae,
            'target_mae': target_mae,
            'mae_compression': mae_compression,
            'day_compression': day_compression,
            'total_compression': total_compression,
            'final_mae': final_mae,
            'mae_target_met': final_mae <= target_mae * 1.05  # 5% tolerance
        }
        
        return compressed_range_pct, compression_results

    def enhanced_range_calibration(self, mean_price, current_price, volatility, ticker, prediction_day):
        """ENHANCED range calibration with MAE enforcement and volatility-aware targets"""
        print(f"[CHART] Applying MAE-enforced range calibration for day {prediction_day}...")
        
        # Calculate basic volatility metrics
        daily_volatility = volatility / math.sqrt(252)
        
        # Tech stock identification
        tech_tickers = ['NVDA', 'AMD', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'TEM', 'SMCI']
        is_tech_stock = any(t in ticker.upper() for t in tech_tickers)
        
        # STEP 1: Get volatility-specific MAE target
        mae_target_info = self.get_volatility_mae_targets(volatility, is_tech_stock)
        target_mae = mae_target_info['target']
        volatility_tier = mae_target_info['tier']
        
        print(f"   [TARGET] MAE target: {target_mae:.1%} ({volatility_tier} volatility tier)")
        if mae_target_info['is_tech_adjusted']:
            print(f"   🔬 Tech stock adjustment applied: +0.3%")
        
        # STEP 2: Volatility-tier specific base multipliers (REPLACING fixed 2.2x)
        volatility_multipliers = {
            'low': 1.4,      # <20% volatility - much tighter
            'medium': 1.6,   # 20-35% volatility - tighter  
            'high': 1.8,     # 35-50% volatility - moderate
            'extreme': 2.0   # >50% volatility - still tighter than old 2.2x
        }
        
        base_multiplier = volatility_multipliers[volatility_tier]
        base_range_pct = daily_volatility * base_multiplier
        
        print(f"   [CHART] Volatility tier: {volatility_tier} → {base_multiplier}x multiplier (was 2.2x)")
        
        # STEP 3: Enhanced time expansion (reduced from original)
        if volatility_tier == 'extreme':
            time_expansion = 1.0 + (prediction_day - 1) * 0.04  # Reduced from 0.06
        elif volatility_tier == 'high':
            time_expansion = 1.0 + (prediction_day - 1) * 0.03  # Reduced from 0.05
        else:
            time_expansion = 1.0 + (prediction_day - 1) * 0.02  # Much more conservative
        
        expanded_range_pct = base_range_pct * time_expansion
        
        # STEP 4: Enhanced volatility buffer (reduced)
        if is_tech_stock and volatility > 30:
            volatility_buffer = 0.004  # Reduced from 0.007
            print(f"   🔬 Tech stock buffer: +{volatility_buffer:.1%} (reduced)")
        elif volatility > 50:
            volatility_buffer = 0.003  # Reduced from 0.006
        else:
            volatility_buffer = 0.002  # Reduced from 0.003
        
        # STEP 5: Initial range calculation
        initial_range_pct = expanded_range_pct + volatility_buffer
        
        # STEP 6: APPLY MAE ENFORCEMENT
        compressed_range_pct, compression_results = self.enforce_mae_target(
            initial_range_pct, mean_price, target_mae, prediction_day
        )
        
        print(f"   [TARGET] MAE enforcement: {initial_range_pct:.1%} → {compressed_range_pct:.1%}")
        print(f"   [CHART] Compression: {compression_results['total_compression']:.3f}x")
        
        # STEP 7: Calculate final price range
        range_amount = mean_price * compressed_range_pct
        enhanced_low = mean_price - range_amount
        enhanced_high = mean_price + range_amount
        
        # STEP 8: Downside protection (adjusted for new tighter ranges)
        distance_from_current = abs(mean_price - current_price) / current_price
        if distance_from_current > 0.10:  # Slightly higher threshold due to tighter ranges
            # Reduced extra protection since ranges are already tighter
            extra_downside = current_price * 0.005  # Reduced from 0.008
            enhanced_low -= extra_downside
            print(f"   🛡️ Downside protection: -{extra_downside:.2f} (reduced)")
        
        # STEP 9: Tech stock protection (reduced)
        if is_tech_stock and volatility > 40:
            tech_buffer = current_price * 0.003  # Reduced from 0.005
            enhanced_low -= tech_buffer
            print(f"   [ROCKET] Tech buffer: -{tech_buffer:.2f} (reduced)")
        
        # STEP 10: Final MAE validation
        final_range_width = enhanced_high - enhanced_low
        final_mae_pct = (final_range_width / (2 * mean_price)) * 100
        mae_success = final_mae_pct <= (target_mae * 100 * 1.05)  # 5% tolerance
        
        # STEP 11: Emergency compression if still over target
        if not mae_success:
            emergency_compression = (target_mae * 100) / final_mae_pct
            emergency_range = final_range_width * emergency_compression * 0.95  # Extra 5% safety
            enhanced_low = mean_price - (emergency_range / 2)
            enhanced_high = mean_price + (emergency_range / 2)
            final_mae_pct = (emergency_range / (2 * mean_price)) * 100
            print(f"   🚨 Emergency compression: {emergency_compression:.3f}x")
        
        # STEP 12: Comprehensive results
        range_calibration_results = {
            # MAE-specific results
            'mae_target_pct': target_mae * 100,
            'final_mae_pct': final_mae_pct,
            'mae_target_met': final_mae_pct <= (target_mae * 100 * 1.05),
            'volatility_tier': volatility_tier,
            'compression_applied': compression_results,
            
            # Traditional results (updated)
            'base_range_pct': base_range_pct * 100,
            'base_multiplier': base_multiplier,
            'time_expansion': time_expansion,
            'volatility_buffer_pct': volatility_buffer * 100,
            'initial_range_pct': initial_range_pct * 100,
            'final_range_pct': compressed_range_pct * 100,
            'range_width': enhanced_high - enhanced_low,
            'is_tech_stock': is_tech_stock,
            'enhanced_protection_applied': distance_from_current > 0.10,
            
            # Quality metrics
            'mae_improvement': ((initial_range_pct - compressed_range_pct) / initial_range_pct) * 100,
            'range_tightening_pct': (1 - compression_results['total_compression']) * 100
        }
        
        # Enhanced logging
        status_emoji = "[OK]" if mae_success else "[ERROR]"
        print(f"   [UP] Final Range: ${enhanced_low:.2f} - ${enhanced_high:.2f}")
        print(f"   [TARGET] MAE Result: {final_mae_pct:.1f}% {status_emoji} (target: {target_mae*100:.1f}%)")
        print(f"   [CHART] Range tightening: {range_calibration_results['range_tightening_pct']:.1f}%")
        
        return enhanced_low, enhanced_high, range_calibration_results
    # =============================================================================
    # METHOD 4: Enhanced Confidence Integration (REPLACE EXISTING)
    # =============================================================================

    def enhanced_confidence_integration(self, signal_strength, news_confidence, drift_factor, volatility, prediction_days, monte_carlo_paths=1000, total_news_articles=0):
        """VOLATILITY-AWARE confidence integration with dynamic penalty system"""
        
        print(f"[TARGET] Calculating volatility-aware confidence for {volatility:.1f}% volatility...")
        
        # STEP 1: Determine volatility tier and base confidence range
        if volatility < 20:
            vol_tier = "Low"
            base_confidence_range = (70, 95)  # High confidence possible
            vol_predictability_factor = 1.2    # Boost for predictable stocks
        elif volatility < 35:
            vol_tier = "Medium" 
            base_confidence_range = (60, 88)  # Standard confidence
            vol_predictability_factor = 1.0    # Neutral
        elif volatility < 50:
            vol_tier = "High"
            base_confidence_range = (45, 80)  # Reduced confidence range
            vol_predictability_factor = 0.85   # Penalty for unpredictable stocks
        else:
            vol_tier = "Extreme"
            base_confidence_range = (30, 70)  # Much lower confidence
            vol_predictability_factor = 0.7    # Heavy penalty
        
        min_confidence, max_confidence = base_confidence_range
        print(f"   [CHART] Volatility tier: {vol_tier} → Confidence range: {min_confidence}-{max_confidence}%")
        
        # STEP 2: Calculate base components with volatility awareness
        
        # Signal component - scaled by volatility tier
        signal_base = min(55, abs(signal_strength) * 140)
        signal_component = signal_base * vol_predictability_factor
        
        # News component - higher volatility stocks need more news for same confidence
        if volatility < 20:
            news_multiplier = 70  # Standard news impact
        elif volatility < 35:
            news_multiplier = 65  # Slightly reduced
        elif volatility < 50:
            news_multiplier = 55  # Reduced news impact
        else:
            news_multiplier = 45  # Much reduced for extreme volatility
        
        news_component = min(35, news_confidence * news_multiplier)
        
        # Drift component - volatility affects drift reliability
        drift_base = min(25, abs(drift_factor) / 35 * 30)
        drift_component = drift_base * vol_predictability_factor
        
        # STEP 3: Enhanced volatility penalty system
        
        # Base volatility penalty - much stronger for high volatility
        if volatility > 60:
            base_vol_penalty = min(35, volatility / 3)  # INCREASED penalty
        elif volatility > 45:
            base_vol_penalty = min(25, volatility / 4)  # INCREASED penalty
        elif volatility > 30:
            base_vol_penalty = min(15, volatility / 6)  # INCREASED penalty
        else:
            base_vol_penalty = min(8, volatility / 10)   # Minimal penalty for low vol
        
        # Signal-volatility interaction - strong signals can partially offset volatility
        if abs(signal_strength) > 0.4 and volatility < 60:
            # Strong signal reduces volatility penalty (but not for extreme volatility)
            signal_vol_offset = min(8, abs(signal_strength) * 15)
            volatility_penalty = max(3, base_vol_penalty - signal_vol_offset)
            print(f"   [TARGET] Strong signal offset: -{signal_vol_offset:.1f} volatility penalty")
        else:
            volatility_penalty = base_vol_penalty
        
        # STEP 4: Data quality adjustments
        
        # Monte Carlo path quality bonus
        if monte_carlo_paths >= 4000:
            mc_quality_bonus = 3  # High path count
        elif monte_carlo_paths >= 2500:
            mc_quality_bonus = 2  # Medium path count
        elif monte_carlo_paths >= 1500:
            mc_quality_bonus = 1  # Low path count
        else:
            mc_quality_bonus = 0  # Very low path count
        
        # News volume quality bonus (but capped for high volatility)
        news_volume_bonus = min(3, total_news_articles / 10)
        if volatility > 40:
            news_volume_bonus *= 0.6  # Reduce news impact for high volatility
        
        data_quality_bonus = mc_quality_bonus + news_volume_bonus
        
        # STEP 5: Time penalty - stronger for volatile stocks
        if prediction_days <= 3:
            time_penalty = 0
        elif prediction_days <= 7:
            base_time_penalty = (prediction_days - 3) * 0.3
            # Increase time penalty for volatile stocks
            time_penalty = base_time_penalty * (1 + volatility / 200)
        elif prediction_days <= 20:
            base_time_penalty = 1.2 + (prediction_days - 7) * 0.4
            time_penalty = base_time_penalty * (1 + volatility / 150)
        else:
            base_time_penalty = 6.4 + (prediction_days - 20) * 0.6
            time_penalty = base_time_penalty * (1 + volatility / 100)
        
        # STEP 6: Calculate final confidence
        base_confidence = signal_component + news_component + drift_component + data_quality_bonus
        adjusted_confidence = base_confidence - volatility_penalty - time_penalty
        
        # Apply volatility tier confidence bounds
        preliminary_confidence = max(min_confidence, min(max_confidence, adjusted_confidence))
        
        # STEP 7: Final signal strength adjustments (within volatility bounds)
        if abs(signal_strength) > 0.6:
            # Very strong signal boost - but capped by volatility tier
            signal_boost = min(5, max_confidence - preliminary_confidence)
            final_confidence = preliminary_confidence + signal_boost
        elif abs(signal_strength) > 0.4:
            # Strong signal boost - smaller and capped
            signal_boost = min(3, max_confidence - preliminary_confidence)
            final_confidence = preliminary_confidence + signal_boost
        else:
            final_confidence = preliminary_confidence
        
        # STEP 8: Absolute bounds enforcement
        final_confidence = max(min_confidence, min(max_confidence, final_confidence))
        
        # STEP 9: Special cases and validation
        
        # Extreme volatility with weak signal - extra penalty
        if volatility > 60 and abs(signal_strength) < 0.2:
            extreme_penalty = 5
            final_confidence = max(min_confidence, final_confidence - extreme_penalty)
            print(f"   [WARNING] Extreme volatility + weak signal penalty: -{extreme_penalty}")
        
        # Very short-term high volatility - additional penalty
        if prediction_days <= 2 and volatility > 45:
            short_term_vol_penalty = 3
            final_confidence = max(min_confidence, final_confidence - short_term_vol_penalty)
            print(f"   [WARNING] Short-term high volatility penalty: -{short_term_vol_penalty}")
        
        # STEP 10: Prepare detailed results
        confidence_results = {
            'volatility_tier': vol_tier,
            'vol_predictability_factor': vol_predictability_factor,
            'confidence_range': base_confidence_range,
            'signal_component': signal_component,
            'news_component': news_component,
            'drift_component': drift_component,
            'volatility_penalty': volatility_penalty,
            'time_penalty': time_penalty,
            'data_quality_bonus': data_quality_bonus,
            'mc_quality_bonus': mc_quality_bonus,
            'news_volume_bonus': news_volume_bonus,
            'base_confidence': base_confidence,
            'preliminary_confidence': preliminary_confidence,
            'final_confidence': final_confidence / 100,
            'signal_boost_applied': abs(signal_strength) > 0.4,
            'volatility_adjusted': True,
            'confidence_factors': {
                'signal_strength': abs(signal_strength),
                'volatility_impact': -volatility_penalty,
                'data_quality_impact': data_quality_bonus,
                'time_impact': -time_penalty
            }
        }
        
        print(f"   [OK] Volatility-aware confidence: {final_confidence:.1f}%")
        print(f"   [CHART] Components: Signal={signal_component:.1f}, News={news_component:.1f}, Drift={drift_component:.1f}")
        print(f"   [CHART] Penalties: Vol={volatility_penalty:.1f}, Time={time_penalty:.1f}")
        print(f"   [CHART] Bonuses: Data Quality={data_quality_bonus:.1f}")
        
        return final_confidence / 100, confidence_results
    
    def add_catalyst_detection_system(self):
        """Add system to detect upcoming catalysts that could affect predictions"""
        
    def detect_upcoming_catalysts(self, ticker, prediction_days):
            """Detect upcoming events that could impact stock price"""
            print(f"🔍 Detecting upcoming catalysts for {ticker} over {prediction_days} days...")
            
            upcoming_catalysts = []
            
            # Check for earnings dates (you'd integrate with earnings calendar API)
            # This is a placeholder - integrate with actual earnings calendar
            catalyst_calendar = {
                'earnings_date': None,  # Would come from API
                'ex_dividend_date': None,
                'product_launches': [],
                'regulatory_deadlines': [],
            }
            
            # FOMC meetings and macro events (static calendar - update quarterly)
            import datetime
            current_date = datetime.datetime.now()
            
            # Example macro events
            macro_events = [
                {'date': '2025-01-29', 'event': 'FOMC Meeting', 'impact': 'high'},
                {'date': '2025-02-14', 'event': 'CPI Release', 'impact': 'medium'},
                # Add more based on actual calendar
            ]
            
            for event in macro_events:
                event_date = datetime.datetime.strptime(event['date'], '%Y-%m-%d')
                days_until = (event_date - current_date).days
                
                if 0 <= days_until <= prediction_days:
                    upcoming_catalysts.append({
                        'type': 'macro',
                        'event': event['event'],
                        'days_until': days_until,
                        'impact_level': event['impact'],
                        'direction_bias': 'neutral'  # Unless specified
                    })
            
            catalyst_impact = {
                'has_major_catalysts': len([c for c in upcoming_catalysts if c['impact_level'] == 'high']) > 0,
                'catalyst_count': len(upcoming_catalysts),
                'catalysts': upcoming_catalysts,
                'net_bias': 'neutral'  # Calculate based on catalyst analysis
            }
            
            print(f"   📅 Found {len(upcoming_catalysts)} upcoming catalysts")
            return catalyst_impact


    def analyze_options_scenarios(self, current_price, daily_stats, options_data):
        """Analyze options trading scenarios with FIXED expected returns"""
        print("[CHART] Analyzing options trading scenarios...")
        
        risk_free_rate = 0.05  # 5% risk-free rate
        scenarios = []
        
        for option in options_data:
            strike = option['strike']
            expiration_days = option['expiration_days']
            option_type = option['type']  # 'call' or 'put'
            current_iv = option.get('implied_volatility', 0.3)
            
            # Get price prediction for expiration day
            if expiration_days <= len(daily_stats):
                target_day_stats = daily_stats[expiration_days - 1]
            else:
                # Extrapolate for longer expirations
                target_day_stats = daily_stats[-1]
            
            # Current option value
            time_to_expiry = expiration_days / 365
            if option_type == 'call':
                current_option_value = self.options_analyzer.black_scholes_call(
                    current_price, strike, time_to_expiry, risk_free_rate, current_iv
                )
            else:
                current_option_value = self.options_analyzer.black_scholes_put(
                    current_price, strike, time_to_expiry, risk_free_rate, current_iv
                )
            
            # Calculate Greeks
            greeks = self.options_analyzer.calculate_greeks(
                current_price, strike, time_to_expiry, risk_free_rate, current_iv, option_type
            )
            
            # Calculate expected value using Monte Carlo results
            target_mean_price = target_day_stats['mean_price']
            
            # Expected payoff at expiration
            if option_type == 'call':
                expected_payoff = max(target_mean_price - strike, 0)
                prob_itm = target_day_stats['prob_above_current'] if strike <= current_price else target_day_stats['prob_above_current'] * 0.5
            else:
                expected_payoff = max(strike - target_mean_price, 0)
                prob_itm = 1 - target_day_stats['prob_above_current'] if strike >= current_price else (1 - target_day_stats['prob_above_current']) * 0.5
            
            # Expected profit/loss
            expected_pnl = expected_payoff - current_option_value
            expected_return_pct = (expected_pnl / current_option_value * 100) if current_option_value > 0 else 0
            
            # Calculate breakeven and probability of profit
            if option_type == 'call':
                breakeven = strike + current_option_value
                prob_profitable = target_day_stats['prob_above_current'] * (target_mean_price / breakeven) if breakeven > 0 else 0
            else:
                breakeven = strike - current_option_value
                prob_profitable = (1 - target_day_stats['prob_above_current']) * (breakeven / target_mean_price) if target_mean_price > 0 else 0
            
            prob_profitable = max(0, min(1, prob_profitable))  # Clamp to [0,1]
            
            # Risk metrics
            max_loss = current_option_value
            max_gain = float('inf') if option_type == 'call' else strike - current_option_value
            risk_reward_ratio = abs(expected_pnl / max_loss) if max_loss > 0 and expected_pnl > 0 else 0
            
            scenarios.append({
                'option_type': option_type,
                'strike': strike,
                'expiration_days': expiration_days,
                'current_option_value': current_option_value,
                'expected_payoff': expected_payoff,
                'expected_pnl': expected_pnl,
                'expected_return_pct': expected_return_pct,
                'max_loss': max_loss,
                'max_gain': max_gain if max_gain != float('inf') else None,
                'prob_profitable': prob_profitable,
                'prob_itm': prob_itm,
                'breakeven': breakeven,
                'greeks': greeks,
                'current_iv': current_iv,
                'target_price': target_mean_price,
                'risk_reward_ratio': risk_reward_ratio
            })
        
        print(f"   [OK] Analyzed {len(scenarios)} options scenarios")
        return scenarios

    def generate_options_recommendations(self, scenarios, market_sentiment, volatility_forecast):
        """Generate options trading recommendations based on scenarios"""
        print("[BULB] Generating options recommendations...")
        
        recommendations = []
        
        for scenario in scenarios:
            # Base score calculation
            score = 0
            confidence = 0.5
            
            # Profitability score
            if scenario['expected_return_pct'] > 10:
                score += 0.3
                confidence += 0.1
            elif scenario['expected_return_pct'] > 0:
                score += 0.1
            else:
                score -= 0.2
                confidence -= 0.1
            
            # Probability score
            if scenario['prob_profitable'] > 0.6:
                score += 0.2
                confidence += 0.1
            elif scenario['prob_profitable'] > 0.4:
                score += 0.1
            else:
                score -= 0.1
            
            # Risk-reward score
            if scenario['risk_reward_ratio'] > 2:
                score += 0.2
            elif scenario['risk_reward_ratio'] > 1:
                score += 0.1
            
            # Market sentiment alignment
            if market_sentiment > 0.2 and scenario['option_type'] == 'call':
                score += 0.2
                confidence += 0.1
            elif market_sentiment < -0.2 and scenario['option_type'] == 'put':
                score += 0.2
                confidence += 0.1
            elif abs(market_sentiment) < 0.1:
                score += 0.05  # Neutral markets favor both
            
            # Volatility considerations
            if volatility_forecast > 0.4:  # High volatility
                if scenario['option_type'] == 'call' and scenario['greeks']['delta'] > 0.5:
                    score += 0.1
                elif scenario['option_type'] == 'put' and scenario['greeks']['delta'] < -0.5:
                    score += 0.1
            
            # Time decay consideration
            if scenario['expiration_days'] < 7:
                score -= 0.15  # Penalize very short-term options
                confidence -= 0.1
            elif scenario['expiration_days'] > 90:
                score -= 0.05  # Slight penalty for long-term
            
            # Clamp values
            score = max(0, min(1, score))
            confidence = max(0.1, min(0.95, confidence))
            
            # Determine action
            if score > 0.6:
                action = "BUY"
            elif score > 0.4:
                action = "CONSIDER"
            else:
                action = "AVOID"
            
            recommendations.append({
                'action': action,
                'option_type': scenario['option_type'],
                'strike': scenario['strike'],
                'expiration_days': scenario['expiration_days'],
                'score': score,
                'confidence': confidence,
                'expected_return_pct': scenario['expected_return_pct'],
                'prob_profitable': scenario['prob_profitable'],
                'risk_reward_ratio': scenario['risk_reward_ratio'],
                'max_loss': scenario['max_loss'],
                'reasoning': self._generate_recommendation_reasoning(scenario, market_sentiment, score)
            })
        
        # Sort by score descending
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"   [OK] Generated {len(recommendations)} recommendations")
        return recommendations

    def _generate_recommendation_reasoning(self, scenario, market_sentiment, score):
        """Generate reasoning for recommendation"""
        reasons = []
        
        if scenario['expected_return_pct'] > 10:
            reasons.append(f"High expected return ({scenario['expected_return_pct']:.1f}%)")
        elif scenario['expected_return_pct'] > 0:
            reasons.append(f"Positive expected return ({scenario['expected_return_pct']:.1f}%)")
        else:
            reasons.append(f"Negative expected return ({scenario['expected_return_pct']:.1f}%)")
        
        if scenario['prob_profitable'] > 0.6:
            reasons.append(f"High probability of profit ({scenario['prob_profitable']:.1%})")
        elif scenario['prob_profitable'] < 0.4:
            reasons.append(f"Low probability of profit ({scenario['prob_profitable']:.1%})")
        
        if scenario['risk_reward_ratio'] > 2:
            reasons.append("Excellent risk-reward ratio")
        elif scenario['risk_reward_ratio'] > 1:
            reasons.append("Good risk-reward ratio")
        
        sentiment_direction = "bullish" if market_sentiment > 0.1 else "bearish" if market_sentiment < -0.1 else "neutral"
        if (market_sentiment > 0.2 and scenario['option_type'] == 'call') or (market_sentiment < -0.2 and scenario['option_type'] == 'put'):
            reasons.append(f"Aligns with {sentiment_direction} market sentiment")
        
        return "; ".join(reasons)
    
    def calculate_daily_stats_from_paths_enhanced(self, price_paths, current_price, prediction_days, volatility, ticker):
        """Enhanced daily statistics with MAE-enforced range calibration"""
        print("[CHART] Calculating enhanced daily statistics with MAE enforcement...")
        
        daily_stats = []
        
        for day in range(1, prediction_days + 1):
            day_prices = price_paths[:, day]
            mean_price = np.mean(day_prices)
            
            # Use NEW MAE-enforced range calibration
            enhanced_low, enhanced_high, range_results = self.enhanced_range_calibration(
                mean_price, current_price, volatility, ticker, day
            )
            
            # Calculate actual MAE achieved
            actual_mae_pct = (enhanced_high - enhanced_low) / (2 * mean_price) * 100
            target_mae_pct = range_results.get('mae_target_pct', 2.5)
            mae_target_met = range_results.get('mae_target_met', False)
            volatility_tier = range_results.get('volatility_tier', 'unknown')
            
            # Determine MAE quality
            if actual_mae_pct <= target_mae_pct:
                mae_quality = "Excellent"
            elif actual_mae_pct <= target_mae_pct * 1.1:
                mae_quality = "Good"
            elif actual_mae_pct <= target_mae_pct * 1.2:
                mae_quality = "Acceptable"
            else:
                mae_quality = "Needs Improvement"
            
            daily_stats.append({
                'day': day,
                'mean_price': mean_price,
                'median_price': np.median(day_prices),
                'std_price': np.std(day_prices),
                'min_price': enhanced_low,  # MAE-enforced range
                'max_price': enhanced_high, # MAE-enforced range
                'percentile_5': np.percentile(day_prices, 5),
                'percentile_25': np.percentile(day_prices, 25),
                'percentile_75': np.percentile(day_prices, 75),
                'percentile_95': np.percentile(day_prices, 95),
                'prob_above_current': np.mean(day_prices > current_price),
                'expected_return': (mean_price - current_price) / current_price,
                
                # Enhanced MAE metrics
                'mae_percent': actual_mae_pct,
                'mae_target': target_mae_pct,
                'mae_quality': mae_quality,
                'mae_target_met': mae_target_met,
                'volatility_tier': volatility_tier,
                'range_width_pct': ((enhanced_high - enhanced_low) / mean_price) * 100,
                'range_tightening_applied': range_results.get('range_tightening_pct', 0),
                
                # Calibration details
                'mae_enforcement_applied': True,
                'enhanced_calibration_applied': True,
                'range_calibration_results': range_results,
                'base_multiplier_used': range_results.get('base_multiplier', 2.2),
                'compression_factor': range_results.get('compression_applied', {}).get('total_compression', 1.0)
            })
        
        # Summary logging with MAE performance
        mae_success_count = sum(1 for stat in daily_stats if stat['mae_target_met'])
        total_days = len(daily_stats)
        mae_success_rate = (mae_success_count / total_days * 100) if total_days > 0 else 0
        
        avg_mae = np.mean([stat['mae_percent'] for stat in daily_stats])
        avg_target = np.mean([stat['mae_target'] for stat in daily_stats])
        avg_tightening = np.mean([stat['range_tightening_applied'] for stat in daily_stats])
        
        print(f"   [OK] MAE enforcement complete: {mae_success_count}/{total_days} days meet target ({mae_success_rate:.1f}%)")
        print(f"   [CHART] Average MAE: {avg_mae:.1f}% vs Target: {avg_target:.1f}%")
        print(f"   [CHART] Average range tightening: {avg_tightening:.1f}%")
        print(f"   [TARGET] Volatility tier: {volatility_tier}")
        
        return daily_stats

    def create_fallback_analysis(self, ticker, excel_data, news_data, market_data, prediction_days, graph_analysis):
        """Create enhanced fallback analysis with fresh search awareness"""
        print("🔄 Creating enhanced fallback analysis...")
        
        current_price = market_data['current_price']
        volatility = excel_data.get('volatility', 25.0)
        thirty_day_return = excel_data.get('performance_return_1_month', 0.0)
        
        # Enhanced expected return calculation with fresh search consideration
        daily_return = thirty_day_return / 30
        
        # Apply fresh search boost if available
        fresh_search_boost = 0.0
        if news_data and news_data.get('fresh_search_performed', False):
            fresh_sentiment = news_data.get('sentiment_1d', 0.0)
            fresh_signal_strength = news_data.get('signal_strength', 0.0)
            fresh_search_boost = fresh_sentiment * fresh_signal_strength * 0.5
            print(f"   🔄 Fresh search boost applied: {fresh_search_boost:+.3f}")
        
        # Calculate expected return with fresh search enhancement
        base_expected_return = daily_return * prediction_days
        enhanced_expected_return = base_expected_return + (fresh_search_boost * prediction_days)
        
        # Enhanced confidence calculation
        base_confidence = 0.6
        if news_data and news_data.get('fresh_search_performed', False):
            fresh_confidence_boost = min(0.15, news_data.get('quality_score', 0.0) * 0.2)
            base_confidence += fresh_confidence_boost
            print(f"   🔄 Fresh search confidence boost: +{fresh_confidence_boost:.3f}")
        
        # Enhanced volatility forecast with fresh search consideration
        base_volatility_forecast = volatility / 100
        if news_data and news_data.get('fresh_search_performed', False):
            # Fresh news can increase short-term volatility
            volatility_adjustment = min(0.05, news_data.get('fresh_articles_found', 0) / 200)
            base_volatility_forecast += volatility_adjustment
            print(f"   🔄 Volatility adjustment for fresh news: +{volatility_adjustment:.3f}")
        
        # Price predictions with enhanced methodology
        predictions_by_day = []
        for day in range(1, prediction_days + 1):
            day_return = (daily_return * day) + (fresh_search_boost * day)
            predicted_price = current_price * (1 + day_return / 100)
            
            # Enhanced intraday range calculation
            daily_vol = (volatility / 100) / math.sqrt(252)
            intraday_range = predicted_price * daily_vol * 1.5
            
            # Apply fresh search volatility if available
            if news_data and news_data.get('fresh_search_performed', False):
                fresh_vol_factor = 1.0 + (news_data.get('signal_strength', 0.0) * 0.2)
                intraday_range *= fresh_vol_factor
            
            predictions_by_day.append({
                "day": day,
                "predicted_open": predicted_price * 0.999,
                "predicted_high": predicted_price + intraday_range,
                "predicted_low": predicted_price - intraday_range,
                "predicted_close": predicted_price
            })
        
        # Enhanced key factors with fresh search context
        key_factors = ["Historical momentum", "Market volatility"]
        
        if news_data and news_data.get('fresh_search_performed', False):
            key_factors.extend([
                f"Fresh search: {news_data.get('fresh_articles_found', 0)} articles",
                f"News sentiment: {news_data.get('sentiment_1d', 0.0):+.3f}"
            ])
            if news_data.get('prediction_ready', False):
                key_factors.append("High-quality news coverage")
        
        if news_data and news_data.get('has_major_catalyst', False):
            key_factors.append("Major catalyst detected")
        
        # Enhanced direction calculation
        if enhanced_expected_return > 1:
            direction = "up"
            probability_up = 0.6 + min(0.2, fresh_search_boost * 2)
        elif enhanced_expected_return < -1:
            direction = "down"
            probability_up = 0.4 - min(0.2, abs(fresh_search_boost) * 2)
        else:
            direction = "sideways"
            probability_up = 0.5 + (fresh_search_boost * 0.5)
        
        # Clamp probability
        probability_up = max(0.2, min(0.8, probability_up))
        
        # Enhanced options sentiment
        options_sentiment = "neutral"
        if news_data and news_data.get('fresh_search_performed', False):
            fresh_sentiment = news_data.get('sentiment_1d', 0.0)
            if fresh_sentiment > 0.15:
                options_sentiment = "bullish"
            elif fresh_sentiment < -0.15:
                options_sentiment = "bearish"
        
        # Enhanced risk assessment
        risk_level = "high" if volatility > 50 else "moderate" if volatility > 25 else "low"
        
        # Increase risk if fresh search shows conflicting signals
        if news_data and news_data.get('fresh_search_performed', False):
            if abs(news_data.get('sentiment_1d', 0.0)) > 0.3 and volatility > 40:
                risk_level = "high"
        
        # Enhanced mathematical basis
        mathematical_basis = f"Enhanced fallback: {thirty_day_return:.1f}% 30-day return"
        if news_data and news_data.get('fresh_search_performed', False):
            mathematical_basis += f" + fresh search boost ({fresh_search_boost:+.3f})"
        
        # Enhanced reasoning
        reasoning = f"Enhanced fallback analysis for {prediction_days} day period"
        if news_data and news_data.get('fresh_search_performed', False):
            reasoning += f" with {news_data.get('fresh_articles_found', 0)} fresh articles"
        
        fallback_analysis = {
            "predictions_by_day": predictions_by_day,
            "final_target_price": current_price * (1 + enhanced_expected_return / 100),
            "total_expected_return_pct": enhanced_expected_return,
            "confidence": base_confidence,
            "direction": direction,
            "probability_up": probability_up,
            "volatility_forecast": base_volatility_forecast,
            "options_sentiment": options_sentiment,
            "key_factors": key_factors,
            "risk_assessment": risk_level,
            "mathematical_basis": mathematical_basis,
            "reasoning": reasoning,
            
            # Fresh search metadata
            "fresh_search_integration": True,
            "fresh_search_applied": news_data.get('fresh_search_performed', False) if news_data else False,
            "fresh_search_metadata": {
                "performed": news_data.get('fresh_search_performed', False) if news_data else False,
                "articles_found": news_data.get('fresh_articles_found', 0) if news_data else 0,
                "sentiment_score": news_data.get('sentiment_1d', 0.0) if news_data else 0.0,
                "signal_strength": news_data.get('signal_strength', 0.0) if news_data else 0.0,
                "quality_score": news_data.get('quality_score', 0.0) if news_data else 0.0,
                "boost_applied": fresh_search_boost
            },
            
            # Enhancement flags
            "fallback_enhancements": {
                "fresh_search_boost": fresh_search_boost,
                "confidence_boost": fresh_confidence_boost if news_data and news_data.get('fresh_search_performed', False) else 0.0,
                "volatility_adjustment": volatility_adjustment if news_data and news_data.get('fresh_search_performed', False) else 0.0,
                "enhanced_risk_assessment": True,
                "enhanced_probability_calculation": True
            }
        }
        
        print(f"[OK] Enhanced fallback analysis complete")
        print(f"   [CHART] Expected return: {enhanced_expected_return:+.2f}% (base: {base_expected_return:+.2f}%)")
        print(f"   [TARGET] Confidence: {base_confidence:.3f}")
        print(f"   🔄 Fresh search integrated: {news_data.get('fresh_search_performed', False) if news_data else False}")
        
        return fallback_analysis
        
    def calculate_recent_momentum(self, excel_data):
        """Calculate recent 5-day momentum if available"""
        try:
            # Try to get 5-day return from Excel data
            if 'recent_5d_return' in excel_data:
                return excel_data['recent_5d_return']
            
            # Fallback: estimate from daily changes
            if 'avg_daily_change' in excel_data:
                return excel_data['avg_daily_change'] * 5
            
            # Default fallback
            return 0.0
            
        except Exception:
            return 0.0

    def detect_enhanced_financial_catalysts(self, article_content, ticker):
        """Enhanced catalyst detection with financial verbs"""
        
        # Expanded catalyst patterns with financial verbs
        catalyst_patterns = {
            'earnings_beats': [
                'beats earnings', 'exceeds eps', 'earnings surprise', 'topped estimates',
                'beat expectations', 'earnings beat', 'outperformed estimates', 'smashed estimates'
            ],
            'guidance_upgrades': [
                'raises guidance', 'upgrades outlook', 'increases forecast', 'boosts guidance',
                'raised full-year', 'upgraded target', 'positive guidance', 'upward revision'
            ],
            'analyst_upgrades': [
                'upgraded to buy', 'raised price target', 'analyst upgrade', 'upgrades rating',
                'overweight rating', 'outperform rating', 'buy rating initiated', 'target raised'
            ],
            'major_contracts': [
                'billion dollar contract', 'major deal', 'significant contract', 'partnership agreement',
                'strategic alliance', 'acquisition deal', 'merger agreement', 'landmark deal'
            ],
            'regulatory_approvals': [
                'fda approval', 'regulatory clearance', 'patent approval', 'license granted',
                'government contract', 'regulatory win', 'approval granted'
            ]
        }
        
        # Financial action verbs indicating market-moving news
        financial_verbs = [
            'reports', 'announces', 'declares', 'beats', 'exceeds', 'raises', 'upgrades',
            'launches', 'acquires', 'merges', 'partners', 'signs', 'wins', 'secures',
            'delivers', 'achieves', 'surpasses', 'outperforms'
        ]
        
        content_lower = article_content.lower()
        ticker_lower = ticker.lower()
        
        catalyst_score = 0
        detected_catalysts = []
        has_financial_verb = False
        
        # Check for financial verbs + ticker combination (stronger signal)
        for verb in financial_verbs:
            if verb in content_lower and ticker_lower in content_lower:
                has_financial_verb = True
                catalyst_score += 0.3  # Increased from 0.2
                break
        
        # Check for specific catalyst patterns
        for catalyst_type, patterns in catalyst_patterns.items():
            for pattern in patterns:
                if pattern in content_lower:
                    detected_catalysts.append(catalyst_type)
                    catalyst_score += 0.5  # Increased from 0.4
                    break
        
        # Enhanced scoring for multiple indicators
        if len(detected_catalysts) > 1:
            catalyst_score *= 1.8  # Increased from 1.5
        
        # Extra boost for earnings-related content
        if any('earnings' in cat for cat in detected_catalysts):
            catalyst_score *= 1.5  # Increased from 1.3
        
        return {
        'catalyst_score': min(catalyst_score, 2.5),  # Increased cap from 2.0
        'detected_catalysts': detected_catalysts,
        'has_financial_verb': has_financial_verb,
        'catalyst_strength': 'strong' if catalyst_score > 1.0 else 'moderate' if catalyst_score > 0.5 else 'weak',
        
        # ADD THESE LINES FOR BACKWARD COMPATIBILITY:
        'has_major_catalyst': catalyst_score > 0.8,  # Convert to boolean
        'catalyst_types': detected_catalysts,        # Alias for detected_catalysts
        'catalyst_count': len(detected_catalysts),   # Count of catalysts
        'impact_multiplier': 1.0 + (catalyst_score / 2.0)  # Convert score to multiplier
    }

    def analyze_enhanced_sentiment_momentum(self, article_content, ticker):
        """Enhanced sentiment analysis with stronger momentum detection"""
        
        # Enhanced momentum indicators with stronger weights
        momentum_indicators = {
            'explosive_bullish': {
                'patterns': ['soaring', 'surging', 'skyrocketing', 'explosive', 'breakout', 'moonshot'],
                'weight': 1.0
            },
            'strong_bullish': {
                'patterns': ['surges', 'rallies', 'climbs', 'jumps', 'spikes', 'all-time high'],
                'weight': 0.8
            },
            'moderate_bullish': {
                'patterns': ['rising', 'growing', 'gaining', 'advancing', 'upward trend'],
                'weight': 0.5
            },
            'explosive_bearish': {
                'patterns': ['plummeting', 'crashing', 'collapsing', 'nosediving', 'free fall'],
                'weight': -1.0
            },
            'strong_bearish': {
                'patterns': ['tumbling', 'plunging', 'sliding', 'dropping sharply'],
                'weight': -0.8
            },
            'moderate_bearish': {
                'patterns': ['declining', 'falling', 'retreating', 'pulling back'],
                'weight': -0.5
            }
        }
        
        content_lower = article_content.lower()
        momentum_score = 0
        detected_patterns = []
        
        for category, data in momentum_indicators.items():
            for pattern in data['patterns']:
                if pattern in content_lower:
                    momentum_score += data['weight']
                    detected_patterns.append(pattern)
                    break  # Only count once per category
        
        # Normalize momentum score with enhanced range
        momentum_score = max(-1.2, min(1.2, momentum_score))
        
        return {
            'momentum_score': momentum_score,
            'detected_patterns': detected_patterns,
            'momentum_direction': 'bullish' if momentum_score > 0.3 else 'bearish' if momentum_score < -0.3 else 'neutral'
        }

    def generate_enhanced_price_paths(self, current_price, volatility, prediction_days, enhanced_drift, overall_signal, num_paths=1000):
        """Generate price paths with log-normal returns and enhanced features"""
        print(f"🎲 Generating enhanced price paths with log-normal returns...")
        
        # Convert to log-normal parameters
        annual_vol_decimal = min(volatility, 100.0) / 100
        daily_vol = annual_vol_decimal / math.sqrt(252)
        
        # Enhanced drift with stronger signal adjustment
        annual_drift_decimal = enhanced_drift / 100
        daily_drift = annual_drift_decimal / 252
        
        # Enhanced signal-based drift adjustment
        if prediction_days <= 5:
            if overall_signal < -0.15:
                signal_drift_adjustment = -0.0015  # Stronger bearish bias
                drift_multiplier = 0.05  # Heavily reduced bullish drift
            elif overall_signal < -0.05:
                signal_drift_adjustment = -0.0008
                drift_multiplier = 0.2
            elif overall_signal > 0.15:
                signal_drift_adjustment = 0.0012  # Enhanced bullish bias
                drift_multiplier = 1.2  # Amplified bullish drift
            elif overall_signal > 0.05:
                signal_drift_adjustment = 0.0006
                drift_multiplier = 0.9
            else:
                signal_drift_adjustment = 0
                drift_multiplier = 0.6
        else:
            signal_drift_adjustment = overall_signal * 0.0003  # Increased from 0.0002
            drift_multiplier = 0.9  # Increased from 0.8
        
        adjusted_daily_drift = (daily_drift * drift_multiplier) + signal_drift_adjustment
        
        print(f"   [CHART] Enhanced daily drift: {adjusted_daily_drift:.6f}, volatility: {daily_vol:.4f}")
        
        # Generate paths using log-normal distribution with enhanced mean reversion
        price_paths = []
        for path_num in range(num_paths):
            log_price = math.log(current_price)
            path = [current_price]
            
            for day in range(prediction_days):
                current_price_path = math.exp(log_price)
                distance_from_start = (current_price_path - current_price) / current_price
                
                # Enhanced signal-aware mean reversion
                if overall_signal > 0.1:  # Bullish - reduced mean reversion
                    mean_reversion_strength = 0.03
                elif overall_signal < -0.1:  # Bearish - increased mean reversion
                    mean_reversion_strength = 0.18
                else:
                    mean_reversion_strength = 0.10
                
                # Progressive mean reversion based on distance
                if abs(distance_from_start) > 0.15:
                    mean_reversion = -mean_reversion_strength * 1.5 * distance_from_start
                elif abs(distance_from_start) > 0.08:
                    mean_reversion = -mean_reversion_strength * distance_from_start
                elif abs(distance_from_start) > 0.04:
                    mean_reversion = -mean_reversion_strength * 0.5 * distance_from_start
                else:
                    mean_reversion = 0
                
                # Enhanced momentum component
                if prediction_days <= 5:
                    momentum_component = overall_signal * 0.0008  # Increased impact
                else:
                    momentum_component = overall_signal * 0.0004
                
                # Log-normal random shock with momentum
                random_shock = np.random.normal(
                    adjusted_daily_drift + mean_reversion + momentum_component, 
                    daily_vol
                )
                
                # Update log price
                log_price += random_shock
                
                # Convert back to price
                new_price = math.exp(log_price)
                new_price = max(0.01, new_price)
                
                path.append(new_price)
            
            price_paths.append(path)
        
        return np.array(price_paths)

    def analyze_with_claude_ultimate_enhanced(self, ticker, excel_data, news_data, market_data, custom_articles, graph_analysis=None, prediction_days=1, options_data=None):        
        """Enhanced Claude analysis with VOLATILITY-AWARE confidence integration AND GRAPH ANALYSIS - UPDATED VERSION"""
        print(f"🧮 Running ENHANCED Claude AI analysis with VOLATILITY-AWARE confidence and GRAPH ANALYSIS for {prediction_days} day(s)...")
        
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.claude_key)
        except ImportError:
            print("[ERROR] Anthropic library not installed")
            return self.create_fallback_analysis(ticker, excel_data, news_data, market_data, prediction_days, graph_analysis)
        except AttributeError:
            print("[ERROR] Claude API key not found - check self.claude_key")
            return self.create_fallback_analysis(ticker, excel_data, news_data, market_data, prediction_days, graph_analysis)
        
        # STEP 1: Get volatility and current price with error handling
        try:
            volatility = excel_data.get('volatility', 25.0)
            current_price = market_data['current_price']
            thirty_day_return = excel_data.get('performance_return_1_month', 0.0)
        except KeyError as e:
            print(f"[ERROR] Missing required data: {e}")
            return self.create_fallback_analysis(ticker, excel_data, news_data, market_data, prediction_days, graph_analysis)
        
        print(f"   [CHART] Volatility: {volatility:.1f}% → Applying volatility-aware confidence system")
        if graph_analysis:
            print(f"   [UP] Graph Analysis: [OK] Available → Integrating technical patterns")
        
        # STEP 2: Enhanced drift scaling with fresh search awareness
        base_annualized_drift = ((1 + thirty_day_return/100) ** 12 - 1) * 100
        momentum_score = thirty_day_return / 100 / 12
        news_momentum = news_data.get('sentiment_1d', 0.0) if news_data else 0.0
        recent_performance = thirty_day_return
        
        # Boost drift if fresh search found strong signals
        if news_data and news_data.get('fresh_search_performed', False):
            fresh_boost = news_data.get('signal_strength', 0.0) * 0.5
            print(f"   🔄 Fresh search boost applied: +{fresh_boost:.3f}")
            news_momentum += fresh_boost
        
        # NEW: Boost drift if graph analysis shows strong momentum acceleration
        if graph_analysis:
            graph_momentum_acceleration = graph_analysis.get('momentum_analysis', {}).get('momentum_acceleration', 0.5)
            if graph_momentum_acceleration > 0.75:
                momentum_graph_boost = (graph_momentum_acceleration - 0.5) * 0.3
                news_momentum += momentum_graph_boost
                print(f"   [UP] Graph momentum boost applied: +{momentum_graph_boost:.3f} from acceleration {graph_momentum_acceleration:.3f}")
        
        enhanced_drift, drift_results = self.enhanced_drift_scaling(
            base_annualized_drift, momentum_score, volatility, news_momentum, recent_performance
        )
        
        # STEP 3: Get preliminary signal for drift adjustment WITH GRAPH ANALYSIS
        try:
            preliminary_analysis = self.perform_mathematical_analysis(
                excel_data, market_data, custom_articles, prediction_days, 
                np.array([[current_price]]), graph_analysis
            )
            overall_signal = preliminary_analysis['composite_scores']['overall_signal']
            
            # Apply fresh search signal boost
            if news_data and news_data.get('fresh_search_performed', False):
                fresh_signal_boost = news_data.get('signal_direction', 0.0) * 0.3
                overall_signal += fresh_signal_boost
                overall_signal = max(-1, min(1, overall_signal))  # Clamp to [-1, 1]
                print(f"   🔄 Fresh search signal boost: +{fresh_signal_boost:.3f}, final: {overall_signal:+.3f}")
            
        except Exception as e:
            print(f"   [WARNING] Mathematical analysis failed: {e}, using fallback signal calculation")
            overall_signal = 0.0
        
        # STEP 4: Generate paths with enhanced drift (VOLATILITY-AWARE)
        try:
            price_paths = self.generate_enhanced_price_paths(
                current_price, volatility, prediction_days, enhanced_drift, overall_signal
            )
        except Exception as e:
            print(f"   [WARNING] Enhanced price paths failed: {e}, using fallback Monte Carlo")
            try:
                # Use our NEW volatility-aware Monte Carlo method
                price_paths = self.generate_multi_day_price_paths(
                    current_price, volatility, prediction_days, 
                    drift=enhanced_drift, overall_signal=overall_signal
                )
                print("   [OK] Using volatility-aware Monte Carlo price path generation")
            except Exception as e2:
                print(f"   [ERROR] All price path generation failed: {e2}")
                return self.create_fallback_analysis(ticker, excel_data, news_data, market_data, prediction_days, graph_analysis)
        
        # Log Monte Carlo path count for transparency
        monte_carlo_paths = len(price_paths) if hasattr(price_paths, '__len__') and len(price_paths) > 0 else 1000
        print(f"   [CHART] Monte Carlo paths: {monte_carlo_paths:,} (volatility-adjusted)")
        
        # STEP 5: Calculate daily stats with enhanced range calibration
        try:
            daily_stats = self.calculate_daily_stats_from_paths_enhanced(
                price_paths, current_price, prediction_days, volatility, ticker
            )
        except Exception as e:
            print(f"   [ERROR] Daily stats calculation failed: {e}")
            return self.create_fallback_analysis(ticker, excel_data, news_data, market_data, prediction_days, graph_analysis)
        
        # STEP 6: Final analysis with VOLATILITY-AWARE CONFIDENCE AND GRAPH ANALYSIS
        try:
            math_analysis = self.perform_mathematical_analysis(
                excel_data, market_data, custom_articles, prediction_days, price_paths, graph_analysis
            )
        except Exception as e:
            print(f"   [ERROR] Final mathematical analysis failed: {e}")
            return self.create_fallback_analysis(ticker, excel_data, news_data, market_data, prediction_days, graph_analysis)
        
        # REMOVED OLD CONFIDENCE CALCULATION - Now using the one from perform_mathematical_analysis
        # The enhanced confidence is already calculated in math_analysis with volatility awareness
        enhanced_confidence = math_analysis.get('enhanced_confidence', 0.85)
        confidence_results = math_analysis.get('confidence_breakdown', {})
        
        print(f"   [TARGET] VOLATILITY-AWARE Confidence: {enhanced_confidence:.1%}")
        print(f"   [CHART] Volatility tier: {confidence_results.get('volatility_tier', 'Unknown')}")
        print(f"   [CHART] Confidence factors: Signal={confidence_results.get('signal_component', 0):.1f}, Vol penalty={confidence_results.get('volatility_penalty', 0):.1f}")
        
        # STEP 7: Options analysis if options data provided
        options_analysis = None
        recommendations = []
        options_summary = ""
        
        if options_data:
            try:
                scenarios = self.analyze_options_scenarios(current_price, daily_stats, options_data)
                market_sentiment = math_analysis['composite_scores']['overall_signal']
                volatility_forecast = volatility / 100
                recommendations = self.generate_options_recommendations(scenarios, market_sentiment, volatility_forecast)
                options_analysis = {
                    'scenarios': scenarios,
                    'recommendations': recommendations
                }
                
                options_summary = f"\n\n═══ OPTIONS ANALYSIS ═══\nTop Recommendations:\n"
                for rec in recommendations[:3]:
                    options_summary += f"• {rec['action']} {rec['option_type'].upper()} ${rec['strike']} ({rec['expiration_days']}d): {rec['confidence']:.0%} confidence, {rec['expected_return_pct']:+.1f}% expected return\n"
            except Exception as e:
                print(f"   [WARNING] Options analysis failed: {e}")
        
        # STEP 8: Prepare enhanced prompt with volatility-aware context AND GRAPH ANALYSIS
        overall_signal = math_analysis['composite_scores']['overall_signal']
        expected_return = math_analysis['statistical_metrics']['expected_return_pct']
        
        # Create price predictions summary
        price_predictions_summary = ""
        for i, day_stat in enumerate(daily_stats[:min(5, len(daily_stats))], 1):
            price_predictions_summary += f"\nDay {i}: Mean ${day_stat['mean_price']:.2f}, Range ${day_stat['min_price']:.2f}-${day_stat['max_price']:.2f}, Prob Up: {day_stat['prob_above_current']:.1%}"
        
        # Fresh search summary
        fresh_search_summary = ""
        if news_data and news_data.get('fresh_search_performed', False):
            fresh_search_summary = f"""
FRESH SEARCH RESULTS:
- Articles Found: {news_data.get('fresh_articles_found', 0)}
- Sentiment Score: {news_data.get('sentiment_1d', 0.0):+.3f}
- Signal Strength: {news_data.get('signal_strength', 0.0):.3f}
- Prediction Ready: {news_data.get('prediction_ready', False)}
- Quality Score: {news_data.get('quality_score', 0.0):.3f}
- News-Price Correlation: {news_data.get('news_price_correlation', 0.0):+.3f}
"""
        else:
            fresh_search_summary = "FRESH SEARCH: Not performed (using existing database)"
        
        # NEW: Graph Analysis Summary
        graph_analysis_summary = ""
        primary_pattern = 'None'
        pattern_reliability = 0.0
        breakout_detected = False
        breakout_direction = 'neutral'
        breakout_strength = 0.0
        momentum_acceleration = 0.5
        candlestick_patterns = 0
        nearest_support = 0.0
        nearest_resistance = 0.0
        
        if graph_analysis:
            primary_pattern = graph_analysis.get('pattern_detected', {}).get('primary_pattern', 'None')
            pattern_reliability = graph_analysis.get('pattern_detected', {}).get('pattern_reliability', 0.0)
            breakout_detected = graph_analysis.get('breakout_analysis', {}).get('breakout_detected', False)
            breakout_direction = graph_analysis.get('breakout_analysis', {}).get('breakout_direction', 'neutral')
            breakout_strength = graph_analysis.get('breakout_analysis', {}).get('breakout_strength', 0.0)
            momentum_acceleration = graph_analysis.get('momentum_analysis', {}).get('momentum_acceleration', 0.5)
            candlestick_patterns = graph_analysis.get('candlestick_analysis', {}).get('total_patterns_found', 0)
            nearest_support = graph_analysis.get('support_resistance', {}).get('nearest_support', 0.0)
            nearest_resistance = graph_analysis.get('support_resistance', {}).get('nearest_resistance', 0.0)
            
            graph_analysis_summary = f"""
GRAPH ANALYSIS RESULTS (30-DAY TECHNICAL PATTERNS):
- Primary Chart Pattern: {primary_pattern} (Reliability: {pattern_reliability:.1%})
- Breakout Status: {'[OK] ' + breakout_direction.upper() + ' BREAKOUT' if breakout_detected else '[ERROR] No breakout'} (Strength: {breakout_strength:.1%})
- Momentum Acceleration: {momentum_acceleration:.3f} ({'Strong' if momentum_acceleration > 0.7 else 'Weak' if momentum_acceleration < 0.3 else 'Moderate'})
- Candlestick Patterns: {candlestick_patterns} patterns detected
- Support Level: ${nearest_support:.2f} | Resistance Level: ${nearest_resistance:.2f}
- Graph Technical Alignment: {news_data.get('graph_technical_alignment', 'neutral') if news_data and news_data.get('graph_analysis_integrated') else 'N/A'}
"""
        else:
            graph_analysis_summary = "GRAPH ANALYSIS: Not available (using fundamental analysis only)"
        
        # Get prediction timeframe and other enhanced metrics
        prediction_timeframe = math_analysis['composite_scores'].get('prediction_timeframe', 
                            'short' if prediction_days <= 5 else 'medium' if prediction_days <= 20 else 'long')
        
        bearish_protection = math_analysis['composite_scores'].get('bearish_protection_applied', False)
        drift_adjusted = math_analysis['statistical_metrics'].get('drift_adjustment_applied', False)
        
        # ENHANCED: Add volatility-aware confidence details to prompt
        volatility_tier = confidence_results.get('volatility_tier', 'Unknown')
        volatility_penalty = confidence_results.get('volatility_penalty', 0)
        data_quality_bonus = confidence_results.get('data_quality_bonus', 0)
        
        # NEW: Extract graph integration results for prompt
        graph_integration_applied = False
        graph_boost_summary = ""
        if graph_analysis and math_analysis.get('graph_integration_results', {}).get('graph_analysis_applied', False):
            graph_integration_applied = True
            integration_results = math_analysis['graph_integration_results']
            
            graph_boost_summary = f"""
GRAPH INTEGRATION APPLIED:
- Momentum Boost: {'[OK]' if integration_results.get('momentum_boost_applied', False) else '[ERROR]'}
- Technical Boost: {'[OK]' if integration_results.get('technical_boost_applied', False) else '[ERROR]'}  
- Breakout Override: {'[OK]' if integration_results.get('breakout_override_applied', False) else '[ERROR]'}
- Pattern Signal: {'[OK]' if integration_results.get('pattern_signal_applied', False) else '[ERROR]'}
- Regime Override: {'[OK] ' + str(graph_analysis.get('integration_signals', {}).get('regime_override_signal', '')) if integration_results.get('regime_override_applied', False) else '[ERROR]'}
- Confidence Boost: {'[OK]' if integration_results.get('confidence_boost_applied', False) else '[ERROR]'}
"""
        else:
            graph_boost_summary = "GRAPH INTEGRATION: Not applied (graph analysis not available or failed)"
        
        # Create comprehensive prompt with volatility-aware confidence context AND GRAPH ANALYSIS
        prompt = f"""ENHANCED ANALYSIS with VOLATILITY-AWARE CONFIDENCE and GRAPH ANALYSIS for {ticker} ({prediction_days} days):

    {fresh_search_summary}
    {graph_analysis_summary}

MATHEMATICAL FOUNDATION:
Signal: {overall_signal:+.3f} (Timeframe: {prediction_timeframe})
Expected Return: {expected_return:+.2f}% (Drift-adjusted: {drift_adjusted})
Volatility: {volatility:.1f}% (Tier: {volatility_tier})
30-Day Historical Return: {thirty_day_return:+.2f}%
VOLATILITY-AWARE Confidence: {enhanced_confidence:.3f}

CONFIDENCE BREAKDOWN:
- Volatility Tier: {volatility_tier} → Confidence range adjusted
- Volatility Penalty: -{volatility_penalty:.1f} points
- Data Quality Bonus: +{data_quality_bonus:.1f} points (from {monte_carlo_paths:,} Monte Carlo paths)
- Signal Strength Impact: {confidence_results.get('signal_component', 0):.1f} points
- Final Confidence: {enhanced_confidence:.1%} (volatility-adjusted)

SIGNAL BREAKDOWN:
- Momentum: {math_analysis['composite_scores']['momentum_component']:+.3f}
- Technical: {math_analysis['composite_scores']['technical_component']:+.3f} 
- News: {math_analysis['composite_scores']['news_component']:+.3f}
- Fresh Search Impact: {(news_data.get('signal_direction', 0.0) if news_data else 0.0):+.3f}
- Graph Analysis Impact: {math_analysis['composite_scores'].get('graph_signal_adjustment', 0.0):+.3f}
- Bearish Protection: {bearish_protection}

{graph_boost_summary}

MONTE CARLO PROJECTIONS ({prediction_days} days, {monte_carlo_paths:,} paths):{price_predictions_summary}

Final Expected Price: ${daily_stats[-1]['mean_price']:.2f} ({daily_stats[-1]['expected_return']*100:+.2f}%)
Probability of Gain: {daily_stats[-1]['prob_above_current']:.1%}
{options_summary}

VOLATILITY-AWARE SYSTEM: [OK] Applied (Tier: {volatility_tier}, Penalty: -{volatility_penalty:.1f})
FRESH SEARCH INTEGRATION: {"[OK] Applied" if news_data and news_data.get('fresh_search_performed', False) else "[ERROR] Not Available"}
GRAPH ANALYSIS INTEGRATION: {"[OK] Applied" if graph_integration_applied else "[ERROR] Not Available"}

Based on the ENHANCED analysis with GRAPH ANALYSIS showing {overall_signal:+.3f} signal, {expected_return:+.2f}% expected return, and {enhanced_confidence:.1%} confidence (adjusted for {volatility:.1f}% volatility), provide a JSON prediction that MATCHES these calculations:

{{
   "predictions_by_day": [
      {{"day": 1, "predicted_open": {daily_stats[0]['mean_price']:.2f}, "predicted_high": {daily_stats[0]['max_price']:.2f}, "predicted_low": {daily_stats[0]['min_price']:.2f}, "predicted_close": {daily_stats[0]['mean_price']:.2f}}},
      // ... for each day up to {prediction_days} with REALISTIC ranges
   ],
   "final_target_price": {daily_stats[-1]['mean_price']:.2f},
   "total_expected_return_pct": {expected_return:.2f},
   "confidence": {enhanced_confidence:.3f},
   "direction": "{('up' if expected_return > 0.5 else 'down' if expected_return < -0.5 else 'sideways')}",
   "probability_up": {daily_stats[-1]['prob_above_current']:.3f},
   "volatility_forecast": {volatility/100:.3f},
   "signal_strength": "{('strong' if abs(overall_signal) > 0.3 else 'moderate' if abs(overall_signal) > 0.1 else 'weak')}",
   "volatility_tier": "{volatility_tier}",
   "volatility_adjusted_confidence": true,
   "monte_carlo_paths": {monte_carlo_paths},
   "fresh_search_applied": {str(news_data.get('fresh_search_performed', False) if news_data else False).lower()},
   "fresh_articles_count": {news_data.get('fresh_articles_found', 0) if news_data else 0},
   "graph_analysis_applied": {str(graph_integration_applied).lower()},
   "primary_chart_pattern": "{primary_pattern}",
   "breakout_detected": {str(breakout_detected).lower()},
   "technical_alignment": "{news_data.get('graph_technical_alignment', 'neutral') if news_data and news_data.get('graph_analysis_integrated') else 'neutral'}",
   "options_sentiment": "{('bullish' if overall_signal > 0.2 else 'bearish' if overall_signal < -0.2 else 'neutral')}",
   "key_factors": ["Volatility-aware confidence", "Dynamic Monte Carlo ({monte_carlo_paths:,} paths)", "Signal-aware drift", "{prediction_timeframe}-term analysis"{"," + '"Graph technical patterns"' if graph_integration_applied else ""}],
   "risk_assessment": "{('high' if volatility > 50 else 'moderate' if volatility > 25 else 'low')}",
   "mathematical_basis": "ENHANCED with Graph Analysis: VOLATILITY-AWARE Monte Carlo with {monte_carlo_paths:,} paths, {enhanced_drift:.1f}% drift, {volatility:.1f}% volatility{', ' + str(primary_pattern) + ' pattern' if primary_pattern and primary_pattern != 'None' else ''}",
   "reasoning": "Volatility-aware analysis with {volatility_tier.lower()} volatility tier adjustments, dynamic Monte Carlo simulation{', and graph technical pattern integration' if graph_integration_applied else ''}"
}}

IMPORTANT: Your predictions MUST reflect the ENHANCED confidence of {enhanced_confidence:.1%} (not generic 85%) and incorporate the graph analysis insights."""

        # STEP 9: Call Claude API with Sonnet 4
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",  # Claude Sonnet 4
                max_tokens=1500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            claude_text = response.content[0].text.strip()
            
            # Import required modules
            import re
            import json
            
            # Parse JSON response
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_match = re.search(json_pattern, claude_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                try:
                    analysis = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"   [ERROR] JSON parsing error: {e}")
                    return self.create_fallback_analysis(ticker, excel_data, news_data, market_data, prediction_days, graph_analysis)
                
                # STEP 10: Validate and override if needed (with volatility-aware checks)
                if abs(analysis.get('total_expected_return_pct', 0) - expected_return) > 2:
                    print(f"   [WRENCH] Overriding Claude's return prediction: {analysis.get('total_expected_return_pct', 0):.2f}% → {expected_return:.2f}%")
                    analysis['total_expected_return_pct'] = expected_return
                
                if abs(analysis.get('final_target_price', 0) - daily_stats[-1]['mean_price']) > current_price * 0.02:
                    print(f"   [WRENCH] Overriding Claude's target price: ${analysis.get('final_target_price', 0):.2f} → ${daily_stats[-1]['mean_price']:.2f}")
                    analysis['final_target_price'] = daily_stats[-1]['mean_price']
                
                # CRITICAL: Override confidence if Claude ignored volatility adjustment
                if abs(analysis.get('confidence', 0.85) - enhanced_confidence) > 0.05:
                    print(f"   [WRENCH] Enforcing volatility-aware confidence: {analysis.get('confidence', 0.85):.3f} → {enhanced_confidence:.3f}")
                    analysis['confidence'] = enhanced_confidence
                
                # NEW: Validate graph analysis integration in Claude's response
                if graph_analysis:
                    if not analysis.get('graph_analysis_applied', False):
                        print(f"   [WRENCH] Enforcing graph analysis integration flag: False → True")
                        analysis['graph_analysis_applied'] = True
                    
                    if analysis.get('primary_chart_pattern', 'none') == 'none' and primary_pattern != 'None':
                        print(f"   [WRENCH] Enforcing primary chart pattern: none → {primary_pattern}")
                        analysis['primary_chart_pattern'] = primary_pattern
                    
                    if not analysis.get('breakout_detected', False) and breakout_detected:
                        print(f"   [WRENCH] Enforcing breakout detection: False → True")
                        analysis['breakout_detected'] = True
                
                # STEP 11: Add comprehensive analysis results with VOLATILITY-AWARE metadata AND GRAPH ANALYSIS
                analysis['mathematical_analysis'] = math_analysis
                analysis['monte_carlo_stats'] = daily_stats
                analysis['options_analysis'] = options_analysis
                analysis['fresh_search_integration'] = True
                analysis['volatility_aware_system'] = True  # NEW FLAG
                analysis['graph_analysis_integration'] = graph_integration_applied  # NEW FLAG
                
                # Enhanced metadata with volatility awareness
                analysis['fresh_search_metadata'] = {
                    'performed': news_data.get('fresh_search_performed', False) if news_data else False,
                    'articles_found': news_data.get('fresh_articles_found', 0) if news_data else 0,
                    'sentiment_score': news_data.get('sentiment_1d', 0.0) if news_data else 0.0,
                    'signal_strength': news_data.get('signal_strength', 0.0) if news_data else 0.0,
                    'prediction_ready': news_data.get('prediction_ready', False) if news_data else False,
                    'quality_score': news_data.get('quality_score', 0.0) if news_data else 0.0,
                    # NEW: Graph-enhanced news metadata
                    'graph_sentiment_adjustment': news_data.get('graph_sentiment_adjustment', 0.0) if news_data else 0.0,
                    'graph_confidence_boost': news_data.get('graph_confidence_boost', 0.0) if news_data else 0.0,
                    'graph_technical_alignment': news_data.get('graph_technical_alignment', 'neutral') if news_data else 'neutral'
                }
                
                analysis['prediction_metadata'] = {
                    'timeframe': prediction_timeframe,
                    'signal_strength': abs(overall_signal),
                    'enhanced_confidence': enhanced_confidence,
                    'volatility_aware_confidence': True,  # NEW FLAG
                    'volatility_tier': volatility_tier,
                    'monte_carlo_paths': monte_carlo_paths,
                    'confidence_breakdown': confidence_results,
                    'bearish_protection': bearish_protection,
                    'drift_adjusted': drift_adjusted,
                    # NEW: Graph analysis metadata
                    'graph_analysis_applied': graph_integration_applied,
                    'graph_signal_adjustment': math_analysis['composite_scores'].get('graph_signal_adjustment', 0.0),
                    'graph_regime_override': math_analysis['composite_scores'].get('graph_regime_override_applied', False)
                }
                
                # NEW: Volatility-specific metadata
                analysis['volatility_analysis'] = {
                    'volatility_percentage': volatility,
                    'volatility_tier': volatility_tier,
                    'volatility_penalty': volatility_penalty,
                    'data_quality_bonus': data_quality_bonus,
                    'monte_carlo_path_count': monte_carlo_paths,
                    'confidence_range': confidence_results.get('confidence_range', (0, 100)),
                    'predictability_factor': confidence_results.get('vol_predictability_factor', 1.0)
                }
                
                # NEW: Graph Analysis Results (if available)
                if graph_analysis:
                    analysis['graph_analysis_results'] = {
                        'primary_pattern': primary_pattern,
                        'pattern_reliability': pattern_reliability,
                        'breakout_detected': breakout_detected,
                        'breakout_direction': breakout_direction,
                        'breakout_strength': breakout_strength,
                        'momentum_acceleration': momentum_acceleration,
                        'candlestick_patterns_found': candlestick_patterns,
                        'support_level': nearest_support,
                        'resistance_level': nearest_resistance,
                        'integration_results': math_analysis.get('graph_integration_results', {}),
                        'technical_alignment': news_data.get('graph_technical_alignment', 'neutral') if news_data and news_data.get('graph_analysis_integrated') else 'neutral'
                    }
                else:
                    analysis['graph_analysis_results'] = {
                        'available': False,
                        'reason': 'No graph analysis provided'
                    }
                
                # Enhanced logging
                print(f"[OK] ENHANCED Claude analysis complete with VOLATILITY-AWARE confidence system and GRAPH ANALYSIS")
                print(f"   [CHART] Signal: {overall_signal:+.3f}, Expected Return: {expected_return:+.2f}%")
                print(f"   🔄 Fresh Articles: {news_data.get('fresh_articles_found', 0) if news_data else 0}")
                print(f"   [TARGET] VOLATILITY-AWARE Confidence: {enhanced_confidence:.1%} ({volatility_tier} tier)")
                print(f"   [CHART] Monte Carlo paths: {monte_carlo_paths:,} (volatility-adjusted)")
                if graph_analysis:
                    print(f"   [UP] Graph Analysis: [OK] INTEGRATED")
                    print(f"   [CHART] Primary Pattern: {primary_pattern} (Reliability: {pattern_reliability:.1%})")
                    if breakout_detected:
                        print(f"   [ROCKET] Breakout: {breakout_direction.upper()} (Strength: {breakout_strength:.1%})")
                    print(f"   [CHART] Technical Alignment: {news_data.get('graph_technical_alignment', 'neutral') if news_data and news_data.get('graph_analysis_integrated') else 'neutral'}")
                
                return analysis
                
            else:
                print("[WARNING] Could not parse Claude JSON response")
                return self.create_fallback_analysis(ticker, excel_data, news_data, market_data, prediction_days, graph_analysis)
            
        except Exception as e:
            print(f"[ERROR] Claude API error: {e}")
            return self.create_fallback_analysis(ticker, excel_data, news_data, market_data, prediction_days, graph_analysis)

    def _count_data_sources(self, news_data, custom_articles):
        """Count total data sources used in analysis"""
        sources = 0
        
        # Excel data source
        sources += 1
        
        # Fresh search sources
        if news_data.get('fresh_search_performed', False):
            sources += 1
        
        # Phase 3 database
        if news_data.get('news_volume_7d', 0) > 0:
            sources += 1
        
        # Custom articles
        if custom_articles:
            sources += 1
        
        return sources

    def _calculate_data_quality_metrics(self, news_data, custom_articles, excel_data):
        """Calculate comprehensive data quality metrics"""
        metrics = {
            'overall_quality': 0.0,
            'data_completeness': 0.0,
            'news_coverage': 0.0,
            'technical_coverage': 0.0,
            'recency_score': 0.0
        }
        
        try:
            # News coverage quality
            fresh_articles = news_data.get('fresh_articles_found', 0)
            total_articles = news_data.get('total_articles', 0)
            
            if total_articles > 0:
                metrics['news_coverage'] = min(1.0, total_articles / 20)  # Max at 20 articles
            
            # Technical coverage quality
            technical_indicators = ['volatility', 'current_rsi', 'sma_20', 'sma_50', 'performance_return_1_month']
            available_indicators = sum(1 for indicator in technical_indicators if excel_data.get(indicator) is not None)
            metrics['technical_coverage'] = available_indicators / len(technical_indicators)
            
            # Recency score (fresh search gets higher score)
            if news_data.get('fresh_search_performed', False):
                metrics['recency_score'] = 0.9
            elif news_data.get('news_volume_1d', 0) > 0:
                metrics['recency_score'] = 0.6
            else:
                metrics['recency_score'] = 0.3
            
            # Data completeness
            completeness_factors = [
                1.0 if excel_data.get('volatility') else 0.0,
                1.0 if total_articles > 0 else 0.0,
                1.0 if custom_articles else 0.0,
                1.0 if news_data.get('fresh_search_performed', False) else 0.0
            ]
            metrics['data_completeness'] = sum(completeness_factors) / len(completeness_factors)
            
            # Overall quality score
            metrics['overall_quality'] = (
                metrics['news_coverage'] * 0.3 +
                metrics['technical_coverage'] * 0.3 +
                metrics['recency_score'] * 0.2 +
                metrics['data_completeness'] * 0.2
            )
            
        except Exception as e:
            print(f"   [WARNING] Data quality calculation error: {e}")
        
        return metrics

    def save_ultimate_prediction_enhanced(self, ticker, excel_data, news_data, market_data, claude_analysis, custom_articles, excel_file_path, options_data=None):
        """Wrapper method for backwards compatibility"""
        return self.save_prediction_data(ticker, excel_data, news_data, market_data, claude_analysis, custom_articles, excel_file_path, options_data)
    
    def save_prediction_data(self, ticker, excel_data, news_data, market_data, claude_analysis, custom_articles, excel_file_path, options_data=None):
        """Save comprehensive prediction data with MAE tracking"""
        print(f"💾 Saving enhanced prediction data for {ticker}...")
        
        try:
            from datetime import datetime
            import json
            import os
            
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Prepare comprehensive data structure
            prediction_data = {
                'metadata': {
                    'ticker': ticker,
                    'timestamp': timestamp,
                    'prediction_engine_version': 'MAE_ENHANCED_v1.0',
                    'excel_file_path': excel_file_path,
                    'mae_system_enabled': True
                },
                
                # Core prediction results
                'claude_analysis': claude_analysis,
                'excel_data': excel_data,
                'news_data': news_data,
                'market_data': market_data,
                'custom_articles': custom_articles or [],
                
                # MAE-specific results
                'mae_performance': {
                    'system_enabled': True,
                    'final_day_mae': claude_analysis.get('mathematical_analysis', {}).get('statistical_metrics', {}).get('mae_performance', {}).get('final_day_mae', 0),
                    'mae_target': claude_analysis.get('mathematical_analysis', {}).get('statistical_metrics', {}).get('mae_performance', {}).get('mae_target', 0),
                    'mae_success_rate': claude_analysis.get('mathematical_analysis', {}).get('statistical_metrics', {}).get('mae_performance', {}).get('mae_success_rate', 0),
                    'volatility_tier': claude_analysis.get('volatility_analysis', {}).get('volatility_tier', 'unknown'),
                    'range_tightening_applied': claude_analysis.get('mathematical_analysis', {}).get('statistical_metrics', {}).get('mae_performance', {}).get('avg_range_tightening', 0)
                },
                
                # Enhanced analysis metadata
                'analysis_metadata': {
                    'volatility_aware_confidence': claude_analysis.get('volatility_aware_system', False),
                    'monte_carlo_paths': claude_analysis.get('prediction_metadata', {}).get('monte_carlo_paths', 0),
                    'fresh_search_applied': claude_analysis.get('fresh_search_metadata', {}).get('performed', False),
                    'regime_adaptive_weights': claude_analysis.get('mathematical_analysis', {}).get('composite_scores', {}).get('regime_adaptive_weights', False)
                },
                
                # Options data if provided
                'options_analysis': claude_analysis.get('options_analysis') if options_data else None,
                
                # Quality metrics
                'quality_metrics': {
                    'prediction_confidence': claude_analysis.get('confidence', 0),
                    'signal_strength': claude_analysis.get('mathematical_analysis', {}).get('composite_scores', {}).get('signal_strength', 0),
                    'volatility_adjusted_confidence': claude_analysis.get('prediction_metadata', {}).get('volatility_aware_confidence', False)
                }
            }
            
            # Create output directory if it doesn't exist
            output_dir = "predictions_output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save to JSON file
            json_filename = f"{output_dir}/{ticker}_prediction_{timestamp}.json"
            with open(json_filename, 'w') as f:
                json.dump(prediction_data, f, indent=2, default=str)
            
            # Log MAE performance to error tracker if available
            if hasattr(self, 'error_tracker') and prediction_data['mae_performance']['system_enabled']:
                try:
                    predicted_range = (
                        claude_analysis.get('mathematical_analysis', {}).get('statistical_metrics', {}).get('daily_stats_with_mae', [{}])[-1].get('min_price'),
                        claude_analysis.get('mathematical_analysis', {}).get('statistical_metrics', {}).get('daily_stats_with_mae', [{}])[-1].get('max_price')
                    )
                    
                    # For now, we don't have actual price to compare, so we'll skip the error logging
                    # In real usage, you'd call this after you have the actual price:
                    # self.error_tracker.log_prediction_error(ticker, predicted_price, actual_price, timestamp, volatility, predicted_range)
                    
                except Exception as e:
                    print(f"   [WARNING] MAE tracking setup failed: {e}")
            
            # Enhanced logging with MAE details
            mae_info = prediction_data['mae_performance']
            print(f"[OK] Prediction data saved successfully:")
            print(f"   📄 File: {json_filename}")
            print(f"   [TARGET] MAE System: {'[OK] Enabled' if mae_info['system_enabled'] else '[ERROR] Disabled'}")
            if mae_info['system_enabled']:
                print(f"   [CHART] Final MAE: {mae_info['final_day_mae']:.1f}% (target: {mae_info['mae_target']:.1f}%)")
                print(f"   [UP] Success Rate: {mae_info['mae_success_rate']:.1%}")
                print(f"   🏷️ Volatility Tier: {mae_info['volatility_tier']}")
            print(f"   [TARGET] Confidence: {prediction_data['quality_metrics']['prediction_confidence']:.1%}")
            print(f"   [CHART] Monte Carlo Paths: {prediction_data['analysis_metadata']['monte_carlo_paths']:,}")
            
            return json_filename
            
        except Exception as e:
            print(f"[ERROR] Error saving prediction data: {e}")
            
            # Fallback: create minimal save
            try:
                fallback_filename = f"predictions_output/{ticker}_prediction_fallback_{timestamp}.json"
                fallback_data = {
                    'ticker': ticker,
                    'timestamp': timestamp,
                    'error': str(e),
                    'basic_prediction': claude_analysis
                }
                
                os.makedirs("predictions_output", exist_ok=True)
                with open(fallback_filename, 'w') as f:
                    json.dump(fallback_data, f, indent=2, default=str)
                
                print(f"   💾 Fallback save created: {fallback_filename}")
                return fallback_filename
                
            except Exception as fallback_error:
                print(f"[ERROR] Fallback save also failed: {fallback_error}")
                return None

# Example usage and testing
if __name__ == "__main__":
    # Initialize the engine
    engine = StockPredictionEngine()
    
    print("[ROCKET] Stock Prediction Engine - FIXED VERSION")
    print("=" * 50)
    
    # Example test with sample data
    sample_excel_data = {
        'performance_return_1_month': 8.5,
        'volatility': 28.0,
        'sector': 'Technology',
        'current_rsi': 62.0,
        'sma_20': 155.0,
        'sma_50': 150.0
    }
    
    sample_market_data = {
        'current_price': 160.0,
        'volume': 1000000,
        'change_percent': 2.1
    }
    
    sample_custom_articles = [
        {
            'title': 'Company beats earnings expectations',
            'sentiment_score': 0.7,
            'confidence': 0.8,
            'timestamp': datetime.now().isoformat()
        }
    ]
    
    # Test the mathematical analysis
    price_paths = engine.generate_multi_day_price_paths(160.0, 28.0, 5, num_paths=1000, drift=15.0)
    daily_stats = engine.calculate_daily_stats_from_paths(price_paths, 160.0, 5, 28.0)
    
    math_analysis = engine.perform_mathematical_analysis(
        sample_excel_data, sample_market_data, sample_custom_articles, 5, price_paths
    )
    
    print(f"\n[CHART] Test Results:")
    print(f"Expected 5-day return: {math_analysis['statistical_metrics']['expected_return_pct']:.2f}%")
    print(f"Overall signal: {math_analysis['composite_scores']['overall_signal']:+.3f}")
    print(f"Probability of gain: {math_analysis['statistical_metrics']['probability_of_gain']:.1%}")
    
    print("\n[OK] Fixed Stock Prediction Engine ready for use!")