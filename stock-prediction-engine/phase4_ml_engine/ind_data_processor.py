#!/usr/bin/env python3
"""
Indian Market Data Processor
Handles data processing and conversion for Indian markets with Zerodha API
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class IndianMarketProcessor:
    """Process and convert Indian market data for analysis"""
    
    def __init__(self):
        """Initialize the market processor"""
        
        # Lot sizes for major stocks (as of 2024)
        self.lot_sizes = {
            'RELIANCE': 250,
            'TCS': 125,
            'HDFCBANK': 550,
            'INFY': 300,
            'HINDUNILVR': 300,
            'ITC': 1600,
            'SBIN': 750,
            'BHARTIARTL': 475,
            'AXISBANK': 1200,
            'ICICIBANK': 700,
            'KOTAKBANK': 400,
            'LT': 150,
            'WIPRO': 400,
            'MARUTI': 100,
            'TITAN': 375,
            'NIFTY 50': 50,
            'NIFTY BANK': 25,
            'NIFTY FIN SERVICE': 40,
            'MIDCPNIFTY': 75
        }
        
        logger.info("Indian Market Processor initialized")
    
    def process_for_analyzer(self, ticker: str, current_data: Dict, 
                           historical_data: pd.DataFrame) -> Dict:
        """Process market data for the options analyzer"""
        
        try:
            # Check if we have sufficient data
            if historical_data.empty:
                logger.warning(f"No historical data available for {ticker}")
                return self._create_basic_data(ticker, current_data)
            
            # Calculate technical indicators
            technical_data = self._calculate_technical_indicators(historical_data)
            
            # Calculate support and resistance
            support_resistance = self._calculate_support_resistance(historical_data)
            
            # Prepare data in format expected by analyzer
            processed_data = {
                'ticker': ticker,
                'current_price': current_data.get('price', 0),
                'price_data': {
                    'open': current_data.get('open', 0),
                    'high': current_data.get('high', 0),
                    'low': current_data.get('low', 0),
                    'close': current_data.get('price', 0),
                    'volume': current_data.get('volume', 0),
                    'change': current_data.get('change', 0),
                    'change_percent': current_data.get('change_percent', 0)
                },
                'technical_indicators': technical_data,
                'support_resistance': support_resistance,
                'historical_data': historical_data.to_dict('records') if not historical_data.empty else []
            }
            
            logger.info(f"Successfully processed data for {ticker}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing data for {ticker}: {e}")
            return self._create_basic_data(ticker, current_data)
    
    def _create_basic_data(self, ticker: str, current_data: Dict) -> Dict:
        """Create basic data structure when full processing fails"""
        
        current_price = current_data.get('price', 0)
        
        return {
            'ticker': ticker,
            'current_price': current_price,
            'price_data': {
                'open': current_data.get('open', current_price),
                'high': current_data.get('high', current_price),
                'low': current_data.get('low', current_price),
                'close': current_price,
                'volume': current_data.get('volume', 0),
                'change': current_data.get('change', 0),
                'change_percent': current_data.get('change_percent', 0)
            },
            'technical_indicators': self._get_default_indicators(current_price),
            'support_resistance': self._get_default_levels(current_price),
            'historical_data': []
        }
    
    def _get_default_indicators(self, price: float) -> Dict:
        """Get default technical indicators when calculation fails"""
        
        return {
            'sma_20': price,
            'sma_50': price,
            'sma_200': price,
            'ema_9': price,
            'ema_21': price,
            'rsi': 50.0,
            'macd': 0.0,
            'macd_signal': 0.0,
            'macd_histogram': 0.0,
            'bb_upper': price * 1.02,
            'bb_lower': price * 0.98,
            'bb_middle': price,
            'volume_sma': 1000000,
            'volume_ratio': 1.0,
            'atr': price * 0.02,
            'stoch_k': 50.0,
            'stoch_d': 50.0
        }
    
    def _get_default_levels(self, price: float) -> Dict:
        """Get default support/resistance levels"""
        
        return {
            'immediate_support': price * 0.98,
            'immediate_resistance': price * 1.02,
            'strong_support': price * 0.95,
            'strong_resistance': price * 1.05,
            'pivot': price,
            'recent_high': price * 1.01,
            'recent_low': price * 0.99,
            'volume_support': price * 0.97,
            'volume_resistance': price * 1.03
        }
    
    def convert_options_chain(self, options_chain: Dict, spot_price: float) -> Dict:
        """Convert Zerodha options chain to analyzer format"""
        
        try:
            if not options_chain or 'strikes' not in options_chain:
                logger.warning("Invalid options chain data")
                return self._create_dummy_options_chain(spot_price)
            
            # Process strikes
            processed_strikes = []
            
            for strike_data in options_chain['strikes']:
                strike = strike_data.get('strike', 0)
                
                if strike == 0:
                    continue
                
                # Calculate moneyness
                moneyness = (strike - spot_price) / spot_price if spot_price > 0 else 0
                
                # Get call data with defaults
                call_data = strike_data.get('call', {})
                put_data = strike_data.get('put', {})
                
                processed_strike = {
                    'strike': strike,
                    'moneyness': moneyness,
                    'call': {
                        'bid': call_data.get('bid', 0),
                        'ask': call_data.get('ask', 0),
                        'last': call_data.get('ltp', 0),
                        'volume': call_data.get('volume', 0),
                        'open_interest': call_data.get('oi', 0),
                        'implied_volatility': call_data.get('iv', 20) / 100,  # Default 20% IV
                        'delta': self._estimate_delta(spot_price, strike, 'call'),
                        'gamma': call_data.get('gamma', 0),
                        'theta': call_data.get('theta', 0),
                        'vega': call_data.get('vega', 0)
                    },
                    'put': {
                        'bid': put_data.get('bid', 0),
                        'ask': put_data.get('ask', 0),
                        'last': put_data.get('ltp', 0),
                        'volume': put_data.get('volume', 0),
                        'open_interest': put_data.get('oi', 0),
                        'implied_volatility': put_data.get('iv', 20) / 100,  # Default 20% IV
                        'delta': self._estimate_delta(spot_price, strike, 'put'),
                        'gamma': put_data.get('gamma', 0),
                        'theta': put_data.get('theta', 0),
                        'vega': put_data.get('vega', 0)
                    }
                }
                
                processed_strikes.append(processed_strike)
            
            if not processed_strikes:
                logger.warning("No valid strikes found in options chain")
                return self._create_dummy_options_chain(spot_price)
            
            # Sort by strike
            processed_strikes.sort(key=lambda x: x['strike'])
            
            # Calculate additional metrics
            atm_strike = min(processed_strikes, 
                           key=lambda x: abs(x['strike'] - spot_price))['strike']
            
            # Calculate Put-Call Ratio
            total_put_oi = sum(s['put']['open_interest'] for s in processed_strikes)
            total_call_oi = sum(s['call']['open_interest'] for s in processed_strikes)
            pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1
            
            # Find max pain
            max_pain_strike = self._calculate_max_pain(processed_strikes)
            
            result = {
                'symbol': options_chain.get('symbol', 'UNKNOWN'),
                'spot_price': spot_price,
                'expiry': options_chain.get('expiry', ''),
                'strikes': processed_strikes,
                'atm_strike': atm_strike,
                'put_call_ratio': pcr,
                'max_pain': max_pain_strike,
                'total_volume': sum(s['call']['volume'] + s['put']['volume'] 
                                  for s in processed_strikes),
                'total_oi': total_put_oi + total_call_oi
            }
            
            logger.info(f"Successfully converted options chain for {options_chain.get('symbol')}")
            return result
            
        except Exception as e:
            logger.error(f"Error converting options chain: {e}")
            return self._create_dummy_options_chain(spot_price)
    
    def _create_dummy_options_chain(self, spot_price: float) -> Dict:
        """Create a dummy options chain when real data is unavailable"""
        
        strikes = []
        base_strike = round(spot_price / 50) * 50  # Round to nearest 50
        
        # Create 5 strikes around current price
        for i in range(-2, 3):
            strike = base_strike + (i * 50)
            strikes.append({
                'strike': strike,
                'moneyness': (strike - spot_price) / spot_price,
                'call': {
                    'bid': 0, 'ask': 0, 'last': max(0, spot_price - strike),
                    'volume': 0, 'open_interest': 0, 'implied_volatility': 0.20,
                    'delta': self._estimate_delta(spot_price, strike, 'call'),
                    'gamma': 0, 'theta': 0, 'vega': 0
                },
                'put': {
                    'bid': 0, 'ask': 0, 'last': max(0, strike - spot_price),
                    'volume': 0, 'open_interest': 0, 'implied_volatility': 0.20,
                    'delta': self._estimate_delta(spot_price, strike, 'put'),
                    'gamma': 0, 'theta': 0, 'vega': 0
                }
            })
        
        return {
            'symbol': 'DUMMY',
            'spot_price': spot_price,
            'expiry': '',
            'strikes': strikes,
            'atm_strike': base_strike,
            'put_call_ratio': 1.0,
            'max_pain': base_strike,
            'total_volume': 0,
            'total_oi': 0
        }
    
    def _estimate_delta(self, spot_price: float, strike: float, option_type: str) -> float:
        """Estimate delta when not provided by API"""
        
        try:
            if spot_price <= 0 or strike <= 0:
                return 0.5 if option_type == 'call' else -0.5
            
            # Simple approximation based on moneyness
            moneyness = spot_price / strike
            
            if option_type == 'call':
                if moneyness > 1.1:  # Deep ITM
                    return 0.8
                elif moneyness > 1.05:  # ITM
                    return 0.7
                elif moneyness > 0.95:  # ATM
                    return 0.5
                elif moneyness > 0.9:  # OTM
                    return 0.3
                else:  # Deep OTM
                    return 0.1
            else:  # put
                if moneyness < 0.9:  # Deep ITM
                    return -0.8
                elif moneyness < 0.95:  # ITM
                    return -0.7
                elif moneyness < 1.05:  # ATM
                    return -0.5
                elif moneyness < 1.1:  # OTM
                    return -0.3
                else:  # Deep OTM
                    return -0.1
        
        except Exception as e:
            logger.error(f"Error estimating delta: {e}")
            return 0.5 if option_type == 'call' else -0.5
    
    def get_lot_size(self, ticker: str) -> int:
        """Get lot size for a ticker"""
        return self.lot_sizes.get(ticker.upper(), 1)
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators with error handling"""
        
        try:
            if df.empty or len(df) < 2:
                logger.warning("Insufficient data for technical indicators")
                return self._get_default_indicators(df['close'].iloc[-1] if not df.empty else 0)
            
            indicators = {}
            current_price = df['close'].iloc[-1]
            
            # Simple Moving Averages
            try:
                indicators['sma_20'] = df['close'].rolling(window=min(20, len(df))).mean().iloc[-1]
                indicators['sma_50'] = df['close'].rolling(window=min(50, len(df))).mean().iloc[-1]
                indicators['sma_200'] = df['close'].rolling(window=min(200, len(df))).mean().iloc[-1] if len(df) >= 10 else current_price
            except Exception as e:
                logger.warning(f"Error calculating SMAs: {e}")
                indicators.update({
                    'sma_20': current_price,
                    'sma_50': current_price,
                    'sma_200': current_price
                })
            
            # Exponential Moving Averages
            try:
                indicators['ema_9'] = df['close'].ewm(span=min(9, len(df)), adjust=False).mean().iloc[-1]
                indicators['ema_21'] = df['close'].ewm(span=min(21, len(df)), adjust=False).mean().iloc[-1]
            except Exception as e:
                logger.warning(f"Error calculating EMAs: {e}")
                indicators.update({
                    'ema_9': current_price,
                    'ema_21': current_price
                })
            
            # RSI
            try:
                indicators['rsi'] = self._calculate_rsi(df['close'])
            except Exception as e:
                logger.warning(f"Error calculating RSI: {e}")
                indicators['rsi'] = 50.0
            
            # MACD
            try:
                if len(df) >= 26:
                    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
                    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
                    macd_line = ema_12 - ema_26
                    signal_line = macd_line.ewm(span=9, adjust=False).mean()
                    
                    indicators['macd'] = macd_line.iloc[-1]
                    indicators['macd_signal'] = signal_line.iloc[-1]
                    indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
                else:
                    indicators.update({
                        'macd': 0.0,
                        'macd_signal': 0.0,
                        'macd_histogram': 0.0
                    })
            except Exception as e:
                logger.warning(f"Error calculating MACD: {e}")
                indicators.update({
                    'macd': 0.0,
                    'macd_signal': 0.0,
                    'macd_histogram': 0.0
                })
            
            # Bollinger Bands
            try:
                if len(df) >= 20:
                    sma_20 = df['close'].rolling(window=20).mean()
                    std_20 = df['close'].rolling(window=20).std()
                    indicators['bb_upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
                    indicators['bb_lower'] = (sma_20 - (std_20 * 2)).iloc[-1]
                    indicators['bb_middle'] = sma_20.iloc[-1]
                else:
                    indicators.update({
                        'bb_upper': current_price * 1.02,
                        'bb_lower': current_price * 0.98,
                        'bb_middle': current_price
                    })
            except Exception as e:
                logger.warning(f"Error calculating Bollinger Bands: {e}")
                indicators.update({
                    'bb_upper': current_price * 1.02,
                    'bb_lower': current_price * 0.98,
                    'bb_middle': current_price
                })
            
            # Volume indicators
            try:
                indicators['volume_sma'] = df['volume'].rolling(window=min(20, len(df))).mean().iloc[-1]
                indicators['volume_ratio'] = df['volume'].iloc[-1] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1.0
            except Exception as e:
                logger.warning(f"Error calculating volume indicators: {e}")
                indicators.update({
                    'volume_sma': df['volume'].iloc[-1] if not df.empty else 1000000,
                    'volume_ratio': 1.0
                })
            
            # ATR (Average True Range)
            try:
                if len(df) >= 14:
                    high_low = df['high'] - df['low']
                    high_close = np.abs(df['high'] - df['close'].shift())
                    low_close = np.abs(df['low'] - df['close'].shift())
                    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    indicators['atr'] = true_range.rolling(window=14).mean().iloc[-1]
                else:
                    indicators['atr'] = current_price * 0.02
            except Exception as e:
                logger.warning(f"Error calculating ATR: {e}")
                indicators['atr'] = current_price * 0.02
            
            # Stochastic
            try:
                if len(df) >= 14:
                    low_14 = df['low'].rolling(window=14).min()
                    high_14 = df['high'].rolling(window=14).max()
                    k_percent = 100 * ((df['close'] - low_14) / (high_14 - low_14))
                    indicators['stoch_k'] = k_percent.iloc[-1] if not pd.isna(k_percent.iloc[-1]) else 50.0
                    indicators['stoch_d'] = k_percent.rolling(window=3).mean().iloc[-1] if not pd.isna(k_percent.rolling(window=3).mean().iloc[-1]) else 50.0
                else:
                    indicators.update({
                        'stoch_k': 50.0,
                        'stoch_d': 50.0
                    })
            except Exception as e:
                logger.warning(f"Error calculating Stochastic: {e}")
                indicators.update({
                    'stoch_k': 50.0,
                    'stoch_d': 50.0
                })
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return self._get_default_indicators(df['close'].iloc[-1] if not df.empty else 0)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI with error handling"""
        
        try:
            if len(prices) < period + 1:
                return 50.0
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            result = rsi.iloc[-1]
            return result if not pd.isna(result) else 50.0
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels with error handling"""
        
        try:
            if df.empty:
                return self._get_default_levels(0)
            
            current_price = df['close'].iloc[-1]
            
            # Method 1: Recent highs and lows
            window = min(20, len(df))
            recent_high = df['high'].rolling(window=window).max().iloc[-1]
            recent_low = df['low'].rolling(window=window).min().iloc[-1]
            
            # Method 2: Pivot points
            last_close = df['close'].iloc[-1]
            last_high = df['high'].iloc[-1]
            last_low = df['low'].iloc[-1]
            
            pivot = (last_high + last_low + last_close) / 3
            r1 = (2 * pivot) - last_low
            s1 = (2 * pivot) - last_high
            r2 = pivot + (last_high - last_low)
            s2 = pivot - (last_high - last_low)
            
            # Method 3: Volume-weighted levels
            try:
                volume_levels = self._find_volume_levels(df)
            except Exception as e:
                logger.warning(f"Error calculating volume levels: {e}")
                volume_levels = {
                    'support': current_price * 0.97,
                    'resistance': current_price * 1.03
                }
            
            return {
                'immediate_support': max(s1, 0),
                'immediate_resistance': r1,
                'strong_support': max(s2, 0),
                'strong_resistance': r2,
                'pivot': pivot,
                'recent_high': recent_high,
                'recent_low': recent_low,
                'volume_support': volume_levels.get('support', s1),
                'volume_resistance': volume_levels.get('resistance', r1)
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            current_price = df['close'].iloc[-1] if not df.empty else 0
            return self._get_default_levels(current_price)
    
    def _find_volume_levels(self, df: pd.DataFrame) -> Dict:
        """Find price levels with high volume"""
        
        try:
            if len(df) < 10:
                current_price = df['close'].iloc[-1]
                return {
                    'support': current_price * 0.97,
                    'resistance': current_price * 1.03
                }
            
            # Create price bins
            price_range = df['high'].max() - df['low'].min()
            if price_range <= 0:
                current_price = df['close'].iloc[-1]
                return {
                    'support': current_price * 0.97,
                    'resistance': current_price * 1.03
                }
            
            bin_size = price_range / 50
            
            # Volume profile
            volume_profile = {}
            
            for idx, row in df.iterrows():
                price = row['close']
                volume = row['volume']
                bin_price = round(price / bin_size) * bin_size
                
                if bin_price not in volume_profile:
                    volume_profile[bin_price] = 0
                volume_profile[bin_price] += volume
            
            if not volume_profile:
                current_price = df['close'].iloc[-1]
                return {
                    'support': current_price * 0.97,
                    'resistance': current_price * 1.03
                }
            
            # Sort by volume
            sorted_levels = sorted(volume_profile.items(), 
                                 key=lambda x: x[1], reverse=True)
            
            current_price = df['close'].iloc[-1]
            
            # Find nearest high-volume levels
            support = None
            resistance = None
            
            for price, volume in sorted_levels[:10]:
                if price < current_price and support is None:
                    support = price
                elif price > current_price and resistance is None:
                    resistance = price
                
                if support and resistance:
                    break
            
            return {
                'support': support if support else current_price * 0.97,
                'resistance': resistance if resistance else current_price * 1.03
            }
            
        except Exception as e:
            logger.error(f"Error finding volume levels: {e}")
            current_price = df['close'].iloc[-1] if not df.empty else 0
            return {
                'support': current_price * 0.97,
                'resistance': current_price * 1.03
            }
    
    def _calculate_max_pain(self, strikes: List[Dict]) -> float:
        """Calculate max pain strike price"""
        
        try:
            if not strikes:
                return 0
            
            max_pain_values = []
            
            for strike_data in strikes:
                strike = strike_data['strike']
                total_pain = 0
                
                # Calculate pain for this strike
                for other_strike in strikes:
                    other_price = other_strike['strike']
                    
                    # Call pain
                    if other_price < strike:
                        call_pain = (strike - other_price) * other_strike['call']['open_interest']
                        total_pain += call_pain
                    
                    # Put pain
                    if other_price > strike:
                        put_pain = (other_price - strike) * other_strike['put']['open_interest']
                        total_pain += put_pain
                
                max_pain_values.append((strike, total_pain))
            
            # Find strike with minimum total pain
            if max_pain_values:
                max_pain_strike = min(max_pain_values, key=lambda x: x[1])[0]
                return max_pain_strike
            
            return strikes[0]['strike'] if strikes else 0
            
        except Exception as e:
            logger.error(f"Error calculating max pain: {e}")
            return strikes[0]['strike'] if strikes else 0
    
    def calculate_option_metrics(self, option_data: Dict, spot_price: float, 
                               days_to_expiry: int, risk_free_rate: float = 0.07) -> Dict:
        """Calculate additional option metrics"""
        
        try:
            metrics = {}
            
            # Time value
            if option_data.get('type') == 'call':
                intrinsic_value = max(0, spot_price - option_data['strike'])
            else:
                intrinsic_value = max(0, option_data['strike'] - spot_price)
            
            last_price = option_data.get('last', 0)
            metrics['intrinsic_value'] = intrinsic_value
            metrics['time_value'] = max(0, last_price - intrinsic_value)
            
            # Probability ITM (simplified)
            delta = option_data.get('delta', 0)
            if option_data.get('type') == 'call':
                metrics['prob_itm'] = max(0, min(1, delta))
            else:
                metrics['prob_itm'] = max(0, min(1, abs(delta)))
            
            # Breakeven
            if option_data.get('type') == 'call':
                metrics['breakeven'] = option_data['strike'] + last_price
            else:
                metrics['breakeven'] = option_data['strike'] - last_price
            
            # Risk/Reward
            metrics['risk_reward_ratio'] = intrinsic_value / last_price if last_price > 0 else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating option metrics: {e}")
            return {}