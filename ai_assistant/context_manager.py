"""
Context Manager for TradeLens AI
Integrates with existing stock analyzer to provide real-time stock data context
"""

import sys
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data_processor import StockDataProcessor
    from utils import get_logger, is_market_open
    from config import config
except ImportError as e:
    print(f"Warning: Could not import stock analyzer modules: {e}")
    print("Context manager will use simulated data.")

logger = logging.getLogger(__name__)

class StockContextManager:
    """Manages stock data context for AI responses"""

    def __init__(self):
        self.logger = get_logger("StockContextManager") if 'get_logger' in globals() else logging.getLogger(__name__)

        # Try to initialize data processor
        try:
            self.data_processor = StockDataProcessor()
            self.has_data_processor = True
        except NameError:
            self.data_processor = None
            self.has_data_processor = False
            self.logger.warning("StockDataProcessor not available. Using simulated data.")

        # Cache for recent data to avoid repeated API calls
        self.data_cache = {}
        self.cache_timeout = 300  # 5 minutes

    def get_enhanced_context(self, symbol: str) -> Dict[str, Any]:
        """Get enhanced stock context including technical indicators"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            if cache_key in self.data_cache:
                cache_age = datetime.now() - self.data_cache[cache_key]['timestamp']
                if cache_age.seconds < self.cache_timeout:
                    return self.data_cache[cache_key]['data']

            # Get fresh data
            if self.has_data_processor:
                context = self._get_real_context(symbol)
            else:
                context = self._get_simulated_context(symbol)

            # Cache the result
            self.data_cache[cache_key] = {
                'data': context,
                'timestamp': datetime.now()
            }

            return context

        except Exception as e:
            self.logger.error(f"Error getting enhanced context for {symbol}: {str(e)}")
            return self._get_simulated_context(symbol)

    def _get_real_context(self, symbol: str) -> Dict[str, Any]:
        """Get real stock data using the existing data processor"""
        try:
            # Process stock data using existing analyzer
            stock_data = self.data_processor.process_stock(symbol)

            # Extract relevant information for AI context
            context = {
                'symbol': symbol,
                'price': stock_data.get('current_price', 0),
                'dayChange': stock_data.get('day_change_percent', 0),
                'volume': stock_data.get('volume', 0),
                'marketCap': stock_data.get('market_cap', 'N/A'),
                'peRatio': stock_data.get('pe_ratio', 'N/A'),

                # Technical indicators
                'rsi': stock_data.get('technical_indicators', {}).get('RSI', 50),
                'macd': stock_data.get('technical_indicators', {}).get('MACD', 0),
                'macd_signal': stock_data.get('technical_indicators', {}).get('MACD_signal', 0),
                'sma_20': stock_data.get('technical_indicators', {}).get('SMA_20', 0),
                'sma_50': stock_data.get('technical_indicators', {}).get('SMA_50', 0),
                'bollinger_upper': stock_data.get('technical_indicators', {}).get('BB_upper', 0),
                'bollinger_lower': stock_data.get('technical_indicators', {}).get('BB_lower', 0),

                # Additional context
                'market_status': self._get_market_status(),
                'volatility': stock_data.get('volatility', 'normal'),
                'trend': self._determine_trend(stock_data),
                'support_level': stock_data.get('support_resistance', {}).get('support', 0),
                'resistance_level': stock_data.get('support_resistance', {}).get('resistance', 0),

                # News sentiment if available
                'sentiment_score': stock_data.get('sentiment', {}).get('score', 0),
                'news_count': stock_data.get('sentiment', {}).get('news_count', 0),

                # Performance metrics
                'week_performance': stock_data.get('performance', {}).get('1_week', 0),
                'month_performance': stock_data.get('performance', {}).get('1_month', 0),
                'ytd_performance': stock_data.get('performance', {}).get('ytd', 0),

                'last_updated': datetime.now().isoformat()
            }

            return context

        except Exception as e:
            self.logger.error(f"Error getting real context: {str(e)}")
            return self._get_simulated_context(symbol)

    def _get_simulated_context(self, symbol: str) -> Dict[str, Any]:
        """Get simulated stock data for testing/fallback"""
        import random

        # Simulate realistic stock data
        base_price = random.uniform(50, 500)
        day_change = random.uniform(-5, 5)

        context = {
            'symbol': symbol,
            'price': round(base_price, 2),
            'dayChange': round(day_change, 2),
            'volume': random.randint(1000000, 100000000),
            'marketCap': f"{random.uniform(1, 3000):.1f}B",
            'peRatio': round(random.uniform(15, 45), 1),

            # Technical indicators
            'rsi': round(random.uniform(20, 80), 1),
            'macd': round(random.uniform(-2, 2), 3),
            'macd_signal': round(random.uniform(-2, 2), 3),
            'sma_20': round(base_price * random.uniform(0.95, 1.05), 2),
            'sma_50': round(base_price * random.uniform(0.90, 1.10), 2),
            'bollinger_upper': round(base_price * 1.02, 2),
            'bollinger_lower': round(base_price * 0.98, 2),

            # Additional context
            'market_status': self._get_market_status(),
            'volatility': random.choice(['low', 'normal', 'high']),
            'trend': self._get_simulated_trend(day_change),
            'support_level': round(base_price * 0.95, 2),
            'resistance_level': round(base_price * 1.05, 2),

            # News sentiment
            'sentiment_score': round(random.uniform(-1, 1), 2),
            'news_count': random.randint(0, 20),

            # Performance metrics
            'week_performance': round(random.uniform(-10, 10), 2),
            'month_performance': round(random.uniform(-20, 20), 2),
            'ytd_performance': round(random.uniform(-30, 30), 2),

            'last_updated': datetime.now().isoformat(),
            'simulated': True
        }

        return context

    def _get_market_status(self) -> str:
        """Get current market status"""
        try:
            if 'is_market_open' in globals():
                return "open" if is_market_open() else "closed"
            else:
                # Simple time-based check for US markets
                now = datetime.now()
                weekday = now.weekday()
                hour = now.hour

                # Basic US market hours check (9:30 AM - 4:00 PM ET, Mon-Fri)
                if weekday < 5 and 9 <= hour < 16:
                    return "open"
                else:
                    return "closed"
        except Exception:
            return "unknown"

    def _determine_trend(self, stock_data: Dict[str, Any]) -> str:
        """Determine current trend based on technical indicators"""
        try:
            price = stock_data.get('current_price', 0)
            sma_20 = stock_data.get('technical_indicators', {}).get('SMA_20', 0)
            sma_50 = stock_data.get('technical_indicators', {}).get('SMA_50', 0)
            day_change = stock_data.get('day_change_percent', 0)

            if price > sma_20 > sma_50 and day_change > 1:
                return "bullish"
            elif price < sma_20 < sma_50 and day_change < -1:
                return "bearish"
            else:
                return "neutral"
        except Exception:
            return "neutral"

    def _get_simulated_trend(self, day_change: float) -> str:
        """Get simulated trend based on day change"""
        if day_change > 2:
            return "bullish"
        elif day_change < -2:
            return "bearish"
        else:
            return "neutral"

    def get_quick_insights(self, context: Dict[str, Any]) -> List[str]:
        """Generate quick insights based on current context"""
        insights = []

        try:
            symbol = context.get('symbol', '')
            price = context.get('price', 0)
            rsi = context.get('rsi', 50)
            day_change = context.get('dayChange', 0)
            trend = context.get('trend', 'neutral')
            volume = context.get('volume', 0)

            # RSI insights
            if rsi > 70:
                insights.append(f"{symbol} RSI at {rsi} suggests overbought conditions")
            elif rsi < 30:
                insights.append(f"{symbol} RSI at {rsi} suggests oversold conditions")

            # Price movement insights
            if abs(day_change) > 3:
                direction = "up" if day_change > 0 else "down"
                insights.append(f"Strong {direction} movement: {abs(day_change):.1f}% today")

            # Trend insights
            if trend == "bullish":
                insights.append("Technical indicators show bullish momentum")
            elif trend == "bearish":
                insights.append("Technical indicators show bearish pressure")

            # Volume insights
            if volume > 50000000:  # Arbitrary threshold for high volume
                insights.append("Higher than average trading volume detected")

            return insights[:3]  # Return top 3 insights

        except Exception as e:
            self.logger.error(f"Error generating insights: {str(e)}")
            return ["Market data analysis in progress"]

    def get_risk_assessment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk assessment based on current context"""
        try:
            volatility = context.get('volatility', 'normal')
            rsi = context.get('rsi', 50)
            day_change = abs(context.get('dayChange', 0))

            # Calculate risk score (0-100)
            risk_score = 50  # Base score

            if volatility == 'high':
                risk_score += 20
            elif volatility == 'low':
                risk_score -= 10

            if rsi > 80 or rsi < 20:
                risk_score += 15

            if day_change > 5:
                risk_score += 15
            elif day_change > 3:
                risk_score += 10

            risk_score = max(0, min(100, risk_score))

            # Risk level
            if risk_score > 80:
                risk_level = "high"
                risk_message = "High volatility and extreme conditions"
            elif risk_score > 60:
                risk_level = "moderate-high"
                risk_message = "Elevated risk due to market conditions"
            elif risk_score > 40:
                risk_level = "moderate"
                risk_message = "Normal market risk levels"
            else:
                risk_level = "low"
                risk_message = "Relatively stable conditions"

            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_message': risk_message,
                'factors': {
                    'volatility': volatility,
                    'rsi_extreme': rsi > 80 or rsi < 20,
                    'high_movement': day_change > 3
                }
            }

        except Exception as e:
            self.logger.error(f"Error calculating risk assessment: {str(e)}")
            return {
                'risk_score': 50,
                'risk_level': 'moderate',
                'risk_message': 'Unable to calculate risk at this time',
                'factors': {}
            }

    def clear_cache(self):
        """Clear the data cache"""
        self.data_cache.clear()
        self.logger.info("Context cache cleared")

# For testing purposes
if __name__ == "__main__":
    manager = StockContextManager()

    # Test with a sample symbol
    test_symbol = "AAPL"
    context = manager.get_enhanced_context(test_symbol)

    print(f"Context for {test_symbol}:")
    for key, value in context.items():
        print(f"  {key}: {value}")

    print("\nQuick Insights:")
    insights = manager.get_quick_insights(context)
    for insight in insights:
        print(f"  - {insight}")

    print("\nRisk Assessment:")
    risk = manager.get_risk_assessment(context)
    print(f"  Risk Level: {risk['risk_level']}")
    print(f"  Risk Score: {risk['risk_score']}")
    print(f"  Message: {risk['risk_message']}")