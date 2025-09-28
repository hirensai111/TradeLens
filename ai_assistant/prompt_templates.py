"""
Prompt Template Manager for TradeLens AI
Manages AI response formatting and provides context-aware prompts
"""

from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PromptTemplateManager:
    """Manages prompt templates and AI response formatting"""

    def __init__(self):
        self.max_response_sentences = 3
        self.friendly_explanations = True

    def get_system_prompt(self, context: Dict[str, Any]) -> str:
        """Generate system prompt with current stock context"""
        symbol = context.get('symbol', 'UNKNOWN')
        price = context.get('price', 0)
        rsi = context.get('rsi', 50)
        macd = context.get('macd', 0)
        day_change = context.get('dayChange', 0)
        volume = context.get('volume', 0)
        market_status = context.get('market_status', 'unknown')
        trend = context.get('trend', 'neutral')

        # Format volume for readability
        volume_formatted = self._format_volume(volume)

        # Get market context
        market_context = self._get_market_context(market_status)

        # Get trend description
        trend_description = self._get_trend_description(trend, day_change)

        system_prompt = f"""
You are TradeLens AI, a friendly and knowledgeable stock analysis assistant. You help users understand stock market data using simple, conversational language.

CURRENT STOCK DATA FOR {symbol}:
- Current Price: ${price:,.2f}
- Day Change: {day_change:+.1f}%
- RSI: {rsi:.1f}
- MACD: {macd:.3f}
- Volume: {volume_formatted}
- Market Status: {market_status}
- Current Trend: {trend_description}

{market_context}

RESPONSE GUIDELINES:
1. Keep responses to 3 sentences maximum
2. Use friendly, conversational language (like explaining to a friend)
3. For "what is" questions (like "what is RSI"), provide educational explanations using analogies
4. For ticker-specific questions, reference actual numbers from the current data
5. Use analogies to explain complex concepts
6. Be specific and actionable when possible
7. Focus on education, not financial advice
8. If RSI is mentioned, explain what the current {rsi:.1f} means
9. If trend questions come up, reference the {trend} trend and {day_change:+.1f}% daily move

EDUCATIONAL EXPLANATIONS FOR "WHAT IS" QUESTIONS:
- "What is RSI?": RSI (Relative Strength Index) measures momentum on a 0-100 scale. Think of it like a speedometer for stocks - above 70 means overbought (going too fast), below 30 means oversold (going too slow), and 30-70 is the normal cruising range.
- "What is MACD?": MACD (Moving Average Convergence Divergence) shows the relationship between two moving averages. Like two cars on a highway - when the faster one (MACD line) is above the slower one (signal line), it suggests upward momentum, and vice versa.
- "What is SMA?": SMA (Simple Moving Average) is the average price over a set period, like the last 20 days. Think of it as the stock's 'normal' price - when current price is above SMA, it's running hot; below SMA means it's running cool.

EXAMPLE RESPONSES:
- For trend questions: "{symbol} at ${price:,.2f} is showing a {trend} trend with {abs(day_change):.1f}% {'gain' if day_change > 0 else 'decline'} today - {'momentum is building' if abs(day_change) > 2 else 'moving at a steady pace'}."
- For RSI questions: "RSI at {rsi:.1f} means {symbol} is {'overbought (like a car going too fast)' if rsi > 70 else 'oversold (like a spring compressed too much)' if rsi < 30 else 'in the balanced zone (steady cruising speed)'}."
- For buy questions: "With {symbol} at ${price:,.2f} and RSI at {rsi:.1f}, the technicals show {'strength but watch for a pullback' if rsi > 65 else 'potential value but confirm the trend' if rsi < 35 else 'balanced conditions'}. Always consider your risk tolerance and do your research."

IMPORTANT REMINDERS:
- Never give direct financial advice (avoid "you should buy/sell")
- Always mention risk management when discussing positions
- Use the actual current numbers in your responses
- Keep explanations simple and relatable
- Focus on helping users understand what the data means
"""

        return system_prompt

    def get_context_specific_prompt(self, user_message: str, context: Dict[str, Any]) -> str:
        """Generate context-specific prompt enhancements based on user question"""
        user_lower = user_message.lower()
        symbol = context.get('symbol', 'UNKNOWN')
        rsi = context.get('rsi', 50)
        price = context.get('price', 0)
        day_change = context.get('dayChange', 0)

        # Educational "what is" questions
        if any(phrase in user_lower for phrase in ['what is rsi', 'what\'s rsi', 'explain rsi', 'rsi means']):
            return """
EDUCATIONAL RESPONSE REQUIRED:
The user is asking to learn what RSI is. Provide a clear, beginner-friendly explanation using the car speedometer analogy. Focus on education, not the current stock's RSI value.
"""

        elif any(phrase in user_lower for phrase in ['what is macd', 'what\'s macd', 'explain macd', 'macd means']):
            return """
EDUCATIONAL RESPONSE REQUIRED:
The user is asking to learn what MACD is. Provide a clear, beginner-friendly explanation using the two cars on highway analogy. Focus on education, not the current stock's MACD value.
"""

        elif any(phrase in user_lower for phrase in ['what is sma', 'what\'s sma', 'explain sma', 'sma means', 'simple moving average']):
            return """
EDUCATIONAL RESPONSE REQUIRED:
The user is asking to learn what SMA is. Provide a clear, beginner-friendly explanation using the temperature analogy. Focus on education, not the current stock's SMA value.
"""

        # RSI-focused analysis questions (not educational)
        elif any(word in user_lower for word in ['rsi', 'overbought', 'oversold', 'momentum']) and not any(phrase in user_lower for phrase in ['what is', 'what\'s', 'explain', 'means']):
            return f"""
FOCUS ON RSI ANALYSIS:
Current RSI for {symbol}: {rsi:.1f}
- Above 70 = Overbought (potentially due for a pullback)
- Below 30 = Oversold (potentially due for a bounce)
- 30-70 = Normal range

Explain what {rsi:.1f} means in simple terms with an analogy.
"""

        # Trend/direction questions
        elif any(word in user_lower for word in ['trend', 'direction', 'going', 'movement']):
            return f"""
FOCUS ON TREND ANALYSIS:
{symbol} current trend: {context.get('trend', 'neutral')}
Today's move: {day_change:+.1f}%
Price vs 20-day average: {'above' if price > context.get('sma_20', price) else 'below'}

Explain the overall direction and what today's {abs(day_change):.1f}% move means.
"""

        # Buy/investment questions
        elif any(word in user_lower for word in ['buy', 'invest', 'purchase', 'should i']):
            return f"""
FOCUS ON INVESTMENT CONSIDERATIONS:
Current conditions for {symbol}:
- Price: ${price:,.2f}
- RSI: {rsi:.1f} ({'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'})
- Trend: {context.get('trend', 'neutral')}

Provide educational guidance about what these indicators suggest, but remind about personal research and risk tolerance.
"""

        # Risk analysis questions
        elif any(word in user_lower for word in ['risk', 'danger', 'safe', 'volatile']):
            volatility = context.get('volatility', 'normal')
            return f"""
FOCUS ON RISK ASSESSMENT:
Risk factors for {symbol}:
- Volatility: {volatility}
- Daily move: {abs(day_change):.1f}%
- RSI level: {rsi:.1f}

Explain risk management and what current conditions mean for position sizing.
"""

        # Comparison questions
        elif any(word in user_lower for word in ['compare', 'vs', 'versus', 'sector', 'market']):
            return f"""
FOCUS ON COMPARATIVE ANALYSIS:
{symbol} current metrics:
- Day change: {day_change:+.1f}%
- RSI: {rsi:.1f}
- Trend: {context.get('trend', 'neutral')}

Provide context about how these numbers compare to typical market behavior.
"""

        return ""

    def format_response_for_context(self, response: str, context: Dict[str, Any]) -> str:
        """Format and enhance AI response with context-specific information"""
        try:
            # Ensure response is under sentence limit
            sentences = response.split('. ')
            if len(sentences) > self.max_response_sentences:
                response = '. '.join(sentences[:self.max_response_sentences])
                if not response.endswith('.'):
                    response += '.'

            # Add context-specific enhancements if needed
            symbol = context.get('symbol', 'UNKNOWN')

            # Replace generic references with specific data
            price = context.get('price', 0)
            if 'current price' in response.lower() and '$' not in response:
                response = response.replace('current price', f'${price:,.2f}')

            return response

        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return response

    def get_quick_action_responses(self, action: str, context: Dict[str, Any]) -> str:
        """Generate responses for quick action buttons"""
        symbol = context.get('symbol', 'UNKNOWN')
        price = context.get('price', 0)
        rsi = context.get('rsi', 50)
        day_change = context.get('dayChange', 0)
        trend = context.get('trend', 'neutral')

        responses = {
            "What's the trend?": f"{symbol} at ${price:,.2f} is showing a {trend} trend with {day_change:+.1f}% movement today. RSI at {rsi:.1f} suggests {'strong momentum' if abs(day_change) > 2 else 'steady conditions'}.",

            "Should I buy?": f"With {symbol} at ${price:,.2f} and RSI at {rsi:.1f}, the technicals show {'overbought conditions - consider waiting for a pullback' if rsi > 70 else 'oversold conditions - could be a value opportunity' if rsi < 30 else 'balanced conditions'}. Always research fundamentals and consider your risk tolerance.",

            "Risk analysis": f"{symbol} shows {'high' if abs(day_change) > 3 else 'moderate' if abs(day_change) > 1 else 'low'} volatility today with {abs(day_change):.1f}% movement. RSI at {rsi:.1f} {'suggests extended conditions' if rsi > 70 or rsi < 30 else 'indicates normal momentum'}. Use proper position sizing and stop-losses.",

            "Compare with sector": f"{symbol}'s {day_change:+.1f}% move {'outperforms' if day_change > 1 else 'underperforms' if day_change < -1 else 'aligns with'} typical sector movements. RSI at {rsi:.1f} shows {'stronger' if rsi > 55 else 'weaker' if rsi < 45 else 'similar'} momentum compared to market averages."
        }

        return responses.get(action, f"Let me analyze {symbol} at ${price:,.2f} for you. What specific aspect would you like me to explain?")

    def _format_volume(self, volume: int) -> str:
        """Format volume for readable display"""
        if volume >= 1_000_000_000:
            return f"{volume / 1_000_000_000:.1f}B"
        elif volume >= 1_000_000:
            return f"{volume / 1_000_000:.1f}M"
        elif volume >= 1_000:
            return f"{volume / 1_000:.1f}K"
        else:
            return str(volume)

    def _get_market_context(self, market_status: str) -> str:
        """Get market context based on current status"""
        if market_status == "closed":
            return "MARKET CONTEXT: Markets are currently closed. Data reflects the last trading session."
        elif market_status == "open":
            return "MARKET CONTEXT: Markets are currently open and data is live."
        else:
            return "MARKET CONTEXT: Market status unknown. Data may be delayed."

    def _get_trend_description(self, trend: str, day_change: float) -> str:
        """Get descriptive trend text"""
        if trend == "bullish":
            return f"bullish (up {abs(day_change):.1f}% with positive momentum)"
        elif trend == "bearish":
            return f"bearish (down {abs(day_change):.1f}% with negative pressure)"
        else:
            return f"neutral (sideways movement, {day_change:+.1f}% today)"

    def get_error_response(self, error_type: str = "general") -> str:
        """Generate user-friendly error responses"""
        error_responses = {
            "api_limit": "I'm getting a lot of questions right now! Please wait a moment and try again.",
            "data_unavailable": "I'm having trouble getting the latest data for this stock. Please try again in a moment.",
            "invalid_symbol": "I couldn't find that stock symbol. Please check the spelling and try again.",
            "network_error": "I'm having connectivity issues. Please try again in a few seconds.",
            "general": "Something went wrong on my end. Please try your question again."
        }

        return error_responses.get(error_type, error_responses["general"])

    def get_welcome_message(self, symbol: str = None) -> str:
        """Generate welcome message"""
        if symbol:
            return f"Hi! I'm TradeLens AI, ready to help analyze {symbol} and answer your trading questions. What would you like to know?"
        else:
            return "Hi! I'm TradeLens AI, your personal trading assistant. I can help analyze stocks, explain indicators, and provide trading insights based on real-time data."

# For testing purposes
if __name__ == "__main__":
    manager = PromptTemplateManager()

    # Test context
    test_context = {
        'symbol': 'AAPL',
        'price': 220.45,
        'rsi': 65.2,
        'macd': 0.75,
        'dayChange': 2.1,
        'volume': 52000000,
        'market_status': 'open',
        'trend': 'bullish'
    }

    print("System Prompt:")
    print(manager.get_system_prompt(test_context))
    print("\n" + "="*50 + "\n")

    print("Quick Action Responses:")
    actions = ["What's the trend?", "Should I buy?", "Risk analysis", "Compare with sector"]
    for action in actions:
        response = manager.get_quick_action_responses(action, test_context)
        print(f"{action}: {response}")
        print()