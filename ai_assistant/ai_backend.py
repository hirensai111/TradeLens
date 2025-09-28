"""
TradeLens AI Backend - Flask API endpoint for chat functionality
Integrates with existing stock analyzer and provides AI-powered responses
"""

from flask import Flask, request, jsonify, session
from flask_cors import CORS
import openai
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from context_manager import StockContextManager
    from prompt_templates import PromptTemplateManager
except ImportError:
    # Fallback if modules aren't available yet
    print("Warning: context_manager or prompt_templates not found. Some features may be limited.")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'tradeLens-ai-secret-key-2024')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradeLensAI:
    def __init__(self):
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not self.openai_api_key:
            logger.warning("OpenAI API key not found. AI responses will be simulated.")

        openai.api_key = self.openai_api_key

        # Initialize context and prompt managers
        try:
            self.context_manager = StockContextManager()
            self.prompt_manager = PromptTemplateManager()
        except NameError:
            self.context_manager = None
            self.prompt_manager = None
            logger.warning("Context or Prompt managers not available. Using fallback responses.")

        self.max_session_messages = 50
        self.model = "gpt-3.5-turbo"

    def get_or_create_session_id(self) -> str:
        """Get or create a session ID for message tracking"""
        if 'session_id' not in session:
            session['session_id'] = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            session['message_count'] = 0
            session['chat_history'] = []
        return session['session_id']

    def add_to_session_history(self, message: str, response: str):
        """Add message and response to session history"""
        if 'chat_history' not in session:
            session['chat_history'] = []

        session['chat_history'].append({
            'user_message': message,
            'ai_response': response,
            'timestamp': datetime.now().isoformat()
        })

        # Limit session history
        if len(session['chat_history']) > self.max_session_messages:
            session['chat_history'] = session['chat_history'][-self.max_session_messages:]

    def check_rate_limit(self) -> tuple[bool, str]:
        """Check if user has exceeded rate limits"""
        session['message_count'] = session.get('message_count', 0) + 1

        if session['message_count'] > self.max_session_messages:
            return False, "You've reached the maximum number of messages for this session. Please refresh to start a new session."

        return True, ""

    def generate_ai_response(self, user_message: str, context: Dict[str, Any], history: List[Dict]) -> str:
        """Generate AI response using OpenAI API or fallback"""
        try:
            if not self.openai_api_key:
                return self.generate_fallback_response(user_message, context)

            # Build system prompt with context
            system_prompt = self.build_system_prompt(context)

            # Build conversation history
            messages = [{"role": "system", "content": system_prompt}]

            # Add recent history for context
            for msg in history[-6:]:  # Last 6 messages
                messages.append({"role": "user", "content": msg.get('content', '')})

            # Add current message
            messages.append({"role": "user", "content": user_message})

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=150,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.5,
                presence_penalty=0.3
            )

            ai_response = response.choices[0].message.content.strip()

            # Ensure response is under 3 sentences
            sentences = ai_response.split('. ')
            if len(sentences) > 3:
                ai_response = '. '.join(sentences[:3]) + '.'

            return ai_response

        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            return self.generate_fallback_response(user_message, context)

    def build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build system prompt with current stock context"""
        if self.prompt_manager:
            return self.prompt_manager.get_system_prompt(context)

        # Fallback system prompt
        symbol = context.get('symbol', 'UNKNOWN')
        price = context.get('price', 0)
        rsi = context.get('rsi', 50)
        macd = context.get('macd', 0)
        day_change = context.get('dayChange', 0)
        volume = context.get('volume', 0)

        return f"""
You are TradeLens AI, a friendly stock analysis assistant. You have access to real-time data for {symbol}.

Current Data:
- Price: ${price}
- RSI: {rsi}
- MACD: {macd}
- Day Change: {day_change}%
- Volume: {volume:,}

Guidelines:
1. Keep responses under 3 sentences
2. Use simple language (explain like to a friend)
3. Reference the actual numbers they're seeing
4. Be specific and actionable
5. Use analogies to explain complex concepts
6. Never give financial advice - only educational information

Example: "{symbol} at ${price} with RSI of {rsi} shows good momentum - like a car at comfortable highway speed. The price is {'up' if day_change > 0 else 'down'} {abs(day_change):.1f}% today on {'higher' if volume > 50000000 else 'normal'} volume, suggesting {'real buying interest' if day_change > 0 else 'some selling pressure'}."
"""

    def generate_fallback_response(self, user_message: str, context: Dict[str, Any]) -> str:
        """Generate fallback response when OpenAI is not available"""
        symbol = context.get('symbol', 'UNKNOWN')
        price = context.get('price', 0)
        rsi = context.get('rsi', 50)
        day_change = context.get('dayChange', 0)

        user_lower = user_message.lower()

        if any(word in user_lower for word in ['trend', 'direction', 'going']):
            if day_change > 2:
                return f"{symbol} is trending up strongly at ${price}, up {day_change:.1f}% today. The momentum looks positive."
            elif day_change < -2:
                return f"{symbol} is trending down at ${price}, down {abs(day_change):.1f}% today. Watch for support levels."
            else:
                return f"{symbol} at ${price} is moving sideways today with {day_change:+.1f}% change. Look for breakout signals."

        elif any(word in user_lower for word in ['buy', 'purchase', 'invest']):
            if rsi < 30:
                return f"With RSI at {rsi}, {symbol} might be oversold at ${price}. Consider dollar-cost averaging if you believe in the company."
            elif rsi > 70:
                return f"RSI at {rsi} suggests {symbol} is overbought at ${price}. Maybe wait for a pullback."
            else:
                return f"{symbol} at ${price} with RSI of {rsi} shows balanced momentum. Do your research and consider your risk tolerance."

        elif any(word in user_lower for word in ['rsi', 'indicator', 'technical']):
            if rsi > 70:
                return f"RSI at {rsi} means {symbol} is overbought - like a car going too fast. It might slow down soon."
            elif rsi < 30:
                return f"RSI at {rsi} means {symbol} is oversold - like a stretched rubber band ready to snap back."
            else:
                return f"RSI at {rsi} is in the neutral zone - {symbol} has room to move either direction."

        elif any(word in user_lower for word in ['risk', 'danger', 'safe']):
            volatility = "high" if abs(day_change) > 3 else "moderate" if abs(day_change) > 1 else "low"
            return f"{symbol} shows {volatility} volatility today with {abs(day_change):.1f}% movement. Always use stop-losses and position sizing."

        else:
            return f"{symbol} is trading at ${price} with RSI of {rsi}. I'm here to help explain what these numbers mean for your analysis."

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        user_message = data.get('message', '').strip()
        context = data.get('context', {})
        history = data.get('history', [])

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Initialize AI instance
        ai = TradeLensAI()

        # Check rate limits
        is_allowed, rate_limit_msg = ai.check_rate_limit()
        if not is_allowed:
            return jsonify({'response': rate_limit_msg}), 429

        # Get session ID
        session_id = ai.get_or_create_session_id()

        # Enhance context if context manager is available
        if ai.context_manager and context.get('symbol'):
            try:
                enhanced_context = ai.context_manager.get_enhanced_context(context['symbol'])
                context.update(enhanced_context)
            except Exception as e:
                logger.warning(f"Failed to enhance context: {str(e)}")

        # Generate AI response
        ai_response = ai.generate_ai_response(user_message, context, history)

        # Add to session history
        ai.add_to_session_history(user_message, ai_response)

        # Log the interaction
        logger.info(f"Session {session_id}: User: {user_message[:50]}... | AI: {ai_response[:50]}...")

        return jsonify({
            'response': ai_response,
            'session_id': session_id,
            'message_count': session.get('message_count', 0),
            'context_used': context.get('symbol', 'unknown')
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'response': "I'm experiencing some technical difficulties. Please try again in a moment.",
            'error': True
        }), 500

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """Clear chat history for current session"""
    try:
        session.pop('chat_history', None)
        session['message_count'] = 0
        return jsonify({'message': 'Chat history cleared successfully'})
    except Exception as e:
        logger.error(f"Error clearing chat: {str(e)}")
        return jsonify({'error': 'Failed to clear chat'}), 500

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """Get chat history for current session"""
    try:
        history = session.get('chat_history', [])
        return jsonify({
            'history': history,
            'message_count': session.get('message_count', 0),
            'session_id': session.get('session_id', 'unknown')
        })
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        return jsonify({'error': 'Failed to get chat history'}), 500

@app.route('/api/chat/status', methods=['GET'])
def get_chat_status():
    """Get chat service status"""
    ai = TradeLensAI()
    return jsonify({
        'status': 'online',
        'openai_available': bool(ai.openai_api_key),
        'context_manager_available': bool(ai.context_manager),
        'prompt_manager_available': bool(ai.prompt_manager),
        'max_messages': ai.max_session_messages
    })

if __name__ == '__main__':
    # Set environment variables for development
    if not os.environ.get('OPENAI_API_KEY'):
        print("\nWarning: OPENAI_API_KEY not set. AI responses will be simulated.")
        print("To use real AI responses, set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print()

    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5001)