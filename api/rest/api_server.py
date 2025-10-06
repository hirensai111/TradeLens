"""
Flask API Server for Stock Analyzer
Provides REST API endpoints for stock analysis and JSON data export
"""

import time
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
from datetime import datetime
from typing import Dict, Any
from flask import Flask, request, jsonify
from flask_cors import CORS

from core.config.config import config
from core.utils.utils import get_logger, get_market_hours_status
from api.rest.stock_analyzer_handler import StockAnalyzerHandler
from core.validators.validators import validate_ticker, ValidationError

# Import chat functionality
try:
    import openai
    from ai_assistant import ai_backend
    CHAT_AVAILABLE = True
except ImportError:
    CHAT_AVAILABLE = False
    print("Warning: OpenAI not available. Chat functionality will be limited.")


class StockAnalyzerAPI:
    """Flask API wrapper for Stock Analyzer functionality."""

    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app, origins=config.CORS_ORIGINS)

        self.logger = get_logger("StockAnalyzerAPI")
        self.stock_handler = StockAnalyzerHandler()

        # Free tier allowed stocks (from config)
        self.FREE_TIER_STOCKS = config.get_allowed_stocks()

        # Register routes
        self._register_routes()

        self.logger.info("Stock Analyzer API initialized")

    def _check_free_tier_access(self, ticker: str) -> tuple[bool, dict]:
        """
        Check if ticker is allowed in free tier.

        Returns:
            tuple: (is_allowed, error_response_dict)
        """
        # Empty list means premium tier - no restrictions
        if len(self.FREE_TIER_STOCKS) == 0:
            return True, {}

        if ticker.upper() not in self.FREE_TIER_STOCKS:
            return False, {
                'success': False,
                'error': 'Premium access required',
                'message': f'This stock is not available in the free tier. Available stocks: {self.FREE_TIER_STOCKS}',
                'available_stocks': self.FREE_TIER_STOCKS,
                'ticker': ticker,
                'timestamp': datetime.now().isoformat()
            }
        return True, {}

    def _load_json_file(self, file_path: str) -> Dict[str, Any]:
        """Load JSON data from file."""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"JSON file not found: {file_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading JSON file {file_path}: {e}")
            return {}

    def _generate_chat_response(self, message: str, ticker: str = None, context: str = "stock_analysis") -> str:
        """Generate AI chat response for stock analysis queries."""
        try:
            # Get stock context if ticker is provided
            stock_context = {}
            if ticker:
                try:
                    # Try to get existing stock data
                    result = self.stock_handler.get_stock_data(ticker)
                    if result.get('success'):
                        viz_data = result.get('viz_data', {})
                        price_data = self._load_json_file(viz_data.get('price_data', ''))
                        summary_data = self._load_json_file(viz_data.get('summary', ''))

                        # Extract key metrics for context
                        if summary_data:
                            stock_context = {
                                'symbol': ticker,
                                'price': summary_data.get('current_price', 0),
                                'rsi': summary_data.get('technical_indicators', {}).get('rsi', {}).get('value', 50),
                                'macd': summary_data.get('technical_indicators', {}).get('macd', {}).get('macd_line', 0),
                                'day_change': summary_data.get('key_metrics', {}).get('price_change_1d_pct', 0),
                                'volume': price_data.get('ohlcv', [{}])[-1].get('volume', 0) if price_data.get('ohlcv') else 0
                            }
                except Exception as e:
                    self.logger.warning(f"Could not get stock context for {ticker}: {e}")

            # Generate intelligent response based on the query
            response = self._get_intelligent_response(message.lower(), stock_context)
            return response

        except Exception as e:
            self.logger.error(f"Error generating chat response: {e}")
            return "I'm experiencing some technical difficulties. Please try again in a moment."

    def _get_intelligent_response(self, message: str, context: Dict[str, Any]) -> str:
        """Generate intelligent response based on message and stock context."""
        symbol = context.get('symbol', 'the stock')
        price = context.get('price', 0)
        rsi = context.get('rsi', 50)
        macd = context.get('macd', 0)
        day_change = context.get('day_change', 0)

        # Educational "what is" questions first
        if any(phrase in message for phrase in ['what is rsi', 'what\'s rsi', 'explain rsi', 'rsi means']):
            return "RSI (Relative Strength Index) measures momentum on a 0-100 scale. Think of it like a speedometer for stocks - above 70 means overbought (going too fast), below 30 means oversold (going too slow), and 30-70 is the normal cruising range."

        elif any(phrase in message for phrase in ['what is macd', 'what\'s macd', 'explain macd', 'macd means']):
            return "MACD (Moving Average Convergence Divergence) shows the relationship between two moving averages. Like two cars on a highway - when the faster one (MACD line) is above the slower one (signal line), it suggests upward momentum, and vice versa."

        elif any(phrase in message for phrase in ['what is sma', 'what\'s sma', 'explain sma', 'sma means', 'simple moving average']):
            return "SMA (Simple Moving Average) is the average price over a set period, like the last 20 days. Think of it as the stock's 'normal' price - when current price is above SMA, it's running hot; below SMA means it's running cool."

        # RSI analysis questions (not educational)
        elif ('rsi' in message or 'relative strength' in message) and not any(phrase in message for phrase in ['what is', 'what\'s', 'explain', 'means']):
            if rsi > 70:
                return f"RSI at {rsi:.1f} means {symbol} is overbought - like a car going too fast. It might slow down soon. Consider waiting for a pullback."
            elif rsi < 30:
                return f"RSI at {rsi:.1f} means {symbol} is oversold - like a stretched rubber band ready to snap back. Could be a buying opportunity if fundamentals are strong."
            else:
                return f"RSI at {rsi:.1f} is in the neutral zone - {symbol} has room to move either direction. This is a balanced momentum level."

        # Trend questions
        elif any(word in message for word in ['trend', 'direction', 'going']):
            if day_change > 2:
                return f"{symbol} is trending up strongly at ${price:.2f}, up {day_change:.1f}% today. The momentum looks positive."
            elif day_change < -2:
                return f"{symbol} is trending down at ${price:.2f}, down {abs(day_change):.1f}% today. Watch for support levels."
            else:
                return f"{symbol} at ${price:.2f} is moving sideways today with {day_change:+.1f}% change. Look for breakout signals."

        # Buy/investment questions
        elif any(word in message for word in ['buy', 'purchase', 'invest', 'should i']):
            if rsi < 30:
                return f"With RSI at {rsi:.1f}, {symbol} might be oversold at ${price:.2f}. Consider dollar-cost averaging if you believe in the company."
            elif rsi > 70:
                return f"RSI at {rsi:.1f} suggests {symbol} is overbought at ${price:.2f}. Maybe wait for a pullback."
            else:
                return f"{symbol} at ${price:.2f} with RSI of {rsi:.1f} shows balanced momentum. Do your research and consider your risk tolerance."

        # Risk questions
        elif any(word in message for word in ['risk', 'danger', 'safe']):
            volatility = "high" if abs(day_change) > 3 else "moderate" if abs(day_change) > 1 else "low"
            return f"{symbol} shows {volatility} volatility today with {abs(day_change):.1f}% movement. Always use stop-losses and position sizing."

        # Technical analysis questions
        elif any(word in message for word in ['technical', 'indicator', 'macd', 'signal']):
            return f"For {symbol} at ${price:.2f}, check the charts on this dashboard for technical indicators like RSI ({rsi:.1f}), MACD, and moving averages."

        # Comparison questions
        elif any(word in message for word in ['compare', 'sector', 'peers']):
            return f"To compare {symbol} with its sector, look at relative performance, P/E ratios, growth rates, and competitive advantages. Sector ETFs provide good benchmarks."

        # Default response
        else:
            if symbol != 'the stock':
                return f"I'm here to help analyze {symbol} at ${price:.2f}. I can discuss trends, RSI signals, risk factors, and investment considerations. What would you like to know?"
            else:
                return "I'm here to help with stock analysis and trading insights. I can discuss trends, technical indicators, risk analysis, and investment considerations. What specific aspect would you like to explore?"

    def _register_routes(self):
        """Register all API routes."""

        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            market_status = get_market_hours_status()
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': config.VERSION,
                'market_status': market_status
            })

        @self.app.route('/api/analyze/<ticker>', methods=['POST'])
        def analyze_stock(ticker: str):
            """
            Analyze stock and return JSON response with data paths.

            Args:
                ticker: Stock ticker symbol

            Returns:
                JSON response with analysis status and file paths
            """
            start_time = time.time()

            try:
                self.logger.info(f"API request received for ticker: {ticker}")

                # Validate ticker
                ticker = validate_ticker(ticker)

                # Check free tier access
                is_allowed, error_response = self._check_free_tier_access(ticker)
                if not is_allowed:
                    return jsonify(error_response), 403

                # Check for force refresh parameter
                force_refresh = request.json.get('force_refresh', False) if request.is_json else False

                # Perform analysis
                result = self.stock_handler.analyze_stock(
                    ticker,
                    force_refresh=force_refresh
                )

                analysis_time = time.time() - start_time

                if result['success']:
                    # Load JSON data from files
                    viz_data_paths = result.get('viz_data', {})

                    # Load the actual JSON data
                    price_data = self._load_json_file(viz_data_paths.get('price_data', ''))
                    company_data = self._load_json_file(viz_data_paths.get('company', ''))
                    events_data = self._load_json_file(viz_data_paths.get('events', ''))
                    summary_data = self._load_json_file(viz_data_paths.get('summary', ''))

                    # Get market status
                    market_status = get_market_hours_status()

                    # Prepare successful response with loaded data
                    response_data = {
                        'success': True,
                        'data': {
                            'price_data': price_data,
                            'company_data': company_data,
                            'events_data': events_data,
                            'summary_data': summary_data
                        },
                        'ticker': ticker,
                        'analysis_time_seconds': round(analysis_time, 2),
                        'timestamp': datetime.now().isoformat(),
                        'market_status': market_status
                    }

                    self.logger.info(f"Analysis successful for {ticker} in {analysis_time:.2f}s")
                    return jsonify(response_data), 200

                else:
                    # Handle analysis failure
                    error_response = {
                        'success': False,
                        'ticker': ticker,
                        'error': result.get('error', 'Unknown error occurred'),
                        'analysis_time_seconds': round(analysis_time, 2),
                        'timestamp': datetime.now().isoformat()
                    }

                    self.logger.error(f"Analysis failed for {ticker}: {result.get('error')}")
                    return jsonify(error_response), 500

            except ValidationError as e:
                self.logger.warning(f"Invalid ticker {ticker}: {e}")
                return jsonify({
                    'success': False,
                    'error': f'Invalid ticker symbol: {e}',
                    'ticker': ticker,
                    'timestamp': datetime.now().isoformat()
                }), 400

            except Exception as e:
                analysis_time = time.time() - start_time
                self.logger.error(f"Unexpected error analyzing {ticker}: {e}")
                return jsonify({
                    'success': False,
                    'error': f'Internal server error: {str(e)}',
                    'ticker': ticker,
                    'analysis_time_seconds': round(analysis_time, 2),
                    'timestamp': datetime.now().isoformat()
                }), 500

        @self.app.route('/api/status/<ticker>', methods=['GET'])
        def check_data_status(ticker: str):
            """
            Check the freshness status of cached data for a ticker.

            Args:
                ticker: Stock ticker symbol

            Returns:
                JSON response with data freshness information
            """
            try:
                ticker = validate_ticker(ticker)

                # Check free tier access
                is_allowed, error_response = self._check_free_tier_access(ticker)
                if not is_allowed:
                    return jsonify(error_response), 403

                # Check data freshness
                freshness_info = self.stock_handler.check_data_freshness(ticker)

                return jsonify({
                    'success': True,
                    'ticker': ticker,
                    'data_status': freshness_info.get('status', 'unknown'),
                    'last_updated': freshness_info.get('last_updated'),
                    'cache_age_hours': freshness_info.get('cache_age_hours'),
                    'needs_update': freshness_info.get('needs_update', True),
                    'timestamp': datetime.now().isoformat()
                }), 200

            except ValidationError as e:
                return jsonify({
                    'success': False,
                    'error': f'Invalid ticker symbol: {e}',
                    'ticker': ticker,
                    'timestamp': datetime.now().isoformat()
                }), 400

            except Exception as e:
                self.logger.error(f"Error checking status for {ticker}: {e}")
                return jsonify({
                    'success': False,
                    'error': f'Internal server error: {str(e)}',
                    'ticker': ticker,
                    'timestamp': datetime.now().isoformat()
                }), 500

        @self.app.route('/api/data/<ticker>', methods=['GET'])
        def get_stock_data(ticker: str):
            """
            Get existing JSON data for a ticker without running full analysis.

            Args:
                ticker: Stock ticker symbol

            Returns:
                JSON response with existing data or error if not found
            """
            try:
                ticker = validate_ticker(ticker)

                # Check free tier access
                is_allowed, error_response = self._check_free_tier_access(ticker)
                if not is_allowed:
                    return jsonify(error_response), 403

                # Check if JSON files exist
                viz_data_dir = config.VIZ_DATA_DIR
                files = {
                    'price_data': viz_data_dir / f"{ticker}_price_data.json",
                    'company': viz_data_dir / f"{ticker}_company.json",
                    'events': viz_data_dir / f"{ticker}_events.json",
                    'summary': viz_data_dir / f"{ticker}_summary.json"
                }

                # Check if all files exist
                missing_files = [name for name, path in files.items() if not path.exists()]
                if missing_files:
                    return jsonify({
                        'success': False,
                        'error': f'Data not found for {ticker}. Missing files: {missing_files}',
                        'ticker': ticker,
                        'timestamp': datetime.now().isoformat()
                    }), 404

                # Load all JSON data
                data = {}
                for name, path in files.items():
                    data[name] = self._load_json_file(str(path))

                return jsonify({
                    'success': True,
                    'data': data,
                    'ticker': ticker,
                    'timestamp': datetime.now().isoformat()
                }), 200

            except ValidationError as e:
                return jsonify({
                    'success': False,
                    'error': f'Invalid ticker symbol: {e}',
                    'ticker': ticker,
                    'timestamp': datetime.now().isoformat()
                }), 400

            except Exception as e:
                self.logger.error(f"Error getting data for {ticker}: {e}")
                return jsonify({
                    'success': False,
                    'error': f'Internal server error: {str(e)}',
                    'ticker': ticker,
                    'timestamp': datetime.now().isoformat()
                }), 500

        @self.app.route('/api/config', methods=['GET'])
        def get_api_config():
            """Get API configuration information."""
            return jsonify({
                'api_version': config.VERSION,
                'cache_freshness_hours': config.CACHE_FRESHNESS_HOURS,
                'supported_data_sources': config.DATA_SOURCES,
                'api_settings': {
                    'timeout_seconds': config.API_TIMEOUT_SECONDS,
                    'max_concurrent_analysis': config.MAX_CONCURRENT_ANALYSIS
                },
                'data_period_years': config.DATA_PERIOD_YEARS,
                'timestamp': datetime.now().isoformat()
            })

        @self.app.route('/api/market-status', methods=['GET'])
        def get_market_status():
            """Get current market status and hours."""
            market_status = get_market_hours_status()
            return jsonify({
                'success': True,
                'market_status': market_status,
                'timestamp': datetime.now().isoformat()
            })

        @self.app.route('/api/free-tier-stocks', methods=['GET'])
        def get_free_tier_stocks():
            """Get list of stocks available in the free tier."""
            return jsonify({
                'success': True,
                'available_stocks': self.FREE_TIER_STOCKS,
                'count': len(self.FREE_TIER_STOCKS),
                'message': 'These stocks are available in the free tier with cached data',
                'timestamp': datetime.now().isoformat()
            })

        @self.app.route('/chat', methods=['POST'])
        def chat():
            """Chat endpoint for AI-powered stock analysis assistance."""
            try:
                # Get request data
                data = request.get_json()
                if not data:
                    return jsonify({
                        'success': False,
                        'error': 'No data provided'
                    }), 400

                user_message = data.get('message', '').strip()
                ticker = data.get('ticker', '').strip().upper()
                context = data.get('context', 'stock_analysis')

                if not user_message:
                    return jsonify({
                        'success': False,
                        'error': 'No message provided'
                    }), 400

                # Generate AI response
                ai_response = self._generate_chat_response(user_message, ticker, context)

                return jsonify({
                    'success': True,
                    'data': {
                        'response': ai_response
                    },
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                self.logger.error(f"Error in chat endpoint: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': 'Failed to process chat message',
                    'timestamp': datetime.now().isoformat()
                }), 500

        @self.app.errorhandler(404)
        def not_found(error):
            """Handle 404 errors."""
            return jsonify({
                'success': False,
                'error': 'Endpoint not found',
                'available_endpoints': [
                    'GET /health',
                    'GET /api/data/<ticker>',
                    'POST /api/analyze/<ticker>',
                    'GET /api/status/<ticker>',
                    'GET /api/config',
                    'GET /api/market-status',
                    'GET /api/free-tier-stocks',
                    'POST /chat'
                ],
                'timestamp': datetime.now().isoformat()
            }), 404

        @self.app.errorhandler(405)
        def method_not_allowed(error):
            """Handle 405 errors."""
            return jsonify({
                'success': False,
                'error': 'Method not allowed',
                'timestamp': datetime.now().isoformat()
            }), 405

        @self.app.errorhandler(500)
        def internal_error(error):
            """Handle 500 errors."""
            self.logger.error(f"Internal server error: {error}")
            return jsonify({
                'success': False,
                'error': 'Internal server error',
                'timestamp': datetime.now().isoformat()
            }), 500

    def run(self, host: str = None, port: int = None, debug: bool = None):
        """
        Run the Flask development server.

        Args:
            host: Host to bind to (defaults to config.API_HOST)
            port: Port to bind to (defaults to config.API_PORT)
            debug: Enable debug mode (defaults to config.API_DEBUG)
        """
        host = host or config.API_HOST
        port = port or config.API_PORT
        debug = debug if debug is not None else config.API_DEBUG

        self.logger.info(f"Starting API server on {host}:{port} (debug={debug})")

        try:
            self.app.run(
                host=host,
                port=port,
                debug=debug,
                threaded=True
            )
        except Exception as e:
            self.logger.error(f"Failed to start API server: {e}")
            raise


def create_app() -> Flask:
    """
    Application factory for creating Flask app instance.

    Returns:
        Configured Flask application
    """
    api = StockAnalyzerAPI()
    return api.app


def main():
    """Main entry point for running the API server."""
    import argparse

    parser = argparse.ArgumentParser(description='Stock Analyzer API Server')
    parser.add_argument('--host', default=config.API_HOST, help='Host to bind to')
    parser.add_argument('--port', type=int, default=config.API_PORT, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    # Create and run API
    api = StockAnalyzerAPI()
    api.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()