#!/usr/bin/env python3
"""
Zerodha Kite Connect API Client for Indian Trading Bot
Handles all interactions with Zerodha Kite Connect API for market data and trading
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
from kiteconnect import KiteConnect
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass  # dotenv not installed, skip

logger = logging.getLogger(__name__)

class ZerodhaAPIClient:
    """Client for interacting with Zerodha Kite Connect API"""
    
    def __init__(self, api_key: str = None, access_token: str = None):
        """Initialize Zerodha Kite Connect API client"""
        
        # Auto-load from environment variables if not provided
        api_key = api_key or os.getenv("ZERODHA_API_KEY")
        access_token = access_token or os.getenv("ZERODHA_ACCESS_TOKEN")
        
        if not api_key:
            raise ValueError("API key is required. Set ZERODHA_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize KiteConnect
        self.kite = KiteConnect(api_key=api_key)
        
        if access_token:
            self.kite.set_access_token(access_token)
            self.access_token = access_token
        else:
            # If no access token, you'll need to generate one using login flow
            logger.warning("No access token provided. You'll need to complete the login flow to get access token.")
            self.access_token = None
        
        # Exchange constants
        self.NSE = self.kite.EXCHANGE_NSE
        self.BSE = self.kite.EXCHANGE_BSE
        self.NFO = self.kite.EXCHANGE_NFO  # NSE F&O
        self.BFO = self.kite.EXCHANGE_BFO  # BSE F&O
        self.MCX = self.kite.EXCHANGE_MCX
        
        # Order types
        self.MARKET = self.kite.ORDER_TYPE_MARKET
        self.LIMIT = self.kite.ORDER_TYPE_LIMIT
        self.SL = self.kite.ORDER_TYPE_SL
        self.SLM = self.kite.ORDER_TYPE_SLM
        
        # Product types (with fallback for different KiteConnect versions)
        self.CNC = getattr(self.kite, 'PRODUCT_CNC', 'CNC')
        self.MIS = getattr(self.kite, 'PRODUCT_MIS', 'MIS')  # Intraday
        self.NRML = getattr(self.kite, 'PRODUCT_NRML', 'NRML')  # Normal/Carryforward
        self.BO = getattr(self.kite, 'PRODUCT_BO', 'BO')  # Bracket Order
        self.CO = getattr(self.kite, 'PRODUCT_CO', 'CO')  # Cover Order
        
        # Transaction types
        self.BUY = self.kite.TRANSACTION_TYPE_BUY
        self.SELL = self.kite.TRANSACTION_TYPE_SELL
        
        # Validity types (with fallback for different KiteConnect versions)
        self.DAY = getattr(self.kite, 'VALIDITY_DAY', 'DAY')
        self.IOC = getattr(self.kite, 'VALIDITY_IOC', 'IOC')
        self.TTL = getattr(self.kite, 'VALIDITY_TTL', 'TTL')
        
        # Position types (with fallback for different KiteConnect versions)
        self.OVERNIGHT = getattr(self.kite, 'POSITION_TYPE_OVERNIGHT', 'overnight')
        self.DAY_POSITION = getattr(self.kite, 'POSITION_TYPE_DAY', 'day')
        
        # Load instruments data
        self._load_instruments()
        
        logger.info("Zerodha Kite Connect API client initialized successfully")
    
    def generate_login_url(self) -> str:
        """Generate login URL for getting request token"""
        if not hasattr(self.kite, 'api_key'):
            raise ValueError("API key not set")
        return f"https://kite.zerodha.com/connect/login?api_key={self.kite.api_key}&v=3"
    
    def generate_session(self, request_token: str, api_secret: str) -> Dict:
        """Generate session using request token and API secret"""
        try:
            data = self.kite.generate_session(request_token, api_secret=api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            logger.info("✅ Session generated successfully")
            return data
        except Exception as e:
            logger.error(f"Error generating session: {e}")
            raise
        
    def get_profile(self):
        """Get user profile"""
        try:
            return self.kite.profile()
        except Exception as e:
            logger.error(f"Error getting profile: {e}")
            return None
    
    def _load_instruments(self):
        """Load instruments data for symbol to instrument token mapping"""
        
        try:
            if not self.access_token:
                logger.warning("No access token available. Cannot load instruments.")
                self._create_fallback_mapping()
                return
            
            # Download instruments for NSE and NFO
            nse_instruments = self.kite.instruments(self.NSE)
            nfo_instruments = self.kite.instruments(self.NFO)
            
            # Combine all instruments
            all_instruments = nse_instruments + nfo_instruments
            
            # Create symbol mapping
            self.instruments_df = pd.DataFrame(all_instruments)
            self.symbol_map = {}
            
            for instrument in all_instruments:
                symbol = instrument['tradingsymbol'].upper()
                self.symbol_map[symbol] = {
                    'instrument_token': instrument['instrument_token'],
                    'exchange_token': instrument['exchange_token'],
                    'exchange': instrument['exchange'],
                    'segment': instrument['segment'],
                    'lot_size': instrument['lot_size'],
                    'tick_size': instrument['tick_size'],
                    'name': instrument['name'],
                    'expiry': instrument.get('expiry'),
                    'strike': instrument.get('strike'),
                    'instrument_type': instrument.get('instrument_type')
                }
            
            logger.info(f"Loaded {len(self.symbol_map)} instruments")
            
        except Exception as e:
            logger.error(f"Failed to load instruments: {e}")
            self._create_fallback_mapping()
    
    def _create_fallback_mapping(self):
        """Create fallback symbol mapping for major stocks"""
        self.symbol_map = {
            # Existing stocks
            'RELIANCE': {'instrument_token': 738561, 'exchange': 'NSE', 'segment': 'NSE', 'lot_size': 1},
            'TCS': {'instrument_token': 2953217, 'exchange': 'NSE', 'segment': 'NSE', 'lot_size': 1},
            'HDFCBANK': {'instrument_token': 341249, 'exchange': 'NSE', 'segment': 'NSE', 'lot_size': 1},
            'INFY': {'instrument_token': 408065, 'exchange': 'NSE', 'segment': 'NSE', 'lot_size': 1},
            'HINDUNILVR': {'instrument_token': 356865, 'exchange': 'NSE', 'segment': 'NSE', 'lot_size': 1},
            'ITC': {'instrument_token': 424961, 'exchange': 'NSE', 'segment': 'NSE', 'lot_size': 1},
            'SBIN': {'instrument_token': 779521, 'exchange': 'NSE', 'segment': 'NSE', 'lot_size': 1},
            'BHARTIARTL': {'instrument_token': 2714625, 'exchange': 'NSE', 'segment': 'NSE', 'lot_size': 1},
            
            # ADD THESE NEW STOCKS:
            'BAJFINANCE': {'instrument_token': 81153, 'exchange': 'NSE', 'segment': 'NSE', 'lot_size': 1},
            'MARUTI': {'instrument_token': 519937, 'exchange': 'NSE', 'segment': 'NSE', 'lot_size': 1},
            'ICICIBANK': {'instrument_token': 1270529, 'exchange': 'NSE', 'segment': 'NSE', 'lot_size': 1},
            'LT': {'instrument_token': 2939649, 'exchange': 'NSE', 'segment': 'NSE', 'lot_size': 1},
            'HCLTECH': {'instrument_token': 1850625, 'exchange': 'NSE', 'segment': 'NSE', 'lot_size': 1},
            'WIPRO': {'instrument_token': 3050241, 'exchange': 'NSE', 'segment': 'NSE', 'lot_size': 1},
            'TECHM': {'instrument_token': 3465729, 'exchange': 'NSE', 'segment': 'NSE', 'lot_size': 1},
            'ASIANPAINT': {'instrument_token': 60417, 'exchange': 'NSE', 'segment': 'NSE', 'lot_size': 1},
            'MPHASIS': {'instrument_token': 2815745, 'exchange': 'NSE', 'segment': 'NSE', 'lot_size': 1},
            'ADANIPORTS': {'instrument_token': 3861249, 'exchange': 'NSE', 'segment': 'NSE', 'lot_size': 1},
            'TATAMOTORS': {'instrument_token': 884737, 'exchange': 'NSE', 'segment': 'NSE', 'lot_size': 1},
            
            # Indices (unchanged)
            'NIFTY 50': {'instrument_token': 256265, 'exchange': 'NSE', 'segment': 'INDICES', 'lot_size': 50},
            'NIFTY BANK': {'instrument_token': 260105, 'exchange': 'NSE', 'segment': 'INDICES', 'lot_size': 25},
            'NIFTY FIN SERVICE': {'instrument_token': 257801, 'exchange': 'NSE', 'segment': 'INDICES', 'lot_size': 40}
        }
        logger.warning("Using fallback symbol mapping")
    
    def get_instrument_info(self, symbol: str) -> Optional[Dict]:
        """Get instrument info from symbol"""
        return self.symbol_map.get(symbol.upper())
    
    def get_live_quotes(self, symbols: List[str]) -> Dict:
        """Get real-time quotes for multiple symbols"""
        
        try:
            if not self.access_token:
                logger.error("Access token not available")
                return {}
            
            # Get instrument tokens for symbols
            instrument_tokens = []
            symbol_token_map = {}
            
            for symbol in symbols:
                instrument_info = self.get_instrument_info(symbol)
                if instrument_info and 'instrument_token' in instrument_info:
                    token = instrument_info['instrument_token']
                    instrument_tokens.append(token)
                    symbol_token_map[str(token)] = symbol  # Ensure string key for mapping
                else:
                    logger.warning(f"Instrument token not found for {symbol}")
                    # Try alternative symbol formats for indices
                    if symbol in ['NIFTY 50', 'NIFTY BANK', 'NIFTY FIN SERVICE']:
                        alt_symbol = symbol.replace(' ', '')  # Try without space
                        alt_info = self.get_instrument_info(alt_symbol)
                        if alt_info and 'instrument_token' in alt_info:
                            token = alt_info['instrument_token']
                            instrument_tokens.append(token)
                            symbol_token_map[str(token)] = symbol
            
            if not instrument_tokens:
                logger.warning("No valid instrument tokens found for any symbols")
                return {}
            
            logger.info(f"Fetching quotes for {len(instrument_tokens)} instruments")
            
            # Get quotes from Kite API
            quotes_response = self.kite.quote(instrument_tokens)
            
            quotes = {}
            for token, quote_data in quotes_response.items():
                # Map back to original symbol using string token
                symbol = symbol_token_map.get(str(token))
                
                if symbol and quote_data:
                    ohlc = quote_data.get('ohlc', {})
                    depth = quote_data.get('depth', {})
                    
                    quotes[symbol] = {
                        'price': quote_data.get('last_price', 0),
                        'open': ohlc.get('open', 0),
                        'high': ohlc.get('high', 0),
                        'low': ohlc.get('low', 0),
                        'close': ohlc.get('close', 0),
                        'volume': quote_data.get('volume', 0),
                        'change': quote_data.get('net_change', 0),
                        'change_percent': quote_data.get('oi_day_change_percentage', 0),
                        'bid': depth.get('buy', [{}])[0].get('price', 0) if depth.get('buy') else 0,
                        'ask': depth.get('sell', [{}])[0].get('price', 0) if depth.get('sell') else 0,
                        'oi': quote_data.get('oi', 0),
                        'oi_change': quote_data.get('oi_day_change', 0)
                    }
                else:
                    if not symbol:
                        logger.warning(f"Could not map token {token} back to symbol")
            
            logger.info(f"Successfully fetched quotes for {len(quotes)} symbols")
            return quotes
            
        except Exception as e:
            logger.error(f"Error fetching live quotes: {e}")
            return {}
    
    def get_ltp(self, symbols: List[str]) -> Dict:
        """Get Last Traded Price for symbols"""
        
        try:
            if not self.access_token:
                logger.error("Access token not available")
                return {}
            
            # Get instrument tokens
            instrument_tokens = []
            symbol_token_map = {}
            
            for symbol in symbols:
                instrument_info = self.get_instrument_info(symbol)
                if instrument_info and 'instrument_token' in instrument_info:
                    token = instrument_info['instrument_token']
                    instrument_tokens.append(token)
                    symbol_token_map[token] = symbol
            
            if not instrument_tokens:
                return {}
            
            # Get LTP from Kite API
            ltp_response = self.kite.ltp(instrument_tokens)
            
            result = {}
            for token, ltp_data in ltp_response.items():
                # Map back to original symbol
                symbol = None
                for orig_token, orig_symbol in symbol_token_map.items():
                    if str(orig_token) == str(token):
                        symbol = orig_symbol
                        break
                
                if symbol and ltp_data:
                    result[symbol] = ltp_data.get('last_price', 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching LTP: {e}")
            return {}
    
    def get_historical_data(self, symbol: str, timeframe: str = 'day', days: int = 100) -> pd.DataFrame:
        """Get historical OHLCV data"""
        
        try:
            if not self.access_token:
                logger.error("Access token not available")
                return pd.DataFrame()
            
            instrument_info = self.get_instrument_info(symbol)
            if not instrument_info or 'instrument_token' not in instrument_info:
                logger.error(f"Instrument token not found for {symbol}")
                return pd.DataFrame()
            
            instrument_token = instrument_info['instrument_token']
            
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            # Map timeframe to Kite interval
            interval_map = {
                'minute': 'minute',
                '3minute': '3minute',
                '5minute': '5minute',
                '10minute': '10minute',
                '15minute': '15minute',
                '30minute': '30minute',
                'hour': '60minute',
                'day': 'day'
            }
            
            interval = interval_map.get(timeframe, 'day')
            
            # Fetch historical data
            historical_data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            if historical_data:
                # Convert to DataFrame
                df = pd.DataFrame(historical_data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df[['open', 'high', 'low', 'close', 'volume']]
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def get_market_status(self) -> Dict:
        """Check if markets are open"""
        
        try:
            if not self.access_token:
                # Fallback market status check
                now = datetime.now()
                
                # Check if weekday
                if now.weekday() >= 5:  # Saturday or Sunday
                    return {'status': 'closed', 'reason': 'weekend'}
                
                # Check market hours (9:15 AM - 3:30 PM IST)
                market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
                market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
                
                if market_open <= now <= market_close:
                    return {'status': 'open', 'session': 'normal'}
                elif now < market_open:
                    return {'status': 'closed', 'reason': 'pre-market'}
                else:
                    return {'status': 'closed', 'reason': 'post-market'}
            
            # Get market status from API
            margins = self.kite.margins()
            
            # If we can get margins, market is likely accessible
            return {'status': 'open', 'session': 'normal'}
            
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return {'status': 'unknown', 'error': str(e)}
    
    def get_lot_sizes(self, symbols: List[str]) -> Dict[str, int]:
        """Get lot sizes for symbols"""
        
        result = {}
        for symbol in symbols:
            instrument_info = self.get_instrument_info(symbol)
            if instrument_info:
                result[symbol] = instrument_info.get('lot_size', 1)
            else:
                result[symbol] = 1
        
        return result
    
    def place_order(self, symbol, exchange, transaction_type, quantity, 
                product, order_type, price=None, trigger_price=None, 
                validity='DAY', tag=None):
        """Place an order via Kite Connect"""
        try:
            # Validate access token
            if not self.access_token:
                return {
                    'status': 'error',
                    'message': 'Access token not available. Please login first.'
                }
            
            # Build order parameters
            order_params = {
                'tradingsymbol': symbol.upper(),  # Ensure uppercase
                'exchange': exchange,
                'transaction_type': transaction_type,
                'quantity': int(quantity),  # Ensure integer
                'product': product,
                'order_type': order_type,
                'validity': validity or self.DAY  # Use default if None
            }
            
            # Add optional price parameters only if needed
            if order_type in [self.LIMIT, self.SL]:
                if price is not None:
                    order_params['price'] = float(price)
                else:
                    return {
                        'status': 'error',
                        'message': f'Price is required for {order_type} orders'
                    }
            
            # Add trigger price for SL and SLM orders
            if order_type in [self.SL, self.SLM]:
                if trigger_price is not None:
                    order_params['trigger_price'] = float(trigger_price)
                else:
                    return {
                        'status': 'error',
                        'message': f'Trigger price is required for {order_type} orders'
                    }
            
            # Add tag if provided (max 20 characters)
            if tag:
                order_params['tag'] = str(tag)[:20]
            
            # Place the order with variety parameter
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                **order_params
            )
            
            logger.info(f"Order placed successfully: {order_id} for {symbol}")
            
            return {
                'status': 'success',
                'order_id': order_id,
                'message': f'Order placed successfully: {order_id}',
                'details': {
                    'symbol': symbol,
                    'quantity': quantity,
                    'order_type': order_type,
                    'transaction_type': transaction_type
                }
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error placing order for {symbol}: {error_msg}")
            
            # Provide more specific error messages
            if 'insufficient' in error_msg.lower():
                return {
                    'status': 'error',
                    'message': 'Insufficient funds/margin to place order',
                    'original_error': error_msg
                }
            elif 'invalid' in error_msg.lower():
                return {
                    'status': 'error',
                    'message': 'Invalid order parameters. Please check symbol, quantity, or price',
                    'original_error': error_msg
                }
            elif 'market closed' in error_msg.lower():
                return {
                    'status': 'error',
                    'message': 'Market is closed. Orders can only be placed during market hours',
                    'original_error': error_msg
                }
            else:
                return {
                    'status': 'error',
                    'message': error_msg
                }
    
    def modify_order(self, order_id: str, **kwargs) -> Dict:
        """Modify an existing order"""
        
        try:
            if not self.access_token:
                return {'status': 'error', 'message': 'Access token not available'}
            
            self.kite.modify_order(
                variety=kwargs.get('variety', self.kite.VARIETY_REGULAR),
                order_id=order_id,
                quantity=kwargs.get('quantity'),
                price=kwargs.get('price'),
                order_type=kwargs.get('order_type'),
                validity=kwargs.get('validity'),
                disclosed_quantity=kwargs.get('disclosed_quantity'),
                trigger_price=kwargs.get('trigger_price')
            )
            
            return {'status': 'success', 'order_id': order_id}
            
        except Exception as e:
            logger.error(f"Error modifying order: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def cancel_order(self, order_id: str, variety: str = None) -> Dict:
        """Cancel an order"""
        
        try:
            if not self.access_token:
                return {'status': 'error', 'message': 'Access token not available'}
            
            self.kite.cancel_order(
                variety=variety or self.kite.VARIETY_REGULAR,
                order_id=order_id
            )
            
            return {'status': 'success', 'order_id': order_id}
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_orders(self) -> List[Dict]:
        """Get order list"""
        
        try:
            if not self.access_token:
                logger.error("Access token not available")
                return []
            
            orders = self.kite.orders()
            return orders if isinstance(orders, list) else []
            
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return []
    
    def get_order_history(self, order_id: str) -> List[Dict]:
        """Get order history"""
        
        try:
            if not self.access_token:
                logger.error("Access token not available")
                return []
            
            history = self.kite.order_history(order_id=order_id)
            return history if isinstance(history, list) else []
            
        except Exception as e:
            logger.error(f"Error fetching order history: {e}")
            return []
    
    def get_trades(self) -> List[Dict]:
        """Get trade list"""
        
        try:
            if not self.access_token:
                logger.error("Access token not available")
                return []
            
            trades = self.kite.trades()
            return trades if isinstance(trades, list) else []
            
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return []
    
    def get_positions(self) -> Dict:
        """Get current positions"""
        
        try:
            if not self.access_token:
                logger.error("Access token not available")
                return {'net': [], 'day': []}
            
            positions = self.kite.positions()
            return positions
            
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return {'net': [], 'day': []}
    
    def get_holdings(self) -> List[Dict]:
        """Get holdings"""
        
        try:
            if not self.access_token:
                logger.error("Access token not available")
                return []
            
            holdings = self.kite.holdings()
            return holdings if isinstance(holdings, list) else []
            
        except Exception as e:
            logger.error(f"Error fetching holdings: {e}")
            return []
    
    def get_margins(self, segment: str = None) -> Dict:
        """Get margin details"""
        
        try:
            if not self.access_token:
                logger.error("Access token not available")
                return {}
            
            if segment:
                margins = self.kite.margins(segment=segment)
            else:
                margins = self.kite.margins()
            
            return margins
            
        except Exception as e:
            logger.error(f"Error fetching margins: {e}")
            return {}
    
    def get_order_margins(self, orders: List[Dict]) -> Dict:
        """Calculate order margins"""
        
        try:
            if not self.access_token:
                logger.error("Access token not available")
                return {}
            
            margins = self.kite.order_margins(orders)
            return margins
            
        except Exception as e:
            logger.error(f"Error calculating order margins: {e}")
            return {}
    
    def convert_position(self, **kwargs) -> Dict:
        """Convert position from one product type to another"""
        
        try:
            if not self.access_token:
                return {'status': 'error', 'message': 'Access token not available'}
            
            result = self.kite.convert_position(
                exchange=kwargs.get('exchange'),
                tradingsymbol=kwargs.get('tradingsymbol'),
                transaction_type=kwargs.get('transaction_type'),
                position_type=kwargs.get('position_type'),
                quantity=kwargs.get('quantity'),
                old_product=kwargs.get('old_product'),
                new_product=kwargs.get('new_product')
            )
            
            return {'status': 'success', 'result': result}
            
        except Exception as e:
            logger.error(f"Error converting position: {e}")
            return {'status': 'error', 'message': str(e)}

# Example usage
if __name__ == "__main__":
    # Method 1: Auto-load from environment variables (recommended)
    client = ZerodhaAPIClient()  # Will automatically read from .env file
    
    # Method 2: Explicit API Key + Access Token
    # client = ZerodhaAPIClient(api_key="your_api_key", access_token="your_access_token")
    
    # For first time setup (login flow):
    # 1. Generate login URL
    # login_url = client.generate_login_url()
    # print(f"Login URL: {login_url}")
    # 2. After login, get request token from redirect URL
    # 3. Generate session
    # session_data = client.generate_session(request_token="your_request_token", api_secret="your_api_secret")
    # print(f"Access Token: {session_data['access_token']}")
    
    # Test basic functionality
    try:
        # Get market status
        market_status = client.get_market_status()
        print(f"Market Status: {market_status}")
        
        # Get live quotes
        quotes = client.get_live_quotes(['RELIANCE', 'TCS', 'NIFTY 50'])
        print(f"Live Quotes: {quotes}")
        
        # Get LTP
        ltp = client.get_ltp(['RELIANCE', 'TCS'])
        print(f"LTP: {ltp}")
        
        # Get holdings
        holdings = client.get_holdings()
        print(f"Holdings: {len(holdings)} items")
        
        # Get positions
        positions = client.get_positions()
        print(f"Positions: Net={len(positions.get('net', []))}, Day={len(positions.get('day', []))}")
        
        # Get margins
        margins = client.get_margins()
        print(f"Margins: {margins}")
        
    except Exception as e:
        print(f"Error testing client: {e}")