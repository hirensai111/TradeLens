#!/usr/bin/env python3
"""
Groww API Client for Indian Trading Bot
Handles all interactions with Groww API for market data and trading
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
from growwapi import GrowwAPI, GrowwFeed
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class GrowwAPIClient:
    """Client for interacting with Groww API"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, access_token: str = None):
        """Initialize Groww API client"""
        
        # Auto-load from environment variables if not provided
        if not access_token and not (api_key and api_secret):
            api_key = api_key or os.getenv("GROWW_API_KEY")
            api_secret = api_secret or os.getenv("GROWW_API_SECRET")
            access_token = access_token or os.getenv("GROWW_ACCESS_TOKEN")
        
        if access_token:
            # Method 1: Direct access token
            self.access_token = access_token
            self.groww = GrowwAPI(access_token)
        elif api_key and api_secret:
            # Method 2: API Key + Secret with TOTP
            import pyotp
            self.api_key = api_key
            self.api_secret = api_secret
            
            # Generate TOTP and get access token
            totp_gen = pyotp.TOTP(api_secret)
            totp = totp_gen.now()
            self.access_token = GrowwAPI.get_access_token(api_key, totp)
            self.groww = GrowwAPI(self.access_token)
        else:
            raise ValueError("Either provide access_token or both api_key and api_secret. Check your environment variables or pass them directly.")
        self.feed = GrowwFeed(self.groww)
        
        # Exchange constants
        self.NSE = self.groww.EXCHANGE_NSE
        self.BSE = self.groww.EXCHANGE_BSE
        
        # Segment constants
        self.CASH = self.groww.SEGMENT_CASH
        self.FNO = self.groww.SEGMENT_FNO
        
        # Order types
        self.MARKET = self.groww.ORDER_TYPE_MARKET
        self.LIMIT = self.groww.ORDER_TYPE_LIMIT
        # Note: SL and SLM might not be available in current SDK version
        
        # Product types
        self.CNC = self.groww.PRODUCT_CNC
        self.INTRA = self.groww.PRODUCT_INTRADAY if hasattr(self.groww, 'PRODUCT_INTRADAY') else 'INTRADAY'
        self.MARGIN = self.groww.PRODUCT_MARGIN if hasattr(self.groww, 'PRODUCT_MARGIN') else 'MARGIN'
        
        # Transaction types
        self.BUY = self.groww.TRANSACTION_TYPE_BUY
        self.SELL = self.groww.TRANSACTION_TYPE_SELL
        
        # Validity types
        self.DAY = self.groww.VALIDITY_DAY
        self.IOC = self.groww.VALIDITY_IOC if hasattr(self.groww, 'VALIDITY_IOC') else 'IOC'
        
        # Load instruments data
        self._load_instruments()
        
        logger.info("✅ Groww API client initialized")
    
    def _load_instruments(self):
        """Load instruments data for symbol to exchange token mapping"""
        
        try:
            # Use the new get_all_instruments method
            instruments_df = self.groww.get_all_instruments()
            
            if instruments_df is not None and not instruments_df.empty:
                # Create symbol mapping from instruments data
                self.instruments_df = instruments_df
                self.symbol_map = {}
                
                for _, row in self.instruments_df.iterrows():
                    symbol = row.get('trading_symbol', '').upper()
                    if symbol:
                        self.symbol_map[symbol] = {
                            'exchange_token': row.get('exchange_token'),
                            'exchange': row.get('exchange'),
                            'segment': row.get('segment'),
                            'lot_size': row.get('lot_size', 1),
                            'tick_size': row.get('tick_size', 0.05),
                            'groww_symbol': row.get('groww_symbol', '')
                        }
                
                logger.info(f"Loaded {len(self.symbol_map)} instruments")
            else:
                # Fallback symbol mapping for major stocks
                self.symbol_map = {
                    'RELIANCE': {'exchange_token': '2885', 'exchange': 'NSE', 'segment': 'CASH', 'lot_size': 1, 'groww_symbol': 'NSE-RELIANCE'},
                    'TCS': {'exchange_token': '3499', 'exchange': 'NSE', 'segment': 'CASH', 'lot_size': 1, 'groww_symbol': 'NSE-TCS'},
                    'HDFCBANK': {'exchange_token': '1333', 'exchange': 'NSE', 'segment': 'CASH', 'lot_size': 1, 'groww_symbol': 'NSE-HDFCBANK'},
                    'INFY': {'exchange_token': '1594', 'exchange': 'NSE', 'segment': 'CASH', 'lot_size': 1, 'groww_symbol': 'NSE-INFY'},
                    'HINDUNILVR': {'exchange_token': '1394', 'exchange': 'NSE', 'segment': 'CASH', 'lot_size': 1, 'groww_symbol': 'NSE-HINDUNILVR'},
                    'NIFTYNXT50': {
                        'exchange_token': '26013',  # NIFTY NEXT 50
                        'exchange': 'NSE', 
                        'segment': 'INDICES', 
                        'lot_size': 25,  # Standard F&O lot size for NIFTY NEXT 50
                        'groww_symbol': 'NSE-NIFTY NEXT 50',
                        'full_name': 'NIFTY NEXT 50',
                        'alternate_tokens': {
                            'angel_new': '99926013'
                        }
                    },
                    
                    'MIDCPNIFTY': {
                        'exchange_token': '26011',  # NIFTY MIDCAP 100
                        'exchange': 'NSE', 
                        'segment': 'INDICES', 
                        'lot_size': 75,  # Standard F&O lot size for MIDCAP NIFTY
                        'groww_symbol': 'NSE-NIFTY MIDCAP 100',
                        'full_name': 'NIFTY MIDCAP 100',
                        'alternate_tokens': {
                            'angel_new': '99926011'
                        }
                    },
                    
                    'FINNIFTY': {
                        'exchange_token': '26037',  # NIFTY FIN SERVICE  
                        'exchange': 'NSE', 
                        'segment': 'INDICES', 
                        'lot_size': 40,  # Standard F&O lot size for FINNIFTY
                        'groww_symbol': 'NSE-NIFTY FIN SERVICE',
                        'full_name': 'NIFTY FIN SERVICE', 
                        'alternate_tokens': {
                            'angel_new': '99926037'  # Extrapolated based on pattern
                        }
                    },
                    'NIFTY': {'exchange_token': '13', 'exchange': 'NSE', 'segment': 'CASH', 'lot_size': 50, 'groww_symbol': 'NSE-NIFTY'},
                    'BANKNIFTY': {'exchange_token': '25', 'exchange': 'NSE', 'segment': 'CASH', 'lot_size': 25, 'groww_symbol': 'NSE-BANKNIFTY'},
                    'FINNIFTY': {'exchange_token': '27', 'exchange': 'NSE', 'segment': 'CASH', 'lot_size': 40, 'groww_symbol': 'NSE-FINNIFTY'}
                }
                logger.warning("Using fallback symbol mapping")
            
        except Exception as e:
            logger.error(f"Failed to load instruments: {e}")
            # Use fallback mapping
            self.symbol_map = {
                'RELIANCE': {'exchange_token': '2885', 'exchange': 'NSE', 'segment': 'CASH', 'lot_size': 1, 'groww_symbol': 'NSE-RELIANCE'},
                'TCS': {'exchange_token': '3499', 'exchange': 'NSE', 'segment': 'CASH', 'lot_size': 1, 'groww_symbol': 'NSE-TCS'},
                'HDFCBANK': {'exchange_token': '1333', 'exchange': 'NSE', 'segment': 'CASH', 'lot_size': 1, 'groww_symbol': 'NSE-HDFCBANK'},
                'INFY': {'exchange_token': '1594', 'exchange': 'NSE', 'segment': 'CASH', 'lot_size': 1, 'groww_symbol': 'NSE-INFY'}
            }
    
    def get_instrument_info(self, symbol: str) -> Optional[Dict]:
        """Get instrument info from symbol"""
        return self.symbol_map.get(symbol.upper())
    
    def get_live_quotes(self, symbols: List[str]) -> Dict:
        """Get real-time quotes for multiple symbols"""
        
        try:
            quotes = {}
            
            for symbol in symbols:
                instrument_info = self.get_instrument_info(symbol)
                if not instrument_info:
                    logger.warning(f"Instrument info not found for {symbol}")
                    continue
                
                try:
                    # Use the correct method: get_quote for individual quotes
                    quote_response = self.groww.get_quote(
                        exchange=instrument_info['exchange'],
                        segment=instrument_info['segment'],
                        trading_symbol=symbol
                    )
                    
                    if quote_response:
                        ohlc = quote_response.get('ohlc', {})
                        # Get LTP from multiple possible fields
                        ltp = (quote_response.get('ltp', 0) or 
                               quote_response.get('last_price', 0) or 
                               quote_response.get('close_price', 0) or
                               ohlc.get('close', 0))
                        
                        quotes[symbol] = {
                            'price': ltp,
                            'open': ohlc.get('open', 0),
                            'high': ohlc.get('high', 0),
                            'low': ohlc.get('low', 0),
                            'close': ohlc.get('close', 0),
                            'volume': quote_response.get('volume', 0),
                            'change': quote_response.get('day_change', 0),
                            'change_percent': quote_response.get('day_change_perc', 0)
                        }
                    else:
                        # Fallback: try get_ltp method
                        ltp_response = self.groww.get_ltp(
                            segment=instrument_info['segment'],
                            exchange_trading_symbols=f"{instrument_info['exchange']}_{symbol}"
                        )
                        
                        if ltp_response:
                            quotes[symbol] = {
                                'price': ltp_response.get('ltp', 0),
                                'open': 0,
                                'high': 0,
                                'low': 0,
                                'close': 0,
                                'volume': 0,
                                'change': 0,
                                'change_percent': 0
                            }
                    
                except Exception as e:
                    logger.error(f"Error fetching quote for {symbol}: {e}")
                    continue
            
            return quotes
            
        except Exception as e:
            logger.error(f"Error fetching live quotes: {e}")
            return {}
    
    def get_options_chain(self, symbol: str, expiry_date: str) -> Dict:
        """Get options chain for a symbol"""
        
        try:
            instrument_info = self.get_instrument_info(symbol)
            if not instrument_info:
                logger.error(f"Instrument info not found for {symbol}")
                return {}
            
            # Get option chain data
            response = self.groww.get_option_chain(
                trading_symbol=symbol,
                expiry=expiry_date
            )
            
            if response.get('status') == 'success':
                return self._process_option_chain(response.get('data', {}), symbol)
            else:
                logger.error(f"Failed to fetch option chain: {response}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching option chain: {e}")
            return {}
    
    def _process_option_chain(self, data: Dict, symbol: str) -> Dict:
        """Process raw option chain data"""
        
        try:
            processed_chain = {
                'symbol': symbol,
                'spot_price': data.get('underlying_price', 0),
                'expiry': data.get('expiry', ''),
                'strikes': []
            }
            
            # Process each strike
            for strike_data in data.get('option_chain', []):
                strike_info = {
                    'strike': strike_data.get('strike_price', 0),
                    'call': {
                        'oi': strike_data.get('call_oi', 0),
                        'volume': strike_data.get('call_volume', 0),
                        'ltp': strike_data.get('call_ltp', 0),
                        'bid': strike_data.get('call_bid', 0),
                        'ask': strike_data.get('call_ask', 0),
                        'iv': strike_data.get('call_iv', 0),
                        'delta': strike_data.get('call_delta', 0),
                        'gamma': strike_data.get('call_gamma', 0),
                        'theta': strike_data.get('call_theta', 0),
                        'vega': strike_data.get('call_vega', 0),
                        'exchange_token': strike_data.get('call_exchange_token', '')
                    },
                    'put': {
                        'oi': strike_data.get('put_oi', 0),
                        'volume': strike_data.get('put_volume', 0),
                        'ltp': strike_data.get('put_ltp', 0),
                        'bid': strike_data.get('put_bid', 0),
                        'ask': strike_data.get('put_ask', 0),
                        'iv': strike_data.get('put_iv', 0),
                        'delta': strike_data.get('put_delta', 0),
                        'gamma': strike_data.get('put_gamma', 0),
                        'theta': strike_data.get('put_theta', 0),
                        'vega': strike_data.get('put_vega', 0),
                        'exchange_token': strike_data.get('put_exchange_token', '')
                    }
                }
                processed_chain['strikes'].append(strike_info)
            
            return processed_chain
            
        except Exception as e:
            logger.error(f"Error processing option chain: {e}")
            return {}
    
    def get_historical_data(self, symbol: str, timeframe: str = 'day', days: int = 100) -> pd.DataFrame:
        """Get historical OHLCV data"""
        
        try:
            instrument_info = self.get_instrument_info(symbol)
            if not instrument_info:
                logger.error(f"Instrument info not found for {symbol}")
                return pd.DataFrame()
            
            # Calculate date range - use shorter period to test
            end_time = datetime.now()
            start_time = end_time - timedelta(days=min(days, 10))  # Limit to 10 days for testing
            
            # Format dates for API (based on documentation)
            end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
            start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Set interval based on timeframe
            interval_minutes = 1440 if timeframe == 'day' else 5  # 1440 minutes = 1 day
            
            # Fetch historical data using correct method
            response = self.groww.get_historical_candle_data(
                trading_symbol=symbol,
                exchange=instrument_info['exchange'],
                segment=instrument_info['segment'],
                start_time=start_time_str,
                end_time=end_time_str,
                interval_in_minutes=interval_minutes
            )
            
            # Handle different response formats
            if response:
                # Check if response is a list directly or nested in data
                candles = response
                if isinstance(response, dict) and 'data' in response:
                    candles = response['data']
                if isinstance(candles, dict) and 'candles' in candles:
                    candles = candles['candles']
                
                if isinstance(candles, list) and len(candles) > 0:
                    # Convert to DataFrame
                    df_data = []
                    for candle in candles:
                        if isinstance(candle, list) and len(candle) >= 5:  # At least timestamp, OHLC
                            try:
                                df_data.append({
                                    'datetime': pd.to_datetime(candle[0]),
                                    'open': float(candle[1]),
                                    'high': float(candle[2]),
                                    'low': float(candle[3]),
                                    'close': float(candle[4]),
                                    'volume': int(candle[5]) if len(candle) > 5 and candle[5] else 0
                                })
                            except (ValueError, TypeError, IndexError) as e:
                                logger.warning(f"Skipping invalid candle data: {candle}, error: {e}")
                                continue
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df.set_index('datetime', inplace=True)
                        return df[['open', 'high', 'low', 'close', 'volume']]
                    else:
                        logger.warning(f"No valid candle data found in response: {response}")
                else:
                    logger.warning(f"No candles in response: {response}")
            else:
                logger.warning("Empty response from historical data API")
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def get_market_status(self) -> Dict:
        """Check if markets are open"""
        
        try:
            # Groww API might not have a direct market status method
            # So we'll use a fallback approach
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
    
    def get_expiry_list(self, symbol: str) -> List[str]:
        """Get list of available expiry dates"""
        
        try:
            response = self.groww.get_expiry_list(trading_symbol=symbol)
            
            if response.get('status') == 'success':
                return response.get('data', [])
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching expiry list: {e}")
            return []
    
    def get_margin_calculator(self, positions: List[Dict]) -> Dict:
        """Calculate margin requirements"""
        
        try:
            # Format positions for API
            formatted_positions = []
            
            for pos in positions:
                instrument_info = self.get_instrument_info(pos['symbol'])
                if not instrument_info:
                    continue
                
                formatted_positions.append({
                    'trading_symbol': pos['symbol'],
                    'exchange': instrument_info['exchange'],
                    'segment': instrument_info['segment'],
                    'transaction_type': self.BUY if pos['action'] == 'BUY' else self.SELL,
                    'quantity': pos['quantity'],
                    'product': self.INTRA,
                    'price': pos.get('price', 0)
                })
            
            response = self.groww.get_margin_calculator(positions=formatted_positions)
            
            return response.get('data', {})
            
        except Exception as e:
            logger.error(f"Error calculating margin: {e}")
            return {}
    
    def place_order(self, **kwargs) -> Dict:
        """Place an order"""
        
        try:
            # Get instrument info
            symbol = kwargs.get('symbol')
            instrument_info = self.get_instrument_info(symbol)
            
            if not instrument_info:
                return {'status': 'error', 'message': f'Instrument info not found for {symbol}'}
            
            # Place order via Groww API
            response = self.groww.place_order(
                trading_symbol=symbol,
                exchange=kwargs.get('exchange', instrument_info['exchange']),
                segment=kwargs.get('segment', instrument_info['segment']),
                transaction_type=kwargs.get('transaction_type', self.BUY),
                quantity=kwargs.get('quantity', 1),
                order_type=kwargs.get('order_type', self.LIMIT),
                product=kwargs.get('product', self.INTRA),
                validity=kwargs.get('validity', self.DAY),
                price=kwargs.get('price', 0),
                trigger_price=kwargs.get('trigger_price', 0),
                disclosed_quantity=kwargs.get('disclosed_quantity', 0),
                order_reference_id=kwargs.get('order_reference_id', '')
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def modify_order(self, order_id: str, **kwargs) -> Dict:
        """Modify an existing order"""
        
        try:
            response = self.groww.modify_order(
                order_id=order_id,
                quantity=kwargs.get('quantity'),
                order_type=kwargs.get('order_type'),
                validity=kwargs.get('validity'),
                price=kwargs.get('price'),
                trigger_price=kwargs.get('trigger_price'),
                disclosed_quantity=kwargs.get('disclosed_quantity')
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error modifying order: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order"""
        
        try:
            response = self.groww.cancel_order(order_id=order_id)
            return response
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_orders(self) -> List[Dict]:
        """Get order list"""
        
        try:
            response = self.groww.get_order_list()
            return response if isinstance(response, list) else []
            
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return []
    
    def get_order_history(self, order_id: str) -> List[Dict]:
        """Get order history"""
        
        try:
            response = self.groww.get_order_history(order_id=order_id)
            if response.get('status') == 'success':
                return response.get('data', [])
            return []
            
        except Exception as e:
            logger.error(f"Error fetching order history: {e}")
            return []
    
    def get_trades(self) -> List[Dict]:
        """Get trade list"""
        
        try:
            response = self.groww.get_trade_list()
            if response.get('status') == 'success':
                return response.get('data', [])
            return []
            
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return []
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        
        try:
            response = self.groww.get_positions_for_user()
            return response if isinstance(response, list) else []
            
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
    def get_position_for_symbol(self, symbol: str) -> Dict:
        """Get position for specific trading symbol"""
        
        try:
            instrument_info = self.get_instrument_info(symbol)
            if not instrument_info:
                return {}
            
            response = self.groww.get_position_for_trading_symbol(
                trading_symbol=symbol,
                segment=instrument_info['segment']
            )
            
            if response.get('status') == 'success':
                return response.get('data', {})
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching position for {symbol}: {e}")
            return {}
    
    def get_holdings(self) -> List[Dict]:
        """Get holdings"""
        
        try:
            response = self.groww.get_holdings_for_user()
            return response if isinstance(response, list) else []
            
        except Exception as e:
            logger.error(f"Error fetching holdings: {e}")
            return []
    
    def get_fund_limits(self) -> Dict:
        """Get fund limits and margins"""
        
        try:
            # Groww API might not have this method yet, so return empty dict
            # This can be implemented when the method becomes available
            logger.warning("Fund limits API not available in current Groww SDK version")
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching fund limits: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    # Method 1: Auto-load from environment variables (recommended)
    client = GrowwAPIClient()  # Will automatically read from .env file
    
    # Method 2: Explicit API Key + Secret
    # client = GrowwAPIClient(api_key="your_api_key", api_secret="your_api_secret")
    
    # Method 3: Using pre-generated access token
    # client = GrowwAPIClient(access_token="your_access_token")
    
    # Test basic functionality
    try:
        # Get market status
        market_status = client.get_market_status()
        print(f"Market Status: {market_status}")
        
        # Get live quotes
        quotes = client.get_live_quotes(['RELIANCE', 'TCS', 'NIFTY'])
        print(f"Live Quotes: {quotes}")
        
        # Get holdings
        holdings = client.get_holdings()
        print(f"Holdings: {len(holdings)} items")
        
        # Get positions
        positions = client.get_positions()
        print(f"Positions: {len(positions)} items")
        
    except Exception as e:
        print(f"Error testing client: {e}")