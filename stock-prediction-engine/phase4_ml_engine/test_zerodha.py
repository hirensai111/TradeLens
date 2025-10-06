#!/usr/bin/env python3
"""
Comprehensive test script for Zerodha Kite Connect API Client
Tests all functionality including live options data fetching for NIFTY and INFY
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict
import time

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
    print("📄 Loaded environment variables from .env file")
except ImportError:
    print("💡 python-dotenv not installed. Install with: pip install python-dotenv")
    print("💡 Or set environment variables manually")

# Add the path where your ZerodhaAPIClient is located
# sys.path.append('/path/to/your/client')

# Try to import the client - adjust the import based on your file structure
try:
    from zerodha_api_client import ZerodhaAPIClient
except ImportError:
    try:
        # If the file is in the same directory but named differently
        import importlib.util
        spec = importlib.util.spec_from_file_location("zerodha_api_client", "zerodha_api_client.py")
        if spec is None:
            raise ImportError("Could not find zerodha_api_client.py")
        zerodha_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(zerodha_module)
        ZerodhaAPIClient = zerodha_module.ZerodhaAPIClient
    except Exception as e:
        print(f"❌ Could not import ZerodhaAPIClient: {e}")
        print("Please ensure zerodha_api_client.py is in the same directory or adjust the import path")
        sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('zerodha_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ZerodhaAPITester:
    """Comprehensive tester for Zerodha API functionality"""
    
    def __init__(self):
        """Initialize the tester"""
        self.client = None
        self.test_symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'SBIN']
        self.index_symbols = ['NIFTY 50', 'NIFTY BANK']
        self.test_results = {}
        
    def setup_client(self):
        """Setup Zerodha API client"""
        try:
            print("🔧 Setting up Zerodha API client...")
            
            # Try to initialize with environment variables
            api_key = os.getenv("ZERODHA_API_KEY")
            access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
            
            if not api_key:
                print("❌ ZERODHA_API_KEY not found in environment variables")
                api_key = input("Enter your Zerodha API Key: ").strip()
            
            if not access_token:
                print("❌ ZERODHA_ACCESS_TOKEN not found in environment variables")
                print("You'll need to complete the login flow first...")
                
                try:
                    # Initialize client with just API key
                    self.client = ZerodhaAPIClient(api_key=api_key)
                    
                    # Generate login URL
                    login_url = self.client.generate_login_url()
                    print(f"📱 Please visit this URL to login: {login_url}")
                    print("\nAfter login, you'll be redirected to a URL like:")
                    print("https://127.0.0.1:5000/?request_token=XXXXXX&action=login&status=success")
                    
                    # Get request token from user
                    request_token = input("\nEnter the request_token from redirect URL: ").strip()
                    api_secret = input("Enter your API Secret: ").strip()
                    
                    if not request_token or not api_secret:
                        print("❌ Request token and API secret are required")
                        return False
                    
                    # Generate session
                    session_data = self.client.generate_session(request_token, api_secret)
                    access_token = session_data['access_token']
                    
                    print(f"✅ Access Token generated: {access_token}")
                    print("💡 Save this access token as ZERODHA_ACCESS_TOKEN environment variable for future use")
                    print(f"💡 You can set it by running: set ZERODHA_ACCESS_TOKEN={access_token}")
                    
                except Exception as e:
                    print(f"❌ Error during login flow: {e}")
                    print("💡 Common issues:")
                    print("   - Make sure you completed the login in browser")
                    print("   - Check that request_token is correct (no extra spaces)")
                    print("   - Verify your API secret is correct")
                    return False
            else:
                try:
                    self.client = ZerodhaAPIClient(api_key=api_key, access_token=access_token)
                except Exception as e:
                    print(f"❌ Error initializing client with stored token: {e}")
                    print("💡 Your access token might be expired. Try removing it and running the login flow again.")
                    return False
            
            print("✅ Zerodha API client setup complete")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up client: {e}")
            print("💡 Make sure you have the kiteconnect library installed: pip install kiteconnect")
            return False
    
    def test_market_status(self):
        """Test market status functionality"""
        print("\n📊 Testing Market Status...")
        
        try:
            market_status = self.client.get_market_status()
            print(f"Market Status: {market_status}")
            
            self.test_results['market_status'] = {
                'status': 'success',
                'data': market_status
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Market status test failed: {e}")
            self.test_results['market_status'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_live_quotes(self):
        """Test live quotes functionality"""
        print("\n💰 Testing Live Quotes...")
        
        try:
            # Test equity quotes
            print("Testing equity quotes...")
            quotes = self.client.get_live_quotes(self.test_symbols)
            
            for symbol, quote in quotes.items():
                print(f"{symbol}: Price={quote.get('price', 0):.2f}, "
                      f"Change={quote.get('change', 0):.2f} "
                      f"({quote.get('change_percent', 0):.2f}%)")
            
            # Test index quotes
            print("\nTesting index quotes...")
            index_quotes = self.client.get_live_quotes(self.index_symbols)
            
            for symbol, quote in index_quotes.items():
                print(f"{symbol}: Price={quote.get('price', 0):.2f}, "
                      f"Change={quote.get('change', 0):.2f}")
            
            self.test_results['live_quotes'] = {
                'status': 'success',
                'equity_count': len(quotes),
                'index_count': len(index_quotes)
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Live quotes test failed: {e}")
            self.test_results['live_quotes'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_ltp(self):
        """Test Last Traded Price functionality"""
        print("\n🎯 Testing LTP (Last Traded Price)...")
        
        try:
            ltp_data = self.client.get_ltp(self.test_symbols)
            
            for symbol, ltp in ltp_data.items():
                print(f"{symbol}: LTP = ₹{ltp:.2f}")
            
            self.test_results['ltp'] = {
                'status': 'success',
                'count': len(ltp_data)
            }
            
            return True
            
        except Exception as e:
            print(f"❌ LTP test failed: {e}")
            self.test_results['ltp'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_options_data(self):
        """Test options data functionality for both NIFTY and INFY"""
        print("\n📈 Testing Options Data (NIFTY & INFY)...")
        
        try:
            total_contracts_tested = 0
            total_options_found = 0
            
            # Test NIFTY options
            print("Testing NIFTY options data...")
            nifty_options = self.find_options('NIFTY', 'NFO')
            
            if nifty_options:
                # Get quotes for some NIFTY options
                nifty_option_symbols = list(nifty_options.keys())[:8]  # Test first 8 options
                print(f"Testing {len(nifty_option_symbols)} NIFTY option contracts...")
                
                nifty_option_quotes = self.client.get_live_quotes(nifty_option_symbols)
                
                print("\n📊 NIFTY Options Data:")
                print("-" * 85)
                print(f"{'Symbol':<30} {'LTP':<10} {'Volume':<10} {'OI':<10} {'Bid':<10} {'Ask':<10}")
                print("-" * 85)
                
                for symbol, quote in nifty_option_quotes.items():
                    print(f"{symbol:<30} "
                          f"{quote.get('price', 0):<10.2f} "
                          f"{quote.get('volume', 0):<10} "
                          f"{quote.get('oi', 0):<10} "
                          f"{quote.get('bid', 0):<10.2f} "
                          f"{quote.get('ask', 0):<10.2f}")
                
                total_contracts_tested += len(nifty_option_quotes)
                total_options_found += len(nifty_options)
            
            # Test INFY options
            print("\n\nTesting INFY options data...")
            infy_options = self.find_options('INFY', 'NFO')
            
            if infy_options:
                # Get quotes for some INFY options
                infy_option_symbols = list(infy_options.keys())[:8]  # Test first 8 options
                print(f"Testing {len(infy_option_symbols)} INFY option contracts...")
                
                infy_option_quotes = self.client.get_live_quotes(infy_option_symbols)
                
                print("\n📊 INFY Options Data:")
                print("-" * 85)
                print(f"{'Symbol':<30} {'LTP':<10} {'Volume':<10} {'OI':<10} {'Bid':<10} {'Ask':<10}")
                print("-" * 85)
                
                for symbol, quote in infy_option_quotes.items():
                    print(f"{symbol:<30} "
                          f"{quote.get('price', 0):<10.2f} "
                          f"{quote.get('volume', 0):<10} "
                          f"{quote.get('oi', 0):<10} "
                          f"{quote.get('bid', 0):<10.2f} "
                          f"{quote.get('ask', 0):<10.2f}")
                
                total_contracts_tested += len(infy_option_quotes)
                total_options_found += len(infy_options)
            
            if total_contracts_tested > 0:
                self.test_results['options_data'] = {
                    'status': 'success',
                    'contracts_tested': total_contracts_tested,
                    'total_options_found': total_options_found,
                    'nifty_options': len(nifty_options) if nifty_options else 0,
                    'infy_options': len(infy_options) if infy_options else 0
                }
                
                return True
            else:
                print("❌ No options contracts found for testing")
                self.test_results['options_data'] = {
                    'status': 'failed',
                    'error': 'No options instruments found'
                }
                return False
            
        except Exception as e:
            print(f"❌ Options data test failed: {e}")
            self.test_results['options_data'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def find_options(self, underlying: str, exchange: str = 'NFO') -> Dict:
        """Find option contracts for a given underlying"""
        try:
            if not hasattr(self.client, 'instruments_df') or self.client.instruments_df is None:
                print(f"Instruments data not loaded, loading now...")
                self.client._load_instruments()
            
            if hasattr(self.client, 'instruments_df') and self.client.instruments_df is not None:
                # Filter for options of the given underlying
                options = self.client.instruments_df[
                    (self.client.instruments_df['name'] == underlying) & 
                    (self.client.instruments_df['instrument_type'].isin(['CE', 'PE'])) &
                    (self.client.instruments_df['exchange'] == exchange)
                ]
                
                # Get current week/nearest expiry options
                if not options.empty:
                    # Sort by expiry and get nearest expiry
                    options = options.sort_values('expiry')
                    nearest_expiry = options['expiry'].iloc[0]
                    
                    # Filter for nearest expiry
                    current_expiry_options = options[options['expiry'] == nearest_expiry]
                    
                    print(f"Found {len(current_expiry_options)} {underlying} options for expiry: {nearest_expiry}")
                    
                    # Create options mapping
                    options_map = {}
                    for _, option in current_expiry_options.iterrows():
                        symbol = option['tradingsymbol']
                        options_map[symbol] = {
                            'instrument_token': option['instrument_token'],
                            'exchange': option['exchange'],
                            'strike': option['strike'],
                            'option_type': option['instrument_type'],
                            'expiry': option['expiry'],
                            'underlying': underlying
                        }
                        
                        # Update client's symbol map if not already present
                        if symbol not in self.client.symbol_map:
                            self.client.symbol_map[symbol] = {
                                'instrument_token': option['instrument_token'],
                                'exchange_token': option['exchange_token'],
                                'exchange': option['exchange'],
                                'segment': option['segment'],
                                'lot_size': option['lot_size'],
                                'tick_size': option['tick_size'],
                                'name': option['name'],
                                'expiry': option['expiry'],
                                'strike': option['strike'],
                                'instrument_type': option['instrument_type']
                            }
                    
                    return options_map
            
            return {}
            
        except Exception as e:
            print(f"Error finding {underlying} options: {e}")
            return {}
    
    def test_historical_data(self):
        """Test historical data functionality"""
        print("\n📊 Testing Historical Data...")
        
        try:
            # Test with RELIANCE
            symbol = 'RELIANCE'
            print(f"Fetching historical data for {symbol}...")
            
            # Get daily data for last 30 days
            hist_data = self.client.get_historical_data(symbol, timeframe='day', days=30)
            
            if not hist_data.empty:
                print(f"Historical data shape: {hist_data.shape}")
                print(f"Date range: {hist_data.index[0]} to {hist_data.index[-1]}")
                print("\nLast 5 days data:")
                print(hist_data.tail())
                
                self.test_results['historical_data'] = {
                    'status': 'success',
                    'records': len(hist_data),
                    'date_range': f"{hist_data.index[0]} to {hist_data.index[-1]}"
                }
                
                return True
            else:
                print("❌ No historical data received")
                self.test_results['historical_data'] = {
                    'status': 'failed',
                    'error': 'No data received'
                }
                return False
                
        except Exception as e:
            print(f"❌ Historical data test failed: {e}")
            self.test_results['historical_data'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_account_info(self):
        """Test account related functionality"""
        print("\n👤 Testing Account Information...")
        
        try:
            # Test margins
            print("Testing margins...")
            margins = self.client.get_margins()
            
            if margins:
                print("Margin details:")
                for segment, margin_data in margins.items():
                    if isinstance(margin_data, dict):
                        available = margin_data.get('available', {})
                        print(f"  {segment}: Available Cash = ₹{available.get('cash', 0):,.2f}")
            
            # Test positions
            print("\nTesting positions...")
            positions = self.client.get_positions()
            
            net_positions = positions.get('net', [])
            day_positions = positions.get('day', [])
            
            print(f"Net positions: {len(net_positions)}")
            print(f"Day positions: {len(day_positions)}")
            
            if net_positions:
                print("Net positions details:")
                for pos in net_positions[:5]:  # Show first 5
                    print(f"  {pos.get('tradingsymbol', 'N/A')}: "
                          f"Qty={pos.get('quantity', 0)}, "
                          f"PnL=₹{pos.get('pnl', 0):.2f}")
            
            # Test holdings
            print("\nTesting holdings...")
            holdings = self.client.get_holdings()
            
            print(f"Holdings count: {len(holdings)}")
            
            if holdings:
                print("Holdings details:")
                for holding in holdings[:5]:  # Show first 5
                    print(f"  {holding.get('tradingsymbol', 'N/A')}: "
                          f"Qty={holding.get('quantity', 0)}, "
                          f"LTP=₹{holding.get('last_price', 0):.2f}")
            
            self.test_results['account_info'] = {
                'status': 'success',
                'net_positions': len(net_positions),
                'day_positions': len(day_positions),
                'holdings': len(holdings)
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Account info test failed: {e}")
            self.test_results['account_info'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_orders(self):
        """Test order related functionality"""
        print("\n📝 Testing Order Management...")
        
        try:
            # Get existing orders
            print("Fetching existing orders...")
            orders = self.client.get_orders()
            
            print(f"Total orders today: {len(orders)}")
            
            if orders:
                print("Recent orders:")
                for order in orders[-5:]:  # Show last 5 orders
                    print(f"  {order.get('tradingsymbol', 'N/A')}: "
                          f"{order.get('transaction_type', 'N/A')} "
                          f"{order.get('quantity', 0)} @ ₹{order.get('price', 0):.2f} "
                          f"Status: {order.get('status', 'N/A')}")
            
            # Get trades
            print("\nFetching trades...")
            trades = self.client.get_trades()
            
            print(f"Total trades today: {len(trades)}")
            
            if trades:
                print("Recent trades:")
                for trade in trades[-5:]:  # Show last 5 trades
                    print(f"  {trade.get('tradingsymbol', 'N/A')}: "
                          f"{trade.get('transaction_type', 'N/A')} "
                          f"{trade.get('quantity', 0)} @ ₹{trade.get('average_price', 0):.2f}")
            
            self.test_results['orders'] = {
                'status': 'success',
                'orders_count': len(orders),
                'trades_count': len(trades)
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Orders test failed: {e}")
            self.test_results['orders'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_instruments_loading(self):
        """Test instruments loading functionality"""
        print("\n🔧 Testing Instruments Loading...")
        
        try:
            # Check if instruments are loaded
            if hasattr(self.client, 'symbol_map') and self.client.symbol_map:
                print(f"Total instruments loaded: {len(self.client.symbol_map)}")
                
                # Show some sample instruments
                print("\nSample instruments:")
                sample_symbols = list(self.client.symbol_map.keys())[:10]
                for symbol in sample_symbols:
                    info = self.client.symbol_map[symbol]
                    print(f"  {symbol}: Token={info.get('instrument_token', 'N/A')}, "
                          f"Exchange={info.get('exchange', 'N/A')}")
                
                # Check for options
                options_count = sum(1 for symbol, info in self.client.symbol_map.items() 
                                  if info.get('instrument_type') in ['CE', 'PE'])
                
                # Check for NIFTY and INFY options specifically
                nifty_options_count = sum(1 for symbol, info in self.client.symbol_map.items() 
                                        if info.get('instrument_type') in ['CE', 'PE'] and 'NIFTY' in symbol)
                
                infy_options_count = sum(1 for symbol, info in self.client.symbol_map.items() 
                                       if info.get('instrument_type') in ['CE', 'PE'] and 'INFY' in symbol)
                
                print(f"\nOptions contracts loaded: {options_count}")
                print(f"NIFTY options loaded: {nifty_options_count}")
                print(f"INFY options loaded: {infy_options_count}")
                
                self.test_results['instruments'] = {
                    'status': 'success',
                    'total_instruments': len(self.client.symbol_map),
                    'options_count': options_count,
                    'nifty_options_count': nifty_options_count,
                    'infy_options_count': infy_options_count
                }
                
                return True
            else:
                print("❌ Instruments not loaded")
                self.test_results['instruments'] = {
                    'status': 'failed',
                    'error': 'Instruments not loaded'
                }
                return False
                
        except Exception as e:
            print(f"❌ Instruments loading test failed: {e}")
            self.test_results['instruments'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*60)
        print("🔍 TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() 
                          if result.get('status') == 'success')
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nDetailed Results:")
        print("-" * 40)
        
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            print(f"{status_icon} {test_name.replace('_', ' ').title()}")
            
            if result['status'] == 'success':
                # Show additional success details
                for key, value in result.items():
                    if key != 'status':
                        print(f"    {key}: {value}")
            else:
                print(f"    Error: {result.get('error', 'Unknown error')}")
        
        print("\n" + "="*60)
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting Zerodha API Comprehensive Tests (NIFTY & INFY Options)")
        print("="*65)
        
        # Setup client
        if not self.setup_client():
            print("❌ Failed to setup client. Exiting...")
            return
        
        # Run all tests
        tests = [
            self.test_market_status,
            self.test_instruments_loading,
            self.test_live_quotes,
            self.test_ltp,
            self.test_options_data,
            self.test_historical_data,
            self.test_account_info,
            self.test_orders
        ]
        
        for test_func in tests:
            try:
                test_func()
                time.sleep(1)  # Small delay between tests
            except Exception as e:
                print(f"❌ Unexpected error in {test_func.__name__}: {e}")
        
        # Print summary
        self.print_test_summary()

def main():
    """Main function"""
    tester = ZerodhaAPITester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()