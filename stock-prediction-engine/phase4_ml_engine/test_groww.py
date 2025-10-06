#!/usr/bin/env python3
"""
Groww API Market Data Test Script
Tests the Groww API client to verify if it can access market data using API keys
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

# Import your Groww API client
try:
    from groww_api_client import GrowwAPIClient
except ImportError:
    print("❌ Error: Cannot import GrowwAPIClient. Make sure the groww_api_client.py file is in the same directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('groww_api_test.log')
    ]
)

logger = logging.getLogger(__name__)

class GrowwAPITester:
    """Test suite for Groww API client"""
    
    def __init__(self):
        """Initialize the tester"""
        self.client = None
        self.test_symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'NIFTY', 'BANKNIFTY']
        self.passed_tests = 0
        self.failed_tests = 0
        self.total_tests = 0
        
    def print_header(self, title: str):
        """Print a formatted header"""
        print(f"\n{'='*60}")
        print(f"🧪 {title}")
        print(f"{'='*60}")
    
    def print_test_result(self, test_name: str, success: bool, message: str = ""):
        """Print test result with formatting"""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
            print(f"✅ {test_name}: PASSED {message}")
        else:
            self.failed_tests += 1
            print(f"❌ {test_name}: FAILED {message}")
    
    def test_environment_variables(self) -> bool:
        """Test if environment variables are properly set"""
        self.print_header("Testing Environment Variables")
        
        try:
            # Load environment variables
            load_dotenv()
            
            api_key = os.getenv("GROWW_API_KEY")
            api_secret = os.getenv("GROWW_API_SECRET")
            access_token = os.getenv("GROWW_ACCESS_TOKEN")
            
            # Test API Key method
            if api_key and api_secret:
                self.print_test_result(
                    "API Key Environment Variables", 
                    True, 
                    f"(Key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else 'short'}, Secret: {api_secret[:8]}...{api_secret[-4:] if len(api_secret) > 12 else 'short'})"
                )
                return True
            elif access_token:
                self.print_test_result(
                    "Access Token Environment Variable", 
                    True, 
                    f"(Token: {access_token[:8]}...{access_token[-4:] if len(access_token) > 12 else 'short'})"
                )
                return True
            else:
                self.print_test_result(
                    "Environment Variables", 
                    False, 
                    "Neither API Key+Secret nor Access Token found in .env file"
                )
                return False
                
        except Exception as e:
            self.print_test_result("Environment Variables", False, f"Error: {str(e)}")
            return False
    
    def test_client_initialization(self) -> bool:
        """Test if the client can be initialized"""
        self.print_header("Testing Client Initialization")
        
        try:
            # Test auto-initialization from environment
            self.client = GrowwAPIClient()
            self.print_test_result("Client Initialization", True, "Successfully initialized from environment variables")
            return True
            
        except Exception as e:
            self.print_test_result("Client Initialization", False, f"Error: {str(e)}")
            logger.error(f"Client initialization failed: {traceback.format_exc()}")
            return False
    
    def test_market_status(self) -> bool:
        """Test market status API"""
        self.print_header("Testing Market Status API")
        
        if not self.client:
            self.print_test_result("Market Status", False, "Client not initialized")
            return False
        
        try:
            market_status = self.client.get_market_status()
            
            if market_status and isinstance(market_status, dict):
                status = market_status.get('status', 'unknown')
                session = market_status.get('session', '')
                reason = market_status.get('reason', '')
                
                self.print_test_result(
                    "Market Status API", 
                    True, 
                    f"Status: {status}, Session: {session}, Reason: {reason}"
                )
                return True
            else:
                self.print_test_result("Market Status API", False, f"Invalid response: {market_status}")
                return False
                
        except Exception as e:
            self.print_test_result("Market Status API", False, f"Error: {str(e)}")
            logger.error(f"Market status test failed: {traceback.format_exc()}")
            return False
    
    def test_instruments_loading(self) -> bool:
        """Test if instruments are loaded properly"""
        self.print_header("Testing Instruments Loading")
        
        if not self.client:
            self.print_test_result("Instruments Loading", False, "Client not initialized")
            return False
        
        try:
            # Check if symbol map is populated
            if hasattr(self.client, 'symbol_map') and self.client.symbol_map:
                symbol_count = len(self.client.symbol_map)
                
                # Test some common symbols
                test_symbols = ['RELIANCE', 'TCS', 'NIFTY']
                found_symbols = []
                
                for symbol in test_symbols:
                    if symbol in self.client.symbol_map:
                        found_symbols.append(symbol)
                
                self.print_test_result(
                    "Instruments Loading", 
                    True, 
                    f"Loaded {symbol_count} instruments, Found test symbols: {found_symbols}"
                )
                return True
            else:
                self.print_test_result("Instruments Loading", False, "Symbol map is empty or not available")
                return False
                
        except Exception as e:
            self.print_test_result("Instruments Loading", False, f"Error: {str(e)}")
            logger.error(f"Instruments loading test failed: {traceback.format_exc()}")
            return False
    
    def test_live_quotes(self) -> bool:
        """Test live quotes API"""
        self.print_header("Testing Live Quotes API")
        
        if not self.client:
            self.print_test_result("Live Quotes", False, "Client not initialized")
            return False
        
        try:
            # Test with a subset of symbols to avoid rate limits
            test_symbols = ['RELIANCE', 'TCS']
            quotes = self.client.get_live_quotes(test_symbols)
            
            if quotes and isinstance(quotes, dict):
                successful_quotes = []
                
                for symbol, quote_data in quotes.items():
                    if quote_data and quote_data.get('price', 0) > 0:
                        successful_quotes.append(f"{symbol}: ₹{quote_data['price']}")
                
                if successful_quotes:
                    self.print_test_result(
                        "Live Quotes API", 
                        True, 
                        f"Retrieved quotes for: {', '.join(successful_quotes)}"
                    )
                    return True
                else:
                    self.print_test_result("Live Quotes API", False, f"No valid quotes received: {quotes}")
                    return False
            else:
                self.print_test_result("Live Quotes API", False, f"Invalid response: {quotes}")
                return False
                
        except Exception as e:
            self.print_test_result("Live Quotes API", False, f"Error: {str(e)}")
            logger.error(f"Live quotes test failed: {traceback.format_exc()}")
            return False
    
    def test_historical_data(self) -> bool:
        """Test historical data API"""
        self.print_header("Testing Historical Data API")
        
        if not self.client:
            self.print_test_result("Historical Data", False, "Client not initialized")
            return False
        
        try:
            # Test with RELIANCE for last 30 days
            df = self.client.get_historical_data('RELIANCE', timeframe='day', days=30)
            
            if df is not None and not df.empty:
                rows, cols = df.shape
                latest_date = df.index[-1] if len(df) > 0 else "N/A"
                latest_close = df['close'].iloc[-1] if 'close' in df.columns and len(df) > 0 else "N/A"
                
                self.print_test_result(
                    "Historical Data API", 
                    True, 
                    f"Retrieved {rows} rows, {cols} columns. Latest: {latest_date}, Close: ₹{latest_close}"
                )
                return True
            else:
                self.print_test_result("Historical Data API", False, "No data returned or empty DataFrame")
                return False
                
        except Exception as e:
            self.print_test_result("Historical Data API", False, f"Error: {str(e)}")
            logger.error(f"Historical data test failed: {traceback.format_exc()}")
            return False
    
    def test_account_info(self) -> bool:
        """Test account-related APIs"""
        self.print_header("Testing Account Information APIs")
        
        if not self.client:
            self.print_test_result("Account Info", False, "Client not initialized")
            return False
        
        success_count = 0
        total_account_tests = 0
        
        # Test Holdings
        try:
            total_account_tests += 1
            holdings = self.client.get_holdings()
            if isinstance(holdings, list):
                self.print_test_result("Holdings API", True, f"Retrieved {len(holdings)} holdings")
                success_count += 1
            else:
                self.print_test_result("Holdings API", False, f"Invalid response: {holdings}")
        except Exception as e:
            self.print_test_result("Holdings API", False, f"Error: {str(e)}")
        
        # Test Positions
        try:
            total_account_tests += 1
            positions = self.client.get_positions()
            if isinstance(positions, list):
                self.print_test_result("Positions API", True, f"Retrieved {len(positions)} positions")
                success_count += 1
            else:
                self.print_test_result("Positions API", False, f"Invalid response: {positions}")
        except Exception as e:
            self.print_test_result("Positions API", False, f"Error: {str(e)}")
        
        # Test Fund Limits
        try:
            total_account_tests += 1
            funds = self.client.get_fund_limits()
            if isinstance(funds, dict):
                self.print_test_result("Fund Limits API", True, f"Retrieved fund info: {list(funds.keys())}")
                success_count += 1
            else:
                self.print_test_result("Fund Limits API", False, f"Invalid response: {funds}")
        except Exception as e:
            self.print_test_result("Fund Limits API", False, f"Error: {str(e)}")
        
        return success_count == total_account_tests
    
    def test_orders_api(self) -> bool:
        """Test orders-related APIs (read-only)"""
        self.print_header("Testing Orders API (Read-Only)")
        
        if not self.client:
            self.print_test_result("Orders API", False, "Client not initialized")
            return False
        
        try:
            # Test getting order list (should work even if empty)
            orders = self.client.get_orders()
            if isinstance(orders, list):
                self.print_test_result("Orders List API", True, f"Retrieved {len(orders)} orders")
                return True
            else:
                self.print_test_result("Orders List API", False, f"Invalid response: {orders}")
                return False
                
        except Exception as e:
            self.print_test_result("Orders List API", False, f"Error: {str(e)}")
            logger.error(f"Orders API test failed: {traceback.format_exc()}")
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print(f"\n🚀 Starting Groww API Test Suite - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("📝 This will test your API connectivity and basic functionality")
        
        # Run tests in order
        tests = [
            ("Environment Variables", self.test_environment_variables),
            ("Client Initialization", self.test_client_initialization),
            ("Market Status", self.test_market_status),
            ("Instruments Loading", self.test_instruments_loading),
            ("Live Quotes", self.test_live_quotes),
            ("Historical Data", self.test_historical_data),
            ("Account Information", self.test_account_info),
            ("Orders API", self.test_orders_api),
        ]
        
        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                self.print_test_result(f"{test_name} (Exception)", False, f"Unexpected error: {str(e)}")
                logger.error(f"Unexpected error in {test_name}: {traceback.format_exc()}")
        
        # Print final results
        self.print_final_results()
    
    def print_final_results(self):
        """Print final test results"""
        print(f"\n{'='*60}")
        print(f"📊 TEST RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"📝 Total:  {self.total_tests}")
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print(f"\n🎉 EXCELLENT! Your Groww API is working well!")
        elif success_rate >= 60:
            print(f"\n👍 GOOD! Most features are working. Check failed tests above.")
        elif success_rate >= 40:
            print(f"\n⚠️  PARTIAL! Some features working. Review your API setup.")
        else:
            print(f"\n🚨 POOR! Most tests failed. Check your API credentials and network.")
        
        print(f"\n💡 Tips:")
        print(f"   - Make sure your .env file has correct GROWW_API_KEY and GROWW_API_SECRET")
        print(f"   - Ensure you have installed: pip install growwapi pyotp pandas python-dotenv")
        print(f"   - Check that your Groww API subscription is active")
        print(f"   - Review the log file: groww_api_test.log for detailed errors")
        
        if self.failed_tests > 0:
            print(f"\n🔍 For failed tests, check the error messages above and the log file.")

def main():
    """Main function to run the tests"""
    try:
        tester = GrowwAPITester()
        tester.run_all_tests()
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Critical error running tests: {str(e)}")
        logger.error(f"Critical error: {traceback.format_exc()}")

if __name__ == "__main__":
    main()