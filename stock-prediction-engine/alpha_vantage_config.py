#!/usr/bin/env python3
"""
Alpha Vantage Configuration and Testing Script

This script helps you set up and test your Alpha Vantage integration.
Run this script first to configure your API key and test the connection.
"""

import os
import sys
import json
import requests
from datetime import datetime
from typing import Dict, Any, Optional

class AlphaVantageSetup:
    """Setup and configuration helper for Alpha Vantage API."""
    
    def __init__(self):
        self.base_url = "https://www.alphavantage.co/query"
        self.api_key = None
        self.config_file = ".env"
    
    def check_api_key(self) -> bool:
        """Check if API key is already configured."""
        # Check environment variable
        self.api_key = os.getenv('ALPHA_VANTAGE_KEY')
        if self.api_key:
            print(f"✅ Found API key in environment: {self.api_key[:8]}...")
            return True
        
        # Check .env file
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    for line in f:
                        if line.startswith('ALPHA_VANTAGE_KEY='):
                            self.api_key = line.split('=', 1)[1].strip().strip('"\'')
                            if self.api_key:
                                print(f"✅ Found API key in .env file: {self.api_key[:8]}...")
                                return True
            except Exception as e:
                print(f"⚠️  Error reading .env file: {e}")
        
        print("❌ No API key found")
        return False
    
    def get_api_key_from_user(self) -> str:
        """Get API key from user input."""
        print("\n🔑 Alpha Vantage API Key Setup")
        print("-" * 40)
        print("To get your free API key:")
        print("1. Go to: https://www.alphavantage.co/support/#api-key")
        print("2. Fill out the form with your email")
        print("3. Copy the API key you receive")
        print("\nFree tier includes:")
        print("• 500 API calls per day")
        print("• 5 API calls per minute")
        print("• Perfect for testing and development")
        
        while True:
            api_key = input("\nEnter your Alpha Vantage API key: ").strip()
            if api_key:
                return api_key
            print("❌ Please enter a valid API key")
    
    def save_api_key(self, api_key: str) -> bool:
        """Save API key to .env file."""
        try:
            # Read existing .env file
            env_content = []
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    env_content = f.readlines()
            
            # Update or add API key
            found = False
            for i, line in enumerate(env_content):
                if line.startswith('ALPHA_VANTAGE_KEY='):
                    env_content[i] = f'ALPHA_VANTAGE_KEY={api_key}\n'
                    found = True
                    break
            
            if not found:
                env_content.append(f'ALPHA_VANTAGE_KEY={api_key}\n')
                env_content.append('ALPHA_VANTAGE_BASE_URL=https://www.alphavantage.co/query\n')
                env_content.append('MAX_REQUESTS_PER_MINUTE=5\n')
                env_content.append('CACHE_DURATION_HOURS=1\n')
            
            # Write back to file
            with open(self.config_file, 'w') as f:
                f.writelines(env_content)
            
            print(f"✅ API key saved to {self.config_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving API key: {e}")
            return False
    
    def test_api_connection(self, api_key: str) -> Dict[str, Any]:
        """Test the API connection with a sample request."""
        print(f"\n🔍 Testing API connection...")
        
        # Test with a simple quote request
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': 'AAPL',
            'apikey': api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for various error conditions
            if 'Error Message' in data:
                return {
                    'success': False,
                    'error': f"API Error: {data['Error Message']}",
                    'suggestion': "Check that your API key is valid and the symbol exists"
                }
            
            if 'Note' in data:
                return {
                    'success': False,
                    'error': f"Rate Limit: {data['Note']}",
                    'suggestion': "You've hit the rate limit. Try again in a minute."
                }
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                return {
                    'success': True,
                    'data': {
                        'symbol': quote['01. symbol'],
                        'price': quote['05. price'],
                        'change': quote['09. change'],
                        'change_percent': quote['10. change percent'],
                        'volume': quote['06. volume'],
                        'latest_trading_day': quote['07. latest trading day']
                    }
                }
            
            return {
                'success': False,
                'error': f"Unexpected response format: {list(data.keys())}",
                'suggestion': "The API response format may have changed"
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Network error: {e}",
                'suggestion': "Check your internet connection and try again"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Unexpected error: {e}",
                'suggestion': "Please check your API key and try again"
            }
    
    def test_time_series_data(self, api_key: str) -> Dict[str, Any]:
        """Test time series data retrieval."""
        print(f"\n📊 Testing time series data...")
        
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': 'AAPL',
            'apikey': api_key,
            'outputsize': 'compact'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Error Message' in data:
                return {
                    'success': False,
                    'error': f"API Error: {data['Error Message']}"
                }
            
            if 'Note' in data:
                return {
                    'success': False,
                    'error': f"Rate Limit: {data['Note']}"
                }
            
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
                latest_date = list(time_series.keys())[0]
                latest_data = time_series[latest_date]
                
                return {
                    'success': True,
                    'data': {
                        'symbol': 'AAPL',
                        'latest_date': latest_date,
                        'open': latest_data['1. open'],
                        'high': latest_data['2. high'],
                        'low': latest_data['3. low'],
                        'close': latest_data['4. close'],
                        'volume': latest_data['5. volume'],
                        'total_days': len(time_series)
                    }
                }
            
            return {
                'success': False,
                'error': f"Unexpected response format: {list(data.keys())}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Error: {e}"
            }
    
    def check_rate_limits(self) -> Dict[str, Any]:
        """Check current rate limit status."""
        print(f"\n⏱️  Rate Limit Information:")
        print("Free Tier Limits:")
        print("• 500 API calls per day")
        print("• 5 API calls per minute")
        print("• No historical data older than 20 years")
        
        return {
            'daily_limit': 500,
            'minute_limit': 5,
            'recommendation': "Use caching to minimize API calls"
        }
    
    def create_sample_config(self) -> str:
        """Create a sample configuration file."""
        config = {
            'alpha_vantage': {
                'api_key': 'YOUR_API_KEY_HERE',
                'base_url': 'https://www.alphavantage.co/query',
                'max_requests_per_minute': 5,
                'cache_duration_hours': 1,
                'daily_limit': 500
            },
            'correlation_analysis': {
                'default_analysis_days': 30,
                'min_events_for_analysis': 3,
                'min_price_data_points': 5,
                'significant_event_threshold': 2.0
            },
            'logging': {
                'level': 'INFO',
                'file': 'correlation_analyzer.log'
            }
        }
        
        config_file = 'alpha_vantage_config.json'
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✅ Sample configuration created: {config_file}")
            return config_file
        except Exception as e:
            print(f"❌ Error creating config file: {e}")
            return ""
    
    def run_setup(self) -> bool:
        """Run the complete setup process."""
        print("🚀 Alpha Vantage Integration Setup")
        print("=" * 50)
        
        # Check if API key already exists
        if not self.check_api_key():
            # Get API key from user
            api_key = self.get_api_key_from_user()
            
            # Save API key
            if not self.save_api_key(api_key):
                return False
            
            self.api_key = api_key
        
        # Test API connection
        test_result = self.test_api_connection(self.api_key)
        
        if test_result['success']:
            print("✅ API connection test successful!")
            data = test_result['data']
            print(f"   Symbol: {data['symbol']}")
            print(f"   Price: ${data['price']}")
            print(f"   Change: {data['change']} ({data['change_percent']})")
            print(f"   Volume: {data['volume']}")
            print(f"   Latest Trading Day: {data['latest_trading_day']}")
            
            # Test time series data
            ts_result = self.test_time_series_data(self.api_key)
            if ts_result['success']:
                print("✅ Time series data test successful!")
                ts_data = ts_result['data']
                print(f"   Retrieved {ts_data['total_days']} days of data")
                print(f"   Latest date: {ts_data['latest_date']}")
                print(f"   Close price: ${ts_data['close']}")
            else:
                print(f"❌ Time series test failed: {ts_result['error']}")
                return False
            
        else:
            print(f"❌ API connection test failed: {test_result['error']}")
            print(f"💡 Suggestion: {test_result['suggestion']}")
            return False
        
        # Show rate limit info
        self.check_rate_limits()
        
        # Create sample config
        self.create_sample_config()
        
        print(f"\n🎉 Setup Complete!")
        print("=" * 50)
        print("✅ API key configured and tested")
        print("✅ Connection to Alpha Vantage successful")
        print("✅ Time series data retrieval working")
        print("✅ Configuration files created")
        
        print(f"\n🔧 Environment Setup:")
        print("Your .env file now contains:")
        print(f"• ALPHA_VANTAGE_KEY={self.api_key[:8]}...")
        print("• ALPHA_VANTAGE_BASE_URL=https://www.alphavantage.co/query")
        print("• MAX_REQUESTS_PER_MINUTE=5")
        print("• CACHE_DURATION_HOURS=1")
        
        print(f"\n🚀 Ready to use Enhanced Correlation Analyzer!")
        print("You can now run your correlation analysis with real Alpha Vantage data.")
        
        return True


def main():
    """Main setup function."""
    setup = AlphaVantageSetup()
    
    print("Alpha Vantage Integration Setup")
    print("=" * 40)
    print("This script will help you:")
    print("1. Configure your Alpha Vantage API key")
    print("2. Test the API connection")
    print("3. Create configuration files")
    print("4. Verify everything works correctly")
    
    try:
        if setup.run_setup():
            print(f"\n✨ Next Steps:")
            print("1. Run the enhanced correlation analyzer")
            print("2. Test with your stock symbols")
            print("3. Integrate with your news database")
            print("4. Set up automated analysis")
            
            # Show sample usage
            print(f"\n📖 Sample Usage:")
            print("```python")
            print("from correlation_analyzer import NewsPriceCorrelationAnalyzer")
            print("")
            print("# Initialize with your API key")
            print("analyzer = NewsPriceCorrelationAnalyzer()")
            print("")
            print("# Analyze a single stock")
            print("result = analyzer.analyze_correlation('AAPL', days=30)")
            print("print(f'Correlation: {result.correlation_coefficient:.3f}')")
            print("")
            print("# Analyze multiple stocks")
            print("symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT']")
            print("summary = analyzer.get_market_correlation_summary(symbols)")
            print("print(f'Market correlation: {summary[\"avg_correlation\"]:.3f}')")
            print("```")
            
            return True
        else:
            print(f"\n❌ Setup failed. Please check the errors above and try again.")
            return False
            
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Setup interrupted by user.")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)