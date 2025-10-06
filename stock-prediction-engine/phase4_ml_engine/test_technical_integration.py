#!/usr/bin/env python3
"""
Test script for technical analysis integration
"""

import asyncio
import sys
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get config from environment
ZERODHA_CONFIG = {
    'api_key': os.getenv('ZERODHA_API_KEY', 'your_api_key_here'),
    'api_secret': os.getenv('ZERODHA_API_SECRET', 'your_api_secret_here'),
    'access_token': os.getenv('ZERODHA_ACCESS_TOKEN'),
    'user_id': os.getenv('ZERODHA_USER_ID', 'your_user_id_here')
}

print(f"🔑 Loaded config: API Key: {ZERODHA_CONFIG['api_key'][:8]}...")

# Try to import the analyzer
try:
    from options_analyzer import ZerodhaEnhancedOptionsAnalyzer
    print("✅ Successfully imported ZerodhaEnhancedOptionsAnalyzer")
except ImportError as e:
    print(f"❌ Failed to import ZerodhaEnhancedOptionsAnalyzer: {e}")
    print("Available Python files in current directory:")
    for file in os.listdir('.'):
        if file.endswith('.py'):
            print(f"  - {file}")
    
    # Let's try just the technical analyzer first
    try:
        from zerodha_technical_analyzer import ZerodhaTechnicalAnalyzer
        print("✅ Technical analyzer imported successfully, let's test that first")
        TEST_TECH_ONLY = True
    except ImportError as e2:
        print(f"❌ Technical analyzer also failed: {e2}")
        sys.exit(1)
else:
    TEST_TECH_ONLY = False


# Mock Zerodha client for testing technical analyzer
class MockZerodhaClient:
    def __init__(self):
        self.access_token = "mock_token"
    
    def get_historical_data(self, symbol: str, interval: str, days: int):
        """Generate realistic mock historical data"""
        import pandas as pd
        import numpy as np
        
        print(f"📊 Mock: Getting {days} days of {interval} data for {symbol}")
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        base_price = 2145 if 'RELIANCE' in symbol else 1650 if 'HDFCBANK' in symbol else 1500
        
        # Generate trending price data
        trend = np.linspace(0, 0.05, days)  # 5% uptrend over period
        noise = np.random.normal(0, 0.015, days)  # 1.5% daily noise
        
        prices = []
        for i in range(days):
            if i == 0:
                prices.append(base_price)
            else:
                change = trend[i] + noise[i]
                new_price = prices[0] * (1 + trend[i] + np.cumsum(noise[:i+1])[-1]/days)
                prices.append(max(new_price, base_price * 0.9))  # Minimum 10% drop
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices],
            'close': prices,
            'volume': np.random.randint(500000, 2000000, days)
        }, index=dates)
        
        return df

async def test_technical_only():
    """Test just the technical analyzer"""
    print("\n🔧 Testing Technical Analyzer Only")
    print("-" * 40)
    
    try:
        from zerodha_technical_analyzer import ZerodhaTechnicalAnalyzer
        
        mock_client = MockZerodhaClient()
        analyzer = ZerodhaTechnicalAnalyzer(mock_client)
        
        # Test with mock market data
        test_symbols = ['RELIANCE', 'HDFCBANK']
        
        for symbol in test_symbols:
            print(f"\n📊 Testing {symbol}...")
            
            # Mock market data
            market_data = {
                'current_price': 2145.50 if symbol == 'RELIANCE' else 1650.25,
                'volume': 1500000,
                'change': 25.30,
                'change_percent': 1.19
            }
            
            # Run technical analysis
            result = await analyzer.analyze_symbol_for_options(
                symbol, market_data['current_price'], market_data, 'intraday'
            )
            
            print(f"✅ Analysis completed for {symbol}")
            print(f"   Market Bias: {result['market_bias']}")
            print(f"   Confidence: {result['confidence_score']:.1%}")
            
            # Fix: entry_signal is now a dictionary
            entry_signal = result['entry_signal']
            print(f"   Entry Signal: {entry_signal['signal_type']}")
            print(f"   Entry Reason: {entry_signal['reason']}")
            
            if entry_signal.get('entry_condition'):
                print(f"   Entry Condition: {entry_signal['entry_condition']}")
            
            print(f"   Support: ₹{result['support_resistance']['nearest_support']:.0f}")
            print(f"   Resistance: ₹{result['support_resistance']['nearest_resistance']:.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Technical analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_full_integration():
    """Test full integration with options analyzer"""
    print("\n🚀 Testing Full Integration")
    print("-" * 40)
    
    try:
        # Initialize analyzer the correct way (your __init__ expects ZerodhaAPIClient or None)
        analyzer = ZerodhaEnhancedOptionsAnalyzer(zerodha_client=None, claude_api_key=None)
        
        # Test with one symbol
        symbol = 'RELIANCE'
        print(f"📊 Testing full analysis for {symbol}...")
        
        result = await analyzer.analyze_trade(
            symbol=symbol,
            trading_style='intraday',
            risk_tolerance='medium',
            capital=100000,
            execute_trades=False
        )
        
        print(f"✅ Full analysis completed for {symbol}")
        
        # Display key results
        if 'technical_analysis' in result:
            tech = result['technical_analysis']
            print(f"📈 Market Bias: {tech['market_bias']}")
            print(f"🎯 Confidence: {tech['confidence_score']:.1%}")
        
        if 'trade_recommendation' in result:
            trade = result['trade_recommendation']
            print(f"💡 Strategy: {trade.get('strategy', 'N/A')}")
            print(f"🎖️ Confidence: {trade.get('confidence', 0):.1%}")
            
            if 'entry_rules' in trade:
                entry = trade['entry_rules']
                print(f"🎯 Entry: {entry.get('entry_condition', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🚀 Testing Technical Analysis Integration")
    print("=" * 50)
    
    # First test technical analyzer alone
    tech_success = await test_technical_only()
    
    if tech_success and not TEST_TECH_ONLY:
        # Then test full integration
        full_success = await test_full_integration()
        
        if full_success:
            print("\n🎉 All tests passed! Integration is working!")
        else:
            print("\n⚠️ Technical analyzer works, but full integration has issues")
    elif tech_success:
        print("\n✅ Technical analyzer test passed!")
        print("💡 Next: Fix the options analyzer import to test full integration")
    else:
        print("\n❌ Technical analyzer test failed")

if __name__ == "__main__":
    asyncio.run(main())