#!/usr/bin/env python3
"""
Debug script to check symbol mapping and find correct symbol names
"""

import os
from dotenv import load_dotenv
load_dotenv()

from zerodha_api_client import ZerodhaAPIClient

def debug_symbols():
    """Debug symbol mapping issues"""
    
    try:
        # Initialize client
        client = ZerodhaAPIClient()
        
        # Check what symbols we have for major stocks
        test_symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'SBIN']
        
        print("🔍 Debugging Symbol Mapping")
        print("=" * 50)
        
        print("\n📋 Checking test symbols:")
        for symbol in test_symbols:
            info = client.get_instrument_info(symbol)
            if info:
                print(f"✅ {symbol}: Token={info.get('instrument_token')}, Exchange={info.get('exchange')}")
            else:
                print(f"❌ {symbol}: Not found")
        
        # Check indices
        print("\n📊 Checking index symbols:")
        index_symbols = ['NIFTY 50', 'NIFTY BANK', 'NIFTY FIN SERVICE', 'NIFTY50', 'NIFTYBANK', 'NIFTYFIN']
        
        for symbol in index_symbols:
            info = client.get_instrument_info(symbol)
            if info:
                print(f"✅ {symbol}: Token={info.get('instrument_token')}, Exchange={info.get('exchange')}")
            else:
                print(f"❌ {symbol}: Not found")
        
        # Search for NIFTY related instruments
        print("\n🔎 Searching for NIFTY instruments:")
        nifty_instruments = []
        
        if hasattr(client, 'instruments_df') and client.instruments_df is not None:
            nifty_df = client.instruments_df[
                client.instruments_df['name'].str.contains('NIFTY', na=False) |
                client.instruments_df['tradingsymbol'].str.contains('NIFTY', na=False)
            ]
            
            # Show different NIFTY instruments
            for _, instrument in nifty_df.head(10).iterrows():
                print(f"  {instrument['tradingsymbol']}: {instrument['name']} ({instrument['exchange']}) - Token: {instrument['instrument_token']}")
        
        # Test a simple quote fetch
        print("\n🧪 Testing quote fetch with known token:")
        # NIFTY 50 token from fallback mapping
        test_tokens = [256265]  # NIFTY 50 token
        
        try:
            quotes = client.kite.quote(test_tokens)
            print(f"Quote response: {quotes}")
        except Exception as e:
            print(f"Quote fetch failed: {e}")
        
        # Test LTP fetch
        print("\n🧪 Testing LTP fetch:")
        try:
            ltp = client.kite.ltp(test_tokens)
            print(f"LTP response: {ltp}")
        except Exception as e:
            print(f"LTP fetch failed: {e}")
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")

if __name__ == "__main__":
    debug_symbols()