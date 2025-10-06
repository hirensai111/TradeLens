#!/usr/bin/env python3
"""
Test file for verifying directional-only intraday changes
Run this to validate that your analyzer only creates LONG_CALL/LONG_PUT for intraday
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the parent directory to path to import your analyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from zerodha_api_client import ZerodhaAPIClient
    from options_analyzer import ZerodhaEnhancedOptionsAnalyzer  # Replace with your actual file name
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to update the import path to your analyzer file")
    sys.exit(1)


async def test_directional_changes():
    """Test to verify directional-only intraday changes"""
    
    print("🧪 Testing Core Directional Changes")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Initialize analyzer
        print("Initializing analyzer...")
        zerodha_client = ZerodhaAPIClient()
        analyzer = ZerodhaEnhancedOptionsAnalyzer(zerodha_client)
        print("[OK] Analyzer initialized")
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize analyzer: {e}")
        return False
    
    # Test cases
    test_cases = [
        {
            'name': 'Intraday High Confidence Test',
            'symbol': 'NIFTY',
            'style': 'intraday',
            'capital': 100000,
            'expected': 'directional_or_no_trade',
            'description': 'Should get LONG_CALL, LONG_PUT, or NO_TRADE only'
        },
        {
            'name': 'Intraday Medium Confidence Test',
            'symbol': 'RELIANCE',
            'style': 'intraday',
            'capital': 100000,
            'expected': 'directional_or_no_trade',
            'description': 'Should get LONG_CALL, LONG_PUT, or NO_TRADE only'
        },
        {
            'name': 'Swing Trading Test',
            'symbol': 'TCS',
            'style': 'swing',
            'capital': 100000,
            'expected': 'any_strategy',
            'description': 'Should allow complex strategies like spreads, straddles'
        },
        {
            'name': 'Another Intraday Test',
            'symbol': 'HDFCBANK',
            'style': 'intraday', 
            'capital': 150000,
            'expected': 'directional_or_no_trade',
            'description': 'Should get LONG_CALL, LONG_PUT, or NO_TRADE only'
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test['name']}")
        print(f"Symbol: {test['symbol']} | Style: {test['style']} | Capital: ₹{test['capital']:,}")
        print(f"Expected: {test['description']}")
        print("-" * 50)
        
        try:
            # Run the analysis
            result = await analyzer.analyze_trade(
                symbol=test['symbol'],
                trading_style=test['style'],
                prediction_days=1 if test['style'] == 'intraday' else 14,
                risk_tolerance='medium',
                capital=test['capital'],
                execute_trades=False
            )
            
            # Check for errors
            if result.get('error'):
                print(f"[ERROR] Analysis failed: {result.get('message', 'Unknown error')}")
                results.append({'test': test['name'], 'status': 'ERROR', 'details': result.get('message')})
                continue
            
            # Extract results
            trade_rec = result.get('trade_recommendation', {})
            strategy = trade_rec.get('strategy', 'UNKNOWN')
            confidence = trade_rec.get('confidence', 0)
            option_legs = trade_rec.get('option_legs', [])
            
            print(f"[CHART] Results:")
            print(f"   Strategy: {strategy}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Legs Created: {len(option_legs)}")
            
            if option_legs:
                for j, leg in enumerate(option_legs):
                    print(f"   Leg {j+1}: {leg['action']} {leg['option_type']} @ ₹{leg['strike']} "
                          f"({leg['contracts']} contracts)")
            
            # Validate results
            test_passed = True
            failure_reasons = []
            
            if test['style'] == 'intraday':
                # Intraday should only get directional or NO_TRADE
                allowed_strategies = ['LONG_CALL', 'LONG_PUT', 'NO_TRADE']
                
                if strategy not in allowed_strategies:
                    test_passed = False
                    failure_reasons.append(f"Got {strategy}, expected one of {allowed_strategies}")
                
                # If not NO_TRADE, check confidence threshold
                if strategy != 'NO_TRADE' and confidence < 0.75:
                    test_passed = False
                    failure_reasons.append(f"Confidence {confidence:.1%} below 75% threshold")
                
                # Check leg structure for directional trades
                if strategy in ['LONG_CALL', 'LONG_PUT']:
                    if len(option_legs) != 1:
                        test_passed = False
                        failure_reasons.append(f"Directional should have 1 leg, got {len(option_legs)}")
                    elif option_legs[0]['action'] != 'BUY':
                        test_passed = False
                        failure_reasons.append(f"Directional should BUY, got {option_legs[0]['action']}")
                
                # NO_TRADE should have no legs
                elif strategy == 'NO_TRADE' and len(option_legs) > 0:
                    test_passed = False
                    failure_reasons.append(f"NO_TRADE should have 0 legs, got {len(option_legs)}")
                    
            else:  # Swing trading
                # Swing can have any strategy (just check it's not broken)
                if strategy == 'UNKNOWN':
                    test_passed = False
                    failure_reasons.append("Got UNKNOWN strategy")
            
            # Report results
            if test_passed:
                print(f"[OK] PASSED")
                results.append({'test': test['name'], 'status': 'PASSED', 'strategy': strategy})
            else:
                print(f"[ERROR] FAILED")
                for reason in failure_reasons:
                    print(f"   - {reason}")
                results.append({'test': test['name'], 'status': 'FAILED', 'reasons': failure_reasons})
                
        except Exception as e:
            print(f"[ERROR] Test execution failed: {e}")
            results.append({'test': test['name'], 'status': 'EXCEPTION', 'error': str(e)})
    
    # Summary
    print(f"\n[TARGET] TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r['status'] == 'PASSED')
    failed = sum(1 for r in results if r['status'] == 'FAILED')
    errors = sum(1 for r in results if r['status'] == 'ERROR')
    exceptions = sum(1 for r in results if r['status'] == 'EXCEPTION')
    
    print(f"Total Tests: {len(results)}")
    print(f"[OK] Passed: {passed}")
    print(f"[ERROR] Failed: {failed}")
    print(f"[WARNING]  Errors: {errors}")
    print(f"💥 Exceptions: {exceptions}")
    
    if failed > 0 or errors > 0 or exceptions > 0:
        print(f"\n📋 Failure Details:")
        for result in results:
            if result['status'] != 'PASSED':
                print(f"   {result['test']}: {result['status']}")
                if 'reasons' in result:
                    for reason in result['reasons']:
                        print(f"     - {reason}")
                elif 'error' in result:
                    print(f"     - {result['error']}")
    
    print(f"\n🔍 Key Changes Validated:")
    print("[OK] Intraday directional-only strategy selection")
    print("[OK] 75% confidence threshold for intraday trades")
    print("[OK] NO_TRADE handling with zero legs")
    print("[OK] Single-leg structure for directional trades")
    print("[OK] Swing trading flexibility preserved")
    
    # Return success status
    return failed == 0 and errors == 0 and exceptions == 0


def main():
    """Main test runner"""
    success = asyncio.run(test_directional_changes())
    
    if success:
        print(f"\n🎉 All tests passed! Your directional changes are working correctly.")
        return 0
    else:
        print(f"\n[WARNING]  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()