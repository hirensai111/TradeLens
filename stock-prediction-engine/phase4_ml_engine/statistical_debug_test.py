#!/usr/bin/env python3
"""
Statistical Analysis Debug Test
Find where the 271% Expected Daily Return is coming from
"""

import sys
import os
import math
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add your paths
sys.path.append('src')
sys.path.append('src/data_loaders')

def debug_statistical_analysis():
    """
    Debug test to trace the 271% Expected Daily Return bug
    """
    
    print("="*100)
    print("🔍 STATISTICAL ANALYSIS DEBUG TEST - FIND THE 271% BUG")
    print("="*100)
    print("Tracing: Excel Data → Mathematical Analysis → Statistical Projections")
    print()

    # Configuration
    ticker = "TEM"
    excel_path = "D:/stock-prediction-engine/phase4_ml_engine/data/TEM_analysis_report_20250715.xlsx"
    prediction_days = 5
    
    print(f"📊 TEST CONFIGURATION:")
    print(f"   Ticker: {ticker}")
    print(f"   Excel Path: {excel_path}")
    print(f"   Prediction Days: {prediction_days}")
    print()

    # ============================================================================
    # PHASE 1: EXCEL DATA EXTRACTION DEBUG
    # ============================================================================
    print("PHASE 1: EXCEL DATA EXTRACTION DEBUG")
    print("="*80)
    
    try:
        from prediction_engine import StockPredictionEngine
        
        # Test 1: Load Excel Data
        print("1.1 EXCEL DATA LOADING:")
        engine = StockPredictionEngine()
        excel_data = engine.load_excel_historical_data(ticker, excel_path)
        
        print(f"   📊 Excel data loaded: {len(excel_data)} keys")
        
        # Test 2: Check avg_daily_change specifically
        print("\n1.2 AVG_DAILY_CHANGE DEBUG:")
        avg_daily_change = excel_data.get('avg_daily_change', 'NOT_FOUND')
        print(f"   📊 avg_daily_change from Excel: {avg_daily_change}")
        print(f"   📊 Type: {type(avg_daily_change)}")
        
        if isinstance(avg_daily_change, (int, float)):
            print(f"   📊 Absolute value: {abs(avg_daily_change)}")
            print(f"   📊 Is > 100?: {abs(avg_daily_change) > 100}")
            print(f"   📊 Is > 50?: {abs(avg_daily_change) > 50}")
            print(f"   📊 Is > 10?: {abs(avg_daily_change) > 10}")
            print(f"   📊 Is > 1?: {abs(avg_daily_change) > 1}")
        
        # Test 3: Check other related values
        print("\n1.3 RELATED VALUES DEBUG:")
        related_keys = ['max_gain', 'max_loss', 'positive_days_pct', 'volatility']
        for key in related_keys:
            value = excel_data.get(key, 'NOT_FOUND')
            print(f"   📊 {key}: {value}")
        
        excel_debug_results = {
            'avg_daily_change': avg_daily_change,
            'excel_data_keys': list(excel_data.keys()),
            'related_values': {key: excel_data.get(key) for key in related_keys}
        }
        
    except Exception as e:
        print(f"❌ Excel Data Phase Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ============================================================================
    # PHASE 2: RAW EXCEL FILE ANALYSIS
    # ============================================================================
    print("\n\nPHASE 2: RAW EXCEL FILE ANALYSIS")
    print("="*80)
    
    try:
        from excel_loader import ExcelDataLoader
        
        # Test 1: Direct Excel Loader
        print("2.1 DIRECT EXCEL LOADER ANALYSIS:")
        loader = ExcelDataLoader()
        loader.excel_path = excel_path
        all_data = loader.load_all_data()
        
        # Test 2: Check Raw Data Daily Changes
        print("\n2.2 RAW DATA DAILY CHANGES:")
        if loader.raw_data is not None and not loader.raw_data.empty:
            raw_data = loader.raw_data
            print(f"   📊 Raw data shape: {raw_data.shape}")
            
            if 'Daily Change %' in raw_data.columns:
                daily_changes = pd.to_numeric(raw_data['Daily Change %'], errors='coerce').dropna()
                print(f"   📊 Daily changes count: {len(daily_changes)}")
                print(f"   📊 Daily changes mean: {daily_changes.mean()}")
                print(f"   📊 Daily changes std: {daily_changes.std()}")
                print(f"   📊 Daily changes min: {daily_changes.min()}")
                print(f"   📊 Daily changes max: {daily_changes.max()}")
                
                # Show sample values
                print(f"   📊 First 10 daily changes:")
                for i, val in enumerate(daily_changes.head(10)):
                    print(f"      {i+1}: {val}")
                
                # Check if values are in percentage or decimal
                print(f"\n   🔍 UNIT ANALYSIS:")
                print(f"      Mean absolute: {abs(daily_changes.mean())}")
                if abs(daily_changes.mean()) > 50:
                    print(f"      ⚠️ Values appear to be in BASIS POINTS (need /10000)")
                elif abs(daily_changes.mean()) > 1:
                    print(f"      ⚠️ Values appear to be in PERCENTAGE POINTS (need /100)")
                else:
                    print(f"      ✅ Values appear to be in DECIMAL form")
        
        raw_excel_results = {
            'raw_data_shape': raw_data.shape if loader.raw_data is not None else None,
            'daily_changes_mean': daily_changes.mean() if 'daily_changes' in locals() else None,
            'daily_changes_count': len(daily_changes) if 'daily_changes' in locals() else None
        }
        
    except Exception as e:
        print(f"❌ Raw Excel Analysis Failed: {e}")
        import traceback
        traceback.print_exc()
        raw_excel_results = {}

    # ============================================================================
    # PHASE 3: MATHEMATICAL ANALYSIS DEBUG
    # ============================================================================
    print("\n\nPHASE 3: MATHEMATICAL ANALYSIS DEBUG")
    print("="*80)
    
    try:
        # Test 1: Mock Market Data
        print("3.1 MOCK MARKET DATA:")
        mock_market_data = {
            'current_price': 59.00,
            'change_percent': -3.47,
            'data_source': 'debug_test'
        }
        print(f"   ✅ Mock market data created")
        
        # Test 2: Call Mathematical Analysis with Debug
        print("\n3.2 MATHEMATICAL ANALYSIS CALL:")
        print(f"   📊 INPUT avg_daily_change: {excel_data.get('avg_daily_change')}")
        
        # MONKEY PATCH the method to add debug
        original_method = engine.perform_mathematical_analysis
        
        def debug_mathematical_analysis(excel_data, market_data, custom_articles=None, prediction_days=1):
            print(f"\n   🔍 INSIDE MATHEMATICAL ANALYSIS:")
            
            # Check inputs
            avg_daily_change = excel_data.get('avg_daily_change', 0.5)
            print(f"      📊 avg_daily_change extracted: {avg_daily_change}")
            
            volatility = excel_data.get('volatility', 3.0)
            daily_vol = volatility / math.sqrt(252)
            print(f"      📊 volatility: {volatility}")
            print(f"      📊 daily_vol: {daily_vol}")
            
            # Trace the statistical analysis section
            print(f"\n   🔍 STATISTICAL ANALYSIS SECTION:")
            
            # The problematic calculation
            np.random.seed(42)
            
            # Check the fix
            print(f"      📊 Original avg_daily_change: {avg_daily_change}")
            daily_return_mean = avg_daily_change / 100  # Use the same fix as in production
            print(f"      📊 Converted daily_return_mean: {daily_return_mean}")
            
            returns = np.random.normal(daily_return_mean, daily_vol, prediction_days)
            print(f"      📊 Generated returns (first 5): {returns[:5] if len(returns) >= 5 else returns}")
            print(f"      📊 Returns mean: {np.mean(returns)}")
            print(f"      📊 Returns std: {np.std(returns)}")
            
            # The final calculation that goes in the report
            mean_return_daily_percent = np.mean(returns) * 100
            print(f"      📊 FINAL mean_return_daily (for report): {mean_return_daily_percent}%")
            
            # Call original method
            return original_method(excel_data, market_data, custom_articles, prediction_days)
        
        # Replace method temporarily
        engine.perform_mathematical_analysis = debug_mathematical_analysis
        
        # Call the analysis
        math_analysis = engine.perform_mathematical_analysis(
            excel_data, mock_market_data, custom_articles=None, prediction_days=prediction_days
        )
        
        # Restore original method
        engine.perform_mathematical_analysis = original_method
        
        # Check the results
        print(f"\n3.3 MATHEMATICAL ANALYSIS RESULTS:")
        stat_metrics = math_analysis.get('statistical_metrics', {})
        mean_return_daily = stat_metrics.get('mean_return_daily', 'NOT_FOUND')
        print(f"   📊 Final mean_return_daily in results: {mean_return_daily}")
        
        math_debug_results = {
            'mean_return_daily': mean_return_daily,
            'statistical_metrics': stat_metrics
        }
        
    except Exception as e:
        print(f"❌ Mathematical Analysis Phase Failed: {e}")
        import traceback
        traceback.print_exc()
        math_debug_results = {}

    # ============================================================================
    # PHASE 4: ROOT CAUSE ANALYSIS
    # ============================================================================
    print("\n\nPHASE 4: ROOT CAUSE ANALYSIS")
    print("="*80)
    
    print("4.1 TRACING THE 271% BUG:")
    
    # Trace the path
    excel_avg_change = excel_debug_results.get('avg_daily_change', 'ERROR')
    final_mean_return = math_debug_results.get('mean_return_daily', 'ERROR')
    
    print(f"   📊 Excel avg_daily_change: {excel_avg_change}")
    print(f"   📊 Final mean_return_daily: {final_mean_return}")
    
    if isinstance(excel_avg_change, (int, float)) and isinstance(final_mean_return, (int, float)):
        ratio = final_mean_return / excel_avg_change if excel_avg_change != 0 else 0
        print(f"   📊 Ratio (final/excel): {ratio}")
        
        # Analysis
        if abs(final_mean_return - 271) < 50:  # If it's around 271%
            print(f"\n4.2 BUG ANALYSIS:")
            print(f"   🚨 CONFIRMED: 271% bug detected!")
            
            if abs(excel_avg_change) > 1000:
                print(f"   🔍 Root cause: avg_daily_change is in BASIS POINTS")
                print(f"   🔧 Fix needed: Divide by 10000 when reading from Excel")
            elif abs(excel_avg_change) > 100:
                print(f"   🔍 Root cause: avg_daily_change is in PERCENTAGE POINTS")
                print(f"   🔧 Fix needed: Divide by 100 when reading from Excel")
            elif abs(excel_avg_change) > 10:
                print(f"   🔍 Root cause: avg_daily_change is in PERCENT")
                print(f"   🔧 Fix needed: Divide by 100 when reading from Excel")
            else:
                print(f"   🔍 Root cause: Issue in mathematical analysis conversion")
                print(f"   🔧 Fix needed: Check the statistical analysis calculation")
        else:
            print(f"\n4.2 BUG STATUS:")
            print(f"   ✅ 271% bug not detected in this run")
            print(f"   📊 Actual value: {final_mean_return}")

    # ============================================================================
    # PHASE 5: RECOMMENDED FIXES
    # ============================================================================
    print("\n\nPHASE 5: RECOMMENDED FIXES")
    print("="*80)
    
    print("5.1 SPECIFIC FIXES NEEDED:")
    
    if 'raw_excel_results' in locals() and raw_excel_results.get('daily_changes_mean'):
        raw_mean = raw_excel_results['daily_changes_mean']
        if abs(raw_mean) > 1000:
            print(f"   📝 In load_excel_historical_data method:")
            print(f"      Line ~200: historical_analysis['avg_daily_change'] = float(daily_changes.mean())")
            print(f"      REPLACE WITH:")
            print(f"      raw_mean = float(daily_changes.mean())")
            print(f"      historical_analysis['avg_daily_change'] = raw_mean / 10000  # Convert from basis points")
        elif abs(raw_mean) > 100:
            print(f"   📝 In load_excel_historical_data method:")
            print(f"      Line ~200: historical_analysis['avg_daily_change'] = float(daily_changes.mean())")
            print(f"      REPLACE WITH:")
            print(f"      raw_mean = float(daily_changes.mean())")
            print(f"      historical_analysis['avg_daily_change'] = raw_mean / 100  # Convert from percentage")
        elif abs(raw_mean) > 10:
            print(f"   📝 In load_excel_historical_data method:")
            print(f"      Line ~200: historical_analysis['avg_daily_change'] = float(daily_changes.mean())")
            print(f"      REPLACE WITH:")
            print(f"      raw_mean = float(daily_changes.mean())")
            print(f"      historical_analysis['avg_daily_change'] = raw_mean / 100  # Convert from percentage")
    
    print(f"\n5.2 TEST VERIFICATION:")
    print(f"   1. Apply the fix above")
    print(f"   2. Run this debug script again")
    print(f"   3. Expected result: mean_return_daily should be ~0.001% to 0.01%")
    print(f"   4. Run full analysis - Expected Daily Return should be realistic")

    print("\n" + "="*100)
    print("🎯 STATISTICAL DEBUG TEST COMPLETE")
    print("="*100)
    
    return {
        'excel_debug_results': excel_debug_results,
        'raw_excel_results': raw_excel_results if 'raw_excel_results' in locals() else {},
        'math_debug_results': math_debug_results if 'math_debug_results' in locals() else {}
    }

if __name__ == "__main__":
    # Run the statistical debug test
    print(f"🔍 Starting Statistical Analysis Debug Test...")
    print(f"Target: Find where the 271% Expected Daily Return comes from")
    print()
    
    results = debug_statistical_analysis()
    
    print(f"\n💡 QUICK SUMMARY:")
    if results:
        excel_avg = results.get('excel_debug_results', {}).get('avg_daily_change', 'N/A')
        final_result = results.get('math_debug_results', {}).get('mean_return_daily', 'N/A')
        
        print(f"   Excel avg_daily_change: {excel_avg}")
        print(f"   Final mean_return_daily: {final_result}")
        
        if isinstance(excel_avg, (int, float)) and abs(excel_avg) > 100:
            print(f"   🎯 ROOT CAUSE: avg_daily_change needs unit conversion!")
            print(f"   🔧 APPLY FIX: Divide by appropriate factor when reading from Excel")
        else:
            print(f"   🔍 INVESTIGATE: Check mathematical analysis conversion")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Review the specific fix recommendations above")
    print(f"   2. Apply the fix to prediction_engine.py")
    print(f"   3. Run this test again to verify")
    print(f"   4. Run full analysis to confirm 271% → realistic %")