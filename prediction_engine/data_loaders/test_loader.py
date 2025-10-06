from prediction_engine.data_loaders.excel_loader import ExcelDataLoader
from datetime import datetime
import numpy as np

print("="*80)
print("ENHANCED EXCEL LOADER TEST - VOLATILITY & PERFORMANCE VERIFICATION")
print("="*80)

# Initialize loader
loader = ExcelDataLoader()

# Load all data
print("\n1. LOADING ALL DATA...")
all_data = loader.load_all_data()

# Test volatility calculation
print("\n2. VOLATILITY CALCULATION TEST:")
print("-"*50)
if hasattr(loader, 'calculated_volatility'):
    print(f"[OK] Calculated Volatility: {loader.calculated_volatility:.2f}%")
else:
    print("[ERROR] Volatility not calculated - check Technical Analysis processing")

# Test the get_proper_volatility method
print(f"[OK] get_proper_volatility(): {loader.get_proper_volatility():.2f}%")

# Check technical data details
print("\n3. TECHNICAL DATA ANALYSIS:")
print("-"*50)
if loader.technical_data is not None and not loader.technical_data.empty:
    print(f"Technical data rows: {len(loader.technical_data)}")
    print(f"Technical data columns: {list(loader.technical_data.columns)}")
    
    # Check if we have Close prices and calculated returns
    if 'Close' in loader.technical_data.columns:
        close_prices = loader.technical_data['Close'].dropna()
        print(f"Close price data points: {len(close_prices)}")
        print(f"Price range: ${close_prices.min():.2f} - ${close_prices.max():.2f}")
        
        # Show recent price movement for validation
        if len(close_prices) > 5:
            recent_prices = close_prices.tail(5)
            print(f"Recent 5 closes: {recent_prices.values}")
        
        # Check if daily returns were calculated
        if 'Daily_Return' in loader.technical_data.columns:
            returns = loader.technical_data['Daily_Return'].dropna()
            print(f"Daily returns calculated: {len(returns)} points")
            print(f"Daily return range: {returns.min():.4f} to {returns.max():.4f}")
            print(f"Daily volatility: {returns.std():.4f}")
            print(f"Annualized volatility: {returns.std() * np.sqrt(252):.4f} ({returns.std() * np.sqrt(252) * 100:.2f}%)")
        else:
            print("[ERROR] Daily_Return column not found")
    else:
        print("[ERROR] Close column not found in technical data")
else:
    print("[ERROR] No technical data loaded")

# Test performance data (existing functionality)
print("\n4. PERFORMANCE DATA TEST:")
print("-"*50)
loader.debug_performance_data()

# Test 30-day performance
print(f"\n5. DIRECT 30-DAY PERFORMANCE TEST:")
print("-"*50)
performance_30d = loader.get_30d_performance()
print(f"Direct 30-day performance: {performance_30d}%")

# Test feature extraction with new volatility
print("\n6. FEATURE EXTRACTION TEST (with proper volatility):")
print("-"*50)
features = loader.get_enhanced_features_for_date(datetime.now())

# Check volatility-related features
volatility_features = {k: v for k, v in features.items() 
                      if 'volatility' in k.lower() or 'vol' in k.lower()}

print("Volatility features found:")
for key, value in volatility_features.items():
    print(f"   {key}: {value}")

# Check performance features
performance_features = {k: v for k, v in features.items() 
                       if any(term in k.lower() for term in ['performance', 'return', 'month', '30d'])}

print(f"\nPerformance features found ({len(performance_features)}):")
for key, value in performance_features.items():
    if isinstance(value, (int, float)):
        print(f"   {key}: {value:.2f}%")
    else:
        print(f"   {key}: {value}")

# Validation checks
print("\n7. VALIDATION CHECKS:")
print("-"*50)

# Check 1: Volatility should be realistic (5% - 100%)
if hasattr(loader, 'calculated_volatility'):
    vol = loader.calculated_volatility
    if 5 <= vol <= 100:
        print(f"[OK] Volatility is realistic: {vol:.2f}%")
    else:
        print(f"[WARNING]  Volatility seems unrealistic: {vol:.2f}% (should be 5-100%)")
else:
    print("[ERROR] No calculated volatility to validate")

# Check 2: 30-day performance should be -18.0%
if abs(performance_30d - (-18.0)) < 0.1:
    print(f"[OK] 30-day performance correct: {performance_30d}%")
else:
    print(f"[WARNING]  30-day performance unexpected: {performance_30d}% (expected: -18.0%)")

# Check 3: All key performance aliases should match
key_aliases = ['return_1_month', 'monthly_return', '30d_return', '30d_performance']
alias_values = [features.get(alias, 'NOT_FOUND') for alias in key_aliases]
all_match = all(abs(val - (-18.0)) < 0.1 for val in alias_values if isinstance(val, (int, float)))

if all_match:
    print("[OK] All performance aliases correctly set to -18.0%")
else:
    print("[WARNING]  Performance aliases mismatch:")
    for alias, value in zip(key_aliases, alias_values):
        print(f"     {alias}: {value}")

# Summary
print("\n8. SUMMARY:")
print("-"*50)
print(f"[CHART] Technical data loaded: {'[OK]' if loader.technical_data is not None else '[ERROR]'}")
print(f"[UP] Volatility calculated: {'[OK]' if hasattr(loader, 'calculated_volatility') else '[ERROR]'}")
print(f"[DOWN] Performance data fixed: {'[OK]' if abs(performance_30d - (-18.0)) < 0.1 else '[ERROR]'}")
print(f"[TARGET] Ready for main analysis: {'[OK]' if hasattr(loader, 'calculated_volatility') and abs(performance_30d - (-18.0)) < 0.1 else '[ERROR]'}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)