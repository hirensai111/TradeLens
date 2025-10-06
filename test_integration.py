#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify integration fixes.
Tests data flow from analyzer through prediction engine.
"""

import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_analyzer():
    """Test stock analyzer with unified cache"""
    print("\n" + "=" * 70)
    print("TEST 1: Stock Analyzer with Unified Cache")
    print("=" * 70)

    try:
        from prediction_engine.technical_analysis.data_processor import StockDataProcessor

        processor = StockDataProcessor()
        print("✓ Processor initialized")

        # Test with a fetch to ensure we can get new data
        result = processor.process_stock('NVDA')

        if result and 'ticker' in result:
            print(f"✓ Processing successful for NVDA")
            print(f"  Data points: {result.get('metadata', {}).get('data_points', 0)}")
            print(f"  Current price: ${result.get('summary_statistics', {}).get('current_price', 0):.2f}")
            return True
        else:
            print(f"✗ Processing failed")
            return False

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_location():
    """Verify cache is using unified location"""
    print("\n" + "=" * 70)
    print("TEST 2: Cache Location Verification")
    print("=" * 70)

    try:
        from core.config.config import config
        from pathlib import Path

        print(f"Cache directory: {config.CACHE_DIR}")

        # Check if cache directory exists
        if config.CACHE_DIR.exists():
            print("✓ Unified cache directory exists")

            # Count files
            json_files = list(config.CACHE_DIR.glob("*.json"))
            print(f"✓ Found {len(json_files)} JSON files in cache")

            # Check for NVDA files
            nvda_files = [f for f in json_files if 'NVDA' in f.name]
            if nvda_files:
                print(f"✓ Found {len(nvda_files)} NVDA cache files:")
                for f in nvda_files:
                    size_kb = f.stat().st_size / 1024
                    print(f"    - {f.name} ({size_kb:.1f} KB)")
            else:
                print("⚠ No NVDA files in cache yet")

            return True
        else:
            print("✗ Unified cache directory does not exist")
            return False

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_prediction_engine():
    """Test prediction engine can load data from unified cache"""
    print("\n" + "=" * 70)
    print("TEST 3: Prediction Engine Integration")
    print("=" * 70)

    try:
        from prediction_engine.data_loaders.json_loader import JSONDataLoader

        loader = JSONDataLoader()
        print(f"✓ Loader initialized with cache: {loader.cache_dir}")

        # Try to load NVDA data
        all_data = loader.load_all_data('NVDA')

        if all_data and 'raw_data' in all_data:
            print(f"✓ Successfully loaded NVDA data")
            print(f"  Data sources: {list(all_data.keys())}")
            print(f"  Raw data rows: {len(all_data['raw_data'])}")

            # Test feature extraction
            from datetime import datetime
            features = loader.get_enhanced_features_for_date(datetime.now())
            print(f"✓ Extracted {len(features)} features")

            # Test volatility
            volatility = loader.get_proper_volatility()
            print(f"✓ Volatility: {volatility:.2f}%")

            return True
        else:
            print("✗ Failed to load NVDA data")
            return False

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_date_column():
    """Verify Date column is preserved in cached data"""
    print("\n" + "=" * 70)
    print("TEST 4: Date Column Preservation")
    print("=" * 70)

    try:
        import json
        from core.config.config import config

        cache_file = config.CACHE_DIR / "historical_NVDA_5y.json"

        if not cache_file.exists():
            print("⚠ NVDA historical cache not found, skipping test")
            return True

        with open(cache_file, 'r') as f:
            data = json.load(f)

        if isinstance(data, list) and len(data) > 0:
            first_record = data[0]
            if 'Date' in first_record:
                print(f"✓ Date column found in cached data")
                print(f"  Sample date: {first_record['Date']}")
                print(f"  Total records: {len(data)}")
                return True
            else:
                print(f"✗ Date column NOT found in cached data")
                print(f"  Available columns: {list(first_record.keys())[:5]}")
                return False
        else:
            print("✗ Invalid cache data format")
            return False

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUITE")
    print("Testing all fixes for JSON file update issues")
    print("=" * 70)

    results = {}

    # Run all tests
    results['cache_location'] = test_cache_location()
    results['analyzer'] = test_analyzer()
    results['date_column'] = test_date_column()
    results['prediction_engine'] = test_prediction_engine()

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 70)

    if all_passed:
        print("✓ ALL TESTS PASSED - Integration is working!")
    else:
        print("✗ SOME TESTS FAILED - Check errors above")

    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
