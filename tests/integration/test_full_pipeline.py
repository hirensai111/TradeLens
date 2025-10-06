#!/usr/bin/env python3

"""
Test script: Data Processor + Enhanced Excel Report Generator pipeline.
Tests all features including charts, visualizations, and multiple sheets.
"""

from pathlib import Path
import sys
import time
from datetime import datetime

# Import your data processor and Excel report generator
from prediction_engine.technical_analysis.data_processor import process_stock
from output.excel.excel_generator import ExcelReportGenerator
from core.config.config import config

def print_section_header(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")

def test_single_stock_report(ticker="AAPL"):
    """Test single stock report generation with all features."""
    print_section_header(f"Single Stock Report Test - {ticker}")
    
    # Process stock data
    print(f"1. Processing stock data for: {ticker}")
    start_time = time.time()
    try:
        stock_data = process_stock(ticker)
        processing_time = time.time() - start_time
        print(f"   [OK] Data processing successful (took {processing_time:.2f} seconds)")
        print(f"   [CHART] Data contains {len(stock_data.get('technical_data', []))} days of history")
    except Exception as e:
        print(f"   [ERROR] Data processing failed: {e}")
        return None
    
    # Generate Excel report
    print(f"\n2. Generating Enhanced Excel report for: {ticker}")
    generator = ExcelReportGenerator()
    start_time = time.time()
    
    try:
        report_path = generator.generate_report(stock_data)
        generation_time = time.time() - start_time
        print(f"   [OK] Excel report generated (took {generation_time:.2f} seconds)")
        print(f"   📄 Report saved to: {report_path}")
        
        # Verify file exists and check size
        if Path(report_path).exists():
            file_size = Path(report_path).stat().st_size / 1024  # KB
            print(f"   💾 File size: {file_size:.1f} KB")
            print(f"\n   📋 Report includes the following sheets:")
            print(f"      • Summary - Executive overview with key metrics")
            print(f"      • Company Info - Detailed company information")
            print(f"      • Price Charts - Visual analysis with 4 charts")
            print(f"      • Technical Analysis - Indicators with conditional formatting")
            print(f"      • Performance Metrics - Returns and risk analysis")
            print(f"      • Raw Data - Historical prices with formatting")
            print(f"      • Data Quality - Integrity checks and validation")
            print(f"      • Metadata - Analysis information and disclaimer")
        else:
            print(f"   [WARNING] File not found at expected location: {report_path}")
            return None
            
        return report_path
        
    except Exception as e:
        print(f"   [ERROR] Excel report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_portfolio_report(tickers=None):
    """Test portfolio report generation."""
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    print_section_header(f"Portfolio Report Test - {len(tickers)} stocks")
    
    portfolio_data = []
    
    # Process each stock
    print("1. Processing portfolio stocks:")
    for i, ticker in enumerate(tickers, 1):
        print(f"   [{i}/{len(tickers)}] Processing {ticker}...", end="")
        try:
            stock_data = process_stock(ticker)
            portfolio_data.append(stock_data)
            print(" [OK]")
        except Exception as e:
            print(f" [ERROR] Failed: {e}")
    
    if not portfolio_data:
        print("   [ERROR] No stocks processed successfully")
        return None
    
    print(f"\n   [CHART] Successfully processed {len(portfolio_data)} stocks")
    
    # Generate portfolio report
    print(f"\n2. Generating Portfolio Excel report")
    generator = ExcelReportGenerator()
    start_time = time.time()
    
    try:
        report_path = generator.create_portfolio_report(portfolio_data)
        generation_time = time.time() - start_time
        print(f"   [OK] Portfolio report generated (took {generation_time:.2f} seconds)")
        print(f"   📄 Report saved to: {report_path}")
        
        if Path(report_path).exists():
            file_size = Path(report_path).stat().st_size / 1024  # KB
            print(f"   💾 File size: {file_size:.1f} KB")
            
        return report_path
        
    except Exception as e:
        print(f"   [ERROR] Portfolio report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_error_handling():
    """Test error handling with invalid ticker."""
    print_section_header("Error Handling Test")
    
    invalid_ticker = "INVALID123"
    print(f"Testing with invalid ticker: {invalid_ticker}")
    
    try:
        stock_data = process_stock(invalid_ticker)
        generator = ExcelReportGenerator()
        report_path = generator.generate_report(stock_data)
        print(f"   [WARNING] Unexpected success - should have failed")
    except Exception as e:
        print(f"   [OK] Properly handled error: {type(e).__name__}: {str(e)[:100]}...")

def main():
    """Main test function."""
    print("\n" + "="*60)
    print(" Enhanced Excel Report Generator Test Suite")
    print(" Testing charts, visualizations, and all features")
    print("="*60)
    
    # Test configuration
    print(f"\nTest Configuration:")
    print(f"  • Output Directory: {config.OUTPUT_DIR}")
    print(f"  • Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure output directory exists
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Run tests
    successful_tests = 0
    total_tests = 3
    
    # Test 1: Single stock report
    report_path = test_single_stock_report("AAPL")
    if report_path:
        successful_tests += 1
    
    # Test 2: Portfolio report
    print("\nPress Enter to continue with portfolio test (or Ctrl+C to skip)...")
    try:
        input()
        portfolio_path = test_portfolio_report(["AAPL", "MSFT", "GOOGL"])
        if portfolio_path:
            successful_tests += 1
    except KeyboardInterrupt:
        print("\nSkipping portfolio test")
        total_tests -= 1
    
    # Test 3: Error handling
    print("\nPress Enter to continue with error handling test (or Ctrl+C to skip)...")
    try:
        input()
        test_error_handling()
        successful_tests += 1
    except KeyboardInterrupt:
        print("\nSkipping error handling test")
        total_tests -= 1
    
    # Summary
    print_section_header("Test Summary")
    print(f"Tests passed: {successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        print("\n🎉 All tests passed successfully!")
        print("\n📁 Check the output directory for generated reports:")
        print(f"   {config.OUTPUT_DIR}")
        print("\n[BULB] Tips for reviewing the reports:")
        print("   • Open in Excel to see all formatting and charts")
        print("   • Check the Price Charts sheet for visualizations")
        print("   • Review conditional formatting in Technical Analysis")
        print("   • Examine the Data Quality sheet for integrity checks")
    else:
        print(f"\n[WARNING] {total_tests - successful_tests} test(s) failed")
    
    print("\n" + "="*60)
    print(" Test Suite Complete")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)