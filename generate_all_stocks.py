#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate AI-powered analysis for all stocks
"""

import sys
from pathlib import Path
import time

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# List of all stocks to analyze
STOCKS = [
    'AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA', 'META', 'AMZN',
    'NFLX', 'AMD', 'INTC', 'PLTR', 'RDDT', 'ABNB', 'ADBE',
    'GOOG', 'NKE', 'F', 'COF', 'MSTR', 'BABA', 'ADDYY', 'IBM'
]

def generate_stock_analysis(ticker: str) -> bool:
    """Generate complete analysis for a single stock"""
    try:
        from output.json.json_exporter import VisualizationDataExporter
        from prediction_engine.technical_analysis.data_processor import StockDataProcessor
        from core.analysis.ai_event_analyzer import AIEventAnalyzer
        from core.config.config import config

        print(f"\n{'='*70}")
        print(f"📊 PROCESSING: {ticker}")
        print(f"{'='*70}")

        # Process stock data
        processor = StockDataProcessor()
        data_bundle = processor.process_stock(ticker)

        # Detect events with AI analysis
        print(f"   🔍 Analyzing price events with AI...")
        event_detector = AIEventAnalyzer(threshold_pct=3.0)
        events_data = event_detector.detect_events(data_bundle['raw_data'], ticker)

        # Add events to bundle
        data_bundle['events'] = events_data
        ai_events = len([e for e in events_data['events'] if e.get('analysis_method') == 'ai_powered'])
        print(f"   ✓ Found {len(events_data['events'])} events ({ai_events} AI-powered)")

        # Export to viz_data
        exporter = VisualizationDataExporter()
        exported_files = exporter.export_all_data(data_bundle, ticker)

        print(f"   ✓ Exported to viz_data")

        # Copy to stock_analyzer location
        import shutil
        for file_type, file_path in exported_files.items():
            dest = config.PROJECT_ROOT / "stock_analyzer" / "output" / "viz_data" / Path(file_path).name
            shutil.copy2(file_path, dest)

        print(f"   ✓ Copied to stock_analyzer/output/viz_data")
        return True

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Generate analysis for all stocks"""
    print("\n" + "=" * 70)
    print("AI-POWERED ANALYSIS FOR ALL STOCKS")
    print("=" * 70)
    print(f"\nTotal stocks to process: {len(STOCKS)}")
    print(f"Stocks: {', '.join(STOCKS)}\n")

    results = {}
    start_time = time.time()

    for i, ticker in enumerate(STOCKS, 1):
        print(f"\n[{i}/{len(STOCKS)}] Starting {ticker}...")

        success = generate_stock_analysis(ticker)
        results[ticker] = 'SUCCESS' if success else 'FAILED'

        # Small delay to avoid rate limiting
        if i < len(STOCKS):
            time.sleep(2)

    # Summary
    elapsed_time = time.time() - start_time
    successful = sum(1 for v in results.values() if v == 'SUCCESS')
    failed = sum(1 for v in results.values() if v == 'FAILED')

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n📊 Results:")
    print(f"   Total Stocks:    {len(STOCKS)}")
    print(f"   ✓ Successful:    {successful}")
    print(f"   ✗ Failed:        {failed}")
    print(f"   ⏱  Total Time:    {elapsed_time/60:.1f} minutes")
    print(f"   📈 Avg per Stock: {elapsed_time/len(STOCKS):.1f} seconds")

    # Show status for each stock
    print(f"\n📋 Detailed Results:")
    for ticker, status in results.items():
        symbol = '✓' if status == 'SUCCESS' else '✗'
        print(f"   {symbol} {ticker:<8} {status}")

    # Show failed stocks if any
    if failed > 0:
        failed_stocks = [t for t, s in results.items() if s == 'FAILED']
        print(f"\n⚠️  Failed Stocks: {', '.join(failed_stocks)}")

    print("\n" + "=" * 70)
    print(f"✅ All viz_data files saved to:")
    print(f"   • output/viz_data/")
    print(f"   • stock_analyzer/output/viz_data/")
    print("=" * 70 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
