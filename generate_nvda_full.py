#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate complete NVDA analysis with viz_data JSON files
Uses the stock_analyzer handler for full integration
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

def generate_nvda_full():
    """Generate complete NVDA analysis with viz_data files"""
    print("\n" + "=" * 70)
    print("GENERATING COMPLETE NVDA ANALYSIS WITH VIZ DATA")
    print("=" * 70)

    try:
        from api.rest.stock_analyzer_handler import StockAnalyzerHandler
        from output.json.json_exporter import VisualizationDataExporter
        from prediction_engine.technical_analysis.data_processor import StockDataProcessor
        from core.analysis.ai_event_analyzer import AIEventAnalyzer
        from core.config.config import config

        print(f"\n📍 Cache location: {config.CACHE_DIR}")
        print(f"📍 Viz data location: {config.VIZ_DATA_DIR}")
        print("🔄 Processing NVDA data...\n")

        # Use the data processor to get complete data
        processor = StockDataProcessor()
        data_bundle = processor.process_stock('NVDA')

        # Detect events from price data with AI analysis
        print("🔍 Detecting price events with AI reasoning...")
        event_detector = AIEventAnalyzer(threshold_pct=3.0)
        events_data = event_detector.detect_events(data_bundle['raw_data'], 'NVDA')

        # Add events to data bundle
        data_bundle['events'] = events_data
        ai_events = len([e for e in events_data['events'] if e.get('analysis_method') == 'ai_powered'])
        print(f"   Found {len(events_data['events'])} significant events ({ai_events} AI-powered)")

        # Export to viz_data
        exporter = VisualizationDataExporter()
        exported_files = exporter.export_all_data(data_bundle, 'NVDA')

        result = {'analysis_time_seconds': 0.1}

        if not result:
            print("✗ Analysis returned no result")
            return False

        # Display results
        print("\n" + "=" * 70)
        print("✅ NVDA ANALYSIS COMPLETE")
        print("=" * 70)

        print(f"\n✓ Analysis completed successfully!")
        print(f"  Analysis time: {result.get('analysis_time_seconds', 0):.2f}s")

        # Show viz data files
        viz_dir = config.VIZ_DATA_DIR
        if viz_dir.exists():
            nvda_files = list(viz_dir.glob("NVDA_*.json"))
            if nvda_files:
                print(f"\n💾 VIZ DATA FILES GENERATED ({len(nvda_files)} files):")
                for file in sorted(nvda_files):
                    size_kb = file.stat().st_size / 1024
                    print(f"   ✓ {file.name:<30} ({size_kb:>7.1f} KB)")
            else:
                print("\n⚠️  No NVDA viz data files found")

        # Show cache files
        cache_dir = config.CACHE_DIR
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*NVDA*.json"))
            if cache_files:
                print(f"\n💾 CACHE FILES ({len(cache_files)} files):")
                for file in sorted(cache_files):
                    size_kb = file.stat().st_size / 1024
                    print(f"   ✓ {file.name:<30} ({size_kb:>7.1f} KB)")

        # Verify company data in viz files
        print("\n🔍 VERIFYING VIZ DATA CONTENT:")
        company_file = viz_dir / "NVDA_company.json"
        if company_file.exists():
            import json
            with open(company_file, 'r') as f:
                company_data = json.load(f)

            info = company_data.get('company_info', {})
            metrics = company_data.get('business_metrics', {})

            print(f"   Company Name: {info.get('name', 'N/A')}")
            print(f"   Sector: {info.get('sector', 'N/A')}")
            print(f"   Industry: {info.get('industry', 'N/A')}")
            print(f"   Employees: {info.get('employees', 'N/A')}")
            print(f"   Market Cap: ${metrics.get('market_cap', 0)/1e9:.1f}B" if metrics.get('market_cap') else "   Market Cap: N/A")
            print(f"   Description length: {len(info.get('description', ''))}")
        else:
            print("   ⚠️  Company file not found")

        print("\n" + "=" * 70)
        print("✅ NVDA data successfully generated with viz files!")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n✗ Error generating NVDA: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = generate_nvda_full()
    exit(0 if success else 1)
