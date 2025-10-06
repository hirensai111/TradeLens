#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate complete NVDA analysis with all data
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

def generate_nvda():
    """Generate complete NVDA analysis"""
    print("\n" + "=" * 70)
    print("GENERATING NVDA ANALYSIS")
    print("=" * 70)

    try:
        from prediction_engine.technical_analysis.data_processor import StockDataProcessor
        from core.config.config import config

        print(f"\n📍 Cache location: {config.CACHE_DIR}")
        print("🔄 Processing NVDA data...\n")

        # Process NVDA
        processor = StockDataProcessor()
        result = processor.process_stock('NVDA')

        # Display results
        print("\n" + "=" * 70)
        print("✅ NVDA ANALYSIS COMPLETE")
        print("=" * 70)

        # Basic info
        print(f"\n📊 {result['company_info']['longName']}")
        print(f"   Ticker: {result['ticker']}")
        print(f"   Sector: {result['company_info'].get('sector', 'N/A')}")
        print(f"   Industry: {result['company_info'].get('industry', 'N/A')}")

        # Price info
        stats = result['summary_statistics']
        print(f"\n💰 PRICE INFORMATION")
        print(f"   Current Price:  ${stats['current_price']:.2f}")
        print(f"   Day Change:     ${stats['price_change_1d']:.2f} ({stats['price_change_1d_pct']:.2f}%)")
        print(f"   52-Week Range:  ${stats['52_week_low']:.2f} - ${stats['52_week_high']:.2f}")
        print(f"   All-Time High:  ${stats['all_time_high']:.2f}")
        print(f"   All-Time Low:   ${stats['all_time_low']:.2f}")

        # Performance
        print(f"\n📈 PERFORMANCE RETURNS")
        returns = stats['returns']
        for period, value in returns.items():
            if value is not None:
                period_name = period.replace('_', ' ').title()
                sign = '+' if value > 0 else ''
                print(f"   {period_name:<12} {sign}{value:>7.2f}%")

        # Technical indicators
        print(f"\n🔧 TECHNICAL INDICATORS")
        tech = stats['technical_indicators']
        if tech.get('rsi'):
            print(f"   RSI:            {tech['rsi']:.2f}")
        if tech.get('macd'):
            print(f"   MACD:           {tech['macd']:.4f}")
        if tech.get('sma_20'):
            print(f"   SMA 20:         ${tech['sma_20']:.2f}")
        if tech.get('sma_50'):
            print(f"   SMA 50:         ${tech['sma_50']:.2f}")
        if tech.get('sma_200'):
            print(f"   SMA 200:        ${tech['sma_200']:.2f}")

        # Risk metrics
        print(f"\n⚠️  RISK METRICS")
        print(f"   Annual Volatility:  {stats['volatility_annual']*100:.2f}%")
        print(f"   Sharpe Ratio:       {stats['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown:       {stats['max_drawdown']*100:.2f}%")

        # Trading signals
        signals = stats['signals']
        print(f"\n🎯 TRADING SIGNALS")
        print(f"   Overall:    {signals['overall']}")
        print(f"   Trend:      {signals['trend']}")
        print(f"   Momentum:   {signals['momentum']}")
        print(f"   Volume:     {signals['volume']}")

        if signals['signals']:
            print(f"\n   Key Signals:")
            for signal in signals['signals'][:5]:
                print(f"   • {signal}")

        # Data info
        print(f"\n📦 DATA INFORMATION")
        metadata = result['metadata']
        print(f"   Data Points:    {metadata['data_points']}")
        print(f"   Date Range:     {stats['date_range']['start']} to {stats['date_range']['end']}")
        print(f"   Cache Status:   {metadata['cache_status']}")
        print(f"   Last Updated:   {stats['last_updated']}")

        # Cache files
        print(f"\n💾 CACHE FILES GENERATED")
        cache_dir = config.CACHE_DIR
        nvda_files = list(cache_dir.glob("*NVDA*.json"))
        for file in nvda_files:
            size_kb = file.stat().st_size / 1024
            print(f"   ✓ {file.name:<30} ({size_kb:>7.1f} KB)")

        print("\n" + "=" * 70)
        print("✅ NVDA data successfully generated and cached!")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n✗ Error generating NVDA: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = generate_nvda()
    exit(0 if success else 1)
