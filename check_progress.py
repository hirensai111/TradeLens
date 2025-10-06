#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Monitor progress of stock analysis"""
import sys
from pathlib import Path

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# List of stocks being analyzed
STOCKS = [
    'AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA', 'META', 'AMZN',
    'NFLX', 'AMD', 'INTC', 'PLTR', 'RDDT', 'ABNB', 'ADBE',
    'GOOG', 'NKE', 'F', 'COF', 'MSTR', 'BABA', 'ADDYY', 'IBM'
]

viz_dir = Path('stock_analyzer/output/viz_data')
completed = []

for ticker in STOCKS:
    events_file = viz_dir / f"{ticker}_events.json"
    if events_file.exists():
        completed.append(ticker)

print(f"Progress: {len(completed)}/{len(STOCKS)} stocks completed")
print(f"Completed: {', '.join(completed)}")
print(f"Remaining: {', '.join([s for s in STOCKS if s not in completed])}")
print(f"\nCompletion: {len(completed)/len(STOCKS)*100:.1f}%")
