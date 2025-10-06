#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

data = json.load(open('stock_analyzer/output/viz_data/NVDA_events.json'))
event = data['events'][0]

print('=' * 70)
print('COMPLETE EVENT WITH ENHANCED DESCRIPTION')
print('=' * 70)
print(f'\nDate: {event["date"]}')
print(f'Type: {event["type"]}')
print(f'Price Change: {event["price_change_pct"]:+.2f}%')
print(f'Impact: {event["impact"]}')
print(f'Sentiment: {event["sentiment"]}')
print(f'Confidence: {event["confidence"]:.1f}')
print(f'\nDescription:')
print(f'  {event["description"]}')
print(f'\nPrice Details:')
print(f'  Open: ${event["open_price"]:.2f}')
print(f'  Close: ${event["close_price"]:.2f}')
print(f'  High: ${event["high_price"]:.2f}')
print(f'  Low: ${event["low_price"]:.2f}')
print(f'  Volume: {event["volume"]:,}')
print('=' * 70)
