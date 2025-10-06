#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

data = json.load(open('stock_analyzer/output/viz_data/NVDA_events.json'))

print('=' * 70)
print('AI-POWERED EVENT ANALYSIS')
print('=' * 70)

event = data['events'][0]
print(f'\nDate: {event["date"]}')
print(f'Type: {event["type"]}')
print(f'Change: {event["price_change_pct"]:+.2f}%')
print(f'Analysis Method: {event["analysis_method"]}')
print(f'Confidence: {event["confidence"]:.1f}%')
print(f'Sentiment Score: {event["sentiment_score"]:.1f}/100')

print(f'\n📝 AI Description:')
print(f'   {event["description"]}')

if 'key_factors' in event and event['key_factors']:
    print(f'\n🔑 Key Factors:')
    for f in event['key_factors']:
        print(f'   • {f}')

print('\n' + '=' * 70)
print('COMPARISON: AI vs Basic Analysis')
print('=' * 70)

print('\n🤖 AI-Powered (Current):')
print(f'   {event["description"][:150]}...')

print('\n🔧 Basic (Previous):')
print(f'   NVDA stock surged 3.93% on September 22, 2025, closing at $183.61...')

print('\n' + '=' * 70)
print(f'\nSummary: {data["event_summary"]["ai_powered_events"]}/{data["event_summary"]["total_events"]} events analyzed with AI')
print('=' * 70)
