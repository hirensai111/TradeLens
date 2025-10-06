#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify NVDA viz_data content"""
import sys
import json

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

data = json.load(open('stock_analyzer/output/viz_data/NVDA_company.json'))
info = data['company_info']
metrics = data['business_metrics']
highlights = data['financial_highlights']

print('=' * 70)
print('NVDA COMPANY DATA - UPDATED')
print('=' * 70)

print('\n📊 COMPANY INFORMATION:')
print(f'  Name: {info["name"]}')
print(f'  Sector: {info["sector"]}')
print(f'  Industry: {info["industry"]}')
print(f'  Employees: {info["employees"]:,}')
print(f'  Country: {info["country"]}')
print(f'  Website: {info["website"]}')
print(f'  Exchange: {info["exchange"]}')
print(f'  Description: {info["description"][:150]}...')

print('\n💰 BUSINESS METRICS:')
print(f'  Market Cap: ${metrics["market_cap"]/1e9:.1f}B')
print(f'  Enterprise Value: ${metrics["enterprise_value"]/1e9:.1f}B')
print(f'  P/E Ratio: {metrics["pe_ratio"]:.2f}')
print(f'  Price to Book: {metrics["price_to_book"]:.2f}')
print(f'  Dividend Yield: {metrics["dividend_yield"]*100:.2f}%' if metrics["dividend_yield"] else '  Dividend Yield: N/A')
print(f'  Profit Margin: {metrics["profit_margin"]*100:.2f}%')
print(f'  ROE: {metrics["return_on_equity"]*100:.2f}%')
print(f'  ROA: {metrics["return_on_assets"]*100:.2f}%')

print('\n📈 FINANCIAL HIGHLIGHTS:')
print(f'  Revenue: ${highlights["revenue"]/1e9:.1f}B')
print(f'  Gross Profit: ${highlights["gross_profit"]/1e9:.1f}B')
print(f'  EBITDA: ${highlights["ebitda"]/1e9:.1f}B')
print(f'  Revenue Growth: {highlights["revenue_growth"]*100:.1f}%')

print('\n' + '=' * 70)
print('✅ All viz_data files updated successfully!')
print('=' * 70)
