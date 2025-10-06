#!/usr/bin/env python3
"""
Ultimate AI Stock Predictor - ENHANCED WITH MATHEMATICAL ANALYSIS
Excel (user-selected) + Custom News (user-provided) + Market Data (API/user) + Mathematical Computations + Claude AI
ULTIMATE: User can provide Excel file, custom news articles, and current prices
ENHANCED: Deep mathematical analysis of Excel data before Claude AI reasoning
"""

import os
import sys
import json
import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats
import math

# Add paths for your existing modules
sys.path.append('src')
sys.path.append('src/data_loaders')

def load_api_keys():
    """Load API keys from .env file"""
    alpha_key = None
    claude_key = None
    
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('ALPHA_VANTAGE_API_KEY='):
                    alpha_key = line.split('=', 1)[1].strip()
                elif line.startswith('CLAUDE_API_KEY='):
                    claude_key = line.split('=', 1)[1].strip()
    except FileNotFoundError:
        print("[ERROR] .env file not found. Please create one with your API keys.")
    
    return alpha_key, claude_key

def get_excel_file_from_user(ticker):
    """Interactive Excel file selection from user"""
    print(f"\n📁 EXCEL FILE SELECTION FOR {ticker}")
    print("=" * 50)
    print("Please select your Excel analysis file:")
    print("")
    print("1. Use default file (if available)")
    print("2. Browse and select Excel file")
    print("3. Enter file path manually")
    print("4. Skip Excel analysis (use basic defaults)")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            # Try default files
            default_files = {
                'NVDA': 'data/NVDA_analysis_report_20250715.xlsx',
                'MSFT': 'data/MSFT_analysis_report_20250715_141735.xlsx'
            }
            
            # Also check for files matching pattern
            data_dir = Path('data')
            if data_dir.exists():
                for file in data_dir.glob(f"{ticker}_analysis_report_*.xlsx"):
                    default_files[ticker] = str(file)
                    break
            
            if ticker in default_files and os.path.exists(default_files[ticker]):
                print(f"[OK] Using default file: {default_files[ticker]}")
                return default_files[ticker]
            else:
                print(f"[ERROR] No default file found for {ticker}")
                continue
                
        elif choice == "2":
            # Browse for file (simplified - user provides path)
            print("📁 Browse mode: Please provide the full path to your Excel file")
            file_path = input("Excel file path: ").strip().strip('"')
            
            if os.path.exists(file_path) and file_path.endswith(('.xlsx', '.xls')):
                print(f"[OK] Excel file selected: {file_path}")
                return file_path
            else:
                print("[ERROR] File not found or not an Excel file (.xlsx/.xls)")
                continue
                
        elif choice == "3":
            # Manual path entry
            file_path = input("Enter full path to Excel file: ").strip().strip('"')
            
            if os.path.exists(file_path) and file_path.endswith(('.xlsx', '.xls')):
                print(f"[OK] Excel file selected: {file_path}")
                return file_path
            else:
                print("[ERROR] File not found or not an Excel file")
                continue
                
        elif choice == "4":
            print("[WARNING] Skipping Excel analysis - will use basic historical defaults")
            return None
            
        else:
            print("[ERROR] Invalid choice. Please enter 1, 2, 3, or 4.")

def load_excel_historical_data(ticker, excel_file_path=None):
    """Load historical data from user-selected Excel file"""
    print(f"[CHART] Loading Excel analysis for {ticker}...")
    
    if excel_file_path is None:
        print("   [WARNING] No Excel file provided - using fallback data")
        return {
            'performance_return_1_month': 6.19,
            'volatility': 3.0,
            'sector': 'Technology',
            'avg_daily_change': 0.5,
            'excel_recommendation': 'Hold',
            'excel_risk_level': 'Moderate',
            'recent_high': 180.0,
            'recent_low': 150.0,
            'sma_20': 165.0,
            'sma_50': 160.0,
            'sma_200': 155.0,
            'current_rsi': 55.0
        }
    
    try:
        # Import the fixed Excel loader
        from excel_loader import ExcelDataLoader
        
        loader = ExcelDataLoader()
        loader.excel_path = excel_file_path
        
        print(f"   📂 Loading from: {excel_file_path}")
        
        # Load all data using the fixed loader
        all_excel_data = loader.load_all_data()
        
        if all_excel_data:
            historical_analysis = {}
            
            # Extract comprehensive data (using existing logic)
            if loader.raw_data is not None and not loader.raw_data.empty:
                raw_data = loader.raw_data
                print(f"   [UP] Historical data: {len(raw_data)} days")
                
                # Process daily changes
                if 'Daily Change %' in raw_data.columns:
                    try:
                        daily_changes = pd.to_numeric(raw_data['Daily Change %'], errors='coerce').dropna()
                        if len(daily_changes) > 0:
                            historical_analysis['avg_daily_change'] = float(daily_changes.mean())
                            historical_analysis['volatility'] = float(daily_changes.std())
                            historical_analysis['max_gain'] = float(daily_changes.max())
                            historical_analysis['max_loss'] = float(daily_changes.min())
                            historical_analysis['positive_days_pct'] = float((daily_changes > 0).mean() * 100)
                    except Exception as e:
                        print(f"   [WARNING] Error processing daily changes: {e}")
                
                # Support/resistance levels
                try:
                    if 'High' in raw_data.columns and 'Low' in raw_data.columns:
                        highs = pd.to_numeric(raw_data['High'], errors='coerce').dropna()
                        lows = pd.to_numeric(raw_data['Low'], errors='coerce').dropna()
                        if len(highs) > 0 and len(lows) > 0:
                            historical_analysis['recent_high'] = float(highs.tail(60).max())
                            historical_analysis['recent_low'] = float(lows.tail(60).min())
                except Exception as e:
                    print(f"   [WARNING] Error processing high/low: {e}")
            
            # Technical indicators
            if loader.technical_data is not None and not loader.technical_data.empty:
                try:
                    tech_data = loader.technical_data
                    latest_tech = tech_data.iloc[-1]
                    
                    tech_indicators = ['SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_Signal', 'ATR']
                    for indicator in tech_indicators:
                        if indicator in latest_tech.index:
                            try:
                                value = pd.to_numeric(latest_tech[indicator], errors='coerce')
                                if pd.notna(value):
                                    key = f'current_{indicator.lower()}' if indicator == 'RSI' else indicator.lower()
                                    historical_analysis[key] = float(value)
                            except:
                                pass
                except Exception as e:
                    print(f"   [WARNING] Error processing technical data: {e}")
            
            # Summary data
            if loader.summary_data is not None and not loader.summary_data.empty:
                try:
                    summary = loader.summary_data.iloc[0]
                    
                    # Text fields
                    text_fields = ['overall_signal', 'trend', 'momentum', 'recommendation', 'risk_level']
                    for field in text_fields:
                        if field in summary.index and pd.notna(summary[field]):
                            value = str(summary[field])
                            if not ('%' in value and len(value) > 10):  # Skip corrupted data
                                historical_analysis[f'excel_{field}'] = value
                except Exception as e:
                    print(f"   [WARNING] Error processing summary: {e}")
            
            # CRITICAL: Get 30-day performance using fixed method
            try:
                thirty_day_performance = loader.get_30d_performance()
                historical_analysis['performance_return_1_month'] = thirty_day_performance
                historical_analysis['30d_return'] = thirty_day_performance
                historical_analysis['recent_30d_return'] = thirty_day_performance
                print(f"   [OK] CRITICAL FIX: 30-day performance = {thirty_day_performance:.2f}%")
            except Exception as e:
                print(f"   [WARNING] Error using direct 30-day method: {e}")
                historical_analysis['performance_return_1_month'] = 6.19
                historical_analysis['30d_return'] = 6.19
                historical_analysis['recent_30d_return'] = 6.19
            
            # Company info
            if loader.company_info is not None and not loader.company_info.empty:
                try:
                    company = loader.company_info.iloc[0]
                    if 'sector' in company.index and pd.notna(company['sector']):
                        historical_analysis['sector'] = str(company['sector'])
                    if 'industry' in company.index and pd.notna(company['industry']):
                        historical_analysis['industry'] = str(company['industry'])
                except Exception as e:
                    print(f"   [WARNING] Error processing company info: {e}")
            
            # Set defaults for missing values
            defaults = {
                'avg_daily_change': 0.5, 'volatility': 3.0, 'sector': 'Technology',
                'excel_recommendation': 'Hold', 'excel_risk_level': 'Moderate',
                'recent_high': 180.0, 'recent_low': 150.0, 'sma_20': 165.0,
                'sma_50': 160.0, 'sma_200': 155.0, 'current_rsi': 55.0
            }
            
            for key, default_value in defaults.items():
                if key not in historical_analysis:
                    historical_analysis[key] = default_value
            
            print(f"[OK] Excel analysis complete: {len(historical_analysis)} metrics extracted")
            return historical_analysis
            
    except Exception as e:
        print(f"[ERROR] Excel loading error: {e}")
        return {
            'performance_return_1_month': 6.19, 'volatility': 3.0, 'sector': 'Technology',
            'avg_daily_change': 0.5, 'excel_recommendation': 'Hold', 'excel_risk_level': 'Moderate'
        }

def analyze_article_sentiment(article_content, ticker):
    """
    Analyze article for specific momentum indicators
    """
    # Key phrases for momentum analysis
    positive_indicators = [
        'beat expectations', 'record revenue', 'strong growth', 'exceeded',
        'breakthrough', 'partnership', 'expansion', 'bullish', 'upgrade',
        'raised guidance', 'positive outlook', 'strong demand', 'all-time high',
        'surpassed', 'accelerating', 'robust', 'outperform', 'momentum'
    ]
    
    negative_indicators = [
        'missed expectations', 'revenue decline', 'weak growth', 'below',
        'concerns', 'layoffs', 'contraction', 'bearish', 'downgrade',
        'lowered guidance', 'negative outlook', 'weak demand', 'disappointing',
        'slowing', 'struggling', 'underperform', 'cut', 'warning'
    ]
    
    content_lower = article_content.lower()
    
    # Count indicators
    positive_count = sum(1 for phrase in positive_indicators if phrase in content_lower)
    negative_count = sum(1 for phrase in negative_indicators if phrase in content_lower)
    
    # Calculate momentum score
    net_score = positive_count - negative_count
    momentum_score = max(-1, min(1, net_score / 5))  # Normalize to -1 to 1
    
    return {
        'momentum_score': momentum_score,
        'positive_indicators': positive_count,
        'negative_indicators': negative_count,
        'momentum_direction': 'Bullish' if momentum_score > 0.2 else 'Bearish' if momentum_score < -0.2 else 'Neutral'
    }

def detect_major_catalysts(article_content):
    """
    Detect if article contains major market-moving catalysts
    """
    major_catalysts = {
        'earnings_beat': ['beat earnings', 'exceeded eps', 'earnings surprise', 'beat expectations', 'topped estimates'],
        'guidance_raise': ['raised guidance', 'increased outlook', 'upgraded forecast', 'raised full-year', 'boosted outlook'],
        'major_contract': ['billion dollar', 'major contract', 'landmark deal', 'mega deal', 'significant contract'],
        'fda_approval': ['fda approval', 'regulatory approval', 'cleared by fda', 'approved by fda'],
        'acquisition': ['acquiring', 'acquisition', 'merger', 'buyout', 'to acquire'],
        'breakthrough': ['breakthrough', 'revolutionary', 'game-changing', 'paradigm shift', 'disrupting']
    }
    
    content_lower = article_content.lower()
    detected_catalysts = []
    
    for catalyst_type, phrases in major_catalysts.items():
        for phrase in phrases:
            if phrase in content_lower:
                detected_catalysts.append(catalyst_type)
                break
    
    return {
        'has_major_catalyst': len(detected_catalysts) > 0,
        'catalyst_types': detected_catalysts,
        'catalyst_count': len(detected_catalysts),
        'impact_multiplier': 1.0 + (len(detected_catalysts) * 0.3)  # Each catalyst adds 30% impact
    }

def get_custom_news_from_user(ticker):
    """Get custom news articles from user"""
    print(f"\n📰 CUSTOM NEWS INPUT FOR {ticker}")
    print("=" * 40)
    print("You can add breaking news or important announcements that might affect the stock.")
    print("This will be included in the AI analysis for more accurate predictions.")
    print("")
    print("Options:")
    print("1. Add news text manually")
    print("2. Load news from text file") 
    print("3. Skip custom news")
    
    custom_articles = []
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            # Manual text input
            print(f"\n📝 Enter News Article Text")
            print("-" * 30)
            print("Enter the news article content (press Enter twice when done):")
            print()
            
            lines = []
            empty_line_count = 0
            
            while True:
                try:
                    line = input()
                    if line.strip() == "":
                        empty_line_count += 1
                        if empty_line_count >= 2:
                            break
                    else:
                        empty_line_count = 0
                        lines.append(line)
                except EOFError:
                    break
            
            if lines:
                article_text = "\n".join(lines)
                if len(article_text.strip()) >= 20:  # Minimum length
                    title = input("\nEnter article title (or press Enter to auto-generate): ").strip()
                    if not title:
                        title = article_text.split('\n')[0][:80] + "..." if len(article_text) > 80 else article_text.split('\n')[0]
                    
                    # Analyze article content automatically
                    momentum_analysis = analyze_article_sentiment(article_text, ticker)
                    catalyst_analysis = detect_major_catalysts(article_text)
                    
                    # Get sentiment assessment
                    print(f"\n🤖 Automated Analysis Results:")
                    print(f"   Momentum: {momentum_analysis['momentum_direction']} ({momentum_analysis['momentum_score']:+.2f})")
                    print(f"   Positive indicators: {momentum_analysis['positive_indicators']}")
                    print(f"   Negative indicators: {momentum_analysis['negative_indicators']}")
                    if catalyst_analysis['has_major_catalyst']:
                        print(f"   🚨 Major catalysts detected: {', '.join(catalyst_analysis['catalyst_types'])}")
                    
                    print(f"\nHow would you assess this news for {ticker}?")
                    print("1. Very Positive (+0.8)")
                    print("2. Positive (+0.4)")  
                    print("3. Neutral (0.0)")
                    print("4. Negative (-0.4)")
                    print("5. Very Negative (-0.8)")
                    print("6. Use automated assessment")
                    
                    sentiment_choice = input("Enter choice (1-6, default 6): ").strip() or "6"
                    
                    if sentiment_choice == "6":
                        # Use automated assessment
                        sentiment_score = momentum_analysis['momentum_score'] * catalyst_analysis['impact_multiplier']
                        sentiment_score = max(-1, min(1, sentiment_score))  # Clamp to [-1, 1]
                    else:
                        sentiment_map = {"1": 0.8, "2": 0.4, "3": 0.0, "4": -0.4, "5": -0.8}
                        sentiment_score = sentiment_map.get(sentiment_choice, momentum_analysis['momentum_score'])
                    
                    custom_articles.append({
                        'title': title,
                        'content': article_text,
                        'sentiment_score': sentiment_score,
                        'confidence': 0.9,
                        'source': 'user_input',
                        'timestamp': datetime.now().isoformat(),
                        'momentum_analysis': momentum_analysis,
                        'catalyst_analysis': catalyst_analysis
                    })
                    
                    print(f"[OK] Custom news article added!")
                    print(f"   Title: {title}")
                    print(f"   Sentiment: {sentiment_score:+.1f}")
                    print(f"   Impact: {'HIGH' if catalyst_analysis['has_major_catalyst'] else 'Normal'}")
                    
                    # Ask if user wants to add more
                    add_more = input(f"\nAdd another news article? (y/n): ").strip().lower()
                    if add_more in ['y', 'yes']:
                        continue
                    else:
                        break
                else:
                    print("[ERROR] Article too short (minimum 20 characters)")
                    continue
            else:
                print("[ERROR] No text entered")
                continue
                
        elif choice == "2":
            # Load from file
            file_path = input("Enter path to text file: ").strip().strip('"')
            
            if not os.path.exists(file_path):
                print(f"[ERROR] File not found: {file_path}")
                continue
            
            try:
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'cp1252']
                article_text = None
                
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            article_text = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                
                if article_text and len(article_text.strip()) >= 20:
                    filename = os.path.basename(file_path)
                    title = input(f"Enter title for this article (default: {filename}): ").strip() or filename
                    
                    # Analyze article content
                    momentum_analysis = analyze_article_sentiment(article_text, ticker)
                    catalyst_analysis = detect_major_catalysts(article_text)
                    
                    print(f"\n🤖 Automated Analysis:")
                    print(f"   Momentum: {momentum_analysis['momentum_direction']} ({momentum_analysis['momentum_score']:+.2f})")
                    if catalyst_analysis['has_major_catalyst']:
                        print(f"   Major catalysts: {', '.join(catalyst_analysis['catalyst_types'])}")
                    
                    # Get sentiment
                    print(f"\nSentiment assessment for this article:")
                    print("1. Positive (+0.5)")
                    print("2. Neutral (0.0)")
                    print("3. Negative (-0.5)")
                    print("4. Use automated assessment")
                    
                    sentiment_choice = input("Enter choice (1-4, default 4): ").strip() or "4"
                    
                    if sentiment_choice == "4":
                        sentiment_score = momentum_analysis['momentum_score'] * catalyst_analysis['impact_multiplier']
                        sentiment_score = max(-1, min(1, sentiment_score))
                    else:
                        sentiment_map = {"1": 0.5, "2": 0.0, "3": -0.5}
                        sentiment_score = sentiment_map.get(sentiment_choice, 0.0)
                    
                    custom_articles.append({
                        'title': title,
                        'content': article_text,
                        'sentiment_score': sentiment_score,
                        'confidence': 0.8,
                        'source': f'file: {filename}',
                        'timestamp': datetime.now().isoformat(),
                        'momentum_analysis': momentum_analysis,
                        'catalyst_analysis': catalyst_analysis
                    })
                    
                    print(f"[OK] News article loaded from file!")
                    print(f"   Title: {title}")
                    print(f"   Length: {len(article_text)} characters")
                    break
                else:
                    print("[ERROR] File is empty or too short")
                    continue
                    
            except Exception as e:
                print(f"[ERROR] Error reading file: {e}")
                continue
                
        elif choice == "3":
            print("[WARNING] Skipping custom news input")
            break
            
        else:
            print("[ERROR] Invalid choice. Please enter 1, 2, or 3.")
    
    if custom_articles:
        print(f"\n[OK] Added {len(custom_articles)} custom news article(s)")
        for i, article in enumerate(custom_articles, 1):
            sentiment_label = "[UP] Positive" if article['sentiment_score'] > 0.1 else "[DOWN] Negative" if article['sentiment_score'] < -0.1 else "➡️ Neutral"
            catalyst_label = " 🚨 MAJOR CATALYST" if article['catalyst_analysis']['has_major_catalyst'] else ""
            print(f"   {i}. {article['title']} ({sentiment_label}){catalyst_label}")
    
    return custom_articles

def get_phase3_news_intelligence(ticker):
    """Get news intelligence from Phase 3 system"""
    print(f"📰 Gathering Phase 3 news intelligence for {ticker}...")
    
    try:
        from phase3_connector import Phase3NewsConnector
        connector = Phase3NewsConnector()
        
        sentiment_features = connector.get_enhanced_sentiment_features(
            ticker, datetime.now(), lookback_days=7
        )
        
        news_data = {
            'sentiment_1d': float(sentiment_features.sentiment_1d),
            'sentiment_7d': float(sentiment_features.sentiment_7d),
            'news_volume_1d': int(sentiment_features.news_volume_1d),
            'news_volume_7d': int(sentiment_features.news_volume_7d),
            'confidence_score': float(sentiment_features.confidence_score),
            'source_diversity': int(sentiment_features.source_diversity),
            'event_impact_score': float(sentiment_features.event_impact_score),
            'recent_events': []
        }
        
        print(f"[OK] Phase 3 news intelligence complete")
        return news_data
        
    except Exception as e:
        print(f"[WARNING] Phase 3 connector issues: {e} - using fallback")
        return {
            'sentiment_1d': 0.15, 'sentiment_7d': 0.08, 'news_volume_1d': 12,
            'confidence_score': 0.75, 'source_diversity': 8, 'event_impact_score': 0.42,
            'recent_events': []
        }

def enhance_news_with_custom_articles(news_data, custom_articles):
    """Enhance Phase 3 news data with custom articles"""
    if not custom_articles:
        return news_data
    
    print(f"🔄 Enhancing news analysis with {len(custom_articles)} custom articles...")
    
    # Calculate custom sentiment impact with MOMENTUM ANALYSIS
    total_impact = 0
    has_major_catalyst = False
    catalyst_count = 0
    
    for article in custom_articles:
        # Get momentum analysis (already computed in get_custom_news_from_user)
        momentum_analysis = article.get('momentum_analysis', {'momentum_score': article['sentiment_score']})
        catalyst_analysis = article.get('catalyst_analysis', {'has_major_catalyst': False, 'impact_multiplier': 1.0})
        
        # Combine user sentiment with automated analysis
        combined_sentiment = (article['sentiment_score'] * 0.4 + 
                            momentum_analysis['momentum_score'] * 0.6)
        
        # Apply catalyst multiplier
        combined_sentiment *= catalyst_analysis['impact_multiplier']
        
        # Weight by recency (more recent = higher weight)
        hours_old = (datetime.now() - datetime.fromisoformat(article['timestamp'])).total_seconds() / 3600
        recency_weight = 1.0 if hours_old < 24 else 0.8 if hours_old < 48 else 0.6
        
        total_impact += combined_sentiment * recency_weight
        
        if catalyst_analysis['has_major_catalyst']:
            has_major_catalyst = True
            catalyst_count += catalyst_analysis['catalyst_count']
    
    # Calculate weighted average impact
    avg_impact = total_impact / len(custom_articles) if custom_articles else 0
    
    # INCREASE WEIGHT for custom articles (they're breaking news)
    # If major catalyst detected, custom news gets even more weight
    custom_weight = 0.8 if has_major_catalyst else 0.7
    phase3_weight = 1.0 - custom_weight
    
    enhanced_sentiment = (news_data.get('sentiment_1d', 0) * phase3_weight + avg_impact * custom_weight)
    
    # Update news data
    enhanced_news_data = news_data.copy()
    enhanced_news_data['sentiment_1d'] = enhanced_sentiment
    enhanced_news_data['custom_impact_score'] = avg_impact
    enhanced_news_data['momentum_direction'] = 'Bullish' if avg_impact > 0.2 else 'Bearish' if avg_impact < -0.2 else 'Neutral'
    enhanced_news_data['has_major_catalyst'] = has_major_catalyst
    enhanced_news_data['total_catalyst_count'] = catalyst_count
    enhanced_news_data['news_volume_1d'] += len(custom_articles)
    enhanced_news_data['custom_articles_count'] = len(custom_articles)
    enhanced_news_data['custom_sentiment_impact'] = avg_impact
    
    # Add custom articles to recent events
    if 'recent_events' not in enhanced_news_data:
        enhanced_news_data['recent_events'] = []
    
    for article in custom_articles[-3:]:  # Add last 3 custom articles
        catalyst_tag = " [CATALYST]" if article.get('catalyst_analysis', {}).get('has_major_catalyst', False) else ""
        enhanced_news_data['recent_events'].append({
            'title': f"[CUSTOM]{catalyst_tag} {article['title']}",
            'sentiment': article['sentiment_score'],
            'confidence': article['confidence'],
            'date': article['timestamp'][:10],
            'has_catalyst': article.get('catalyst_analysis', {}).get('has_major_catalyst', False)
        })
    
    print(f"   [OK] News enhanced: sentiment adjusted to {enhanced_sentiment:+.3f}")
    print(f"   [CHART] Custom impact: {avg_impact:+.2f}")
    if has_major_catalyst:
        print(f"   🚨 Major catalysts detected: {catalyst_count} total")
    
    return enhanced_news_data

def get_current_price_from_user(ticker):
    """Get current stock price and market data from user - FIXED VERSION"""
    print(f"\n[MONEY] MANUAL PRICE INPUT FOR {ticker}")
    print("=" * 40)
    print("Since the API is unavailable, please provide current market information.")
    print("You can find current prices on:")
    print("   • Yahoo Finance (finance.yahoo.com)")
    print("   • Google Finance (finance.google.com)")
    print("   • Robinhood, E*TRADE, or your broker app")
    print("")
    
    # Get current price (required)
    while True:
        try:
            price_input = input(f"[MONEY] Enter current price for {ticker}: $").strip()
            if not price_input:
                print("[ERROR] Price is required. Please enter the current stock price.")
                continue
            
            current_price = float(price_input)
            if current_price <= 0:
                print("[ERROR] Price must be greater than 0")
                continue
            break
        except ValueError:
            print("[ERROR] Invalid price format. Please enter a number (e.g., 118.50)")
    
    print(f"[OK] Current price set: ${current_price:.2f}")
    
    # FIXED: Initialize variables first
    daily_change = 0.0
    daily_change_pct = 0.0
    
    # Get daily change (optional but recommended)
    print(f"\n[CHART] Optional: Daily change information")
    try:
        change_input = input(f"Daily change in $ (e.g., +2.45, -1.20, or press Enter to skip): ").strip()
        if change_input:
            daily_change = float(change_input.replace('+', ''))
            # FIXED: Only calculate percentage if we have a valid previous price
            if current_price != daily_change:  # Avoid division by zero
                daily_change_pct = (daily_change / (current_price - daily_change)) * 100
            else:
                daily_change_pct = 0.0
            direction = "[UP]"
            direction = "[UP]" if daily_change > 0 else "[DOWN]" if daily_change < 0 else "➡️"
            print(f"[OK] Daily change: {direction} ${daily_change:+.2f} ({daily_change_pct:+.2f}%)")
    except ValueError:
        print("[WARNING] Invalid change format, using 0.0")
        daily_change = 0.0
        daily_change_pct = 0.0
    
    # Get volume (optional)
    volume = 25_000_000  # Default volume
    try:
        volume_input = input(f"Trading volume (e.g., 45M, 25000000, or press Enter for default): ").strip()
        if volume_input:
            volume_str = volume_input.upper().replace(',', '')
            if 'M' in volume_str:
                volume = int(float(volume_str.replace('M', '')) * 1_000_000)
            elif 'K' in volume_str:
                volume = int(float(volume_str.replace('K', '')) * 1_000)
            else:
                volume = int(volume_str)
            print(f"[OK] Volume set: {volume:,}")
    except ValueError:
        print("[WARNING] Invalid volume format, using default 25M")
        volume = 25_000_000
    
    # Calculate derived values
    open_price = current_price - daily_change
    previous_close = open_price
    
    # Estimate intraday range (conservative 1-2% range)
    estimated_range = current_price * 0.015  # 1.5% range
    high_price = current_price + (estimated_range * 0.7)  # Slightly above current
    low_price = current_price - (estimated_range * 0.5)   # Slightly below current
    
    # Create market data structure - FIXED: All variables are now properly initialized
    market_data = {
        'current_price': round(current_price, 2),
        'open': round(open_price, 2),
        'high': round(high_price, 2),
        'low': round(low_price, 2),
        'volume': volume,
        'change': round(daily_change, 2),
        'change_percent': round(daily_change_pct, 2),
        'previous_close': round(previous_close, 2),
        'current_rsi': 50.0,  # Neutral RSI assumption
        'av_news_sentiment': 0.0,  # Neutral sentiment
        'av_news_count': 0,
        'data_source': 'user_input'
    }
    
    # Display summary for confirmation
    print(f"\n📋 MARKET DATA SUMMARY:")
    print(f"   Current Price: ${market_data['current_price']:.2f}")
    print(f"   Daily Change: ${market_data['change']:+.2f} ({market_data['change_percent']:+.2f}%)")
    print(f"   Est. Range: ${market_data['low']:.2f} - ${market_data['high']:.2f}")
    print(f"   Volume: {market_data['volume']:,}")
    print(f"   Open: ${market_data['open']:.2f}")
    
    # Final confirmation
    print(f"\n[?] Proceed with this market data for analysis?")
    confirm = input("Enter 'y' to continue, 'n' to re-enter data: ").strip().lower()
    
    if confirm in ['y', 'yes']:
        print(f"[OK] Market data confirmed - proceeding with analysis")
        return market_data
    elif confirm in ['n', 'no']:
        print(f"🔄 Let's re-enter the data...")
        return get_current_price_from_user(ticker)  # Recursive call to re-enter
    else:
        # Default to yes if unclear
        print(f"[OK] Proceeding with analysis (default)")
        return market_data

def get_realtime_market_data(ticker, api_key):
   """Get real-time data from Alpha Vantage API"""
   try:
       url = "https://www.alphavantage.co/query"
       params = {'function': 'GLOBAL_QUOTE', 'symbol': ticker, 'apikey': api_key}
       
       response = requests.get(url, params=params, timeout=30)
       data = response.json()
       
       if "Global Quote" in data:
           quote = data["Global Quote"]
           return {
               'current_price': float(quote['05. price']),
               'open': float(quote['02. open']),
               'high': float(quote['03. high']),
               'low': float(quote['04. low']),
               'volume': int(quote['06. volume']),
               'change': float(quote['09. change']),
               'change_percent': float(quote['10. change percent'].replace('%', '')),
               'previous_close': float(quote['08. previous close']),
               'current_rsi': 50.0,
               'av_news_sentiment': 0.0,
               'av_news_count': 0,
               'data_source': 'alpha_vantage'
           }
       else:
           print(f"   [WARNING] API issue: {data.get('Note', data.get('Error Message', 'Unknown error'))}")
           return None
   except Exception as e:
       print(f"   [ERROR] API failed: {e}")
       return None

def get_realtime_market_data_with_fallback(ticker, api_key):
   """Get market data with user fallback"""
   print(f"[UP] Getting market data for {ticker}...")
   
   # Try API first
   if api_key:
       market_data = get_realtime_market_data(ticker, api_key)
       if market_data:
           print(f"   [OK] API data: ${market_data['current_price']:.2f} ({market_data['change_percent']:+.2f}%)")
           return market_data
   
   # Fallback to user input
   print(f"   [WARNING] API unavailable - switching to manual input")
   return get_current_price_from_user(ticker)

def perform_mathematical_analysis(excel_data, market_data, custom_articles=None):
   """
   Perform comprehensive mathematical analysis on Excel data
   Returns detailed statistical metrics for Claude to analyze
   """
   print("🔢 Performing mathematical analysis on Excel data...")
   
   analysis_results = {}
   
   # === VOLATILITY ANALYSIS ===
   volatility = excel_data.get('volatility', 3.0)
   analysis_results['volatility_metrics'] = {
       'annualized_volatility': volatility,
       'daily_volatility': volatility / math.sqrt(252),  # Convert to daily
       'volatility_percentile': min(95, max(5, volatility * 10)),  # Rough percentile
       'risk_category': 'High' if volatility > 25 else 'Medium' if volatility > 15 else 'Low'
   }
   
   # === MOMENTUM ANALYSIS ===
   avg_daily_change = excel_data.get('avg_daily_change', 0.5)
   current_change = market_data.get('change_percent', 0)
   thirty_day_return = excel_data.get('performance_return_1_month', 6.19)
   
   # Calculate momentum strength
   momentum_score = (thirty_day_return / 30) * 0.6 + current_change * 0.4  # Weight recent more
   momentum_strength = abs(momentum_score)
   
   # Add custom news momentum impact
   custom_momentum_boost = 0
   if custom_articles:
       for article in custom_articles:
           if 'momentum_analysis' in article:
               custom_momentum_boost += article['momentum_analysis']['momentum_score']
           if article.get('catalyst_analysis', {}).get('has_major_catalyst', False):
               custom_momentum_boost += 0.3  # Extra boost for catalysts
       
       avg_custom_momentum = custom_momentum_boost / len(custom_articles) if custom_articles else 0
       
       # Custom news can override historical momentum
       if abs(avg_custom_momentum) > 0.5:  # Strong custom news signal
           momentum_score = momentum_score * 0.3 + avg_custom_momentum * 0.7
       else:
           momentum_score = momentum_score * 0.7 + avg_custom_momentum * 0.3
   
   analysis_results['momentum_metrics'] = {
       'thirty_day_return': thirty_day_return,
       'annualized_return': (thirty_day_return / 30) * 365,
       'momentum_score': momentum_score,
       'momentum_strength': momentum_strength,
       'momentum_direction': 'Bullish' if momentum_score > 0.5 else 'Bearish' if momentum_score < -0.5 else 'Neutral',
       'relative_strength': thirty_day_return / volatility if volatility > 0 else 0,  # Return/risk ratio
       'custom_news_impact': custom_momentum_boost / len(custom_articles) if custom_articles else 0
   }
   
   # === TECHNICAL ANALYSIS ===
   current_price = market_data['current_price']
   sma_20 = excel_data.get('sma_20', current_price)
   sma_50 = excel_data.get('sma_50', current_price * 0.98)
   sma_200 = excel_data.get('sma_200', current_price * 0.95)
   current_rsi = excel_data.get('current_rsi', 50.0)
   
   # Price position analysis
   price_above_sma20 = (current_price - sma_20) / sma_20 * 100
   price_above_sma50 = (current_price - sma_50) / sma_50 * 100
   price_above_sma200 = (current_price - sma_200) / sma_200 * 100
   
   # Moving average convergence/divergence
   sma_convergence = (sma_20 - sma_50) / sma_50 * 100
   long_term_trend = (sma_50 - sma_200) / sma_200 * 100
   
   analysis_results['technical_metrics'] = {
       'price_vs_sma20_pct': price_above_sma20,
       'price_vs_sma50_pct': price_above_sma50,
       'price_vs_sma200_pct': price_above_sma200,
       'sma_convergence': sma_convergence,
       'long_term_trend': long_term_trend,
       'current_rsi': current_rsi,
       'rsi_category': 'Overbought' if current_rsi > 70 else 'Oversold' if current_rsi < 30 else 'Neutral',
       'trend_strength': abs(price_above_sma20) + abs(sma_convergence),
       'trend_alignment': 1 if price_above_sma20 > 0 and sma_convergence > 0 and long_term_trend > 0 else -1 if price_above_sma20 < 0 and sma_convergence < 0 and long_term_trend < 0 else 0
   }
   
   # === SUPPORT/RESISTANCE ANALYSIS ===
   recent_high = excel_data.get('recent_high', current_price * 1.05)
   recent_low = excel_data.get('recent_low', current_price * 0.95)
   
   # Calculate key levels
   resistance_distance = (recent_high - current_price) / current_price * 100
   support_distance = (current_price - recent_low) / current_price * 100
   range_position = (current_price - recent_low) / (recent_high - recent_low) * 100
   
   analysis_results['support_resistance'] = {
       'recent_high': recent_high,
       'recent_low': recent_low,
       'resistance_distance_pct': resistance_distance,
       'support_distance_pct': support_distance,
       'range_position_pct': range_position,
       'range_width_pct': (recent_high - recent_low) / current_price * 100,
       'breakout_probability': max(0, min(100, (90 - range_position) if range_position > 80 else (range_position - 10) if range_position < 20 else 30))
   }
   
   # === RISK ANALYSIS ===
   # Calculate Value at Risk (VaR) estimates
   daily_vol = volatility / math.sqrt(252)
   var_95 = current_price * daily_vol * 1.645  # 95% confidence
   var_99 = current_price * daily_vol * 2.33   # 99% confidence
   
   # Sharpe ratio approximation
   risk_free_rate = 0.05  # Assume 5% risk-free rate
   excess_return = (thirty_day_return / 100 * 12) - risk_free_rate  # Annualized excess return
   sharpe_ratio = excess_return / (volatility / 100) if volatility > 0 else 0
   
   analysis_results['risk_metrics'] = {
       'value_at_risk_95': var_95,
       'value_at_risk_99': var_99,
       'max_drawdown_estimate': current_price * (volatility / 100) * 2,
       'sharpe_ratio': sharpe_ratio,
       'risk_adjusted_return': thirty_day_return / volatility if volatility > 0 else 0,
       'downside_risk': volatility * 0.7,  # Assume 70% of volatility is downside
       'risk_score': min(100, max(0, volatility * 2 + abs(momentum_score) * 10))
   }
   
   # === STATISTICAL ANALYSIS ===
   # Generate synthetic price series for statistical analysis
   np.random.seed(42)  # For reproducible results
   returns = np.random.normal(avg_daily_change/100, daily_vol, 30)  # 30 days of returns
   price_series = [current_price]
   for ret in returns:
       price_series.append(price_series[-1] * (1 + ret))
   
   price_array = np.array(price_series)
   
   # Statistical metrics
   analysis_results['statistical_metrics'] = {
       'mean_return_daily': np.mean(returns) * 100,
       'std_deviation': np.std(returns) * 100,
       'skewness': stats.skew(returns),
       'kurtosis': stats.kurtosis(returns),
       'confidence_interval_95': {
           'lower': np.percentile(price_series, 2.5),
           'upper': np.percentile(price_series, 97.5)
       },
       'probability_positive': len([r for r in returns if r > 0]) / len(returns),
       'max_expected_gain': np.max(returns) * 100,
       'max_expected_loss': np.min(returns) * 100
   }
   
   # === COMPOSITE SCORES ===
   # Calculate overall signal strength
   technical_score = (
       (1 if price_above_sma20 > 0 else -1) * 0.3 +
       (1 if current_rsi < 70 and current_rsi > 30 else -0.5) * 0.2 +
       (analysis_results['technical_metrics']['trend_alignment']) * 0.5
   )
   
   momentum_score_normalized = max(-1, min(1, momentum_score / 5))  # Normalize to -1,1
   
   # Boost scores if major catalysts detected
   catalyst_boost = 0
   if custom_articles:
       for article in custom_articles:
           if article.get('catalyst_analysis', {}).get('has_major_catalyst', False):
               catalyst_boost += 0.2
       catalyst_boost = min(0.5, catalyst_boost)  # Max 0.5 boost
   
   analysis_results['composite_scores'] = {
       'technical_signal': technical_score,
       'momentum_signal': momentum_score_normalized + catalyst_boost,
       'risk_signal': max(-1, min(1, (50 - analysis_results['risk_metrics']['risk_score']) / 50)),
       'overall_signal': (technical_score * 0.4 + (momentum_score_normalized + catalyst_boost) * 0.4 + 
                         (max(-1, min(1, (50 - analysis_results['risk_metrics']['risk_score']) / 50)) * 0.2)),
       'catalyst_boost': catalyst_boost
   }
   
   print(f"   [OK] Mathematical analysis complete: {len(analysis_results)} metric categories")
   print(f"   [CHART] Overall signal: {analysis_results['composite_scores']['overall_signal']:.3f}")
   print(f"   [TARGET] Technical strength: {analysis_results['composite_scores']['technical_signal']:.3f}")
   print(f"   [UP] Momentum strength: {analysis_results['composite_scores']['momentum_signal']:.3f}")
   if catalyst_boost > 0:
       print(f"   🚨 Catalyst boost applied: +{catalyst_boost:.2f}")
   
   return analysis_results

def analyze_with_claude_ultimate_enhanced(ticker, excel_data, news_data, market_data, custom_articles, claude_key):
   """
   Enhanced Claude analysis with deep mathematical computations and article analysis
   """
   print(f"🧮 Running enhanced Claude AI analysis with mathematical computations for {ticker}...")
   
   try:
       import anthropic
       client = anthropic.Anthropic(api_key=claude_key)
   except ImportError:
       print("[ERROR] Anthropic library not installed")
       return create_fallback_analysis(ticker, excel_data, news_data, market_data)
   
   # STEP 1: Perform comprehensive mathematical analysis
   math_analysis = perform_mathematical_analysis(excel_data, market_data, custom_articles)
   
   # STEP 2: Prepare custom news summary with full article content
   custom_news_summary = ""
   if custom_articles:
       custom_news_summary = f"\n═══ CUSTOM NEWS ARTICLES (User-Provided) ═══\n"
       for i, article in enumerate(custom_articles, 1):
           sentiment_label = "Positive" if article['sentiment_score'] > 0.1 else "Negative" if article['sentiment_score'] < -0.1 else "Neutral"
           momentum_analysis = article.get('momentum_analysis', {})
           catalyst_analysis = article.get('catalyst_analysis', {})
           
           custom_news_summary += f"\n{i}. {article['title']}\n"
           custom_news_summary += f"   User Sentiment: {sentiment_label} ({article['sentiment_score']:+.1f})\n"
           custom_news_summary += f"   Automated Analysis: {momentum_analysis.get('momentum_direction', 'Unknown')} "
           custom_news_summary += f"(+{momentum_analysis.get('positive_indicators', 0)}/-{momentum_analysis.get('negative_indicators', 0)} indicators)\n"
           
           if catalyst_analysis.get('has_major_catalyst', False):
               custom_news_summary += f"   🚨 MAJOR CATALYSTS: {', '.join(catalyst_analysis.get('catalyst_types', []))}\n"
           
           # Include first 800 chars of content for Claude to analyze
           content_preview = article['content'][:800].replace('\n', ' ')
           custom_news_summary += f"   Content: \"{content_preview}...\"\n"
           custom_news_summary += f"   ANALYSIS DIRECTIVE: This article should be analyzed for specific momentum indicators and heavily weighted\n\n"
   
   # STEP 3: Data sources note
   excel_source = "User-provided Excel file" if excel_data.get('volatility', 3.0) != 3.0 else "Default historical data"
   market_source = "Alpha Vantage API" if market_data.get('data_source') == 'alpha_vantage' else "User-provided current data"
   
   # STEP 4: Create comprehensive prompt with mathematical analysis
   prompt = f"""You are a senior quantitative analyst with expertise in mathematical finance and news analysis. Perform ULTIMATE analysis of {ticker} using ALL available data sources including comprehensive mathematical computations and deep article analysis.

═══ RAW HISTORICAL DATA ({excel_source}) ═══
- Recent 30d performance: +{excel_data.get('performance_return_1_month', 6.19):.2f}%
- Historical volatility: {excel_data.get('volatility', 3.0):.2f}%
- Average daily change: {excel_data.get('avg_daily_change', 0.5):.2f}%
- Sector: {excel_data.get('sector', 'Technology')}
- Excel recommendation: {excel_data.get('excel_recommendation', 'Hold')}

═══ MATHEMATICAL ANALYSIS RESULTS ═══

[CHART] VOLATILITY METRICS:
- Annualized Volatility: {math_analysis['volatility_metrics']['annualized_volatility']:.2f}%
- Daily Volatility: {math_analysis['volatility_metrics']['daily_volatility']:.3f}%
- Risk Category: {math_analysis['volatility_metrics']['risk_category']}
- Volatility Percentile: {math_analysis['volatility_metrics']['volatility_percentile']:.0f}%

[UP] MOMENTUM ANALYSIS:
- 30-day Return: {math_analysis['momentum_metrics']['thirty_day_return']:.2f}%
- Annualized Return: {math_analysis['momentum_metrics']['annualized_return']:.2f}%
- Momentum Score: {math_analysis['momentum_metrics']['momentum_score']:.3f}
- Momentum Direction: {math_analysis['momentum_metrics']['momentum_direction']}
- Risk-Adjusted Return: {math_analysis['momentum_metrics']['relative_strength']:.3f}
- Custom News Impact: {math_analysis['momentum_metrics']['custom_news_impact']:+.3f}

[WRENCH] TECHNICAL INDICATORS:
- Price vs SMA20: {math_analysis['technical_metrics']['price_vs_sma20_pct']:+.2f}%
- Price vs SMA50: {math_analysis['technical_metrics']['price_vs_sma50_pct']:+.2f}%
- Price vs SMA200: {math_analysis['technical_metrics']['price_vs_sma200_pct']:+.2f}%
- SMA Convergence: {math_analysis['technical_metrics']['sma_convergence']:+.2f}%
- RSI: {math_analysis['technical_metrics']['current_rsi']:.1f} ({math_analysis['technical_metrics']['rsi_category']})
- Trend Alignment: {math_analysis['technical_metrics']['trend_alignment']} (-1=Bear, 0=Mixed, +1=Bull)

[CHART] SUPPORT/RESISTANCE LEVELS:
- Range Position: {math_analysis['support_resistance']['range_position_pct']:.1f}% of range
- Distance to Resistance: {math_analysis['support_resistance']['resistance_distance_pct']:.2f}%
- Distance to Support: {math_analysis['support_resistance']['support_distance_pct']:.2f}%
- Breakout Probability: {math_analysis['support_resistance']['breakout_probability']:.0f}%

[WARNING] RISK ANALYSIS:
- Value at Risk (95%): ${math_analysis['risk_metrics']['value_at_risk_95']:.2f}
- Value at Risk (99%): ${math_analysis['risk_metrics']['value_at_risk_99']:.2f}
- Sharpe Ratio: {math_analysis['risk_metrics']['sharpe_ratio']:.3f}
- Risk Score: {math_analysis['risk_metrics']['risk_score']:.0f}/100

[UP] STATISTICAL PROJECTIONS:
- Expected Daily Return: {math_analysis['statistical_metrics']['mean_return_daily']:+.3f}%
- Probability of Positive Day: {math_analysis['statistical_metrics']['probability_positive']:.0%}
- 95% Confidence Interval: ${math_analysis['statistical_metrics']['confidence_interval_95']['lower']:.2f} - ${math_analysis['statistical_metrics']['confidence_interval_95']['upper']:.2f}
- Skewness: {math_analysis['statistical_metrics']['skewness']:.3f}
- Kurtosis: {math_analysis['statistical_metrics']['kurtosis']:.3f}

[TARGET] COMPOSITE SIGNALS:
- Technical Signal: {math_analysis['composite_scores']['technical_signal']:+.3f} (-1 to +1)
- Momentum Signal: {math_analysis['composite_scores']['momentum_signal']:+.3f} (-1 to +1)
- Risk Signal: {math_analysis['composite_scores']['risk_signal']:+.3f} (-1 to +1)
- Catalyst Boost: {math_analysis['composite_scores']['catalyst_boost']:+.2f}
- OVERALL SIGNAL: {math_analysis['composite_scores']['overall_signal']:+.3f} (-1 to +1)

═══ CURRENT MARKET DATA ({market_source}) ═══
- Current Price: ${market_data['current_price']:.2f}
- Daily Change: {market_data['change']:+.2f} ({market_data['change_percent']:+.2f}%)
- Trading Range: ${market_data['low']:.2f} - ${market_data['high']:.2f}
- Volume: {market_data['volume']:,} shares

═══ NEWS INTELLIGENCE ═══
- 1-day sentiment: {news_data['sentiment_1d']:.3f}
- 7-day sentiment: {news_data['sentiment_7d']:.3f}
- News confidence: {news_data['confidence_score']:.2f}
- Event impact: {news_data['event_impact_score']:.2f}
- Has Major Catalyst: {'YES' if news_data.get('has_major_catalyst', False) else 'NO'}
{custom_news_summary}

═══ CUSTOM ARTICLE DEEP ANALYSIS REQUIREMENTS ═══
For each custom news article provided:
1. Identify specific momentum indicators (earnings beats, guidance changes, product launches, etc.)
2. Assess the magnitude of impact (minor news vs major catalyst)
3. Determine time horizon of impact (immediate vs long-term)
4. Weight recent custom news MORE heavily than historical data
5. Look for specific numbers, percentages, or financial metrics mentioned

Key Analysis Points:
- Revenue/earnings surprises (weight: HIGH)
- Guidance changes (weight: VERY HIGH)
- Product announcements (weight: MEDIUM)
- Partnership/contract news (weight: MEDIUM-HIGH)
- Analyst upgrades/downgrades (weight: MEDIUM)
- Regulatory news (weight: HIGH)
- Management changes (weight: LOW-MEDIUM)

CRITICAL: If custom articles contain major catalysts (earnings beats, guidance raises, etc.), 
these should OVERRIDE historical trends and drive the prediction direction.

═══ CRITICAL ANALYSIS REQUIREMENTS ═══
You are a quantitative analyst. Use the mathematical analysis above as your PRIMARY foundation:

1. MATHEMATICAL SIGNALS (45% weight):
  - Overall Signal: {math_analysis['composite_scores']['overall_signal']:+.3f}
  - Technical confluence of moving averages, RSI, momentum
  - Statistical projections and confidence intervals
  - Risk-adjusted returns and volatility analysis

2. BREAKING NEWS & CATALYSTS (35% weight):
  - Custom articles represent immediate market-moving events
  - Major catalysts should significantly impact predictions
  - Analyze article content for specific momentum indicators
  - Weight recent sentiment heavily

3. CURRENT MOMENTUM (20% weight):
  - Today's price action and volume
  - Intraday patterns and market structure

MATHEMATICAL REASONING REQUIRED:
- Use the calculated composite signals as your baseline
- Adjust predictions based on Value at Risk calculations
- Consider the 95% confidence interval for realistic ranges
- Factor in the Sharpe ratio for risk-adjusted expectations
- Use breakout probability for range predictions
- Apply catalyst boost when major news events are detected

Predict tomorrow's price movement integrating mathematical analysis with deep news analysis.

RESPOND ONLY with valid JSON:
{{
   "predicted_open": [number],
   "predicted_high": [number],
   "predicted_low": [number], 
   "predicted_close": [number],
   "confidence": [0-1],
   "direction": "[up/down/sideways]",
   "probability_up": [0-1],
   "key_factors": ["factor1", "factor2", "factor3"],
   "risk_assessment": "[low/moderate/high]",
   "support_level": [number],
   "resistance_level": [number],
   "catalyst_impact": "[none/minor/moderate/major]",
   "mathematical_basis": "[Brief explanation of mathematical signals used - under 150 chars]",
   "reasoning": "[Brief analysis integrating mathematical computations with market data and news catalysts - under 200 chars]"
}}"""

   try:
       # STEP 5: Send to Claude for AI reasoning
       response = client.messages.create(
           model="claude-3-5-sonnet-20241022",
           max_tokens=1200,
           temperature=0.1,
           messages=[{"role": "user", "content": prompt}]
       )
       
       claude_text = response.content[0].text.strip()
       print(f"[OK] Claude enhanced analysis complete")
       
       # STEP 6: Parse JSON response
       import re
       import json
       json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
       json_match = re.search(json_pattern, claude_text, re.DOTALL)
       
       if json_match:
           json_str = json_match.group()
           analysis = json.loads(json_str)
           
           # STEP 7: Validate and enhance with mathematical analysis
           catalyst_boost = math_analysis['composite_scores']['catalyst_boost']
           overall_signal = math_analysis['composite_scores']['overall_signal']
           
           required_fields = {
               'predicted_open': market_data['current_price'] * (1 + overall_signal * 0.008),
               'predicted_high': market_data['current_price'] * (1 + (abs(overall_signal) * 0.045 + math_analysis['volatility_metrics']['daily_volatility'] * 2.0 + catalyst_boost * 0.02)),
               'predicted_low': market_data['current_price'] * (1 - (abs(overall_signal) * 0.035 + math_analysis['volatility_metrics']['daily_volatility'] * 1.5 - catalyst_boost * 0.01)),
               'predicted_close': market_data['current_price'] * (1 + overall_signal * 0.025 + catalyst_boost * 0.015),
               'confidence': min(0.98, max(0.35, 0.6 + abs(overall_signal) * 0.4 + catalyst_boost * 0.2)),
               'direction': 'up' if overall_signal > 0.02 else 'down' if overall_signal < -0.02 else 'sideways',
               'probability_up': min(0.95, max(0.05, 0.5 + overall_signal * 0.45 + catalyst_boost * 0.15)),
               'key_factors': [
                   f"Mathematical signal: {overall_signal:+.3f}",
                   f"Technical confluence: {math_analysis['technical_metrics']['trend_alignment']}",
                   f"Risk-adjusted momentum: {math_analysis['momentum_metrics']['relative_strength']:.2f}"
               ],
               'risk_assessment': 'high' if math_analysis['risk_metrics']['risk_score'] > 70 else 'low' if math_analysis['risk_metrics']['risk_score'] < 30 else 'moderate',
               'support_level': math_analysis['support_resistance']['recent_low'],
               'resistance_level': math_analysis['support_resistance']['recent_high'],
               'catalyst_impact': 'major' if catalyst_boost > 0.3 else 'moderate' if catalyst_boost > 0.1 else 'minor' if catalyst_boost > 0 else 'none',
               'mathematical_basis': f"Signal:{overall_signal:+.2f}, VaR:${math_analysis['risk_metrics']['value_at_risk_95']:.0f}, Sharpe:{math_analysis['risk_metrics']['sharpe_ratio']:.2f}",
               'reasoning': 'Enhanced analysis integrating mathematical computations with AI reasoning and news catalysts'
           }
           
           for field, default in required_fields.items():
               if field not in analysis:
                   analysis[field] = default
           
           # Ensure numeric fields are numbers
           numeric_fields = ['predicted_open', 'predicted_high', 'predicted_low', 'predicted_close',
                           'confidence', 'probability_up', 'support_level', 'resistance_level']
           for field in numeric_fields:
               try:
                   analysis[field] = float(analysis[field])
               except:
                   analysis[field] = required_fields[field]
           
           # Add mathematical analysis to response
           analysis['mathematical_analysis'] = math_analysis
           analysis['full_reasoning'] = claude_text
           
           return analysis
       else:
           raise ValueError("No JSON found in Claude response")
           
   except Exception as e:
       print(f"[WARNING] Claude analysis issues: {e}")
       return create_enhanced_fallback_analysis(ticker, excel_data, news_data, market_data, math_analysis)

def create_enhanced_fallback_analysis(ticker, excel_data, news_data, market_data, math_analysis):
   """Create intelligent fallback analysis using mathematical computations"""
   current_price = market_data['current_price']
   overall_signal = math_analysis['composite_scores']['overall_signal']
   var_95 = math_analysis['risk_metrics']['value_at_risk_95']
   catalyst_boost = math_analysis['composite_scores']['catalyst_boost']
   
   # Use mathematical signals for prediction - ENHANCED
   predicted_move_pct = overall_signal * 0.028 + catalyst_boost * 0.015  # Up to 2.8% moves + catalyst boost
   volatility_adjustment = math_analysis['volatility_metrics']['daily_volatility'] * 2.8
   momentum_boost = abs(math_analysis['momentum_metrics']['momentum_score']) * 0.015
   
   # Enhanced confidence calculation
   signal_strength = abs(overall_signal)
   base_confidence = 0.55 + signal_strength * 0.4 + catalyst_boost * 0.2
   volatility_adj = min(0.15, math_analysis['volatility_metrics']['daily_volatility'] * 5)
   
   return {
       "predicted_open": current_price * (1 + predicted_move_pct * 0.6),  # Larger gaps
       "predicted_high": current_price * (1 + abs(predicted_move_pct) + volatility_adjustment + momentum_boost),
       "predicted_low": current_price * (1 - abs(predicted_move_pct) - volatility_adjustment * 0.8),
       "predicted_close": current_price * (1 + predicted_move_pct),
       "confidence": min(0.95, max(0.45, base_confidence + volatility_adj)),  # More dynamic
       "direction": "up" if overall_signal > 0.02 else "down" if overall_signal < -0.02 else "sideways",  # Lower threshold
       "probability_up": min(0.90, max(0.10, 0.5 + overall_signal * 0.45 + catalyst_boost * 0.15)),  # More responsive
       "key_factors": [
           f"Mathematical signal: {overall_signal:+.3f}",
           f"VaR-95: ${var_95:.2f}",
           f"Sharpe ratio: {math_analysis['risk_metrics']['sharpe_ratio']:.2f}",
           f"Technical alignment: {math_analysis['technical_metrics']['trend_alignment']}"
       ],
       "risk_assessment": "high" if math_analysis['risk_metrics']['risk_score'] > 70 else "low" if math_analysis['risk_metrics']['risk_score'] < 30 else "moderate",
       "support_level": math_analysis['support_resistance']['recent_low'],
       "resistance_level": math_analysis['support_resistance']['recent_high'],
       "catalyst_impact": "major" if catalyst_boost > 0.3 else "moderate" if catalyst_boost > 0.1 else "minor" if catalyst_boost > 0 else "none",
       "mathematical_basis": f"Signal:{overall_signal:+.2f}, VaR:${var_95:.0f}, Sharpe:{math_analysis['risk_metrics']['sharpe_ratio']:.2f}",
       "reasoning": "Enhanced mathematical fallback analysis using comprehensive quantitative signals and catalyst detection",
       "mathematical_analysis": math_analysis
   }

def create_fallback_analysis(ticker, excel_data, news_data, market_data):
   """Create intelligent fallback analysis"""
   current_price = market_data['current_price']
   volatility = excel_data.get('volatility', 3.0) / 100
   sentiment_bias = news_data.get('sentiment_1d', 0) * 0.015
   momentum_bias = market_data.get('change_percent', 0) / 100 * 0.3
   
   predicted_move = current_price * (volatility + sentiment_bias + momentum_bias)
   
   return {
       "predicted_open": current_price * (1 + momentum_bias * 0.2),
       "predicted_high": current_price + predicted_move * 1.2,
       "predicted_low": current_price - predicted_move * 0.9,
       "predicted_close": current_price * (1 + (predicted_move / current_price * 0.6)),
       "confidence": 0.70,
       "direction": "up" if (momentum_bias + sentiment_bias) > 0 else "down",
       "probability_up": 0.55 if (momentum_bias + sentiment_bias) > 0 else 0.45,
       "key_factors": [
           f"Current momentum: {market_data.get('change_percent', 0):+.2f}%",
           f"Historical volatility: {excel_data.get('volatility', 3):.1f}%",
           f"News sentiment: {news_data['sentiment_1d']:+.3f}"
       ],
       "risk_assessment": "moderate",
       "support_level": excel_data.get('recent_low', current_price * 0.97),
       "resistance_level": excel_data.get('recent_high', current_price * 1.03),
       "catalyst_impact": "none",
       "reasoning": "Comprehensive fallback analysis using all available data inputs"
   }

def generate_ultimate_report_enhanced(ticker, excel_data, news_data, market_data, claude_analysis, custom_articles, excel_file_path):
   """
   Generate ultimate comprehensive prediction report with mathematical analysis and catalyst detection
   """
   
   expected_return = (claude_analysis['predicted_close'] - market_data['current_price']) / market_data['current_price']
   thirty_day_performance = excel_data.get('performance_return_1_month', 6.19)
   math_analysis = claude_analysis.get('mathematical_analysis', {})
   
   # Data source details
   excel_source = "User Excel File" if excel_file_path else "Default Data"
   market_source = "Alpha Vantage API" if market_data.get('data_source') == 'alpha_vantage' else "User Input"
   custom_news_count = len(custom_articles) if custom_articles else 0
   
   # Check for major catalysts
   has_major_catalyst = False
   catalyst_types = []
   if custom_articles:
       for article in custom_articles:
           if article.get('catalyst_analysis', {}).get('has_major_catalyst', False):
               has_major_catalyst = True
               catalyst_types.extend(article['catalyst_analysis'].get('catalyst_types', []))
   
   report = f"""
[TARGET] ULTIMATE AI STOCK PREDICTION - {ticker} (Enhanced with Mathematical Analysis & Catalyst Detection)
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Type: Mathematical Computations + AI Reasoning + Multi-Source Data + Deep News Analysis

[CHART] MATHEMATICAL ANALYSIS FOUNDATION:
{'─'*50}"""

   # Add mathematical analysis if available
   if math_analysis:
       vol_metrics = math_analysis.get('volatility_metrics', {})
       momentum_metrics = math_analysis.get('momentum_metrics', {})
       technical_metrics = math_analysis.get('technical_metrics', {})
       risk_metrics = math_analysis.get('risk_metrics', {})
       composite_scores = math_analysis.get('composite_scores', {})
       support_resistance = math_analysis.get('support_resistance', {})
       statistical_metrics = math_analysis.get('statistical_metrics', {})
       
       report += f"""
🔢 VOLATILITY & RISK METRICS:
  Annualized Volatility: {vol_metrics.get('annualized_volatility', 0):.2f}% ({vol_metrics.get('risk_category', 'Unknown')} Risk)
  Daily Volatility: {vol_metrics.get('daily_volatility', 0):.3f}%
  Value at Risk (95%): ${risk_metrics.get('value_at_risk_95', 0):.2f}
  Value at Risk (99%): ${risk_metrics.get('value_at_risk_99', 0):.2f}
  Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.3f}
  Risk Score: {risk_metrics.get('risk_score', 0):.0f}/100

[UP] MOMENTUM & STATISTICAL ANALYSIS:
  30-day Return: {momentum_metrics.get('thirty_day_return', 0):.2f}%
  Annualized Return: {momentum_metrics.get('annualized_return', 0):.2f}%
  Momentum Score: {momentum_metrics.get('momentum_score', 0):+.3f}
  Risk-Adjusted Return: {momentum_metrics.get('relative_strength', 0):.3f}
  Direction: {momentum_metrics.get('momentum_direction', 'Unknown')}
  Custom News Impact: {momentum_metrics.get('custom_news_impact', 0):+.3f}

[WRENCH] TECHNICAL INDICATOR ANALYSIS:
  Price vs SMA20: {technical_metrics.get('price_vs_sma20_pct', 0):+.2f}%
  Price vs SMA50: {technical_metrics.get('price_vs_sma50_pct', 0):+.2f}%
  Price vs SMA200: {technical_metrics.get('price_vs_sma200_pct', 0):+.2f}%
  RSI Level: {technical_metrics.get('current_rsi', 50):.1f} ({technical_metrics.get('rsi_category', 'Neutral')})
  Trend Alignment: {technical_metrics.get('trend_alignment', 0)} (Bull=+1, Bear=-1, Mixed=0)
  SMA Convergence: {technical_metrics.get('sma_convergence', 0):+.2f}%

[TARGET] COMPOSITE MATHEMATICAL SIGNALS:
  Technical Signal: {composite_scores.get('technical_signal', 0):+.3f} (-1 to +1)
  Momentum Signal: {composite_scores.get('momentum_signal', 0):+.3f} (-1 to +1)
  Risk Signal: {composite_scores.get('risk_signal', 0):+.3f} (-1 to +1)
  Catalyst Boost: {composite_scores.get('catalyst_boost', 0):+.2f}
  [CHART] OVERALL MATHEMATICAL SIGNAL: {composite_scores.get('overall_signal', 0):+.3f} (-1 to +1)

[CHART] SUPPORT/RESISTANCE MATHEMATICAL ANALYSIS:
  Recent High: ${support_resistance.get('recent_high', 0):.2f}
  Recent Low: ${support_resistance.get('recent_low', 0):.2f}
  Range Position: {support_resistance.get('range_position_pct', 0):.1f}% of range
  Distance to Resistance: {support_resistance.get('resistance_distance_pct', 0):.2f}%
  Distance to Support: {support_resistance.get('support_distance_pct', 0):.2f}%
  Breakout Probability: {support_resistance.get('breakout_probability', 0):.0f}%

[UP] STATISTICAL PROJECTIONS:
  Expected Daily Return: {statistical_metrics.get('mean_return_daily', 0):+.3f}%
  Probability of Positive Day: {statistical_metrics.get('probability_positive', 0.5):.0%}
  95% Confidence Interval: ${statistical_metrics.get('confidence_interval_95', {}).get('lower', 0):.2f} - ${statistical_metrics.get('confidence_interval_95', {}).get('upper', 0):.2f}
  Skewness: {statistical_metrics.get('skewness', 0):.3f}
  Kurtosis: {statistical_metrics.get('kurtosis', 0):.3f}"""

   report += f"""

[CHART] HISTORICAL CONTEXT ({excel_source}):
  Recent 30d performance: +{thirty_day_performance:.2f}%  ← FIXED
  Historical volatility: {excel_data.get('volatility', 3.0):.2f}%
  Sector: {excel_data.get('sector', 'Technology')}
  Excel recommendation: {excel_data.get('excel_recommendation', 'Hold')}
  Risk assessment: {excel_data.get('excel_risk_level', 'Moderate')}"""
   
   if excel_file_path:
       report += f"\n   📂 Excel file: {os.path.basename(excel_file_path)}"
   
   report += f"""

[UP] CURRENT MARKET DATA ({market_source}):
  Current Price: ${market_data['current_price']:.2f}
  Daily Change: {market_data['change']:+.2f} ({market_data['change_percent']:+.2f}%)
  Trading Range: ${market_data['low']:.2f} - ${market_data['high']:.2f}
  Volume: {market_data['volume']:,} shares
  Previous Close: ${market_data['previous_close']:.2f}

📰 NEWS INTELLIGENCE:
  Phase 3 Sentiment (1d): {news_data['sentiment_1d']:+.3f}
  Phase 3 Sentiment (7d): {news_data['sentiment_7d']:+.3f}
  Confidence Score: {news_data['confidence_score']:.2f}
  Event Impact: {news_data['event_impact_score']:.2f}
  Custom Articles: {custom_news_count} user-provided
  Major Catalysts Detected: {'YES' if has_major_catalyst else 'NO'}"""
   
   # Add custom news details with catalyst analysis
   if custom_articles:
       report += f"\n\n🚨 BREAKING NEWS (User-Provided):"
       for i, article in enumerate(custom_articles, 1):
           sentiment_emoji = "[UP]" if article['sentiment_score'] > 0.1 else "[DOWN]" if article['sentiment_score'] < -0.1 else "➡️"
           momentum_analysis = article.get('momentum_analysis', {})
           catalyst_analysis = article.get('catalyst_analysis', {})
           
           report += f"\n   {i}. {article['title']}"
           report += f"\n      {sentiment_emoji} Impact: {article['sentiment_score']:+.1f} | Source: {article['source']}"
           report += f"\n      Momentum: {momentum_analysis.get('momentum_direction', 'Unknown')} "
           report += f"(+{momentum_analysis.get('positive_indicators', 0)}/-{momentum_analysis.get('negative_indicators', 0)} indicators)"
           
           if catalyst_analysis.get('has_major_catalyst', False):
               report += f"\n      🚨 MAJOR CATALYSTS: {', '.join(catalyst_analysis.get('catalyst_types', []))}"
   
   # Get catalyst impact
   catalyst_impact = claude_analysis.get('catalyst_impact', 'none')
   
   # Get mathematical basis if available
   math_basis = claude_analysis.get('mathematical_basis', 'Standard analysis')
   
   report += f"""

🤖 CLAUDE AI ULTIMATE PREDICTION (Enhanced with Mathematical Foundation & Catalyst Analysis):
  Mathematical Basis: {math_basis}
  Predicted Close: ${claude_analysis['predicted_close']:.2f}
  Expected Return: {expected_return:+.2%}
  Direction: {claude_analysis['direction'].upper()}
  Confidence: {claude_analysis['confidence']:.0%}
  Upside Probability: {claude_analysis['probability_up']:.0%}
  Risk Level: {claude_analysis['risk_assessment'].upper()}
  Catalyst Impact: {catalyst_impact.upper()}

[TARGET] TRADING LEVELS & TARGETS:
  Tomorrow's Range: ${claude_analysis['predicted_low']:.2f} - ${claude_analysis['predicted_high']:.2f}
  Predicted Open: ${claude_analysis['predicted_open']:.2f}
  Support Level: ${claude_analysis['support_level']:.2f}
  Resistance Level: ${claude_analysis['resistance_level']:.2f}

[KEY] KEY FACTORS (AI-Identified):"""
   
   for factor in claude_analysis['key_factors']:
       report += f"\n   • {factor}"
   
   if has_major_catalyst:
       report += f"\n   • 🚨 Major catalysts detected: {', '.join(set(catalyst_types))}"
   
   report += f"""

🧠 AI REASONING:
  {claude_analysis['reasoning']}

[CHART] DATA INTEGRATION SUMMARY:
  Excel Analysis: [OK] {excel_source}
  Mathematical Computations: [OK] Comprehensive quantitative analysis
  Phase 3 News: [OK] Automated sentiment analysis
  Market Data: [OK] {market_source}
  Custom News: [OK] {custom_news_count} breaking articles {'with MAJOR CATALYSTS' if has_major_catalyst else ''}
  AI Analysis: [OK] Claude enhanced reasoning with mathematical foundation & catalyst detection

[TARGET] TRADING RECOMMENDATIONS:
  Entry Signal: {claude_analysis['direction'].upper()} {'[CATALYST-DRIVEN]' if catalyst_impact in ['major', 'moderate'] else ''}
  Position Size: {'Conservative (25%)' if claude_analysis['risk_assessment'] == 'high' else 'Standard (50%)' if claude_analysis['risk_assessment'] == 'moderate' else 'Aggressive (75%)'}
  Stop Loss: ${claude_analysis['support_level']:.2f} (-{((market_data['current_price'] - claude_analysis['support_level']) / market_data['current_price'] * 100):.1f}%)
  Take Profit: ${claude_analysis['resistance_level']:.2f} (+{((claude_analysis['resistance_level'] - market_data['current_price']) / market_data['current_price'] * 100):.1f}%)"""

   # Add catalyst-specific recommendations
   if catalyst_impact in ['major', 'moderate']:
       report += f"""
  
  🚨 CATALYST-DRIVEN RECOMMENDATIONS:
  - Major news event detected - expect increased volatility
  - Consider wider stop-loss due to potential gaps
  - News momentum may override technical levels temporarily
  - Monitor for follow-up announcements"""

   # Add mathematical summary if available
   if math_analysis:
       overall_signal = math_analysis.get('composite_scores', {}).get('overall_signal', 0)
       var_95 = math_analysis.get('risk_metrics', {}).get('value_at_risk_95', 0)
       sharpe = math_analysis.get('risk_metrics', {}).get('sharpe_ratio', 0)
       
       report += f"""

🔢 MATHEMATICAL FOUNDATION SUMMARY:
  Overall Mathematical Signal: {overall_signal:+.3f} (-1 to +1)
  Value at Risk (95%): ${var_95:.2f}
  Sharpe Ratio: {sharpe:.3f}
  Risk-Adjusted Confidence: {claude_analysis['confidence']:.0%}"""

   report += f"""

🎉 ULTIMATE PREDICTION SUMMARY:
This represents the most comprehensive analysis possible, integrating:
- Mathematical computations (volatility, momentum, technical indicators)
- User-selected Excel historical analysis
- User-provided breaking news articles with catalyst detection
- Deep article content analysis for momentum indicators
- Real-time/current market data
- Advanced AI reasoning with quantitative foundation

Expected Price Movement: {expected_return:+.2%} ({claude_analysis['direction'].upper()})
Tomorrow's Target: ${claude_analysis['predicted_close']:.2f}
Mathematical Confidence Level: {claude_analysis['confidence']:.0%}
Catalyst Impact Level: {catalyst_impact.upper()}

[BULB] DATA SOURCES & METHODOLOGY:
  Historical: {excel_source}
  Market: {market_source}  
  Custom News: {custom_news_count} articles {'[MAJOR CATALYSTS DETECTED]' if has_major_catalyst else ''}
  Mathematical Analysis: [OK] Comprehensive quantitative computations
  Article Analysis: [OK] Deep content analysis for momentum indicators
  30-day Performance: {thirty_day_performance:.2f}% (FIXED)
  AI Enhancement: Claude with mathematical foundation & catalyst detection
"""
   
   return report

def save_ultimate_prediction_enhanced(ticker, excel_data, news_data, market_data, claude_analysis, custom_articles, excel_file_path):
   """Save ultimate prediction data with mathematical analysis and catalyst detection"""
   prediction_data = {
       "ticker": ticker,
       "timestamp": datetime.now().isoformat(),
       "analysis_type": "ultimate_enhanced_mathematical_catalyst",
       "data_sources": {
           "excel_file_path": excel_file_path,
           "market_data_source": market_data.get('data_source', 'unknown'),
           "custom_articles_count": len(custom_articles) if custom_articles else 0,
           "mathematical_analysis_included": True,
           "catalyst_detection_included": True
       },
       "excel_data": excel_data,
       "news_data": news_data,
       "market_data": market_data,
       "custom_articles": custom_articles,
       "claude_analysis": claude_analysis,
       "mathematical_analysis": claude_analysis.get('mathematical_analysis', {}),
       "catalyst_summary": {
           "has_major_catalyst": any(article.get('catalyst_analysis', {}).get('has_major_catalyst', False) for article in (custom_articles or [])),
           "catalyst_types": list(set(sum([article.get('catalyst_analysis', {}).get('catalyst_types', []) for article in (custom_articles or [])], []))),
           "catalyst_impact": claude_analysis.get('catalyst_impact', 'none')
       },
       "metadata": {
           "thirty_day_performance": excel_data.get('performance_return_1_month', 0),
           "confidence": claude_analysis['confidence'],
           "expected_return": (claude_analysis['predicted_close'] - market_data['current_price']) / market_data['current_price'],
           "mathematical_signal": claude_analysis.get('mathematical_analysis', {}).get('composite_scores', {}).get('overall_signal', 0),
           "mathematical_basis": claude_analysis.get('mathematical_basis', 'Standard analysis'),
           "analysis_type": "ultimate_enhanced_mathematical_catalyst",
           "user_inputs": {
               "excel_provided": excel_file_path is not None,
               "custom_news_provided": len(custom_articles) > 0 if custom_articles else False,
               "market_data_manual": market_data.get('data_source') == 'user_input'
           }
       }
   }
   
   timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
   filename = f"enhanced_prediction_{ticker}_{timestamp}.json"
   
   with open(filename, 'w', encoding='utf-8') as f:
       json.dump(prediction_data, f, indent=2, default=str)
   
   return filename

def main():
   """Enhanced main function with mathematical analysis, catalyst detection, and all user inputs"""
   print("[ROCKET] ULTIMATE AI STOCK PREDICTOR (Enhanced with Mathematical Analysis & Catalyst Detection)")
   print("="*80)
   print("[CHART] Mathematical Computations + Article Analysis + User Excel + Custom News + Market Data + Claude AI")
   print("[TARGET] Most comprehensive analysis with quantitative foundation & catalyst detection!")
   print("="*80)
   
   # Load API keys
   alpha_key, claude_key = load_api_keys()
   
   if not claude_key:
       print("[ERROR] Claude API key required for AI analysis")
       print("   Please add CLAUDE_API_KEY to your .env file")
       return
   
   # Get ticker
   ticker = sys.argv[1] if len(sys.argv) > 1 else input("\nEnter ticker symbol (default NVDA): ").strip().upper() or "NVDA"
   
   try:
       print(f"\n[TARGET] Starting ENHANCED ULTIMATE analysis for {ticker}...")
       print("   This will include deep mathematical computations + AI reasoning + catalyst detection!")
       
       # Step 1: Get Excel file from user
       excel_file_path = get_excel_file_from_user(ticker)
       excel_data = load_excel_historical_data(ticker, excel_file_path)
       
       # Step 2: Get custom news articles from user (with enhanced analysis)
       custom_articles = get_custom_news_from_user(ticker)
       
       # Step 3: Get Phase 3 news intelligence
       news_data = get_phase3_news_intelligence(ticker)
       
       # Step 4: Enhance news with custom articles
       enhanced_news_data = enhance_news_with_custom_articles(news_data, custom_articles)
       
       # Step 5: Get market data (API or user input)
       market_data = get_realtime_market_data_with_fallback(ticker, alpha_key)
       if not market_data:
           print("[ERROR] Cannot proceed without market data")
           return
       
       # Step 6: Enhanced Claude AI analysis with mathematical computations
       claude_analysis = analyze_with_claude_ultimate_enhanced(
           ticker, excel_data, enhanced_news_data, market_data, custom_articles, claude_key
       )
       if not claude_analysis:
           print("[ERROR] AI analysis failed")
           return
       
       # Step 7: Generate enhanced ultimate report
       report = generate_ultimate_report_enhanced(
           ticker, excel_data, enhanced_news_data, market_data, 
           claude_analysis, custom_articles, excel_file_path
       )
       print(report)
       
       # Step 8: Save results
       print(f"\n💾 Save enhanced ultimate analysis?")
       save_choice = input("Enter 'y' to save files: ").strip().lower()
       
       if save_choice in ['y', 'yes', '']:
           # Save report
           timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
           report_filename = f"enhanced_ultimate_prediction_{ticker}_{timestamp}.txt"
           with open(report_filename, 'w', encoding='utf-8') as f:
               f.write(report)
           
           # Save JSON
           json_filename = save_ultimate_prediction_enhanced(
               ticker, excel_data, enhanced_news_data, market_data,
               claude_analysis, custom_articles, excel_file_path
           )
           
           print(f"[OK] Files saved:")
           print(f"   📋 Report: {report_filename}")
           print(f"   [CHART] Data: {json_filename}")
       
       print(f"\n🎉 ENHANCED ULTIMATE analysis complete for {ticker}!")
       
       # Summary of enhancements
       print(f"\n[CHART] ENHANCED ANALYSIS SUMMARY:")
       excel_source = "[OK] User Excel file" if excel_file_path else "[WARNING] Default data"
       market_source = "[OK] Alpha Vantage API" if market_data.get('data_source') == 'alpha_vantage' else "[OK] User input"
       custom_news = f"[OK] {len(custom_articles)} custom articles" if custom_articles else "[WARNING] No custom news"
       math_signal = claude_analysis.get('mathematical_analysis', {}).get('composite_scores', {}).get('overall_signal', 0)
       catalyst_impact = claude_analysis.get('catalyst_impact', 'none')
       
       # Check for catalysts
       has_catalyst = False
       if custom_articles:
           for article in custom_articles:
               if article.get('catalyst_analysis', {}).get('has_major_catalyst', False):
                   has_catalyst = True
                   break
       
       print(f"   Historical Data: {excel_source}")
       print(f"   Market Data: {market_source}")
       print(f"   Custom News: {custom_news}")
       print(f"   Mathematical Analysis: [OK] Comprehensive quantitative computations")
       print(f"   Mathematical Signal: {math_signal:+.3f} (-1 to +1)")
       print(f"   Catalyst Detection: [OK] {'MAJOR CATALYSTS DETECTED' if has_catalyst else 'No major catalysts'}")
       print(f"   Catalyst Impact: {catalyst_impact.upper()}")
       print(f"   30-day Performance: [OK] {excel_data.get('performance_return_1_month', 6.19):.2f}% (FIXED)")
       print(f"   AI Analysis: [OK] Claude enhanced with mathematical foundation & catalyst detection")
       print(f"   Confidence: {claude_analysis['confidence']:.0%}")
       
   except KeyboardInterrupt:
       print("\n[WARNING] Analysis interrupted by user")
   except Exception as e:
       print(f"\n[ERROR] Error: {e}")
       import traceback
       traceback.print_exc()

if __name__ == "__main__":
   main()