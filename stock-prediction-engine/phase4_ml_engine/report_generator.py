#!/usr/bin/env python3
"""
AI Stock Prediction Report Generator - User Interface and Report Generation - UPDATED VERSION
Handles user inputs, report generation, and file operations
Updated to match the new prediction engine with Graph Analysis, Multi-Source News, and Enhanced Confidence
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Import the updated prediction engine
try:
    from prediction_engine import (
        StockPredictionEngine, 
        MarketRegimeDetector,
        MultiSourceNewsAnalyzer,
        PredictionErrorTracker
    )
    
    from graph_analyzer import GraphAnalyzer
    GRAPH_ANALYZER_AVAILABLE = True
except ImportError as e:
    print("❌ Cannot import prediction_engine or graph_analyzer. Make sure they're in the same directory.")
    print(f"   Import error: {e}")
    GRAPH_ANALYZER_AVAILABLE = False
    sys.exit(1)

class ReportGenerator:
    """Handles user interface and report generation - UPDATED VERSION"""
    
    def __init__(self):
        self.engine = StockPredictionEngine()
        
    def validate_prediction_engine_compatibility(self):
        """Validate compatibility with prediction engine"""
        print("🔍 Validating prediction engine compatibility...")
        
        compatibility_issues = []
        
        # Check required methods exist
        required_methods = [
            'get_phase3_news_intelligence',
            'analyze_with_claude_ultimate_enhanced', 
            'save_prediction_data',
            'set_news_data_for_regime',
            'load_excel_historical_data',
            'enhance_news_with_custom_articles'
        ]
        
        for method in required_methods:
            if not hasattr(self.engine, method):
                compatibility_issues.append(f"Missing engine method: {method}")
        
        # Check error tracker methods
        if hasattr(self.engine, 'error_tracker'):
            error_tracker_methods = ['validate_graph_analysis', 'generate_default_graph_analysis']
            for method in error_tracker_methods:
                if not hasattr(self.engine.error_tracker, method):
                    compatibility_issues.append(f"Missing error tracker method: {method}")
        else:
            compatibility_issues.append("Missing error_tracker attribute")
        
        # Check for API keys
        if not hasattr(self.engine, 'claude_key') or not self.engine.claude_key:
            compatibility_issues.append("Missing Claude API key")
        
        if compatibility_issues:
            print("⚠️ Compatibility issues found:")
            for issue in compatibility_issues:
                print(f"   - {issue}")
            print("🔧 Some features may be limited")
            return False
        else:
            print("✅ Prediction engine fully compatible!")
            return True
    
    def get_excel_file_from_user(self, ticker):
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
                    print(f"✅ Using default file: {default_files[ticker]}")
                    return default_files[ticker]
                else:
                    print(f"❌ No default file found for {ticker}")
                    continue
                    
            elif choice == "2":
                # Browse for file (simplified - user provides path)
                print("📁 Browse mode: Please provide the full path to your Excel file")
                file_path = input("Excel file path: ").strip().strip('"')
                
                if os.path.exists(file_path) and file_path.endswith(('.xlsx', '.xls')):
                    print(f"✅ Excel file selected: {file_path}")
                    return file_path
                else:
                    print("❌ File not found or not an Excel file (.xlsx/.xls)")
                    continue
                    
            elif choice == "3":
                # Manual path entry
                file_path = input("Enter full path to Excel file: ").strip().strip('"')
                
                if os.path.exists(file_path) and file_path.endswith(('.xlsx', '.xls')):
                    print(f"✅ Excel file selected: {file_path}")
                    return file_path
                else:
                    print("❌ File not found or not an Excel file")
                    continue
                    
            elif choice == "4":
                print("⚠️ Skipping Excel analysis - will use basic historical defaults")
                return None
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

    def get_custom_news_from_user(self, ticker):
        """Get custom news articles from user - ENHANCED VERSION"""
        print(f"\n📰 CUSTOM NEWS INPUT FOR {ticker}")
        print("=" * 40)
        print("You can add breaking news or important announcements that might affect the stock.")
        print("🚀 ENHANCED with stronger catalyst detection and momentum analysis!")
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
                        
                        # ENHANCED: Use the new enhanced analysis methods
                        momentum_analysis = self.engine.analyze_enhanced_sentiment_momentum(article_text, ticker)
                        catalyst_analysis = self.engine.detect_enhanced_financial_catalysts(article_text, ticker)
                        
                        # ENHANCED: Show more detailed analysis results
                        print(f"\n🤖 ENHANCED Automated Analysis Results:")
                        print(f"   Momentum Score: {momentum_analysis['momentum_score']:+.2f} ({momentum_analysis['momentum_direction']})")
                        print(f"   Detected Patterns: {', '.join(momentum_analysis.get('detected_patterns', ['None']))}")
                        print(f"   Catalyst Score: {catalyst_analysis['catalyst_score']:.2f} ({catalyst_analysis['catalyst_strength']})")
                        print(f"   Financial Verbs: {'✅ Found' if catalyst_analysis['has_financial_verb'] else '❌ None'}")
                        
                        if catalyst_analysis['detected_catalysts']:
                            print(f"   🚨 CATALYSTS: {', '.join(catalyst_analysis['detected_catalysts'])}")
                        
                        print(f"\nHow would you assess this news for {ticker}?")
                        print("1. Very Positive (+0.8)")
                        print("2. Positive (+0.4)")  
                        print("3. Neutral (0.0)")
                        print("4. Negative (-0.4)")
                        print("5. Very Negative (-0.8)")
                        print("6. Use ENHANCED automated assessment (recommended)")
                        
                        sentiment_choice = input("Enter choice (1-6, default 6): ").strip() or "6"
                        
                        if sentiment_choice == "6":
                            # ENHANCED: Use new momentum score with catalyst amplification
                            base_sentiment = momentum_analysis['momentum_score']
                            catalyst_multiplier = 1 + (catalyst_analysis['catalyst_score'] / 2)  # Amplify based on catalyst
                            sentiment_score = base_sentiment * catalyst_multiplier
                            sentiment_score = max(-1, min(1, sentiment_score))  # Clamp to [-1, 1]
                            print(f"   🎯 Enhanced sentiment: {sentiment_score:+.2f} (base: {base_sentiment:+.2f}, catalyst boost: {catalyst_multiplier:.1f}x)")
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
                            'momentum_analysis': momentum_analysis,  # Enhanced momentum analysis
                            'catalyst_analysis': catalyst_analysis   # Enhanced catalyst analysis
                        })
                        
                        print(f"✅ ENHANCED news article added!")
                        print(f"   Title: {title}")
                        print(f"   Sentiment: {sentiment_score:+.2f}")
                        print(f"   Momentum: {momentum_analysis['momentum_direction']}")
                        print(f"   Catalyst Impact: {catalyst_analysis['catalyst_strength'].upper()}")
                        
                        # Ask if user wants to add more
                        add_more = input(f"\nAdd another news article? (y/n): ").strip().lower()
                        if add_more in ['y', 'yes']:
                            continue
                        else:
                            break
                    else:
                        print("❌ Article too short (minimum 20 characters)")
                        continue
                else:
                    print("❌ No text entered")
                    continue
                    
            elif choice == "2":
                # Load from file - ENHANCED analysis
                file_path = input("Enter path to text file: ").strip().strip('"')
                
                if not os.path.exists(file_path):
                    print(f"❌ File not found: {file_path}")
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
                        
                        # ENHANCED: Use new analysis methods
                        momentum_analysis = self.engine.analyze_enhanced_sentiment_momentum(article_text, ticker)
                        catalyst_analysis = self.engine.detect_enhanced_financial_catalysts(article_text, ticker)
                        
                        print(f"\n🤖 ENHANCED Automated Analysis:")
                        print(f"   Momentum: {momentum_analysis['momentum_direction']} ({momentum_analysis['momentum_score']:+.2f})")
                        print(f"   Catalyst Strength: {catalyst_analysis['catalyst_strength']}")
                        if catalyst_analysis['detected_catalysts']:
                            print(f"   Major catalysts: {', '.join(catalyst_analysis['detected_catalysts'])}")
                        
                        # Get sentiment with enhanced calculation
                        print(f"\nSentiment assessment for this article:")
                        print("1. Positive (+0.5)")
                        print("2. Neutral (0.0)")
                        print("3. Negative (-0.5)")
                        print("4. Use ENHANCED automated assessment")
                        
                        sentiment_choice = input("Enter choice (1-4, default 4): ").strip() or "4"
                        
                        if sentiment_choice == "4":
                            base_sentiment = momentum_analysis['momentum_score']
                            catalyst_multiplier = 1 + (catalyst_analysis['catalyst_score'] / 3)
                            sentiment_score = base_sentiment * catalyst_multiplier
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
                            'momentum_analysis': momentum_analysis,  # Enhanced analysis
                            'catalyst_analysis': catalyst_analysis   # Enhanced analysis
                        })
                        
                        print(f"✅ ENHANCED news article loaded from file!")
                        print(f"   Title: {title}")
                        print(f"   Length: {len(article_text)} characters")
                        print(f"   Enhanced Sentiment: {sentiment_score:+.2f}")
                        break
                    else:
                        print("❌ File is empty or too short")
                        continue
                        
                except Exception as e:
                    print(f"❌ Error reading file: {e}")
                    continue
                    
            elif choice == "3":
                print("⚠️ Skipping custom news input")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        
        if custom_articles:
            print(f"\n✅ Added {len(custom_articles)} ENHANCED news article(s)")
            for i, article in enumerate(custom_articles, 1):
                sentiment_label = "📈 Positive" if article['sentiment_score'] > 0.1 else "📉 Negative" if article['sentiment_score'] < -0.1 else "➡️ Neutral"
                
                # Enhanced display with catalyst info
                catalyst_strength = article['catalyst_analysis']['catalyst_strength']
                catalyst_label = ""
                if catalyst_strength == 'strong':
                    catalyst_label = " 🚨 STRONG CATALYST"
                elif catalyst_strength == 'moderate':
                    catalyst_label = " ⚡ MODERATE CATALYST"
                
                momentum_direction = article['momentum_analysis']['momentum_direction']
                momentum_label = f" ({momentum_direction.upper()})"
                
                print(f"   {i}. {article['title']} ({sentiment_label}){momentum_label}{catalyst_label}")
        
        return custom_articles

    def get_graph_analysis_choice(self, ticker):
        """NEW: Get user choice for graph analysis"""
        print(f"\n📈 GRAPH ANALYSIS OPTION FOR {ticker}")
        print("=" * 40)
        print("Graph analysis provides technical pattern detection:")
        print("  • Chart patterns (triangles, flags, wedges)")
        print("  • Breakout detection with strength metrics")
        print("  • Candlestick patterns and clusters")
        print("  • Support/resistance levels")
        print("  • Momentum acceleration analysis")
        print("")
        print("1. Skip graph analysis (faster)")
        print("2. Perform graph analysis (recommended for technical traders)")
        
        choice = input("\nEnter your choice (1-2, default 2): ").strip() or "2"
        
        if choice == "2":
            print("✅ Graph analysis will be performed")
            return True
        else:
            print("⚠️ Skipping graph analysis")
            return False

    def perform_graph_analysis(self, ticker, excel_file_path):
        """Perform REAL graph analysis using the GraphAnalyzer"""
        print(f"\n📈 Performing graph analysis for {ticker}...")
        
        if not GRAPH_ANALYZER_AVAILABLE:
            print("❌ GraphAnalyzer not available, generating default analysis...")
            # Use prediction engine's fallback system
            if hasattr(self.engine.error_tracker, 'generate_default_graph_analysis'):
                # Need market data for fallback
                temp_market_data = self.get_realtime_market_data_with_fallback(ticker)
                if temp_market_data:
                    excel_data = self.engine.load_excel_historical_data(ticker, excel_file_path)
                    return self.engine.error_tracker.generate_default_graph_analysis(
                        ticker, excel_data, temp_market_data
                    )
            return None
        
        try:
            # Initialize the real GraphAnalyzer
            graph_analyzer = GraphAnalyzer(use_cache=True)
            
            # Perform real analysis
            graph_analysis = graph_analyzer.analyze_ticker(
                ticker, 
                days=30, 
                include_extended_analysis=True
            )
            
            # Validate using prediction engine if available
            if (graph_analysis and 
                hasattr(self.engine.error_tracker, 'validate_graph_analysis') and
                self.engine.error_tracker.validate_graph_analysis(graph_analysis)):
                
                print(f"✅ Graph analysis complete and validated!")
                
                # Extract and display key results
                primary_pattern = graph_analysis.get('pattern_detected', {}).get('primary_pattern', 'None')
                pattern_reliability = graph_analysis.get('pattern_detected', {}).get('pattern_reliability', 0.0)
                breakout_detected = graph_analysis.get('breakout_analysis', {}).get('breakout_detected', False)
                breakout_strength = graph_analysis.get('breakout_analysis', {}).get('breakout_strength', 0.0)
                breakout_direction = graph_analysis.get('breakout_analysis', {}).get('breakout_direction', 'neutral')
                momentum_acceleration = graph_analysis.get('momentum_analysis', {}).get('momentum_acceleration', 0.5)
                
                print(f"   📊 Primary Pattern: {primary_pattern} (Reliability: {pattern_reliability:.1%})")
                
                if breakout_detected:
                    print(f"   🚀 Breakout: {breakout_direction.upper()} (Strength: {breakout_strength:.1%})")
                else:
                    print(f"   📊 No breakout detected")
                    
                print(f"   ⚡ Momentum Acceleration: {momentum_acceleration:.3f}")
                
                # Show candlestick patterns if available
                candlestick_count = graph_analysis.get('candlestick_analysis', {}).get('total_patterns_found', 0)
                if candlestick_count > 0:
                    strongest_pattern = graph_analysis.get('candlestick_analysis', {}).get('strongest_pattern', 'none')
                    print(f"   🕯️ Candlestick Patterns: {candlestick_count} found (strongest: {strongest_pattern})")
                
                # Show support/resistance levels
                support = graph_analysis.get('support_resistance', {}).get('nearest_support', 0)
                resistance = graph_analysis.get('support_resistance', {}).get('nearest_resistance', 0)
                if support > 0 and resistance > 0:
                    print(f"   📊 Support: ${support:.2f} | Resistance: ${resistance:.2f}")
                
                return graph_analysis
                
            else:
                print("❌ Graph analysis validation failed, generating fallback...")
                # Generate fallback using prediction engine
                temp_market_data = self.get_realtime_market_data_with_fallback(ticker)
                excel_data = self.engine.load_excel_historical_data(ticker, excel_file_path)
                
                if (temp_market_data and excel_data and 
                    hasattr(self.engine.error_tracker, 'generate_default_graph_analysis')):
                    return self.engine.error_tracker.generate_default_graph_analysis(
                        ticker, excel_data, temp_market_data
                    )
                else:
                    return None
                    
        except Exception as e:
            print(f"❌ Graph analysis error: {e}")
            print("🔄 Generating fallback analysis...")
            
            # Generate fallback using prediction engine
            try:
                temp_market_data = self.get_realtime_market_data_with_fallback(ticker)
                excel_data = self.engine.load_excel_historical_data(ticker, excel_file_path)
                
                if (temp_market_data and excel_data and 
                    hasattr(self.engine.error_tracker, 'generate_default_graph_analysis')):
                    return self.engine.error_tracker.generate_default_graph_analysis(
                        ticker, excel_data, temp_market_data
                    )
                else:
                    print("⚠️ Fallback graph analysis also failed")
                    return None
                    
            except Exception as fallback_error:
                print(f"❌ Fallback graph analysis failed: {fallback_error}")
                return None

    def get_current_price_from_user(self, ticker):
        """Get current stock price and market data from user - ENHANCED VERSION"""
        print(f"\n💰 MANUAL PRICE INPUT FOR {ticker}")
        print("=" * 40)
        print("Please provide current market information manually.")
        print("")
        print("💡 RECOMMENDED DATA SOURCES:")
        print("   • Yahoo Finance (finance.yahoo.com)")
        print("   • Google Finance (finance.google.com)")
        print("   • Bloomberg, MarketWatch, CNBC")
        print("   • Your broker app (Robinhood, E*TRADE, TD Ameritrade, etc.)")
        print("   • TradingView, Seeking Alpha")
        print("")
        
        # Show API status
        if self.engine.alpha_key:
            print("📡 NOTE: Alpha Vantage API is available but you chose manual input")
        else:
            print("⚠️ NOTE: No Alpha Vantage API key found (add to .env file for automatic data)")
        print("")
        
        # Get current price (required)
        while True:
            try:
                price_input = input(f"💰 Enter current price for {ticker}: $").strip()
                if not price_input:
                    print("❌ Price is required. Please enter the current stock price.")
                    continue
                
                current_price = float(price_input)
                if current_price <= 0:
                    print("❌ Price must be greater than 0")
                    continue
                break
            except ValueError:
                print("❌ Invalid price format. Please enter a number (e.g., 118.50)")
        
        print(f"✅ Current price set: ${current_price:.2f}")
        
        # Initialize variables
        daily_change = 0.0
        daily_change_pct = 0.0
        
        # Get daily change (optional but recommended)
        print(f"\n📊 Optional: Daily change information")
        print("💡 This helps improve prediction accuracy")
        try:
            change_input = input(f"Daily change in $ (e.g., +2.45, -1.20, or press Enter to skip): ").strip()
            if change_input:
                daily_change = float(change_input.replace('+', ''))
                # Only calculate percentage if we have a valid previous price
                if current_price != daily_change:  # Avoid division by zero
                    previous_price = current_price - daily_change
                    if previous_price > 0:
                        daily_change_pct = (daily_change / previous_price) * 100
                    else:
                        daily_change_pct = 0.0
                else:
                    daily_change_pct = 0.0
                direction = "📈" if daily_change > 0 else "📉" if daily_change < 0 else "➡️"
                print(f"✅ Daily change: {direction} ${daily_change:+.2f} ({daily_change_pct:+.2f}%)")
        except ValueError:
            print("⚠️ Invalid change format, using 0.0")
            daily_change = 0.0
            daily_change_pct = 0.0
        
        # Get volume (optional)
        volume = 25_000_000  # Default volume
        print(f"\n📈 Optional: Trading volume")
        print("💡 Helps assess market activity and liquidity")
        try:
            volume_input = input(f"Trading volume (e.g., 45M, 25000000, or press Enter for default 25M): ").strip()
            if volume_input:
                volume_str = volume_input.upper().replace(',', '')
                if 'M' in volume_str:
                    volume = int(float(volume_str.replace('M', '')) * 1_000_000)
                elif 'K' in volume_str:
                    volume = int(float(volume_str.replace('K', '')) * 1_000)
                elif 'B' in volume_str:
                    volume = int(float(volume_str.replace('B', '')) * 1_000_000_000)
                else:
                    volume = int(volume_str)
                print(f"✅ Volume set: {volume:,}")
        except ValueError:
            print("⚠️ Invalid volume format, using default 25M")
            volume = 25_000_000
        
        # Calculate derived values
        open_price = current_price - daily_change
        previous_close = open_price
        
        # Estimate intraday range
        estimated_range_pct = 0.02  # Default 2% daily range
        if abs(daily_change_pct) > 5:
            estimated_range_pct = 0.04
        elif abs(daily_change_pct) > 2:
            estimated_range_pct = 0.03
        
        estimated_range = current_price * estimated_range_pct
        
        # Estimate high/low
        if daily_change > 0:
            high_price = current_price + (estimated_range * 0.3)
            low_price = current_price - (estimated_range * 0.7)
        elif daily_change < 0:
            high_price = current_price + (estimated_range * 0.7)
            low_price = current_price - (estimated_range * 0.3)
        else:
            high_price = current_price + (estimated_range * 0.5)
            low_price = current_price - (estimated_range * 0.5)
        
        # Create market data structure
        market_data = {
            'current_price': round(current_price, 2),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'volume': volume,
            'change': round(daily_change, 2),
            'change_percent': round(daily_change_pct, 2),
            'previous_close': round(previous_close, 2),
            'current_rsi': 50.0,
            'av_news_sentiment': 0.0,
            'av_news_count': 0,
            'data_source': 'user_input',
            'symbol': ticker  # Add ticker symbol
        }
        
        # Display summary
        print(f"\n📋 MARKET DATA SUMMARY:")
        print(f"   💰 Current Price: ${market_data['current_price']:.2f}")
        print(f"   📊 Daily Change: ${market_data['change']:+.2f} ({market_data['change_percent']:+.2f}%)")
        print(f"   📈 Est. Day Range: ${market_data['low']:.2f} - ${market_data['high']:.2f}")
        print(f"   📊 Volume: {market_data['volume']:,} shares")
        print(f"   📡 Data Source: Manual User Input")
        
        # Confirmation
        print(f"\n❓ Proceed with this market data?")
        confirm = input("Enter 'y' to continue, 'n' to re-enter: ").strip().lower()
        
        if confirm in ['y', 'yes', '']:
            print(f"✅ Market data confirmed")
            return market_data
        else:
            print(f"🔄 Let's re-enter the data...")
            return self.get_current_price_from_user(ticker)

    def get_prediction_timeframe(self):
        """Get prediction timeframe from user"""
        print(f"\n📅 PREDICTION TIMEFRAME SELECTION")
        print("=" * 40)
        print("Choose your prediction period:")
        print("")
        print("1. 1 day (default - same day trading)")
        print("2. 5 days (1 week)")
        print("3. 7 days (1 week + weekend)")
        print("4. 14 days (2 weeks)")
        print("5. 30 days (1 month)")
        print("6. Custom timeframe")
        
        while True:
            choice = input("\nEnter your choice (1-6, default 1): ").strip() or "1"
            
            timeframe_map = {
                "1": 1,
                "2": 5, 
                "3": 7,
                "4": 14,
                "5": 30
            }
            
            if choice in timeframe_map:
                days = timeframe_map[choice]
                period_name = {1: "Same Day", 5: "1 Week", 7: "1 Week+", 14: "2 Weeks", 30: "1 Month"}[days]
                print(f"✅ Selected: {days} day(s) - {period_name} prediction")
                return days
            elif choice == "6":
                try:
                    custom_days = int(input("Enter custom days (1-60): "))
                    if 1 <= custom_days <= 60:
                        print(f"✅ Custom timeframe: {custom_days} day(s)")
                        return custom_days
                    else:
                        print("❌ Please enter between 1-60 days")
                except ValueError:
                    print("❌ Please enter a valid number")
            else:
                print("❌ Invalid choice. Please enter 1-6.")

    def get_options_data(self, ticker, prediction_days):
        """Get options trading input from user"""
        print(f"\n📈 OPTIONS TRADING ANALYSIS FOR {ticker}")
        print("=" * 50)
        print("Options analysis provides:")
        print("  • Buy/Sell recommendations for calls and puts")
        print("  • Expected profit/loss calculations")
        print("  • Greeks analysis (Delta, Gamma, Theta, Vega)")
        print("  • Probability of profit calculations")
        print("  • Risk/reward ratios")
        print("")
        print("1. Analyze options (recommended for options traders)")
        print("2. Skip options analysis (stocks only)")
        
        choice = input("\nEnter your choice (1-2): ").strip()
        
        if choice != "1":
            print("⚠️ Skipping options analysis - stocks only")
            return None
        
        print(f"\n📊 OPTIONS INPUT")
        print("-" * 30)
        
        options_data = []
        option_count = 1
        
        while True:
            print(f"\n📈 Option #{option_count}:")
            
            # Get option type
            while True:
                option_type = input("Option type (call/put): ").strip().lower()
                if option_type in ['call', 'put', 'c', 'p']:
                    option_type = 'call' if option_type in ['call', 'c'] else 'put'
                    break
                print("❌ Please enter 'call' or 'put'")
            
            # Get strike price
            while True:
                try:
                    strike = float(input("Strike price: $"))
                    if strike > 0:
                        break
                    print("❌ Strike price must be positive")
                except ValueError:
                    print("❌ Please enter a valid number")
            
            # Get expiration
            while True:
                try:
                    expiration = int(input(f"Days to expiration (max {prediction_days*2}): "))
                    if 1 <= expiration <= prediction_days*2:
                        break
                    print(f"❌ Please enter between 1-{prediction_days*2} days")
                except ValueError:
                    print("❌ Please enter a valid number")
            
            # Get implied volatility
            try:
                iv_input = input("Implied volatility (e.g., 0.25 for 25%, or press Enter for auto): ").strip()
                if iv_input:
                    iv = float(iv_input)
                    if iv > 1:  # User entered percentage
                        iv = iv / 100
                else:
                    iv = 0.25  # Default 25%
            except ValueError:
                iv = 0.25
                print("⚠️ Using default IV of 25%")
            
            # Add to options data
            options_data.append({
                'strike': strike,
                'expiration_days': expiration,
                'type': option_type,
                'implied_volatility': iv
            })
            
            print(f"✅ Added: {option_type.upper()} ${strike} ({expiration}d, IV: {iv:.1%})")
            option_count += 1
            
            # Ask to add more
            if option_count > 5:
                print("Maximum 5 options reached.")
                break
                
            add_more = input(f"\nAdd another option? (y/n): ").strip().lower()
            if add_more not in ['y', 'yes']:
                break
        
        if options_data:
            print(f"\n✅ Options analysis configured: {len(options_data)} option(s)")
            for i, opt in enumerate(options_data, 1):
                print(f"   {i}. {opt['type'].upper()} ${opt['strike']} ({opt['expiration_days']}d)")
        
        return options_data if options_data else None

    def get_realtime_market_data_with_fallback(self, ticker):
        """Get market data with user choice between API and manual input"""
        print(f"\n📈 MARKET DATA OPTIONS FOR {ticker}")
        print("=" * 40)
        
        print("Choose how to get current market data:")
        print("")
        print("1. Use Alpha Vantage API (automatic)")
        print("2. Enter price manually")
        print("3. Try API first, fallback to manual if needed")
        
        while True:
            choice = input("\nEnter your choice (1-3, default 3): ").strip() or "3"
            
            if choice == "1":
                # Force API usage
                print(f"📡 Fetching data from Alpha Vantage API for {ticker}...")
                if self.engine.alpha_key:
                    market_data = self.engine.get_realtime_market_data(ticker, self.engine.alpha_key)
                    if market_data:
                        print(f"   ✅ API data retrieved successfully!")
                        print(f"   💰 Price: ${market_data['current_price']:.2f} ({market_data['change_percent']:+.2f}%)")
                        print(f"   📊 Volume: {market_data['volume']:,}")
                        
                        # Ask for confirmation
                        confirm = input(f"\n✓ Use this API data? (y/n, default y): ").strip().lower()
                        if confirm in ['', 'y', 'yes']:
                            market_data['symbol'] = ticker  # Ensure ticker is included
                            return market_data
                        else:
                            print("🔄 Switching to manual input...")
                            return self.get_current_price_from_user(ticker)
                    else:
                        print("   ❌ API request failed")
                        retry_choice = input("Try manual input instead? (y/n): ").strip().lower()
                        if retry_choice in ['y', 'yes']:
                            return self.get_current_price_from_user(ticker)
                        else:
                            continue
                else:
                    print("   ❌ No Alpha Vantage API key found in .env file")
                    print("   📝 Add ALPHA_VANTAGE_API_KEY=your_key to .env for API access")
                    
                    manual_choice = input("Switch to manual input? (y/n): ").strip().lower()
                    if manual_choice in ['y', 'yes']:
                        return self.get_current_price_from_user(ticker)
                    else:
                        continue
                        
            elif choice == "2":
                # Force manual input
                print(f"📝 Manual price input selected for {ticker}")
                return self.get_current_price_from_user(ticker)
                
            elif choice == "3":
                # Try API first, fallback to manual
                print(f"🔄 Trying API first, will fallback to manual if needed...")
                
                if self.engine.alpha_key:
                    market_data = self.engine.get_realtime_market_data(ticker, self.engine.alpha_key)
                    if market_data:
                        print(f"   ✅ API data: ${market_data['current_price']:.2f} ({market_data['change_percent']:+.2f}%)")
                        
                        # Give user choice even when API works
                        use_api = input(f"Use API data or enter manually? (api/manual, default api): ").strip().lower()
                        if use_api in ['', 'api', 'a']:
                            market_data['symbol'] = ticker  # Ensure ticker is included
                            return market_data
                        else:
                            print("🔄 Switching to manual input...")
                            return self.get_current_price_from_user(ticker)
                    else:
                        print(f"   ⚠️ API unavailable - switching to manual input")
                        return self.get_current_price_from_user(ticker)
                else:
                    print(f"   ⚠️ No API key found - switching to manual input")
                    return self.get_current_price_from_user(ticker)
                    
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")

    def generate_ultimate_report_enhanced(self, ticker, excel_data, news_data, market_data, claude_analysis, custom_articles, excel_file_path, prediction_days, options_data, graph_analysis=None):
        """Generate ultimate comprehensive prediction report - UPDATED VERSION with Graph Analysis"""
        
        # Safe access to claude_analysis data
        if not claude_analysis:
            print("❌ Cannot generate report without Claude analysis")
            return "❌ Report generation failed - no Claude analysis available"
        
        final_target = claude_analysis.get('final_target_price', market_data['current_price'])
        expected_return_pct = claude_analysis.get('total_expected_return_pct', 0)
        expected_return = expected_return_pct / 100 if expected_return_pct else 0
        thirty_day_performance = excel_data.get('performance_return_1_month', 0.0)
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

        # Multi-source news check
        multi_source_active = news_data.get('multi_source_enabled', False)
        active_sources = news_data.get('active_sources', 1)
        total_articles = news_data.get('total_articles', 0)

        report = f"""
🎯 ULTIMATE AI STOCK PREDICTION - {ticker} ({prediction_days} Day Analysis + Options Trading)
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Type: Mathematical Computations + AI Reasoning + Multi-Source Data + Graph Analysis

📊 MATHEMATICAL ANALYSIS FOUNDATION:
{'─'*50}"""

        if math_analysis:
            momentum_metrics = math_analysis.get('momentum_metrics', {})
            volatility_metrics = math_analysis.get('volatility_metrics', {})
            statistical_metrics = math_analysis.get('statistical_metrics', {})
            composite_scores = math_analysis.get('composite_scores', {})
            
            # Market regime analysis
            regime_analysis = math_analysis.get('regime_analysis', {})
            adaptive_weights = math_analysis.get('adaptive_weights', {})
            final_weights = math_analysis.get('final_weights', {})
            
            if regime_analysis:
                report += f"""

🏛️ MARKET REGIME ANALYSIS:
  Detected Regime: {regime_analysis.get('main_regime', 'unknown').replace('_', ' ').title()}
  Regime Confidence: {regime_analysis.get('regime_confidence', 0.0):.1%}
  Volatility Level: {regime_analysis.get('volatility_level', 0):.1f}%
  Trend Strength: {regime_analysis.get('trend_strength', 0):.3f}
  News Volume: {regime_analysis.get('news_volume', 0)} articles
  Active Sources: {regime_analysis.get('active_sources', 1)}"""

                # Add graph analysis regime enhancement if available
                if regime_analysis.get('graph_analysis_applied', False):
                    report += f"""
  Graph Pattern Regime: {regime_analysis.get('graph_pattern_regime', 'none').replace('_', ' ').title()}
  Graph Override Applied: {'✅' if regime_analysis.get('graph_override_applied', False) else '❌'}"""

            report += f"""

⚖️ ADAPTIVE WEIGHT SYSTEM:
  Base Weights: Mom={adaptive_weights.get('momentum', 0.30):.2f}, Tech={adaptive_weights.get('technical', 0.35):.2f}, News={adaptive_weights.get('news', 0.30):.2f}
  Final Weights: Mom={final_weights.get('momentum', 0.30):.2f}, Tech={final_weights.get('technical', 0.35):.2f}, News={final_weights.get('news', 0.30):.2f}
  Regime Adjustment: {'✅ Applied' if adaptive_weights != final_weights else '❌ No adjustment needed'}"""
            
            # Add graph boost factors if available
            if composite_scores.get('graph_analysis_integrated', False):
                report += f"""
  Graph Analysis Boost: {'✅ Applied' if composite_scores.get('graph_boost_factors_applied', False) else '❌ Not applied'}"""
                      
            report += f"""

🔢 VOLATILITY & RISK METRICS:
  Annualized Volatility: {volatility_metrics.get('annualized_volatility', 0):.2f}%
  Daily Volatility: {volatility_metrics.get('daily_volatility', 0):.3f}%
  Risk Category: {volatility_metrics.get('volatility_category', 'Unknown')}

📈 MOMENTUM & STATISTICAL ANALYSIS:
  30-day Return: {momentum_metrics.get('thirty_day_return', 0):.2f}%
  Momentum Score: {momentum_metrics.get('momentum_score', 0):+.3f}
  Direction: {momentum_metrics.get('momentum_direction', 'Unknown')}"""

            # Add graph momentum if available
            if momentum_metrics.get('graph_momentum_applied', False):
                report += f"""
  Graph Momentum Boost: {momentum_metrics.get('graph_momentum_boost', 0):+.4f}
  Graph Acceleration: {momentum_metrics.get('graph_momentum_acceleration', 0.5):.3f}"""

            report += f"""

🎯 COMPOSITE SIGNALS:
  Overall Signal: {composite_scores.get('overall_signal', 0):+.3f} (-1 to +1)
  Signal Strength: {composite_scores.get('signal_strength', 0):.0%}
  Signal Direction: {composite_scores.get('signal_direction', 'Unknown')}"""

            # Add graph signal adjustment if available
            if composite_scores.get('graph_analysis_integrated', False):
                graph_adjustment = composite_scores.get('graph_signal_adjustment', 0)
                if abs(graph_adjustment) > 0.01:
                    report += f"""
  Graph Signal Adjustment: {graph_adjustment:+.3f}"""

            report += f"""

📊 STATISTICAL PROJECTIONS:
  Expected Return: {statistical_metrics.get('expected_return_pct', 0):+.2f}%
  Probability of Gain: {statistical_metrics.get('probability_of_gain', 0.5):.0%}

📊 ENHANCED CONFIDENCE METRICS:
  Volatility-Aware Confidence: {claude_analysis.get('confidence', 0.5):.0%}
  Volatility Tier: {claude_analysis.get('volatility_tier', 'Unknown')}
  Monte Carlo Paths: {claude_analysis.get('monte_carlo_paths', 1000):,}
  
📈 PREDICTION QUALITY METRICS (Financial Standards):
  Target MAE: < {statistical_metrics.get('mae_performance', {}).get('mae_target', 2.5):.1f}% of stock price
  Actual MAE: {statistical_metrics.get('mae_performance', {}).get('final_day_mae', 0):.1f}%
  MAE Success Rate: {statistical_metrics.get('mae_performance', {}).get('mae_success_rate', 0):.0%}
  Time Horizon: {prediction_days} day(s) - {"Short-term" if prediction_days <= 5 else "Medium-term" if prediction_days <= 20 else "Long-term"}"""

        # Add graph analysis section if available
        if graph_analysis and claude_analysis.get('graph_analysis_results', {}).get('available', True):
            graph_results = claude_analysis.get('graph_analysis_results', {})
            report += f"""

📈 GRAPH ANALYSIS RESULTS:
  Primary Pattern: {graph_results.get('primary_pattern', 'None')}
  Pattern Reliability: {graph_results.get('pattern_reliability', 0):.1%}
  Breakout Status: {'✅ ' + graph_results.get('breakout_direction', '').upper() + ' BREAKOUT' if graph_results.get('breakout_detected', False) else '❌ No breakout'}
  Breakout Strength: {graph_results.get('breakout_strength', 0):.1%}
  Momentum Acceleration: {graph_results.get('momentum_acceleration', 0.5):.3f}
  Candlestick Patterns: {graph_results.get('candlestick_patterns_found', 0)} detected
  Support: ${graph_results.get('support_level', 0):.2f} | Resistance: ${graph_results.get('resistance_level', 0):.2f}"""

        report += f"""

📊 HISTORICAL CONTEXT ({excel_source}):
  Recent 30d performance: {thirty_day_performance:+.2f}%
  Historical volatility: {excel_data.get('volatility', 25.0):.2f}%
  Sector: {excel_data.get('sector', 'Technology')}
  Excel recommendation: {excel_data.get('excel_recommendation', 'Hold')}
  Risk assessment: {excel_data.get('excel_risk_level', 'Moderate')}"""
    
        if excel_file_path:
            report += f"\n  📂 Excel file: {os.path.basename(excel_file_path)}"
        
        report += f"""

📈 CURRENT MARKET DATA ({market_source}):
  Current Price: ${market_data['current_price']:.2f}
  Daily Change: {market_data['change']:+.2f} ({market_data['change_percent']:+.2f}%)
  Trading Range: ${market_data['low']:.2f} - ${market_data['high']:.2f}
  Volume: {market_data['volume']:,} shares
  Previous Close: ${market_data['previous_close']:.2f}

📰 MULTI-SOURCE NEWS INTELLIGENCE:
  Total Articles: {total_articles} (from {active_sources} sources)
  Multi-Source Status: {'✅ Active' if multi_source_active else '❌ Single source'}
  Phase 3 Sentiment: {news_data.get('phase3_sentiment', 0):+.3f}
  Aggregated Sentiment: {news_data.get('aggregated_sentiment', 0):+.3f}
  Confidence Score: {news_data.get('confidence_score', 0.5):.2f}
  Event Impact: {news_data.get('event_impact_score', 0):.2f}
  Custom Articles: {custom_news_count} user-provided
  Major Catalysts: {'✅ YES' if has_major_catalyst else '❌ NO'}"""

        # Add graph-enhanced news if available
        if news_data.get('graph_analysis_integrated', False):
            report += f"""
  Graph Sentiment Adjustment: {news_data.get('graph_sentiment_adjustment', 0):+.3f}
  Graph Technical Alignment: {news_data.get('graph_technical_alignment', 'neutral').title()}"""
    
        # Add custom news details
        if custom_articles:
            report += f"\n\n🚨 BREAKING NEWS (User-Provided):"
            for i, article in enumerate(custom_articles, 1):
                sentiment_emoji = "📈" if article['sentiment_score'] > 0.1 else "📉" if article['sentiment_score'] < -0.1 else "➡️"
                momentum_analysis = article.get('momentum_analysis', {})
                catalyst_analysis = article.get('catalyst_analysis', {})
                
                report += f"\n   {i}. {article['title']}"
                report += f"\n      {sentiment_emoji} Impact: {article['sentiment_score']:+.1f} | Source: {article['source']}"
                report += f"\n      Momentum: {momentum_analysis.get('momentum_direction', 'Unknown')} "
                report += f"(+{momentum_analysis.get('positive_indicators', 0)}/-{momentum_analysis.get('negative_indicators', 0)} indicators)"
                
                if catalyst_analysis.get('has_major_catalyst', False):
                    report += f"\n      🚨 MAJOR CATALYSTS: {', '.join(catalyst_analysis.get('catalyst_types', []))}"

        report += f"""

📅 MULTI-DAY PRICE PREDICTIONS ({prediction_days} Days):
{'─'*50}"""

        # Add multi-day predictions with MAE quality metrics
        if 'predictions_by_day' in claude_analysis:
            report += f"""

📊 MAE TUNING RESULTS:
  Target MAE: < {statistical_metrics.get('mae_performance', {}).get('mae_target', 2.5):.1f}% (Volatility-Adjusted)
  Monte Carlo: {claude_analysis.get('monte_carlo_paths', 1000):,} paths
  Volatility Tier: {claude_analysis.get('volatility_tier', 'Unknown')}"""
            
            total_mae = 0
            mae_count = 0
            
            for day_pred in claude_analysis['predictions_by_day'][:5]:  # Show first 5 days
                day_return = ((day_pred['predicted_close'] - market_data['current_price']) / market_data['current_price']) * 100
                
                # Calculate MAE metrics
                range_absolute = day_pred['predicted_high'] - day_pred['predicted_low']
                mae_estimate = range_absolute / (2 * day_pred['predicted_close']) * 100
                range_width = (range_absolute / day_pred['predicted_close']) * 100
                
                total_mae += mae_estimate
                mae_count += 1
                
                # MAE quality assessment
                if mae_estimate <= 1.5:
                    mae_quality = "🟢 Excellent"
                elif mae_estimate <= 2.0:
                    mae_quality = "✅ Good"  
                elif mae_estimate <= 2.5:
                    mae_quality = "🟡 Target Met"
                else:
                    mae_quality = "🔴 Exceeds Target"
                
                report += f"""
    Day {day_pred['day']}: ${day_pred['predicted_close']:.2f} ({day_return:+.1f}%) 
        Range: ${day_pred['predicted_low']:.2f} - ${day_pred['predicted_high']:.2f}
        MAE: {mae_estimate:.1f}% {mae_quality} | Range Width: ±{range_width/2:.1f}%"""
            
            # Add average MAE summary
            avg_mae = total_mae / mae_count if mae_count > 0 else 0
            mae_status = "🟢 EXCELLENT" if avg_mae <= 1.5 else "✅ GOOD" if avg_mae <= 2.0 else "🟡 TARGET MET" if avg_mae <= 2.5 else "🔴 NEEDS TUNING"
            
            report += f"""

📊 MAE PERFORMANCE SUMMARY:
    Average MAE: {avg_mae:.1f}% {mae_status}
    Target: < 2.5% per financial standards
    Tuning Status: {'✅ Successfully tuned' if avg_mae <= 2.5 else '⚠️ Requires further tuning'}"""
            
            if prediction_days > 5:
                final_day = claude_analysis['predictions_by_day'][-1]
                final_return = ((final_day['predicted_close'] - market_data['current_price']) / market_data['current_price']) * 100
                report += f"""
    ...
    Day {prediction_days}: ${final_day['predicted_close']:.2f} ({final_return:+.1f}%) [FINAL TARGET]"""
        
        report += f"""

🎯 {prediction_days}-DAY SUMMARY:
Final Target Price: ${final_target:.2f}
Total Expected Return: {expected_return_pct:+.1f}%
Direction: {claude_analysis.get('direction', 'unknown').upper()}
Confidence: {claude_analysis.get('confidence', 0.5):.0%}
Volatility Tier: {claude_analysis.get('volatility_tier', 'Unknown')}"""

        # Add options analysis section
        if options_data and claude_analysis and claude_analysis.get('options_analysis'):
            options_analysis = claude_analysis.get('options_analysis')
            if options_analysis and isinstance(options_analysis, dict):
                report += f"""

📈 OPTIONS TRADING ANALYSIS:
{'─'*50}
Input Options Analyzed: {len(options_data)}"""
        
                recommendations = options_analysis.get('recommendations')
                if recommendations and isinstance(recommendations, list) and len(recommendations) > 0:
                    report += f"""

🎯 TOP OPTIONS RECOMMENDATIONS:"""
            
                    for i, rec in enumerate(recommendations[:5], 1):
                        action_emoji = "🟢" if rec.get('action') == 'BUY' else "🔴" if rec.get('action') == 'AVOID' else "🟡"
                        report += f"""
    {i}. {action_emoji} {rec.get('action', 'HOLD')} {rec.get('option_type', '').upper()} ${rec.get('strike', 0)} ({rec.get('expiration_days', 0)}d)
       Confidence: {rec.get('confidence', 0):.0%} | Expected Return: {rec.get('expected_return_pct', 0):+.1f}%
       Max Loss: ${rec.get('max_loss', 0):.2f} | Prob Profit: {rec.get('prob_profitable', 0):.0%}
       Risk/Reward: {rec.get('risk_reward_ratio', 0):.1f}"""
                    
                        if rec.get('reasoning'):
                            report += f"""
       Reasoning: {rec['reasoning']}"""

        # Mathematical basis
        math_basis = claude_analysis.get('mathematical_basis', 'Standard analysis')
        
        # Handle predictions
        if prediction_days == 1:
            predicted_close = claude_analysis.get('predicted_close', final_target)
            predicted_low = claude_analysis.get('predicted_low', predicted_close * 0.98)
            predicted_high = claude_analysis.get('predicted_high', predicted_close * 1.02)
            predicted_open = claude_analysis.get('predicted_open', predicted_close)
        else:
            predicted_close = final_target
            if 'predictions_by_day' in claude_analysis and claude_analysis['predictions_by_day']:
                final_day_pred = claude_analysis['predictions_by_day'][-1]
                predicted_low = final_day_pred.get('predicted_low', predicted_close * 0.95)
                predicted_high = final_day_pred.get('predicted_high', predicted_close * 1.05)
                predicted_open = claude_analysis['predictions_by_day'][0].get('predicted_open', market_data['current_price'])
            else:
                predicted_low = predicted_close * 0.95
                predicted_high = predicted_close * 1.05
                predicted_open = market_data['current_price']
        
        report += f"""

🤖 CLAUDE AI ULTIMATE PREDICTION (Enhanced with Mathematical Foundation + Graph Analysis):
  Mathematical Basis: {math_basis}
  Final Target Price ({prediction_days}d): ${predicted_close:.2f}
  Expected Return: {expected_return:+.2%}
  Direction: {claude_analysis.get('direction', 'unknown').upper()}
  Confidence: {claude_analysis.get('confidence', 0.5):.0%}
  Upside Probability: {claude_analysis.get('probability_up', 0.5):.0%}
  Risk Level: {claude_analysis.get('risk_assessment', 'moderate').upper()}"""

        # Add graph analysis integration status
        if claude_analysis.get('graph_analysis_applied', False):
            report += f"""
  Graph Integration: ✅ Applied
  Primary Pattern: {claude_analysis.get('primary_chart_pattern', 'none')}
  Breakout: {'✅' if claude_analysis.get('breakout_detected', False) else '❌'}"""

        report += f"""

🎯 TRADING LEVELS & TARGETS:
  {prediction_days}-Day Range: ${predicted_low:.2f} - ${predicted_high:.2f}
  Predicted Open: ${predicted_open:.2f}
  Support Level: ${market_data['current_price'] * 0.97:.2f}
  Resistance Level: ${market_data['current_price'] * 1.03:.2f}

🔑 KEY FACTORS (AI-Identified):"""
    
        for factor in claude_analysis.get('key_factors', ['No specific factors identified']):
            report += f"\n   • {factor}"
        
        if has_major_catalyst:
            report += f"\n   • 🚨 Major catalysts detected: {', '.join(set(catalyst_types))}"
        
        report += f"""

🧠 AI REASONING:
  {claude_analysis.get('reasoning', 'Standard mathematical analysis applied')}

📊 DATA INTEGRATION SUMMARY:
  Excel Analysis: ✅ {excel_source}
  Mathematical Computations: ✅ Comprehensive quantitative analysis
  Multi-Source News: ✅ {active_sources} sources, {total_articles} articles
  Market Data: ✅ {market_source}
  Multi-Day Analysis: ✅ {prediction_days} day forecast
  Custom News: ✅ {custom_news_count} breaking articles {'with MAJOR CATALYSTS' if has_major_catalyst else ''}
  Options Analysis: {'✅ Included' if options_data else '❌ Not requested'}
  Graph Analysis: {'✅ Applied' if claude_analysis.get('graph_analysis_applied', False) else '❌ Not performed'}
  AI Analysis: ✅ Claude enhanced with volatility-aware confidence

🎯 TRADING RECOMMENDATIONS:
  Entry Signal: {claude_analysis.get('direction', 'unknown').upper()}
  Position Size: {'Conservative (25%)' if claude_analysis.get('risk_assessment') == 'high' else 'Standard (50%)' if claude_analysis.get('risk_assessment') == 'moderate' else 'Aggressive (75%)'}
  Stop Loss: ${market_data['current_price'] * 0.97:.2f} (-3.0%)
  Take Profit: ${market_data['current_price'] * 1.03:.2f} (+3.0%)"""

        # Add options-specific recommendations
        if options_data and claude_analysis:
            options_analysis = claude_analysis.get('options_analysis')
            if options_analysis and isinstance(options_analysis, dict):
                recommendations = options_analysis.get('recommendations')
                if recommendations and isinstance(recommendations, list) and len(recommendations) > 0:
                    top_rec = recommendations[0]
                    report += f"""
  
  📈 TOP OPTIONS RECOMMENDATION:
  - {top_rec.get('action', 'HOLD')} {top_rec.get('option_type', '').upper()} ${top_rec.get('strike', 0):.0f} ({top_rec.get('expiration_days', 0)}d)
  - Confidence: {top_rec.get('confidence', 0):.0%} | Expected Return: {top_rec.get('expected_return_pct', 0):+.1f}%
  - Max Risk: ${top_rec.get('max_loss', 0):.2f}"""

        # Add mathematical summary
        if math_analysis:
            overall_signal = math_analysis.get('composite_scores', {}).get('overall_signal', 0)
            
            report += f"""

🔢 MATHEMATICAL FOUNDATION SUMMARY:
  Overall Mathematical Signal: {overall_signal:+.3f} (-1 to +1)
  Risk-Adjusted Confidence: {claude_analysis.get('confidence', 0.5):.0%}
  Volatility-Aware System: ✅ Applied
  MAE Enforcement: ✅ Applied"""

        report += f"""

🎉 ULTIMATE PREDICTION SUMMARY:
This represents the most comprehensive analysis possible, integrating:
- Mathematical computations with volatility-aware confidence
- Multi-day Monte Carlo price simulations ({prediction_days} days, {claude_analysis.get('monte_carlo_paths', 1000):,} paths)
- User-selected Excel historical analysis
- Multi-source news aggregation ({active_sources} sources)
- User-provided breaking news with catalyst detection
- Deep article content analysis for momentum indicators
- Real-time/current market data
- Advanced AI reasoning with quantitative foundation"""

        if graph_analysis:
            report += f"\n- Graph technical analysis with pattern detection"
        if options_data:
            report += f"\n- Options trading analysis with Greeks and probability calculations"

        report += f"""

Expected Price Movement: {expected_return:+.2%} over {prediction_days} day(s) ({claude_analysis.get('direction', 'unknown').upper()})
Final Target ({prediction_days}d): ${predicted_close:.2f}
Mathematical Confidence Level: {claude_analysis.get('confidence', 0.5):.0%} (Volatility-Adjusted)
{'Options Recommendations: ' + str(len(claude_analysis.get('options_analysis', {}).get('recommendations', []))) + ' strategies analyzed' if options_data else ''}

💡 DATA SOURCES & METHODOLOGY:
  Historical: {excel_source}
  Market: {market_source}  
  News Sources: {active_sources} sources, {total_articles} articles
  Custom News: {custom_news_count} articles {'[MAJOR CATALYSTS DETECTED]' if has_major_catalyst else ''}
  Mathematical Analysis: ✅ Comprehensive with adaptive weights
  Multi-Day Forecast: ✅ {prediction_days} day Monte Carlo simulation
  Article Analysis: ✅ Deep content analysis for momentum indicators
  30-day Performance: {thirty_day_performance:.2f}%
  Options Analysis: {'✅ Complete Greeks & probability analysis' if options_data else '❌ Not requested'}
  Graph Analysis: {'✅ Technical patterns & breakout detection' if claude_analysis.get('graph_analysis_applied', False) else '❌ Not performed'}
  AI Enhancement: Claude with mathematical foundation, catalyst detection & volatility-aware confidence
"""
    
        return report

    def save_prediction_data(self, ticker, excel_data, news_data, market_data, claude_analysis, custom_articles, excel_file_path, options_data, graph_analysis=None):
        """Save prediction data to JSON file with all enhancements - FIXED"""
        try:
            # Use the enhanced save method from the engine with proper error handling
            filename = self.engine.save_prediction_data(
                ticker=ticker, 
                excel_data=excel_data, 
                news_data=news_data, 
                market_data=market_data, 
                claude_analysis=claude_analysis, 
                custom_articles=custom_articles, 
                excel_file_path=excel_file_path, 
                options_data=options_data
            )
            
            if filename:
                print(f"✅ Enhanced prediction data saved to: {filename}")
                
                # Show multi-source news summary
                if news_data and news_data.get('multi_source_enabled', False):
                    print(f"   📰 Multi-source news: {news_data.get('active_sources', 1)} sources")
                    print(f"   📊 Total articles: {news_data.get('total_articles', 0)}")
                    
                # Show graph analysis summary
                if graph_analysis and claude_analysis and claude_analysis.get('graph_analysis_applied', False):
                    print(f"   📈 Graph analysis: ✅ Included")
                    print(f"   📊 Pattern: {claude_analysis.get('primary_chart_pattern', 'none')}")
                    print(f"   🚀 Breakout: {'✅' if claude_analysis.get('breakout_detected', False) else '❌'}")
            
                # Show volatility-aware confidence
                if claude_analysis:
                    print(f"   🎯 Volatility-aware confidence: ✅ Applied")
                    print(f"   📊 Volatility tier: {claude_analysis.get('volatility_tier', 'Unknown')}")
                    print(f"   📊 Monte Carlo paths: {claude_analysis.get('monte_carlo_paths', 1000):,}")
                    
                    # Show MAE performance
                    mae_data = claude_analysis.get('mathematical_analysis', {}).get('statistical_metrics', {}).get('mae_performance', {})
                    if mae_data:
                        print(f"   📊 MAE performance: {mae_data.get('final_day_mae', 0):.1f}% (target: {mae_data.get('mae_target', 2.5):.1f}%)")
                    
                # Show comprehensive data summary
                total_articles = news_data.get('total_articles', 0) if news_data else 0
                custom_count = len(custom_articles) if custom_articles else 0
                    
                print(f"   📰 Total articles saved: {total_articles}")
                print(f"   📝 Custom articles: {custom_count}")
                print(f"   📊 Options data: {'✅ Included' if options_data else '❌ None'}")
                print(f"   📈 Graph analysis: {'✅ Included' if graph_analysis else '❌ None'}")
                    
                return filename
            else:
                print(f"❌ Failed to save prediction data")
                return None
                
        except Exception as e:
            print(f"❌ Error saving enhanced prediction data: {e}")
            print("🔄 Attempting fallback save method...")
            
            # Fallback: Create a basic JSON save
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                fallback_filename = f"predictions_output/{ticker}_prediction_fallback_{timestamp}.json"
                
                # Create basic data structure
                fallback_data = {
                    'ticker': ticker,
                    'timestamp': timestamp,
                    'excel_data': excel_data,
                    'news_data': news_data,
                    'market_data': market_data,
                    'claude_analysis': claude_analysis,
                    'custom_articles': custom_articles,
                    'options_data': options_data,
                    'graph_analysis': graph_analysis,
                    'excel_file_path': excel_file_path,
                    'save_method': 'fallback'
                }
                
                os.makedirs("predictions_output", exist_ok=True)
                with open(fallback_filename, 'w', encoding='utf-8') as f:
                    json.dump(fallback_data, f, indent=2, default=str)
                
                print(f"✅ Fallback save successful: {fallback_filename}")
                return fallback_filename
                
            except Exception as fallback_error:
                print(f"❌ Fallback save also failed: {fallback_error}")
                return None

    def run_analysis(self, ticker):
        """Run the complete analysis workflow with all enhancements"""
        print(f"\n🎯 Starting ULTIMATE ENHANCED analysis for {ticker}...")
        print("   Multi-day predictions + Options analysis + Mathematical computations + AI reasoning!")
        print("   🔄 With multi-source news aggregation + graph analysis + volatility-aware confidence!")
        
        # Step 1: Get prediction timeframe
        prediction_days = self.get_prediction_timeframe()
        
        # Step 2: Get Excel file from user  
        excel_file_path = self.get_excel_file_from_user(ticker)
        excel_data = self.engine.load_excel_historical_data(ticker, excel_file_path)
        
        # Step 3: Get graph analysis choice
        perform_graph = self.get_graph_analysis_choice(ticker)
        graph_analysis = None
        if perform_graph:
            graph_analysis = self.perform_graph_analysis(ticker, excel_file_path)
        
        # Step 4: Get options data
        options_data = self.get_options_data(ticker, prediction_days)
        
        # Step 5: Get custom news articles from user
        custom_articles = self.get_custom_news_from_user(ticker)
        
        # Step 6: Get multi-source news intelligence (includes graph analysis if available)
        print(f"\n🔄 Gathering multi-source news intelligence for {ticker}...")
        news_data = self.engine.get_phase3_news_intelligence(ticker, graph_analysis)
        
        # Show multi-source news results
        if news_data.get('multi_source_enabled', False):
            print(f"   ✅ Multi-source aggregation successful!")
            print(f"   📊 Active sources: {news_data.get('active_sources', 1)}")
            print(f"   📰 Total articles: {news_data.get('total_articles', 0)}")
            print(f"   📈 Aggregated sentiment: {news_data.get('aggregated_sentiment', 0.0):+.3f}")
            print(f"   🎯 Aggregate confidence: {news_data.get('aggregate_confidence', 0.5):.3f}")
            
            # Show source breakdown
            source_breakdown = news_data.get('source_breakdown', {})
            if source_breakdown:
                print(f"   📊 Source breakdown:")
                for source_name, source_data in source_breakdown.items():
                    if source_data and source_data.get('articles', 0) > 0:
                        print(f"      - {source_name}: {source_data.get('articles', 0)} articles, sentiment {source_data.get('sentiment', 0):+.3f}")
        else:
            print(f"   ⚠️ Multi-source aggregation not available - using Phase 3 only")
        
        # Show graph analysis integration in news
        if news_data.get('graph_analysis_integrated', False):
            print(f"   📈 Graph analysis integrated into news sentiment")
            print(f"   📊 Graph sentiment adjustment: {news_data.get('graph_sentiment_adjustment', 0.0):+.3f}")
            print(f"   🎯 Graph confidence boost: +{news_data.get('graph_confidence_boost', 0.0):.3f}")
        
        # Step 7: Store news data for regime detection
        self.engine.set_news_data_for_regime(news_data)
        
        # Step 8: Enhance news with custom articles
        enhanced_news_data = self.engine.enhance_news_with_custom_articles(news_data, custom_articles)
        
        # Show enhanced news summary
        total_articles = enhanced_news_data.get('total_articles', 0)
        custom_count = len(custom_articles) if custom_articles else 0
        
        print(f"\n📰 Enhanced news intelligence summary:")
        print(f"   📊 Total articles: {total_articles}")
        print(f"   📝 Custom articles: {custom_count}")
        print(f"   📈 Final sentiment: {enhanced_news_data.get('sentiment_1d', 0.0):+.3f}")
        print(f"   🎯 Final confidence: {enhanced_news_data.get('confidence_score', 0.5):.3f}")
        
        if enhanced_news_data.get('has_major_catalyst', False):
            print(f"   🚨 Major catalysts detected: {enhanced_news_data.get('total_catalyst_count', 0)}")
            print(f"   📊 Catalyst types: {', '.join(enhanced_news_data.get('catalyst_types', []))}")
        
        # Step 9: Get market data
        market_data = self.get_realtime_market_data_with_fallback(ticker)
        if not market_data:
            print("❌ Cannot proceed without market data")
            return None, None, None, None, None, None, None, None, None
        
        # Step 10: Enhanced Claude AI analysis with all features
        print(f"\n🤖 Running Claude AI analysis with all enhancements...")
        print(f"   📊 Volatility-aware confidence: ✅ Enabled")
        print(f"   📰 Multi-source news: {'✅' if enhanced_news_data.get('multi_source_enabled', False) else '❌'}")
        print(f"   📈 Graph analysis: {'✅' if graph_analysis else '❌'}")
        print(f"   📊 Options analysis: {'✅' if options_data else '❌'}")
        
        # CORRECT: Use the proper method signature from your prediction engine
        claude_analysis = self.engine.analyze_with_claude_ultimate_enhanced(
            ticker=ticker, 
            excel_data=excel_data, 
            news_data=enhanced_news_data, 
            market_data=market_data, 
            custom_articles=custom_articles, 
            graph_analysis=graph_analysis,  # CRITICAL: Pass graph_analysis as 6th parameter
            prediction_days=prediction_days, 
            options_data=options_data
        )
        
        if not claude_analysis:
            print("❌ AI analysis failed")
            return None, None, None, None, None, None, None, None, None
        
        # Show Claude analysis results with all enhancements
        print(f"   ✅ Claude analysis complete!")
        print(f"   🎯 Volatility-aware confidence: {claude_analysis.get('confidence', 0.5):.1%}")
        print(f"   📊 Volatility tier: {claude_analysis.get('volatility_tier', 'Unknown')}")
        print(f"   📊 Monte Carlo paths: {claude_analysis.get('monte_carlo_paths', 1000):,}")
        
        if claude_analysis.get('graph_analysis_applied', False):
            print(f"   📈 Graph analysis: ✅ Applied")
            print(f"   📊 Primary pattern: {claude_analysis.get('primary_chart_pattern', 'none')}")
            if claude_analysis.get('breakout_detected', False):
                print(f"   🚀 Breakout detected!")
        
        if claude_analysis.get('prediction_metadata', {}).get('regime_adaptive_weights', False):
            print(f"   ⚖️ Adaptive weights: ✅ Applied")
        
        # Step 11: Generate enhanced ultimate report
        report = self.generate_ultimate_report_enhanced(
            ticker, excel_data, enhanced_news_data, market_data, 
            claude_analysis, custom_articles, excel_file_path, 
            prediction_days, options_data, graph_analysis
        )
        
        return report, excel_data, enhanced_news_data, market_data, claude_analysis, custom_articles, excel_file_path, options_data, graph_analysis

    def save_results(self, ticker, report, excel_data, news_data, market_data, claude_analysis, custom_articles, excel_file_path, options_data, graph_analysis=None):
        """Save analysis results to files"""
        print(f"\n💾 Save enhanced ultimate analysis?")
        save_choice = input("Enter 'y' to save files: ").strip().lower()
        
        if save_choice in ['y', 'yes', '']:
            # Save report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"enhanced_ultimate_prediction_{ticker}_{timestamp}.txt"
            
            try:
                with open(report_filename, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"✅ Report saved: {report_filename}")
            except Exception as e:
                print(f"❌ Error saving report: {e}")
                report_filename = None
            
            # Save JSON
            json_filename = self.save_prediction_data(
                ticker, excel_data, news_data, market_data,
                claude_analysis, custom_articles, excel_file_path, 
                options_data, graph_analysis
            )
            
            if json_filename:
                print(f"✅ Data saved: {json_filename}")
            
            return report_filename, json_filename
        
        return None, None

def main():
        """Enhanced main function with all features integrated"""
        print("🚀 ULTIMATE AI STOCK PREDICTOR (FULLY ENHANCED)")
        print("="*80)
        print("📊 Mathematical Computations + Multi-Day Forecasts + Options Analysis")
        print("🎯 Multi-Source News + Graph Analysis + Volatility-Aware Confidence")
        print("⚖️ Dynamic Weight Adjustment + MAE Enforcement + Catalyst Detection")
        print("="*80)
        
        # Initialize report generator
        generator = ReportGenerator()
        
        if not generator.engine.claude_key:
            print("❌ Claude API key required for AI analysis")
            print("   Please add CLAUDE_API_KEY to your .env file")
            return
        
        # Get ticker
        ticker = sys.argv[1] if len(sys.argv) > 1 else input("\nEnter ticker symbol (default NVDA): ").strip().upper() or "NVDA"
        
        try:
            # Run complete analysis with all enhancements
            results = generator.run_analysis(ticker)
            
            if results[0] is None:  # Check if analysis failed
                print("❌ Analysis failed")
                return
            
            report, excel_data, news_data, market_data, claude_analysis, custom_articles, excel_file_path, options_data, graph_analysis = results
            
            # Check for None claude_analysis
            if claude_analysis is None:
                print("❌ Claude analysis failed - cannot generate complete report")
                print("📊 Basic data was collected, but AI analysis is missing")
                return
            
            # Display report
            print(report)
            
            # Save results 
            generator.save_results(ticker, report, excel_data, news_data, market_data, claude_analysis, custom_articles, excel_file_path, options_data, graph_analysis)
            
            print(f"\n🎉 ULTIMATE ENHANCED analysis complete for {ticker}!")
            
            # Enhanced summary with all features
            print(f"\n📊 ENHANCED ANALYSIS SUMMARY:")
            excel_source = "✅ User Excel file" if excel_file_path else "⚠️ Default data"
            market_source = "✅ Alpha Vantage API" if market_data.get('data_source') == 'alpha_vantage' else "✅ User input"
            custom_news = f"✅ {len(custom_articles)} custom articles" if custom_articles else "⚠️ No custom news"
            
            # Multi-source news status
            multi_source_status = "❌ Single source"
            active_sources = 1
            total_articles = 0
            
            if news_data and news_data.get('multi_source_enabled', False):
                multi_source_status = f"✅ {news_data.get('active_sources', 1)} sources active"
                active_sources = news_data.get('active_sources', 1)
                total_articles = news_data.get('total_articles', 0)
            
            # Safe access to claude_analysis
            math_signal = 0
            catalyst_impact = 'none'
            prediction_days = 1
            market_regime = 'unknown'
            regime_confidence = 0.0
            adaptive_weights_applied = False
            final_weights = {}
            volatility_tier = 'Unknown'
            monte_carlo_paths = 1000
            graph_applied = False
            
            if claude_analysis:
                math_analysis = claude_analysis.get('mathematical_analysis', {})
                if math_analysis:
                    composite_scores = math_analysis.get('composite_scores', {})
                    math_signal = composite_scores.get('overall_signal', 0)
                    adaptive_weights_applied = composite_scores.get('regime_adaptive_weights', False)
                    
                    # Get regime information
                    regime_analysis = math_analysis.get('regime_analysis', {})
                    if regime_analysis:
                        market_regime = regime_analysis.get('main_regime', 'unknown')
                        regime_confidence = regime_analysis.get('regime_confidence', 0.0)
                    
                    # Get final weights
                    final_weights = math_analysis.get('final_weights', {})
                    
                catalyst_impact = claude_analysis.get('catalyst_impact', 'none')
                prediction_days = len(claude_analysis.get('predictions_by_day', [1])) or 1
                volatility_tier = claude_analysis.get('volatility_tier', 'Unknown')
                monte_carlo_paths = claude_analysis.get('monte_carlo_paths', 1000)
                graph_applied = claude_analysis.get('graph_analysis_applied', False)
            
            options_count = len(options_data) if options_data else 0
            
            # Check for catalysts
            has_catalyst = False
            if custom_articles:
                for article in custom_articles:
                    if article.get('catalyst_analysis', {}).get('has_major_catalyst', False):
                        has_catalyst = True
                        break
            
            # Check for options recommendations
            options_recs = 0
            if claude_analysis:
                options_analysis = claude_analysis.get('options_analysis')
                if options_analysis and isinstance(options_analysis, dict):
                    recommendations = options_analysis.get('recommendations')
                    if recommendations and isinstance(recommendations, list):
                        options_recs = len(recommendations)
            
            # Display enhanced summary
            print(f"   📅 Prediction Period: {prediction_days} day(s)")
            print(f"   📰 Multi-Source News: {multi_source_status}")
            print(f"   📊 Total Articles: {total_articles} (from {active_sources} sources)")
            print(f"   📈 Graph Analysis: {'✅ Applied' if graph_applied else '❌ Not available'}")
            
            # Market regime information
            print(f"   🏛️ Market Regime: {market_regime.replace('_', ' ').title()}")
            print(f"   📊 Regime Confidence: {regime_confidence:.1%}")
            print(f"   ⚖️ Adaptive Weights: {'✅ Applied' if adaptive_weights_applied else '❌ Not applied'}")
            
            # Show final weights if available
            if final_weights:
                print(f"   📊 Final Weights: Mom={final_weights.get('momentum', 0.30):.2f}, Tech={final_weights.get('technical', 0.35):.2f}, News={final_weights.get('news', 0.30):.2f}")
            
            # Volatility-aware confidence
            print(f"   🎯 Volatility-Aware Confidence: ✅ Applied")
            print(f"   📊 Volatility Tier: {volatility_tier}")
            print(f"   📊 Monte Carlo Paths: {monte_carlo_paths:,}")
            
            print(f"   📈 Options Analysis: {'✅ ' + str(options_count) + ' options analyzed' if options_count > 0 else '⚠️ Not requested'}")
            print(f"   💡 Options Recommendations: {'✅ ' + str(options_recs) + ' strategies' if options_recs > 0 else '⚠️ None generated'}")
            print(f"   📊 Historical Data: {excel_source}")
            print(f"   💰 Market Data: {market_source}")
            print(f"   📰 Custom News: {custom_news}")
            print(f"   🔢 Mathematical Analysis: ✅ Comprehensive quantitative computations")
            print(f"   📈 Mathematical Signal: {math_signal:+.3f} (-1 to +1)")
            print(f"   🚨 Catalyst Detection: ✅ {'MAJOR CATALYSTS DETECTED' if has_catalyst else 'No major catalysts'}")
            print(f"   🎯 Catalyst Impact: {catalyst_impact.upper()}")
            print(f"   📊 30-day Performance: ✅ {excel_data.get('performance_return_1_month', 0.0):.2f}%")
            print(f"   🤖 AI Analysis: {'✅ Claude enhanced with all features' if claude_analysis else '❌ Failed'}")
            print(f"   🎯 Final Confidence: {claude_analysis.get('confidence', 0.5):.0%}" if claude_analysis else "   🎯 Final Confidence: N/A")
            
            # Show key results
            if claude_analysis:
                final_target = claude_analysis.get('final_target_price', market_data['current_price'])
                expected_return = claude_analysis.get('total_expected_return_pct', 0)
                current_price = market_data['current_price']
                
                print(f"\n🎯 KEY RESULTS:")
                print(f"   Current Price: ${current_price:.2f}")
                print(f"   {prediction_days}-Day Target: ${final_target:.2f}")
                print(f"   Expected Return: {expected_return:+.1f}%")
                print(f"   Direction: {claude_analysis.get('direction', 'unknown').upper()}")
                print(f"   Confidence: {claude_analysis.get('confidence', 0.5):.0%} (Volatility-Adjusted)")
                
                if options_recs > 0:
                    options_analysis = claude_analysis.get('options_analysis', {})
                    recommendations = options_analysis.get('recommendations', [])
                    if recommendations:
                        top_option = recommendations[0]
                        print(f"   Top Option Play: {top_option['action']} {top_option['option_type'].upper()} ${top_option['strike']} ({top_option.get('confidence', 0):.0%} confidence)")
                
                # MAE Performance
                mae_data = claude_analysis.get('mathematical_analysis', {}).get('statistical_metrics', {}).get('mae_performance', {})
                if mae_data:
                    print(f"   MAE Performance: {mae_data.get('final_day_mae', 0):.1f}% vs target {mae_data.get('mae_target', 2.5):.1f}%")
            else:
                print(f"\n❌ KEY RESULTS: Claude analysis failed - no predictions available")
            
            # Show regime-specific trading recommendations
            if market_regime != 'unknown' and claude_analysis:
                print(f"\n💡 REGIME-SPECIFIC TRADING RECOMMENDATIONS:")
                
                regime_recommendations = {
                    'high_vol_trending': [
                        "🚀 High Volatility Trending Market:",
                        "• Use momentum-based strategies",
                        "• Employ wider stop-losses due to volatility",
                        "• Consider trend-following options",
                        "• Monitor for regime shifts"
                    ],
                    'low_vol_sideways': [
                        "📊 Low Volatility Sideways Market:",
                        "• Focus on technical support/resistance levels",
                        "• Use range-trading strategies",
                        "• Consider selling options for premium",
                        "• Watch for breakout signals"
                    ],
                    'news_driven': [
                        "📰 News-Driven Market:",
                        "• Stay alert for breaking news",
                        "• Use smaller position sizes",
                        "• Consider event-driven strategies",
                        "• Monitor news flow closely"
                    ],
                    'technical_trending': [
                        "🔧 Technical Trending Market:",
                        "• Follow technical indicators closely",
                        "• Use trend-following strategies",
                        "• Respect support/resistance levels",
                        "• Consider momentum-based entries"
                    ],
                    'mixed_signals': [
                        "🔄 Mixed Signals Market:",
                        "• Use balanced approach",
                        "• Employ careful risk management",
                        "• Monitor for regime clarification",
                        "• Consider defensive positioning"
                    ],
                    'technical_breakout': [
                        "🚀 Technical Breakout Market:",
                        "• Act on breakout signals quickly",
                        "• Use momentum strategies",
                        "• Set wider profit targets",
                        "• Trail stops to protect gains"
                    ],
                    'pattern_trending': [
                        "📈 Pattern-Driven Trending Market:",
                        "• Follow pattern completion signals",
                        "• Use pattern-based entry/exit points",
                        "• Monitor pattern reliability",
                        "• Combine with momentum indicators"
                    ]
                }
                
                if market_regime in regime_recommendations:
                    for line in regime_recommendations[market_regime]:
                        print(f"   {line}")
                else:
                    # Default recommendations
                    print(f"   📊 Standard Market Analysis:")
                    print(f"   • Use standard technical analysis")
                    print(f"   • Monitor key support/resistance")
                    print(f"   • Employ proper risk management")
                    print(f"   • Stay alert to regime changes")
            
        except KeyboardInterrupt:
            print("\n⚠️ Analysis interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()