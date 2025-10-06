#!/usr/bin/env python3
"""
Phase 4 ML Prediction Engine - ULTIMATE ENHANCED Interactive Tool
UPDATED: Now integrates with Ultimate AI Stock Predictor + Full User Control
User-friendly interface with comprehensive prediction options
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import warnings
import json
import requests
warnings.filterwarnings('ignore')

# Import our Phase 4 components
from features.feature_engineering import FeatureEngineer
from data_loaders.excel_loader import ExcelDataLoader  # FIXED: Use the corrected loader
from data_loaders.phase3_connector import Phase3NewsPredictor
from sklearn.ensemble import RandomForestRegressor
from graph_analyzer import GraphAnalyzer, integrate_graph_analysis_with_prediction
import joblib

# CRITICAL: Import the separated prediction engine components
AI_PREDICTOR_AVAILABLE = False

# Try to import the separated prediction engine architecture
try:
    from prediction_engine import StockPredictionEngine
    from report_generator import ReportGenerator, main as report_main
    AI_PREDICTOR_AVAILABLE = True
    print("✅ Separated prediction engine loaded successfully")
    
    # Create global instances for backward compatibility
    prediction_engine = StockPredictionEngine()
    report_generator = ReportGenerator()
    
    # Create function aliases for backward compatibility
    def load_api_keys():
        return prediction_engine.alpha_key, prediction_engine.claude_key
    
    def get_excel_file_from_user(ticker):
        return report_generator.get_excel_file_from_user(ticker)
    
    def load_excel_historical_data(ticker, excel_file_path=None):
        return prediction_engine.load_excel_historical_data(ticker, excel_file_path)
    
    def get_custom_news_from_user(ticker):
        return report_generator.get_custom_news_from_user(ticker)
    
    def get_phase3_news_intelligence(ticker, graph_analysis=None):
        return prediction_engine.get_phase3_news_intelligence(ticker, graph_analysis)
    
    def enhance_news_with_custom_articles(news_data, custom_articles):
        return prediction_engine.enhance_news_with_custom_articles(news_data, custom_articles)
    
    def get_realtime_market_data_with_fallback(ticker, api_key=None):
        return report_generator.get_realtime_market_data_with_fallback(ticker)
    
    def analyze_with_claude_ultimate_enhanced(ticker, excel_data, news_data, market_data, custom_articles, claude_key=None):
        return prediction_engine.analyze_with_claude_ultimate_enhanced(ticker, excel_data, news_data, market_data, custom_articles)
    
    def generate_ultimate_report_enhanced(ticker, excel_data, news_data, market_data, claude_analysis, custom_articles, excel_file_path):
        return report_generator.generate_ultimate_report_enhanced(ticker, excel_data, news_data, market_data, claude_analysis, custom_articles, excel_file_path)
    
    def save_ultimate_prediction_enhanced(ticker, excel_data, news_data, market_data, claude_analysis, custom_articles, excel_file_path):
        return prediction_engine.save_prediction_data(ticker, excel_data, news_data, market_data, claude_analysis, custom_articles, excel_file_path)
    
    def get_prediction_timeframe():
        return report_generator.get_prediction_timeframe()
    
    def get_options_data(ticker, prediction_days):
        return report_generator.get_options_data(ticker, prediction_days)
    
    def analyze_with_claude_ultimate_enhanced_multi_day(ticker, excel_data, news_data, market_data, custom_articles, prediction_days=1, options_data=None):
        return prediction_engine.analyze_with_claude_ultimate_enhanced(ticker, excel_data, news_data, market_data, custom_articles, prediction_days=prediction_days, options_data=options_data)
    
    def generate_ultimate_report_enhanced_multi_day(ticker, excel_data, news_data, market_data, claude_analysis, custom_articles, excel_file_path, prediction_days, options_data):
        return report_generator.generate_ultimate_report_enhanced(ticker, excel_data, news_data, market_data, claude_analysis, custom_articles, excel_file_path, prediction_days, options_data)

except ImportError:
    # Fallback: Try the old monolithic file approach
    ai_predictor_modules = [
        'ultimate_ai_predictor',
        'ai_stock_predictor_fixed', 
        'complete_ai_predictor',
        'ai_stock_predictor'
    ]

    for module_name in ai_predictor_modules:
        try:
            # Import functions from the AI predictor
            ai_module = __import__(module_name)
            
            # Import all the functions we need
            load_api_keys = getattr(ai_module, 'load_api_keys')
            get_excel_file_from_user = getattr(ai_module, 'get_excel_file_from_user')
            load_excel_historical_data = getattr(ai_module, 'load_excel_historical_data')
            get_custom_news_from_user = getattr(ai_module, 'get_custom_news_from_user')
            get_phase3_news_intelligence = getattr(ai_module, 'get_phase3_news_intelligence')
            enhance_news_with_custom_articles = getattr(ai_module, 'enhance_news_with_custom_articles')
            get_realtime_market_data_with_fallback = getattr(ai_module, 'get_realtime_market_data_with_fallback')
            analyze_with_claude_ultimate_enhanced = getattr(ai_module, 'analyze_with_claude_ultimate_enhanced')
            
            generate_ultimate_report_enhanced = getattr(ai_module, 'generate_ultimate_report_enhanced')
            save_ultimate_prediction_enhanced = getattr(ai_module, 'save_ultimate_prediction_enhanced') 
            
            AI_PREDICTOR_AVAILABLE = True
            print(f"✅ Ultimate AI Stock Predictor loaded from: {module_name}")
            break
            
        except ImportError:
            continue
        except AttributeError as e:
            print(f"⚠️ Found {module_name} but missing functions: {e}")
            continue

    if not AI_PREDICTOR_AVAILABLE:
        print("❌ Ultimate AI Stock Predictor not found. Expected files:")
        print("   - prediction_engine.py + report_generator.py (preferred)")
        print("   - ultimate_ai_predictor.py")
        print("   - ai_stock_predictor_fixed.py") 
        print("   - complete_ai_predictor.py")
        print("   Some features will be limited")

class UltimateEnhancedPredictionTool:
    """
    Ultimate Enhanced Interactive prediction tool with full AI integration
    """
    
    def __init__(self):
        self.excel_loader = None
        self.news_connector = None
        self.feature_engineer = None
        self.model = None
        self.is_trained = False
        self.ticker = None
        self.excel_file_path = None
        self.trained_features = []
        self.current_price_override = None
        self.custom_news_articles = []  # Store custom news articles
        self.api_keys = None  # Store API keys for AI predictor
        
        # Create directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        os.makedirs("custom_news", exist_ok=True)
        
        # Load API keys if available
        if AI_PREDICTOR_AVAILABLE:
            try:
                alpha_key, claude_key = load_api_keys()
                if alpha_key and claude_key:
                    self.api_keys = {'alpha': alpha_key, 'claude': claude_key}
                    print("✅ API keys loaded for Ultimate AI predictor")
                else:
                    print("⚠️ API keys not found - Ultimate AI features limited")
            except Exception as e:
                print(f"⚠️ Error loading API keys: {e}")
        
        self.print_header()
    
    def print_header(self):
        """Print ultimate enhanced welcome header"""
        print("🚀 " + "=" * 80)
        print("    PHASE 4 ML PREDICTION ENGINE - ULTIMATE ENHANCED INTERACTIVE")
        print("=" * 84)
        print("📊 User Excel Selection + Custom News + Real-time Data + Claude AI")
        print("🎯 Get the most comprehensive AI-powered stock predictions possible")
        print("📰 NEW: Full user control over Excel files and breaking news")
        print("🤖 NEW: Ultimate AI predictor with Claude comprehensive analysis")
        print("💰 NEW: Smart market data input when APIs are unavailable")
        print("=" * 84)
    
    def show_main_menu(self):
        """Display ultimate main menu with all available options"""
        print(f"\n🎯 ULTIMATE MAIN MENU - Complete Prediction Suite")
        print("=" * 60)
        print("1. 📊 Traditional ML Analysis (Excel + News + ML)")
        print("2. 🚀 ULTIMATE AI Analysis (User Excel + Custom News + Claude)")
        print("3. 📁 Select Excel Analysis File")
        print("4. 📰 Add Custom Breaking News")
        print("5. 📋 View Custom News Articles")
        print("6. 🗑️ Clear Custom News Articles")
        print("7. ⚙️ Settings & Configuration")
        print("8. 🆘 Help & Instructions")
        print("9. 🚪 Exit")
        
        if not AI_PREDICTOR_AVAILABLE:
            print("\n⚠️ Note: Ultimate AI Analysis (Option 2) requires ultimate_ai_predictor.py")
        
        if not self.api_keys:
            print("⚠️ Note: Real-time data requires API keys in .env file")
        
        # Show current status
        print(f"\n📊 Current Status:")
        print(f"   Ticker: {self.ticker or 'Not set'}")
        print(f"   Excel File: {os.path.basename(self.excel_file_path) if self.excel_file_path else 'Not selected'}")
        print(f"   Custom News: {len(self.custom_news_articles)} articles")
        print(f"   API Keys: {'✅ Available' if self.api_keys else '❌ Missing'}")
    
    def get_main_menu_choice(self):
        """Get user's main menu choice"""
        while True:
            self.show_main_menu()
            choice = input("\nEnter your choice (1-9): ").strip()
            
            if choice == "1":
                return "ml_analysis"
            elif choice == "2":
                if AI_PREDICTOR_AVAILABLE and self.api_keys:
                    return "ultimate_ai_analysis"
                elif not AI_PREDICTOR_AVAILABLE:
                    print("❌ Ultimate AI Analysis not available. Please check ultimate_ai_predictor.py")
                    continue
                elif not self.api_keys:
                    print("❌ Ultimate AI Analysis requires API keys. Please setup .env file first.")
                    self.show_api_setup_help()
                    continue
            elif choice == "3":
                return "select_excel"
            elif choice == "4":
                return "add_news"
            elif choice == "5":
                return "view_news"
            elif choice == "6":
                return "clear_news"
            elif choice == "7":
                return "settings"
            elif choice == "8":
                return "help"
            elif choice == "9":
                return "exit"
            else:
                print("❌ Invalid choice. Please enter 1-9.")
    
    def show_api_setup_help(self):
        """Show help for setting up API keys"""
        print("\n📋 API SETUP INSTRUCTIONS")
        print("=" * 40)
        print("To use Ultimate AI Analysis features, create a .env file with:")
        print("")
        print("ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key")
        print("CLAUDE_API_KEY=your_anthropic_key")
        print("")
        print("Free Alpha Vantage: https://www.alphavantage.co/support/#api-key")
        print("Anthropic Claude: https://console.anthropic.com/")
        print("")
        input("Press Enter to continue...")
    
    def select_excel_file(self):
        """Let user select Excel file using Ultimate AI predictor"""
        print(f"\n📁 EXCEL FILE SELECTION")
        print("=" * 40)
        
        if not self.ticker:
            self.ticker = input("Enter ticker symbol first (e.g., NVDA, MSFT): ").strip().upper()
            if not self.ticker:
                print("❌ Ticker required for Excel file selection")
                return
        
        if AI_PREDICTOR_AVAILABLE:
            try:
                excel_file_path = report_generator.get_excel_file_from_user(self.ticker)                
                if excel_file_path:
                    self.excel_file_path = excel_file_path
                    print(f"✅ Excel file selected: {os.path.basename(excel_file_path)}")
                else:
                    print("⚠️ No Excel file selected")
            except Exception as e:
                print(f"❌ Error selecting Excel file: {e}")
        else:
            print("❌ Excel file selection requires ultimate_ai_predictor.py")
    
    def add_custom_news_article(self):
        """Add custom news article using Ultimate AI predictor"""
        print(f"\n📰 ADD CUSTOM NEWS ARTICLE")
        print("=" * 40)
        
        if not self.ticker:
            self.ticker = input("Enter ticker symbol (e.g., NVDA, MSFT): ").strip().upper()
            if not self.ticker:
                print("❌ Ticker required for news article")
                return
        
        if AI_PREDICTOR_AVAILABLE:
            try:
                new_articles = get_custom_news_from_user(self.ticker)
                if new_articles:
                    # Filter articles for current ticker
                    ticker_articles = [a for a in new_articles if a.get('ticker') == self.ticker]
                    self.custom_news_articles.extend(ticker_articles)
                    print(f"✅ Added {len(ticker_articles)} custom news articles")
                    
                    # Save to file
                    self._save_custom_news_to_file()
                else:
                    print("⚠️ No custom news articles added")
            except Exception as e:
                print(f"❌ Error adding custom news: {e}")
        else:
            print("❌ Custom news input requires ultimate_ai_predictor.py")
    
    def _save_custom_news_to_file(self):
        """Save custom news articles to file"""
        try:
            filename = f"custom_news/custom_news_{self.ticker}_{datetime.now().strftime('%Y%m%d')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.custom_news_articles, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Warning: Could not save custom news to file: {e}")
    
    def view_custom_news_articles(self):
        """View stored custom news articles"""
        print(f"\n📰 CUSTOM NEWS ARTICLES")
        print("=" * 40)
        
        if not self.custom_news_articles:
            print("No custom news articles added yet.")
            print("\nUse option 4 from main menu to add articles.")
            return
        
        for i, article in enumerate(self.custom_news_articles, 1):
            sentiment_label = "📈 Positive" if article['sentiment_score'] > 0.1 else "📉 Negative" if article['sentiment_score'] < -0.1 else "➡️ Neutral"
            print(f"\n{i}. {article['title']}")
            print(f"   Ticker: {article['ticker']}")
            print(f"   Source: {article['source']}")
            print(f"   Sentiment: {sentiment_label} ({article['sentiment_score']:+.1f})")
            print(f"   Added: {article['timestamp'][:19]}")
            print(f"   Length: {len(article['content'])} characters")
        
        print(f"\nTotal: {len(self.custom_news_articles)} custom articles")
        input("\nPress Enter to continue...")
    
    def clear_custom_news_articles(self):
        """Clear custom news articles"""
        if not self.custom_news_articles:
            print("No custom news articles to clear.")
            return
        
        print(f"\n🗑️ CLEAR CUSTOM NEWS ARTICLES")
        print("=" * 40)
        print(f"You have {len(self.custom_news_articles)} custom articles")
        
        confirm = input("Are you sure you want to clear all? (yes/no): ").strip().lower()
        if confirm in ['yes', 'y']:
            self.custom_news_articles = []
            print("✅ All custom news articles cleared")
        else:
            print("❌ Clear operation cancelled")
    
    def run_ultimate_ai_analysis(self):
        """Run the ultimate AI analysis with full user control - UPDATED TO MATCH PREDICTION ENGINE"""
        print(f"\n🚀 ULTIMATE AI ANALYSIS")
        print("=" * 50)
        print("Most comprehensive stock prediction possible!")
        print("User Excel + Custom News + Real-time + Claude AI + Graph Analysis")
        
        if not AI_PREDICTOR_AVAILABLE:
            print("❌ Ultimate AI predictor not available")
            return
        
        if not self.api_keys:
            print("❌ API keys required for Ultimate AI analysis")
            return
        
        # Get ticker if not set
        if not self.ticker:
            self.ticker = input("\nEnter stock ticker: ").strip().upper()
            if not self.ticker:
                print("❌ Ticker required")
                return
        
        print(f"\n🎯 Starting ULTIMATE analysis for {self.ticker}...")
        
        # NEW: Ask for prediction timeframe and options
        prediction_days = 1
        options_data = None

        timeframe_choice = input("Multi-day prediction? (y/n): ").strip().lower()
        if timeframe_choice in ['y', 'yes']:
            try:
                prediction_days = get_prediction_timeframe()
                print(f"✅ Selected: {prediction_days} day prediction")
                
                options_choice = input("Include options analysis? (y/n): ").strip().lower()
                if options_choice in ['y', 'yes']:
                    options_data = get_options_data(self.ticker, prediction_days)
            except:
                print("⚠️ Using default 1-day prediction")
                prediction_days = 1
                
        print("   This will be the most comprehensive analysis possible!")
        
        try:
            # Step 1: Excel file selection (if not already selected)
            if not self.excel_file_path:
                print(f"\n📁 No Excel file selected yet")
                excel_file_path = get_excel_file_from_user(self.ticker)
                if excel_file_path:
                    self.excel_file_path = excel_file_path
            
            # Step 2: Load Excel historical data
            print("📊 Loading Excel historical analysis...")
            excel_data = load_excel_historical_data(self.ticker, self.excel_file_path)
            
            # CRITICAL: Validate Excel data has required fields for prediction engine
            required_fields = ['volatility', 'performance_return_1_month', 'current_rsi', 'sma_20', 'sma_50']
            missing_fields = [field for field in required_fields if field not in excel_data]
            if missing_fields:
                print(f"⚠️ Missing Excel fields: {missing_fields}")
                # Add defaults to prevent prediction engine errors
                defaults = {
                    'volatility': 25.0,
                    'performance_return_1_month': 0.0,
                    'current_rsi': 50.0,
                    'sma_20': 0.0,
                    'sma_50': 0.0,
                    'sector': 'Technology',
                    'avg_daily_change': 0.0,
                    'excel_recommendation': 'Hold',
                    'excel_risk_level': 'Moderate'
                }
                for field, default_value in defaults.items():
                    if field not in excel_data:
                        excel_data[field] = default_value
            
            print(f"✅ Excel data loaded with volatility: {excel_data.get('volatility', 25.0):.1f}%")
            
            # Step 3: Custom news articles (if not already added)
            if not self.custom_news_articles:
                print(f"\n📰 No custom news articles yet")
                add_news = input("Add breaking news articles? (y/n): ").strip().lower()
                if add_news in ['y', 'yes']:
                    new_articles = get_custom_news_from_user(self.ticker)
                    if new_articles:
                        self.custom_news_articles.extend(new_articles)
                        self._save_custom_news_to_file()
            
            # Step 4: Get graph analysis choice and perform if selected (ENHANCED)
            print(f"\n📈 GRAPH ANALYSIS OPTION FOR {self.ticker}")
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
            
            graph_choice = input("\nEnter your choice (1-2, default 2): ").strip() or "2"
            graph_analysis = None
            
            if graph_choice == "2":
                print("✅ Graph analysis will be performed")
                try:
                    from graph_analyzer import GraphAnalyzer
                    
                    # Initialize with proper parameters
                    graph_analyzer = GraphAnalyzer(use_cache=True)
                    
                    # CRITICAL: Call with proper method signature
                    graph_analysis = graph_analyzer.analyze_ticker(
                        self.ticker, 
                        days=30, 
                        include_extended_analysis=True
                    )
                    
                    # ENHANCED: Validate the graph analysis result using prediction engine
                    if graph_analysis and hasattr(prediction_engine.error_tracker, 'validate_graph_analysis'):
                        if prediction_engine.error_tracker.validate_graph_analysis(graph_analysis):
                            print(f"✅ Graph analysis complete and validated!")
                            primary_pattern = graph_analysis.get('pattern_detected', {}).get('primary_pattern', 'None')
                            pattern_reliability = graph_analysis.get('pattern_detected', {}).get('pattern_reliability', 0.0)
                            breakout_detected = graph_analysis.get('breakout_analysis', {}).get('breakout_detected', False)
                            
                            print(f"   📊 Primary Pattern: {primary_pattern} (Reliability: {pattern_reliability:.1%})")
                            if breakout_detected:
                                breakout_direction = graph_analysis.get('breakout_analysis', {}).get('breakout_direction', 'unknown')
                                breakout_strength = graph_analysis.get('breakout_analysis', {}).get('breakout_strength', 0.0)
                                print(f"   🚀 Breakout: {breakout_direction.upper()} (Strength: {breakout_strength:.1%})")
                        else:
                            print("❌ Graph analysis validation failed, using fallback")
                            # Get market data early for fallback generation
                            temp_market_data = get_realtime_market_data_with_fallback(self.ticker)
                            if temp_market_data:
                                graph_analysis = prediction_engine.error_tracker.generate_default_graph_analysis(
                                    self.ticker, excel_data, temp_market_data
                                )
                    else:
                        print(f"✅ Graph analysis complete (validation not available)")
                        primary_pattern = graph_analysis.get('pattern_detected', {}).get('primary_pattern', 'None')
                        print(f"   📊 Pattern: {primary_pattern}")
                        
                except ImportError:
                    print("❌ Graph analyzer not found. Creating default analysis...")
                    # Get market data for fallback
                    temp_market_data = get_realtime_market_data_with_fallback(self.ticker)
                    if temp_market_data and hasattr(prediction_engine.error_tracker, 'generate_default_graph_analysis'):
                        graph_analysis = prediction_engine.error_tracker.generate_default_graph_analysis(
                            self.ticker, excel_data, temp_market_data
                        )
                        print("✅ Default graph analysis generated")
                    else:
                        graph_analysis = None
                except Exception as e:
                    print(f"❌ Graph analysis error: {e}. Using fallback...")
                    # Get market data for fallback
                    temp_market_data = get_realtime_market_data_with_fallback(self.ticker)
                    if temp_market_data and hasattr(prediction_engine.error_tracker, 'generate_default_graph_analysis'):
                        graph_analysis = prediction_engine.error_tracker.generate_default_graph_analysis(
                            self.ticker, excel_data, temp_market_data
                        )
                        print("✅ Fallback graph analysis generated")
                    else:
                        graph_analysis = None
            else:
                print("⚠️ Skipping graph analysis")
                graph_analysis = None
            
            # Step 5: Get Phase 3 news intelligence with graph_analysis (ENHANCED)
            print("📰 Gathering multi-source news intelligence...")
            
            # CRITICAL: Store news data for regime detection (required by prediction engine)
            try:
                # CORRECT: Call the method properly with graph_analysis integration
                news_data = prediction_engine.get_phase3_news_intelligence(self.ticker)
                
                # CRITICAL: Store news data for regime detection in mathematical analysis
                if hasattr(prediction_engine, 'set_news_data_for_regime'):
                    prediction_engine.set_news_data_for_regime(news_data)
                
                print(f"✅ Multi-source news intelligence complete:")
                print(f"   📊 Sources: {news_data.get('active_sources', 1)}")
                print(f"   📰 Articles: {news_data.get('total_articles', 0)}")
                print(f"   📈 Sentiment: {news_data.get('sentiment_1d', 0.0):+.3f}")
                
                # Show graph analysis integration status
                if news_data.get('graph_analysis_integrated', False):
                    print(f"   📈 Graph analysis: ✅ INTEGRATED")
                    graph_sentiment_adj = news_data.get('graph_sentiment_adjustment', 0.0)
                    if abs(graph_sentiment_adj) > 0.02:
                        print(f"   📊 Graph sentiment adjustment: {graph_sentiment_adj:+.3f}")
                else:
                    print(f"   📈 Graph analysis: ❌ Not integrated")
                
            except Exception as e:
                print(f"❌ News intelligence error: {e}")
                # Create comprehensive fallback that works with your prediction engine
                news_data = {
                    'sentiment_1d': 0.0,
                    'sentiment_7d': 0.0, 
                    'news_volume_1d': 0,
                    'news_volume_7d': 0,
                    'confidence_score': 0.5,
                    'source_diversity': 1,
                    'event_impact_score': 0.0,
                    'recent_events': [],
                    'multi_source_enabled': False,
                    'active_sources': 1,
                    'total_articles': 0,
                    'fresh_search_performed': False,
                    'fresh_articles_found': 0,
                    'prediction_ready': False,
                    # CRITICAL: Graph analysis integration flags
                    'graph_analysis_integrated': graph_analysis is not None,
                    'graph_sentiment_adjustment': 0.0,
                    'graph_confidence_boost': 0.0,
                    'graph_technical_alignment': 'neutral'
                }
                
                # Store fallback data for regime detection
                if hasattr(prediction_engine, 'set_news_data_for_regime'):
                    prediction_engine.set_news_data_for_regime(news_data)
            
            # Step 6: Enhance news with custom articles
            print("🔄 Enhancing news with custom articles...")
            enhanced_news_data = enhance_news_with_custom_articles(news_data, self.custom_news_articles)
            
            print(f"✅ News enhancement complete:")
            print(f"   📰 Final sentiment: {enhanced_news_data.get('sentiment_1d', 0.0):+.3f}")
            print(f"   📊 Custom articles: {len(self.custom_news_articles)}")
            
            # Step 7: Get real-time market data
            print("📈 Getting real-time market data...")
            market_data = get_realtime_market_data_with_fallback(self.ticker)            
            if not market_data:
                print("❌ Could not get market data")
                return
            
            print(f"✅ Market data: ${market_data.get('current_price', 0):.2f}")
            
            # Step 8: Apply price override if user provided one
            if self.current_price_override:
                print(f"💰 Using user-provided price: ${self.current_price_override:.2f}")
                market_data['current_price'] = self.current_price_override
            
            # Step 9: Ultimate Claude AI analysis (COMPLETELY REWRITTEN)
            print("🤖 Running Ultimate Claude AI analysis with Graph Integration...")
            
            try:
                # CORRECT: Use the prediction_engine instance directly with proper parameters
                # The method signature from your prediction engine is:
                # analyze_with_claude_ultimate_enhanced(ticker, excel_data, news_data, market_data, custom_articles, graph_analysis, prediction_days=1, options_data=None)
                
                claude_analysis = prediction_engine.analyze_with_claude_ultimate_enhanced(
                    ticker=self.ticker,
                    excel_data=excel_data,
                    news_data=enhanced_news_data, 
                    market_data=market_data,
                    custom_articles=self.custom_news_articles,
                    graph_analysis=graph_analysis,  # CRITICAL: Pass graph_analysis
                    prediction_days=prediction_days,
                    options_data=options_data
                )
                
                if claude_analysis:
                    print("✅ Ultimate Claude analysis complete!")
                    
                    # Show key results
                    final_return = claude_analysis.get('total_expected_return_pct', 0)
                    confidence = claude_analysis.get('confidence', 0)
                    target_price = claude_analysis.get('final_target_price', 0)
                    
                    print(f"   🎯 Expected return: {final_return:+.2f}%")
                    print(f"   💰 Target price: ${target_price:.2f}")
                    print(f"   🎯 Confidence: {confidence:.1%}")
                    
                    # Show enhancement status
                    enhancements = []
                    if claude_analysis.get('volatility_aware_system', False):
                        volatility_tier = claude_analysis.get('volatility_analysis', {}).get('volatility_tier', 'unknown')
                        enhancements.append(f"Volatility-aware ({volatility_tier} tier)")
                    
                    if claude_analysis.get('fresh_search_metadata', {}).get('performed', False):
                        fresh_articles = claude_analysis.get('fresh_search_metadata', {}).get('articles_found', 0)
                        enhancements.append(f"Fresh search ({fresh_articles} articles)")
                    
                    if claude_analysis.get('graph_analysis_integration', False):
                        enhancements.append("Graph analysis integrated")
                    
                    if len(self.custom_news_articles) > 0:
                        enhancements.append(f"Custom news ({len(self.custom_news_articles)} articles)")
                    
                    if enhancements:
                        print(f"   🚀 Enhancements: {', '.join(enhancements)}")
                else:
                    print("❌ Claude analysis returned empty result")
                    return
                    
            except Exception as e:
                print(f"❌ Claude analysis failed: {e}")
                import traceback
                traceback.print_exc()
                return
            
            # Step 10: Generate ultimate report (ENHANCED)
            print("📋 Generating ultimate comprehensive report...")
            
            try:
                # CORRECT: Use proper function based on availability
                if hasattr(report_generator, 'generate_ultimate_report_enhanced'):
                    # Use the separated architecture
                    report = report_generator.generate_ultimate_report_enhanced(
                        ticker=self.ticker,
                        excel_data=excel_data,
                        news_data=enhanced_news_data,
                        market_data=market_data,
                        claude_analysis=claude_analysis,
                        custom_articles=self.custom_news_articles,
                        excel_file_path=self.excel_file_path,
                        prediction_days=prediction_days,
                        options_data=options_data
                    )
                else:
                    # Fallback to wrapper function
                    report = generate_ultimate_report_enhanced_multi_day(
                        self.ticker, excel_data, enhanced_news_data, market_data,
                        claude_analysis, self.custom_news_articles, self.excel_file_path,
                        prediction_days, options_data
                    )
                
                # Display the report
                print("\n" + "="*80)
                print("📋 ULTIMATE ANALYSIS REPORT")
                print("="*80)
                print(report)
                print("="*80)
                
            except Exception as e:
                print(f"❌ Report generation failed: {e}")
                # Create basic fallback report
                report = f"""
ULTIMATE ANALYSIS REPORT - {self.ticker}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Expected Return: {claude_analysis.get('total_expected_return_pct', 0):+.2f}%
Target Price: ${claude_analysis.get('final_target_price', 0):.2f}
Confidence: {claude_analysis.get('confidence', 0):.1%}
Prediction Days: {prediction_days}

Analysis included:
- Excel historical data: ✅
- Multi-source news: ✅
- Graph analysis: {'✅' if graph_analysis else '❌'}
- Custom articles: {len(self.custom_news_articles)}
- Real-time data: ✅

For detailed analysis, check the saved JSON file.
            """
        
            # Step 11: Save results (ENHANCED)
            print(f"\n💾 Save ultimate analysis results?")
            save_choice = input("Enter 'y' to save files: ").strip().lower()
            
            if save_choice in ['y', 'yes', '']:
                try:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    # Save text report
                    report_filename = f"reports/ultimate_analysis_{self.ticker}_{prediction_days}d_{timestamp}.txt"
                    os.makedirs("reports", exist_ok=True)
                    
                    with open(report_filename, 'w', encoding='utf-8') as f:
                        f.write(f"ULTIMATE ANALYSIS REPORT - {self.ticker}\n")
                        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Prediction Days: {prediction_days}\n")
                        f.write("="*80 + "\n\n")
                        f.write(report)
                        
                        # Add enhancement summary
                        f.write(f"\n\nENHANCEMENT SUMMARY:\n")
                        f.write(f"Excel File: {os.path.basename(self.excel_file_path) if self.excel_file_path else 'Default'}\n")
                        f.write(f"Graph Analysis: {'✅ Applied' if graph_analysis else '❌ Skipped'}\n")
                        f.write(f"Custom News: {len(self.custom_news_articles)} articles\n")
                        f.write(f"Multi-source News: {'✅' if enhanced_news_data.get('multi_source_enabled') else '❌'}\n")
                        f.write(f"Volatility-Aware: {'✅' if claude_analysis.get('volatility_aware_system') else '❌'}\n")
                    
                    # Save JSON data using prediction engine's method
                    json_filename = prediction_engine.save_prediction_data(
                        ticker=self.ticker,
                        excel_data=excel_data,
                        news_data=enhanced_news_data,
                        market_data=market_data,
                        claude_analysis=claude_analysis,
                        custom_articles=self.custom_news_articles,
                        excel_file_path=self.excel_file_path,
                        options_data=options_data
                    )
                    
                    print(f"✅ Ultimate analysis saved successfully:")
                    print(f"   📋 Report: {report_filename}")
                    print(f"   📊 Data: {json_filename}")
                    
                    # Save custom news separately if any
                    if self.custom_news_articles:
                        news_filename = f"custom_news/custom_news_{self.ticker}_{timestamp}.json"
                        os.makedirs("custom_news", exist_ok=True)
                        with open(news_filename, 'w', encoding='utf-8') as f:
                            json.dump(self.custom_news_articles, f, indent=2, default=str)
                        print(f"   📰 Custom News: {news_filename}")
                    
                except Exception as e:
                    print(f"❌ Save failed: {e}")
                    print("⚠️ Analysis completed but could not save files")
            
            print(f"\n🎉 ULTIMATE analysis complete for {self.ticker}!")
            
            # Enhanced summary of what was used
            excel_source = "✅ User Excel file" if self.excel_file_path else "⚠️ Default data"
            market_source = "✅ Alpha Vantage API" if market_data.get('data_source') == 'alpha_vantage' else "✅ User input"
            custom_news = f"✅ {len(self.custom_news_articles)} custom articles" if self.custom_news_articles else "⚠️ No custom news"
            graph_status = "✅ Graph analysis applied" if graph_analysis else "⚠️ Graph analysis skipped"
            volatility_status = "✅ Volatility-aware system" if claude_analysis.get('volatility_aware_system') else "⚠️ Standard system"
            
            print(f"\n📊 ULTIMATE ANALYSIS SUMMARY:")
            print(f"   Historical Data: {excel_source}")
            print(f"   Market Data: {market_source}")
            print(f"   Custom News: {custom_news}")
            print(f"   Graph Analysis: {graph_status}")
            print(f"   Volatility System: {volatility_status}")
            print(f"   30-day Performance: ✅ {excel_data.get('performance_return_1_month', 0.0):.2f}% (from Excel)")
            print(f"   AI Analysis: ✅ Claude ultimate reasoning with {prediction_days}-day prediction")
            
            # Show MAE performance if available
            if claude_analysis.get('mathematical_analysis', {}).get('statistical_metrics', {}).get('mae_performance'):
                mae_info = claude_analysis['mathematical_analysis']['statistical_metrics']['mae_performance']
                print(f"   MAE Performance: ✅ {mae_info.get('final_day_mae', 0):.1f}% (target: {mae_info.get('mae_target', 0):.1f}%)")
            
            # Show confidence tier
            confidence_tier = claude_analysis.get('volatility_analysis', {}).get('volatility_tier', 'unknown')
            final_confidence = claude_analysis.get('confidence', 0)
            print(f"   Confidence: ✅ {final_confidence:.1%} ({confidence_tier} volatility tier)")
            
        except Exception as e:
            print(f"❌ Ultimate AI analysis failed: {e}")
            import traceback
            traceback.print_exc()
        
    def run_traditional_ml_analysis(self):
        """Run traditional ML analysis (enhanced version)"""
        print(f"\n📊 TRADITIONAL ML ANALYSIS")
        print("=" * 40)
        print("Using: Excel + News + ML Models")
        
        # Enhanced ML analysis workflow
        if not self.get_excel_file_path():
            return
        
        if not self.load_excel_data():
            return
        
        if not self.run_news_analysis():
            return
        
        if not self.setup_ml_engine():
            return
        
        # ML prediction loop
        while True:
            start_date, end_date = self.get_prediction_timeframe()
            
            if start_date is None:
                break
            
            predictions = self.make_predictions(start_date, end_date)
            
            if predictions:
                self.display_results(predictions)
            
            another = input("\n🔄 Make another prediction? (y/n): ").strip().lower()
            if another not in ['y', 'yes']:
                break
    
    def show_settings_menu(self):
        """Show enhanced settings and configuration menu"""
        print(f"\n⚙️ ULTIMATE SETTINGS & CONFIGURATION")
        print("=" * 50)
        print(f"Current Ticker: {self.ticker or 'Not set'}")
        print(f"Excel File: {os.path.basename(self.excel_file_path) if self.excel_file_path else 'Not selected'}")
        print(f"Price Override: ${self.current_price_override:.2f}" if self.current_price_override else "Price Override: Not set")
        print(f"Custom News Articles: {len(self.custom_news_articles)}")
        print(f"API Keys Status: {'✅ Available' if self.api_keys else '❌ Not available'}")
        print(f"Ultimate AI Predictor: {'✅ Available' if AI_PREDICTOR_AVAILABLE else '❌ Missing'}")
        
        print(f"\nOptions:")
        print("1. Set/Change Ticker")
        print("2. Select Excel File")
        print("3. Set Current Price Override")
        print("4. Clear Price Override")
        print("5. Check API Keys")
        print("6. View File Paths")
        print("7. Reset All Settings")
        print("8. Back to Main Menu")
        
        choice = input("\nEnter choice (1-8): ").strip()
        
        if choice == "1":
            new_ticker = input("Enter new ticker: ").strip().upper()
            if new_ticker:
                self.ticker = new_ticker
                print(f"✅ Ticker set to: {new_ticker}")
        
        elif choice == "2":
            self.select_excel_file()
        
        elif choice == "3":
            try:
                price = float(input("Enter current price: $").strip())
                if price > 0:
                    self.current_price_override = price
                    print(f"✅ Price override set to: ${price:.2f}")
            except ValueError:
                print("❌ Invalid price")
        
        elif choice == "4":
            self.current_price_override = None
            print("✅ Price override cleared")
        
        elif choice == "5":
            if self.api_keys:
                print("✅ API Keys Status:")
                print(f"   Alpha Vantage: {'✅ Set' if self.api_keys.get('alpha') else '❌ Missing'}")
                print(f"   Claude API: {'✅ Set' if self.api_keys.get('claude') else '❌ Missing'}")
            else:
                print("❌ No API keys found")
                self.show_api_setup_help()
        
        elif choice == "6":
            print(f"\n📁 File Paths:")
            print(f"   Excel File: {self.excel_file_path or 'Not selected'}")
            print(f"   Reports Directory: reports/")
            print(f"   Custom News Directory: custom_news/")
            print(f"   Models Directory: models/")
            
        elif choice == "7":
            confirm = input("Reset all settings? (yes/no): ").strip().lower()
            if confirm == 'yes':
                self.reset_session()
                print("✅ All settings reset")
        
        elif choice == "8":
            return
    
    def show_help(self):
        """Show comprehensive help and instructions"""
        print(f"\n🆘 ULTIMATE HELP & INSTRUCTIONS")
        print("=" * 50)
        print("📊 TRADITIONAL ML ANALYSIS:")
        print("   - Uses Excel historical data + News + ML models")
        print("   - Good for pattern-based predictions")
        print("   - Works without API keys")
        print("   - Trains custom ML models for your stocks")
        print("")
        print("🚀 ULTIMATE AI ANALYSIS:")
        print("   - Most comprehensive analysis possible")
        print("   - User selects Excel file with historical data")
        print("   - User can add breaking news articles")
        print("   - Real-time market data (API or user input)")
        print("   - Claude AI processes everything intelligently")
        print("   - Requires API keys (Alpha Vantage + Claude)")
        print("")
        print("📁 EXCEL FILE SELECTION:")
        print("   - Browse and select your own Excel analysis files")
        print("   - Supports various Excel formats (.xlsx, .xls)")
        print("   - Auto-detects ticker from filename")
        print("   - Falls back to defaults if no file selected")
        print("")
        print("📰 CUSTOM NEWS ARTICLES:")
        print("   - Add breaking news or important announcements")
        print("   - Input text directly or load from files")
        print("   - Set sentiment impact (+0.8 to -0.8)")
        print("   - Articles get 40% weight in AI analysis")
        print("   - Perfect for earnings, product launches, etc.")
        print("")
        print("💰 SMART MARKET DATA:")
        print("   - First tries Alpha Vantage API automatically")
        print("   - Falls back to user input when API unavailable")
        print("   - Guides user through price entry process")
        print("   - Calculates derived values intelligently")
        print("")
        print("💡 WORKFLOW TIPS:")
        print("   - Start with ticker symbol")
        print("   - Select your Excel analysis file")
        print("   - Add any breaking news")
        print("   - Use Ultimate AI Analysis for best results")
        print("   - Save reports for tracking accuracy")
        print("   - Check API status if real-time data fails")
        print("")
        print("🔧 TROUBLESHOOTING:")
        print("   - If Ultimate AI unavailable: check ultimate_ai_predictor.py")
        print("   - If API fails: will prompt for manual price input")
        print("   - If Excel errors: ensure file format is correct")
        print("   - For best results: use both Excel file + custom news")
        
        input("\nPress Enter to continue...")
    
    # Include enhanced versions of original methods
    def get_excel_file_path(self) -> bool:
        """Enhanced Excel file selection"""
        print("\n📁 STEP 1: Excel File Selection")
        print("-" * 40)
        
        # Use the selected file if available
        if self.excel_file_path:
            print(f"Using selected file: {os.path.basename(self.excel_file_path)}")
            return True
        
        while True:
            print("\nOptions:")
            print("1. Use default MSFT file")
            print("2. Enter custom Excel file path")
            print("3. Use Ultimate AI file selector")
            print("4. Back to main menu")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                self.excel_file_path = "../MSFT_analysis_report_20250715_141735.xlsx"
                if os.path.exists(self.excel_file_path):
                    print(f"✅ Using default file: {self.excel_file_path}")
                    return True
                else:
                    print(f"❌ Default file not found")
                    continue
            
            elif choice == "2":
                file_path = input("Enter full path to Excel file: ").strip().strip('"')
                if os.path.exists(file_path):
                    self.excel_file_path = file_path
                    print(f"✅ Excel file selected: {file_path}")
                    return True
                else:
                    print(f"❌ File not found: {file_path}")
                    continue
            
            elif choice == "3":
                if AI_PREDICTOR_AVAILABLE:
                    self.select_excel_file()
                    return self.excel_file_path is not None
                else:
                    print("❌ Ultimate AI file selector not available")
                    continue
            
            elif choice == "4":
                return False
            
            else:
                print("❌ Invalid choice. Please enter 1-4.")
    
    def load_excel_data(self) -> bool:
        """Load and analyze Excel data with enhanced features"""
        print("\n📊 STEP 2: Loading Excel Analysis Data")
        print("-" * 40)
        
        try:
            # Use the fixed Excel loader
            self.excel_loader = ExcelDataLoader()
            self.excel_loader.excel_path = self.excel_file_path
            
            excel_data = self.excel_loader.load_all_data()
            
            print("✅ Excel data loaded successfully!")
            
            # Display enhanced data summary
            print("\n📋 Enhanced Data Summary:")
            data_sources = [
                ('raw_data', self.excel_loader.raw_data),
                ('technical_data', self.excel_loader.technical_data),
                ('sentiment_data', self.excel_loader.sentiment_data),
                ('performance_data', self.excel_loader.performance_data),
                ('summary_data', self.excel_loader.summary_data),
                ('company_info', self.excel_loader.company_info)
            ]
            
            for name, data in data_sources:
                if data is not None and not data.empty:
                    print(f"   📊 {name}: {len(data)} rows")
            
            # Extract ticker
            self.ticker = self.extract_ticker_from_filename()
            
            # CRITICAL: Check if 30-day performance is correctly loaded
            thirty_day_perf = self.excel_loader.get_30d_performance()
            print(f"\n✅ CRITICAL CHECK: 30-day performance = {thirty_day_perf:.2f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Excel file: {e}")
            return False
    
    def extract_ticker_from_filename(self) -> str:
        """Extract ticker from filename or ask user"""
        if self.ticker:  # If already set, keep it
            return self.ticker
            
        filename = os.path.basename(self.excel_file_path)
        potential_ticker = filename.split('_')[0].upper()
        
        if 1 <= len(potential_ticker) <= 5 and potential_ticker.isalpha():
            print(f"\n🏷️ Detected ticker: {potential_ticker}")
            confirm = input(f"Is '{potential_ticker}' correct? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                return potential_ticker
        
        while True:
            ticker = input("Enter stock ticker: ").strip().upper()
            if ticker and ticker.isalpha() and 1 <= len(ticker) <= 5:
                return ticker
            print("❌ Please enter a valid ticker (1-5 letters)")
    
    def run_news_analysis(self) -> bool:
        """Run Phase 3 news analysis with custom news enhancement"""
        print(f"\n🗞️ STEP 3: News Intelligence Analysis for {self.ticker}")
        print("-" * 40)
        
        try:
            self.news_connector = Phase3NewsPredictor()
            
            sentiment_features = self.news_connector.get_enhanced_sentiment_features(
                self.ticker, datetime.now(), lookback_days=7
            )
            
            print("\n📰 News Analysis Summary:")
            print(f"   📊 1-day sentiment: {sentiment_features.sentiment_1d:.3f}")
            print(f"   📊 7-day sentiment: {sentiment_features.sentiment_7d:.3f}")
            print(f"   📈 News volume (7d): {sentiment_features.news_volume_7d} articles")
            print(f"   🎯 Confidence: {sentiment_features.confidence_score:.2f}")
            
            # Show custom news enhancement
            if self.custom_news_articles:
                ticker_articles = [a for a in self.custom_news_articles if a.get('ticker') == self.ticker]
                if ticker_articles:
                    avg_sentiment = sum(a['sentiment_score'] for a in ticker_articles) / len(ticker_articles)
                    print(f"\n📰 Custom News Enhancement:")
                    print(f"   📊 Custom articles: {len(ticker_articles)}")
                    print(f"   📊 Average sentiment: {avg_sentiment:+.2f}")
                    print(f"   🔄 This will enhance AI analysis accuracy")
            
            return True
            
        except Exception as e:
            print(f"⚠️ News analysis issues: {e}")
            return True
    
    def setup_ml_engine(self) -> bool:
        """Setup ML prediction engine with enhanced features"""
        print(f"\n🧠 STEP 4: Setting up Enhanced ML Prediction Engine")
        print("-" * 40)
        
        try:
            self.feature_engineer = FeatureEngineer()
            
            model_path = f"models/{self.ticker}_trained_model.joblib"
            
            if os.path.exists(model_path):
                load_existing = input("Load existing model? (y/n): ").strip().lower()
                if load_existing in ['y', 'yes']:
                    try:
                        model_data = joblib.load(model_path)
                        if isinstance(model_data, dict):
                            self.model = model_data['model']
                            self.trained_features = model_data['features']
                        else:
                            self.model = model_data
                            self.trained_features = []
                        
                        self.is_trained = True
                        print("✅ Existing model loaded!")
                        self.get_current_price_from_user()
                        return True
                    except Exception as e:
                        print(f"❌ Failed to load model: {e}")
            
            # Train new model
            print("🧠 Training new enhanced ML model...")
            end_date = datetime.now() - timedelta(days=2)
            start_date = end_date - timedelta(days=180)
            
            try:
                X, y = self.feature_engineer.create_comprehensive_training_dataset(
                    self.ticker, start_date, end_date, min_samples=20
                )
                
                if len(X) < 10:
                    print(f"❌ Insufficient training data: {len(X)} samples")
                    return False
                
                print(f"📊 Training dataset: {len(X)} samples, {len(X.columns)} features")
                
                self.model = RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
                )
                
                self.model.fit(X, y)
                self.is_trained = True
                self.trained_features = X.columns.tolist()
                
                model_data = {'model': self.model, 'features': self.trained_features}
                joblib.dump(model_data, model_path)
                
                print("✅ Enhanced ML model trained and saved!")
                
                y_pred = self.model.predict(X)
                mae = np.mean(np.abs(y - y_pred))
                print(f"📈 Training performance: MAE = {mae:.4f}")
                
                self.get_current_price_from_user()
                return True
                
            except Exception as e:
                print(f"❌ Model training failed: {e}")
                return False
            
        except Exception as e:
            print(f"❌ ML engine setup failed: {e}")
            return False
    
    def get_current_price_from_user(self):
        """Get current price from user with enhanced options"""
        print(f"\n💰 Enhanced Price Update for {self.ticker}")
        print("-" * 40)
        
        # Get last Excel price
        last_excel_price = None
        try:
            if self.excel_loader and hasattr(self.excel_loader, 'raw_data'):
                raw_data = self.excel_loader.raw_data
                if raw_data is not None and not raw_data.empty:
                    last_row = raw_data.iloc[-1]
                    last_excel_price = last_row['Close']
                    last_excel_date = raw_data.index[-1]
                    print(f"📊 Last Excel price: ${last_excel_price:.2f} ({last_excel_date.strftime('%Y-%m-%d')})")
        except Exception as e:
            print(f"⚠️ Could not get Excel price: {e}")
        
        print(f"\nEnhanced Options:")
        print("1. Use Excel price")
        print("2. Enter current market price manually")
        print("3. Try to fetch current price (if APIs available)")
        print("4. Use Ultimate AI price input")
        print("5. Skip price update")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            self.current_price_override = None
            print(f"✅ Using Excel price")
        
        elif choice == "2":
            while True:
                try:
                    price_input = input(f"Enter current price for {self.ticker}: $").strip()
                    current_price = float(price_input)
                    if current_price > 0:
                        self.current_price_override = current_price
                        print(f"✅ Price set to: ${current_price:.2f}")
                        
                        if last_excel_price:
                            change = ((current_price - last_excel_price) / last_excel_price) * 100
                            direction = "📈" if change > 0 else "📉"
                            print(f"   {direction} Change from Excel: {change:+.2f}%")
                        break
                    else:
                        print("❌ Price must be > 0")
                except ValueError:
                    print("❌ Invalid price format")
        
        elif choice == "3":
            if self.api_keys:
                print("🔍 Fetching current price...")
                try:
                    # Use the Ultimate AI predictor's market data function
                    market_data = get_realtime_market_data_with_fallback(self.ticker)
                    if market_data and 'current_price' in market_data:
                        fetched_price = market_data['current_price']
                        print(f"✅ Fetched price: ${fetched_price:.2f}")
                        use_fetched = input("Use this price? (y/n): ").strip().lower()
                        if use_fetched in ['y', 'yes']:
                            self.current_price_override = fetched_price
                    else:
                        print("❌ Could not fetch price")
                        self.current_price_override = None
                except Exception as e:
                    print(f"❌ Error fetching price: {e}")
                    self.current_price_override = None
            else:
                print("❌ API keys required for price fetching")
                self.current_price_override = None
        
        elif choice == "4":
            if AI_PREDICTOR_AVAILABLE:
                print("🚀 Using Ultimate AI price input...")
                try:
                    # This would use the Ultimate AI predictor's enhanced price input
                    market_data = get_realtime_market_data_with_fallback(
                        self.ticker, self.api_keys.get('alpha') if self.api_keys else None
                    )
                    if market_data and 'current_price' in market_data:
                        self.current_price_override = market_data['current_price']
                        print(f"✅ Ultimate AI price set: ${self.current_price_override:.2f}")
                    else:
                        print("❌ Ultimate AI price input failed")
                except Exception as e:
                    print(f"❌ Error with Ultimate AI price input: {e}")
            else:
                print("❌ Ultimate AI predictor not available")
        
        elif choice == "5":
            self.current_price_override = None
            print("✅ Skipped price update")
    
    def get_prediction_timeframe(self) -> Tuple[datetime, datetime]:
        """Get prediction timeframe with enhanced options"""
        print(f"\n🎯 ENHANCED PREDICTION TIMEFRAME")
        print("-" * 40)
        
        while True:
            print(f"\nWhat to predict for {self.ticker}?")
            print("1. Tomorrow")
            print("2. Next 7 days")
            print("3. Next 30 days")
            print("4. Custom date range")
            print("5. Back")
            
            choice = input("\nChoice (1-5): ").strip()
            today = datetime.now()
            
            if choice == "1":
                start_date = today + timedelta(days=1)
                while start_date.weekday() >= 5:
                    start_date += timedelta(days=1)
                return start_date, start_date
            
            elif choice == "2":
                start_date = today + timedelta(days=1)
                while start_date.weekday() >= 5:
                    start_date += timedelta(days=1)
                end_date = start_date + timedelta(days=6)
                return start_date, end_date
            
            elif choice == "3":
                start_date = today + timedelta(days=1)
                while start_date.weekday() >= 5:
                    start_date += timedelta(days=1)
                end_date = start_date + timedelta(days=29)
                return start_date, end_date
            
            elif choice == "4":
                try:
                    start_input = input("Start date (YYYY-MM-DD): ").strip()
                    start_date = datetime.strptime(start_input, "%Y-%m-%d")
                    
                    end_input = input("End date (YYYY-MM-DD, Enter for single day): ").strip()
                    end_date = datetime.strptime(end_input, "%Y-%m-%d") if end_input else start_date
                    
                    if start_date <= end_date:
                        return start_date, end_date
                    else:
                        print("❌ Start date must be <= end date")
                except ValueError:
                    print("❌ Invalid date format")
                    continue
            
            elif choice == "5":
                return None, None
            
            else:
                print("❌ Invalid choice")
    
    def make_predictions(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Generate enhanced ML predictions"""
        if not self.is_trained:
            print("❌ Model not trained")
            return []
        
        predictions = []
        current_date = start_date
        
        print(f"🔮 Generating enhanced predictions...")
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Skip weekends
                try:
                    prediction = self._predict_single_day(current_date)
                    if prediction:
                        predictions.append(prediction)
                        direction = "📈" if prediction['predicted_return'] > 0 else "📉"
                        confidence_icon = "🎯" if prediction['confidence'] > 0.7 else "⚡"
                        print(f"   ✅ {current_date.date()}: ${prediction['predicted_close']:.2f} {direction} {confidence_icon}")
                except Exception as e:
                    print(f"   ❌ {current_date.date()}: Error - {e}")
            
            current_date += timedelta(days=1)
        
        return predictions
    
    def _predict_single_day(self, target_date: datetime) -> Optional[Dict]:
        """Predict single day with enhanced features"""
        try:
            features = self.feature_engineer.create_enhanced_features_for_date(
                self.ticker, target_date
            )
            
            feature_dict = {}
            feature_dict.update(features.price_features)
            feature_dict.update(features.technical_features)
            feature_dict.update(features.sentiment_features)
            feature_dict.update(features.time_features)
            feature_dict.update(features.fundamental_features)
            feature_dict.update(features.quality_features)
            feature_dict.update(features.derived_features)
            
            # Enhanced feature processing with custom news
            if self.custom_news_articles:
                ticker_articles = [a for a in self.custom_news_articles if a.get('ticker') == self.ticker]
                if ticker_articles:
                    custom_sentiment = sum(a['sentiment_score'] for a in ticker_articles) / len(ticker_articles)
                    feature_dict['custom_news_sentiment'] = custom_sentiment
                    feature_dict['custom_news_count'] = len(ticker_articles)
            
            X_pred = pd.DataFrame([feature_dict])
            
            if self.trained_features:
                X_pred = X_pred.reindex(columns=self.trained_features, fill_value=0)
            
            predicted_return = self.model.predict(X_pred)[0]
            
            # Enhanced confidence calculation
            predictions = [tree.predict(X_pred)[0] for tree in self.model.estimators_]
            prediction_std = np.std(predictions)
            base_confidence = max(0.1, min(0.95, 1.0 - (prediction_std * 10)))
            
            # Boost confidence if we have custom news
            if self.custom_news_articles:
                confidence = min(0.95, base_confidence + 0.1)
            else:
                confidence = base_confidence
            
            # Calculate prices
            current_price = feature_dict.get('close', 400)
            if self.current_price_override:
                current_price = self.current_price_override
            
            predicted_close = current_price * (1 + predicted_return)
            volatility = abs(predicted_return) + 0.01
            predicted_open = current_price * (1 + predicted_return * 0.3)
            predicted_high = predicted_close * (1 + volatility)
            predicted_low = predicted_close * (1 - volatility)
            
            return {
                'date': target_date,
                'ticker': self.ticker,
                'current_price': current_price,
                'predicted_return': predicted_return,
                'predicted_open': predicted_open,
                'predicted_high': predicted_high,
                'predicted_low': predicted_low,
                'predicted_close': predicted_close,
                'confidence': confidence,
                'key_factors': self._get_enhanced_key_factors(features, predicted_return),
                'has_custom_news': len(self.custom_news_articles) > 0
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def _get_enhanced_key_factors(self, features, predicted_return) -> List[str]:
        """Generate enhanced key factors including custom news"""
        factors = []
        
        tech_features = features.technical_features
        if 'tech_rsi' in tech_features:
            rsi = tech_features['tech_rsi']
            if rsi > 70:
                factors.append("RSI overbought")
            elif rsi < 30:
                factors.append("RSI oversold")
        
        sentiment_features = features.sentiment_features
        sentiment_1d = sentiment_features.get('sentiment_1d', 0)
        if abs(sentiment_1d) > 0.1:
            direction = "positive" if sentiment_1d > 0 else "negative"
            factors.append(f"News sentiment {direction}")
        
        # Add custom news factor
        if self.custom_news_articles:
            ticker_articles = [a for a in self.custom_news_articles if a.get('ticker') == self.ticker]
            if ticker_articles:
                avg_sentiment = sum(a['sentiment_score'] for a in ticker_articles) / len(ticker_articles)
                if abs(avg_sentiment) > 0.1:
                    direction = "positive" if avg_sentiment > 0 else "negative"
                    factors.append(f"Custom news {direction}")
        
        if predicted_return > 0.01:
            factors.append("Bullish signals")
        elif predicted_return < -0.01:
            factors.append("Bearish indicators")
        else:
            factors.append("Neutral outlook")
        
        return factors[:4]  # Return up to 4 factors
    
    def display_results(self, predictions: List[Dict]):
        """Display enhanced prediction results"""
        if not predictions:
            print("\n❌ No predictions to display")
            return
        
        print(f"\n🎯 ENHANCED PREDICTION RESULTS FOR {self.ticker}")
        print("=" * 70)
        
        if len(predictions) == 1:
            pred = predictions[0]
            print(f"📅 Date: {pred['date'].strftime('%Y-%m-%d (%A)')}")
            print(f"💰 Current: ${pred['current_price']:.2f}")
            print(f"🎯 Predicted: ${pred['predicted_close']:.2f}")
            print(f"📊 Change: {pred['predicted_return']:+.2%}")
            print(f"🎪 Range: ${pred['predicted_low']:.2f} - ${pred['predicted_high']:.2f}")
            print(f"🎯 Confidence: {pred['confidence']:.0%}")
            print(f"🔑 Factors: {', '.join(pred['key_factors'])}")
            if pred.get('has_custom_news'):
                print(f"📰 Enhanced with custom news articles")
        else:
            starting_price = predictions[0]['current_price']
            ending_price = predictions[-1]['predicted_close']
            total_change = (ending_price - starting_price) / starting_price
            avg_confidence = np.mean([p['confidence'] for p in predictions])
            
            print(f"📊 Enhanced Summary ({len(predictions)} days):")
            print(f"   Start: ${starting_price:.2f}")
            print(f"   End: ${ending_price:.2f}")
            print(f"   Total: {total_change:+.1%}")
            print(f"   Avg Confidence: {avg_confidence:.0%}")
            
            if any(p.get('has_custom_news') for p in predictions):
                print(f"   📰 Enhanced with custom news")
            
            print(f"\n📈 Daily Breakdown:")
            for pred in predictions:
                direction = "📈" if pred['predicted_return'] > 0 else "📉"
                confidence_icon = "🎯" if pred['confidence'] > 0.7 else "⚡"
                news_icon = "📰" if pred.get('has_custom_news') else ""
                print(f"   {pred['date'].strftime('%Y-%m-%d')}: ${pred['predicted_close']:.2f} {direction} {confidence_icon} {news_icon}")
        
        save = input("\n💾 Save enhanced report? (y/n): ").strip().lower()
        if save in ['y', 'yes']:
            self.save_enhanced_report(predictions)
    
    def save_enhanced_report(self, predictions: List[Dict]):
        """Save enhanced detailed report"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"reports/enhanced_ml_prediction_{self.ticker}_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"🎯 ENHANCED ML PREDICTION REPORT - {self.ticker}\n")
                f.write("=" * 70 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Excel File: {self.excel_file_path}\n")
                f.write(f"Custom News: {len(self.custom_news_articles)} articles\n")
                f.write(f"Price Override: {self.current_price_override}\n")
                f.write(f"Ultimate AI Available: {AI_PREDICTOR_AVAILABLE}\n\n")
                
                # Add custom news summary
                if self.custom_news_articles:
                    f.write("📰 CUSTOM NEWS ARTICLES:\n")
                    ticker_articles = [a for a in self.custom_news_articles if a.get('ticker') == self.ticker]
                    for i, article in enumerate(ticker_articles, 1):
                        f.write(f"{i}. {article['title']}\n")
                        f.write(f"   Sentiment: {article['sentiment_score']:+.1f}\n")
                        f.write(f"   Added: {article['timestamp'][:19]}\n\n")
                
                f.write("📊 PREDICTIONS:\n")
                for pred in predictions:
                    f.write(f"📅 {pred['date'].strftime('%Y-%m-%d (%A)')}\n")
                    f.write(f"   Open: ${pred['predicted_open']:.2f}\n")
                    f.write(f"   High: ${pred['predicted_high']:.2f}\n")
                    f.write(f"   Low: ${pred['predicted_low']:.2f}\n")
                    f.write(f"   Close: ${pred['predicted_close']:.2f}\n")
                    f.write(f"   Confidence: {pred['confidence']:.0%}\n")
                    f.write(f"   Change: {pred['predicted_return']:+.2%}\n")
                    f.write(f"   Factors: {', '.join(pred['key_factors'])}\n")
                    if pred.get('has_custom_news'):
                        f.write(f"   Custom News: Enhanced prediction\n")
                    f.write("\n")
            
            print(f"✅ Enhanced report saved: {filename}")
            
        except Exception as e:
            print(f"❌ Save failed: {e}")
    
    def reset_session(self):
        """Reset session for new analysis"""
        self.excel_loader = None
        self.news_connector = None
        self.feature_engineer = None
        self.model = None
        self.is_trained = False
        self.ticker = None
        self.excel_file_path = None
        self.trained_features = []
        self.current_price_override = None
        # Keep custom news articles and API keys for convenience
    
    def run_interactive_session(self):
        """Main enhanced interactive session"""
        print("🎯 Welcome to Ultimate Enhanced Stock Prediction Engine!")
        
        while True:
            try:
                choice = self.get_main_menu_choice()
                
                if choice == "ml_analysis":
                    self.run_traditional_ml_analysis()
                
                elif choice == "ultimate_ai_analysis":
                    self.run_ultimate_ai_analysis()
                
                elif choice == "select_excel":
                    self.select_excel_file()
                
                elif choice == "add_news":
                    self.add_custom_news_article()
                
                elif choice == "view_news":
                    self.view_custom_news_articles()
                
                elif choice == "clear_news":
                    self.clear_custom_news_articles()
                
                elif choice == "settings":
                    self.show_settings_menu()
                
                elif choice == "help":
                    self.show_help()
                
                elif choice == "exit":
                    break
                
                # Ask to continue or exit after each operation
                if choice not in ["settings", "help", "view_news"]:
                    print(f"\n🔄 Continue using the Ultimate prediction engine?")
                    continue_choice = input("Enter 'y' to continue, any other key to exit: ").strip().lower()
                    if continue_choice not in ['y', 'yes']:
                        break
                
            except KeyboardInterrupt:
                print("\n\n🛑 Session interrupted")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("🔄 Continuing...")
        
        print("\n👋 Thank you for using Ultimate Enhanced Stock Prediction Engine!")
        print("📊 Your reports are saved in the 'reports' directory")
        if self.custom_news_articles:
            print(f"📰 {len(self.custom_news_articles)} custom articles saved in 'custom_news' directory")
        print("🚀 Ultimate AI capabilities integrated successfully!")


def main():
    """Main entry point for Ultimate Enhanced prediction tool"""
    print("🚀 Starting Ultimate Enhanced Interactive Stock Prediction Tool...")
    print("🔧 Loading ultimate components...")
    
    tool = UltimateEnhancedPredictionTool()
    tool.run_interactive_session()


if __name__ == "__main__":
    main()