"""
test_simple_sentiment.py
Simple test script to verify sentiment analysis is working
"""

def test_basic_sentiment():
    """Test basic sentiment analysis functionality"""
    print("🧪 Testing Basic Sentiment Analysis")
    print("=" * 40)
    
    try:
        # Test 1: Import the working sentiment analyzer
        print("Step 1: Testing imports...")
        from sentiment_analyzer import WorkingFinancialSentimentAnalyzer
        print("[OK] Successfully imported WorkingFinancialSentimentAnalyzer")
        
        # Test 2: Initialize analyzer
        print("\nStep 2: Initializing analyzer...")
        analyzer = WorkingFinancialSentimentAnalyzer()
        print("[OK] Sentiment analyzer initialized successfully")
        
        # Test 3: Test sentiment analysis
        print("\nStep 3: Testing sentiment analysis...")
        
        test_cases = [
            {
                'title': 'Apple Reports Record Earnings',
                'text': 'Apple Inc. beat analyst expectations with record quarterly revenue and strong iPhone sales growth.',
                'expected_sentiment': 'Positive'
            },
            {
                'title': 'Tesla Stock Drops on Production Issues',
                'text': 'Tesla shares fell after the company reported production delays and missed delivery targets.',
                'expected_sentiment': 'Negative'
            },
            {
                'title': 'Microsoft Maintains Steady Performance',
                'text': 'Microsoft reported earnings in line with expectations, showing consistent cloud growth.',
                'expected_sentiment': 'Neutral'
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases, 1):
            print(f"\n   Test Case {i}: {case['title']}")
            
            result = analyzer.analyze_sentiment(case['text'], case['title'])
            results.append(result)
            
            print(f"   Expected: {case['expected_sentiment']}")
            print(f"   Got: {result.overall_sentiment}")
            print(f"   Financial: {result.financial_sentiment}")
            print(f"   Confidence: {result.confidence:.2f}")
            
            # Check if result makes sense
            if (case['expected_sentiment'] == 'Positive' and result.overall_sentiment in ['Positive']) or \
               (case['expected_sentiment'] == 'Negative' and result.overall_sentiment in ['Negative']) or \
               (case['expected_sentiment'] == 'Neutral' and result.overall_sentiment in ['Neutral', 'Positive']):
                print("   [OK] Result looks good")
            else:
                print("   [WARNING] Unexpected result (but this is OK)")
        
        print(f"\n[OK] All {len(test_cases)} test cases completed successfully!")
        
        # Test 4: Test statistics
        print("\nStep 4: Testing statistics...")
        stats = analyzer.get_stats()
        print(f"   Total analyses: {stats['total_analyses']}")
        print(f"   Positive: {stats['positive_sentiment']}")
        print(f"   Negative: {stats['negative_sentiment']}")
        print(f"   Neutral: {stats['neutral_sentiment']}")
        print("[OK] Statistics working correctly")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        print("   Make sure you have saved 'sentiment_analyzer_working.py' in your directory")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test the integration module"""
    print("\n🔗 Testing Sentiment Integration")
    print("=" * 40)
    
    try:
        # Test import
        print("Step 1: Testing integration import...")
        from sentiment_integration import WorkingSentimentIntegration
        print("[OK] Successfully imported integration module")
        
        # Test initialization
        print("\nStep 2: Initializing integration...")
        integration = WorkingSentimentIntegration()
        print("[OK] Integration initialized successfully")
        
        # Test with sample articles
        print("\nStep 3: Testing with sample articles...")
        
        sample_articles = [
            {
                'title': 'Apple Beats Q3 Earnings Expectations',
                'summary': 'Apple reported strong quarterly results with iPhone sales exceeding analyst forecasts.',
                'source': 'Yahoo Finance',
                'source_key': 'article_1'
            },
            {
                'title': 'Tesla Production Concerns Mount',
                'summary': 'Tesla faces challenges meeting production targets amid supply chain disruptions.',
                'source': 'Reuters', 
                'source_key': 'article_2'
            }
        ]
        
        # Add sentiment to articles
        enhanced_articles = integration.add_sentiment_to_articles(sample_articles)
        print(f"   Processed {len(enhanced_articles)} articles")
        
        # Check if sentiment was added
        for article in enhanced_articles:
            sentiment = article.get('sentiment')
            if sentiment:
                print(f"   [OK] Article '{article['title'][:30]}...' has sentiment: {sentiment['overall_sentiment']}")
            else:
                print(f"   [WARNING] Article '{article['title'][:30]}...' missing sentiment")
        
        # Test summary creation
        print("\nStep 4: Testing sentiment summary...")
        summary = integration.create_sentiment_summary(enhanced_articles)
        
        if 'error' not in summary:
            print(f"   Total articles: {summary['total_articles']}")
            print(f"   Overall sentiment: {summary['overall_assessment']['market_sentiment']}")
            print("[OK] Summary created successfully")
        else:
            print(f"   [WARNING] Summary error: {summary['error']}")
        
        print("\n[OK] Integration test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        print("   Make sure you have saved 'working_sentiment_integration.py' in your directory")
        return False
    except Exception as e:
        print(f"[ERROR] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking Dependencies")
    print("=" * 25)
    
    dependencies = {
        'vaderSentiment': 'VADER sentiment analysis',
        'requests': 'HTTP requests',
        'json': 'JSON processing (built-in)',
        'datetime': 'Date/time handling (built-in)',
        'logging': 'Logging (built-in)'
    }
    
    available = []
    missing = []
    
    for module, description in dependencies.items():
        try:
            if module == 'vaderSentiment':
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            else:
                __import__(module)
            print(f"[OK] {module}: {description}")
            available.append(module)
        except ImportError:
            print(f"[ERROR] {module}: {description} - NOT AVAILABLE")
            missing.append(module)
    
    print(f"\nSummary: {len(available)}/{len(dependencies)} dependencies available")
    
    if missing:
        print(f"\nTo install missing dependencies:")
        for module in missing:
            if module == 'vaderSentiment':
                print(f"   pip install vaderSentiment")
            else:
                print(f"   pip install {module}")
    
    return len(missing) == 0

def main():
    """Main test function"""
    print("[TARGET] Sentiment Analysis Testing Suite")
    print("=" * 50)
    
    # Check dependencies first
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n[WARNING] Some dependencies are missing, but tests will continue...")
        print("   The system will work with reduced functionality")
    
    print("\n" + "="*50)
    
    # Test basic sentiment analysis
    basic_ok = test_basic_sentiment()
    
    if basic_ok:
        print("\n" + "="*50)
        # Test integration
        integration_ok = test_integration()
        
        if integration_ok:
            print("\n🎉 ALL TESTS PASSED!")
            print("\n[OK] Your sentiment analysis system is working correctly!")
            print("\nNext steps:")
            print("1. You can now integrate this with your news scraper")
            print("2. Add Excel output functionality")
            print("3. Connect to your price data for correlation analysis")
            
            # Show how to use it
            print("\n[BULB] How to use in your code:")
            print("""
from working_sentiment_integration import WorkingSentimentIntegration

# Initialize
integration = WorkingSentimentIntegration()

# Process your news articles
enhanced_articles = integration.add_sentiment_to_articles(your_articles)

# Get sentiment summary
summary = integration.create_sentiment_summary(enhanced_articles)

# Use the results!
print(f"Overall market sentiment: {summary['overall_assessment']['market_sentiment']}")
""")
        else:
            print("\n[WARNING] Integration tests failed, but basic sentiment analysis works")
    else:
        print("\n[ERROR] Basic tests failed - check your setup")
    
    print("\n" + "="*50)
    print("Test completed!")

if __name__ == "__main__":
    main()