# test_integration.py
from datetime import datetime, timedelta
from news_integration_bridge import news_bridge

def test_integration():
    print("Testing Complete News Integration")
    print("="*60)
    
    # Test cases that match your stock analyzer scenarios
    test_cases = [
        {
            'ticker': 'MSFT',
            'date': datetime(2024, 4, 30),  # From your logs
            'expected': 'Historical analysis'
        },
        {
            'ticker': 'AAPL',
            'date': datetime.now() - timedelta(days=1),
            'expected': 'Recent news analysis'
        }
    ]
    
    for test in test_cases:
        print(f"\nTest: {test['ticker']} on {test['date'].date()}")
        print("-" * 40)
        
        articles, summary = news_bridge.get_news_with_sentiment(
            ticker=test['ticker'],
            event_date=test['date']
        )
        
        print(f"Articles found: {len(articles)}")
        
        if summary.get('error'):
            print(f"Error: {summary['error']}")
        else:
            print(f"Overall sentiment: {summary['overall_sentiment']}")
            print(f"Confidence: {summary['confidence']:.2f}")
            print(f"Recommendation: {summary['recommendation']}")
            print(f"Key themes: {', '.join(summary['key_themes'][:3])}")
            
            # Show sentiment breakdown
            breakdown = summary['sentiment_breakdown']
            print(f"Sentiment breakdown: Bullish={breakdown['bullish']}, "
                  f"Bearish={breakdown['bearish']}, Neutral={breakdown['neutral']}")

if __name__ == "__main__":
    test_integration()