# test_news_client.py
from datetime import datetime, timedelta
from news_api_client import news_client

def test_news_client():
    print("Testing Integrated News API Client")
    print("="*60)
    
    test_cases = [
        {
            'name': 'Today\'s News',
            'ticker': 'AAPL',
            'date': datetime.now(),
            'lookback_days': 1,
        },
        {
            'name': 'Last Week\'s News',
            'ticker': 'MSFT',
            'date': datetime.now() - timedelta(days=7),
            'lookback_days': 3,
        },
        {
            'name': 'Historical News (2 months ago)',
            'ticker': 'GOOGL',
            'date': datetime.now() - timedelta(days=60),
            'lookback_days': 5,
        }
    ]
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print("-" * 40)
        
        articles = news_client.get_news(
            ticker=test['ticker'],
            date=test['date'],
            lookback_days=test['lookback_days']
        )
        
        print(f"Ticker: {test['ticker']}")
        print(f"Date: {test['date'].strftime('%Y-%m-%d')}")
        print(f"Articles found: {len(articles)}")
        
        if articles:
            # Show sources distribution
            sources = {}
            for article in articles:
                source = article.get('api_source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            
            print("Sources:", sources)
            
            # Show sample article
            article = articles[0]
            print(f"\nSample article:")
            print(f"  Title: {article['title'][:70]}...")
            print(f"  Date: {article['published_date']}")
            print(f"  Source: {article['source']}")
            print(f"  API: {article['api_source']}")
    
    # Show statistics
    print(f"\n{'='*60}")
    print("API Usage Statistics:")
    stats = news_client.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_news_client()