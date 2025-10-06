"""
Integration test for processing pipeline with collected news articles.

This script demonstrates how to process articles from the database using
the text processor and sentiment analyzer.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import get_session
from src.processors import process_article, get_market_sentiment_overview

def test_processing_pipeline():
    """Test the processing pipeline with real articles from database."""
    
    print("=== Processing Pipeline Integration Test ===\n")
    
    # Get articles from database
    session = get_session()
    articles = session.get_articles(limit=10)  # Get 10 recent articles
    
    if not articles:
        print("No articles found in database. Run news collection first!")
        return
    
    print(f"Found {len(articles)} articles to process\n")
    
    # Process each article
    processed_articles = []
    for i, article in enumerate(articles, 1):
        print(f"Processing Article {i}: {article.title[:60]}...")
        
        try:
            # Process the article
            result = process_article(article.title, article.content)
            processed_articles.append((article.title, article.content))
            
            # Display key results
            text_analysis = result['text_analysis']
            sentiment_analysis = result['sentiment_analysis']
            
            print(f"  Keywords: {text_analysis['keywords'][:5]}")  # First 5 keywords
            print(f"  Stock Symbols: {text_analysis['stock_symbols']}")
            print(f"  Sentiment: {sentiment_analysis['sentiment_label']} ({sentiment_analysis['sentiment_score']:.3f})")
            print(f"  Market Sentiment: {sentiment_analysis['market_sentiment']:.3f}")
            print(f"  Confidence: {sentiment_analysis['confidence']:.3f}")
            print(f"  Urgency Score: {text_analysis['urgency_score']:.3f}")
            
            # Update article in database with processed data
            session.update_article(
                article.id,
                sentiment_score=sentiment_analysis['sentiment_score'],
                sentiment_label=sentiment_analysis['sentiment_label'],
                keywords=text_analysis['keywords'],
                stock_symbols=text_analysis['stock_symbols'],
                processed=True
            )
            
            print("  ✅ Article processed and updated in database\n")
            
        except Exception as e:
            print(f"  ❌ Error processing article: {e}\n")
    
    # Get market sentiment overview
    if processed_articles:
        print("=== Market Sentiment Overview ===")
        overview = get_market_sentiment_overview(processed_articles)
        
        print(f"Overall Sentiment: {overview.get('overall_sentiment', 0):.3f}")
        print(f"Market Sentiment: {overview.get('market_sentiment', 0):.3f}")
        print(f"Confidence: {overview.get('confidence', 0):.3f}")
        print(f"Positive Ratio: {overview.get('positive_ratio', 0):.3f}")
        print(f"Risk Level: {overview.get('risk_level', 'unknown')}")
        
        sentiment_dist = overview.get('sentiment_distribution', {})
        print(f"Sentiment Distribution:")
        print(f"  Positive: {sentiment_dist.get('positive', 0)}")
        print(f"  Negative: {sentiment_dist.get('negative', 0)}")
        print(f"  Neutral: {sentiment_dist.get('neutral', 0)}")
    
    print("\n=== Processing Pipeline Test Complete ===")

if __name__ == "__main__":
    test_processing_pipeline()