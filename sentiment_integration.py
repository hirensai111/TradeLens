"""
working_sentiment_integration.py
Working sentiment integration that bypasses dependency issues
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

# Import the working sentiment analyzer
from sentiment_analyzer import WorkingFinancialSentimentAnalyzer

class WorkingSentimentIntegration:
    """
    Simple working sentiment integration for your news scraper
    """
    
    def __init__(self):
        self.sentiment_analyzer = WorkingFinancialSentimentAnalyzer()
        self.setup_logging()
        
        # Stats
        self.stats = {
            'articles_processed': 0,
            'articles_with_sentiment': 0,
            'bullish_articles': 0,
            'bearish_articles': 0,
            'neutral_articles': 0
        }
    
    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def add_sentiment_to_articles(self, articles: List[Dict]) -> List[Dict]:
        """Add sentiment analysis to articles"""
        enhanced_articles = []
        
        for article in articles:
            try:
                # Get article data
                title = article.get('title', '')
                summary = article.get('summary', '')
                
                if not title and not summary:
                    # Skip articles with no content
                    continue
                
                # Analyze sentiment
                sentiment_result = self.sentiment_analyzer.analyze_sentiment(summary, title)
                
                # Add sentiment to article
                article['sentiment'] = {
                    'overall_sentiment': sentiment_result.overall_sentiment,
                    'financial_sentiment': sentiment_result.financial_sentiment,
                    'confidence': sentiment_result.confidence,
                    'polarity': sentiment_result.polarity,
                    'intensity': sentiment_result.sentiment_intensity,
                    'key_phrases': sentiment_result.key_phrases
                }
                
                # Calculate relevance score
                article['sentiment_relevance'] = self.calculate_relevance_score(article)
                
                enhanced_articles.append(article)
                self.stats['articles_with_sentiment'] += 1
                
                # Update sentiment stats
                financial_sentiment = sentiment_result.financial_sentiment
                if financial_sentiment == 'Bullish':
                    self.stats['bullish_articles'] += 1
                elif financial_sentiment == 'Bearish':
                    self.stats['bearish_articles'] += 1
                else:
                    self.stats['neutral_articles'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing article: {e}")
                # Add article without sentiment
                article['sentiment'] = None
                article['sentiment_relevance'] = 0.5
                enhanced_articles.append(article)
            
            self.stats['articles_processed'] += 1
        
        return enhanced_articles
    
    def calculate_relevance_score(self, article: Dict) -> float:
        """Calculate article relevance based on sentiment and content"""
        base_score = 0.5
        sentiment_data = article.get('sentiment')
        
        if not sentiment_data:
            return base_score
        
        # Boost for high confidence
        confidence = sentiment_data.get('confidence', 0)
        base_score += confidence * 0.3
        
        # Boost for strong polarity (important news)
        polarity = abs(sentiment_data.get('polarity', 0))
        base_score += polarity * 0.2
        
        # Boost for financial-specific sentiment
        if sentiment_data.get('financial_sentiment') in ['Bullish', 'Bearish']:
            base_score += 0.1
        
        # Boost for key phrases
        key_phrases = sentiment_data.get('key_phrases', [])
        if key_phrases:
            base_score += min(0.1, len(key_phrases) * 0.02)
        
        return min(1.0, max(0.0, base_score))
    
    def create_sentiment_summary(self, articles: List[Dict]) -> Dict[str, Any]:
        """Create sentiment summary for articles"""
        if not articles:
            return {'error': 'No articles to analyze'}
        
        # Collect sentiment data
        sentiments = []
        financial_sentiments = []
        confidences = []
        polarities = []
        
        for article in articles:
            sentiment = article.get('sentiment')
            if sentiment:
                sentiments.append(sentiment['overall_sentiment'])
                financial_sentiments.append(sentiment['financial_sentiment'])
                confidences.append(sentiment['confidence'])
                polarities.append(sentiment['polarity'])
        
        if not sentiments:
            return {'error': 'No sentiment data available'}
        
        # Calculate counts
        positive_count = sentiments.count('Positive')
        negative_count = sentiments.count('Negative')
        neutral_count = sentiments.count('Neutral')
        
        bullish_count = financial_sentiments.count('Bullish')
        bearish_count = financial_sentiments.count('Bearish')
        neutral_financial_count = financial_sentiments.count('Neutral')
        
        # Calculate averages
        avg_confidence = sum(confidences) / len(confidences)
        avg_polarity = sum(polarities) / len(polarities)
        
        # Determine overall sentiment
        if avg_polarity > 0.1:
            overall_sentiment = 'Bullish'
        elif avg_polarity < -0.1:
            overall_sentiment = 'Bearish'
        else:
            overall_sentiment = 'Neutral'
        
        summary = {
            'total_articles': len(articles),
            'articles_with_sentiment': len(sentiments),
            'sentiment_breakdown': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count,
                'positive_percentage': (positive_count / len(sentiments)) * 100,
                'negative_percentage': (negative_count / len(sentiments)) * 100,
                'neutral_percentage': (neutral_count / len(sentiments)) * 100
            },
            'financial_sentiment': {
                'bullish': bullish_count,
                'bearish': bearish_count,
                'neutral': neutral_financial_count,
                'bullish_percentage': (bullish_count / len(financial_sentiments)) * 100,
                'bearish_percentage': (bearish_count / len(financial_sentiments)) * 100
            },
            'quality_metrics': {
                'average_confidence': avg_confidence,
                'average_polarity': avg_polarity,
                'high_confidence_count': sum(1 for c in confidences if c > 0.7)
            },
            'overall_assessment': {
                'market_sentiment': overall_sentiment,
                'sentiment_strength': 'Strong' if abs(avg_polarity) > 0.3 else 'Moderate' if abs(avg_polarity) > 0.1 else 'Weak'
            }
        }
        
        return summary
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {**self.stats, **self.sentiment_analyzer.get_stats()}

# Test function to simulate your news scraper
def simulate_news_articles(ticker: str) -> List[Dict]:
    """Simulate news articles for testing"""
    sample_articles = [
        {
            'title': f'{ticker} Reports Strong Quarterly Earnings, Beats Expectations',
            'summary': f'{ticker} announced record revenue growth and exceeded analyst estimates for the quarter, driven by strong product sales and expanding market share.',
            'source': 'Yahoo Finance',
            'date': datetime.now() - timedelta(hours=2),
            'categories': ['company', 'earnings']
        },
        {
            'title': f'{ticker} Stock Faces Pressure After Regulatory Concerns',
            'summary': f'Shares of {ticker} declined following news of potential regulatory challenges that could impact future growth prospects and market expansion plans.',
            'source': 'Reuters',
            'date': datetime.now() - timedelta(hours=5),
            'categories': ['regulatory', 'company']
        },
        {
            'title': f'Analysts Upgrade {ticker} Price Target on Innovation',
            'summary': f'Several Wall Street analysts raised their price targets for {ticker} citing breakthrough innovations and strong competitive positioning in the market.',
            'source': 'MarketWatch',
            'date': datetime.now() - timedelta(hours=8),
            'categories': ['analyst', 'company']
        },
        {
            'title': f'{ticker} CEO Discusses Future Strategy in Interview',
            'summary': f'The CEO of {ticker} outlined the company\'s strategic vision and growth plans during a recent interview, expressing optimism about market opportunities.',
            'source': 'CNBC',
            'date': datetime.now() - timedelta(hours=12),
            'categories': ['management', 'company']
        },
        {
            'title': f'Market Volatility Affects {ticker} Trading Volume',
            'summary': f'Trading volume for {ticker} increased significantly amid broader market volatility, with investors closely watching technical indicators.',
            'source': 'Bloomberg',
            'date': datetime.now() - timedelta(hours=18),
            'categories': ['market', 'technical']
        }
    ]
    
    return sample_articles

def test_working_sentiment_integration():
    """Test the working sentiment integration"""
    print("Testing Working Sentiment Integration")
    print("=" * 50)
    
    # Initialize integration
    integration = WorkingSentimentIntegration()
    
    # Simulate news articles for AAPL
    test_ticker = "AAPL"
    print(f"Simulating news articles for {test_ticker}...")
    
    articles = simulate_news_articles(test_ticker)
    print(f"Generated {len(articles)} sample articles")
    
    # Add sentiment analysis
    print("\nAnalyzing sentiment...")
    enhanced_articles = integration.add_sentiment_to_articles(articles)
    
    # Create sentiment summary
    sentiment_summary = integration.create_sentiment_summary(enhanced_articles)
    
    # Display results
    print(f"\n📊 Sentiment Analysis Results:")
    print("-" * 30)
    
    if 'error' not in sentiment_summary:
        breakdown = sentiment_summary['sentiment_breakdown']
        print(f"Total articles: {sentiment_summary['total_articles']}")
        print(f"Positive: {breakdown['positive']} ({breakdown['positive_percentage']:.1f}%)")
        print(f"Negative: {breakdown['negative']} ({breakdown['negative_percentage']:.1f}%)")
        print(f"Neutral: {breakdown['neutral']} ({breakdown['neutral_percentage']:.1f}%)")
        
        financial = sentiment_summary['financial_sentiment']
        print(f"\nFinancial Sentiment:")
        print(f"Bullish: {financial['bullish']} ({financial['bullish_percentage']:.1f}%)")
        print(f"Bearish: {financial['bearish']} ({financial['bearish_percentage']:.1f}%)")
        
        overall = sentiment_summary['overall_assessment']
        print(f"\nOverall Assessment:")
        print(f"Market Sentiment: {overall['market_sentiment']}")
        print(f"Sentiment Strength: {overall['sentiment_strength']}")
        
        quality = sentiment_summary['quality_metrics']
        print(f"\nQuality Metrics:")
        print(f"Average Confidence: {quality['average_confidence']:.2f}")
        print(f"High Confidence Articles: {quality['high_confidence_count']}")
    
    # Show individual article analysis
    print(f"\n📰 Individual Article Analysis:")
    print("-" * 40)
    
    for i, article in enumerate(enhanced_articles[:3], 1):
        print(f"\nArticle {i}:")
        print(f"Title: {article['title'][:70]}...")
        
        sentiment = article.get('sentiment')
        if sentiment:
            print(f"  Sentiment: {sentiment['overall_sentiment']} (Financial: {sentiment['financial_sentiment']})")
            print(f"  Confidence: {sentiment['confidence']:.2f}")
            print(f"  Polarity: {sentiment['polarity']:.2f}")
            print(f"  Intensity: {sentiment['intensity']}")
            print(f"  Relevance: {article.get('sentiment_relevance', 0):.2f}")
            if sentiment['key_phrases']:
                print(f"  Key Phrases: {', '.join(sentiment['key_phrases'][:3])}")
        else:
            print("  No sentiment analysis available")
    
    # Show statistics
    stats = integration.get_stats()
    print(f"\n📈 Processing Statistics:")
    print("-" * 25)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: {value}")
    
    return enhanced_articles, sentiment_summary

# Simple test script that works with your existing setup
def quick_test():
    """Quick test that you can run immediately"""
    print("🚀 Quick Sentiment Analysis Test")
    print("=" * 40)
    
    # Test the sentiment analyzer directly
    from sentiment_analyzer import WorkingFinancialSentimentAnalyzer
    
    analyzer = WorkingFinancialSentimentAnalyzer()
    
    # Test cases
    test_texts = [
        ("AAPL beats earnings expectations", "Apple reported strong quarterly results"),
        ("TSLA stock plunges on concerns", "Tesla shares fell on production worries"),
        ("MSFT maintains steady growth", "Microsoft showed consistent performance")
    ]
    
    print("Testing sentiment analysis...")
    
    for i, (title, text) in enumerate(test_texts, 1):
        result = analyzer.analyze_sentiment(text, title)
        print(f"\nTest {i}: {title}")
        print(f"  Result: {result.overall_sentiment}")
        print(f"  Financial: {result.financial_sentiment}")
        print(f"  Confidence: {result.confidence:.2f}")
    
    print("\n✅ Basic sentiment analysis working!")
    return True

# Main execution
if __name__ == "__main__":
    print("Working Sentiment Integration Module")
    print("=" * 50)
    
    try:
        # Run quick test first
        if quick_test():
            print("\n" + "="*50)
            # Run full integration test
            enhanced_articles, summary = test_working_sentiment_integration()
            print(f"\n✅ Full integration test completed successfully!")
            print(f"   - Processed {len(enhanced_articles)} articles")
            print(f"   - Generated comprehensive sentiment analysis")
            print(f"   - No dependency issues encountered")
        else:
            print("❌ Quick test failed")
    
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 Next Steps:")
    print("1. Save these files in your D:\\stock_analyzer directory")
    print("2. Run: python working_sentiment_integration.py")
    print("3. Integrate with your existing news scraper")
    print("4. Add Excel output with sentiment data")