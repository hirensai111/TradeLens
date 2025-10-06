# news_integration_bridge.py
"""
Bridge between new API-based news system and existing Excel report generator
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

from news_system.collectors.news_api_client import news_client
from news_system.analyzers.sentiment_analyzer import WorkingFinancialSentimentAnalyzer

class NewsIntegrationBridge:
    """Integrates new API-based news with existing sentiment analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.news_client = news_client
        self.sentiment_analyzer = WorkingFinancialSentimentAnalyzer()
        
    def get_news_with_sentiment(self, ticker: str, event_date: datetime, 
                               lookback_days: int = 3) -> tuple[List[Dict], Dict[str, Any]]:
        """
        Get news articles with sentiment analysis.
        Returns: (articles, analysis_summary)
        """
        # Fetch news from APIs
        self.logger.info(f"Fetching news for {ticker} around {event_date.date()}")
        articles = self.news_client.get_news(
            ticker=ticker,
            date=event_date,
            lookback_days=lookback_days,
            min_articles=3  # Lower threshold for better coverage
        )
        
        if not articles:
            self.logger.warning(f"No articles found for {ticker} on {event_date.date()}")
            return [], {"error": "No articles found"}
        
        # Add sentiment analysis to each article
        analyzed_articles = []
        for article in articles:
            # Analyze sentiment
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(
                text=article.get('summary', ''),
                title=article.get('title', '')
            )
            
            # Add sentiment to article
            article['sentiment'] = {
                'overall_sentiment': sentiment_result.overall_sentiment,
                'financial_sentiment': sentiment_result.financial_sentiment,
                'confidence': sentiment_result.confidence,
                'polarity': sentiment_result.polarity,
                'intensity': sentiment_result.sentiment_intensity,
                'key_phrases': sentiment_result.key_phrases
            }
            
            analyzed_articles.append(article)
        
        # Create analysis summary
        analysis_summary = self._create_analysis_summary(analyzed_articles, ticker, event_date)
        
        return analyzed_articles, analysis_summary
    
    def _create_analysis_summary(self, articles: List[Dict], ticker: str, 
                                event_date: datetime) -> Dict[str, Any]:
        """Create a summary of the news analysis"""
        if not articles:
            return {"error": "No articles to analyze"}
        
        # Aggregate sentiment data
        sentiments = [a['sentiment'] for a in articles if 'sentiment' in a]
        
        if not sentiments:
            return {"error": "No sentiment data available"}
        
        # Calculate overall metrics
        avg_confidence = sum(s['confidence'] for s in sentiments) / len(sentiments)
        avg_polarity = sum(s['polarity'] for s in sentiments) / len(sentiments)
        
        # Count sentiment types
        financial_sentiments = [s['financial_sentiment'] for s in sentiments]
        bullish_count = financial_sentiments.count('Bullish')
        bearish_count = financial_sentiments.count('Bearish')
        neutral_count = financial_sentiments.count('Neutral')
        
        # Determine overall sentiment
        if bullish_count > bearish_count + neutral_count:
            overall_sentiment = 'Bullish'
        elif bearish_count > bullish_count + neutral_count:
            overall_sentiment = 'Bearish'
        else:
            overall_sentiment = 'Mixed/Neutral'
        
        # Get key phrases
        all_key_phrases = []
        for s in sentiments:
            all_key_phrases.extend(s.get('key_phrases', []))
        
        # Count phrase frequency
        phrase_counts = {}
        for phrase in all_key_phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        # Get top phrases
        top_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        summary = {
            'ticker': ticker,
            'event_date': event_date.isoformat(),
            'total_articles': len(articles),
            'articles_analyzed': len(sentiments),
            'confidence': avg_confidence,
            'overall_sentiment': overall_sentiment,
            'sentiment_breakdown': {
                'bullish': bullish_count,
                'bearish': bearish_count,
                'neutral': neutral_count
            },
            'average_polarity': avg_polarity,
            'key_themes': [phrase for phrase, count in top_phrases],
            'sources': list(set(a.get('api_source', 'unknown') for a in articles)),
            'recommendation': self._generate_recommendation(
                overall_sentiment, avg_confidence, len(articles)
            )
        }
        
        return summary
    
    def _generate_recommendation(self, sentiment: str, confidence: float, 
                               article_count: int) -> str:
        """Generate a recommendation based on analysis"""
        if confidence < 0.5:
            return "Low confidence - Insufficient or unclear news sentiment"
        
        if article_count < 3:
            return "Limited news coverage - Consider additional research"
        
        if sentiment == 'Bullish':
            if confidence > 0.7:
                return "Strong positive sentiment - Potential upward catalyst"
            else:
                return "Moderate positive sentiment - Cautiously optimistic outlook"
        elif sentiment == 'Bearish':
            if confidence > 0.7:
                return "Strong negative sentiment - Potential downward catalyst"
            else:
                return "Moderate negative sentiment - Monitor for further developments"
        else:
            return "Mixed sentiment - No clear directional bias from news"

# Create global instance
news_bridge = NewsIntegrationBridge()