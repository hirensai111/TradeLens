"""
Processing pipeline package for Phase 3 News Intelligence Engine.

This package provides text processing, sentiment analysis, event extraction,
and content analysis capabilities for news intelligence.
"""

from .text_processor import TextProcessor, TextAnalysisResult
from .sentiment_analyzer import FinancialSentimentAnalyzer, SentimentResult
from .event_extractor import FinancialEventExtractor, ExtractedEvent, EventExtractionResult

__version__ = "1.0.0"
__all__ = [
    # Text processing
    "TextProcessor",
    "TextAnalysisResult",
    
    # Sentiment analysis
    "FinancialSentimentAnalyzer", 
    "SentimentResult",
    
    # Event extraction
    "FinancialEventExtractor",
    "ExtractedEvent", 
    "EventExtractionResult"
]

# Package-level convenience functions
def process_article(title: str, content: str) -> dict:
    """
    Process a single article with both text analysis and sentiment analysis.
    
    Args:
        title: Article title
        content: Article content
        
    Returns:
        dict: Combined analysis results
    """
    # Initialize processors
    text_processor = TextProcessor()
    sentiment_analyzer = FinancialSentimentAnalyzer()
    
    # Perform analyses
    text_result = text_processor.process_text(content, title)
    sentiment_result = sentiment_analyzer.analyze_sentiment(content, title)
    
    return {
        'text_analysis': {
            'keywords': text_result.keywords,
            'entities': text_result.entities,
            'financial_terms': text_result.financial_terms,
            'stock_symbols': text_result.stock_symbols,
            'confidence_indicators': text_result.confidence_indicators,
            'readability_score': text_result.readability_score,
            'urgency_score': text_result.urgency_score,
            'word_count': text_result.word_count
        },
        'sentiment_analysis': {
            'sentiment_score': sentiment_result.sentiment_score,
            'sentiment_label': sentiment_result.sentiment_label,
            'market_sentiment': sentiment_result.market_sentiment,
            'risk_sentiment': sentiment_result.risk_sentiment,
            'confidence': sentiment_result.confidence,
            'emotional_indicators': sentiment_result.emotional_indicators
        },
        'processing_metadata': {
            'processing_time': sentiment_result.processing_time,
            'analysis_method': sentiment_result.analysis_method,
            'text_length': text_result.text_length
        }
    }

def batch_process_articles(articles: list) -> list:
    """
    Process multiple articles efficiently.
    
    Args:
        articles: List of (title, content) tuples
        
    Returns:
        list: Analysis results for each article
    """
    results = []
    
    # Initialize processors once
    text_processor = TextProcessor()
    sentiment_analyzer = FinancialSentimentAnalyzer()
    
    for title, content in articles:
        try:
            result = process_article(title, content)
            results.append({
                'title': title,
                'success': True,
                'analysis': result
            })
        except Exception as e:
            results.append({
                'title': title,
                'success': False,
                'error': str(e)
            })
    
    return results

def get_market_sentiment_overview(articles: list) -> dict:
    """
    Get overall market sentiment from a collection of articles.
    
    Args:
        articles: List of (title, content) tuples
        
    Returns:
        dict: Market sentiment overview
    """
    sentiment_analyzer = FinancialSentimentAnalyzer()
    
    # Analyze all articles
    sentiment_results = []
    for title, content in articles:
        try:
            result = sentiment_analyzer.analyze_sentiment(content, title)
            sentiment_results.append(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to analyze sentiment for '{title}': {e}")
    
    if not sentiment_results:
        return {'error': 'No articles could be analyzed'}
    
    # Get market summary
    summary = sentiment_analyzer.get_market_sentiment_summary(sentiment_results)
    
    # Add additional insights
    summary['sentiment_distribution'] = {
        'positive': len([r for r in sentiment_results if r.sentiment_label == 'positive']),
        'negative': len([r for r in sentiment_results if r.sentiment_label == 'negative']),
        'neutral': len([r for r in sentiment_results if r.sentiment_label == 'neutral'])
    }
    
    # Calculate risk level
    avg_risk = sum(r.risk_sentiment for r in sentiment_results) / len(sentiment_results)
    if avg_risk > 0.3:
        risk_level = 'high'
    elif avg_risk > 0.0:
        risk_level = 'medium'
    else:
        risk_level = 'low'
    
    summary['risk_level'] = risk_level
    summary['avg_risk_sentiment'] = avg_risk
    
    return summary

# Convenience processors for common use cases
def quick_sentiment(text: str) -> dict:
    """Quick sentiment analysis for a text snippet."""
    analyzer = FinancialSentimentAnalyzer()
    result = analyzer.analyze_sentiment(text)
    
    return {
        'sentiment': result.sentiment_label,
        'score': result.sentiment_score,
        'confidence': result.confidence
    }

def extract_financial_info(text: str) -> dict:
    """Quick extraction of financial information from text."""
    processor = TextProcessor()
    result = processor.process_text(text)
    
    return {
        'stock_symbols': result.stock_symbols,
        'financial_terms': result.financial_terms,
        'dollar_amounts': result.entities.get('dollar_amounts', []),
        'percentages': result.entities.get('percentages', [])
    }

# Initialize logging for the package
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())