"""
sentiment_analyzer_working.py
A working sentiment analyzer that bypasses dependency issues
Uses only VADER + custom financial lexicon (no TextBlob/spaCy/sklearn)
"""

import re
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Try to import VADER, fallback if not available
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

@dataclass
class SentimentResult:
    """Data class for sentiment analysis results"""
    overall_sentiment: str  # 'Positive', 'Negative', 'Neutral'
    confidence: float  # 0.0 to 1.0
    polarity: float  # -1.0 to 1.0
    subjectivity: float  # 0.0 to 1.0
    financial_sentiment: str  # 'Bullish', 'Bearish', 'Neutral'
    vader_scores: Dict[str, float]
    key_phrases: List[str]
    sentiment_intensity: str  # 'Strong', 'Moderate', 'Weak'
    sentiment_score: float  # Combined sentiment score (0-100 scale)
    analyzed_at: datetime

class WorkingFinancialSentimentAnalyzer:
    """
    Working sentiment analyzer - no problematic dependencies
    """
    
    def __init__(self):
        self.setup_logging()
        self.initialize_analyzers()
        self.load_financial_lexicon()
        
        # Statistics tracking
        self.stats = {
            'total_analyses': 0,
            'positive_sentiment': 0,
            'negative_sentiment': 0,
            'neutral_sentiment': 0,
            'high_confidence_analyses': 0,
            'financial_terms_detected': 0
        }
    
    def setup_logging(self):
        """Setup logging for sentiment analysis"""
        self.logger = logging.getLogger(__name__)
    
    def initialize_analyzers(self):
        """Initialize available sentiment analyzers"""
        self.analyzers = {}
        
        # Initialize VADER if available
        if VADER_AVAILABLE:
            self.analyzers['vader'] = SentimentIntensityAnalyzer()
            self.logger.info("VADER sentiment analyzer initialized")
        else:
            self.logger.warning("VADER not available - using financial lexicon only")
    
    def load_financial_lexicon(self):
        """Load financial-specific sentiment words and phrases"""
        self.financial_lexicon = {
            'positive': {
                'strong': [
                    'surge', 'soar', 'rally', 'boom', 'breakthrough', 'record-high', 
                    'outperform', 'beat expectations', 'exceeded estimates', 'blowout earnings',
                    'strong growth', 'record profit', 'stellar performance', 'crushing it'
                ],
                'moderate': [
                    'rise', 'gain', 'up', 'increase', 'growth', 'positive', 'bullish', 
                    'optimistic', 'improved', 'better than expected', 'solid results',
                    'good news', 'upbeat', 'encouraging'
                ],
                'weak': [
                    'slight increase', 'marginal gain', 'modest growth', 'cautiously optimistic',
                    'mild improvement', 'gradual increase'
                ]
            },
            'negative': {
                'strong': [
                    'crash', 'plunge', 'collapse', 'tumble', 'plummet', 'disaster', 
                    'catastrophic', 'bankruptcy', 'massive loss', 'devastating',
                    'terrible results', 'worst performance', 'major disappointment'
                ],
                'moderate': [
                    'fall', 'drop', 'decline', 'loss', 'down', 'bearish', 'pessimistic', 
                    'concern', 'worried', 'disappointing', 'missed expectations',
                    'weak results', 'poor performance', 'troubling'
                ],
                'weak': [
                    'slight decline', 'marginal loss', 'modest decrease', 'cautious',
                    'mild concern', 'minor setback'
                ]
            },
            'neutral': [
                'stable', 'unchanged', 'flat', 'sideways', 'mixed', 'neutral', 
                'steady', 'maintained', 'consistent', 'as expected'
            ]
        }
        
        # Financial impact amplifiers
        self.financial_amplifiers = {
            'high_impact': [
                'earnings', 'revenue', 'profit', 'loss', 'merger', 'acquisition', 
                'ipo', 'bankruptcy', 'ceo', 'management change'
            ],
            'medium_impact': [
                'guidance', 'forecast', 'outlook', 'target', 'rating', 'upgrade', 
                'downgrade', 'analyst', 'recommendation'
            ],
            'market_terms': [
                'bull market', 'bear market', 'correction', 'volatility', 'index', 
                'sector', 'fed', 'interest rate', 'inflation'
            ]
        }
        
        # Create combined word lists for quick lookup
        self.all_positive_words = []
        self.all_negative_words = []
        
        for intensity in self.financial_lexicon['positive'].values():
            self.all_positive_words.extend(intensity)
        
        for intensity in self.financial_lexicon['negative'].values():
            self.all_negative_words.extend(intensity)
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis"""
        if not text:
            return ""
        
        # Convert to lowercase for processing
        processed_text = text.lower()
        
        # Remove extra whitespace
        processed_text = re.sub(r'\s+', ' ', processed_text.strip())
        
        # Handle financial abbreviations
        financial_replacements = {
            'q1': 'first quarter', 'q2': 'second quarter', 'q3': 'third quarter', 'q4': 'fourth quarter',
            'yoy': 'year over year', 'qoq': 'quarter over quarter',
            'eps': 'earnings per share', 'pe': 'price to earnings',
            'ipo': 'initial public offering', 'ceo': 'chief executive officer',
            'cfo': 'chief financial officer', 'cto': 'chief technology officer'
        }
        
        for abbr, expansion in financial_replacements.items():
            processed_text = processed_text.replace(f' {abbr} ', f' {expansion} ')
            processed_text = processed_text.replace(f' {abbr}.', f' {expansion}.')
        
        return processed_text
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key financial phrases from text"""
        key_phrases = []
        text_lower = text.lower()
        
        # Look for financial phrases
        financial_phrases = [
            'beat expectations', 'miss expectations', 'exceeded estimates', 'fell short',
            'raised guidance', 'lowered guidance', 'strong quarter', 'weak quarter',
            'record revenue', 'record profit', 'market share', 'competitive advantage',
            'supply chain', 'demand growth', 'price increase', 'cost reduction',
            'new product', 'product launch', 'partnership', 'acquisition'
        ]
        
        for phrase in financial_phrases:
            if phrase in text_lower:
                key_phrases.append(phrase)
        
        # Extract quoted text (often contains key information)
        quotes = re.findall(r'"([^"]*)"', text)
        for quote in quotes:
            if len(quote) > 10 and len(quote) < 100:  # Reasonable quote length
                key_phrases.append(f'"{quote}"')
        
        return key_phrases[:5]  # Limit to top 5 phrases
    
    def analyze_with_vader(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER"""
        if 'vader' not in self.analyzers:
            return {'vader_compound': 0.0, 'vader_positive': 0.0, 'vader_negative': 0.0, 'vader_neutral': 1.0}
        
        scores = self.analyzers['vader'].polarity_scores(text)
        return {
            'vader_positive': scores['pos'],
            'vader_negative': scores['neg'],
            'vader_neutral': scores['neu'],
            'vader_compound': scores['compound']
        }
    
    def analyze_financial_context(self, text: str) -> Dict[str, Any]:
        """Analyze text for financial-specific sentiment indicators"""
        text_lower = text.lower()
        financial_analysis = {
            'positive_terms': 0,
            'negative_terms': 0,
            'neutral_terms': 0,
            'financial_intensity': 'weak',
            'detected_terms': [],
            'amplifier_present': False,
            'amplifier_count': 0
        }
        
        # Count positive terms by intensity
        for intensity, terms in self.financial_lexicon['positive'].items():
            for term in terms:
                if term in text_lower:
                    financial_analysis['positive_terms'] += 1
                    financial_analysis['detected_terms'].append(f"positive:{term}")
                    if intensity == 'strong':
                        financial_analysis['financial_intensity'] = 'strong'
                    elif intensity == 'moderate' and financial_analysis['financial_intensity'] == 'weak':
                        financial_analysis['financial_intensity'] = 'moderate'
        
        # Count negative terms by intensity
        for intensity, terms in self.financial_lexicon['negative'].items():
            for term in terms:
                if term in text_lower:
                    financial_analysis['negative_terms'] += 1
                    financial_analysis['detected_terms'].append(f"negative:{term}")
                    if intensity == 'strong':
                        financial_analysis['financial_intensity'] = 'strong'
                    elif intensity == 'moderate' and financial_analysis['financial_intensity'] == 'weak':
                        financial_analysis['financial_intensity'] = 'moderate'
        
        # Count neutral terms
        for term in self.financial_lexicon['neutral']:
            if term in text_lower:
                financial_analysis['neutral_terms'] += 1
                financial_analysis['detected_terms'].append(f"neutral:{term}")
        
        # Check for amplifiers
        for impact_level, terms in self.financial_amplifiers.items():
            for term in terms:
                if term in text_lower:
                    financial_analysis['amplifier_present'] = True
                    financial_analysis['amplifier_count'] += 1
                    financial_analysis['detected_terms'].append(f"amplifier:{term}")
        
        return financial_analysis
    
    def calculate_ensemble_sentiment(self, vader_scores: Dict, financial_analysis: Dict) -> Tuple[str, float, str]:
        """Combine VADER and financial analysis into final sentiment"""
        
        # Get VADER sentiment
        vader_sentiment = vader_scores.get('vader_compound', 0.0)
        vader_confidence = abs(vader_sentiment)
        
        # Calculate financial sentiment
        pos_count = financial_analysis.get('positive_terms', 0)
        neg_count = financial_analysis.get('negative_terms', 0)
        total_financial = pos_count + neg_count
        
        if total_financial > 0:
            # Financial terms detected
            financial_sentiment = (pos_count - neg_count) / total_financial
            
            # Apply intensity multiplier
            intensity_multiplier = {'weak': 0.5, 'moderate': 0.75, 'strong': 1.0}
            multiplier = intensity_multiplier.get(financial_analysis.get('financial_intensity', 'weak'), 0.5)
            financial_sentiment *= multiplier
            
            # Apply amplifier boost
            if financial_analysis.get('amplifier_present', False):
                amplifier_boost = min(0.3, financial_analysis.get('amplifier_count', 0) * 0.1)
                financial_sentiment *= (1 + amplifier_boost)
            
            financial_confidence = multiplier
        else:
            financial_sentiment = 0.0
            financial_confidence = 0.0
        
        # Combine sentiments
        if VADER_AVAILABLE and total_financial > 0:
            # Both VADER and financial analysis available
            final_sentiment = (vader_sentiment * 0.4) + (financial_sentiment * 0.6)  # Weight financial higher
            overall_confidence = (vader_confidence * 0.4) + (financial_confidence * 0.6)
        elif VADER_AVAILABLE:
            # Only VADER available
            final_sentiment = vader_sentiment
            overall_confidence = vader_confidence
        elif total_financial > 0:
            # Only financial analysis available
            final_sentiment = financial_sentiment
            overall_confidence = financial_confidence
        else:
            # No clear sentiment
            final_sentiment = 0.0
            overall_confidence = 0.2  # Low confidence
        
        # Determine sentiment labels
        if final_sentiment > 0.1:
            sentiment_label = 'Positive'
            financial_sentiment_label = 'Bullish'
        elif final_sentiment < -0.1:
            sentiment_label = 'Negative'
            financial_sentiment_label = 'Bearish'
        else:
            sentiment_label = 'Neutral'
            financial_sentiment_label = 'Neutral'
        
        # Ensure confidence is reasonable
        overall_confidence = max(0.1, min(1.0, overall_confidence))
        
        return sentiment_label, overall_confidence, financial_sentiment_label
    
    def analyze_sentiment(self, text: str, title: str = None) -> SentimentResult:
        """
        Main method to analyze sentiment of financial text
        """
        if not text:
            # Return neutral sentiment for empty text
            return SentimentResult(
                overall_sentiment='Neutral',
                confidence=0.0,
                polarity=0.0,
                subjectivity=0.5,
                financial_sentiment='Neutral',
                vader_scores={},
                key_phrases=[],
                sentiment_intensity='Weak',
                sentiment_score=50.0,  # ADD THIS LINE
                analyzed_at=datetime.now()
            )
        
        # Combine title and text, giving title more weight
        full_text = text
        if title:
            full_text = f"{title}. {text}"
        
        # Preprocess text
        processed_text = self.preprocess_text(full_text)
        
        # Run analyses
        vader_scores = self.analyze_with_vader(processed_text)
        financial_analysis = self.analyze_financial_context(processed_text)
        
        # Extract key phrases
        key_phrases = self.extract_key_phrases(full_text)
        
        # Calculate ensemble sentiment
        sentiment_label, confidence, financial_sentiment = self.calculate_ensemble_sentiment(
            vader_scores, financial_analysis
        )
        
        # Determine sentiment intensity
        polarity = vader_scores.get('vader_compound', 0.0)
        intensity_score = confidence * abs(polarity) if polarity != 0 else confidence
        
        if intensity_score > 0.6:
            sentiment_intensity = 'Strong'
        elif intensity_score > 0.3:
            sentiment_intensity = 'Moderate'
        else:
            sentiment_intensity = 'Weak'
        
        # Estimate subjectivity (simple heuristic)
        subjectivity = 0.7 if financial_analysis.get('detected_terms') else 0.3
        
        # Update statistics
        self.stats['total_analyses'] += 1
        if sentiment_label == 'Positive':
            self.stats['positive_sentiment'] += 1
        elif sentiment_label == 'Negative':
            self.stats['negative_sentiment'] += 1
        else:
            self.stats['neutral_sentiment'] += 1
        
        if confidence > 0.7:
            self.stats['high_confidence_analyses'] += 1
        
        if financial_analysis.get('detected_terms'):
            self.stats['financial_terms_detected'] += 1
        
        # Create result object
        # Calculate sentiment score (0-100 scale)
        sentiment_score = ((polarity + 1) / 2) * 100
        
        # Create result object
        result = SentimentResult(
            overall_sentiment=sentiment_label,
            confidence=confidence,
            polarity=polarity,
            subjectivity=subjectivity,
            financial_sentiment=financial_sentiment,
            vader_scores=vader_scores,
            key_phrases=key_phrases,
            sentiment_intensity=sentiment_intensity,
            sentiment_score=sentiment_score,
            analyzed_at=datetime.now()
        )
        
        return result
    
    def analyze_batch(self, articles: List[Dict]) -> Dict[str, SentimentResult]:
        """Analyze sentiment for a batch of articles"""
        results = {}
        
        for article in articles:
            article_key = article.get('source_key', f"article_{len(results)}")
            title = article.get('title', '')
            summary = article.get('summary', '')
            
            # Analyze sentiment
            sentiment_result = self.analyze_sentiment(summary, title)
            results[article_key] = sentiment_result
            
            # Add sentiment data back to article
            article['sentiment_analysis'] = {
                'overall_sentiment': sentiment_result.overall_sentiment,
                'confidence': sentiment_result.confidence,
                'financial_sentiment': sentiment_result.financial_sentiment,
                'polarity': sentiment_result.polarity,
                'intensity': sentiment_result.sentiment_intensity
            }
        
        return results
    
    def get_sentiment_summary(self, articles: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics for sentiment analysis"""
        if not articles:
            return {}
        
        sentiments = []
        confidences = []
        polarities = []
        
        for article in articles:
            sentiment_data = article.get('sentiment_analysis')
            if sentiment_data:
                sentiments.append(sentiment_data['overall_sentiment'])
                confidences.append(sentiment_data['confidence'])
                polarities.append(sentiment_data['polarity'])
        
        if not sentiments:
            return {}
        
        # Calculate summary statistics
        positive_count = sentiments.count('Positive')
        negative_count = sentiments.count('Negative')
        neutral_count = sentiments.count('Neutral')
        
        # Simple average polarity calculation
        avg_polarity = sum(polarities) / len(polarities) if polarities else 0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        summary = {
            'total_articles': len(sentiments),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'positive_percentage': (positive_count / len(sentiments)) * 100,
            'negative_percentage': (negative_count / len(sentiments)) * 100,
            'neutral_percentage': (neutral_count / len(sentiments)) * 100,
            'average_confidence': avg_confidence,
            'average_polarity': avg_polarity,
            'overall_sentiment': 'Positive' if positive_count > negative_count else 'Negative' if negative_count > positive_count else 'Neutral'
        }
        
        return summary
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sentiment analysis statistics"""
        total = self.stats['total_analyses']
        if total == 0:
            return self.stats
        
        enhanced_stats = self.stats.copy()
        enhanced_stats.update({
            'positive_percentage': (self.stats['positive_sentiment'] / total) * 100,
            'negative_percentage': (self.stats['negative_sentiment'] / total) * 100,
            'neutral_percentage': (self.stats['neutral_sentiment'] / total) * 100,
            'high_confidence_percentage': (self.stats['high_confidence_analyses'] / total) * 100,
            'financial_terms_percentage': (self.stats['financial_terms_detected'] / total) * 100
        })
        
        return enhanced_stats
    
    def reset_stats(self):
        """Reset sentiment analysis statistics"""
        self.stats = {
            'total_analyses': 0,
            'positive_sentiment': 0,
            'negative_sentiment': 0,
            'neutral_sentiment': 0,
            'high_confidence_analyses': 0,
            'financial_terms_detected': 0
        }

# Global sentiment analyzer instance
sentiment_analyzer = WorkingFinancialSentimentAnalyzer()

if __name__ == "__main__":
    # Test the working analyzer
    print("Working Financial Sentiment Analyzer Test")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            'title': 'Apple Beats Earnings Expectations, Stock Soars 5%',
            'text': 'Apple Inc. reported strong quarterly earnings that exceeded analyst expectations, driven by robust iPhone sales and growing services revenue.',
            'expected': 'Positive'
        },
        {
            'title': 'Tesla Stock Plunges on Production Concerns',
            'text': 'Tesla shares fell sharply after the company reported lower than expected vehicle deliveries and warned of potential production delays.',
            'expected': 'Negative'
        },
        {
            'title': 'Microsoft Reports Steady Growth',
            'text': 'Microsoft Corporation announced modest growth in its cloud computing division, with revenue increasing in line with forecasts.',
            'expected': 'Neutral'
        }
    ]
    
    analyzer = WorkingFinancialSentimentAnalyzer()
    
    print(f"VADER Available: {VADER_AVAILABLE}")
    print(f"Testing {len(test_cases)} scenarios:")
    print("-" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['title']}")
        
        # Analyze sentiment
        result = analyzer.analyze_sentiment(test_case['text'], test_case['title'])
        
        print(f"  Expected: {test_case['expected']}")
        print(f"  Result: {result.overall_sentiment}")
        print(f"  Financial: {result.financial_sentiment}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Polarity: {result.polarity:.2f}")
        print(f"  Intensity: {result.sentiment_intensity}")
        if result.key_phrases:
            print(f"  Key Phrases: {', '.join(result.key_phrases[:3])}")
        
        # Check accuracy
        if result.overall_sentiment == test_case['expected']:
            print("  ✅ CORRECT")
        else:
            print("  ❌ Different from expected")
    
    # Show statistics
    stats = analyzer.get_stats()
    print(f"\nAnalysis Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ Working sentiment analyzer test complete!")