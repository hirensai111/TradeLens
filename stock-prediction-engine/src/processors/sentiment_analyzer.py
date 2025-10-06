"""
Sentiment analyzer for Phase 3 News Intelligence Engine.

This module provides sophisticated sentiment analysis specifically tuned for 
financial news content, with support for multiple analysis methods and 
market-specific sentiment scoring.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    sentiment_score: float  # -1.0 (very negative) to 1.0 (very positive)
    sentiment_label: str    # 'positive', 'negative', 'neutral'
    confidence: float       # 0.0 to 1.0
    market_sentiment: float # Market-specific sentiment (-1.0 to 1.0)
    emotional_indicators: Dict[str, float]  # emotion: strength
    risk_sentiment: float   # Risk-specific sentiment
    analysis_method: str    # Method used for analysis
    processing_time: float  # Time taken for analysis

class FinancialSentimentAnalyzer:
    """Advanced sentiment analyzer tuned for financial news."""
    
    def __init__(self):
        self.positive_words = self._load_positive_words()
        self.negative_words = self._load_negative_words()
        self.market_positive = self._load_market_positive_terms()
        self.market_negative = self._load_market_negative_terms()
        self.risk_indicators = self._load_risk_indicators()
        self.intensifiers = self._load_intensifiers()
        self.negators = self._load_negators()
        
        # Sentiment patterns
        self.sentiment_patterns = self._compile_sentiment_patterns()
    
    def _load_positive_words(self) -> Dict[str, float]:
        """Load positive sentiment words with weights."""
        return {
            # Strong positive
            'excellent': 0.8, 'outstanding': 0.9, 'exceptional': 0.8, 'superb': 0.8,
            'fantastic': 0.7, 'amazing': 0.7, 'incredible': 0.7, 'wonderful': 0.6,
            
            # Financial positive
            'profit': 0.6, 'gain': 0.6, 'growth': 0.7, 'boom': 0.8, 'surge': 0.7,
            'rally': 0.7, 'bullish': 0.8, 'upbeat': 0.6, 'optimistic': 0.6,
            'breakthrough': 0.7, 'success': 0.6, 'achievement': 0.6,
            
            # Market positive
            'beat': 0.7, 'exceed': 0.6, 'outperform': 0.7, 'strong': 0.6,
            'robust': 0.6, 'solid': 0.5, 'healthy': 0.5, 'stable': 0.4,
            'confident': 0.6, 'promising': 0.6, 'favorable': 0.5,
            
            # Moderate positive
            'good': 0.4, 'positive': 0.5, 'improve': 0.5, 'better': 0.4,
            'increase': 0.4, 'rise': 0.4, 'advance': 0.4, 'progress': 0.5
        }
    
    def _load_negative_words(self) -> Dict[str, float]:
        """Load negative sentiment words with weights."""
        return {
            # Strong negative
            'terrible': -0.8, 'awful': -0.8, 'horrible': -0.8, 'devastating': -0.9,
            'catastrophic': -0.9, 'disastrous': -0.9, 'crisis': -0.8,
            
            # Financial negative
            'loss': -0.6, 'decline': -0.6, 'crash': -0.9, 'plunge': -0.8,
            'collapse': -0.9, 'bearish': -0.8, 'pessimistic': -0.6,
            'recession': -0.8, 'bankruptcy': -0.9, 'default': -0.8,
            
            # Market negative
            'miss': -0.7, 'disappoint': -0.6, 'underperform': -0.7, 'weak': -0.6,
            'poor': -0.6, 'troubled': -0.6, 'volatile': -0.5, 'uncertain': -0.4,
            'concern': -0.5, 'worry': -0.5, 'fear': -0.6, 'risk': -0.4,
            
            # Moderate negative
            'bad': -0.4, 'negative': -0.5, 'decrease': -0.4, 'fall': -0.4,
            'drop': -0.4, 'lower': -0.3, 'reduce': -0.3, 'cut': -0.4
        }
    
    def _load_market_positive_terms(self) -> Dict[str, float]:
        """Load market-specific positive terms."""
        return {
            'earnings beat': 0.8, 'revenue beat': 0.7, 'guidance raised': 0.8,
            'dividend increase': 0.6, 'buyback': 0.6, 'acquisition': 0.5,
            'partnership': 0.4, 'expansion': 0.5, 'innovation': 0.5,
            'approval': 0.6, 'launch': 0.4, 'upgrade': 0.6, 'buy rating': 0.7,
            'price target raised': 0.7, 'market share': 0.4, 'competitive advantage': 0.6
        }
    
    def _load_market_negative_terms(self) -> Dict[str, float]:
        """Load market-specific negative terms."""
        return {
            'earnings miss': -0.8, 'revenue miss': -0.7, 'guidance lowered': -0.8,
            'dividend cut': -0.7, 'layoffs': -0.6, 'investigation': -0.6,
            'lawsuit': -0.5, 'recall': -0.6, 'downgrade': -0.6, 'sell rating': -0.7,
            'price target lowered': -0.7, 'market share loss': -0.5, 'competition': -0.3,
            'regulatory': -0.4, 'fine': -0.5, 'penalty': -0.5
        }
    
    def _load_risk_indicators(self) -> Dict[str, float]:
        """Load risk-related terms and their weights."""
        return {
            'high risk': 0.8, 'risky': 0.6, 'uncertain': 0.5, 'volatile': 0.7,
            'unpredictable': 0.6, 'unstable': 0.7, 'speculation': 0.5,
            'bubble': 0.8, 'overvalued': 0.6, 'correction': 0.7,
            'safe': -0.6, 'stable': -0.5, 'secure': -0.5, 'predictable': -0.4
        }
    
    def _load_intensifiers(self) -> Dict[str, float]:
        """Load words that intensify sentiment."""
        return {
            'very': 1.5, 'extremely': 2.0, 'highly': 1.6, 'significantly': 1.7,
            'substantially': 1.8, 'dramatically': 2.0, 'considerably': 1.5,
            'remarkably': 1.6, 'exceptionally': 1.8, 'particularly': 1.3,
            'especially': 1.3, 'quite': 1.2, 'rather': 1.1, 'somewhat': 0.8,
            'slightly': 0.7, 'barely': 0.5, 'hardly': 0.4
        }
    
    def _load_negators(self) -> List[str]:
        """Load negation words that flip sentiment."""
        return [
            'not', 'no', 'never', 'none', 'nothing', 'neither', 'nor',
            'cannot', 'can\'t', 'won\'t', 'wouldn\'t', 'shouldn\'t',
            'doesn\'t', 'don\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t'
        ]
    
    def _compile_sentiment_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for sentiment analysis."""
        return {
            'negation': re.compile(r'\b(?:' + '|'.join(self.negators) + r')\b', re.IGNORECASE),
            'intensifier': re.compile(r'\b(?:' + '|'.join(self.intensifiers.keys()) + r')\b', re.IGNORECASE),
            'financial_context': re.compile(r'\b(?:stock|share|market|trading|investor|analyst)\b', re.IGNORECASE),
            'earnings_context': re.compile(r'\b(?:earnings|revenue|profit|quarterly|q[1-4])\b', re.IGNORECASE),
            'percentage': re.compile(r'(\d+(?:\.\d+)?)\s*%'),
            'dollar_change': re.compile(r'\$(\d+(?:\.\d+)?)\s*(?:billion|million|B|M)?', re.IGNORECASE)
        }
    
    def analyze_sentiment(self, text: str, title: str = None) -> SentimentResult:
        """
        Perform comprehensive sentiment analysis on financial text.
        
        Args:
            text: Article content to analyze
            title: Article title (optional, weighted higher)
            
        Returns:
            SentimentResult: Comprehensive sentiment analysis
        """
        start_time = datetime.now()
        
        if not text:
            return self._empty_result()
        
        # Combine title and text, weighting title higher
        full_text = text
        if title:
            full_text = f"{title} {title} {text}"  # Title appears twice for higher weight
        
        # Analyze different sentiment aspects
        basic_sentiment = self._analyze_basic_sentiment(full_text)
        market_sentiment = self._analyze_market_sentiment(full_text)
        risk_sentiment = self._analyze_risk_sentiment(full_text)
        emotional_indicators = self._analyze_emotional_indicators(full_text)
        
        # Calculate overall sentiment score (weighted combination)
        sentiment_score = (
            basic_sentiment * 0.4 +
            market_sentiment * 0.5 +
            risk_sentiment * 0.1
        )
        
        # Determine sentiment label
        sentiment_label = self._get_sentiment_label(sentiment_score)
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence(
            full_text, sentiment_score, emotional_indicators
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return SentimentResult(
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            confidence=confidence,
            market_sentiment=market_sentiment,
            emotional_indicators=emotional_indicators,
            risk_sentiment=risk_sentiment,
            analysis_method="financial_lexicon",
            processing_time=processing_time
        )
    
    def _analyze_basic_sentiment(self, text: str) -> float:
        """Analyze basic positive/negative sentiment."""
        words = re.findall(r'\b\w+\b', text.lower())
        sentiment_score = 0.0
        word_count = 0
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # Check for negation in previous 2 words
            negated = False
            for j in range(max(0, i-2), i):
                if words[j] in self.negators:
                    negated = True
                    break
            
            # Check for intensifiers in previous 2 words
            intensifier = 1.0
            for j in range(max(0, i-2), i):
                if words[j] in self.intensifiers:
                    intensifier = self.intensifiers[words[j]]
                    break
            
            # Get word sentiment
            word_sentiment = 0.0
            if word in self.positive_words:
                word_sentiment = self.positive_words[word]
            elif word in self.negative_words:
                word_sentiment = self.negative_words[word]
            
            # Apply modifications
            if word_sentiment != 0:
                if negated:
                    word_sentiment *= -1
                word_sentiment *= intensifier
                sentiment_score += word_sentiment
                word_count += 1
            
            i += 1
        
        # Normalize by word count
        if word_count > 0:
            sentiment_score = sentiment_score / word_count
        
        # Clamp to [-1, 1] range
        return max(-1.0, min(1.0, sentiment_score))
    
    def _analyze_market_sentiment(self, text: str) -> float:
        """Analyze market-specific sentiment."""
        text_lower = text.lower()
        market_score = 0.0
        term_count = 0
        
        # Check positive market terms
        for term, weight in self.market_positive.items():
            if term in text_lower:
                market_score += weight
                term_count += 1
        
        # Check negative market terms
        for term, weight in self.market_negative.items():
            if term in text_lower:
                market_score += weight  # weight is already negative
                term_count += 1
        
        # Analyze percentage movements
        percentage_matches = self.sentiment_patterns['percentage'].findall(text)
        for pct_str in percentage_matches:
            pct = float(pct_str)
            context = self._get_percentage_context(text, pct_str)
            
            if 'gain' in context or 'rise' in context or 'up' in context:
                market_score += min(pct / 10.0, 0.5)  # Cap at 0.5
                term_count += 1
            elif 'loss' in context or 'fall' in context or 'down' in context:
                market_score -= min(pct / 10.0, 0.5)  # Cap at 0.5
                term_count += 1
        
        # Normalize
        if term_count > 0:
            market_score = market_score / term_count
        
        return max(-1.0, min(1.0, market_score))
    
    def _analyze_risk_sentiment(self, text: str) -> float:
        """Analyze risk-related sentiment."""
        text_lower = text.lower()
        risk_score = 0.0
        term_count = 0
        
        for term, weight in self.risk_indicators.items():
            if term in text_lower:
                risk_score += weight
                term_count += 1
        
        if term_count > 0:
            risk_score = risk_score / term_count
        
        return max(-1.0, min(1.0, risk_score))
    
    def _analyze_emotional_indicators(self, text: str) -> Dict[str, float]:
        """Analyze specific emotional indicators."""
        text_lower = text.lower()
        emotions = {
            'optimism': 0.0,
            'pessimism': 0.0,
            'fear': 0.0,
            'greed': 0.0,
            'uncertainty': 0.0,
            'confidence': 0.0
        }
        
        # Optimism indicators
        optimism_terms = ['optimistic', 'confident', 'bullish', 'positive', 'hopeful', 'promising']
        emotions['optimism'] = sum(1 for term in optimism_terms if term in text_lower) / len(optimism_terms)
        
        # Pessimism indicators
        pessimism_terms = ['pessimistic', 'bearish', 'negative', 'gloomy', 'dire', 'bleak']
        emotions['pessimism'] = sum(1 for term in pessimism_terms if term in text_lower) / len(pessimism_terms)
        
        # Fear indicators
        fear_terms = ['fear', 'panic', 'worry', 'concern', 'anxiety', 'nervous', 'scared']
        emotions['fear'] = sum(1 for term in fear_terms if term in text_lower) / len(fear_terms)
        
        # Greed indicators
        greed_terms = ['rally', 'surge', 'boom', 'bubble', 'euphoria', 'frenzy']
        emotions['greed'] = sum(1 for term in greed_terms if term in text_lower) / len(greed_terms)
        
        # Uncertainty indicators
        uncertainty_terms = ['uncertain', 'unclear', 'volatile', 'unpredictable', 'mixed', 'conflicted']
        emotions['uncertainty'] = sum(1 for term in uncertainty_terms if term in text_lower) / len(uncertainty_terms)
        
        # Confidence indicators
        confidence_terms = ['certain', 'confident', 'sure', 'definite', 'confirmed', 'solid']
        emotions['confidence'] = sum(1 for term in confidence_terms if term in text_lower) / len(confidence_terms)
        
        return emotions
    
    def _get_percentage_context(self, text: str, percentage: str) -> str:
        """Get context around a percentage mention."""
        # Find the percentage in text and get surrounding words
        pattern = re.compile(rf'\b\w+\s+{re.escape(percentage)}%|\b{re.escape(percentage)}%\s+\w+', re.IGNORECASE)
        match = pattern.search(text)
        
        if match:
            return match.group().lower()
        return ""
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.1:
            return "positive"
        elif score < -0.1:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_confidence(self, text: str, sentiment_score: float, 
                            emotional_indicators: Dict[str, float]) -> float:
        """Calculate confidence in sentiment analysis."""
        confidence_factors = []
        
        # Factor 1: Absolute sentiment score (higher = more confident)
        confidence_factors.append(abs(sentiment_score))
        
        # Factor 2: Text length (longer = more confident)
        word_count = len(text.split())
        length_confidence = min(word_count / 200.0, 1.0)  # Cap at 200 words
        confidence_factors.append(length_confidence)
        
        # Factor 3: Presence of financial context
        financial_context = bool(self.sentiment_patterns['financial_context'].search(text))
        confidence_factors.append(0.8 if financial_context else 0.3)
        
        # Factor 4: Consistency of emotional indicators
        emotion_values = list(emotional_indicators.values())
        if emotion_values:
            emotion_consistency = 1.0 - (max(emotion_values) - min(emotion_values))
            confidence_factors.append(emotion_consistency)
        
        # Factor 5: Presence of specific numbers/percentages
        has_numbers = bool(self.sentiment_patterns['percentage'].search(text) or 
                          self.sentiment_patterns['dollar_change'].search(text))
        confidence_factors.append(0.7 if has_numbers else 0.5)
        
        # Calculate weighted average
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]
        confidence = sum(factor * weight for factor, weight in zip(confidence_factors, weights))
        
        return max(0.0, min(1.0, confidence))
    
    def _empty_result(self) -> SentimentResult:
        """Return empty sentiment result."""
        return SentimentResult(
            sentiment_score=0.0,
            sentiment_label="neutral",
            confidence=0.0,
            market_sentiment=0.0,
            emotional_indicators={},
            risk_sentiment=0.0,
            analysis_method="none",
            processing_time=0.0
        )
    
    def batch_analyze(self, texts: List[Tuple[str, str]]) -> List[SentimentResult]:
        """
        Analyze sentiment for multiple texts efficiently.
        
        Args:
            texts: List of (text, title) tuples
            
        Returns:
            List[SentimentResult]: Sentiment results for each text
        """
        results = []
        for text, title in texts:
            result = self.analyze_sentiment(text, title)
            results.append(result)
        
        return results
    
    def get_market_sentiment_summary(self, results: List[SentimentResult]) -> Dict[str, float]:
        """
        Get overall market sentiment summary from multiple analyses.
        
        Args:
            results: List of sentiment results
            
        Returns:
            Dict[str, float]: Market sentiment summary
        """
        if not results:
            return {'overall': 0.0, 'confidence': 0.0, 'positive_ratio': 0.0}
        
        # Calculate averages
        avg_sentiment = sum(r.sentiment_score for r in results) / len(results)
        avg_confidence = sum(r.confidence for r in results) / len(results)
        avg_market_sentiment = sum(r.market_sentiment for r in results) / len(results)
        
        # Calculate positive ratio
        positive_count = sum(1 for r in results if r.sentiment_score > 0.1)
        positive_ratio = positive_count / len(results)
        
        return {
            'overall_sentiment': avg_sentiment,
            'market_sentiment': avg_market_sentiment,
            'confidence': avg_confidence,
            'positive_ratio': positive_ratio,
            'total_articles': len(results)
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = FinancialSentimentAnalyzer()
    
    # Sample financial news texts
    test_articles = [
        (
            "Apple Reports Strong Q3 Earnings Beat",
            """Apple Inc. exceeded Wall Street expectations with outstanding quarterly results, 
            reporting revenue of $94.8 billion, a 15% increase year-over-year. The tech giant's 
            iPhone sales surged, driven by robust demand in international markets. CEO Tim Cook 
            expressed optimism about future growth prospects, citing strong consumer confidence 
            and innovative product pipeline."""
        ),
        (
            "Market Volatility Continues Amid Economic Uncertainty",
            """Stock markets plunged today as investors grappled with mounting concerns over 
            inflation and potential recession risks. The Dow Jones fell 3.2%, while the S&P 500 
            dropped 2.8% in volatile trading. Analysts warn of continued uncertainty as the 
            Federal Reserve considers aggressive interest rate measures."""
        ),
        (
            "Tesla Stock Remains Stable Despite Mixed Quarterly Results",
            """Tesla (TSLA) delivered mixed results in its latest quarterly report, with revenue 
            slightly below analyst estimates but maintaining steady profitability. The electric 
            vehicle manufacturer reported moderate growth in deliveries while facing increased 
            competition in the EV market. Investors remain cautiously optimistic about the 
            company's long-term prospects."""
        )
    ]
    
    print("=== Financial Sentiment Analysis ===\n")
    
    all_results = []
    for title, content in test_articles:
        print(f"Article: {title}")
        print("-" * 50)
        
        result = analyzer.analyze_sentiment(content, title)
        all_results.append(result)
        
        print(f"Sentiment Score: {result.sentiment_score:.3f}")
        print(f"Sentiment Label: {result.sentiment_label}")
        print(f"Market Sentiment: {result.market_sentiment:.3f}")
        print(f"Risk Sentiment: {result.risk_sentiment:.3f}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Processing Time: {result.processing_time:.3f}s")
        print(f"Emotional Indicators: {result.emotional_indicators}")
        print()
    
    # Market summary
    summary = analyzer.get_market_sentiment_summary(all_results)
    print("=== Market Sentiment Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value:.3f}")
    
    print("\nSentiment analysis test completed!")