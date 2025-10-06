"""
Text processor for Phase 3 News Intelligence Engine.

This module handles text cleaning, keyword extraction, entity recognition,
and content preprocessing for sentiment analysis and event extraction.
"""

import re
import string
import logging
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class TextAnalysisResult:
    """Result of text analysis."""
    cleaned_text: str
    keywords: List[str]
    entities: Dict[str, List[str]]  # entity_type: [entities]
    readability_score: float
    text_length: int
    word_count: int
    sentence_count: int
    financial_terms: List[str]
    stock_symbols: List[str]
    confidence_indicators: List[str]
    urgency_score: float

class TextProcessor:
    """Advanced text processing for news content analysis."""
    
    def __init__(self):
        self.stop_words = self._load_stop_words()
        self.financial_terms = self._load_financial_terms()
        self.confidence_indicators = self._load_confidence_indicators()
        self.stock_symbols = self._load_common_stock_symbols()
        
        # Precompiled regex patterns for efficiency
        self.patterns = {
            'stock_symbol': re.compile(r'\b([A-Z]{1,5})(?:\s+(?:stock|shares|ticker))?\b'),
            'dollar_amount': re.compile(r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion|M|B|T))?', re.IGNORECASE),
            'percentage': re.compile(r'\d+(?:\.\d+)?%'),
            'date': re.compile(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})\b'),
            'url': re.compile(r'https?://[^\s]+'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'company_suffix': re.compile(r'\b\w+\s+(?:Inc|Corp|LLC|Ltd|Co|Company|Corporation)\b', re.IGNORECASE)
        }
    
    def _load_stop_words(self) -> Set[str]:
        """Load common English stop words."""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'would', 'could', 'should',
            'this', 'these', 'they', 'them', 'their', 'have', 'had', 'been',
            'said', 'says', 'can', 'may', 'also', 'more', 'than', 'very',
            'when', 'where', 'who', 'what', 'why', 'how', 'but', 'or', 'not',
            'up', 'out', 'if', 'about', 'into', 'over', 'after'
        }
    
    def _load_financial_terms(self) -> Set[str]:
        """Load financial and market terminology."""
        return {
            # Market terms
            'market', 'stock', 'share', 'equity', 'bond', 'security', 'portfolio',
            'trading', 'trader', 'investor', 'investment', 'fund', 'etf', 'mutual',
            
            # Financial metrics
            'revenue', 'earnings', 'profit', 'loss', 'ebitda', 'margin', 'growth',
            'dividend', 'yield', 'pe', 'ratio', 'valuation', 'capitalization',
            
            # Market movements
            'bull', 'bear', 'rally', 'correction', 'crash', 'volatility', 'surge',
            'decline', 'rise', 'fall', 'gain', 'climb', 'drop', 'plunge',
            
            # Corporate actions
            'merger', 'acquisition', 'ipo', 'buyback', 'split', 'spinoff',
            'bankruptcy', 'restructuring', 'delisting',
            
            # Economic indicators
            'gdp', 'inflation', 'recession', 'recovery', 'unemployment', 'cpi',
            'fed', 'federal reserve', 'interest rate', 'monetary policy',
            
            # Analyst terms
            'forecast', 'estimate', 'guidance', 'outlook', 'projection', 'target',
            'upgrade', 'downgrade', 'rating', 'analyst', 'consensus'
        }
    
    def _load_confidence_indicators(self) -> Dict[str, float]:
        """Load confidence/uncertainty indicators with weights."""
        return {
            # High confidence
            'confirmed': 0.9, 'official': 0.9, 'announced': 0.9, 'reported': 0.8,
            'stated': 0.8, 'disclosed': 0.8, 'revealed': 0.8,
            
            # Medium confidence
            'indicated': 0.6, 'suggested': 0.6, 'expected': 0.6, 'projected': 0.6,
            'estimated': 0.6, 'forecast': 0.6, 'predicted': 0.6,
            
            # Low confidence
            'rumored': 0.3, 'alleged': 0.3, 'speculated': 0.3, 'possible': 0.4,
            'potential': 0.4, 'may': 0.4, 'might': 0.3, 'could': 0.4,
            
            # Uncertainty indicators
            'uncertain': 0.2, 'unclear': 0.2, 'unknown': 0.1, 'unconfirmed': 0.2
        }
    
    def _load_common_stock_symbols(self) -> Set[str]:
        """Load common stock symbols for better recognition."""
        return {
            # Tech giants
            'AAPL', 'GOOGL', 'GOOG', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA',
            'ORCL', 'IBM', 'CSCO', 'INTC', 'AMD', 'QCOM', 'ADBE', 'CRM',
            
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'BLK',
            
            # Healthcare
            'JNJ', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK', 'UNH', 'MDT',
            
            # Consumer
            'WMT', 'PG', 'KO', 'PEP', 'NKE', 'MCD', 'SBUX', 'HD', 'LOW',
            
            # Industrial
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'FDX',
            
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG',
            
            # Crypto/Fintech
            'COIN', 'PYPL', 'SQ', 'HOOD'
        }
    
    def process_text(self, text: str, title: str = None) -> TextAnalysisResult:
        """
        Perform comprehensive text analysis.
        
        Args:
            text: Article content to analyze
            title: Article title (optional, for enhanced analysis)
            
        Returns:
            TextAnalysisResult: Comprehensive analysis results
        """
        if not text:
            return self._empty_result()
        
        # Combine title and text for analysis
        full_text = f"{title or ''} {text}".strip()
        
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Extract various elements
        keywords = self.extract_keywords(full_text)
        entities = self.extract_entities(full_text)
        financial_terms = self.extract_financial_terms(full_text)
        stock_symbols = self.extract_stock_symbols(full_text)
        confidence_indicators = self.extract_confidence_indicators(full_text)
        
        # Calculate metrics
        readability_score = self.calculate_readability(cleaned_text)
        urgency_score = self.calculate_urgency_score(full_text)
        word_count = len(cleaned_text.split())
        sentence_count = len([s for s in cleaned_text.split('.') if s.strip()])
        
        return TextAnalysisResult(
            cleaned_text=cleaned_text,
            keywords=keywords,
            entities=entities,
            readability_score=readability_score,
            text_length=len(text),
            word_count=word_count,
            sentence_count=sentence_count,
            financial_terms=financial_terms,
            stock_symbols=stock_symbols,
            confidence_indicators=confidence_indicators,
            urgency_score=urgency_score
        )
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove URLs
        text = self.patterns['url'].sub('', text)
        
        # Remove email addresses
        text = self.patterns['email'].sub('', text)
        
        # Remove phone numbers
        text = self.patterns['phone'].sub('', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        # Fix common encoding issues
        text = text.replace('â€™', "'").replace('â€œ', '"').replace('â€', '"')
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        return text.strip()
    
    def extract_keywords(self, text: str, max_keywords: int = 15) -> List[str]:
        """
        Extract important keywords from text.
        
        Args:
            text: Text to analyze
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List[str]: Important keywords
        """
        if not text:
            return []
        
        # Tokenize and clean words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove stop words
        meaningful_words = [w for w in words if w not in self.stop_words]
        
        # Count word frequency
        word_freq = Counter(meaningful_words)
        
        # Boost financial terms
        for word in meaningful_words:
            if word in self.financial_terms:
                word_freq[word] *= 2
        
        # Get most common words
        keywords = [word for word, count in word_freq.most_common(max_keywords)]
        
        return keywords
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict[str, List[str]]: Entities by type
        """
        entities = {
            'companies': [],
            'people': [],
            'locations': [],
            'dollar_amounts': [],
            'percentages': [],
            'dates': []
        }
        
        # Extract dollar amounts
        dollar_matches = self.patterns['dollar_amount'].findall(text)
        entities['dollar_amounts'] = list(set(dollar_matches))
        
        # Extract percentages
        percentage_matches = self.patterns['percentage'].findall(text)
        entities['percentages'] = list(set(percentage_matches))
        
        # Extract dates
        date_matches = self.patterns['date'].findall(text)
        entities['dates'] = list(set(date_matches))
        
        # Extract company names (basic pattern matching)
        company_matches = self.patterns['company_suffix'].findall(text)
        entities['companies'] = list(set(company_matches))
        
        # Extract potential people names (capitalized words)
        people_pattern = re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b')
        people_matches = people_pattern.findall(text)
        # Filter out common false positives
        filtered_people = [name for name in people_matches 
                          if not any(word in name.lower() for word in 
                                   ['new york', 'wall street', 'united states', 'federal reserve'])]
        entities['people'] = list(set(filtered_people))
        
        return entities
    
    def extract_financial_terms(self, text: str) -> List[str]:
        """Extract financial terms from text."""
        text_lower = text.lower()
        found_terms = []
        
        for term in self.financial_terms:
            if term in text_lower:
                found_terms.append(term)
        
        return list(set(found_terms))
    
    def extract_stock_symbols(self, text: str) -> List[str]:
        """
        Extract stock symbols from text with improved accuracy.
        
        Args:
            text: Text to analyze
            
        Returns:
            List[str]: Stock symbols found
        """
        symbols = set()
        
        # Pattern 1: Explicit stock mentions ($AAPL, AAPL stock, etc.)
        explicit_patterns = [
            r'\$([A-Z]{1,5})\b',  # $AAPL
            r'\b([A-Z]{1,5})\s+(?:stock|shares|ticker|equity)\b',  # AAPL stock
            r'\(([A-Z]{1,5})\)',  # (AAPL)
            r'\b([A-Z]{1,5}):\s*[A-Z]',  # AAPL: NASDAQ
        ]
        
        for pattern in explicit_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                symbols.add(match.upper())
        
        # Filter against known symbols for higher accuracy
        valid_symbols = [s for s in symbols if s in self.stock_symbols]
        
        return list(set(valid_symbols))
    
    def extract_confidence_indicators(self, text: str) -> List[str]:
        """Extract confidence/uncertainty indicators from text."""
        text_lower = text.lower()
        found_indicators = []
        
        for indicator in self.confidence_indicators:
            if indicator in text_lower:
                found_indicators.append(indicator)
        
        return found_indicators
    
    def calculate_readability(self, text: str) -> float:
        """
        Calculate readability score (simplified Flesch Reading Ease).
        
        Args:
            text: Text to analyze
            
        Returns:
            float: Readability score (0-100, higher = more readable)
        """
        if not text:
            return 0.0
        
        sentences = len([s for s in text.split('.') if s.strip()])
        words = len(text.split())
        syllables = sum(self._count_syllables(word) for word in text.split())
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Simplified Flesch Reading Ease formula
        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        
        # Normalize to 0-100 range
        return max(0.0, min(100.0, score))
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximate)."""
        word = word.lower().strip()
        if len(word) <= 3:
            return 1
        
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def calculate_urgency_score(self, text: str) -> float:
        """
        Calculate urgency score based on language indicators.
        
        Args:
            text: Text to analyze
            
        Returns:
            float: Urgency score (0-1, higher = more urgent)
        """
        text_lower = text.lower()
        urgency_score = 0.0
        
        # Urgent words and their weights
        urgent_indicators = {
            'breaking': 0.9, 'urgent': 0.9, 'alert': 0.8, 'immediate': 0.8,
            'emergency': 0.9, 'critical': 0.7, 'major': 0.6, 'significant': 0.5,
            'suddenly': 0.6, 'unexpected': 0.6, 'shock': 0.7, 'crash': 0.8,
            'plunge': 0.7, 'surge': 0.6, 'spike': 0.6, 'jump': 0.5,
            'just in': 0.8, 'developing': 0.6, 'live': 0.5
        }
        
        # Check for urgent indicators
        for indicator, weight in urgent_indicators.items():
            if indicator in text_lower:
                urgency_score = max(urgency_score, weight)
        
        # Check for time-sensitive language
        time_sensitive = ['today', 'now', 'this morning', 'tonight', 'minutes ago']
        for term in time_sensitive:
            if term in text_lower:
                urgency_score = max(urgency_score, 0.4)
        
        return urgency_score
    
    def _empty_result(self) -> TextAnalysisResult:
        """Return empty analysis result."""
        return TextAnalysisResult(
            cleaned_text="",
            keywords=[],
            entities={},
            readability_score=0.0,
            text_length=0,
            word_count=0,
            sentence_count=0,
            financial_terms=[],
            stock_symbols=[],
            confidence_indicators=[],
            urgency_score=0.0
        )

# Example usage and testing
if __name__ == "__main__":
    # Test the text processor
    processor = TextProcessor()
    
    # Sample financial news text
    sample_text = """
    Apple Inc. (AAPL) stock surged 5.2% in after-hours trading following the company's 
    announcement of better-than-expected quarterly earnings. The tech giant reported 
    revenue of $94.8 billion, beating analyst estimates of $92.1 billion. CEO Tim Cook 
    confirmed that iPhone sales grew 15% year-over-year, driven by strong demand in China.
    
    The Federal Reserve's recent interest rate decision has created market volatility, 
    with analysts projecting further gains for major tech stocks including Google (GOOGL), 
    Microsoft (MSFT), and Amazon (AMZN). Wall Street expects continued growth momentum 
    through Q4 2024.
    """
    
    # Analyze the text
    result = processor.process_text(sample_text, "Apple Reports Strong Q3 Earnings")
    
    print("=== Text Analysis Results ===")
    print(f"Word count: {result.word_count}")
    print(f"Readability score: {result.readability_score:.1f}")
    print(f"Urgency score: {result.urgency_score:.2f}")
    print(f"Keywords: {result.keywords}")
    print(f"Stock symbols: {result.stock_symbols}")
    print(f"Financial terms: {result.financial_terms}")
    print(f"Confidence indicators: {result.confidence_indicators}")
    print(f"Dollar amounts: {result.entities.get('dollar_amounts', [])}")
    print(f"Percentages: {result.entities.get('percentages', [])}")
    print(f"Companies: {result.entities.get('companies', [])}")
    
    print("\nText processor test completed!")