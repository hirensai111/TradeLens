"""
Event extractor for Phase 3 News Intelligence Engine.

This module identifies and extracts specific financial events from news content,
including earnings reports, mergers & acquisitions, IPOs, regulatory changes,
and other market-moving events.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of financial events."""
    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    IPO = "ipo"
    DIVIDEND = "dividend"
    BUYBACK = "buyback"
    LAYOFFS = "layoffs"
    PRODUCT_LAUNCH = "product_launch"
    REGULATORY = "regulatory"
    MONETARY_POLICY = "monetary_policy"
    BANKRUPTCY = "bankruptcy"
    PARTNERSHIP = "partnership"
    ANALYST_RATING = "analyst_rating"
    GUIDANCE_UPDATE = "guidance_update"
    LEADERSHIP_CHANGE = "leadership_change"
    BREAKING_NEWS = "breaking_news"
    MARKET_MOVEMENT = "market_movement"

class EventSeverity(Enum):
    """Event severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ExtractedEvent:
    """Represents an extracted financial event."""
    event_type: EventType
    event_subtype: Optional[str] = None
    severity: EventSeverity = EventSeverity.MEDIUM
    confidence: float = 0.0  # 0-1 confidence score
    companies: List[str] = field(default_factory=list)
    stock_symbols: List[str] = field(default_factory=list)
    key_figures: Dict[str, str] = field(default_factory=dict)  # amounts, percentages, etc.
    event_date: Optional[datetime] = None
    description: str = ""
    impact_indicators: List[str] = field(default_factory=list)
    source_text: str = ""
    extraction_confidence: float = 0.0

@dataclass
class EventExtractionResult:
    """Result of event extraction from text."""
    events: List[ExtractedEvent]
    processing_time: float
    text_length: int
    events_found: int
    high_confidence_events: int

class FinancialEventExtractor:
    """Advanced financial event extraction from news content."""
    
    def __init__(self):
        self.event_patterns = self._compile_event_patterns()
        self.company_indicators = self._load_company_indicators()
        self.financial_figures_pattern = self._compile_financial_patterns()
        self.temporal_patterns = self._compile_temporal_patterns()
        
    def _compile_event_patterns(self) -> Dict[EventType, List[re.Pattern]]:
        """Compile regex patterns for different event types."""
        patterns = {}
        
        # Earnings patterns
        patterns[EventType.EARNINGS] = [
            re.compile(r'\b(?:earnings|quarterly\s+results?|q[1-4]\s+(?:results?|earnings))\b', re.IGNORECASE),
            re.compile(r'\b(?:reports?|reported|announces?|announced)\s+(?:quarterly|q[1-4])?\s*(?:earnings|results?|revenue)\b', re.IGNORECASE),
            re.compile(r'\b(?:beats?|misses?|meets?)\s+(?:earnings|revenue|estimates?)\b', re.IGNORECASE),
            re.compile(r'\b(?:eps|earnings\s+per\s+share)\b', re.IGNORECASE),
            re.compile(r'\b(?:guidance|outlook)\s+(?:raised|lowered|updated|revised)\b', re.IGNORECASE)
        ]
        
        # M&A patterns
        patterns[EventType.MERGER_ACQUISITION] = [
            re.compile(r'\b(?:merger|merges?|merging)\s+with\b', re.IGNORECASE),
            re.compile(r'\b(?:acquires?|acquiring|acquisition|acquired)\b', re.IGNORECASE),
            re.compile(r'\b(?:buyout|takeover|deal|transaction)\b', re.IGNORECASE),
            re.compile(r'\b(?:agrees?\s+to\s+(?:buy|purchase|acquire))\b', re.IGNORECASE),
            re.compile(r'\$\d+(?:\.\d+)?\s*(?:billion|million)\s+(?:deal|acquisition|merger)', re.IGNORECASE)
        ]
        
        # IPO patterns
        patterns[EventType.IPO] = [
            re.compile(r'\b(?:ipo|initial\s+public\s+offering)\b', re.IGNORECASE),
            re.compile(r'\b(?:goes?\s+public|going\s+public)\b', re.IGNORECASE),
            re.compile(r'\b(?:public\s+debut|market\s+debut)\b', re.IGNORECASE),
            re.compile(r'\b(?:lists?|listing)\s+(?:on|at)\s+(?:nasdaq|nyse|stock\s+exchange)\b', re.IGNORECASE),
            re.compile(r'\b(?:shares?\s+began?\s+trading|trading\s+debut)\b', re.IGNORECASE)
        ]
        
        # Dividend patterns
        patterns[EventType.DIVIDEND] = [
            re.compile(r'\b(?:dividend|dividends?)\s+(?:declared|announced|increased|decreased|cut|suspended)\b', re.IGNORECASE),
            re.compile(r'\b(?:quarterly|annual)\s+dividend\b', re.IGNORECASE),
            re.compile(r'\b(?:dividend\s+yield|payout\s+ratio)\b', re.IGNORECASE),
            re.compile(r'\$\d+(?:\.\d+)?\s+(?:per\s+share\s+)?dividend', re.IGNORECASE)
        ]
        
        # Buyback patterns
        patterns[EventType.BUYBACK] = [
            re.compile(r'\b(?:share\s+buyback|stock\s+buyback|repurchase\s+program)\b', re.IGNORECASE),
            re.compile(r'\b(?:buyback|repurchases?|repurchasing)\s+(?:shares?|stock)\b', re.IGNORECASE),
            re.compile(r'\$\d+(?:\.\d+)?\s*(?:billion|million)\s+(?:buyback|repurchase)', re.IGNORECASE)
        ]
        
        # Layoffs patterns
        patterns[EventType.LAYOFFS] = [
            re.compile(r'\b(?:layoffs?|lay\s+off|laying\s+off)\b', re.IGNORECASE),
            re.compile(r'\b(?:job\s+cuts?|cutting\s+jobs?|eliminate\s+jobs?)\b', re.IGNORECASE),
            re.compile(r'\b(?:workforce\s+reduction|downsizing|restructuring)\b', re.IGNORECASE),
            re.compile(r'\b(?:fires?|firing|terminated|dismisses?)\s+\d+\s+(?:employees?|workers?)\b', re.IGNORECASE)
        ]
        
        # Product launch patterns
        patterns[EventType.PRODUCT_LAUNCH] = [
            re.compile(r'\b(?:launches?|launching|unveiled?|unveiling|introduces?|introducing)\s+(?:new|latest)\b', re.IGNORECASE),
            re.compile(r'\b(?:product\s+launch|new\s+product|latest\s+version)\b', re.IGNORECASE),
            re.compile(r'\b(?:announces?|announced)\s+(?:new|upcoming)\s+(?:product|service|feature)\b', re.IGNORECASE)
        ]
        
        # Regulatory patterns
        patterns[EventType.REGULATORY] = [
            re.compile(r'\b(?:fda|sec|ftc|doj|cftc)\s+(?:approves?|approved|denies?|denied|investigates?)\b', re.IGNORECASE),
            re.compile(r'\b(?:regulatory\s+approval|government\s+approval)\b', re.IGNORECASE),
            re.compile(r'\b(?:fined?|penalty|settlement|lawsuit|litigation)\b', re.IGNORECASE),
            re.compile(r'\b(?:investigation|probe|inquiry)\s+(?:into|regarding)\b', re.IGNORECASE)
        ]
        
        # Monetary policy patterns
        patterns[EventType.MONETARY_POLICY] = [
            re.compile(r'\b(?:federal\s+reserve|fed|central\s+bank)\s+(?:raises?|cuts?|maintains?)\s+(?:interest\s+rates?|rates?)\b', re.IGNORECASE),
            re.compile(r'\b(?:interest\s+rate\s+(?:decision|announcement|cut|hike))\b', re.IGNORECASE),
            re.compile(r'\b(?:monetary\s+policy|fomc\s+meeting|fed\s+meeting)\b', re.IGNORECASE),
            re.compile(r'\b(?:quantitative\s+easing|qe|tapering)\b', re.IGNORECASE)
        ]
        
        # Analyst rating patterns
        patterns[EventType.ANALYST_RATING] = [
            re.compile(r'\b(?:upgrades?|upgraded|downgrades?|downgraded)\s+(?:to|from)\b', re.IGNORECASE),
            re.compile(r'\b(?:price\s+target)\s+(?:raised|increased|lowered|decreased|cut)\b', re.IGNORECASE),
            re.compile(r'\b(?:buy|sell|hold|strong\s+buy|strong\s+sell)\s+rating\b', re.IGNORECASE),
            re.compile(r'\b(?:analyst|analysts?)\s+(?:recommend|recommends?|expects?|forecasts?)\b', re.IGNORECASE)
        ]
        
        # Breaking news patterns
        patterns[EventType.BREAKING_NEWS] = [
            re.compile(r'\b(?:breaking|urgent|alert|just\s+in|developing)\b', re.IGNORECASE),
            re.compile(r'\b(?:emergency|immediate|critical|major\s+announcement)\b', re.IGNORECASE)
        ]
        
        return patterns
    
    def _load_company_indicators(self) -> List[str]:
        """Load indicators that suggest company names."""
        return [
            'inc', 'corp', 'corporation', 'company', 'ltd', 'llc', 'co',
            'group', 'holdings', 'enterprises', 'industries', 'systems',
            'technologies', 'tech', 'international', 'global', 'worldwide'
        ]
    
    def _compile_financial_patterns(self) -> Dict[str, re.Pattern]:
        """Compile patterns for extracting financial figures."""
        return {
            'revenue': re.compile(r'revenue\s+of\s+\$?([\d,]+(?:\.\d+)?)\s*(?:billion|million|B|M)?', re.IGNORECASE),
            'profit': re.compile(r'(?:profit|earnings|income)\s+of\s+\$?([\d,]+(?:\.\d+)?)\s*(?:billion|million|B|M)?', re.IGNORECASE),
            'eps': re.compile(r'(?:eps|earnings\s+per\s+share)\s+of\s+\$?([\d,]+(?:\.\d+)?)', re.IGNORECASE),
            'percentage': re.compile(r'([\d,]+(?:\.\d+)?)%'),
            'dollar_amount': re.compile(r'\$?([\d,]+(?:\.\d+)?)\s*(?:billion|million|B|M)', re.IGNORECASE),
            'share_price': re.compile(r'(?:share\s+price|stock\s+price)\s+(?:of\s+)?\$?([\d,]+(?:\.\d+)?)', re.IGNORECASE)
        }
    
    def _compile_temporal_patterns(self) -> List[re.Pattern]:
        """Compile patterns for extracting dates and times."""
        return [
            re.compile(r'\b(?:q[1-4]|quarter)\s+\d{4}\b', re.IGNORECASE),
            re.compile(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE),
            re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
            re.compile(r'\b(?:today|yesterday|tomorrow|this\s+week|next\s+week|last\s+week)\b', re.IGNORECASE),
            re.compile(r'\b(?:fiscal\s+year|fy)\s+\d{4}\b', re.IGNORECASE)
        ]
    
    def extract_events(self, text: str, title: str = None) -> EventExtractionResult:
        """
        Extract financial events from text content.
        
        Args:
            text: Article content
            title: Article title (optional)
            
        Returns:
            EventExtractionResult: Extracted events and metadata
        """
        start_time = datetime.now()
        
        if not text:
            return EventExtractionResult([], 0.0, 0, 0, 0)
        
        # Combine title and text for analysis
        full_text = f"{title or ''} {text}".strip()
        
        events = []
        
        # Extract events for each type
        for event_type, patterns in self.event_patterns.items():
            type_events = self._extract_events_by_type(full_text, event_type, patterns)
            events.extend(type_events)
        
        # Post-process events
        events = self._post_process_events(events, full_text)
        
        # Calculate metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        high_confidence_events = len([e for e in events if e.confidence > 0.7])
        
        return EventExtractionResult(
            events=events,
            processing_time=processing_time,
            text_length=len(text),
            events_found=len(events),
            high_confidence_events=high_confidence_events
        )
    
    def _extract_events_by_type(self, text: str, event_type: EventType, 
                               patterns: List[re.Pattern]) -> List[ExtractedEvent]:
        """Extract events of a specific type."""
        events = []
        
        for pattern in patterns:
            matches = pattern.finditer(text)
            
            for match in matches:
                event = ExtractedEvent(
                    event_type=event_type,
                    source_text=match.group(),
                    description=self._extract_description(text, match),
                    confidence=self._calculate_pattern_confidence(match, text)
                )
                
                # Extract additional details
                event.companies = self._extract_companies_near_match(text, match)
                event.stock_symbols = self._extract_stock_symbols_near_match(text, match)
                event.key_figures = self._extract_financial_figures_near_match(text, match)
                event.event_date = self._extract_date_near_match(text, match)
                event.severity = self._determine_severity(event, text)
                event.impact_indicators = self._extract_impact_indicators(text, match)
                
                events.append(event)
        
        return events
    
    def _extract_description(self, text: str, match: re.Match) -> str:
        """Extract a descriptive sentence around the match."""
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 100)
        context = text[start:end]
        
        # Find sentence boundaries
        sentences = re.split(r'[.!?]', context)
        if sentences:
            # Return the sentence containing the match
            for sentence in sentences:
                if match.group().lower() in sentence.lower():
                    return sentence.strip()
        
        return context.strip()
    
    def _calculate_pattern_confidence(self, match: re.Match, text: str) -> float:
        """Calculate confidence score for a pattern match."""
        confidence = 0.5  # Base confidence
        
        match_text = match.group().lower()
        
        # Boost confidence for specific keywords
        high_confidence_terms = ['announced', 'reported', 'confirmed', 'official']
        if any(term in match_text for term in high_confidence_terms):
            confidence += 0.2
        
        # Boost confidence for numbers/figures nearby
        context_start = max(0, match.start() - 50)
        context_end = min(len(text), match.end() + 50)
        context = text[context_start:context_end]
        
        if re.search(r'\$[\d,]+|\d+%|\d+\.\d+', context):
            confidence += 0.2
        
        # Reduce confidence for uncertain language
        uncertain_terms = ['rumored', 'alleged', 'possible', 'may', 'might', 'could']
        if any(term in match_text for term in uncertain_terms):
            confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))
    
    def _extract_companies_near_match(self, text: str, match: re.Match) -> List[str]:
        """Extract company names near the match."""
        # Get context around match
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 100)
        context = text[start:end]
        
        companies = []
        
        # Look for capitalized words that might be company names
        company_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Co|Company|Corporation)\b')
        company_matches = company_pattern.findall(context)
        companies.extend(company_matches)
        
        # Look for well-known company names (simple heuristic)
        words = context.split()
        for i, word in enumerate(words):
            if word.istitle() and len(word) > 2:
                # Check if next word is also capitalized (potential company name)
                if i + 1 < len(words) and words[i + 1].istitle():
                    potential_company = f"{word} {words[i + 1]}"
                    if len(potential_company) > 4:
                        companies.append(potential_company)
        
        return list(set(companies))[:3]  # Return up to 3 unique companies
    
    def _extract_stock_symbols_near_match(self, text: str, match: re.Match) -> List[str]:
        """Extract stock symbols near the match."""
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 100)
        context = text[start:end]
        
        symbols = []
        
        # Pattern for stock symbols
        symbol_patterns = [
            r'\$([A-Z]{1,5})\b',  # $AAPL
            r'\(([A-Z]{1,5})\)',  # (AAPL)
            r'\b([A-Z]{1,5}):\s*[A-Z]',  # AAPL: NASDAQ
        ]
        
        for pattern in symbol_patterns:
            matches = re.findall(pattern, context)
            symbols.extend(matches)
        
        return list(set(symbols))
    
    def _extract_financial_figures_near_match(self, text: str, match: re.Match) -> Dict[str, str]:
        """Extract financial figures near the match."""
        start = max(0, match.start() - 150)
        end = min(len(text), match.end() + 150)
        context = text[start:end]
        
        figures = {}
        
        for figure_type, pattern in self.financial_figures_pattern.items():
            matches = pattern.findall(context)
            if matches:
                figures[figure_type] = matches[0] if isinstance(matches[0], str) else matches[0][0]
        
        return figures
    
    def _extract_date_near_match(self, text: str, match: re.Match) -> Optional[datetime]:
        """Extract date information near the match."""
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 100)
        context = text[start:end]
        
        for pattern in self.temporal_patterns:
            date_match = pattern.search(context)
            if date_match:
                # Simple date parsing (could be enhanced)
                date_str = date_match.group()
                try:
                    # Handle some common formats
                    if 'q' in date_str.lower():
                        # Quarterly format like "Q3 2024"
                        return None  # Would need more sophisticated parsing
                    elif '/' in date_str or '-' in date_str:
                        # Try parsing date formats
                        return None  # Would need more sophisticated parsing
                except:
                    pass
        
        return None
    
    def _determine_severity(self, event: ExtractedEvent, text: str) -> EventSeverity:
        """Determine the severity of an event."""
        # Base severity by event type
        severity_map = {
            EventType.EARNINGS: EventSeverity.MEDIUM,
            EventType.MERGER_ACQUISITION: EventSeverity.HIGH,
            EventType.IPO: EventSeverity.HIGH,
            EventType.DIVIDEND: EventSeverity.LOW,
            EventType.BUYBACK: EventSeverity.MEDIUM,
            EventType.LAYOFFS: EventSeverity.MEDIUM,
            EventType.PRODUCT_LAUNCH: EventSeverity.LOW,
            EventType.REGULATORY: EventSeverity.HIGH,
            EventType.MONETARY_POLICY: EventSeverity.CRITICAL,
            EventType.BANKRUPTCY: EventSeverity.CRITICAL,
            EventType.BREAKING_NEWS: EventSeverity.HIGH,
        }
        
        base_severity = severity_map.get(event.event_type, EventSeverity.MEDIUM)
        
        # Adjust based on content
        text_lower = text.lower()
        
        # Upgrade severity for major indicators
        if any(term in text_lower for term in ['major', 'significant', 'historic', 'unprecedented']):
            if base_severity == EventSeverity.LOW:
                return EventSeverity.MEDIUM
            elif base_severity == EventSeverity.MEDIUM:
                return EventSeverity.HIGH
        
        # Upgrade for large financial figures
        if event.key_figures:
            for figure_type, value in event.key_figures.items():
                if 'billion' in value.lower():
                    return EventSeverity.HIGH
        
        return base_severity
    
    def _extract_impact_indicators(self, text: str, match: re.Match) -> List[str]:
        """Extract indicators of market impact."""
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 100)
        context = text[start:end].lower()
        
        impact_indicators = []
        
        # Market reaction indicators
        market_terms = ['surge', 'plunge', 'rally', 'crash', 'volatility', 'spike', 'drop']
        for term in market_terms:
            if term in context:
                impact_indicators.append(term)
        
        # Sentiment indicators
        sentiment_terms = ['positive', 'negative', 'bullish', 'bearish', 'optimistic', 'pessimistic']
        for term in sentiment_terms:
            if term in context:
                impact_indicators.append(term)
        
        return impact_indicators
    
    def _post_process_events(self, events: List[ExtractedEvent], text: str) -> List[ExtractedEvent]:
        """Post-process and filter events."""
        # Remove duplicate events
        unique_events = []
        seen_descriptions = set()
        
        for event in events:
            # Create a simple hash of the event
            event_hash = f"{event.event_type}_{event.description[:50]}"
            if event_hash not in seen_descriptions:
                seen_descriptions.add(event_hash)
                unique_events.append(event)
        
        # Sort by confidence
        unique_events.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit to top 10 events
        return unique_events[:10]
    
    def get_event_summary(self, events: List[ExtractedEvent]) -> Dict[str, any]:
        """Get summary statistics for extracted events."""
        if not events:
            return {'total_events': 0}
        
        event_types = {}
        severities = {}
        total_confidence = 0
        
        for event in events:
            # Count by type
            event_type = event.event_type.value
            event_types[event_type] = event_types.get(event_type, 0) + 1
            
            # Count by severity
            severity = event.severity.value
            severities[severity] = severities.get(severity, 0) + 1
            
            total_confidence += event.confidence
        
        avg_confidence = total_confidence / len(events)
        
        return {
            'total_events': len(events),
            'event_types': event_types,
            'severities': severities,
            'average_confidence': avg_confidence,
            'high_confidence_events': len([e for e in events if e.confidence > 0.7]),
            'critical_events': len([e for e in events if e.severity == EventSeverity.CRITICAL])
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the event extractor
    extractor = FinancialEventExtractor()
    
    # Sample financial news texts
    test_articles = [
        (
            "Apple Reports Q3 Earnings Beat",
            """Apple Inc. (AAPL) announced strong quarterly earnings results today, 
            reporting revenue of $94.8 billion and earnings per share of $1.85, beating 
            analyst estimates. The company also announced a $25 billion share buyback 
            program and raised its quarterly dividend to $0.95 per share. CEO Tim Cook 
            expressed optimism about the upcoming iPhone launch in Q4 2024."""
        ),
        (
            "Microsoft Acquires Gaming Studio for $2.5 Billion",
            """Microsoft Corporation today announced the acquisition of indie gaming 
            studio GameCorp for $2.5 billion in cash. The deal is expected to close 
            in Q1 2025, pending regulatory approval. This represents Microsoft's largest 
            gaming acquisition since the Activision Blizzard merger."""
        ),
        (
            "Fed Raises Interest Rates by 0.25%",
            """The Federal Reserve announced today that it is raising interest rates 
            by 0.25 percentage points, bringing the federal funds rate to 5.50%. 
            Fed Chair Jerome Powell cited persistent inflation concerns and strong 
            employment data as reasons for the monetary policy decision."""
        )
    ]
    
    print("=== Financial Event Extraction ===\n")
    
    for title, content in test_articles:
        print(f"Article: {title}")
        print("-" * 50)
        
        result = extractor.extract_events(content, title)
        
        print(f"Events found: {result.events_found}")
        print(f"High confidence events: {result.high_confidence_events}")
        print(f"Processing time: {result.processing_time:.3f}s")
        
        for i, event in enumerate(result.events, 1):
            print(f"\nEvent {i}:")
            print(f"  Type: {event.event_type.value}")
            print(f"  Severity: {event.severity.value}")
            print(f"  Confidence: {event.confidence:.3f}")
            print(f"  Companies: {event.companies}")
            print(f"  Stock symbols: {event.stock_symbols}")
            print(f"  Key figures: {event.key_figures}")
            print(f"  Description: {event.description[:100]}...")
        
        # Get summary
        summary = extractor.get_event_summary(result.events)
        print(f"\nEvent Summary: {summary}")
        print()
    
    print("Event extraction test completed!")