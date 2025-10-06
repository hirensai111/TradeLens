"""
Test event extractor with real articles from the database.

This script demonstrates event extraction on articles collected from NewsAPI.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_system.database import get_session
from news_system.processors.event_extractor import FinancialEventExtractor

def test_event_extraction():
    """Test event extraction with real articles from database."""
    
    print("=== Event Extraction Test with Real Articles ===\n")
    
    # Get articles from database
    session = get_session()
    articles = session.get_articles(limit=10)  # Get 10 recent articles
    
    if not articles:
        print("No articles found in database. Run news collection first!")
        return
    
    print(f"Found {len(articles)} articles to analyze for events\n")
    
    # Initialize event extractor
    extractor = FinancialEventExtractor()
    
    all_events = []
    
    # Extract events from each article
    for i, article in enumerate(articles, 1):
        print(f"Analyzing Article {i}: {article.title[:60]}...")
        
        try:
            # Extract events
            result = extractor.extract_events(article.content, article.title)
            
            if result.events_found > 0:
                print(f"  [OK] Found {result.events_found} events ({result.high_confidence_events} high confidence)")
                
                for j, event in enumerate(result.events, 1):
                    print(f"    Event {j}: {event.event_type.value} (confidence: {event.confidence:.3f})")
                    if event.companies:
                        print(f"      Companies: {', '.join(event.companies)}")
                    if event.stock_symbols:
                        print(f"      Stocks: {', '.join(event.stock_symbols)}")
                    if event.key_figures:
                        print(f"      Figures: {event.key_figures}")
                    if event.severity.value != 'medium':
                        print(f"      Severity: {event.severity.value}")
                    print(f"      Description: {event.description[:80]}...")
                
                all_events.extend(result.events)
                
                # Update article with event information
                if result.events:
                    # Store the most significant event type
                    primary_event = max(result.events, key=lambda x: x.confidence)
                    session.update_article(
                        article.id,
                        event_type=primary_event.event_type.value,
                        impact_score=min(primary_event.confidence, 1.0)
                    )
                    print(f"      Updated article with event: {primary_event.event_type.value}")
            else:
                print(f"  📄 No significant events detected")
                
        except Exception as e:
            print(f"  [ERROR] Error extracting events: {e}")
        
        print()
    
    # Overall event analysis
    if all_events:
        print("=== Overall Event Analysis ===")
        summary = extractor.get_event_summary(all_events)
        
        print(f"Total events extracted: {summary['total_events']}")
        print(f"Average confidence: {summary['average_confidence']:.3f}")
        print(f"High confidence events: {summary['high_confidence_events']}")
        print(f"Critical events: {summary['critical_events']}")
        
        print("\nEvent types found:")
        for event_type, count in summary['event_types'].items():
            print(f"  {event_type}: {count}")
        
        print("\nSeverity distribution:")
        for severity, count in summary['severities'].items():
            print(f"  {severity}: {count}")
        
        # Find most significant events
        high_impact_events = [e for e in all_events if e.confidence > 0.7 or e.severity.value in ['high', 'critical']]
        if high_impact_events:
            print(f"\n=== High Impact Events ({len(high_impact_events)}) ===")
            for event in high_impact_events:
                print(f"• {event.event_type.value} - {event.description[:60]}...")
                print(f"  Confidence: {event.confidence:.3f}, Severity: {event.severity.value}")
                if event.companies:
                    print(f"  Companies: {', '.join(event.companies)}")
                if event.stock_symbols:
                    print(f"  Stocks: {', '.join(event.stock_symbols)}")
                print()
    
    else:
        print("No events were extracted from the articles.")
    
    print("=== Event Extraction Test Complete ===")

def test_event_extractor_examples():
    """Test event extractor with predefined examples."""
    
    print("\n=== Testing Event Extractor with Sample Articles ===\n")
    
    extractor = FinancialEventExtractor()
    
    # Sample financial news with clear events
    sample_articles = [
        (
            "Tesla Q3 Earnings Beat Expectations",
            """Tesla Inc. (TSLA) reported third-quarter earnings that exceeded Wall Street 
            expectations, with revenue of $23.4 billion and earnings per share of $0.66. 
            The electric vehicle maker also announced a new $5 billion share buyback program. 
            CEO Elon Musk expressed confidence about achieving full-year delivery targets."""
        ),
        (
            "Microsoft to Acquire AI Startup for $10 Billion",
            """Microsoft Corporation announced today it has agreed to acquire artificial 
            intelligence startup TechCorp for $10 billion in cash and stock. The deal, 
            expected to close in Q2 2025, will strengthen Microsoft's AI capabilities 
            and expand its enterprise software offerings."""
        ),
        (
            "Federal Reserve Cuts Interest Rates by 0.5%",
            """The Federal Reserve announced an emergency 0.5 percentage point cut to 
            interest rates, bringing the federal funds rate to 4.75%. Fed Chair Jerome 
            Powell cited concerns about economic slowdown and declining inflation as 
            reasons for the aggressive monetary policy action."""
        ),
        (
            "Apple Launches Revolutionary VR Headset",
            """Apple Inc. unveiled its highly anticipated Vision Pro virtual reality 
            headset at today's product launch event. The device, priced at $3,499, 
            features advanced spatial computing capabilities and is expected to ship 
            in early 2024. Apple stock surged 5% in after-hours trading."""
        ),
        (
            "Meta Announces Major Layoffs Affecting 10,000 Employees",
            """Meta Platforms announced it will lay off approximately 10,000 employees, 
            representing about 13% of its workforce. The company cited challenging 
            economic conditions and the need to improve operational efficiency. 
            The layoffs will primarily affect engineering and business teams."""
        )
    ]
    
    all_sample_events = []
    
    for title, content in sample_articles:
        print(f"Analyzing: {title}")
        print("-" * 60)
        
        result = extractor.extract_events(content, title)
        
        print(f"Events found: {result.events_found}")
        print(f"Processing time: {result.processing_time:.3f}s")
        
        for i, event in enumerate(result.events, 1):
            print(f"\n  Event {i}: {event.event_type.value}")
            print(f"    Confidence: {event.confidence:.3f}")
            print(f"    Severity: {event.severity.value}")
            print(f"    Companies: {event.companies}")
            print(f"    Stock symbols: {event.stock_symbols}")
            print(f"    Key figures: {event.key_figures}")
            print(f"    Impact indicators: {event.impact_indicators}")
            print(f"    Description: {event.description}")
        
        all_sample_events.extend(result.events)
        print("\n" + "="*60 + "\n")
    
    # Summary of sample analysis
    if all_sample_events:
        summary = extractor.get_event_summary(all_sample_events)
        print("=== Sample Event Analysis Summary ===")
        print(f"Total events: {summary['total_events']}")
        print(f"Average confidence: {summary['average_confidence']:.3f}")
        print(f"Event types: {summary['event_types']}")
        print(f"Severities: {summary['severities']}")
    
    print("Sample event extraction test completed!")

if __name__ == "__main__":
    # Test with both real articles and samples
    test_event_extraction()
    test_event_extractor_examples()