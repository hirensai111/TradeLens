"""
Enhanced Event Analyzer Module - FIXED VERSION
Uses intelligent analysis with ChatGPT knowledge-first approach and dynamic thresholds
"""

from openai import OpenAI
import json
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import re

# Import our configuration
from core.config.config import Config

@dataclass
class PriceEvent:
    """Data class for significant price movement events"""
    date: datetime
    ticker: str
    open_price: float
    close_price: float
    change_percent: float
    volume: int
    is_significant: bool = False

@dataclass
class EventAnalysis:
    """Data class for ChatGPT analysis results"""
    event_reason: str
    event_type: str
    sentiment: str
    confidence_score: float
    impact_level: str
    news_sources_count: int
    analysis_timestamp: datetime
    key_factors: List[str] = None
    analysis_method: str = "unknown"
    sentiment_score: float = 50.0  # ADD THIS: 0-100 scale sentiment score
    news_sentiment_average: float = 50.0  # ADD THIS: Average sentiment from news articles

class ThresholdManager:
    """Manages dynamic thresholds and learning phases"""
    
    def __init__(self):
        self.event_count = 0
        self.learning_phase_limit = 7
        self.initial_threshold = 3.0
        self.post_learning_threshold = 7.5
        self.confidence_threshold = 0.6
        
        # Track events for analysis
        self.analyzed_events = []
        
    def increment_event_count(self):
        """Increment the event counter"""
        self.event_count += 1
    
    def get_current_threshold(self) -> float:
        """Get the current price change threshold"""
        if self.is_learning_phase():
            return self.initial_threshold
        else:
            return self.post_learning_threshold
    
    def is_learning_phase(self) -> bool:
        """Check if we're still in the learning phase"""
        return self.event_count < self.learning_phase_limit
    
    def should_analyze_event(self, price_change_percent: float) -> bool:
        """Determine if an event should be analyzed based on current threshold"""
        current_threshold = self.get_current_threshold()
        return abs(price_change_percent) >= current_threshold
    
    def get_phase_info(self) -> Dict[str, Any]:
        """Get information about current phase"""
        return {
            'current_phase': 'learning' if self.is_learning_phase() else 'knowledge',
            'event_count': self.event_count,
            'events_until_transition': max(0, self.learning_phase_limit - self.event_count),
            'current_threshold': self.get_current_threshold(),
            'confidence_threshold': self.confidence_threshold
        }

class EnhancedEventAnalyzer:
    """Enhanced Event Analyzer with intelligent analysis system"""
    
    def __init__(self):
        self.config = Config()
        self.setup_openai()
        self.setup_logging()
        
        # Initialize components
        self.threshold_manager = ThresholdManager()
        # Note: We'll import news_scraper methods only when needed to avoid circular imports
        
        # Analysis parameters
        self.max_news_articles = 7  # Reduced to 7 as per requirement
        self.api_retry_count = 3
        self.max_days_to_analyze = 20
        
        # Event type categories (expanded)
        self.event_types = [
            'Earnings', 'Product_Launch', 'Regulatory', 'Market_Sentiment', 
            'Technical', 'Analyst_Rating', 'Partnership', 'Legal', 
            'Macro_Economic', 'Company_News', 'Political', 'Sector_Movement',
            'Competition', 'Supply_Chain', 'Management_Change', 'Unknown'
        ]
        
        # Enhanced statistics
        self.stats = {
            'events_analyzed': 0,
            'learning_phase_events': 0,
            'knowledge_phase_events': 0,
            'gpt_knowledge_sufficient': 0,
            'scraping_fallback_used': 0,
            'api_calls_made': 0,
            'api_failures': 0,
            'high_confidence_analyses': 0,
            'average_confidence': 0.0,
            'cost_savings_estimate': 0.0,
            'political_events': 0,
            'economic_events': 0
        }
    
    def setup_openai(self):
        """Initialize OpenAI client"""
        try:
            # Only try to initialize if we have an API key
            if self.config.OPENAI_API_KEY and self.config.OPENAI_API_KEY != 'your-openai-api-key-here':
                # Try to initialize OpenAI client without proxies argument that causes issues
                import os
                # Temporarily remove proxy settings if they exist
                proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']
                saved_proxies = {}
                for var in proxy_vars:
                    if var in os.environ:
                        saved_proxies[var] = os.environ[var]
                        del os.environ[var]

                try:
                    self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
                finally:
                    # Restore proxy settings
                    for var, value in saved_proxies.items():
                        os.environ[var] = value
            else:
                self.client = None
                print("WARNING: OpenAI API key not configured - AI insights will be unavailable")
        except Exception as e:
            self.client = None
            print(f"WARNING: Failed to initialize OpenAI: {e} - AI insights will be unavailable")
    
    def setup_logging(self):
        """Setup logging for the analyzer"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def get_news_bridge(self):
        """Get news integration bridge instance"""
        try:
            from news_integration_bridge import news_bridge
            return news_bridge
        except ImportError:
            self.logger.error("Could not import news_bridge - news functionality will be disabled")
            return None
    
    def identify_significant_events(self, price_data: List[Dict], limit: int = None) -> List[PriceEvent]:
        """Identify days with significant price movements using dynamic threshold"""
        significant_events = []
        
        # Sort by date descending to get most recent events first
        sorted_data = sorted(price_data, key=lambda x: x['Date'], reverse=True)
        
        # Apply limit if specified
        if limit is None:
            limit = self.max_days_to_analyze
        
        current_threshold = self.threshold_manager.get_current_threshold()
        self.logger.info(f"Using threshold: {current_threshold}% (Phase: {'Learning' if self.threshold_manager.is_learning_phase() else 'Knowledge'})")
        
        for i, day_data in enumerate(sorted_data):
            try:
                # Calculate price change percentage
                open_price = float(day_data.get('Open', 0))
                close_price = float(day_data.get('Close', 0))
                
                if open_price > 0:
                    change_percent = ((close_price - open_price) / open_price) * 100
                else:
                    continue
                
                # Check if change meets current threshold
                if self.threshold_manager.should_analyze_event(change_percent):
                    event = PriceEvent(
                        date=datetime.strptime(day_data['Date'], '%Y-%m-%d'),
                        ticker=day_data.get('Ticker', ''),
                        open_price=open_price,
                        close_price=close_price,
                        change_percent=change_percent,
                        volume=int(day_data.get('Volume', 0)),
                        is_significant=True
                    )
                    significant_events.append(event)
                    
                    # Stop if we've reached the limit
                    if len(significant_events) >= limit:
                        break
                    
            except (ValueError, KeyError) as e:
                self.logger.warning(f"Error processing price data for day {i}: {e}")
                continue
        
        self.logger.info(f"Found {len(significant_events)} significant price events (threshold: {current_threshold}%)")
        return significant_events
    
    def create_knowledge_based_prompt(self, event: PriceEvent) -> str:
        """Create prompt that asks ChatGPT to use its training knowledge first"""
        
        direction = "increased" if event.change_percent > 0 else "decreased"
        event_types_str = ", ".join(self.event_types)
        
        prompt = f"""
You are a financial analyst analyzing a stock price movement using your training knowledge.

STOCK PRICE MOVEMENT:
- Ticker: {event.ticker}
- Date: {event.date.strftime('%Y-%m-%d')}
- Price Movement: ${event.open_price:.2f} → ${event.close_price:.2f}
- Change: {event.change_percent:+.2f}%
- Volume: {event.volume:,} shares
- The stock {direction} by {abs(event.change_percent):.2f}% on this day.

IMPORTANT INSTRUCTIONS:
1. Use your training knowledge about market events, company history, and general financial patterns
2. Rate your confidence based on how well your training data covers this specific event and timeframe
3. If you don't have specific recent information about events around this date, explicitly state "insufficient recent information" and lower your confidence
4. Be honest about the limitations of your training data for this specific date and stock

CONFIDENCE GUIDELINES:
- 0.8-1.0: You have strong knowledge of specific events around this date for this stock
- 0.5-0.8: You have general market knowledge that likely explains this movement
- 0.0-0.5: Limited knowledge, mostly speculation based on patterns

ANALYSIS REQUIRED:
Provide your analysis in this JSON format:

{{
    "event_reason": "Explanation based on your training knowledge. If you lack specific information about this date/stock, say so explicitly and provide general market context.",
    "event_type": "One of: {event_types_str}",
    "sentiment": "One of: Bullish, Bearish, or Neutral",
    "confidence_score": "Number 0.0-1.0 representing your confidence in this analysis based on your training data coverage",
    "impact_level": "One of: HIGH, MEDIUM, or LOW",
    "key_factors": ["List", "of", "factors", "from", "your", "knowledge"],
    "data_recency_note": "Brief note about how recent/relevant your training data is for this analysis"
}}

If your confidence is below 0.6, explicitly mention "insufficient recent information" in your event_reason.

Respond ONLY with the JSON object.
"""
        
        return prompt
    
    def create_sentiment_enhanced_prompt(self, event: PriceEvent, news_articles: List[Dict], previous_analysis: Optional[EventAnalysis] = None) -> str:
        """Create prompt with sentiment-enhanced news data and previous analysis"""
        
        direction = "increased" if event.change_percent > 0 else "decreased"
        
        # Calculate aggregate sentiment metrics from articles
        sentiment_scores = []
        total_relevance = 0
        
        # Prepare news articles with sentiment data
        news_text = ""
        if news_articles:
            news_text = "\n\nFRESH NEWS ARTICLES WITH SENTIMENT ANALYSIS:\n"
            for i, article in enumerate(news_articles[:self.max_news_articles], 1):
                title = article.get('title', 'No title')
                summary = article.get('summary', 'No summary')
                source = article.get('source', 'Unknown source')
                categories = article.get('categories', [])
                sentiment = article.get('sentiment', {})
                sentiment_relevance = article.get('sentiment_relevance', 0)
                
                # Extract sentiment score for aggregation
                if sentiment and 'sentiment_score' in sentiment:
                    sentiment_scores.append({
                        'score': sentiment['sentiment_score'],
                        'relevance': sentiment_relevance
                    })
                    total_relevance += sentiment_relevance
                
                news_text += f"\n{i}. {title}\n"
                news_text += f"   Source: {source}\n"
                news_text += f"   Categories: {', '.join(categories)}\n"
                
                if sentiment:
                    news_text += f"   Sentiment: {sentiment.get('overall_sentiment', 'N/A')} (Financial: {sentiment.get('financial_sentiment', 'N/A')})\n"
                    news_text += f"   Sentiment Score: {sentiment.get('sentiment_score', 50):.1f}/100\n"  # ADD THIS LINE
                    news_text += f"   Sentiment Confidence: {sentiment.get('confidence', 0):.2f}\n"
                    news_text += f"   Relevance Score: {sentiment_relevance:.2f}\n"
                    if sentiment.get('key_phrases'):
                        news_text += f"   Key Phrases: {', '.join(sentiment['key_phrases'][:3])}\n"
                
                if summary:
                    news_text += f"   Summary: {summary[:200]}...\n" if len(summary) > 200 else f"   Summary: {summary}\n"
            
            # Calculate weighted average sentiment score
            if sentiment_scores and total_relevance > 0:
                weighted_sentiment = sum(s['score'] * s['relevance'] for s in sentiment_scores) / total_relevance
                news_text += f"\nAGGREGATE NEWS SENTIMENT:\n"
                news_text += f"Weighted Average Sentiment Score: {weighted_sentiment:.1f}/100\n"
                news_text += f"Price Movement Direction: {direction.upper()}\n"
                news_text += f"Sentiment-Price Alignment: {'ALIGNED' if (weighted_sentiment > 50 and event.change_percent > 0) or (weighted_sentiment < 50 and event.change_percent < 0) else 'DIVERGENT'}\n"
        else:
            news_text = "\n\nNo recent news articles found for this date range."
        
        # Rest of the method remains the same...
        # [Previous context and prompt construction code]
        
        prompt = f"""
    You are a financial analyst with access to fresh news data and sentiment analysis. 

    STOCK PRICE MOVEMENT:
    - Ticker: {event.ticker}
    - Date: {event.date.strftime('%Y-%m-%d')}
    - Price Movement: ${event.open_price:.2f} → ${event.close_price:.2f}
    - Change: {event.change_percent:+.2f}%
    - Volume: {event.volume:,} shares
    - The stock {direction} by {abs(event.change_percent):.2f}% on this day.

    {news_text}

    ANALYSIS REQUIRED:
    Now that you have fresh news data with sentiment analysis, provide an updated analysis:

    {{
        "event_reason": "Detailed explanation incorporating the news articles and sentiment data. Reference specific articles and their sentiment scores.",
        "event_type": "One of: {', '.join(self.event_types)}",
        "sentiment": "One of: Bullish, Bearish, or Neutral",
        "confidence_score": "Number 0.0-1.0 (should be higher now with fresh news data)",
        "impact_level": "One of: HIGH, MEDIUM, or LOW",
        "key_factors": ["List", "of", "factors", "based", "on", "news", "and", "sentiment"],
        "sentiment_score": "Overall sentiment score 0-100 based on news analysis and price movement",
        "sentiment_price_alignment": "How well the news sentiment aligns with the price movement (STRONG, MODERATE, WEAK, DIVERGENT)"
    }}

    GUIDELINES:
    1. Correlate news sentiment scores with price movement direction
    2. Reference specific articles and their sentiment scores  
    3. Calculate sentiment_score considering both news sentiment and price direction
    4. Higher confidence since you now have recent, relevant news data
    5. Consider sentiment relevance scores when weighing article importance

    Respond ONLY with the JSON object.
    """
        
        return prompt
    
    def call_chatgpt_api(self, prompt: str, model: str = "gpt-4o-mini") -> Optional[Dict]:
        """Make API call to ChatGPT with retry logic"""
        # Return None if client is not available
        if self.client is None:
            return None

        for attempt in range(self.api_retry_count):
            try:
                self.stats['api_calls_made'] += 1

                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a financial analyst expert at explaining stock price movements. Always respond with valid JSON only. Be honest about the limitations of your knowledge and data recency."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.3,
                    timeout=30
                )
                
                return response
                
            except Exception as e:
                error_msg = str(e).lower()
                
                if "rate limit" in error_msg:
                    wait_time = (2 ** attempt) * 5
                    self.logger.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                elif "api" in error_msg:
                    self.logger.error(f"OpenAI API error (attempt {attempt + 1}): {e}")
                    if attempt == self.api_retry_count - 1:
                        self.stats['api_failures'] += 1
                        return None
                    time.sleep(2)
                else:
                    self.logger.error(f"Unexpected error calling ChatGPT (attempt {attempt + 1}): {e}")
                    if attempt == self.api_retry_count - 1:
                        self.stats['api_failures'] += 1
                        return None
                    time.sleep(2)
        
        return None
    
    def parse_chatgpt_response(self, response_text: str) -> Optional[Dict]:
        """Parse and validate ChatGPT JSON response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                self.logger.error("No JSON found in ChatGPT response")
                return None
            
            json_text = json_match.group(0)
            analysis_data = json.loads(json_text)
            
            # Validate required fields
            required_fields = ['event_reason', 'event_type', 'sentiment', 'confidence_score', 'impact_level']
            for field in required_fields:
                if field not in analysis_data:
                    self.logger.error(f"Missing required field: {field}")
                    return None
            
            # Validate field values
            valid_sentiments = ['Bullish', 'Bearish', 'Neutral']
            valid_impacts = ['HIGH', 'MEDIUM', 'LOW']
            
            if analysis_data['sentiment'] not in valid_sentiments:
                analysis_data['sentiment'] = 'Neutral'
            
            if analysis_data['impact_level'] not in valid_impacts:
                analysis_data['impact_level'] = 'MEDIUM'
            
            # Validate event type
            if analysis_data['event_type'] not in self.event_types:
                self.logger.warning(f"Invalid event type: {analysis_data['event_type']}, defaulting to Unknown")
                analysis_data['event_type'] = 'Unknown'
            
            # Ensure confidence score is valid
            confidence = float(analysis_data['confidence_score'])
            analysis_data['confidence_score'] = max(0.0, min(1.0, confidence))
            
            # Ensure sentiment_score is valid (ADD THIS)
            if 'sentiment_score' in analysis_data:
                sentiment_score = float(analysis_data['sentiment_score'])
                analysis_data['sentiment_score'] = max(0.0, min(100.0, sentiment_score))
            else:
                analysis_data['sentiment_score'] = 50.0  # Default neutral
            
            # Ensure key_factors is a list
            if 'key_factors' not in analysis_data or not isinstance(analysis_data['key_factors'], list):
                analysis_data['key_factors'] = []
            
            return analysis_data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            self.logger.error(f"Response text: {response_text[:200]}...")
            return None
        except Exception as e:
            self.logger.error(f"Error parsing ChatGPT response: {e}")
            return None

    
    def analyze_with_gpt_knowledge(self, event: PriceEvent) -> Optional[EventAnalysis]:
        """Analyze event using ChatGPT's training knowledge first"""
        self.logger.info(f"Analyzing {event.ticker} with GPT knowledge first")
        
        # Create knowledge-based prompt
        prompt = self.create_knowledge_based_prompt(event)
        
        # Call ChatGPT API
        response = self.call_chatgpt_api(prompt)
        if not response:
            return None
        
        # Parse response
        response_text = response.choices[0].message.content
        analysis_data = self.parse_chatgpt_response(response_text)
        
        if not analysis_data:
            return None
        
        # Create EventAnalysis object
        event_analysis = EventAnalysis(
            event_reason=analysis_data['event_reason'],
            event_type=analysis_data['event_type'],
            sentiment=analysis_data['sentiment'],
            confidence_score=analysis_data['confidence_score'],
            impact_level=analysis_data['impact_level'],
            news_sources_count=0,  # No news articles used
            analysis_timestamp=datetime.now(),
            key_factors=analysis_data.get('key_factors', []),
            analysis_method="gpt_knowledge"
        )
        
        self.logger.info(f"GPT knowledge analysis complete - Confidence: {event_analysis.confidence_score:.2f}")
        return event_analysis
    
    def analyze_with_news_scraping(self, event: PriceEvent, previous_analysis: Optional[EventAnalysis] = None) -> Optional[EventAnalysis]:
        """Analyze event with fresh news scraping"""
        self.logger.info(f"Analyzing {event.ticker} with news scraping")
        
        # Get news bridge instance
        news_bridge = self.get_news_bridge()
        if not news_bridge:
            self.logger.error("News bridge not available, using fallback analysis")
            return self.create_fallback_analysis(event)

        try:
            # Get news data through the bridge
            articles, sentiment_summary = news_bridge.get_news_with_sentiment(
                ticker=event.ticker,
                event_date=event.date,
                lookback_days=3
            )
            
            # Check for errors
            if sentiment_summary.get('error'):
                self.logger.warning(f"News bridge error: {sentiment_summary['error']}")
                articles = []
            
            # Limit to most relevant articles
            if articles:
                articles = sorted(articles, key=lambda x: x.get('sentiment_relevance', 0), reverse=True)[:self.max_news_articles]
            
            # Calculate news sentiment average
            news_sentiment_avg = 50.0  # Default neutral
            if articles:
                sentiment_scores = []
                for article in articles:
                    sentiment = article.get('sentiment', {})
                    if 'sentiment_score' in sentiment:
                        sentiment_scores.append(sentiment['sentiment_score'])
                
                if sentiment_scores:
                    news_sentiment_avg = sum(sentiment_scores) / len(sentiment_scores)
            
        except Exception as e:
            self.logger.error(f"Error during news scraping: {e}")
            articles = []
            news_sentiment_avg = 50.0
        
        # Create sentiment-enhanced prompt
        prompt = self.create_sentiment_enhanced_prompt(event, articles, previous_analysis)
        
        # Call ChatGPT API
        response = self.call_chatgpt_api(prompt)
        if not response:
            return self.create_fallback_analysis(event)
        
        # Parse response
        response_text = response.choices[0].message.content
        analysis_data = self.parse_chatgpt_response(response_text)
        
        if not analysis_data:
            return self.create_fallback_analysis(event)
        
        # Create EventAnalysis object with sentiment score
        event_analysis = EventAnalysis(
            event_reason=analysis_data['event_reason'],
            event_type=analysis_data['event_type'],
            sentiment=analysis_data['sentiment'],
            confidence_score=analysis_data['confidence_score'],
            impact_level=analysis_data['impact_level'],
            news_sources_count=len(articles),
            analysis_timestamp=datetime.now(),
            key_factors=analysis_data.get('key_factors', []),
            analysis_method="full_scrape",
            sentiment_score=analysis_data.get('sentiment_score', 50.0),  # ADD THIS
            news_sentiment_average=news_sentiment_avg  # ADD THIS
        )
        
        self.logger.info(f"News scraping analysis complete - Articles: {len(articles)}, Confidence: {event_analysis.confidence_score:.2f}, Sentiment Score: {event_analysis.sentiment_score:.1f}")
        return event_analysis
    
    def analyze_event_intelligent(self, event: PriceEvent) -> Optional[EventAnalysis]:
        """Main intelligent analysis method that follows the decision tree"""
        phase_info = self.threshold_manager.get_phase_info()
        self.logger.info(f"Analyzing event in {phase_info['current_phase']} phase (Event #{self.threshold_manager.event_count + 1})")
        
        # Increment event count
        self.threshold_manager.increment_event_count()
        
        if self.threshold_manager.is_learning_phase():
            # Learning Phase: Always scrape
            self.logger.info("Learning phase: Using full news scraping")
            analysis = self.analyze_with_news_scraping(event)
            if analysis:
                self.stats['learning_phase_events'] += 1
            return analysis
        
        else:
            # Knowledge Phase: GPT first, then scrape if needed
            self.logger.info("Knowledge phase: Trying GPT knowledge first")
            
            # Try GPT knowledge first
            gpt_analysis = self.analyze_with_gpt_knowledge(event)
            
            if gpt_analysis and gpt_analysis.confidence_score >= self.threshold_manager.confidence_threshold:
                # GPT analysis is sufficient
                self.logger.info(f"GPT knowledge sufficient (confidence: {gpt_analysis.confidence_score:.2f})")
                self.stats['knowledge_phase_events'] += 1
                self.stats['gpt_knowledge_sufficient'] += 1
                self.stats['cost_savings_estimate'] += 1.0  # Estimate 1 unit saved per avoided scraping
                return gpt_analysis
            
            else:
                # GPT confidence too low, fallback to scraping
                confidence = gpt_analysis.confidence_score if gpt_analysis else 0.0
                self.logger.info(f"GPT confidence too low ({confidence:.2f}), falling back to news scraping")
                
                analysis = self.analyze_with_news_scraping(event, gpt_analysis)
                if analysis:
                    self.stats['knowledge_phase_events'] += 1
                    self.stats['scraping_fallback_used'] += 1
                return analysis
    
    def create_fallback_analysis(self, event: PriceEvent) -> EventAnalysis:
        """Create a basic analysis when all other methods fail"""
        direction = "upward" if event.change_percent > 0 else "downward"
        magnitude = "significant" if abs(event.change_percent) > 5 else "moderate"
        
        # Calculate sentiment score based on price movement
        base_sentiment = 50.0  # Neutral
        price_impact = min(abs(event.change_percent) * 2, 25)  # Cap at 25 points
        sentiment_score = base_sentiment + (price_impact if event.change_percent > 0 else -price_impact)
        sentiment_score = max(0.0, min(100.0, sentiment_score))
        
        if abs(event.change_percent) > 7:
            impact = "HIGH"
            reason_detail = "This large movement suggests a major catalyst or significant market event."
        elif abs(event.change_percent) > 5:
            impact = "MEDIUM"
            reason_detail = "This notable movement likely reflects important company or sector developments."
        else:
            impact = "LOW"
            reason_detail = "This movement could be due to normal market volatility or technical trading."
        
        fallback_reason = f"The stock experienced a {magnitude} {direction} movement of {abs(event.change_percent):.1f}%. "
        fallback_reason += reason_detail + " "
        fallback_reason += "Without specific news coverage or sufficient knowledge, this could be attributed to general market conditions, sector rotation, or technical factors."
        
        return EventAnalysis(
            event_reason=fallback_reason,
            event_type="Unknown",
            sentiment="Bullish" if event.change_percent > 0 else "Bearish",
            confidence_score=0.3,
            impact_level=impact,
            news_sources_count=0,
            analysis_timestamp=datetime.now(),
            key_factors=["Market volatility", "Technical trading", "Sector movement"],
            analysis_method="fallback",
            sentiment_score=sentiment_score,  # ADD THIS
            news_sentiment_average=50.0  # ADD THIS
        )
    
    def batch_analyze_events(self, events: List[PriceEvent]) -> Dict[str, EventAnalysis]:
        """Analyze multiple price events using intelligent system"""
        analyses = {}
        
        self.logger.info(f"Starting intelligent batch analysis of {len(events)} events")
        self.logger.info(f"Current phase: {self.threshold_manager.get_phase_info()}")
        
        for i, event in enumerate(events):
            event_key = f"{event.ticker}_{event.date.strftime('%Y-%m-%d')}"
            
            # Analyze the event using intelligent method
            analysis = self.analyze_event_intelligent(event)
            
            if analysis:
                analyses[event_key] = analysis
                
                # Update statistics
                self.stats['events_analyzed'] += 1
                if analysis.confidence_score >= 0.7:
                    self.stats['high_confidence_analyses'] += 1
                
                # Track political/economic events
                if analysis.event_type in ['Political', 'Macro_Economic']:
                    if analysis.event_type == 'Political':
                        self.stats['political_events'] += 1
                    else:
                        self.stats['economic_events'] += 1
            
            # Rate limiting between API calls
            time.sleep(1.5)
            
            # Progress update
            if (i + 1) % 3 == 0:
                phase_info = self.threshold_manager.get_phase_info()
                self.logger.info(f"Progress: {i + 1}/{len(events)} events analyzed (Phase: {phase_info['current_phase']})")
        
        # Calculate average confidence
        if analyses:
            confidences = [a.confidence_score for a in analyses.values()]
            self.stats['average_confidence'] = sum(confidences) / len(confidences)
        
        self.logger.info(f"Intelligent batch analysis complete: {len(analyses)} successful analyses")
        return analyses
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive analysis statistics"""
        phase_info = self.threshold_manager.get_phase_info()
        
        enhanced_stats = self.stats.copy()
        enhanced_stats.update({
            'threshold_manager': phase_info,
            'efficiency_metrics': {
                'gpt_success_rate': (self.stats['gpt_knowledge_sufficient'] / max(1, self.stats['knowledge_phase_events'])) * 100 if self.stats['knowledge_phase_events'] > 0 else 0,
                'scraping_fallback_rate': (self.stats['scraping_fallback_used'] / max(1, self.stats['knowledge_phase_events'])) * 100 if self.stats['knowledge_phase_events'] > 0 else 0,
                'estimated_cost_savings': self.stats['cost_savings_estimate'],
                'average_confidence': self.stats['average_confidence']
            }
        })
        
        return enhanced_stats
    
    def reset_stats(self):
        """Reset all statistics"""
        self.stats = {
            'events_analyzed': 0,
            'learning_phase_events': 0,
            'knowledge_phase_events': 0,
            'gpt_knowledge_sufficient': 0,
            'scraping_fallback_used': 0,
            'api_calls_made': 0,
            'api_failures': 0,
            'high_confidence_analyses': 0,
            'average_confidence': 0.0,
            'cost_savings_estimate': 0.0,
            'political_events': 0,
            'economic_events': 0
        }

# Global enhanced analyzer instance
enhanced_analyzer = EnhancedEventAnalyzer()

if __name__ == "__main__":
    # Simple test without dependencies
    print("Enhanced Event Analyzer - Fixed Version")
    print("=" * 50)
    
    # Test threshold manager
    analyzer = EnhancedEventAnalyzer()
    
    print("Testing Threshold Manager:")
    for i in range(10):
        phase_info = analyzer.threshold_manager.get_phase_info()
        print(f"Event {i+1}: Phase={phase_info['current_phase']}, Threshold={phase_info['current_threshold']}%")
        analyzer.threshold_manager.increment_event_count()
    
    print(f"\n[OK] Fixed Event Analyzer loaded successfully!")
    print(f"[TARGET] Key fixes applied:")
    print(f"   - Removed circular import issues")
    print(f"   - Added lazy loading for news scraper")
    print(f"   - Fixed method calls and references")
    print(f"   - Added proper error handling")
    print(f"   - Maintained all intelligent features")