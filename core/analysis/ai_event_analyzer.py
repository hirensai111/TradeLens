#!/usr/bin/env python3
"""
AI-Enhanced Event Analyzer
Integrates ChatGPT for detailed event analysis and reasoning
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from openai import OpenAI
import json
import re
import logging

class AIEventAnalyzer:
    """Event analyzer with ChatGPT integration for detailed reasoning"""

    def __init__(self, api_key: Optional[str] = None, threshold_pct: float = 3.0):
        """
        Initialize AI event analyzer

        Args:
            api_key: OpenAI API key (reads from config if not provided)
            threshold_pct: Minimum price change percentage to consider as event
        """
        self.threshold_pct = threshold_pct
        self.setup_logging()
        self.setup_openai(api_key)

    def setup_openai(self, api_key: Optional[str]):
        """Initialize OpenAI client"""
        try:
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                # Try to get from config
                try:
                    from core.config.config import config
                    if config.OPENAI_API_KEY and config.OPENAI_API_KEY != 'your-openai-api-key-here':
                        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
                    else:
                        self.client = None
                except:
                    self.client = None

            if self.client:
                self.logger.info("✓ OpenAI client initialized successfully")
            else:
                self.logger.warning("⚠ OpenAI not configured - using basic analysis only")
        except Exception as e:
            self.client = None
            self.logger.warning(f"⚠ Failed to initialize OpenAI: {e}")

    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def detect_events(self, raw_data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """
        Detect and analyze significant price events with AI reasoning

        Args:
            raw_data: DataFrame with OHLCV data
            ticker: Stock ticker symbol

        Returns:
            Dictionary with events, summaries, and AI analysis
        """
        if raw_data.empty or 'Close' not in raw_data.columns:
            return self._empty_events(ticker)

        events_list = []

        # Calculate daily returns
        raw_data = raw_data.copy()
        raw_data['daily_return'] = raw_data['Close'].pct_change() * 100

        # Find significant moves (last 90 days)
        recent_data = raw_data.tail(90)

        for date, row in recent_data.iterrows():
            daily_return = row.get('daily_return', 0)

            if abs(daily_return) >= self.threshold_pct:
                # Generate AI-powered analysis
                analysis = self._analyze_event_with_ai(
                    ticker, date, daily_return, row, raw_data
                )

                events_list.append(analysis)

        # Sort by date descending
        events_list.sort(key=lambda x: x['date'], reverse=True)

        # Create summary
        event_summary = self._create_event_summary(events_list)
        sentiment_analysis = self._analyze_sentiment(events_list)

        return {
            'ticker': ticker,
            'last_updated': datetime.now().isoformat(),
            'events': events_list,
            'event_summary': event_summary,
            'sentiment_analysis': sentiment_analysis
        }

    def _analyze_event_with_ai(
        self, ticker: str, date: pd.Timestamp, daily_return: float,
        row: pd.Series, raw_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze event with ChatGPT for detailed reasoning"""

        # Determine impact level
        if abs(daily_return) >= 10:
            impact = 'HIGH'
        elif abs(daily_return) >= 7:
            impact = 'MEDIUM'
        else:
            impact = 'LOW'

        # Get technical context
        technical_context = self._get_technical_context(date, row, raw_data)

        # Try AI analysis if available
        if self.client:
            ai_analysis = self._get_ai_reasoning(
                ticker, date, daily_return, row, technical_context
            )
            if ai_analysis:
                return self._build_event_with_ai(
                    date, daily_return, row, impact, ai_analysis
                )

        # Fallback to basic analysis
        return self._build_basic_event(
            ticker, date, daily_return, row, raw_data, impact, technical_context
        )

    def _get_ai_reasoning(
        self, ticker: str, date: pd.Timestamp, daily_return: float,
        row: pd.Series, technical_context: Dict
    ) -> Optional[Dict]:
        """Get AI-powered reasoning from ChatGPT"""

        prompt = f"""Analyze this stock price movement and provide detailed reasoning:

Stock: {ticker}
Date: {date.strftime('%Y-%m-%d')}
Price Change: {daily_return:+.2f}%
Close Price: ${row['Close']:.2f}
Volume: {int(row['Volume']):,}

Technical Context:
{json.dumps(technical_context, indent=2)}

Provide a JSON response with:
{{
  "event_reason": "Detailed explanation of what likely caused this movement (2-3 sentences)",
  "event_type": "One of: Earnings, Product_Launch, Market_Sentiment, Technical, Regulatory, Partnership, Macro_Economic, Company_News, Unknown",
  "sentiment": "Bullish, Bearish, or Neutral",
  "confidence_score": 0.0-1.0 (how confident you are in this analysis),
  "impact_level": "HIGH, MEDIUM, or LOW",
  "key_factors": ["factor1", "factor2", "factor3"],
  "sentiment_score": 0-100 (0=very bearish, 50=neutral, 100=very bullish)
}}

Be honest about your limitations - if you don't know the specific reason, explain based on technical analysis and general market context."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst expert at explaining stock price movements. Always respond with valid JSON only."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3,
                timeout=30
            )

            response_text = response.choices[0].message.content
            return self._parse_ai_response(response_text)

        except Exception as e:
            self.logger.warning(f"AI analysis failed: {e}")
            return None

    def _parse_ai_response(self, response_text: str) -> Optional[Dict]:
        """Parse AI JSON response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                return None

            json_text = json_match.group(0)
            analysis_data = json.loads(json_text)

            # Validate required fields
            required = ['event_reason', 'event_type', 'sentiment', 'confidence_score', 'impact_level']
            if not all(field in analysis_data for field in required):
                return None

            # Validate confidence
            analysis_data['confidence_score'] = max(0.0, min(1.0, float(analysis_data['confidence_score'])))

            # Validate sentiment_score
            if 'sentiment_score' not in analysis_data:
                analysis_data['sentiment_score'] = 50.0
            else:
                analysis_data['sentiment_score'] = max(0.0, min(100.0, float(analysis_data['sentiment_score'])))

            return analysis_data

        except Exception as e:
            self.logger.warning(f"Failed to parse AI response: {e}")
            return None

    def _get_technical_context(self, date: pd.Timestamp, row: pd.Series, raw_data: pd.DataFrame) -> Dict:
        """Get technical indicators context"""
        context = {}

        try:
            idx = raw_data.index.get_loc(date)

            # RSI
            if 'RSI' in raw_data.columns and not pd.isna(row.get('RSI')):
                context['rsi'] = float(row['RSI'])

            # Moving averages
            for ma in ['SMA_20', 'SMA_50', 'SMA_200']:
                if ma in raw_data.columns and not pd.isna(row.get(ma)):
                    context[ma.lower()] = float(row[ma])

            # Volume trend
            if idx > 20:
                avg_volume = raw_data.iloc[idx-20:idx]['Volume'].mean()
                context['volume_vs_avg'] = f"{((row['Volume'] / avg_volume - 1) * 100):.1f}%"

            # Multi-day trend
            if idx > 2:
                last_3_returns = raw_data.iloc[idx-2:idx+1]['daily_return']
                if (last_3_returns > 0).all():
                    context['trend'] = "3-day winning streak"
                elif (last_3_returns < 0).all():
                    context['trend'] = "3-day losing streak"

        except:
            pass

        return context

    def _build_event_with_ai(
        self, date: pd.Timestamp, daily_return: float,
        row: pd.Series, impact: str, ai_analysis: Dict
    ) -> Dict[str, Any]:
        """Build event dict with AI analysis"""

        confidence_pct = ai_analysis['confidence_score'] * 100

        return {
            'date': date.strftime('%Y-%m-%d'),
            'type': ai_analysis['event_type'],
            'description': ai_analysis['event_reason'],
            'sentiment': ai_analysis['sentiment'],
            'confidence': confidence_pct,
            'impact': impact,
            'price_change_pct': round(daily_return, 2),
            'open_price': float(row.get('Open', 0)),
            'close_price': float(row.get('Close', 0)),
            'high_price': float(row.get('High', 0)),
            'low_price': float(row.get('Low', 0)),
            'volume': int(row.get('Volume', 0)),
            'news_count': 0,
            'sentiment_score': ai_analysis['sentiment_score'],
            'analysis_method': 'ai_powered',
            'key_factors': ai_analysis.get('key_factors', [])
        }

    def _build_basic_event(
        self, ticker: str, date: pd.Timestamp, daily_return: float,
        row: pd.Series, raw_data: pd.DataFrame, impact: str, technical_context: Dict
    ) -> Dict[str, Any]:
        """Build event with basic technical analysis (fallback)"""

        direction = "surged" if daily_return > 0 else "dropped"
        sentiment = 'Bullish' if daily_return > 0 else 'Bearish'

        # Build description from technical context
        description = (
            f"{ticker} stock {direction} {abs(daily_return):.2f}% on {date.strftime('%B %d, %Y')}, "
            f"closing at ${row['Close']:.2f}."
        )

        if 'volume_vs_avg' in technical_context:
            description += f" Trading volume was {technical_context['volume_vs_avg']} vs average."

        if 'trend' in technical_context:
            description += f" {technical_context['trend']}."

        return {
            'date': date.strftime('%Y-%m-%d'),
            'type': 'Price_Movement',
            'description': description,
            'sentiment': sentiment,
            'confidence': min(95, 50 + abs(daily_return) * 3),
            'impact': impact,
            'price_change_pct': round(daily_return, 2),
            'open_price': float(row.get('Open', 0)),
            'close_price': float(row.get('Close', 0)),
            'high_price': float(row.get('High', 0)),
            'low_price': float(row.get('Low', 0)),
            'volume': int(row.get('Volume', 0)),
            'news_count': 0,
            'sentiment_score': 50 + (daily_return * 2),
            'analysis_method': 'technical'
        }

    def _create_event_summary(self, events: List[Dict]) -> Dict:
        """Create summary statistics"""
        if not events:
            return {
                'total_events': 0,
                'positive_events': 0,
                'negative_events': 0,
                'avg_magnitude': 0,
                'ai_powered_events': 0,
                'largest_gain': None,
                'largest_loss': None
            }

        positive_events = [e for e in events if e['price_change_pct'] > 0]
        negative_events = [e for e in events if e['price_change_pct'] < 0]
        ai_events = [e for e in events if e.get('analysis_method') == 'ai_powered']

        largest_gain = max(events, key=lambda x: x['price_change_pct']) if positive_events else None
        largest_loss = min(events, key=lambda x: x['price_change_pct']) if negative_events else None

        return {
            'total_events': len(events),
            'positive_events': len(positive_events),
            'negative_events': len(negative_events),
            'avg_magnitude': round(np.mean([abs(e['price_change_pct']) for e in events]), 2),
            'ai_powered_events': len(ai_events),
            'largest_gain': {
                'date': largest_gain['date'],
                'change_pct': largest_gain['price_change_pct']
            } if largest_gain else None,
            'largest_loss': {
                'date': largest_loss['date'],
                'change_pct': largest_loss['price_change_pct']
            } if largest_loss else None
        }

    def _analyze_sentiment(self, events: List[Dict]) -> Dict:
        """Analyze sentiment distribution"""
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}

        for event in events:
            sentiment = event.get('sentiment', 'Neutral').lower()
            if sentiment == 'bullish':
                sentiment_counts['positive'] += 1
            elif sentiment == 'bearish':
                sentiment_counts['negative'] += 1
            else:
                sentiment_counts['neutral'] += 1

        total = sum(sentiment_counts.values())
        if total > 0:
            sentiment_percentages = {
                k: round((v / total) * 100, 1)
                for k, v in sentiment_counts.items()
            }
        else:
            sentiment_percentages = {k: 0 for k in sentiment_counts.keys()}

        return {
            'sentiment_counts': sentiment_counts,
            'sentiment_percentages': sentiment_percentages
        }

    def _empty_events(self, ticker: str) -> Dict:
        """Return empty events structure"""
        return {
            'ticker': ticker,
            'last_updated': datetime.now().isoformat(),
            'events': [],
            'event_summary': {
                'total_events': 0,
                'positive_events': 0,
                'negative_events': 0,
                'avg_magnitude': 0,
                'ai_powered_events': 0,
                'largest_gain': None,
                'largest_loss': None
            },
            'sentiment_analysis': {
                'sentiment_counts': {'positive': 0, 'negative': 0, 'neutral': 0},
                'sentiment_percentages': {'positive': 0, 'negative': 0, 'neutral': 0}
            }
        }
