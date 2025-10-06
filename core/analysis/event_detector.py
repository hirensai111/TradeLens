#!/usr/bin/env python3
"""
Simple Event Detector
Identifies significant price movements and creates events for visualization
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

class SimpleEventDetector:
    """Detect significant price movement events"""

    def __init__(self, threshold_pct: float = 5.0):
        """
        Initialize event detector

        Args:
            threshold_pct: Minimum price change percentage to consider as event
        """
        self.threshold_pct = threshold_pct

    def detect_events(self, raw_data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """
        Detect significant price events from historical data

        Args:
            raw_data: DataFrame with OHLCV data
            ticker: Stock ticker symbol

        Returns:
            Dictionary with events and summary
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
                event_type = 'Price_Movement' if abs(daily_return) < 5 else 'Significant_Move'
                sentiment = 'Bullish' if daily_return > 0 else 'Bearish'

                # Determine impact level
                if abs(daily_return) >= 10:
                    impact = 'HIGH'
                elif abs(daily_return) >= 7:
                    impact = 'MEDIUM'
                else:
                    impact = 'LOW'

                # Generate detailed description with technical context
                description = self._generate_event_description(
                    ticker, date, daily_return, row, raw_data, impact
                )

                # Create event
                events_list.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'type': event_type,
                    'description': description,
                    'sentiment': sentiment,
                    'confidence': min(95, 50 + abs(daily_return) * 3),  # Higher move = higher confidence
                    'impact': impact,
                    'price_change_pct': round(daily_return, 2),
                    'open_price': float(row.get('Open', 0)),
                    'close_price': float(row.get('Close', 0)),
                    'high_price': float(row.get('High', 0)),
                    'low_price': float(row.get('Low', 0)),
                    'volume': int(row.get('Volume', 0)),
                    'news_count': 0,  # Would be populated by news system
                    'sentiment_score': 50 + (daily_return * 2) if daily_return > 0 else 50 + (daily_return * 2),
                })

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

    def _generate_event_description(
        self, ticker: str, date: pd.Timestamp, daily_return: float,
        row: pd.Series, raw_data: pd.DataFrame, impact: str
    ) -> str:
        """Generate detailed event description with technical context"""

        direction = "surged" if daily_return > 0 else "dropped"
        close_price = row.get('Close', 0)
        open_price = row.get('Open', 0)
        volume = row.get('Volume', 0)

        # Get previous data for context
        try:
            idx = raw_data.index.get_loc(date)
            if idx > 0:
                prev_close = raw_data.iloc[idx-1]['Close']
                avg_volume = raw_data.iloc[max(0, idx-20):idx]['Volume'].mean()
            else:
                prev_close = open_price
                avg_volume = volume
        except:
            prev_close = open_price
            avg_volume = volume

        # Calculate volume comparison
        volume_pct = ((volume / avg_volume) - 1) * 100 if avg_volume > 0 else 0
        volume_context = ""
        if abs(volume_pct) > 50:
            volume_context = f" on {abs(volume_pct):.0f}% {'higher' if volume_pct > 0 else 'lower'} than average volume"
        elif abs(volume_pct) > 20:
            volume_context = f" with {'elevated' if volume_pct > 0 else 'reduced'} trading volume"

        # Get technical indicators if available
        technical_context = ""
        if 'RSI' in raw_data.columns and not pd.isna(row.get('RSI')):
            rsi = row['RSI']
            if rsi > 70:
                technical_context = ", with RSI indicating overbought conditions"
            elif rsi < 30:
                technical_context = ", with RSI indicating oversold conditions"

        # Check for multi-day trends
        trend_context = ""
        if idx > 2:
            last_3_returns = raw_data.iloc[idx-2:idx+1]['daily_return']
            if (last_3_returns > 0).all():
                trend_context = " This marks the third consecutive day of gains."
            elif (last_3_returns < 0).all():
                trend_context = " This extends a three-day losing streak."

        # Build comprehensive description
        description = (
            f"{ticker} stock {direction} {abs(daily_return):.2f}% on {date.strftime('%B %d, %Y')}, "
            f"closing at ${close_price:.2f} (from ${prev_close:.2f}){volume_context}"
            f"{technical_context}.{trend_context}"
        )

        # Add impact context
        if impact == 'HIGH':
            description += f" This represents a significant {abs(daily_return):.1f}% move that may indicate major news or market events."
        elif impact == 'MEDIUM':
            description += f" The {abs(daily_return):.1f}% movement suggests notable market attention or company developments."

        return description

    def _create_event_summary(self, events: List[Dict]) -> Dict:
        """Create summary statistics from events"""
        if not events:
            return {
                'total_events': 0,
                'positive_events': 0,
                'negative_events': 0,
                'avg_magnitude': 0,
                'largest_gain': None,
                'largest_loss': None
            }

        positive_events = [e for e in events if e['price_change_pct'] > 0]
        negative_events = [e for e in events if e['price_change_pct'] < 0]

        largest_gain = max(events, key=lambda x: x['price_change_pct']) if positive_events else None
        largest_loss = min(events, key=lambda x: x['price_change_pct']) if negative_events else None

        return {
            'total_events': len(events),
            'positive_events': len(positive_events),
            'negative_events': len(negative_events),
            'avg_magnitude': round(np.mean([abs(e['price_change_pct']) for e in events]), 2),
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
        """Analyze sentiment distribution from events"""
        sentiment_counts = {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }

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
                'largest_gain': None,
                'largest_loss': None
            },
            'sentiment_analysis': {
                'sentiment_counts': {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                },
                'sentiment_percentages': {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                }
            }
        }


# Quick test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__.parent.parent.parent))

    from prediction_engine.technical_analysis.data_processor import StockDataProcessor

    processor = StockDataProcessor()
    result = processor.process_stock('NVDA')

    detector = SimpleEventDetector(threshold_pct=3.0)
    events = detector.detect_events(result['raw_data'], 'NVDA')

    print(f"Found {len(events['events'])} significant events")
    print(f"Positive: {events['event_summary']['positive_events']}")
    print(f"Negative: {events['event_summary']['negative_events']}")

    if events['events']:
        print("\nMost recent events:")
        for event in events['events'][:5]:
            print(f"  {event['date']}: {event['type']} {event['price_change_pct']:+.2f}%")
