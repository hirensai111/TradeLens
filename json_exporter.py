"""
JSON Data Exporter for Stock Analyzer
Exports analysis data in JSON format for web dashboard visualization
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from config import config
from utils import get_logger


class VisualizationDataExporter:
    """Exports stock analysis data to JSON format for frontend visualization."""

    def __init__(self):
        self.logger = get_logger("VisualizationDataExporter")
        self.output_dir = config.VIZ_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all_data(self, data_bundle: Dict[str, Any], ticker: str) -> Dict[str, str]:
        """
        Export all data types for a given ticker.

        Args:
            data_bundle: Complete analysis data from StockDataProcessor
            ticker: Stock ticker symbol

        Returns:
            Dictionary mapping data type to file path
        """
        try:
            self.logger.info(f"Starting JSON export for {ticker}")

            exported_files = {}

            # Export price data with technical indicators
            exported_files['price_data'] = self.export_price_data(data_bundle, ticker)

            # Export events data with sentiment analysis
            exported_files['events'] = self.export_events_data(data_bundle, ticker)

            # Export summary data with key metrics
            exported_files['summary'] = self.export_summary_data(data_bundle, ticker)

            # Export company information
            exported_files['company'] = self.export_company_data(data_bundle, ticker)

            self.logger.info(f"JSON export completed for {ticker}")
            return exported_files

        except Exception as e:
            self.logger.error(f"Failed to export JSON data for {ticker}: {e}")
            raise

    def export_price_data(self, data_bundle: Dict[str, Any], ticker: str) -> str:
        """
        Export OHLCV data with technical indicators.

        Args:
            data_bundle: Analysis data bundle
            ticker: Stock ticker symbol

        Returns:
            Path to exported JSON file
        """
        try:
            # Get raw data and technical indicators
            raw_data = data_bundle.get('raw_data', pd.DataFrame())
            technical_indicators = data_bundle.get('technical_indicators', {})

            if raw_data.empty:
                raise ValueError(f"No raw data available for {ticker}")

            # Prepare price data
            price_data = {
                'ticker': ticker,
                'last_updated': datetime.now().isoformat(),
                'data_period': {
                    'start_date': raw_data.index.min().isoformat() if not raw_data.empty else None,
                    'end_date': raw_data.index.max().isoformat() if not raw_data.empty else None,
                    'total_days': len(raw_data)
                },
                'ohlcv': self._prepare_ohlcv_data(raw_data),
                'technical_indicators': self._prepare_technical_indicators(technical_indicators, raw_data),
                'statistics': self._calculate_price_statistics(raw_data)
            }

            # Save to file
            filename = f"{ticker}_price_data.json"
            filepath = self.output_dir / filename

            with open(filepath, 'w') as f:
                json.dump(price_data, f, indent=2, default=self._json_serializer)

            self.logger.info(f"Price data exported to {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Failed to export price data for {ticker}: {e}")
            raise

    def export_events_data(self, data_bundle: Dict[str, Any], ticker: str) -> str:
        """
        Export significant events with sentiment analysis.

        Args:
            data_bundle: Analysis data bundle
            ticker: Stock ticker symbol

        Returns:
            Path to exported JSON file
        """
        try:
            events_data = data_bundle.get('events', {})
            raw_data = data_bundle.get('raw_data', pd.DataFrame())

            # Prepare events data
            events_export = {
                'ticker': ticker,
                'last_updated': datetime.now().isoformat(),
                'events': self._prepare_events_data(events_data, raw_data),
                'event_summary': self._prepare_events_summary(events_data),
                'sentiment_analysis': self._prepare_sentiment_data(events_data)
            }

            # Save to file
            filename = f"{ticker}_events.json"
            filepath = self.output_dir / filename

            with open(filepath, 'w') as f:
                json.dump(events_export, f, indent=2, default=self._json_serializer)

            self.logger.info(f"Events data exported to {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Failed to export events data for {ticker}: {e}")
            raise

    def export_summary_data(self, data_bundle: Dict[str, Any], ticker: str) -> str:
        """
        Export key metrics and trading signals.

        Args:
            data_bundle: Analysis data bundle
            ticker: Stock ticker symbol

        Returns:
            Path to exported JSON file
        """
        try:
            raw_data = data_bundle.get('raw_data', pd.DataFrame())
            technical_data = data_bundle.get('technical_data', pd.DataFrame())
            technical_indicators = data_bundle.get('technical_indicators', {})
            analysis_summary = data_bundle.get('analysis_summary', {})

            # Use technical_data if available, otherwise fall back to raw_data
            data_for_indicators = technical_data if not technical_data.empty else raw_data

            # Prepare summary data
            summary_data = {
                'ticker': ticker,
                'last_updated': datetime.now().isoformat(),
                'current_price': self._get_current_price(data_for_indicators),
                'key_metrics': self._prepare_key_metrics(data_for_indicators, technical_indicators),
                'trading_signals': self._prepare_trading_signals_enhanced(data_for_indicators),
                'performance_metrics': self._prepare_performance_metrics(data_for_indicators),
                'risk_metrics': self._prepare_risk_metrics(data_for_indicators),
                'technical_indicators': self._prepare_technical_indicators_summary(data_for_indicators),
                'analysis_summary': analysis_summary
            }

            # Save to file
            filename = f"{ticker}_summary.json"
            filepath = self.output_dir / filename

            with open(filepath, 'w') as f:
                json.dump(summary_data, f, indent=2, default=self._json_serializer)

            self.logger.info(f"Summary data exported to {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Failed to export summary data for {ticker}: {e}")
            raise

    def export_company_data(self, data_bundle: Dict[str, Any], ticker: str) -> str:
        """
        Export company information and business data.

        Args:
            data_bundle: Analysis data bundle
            ticker: Stock ticker symbol

        Returns:
            Path to exported JSON file
        """
        try:
            company_info = data_bundle.get('company_info', {})

            # Prepare company data
            company_data = {
                'ticker': ticker,
                'last_updated': datetime.now().isoformat(),
                'company_info': self._prepare_company_info(company_info),
                'business_metrics': self._prepare_business_metrics(company_info),
                'financial_highlights': self._prepare_financial_highlights(company_info)
            }

            # Save to file
            filename = f"{ticker}_company.json"
            filepath = self.output_dir / filename

            with open(filepath, 'w') as f:
                json.dump(company_data, f, indent=2, default=self._json_serializer)

            self.logger.info(f"Company data exported to {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Failed to export company data for {ticker}: {e}")
            raise

    def _prepare_ohlcv_data(self, raw_data: pd.DataFrame) -> List[Dict]:
        """Convert OHLCV data to JSON-friendly format."""
        if raw_data.empty:
            return []

        ohlcv_data = []
        for date, row in raw_data.iterrows():
            ohlcv_data.append({
                'date': date.isoformat(),
                'open': float(row.get('Open', 0)),
                'high': float(row.get('High', 0)),
                'low': float(row.get('Low', 0)),
                'close': float(row.get('Close', 0)),
                'volume': int(row.get('Volume', 0))
            })

        return ohlcv_data

    def _prepare_technical_indicators(self, indicators: Dict, raw_data: pd.DataFrame) -> Dict:
        """Prepare technical indicators for JSON export."""
        prepared_indicators = {}

        # If raw_data has technical indicator columns, extract them
        if not raw_data.empty:
            latest_data = raw_data.iloc[-1]  # Get the most recent data point

            # RSI
            if 'RSI' in raw_data.columns:
                rsi_value = latest_data.get('RSI')
                if not pd.isna(rsi_value):
                    signal = 'overbought' if rsi_value > 70 else 'oversold' if rsi_value < 30 else 'neutral'
                    prepared_indicators['rsi'] = {
                        'value': float(rsi_value),
                        'signal': signal,
                        'period': 14
                    }

            # MACD
            if 'MACD' in raw_data.columns and 'MACD_Signal' in raw_data.columns:
                macd_value = latest_data.get('MACD')
                signal_value = latest_data.get('MACD_Signal')
                if not pd.isna(macd_value) and not pd.isna(signal_value):
                    histogram = macd_value - signal_value
                    signal = 'bullish' if macd_value > signal_value else 'bearish'
                    prepared_indicators['macd'] = {
                        'macd_line': float(macd_value),
                        'signal_line': float(signal_value),
                        'histogram': float(histogram),
                        'signal': signal
                    }

            # Moving Averages
            moving_averages = {}
            if 'SMA_20' in raw_data.columns and not pd.isna(latest_data.get('SMA_20')):
                moving_averages['sma_20'] = float(latest_data['SMA_20'])
            if 'SMA_50' in raw_data.columns and not pd.isna(latest_data.get('SMA_50')):
                moving_averages['sma_50'] = float(latest_data['SMA_50'])
            if 'SMA_200' in raw_data.columns and not pd.isna(latest_data.get('SMA_200')):
                moving_averages['sma_200'] = float(latest_data['SMA_200'])
            if 'EMA_12' in raw_data.columns and not pd.isna(latest_data.get('EMA_12')):
                moving_averages['ema_12'] = float(latest_data['EMA_12'])
            if 'EMA_26' in raw_data.columns and not pd.isna(latest_data.get('EMA_26')):
                moving_averages['ema_26'] = float(latest_data['EMA_26'])

            if moving_averages:
                prepared_indicators['moving_averages'] = moving_averages

            # Bollinger Bands
            if 'BB_Upper' in raw_data.columns and 'BB_Lower' in raw_data.columns:
                bb_upper = latest_data.get('BB_Upper')
                bb_lower = latest_data.get('BB_Lower')
                if not pd.isna(bb_upper) and not pd.isna(bb_lower):
                    current_price = latest_data.get('Close', 0)
                    if current_price > bb_upper:
                        bb_signal = 'overbought'
                    elif current_price < bb_lower:
                        bb_signal = 'oversold'
                    else:
                        bb_signal = 'neutral'

                    prepared_indicators['bollinger_bands'] = {
                        'upper_band': float(bb_upper),
                        'middle_band': float(latest_data.get('SMA_20', (bb_upper + bb_lower) / 2)),
                        'lower_band': float(bb_lower),
                        'signal': bb_signal
                    }

        # Fallback to old structure if available
        if not prepared_indicators:
            # Moving averages (old structure)
            if 'moving_averages' in indicators:
                prepared_indicators['moving_averages'] = {}
                for period, values in indicators['moving_averages'].items():
                    if isinstance(values, pd.Series):
                        prepared_indicators['moving_averages'][f'ma_{period}'] = [
                            {'date': date.isoformat(), 'value': float(value) if not pd.isna(value) else None}
                            for date, value in values.items()
                        ]

            # RSI (old structure)
            if 'rsi' in indicators and isinstance(indicators['rsi'], pd.Series):
                prepared_indicators['rsi'] = [
                    {'date': date.isoformat(), 'value': float(value) if not pd.isna(value) else None}
                    for date, value in indicators['rsi'].items()
                ]

            # MACD (old structure)
            if 'macd' in indicators:
                macd_data = indicators['macd']
                if isinstance(macd_data, dict):
                    prepared_indicators['macd'] = {}
                    for key, series in macd_data.items():
                        if isinstance(series, pd.Series):
                            prepared_indicators['macd'][key] = [
                                {'date': date.isoformat(), 'value': float(value) if not pd.isna(value) else None}
                                for date, value in series.items()
                            ]

            # Bollinger Bands (old structure)
            if 'bollinger_bands' in indicators:
                bb_data = indicators['bollinger_bands']
                if isinstance(bb_data, dict):
                    prepared_indicators['bollinger_bands'] = {}
                    for key, series in bb_data.items():
                        if isinstance(series, pd.Series):
                            prepared_indicators['bollinger_bands'][key] = [
                                {'date': date.isoformat(), 'value': float(value) if not pd.isna(value) else None}
                                for date, value in series.items()
                            ]

        return prepared_indicators

    def _calculate_price_statistics(self, raw_data: pd.DataFrame) -> Dict:
        """Calculate price statistics."""
        if raw_data.empty:
            return {}

        close_prices = raw_data.get('Close', pd.Series())

        return {
            'current_price': float(close_prices.iloc[-1]) if not close_prices.empty else None,
            'price_change_1d': float(close_prices.iloc[-1] - close_prices.iloc[-2]) if len(close_prices) >= 2 else None,
            'price_change_1d_pct': float((close_prices.iloc[-1] / close_prices.iloc[-2] - 1) * 100) if len(close_prices) >= 2 else None,
            'high_52w': float(close_prices.max()) if not close_prices.empty else None,
            'low_52w': float(close_prices.min()) if not close_prices.empty else None,
            'avg_volume': float(raw_data.get('Volume', pd.Series()).mean()) if 'Volume' in raw_data.columns else None,
            'volatility': float(close_prices.pct_change().std() * np.sqrt(252)) if not close_prices.empty else None
        }

    def _prepare_events_data(self, events_data: Dict, raw_data: pd.DataFrame) -> List[Dict]:
        """Prepare events data for JSON export."""
        if not events_data:
            return []

        events_list = []

        # Handle new events format from enhanced analyzer
        if 'events_analysis' in events_data:
            for event in events_data['events_analysis']:
                events_list.append({
                    'date': event.get('date'),
                    'type': event.get('event_type', 'Unknown'),
                    'description': event.get('event_reason', ''),
                    'sentiment': event.get('sentiment', 'Neutral'),
                    'confidence': event.get('confidence_score', 0.0),
                    'impact': event.get('impact_level', 'MEDIUM'),
                    'price_change_pct': event.get('price_change_pct', 0.0),
                    'open_price': event.get('open_price', 0.0),
                    'close_price': event.get('close_price', 0.0),
                    'volume': event.get('volume', 0),
                    'news_count': event.get('news_count', 0),
                    'sentiment_score': event.get('sentiment_score', 50.0),
                    'analysis_method': event.get('analysis_method', 'unknown'),
                    'analysis_phase': event.get('analysis_phase', 'unknown'),
                    'key_phrases': event.get('key_phrases', ''),
                    'sentiment_overall': event.get('sentiment_overall', ''),
                    'sentiment_financial': event.get('sentiment_financial', ''),
                    'sentiment_confidence': event.get('sentiment_confidence', 0.0),
                    'sentiment_relevance': event.get('sentiment_relevance', 0.0)
                })
        else:
            # Handle legacy events format
            for event_type, events in events_data.items():
                if isinstance(events, list):
                    for event in events:
                        if isinstance(event, dict):
                            events_list.append({
                                'type': event_type,
                                'date': event.get('date'),
                                'description': event.get('description', ''),
                                'impact': event.get('impact', 'unknown'),
                                'sentiment': event.get('sentiment', 'neutral'),
                                'confidence': event.get('confidence', 0.5)
                            })

        return sorted(events_list, key=lambda x: x.get('date', ''), reverse=True)

    def _prepare_events_summary(self, events_data: Dict) -> Dict:
        """Prepare events summary statistics."""
        if not events_data:
            return {}

        # Handle new events format from enhanced analyzer
        if 'summary' in events_data:
            return events_data['summary']

        # Handle legacy format
        total_events = sum(len(events) if isinstance(events, list) else 0 for events in events_data.values())

        return {
            'total_events': total_events,
            'event_types': list(events_data.keys()),
            'events_by_type': {k: len(v) if isinstance(v, list) else 0 for k, v in events_data.items()}
        }

    def _prepare_sentiment_data(self, events_data: Dict) -> Dict:
        """Prepare sentiment analysis data."""
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}

        # Handle new events format from enhanced analyzer
        if 'events_analysis' in events_data:
            events_list = events_data['events_analysis']
            for event in events_list:
                sentiment = event.get('sentiment', 'Neutral').lower()
                if sentiment == 'bullish':
                    sentiment_counts['positive'] += 1
                elif sentiment == 'bearish':
                    sentiment_counts['negative'] += 1
                else:
                    sentiment_counts['neutral'] += 1
        else:
            # Handle legacy format
            for events in events_data.values():
                if isinstance(events, list):
                    for event in events:
                        if isinstance(event, dict):
                            sentiment = event.get('sentiment', 'neutral').lower()
                            if sentiment == 'positive' or sentiment == 'bullish':
                                sentiment_counts['positive'] += 1
                            elif sentiment == 'negative' or sentiment == 'bearish':
                                sentiment_counts['negative'] += 1
                            else:
                                sentiment_counts['neutral'] += 1

        total = sum(sentiment_counts.values())
        if total > 0:
            sentiment_percentages = {k: (v / total) * 100 for k, v in sentiment_counts.items()}
        else:
            sentiment_percentages = {k: 0 for k in sentiment_counts.keys()}

        return {
            'sentiment_counts': sentiment_counts,
            'sentiment_percentages': sentiment_percentages
        }

    def _get_current_price(self, raw_data: pd.DataFrame) -> Optional[float]:
        """Get current price from raw data."""
        if raw_data.empty or 'Close' not in raw_data.columns:
            return None
        return float(raw_data['Close'].iloc[-1])

    def _prepare_key_metrics(self, raw_data: pd.DataFrame, indicators: Dict) -> Dict:
        """Prepare key financial metrics."""
        if raw_data.empty:
            return {}

        close_prices = raw_data.get('Close', pd.Series())

        metrics = {}

        # Price metrics
        if not close_prices.empty:
            metrics['current_price'] = float(close_prices.iloc[-1])
            metrics['price_change_1d'] = float(close_prices.iloc[-1] - close_prices.iloc[-2]) if len(close_prices) >= 2 else None
            metrics['price_change_1d_pct'] = float((close_prices.iloc[-1] / close_prices.iloc[-2] - 1) * 100) if len(close_prices) >= 2 else None

        # Technical indicator values
        if 'rsi' in indicators and isinstance(indicators['rsi'], pd.Series):
            metrics['rsi_current'] = float(indicators['rsi'].iloc[-1]) if not indicators['rsi'].empty else None

        if 'moving_averages' in indicators:
            mas = indicators['moving_averages']
            for period, ma_series in mas.items():
                if isinstance(ma_series, pd.Series) and not ma_series.empty:
                    metrics[f'ma_{period}'] = float(ma_series.iloc[-1])

        return metrics

    def _prepare_trading_signals(self, indicators: Dict) -> Dict:
        """Prepare trading signals based on technical indicators."""
        signals = {
            'overall_signal': 'HOLD',
            'signal_strength': 'WEAK',
            'signals': []
        }

        signal_score = 0
        total_signals = 0

        # RSI signals
        if 'rsi' in indicators and isinstance(indicators['rsi'], pd.Series):
            rsi_current = indicators['rsi'].iloc[-1] if not indicators['rsi'].empty else None
            if rsi_current is not None:
                if rsi_current > 70:
                    signals['signals'].append({'type': 'RSI', 'signal': 'SELL', 'reason': 'Overbought'})
                    signal_score -= 1
                elif rsi_current < 30:
                    signals['signals'].append({'type': 'RSI', 'signal': 'BUY', 'reason': 'Oversold'})
                    signal_score += 1
                total_signals += 1

        # Moving average signals
        if 'moving_averages' in indicators:
            mas = indicators['moving_averages']
            if 50 in mas and 200 in mas:
                ma_50 = mas[50].iloc[-1] if not mas[50].empty else None
                ma_200 = mas[200].iloc[-1] if not mas[200].empty else None

                if ma_50 is not None and ma_200 is not None:
                    if ma_50 > ma_200:
                        signals['signals'].append({'type': 'MA_CROSS', 'signal': 'BUY', 'reason': 'Golden Cross'})
                        signal_score += 1
                    else:
                        signals['signals'].append({'type': 'MA_CROSS', 'signal': 'SELL', 'reason': 'Death Cross'})
                        signal_score -= 1
                    total_signals += 1

        # Overall signal
        if total_signals > 0:
            avg_score = signal_score / total_signals
            if avg_score > 0.3:
                signals['overall_signal'] = 'BUY'
                signals['signal_strength'] = 'STRONG' if avg_score > 0.7 else 'MODERATE'
            elif avg_score < -0.3:
                signals['overall_signal'] = 'SELL'
                signals['signal_strength'] = 'STRONG' if avg_score < -0.7 else 'MODERATE'

        return signals

    def _prepare_trading_signals_enhanced(self, data: pd.DataFrame) -> Dict:
        """Prepare enhanced trading signals based on technical indicators from data."""
        signals = {
            'overall_signal': 'HOLD',
            'signal_strength': 'WEAK',
            'signals': []
        }

        if data.empty:
            return signals

        signal_score = 0
        total_signals = 0
        latest_data = data.iloc[-1]

        # RSI signals
        if 'RSI' in data.columns and not pd.isna(latest_data.get('RSI')):
            rsi_current = latest_data['RSI']
            if rsi_current > 70:
                signals['signals'].append({'type': 'RSI', 'signal': 'SELL', 'reason': 'Overbought', 'value': rsi_current})
                signal_score -= 1
            elif rsi_current < 30:
                signals['signals'].append({'type': 'RSI', 'signal': 'BUY', 'reason': 'Oversold', 'value': rsi_current})
                signal_score += 1
            total_signals += 1

        # MACD signals
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            macd_current = latest_data.get('MACD')
            signal_current = latest_data.get('MACD_Signal')
            if not pd.isna(macd_current) and not pd.isna(signal_current):
                if macd_current > signal_current:
                    signals['signals'].append({'type': 'MACD', 'signal': 'BUY', 'reason': 'MACD above signal line', 'value': macd_current})
                    signal_score += 1
                else:
                    signals['signals'].append({'type': 'MACD', 'signal': 'SELL', 'reason': 'MACD below signal line', 'value': macd_current})
                    signal_score -= 1
                total_signals += 1

        # Moving average signals
        if 'SMA_50' in data.columns and 'SMA_200' in data.columns:
            sma_50 = latest_data.get('SMA_50')
            sma_200 = latest_data.get('SMA_200')
            if not pd.isna(sma_50) and not pd.isna(sma_200):
                if sma_50 > sma_200:
                    signals['signals'].append({'type': 'MA_CROSS', 'signal': 'BUY', 'reason': 'Golden Cross (50 > 200)', 'value': sma_50})
                    signal_score += 1
                else:
                    signals['signals'].append({'type': 'MA_CROSS', 'signal': 'SELL', 'reason': 'Death Cross (50 < 200)', 'value': sma_50})
                    signal_score -= 1
                total_signals += 1

        # Overall signal calculation
        if total_signals > 0:
            avg_score = signal_score / total_signals
            if avg_score > 0.3:
                signals['overall_signal'] = 'BUY'
                signals['signal_strength'] = 'STRONG' if avg_score > 0.7 else 'MODERATE'
            elif avg_score < -0.3:
                signals['overall_signal'] = 'SELL'
                signals['signal_strength'] = 'STRONG' if avg_score < -0.7 else 'MODERATE'

        return signals

    def _prepare_technical_indicators_summary(self, data: pd.DataFrame) -> Dict:
        """Prepare technical indicators summary for the frontend."""
        if data.empty:
            return {}

        latest_data = data.iloc[-1]
        indicators = {}

        # RSI
        if 'RSI' in data.columns and not pd.isna(latest_data.get('RSI')):
            rsi_value = float(latest_data['RSI'])
            rsi_signal = 'overbought' if rsi_value > 70 else 'oversold' if rsi_value < 30 else 'neutral'
            indicators['rsi'] = {
                'value': rsi_value,
                'signal': rsi_signal,
                'period': 14
            }

        # MACD
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            macd_value = latest_data.get('MACD')
            signal_value = latest_data.get('MACD_Signal')
            if not pd.isna(macd_value) and not pd.isna(signal_value):
                histogram = float(macd_value - signal_value)
                macd_signal = 'bullish' if macd_value > signal_value else 'bearish'
                indicators['macd'] = {
                    'macd_line': float(macd_value),
                    'signal_line': float(signal_value),
                    'histogram': histogram,
                    'signal': macd_signal
                }

        # Moving Averages
        if 'SMA_20' in data.columns and not pd.isna(latest_data.get('SMA_20')):
            indicators['sma_20'] = float(latest_data['SMA_20'])

        if 'EMA_12' in data.columns and not pd.isna(latest_data.get('EMA_12')):
            indicators['ema_12'] = float(latest_data['EMA_12'])

        # Moving averages structure
        moving_averages = {}
        if 'SMA_20' in data.columns and not pd.isna(latest_data.get('SMA_20')):
            moving_averages['sma_20'] = float(latest_data['SMA_20'])
        if 'SMA_50' in data.columns and not pd.isna(latest_data.get('SMA_50')):
            moving_averages['sma_50'] = float(latest_data['SMA_50'])
        if 'SMA_200' in data.columns and not pd.isna(latest_data.get('SMA_200')):
            moving_averages['sma_200'] = float(latest_data['SMA_200'])

        if moving_averages:
            indicators['moving_averages'] = moving_averages

        return indicators

    def _prepare_performance_metrics(self, raw_data: pd.DataFrame) -> Dict:
        """Prepare performance metrics."""
        if raw_data.empty or 'Close' not in raw_data.columns:
            return {}

        close_prices = raw_data['Close']
        returns = close_prices.pct_change().dropna()

        return {
            'total_return_pct': float((close_prices.iloc[-1] / close_prices.iloc[0] - 1) * 100),
            'annualized_return_pct': float(returns.mean() * 252 * 100),
            'max_drawdown_pct': float((close_prices / close_prices.cummax() - 1).min() * 100),
            'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
        }

    def _prepare_risk_metrics(self, raw_data: pd.DataFrame) -> Dict:
        """Prepare risk metrics."""
        if raw_data.empty or 'Close' not in raw_data.columns:
            return {}

        close_prices = raw_data['Close']
        returns = close_prices.pct_change().dropna()

        return {
            'volatility_annualized_pct': float(returns.std() * np.sqrt(252) * 100),
            'beta': None,  # Would need market data for calculation
            'var_95_pct': float(returns.quantile(0.05) * 100),
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis())
        }

    def _prepare_company_info(self, company_info: Dict) -> Dict:
        """Prepare company information."""
        return {
            'name': company_info.get('longName', ''),
            'sector': company_info.get('sector', ''),
            'industry': company_info.get('industry', ''),
            'country': company_info.get('country', ''),
            'website': company_info.get('website', ''),
            'description': company_info.get('longBusinessSummary', ''),
            'employees': company_info.get('fullTimeEmployees', None),
            'exchange': company_info.get('exchange', ''),
            'currency': company_info.get('currency', ''),
        }

    def _prepare_business_metrics(self, company_info: Dict) -> Dict:
        """Prepare business metrics."""
        return {
            'market_cap': company_info.get('marketCap', None),
            'enterprise_value': company_info.get('enterpriseValue', None),
            'pe_ratio': company_info.get('forwardPE', None),
            'price_to_book': company_info.get('priceToBook', None),
            'dividend_yield': company_info.get('dividendYield', None),
            'profit_margin': company_info.get('profitMargins', None),
            'return_on_equity': company_info.get('returnOnEquity', None),
            'return_on_assets': company_info.get('returnOnAssets', None)
        }

    def _prepare_financial_highlights(self, company_info: Dict) -> Dict:
        """Prepare financial highlights."""
        return {
            'revenue': company_info.get('totalRevenue', None),
            'gross_profit': company_info.get('grossProfits', None),
            'operating_margin': company_info.get('operatingMargins', None),
            'ebitda': company_info.get('ebitda', None),
            'total_cash': company_info.get('totalCash', None),
            'total_debt': company_info.get('totalDebt', None),
            'free_cash_flow': company_info.get('freeCashflow', None),
            'revenue_growth': company_info.get('revenueGrowth', None)
        }

    def _json_serializer(self, obj):
        """JSON serializer for numpy and pandas objects."""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")