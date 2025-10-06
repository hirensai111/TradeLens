"""
Stock Analyzer Handler
Coordinates stock data processing and Excel report generation
"""

from prediction_engine.technical_analysis.data_processor import StockDataProcessor
from output.excel.excel_generator import ExcelReportGenerator
from output.json.json_exporter import VisualizationDataExporter
from core.utils.utils import get_logger
from core.config.config import config
import os
from datetime import datetime, timedelta
from typing import Dict, Optional


class StockAnalyzerHandler:
    """Handles the full stock analysis pipeline using internal processors."""

    def __init__(self):
        self.logger = get_logger("StockAnalyzerHandler")
        self.data_processor = StockDataProcessor()
        self.excel_generator = ExcelReportGenerator()
        self.json_exporter = VisualizationDataExporter()
        self.logger.info("Stock Analyzer Handler initialized")

    def analyze_stock(self, ticker: str, force_refresh: bool = False) -> Dict:
        """
        Run stock analysis pipeline for the given ticker.

        Args:
            ticker: Stock ticker symbol
            force_refresh: Force refresh of data even if cache is fresh

        Returns:
            Dict with success status, output file path, and JSON data paths
        """
        analysis_start_time = datetime.now()

        try:
            self.logger.info(f"Starting analysis for {ticker} (force_refresh={force_refresh})")

            # Check data freshness first
            freshness_info = self.check_data_freshness(ticker)
            data_status = 'fresh'

            # Determine if we need to process data
            needs_processing = (
                force_refresh or
                freshness_info.get('needs_update', True) or
                not freshness_info.get('data_exists', False)
            )

            if needs_processing:
                # Step 1: Process stock data
                data_bundle = self.data_processor.process_stock(ticker)
                data_status = 'updated'
            else:
                # Use cached data
                data_bundle = self.data_processor.get_cached_data(ticker)
                data_status = 'current'

            # Step 2: Generate Excel report (always generate to ensure consistency)
            output_path = self.excel_generator.generate_report(data_bundle)

            # Step 2.5: Extract analyzed events from Excel generator for JSON export
            analyzed_events = self.excel_generator.get_analyzed_events_for_json()
            if analyzed_events:
                data_bundle['events'] = analyzed_events
                self.logger.info(f"Captured {len(analyzed_events.get('events_analysis', []))} analyzed events for JSON export")

            # Step 3: Export JSON data for visualization
            viz_data_paths = {}
            if config.JSON_EXPORT_ENABLED:
                try:
                    viz_data_paths = self.json_exporter.export_all_data(data_bundle, ticker)
                    self.logger.info(f"JSON export completed for {ticker}")
                except Exception as e:
                    self.logger.warning(f"JSON export failed for {ticker}: {e}")

            analysis_time = (datetime.now() - analysis_start_time).total_seconds()

            self.logger.info(f"Analysis completed for {ticker} in {analysis_time:.2f}s")
            return {
                'success': True,
                'ticker': ticker,
                'output_file': output_path,
                'viz_data': viz_data_paths,
                'data_status': data_status,
                'last_updated': analysis_start_time.isoformat(),
                'analysis_time_seconds': round(analysis_time, 2)
            }

        except Exception as e:
            analysis_time = (datetime.now() - analysis_start_time).total_seconds()
            self.logger.error(f"Analysis failed for {ticker}: {e}")
            return {
                'success': False,
                'ticker': ticker,
                'error': str(e),
                'analysis_time_seconds': round(analysis_time, 2)
            }

    def check_data_freshness(self, ticker: str) -> Dict:
        """
        Check the freshness of cached data for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with freshness information
        """
        try:
            return self.data_processor.check_data_freshness(ticker)
        except Exception as e:
            self.logger.error(f"Error checking data freshness for {ticker}: {e}")
            return {
                'status': 'unknown',
                'needs_update': True,
                'data_exists': False,
                'error': str(e)
            }


# Optional test entry point
if __name__ == "__main__":
    handler = StockAnalyzerHandler()
    ticker = "AAPL"
    result = handler.analyze_stock(ticker)

    if result['success']:
        print(f"[OK] Analysis successful: {result['output_file']}")
    else:
        print(f"✗ Analysis failed: {result['error']}")
