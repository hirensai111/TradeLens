"""
Main entry point for Stock Analyzer.
Provides CLI interface for stock analysis and Excel report generation.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List

from config import config
from utils import get_logger, print_banner, ProgressTracker
from validators import validate_ticker
from stock_analyzer_handler import StockAnalyzerHandler


class StockAnalyzer:
    """Main application class for stock analysis."""

    def __init__(self):
        self.logger = get_logger("StockAnalyzer")
        self.stock_handler = StockAnalyzerHandler()

        # Ensure output directory exists
        config.OUTPUT_DIR.mkdir(exist_ok=True)

    def analyze_stock(self, ticker: str) -> bool:
        try:
            ticker = validate_ticker(ticker)
            self.logger.info(f"Starting analysis for {ticker}")

            print(f"\nAnalyzing {ticker}...")
            result = self.stock_handler.analyze_stock(ticker)

            if result['success']:
                print(f"\n✅ Success! Report saved to: {result['output_file']}")
                if sys.platform == "win32":
                    os.startfile(os.path.dirname(result['output_file']))
                elif sys.platform == "darwin":
                    os.system(f"open {os.path.dirname(result['output_file'])}")
                return True
            else:
                self.logger.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
                print(f"\nAnalysis failed: {result.get('error', 'Unknown error')}")
                return False

        except Exception as e:
            self.logger.error(f"Analysis failed for {ticker}: {e}")
            print(f"\nError analyzing {ticker}: {e}")
            return False

    def analyze_multiple_stocks(self, tickers: List[str]) -> None:
        print(f"\nAnalyzing {len(tickers)} stocks...")

        successful = 0
        failed = []

        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] Processing {ticker}")
            if self.analyze_stock(ticker):
                successful += 1
            else:
                failed.append(ticker)

        print("\n" + "=" * 50)
        print("ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total stocks: {len(tickers)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(failed)}")

        if failed:
            print(f"\nFailed stocks: {', '.join(failed)}")

    def run_interactive_mode(self):
        print_banner("Stock Analyzer - Interactive Mode")
        print("\nWelcome to Stock Analyzer!")
        print("Type 'help' for commands or 'quit' to exit.\n")

        while True:
            try:
                command = input("stock-analyzer> ").strip().lower()

                if command in ['quit', 'exit']:
                    print("Goodbye!")
                    break

                elif command == 'help':
                    self.print_help()

                elif command.startswith('analyze '):
                    ticker = command.split()[1].upper()
                    self.analyze_stock(ticker)

                elif command.startswith('batch '):
                    filename = command.split()[1]
                    self.process_batch_file(filename)

                elif command == 'settings':
                    self.show_settings()

                elif command == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')

                elif command == '':
                    continue

                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\nUse 'quit' to exit.")
            except Exception as e:
                print(f"Error: {e}")

    def print_help(self):
        help_text = """
Available Commands:
==================
analyze <TICKER>    - Analyze a single stock (e.g., analyze AAPL)
batch <filename>    - Analyze multiple stocks from a file
settings            - Show current settings
clear               - Clear the screen
help                - Show this help message
quit/exit           - Exit the program

Examples:
=========
analyze AAPL        - Analyze Apple stock
batch stocks.txt    - Analyze all stocks listed in stocks.txt

Notes:
======
- Stock tickers should be in uppercase (e.g., AAPL, MSFT, GOOGL)
- Batch files should have one ticker per line
- Reports are saved in the 'output' directory
"""
        print(help_text)

    def show_settings(self):
        print("\nCurrent Settings:")
        print("=" * 40)
        print(f"Output Directory: {config.OUTPUT_DIR}")
        print(f"Data Period: {config.DATA_PERIOD_YEARS} years")
        print(f"Cache Enabled: {'Yes' if config.CACHE_ENABLED else 'No'}")
        print("=" * 40)

    def process_batch_file(self, filename: str):
        try:
            filepath = Path(filename)
            if not filepath.exists():
                print(f"Error: File '{filename}' not found.")
                return

            with open(filepath, 'r') as f:
                tickers = [line.strip().upper() for line in f if line.strip()]

            if not tickers:
                print("Error: No tickers found in file.")
                return

            print(f"Found {len(tickers)} tickers in {filename}")
            self.analyze_multiple_stocks(tickers)

        except Exception as e:
            print(f"Error processing batch file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Stock Analyzer - Generate technical reports for any stock.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --ticker AAPL           # Analyze Apple stock
  %(prog)s --batch stocks.txt      # Analyze multiple stocks
  %(prog)s --interactive           # Run in interactive mode
        """
    )

    parser.add_argument(
        '--ticker', '-t',
        type=str,
        help='Stock ticker symbol to analyze (e.g., AAPL)'
    )

    parser.add_argument(
        '--batch', '-b',
        type=str,
        help='Path to file containing list of tickers (one per line)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory for reports (default: output)'
    )

    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )

    parser.add_argument(
        '--version', '-v',
        action='version',
        version='%(prog)s 1.0.0'
    )

    args = parser.parse_args()

    print_banner("Stock Analyzer")

    analyzer = StockAnalyzer()

    # Set output directory
    if args.output != 'output':
        config.OUTPUT_DIR = Path(args.output)
        config.OUTPUT_DIR.mkdir(exist_ok=True)

    # Run based on mode
    if args.interactive:
        analyzer.run_interactive_mode()

    elif args.ticker:
        analyzer.analyze_stock(args.ticker)

    elif args.batch:
        try:
            with open(args.batch, 'r') as f:
                tickers = [line.strip().upper() for line in f if line.strip()]

            if tickers:
                analyzer.analyze_multiple_stocks(tickers)
            else:
                print("Error: No tickers found in batch file.")
                sys.exit(1)

        except FileNotFoundError:
            print(f"Error: Batch file '{args.batch}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading batch file: {e}")
            sys.exit(1)

    else:
        parser.print_help()
        print("\nTip: Try running with --interactive for an interactive session!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
