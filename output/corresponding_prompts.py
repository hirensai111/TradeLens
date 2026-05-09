"""
Prompt templates for ChatGPT API interactions.
Contains various prompts for different types of stock analysis.
"""

from typing import Dict, List, Optional


class PromptTemplates:
    """Collection of prompt templates for stock analysis."""
    
    @staticmethod
    def get_basic_analysis_prompt(ticker: str) -> str:
        """Get basic stock analysis prompt."""
        return f"""You are a financial data analyst and Python developer. Generate complete Python code to analyze the stock {ticker}.

Requirements:
1. Fetch 5 years of historical data using yfinance
2. Calculate technical indicators: SMA (20, 50, 200), RSI (14), MACD (12, 26, 9), Bollinger Bands
3. Create Excel file with multiple sheets (Raw Data, Technical Analysis, Summary)
4. Handle errors and missing data appropriately
5. Save to 'output' directory with timestamp

Provide ONLY Python code, no explanations."""

    @staticmethod
    def get_comprehensive_analysis_prompt(ticker: str, 
                                         indicators: Optional[List[str]] = None,
                                         period_years: int = 5) -> str:
        """Get comprehensive analysis prompt with customizable parameters."""
        
        # Default indicators if none specified
        if not indicators:
            indicators = [
                "SMA (5, 10, 20, 50, 200 days)",
                "EMA (12, 26 days)",
                "RSI (14-day period)",
                "MACD with signal line (12, 26, 9)",
                "Bollinger Bands (20-day, 2 std)",
                "Volume SMA (20-day)",
                "ATR (14-day)",
                "Stochastic Oscillator (14, 3, 3)"
            ]
        
        indicators_text = "\n   - ".join(indicators)
        
        return f"""You are a financial data analyst and Python developer. Generate complete Python code to analyze the stock {ticker}.

REQUIREMENTS:

1. DATA COLLECTION:
   - Fetch {period_years} years of historical data using yfinance
   - Validate data quality and completeness
   - Handle missing values appropriately
   - Include dividend and split adjustments

2. TECHNICAL INDICATORS (calculate all):
   - {indicators_text}

3. PERFORMANCE METRICS:
   - Daily, monthly, and annual returns
   - Volatility (20-day and annual)
   - Sharpe ratio (assuming 2% risk-free rate)
   - Maximum drawdown and recovery period
   - Beta vs S&P 500
   - 52-week high/low
   - Current price vs moving averages

4. EXCEL OUTPUT STRUCTURE:
   Sheet 1 "Raw Data":
   - Date, Open, High, Low, Close, Volume, Adj Close
   - Data quality indicators
   
   Sheet 2 "Technical Analysis":
   - All calculated technical indicators
   - Trading signals (Buy/Hold/Sell)
   
   Sheet 3 "Performance Metrics":
   - Returns analysis
   - Risk metrics
   - Comparison to benchmarks
   
   Sheet 4 "Summary Dashboard":
   - Current price and change
   - Key technical levels
   - Performance summary
   - Latest indicator values
   
   Sheet 5 "Charts Data":
   - Prepared data for charting

5. EXCEL FORMATTING:
   - Professional styling with company colors
   - Conditional formatting for positive/negative values
   - Number formatting (currency, percentages)
   - Frozen panes for headers
   - Column width adjustments
   - Data validation where appropriate

6. ERROR HANDLING:
   - Try-except blocks for API calls
   - Validation for ticker symbol
   - Check for sufficient data
   - Graceful handling of calculation errors

7. OUTPUT:
   - Save to 'output' directory
   - Filename: {ticker}_analysis_YYYYMMDD_HHMMSS.xlsx
   - Include metadata (analysis date, data period)

Use only these libraries: yfinance, pandas, numpy, openpyxl, datetime
Ensure the code is complete, production-ready, and well-structured.

Provide ONLY Python code, no explanations or comments outside the code."""

    @staticmethod
    def get_comparison_analysis_prompt(tickers: List[str]) -> str:
        """Get prompt for comparing multiple stocks."""
        tickers_str = ", ".join(tickers)
        
        return f"""Generate Python code to compare these stocks: {tickers_str}

Requirements:
1. Fetch 5 years of data for each stock
2. Calculate key metrics for comparison:
   - Returns (1M, 3M, 6M, 1Y, 3Y, 5Y)
   - Volatility
   - Sharpe ratio
   - Maximum drawdown
   - Correlation matrix
   
3. Create Excel with:
   - Individual stock sheets
   - Comparison summary sheet
   - Correlation analysis
   - Relative performance charts
   
4. Rank stocks by various metrics

Provide ONLY Python code."""

    @staticmethod
    def get_sector_analysis_prompt(ticker: str) -> str:
        """Get prompt for sector-relative analysis."""
        return f"""Generate Python code to analyze {ticker} relative to its sector.

Requirements:
1. Fetch stock data and identify its sector
2. Get sector ETF data for comparison
3. Calculate relative performance metrics
4. Show outperformance/underperformance periods
5. Create Excel with sector comparison analysis

Provide ONLY Python code."""

    @staticmethod
    def get_options_analysis_prompt(ticker: str) -> str:
        """Get prompt for options analysis."""
        return f"""Generate Python code to analyze options data for {ticker}.

Requirements:
1. Fetch current stock price and historical volatility
2. Get options chain data
3. Calculate implied volatility
4. Identify unusual options activity
5. Create Excel with options analysis

Provide ONLY Python code."""

    @staticmethod
    def get_fundamental_analysis_prompt(ticker: str) -> str:
        """Get prompt for fundamental analysis."""
        return f"""Generate Python code for fundamental analysis of {ticker}.

Requirements:
1. Fetch company info and financials
2. Calculate key ratios (P/E, P/B, ROE, etc.)
3. Get revenue and earnings data
4. Compare to industry averages
5. Create Excel with fundamental analysis

Provide ONLY Python code."""

    @staticmethod
    def get_custom_prompt(ticker: str, requirements: str) -> str:
        """Get custom analysis prompt."""
        return f"""Generate Python code to analyze {ticker} with these specific requirements:

{requirements}

Ensure the code:
- Uses yfinance for data
- Creates professional Excel output
- Includes error handling
- Saves to 'output' directory

Provide ONLY Python code."""


class PromptEnhancer:
    """Enhance prompts with additional context and examples."""
    
    @staticmethod
    def add_code_quality_requirements(prompt: str) -> str:
        """Add code quality requirements to prompt."""
        quality_requirements = """

CODE QUALITY REQUIREMENTS:
- Use descriptive variable names
- Add helpful comments
- Structure code with functions
- Include docstrings
- Follow PEP 8 style guide
- Add type hints where appropriate"""
        
        return prompt + quality_requirements
    
    @staticmethod
    def add_data_validation_requirements(prompt: str) -> str:
        """Add data validation requirements to prompt."""
        validation_requirements = """

DATA VALIDATION:
- Check for NaN values
- Verify date continuity
- Validate price data (no negatives)
- Check volume data integrity
- Ensure sufficient data points for calculations"""
        
        return prompt + validation_requirements
    
    @staticmethod
    def add_performance_requirements(prompt: str) -> str:
        """Add performance optimization requirements."""
        performance_requirements = """

PERFORMANCE OPTIMIZATION:
- Use vectorized operations
- Minimize API calls
- Efficient memory usage
- Cache calculated values
- Use appropriate data types"""
        
        return prompt + performance_requirements


# Utility functions for prompt management
def get_prompt_for_analysis_type(analysis_type: str, ticker: str, **kwargs) -> str:
    """
    Get appropriate prompt based on analysis type.
    
    Args:
        analysis_type: Type of analysis (basic, comprehensive, comparison, etc.)
        ticker: Stock ticker symbol
        **kwargs: Additional parameters for specific prompts
        
    Returns:
        Formatted prompt string
    """
    templates = PromptTemplates()
    
    prompt_map = {
        'basic': templates.get_basic_analysis_prompt,
        'comprehensive': templates.get_comprehensive_analysis_prompt,
        'sector': templates.get_sector_analysis_prompt,
        'options': templates.get_options_analysis_prompt,
        'fundamental': templates.get_fundamental_analysis_prompt,
    }
    
    if analysis_type in prompt_map:
        if analysis_type == 'comprehensive':
            return prompt_map[analysis_type](ticker, **kwargs)
        else:
            return prompt_map[analysis_type](ticker)
    else:
        # Default to basic analysis
        return templates.get_basic_analysis_prompt(ticker)


def enhance_prompt(prompt: str, 
                  add_quality: bool = True,
                  add_validation: bool = True,
                  add_performance: bool = False) -> str:
    """
    Enhance prompt with additional requirements.
    
    Args:
        prompt: Base prompt
        add_quality: Add code quality requirements
        add_validation: Add data validation requirements
        add_performance: Add performance requirements
        
    Returns:
        Enhanced prompt
    """
    enhancer = PromptEnhancer()
    
    if add_quality:
        prompt = enhancer.add_code_quality_requirements(prompt)
    
    if add_validation:
        prompt = enhancer.add_data_validation_requirements(prompt)
    
    if add_performance:
        prompt = enhancer.add_performance_requirements(prompt)
    
    return prompt


# Example prompt configurations for different use cases
PROMPT_CONFIGS = {
    'quick_analysis': {
        'type': 'basic',
        'enhance_quality': False,
        'enhance_validation': True,
        'enhance_performance': False
    },
    'detailed_analysis': {
        'type': 'comprehensive',
        'enhance_quality': True,
        'enhance_validation': True,
        'enhance_performance': True
    },
    'portfolio_analysis': {
        'type': 'comparison',
        'enhance_quality': True,
        'enhance_validation': True,
        'enhance_performance': True
    }
}


if __name__ == "__main__":
    # Test prompt generation
    print("Testing Prompt Templates")
    print("=" * 50)
    
    # Test basic prompt
    basic_prompt = PromptTemplates.get_basic_analysis_prompt("AAPL")
    print("Basic Analysis Prompt:")
    print(basic_prompt[:200] + "...")
    print()
    
    # Test comprehensive prompt
    comp_prompt = PromptTemplates.get_comprehensive_analysis_prompt(
        "MSFT", 
        period_years=3
    )
    print("Comprehensive Analysis Prompt:")
    print(comp_prompt[:200] + "...")
    print()
    
    # Test enhanced prompt
    enhanced = enhance_prompt(
        PromptTemplates.get_basic_analysis_prompt("GOOGL"),
        add_quality=True,
        add_validation=True
    )
    print("Enhanced Prompt:")
    print(enhanced[:200] + "...")