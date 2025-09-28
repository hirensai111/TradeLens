import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useStockAnalysis, useStockSearch, useDebounce } from '../utils/hooks';
import AnalysisProgress from '../components/common/AnalysisProgress';

const POPULAR_STOCKS = [
  { ticker: 'AAPL', name: 'Apple Inc.' },
  { ticker: 'MSFT', name: 'Microsoft Corp.' },
  { ticker: 'GOOGL', name: 'Alphabet Inc.' },
  { ticker: 'AMZN', name: 'Amazon.com Inc.' },
  { ticker: 'TSLA', name: 'Tesla Inc.' },
  { ticker: 'NVDA', name: 'NVIDIA Corp.' },
  { ticker: 'META', name: 'Meta Platforms' },
  { ticker: 'NFLX', name: 'Netflix Inc.' },
];

const LandingPage: React.FC = () => {
  const [ticker, setTicker] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearchFocused, setIsSearchFocused] = useState(false);
  const [validationError, setValidationError] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [showProgress, setShowProgress] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisStep, setAnalysisStep] = useState('');

  const debouncedQuery = useDebounce(searchQuery, 300);
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);

  const {
    data: analysisData,
    isLoading: isAnalyzing,
    error: analysisError,
    analyzeStock
  } = useStockAnalysis();

  const {
    searchResults,
    search,
    clearResults,
    isLoading: isSearching,
  } = useStockSearch();

  // Auto-focus on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Handle search suggestions
  useEffect(() => {
    if (debouncedQuery && debouncedQuery !== ticker) {
      search(debouncedQuery);
      setShowSuggestions(true);
    } else {
      clearResults();
      setShowSuggestions(false);
    }
  }, [debouncedQuery, search, clearResults, ticker]);

  // Navigate to dashboard when analysis completes
  useEffect(() => {
    if (analysisData) {
      console.log('LandingPage: Analysis data received, navigating to dashboard:', {
        ticker,
        hasData: !!analysisData,
        dataKeys: Object.keys(analysisData)
      });
      navigate(`/dashboard/${ticker}`, { state: { analysisData } });
    }
  }, [analysisData, navigate, ticker]);

  const validateTicker = (tickerValue: string): boolean => {
    const cleanTicker = tickerValue.trim().toUpperCase();

    if (!cleanTicker) {
      setValidationError('Please enter a stock ticker symbol');
      return false;
    }

    if (cleanTicker.length > 10) {
      setValidationError('Ticker symbol must be 10 characters or less');
      return false;
    }

    if (!/^[A-Z]+$/.test(cleanTicker)) {
      setValidationError('Ticker symbol must contain only letters');
      return false;
    }

    setValidationError('');
    return true;
  };

  const handleAnalyze = async () => {
    const cleanTicker = ticker.trim().toUpperCase();

    if (!validateTicker(cleanTicker)) {
      return;
    }

    try {
      // Don't show progress bar immediately - wait to see if we get progress updates
      setAnalysisProgress(0);
      setAnalysisStep('Starting analysis...');

      let progressReceived = false;
      let progressTimeout: NodeJS.Timeout;

      // Set a timeout to show progress bar only if analysis takes more than 500ms
      progressTimeout = setTimeout(() => {
        if (!progressReceived) {
          setShowProgress(true);
        }
      }, 500);

      await analyzeStock(cleanTicker, (progress, step) => {
        progressReceived = true;
        clearTimeout(progressTimeout);
        setShowProgress(true);
        setAnalysisProgress(progress);
        setAnalysisStep(step);
      });

      // Clear the timeout in case analysis completed before it fired
      clearTimeout(progressTimeout);

      // If no progress was received, it's cached data - don't show progress bar at all
      if (!progressReceived) {
        setShowProgress(false);
      }
    } catch (error) {
      console.error('Analysis failed:', error);
      setShowProgress(false);
    }
  };

  const handleCancelAnalysis = () => {
    setShowProgress(false);
    setAnalysisProgress(0);
    setAnalysisStep('');
    // Note: In a real implementation, you might want to cancel the actual API request
  };

  const handleAnalysisComplete = () => {
    console.log('LandingPage: handleAnalysisComplete called, current analysisData:', !!analysisData);
    setShowProgress(false);
    setAnalysisProgress(0);
    setAnalysisStep('');
    // Navigation will be handled by the existing useEffect
  };

  const handleTickerChange = (value: string) => {
    const upperValue = value.toUpperCase();
    setTicker(upperValue);
    setSearchQuery(upperValue);
    setValidationError('');
  };

  const handlePopularStockClick = (stockTicker: string) => {
    setTicker(stockTicker);
    setSearchQuery(stockTicker);
    setValidationError('');
    setShowSuggestions(false);
    clearResults();
  };

  const handleSuggestionClick = (suggestion: any) => {
    setTicker(suggestion.ticker);
    setSearchQuery(suggestion.ticker);
    setShowSuggestions(false);
    clearResults();
    setValidationError('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleAnalyze();
    }
  };

  const handleInputFocus = () => {
    setIsSearchFocused(true);
    if (searchResults.length > 0) {
      setShowSuggestions(true);
    }
  };

  const handleInputBlur = () => {
    setIsSearchFocused(false);
    // Delay hiding suggestions to allow clicks
    setTimeout(() => setShowSuggestions(false), 200);
  };

  return (
    <div
      className="min-h-screen"
      style={{ backgroundColor: '#0b0e14' }}
    >
      <div className="flex items-center justify-center min-h-screen px-4 py-12">
        <div className="max-w-2xl w-full space-y-12">
          {/* Header Section */}
          <div className="text-center space-y-6">
            <h1 className="text-5xl md:text-6xl font-bold text-white mb-4">
              <span style={{ color: '#00ff99' }}>Trade</span>Lens
            </h1>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Easy market analysis for smarter decisions
            </p>
          </div>

          {/* Main Search Section */}
          <div
            className="border p-8"
            style={{
              backgroundColor: '#1a202c',
              borderColor: '#2d3748'
            }}
          >
            <div className="space-y-6">
              {/* Search Input */}
              <div className="relative">
                <label htmlFor="ticker-input" className="block text-sm font-medium text-gray-300 mb-3">
                  Enter Stock Symbol
                </label>
                <div className="relative">
                  <input
                    ref={inputRef}
                    id="ticker-input"
                    type="text"
                    value={searchQuery}
                    onChange={(e) => handleTickerChange(e.target.value)}
                    onKeyPress={handleKeyPress}
                    onFocus={handleInputFocus}
                    onBlur={handleInputBlur}
                    placeholder="e.g., AAPL, MSFT, GOOGL..."
                    className={`w-full px-4 py-3 text-white text-lg font-medium placeholder-gray-500 transition-all duration-200 ${
                      isSearchFocused
                        ? 'outline-none ring-2'
                        : validationError
                        ? 'border-red-500'
                        : 'border-gray-600'
                    }`}
                    style={{
                      backgroundColor: '#0b0e14',
                      border: isSearchFocused ? 'none' : '2px solid #2d3748',
                      ...(isSearchFocused && {
                        ringColor: '#00ff99',
                        ringWidth: '2px'
                      })
                    }}
                    maxLength={10}
                  />

                  {/* Search Loading Indicator */}
                  {isSearching && (
                    <div className="absolute right-4 top-1/2 transform -translate-y-1/2">
                      <div
                        className="animate-spin rounded-full h-5 w-5 border-b-2"
                        style={{ borderColor: '#00ff99' }}
                      ></div>
                    </div>
                  )}
                </div>

                {/* Validation Error */}
                {validationError && (
                  <p className="mt-2 text-red-400 text-sm">
                    {validationError}
                  </p>
                )}

                {/* Search Results Dropdown */}
                {showSuggestions && searchResults.length > 0 && (
                  <div
                    className="absolute top-full left-0 right-0 mt-2 border z-50 max-h-60 overflow-y-auto"
                    style={{
                      backgroundColor: '#1a202c',
                      borderColor: '#2d3748'
                    }}
                  >
                    {searchResults.map((result) => (
                      <button
                        key={result.ticker}
                        onClick={() => handleSuggestionClick(result)}
                        className="w-full px-4 py-3 text-left transition-colors border-b border-gray-700 last:border-b-0 hover:bg-gray-600"
                        style={{ backgroundColor: 'transparent' }}
                      >
                        <div className="flex justify-between items-center">
                          <div>
                            <span className="font-semibold text-white">{result.ticker}</span>
                            <span className="ml-2 text-gray-400 text-sm">{result.company_name}</span>
                          </div>
                          <div className="text-right">
                            <div className="text-white font-medium">${result.current_price?.toFixed(2)}</div>
                            <div
                              className="text-xs"
                              style={{
                                color: result.price_change >= 0 ? '#00ff99' : '#ff4757'
                              }}
                            >
                              {result.price_change >= 0 ? '+' : ''}
                              {result.price_change_percent?.toFixed(2)}%
                            </div>
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* Analyze Button */}
              <button
                onClick={handleAnalyze}
                disabled={isAnalyzing || !ticker.trim()}
                className={`w-full font-bold py-3 px-6 text-lg transition-all duration-200 ${
                  isAnalyzing || !ticker.trim()
                    ? 'opacity-50 cursor-not-allowed'
                    : 'hover:opacity-90'
                }`}
                style={{
                  backgroundColor: '#00ff99',
                  color: '#000000'
                }}
              >
                {isAnalyzing ? (
                  <div className="flex items-center justify-center space-x-3">
                    <div
                      className="animate-spin rounded-full h-5 w-5 border-b-2"
                      style={{ borderColor: '#000000' }}
                    ></div>
                    <span>Analyzing {ticker}...</span>
                  </div>
                ) : (
                  'Analyze Stock'
                )}
              </button>

              {/* Analysis Error */}
              {analysisError && (
                <div
                  className="border p-4"
                  style={{
                    backgroundColor: '#2d1b1b',
                    borderColor: '#ff4757'
                  }}
                >
                  <p className="text-red-400">
                    {analysisError.message}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Popular Stocks */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-300 text-center">Popular Stocks</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {POPULAR_STOCKS.map((stock) => (
                <button
                  key={stock.ticker}
                  onClick={() => handlePopularStockClick(stock.ticker)}
                  className="border p-4 transition-all duration-200"
                  style={{
                    backgroundColor: '#1a202c',
                    borderColor: '#2d3748'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = '#00ff99';
                    e.currentTarget.querySelector('.ticker')!.setAttribute('style', 'color: #00ff99');
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = '#2d3748';
                    e.currentTarget.querySelector('.ticker')!.setAttribute('style', 'color: white');
                  }}
                >
                  <div className="text-center">
                    <div className="ticker font-bold text-white transition-colors">
                      {stock.ticker}
                    </div>
                    <div className="text-xs text-gray-400 mt-1 truncate">
                      {stock.name}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Analysis Progress Modal */}
      <AnalysisProgress
        ticker={ticker}
        isVisible={showProgress}
        onCancel={handleCancelAnalysis}
        onComplete={handleAnalysisComplete}
        externalProgress={analysisProgress}
        externalStep={analysisStep}
      />

    </div>
  );
};

export default LandingPage;