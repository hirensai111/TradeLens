import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useStockAnalysis, useStockSearch, useDebounce } from '../utils/hooks';
import AnalysisProgress from '../components/common/AnalysisProgress';
import ModeSelector from '../components/common/ModeSelector';
import AnimatedBackground from '../components/common/AnimatedBackground';

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
    <div className="min-h-screen relative overflow-hidden">
      <AnimatedBackground />

      {/* Header */}
      <div className="py-8 px-4 text-center animate-[fadeInUp_0.6s_ease-out]">
        <h1 className="text-5xl md:text-6xl font-bold text-white mb-4">
          <span
            style={{
              background: 'linear-gradient(135deg, #00ff99 0%, #00e5ff 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text'
            }}
          >
            Trade
          </span>
          <span>Lens</span>
        </h1>
        <p className="text-xl text-gray-400 mb-8">
          Easy market analysis for smarter decisions
        </p>
        <ModeSelector />
      </div>

      <div className="flex items-center justify-center px-4 py-12">
        <div className="max-w-2xl w-full space-y-12">

          {/* Main Search Section */}
          <div
            className="p-8 rounded-3xl animate-[fadeInUp_0.8s_ease-out_0.2s] animate-fill-both"
            style={{
              background: 'rgba(26, 32, 44, 0.6)',
              backdropFilter: 'blur(20px)',
              WebkitBackdropFilter: 'blur(20px)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1)'
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
                    className="w-full px-5 py-4 text-white text-lg font-medium placeholder-gray-500 transition-all duration-300 outline-none rounded-2xl"
                    style={{
                      background: isSearchFocused
                        ? 'rgba(11, 14, 20, 0.8)'
                        : 'rgba(11, 14, 20, 0.6)',
                      backdropFilter: 'blur(10px)',
                      WebkitBackdropFilter: 'blur(10px)',
                      border: isSearchFocused
                        ? '2px solid rgba(0, 255, 153, 0.5)'
                        : '1px solid rgba(255, 255, 255, 0.1)',
                      boxShadow: isSearchFocused
                        ? '0 8px 32px rgba(0, 255, 153, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1)'
                        : '0 4px 16px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05)'
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
                    className="absolute top-full left-0 right-0 mt-2 z-50 max-h-60 overflow-y-auto rounded-2xl animate-[fadeIn_0.3s_ease-out]"
                    style={{
                      background: 'rgba(26, 32, 44, 0.95)',
                      backdropFilter: 'blur(20px)',
                      WebkitBackdropFilter: 'blur(20px)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)'
                    }}
                  >
                    {searchResults.map((result) => (
                      <button
                        key={result.ticker}
                        onClick={() => handleSuggestionClick(result)}
                        className="w-full px-4 py-3 text-left transition-all duration-200 border-b last:border-b-0 hover:bg-white/10"
                        style={{
                          backgroundColor: 'transparent',
                          borderColor: 'rgba(255, 255, 255, 0.05)'
                        }}
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
                className={`w-full font-bold py-4 px-6 text-lg transition-all duration-300 rounded-2xl transform active:scale-95 ${
                  isAnalyzing || !ticker.trim()
                    ? 'opacity-50 cursor-not-allowed'
                    : 'hover:scale-[1.02] hover:shadow-2xl'
                }`}
                style={{
                  background: 'linear-gradient(135deg, #00ff99 0%, #00e5ff 100%)',
                  color: '#000000',
                  boxShadow: '0 8px 32px rgba(0, 255, 153, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.3)'
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
          <div className="space-y-4 animate-[fadeInUp_0.8s_ease-out_0.4s] animate-fill-both">
            <h3 className="text-lg font-semibold text-gray-300 text-center">Popular Stocks</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {POPULAR_STOCKS.map((stock, index) => (
                <button
                  key={stock.ticker}
                  onClick={() => handlePopularStockClick(stock.ticker)}
                  className="p-4 transition-all duration-300 rounded-2xl transform hover:scale-105 hover:-translate-y-1 active:scale-95"
                  style={{
                    background: 'rgba(26, 32, 44, 0.6)',
                    backdropFilter: 'blur(20px)',
                    WebkitBackdropFilter: 'blur(20px)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    boxShadow: '0 4px 16px rgba(0, 0, 0, 0.2)',
                    animationDelay: `${index * 0.05}s`
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(0, 255, 153, 0.5)';
                    e.currentTarget.style.boxShadow = '0 8px 32px rgba(0, 255, 153, 0.3)';
                    e.currentTarget.querySelector('.ticker')!.setAttribute('style', 'background: linear-gradient(135deg, #00ff99 0%, #00e5ff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;');
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                    e.currentTarget.style.boxShadow = '0 4px 16px rgba(0, 0, 0, 0.2)';
                    e.currentTarget.querySelector('.ticker')!.setAttribute('style', 'color: white');
                  }}
                >
                  <div className="text-center">
                    <div className="ticker font-bold text-white transition-all duration-300">
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