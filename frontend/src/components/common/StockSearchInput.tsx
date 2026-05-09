import React, { useState, useRef, useEffect } from 'react';
import { useStockSearch, useDebounce } from '../../utils/hooks';

interface StockSearchInputProps {
  value: string;
  onChange: (value: string) => void;
  onSelect: (ticker: string) => void;
  placeholder?: string;
  className?: string;
  autoFocus?: boolean;
  disabled?: boolean;
}

const StockSearchInput: React.FC<StockSearchInputProps> = ({
  value,
  onChange,
  onSelect,
  placeholder = "e.g., AAPL, MSFT, GOOGL...",
  className = "",
  autoFocus = false,
  disabled = false,
}) => {
  const [isFocused, setIsFocused] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const debouncedValue = useDebounce(value, 300);

  const {
    searchResults,
    search,
    clearResults,
    isLoading: isSearching,
  } = useStockSearch();

  useEffect(() => {
    if (autoFocus && inputRef.current) {
      inputRef.current.focus();
    }
  }, [autoFocus]);

  useEffect(() => {
    if (debouncedValue && debouncedValue.length > 0) {
      search(debouncedValue);
      setShowSuggestions(true);
    } else {
      clearResults();
      setShowSuggestions(false);
    }
  }, [debouncedValue, search, clearResults]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value.toUpperCase();
    onChange(newValue);
  };

  const handleFocus = () => {
    setIsFocused(true);
    if (searchResults.length > 0) {
      setShowSuggestions(true);
    }
  };

  const handleBlur = () => {
    setIsFocused(false);
    // Delay hiding suggestions to allow clicks
    setTimeout(() => setShowSuggestions(false), 200);
  };

  const handleSuggestionClick = (ticker: string) => {
    onSelect(ticker);
    setShowSuggestions(false);
    clearResults();
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && value.trim()) {
      onSelect(value.trim());
      setShowSuggestions(false);
    }
  };

  return (
    <div className={`relative ${className}`}>
      <div className="relative">
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={handleInputChange}
          onFocus={handleFocus}
          onBlur={handleBlur}
          onKeyPress={handleKeyPress}
          placeholder={placeholder}
          disabled={disabled}
          className={`w-full px-6 py-4 bg-slate-900/50 border-2 rounded-xl text-white text-lg font-medium placeholder-slate-400 transition-all duration-300 ${
            isFocused
              ? 'border-emerald-500 ring-4 ring-emerald-500/20'
              : 'border-slate-600 hover:border-slate-500'
          } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
          maxLength={10}
        />

        {/* Search Loading Indicator */}
        {isSearching && (
          <div className="absolute right-4 top-1/2 transform -translate-y-1/2">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-emerald-500"></div>
          </div>
        )}
      </div>

      {/* Search Results Dropdown */}
      {showSuggestions && searchResults.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-slate-800 border border-slate-600 rounded-xl shadow-2xl z-50 max-h-60 overflow-y-auto animate-fade-in">
          {searchResults.map((result) => (
            <button
              key={result.ticker}
              onClick={() => handleSuggestionClick(result.ticker)}
              className="w-full px-4 py-3 text-left hover:bg-slate-700 border-b border-slate-600 last:border-b-0 transition-colors"
            >
              <div className="flex justify-between items-center">
                <div>
                  <span className="font-semibold text-white">{result.ticker}</span>
                  <span className="ml-2 text-slate-400 text-sm">{result.company_name}</span>
                </div>
                <div className="text-right">
                  <div className="text-white font-medium">${result.current_price?.toFixed(2)}</div>
                  <div className={`text-xs ${result.price_change >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
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
  );
};

export default StockSearchInput;