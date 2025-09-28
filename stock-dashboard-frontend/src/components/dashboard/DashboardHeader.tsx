import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AnalysisResult } from '../../types/backend';
import { formatCurrency, formatPercentage } from '../../utils';
import CustomDatePicker from './CustomDatePicker';

interface DashboardHeaderProps {
  ticker: string;
  analysisData: AnalysisResult;
  selectedDateRange: {
    start: string;
    end: string;
  };
  onDateRangeChange: (range: { start: string; end: string }) => void;
}

const DashboardHeader: React.FC<DashboardHeaderProps> = ({
  ticker,
  analysisData,
  selectedDateRange,
  onDateRangeChange,
}) => {
  const navigate = useNavigate();
  const [showDatePicker, setShowDatePicker] = useState(false);

  const { stock_data, company_info, market_status } = analysisData;
  const priceChange = stock_data.price_change_1d || 0;
  const priceChangePercent = stock_data.price_change_1d_pct || 0;

  // Market status color and text logic
  const getMarketStatusDisplay = () => {
    if (!market_status) {
      return { color: '#6b7280', text: 'Market Status Unknown' };
    }

    switch (market_status.status) {
      case 'OPEN':
        return { color: '#00ff99', text: 'Market Open' };
      case 'CLOSED':
        return { color: '#ef4444', text: 'Market Closed' };
      case 'CLOSED_WEEKEND':
        return { color: '#ef4444', text: 'Market Closed (Weekend)' };
      case 'CLOSED_HOLIDAY':
        return { color: '#ef4444', text: 'Market Closed (Holiday)' };
      case 'PREMARKET':
        return { color: '#f59e0b', text: 'Pre-Market' };
      case 'AFTERHOURS':
        return { color: '#f59e0b', text: 'After Hours' };
      default:
        return { color: '#6b7280', text: market_status.status_message || 'Market Status Unknown' };
    }
  };

  const marketDisplay = getMarketStatusDisplay();

  const handleBackToSearch = () => {
    navigate('/');
  };

  return (
    <div
      className="border-b sticky top-0 z-40"
      style={{
        backgroundColor: '#1a202c',
        borderColor: '#2d3748'
      }}
    >
      <div className="container mx-auto px-4 py-4">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">

          {/* Left Section - Stock Info */}
          <div className="flex items-center space-x-6">
            <button
              onClick={handleBackToSearch}
              className="flex items-center space-x-2 px-3 py-2 border transition-colors hover:bg-gray-700"
              style={{
                borderColor: '#2d3748',
                color: '#00ff99'
              }}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              <span className="text-sm font-medium">Back to Search</span>
            </button>

            <div className="flex items-center space-x-4">
              <div>
                <h1 className="text-2xl lg:text-3xl font-bold text-white">
                  {ticker}
                </h1>
                <p className="text-sm text-gray-400 truncate max-w-xs lg:max-w-none">
                  {company_info.name}
                </p>
              </div>

              <div className="text-right">
                <div className="text-xl lg:text-2xl font-bold text-white">
                  {formatCurrency(stock_data.current_price)}
                </div>
                <div className={`text-sm font-medium ${
                  priceChange >= 0 ? 'text-trading-green' : 'text-trading-red'
                }`}>
                  {priceChange >= 0 ? '+' : ''}{formatCurrency(priceChange)}
                  ({formatPercentage(priceChangePercent)})
                </div>
              </div>
            </div>
          </div>

          {/* Right Section - Controls */}
          <div className="flex items-center space-x-4">

            {/* Market Status Indicator */}
            <div className="hidden lg:flex items-center space-x-2">
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: marketDisplay.color }}
              ></div>
              <span className="text-sm text-gray-400">{marketDisplay.text}</span>
              {market_status && market_status.next_market_open && !market_status.is_open && (
                <span className="text-xs text-gray-500 ml-2">
                  Next: {new Date(market_status.next_market_open).toLocaleDateString('en-US', {
                    weekday: 'short',
                    month: 'short',
                    day: 'numeric',
                    hour: 'numeric',
                    minute: '2-digit'
                  })}
                </span>
              )}
            </div>

            {/* Date Range Picker */}
            <div className="relative">
              <button
                onClick={() => setShowDatePicker(!showDatePicker)}
                className="flex items-center space-x-2 px-4 py-2 border transition-colors"
                style={{
                  backgroundColor: '#0b0e14',
                  borderColor: '#2d3748',
                  color: '#00ff99'
                }}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <span className="text-sm font-medium whitespace-nowrap">
                  {new Date(selectedDateRange.start).toLocaleDateString()} - {new Date(selectedDateRange.end).toLocaleDateString()}
                </span>
              </button>

              {showDatePicker && (
                <CustomDatePicker
                  selectedRange={selectedDateRange}
                  onRangeChange={onDateRangeChange}
                  onClose={() => setShowDatePicker(false)}
                />
              )}
            </div>

            {/* Refresh Data Button */}
            <button
              className="p-2 border transition-colors hover:bg-gray-700"
              style={{
                borderColor: '#2d3748'
              }}
              title="Refresh Data"
            >
              <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardHeader;