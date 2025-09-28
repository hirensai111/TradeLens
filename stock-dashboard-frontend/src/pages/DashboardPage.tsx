import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useStockAnalysis } from '../utils/hooks';
import { AnalysisResult } from '../types/backend';

const DashboardPage: React.FC = () => {
  const { ticker } = useParams<{ ticker: string }>();
  const navigate = useNavigate();
  const { data: analysisData, isLoading, error, analyzeStock } = useStockAnalysis();
  const [searchTicker, setSearchTicker] = useState('');

  useEffect(() => {
    if (ticker) {
      analyzeStock(ticker);
    }
  }, [ticker, analyzeStock]);

  const handleNewSearch = () => {
    if (searchTicker.trim()) {
      navigate(`/dashboard/${searchTicker.trim().toUpperCase()}`);
      setSearchTicker('');
    }
  };

  const handleBackToSearch = () => {
    navigate('/');
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <h2 className="text-xl font-semibold text-gray-900">Analyzing {ticker}...</h2>
          <p className="text-gray-600 mt-2">Fetching real-time data and generating insights</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center px-4">
        <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-8 text-center">
          <div className="text-red-500 text-6xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Analysis Failed</h2>
          <p className="text-gray-600 mb-6">{error.message}</p>
          <div className="space-y-3">
            <button
              onClick={() => ticker && analyzeStock(ticker)}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded"
            >
              Try Again
            </button>
            <button
              onClick={handleBackToSearch}
              className="w-full bg-gray-300 hover:bg-gray-400 text-gray-700 font-semibold py-2 px-4 rounded"
            >
              Back to Search
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!analysisData || !analysisData.stock_data || !analysisData.company_info) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-gray-900">No data available</h2>
          <button
            onClick={handleBackToSearch}
            className="mt-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded"
          >
            Back to Search
          </button>
        </div>
      </div>
    );
  }

  const { stock_data, company_info, technical_indicators, events, sentiment, analysis_summary } = analysisData;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <button
                onClick={handleBackToSearch}
                className="text-gray-600 hover:text-gray-900"
              >
                ← Back
              </button>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  {stock_data.ticker} - {company_info.name}
                </h1>
                <p className="text-sm text-gray-600">{company_info.sector} • {company_info.industry}</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <input
                type="text"
                value={searchTicker}
                onChange={(e) => setSearchTicker(e.target.value.toUpperCase())}
                placeholder="Enter ticker..."
                className="px-3 py-1 border border-gray-300 rounded text-sm"
                onKeyPress={(e) => e.key === 'Enter' && handleNewSearch()}
              />
              <button
                onClick={handleNewSearch}
                disabled={!searchTicker.trim()}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white px-4 py-1 rounded text-sm"
              >
                Analyze
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stock Price Overview */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
          <div className="lg:col-span-2 bg-white rounded-lg shadow-md p-6">
            <div className="flex justify-between items-start">
              <div>
                <h2 className="text-lg font-semibold text-gray-900 mb-2">Current Price</h2>
                <div className="text-3xl font-bold text-gray-900">${stock_data.current_price?.toFixed(2)}</div>
                <div className={`text-lg ${(stock_data.price_change_1d || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {(stock_data.price_change_1d || 0) >= 0 ? '+' : ''}${stock_data.price_change_1d?.toFixed(2)}
                  ({(stock_data.price_change_1d_pct || 0) >= 0 ? '+' : ''}{stock_data.price_change_1d_pct?.toFixed(2)}%)
                </div>
              </div>
              {sentiment?.overall &&
               sentiment.overall.toLowerCase() !== 'neutral' &&
               sentiment.overall.toLowerCase() !== 'n/a' && (
                <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                  sentiment.overall.toLowerCase() === 'bullish' ? 'bg-green-100 text-green-800' :
                  sentiment.overall.toLowerCase() === 'bearish' ? 'bg-red-100 text-red-800' :
                  'bg-blue-100 text-blue-800'
                }`}>
                  {sentiment.overall.toUpperCase()}
                </div>
              )}
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Key Metrics</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-600">Volume</span>
                <span className="font-medium">{stock_data.volume?.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Market Cap</span>
                <span className="font-medium">${stock_data.market_cap?.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">P/E Ratio</span>
                <span className="font-medium">{stock_data.pe_ratio?.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Beta</span>
                <span className="font-medium">{stock_data.beta?.toFixed(2)}</span>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">52-Week Range</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-600">High</span>
                <span className="font-medium text-green-600">N/A</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Low</span>
                <span className="font-medium text-red-600">N/A</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Annualized Return</span>
                <span className="font-medium">
                  {stock_data.annualized_return_pct ? `${stock_data.annualized_return_pct.toFixed(2)}%` : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Dividend Yield</span>
                <span className="font-medium">N/A</span>
              </div>
            </div>
          </div>
        </div>

        {/* Technical Indicators */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Technical Indicators</h3>
            {technical_indicators ? (
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">RSI ({technical_indicators?.rsi?.period ?? 'N/A'})</span>
                  <div className="text-right">
                    <span className="font-medium">{technical_indicators?.rsi?.value?.toFixed(2) ?? 'N/A'}</span>
                    <span className={`ml-2 px-2 py-1 rounded text-xs ${
                      technical_indicators?.rsi?.signal === 'overbought' ? 'bg-red-100 text-red-800' :
                      technical_indicators?.rsi?.signal === 'oversold' ? 'bg-green-100 text-green-800' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {technical_indicators?.rsi?.signal ?? 'N/A'}
                    </span>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">MACD</span>
                  <div className="text-right">
                    <span className="font-medium">{technical_indicators?.macd?.macd_line?.toFixed(4) ?? 'N/A'}</span>
                    <span className={`ml-2 px-2 py-1 rounded text-xs ${
                      technical_indicators?.macd?.signal === 'bullish' ? 'bg-green-100 text-green-800' :
                      technical_indicators?.macd?.signal === 'bearish' ? 'bg-red-100 text-red-800' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {technical_indicators?.macd?.signal ?? 'N/A'}
                    </span>
                  </div>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">SMA 20</span>
                  <span className="font-medium">${technical_indicators?.moving_averages?.sma_20?.toFixed(2) ?? 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">SMA 50</span>
                  <span className="font-medium">${technical_indicators?.moving_averages?.sma_50?.toFixed(2) ?? 'N/A'}</span>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-500 py-8">
                <p>Technical indicators not available</p>
              </div>
            )}
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Analysis Summary</h3>
            <div className="space-y-4">
              {analysis_summary?.overall_sentiment &&
               analysis_summary.overall_sentiment.toLowerCase() !== 'neutral' &&
               analysis_summary.overall_sentiment.toLowerCase() !== 'n/a' && (
                <div>
                  <span className="text-gray-600">Overall Sentiment</span>
                  <div className={`mt-1 px-3 py-1 rounded-full text-sm font-medium inline-block ${
                    analysis_summary.overall_sentiment === 'bullish' ? 'bg-green-100 text-green-800' :
                    analysis_summary.overall_sentiment === 'bearish' ? 'bg-red-100 text-red-800' :
                    'bg-yellow-100 text-yellow-800'
                  }`}>
                    {analysis_summary.overall_sentiment.toUpperCase()}
                  </div>
                </div>
              )}
              {sentiment?.overall &&
               sentiment.overall.toLowerCase() !== 'neutral' &&
               sentiment.overall.toLowerCase() !== 'n/a' && (
                <div>
                  <span className="text-gray-600">Market Sentiment</span>
                  <div className="mt-1 text-lg font-medium">{sentiment.overall}</div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Recent Events */}
        {events && events.length > 0 && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Price Events</h3>
            <div className="space-y-3">
              {events.slice(0, 5).map((event, index) => (
                <div key={index} className="flex justify-between items-center py-2 border-b border-gray-100 last:border-b-0">
                  <div>
                    <span className="font-medium text-gray-900">{event.type}</span>
                    <p className="text-sm text-gray-600">{event.description}</p>
                  </div>
                  <div className="text-right">
                    <div className="font-medium">${event.close_price?.toFixed(2)}</div>
                    <div className={`text-xs px-2 py-1 rounded ${
                      event.impact === 'positive' ? 'bg-green-100 text-green-800' :
                      event.impact === 'negative' ? 'bg-red-100 text-red-800' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {(event.confidence * 100)?.toFixed(1)}% confidence
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DashboardPage;