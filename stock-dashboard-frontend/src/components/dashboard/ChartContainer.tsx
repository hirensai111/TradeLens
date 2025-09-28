import React, { useState, useMemo } from 'react';
import { AnalysisResult } from '../../types/backend';
import OHLCDisplay from './OHLCDisplay';
import FinancialChart from './charts/FinancialChart';
import { calculateOHLCForDateRange } from '../../utils/ohlcCalculations';

interface ChartContainerProps {
  analysisData: AnalysisResult;
  selectedDateRange: {
    start: string;
    end: string;
  };
  onDateRangeChange: (range: { start: string; end: string }) => void;
}

const TIME_PERIODS = [
  { label: '1D', value: '1d' },
  { label: '5D', value: '5d' },
  { label: '1M', value: '1m' },
  { label: '3M', value: '3m' },
  { label: '6M', value: '6m' },
  { label: '1Y', value: '1y' },
  { label: 'MAX', value: 'max' },
];

const ChartContainer: React.FC<ChartContainerProps> = ({
  analysisData,
  selectedDateRange,
  onDateRangeChange,
}) => {
  const [selectedPeriod, setSelectedPeriod] = useState('1m');
  const [chartType, setChartType] = useState<'candlestick' | 'line'>('line');
  const [showVolume, setShowVolume] = useState(false);
  const [showMovingAverages, setShowMovingAverages] = useState(false);
  const [showBollingerBands, setShowBollingerBands] = useState(false);
  const [showEvents, setShowEvents] = useState(true);

  // Calculate OHLC data for the selected date range
  const ohlcData = useMemo(() => {
    if (!analysisData.price_history || analysisData.price_history.length === 0) {
      return null;
    }

    return calculateOHLCForDateRange(
      analysisData.price_history,
      selectedDateRange.start,
      selectedDateRange.end
    );
  }, [analysisData.price_history, selectedDateRange.start, selectedDateRange.end]);

  const handlePeriodClick = (period: string) => {
    setSelectedPeriod(period);

    // Use a fixed end date to prevent constant re-renders
    const endDate = new Date();
    endDate.setHours(23, 59, 59, 999); // End of day
    const startDate = new Date();

    switch (period) {
      case '1d':
        startDate.setDate(endDate.getDate() - 1);
        break;
      case '5d':
        startDate.setDate(endDate.getDate() - 5);
        break;
      case '1m':
        startDate.setMonth(endDate.getMonth() - 1);
        break;
      case '3m':
        startDate.setMonth(endDate.getMonth() - 3);
        break;
      case '6m':
        startDate.setMonth(endDate.getMonth() - 6);
        break;
      case '1y':
        startDate.setFullYear(endDate.getFullYear() - 1);
        break;
      case 'max':
        startDate.setFullYear(endDate.getFullYear() - 5);
        break;
    }

    // Use fixed date strings to prevent infinite updates
    const newDateRange = {
      start: startDate.toISOString().split('T')[0],
      end: endDate.toISOString().split('T')[0],
    };

    // Only update if the date range has actually changed
    if (newDateRange.start !== selectedDateRange.start || newDateRange.end !== selectedDateRange.end) {
      onDateRangeChange(newDateRange);
    }
  };

  return (
    <div
      className="border"
      style={{
        backgroundColor: '#1a202c',
        borderColor: '#2d3748'
      }}
    >
      {/* Chart Header */}
      <div
        className="border-b px-6 py-4"
        style={{ borderColor: '#2d3748' }}
      >
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0">
          <h2 className="text-xl font-semibold text-white">Price Chart</h2>

          <div className="flex items-center space-x-4">
            {/* Chart Type Toggle */}
            <div className="flex border">
              <button
                onClick={() => setChartType('line')}
                className={`px-3 py-1 text-sm transition-colors ${
                  chartType === 'line'
                    ? 'text-black'
                    : 'text-gray-400 hover:text-white'
                }`}
                style={{
                  backgroundColor: chartType === 'line' ? '#00ff99' : 'transparent',
                  borderColor: '#2d3748'
                }}
              >
                Simple
              </button>
              <button
                onClick={() => setChartType('candlestick')}
                className={`px-3 py-1 text-sm border-l transition-colors ${
                  chartType === 'candlestick'
                    ? 'text-black'
                    : 'text-gray-400 hover:text-white'
                }`}
                style={{
                  backgroundColor: chartType === 'candlestick' ? '#00ff99' : 'transparent',
                  borderColor: '#2d3748'
                }}
              >
                Detailed
              </button>
            </div>

            {/* Time Period Buttons */}
            <div className="flex border">
              {TIME_PERIODS.map((period, index) => (
                <button
                  key={period.value}
                  onClick={() => handlePeriodClick(period.value)}
                  className={`px-3 py-1 text-sm transition-colors ${
                    index > 0 ? 'border-l' : ''
                  } ${
                    selectedPeriod === period.value
                      ? 'text-black'
                      : 'text-gray-400 hover:text-white'
                  }`}
                  style={{
                    backgroundColor: selectedPeriod === period.value ? '#00ff99' : 'transparent',
                    borderColor: '#2d3748'
                  }}
                >
                  {period.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Chart Content */}
      <div className="p-6">
        {/* OHLC Display */}
        <div className="mb-6">
          <OHLCDisplay
            ohlcData={ohlcData}
            selectedDateRange={selectedDateRange}
          />
        </div>

        <FinancialChart
          analysisData={analysisData}
          selectedDateRange={selectedDateRange}
          chartType={chartType}
          showVolume={showVolume}
          showMovingAverages={showMovingAverages}
          showBollingerBands={showBollingerBands}
          showEvents={showEvents}
        />

        {/* Chart Controls */}
        <div className="mt-4 flex flex-wrap items-center justify-between">
          <div className="flex items-center space-x-4">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                className="form-checkbox text-trading-accent"
                checked={showVolume}
                onChange={(e) => setShowVolume(e.target.checked)}
              />
              <span className="text-sm text-gray-400">Volume</span>
            </label>
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                className="form-checkbox text-trading-accent"
                checked={showMovingAverages}
                onChange={(e) => setShowMovingAverages(e.target.checked)}
              />
              <span className="text-sm text-gray-400">Moving Averages</span>
            </label>
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                className="form-checkbox text-trading-accent"
                checked={showBollingerBands}
                onChange={(e) => setShowBollingerBands(e.target.checked)}
              />
              <span className="text-sm text-gray-400">Bollinger Bands</span>
            </label>
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                className="form-checkbox text-trading-accent"
                checked={showEvents}
                onChange={(e) => setShowEvents(e.target.checked)}
              />
              <span className="text-sm text-gray-400">Events</span>
            </label>
          </div>

          <div className="text-xs text-gray-500">
            Data range: {new Date(selectedDateRange.start).toLocaleDateString()} - {new Date(selectedDateRange.end).toLocaleDateString()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChartContainer;