import React from 'react';

interface OHLCData {
  open: number;
  high: number;
  low: number;
  close: number;
  change: number;
  changePercent: number;
}

interface OHLCDisplayProps {
  ohlcData: OHLCData | null;
  selectedDateRange: {
    start: string;
    end: string;
  };
}

const OHLCDisplay: React.FC<OHLCDisplayProps> = ({
  ohlcData,
  selectedDateRange,
}) => {
  if (!ohlcData) {
    return null;
  }

  const formatCurrency = (value: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  const formatDate = (dateStr: string): string => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  const isRange = selectedDateRange.start !== selectedDateRange.end;
  const isPositive = ohlcData.change >= 0;

  return (
    <div
      className="border rounded-lg"
      style={{
        backgroundColor: '#1a202c',
        borderColor: '#2d3748'
      }}
    >
      {/* Header */}
      <div
        className="border-b px-4 py-3"
        style={{ borderColor: '#2d3748' }}
      >
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-white">
              {isRange ? 'Period Summary' : 'Daily OHLC'}
            </h3>
            <p className="text-sm text-gray-400">
              {isRange
                ? `${formatDate(selectedDateRange.start)} to ${formatDate(selectedDateRange.end)}`
                : formatDate(selectedDateRange.start)
              }
            </p>
          </div>
          <div className="text-right">
            <div
              className={`text-xl font-bold ${
                isPositive ? 'text-trading-green' : 'text-trading-red'
              }`}
            >
              {isPositive ? '+' : ''}{formatCurrency(ohlcData.change)}
            </div>
            <div
              className={`text-sm ${
                isPositive ? 'text-trading-green' : 'text-trading-red'
              }`}
            >
              ({isPositive ? '+' : ''}{ohlcData.changePercent.toFixed(2)}%)
            </div>
          </div>
        </div>
      </div>

      {/* OHLC Grid */}
      <div className="p-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div
          className="border p-3 rounded"
          style={{
            backgroundColor: '#0b0e14',
            borderColor: '#2d3748'
          }}
        >
          <div className="text-xs text-gray-400 mb-1">Open</div>
          <div className="text-sm font-semibold text-white">
            {formatCurrency(ohlcData.open)}
          </div>
        </div>

        <div
          className="border p-3 rounded"
          style={{
            backgroundColor: '#0b0e14',
            borderColor: '#2d3748'
          }}
        >
          <div className="text-xs text-gray-400 mb-1">High</div>
          <div className="text-sm font-semibold text-trading-green">
            {formatCurrency(ohlcData.high)}
          </div>
        </div>

        <div
          className="border p-3 rounded"
          style={{
            backgroundColor: '#0b0e14',
            borderColor: '#2d3748'
          }}
        >
          <div className="text-xs text-gray-400 mb-1">Low</div>
          <div className="text-sm font-semibold text-trading-red">
            {formatCurrency(ohlcData.low)}
          </div>
        </div>

        <div
          className="border p-3 rounded"
          style={{
            backgroundColor: '#0b0e14',
            borderColor: '#2d3748'
          }}
        >
          <div className="text-xs text-gray-400 mb-1">Close</div>
          <div className="text-sm font-semibold text-white">
            {formatCurrency(ohlcData.close)}
          </div>
        </div>
        </div>

        {/* Additional Info for Range */}
        {isRange && (
          <div className="mt-3 pt-3 border-t" style={{ borderColor: '#2d3748' }}>
            <div className="text-xs text-gray-400">
              Range Span: {Math.abs(new Date(selectedDateRange.end).getTime() - new Date(selectedDateRange.start).getTime()) / (1000 * 60 * 60 * 24)} days
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default OHLCDisplay;