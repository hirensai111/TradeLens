import React from 'react';
import { StockData, TechnicalIndicators } from '../../types/backend';
import { formatCurrency, formatNumber } from '../../utils';

interface MarketDataGridProps {
  stockData: StockData;
  technicalIndicators: TechnicalIndicators;
}

const MarketDataGrid: React.FC<MarketDataGridProps> = ({
  stockData,
  technicalIndicators,
}) => {
  const marketDataItems = [
    {
      label: 'Current Price',
      value: formatCurrency(stockData.current_price),
      isPrice: true,
    },
    {
      label: '1D Change',
      value: stockData.price_change_1d ? formatCurrency(stockData.price_change_1d) : 'N/A',
      change: stockData.price_change_1d,
      isPrice: true,
    },
    {
      label: '1D Change %',
      value: stockData.price_change_1d_pct ? `${stockData.price_change_1d_pct.toFixed(2)}%` : 'N/A',
      change: stockData.price_change_1d_pct,
      isPercentage: true,
    },
    {
      label: 'Volume',
      value: stockData.volume ? formatNumber(stockData.volume) : 'N/A',
      subValue: 'shares',
      isVolume: true,
    },
    {
      label: 'Market Cap',
      value: stockData.market_cap ? `$${(stockData.market_cap / 1000000000).toFixed(2)}B` : 'N/A',
      subValue: stockData.market_cap ? 'USD' : '',
    },
    {
      label: 'P/E Ratio',
      value: stockData.pe_ratio?.toFixed(2) || 'N/A',
      subValue: stockData.pe_ratio ? 'TTM' : '',
    },
    {
      label: 'Sharpe Ratio',
      value: stockData.sharpe_ratio?.toFixed(2) || 'N/A',
      subValue: stockData.sharpe_ratio ? 'Risk-Adjusted' : '',
    },
    {
      label: 'Annualized Return',
      value: stockData.annualized_return_pct ? `${stockData.annualized_return_pct.toFixed(2)}%` : 'N/A',
      subValue: stockData.annualized_return_pct ? 'Return' : '',
      change: stockData.annualized_return_pct,
      isPercentage: true,
    },
  ];

  const technicalItems = [
    {
      label: 'RSI (14)',
      value: technicalIndicators.rsi?.value?.toFixed(2) || 'N/A',
      indicator: technicalIndicators.rsi?.signal || '',
      color: technicalIndicators.rsi?.signal === 'overbought' ? '#ff4757' :
            technicalIndicators.rsi?.signal === 'oversold' ? '#00ff99' : '#ffffff',
    },
    {
      label: 'MACD',
      value: technicalIndicators.macd?.macd_line?.toFixed(4) || 'N/A',
      indicator: technicalIndicators.macd?.signal || '',
      color: technicalIndicators.macd?.signal === 'bullish' ? '#00ff99' :
            technicalIndicators.macd?.signal === 'bearish' ? '#ff4757' : '#ffffff',
    },
    {
      label: 'SMA (20)',
      value: technicalIndicators.sma_20?.toFixed(2) || 'N/A',
      change: technicalIndicators.sma_20 ? stockData.current_price - technicalIndicators.sma_20 : null,
      isPrice: true,
    },
    {
      label: 'EMA (12)',
      value: technicalIndicators.ema_12?.toFixed(2) || 'N/A',
      change: technicalIndicators.ema_12 ? stockData.current_price - technicalIndicators.ema_12 : null,
      isPrice: true,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Market Data Section */}
      <div
        className="border"
        style={{
          backgroundColor: '#1a202c',
          borderColor: '#2d3748'
        }}
      >
        <div
          className="border-b px-6 py-4"
          style={{ borderColor: '#2d3748' }}
        >
          <h2 className="text-xl font-semibold text-white">Market Data</h2>
        </div>

        <div className="p-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {marketDataItems.map((item, index) => (
              <div
                key={index}
                className="border p-4"
                style={{
                  backgroundColor: '#0b0e14',
                  borderColor: '#2d3748'
                }}
              >
                <div className="text-sm text-gray-400 mb-1">{item.label}</div>
                <div className="text-lg font-semibold text-white">{item.value}</div>

                {item.change !== undefined && (item.isPrice || item.isPercentage) && (
                  <div className={`text-sm ${
                    item.change > 0 ? 'text-trading-green' :
                    item.change < 0 ? 'text-trading-red' : 'text-gray-400'
                  }`}>
                    {item.change > 0 ? '+' : ''}
                    {item.isPercentage ?
                      `${item.change.toFixed(2)}%` :
                      formatCurrency(item.change)}
                  </div>
                )}

                {item.subValue && (
                  <div className="text-xs text-gray-500 mt-1">{item.subValue}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Technical Indicators Section */}
      <div
        className="border"
        style={{
          backgroundColor: '#1a202c',
          borderColor: '#2d3748'
        }}
      >
        <div
          className="border-b px-6 py-4"
          style={{ borderColor: '#2d3748' }}
        >
          <h2 className="text-xl font-semibold text-white">Technical Indicators</h2>
        </div>

        <div className="p-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {technicalItems.map((item, index) => (
              <div
                key={index}
                className="border p-4"
                style={{
                  backgroundColor: '#0b0e14',
                  borderColor: '#2d3748'
                }}
              >
                <div className="text-sm text-gray-400 mb-1">{item.label}</div>
                <div className="text-lg font-semibold text-white">{item.value}</div>

                {item.indicator && (
                  <div
                    className="text-sm font-medium mt-1"
                    style={{ color: item.color }}
                  >
                    {item.indicator}
                  </div>
                )}

                {item.change !== undefined && item.isPrice && item.change !== null && (
                  <div className={`text-sm ${
                    item.change > 0 ? 'text-trading-green' :
                    item.change < 0 ? 'text-trading-red' : 'text-gray-400'
                  }`}>
                    {item.change > 0 ? '+' : ''}{formatCurrency(item.change)}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MarketDataGrid;