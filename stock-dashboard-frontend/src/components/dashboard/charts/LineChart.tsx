import React from 'react';
import {
  LineChart as RechartsLineChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ComposedChart,
} from 'recharts';

interface LineDataPoint {
  date: string;
  close: number;
  volume?: number;
}

interface LineChartProps {
  data: LineDataPoint[];
  height?: number;
  showMovingAverages?: boolean;
  showBollingerBands?: boolean;
  movingAverages?: {
    sma_20?: number;
    sma_50?: number;
    sma_200?: number;
  };
}

// Custom tooltip for line chart
const LineTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length > 0) {
    const data = payload[0].payload;
    const priceData = payload.find((p: any) => p.dataKey === 'close');

    return (
      <div
        className="p-3 border rounded-lg shadow-lg"
        style={{
          backgroundColor: '#1a202c',
          borderColor: '#2d3748',
          color: '#ffffff'
        }}
      >
        <div className="text-sm font-medium mb-2">
          {new Date(payload[0].payload.date).toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric'
          })}
        </div>

        <div className="space-y-1 text-xs">
          <div className="flex justify-between">
            <span className="text-gray-400">Price:</span>
            <span>${priceData?.value.toFixed(2)}</span>
          </div>
          {payload.map((entry: any, index: number) => {
            if (entry.dataKey.startsWith('sma_') || entry.dataKey.startsWith('ema_')) {
              return (
                <div key={index} className="flex justify-between">
                  <span className="text-gray-400">{entry.dataKey.toUpperCase()}:</span>
                  <span style={{ color: entry.color }}>${entry.value.toFixed(2)}</span>
                </div>
              );
            }
            return null;
          })}
          {data.volume && (
            <div className="flex justify-between">
              <span className="text-gray-400">Volume:</span>
              <span>{data.volume.toLocaleString()}</span>
            </div>
          )}
        </div>
      </div>
    );
  }
  return null;
};

const LineChart: React.FC<LineChartProps> = ({
  data,
  height = 400,
  showMovingAverages = false,
  showBollingerBands = false,
  movingAverages
}) => {
  // Transform data for Recharts format and calculate moving averages
  const chartData = data.map((item, index) => {
    const chartItem: any = {
      ...item,
      timestamp: new Date(item.date).getTime(),
      dateLabel: new Date(item.date).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric'
      })
    };

    // Calculate moving averages dynamically - start from beginning
    if (showMovingAverages) {
      // SMA 20 - use available data points (minimum 1)
      const sma20Period = Math.min(20, index + 1);
      const sma20Data = data.slice(Math.max(0, index - sma20Period + 1), index + 1);
      chartItem.sma20 = sma20Data.reduce((sum, d) => sum + d.close, 0) / sma20Data.length;

      // SMA 50 - use available data points (minimum 1)
      const sma50Period = Math.min(50, index + 1);
      const sma50Data = data.slice(Math.max(0, index - sma50Period + 1), index + 1);
      chartItem.sma50 = sma50Data.reduce((sum, d) => sum + d.close, 0) / sma50Data.length;

      // SMA 200 - use available data points (minimum 1)
      const sma200Period = Math.min(200, index + 1);
      const sma200Data = data.slice(Math.max(0, index - sma200Period + 1), index + 1);
      chartItem.sma200 = sma200Data.reduce((sum, d) => sum + d.close, 0) / sma200Data.length;
    }

    // Calculate Bollinger Bands - start from beginning
    if (showBollingerBands) {
      const bbPeriod = Math.min(20, index + 1);
      const bbData = data.slice(Math.max(0, index - bbPeriod + 1), index + 1);
      const sma = bbData.reduce((sum, d) => sum + d.close, 0) / bbData.length;
      const variance = bbData.reduce((sum, d) => sum + Math.pow(d.close - sma, 2), 0) / bbData.length;
      const stdDev = Math.sqrt(variance);

      chartItem.bbUpper = sma + (2 * stdDev);
      chartItem.bbLower = sma - (2 * stdDev);
      chartItem.bbMiddle = sma;
    }

    return chartItem;
  });

  // Calculate price change for gradient
  const firstPrice = chartData[0]?.close || 0;
  const lastPrice = chartData[chartData.length - 1]?.close || 0;
  const isPositive = lastPrice >= firstPrice;

  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          <defs>
            <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
              <stop
                offset="5%"
                stopColor={isPositive ? '#00ff99' : '#ff4757'}
                stopOpacity={0.3}
              />
              <stop
                offset="95%"
                stopColor={isPositive ? '#00ff99' : '#ff4757'}
                stopOpacity={0.05}
              />
            </linearGradient>
          </defs>

          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#2d3748"
            opacity={0.3}
          />

          <XAxis
            dataKey="dateLabel"
            axisLine={false}
            tickLine={false}
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            interval="preserveStartEnd"
          />

          <YAxis
            domain={['dataMin - 2', 'dataMax + 2']}
            axisLine={false}
            tickLine={false}
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            tickFormatter={(value) => `$${value.toFixed(0)}`}
          />

          <Tooltip
            content={<LineTooltip />}
            cursor={{ stroke: '#00ff99', strokeWidth: 1, strokeOpacity: 0.5 }}
          />

          {/* Main price line - thick, bright green */}
          <Line
            type="monotone"
            dataKey="close"
            stroke="#00ff99"
            strokeWidth={3}
            dot={false}
            activeDot={{
              r: 5,
              fill: '#00ff99',
              stroke: '#0b0e14',
              strokeWidth: 2
            }}
            fill="url(#priceGradient)"
            fillOpacity={0.1}
          />

          {/* Moving averages with distinct styling */}
          {showMovingAverages && (
            <>
              {/* SMA 20: Thin line, orange */}
              <Line
                type="monotone"
                dataKey="sma20"
                stroke="#ffa500"
                strokeWidth={1}
                dot={false}
                connectNulls={false}
                strokeOpacity={0.8}
              />
              {/* SMA 50: Medium weight, red */}
              <Line
                type="monotone"
                dataKey="sma50"
                stroke="#ff6b6b"
                strokeWidth={1.5}
                dot={false}
                connectNulls={false}
                strokeOpacity={0.8}
              />
              {/* SMA 200: Thicker line, blue */}
              <Line
                type="monotone"
                dataKey="sma200"
                stroke="#74b9ff"
                strokeWidth={2}
                dot={false}
                connectNulls={false}
                strokeOpacity={0.8}
              />
            </>
          )}

          {/* Bollinger Bands - subtle fill area */}
          {showBollingerBands && (
            <>
              {/* Fill area between upper and lower bands */}
              <Area
                type="monotone"
                dataKey="bbUpper"
                stroke="none"
                fill="#a855f7"
                fillOpacity={0.1}
                dot={false}
                connectNulls={false}
              />
              <Area
                type="monotone"
                dataKey="bbLower"
                stroke="none"
                fill="#0b0e14"
                fillOpacity={1}
                dot={false}
                connectNulls={false}
              />
              {/* Upper band line */}
              <Line
                type="monotone"
                dataKey="bbUpper"
                stroke="#a855f7"
                strokeWidth={1}
                strokeOpacity={0.5}
                dot={false}
                connectNulls={false}
              />
              {/* Lower band line */}
              <Line
                type="monotone"
                dataKey="bbLower"
                stroke="#a855f7"
                strokeWidth={1}
                strokeOpacity={0.5}
                dot={false}
                connectNulls={false}
              />
              {/* Middle line (20-day SMA) - very subtle */}
              <Line
                type="monotone"
                dataKey="bbMiddle"
                stroke="#a855f7"
                strokeWidth={1}
                strokeOpacity={0.3}
                dot={false}
                connectNulls={false}
              />
            </>
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

export default LineChart;