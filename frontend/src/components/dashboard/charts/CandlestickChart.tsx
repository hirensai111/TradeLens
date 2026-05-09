import React from 'react';
import {
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Line,
} from 'recharts';

interface CandlestickDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface CandlestickChartProps {
  data: CandlestickDataPoint[];
  height?: number;
}


// Custom tooltip for candlestick data
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length > 0) {
    const data = payload[0].payload;
    const isPositive = data.close >= data.open;

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
          {new Date(data.date).toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric'
          })}
        </div>

        <div className="space-y-1 text-xs">
          <div className="flex justify-between">
            <span className="text-gray-400">Open:</span>
            <span>${data.open.toFixed(2)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">High:</span>
            <span className="text-trading-green">${data.high.toFixed(2)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Low:</span>
            <span className="text-trading-red">${data.low.toFixed(2)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Close:</span>
            <span style={{ color: isPositive ? '#00ff99' : '#ff4757' }}>
              ${data.close.toFixed(2)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Volume:</span>
            <span>{data.volume.toLocaleString()}</span>
          </div>
          <div className="flex justify-between border-t pt-1" style={{ borderColor: '#2d3748' }}>
            <span className="text-gray-400">Change:</span>
            <span style={{ color: isPositive ? '#00ff99' : '#ff4757' }}>
              {isPositive ? '+' : ''}${(data.close - data.open).toFixed(2)}
              ({isPositive ? '+' : ''}{(((data.close - data.open) / data.open) * 100).toFixed(2)}%)
            </span>
          </div>
        </div>
      </div>
    );
  }
  return null;
};

const CandlestickChart: React.FC<CandlestickChartProps> = ({ data, height = 400 }) => {
  // Transform data for chart format
  const chartData = data.map(item => ({
    ...item,
    dateLabel: new Date(item.date).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric'
    })
  }));

  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
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
            content={<CustomTooltip />}
            cursor={{ stroke: '#00ff99', strokeWidth: 1, strokeOpacity: 0.5 }}
          />

          {/* High-Low line (wick) */}
          <Line
            type="linear"
            dataKey="high"
            stroke="#666666"
            strokeWidth={1}
            dot={false}
            connectNulls={false}
          />

          <Line
            type="linear"
            dataKey="low"
            stroke="#666666"
            strokeWidth={1}
            dot={false}
            connectNulls={false}
          />

          {/* Close price line for trend */}
          <Line
            type="monotone"
            dataKey="close"
            stroke="#00ff99"
            strokeWidth={1.5}
            dot={{ r: 2, fill: '#00ff99' }}
            connectNulls={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

export default CandlestickChart;