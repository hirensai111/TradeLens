import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

interface VolumeDataPoint {
  date: string;
  volume: number;
  open: number;
  close: number;
}

interface VolumeChartProps {
  data: VolumeDataPoint[];
  height?: number;
}

// Custom tooltip for volume data
const VolumeTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length > 0) {
    const data = payload[0].payload;

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
            <span className="text-gray-400">Volume:</span>
            <span>{data.volume.toLocaleString()}</span>
          </div>
        </div>
      </div>
    );
  }
  return null;
};

const VolumeChart: React.FC<VolumeChartProps> = ({ data, height = 120 }) => {
  // Transform data for Recharts format and add color based on price movement
  const chartData = data.map(item => {
    const isPositive = item.close >= item.open;
    return {
      ...item,
      timestamp: new Date(item.date).getTime(),
      dateLabel: new Date(item.date).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric'
      }),
      volumeColor: isPositive ? '#00ff99' : '#ff4757',
      isPositive
    };
  });

  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={chartData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
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
            tick={{ fill: '#9ca3af', fontSize: 10 }}
            interval="preserveStartEnd"
          />

          <YAxis
            axisLine={false}
            tickLine={false}
            tick={{ fill: '#9ca3af', fontSize: 10 }}
            tickFormatter={(value) => {
              if (value >= 1000000) {
                return `${(value / 1000000).toFixed(1)}M`;
              } else if (value >= 1000) {
                return `${(value / 1000).toFixed(0)}K`;
              }
              return value.toString();
            }}
          />

          <Tooltip
            content={<VolumeTooltip />}
            cursor={{ fill: 'rgba(255, 255, 255, 0.1)' }}
          />

          <Bar
            dataKey="volume"
            radius={[1, 1, 0, 0]}
          >
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.volumeColor}
                fillOpacity={0.8}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default VolumeChart;