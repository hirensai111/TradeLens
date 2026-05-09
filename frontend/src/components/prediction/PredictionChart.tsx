import React from 'react';
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot,
  Area,
} from 'recharts';

interface PredictionDataPoint {
  date: string;
  actualPrice?: number;
  predictedPrice?: number;
  upperBound?: number;
  lowerBound?: number;
  isActual: boolean;
  isBullish?: boolean;
}

interface PredictionChartProps {
  historicalData: Array<{ date: string; price: number }>;
  predictions: Array<{
    date: string;
    price: number;
    upperBound?: number;
    lowerBound?: number;
    changePct: number;
  }>;
  currentPrice: number;
  height?: number;
}

const PredictionTooltip = ({ active, payload }: any) => {
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
          {new Date(data.date).toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric'
          })}
        </div>

        {data.actualPrice !== undefined && (
          <div className="text-xs mb-1">
            <span className="text-gray-400">Actual: </span>
            <span className="font-semibold text-green-400">
              ${data.actualPrice.toFixed(2)}
            </span>
          </div>
        )}

        {data.predictedPrice !== undefined && (
          <>
            <div className="text-xs mb-1">
              <span className="text-gray-400">Predicted: </span>
              <span className="font-semibold" style={{
                color: data.isBullish ? '#00ff99' : '#ff4757'
              }}>
                ${data.predictedPrice.toFixed(2)}
              </span>
            </div>

            {data.upperBound !== undefined && data.lowerBound !== undefined && (
              <div className="text-xs text-gray-400 mt-1">
                Range: ${data.lowerBound.toFixed(2)} - ${data.upperBound.toFixed(2)}
              </div>
            )}
          </>
        )}
      </div>
    );
  }
  return null;
};

const PredictionChart: React.FC<PredictionChartProps> = ({
  historicalData,
  predictions,
  currentPrice,
  height = 400,
}) => {
  // Combine historical and prediction data
  const chartData: PredictionDataPoint[] = [
    // Add historical data (last 5-7 days)
    ...historicalData.map(d => ({
      date: d.date,
      actualPrice: d.price,
      isActual: true,
    })),
    // Add current price as transition point
    {
      date: new Date().toISOString().split('T')[0],
      actualPrice: currentPrice,
      predictedPrice: currentPrice,
      isActual: true,
      isBullish: true,
    },
    // Add predictions
    ...predictions.map(p => ({
      date: p.date,
      predictedPrice: p.price,
      upperBound: p.upperBound,
      lowerBound: p.lowerBound,
      isActual: false,
      isBullish: p.changePct >= 0,
    })),
  ];

  // Calculate Y-axis domain
  const allPrices = [
    ...historicalData.map(d => d.price),
    currentPrice,
    ...predictions.map(p => p.price),
    ...predictions.filter(p => p.lowerBound).map(p => p.lowerBound!),
    ...predictions.filter(p => p.upperBound).map(p => p.upperBound!),
  ];

  const minPrice = Math.min(...allPrices);
  const maxPrice = Math.max(...allPrices);
  const padding = (maxPrice - minPrice) * 0.1;
  const yDomain = [minPrice - padding, maxPrice + padding];

  // Determine overall prediction trend
  const lastPrediction = predictions[predictions.length - 1];
  const isBullishOverall = lastPrediction && lastPrediction.changePct >= 0;

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          <defs>
            <linearGradient id="predictionGradient" x1="0" y1="0" x2="0" y2="1">
              <stop
                offset="5%"
                stopColor={isBullishOverall ? "#00ff99" : "#ff4757"}
                stopOpacity={0.3}
              />
              <stop
                offset="95%"
                stopColor={isBullishOverall ? "#00ff99" : "#ff4757"}
                stopOpacity={0.05}
              />
            </linearGradient>
          </defs>

          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(255, 255, 255, 0.1)"
            vertical={false}
          />

          <XAxis
            dataKey="date"
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            tickLine={{ stroke: 'rgba(255, 255, 255, 0.1)' }}
            axisLine={{ stroke: 'rgba(255, 255, 255, 0.1)' }}
            tickFormatter={(value) => {
              const date = new Date(value);
              return `${date.getMonth() + 1}/${date.getDate()}`;
            }}
          />

          <YAxis
            domain={yDomain}
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            tickLine={{ stroke: 'rgba(255, 255, 255, 0.1)' }}
            axisLine={{ stroke: 'rgba(255, 255, 255, 0.1)' }}
            tickFormatter={(value) => `$${value.toFixed(0)}`}
          />

          <Tooltip content={<PredictionTooltip />} />

          {/* Confidence interval area for predictions */}
          <Area
            type="monotone"
            dataKey="upperBound"
            stroke="none"
            fill="url(#predictionGradient)"
            fillOpacity={0.3}
          />
          <Area
            type="monotone"
            dataKey="lowerBound"
            stroke="none"
            fill="url(#predictionGradient)"
            fillOpacity={0.3}
          />

          {/* Historical price line (solid green) */}
          <Line
            type="monotone"
            dataKey="actualPrice"
            stroke="#00ff99"
            strokeWidth={2.5}
            dot={false}
            connectNulls={false}
          />

          {/* Predicted price line (dotted, color based on trend) */}
          <Line
            type="monotone"
            dataKey="predictedPrice"
            stroke={isBullishOverall ? "#00ff99" : "#ff4757"}
            strokeWidth={2.5}
            strokeDasharray="5 5"
            dot={false}
            connectNulls={true}
          />

          {/* Current price marker */}
          <ReferenceDot
            x={new Date().toISOString().split('T')[0]}
            y={currentPrice}
            r={6}
            fill="#00e5ff"
            stroke="#fff"
            strokeWidth={2}
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-8 h-0.5 bg-green-400"></div>
          <span className="text-gray-400">Historical</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-8 h-0.5 border-t-2 border-dashed" style={{
            borderColor: isBullishOverall ? '#00ff99' : '#ff4757'
          }}></div>
          <span className="text-gray-400">
            Predicted ({isBullishOverall ? 'Bullish' : 'Bearish'})
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-cyan-400 border-2 border-white"></div>
          <span className="text-gray-400">Current Price</span>
        </div>
      </div>
    </div>
  );
};

export default PredictionChart;
