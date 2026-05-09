import React, { useState, useMemo } from 'react';
import CandlestickChart from './CandlestickChart';
import VolumeChart from './VolumeChart';
import LineChart from './LineChart';
import { AnalysisResult, EventData } from '../../../types/backend';

interface FinancialChartProps {
  analysisData: AnalysisResult;
  selectedDateRange: {
    start: string;
    end: string;
  };
  chartType: 'candlestick' | 'line';
  showVolume: boolean;
  showMovingAverages: boolean;
  showBollingerBands: boolean;
  showEvents: boolean;
}


const FinancialChart: React.FC<FinancialChartProps> = ({
  analysisData,
  selectedDateRange,
  chartType,
  showVolume,
  showMovingAverages,
  showBollingerBands,
  showEvents
}) => {
  const [hoveredEvent, setHoveredEvent] = useState<EventData | null>(null);
  const [hoverTimeout, setHoverTimeout] = useState<NodeJS.Timeout | null>(null);

  const handleEventMouseEnter = (event: EventData) => {
    if (hoverTimeout) {
      clearTimeout(hoverTimeout);
      setHoverTimeout(null);
    }
    setHoveredEvent(event);
  };

  const handleEventMouseLeave = (e: React.MouseEvent) => {
    // Only clear hover if we're not moving to the tooltip
    if (!e.relatedTarget || !(e.relatedTarget as Element)?.closest('.event-tooltip')) {
      const timeout = setTimeout(() => {
        setHoveredEvent(null);
      }, 100); // Small delay to prevent flickering
      setHoverTimeout(timeout);
    }
  };

  // Filter and transform price data based on selected date range
  const chartData = useMemo(() => {
    if (!analysisData.price_history || analysisData.price_history.length === 0) {
      return [];
    }

    const startDate = new Date(selectedDateRange.start);
    const endDate = new Date(selectedDateRange.end);

    return analysisData.price_history
      .filter(item => {
        const itemDate = new Date(item.date);
        return itemDate >= startDate && itemDate <= endDate;
      })
      .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
      .map(item => ({
        date: item.date,
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
        volume: item.volume
      }));
  }, [analysisData.price_history, selectedDateRange.start, selectedDateRange.end]);

  // Filter events within the date range
  const filteredEvents = useMemo(() => {
    if (!analysisData.events || !showEvents) {
      return [];
    }

    const startDate = new Date(selectedDateRange.start);
    const endDate = new Date(selectedDateRange.end);

    return analysisData.events.filter(event => {
      const eventDate = new Date(event.date);
      return eventDate >= startDate && eventDate <= endDate;
    });
  }, [analysisData.events, selectedDateRange.start, selectedDateRange.end, showEvents]);

  // Get technical indicators for moving averages
  const technicalIndicators = useMemo(() => {
    return {
      sma_20: analysisData.technical_indicators?.sma_20,
      sma_50: analysisData.technical_indicators?.moving_averages?.sma_50,
      sma_200: analysisData.technical_indicators?.moving_averages?.sma_200,
    };
  }, [analysisData.technical_indicators]);

  if (chartData.length === 0) {
    return (
      <div
        className="w-full flex items-center justify-center"
        style={{
          backgroundColor: '#0b0e14',
          height: '400px'
        }}
      >
        <div className="text-center">
          <div className="text-gray-400 text-sm">No price data available</div>
          <div className="text-gray-500 text-xs mt-2">
            Try selecting a different date range
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Main Price Chart */}
      <div
        className="relative"
        style={{
          backgroundColor: '#0b0e14',
          borderRadius: '8px',
          padding: '16px'
        }}
      >
        {chartType === 'candlestick' ? (
          <CandlestickChart data={chartData} height={400} />
        ) : (
          <LineChart
            data={chartData}
            height={400}
            showMovingAverages={showMovingAverages}
            showBollingerBands={showBollingerBands}
            movingAverages={technicalIndicators}
          />
        )}

        {/* Event Markers Overlay */}
        {showEvents && filteredEvents.length > 0 && (
          <div className="absolute top-4 right-4 space-y-1 z-10">
            <div className="text-xs text-gray-400 mb-2">Events</div>
            {filteredEvents.slice(0, 5).map((event, index) => (
              <div
                key={`${event.date}-${index}`}
                className="flex items-center space-x-2 text-xs cursor-pointer p-2 rounded transition-colors duration-150"
                onMouseEnter={() => setHoveredEvent(event)}
                onMouseLeave={(e) => {
                  // Only clear hover if we're not moving to the tooltip
                  if (!e.relatedTarget || !(e.relatedTarget as Element)?.closest('.event-tooltip')) {
                    setHoveredEvent(null);
                  }
                }}
                style={{
                  backgroundColor: hoveredEvent?.date === event.date && hoveredEvent?.type === event.type ? '#2d3748' : 'transparent'
                }}
              >
                <div
                  className="w-2 h-2 rounded-full"
                  style={{
                    backgroundColor: event.sentiment.toLowerCase().includes('bullish')
                      ? '#00ff99'
                      : event.sentiment.toLowerCase().includes('bearish')
                      ? '#ff4757'
                      : '#ffa500'
                  }}
                />
                <span className="text-gray-300 truncate max-w-32">
                  {new Date(event.date).toLocaleDateString('en-US', {
                    month: 'short',
                    day: 'numeric'
                  })}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Event Tooltip */}
        {hoveredEvent && (
          <div
            className="event-tooltip absolute top-16 right-4 p-3 border rounded-lg shadow-lg max-w-sm z-20"
            style={{
              backgroundColor: '#1a202c',
              borderColor: '#2d3748',
              color: '#ffffff'
            }}
            onMouseEnter={() => setHoveredEvent(hoveredEvent)}
            onMouseLeave={() => setHoveredEvent(null)}
          >
            <div className="text-sm font-medium mb-2">
              {new Date(hoveredEvent.date).toLocaleDateString('en-US', {
                month: 'long',
                day: 'numeric',
                year: 'numeric'
              })}
            </div>
            <div className="text-xs space-y-1">
              <div className="flex justify-between">
                <span className="text-gray-400">Type:</span>
                <span>{hoveredEvent.type}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Sentiment:</span>
                <span
                  style={{
                    color: hoveredEvent.sentiment.toLowerCase().includes('bullish')
                      ? '#00ff99'
                      : hoveredEvent.sentiment.toLowerCase().includes('bearish')
                      ? '#ff4757'
                      : '#ffa500'
                  }}
                >
                  {hoveredEvent.sentiment}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Confidence:</span>
                <span>{(hoveredEvent.confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="mt-2 pt-2 border-t" style={{ borderColor: '#2d3748' }}>
                <p className="text-gray-300 text-xs leading-relaxed">
                  {hoveredEvent.description.length > 150
                    ? hoveredEvent.description.substring(0, 150) + '...'
                    : hoveredEvent.description
                  }
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Volume Chart */}
      {showVolume && (
        <div
          style={{
            backgroundColor: '#0b0e14',
            borderRadius: '8px',
            padding: '16px'
          }}
        >
          <div className="text-sm text-gray-400 mb-2">Volume</div>
          <VolumeChart data={chartData} height={120} />
        </div>
      )}

      {/* Chart Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
        <div
          className="p-3 border rounded"
          style={{
            backgroundColor: '#0b0e14',
            borderColor: '#2d3748'
          }}
        >
          <div className="text-gray-400 mb-1">Period High</div>
          <div className="text-trading-green font-semibold">
            ${Math.max(...chartData.map(d => d.high)).toFixed(2)}
          </div>
        </div>

        <div
          className="p-3 border rounded"
          style={{
            backgroundColor: '#0b0e14',
            borderColor: '#2d3748'
          }}
        >
          <div className="text-gray-400 mb-1">Period Low</div>
          <div className="text-trading-red font-semibold">
            ${Math.min(...chartData.map(d => d.low)).toFixed(2)}
          </div>
        </div>

        <div
          className="p-3 border rounded"
          style={{
            backgroundColor: '#0b0e14',
            borderColor: '#2d3748'
          }}
        >
          <div className="text-gray-400 mb-1">Avg Volume</div>
          <div className="text-white font-semibold">
            {(chartData.reduce((sum, d) => sum + d.volume, 0) / chartData.length).toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </div>
        </div>

        <div
          className="p-3 border rounded"
          style={{
            backgroundColor: '#0b0e14',
            borderColor: '#2d3748'
          }}
        >
          <div className="text-gray-400 mb-1">Data Points</div>
          <div className="text-white font-semibold">
            {chartData.length}
          </div>
        </div>
      </div>
    </div>
  );
};

export default FinancialChart;