import React from 'react';
import { TechnicalIndicators } from '../../types/backend';

interface TradingSignalsProps {
  technicalIndicators: TechnicalIndicators;
  recommendation?: {
    action: string;
    confidence: number;
    reasoning: string;
  };
}

const TradingSignals: React.FC<TradingSignalsProps> = ({
  technicalIndicators,
  recommendation,
}) => {
  const getRecommendationColor = (action: string) => {
    switch (action.toLowerCase()) {
      case 'buy':
        return '#00ff99';
      case 'sell':
        return '#ff4757';
      case 'hold':
      default:
        return '#ffa500';
    }
  };

  const getConfidenceLevel = (confidence: number) => {
    if (confidence >= 80) return { label: 'High', color: '#00ff99' };
    if (confidence >= 60) return { label: 'Medium', color: '#ffa500' };
    return { label: 'Low', color: '#ff4757' };
  };


  const confidenceLevel = getConfidenceLevel(recommendation?.confidence || 50);

  return (
    <div
      className="border"
      style={{
        backgroundColor: '#1a202c',
        borderColor: '#2d3748'
      }}
    >
      {/* Header */}
      <div
        className="border-b px-6 py-4"
        style={{ borderColor: '#2d3748' }}
      >
        <h2 className="text-xl font-semibold text-white">Trading Signals</h2>
      </div>

      <div className="p-6 space-y-6">
        {/* Main Recommendation */}
        <div
          className="border p-4"
          style={{
            backgroundColor: '#0b0e14',
            borderColor: '#2d3748'
          }}
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold text-white">Recommendation</h3>
            <div
              className="px-3 py-1 text-sm font-bold uppercase tracking-wide"
              style={{
                backgroundColor: getRecommendationColor(recommendation?.action || 'hold'),
                color: '#000000'
              }}
            >
              {recommendation?.action || 'N/A'}
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Confidence</span>
              <div className="flex items-center space-x-2">
                <div className="w-20 bg-gray-700 h-2">
                  <div
                    className="h-2 transition-all duration-500"
                    style={{
                      width: `${recommendation?.confidence || 50}%`,
                      backgroundColor: confidenceLevel.color
                    }}
                  ></div>
                </div>
                <span
                  className="text-sm font-medium"
                  style={{ color: confidenceLevel.color }}
                >
                  {recommendation?.confidence || 50}% ({confidenceLevel.label})
                </span>
              </div>
            </div>
          </div>

          {recommendation?.reasoning && (
            <p className="text-sm text-gray-400 mt-3 leading-relaxed">
              {recommendation.reasoning}
            </p>
          )}
        </div>


        {/* Risk Warning */}
        <div
          className="border p-3"
          style={{
            backgroundColor: '#2d1b1b',
            borderColor: '#ff4757'
          }}
        >
          <div className="flex items-start space-x-2">
            <span className="text-red-400 text-sm">⚠️</span>
            <div>
              <p className="text-red-400 text-xs font-medium mb-1">Risk Warning</p>
              <p className="text-red-300 text-xs leading-relaxed">
                Trading signals are for informational purposes only. Past performance does not guarantee future results. Always conduct your own research before making investment decisions.
              </p>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default TradingSignals;