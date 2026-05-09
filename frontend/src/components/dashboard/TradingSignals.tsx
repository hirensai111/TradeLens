import React from 'react';
import { TechnicalIndicators } from '../../types/backend';
import GlassCard from '../common/GlassCard';

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
    <GlassCard title="Trading Signals">
      <div className="space-y-6">
        {/* Main Recommendation */}
        <div
          className="p-4 rounded-2xl"
          style={{
            background: 'rgba(11, 14, 20, 0.6)',
            border: '1px solid rgba(255, 255, 255, 0.1)'
          }}
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold text-white">Recommendation</h3>
            <div
              className="px-3 py-1 text-sm font-bold uppercase tracking-wide rounded-xl"
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
          className="p-3 rounded-2xl"
          style={{
            backgroundColor: 'rgba(45, 27, 27, 0.6)',
            border: '1px solid rgba(255, 71, 87, 0.3)'
          }}
        >
          <div>
            <p className="text-red-400 text-xs font-medium mb-1">Risk Warning</p>
            <p className="text-red-300 text-xs leading-relaxed">
              Trading signals are for informational purposes only. Past performance does not guarantee future results. Always conduct your own research before making investment decisions.
            </p>
          </div>
        </div>

      </div>
    </GlassCard>
  );
};

export default TradingSignals;