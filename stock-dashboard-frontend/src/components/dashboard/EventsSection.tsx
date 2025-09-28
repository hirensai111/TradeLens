import React, { useState } from 'react';
import { EventData } from '../../types/backend';
import { formatDateTime } from '../../utils';

interface EventsSectionProps {
  events: EventData[];
  sentiment?: {
    overall: string;
    score: number;
  };
}

const EventsSection: React.FC<EventsSectionProps> = ({
  events,
  sentiment,
}) => {
  const [showAllEvents, setShowAllEvents] = useState(false);

  const getSentimentColor = (sentimentValue: string) => {
    switch (sentimentValue.toLowerCase()) {
      case 'bullish':
      case 'positive':
        return '#00ff99';
      case 'bearish':
      case 'negative':
        return '#ff3d71';
      case 'neutral':
      default:
        return '#ffa500';
    }
  };

  const getSentimentColorWithIntensity = (sentimentValue: string, impact: string) => {
    const sentiment = sentimentValue.toLowerCase();
    const impactLevel = impact.toLowerCase();

    if (sentiment === 'bullish' || sentiment === 'positive') {
      // Green variants for bullish - brighter for higher impact
      switch (impactLevel) {
        case 'high': return { bg: '#0d4f2b', border: '#00ff99', text: '#00ff99' };
        case 'medium': return { bg: '#1a4d3a', border: '#00e68a', text: '#00e68a' };
        case 'low': return { bg: '#264d3a', border: '#66ff99', text: '#66ff99' };
        default: return { bg: '#1a4d3a', border: '#00ff99', text: '#00ff99' };
      }
    } else if (sentiment === 'bearish' || sentiment === 'negative') {
      // Red variants for bearish - brighter for higher impact
      switch (impactLevel) {
        case 'high': return { bg: '#4d1a2b', border: '#ff3d71', text: '#ff3d71' };
        case 'medium': return { bg: '#4d1a1a', border: '#ff4757', text: '#ff4757' };
        case 'low': return { bg: '#4d2626', border: '#ff7a7a', text: '#ff7a7a' };
        default: return { bg: '#4d1a1a', border: '#ff3d71', text: '#ff3d71' };
      }
    } else {
      // Neutral/yellow variants
      switch (impactLevel) {
        case 'high': return { bg: '#4d3a1a', border: '#ffb84d', text: '#ffb84d' };
        case 'medium': return { bg: '#4d3a1a', border: '#ffa500', text: '#ffa500' };
        case 'low': return { bg: '#4d4026', border: '#ffd966', text: '#ffd966' };
        default: return { bg: '#4d3a1a', border: '#ffa500', text: '#ffa500' };
      }
    }
  };

  const getConfidenceBadgeColor = (confidence: number) => {
    if (confidence >= 80) return { bg: '#1a4d3a', border: '#00ff99', text: '#00ff99' };
    if (confidence >= 60) return { bg: '#4d3a1a', border: '#ffa500', text: '#ffa500' };
    return { bg: '#4d1a1a', border: '#ff4757', text: '#ff4757' };
  };

  const getSentimentIndicator = (sentiment: string) => {
    switch (sentiment.toLowerCase()) {
      case 'bullish':
      case 'positive':
        return '🟢'; // Green circle for bullish
      case 'bearish':
      case 'negative':
        return '🔴'; // Red circle for bearish
      case 'neutral':
      default:
        return '🟡'; // Yellow circle for neutral
    }
  };

  const displayedEvents = showAllEvents ? events : events.slice(0, 3);

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
        <h2 className="text-xl font-semibold text-white">Market Events</h2>
      </div>

      <div className="p-6">
        {/* Recent Events */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-gray-300">Recent Events</h3>
            {events.length > 3 && (
              <button
                onClick={() => setShowAllEvents(!showAllEvents)}
                className="text-sm transition-colors"
                style={{ color: '#00ff99' }}
              >
                {showAllEvents ? 'Show Less' : 'Show All'}
              </button>
            )}
          </div>

          {events.length === 0 ? (
            <div
              className="border p-5 text-center rounded-lg"
              style={{
                backgroundColor: '#0b0e14',
                borderColor: '#2d3748',
                minHeight: '120px'
              }}
            >
              <div className="flex items-center justify-center h-full">
                <p className="text-gray-400 text-sm">No recent events available</p>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              {displayedEvents.map((event, index) => {
                const confidenceBadge = getConfidenceBadgeColor(event.confidence);
                return (
                  <div
                    key={index}
                    className="border p-5 rounded-lg"
                    style={{
                      backgroundColor: '#0b0e14',
                      borderColor: '#2d3748',
                      minHeight: '120px'
                    }}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-start space-x-2 flex-1 mr-3">
                        <span className="text-lg flex-shrink-0 mt-0.5">{getSentimentIndicator(event.sentiment)}</span>
                        <div className="min-w-0 flex-1">
                          <h4 className="text-sm font-medium text-white leading-relaxed">
                            {event.type}
                          </h4>
                          <p className="text-sm text-gray-300 mt-2 leading-relaxed whitespace-normal break-words">
                            {event.description}
                          </p>
                        </div>
                      </div>
                      <div
                        className="px-2 py-1 text-xs font-medium border flex-shrink-0 rounded"
                        style={{
                          backgroundColor: confidenceBadge.bg,
                          borderColor: confidenceBadge.border,
                          color: confidenceBadge.text
                        }}
                      >
                        {event.confidence}%
                      </div>
                    </div>

                    <div className="flex items-center justify-between text-xs mt-4 pt-3 border-t" style={{ borderColor: '#2d3748' }}>
                      <div className="flex items-center space-x-4">
                        <span className="text-gray-400">
                          {formatDateTime(event.date)}
                        </span>
                        <div className="flex items-center space-x-1">
                          <span
                            className="font-medium px-2 py-1 rounded"
                            style={{
                              backgroundColor: getSentimentColorWithIntensity(event.sentiment, 'medium').bg,
                              borderColor: getSentimentColorWithIntensity(event.sentiment, 'medium').border,
                              color: getSentimentColorWithIntensity(event.sentiment, 'medium').text,
                              border: '1px solid'
                            }}
                          >
                            {event.sentiment}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-gray-400">Impact:</span>
                        <span
                          className="font-medium text-xs px-2 py-1 rounded border"
                          style={{
                            backgroundColor: getSentimentColorWithIntensity(event.sentiment, event.impact).bg,
                            borderColor: getSentimentColorWithIntensity(event.sentiment, event.impact).border,
                            color: getSentimentColorWithIntensity(event.sentiment, event.impact).text
                          }}
                        >
                          {event.impact}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Legend */}
        <div
          className="border-t pt-4"
          style={{ borderColor: '#2d3748' }}
        >
          <h4 className="text-xs font-medium text-gray-400 mb-2">Legend</h4>
          <div className="flex items-center space-x-6 text-xs text-gray-500">
            <div className="flex items-center space-x-1">
              <span>🟢</span>
              <span>Bullish</span>
            </div>
            <div className="flex items-center space-x-1">
              <span>🔴</span>
              <span>Bearish</span>
            </div>
            <div className="flex items-center space-x-1">
              <span>🟡</span>
              <span>Neutral</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EventsSection;