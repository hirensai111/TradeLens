export * from './hooks';
export * from './context';

export const formatCurrency = (value: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(value);
};

export const formatNumber = (value: number): string => {
  return new Intl.NumberFormat('en-US').format(value);
};

export const formatPercentage = (value: number): string => {
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
};

export const formatDate = (dateString: string): string => {
  return new Date(dateString).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
};

export const formatDateTime = (dateString: string): string => {
  return new Date(dateString).toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

export const getChangeColorClass = (value: number): string => {
  if (value > 0) return 'text-trading-green';
  if (value < 0) return 'text-trading-red';
  return 'text-gray-600';
};

export const getRecommendationColorClass = (recommendation: string): string => {
  switch (recommendation.toLowerCase()) {
    case 'buy':
      return 'bg-green-900 text-trading-green border border-trading-green';
    case 'sell':
      return 'bg-red-900 text-trading-red border border-trading-red';
    case 'hold':
    default:
      return 'bg-yellow-900 text-yellow-400 border border-yellow-400';
  }
};

export const getSentimentColorClass = (sentiment: string): string => {
  switch (sentiment.toLowerCase()) {
    case 'bullish':
      return 'bg-green-900 text-trading-green border border-trading-green';
    case 'bearish':
      return 'bg-red-900 text-trading-red border border-trading-red';
    case 'neutral':
    default:
      return 'bg-yellow-900 text-yellow-400 border border-yellow-400';
  }
};