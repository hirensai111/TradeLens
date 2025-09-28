interface PriceDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface OHLCResult {
  open: number;
  high: number;
  low: number;
  close: number;
  change: number;
  changePercent: number;
}

export const calculateOHLCForDateRange = (
  priceData: PriceDataPoint[],
  startDate: string,
  endDate: string
): OHLCResult | null => {
  if (!priceData || priceData.length === 0) {
    return null;
  }

  // Convert dates to comparable format (YYYY-MM-DD)
  const normalizeDate = (dateStr: string): string => {
    return new Date(dateStr).toISOString().split('T')[0];
  };

  const normalizedStartDate = normalizeDate(startDate);
  const normalizedEndDate = normalizeDate(endDate);

  // Filter data within the date range
  const filteredData = priceData.filter(point => {
    const pointDate = normalizeDate(point.date);
    return pointDate >= normalizedStartDate && pointDate <= normalizedEndDate;
  });

  if (filteredData.length === 0) {
    return null;
  }

  // Sort by date to ensure proper ordering
  filteredData.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

  // Calculate OHLC based on logic:
  // Single date: use that day's OHLC
  // Date range: Open from first day, Close from last day, High/Low across range
  const firstDay = filteredData[0];
  const lastDay = filteredData[filteredData.length - 1];

  const open = firstDay.open;
  const close = lastDay.close;
  const high = Math.max(...filteredData.map(point => point.high));
  const low = Math.min(...filteredData.map(point => point.low));

  // Calculate change and change percentage
  const change = close - open;
  const changePercent = (change / open) * 100;

  return {
    open,
    high,
    low,
    close,
    change,
    changePercent
  };
};

export const getLatestPrice = (priceData: PriceDataPoint[]): number => {
  if (!priceData || priceData.length === 0) {
    return 0;
  }

  // Sort by date and get the latest close price
  const sorted = [...priceData].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
  return sorted[0]?.close || 0;
};

export const formatDateForDisplay = (dateStr: string): string => {
  return new Date(dateStr).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  });
};

export const isValidDateRange = (startDate: string, endDate: string): boolean => {
  const start = new Date(startDate);
  const end = new Date(endDate);
  return start <= end && !isNaN(start.getTime()) && !isNaN(end.getTime());
};