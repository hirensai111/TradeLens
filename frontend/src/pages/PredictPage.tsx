import React, { useState, useRef, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import ModeSelector from '../components/common/ModeSelector';
import AnimatedBackground from '../components/common/AnimatedBackground';
import PredictionChart from '../components/prediction/PredictionChart';
import apiService from '../services/api';

interface PredictionDay {
  date: string; // ISO format YYYY-MM-DD
  displayDate?: string; // Display format like "Jan 1"
  price: number;
  change: number;
  confidence: number;
  changePct?: number;
}

interface AIReasoning {
  source: string;
  reasoning: string;
  key_insights: string[];
  risk_assessment: any;
  recommendation: string;
}

interface NewsAnalysis {
  total_articles: number;
  sentiment_label: string;
  sentiment_score: number;
  key_events: any[];
}

interface PredictionData {
  dataPoints: number;
  days: PredictionDay[];
  outlook: string;
  targetPrice: string;
  supportLevel: string;
  resistanceLevel: string;
  volatility: string;
  riskLevel: string;
  positionSize: string;
  avgReturn: number;
  winRate: number;
  currentPrice?: number;
  aiReasoning?: AIReasoning;
  newsAnalysis?: NewsAnalysis;
}

interface AllPredictions {
  [key: number]: PredictionData;
}

const POPULAR_STOCKS = [
  { ticker: 'AAPL', name: 'Apple Inc.' },
  { ticker: 'MSFT', name: 'Microsoft Corp.' },
  { ticker: 'GOOGL', name: 'Alphabet Inc.' },
  { ticker: 'AMZN', name: 'Amazon.com Inc.' },
  { ticker: 'TSLA', name: 'Tesla Inc.' },
  { ticker: 'NVDA', name: 'NVIDIA Corp.' },
  { ticker: 'META', name: 'Meta Platforms' },
  { ticker: 'NFLX', name: 'Netflix Inc.' },
];

const PredictPage: React.FC = () => {
  const [ticker, setTicker] = useState('');
  const [selectedDays, setSelectedDays] = useState(5);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [loadingStage, setLoadingStage] = useState('');
  const [allPredictions, setAllPredictions] = useState<AllPredictions | null>(null);
  const [historicalData, setHistoricalData] = useState<Array<{date: string; price: number}>>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    inputRef.current?.focus();

    // Check if we have ticker from navigation state
    const locationState = location.state as { ticker?: string } | null;
    if (locationState?.ticker) {
      const tickerUpper = locationState.ticker.toUpperCase();

      // Try to load cached predictions from sessionStorage
      const cachedKey = `predictions_${tickerUpper}`;
      const cached = sessionStorage.getItem(cachedKey);

      console.log('Navigation to predict with ticker:', tickerUpper);
      console.log('Cache found:', !!cached);

      if (cached) {
        try {
          const parsedCache = JSON.parse(cached);
          console.log('Loading cached predictions for', tickerUpper);
          setAllPredictions(parsedCache.predictions);
          setHistoricalData(parsedCache.historicalData);
          setTicker(tickerUpper);
        } catch (e) {
          console.error('Failed to parse cached predictions:', e);
          // If cache is invalid, just set the ticker
          setTicker(tickerUpper);
        }
      } else {
        // No cache found, just set the ticker (don't auto-fetch)
        console.log('No cache found for', tickerUpper);
        setTicker(tickerUpper);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location]);

  const generateMockPredictions = (tickerSymbol: string, days: number): PredictionData => {
    const basePrice = 150 + Math.random() * 100;
    const trend = Math.random() > 0.5 ? 1 : -1;
    const volatility = 0.5 + Math.random() * 2;

    const predictions: PredictionData = {
      dataPoints: Math.floor(800 + Math.random() * 400),
      days: [],
      outlook: trend > 0 ? 'Bullish - Upward momentum expected' : 'Cautious - Potential downside risk',
      targetPrice: (basePrice * (1 + trend * 0.05)).toFixed(2),
      supportLevel: (basePrice * 0.95).toFixed(2),
      resistanceLevel: (basePrice * 1.05).toFixed(2),
      volatility: volatility > 1.5 ? 'High' : volatility > 1 ? 'Medium' : 'Low',
      riskLevel: volatility > 1.5 ? 'High Risk' : volatility > 1 ? 'Moderate Risk' : 'Low Risk',
      positionSize: volatility > 1.5 ? '1-2% of portfolio' : volatility > 1 ? '2-5% of portfolio' : '3-7% of portfolio',
      avgReturn: (5 + Math.random() * 10) * trend,
      winRate: 65 + Math.random() * 20
    };

    let currentPrice = basePrice;
    const today = new Date();

    for (let i = 1; i <= days; i++) {
      const date = new Date(today);
      date.setDate(date.getDate() + i);

      const dailyChange = (Math.random() - 0.45) * volatility * trend;
      currentPrice *= (1 + dailyChange / 100);

      predictions.days.push({
        date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        price: currentPrice,
        change: ((currentPrice - basePrice) / basePrice * 100),
        confidence: 70 + Math.random() * 25
      });
    }

    return predictions;
  };

  const handlePredict = async (tickerToPredict?: string) => {
    const cleanTicker = (tickerToPredict || ticker).trim().toUpperCase();

    if (!cleanTicker) {
      alert('Please enter a stock ticker symbol');
      return;
    }

    if (cleanTicker.length > 10 || !/^[A-Z]+$/.test(cleanTicker)) {
      alert('Invalid ticker symbol. Please use only letters (max 10 characters)');
      return;
    }

    setTicker(cleanTicker);
    setIsLoading(true);
    setAllPredictions(null);

    // Clear any cached predictions for this ticker
    sessionStorage.removeItem(`predictions_${cleanTicker}`);

    const stages = [
      { progress: 20, text: 'Fetching historical data...' },
      { progress: 40, text: 'Running mathematical models...' },
      { progress: 60, text: 'Calculating technical indicators...' },
      { progress: 80, text: 'Generating predictions...' },
      { progress: 100, text: 'Finalizing report...' }
    ];

    try {
      // Show loading stages
      for (let i = 0; i < stages.length - 1; i++) {
        setLoadingProgress(stages[i].progress);
        setLoadingStage(stages[i].text);
        await new Promise(resolve => setTimeout(resolve, 500));
      }

      // Fetch prediction data once
      setLoadingStage('Generating predictions for all timeframes...');
      const response = await apiService.getPredictions(cleanTicker);

      const apiPredictions = response.predictions || [];
      const summary = response.summary || {};
      const aiReasoning = response.aiReasoning || null;
      const newsAnalysis = response.newsAnalysis || null;

      // Generate stable historical data (last 7 days) - only once
      const historical = Array.from({ length: 7 }, (_, i) => {
        const date = new Date();
        date.setDate(date.getDate() - (7 - i));
        // Use a deterministic variation based on the day to avoid random changes
        const variation = (Math.sin(i) * 5);
        return {
          date: date.toISOString().split('T')[0],
          price: (response.currentPrice || 100) + variation,
        };
      });
      setHistoricalData(historical);

      // Generate predictions for all 3 timeframes from the same data
      const predictions: AllPredictions = {};
      const timeframes = [5, 15, 30];

      for (const days of timeframes) {
        // Extend predictions if API doesn't provide enough days
        const extendedPredictions: Array<{
          date: string;
          price: number;
          changePct: number;
          confidence: number;
        }> = [];

        for (let i = 0; i < days; i++) {
          if (i < apiPredictions.length) {
            // Use API data if available
            extendedPredictions.push(apiPredictions[i]);
          } else {
            // Extend predictions based on trend with realistic market volatility
            const lastPrediction = extendedPredictions[extendedPredictions.length - 1];
            const prevPrediction = extendedPredictions[Math.max(0, extendedPredictions.length - 2)];

            // Calculate recent momentum from last two predictions
            const recentMomentum = extendedPredictions.length > 1
              ? ((lastPrediction.price - prevPrediction.price) / prevPrediction.price)
              : 0;

            // Base trend direction with mean reversion
            // Stronger trends get pulled back gradually (mean reversion)
            const baseTrend = summary.trend === 'bullish' ? 0.003 : -0.003;
            const meanReversion = -recentMomentum * 0.3; // 30% mean reversion

            // Multi-frequency volatility for realistic price action
            const dayIndex = i - apiPredictions.length;
            const highFreqVol = Math.sin(dayIndex * 1.3 + cleanTicker.charCodeAt(0)) * 0.012; // Daily noise
            const medFreqVol = Math.sin(dayIndex * 0.4 + cleanTicker.charCodeAt(1)) * 0.008; // Multi-day swings
            const lowFreqVol = Math.cos(dayIndex * 0.15) * 0.005; // Trend variations

            // Random-like component using ticker hash for consistency
            const tickerHash = cleanTicker.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0);
            const pseudoRandom = Math.sin(dayIndex * 2.7 + tickerHash) * 0.006;

            // Risk level affects volatility magnitude
            const volMultiplier = summary.riskLevel === 'high' ? 1.5 :
                                 summary.riskLevel === 'medium' ? 1.0 : 0.6;

            // Combine all components for realistic daily movement
            const totalVolatility = (highFreqVol + medFreqVol + lowFreqVol + pseudoRandom) * volMultiplier;
            const dailyChange = baseTrend + meanReversion + totalVolatility;
            const newPrice = lastPrediction.price * (1 + dailyChange);

            const lastDate = new Date(lastPrediction.date);
            lastDate.setDate(lastDate.getDate() + 1);

            extendedPredictions.push({
              date: lastDate.toISOString().split('T')[0],
              price: newPrice,
              changePct: dailyChange * 100,
              confidence: Math.max(50, lastPrediction.confidence - 1.5), // Gradual confidence decay
            });
          }
        }

        predictions[days] = {
          dataPoints: response.dataPoints || 1000,
          days: extendedPredictions.map((p: any, index: number) => {
            // Keep the original date format from API, or generate sequential dates
            const predictionDate = p.date ? new Date(p.date) : new Date(Date.now() + (index + 1) * 24 * 60 * 60 * 1000);

            return {
              date: predictionDate.toISOString().split('T')[0], // Store as YYYY-MM-DD
              displayDate: predictionDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }), // For table display
              price: p.price,
              change: p.changePct,
              confidence: p.confidence,
              changePct: p.changePct
            };
          }),
          outlook: summary.trend === 'bullish' ? 'Bullish - Upward momentum expected' : 'Cautious - Potential downside risk',
          targetPrice: extendedPredictions.length > 0 ? extendedPredictions[extendedPredictions.length - 1].price.toFixed(2) : response.currentPrice?.toFixed(2),
          supportLevel: (response.currentPrice * 0.95).toFixed(2),
          resistanceLevel: (response.currentPrice * 1.05).toFixed(2),
          volatility: summary.riskLevel === 'high' ? 'High' : summary.riskLevel === 'medium' ? 'Medium' : 'Low',
          riskLevel: summary.riskLevel === 'high' ? 'High Risk' : summary.riskLevel === 'medium' ? 'Moderate Risk' : 'Low Risk',
          positionSize: summary.riskLevel === 'high' ? '1-2% of portfolio' : summary.riskLevel === 'medium' ? '2-5% of portfolio' : '3-7% of portfolio',
          avgReturn: summary.avgChange || 0,
          winRate: summary.winRate || 65,
          currentPrice: response.currentPrice,
          aiReasoning: aiReasoning,
          newsAnalysis: newsAnalysis
        };
      }

      setLoadingProgress(stages[4].progress);
      setLoadingStage(stages[4].text);
      await new Promise(resolve => setTimeout(resolve, 300));

      // Cache predictions in sessionStorage
      const cacheKey = `predictions_${cleanTicker}`;
      const cacheData = {
        predictions,
        historicalData: historical
      };
      sessionStorage.setItem(cacheKey, JSON.stringify(cacheData));
      console.log('Cached predictions for', cleanTicker, 'with', Object.keys(predictions).length, 'timeframes');

      setAllPredictions(predictions);
      setIsLoading(false);

    } catch (error: any) {
      console.error('Error fetching predictions:', error);
      setIsLoading(false);

      // Show error message
      if (error.message?.includes('No data available')) {
        alert(`No data available for ${cleanTicker}. Please run analysis first from the Dashboard.`);
      } else if (error.code === 'PREMIUM_ACCESS_REQUIRED') {
        alert(`${cleanTicker} is not available in free tier. Available stocks: ${error.available_stocks?.join(', ')}`);
      } else {
        alert(`Failed to generate predictions: ${error.message || 'Unknown error'}`);
      }
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handlePredict();
    }
  };

  const handleReset = () => {
    // Clear cached predictions for current ticker
    if (ticker) {
      sessionStorage.removeItem(`predictions_${ticker}`);
    }

    setAllPredictions(null);
    setHistoricalData([]);
    setTicker('');
    setLoadingProgress(0);
    setLoadingStage('');
  };

  const handlePopularStockClick = (stockTicker: string) => {
    handlePredict(stockTicker);
  };

  const predictionData = allPredictions?.[selectedDays] || null;

  return (
    <div className="min-h-screen relative overflow-hidden">
      <AnimatedBackground />

      {/* Conditional Header - TradeLens style before prediction, Dashboard style after */}
      {allPredictions ? (
        // Dashboard-style header (after prediction)
        <div
          className="sticky top-0 z-40 rounded-b-3xl"
          style={{
            background: 'rgba(26, 32, 44, 0.8)',
            backdropFilter: 'blur(20px)',
            WebkitBackdropFilter: 'blur(20px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderTop: 'none',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
          }}
        >
          <div className="container mx-auto px-4 py-4">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">

              {/* Left Section - Stock Info */}
              <div className="flex items-center space-x-6">
                <button
                  onClick={() => navigate('/')}
                  className="flex items-center space-x-2 px-4 py-2 rounded-2xl transition-all duration-300 transform hover:scale-105 active:scale-95"
                  style={{
                    background: 'rgba(255, 255, 255, 0.05)',
                    backdropFilter: 'blur(10px)',
                    WebkitBackdropFilter: 'blur(10px)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    color: '#00ff99'
                  }}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                  <span className="text-sm font-medium">Back to Search</span>
                </button>

                <div className="flex items-center space-x-4">
                  <div>
                    <h1 className="text-2xl lg:text-3xl font-bold text-white">
                      {ticker}
                    </h1>
                    <p className="text-sm text-gray-400">
                      Price Prediction - {selectedDays} Days
                    </p>
                  </div>

                  {predictionData?.currentPrice && (
                    <div className="text-right">
                      <div className="text-xl lg:text-2xl font-bold text-white">
                        ${predictionData.currentPrice.toFixed(2)}
                      </div>
                      <div className={`text-sm font-medium ${
                        predictionData.avgReturn >= 0 ? 'text-trading-green' : 'text-trading-red'
                      }`}>
                        {predictionData.avgReturn >= 0 ? '+' : ''}{predictionData.avgReturn.toFixed(2)}%
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Right Section - Controls */}
              <div className="flex items-center space-x-4">
                <ModeSelector ticker={ticker} />

                <button
                  onClick={handleReset}
                  className="px-4 py-2 rounded-2xl transition-all duration-300 transform hover:scale-105 active:scale-95"
                  style={{
                    background: 'rgba(11, 14, 20, 0.6)',
                    backdropFilter: 'blur(10px)',
                    WebkitBackdropFilter: 'blur(10px)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    color: '#00ff99'
                  }}
                >
                  <span className="text-sm font-medium">New Prediction</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : (
        // TradeLens-style header (before prediction)
        <div className="py-8 px-4 text-center animate-[fadeInUp_0.6s_ease-out]">
          <h1 className="text-5xl md:text-6xl font-bold text-white mb-4">
            <span
              style={{
                background: 'linear-gradient(135deg, #00ff99 0%, #00e5ff 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text'
              } as React.CSSProperties}
            >
              Trade
            </span>
            <span>Lens</span>
          </h1>
          <p className="text-xl text-gray-400 mb-8">
            Easy market analysis for smarter decisions
          </p>
          <ModeSelector />
        </div>
      )}

      <div className="container mx-auto px-4 py-6">
          {!isLoading && !allPredictions && (
            <div className="max-w-2xl mx-auto space-y-12">
            <>
              <div
                className="p-8 rounded-3xl animate-[fadeInUp_0.8s_ease-out_0.2s] animate-fill-both"
                style={{
                  background: 'rgba(26, 32, 44, 0.6)',
                  backdropFilter: 'blur(20px)',
                  WebkitBackdropFilter: 'blur(20px)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1)'
                }}
              >
                <div className="mb-6">
                  <label className="block text-sm font-medium text-gray-300 mb-3">
                    Enter Stock Symbol
                  </label>
                  <input
                    ref={inputRef}
                    type="text"
                    value={ticker}
                    onChange={(e) => setTicker(e.target.value.toUpperCase())}
                    onKeyPress={handleKeyPress}
                    placeholder="e.g., AAPL, MSFT, GOOGL..."
                    maxLength={10}
                    className="w-full px-5 py-4 text-white text-lg font-medium placeholder-gray-500 transition-all duration-300 outline-none rounded-2xl"
                    style={{
                      background: 'rgba(11, 14, 20, 0.6)',
                      backdropFilter: 'blur(10px)',
                      WebkitBackdropFilter: 'blur(10px)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      boxShadow: '0 4px 16px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05)'
                    }}
                    onFocus={(e) => {
                      e.target.style.borderColor = 'rgba(0, 255, 153, 0.5)';
                      e.target.style.boxShadow = '0 8px 32px rgba(0, 255, 153, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1)';
                    }}
                    onBlur={(e) => {
                      e.target.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                      e.target.style.boxShadow = '0 4px 16px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05)';
                    }}
                  />
                </div>

                <button
                  onClick={() => handlePredict()}
                  disabled={!ticker.trim()}
                  className="w-full font-bold py-4 px-6 text-lg transition-all duration-300 rounded-2xl transform active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed hover:scale-[1.02]"
                  style={{
                    background: 'linear-gradient(135deg, #00ff99 0%, #00e5ff 100%)',
                    color: '#000000',
                    boxShadow: '0 8px 32px rgba(0, 255, 153, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.3)'
                  }}
                >
                  Generate Price Forecast
                </button>
              </div>

              {/* Popular Stocks */}
              <div className="space-y-4 animate-[fadeInUp_0.8s_ease-out_0.4s] animate-fill-both">
                <h3 className="text-lg font-semibold text-gray-300 text-center">Popular Stocks</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {POPULAR_STOCKS.map((stock, index) => (
                    <button
                      key={stock.ticker}
                      onClick={() => handlePopularStockClick(stock.ticker)}
                      className="p-4 transition-all duration-300 rounded-2xl transform hover:scale-105 hover:-translate-y-1 active:scale-95"
                      style={{
                        background: 'rgba(26, 32, 44, 0.6)',
                        backdropFilter: 'blur(20px)',
                        WebkitBackdropFilter: 'blur(20px)',
                        border: '1px solid rgba(255, 255, 255, 0.1)',
                        boxShadow: '0 4px 16px rgba(0, 0, 0, 0.2)',
                        animationDelay: `${index * 0.05}s`
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.borderColor = 'rgba(0, 255, 153, 0.5)';
                        e.currentTarget.style.boxShadow = '0 8px 32px rgba(0, 255, 153, 0.3)';
                        const tickerEl = e.currentTarget.querySelector('.ticker');
                        if (tickerEl) {
                          tickerEl.setAttribute('style', 'background: linear-gradient(135deg, #00ff99 0%, #00e5ff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;');
                        }
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                        e.currentTarget.style.boxShadow = '0 4px 16px rgba(0, 0, 0, 0.2)';
                        const tickerEl = e.currentTarget.querySelector('.ticker');
                        if (tickerEl) {
                          tickerEl.setAttribute('style', 'color: white');
                        }
                      }}
                    >
                      <div className="text-center">
                        <div className="ticker font-bold text-white transition-all duration-300">
                          {stock.ticker}
                        </div>
                        <div className="text-xs text-gray-400 mt-1 truncate">
                          {stock.name}
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </>
            </div>
          )}

          {isLoading && (
            <div className="max-w-md mx-auto">
            <div className="text-center py-12 animate-[fadeIn_0.3s_ease-out]">
              <div
                className="inline-block p-12 rounded-3xl max-w-md"
                style={{
                  background: 'rgba(26, 32, 44, 0.6)',
                  backdropFilter: 'blur(20px)',
                  WebkitBackdropFilter: 'blur(20px)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
                }}
              >
                <h2 className="text-2xl font-semibold text-white mb-8">
                  Processing Forecast
                </h2>
                <div className="w-full">
                  <div className="w-full h-2 mb-6 rounded-full overflow-hidden" style={{ background: 'rgba(255, 255, 255, 0.1)' }}>
                    <div
                      className="h-full transition-all duration-500 rounded-full"
                      style={{
                        width: `${loadingProgress}%`,
                        background: 'linear-gradient(90deg, #00ff99, #00e5ff)',
                        boxShadow: '0 0 20px rgba(0, 255, 153, 0.6)'
                      }}
                    ></div>
                  </div>
                  <div className="text-gray-300 text-base">{loadingStage}</div>
                  <div className="text-gray-500 text-sm mt-2">{loadingProgress}%</div>
                </div>
              </div>
            </div>
            </div>
          )}

          {allPredictions && predictionData && (
            <div className="animate-[fadeInUp_0.6s_ease-out]">
              {/* Time Period Tabs */}
              <div className="mb-6 flex justify-center">
                <div
                  className="inline-flex gap-3 p-2 rounded-2xl"
                  style={{
                    background: 'rgba(26, 32, 44, 0.6)',
                    backdropFilter: 'blur(20px)',
                    WebkitBackdropFilter: 'blur(20px)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
                  }}
                >
                  {[5, 15, 30].map((days) => (
                    <button
                      key={days}
                      onClick={() => setSelectedDays(days)}
                      className={`px-6 py-3 rounded-xl text-base font-semibold transition-all duration-500 transform hover:scale-105 active:scale-95 ${
                        selectedDays === days ? 'text-black' : 'text-white'
                      }`}
                      style={
                        selectedDays === days
                          ? {
                              background: 'linear-gradient(135deg, #00ff99 0%, #00e5ff 100%)',
                              boxShadow: '0 8px 32px rgba(0, 255, 153, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.2) inset',
                            }
                          : {
                              background: 'rgba(255, 255, 255, 0.05)',
                              backdropFilter: 'blur(10px)',
                              WebkitBackdropFilter: 'blur(10px)',
                              border: '1px solid rgba(255, 255, 255, 0.1)',
                            }
                      }
                    >
                      {days} Days
                    </button>
                  ))}
                </div>
              </div>

              {/* Price Forecast Chart with Key Levels & Risk */}
              <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 mb-6">
                {/* Price Forecast Chart (2/3 width) */}
                <div className="xl:col-span-2">
                  <div
                    className="p-6 rounded-3xl"
                    style={{
                      background: 'rgba(26, 32, 44, 0.6)',
                      backdropFilter: 'blur(20px)',
                      WebkitBackdropFilter: 'blur(20px)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
                    }}
                  >
                    <div className="flex items-center justify-between mb-4 pb-2" style={{
                      borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
                    }}>
                      <h3
                        className="text-xl font-semibold"
                        style={{
                          background: 'linear-gradient(135deg, #00ff99 0%, #00e5ff 100%)',
                          WebkitBackgroundClip: 'text',
                          WebkitTextFillColor: 'transparent',
                          backgroundClip: 'text'
                        } as React.CSSProperties}
                      >
                        📈 Price Forecast
                      </h3>
                      <span className="text-sm text-gray-400">{selectedDays}-Day Prediction</span>
                    </div>
                    <PredictionChart
                      historicalData={historicalData}
                      predictions={predictionData.days.map(day => ({
                        date: day.date, // Already in ISO format
                        price: day.price,
                        upperBound: day.price * 1.05,
                        lowerBound: day.price * 0.95,
                        changePct: day.changePct || 0,
                      }))}
                      currentPrice={predictionData.currentPrice || 100}
                      height={400}
                    />
                  </div>
                </div>

                {/* Right Column - Key Levels & Risk (1/3 width) */}
                <div className="space-y-6">
                  {/* Key Price Levels */}
                  <div
                    className="p-6 rounded-3xl"
                    style={{
                      background: 'rgba(26, 32, 44, 0.6)',
                      backdropFilter: 'blur(20px)',
                      WebkitBackdropFilter: 'blur(20px)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
                    }}
                  >
                    <h3
                      className="text-xl font-semibold mb-4 pb-2"
                      style={{
                        background: 'linear-gradient(135deg, #00ff99 0%, #00e5ff 100%)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                        backgroundClip: 'text',
                        borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
                      } as React.CSSProperties}
                    >
                      Key Price Levels
                    </h3>
                    <div className="space-y-4 text-gray-300">
                      <div className="flex justify-between items-center">
                        <span className="text-gray-400">Target Price</span>
                        <span className="text-xl font-bold" style={{ color: '#00ff99' }}>${predictionData.targetPrice}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-gray-400">Resistance</span>
                        <span className="text-lg font-semibold text-white">${predictionData.resistanceLevel}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-gray-400">Support</span>
                        <span className="text-lg font-semibold text-white">${predictionData.supportLevel}</span>
                      </div>
                    </div>
                  </div>

                  {/* Risk Assessment */}
                  <div
                    className="p-6 rounded-3xl"
                    style={{
                      background: 'rgba(26, 32, 44, 0.6)',
                      backdropFilter: 'blur(20px)',
                      WebkitBackdropFilter: 'blur(20px)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
                    }}
                  >
                    <h3
                      className="text-xl font-semibold mb-4 pb-2"
                      style={{
                        background: 'linear-gradient(135deg, #00ff99 0%, #00e5ff 100%)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                        backgroundClip: 'text',
                        borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
                      } as React.CSSProperties}
                    >
                      Risk Assessment
                    </h3>
                    <div className="space-y-4 text-gray-300">
                      <div>
                        <div className="text-gray-400 text-sm mb-1">Market Outlook</div>
                        <div className="text-lg font-semibold text-white">{predictionData.outlook}</div>
                      </div>
                      <div>
                        <div className="text-gray-400 text-sm mb-1">Risk Level</div>
                        <div className="text-lg font-semibold" style={{ color: predictionData.riskLevel.includes('High') ? '#ff4757' : predictionData.riskLevel.includes('Moderate') ? '#ffa502' : '#00ff99' }}>
                          {predictionData.riskLevel}
                        </div>
                      </div>
                      <div>
                        <div className="text-gray-400 text-sm mb-1">Position Size</div>
                        <div className="text-lg font-semibold text-white">{predictionData.positionSize}</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Price Forecast Table */}
              <div
                className="p-6 mb-6 overflow-x-auto rounded-3xl"
                style={{
                  background: 'rgba(26, 32, 44, 0.6)',
                  backdropFilter: 'blur(20px)',
                  WebkitBackdropFilter: 'blur(20px)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
                }}
              >
                <h3
                  className="text-xl font-semibold mb-4 pb-2"
                  style={{
                    background: 'linear-gradient(135deg, #00ff99 0%, #00e5ff 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    backgroundClip: 'text',
                    borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
                  } as React.CSSProperties}
                >
                  Price Forecast
                </h3>
                <table className="w-full">
                  <thead>
                    <tr style={{ background: 'rgba(11, 14, 20, 0.6)' }}>
                      <th className="px-4 py-3 text-left font-semibold" style={{ color: '#00ff99' }}>Date</th>
                      <th className="px-4 py-3 text-left font-semibold" style={{ color: '#00ff99' }}>Predicted Price</th>
                      <th className="px-4 py-3 text-left font-semibold" style={{ color: '#00ff99' }}>Change</th>
                      <th className="px-4 py-3 text-left font-semibold" style={{ color: '#00ff99' }}>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {predictionData.days.map((day, index) => (
                      <tr key={index} className="border-b" style={{ borderColor: 'rgba(255, 255, 255, 0.05)' }}>
                        <td className="px-4 py-3 text-gray-300">{day.displayDate || day.date}</td>
                        <td className="px-4 py-3 text-gray-300 font-semibold">${day.price.toFixed(2)}</td>
                        <td className="px-4 py-3 font-semibold" style={{ color: day.change >= 0 ? '#00ff99' : '#ff4757' }}>
                          {day.change >= 0 ? '+' : ''}{day.change.toFixed(2)}%
                        </td>
                        <td className="px-4 py-3 text-gray-400">{day.confidence.toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* AI Reasoning Section */}
              {predictionData.aiReasoning && (
                <div
                  className="p-6 mb-6 rounded-3xl"
                  style={{
                    background: 'linear-gradient(135deg, rgba(138, 43, 226, 0.1) 0%, rgba(75, 0, 130, 0.1) 100%)',
                    backdropFilter: 'blur(20px)',
                    WebkitBackdropFilter: 'blur(20px)',
                    border: '1px solid rgba(138, 43, 226, 0.3)',
                    boxShadow: '0 8px 32px rgba(138, 43, 226, 0.2)'
                  }}
                >
                  <div className="flex items-center mb-4">
                    <div className="text-2xl mr-3">🤖</div>
                    <h3
                      className="text-xl font-semibold"
                      style={{
                        background: 'linear-gradient(135deg, #ba55d3 0%, #9370db 100%)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                        backgroundClip: 'text'
                      } as React.CSSProperties}
                    >
                      AI Reasoning
                    </h3>
                  </div>

                  {/* Key Insights */}
                  {predictionData.aiReasoning.key_insights && predictionData.aiReasoning.key_insights.length > 0 && (
                    <div className="mb-4">
                      <div className="text-sm font-semibold text-purple-300 mb-2">Key Insights</div>
                      <ul className="space-y-2">
                        {predictionData.aiReasoning.key_insights.map((insight, idx) => (
                          <li key={idx} className="flex items-start text-gray-300">
                            <span className="text-purple-400 mr-2">•</span>
                            <span>{insight}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Detailed Reasoning */}
                  {predictionData.aiReasoning.reasoning && (
                    <div className="mb-4">
                      <div className="text-sm font-semibold text-purple-300 mb-2">Detailed Analysis</div>
                      <div className="text-gray-300 text-sm whitespace-pre-line leading-relaxed">
                        {predictionData.aiReasoning.reasoning}
                      </div>
                    </div>
                  )}

                  {/* Recommendation */}
                  {predictionData.aiReasoning.recommendation && (
                    <div className="mt-4 p-4 rounded-xl" style={{
                      background: 'rgba(138, 43, 226, 0.1)',
                      border: '1px solid rgba(138, 43, 226, 0.2)'
                    }}>
                      <div className="text-sm font-semibold text-purple-300 mb-1">Recommendation</div>
                      <div className="text-gray-200 font-medium">{predictionData.aiReasoning.recommendation}</div>
                    </div>
                  )}
                </div>
              )}

              {/* News Analysis Section */}
              {predictionData.newsAnalysis && predictionData.newsAnalysis.total_articles > 0 && (
                <div
                  className="p-6 mb-6 rounded-3xl"
                  style={{
                    background: 'rgba(26, 32, 44, 0.6)',
                    backdropFilter: 'blur(20px)',
                    WebkitBackdropFilter: 'blur(20px)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
                  }}
                >
                  <h3
                    className="text-xl font-semibold mb-4 pb-2"
                    style={{
                      background: 'linear-gradient(135deg, #00ff99 0%, #00e5ff 100%)',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      backgroundClip: 'text',
                      borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
                    } as React.CSSProperties}
                  >
                    📰 News Sentiment Analysis
                  </h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-gray-400 text-sm mb-1">Articles Analyzed</div>
                      <div className="text-2xl font-bold text-white">{predictionData.newsAnalysis.total_articles}</div>
                    </div>
                    <div>
                      <div className="text-gray-400 text-sm mb-1">Market Sentiment</div>
                      <div className="text-2xl font-bold" style={{
                        color: predictionData.newsAnalysis.sentiment_label === 'Bullish' ? '#00ff99' :
                               predictionData.newsAnalysis.sentiment_label === 'Bearish' ? '#ff4757' : '#ffa502'
                      }}>
                        {predictionData.newsAnalysis.sentiment_label}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Disclaimer */}
              <div
                className="p-4 text-center text-sm text-gray-400 rounded-2xl mt-6"
                style={{
                  background: 'rgba(255, 255, 255, 0.02)',
                  border: '1px solid rgba(255, 255, 255, 0.05)'
                }}
              >
                Predictions are based on AI analysis and should not be considered financial advice. Always conduct thorough research before making investment decisions.
              </div>
            </div>
          )}
      </div>
    </div>
  );
};

export default PredictPage;
