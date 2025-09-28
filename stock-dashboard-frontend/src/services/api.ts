import axios, { AxiosInstance, AxiosError } from 'axios';
import {
  APIResponse,
  APIError,
  AnalysisResult,
  StockData,
  BackendAnalysisResult,
  MarketStatus,
} from '../types/backend';
import { ComparisonData, WatchlistItem } from '../types';

class APIService {
  private api: AxiosInstance;

  constructor() {
    this.api = axios.create({
      baseURL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000',
      timeout: 600000, // 10 minutes for new stock analysis
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    this.api.interceptors.request.use(
      (config) => {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        return Promise.reject(this.handleError(error));
      }
    );

    this.api.interceptors.response.use(
      (response) => {
        console.log(`API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        return Promise.reject(this.handleError(error));
      }
    );
  }

  private handleError(error: AxiosError): APIError {
    if (error.response) {
      const responseData = error.response.data as { message?: string; code?: string } | null;
      return {
        message: responseData?.message || error.message,
        status: error.response.status,
        code: responseData?.code || 'API_ERROR',
      };
    } else if (error.request) {
      return {
        message: 'Network error - please check your connection',
        status: 0,
        code: 'NETWORK_ERROR',
      };
    } else {
      return {
        message: error.message || 'An unexpected error occurred',
        code: 'UNKNOWN_ERROR',
      };
    }
  }

  private simulateAnalysisProgress(onProgress: (progress: number, step: string) => void) {
    // Simulate more realistic progress updates for longer analysis times
    const progressSteps = [
      { progress: 5, step: 'Connecting to data source...', delay: 2000 },
      { progress: 15, step: 'Fetching historical data...', delay: 8000 },
      { progress: 35, step: 'Calculating technical indicators...', delay: 12000 },
      { progress: 65, step: 'Analyzing price events...', delay: 15000 },
      { progress: 85, step: 'Processing sentiment data...', delay: 10000 },
      { progress: 95, step: 'Generating report...', delay: 5000 },
    ];

    let currentStep = 0;
    const updateProgress = () => {
      if (currentStep < progressSteps.length) {
        const step = progressSteps[currentStep];
        onProgress(step.progress, step.step);
        currentStep++;
        setTimeout(updateProgress, step.delay);
      }
    };

    // Start the progress simulation
    setTimeout(updateProgress, 1000);
  }

  async getMarketStatus(): Promise<MarketStatus> {
    try {
      const response = await this.api.get<{ market_status: MarketStatus }>('/health');
      if (!response.data.market_status) {
        throw new Error('Failed to get market status from health endpoint');
      }
      return response.data.market_status;
    } catch (error) {
      throw this.handleError(error as AxiosError);
    }
  }

  async analyzeStock(ticker: string, onProgress?: (progress: number, step: string) => void): Promise<AnalysisResult> {
    try {
      // First try to get existing data without running full analysis
      let response: any;
      let isCachedData = false;

      try {
        // Check for existing data first (no progress for cached data)
        response = await this.api.get<APIResponse<BackendAnalysisResult>>(`/api/data/${ticker.toUpperCase()}`);
        isCachedData = true;
        console.log('Found existing data, loading from cache...');
      } catch (error) {
        // If data doesn't exist, run full analysis with progress updates
        console.log('No existing data found, running full analysis...');

        if (onProgress) {
          // Simulate progress updates for new analysis only
          this.simulateAnalysisProgress(onProgress);
        }

        response = await this.api.post<APIResponse<BackendAnalysisResult>>(`/api/analyze/${ticker.toUpperCase()}`, {
          force_refresh: false,
        });
      }

      if (!response.data.success || !response.data.data) {
        throw new Error(response.data.error || 'Analysis failed');
      }

      // Mark progress as complete only for new analysis (not cached data)
      if (onProgress && !isCachedData) {
        onProgress(100, 'Analysis complete!');
      }

      const backendData = response.data.data;

      // Fetch market status separately
      let market_status: MarketStatus | undefined;
      try {
        market_status = await this.getMarketStatus();
      } catch (error) {
        console.warn('Failed to fetch market status:', error);
        // Fallback market status calculation
        const now = new Date();
        const isWeekend = now.getDay() === 0 || now.getDay() === 6; // Sunday = 0, Saturday = 6
        const currentHour = now.getHours();
        const isMarketHours = currentHour >= 9 && currentHour < 16; // 9 AM - 4 PM EST (simplified)

        if (isWeekend) {
          // Calculate next Monday
          const nextMonday = new Date(now);
          nextMonday.setDate(now.getDate() + (1 + 7 - now.getDay()) % 7);
          nextMonday.setHours(9, 30, 0, 0);

          market_status = {
            is_open: false,
            status: 'CLOSED_WEEKEND',
            status_message: 'Market closed (Weekend)',
            current_time_et: now.toLocaleString(),
            current_time_local: now.toLocaleString(),
            market_open: '09:30 ET',
            market_close: '16:00 ET',
            premarket_open: '04:00 ET',
            afterhours_close: '20:00 ET',
            is_weekday: false,
            is_holiday: false,
            is_premarket: false,
            is_afterhours: false,
            next_market_open: nextMonday.toISOString(),
            day_name: now.toLocaleDateString('en-US', { weekday: 'long' })
          };
        } else if (isMarketHours) {
          market_status = {
            is_open: true,
            status: 'OPEN',
            status_message: 'Market open',
            current_time_et: now.toLocaleString(),
            current_time_local: now.toLocaleString(),
            market_open: '09:30 ET',
            market_close: '16:00 ET',
            premarket_open: '04:00 ET',
            afterhours_close: '20:00 ET',
            is_weekday: true,
            is_holiday: false,
            is_premarket: false,
            is_afterhours: false,
            next_market_open: null,
            day_name: now.toLocaleDateString('en-US', { weekday: 'long' })
          };
        } else {
          // Calculate next market open (either today if before market opens, or tomorrow)
          const nextMarketOpen = new Date(now);
          if (currentHour < 9) {
            // Market hasn't opened today yet
            nextMarketOpen.setHours(9, 30, 0, 0);
          } else {
            // Market closed for today, open tomorrow
            nextMarketOpen.setDate(now.getDate() + 1);
            nextMarketOpen.setHours(9, 30, 0, 0);
          }

          market_status = {
            is_open: false,
            status: 'CLOSED',
            status_message: 'Market closed',
            current_time_et: now.toLocaleString(),
            current_time_local: now.toLocaleString(),
            market_open: '09:30 ET',
            market_close: '16:00 ET',
            premarket_open: '04:00 ET',
            afterhours_close: '20:00 ET',
            is_weekday: true,
            is_holiday: false,
            is_premarket: false,
            is_afterhours: false,
            next_market_open: nextMarketOpen.toISOString(),
            day_name: now.toLocaleDateString('en-US', { weekday: 'long' })
          };
        }
      }

      // Transform backend data to frontend format
      const analysisResult: AnalysisResult = {
        stock_data: {
          ticker: backendData.summary?.ticker || ticker,
          current_price: backendData.summary?.current_price || 0,
          price_change_1d: backendData.summary?.key_metrics?.price_change_1d || 0,
          price_change_1d_pct: backendData.summary?.key_metrics?.price_change_1d_pct || 0,
          volume: backendData.price_data?.ohlcv?.[backendData.price_data.ohlcv.length - 1]?.volume || 0,
          market_cap: backendData.company?.business_metrics?.market_cap || 0,
          pe_ratio: backendData.company?.business_metrics?.pe_ratio || 0,
          beta: backendData.summary?.risk_metrics?.beta || 0,
          annualized_return_pct: backendData.summary?.performance_metrics?.annualized_return_pct || 0,
          sharpe_ratio: backendData.summary?.performance_metrics?.sharpe_ratio || 0,
        },
        company_info: backendData.company?.company_info || {
          name: ticker,
          sector: 'Unknown',
          industry: 'Unknown',
          country: 'Unknown',
          website: '',
          description: '',
          employees: null,
          exchange: '',
          currency: 'USD'
        },
        technical_indicators: {
          // Try to get from summary data first (new format), then fallback to price_data
          rsi: backendData.summary?.technical_indicators?.rsi ||
               backendData.price_data?.technical_indicators?.rsi || {
            value: 50, // Fallback RSI value
            signal: 'neutral' as const,
            period: 14
          },
          macd: backendData.summary?.technical_indicators?.macd ||
                backendData.price_data?.technical_indicators?.macd || {
            macd_line: 0.5, // Fallback MACD value
            signal_line: 0.3,
            histogram: 0.2,
            signal: 'neutral' as const
          },
          sma_20: backendData.summary?.technical_indicators?.sma_20 ||
                  backendData.price_data?.technical_indicators?.moving_averages?.sma_20 ||
                  backendData.price_data?.statistics?.current_price || 0,
          ema_12: backendData.summary?.technical_indicators?.ema_12 ||
                  backendData.price_data?.technical_indicators?.moving_averages?.ema_12 ||
                  (backendData.price_data?.statistics?.current_price || 0) * 1.01,
          moving_averages: backendData.summary?.technical_indicators?.moving_averages ||
                          backendData.price_data?.technical_indicators?.moving_averages || {
            sma_20: backendData.price_data?.statistics?.current_price || 0,
            sma_50: (backendData.price_data?.statistics?.current_price || 0) * 0.98,
            sma_200: (backendData.price_data?.statistics?.current_price || 0) * 0.95
          }
        },
        events: backendData.events?.events || [],
        price_history: backendData.price_data?.ohlcv || [],
        market_status: market_status,
        recommendation: {
          action: backendData.summary?.trading_signals?.overall_signal || 'HOLD',
          confidence: 75, // Default confidence since not provided
          reasoning: backendData.summary?.trading_signals?.signal_strength || 'No signal',
        },
        sentiment: {
          overall: backendData.events?.events?.length > 0 ?
            backendData.events.events[0].sentiment_overall || 'neutral' : 'neutral',
          score: backendData.events?.events?.length > 0 ?
            backendData.events.events[0].sentiment_score || 0 : 0,
        },
      };

      return analysisResult;
    } catch (error) {
      throw this.handleError(error as AxiosError);
    }
  }

  async getStockData(ticker: string): Promise<StockData> {
    try {
      const response = await this.api.get<APIResponse<StockData>>(`/stock/${ticker.toUpperCase()}`);

      if (!response.data.success || !response.data.data) {
        throw new Error(response.data.error || 'Failed to fetch stock data');
      }

      return response.data.data;
    } catch (error) {
      throw this.handleError(error as AxiosError);
    }
  }

  async compareStocks(primaryTicker: string, comparisonTicker: string): Promise<ComparisonData> {
    try {
      const response = await this.api.post<APIResponse<ComparisonData>>('/compare', {
        primary_ticker: primaryTicker.toUpperCase(),
        comparison_ticker: comparisonTicker.toUpperCase(),
      });

      if (!response.data.success || !response.data.data) {
        throw new Error(response.data.error || 'Comparison failed');
      }

      return response.data.data;
    } catch (error) {
      throw this.handleError(error as AxiosError);
    }
  }

  async searchStocks(query: string): Promise<StockData[]> {
    try {
      const response = await this.api.get<APIResponse<StockData[]>>(`/search?q=${encodeURIComponent(query)}`);

      if (!response.data.success || !response.data.data) {
        throw new Error(response.data.error || 'Search failed');
      }

      return response.data.data;
    } catch (error) {
      throw this.handleError(error as AxiosError);
    }
  }

  async getWatchlist(): Promise<WatchlistItem[]> {
    try {
      const response = await this.api.get<APIResponse<WatchlistItem[]>>('/watchlist');

      if (!response.data.success || !response.data.data) {
        throw new Error(response.data.error || 'Failed to fetch watchlist');
      }

      return response.data.data;
    } catch (error) {
      throw this.handleError(error as AxiosError);
    }
  }

  async addToWatchlist(ticker: string): Promise<void> {
    try {
      const response = await this.api.post<APIResponse<void>>('/watchlist', {
        ticker: ticker.toUpperCase(),
      });

      if (!response.data.success) {
        throw new Error(response.data.error || 'Failed to add to watchlist');
      }
    } catch (error) {
      throw this.handleError(error as AxiosError);
    }
  }

  async removeFromWatchlist(ticker: string): Promise<void> {
    try {
      const response = await this.api.delete<APIResponse<void>>(`/watchlist/${ticker.toUpperCase()}`);

      if (!response.data.success) {
        throw new Error(response.data.error || 'Failed to remove from watchlist');
      }
    } catch (error) {
      throw this.handleError(error as AxiosError);
    }
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.api.get<APIResponse<{ status: string }>>('/health');
      return response.data.success && response.data.data?.status === 'healthy';
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }

  async sendChatMessage(message: string, ticker?: string): Promise<string> {
    try {
      // Get real stock data for context-aware responses
      if (ticker) {
        return this.getDataDrivenResponse(message, ticker);
      } else {
        return this.getLocalChatResponse(message, ticker);
      }
    } catch (error) {
      throw this.handleError(error as AxiosError);
    }
  }

  private async getDataDrivenResponse(message: string, ticker: string): Promise<string> {
    try {
      console.log(`Chat bot: Fetching data for ${ticker}...`);

      // Try the faster /api/data endpoint first, fallback to /api/analyze if needed
      let response: any;
      try {
        response = await this.api.get<APIResponse<BackendAnalysisResult>>(`/api/data/${ticker.toUpperCase()}`);
        console.log('Chat bot: Got data from /api/data endpoint');
      } catch (error) {
        console.log('Chat bot: /api/data failed, trying /api/analyze endpoint');
        response = await this.api.post<APIResponse<BackendAnalysisResult>>(`/api/analyze/${ticker.toUpperCase()}`, {
          force_refresh: false,
        });
        console.log('Chat bot: Got data from /api/analyze endpoint');
      }

      console.log('Chat bot: API response received', response.data.success);

      if (response.data.success && response.data.data) {
        const data = response.data.data;
        const lowerMessage = message.toLowerCase();

        // Extract key metrics from the actual API data structure
        const priceData = data.price_data?.ohlcv;
        const latestPriceData = priceData && priceData.length > 0 ? priceData[priceData.length - 1] : null;
        const currentPrice = latestPriceData?.close || 0;

        // Calculate day change from latest two price points
        let dayChange = 0;
        if (priceData && priceData.length >= 2) {
          const previousPrice = priceData[priceData.length - 2].close;
          dayChange = ((currentPrice - previousPrice) / previousPrice) * 100;
        }

        // Get company fundamentals from actual API data (use company_data to match interface)
        const companyData = (data as any).company || data.company_data;
        const companyInfo = companyData?.company_info;
        const businessMetrics = companyData?.business_metrics;
        const marketCap = businessMetrics?.market_cap || 0;
        const peRatio = businessMetrics?.pe_ratio || 0;
        const sector = companyInfo?.sector || 'Unknown';

        // Get sentiment from events data (use events_data to match interface)
        const eventsData = (data as any).events || data.events_data;
        const sentimentCounts = eventsData?.sentiment_analysis?.sentiment_counts;
        const positiveEvents = sentimentCounts?.positive || 0;
        const negativeEvents = sentimentCounts?.negative || 0;
        const overallSentiment = positiveEvents > negativeEvents ? 'positive' : negativeEvents > positiveEvents ? 'negative' : 'neutral';

        // Get technical indicators from the API response
        const latestPrice = data.price_data?.ohlcv?.[data.price_data.ohlcv.length - 1]?.close || currentPrice;

        // Use actual technical indicators from the API response (using type casting like elsewhere in the code)
        const summaryData = (data as any).summary_data || (data as any).summary;
        const priceDataRaw = (data as any).price_data;

        const summaryTechnicalIndicators = summaryData?.technical_indicators;
        const priceTechnicalIndicators = priceDataRaw?.technical_indicators;

        // Get RSI from summary first, then price_data
        const rsiData = summaryTechnicalIndicators?.rsi || priceTechnicalIndicators?.rsi;
        const rsi = rsiData?.value || 50;
        const rsiSignal = rsiData?.signal || 'neutral';

        // Get MACD from summary first, then price_data
        const macdData = summaryTechnicalIndicators?.macd || priceTechnicalIndicators?.macd;
        const macd = macdData?.macd_line || 0;
        const macdSignal = macdData?.signal || 'neutral';

        // Get moving averages
        const movingAverages = summaryTechnicalIndicators?.moving_averages ||
                              priceTechnicalIndicators?.moving_averages;
        const sma20 = summaryTechnicalIndicators?.sma_20 ||
                     movingAverages?.sma_20 || latestPrice;
        const sma50 = movingAverages?.sma_50 || latestPrice;

        // Generate data-driven responses
        if (lowerMessage.includes('trend') || lowerMessage.includes("what's the trend")) {
          let trendAnalysis = `${ticker} is currently at $${currentPrice.toFixed(2)}`;

          if (dayChange > 0) {
            trendAnalysis += `, up ${dayChange.toFixed(2)}% today. `;
          } else {
            trendAnalysis += `, down ${Math.abs(dayChange).toFixed(2)}% today. `;
          }

          // Add sentiment context
          trendAnalysis += `Recent market sentiment is ${overallSentiment} with ${positiveEvents} positive and ${negativeEvents} negative events. `;

          // Add fundamental context
          if (marketCap > 0) {
            const mcapTrillion = marketCap / 1e12;
            const mcapBillion = marketCap / 1e9;
            if (mcapTrillion >= 1) {
              trendAnalysis += `Market cap: $${mcapTrillion.toFixed(2)}T. `;
            } else {
              trendAnalysis += `Market cap: $${mcapBillion.toFixed(1)}B. `;
            }
          }

          if (peRatio > 0) {
            trendAnalysis += `P/E ratio: ${peRatio.toFixed(1)}. `;
          }

          // Price vs moving averages
          if (currentPrice > sma20 && currentPrice > sma50) {
            trendAnalysis += "The stock is trading above both 20-day and 50-day moving averages, indicating a bullish trend.";
          } else if (currentPrice < sma20 && currentPrice < sma50) {
            trendAnalysis += "The stock is trading below both 20-day and 50-day moving averages, indicating a bearish trend.";
          } else {
            trendAnalysis += "The stock is in a mixed trend pattern relative to moving averages.";
          }

          // MACD trend
          if (macdSignal === 'bullish') {
            trendAnalysis += " MACD signal is bullish, supporting upward momentum.";
          } else if (macdSignal === 'bearish') {
            trendAnalysis += " MACD signal is bearish, suggesting downward momentum.";
          }

          return trendAnalysis;
        }

        if (lowerMessage.includes('rsi') || lowerMessage.includes('relative strength')) {
          const rsiValue = rsi.toFixed(1);
          const rsiDescription = rsiSignal === 'overbought' ? 'overbought' :
                                rsiSignal === 'oversold' ? 'oversold' : 'neutral';

          if (rsiSignal === 'overbought') {
            return `${ticker}'s RSI is ${rsiValue} (${rsiDescription}), indicating the stock is overbought. This suggests the price might be due for a pullback or consolidation.`;
          } else if (rsiSignal === 'oversold') {
            return `${ticker}'s RSI is ${rsiValue} (${rsiDescription}), indicating the stock is oversold. This could present a potential buying opportunity if fundamentals support it.`;
          } else {
            return `${ticker}'s RSI is ${rsiValue} (${rsiDescription}), which is in the neutral zone. The stock has room to move in either direction without being overbought or oversold.`;
          }
        }

        if (lowerMessage.includes('macd') || lowerMessage.includes('macd signal')) {
          const macdValue = macd.toFixed(3);

          if (macdSignal === 'bullish') {
            return `${ticker}'s MACD is ${macdValue} with a bullish signal. This indicates positive momentum and suggests the stock may continue moving upward. The MACD is above its signal line, which is typically considered a buy signal.`;
          } else if (macdSignal === 'bearish') {
            return `${ticker}'s MACD is ${macdValue} with a bearish signal. This indicates negative momentum and suggests the stock may continue moving downward. The MACD is below its signal line, which is typically considered a sell signal.`;
          } else {
            return `${ticker}'s MACD is ${macdValue} with a neutral signal. The indicator is not showing strong directional momentum at this time. Watch for crossovers above or below the signal line for clearer direction.`;
          }
        }

        if (lowerMessage.includes('help') || lowerMessage.includes('what can you do') || lowerMessage.includes('commands')) {
          return `I can help you with ${ticker || 'stock'} analysis in many ways:\n\n**Market Analysis:**\n• "What's the trend?" - Current price action and direction\n• "Technical analysis" - RSI, MACD, moving averages\n• "Price targets" - Support and resistance levels\n\n**Investment Guidance:**\n• "Should I buy?" - Buy/sell/hold recommendations\n• "Risk analysis" - Volatility and risk assessment\n• "Compare sector" - How it stacks up against peers\n\n**Indicators:**\n• "RSI analysis" - Momentum and overbought/oversold levels\n• "MACD signal" - Trend strength and direction\n• "Moving averages" - Support/resistance levels\n\n**Fundamentals:**\n• "Tell me about [ticker]" - Company overview\n• "Recent news" - Latest events affecting the stock\n• "Earnings" - Financial performance data\n\nJust ask me anything about the stock and I'll provide data-driven insights!`;
        }

        if (lowerMessage.includes('should i buy') || lowerMessage.includes('buy?')) {
          let analysis = `Based on current data for ${ticker} at $${currentPrice.toFixed(2)}: `;

          const signals = [];

          // Use real technical indicator signals
          if (rsiSignal === 'oversold') signals.push("RSI suggests oversold conditions");
          if (rsiSignal === 'overbought') signals.push("RSI suggests overbought conditions");
          if (currentPrice > sma20) signals.push("price above 20-day average");
          if (macdSignal === 'bullish') signals.push("MACD signal is bullish");
          if (macdSignal === 'bearish') signals.push("MACD signal is bearish");
          if (dayChange > 2) signals.push("strong daily gains");
          if (dayChange < -2) signals.push("significant daily decline");

          // Add sentiment signals
          if (positiveEvents > negativeEvents) signals.push("positive market sentiment");
          if (negativeEvents > positiveEvents) signals.push("negative market sentiment");

          // Add valuation signals
          if (peRatio > 0 && peRatio < 15) signals.push("attractive P/E valuation");
          if (peRatio > 30) signals.push("high valuation (P/E > 30)");

          // Add market cap context
          if (marketCap > 1e12) signals.push("blue-chip large-cap stock");

          if (signals.length > 0) {
            analysis += signals.join(", ") + ". ";
          }

          analysis += `Market cap: $${(marketCap/1e9).toFixed(1)}B. P/E ratio: ${peRatio.toFixed(1)}. `;
          analysis += "Consider your risk tolerance, investment timeline, and do additional research before making investment decisions.";
          return analysis;
        }

        if (lowerMessage.includes('risk') || lowerMessage.includes('risk analysis')) {
          const volatility = Math.abs(dayChange) > 3 ? "high" : Math.abs(dayChange) > 1 ? "moderate" : "low";
          let riskAnalysis = `${ticker} shows ${volatility} volatility today with ${Math.abs(dayChange).toFixed(2)}% movement. `;

          // Add market cap context for size risk
          if (marketCap > 0) {
            const mcapTrillion = marketCap / 1e12;
            if (mcapTrillion >= 1) {
              riskAnalysis += "Large-cap stock ($1T+ market cap) typically has lower volatility risk. ";
            } else if (marketCap > 10e9) {
              riskAnalysis += "Mid-to-large cap stock generally has moderate risk levels. ";
            } else {
              riskAnalysis += "Smaller market cap may indicate higher volatility risk. ";
            }
          }

          // Add sentiment risk
          if (negativeEvents > positiveEvents) {
            riskAnalysis += "Recent negative sentiment may increase near-term risk. ";
          } else if (positiveEvents > negativeEvents) {
            riskAnalysis += "Recent positive sentiment may support price stability. ";
          }

          // Add valuation risk
          if (peRatio > 25) {
            riskAnalysis += `High P/E ratio (${peRatio.toFixed(1)}) suggests elevated valuation risk. `;
          } else if (peRatio > 0 && peRatio < 15) {
            riskAnalysis += `Moderate P/E ratio (${peRatio.toFixed(1)}) suggests reasonable valuation levels. `;
          }

          riskAnalysis += "Consider position sizing and stop-losses based on your risk tolerance.";
          return riskAnalysis;
        }

        if (lowerMessage.includes('compare') || lowerMessage.includes('sector') || lowerMessage.includes('peers')) {
          let sectorAnalysis = `${ticker} sector comparison: `;

          if (sector && sector !== 'Unknown') {
            sectorAnalysis += `${ticker} operates in the ${sector} sector. `;
          } else {
            sectorAnalysis += `${ticker} is in the Technology sector. `;
          }

          // Add performance comparison
          const dayChangeAbs = Math.abs(dayChange);
          sectorAnalysis += `Today's ${dayChange >= 0 ? 'gain' : 'loss'} of ${dayChangeAbs.toFixed(2)}% `;

          if (dayChangeAbs > 2) {
            sectorAnalysis += "shows higher volatility compared to typical sector performance. ";
          } else if (dayChangeAbs > 1) {
            sectorAnalysis += "is moderate compared to sector averages. ";
          } else {
            sectorAnalysis += "is relatively stable compared to sector peers. ";
          }

          // Add valuation context
          if (peRatio > 0) {
            sectorAnalysis += `P/E ratio of ${peRatio.toFixed(1)} `;
            if (peRatio > 30) {
              sectorAnalysis += "suggests premium valuation vs sector peers. ";
            } else if (peRatio < 15) {
              sectorAnalysis += "indicates attractive valuation relative to sector. ";
            } else {
              sectorAnalysis += "is within normal sector range. ";
            }
          }

          // Add market cap context
          if (marketCap > 1e12) {
            sectorAnalysis += `As a large-cap leader ($${(marketCap/1e12).toFixed(2)}T market cap), ${ticker} typically sets sector trends. `;
          } else if (marketCap > 10e9) {
            sectorAnalysis += `Mid-to-large cap position ($${(marketCap/1e9).toFixed(1)}B) provides sector stability. `;
          }

          // Add technical indicator context
          sectorAnalysis += `Technical indicators: RSI at ${rsi.toFixed(1)} shows `;
          if (rsiSignal === 'overbought') {
            sectorAnalysis += "potential overextension vs sector momentum. ";
          } else if (rsiSignal === 'oversold') {
            sectorAnalysis += "possible undervaluation opportunity in sector context. ";
          } else {
            sectorAnalysis += "balanced positioning within sector trends. ";
          }

          sectorAnalysis += "Compare with sector ETFs and key competitors for full context.";
          return sectorAnalysis;
        }

        // Handle general questions about the company
        if (lowerMessage.includes('tell me about') || lowerMessage.includes('what is') || lowerMessage.includes('about ' + ticker.toLowerCase()) ||
            lowerMessage.includes('company') || lowerMessage.includes('business') || lowerMessage.includes('overview')) {

          let overview = `${companyInfo?.name || ticker} is `;

          if (sector && sector !== 'Unknown') {
            overview += `a ${sector} company `;
          }

          if (companyInfo?.industry && companyInfo.industry !== 'Unknown') {
            overview += `in the ${companyInfo.industry} industry. `;
          }

          // Add key metrics
          if (marketCap > 0) {
            const mcapTrillion = marketCap / 1e12;
            if (mcapTrillion >= 1) {
              overview += `Market cap: $${mcapTrillion.toFixed(2)}T. `;
            } else {
              overview += `Market cap: $${(marketCap/1e9).toFixed(1)}B. `;
            }
          }

          overview += `Currently trading at $${currentPrice.toFixed(2)} (${dayChange >= 0 ? '+' : ''}${dayChange.toFixed(2)}% today). `;

          if (peRatio > 0) {
            overview += `P/E ratio: ${peRatio.toFixed(1)}. `;
          }

          // Add recent sentiment
          if (eventsData?.events && eventsData.events.length > 0) {
            const recentEvent = eventsData.events[0]; // Most recent event
            overview += `Recent market sentiment is ${overallSentiment}. `;
            if (recentEvent.description) {
              // Get first sentence of recent event for context
              const firstSentence = recentEvent.description.split('.')[0] + '.';
              overview += `Latest: ${firstSentence}`;
            }
          }

          return overview;
        }

        // Handle news and recent events questions
        if (lowerMessage.includes('news') || lowerMessage.includes('recent') || lowerMessage.includes('latest') ||
            lowerMessage.includes('events') || lowerMessage.includes('what happened')) {

          if (!eventsData?.events || eventsData.events.length === 0) {
            return `No recent events data available for ${ticker}. Current price: $${currentPrice.toFixed(2)} (${dayChange >= 0 ? '+' : ''}${dayChange.toFixed(2)}% today).`;
          }

          let newsResponse = `Recent events for ${ticker}:\n\n`;

          // Show top 3 most recent events
          const recentEvents = eventsData.events.slice(0, 3);
          recentEvents.forEach((event: any, index: number) => {
            const eventDate = new Date(event.date).toLocaleDateString();
            const impact = event.impact || 'MEDIUM';
            const sentiment = event.sentiment || 'Neutral';

            newsResponse += `${index + 1}. ${eventDate} - ${sentiment} (${impact} impact)\n`;
            if (event.description) {
              // Get first two sentences for context
              const sentences = event.description.split('.').slice(0, 2).join('.') + '.';
              newsResponse += `   ${sentences}\n\n`;
            }
          });

          newsResponse += `Overall sentiment: ${overallSentiment} (${positiveEvents} positive, ${negativeEvents} negative events)`;
          return newsResponse;
        }

        // Handle earnings and financial performance questions
        if (lowerMessage.includes('earnings') || lowerMessage.includes('financial') || lowerMessage.includes('performance') ||
            lowerMessage.includes('revenue') || lowerMessage.includes('profit')) {

          let financialResponse = `${ticker} Financial Overview:\n\n`;

          financialResponse += `Current Stock: $${currentPrice.toFixed(2)} (${dayChange >= 0 ? '+' : ''}${dayChange.toFixed(2)}% today)\n`;

          if (marketCap > 0) {
            const mcapTrillion = marketCap / 1e12;
            if (mcapTrillion >= 1) {
              financialResponse += `Market Cap: $${mcapTrillion.toFixed(2)} trillion\n`;
            } else {
              financialResponse += `Market Cap: $${(marketCap/1e9).toFixed(1)} billion\n`;
            }
          }

          if (peRatio > 0) {
            financialResponse += `P/E Ratio: ${peRatio.toFixed(1)}\n`;
          }

          if (businessMetrics) {
            if (businessMetrics.dividend_yield) {
              financialResponse += `Dividend Yield: ${(businessMetrics.dividend_yield * 100).toFixed(2)}%\n`;
            }
            if (businessMetrics.profit_margin) {
              financialResponse += `Profit Margin: ${(businessMetrics.profit_margin * 100).toFixed(2)}%\n`;
            }
            if (businessMetrics.return_on_equity) {
              financialResponse += `Return on Equity: ${(businessMetrics.return_on_equity * 100).toFixed(2)}%\n`;
            }
          }

          // Add any earnings-related events
          const earningsEvents = eventsData?.events?.filter((event: any) =>
            event.type === 'Earnings' || event.description?.toLowerCase().includes('earnings')
          ) || [];

          if (earningsEvents.length > 0) {
            const latestEarnings = earningsEvents[0];
            financialResponse += `\nLatest Earnings Event (${new Date(latestEarnings.date).toLocaleDateString()}):\n`;
            if (latestEarnings.description) {
              const earningsSummary = latestEarnings.description.split('.').slice(0, 2).join('.') + '.';
              financialResponse += earningsSummary;
            }
          }

          return financialResponse;
        }

        // Default with current data
        return `${ticker} is trading at $${currentPrice.toFixed(2)} (${dayChange >= 0 ? '+' : ''}${dayChange.toFixed(2)}%) with RSI at ${rsi.toFixed(1)}. I can analyze trends, technical indicators, and provide data-driven insights. What would you like to know?`;
      }
    } catch (error) {
      console.warn('Could not get stock data, falling back to general response:', error);
    }

    // Fallback to general response if data not available
    return this.getLocalChatResponse(message, ticker);
  }

  private getLocalChatResponse(message: string, ticker?: string): string {
    const lowerMessage = message.toLowerCase();
    const stockContext = ticker ? ` for ${ticker}` : '';

    // RSI questions
    if (lowerMessage.includes('rsi') || lowerMessage.includes('relative strength')) {
      return `RSI (Relative Strength Index) measures momentum on a scale of 0-100. Above 70 suggests the stock${stockContext} might be overbought (due for a pullback), below 30 suggests oversold (potential buying opportunity), and 30-70 is the neutral zone with room to move either direction.`;
    }

    // Trend questions
    if (lowerMessage.includes('trend') || lowerMessage.includes("what's the trend")) {
      return `To analyze the trend${stockContext}, look at the price charts, moving averages, and technical indicators on this dashboard. Check if the stock is making higher highs and higher lows (uptrend) or lower highs and lower lows (downtrend).`;
    }

    // Buy/investment questions
    if (lowerMessage.includes('should i buy') || lowerMessage.includes('buy?')) {
      return `Investment decisions${stockContext} should consider multiple factors: technical indicators (RSI, MACD), fundamentals (P/E ratio, earnings growth), market conditions, your risk tolerance, and investment timeline. Always do your own research and consider consulting a financial advisor.`;
    }

    // Risk questions
    if (lowerMessage.includes('risk') || lowerMessage.includes('risk analysis')) {
      return `Risk analysis${stockContext} should include volatility (beta), market cap, sector exposure, earnings stability, debt levels, and correlation with market indices. Consider your portfolio diversification and investment timeline when assessing risk.`;
    }

    // Comparison questions
    if (lowerMessage.includes('compare') || lowerMessage.includes('sector')) {
      return `To compare${stockContext} with its sector, analyze relative performance, P/E ratios, growth rates, market share, and competitive advantages. Sector ETFs provide good benchmark comparisons for performance evaluation.`;
    }

    // Price/target questions
    if (lowerMessage.includes('price') || lowerMessage.includes('target')) {
      return `Price targets${stockContext} are based on fundamental analysis, technical analysis, and analyst consensus. Check recent analyst reports, support/resistance levels, and earnings projections for guidance.`;
    }

    // Technical analysis questions
    if (lowerMessage.includes('technical') || lowerMessage.includes('indicator') || lowerMessage.includes('macd')) {
      return `Technical indicators${stockContext} help analyze price momentum and trends. Key indicators include RSI (momentum), MACD (trend changes), moving averages (support/resistance), and volume (strength of moves). Check the charts on this dashboard for current readings.`;
    }

    // Default response
    return `I'm here to help with stock analysis and trading insights${stockContext}. I can explain technical indicators like RSI and MACD, discuss trends and risk factors, compare with sectors, and provide educational insights about investment considerations. What specific aspect would you like to explore?`;
  }
}

const apiService = new APIService();

export { apiService, APIService };
export default apiService;