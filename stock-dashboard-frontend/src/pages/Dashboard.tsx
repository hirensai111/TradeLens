import React, { useEffect, useState } from 'react';
import { useParams, useLocation } from 'react-router-dom';
import { useAppContext } from '../utils/context';
import { AnalysisResult } from '../types/backend';
import DashboardHeader from '../components/dashboard/DashboardHeader';
import ChartContainer from '../components/dashboard/ChartContainer';
import MarketDataGrid from '../components/dashboard/MarketDataGrid';
import CompanySummary from '../components/dashboard/CompanySummary';
import EventsSection from '../components/dashboard/EventsSection';
import TradingSignals from '../components/dashboard/TradingSignals';
import LoadingSpinner from '../components/common/LoadingSpinner';
import InlineChatBot from '../components/chat/InlineChatBot';

const Dashboard: React.FC = () => {
  const { ticker } = useParams<{ ticker: string }>();
  const location = useLocation();
  const { actions } = useAppContext();
  const [analysisData, setAnalysisData] = useState<AnalysisResult | null>(null);
  // Initialize date range once to prevent infinite re-renders
  const [selectedDateRange, setSelectedDateRange] = useState<{
    start: string;
    end: string;
  }>(() => {
    const endDate = new Date();
    const startDate = new Date(endDate.getTime() - 30 * 24 * 60 * 60 * 1000);
    return {
      start: startDate.toISOString().split('T')[0],
      end: endDate.toISOString().split('T')[0]
    };
  });

  useEffect(() => {
    console.log('Dashboard useEffect triggered:', {
      hasLocationState: !!location.state?.analysisData,
      ticker,
      analysisDataExists: !!analysisData
    });

    if (location.state?.analysisData) {
      console.log('Setting analysis data from location state');
      setAnalysisData(location.state.analysisData);
      actions.setCurrentStock(location.state.analysisData);

      // Scroll to top when new analysis data is loaded
      setTimeout(() => {
        window.scrollTo(0, 0);
        document.documentElement.scrollTop = 0;
        document.body.scrollTop = 0;
      }, 100);
    } else {
      console.log('No analysis data in location state, current analysisData:', !!analysisData);
    }
  }, [location.state?.analysisData, ticker]);

  // Scroll to top when component mounts
  useEffect(() => {
    // Multiple methods to ensure scroll to top
    window.scrollTo(0, 0);
    document.documentElement.scrollTop = 0;
    document.body.scrollTop = 0;

    // Also try with timeout to ensure it happens after render
    const timer = setTimeout(() => {
      window.scrollTo(0, 0);
      document.documentElement.scrollTop = 0;
      document.body.scrollTop = 0;
    }, 0);

    return () => clearTimeout(timer);
  }, []);

  if (!analysisData) {
    return (
      <div
        className="min-h-screen flex items-center justify-center"
        style={{ backgroundColor: '#0b0e14' }}
      >
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <div
      className="min-h-screen"
      style={{ backgroundColor: '#0b0e14' }}
    >
      {/* Dashboard Header */}
      <DashboardHeader
        ticker={ticker?.toUpperCase() || ''}
        analysisData={analysisData}
        selectedDateRange={selectedDateRange}
        onDateRangeChange={setSelectedDateRange}
      />

      {/* Main Content Grid */}
      <div className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 mb-6">

          {/* Left Column - Chart Section (2/3 width on xl screens) */}
          <div className="xl:col-span-2 space-y-6">
            <ChartContainer
              analysisData={analysisData}
              selectedDateRange={selectedDateRange}
              onDateRangeChange={setSelectedDateRange}
            />

            <MarketDataGrid
              stockData={analysisData.stock_data}
              technicalIndicators={analysisData.technical_indicators}
            />
          </div>

          {/* Right Column - Info Panels (1/3 width on xl screens) */}
          <div className="space-y-6">
            <CompanySummary
              companyInfo={analysisData.company_info}
              stockData={analysisData.stock_data}
            />

            <TradingSignals
              technicalIndicators={analysisData.technical_indicators}
              recommendation={analysisData.recommendation}
            />

            {/* AI Assistant */}
            <InlineChatBot
              ticker={ticker?.toUpperCase()}
              stockData={{
                price: analysisData?.stock_data?.current_price,
                change: analysisData?.stock_data?.price_change_1d,
                changePercent: analysisData?.stock_data?.price_change_1d_pct,
                rsi: analysisData?.technical_indicators?.rsi?.value,
                macd: analysisData?.technical_indicators?.macd?.macd_line
              }}
            />
          </div>
        </div>

        {/* Full Width Events Section */}
        <div className="w-full">
          <EventsSection
            events={analysisData.events}
            sentiment={analysisData.sentiment}
          />
        </div>
      </div>

    </div>
  );
};

export default Dashboard;