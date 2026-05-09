import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import apiService from '../../services/api';

type Mode = 'analyze' | 'predict' | 'trade';

interface ModeSelectorProps {
  className?: string;
  ticker?: string;
}

const ModeSelector: React.FC<ModeSelectorProps> = ({ className = '', ticker }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [isLoading, setIsLoading] = useState(false);

  const getCurrentMode = (): Mode => {
    const path = location.pathname;
    if (path === '/' || path === '/analyze' || path.startsWith('/dashboard/')) return 'analyze';
    if (path.startsWith('/predict')) return 'predict';
    if (path.startsWith('/trade')) return 'trade';
    return 'analyze';
  };

  const currentMode = getCurrentMode();

  const handleModeChange = async (mode: Mode) => {
    switch (mode) {
      case 'analyze':
        if (ticker) {
          // If we have a ticker, fetch analysis data and navigate to dashboard
          setIsLoading(true);
          try {
            const analysisData = await apiService.analyzeStock(ticker.toUpperCase());
            navigate(`/dashboard/${ticker.toUpperCase()}`, {
              state: { analysisData }
            });
          } catch (error) {
            console.error('Error fetching analysis data:', error);
            // Still navigate but without data - dashboard will show error
            navigate(`/dashboard/${ticker.toUpperCase()}`);
          } finally {
            setIsLoading(false);
          }
        } else {
          // Otherwise go to homepage
          navigate('/');
        }
        break;
      case 'predict':
        if (ticker) {
          // If we have a ticker, pass it to predict page
          navigate('/predict', { state: { ticker: ticker.toUpperCase() } });
        } else {
          navigate('/predict');
        }
        break;
      case 'trade':
        navigate('/trade');
        break;
    }
  };

  return (
    <div
      className={`inline-flex gap-2 p-1.5 rounded-xl ${className}`}
      style={{
        background: 'rgba(26, 32, 44, 0.6)',
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
      }}
    >
      <button
        onClick={() => handleModeChange('analyze')}
        disabled={isLoading}
        className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-500 transform hover:scale-105 active:scale-95 ${
          currentMode === 'analyze' ? 'text-black' : 'text-white'
        } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
        style={
          currentMode === 'analyze'
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
        {isLoading ? (
          <span className="flex items-center justify-center">
            <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          </span>
        ) : (
          'Analyze'
        )}
      </button>
      <button
        onClick={() => handleModeChange('predict')}
        className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-500 transform hover:scale-105 active:scale-95 ${
          currentMode === 'predict' ? 'text-black' : 'text-white'
        }`}
        style={
          currentMode === 'predict'
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
        Predict
      </button>
      <button
        onClick={() => handleModeChange('trade')}
        className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-500 transform hover:scale-105 active:scale-95 ${
          currentMode === 'trade' ? 'text-black' : 'text-white'
        }`}
        style={
          currentMode === 'trade'
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
        Trade
      </button>
    </div>
  );
};

export default ModeSelector;
