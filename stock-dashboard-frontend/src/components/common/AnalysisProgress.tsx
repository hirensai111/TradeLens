import React, { useState, useEffect } from 'react';

// Add CSS styles for animations
const styles = `
  @keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }

  .shimmer-animation {
    animation: shimmer 2s infinite;
  }
`;

interface ProgressStep {
  id: string;
  label: string;
  progress: number;
}

interface AnalysisProgressProps {
  ticker: string;
  isVisible: boolean;
  onCancel?: () => void;
  onComplete?: () => void;
  externalProgress?: number;
  externalStep?: string;
}

const PROGRESS_STEPS: ProgressStep[] = [
  {
    id: 'data',
    label: 'Fetching historical data...',
    progress: 15,
  },
  {
    id: 'indicators',
    label: 'Calculating technical indicators...',
    progress: 35,
  },
  {
    id: 'events',
    label: 'Analyzing price events...',
    progress: 65,
  },
  {
    id: 'sentiment',
    label: 'Processing sentiment data...',
    progress: 85,
  },
  {
    id: 'report',
    label: 'Generating report...',
    progress: 100,
  }
];

const AnalysisProgress: React.FC<AnalysisProgressProps> = ({
  ticker,
  isVisible,
  onCancel,
  onComplete,
  externalProgress,
  externalStep
}) => {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [currentProgress, setCurrentProgress] = useState(0);
  const [startTime, setStartTime] = useState<number>(Date.now());
  const [timeElapsed, setTimeElapsed] = useState(0);
  const [currentStepLabel, setCurrentStepLabel] = useState('');

  // Handle external progress updates from API
  useEffect(() => {
    if (externalProgress !== undefined) {
      setCurrentProgress(externalProgress);

      // Update step index based on progress percentage
      let stepIndex = 0;
      if (externalProgress >= 85) {
        stepIndex = 4; // Generating report
      } else if (externalProgress >= 65) {
        stepIndex = 3; // Processing sentiment
      } else if (externalProgress >= 35) {
        stepIndex = 2; // Analyzing events
      } else if (externalProgress >= 15) {
        stepIndex = 1; // Calculating indicators
      } else if (externalProgress >= 5) {
        stepIndex = 0; // Fetching data
      }
      setCurrentStepIndex(stepIndex);
      console.log(`External progress: ${externalProgress}% -> Step ${stepIndex}`);

      if (externalProgress === 100) {
        setCurrentStepIndex(PROGRESS_STEPS.length - 1);
        setCurrentStepLabel('Analysis complete!');

        // Call onComplete after a brief delay to show 100%
        setTimeout(() => {
          onComplete?.();
        }, 1500);
        return;
      }
    }

    if (externalStep) {
      setCurrentStepLabel(externalStep);
    }
  }, [externalProgress, externalStep, onComplete]);

  useEffect(() => {
    if (!isVisible) {
      // Reset state when not visible
      setCurrentStepIndex(0);
      setCurrentProgress(0);
      setStartTime(Date.now());
      setTimeElapsed(0);
      setCurrentStepLabel('');
      return;
    }

    // Start the progress simulation
    setStartTime(Date.now());

    const progressTimer = setInterval(() => {
      const now = Date.now();
      const elapsed = now - startTime;
      setTimeElapsed(elapsed);

      // Only update progress if we don't have external progress
      if (externalProgress === undefined) {
        // More realistic progress based on actual analysis time
        // Since real analysis takes 30-60+ seconds, use a slower progression
        let targetStepIndex = 0;
        let targetProgress = 0;

        // Progress through steps based on elapsed time intervals
        // Extended timing for longer analysis periods
        if (elapsed < 15000) {
          // First 15 seconds - fetching data
          targetStepIndex = 0;
          targetProgress = Math.min(15, (elapsed / 15000) * 15);
          setCurrentStepLabel('Fetching historical data...');
        } else if (elapsed < 35000) {
          // 15-35 seconds - calculating indicators
          targetStepIndex = 1;
          targetProgress = 15 + Math.min(20, ((elapsed - 15000) / 20000) * 20);
          setCurrentStepLabel('Calculating technical indicators...');
        } else if (elapsed < 60000) {
          // 35-60 seconds - analyzing events
          targetStepIndex = 2;
          targetProgress = 35 + Math.min(30, ((elapsed - 35000) / 25000) * 30);
          setCurrentStepLabel('Analyzing price events...');
        } else if (elapsed < 90000) {
          // 60-90 seconds - processing sentiment
          targetStepIndex = 3;
          targetProgress = 65 + Math.min(20, ((elapsed - 60000) / 30000) * 20);
          setCurrentStepLabel('Processing sentiment data...');
        } else {
          // 90+ seconds - generating report (keep progressing slowly)
          targetStepIndex = 4;
          // Continue progressing slowly after 90 seconds, approaching but never reaching 100%
          const extraTime = elapsed - 90000;
          const extraProgress = Math.min(14, (extraTime / 60000) * 14); // Very slow progress
          targetProgress = 85 + extraProgress;
          setCurrentStepLabel('Generating comprehensive report...');
        }

        setCurrentStepIndex(targetStepIndex);
        setCurrentProgress(Math.min(targetProgress, 99)); // Cap at 99% until actual completion

        // Debug logging to track step progression
        if (targetStepIndex !== currentStepIndex) {
          console.log(`Step progression: ${currentStepIndex} -> ${targetStepIndex}, Progress: ${Math.round(targetProgress)}%`);
        }
      }
    }, 100);

    return () => clearInterval(progressTimer);
  }, [isVisible, onComplete, externalProgress]);

  const formatTime = (milliseconds: number) => {
    const seconds = Math.ceil(milliseconds / 1000);
    if (seconds < 60) {
      return `${seconds}s`;
    }
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  if (!isVisible) {
    return null;
  }

  const currentStep = PROGRESS_STEPS[currentStepIndex];

  return (
    <>
      <style>{styles}</style>
      <div
        className="fixed inset-0 flex items-center justify-center z-50"
        style={{ backgroundColor: 'rgba(11, 14, 20, 0.95)' }}
      >
      <div
        className="max-w-md w-full mx-4 p-8 rounded-xl border-2 shadow-2xl"
        style={{
          backgroundColor: '#1a202c',
          borderColor: '#2d3748',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(45, 55, 72, 0.5)'
        }}
      >
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-3">
            <div
              className="w-12 h-12 rounded-full flex items-center justify-center mr-3 p-2"
              style={{
                backgroundColor: 'rgba(0, 255, 153, 0.1)',
                border: '2px solid #00ff99'
              }}
            >
              <img
                src="/tradeLens.png"
                alt="TradeLens Logo"
                className="w-8 h-8"
              />
            </div>
            <h2 className="text-2xl font-bold text-white">
              Analyzing <span style={{ color: '#00ff99' }}>{ticker}</span>
            </h2>
          </div>
          <p className="text-gray-400 text-sm">
            Professional market analysis powered by TradeLens
          </p>
        </div>

        {/* Progress Bar */}
        <div className="mb-6">
          <div
            className="w-full h-4 rounded-full mb-4 relative overflow-hidden"
            style={{
              backgroundColor: '#0b0e14',
              border: '1px solid #2d3748'
            }}
          >
            <div
              className="h-full rounded-full transition-all duration-500 ease-out relative"
              style={{
                backgroundColor: '#00ff99',
                width: `${currentProgress}%`,
                boxShadow: '0 0 15px rgba(0, 255, 153, 0.6)',
              }}
            >
              {/* Animated shimmer effect */}
              <div
                className="absolute inset-0 opacity-50 shimmer-animation"
                style={{
                  background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)'
                }}
              />
            </div>
          </div>

          <div className="flex justify-between text-sm text-gray-400">
            <span>{Math.round(currentProgress)}% Complete</span>
            <span>Elapsed: {formatTime(timeElapsed)}</span>
          </div>
        </div>

        {/* Current Step */}
        <div className="mb-8">
          <div className="flex items-center mb-4 p-3 rounded-lg" style={{ backgroundColor: 'rgba(0, 255, 153, 0.05)' }}>
            <div
              className="w-3 h-3 rounded-full mr-3 animate-pulse"
              style={{ backgroundColor: '#00ff99', boxShadow: '0 0 8px rgba(0, 255, 153, 0.8)' }}
            />
            <span className="text-white font-medium">
              {currentStepLabel || currentStep?.label || 'Processing...'}
            </span>
          </div>

          {/* Step Progress Indicators */}
          <div className="flex justify-between px-2">
            {PROGRESS_STEPS.map((step, index) => (
              <div key={step.id} className="flex flex-col items-center">
                <div
                  className={`w-3 h-3 rounded-full transition-all duration-500 ${
                    index <= currentStepIndex
                      ? 'shadow-lg'
                      : ''
                  }`}
                  style={{
                    backgroundColor: index <= currentStepIndex ? '#00ff99' : '#4a5568',
                    boxShadow: index <= currentStepIndex ? '0 0 8px rgba(0, 255, 153, 0.6)' : 'none',
                    transform: index <= currentStepIndex ? 'scale(1.1)' : 'scale(1)'
                  }}
                />
                <span className="text-xs text-gray-500 mt-2 text-center max-w-[60px] leading-tight">
                  {step.label.split(' ')[0]}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Time Information */}
        <div className="text-center text-sm text-gray-400 mb-6">
          <span>Analysis in progress • {formatTime(timeElapsed)} elapsed</span>
        </div>

        {/* Cancel Button */}
        {onCancel && (
          <button
            onClick={onCancel}
            className="w-full py-3 px-4 border-2 rounded-lg font-medium transition-all duration-200"
            style={{
              borderColor: '#4a5568',
              color: '#9ca3af',
              backgroundColor: 'transparent'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = '#6b7280';
              e.currentTarget.style.color = '#ffffff';
              e.currentTarget.style.backgroundColor = 'rgba(107, 114, 128, 0.1)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = '#4a5568';
              e.currentTarget.style.color = '#9ca3af';
              e.currentTarget.style.backgroundColor = 'transparent';
            }}
          >
            Cancel Analysis
          </button>
        )}

        {/* Professional Trading Platform Footer */}
        <div className="mt-6 pt-4 border-t border-gray-700">
          <div className="flex items-center justify-center text-xs text-gray-500">
            <div
              className="w-2 h-2 rounded-full mr-2"
              style={{ backgroundColor: '#00ff99' }}
            />
            Secure real-time market data analysis
          </div>
        </div>
      </div>
    </div>
    </>
  );
};

export default AnalysisProgress;