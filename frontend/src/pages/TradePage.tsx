import React from 'react';
import ModeSelector from '../components/common/ModeSelector';
import AnimatedBackground from '../components/common/AnimatedBackground';

const TradePage: React.FC = () => {
  return (
    <div className="min-h-screen relative overflow-hidden">
      <AnimatedBackground />

      {/* Header */}
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
    </div>
  );
};

export default TradePage;
