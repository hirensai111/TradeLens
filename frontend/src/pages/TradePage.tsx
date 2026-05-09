import React from 'react';
import ModeSelector from '../components/common/ModeSelector';
import AnimatedBackground from '../components/common/AnimatedBackground';

const TradePage: React.FC = () => {
  const features = [
    {
      icon: '📊',
      title: 'Live Trading',
      description: 'Execute trades in real-time with live market data'
    },
    {
      icon: '🤖',
      title: 'Auto Trading',
      description: 'Set up automated trading strategies'
    },
    {
      icon: '💼',
      title: 'Portfolio Tracking',
      description: 'Monitor your investments in one place'
    },
    {
      icon: '⚡',
      title: 'Smart Orders',
      description: 'Advanced order types for better control'
    },
    {
      icon: '🔔',
      title: 'Price Alerts',
      description: 'Get notified of important price movements'
    },
    {
      icon: '🛡️',
      title: 'Risk Management',
      description: 'Protect your portfolio with smart limits'
    }
  ];

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

      {/* Coming Soon Content */}
      <div className="text-center px-4 py-16">
        <div className="max-w-6xl mx-auto">
          {/* Animated Rocket */}
          <div
            className="text-9xl mb-8 inline-block animate-[float_3s_ease-in-out_infinite]"
          >
            🚀
          </div>

          {/* Title with Gradient */}
          <h2
            className="text-5xl font-bold mb-4 animate-[fadeInUp_0.8s_ease-out_0.2s] animate-fill-both"
            style={{
              background: 'linear-gradient(135deg, #00ff99, #00e5ff)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text'
            } as React.CSSProperties}
          >
            Trading Platform Coming Soon
          </h2>

          <p className="text-xl text-gray-400 mb-16 animate-[fadeInUp_0.8s_ease-out_0.3s] animate-fill-both">
            We're building something amazing for you!
          </p>

          {/* Features Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <div
                key={index}
                className="p-8 text-center transition-all duration-300 rounded-3xl transform hover:scale-105 hover:-translate-y-2 active:scale-95 animate-[fadeInUp_0.8s_ease-out] animate-fill-both"
                style={{
                  background: 'rgba(26, 32, 44, 0.6)',
                  backdropFilter: 'blur(20px)',
                  WebkitBackdropFilter: 'blur(20px)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
                  animationDelay: `${0.4 + index * 0.1}s`
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = 'rgba(0, 255, 153, 0.5)';
                  e.currentTarget.style.boxShadow = '0 12px 48px rgba(0, 255, 153, 0.3)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                  e.currentTarget.style.boxShadow = '0 8px 32px rgba(0, 0, 0, 0.3)';
                }}
              >
                <div className="text-6xl mb-4 animate-[float_3s_ease-in-out_infinite]" style={{ animationDelay: `${index * 0.2}s` }}>
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">
                  {feature.title}
                </h3>
                <p className="text-sm text-gray-400">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>

          {/* Additional Coming Soon Info */}
          <div
            className="mt-16 p-8 rounded-3xl max-w-2xl mx-auto animate-[fadeInUp_1s_ease-out_1s] animate-fill-both"
            style={{
              background: 'linear-gradient(135deg, rgba(0, 255, 153, 0.1) 0%, rgba(0, 229, 255, 0.1) 100%)',
              backdropFilter: 'blur(20px)',
              WebkitBackdropFilter: 'blur(20px)',
              border: '1px solid rgba(0, 255, 153, 0.2)',
              boxShadow: '0 8px 32px rgba(0, 255, 153, 0.15)'
            }}
          >
            <h3
              className="text-2xl font-bold mb-4"
              style={{
                background: 'linear-gradient(135deg, #00ff99 0%, #00e5ff 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text'
              } as React.CSSProperties}
            >
              🎉 Get Early Access
            </h3>
            <p className="text-gray-300 mb-4">
              Be the first to know when our trading platform launches. Join our waitlist for exclusive early access and special launch features!
            </p>
            <button
              className="px-8 py-4 rounded-2xl font-semibold text-lg transition-all duration-300 transform hover:scale-105 active:scale-95"
              style={{
                background: 'linear-gradient(135deg, #00ff99 0%, #00e5ff 100%)',
                color: '#000000',
                boxShadow: '0 8px 32px rgba(0, 255, 153, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.3)'
              }}
            >
              Join Waitlist
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TradePage;
