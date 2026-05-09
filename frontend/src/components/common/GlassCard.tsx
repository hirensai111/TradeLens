import React, { ReactNode } from 'react';

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  title?: string;
  hoverable?: boolean;
}

const GlassCard: React.FC<GlassCardProps> = ({
  children,
  className = '',
  title,
  hoverable = false
}) => {
  return (
    <div
      className={`p-6 rounded-3xl transition-all duration-300 ${
        hoverable ? 'hover:scale-[1.02] hover:shadow-2xl' : ''
      } ${className}`}
      style={{
        background: 'rgba(26, 32, 44, 0.6)',
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1)'
      }}
    >
      {title && (
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
          {title}
        </h3>
      )}
      {children}
    </div>
  );
};

export default GlassCard;
