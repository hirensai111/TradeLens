import React, { useState, useRef, useEffect } from 'react';
import apiService from '../../services/api';

interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: Date;
}

interface InlineChatBotProps {
  ticker?: string;
  stockData?: {
    price?: number;
    change?: number;
    changePercent?: number;
    rsi?: number;
    macd?: number;
  };
}

const InlineChatBot: React.FC<InlineChatBotProps> = ({ ticker, stockData }) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: `Hi! I'm TradeLens AI, your personal trading assistant. I can help analyze stocks, explain indicators, and provide trading insights based on real-time data.${ticker ? ` I see you're looking at ${ticker}. How can I help?` : ''}`,
      isUser: false,
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const quickActions = [
    "What's the trend?",
    "Should I buy?",
    "Risk analysis",
    "Compare sector",
    "RSI analysis",
    "MACD signal",
    "Help"
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({
      behavior: 'smooth',
      block: 'nearest',
      inline: 'nearest'
    });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async (content: string) => {
    if (!content.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: content.trim(),
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    try {
      const response = await apiService.sendChatMessage(content.trim(), ticker);

      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: response,
        isUser: false,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiResponse]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: 'Sorry, I encountered an error. Please try again.',
        isUser: false,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(inputValue);
  };

  const handleQuickAction = (action: string, e?: React.MouseEvent) => {
    e?.preventDefault();
    e?.stopPropagation();
    sendMessage(action);
  };

  const clearChat = () => {
    setMessages([
      {
        id: '1',
        content: `Hi! I'm TradeLens AI, your personal trading assistant. I can help analyze stocks, explain indicators, and provide trading insights based on real-time data.${ticker ? ` I see you're looking at ${ticker}. How can I help?` : ''}`,
        isUser: false,
        timestamp: new Date()
      }
    ]);
  };

  const formatTime = (date: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <div
      className="border mt-6"
      style={{
        backgroundColor: '#1a202c',
        borderColor: '#2d3748'
      }}
    >
      {/* Chat Header */}
      <div
        className="border-b px-6 py-4"
        style={{ borderColor: '#2d3748' }}
      >
        <div className="flex items-center gap-3">
          <div className="relative w-8 h-8 rounded-full flex items-center justify-center"
               style={{ backgroundColor: '#000000' }}>
            <img
              src="/tradeLens.png"
              alt="TradeLens"
              className="w-6 h-6 object-contain"
            />
            <div
              className="absolute -top-1 -right-1 w-2 h-2 border rounded-full"
              style={{ backgroundColor: '#10b981', borderColor: '#1a202c' }}
            ></div>
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-white">TradeLens AI Assistant</h3>
            <span className="text-gray-400 text-sm">Ready to assist with {ticker || 'trading'}</span>
          </div>
          <button
            onClick={clearChat}
            className="text-xs px-3 py-1 rounded transition-colors"
            style={{
              color: '#9ca3af',
              border: '1px solid rgba(255, 255, 255, 0.1)'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = '#e5e7eb';
              e.currentTarget.style.backgroundColor = '#0b0e14';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = '#9ca3af';
              e.currentTarget.style.backgroundColor = 'transparent';
            }}
          >
            Clear Chat
          </button>
        </div>

        {/* Stock Context Bar */}
        {ticker && (
          <div
            className="flex gap-4 p-3 rounded-lg text-xs mt-3"
            style={{ backgroundColor: '#0b0e14', border: '1px solid rgba(255, 255, 255, 0.1)' }}
          >
            <div className="flex flex-col items-center">
              <span className="text-gray-400 mb-1">Price</span>
              <span className="font-medium" style={{ color: '#10b981' }}>
                ${stockData?.price?.toFixed(2) || '—'}
              </span>
            </div>
            <div className="flex flex-col items-center">
              <span className="text-gray-400 mb-1">Change</span>
              <span
                className="font-medium"
                style={{
                  color: (stockData?.changePercent || 0) >= 0 ? '#10b981' : '#ef4444'
                }}
              >
                {stockData?.changePercent !== undefined
                  ? `${stockData.changePercent > 0 ? '+' : ''}${stockData.changePercent.toFixed(2)}%`
                  : '—'
                }
              </span>
            </div>
            <div className="flex flex-col items-center">
              <span className="text-gray-400 mb-1">RSI</span>
              <span className="font-medium text-white">
                {stockData?.rsi?.toFixed(1) || '—'}
              </span>
            </div>
            <div className="flex flex-col items-center">
              <span className="text-gray-400 mb-1">MACD</span>
              <span className="font-medium text-white">
                {stockData?.macd?.toFixed(3) || '—'}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Messages Area */}
      <div
        className="p-4 space-y-4 h-64 overflow-y-auto"
        style={{ backgroundColor: '#1a202c' }}
      >
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
          >
            <div className={`flex items-start gap-3 max-w-[80%] ${message.isUser ? 'flex-row-reverse' : ''}`}>
              {!message.isUser && (
                <div
                  className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0"
                  style={{ backgroundColor: '#000000' }}
                >
                  <img
                    src="/tradeLens.png"
                    alt="TradeLens"
                    className="w-4 h-4 object-contain"
                  />
                </div>
              )}
              <div
                className={`p-3 rounded-xl shadow-sm ${
                  message.isUser ? 'rounded-br-sm' : 'rounded-bl-sm'
                }`}
                style={{
                  backgroundColor: message.isUser ? '#10b981' : '#0b0e14',
                  color: message.isUser ? 'white' : '#e5e7eb',
                  border: '1px solid rgba(255, 255, 255, 0.1)'
                }}
              >
                <p className="text-sm leading-relaxed mb-1">{message.content}</p>
                <span className="text-xs opacity-60">
                  {formatTime(message.timestamp)}
                </span>
              </div>
            </div>
          </div>
        ))}

        {/* Typing Indicator */}
        {isTyping && (
          <div className="flex items-start gap-3">
            <div
              className="w-6 h-6 rounded-full flex items-center justify-center"
              style={{ backgroundColor: '#000000' }}
            >
              <img
                src="/tradeLens.png"
                alt="TradeLens"
                className="w-4 h-4 object-contain"
              />
            </div>
            <div
              className="p-3 rounded-xl rounded-bl-sm shadow-sm"
              style={{
                backgroundColor: '#0b0e14',
                border: '1px solid rgba(255, 255, 255, 0.1)'
              }}
            >
              <div className="flex gap-1">
                <div
                  className="w-2 h-2 rounded-full animate-bounce"
                  style={{ backgroundColor: '#10b981' }}
                ></div>
                <div
                  className="w-2 h-2 rounded-full animate-bounce"
                  style={{ backgroundColor: '#10b981', animationDelay: '0.1s' }}
                ></div>
                <div
                  className="w-2 h-2 rounded-full animate-bounce"
                  style={{ backgroundColor: '#10b981', animationDelay: '0.2s' }}
                ></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Quick Actions */}
      <div
        className="px-4 py-3 border-t"
        style={{
          backgroundColor: '#0b0e14',
          borderColor: 'rgba(255, 255, 255, 0.1)'
        }}
      >
        <div className="flex flex-wrap gap-2">
          {quickActions.map((action) => (
            <button
              key={action}
              onClick={(e) => handleQuickAction(action, e)}
              className="px-3 py-1.5 text-xs rounded-full transition-all duration-200"
              style={{
                backgroundColor: '#1a202c',
                color: '#e5e7eb',
                border: '1px solid rgba(255, 255, 255, 0.1)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = '#10b981';
                e.currentTarget.style.color = 'white';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = '#1a202c';
                e.currentTarget.style.color = '#e5e7eb';
              }}
            >
              {action}
            </button>
          ))}
        </div>
      </div>

      {/* Input Area */}
      <form
        onSubmit={handleSubmit}
        className="p-4 border-t"
        style={{
          backgroundColor: '#0b0e14',
          borderColor: 'rgba(255, 255, 255, 0.1)'
        }}
      >
        <div className="flex gap-3">
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder={`Ask about ${ticker || 'stocks'}...`}
            className="flex-1 px-4 py-2 rounded-full focus:outline-none text-sm"
            style={{
              backgroundColor: '#1a202c',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              color: '#e5e7eb'
            }}
            disabled={isTyping}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = '#10b981';
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
            }}
          />
          <button
            type="submit"
            disabled={!inputValue.trim() || isTyping}
            className="w-10 h-10 rounded-full flex items-center justify-center transition-colors"
            style={{
              backgroundColor: !inputValue.trim() || isTyping ? '#374151' : '#10b981',
              color: 'white'
            }}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        </div>
      </form>
    </div>
  );
};

export default InlineChatBot;