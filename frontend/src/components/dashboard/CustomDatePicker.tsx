import React, { useState, useRef, useEffect } from 'react';

interface CustomDatePickerProps {
  selectedRange: {
    start: string;
    end: string;
  };
  onRangeChange: (range: { start: string; end: string }) => void;
  onClose: () => void;
}

const PRESET_RANGES = [
  { label: '1 Week', days: 7 },
  { label: '1 Month', days: 30 },
  { label: '3 Months', days: 90 },
  { label: '6 Months', days: 180 },
  { label: '1 Year', days: 365 },
];

const CustomDatePicker: React.FC<CustomDatePickerProps> = ({
  selectedRange,
  onRangeChange,
  onClose,
}) => {
  const [startDate, setStartDate] = useState(selectedRange.start);
  const [endDate, setEndDate] = useState(selectedRange.end);
  const pickerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (pickerRef.current && !pickerRef.current.contains(event.target as Node)) {
        onClose();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [onClose]);

  const handlePresetClick = (days: number) => {
    const endDate = new Date();
    endDate.setHours(23, 59, 59, 999); // End of day to prevent drift
    const startDate = new Date(endDate.getTime() - days * 24 * 60 * 60 * 1000);

    const newRange = {
      start: startDate.toISOString().split('T')[0],
      end: endDate.toISOString().split('T')[0],
    };

    setStartDate(newRange.start);
    setEndDate(newRange.end);
    onRangeChange(newRange);
  };

  const handleApply = () => {
    if (startDate && endDate && new Date(startDate) <= new Date(endDate)) {
      onRangeChange({ start: startDate, end: endDate });
      onClose();
    }
  };

  const handleCancel = () => {
    setStartDate(selectedRange.start);
    setEndDate(selectedRange.end);
    onClose();
  };

  const isValidRange = startDate && endDate && new Date(startDate) <= new Date(endDate);

  return (
    <div
      ref={pickerRef}
      className="absolute top-full right-0 mt-2 border shadow-lg z-50 animate-fade-in"
      style={{
        backgroundColor: '#1a202c',
        borderColor: '#2d3748',
        width: '320px'
      }}
    >
      <div className="p-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">Select Date Range</h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Preset Ranges */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Quick Select
          </label>
          <div className="grid grid-cols-2 gap-2">
            {PRESET_RANGES.map((preset) => (
              <button
                key={preset.label}
                onClick={() => handlePresetClick(preset.days)}
                className="px-3 py-2 text-sm border transition-colors hover:bg-gray-600"
                style={{
                  backgroundColor: '#0b0e14',
                  borderColor: '#2d3748',
                  color: '#ffffff'
                }}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>

        {/* Custom Date Inputs */}
        <div className="space-y-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Start Date
            </label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full px-3 py-2 border text-white transition-colors"
              style={{
                backgroundColor: '#0b0e14',
                borderColor: '#2d3748'
              }}
              max={endDate || new Date().toISOString().split('T')[0]}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              End Date
            </label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="w-full px-3 py-2 border text-white transition-colors"
              style={{
                backgroundColor: '#0b0e14',
                borderColor: '#2d3748'
              }}
              min={startDate}
              max={new Date().toISOString().split('T')[0]}
            />
          </div>
        </div>

        {/* Validation Error */}
        {!isValidRange && startDate && endDate && (
          <div className="mb-4 p-2 border text-sm text-red-400"
            style={{
              backgroundColor: '#2d1b1b',
              borderColor: '#ff4757'
            }}
          >
            Start date must be before or equal to end date
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex space-x-2">
          <button
            onClick={handleCancel}
            className="flex-1 px-4 py-2 border text-gray-300 transition-colors hover:bg-gray-600"
            style={{
              borderColor: '#2d3748'
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleApply}
            disabled={!isValidRange}
            className={`flex-1 px-4 py-2 font-medium transition-colors ${
              isValidRange
                ? 'hover:opacity-90'
                : 'opacity-50 cursor-not-allowed'
            }`}
            style={{
              backgroundColor: '#00ff99',
              color: '#000000'
            }}
          >
            Apply
          </button>
        </div>
      </div>
    </div>
  );
};

export default CustomDatePicker;