import React from 'react';
import { CompanyInfo, StockData } from '../../types/backend';
import { formatNumber } from '../../utils';

interface CompanySummaryProps {
  companyInfo: CompanyInfo;
  stockData: StockData;
}

const CompanySummary: React.FC<CompanySummaryProps> = ({
  companyInfo,
  stockData,
}) => {
  const summaryItems = [
    {
      label: 'Industry',
      value: companyInfo.industry || 'N/A',
    },
    {
      label: 'Sector',
      value: companyInfo.sector || 'N/A',
    },
    {
      label: 'Country',
      value: companyInfo.country || 'N/A',
    },
    {
      label: 'Employees',
      value: companyInfo.employees ? formatNumber(companyInfo.employees) : 'N/A',
    },
  ];

  const financialMetrics = [
    {
      label: 'Market Cap',
      value: stockData.market_cap ? `$${(stockData.market_cap / 1000000000).toFixed(2)}B` : 'N/A',
    },
    {
      label: 'P/E Ratio',
      value: stockData.pe_ratio?.toFixed(2) || 'N/A',
    },
    {
      label: 'Annualized Return',
      value: stockData.annualized_return_pct ? `${stockData.annualized_return_pct.toFixed(2)}%` : 'N/A',
    },
    {
      label: 'Beta',
      value: stockData.beta?.toFixed(2) || 'N/A',
    },
  ];

  return (
    <div
      className="border"
      style={{
        backgroundColor: '#1a202c',
        borderColor: '#2d3748'
      }}
    >
      {/* Header */}
      <div
        className="border-b px-6 py-4"
        style={{ borderColor: '#2d3748' }}
      >
        <h2 className="text-xl font-semibold text-white">Company Overview</h2>
      </div>

      <div className="p-6 space-y-6">
        {/* Company Name and Description */}
        <div>
          <h3 className="text-lg font-semibold text-white mb-2">
            {companyInfo.name}
          </h3>
          {companyInfo.description && (
            <p className="text-sm text-gray-400 leading-relaxed line-clamp-4">
              {companyInfo.description}
            </p>
          )}
        </div>

        {/* Basic Information */}
        <div>
          <h4 className="text-sm font-medium text-gray-300 mb-3">Company Details</h4>
          <div className="space-y-3">
            {summaryItems.map((item, index) => (
              <div key={index} className="flex justify-between items-center">
                <span className="text-sm text-gray-400">{item.label}</span>
                <span className="text-sm text-white font-medium">{item.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Financial Metrics */}
        <div>
          <h4 className="text-sm font-medium text-gray-300 mb-3">Key Metrics</h4>
          <div className="space-y-3">
            {financialMetrics.map((metric, index) => (
              <div key={index} className="flex justify-between items-center">
                <span className="text-sm text-gray-400">{metric.label}</span>
                <span className="text-sm text-white font-medium">{metric.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Website Link */}
        {companyInfo.website && (
          <div className="pt-4 border-t" style={{ borderColor: '#2d3748' }}>
            <a
              href={companyInfo.website}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center space-x-2 text-sm transition-colors"
              style={{ color: '#00ff99' }}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
              <span>Visit Company Website</span>
            </a>
          </div>
        )}

        {/* Trading Information */}
        <div
          className="border-t pt-4"
          style={{ borderColor: '#2d3748' }}
        >
          <h4 className="text-sm font-medium text-gray-300 mb-3">Trading Info</h4>
          <div className="grid grid-cols-2 gap-4">
            <div
              className="p-3 border"
              style={{
                backgroundColor: '#0b0e14',
                borderColor: '#2d3748'
              }}
            >
              <div className="text-xs text-gray-400">Exchange</div>
              <div className="text-sm font-semibold text-white">
                {companyInfo.exchange || 'N/A'}
              </div>
            </div>
            <div
              className="p-3 border"
              style={{
                backgroundColor: '#0b0e14',
                borderColor: '#2d3748'
              }}
            >
              <div className="text-xs text-gray-400">Currency</div>
              <div className="text-sm font-semibold text-white">
                {companyInfo.currency || 'N/A'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CompanySummary;