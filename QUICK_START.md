# TradeLens - Quick Start Guide

## 📁 New Project Structure

Your project is now organized into three main functional areas:

```
tradelens/
├── 📰 news_system/          # News collection & sentiment analysis
├── 🤖 prediction_engine/    # ML models & stock prediction
├── 📊 options_analyzer/     # Options trading & bots
├── 🔧 core/                 # Shared utilities
├── 🌐 api/                  # REST API & Telegram bot
├── 📄 output/               # Excel & JSON reports
├── 🤝 ai_assistant/         # AI chat
└── 🧪 tests/                # All tests
```

---

## 🚀 Quick Actions

### 1. News Analysis
```bash
# Collect and analyze news for a stock
python -c "
from news_system.collectors.news_api_client import NewsAPIClient
from news_system.analyzers.sentiment_analyzer import SentimentAnalyzer

client = NewsAPIClient()
news = client.get_news('AAPL')
analyzer = SentimentAnalyzer()
for article in news[:5]:
    sentiment = analyzer.analyze(article['content'])
    print(f'{article[\"title\"]}: {sentiment}')
"
```

### 2. Stock Prediction
```bash
# Predict stock movement
python -c "
from prediction_engine.predictors.prediction_engine import PredictionEngine

predictor = PredictionEngine()
predictions = predictor.predict_multi_day('AAPL', days=5)
print(predictions)
"
```

### 3. Options Analysis
```bash
# Analyze options
python -c "
from options_analyzer.analyzers.options_analyzer import OptionsAnalyzer

analyzer = OptionsAnalyzer()
greeks = analyzer.calculate_greeks(
    spot=18500, strike=18600,
    volatility=0.15, time_to_expiry=7
)
print(greeks)
"
```

### 4. Start REST API
```bash
python api/rest/api_server.py
# API available at http://localhost:5000
```

### 5. Run Trading Bot
```bash
python options_analyzer/bots/automated_options_bot.py
```

---

## 📚 Module Overview

### 📰 News System
**Location**: `news_system/`

**What it does**:
- Collects news from NewsAPI, Polygon, Reddit, Yahoo Finance
- Analyzes sentiment using advanced NLP
- Extracts financial events
- Correlates news with price movements

**Main files**:
- `collectors/news_api_client.py` - Main news orchestrator
- `analyzers/sentiment_analyzer.py` - Sentiment analysis
- `analyzers/event_analyzer.py` - Event extraction
- `processors/` - Text processing, deduplication

### 🤖 Prediction Engine
**Location**: `prediction_engine/`

**What it does**:
- Technical analysis (RSI, MACD, Bollinger Bands, etc.)
- Machine learning predictions
- Multi-day forecasts
- Feature engineering

**Main files**:
- `technical_analysis/data_processor.py` - Technical indicators
- `predictors/prediction_engine.py` - ML predictions
- `features/feature_engineering.py` - Feature creation
- `data_loaders/` - Data loading from various sources

### 📊 Options Analyzer
**Location**: `options_analyzer/`

**What it does**:
- Options Greeks calculation
- Automated intraday trading
- Broker integrations (Zerodha, Groww)
- Indian market support (NSE, BSE, Nifty)

**Main files**:
- `analyzers/options_analyzer.py` - Options analysis & Greeks
- `bots/automated_options_bot.py` - Automated trading bot
- `brokers/zerodha_api_client.py` - Zerodha integration
- `brokers/groww_api_client.py` - Groww integration

---

## 🔑 Environment Setup

1. **Copy environment template**:
```bash
cp .env.example .env
```

2. **Add your API keys** to `.env`:
```env
# News APIs
NEWSAPI_KEY=your_newsapi_key
POLYGON_API_KEY=your_polygon_key

# OpenAI (for AI assistant)
OPENAI_API_KEY=your_openai_key

# Brokers
ZERODHA_API_KEY=your_zerodha_key
ZERODHA_API_SECRET=your_zerodha_secret
GROWW_API_KEY=your_groww_key

# Telegram
TELEGRAM_BOT_TOKEN=your_telegram_token

# Alpha Vantage (fallback)
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

---

## 🧪 Running Tests

```bash
# All tests
python -m pytest tests/

# By module
python -m pytest tests/news_system/
python -m pytest tests/prediction_engine/
python -m pytest tests/options_analyzer/

# Integration tests
python -m pytest tests/integration/

# Specific test file
python -m pytest tests/news_system/test_sentiment_integration.py
```

---

## 📊 Common Workflows

### Workflow 1: Complete Stock Analysis
```python
from news_system.collectors.news_api_client import NewsAPIClient
from news_system.analyzers.sentiment_analyzer import SentimentAnalyzer
from prediction_engine.technical_analysis.data_processor import DataProcessor
from prediction_engine.predictors.prediction_engine import PredictionEngine
from output.excel.excel_generator import ExcelGenerator

# 1. Collect news
news_client = NewsAPIClient()
news = news_client.get_news('AAPL', days_back=7)

# 2. Analyze sentiment
sentiment_analyzer = SentimentAnalyzer()
sentiment_scores = [sentiment_analyzer.analyze(article['content']) for article in news]

# 3. Get technical data
processor = DataProcessor()
technical_data = processor.process_stock_data('AAPL')

# 4. Make predictions
predictor = PredictionEngine()
predictions = predictor.predict_multi_day('AAPL', days=5)

# 5. Generate report
excel_gen = ExcelGenerator()
excel_gen.generate_comprehensive_report(
    symbol='AAPL',
    technical_data=technical_data,
    news=news,
    sentiment=sentiment_scores,
    predictions=predictions,
    output_file='AAPL_analysis.xlsx'
)
```

### Workflow 2: Automated Options Trading
```python
from options_analyzer.bots.automated_options_bot import AutomatedOptionsBot
from options_analyzer.brokers.zerodha_api_client import ZerodhaClient

# 1. Initialize broker connection
zerodha = ZerodhaClient(api_key='your_key', access_token='token')

# 2. Initialize trading bot
bot = AutomatedOptionsBot(
    broker=zerodha,
    capital=100000,
    risk_per_trade=0.02,
    symbols=['NIFTY', 'BANKNIFTY']
)

# 3. Start automated trading
bot.start()
```

### Workflow 3: Real-time News Monitoring
```python
from news_system.collectors.scheduler import NewsScheduler
from news_system.analyzers.correlation_analyzer import CorrelationAnalyzer
from api.telegram.telegram_bot import TelegramBot

# 1. Setup news scheduler
scheduler = NewsScheduler(interval_minutes=15)
scheduler.add_symbols(['AAPL', 'GOOGL', 'MSFT', 'TSLA'])

# 2. Setup correlation analyzer
correlator = CorrelationAnalyzer()

# 3. Start Telegram bot for alerts
bot = TelegramBot(token='your_token')
bot.start()

# 4. Monitor and send alerts
scheduler.on_news(lambda news: correlator.analyze(news))
correlator.on_significant_event(lambda event: bot.send_alert(event))
```

---

## 🌐 REST API Endpoints

Start the API server:
```bash
python api/rest/api_server.py
```

Available endpoints:
```
GET  /api/analyze/<symbol>           # Complete stock analysis
GET  /api/news/<symbol>              # Get news for symbol
POST /api/predict                    # Make predictions
GET  /api/options/<symbol>           # Get options data
POST /api/chat                       # AI assistant chat
GET  /api/technical/<symbol>         # Technical indicators
GET  /api/sentiment/<symbol>         # Sentiment analysis
```

Example:
```bash
# Get complete analysis
curl http://localhost:5000/api/analyze/AAPL

# Get news
curl http://localhost:5000/api/news/AAPL

# Make prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days": 5}'
```

---

## 📖 Import Reference

### Old → New Import Paths

| Old Import | New Import |
|-----------|-----------|
| `from stock_analyzer.news_api_client import ...` | `from news_system.collectors.news_api_client import ...` |
| `from stock_analyzer.sentiment_analyzer import ...` | `from news_system.analyzers.sentiment_analyzer import ...` |
| `from stock_analyzer.data_processor import ...` | `from prediction_engine.technical_analysis.data_processor import ...` |
| `from phase4_ml_engine.prediction_engine import ...` | `from prediction_engine.predictors.prediction_engine import ...` |
| `from phase4_ml_engine.options_analyzer import ...` | `from options_analyzer.analyzers.options_analyzer import ...` |
| `from phase4_ml_engine.automated_options_bot import ...` | `from options_analyzer.bots.automated_options_bot import ...` |
| `from stock_analyzer.utils import ...` | `from core.utils.utils import ...` |
| `from stock_analyzer.config import ...` | `from core.config.config import ...` |

---

## 🔧 Troubleshooting

### Import Errors
If you get import errors, run the import updater:
```bash
python update_imports.py
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### API Key Issues
Check your `.env` file has all required keys:
```bash
python -c "from core.config.config import Config; Config().validate()"
```

### Database Issues
For news system database:
```bash
python -c "from news_system.database.migrations import run_migrations; run_migrations()"
```

---

## 📝 Next Steps

1. ✅ Review the new structure - everything is organized by function
2. ✅ Update any custom scripts using `update_imports.py`
3. ✅ Run tests to verify everything works
4. ✅ Configure your API keys in `.env`
5. ✅ Start using the modular components!

---

## 📚 Full Documentation

- **Complete README**: [README_NEW_STRUCTURE.md](README_NEW_STRUCTURE.md)
- **Original docs**: Check `stock_analyzer/` and `stock-prediction-engine/` (preserved for reference)

---

## ❓ Need Help?

- Check module-specific README files (coming soon)
- Review test files for usage examples
- Open an issue on GitHub

---

**Happy Trading! 🚀📈**
