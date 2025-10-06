# TradeLens - Reorganized Project Structure

## 📁 Project Organization

This project has been reorganized into a clean, modular structure with three main functional areas:

```
tradelens/
├── 📰 news_system/          # News collection, processing & sentiment analysis
├── 🤖 prediction_engine/    # ML models & stock prediction
├── 📊 options_analyzer/     # Options trading & analysis
├── 🔧 core/                 # Shared utilities & configuration
├── 🌐 api/                  # API servers & interfaces
├── 📄 output/               # Report generators
├── 🤝 ai_assistant/         # AI chat functionality
├── 🧪 tests/                # All test files
├── 📦 data/                 # Data storage
└── 📝 logs/                 # Log files
```

---

## 📰 News System (`news_system/`)

Complete news intelligence pipeline for market analysis.

### Structure:
```
news_system/
├── collectors/              # News data collection
│   ├── adapters/           # API adapters (NewsAPI, Polygon, YFinance)
│   ├── newsapi_collector.py
│   ├── reddit_collector.py
│   ├── hybrid_collector.py
│   ├── news_api_client.py  # Main news orchestrator
│   ├── news_integration_bridge.py
│   ├── api_config.py
│   ├── scheduler.py
│   └── base_collector.py
├── processors/              # News processing
│   ├── sentiment_analyzer.py  # Advanced financial sentiment
│   ├── text_processor.py      # Text cleaning
│   ├── deduplicator.py        # Remove duplicates
│   └── event_extractor.py     # Extract financial events
├── analyzers/              # News analysis
│   ├── event_analyzer.py      # GPT-powered event analysis
│   ├── sentiment_analyzer.py  # VADER + custom lexicon
│   ├── sentiment_integration.py
│   └── correlation_analyzer.py  # News-price correlation
└── database/               # News storage
    ├── models/
    │   └── news_models.py  # SQLAlchemy models
    ├── connection.py
    ├── session.py
    └── migrations.py
```

### Key Features:
- **Multi-source news collection** (NewsAPI, Polygon, Reddit, Yahoo Finance)
- **Advanced sentiment analysis** with financial lexicon
- **Event extraction** and correlation with price movements
- **Deduplication** and text processing
- **Database persistence** with SQLAlchemy

### Usage:
```python
from news_system.collectors.news_api_client import NewsAPIClient
from news_system.analyzers.sentiment_analyzer import SentimentAnalyzer

# Collect news
news_client = NewsAPIClient()
articles = news_client.get_news("AAPL")

# Analyze sentiment
analyzer = SentimentAnalyzer()
sentiment = analyzer.analyze(articles[0]['content'])
```

---

## 🤖 Prediction Engine (`prediction_engine/`)

Machine learning models and stock prediction algorithms.

### Structure:
```
prediction_engine/
├── models/                 # ML model implementations
├── features/               # Feature engineering
│   └── feature_engineering.py
├── data_loaders/           # Data loading utilities
│   ├── excel_loader.py
│   ├── phase3_connector.py
│   ├── alpha_vantage_client.py
│   ├── alpha_vantage_config.py
│   └── test_loader.py
├── technical_analysis/     # Technical indicators
│   └── data_processor.py   # SMA, EMA, RSI, MACD, Bollinger, ATR, etc.
└── predictors/             # Prediction engines
    ├── prediction_engine.py    # Multi-day AI predictions
    ├── ultimate_ai_predictor.py
    ├── predict.py
    ├── graph_analyzer.py
    └── report_generator.py
```

### Key Features:
- **Technical indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic, OBV
- **Feature engineering** for ML models
- **Multi-day predictions** with confidence scores
- **Data loaders** for multiple sources (Excel, Alpha Vantage)
- **Graph analysis** and visualization

### Usage:
```python
from prediction_engine.predictors.prediction_engine import PredictionEngine
from prediction_engine.technical_analysis.data_processor import DataProcessor

# Load and process data
processor = DataProcessor()
df = processor.process_stock_data("AAPL")

# Make predictions
predictor = PredictionEngine()
predictions = predictor.predict_multi_day("AAPL", days=5)
```

---

## 📊 Options Analyzer (`options_analyzer/`)

Options trading, analysis, and automated bot functionality.

### Structure:
```
options_analyzer/
├── analyzers/              # Options analysis tools
│   ├── options_analyzer.py    # Greeks, pricing, volatility
│   └── check_nifty_options.py # Nifty options checker
├── bots/                   # Automated trading bots
│   ├── automated_options_bot.py  # Intraday options bot v2.0
│   └── indian_trading_bot.py     # Indian market bot
├── brokers/                # Broker integrations
│   ├── zerodha_api_client.py        # Zerodha Kite Connect
│   ├── zerodha_technical_analyzer.py
│   ├── groww_api_client.py          # Groww API
│   └── get_access_token.py          # OAuth management
└── indian_market/          # India-specific tools
    ├── ind_trade_logger.py
    └── ind_data_processor.py
```

### Key Features:
- **Options Greeks calculation** (Delta, Gamma, Theta, Vega, Rho)
- **Automated trading bots** with risk management
- **Broker integrations** (Zerodha, Groww)
- **Indian market support** (NSE, BSE, Nifty)
- **Technical analysis** for options
- **Telegram notifications** for signals

### Usage:
```python
from options_analyzer.analyzers.options_analyzer import OptionsAnalyzer
from options_analyzer.brokers.zerodha_api_client import ZerodhaClient

# Analyze options
analyzer = OptionsAnalyzer()
greeks = analyzer.calculate_greeks(
    spot=18500, strike=18600, volatility=0.15,
    time_to_expiry=7, risk_free_rate=0.06
)

# Connect to Zerodha
zerodha = ZerodhaClient(api_key="your_key", access_token="token")
positions = zerodha.get_positions()
```

---

## 🔧 Core (`core/`)

Shared utilities, configuration, and common functionality.

### Structure:
```
core/
├── config/                 # Configuration management
│   ├── config.py          # Stock analyzer config
│   └── settings.py        # Prediction engine settings
├── utils/                  # Utility functions
│   └── utils.py           # Logging, progress, cache, market hours
├── validators/             # Input validation
│   └── validators.py      # Data quality checks, API validation
└── cache/                  # Caching utilities
    └── __init__.py
```

### Key Features:
- **Centralized configuration** for all modules
- **Logging utilities** with progress tracking
- **Input validation** and data quality checks
- **Cache management** for API responses
- **Market hours** detection

### Usage:
```python
from core.config.config import Config
from core.utils.utils import setup_logging, is_market_open

# Load configuration
config = Config()

# Setup logging
logger = setup_logging("my_module")

# Check market hours
if is_market_open():
    logger.info("Market is open for trading")
```

---

## 🌐 API (`api/`)

REST API servers and communication interfaces.

### Structure:
```
api/
├── rest/                   # Flask REST API
│   ├── api_server.py      # Main Flask server
│   └── stock_analyzer_handler.py
└── telegram/               # Telegram bot
    └── telegram_bot.py    # Trading signals bot
```

### Key Features:
- **Flask REST API** with endpoints for:
  - Stock analysis
  - News retrieval
  - Predictions
  - Options data
  - Chat functionality
- **Telegram bot** for trading signals
- **CORS support** for web frontends
- **JSON response** formatting

### Endpoints:
```
GET  /api/analyze/<symbol>           # Analyze stock
GET  /api/news/<symbol>              # Get news
POST /api/predict                    # Make predictions
GET  /api/options/<symbol>           # Get options data
POST /api/chat                       # AI chat
```

---

## 📄 Output (`output/`)

Report generation and data export utilities.

### Structure:
```
output/
├── excel/                  # Excel report generation
│   └── excel_generator.py # Professional Excel reports with charts
├── json/                   # JSON export
│   └── json_exporter.py   # Dashboard data export
└── corresponding_prompts.py  # ChatGPT prompt templates
```

### Key Features:
- **Professional Excel reports** with:
  - Charts and visualizations
  - Multiple worksheets
  - Technical analysis
  - News sentiment
- **JSON export** for web dashboards
- **Customizable templates**

### Usage:
```python
from output.excel.excel_generator import ExcelGenerator
from output.json.json_exporter import JSONExporter

# Generate Excel report
excel_gen = ExcelGenerator()
excel_gen.generate_report(stock_data, "AAPL_report.xlsx")

# Export to JSON
json_exp = JSONExporter()
json_exp.export_dashboard_data(stock_data, "dashboard_data.json")
```

---

## 🤝 AI Assistant (`ai_assistant/`)

ChatGPT integration for natural language stock analysis.

### Structure:
```
ai_assistant/
├── ai_backend.py          # ChatGPT integration
├── context_manager.py     # Conversation context
└── prompt_templates.py    # Prompt templates
```

---

## 🧪 Tests (`tests/`)

Comprehensive test suite organized by module.

### Structure:
```
tests/
├── news_system/            # News system tests
│   ├── test_news_client.py
│   ├── test_sentiment_integration.py
│   ├── test_event_extractor.py
│   └── test_processing_pipeline.py
├── prediction_engine/      # Prediction engine tests
│   ├── test_directional_changes.py
│   └── test_technical_integration.py
├── options_analyzer/       # Options analyzer tests
│   ├── test_options.py
│   ├── test_bot.py
│   ├── test_zerodha.py
│   ├── test_groww.py
│   └── simple_bot_test.py
└── integration/            # Integration tests
    ├── test_full_pipeline.py
    ├── test_integration.py
    ├── test_api_server.py
    ├── test_complete_system.py
    ├── integration_test.py
    └── integration_test_suite.py
```

---

## 🚀 Getting Started

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running the Application

```bash
# Run main analysis
python main.py

# Start REST API server
python api/rest/api_server.py

# Run automated trading bot
python options_analyzer/bots/automated_options_bot.py

# Start Telegram bot
python api/telegram/telegram_bot.py
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific module tests
python -m pytest tests/news_system/
python -m pytest tests/prediction_engine/
python -m pytest tests/options_analyzer/

# Run integration tests
python -m pytest tests/integration/
```

---

## 📝 Configuration

### Environment Variables (.env)

```env
# News APIs
NEWSAPI_KEY=your_newsapi_key
POLYGON_API_KEY=your_polygon_key

# OpenAI
OPENAI_API_KEY=your_openai_key

# Broker APIs
ZERODHA_API_KEY=your_zerodha_key
ZERODHA_API_SECRET=your_zerodha_secret
GROWW_API_KEY=your_groww_key

# Telegram
TELEGRAM_BOT_TOKEN=your_telegram_token

# Alpha Vantage
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
```

---

## 📊 Key Improvements

### Before:
- ❌ Files scattered across `stock_analyzer` and `stock-prediction-engine`
- ❌ Duplicate functionality (sentiment, news collection, config)
- ❌ No clear module boundaries
- ❌ Tests mixed with source code

### After:
- ✅ **Clear module separation**: News, Prediction, Options
- ✅ **No duplication**: Consolidated overlapping functionality
- ✅ **Logical organization**: Related files grouped together
- ✅ **Centralized utilities**: Shared code in `core/`
- ✅ **Organized tests**: Separate test directories by module
- ✅ **Easy navigation**: Find what you need quickly

---

## 🔄 Migration Guide

### Old Structure → New Structure

| Old Location | New Location |
|-------------|-------------|
| `stock_analyzer/news_api_client.py` | `news_system/collectors/news_api_client.py` |
| `stock_analyzer/sentiment_analyzer.py` | `news_system/analyzers/sentiment_analyzer.py` |
| `stock_analyzer/data_processor.py` | `prediction_engine/technical_analysis/data_processor.py` |
| `phase4_ml_engine/prediction_engine.py` | `prediction_engine/predictors/prediction_engine.py` |
| `phase4_ml_engine/options_analyzer.py` | `options_analyzer/analyzers/options_analyzer.py` |
| `phase4_ml_engine/automated_options_bot.py` | `options_analyzer/bots/automated_options_bot.py` |
| `stock_analyzer/api_server.py` | `api/rest/api_server.py` |
| `stock_analyzer/utils.py` | `core/utils/utils.py` |
| `stock_analyzer/config.py` | `core/config/config.py` |

---

## 📚 Documentation

- **News System**: See [news_system/README.md](news_system/README.md)
- **Prediction Engine**: See [prediction_engine/README.md](prediction_engine/README.md)
- **Options Analyzer**: See [options_analyzer/README.md](options_analyzer/README.md)
- **API Documentation**: See [api/README.md](api/README.md)

---

## 🤝 Contributing

When adding new features:
1. Place files in the appropriate module directory
2. Add tests in the corresponding `tests/` subdirectory
3. Update the module's `__init__.py` if needed
4. Follow the existing structure and naming conventions

---

## 📧 Support

For issues or questions:
- Check the module-specific README files
- Review the test files for usage examples
- Open an issue on GitHub

---

## 📜 License

See LICENSE file for details.

---

**Note**: The old `stock_analyzer/` and `stock-prediction-engine/` directories are preserved for reference. You can safely delete them after verifying the new structure works correctly.
