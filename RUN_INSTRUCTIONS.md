# How to Run TradeLens Applications

## ✅ Import Issues Fixed!

All scripts have been updated with proper path configuration. You can now run them from anywhere.

---

## 🚀 Running Applications

### 1. **Main Stock Analyzer**
Run from project root:
```bash
cd D:\tradelens
python main.py AAPL
```

Or run directly:
```bash
python D:\tradelens\main.py AAPL
```

---

### 2. **REST API Server**
Run from anywhere:
```bash
python D:\tradelens\api\rest\api_server.py
```

Or from api/rest directory:
```bash
cd D:\tradelens\api\rest
python api_server.py
```

**Access API at:** `http://localhost:5000`

**Endpoints:**
- `GET /api/analyze/<symbol>` - Complete stock analysis
- `GET /api/news/<symbol>` - Get news for symbol
- `POST /api/predict` - Make predictions
- `GET /api/options/<symbol>` - Get options data

---

### 3. **Automated Options Trading Bot**
Run from anywhere:
```bash
python D:\tradelens\options_analyzer\bots\automated_options_bot.py
```

Or from bots directory:
```bash
cd D:\tradelens\options_analyzer\bots
python automated_options_bot.py
```

---

### 4. **Indian Trading Bot**
Run from anywhere:
```bash
python D:\tradelens\options_analyzer\bots\indian_trading_bot.py
```

Or from bots directory:
```bash
cd D:\tradelens\options_analyzer\bots
python indian_trading_bot.py
```

---

### 5. **Telegram Bot**
Run from anywhere:
```bash
python D:\tradelens\api\telegram\telegram_bot.py
```

---

## 📝 How It Works

Each main entry point file now includes:

```python
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent  # Adjust depth as needed
sys.path.insert(0, str(project_root))

# Now imports work!
from news_system.collectors.news_api_client import NewsAPIClient
from prediction_engine.technical_analysis.data_processor import StockDataProcessor
# etc...
```

This allows Python to find the modules regardless of where you run the script from.

---

## 🧪 Running Tests

Always run tests from project root:

```bash
cd D:\tradelens

# All tests
python -m pytest tests/

# Specific module
python -m pytest tests/news_system/
python -m pytest tests/prediction_engine/
python -m pytest tests/options_analyzer/

# With verbose output
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/news_system/test_sentiment_integration.py -v
```

---

## 🔧 Using as a Module

If you want to import TradeLens in other projects:

```python
import sys
sys.path.append('D:/tradelens')

from news_system.collectors.news_api_client import NewsAPIClient
from prediction_engine.technical_analysis.data_processor import StockDataProcessor
from options_analyzer.analyzers.options_analyzer import ZerodhaEnhancedOptionsAnalyzer
```

---

## 🌐 Environment Setup

Make sure you have configured `.env`:

```bash
# Copy example if not done
cp .env.example .env

# Edit with your API keys
# NEWSAPI_KEY=your_key
# OPENAI_API_KEY=your_key
# ZERODHA_API_KEY=your_key
# ZERODHA_API_SECRET=your_secret
# TELEGRAM_BOT_TOKEN=your_token
```

---

## 📋 Common Commands

### **Stock Analysis**
```bash
# Single stock
python main.py AAPL

# Multiple stocks
python main.py AAPL GOOGL MSFT

# With date range
python main.py AAPL --start-date 2024-01-01 --end-date 2024-12-31
```

### **API Server**
```bash
# Start server
python api/rest/api_server.py

# Test endpoints
curl http://localhost:5000/api/analyze/AAPL
curl http://localhost:5000/api/news/AAPL
```

### **Trading Bots**
```bash
# Automated options bot
python options_analyzer/bots/automated_options_bot.py

# Indian trading bot
python options_analyzer/bots/indian_trading_bot.py

# With specific config
python options_analyzer/bots/automated_options_bot.py --config bot_config.json
```

---

## ⚠️ Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'options_analyzer'`
**Solution:** The script now auto-adds project root to path. Just run it again.

### Issue: `ImportError: cannot import name 'X'`
**Solution:** Check the class name. For example:
- `StockDataProcessor` (not `DataProcessor`)
- `WorkingFinancialSentimentAnalyzer` (not `SentimentAnalyzer`)
- `ZerodhaEnhancedOptionsAnalyzer` (not `OptionsAnalyzer`)

### Issue: API keys not found
**Solution:** Make sure `.env` file exists and has valid keys:
```bash
cat .env  # Check if file exists
# Edit if needed
```

### Issue: Port already in use (Flask)
**Solution:** Kill the process or use a different port:
```bash
# Use different port
python api/rest/api_server.py --port 5001
```

---

## 🎯 Quick Start Checklist

- [ ] Activate virtual environment (if using one)
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Configure `.env` file with API keys
- [ ] Run from project root: `cd D:\tradelens`
- [ ] Test import: `python -c "import news_system, prediction_engine, options_analyzer"`
- [ ] Run your desired script

---

## 📚 Module Import Reference

### **News System**
```python
from news_system.collectors.news_api_client import NewsAPIClient
from news_system.analyzers.sentiment_analyzer import WorkingFinancialSentimentAnalyzer
from news_system.processors.event_extractor import FinancialEventExtractor
from news_system.database.connection import get_db_connection
```

### **Prediction Engine**
```python
from prediction_engine.technical_analysis.data_processor import StockDataProcessor
from prediction_engine.predictors.prediction_engine import PredictionEngine
from prediction_engine.features.feature_engineering import FeatureEngineer
from prediction_engine.data_loaders.alpha_vantage_client import AlphaVantageClient
```

### **Options Analyzer**
```python
from options_analyzer.analyzers.options_analyzer import ZerodhaEnhancedOptionsAnalyzer
from options_analyzer.brokers.zerodha_api_client import ZerodhaAPIClient
from options_analyzer.bots.automated_options_bot import AutomatedOptionsBot
from options_analyzer.indian_market.ind_data_processor import IndianMarketProcessor
```

### **Core**
```python
from core.config.config import Config, config
from core.utils.utils import setup_logging, get_logger
from core.validators.validators import validate_ticker, validate_date_range
```

---

## 🚀 You're Ready to Go!

All import issues are fixed. Run your applications from anywhere using the commands above.

**Need help?** Check:
- [README_NEW_STRUCTURE.md](README_NEW_STRUCTURE.md) - Complete documentation
- [QUICK_START.md](QUICK_START.md) - Quick examples
- [IMPORT_FIXES_COMPLETE.md](IMPORT_FIXES_COMPLETE.md) - Import fixes summary
