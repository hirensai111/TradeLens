# TradeLens AI Assistant

A context-aware AI financial assistant chat feature for stock trading dashboards. Provides real-time, personalized guidance integrated with your existing stock analyzer.

## Features

- 🤖 **AI-Powered Chat**: OpenAI-powered responses with financial expertise
- 📊 **Real-Time Context**: Integrates with stock data (RSI, MACD, price, volume)
- 🎯 **Smart Suggestions**: Context-aware quick action buttons
- 📱 **Responsive Design**: Modern floating chat widget
- 🔒 **Session Management**: Maintains conversation context
- ⚡ **Fast Integration**: Modular design for easy dashboard integration

## Quick Start

### 1. Install Dependencies

```bash
pip install flask flask-cors openai requests
```

### 2. Set OpenAI API Key

```bash
# Windows
set OPENAI_API_KEY=your-api-key-here

# Linux/Mac
export OPENAI_API_KEY=your-api-key-here
```

### 3. Start the AI Backend

```bash
python ai_assistant/ai_backend.py
```

The backend will start on `http://localhost:5001`

### 4. Integrate with Your Dashboard

Add the chat widget to any HTML page:

```html
<!-- Add to your page -->
<link rel="stylesheet" href="ai_assistant/chat_styles.css">
<script src="ai_assistant/chat_handler.js"></script>

<!-- Include the chat interface -->
<div id="chat-widget" class="chat-widget hidden">
    <!-- Content from chat_interface.html -->
</div>
```

## File Structure

```
ai_assistant/
├── chat_interface.html      # Chat UI component
├── chat_styles.css         # Modern styling
├── chat_handler.js         # Frontend logic
├── ai_backend.py          # Flask API endpoint
├── context_manager.py     # Stock data integration
├── prompt_templates.py    # AI response formatting
├── test_ai_backend.py     # Test suite
└── README.md             # This file
```

## API Endpoints

### POST /api/chat
Send message to AI assistant.

**Request:**
```json
{
  "message": "What's the trend?",
  "context": {
    "symbol": "AAPL",
    "price": 220.45,
    "rsi": 65.2,
    "dayChange": 2.1,
    "volume": 52000000
  },
  "history": []
}
```

**Response:**
```json
{
  "response": "AAPL at $220.45 is showing bullish momentum with 2.1% gain today. RSI at 65.2 indicates healthy strength without being overbought.",
  "session_id": "session_20240327_143022",
  "message_count": 1
}
```

### GET /api/chat/status
Check service status and capabilities.

### GET /api/chat/history
Retrieve chat history for current session.

### POST /api/chat/clear
Clear chat history for current session.

## Integration with Existing Stock Analyzer

The context manager automatically integrates with your existing `data_processor.py`:

```python
# In context_manager.py
from data_processor import StockDataProcessor

# Pulls real RSI, MACD, price data
context = context_manager.get_enhanced_context("AAPL")
```

If the stock analyzer isn't available, it falls back to simulated data for testing.

## Configuration

### Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key (required for AI responses)
- `SECRET_KEY` - Flask session secret (optional, has default)

### Customization

#### AI Response Style
Edit `prompt_templates.py` to modify:
- Response length (default: 3 sentences max)
- Tone and language style
- Technical explanation depth

#### Chat Appearance
Edit `chat_styles.css` to customize:
- Colors and branding
- Widget size and position
- Animation effects

#### Quick Actions
Modify the quick action buttons in `chat_handler.js`:
```javascript
// Update suggestions based on stock context
getSmartSuggestions(context) {
    if (context.rsi > 70) {
        return ["Is it overbought?", "Risk analysis", ...];
    }
    // ... your custom logic
}
```

## Testing

Run the test suite to verify everything works:

```bash
# Start the backend first
python ai_assistant/ai_backend.py

# In another terminal, run tests
python ai_assistant/test_ai_backend.py
```

Test coverage includes:
- ✅ API endpoint functionality
- ✅ Context integration
- ✅ Response quality and length
- ✅ Session management
- ✅ Quick action responses

## Example Interactions

**User:** "What's happening with this stock?"
**AI:** "AAPL is showing strength at $220 with RSI at 62 - good momentum without being overheated. Volume is 20% above average, suggesting real interest from buyers."

**User:** "Should I buy?"
**AI:** "The technicals look positive with MACD trending up. If you have $1000, consider starting with $300-400. Set a stop-loss at $215 (recent support level)."

**User:** "Explain RSI in simple terms"
**AI:** "RSI at 62 is like a speedometer showing the stock's momentum. Under 30 is slow (oversold), over 70 is fast (overbought). Your stock is at a nice cruising speed right now."

## Error Handling

The system gracefully handles:
- ✅ API failures (falls back to simulated responses)
- ✅ Rate limiting (max 50 messages per session)
- ✅ Invalid stock symbols
- ✅ Network connectivity issues
- ✅ Missing context data

## Security Features

- 🔒 Session-based rate limiting
- 🔒 Input sanitization
- 🔒 No financial advice (educational only)
- 🔒 CORS protection
- 🔒 API key protection

## Browser Support

- ✅ Chrome, Firefox, Safari (latest versions)
- ✅ Mobile responsive design
- ✅ Touch-friendly interface

## Contributing

To extend the AI assistant:

1. **Add new indicators**: Update `context_manager.py` to include new data
2. **Modify responses**: Edit `prompt_templates.py` for new response patterns
3. **UI changes**: Update `chat_styles.css` and `chat_interface.html`
4. **New endpoints**: Add routes to `ai_backend.py`

## Troubleshooting

### Common Issues

**Chat widget not appearing:**
- Check console for JavaScript errors
- Ensure CSS and JS files are properly linked

**AI not responding:**
- Verify OpenAI API key is set
- Check backend is running on port 5001
- Review browser network tab for failed requests

**Context data is empty:**
- Check integration with your stock analyzer
- Verify `data_processor.py` is accessible
- Test with simulated data first

**Responses are too generic:**
- Ensure context data is being passed correctly
- Check prompt templates for your use case
- Verify symbol parameter is being sent

### Getting Help

1. Run the test suite to identify issues
2. Check browser console for errors
3. Review backend logs for API errors
4. Test with simulated data to isolate problems

## License

This component is part of the Stock Analyzer project and follows the same licensing terms.