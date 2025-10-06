#!/usr/bin/env python3
"""
Test script to verify Telegram bot connectivity
Run this to make sure your bot can send messages properly
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import your telegram bot
try:
    from telegram_bot import TelegramSignalBot
except ImportError:
    print("❌ Error: Could not import telegram_bot.py")
    print("Make sure telegram_bot.py is in the same directory")
    sys.exit(1)

def test_telegram_bot():
    """Test all Telegram bot functions"""
    
    print("🔧 Starting Telegram Bot Test...")
    print("-" * 50)
    
    # Get credentials from environment
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    # Check if credentials exist
    if not bot_token or not chat_id:
        print("❌ Error: Missing credentials in .env file")
        print(f"TELEGRAM_BOT_TOKEN: {'✅ Found' if bot_token else '❌ Missing'}")
        print(f"TELEGRAM_CHAT_ID: {'✅ Found' if chat_id else '❌ Missing'}")
        return False
    
    print(f"✅ Bot Token: {bot_token[:20]}...")
    print(f"✅ Chat ID: {chat_id}")
    print("-" * 50)
    
    try:
        # Initialize bot
        print("📱 Initializing Telegram bot...")
        bot = TelegramSignalBot(bot_token=bot_token, chat_id=chat_id)
        print("✅ Bot initialized successfully!")
        
        # Test 1: Simple message
        print("\n📤 Test 1: Sending simple message...")
        success = bot.send_message(
            "🎉 **Test Message 1**\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "Your Telegram bot is working!"
        )
        print(f"Result: {'✅ Success' if success else '❌ Failed'}")
        
        # Test 2: Trade signal format
        print("\n📤 Test 2: Sending trade signal format...")
        test_signal = """
🚨 **TEST SIGNAL** - RELIANCE
📊 Strategy: **BULLISH_CALL_SPREAD**
💪 Confidence: 85% ⭐⭐⭐

🎯 **TRADE SETUP**:
BUY RELIANCE 2024-01-25 2600 CALL @ ₹45.50
SELL RELIANCE 2024-01-25 2650 CALL @ ₹22.30
Net Debit: ₹23.20 per share
Lot Size: 250 shares

💰 **PAYOFF ANALYSIS**:
Max Profit: ₹6,700 (115.5%)
Max Loss: ₹5,800 (100.0%)
Breakeven: ₹2,623.20

⚠️ This is a TEST signal only!
        """
        success = bot.send_trade_signal(test_signal)
        print(f"Result: {'✅ Success' if success else '❌ Failed'}")
        
        # Test 3: Error alert
        print("\n📤 Test 3: Sending error alert...")
        success = bot.send_error_alert("This is a test error message")
        print(f"Result: {'✅ Success' if success else '❌ Failed'}")
        
        # Test 4: Market alert
        print("\n📤 Test 4: Sending market alert...")
        success = bot.send_market_alert(
            'volume_spike',
            {
                'Stock': 'RELIANCE',
                'Volume': '5x normal',
                'Price Change': '+2.5%'
            }
        )
        print(f"Result: {'✅ Success' if success else '❌ Failed'}")
        
        # Test 5: Silent message
        print("\n📤 Test 5: Sending silent message...")
        success = bot.send_message(
            "🔇 This is a silent test message (no notification)",
            silent=True
        )
        print(f"Result: {'✅ Success' if success else '❌ Failed'}")
        
        # Test 6: Long message (test splitting)
        print("\n📤 Test 6: Sending long message...")
        long_message = "📝 **Long Message Test**\n\n"
        long_message += "This is a test of message splitting.\n" * 100
        long_message += "\nEnd of long message."
        success = bot.send_message(long_message)
        print(f"Result: {'✅ Success' if success else '❌ Failed'}")
        
        # Final summary
        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        print("Check your Telegram for the test messages.")
        print("=" * 50)
        
        # Send completion message
        bot.send_message(
            "✅ **All Tests Completed!**\n\n"
            "Your Indian Trading Bot's Telegram integration is working perfectly!\n\n"
            "You should have received:\n"
            "• Simple message\n"
            "• Trade signal\n"
            "• Error alert\n"
            "• Market alert\n"
            "• Silent message\n"
            "• Long message (split if needed)\n\n"
            "Ready to start trading! 🚀"
        )
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        print("\nPossible issues:")
        print("1. Invalid bot token (regenerate from @BotFather)")
        print("2. Wrong chat ID")
        print("3. Bot not started (send /start to your bot)")
        print("4. Network issues")
        return False

def quick_test():
    """Quick connection test only"""
    
    print("🚀 Quick Telegram Connection Test")
    print("-" * 30)
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("❌ Missing credentials in .env file")
        return
    
    try:
        from telegram_bot import TelegramSignalBot
        bot = TelegramSignalBot(bot_token=bot_token, chat_id=chat_id)
        
        success = bot.send_message(
            f"✅ Quick test successful!\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        
        if success:
            print("✅ Message sent successfully!")
        else:
            print("❌ Failed to send message")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    else:
        test_telegram_bot()