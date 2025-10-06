#!/usr/bin/env python3
"""
Telegram Signal Bot for Indian Trading
Handles all Telegram communications and formatting
Compatible with python-telegram-bot v13.x (stable version)
"""

import logging
import time
from typing import Dict, List, Optional

# Use stable v13.x version for synchronous operations
try:
    import telegram
    from telegram import Bot, ParseMode
    from telegram.error import TelegramError, NetworkError, TimedOut, RetryAfter
    Bot = telegram.Bot
    PARSE_MODE_MARKDOWN = ParseMode.MARKDOWN
    print("Using python-telegram-bot v13.x")
except ImportError:
    print("ERROR: python-telegram-bot not installed properly!")
    print("Please run: pip install python-telegram-bot==13.15")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TelegramSignalBot:
    """Telegram bot for sending trading signals"""
    
    def __init__(self, bot_token: str, chat_id: str):
        """Initialize Telegram bot"""
        
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.last_message_time = 0
        self.message_count = 0
        
        try:
            # Initialize bot
            self.bot = Bot(token=bot_token)
            
            # Test connection
            self._test_connection()
            
            logger.info("[OK] Telegram bot initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            raise
    
    def _test_connection(self):
        """Test Telegram connection"""
        
        try:
            bot_info = self.bot.get_me()
            logger.info(f"Connected to Telegram as @{bot_info.username}")
            return True
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False
    
    def _apply_rate_limiting(self):
        """Apply rate limiting to avoid Telegram API limits"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.last_message_time > 60:
            self.message_count = 0
        
        # If sending too many messages, add delay
        if self.message_count >= 19:  # Stay under 20 messages/minute limit
            sleep_time = 60 - (current_time - self.last_message_time)
            if sleep_time > 0:
                logger.info(f"Rate limiting: sleeping for {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
            self.message_count = 0
        
        # Also ensure we don't exceed 30 messages/second
        time_since_last = current_time - self.last_message_time
        if time_since_last < 0.034:  # ~30 messages/second
            time.sleep(0.034 - time_since_last)
        
        self.last_message_time = time.time()
        self.message_count += 1
    
    def send_message(self, message: str, silent: bool = False, parse_mode: Optional[str] = PARSE_MODE_MARKDOWN) -> bool:
        """Send message via Telegram with better error handling"""
        
        try:
            # Split long messages first
            messages = self._split_message(message)
            
            for msg in messages:
                # Apply rate limiting
                self._apply_rate_limiting()
                
                # Send message with or without parse_mode
                if parse_mode is None:
                    self.bot.send_message(
                        chat_id=self.chat_id,
                        text=msg,
                        disable_notification=silent,
                        disable_web_page_preview=True
                    )
                else:
                    self.bot.send_message(
                        chat_id=self.chat_id,
                        text=msg,
                        parse_mode=parse_mode,
                        disable_notification=silent,
                        disable_web_page_preview=True
                    )
                
                logger.info(f"📱 Message sent successfully")
            
            return True
            
        except RetryAfter as e:
            # Handle rate limit errors with exponential backoff
            logger.warning(f"Rate limited by Telegram. Retrying after {e.retry_after} seconds")
            time.sleep(e.retry_after)
            return self.send_message(message, silent, parse_mode)
            
        except NetworkError as e:
            logger.error(f"Network error sending message: {e}")
            return self._retry_send(message, silent, parse_mode)
            
        except TimedOut as e:
            logger.error(f"Telegram request timed out: {e}")
            return self._retry_send(message, silent, parse_mode)
            
        except TelegramError as e:
            logger.error(f"Telegram error sending message: {e}")
            # Try sending without formatting if markdown fails
            if parse_mode and "can't parse entities" in str(e).lower():
                logger.info("Retrying without markdown formatting")
                return self.send_message(message, silent, parse_mode=None)
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error sending message: {e}")
            return False
    
    def send_trade_signal(self, signal_message: str) -> bool:
        """Send a trade signal with special formatting"""
        
        try:
            # Add signal emoji and formatting
            formatted_message = f"📢 **TRADE SIGNAL**\n\n{signal_message}"
            
            # Send with notification (not silent)
            return self.send_message(formatted_message, silent=False)
            
        except Exception as e:
            logger.error(f"Failed to send trade signal: {e}")
            return False
    
    def send_error_alert(self, error_message: str) -> bool:
        """Send error alert"""
        
        try:
            formatted_message = f"[ERROR] **ERROR ALERT**\n\n{error_message}\n\nPlease check logs."
            
            # Send error alerts with high priority (not silent)
            return self.send_message(formatted_message, silent=False)
            
        except Exception as e:
            logger.error(f"Failed to send error alert: {e}")
            return False
    
    def send_photo(self, photo_path: str, caption: str = "") -> bool:
        """Send a photo (for charts)"""
        
        try:
            # Apply rate limiting
            self._apply_rate_limiting()
            
            with open(photo_path, 'rb') as photo:
                self.bot.send_photo(
                    chat_id=self.chat_id,
                    photo=photo,
                    caption=caption[:1024],  # Telegram caption limit
                    parse_mode=PARSE_MODE_MARKDOWN
                )
            
            logger.info("Photo sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send photo: {e}")
            return False
    
    def send_document(self, file_path: str, caption: str = "") -> bool:
        """Send a document (for reports)"""
        
        try:
            # Apply rate limiting
            self._apply_rate_limiting()
            
            with open(file_path, 'rb') as doc:
                self.bot.send_document(
                    chat_id=self.chat_id,
                    document=doc,
                    caption=caption[:1024],  # Telegram caption limit
                    parse_mode=PARSE_MODE_MARKDOWN
                )
            
            logger.info("Document sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send document: {e}")
            return False
    
    def _split_message(self, message: str, max_length: int = 4000) -> List[str]:
        """Split long messages for Telegram's character limit"""
        
        if len(message) <= max_length:
            return [message]
        
        messages = []
        lines = message.split('\n')
        current_message = ""
        
        for line in lines:
            # If adding this line would exceed limit
            if len(current_message) + len(line) + 1 > max_length:
                # Save current message if it has content
                if current_message.strip():
                    messages.append(current_message.strip())
                # Start new message with current line
                current_message = line + '\n'
            else:
                current_message += line + '\n'
        
        # Don't forget the last message
        if current_message.strip():
            messages.append(current_message.strip())
        
        # If no messages were created (single very long line), force split
        if not messages and message:
            for i in range(0, len(message), max_length):
                messages.append(message[i:i+max_length])
        
        return messages
    
    def _retry_send(self, message: str, silent: bool, parse_mode: Optional[str], max_retries: int = 3) -> bool:
        """Retry sending message on network errors with exponential backoff"""
        
        for attempt in range(max_retries):
            try:
                # Exponential backoff: 2, 4, 8 seconds
                sleep_time = 2 ** (attempt + 1)
                logger.info(f"Retrying send in {sleep_time} seconds (attempt {attempt + 1}/{max_retries})")
                time.sleep(sleep_time)
                
                # Apply rate limiting
                self._apply_rate_limiting()
                
                if parse_mode is None:
                    self.bot.send_message(
                        chat_id=self.chat_id,
                        text=message,
                        disable_notification=silent,
                        disable_web_page_preview=True
                    )
                else:
                    self.bot.send_message(
                        chat_id=self.chat_id,
                        text=message,
                        parse_mode=parse_mode,
                        disable_notification=silent,
                        disable_web_page_preview=True
                    )
                
                logger.info(f"Message sent on retry attempt {attempt + 1}")
                return True
                
            except RetryAfter as e:
                # If we get rate limited, wait the required time
                logger.warning(f"Rate limited on retry. Waiting {e.retry_after} seconds")
                time.sleep(e.retry_after)
                continue
                
            except (NetworkError, TimedOut) as e:
                logger.error(f"Retry {attempt + 1} failed with network error: {e}")
                continue
                
            except Exception as e:
                logger.error(f"Retry {attempt + 1} failed: {e}")
                continue
        
        logger.error(f"Failed to send message after {max_retries} retries")
        return False
    
    def format_signal_summary(self, signals: List[Dict]) -> str:
        """Format multiple signals into a summary"""
        
        if not signals:
            return "No signals found in this scan."
        
        lines = ["[CHART] **SIGNAL SUMMARY**", ""]
        
        # Sort signals by confidence
        signals_sorted = sorted(signals, key=lambda x: x.get('confidence', 0), reverse=True)
        
        for signal in signals_sorted[:20]:  # Limit to top 20 to avoid message size issues
            confidence = signal.get('confidence', 0)
            confidence_emoji = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"
            
            ticker = signal.get('ticker', 'Unknown')
            strategy = signal.get('strategy', 'Unknown')
            entry_price = signal.get('entry_price', 0)
            
            lines.append(
                f"{confidence_emoji} **{ticker}** - {strategy}\n"
                f"   Confidence: {confidence:.0%} | Entry: ₹{entry_price:.2f}"
            )
        
        # Add summary statistics
        total_signals = len(signals)
        high_conf = sum(1 for s in signals if s.get('confidence', 0) >= 0.8)
        med_conf = sum(1 for s in signals if 0.6 <= s.get('confidence', 0) < 0.8)
        low_conf = sum(1 for s in signals if s.get('confidence', 0) < 0.6)
        
        lines.extend([
            "",
            "[UP] **Statistics:**",
            f"Total signals: {total_signals}",
            f"High confidence (≥80%): {high_conf}",
            f"Medium confidence (60-79%): {med_conf}",
            f"Low confidence (<60%): {low_conf}"
        ])
        
        if total_signals > 20:
            lines.append(f"\n_Showing top 20 of {total_signals} signals_")
        
        return "\n".join(lines)
    
    def format_position_update(self, position: Dict) -> str:
        """Format position update message"""
        
        ticker = position.get('ticker', 'Unknown')
        strategy = position.get('strategy', 'Unknown')
        entry_price = position.get('entry_price', 0)
        current_price = position.get('current_price', 0)
        pnl = position.get('unrealized_pnl', 0)
        pnl_percent = position.get('pnl_percent', 0)
        days_held = position.get('days_held', 0)
        
        emoji = "[UP]" if pnl > 0 else "[DOWN]" if pnl < 0 else "➡️"
        
        lines = [
            f"{emoji} **POSITION UPDATE - {ticker}**",
            "",
            f"Strategy: {strategy}",
            f"Entry: ₹{entry_price:.2f}",
            f"Current: ₹{current_price:.2f}",
            f"P&L: ₹{pnl:+,.2f} ({pnl_percent:+.1f}%)",
            f"Days held: {days_held}",
            ""
        ]
        
        if position.get('action'):
            action_emoji = "[BELL]" if position['action'] == 'HOLD' else "[WARNING]"
            lines.append(f"{action_emoji} **Action: {position['action']}**")
        
        if position.get('stop_loss'):
            lines.append(f"Stop Loss: ₹{position['stop_loss']:.2f}")
        
        if position.get('target'):
            lines.append(f"Target: ₹{position['target']:.2f}")
        
        return "\n".join(lines)
    
    def send_market_alert(self, alert_type: str, details: Dict = None) -> bool:
        """Send market-wide alerts"""
        
        alert_messages = {
            'circuit_breaker': "🚨 **CIRCUIT BREAKER** - Market halted!",
            'high_volatility': "[WARNING] **HIGH VOLATILITY** - VIX spike detected",
            'news_alert': "📰 **BREAKING NEWS** - Market moving event",
            'technical_breakout': "🔥 **TECHNICAL BREAKOUT** - Major level broken",
            'volume_spike': "[CHART] **VOLUME SPIKE** - Unusual activity detected",
            'market_open': "[BELL] **MARKET OPEN** - Trading session started",
            'market_close': "[BELL] **MARKET CLOSE** - Trading session ended",
            'pre_market': "📍 **PRE-MARKET UPDATE** - Futures and global markets"
        }
        
        message = alert_messages.get(alert_type, "📢 **MARKET ALERT**")
        
        if details:
            message += "\n\n**Details:**\n"
            for key, value in details.items():
                # Format key nicely
                formatted_key = key.replace('_', ' ').title()
                message += f"• {formatted_key}: {value}\n"
        
        message += f"\n_Alert Time: {time.strftime('%H:%M:%S')}_"
        
        # Market alerts are high priority
        return self.send_message(message, silent=False)
    
    def send_daily_summary(self, summary_data: Dict) -> str:
        """Send daily trading summary"""
        
        lines = [
            "[CHART] **DAILY TRADING SUMMARY**",
            f"_Date: {time.strftime('%Y-%m-%d')}_",
            "",
            "**Performance:**"
        ]
        
        if 'total_pnl' in summary_data:
            pnl = summary_data['total_pnl']
            pnl_emoji = "💚" if pnl > 0 else "💔" if pnl < 0 else "💛"
            lines.append(f"{pnl_emoji} Total P&L: ₹{pnl:+,.2f}")
        
        if 'win_rate' in summary_data:
            lines.append(f"[UP] Win Rate: {summary_data['win_rate']:.1f}%")
        
        if 'total_trades' in summary_data:
            lines.append(f"🔄 Total Trades: {summary_data['total_trades']}")
        
        if 'best_performer' in summary_data:
            lines.extend([
                "",
                "**Best Performer:**",
                f"🏆 {summary_data['best_performer']['ticker']}: "
                f"₹{summary_data['best_performer']['pnl']:+,.2f} "
                f"({summary_data['best_performer']['pnl_percent']:+.1f}%)"
            ])
        
        if 'worst_performer' in summary_data:
            lines.extend([
                "",
                "**Worst Performer:**",
                f"[DOWN] {summary_data['worst_performer']['ticker']}: "
                f"₹{summary_data['worst_performer']['pnl']:+,.2f} "
                f"({summary_data['worst_performer']['pnl_percent']:+.1f}%)"
            ])
        
        if 'open_positions' in summary_data:
            lines.extend([
                "",
                f"**Open Positions:** {summary_data['open_positions']}"
            ])
        
        message = "\n".join(lines)
        return self.send_message(message, silent=False)


# Quick test function
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if bot_token and chat_id:
        print(f"Testing bot with token: {bot_token[:20]}...")
        print(f"Chat ID: {chat_id}")
        
        try:
            bot = TelegramSignalBot(bot_token, chat_id)
            
            # Test basic message
            if bot.send_message("[OK] Telegram bot is working! This is a test message."):
                print("[OK] Test message sent successfully!")
            
            # Test formatted message
            test_signal = {
                'ticker': 'RELIANCE',
                'strategy': 'Momentum',
                'confidence': 0.85,
                'entry_price': 2450.50
            }
            
            if bot.send_trade_signal(f"Buy {test_signal['ticker']} at ₹{test_signal['entry_price']:.2f}"):
                print("[OK] Trade signal test successful!")
            
            # Test market alert
            if bot.send_market_alert('market_open', {'index': 'NIFTY50', 'opening': 19245.50}):
                print("[OK] Market alert test successful!")
                
        except Exception as e:
            print(f"[ERROR] Test failed: {e}")
    else:
        print("[ERROR] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env file")
        print("Please add these to your .env file:")
        print("TELEGRAM_BOT_TOKEN=your_bot_token_here")
        print("TELEGRAM_CHAT_ID=your_chat_id_here")