#!/usr/bin/env python3
"""
Trade Logger for Indian Trading Bot - COMPLETE FIXED VERSION
Resolves all database locking issues with proper connection management
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import pytz
import time
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class IndianTradeLogger:
    """Log and track trading activities with improved connection management"""
    
    def __init__(self, db_path: str = "indian_trading_bot.db"):
        """Initialize trade logger with SQLite database"""
        
        self.db_path = db_path
        self.timezone = pytz.timezone('Asia/Kolkata')
        
        # Thread lock for database operations
        self._db_lock = threading.Lock()
        
        # Connection pool (simple implementation)
        self._connection = None
        
        # Initialize database
        self._init_database()
        
        logger.info("Database initialized successfully")
    
    @contextmanager
    def _get_db_connection(self):
        """Context manager for database connections with proper locking"""
        conn = None
        try:
            with self._db_lock:
                # Create a new connection for this operation
                conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
                conn.execute("PRAGMA busy_timeout = 30000")
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                conn.execute("PRAGMA cache_size = -2000")
                conn.execute("PRAGMA temp_store = MEMORY")
                
                # Enable row factory for better results
                conn.row_factory = sqlite3.Row
                
                yield conn
                
                # Commit any pending transactions
                if conn.in_transaction:
                    conn.commit()
                    
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                logger.warning(f"Database locked, retrying... {e}")
                time.sleep(0.5)  # Wait before retry
                # Retry once
                if conn:
                    conn.close()
                conn = sqlite3.connect(self.db_path, timeout=60.0, check_same_thread=False)
                conn.execute("PRAGMA busy_timeout = 60000")
                conn.execute("PRAGMA journal_mode = WAL")
                conn.row_factory = sqlite3.Row
                yield conn
            else:
                raise
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Create signals table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        ticker TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        entry_price REAL,
                        target_price REAL,
                        stop_loss REAL,
                        max_profit REAL,
                        max_loss REAL,
                        expiry_date TEXT,
                        signal_data TEXT,
                        status TEXT DEFAULT 'active',
                        processed_by_automation BOOLEAN DEFAULT 0,
                        automation_timestamp DATETIME,
                        source TEXT DEFAULT 'signal_generator'
                    )
                """)
                
                # Create indices for better performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_signals_processed 
                    ON signals(processed_by_automation, status)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_signals_timestamp 
                    ON signals(timestamp DESC)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_signals_ticker 
                    ON signals(ticker)
                """)
                
                # Create trades table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_id INTEGER,
                        timestamp DATETIME NOT NULL,
                        ticker TEXT NOT NULL,
                        action TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        price REAL NOT NULL,
                        commission REAL DEFAULT 0,
                        trade_type TEXT,
                        notes TEXT,
                        FOREIGN KEY (signal_id) REFERENCES signals (id)
                    )
                """)
                
                # Create performance table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE NOT NULL UNIQUE,
                        signals_sent INTEGER DEFAULT 0,
                        high_confidence_signals INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        losing_trades INTEGER DEFAULT 0,
                        total_pnl REAL DEFAULT 0,
                        best_trade REAL DEFAULT 0,
                        worst_trade REAL DEFAULT 0,
                        daily_summary TEXT
                    )
                """)
                
                # Create patterns table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_name TEXT NOT NULL UNIQUE,
                        occurrences INTEGER DEFAULT 0,
                        successful INTEGER DEFAULT 0,
                        average_return REAL DEFAULT 0,
                        last_seen DATETIME
                    )
                """)
                
                conn.commit()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def save_automation_signal(self, signal_data: Dict) -> int:
        """Save signal specifically for automation pickup - FIXED VERSION"""
        
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Validate required fields
                    if not signal_data.get('ticker'):
                        raise ValueError("Ticker is required")
                    if not signal_data.get('strategy'):
                        raise ValueError("Strategy is required")
                    
                    cursor.execute("""
                        INSERT INTO signals (
                            timestamp, ticker, strategy, confidence, entry_price,
                            signal_data, source, processed_by_automation
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        signal_data.get('timestamp', datetime.now(self.timezone).isoformat()),
                        signal_data['ticker'],
                        signal_data['strategy'],
                        signal_data.get('confidence', 0),
                        signal_data.get('current_price', 0),
                        json.dumps(signal_data, default=str),
                        'automation_ready',
                        0  # Not processed yet
                    ))
                    
                    signal_id = cursor.lastrowid
                    conn.commit()
                    
                    logger.info(f"Automation signal saved: {signal_data['ticker']} (ID: {signal_id})")
                    return signal_id
                    
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Database locked on attempt {attempt + 1}, retrying...")
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    logger.error(f"Error saving automation signal after {attempt + 1} attempts: {e}")
                    return -1
            except Exception as e:
                logger.error(f"Error saving automation signal: {e}")
                return -1
        
        return -1
    
    def get_unprocessed_signals(self, limit: int = 10) -> List[Dict]:
        """Get unprocessed signals for automation bot - FIXED VERSION"""
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, timestamp, ticker, strategy, confidence, entry_price, 
                        signal_data, source
                    FROM signals 
                    WHERE processed_by_automation = 0 
                    AND status = 'active'
                    AND source IN ('signal_generator', 'automated_scan', 'automation_ready')
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                results = cursor.fetchall()
                
                signals = []
                for row in results:
                    try:
                        signal_data = json.loads(row['signal_data']) if row['signal_data'] else {}
                    except (json.JSONDecodeError, TypeError):
                        signal_data = {}
                    
                    signals.append({
                        'id': row['id'],
                        'timestamp': row['timestamp'],
                        'ticker': row['ticker'],
                        'strategy': row['strategy'],
                        'confidence': row['confidence'],
                        'current_price': row['entry_price'] or 0,
                        'signal_data': signal_data,
                        'source': row['source']
                    })
                
                logger.info(f"Retrieved {len(signals)} unprocessed signals")
                return signals
                
        except Exception as e:
            logger.error(f"Error getting unprocessed signals: {e}")
            return []
    
    def mark_signal_processed(self, signal_id: int, notes: str = None) -> bool:
        """Mark signal as processed by automation bot - FIXED VERSION"""
        
        max_retries = 3
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        UPDATE signals 
                        SET processed_by_automation = 1,
                            automation_timestamp = ?
                        WHERE id = ?
                    """, (datetime.now(self.timezone).isoformat(), signal_id))
                    
                    conn.commit()
                    
                    logger.info(f"Signal {signal_id} marked as processed by automation")
                    return True
                    
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Database locked on attempt {attempt + 1}, retrying...")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    logger.error(f"Error marking signal as processed after {attempt + 1} attempts: {e}")
                    return False
            except Exception as e:
                logger.error(f"Error marking signal as processed: {e}")
                return False
        
        return False
    
    def log_signal(self, ticker: str, analysis_result: Dict, source: str = 'signal_generator') -> int:
        """Log a new signal - FIXED VERSION with better connection handling"""
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Extract trade recommendation safely
                trade_rec = analysis_result.get('trade_recommendation', {})
                if not trade_rec:
                    logger.warning(f"No trade recommendation found for {ticker}")
                    return -1
                
                # Extract market data properly
                market_data = analysis_result.get('market_data', {})
                if not market_data:
                    market_data = analysis_result.get('indian_market_data', {})
                    if not market_data:
                        market_data = {'current_price': 0}
                
                # Extract key data with safe defaults
                expected_outcomes = trade_rec.get('expected_outcomes', {})
                options_legs = trade_rec.get('option_legs', [])
                
                signal_data = {
                    'ticker': ticker,
                    'timestamp': datetime.now(self.timezone).isoformat(),
                    'strategy': trade_rec.get('strategy', 'UNKNOWN'),
                    'confidence': trade_rec.get('confidence', 0),
                    'entry_price': market_data.get('current_price', 0),
                    'expected_outcomes': expected_outcomes,
                    'option_legs': options_legs,
                    'risk_analysis': analysis_result.get('risk_analysis', {}),
                    'market_analysis': analysis_result.get('market_analysis_summary', {}),
                    'volatility_analysis': analysis_result.get('volatility_analysis', {}),
                    'source': source
                }
                
                # Get expiry date safely
                expiry_date = None
                if options_legs:
                    expiry_date = options_legs[0].get('expiry')
                
                cursor.execute("""
                    INSERT INTO signals (
                        timestamp, ticker, strategy, confidence, entry_price,
                        target_price, stop_loss, max_profit, max_loss, expiry_date,
                        signal_data, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(self.timezone).isoformat(),
                    ticker,
                    trade_rec.get('strategy', 'UNKNOWN'),
                    trade_rec.get('confidence', 0),
                    signal_data['entry_price'],
                    expected_outcomes.get('target_price'),
                    expected_outcomes.get('stop_loss'),
                    expected_outcomes.get('max_profit'),
                    expected_outcomes.get('max_loss'),
                    expiry_date,
                    json.dumps(signal_data, default=str),
                    source
                ))
                
                signal_id = cursor.lastrowid
                
                # Update daily performance
                self._update_daily_performance(cursor, 'signals_sent')
                if trade_rec.get('confidence', 0) >= 0.8:
                    self._update_daily_performance(cursor, 'high_confidence_signals')
                
                # Log pattern
                market_analysis = analysis_result.get('market_analysis_summary', {})
                pattern = market_analysis.get('primary_pattern')
                if pattern:
                    self._log_pattern(cursor, pattern)
                
                conn.commit()
                
                logger.info(f"Signal logged: {ticker} - {trade_rec.get('strategy', 'UNKNOWN')} (ID: {signal_id})")
                return signal_id
                
        except Exception as e:
            logger.error(f"Error logging signal for {ticker}: {e}", exc_info=True)
            return -1
    
    def log_trade(self, signal_id: int, ticker: str, action: str, 
                  quantity: int, price: float, **kwargs) -> int:
        """Log a trade execution"""
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO trades (
                        signal_id, timestamp, ticker, action, quantity, price,
                        commission, trade_type, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal_id,
                    datetime.now(self.timezone).isoformat(),
                    ticker,
                    action,
                    quantity,
                    price,
                    kwargs.get('commission', 0),
                    kwargs.get('trade_type', 'OPTION'),
                    kwargs.get('notes', '')
                ))
                
                trade_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Trade logged: {ticker} {action} {quantity} @ {price} (ID: {trade_id})")
                return trade_id
                
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
            return -1
    
    def update_signal_status(self, signal_id: int, status: str, pnl: float = None, notes: str = None):
        """Update signal status and P&L with optional notes"""
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # First, update the status
                cursor.execute("""
                    UPDATE signals SET status = ? WHERE id = ?
                """, (status, signal_id))
                
                # If notes are provided, update the signal data to include notes
                if notes:
                    # Fetch current signal data
                    cursor.execute("""
                        SELECT signal_data FROM signals WHERE id = ?
                    """, (signal_id,))
                    
                    result = cursor.fetchone()
                    if result and result['signal_data']:
                        try:
                            signal_data = json.loads(result['signal_data'])
                            # Add or update notes in the signal data
                            if 'notes' not in signal_data:
                                signal_data['notes'] = []
                            if isinstance(signal_data['notes'], list):
                                signal_data['notes'].append({
                                    'timestamp': datetime.now(self.timezone).isoformat(),
                                    'status': status,
                                    'note': notes
                                })
                            else:
                                signal_data['notes'] = [{
                                    'timestamp': datetime.now(self.timezone).isoformat(),
                                    'status': status,
                                    'note': notes
                                }]
                            
                            # Update the signal data
                            cursor.execute("""
                                UPDATE signals SET signal_data = ? WHERE id = ?
                            """, (json.dumps(signal_data, default=str), signal_id))
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse signal data for signal {signal_id}")
                
                # Update performance if closed
                if status == 'closed' and pnl is not None:
                    today = datetime.now(self.timezone).strftime('%Y-%m-%d')
                    
                    if pnl > 0:
                        self._update_daily_performance(cursor, 'winning_trades')
                    else:
                        self._update_daily_performance(cursor, 'losing_trades')
                    
                    # Update total P&L and best/worst trades
                    cursor.execute("""
                        SELECT total_pnl, best_trade, worst_trade 
                        FROM performance WHERE date = ?
                    """, (today,))
                    
                    result = cursor.fetchone()
                    if result:
                        new_total = result['total_pnl'] + pnl
                        new_best = max(result['best_trade'], pnl)
                        new_worst = min(result['worst_trade'], pnl)
                        
                        cursor.execute("""
                            UPDATE performance 
                            SET total_pnl = ?, best_trade = ?, worst_trade = ?
                            WHERE date = ?
                        """, (new_total, new_best, new_worst, today))
                    else:
                        # Insert new performance record
                        cursor.execute("""
                            INSERT INTO performance (date, total_pnl, best_trade, worst_trade)
                            VALUES (?, ?, ?, ?)
                        """, (today, pnl, pnl, pnl))
                
                conn.commit()
                
                logger.info(f"Signal {signal_id} status updated to {status}" + 
                          (f" with note: {notes[:50]}..." if notes else ""))
                
        except Exception as e:
            logger.error(f"Error updating signal status: {e}")
    
    def get_daily_summary(self, date: str = None) -> Dict:
        """Get daily trading summary"""
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                if date is None:
                    date = datetime.now(self.timezone).strftime('%Y-%m-%d')
                
                # Get performance metrics
                cursor.execute("""
                    SELECT signals_sent, high_confidence_signals, winning_trades, 
                           losing_trades, total_pnl, best_trade, worst_trade
                    FROM performance WHERE date = ?
                """, (date,))
                
                perf_data = cursor.fetchone()
                
                # Get signals for the day
                cursor.execute("""
                    SELECT ticker, strategy, confidence, entry_price, status
                    FROM signals 
                    WHERE DATE(timestamp) = ?
                    ORDER BY confidence DESC
                    LIMIT 10
                """, (date,))
                
                signals = cursor.fetchall()
                
                # Get top patterns
                cursor.execute("""
                    SELECT pattern_name, occurrences, successful, average_return
                    FROM patterns
                    WHERE DATE(last_seen) = ?
                    ORDER BY successful DESC
                    LIMIT 5
                """, (date,))
                
                patterns = cursor.fetchall()
                
                # Build summary
                summary = {
                    'date': date,
                    'signals_sent': perf_data['signals_sent'] if perf_data else 0,
                    'high_confidence_signals': perf_data['high_confidence_signals'] if perf_data else 0,
                    'winning_trades': perf_data['winning_trades'] if perf_data else 0,
                    'losing_trades': perf_data['losing_trades'] if perf_data else 0,
                    'total_pnl': perf_data['total_pnl'] if perf_data else 0,
                    'best_trade': perf_data['best_trade'] if perf_data else 0,
                    'worst_trade': perf_data['worst_trade'] if perf_data else 0,
                    'top_signals': [
                        {
                            'ticker': s['ticker'],
                            'strategy': s['strategy'],
                            'confidence': s['confidence'],
                            'entry_price': s['entry_price'],
                            'status': s['status']
                        } for s in signals
                    ],
                    'top_patterns': [
                        {
                            'pattern': p['pattern_name'],
                            'occurrences': p['occurrences'],
                            'successful': p['successful'],
                            'avg_return': p['average_return']
                        } for p in patterns
                    ],
                    'learning_insights': self._generate_insights(signals, patterns)
                }
                
                return summary
                
        except Exception as e:
            logger.error(f"Error getting daily summary: {e}")
            return {
                'date': date or datetime.now(self.timezone).strftime('%Y-%m-%d'),
                'signals_sent': 0,
                'high_confidence_signals': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'top_signals': [],
                'top_patterns': [],
                'learning_insights': {}
            }
    
    def get_performance_history(self, days: int = 30) -> pd.DataFrame:
        """Get performance history"""
        
        try:
            with self._get_db_connection() as conn:
                # Calculate start date
                start_date = (datetime.now(self.timezone) - timedelta(days=days)).strftime('%Y-%m-%d')
                
                query = """
                    SELECT date, signals_sent, high_confidence_signals,
                           winning_trades, losing_trades, total_pnl
                    FROM performance
                    WHERE date >= ?
                    ORDER BY date
                """
                
                df = pd.read_sql_query(query, conn, params=(start_date,))
                
                # Calculate additional metrics
                if not df.empty:
                    df['total_trades'] = df['winning_trades'] + df['losing_trades']
                    df['win_rate'] = df.apply(
                        lambda row: row['winning_trades'] / row['total_trades'] if row['total_trades'] > 0 else 0, 
                        axis=1
                    )
                    df['cumulative_pnl'] = df['total_pnl'].cumsum()
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            return pd.DataFrame()
    
    def get_strategy_performance(self) -> Dict:
        """Get performance by strategy"""
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        strategy,
                        COUNT(*) as total_signals,
                        AVG(confidence) as avg_confidence,
                        SUM(CASE WHEN status = 'closed' THEN 1 ELSE 0 END) as closed_trades,
                        SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active_trades
                    FROM signals
                    WHERE strategy != 'UNKNOWN'
                    GROUP BY strategy
                    ORDER BY total_signals DESC
                """)
                
                results = cursor.fetchall()
                
                performance = {}
                for row in results:
                    performance[row['strategy']] = {
                        'total_signals': row['total_signals'],
                        'avg_confidence': round(row['avg_confidence'], 3) if row['avg_confidence'] else 0,
                        'closed_trades': row['closed_trades'],
                        'active_trades': row['active_trades']
                    }
                
                return performance
                
        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            return {}
    
    def get_automation_stats(self) -> Dict:
        """Get automation processing statistics - FIXED VERSION"""
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get processing stats
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_signals,
                        SUM(CASE WHEN processed_by_automation = 1 THEN 1 ELSE 0 END) as processed,
                        SUM(CASE WHEN processed_by_automation = 0 THEN 1 ELSE 0 END) as pending
                    FROM signals 
                    WHERE source IN ('signal_generator', 'automated_scan', 'automation_ready')
                """)
                
                result = cursor.fetchone()
                
                # Get processing rate by day
                cursor.execute("""
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as signals_generated,
                        SUM(CASE WHEN processed_by_automation = 1 THEN 1 ELSE 0 END) as processed
                    FROM signals 
                    WHERE source IN ('signal_generator', 'automated_scan', 'automation_ready')
                    AND DATE(timestamp) >= DATE('now', '-7 days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                """)
                
                daily_stats = cursor.fetchall()
                
                return {
                    'total_signals': result['total_signals'] if result else 0,
                    'processed_signals': result['processed'] if result else 0,
                    'pending_signals': result['pending'] if result else 0,
                    'processing_rate': (result['processed'] / result['total_signals'] * 100) 
                                    if result and result['total_signals'] > 0 else 0,
                    'daily_processing': [
                        {
                            'date': row['date'],
                            'generated': row['signals_generated'],
                            'processed': row['processed'],
                            'rate': (row['processed'] / row['signals_generated'] * 100) 
                                if row['signals_generated'] > 0 else 0
                        }
                        for row in daily_stats
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error getting automation stats: {e}")
            return {
                'total_signals': 0,
                'processed_signals': 0,
                'pending_signals': 0,
                'processing_rate': 0,
                'daily_processing': []
            }
    
    def export_to_excel(self, filename: str = None) -> str:
        """Export all data to Excel"""
        
        try:
            if filename is None:
                filename = f"trading_report_{datetime.now(self.timezone).strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            with self._get_db_connection() as conn:
                # Create Excel writer
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    # Export signals
                    try:
                        signals_df = pd.read_sql_query("SELECT * FROM signals", conn)
                        signals_df.to_excel(writer, sheet_name='Signals', index=False)
                    except Exception as e:
                        logger.warning(f"Could not export signals: {e}")
                    
                    # Export trades
                    try:
                        trades_df = pd.read_sql_query("SELECT * FROM trades", conn)
                        trades_df.to_excel(writer, sheet_name='Trades', index=False)
                    except Exception as e:
                        logger.warning(f"Could not export trades: {e}")
                    
                    # Export performance
                    try:
                        perf_df = pd.read_sql_query("SELECT * FROM performance", conn)
                        perf_df.to_excel(writer, sheet_name='Performance', index=False)
                    except Exception as e:
                        logger.warning(f"Could not export performance: {e}")
                    
                    # Export patterns
                    try:
                        patterns_df = pd.read_sql_query("SELECT * FROM patterns", conn)
                        patterns_df.to_excel(writer, sheet_name='Patterns', index=False)
                    except Exception as e:
                        logger.warning(f"Could not export patterns: {e}")
                    
                    # Export automation stats
                    try:
                        automation_df = pd.read_sql_query("""
                            SELECT ticker, strategy, confidence, 
                                   processed_by_automation, automation_timestamp, source
                            FROM signals 
                            WHERE source IN ('signal_generator', 'automated_scan', 'automation_ready')
                            ORDER BY timestamp DESC
                        """, conn)
                        automation_df.to_excel(writer, sheet_name='Automation', index=False)
                    except Exception as e:
                        logger.warning(f"Could not export automation data: {e}")
                
                logger.info(f"Data exported to {filename}")
                return filename
                
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            return ""
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to keep database size manageable"""
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cutoff_date = (datetime.now(self.timezone) - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
                
                # Delete old closed signals
                cursor.execute("""
                    DELETE FROM signals 
                    WHERE DATE(timestamp) < ? AND status = 'closed'
                """, (cutoff_date,))
                
                deleted_signals = cursor.rowcount
                
                # Delete old performance records
                cursor.execute("""
                    DELETE FROM performance 
                    WHERE date < ?
                """, (cutoff_date,))
                
                deleted_performance = cursor.rowcount
                
                conn.commit()
                
                logger.info(f"Cleanup completed: {deleted_signals} signals, {deleted_performance} performance records deleted")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_database_stats(self) -> Dict:
        """Get database statistics - FIXED VERSION"""
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count records in each table
                for table in ['signals', 'trades', 'performance', 'patterns']:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                    result = cursor.fetchone()
                    stats[f'{table}_count'] = result['count'] if result else 0
                
                # Database file size
                if os.path.exists(self.db_path):
                    stats['db_size_mb'] = round(os.path.getsize(self.db_path) / (1024 * 1024), 2)
                else:
                    stats['db_size_mb'] = 0
                
                # Date range
                cursor.execute("SELECT MIN(DATE(timestamp)) as min_date, MAX(DATE(timestamp)) as max_date FROM signals")
                date_range = cursor.fetchone()
                if date_range and date_range['min_date']:
                    stats['date_range'] = f"{date_range['min_date']} to {date_range['max_date']}"
                else:
                    stats['date_range'] = "No data"
                
                # Automation processing stats
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN processed_by_automation = 1 THEN 1 ELSE 0 END) as processed,
                        SUM(CASE WHEN source = 'automation_ready' THEN 1 ELSE 0 END) as automation_ready
                    FROM signals
                """)
                
                automation_stats = cursor.fetchone()
                if automation_stats:
                    stats['automation_total'] = automation_stats['total']
                    stats['automation_processed'] = automation_stats['processed']
                    stats['automation_ready'] = automation_stats['automation_ready']
                else:
                    stats['automation_total'] = 0
                    stats['automation_processed'] = 0
                    stats['automation_ready'] = 0
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def test_automation_integration(self) -> Dict:
        """Test automation integration - FIXED VERSION"""
        
        try:
            # Test saving a signal
            test_signal_data = {
                'ticker': 'TEST',
                'strategy': 'TEST_STRATEGY',
                'confidence': 0.85,
                'current_price': 1000,
                'timestamp': datetime.now(self.timezone).isoformat()
            }
            
            signal_id = self.save_automation_signal(test_signal_data)
            
            if signal_id > 0:
                # Small delay to ensure write completes
                time.sleep(0.1)
                
                # Test retrieving unprocessed signals
                unprocessed = self.get_unprocessed_signals(limit=1)
                
                if unprocessed:
                    # Test marking as processed
                    success = self.mark_signal_processed(signal_id, "Test processing")
                    
                    return {
                        'status': 'success',
                        'signal_saved': True,
                        'signal_id': signal_id,
                        'signal_retrieved': True,
                        'marked_processed': success,
                        'message': 'Automation integration working correctly'
                    }
            
            return {
                'status': 'error',
                'message': 'Failed to save test signal'
            }
            
        except Exception as e:
            logger.error(f"Automation integration test failed: {e}")
            return {
                'status': 'error',
                'message': f'Test failed: {str(e)}'
            }
    
    def _update_daily_performance(self, cursor, field: str, value: float = 1):
        """Update daily performance metrics"""
        
        today = datetime.now(self.timezone).strftime('%Y-%m-%d')
        
        try:
            # First, try to get existing record
            cursor.execute("SELECT * FROM performance WHERE date = ?", (today,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record
                field_indices = {
                    'signals_sent': 'signals_sent',
                    'high_confidence_signals': 'high_confidence_signals',
                    'winning_trades': 'winning_trades',
                    'losing_trades': 'losing_trades',
                    'total_pnl': 'total_pnl',
                    'best_trade': 'best_trade',
                    'worst_trade': 'worst_trade'
                }
                
                if field in field_indices:
                    current_value = existing[field_indices[field]] or 0
                    new_value = current_value + value
                    
                    cursor.execute(f"""
                        UPDATE performance SET {field} = ? WHERE date = ?
                    """, (new_value, today))
            else:
                # Insert new record
                cursor.execute(f"""
                    INSERT INTO performance (date, {field}) VALUES (?, ?)
                """, (today, value))
                
        except Exception as e:
            logger.error(f"Error updating daily performance: {e}")
    
    def _log_pattern(self, cursor, pattern_name: str):
        """Log pattern occurrence"""
        
        try:
            # Check if pattern exists
            cursor.execute("SELECT occurrences FROM patterns WHERE pattern_name = ?", (pattern_name,))
            existing = cursor.fetchone()
            
            current_time = datetime.now(self.timezone).isoformat()
            
            if existing:
                # Update existing pattern
                new_count = existing['occurrences'] + 1
                cursor.execute("""
                    UPDATE patterns 
                    SET occurrences = ?, last_seen = ?
                    WHERE pattern_name = ?
                """, (new_count, current_time, pattern_name))
            else:
                # Insert new pattern
                cursor.execute("""
                    INSERT INTO patterns (pattern_name, occurrences, successful, average_return, last_seen)
                    VALUES (?, 1, 0, 0, ?)
                """, (pattern_name, current_time))
                
        except Exception as e:
            logger.error(f"Error logging pattern: {e}")
    
    def _generate_insights(self, signals: List, patterns: List) -> Dict:
        """Generate trading insights from data"""
        
        insights = {}
        
        try:
            if signals:
                # Most common strategy
                strategies = [s['strategy'] for s in signals if s['strategy'] != 'UNKNOWN']
                if strategies:
                    insights['most_used_strategy'] = max(set(strategies), key=strategies.count)
                
                # Average confidence
                confidences = [s['confidence'] for s in signals if s['confidence'] > 0]
                if confidences:
                    insights['avg_confidence'] = round(sum(confidences) / len(confidences), 3)
            
            if patterns:
                # Best performing pattern
                valid_patterns = [p for p in patterns if p['occurrences'] > 0]
                if valid_patterns:
                    best_pattern = max(valid_patterns, 
                                     key=lambda x: x['successful'] / x['occurrences'] if x['occurrences'] > 0 else 0)
                    insights['best_pattern'] = best_pattern['pattern_name']
                    insights['pattern_success_rate'] = round(
                        best_pattern['successful'] / best_pattern['occurrences'] 
                        if best_pattern['occurrences'] > 0 else 0, 3
                    )
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
        
        return insights
    
    # Additional utility methods
    
    def get_active_signals(self, limit: int = 50) -> List[Dict]:
        """Get all active signals"""
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, timestamp, ticker, strategy, confidence, 
                           entry_price, status, source
                    FROM signals 
                    WHERE status = 'active'
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
                
                results = cursor.fetchall()
                
                signals = []
                for row in results:
                    signals.append({
                        'id': row['id'],
                        'timestamp': row['timestamp'],
                        'ticker': row['ticker'],
                        'strategy': row['strategy'],
                        'confidence': row['confidence'],
                        'entry_price': row['entry_price'],
                        'status': row['status'],
                        'source': row['source']
                    })
                
                return signals
                
        except Exception as e:
            logger.error(f"Error getting active signals: {e}")
            return []
    
    def get_signal_by_id(self, signal_id: int) -> Dict:
        """Get detailed signal information by ID"""
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM signals WHERE id = ?
                """, (signal_id,))
                
                result = cursor.fetchone()
                
                if result:
                    signal_data = {}
                    try:
                        signal_data = json.loads(result['signal_data']) if result['signal_data'] else {}
                    except (json.JSONDecodeError, TypeError):
                        pass
                    
                    return {
                        'id': result['id'],
                        'timestamp': result['timestamp'],
                        'ticker': result['ticker'],
                        'strategy': result['strategy'],
                        'confidence': result['confidence'],
                        'entry_price': result['entry_price'],
                        'target_price': result['target_price'],
                        'stop_loss': result['stop_loss'],
                        'max_profit': result['max_profit'],
                        'max_loss': result['max_loss'],
                        'expiry_date': result['expiry_date'],
                        'status': result['status'],
                        'processed_by_automation': result['processed_by_automation'],
                        'automation_timestamp': result['automation_timestamp'],
                        'source': result['source'],
                        'signal_data': signal_data
                    }
                
                return {}
                
        except Exception as e:
            logger.error(f"Error getting signal by ID: {e}")
            return {}
    
    def update_pattern_performance(self, pattern_name: str, success: bool, return_pct: float):
        """Update pattern performance metrics"""
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get current pattern stats
                cursor.execute("""
                    SELECT successful, average_return, occurrences 
                    FROM patterns WHERE pattern_name = ?
                """, (pattern_name,))
                
                result = cursor.fetchone()
                
                if result:
                    successful = result['successful'] + (1 if success else 0)
                    occurrences = result['occurrences']
                    
                    # Calculate new average return
                    current_avg = result['average_return']
                    new_avg = ((current_avg * occurrences) + return_pct) / (occurrences + 1)
                    
                    cursor.execute("""
                        UPDATE patterns 
                        SET successful = ?, average_return = ?
                        WHERE pattern_name = ?
                    """, (successful, new_avg, pattern_name))
                    
                    conn.commit()
                    logger.info(f"Updated pattern performance for {pattern_name}")
                
        except Exception as e:
            logger.error(f"Error updating pattern performance: {e}")
    
    def close_connection(self):
        """Close any remaining connections (cleanup method)"""
        try:
            if self._connection:
                self._connection.close()
                self._connection = None
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")