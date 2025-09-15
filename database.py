import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging
import threading
import time

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: str = "data/crypto_bot.db"):
        self.db_path = db_path
        self.log_cache = []
        self.cache_lock = threading.Lock()
        self.last_flush = time.time()
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute('PRAGMA journal_mode=WAL')  # Better concurrency
        conn.execute('PRAGMA synchronous=NORMAL')  # Faster writes
        cursor = conn.cursor()
        
        # Signals table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id TEXT UNIQUE NOT NULL,
            timestamp DATETIME NOT NULL,
            coin TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            current_price REAL,
            take_profit REAL NOT NULL,
            stop_loss REAL NOT NULL,
            confidence INTEGER NOT NULL,
            analysis_data TEXT NOT NULL,
            indicators TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Trade results table for P&L tracking
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trade_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            pnl_usd REAL DEFAULT 0,
            pnl_percentage REAL DEFAULT 0,
            max_profit REAL DEFAULT 0,
            max_loss REAL DEFAULT 0,
            duration_minutes INTEGER DEFAULT 0,
            exit_reason TEXT,
            leverage INTEGER DEFAULT 10,
            position_size_usd REAL DEFAULT 1000,
            status TEXT DEFAULT 'open',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (signal_id) REFERENCES signals (signal_id)
        )
        ''')
        
        # Portfolio tracking
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            balance_usd REAL DEFAULT 1000,
            total_trades INTEGER DEFAULT 0,
            winning_trades INTEGER DEFAULT 0,
            losing_trades INTEGER DEFAULT 0,
            total_pnl REAL DEFAULT 0,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Bot activity logs table (simplified)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS bot_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            level TEXT NOT NULL,
            component TEXT NOT NULL,
            message TEXT NOT NULL,
            details TEXT,
            coin TEXT,
            data TEXT
        )
        ''')
        
        # Market analysis logs (simplified)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            coin TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            analysis_type TEXT NOT NULL,
            result TEXT NOT NULL,
            confidence REAL,
            indicators TEXT,
            patterns TEXT,
            support_resistance TEXT
        )
        ''')
        
        # Initialize portfolio if empty
        cursor.execute('SELECT COUNT(*) FROM portfolio')
        if cursor.fetchone()[0] == 0:
            cursor.execute('''
            INSERT INTO portfolio (balance_usd, total_trades, winning_trades, losing_trades, total_pnl)
            VALUES (1000, 0, 0, 0, 0)
            ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def log_bot_activity(self, level: str, component: str, message: str, 
                        details: str = None, coin: str = None, data: Dict = None):
        """Log bot activity for monitoring - REDUCED FREQUENCY"""
        # Only log important events to reduce database locks
        important_events = [
            'SIGNAL_GENERATOR', 'SIGNAL_MANAGER', 'SYSTEM', 'MARKET_SCANNER'
        ]
        
        # Skip DEBUG logs and less important components
        if level == 'DEBUG' or component not in important_events:
            # Just log to console for monitoring
            if level == 'ERROR':
                logger.error(f"[{component}] {message} - {details}")
            elif level == 'WARNING':
                logger.warning(f"[{component}] {message} - {details}")
            else:
                logger.info(f"[{component}] {message}")
            return
        
        # Cache logs and flush periodically to reduce database writes
        with self.cache_lock:
            self.log_cache.append({
                'level': level,
                'component': component,
                'message': message,
                'details': details,
                'coin': coin,
                'data': json.dumps(data) if data else None,
                'timestamp': datetime.now().isoformat()
            })
            
            # Flush cache every 60 seconds or when it gets too big
            current_time = time.time()
            if (current_time - self.last_flush > 60) or len(self.log_cache) > 10:
                self._flush_log_cache()
                self.last_flush = current_time
        
        # Always log to console immediately
        if level == 'ERROR':
            logger.error(f"[{component}] {message} - {details}")
        elif level == 'WARNING':
            logger.warning(f"[{component}] {message} - {details}")
        else:
            logger.info(f"[{component}] {message}")
    
    def _flush_log_cache(self):
        """Flush cached logs to database in batch"""
        if not self.log_cache:
            return
        
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            # Batch insert all cached logs
            for log_entry in self.log_cache:
                cursor.execute('''
                INSERT INTO bot_logs (level, component, message, details, coin, data, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    log_entry['level'],
                    log_entry['component'],
                    log_entry['message'],
                    log_entry['details'],
                    log_entry['coin'],
                    log_entry['data'],
                    log_entry['timestamp']
                ))
            
            conn.commit()
            conn.close()
            self.log_cache.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush log cache: {e}")
            # Clear cache anyway to prevent memory buildup
            self.log_cache.clear()
    
    def log_analysis(self, coin: str, timeframe: str, analysis_type: str, result: str,
                    confidence: float = None, indicators: Dict = None, 
                    patterns: Dict = None, support_resistance: Dict = None):
        """Log market analysis results - REDUCED FREQUENCY"""
        # Only log signal generation analysis, skip intermediate steps
        if analysis_type != 'SIGNAL_GENERATION':
            return
        
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO analysis_logs (coin, timeframe, analysis_type, result, confidence, 
                                     indicators, patterns, support_resistance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                coin,
                timeframe,
                analysis_type,
                result,
                confidence,
                json.dumps(indicators) if indicators else None,
                json.dumps(patterns) if patterns else None,
                json.dumps(support_resistance) if support_resistance else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log analysis: {e}")
    
    def get_recent_logs(self, limit: int = 50) -> List[Dict]:
        """Get recent bot activity logs"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM bot_logs 
            ORDER BY timestamp DESC 
            LIMIT ?
            ''', (limit,))
            
            logs = []
            for row in cursor.fetchall():
                log = {
                    'id': row[0],
                    'timestamp': row[1],
                    'level': row[2],
                    'component': row[3],
                    'message': row[4],
                    'details': row[5],
                    'coin': row[6],
                    'data': json.loads(row[7]) if row[7] else None
                }
                logs.append(log)
            
            conn.close()
            return logs
        except Exception as e:
            logger.error(f"Failed to get recent logs: {e}")
            return []
    
    def get_analysis_summary(self, hours: int = 24) -> Dict:
        """Get analysis summary for the last N hours"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT 
                coin,
                COUNT(*) as analysis_count,
                AVG(confidence) as avg_confidence,
                COUNT(CASE WHEN result LIKE '%bullish%' THEN 1 END) as bullish_signals,
                COUNT(CASE WHEN result LIKE '%bearish%' THEN 1 END) as bearish_signals
            FROM analysis_logs 
            WHERE timestamp >= datetime('now', '-{} hours')
            GROUP BY coin
            ORDER BY analysis_count DESC
            '''.format(hours))
            
            summary = {}
            for row in cursor.fetchall():
                summary[row[0]] = {
                    'analysis_count': row[1],
                    'avg_confidence': row[2] or 0,
                    'bullish_signals': row[3],
                    'bearish_signals': row[4]
                }
            
            conn.close()
            return summary
        except Exception as e:
            logger.error(f"Failed to get analysis summary: {e}")
            return {}

    def save_signal(self, signal_data: Dict) -> bool:
        """Save a new trading signal"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO signals (
                signal_id, timestamp, coin, direction, entry_price, current_price,
                take_profit, stop_loss, confidence, analysis_data, indicators, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_data['signal_id'],
                signal_data['timestamp'],
                signal_data['coin'],
                signal_data['direction'],
                signal_data['entry_price'],
                signal_data['current_price'],
                signal_data['take_profit'],
                signal_data['stop_loss'],
                signal_data['confidence'],
                json.dumps(signal_data['analysis_data']),
                json.dumps(signal_data['indicators']),
                'active'
            ))
            
            # Create corresponding trade result entry
            cursor.execute('''
            INSERT INTO trade_results (
                signal_id, entry_price, position_size_usd, leverage, status
            ) VALUES (?, ?, ?, ?, ?)
            ''', (
                signal_data['signal_id'],
                signal_data['entry_price'],
                1000,  # $1000 position size
                10,    # 10x leverage
                'open'
            ))
            
            conn.commit()
            conn.close()
            
            # Log the signal creation (will be cached)
            self.log_bot_activity(
                'INFO', 
                'SIGNAL_GENERATOR', 
                f"New {signal_data['direction']} signal created",
                f"Entry: ${signal_data['entry_price']}, Confidence: {signal_data['confidence']}%",
                signal_data['coin'],
                {'signal_id': signal_data['signal_id']}
            )
            
            logger.info(f"Signal saved: {signal_data['signal_id']}")
            return True
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
            return False
    
    def get_active_signals(self) -> List[Dict]:
        """Get all active signals"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT s.*, tr.pnl_usd, tr.pnl_percentage, tr.max_profit, tr.max_loss, tr.status as trade_status
            FROM signals s
            LEFT JOIN trade_results tr ON s.signal_id = tr.signal_id
            WHERE s.status = 'active'
            ORDER BY s.created_at DESC
            ''')
            
            signals = []
            for row in cursor.fetchall():
                signal = {
                    'id': row[0],
                    'signal_id': row[1],
                    'timestamp': row[2],
                    'coin': row[3],
                    'direction': row[4],
                    'entry_price': row[5],
                    'current_price': row[6],
                    'take_profit': row[7],
                    'stop_loss': row[8],
                    'confidence': row[9],
                    'analysis_data': json.loads(row[10]),
                    'indicators': json.loads(row[11]),
                    'status': row[12],
                    'created_at': row[13],
                    'updated_at': row[14],
                    'pnl_usd': row[15] or 0,
                    'pnl_percentage': row[16] or 0,
                    'max_profit': row[17] or 0,
                    'max_loss': row[18] or 0,
                    'trade_status': row[19] or 'open'
                }
                signals.append(signal)
            
            conn.close()
            return signals
        except Exception as e:
            logger.error(f"Error getting active signals: {e}")
            return []
    
    def update_signal_price(self, signal_id: str, current_price: float) -> bool:
        """Update current price for a signal and calculate P&L"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            # Update signal current price
            cursor.execute('''
            UPDATE signals SET current_price = ?, updated_at = CURRENT_TIMESTAMP
            WHERE signal_id = ?
            ''', (current_price, signal_id))
            
            # Get signal details for P&L calculation
            cursor.execute('''
            SELECT direction, entry_price, take_profit, stop_loss, coin
            FROM signals WHERE signal_id = ?
            ''', (signal_id,))
            
            signal = cursor.fetchone()
            if signal:
                direction, entry_price, take_profit, stop_loss, coin = signal
                
                # Calculate P&L
                if direction == 'LONG':
                    pnl_percentage = ((current_price - entry_price) / entry_price) * 100 * 10  # 10x leverage
                else:  # SHORT
                    pnl_percentage = ((entry_price - current_price) / entry_price) * 100 * 10  # 10x leverage
                
                pnl_usd = (pnl_percentage / 100) * 1000  # $1000 position size
                
                # Update trade result
                cursor.execute('''
                UPDATE trade_results 
                SET pnl_usd = ?, pnl_percentage = ?, updated_at = CURRENT_TIMESTAMP
                WHERE signal_id = ?
                ''', (pnl_usd, pnl_percentage, signal_id))
                
                # Update max profit/loss
                cursor.execute('''
                UPDATE trade_results 
                SET max_profit = CASE WHEN ? > max_profit THEN ? ELSE max_profit END,
                    max_loss = CASE WHEN ? < max_loss THEN ? ELSE max_loss END
                WHERE signal_id = ?
                ''', (pnl_usd, pnl_usd, pnl_usd, pnl_usd, signal_id))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error updating signal price: {e}")
            return False
    
    def close_signal(self, signal_id: str, exit_price: float, exit_reason: str) -> bool:
        """Close a signal and finalize P&L"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            # Get signal info before closing
            cursor.execute('SELECT coin, direction FROM signals WHERE signal_id = ?', (signal_id,))
            signal_info = cursor.fetchone()
            
            # Update signal status
            cursor.execute('''
            UPDATE signals SET status = 'closed', updated_at = CURRENT_TIMESTAMP
            WHERE signal_id = ?
            ''', (signal_id,))
            
            # Update trade result
            cursor.execute('''
            UPDATE trade_results 
            SET exit_price = ?, exit_reason = ?, status = 'closed', updated_at = CURRENT_TIMESTAMP
            WHERE signal_id = ?
            ''', (exit_price, exit_reason, signal_id))
            
            conn.commit()
            conn.close()
            
            # Log signal closure
            if signal_info:
                self.log_bot_activity(
                    'INFO', 
                    'SIGNAL_MANAGER', 
                    f"Signal closed: {signal_info[0]} {signal_info[1]}",
                    f"Exit: ${exit_price}, Reason: {exit_reason}",
                    signal_info[0],
                    {'signal_id': signal_id, 'exit_reason': exit_reason}
                )
            
            logger.info(f"Signal closed: {signal_id} at {exit_price}")
            return True
        except Exception as e:
            logger.error(f"Error closing signal: {e}")
            return False
    
    def get_portfolio_stats(self) -> Dict:
        """Get overall portfolio statistics"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            # Get total P&L from closed trades
            cursor.execute('''
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl_usd < 0 THEN 1 ELSE 0 END) as losing_trades,
                SUM(pnl_usd) as total_pnl,
                AVG(pnl_usd) as avg_pnl,
                MAX(pnl_usd) as best_trade,
                MIN(pnl_usd) as worst_trade
            FROM trade_results WHERE status = 'closed'
            ''')
            
            stats = cursor.fetchone()
            
            # Get current open trades P&L
            cursor.execute('''
            SELECT SUM(pnl_usd) FROM trade_results WHERE status = 'open'
            ''')
            open_pnl = cursor.fetchone()[0] or 0
            
            conn.close()
            
            total_trades, winning_trades, losing_trades, total_pnl, avg_pnl, best_trade, worst_trade = stats
            
            return {
                'total_balance': 1000 + (total_pnl or 0) + open_pnl,
                'total_trades': total_trades or 0,
                'winning_trades': winning_trades or 0,
                'losing_trades': losing_trades or 0,
                'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                'total_pnl': (total_pnl or 0) + open_pnl,
                'avg_pnl': avg_pnl or 0,
                'best_trade': best_trade or 0,
                'worst_trade': worst_trade or 0,
                'open_pnl': open_pnl
            }
        except Exception as e:
            logger.error(f"Error getting portfolio stats: {e}")
            return {
                'total_balance': 1000, 'total_trades': 0, 'winning_trades': 0,
                'losing_trades': 0, 'win_rate': 0, 'total_pnl': 0,
                'avg_pnl': 0, 'best_trade': 0, 'worst_trade': 0, 'open_pnl': 0
            }
    
    def get_signal_history(self, limit: int = 50) -> List[Dict]:
        """Get signal history with P&L data"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT s.*, tr.pnl_usd, tr.pnl_percentage, tr.exit_price, tr.exit_reason, tr.duration_minutes
            FROM signals s
            LEFT JOIN trade_results tr ON s.signal_id = tr.signal_id
            ORDER BY s.created_at DESC
            LIMIT ?
            ''', (limit,))
            
            signals = []
            for row in cursor.fetchall():
                signal = {
                    'signal_id': row[1],
                    'timestamp': row[2],
                    'coin': row[3],
                    'direction': row[4],
                    'entry_price': row[5],
                    'current_price': row[6],
                    'take_profit': row[7],
                    'stop_loss': row[8],
                    'confidence': row[9],
                    'analysis_data': json.loads(row[10]),
                    'status': row[12],
                    'created_at': row[13],
                    'pnl_usd': row[15] or 0,
                    'pnl_percentage': row[16] or 0,
                    'exit_price': row[17],
                    'exit_reason': row[18],
                    'duration_minutes': row[19] or 0
                }
                signals.append(signal)
            
            conn.close()
            return signals
        except Exception as e:
            logger.error(f"Error getting signal history: {e}")
            return []