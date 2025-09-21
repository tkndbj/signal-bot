import sqlite3
import json
import pickle
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
import threading
import time
from contextlib import contextmanager
import os

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: str = "data/crypto_bot.db"):
        self.db_path = db_path
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Thread-safe logging cache
        self.log_cache = []
        self.cache_lock = threading.Lock()
        self.last_flush = time.time()
        
        # Initialize database
        self.init_database()
        
        # Clean up on startup
        self.cleanup_on_startup()
    
    @contextmanager
    def get_db_connection(self):
        """Thread-safe database connection with consistent retry logic"""
        conn = None
        max_retries = 5
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(self.db_path, timeout=30)
                conn.execute('PRAGMA journal_mode=WAL')
                conn.execute('PRAGMA synchronous=NORMAL')
                conn.execute('PRAGMA cache_size=10000')
                conn.execute('PRAGMA busy_timeout=30000')
                conn.row_factory = sqlite3.Row
                yield conn
                conn.commit()
                return
                
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"Database locked, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    raise e
                    
            except Exception as e:
                if conn:
                    conn.rollback()
                logger.error(f"Database error: {e}")
                raise e
                
            finally:
                if conn:
                    conn.close()
    
    def init_database(self):
        """Initialize simplified database schema"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Main signals table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT UNIQUE NOT NULL,
                timestamp DATETIME NOT NULL,
                coin TEXT NOT NULL,
                direction TEXT NOT NULL CHECK (direction IN ('LONG', 'SHORT')),
                entry_price REAL NOT NULL CHECK (entry_price > 0),
                current_price REAL CHECK (current_price > 0),
                take_profit REAL NOT NULL CHECK (take_profit > 0),
                stop_loss REAL NOT NULL CHECK (stop_loss > 0),
                confidence INTEGER NOT NULL CHECK (confidence >= 0 AND confidence <= 100),
                analysis_data TEXT NOT NULL,
                indicators TEXT NOT NULL,
                
                -- ML fields (simplified)
                ml_prediction REAL,
                model_confidence REAL CHECK (model_confidence >= 0 AND model_confidence <= 1),
                feature_importance TEXT,
                
                -- Position tracking
                position_value REAL DEFAULT NULL,
                leverage INTEGER DEFAULT 15,
                
                -- Status tracking
                status TEXT DEFAULT 'active' CHECK (status IN ('active', 'closed', 'cancelled')),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                closed_at DATETIME,
                exit_price REAL CHECK (exit_price > 0 OR exit_price IS NULL),
                exit_reason TEXT
            )
            ''')
            
            # Trade results table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT NOT NULL,
                entry_price REAL NOT NULL CHECK (entry_price > 0),
                exit_price REAL CHECK (exit_price > 0 OR exit_price IS NULL),
                pnl_usd REAL DEFAULT 0,
                pnl_percentage REAL DEFAULT 0,
                duration_minutes INTEGER DEFAULT 0,
                exit_reason TEXT,
                leverage INTEGER DEFAULT 15,
                position_value REAL DEFAULT NULL,
                status TEXT DEFAULT 'open' CHECK (status IN ('open', 'closed')),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (signal_id) REFERENCES signals (signal_id)
            )
            ''')
            
            # ML Models table (simplified)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                model_type TEXT NOT NULL,
                model_data BLOB,
                scaler_data BLOB,
                feature_names TEXT,
                feature_importance TEXT,
                cv_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(coin, model_type)
            )
            ''')
            
            # Portfolio tracking
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                balance_usd REAL DEFAULT 1000 CHECK (balance_usd >= 0),
                total_trades INTEGER DEFAULT 0 CHECK (total_trades >= 0),
                winning_trades INTEGER DEFAULT 0 CHECK (winning_trades >= 0),
                losing_trades INTEGER DEFAULT 0 CHECK (losing_trades >= 0),
                total_pnl REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0 CHECK (max_drawdown >= 0),
                peak_balance REAL DEFAULT 1000 CHECK (peak_balance > 0),
                win_rate REAL DEFAULT 0,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Bot logs
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS bot_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                level TEXT NOT NULL CHECK (level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR')),
                component TEXT NOT NULL,
                message TEXT NOT NULL,
                details TEXT
            )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_coin ON signals(coin)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_results_signal ON trade_results(signal_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bot_logs_timestamp ON bot_logs(timestamp)')
            
            # Initialize portfolio if empty
            cursor.execute('SELECT COUNT(*) FROM portfolio')
            if cursor.fetchone()[0] == 0:
                cursor.execute('''
                INSERT INTO portfolio (balance_usd, total_trades, winning_trades, 
                                      losing_trades, total_pnl, peak_balance)
                VALUES (1000, 0, 0, 0, 0, 1000)
                ''')
            
            logger.info("Database initialized successfully")
    
    def cleanup_on_startup(self):
        """Clean up stale data on startup"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Remove orphaned trade results
                cursor.execute('''
                DELETE FROM trade_results 
                WHERE signal_id NOT IN (SELECT signal_id FROM signals)
                ''')
                
                # Close signals older than 24 hours that are still active
                cursor.execute('''
                UPDATE signals 
                SET status = 'cancelled', 
                    exit_reason = 'Stale signal', 
                    closed_at = CURRENT_TIMESTAMP
                WHERE status = 'active' 
                AND datetime(created_at) < datetime('now', '-24 hours')
                ''')
                
                stale_count = cursor.rowcount
                if stale_count > 0:
                    logger.info(f"Cleaned up {stale_count} stale signals")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def save_signal(self, signal_data: Dict) -> bool:
        """Save trading signal"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check if signal already exists
                cursor.execute('SELECT id FROM signals WHERE signal_id = ?', 
                             (signal_data['signal_id'],))
                if cursor.fetchone():
                    logger.warning(f"Signal {signal_data['signal_id']} already exists")
                    return False
                
                # Check if coin already has active signal
                cursor.execute('''
                    SELECT signal_id FROM signals 
                    WHERE coin = ? AND status = 'active'
                    LIMIT 1
                ''', (signal_data['coin'],))
                
                if cursor.fetchone():
                    logger.warning(f"Coin {signal_data['coin']} already has active signal")
                    return False
                
                # Insert signal
                cursor.execute('''
                INSERT INTO signals (
                    signal_id, timestamp, coin, direction, entry_price, current_price,
                    take_profit, stop_loss, confidence, analysis_data, indicators,
                    ml_prediction, model_confidence, feature_importance, position_value,
                    leverage, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_data['signal_id'],
                    signal_data['timestamp'],
                    signal_data['coin'],
                    signal_data['direction'],
                    signal_data['entry_price'],
                    signal_data.get('current_price', signal_data['entry_price']),
                    signal_data['take_profit'],
                    signal_data['stop_loss'],
                    signal_data['confidence'],
                    json.dumps(signal_data.get('analysis_data', {})),
                    json.dumps(signal_data.get('indicators', {})),
                    signal_data.get('ml_prediction'),
                    signal_data.get('model_confidence'),
                    json.dumps(signal_data.get('feature_importance', {})),
                    signal_data.get('position_value'),
                    15,  # Fixed leverage
                    'active'
                ))
                
                # Create trade result entry
                cursor.execute('''
                INSERT INTO trade_results (
                    signal_id, entry_price, position_value, leverage, status
                ) VALUES (?, ?, ?, ?, ?)
                ''', (
                    signal_data['signal_id'],
                    signal_data['entry_price'],
                    signal_data.get('position_value'),
                    15,
                    'open'
                ))
                
                logger.info(f"Signal saved: {signal_data['signal_id']}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
            return False
    
    def get_active_signals(self) -> List[Dict]:
        """Get all active signals"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT s.*, tr.pnl_usd, tr.pnl_percentage
                FROM signals s
                LEFT JOIN trade_results tr ON s.signal_id = tr.signal_id
                WHERE s.status = 'active'
                ORDER BY s.created_at DESC
                ''')
                
                signals = []
                for row in cursor.fetchall():
                    signal_dict = dict(row)
                    
                    # Parse JSON fields
                    try:
                        signal_dict['analysis_data'] = json.loads(signal_dict.get('analysis_data', '{}'))
                        signal_dict['indicators'] = json.loads(signal_dict.get('indicators', '{}'))
                        signal_dict['feature_importance'] = json.loads(signal_dict.get('feature_importance', '{}'))
                    except:
                        signal_dict['analysis_data'] = {}
                        signal_dict['indicators'] = {}
                        signal_dict['feature_importance'] = {}
                    
                    signals.append(signal_dict)
                
                return signals
                
        except Exception as e:
            logger.error(f"Error getting active signals: {e}")
            return []
    
    def close_signal(self, signal_id: str, exit_price: float, exit_reason: str) -> bool:
        """Close a signal"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get signal details
                cursor.execute('''
                SELECT coin, direction, entry_price, position_value, leverage, created_at
                FROM signals 
                WHERE signal_id = ? AND status = 'active'
                ''', (signal_id,))
                
                result = cursor.fetchone()
                if not result:
                    logger.warning(f"Signal {signal_id} not found or already closed")
                    return False
                
                signal = dict(result)
                
                # Calculate P&L with actual leverage
                leverage = signal.get('leverage', 15)
                position_value = signal.get('position_value')
                
                if position_value:
                    if signal['direction'] == 'LONG':
                        pnl_percentage = ((exit_price - signal['entry_price']) / signal['entry_price']) * 100 * leverage
                    else:
                        pnl_percentage = ((signal['entry_price'] - exit_price) / signal['entry_price']) * 100 * leverage
                    
                    pnl_usd = (pnl_percentage / 100) * position_value
                else:
                    pnl_usd = 0
                    pnl_percentage = 0
                
                # Calculate duration
                try:
                    created_time = datetime.fromisoformat(signal['created_at'].replace('Z', '+00:00'))
                    duration_minutes = int((datetime.now() - created_time).total_seconds() / 60)
                except:
                    duration_minutes = 0
                
                # Update signal
                cursor.execute('''
                UPDATE signals 
                SET status = 'closed', 
                    exit_price = ?, 
                    exit_reason = ?,
                    closed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE signal_id = ?
                ''', (exit_price, exit_reason, signal_id))
                
                # Update trade result
                cursor.execute('''
                UPDATE trade_results 
                SET exit_price = ?, 
                    exit_reason = ?, 
                    status = 'closed',
                    pnl_usd = ?, 
                    pnl_percentage = ?, 
                    duration_minutes = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE signal_id = ?
                ''', (exit_price, exit_reason, pnl_usd, pnl_percentage, 
                      duration_minutes, signal_id))
                
                # Update portfolio
                self._update_portfolio(conn, pnl_usd)
                
                logger.info(f"Signal closed: {signal_id} - P&L: ${pnl_usd:.2f}")
                return True
                
        except Exception as e:
            logger.error(f"Error closing signal {signal_id}: {e}")
            return False
    
    def _update_portfolio(self, conn, pnl_usd: float):
        """Update portfolio statistics"""
        cursor = conn.cursor()
        
        # Get current portfolio
        cursor.execute('SELECT * FROM portfolio ORDER BY id DESC LIMIT 1')
        portfolio = dict(cursor.fetchone())
        
        # Update stats
        new_balance = portfolio['balance_usd'] + pnl_usd
        new_total_trades = portfolio['total_trades'] + 1
        new_winning_trades = portfolio['winning_trades'] + (1 if pnl_usd > 0 else 0)
        new_losing_trades = portfolio['losing_trades'] + (1 if pnl_usd <= 0 else 0)
        new_total_pnl = portfolio['total_pnl'] + pnl_usd
        
        # Update peak and drawdown
        if new_balance > portfolio['peak_balance']:
            new_peak_balance = new_balance
            new_max_drawdown = portfolio['max_drawdown']
        else:
            new_peak_balance = portfolio['peak_balance']
            current_drawdown = ((new_peak_balance - new_balance) / new_peak_balance) * 100
            new_max_drawdown = max(portfolio['max_drawdown'], current_drawdown)
        
        # Calculate win rate
        win_rate = (new_winning_trades / new_total_trades * 100) if new_total_trades > 0 else 0
        
        # Update portfolio
        cursor.execute('''
        UPDATE portfolio 
        SET balance_usd = ?, 
            total_trades = ?, 
            winning_trades = ?,
            losing_trades = ?, 
            total_pnl = ?, 
            max_drawdown = ?,
            peak_balance = ?, 
            win_rate = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        ''', (new_balance, new_total_trades, new_winning_trades,
              new_losing_trades, new_total_pnl, new_max_drawdown,
              new_peak_balance, win_rate, portfolio['id']))
    
    def update_signal_price(self, signal_id: str, current_price: float) -> bool:
        """Update signal current price"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                UPDATE signals 
                SET current_price = ?, updated_at = CURRENT_TIMESTAMP
                WHERE signal_id = ? AND status = 'active'
                ''', (current_price, signal_id))
                
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error updating signal price: {e}")
            return False
    
    def get_portfolio_stats(self) -> Dict:
        """Get portfolio statistics"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get portfolio
                cursor.execute('SELECT * FROM portfolio ORDER BY id DESC LIMIT 1')
                portfolio_row = cursor.fetchone()
                
                if not portfolio_row:
                    return {
                        'total_balance': 1000,
                        'total_trades': 0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'win_rate': 0,
                        'total_pnl': 0,
                        'max_drawdown': 0,
                        'peak_balance': 1000
                    }
                
                return dict(portfolio_row)
                
        except Exception as e:
            logger.error(f"Error getting portfolio stats: {e}")
            return {}
    
    def get_signal_by_coin(self, coin: str) -> Optional[Dict]:
        """Get active signal for a coin"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT * FROM signals 
                WHERE coin = ? AND status = 'active'
                LIMIT 1
                ''', (coin,))
                
                row = cursor.fetchone()
                if row:
                    signal_dict = dict(row)
                    try:
                        signal_dict['analysis_data'] = json.loads(signal_dict.get('analysis_data', '{}'))
                        signal_dict['indicators'] = json.loads(signal_dict.get('indicators', '{}'))
                        signal_dict['feature_importance'] = json.loads(signal_dict.get('feature_importance', '{}'))
                    except:
                        pass
                    return signal_dict
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting signal for {coin}: {e}")
            return None
    
    def save_ml_model(self, coin: str, model_data: Dict) -> bool:
        """Save ML model"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Serialize model and scaler
                model_blob = pickle.dumps(model_data.get('model')) if model_data.get('model') else None
                scaler_blob = pickle.dumps(model_data.get('scaler')) if model_data.get('scaler') else None
                
                cursor.execute('''
                INSERT OR REPLACE INTO ml_models (
                    coin, model_type, model_data, scaler_data, 
                    feature_names, feature_importance, cv_score, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    coin,
                    model_data.get('model_type', 'unknown'),
                    model_blob,
                    scaler_blob,
                    json.dumps(model_data.get('features', [])),
                    json.dumps(model_data.get('importance', {})),
                    model_data.get('cv_score', 0)
                ))
                
                return True
                
        except Exception as e:
            logger.error(f"Error saving ML model: {e}")
            return False
    
    def load_ml_model(self, coin: str) -> Optional[Dict]:
        """Load ML model"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT * FROM ml_models 
                WHERE coin = ? 
                ORDER BY updated_at DESC 
                LIMIT 1
                ''', (coin,))
                
                row = cursor.fetchone()
                if row:
                    model_dict = dict(row)
                    
                    # Deserialize
                    if model_dict['model_data']:
                        model_dict['model'] = pickle.loads(model_dict['model_data'])
                    if model_dict['scaler_data']:
                        model_dict['scaler'] = pickle.loads(model_dict['scaler_data'])
                    
                    # Parse JSON
                    model_dict['features'] = json.loads(model_dict.get('feature_names', '[]'))
                    model_dict['importance'] = json.loads(model_dict.get('feature_importance', '{}'))
                    
                    return model_dict
                
                return None
                
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            return None
    
    def log_bot_activity(self, level: str, component: str, message: str, 
                        details: str = None):
        """Log bot activity"""
        # Skip verbose DEBUG logs
        if level == 'DEBUG':
            return
        
        # Cache log entry
        with self.cache_lock:
            self.log_cache.append({
                'level': level,
                'component': component,
                'message': message,
                'details': details,
                'timestamp': datetime.now().isoformat()
            })
            
            # Flush if needed
            if len(self.log_cache) > 20 or time.time() - self.last_flush > 30:
                self._flush_log_cache()
                self.last_flush = time.time()
    
    def _flush_log_cache(self):
        """Flush log cache to database"""
        if not self.log_cache:
            return
        
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.executemany('''
                INSERT INTO bot_logs (level, component, message, details, timestamp)
                VALUES (?, ?, ?, ?, ?)
                ''', [
                    (log['level'], log['component'], log['message'], 
                     log['details'], log['timestamp'])
                    for log in self.log_cache
                ])
                
                self.log_cache.clear()
                
        except Exception as e:
            logger.error(f"Failed to flush log cache: {e}")
            self.log_cache.clear()
    
    def get_recent_logs(self, limit: int = 50) -> List[Dict]:
        """Get recent logs"""
        try:
            # Flush cache first
            with self.cache_lock:
                if self.log_cache:
                    self._flush_log_cache()
            
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT * FROM bot_logs 
                ORDER BY timestamp DESC 
                LIMIT ?
                ''', (limit,))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting logs: {e}")
            return []
    
    def clean_old_data(self, days: int = 7):
        """Clean old data"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Clean old closed signals
                cursor.execute('''
                DELETE FROM signals 
                WHERE status = 'closed' 
                AND closed_at < datetime('now', '-{} days')
                '''.format(days))
                
                # Clean old logs (keep last 1000)
                cursor.execute('''
                DELETE FROM bot_logs 
                WHERE id NOT IN (
                    SELECT id FROM bot_logs 
                    ORDER BY timestamp DESC 
                    LIMIT 1000
                )
                ''')
                
                logger.info(f"Cleaned old data (>{days} days)")
                
        except Exception as e:
            logger.error(f"Error cleaning old data: {e}")

# For backward compatibility
MLEnhancedDatabase = Database