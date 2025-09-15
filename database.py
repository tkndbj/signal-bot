import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
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
        
        # Clean up any inconsistent state on startup
        self.cleanup_on_startup()
    
    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections with proper error handling"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            # Enable WAL mode for better concurrency
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            conn.execute('PRAGMA foreign_keys=ON')
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise e
        finally:
            if conn:
                conn.close()
    
    def init_database(self):
        """Initialize database with required tables and proper constraints"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
        
            # Add this SIGNALS table first:
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
                status TEXT DEFAULT 'active' CHECK (status IN ('active', 'closed', 'cancelled')),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                closed_at DATETIME,
                exit_price REAL CHECK (exit_price > 0),
                exit_reason TEXT
            )
            ''')
            
            # Enhanced signals table with better constraints
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT NOT NULL,
                entry_price REAL NOT NULL CHECK (entry_price > 0),
                exit_price REAL CHECK (exit_price > 0),
                pnl_usd REAL DEFAULT 0,
                pnl_percentage REAL DEFAULT 0,
                max_profit REAL DEFAULT 0,
                max_loss REAL DEFAULT 0,
                duration_minutes INTEGER DEFAULT 0,
                exit_reason TEXT,
                leverage INTEGER DEFAULT 10 CHECK (leverage > 0),
                position_size_usd REAL DEFAULT 1000 CHECK (position_size_usd > 0),
                status TEXT DEFAULT 'open' CHECK (status IN ('open', 'closed')),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (signal_id) REFERENCES signals (signal_id) ON DELETE CASCADE
            )
            ''')
            
            # Portfolio table
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
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                CHECK (winning_trades + losing_trades <= total_trades)
            )
            ''')
            
            # Bot logs table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS bot_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                level TEXT NOT NULL CHECK (level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR')),
                component TEXT NOT NULL,
                message TEXT NOT NULL,
                details TEXT,
                coin TEXT,
                data TEXT
            )
            ''')
            
            # Analysis logs table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                coin TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                result TEXT NOT NULL,
                confidence REAL CHECK (confidence >= 0 AND confidence <= 100),
                indicators TEXT,
                patterns TEXT,
                support_resistance TEXT
            )
            ''')
            
            # Create optimized indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_coin ON signals(coin)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_signal_id ON signals(signal_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_results_signal_id ON trade_results(signal_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bot_logs_timestamp ON bot_logs(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bot_logs_level ON bot_logs(level)')
            
            # Initialize portfolio if empty
            cursor.execute('SELECT COUNT(*) FROM portfolio')
            if cursor.fetchone()[0] == 0:
                cursor.execute('''
                INSERT INTO portfolio (balance_usd, total_trades, winning_trades, losing_trades, total_pnl, peak_balance)
                VALUES (1000, 0, 0, 0, 0, 1000)
                ''')
            
            logger.info("Database initialized successfully")
    
    def cleanup_on_startup(self):
        """Clean up any inconsistent state on startup"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check for orphaned trade results without corresponding signals
                cursor.execute('''
                DELETE FROM trade_results 
                WHERE signal_id NOT IN (SELECT signal_id FROM signals)
                ''')
                
                # Check for signals with inconsistent price relationships
                cursor.execute('''
                SELECT signal_id, direction, entry_price, take_profit, stop_loss 
                FROM signals 
                WHERE status = 'active'
                ''')
                
                invalid_signals = []
                for row in cursor.fetchall():
                    signal_id, direction, entry, tp, sl = row
                    
                    # Validate price relationships
                    if direction == 'LONG' and not (sl < entry < tp):
                        invalid_signals.append(signal_id)
                    elif direction == 'SHORT' and not (tp < entry < sl):
                        invalid_signals.append(signal_id)
                
                # Close invalid signals
                for signal_id in invalid_signals:
                    cursor.execute('''
                    UPDATE signals 
                    SET status = 'cancelled', exit_reason = 'Invalid price configuration', closed_at = CURRENT_TIMESTAMP
                    WHERE signal_id = ?
                    ''', (signal_id,))
                    
                    cursor.execute('''
                    UPDATE trade_results 
                    SET status = 'closed', exit_reason = 'Invalid price configuration'
                    WHERE signal_id = ?
                    ''', (signal_id,))
                
                if invalid_signals:
                    logger.warning(f"Cleaned up {len(invalid_signals)} invalid signals on startup")
                
        except Exception as e:
            logger.error(f"Error during startup cleanup: {e}")
    
    def log_bot_activity(self, level: str, component: str, message: str, 
                        details: str = None, coin: str = None, data: Dict = None):
        """Log bot activity with intelligent caching"""
        
        # Skip verbose DEBUG logs for non-critical components
        if level == 'DEBUG' and component not in ['SIGNAL_MANAGER', 'SIGNAL_MONITOR']:
            return
        
        # Cache log entry
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
            
            # Flush cache when it gets large or after time interval
            current_time = time.time()
            if (current_time - self.last_flush > 30) or len(self.log_cache) > 15:
                self._flush_log_cache()
                self.last_flush = current_time
        
        # Always log important messages to console
        log_msg = f"[{component}] {message}"
        if details:
            log_msg += f" - {details}"
        
        if level == 'ERROR':
            logger.error(log_msg)
        elif level == 'WARNING':
            logger.warning(log_msg)
        elif level == 'INFO':
            logger.info(log_msg)
    
    def _flush_log_cache(self):
        """Flush cached logs to database"""
        if not self.log_cache:
            return
        
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.executemany('''
                INSERT INTO bot_logs (level, component, message, details, coin, data, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', [
                    (log['level'], log['component'], log['message'], 
                     log['details'], log['coin'], log['data'], log['timestamp'])
                    for log in self.log_cache
                ])
                
                self.log_cache.clear()
                
        except Exception as e:
            logger.error(f"Failed to flush log cache: {e}")
            self.log_cache.clear()
    
    def save_signal(self, signal_data: Dict) -> bool:
        """Save a new trading signal with comprehensive validation"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check if signal already exists
                cursor.execute('SELECT id FROM signals WHERE signal_id = ?', (signal_data['signal_id'],))
                if cursor.fetchone():
                    logger.warning(f"Signal {signal_data['signal_id']} already exists")
                    return False
                
                # Validate signal data
                required_fields = ['signal_id', 'timestamp', 'coin', 'direction', 
                                 'entry_price', 'take_profit', 'stop_loss', 'confidence']
                for field in required_fields:
                    if field not in signal_data:
                        logger.error(f"Missing required field: {field}")
                        return False
                
                # Insert signal
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
                    signal_data.get('current_price', signal_data['entry_price']),
                    signal_data['take_profit'],
                    signal_data['stop_loss'],
                    signal_data['confidence'],
                    json.dumps(signal_data.get('analysis_data', {})),
                    json.dumps(signal_data.get('indicators', {})),
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
                
                logger.info(f"Signal saved successfully: {signal_data['signal_id']}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
            return False
    
    def get_active_signals(self) -> List[Dict]:
        """Get all active signals with comprehensive data"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT s.*, tr.pnl_usd, tr.pnl_percentage, tr.max_profit, tr.max_loss, 
                       tr.position_size_usd, tr.leverage
                FROM signals s
                LEFT JOIN trade_results tr ON s.signal_id = tr.signal_id
                WHERE s.status = 'active'
                ORDER BY s.created_at DESC
                ''')
                
                signals = []
                for row in cursor.fetchall():
                    signal_dict = dict(row)
                    
                    # Parse JSON fields safely
                    try:
                        signal_dict['analysis_data'] = json.loads(signal_dict['analysis_data'] or '{}')
                        signal_dict['indicators'] = json.loads(signal_dict['indicators'] or '{}')
                    except json.JSONDecodeError:
                        signal_dict['analysis_data'] = {}
                        signal_dict['indicators'] = {}
                    
                    # Ensure numeric fields have safe defaults
                    signal_dict['pnl_usd'] = signal_dict.get('pnl_usd') or 0
                    signal_dict['pnl_percentage'] = signal_dict.get('pnl_percentage') or 0
                    signal_dict['max_profit'] = signal_dict.get('max_profit') or 0
                    signal_dict['max_loss'] = signal_dict.get('max_loss') or 0
                    
                    signals.append(signal_dict)
                
                return signals
                
        except Exception as e:
            logger.error(f"Error getting active signals: {e}")
            return []
    
    def update_signal_price(self, signal_id: str, current_price: float) -> bool:
        """Update signal current price and calculate live P&L"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Update signal current price
                cursor.execute('''
                UPDATE signals 
                SET current_price = ?, updated_at = CURRENT_TIMESTAMP
                WHERE signal_id = ? AND status = 'active'
                ''', (current_price, signal_id))
                
                if cursor.rowcount == 0:
                    return False
                
                # Get signal details for P&L calculation
                cursor.execute('''
                SELECT direction, entry_price FROM signals 
                WHERE signal_id = ? AND status = 'active'
                ''', (signal_id,))
                
                result = cursor.fetchone()
                if not result:
                    return False
                
                direction, entry_price = result
                
                # Calculate P&L with 10x leverage
                if direction == 'LONG':
                    pnl_percentage = ((current_price - entry_price) / entry_price) * 100 * 10
                else:  # SHORT
                    pnl_percentage = ((entry_price - current_price) / entry_price) * 100 * 10
                
                pnl_usd = (pnl_percentage / 100) * 1000  # $1000 position size
                
                # Update trade result with current P&L and track extremes
                cursor.execute('''
                UPDATE trade_results 
                SET pnl_usd = ?, pnl_percentage = ?, updated_at = CURRENT_TIMESTAMP,
                    max_profit = CASE WHEN ? > max_profit THEN ? ELSE max_profit END,
                    max_loss = CASE WHEN ? < max_loss THEN ? ELSE max_loss END
                WHERE signal_id = ? AND status = 'open'
                ''', (pnl_usd, pnl_percentage, pnl_usd, pnl_usd, pnl_usd, pnl_usd, signal_id))
                
                return True
                
        except Exception as e:
            logger.error(f"Error updating signal price for {signal_id}: {e}")
            return False
    
    def close_signal(self, signal_id: str, exit_price: float, exit_reason: str) -> bool:
        """Close a signal and finalize all related data"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get signal details
                cursor.execute('''
                SELECT coin, direction, entry_price, created_at 
                FROM signals 
                WHERE signal_id = ? AND status = 'active'
                ''', (signal_id,))
                
                result = cursor.fetchone()
                if not result:
                    logger.warning(f"Signal {signal_id} not found or already closed")
                    return False
                
                coin, direction, entry_price, created_at = result
                
                # Calculate trade duration
                try:
                    if isinstance(created_at, str):
                        created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    else:
                        created_time = created_at
                    duration_minutes = int((datetime.now() - created_time).total_seconds() / 60)
                except:
                    duration_minutes = 0
                
                # Calculate final P&L with 10x leverage
                if direction == 'LONG':
                    pnl_percentage = ((exit_price - entry_price) / entry_price) * 100 * 10
                else:  # SHORT
                    pnl_percentage = ((entry_price - exit_price) / entry_price) * 100 * 10
                
                pnl_usd = (pnl_percentage / 100) * 1000
                
                # Update signal status
                cursor.execute('''
                UPDATE signals 
                SET status = 'closed', exit_price = ?, exit_reason = ?, 
                    closed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE signal_id = ?
                ''', (exit_price, exit_reason, signal_id))
                
                # Update trade result
                cursor.execute('''
                UPDATE trade_results 
                SET exit_price = ?, exit_reason = ?, status = 'closed',
                    pnl_usd = ?, pnl_percentage = ?, duration_minutes = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE signal_id = ?
                ''', (exit_price, exit_reason, pnl_usd, pnl_percentage, duration_minutes, signal_id))
                
                # Update portfolio statistics
                self._update_portfolio_stats(conn, pnl_usd)
                
                logger.info(f"Signal closed successfully: {signal_id} - P&L: ${pnl_usd:.2f}")
                return True
                
        except Exception as e:
            logger.error(f"Error closing signal {signal_id}: {e}")
            return False
    
    def _update_portfolio_stats(self, conn, pnl_usd: float):
        """Update portfolio statistics after trade closure"""
        try:
            cursor = conn.cursor()
            
            # Get current portfolio stats
            cursor.execute('SELECT * FROM portfolio ORDER BY id DESC LIMIT 1')
            portfolio = cursor.fetchone()
            
            if not portfolio:
                # Initialize portfolio if it doesn't exist
                cursor.execute('''
                INSERT INTO portfolio (balance_usd, total_trades, winning_trades, losing_trades, total_pnl, peak_balance)
                VALUES (1000, 1, ?, 0, ?, 1000)
                ''', (1 if pnl_usd > 0 else 0, pnl_usd))
                return
            
            # Calculate new stats
            balance = portfolio['balance_usd']
            total_trades = portfolio['total_trades']
            winning_trades = portfolio['winning_trades']
            losing_trades = portfolio['losing_trades']
            total_pnl = portfolio['total_pnl']
            max_drawdown = portfolio['max_drawdown']
            peak_balance = portfolio['peak_balance']
            
            # Update counters
            new_balance = balance + pnl_usd
            new_total_trades = total_trades + 1
            new_winning_trades = winning_trades + (1 if pnl_usd > 0 else 0)
            new_losing_trades = losing_trades + (1 if pnl_usd <= 0 else 0)
            new_total_pnl = total_pnl + pnl_usd
            
            # Update peak balance and calculate drawdown
            if new_balance > peak_balance:
                new_peak_balance = new_balance
                new_max_drawdown = max_drawdown  # Keep existing if we're at new peak
            else:
                new_peak_balance = peak_balance
                current_drawdown = ((peak_balance - new_balance) / peak_balance) * 100
                new_max_drawdown = max(max_drawdown, current_drawdown)
            
            # Update portfolio record
            cursor.execute('''
            UPDATE portfolio 
            SET balance_usd = ?, total_trades = ?, winning_trades = ?, 
                losing_trades = ?, total_pnl = ?, max_drawdown = ?, 
                peak_balance = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            ''', (new_balance, new_total_trades, new_winning_trades, 
                  new_losing_trades, new_total_pnl, new_max_drawdown, 
                  new_peak_balance, portfolio['id']))
            
        except Exception as e:
            logger.error(f"Error updating portfolio stats: {e}")
    
    def get_portfolio_stats(self) -> Dict:
        """Get comprehensive portfolio statistics"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get latest portfolio record
                cursor.execute('SELECT * FROM portfolio ORDER BY id DESC LIMIT 1')
                portfolio = cursor.fetchone()
                
                if not portfolio:
                    return self._get_default_portfolio_stats()
                
                # Get current open P&L
                cursor.execute('''
                SELECT COALESCE(SUM(pnl_usd), 0) as open_pnl
                FROM trade_results 
                WHERE status = 'open'
                ''')
                open_pnl = cursor.fetchone()['open_pnl']
                
                # Get additional trade statistics
                cursor.execute('''
                SELECT 
                    COALESCE(AVG(pnl_usd), 0) as avg_pnl,
                    COALESCE(MAX(pnl_usd), 0) as best_trade,
                    COALESCE(MIN(pnl_usd), 0) as worst_trade,
                    COALESCE(AVG(CASE WHEN pnl_usd > 0 THEN pnl_usd END), 0) as avg_win,
                    COALESCE(AVG(CASE WHEN pnl_usd <= 0 THEN pnl_usd END), 0) as avg_loss,
                    COALESCE(AVG(duration_minutes), 0) as avg_duration
                FROM trade_results 
                WHERE status = 'closed'
                ''')
                
                trade_stats = cursor.fetchone()
                
                total_trades = portfolio['total_trades']
                winning_trades = portfolio['winning_trades']
                
                return {
                    'total_balance': portfolio['balance_usd'] + open_pnl,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': portfolio['losing_trades'],
                    'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                    'total_pnl': portfolio['total_pnl'] + open_pnl,
                    'avg_pnl': trade_stats['avg_pnl'],
                    'best_trade': trade_stats['best_trade'],
                    'worst_trade': trade_stats['worst_trade'],
                    'avg_win': trade_stats['avg_win'],
                    'avg_loss': trade_stats['avg_loss'],
                    'max_drawdown': portfolio['max_drawdown'],
                    'open_pnl': open_pnl,
                    'peak_balance': portfolio['peak_balance'],
                    'avg_duration_minutes': trade_stats['avg_duration']
                }
                
        except Exception as e:
            logger.error(f"Error getting portfolio stats: {e}")
            return self._get_default_portfolio_stats()
    
    def _get_default_portfolio_stats(self) -> Dict:
        """Return default portfolio stats when data is unavailable"""
        return {
            'total_balance': 1000,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_drawdown': 0,
            'open_pnl': 0,
            'peak_balance': 1000,
            'avg_duration_minutes': 0
        }
    
    def get_recent_logs(self, limit: int = 100) -> List[Dict]:
        """Get recent bot activity logs"""
        try:
            # Flush any pending logs first
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
                
                logs = []
                for row in cursor.fetchall():
                    log_dict = dict(row)
                    if log_dict.get('data'):
                        try:
                            log_dict['data'] = json.loads(log_dict['data'])
                        except:
                            log_dict['data'] = None
                    logs.append(log_dict)
                
                return logs
                
        except Exception as e:
            logger.error(f"Error getting recent logs: {e}")
            return []
    
    def clean_old_data(self, days: int = 7):
        """Clean old data to maintain database performance"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Clean old closed signals
                cursor.execute('''
                DELETE FROM signals 
                WHERE status = 'closed' 
                AND closed_at < datetime('now', '-{} days')
                '''.format(days))
                deleted_signals = cursor.rowcount
                
                # Clean old logs (keep last 1000 important ones)
                cursor.execute('''
                DELETE FROM bot_logs 
                WHERE id NOT IN (
                    SELECT id FROM bot_logs 
                    WHERE level IN ('ERROR', 'WARNING', 'INFO')
                    ORDER BY timestamp DESC 
                    LIMIT 1000
                )
                ''')
                deleted_logs = cursor.rowcount
                
                # Clean old analysis logs
                cursor.execute('''
                DELETE FROM analysis_logs 
                WHERE timestamp < datetime('now', '-{} days')
                '''.format(days))
                deleted_analysis = cursor.rowcount
                
                logger.info(f"Cleaned old data: {deleted_signals} signals, {deleted_logs} logs, {deleted_analysis} analysis records")
                
        except Exception as e:
            logger.error(f"Error cleaning old data: {e}")
    
    def get_signal_by_id(self, signal_id: str) -> Optional[Dict]:
        """Get a specific signal by ID"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT s.*, tr.pnl_usd, tr.pnl_percentage 
                FROM signals s
                LEFT JOIN trade_results tr ON s.signal_id = tr.signal_id
                WHERE s.signal_id = ?
                ''', (signal_id,))
                
                row = cursor.fetchone()
                if row:
                    signal_dict = dict(row)
                    try:
                        signal_dict['analysis_data'] = json.loads(signal_dict['analysis_data'] or '{}')
                        signal_dict['indicators'] = json.loads(signal_dict['indicators'] or '{}')
                    except:
                        signal_dict['analysis_data'] = {}
                        signal_dict['indicators'] = {}
                    return signal_dict
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting signal {signal_id}: {e}")
            return None