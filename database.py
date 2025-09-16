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

class MLEnhancedDatabase:
    def __init__(self, db_path: str = "data/ml_crypto_bot.db"):
        self.db_path = db_path
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Thread-safe logging cache
        self.log_cache = []
        self.cache_lock = threading.Lock()
        self.last_flush = time.time()
        
        # Initialize ML-enhanced database
        self.init_ml_database()
        
        # Clean up any inconsistent state on startup
        self.cleanup_on_startup()
    
    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections with ML optimizations"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            # Enable WAL mode for better concurrency
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            conn.execute('PRAGMA foreign_keys=ON')
            conn.execute('PRAGMA cache_size=10000')  # Increased cache for ML data
            conn.row_factory = sqlite3.Row
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

    def row_to_dict(self, row):
        """Convert sqlite3.Row to dictionary safely"""
        if row is None:
            return None
        return dict(row) if hasattr(row, 'keys') else row
    
    def init_ml_database(self):
        """Initialize ML-enhanced database with new tables for ML features"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
        
            # Enhanced signals table with ML fields
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
                
                -- ML-specific fields
                ml_prediction REAL,
                model_confidence REAL CHECK (model_confidence >= 0 AND model_confidence <= 1),
                feature_importance TEXT,  -- JSON of feature importance scores
                model_type TEXT,  -- rf, gbm, etc.
                prediction_horizon INTEGER DEFAULT 24,  -- hours
                feature_count INTEGER DEFAULT 0,
                orthogonality_score REAL DEFAULT 0,
                
                status TEXT DEFAULT 'active' CHECK (status IN ('active', 'closed', 'cancelled')),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                closed_at DATETIME,
                exit_price REAL CHECK (exit_price > 0),
                exit_reason TEXT
            )
            ''')
            
            # ML Models tracking table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                model_type TEXT NOT NULL,
                model_data BLOB,  -- Pickled model
                scaler_data BLOB,  -- Pickled scaler
                feature_names TEXT,  -- JSON array of feature names
                feature_importance TEXT,  -- JSON of feature importance
                cv_score REAL,
                accuracy REAL DEFAULT 0,
                precision_score REAL DEFAULT 0,
                recall_score REAL DEFAULT 0,
                training_samples INTEGER DEFAULT 0,
                validation_score REAL DEFAULT 0,
                feature_count INTEGER DEFAULT 0,
                orthogonality_applied BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_used DATETIME,
                performance_score REAL DEFAULT 0,
                UNIQUE(coin, model_type)
            )
            ''')
            
            # Feature engineering tracking
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_engineering_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                raw_features_count INTEGER,
                engineered_features_count INTEGER,
                selected_features_count INTEGER,
                orthogonalized_features_count INTEGER,
                correlation_threshold REAL,
                selection_method TEXT,
                feature_categories TEXT,  -- JSON breakdown by category
                computation_time_ms INTEGER,
                data_quality_score REAL
            )
            ''')
            
            # ML Predictions tracking
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT NOT NULL,
                coin TEXT NOT NULL,
                prediction_value REAL NOT NULL,
                model_confidence REAL NOT NULL,
                model_type TEXT NOT NULL,
                actual_outcome REAL,  -- filled when signal closes
                prediction_accuracy REAL,  -- calculated when closed
                feature_importance TEXT,  -- JSON of top features
                shap_values TEXT,  -- JSON of SHAP values if available
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                outcome_at DATETIME,
                FOREIGN KEY (signal_id) REFERENCES signals (signal_id) ON DELETE CASCADE
            )
            ''')
            
            # Enhanced trade results with ML metrics
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
                
                -- ML performance tracking
                ml_prediction_accuracy REAL,
                model_confidence_at_entry REAL,
                prediction_error REAL,
                feature_stability_score REAL,
                
                status TEXT DEFAULT 'open' CHECK (status IN ('open', 'closed')),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (signal_id) REFERENCES signals (signal_id) ON DELETE CASCADE
            )
            ''')
            
            # Feature importance trends
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_importance_trends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                importance_score REAL NOT NULL,
                rank_position INTEGER,
                model_type TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                stability_score REAL DEFAULT 0,
                trend_direction TEXT CHECK (trend_direction IN ('up', 'down', 'stable'))                
            )
            ''')
            
            # Model performance history
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                model_type TEXT NOT NULL,
                accuracy REAL NOT NULL,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                cv_score REAL,
                prediction_count INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0,
                feature_count INTEGER DEFAULT 0,
                training_time_ms INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP                
            )
            ''')
            
            # Enhanced portfolio with ML metrics
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
                
                -- ML-specific portfolio metrics
                ml_model_accuracy REAL DEFAULT 0,
                avg_prediction_confidence REAL DEFAULT 0,
                feature_orthogonality_score REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,
                sortino_ratio REAL DEFAULT 0,
                calmar_ratio REAL DEFAULT 0,
                ml_signals_count INTEGER DEFAULT 0,
                ml_success_rate REAL DEFAULT 0,
                
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                CHECK (winning_trades + losing_trades <= total_trades)
            )
            ''')
            
            # Bot logs (unchanged but indexed for ML queries)
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
            
            # Time series validation results
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS time_series_validation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                model_type TEXT NOT NULL,
                fold_number INTEGER NOT NULL,
                train_start DATETIME,
                train_end DATETIME,
                validation_start DATETIME,
                validation_end DATETIME,
                validation_score REAL NOT NULL,
                feature_count INTEGER,
                samples_count INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP                
            )
            ''')
            
            # Create optimized indexes for ML queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_ml_confidence ON signals(model_confidence DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_ml_prediction ON signals(ml_prediction)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_coin_status ON signals(coin, status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ml_models_coin ON ml_models(coin)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ml_models_performance ON ml_models(performance_score DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ml_predictions_coin ON ml_predictions(coin, created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_feature_trends_coin_time ON feature_importance_trends(coin, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_performance_coin_time ON model_performance_history(coin, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bot_logs_timestamp ON bot_logs(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bot_logs_level_component ON bot_logs(level, component)')
            
            # Initialize portfolio if empty
            cursor.execute('SELECT COUNT(*) FROM portfolio')
            if cursor.fetchone()[0] == 0:
                cursor.execute('''
                INSERT INTO portfolio (
                    balance_usd, total_trades, winning_trades, losing_trades, total_pnl, 
                    peak_balance, ml_model_accuracy, avg_prediction_confidence, 
                    feature_orthogonality_score
                ) VALUES (1000, 0, 0, 0, 0, 1000, 0, 0, 0)
                ''')
            
            logger.info("ML-enhanced database initialized successfully")
    
    def cleanup_on_startup(self):
        """Enhanced cleanup for ML components"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Standard cleanup from parent class
                cursor.execute('''
                DELETE FROM trade_results 
                WHERE signal_id NOT IN (SELECT signal_id FROM signals)
                ''')
                
                # ML-specific cleanup: orphaned predictions
                cursor.execute('''
                DELETE FROM ml_predictions 
                WHERE signal_id NOT IN (SELECT signal_id FROM signals)
                ''')
                
                # Clean up old ML model data (keep last 3 versions per coin)
                cursor.execute('''
                DELETE FROM ml_models 
                WHERE id NOT IN (
                    SELECT id FROM ml_models m1
                    WHERE (
                        SELECT COUNT(*) FROM ml_models m2 
                        WHERE m2.coin = m1.coin AND m2.model_type = m1.model_type 
                        AND m2.updated_at >= m1.updated_at
                    ) <= 3
                )
                ''')
                
                # Validate signal price relationships
                cursor.execute('''
                SELECT signal_id, direction, entry_price, take_profit, stop_loss 
                FROM signals 
                WHERE status = 'active'
                ''')
                
                invalid_signals = []
                for row in cursor.fetchall():
                    signal_id, direction, entry, tp, sl = row
                    
                    if direction == 'LONG' and not (sl < entry < tp):
                        invalid_signals.append(signal_id)
                    elif direction == 'SHORT' and not (tp < entry < sl):
                        invalid_signals.append(signal_id)
                
                # Close invalid signals
                for signal_id in invalid_signals:
                    cursor.execute('''
                    UPDATE signals 
                    SET status = 'cancelled', exit_reason = 'Invalid price configuration', 
                        closed_at = CURRENT_TIMESTAMP
                    WHERE signal_id = ?
                    ''', (signal_id,))
                    
                    cursor.execute('''
                    UPDATE trade_results 
                    SET status = 'closed', exit_reason = 'Invalid price configuration'
                    WHERE signal_id = ?
                    ''', (signal_id,))
                
                if invalid_signals:
                    logger.warning(f"ML cleanup: Fixed {len(invalid_signals)} invalid signals")
                
        except Exception as e:
            logger.error(f"Error during ML startup cleanup: {e}")
    
    def save_ml_signal(self, signal_data: Dict) -> bool:
        """Save ML-enhanced trading signal"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
        
                # Check if signal already exists
                cursor.execute('SELECT id FROM signals WHERE signal_id = ?', (signal_data['signal_id'],))
                if cursor.fetchone():
                    logger.warning(f"Signal {signal_data['signal_id']} already exists")
                    return False
            
                # Check if coin already has active signal
                cursor.execute('''
                    SELECT signal_id FROM signals 
                    WHERE coin = ? AND status = 'active'
                    LIMIT 1
                ''', (signal_data['coin'],))
            
                existing_signal = cursor.fetchone()
                if existing_signal:
                    logger.warning(f"Coin {signal_data['coin']} already has active signal: {existing_signal[0]}")
                    return False
        
                # Validate required fields
                required_fields = ['signal_id', 'timestamp', 'coin', 'direction', 
                                'entry_price', 'take_profit', 'stop_loss', 'confidence']
                for field in required_fields:
                    if field not in signal_data:
                        logger.error(f"Missing required field: {field}")
                        return False
        
                # Insert ML-enhanced signal
                cursor.execute('''
                INSERT INTO signals (
                    signal_id, timestamp, coin, direction, entry_price, current_price,
                    take_profit, stop_loss, confidence, analysis_data, indicators, 
                    ml_prediction, model_confidence, feature_importance, model_type,
                    prediction_horizon, feature_count, orthogonality_score, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    signal_data.get('analysis_data', {}).get('model_type'),
                    signal_data.get('prediction_horizon', 24),
                    len(signal_data.get('feature_importance', {})),
                    signal_data.get('orthogonality_score', 0),
                    'active'
                ))
        
                # Create corresponding trade result entry with ML metrics
                cursor.execute('''
                INSERT INTO trade_results (
                    signal_id, entry_price, position_size_usd, leverage, status,
                    model_confidence_at_entry, feature_stability_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_data['signal_id'],
                    signal_data['entry_price'],
                    1000,  # $1000 position size
                    10,    # 10x leverage
                    'open',
                    signal_data.get('model_confidence', 0),
                    signal_data.get('feature_stability_score', 0.5)
                ))
                
                # Save ML prediction tracking
                self.save_ml_prediction(signal_data)
        
                logger.info(f"ML signal saved successfully: {signal_data['signal_id']}")
                return True
        
        except Exception as e:
            logger.error(f"Error saving ML signal: {e}")
            return False
    
    def save_ml_prediction(self, signal_data: Dict) -> bool:
        """Save ML prediction for tracking"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                feature_importance = signal_data.get('feature_importance', {})
                shap_values = signal_data.get('shap_importance', {})
                
                cursor.execute('''
                INSERT INTO ml_predictions (
                    signal_id, coin, prediction_value, model_confidence, model_type,
                    feature_importance, shap_values
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_data['signal_id'],
                    signal_data['coin'],
                    signal_data.get('ml_prediction', 0),
                    signal_data.get('model_confidence', 0),
                    signal_data.get('analysis_data', {}).get('model_type', 'unknown'),
                    json.dumps(feature_importance),
                    json.dumps(shap_values)
                ))
                
                return True
                
        except Exception as e:
            logger.error(f"Error saving ML prediction: {e}")
            return False
    
    def save_ml_model(self, coin: str, model_data: Dict) -> bool:
        """Save ML model and associated data"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Serialize model and scaler
                model_blob = pickle.dumps(model_data['model']) if model_data.get('model') else None
                scaler_blob = pickle.dumps(model_data['scaler']) if model_data.get('scaler') else None
                
                # Insert or update ML model
                cursor.execute('''
                INSERT OR REPLACE INTO ml_models (
                    coin, model_type, model_data, scaler_data, feature_names,
                    feature_importance, cv_score, training_samples, feature_count,
                    orthogonality_applied, performance_score, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    coin,
                    model_data.get('model_type', 'unknown'),
                    model_blob,
                    scaler_blob,
                    json.dumps(model_data.get('features', [])),
                    json.dumps(model_data.get('importance', {})),
                    model_data.get('cv_score', 0),
                    model_data.get('training_samples', 0),
                    len(model_data.get('features', [])),
                    model_data.get('orthogonality_applied', False),
                    1.0 / (1.0 + model_data.get('cv_score', 1.0)),  # Convert error to score
                ))
                
                # Save feature importance trends
                self.save_feature_importance_trend(coin, model_data)
                
                logger.info(f"ML model saved for {coin}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving ML model for {coin}: {e}")
            return False
    
    def save_feature_importance_trend(self, coin: str, model_data: Dict):
        """Save feature importance trends for analysis"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                importance_dict = model_data.get('importance', {})
                model_type = model_data.get('model_type', 'unknown')
                
                for rank, (feature, importance) in enumerate(
                    sorted(importance_dict.items(), key=lambda x: x[1], reverse=True), 1):
                    
                    cursor.execute('''
                    INSERT INTO feature_importance_trends (
                        coin, feature_name, importance_score, rank_position, model_type
                    ) VALUES (?, ?, ?, ?, ?)
                    ''', (coin, feature, importance, rank, model_type))
                
        except Exception as e:
            logger.error(f"Error saving feature importance trends: {e}")
    
    def load_ml_model(self, coin: str, model_type: str = None) -> Optional[Dict]:
        """Load ML model and associated data"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                query = '''
                SELECT * FROM ml_models 
                WHERE coin = ? AND performance_score > 0
                '''
                params = [coin]
                
                if model_type:
                    query += ' AND model_type = ?'
                    params.append(model_type)
                
                query += ' ORDER BY performance_score DESC, updated_at DESC LIMIT 1'
                
                cursor.execute(query, params)
                row = cursor.fetchone()
                
                if row:
                    model_dict = dict(row)
                    
                    # Deserialize model and scaler
                    if model_dict['model_data']:
                        model_dict['model'] = pickle.loads(model_dict['model_data'])
                    if model_dict['scaler_data']:
                        model_dict['scaler'] = pickle.loads(model_dict['scaler_data'])
                    
                    # Parse JSON fields
                    model_dict['features'] = json.loads(model_dict['feature_names'] or '[]')
                    model_dict['importance'] = json.loads(model_dict['feature_importance'] or '{}')
                    
                    # Update last used timestamp
                    cursor.execute('''
                    UPDATE ml_models SET last_used = CURRENT_TIMESTAMP WHERE id = ?
                    ''', (model_dict['id'],))
                    
                    return model_dict
                
                return None
                
        except Exception as e:
            logger.error(f"Error loading ML model for {coin}: {e}")
            return None
    
    
    def get_active_signals(self) -> List[Dict]:
        """Get all active signals with ML enhancement data"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT s.*, tr.pnl_usd, tr.pnl_percentage, tr.max_profit, tr.max_loss, 
                       tr.position_size_usd, tr.leverage, tr.model_confidence_at_entry,
                       tr.feature_stability_score
                FROM signals s
                LEFT JOIN trade_results tr ON s.signal_id = tr.signal_id
                WHERE s.status = 'active'
                ORDER BY s.model_confidence DESC, s.created_at DESC
                ''')
                
                signals = []
                for row in cursor.fetchall():
                    signal_dict = dict(row)
                    
                    # Parse JSON fields safely
                    try:
                        signal_dict['analysis_data'] = json.loads(signal_dict['analysis_data'] or '{}')
                        signal_dict['indicators'] = json.loads(signal_dict['indicators'] or '{}')
                        signal_dict['feature_importance'] = json.loads(signal_dict['feature_importance'] or '{}')
                    except json.JSONDecodeError:
                        signal_dict['analysis_data'] = {}
                        signal_dict['indicators'] = {}
                        signal_dict['feature_importance'] = {}
                    
                    # Ensure numeric fields have safe defaults
                    signal_dict['pnl_usd'] = signal_dict.get('pnl_usd') or 0
                    signal_dict['pnl_percentage'] = signal_dict.get('pnl_percentage') or 0
                    signal_dict['max_profit'] = signal_dict.get('max_profit') or 0
                    signal_dict['max_loss'] = signal_dict.get('max_loss') or 0
                    signal_dict['ml_prediction'] = signal_dict.get('ml_prediction') or 0
                    signal_dict['model_confidence'] = signal_dict.get('model_confidence') or 0
                    
                    signals.append(signal_dict)
                
                return signals
                
        except Exception as e:
            logger.error(f"Error getting active signals: {e}")
            return []
    
    def close_ml_signal(self, signal_id: str, exit_price: float, exit_reason: str) -> bool:
        """Close ML signal with enhanced tracking"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get signal details including ML data
                cursor.execute('''
                SELECT coin, direction, entry_price, created_at, ml_prediction, model_confidence
                FROM signals 
                WHERE signal_id = ? AND status = 'active'
                ''', (signal_id,))
                
                result = cursor.fetchone()
                if not result:
                    logger.warning(f"Signal {signal_id} not found or already closed")
                    return False
                
                coin, direction, entry_price, created_at, ml_prediction, model_confidence = result
                
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
                
                # Calculate ML prediction accuracy
                if ml_prediction is not None:
                    actual_return = ((exit_price - entry_price) / entry_price) if direction == 'LONG' else ((entry_price - exit_price) / entry_price)
                    prediction_error = abs(ml_prediction - actual_return)
                    prediction_accuracy = max(0, 1 - (prediction_error / max(abs(ml_prediction), abs(actual_return), 0.01)))
                else:
                    prediction_accuracy = 0
                
                # Update signal status
                cursor.execute('''
                UPDATE signals 
                SET status = 'closed', exit_price = ?, exit_reason = ?, 
                    closed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE signal_id = ?
                ''', (exit_price, exit_reason, signal_id))
                
                # Update trade result with ML metrics
                cursor.execute('''
                UPDATE trade_results 
                SET exit_price = ?, exit_reason = ?, status = 'closed',
                    pnl_usd = ?, pnl_percentage = ?, duration_minutes = ?,
                    ml_prediction_accuracy = ?, prediction_error = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE signal_id = ?
                ''', (exit_price, exit_reason, pnl_usd, pnl_percentage, 
                      duration_minutes, prediction_accuracy, prediction_error, signal_id))
                
                # Update ML prediction outcome
                cursor.execute('''
                UPDATE ml_predictions 
                SET actual_outcome = ?, prediction_accuracy = ?, outcome_at = CURRENT_TIMESTAMP
                WHERE signal_id = ?
                ''', (pnl_percentage, prediction_accuracy, signal_id))
                
                # Update portfolio statistics with ML metrics
                self._update_ml_portfolio_stats(conn, pnl_usd, prediction_accuracy, model_confidence)
                
                # Log model performance
                self.log_model_performance(coin, prediction_accuracy, model_confidence or 0)
                
                logger.info(f"ML signal closed: {signal_id} - P&L: ${pnl_usd:.2f}, Accuracy: {prediction_accuracy:.3f}")
                return True
                
        except Exception as e:
            logger.error(f"Error closing ML signal {signal_id}: {e}")
            return False
    
    def _update_ml_portfolio_stats(self, conn, pnl_usd: float, prediction_accuracy: float, model_confidence: float):
        """Update portfolio statistics with ML enhancements"""
        try:
            cursor = conn.cursor()
            
            # Get current portfolio stats
            cursor.execute('SELECT * FROM portfolio ORDER BY id DESC LIMIT 1')
            portfolio_row = cursor.fetchone()
            
            if not portfolio_row:
                # Initialize portfolio
                cursor.execute('''
                INSERT INTO portfolio (
                    balance_usd, total_trades, winning_trades, losing_trades, total_pnl, 
                    peak_balance, ml_signals_count, ml_model_accuracy, avg_prediction_confidence
                ) VALUES (1000, 1, ?, 0, ?, 1000, 1, ?, ?)
                ''', (1 if pnl_usd > 0 else 0, pnl_usd, prediction_accuracy, model_confidence))
                return
            
            # Convert sqlite3.Row to dict
            portfolio = dict(portfolio_row)
            
            # Calculate new stats
            balance = portfolio['balance_usd']
            total_trades = portfolio['total_trades']
            winning_trades = portfolio['winning_trades']
            losing_trades = portfolio['losing_trades']
            total_pnl = portfolio['total_pnl']
            max_drawdown = portfolio['max_drawdown']
            peak_balance = portfolio['peak_balance']
            
            # ML-specific stats
            ml_signals_count = portfolio.get('ml_signals_count', 0)
            prev_ml_accuracy = portfolio.get('ml_model_accuracy', 0)
            prev_avg_confidence = portfolio.get('avg_prediction_confidence', 0)
            
            # Update counters
            new_balance = balance + pnl_usd
            new_total_trades = total_trades + 1
            new_winning_trades = winning_trades + (1 if pnl_usd > 0 else 0)
            new_losing_trades = losing_trades + (1 if pnl_usd <= 0 else 0)
            new_total_pnl = total_pnl + pnl_usd
            new_ml_signals_count = ml_signals_count + 1
            
            # Update ML metrics (running averages)
            new_ml_accuracy = ((prev_ml_accuracy * ml_signals_count) + prediction_accuracy) / new_ml_signals_count
            new_avg_confidence = ((prev_avg_confidence * ml_signals_count) + model_confidence) / new_ml_signals_count
            
            # Calculate ML success rate
            ml_success_rate = new_winning_trades / new_total_trades * 100 if new_total_trades > 0 else 0
            
            # Update peak balance and drawdown
            if new_balance > peak_balance:
                new_peak_balance = new_balance
                new_max_drawdown = max_drawdown
            else:
                new_peak_balance = peak_balance
                current_drawdown = ((peak_balance - new_balance) / peak_balance) * 100
                new_max_drawdown = max(max_drawdown, current_drawdown)
            
            # Calculate enhanced ratios
            sharpe_ratio = self._calculate_sharpe_ratio(new_total_pnl, new_total_trades)
            
            # Update portfolio record
            cursor.execute('''
            UPDATE portfolio 
            SET balance_usd = ?, total_trades = ?, winning_trades = ?, 
                losing_trades = ?, total_pnl = ?, max_drawdown = ?, 
                peak_balance = ?, ml_signals_count = ?, ml_model_accuracy = ?,
                avg_prediction_confidence = ?, ml_success_rate = ?, sharpe_ratio = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            ''', (new_balance, new_total_trades, new_winning_trades, 
                  new_losing_trades, new_total_pnl, new_max_drawdown, 
                  new_peak_balance, new_ml_signals_count, new_ml_accuracy,
                  new_avg_confidence, ml_success_rate, sharpe_ratio, portfolio['id']))
            
        except Exception as e:
            logger.error(f"Error updating ML portfolio stats: {e}")
    
    def _calculate_sharpe_ratio(self, total_pnl: float, total_trades: int, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for portfolio"""
        try:
            if total_trades < 2:
                return 0.0
        
            # Get trade returns for volatility calculation
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT pnl_percentage FROM trade_results 
                WHERE status = 'closed' AND pnl_percentage IS NOT NULL
                ORDER BY updated_at DESC 
                LIMIT 50
                ''')
            
                # Fix: Convert each row to dict before accessing
                returns = []
                for row in cursor.fetchall():
                    row_dict = self.row_to_dict(row)
                    if row_dict:
                        returns.append(row_dict['pnl_percentage'] / 100)
            
                if len(returns) < 2:
                    return 0.0
            
                avg_return = np.mean(returns)
                std_return = np.std(returns)
            
                if std_return == 0:
                    return 0.0
            
                # Annualized Sharpe ratio (assuming daily trades)
                sharpe = (avg_return - risk_free_rate / 365) / std_return * np.sqrt(365)
                return min(5.0, max(-5.0, sharpe))  # Cap at reasonable bounds
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def log_model_performance(self, coin: str, accuracy: float, confidence: float):
        """Log model performance for tracking"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO model_performance_history (
                    coin, model_type, accuracy, avg_confidence, prediction_count
                ) VALUES (?, ?, ?, ?, ?)
                ''', (coin, 'ml_ensemble', accuracy, confidence, 1))
                
        except Exception as e:
            logger.error(f"Error logging model performance: {e}")
    
    def get_ml_model_stats(self, coin: str = None) -> Dict:
        """Get ML model statistics"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Base query
                base_query = '''
                SELECT coin, model_type, accuracy, performance_score, feature_count,
                       cv_score, last_used, updated_at
                FROM ml_models 
                '''
                
                if coin:
                    base_query += 'WHERE coin = ? '
                    cursor.execute(base_query + 'ORDER BY performance_score DESC', (coin,))
                else:
                    cursor.execute(base_query + 'ORDER BY performance_score DESC')
                
                models = []
                for row in cursor.fetchall():
                    models.append(dict(row))
                
                # Get overall ML performance
                cursor.execute('''
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(prediction_accuracy) as avg_accuracy,
                    AVG(model_confidence) as avg_confidence
                FROM ml_predictions 
                WHERE prediction_accuracy IS NOT NULL
                ''' + ('AND coin = ?' if coin else ''), (coin,) if coin else ())
                
                overall_stats = cursor.fetchone()
                
                return {
                    'models': models,
                    'overall_stats': dict(overall_stats) if overall_stats else {}
                }
                
        except Exception as e:
            logger.error(f"Error getting ML model stats: {e}")
            return {'models': [], 'overall_stats': {}}
    
    def get_feature_importance_trends(self, coin: str, days: int = 7) -> List[Dict]:
        """Get feature importance trends over time"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT feature_name, importance_score, rank_position, timestamp, model_type
                FROM feature_importance_trends 
                WHERE coin = ? AND timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC, rank_position ASC
                '''.format(days), (coin,))
                
                trends = []
                for row in cursor.fetchall():
                    trends.append(dict(row))
                
                return trends
                
        except Exception as e:
            logger.error(f"Error getting feature importance trends: {e}")
            return []
    
    def save_feature_engineering_log(self, coin: str, feature_stats: Dict):
        """Log feature engineering process"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO feature_engineering_log (
                    coin, raw_features_count, engineered_features_count, 
                    selected_features_count, orthogonalized_features_count,
                    correlation_threshold, selection_method, feature_categories,
                    computation_time_ms, data_quality_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    coin,
                    feature_stats.get('raw_features_count', 0),
                    feature_stats.get('engineered_features_count', 0),
                    feature_stats.get('selected_features_count', 0),
                    feature_stats.get('orthogonalized_features_count', 0),
                    feature_stats.get('correlation_threshold', 0.85),
                    feature_stats.get('selection_method', 'ml_ensemble'),
                    json.dumps(feature_stats.get('feature_categories', {})),
                    feature_stats.get('computation_time_ms', 0),
                    feature_stats.get('data_quality_score', 0.5)
                ))
                
        except Exception as e:
            logger.error(f"Error saving feature engineering log: {e}")
    
    def log_bot_activity(self, level: str, component: str, message: str, 
                        details: str = None, coin: str = None, data: Dict = None):
        """Enhanced bot activity logging with ML context"""
        
        # Skip verbose DEBUG logs except for ML components
        if level == 'DEBUG' and component not in ['ML_TRAINER', 'ML_PREDICTOR', 'FEATURE_ENGINEER']:
            return
        
        # Cache log entry with ML enhancements
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
            
            # Flush cache more frequently for ML components
            current_time = time.time()
            flush_interval = 15 if component.startswith('ML_') else 30
            
            if (current_time - self.last_flush > flush_interval) or len(self.log_cache) > 20:
                self._flush_log_cache()
                self.last_flush = current_time
        
        # Console logging
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
    
    def clean_old_ml_data(self, days: int = 7):
        """Clean old ML data to maintain performance"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Clean old feature importance trends (keep recent trends)
                cursor.execute('''
                DELETE FROM feature_importance_trends 
                WHERE timestamp < datetime('now', '-{} days')
                '''.format(days * 2))  # Keep longer history for trends
                
                # Clean old model performance history (keep recent performance)
                cursor.execute('''
                DELETE FROM model_performance_history 
                WHERE timestamp < datetime('now', '-{} days')
                AND id NOT IN (
                    SELECT id FROM model_performance_history 
                    ORDER BY timestamp DESC 
                    LIMIT 1000
                )
                '''.format(days))
                
                # Clean old feature engineering logs
                cursor.execute('''
                DELETE FROM feature_engineering_log 
                WHERE timestamp < datetime('now', '-{} days')
                '''.format(days))
                
                # Clean old closed ML predictions
                cursor.execute('''
                DELETE FROM ml_predictions 
                WHERE outcome_at < datetime('now', '-{} days')
                AND outcome_at IS NOT NULL
                '''.format(days))
                
                logger.info("ML data cleanup completed")
                
        except Exception as e:
            logger.error(f"Error cleaning old ML data: {e}")
    
    # Compatibility methods for existing interface
    def save_signal(self, signal_data: Dict) -> bool:
        """Compatibility wrapper for save_ml_signal"""
        return self.save_ml_signal(signal_data)
    
    def close_signal(self, signal_id: str, exit_price: float, exit_reason: str) -> bool:
        """Compatibility wrapper for close_ml_signal"""
        return self.close_ml_signal(signal_id, exit_price, exit_reason)
    
    def update_signal_price(self, signal_id: str, current_price: float) -> bool:
        """Update signal price with ML context"""
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
                
                pnl_usd = (pnl_percentage / 100) * 1000
                
                # Update trade result with enhanced tracking
                cursor.execute('''
                UPDATE trade_results 
                SET pnl_usd = ?, pnl_percentage = ?, updated_at = CURRENT_TIMESTAMP,
                    max_profit = CASE WHEN ? > max_profit THEN ? ELSE max_profit END,
                    max_loss = CASE WHEN ? < max_loss THEN ? ELSE max_loss END
                WHERE signal_id = ? AND status = 'open'
                ''', (pnl_usd, pnl_percentage, pnl_usd, pnl_usd, pnl_usd, pnl_usd, signal_id))
                
                return True
                
        except Exception as e:
            logger.error(f"Error updating ML signal price for {signal_id}: {e}")
            return False    
    
    def get_portfolio_stats(self) -> Dict:
        """Get ML-enhanced portfolio statistics"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
            
                # Get latest portfolio record
                cursor.execute('SELECT * FROM portfolio ORDER BY id DESC LIMIT 1')
                portfolio_row = cursor.fetchone()
            
                if not portfolio_row:
                    return self._get_default_ml_portfolio_stats()
            
                # Convert sqlite3.Row to dict properly
                portfolio = dict(portfolio_row)
            
                # Get current open P&L
                cursor.execute('''
                SELECT COALESCE(SUM(pnl_usd), 0) as open_pnl
                FROM trade_results 
                WHERE status = 'open'
                ''')
                open_pnl_row = cursor.fetchone()
                open_pnl = dict(open_pnl_row)['open_pnl'] if open_pnl_row else 0
            
                # Get ML-specific trade statistics
                cursor.execute('''
                SELECT 
                    COALESCE(AVG(pnl_usd), 0) as avg_pnl,
                    COALESCE(MAX(pnl_usd), 0) as best_trade,
                    COALESCE(MIN(pnl_usd), 0) as worst_trade,
                    COALESCE(AVG(CASE WHEN pnl_usd > 0 THEN pnl_usd END), 0) as avg_win,
                    COALESCE(AVG(CASE WHEN pnl_usd <= 0 THEN pnl_usd END), 0) as avg_loss,
                    COALESCE(AVG(duration_minutes), 0) as avg_duration,
                    COALESCE(AVG(ml_prediction_accuracy), 0) as avg_ml_accuracy,
                    COALESCE(AVG(model_confidence_at_entry), 0) as avg_model_confidence
                FROM trade_results 
                WHERE status = 'closed'
                ''')
            
                trade_stats_row = cursor.fetchone()
                trade_stats = self.row_to_dict(trade_stats_row)
            
                total_trades = portfolio.get('total_trades', 0)
                winning_trades = portfolio.get('winning_trades', 0)
            
                return {
                    'total_balance': portfolio.get('balance_usd', 1000) + open_pnl,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': portfolio.get('losing_trades', 0),
                    'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                    'total_pnl': portfolio.get('total_pnl', 0) + open_pnl,
                    'avg_pnl': trade_stats.get('avg_pnl', 0),
                    'best_trade': trade_stats.get('best_trade', 0),
                    'worst_trade': trade_stats.get('worst_trade', 0),
                    'avg_win': trade_stats.get('avg_win', 0),
                    'avg_loss': trade_stats.get('avg_loss', 0),
                    'max_drawdown': portfolio.get('max_drawdown', 0),
                    'open_pnl': open_pnl,
                    'peak_balance': portfolio.get('peak_balance', 1000),
                    'avg_duration_minutes': trade_stats.get('avg_duration', 0),
                
                    # ML-specific metrics
                    'ml_signals_count': portfolio.get('ml_signals_count', 0),
                    'ml_model_accuracy': portfolio.get('ml_model_accuracy', 0),
                    'avg_prediction_confidence': portfolio.get('avg_prediction_confidence', 0),
                    'feature_orthogonality_score': portfolio.get('feature_orthogonality_score', 0),
                    'sharpe_ratio': portfolio.get('sharpe_ratio', 0),
                    'ml_success_rate': portfolio.get('ml_success_rate', 0),
                    'avg_ml_prediction_accuracy': trade_stats.get('avg_ml_accuracy', 0),
                    'avg_model_confidence_at_entry': trade_stats.get('avg_model_confidence', 0)
                }
            
        except Exception as e:
            logger.error(f"Error getting ML portfolio stats: {e}")
            return self._get_default_ml_portfolio_stats()
    
    def _get_default_ml_portfolio_stats(self) -> Dict:
        """Return default ML portfolio stats"""
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
            'avg_duration_minutes': 0,
            'ml_signals_count': 0,
            'ml_model_accuracy': 0,
            'avg_prediction_confidence': 0,
            'feature_orthogonality_score': 0,
            'sharpe_ratio': 0,
            'ml_success_rate': 0,
            'avg_ml_prediction_accuracy': 0,
            'avg_model_confidence_at_entry': 0
        }
    
    def get_recent_logs(self, limit: int = 100) -> List[Dict]:
        """Get recent bot activity logs with ML context"""
        try:
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
    
    def get_signal_by_id(self, signal_id: str) -> Optional[Dict]:
        """Get specific signal with ML enhancement"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT s.*, tr.pnl_usd, tr.pnl_percentage, tr.ml_prediction_accuracy,
                       tr.model_confidence_at_entry, tr.feature_stability_score
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
                        signal_dict['feature_importance'] = json.loads(signal_dict['feature_importance'] or '{}')
                    except:
                        signal_dict['analysis_data'] = {}
                        signal_dict['indicators'] = {}
                        signal_dict['feature_importance'] = {}
                    return signal_dict
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting ML signal {signal_id}: {e}")
            return None
    
    def clean_old_data(self, days: int = 7):
        """Enhanced cleanup including ML data"""
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
                
                # Clean ML data
                self.clean_old_ml_data(days)
                
                logger.info(f"Data cleanup: {deleted_signals} signals, {deleted_logs} logs + ML data")
                
        except Exception as e:
            logger.error(f"Error cleaning old data: {e}")

# Create alias for backward compatibility
Database = MLEnhancedDatabase