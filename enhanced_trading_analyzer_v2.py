import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union
import asyncio
import json
from dotenv import load_dotenv
import talib
from dataclasses import dataclass
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# ML and feature engineering imports
import sklearn
import numba
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy
from scipy import stats
from scipy.linalg import qr
import joblib

# Define logger before using it
logger = logging.getLogger(__name__)
logger.info(f"Library versions - NumPy: {np.__version__}, SciPy: {scipy.__version__}, scikit-learn: {sklearn.__version__}, pandas: {pd.__version__}, numba: {numba.__version__}")
warnings.filterwarnings('ignore')

class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

class SignalStrength(Enum):
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    INSTITUTIONAL = 4

@dataclass
class TradingSignal:
    signal_id: str
    timestamp: str
    coin: str
    direction: str
    entry_price: float
    current_price: float
    take_profit: float
    stop_loss: float
    confidence: int
    strength: SignalStrength
    risk_reward_ratio: float
    risk_percentage: float
    analysis_summary: Dict
    confluence_factors: List[str]
    ml_prediction: float
    feature_importance: Dict
    model_confidence: float

class MLTradingAnalyzer:
    def __init__(self, database=None):
        load_dotenv()
        
        # Initialize exchange
        self.exchange = self._initialize_exchange()
        self.database = database
        
        # Core parameters - FIXED VALUES
        self.coins = [
            'WLFIUSDT', 'DOGEUSDT', '1000BONKUSDT', '1000FLOKIUSDT', 'LINKUSDT', '1000PEPEUSDT',
            'NEARUSDT', 'TIAUSDT', 'ARBUSDT', 'APTUSDT', 'TAOUSDT', 'FETUSDT',
            'SUIUSDT', 'SEIUSDT', 'OPUSDT', 'LDOUSDT', 'SHIB1000USDT', 'BOMEUSDT',
            'PENDLEUSDT', 'JUPUSDT', 'LINEAUSDT', 'UBUSDT', 'CGPTUSDT',
            'POPCATUSDT', 'WIFUSDT', 'OLUSDT', 'JASMYUSDT', 'BLURUSDT', 'GMXUSDT',
            'COMPUSDT', 'CRVUSDT', 'TRUMPUSDT', '1INCHUSDT', 'SUSHIUSDT', 'YFIUSDT', 'MOVEUSDT'
        ]
        
        # Enhanced trading parameters
        self.min_model_confidence = 0.65
        self.min_feature_importance_sum = 0.5
        self.max_risk_per_trade = 1.0
        self.min_rr_ratio = 2.0
        self.max_signals_per_scan = 7
        
        # ML Model parameters - OPTIMIZED
        self.lookback_periods = 200  # REDUCED from 300 to avoid data issues
        self.min_data_points = 100
        self.prediction_horizon = 6  # REDUCED from 24 for better data preservation
        self.feature_selection_k = 15  # Top K features to keep
        self.cv_folds = 5  # Time series CV folds
        
        # Feature engineering parameters - OPTIMIZED
        self.orthogonal_threshold = 0.95  # INCREASED from 0.85 to be less aggressive
        self.pca_variance_threshold = 0.95
        
        # Models storage
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.pca_transformers = {}
        self.feature_importance_history = {}
        
        # Rate limiting
        self.request_delays = {}
        self.min_request_interval = 1
        self.cache = {}
        self.cache_duration = 60
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("ML Trading Analyzer initialized with production fixes")
    
    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initialize Bybit exchange for real trading - FIXED"""
        try:
            api_key = os.getenv('BYBIT_API_KEY', '')
            secret_key = os.getenv('BYBIT_SECRET_KEY', '')
        
            if not api_key or not secret_key:
                raise ValueError("Bybit API credentials not found in environment")
        
            exchange = ccxt.bybit({
                'apiKey': api_key,
                'secret': secret_key,
                'enableRateLimit': True,
                'rateLimit': 200,  # INCREASED from 100
                'timeout': 30000,  # ADDED timeout
                'options': {
                    'defaultType': 'linear',  # For USDT perpetuals
                    'adjustForTimeDifference': True,
                    'recvWindow': 10000
                }
            })

            exchange.options['createMarketBuyOrderRequiresPrice'] = False
        
            # Load markets with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    exchange.load_markets()
                    if exchange.markets:
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(2 ** attempt)
        
            if not exchange.markets:
                logger.error("Failed to load markets from Bybit")
                raise ValueError("No markets loaded")
        
            # Cache markets for fast access
            self.markets = exchange.markets
            logger.info(f"Loaded {len(exchange.markets)} markets from Bybit")
        
            # Test balance fetch
            try:
                balance = exchange.fetch_balance({'type': 'unified'})
                if 'USDT' in balance:
                    total = balance['USDT'].get('total', 0)
                    logger.info(f"Bybit connected - USDT balance: {total}")
            except Exception as e:
                logger.warning(f"Balance test: {e}")
        
            logger.info("Bybit exchange connected for REAL TRADING")
            return exchange   
        
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    async def get_market_data(self, symbol: str, timeframe: str = '1h', 
                            limit: int = 500) -> pd.DataFrame:
        """Enhanced market data fetching - FIXED SYMBOL HANDLING"""
        # FIX: Normalize symbol format
        symbol = symbol.replace('/', '')
        if not symbol.endswith('USDT'):
            symbol = symbol + 'USDT'
        
        cache_key = f"{symbol}_{timeframe}_{limit}"
        current_time = time.time()
        
        # Check cache
        if (cache_key in self.cache and 
            current_time - self.cache[cache_key]['timestamp'] < self.cache_duration):
            return self.cache[cache_key]['data'].copy()
        
        try:
            # Rate limiting
            if symbol in self.request_delays:
                time_since_last = current_time - self.request_delays[symbol]
                if time_since_last < self.min_request_interval:
                    await asyncio.sleep(self.min_request_interval - time_since_last)
            
            self.request_delays[symbol] = current_time
            
            # Fetch with retry logic
            ohlcv = await self._fetch_with_retry(symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) < self.min_data_points:
                logger.warning(f"Insufficient data for {symbol}: {len(ohlcv) if ohlcv else 0} rows")
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Data quality validation
            if not self._validate_data_quality(df):
                logger.warning(f"Poor data quality for {symbol}")
                return pd.DataFrame()
            
            # Cache result
            self.cache[cache_key] = {
                'data': df.copy(),
                'timestamp': current_time
            }
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _fetch_with_retry(self, symbol: str, timeframe: str, 
                            limit: int = 500, max_retries: int = 3) -> List:
        """Fetch data with retry logic - FIXED"""
        for attempt in range(max_retries):
            try:
                # Use sync version since we're not using ccxt.async
                return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            except Exception as e:
                logger.warning(f"Fetch attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt == max_retries - 1:
                    return []  # Return empty list instead of raising
                await asyncio.sleep(2 ** (attempt + 1))
        return []
    
    def _validate_data_quality(self, df: pd.DataFrame) -> bool:
        """Validate data quality - RELAXED THRESHOLDS"""
        if df.empty or len(df) < 50:  # REDUCED from 100
            return False
        if df.isnull().sum().sum() > len(df) * 0.15:  # INCREASED tolerance from 0.10
            return False
        if (df['volume'] == 0).sum() > len(df) * 0.25:  # INCREASED from 0.20
            return False
        
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if df[col].std() == 0:  # All same value
                return False
            z_scores = np.abs(stats.zscore(df[col].fillna(df[col].median())))
            if (z_scores > 10).sum() > len(df) * 0.05:  # INCREASED outlier tolerance
                return False
        
        if (df['high'] < df['low']).any() or (df['high'] < df['close']).any():
            return False
        
        return True
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering - FIXED DATA PRESERVATION"""
        if df.empty or len(df) < 50:
            return df
        
        try:
            initial_rows = len(df)
            df = df.copy()  # Work on copy
            
            # 1. Price-based features
            df = self._create_price_features(df)
            
            # 2. Volume features
            df = self._create_volume_features(df)
            
            # 3. Volatility features
            df = self._create_volatility_features(df)
            
            # 4. Technical indicators
            df = self._create_technical_features(df)
            
            # 5. Momentum features
            df = self._create_momentum_features(df)
            
            # 6. Create target variable with SHORT horizon
            df = self._create_target_variable(df)
            
            # CRITICAL: Selective cleaning to preserve data
            # Only remove rows with NaN in target
            df = df.dropna(subset=['target_return'])
            
            # For features, be more permissive
            feature_cols = [col for col in df.columns if not any(x in col.lower() for x in 
                        ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'target'])]
            
            if feature_cols:
                # Only remove rows with >70% NaN features
                nan_pct_per_row = df[feature_cols].isna().sum(axis=1) / len(feature_cols)
                df = df[nan_pct_per_row < 0.7]
            
            final_rows = len(df)
            if final_rows < initial_rows * 0.5:
                logger.warning(f"Too much data lost in feature engineering: {initial_rows} -> {final_rows}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return df
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features"""
        # Returns at different horizons
        for period in [1, 2, 4, 8, 12, 24]:
            df[f'return_{period}h'] = df['close'].pct_change(period)
        
        # Log returns
        df['log_return_1h'] = np.log(df['close'] / df['close'].shift(1))
        df['log_return_4h'] = np.log(df['close'] / df['close'].shift(4))
        
        # Price position within range
        for period in [10, 20, 50]:
            df[f'price_position_{period}'] = (df['close'] - df['close'].rolling(period).min()) / \
                                           (df['close'].rolling(period).max() - df['close'].rolling(period).min() + 1e-10)
        
        # Gap features
        df['gap_up'] = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-10)
        df['gap_down'] = df['gap_up'] * -1
        
        # Intrabar features
        df['body_size'] = abs(df['close'] - df['open']) / (df['open'] + 1e-10)
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['open'] + 1e-10)
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / (df['open'] + 1e-10)
        df['total_range'] = (df['high'] - df['low']) / (df['open'] + 1e-10)
        
        return df
    
    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features"""
        # Volume ratios
        for period in [5, 10, 20]:
            df[f'volume_ratio_{period}'] = df['volume'] / (df['volume'].rolling(period).mean() + 1e-10)
        
        # Volume-price features
        df['volume_price_trend'] = talib.OBV(df['close'].values, df['volume'].values)
        
        # VWAP variations
        df['vwap_1d'] = self._calculate_vwap(df, period=24)
        df['price_vs_vwap_1d'] = (df['close'] - df['vwap_1d']) / (df['vwap_1d'] + 1e-10)
        
        # Volume momentum
        df['volume_momentum_5'] = df['volume'].pct_change(5)
        df['volume_acceleration'] = df['volume_momentum_5'].diff()
        
        # Volume percentiles
        for period in [20, 50]:
            df[f'volume_percentile_{period}'] = df['volume'].rolling(period).rank(pct=True)
        
        return df
    
    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility features"""
        # ATR variations
        for period in [7, 14, 21]:
            df[f'atr_{period}'] = talib.ATR(df['high'].values, df['low'].values, 
                                          df['close'].values, timeperiod=period)
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / (df['close'] + 1e-10)
        
        # Realized volatility
        for period in [24, 48]:  # Removed 168 to reduce NaN
            returns = np.log(df['close'] / df['close'].shift(1))
            df[f'realized_vol_{period}'] = returns.rolling(period).std() * np.sqrt(period)
        
        # Volatility of volatility
        df['vol_of_vol'] = df['realized_vol_24'].rolling(24).std()
        
        return df
    
    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create diverse technical indicators"""
        # Trend indicators
        df['sma_10'] = talib.SMA(df['close'].values, timeperiod=10)
        df['sma_50'] = talib.SMA(df['close'].values, timeperiod=50)
        df['ema_12'] = talib.EMA(df['close'].values, timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'].values, timeperiod=26)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'].values)
        df['macd_normalized'] = df['macd'] / (df['close'] + 1e-10)
        
        # Oscillators
        df['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'].values, df['low'].values, 
                                                  df['close'].values)
        df['cci'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'].values)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
        
        # Trend strength
        df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values)
        
        return df
    
    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum features"""
        # ROC
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = talib.ROC(df['close'].values, timeperiod=period)
        
        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'].values, df['low'].values, 
                                      df['close'].values)
        
        # Momentum
        for period in [10, 20]:
            df[f'momentum_{period}'] = talib.MOM(df['close'].values, timeperiod=period)
        
        # Trend consistency
        for period in [10, 20]:
            df[f'trend_consistency_{period}'] = (df['close'] > df['close'].shift(1)).rolling(period).sum() / period
        
        # Moving average convergence
        df['ma_convergence'] = (df['sma_10'] - df['sma_50']) / (df['sma_50'] + 1e-10)
        
        return df
    
    def _calculate_vwap(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate VWAP over rolling period"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).rolling(period).sum() / (df['volume'].rolling(period).sum() + 1e-10)
    
    def _calculate_hurst(self, price_series) -> float:
        """Calculate Hurst exponent"""
        try:
            if len(price_series) < 10:
                return 0.5
            
            lags = range(2, min(20, len(price_series) // 2))
            tau = [np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 0.5
    
    def _create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable - SHORT HORIZON FOR DATA PRESERVATION"""
        # Use 6 hour horizon instead of 24 to preserve more data
        horizon = self.prediction_horizon  # 6 hours
        
        # Simple future return
        df['target_return'] = df['close'].shift(-horizon) / df['close'] - 1
        
        # Binary targets
        df['target_up'] = (df['target_return'] > 0.015).astype(int)
        df['target_down'] = (df['target_return'] < -0.015).astype(int)
        
        # Only remove the last few rows
        df = df[:-horizon]
        
        return df
    
    def orthogonalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simplified orthogonalization - LESS AGGRESSIVE"""
        try:
            if df.empty or len(df) < 50:
                return df
            
            # Identify feature columns
            exclude_terms = ['timestamp', 'target', 'open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in df.columns if not any(term in col.lower() for term in exclude_terms)]
            
            if len(feature_cols) < 5:
                return df
            
            # Get feature data
            feature_data = df[feature_cols].copy()
            
            # Clean data
            feature_data = feature_data.ffill().bfill().fillna(0)
            feature_data = feature_data.replace([np.inf, -np.inf], 0)
            
            # If insufficient samples, just return original
            if len(feature_data) < len(feature_cols) * 2:
                logger.info("Insufficient samples for orthogonalization, skipping")
                return df
            
            # Calculate correlation matrix
            try:
                corr_matrix = feature_data.corr()
                
                # Find highly correlated groups with HIGH threshold (0.95)
                highly_correlated_groups = self._find_correlated_groups(corr_matrix, self.orthogonal_threshold)
                
                if not highly_correlated_groups:
                    return df
                
                # Only orthogonalize the most correlated features
                for group in highly_correlated_groups[:5]:  # Limit to 5 groups
                    if len(group) < 2:
                        continue
                    
                    try:
                        # Just keep the most important feature from each group
                        # (the one with highest variance)
                        variances = feature_data[group].var()
                        keep_feature = variances.idxmax()
                        drop_features = [f for f in group if f != keep_feature]
                        
                        # Drop the redundant features
                        df = df.drop(columns=drop_features, errors='ignore')
                        
                    except Exception as e:
                        logger.warning(f"Error processing correlated group: {e}")
                        continue
                
            except Exception as e:
                logger.warning(f"Correlation calculation failed: {e}")
                return df
            
            return df
            
        except Exception as e:
            logger.error(f"Error in orthogonalization: {e}")
            return df
    
    def _find_correlated_groups(self, corr_matrix: pd.DataFrame, threshold: float) -> List[List[str]]:
        """Find groups of highly correlated features"""
        groups = []
        processed = set()
        
        for feature in corr_matrix.columns:
            if feature in processed:
                continue
            
            # Find features correlated with current feature
            correlated = corr_matrix[feature][abs(corr_matrix[feature]) > threshold].index.tolist()
            correlated = [f for f in correlated if f not in processed and f != feature]
            
            if correlated:
                group = [feature] + correlated
                groups.append(group)
                processed.update(group)
        
        return groups

    def update_model_from_trades(self, symbol: str, trade_results: List[Dict]) -> bool:
        """Update model based on trade outcomes"""
        try:
            symbol = symbol.replace('/', '')
            if not symbol.endswith('USDT'):
                symbol = symbol + 'USDT'
        
            if not trade_results or symbol not in self.models:
                return False
        
            # Get recent market data for retraining
            df = asyncio.run(self.get_market_data(symbol, '1h', self.lookback_periods))
            if df.empty or len(df) < self.lookback_periods:
                return False
        
            # Engineer features
            df = self.engineer_features(df)
            df = self.orthogonalize_features(df)
        
            # Add trade outcome features
            df = self._add_trade_feedback_features(df, trade_results)
        
            # Retrain with weighted samples
            model_info = self._retrain_with_weights(df, symbol, trade_results)
        
            if model_info:
                logger.info(f"Model updated for {symbol} with trade feedback")
                return True
            return False
        
        except Exception as e:
            logger.error(f"Error updating model from trades: {e}")
            return False

    def _add_trade_feedback_features(self, df: pd.DataFrame, trade_results: List[Dict]) -> pd.DataFrame:
        """Add trade outcome features to training data"""
        try:
            # Initialize feedback columns
            df['trade_success_rate'] = 0.5  # Default neutral
            df['avg_trade_return'] = 0.0
            df['trade_signal_quality'] = 0.5
            
            for trade in trade_results[-20:]:  # Use last 20 trades
                trade_time = pd.to_datetime(trade['created_at'])
                
                # Find closest timestamp in df
                time_diffs = abs(df.index - trade_time)
                if len(time_diffs) > 0:
                    closest_idx = time_diffs.argmin()
                    
                    # Calculate trade success
                    pnl_pct = trade.get('pnl_percentage', 0)
                    trade_success = 1.0 if pnl_pct > 0 else 0.0
                    
                    # Update feedback features in a window around trade
                    window_size = 5
                    start_idx = max(0, closest_idx - window_size)
                    end_idx = min(len(df), closest_idx + window_size)
                    
                    # Weighted update based on proximity
                    for i in range(start_idx, end_idx):
                        weight = 1.0 / (1.0 + abs(i - closest_idx))
                        df.iloc[i, df.columns.get_loc('trade_success_rate')] += trade_success * weight * 0.1
                        df.iloc[i, df.columns.get_loc('avg_trade_return')] += pnl_pct * weight * 0.01
                        
                        # Signal quality based on prediction accuracy
                        if trade.get('ml_prediction'):
                            pred_accuracy = 1.0 if (trade['ml_prediction'] > 0 and pnl_pct > 0) or \
                                                 (trade['ml_prediction'] < 0 and pnl_pct < 0) else 0.0
                            df.iloc[i, df.columns.get_loc('trade_signal_quality')] += pred_accuracy * weight * 0.1
            
            # Clip values to reasonable ranges
            df['trade_success_rate'] = df['trade_success_rate'].clip(0, 1)
            df['trade_signal_quality'] = df['trade_signal_quality'].clip(0, 1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding trade feedback features: {e}")
            return df

    def _retrain_with_weights(self, df: pd.DataFrame, symbol: str, trade_results: List[Dict]) -> Dict:
        """Retrain model with sample weights based on prediction accuracy"""
        try:
            # Select features including new feedback features
            df, selected_features = self.select_features(df)
            
            if not selected_features:
                return {}
            
            # Prepare data
            X = df[selected_features].ffill().bfill().fillna(0)
            y = df['target_return'].fillna(0)
            
            # Calculate sample weights based on trade feedback
            sample_weights = np.ones(len(X))
            
            # Give more weight to periods with successful trades
            if 'trade_success_rate' in df.columns:
                success_rates = df['trade_success_rate'].values
                sample_weights = 0.5 + 0.5 * success_rates  # Range: 0.5 to 1.0
            
            # Clean data
            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X = X[mask]
            y = y[mask]
            sample_weights = sample_weights[mask]
            
            if len(X) < 50:
                return {}
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            weights_train = sample_weights[:split_idx]
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train with sample weights
            model = RandomForestRegressor(
                n_estimators=75,  # Slightly more trees for learning
                max_depth=6,
                min_samples_split=8,
                random_state=42,
                n_jobs=1
            )
            model.fit(X_train_scaled, y_train, sample_weight=weights_train)
            
            # Validate
            y_pred = model.predict(X_test_scaled)
            score = mean_squared_error(y_test, y_pred)
            
            # Update feature importance based on trade success correlation
            feature_importance = dict(zip(selected_features, model.feature_importances_))
            
            # Boost importance of features that correlated with successful trades
            if trade_results:
                feature_importance = self._adjust_feature_importance_by_trades(
                    feature_importance, trade_results, selected_features
                )
            
            # Update stored models
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.feature_selectors[symbol] = selected_features
            self.feature_importance_history[symbol] = {
                'feature_importance': feature_importance,
                'shap_importance': {},
                'best_score': score,
                'best_model': 'rf_adaptive',
                'last_retrain': datetime.now().isoformat(),
                'trade_feedback_count': len(trade_results)
            }
            
            return {
                'model': model,
                'scaler': scaler,
                'features': selected_features,
                'importance': feature_importance,
                'cv_score': score,
                'model_type': 'rf_adaptive'
            }
            
        except Exception as e:
            logger.error(f"Error in retrain with weights: {e}")
            return {}

    def _adjust_feature_importance_by_trades(self, feature_importance: Dict, 
                                            trade_results: List[Dict], 
                                            features: List[str]) -> Dict:
        """Adjust feature importance based on trade success"""
        try:
            # Analyze which features were most important in successful trades
            successful_trades = [t for t in trade_results if t.get('pnl_percentage', 0) > 0]
            failed_trades = [t for t in trade_results if t.get('pnl_percentage', 0) <= 0]
            
            if not successful_trades:
                return feature_importance
            
            # Boost features that were important in successful trades
            boost_factors = {}
            for feature in features:
                boost_factors[feature] = 1.0
                
                # Check correlation with success
                for trade in successful_trades[-10:]:  # Last 10 successful trades
                    trade_features = trade.get('feature_importance', {})
                    if feature in trade_features and trade_features[feature] > 0.1:
                        boost_factors[feature] *= 1.05  # 5% boost
                
                # Penalize features important in failed trades
                for trade in failed_trades[-5:]:  # Last 5 failed trades
                    trade_features = trade.get('feature_importance', {})
                    if feature in trade_features and trade_features[feature] > 0.15:
                        boost_factors[feature] *= 0.95  # 5% penalty
            
            # Apply adjustments
            adjusted_importance = {}
            for feature, importance in feature_importance.items():
                adjusted_importance[feature] = importance * boost_factors.get(feature, 1.0)
            
            # Normalize
            total = sum(adjusted_importance.values())
            if total > 0:
                adjusted_importance = {k: v/total for k, v in adjusted_importance.items()}
            
            return adjusted_importance
            
        except Exception as e:
            logger.error(f"Error adjusting feature importance: {e}")
            return feature_importance

    async def periodic_model_update(self, database) -> None:
        """Periodically update models with trade feedback"""
        try:
            # Get recent closed trades from database
            with database.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT s.*, tr.pnl_percentage, tr.pnl_usd, tr.exit_reason,
                           s.ml_prediction, s.feature_importance
                    FROM signals s
                    JOIN trade_results tr ON s.signal_id = tr.signal_id
                    WHERE s.status = 'closed' 
                    AND s.closed_at > datetime('now', '-7 days')
                    ORDER BY s.closed_at DESC
                ''')
                
                recent_trades = []
                for row in cursor.fetchall():
                    trade_dict = dict(row)
                    # Parse JSON fields
                    try:
                        trade_dict['feature_importance'] = json.loads(
                            trade_dict.get('feature_importance', '{}')
                        )
                    except:
                        trade_dict['feature_importance'] = {}
                    recent_trades.append(trade_dict)
            
            if not recent_trades:
                logger.info("No recent trades for model update")
                return
            
            # Group trades by coin
            trades_by_coin = {}
            for trade in recent_trades:
                coin = trade['coin']
                if coin not in trades_by_coin:
                    trades_by_coin[coin] = []
                trades_by_coin[coin].append(trade)
            
            # Update models for coins with enough trades
            for coin, trades in trades_by_coin.items():
                if len(trades) >= 3:  # Need at least 3 trades
                    logger.info(f"Updating model for {coin} with {len(trades)} trade results")
                    self.update_model_from_trades(coin, trades)
                    await asyncio.sleep(2)  # Rate limiting
            
            # Save updated models
            self.save_models()
            logger.info("Periodic model update completed")
            
        except Exception as e:
            logger.error(f"Error in periodic model update: {e}")
    
    def select_features(self, df: pd.DataFrame, target_col: str = 'target_return') -> Tuple[pd.DataFrame, List[str]]:
        """Feature selection - SIMPLIFIED"""
        try:
            # Get feature columns
            feature_cols = [col for col in df.columns if not any(x in col.lower() for x in 
                        ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'target'])]
            
            if len(feature_cols) <= self.feature_selection_k:
                return df, feature_cols
            
            # Prepare data
            X = df[feature_cols].ffill().bfill().fillna(0)
            y = df[target_col].fillna(0)
            
            # Remove infinite values
            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X = X[mask]
            y = y[mask]
            
            if len(X) < 50:
                return df, feature_cols[:self.feature_selection_k]
            
            # Use simple Random Forest importance
            rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=1)
            rf.fit(X, y)
            
            # Get top features
            importances = rf.feature_importances_
            top_indices = np.argsort(importances)[-self.feature_selection_k:]
            selected_features = [feature_cols[i] for i in top_indices]
            
            return df, selected_features
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            return df, feature_cols[:self.feature_selection_k]
    
    def train_model(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Train ML model - SIMPLIFIED AND ROBUST"""
        try:
            # Engineer features
            df = self.engineer_features(df)
            
            if df.empty or len(df) < 50:
                logger.warning(f"Insufficient data after feature engineering for {symbol}")
                return {}
            
            # Orthogonalize features
            df = self.orthogonalize_features(df)
            
            # Select features
            df, selected_features = self.select_features(df)
            
            if not selected_features:
                logger.warning(f"No features selected for {symbol}")
                return {}
            
            # Prepare data
            X = df[selected_features].ffill().bfill().fillna(0)
            y = df['target_return'].fillna(0)
            
            # Remove infinite values
            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X = X[mask]
            y = y[mask]
            
            if len(X) < 50:
                logger.warning(f"Insufficient clean data for {symbol}")
                return {}
            
            # Simple train/test split (last 20% for validation)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train simple model
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                min_samples_split=10,
                random_state=42,
                n_jobs=1
            )
            model.fit(X_train_scaled, y_train)
            
            # Validate
            y_pred = model.predict(X_test_scaled)
            score = mean_squared_error(y_test, y_pred)
            
            # Get feature importance
            feature_importance = dict(zip(selected_features, model.feature_importances_))
            
            # Store model
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.feature_selectors[symbol] = selected_features
            self.feature_importance_history[symbol] = {
                'feature_importance': feature_importance,
                'shap_importance': {},
                'best_score': score,
                'best_model': 'rf'
            }
            
            logger.info(f"Model trained for {symbol} - Score: {score:.6f}")
            
            return {
                'model': model,
                'scaler': scaler,
                'features': selected_features,
                'importance': feature_importance,
                'cv_score': score,
                'model_type': 'rf'
            }
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return {}
    
    def predict_price_movement(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Make prediction using trained model"""
        try:
            # Normalize symbol
            symbol = symbol.replace('/', '')
            if not symbol.endswith('USDT'):
                symbol = symbol + 'USDT'
            
            if symbol not in self.models:
                return {}
            
            # Engineer features
            df = self.engineer_features(df)
            if df.empty:
                return {}
            
            df = self.orthogonalize_features(df)
            
            # Use selected features
            selected_features = self.feature_selectors[symbol]
            
            # Check if we have the required features
            missing_features = [f for f in selected_features if f not in df.columns]
            if missing_features:
                logger.warning(f"Missing features for prediction: {missing_features}")
                return {}
            
            X = df[selected_features].ffill().bfill().fillna(0).tail(1)
            
            if X.empty:
                return {}
            
            # Scale features
            scaler = self.scalers[symbol]
            X_scaled = scaler.transform(X)
            
            # Make prediction
            model = self.models[symbol]
            prediction = model.predict(X_scaled)[0]
            
            # Calculate confidence
            model_info = self.feature_importance_history[symbol]
            base_confidence = 1.0 / (1.0 + model_info['best_score'])
            
            # Adjust confidence based on feature importance
            feature_importance_sum = sum(model_info['feature_importance'].values())
            confidence = base_confidence * min(1.0, feature_importance_sum / self.min_feature_importance_sum)
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'feature_importance': model_info['feature_importance'],
                'shap_importance': {},
                'model_type': 'rf'
            }
            
        except Exception as e:
            logger.error(f"Error making prediction for {symbol}: {e}")
            return {}
    
    def detect_market_regime(self, df: pd.DataFrame) -> Dict:
        """Detect current market regime"""
        try:
            if len(df) < 50:
                return {'regime': 'unknown', 'confidence': 0}
            
            current_price = df['close'].iloc[-1]
            
            # ATR for volatility
            atr_14 = talib.ATR(df['high'].values, df['low'].values, df['close'].values, 14)[-1]
            atr_pct = atr_14 / current_price
            
            # Trend indicators
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, 14)[-1]
            
            # RSI for momentum
            rsi = talib.RSI(df['close'].values, 14)[-1]
            
            # Classify regime
            regime = 'normal'
            confidence = 0.5
            
            if atr_pct > 0.05:
                regime = 'high_volatility'
                confidence = 0.8
            elif adx > 35:
                if current_price > sma_20 > sma_50:
                    regime = 'trending_up'
                    confidence = 0.75
                elif current_price < sma_20 < sma_50:
                    regime = 'trending_down'
                    confidence = 0.75
            elif adx < 20:
                regime = 'ranging'
                confidence = 0.6
            
            return {
                'regime': regime,
                'confidence': confidence,
                'indicators': {
                    'atr_pct': atr_pct,
                    'rsi': rsi,
                    'adx': adx
                }
            }
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return {'regime': 'unknown', 'confidence': 0}
    
    def evaluate_position_health(self, signal_data: Dict, df: pd.DataFrame) -> Dict:
        """Evaluate the health of an active position"""
        try:
            # Normalize symbol for prediction
            symbol = signal_data['coin']
            if not symbol.endswith('USDT'):
                symbol = symbol + 'USDT'
            
            current_price = df['close'].iloc[-1]
            entry_price = signal_data['entry_price']
            direction = signal_data['direction']
            
            # Calculate P&L
            if direction == 'LONG':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            # Get fresh ML prediction if model exists
            ml_prediction = 0
            ml_confidence = 0
            
            if symbol in self.models and len(df) >= 100:
                prediction_result = self.predict_price_movement(df, symbol)
                if prediction_result:
                    ml_prediction = prediction_result.get('prediction', 0)
                    ml_confidence = prediction_result.get('confidence', 0)
            
            # Check if prediction aligns with position
            original_ml_pred = signal_data.get('ml_prediction', 0)
            prediction_aligned = (ml_prediction * original_ml_pred) > 0 if original_ml_pred != 0 else True
            
            # Detect market regime
            regime_info = self.detect_market_regime(df)
            
            # Calculate health score
            health_score = 1.0
            
            if not prediction_aligned and abs(ml_prediction) > 0.01:
                health_score *= 0.5
            
            if pnl_pct < -2:
                health_score *= 0.7
            elif pnl_pct > 2:
                health_score = min(1.0, health_score * 1.2)
            
            # Determine recommendation
            if health_score < 0.3:
                recommendation = 'exit_immediately'
            elif health_score < 0.5:
                recommendation = 'exit_soon'
            elif pnl_pct > 1.5:
                recommendation = 'move_sl_breakeven'
            elif pnl_pct > 5:
                recommendation = 'trail_stop'
            else:
                recommendation = 'hold'
            
            return {
                'health_score': health_score,
                'pnl_pct': pnl_pct,
                'ml_prediction': ml_prediction,
                'ml_confidence': ml_confidence,
                'prediction_aligned': prediction_aligned,
                'regime': regime_info['regime'],
                'current_price': current_price,
                'recommendation': recommendation
            }
            
        except Exception as e:
            logger.error(f"Error evaluating position health: {e}")
            return {'health_score': 0.5, 'recommendation': 'hold'}
    
    def calculate_dynamic_levels(self, signal_data: Dict, df: pd.DataFrame, 
                                  evaluation: Dict) -> Dict:
        """Calculate new dynamic stop loss and take profit levels"""
        try:
            current_price = evaluation['current_price']
            direction = signal_data['direction']
            recommendation = evaluation['recommendation']
            
            current_sl = signal_data['stop_loss']
            current_tp = signal_data['take_profit']
            entry_price = signal_data['entry_price']
            
            # Calculate ATR for adjustments
            atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, 14)[-1]
            
            if recommendation == 'exit_immediately':
                return {'action': 'exit', 'reason': 'Low health score'}
            
            elif recommendation == 'move_sl_breakeven':
                if direction == 'LONG':
                    new_sl = entry_price * 1.002
                    new_sl = max(new_sl, current_sl)
                else:
                    new_sl = entry_price * 0.998
                    new_sl = min(new_sl, current_sl)
                return {'action': 'update', 'new_sl': new_sl, 'new_tp': current_tp,
                        'reason': 'Moving to breakeven'}
            
            elif recommendation == 'trail_stop':
                if direction == 'LONG':
                    new_sl = current_price - (atr * 2)
                    new_sl = max(new_sl, current_sl)
                else:
                    new_sl = current_price + (atr * 2)
                    new_sl = min(new_sl, current_sl)
                return {'action': 'update', 'new_sl': new_sl, 'new_tp': current_tp,
                        'reason': 'Trailing stop'}
            
            return {'action': 'hold', 'reason': 'No adjustment needed'}
            
        except Exception as e:
            logger.error(f"Error calculating dynamic levels: {e}")
            return {'action': 'hold', 'reason': 'Error in calculation'}
    
    async def generate_ml_signal(self, df: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """Generate ML-based trading signal - FIXED"""
        try:
            # Normalize symbol
            symbol = symbol.replace('/', '')
            if not symbol.endswith('USDT'):
                symbol = symbol + 'USDT'
            
            if len(df) < self.lookback_periods:
                return None
            
            # Train model if not exists
            if symbol not in self.models:
                model_info = self.train_model(df, symbol)
                if not model_info:
                    return None
            
            # Make prediction
            prediction_result = self.predict_price_movement(df, symbol)
            if not prediction_result:
                return None
            
            prediction = prediction_result['prediction']
            model_confidence = prediction_result['confidence']
            
            # Quality gates
            if model_confidence < self.min_model_confidence or abs(prediction) < 0.003:
                return None
            
            current_price = df['close'].iloc[-1]
            
            # Get symbol info
            try:
                if symbol in self.markets:
                    symbol_info = self.markets[symbol]
                    tick_size = symbol_info.get('precision', {}).get('price', 0.00001)
                    min_price = symbol_info.get('limits', {}).get('price', {}).get('min', 0.0000001)
                else:
                    tick_size = 0.00001
                    min_price = 0.0000001
            except:
                tick_size = 0.00001
                min_price = 0.0000001
            
            if current_price <= 0:
                return None
            
            # Determine direction
            direction = 'LONG' if prediction > 0 else 'SHORT'
            
            # Calculate risk parameters
            volatility = df['close'].pct_change().std() * np.sqrt(24)
            atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, 14)[-1]
            atr_pct = atr / current_price
            
            # Dynamic stop loss
            base_stop_pct = max(0.015, atr_pct * 1.5)
            stop_pct = min(base_stop_pct * (2.0 - model_confidence), 0.03)
            
            # Minimum price differences
            if current_price > 100:
                min_price_diff = current_price * 0.005
            elif current_price > 1:
                min_price_diff = current_price * 0.007
            else:
                min_price_diff = current_price * 0.01
            
            min_price_diff = max(min_price_diff, tick_size * 10)
            
            # Calculate levels
            if direction == 'LONG':
                stop_loss = current_price * (1 - stop_pct)
                predicted_tp = current_price * (1 + abs(prediction))
                min_tp = current_price + min_price_diff * 2
                take_profit = max(min_tp, predicted_tp)
            else:
                stop_loss = current_price * (1 + stop_pct)
                predicted_tp = current_price * (1 - abs(prediction))
                min_tp = current_price - min_price_diff * 2
                take_profit = min(min_tp, predicted_tp)
            
            # Apply precision
            try:
                current_price = float(self.exchange.price_to_precision(symbol, current_price))
                stop_loss = float(self.exchange.price_to_precision(symbol, stop_loss))
                take_profit = float(self.exchange.price_to_precision(symbol, take_profit))
            except:
                pass
            
            # Validate prices
            if direction == 'LONG':
                if not (stop_loss < current_price < take_profit):
                    return None
            else:
                if not (take_profit < current_price < stop_loss):
                    return None
            
            # Calculate R:R ratio
            if direction == 'LONG':
                risk = current_price - stop_loss
                reward = take_profit - current_price
            else:
                risk = stop_loss - current_price
                reward = current_price - take_profit
            
            rr_ratio = reward / risk if risk > 0 else 0
            
            if rr_ratio < self.min_rr_ratio:
                return None
            
            # Determine signal strength
            if model_confidence > 0.85:
                strength = SignalStrength.STRONG
            elif model_confidence > 0.75:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            # Create signal
            signal_id = f"{symbol}_{int(datetime.now().timestamp())}_ML_{direction[0]}"
            
            feature_importance = prediction_result['feature_importance']
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            confluence_factors = [f"{feature}: {importance:.3f}" for feature, importance in top_features]
            
            return TradingSignal(
                signal_id=signal_id,
                timestamp=datetime.now().isoformat(),
                coin=symbol.replace('USDT', ''),
                direction=direction,
                entry_price=round(current_price, 6),
                current_price=round(current_price, 6),
                take_profit=round(take_profit, 6),
                stop_loss=round(stop_loss, 6),
                confidence=min(95, int(model_confidence * 100)),
                strength=strength,
                risk_reward_ratio=round(rr_ratio, 2),
                risk_percentage=round(stop_pct * 100, 2),
                analysis_summary={
                    'ml_prediction': round(prediction, 4),
                    'model_confidence': round(model_confidence, 3),
                    'model_type': 'rf',
                    'volatility': round(volatility, 4),
                    'atr_pct': round(atr_pct, 4)
                },
                confluence_factors=confluence_factors,
                ml_prediction=prediction,
                feature_importance=feature_importance,
                model_confidence=model_confidence
            )
            
        except Exception as e:
            logger.error(f"Error generating ML signal for {symbol}: {e}")
            return None
    
    async def scan_all_coins(self) -> List[Dict]:
        """Scan all coins - OPTIMIZED BATCHING"""
        signals = []
        processed_count = 0
        
        # Randomize order
        import random
        coins_to_scan = self.coins.copy()
        random.shuffle(coins_to_scan)
        
        # REDUCED batch size for memory efficiency
        BATCH_SIZE = 5  # REDUCED from 8
        
        logger.info(f"Starting ML scan of {len(coins_to_scan)} coins in batches of {BATCH_SIZE}")
        
        # Process coins in batches
        for batch_idx in range(0, len(coins_to_scan), BATCH_SIZE):
            batch = coins_to_scan[batch_idx:batch_idx + BATCH_SIZE]
            batch_signals = []
            
            for symbol in batch:
                try:
                    if len(signals) >= self.max_signals_per_scan:
                        break
                    
                    processed_count += 1
                    
                    # Get market data
                    df = await self.get_market_data(symbol, '1h', self.lookback_periods)
                    if df.empty or len(df) < self.lookback_periods:
                        continue
                    
                    # Generate ML signal
                    signal = await self.generate_ml_signal(df, symbol)
                    
                    if signal:
                        # Convert to dict
                        signal_dict = {
                            'signal_id': signal.signal_id,
                            'timestamp': signal.timestamp,
                            'coin': signal.coin,
                            'direction': signal.direction,
                            'entry_price': signal.entry_price,
                            'current_price': signal.current_price,
                            'take_profit': signal.take_profit,
                            'stop_loss': signal.stop_loss,
                            'confidence': signal.confidence,
                            'ml_prediction': signal.ml_prediction,
                            'model_confidence': signal.model_confidence,
                            'analysis_data': {
                                'ml_prediction': signal.analysis_summary['ml_prediction'],
                                'model_confidence': signal.analysis_summary['model_confidence'],
                                'model_type': 'rf',
                                'risk_reward_ratio': signal.risk_reward_ratio,
                                'risk_percentage': signal.risk_percentage,
                                'feature_importance_top5': dict(list(signal.feature_importance.items())[:5])
                            },
                            'indicators': {
                                'ml_prediction_pct': round(signal.ml_prediction * 100, 2),
                                'model_confidence': signal.model_confidence,
                                'volatility': signal.analysis_summary['volatility']
                            }
                        }
                        
                        batch_signals.append(signal_dict)
                        signals.append(signal_dict)
                        logger.info(f"ML Signal: {symbol} {signal.direction} (Conf: {signal.model_confidence:.3f})")
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}")
                    continue
            
            # Clear memory after each batch
            if batch_idx + BATCH_SIZE < len(coins_to_scan):
                self._clear_batch_memory()
            
            # Stop if we have enough signals
            if len(signals) >= self.max_signals_per_scan:
                break
        
        logger.info(f"ML scan complete: {len(signals)} signals from {processed_count} coins")
        return signals
    
    def _clear_batch_memory(self):
        """Clear memory between batches"""
        try:
            self.cache.clear()
            import gc
            gc.collect()
        except Exception as e:
            logger.warning(f"Memory cleanup warning: {e}")
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price - FIXED"""
        try:
            # Fix double USDT issue
            symbol = symbol.replace('/', '')
            if symbol.endswith('USDTUSDT'):
                symbol = symbol.replace('USDTUSDT', 'USDT')
            elif not symbol.endswith('USDT'):
                symbol = symbol + 'USDT'
            
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker.get('last', 0))
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return 0.0
    
    def save_models(self, filepath: str = 'models/ml_models.joblib'):
        """Save trained models"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_selectors': self.feature_selectors,
                'feature_importance_history': self.feature_importance_history
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Models saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str = 'models/ml_models.joblib'):
        """Load trained models"""
        try:
            if os.path.exists(filepath):
                model_data = joblib.load(filepath)
                self.models = model_data.get('models', {})
                self.scalers = model_data.get('scalers', {})
                self.feature_selectors = model_data.get('feature_selectors', {})
                self.feature_importance_history = model_data.get('feature_importance_history', {})
                logger.info(f"Loaded {len(self.models)} models from {filepath}")
            else:
                logger.info(f"No saved models found at {filepath}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def __del__(self):
        """Cleanup"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except:
            pass