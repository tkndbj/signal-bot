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
import numba  # Add this import
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
import shap
explainer = shap.TreeExplainer(model, feature_perturbation='interventional', check_additivity=False)
shap_values = explainer.shap_values(X)

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
        
        # Core parameters
        self.coins = [
            'SOL/USDT','DOGE/USDT','BONK/USDT','FLOKI/USDT','LINK/USDT','PEPE/USDT',
            'NEAR/USDT','TIA/USDT','ARB/USDT','APT/USDT','TAO/USDT','FET/USDT',
            'SUI/USDT','SEI/USDT','OP/USDT','LDO/USDT','SHIB/USDT','BOME/USDT',
            'PENDLE/USDT','JUP/USDT','LINEA/USDT','UB/USDT','ZEC/USDT','CGPT/USDT',
            'POPCAT/USDT','WIF/USDT','OL/USDT','JASMY/USDT','BLUR/USDT','GMX/USDT',
            'COMP/USDT','CRV/USDT','SNX/USDT','1INCH/USDT','SUSHI/USDT','YFI/USDT',
            'BAL/USDT','MKR/USDT'
        ]
        
        # Enhanced trading parameters
        self.min_model_confidence = 0.65
        self.min_feature_importance_sum = 0.5
        self.max_risk_per_trade = 1.0
        self.min_rr_ratio = 2.0
        self.max_signals_per_scan = 7
        
        # ML Model parameters
        self.lookback_periods = 300  # For training
        self.min_data_points = 100
        self.prediction_horizon = 24  # Hours ahead to predict
        self.feature_selection_k = 15  # Top K features to keep
        self.cv_folds = 5  # Time series CV folds
        
        # Feature engineering parameters
        self.orthogonal_threshold = 0.85  # Correlation threshold for orthogonalization
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
        
        logger.info("ML Trading Analyzer initialized with advanced feature engineering")
    
    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initialize exchange with robust error handling"""
        try:
            exchange = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY', ''),
                'secret': os.getenv('BINANCE_SECRET_KEY', ''),
                'sandbox': False,
                'rateLimit': 600,
                'enableRateLimit': True,
                'timeout': 10000,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True
                }
            })
            
            exchange.load_markets()
            logger.info("Exchange connection established successfully")
            return exchange
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            return ccxt.binance({
                'rateLimit': 1000,
                'enableRateLimit': True,
                'timeout': 15000
            })
    
    async def get_market_data(self, symbol: str, timeframe: str = '1h', 
                            limit: int = 500) -> pd.DataFrame:
        """Enhanced market data fetching with extended history"""
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
            ohlcv = await self._fetch_with_retry(symbol, timeframe, limit=limit)  # Pass limit explicitly
            logger.info(f"Fetched {len(ohlcv)} rows for {symbol}")
            if not ohlcv or len(ohlcv) < self.min_data_points:  # Use min_data_points (100) instead of lookback_periods
                logger.warning(f"Insufficient rows for {symbol}: {len(ohlcv)}")
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
        """Fetch data with retry logic"""
        for attempt in range(max_retries):
            try:
                # Use sync version of fetch_ohlcv since ccxt.async is not used
                return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** (attempt + 1))  # Use sync sleep
        return []
    
    def _validate_data_quality(self, df: pd.DataFrame) -> bool:
        if df.empty or len(df) < 100:  # Reduce from 150
            return False
        if df.isnull().sum().sum() > len(df) * 0.10:  # Increase null tolerance to 10%
            return False
        if (df['volume'] == 0).sum() > len(df) * 0.20:  # Increase zero-volume tolerance to 20%
            return False
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            z_scores = np.abs(stats.zscore(df[col].fillna(df[col].median())))
            if (z_scores > 5).sum() > len(df) * 0.02:  # Increase outlier tolerance to 2%
                return False
        if (df['high'] < df['low']).any() or (df['high'] < df['close']).any():
            return False
        return True
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering with multiple domains"""

        initial_rows = len(df)

        if df.empty or len(df) < 50:
            return df        
    
        try:            
            df = df.dropna().copy()  # Only clean initial data
            logger.info(f"Feature engineering starting with {len(df)} clean rows")
        
            # 1. Price-based features
            df = self._create_price_features(df)
        
            # 2. Volume features
            df = self._create_volume_features(df)
        
            # 3. Volatility features
            df = self._create_volatility_features(df)
        
            # 4. Technical indicators (diverse)
            df = self._create_technical_features(df)
        
            # 5. Momentum and trend features
            df = self._create_momentum_features(df)  
        
            # 6. Create target variable (future returns)
            df = self._create_target_variable(df)
        
            # CRITICAL FIX: Instead of dropna(), do selective cleaning
            # Remove rows where target is NaN (essential)
            df = df.dropna(subset=['target_return'])
        
            # For other columns, be more permissive
            # Remove rows only if more than 50% of feature columns are NaN
            feature_cols = [col for col in df.columns if not any(x in col.lower() for x in 
                        ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'target'])]
        
            if feature_cols:
                # Calculate NaN percentage per row for feature columns only
                nan_pct_per_row = df[feature_cols].isna().sum(axis=1) / len(feature_cols)
                # Keep rows where less than 50% of features are NaN
                df = df[nan_pct_per_row < 0.7]
        
            final_rows = len(df)
            logger.info(f"Feature engineering completed: {final_rows} rows ({initial_rows - final_rows} removed)")
        
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
                                           (df['close'].rolling(period).max() - df['close'].rolling(period).min())
        
        # Gap features
        df['gap_up'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['gap_down'] = df['gap_up'] * -1
        
        # Intrabar features
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['open']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['open']
        df['total_range'] = (df['high'] - df['low']) / df['open']
        
        return df
    
    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features"""
        # Volume ratios
        for period in [5, 10, 20]:
            df[f'volume_ratio_{period}'] = df['volume'] / df['volume'].rolling(period).mean()
        
        # Volume-price features
        df['volume_price_trend'] = talib.OBV(df['close'].values, df['volume'].values)
        df['vpt'] = talib.OBV(df['close'].values, df['volume'].values)  # Volume Price Trend
        
        # VWAP variations
        df['vwap_1d'] = self._calculate_vwap(df, period=24)
        df['vwap_3d'] = self._calculate_vwap(df, period=72)
        df['price_vs_vwap_1d'] = (df['close'] - df['vwap_1d']) / df['vwap_1d']
        
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
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['close']
        
        # Realized volatility
        for period in [24, 48, 168]:  # 1d, 2d, 1w
            returns = np.log(df['close'] / df['close'].shift(1))
            df[f'realized_vol_{period}'] = returns.rolling(period).std() * np.sqrt(period)
        
        # Volatility of volatility
        df['vol_of_vol'] = df['realized_vol_24'].rolling(24).std()
        
        # Parkinson volatility (using high-low)
        df['parkinson_vol'] = np.sqrt(0.25 * np.log(df['high']/df['low'])**2)
        
        # Garman-Klass volatility
        df['gk_vol'] = np.sqrt(0.5 * np.log(df['high']/df['low'])**2 - 
                              (2*np.log(2)-1) * np.log(df['close']/df['open'])**2)
        
        return df
    
    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create diverse technical indicators"""
        # Trend indicators
        df['sma_10'] = talib.SMA(df['close'].values, timeperiod=10)
        df['sma_50'] = talib.SMA(df['close'].values, timeperiod=50)
        df['ema_12'] = talib.EMA(df['close'].values, timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'].values, timeperiod=26)
        
        # MACD family
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'].values)
        df['macd_normalized'] = df['macd'] / df['close']
        
        # Oscillators (but we'll orthogonalize these)
        df['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'].values, df['low'].values, 
                                                  df['close'].values)
        df['cci'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'].values)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Trend strength
        df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values)
        df['di_plus'] = talib.PLUS_DI(df['high'].values, df['low'].values, df['close'].values)
        df['di_minus'] = talib.MINUS_DI(df['high'].values, df['low'].values, df['close'].values)
        
        return df
    
    def _create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features"""
        # Order flow proxies
        df['buy_pressure'] = np.where(df['close'] > df['open'], df['volume'], 0)
        df['sell_pressure'] = np.where(df['close'] < df['open'], df['volume'], 0)
        df['net_pressure'] = df['buy_pressure'] - df['sell_pressure']
        
        # Price impact features
        df['price_impact'] = abs(df['close'].pct_change()) / (df['volume'] / df['volume'].rolling(20).mean())
        
        # Tick direction
        df['tick_direction'] = np.sign(df['close'].diff())
        df['tick_runs'] = (df['tick_direction'] != df['tick_direction'].shift()).cumsum()
        
        # Support/Resistance levels
        df['recent_high'] = df['high'].rolling(20).max()
        df['recent_low'] = df['low'].rolling(20).min()
        df['resistance_distance'] = (df['recent_high'] - df['close']) / df['close']
        df['support_distance'] = (df['close'] - df['recent_low']) / df['close']
        
        return df
    
    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum and trend features"""
        # ROC (Rate of Change)
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
        df['ma_convergence'] = (df['sma_10'] - df['sma_50']) / df['sma_50']
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        # Skewness and kurtosis of returns
        for period in [20, 50]:
            returns = df['close'].pct_change()
            df[f'return_skew_{period}'] = returns.rolling(period).skew()
            df[f'return_kurtosis_{period}'] = returns.rolling(period).kurt()
        
        # Autocorrelation
        for lag in [1, 5, 10]:
            df[f'autocorr_lag_{lag}'] = df['close'].pct_change().rolling(50).apply(
                lambda x: x.autocorr(lag=lag), raw=False)
        
        # Hurst exponent (trend persistence)
        df['hurst_50'] = df['close'].rolling(50).apply(self._calculate_hurst, raw=False)
        
        # Entropy (randomness measure)
        df['entropy_20'] = df['close'].pct_change().rolling(20).apply(
            lambda x: stats.entropy(np.histogram(x.dropna(), bins=10)[0] + 1e-10), raw=False)
        
        return df
    
    def _calculate_vwap(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate VWAP over rolling period"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
    
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
        """Create target variable - use shorter horizon to preserve data"""
        initial_rows = len(df)  # ADD THIS LINE
    
        # Use much shorter horizon to preserve training data
        short_horizon = 6  # 6 hours instead of 24
    
        # Simple future return
        df['target_return'] = df['close'].shift(-short_horizon) / df['close'] - 1
    
        # Binary targets  
        df['target_up'] = (df['target_return'] > 0.015).astype(int)  # 1.5% threshold
        df['target_down'] = (df['target_return'] < -0.015).astype(int)
    
        # Remove only the last few rows that have NaN targets
        df = df[:-short_horizon]  # Remove last 6 rows instead of using dropna()

        final_rows = len(df)
        logger.info(f"Target creation: {initial_rows} -> {final_rows} rows ({initial_rows - final_rows} removed)")
    
        return df
    
    def orthogonalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply simplified orthogonalization with better error handling"""
        try:
            # CRITICAL: Check input data first
            if df.empty or len(df) < 50:
                logger.warning(f"Input data too small for orthogonalization: {len(df)} rows")
                return df
            
            logger.info(f"Starting orthogonalization with {len(df)} rows, {len(df.columns)} columns")
        
            # Identify feature columns (be more permissive)
            exclude_terms = ['timestamp', 'target']  # Reduced exclusions
            feature_cols = []
        
            for col in df.columns:
                # Only exclude if column name contains these exact patterns
                if not any(term in col.lower() for term in exclude_terms):
                    # Also exclude basic OHLCV unless they're derived features
                    if col not in ['open', 'high', 'low', 'close', 'volume']:
                        feature_cols.append(col)
        
            logger.info(f"Found {len(feature_cols)} potential feature columns")
        
            if len(feature_cols) < 5:  # Need minimum features
                logger.warning("Too few features for orthogonalization, returning original data")
                return df
        
            # Get feature data with robust cleaning
            feature_data = df[feature_cols].copy()
        
            # More aggressive data cleaning
            initial_rows = len(feature_data)
        
            # Forward fill first, then backward fill, then zero fill
            feature_data = feature_data.ffill().bfill().fillna(0)
        
            # Handle infinite values more carefully
            feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
        
            # For remaining NaNs, use column median
            for col in feature_data.columns:
                if feature_data[col].isna().any():
                    median_val = feature_data[col].median()
                    feature_data[col] = feature_data[col].fillna(median_val if pd.notna(median_val) else 0)
        
            # Check for remaining issues
            if feature_data.isna().any().any():
                logger.warning("Still have NaN values after cleaning, filling with 0")
                feature_data = feature_data.fillna(0)
            
            if feature_data.empty or len(feature_data) < 20:
                logger.warning(f"Data too small after cleaning: {len(feature_data)} rows")
                return df
        
            logger.info(f"Data cleaned: {len(feature_data)} rows, {len(feature_data.columns)} features")
        
            # Skip correlation-based orthogonalization if we don't have enough data
            if len(feature_data) < len(feature_data.columns) * 2:
                logger.warning("Insufficient samples for correlation analysis, using PCA instead")
                return self._apply_simple_pca(df, feature_data, feature_cols)
        
            # Calculate correlation matrix with error handling
            try:
                corr_matrix = feature_data.corr()
            
                # Check if correlation calculation succeeded
                if corr_matrix.isna().all().all() or corr_matrix.empty:
                    logger.warning("Correlation matrix calculation failed, returning original data")
                    return df
                
            except Exception as e:
                logger.warning(f"Correlation calculation failed: {e}, returning original data")
                return df
        
            # Find correlated groups with relaxed threshold
            highly_correlated_groups = self._find_correlated_groups(corr_matrix, 0.9)  # Increased threshold
        
            if not highly_correlated_groups:
                logger.info("No highly correlated groups found, keeping original features")
                return df
        
            logger.info(f"Found {len(highly_correlated_groups)} correlated groups")
        
            # Apply orthogonalization only to groups with sufficient data
            orthogonal_features = feature_data.copy()
            orthogonalized_count = 0
        
            for i, group in enumerate(highly_correlated_groups):
                if len(group) < 2:
                 continue
                
                try:
                    group_data = feature_data[group].values
                
                    # Ensure we have enough samples (at least 2x the number of features)
                    if group_data.shape[0] < group_data.shape[1] * 2:
                        logger.warning(f"Group {i} has insufficient samples ({group_data.shape[0]} vs {group_data.shape[1]}), skipping")
                        continue
                
                    # Check for valid data
                    if np.any(np.isnan(group_data)) or np.any(np.isinf(group_data)):
                        logger.warning(f"Group {i} has invalid data, skipping")
                        continue
                
                    # Apply QR decomposition
                    q, r = qr(group_data, mode='economic')
                
                    if q.shape[1] != len(group):
                        logger.warning(f"QR decomposition dimension mismatch for group {i}, skipping")
                        continue
                
                    # Replace with orthogonal features (keep original names with suffix)
                    for j, feature in enumerate(group[:q.shape[1]]):  # Ensure we don't exceed dimensions
                        new_name = f"{feature}_orth"
                        orthogonal_features[new_name] = q[:, j]
                        # Remove original
                        if feature in orthogonal_features.columns:
                            orthogonal_features = orthogonal_features.drop(feature, axis=1)
                
                    orthogonalized_count += 1
                
                except Exception as e:
                    logger.warning(f"Error orthogonalizing group {i}: {e}, skipping group")
                    continue
        
            # Update dataframe with orthogonal features
            df_result = df.copy()
        
            # Remove old feature columns
            for col in feature_cols:
                if col in df_result.columns:
                    df_result = df_result.drop(col, axis=1)
        
            # Add orthogonal features
            for col in orthogonal_features.columns:
                if col not in ['open', 'high', 'low', 'close', 'volume'] and 'target' not in col.lower():
                    df_result[col] = orthogonal_features[col]
        
            final_rows = len(df_result)
            logger.info(f"Orthogonalized {orthogonalized_count} feature groups. "
                    f"Final data: {final_rows} rows, {len(df_result.columns)} total columns")
        
            # CRITICAL: Ensure we didn't lose all our data
            if final_rows == 0:
                logger.error("Orthogonalization resulted in 0 rows! Returning original data")
                return df
            
            return df_result

        except Exception as e:
            logger.error(f"Critical error in orthogonalization: {e}")
            logger.info("Returning original dataframe due to orthogonalization failure")
            return df

    def _apply_simple_pca(self, df: pd.DataFrame, feature_data: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Apply simple PCA when correlation-based orthogonalization isn't feasible"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        
            # Scale features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(feature_data)
        
            # Apply PCA to reduce to reasonable number of components
            n_components = min(10, len(feature_cols), len(feature_data) - 1)
            pca = PCA(n_components=n_components)
            pca_features = pca.fit_transform(scaled_data)
        
            # Create PCA feature names
            pca_df = pd.DataFrame(pca_features, 
                                columns=[f'pca_{i}' for i in range(n_components)],
                                index=feature_data.index)
        
            # Combine with original dataframe (removing original features)
            df_result = df.copy()
            for col in feature_cols:
                if col in df_result.columns:
                    df_result = df_result.drop(col, axis=1)
        
            # Add PCA features
            for col in pca_df.columns:
                df_result[col] = pca_df[col]
        
            logger.info(f"Applied PCA: reduced {len(feature_cols)} features to {n_components} components")
            return df_result
        
        except Exception as e:
            logger.error(f"PCA fallback failed: {e}, returning original data")
            return df        
    
    def _find_correlated_groups(self, corr_matrix: pd.DataFrame, threshold: float) -> List[List[str]]:
        """Find groups of highly correlated features"""
        groups = []
        processed = set()
        
        for feature in corr_matrix.columns:
            if feature in processed:
                continue
            
            # Find features correlated with current feature
            correlated = corr_matrix[feature][corr_matrix[feature] > threshold].index.tolist()
            correlated = [f for f in correlated if f not in processed]
            
            if len(correlated) > 1:
                groups.append(correlated)
                processed.update(correlated)
        
        return groups
    
    def select_features(self, df: pd.DataFrame, target_col: str = 'target_return') -> Tuple[pd.DataFrame, List[str]]:
        """Advanced feature selection using multiple methods"""
        try:
            # Get feature columns
            feature_cols = [col for col in df.columns if not any(x in col.lower() for x in 
                        ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'target'])]
        
            if len(feature_cols) < self.feature_selection_k:
                return df, feature_cols
        
            # Prepare data - FIX: Replace fillna(method='ffill')
            X = df[feature_cols].ffill().fillna(0)
            y = df[target_col].fillna(0)
        
            # Remove rows with infinite values
            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X = X[mask]
            y = y[mask]
        
            if len(X) < 50:
                return df, feature_cols[:self.feature_selection_k]
        
            # Method 1: Random Forest feature importance
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
            rf.fit(X, y)
            rf_importance = rf.feature_importances_
        
            # Method 2: Mutual information
            mi_scores = mutual_info_regression(X, y, random_state=42)
        
            # Method 3: Statistical correlation
            correlations = abs(X.corrwith(y))
        
            # Combine scores (normalized)
            rf_importance = rf_importance / rf_importance.sum()
            mi_scores = mi_scores / mi_scores.sum() if mi_scores.sum() > 0 else mi_scores
            correlations = correlations / correlations.sum() if correlations.sum() > 0 else correlations
        
            # Weighted combination
            combined_scores = (0.5 * rf_importance + 0.3 * mi_scores + 0.2 * correlations.values)
        
            # Select top features
            top_indices = np.argsort(combined_scores)[-self.feature_selection_k:]
            selected_features = [feature_cols[i] for i in top_indices]
        
            logger.info(f"Selected {len(selected_features)} features from {len(feature_cols)}")
            return df, selected_features
        
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            return df, feature_cols[:self.feature_selection_k]
    
    def train_model(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Train ML model with time series validation"""
        try:
            logger.info(f"Starting model training for {symbol} with {len(df)} rows")
            # Engineer features
            df = self.engineer_features(df)
            
            # Orthogonalize features
            df = self.orthogonalize_features(df)
            
            # Select features
            df, selected_features = self.select_features(df)
            
            # Prepare data
            X = df[selected_features].ffill().fillna(0)
            y = df['target_return'].fillna(0)
            
            # Remove infinite values
            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X = X[mask]
            y = y[mask]
            
            if len(X) < 50:
                logger.warning(f"Insufficient data for {symbol}: {len(X)} rows after processing")
                return {}
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            
            # Initialize models
            models = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
                'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
            
            # Train and validate models
            best_model = None
            best_score = float('inf')
            cv_scores = {}
            
            for name, model in models.items():
                scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Scale features
                    scaler = RobustScaler()
                    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), 
                                                columns=X_train.columns, index=X_train.index)
                    X_val_scaled = pd.DataFrame(scaler.transform(X_val), 
                                              columns=X_val.columns, index=X_val.index)
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Predict and score
                    y_pred = model.predict(X_val_scaled)
                    score = mean_squared_error(y_val, y_pred)
                    scores.append(score)
                
                avg_score = np.mean(scores)
                cv_scores[name] = avg_score
                
                if avg_score < best_score:
                    best_score = avg_score
                    best_model = name
            
            # Train final model on all data
            final_model = models[best_model]
            scaler = RobustScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
            final_model.fit(X_scaled, y)
            
            # Get feature importance
            if hasattr(final_model, 'feature_importances_'):
                feature_importance = dict(zip(selected_features, final_model.feature_importances_))
            else:
                feature_importance = {}
            
            # Calculate SHAP values (for top 10 features only to save time)
            shap_importance = {}
            try:
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                top_feature_names = [f[0] for f in top_features]
                X_sample = X_scaled[top_feature_names].tail(50)  # Reduce sample size from 100 to 50
    
                # Use TreeExplainer with additivity check disabled
                explainer = shap.TreeExplainer(final_model, check_additivity=False)
                shap_values = explainer.shap_values(X_sample)
    
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]  # For multi-output models, take first output
    
                shap_importance = dict(zip(top_feature_names, np.abs(shap_values).mean(axis=0)))
    
            except Exception as e:
                logger.warning(f"SHAP calculation failed for {symbol}: {e}")
            
            # Store model artifacts
            self.models[symbol] = final_model
            self.scalers[symbol] = scaler
            self.feature_selectors[symbol] = selected_features
            self.feature_importance_history[symbol] = {
                'feature_importance': feature_importance,
                'shap_importance': shap_importance,
                'cv_scores': cv_scores,
                'best_model': best_model,
                'best_score': best_score
            }
            
            logger.info(f"Model trained for {symbol} - Best: {best_model}, Score: {best_score:.6f}")
            
            return {
                'model': final_model,
                'scaler': scaler,
                'features': selected_features,
                'importance': feature_importance,
                'shap_importance': shap_importance,
                'cv_score': best_score,
                'model_type': best_model
            }
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return {}
    
    def predict_price_movement(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Make prediction using trained model"""
        try:
            if symbol not in self.models:
                return {}
            
            # Engineer features (same as training)
            df = self.engineer_features(df)
            df = self.orthogonalize_features(df)
            
            # Use selected features
            selected_features = self.feature_selectors[symbol]
            X = df[selected_features].ffill().fillna(0).tail(1)
            
            # Scale features
            scaler = self.scalers[symbol]
            X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
            
            # Make prediction
            model = self.models[symbol]
            prediction = model.predict(X_scaled)[0]
            
            # Calculate confidence (based on model's training performance)
            model_info = self.feature_importance_history[symbol]
            base_confidence = 1.0 / (1.0 + model_info['best_score'])  # Convert MSE to confidence
            
            # Adjust confidence based on feature importance sum
            feature_importance_sum = sum(model_info['feature_importance'].values())
            confidence = base_confidence * min(1.0, feature_importance_sum / self.min_feature_importance_sum)
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'feature_importance': model_info['feature_importance'],
                'shap_importance': model_info['shap_importance'],
                'model_type': model_info['best_model']
            }
            
        except Exception as e:
            logger.error(f"Error making prediction for {symbol}: {e}")
            return {}
    
    async def generate_ml_signal(self, df: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """Generate ML-based trading signal"""
        try:
            if len(df) < self.lookback_periods:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Train model if not exists or retrain periodically
            if symbol not in self.models or len(df) % 100 == 0:  # Retrain every 100 periods
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
            if (model_confidence < self.min_model_confidence or
                abs(prediction) < 0.003):  # Minimum 1% predicted move
                return None
            
            current_price = df['close'].iloc[-1]
            
            # Determine direction
            direction = 'LONG' if prediction > 0 else 'SHORT'
            
            # Calculate position sizing and risk management
            volatility = df['close'].pct_change().std() * np.sqrt(24)  # Daily volatility
            atr_pct = (talib.ATR(df['high'].values, df['low'].values, df['close'].values, 14)[-1] / current_price)
            
            # Dynamic stop loss based on volatility and prediction confidence
            base_stop_pct = max(0.015, atr_pct * 1.5)  # Minimum 1.5% or 1.5x ATR
            confidence_multiplier = 2.0 - model_confidence  # Higher confidence = tighter stops
            stop_pct = base_stop_pct * confidence_multiplier
            
            # Ensure max risk limit
            stop_pct = min(stop_pct, self.max_risk_per_trade / 100)
            
            if direction == 'LONG':
                stop_loss = current_price * (1 - stop_pct)
                # Take profit based on prediction and risk-reward ratio
                min_tp = current_price * (1 + stop_pct * self.min_rr_ratio)
                predicted_tp = current_price * (1 + abs(prediction))
                take_profit = max(min_tp, predicted_tp)
            else:
                stop_loss = current_price * (1 + stop_pct)
                min_tp = current_price * (1 - stop_pct * self.min_rr_ratio)
                predicted_tp = current_price * (1 - abs(prediction))
                take_profit = min(min_tp, predicted_tp)
            
            # Calculate final metrics
            if direction == 'LONG':
                risk = current_price - stop_loss
                reward = take_profit - current_price
            else:
                risk = stop_loss - current_price
                reward = current_price - take_profit
            
            rr_ratio = reward / risk if risk > 0 else 0
            
            if rr_ratio < self.min_rr_ratio:
                return None
            
            # Determine signal strength based on ML metrics
            if model_confidence > 0.9 and abs(prediction) > 0.05:
                strength = SignalStrength.INSTITUTIONAL
            elif model_confidence > 0.8 and abs(prediction) > 0.03:
                strength = SignalStrength.STRONG
            else:
                strength = SignalStrength.MODERATE
            
            # Create confluence factors from top features
            feature_importance = prediction_result['feature_importance']
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            confluence_factors = [f"ML Feature: {feature} (imp: {importance:.3f})" 
                                for feature, importance in top_features]
            
            signal_id = f"{symbol.replace('/', '')}_{int(datetime.now().timestamp())}_ML_{direction[0]}"
            
            return TradingSignal(
                signal_id=signal_id,
                timestamp=datetime.now().isoformat(),
                coin=symbol.replace('/USDT', ''),
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
                    'model_type': prediction_result['model_type'],
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
        """Scan all coins using ML approach"""
        signals = []
        processed_count = 0
        
        # Randomize order
        import random
        coins_to_scan = self.coins.copy()
        random.shuffle(coins_to_scan)
        
        logger.info(f"Starting ML scan of {len(coins_to_scan)} coins")
        
        for symbol in coins_to_scan:
            try:
                if len(signals) >= self.max_signals_per_scan:
                    break
                
                processed_count += 1
                
                # Get extended market data for ML
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
                        'analysis_data': {
                            'ml_prediction': signal.analysis_summary['ml_prediction'],
                            'model_confidence': signal.analysis_summary['model_confidence'],
                            'model_type': signal.analysis_summary['model_type'],
                            'confluence_factors': signal.confluence_factors,
                            'risk_reward_ratio': signal.risk_reward_ratio,
                            'risk_percentage': signal.risk_percentage,
                            'signal_grade': 'institutional' if signal.strength == SignalStrength.INSTITUTIONAL else 'ml_based',
                            'feature_importance_top5': dict(list(signal.feature_importance.items())[:5])
                        },
                        'indicators': {
                            'ml_prediction_pct': round(signal.ml_prediction * 100, 2),
                            'model_confidence': signal.model_confidence,
                            'volatility': signal.analysis_summary['volatility']
                        }
                    }
                    
                    signals.append(signal_dict)
                    logger.info(f"ML Signal: {symbol} {signal.direction} "
                              f"(Pred: {signal.ml_prediction:.4f}, Conf: {signal.model_confidence:.3f})")
                
                # Rate limiting
                await asyncio.sleep(0.8)  # Slightly slower for ML processing
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        logger.info(f"ML scan complete: {len(signals)} signals from {processed_count} coins")
        return signals
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price"""
        try:
            ticker = self.exchange.fetch_ticker(f"{symbol}/USDT")
            return float(ticker.get('last', 0))
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return 0.0
    
    def save_models(self, filepath: str = 'ml_models.joblib'):
        """Save trained models"""
        try:
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
    
    def load_models(self, filepath: str = 'ml_models.joblib'):
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
                logger.warning(f"Model file {filepath} not found")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def __del__(self):
        """Cleanup"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except:
            pass