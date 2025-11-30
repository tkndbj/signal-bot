"""
Multi-Strategy Trading Engine
Combines: Trend Following + Funding Rate Harvesting + ML Filtering
"""

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

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class StrategyType(Enum):
    TREND_FOLLOWING = "trend_following"
    FUNDING_RATE = "funding_rate"
    ML_PREDICTION = "ml_prediction"


@dataclass
class TradingSignal:
    signal_id: str
    timestamp: str
    coin: str
    direction: str
    entry_price: float
    take_profit: float
    stop_loss: float
    confidence: float
    strategy_type: str
    risk_reward_ratio: float
    analysis_data: Dict
    ml_filter_passed: bool = True
    model_confidence: float = 0.0


class StrategyEngine:
    """
    Production trading engine with multiple strategies:
    1. Trend Following (primary) - Trade with the trend on pullbacks
    2. Funding Rate Harvesting - Collect funding when rates are extreme
    3. ML Filter - Use ML to filter trades, not generate them
    """
    
    def __init__(self, database=None):
        load_dotenv()
        
        self.exchange = self._initialize_exchange()
        self.database = database
        
        # ============== COIN UNIVERSE ==============
        # Focus on liquid, established coins for trend following
        self.trend_coins = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
            'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT',
            'ATOMUSDT', 'NEARUSDT', 'APTUSDT', 'ARBUSDT', 'OPUSDT',
            'SUIUSDT', 'INJUSDT', 'FETUSDT', 'TIAUSDT', 'SEIUSDT'
        ]
        
        # Higher volatility coins for funding rate plays
        self.funding_coins = [
            'DOGEUSDT', '1000PEPEUSDT', '1000BONKUSDT', 'WIFUSDT',
            '1000FLOKIUSDT', 'SHIBUSDT', 'TRUMPUSDT', 'MEMEUSDT'
        ]
        
        # ============== STRATEGY PARAMETERS ==============
        
        # Trend Following Parameters
        self.trend_params = {
            'fast_ma': 21,           # Fast moving average
            'slow_ma': 50,           # Slow moving average
            'trend_ma': 200,         # Long-term trend MA
            'rsi_period': 14,
            'rsi_oversold': 40,      # Buy zone in uptrend
            'rsi_overbought': 60,    # Sell zone in downtrend
            'adx_threshold': 25,     # Minimum trend strength
            'atr_period': 14,
            'pullback_pct': 0.02,    # 2% pullback from recent high/low
            'min_rr_ratio': 2.5,     # Minimum risk:reward
        }
        
        # Funding Rate Parameters
        self.funding_params = {
            'min_funding_rate': 0.0003,   # 0.03% per 8h = 0.09%/day minimum
            'extreme_funding_rate': 0.001, # 0.1% per 8h = very extreme
            'max_position_hours': 48,      # Max hours to hold funding position
        }
        
        # ML Filter Parameters - Used to FILTER, not generate signals
        self.ml_params = {
            'min_filter_confidence': 0.55,  # Just above random
            'feature_count': 12,
            'lookback_periods': 100,
        }
        
        # General parameters
        self.max_signals_per_scan = 3  # Quality over quantity
        self.min_confidence = 0.70
        
        # ML models storage (for filtering)
        self.ml_models = {}
        self.ml_scalers = {}
        self.ml_features = {}
        
        # Rate limiting
        self.request_delays = {}
        self.min_request_interval = 0.5
        self.cache = {}
        self.cache_duration = 30
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("=" * 50)
        logger.info("STRATEGY ENGINE INITIALIZED")
        logger.info(f"Trend Following Coins: {len(self.trend_coins)}")
        logger.info(f"Funding Rate Coins: {len(self.funding_coins)}")
        logger.info("Strategies: Trend Following + Funding Rate + ML Filter")
        logger.info("=" * 50)
    
    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initialize Bybit exchange"""
        try:
            api_key = os.getenv('BYBIT_API_KEY', '')
            secret_key = os.getenv('BYBIT_SECRET_KEY', '')
            
            if not api_key or not secret_key:
                raise ValueError("Bybit API credentials not found")
            
            exchange = ccxt.bybit({
                'apiKey': api_key,
                'secret': secret_key,
                'enableRateLimit': True,
                'rateLimit': 200,
                'timeout': 30000,
                'options': {
                    'defaultType': 'linear',
                    'adjustForTimeDifference': True,
                    'recvWindow': 10000
                }
            })
            
            exchange.options['createMarketBuyOrderRequiresPrice'] = False
            
            # Load markets
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
            
            self.markets = exchange.markets
            logger.info(f"Bybit connected - {len(exchange.markets)} markets loaded")
            
            # Test connection
            try:
                balance = exchange.fetch_balance({'type': 'unified'})
                if 'USDT' in balance:
                    logger.info(f"USDT Balance: {balance['USDT'].get('total', 0)}")
            except Exception as e:
                logger.warning(f"Balance check: {e}")
            
            return exchange
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    # ============== DATA FETCHING ==============
    
    async def get_market_data(self, symbol: str, timeframe: str = '1h', 
                              limit: int = 300) -> pd.DataFrame:
        """Fetch and cache market data"""
        symbol = self._normalize_symbol(symbol)
        cache_key = f"{symbol}_{timeframe}_{limit}"
        current_time = time.time()
        
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
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) < 50:
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Data quality check
            if not self._validate_data(df):
                return pd.DataFrame()
            
            self.cache[cache_key] = {'data': df.copy(), 'timestamp': current_time}
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format"""
        symbol = symbol.replace('/', '').replace(':', '')
        if symbol.endswith('USDTUSDT'):
            symbol = symbol.replace('USDTUSDT', 'USDT')
        if not symbol.endswith('USDT'):
            symbol = symbol + 'USDT'
        return symbol
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality"""
        if df.empty or len(df) < 50:
            return False
        if df.isnull().sum().sum() > len(df) * 0.1:
            return False
        if (df['volume'] == 0).sum() > len(df) * 0.2:
            return False
        if (df['high'] < df['low']).any():
            return False
        return True
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price"""
        try:
            symbol = self._normalize_symbol(symbol)
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker.get('last', 0))
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return 0.0
    
    # ============== TREND FOLLOWING STRATEGY ==============
    
    def calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all trend following indicators"""
        df = df.copy()
        
        # Moving averages
        df['ema_fast'] = talib.EMA(df['close'].values, timeperiod=self.trend_params['fast_ma'])
        df['ema_slow'] = talib.EMA(df['close'].values, timeperiod=self.trend_params['slow_ma'])
        df['sma_trend'] = talib.SMA(df['close'].values, timeperiod=self.trend_params['trend_ma'])
        
        # RSI
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=self.trend_params['rsi_period'])
        
        # ADX for trend strength
        df['adx'] = talib.ADX(df['high'].values, df['low'].values, 
                             df['close'].values, timeperiod=14)
        df['plus_di'] = talib.PLUS_DI(df['high'].values, df['low'].values, 
                                      df['close'].values, timeperiod=14)
        df['minus_di'] = talib.MINUS_DI(df['high'].values, df['low'].values, 
                                        df['close'].values, timeperiod=14)
        
        # ATR for stop loss
        df['atr'] = talib.ATR(df['high'].values, df['low'].values, 
                             df['close'].values, timeperiod=self.trend_params['atr_period'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'].values)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'].values, timeperiod=20)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # Recent high/low for pullback detection
        df['recent_high'] = df['high'].rolling(20).max()
        df['recent_low'] = df['low'].rolling(20).min()
        
        # Pullback percentage
        df['pullback_from_high'] = (df['recent_high'] - df['close']) / df['recent_high']
        df['pullback_from_low'] = (df['close'] - df['recent_low']) / df['recent_low']
        
        return df
    
    def detect_trend(self, df: pd.DataFrame) -> Dict:
        """Detect current market trend with confidence"""
        if len(df) < 200:
            return {'trend': 'unknown', 'strength': 0, 'confidence': 0}
        
        current = df.iloc[-1]
        
        # Price vs MAs
        above_fast = current['close'] > current['ema_fast']
        above_slow = current['close'] > current['ema_slow']
        above_trend = current['close'] > current['sma_trend']
        
        fast_above_slow = current['ema_fast'] > current['ema_slow']
        slow_above_trend = current['ema_slow'] > current['sma_trend']
        
        # ADX strength
        adx = current['adx']
        plus_di = current['plus_di']
        minus_di = current['minus_di']
        
        # Determine trend
        if above_fast and above_slow and above_trend and fast_above_slow:
            if adx > self.trend_params['adx_threshold'] and plus_di > minus_di:
                trend = 'strong_uptrend'
                strength = min(100, adx * 2)
                confidence = 0.85
            else:
                trend = 'uptrend'
                strength = adx
                confidence = 0.70
        elif not above_fast and not above_slow and not above_trend and not fast_above_slow:
            if adx > self.trend_params['adx_threshold'] and minus_di > plus_di:
                trend = 'strong_downtrend'
                strength = min(100, adx * 2)
                confidence = 0.85
            else:
                trend = 'downtrend'
                strength = adx
                confidence = 0.70
        else:
            trend = 'ranging'
            strength = 100 - adx
            confidence = 0.5
        
        return {
            'trend': trend,
            'strength': strength,
            'confidence': confidence,
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }
    
    async def generate_trend_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate trend following signal"""
        try:
            # Get multi-timeframe data
            df_1h = await self.get_market_data(symbol, '1h', 300)
            df_4h = await self.get_market_data(symbol, '4h', 100)
            
            if df_1h.empty or df_4h.empty:
                return None
            
            # Calculate indicators
            df_1h = self.calculate_trend_indicators(df_1h)
            df_4h = self.calculate_trend_indicators(df_4h)
            
            # Detect trends
            trend_1h = self.detect_trend(df_1h)
            trend_4h = self.detect_trend(df_4h)
            
            current = df_1h.iloc[-1]
            current_price = current['close']
            atr = current['atr']
            
            signal = None
            
            # ===== LONG SETUP =====
            # 4H in uptrend + 1H pullback
            if trend_4h['trend'] in ['uptrend', 'strong_uptrend']:
                if trend_1h['trend'] in ['ranging', 'uptrend']:
                    # Look for pullback entry
                    rsi = current['rsi']
                    pullback = current['pullback_from_high']
                    macd_bullish = current['macd'] > current['macd_signal']
                    
                    # Entry conditions:
                    # 1. RSI pulled back (not overbought)
                    # 2. Price pulled back 1-4% from recent high
                    # 3. MACD turning bullish
                    # 4. Price above slow MA
                    
                    if (rsi < self.trend_params['rsi_overbought'] and
                        0.01 < pullback < 0.04 and
                        current['close'] > current['ema_slow'] and
                        macd_bullish):
                        
                        # Calculate levels
                        stop_loss = current_price - (atr * 2.5)
                        
                        # TP based on ATR and recent swing
                        target_distance = atr * 5
                        take_profit = current_price + target_distance
                        
                        # Validate R:R
                        risk = current_price - stop_loss
                        reward = take_profit - current_price
                        rr_ratio = reward / risk if risk > 0 else 0
                        
                        if rr_ratio >= self.trend_params['min_rr_ratio']:
                            confidence = (trend_4h['confidence'] + trend_1h['confidence']) / 2
                            
                            signal = TradingSignal(
                                signal_id=f"{symbol}_{int(time.time())}_TF_L",
                                timestamp=datetime.now().isoformat(),
                                coin=symbol.replace('USDT', ''),
                                direction='LONG',
                                entry_price=round(current_price, 6),
                                take_profit=round(take_profit, 6),
                                stop_loss=round(stop_loss, 6),
                                confidence=confidence,
                                strategy_type='trend_following',
                                risk_reward_ratio=round(rr_ratio, 2),
                                analysis_data={
                                    'trend_4h': trend_4h['trend'],
                                    'trend_1h': trend_1h['trend'],
                                    'rsi': round(rsi, 2),
                                    'adx': round(current['adx'], 2),
                                    'pullback_pct': round(pullback * 100, 2),
                                    'atr': round(atr, 6)
                                }
                            )
            
            # ===== SHORT SETUP =====
            # 4H in downtrend + 1H rally
            elif trend_4h['trend'] in ['downtrend', 'strong_downtrend']:
                if trend_1h['trend'] in ['ranging', 'downtrend']:
                    rsi = current['rsi']
                    rally = current['pullback_from_low']
                    macd_bearish = current['macd'] < current['macd_signal']
                    
                    if (rsi > self.trend_params['rsi_oversold'] and
                        0.01 < rally < 0.04 and
                        current['close'] < current['ema_slow'] and
                        macd_bearish):
                        
                        stop_loss = current_price + (atr * 2.5)
                        target_distance = atr * 5
                        take_profit = current_price - target_distance
                        
                        risk = stop_loss - current_price
                        reward = current_price - take_profit
                        rr_ratio = reward / risk if risk > 0 else 0
                        
                        if rr_ratio >= self.trend_params['min_rr_ratio']:
                            confidence = (trend_4h['confidence'] + trend_1h['confidence']) / 2
                            
                            signal = TradingSignal(
                                signal_id=f"{symbol}_{int(time.time())}_TF_S",
                                timestamp=datetime.now().isoformat(),
                                coin=symbol.replace('USDT', ''),
                                direction='SHORT',
                                entry_price=round(current_price, 6),
                                take_profit=round(take_profit, 6),
                                stop_loss=round(stop_loss, 6),
                                confidence=confidence,
                                strategy_type='trend_following',
                                risk_reward_ratio=round(rr_ratio, 2),
                                analysis_data={
                                    'trend_4h': trend_4h['trend'],
                                    'trend_1h': trend_1h['trend'],
                                    'rsi': round(rsi, 2),
                                    'adx': round(current['adx'], 2),
                                    'rally_pct': round(rally * 100, 2),
                                    'atr': round(atr, 6)
                                }
                            )
            
            # Apply ML filter if signal generated
            if signal and symbol in self.ml_models:
                ml_result = self._apply_ml_filter(df_1h, symbol, signal.direction)
                signal.ml_filter_passed = ml_result['passed']
                signal.model_confidence = ml_result['confidence']
                
                if not ml_result['passed']:
                    logger.info(f"Signal filtered by ML: {symbol} {signal.direction}")
                    return None
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating trend signal for {symbol}: {e}")
            return None
    
    # ============== FUNDING RATE STRATEGY ==============
    
    async def scan_funding_opportunities(self) -> List[Dict]:
        """Scan for funding rate opportunities"""
        opportunities = []
        
        try:
            for symbol in self.funding_coins:
                try:
                    symbol = self._normalize_symbol(symbol)
                    
                    # Get funding rate
                    funding_info = self.exchange.fetch_funding_rate(symbol)
                    funding_rate = funding_info.get('fundingRate', 0) or 0
                    
                    if abs(funding_rate) >= self.funding_params['min_funding_rate']:
                        daily_yield = abs(funding_rate) * 3 * 100  # 3 funding periods per day
                        
                        opportunities.append({
                            'symbol': symbol,
                            'funding_rate': funding_rate,
                            'daily_yield': round(daily_yield, 4),
                            'direction': 'SHORT' if funding_rate > 0 else 'LONG',
                            'next_funding': funding_info.get('fundingDatetime'),
                            'is_extreme': abs(funding_rate) >= self.funding_params['extreme_funding_rate']
                        })
                    
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    logger.debug(f"Funding rate error for {symbol}: {e}")
                    continue
            
            # Sort by yield
            opportunities.sort(key=lambda x: x['daily_yield'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error scanning funding rates: {e}")
        
        return opportunities[:5]
    
    async def get_funding_rate_signals(self) -> List[Dict]:
        """Generate signals based on extreme funding rates"""
        signals = []
        
        try:
            opportunities = await self.scan_funding_opportunities()
            
            for opp in opportunities:
                if not opp['is_extreme']:
                    continue
                
                symbol = opp['symbol']
                df = await self.get_market_data(symbol, '1h', 100)
                
                if df.empty:
                    continue
                
                current_price = df['close'].iloc[-1]
                atr = talib.ATR(df['high'].values, df['low'].values, 
                               df['close'].values, 14)[-1]
                
                # Wider stops for funding plays (we want to collect funding, not trade direction)
                if opp['direction'] == 'LONG':
                    stop_loss = current_price - (atr * 4)
                    take_profit = current_price + (atr * 3)
                else:
                    stop_loss = current_price + (atr * 4)
                    take_profit = current_price - (atr * 3)
                
                signal_dict = {
                    'signal_id': f"{symbol}_{int(time.time())}_FR_{opp['direction'][0]}",
                    'timestamp': datetime.now().isoformat(),
                    'coin': symbol.replace('USDT', ''),
                    'direction': opp['direction'],
                    'entry_price': round(current_price, 6),
                    'take_profit': round(take_profit, 6),
                    'stop_loss': round(stop_loss, 6),
                    'confidence': 0.75 if opp['is_extreme'] else 0.65,
                    'model_confidence': 0.75,
                    'strategy_type': 'funding_rate',
                    'risk_reward_ratio': 0.75,  # Lower RR acceptable for funding
                    'analysis_data': {
                        'funding_rate': opp['funding_rate'],
                        'daily_yield': opp['daily_yield'],
                        'is_extreme': opp['is_extreme']
                    },
                    'indicators': {
                        'funding_rate_pct': round(opp['funding_rate'] * 100, 4),
                        'daily_yield_pct': opp['daily_yield']
                    }
                }
                
                signals.append(signal_dict)
            
        except Exception as e:
            logger.error(f"Error generating funding rate signals: {e}")
        
        return signals[:1]  # Max 1 funding rate position
    
    # ============== ML FILTER (NOT PREDICTOR) ==============
    
    def _create_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for ML filter - focus on regime, not price prediction"""
        df = df.copy()
        
        # Volatility features (regime detection)
        df['volatility_10'] = df['close'].pct_change().rolling(10).std()
        df['volatility_30'] = df['close'].pct_change().rolling(30).std()
        df['volatility_ratio'] = df['volatility_10'] / (df['volatility_30'] + 1e-10)
        
        # Trend strength
        df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, 14)
        
        # Momentum
        df['rsi'] = talib.RSI(df['close'].values, 14)
        df['rsi_slope'] = df['rsi'].diff(5)
        
        # Volume
        df['volume_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
        
        # Price position
        sma_50 = talib.SMA(df['close'].values, 50)
        df['price_vs_sma50'] = (df['close'] - sma_50) / (sma_50 + 1e-10)
        
        # Bollinger position
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'].values, 20)
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        
        # Recent returns
        df['return_5'] = df['close'].pct_change(5)
        df['return_10'] = df['close'].pct_change(10)
        
        # Trend consistency
        df['trend_consistency'] = (df['close'] > df['close'].shift(1)).rolling(10).mean()
        
        return df
    
    def train_ml_filter(self, df: pd.DataFrame, symbol: str) -> bool:
        """Train ML filter model - predicts if setup is likely to work"""
        try:
            df = self._create_ml_features(df)
            
            # Create target: Did price move favorably in next 6 bars?
            df['future_return'] = df['close'].shift(-6) / df['close'] - 1
            df['favorable_move'] = (abs(df['future_return']) > 0.01).astype(int)  # 1% move
            
            # Clean data
            feature_cols = ['volatility_ratio', 'adx', 'rsi', 'rsi_slope', 
                          'volume_ratio', 'price_vs_sma50', 'bb_position',
                          'return_5', 'return_10', 'trend_consistency']
            
            df = df.dropna(subset=feature_cols + ['favorable_move'])
            
            if len(df) < 100:
                return False
            
            X = df[feature_cols].values
            y = df['favorable_move'].values
            
            # Time series split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train classifier
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=4,
                min_samples_split=20,
                random_state=42,
                n_jobs=1
            )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            accuracy = model.score(X_test_scaled, y_test)
            
            if accuracy > 0.52:  # Better than random
                self.ml_models[symbol] = model
                self.ml_scalers[symbol] = scaler
                self.ml_features[symbol] = feature_cols
                logger.info(f"ML filter trained for {symbol} - Accuracy: {accuracy:.3f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error training ML filter for {symbol}: {e}")
            return False
    
    def _apply_ml_filter(self, df: pd.DataFrame, symbol: str, direction: str) -> Dict:
        """Apply ML filter to a potential signal"""
        try:
            if symbol not in self.ml_models:
                return {'passed': True, 'confidence': 0.5}
            
            df = self._create_ml_features(df)
            feature_cols = self.ml_features[symbol]
            
            # Get latest features
            latest = df[feature_cols].iloc[-1:].values
            
            if np.isnan(latest).any():
                return {'passed': True, 'confidence': 0.5}
            
            # Scale and predict
            scaler = self.ml_scalers[symbol]
            latest_scaled = scaler.transform(latest)
            
            model = self.ml_models[symbol]
            proba = model.predict_proba(latest_scaled)[0]
            
            # Probability of favorable move
            confidence = proba[1] if len(proba) > 1 else 0.5
            
            passed = confidence >= self.ml_params['min_filter_confidence']
            
            return {'passed': passed, 'confidence': confidence}
            
        except Exception as e:
            logger.error(f"Error applying ML filter: {e}")
            return {'passed': True, 'confidence': 0.5}
    
    # ============== POSITION MANAGEMENT ==============
    
    def evaluate_position_health(self, signal_data: Dict, df: pd.DataFrame) -> Dict:
        """Evaluate health of an active position"""
        try:
            current_price = df['close'].iloc[-1]
            entry_price = signal_data['entry_price']
            direction = signal_data['direction']
            
            # Calculate P&L
            if direction == 'LONG':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            # Get indicators
            df = self.calculate_trend_indicators(df)
            trend_info = self.detect_trend(df)
            
            # Health score
            health_score = 0.5
            recommendation = 'hold'
            
            # Trend alignment check
            if direction == 'LONG':
                if trend_info['trend'] in ['strong_uptrend', 'uptrend']:
                    health_score += 0.2
                elif trend_info['trend'] in ['strong_downtrend', 'downtrend']:
                    health_score -= 0.3
            else:
                if trend_info['trend'] in ['strong_downtrend', 'downtrend']:
                    health_score += 0.2
                elif trend_info['trend'] in ['strong_uptrend', 'uptrend']:
                    health_score -= 0.3
            
            # P&L adjustment
            if pnl_pct > 3:
                health_score += 0.2
                recommendation = 'trail_stop'
            elif pnl_pct > 1.5:
                recommendation = 'move_sl_breakeven'
            elif pnl_pct < -2:
                health_score -= 0.2
            
            # RSI extreme check
            rsi = df['rsi'].iloc[-1]
            if direction == 'LONG' and rsi > 75:
                health_score -= 0.1
            elif direction == 'SHORT' and rsi < 25:
                health_score -= 0.1
            
            # Final recommendation
            if health_score < 0.2:
                recommendation = 'exit_soon'
            elif health_score < 0.1:
                recommendation = 'exit_immediately'
            
            return {
                'health_score': max(0, min(1, health_score)),
                'pnl_pct': pnl_pct,
                'trend': trend_info['trend'],
                'current_price': current_price,
                'recommendation': recommendation,
                'rsi': rsi
            }
            
        except Exception as e:
            logger.error(f"Error evaluating position: {e}")
            return {'health_score': 0.5, 'recommendation': 'hold'}
    
    def calculate_dynamic_levels(self, signal_data: Dict, df: pd.DataFrame, 
                                  evaluation: Dict) -> Dict:
        """Calculate dynamic SL/TP adjustments"""
        try:
            current_price = evaluation['current_price']
            direction = signal_data['direction']
            recommendation = evaluation['recommendation']
            entry_price = signal_data['entry_price']
            current_sl = signal_data['stop_loss']
            current_tp = signal_data['take_profit']
            
            atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, 14)[-1]
            
            if recommendation == 'exit_immediately':
                return {'action': 'exit', 'reason': 'Health score critical'}
            
            elif recommendation == 'move_sl_breakeven':
                if direction == 'LONG':
                    new_sl = entry_price * 1.002  # Tiny profit
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
            return {'action': 'hold', 'reason': 'Calculation error'}
    
    # ============== MAIN SIGNAL GENERATION ==============
    
    async def generate_signals(self) -> List[Dict]:
        """Generate signals from all strategies"""
        signals = []
        
        try:
            logger.info("Scanning for trend following setups...")
            
            # Randomize order to avoid always checking same coins first
            import random
            coins = self.trend_coins.copy()
            random.shuffle(coins)
            
            for symbol in coins:
                if len(signals) >= self.max_signals_per_scan:
                    break
                
                try:
                    signal = await self.generate_trend_signal(symbol)
                    
                    if signal and signal.confidence >= self.min_confidence:
                        if signal.ml_filter_passed:
                            signal_dict = {
                                'signal_id': signal.signal_id,
                                'timestamp': signal.timestamp,
                                'coin': signal.coin,
                                'direction': signal.direction,
                                'entry_price': signal.entry_price,
                                'take_profit': signal.take_profit,
                                'stop_loss': signal.stop_loss,
                                'confidence': int(signal.confidence * 100),
                                'model_confidence': signal.model_confidence,
                                'strategy_type': signal.strategy_type,
                                'risk_reward_ratio': signal.risk_reward_ratio,
                                'ml_filter_passed': signal.ml_filter_passed,
                                'analysis_data': signal.analysis_data,
                                'indicators': signal.analysis_data
                            }
                            signals.append(signal_dict)
                            logger.info(f"Signal: {symbol} {signal.direction} (Conf: {signal.confidence:.2f}, RR: {signal.risk_reward_ratio})")
                    
                    await asyncio.sleep(0.3)
                    
                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}")
                    continue
            
            logger.info(f"Scan complete: {len(signals)} signals generated")
            
        except Exception as e:
            logger.error(f"Error in signal generation: {e}")
        
        return signals

    async def scan_all_coins(self):
        """Alias for generate_signals - maintains compatibility with app.py"""
        return await self.generate_signals()
    
    async def periodic_model_update(self, database=None) -> None:
        """Periodically update ML filter models"""
        try:
            logger.info("Updating ML filter models...")
            
            for symbol in self.trend_coins[:10]:  # Top 10 coins
                try:
                    df = await self.get_market_data(symbol, '1h', 500)
                    if not df.empty and len(df) >= 200:
                        self.train_ml_filter(df, symbol)
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.debug(f"ML update error for {symbol}: {e}")
            
            self.save_models()
            logger.info("ML filter models updated")
            
        except Exception as e:
            logger.error(f"Error in periodic model update: {e}")
    
    # ============== MODEL PERSISTENCE ==============
    
    def save_models(self, filepath: str = 'models/strategy_models.joblib'):
        """Save trained models"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            model_data = {
                'ml_models': self.ml_models,
                'ml_scalers': self.ml_scalers,
                'ml_features': self.ml_features
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Models saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str = 'models/strategy_models.joblib'):
        """Load trained models"""
        try:
            if os.path.exists(filepath):
                model_data = joblib.load(filepath)
                self.ml_models = model_data.get('ml_models', {})
                self.ml_scalers = model_data.get('ml_scalers', {})
                self.ml_features = model_data.get('ml_features', {})
                logger.info(f"Loaded {len(self.ml_models)} ML filter models")
            else:
                logger.info("No saved models found - will train on first run")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def __del__(self):
        """Cleanup"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except:
            pass