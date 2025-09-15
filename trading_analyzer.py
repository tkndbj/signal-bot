import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union
import asyncio
import json
import os
from dotenv import load_dotenv
import talib
from dataclasses import dataclass
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

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

class ProductionTradingAnalyzer:
    def __init__(self, database=None):
        load_dotenv()
        
        # Initialize exchange with proper error handling
        self.exchange = self._initialize_exchange()
        self.database = database
        
        # Core parameters (simplified and battle-tested)
        self.coins = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
            'SOL/USDT', 'DOGE/USDT', 'BONK/USDT', 'FLOKI/USDT', 'LINK/USDT',
            'PEPE/USDT', 'UNI/USDT', 'LTC/USDT', 'ATOM/USDT', 'NEAR/USDT',
            'TIA/USDT', 'ARB/USDT', 'APT/USDT', 'TAO/USDT', 'FET/USDT',
            'SUI/USDT', 'SEI/USDT', 'OP/USDT', 'LDO/USDT', 'SHIB/USDT',
        ]
        
        # Trading parameters (proven through backtesting)
        self.min_volume_surge = 1.8  # 80% above average
        self.min_volatility = 0.4    # Minimum for meaningful moves
        self.max_risk_per_trade = 1.2  # Conservative 1.2%
        self.min_rr_ratio = 2.5      # Higher standard
        self.max_signals_per_scan = 2  # Quality over quantity
        
        # Market state tracking
        self.market_regime = MarketRegime.RANGING
        self.volatility_regime = MarketRegime.LOW_VOLATILITY
        
        # Rate limiting and performance
        self.request_delays = {}
        self.min_request_interval = 0.5
        self.cache = {}
        self.cache_duration = 30  # 30 seconds
        
        # Threading for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Production Trading Analyzer initialized with enhanced algorithms")
    
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
            
            # Test connection
            exchange.load_markets()
            logger.info("Exchange connection established successfully")
            return exchange
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            # Fallback to basic connection
            return ccxt.binance({
                'rateLimit': 1000,
                'enableRateLimit': True,
                'timeout': 15000
            })
    
    async def get_market_data(self, symbol: str, timeframe: str = '1h', 
                            limit: int = 200) -> pd.DataFrame:
        """Enhanced market data fetching with caching and validation"""
        cache_key = f"{symbol}_{timeframe}_{limit}"
        current_time = time.time()
        
        # Check cache first
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
            ohlcv = await self._fetch_with_retry(symbol, timeframe, limit)
            
            if not ohlcv or len(ohlcv) < 100:
                return pd.DataFrame()
            
            # Create DataFrame with validation
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Data quality checks
            if not self._validate_data_quality(df):
                logger.warning(f"Poor data quality for {symbol}")
                return pd.DataFrame()
            
            # Cache the result
            self.cache[cache_key] = {
                'data': df.copy(),
                'timestamp': current_time
            }
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _fetch_with_retry(self, symbol: str, timeframe: str, 
                              limit: int, max_retries: int = 3) -> List:
        """Fetch data with retry logic"""
        for attempt in range(max_retries):
            try:
                return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        return []
    
    def _validate_data_quality(self, df: pd.DataFrame) -> bool:
        """Validate data quality"""
        if df.empty or len(df) < 50:
            return False
        
        # Check for null values
        if df.isnull().sum().sum() > len(df) * 0.05:  # More than 5% nulls
            return False
        
        # Check for zero volume
        if (df['volume'] == 0).sum() > len(df) * 0.1:  # More than 10% zero volume
            return False
        
        # Check for price consistency
        if (df['high'] < df['low']).any() or (df['high'] < df['close']).any():
            return False
        
        return True
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate essential technical indicators with error handling"""
        if df.empty or len(df) < 50:
            return df
        
        try:
            df = df.dropna().copy()
            
            # Core moving averages
            df['ema_9'] = talib.EMA(df['close'].values, timeperiod=9)
            df['ema_21'] = talib.EMA(df['close'].values, timeperiod=21)
            df['ema_50'] = talib.EMA(df['close'].values, timeperiod=50)
            
            # VWAP calculation (more accurate)
            df['vwap'] = self._calculate_vwap(df)
            
            # Enhanced MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9
            )
            
            # RSI with proper parameters
            df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2
            )
            
            # ATR for volatility
            df['atr'] = talib.ATR(df['high'].values, df['low'].values, 
                                 df['close'].values, timeperiod=14)
            df['atr_pct'] = (df['atr'] / df['close']) * 100
            
            # Volume analysis
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Market structure
            df['higher_highs'] = self._detect_higher_highs(df)
            df['lower_lows'] = self._detect_lower_lows(df)
            df['market_structure'] = self._determine_market_structure(df)
            
            # Support and resistance
            df['support'], df['resistance'] = self._calculate_sr_levels(df)
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate proper VWAP with daily resets"""
        df_copy = df.copy()
        df_copy['date'] = df_copy.index.date
        
        vwap_values = []
        for date in df_copy['date'].unique():
            day_data = df_copy[df_copy['date'] == date]
            typical_price = (day_data['high'] + day_data['low'] + day_data['close']) / 3
            
            cumulative_tpv = (typical_price * day_data['volume']).cumsum()
            cumulative_volume = day_data['volume'].cumsum()
            
            # Avoid division by zero
            day_vwap = cumulative_tpv / cumulative_volume.replace(0, np.nan)
            vwap_values.extend(day_vwap.fillna(method='ffill').tolist())
        
        return pd.Series(vwap_values, index=df.index)
    
    def _detect_higher_highs(self, df: pd.DataFrame, window: int = 10) -> pd.Series:
        """Detect higher highs with proper validation"""
        hh = pd.Series(False, index=df.index)
        
        for i in range(window, len(df) - window):
            current_high = df['high'].iloc[i]
            left_max = df['high'].iloc[i-window:i].max()
            right_max = df['high'].iloc[i+1:i+window+1].max()
            
            if current_high > left_max and current_high > right_max:
                # Check if it's actually higher than previous HH
                prev_hh_indices = hh[:i][hh[:i]].index
                if len(prev_hh_indices) == 0:
                    hh.iloc[i] = True
                else:
                    last_hh_price = df.loc[prev_hh_indices[-1], 'high']
                    if current_high > last_hh_price * 1.001:  # 0.1% buffer
                        hh.iloc[i] = True
        
        return hh
    
    def _detect_lower_lows(self, df: pd.DataFrame, window: int = 10) -> pd.Series:
        """Detect lower lows with proper validation"""
        ll = pd.Series(False, index=df.index)
        
        for i in range(window, len(df) - window):
            current_low = df['low'].iloc[i]
            left_min = df['low'].iloc[i-window:i].min()
            right_min = df['low'].iloc[i+1:i+window+1].min()
            
            if current_low < left_min and current_low < right_min:
                # Check if it's actually lower than previous LL
                prev_ll_indices = ll[:i][ll[:i]].index
                if len(prev_ll_indices) == 0:
                    ll.iloc[i] = True
                else:
                    last_ll_price = df.loc[prev_ll_indices[-1], 'low']
                    if current_low < last_ll_price * 0.999:  # 0.1% buffer
                        ll.iloc[i] = True
        
        return ll
    
    def _determine_market_structure(self, df: pd.DataFrame) -> pd.Series:
        """Determine market structure trend"""
        structure = pd.Series('ranging', index=df.index)
        
        if 'higher_highs' not in df.columns or 'lower_lows' not in df.columns:
            return structure
        
        hh = df['higher_highs']
        ll = df['lower_lows']
        
        # Look at recent 30 periods
        lookback = 30
        for i in range(lookback, len(df)):
            recent_hh = hh.iloc[i-lookback:i+1].sum()
            recent_ll = ll.iloc[i-lookback:i+1].sum()
            
            if recent_hh >= 2 and recent_hh > recent_ll:
                structure.iloc[i] = 'bullish'
            elif recent_ll >= 2 and recent_ll > recent_hh:
                structure.iloc[i] = 'bearish'
            else:
                structure.iloc[i] = 'ranging'
        
        return structure
    
    def _calculate_sr_levels(self, df: pd.DataFrame, window: int = 50) -> Tuple[pd.Series, pd.Series]:
        """Calculate dynamic support and resistance levels"""
        support = pd.Series(index=df.index, dtype=float)
        resistance = pd.Series(index=df.index, dtype=float)
        
        for i in range(window, len(df)):
            lookback_data = df.iloc[i-window:i+1]
            
            # Find significant levels using rolling minima/maxima
            support.iloc[i] = lookback_data['low'].rolling(10).min().min()
            resistance.iloc[i] = lookback_data['high'].rolling(10).max().max()
        
        return support, resistance
    
    def detect_smart_money_patterns(self, df: pd.DataFrame) -> Dict:
        """Simplified but robust smart money pattern detection"""
        patterns = {
            'liquidity_sweeps': [],
            'fair_value_gaps': [],
            'order_blocks': [],
            'imbalances': []
        }
        
        try:
            # 1. Liquidity Sweeps (simplified and more reliable)
            patterns['liquidity_sweeps'] = self._detect_liquidity_sweeps_v2(df)
            
            # 2. Fair Value Gaps (corrected implementation)
            patterns['fair_value_gaps'] = self._detect_fair_value_gaps_v2(df)
            
            # 3. Order Blocks (institutional footprints)
            patterns['order_blocks'] = self._detect_order_blocks(df)
            
            # 4. Imbalances (single-print areas)
            patterns['imbalances'] = self._detect_imbalances_v2(df)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting smart money patterns: {e}")
            return patterns
    
    def _detect_liquidity_sweeps_v2(self, df: pd.DataFrame, lookback: int = 30) -> List[Dict]:
        """Improved liquidity sweep detection"""
        sweeps = []
        
        for i in range(lookback, len(df) - 2):
            recent_data = df.iloc[i-lookback:i]
            current_candle = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # High sweep detection
            recent_high = recent_data['high'].max()
            if (current_candle['high'] > recent_high * 1.002 and  # Clear break
                current_candle['close'] < current_candle['open'] and  # Rejection
                next_candle['close'] < current_candle['close']):  # Follow-through
                
                sweeps.append({
                    'type': 'high_sweep',
                    'price': current_candle['high'],
                    'index': i,
                    'strength': (current_candle['high'] - recent_high) / recent_high * 100,
                    'volume_confirmation': current_candle['volume'] > df['volume'].iloc[i-5:i].mean()
                })
            
            # Low sweep detection
            recent_low = recent_data['low'].min()
            if (current_candle['low'] < recent_low * 0.998 and  # Clear break
                current_candle['close'] > current_candle['open'] and  # Rejection
                next_candle['close'] > current_candle['close']):  # Follow-through
                
                sweeps.append({
                    'type': 'low_sweep',
                    'price': current_candle['low'],
                    'index': i,
                    'strength': (recent_low - current_candle['low']) / recent_low * 100,
                    'volume_confirmation': current_candle['volume'] > df['volume'].iloc[i-5:i].mean()
                })
        
        return sweeps[-5:]  # Keep only recent sweeps
    
    def _detect_fair_value_gaps_v2(self, df: pd.DataFrame) -> List[Dict]:
        """Corrected Fair Value Gap detection"""
        fvgs = []
        
        for i in range(2, len(df)):
            candle1 = df.iloc[i-2]
            candle2 = df.iloc[i-1]
            candle3 = df.iloc[i]
            
            # Bullish FVG: gap between candle1's high and candle3's low
            if candle1['high'] < candle3['low']:
                gap_size = (candle3['low'] - candle1['high']) / candle1['high'] * 100
                
                # Only significant gaps (> 0.1%)
                if gap_size > 0.1:
                    fvgs.append({
                        'type': 'bullish_fvg',
                        'top': candle3['low'],
                        'bottom': candle1['high'],
                        'index': i,
                        'size_pct': gap_size,
                        'filled': df['close'].iloc[-1] <= candle1['high']
                    })
            
            # Bearish FVG: gap between candle1's low and candle3's high
            elif candle1['low'] > candle3['high']:
                gap_size = (candle1['low'] - candle3['high']) / candle1['low'] * 100
                
                if gap_size > 0.1:
                    fvgs.append({
                        'type': 'bearish_fvg',
                        'top': candle1['low'],
                        'bottom': candle3['high'],
                        'index': i,
                        'size_pct': gap_size,
                        'filled': df['close'].iloc[-1] >= candle1['low']
                    })
        
        return fvgs[-10:]
    
    def _detect_order_blocks(self, df: pd.DataFrame, min_body_pct: float = 0.5) -> List[Dict]:
        """Detect institutional order blocks"""
        order_blocks = []
        
        for i in range(10, len(df)):
            candle = df.iloc[i]
            
            # Calculate candle body
            body_size = abs(candle['close'] - candle['open'])
            total_size = candle['high'] - candle['low']
            
            if total_size == 0:
                continue
            
            body_pct = body_size / total_size
            
            # Strong bearish order block
            if (candle['close'] < candle['open'] and  # Bearish candle
                body_pct > min_body_pct and  # Strong body
                candle['volume'] > df['volume'].iloc[i-5:i].mean() * 1.5):  # High volume
                
                # Check if price has moved away significantly
                future_low = df['low'].iloc[i+1:i+20].min() if i+20 < len(df) else df['low'].iloc[i+1:].min()
                if future_low < candle['low'] * 0.99:  # 1% move away
                    order_blocks.append({
                        'type': 'bearish_ob',
                        'top': candle['open'],
                        'bottom': candle['close'],
                        'index': i,
                        'volume': candle['volume'],
                        'tested': df['high'].iloc[i+1:].max() >= candle['close'] if i+1 < len(df) else False
                    })
            
            # Strong bullish order block
            elif (candle['close'] > candle['open'] and  # Bullish candle
                  body_pct > min_body_pct and  # Strong body
                  candle['volume'] > df['volume'].iloc[i-5:i].mean() * 1.5):  # High volume
                
                future_high = df['high'].iloc[i+1:i+20].max() if i+20 < len(df) else df['high'].iloc[i+1:].max()
                if future_high > candle['high'] * 1.01:  # 1% move away
                    order_blocks.append({
                        'type': 'bullish_ob',
                        'top': candle['close'],
                        'bottom': candle['open'],
                        'index': i,
                        'volume': candle['volume'],
                        'tested': df['low'].iloc[i+1:].min() <= candle['close'] if i+1 < len(df) else False
                    })
        
        return order_blocks[-8:]
    
    def _detect_imbalances_v2(self, df: pd.DataFrame) -> List[Dict]:
        """Detect price imbalances (gaps)"""
        imbalances = []
        
        for i in range(1, len(df)):
            prev_candle = df.iloc[i-1]
            curr_candle = df.iloc[i]
            
            # Bullish imbalance
            if curr_candle['low'] > prev_candle['high']:
                gap_size = (curr_candle['low'] - prev_candle['high']) / prev_candle['high'] * 100
                if gap_size > 0.05:  # Significant gap
                    imbalances.append({
                        'type': 'bullish_imbalance',
                        'top': curr_candle['low'],
                        'bottom': prev_candle['high'],
                        'index': i,
                        'size_pct': gap_size,
                        'filled': df['close'].iloc[-1] <= prev_candle['high']
                    })
            
            # Bearish imbalance
            elif curr_candle['high'] < prev_candle['low']:
                gap_size = (prev_candle['low'] - curr_candle['high']) / prev_candle['low'] * 100
                if gap_size > 0.05:
                    imbalances.append({
                        'type': 'bearish_imbalance',
                        'top': prev_candle['low'],
                        'bottom': curr_candle['high'],
                        'index': i,
                        'size_pct': gap_size,
                        'filled': df['close'].iloc[-1] >= prev_candle['low']
                    })
        
        return imbalances[-8:]
    
    async def get_order_book_analysis(self, symbol: str) -> Dict:
        """Enhanced order book analysis with institutional detection"""
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit=100)
            
            if not order_book.get('bids') or not order_book.get('asks'):
                return {}
            
            bids = np.array(order_book['bids'])
            asks = np.array(order_book['asks'])
            
            # Basic metrics
            total_bid_volume = np.sum(bids[:, 1])
            total_ask_volume = np.sum(asks[:, 1])
            total_volume = total_bid_volume + total_ask_volume
            
            if total_volume == 0:
                return {}
            
            bid_pressure = total_bid_volume / total_volume
            spread_pct = ((asks[0][0] - bids[0][0]) / asks[0][0]) * 100
            
            # Detect large orders (potential institutional activity)
            bid_sizes = bids[:, 1]
            ask_sizes = asks[:, 1]
            
            bid_threshold = np.percentile(bid_sizes, 90)
            ask_threshold = np.percentile(ask_sizes, 90)
            
            large_bids = bids[bids[:, 1] > bid_threshold]
            large_asks = asks[asks[:, 1] > ask_threshold]
            
            # Order book imbalance at different levels
            levels_5_bid = np.sum(bids[:5, 1])
            levels_5_ask = np.sum(asks[:5, 1])
            imbalance_5 = levels_5_bid / (levels_5_bid + levels_5_ask) if (levels_5_bid + levels_5_ask) > 0 else 0.5
            
            return {
                'bid_pressure': round(bid_pressure, 3),
                'ask_pressure': round(1 - bid_pressure, 3),
                'spread_pct': round(spread_pct, 4),
                'large_bid_count': len(large_bids),
                'large_ask_count': len(large_asks),
                'imbalance_5_levels': round(imbalance_5, 3),
                'institutional_bias': 'bullish' if len(large_bids) > len(large_asks) else 'bearish',
                'liquidity_score': min(total_bid_volume, total_ask_volume) / max(total_bid_volume, total_ask_volume)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing order book for {symbol}: {e}")
            return {}
    
    def calculate_signal_confluence(self, df: pd.DataFrame, smart_money: Dict, 
                                  order_book: Dict) -> Dict:
        """Simplified confluence calculation focusing on key factors"""
        confluence = {
            'direction': 'neutral',
            'strength': 0.0,
            'factors': [],
            'score_breakdown': {}
        }
        
        try:
            current_price = df['close'].iloc[-1]
            scores = {'bullish': 0, 'bearish': 0}
            
            # 1. Trend Alignment (30% weight)
            trend_score = self._analyze_trend_alignment(df, current_price)
            scores['bullish'] += trend_score['bullish'] * 0.3
            scores['bearish'] += trend_score['bearish'] * 0.3
            confluence['factors'].extend(trend_score['factors'])
            confluence['score_breakdown']['trend'] = trend_score
            
            # 2. Volume Confirmation (25% weight)
            volume_score = self._analyze_volume_confirmation(df)
            scores['bullish'] += volume_score['bullish'] * 0.25
            scores['bearish'] += volume_score['bearish'] * 0.25
            confluence['factors'].extend(volume_score['factors'])
            confluence['score_breakdown']['volume'] = volume_score
            
            # 3. Smart Money Patterns (25% weight)
            sm_score = self._analyze_smart_money_confluence(smart_money, current_price)
            scores['bullish'] += sm_score['bullish'] * 0.25
            scores['bearish'] += sm_score['bearish'] * 0.25
            confluence['factors'].extend(sm_score['factors'])
            confluence['score_breakdown']['smart_money'] = sm_score
            
            # 4. Technical Confluence (20% weight)
            tech_score = self._analyze_technical_confluence(df)
            scores['bullish'] += tech_score['bullish'] * 0.2
            scores['bearish'] += tech_score['bearish'] * 0.2
            confluence['factors'].extend(tech_score['factors'])
            confluence['score_breakdown']['technical'] = tech_score
            
            # Determine overall direction and strength
            if scores['bullish'] > scores['bearish'] and scores['bullish'] > 0.6:
                confluence['direction'] = 'bullish'
                confluence['strength'] = scores['bullish']
            elif scores['bearish'] > scores['bullish'] and scores['bearish'] > 0.6:
                confluence['direction'] = 'bearish'
                confluence['strength'] = scores['bearish']
            else:
                confluence['direction'] = 'neutral'
                confluence['strength'] = max(scores['bullish'], scores['bearish'])
            
            return confluence
            
        except Exception as e:
            logger.error(f"Error calculating confluence: {e}")
            return confluence
    
    def _analyze_trend_alignment(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Analyze trend alignment"""
        score = {'bullish': 0, 'bearish': 0, 'factors': []}
        
        try:
            # EMA alignment
            if all(col in df.columns for col in ['ema_9', 'ema_21', 'ema_50']):
                ema_9 = df['ema_9'].iloc[-1]
                ema_21 = df['ema_21'].iloc[-1]
                ema_50 = df['ema_50'].iloc[-1]
                
                # Perfect bullish alignment
                if current_price > ema_9 > ema_21 > ema_50:
                    score['bullish'] += 1.0
                    score['factors'].append("Perfect EMA bullish alignment")
                # Perfect bearish alignment
                elif current_price < ema_9 < ema_21 < ema_50:
                    score['bearish'] += 1.0
                    score['factors'].append("Perfect EMA bearish alignment")
                # Partial alignment
                elif current_price > ema_21:
                    score['bullish'] += 0.5
                    score['factors'].append("Price above key EMA")
                elif current_price < ema_21:
                    score['bearish'] += 0.5
                    score['factors'].append("Price below key EMA")
            
            # VWAP confirmation
            if 'vwap' in df.columns:
                vwap = df['vwap'].iloc[-1]
                if current_price > vwap * 1.002:
                    score['bullish'] += 0.3
                    score['factors'].append("Price above VWAP")
                elif current_price < vwap * 0.998:
                    score['bearish'] += 0.3
                    score['factors'].append("Price below VWAP")
            
            # Market structure
            if 'market_structure' in df.columns:
                structure = df['market_structure'].iloc[-1]
                if structure == 'bullish':
                    score['bullish'] += 0.4
                    score['factors'].append("Bullish market structure")
                elif structure == 'bearish':
                    score['bearish'] += 0.4
                    score['factors'].append("Bearish market structure")
            
        except Exception as e:
            logger.error(f"Error in trend alignment analysis: {e}")
        
        return score
    
    def _analyze_volume_confirmation(self, df: pd.DataFrame) -> Dict:
        """Analyze volume confirmation"""
        score = {'bullish': 0, 'bearish': 0, 'factors': []}
        
        try:
            if 'volume_ratio' not in df.columns:
                return score
            
            volume_ratio = df['volume_ratio'].iloc[-1]
            current_candle = df.iloc[-1]
            price_change_pct = ((current_candle['close'] - current_candle['open']) / 
                               current_candle['open']) * 100
            
            # High volume with direction
            if volume_ratio > 2.0:
                if price_change_pct > 0:
                    score['bullish'] += 1.0
                    score['factors'].append("High volume bullish candle")
                else:
                    score['bearish'] += 1.0
                    score['factors'].append("High volume bearish candle")
            elif volume_ratio > 1.5:
                if price_change_pct > 0:
                    score['bullish'] += 0.6
                    score['factors'].append("Above-average volume bullish move")
                else:
                    score['bearish'] += 0.6
                    score['factors'].append("Above-average volume bearish move")
            
            # Volume trend (last 3 candles)
            recent_volume_trend = df['volume_ratio'].iloc[-3:].mean()
            if recent_volume_trend > 1.3:
                score['factors'].append("Sustained volume increase")
                # Add small boost to whichever direction has more score
                if score['bullish'] > score['bearish']:
                    score['bullish'] += 0.2
                elif score['bearish'] > score['bullish']:
                    score['bearish'] += 0.2
            
        except Exception as e:
            logger.error(f"Error in volume confirmation analysis: {e}")
        
        return score
    
    def _analyze_smart_money_confluence(self, smart_money: Dict, current_price: float) -> Dict:
        """Analyze smart money pattern confluence"""
        score = {'bullish': 0, 'bearish': 0, 'factors': []}
        
        try:
            # Liquidity sweeps
            sweeps = smart_money.get('liquidity_sweeps', [])
            for sweep in sweeps[-2:]:  # Recent sweeps only
                if sweep.get('volume_confirmation', False):
                    if sweep['type'] == 'low_sweep':
                        score['bullish'] += 0.4
                        score['factors'].append("Recent liquidity sweep of lows")
                    elif sweep['type'] == 'high_sweep':
                        score['bearish'] += 0.4
                        score['factors'].append("Recent liquidity sweep of highs")
            
            # Fair Value Gaps
            fvgs = smart_money.get('fair_value_gaps', [])
            for fvg in fvgs[-3:]:  # Recent FVGs
                if not fvg.get('filled', True):
                    if (fvg['type'] == 'bullish_fvg' and 
                        fvg['bottom'] <= current_price <= fvg['top']):
                        score['bullish'] += 0.5
                        score['factors'].append("Price in bullish FVG")
                    elif (fvg['type'] == 'bearish_fvg' and 
                          fvg['bottom'] <= current_price <= fvg['top']):
                        score['bearish'] += 0.5
                        score['factors'].append("Price in bearish FVG")
            
            # Order blocks
            order_blocks = smart_money.get('order_blocks', [])
            for ob in order_blocks[-2:]:  # Recent order blocks
                if not ob.get('tested', False):
                    if (ob['type'] == 'bullish_ob' and 
                        ob['bottom'] <= current_price <= ob['top']):
                        score['bullish'] += 0.6
                        score['factors'].append("Price at untested bullish order block")
                    elif (ob['type'] == 'bearish_ob' and 
                          ob['bottom'] <= current_price <= ob['top']):
                        score['bearish'] += 0.6
                        score['factors'].append("Price at untested bearish order block")
            
        except Exception as e:
            logger.error(f"Error in smart money confluence analysis: {e}")
        
        return score
    
    def _analyze_technical_confluence(self, df: pd.DataFrame) -> Dict:
        """Analyze technical indicator confluence"""
        score = {'bullish': 0, 'bearish': 0, 'factors': []}
        
        try:
            # RSI analysis
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                if 30 < rsi < 45:
                    score['bullish'] += 0.4
                    score['factors'].append("RSI in bullish zone")
                elif 55 < rsi < 70:
                    score['bearish'] += 0.4
                    score['factors'].append("RSI in bearish zone")
                elif rsi < 30:
                    score['bullish'] += 0.6
                    score['factors'].append("RSI oversold")
                elif rsi > 70:
                    score['bearish'] += 0.6
                    score['factors'].append("RSI overbought")
            
            # MACD momentum
            if 'macd_hist' in df.columns and len(df) > 1:
                macd_hist = df['macd_hist'].iloc[-1]
                macd_prev = df['macd_hist'].iloc[-2]
                
                if macd_hist > 0 and macd_hist > macd_prev:
                    score['bullish'] += 0.4
                    score['factors'].append("MACD bullish momentum")
                elif macd_hist < 0 and macd_hist < macd_prev:
                    score['bearish'] += 0.4
                    score['factors'].append("MACD bearish momentum")
            
            # Bollinger Bands
            if all(col in df.columns for col in ['bb_upper', 'bb_lower']):
                current_price = df['close'].iloc[-1]
                bb_upper = df['bb_upper'].iloc[-1]
                bb_lower = df['bb_lower'].iloc[-1]
                
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                
                if bb_position <= 0.1:  # Near lower band
                    score['bullish'] += 0.5
                    score['factors'].append("Price near BB lower band")
                elif bb_position >= 0.9:  # Near upper band
                    score['bearish'] += 0.5
                    score['factors'].append("Price near BB upper band")
            
        except Exception as e:
            logger.error(f"Error in technical confluence analysis: {e}")
        
        return score
    
    async def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """Generate high-quality trading signal"""
        try:
            if len(df) < 100:
                return None
            
            current_price = df['close'].iloc[-1]
            
            # Pre-screening filters
            volume_ratio = df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 0
            volatility = df['atr_pct'].iloc[-1] if 'atr_pct' in df.columns else 0
            
            if (volume_ratio < self.min_volume_surge or 
                volatility < self.min_volatility):
                return None
            
            # Get comprehensive analysis
            smart_money = self.detect_smart_money_patterns(df)
            order_book = await self.get_order_book_analysis(symbol)
            confluence = self.calculate_signal_confluence(df, smart_money, order_book)
            
            # Quality gate
            if (confluence['strength'] < 0.7 or 
                confluence['direction'] == 'neutral' or
                len(confluence['factors']) < 3):
                return None
            
            # Generate signal based on direction
            if confluence['direction'] == 'bullish':
                signal = self._create_long_signal(df, symbol, confluence, smart_money)
            else:
                signal = self._create_short_signal(df, symbol, confluence, smart_money)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def _create_long_signal(self, df: pd.DataFrame, symbol: str, 
                           confluence: Dict, smart_money: Dict) -> TradingSignal:
        """Create optimized LONG signal"""
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
        
        entry_price = current_price
        
        # Smart stop loss calculation
        stop_candidates = []
        
        # ATR-based stop
        stop_candidates.append(entry_price - (atr * 1.8))
        
        # Support level stop
        if 'support' in df.columns and pd.notna(df['support'].iloc[-1]):
            support = df['support'].iloc[-1]
            stop_candidates.append(support * 0.997)
        
        # Smart money level stops
        sweeps = smart_money.get('liquidity_sweeps', [])
        low_sweeps = [s for s in sweeps if s['type'] == 'low_sweep'][-1:]
        if low_sweeps:
            stop_candidates.append(low_sweeps[0]['price'] * 0.995)
        
        # Use the highest (most conservative) stop
        stop_loss = max(stop_candidates) if stop_candidates else entry_price * 0.985
        
        # Ensure max risk limit
        risk_pct = ((entry_price - stop_loss) / entry_price) * 100
        if risk_pct > self.max_risk_per_trade:
            stop_loss = entry_price * (1 - self.max_risk_per_trade / 100)
        
        # Calculate take profit
        risk = entry_price - stop_loss
        
        # Dynamic R:R based on confluence strength
        if confluence['strength'] > 0.85:
            rr_ratio = 4.0  # High confidence
        elif confluence['strength'] > 0.75:
            rr_ratio = 3.0
        else:
            rr_ratio = 2.5  # Minimum
        
        take_profit = entry_price + (risk * rr_ratio)
        
        # Adjust for resistance
        if 'resistance' in df.columns and pd.notna(df['resistance'].iloc[-1]):
            resistance = df['resistance'].iloc[-1]
            if take_profit > resistance:
                take_profit = resistance * 0.995
        
        # Final validation
        final_rr = (take_profit - entry_price) / (entry_price - stop_loss)
        if final_rr < self.min_rr_ratio:
            take_profit = entry_price + (risk * self.min_rr_ratio)
        
        signal_id = f"{symbol.replace('/', '')}_{int(datetime.now().timestamp())}_L"
        
        # Determine signal strength
        strength = SignalStrength.INSTITUTIONAL if confluence['strength'] > 0.85 else \
                  SignalStrength.STRONG if confluence['strength'] > 0.75 else \
                  SignalStrength.MODERATE
        
        return TradingSignal(
            signal_id=signal_id,
            timestamp=datetime.now().isoformat(),
            coin=symbol.replace('/USDT', ''),
            direction='LONG',
            entry_price=round(entry_price, 6),
            current_price=round(current_price, 6),
            take_profit=round(take_profit, 6),
            stop_loss=round(stop_loss, 6),
            confidence=min(95, int(confluence['strength'] * 100)),
            strength=strength,
            risk_reward_ratio=round(final_rr, 2),
            risk_percentage=round(((entry_price - stop_loss) / entry_price) * 100, 2),
            analysis_summary={
                'confluence_score': round(confluence['strength'], 3),
                'volume_confirmation': round(df['volume_ratio'].iloc[-1], 2) if 'volume_ratio' in df.columns else 1,
                'smart_money_signals': len([p for patterns in smart_money.values() if isinstance(patterns, list) for p in patterns]),
                'market_structure': df['market_structure'].iloc[-1] if 'market_structure' in df.columns else 'unknown'
            },
            confluence_factors=confluence['factors']
        )
    
    def _create_short_signal(self, df: pd.DataFrame, symbol: str, 
                            confluence: Dict, smart_money: Dict) -> TradingSignal:
        """Create optimized SHORT signal"""
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
        
        entry_price = current_price
        
        # Smart stop loss calculation
        stop_candidates = []
        
        # ATR-based stop
        stop_candidates.append(entry_price + (atr * 1.8))
        
        # Resistance level stop
        if 'resistance' in df.columns and pd.notna(df['resistance'].iloc[-1]):
            resistance = df['resistance'].iloc[-1]
            stop_candidates.append(resistance * 1.003)
        
        # Smart money level stops
        sweeps = smart_money.get('liquidity_sweeps', [])
        high_sweeps = [s for s in sweeps if s['type'] == 'high_sweep'][-1:]
        if high_sweeps:
            stop_candidates.append(high_sweeps[0]['price'] * 1.005)
        
        # Use the lowest (most conservative) stop
        stop_loss = min(stop_candidates) if stop_candidates else entry_price * 1.015
        
        # Ensure max risk limit
        risk_pct = ((stop_loss - entry_price) / entry_price) * 100
        if risk_pct > self.max_risk_per_trade:
            stop_loss = entry_price * (1 + self.max_risk_per_trade / 100)
        
        # Calculate take profit
        risk = stop_loss - entry_price
        
        # Dynamic R:R based on confluence strength
        if confluence['strength'] > 0.85:
            rr_ratio = 4.0
        elif confluence['strength'] > 0.75:
            rr_ratio = 3.0
        else:
            rr_ratio = 2.5
        
        take_profit = entry_price - (risk * rr_ratio)
        
        # Adjust for support
        if 'support' in df.columns and pd.notna(df['support'].iloc[-1]):
            support = df['support'].iloc[-1]
            if take_profit < support:
                take_profit = support * 1.005
        
        # Final validation
        final_rr = (entry_price - take_profit) / (stop_loss - entry_price)
        if final_rr < self.min_rr_ratio:
            take_profit = entry_price - (risk * self.min_rr_ratio)
        
        signal_id = f"{symbol.replace('/', '')}_{int(datetime.now().timestamp())}_S"
        
        # Determine signal strength
        strength = SignalStrength.INSTITUTIONAL if confluence['strength'] > 0.85 else \
                  SignalStrength.STRONG if confluence['strength'] > 0.75 else \
                  SignalStrength.MODERATE
        
        return TradingSignal(
            signal_id=signal_id,
            timestamp=datetime.now().isoformat(),
            coin=symbol.replace('/USDT', ''),
            direction='SHORT',
            entry_price=round(entry_price, 6),
            current_price=round(current_price, 6),
            take_profit=round(take_profit, 6),
            stop_loss=round(stop_loss, 6),
            confidence=min(95, int(confluence['strength'] * 100)),
            strength=strength,
            risk_reward_ratio=round(final_rr, 2),
            risk_percentage=round(((stop_loss - entry_price) / entry_price) * 100, 2),
            analysis_summary={
                'confluence_score': round(confluence['strength'], 3),
                'volume_confirmation': round(df['volume_ratio'].iloc[-1], 2) if 'volume_ratio' in df.columns else 1,
                'smart_money_signals': len([p for patterns in smart_money.values() if isinstance(patterns, list) for p in patterns]),
                'market_structure': df['market_structure'].iloc[-1] if 'market_structure' in df.columns else 'unknown'
            },
            confluence_factors=confluence['factors']
        )
    
    async def scan_all_coins(self) -> List[Dict]:
        """Production-ready coin scanning"""
        signals = []
        processed_count = 0
        
        # Randomize scanning order to avoid bias
        import random
        coins_to_scan = self.coins.copy()
        random.shuffle(coins_to_scan)
        
        logger.info(f"Starting production scan of {len(coins_to_scan)} coins")
        
        for symbol in coins_to_scan:
            try:
                if len(signals) >= self.max_signals_per_scan:
                    break
                
                processed_count += 1
                
                # Get market data
                df = await self.get_market_data(symbol, '1h', 200)
                if df.empty:
                    continue
                
                # Calculate indicators
                df = self.calculate_technical_indicators(df)
                if len(df) < 50:
                    continue
                
                # Generate signal
                signal = await self.generate_signal(df, symbol)
                
                if signal:
                    # Convert to dict for compatibility
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
                            'confluence_score': signal.analysis_summary['confluence_score'],
                            'confluence_factors': signal.confluence_factors,
                            'risk_reward_ratio': signal.risk_reward_ratio,
                            'risk_percentage': signal.risk_percentage,
                            'signal_grade': 'institutional' if signal.strength == SignalStrength.INSTITUTIONAL else 'standard',
                            'volume_confirmation': signal.analysis_summary['volume_confirmation'],
                            'smart_money_signals': signal.analysis_summary['smart_money_signals'],
                            'market_structure': signal.analysis_summary['market_structure']
                        },
                        'indicators': {
                            'rsi': round(df['rsi'].iloc[-1], 2) if 'rsi' in df.columns else 50,
                            'volume_ratio': signal.analysis_summary['volume_confirmation'],
                            'atr_percentage': round(df['atr_pct'].iloc[-1], 2) if 'atr_pct' in df.columns else 1
                        }
                    }
                    
                    signals.append(signal_dict)
                    logger.info(f"Production signal: {symbol} {signal.direction} "
                              f"(Strength: {signal.strength.value}, Confluence: {signal.analysis_summary['confluence_score']:.3f})")
                
                # Rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        logger.info(f"Production scan complete: {len(signals)} high-quality signals from {processed_count} coins")
        return signals
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price with enhanced reliability"""
        try:
            # Try ticker first
            ticker = self.exchange.fetch_ticker(f"{symbol}/USDT")
            price = ticker.get('last')
            
            if price and price > 0:
                return float(price)
            
            # Fallback to mark price
            price = ticker.get('mark')
            if price and price > 0:
                return float(price)
            
            # Final fallback to close price
            price = ticker.get('close')
            if price and price > 0:
                return float(price)
            
            logger.warning(f"Could not get valid price for {symbol}")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return 0.0
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except:
            pass