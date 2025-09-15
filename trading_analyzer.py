import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import asyncio
import json
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class AdvancedTradingAnalyzer:
    def __init__(self, database=None):
        # Load environment variables
        load_dotenv()
        
        # Initialize exchange with proper error handling
        try:
            self.exchange = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY', ''),
                'secret': os.getenv('BINANCE_SECRET_KEY', ''),
                'sandbox': False,
                'rateLimit': 1200,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })
            # Test connection
            self.exchange.load_markets()
            logger.info("Connected to Binance exchange successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            # Initialize without API keys for testing
            self.exchange = ccxt.binance({
                'rateLimit': 1200,
                'enableRateLimit': True,
                'sandbox': False
            })
        
        self.database = database
        
        # Carefully selected coins with good liquidity and volume
        self.coins = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
            'SOL/USDT', 'DOGE/USDT', 'DOT/USDT', 'MATIC/USDT', 'LINK/USDT',
            'AVAX/USDT', 'UNI/USDT', 'LTC/USDT', 'ATOM/USDT', 'NEAR/USDT'
        ]
        
        # Enhanced signal generation parameters - optimized for smart but not over-conservative
        self.min_confidence_score = 6  # Reduced from 7 (60% confidence minimum)
        self.min_risk_reward_ratio = 2.5  # Increased from 2.0 for better quality
        self.max_signals_per_scan = 2  # Reduced from 3 for higher quality
        self.volume_threshold = 1.3  # Reduced from 2.0 (less conservative)
        self.volatility_threshold = 0.8  # Reduced from 0.02 (in percentage terms)
        
        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 0.5  # 500ms between requests per symbol
        
        if self.database:
            self.database.log_bot_activity(
                'INFO', 'ANALYZER', 'Enhanced trading analyzer initialized',
                f'Monitoring {len(self.coins)} coins with institutional-grade criteria'
            )
    
    async def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> pd.DataFrame:
        """Fetch market data with enhanced error handling and rate limiting"""
        try:
            # Rate limiting per symbol
            now = datetime.now().timestamp()
            if symbol in self.last_request_time:
                time_since_last = now - self.last_request_time[symbol]
                if time_since_last < self.min_request_interval:
                    await asyncio.sleep(self.min_request_interval - time_since_last)
            
            self.last_request_time[symbol] = now
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) < 100:
                logger.warning(f"Insufficient data for {symbol}: {len(ohlcv) if ohlcv else 0} candles")
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Validate data quality
            if df['volume'].sum() == 0:
                logger.warning(f"No volume data for {symbol}")
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators with validation"""
        if df.empty or len(df) < 100:
            return df
        
        try:
            # Ensure we have clean data
            df = df.dropna()
            
            # Moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # RSI with proper calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume analysis
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # ATR for volatility
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            df['atr_percentage'] = (df['atr'] / df['close']) * 100
            
            # Stochastic Oscillator
            lowest_low = df['low'].rolling(window=14).min()
            highest_high = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # Momentum and volatility
            df['price_change'] = df['close'].pct_change(periods=1)
            df['volatility'] = df['price_change'].rolling(20).std() * np.sqrt(252)
            
            # Support and resistance levels
            window = 10
            df['local_high'] = df['high'].rolling(window=window, center=True).max()
            df['local_low'] = df['low'].rolling(window=window, center=True).min()
            df['is_resistance'] = df['high'] == df['local_high']
            df['is_support'] = df['low'] == df['local_low']
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def calculate_volume_profile(self, df: pd.DataFrame, price_levels: int = 30) -> Dict:
        """Calculate volume profile to find high volume nodes and fair value areas"""
        try:
            if len(df) < 50:
                return {}
            
            # Get price range
            price_min = df['low'].min()
            price_max = df['high'].max()
            price_step = (price_max - price_min) / price_levels
            
            # Initialize volume buckets
            volume_profile = {}
            price_levels_list = []
            
            for i in range(price_levels):
                level_price = price_min + (i * price_step)
                price_levels_list.append(level_price)
                volume_profile[level_price] = 0
            
            # Distribute volume across price levels
            for _, row in df.iterrows():
                candle_range = row['high'] - row['low']
                if candle_range > 0:
                    # Distribute volume proportionally across the candle's range
                    for level in price_levels_list:
                        if row['low'] <= level <= row['high']:
                            # Weight by proximity to OHLC average
                            ohlc_avg = (row['open'] + row['high'] + row['low'] + row['close']) / 4
                            weight = 1 - abs(level - ohlc_avg) / candle_range
                            volume_profile[level] += row['volume'] * weight
            
            # Find high volume nodes (HVN) and low volume nodes (LVN)
            volumes = list(volume_profile.values())
            volume_threshold_high = np.percentile(volumes, 80)
            volume_threshold_low = np.percentile(volumes, 20)
            
            hvn_levels = []
            lvn_levels = []
            
            for price, volume in volume_profile.items():
                if volume >= volume_threshold_high:
                    hvn_levels.append({'price': price, 'volume': volume})
                elif volume <= volume_threshold_low:
                    lvn_levels.append({'price': price, 'volume': volume})
            
            # Find Point of Control (POC) - highest volume level
            poc_price = max(volume_profile.items(), key=lambda x: x[1])[0]
            
            # Calculate Value Area (70% of volume)
            sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
            total_volume = sum(volumes)
            value_area_volume = 0
            value_area_high = poc_price
            value_area_low = poc_price
            
            for price, volume in sorted_levels:
                value_area_volume += volume
                value_area_high = max(value_area_high, price)
                value_area_low = min(value_area_low, price)
                if value_area_volume >= total_volume * 0.7:
                    break
            
            return {
                'poc_price': poc_price,
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'hvn_levels': hvn_levels,
                'lvn_levels': lvn_levels,
                'volume_profile': volume_profile
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {e}")
            return {}

    def detect_smart_money_concepts(self, df: pd.DataFrame) -> Dict:
        """Detect smart money concepts like liquidity sweeps, fair value gaps, imbalances"""
        try:
            if len(df) < 20:
                return {}
            
            current_price = df['close'].iloc[-1]
            concepts = {
                'liquidity_sweeps': [],
                'fair_value_gaps': [],
                'imbalances': [],
                'break_of_structure': None
            }
            
            # 1. Detect Liquidity Sweeps
            # Look for moves that sweep previous highs/lows then reverse quickly
            for i in range(10, len(df) - 5):
                # Check for high sweep
                prev_high = df['high'].iloc[i-10:i].max()
                current_high = df['high'].iloc[i]
                next_candles = df.iloc[i+1:i+4]
                
                if (current_high > prev_high and 
                    len(next_candles) > 0 and
                    next_candles['close'].min() < df['close'].iloc[i] * 0.998):
                    
                    concepts['liquidity_sweeps'].append({
                        'type': 'high_sweep',
                        'price': current_high,
                        'index': i,
                        'strength': (current_high - prev_high) / prev_high * 100
                    })
                
                # Check for low sweep
                prev_low = df['low'].iloc[i-10:i].min()
                current_low = df['low'].iloc[i]
                
                if (current_low < prev_low and 
                    len(next_candles) > 0 and
                    next_candles['close'].max() > df['close'].iloc[i] * 1.002):
                    
                    concepts['liquidity_sweeps'].append({
                        'type': 'low_sweep',
                        'price': current_low,
                        'index': i,
                        'strength': (prev_low - current_low) / prev_low * 100
                    })
            
            # 2. Detect Fair Value Gaps (FVG)
            for i in range(2, len(df)):
                # Bullish FVG: gap between candle 1 high and candle 3 low
                if i >= 2:
                    gap_low = df['high'].iloc[i-2]
                    gap_high = df['low'].iloc[i]
                    gap_middle = df.iloc[i-1]
                    
                    # Bullish FVG condition
                    if (gap_low < gap_high and 
                        gap_middle['low'] > gap_low and 
                        gap_middle['high'] < gap_high):
                        
                        concepts['fair_value_gaps'].append({
                            'type': 'bullish_fvg',
                            'low': gap_low,
                            'high': gap_high,
                            'index': i,
                            'filled': current_price <= gap_low
                        })
                    
                    # Bearish FVG condition
                    elif (gap_low > gap_high and 
                          gap_middle['high'] < gap_low and 
                          gap_middle['low'] > gap_high):
                        
                        concepts['fair_value_gaps'].append({
                            'type': 'bearish_fvg',
                            'low': gap_high,
                            'high': gap_low,
                            'index': i,
                            'filled': current_price >= gap_low
                        })
            
            # 3. Detect Imbalances (single candle gaps)
            for i in range(1, len(df)):
                prev_candle = df.iloc[i-1]
                curr_candle = df.iloc[i]
                
                # Bullish imbalance (gap up)
                if curr_candle['low'] > prev_candle['high']:
                    concepts['imbalances'].append({
                        'type': 'bullish_imbalance',
                        'low': prev_candle['high'],
                        'high': curr_candle['low'],
                        'index': i,
                        'filled': current_price <= prev_candle['high']
                    })
                
                # Bearish imbalance (gap down)
                elif curr_candle['high'] < prev_candle['low']:
                    concepts['imbalances'].append({
                        'type': 'bearish_imbalance',
                        'low': curr_candle['high'],
                        'high': prev_candle['low'],
                        'index': i,
                        'filled': current_price >= prev_candle['low']
                    })
            
            # 4. Detect Break of Structure (BOS)
            # Look for significant breaks of recent highs/lows
            recent_high = df['high'].iloc[-20:].max()
            recent_low = df['low'].iloc[-20:].min()
            
            if current_price > recent_high * 1.001:
                concepts['break_of_structure'] = {
                    'type': 'bullish_bos',
                    'broken_level': recent_high,
                    'current_price': current_price
                }
            elif current_price < recent_low * 0.999:
                concepts['break_of_structure'] = {
                    'type': 'bearish_bos',
                    'broken_level': recent_low,
                    'current_price': current_price
                }
            
            return concepts
            
        except Exception as e:
            logger.error(f"Error detecting smart money concepts: {e}")
            return {}

    def multi_timeframe_confluence(self, symbol: str) -> Dict:
        """Analyze multiple timeframes for confluence - simplified but effective"""
        try:
            timeframes = {
                '15m': {'weight': 0.2, 'trend': None, 'strength': 0},
                '1h': {'weight': 0.4, 'trend': None, 'strength': 0},  # Main timeframe
                '4h': {'weight': 0.4, 'trend': None, 'strength': 0}
            }
            
            # This is a simplified version - in practice you'd fetch all timeframes
            # For now, we'll use the 1h data and simulate others
            df_1h = asyncio.run(self.get_market_data(symbol, '1h', 100))
            if df_1h.empty:
                return {}
            
            df_1h = self.calculate_technical_indicators(df_1h)
            
            # Analyze 1h trend
            sma_20 = df_1h['sma_20'].iloc[-1]
            sma_50 = df_1h['sma_50'].iloc[-1]
            macd_hist = df_1h['macd_hist'].iloc[-1]
            rsi = df_1h['rsi'].iloc[-1]
            
            # Determine trend strength for 1h
            bullish_signals = 0
            bearish_signals = 0
            
            if sma_20 > sma_50:
                bullish_signals += 1
            else:
                bearish_signals += 1
                
            if macd_hist > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1
                
            if rsi > 50:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            if bullish_signals > bearish_signals:
                timeframes['1h']['trend'] = 'bullish'
                timeframes['1h']['strength'] = bullish_signals / 3
            else:
                timeframes['1h']['trend'] = 'bearish'
                timeframes['1h']['strength'] = bearish_signals / 3
            
            # Simulate higher timeframe bias (4h)
            # Use longer period moving averages to simulate higher timeframe
            sma_80 = df_1h['close'].rolling(80).mean().iloc[-1]  # Simulates 4h trend
            current_price = df_1h['close'].iloc[-1]
            
            if current_price > sma_80:
                timeframes['4h']['trend'] = 'bullish'
                timeframes['4h']['strength'] = min(1.0, (current_price - sma_80) / sma_80 * 100)
            else:
                timeframes['4h']['trend'] = 'bearish'
                timeframes['4h']['strength'] = min(1.0, (sma_80 - current_price) / sma_80 * 100)
            
            # Calculate overall confluence score
            bullish_weight = 0
            bearish_weight = 0
            
            for tf, data in timeframes.items():
                if tf == '15m':  # Skip 15m for now
                    continue
                    
                weight = data['weight'] * data['strength']
                if data['trend'] == 'bullish':
                    bullish_weight += weight
                elif data['trend'] == 'bearish':
                    bearish_weight += weight
            
            return {
                'timeframes': timeframes,
                'bullish_confluence': bullish_weight,
                'bearish_confluence': bearish_weight,
                'dominant_bias': 'bullish' if bullish_weight > bearish_weight else 'bearish',
                'confluence_strength': abs(bullish_weight - bearish_weight)
            }
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {e}")
            return {}

    def analyze_order_book_pressure(self, symbol: str) -> Dict:
        """Analyze order book for supply/demand pressure"""
        try:
            # Get order book
            order_book = self.exchange.fetch_order_book(symbol, limit=20)
            
            bids = np.array(order_book['bids'])
            asks = np.array(order_book['asks'])
            
            if len(bids) == 0 or len(asks) == 0:
                return {}
            
            # Calculate bid/ask pressure
            bid_volume = np.sum(bids[:, 1])
            ask_volume = np.sum(asks[:, 1])
            total_volume = bid_volume + ask_volume
            
            if total_volume == 0:
                return {}
            
            bid_pressure = bid_volume / total_volume
            ask_pressure = ask_volume / total_volume
            
            # Calculate spread
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread_pct = ((best_ask - best_bid) / best_ask) * 100
            
            # Analyze volume at key levels
            mid_price = (best_bid + best_ask) / 2
            significant_bid_levels = []
            significant_ask_levels = []
            
            for bid in bids:
                if bid[1] > bid_volume * 0.1:  # More than 10% of total bid volume
                    significant_bid_levels.append(bid[0])
            
            for ask in asks:
                if ask[1] > ask_volume * 0.1:  # More than 10% of total ask volume
                    significant_ask_levels.append(ask[0])
            
            return {
                'bid_pressure': bid_pressure,
                'ask_pressure': ask_pressure,
                'spread_percentage': spread_pct,
                'significant_support': significant_bid_levels,
                'significant_resistance': significant_ask_levels,
                'liquidity_ratio': min(bid_volume, ask_volume) / max(bid_volume, ask_volume)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing order book for {symbol}: {e}")
            return {}

    def enhanced_signal_generation(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Enhanced signal generation using institutional-grade concepts"""
        try:
            if len(df) < 100:
                return None
            
            current_price = df['close'].iloc[-1]
            
            # Get enhanced analysis data
            volume_profile = self.calculate_volume_profile(df)
            smart_money = self.detect_smart_money_concepts(df)
            mtf_analysis = self.multi_timeframe_confluence(symbol)
            order_book_data = self.analyze_order_book_pressure(symbol)
            
            # Start with cleaner scoring system
            signal_strength = 0
            max_strength = 10
            confidence_reasons = []
            direction = None
            
            # 1. Multi-timeframe confluence (40% weight)
            if mtf_analysis:
                confluence_strength = mtf_analysis.get('confluence_strength', 0)
                dominant_bias = mtf_analysis.get('dominant_bias')
                
                if confluence_strength > 0.6:  # Strong confluence
                    signal_strength += 4
                    direction = dominant_bias
                    confidence_reasons.append(f"Strong {dominant_bias} confluence across timeframes")
                elif confluence_strength > 0.3:  # Moderate confluence
                    signal_strength += 2
                    direction = dominant_bias
                    confidence_reasons.append(f"Moderate {dominant_bias} confluence")
            
            # 2. Volume Profile Analysis (30% weight)
            if volume_profile and direction:
                poc_price = volume_profile.get('poc_price', current_price)
                value_area_high = volume_profile.get('value_area_high', current_price)
                value_area_low = volume_profile.get('value_area_low', current_price)
                
                # Check if price is at key volume levels
                poc_distance = abs(current_price - poc_price) / current_price * 100
                
                if poc_distance < 0.5:  # Very close to POC
                    signal_strength += 3
                    confidence_reasons.append("Price at Point of Control (high volume area)")
                elif (direction == 'bullish' and current_price <= value_area_low * 1.002):
                    signal_strength += 2
                    confidence_reasons.append("Price at value area low - potential bounce")
                elif (direction == 'bearish' and current_price >= value_area_high * 0.998):
                    signal_strength += 2
                    confidence_reasons.append("Price at value area high - potential rejection")
            
            # 3. Smart Money Concepts (20% weight)
            if smart_money and direction:
                # Check for recent liquidity sweeps
                recent_sweeps = [s for s in smart_money.get('liquidity_sweeps', []) 
                               if s['index'] >= len(df) - 10]
                
                if recent_sweeps:
                    for sweep in recent_sweeps:
                        if ((direction == 'bullish' and sweep['type'] == 'low_sweep') or
                            (direction == 'bearish' and sweep['type'] == 'high_sweep')):
                            signal_strength += 2
                            confidence_reasons.append(f"Recent {sweep['type']} - smart money move")
                            break
                
                # Check for unfilled fair value gaps
                unfilled_fvgs = [fvg for fvg in smart_money.get('fair_value_gaps', []) 
                               if not fvg.get('filled', True)]
                
                for fvg in unfilled_fvgs[-3:]:  # Check last 3 FVGs
                    if ((direction == 'bullish' and fvg['type'] == 'bullish_fvg' and 
                         fvg['low'] <= current_price <= fvg['high']) or
                        (direction == 'bearish' and fvg['type'] == 'bearish_fvg' and 
                         fvg['low'] <= current_price <= fvg['high'])):
                        signal_strength += 1
                        confidence_reasons.append("Price in unfilled fair value gap")
                        break
            
            # 4. Order Book Confirmation (10% weight)
            if order_book_data and direction:
                bid_pressure = order_book_data.get('bid_pressure', 0.5)
                
                if ((direction == 'bullish' and bid_pressure > 0.6) or
                    (direction == 'bearish' and bid_pressure < 0.4)):
                    signal_strength += 1
                    confidence_reasons.append("Order book pressure alignment")
            
            # Quality filters
            volume_ratio = df['volume_ratio'].iloc[-1]
            volatility = df['atr_percentage'].iloc[-1]
            
            # Minimum requirements
            if (signal_strength < self.min_confidence_score or  # Minimum strength
                volume_ratio < self.volume_threshold or   # Minimum volume
                volatility < self.volatility_threshold or     # Minimum volatility
                not direction):
                return None
            
            # Generate the actual signal
            if direction == 'bullish':
                signal = self.create_smart_long_signal(
                    df, symbol, signal_strength, confidence_reasons, 
                    volume_profile, smart_money
                )
            else:
                signal = self.create_smart_short_signal(
                    df, symbol, signal_strength, confidence_reasons, 
                    volume_profile, smart_money
                )
            
            if signal and self.validate_enhanced_signal(signal):
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error in enhanced signal generation for {symbol}: {e}")
            return None

    def create_smart_long_signal(self, df: pd.DataFrame, symbol: str, strength: int, 
                               reasons: List[str], volume_profile: Dict, smart_money: Dict) -> Dict:
        """Create optimized LONG signal using advanced concepts"""
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        entry_price = current_price
        
        # Smart stop loss using multiple concepts
        stop_options = []
        
        # ATR-based stop
        stop_options.append(entry_price - (atr * 1.2))
        
        # Volume profile based stop
        if volume_profile and volume_profile.get('value_area_low'):
            vp_stop = volume_profile['value_area_low'] * 0.995
            stop_options.append(vp_stop)
        
        # Smart money levels
        if smart_money:
            # Use recent liquidity sweep low as stop
            recent_sweeps = [s for s in smart_money.get('liquidity_sweeps', []) 
                            if s['type'] == 'low_sweep' and s['index'] >= len(df) - 20]
            if recent_sweeps:
                sweep_stop = min([s['price'] for s in recent_sweeps]) * 0.998
                stop_options.append(sweep_stop)
        
        # Choose the highest (most conservative) stop
        stop_loss = max(stop_options) if stop_options else entry_price * 0.985
        
        # Smart take profit
        risk = entry_price - stop_loss
        
        # Dynamic R:R based on signal strength
        if strength >= 9:
            rr_ratio = 4.0  # Very strong signal
        elif strength >= 7:
            rr_ratio = 3.0  # Strong signal
        else:
            rr_ratio = 2.5  # Moderate signal
        
        take_profit = entry_price + (risk * rr_ratio)
        
        # Adjust TP based on volume profile resistance
        if volume_profile and volume_profile.get('value_area_high'):
            vp_resistance = volume_profile['value_area_high']
            if take_profit > vp_resistance:
                take_profit = vp_resistance * 0.998
        
        signal_id = f"{symbol.replace('/', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_L"
        
        return {
            'signal_id': signal_id,
            'timestamp': datetime.now().isoformat(),
            'coin': symbol.replace('/USDT', ''),
            'direction': 'LONG',
            'entry_price': round(entry_price, 6),
            'current_price': round(current_price, 6),
            'take_profit': round(take_profit, 6),
            'stop_loss': round(stop_loss, 6),
            'confidence': min(95, strength * 10),
            'analysis_data': {
                'reasons': reasons,
                'strength': strength,
                'risk_reward_ratio': round((take_profit - entry_price) / (entry_price - stop_loss), 2),
                'volume_profile': volume_profile,
                'smart_money_concepts': smart_money,
                'risk_percentage': round(((entry_price - stop_loss) / entry_price) * 100, 2)
            },
            'indicators': {
                'rsi': round(df['rsi'].iloc[-1], 2),
                'volume_ratio': round(df['volume_ratio'].iloc[-1], 2),
                'atr_percentage': round(df['atr_percentage'].iloc[-1], 2)
            }
        }

    def create_smart_short_signal(self, df: pd.DataFrame, symbol: str, strength: int, 
                                reasons: List[str], volume_profile: Dict, smart_money: Dict) -> Dict:
        """Create optimized SHORT signal using advanced concepts"""
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        entry_price = current_price
        
        # Smart stop loss
        stop_options = []
        
        # ATR-based stop
        stop_options.append(entry_price + (atr * 1.2))
        
        # Volume profile based stop
        if volume_profile and volume_profile.get('value_area_high'):
            vp_stop = volume_profile['value_area_high'] * 1.005
            stop_options.append(vp_stop)
        
        # Smart money levels
        if smart_money:
            recent_sweeps = [s for s in smart_money.get('liquidity_sweeps', []) 
                            if s['type'] == 'high_sweep' and s['index'] >= len(df) - 20]
            if recent_sweeps:
                sweep_stop = max([s['price'] for s in recent_sweeps]) * 1.002
                stop_options.append(sweep_stop)
        
        # Choose the lowest (most conservative) stop
        stop_loss = min(stop_options) if stop_options else entry_price * 1.015
        
        # Smart take profit
        risk = stop_loss - entry_price
        
        if strength >= 9:
            rr_ratio = 4.0
        elif strength >= 7:
            rr_ratio = 3.0
        else:
            rr_ratio = 2.5
        
        take_profit = entry_price - (risk * rr_ratio)
        
        # Adjust TP based on volume profile support
        if volume_profile and volume_profile.get('value_area_low'):
            vp_support = volume_profile['value_area_low']
            if take_profit < vp_support:
                take_profit = vp_support * 1.002
        
        signal_id = f"{symbol.replace('/', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_S"
        
        return {
            'signal_id': signal_id,
            'timestamp': datetime.now().isoformat(),
            'coin': symbol.replace('/USDT', ''),
            'direction': 'SHORT',
            'entry_price': round(entry_price, 6),
            'current_price': round(current_price, 6),
            'take_profit': round(take_profit, 6),
            'stop_loss': round(stop_loss, 6),
            'confidence': min(95, strength * 10),
            'analysis_data': {
                'reasons': reasons,
                'strength': strength,
                'risk_reward_ratio': round((entry_price - take_profit) / (stop_loss - entry_price), 2),
                'volume_profile': volume_profile,
                'smart_money_concepts': smart_money,
                'risk_percentage': round(((stop_loss - entry_price) / entry_price) * 100, 2)
            },
            'indicators': {
                'rsi': round(df['rsi'].iloc[-1], 2),
                'volume_ratio': round(df['volume_ratio'].iloc[-1], 2),
                'atr_percentage': round(df['atr_percentage'].iloc[-1], 2)
            }
        }

    def validate_enhanced_signal(self, signal: Dict) -> bool:
        """Enhanced validation for smarter signals"""
        try:
            # Basic validations
            if not self.validate_signal_quality(signal):
                return False
            
            # Additional smart validations
            analysis_data = signal.get('analysis_data', {})
            
            # Ensure minimum signal strength
            if analysis_data.get('strength', 0) < self.min_confidence_score:
                return False
            
            # Ensure reasonable risk (max 1.5% per trade for smart signals)
            if analysis_data.get('risk_percentage', 0) > 1.5:
                return False
            
            # Ensure we have quality reasons
            reasons = analysis_data.get('reasons', [])
            if len(reasons) < 2:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating enhanced signal: {e}")
            return False

    def validate_signal_quality(self, signal: Dict) -> bool:
        """Comprehensive signal validation"""
        try:
            entry = signal['entry_price']
            tp = signal['take_profit']
            sl = signal['stop_loss']
            direction = signal['direction']
            
            # Basic price logic validation
            if direction == 'LONG':
                if not (sl < entry < tp):
                    return False
            else:
                if not (tp < entry < sl):
                    return False
            
            # Risk-reward validation
            rr_ratio = signal['analysis_data']['risk_reward_ratio']
            if rr_ratio < self.min_risk_reward_ratio:
                return False
            
            # Risk percentage validation (max 2% risk per trade)
            risk_pct = signal['analysis_data']['risk_percentage']
            if risk_pct > 2.0:
                return False
            
            # Confidence validation
            if signal['confidence'] < 60:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False

    async def scan_all_coins(self) -> List[Dict]:
        """Scan all coins with improved filtering and limits"""
        signals = []
        processed_count = 0
        
        # Randomize coin order to avoid bias
        import random
        coins_to_scan = self.coins.copy()
        random.shuffle(coins_to_scan)
        
        for symbol in coins_to_scan:
            try:
                if len(signals) >= self.max_signals_per_scan:
                    break
                
                logger.debug(f"Scanning {symbol}")
                processed_count += 1
                
                # Fetch and validate market data
                df = await self.get_market_data(symbol, '1h', 200)
                if df.empty or len(df) < 100:
                    continue
                
                # Calculate indicators
                df = self.calculate_technical_indicators(df)
                if len(df) < 50:
                    continue
                
                # Pre-filter based on basic conditions
                volume_ratio = df['volume_ratio'].iloc[-1]
                volatility = df['atr_percentage'].iloc[-1]
                
                if volume_ratio < self.volume_threshold or volatility < self.volatility_threshold:
                    continue
                
                # Generate signal using enhanced analysis
                signal = self.enhanced_signal_generation(df, symbol)
                
                if signal:
                    signals.append(signal)
                    logger.info(f"Institutional-grade signal generated: {symbol} {signal['direction']} (Strength: {signal['analysis_data']['strength']}/10)")
                
                # Rate limiting
                await asyncio.sleep(0.3)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        logger.info(f"Enhanced scan complete: {len(signals)} high-quality signals from {processed_count} coins analyzed")
        return signals
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price with caching and error handling"""
        try:
            ticker = self.exchange.fetch_ticker(f"{symbol}/USDT")
            price = ticker.get('last', 0)
            
            if price <= 0:
                logger.warning(f"Invalid price for {symbol}: {price}")
                return 0.0
            
            return float(price)
            
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return 0.0