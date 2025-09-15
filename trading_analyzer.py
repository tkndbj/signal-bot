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
        
        # Enhanced signal generation parameters
        self.min_confidence_score = 7  # Increased minimum score (out of 10)
        self.min_risk_reward_ratio = 2.0  # Minimum 2:1 R:R ratio
        self.max_signals_per_scan = 3  # Maximum signals per scan cycle
        self.volume_threshold = 2.0  # Minimum volume ratio
        self.volatility_threshold = 0.02  # Minimum volatility for signals
        
        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 0.5  # 500ms between requests per symbol
        
        if self.database:
            self.database.log_bot_activity(
                'INFO', 'ANALYZER', 'Trading analyzer initialized',
                f'Monitoring {len(self.coins)} coins with strict criteria'
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
    
    def detect_supply_demand_zones(self, df: pd.DataFrame) -> Dict:
        """Detect supply and demand zones using price action"""
        try:
            if len(df) < 50:
                return {}
            
            current_price = df['close'].iloc[-1]
            zones = {'demand_zones': [], 'supply_zones': []}
            
            # Look for significant moves and consolidations
            window = 10
            price_changes = df['close'].pct_change(periods=window)
            volume_spikes = df['volume_ratio'] > 1.5
            
            # Find demand zones (strong bounces with volume)
            for i in range(window, len(df) - window):
                # Check for strong bounce (>2% move up with volume)
                if (price_changes.iloc[i] > 0.02 and 
                    volume_spikes.iloc[i] and
                    df['low'].iloc[i] == df['low'].iloc[i-window:i+window].min()):
                    
                    zone_low = df['low'].iloc[i-2:i+3].min()
                    zone_high = df['high'].iloc[i-2:i+3].min()
                    
                    if zone_low < current_price < zone_high * 1.10:  # Within 10% above zone
                        zones['demand_zones'].append({
                            'low': zone_low,
                            'high': zone_high,
                            'strength': volume_spikes.iloc[i-2:i+3].sum(),
                            'distance_pct': ((current_price - zone_high) / current_price) * 100
                        })
            
            # Find supply zones (strong drops with volume)
            for i in range(window, len(df) - window):
                if (price_changes.iloc[i] < -0.02 and 
                    volume_spikes.iloc[i] and
                    df['high'].iloc[i] == df['high'].iloc[i-window:i+window].max()):
                    
                    zone_high = df['high'].iloc[i-2:i+3].max()
                    zone_low = df['low'].iloc[i-2:i+3].max()
                    
                    if zone_high > current_price > zone_low * 0.90:  # Within 10% below zone
                        zones['supply_zones'].append({
                            'low': zone_low,
                            'high': zone_high,
                            'strength': volume_spikes.iloc[i-2:i+3].sum(),
                            'distance_pct': ((zone_low - current_price) / current_price) * 100
                        })
            
            # Sort by proximity to current price
            zones['demand_zones'] = sorted(zones['demand_zones'], 
                                         key=lambda x: abs(x['distance_pct']))[:3]
            zones['supply_zones'] = sorted(zones['supply_zones'], 
                                         key=lambda x: abs(x['distance_pct']))[:3]
            
            return zones
            
        except Exception as e:
            logger.error(f"Error detecting supply/demand zones: {e}")
            return {}
    
    def advanced_signal_generation(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Generate signals using advanced order flow and supply/demand analysis"""
        try:
            if len(df) < 100:
                return None
            
            current_price = df['close'].iloc[-1]
            coin = symbol.replace('/USDT', '')
            
            # Pre-filtering checks
            volatility = df['volatility'].iloc[-1]
            volume_ratio = df['volume_ratio'].iloc[-1]
            
            # Skip if insufficient volatility or volume
            if volatility < self.volatility_threshold or volume_ratio < self.volume_threshold:
                return None
            
            # Get order book analysis
            order_book_data = self.analyze_order_book_pressure(symbol)
            
            # Get supply/demand zones
            sd_zones = self.detect_supply_demand_zones(df)
            
            # Advanced scoring system
            signal_score = 0
            confidence_factors = []
            direction_bias = None
            
            # 1. Order Book Analysis (Weight: 30%)
            if order_book_data:
                bid_pressure = order_book_data.get('bid_pressure', 0.5)
                liquidity_ratio = order_book_data.get('liquidity_ratio', 0.5)
                spread_pct = order_book_data.get('spread_percentage', 1.0)
                
                # Strong buying pressure
                if bid_pressure > 0.65 and liquidity_ratio > 0.7:
                    signal_score += 3
                    confidence_factors.append("Strong bid pressure in order book")
                    direction_bias = 'LONG'
                # Strong selling pressure
                elif bid_pressure < 0.35 and liquidity_ratio > 0.7:
                    signal_score += 3
                    confidence_factors.append("Strong ask pressure in order book")
                    direction_bias = 'SHORT'
                
                # Tight spreads indicate good liquidity
                if spread_pct < 0.1:
                    signal_score += 1
                    confidence_factors.append("Tight bid-ask spread")
            
            # 2. Supply/Demand Zone Analysis (Weight: 25%)
            if sd_zones:
                # Check proximity to demand zones for LONG
                for zone in sd_zones.get('demand_zones', []):
                    if abs(zone['distance_pct']) < 1.0 and zone['strength'] >= 2:
                        if direction_bias == 'LONG' or direction_bias is None:
                            signal_score += 2
                            confidence_factors.append(f"Near strong demand zone at ${zone['low']:.6f}")
                            direction_bias = 'LONG'
                
                # Check proximity to supply zones for SHORT
                for zone in sd_zones.get('supply_zones', []):
                    if abs(zone['distance_pct']) < 1.0 and zone['strength'] >= 2:
                        if direction_bias == 'SHORT' or direction_bias is None:
                            signal_score += 2
                            confidence_factors.append(f"Near strong supply zone at ${zone['high']:.6f}")
                            direction_bias = 'SHORT'
            
            # 3. Technical Confirmation (Weight: 25%)
            rsi = df['rsi'].iloc[-1]
            macd_hist = df['macd_hist'].iloc[-1]
            macd_hist_prev = df['macd_hist'].iloc[-2]
            bb_position = df['bb_position'].iloc[-1]
            
            # RSI divergence and extremes
            if rsi < 30 and direction_bias == 'LONG':
                signal_score += 2
                confidence_factors.append("RSI oversold confirmation")
            elif rsi > 70 and direction_bias == 'SHORT':
                signal_score += 2
                confidence_factors.append("RSI overbought confirmation")
            
            # MACD momentum
            if macd_hist > macd_hist_prev and direction_bias == 'LONG':
                signal_score += 1
                confidence_factors.append("MACD momentum increasing")
            elif macd_hist < macd_hist_prev and direction_bias == 'SHORT':
                signal_score += 1
                confidence_factors.append("MACD momentum decreasing")
            
            # 4. Volume Profile Analysis (Weight: 20%)
            recent_volume = df['volume'].iloc[-5:].mean()
            avg_volume = df['volume'].rolling(50).mean().iloc[-1]
            
            if recent_volume > avg_volume * 1.5:
                signal_score += 1
                confidence_factors.append("Above-average volume confirmation")
            
            # Volume price trend alignment
            price_trend = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
            volume_trend = (recent_volume - df['volume'].iloc[-15:-10].mean()) / df['volume'].iloc[-15:-10].mean()
            
            if price_trend > 0 and volume_trend > 0 and direction_bias == 'LONG':
                signal_score += 1
                confidence_factors.append("Price-volume trend alignment (bullish)")
            elif price_trend < 0 and volume_trend > 0 and direction_bias == 'SHORT':
                signal_score += 1
                confidence_factors.append("Price-volume trend alignment (bearish)")
            
            # Final signal generation
            if signal_score >= self.min_confidence_score and direction_bias:
                if direction_bias == 'LONG':
                    signal = self.create_optimized_long_signal(
                        df, symbol, signal_score, confidence_factors, sd_zones, order_book_data
                    )
                else:
                    signal = self.create_optimized_short_signal(
                        df, symbol, signal_score, confidence_factors, sd_zones, order_book_data
                    )
                
                if signal and self.validate_signal_quality(signal):
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def create_optimized_long_signal(self, df: pd.DataFrame, symbol: str, score: int, 
                                   factors: List[str], sd_zones: Dict, order_book: Dict) -> Dict:
        """Create optimized LONG signal with dynamic SL/TP based on market structure"""
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        # Dynamic entry strategy
        entry_price = current_price
        
        # Smart stop loss using multiple factors
        atr_stop = entry_price - (atr * 1.5)
        
        # Use demand zone if available
        zone_stop = None
        if sd_zones.get('demand_zones'):
            nearest_zone = min(sd_zones['demand_zones'], 
                             key=lambda x: abs(x['distance_pct']))
            zone_stop = nearest_zone['low'] * 0.998  # Slightly below zone
        
        # Use order book support if available
        ob_stop = None
        if order_book.get('significant_support'):
            ob_stop = min(order_book['significant_support']) * 0.999
        
        # Choose the most conservative (highest) stop loss
        stop_options = [s for s in [atr_stop, zone_stop, ob_stop] if s is not None]
        stop_loss = max(stop_options) if stop_options else atr_stop
        
        # Ensure minimum stop distance
        min_stop = entry_price * 0.995
        stop_loss = max(stop_loss, min_stop)
        
        # Dynamic take profit based on resistance and R:R
        risk = entry_price - stop_loss
        
        # Base R:R ratio adjusted for score
        if score >= 9:
            rr_ratio = 3.5
        elif score >= 8:
            rr_ratio = 3.0
        else:
            rr_ratio = 2.5
        
        take_profit = entry_price + (risk * rr_ratio)
        
        # Adjust for supply zones
        if sd_zones.get('supply_zones'):
            nearest_resistance = min(sd_zones['supply_zones'], 
                                   key=lambda x: abs(x['distance_pct']))
            if take_profit > nearest_resistance['low']:
                take_profit = nearest_resistance['low'] * 0.998
                # Recalculate R:R
                new_rr = (take_profit - entry_price) / risk
                if new_rr < 2.0:
                    return None  # Skip if R:R becomes too low
        
        # Adjust for order book resistance
        if order_book.get('significant_resistance'):
            max_resistance = min(order_book['significant_resistance'])
            if take_profit > max_resistance:
                take_profit = max_resistance * 0.999
        
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
            'confidence': min(95, score * 10),
            'analysis_data': {
                'reasons': factors,
                'score': score,
                'risk_reward_ratio': round((take_profit - entry_price) / (entry_price - stop_loss), 2),
                'atr': round(atr, 6),
                'supply_demand_zones': sd_zones,
                'order_book_analysis': order_book,
                'risk_percentage': round(((entry_price - stop_loss) / entry_price) * 100, 2)
            },
            'indicators': {
                'rsi': round(df['rsi'].iloc[-1], 2),
                'macd_hist': round(df['macd_hist'].iloc[-1], 6),
                'volume_ratio': round(df['volume_ratio'].iloc[-1], 2),
                'bb_position': round(df['bb_position'].iloc[-1], 3),
                'atr_percentage': round(df['atr_percentage'].iloc[-1], 2)
            }
        }
    
    def create_optimized_short_signal(self, df: pd.DataFrame, symbol: str, score: int, 
                                    factors: List[str], sd_zones: Dict, order_book: Dict) -> Dict:
        """Create optimized SHORT signal with dynamic SL/TP based on market structure"""
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        entry_price = current_price
        
        # Smart stop loss using multiple factors
        atr_stop = entry_price + (atr * 1.5)
        
        # Use supply zone if available
        zone_stop = None
        if sd_zones.get('supply_zones'):
            nearest_zone = min(sd_zones['supply_zones'], 
                             key=lambda x: abs(x['distance_pct']))
            zone_stop = nearest_zone['high'] * 1.002
        
        # Use order book resistance
        ob_stop = None
        if order_book.get('significant_resistance'):
            ob_stop = max(order_book['significant_resistance']) * 1.001
        
        # Choose the most conservative (lowest) stop loss
        stop_options = [s for s in [atr_stop, zone_stop, ob_stop] if s is not None]
        stop_loss = min(stop_options) if stop_options else atr_stop
        
        # Ensure minimum stop distance
        max_stop = entry_price * 1.005
        stop_loss = min(stop_loss, max_stop)
        
        # Dynamic take profit
        risk = stop_loss - entry_price
        
        if score >= 9:
            rr_ratio = 3.5
        elif score >= 8:
            rr_ratio = 3.0
        else:
            rr_ratio = 2.5
        
        take_profit = entry_price - (risk * rr_ratio)
        
        # Adjust for demand zones
        if sd_zones.get('demand_zones'):
            nearest_support = min(sd_zones['demand_zones'], 
                                key=lambda x: abs(x['distance_pct']))
            if take_profit < nearest_support['high']:
                take_profit = nearest_support['high'] * 1.002
                new_rr = (entry_price - take_profit) / risk
                if new_rr < 2.0:
                    return None
        
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
            'confidence': min(95, score * 10),
            'analysis_data': {
                'reasons': factors,
                'score': score,
                'risk_reward_ratio': round((entry_price - take_profit) / (stop_loss - entry_price), 2),
                'atr': round(atr, 6),
                'supply_demand_zones': sd_zones,
                'order_book_analysis': order_book,
                'risk_percentage': round(((stop_loss - entry_price) / entry_price) * 100, 2)
            },
            'indicators': {
                'rsi': round(df['rsi'].iloc[-1], 2),
                'macd_hist': round(df['macd_hist'].iloc[-1], 6),
                'volume_ratio': round(df['volume_ratio'].iloc[-1], 2),
                'bb_position': round(df['bb_position'].iloc[-1], 3),
                'atr_percentage': round(df['atr_percentage'].iloc[-1], 2)
            }
        }
    
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
            if signal['confidence'] < 70:
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
                volatility = df['volatility'].iloc[-1]
                
                if volume_ratio < self.volume_threshold or volatility < self.volatility_threshold:
                    continue
                
                # Generate signal using advanced analysis
                signal = self.advanced_signal_generation(df, symbol)
                
                if signal:
                    signals.append(signal)
                    logger.info(f"High-quality signal generated: {symbol} {signal['direction']} (Score: {signal['analysis_data']['score']})")
                
                # Rate limiting
                await asyncio.sleep(0.3)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        logger.info(f"Scan complete: {len(signals)} signals from {processed_count} coins analyzed")
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