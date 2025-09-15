import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
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
        
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY'),
            'sandbox': False,
            'rateLimit': 1200,
            'enableRateLimit': True
        })
        
        self.scaler = StandardScaler()
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_model_trained = False
        self.database = database
        
        # Coins to monitor
        self.coins = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
            'SOL/USDT', 'DOGE/USDT', 'DOT/USDT', 'MATIC/USDT', 'SHIB/USDT',
            'AVAX/USDT', 'LINK/USDT', 'UNI/USDT', 'LTC/USDT', 'ATOM/USDT',
            'NEAR/USDT', 'ALGO/USDT', 'MANA/USDT', 'SAND/USDT', 'FTM/USDT'
        ]
        
        # Signal generation thresholds
        self.min_confidence_score = 5  # Minimum score to generate signal
        self.min_risk_reward_ratio = 2.0  # Minimum R:R ratio
        self.max_signals_per_coin = 1  # Max active signals per coin
        
        # Log initialization
        if self.database:
            self.database.log_bot_activity(
                'INFO', 'ANALYZER', 'Trading analyzer initialized',
                f'Monitoring {len(self.coins)} coins'
            )
    
    async def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> pd.DataFrame:
        """Fetch market data for a symbol with rate limiting"""
        try:
            # Rate limiting
            await asyncio.sleep(0.1)
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if df.empty or len(df) < 50:
            return df
        
        try:
            # Price-based indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)  # Avoid division by zero
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)
            
            # ATR (Average True Range)
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
            df['stoch_k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # Momentum indicators
            df['momentum'] = df['close'].pct_change(periods=10) * 100
            df['roc'] = df['close'].pct_change(periods=10) * 100
            
            # Volatility
            df['volatility'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
            
            # Support and Resistance
            df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
            df['resistance1'] = 2 * df['pivot'] - df['low']
            df['support1'] = 2 * df['pivot'] - df['high']
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def detect_patterns(self, df: pd.DataFrame) -> Dict:
        """Enhanced pattern detection"""
        try:
            if len(df) < 20:
                return {}
            
            patterns = {}
            
            # Trend detection
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            current_price = df['close'].iloc[-1]
            
            if current_price > sma_20 > sma_50:
                patterns['trend'] = 'strong_bullish'
            elif current_price > sma_20:
                patterns['trend'] = 'bullish'
            elif current_price < sma_20 < sma_50:
                patterns['trend'] = 'strong_bearish'
            elif current_price < sma_20:
                patterns['trend'] = 'bearish'
            else:
                patterns['trend'] = 'neutral'
            
            # Divergence detection
            if len(df) >= 20:
                price_trend = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
                rsi_trend = df['rsi'].iloc[-1] - df['rsi'].iloc[-20]
                
                if price_trend > 0.02 and rsi_trend < -5:
                    patterns['divergence'] = 'bearish'
                elif price_trend < -0.02 and rsi_trend > 5:
                    patterns['divergence'] = 'bullish'
            
            # Momentum patterns
            macd_hist = df['macd_hist'].iloc[-3:].values
            if len(macd_hist) == 3:
                if macd_hist[-1] > macd_hist[-2] > macd_hist[-3]:
                    patterns['momentum'] = 'increasing_bullish'
                elif macd_hist[-1] < macd_hist[-2] < macd_hist[-3]:
                    patterns['momentum'] = 'increasing_bearish'
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return {}
    
    def calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Calculate dynamic support and resistance levels"""
        try:
            if len(df) < 20:
                return {}
            
            # Find local highs and lows
            window = 5
            df_copy = df.copy()
            df_copy['high_rolling'] = df_copy['high'].rolling(window=window, center=True).max()
            df_copy['low_rolling'] = df_copy['low'].rolling(window=window, center=True).min()
            
            # Identify pivot points
            pivot_highs = df_copy[df_copy['high'] == df_copy['high_rolling']]['high']
            pivot_lows = df_copy[df_copy['low'] == df_copy['low_rolling']]['low']
            
            current_price = df['close'].iloc[-1]
            
            # Find nearest levels
            resistance_levels = pivot_highs[pivot_highs > current_price].sort_values()
            support_levels = pivot_lows[pivot_lows < current_price].sort_values(ascending=False)
            
            # Calculate strength of levels based on touches
            def calculate_level_strength(level, df, tolerance=0.002):
                touches = 0
                for _, row in df.iterrows():
                    if abs(row['high'] - level) / level < tolerance or \
                       abs(row['low'] - level) / level < tolerance or \
                       abs(row['close'] - level) / level < tolerance:
                        touches += 1
                return touches
            
            result = {
                'current_price': current_price,
                'immediate_resistance': resistance_levels.iloc[0] if len(resistance_levels) > 0 else current_price * 1.05,
                'immediate_support': support_levels.iloc[0] if len(support_levels) > 0 else current_price * 0.95,
                'strong_resistance': resistance_levels.iloc[1] if len(resistance_levels) > 1 else current_price * 1.10,
                'strong_support': support_levels.iloc[1] if len(support_levels) > 1 else current_price * 0.90
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {}
    
    def smart_signal_generation(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Generate smart trading signals with enhanced validation"""
        try:
            if len(df) < 50:
                return None
            
            current_price = df['close'].iloc[-1]
            coin = symbol.replace('/USDT', '')
            
            # Check if we already have an active signal for this coin
            if self.database:
                active_signals = self.database.get_active_signals()
                active_coins = [s['coin'] for s in active_signals]
                if coin in active_coins:
                    logger.debug(f"Already have active signal for {coin}, skipping")
                    return None
            
            # Get analysis components
            patterns = self.detect_patterns(df)
            levels = self.calculate_support_resistance(df)
            
            # Signal scoring system
            long_score = 0
            short_score = 0
            confidence_factors = []
            
            # RSI analysis (weighted)
            rsi = df['rsi'].iloc[-1]
            rsi_prev = df['rsi'].iloc[-2]
            
            if rsi < 30:
                long_score += 3
                confidence_factors.append("RSI oversold")
            elif rsi < 35 and rsi > rsi_prev:
                long_score += 2
                confidence_factors.append("RSI recovering from oversold")
            elif rsi > 70:
                short_score += 3
                confidence_factors.append("RSI overbought")
            elif rsi > 65 and rsi < rsi_prev:
                short_score += 2
                confidence_factors.append("RSI declining from overbought")
            
            # MACD analysis (weighted)
            macd_hist = df['macd_hist'].iloc[-1]
            macd_hist_prev = df['macd_hist'].iloc[-2]
            macd_hist_prev2 = df['macd_hist'].iloc[-3]
            
            # Strong crossover signals
            if macd_hist > 0 and macd_hist_prev <= 0:
                long_score += 4
                confidence_factors.append("MACD bullish crossover")
            elif macd_hist < 0 and macd_hist_prev >= 0:
                short_score += 4
                confidence_factors.append("MACD bearish crossover")
            # Momentum confirmation
            elif macd_hist > macd_hist_prev > macd_hist_prev2:
                long_score += 2
                confidence_factors.append("MACD momentum increasing")
            elif macd_hist < macd_hist_prev < macd_hist_prev2:
                short_score += 2
                confidence_factors.append("MACD momentum decreasing")
            
            # Volume analysis
            volume_ratio = df['volume_ratio'].iloc[-1]
            if volume_ratio > 1.5:
                if long_score > short_score:
                    long_score += 2
                    confidence_factors.append("High volume confirmation (bullish)")
                elif short_score > long_score:
                    short_score += 2
                    confidence_factors.append("High volume confirmation (bearish)")
            
            # Bollinger Bands
            bb_position = df['bb_position'].iloc[-1]
            if bb_position < 0.1:
                long_score += 2
                confidence_factors.append("Near lower Bollinger Band")
            elif bb_position > 0.9:
                short_score += 2
                confidence_factors.append("Near upper Bollinger Band")
            
            # Support/Resistance analysis
            if levels:
                price_to_support = (current_price - levels.get('immediate_support', 0)) / current_price
                price_to_resistance = (levels.get('immediate_resistance', current_price) - current_price) / current_price
                
                if price_to_support < 0.015:  # Within 1.5% of support
                    long_score += 3
                    confidence_factors.append("Near strong support")
                if price_to_resistance < 0.015:  # Within 1.5% of resistance
                    short_score += 3
                    confidence_factors.append("Near strong resistance")
            
            # Pattern recognition bonus
            if patterns.get('divergence') == 'bullish':
                long_score += 3
                confidence_factors.append("Bullish divergence detected")
            elif patterns.get('divergence') == 'bearish':
                short_score += 3
                confidence_factors.append("Bearish divergence detected")
            
            # Trend alignment
            trend = patterns.get('trend', 'neutral')
            if trend == 'strong_bullish':
                long_score += 2
            elif trend == 'strong_bearish':
                short_score += 2
            elif trend == 'bullish':
                long_score += 1
            elif trend == 'bearish':
                short_score += 1
            
            # Stochastic confirmation
            stoch_k = df['stoch_k'].iloc[-1]
            stoch_d = df['stoch_d'].iloc[-1]
            
            if stoch_k < 20 and stoch_k > stoch_d:
                long_score += 1
                confidence_factors.append("Stochastic oversold with bullish cross")
            elif stoch_k > 80 and stoch_k < stoch_d:
                short_score += 1
                confidence_factors.append("Stochastic overbought with bearish cross")
            
            # Log analysis
            if self.database:
                self.database.log_analysis(
                    coin, '1h', 'SIGNAL_GENERATION',
                    f"Long: {long_score}, Short: {short_score}",
                    max(long_score, short_score) * 10,
                    {'rsi': rsi, 'macd_hist': macd_hist, 'bb_position': bb_position},
                    patterns, levels
                )
            
            # Generate signal if conditions are met
            if long_score >= self.min_confidence_score and long_score > short_score:
                signal = self.create_long_signal(df, symbol, long_score, confidence_factors, levels)
                if signal and self.validate_signal_params(signal):
                    logger.info(f"LONG signal generated for {coin}: Score {long_score}")
                    return signal
            elif short_score >= self.min_confidence_score and short_score > long_score:
                signal = self.create_short_signal(df, symbol, short_score, confidence_factors, levels)
                if signal and self.validate_signal_params(signal):
                    logger.info(f"SHORT signal generated for {coin}: Score {short_score}")
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def validate_signal_params(self, signal: Dict) -> bool:
        """Validate signal parameters before returning"""
        try:
            direction = signal['direction']
            entry = signal['entry_price']
            tp = signal['take_profit']
            sl = signal['stop_loss']
            current = signal['current_price']
            
            # Check basic logic
            if direction == 'LONG':
                # For LONG: SL < Entry < Current < TP
                if not (sl < entry <= current < tp):
                    logger.warning(f"Invalid LONG signal params: SL={sl:.6f}, Entry={entry:.6f}, Current={current:.6f}, TP={tp:.6f}")
                    return False
            else:  # SHORT
                # For SHORT: TP < Current <= Entry < SL
                if not (tp < current <= entry < sl):
                    logger.warning(f"Invalid SHORT signal params: TP={tp:.6f}, Current={current:.6f}, Entry={entry:.6f}, SL={sl:.6f}")
                    return False
            
            # Check risk-reward ratio
            rr_ratio = signal['analysis_data'].get('risk_reward_ratio', 0)
            if rr_ratio < self.min_risk_reward_ratio:
                logger.warning(f"Risk-reward ratio too low: {rr_ratio:.2f}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    
    def create_long_signal(self, df: pd.DataFrame, symbol: str, score: int, factors: List[str], levels: Dict) -> Dict:
        """Create a LONG signal with validated parameters"""
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        atr_percentage = df['atr_percentage'].iloc[-1]
        
        # Entry price: Current price (we enter at market)
        entry_price = current_price
        
        # Stop loss calculation
        # Use ATR-based stop with support level validation
        atr_multiplier = 2.0 if atr_percentage > 3 else 1.5  # Adjust for volatility
        atr_stop = entry_price - (atr * atr_multiplier)
        
        # Use support level if available and reasonable
        if levels.get('immediate_support'):
            support_stop = levels['immediate_support'] * 0.995
            # Use the higher of the two (closer to entry) for better risk management
            stop_loss = max(atr_stop, support_stop)
        else:
            stop_loss = atr_stop
        
        # Ensure minimum stop distance (0.5%)
        min_stop_distance = entry_price * 0.005
        if entry_price - stop_loss < min_stop_distance:
            stop_loss = entry_price - min_stop_distance
        
        # Take profit calculation
        risk = entry_price - stop_loss
        base_rr_ratio = 2.5
        
        # Adjust R:R based on confidence
        if score >= 8:
            rr_ratio = 3.0
        elif score >= 6:
            rr_ratio = 2.5
        else:
            rr_ratio = 2.0
        
        take_profit = entry_price + (risk * rr_ratio)
        
        # Validate against resistance
        if levels.get('immediate_resistance'):
            resistance_limit = levels['immediate_resistance'] * 0.995
            if take_profit > resistance_limit:
                # Adjust TP to just below resistance
                take_profit = resistance_limit
                # Recalculate R:R
                new_rr = (take_profit - entry_price) / risk
                if new_rr < self.min_risk_reward_ratio:
                    # If R:R becomes too low, widen the stop
                    stop_loss = entry_price - ((take_profit - entry_price) / self.min_risk_reward_ratio)
        
        # Ensure TP is at least 1% above entry
        min_tp_distance = entry_price * 1.01
        if take_profit < min_tp_distance:
            take_profit = min_tp_distance
        
        signal_id = f"{symbol.replace('/', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
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
                'atr': round(atr, 6),
                'atr_percentage': round(atr_percentage, 2),
                'risk_reward_ratio': round((take_profit - entry_price) / (entry_price - stop_loss), 2),
                'risk_percentage': round(((entry_price - stop_loss) / entry_price) * 100, 2),
                'profit_percentage': round(((take_profit - entry_price) / entry_price) * 100, 2),
                'support_level': levels.get('immediate_support'),
                'resistance_level': levels.get('immediate_resistance'),
                'score': score
            },
            'indicators': {
                'rsi': round(df['rsi'].iloc[-1], 2),
                'macd': round(df['macd'].iloc[-1], 6),
                'macd_hist': round(df['macd_hist'].iloc[-1], 6),
                'bb_position': round(df['bb_position'].iloc[-1], 3),
                'volume_ratio': round(df['volume_ratio'].iloc[-1], 2),
                'stoch_k': round(df['stoch_k'].iloc[-1], 2),
                'atr': round(atr, 6)
            }
        }
    
    def create_short_signal(self, df: pd.DataFrame, symbol: str, score: int, factors: List[str], levels: Dict) -> Dict:
        """Create a SHORT signal with validated parameters"""
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        atr_percentage = df['atr_percentage'].iloc[-1]
        
        # Entry price: Current price (we enter at market)
        entry_price = current_price
        
        # Stop loss calculation
        atr_multiplier = 2.0 if atr_percentage > 3 else 1.5
        atr_stop = entry_price + (atr * atr_multiplier)
        
        # Use resistance level if available
        if levels.get('immediate_resistance'):
            resistance_stop = levels['immediate_resistance'] * 1.005
            # Use the lower of the two (closer to entry)
            stop_loss = min(atr_stop, resistance_stop)
        else:
            stop_loss = atr_stop
        
        # Ensure minimum stop distance
        min_stop_distance = entry_price * 0.005
        if stop_loss - entry_price < min_stop_distance:
            stop_loss = entry_price + min_stop_distance
        
        # Take profit calculation
        risk = stop_loss - entry_price
        
        # Adjust R:R based on confidence
        if score >= 8:
            rr_ratio = 3.0
        elif score >= 6:
            rr_ratio = 2.5
        else:
            rr_ratio = 2.0
        
        take_profit = entry_price - (risk * rr_ratio)
        
        # Validate against support
        if levels.get('immediate_support'):
            support_limit = levels['immediate_support'] * 1.005
            if take_profit < support_limit:
                take_profit = support_limit
                # Recalculate R:R
                new_rr = (entry_price - take_profit) / risk
                if new_rr < self.min_risk_reward_ratio:
                    stop_loss = entry_price + ((entry_price - take_profit) / self.min_risk_reward_ratio)
        
        # Ensure TP is at least 1% below entry
        min_tp_distance = entry_price * 0.99
        if take_profit > min_tp_distance:
            take_profit = min_tp_distance
        
        signal_id = f"{symbol.replace('/', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
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
                'atr': round(atr, 6),
                'atr_percentage': round(atr_percentage, 2),
                'risk_reward_ratio': round((entry_price - take_profit) / (stop_loss - entry_price), 2),
                'risk_percentage': round(((stop_loss - entry_price) / entry_price) * 100, 2),
                'profit_percentage': round(((entry_price - take_profit) / entry_price) * 100, 2),
                'support_level': levels.get('immediate_support'),
                'resistance_level': levels.get('immediate_resistance'),
                'score': score
            },
            'indicators': {
                'rsi': round(df['rsi'].iloc[-1], 2),
                'macd': round(df['macd'].iloc[-1], 6),
                'macd_hist': round(df['macd_hist'].iloc[-1], 6),
                'bb_position': round(df['bb_position'].iloc[-1], 3),
                'volume_ratio': round(df['volume_ratio'].iloc[-1], 2),
                'stoch_k': round(df['stoch_k'].iloc[-1], 2),
                'atr': round(atr, 6)
            }
        }
    
    async def scan_all_coins(self) -> List[Dict]:
        """Scan all coins for trading opportunities"""
        signals = []
        
        for i, symbol in enumerate(self.coins):
            try:
                logger.debug(f"Scanning {symbol} ({i+1}/{len(self.coins)})")
                
                # Fetch market data
                df = await self.get_market_data(symbol, '1h', 200)
                
                if df.empty or len(df) < 50:
                    continue
                
                # Calculate indicators
                df = self.calculate_technical_indicators(df)
                
                if len(df) < 50:
                    continue
                
                # Generate signal
                signal = self.smart_signal_generation(df, symbol)
                
                if signal:
                    signals.append(signal)
                    logger.info(f"Signal generated for {symbol}: {signal['direction']}")
                
                # Small delay between requests
                await asyncio.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        logger.info(f"Scan complete: {len(signals)} signals generated from {len(self.coins)} coins")
        return signals
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            ticker = self.exchange.fetch_ticker(f"{symbol}/USDT")
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return 0.0