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
        # Load environment variables from .env file
        load_dotenv()
        
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY'),
            'sandbox': False,
            'rateLimit': 1200,
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
        
        # Log initialization
        if self.database:
            self.database.log_bot_activity(
                'INFO', 'ANALYZER', 'Trading analyzer initialized',
                f'Monitoring {len(self.coins)} coins'
            )
    
    async def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> pd.DataFrame:
        """Fetch market data for a symbol"""
        try:
            if self.database:
                self.database.log_bot_activity(
                    'DEBUG', 'DATA_FETCHER', f'Fetching {timeframe} data for {symbol}',
                    f'Limit: {limit} candles'
                )
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            if self.database:
                self.database.log_bot_activity(
                    'DEBUG', 'DATA_FETCHER', f'Successfully fetched {len(df)} candles for {symbol}',
                    f'Latest price: ${df["close"].iloc[-1]:.6f}'
                )
            
            return df
        except Exception as e:
            if self.database:
                self.database.log_bot_activity(
                    'ERROR', 'DATA_FETCHER', f'Failed to fetch data for {symbol}',
                    str(e), symbol
                )
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators without TA-Lib"""
        if df.empty or len(df) < 50:
            return df
        
        try:
            if self.database:
                self.database.log_bot_activity(
                    'DEBUG', 'INDICATORS', 'Calculating technical indicators',
                    f'Processing {len(df)} data points'
                )
            
            # Simple Moving Averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Exponential Moving Averages
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # Stochastic Oscillator
            lowest_low = df['low'].rolling(window=14).min()
            highest_high = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # Williams %R
            df['williams_r'] = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
            
            # Momentum
            df['momentum'] = df['close'] / df['close'].shift(10) - 1
            df['roc'] = df['close'].pct_change(periods=10) * 100
            
            # Volatility
            df['volatility'] = df['close'].rolling(20).std()
            
            # Support and Resistance levels
            df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
            df['resistance1'] = 2 * df['pivot'] - df['low']
            df['support1'] = 2 * df['pivot'] - df['high']
            
            if self.database:
                indicators_summary = {
                    'rsi': df['rsi'].iloc[-1],
                    'macd_hist': df['macd_hist'].iloc[-1],
                    'bb_position': df['bb_position'].iloc[-1],
                    'volume_ratio': df['volume_ratio'].iloc[-1]
                }
                self.database.log_bot_activity(
                    'DEBUG', 'INDICATORS', 'Technical indicators calculated',
                    'All indicators computed successfully',
                    data=indicators_summary
                )
            
            return df.dropna()
        except Exception as e:
            if self.database:
                self.database.log_bot_activity(
                    'ERROR', 'INDICATORS', 'Failed to calculate indicators',
                    str(e)
                )
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Analyze volume profile for support/resistance levels"""
        try:
            if len(df) < 50:
                return {}
            
            if self.database:
                self.database.log_bot_activity(
                    'DEBUG', 'VOLUME_ANALYZER', 'Analyzing volume profile',
                    f'Processing {len(df)} price levels'
                )
            
            # Create price bins
            price_min, price_max = df['low'].min(), df['high'].max()
            num_bins = 50
            bins = np.linspace(price_min, price_max, num_bins)
            
            # Calculate volume at each price level
            volume_profile = {}
            for i in range(len(bins) - 1):
                mask = (df['low'] <= bins[i+1]) & (df['high'] >= bins[i])
                volume_profile[bins[i]] = df[mask]['volume'].sum()
            
            # Find VPOC (Volume Point of Control)
            if volume_profile:
                vpoc_price = max(volume_profile, key=volume_profile.get)
                vpoc_volume = volume_profile[vpoc_price]
                
                # Find high volume nodes
                sorted_volumes = sorted(volume_profile.values(), reverse=True)
                high_volume_threshold = sorted_volumes[min(5, len(sorted_volumes)-1)]
                
                hvn_levels = [price for price, volume in volume_profile.items() 
                             if volume >= high_volume_threshold]
                
                result = {
                    'vpoc_price': vpoc_price,
                    'vpoc_volume': vpoc_volume,
                    'high_volume_nodes': hvn_levels,
                    'volume_profile': volume_profile
                }
                
                if self.database:
                    self.database.log_bot_activity(
                        'DEBUG', 'VOLUME_ANALYZER', 'Volume profile analysis completed',
                        f'VPOC: ${vpoc_price:.6f}, HVN levels: {len(hvn_levels)}'
                    )
                
                return result
            return {}
        except Exception as e:
            if self.database:
                self.database.log_bot_activity(
                    'ERROR', 'VOLUME_ANALYZER', 'Volume profile analysis failed',
                    str(e)
                )
            logger.error(f"Error in volume profile analysis: {e}")
            return {}
    
    def detect_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect chart patterns"""
        try:
            if len(df) < 20:
                return {}
            
            if self.database:
                self.database.log_bot_activity(
                    'DEBUG', 'PATTERN_DETECTOR', 'Analyzing chart patterns',
                    f'Scanning {len(df)} candles for patterns'
                )
            
            patterns = {}
            
            # Trend patterns
            recent_highs = df['high'].rolling(5).max()
            recent_lows = df['low'].rolling(5).min()
            
            # Higher highs and higher lows (uptrend)
            higher_highs = (recent_highs > recent_highs.shift(5)).sum()
            higher_lows = (recent_lows > recent_lows.shift(5)).sum()
            
            if higher_highs >= 3 and higher_lows >= 3:
                patterns['uptrend'] = True
            elif higher_highs <= 1 and higher_lows <= 1:
                patterns['downtrend'] = True
            else:
                patterns['sideways'] = True
            
            # Divergence detection
            if len(df) >= 20:
                price_trend = df['close'].iloc[-1] - df['close'].iloc[-20]
                rsi_trend = df['rsi'].iloc[-1] - df['rsi'].iloc[-20]
                
                if price_trend > 0 and rsi_trend < 0:
                    patterns['bearish_divergence'] = True
                elif price_trend < 0 and rsi_trend > 0:
                    patterns['bullish_divergence'] = True
            
            if self.database:
                pattern_summary = []
                for pattern, value in patterns.items():
                    if value:
                        pattern_summary.append(pattern)
                
                self.database.log_bot_activity(
                    'DEBUG', 'PATTERN_DETECTOR', 'Pattern detection completed',
                    f'Detected patterns: {", ".join(pattern_summary) if pattern_summary else "None"}'
                )
            
            return patterns
        except Exception as e:
            if self.database:
                self.database.log_bot_activity(
                    'ERROR', 'PATTERN_DETECTOR', 'Pattern detection failed',
                    str(e)
                )
            logger.error(f"Error detecting patterns: {e}")
            return {}
    
    def calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Calculate dynamic support and resistance levels"""
        try:
            if len(df) < 20:
                return {}
            
            if self.database:
                self.database.log_bot_activity(
                    'DEBUG', 'SUPPORT_RESISTANCE', 'Calculating S/R levels',
                    'Finding pivot points and key levels'
                )
            
            # Find pivot points
            highs = df['high'].rolling(window=5, center=True).max()
            lows = df['low'].rolling(window=5, center=True).min()
            
            pivot_highs = df[df['high'] == highs]['high'].dropna()
            pivot_lows = df[df['low'] == lows]['low'].dropna()
            
            current_price = df['close'].iloc[-1]
            
            # Find nearest support and resistance
            resistance_levels = pivot_highs[pivot_highs > current_price].sort_values()
            support_levels = pivot_lows[pivot_lows < current_price].sort_values(ascending=False)
            
            result = {
                'immediate_resistance': resistance_levels.iloc[0] if len(resistance_levels) > 0 else None,
                'immediate_support': support_levels.iloc[0] if len(support_levels) > 0 else None,
                'all_resistance': resistance_levels.head(3).tolist(),
                'all_support': support_levels.head(3).tolist()
            }
            
            if self.database:
                support_str = f"${result['immediate_support']:.6f}" if result['immediate_support'] else "None"
                resistance_str = f"${result['immediate_resistance']:.6f}" if result['immediate_resistance'] else "None"
                
                self.database.log_bot_activity(
                    'DEBUG', 'SUPPORT_RESISTANCE', 'S/R levels calculated',
                    f'Support: {support_str}, Resistance: {resistance_str}'
                )
            
            return result
        except Exception as e:
            if self.database:
                self.database.log_bot_activity(
                    'ERROR', 'SUPPORT_RESISTANCE', 'S/R calculation failed',
                    str(e)
                )
            logger.error(f"Error calculating support/resistance: {e}")
            return {}
    
    def smart_signal_generation(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Generate smart trading signals using multiple confirmations"""
        try:
            if len(df) < 50:
                return None
            
            current_price = df['close'].iloc[-1]
            coin = symbol.replace('/USDT', '')
            
            if self.database:
                self.database.log_bot_activity(
                    'INFO', 'SIGNAL_ANALYZER', f'Analyzing {coin} for signals',
                    f'Current price: ${current_price:.6f}', coin
                )
            
            # Get analysis components
            volume_analysis = self.analyze_volume_profile(df)
            patterns = self.detect_patterns(df)
            levels = self.calculate_support_resistance(df)
            
            # Signal scoring system
            long_score = 0
            short_score = 0
            confidence_factors = []
            
            # RSI analysis
            rsi = df['rsi'].iloc[-1]
            if rsi < 30:
                long_score += 2
                confidence_factors.append("RSI oversold (bullish)")
            elif rsi > 70:
                short_score += 2
                confidence_factors.append("RSI overbought (bearish)")
            
            # MACD analysis
            macd_hist = df['macd_hist'].iloc[-1]
            macd_hist_prev = df['macd_hist'].iloc[-2]
            if macd_hist > 0 and macd_hist_prev <= 0:
                long_score += 3
                confidence_factors.append("MACD bullish crossover")
            elif macd_hist < 0 and macd_hist_prev >= 0:
                short_score += 3
                confidence_factors.append("MACD bearish crossover")
            
            # Volume confirmation
            volume_ratio = df['volume_ratio'].iloc[-1]
            if volume_ratio > 1.5:
                confidence_factors.append("High volume confirmation")
                if long_score > short_score:
                    long_score += 1
                else:
                    short_score += 1
            
            # Bollinger Bands
            bb_position = df['bb_position'].iloc[-1]
            if bb_position < 0.2:
                long_score += 1
                confidence_factors.append("Price near lower Bollinger Band")
            elif bb_position > 0.8:
                short_score += 1
                confidence_factors.append("Price near upper Bollinger Band")
            
            # Support/Resistance confirmation
            if levels.get('immediate_support'):
                support_distance = abs(current_price - levels['immediate_support']) / current_price
                if support_distance < 0.02:  # Within 2% of support
                    long_score += 2
                    confidence_factors.append("Price near strong support")
            
            if levels.get('immediate_resistance'):
                resistance_distance = abs(current_price - levels['immediate_resistance']) / current_price
                if resistance_distance < 0.02:  # Within 2% of resistance
                    short_score += 2
                    confidence_factors.append("Price near strong resistance")
            
            # Pattern recognition bonus
            if patterns.get('bullish_divergence'):
                long_score += 2
                confidence_factors.append("Bullish divergence detected")
            if patterns.get('bearish_divergence'):
                short_score += 2
                confidence_factors.append("Bearish divergence detected")
            
            # Trend confirmation
            if patterns.get('uptrend'):
                long_score += 1
            elif patterns.get('downtrend'):
                short_score += 1
            
            # Log analysis results
            if self.database:
                analysis_result = f"Long score: {long_score}, Short score: {short_score}"
                self.database.log_analysis(
                    coin, '1h', 'SIGNAL_GENERATION', analysis_result,
                    max(long_score, short_score) * 10,
                    {
                        'rsi': df['rsi'].iloc[-1],
                        'macd_hist': df['macd_hist'].iloc[-1],
                        'bb_position': df['bb_position'].iloc[-1],
                        'volume_ratio': df['volume_ratio'].iloc[-1]
                    },
                    patterns,
                    levels
                )
                
                self.database.log_bot_activity(
                    'INFO', 'SIGNAL_ANALYZER', f'{coin} analysis completed',
                    f'Long: {long_score}, Short: {short_score}, Factors: {len(confidence_factors)}',
                    coin
                )
            
            # Generate signal if confidence is high enough
            min_score = 4  # Minimum score to generate signal
            
            if long_score >= min_score and long_score > short_score:
                signal = self.create_long_signal(df, symbol, long_score, confidence_factors, levels)
                if self.database:
                    self.database.log_bot_activity(
                        'INFO', 'SIGNAL_GENERATOR', f'LONG signal generated for {coin}',
                        f'Confidence: {signal["confidence"]}%, Entry: ${signal["entry_price"]}',
                        coin
                    )
                return signal
            elif short_score >= min_score and short_score > long_score:
                signal = self.create_short_signal(df, symbol, short_score, confidence_factors, levels)
                if self.database:
                    self.database.log_bot_activity(
                        'INFO', 'SIGNAL_GENERATOR', f'SHORT signal generated for {coin}',
                        f'Confidence: {signal["confidence"]}%, Entry: ${signal["entry_price"]}',
                        coin
                    )
                return signal
            else:
                if self.database:
                    self.database.log_bot_activity(
                        'DEBUG', 'SIGNAL_ANALYZER', f'No signal for {coin}',
                        f'Insufficient confidence (min: {min_score})', coin
                    )
            
            return None
            
        except Exception as e:
            if self.database:
                self.database.log_bot_activity(
                    'ERROR', 'SIGNAL_ANALYZER', f'Signal generation failed for {symbol}',
                    str(e), symbol.replace('/USDT', '')
                )
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def create_long_signal(self, df: pd.DataFrame, symbol: str, score: int, factors: List[str], levels: Dict) -> Dict:
        """Create a LONG signal with proper entry, TP, and SL"""
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        # Entry price (current price with small buffer)
        entry_price = current_price * 0.999
        
        # Stop loss based on ATR and support levels
        if levels.get('immediate_support'):
            stop_loss = min(levels['immediate_support'] * 0.995, entry_price - (atr * 2))
        else:
            stop_loss = entry_price - (atr * 2.5)
        
        # Take profit based on risk-reward ratio
        risk = entry_price - stop_loss
        take_profit = entry_price + (risk * 2.5)  # 2.5:1 risk-reward ratio
        
        # Adjust TP if resistance level is closer
        if levels.get('immediate_resistance') and levels['immediate_resistance'] < take_profit:
            take_profit = levels['immediate_resistance'] * 0.995
        
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
                'risk_reward_ratio': round((take_profit - entry_price) / (entry_price - stop_loss), 2),
                'support_level': levels.get('immediate_support'),
                'resistance_level': levels.get('immediate_resistance'),
                'volume_analysis': f"Volume ratio: {df['volume_ratio'].iloc[-1]:.2f}",
                'technical_summary': f"RSI: {df['rsi'].iloc[-1]:.1f}, MACD trending positive"
            },
            'indicators': {
                'rsi': round(df['rsi'].iloc[-1], 2),
                'macd': round(df['macd'].iloc[-1], 6),
                'bb_position': round(df['bb_position'].iloc[-1], 3),
                'volume_ratio': round(df['volume_ratio'].iloc[-1], 2),
                'atr': round(atr, 6)
            }
        }
    
    def create_short_signal(self, df: pd.DataFrame, symbol: str, score: int, factors: List[str], levels: Dict) -> Dict:
        """Create a SHORT signal with proper entry, TP, and SL"""
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        # Entry price (current price with small buffer)
        entry_price = current_price * 1.001
        
        # Stop loss based on ATR and resistance levels
        if levels.get('immediate_resistance'):
            stop_loss = max(levels['immediate_resistance'] * 1.005, entry_price + (atr * 2))
        else:
            stop_loss = entry_price + (atr * 2.5)
        
        # Take profit based on risk-reward ratio
        risk = stop_loss - entry_price
        take_profit = entry_price - (risk * 2.5)  # 2.5:1 risk-reward ratio
        
        # Adjust TP if support level is closer
        if levels.get('immediate_support') and levels['immediate_support'] > take_profit:
            take_profit = levels['immediate_support'] * 1.005
        
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
                'risk_reward_ratio': round((entry_price - take_profit) / (stop_loss - entry_price), 2),
                'support_level': levels.get('immediate_support'),
                'resistance_level': levels.get('immediate_resistance'),
                'volume_analysis': f"Volume ratio: {df['volume_ratio'].iloc[-1]:.2f}",
                'technical_summary': f"RSI: {df['rsi'].iloc[-1]:.1f}, MACD trending negative"
            },
            'indicators': {
                'rsi': round(df['rsi'].iloc[-1], 2),
                'macd': round(df['macd'].iloc[-1], 6),
                'bb_position': round(df['bb_position'].iloc[-1], 3),
                'volume_ratio': round(df['volume_ratio'].iloc[-1], 2),
                'atr': round(atr, 6)
            }
        }
    
    async def scan_all_coins(self) -> List[Dict]:
        """Scan all coins for trading opportunities"""
        signals = []
        
        if self.database:
            self.database.log_bot_activity(
                'INFO', 'MARKET_SCANNER', f'Starting market scan',
                f'Scanning {len(self.coins)} coins for opportunities'
            )
        
        for i, symbol in enumerate(self.coins):
            try:
                if self.database:
                    self.database.log_bot_activity(
                        'DEBUG', 'MARKET_SCANNER', f'Scanning {symbol} ({i+1}/{len(self.coins)})',
                        'Fetching market data and analyzing'
                    )
                
                # Fetch data for multiple timeframes
                df_1h = await self.get_market_data(symbol, '1h', 200)
                df_4h = await self.get_market_data(symbol, '4h', 100)
                
                if df_1h.empty or df_4h.empty:
                    if self.database:
                        self.database.log_bot_activity(
                            'WARNING', 'MARKET_SCANNER', f'No data available for {symbol}',
                            'Skipping to next coin'
                        )
                    continue
                
                # Calculate indicators
                df_1h = self.calculate_technical_indicators(df_1h)
                df_4h = self.calculate_technical_indicators(df_4h)
                
                if len(df_1h) < 50:
                    if self.database:
                        self.database.log_bot_activity(
                            'WARNING', 'MARKET_SCANNER', f'Insufficient data for {symbol}',
                            f'Only {len(df_1h)} candles available'
                        )
                    continue
                
                # Generate signal (using 1h timeframe for entries, 4h for trend confirmation)
                signal = self.smart_signal_generation(df_1h, symbol)
                
                if signal:
                    # Confirm with 4h timeframe
                    h4_trend = self.get_trend_confirmation(df_4h)
                    if self.is_signal_confirmed(signal, h4_trend):
                        signals.append(signal)
                        if self.database:
                            self.database.log_bot_activity(
                                'INFO', 'MARKET_SCANNER', f'Signal confirmed for {symbol}',
                                f'{signal["direction"]} with {signal["confidence"]}% confidence'
                            )
                    else:
                        if self.database:
                            self.database.log_bot_activity(
                                'INFO', 'MARKET_SCANNER', f'Signal rejected for {symbol}',
                                f'{signal["direction"]} signal conflicts with {h4_trend} H4 trend'
                            )
                
                # Small delay to respect rate limits
                await asyncio.sleep(0.5)
                
            except Exception as e:
                if self.database:
                    self.database.log_bot_activity(
                        'ERROR', 'MARKET_SCANNER', f'Error scanning {symbol}',
                        str(e), symbol.replace('/USDT', '')
                    )
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        if self.database:
            self.database.log_bot_activity(
                'INFO', 'MARKET_SCANNER', 'Market scan completed',
                f'Generated {len(signals)} signals from {len(self.coins)} coins scanned'
            )
        
        return signals
    
    def get_trend_confirmation(self, df: pd.DataFrame) -> str:
        """Get trend confirmation from higher timeframe"""
        if len(df) < 20:
            return "neutral"
        
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            return "bullish"
        elif current_price < sma_20 < sma_50:
            return "bearish"
        else:
            return "neutral"
    
    def is_signal_confirmed(self, signal: Dict, h4_trend: str) -> bool:
        """Check if signal is confirmed by higher timeframe"""
        if signal['direction'] == 'LONG' and h4_trend == 'bearish':
            return False
        if signal['direction'] == 'SHORT' and h4_trend == 'bullish':
            return False
        return True
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            ticker = self.exchange.fetch_ticker(f"{symbol}/USDT")
            return ticker['last']
        except Exception as e:
            if self.database:
                self.database.log_bot_activity(
                    'ERROR', 'PRICE_FETCHER', f'Failed to get price for {symbol}',
                    str(e), symbol
                )
            logger.error(f"Error fetching price for {symbol}: {e}")
            return 0.0