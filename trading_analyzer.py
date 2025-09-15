def smart_signal_generation(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
    """Generate smart trading signals using multiple confirmations"""
    try:
        if len(df) < 50:
            return None
        
        current_price = df['close'].iloc[-1]
        coin = symbol.replace('/USDT', '')
        
        # Get analysis components
        volume_analysis = self.analyze_volume_profile(df)
        patterns = self.detect_patterns(df)
        levels = self.calculate_support_resistance(df)
        
        # Signal scoring system - MORE STRICT
        long_score = 0
        short_score = 0
        confidence_factors = []
        
        # RSI analysis - MORE STRICT ZONES
        rsi = df['rsi'].iloc[-1]
        if rsi < 25:  # Changed from 30 to 25 for stronger oversold
            long_score += 3
            confidence_factors.append("RSI strongly oversold")
        elif rsi < 35:
            long_score += 1
            confidence_factors.append("RSI oversold")
        elif rsi > 75:  # Changed from 70 to 75 for stronger overbought
            short_score += 3
            confidence_factors.append("RSI strongly overbought")
        elif rsi > 65:
            short_score += 1
            confidence_factors.append("RSI overbought")
        
        # MACD analysis with stronger confirmation
        macd_hist = df['macd_hist'].iloc[-1]
        macd_hist_prev = df['macd_hist'].iloc[-2]
        macd_hist_prev2 = df['macd_hist'].iloc[-3]
        
        # Check for confirmed crossover (not just single bar)
        if macd_hist > 0 and macd_hist_prev <= 0 and macd_hist > macd_hist_prev:
            long_score += 3
            confidence_factors.append("MACD bullish crossover confirmed")
        elif macd_hist < 0 and macd_hist_prev >= 0 and macd_hist < macd_hist_prev:
            short_score += 3
            confidence_factors.append("MACD bearish crossover confirmed")
        
        # Volume confirmation - REQUIRE HIGHER VOLUME
        volume_ratio = df['volume_ratio'].iloc[-1]
        if volume_ratio > 2.0:  # Changed from 1.5 to 2.0
            confidence_factors.append("Very high volume confirmation")
            if long_score > short_score:
                long_score += 2
            else:
                short_score += 2
        elif volume_ratio > 1.5:
            confidence_factors.append("High volume")
            if long_score > short_score:
                long_score += 1
            else:
                short_score += 1
        
        # Bollinger Bands - MORE EXTREME POSITIONS
        bb_position = df['bb_position'].iloc[-1]
        if bb_position < 0.1:  # Changed from 0.2 to 0.1
            long_score += 2
            confidence_factors.append("Price at lower Bollinger Band extreme")
        elif bb_position > 0.9:  # Changed from 0.8 to 0.9
            short_score += 2
            confidence_factors.append("Price at upper Bollinger Band extreme")
        
        # Support/Resistance - TIGHTER ZONES
        if levels.get('immediate_support'):
            support_distance = abs(current_price - levels['immediate_support']) / current_price
            if support_distance < 0.01:  # Changed from 0.02 to 0.01 (within 1%)
                long_score += 3
                confidence_factors.append("Price at strong support")
        
        if levels.get('immediate_resistance'):
            resistance_distance = abs(current_price - levels['immediate_resistance']) / current_price
            if resistance_distance < 0.01:  # Changed from 0.02 to 0.01
                short_score += 3
                confidence_factors.append("Price at strong resistance")
        
        # Pattern recognition bonus
        if patterns.get('bullish_divergence'):
            long_score += 2
            confidence_factors.append("Bullish divergence detected")
        if patterns.get('bearish_divergence'):
            short_score += 2
            confidence_factors.append("Bearish divergence detected")
        
        # Trend confirmation - STRONGER WEIGHT
        if patterns.get('uptrend'):
            long_score += 2  # Changed from 1 to 2
        elif patterns.get('downtrend'):
            short_score += 2  # Changed from 1 to 2
        
        # Additional filters to reduce false signals
        # Check momentum
        momentum = df['momentum'].iloc[-1]
        if abs(momentum) < 0.01:  # Avoid signals in low momentum periods
            long_score -= 1
            short_score -= 1
            
        # Check ATR for volatility
        atr_ratio = df['atr'].iloc[-1] / current_price
        if atr_ratio < 0.005:  # Low volatility, reduce scores
            long_score -= 1
            short_score -= 1
        
        # INCREASED MINIMUM SCORE REQUIREMENT
        min_score = 7  # Changed from 4 to 7 - MUCH STRICTER
        
        # Also require minimum confidence factors
        min_factors = 3  # Require at least 3 confidence factors
        
        if long_score >= min_score and long_score > short_score and len(confidence_factors) >= min_factors:
            signal = self.create_long_signal(df, symbol, long_score, confidence_factors, levels)
            return signal
        elif short_score >= min_score and short_score > long_score and len(confidence_factors) >= min_factors:
            signal = self.create_short_signal(df, symbol, short_score, confidence_factors, levels)
            return signal
        
        return None
        
    except Exception as e:
        logger.error(f"Error generating signal for {symbol}: {e}")
        return None

def create_long_signal(self, df: pd.DataFrame, symbol: str, score: int, factors: List[str], levels: Dict) -> Dict:
    """Create a LONG signal with PROPER entry, TP, and SL"""
    current_price = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1]
    
    # Entry price - wait for slight pullback
    entry_price = current_price * 0.998  # Changed from 0.999
    
    # Stop loss - MORE CONSERVATIVE
    if levels.get('immediate_support'):
        # Use support but with more room
        stop_loss = min(levels['immediate_support'] * 0.99, entry_price - (atr * 1.5))
    else:
        stop_loss = entry_price - (atr * 2.0)  # Changed from 2.5 to 2.0
    
    # Take profit - MORE REALISTIC
    risk = entry_price - stop_loss
    
    # Dynamic risk-reward based on market conditions
    if df['volatility'].iloc[-1] > df['volatility'].mean():
        # High volatility = larger targets
        risk_reward = 3.0
    else:
        # Normal volatility = standard targets
        risk_reward = 2.0
    
    take_profit = entry_price + (risk * risk_reward)
    
    # Check if resistance is closer and adjust
    if levels.get('immediate_resistance'):
        # Set TP just below resistance
        resistance_tp = levels['immediate_resistance'] * 0.995
        if resistance_tp < take_profit and resistance_tp > entry_price * 1.01:
            # Only use resistance if it gives at least 1% profit
            take_profit = resistance_tp
    
    # Ensure minimum profit target (at least 1% with 10x leverage = 10%)
    min_profit_target = entry_price * 1.01
    if take_profit < min_profit_target:
        take_profit = min_profit_target
    
    signal_id = f"{symbol.replace('/', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Calculate more accurate confidence
    base_confidence = min(95, score * 8)  # Changed from *10 to *8
    
    # Adjust confidence based on factors
    if len(factors) >= 5:
        base_confidence = min(95, base_confidence + 5)
    
    return {
        'signal_id': signal_id,
        'timestamp': datetime.now().isoformat(),
        'coin': symbol.replace('/USDT', ''),
        'direction': 'LONG',
        'entry_price': round(entry_price, 6),
        'current_price': round(current_price, 6),
        'take_profit': round(take_profit, 6),
        'stop_loss': round(stop_loss, 6),
        'confidence': base_confidence,
        'analysis_data': {
            'reasons': factors,
            'atr': round(atr, 6),
            'risk_reward_ratio': round((take_profit - entry_price) / (entry_price - stop_loss), 2),
            'support_level': levels.get('immediate_support'),
            'resistance_level': levels.get('immediate_resistance'),
            'volume_analysis': f"Volume ratio: {df['volume_ratio'].iloc[-1]:.2f}",
            'technical_summary': f"RSI: {df['rsi'].iloc[-1]:.1f}, MACD: {df['macd_hist'].iloc[-1]:.6f}"
        },
        'indicators': {
            'rsi': round(df['rsi'].iloc[-1], 2),
            'macd': round(df['macd'].iloc[-1], 6),
            'macd_hist': round(df['macd_hist'].iloc[-1], 6),
            'bb_position': round(df['bb_position'].iloc[-1], 3),
            'volume_ratio': round(df['volume_ratio'].iloc[-1], 2),
            'atr': round(atr, 6),
            'momentum': round(df['momentum'].iloc[-1], 4)
        }
    }