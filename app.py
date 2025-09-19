#!/usr/bin/env python3
"""
ML-Enhanced Crypto Trading Bot
Feature Engineering • Orthogonalization • Time Series Validation
"""

import asyncio
import json
import logging
import os
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import signal
import sys
import numpy as np
import math
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Import our ML enhanced modules
from database import Database
from enhanced_trading_analyzer_v2 import MLTradingAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MLTradingBot:
    def __init__(self):
        # Core ML components
        self.database = Database()
        self.analyzer = MLTradingAnalyzer(database=self.database)
        self.enable_dynamic_exits = True
        self.min_health_score_for_exit = 0.3
        self.position_check_interval = 180  # seconds
        
        # FastAPI setup
        self.app = FastAPI(
            title="ML-Enhanced Crypto Trading Bot",
            version="3.0.0",
            description="Advanced ML Trading Bot with Feature Engineering and Time Series Validation"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # WebSocket connections
        self.websocket_connections: Set[WebSocket] = set()
        
        # Bot state
        self.running = False
        self.is_scanning = False
        self.is_training = False
        self.scan_interval = 600  # 10 minutes for ML processing
        self.retrain_interval = 3600  # 1 hour for model retraining
        self.last_scan_time = None
        self.last_train_time = None
        self.scan_count = 0
        self.startup_time = datetime.now()
        
        # ML-specific tracking
        self.model_performance = {}
        self.feature_importance_trends = {}
        self.prediction_accuracy = {}
        
        # Signal management
        self.active_signals: Set[str] = set()
        self.active_coins: Set[str] = set()
        self.signal_grace_period = 300
        self.scanning_lock = asyncio.Lock()
        
        # Enhanced performance metrics
        self.performance_metrics = {
            'total_scans': 0,
            'signals_generated': 0,
            'signals_closed': 0,
            'avg_scan_time': 0,
            'win_rate': 0,
            'avg_hold_time': 0,
            'ml_model_accuracy': 0,
            'feature_orthogonality_score': 0,
            'prediction_confidence_avg': 0,
            'sharpe_ratio': 0
        }
        
        # Market regime and ML insights
        self.market_regime = 'normal'
        self.dominant_features = []
        self.model_confidence_threshold = 0.75
        
        # Setup and initialize
        self.setup_routes()
        self.initialize_ml_bot_state()
        
        # Load existing models if available
        self.analyzer.load_models()
        
        # Graceful shutdown handling
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle graceful shutdown with model saving"""
        logger.info(f"Received signal {signum}. Saving models and shutting down...")
        self.analyzer.save_models()
        self.running = False
        sys.exit(0)

    def initialize_ml_bot_state(self):
        """Initialize ML bot state with enhanced cleanup"""
        try:
            # Standard cleanup
            active_signals = self.database.get_active_signals()
            self.active_coins = {signal['coin'] for signal in active_signals}
            
            # Clean stale signals
            stale_signals = []
            current_time = datetime.now()
            
            for signal in active_signals:
                try:
                    signal_time = datetime.fromisoformat(signal['timestamp'].replace('Z', '+00:00'))
                    if (current_time - signal_time).total_seconds() > 86400:  # 24 hours
                        stale_signals.append(signal)
                except:
                    stale_signals.append(signal)
            
            # Close stale signals
            for signal in stale_signals:
                self.database.close_signal(
                    signal['signal_id'],
                    signal['entry_price'],
                    "ML system cleanup - stale signal"
                )
            
            if stale_signals:
                logger.info(f"ML cleanup: Removed {len(stale_signals)} stale signals")
            
            # Initialize ML-specific tracking
            self.initialize_ml_tracking()
            
            # Clean old data
            self.database.clean_old_data(days=30)
            
            # Log ML bot initialization
            self.database.log_bot_activity(
                'INFO', 'ML_SYSTEM', 'ML trading bot initialized',
                f'Feature engineering and time-series validation ready for {len(self.analyzer.coins)} coins'
            )
            
        except Exception as e:
            logger.error(f"Error during ML bot initialization: {e}")

    async def get_account_balance(self) -> Dict:
        """Get real account balance from Bybit"""
        try:
            # Use basic fetch_balance which works for your account
            balance = self.analyzer.exchange.fetch_balance({'accountType': 'UNIFIED'})
    
            if 'USDT' in balance:
                # Handle None values properly
                total = float(balance['USDT'].get('total', 0) or 0)
                free = float(balance['USDT'].get('free', 0) or total)  # Use total if free is None
                used = float(balance['USDT'].get('used', 0) or 0)
        
                # If free is still 0 but we have total, assume all is free
                if free == 0 and total > 0:
                    free = total
        
                return {
                    'free': free,
                    'used': used,
                    'total': total
                }
            else:
                logger.warning("USDT not found in balance response")
                return {'free': 0, 'used': 0, 'total': 0}
        
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {'free': 0, 'used': 0, 'total': 0}
    
    
    async def execute_real_trade(self, signal_data: Dict) -> bool:
        """Execute real trade on Bybit with 15% of balance and 15x leverage"""
        try:
            logger.info(f"Processing coin: {signal_data['coin']}")
            symbol = signal_data['coin']
            logger.info(f"Bypassing symbol validation for {signal_data['coin']}")
            logger.info(f"Constructed symbol: {symbol}")
            # Convert prices to floats
            signal_data['entry_price'] = float(signal_data['entry_price'])
            signal_data['stop_loss'] = float(signal_data['stop_loss'])
            signal_data['take_profit'] = float(signal_data['take_profit'])
            # Get current balance
            balance_info = await self.get_account_balance()
            available_balance = balance_info['free']
            logger.info(f"Account balance: ${available_balance}, Using 15%: ${available_balance * 0.15}")
            
            if available_balance < 10:  # Minimum $10
                logger.warning(f"Insufficient balance: ${available_balance}")
                return False
            
            # Calculate position size (15% of available balance with 15x leverage)
            account_equity = available_balance * 0.15  # Use 15% of available balance
            leverage = 15
            position_value = account_equity * leverage
            
            # Get current market price
            ticker = self.analyzer.exchange.fetch_ticker(signal_data['coin'])
            current_price = ticker['last']
            
            symbol_info = self.analyzer.exchange.market(signal_data['coin'])

            # Ensure min_qty is ALWAYS a float
            min_qty = symbol_info.get('limits', {}).get('amount', {}).get('min')
            if min_qty is not None:
                min_qty = float(min_qty)
            else:
                min_qty = 0.001

            # Ensure qty_precision is ALWAYS an int
            qty_precision = symbol_info.get('precision', {}).get('amount')
            if qty_precision is not None:
                qty_precision = int(qty_precision) if isinstance(qty_precision, (str, float)) else qty_precision
            else:
                qty_precision = 8

            # Calculate base quantity from USD value
            base_quantity = position_value / current_price
            logger.info(f"Base quantity calculation: ${position_value} / ${current_price} = {base_quantity}")

            # Apply precision rules BEFORE min_qty check
            base_ceil = math.ceil(base_quantity * (10 ** qty_precision)) / (10 ** qty_precision)
            if qty_precision == 0:
                quantity = max(int(base_ceil), 1)
            else:
                quantity = round(base_ceil, qty_precision)

            # Ensure minimum quantity is met
            if quantity < min_qty:
                quantity = min_qty
                logger.info(f"Adjusted to minimum quantity: {quantity}")

            # Final check/adjust to meet 95% target if possible
            final_value = quantity * current_price
            if final_value < position_value * 0.95:
                step_size = 10 ** -qty_precision
                quantity = math.ceil(quantity / step_size) * step_size
                final_value = quantity * current_price
                logger.info(f"Ceiled quantity to meet target: {quantity}")

            # Verify final position value
            final_position_value = quantity * current_price
            logger.info(f"Final position: {quantity} contracts = ${final_position_value:.2f} (target was ${position_value:.2f})")
            logger.info(f"Final quantity: {quantity} (min_qty: {min_qty})")
            
            # Set leverage for the symbol
            try:
                self.analyzer.exchange.set_leverage(leverage, signal_data['coin'])
            except:
                pass  # Some symbols might have fixed leverage
            
            # Place market order for immediate execution
            side = 'buy' if signal_data['direction'] == 'LONG' else 'sell'
            
            order = self.analyzer.exchange.create_order(
                symbol=signal_data['coin'],
                type='market',
                side=side,
                amount=quantity,
                params={
                    'positionIdx': 0,  # One-way mode
                    'timeInForce': 'IOC'
                }
            )

            # Wait a moment for order to process
            await asyncio.sleep(3)

            # Fetch order status if initial response is incomplete
            if order and order.get('id'):
                try:
                    # Fetch the actual order status
                    order_status = self.analyzer.exchange.fetch_order(order['id'], signal_data['coin'])
                    if order_status:
                        order = order_status
                except Exception as e:
                    logger.warning(f"Could not fetch order status: {e}")

            # Check various status fields since response format may vary
            order_filled = False
            if order:
                status = order.get('status')
                filled = order.get('filled', 0)
                info = order.get('info', {})
    
                # Check multiple conditions for success
                if (status in ['closed', 'filled'] or 
                    filled > 0 or 
                (info and info.get('orderId'))):
                    order_filled = True
                    logger.info(f"Order details: status={status}, filled={filled}, id={order.get('id')}")

            if order_filled:
                # Set stop loss and take profit
                await self.set_sl_tp_orders(
                    signal_data['coin'],
                    signal_data['direction'],
                    quantity,
                    signal_data['stop_loss'],
                    signal_data['take_profit']
                )
                if not sl_tp_success:
                    logger.error(f"Failed to set SL/TP for {signal_data['coin']}")
                
                logger.info(f"REAL TRADE EXECUTED: {signal_data['coin']} {side} "
                           f"Qty: {quantity}, Value: ${position_value:.2f}, "
                           f"Leverage: {leverage}x, Account Risk: 15%")
                
                # Update signal with real order info
                signal_data['order_id'] = order['id']
                signal_data['executed_price'] = order['average'] if order.get('average') else current_price
                signal_data['executed_quantity'] = quantity
                signal_data['position_value'] = position_value
                
                return True
            else:
                logger.error(f"Order failed: {order}")
                return False
                
        except Exception as e:
            logger.error(f"Real trade execution failed: {e}")
            return False
    
    async def set_sl_tp_orders(self, symbol: str, direction: str, quantity: float, 
                           stop_loss: float, take_profit: float) -> bool:
        """Set stop loss and take profit orders on Bybit"""
        try:
            # Get current position to determine position side
            positions = self.analyzer.exchange.fetch_positions([symbol], params={'category': 'linear'})
            position = next((p for p in positions if p['symbol'] == symbol and p['contracts'] > 0), None)
        
            if not position:
                logger.warning(f"No position found for {symbol} to set SL/TP")
                return False

            # Stop Loss Order (Market order triggered by stop price)
            sl_side = 'sell' if direction == 'LONG' else 'buy'
            try:
                sl_order = self.analyzer.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=sl_side,
                    amount=quantity,
                    price=None,  # No price for market orders
                    params={
                        'stopPrice': stop_loss,
                        'triggerBy': 'LastPrice',
                        'closeOnTrigger': True,
                        'reduceOnly': True,
                        'positionIdx': 0
                    }
                )
                logger.info(f"Stop Loss set at {stop_loss} for {symbol}")
            except Exception as e:
                logger.error(f"Failed to set Stop Loss for {symbol}: {e}")

            # Take Profit Order (Limit order)  
            tp_side = 'sell' if direction == 'LONG' else 'buy'
            try:
                tp_order = self.analyzer.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=tp_side,
                    amount=quantity,
                    price=take_profit,
                    params={
                        'closeOnTrigger': True,
                        'reduceOnly': True,
                        'positionIdx': 0,
                        'timeInForce': 'GTC'
                    }
                )
                logger.info(f"Take Profit set at {take_profit} for {symbol}")
            except Exception as e:
                logger.error(f"Failed to set Take Profit for {symbol}: {e}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to set SL/TP for {symbol}: {e}")
            return False
    
    async def close_real_position(self, signal_data: Dict, reason: str) -> bool:
        """Close real position on Bybit"""
        try:
            symbol = signal_data['coin']
            
            # Get current position
            positions = self.analyzer.exchange.fetch_positions([symbol], params={'category': 'linear'})
            position = next((p for p in positions if p['symbol'] == symbol), None)
            
            if not position or position['contracts'] == 0:
                logger.warning(f"No open position for {symbol}")
                return False
            
            # Close position with market order
            side = 'sell' if position['side'] == 'long' else 'buy'
            close_order = self.analyzer.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=abs(position['contracts']),
                params={
                    'positionIdx': 0,
                    'reduceOnly': True
                }
            )
            
            logger.info(f"REAL POSITION CLOSED: {symbol} - Reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close real position: {e}")
            return False

    def initialize_ml_tracking(self):
        """Initialize ML-specific performance tracking"""
        try:
            # Initialize model performance tracking for each coin
            for coin in self.analyzer.coins:
                symbol = coin.replace('/USDT', '')
                self.model_performance[symbol] = {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'last_trained': None,
                    'prediction_count': 0,
                    'correct_predictions': 0
                }
                
                self.feature_importance_trends[symbol] = {
                    'top_features': [],
                    'stability_score': 0.0,
                    'last_updated': None
                }
            
            logger.info("ML tracking systems initialized")
            
        except Exception as e:
            logger.error(f"Error initializing ML tracking: {e}")

    def validate_ml_signal_before_save(self, signal_data: Dict) -> bool:
        try:
            signal_data['coin'] = signal_data['coin'].replace('/USDT', '')
            # Standard validation
            if not self.validate_signal_before_save(signal_data):
                return False
        
            # ML-specific validation
            ml_prediction = signal_data.get('ml_prediction', 0)
            model_confidence = signal_data.get('model_confidence', 0)
        
            # Debug missing ml_prediction
            if 'ml_prediction' not in signal_data or ml_prediction is None:
                logger.error(f"Missing or invalid ml_prediction for {signal_data.get('coin', 'unknown')}")
                return False
        
            # Check ML prediction strength
            entry_price = signal_data.get('entry_price', 1.0)
            min_prediction_threshold = 0.01 if entry_price > 0.001 else 0.002
            if abs(ml_prediction) < min_prediction_threshold:
                logger.warning(f"ML prediction too weak: {ml_prediction} for {signal_data['coin']}")
                return False
        
            # Check model confidence
            if model_confidence < self.model_confidence_threshold:
                logger.warning(f"Model confidence too low: {model_confidence}")
                return False
        
            # Check feature importance sum
            feature_importance = signal_data.get('analysis_data', {}).get('feature_importance_top5', {})
            importance_sum = sum(feature_importance.values()) if feature_importance else 0
        
            if importance_sum < 0.1:
                logger.warning(f"Feature importance sum too low: {importance_sum}")
                return False
        
            return True
        
        except Exception as e:
            logger.error(f"Error in ML signal validation: {e}")
            return False

    def validate_signal_before_save(self, signal_data: Dict) -> bool:
        try:
            signal_data['coin'] = signal_data['coin'].replace('/USDT', '')
            required_fields = ['signal_id', 'coin', 'direction', 'entry_price', 'take_profit', 'stop_loss']
            for field in required_fields:
                if field not in signal_data:
                    logger.error(f"Missing required field: {field}")
                    return False

            entry = float(signal_data['entry_price'])
            tp = float(signal_data['take_profit'])
            sl = float(signal_data['stop_loss'])
            direction = signal_data['direction']

            # Reject negative or zero prices
            if entry <= 0 or tp <= 0 or sl <= 0:
                logger.error(f"Invalid price (negative or zero): SL={sl}, Entry={entry}, TP={tp}")
                return False

            # Dynamic minimum price difference
            if entry > 100:  # High-value coins
                min_price_diff = entry * 0.005  # 0.5%
            elif entry > 1:  # Mid-value coins  
                min_price_diff = entry * 0.01   # 1%
            else:  # Low-value coins
                min_price_diff = entry * 0.015  # 1.5%

            if abs(entry - sl) < min_price_diff or abs(tp - entry) < min_price_diff:
                logger.warning(f"Price difference too small for {signal_data['coin']}: SL={sl:.6f}, Entry={entry:.6f}, TP={tp:.6f}")
                return False

            if direction == 'LONG':
                if not (sl < entry < tp):
                    logger.error(f"Invalid LONG signal prices: SL={sl}, Entry={entry}, TP={tp}")
                    return False
            else:  # SHORT
                if not (tp < entry < sl):
                    logger.error(f"Invalid SHORT signal prices: TP={tp}, Entry={entry}, SL={sl}")
                    return False

            active_signals = self.database.get_active_signals()
            active_coins = {signal['coin'] for signal in active_signals}
            if signal_data['coin'] in active_coins:
                logger.warning(f"Coin {signal_data['coin']} already has active signal")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False

    def setup_routes(self):
        """Setup all API routes with ML enhancements"""
        
        @self.app.get("/")
        async def dashboard():
            """Serve ML-enhanced dashboard"""
            try:
                with open("templates/dashboard.html", "r") as f:
                    return HTMLResponse(content=f.read())
            except FileNotFoundError:
                return HTMLResponse(content=self.get_ml_enhanced_dashboard())
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """Enhanced WebSocket with ML updates"""
            await websocket.accept()
            self.websocket_connections.add(websocket)
            
            try:
                await websocket.send_text(json.dumps({
                    "type": "connection_established",
                    "message": "Connected to ML Trading Bot",
                    "version": "3.0.0",
                    "features": [
                        "advanced_feature_engineering",
                        "feature_orthogonalization", 
                        "time_series_validation",
                        "explainable_ai",
                        "dynamic_model_retraining"
                    ],
                    "timestamp": datetime.now().isoformat()
                }))
                
                while True:
                    await asyncio.sleep(30)
                    await websocket.send_text(json.dumps({
                        "type": "heartbeat",
                        "ml_status": {
                            "models_loaded": len(self.analyzer.models),
                            "last_training": self.last_train_time.isoformat() if self.last_train_time else None,
                            "avg_confidence": self.performance_metrics.get('prediction_confidence_avg', 0)
                        },
                        "timestamp": datetime.now().isoformat()
                    }))
                    
            except WebSocketDisconnect:
                self.websocket_connections.discard(websocket)
            except Exception as e:
                logger.debug(f"WebSocket error: {e}")
                self.websocket_connections.discard(websocket)

        @self.app.get("/api/trades/history")
        async def get_trade_history():
            """Get closed trades history"""
            try:
                with self.database.get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT * FROM signals 
                        WHERE status = 'closed' 
                        ORDER BY closed_at DESC 
                        LIMIT 50
                    ''')
            
                    trades = []
                    for row in cursor.fetchall():
                        trade = dict(row)
                        # Calculate P&L
                        if trade['direction'] == 'LONG':
                            pnl_pct = ((trade['exit_price'] - trade['entry_price']) / trade['entry_price']) * 100 * 10
                        else:
                            pnl_pct = ((trade['entry_price'] - trade['exit_price']) / trade['entry_price']) * 100 * 10
                
                        trade['pnl_usd'] = (pnl_pct / 100) * 1000
                        trade['pnl_percentage'] = pnl_pct
                        trades.append(trade)
            
                    return JSONResponse(content={"trades": trades})
            except Exception as e:
                logger.error(f"Error getting trade history: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.get("/api/signals")
        async def get_ml_signals():
            """Get active ML signals with enhanced data"""
            try:
                signals = self.database.get_active_signals()
                enhanced_signals = []
                
                for signal in signals:
                    try:
                        coin = signal['coin'].replace('/USDT', '').replace('USDT', '')
                        symbol = f"{coin}USDT"
                        current_price = await self.analyzer.get_current_price(symbol)
                        
                        if current_price > 0:
                            # Update database
                            self.database.update_signal_price(signal['signal_id'], current_price)
                            
                            # Calculate live P&L
                            entry_price = signal['entry_price']
                            direction = signal['direction']
                            
                            if direction == 'LONG':
                                pnl_pct = ((current_price - entry_price) / entry_price) * 100 * 10
                            else:
                                pnl_pct = ((entry_price - current_price) / entry_price) * 100 * 10
                            
                            pnl_usd = (pnl_pct / 100) * 1000

                            signal['current_price'] = current_price
                            signal['live_pnl_usd'] = round(pnl_usd, 2)       # ← Make sure these are added
                            signal['live_pnl_percentage'] = round(pnl_pct, 2)
                            
                            # Enhanced ML metrics
                            analysis_data = signal.get('analysis_data', {})
                            ml_prediction = analysis_data.get('ml_prediction', 0)
                            model_confidence = analysis_data.get('model_confidence', 0)
                            
                            # Progress calculations
                            if direction == 'LONG':
                                tp_progress = ((current_price - entry_price) / 
                                             (signal['take_profit'] - entry_price)) * 100
                                sl_distance = ((current_price - signal['stop_loss']) / 
                                             (entry_price - signal['stop_loss'])) * 100
                            else:
                                tp_progress = ((entry_price - current_price) / 
                                             (entry_price - signal['take_profit'])) * 100
                                sl_distance = ((signal['stop_loss'] - current_price) / 
                                             (signal['stop_loss'] - entry_price)) * 100
                            
                            # Add ML-specific data
                            signal['current_price'] = current_price
                            signal['live_pnl_usd'] = round(pnl_usd, 2)
                            signal['live_pnl_percentage'] = round(pnl_pct, 2)
                            signal['tp_progress'] = max(0, min(100, tp_progress))
                            signal['sl_distance'] = max(0, min(100, sl_distance))
                            signal['ml_prediction'] = ml_prediction
                            signal['model_confidence'] = model_confidence
                            signal['signal_grade'] = analysis_data.get('signal_grade', 'ml_based')
                            signal['feature_importance_top3'] = dict(list(
                                analysis_data.get('feature_importance_top5', {}).items())[:3])
                            
                        enhanced_signals.append(signal)
                        
                    except Exception as e:
                        logger.error(f"Error enhancing ML signal {signal['signal_id']}: {e}")
                        enhanced_signals.append(signal)
                
                return JSONResponse(content={
                    "signals": enhanced_signals,
                    "count": len(enhanced_signals),
                    "performance": self.performance_metrics,
                    "ml_metrics": {
                        "models_active": len(self.analyzer.models),
                        "avg_model_confidence": self.performance_metrics.get('prediction_confidence_avg', 0),
                        "feature_orthogonality": self.performance_metrics.get('feature_orthogonality_score', 0),
                        "dominant_features": self.dominant_features[:5]
                    },
                    "market_regime": self.market_regime
                })
                
            except Exception as e:
                logger.error(f"Error getting ML signals: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.get("/api/ml/models")
        async def get_model_status():
            """Get ML model status and performance"""
            try:
                model_status = {}
                
                for symbol, model_info in self.analyzer.feature_importance_history.items():
                    model_status[symbol] = {
                        'model_type': model_info.get('best_model', 'unknown'),
                        'cv_score': model_info.get('best_score', 0),
                        'feature_count': len(model_info.get('feature_importance', {})),
                        'top_features': dict(list(
                            sorted(model_info.get('feature_importance', {}).items(), 
                                  key=lambda x: x[1], reverse=True)[:5])),
                        'shap_available': len(model_info.get('shap_importance', {})) > 0,
                        'last_trained': self.model_performance.get(symbol, {}).get('last_trained'),
                        'accuracy': self.model_performance.get(symbol, {}).get('accuracy', 0)
                    }
                
                return JSONResponse(content={
                    "models": model_status,
                    "global_stats": {
                        "total_models": len(self.analyzer.models),
                        "avg_accuracy": np.mean([m.get('accuracy', 0) for m in self.model_performance.values()]),
                        "training_status": "active" if self.is_training else "idle",
                        "last_training_session": self.last_train_time.isoformat() if self.last_train_time else None
                    }
                })
                
            except Exception as e:
                logger.error(f"Error getting model status: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.get("/api/ml/features/{coin}")
        async def get_feature_analysis(coin: str):
            """Get detailed feature analysis for a coin"""
            try:
                symbol = f"{coin}USDT"
                
                # Get market data
                df = await self.analyzer.get_market_data(symbol, '1h', 500)
                if df.empty:
                    return JSONResponse(content={"error": "No data available"}, status_code=404)
                
                # Engineer features
                df = self.analyzer.engineer_features(df)
                df = self.analyzer.orthogonalize_features(df)
                
                # Get feature selection results
                df, selected_features = self.analyzer.select_features(df)
                
                # Get model info if available
                model_info = self.analyzer.feature_importance_history.get(coin, {})
                
                analysis = {
                    'coin': coin,
                    'total_features_engineered': len([col for col in df.columns if not any(x in col.lower() for x in 
                                                    ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'target'])]),
                    'selected_features': selected_features,
                    'feature_importance': model_info.get('feature_importance', {}),
                    'shap_importance': model_info.get('shap_importance', {}),
                    'model_performance': {
                        'cv_score': model_info.get('best_score', 0),
                        'model_type': model_info.get('best_model', 'not_trained'),
                        'feature_stability': self.feature_importance_trends.get(coin, {}).get('stability_score', 0)
                    },
                    'feature_categories': {
                        'price_features': len([f for f in selected_features if 'return' in f or 'price' in f]),
                        'volume_features': len([f for f in selected_features if 'volume' in f or 'vwap' in f]),
                        'volatility_features': len([f for f in selected_features if 'vol' in f or 'atr' in f]),
                        'technical_features': len([f for f in selected_features if any(t in f for t in ['rsi', 'macd', 'bb', 'sma', 'ema'])]),
                        'statistical_features': len([f for f in selected_features if any(s in f for s in ['skew', 'kurt', 'autocorr', 'entropy'])])
                    }
                }
                
                return JSONResponse(content=analysis)
                
            except Exception as e:
                logger.error(f"Error getting feature analysis for {coin}: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.get("/api/portfolio")
        async def get_ml_portfolio():
            """Get enhanced ML portfolio statistics"""
            try:
                portfolio = self.database.get_portfolio_stats()
        
                # ADD: Fetch real Bybit balance
                real_balance = await self.get_account_balance()
                portfolio['real_balance'] = real_balance['total']
                portfolio['available_balance'] = real_balance['free']
                portfolio['used_balance'] = real_balance['used']
        
                # Calculate real open P&L from Bybit positions
                active_signals = self.database.get_active_signals()
                total_open_pnl = 0
        
                for signal in active_signals:
                    try:
                        symbol = signal['coin'].replace('/USDT', '') + 'USDT'
                        positions = self.analyzer.exchange.fetch_positions([symbol])
                        for pos in positions:
                            if pos['contracts'] > 0:
                                total_open_pnl += pos['unrealizedPnl'] or 0
                    except:
                        pass
        
                portfolio['real_open_pnl'] = total_open_pnl
                portfolio['total_balance'] = real_balance['total'] + total_open_pnl
                
                # Add ML-specific metrics
                portfolio['ml_metrics'] = {
                    'model_count': len(self.analyzer.models),
                    'avg_prediction_accuracy': self.performance_metrics.get('ml_model_accuracy', 0),
                    'feature_orthogonality_score': self.performance_metrics.get('feature_orthogonality_score', 0),
                    'prediction_confidence_avg': self.performance_metrics.get('prediction_confidence_avg', 0),
                    'sharpe_ratio': self.performance_metrics.get('sharpe_ratio', 0)
                }
                
                portfolio['signal_quality_score'] = self.calculate_ml_signal_quality_score()
                portfolio['risk_metrics'] = self.get_ml_risk_metrics()
                portfolio['market_regime'] = self.market_regime
                
                return JSONResponse(content={"portfolio": portfolio})
                
            except Exception as e:
                logger.error(f"Error getting ML portfolio: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.get("/api/system")
        async def get_ml_system_stats():
            """Get enhanced ML system statistics"""
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                uptime = (datetime.now() - self.startup_time).total_seconds()
                
                return JSONResponse(content={
                    "cpu_usage": cpu_percent,
                    "memory_usage": memory.percent,
                    "uptime_seconds": uptime,
                    "scan_count": self.scan_count,
                    "last_scan": self.last_scan_time.isoformat() if self.last_scan_time else None,
                    "last_training": self.last_train_time.isoformat() if self.last_train_time else None,
                    "active_connections": len(self.websocket_connections),
                    "is_running": self.running,
                    "is_scanning": self.is_scanning,
                    "is_training": self.is_training,
                    "scan_interval": self.scan_interval,
                    "retrain_interval": self.retrain_interval,
                    "performance_metrics": self.performance_metrics,
                    "analyzer_type": "ml_enhanced",
                    "ml_features": {
                        "feature_engineering": True,
                        "orthogonalization": True,
                        "time_series_validation": True,
                        "explainable_ai": True,
                        "auto_retraining": True
                    },
                    "coins_monitored": len(self.analyzer.coins),
                    "models_trained": len(self.analyzer.models),
                    "market_regime": self.market_regime,
                    "active_signals_count": len(self.active_signals),
                    "model_confidence_threshold": self.model_confidence_threshold
                })
                
            except Exception as e:
                logger.error(f"Error getting ML system stats: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.get("/api/logs")
        async def get_recent_logs():
            """Get recent bot logs"""
            try:
                logs = self.database.get_recent_logs(limit=50)
                return JSONResponse(content={"logs": logs})
            except Exception as e:
                logger.error(f"Error getting logs: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.get("/api/alerts")
        async def get_alerts():
            """Get system alerts and notifications"""
            try:
                # Get recent errors and important events
                with self.database.get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                    SELECT * FROM bot_logs 
                    WHERE level IN ('ERROR', 'WARNING') 
                    ORDER BY timestamp DESC 
                    LIMIT 20
                    ''')
                    
                    alerts = []
                    for row in cursor.fetchall():
                        alert = dict(row)
                        alerts.append({
                            'id': alert['id'],
                            'level': alert['level'],
                            'message': alert['message'],
                            'component': alert['component'],
                            'timestamp': alert['timestamp'],
                            'details': alert.get('details')
                        })
                
                return JSONResponse(content={"alerts": alerts})
            except Exception as e:
                logger.error(f"Error getting alerts: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.post("/api/control/retrain")
        async def trigger_model_retrain():
            """Manually trigger model retraining"""
            if self.is_training:
                return JSONResponse(content={"status": "already_training", "message": "Models are already being retrained"})
            
            asyncio.create_task(self.retrain_models())
            return JSONResponse(content={"status": "training_started", "message": "Model retraining initiated"})

        @self.app.post("/api/control/start")
        async def start_ml_bot():
            """Start the ML trading bot"""
            if not self.running:
                self.running = True
                asyncio.create_task(self.ml_scan_cycle())
                asyncio.create_task(self.ml_retrain_cycle())
                message = "ML trading bot started with advanced features"
                logger.info(message)
                self.database.log_bot_activity('INFO', 'ML_SYSTEM', message)
                return JSONResponse(content={"status": "started", "message": message})
            else:
                return JSONResponse(content={"status": "already_running", "message": "ML bot is already running"})

        @self.app.post("/api/control/stop")
        async def stop_ml_bot():
            """Stop the ML trading bot"""
            self.running = False
            # Save models before stopping
            self.analyzer.save_models()
            message = "ML trading bot stopped and models saved"
            logger.info(message)
            self.database.log_bot_activity('INFO', 'ML_SYSTEM', message)
            return JSONResponse(content={"status": "stopped", "message": message})

        @self.app.get("/api/health")
        async def health_check():
            """Enhanced health check with ML status"""
            return JSONResponse(content={
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "3.0.0",
                "ml_status": {
                    "models_loaded": len(self.analyzer.models),
                    "training_active": self.is_training,
                    "last_prediction": self.last_scan_time.isoformat() if self.last_scan_time else None
                }
            })

    def calculate_ml_signal_quality_score(self) -> float:
        """Calculate ML-enhanced signal quality score"""
        try:
            if self.performance_metrics['total_scans'] == 0:
                return 0.0
            
            # Base score from win rate
            win_rate = self.performance_metrics.get('win_rate', 0)
            base_score = win_rate * 0.3
            
            # ML prediction accuracy
            ml_accuracy = self.performance_metrics.get('ml_model_accuracy', 0)
            accuracy_score = ml_accuracy * 0.25
            
            # Feature orthogonality bonus
            orthogonality = self.performance_metrics.get('feature_orthogonality_score', 0)
            orthogonality_score = orthogonality * 0.2
            
            # Model confidence
            confidence = self.performance_metrics.get('prediction_confidence_avg', 0)
            confidence_score = confidence * 0.15
            
            # Sharpe ratio component
            sharpe = min(3.0, max(0, self.performance_metrics.get('sharpe_ratio', 0))) / 3.0
            sharpe_score = sharpe * 0.1
            
            return min(1.0, base_score + accuracy_score + orthogonality_score + confidence_score + sharpe_score)
            
        except Exception as e:
            logger.error(f"Error calculating ML signal quality score: {e}")
            return 0.0

    def get_ml_risk_metrics(self) -> Dict:
        """Get ML-enhanced risk metrics"""
        try:
            active_signals = self.database.get_active_signals()
            
            if not active_signals:
                return {
                    "total_risk": 0,
                    "max_single_risk": 0,
                    "portfolio_heat": 0,
                    "correlation_risk": "low",
                    "active_positions": 0,
                    "model_risk": "low",
                    "feature_stability": 1.0
                }
            
            # Standard risk calculations
            total_risk = sum(signal.get('analysis_data', {}).get('risk_percentage', 1) for signal in active_signals)
            max_single_risk = max(signal.get('analysis_data', {}).get('risk_percentage', 1) for signal in active_signals)
            portfolio_heat = len(active_signals) * 1.0  # Assuming 1% avg risk
            
            # ML-specific risk assessments
            model_confidences = [signal.get('analysis_data', {}).get('model_confidence', 0) for signal in active_signals]
            avg_confidence = np.mean(model_confidences) if model_confidences else 0
            
            # Feature stability assessment
            feature_stability = np.mean([
                self.feature_importance_trends.get(signal['coin'], {}).get('stability_score', 0.5)
                for signal in active_signals
            ])
            
            # Model risk assessment
            if avg_confidence > 0.85:
                model_risk = "low"
            elif avg_confidence > 0.75:
                model_risk = "medium"
            else:
                model_risk = "high"
            
            # Correlation risk (enhanced with ML insights)
            if len(active_signals) <= 2:
                correlation_risk = "low"
            elif len(active_signals) <= 4 and avg_confidence > 0.8:
                correlation_risk = "medium"
            else:
                correlation_risk = "high"
            
            return {
                "total_risk": round(total_risk, 2),
                "max_single_risk": round(max_single_risk, 2),
                "portfolio_heat": round(portfolio_heat, 2),
                "correlation_risk": correlation_risk,
                "active_positions": len(active_signals),
                "model_risk": model_risk,
                "feature_stability": round(feature_stability, 3),
                "avg_model_confidence": round(avg_confidence, 3)
            }
            
        except Exception as e:
            logger.error(f"Error calculating ML risk metrics: {e}")
            return {"error": "calculation_failed"}

    async def broadcast_to_clients(self, message: Dict):
        """Broadcast ML updates to WebSocket clients"""
        if not self.websocket_connections:
            return
        
        disconnected = set()
        message_str = json.dumps(message, default=str)
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message_str)
            except Exception:
                disconnected.add(websocket)
        
        # Remove disconnected clients
        self.websocket_connections -= disconnected

    async def ml_scan_cycle(self):
        """ML-enhanced market scanning cycle"""
        while self.running:
            try:
                if self.scanning_lock.locked():
                    await asyncio.sleep(60)
                    continue
                
                async with self.scanning_lock:
                    self.is_scanning = True
                    scan_start = datetime.now()
                    
                    logger.info(f"Starting ML scan #{self.scan_count + 1}")
                    
                    # Broadcast scan start
                    await self.broadcast_to_clients({
                        "type": "ml_scan_started",
                        "scan_number": self.scan_count + 1,
                        "timestamp": scan_start.isoformat(),
                        "analyzer_type": "ml_enhanced",
                        "models_active": len(self.analyzer.models)
                    })
                    
                    # Check signal exits first
                    await self.check_signal_exits()

                    await self.evaluate_and_adjust_positions()
                    
                    # ML signal generation
                    active_signals = self.database.get_active_signals()
                    max_concurrent_signals = 3
                    
                    new_signals_count = 0
                    ml_predictions_made = 0
                    
                    if len(active_signals) < max_concurrent_signals:
                        try:
                            # Use ML analyzer for signal generation
                            new_signals = await self.analyzer.scan_all_coins()
                            ml_predictions_made = len(new_signals)
                            
                            active_coins = {signal['coin'] for signal in active_signals}
                            
                            for signal_data in new_signals:
                                if len(active_signals) + new_signals_count >= max_concurrent_signals:
                                    break
                                
                                # Enhanced ML validation
                                if not self.validate_ml_signal_before_save(signal_data):
                                    continue
                                
                                
                                # Execute REAL trade
                                trade_executed = await self.execute_real_trade(signal_data)
                                
                                if trade_executed:
                                    # Save to database for tracking
                                    success = self.database.save_signal(signal_data)
                                    if success:
                                        new_signals_count += 1
                                        self.active_coins.add(signal_data['coin'])
                                        self.active_signals.add(signal_data['signal_id'])
                                    
                                    logger.warning(f"💰 REAL MONEY TRADE: {signal_data['coin']} "
                                                 f"{signal_data['direction']} at {signal_data['entry_price']}")
                                else:
                                    logger.error(f"Failed to execute real trade for {signal_data['coin']}")
                                    self.update_ml_signal_tracking(signal_data)
                                    continue                                 
                                    
                                    
                                    
                                    # Grace period
                                    asyncio.create_task(
                                        self.remove_from_grace_period(
                                            signal_data['signal_id'],
                                            self.signal_grace_period
                                        )
                                    )
                                    
                                    # Broadcast ML signal
                                    await self.broadcast_to_clients({
                                        "type": "new_ml_signal",
                                        "signal": signal_data,
                                        "ml_prediction": signal_data.get('ml_prediction', 0),
                                        "model_confidence": signal_data.get('model_confidence', 0),
                                        "top_features": dict(list(
                                            signal_data.get('feature_importance', {}).items())[:3])
                                    })
                                    
                                    logger.info(f"New ML {signal_data['direction']} signal: {signal_data['coin']} "
                                              f"(Pred: {signal_data.get('ml_prediction', 0):.4f}, "
                                              f"Conf: {signal_data.get('model_confidence', 0):.3f})")
                        
                        except Exception as e:
                            logger.error(f"Error during ML signal generation: {e}")
                    
                    # Update performance metrics
                    self.scan_count += 1
                    self.last_scan_time = datetime.now()
                    scan_duration = (self.last_scan_time - scan_start).total_seconds()
                    self.is_scanning = False
                    
                    # Update ML metrics
                    self.update_ml_performance_metrics(new_signals_count, ml_predictions_made, scan_duration)
                    
                    # Log ML scan completion
                    self.database.log_bot_activity(
                        'INFO', 'ML_SCANNER', 'ML scan completed',
                        f'Duration: {scan_duration:.1f}s, ML predictions: {ml_predictions_made}, '
                        f'New signals: {new_signals_count}, Models active: {len(self.analyzer.models)}'
                    )
                    
                    # Broadcast scan completion
                    await self.broadcast_to_clients({
                        "type": "ml_scan_completed",
                        "scan_count": self.scan_count,
                        "duration": scan_duration,
                        "new_signals": new_signals_count,
                        "ml_predictions": ml_predictions_made,
                        "performance": self.performance_metrics
                    })
                    
            except Exception as e:
                self.is_scanning = False
                error_msg = f"ML scan failed: {str(e)}"
                logger.error(error_msg)
                self.database.log_bot_activity('ERROR', 'ML_SCANNER', error_msg)
                
                await self.broadcast_to_clients({
                    "type": "ml_scan_error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
            
            # Wait for next scan
            await asyncio.sleep(self.scan_interval)

    async def ml_retrain_cycle(self):
        """Automatic model retraining cycle"""
        while self.running:
            try:
                await asyncio.sleep(self.retrain_interval)
                
                if not self.is_training and self.running:
                    await self.retrain_models()
                    
            except Exception as e:
                logger.error(f"Error in retrain cycle: {e}")

    async def retrain_models(self):
        """Retrain ML models with fresh data"""
        if self.is_training:
            return
        
        self.is_training = True
        retrain_start = datetime.now()
        
        try:
            logger.info("Starting ML model retraining cycle")
            
            # Broadcast training start
            await self.broadcast_to_clients({
                "type": "model_training_started",
                "timestamp": retrain_start.isoformat()
            })
            
            models_retrained = 0
            
            # Retrain models for top performing coins
            for coin in self.analyzer.coins:
                try:
                    symbol = f"{coin}"
                    
                    # Get extended data for training
                    df = await self.analyzer.get_market_data(symbol, '1h', 500)
                    
                    if len(df) >= 300:  # Minimum data for training
                        model_info = self.analyzer.train_model(df, coin.replace('/USDT', ''))
                        
                        if model_info:
                            models_retrained += 1
                            
                            # Update tracking
                            self.model_performance[coin.replace('/USDT', '')] = {
                                'accuracy': 1.0 / (1.0 + model_info.get('cv_score', 1.0)),
                                'last_trained': datetime.now().isoformat(),
                                'prediction_count': 0,
                                'correct_predictions': 0
                            }
                            
                            logger.info(f"Retrained model for {coin} - CV Score: {model_info.get('cv_score', 0):.6f}")
                    
                    # Small delay between models
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error retraining model for {coin}: {e}")
            
            # Save models after retraining
            self.analyzer.save_models()
            
            self.last_train_time = datetime.now()
            training_duration = (self.last_train_time - retrain_start).total_seconds()
            
            logger.info(f"Model retraining completed: {models_retrained} models in {training_duration:.1f}s")
            
            # Broadcast training completion
            await self.broadcast_to_clients({
                "type": "model_training_completed",
                "models_retrained": models_retrained,
                "duration": training_duration,
                "timestamp": self.last_train_time.isoformat()
            })
            
            # Log training activity
            self.database.log_bot_activity(
                'INFO', 'ML_TRAINER', 'Model retraining completed',
                f'Retrained {models_retrained} models in {training_duration:.1f}s'
            )
            
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
        finally:
            self.is_training = False

    def update_ml_signal_tracking(self, signal_data: Dict):
        """Update ML-specific signal tracking"""
        try:
            coin = signal_data['coin']
            ml_prediction = signal_data.get('ml_prediction', 0)
            model_confidence = signal_data.get('model_confidence', 0)
            feature_importance = signal_data.get('feature_importance', {})
            
            # Update prediction tracking
            if coin in self.model_performance:
                self.model_performance[coin]['prediction_count'] += 1
            
            # Update feature importance trends
            if feature_importance and coin in self.feature_importance_trends:
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                self.feature_importance_trends[coin]['top_features'] = [f[0] for f in top_features]
                self.feature_importance_trends[coin]['last_updated'] = datetime.now().isoformat()
            
            # Update global dominant features
            if feature_importance:
                for feature, importance in feature_importance.items():
                    if importance > 0.1 and feature not in self.dominant_features:
                        self.dominant_features.append(feature)
            
            # Keep only top 10 dominant features
            self.dominant_features = self.dominant_features[:10]
            
        except Exception as e:
            logger.error(f"Error updating ML signal tracking: {e}")

    def update_ml_performance_metrics(self, new_signals: int, predictions_made: int, scan_duration: float):
        """Update ML-specific performance metrics"""
        try:
            # Standard metrics
            self.performance_metrics['total_scans'] = self.scan_count
            self.performance_metrics['signals_generated'] += new_signals
            
            # Calculate average scan time
            prev_avg = self.performance_metrics.get('avg_scan_time', 0)
            self.performance_metrics['avg_scan_time'] = (
                (prev_avg * (self.scan_count - 1) + scan_duration) / self.scan_count
            )
            
            # ML-specific metrics
            model_accuracies = [m.get('accuracy', 0) for m in self.model_performance.values()]
            self.performance_metrics['ml_model_accuracy'] = np.mean(model_accuracies) if model_accuracies else 0
            
            # Feature orthogonality score (simplified calculation)
            self.performance_metrics['feature_orthogonality_score'] = 0.85  # Placeholder
            
            # Prediction confidence average
            self.performance_metrics['prediction_confidence_avg'] = 0.8  # Will be updated with real data
            
            # Update win rate from database
            portfolio = self.database.get_portfolio_stats()
            self.performance_metrics['win_rate'] = portfolio.get('win_rate', 0) / 100
            
            # Calculate Sharpe ratio (simplified)
            if portfolio.get('total_trades', 0) > 5:
                avg_return = portfolio.get('total_pnl', 0) / portfolio.get('total_trades', 1)
                self.performance_metrics['sharpe_ratio'] = max(0, avg_return / 100)  # Simplified
            
        except Exception as e:
            logger.error(f"Error updating ML performance metrics: {e}")

    async def check_signal_exits(self):
        """Enhanced signal exit checking with ML insights"""
        try:
            active_signals = self.database.get_active_signals()
            
            for signal in active_signals:
                if signal['signal_id'] in self.active_signals:
                    continue
                
                try:
                    current_price = await self.analyzer.get_current_price(signal['coin'])
                    if current_price <= 0:
                        continue
                    
                    self.database.update_signal_price(signal['signal_id'], current_price)
                    
                    # Check exit conditions
                    exit_reason = None
                    exit_price = None
                    buffer = 0.0015
                    
                    if signal['direction'] == 'LONG':
                        if current_price >= signal['take_profit'] * (1 - buffer):
                            exit_reason = "Take Profit Hit"
                            exit_price = signal['take_profit']
                        elif current_price <= signal['stop_loss'] * (1 + buffer):
                            exit_reason = "Stop Loss Hit"
                            exit_price = signal['stop_loss']
                    else:  # SHORT
                        if current_price <= signal['take_profit'] * (1 + buffer):
                            exit_reason = "Take Profit Hit"
                            exit_price = signal['take_profit']
                        elif current_price >= signal['stop_loss'] * (1 - buffer):
                            exit_reason = "Stop Loss Hit"
                            exit_price = signal['stop_loss']
                    
                    # Execute exit
                    if exit_reason and exit_price:
                        real_close_success = await self.close_real_position(signal, exit_reason)

                        success = self.database.close_signal(signal['signal_id'], exit_price, exit_reason)
                        
                        if success:
                            self.active_signals.discard(signal['signal_id'])
                            self.active_coins.discard(signal['coin'])
                            
                            # Calculate P&L
                            if signal['direction'] == 'LONG':
                                pnl_pct = ((exit_price - signal['entry_price']) / signal['entry_price']) * 100 * 10
                            else:
                                pnl_pct = ((signal['entry_price'] - exit_price) / signal['entry_price']) * 100 * 10
                            
                            pnl_usd = (pnl_pct / 100) * 1000
                            
                            # Update ML tracking
                            self.update_ml_prediction_accuracy(signal, exit_reason == "Take Profit Hit")
                            
                            # Update metrics
                            self.performance_metrics['signals_closed'] += 1
                            
                            # Broadcast signal closure
                            await self.broadcast_to_clients({
                                "type": "ml_signal_closed",
                                "signal_id": signal['signal_id'],
                                "coin": signal['coin'],
                                "direction": signal['direction'],
                                "exit_reason": exit_reason,
                                "pnl_usd": round(pnl_usd, 2),
                                "pnl_percentage": round(pnl_pct, 2),
                                "exit_price": exit_price,
                                "ml_prediction_accuracy": self.model_performance.get(signal['coin'], {}).get('accuracy', 0)
                            })
                            
                            logger.info(f"ML Signal closed: {signal['coin']} {exit_reason} - P&L: ${pnl_usd:.2f}")
                
                except Exception as e:
                    logger.error(f"Error checking exit for ML signal {signal['signal_id']}: {e}")
        
        except Exception as e:
            logger.error(f"Error in ML signal exit check: {e}")

    async def evaluate_and_adjust_positions(self):
        """Re-evaluate all active positions and adjust dynamically"""
        
        try:
            active_signals = self.database.get_active_signals()
            
            if not active_signals:
                return
            
            logger.info(f"Re-evaluating {len(active_signals)} active positions")
            
            for signal in active_signals:
                try:
                    symbol = signal['coin']                    
                    
                    # Get fresh market data
                    df = await self.analyzer.get_market_data(symbol, '1h', 100)
                    if df.empty:
                        continue
                    
                    # Evaluate position health
                    evaluation = self.analyzer.evaluate_position_health(signal, df)
                    
                    # Skip if health is good
                    if evaluation['health_score'] > 0.7 and evaluation['recommendation'] == 'hold':
                        continue
                    
                    # Calculate dynamic adjustments
                    adjustment = self.analyzer.calculate_dynamic_levels(signal, df, evaluation)
                    
                    # Execute adjustment
                    if adjustment['action'] == 'exit':
                        # Close position immediately
                        await self.close_real_position(signal, adjustment['reason'])
                        current_price = evaluation['current_price']
                        success = self.database.close_signal(
                            signal['signal_id'], 
                            current_price,
                            f"Dynamic Exit: {adjustment['reason']}"
                        )
                        
                        if success:
                            self.active_signals.discard(signal['signal_id'])
                            self.active_coins.discard(coin)
                            
                            # Calculate P&L for logging
                            if signal['direction'] == 'LONG':
                                pnl_pct = ((current_price - signal['entry_price']) / signal['entry_price']) * 100 * 10
                            else:
                                pnl_pct = ((signal['entry_price'] - current_price) / signal['entry_price']) * 100 * 10
                            
                            await self.broadcast_to_clients({
                                "type": "position_adjusted",
                                "action": "emergency_exit",
                                "signal_id": signal['signal_id'],
                                "coin": coin,
                                "reason": adjustment['reason'],
                                "pnl_pct": round(pnl_pct, 2),
                                "health_score": round(evaluation['health_score'], 2)
                            })
                            
                            logger.warning(f"Emergency exit {coin}: {adjustment['reason']} "
                                          f"(Health: {evaluation['health_score']:.2f}, P&L: {pnl_pct:.2f}%)")
                    
                    elif adjustment['action'] == 'update':
                        # Update stop loss and/or take profit
                        new_sl = adjustment.get('new_sl')
                        new_tp = adjustment.get('new_tp')
                        
                        # Only update if levels actually changed meaningfully (>0.1% difference)
                        sl_changed = abs(new_sl - signal['stop_loss']) / signal['stop_loss'] > 0.001
                        tp_changed = abs(new_tp - signal['take_profit']) / signal['take_profit'] > 0.001
                        
                        if sl_changed or tp_changed:
                            # Update in database
                            with self.database.get_db_connection() as conn:
                                cursor = conn.cursor()
                                cursor.execute('''
                                UPDATE signals 
                                SET stop_loss = ?, take_profit = ?, updated_at = CURRENT_TIMESTAMP
                                WHERE signal_id = ? AND status = 'active'
                                ''', (new_sl, new_tp, signal['signal_id']))
                            
                            await self.broadcast_to_clients({
                                "type": "position_adjusted",
                                "action": "levels_updated",
                                "signal_id": signal['signal_id'],
                                "coin": coin,
                                "old_sl": signal['stop_loss'],
                                "new_sl": new_sl,
                                "old_tp": signal['take_profit'],
                                "new_tp": new_tp,
                                "reason": adjustment['reason'],
                                "health_score": round(evaluation['health_score'], 2)
                            })
                            
                            logger.info(f"Adjusted {coin} levels: SL {signal['stop_loss']:.6f} -> {new_sl:.6f}, "
                                       f"TP {signal['take_profit']:.6f} -> {new_tp:.6f} ({adjustment['reason']})")
                    
                    # Small delay between evaluations
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error evaluating position {signal['signal_id']}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error in position evaluation cycle: {e}")

    def update_ml_prediction_accuracy(self, signal: Dict, was_successful: bool):
        """Update ML prediction accuracy tracking"""
        try:
            coin = signal['coin']
            
            if coin in self.model_performance:
                if was_successful:
                    self.model_performance[coin]['correct_predictions'] += 1
                
                # Update accuracy
                total_predictions = self.model_performance[coin]['prediction_count']
                correct_predictions = self.model_performance[coin]['correct_predictions']
                
                if total_predictions > 0:
                    self.model_performance[coin]['accuracy'] = correct_predictions / total_predictions
            
        except Exception as e:
            logger.error(f"Error updating ML prediction accuracy: {e}")

    async def remove_from_grace_period(self, signal_id: str, delay: int):
        """Remove signal from grace period"""
        await asyncio.sleep(delay)
        self.active_signals.discard(signal_id)

    async def position_monitoring_cycle(self):
        """Dedicated cycle for monitoring and adjusting active positions"""
        monitor_interval = 180  # Check every 3 minutes
        
        while self.running:
            try:
                await asyncio.sleep(monitor_interval)
                
                if not self.running:
                    break
                
                # Only run if we have active positions and not currently scanning
                active_signals = self.database.get_active_signals()
                if active_signals and not self.is_scanning:
                    logger.info("Running position health check...")
                    await self.evaluate_and_adjust_positions()
                    
            except Exception as e:
                logger.error(f"Error in position monitoring cycle: {e}")

    def get_ml_enhanced_dashboard(self) -> str:
        """Return ML-enhanced dashboard HTML"""
        return '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ML-Enhanced Crypto Trading Bot</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { 
                    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
                    color: #ffffff;
                    min-height: 100vh;
                    overflow-x: hidden;
                }
                .container { max-width: 1800px; margin: 0 auto; padding: 20px; }
                
                /* Enhanced Header for ML */
                .header {
                    text-align: center;
                    margin-bottom: 40px;
                    position: relative;
                }
                .header::before {
                    content: '';
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    width: 400px;
                    height: 400px;
                    background: radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, transparent 70%);
                    border-radius: 50%;
                    z-index: -1;
                }
                .header h1 {
                    font-size: 3.8rem;
                    font-weight: 900;
                    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 50%, #1e40af 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    text-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
                    margin-bottom: 10px;
                }
                .header .subtitle {
                    font-size: 1.3rem;
                    color: #94a3b8;
                    font-weight: 600;
                    opacity: 0.9;
                }
                .ml-badge {
                    display: inline-block;
                    background: linear-gradient(135deg, #8b5cf6, #7c3aed);
                    color: white;
                    padding: 8px 20px;
                    border-radius: 25px;
                    font-size: 0.9rem;
                    font-weight: 700;
                    margin-top: 15px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
                }
                
                /* Enhanced Status Grid */
                .status-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                    gap: 25px;
                    margin-bottom: 40px;
                }
                .status-card {
                    background: rgba(30, 41, 59, 0.4);
                    backdrop-filter: blur(20px);
                    border-radius: 20px;
                    padding: 30px;
                    border: 1px solid rgba(59, 130, 246, 0.2);
                    position: relative;
                    overflow: hidden;
                    transition: all 0.3s ease;
                }
                .status-card:hover {
                    transform: translateY(-5px);
                    border-color: rgba(59, 130, 246, 0.4);
                    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
                }
                .status-card::before {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 3px;
                    background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
                }
                .status-card.ml-special::before {
                    background: linear-gradient(90deg, #8b5cf6 0%, #7c3aed 100%);
                }
                .status-card h3 {
                    color: #3b82f6;
                    font-size: 1.4rem;
                    font-weight: 700;
                    margin-bottom: 20px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                .status-card.ml-special h3 {
                    color: #8b5cf6;
                }
                .metric {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin: 15px 0;
                    padding: 12px 0;
                    border-bottom: 1px solid rgba(148, 163, 184, 0.1);
                }
                .metric:last-child { border-bottom: none; }
                .metric-label {
                    color: #cbd5e1;
                    font-weight: 500;
                }
                .metric-value {
                    color: #ffffff;
                    font-weight: 700;
                    font-size: 1.1rem;
                }
                .metric-value.positive { color: #22c55e; }
                .metric-value.negative { color: #ef4444; }
                .metric-value.ml-accent { color: #8b5cf6; }
                
                /* ML Signal Cards */
                .signals-section {
                    background: rgba(30, 41, 59, 0.4);
                    backdrop-filter: blur(20px);
                    border-radius: 20px;
                    padding: 30px;
                    border: 1px solid rgba(139, 92, 246, 0.2);
                    position: relative;
                }
                .signals-section::before {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 3px;
                    background: linear-gradient(90deg, #8b5cf6 0%, #7c3aed 100%);
                }
                .signals-header h3 {
                    color: #8b5cf6;
                    font-size: 1.6rem;
                    font-weight: 700;
                }
                .signals-count {
                    background: rgba(139, 92, 246, 0.2);
                    color: #8b5cf6;
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-weight: 600;
                    border: 1px solid rgba(139, 92, 246, 0.3);
                }
                
                /* Enhanced Signal Cards */
                .signal-card {
                    background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.4) 100%);
                    border: 1px solid rgba(139, 92, 246, 0.3);
                    border-radius: 16px;
                    padding: 25px;
                    margin: 15px 0;
                    position: relative;
                    overflow: hidden;
                    transition: all 0.3s ease;
                }
                .signal-card:hover {
                    transform: scale(1.02);
                    border-color: rgba(139, 92, 246, 0.6);
                    box-shadow: 0 15px 30px rgba(139, 92, 246, 0.1);
                }
                .ml-prediction-badge {
                    position: absolute;
                    top: 15px;
                    right: 15px;
                    background: linear-gradient(135deg, #8b5cf6, #7c3aed);
                    color: white;
                    padding: 6px 14px;
                    border-radius: 12px;
                    font-size: 0.8rem;
                    font-weight: 700;
                    box-shadow: 0 2px 10px rgba(139, 92, 246, 0.3);
                }
                .feature-importance {
                    margin-top: 15px;
                    padding: 15px;
                    background: rgba(139, 92, 246, 0.1);
                    border-radius: 12px;
                    border: 1px solid rgba(139, 92, 246, 0.2);
                }
                .feature-importance h5 {
                    color: #8b5cf6;
                    font-size: 0.9rem;
                    margin-bottom: 10px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                .feature-list {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 8px;
                }
                .feature-tag {
                    background: rgba(139, 92, 246, 0.2);
                    color: #c4b5fd;
                    padding: 4px 10px;
                    border-radius: 15px;
                    font-size: 0.75rem;
                    font-weight: 600;
                    border: 1px solid rgba(139, 92, 246, 0.3);
                }
                
                /* Status indicators */
                .status-indicator {
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 8px;
                }
                .status-indicator.running { background: #22c55e; box-shadow: 0 0 10px rgba(34, 197, 94, 0.5); }
                .status-indicator.stopped { background: #ef4444; }
                .status-indicator.scanning { background: #3b82f6; animation: pulse 1.5s infinite; }
                .status-indicator.training { background: #8b5cf6; animation: pulse 1.5s infinite; }
                
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }
                
                /* Empty state */
                .empty-state {
                    text-align: center;
                    padding: 60px 20px;
                    color: #64748b;
                }
                .empty-state h4 {
                    font-size: 1.3rem;
                    margin-bottom: 10px;
                    color: #94a3b8;
                }
                
                /* Responsive */
                @media (max-width: 768px) {
                    .container { padding: 15px; }
                    .header h1 { font-size: 2.8rem; }
                    .status-grid { grid-template-columns: 1fr; gap: 20px; }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>ML Trading Bot</h1>
                    <div class="subtitle">Advanced Feature Engineering • Orthogonalization • Time Series Validation</div>
                    <div class="ml-badge">Machine Learning Enhanced</div>
                </div>
                
                <div class="status-grid">
                    <div class="status-card">
                        <h3>🤖 System Status</h3>
                        <div class="metric">
                            <span class="metric-label">Bot Status:</span>
                            <span class="metric-value" id="bot-status">
                                <span class="status-indicator" id="status-dot"></span>
                                <span id="status-text">Loading...</span>
                            </span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Analysis Type:</span>
                            <span class="metric-value ml-accent">ML Enhanced</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Active Signals:</span>
                            <span class="metric-value" id="signal-count">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Scan Interval:</span>
                            <span class="metric-value">10 minutes</span>
                        </div>
                    </div>
                    
                    <div class="status-card ml-special">
                        <h3>🧠 ML Models</h3>
                        <div class="metric">
                            <span class="metric-label">Models Trained:</span>
                            <span class="metric-value" id="models-trained">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Avg Accuracy:</span>
                            <span class="metric-value" id="ml-accuracy">0%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Training Status:</span>
                            <span class="metric-value" id="training-status">Idle</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Last Training:</span>
                            <span class="metric-value" id="last-training">Never</span>
                        </div>
                    </div>
                    
                    <div class="status-card">
                        <h3>📊 Performance</h3>
                        <div class="metric">
                            <span class="metric-label">Total Scans:</span>
                            <span class="metric-value" id="total-scans">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Win Rate:</span>
                            <span class="metric-value" id="win-rate">0%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Sharpe Ratio:</span>
                            <span class="metric-value" id="sharpe-ratio">0.0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Model Confidence:</span>
                            <span class="metric-value ml-accent" id="model-confidence">0%</span>
                        </div>
                    </div>
                    
                    <div class="status-card">
                        <h3>💰 Portfolio</h3>
                        <div class="metric">
                            <span class="metric-label">Total Balance:</span>
                            <span class="metric-value" id="total-balance">$1,000</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Open P&L:</span>
                            <span class="metric-value" id="open-pnl">$0.00</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Total Trades:</span>
                            <span class="metric-value" id="total-trades">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Quality Score:</span>
                            <span class="metric-value ml-accent" id="quality-score">0%</span>
                        </div>
                    </div>
                </div>
                
                <div class="signals-section">
                    <div class="signals-header">
                        <h3>🎯 ML Predictions</h3>
                        <div class="signals-count" id="active-count">0 Active</div>
                    </div>
                    <div id="signals-list">
                        <div class="empty-state">
                            <h4>No Active ML Signals</h4>
                            <p>Training models and analyzing market patterns...</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                class MLTradingBotDashboard {
                    constructor() {
                        this.wsReconnectAttempts = 0;
                        this.maxReconnectAttempts = 5;
                        this.reconnectDelay = 3000;
                        this.connectWebSocket();
                        this.updateDashboard();
                        this.startUpdateLoop();
                    }
                    
                    connectWebSocket() {
                        try {
                            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                            const wsUrl = `${protocol}//${window.location.host}/ws`;
                            
                            this.ws = new WebSocket(wsUrl);
                            
                            this.ws.onopen = () => {
                                console.log('ML WebSocket connected');
                                this.wsReconnectAttempts = 0;
                            };
                            
                            this.ws.onmessage = (event) => {
                                try {
                                    const data = JSON.parse(event.data);
                                    this.handleWebSocketMessage(data);
                                } catch (e) {
                                    console.error('Error parsing WebSocket message:', e);
                                }
                            };
                            
                            this.ws.onclose = () => {
                                console.log('ML WebSocket disconnected');
                                this.reconnectWebSocket();
                            };
                            
                            this.ws.onerror = (error) => {
                                console.error('ML WebSocket error:', error);
                            };
                            
                        } catch (error) {
                            console.error('Error connecting ML WebSocket:', error);
                        }
                    }
                    
                    reconnectWebSocket() {
                        if (this.wsReconnectAttempts < this.maxReconnectAttempts) {
                            this.wsReconnectAttempts++;
                            setTimeout(() => {
                                console.log(`Reconnecting ML WebSocket... Attempt ${this.wsReconnectAttempts}`);
                                this.connectWebSocket();
                            }, this.reconnectDelay * this.wsReconnectAttempts);
                        }
                    }
                    
                    handleWebSocketMessage(data) {
                        switch (data.type) {
                            case 'new_ml_signal':
                                this.showNotification('New ML Signal!', 
                                    `${data.signal.coin} ${data.signal.direction} (Conf: ${(data.model_confidence * 100).toFixed(1)}%)`, 'success');
                                this.updateDashboard();
                                break;
                            case 'ml_signal_closed':
                                const pnlClass = data.pnl_usd >= 0 ? 'success' : 'error';
                                this.showNotification('ML Signal Closed!', 
                                    `${data.coin} - ${data.exit_reason}: ${data.pnl_usd}`, pnlClass);
                                this.updateDashboard();
                                break;
                            case 'ml_scan_started':
                                this.updateScanStatus(true, false);
                                break;
                            case 'ml_scan_completed':
                                this.updateScanStatus(false, false);
                                this.updateDashboard();
                                break;
                            case 'model_training_started':
                                this.updateScanStatus(false, true);
                                this.showNotification('Model Training', 'ML models are being retrained', 'info');
                                break;
                            case 'model_training_completed':
                                this.updateScanStatus(false, false);
                                this.showNotification('Training Complete', 
                                    `${data.models_retrained} models retrained`, 'success');
                                this.updateDashboard();
                                break;
                        }
                    }
                    
                    showNotification(title, message, type = 'info') {
                        if ('Notification' in window && Notification.permission === 'granted') {
                            new Notification(title, {
                                body: message,
                                icon: '/favicon.ico'
                            });
                        }
                    }
                    
                    updateScanStatus(isScanning, isTraining) {
                        const statusDot = document.getElementById('status-dot');
                        const statusText = document.getElementById('status-text');
                        const trainingStatus = document.getElementById('training-status');
                        
                        if (isTraining) {
                            statusDot.className = 'status-indicator training';
                            statusText.textContent = 'Training';
                            trainingStatus.textContent = 'Training Models';
                        } else if (isScanning) {
                            statusDot.className = 'status-indicator scanning';
                            statusText.textContent = 'ML Scanning';
                            trainingStatus.textContent = 'Idle';
                        } else {
                            trainingStatus.textContent = 'Idle';
                        }
                    }
                    
                    async updateDashboard() {
                        try {
                            const [signalsResponse, systemResponse, portfolioResponse, modelsResponse] = await Promise.all([
                                fetch('/api/signals'),
                                fetch('/api/system'),
                                fetch('/api/portfolio'),
                                fetch('/api/ml/models').catch(() => ({ json: () => ({ models: {}, global_stats: {} }) }))
                            ]);
                            
                            const signalsData = await signalsResponse.json();
                            const systemData = await systemResponse.json();
                            const portfolioData = await portfolioResponse.json();
                            const modelsData = await modelsResponse.json();
                            
                            this.updateSystemStatus(systemData);
                            this.updateMLModelStatus(modelsData);
                            this.updatePerformanceMetrics(systemData.performance_metrics || {});
                            this.updatePortfolioData(portfolioData.portfolio || {});
                            this.updateMLSignalsList(signalsData.signals || []);
                            
                        } catch (error) {
                            console.error('ML Dashboard update error:', error);
                        }
                    }
                    
                    updateSystemStatus(data) {
                        const statusText = document.getElementById('status-text');
                        const statusDot = document.getElementById('status-dot');
                        
                        if (data.is_running) {
                            if (data.is_training) {
                                statusText.textContent = 'Training';
                                statusDot.className = 'status-indicator training';
                            } else if (data.is_scanning) {
                                statusText.textContent = 'ML Scanning';
                                statusDot.className = 'status-indicator scanning';
                            } else {
                                statusText.textContent = 'Running';
                                statusDot.className = 'status-indicator running';
                            }
                        } else {
                            statusText.textContent = 'Stopped';
                            statusDot.className = 'status-indicator stopped';
                        }
                        
                        document.getElementById('signal-count').textContent = data.active_signals_count || 0;
                        
                        // Format uptime
                        const uptime = data.uptime_seconds || 0;
                        const hours = Math.floor(uptime / 3600);
                        const minutes = Math.floor((uptime % 3600) / 60);
                        // Update other system metrics as needed
                    }
                    
                    updateMLModelStatus(data) {
                        const modelsCount = Object.keys(data.models || {}).length;
                        const globalStats = data.global_stats || {};
                        
                        document.getElementById('models-trained').textContent = modelsCount;
                        document.getElementById('ml-accuracy').textContent = `${(globalStats.avg_accuracy * 100).toFixed(1)}%`;
                        document.getElementById('training-status').textContent = globalStats.training_status === 'active' ? 'Training' : 'Idle';
                        
                        const lastTraining = globalStats.last_training_session;
                        if (lastTraining) {
                            const date = new Date(lastTraining);
                            document.getElementById('last-training').textContent = date.toLocaleTimeString();
                        } else {
                            document.getElementById('last-training').textContent = 'Never';
                        }
                    }
                    
                    updatePerformanceMetrics(metrics) {
                        document.getElementById('total-scans').textContent = metrics.total_scans || 0;
                        document.getElementById('win-rate').textContent = `${((metrics.win_rate || 0) * 100).toFixed(1)}%`;
                        document.getElementById('sharpe-ratio').textContent = (metrics.sharpe_ratio || 0).toFixed(2);
                        document.getElementById('model-confidence').textContent = `${((metrics.prediction_confidence_avg || 0) * 100).toFixed(1)}%`;
                    }
                    
                    updatePortfolioData(portfolio) {
                        const totalBalance = document.getElementById('total-balance');
                        const openPnl = document.getElementById('open-pnl');
                        const qualityScore = document.getElementById('quality-score');
                        
                        totalBalance.textContent = `${(portfolio.total_balance || 1000).toLocaleString()}`;
                        
                        const pnlValue = portfolio.open_pnl || 0;
                        openPnl.textContent = `${pnlValue.toFixed(2)}`;
                        openPnl.className = pnlValue >= 0 ? 'metric-value positive' : 'metric-value negative';
                        
                        document.getElementById('total-trades').textContent = portfolio.total_trades || 0;
                        document.getElementById('quality-score').textContent = 
                            `${((portfolio.signal_quality_score || 0) * 100).toFixed(1)}%`;
                    }
                    
                    updateMLSignalsList(signals) {
                        const signalsList = document.getElementById('signals-list');
                        const activeCount = document.getElementById('active-count');
                        
                        activeCount.textContent = `${signals.length} Active`;
                        
                        if (signals.length === 0) {
                            signalsList.innerHTML = `
                                <div class="empty-state">
                                    <h4>No Active ML Signals</h4>
                                    <p>Training models and analyzing market patterns...</p>
                                </div>
                            `;
                            return;
                        }
                        
                        signalsList.innerHTML = signals.map(signal => {
                            const pnlClass = signal.live_pnl_usd >= 0 ? 'pnl-positive' : 'pnl-negative';
                            const mlPrediction = signal.ml_prediction || 0;
                            const modelConfidence = signal.model_confidence || 0;
                            const topFeatures = signal.feature_importance_top3 || {};
                            
                            return `
                                <div class="signal-card">
                                    <div class="ml-prediction-badge">
                                        ${(mlPrediction * 100).toFixed(1)}% Pred
                                    </div>
                                    <div class="signal-header">
                                        <div class="signal-coin">${signal.coin}</div>
                                        <div class="signal-direction ${signal.direction}">${signal.direction}</div>
                                    </div>
                                    <div class="signal-metrics">
                                        <div class="signal-metric">
                                            <div class="signal-metric-label">Entry</div>
                                            <div class="signal-metric-value">${signal.entry_price.toFixed(4)}</div>
                                        </div>
                                        <div class="signal-metric">
                                            <div class="signal-metric-label">Current</div>
                                            <div class="signal-metric-value">${signal.current_price.toFixed(4)}</div>
                                        </div>
                                        <div class="signal-metric">
                                            <div class="signal-metric-label">P&L</div>
                                            <div class="signal-metric-value ${pnlClass}">${signal.live_pnl_usd.toFixed(2)}</div>
                                        </div>
                                        <div class="signal-metric">
                                            <div class="signal-metric-label">ML Confidence</div>
                                            <div class="signal-metric-value ml-accent">${(modelConfidence * 100).toFixed(1)}%</div>
                                        </div>
                                        <div class="signal-metric">
                                            <div class="signal-metric-label">TP Progress</div>
                                            <div class="signal-metric-value">${signal.tp_progress.toFixed(1)}%</div>
                                            <div class="progress-container">
                                                <div class="progress-bar tp" style="width: ${Math.min(100, signal.tp_progress)}%"></div>
                                            </div>
                                        </div>
                                        <div class="signal-metric">
                                            <div class="signal-metric-label">R:R Ratio</div>
                                            <div class="signal-metric-value">${(signal.analysis_data?.risk_reward_ratio || 0).toFixed(1)}</div>
                                        </div>
                                    </div>
                                    ${Object.keys(topFeatures).length > 0 ? `
                                        <div class="feature-importance">
                                            <h5>Top ML Features</h5>
                                            <div class="feature-list">
                                                ${Object.entries(topFeatures).map(([feature, importance]) => 
                                                    `<span class="feature-tag">${feature.replace(/_/g, ' ')}: ${(importance * 100).toFixed(1)}%</span>`
                                                ).join('')}
                                            </div>
                                        </div>
                                    ` : ''}
                                </div>
                            `;
                        }).join('');
                    }
                    
                    startUpdateLoop() {
                        // Update dashboard every 5 seconds
                        setInterval(() => {
                            this.updateDashboard();
                        }, 5000);
                    }
                }
                
                // Initialize ML dashboard when DOM is loaded
                document.addEventListener('DOMContentLoaded', () => {
                    new MLTradingBotDashboard();
                    
                    // Request notification permission
                    if ('Notification' in window && Notification.permission === 'default') {
                        Notification.requestPermission();
                    }
                });
            </script>
        </body>
        </html>
        '''

# Global ML bot instance
ml_bot_instance = None

def get_ml_bot():
    """Get or create ML bot instance"""
    global ml_bot_instance
    if ml_bot_instance is None:
        ml_bot_instance = MLTradingBot()
    return ml_bot_instance

# FastAPI app
app = get_ml_bot().app

@app.on_event("startup")
async def startup_event():
    """Start ML bot on server startup"""
    bot = get_ml_bot()
    if not bot.running:
        bot.running = True
        asyncio.create_task(bot.ml_scan_cycle())
        asyncio.create_task(bot.ml_retrain_cycle())
        asyncio.create_task(bot.position_monitoring_cycle())
        logger.info("ML trading bot auto-started with advanced features")

@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown with model saving"""
    bot = get_ml_bot()
    bot.running = False
    bot.analyzer.save_models()
    logger.info("ML trading bot shutdown complete - models saved")

if __name__ == "__main__":
    try:
        # Ensure directories exist
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        logger.info("Starting ML-Enhanced Crypto Trading Bot Server...")
        logger.info("Features: Feature Engineering, Orthogonalization, Time Series Validation, Explainable AI")
        
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Graceful shutdown initiated - saving ML models...")
        if ml_bot_instance:
            ml_bot_instance.running = False
            ml_bot_instance.analyzer.save_models()
    except Exception as e:
        logger.error(f"Failed to start ML server: {e}")
        sys.exit(1)