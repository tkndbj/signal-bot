#!/usr/bin/env python3
"""
Simplified ML-Enhanced Crypto Trading Bot for Bybit
Production-ready version with clean architecture
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import our modules
from database import Database
from enhanced_trading_analyzer_v2 import MLTradingAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        # Core components
        self.database = Database()
        self.analyzer = MLTradingAnalyzer(database=self.database)
        
        # Configuration
        self.position_check_interval = 60  # Check positions every minute
        self.scan_interval = 120  # Scan for signals every 10 minutes
        self.max_concurrent_positions = 8
        self.position_size_percent = 0.15  # Use 15% of balance
        self.leverage = 15
        self.min_confidence_threshold = 0.65
        
        # FastAPI setup
        self.app = FastAPI(
            title="ML Crypto Trading Bot",
            version="2.0.0",
            description="Simplified production-ready trading bot for Bybit"
        )
        
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
        self.last_scan_time = None
        self.scan_count = 0
        self.startup_time = datetime.now()
        
        # Signal management - simplified
        self.active_positions: Dict[str, Dict] = {}  # symbol -> position data
        
        # Performance metrics
        self.performance_metrics = {
            'total_scans': 0,
            'signals_generated': 0,
            'win_rate': 0,
            'total_pnl': 0
        }
        
        # Setup
        self.setup_routes()
        self.initialize_bot_state()
        
        # Load ML models if available
        self.analyzer.load_models()
        
        # Graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle graceful shutdown"""
        logger.info(f"Received signal {signum}. Shutting down...")
        self.analyzer.save_models()
        self.running = False
        sys.exit(0)

    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to consistent format (e.g., BTCUSDT)"""
        # Remove any slashes, colons, or /USDT suffixes
        symbol = symbol.replace('/', '').replace(':', '')
        
        # Ensure it ends with USDT if it doesn't already
        if not symbol.endswith('USDT'):
            symbol = symbol + 'USDT'
        
        return symbol

    def initialize_bot_state(self):
        """Initialize bot state and clean up stale signals"""
        try:
            logger.info("Initializing bot state...")
            
            # Clean stale signals (older than 24 hours)
            active_signals = self.database.get_active_signals()
            current_time = datetime.now()
            stale_count = 0
            
            for signal in active_signals:
                try:
                    signal_time = datetime.fromisoformat(signal['timestamp'].replace('Z', '+00:00'))
                    if (current_time - signal_time).total_seconds() > 86400:  # 24 hours
                        self.database.close_signal(
                            signal['signal_id'],
                            signal['entry_price'],
                            "System cleanup - stale signal"
                        )
                        stale_count += 1
                except Exception as e:
                    logger.error(f"Error cleaning signal {signal['signal_id']}: {e}")
            
            if stale_count > 0:
                logger.info(f"Cleaned {stale_count} stale signals")
            
            # Schedule initial sync after startup
            asyncio.create_task(self.initial_sync())
            
            # Clean old data
            self.database.clean_old_data(days=30)
            
            logger.info("Bot initialization complete")
            
        except Exception as e:
            logger.error(f"Error during bot initialization: {e}")

    async def initial_sync(self):
        """Initial sync with exchange after startup"""
        await asyncio.sleep(3)  # Wait for everything to initialize
        logger.info("Running initial position sync...")
        await self.sync_positions_with_exchange()

    async def get_account_balance(self) -> Dict:
        """Get real account balance from Bybit"""
        try:
            balance = self.analyzer.exchange.fetch_balance({'accountType': 'UNIFIED'})
            
            if 'USDT' in balance:
                total = float(balance['USDT'].get('total', 0) or 0)
                free = float(balance['USDT'].get('free', 0) or total)
                used = float(balance['USDT'].get('used', 0) or 0)
                
                if free == 0 and total > 0:
                    free = total
                
                return {
                    'free': free,
                    'used': used,
                    'total': total
                }
            
            logger.warning("USDT not found in balance response")
            return {'free': 0, 'used': 0, 'total': 0}
            
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {'free': 0, 'used': 0, 'total': 0}

    async def sync_positions_with_exchange(self):
        """Sync database with actual Bybit positions"""
        try:
            # Get all positions from Bybit
            positions = self.analyzer.exchange.fetch_positions(params={'category': 'linear'})
            
            # Map of open positions on Bybit
            open_positions = {}
            for pos in positions:
                if pos['contracts'] > 0:
                    symbol = self.normalize_symbol(pos['symbol'])
                    open_positions[symbol] = {
                        'contracts': pos['contracts'],
                        'side': pos['side'],
                        'unrealizedPnl': pos.get('unrealizedPnl', 0),
                        'markPrice': pos.get('markPrice', 0),
                        'entryPrice': pos.get('avgPrice', pos.get('markPrice', 0))
                    }
                    logger.info(f"Bybit position found: {symbol} - {pos['contracts']} contracts")
            
            # Update active_positions
            self.active_positions = open_positions
            
            # Get active signals from database
            db_signals = self.database.get_active_signals()
            
            # Check which database signals are no longer on exchange
            for signal in db_signals:
                # Fix double USDT issue
                coin = signal['coin']
                if coin.endswith('USDT'):
                    coin = coin.replace('USDT', '')
                coin_symbol = self.normalize_symbol(coin)
                
                if coin_symbol not in open_positions:
                    # Position closed on exchange but still active in DB
                    logger.info(f"Closing orphaned signal: {coin_symbol}")
                    
                    # Get exit price
                    try:
                        ticker = self.analyzer.exchange.fetch_ticker(coin_symbol)
                        exit_price = ticker['last']
                    except:
                        exit_price = signal.get('current_price', signal['entry_price'])
                    
                    # Close in database
                    self.database.close_signal(
                        signal['signal_id'],
                        exit_price,
                        "Position closed on exchange"
                    )
            
            # Check for positions on exchange not in database
            for symbol, pos_data in open_positions.items():
                found_in_db = False
                for signal in db_signals:
                    if self.normalize_symbol(signal['coin']) == symbol:
                        found_in_db = True
                        break
                
                if not found_in_db:
                    logger.warning(f"Found untracked position: {symbol}")
                    # Create tracking entry
                    signal_data = {
                        'signal_id': f"SYNC_{symbol}_{int(time.time())}",
                        'timestamp': datetime.now().isoformat(),
                        'coin': symbol,
                        'direction': 'LONG' if pos_data['side'] == 'long' else 'SHORT',
                        'entry_price': pos_data['entryPrice'],
                        'take_profit': pos_data['entryPrice'] * 1.02,
                        'stop_loss': pos_data['entryPrice'] * 0.98,
                        'confidence': 50,
                        'position_value': pos_data['contracts'] * pos_data['markPrice'],
                        'analysis_data': {'synced_from_exchange': True}
                    }
                    self.database.save_signal(signal_data)
            
            return open_positions
            
        except Exception as e:
            logger.error(f"Error syncing positions: {e}")
            return {}

    async def execute_trade(self, signal_data: Dict) -> bool:
        """Execute real trade on Bybit"""
        try:
            symbol = self.normalize_symbol(signal_data['coin'])
            signal_data['coin'] = symbol  # Update to normalized symbol
            
            logger.info(f"Executing trade for {symbol}")
            
            # Get balance
            balance_info = await self.get_account_balance()
            available_balance = balance_info['free']
            
            if available_balance < 10:
                logger.warning(f"Insufficient balance: ${available_balance}")
                return False
            
            # Calculate position size
            account_equity = available_balance * self.position_size_percent
            position_value = account_equity * self.leverage
            
            # Get current price
            ticker = self.analyzer.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Get symbol info for precision
            symbol_info = self.analyzer.exchange.market(symbol)
            
            min_qty = float(symbol_info.get('limits', {}).get('amount', {}).get('min', 0.001))
            qty_precision = int(symbol_info.get('precision', {}).get('amount', 8))
            
            # Calculate quantity
            base_quantity = position_value / current_price
            
            # Apply precision
            if qty_precision == 0:
                quantity = max(int(base_quantity), 1)
            else:
                quantity = round(base_quantity, qty_precision)
            
            # Ensure minimum quantity
            if quantity < min_qty:
                quantity = min_qty
            
            final_position_value = quantity * current_price
            logger.info(f"Position: {quantity} contracts = ${final_position_value:.2f}")
            
            # Set leverage
            try:
                self.analyzer.exchange.set_leverage(self.leverage, symbol)
            except:
                pass  # Some symbols might have fixed leverage
            
            # Place order
            side = 'buy' if signal_data['direction'] == 'LONG' else 'sell'

            # For market buy orders, Bybit via CCXT requires price to calculate cost
            if side == 'buy':
                order = self.analyzer.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=side,
                    amount=quantity,
                    price=current_price,  # Required for market buy on Bybit
                    params={
                        'positionIdx': 0,
                        'timeInForce': 'IOC',
                        'orderType': 'Market',
                        'category': 'linear',
                        'createMarketBuyOrderRequiresPrice': False  # Tell CCXT amount is in base currency
                    }
                )
            else:
                order = self.analyzer.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=side,
                    amount=quantity,
                    params={
                        'positionIdx': 0,
                        'timeInForce': 'IOC',
                        'orderType': 'Market',
                        'category': 'linear'
                    }
                )
            
            # Wait for order to process
            await asyncio.sleep(2)
            
            # Check if order filled
            if order and (order.get('status') in ['closed', 'filled'] or order.get('filled', 0) > 0):
                # Set stop loss and take profit
                await self.set_sl_tp(
                    symbol,
                    signal_data['direction'],
                    quantity,
                    signal_data['stop_loss'],
                    signal_data['take_profit']
                )
                
                logger.info(f"✓ Trade executed: {symbol} {side} - Qty: {quantity}, Value: ${final_position_value:.2f}")
                
                # Update signal data
                signal_data['order_id'] = order.get('id')
                signal_data['executed_price'] = order.get('average', current_price)
                signal_data['executed_quantity'] = quantity
                signal_data['position_value'] = final_position_value
                
                # Update active positions
                self.active_positions[symbol] = {
                    'signal_id': signal_data['signal_id'],
                    'direction': signal_data['direction'],
                    'entry_price': signal_data['executed_price'],
                    'quantity': quantity,
                    'position_value': final_position_value
                }
                
                return True
            else:
                logger.error(f"Order failed: {order}")
                return False
                
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False

    async def set_sl_tp(self, symbol: str, direction: str, quantity: float, 
                    stop_loss: float, take_profit: float) -> bool:
        """Set stop loss and take profit orders"""
        try:
            await asyncio.sleep(3)  # Wait for position to register
        
            # Use the correct v5 API endpoint with proper parameters
            params = {
                'category': 'linear',
                'symbol': symbol,
                'stopLoss': str(stop_loss),
                'takeProfit': str(take_profit),
                'tpTriggerBy': 'LastPrice',
                'slTriggerBy': 'LastPrice',
                'tpslMode': 'Full',
                'positionIdx': 0
            }
        
            # Use the correct API method
            response = self.analyzer.exchange.private_post_v5_position_trading_stop(params)
        
            if response and response.get('retCode') == 0:
                logger.info(f"SL/TP set successfully for {symbol} - TP: {take_profit}, SL: {stop_loss}")
                return True
            else:
                logger.error(f"Failed to set SL/TP for {symbol}: {response}")
                return False
        
        except Exception as e:
            logger.error(f"Error setting SL/TP for {symbol}: {e}")
            return False

    async def close_position(self, signal_data: Dict, reason: str) -> bool:
        """Close position on Bybit"""
        try:
            symbol = self.normalize_symbol(signal_data['coin'])
            
            # Get current position
            positions = self.analyzer.exchange.fetch_positions([symbol], params={'category': 'linear'})
            position = next((p for p in positions if p['symbol'] == symbol), None)
            
            if not position or position['contracts'] == 0:
                logger.warning(f"No open position for {symbol}")
                return False
            
            # Close position
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
            
            logger.info(f"Position closed: {symbol} - Reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False

    def validate_signal(self, signal_data: Dict) -> bool:
        """Validate signal before saving/executing"""
        try:
            # Normalize symbol
            signal_data['coin'] = self.normalize_symbol(signal_data['coin'])
            
            # Check required fields
            required_fields = ['signal_id', 'coin', 'direction', 'entry_price', 'take_profit', 'stop_loss']
            for field in required_fields:
                if field not in signal_data:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            entry = float(signal_data['entry_price'])
            tp = float(signal_data['take_profit'])
            sl = float(signal_data['stop_loss'])
            direction = signal_data['direction']
            
            # Validate prices
            if entry <= 0 or tp <= 0 or sl <= 0:
                logger.error("Invalid price (negative or zero)")
                return False
            
            # Validate price relationships
            if direction == 'LONG':
                if not (sl < entry < tp):
                    logger.error("Invalid LONG signal prices")
                    return False
            else:  # SHORT
                if not (tp < entry < sl):
                    logger.error("Invalid SHORT signal prices")
                    return False
            
            # Check if already have position for this symbol
            symbol = signal_data['coin']
            if symbol in self.active_positions:
                logger.warning(f"Already have position for {symbol}")
                return False
            
            # Check ML confidence
            model_confidence = signal_data.get('model_confidence')
            if model_confidence is None:
                model_confidence = signal_data.get('confidence', 0) / 100.0
    
            if model_confidence is not None and model_confidence < self.min_confidence_threshold:
                logger.warning(f"Model confidence too low: {model_confidence}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False

    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def dashboard():
            """Serve dashboard"""
            return HTMLResponse(content=self.get_dashboard_html())
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint"""
            await websocket.accept()
            self.websocket_connections.add(websocket)
            
            try:
                await websocket.send_text(json.dumps({
                    "type": "connection_established",
                    "message": "Connected to Trading Bot",
                    "timestamp": datetime.now().isoformat()
                }))
                
                while True:
                    await asyncio.sleep(30)
                    await websocket.send_text(json.dumps({
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat()
                    }))
                    
            except WebSocketDisconnect:
                self.websocket_connections.discard(websocket)
            except Exception as e:
                logger.debug(f"WebSocket error: {e}")
                self.websocket_connections.discard(websocket)

        @self.app.get("/api/signals")
        async def get_signals():
            """Get active signals"""
            try:
                signals = self.database.get_active_signals()
                
                # Enhance with current prices
                for signal in signals:
                    try:
                        symbol = self.normalize_symbol(signal['coin'])
                        current_price = await self.analyzer.get_current_price(symbol)
                        
                        if current_price > 0:
                            signal['current_price'] = current_price
                            
                            # Calculate P&L
                            position_value = signal.get('position_value', 0)
                            if position_value > 0:
                                if signal['direction'] == 'LONG':
                                    pnl_pct = ((current_price - signal['entry_price']) / signal['entry_price']) * 100 * self.leverage
                                else:
                                    pnl_pct = ((signal['entry_price'] - current_price) / signal['entry_price']) * 100 * self.leverage
                                
                                pnl_usd = (pnl_pct / 100) * position_value
                                
                                signal['live_pnl_usd'] = round(pnl_usd, 2)
                                signal['live_pnl_percentage'] = round(pnl_pct, 2)
                    except Exception as e:
                        logger.error(f"Error enhancing signal {signal['signal_id']}: {e}")
                
                return JSONResponse(content={"signals": signals, "count": len(signals)})
                
            except Exception as e:
                logger.error(f"Error getting signals: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.get("/api/portfolio")
        async def get_portfolio():
            """Get portfolio statistics"""
            try:
                portfolio = self.database.get_portfolio_stats()
                
                # Get real balance
                real_balance = await self.get_account_balance()
                portfolio['real_balance'] = real_balance['total']
                portfolio['available_balance'] = real_balance['free']
                
                # Calculate open P&L from exchange
                total_open_pnl = 0
                positions = self.analyzer.exchange.fetch_positions(params={'category': 'linear'})
                for pos in positions:
                    if pos['contracts'] > 0:
                        unrealized_pnl = pos.get('unrealizedPnl', 0) or 0
                        total_open_pnl += unrealized_pnl
                
                portfolio['open_pnl'] = round(total_open_pnl, 2)
                portfolio['total_balance'] = real_balance['total'] + total_open_pnl
                
                return JSONResponse(content={"portfolio": portfolio})
                
            except Exception as e:
                logger.error(f"Error getting portfolio: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.get("/api/system")
        async def get_system_stats():
            """Get system statistics"""
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
                    "active_positions": len(self.active_positions),
                    "is_running": self.running,
                    "is_scanning": self.is_scanning,
                    "scan_interval": self.scan_interval,
                    "performance_metrics": self.performance_metrics
                })
                
            except Exception as e:
                logger.error(f"Error getting system stats: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.get("/api/trades/history")
        async def get_trade_history():
            """Get closed trades"""
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
                        position_value = trade.get('position_value', 0)
                        if position_value > 0:
                            if trade['direction'] == 'LONG':
                                pnl_pct = ((trade['exit_price'] - trade['entry_price']) / trade['entry_price']) * 100 * self.leverage
                            else:
                                pnl_pct = ((trade['entry_price'] - trade['exit_price']) / trade['entry_price']) * 100 * self.leverage
                            
                            trade['pnl_usd'] = (pnl_pct / 100) * position_value
                            trade['pnl_percentage'] = pnl_pct
                        else:
                            trade['pnl_usd'] = 0
                            trade['pnl_percentage'] = 0
                        
                        trades.append(trade)
                    
                    return JSONResponse(content={"trades": trades})
                    
            except Exception as e:
                logger.error(f"Error getting trade history: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.post("/api/control/start")
        async def start_bot():
            """Start the trading bot"""
            if not self.running:
                self.running = True
                asyncio.create_task(self.main_trading_loop())
                asyncio.create_task(self.position_monitor_loop())
                return JSONResponse(content={"status": "started", "message": "Trading bot started"})
            else:
                return JSONResponse(content={"status": "already_running"})

        @self.app.post("/api/control/stop")
        async def stop_bot():
            """Stop the trading bot"""
            self.running = False
            self.analyzer.save_models()
            return JSONResponse(content={"status": "stopped", "message": "Trading bot stopped"})

    async def broadcast_to_clients(self, message: Dict):
        """Broadcast message to WebSocket clients"""
        if not self.websocket_connections:
            return
        
        disconnected = set()
        message_str = json.dumps(message, default=str)
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message_str)
            except:
                disconnected.add(websocket)
        
        self.websocket_connections -= disconnected

    async def main_trading_loop(self):
        """Main trading loop - simplified"""
        while self.running:
            try:
                self.is_scanning = True
                scan_start = datetime.now()
                
                logger.info(f"Starting scan #{self.scan_count + 1}")
                
                # Sync with exchange first
                await self.sync_positions_with_exchange()
                
                # Check for exits
                await self.check_signal_exits()
                
                # Only generate new signals if we have room
                current_positions = len(self.active_positions)
                
                if current_positions < self.max_concurrent_positions:
                    # Get new signals from analyzer
                    new_signals = await self.analyzer.scan_all_coins()
                    
                    for signal_data in new_signals:
                        if len(self.active_positions) >= self.max_concurrent_positions:
                            break
                        
                        # Validate signal
                        if not self.validate_signal(signal_data):
                            continue
                        
                        # Execute trade
                        if await self.execute_trade(signal_data):
                            # Save to database
                            self.database.save_signal(signal_data)
                            
                            await self.broadcast_to_clients({
                                "type": "new_signal",
                                "signal": signal_data
                            })
                            
                            logger.info(f"New position opened: {signal_data['coin']}")
                
                # Update metrics
                self.scan_count += 1
                self.last_scan_time = datetime.now()
                self.is_scanning = False
                
                scan_duration = (datetime.now() - scan_start).total_seconds()
                logger.info(f"Scan completed in {scan_duration:.1f}s")
                
                await self.broadcast_to_clients({
                    "type": "scan_completed",
                    "scan_count": self.scan_count,
                    "duration": scan_duration
                })
                
            except Exception as e:
                self.is_scanning = False
                logger.error(f"Error in main loop: {e}")
            
            # Wait for next scan
            await asyncio.sleep(self.scan_interval)

    async def position_monitor_loop(self):
        """Monitor positions and sync with exchange"""
        while self.running:
            try:
                await asyncio.sleep(self.position_check_interval)
                
                if not self.is_scanning:
                    await self.sync_positions_with_exchange()
                    
            except Exception as e:
                logger.error(f"Error in position monitor: {e}")

    async def check_signal_exits(self):
        """Check if any signals should be closed"""
        try:
            active_signals = self.database.get_active_signals()
            
            for signal in active_signals:
                try:
                    symbol = self.normalize_symbol(signal['coin'])
                    current_price = await self.analyzer.get_current_price(symbol)
                    
                    if current_price <= 0:
                        continue
                    
                    # Update current price
                    self.database.update_signal_price(signal['signal_id'], current_price)
                    
                    # Check exit conditions
                    exit_reason = None
                    exit_price = None
                    
                    if signal['direction'] == 'LONG':
                        if current_price >= signal['take_profit']:
                            exit_reason = "Take Profit Hit"
                            exit_price = signal['take_profit']
                        elif current_price <= signal['stop_loss']:
                            exit_reason = "Stop Loss Hit"
                            exit_price = signal['stop_loss']
                    else:  # SHORT
                        if current_price <= signal['take_profit']:
                            exit_reason = "Take Profit Hit"
                            exit_price = signal['take_profit']
                        elif current_price >= signal['stop_loss']:
                            exit_reason = "Stop Loss Hit"
                            exit_price = signal['stop_loss']
                    
                    if exit_reason and exit_price:
                        # Close position
                        await self.close_position(signal, exit_reason)
                        
                        # Update database
                        self.database.close_signal(signal['signal_id'], exit_price, exit_reason)
                        
                        # Remove from active positions
                        if symbol in self.active_positions:
                            del self.active_positions[symbol]
                        
                        await self.broadcast_to_clients({
                            "type": "signal_closed",
                            "signal_id": signal['signal_id'],
                            "coin": symbol,
                            "exit_reason": exit_reason
                        })
                        
                        logger.info(f"Signal closed: {symbol} - {exit_reason}")
                        
                except Exception as e:
                    logger.error(f"Error checking exit for {signal['signal_id']}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in signal exit check: {e}")

    def get_dashboard_html(self) -> str:
        """Return dashboard HTML from templates file"""
        with open('templates/dashboard.html', 'r', encoding='utf-8') as f:
            return f.read()

# Global bot instance
bot_instance = None

def get_bot():
    """Get or create bot instance"""
    global bot_instance
    if bot_instance is None:
        bot_instance = TradingBot()
    return bot_instance

# FastAPI app
app = get_bot().app

@app.on_event("startup")
async def startup_event():
    """Start bot on server startup"""
    bot = get_bot()
    if not bot.running:
        bot.running = True
        asyncio.create_task(bot.main_trading_loop())
        asyncio.create_task(bot.position_monitor_loop())
        logger.info("Trading bot auto-started")

@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown"""
    bot = get_bot()
    bot.running = False
    bot.analyzer.save_models()
    logger.info("Trading bot shutdown complete")

if __name__ == "__main__":
    try:
        # Ensure directories exist
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        logger.info("Starting Simplified Trading Bot Server...")
        
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\nGraceful shutdown initiated...")
        if bot_instance:
            bot_instance.running = False
            bot_instance.analyzer.save_models()
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)