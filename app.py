#!/usr/bin/env python3
"""
Production-Ready ML-Enhanced Crypto Trading Bot for Bybit
Multi-Strategy: Trend Following + Funding Rate + ML Filter
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
from strategy_engine import StrategyEngine as MLTradingAnalyzer

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
        
        # ============================================================
        # RISK MANAGEMENT CONFIGURATION - PRODUCTION SETTINGS
        # ============================================================
        
        # Position sizing - CONSERVATIVE for sustainable profitability
        self.position_size_percent = 0.05  # 5% of balance per trade (was 15%)
        self.leverage = 5                   # 5x leverage (was 15x)
        self.max_concurrent_positions = 4   # Max 4 positions (was 8)
        # Max total exposure: 5% * 5 * 4 = 100% (was 1800%!)
        
        # Risk limits - Circuit breakers
        self.max_daily_loss_percent = 5.0   # Stop trading if down 5% in a day
        self.max_drawdown_percent = 15.0    # Stop trading if drawdown exceeds 15%
        self.min_confidence_threshold = 0.70  # Higher bar for signals (was 0.65)
        self.min_rr_ratio = 2.5             # Minimum risk:reward ratio
        
        # Win rate tracking for adaptive sizing
        self.recent_trades_window = 20      # Track last 20 trades
        self.min_win_rate_for_full_size = 0.40  # Reduce size if win rate below 40%
        
        # Timing intervals
        self.position_check_interval = 30   # Check positions every 30 seconds (was 60)
        self.scan_interval = 300            # Scan for signals every 5 minutes (was 120)
        self.funding_check_interval = 3600  # Check funding rates every hour
        self.model_update_interval = 7200   # Update models every 2 hours (was 1 hour)
        self.last_model_update = time.time()
        
        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_wins = 0
        self.daily_losses = 0
        self.last_daily_reset = datetime.now().date()
        self.starting_balance_today = 0.0
        
        # Trading state
        self.trading_enabled = True  # Can be disabled by risk management
        
        # ============================================================
        # FASTAPI SETUP
        # ============================================================
        
        self.app = FastAPI(
            title="Production ML Crypto Trading Bot",
            version="3.0.0",
            description="Multi-strategy trading bot: Trend Following + Funding Rate + ML Filter"
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
        
        # Active positions tracking
        self.active_positions: Dict[str, Dict] = {}
        
        # Performance metrics
        self.performance_metrics = {
            'total_scans': 0,
            'signals_generated': 0,
            'signals_filtered': 0,
            'trades_executed': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'profit_factor': 0
        }
        
        # Recent trades for adaptive sizing
        self.recent_trade_results = []
        
        # Setup
        self.setup_routes()
        self.initialize_bot_state()
        
        # Load ML models if available
        self.analyzer.load_models()
        
        # Graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Log configuration
        logger.info("=" * 70)
        logger.info("PRODUCTION TRADING BOT INITIALIZED")
        logger.info("=" * 70)
        logger.info(f"Leverage: {self.leverage}x (reduced from 15x)")
        logger.info(f"Position Size: {self.position_size_percent * 100}% (reduced from 15%)")
        logger.info(f"Max Positions: {self.max_concurrent_positions} (reduced from 8)")
        logger.info(f"Max Exposure: {self.leverage * self.position_size_percent * self.max_concurrent_positions * 100}%")
        logger.info(f"Daily Loss Limit: {self.max_daily_loss_percent}%")
        logger.info(f"Max Drawdown Limit: {self.max_drawdown_percent}%")
        logger.info(f"Min Confidence: {self.min_confidence_threshold}")
        logger.info(f"Min R:R Ratio: {self.min_rr_ratio}")
        logger.info("=" * 70)

    def signal_handler(self, signum, frame):
        """Handle graceful shutdown"""
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self.analyzer.save_models()
        self.running = False
        sys.exit(0)

    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to consistent format (e.g., BTCUSDT)"""
        symbol = symbol.replace('/', '').replace(':', '')
        
        # Fix double USDT issue
        if symbol.endswith('USDTUSDT'):
            symbol = symbol.replace('USDTUSDT', 'USDT')
        
        if not symbol.endswith('USDT'):
            symbol = symbol + 'USDT'
        
        return symbol

    def initialize_bot_state(self):
        """Initialize bot state and clean up stale signals"""
        try:
            logger.info("Initializing bot state...")
            
            # Reset daily tracking
            self._check_daily_reset()
            
            # Clean stale signals (older than 24 hours)
            active_signals = self.database.get_active_signals()
            current_time = datetime.now()
            stale_count = 0
            
            for signal in active_signals:
                try:
                    signal_time = datetime.fromisoformat(signal['timestamp'].replace('Z', '+00:00'))
                    if (current_time - signal_time).total_seconds() > 86400:
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
            
            # Load recent trade results for adaptive sizing
            self._load_recent_trade_results()
            
            # Schedule initial sync
            asyncio.create_task(self.initial_sync())
            
            # Clean old data
            self.database.clean_old_data(days=30)
            
            logger.info("Bot initialization complete")
            
        except Exception as e:
            logger.error(f"Error during bot initialization: {e}")

    def _check_daily_reset(self):
        """Reset daily tracking at midnight"""
        today = datetime.now().date()
        if today > self.last_daily_reset:
            logger.info(f"New trading day. Previous day P&L: ${self.daily_pnl:.2f}")
            logger.info(f"Previous day: {self.daily_wins}W / {self.daily_losses}L")
            
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_wins = 0
            self.daily_losses = 0
            self.last_daily_reset = today
            self.trading_enabled = True  # Re-enable trading for new day
            self.starting_balance_today = 0.0  # Will be set on first balance check

    def _load_recent_trade_results(self):
        """Load recent trade results for adaptive position sizing"""
        try:
            with self.database.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT pnl_usd FROM trade_results 
                    WHERE status = 'closed' 
                    ORDER BY updated_at DESC 
                    LIMIT ?
                ''', (self.recent_trades_window,))
                
                self.recent_trade_results = [row['pnl_usd'] for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error loading recent trades: {e}")
            self.recent_trade_results = []

    def _calculate_recent_win_rate(self) -> float:
        """Calculate win rate from recent trades"""
        if not self.recent_trade_results:
            return 0.5  # Default to 50% if no history
        
        wins = sum(1 for pnl in self.recent_trade_results if pnl > 0)
        return wins / len(self.recent_trade_results)

    def _get_adaptive_position_size(self, base_confidence: float) -> float:
        """
        Calculate adaptive position size based on:
        1. Recent win rate
        2. Signal confidence
        3. Current drawdown
        """
        base_size = self.position_size_percent
        
        # Adjust for win rate
        recent_win_rate = self._calculate_recent_win_rate()
        if recent_win_rate < self.min_win_rate_for_full_size:
            # Reduce size by up to 50% if win rate is poor
            win_rate_multiplier = 0.5 + (recent_win_rate / self.min_win_rate_for_full_size) * 0.5
            base_size *= win_rate_multiplier
            logger.info(f"Reduced position size due to win rate {recent_win_rate:.1%}")
        
        # Adjust for confidence (scale from 50% to 100% of base size)
        confidence_multiplier = 0.5 + (base_confidence * 0.5)
        base_size *= confidence_multiplier
        
        # Adjust for current drawdown
        portfolio = self.database.get_portfolio_stats()
        current_drawdown = portfolio.get('max_drawdown', 0)
        if current_drawdown > 10:
            # Reduce size if in significant drawdown
            drawdown_multiplier = max(0.5, 1 - (current_drawdown - 10) / 20)
            base_size *= drawdown_multiplier
            logger.info(f"Reduced position size due to drawdown {current_drawdown:.1f}%")
        
        return base_size

    async def initial_sync(self):
        """Initial sync with exchange after startup"""
        await asyncio.sleep(3)
        logger.info("Running initial position sync...")
        
        # Get starting balance for the day
        balance = await self.get_account_balance()
        if self.starting_balance_today == 0:
            self.starting_balance_today = balance['total']
            logger.info(f"Starting balance today: ${self.starting_balance_today:.2f}")
        
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

    async def check_risk_limits(self) -> bool:
        """
        Check if we're within risk limits.
        Returns True if trading is allowed, False otherwise.
        """
        try:
            # Get current balance
            balance = await self.get_account_balance()
            current_balance = balance['total']
            
            # Set starting balance if not set
            if self.starting_balance_today == 0:
                self.starting_balance_today = current_balance
            
            # Check daily loss limit
            daily_loss_limit = self.starting_balance_today * (self.max_daily_loss_percent / 100)
            current_daily_loss = self.starting_balance_today - current_balance + self.daily_pnl
            
            if current_daily_loss > daily_loss_limit:
                logger.warning(f"⚠️ DAILY LOSS LIMIT REACHED: ${current_daily_loss:.2f} > ${daily_loss_limit:.2f}")
                logger.warning("Trading disabled for the rest of the day")
                self.trading_enabled = False
                return False
            
            # Check max drawdown
            portfolio = self.database.get_portfolio_stats()
            current_drawdown = portfolio.get('max_drawdown', 0)
            
            if current_drawdown > self.max_drawdown_percent:
                logger.warning(f"⚠️ MAX DRAWDOWN EXCEEDED: {current_drawdown:.1f}% > {self.max_drawdown_percent}%")
                logger.warning("Trading disabled until drawdown recovers")
                self.trading_enabled = False
                return False
            
            # Check if we have minimum balance
            if current_balance < 10:
                logger.warning(f"⚠️ INSUFFICIENT BALANCE: ${current_balance:.2f}")
                self.trading_enabled = False
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return True  # Allow trading on error to not lock out

    async def sync_positions_with_exchange(self):
        """Sync database with actual Bybit positions"""
        try:
            positions = self.analyzer.exchange.fetch_positions(params={'category': 'linear'})
            
            open_positions = {}
            for pos in positions:
                if pos['contracts'] > 0:
                    symbol = pos['symbol']
                    symbol = symbol.replace(':USDT', '')
                    if symbol.endswith('USDTUSDT'):
                        symbol = symbol.replace('USDTUSDT', 'USDT')
                    symbol = self.normalize_symbol(symbol)
                    
                    open_positions[symbol] = {
                        'contracts': pos['contracts'],
                        'side': pos['side'],
                        'unrealizedPnl': pos.get('unrealizedPnl', 0),
                        'markPrice': pos.get('markPrice', 0),
                        'entryPrice': pos.get('avgPrice', pos.get('markPrice', 0)),
                        'leverage': pos.get('leverage', self.leverage)
                    }
                    logger.debug(f"Bybit position: {symbol} - {pos['contracts']} contracts")
            
            self.active_positions = open_positions
            
            # Get active signals from database
            db_signals = self.database.get_active_signals()
            
            # Close orphaned signals (in DB but not on exchange)
            for signal in db_signals:
                coin_symbol = self.normalize_symbol(signal['coin'])
                
                if coin_symbol not in open_positions:
                    logger.info(f"Closing orphaned signal: {coin_symbol}")
                    
                    try:
                        ticker = self.analyzer.exchange.fetch_ticker(coin_symbol)
                        exit_price = ticker['last']
                    except:
                        exit_price = signal.get('current_price', signal['entry_price'])
                    
                    pnl = self.database.close_signal(
                        signal['signal_id'],
                        exit_price,
                        "Position closed on exchange"
                    )
                    
                    # Update daily tracking
                    if pnl:
                        self._record_trade_result(pnl)
            
            # Track untracked positions (on exchange but not in DB)
            for symbol, pos_data in open_positions.items():
                found_in_db = any(
                    self.normalize_symbol(sig['coin']) == symbol 
                    for sig in db_signals
                )
                
                if not found_in_db:
                    logger.warning(f"Found untracked position: {symbol}")
                    signal_data = {
                        'signal_id': f"SYNC_{symbol}_{int(time.time())}",
                        'timestamp': datetime.now().isoformat(),
                        'coin': symbol,
                        'direction': 'LONG' if pos_data['side'] == 'long' else 'SHORT',
                        'entry_price': pos_data['entryPrice'],
                        'take_profit': pos_data['entryPrice'] * (1.03 if pos_data['side'] == 'long' else 0.97),
                        'stop_loss': pos_data['entryPrice'] * (0.97 if pos_data['side'] == 'long' else 1.03),
                        'confidence': 50,
                        'position_value': pos_data['contracts'] * pos_data['markPrice'],
                        'strategy_type': 'synced',
                        'analysis_data': {'synced_from_exchange': True}
                    }
                    self.database.save_signal(signal_data)
            
            return open_positions
            
        except Exception as e:
            logger.error(f"Error syncing positions: {e}")
            return {}

    def _record_trade_result(self, pnl: float):
        """Record trade result for tracking"""
        self.daily_pnl += pnl
        self.daily_trades += 1
        
        if pnl > 0:
            self.daily_wins += 1
        else:
            self.daily_losses += 1
        
        # Add to recent trades for adaptive sizing
        self.recent_trade_results.insert(0, pnl)
        if len(self.recent_trade_results) > self.recent_trades_window:
            self.recent_trade_results.pop()
        
        # Update performance metrics
        if self.daily_trades > 0:
            self.performance_metrics['win_rate'] = (self.daily_wins / self.daily_trades) * 100
        self.performance_metrics['total_pnl'] += pnl

    async def execute_trade(self, signal_data: Dict) -> bool:
        """Execute real trade on Bybit with proper risk management"""
        try:
            symbol = self.normalize_symbol(signal_data['coin'])
            signal_data['coin'] = symbol
            
            logger.info(f"Attempting to execute trade for {symbol}")
            
            # Risk checks
            if not self.trading_enabled:
                logger.warning("Trading disabled due to risk limits")
                return False
            
            if not await self.check_risk_limits():
                return False
            
            # Get balance
            balance_info = await self.get_account_balance()
            available_balance = balance_info['free']
            
            if available_balance < 10:
                logger.warning(f"Insufficient balance: ${available_balance:.2f}")
                return False
            
            # Get confidence for adaptive sizing
            confidence = signal_data.get('model_confidence', signal_data.get('confidence', 50) / 100)
            
            # Calculate adaptive position size
            adaptive_size = self._get_adaptive_position_size(confidence)
            
            # Calculate position value
            account_equity = available_balance * adaptive_size
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
            
            if qty_precision == 0:
                quantity = max(int(base_quantity), 1)
            else:
                quantity = round(base_quantity, qty_precision)
            
            if quantity < min_qty:
                quantity = min_qty
            
            final_position_value = quantity * current_price
            margin_used = final_position_value / self.leverage
            
            # Check if position value is reasonable
            if final_position_value < 5:
                logger.warning(f"Position value too small: ${final_position_value:.2f}")
                return False
            
            logger.info(f"Position: {quantity} contracts @ ${current_price:.4f}")
            logger.info(f"Value: ${final_position_value:.2f}, Margin: ${margin_used:.2f}")
            
            # Set leverage
            try:
                self.analyzer.exchange.set_leverage(self.leverage, symbol)
            except:
                pass
            
            # Place order
            side = 'buy' if signal_data['direction'] == 'LONG' else 'sell'
            
            if side == 'buy':
                order = self.analyzer.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=side,
                    amount=quantity,
                    price=current_price,
                    params={
                        'positionIdx': 0,
                        'timeInForce': 'IOC',
                        'orderType': 'Market',
                        'category': 'linear',
                        'createMarketBuyOrderRequiresPrice': False
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
            
            await asyncio.sleep(2)
            
            logger.info(f"Order response: {order}")
            
            # Verify order
            if order and order.get('id'):
                logger.info(f"Order status: {order.get('status')}, filled: {order.get('filled')}")
                
                if order.get('status') is None:
                    await asyncio.sleep(1)
                    try:
                        actual_order = self.analyzer.exchange.fetch_order(order['id'], symbol)
                        logger.info(f"Fetched order status: {actual_order.get('status')}, filled: {actual_order.get('filled')}")
                        if actual_order.get('status') in ['closed', 'filled', 'Filled'] or actual_order.get('filled', 0) > 0:
                            order = actual_order
                        else:
                            logger.error(f"Order not filled: {actual_order}")
                            return False
                    except Exception as e:
                        logger.error(f"Error fetching order status: {e}")
            
            if order.get('status') in ['closed', 'filled', 'Filled'] or order.get('filled', 0) > 0:
                # Set SL/TP
                await self.set_sl_tp(
                    symbol,
                    signal_data['direction'],
                    quantity,
                    signal_data['stop_loss'],
                    signal_data['take_profit']
                )
                
                logger.info(f"✓ Trade executed: {symbol} {side.upper()}")
                logger.info(f"  Entry: ${current_price:.4f}, SL: ${signal_data['stop_loss']:.4f}, TP: ${signal_data['take_profit']:.4f}")
                
                # Update signal data
                signal_data['order_id'] = order.get('id')
                signal_data['executed_price'] = order.get('average', current_price)
                signal_data['executed_quantity'] = quantity
                signal_data['position_value'] = final_position_value
                signal_data['leverage'] = self.leverage
                signal_data['margin_used'] = margin_used
                
                # Update tracking
                self.active_positions[symbol] = {
                    'signal_id': signal_data['signal_id'],
                    'direction': signal_data['direction'],
                    'entry_price': signal_data['executed_price'],
                    'quantity': quantity,
                    'position_value': final_position_value
                }
                
                self.performance_metrics['trades_executed'] += 1
                
                return True
            else:
                logger.error(f"Order failed: {order}")
                return False
                
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def set_sl_tp(self, symbol: str, direction: str, quantity: float,
                        stop_loss: float, take_profit: float) -> bool:
        """Set stop loss and take profit orders"""
        try:
            await asyncio.sleep(3)
            
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
            
            response = self.analyzer.exchange.privatePostV5PositionTradingStop(params)
            
            if response and (response.get('retCode') in [0, '0'] or response.get('ret_code') in [0, '0']):
                logger.info(f"SL/TP set for {symbol} - TP: {take_profit:.6f}, SL: {stop_loss:.6f}")
                return True
            else:
                logger.error(f"Failed to set SL/TP for {symbol}: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting SL/TP for {symbol}: {e}")
            return False

    async def update_sl_tp_on_exchange(self, symbol: str, direction: str,
                                        new_sl: float, new_tp: float) -> bool:
        """Update stop loss and take profit on exchange"""
        try:
            params = {
                'category': 'linear',
                'symbol': symbol,
                'stopLoss': str(new_sl),
                'takeProfit': str(new_tp),
                'tpTriggerBy': 'LastPrice',
                'slTriggerBy': 'LastPrice',
                'tpslMode': 'Full',
                'positionIdx': 0
            }
            
            response = self.analyzer.exchange.privatePostV5PositionTradingStop(params)
            
            if response and (response.get('retCode') in [0, '0'] or response.get('ret_code') in [0, '0']):
                return True
            else:
                logger.warning(f"Failed to update SL/TP for {symbol}: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating SL/TP for {symbol}: {e}")
            return False

    async def close_position(self, signal_data: Dict, reason: str) -> bool:
        """Close position on Bybit"""
        try:
            symbol = self.normalize_symbol(signal_data['coin'])
            
            positions = self.analyzer.exchange.fetch_positions([symbol], params={'category': 'linear'})
            position = next((p for p in positions if p['symbol'] == symbol), None)
            
            if not position or position['contracts'] == 0:
                logger.warning(f"No open position for {symbol}")
                return False
            
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
        """Validate signal before execution with enhanced checks"""
        try:
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
                    logger.error(f"Invalid LONG signal prices: SL={sl}, Entry={entry}, TP={tp}")
                    return False
            else:
                if not (tp < entry < sl):
                    logger.error(f"Invalid SHORT signal prices: TP={tp}, Entry={entry}, SL={sl}")
                    return False
            
            # Check existing position
            symbol = signal_data['coin']
            if symbol in self.active_positions:
                logger.warning(f"Already have position for {symbol}")
                return False
            
            # Check confidence threshold
            model_confidence = signal_data.get('model_confidence')
            if model_confidence is None:
                model_confidence = signal_data.get('confidence', 0) / 100.0
            
            if model_confidence < self.min_confidence_threshold:
                logger.info(f"Signal filtered - confidence too low: {model_confidence:.2f} < {self.min_confidence_threshold}")
                self.performance_metrics['signals_filtered'] += 1
                return False
            
            # Validate R:R ratio
            if direction == 'LONG':
                risk = entry - sl
                reward = tp - entry
            else:
                risk = sl - entry
                reward = entry - tp
            
            rr_ratio = reward / risk if risk > 0 else 0
            
            if rr_ratio < self.min_rr_ratio:
                logger.info(f"Signal filtered - R:R too low: {rr_ratio:.2f} < {self.min_rr_ratio}")
                self.performance_metrics['signals_filtered'] += 1
                return False
            
            # Store R:R in signal data
            signal_data['risk_reward_ratio'] = rr_ratio
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False

    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def dashboard():
            return HTMLResponse(content=self.get_dashboard_html())
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
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
            try:
                signals = self.database.get_active_signals()
                
                for signal in signals:
                    try:
                        symbol = self.normalize_symbol(signal['coin'])
                        current_price = await self.analyzer.get_current_price(symbol)
                        
                        if current_price > 0:
                            signal['current_price'] = current_price
                            
                            position_value = signal.get('position_value', 0)
                            leverage = signal.get('leverage', self.leverage)
                            
                            if position_value > 0:
                                if signal['direction'] == 'LONG':
                                    pnl_pct = ((current_price - signal['entry_price']) / signal['entry_price']) * 100 * leverage
                                else:
                                    pnl_pct = ((signal['entry_price'] - current_price) / signal['entry_price']) * 100 * leverage
                                
                                margin_used = position_value / leverage
                                pnl_usd = (pnl_pct / 100) * margin_used
                                
                                signal['live_pnl_usd'] = round(pnl_usd, 2)
                                signal['live_pnl_percentage'] = round(pnl_pct, 2)
                    except Exception as e:
                        logger.error(f"Error enhancing signal: {e}")
                
                return JSONResponse(content={"signals": signals, "count": len(signals)})
                
            except Exception as e:
                logger.error(f"Error getting signals: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.get("/api/portfolio")
        async def get_portfolio():
            try:
                portfolio = self.database.get_portfolio_stats()
                
                real_balance = await self.get_account_balance()
                portfolio['real_balance'] = real_balance['total']
                portfolio['available_balance'] = real_balance['free']
                
                total_open_pnl = 0
                positions = self.analyzer.exchange.fetch_positions(params={'category': 'linear'})
                for pos in positions:
                    if pos['contracts'] > 0:
                        unrealized_pnl = pos.get('unrealizedPnl', 0) or 0
                        total_open_pnl += unrealized_pnl
                
                portfolio['open_pnl'] = round(total_open_pnl, 2)
                portfolio['total_balance'] = real_balance['total'] + total_open_pnl
                portfolio['daily_pnl'] = round(self.daily_pnl, 2)
                portfolio['daily_trades'] = self.daily_trades
                portfolio['daily_wins'] = self.daily_wins
                portfolio['daily_losses'] = self.daily_losses
                portfolio['trading_enabled'] = self.trading_enabled
                portfolio['leverage'] = self.leverage
                portfolio['position_size_pct'] = self.position_size_percent * 100
                
                return JSONResponse(content={"portfolio": portfolio})
                
            except Exception as e:
                logger.error(f"Error getting portfolio: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.get("/api/system")
        async def get_system_stats():
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
                    "max_positions": self.max_concurrent_positions,
                    "is_running": self.running,
                    "is_scanning": self.is_scanning,
                    "trading_enabled": self.trading_enabled,
                    "scan_interval": self.scan_interval,
                    "leverage": self.leverage,
                    "position_size_percent": self.position_size_percent * 100,
                    "min_confidence": self.min_confidence_threshold,
                    "min_rr_ratio": self.min_rr_ratio,
                    "performance_metrics": self.performance_metrics,
                    "daily_pnl": self.daily_pnl,
                    "daily_trades": self.daily_trades,
                    "recent_win_rate": self._calculate_recent_win_rate()
                })
                
            except Exception as e:
                logger.error(f"Error getting system stats: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.get("/api/trades/history")
        async def get_trade_history():
            try:
                with self.database.get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT * FROM signals 
                        WHERE status = 'closed' 
                        ORDER BY closed_at DESC 
                        LIMIT 100
                    ''')
                    
                    trades = []
                    for row in cursor.fetchall():
                        trade = dict(row)
                        
                        position_value = trade.get('position_value', 0)
                        leverage = trade.get('leverage', self.leverage)
                        
                        if position_value > 0 and trade.get('exit_price'):
                            if trade['direction'] == 'LONG':
                                pnl_pct = ((trade['exit_price'] - trade['entry_price']) / trade['entry_price']) * 100 * leverage
                            else:
                                pnl_pct = ((trade['entry_price'] - trade['exit_price']) / trade['entry_price']) * 100 * leverage
                            
                            margin_used = position_value / leverage
                            trade['pnl_usd'] = round((pnl_pct / 100) * margin_used, 2)
                            trade['pnl_percentage'] = round(pnl_pct, 2)
                        else:
                            trade['pnl_usd'] = 0
                            trade['pnl_percentage'] = 0
                        
                        trades.append(trade)
                    
                    return JSONResponse(content={"trades": trades})
                    
            except Exception as e:
                logger.error(f"Error getting trade history: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.get("/api/risk")
        async def get_risk_status():
            """Get current risk management status"""
            try:
                balance = await self.get_account_balance()
                portfolio = self.database.get_portfolio_stats()
                
                daily_loss_limit = self.starting_balance_today * (self.max_daily_loss_percent / 100) if self.starting_balance_today > 0 else 0
                current_daily_loss = max(0, self.starting_balance_today - balance['total'] + self.daily_pnl) if self.starting_balance_today > 0 else 0
                
                return JSONResponse(content={
                    "trading_enabled": self.trading_enabled,
                    "daily_loss_limit": round(daily_loss_limit, 2),
                    "current_daily_loss": round(current_daily_loss, 2),
                    "daily_loss_remaining": round(max(0, daily_loss_limit - current_daily_loss), 2),
                    "max_drawdown_limit": self.max_drawdown_percent,
                    "current_drawdown": round(portfolio.get('max_drawdown', 0), 2),
                    "positions_used": len(self.active_positions),
                    "max_positions": self.max_concurrent_positions,
                    "recent_win_rate": round(self._calculate_recent_win_rate() * 100, 1),
                    "leverage": self.leverage,
                    "position_size_pct": self.position_size_percent * 100
                })
                
            except Exception as e:
                logger.error(f"Error getting risk status: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.post("/api/control/start")
        async def start_bot():
            if not self.running:
                self.running = True
                self.trading_enabled = True
                asyncio.create_task(self.main_trading_loop())
                asyncio.create_task(self.position_monitor_loop())
                return JSONResponse(content={"status": "started", "message": "Trading bot started"})
            else:
                return JSONResponse(content={"status": "already_running"})

        @self.app.post("/api/control/stop")
        async def stop_bot():
            self.running = False
            self.analyzer.save_models()
            return JSONResponse(content={"status": "stopped", "message": "Trading bot stopped"})

        @self.app.post("/api/control/enable_trading")
        async def enable_trading():
            self.trading_enabled = True
            logger.info("Trading manually enabled")
            return JSONResponse(content={"status": "enabled", "message": "Trading enabled"})

        @self.app.post("/api/control/disable_trading")
        async def disable_trading():
            self.trading_enabled = False
            logger.info("Trading manually disabled")
            return JSONResponse(content={"status": "disabled", "message": "Trading disabled"})

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
        """Main trading loop with improved logging"""
        while self.running:
            try:
                # Check for daily reset
                self._check_daily_reset()
                
                self.is_scanning = True
                scan_start = datetime.now()
                
                logger.info("=" * 50)
                logger.info(f"SCAN #{self.scan_count + 1}")
                logger.info("=" * 50)
                
                # Sync with exchange
                await self.sync_positions_with_exchange()
                
                # Check for exits
                await self.check_signal_exits()
                
                # Check risk limits
                if not await self.check_risk_limits():
                    logger.warning("Risk limits exceeded - skipping signal generation")
                    self.is_scanning = False
                    await asyncio.sleep(self.scan_interval)
                    continue
                
                # Generate new signals if we have capacity
                current_positions = len(self.active_positions)
                
                if current_positions < self.max_concurrent_positions and self.trading_enabled:
                    logger.info(f"Scanning for signals (positions: {current_positions}/{self.max_concurrent_positions})")
                    
                    # Get new signals from analyzer
                    new_signals = await self.analyzer.scan_all_coins()
                    
                    self.performance_metrics['signals_generated'] += len(new_signals)
                    logger.info(f"Generated {len(new_signals)} raw signals")
                    
                    for signal_data in new_signals:
                        if len(self.active_positions) >= self.max_concurrent_positions:
                            logger.info("Max positions reached, stopping signal processing")
                            break
                        
                        # Validate signal
                        if not self.validate_signal(signal_data):
                            continue
                        
                        # Execute trade
                        if await self.execute_trade(signal_data):
                            self.database.save_signal(signal_data)
                            
                            await self.broadcast_to_clients({
                                "type": "new_signal",
                                "signal": signal_data
                            })
                            
                            logger.info(f"✓ New position opened: {signal_data['coin']} {signal_data['direction']}")
                else:
                    if not self.trading_enabled:
                        logger.info("Trading disabled - skipping signal generation")
                    else:
                        logger.info(f"At max positions ({current_positions}/{self.max_concurrent_positions})")
                
                # Update metrics
                self.scan_count += 1
                self.last_scan_time = datetime.now()
                self.is_scanning = False
                self.performance_metrics['total_scans'] = self.scan_count
                
                scan_duration = (datetime.now() - scan_start).total_seconds()
                
                logger.info("-" * 50)
                logger.info(f"Scan completed in {scan_duration:.1f}s")
                logger.info(f"Positions: {len(self.active_positions)}/{self.max_concurrent_positions}")
                logger.info(f"Daily P&L: ${self.daily_pnl:.2f} | Trades: {self.daily_trades}")
                logger.info("-" * 50)
                
                await self.broadcast_to_clients({
                    "type": "scan_completed",
                    "scan_count": self.scan_count,
                    "duration": scan_duration,
                    "active_positions": len(self.active_positions)
                })
                
                # Model updates
                current_time = time.time()
                if current_time - self.last_model_update > self.model_update_interval:
                    logger.info("Triggering periodic model update...")
                    asyncio.create_task(self.analyzer.periodic_model_update(self.database))
                    self.last_model_update = current_time
                
            except Exception as e:
                self.is_scanning = False
                logger.error(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
            
            await asyncio.sleep(self.scan_interval)

    async def position_monitor_loop(self):
        """Monitor positions frequently"""
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
                    
                    self.database.update_signal_price(signal['signal_id'], current_price)
                    
                    # Get market data for analysis
                    df = await self.analyzer.get_market_data(symbol, '1h', 100)
                    
                    exit_reason = None
                    exit_price = None
                    
                    # Dynamic exit analysis
                    if not df.empty and len(df) >= 50:
                        evaluation = self.analyzer.evaluate_position_health(signal, df)
                        dynamic_levels = self.analyzer.calculate_dynamic_levels(signal, df, evaluation)
                        
                        if dynamic_levels.get('action') == 'exit':
                            exit_reason = f"Dynamic Exit: {dynamic_levels.get('reason', 'Strategy recommendation')}"
                            exit_price = current_price
                        elif dynamic_levels.get('action') == 'update':
                            new_sl = dynamic_levels.get('new_sl')
                            new_tp = dynamic_levels.get('new_tp')
                            if new_sl and new_tp:
                                if await self.update_sl_tp_on_exchange(symbol, signal['direction'], new_sl, new_tp):
                                    with self.database.get_db_connection() as conn:
                                        cursor = conn.cursor()
                                        cursor.execute('''
                                            UPDATE signals 
                                            SET stop_loss = ?, take_profit = ?, updated_at = CURRENT_TIMESTAMP
                                            WHERE signal_id = ?
                                        ''', (new_sl, new_tp, signal['signal_id']))
                                    logger.info(f"Updated SL/TP for {symbol}: SL={new_sl:.6f}, TP={new_tp:.6f}")
                                    continue
                    
                    # Standard exit conditions
                    if not exit_reason:
                        if signal['direction'] == 'LONG':
                            if current_price >= signal['take_profit']:
                                exit_reason = "Take Profit Hit"
                                exit_price = signal['take_profit']
                            elif current_price <= signal['stop_loss']:
                                exit_reason = "Stop Loss Hit"
                                exit_price = signal['stop_loss']
                        else:
                            if current_price <= signal['take_profit']:
                                exit_reason = "Take Profit Hit"
                                exit_price = signal['take_profit']
                            elif current_price >= signal['stop_loss']:
                                exit_reason = "Stop Loss Hit"
                                exit_price = signal['stop_loss']
                    
                    if exit_reason and exit_price:
                        await self.close_position(signal, exit_reason)
                        
                        pnl = self.database.close_signal(signal['signal_id'], exit_price, exit_reason)
                        
                        # Record trade result
                        if pnl is not None:
                            self._record_trade_result(pnl)
                        
                        if symbol in self.active_positions:
                            del self.active_positions[symbol]
                        
                        await self.broadcast_to_clients({
                            "type": "signal_closed",
                            "signal_id": signal['signal_id'],
                            "coin": symbol,
                            "exit_reason": exit_reason,
                            "pnl": pnl
                        })
                        
                        pnl_str = f"${pnl:.2f}" if pnl else "N/A"
                        logger.info(f"Signal closed: {symbol} - {exit_reason} | P&L: {pnl_str}")
                        
                except Exception as e:
                    logger.error(f"Error checking exit for {signal['signal_id']}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in signal exit check: {e}")

    def get_dashboard_html(self) -> str:
        """Return dashboard HTML"""
        try:
            with open('templates/dashboard.html', 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return """
            <!DOCTYPE html>
            <html>
            <head><title>Trading Bot Dashboard</title></head>
            <body>
                <h1>Trading Bot Dashboard</h1>
                <p>Dashboard template not found. API endpoints available at /api/*</p>
                <ul>
                    <li><a href="/api/signals">/api/signals</a> - Active signals</li>
                    <li><a href="/api/portfolio">/api/portfolio</a> - Portfolio stats</li>
                    <li><a href="/api/system">/api/system</a> - System status</li>
                    <li><a href="/api/risk">/api/risk</a> - Risk status</li>
                    <li><a href="/api/trades/history">/api/trades/history</a> - Trade history</li>
                </ul>
            </body>
            </html>
            """


# Global bot instance
bot_instance = None

def get_bot():
    global bot_instance
    if bot_instance is None:
        bot_instance = TradingBot()
    return bot_instance

app = get_bot().app

@app.on_event("startup")
async def startup_event():
    bot = get_bot()
    if not bot.running:
        bot.running = True
        asyncio.create_task(bot.main_trading_loop())
        asyncio.create_task(bot.position_monitor_loop())
        logger.info("Trading bot auto-started")

@app.on_event("shutdown")
async def shutdown_event():
    bot = get_bot()
    bot.running = False
    bot.analyzer.save_models()
    logger.info("Trading bot shutdown complete")


if __name__ == "__main__":
    try:
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("templates", exist_ok=True)
        
        logger.info("Starting Production Trading Bot Server...")
        
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