#!/usr/bin/env python3
"""
Production-Ready Enhanced Crypto Trading Bot
Institutional-Grade Analysis with Smart Money Concepts
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

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Import our enhanced modules
from database import Database
from enhanced_trading_analyzer_v2 import ProductionTradingAnalyzer

# Configure logging with proper formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ProductionTradingBot:
    def __init__(self):
        # Core components
        self.database = Database()
        self.analyzer = ProductionTradingAnalyzer(database=self.database)
        
        # FastAPI setup
        self.app = FastAPI(
            title="Production Crypto Trading Bot",
            version="2.0.0",
            description="Institutional-Grade Trading Bot with Smart Money Analysis"
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
        self.scan_interval = 300  # 5 minutes for production quality
        self.last_scan_time = None
        self.scan_count = 0
        self.startup_time = datetime.now()
        
        # Signal management
        self.active_signals: Set[str] = set()
        self.signal_grace_period = 300  # 5 minutes grace period
        self.scanning_lock = asyncio.Lock()
        
        # Performance metrics
        self.performance_metrics = {
            'total_scans': 0,
            'signals_generated': 0,
            'signals_closed': 0,
            'avg_scan_time': 0,
            'win_rate': 0,
            'avg_hold_time': 0
        }
        
        # Market regime tracking
        self.market_regime = 'normal'
        
        # Setup routes and initialize
        self.setup_routes()
        self.initialize_bot_state()
        
        # Graceful shutdown handling
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle graceful shutdown"""
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self.running = False
        sys.exit(0)
    
    def initialize_bot_state(self):
        """Initialize bot state with cleanup"""
        try:
            # Clean up any orphaned signals
            active_signals = self.database.get_active_signals()
            stale_signals = []
            
            current_time = datetime.now()
            for signal in active_signals:
                # Check if signal is older than 24 hours
                try:
                    signal_time = datetime.fromisoformat(signal['timestamp'].replace('Z', '+00:00'))
                    if (current_time - signal_time).total_seconds() > 86400:  # 24 hours
                        stale_signals.append(signal)
                except:
                    stale_signals.append(signal)  # Invalid timestamp
            
            # Close stale signals
            for signal in stale_signals:
                self.database.close_signal(
                    signal['signal_id'],
                    signal['entry_price'],
                    "System cleanup - stale signal"
                )
            
            if stale_signals:
                logger.info(f"Cleaned up {len(stale_signals)} stale signals")
            
            # Clean old data periodically
            self.database.clean_old_data(days=30)
            
            # Log initialization
            self.database.log_bot_activity(
                'INFO', 'SYSTEM', 'Production bot initialized',
                f'Enhanced algorithms ready for {len(self.analyzer.coins)} coins'
            )
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
    
    def setup_routes(self):
        """Setup all API routes"""
        
        @self.app.get("/")
        async def dashboard():
            """Serve enhanced dashboard"""
            return HTMLResponse(content=self.get_enhanced_dashboard())
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """Enhanced WebSocket endpoint"""
            await websocket.accept()
            self.websocket_connections.add(websocket)
            
            try:
                # Send initial connection message
                await websocket.send_text(json.dumps({
                    "type": "connection_established",
                    "message": "Connected to Production Trading Bot",
                    "version": "2.0.0",
                    "features": [
                        "institutional_analysis",
                        "smart_money_concepts",
                        "production_grade_signals",
                        "real_time_monitoring"
                    ],
                    "timestamp": datetime.now().isoformat()
                }))
                
                # Keep connection alive
                while True:
                    await asyncio.sleep(30)  # Send heartbeat every 30 seconds
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
            """Get active signals with live P&L"""
            try:
                signals = self.database.get_active_signals()
                enhanced_signals = []
                
                for signal in signals:
                    try:
                        current_price = await self.analyzer.get_current_price(signal['coin'])
                        
                        if current_price > 0:
                            # Update database with current price
                            self.database.update_signal_price(signal['signal_id'], current_price)
                            
                            # Calculate live P&L
                            entry_price = signal['entry_price']
                            direction = signal['direction']
                            
                            # P&L calculation with 10x leverage
                            if direction == 'LONG':
                                pnl_pct = ((current_price - entry_price) / entry_price) * 100 * 10
                            else:  # SHORT
                                pnl_pct = ((entry_price - current_price) / entry_price) * 100 * 10
                            
                            pnl_usd = (pnl_pct / 100) * 1000  # $1000 position size
                            
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
                            
                            # Enhanced signal data
                            signal['current_price'] = current_price
                            signal['live_pnl_usd'] = round(pnl_usd, 2)
                            signal['live_pnl_percentage'] = round(pnl_pct, 2)
                            signal['tp_progress'] = max(0, min(100, tp_progress))
                            signal['sl_distance'] = max(0, min(100, sl_distance))
                            signal['signal_grade'] = signal.get('analysis_data', {}).get('signal_grade', 'standard')
                            
                        else:
                            # Fallback if price fetch fails
                            signal['current_price'] = signal['entry_price']
                            signal['live_pnl_usd'] = 0
                            signal['live_pnl_percentage'] = 0
                            signal['tp_progress'] = 0
                            signal['sl_distance'] = 0
                            signal['signal_grade'] = signal.get('analysis_data', {}).get('signal_grade', 'standard')
                        
                        enhanced_signals.append(signal)
                        
                    except Exception as e:
                        logger.error(f"Error enhancing signal {signal['signal_id']}: {e}")
                        # Add signal with default values
                        signal['current_price'] = signal['entry_price']
                        signal['live_pnl_usd'] = 0
                        signal['live_pnl_percentage'] = 0
                        signal['tp_progress'] = 0
                        signal['sl_distance'] = 0
                        enhanced_signals.append(signal)
                
                return JSONResponse(content={
                    "signals": enhanced_signals,
                    "count": len(enhanced_signals),
                    "performance": self.performance_metrics,
                    "market_regime": self.market_regime
                })
                
            except Exception as e:
                logger.error(f"Error getting signals: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)
        
        @self.app.get("/api/portfolio")
        async def get_portfolio():
            """Get enhanced portfolio statistics"""
            try:
                portfolio = self.database.get_portfolio_stats()
                
                # Add enhanced metrics
                portfolio['signal_quality_score'] = self.calculate_signal_quality_score()
                portfolio['risk_metrics'] = self.get_risk_metrics()
                portfolio['market_regime'] = self.market_regime
                
                return JSONResponse(content={"portfolio": portfolio})
            except Exception as e:
                logger.error(f"Error getting portfolio: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)
        
        @self.app.get("/api/analysis/{coin}")
        async def get_coin_analysis(coin: str):
            """Get detailed technical analysis for a coin"""
            try:
                symbol = f"{coin}/USDT"
                
                # Get market data
                df = await self.analyzer.get_market_data(symbol, '1h', 200)
                if df.empty:
                    return JSONResponse(content={"error": "No data available"}, status_code=404)
                
                # Calculate indicators
                df = self.analyzer.calculate_technical_indicators(df)
                
                # Get smart money analysis
                smart_money = self.analyzer.detect_smart_money_patterns(df)
                
                # Get order book
                order_book = await self.analyzer.get_order_book_analysis(symbol)
                
                # Calculate confluence
                confluence = self.analyzer.calculate_signal_confluence(df, smart_money, order_book)
                
                # Prepare analysis response
                analysis = {
                    'coin': coin,
                    'current_price': float(df['close'].iloc[-1]),
                    'confluence': confluence,
                    'smart_money_patterns': smart_money,
                    'order_book_analysis': order_book,
                    'technical_indicators': {
                        'rsi': float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else None,
                        'macd_hist': float(df['macd_hist'].iloc[-1]) if 'macd_hist' in df.columns else None,
                        'volume_ratio': float(df['volume_ratio'].iloc[-1]) if 'volume_ratio' in df.columns else None,
                        'atr_pct': float(df['atr_pct'].iloc[-1]) if 'atr_pct' in df.columns else None,
                        'market_structure': str(df['market_structure'].iloc[-1]) if 'market_structure' in df.columns else None
                    },
                    'support_resistance': {
                        'support': float(df['support'].iloc[-1]) if 'support' in df.columns and df['support'].notna().iloc[-1] else None,
                        'resistance': float(df['resistance'].iloc[-1]) if 'resistance' in df.columns and df['resistance'].notna().iloc[-1] else None
                    },
                    'price_levels': {
                        'ema_9': float(df['ema_9'].iloc[-1]) if 'ema_9' in df.columns else None,
                        'ema_21': float(df['ema_21'].iloc[-1]) if 'ema_21' in df.columns else None,
                        'ema_50': float(df['ema_50'].iloc[-1]) if 'ema_50' in df.columns else None,
                        'vwap': float(df['vwap'].iloc[-1]) if 'vwap' in df.columns else None
                    }
                }
                
                return JSONResponse(content=analysis)
                
            except Exception as e:
                logger.error(f"Error getting analysis for {coin}: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)
        
        @self.app.get("/api/logs")
        async def get_logs():
            """Get recent bot logs"""
            try:
                logs = self.database.get_recent_logs(100)
                return JSONResponse(content={"logs": logs})
            except Exception as e:
                logger.error(f"Error getting logs: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)
        
        @self.app.get("/api/system")
        async def get_system_stats():
            """Get enhanced system statistics"""
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
                    "active_connections": len(self.websocket_connections),
                    "is_running": self.running,
                    "is_scanning": self.is_scanning,
                    "scan_interval": self.scan_interval,
                    "performance_metrics": self.performance_metrics,
                    "analyzer_type": "production_grade",
                    "coins_monitored": len(self.analyzer.coins),
                    "market_regime": self.market_regime,
                    "active_signals_count": len(self.active_signals)
                })
            except Exception as e:
                logger.error(f"Error getting system stats: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)
        
        @self.app.post("/api/control/start")
        async def start_bot():
            """Start the trading bot"""
            if not self.running:
                self.running = True
                asyncio.create_task(self.production_scan_cycle())
                message = "Production trading bot started"
                logger.info(message)
                self.database.log_bot_activity('INFO', 'SYSTEM', message)
                return JSONResponse(content={"status": "started", "message": message})
            else:
                return JSONResponse(content={"status": "already_running", "message": "Bot is already running"})
        
        @self.app.post("/api/control/stop")
        async def stop_bot():
            """Stop the trading bot"""
            self.running = False
            message = "Production trading bot stopped"
            logger.info(message)
            self.database.log_bot_activity('INFO', 'SYSTEM', message)
            return JSONResponse(content={"status": "stopped", "message": message})
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return JSONResponse(content={
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "2.0.0"
            })
    
    def calculate_signal_quality_score(self) -> float:
        """Calculate overall signal quality score"""
        try:
            if self.performance_metrics['total_scans'] == 0:
                return 0.0
            
            # Base score from win rate
            win_rate = self.performance_metrics.get('win_rate', 0)
            base_score = win_rate * 0.4
            
            # Signal generation efficiency
            signal_efficiency = min(1.0, self.performance_metrics.get('signals_generated', 0) / max(1, self.performance_metrics['total_scans']))
            efficiency_score = signal_efficiency * 0.3
            
            # Production grade bonus
            production_bonus = 0.3
            
            return min(1.0, base_score + efficiency_score + production_bonus)
            
        except Exception as e:
            logger.error(f"Error calculating signal quality score: {e}")
            return 0.0
    
    def get_risk_metrics(self) -> Dict:
        """Get comprehensive risk metrics"""
        try:
            active_signals = self.database.get_active_signals()
            
            if not active_signals:
                return {
                    "total_risk": 0,
                    "max_single_risk": 0,
                    "portfolio_heat": 0,
                    "correlation_risk": "low",
                    "active_positions": 0
                }
            
            # Calculate risk metrics
            total_risk = sum(signal.get('analysis_data', {}).get('risk_percentage', 1) for signal in active_signals)
            max_single_risk = max(signal.get('analysis_data', {}).get('risk_percentage', 1) for signal in active_signals)
            
            # Portfolio heat (total exposure)
            portfolio_heat = len(active_signals) * 1.2  # Assuming 1.2% avg risk per trade
            
            # Correlation risk assessment
            if len(active_signals) <= 2:
                correlation_risk = "low"
            elif len(active_signals) <= 4:
                correlation_risk = "medium"
            else:
                correlation_risk = "high"
            
            return {
                "total_risk": round(total_risk, 2),
                "max_single_risk": round(max_single_risk, 2),
                "portfolio_heat": round(portfolio_heat, 2),
                "correlation_risk": correlation_risk,
                "active_positions": len(active_signals)
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {
                "total_risk": 0,
                "max_single_risk": 0,
                "portfolio_heat": 0,
                "correlation_risk": "unknown",
                "active_positions": 0
            }
    
    async def broadcast_to_clients(self, message: Dict):
        """Broadcast message to all connected WebSocket clients"""
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
    
    async def production_scan_cycle(self):
        """Production-grade market scanning cycle"""
        while self.running:
            try:
                # Prevent concurrent scans
                if self.scanning_lock.locked():
                    await asyncio.sleep(60)
                    continue
                
                async with self.scanning_lock:
                    self.is_scanning = True
                    scan_start = datetime.now()
                    
                    logger.info(f"Starting production scan #{self.scan_count + 1}")
                    
                    # Broadcast scan start
                    await self.broadcast_to_clients({
                        "type": "scan_started",
                        "scan_number": self.scan_count + 1,
                        "timestamp": scan_start.isoformat(),
                        "analyzer_type": "production_grade"
                    })
                    
                    # Check signal exits first
                    await self.check_signal_exits()
                    
                    # Scan for new signals
                    active_signals = self.database.get_active_signals()
                    max_concurrent_signals = 3  # Conservative limit for production
                    
                    new_signals_count = 0
                    
                    if len(active_signals) < max_concurrent_signals:
                        try:
                            new_signals = await self.analyzer.scan_all_coins()
                            
                            for signal_data in new_signals:
                                if len(active_signals) + new_signals_count >= max_concurrent_signals:
                                    break
                                
                                # Save signal to database
                                success = self.database.save_signal(signal_data)
                                if success:
                                    new_signals_count += 1
                                    self.active_signals.add(signal_data['signal_id'])
                                    
                                    # Grace period protection
                                    asyncio.create_task(
                                        self.remove_from_grace_period(
                                            signal_data['signal_id'], 
                                            self.signal_grace_period
                                        )
                                    )
                                    
                                    # Broadcast new signal
                                    await self.broadcast_to_clients({
                                        "type": "new_signal",
                                        "signal": signal_data,
                                        "grade": signal_data.get('analysis_data', {}).get('signal_grade', 'standard')
                                    })
                                    
                                    logger.info(f"New {signal_data['direction']} signal: {signal_data['coin']} "
                                              f"(Confidence: {signal_data['confidence']}%)")
                        
                        except Exception as e:
                            logger.error(f"Error during signal generation: {e}")
                    
                    # Update performance metrics
                    self.scan_count += 1
                    self.last_scan_time = datetime.now()
                    scan_duration = (self.last_scan_time - scan_start).total_seconds()
                    self.is_scanning = False
                    
                    # Update metrics
                    self.performance_metrics['total_scans'] = self.scan_count
                    self.performance_metrics['signals_generated'] += new_signals_count
                    
                    # Calculate average scan time
                    prev_avg = self.performance_metrics.get('avg_scan_time', 0)
                    self.performance_metrics['avg_scan_time'] = (
                        (prev_avg * (self.scan_count - 1) + scan_duration) / self.scan_count
                    )
                    
                    # Update win rate from database
                    portfolio = self.database.get_portfolio_stats()
                    self.performance_metrics['win_rate'] = portfolio.get('win_rate', 0) / 100
                    
                    # Log scan completion
                    self.database.log_bot_activity(
                        'INFO', 'SCANNER', 'Production scan completed',
                        f'Duration: {scan_duration:.1f}s, New signals: {new_signals_count}'
                    )
                    
                    # Broadcast scan completion
                    await self.broadcast_to_clients({
                        "type": "scan_completed",
                        "scan_count": self.scan_count,
                        "duration": scan_duration,
                        "new_signals": new_signals_count,
                        "performance": self.performance_metrics
                    })
                    
            except Exception as e:
                self.is_scanning = False
                error_msg = f"Production scan failed: {str(e)}"
                logger.error(error_msg)
                self.database.log_bot_activity('ERROR', 'SCANNER', error_msg)
                
                # Broadcast error
                await self.broadcast_to_clients({
                    "type": "scan_error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
            
            # Wait for next scan cycle
            await asyncio.sleep(self.scan_interval)
    
    async def remove_from_grace_period(self, signal_id: str, delay: int):
        """Remove signal from grace period after delay"""
        await asyncio.sleep(delay)
        self.active_signals.discard(signal_id)
    
    async def check_signal_exits(self):
        """Check for signal exits with enhanced logic"""
        try:
            active_signals = self.database.get_active_signals()
            
            for signal in active_signals:
                # Skip signals in grace period
                if signal['signal_id'] in self.active_signals:
                    continue
                
                try:
                    current_price = await self.analyzer.get_current_price(signal['coin'])
                    if current_price <= 0:
                        continue
                    
                    # Update current price
                    self.database.update_signal_price(signal['signal_id'], current_price)
                    
                    # Check exit conditions with buffer zones
                    exit_reason = None
                    exit_price = None
                    buffer = 0.0015  # 0.15% buffer for slippage
                    
                    if signal['direction'] == 'LONG':
                        # Take profit hit
                        if current_price >= signal['take_profit'] * (1 - buffer):
                            exit_reason = "Take Profit Hit"
                            exit_price = signal['take_profit']
                        # Stop loss hit
                        elif current_price <= signal['stop_loss'] * (1 + buffer):
                            exit_reason = "Stop Loss Hit"
                            exit_price = signal['stop_loss']
                    
                    else:  # SHORT
                        # Take profit hit
                        if current_price <= signal['take_profit'] * (1 + buffer):
                            exit_reason = "Take Profit Hit"
                            exit_price = signal['take_profit']
                        # Stop loss hit
                        elif current_price >= signal['stop_loss'] * (1 - buffer):
                            exit_reason = "Stop Loss Hit"
                            exit_price = signal['stop_loss']
                    
                    # Execute exit if conditions met
                    if exit_reason and exit_price:
                        success = self.database.close_signal(signal['signal_id'], exit_price, exit_reason)
                        
                        if success:
                            # Calculate final P&L
                            if signal['direction'] == 'LONG':
                                pnl_pct = ((exit_price - signal['entry_price']) / signal['entry_price']) * 100 * 10
                            else:
                                pnl_pct = ((signal['entry_price'] - exit_price) / signal['entry_price']) * 100 * 10
                            
                            pnl_usd = (pnl_pct / 100) * 1000
                            
                            # Update metrics
                            self.performance_metrics['signals_closed'] += 1
                            
                            # Broadcast signal closure
                            await self.broadcast_to_clients({
                                "type": "signal_closed",
                                "signal_id": signal['signal_id'],
                                "coin": signal['coin'],
                                "direction": signal['direction'],
                                "exit_reason": exit_reason,
                                "pnl_usd": round(pnl_usd, 2),
                                "pnl_percentage": round(pnl_pct, 2),
                                "exit_price": exit_price
                            })
                            
                            logger.info(f"Signal closed: {signal['coin']} {exit_reason} - P&L: ${pnl_usd:.2f}")
                
                except Exception as e:
                    logger.error(f"Error checking exit for signal {signal['signal_id']}: {e}")
        
        except Exception as e:
            logger.error(f"Error in signal exit check: {e}")
    
    def get_enhanced_dashboard(self) -> str:
        """Return enhanced dashboard HTML"""
        return '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Production Crypto Trading Bot</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { 
                    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
                    color: #ffffff;
                    min-height: 100vh;
                    overflow-x: hidden;
                }
                .container { max-width: 1600px; margin: 0 auto; padding: 20px; }
                
                /* Header */
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
                    width: 300px;
                    height: 300px;
                    background: radial-gradient(circle, rgba(34, 197, 94, 0.1) 0%, transparent 70%);
                    border-radius: 50%;
                    z-index: -1;
                }
                .header h1 {
                    font-size: 3.5rem;
                    font-weight: 800;
                    background: linear-gradient(135deg, #22c55e 0%, #16a34a 50%, #15803d 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    text-shadow: 0 4px 20px rgba(34, 197, 94, 0.3);
                    margin-bottom: 10px;
                }
                .header .subtitle {
                    font-size: 1.2rem;
                    color: #94a3b8;
                    font-weight: 500;
                    opacity: 0.9;
                }
                
                /* Status Grid */
                .status-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                    gap: 25px;
                    margin-bottom: 40px;
                }
                .status-card {
                    background: rgba(30, 41, 59, 0.4);
                    backdrop-filter: blur(20px);
                    border-radius: 20px;
                    padding: 30px;
                    border: 1px solid rgba(34, 197, 94, 0.2);
                    position: relative;
                    overflow: hidden;
                    transition: all 0.3s ease;
                }
                .status-card:hover {
                    transform: translateY(-5px);
                    border-color: rgba(34, 197, 94, 0.4);
                    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
                }
                .status-card::before {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 3px;
                    background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%);
                }
                .status-card h3 {
                    color: #22c55e;
                    font-size: 1.4rem;
                    font-weight: 700;
                    margin-bottom: 20px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
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
                
                /* Signals Section */
                .signals-section {
                    background: rgba(30, 41, 59, 0.4);
                    backdrop-filter: blur(20px);
                    border-radius: 20px;
                    padding: 30px;
                    border: 1px solid rgba(34, 197, 94, 0.2);
                    position: relative;
                }
                .signals-section::before {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 3px;
                    background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
                }
                .signals-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 25px;
                }
                .signals-header h3 {
                    color: #3b82f6;
                    font-size: 1.6rem;
                    font-weight: 700;
                }
                .signals-count {
                    background: rgba(59, 130, 246, 0.2);
                    color: #3b82f6;
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-weight: 600;
                    border: 1px solid rgba(59, 130, 246, 0.3);
                }
                
                /* Signal Cards */
                .signal-card {
                    background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.4) 100%);
                    border: 1px solid rgba(59, 130, 246, 0.3);
                    border-radius: 16px;
                    padding: 25px;
                    margin: 15px 0;
                    position: relative;
                    overflow: hidden;
                    transition: all 0.3s ease;
                }
                .signal-card:hover {
                    transform: scale(1.02);
                    border-color: rgba(59, 130, 246, 0.6);
                    box-shadow: 0 15px 30px rgba(59, 130, 246, 0.1);
                }
                .signal-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }
                .signal-coin {
                    font-size: 1.5rem;
                    font-weight: 800;
                    color: #ffffff;
                }
                .signal-direction {
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-size: 0.9rem;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                .signal-direction.LONG {
                    background: linear-gradient(135deg, #22c55e, #16a34a);
                    color: white;
                    box-shadow: 0 4px 15px rgba(34, 197, 94, 0.3);
                }
                .signal-direction.SHORT {
                    background: linear-gradient(135deg, #ef4444, #dc2626);
                    color: white;
                    box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
                }
                .institutional-badge {
                    position: absolute;
                    top: 15px;
                    right: 15px;
                    background: linear-gradient(135deg, #fbbf24, #f59e0b);
                    color: #000;
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-size: 0.75rem;
                    font-weight: 800;
                    text-transform: uppercase;
                    box-shadow: 0 2px 10px rgba(251, 191, 36, 0.3);
                }
                
                /* Signal Metrics */
                .signal-metrics {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                    gap: 15px;
                    margin-top: 20px;
                }
                .signal-metric {
                    text-align: center;
                    padding: 15px;
                    background: rgba(0, 0, 0, 0.3);
                    border-radius: 12px;
                    border: 1px solid rgba(148, 163, 184, 0.1);
                    transition: all 0.3s ease;
                }
                .signal-metric:hover {
                    background: rgba(59, 130, 246, 0.1);
                    border-color: rgba(59, 130, 246, 0.3);
                }
                .signal-metric-label {
                    font-size: 0.85rem;
                    color: #94a3b8;
                    margin-bottom: 8px;
                    font-weight: 500;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                .signal-metric-value {
                    font-weight: 700;
                    color: #ffffff;
                    font-size: 1.1rem;
                }
                .pnl-positive { color: #22c55e !important; }
                .pnl-negative { color: #ef4444 !important; }
                
                /* Progress Bars */
                .progress-container {
                    width: 100%;
                    height: 6px;
                    background: rgba(0, 0, 0, 0.3);
                    border-radius: 3px;
                    overflow: hidden;
                    margin-top: 5px;
                }
                .progress-bar {
                    height: 100%;
                    border-radius: 3px;
                    transition: width 0.3s ease;
                }
                .progress-bar.tp { background: linear-gradient(90deg, #22c55e, #16a34a); }
                .progress-bar.sl { background: linear-gradient(90deg, #ef4444, #dc2626); }
                
                /* Empty State */
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
                
                /* Status Indicators */
                .status-indicator {
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 8px;
                }
                .status-indicator.running { background: #22c55e; box-shadow: 0 0 10px rgba(34, 197, 94, 0.5); }
                .status-indicator.stopped { background: #ef4444; }
                .status-indicator.scanning { background: #f59e0b; animation: pulse 1.5s infinite; }
                
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }
                
                /* Responsive Design */
                @media (max-width: 768px) {
                    .container { padding: 15px; }
                    .header h1 { font-size: 2.5rem; }
                    .status-grid { grid-template-columns: 1fr; gap: 20px; }
                    .signal-metrics { grid-template-columns: repeat(2, 1fr); }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Production Trading Bot</h1>
                    <div class="subtitle">Institutional-Grade Analysis • Smart Money Concepts • Real-Time Execution</div>
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
                            <span class="metric-label">Analysis Grade:</span>
                            <span class="metric-value">Production</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Active Signals:</span>
                            <span class="metric-value" id="signal-count">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Scan Interval:</span>
                            <span class="metric-value">5 minutes</span>
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
                            <span class="metric-label">Signals Generated:</span>
                            <span class="metric-value" id="signals-generated">0</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Avg Scan Time:</span>
                            <span class="metric-value" id="avg-scan-time">0s</span>
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
                            <span class="metric-value" id="quality-score">0%</span>
                        </div>
                    </div>
                    
                    <div class="status-card">
                        <h3>⚡ System Resources</h3>
                        <div class="metric">
                            <span class="metric-label">CPU Usage:</span>
                            <span class="metric-value" id="cpu-usage">0%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Memory Usage:</span>
                            <span class="metric-value" id="memory-usage">0%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Uptime:</span>
                            <span class="metric-value" id="uptime">0m</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Market Regime:</span>
                            <span class="metric-value" id="market-regime">Normal</span>
                        </div>
                    </div>
                </div>
                
                <div class="signals-section">
                    <div class="signals-header">
                        <h3>🎯 Active Positions</h3>
                        <div class="signals-count" id="active-count">0 Active</div>
                    </div>
                    <div id="signals-list">
                        <div class="empty-state">
                            <h4>No Active Signals</h4>
                            <p>Waiting for high-probability setups...</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                class TradingBotDashboard {
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
                                console.log('WebSocket connected');
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
                                console.log('WebSocket disconnected');
                                this.reconnectWebSocket();
                            };
                            
                            this.ws.onerror = (error) => {
                                console.error('WebSocket error:', error);
                            };
                            
                        } catch (error) {
                            console.error('Error connecting WebSocket:', error);
                        }
                    }
                    
                    reconnectWebSocket() {
                        if (this.wsReconnectAttempts < this.maxReconnectAttempts) {
                            this.wsReconnectAttempts++;
                            setTimeout(() => {
                                console.log(`Reconnecting WebSocket... Attempt ${this.wsReconnectAttempts}`);
                                this.connectWebSocket();
                            }, this.reconnectDelay * this.wsReconnectAttempts);
                        }
                    }
                    
                    handleWebSocketMessage(data) {
                        switch (data.type) {
                            case 'new_signal':
                                this.showNotification('New Signal!', `${data.signal.coin} ${data.signal.direction}`, 'success');
                                this.updateDashboard();
                                break;
                            case 'signal_closed':
                                const pnlClass = data.pnl_usd >= 0 ? 'success' : 'error';
                                this.showNotification('Signal Closed!', 
                                    `${data.coin} - ${data.exit_reason}: ${data.pnl_usd}`, pnlClass);
                                this.updateDashboard();
                                break;
                            case 'scan_started':
                                this.updateScanStatus(true);
                                break;
                            case 'scan_completed':
                                this.updateScanStatus(false);
                                this.updateDashboard();
                                break;
                        }
                    }
                    
                    showNotification(title, message, type = 'info') {
                        // Simple notification system
                        if ('Notification' in window && Notification.permission === 'granted') {
                            new Notification(title, {
                                body: message,
                                icon: '/favicon.ico'
                            });
                        }
                    }
                    
                    updateScanStatus(isScanning) {
                        const statusDot = document.getElementById('status-dot');
                        if (isScanning) {
                            statusDot.className = 'status-indicator scanning';
                        } else {
                            // Will be updated by regular dashboard update
                        }
                    }
                    
                    async updateDashboard() {
                        try {
                            const [signalsResponse, systemResponse, portfolioResponse] = await Promise.all([
                                fetch('/api/signals'),
                                fetch('/api/system'),
                                fetch('/api/portfolio')
                            ]);
                            
                            const signalsData = await signalsResponse.json();
                            const systemData = await systemResponse.json();
                            const portfolioData = await portfolioResponse.json();
                            
                            this.updateSystemStatus(systemData);
                            this.updatePerformanceMetrics(systemData.performance_metrics || {});
                            this.updatePortfolioData(portfolioData.portfolio || {});
                            this.updateSignalsList(signalsData.signals || []);
                            
                        } catch (error) {
                            console.error('Dashboard update error:', error);
                        }
                    }
                    
                    updateSystemStatus(data) {
                        const statusText = document.getElementById('status-text');
                        const statusDot = document.getElementById('status-dot');
                        
                        if (data.is_running) {
                            if (data.is_scanning) {
                                statusText.textContent = 'Scanning';
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
                        document.getElementById('cpu-usage').textContent = `${data.cpu_usage?.toFixed(1) || 0}%`;
                        document.getElementById('memory-usage').textContent = `${data.memory_usage?.toFixed(1) || 0}%`;
                        
                        // Format uptime
                        const uptime = data.uptime_seconds || 0;
                        const hours = Math.floor(uptime / 3600);
                        const minutes = Math.floor((uptime % 3600) / 60);
                        document.getElementById('uptime').textContent = hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
                        
                        document.getElementById('market-regime').textContent = 
                            (data.market_regime || 'normal').charAt(0).toUpperCase() + 
                            (data.market_regime || 'normal').slice(1);
                    }
                    
                    updatePerformanceMetrics(metrics) {
                        document.getElementById('total-scans').textContent = metrics.total_scans || 0;
                        document.getElementById('win-rate').textContent = `${((metrics.win_rate || 0) * 100).toFixed(1)}%`;
                        document.getElementById('signals-generated').textContent = metrics.signals_generated || 0;
                        document.getElementById('avg-scan-time').textContent = `${(metrics.avg_scan_time || 0).toFixed(1)}s`;
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
                    
                    updateSignalsList(signals) {
                        const signalsList = document.getElementById('signals-list');
                        const activeCount = document.getElementById('active-count');
                        
                        activeCount.textContent = `${signals.length} Active`;
                        
                        if (signals.length === 0) {
                            signalsList.innerHTML = `
                                <div class="empty-state">
                                    <h4>No Active Signals</h4>
                                    <p>Scanning markets for high-probability setups...</p>
                                </div>
                            `;
                            return;
                        }
                        
                        signalsList.innerHTML = signals.map(signal => {
                            const pnlClass = signal.live_pnl_usd >= 0 ? 'pnl-positive' : 'pnl-negative';
                            const isInstitutional = signal.signal_grade === 'institutional';
                            
                            return `
                                <div class="signal-card">
                                    ${isInstitutional ? '<div class="institutional-badge">Institutional</div>' : ''}
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
                                            <div class="signal-metric-label">TP Progress</div>
                                            <div class="signal-metric-value">${signal.tp_progress.toFixed(1)}%</div>
                                            <div class="progress-container">
                                                <div class="progress-bar tp" style="width: ${Math.min(100, signal.tp_progress)}%"></div>
                                            </div>
                                        </div>
                                        <div class="signal-metric">
                                            <div class="signal-metric-label">Confidence</div>
                                            <div class="signal-metric-value">${signal.confidence}%</div>
                                        </div>
                                        <div class="signal-metric">
                                            <div class="signal-metric-label">R:R Ratio</div>
                                            <div class="signal-metric-value">${(signal.analysis_data?.risk_reward_ratio || 0).toFixed(1)}</div>
                                        </div>
                                    </div>
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
                
                // Initialize dashboard when DOM is loaded
                document.addEventListener('DOMContentLoaded', () => {
                    new TradingBotDashboard();
                    
                    // Request notification permission
                    if ('Notification' in window && Notification.permission === 'default') {
                        Notification.requestPermission();
                    }
                });
            </script>
        </body>
        </html>
        '''

# Global bot instance
bot_instance = None

def get_bot():
    """Get or create bot instance"""
    global bot_instance
    if bot_instance is None:
        bot_instance = ProductionTradingBot()
    return bot_instance

# FastAPI app
app = get_bot().app

@app.on_event("startup")
async def startup_event():
    """Start bot on server startup"""
    bot = get_bot()
    if not bot.running:
        bot.running = True
        asyncio.create_task(bot.production_scan_cycle())
        logger.info("Production trading bot auto-started")

@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown"""
    bot = get_bot()
    bot.running = False
    logger.info("Production trading bot shutdown complete")

if __name__ == "__main__":
    try:
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        logger.info("Starting Production Crypto Trading Bot Server...")
        logger.info("Features: Institutional Analysis, Smart Money Concepts, Real-time Monitoring")
        
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Graceful shutdown initiated...")
        if bot_instance:
            bot_instance.running = False
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)