#!/usr/bin/env python3
"""
Professional Crypto Trading Bot with Enhanced Signal Management
"""

import asyncio
import json
import logging
import os
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Import our enhanced modules
from database import Database
from trading_analyzer import AdvancedTradingAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        self.database = Database()
        self.analyzer = AdvancedTradingAnalyzer(database=self.database)
        self.app = FastAPI(title="Crypto Trading Bot", version="3.0.0")
        self.websocket_connections = set()
        self.running = False
        self.scan_interval = 300  # 5 minutes between scans
        self.last_scan_time = None
        self.scan_count = 0
        self.startup_time = datetime.now()
        
        # Track recently created signals to prevent immediate closure
        self.new_signals: Set[str] = set()
        self.signal_grace_period = 120  # 2 minutes grace period
        
        # Prevent multiple simultaneous scans
        self.scanning_lock = asyncio.Lock()
        self.is_scanning = False
        
        # Setup FastAPI routes
        self.setup_routes()
        
        # Initialize bot state
        self.initialize_bot_state()
    
    def initialize_bot_state(self):
        """Initialize bot state on startup"""
        # Clean any stale active signals from previous runs
        try:
            # Get the last scan time from logs to determine if this is a fresh start
            recent_logs = self.database.get_recent_logs(10)
            system_start_logs = [log for log in recent_logs if 
                               log['component'] == 'SYSTEM' and 'initialized' in log['message']]
            
            # If no recent system logs, this might be a fresh start
            if not system_start_logs:
                # Close any signals that were active from previous sessions
                active_signals = self.database.get_active_signals()
                for signal in active_signals:
                    self.database.close_signal(
                        signal['signal_id'], 
                        signal['entry_price'], 
                        "System restart - stale signal"
                    )
                logger.info(f"Cleaned {len(active_signals)} stale signals from previous session")
        
        except Exception as e:
            logger.error(f"Error initializing bot state: {e}")
        
        # Log successful initialization
        self.database.log_bot_activity(
            'INFO', 'SYSTEM', 'Trading bot initialized',
            f'Monitoring {len(self.analyzer.coins)} coins, scan interval: {self.scan_interval}s'
        )
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        async def dashboard():
            """Serve the main dashboard"""
            try:
                with open("templates/dashboard.html", "r") as f:
                    return HTMLResponse(content=f.read())
            except FileNotFoundError:
                return HTMLResponse(content=self.get_minimal_dashboard())
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.websocket_connections.add(websocket)
            
            try:
                # Send initial connection confirmation
                await websocket.send_text(json.dumps({
                    "type": "connection_established",
                    "message": "Connected to trading bot",
                    "timestamp": datetime.now().isoformat()
                }))
                
                # Keep connection alive
                while True:
                    await asyncio.sleep(1)
                    
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
            except Exception as e:
                logger.debug(f"WebSocket error: {e}")
                self.websocket_connections.discard(websocket)
        
        @self.app.get("/api/signals")
        async def get_signals():
            """Get active trading signals with live prices"""
            try:
                signals = self.database.get_active_signals()
                
                # Update current prices and P&L for active signals
                updated_signals = []
                for signal in signals:
                    try:
                        # Get current price from exchange
                        current_price = await self.analyzer.get_current_price(signal['coin'])
                        
                        if current_price > 0:
                            # Update signal with current price and live P&L
                            self.database.update_signal_price(signal['signal_id'], current_price)
                            
                            # Calculate live P&L with 10x leverage
                            entry_price = signal['entry_price']
                            direction = signal['direction']
                            
                            if direction == 'LONG':
                                pnl_percentage = ((current_price - entry_price) / entry_price) * 100 * 10
                            else:  # SHORT
                                pnl_percentage = ((entry_price - current_price) / entry_price) * 100 * 10
                            
                            pnl_usd = (pnl_percentage / 100) * 1000  # $1000 position size
                            
                            # Calculate progress to targets
                            if direction == 'LONG':
                                tp_progress = ((current_price - entry_price) / 
                                             (signal['take_profit'] - entry_price)) * 100
                            else:
                                tp_progress = ((entry_price - current_price) / 
                                             (entry_price - signal['take_profit'])) * 100
                            
                            # Update signal data
                            signal['current_price'] = current_price
                            signal['live_pnl_usd'] = round(pnl_usd, 2)
                            signal['live_pnl_percentage'] = round(pnl_percentage, 2)
                            signal['tp_progress'] = max(0, min(100, tp_progress))
                            
                        else:
                            # If price fetch failed, use entry price
                            signal['current_price'] = signal['entry_price']
                            signal['live_pnl_usd'] = 0
                            signal['live_pnl_percentage'] = 0
                            signal['tp_progress'] = 0
                        
                        updated_signals.append(signal)
                        
                    except Exception as e:
                        logger.error(f"Error updating signal {signal['signal_id']}: {e}")
                        # Add signal with default values if update fails
                        signal['current_price'] = signal['entry_price']
                        signal['live_pnl_usd'] = 0
                        signal['live_pnl_percentage'] = 0
                        signal['tp_progress'] = 0
                        updated_signals.append(signal)
                
                return JSONResponse(content={
                    "signals": updated_signals, 
                    "count": len(updated_signals)
                })
                
            except Exception as e:
                logger.error(f"Error getting signals: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)
        
        @self.app.get("/api/portfolio")
        async def get_portfolio():
            """Get portfolio statistics"""
            try:
                portfolio = self.database.get_portfolio_stats()
                return JSONResponse(content={"portfolio": portfolio})
            except Exception as e:
                logger.error(f"Error getting portfolio: {e}")
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
            """Get system performance statistics"""
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                uptime = (datetime.now() - self.startup_time).total_seconds()
                
                return JSONResponse(content={
                    "cpu_usage": cpu_percent,
                    "memory_usage": memory.percent,
                    "uptime": uptime,
                    "scan_count": self.scan_count,
                    "last_scan": self.last_scan_time.isoformat() if self.last_scan_time else None,
                    "active_connections": len(self.websocket_connections),
                    "is_running": self.running,
                    "is_scanning": self.is_scanning,
                    "scan_interval": self.scan_interval
                })
            except Exception as e:
                logger.error(f"Error getting system stats: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)
        
        @self.app.post("/api/control/start")
        async def start_bot():
            """Start the trading bot if not already running"""
            if not self.running:
                self.running = True
                # Start the scanning task
                asyncio.create_task(self.market_scan_cycle())
                message = "Bot started successfully"
                logger.info(message)
                self.database.log_bot_activity('INFO', 'SYSTEM', message)
                return JSONResponse(content={"status": "started", "message": message})
            else:
                return JSONResponse(content={"status": "already_running", "message": "Bot is already running"})
        
        @self.app.post("/api/control/stop")
        async def stop_bot():
            """Stop the trading bot"""
            self.running = False
            message = "Bot stopped successfully"
            logger.info(message)
            self.database.log_bot_activity('INFO', 'SYSTEM', message)
            return JSONResponse(content={"status": "stopped", "message": message})
    
    async def broadcast_to_clients(self, message: Dict):
        """Broadcast message to all connected WebSocket clients"""
        if not self.websocket_connections:
            return
        
        disconnected = set()
        message_str = json.dumps(message)
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message_str)
            except Exception:
                disconnected.add(websocket)
        
        # Remove disconnected clients
        self.websocket_connections -= disconnected
    
    async def market_scan_cycle(self):
        """Main market scanning cycle with improved logic"""
        while self.running:
            try:
                # Prevent multiple simultaneous scans
                if self.scanning_lock.locked():
                    await asyncio.sleep(30)
                    continue
                
                async with self.scanning_lock:
                    self.is_scanning = True
                    scan_start = datetime.now()
                    
                    self.database.log_bot_activity(
                        'INFO', 'MARKET_SCANNER', 'Starting market scan',
                        f'Scan #{self.scan_count + 1}'
                    )
                    
                    # Broadcast scan start
                    await self.broadcast_to_clients({
                        "type": "scan_started",
                        "scan_number": self.scan_count + 1,
                        "timestamp": scan_start.isoformat()
                    })
                    
                    # Check for exit conditions on existing signals first
                    await self.check_signal_exits()
                    
                    # Only scan for new signals if we have capacity
                    active_signals = self.database.get_active_signals()
                    max_total_signals = 10  # Maximum total active signals
                    
                    new_signals_generated = 0
                    
                    if len(active_signals) < max_total_signals:
                        # Scan for new opportunities
                        new_signals = await self.analyzer.scan_all_coins()
                        
                        for signal in new_signals:
                            # Validate and save signal
                            if self.validate_signal(signal):
                                success = self.database.save_signal(signal)
                                if success:
                                    new_signals_generated += 1
                                    
                                    # Add to grace period protection
                                    self.new_signals.add(signal['signal_id'])
                                    asyncio.create_task(self.remove_from_grace_period(
                                        signal['signal_id'], self.signal_grace_period
                                    ))
                                    
                                    # Broadcast new signal
                                    await self.broadcast_to_clients({
                                        "type": "new_signal",
                                        "signal": signal
                                    })
                                    
                                    self.database.log_bot_activity(
                                        'INFO', 'SIGNAL_MANAGER', 
                                        f'New {signal["direction"]} signal: {signal["coin"]}',
                                        f'Entry: ${signal["entry_price"]:.6f}, Confidence: {signal["confidence"]}%'
                                    )
                            
                            # Stop if we reach max signals
                            if len(active_signals) + new_signals_generated >= max_total_signals:
                                break
                    
                    # Update scan statistics
                    self.scan_count += 1
                    self.last_scan_time = datetime.now()
                    scan_duration = (datetime.now() - scan_start).total_seconds()
                    self.is_scanning = False
                    
                    self.database.log_bot_activity(
                        'INFO', 'MARKET_SCANNER', 'Market scan completed',
                        f'Duration: {scan_duration:.1f}s, New signals: {new_signals_generated}'
                    )
                    
                    # Broadcast scan completion
                    await self.broadcast_to_clients({
                        "type": "scan_completed",
                        "scan_count": self.scan_count,
                        "signals_found": new_signals_generated,
                        "duration": scan_duration
                    })
                
            except Exception as e:
                self.is_scanning = False
                error_msg = f"Market scan failed: {str(e)}"
                logger.error(error_msg)
                self.database.log_bot_activity('ERROR', 'MARKET_SCANNER', error_msg)
            
            # Wait for next scan
            await asyncio.sleep(self.scan_interval)
    
    def validate_signal(self, signal: Dict) -> bool:
        """Validate signal parameters before saving"""
        try:
            # Check required fields
            required_fields = ['entry_price', 'take_profit', 'stop_loss', 'direction', 'coin']
            if not all(field in signal for field in required_fields):
                logger.warning(f"Signal missing required fields: {signal.get('coin', 'unknown')}")
                return False
            
            # Check for existing active signal for this coin
            active_signals = self.database.get_active_signals()
            active_coins = [s['coin'] for s in active_signals]
            if signal['coin'] in active_coins:
                logger.debug(f"Already have active signal for {signal['coin']}")
                return False
            
            entry = signal['entry_price']
            tp = signal['take_profit']
            sl = signal['stop_loss']
            direction = signal['direction']
            
            # Validate price relationships
            if direction == 'LONG':
                if tp <= entry or sl >= entry:
                    logger.warning(f"Invalid LONG signal prices for {signal['coin']}")
                    return False
            elif direction == 'SHORT':
                if tp >= entry or sl <= entry:
                    logger.warning(f"Invalid SHORT signal prices for {signal['coin']}")
                    return False
            
            # Check minimum risk-reward ratio
            if direction == 'LONG':
                risk = entry - sl
                reward = tp - entry
            else:
                risk = sl - entry
                reward = entry - tp
            
            if risk <= 0 or reward <= 0:
                logger.warning(f"Invalid risk/reward for {signal['coin']}")
                return False
            
            rr_ratio = reward / risk
            if rr_ratio < 1.5:  # Minimum 1.5:1 R:R
                logger.warning(f"R:R ratio too low for {signal['coin']}: {rr_ratio:.2f}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    
    async def remove_from_grace_period(self, signal_id: str, delay: int):
        """Remove signal from grace period after delay"""
        await asyncio.sleep(delay)
        self.new_signals.discard(signal_id)
        logger.debug(f"Signal {signal_id} grace period ended")
    
    async def check_signal_exits(self):
        """Check active signals for exit conditions"""
        try:
            active_signals = self.database.get_active_signals()
            
            for signal in active_signals:
                # Skip signals in grace period
                if signal['signal_id'] in self.new_signals:
                    continue
                
                try:
                    current_price = await self.analyzer.get_current_price(signal['coin'])
                    if current_price <= 0:
                        continue
                    
                    # Update current price
                    self.database.update_signal_price(signal['signal_id'], current_price)
                    
                    # Check exit conditions with buffer to avoid false triggers
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
                    
                    # Close signal if exit condition met
                    if exit_reason and exit_price:
                        success = self.database.close_signal(
                            signal['signal_id'], exit_price, exit_reason
                        )
                        
                        if success:
                            # Calculate final P&L
                            if signal['direction'] == 'LONG':
                                pnl_pct = ((exit_price - signal['entry_price']) / signal['entry_price']) * 100 * 10
                            else:
                                pnl_pct = ((signal['entry_price'] - exit_price) / signal['entry_price']) * 100 * 10
                            
                            pnl_usd = (pnl_pct / 100) * 1000
                            
                            # Broadcast signal closure
                            await self.broadcast_to_clients({
                                "type": "signal_closed",
                                "signal_id": signal['signal_id'],
                                "coin": signal['coin'],
                                "exit_reason": exit_reason,
                                "pnl_usd": round(pnl_usd, 2),
                                "pnl_percentage": round(pnl_pct, 2)
                            })
                            
                            self.database.log_bot_activity(
                                'INFO', 'SIGNAL_MONITOR', 
                                f'Signal closed: {signal["coin"]} - {exit_reason}',
                                f'P&L: ${pnl_usd:.2f} ({pnl_pct:.1f}%)'
                            )
                
                except Exception as e:
                    logger.error(f"Error checking signal {signal['signal_id']}: {e}")
        
        except Exception as e:
            logger.error(f"Error in signal exit check: {e}")
    
    def get_minimal_dashboard(self) -> str:
        """Minimal dashboard if HTML file not found"""
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crypto Trading Bot</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
                .container { max-width: 1200px; margin: 0 auto; }
                .status { padding: 15px; background: #2a2a2a; border-radius: 8px; margin: 10px 0; }
                .signals { padding: 15px; background: #2a2a2a; border-radius: 8px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Crypto Trading Bot Dashboard</h1>
                <div class="status">
                    <h2>System Status</h2>
                    <p>Status: <span id="status">Loading...</span></p>
                    <p>Active Signals: <span id="signal-count">0</span></p>
                </div>
                <div class="signals">
                    <h2>Active Signals</h2>
                    <div id="signals-list">Loading...</div>
                </div>
            </div>
            <script>
                async function updateDashboard() {
                    try {
                        const response = await fetch('/api/signals');
                        const data = await response.json();
                        document.getElementById('status').textContent = 'Running';
                        document.getElementById('signal-count').textContent = data.count || 0;
                        
                        const signalsList = document.getElementById('signals-list');
                        if (data.signals && data.signals.length > 0) {
                            signalsList.innerHTML = data.signals.map(signal => 
                                `<div style="border: 1px solid #444; padding: 10px; margin: 5px 0; border-radius: 4px;">
                                    <strong>${signal.coin} - ${signal.direction}</strong><br>
                                    Entry: $${signal.entry_price.toFixed(6)}<br>
                                    P&L: $${(signal.live_pnl_usd || 0).toFixed(2)}
                                </div>`
                            ).join('');
                        } else {
                            signalsList.innerHTML = '<p>No active signals</p>';
                        }
                    } catch (error) {
                        document.getElementById('status').textContent = 'Error';
                        console.error('Error:', error);
                    }
                }
                
                setInterval(updateDashboard, 5000);
                updateDashboard();
            </script>
        </body>
        </html>
        '''

# Global bot instance
bot = None

def get_bot():
    global bot
    if bot is None:
        bot = TradingBot()
    return bot

# FastAPI app
app = get_bot().app

@app.on_event("startup")
async def startup_event():
    """Start bot when server starts"""
    bot = get_bot()
    if not bot.running:
        bot.running = True
        # Start scanning task
        asyncio.create_task(bot.market_scan_cycle())
        logger.info("Trading bot started on server startup")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop bot when server shuts down"""
    bot = get_bot()
    bot.running = False
    logger.info("Trading bot stopped on server shutdown")

if __name__ == "__main__":
    try:
        # Create necessary directories
        os.makedirs("data", exist_ok=True)
        
        # Start the server
        logger.info("Starting Crypto Trading Bot Server...")
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        if bot:
            bot.running = False
    except Exception as e:
        logger.error(f"Failed to start server: {e}")