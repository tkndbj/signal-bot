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
        self.scan_interval = 300  # 5 minutes
        self.last_scan_time = None
        self.scan_count = 0
        
        # Track recently created signals to prevent immediate closure
        self.new_signals: Set[str] = set()
        self.signal_grace_period = 60  # 60 seconds grace period before checking exits
        
        # Setup FastAPI routes
        self.setup_routes()
        
        # Log bot initialization
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
                return HTMLResponse(content=self.get_dashboard_html())
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.websocket_connections.add(websocket)
            
            self.database.log_bot_activity(
                'DEBUG', 'WEBSOCKET', 'New client connected',
                f'Total connections: {len(self.websocket_connections)}'
            )
            
            try:
                # Send initial data to new connection
                await websocket.send_text(json.dumps({
                    "type": "connection_established",
                    "message": "Connected to trading bot"
                }))
                
                while True:
                    await websocket.receive_text()  # Keep connection alive
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
                self.database.log_bot_activity(
                    'DEBUG', 'WEBSOCKET', 'Client disconnected',
                    f'Remaining connections: {len(self.websocket_connections)}'
                )
        
        @self.app.get("/api/signals")
        async def get_signals():
            """Get active trading signals with live prices"""
            try:
                signals = self.database.get_active_signals()
                
                # Update current prices for active signals
                for signal in signals:
                    try:
                        current_price = await self.analyzer.get_current_price(signal['coin'])
                        if current_price > 0:
                            # Calculate live P&L
                            if signal['direction'] == 'LONG':
                                pnl_percentage = ((current_price - signal['entry_price']) / signal['entry_price']) * 100 * 10
                            else:  # SHORT
                                pnl_percentage = ((signal['entry_price'] - current_price) / signal['entry_price']) * 100 * 10
                            
                            pnl_usd = (pnl_percentage / 100) * 1000
                            
                            # Update database and signal data
                            self.database.update_signal_price(signal['signal_id'], current_price)
                            signal['current_price'] = current_price
                            signal['live_pnl_usd'] = round(pnl_usd, 2)
                            signal['live_pnl_percentage'] = round(pnl_percentage, 2)
                            
                            # Calculate progress to TP and SL
                            if signal['direction'] == 'LONG':
                                tp_progress = ((current_price - signal['entry_price']) / 
                                             (signal['take_profit'] - signal['entry_price'])) * 100
                                sl_progress = ((signal['entry_price'] - current_price) / 
                                             (signal['entry_price'] - signal['stop_loss'])) * 100
                            else:
                                tp_progress = ((signal['entry_price'] - current_price) / 
                                             (signal['entry_price'] - signal['take_profit'])) * 100
                                sl_progress = ((current_price - signal['entry_price']) / 
                                             (signal['stop_loss'] - signal['entry_price'])) * 100
                            
                            signal['tp_progress'] = max(0, min(100, tp_progress))
                            signal['sl_progress'] = max(0, min(100, sl_progress))
                            
                    except Exception as e:
                        logger.error(f"Error updating price for {signal['coin']}: {e}")
                        signal['current_price'] = signal['entry_price']
                        signal['live_pnl_usd'] = 0
                        signal['live_pnl_percentage'] = 0
                
                return JSONResponse(content={"signals": signals, "count": len(signals)})
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
            """Get recent bot logs and analysis summary"""
            try:
                logs = self.database.get_recent_logs(50)
                analysis_summary = self.database.get_analysis_summary(24)
                
                return JSONResponse(content={
                    "logs": logs,
                    "analysis_summary": analysis_summary
                })
            except Exception as e:
                logger.error(f"Error getting logs: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)
        
        @self.app.get("/api/system")
        async def get_system_stats():
            """Get system performance statistics"""
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                return JSONResponse(content={
                    "cpu_usage": cpu_percent,
                    "memory_usage": memory.percent,
                    "uptime": time.time() - psutil.boot_time(),
                    "scan_count": self.scan_count,
                    "last_scan": self.last_scan_time.isoformat() if self.last_scan_time else None,
                    "active_connections": len(self.websocket_connections),
                    "is_running": self.running,
                    "scan_interval": self.scan_interval
                })
            except Exception as e:
                logger.error(f"Error getting system stats: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)
        
        @self.app.post("/api/control/start")
        async def start_scanning():
            """Start the bot scanning"""
            if not self.running:
                self.running = True
                asyncio.create_task(self.market_scan_cycle())
                return JSONResponse(content={"status": "started"})
            return JSONResponse(content={"status": "already running"})
        
        @self.app.post("/api/control/stop")
        async def stop_scanning():
            """Stop the bot scanning"""
            self.running = False
            return JSONResponse(content={"status": "stopped"})
    
    async def broadcast_to_clients(self, message: Dict):
        """Broadcast message to all connected WebSocket clients"""
        if not self.websocket_connections:
            return
        
        disconnected = set()
        message_str = json.dumps(message)
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message_str)
            except Exception as e:
                logger.debug(f"Client disconnected during broadcast: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected clients
        self.websocket_connections -= disconnected
    
    async def market_scan_cycle(self):
        """Main market scanning cycle with improved logic"""
        while self.running:
            try:
                scan_start = datetime.now()
                self.database.log_bot_activity(
                    'INFO', 'MARKET_SCANNER', 'Starting market scan cycle',
                    f'Scan #{self.scan_count + 1}'
                )
                
                # Broadcast scan start
                await self.broadcast_to_clients({
                    "type": "scan_started",
                    "scan_number": self.scan_count + 1,
                    "timestamp": scan_start.isoformat()
                })
                
                # Scan all coins for signals
                signals = await self.analyzer.scan_all_coins()
                
                # Process new signals
                new_signal_count = 0
                for signal in signals:
                    # Validate signal before saving
                    if self.validate_signal(signal):
                        success = self.database.save_signal(signal)
                        if success:
                            new_signal_count += 1
                            # Add to grace period set
                            self.new_signals.add(signal['signal_id'])
                            
                            # Schedule grace period removal
                            asyncio.create_task(self.remove_from_grace_period(
                                signal['signal_id'], 
                                self.signal_grace_period
                            ))
                            
                            # Broadcast new signal to clients
                            await self.broadcast_to_clients({
                                "type": "new_signal",
                                "signal": signal
                            })
                            
                            self.database.log_bot_activity(
                                'INFO', 'SIGNAL_MANAGER', 
                                f'New {signal["direction"]} signal: {signal["coin"]}',
                                f'Entry: ${signal["entry_price"]:.6f}, TP: ${signal["take_profit"]:.6f}, SL: ${signal["stop_loss"]:.6f}',
                                signal["coin"]
                            )
                    else:
                        logger.warning(f"Invalid signal rejected for {signal['coin']}")
                
                # Check active signals for exit conditions (excluding new ones)
                await self.check_signal_exits()
                
                # Update scan statistics
                self.scan_count += 1
                self.last_scan_time = datetime.now()
                scan_duration = (datetime.now() - scan_start).seconds
                
                self.database.log_bot_activity(
                    'INFO', 'MARKET_SCANNER', 'Market scan completed',
                    f'Found {new_signal_count} new signals in {scan_duration}s. Total scans: {self.scan_count}'
                )
                
                # Broadcast scan completion
                await self.broadcast_to_clients({
                    "type": "scan_completed",
                    "scan_count": self.scan_count,
                    "signals_found": new_signal_count,
                    "duration": scan_duration
                })
                
            except Exception as e:
                self.database.log_bot_activity(
                    'ERROR', 'MARKET_SCANNER', 'Market scan failed',
                    str(e)
                )
                logger.error(f"Error in market scan cycle: {e}")
            
            # Wait for next scan
            await asyncio.sleep(self.scan_interval)
    
    def validate_signal(self, signal: Dict) -> bool:
        """Validate signal parameters before saving"""
        try:
            # Check basic requirements
            if not all(key in signal for key in ['entry_price', 'take_profit', 'stop_loss', 'direction']):
                return False
            
            entry = signal['entry_price']
            tp = signal['take_profit']
            sl = signal['stop_loss']
            direction = signal['direction']
            
            # Validate LONG signal
            if direction == 'LONG':
                if tp <= entry:  # TP must be above entry
                    logger.warning(f"Invalid LONG signal: TP ${tp:.6f} <= Entry ${entry:.6f}")
                    return False
                if sl >= entry:  # SL must be below entry
                    logger.warning(f"Invalid LONG signal: SL ${sl:.6f} >= Entry ${entry:.6f}")
                    return False
                # Minimum 1% distance from entry to avoid immediate triggers
                if (tp - entry) / entry < 0.01:
                    logger.warning(f"LONG signal TP too close to entry")
                    return False
                if (entry - sl) / entry < 0.005:
                    logger.warning(f"LONG signal SL too close to entry")
                    return False
            
            # Validate SHORT signal
            elif direction == 'SHORT':
                if tp >= entry:  # TP must be below entry
                    logger.warning(f"Invalid SHORT signal: TP ${tp:.6f} >= Entry ${entry:.6f}")
                    return False
                if sl <= entry:  # SL must be above entry
                    logger.warning(f"Invalid SHORT signal: SL ${sl:.6f} <= Entry ${entry:.6f}")
                    return False
                # Minimum 1% distance from entry
                if (entry - tp) / entry < 0.01:
                    logger.warning(f"SHORT signal TP too close to entry")
                    return False
                if (sl - entry) / entry < 0.005:
                    logger.warning(f"SHORT signal SL too close to entry")
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
        """Check active signals for exit conditions with grace period"""
        try:
            active_signals = self.database.get_active_signals()
            
            if not active_signals:
                return
            
            checked_count = 0
            closed_count = 0
            
            for signal in active_signals:
                # Skip signals still in grace period
                if signal['signal_id'] in self.new_signals:
                    logger.debug(f"Skipping {signal['signal_id']} - still in grace period")
                    continue
                
                checked_count += 1
                
                try:
                    current_price = await self.analyzer.get_current_price(signal['coin'])
                    if current_price <= 0:
                        continue
                    
                    # Update current price and P&L
                    self.database.update_signal_price(signal['signal_id'], current_price)
                    
                    # Check exit conditions with small buffer to avoid false triggers
                    exit_reason = None
                    exit_price = current_price
                    
                    if signal['direction'] == 'LONG':
                        # Use 99.5% of TP and 100.5% of SL for buffer
                        if current_price >= signal['take_profit'] * 0.995:
                            exit_reason = "Take Profit Hit"
                            exit_price = signal['take_profit']
                        elif current_price <= signal['stop_loss'] * 1.005:
                            exit_reason = "Stop Loss Hit"
                            exit_price = signal['stop_loss']
                    else:  # SHORT
                        if current_price <= signal['take_profit'] * 1.005:
                            exit_reason = "Take Profit Hit"
                            exit_price = signal['take_profit']
                        elif current_price >= signal['stop_loss'] * 0.995:
                            exit_reason = "Stop Loss Hit"
                            exit_price = signal['stop_loss']
                    
                    # Close signal if exit condition met
                    if exit_reason:
                        success = self.database.close_signal(
                            signal['signal_id'], 
                            exit_price, 
                            exit_reason
                        )
                        
                        if success:
                            closed_count += 1
                            
                            # Calculate final P&L
                            if signal['direction'] == 'LONG':
                                pnl_percentage = ((exit_price - signal['entry_price']) / 
                                                signal['entry_price']) * 100 * 10
                            else:
                                pnl_percentage = ((signal['entry_price'] - exit_price) / 
                                                signal['entry_price']) * 100 * 10
                            
                            pnl_usd = (pnl_percentage / 100) * 1000
                            
                            # Broadcast signal closure
                            await self.broadcast_to_clients({
                                "type": "signal_closed",
                                "signal_id": signal['signal_id'],
                                "coin": signal['coin'],
                                "direction": signal['direction'],
                                "exit_price": exit_price,
                                "exit_reason": exit_reason,
                                "pnl_usd": round(pnl_usd, 2),
                                "pnl_percentage": round(pnl_percentage, 2)
                            })
                            
                            self.database.log_bot_activity(
                                'INFO', 'SIGNAL_MONITOR', 
                                f'Signal closed: {signal["coin"]} {signal["direction"]}',
                                f'Exit: ${exit_price:.6f}, Reason: {exit_reason}, P&L: ${pnl_usd:.2f} ({pnl_percentage:.1f}%)',
                                signal['coin']
                            )
                
                except Exception as e:
                    logger.error(f"Error checking signal {signal['signal_id']}: {e}")
            
            if checked_count > 0:
                logger.debug(f"Checked {checked_count} signals, closed {closed_count}")
        
        except Exception as e:
            self.database.log_bot_activity(
                'ERROR', 'SIGNAL_MONITOR', 'Signal exit check failed',
                str(e)
            )
            logger.error(f"Error checking signal exits: {e}")
    
    async def periodic_maintenance(self):
        """Periodic maintenance tasks"""
        while self.running:
            try:
                # Run maintenance every hour
                await asyncio.sleep(3600)
                
                self.database.log_bot_activity(
                    'DEBUG', 'MAINTENANCE', 'Running periodic maintenance',
                    'Cleaning old logs and optimizing database'
                )
                
                # Clean old signals (older than 7 days)
                self.database.clean_old_data(days=7)
                
            except Exception as e:
                logger.error(f"Error in maintenance cycle: {e}")
    
    async def start_bot(self):
        """Start the trading bot"""
        self.running = True
        
        self.database.log_bot_activity(
            'INFO', 'SYSTEM', 'Trading bot starting up',
            f'Scan interval: {self.scan_interval}s, Monitoring: {len(self.analyzer.coins)} coins'
        )
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self.market_scan_cycle()),
            asyncio.create_task(self.periodic_maintenance())
        ]
        
        logger.info("Trading bot started successfully")
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in bot main loop: {e}")
            self.database.log_bot_activity('ERROR', 'SYSTEM', 'Bot crashed', str(e))
        finally:
            self.running = False
    
    def stop_bot(self):
        """Stop the trading bot"""
        self.running = False
        self.database.log_bot_activity('INFO', 'SYSTEM', 'Trading bot shutting down', 'Graceful shutdown initiated')
        logger.info("Trading bot stopped")
    
    def get_dashboard_html(self) -> str:
        """Generate dashboard HTML if template file is missing"""
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crypto Trading Bot Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
                h1 { color: #4CAF50; }
                .status { padding: 10px; background: #2a2a2a; border-radius: 5px; margin: 10px 0; }
                .signal { padding: 10px; background: #2a2a2a; border: 1px solid #444; margin: 5px 0; border-radius: 5px; }
                .long { border-left: 4px solid #4CAF50; }
                .short { border-left: 4px solid #f44336; }
                .profit { color: #4CAF50; }
                .loss { color: #f44336; }
            </style>
        </head>
        <body>
            <h1>Crypto Trading Bot Dashboard</h1>
            <div class="status">
                <h2>System Status</h2>
                <p>Bot Status: <span id="bot-status">Checking...</span></p>
                <p>Active Signals: <span id="active-signals">0</span></p>
                <p>Total P&L: <span id="total-pnl">$0.00</span></p>
            </div>
            <div id="signals-container">
                <h2>Active Signals</h2>
                <div id="signals-list"></div>
            </div>
            <script>
                // Connect to WebSocket
                const ws = new WebSocket('ws://' + window.location.host + '/ws');
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    console.log('Received:', data);
                    updateDashboard();
                };
                
                // Update dashboard every 5 seconds
                setInterval(updateDashboard, 5000);
                
                async function updateDashboard() {
                    try {
                        // Get signals
                        const signalsResp = await fetch('/api/signals');
                        const signalsData = await signalsResp.json();
                        
                        // Get portfolio
                        const portfolioResp = await fetch('/api/portfolio');
                        const portfolioData = await portfolioResp.json();
                        
                        // Update UI
                        document.getElementById('bot-status').textContent = 'Running';
                        document.getElementById('bot-status').style.color = '#4CAF50';
                        document.getElementById('active-signals').textContent = signalsData.count || 0;
                        
                        const pnl = portfolioData.portfolio.total_pnl || 0;
                        const pnlElement = document.getElementById('total-pnl');
                        pnlElement.textContent = '$' + pnl.toFixed(2);
                        pnlElement.className = pnl >= 0 ? 'profit' : 'loss';
                        
                        // Update signals list
                        const signalsList = document.getElementById('signals-list');
                        signalsList.innerHTML = '';
                        
                        if (signalsData.signals && signalsData.signals.length > 0) {
                            signalsData.signals.forEach(signal => {
                                const div = document.createElement('div');
                                div.className = 'signal ' + signal.direction.toLowerCase();
                                
                                const pnlClass = signal.live_pnl_usd >= 0 ? 'profit' : 'loss';
                                
                                div.innerHTML = `
                                    <strong>${signal.coin} - ${signal.direction}</strong><br>
                                    Entry: $${signal.entry_price.toFixed(6)}<br>
                                    Current: $${(signal.current_price || signal.entry_price).toFixed(6)}<br>
                                    TP: $${signal.take_profit.toFixed(6)} | SL: $${signal.stop_loss.toFixed(6)}<br>
                                    P&L: <span class="${pnlClass}">$${(signal.live_pnl_usd || 0).toFixed(2)} (${(signal.live_pnl_percentage || 0).toFixed(1)}%)</span><br>
                                    Confidence: ${signal.confidence}%
                                `;
                                signalsList.appendChild(div);
                            });
                        } else {
                            signalsList.innerHTML = '<p>No active signals</p>';
                        }
                    } catch (error) {
                        console.error('Error updating dashboard:', error);
                        document.getElementById('bot-status').textContent = 'Error';
                        document.getElementById('bot-status').style.color = '#f44336';
                    }
                }
                
                // Initial update
                updateDashboard();
            </script>
        </body>
        </html>
        '''

# Global bot instance
bot = TradingBot()

# FastAPI app for uvicorn
app = bot.app

@app.on_event("startup")
async def startup_event():
    """Start bot when server starts"""
    asyncio.create_task(bot.start_bot())

@app.on_event("shutdown")
async def shutdown_event():
    """Stop bot when server shuts down"""
    bot.stop_bot()

if __name__ == "__main__":
    try:
        # Create necessary directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("templates", exist_ok=True)
        
        # Start the server
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        bot.stop_bot()
    except Exception as e:
        print(f"Failed to start bot: {e}")
        logger.error(f"Failed to start bot: {e}")