#!/usr/bin/env python3
"""
Professional Crypto Trading Bot with Enhanced Logging and Monitoring
"""

import asyncio
import json
import logging
import os
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
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
        self.app = FastAPI(title="Crypto Trading Bot", version="2.1.0")
        self.websocket_connections = set()
        self.running = False
        self.scan_interval = 300  # 5 minutes
        self.last_scan_time = None
        self.scan_count = 0
        
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
            with open("templates/dashboard.html", "r") as f:
                return HTMLResponse(content=f.read())
        
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
            """Get active trading signals"""
            try:
                signals = self.database.get_active_signals()
                
                # Update current prices for active signals
                for signal in signals:
                    try:
                        current_price = await self.analyzer.get_current_price(signal['coin'])
                        if current_price > 0:
                            self.database.update_signal_price(signal['signal_id'], current_price)
                            signal['current_price'] = current_price
                    except Exception as e:
                        logger.error(f"Error updating price for {signal['coin']}: {e}")
                
                return JSONResponse(content={"signals": signals})
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
        
        @self.app.get("/api/chart/{symbol}")
        async def get_chart_data(symbol: str):
            """Get chart data for a symbol"""
            try:
                df = await self.analyzer.get_market_data(f"{symbol}/USDT", "1h", 100)
                
                if df.empty:
                    return JSONResponse(content={"error": "No data available"})
                
                df = self.analyzer.calculate_technical_indicators(df)
                
                chart_data = {
                    "timestamps": [t.isoformat() for t in df.index],
                    "prices": {
                        "open": df['open'].tolist(),
                        "high": df['high'].tolist(),
                        "low": df['low'].tolist(),
                        "close": df['close'].tolist()
                    },
                    "volume": df['volume'].tolist(),
                    "indicators": {
                        "rsi": df['rsi'].tolist(),
                        "macd": df['macd'].tolist(),
                        "bb_upper": df['bb_upper'].tolist(),
                        "bb_lower": df['bb_lower'].tolist()
                    }
                }
                
                return JSONResponse(content=chart_data)
            except Exception as e:
                logger.error(f"Error getting chart data for {symbol}: {e}")
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
                    "active_connections": len(self.websocket_connections)
                })
            except Exception as e:
                logger.error(f"Error getting system stats: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)
    
    async def broadcast_to_clients(self, message: Dict):
        """Broadcast message to all connected WebSocket clients"""
        if not self.websocket_connections:
            return
        
        disconnected = set()
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected clients
        self.websocket_connections -= disconnected
    
    async def market_scan_cycle(self):
        """Main market scanning cycle"""
        while self.running:
            try:
                self.database.log_bot_activity(
                    'INFO', 'MARKET_SCANNER', 'Starting market scan cycle',
                    f'Scan #{self.scan_count + 1}'
                )
                
                # Scan all coins for signals
                signals = await self.analyzer.scan_all_coins()
                
                # Process new signals
                for signal in signals:
                    success = self.database.save_signal(signal)
                    if success:
                        # Broadcast new signal to clients
                        await self.broadcast_to_clients({
                            "type": "new_signal",
                            "signal": signal
                        })
                        
                        self.database.log_bot_activity(
                            'INFO', 'SIGNAL_MANAGER', f'New signal saved: {signal["coin"]} {signal["direction"]}',
                            f'Entry: ${signal["entry_price"]}, Confidence: {signal["confidence"]}%',
                            signal["coin"]
                        )
                
                # Check active signals for exit conditions
                await self.check_signal_exits()
                
                # Update scan statistics
                self.scan_count += 1
                self.last_scan_time = datetime.now()
                
                self.database.log_bot_activity(
                    'INFO', 'MARKET_SCANNER', 'Market scan completed',
                    f'Found {len(signals)} new signals. Total scans: {self.scan_count}'
                )
                
                # Broadcast update to clients
                await self.broadcast_to_clients({
                    "type": "scan_completed",
                    "scan_count": self.scan_count,
                    "signals_found": len(signals)
                })
                
            except Exception as e:
                self.database.log_bot_activity(
                    'ERROR', 'MARKET_SCANNER', 'Market scan failed',
                    str(e)
                )
                logger.error(f"Error in market scan cycle: {e}")
            
            # Wait for next scan
            await asyncio.sleep(self.scan_interval)
    
    async def check_signal_exits(self):
        """Check active signals for exit conditions"""
        try:
            active_signals = self.database.get_active_signals()
            
            if not active_signals:
                return
            
            self.database.log_bot_activity(
                'DEBUG', 'SIGNAL_MONITOR', 'Checking signal exits',
                f'Monitoring {len(active_signals)} active signals'
            )
            
            for signal in active_signals:
                try:
                    current_price = await self.analyzer.get_current_price(signal['coin'])
                    if current_price <= 0:
                        continue
                    
                    # Update current price
                    self.database.update_signal_price(signal['signal_id'], current_price)
                    
                    # Check exit conditions
                    exit_reason = None
                    
                    if signal['direction'] == 'LONG':
                        if current_price >= signal['take_profit']:
                            exit_reason = "Take Profit Hit"
                        elif current_price <= signal['stop_loss']:
                            exit_reason = "Stop Loss Hit"
                    else:  # SHORT
                        if current_price <= signal['take_profit']:
                            exit_reason = "Take Profit Hit"
                        elif current_price >= signal['stop_loss']:
                            exit_reason = "Stop Loss Hit"
                    
                    # Close signal if exit condition met
                    if exit_reason:
                        success = self.database.close_signal(signal['signal_id'], current_price, exit_reason)
                        
                        if success:
                            # Broadcast signal closure
                            await self.broadcast_to_clients({
                                "type": "signal_closed",
                                "signal_id": signal['signal_id'],
                                "exit_price": current_price,
                                "exit_reason": exit_reason
                            })
                            
                            self.database.log_bot_activity(
                                'INFO', 'SIGNAL_MONITOR', f'Signal closed: {signal["coin"]} {signal["direction"]}',
                                f'Exit: ${current_price}, Reason: {exit_reason}',
                                signal['coin']
                            )
                
                except Exception as e:
                    self.database.log_bot_activity(
                        'ERROR', 'SIGNAL_MONITOR', f'Error checking signal {signal["signal_id"]}',
                        str(e), signal['coin']
                    )
                    logger.error(f"Error checking signal {signal['signal_id']}: {e}")
        
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
                # Clean old logs (keep last 1000 entries)
                self.database.log_bot_activity(
                    'DEBUG', 'MAINTENANCE', 'Running periodic maintenance',
                    'Cleaning old logs and optimizing database'
                )
                
                # Add any maintenance tasks here
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Error in maintenance cycle: {e}")
                await asyncio.sleep(3600)
    
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
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
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