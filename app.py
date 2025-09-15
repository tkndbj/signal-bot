class TradingBot:
    def __init__(self):
        self.database = Database()
        self.analyzer = AdvancedTradingAnalyzer(database=self.database)
        self.app = FastAPI(title="Crypto Trading Bot", version="2.1.0")
        self.websocket_connections = set()
        self.running = False
        self.scan_interval = 900  # Changed from 300 to 900 (15 minutes)
        self.last_scan_time = None
        self.scan_count = 0
        self.max_signals_per_scan = 3  # Add limit to prevent spam
        self.signal_cooldown = {}  # Track last signal time per coin
        
        # Setup FastAPI routes
        self.setup_routes()
        
        # Log bot initialization
        self.database.log_bot_activity(
            'INFO', 'SYSTEM', 'Trading bot initialized',
            f'Monitoring {len(self.analyzer.coins)} coins, scan interval: {self.scan_interval}s'
        )
    
    async def market_scan_cycle(self):
        """Main market scanning cycle with SIGNAL LIMITING"""
        while self.running:
            try:
                self.database.log_bot_activity(
                    'INFO', 'MARKET_SCANNER', 'Starting market scan cycle',
                    f'Scan #{self.scan_count + 1}'
                )
                
                # Scan all coins for signals
                signals = await self.analyzer.scan_all_coins()
                
                # Sort signals by confidence
                signals.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Apply limits and cooldowns
                processed_signals = []
                for signal in signals:
                    # Check cooldown (don't generate same coin signal within 1 hour)
                    coin = signal['coin']
                    last_signal_time = self.signal_cooldown.get(coin, 0)
                    current_time = time.time()
                    
                    if current_time - last_signal_time < 3600:  # 1 hour cooldown
                        self.database.log_bot_activity(
                            'INFO', 'SIGNAL_FILTER', f'Signal for {coin} filtered (cooldown)',
                            f'Last signal was {int((current_time - last_signal_time)/60)} minutes ago'
                        )
                        continue
                    
                    # Check if we haven't exceeded max signals
                    if len(processed_signals) >= self.max_signals_per_scan:
                        self.database.log_bot_activity(
                            'INFO', 'SIGNAL_FILTER', f'Max signals reached ({self.max_signals_per_scan})',
                            f'Skipping remaining {len(signals) - len(processed_signals)} signals'
                        )
                        break
                    
                    # Add to processed signals
                    processed_signals.append(signal)
                    self.signal_cooldown[coin] = current_time
                
                # Process filtered signals
                for signal in processed_signals:
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
                    f'Found {len(processed_signals)} valid signals from {len(signals)} potential. Total scans: {self.scan_count}'
                )
                
                # Broadcast update to clients
                await self.broadcast_to_clients({
                    "type": "scan_completed",
                    "scan_count": self.scan_count,
                    "signals_found": len(processed_signals)
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
        """Check active signals for exit conditions with PROPER PRICE TRACKING"""
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
                    # Get FRESH price data
                    current_price = await self.analyzer.get_current_price(signal['coin'])
                    if current_price <= 0:
                        continue
                    
                    # Update current price in database
                    self.database.update_signal_price(signal['signal_id'], current_price)
                    
                    # Calculate actual P&L
                    entry_price = signal['entry_price']
                    
                    # Check exit conditions based on ACTUAL prices
                    exit_reason = None
                    
                    if signal['direction'] == 'LONG':
                        # For LONG: profit when price goes UP
                        actual_pnl_percent = ((current_price - entry_price) / entry_price) * 100
                        
                        if current_price >= signal['take_profit']:
                            exit_reason = "Take Profit Hit"
                        elif current_price <= signal['stop_loss']:
                            exit_reason = "Stop Loss Hit"
                        elif actual_pnl_percent >= 5.0:  # Take partial profit at 5% (50% with 10x)
                            exit_reason = "Partial TP (5% move)"
                    else:  # SHORT
                        # For SHORT: profit when price goes DOWN
                        actual_pnl_percent = ((entry_price - current_price) / entry_price) * 100
                        
                        if current_price <= signal['take_profit']:
                            exit_reason = "Take Profit Hit"
                        elif current_price >= signal['stop_loss']:
                            exit_reason = "Stop Loss Hit"
                        elif actual_pnl_percent >= 5.0:  # Take partial profit at 5%
                            exit_reason = "Partial TP (5% move)"
                    
                    # Check time-based exit (close after 24 hours)
                    signal_age = (datetime.now() - datetime.fromisoformat(signal['created_at'])).total_seconds()
                    if signal_age > 86400:  # 24 hours
                        exit_reason = "Time limit (24h)"
                    
                    # Close signal if exit condition met
                    if exit_reason:
                        success = self.database.close_signal(signal['signal_id'], current_price, exit_reason)
                        
                        if success:
                            # Calculate final P&L for broadcast
                            if signal['direction'] == 'LONG':
                                final_pnl = ((current_price - entry_price) / entry_price) * 100 * 10
                            else:
                                final_pnl = ((entry_price - current_price) / entry_price) * 100 * 10
                            
                            # Broadcast signal closure
                            await self.broadcast_to_clients({
                                "type": "signal_closed",
                                "signal_id": signal['signal_id'],
                                "exit_price": current_price,
                                "exit_reason": exit_reason,
                                "pnl_percent": round(final_pnl, 2)
                            })
                            
                            self.database.log_bot_activity(
                                'INFO', 'SIGNAL_MONITOR', f'Signal closed: {signal["coin"]} {signal["direction"]}',
                                f'Exit: ${current_price}, Reason: {exit_reason}, P&L: {final_pnl:.2f}%',
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