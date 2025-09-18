#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import MLTradingBot

async def check():
    bot = MLTradingBot()
    balance = await bot.get_account_balance()
    print(f"Balance from bot: {balance}")
    
    # Also test the exchange directly
    balance2 = bot.analyzer.exchange.fetch_balance()
    print(f"Direct exchange balance: {balance2.get('USDT', {})}")

asyncio.run(check())