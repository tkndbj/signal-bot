#!/usr/bin/env python3
import ccxt
import os
from dotenv import load_dotenv
import json

load_dotenv()

def test_bybit_balance():
    exchange = ccxt.bybit({
        'apiKey': os.getenv('BYBIT_API_KEY'),
        'secret': os.getenv('BYBIT_SECRET_KEY'),
        'options': {
            'defaultType': 'linear',
            'accountType': 'UNIFIED'
        }
    })
    
    print("Testing Bybit Balance Fetch Methods...")
    print("="*50)
    
    # Method 1: Direct V5 API
    try:
        response = exchange.private_get_v5_account_wallet_balance({'accountType': 'UNIFIED'})
        print("\nMethod 1 (V5 API Direct):")
        print(f"Status: {'SUCCESS' if response.get('retCode') == 0 else 'FAILED'}")
        print(f"Message: {response.get('retMsg')}")
        
        if response.get('retCode') == 0:
            for account in response['result']['list']:
                for coin in account.get('coin', []):
                    if coin['coin'] == 'USDT':
                        print(f"USDT Balance: {coin.get('walletBalance')}")
                        print(f"Available: {coin.get('availableToWithdraw')}")
    except Exception as e:
        print(f"Method 1 Failed: {e}")
    
    print("\n" + "="*50)
    
    # Method 2: CCXT fetch_balance
    try:
        balance = exchange.fetch_balance()
        print("\nMethod 2 (CCXT fetch_balance):")
        if 'USDT' in balance:
            print(f"Free: {balance['USDT'].get('free')}")
            print(f"Used: {balance['USDT'].get('used')}")
            print(f"Total: {balance['USDT'].get('total')}")
        else:
            print("USDT not found in balance")
    except Exception as e:
        print(f"Method 2 Failed: {e}")

if __name__ == "__main__":
    test_bybit_balance()