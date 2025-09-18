#!/usr/bin/env python3
import ccxt
import os
from dotenv import load_dotenv
import json

load_dotenv()

def test_bybit_balance():
    api_key = os.getenv('BYBIT_API_KEY')
    secret = os.getenv('BYBIT_SECRET_KEY')
    
    print(f"API Key exists: {bool(api_key)}")
    print(f"Secret exists: {bool(secret)}")
    print("="*50)
    
    exchange = ccxt.bybit({
        'apiKey': api_key,
        'secret': secret,
        'options': {
            'defaultType': 'linear'
        }
    })
    
    # Test 1: Try CONTRACT account type
    print("\nTest 1: CONTRACT account type")
    try:
        response = exchange.private_get_v5_account_wallet_balance({'accountType': 'CONTRACT'})
        print(f"Result: {response.get('retCode')} - {response.get('retMsg')}")
        if response.get('retCode') == 0:
            print(json.dumps(response.get('result', {}), indent=2))
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test 2: Try without specifying account type
    print("\nTest 2: Default fetch_balance")
    try:
        exchange2 = ccxt.bybit({
            'apiKey': api_key,
            'secret': secret
        })
        balance = exchange2.fetch_balance()
        print(f"Success! USDT balance: {balance.get('USDT', {})}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test 3: Try SPOT account
    print("\nTest 3: SPOT account type")
    try:
        response = exchange.private_get_v5_account_wallet_balance({'accountType': 'SPOT'})
        print(f"Result: {response.get('retCode')} - {response.get('retMsg')}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test 4: Get account info
    print("\nTest 4: Account Info")
    try:
        response = exchange.private_get_v5_user_query_api()
        print(f"API Key Info: {response.get('retMsg')}")
        if response.get('retCode') == 0:
            print(f"Permissions: {response.get('result', {})}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_bybit_balance()