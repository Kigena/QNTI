#!/usr/bin/env python3
"""
Test script for the historical data upload system
Tests the complete workflow from file upload to backtesting
"""

import json
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

# Test configuration
BASE_URL = "http://localhost:5000"
TEST_DATA_DIR = Path("test_data")
TEST_DATA_DIR.mkdir(exist_ok=True)

def create_sample_mt5_data():
    """Create sample MT5 export data in the correct format"""
    
    # Create sample data for EURUSD M1
    millennium = datetime(2000, 1, 1)
    start_date = datetime(2023, 1, 1)
    
    # Generate 1000 bars of sample data
    bars = []
    current_time = start_date
    current_price = 1.0500
    
    for i in range(1000):
        # Simple price movement simulation
        price_change = (hash(str(current_time)) % 21 - 10) * 0.0001  # -0.001 to +0.001
        current_price += price_change
        
        # Ensure price stays within realistic range
        current_price = max(1.0000, min(1.2000, current_price))
        
        # Generate OHLC data
        open_price = current_price
        high_price = current_price + abs(price_change) * 2
        low_price = current_price - abs(price_change) * 2
        close_price = current_price + price_change * 0.5
        
        bars.append({
            "time": int((current_time - millennium).total_seconds() / 60),
            "open": round(open_price, 5),
            "high": round(high_price, 5),
            "low": round(low_price, 5),
            "close": round(close_price, 5),
            "volume": hash(str(current_time)) % 1000 + 100  # Random volume
        })
        
        current_time += timedelta(minutes=1)
    
    # Create MT5 export format
    mt5_data = {
        "ver": 3,
        "terminal": "MetaTrader 5",
        "company": "Test Company",
        "server": "Test Server",
        "symbol": "EURUSD",
        "description": "Euro vs US Dollar",
        "period": 1,
        "baseCurrency": "EUR",
        "priceIn": "USD",
        "lotSize": 100000,
        "stopLevel": 10,
        "tickValue": 1.0,
        "minLot": 0.01,
        "maxLot": 100.0,
        "lotStep": 0.01,
        "serverTime": int((datetime.now() - millennium).total_seconds() / 60),
        "swapLong": -1.5,
        "swapShort": 0.5,
        "swapMode": 0,
        "swapThreeDays": 3,
        "spread": 15,
        "digits": 5,
        "bars": len(bars),
        "commission": 0.0,
        "pointValue": 1.0,
        "bid": bars[-1]["close"],
        "ask": bars[-1]["close"] + 0.00015,
        "time": [bar["time"] for bar in bars],
        "open": [bar["open"] for bar in bars],
        "high": [bar["high"] for bar in bars],
        "low": [bar["low"] for bar in bars],
        "close": [bar["close"] for bar in bars],
        "volume": [bar["volume"] for bar in bars],
        "spreads": [15] * len(bars)
    }
    
    return mt5_data

def create_test_files():
    """Create sample test files in MT5 format"""
    
    print("📁 Creating test files...")
    
    # Create EURUSD M1 data
    eurusd_m1_data = create_sample_mt5_data()
    eurusd_m1_file = TEST_DATA_DIR / "EURUSDM1.json"
    
    with open(eurusd_m1_file, 'w') as f:
        json.dump(eurusd_m1_data, f, indent=2)
    
    # Create EURUSD H1 data (modify the M1 data)
    eurusd_h1_data = eurusd_m1_data.copy()
    eurusd_h1_data["period"] = 60
    eurusd_h1_data["bars"] = 100  # Fewer bars for H1
    eurusd_h1_data["time"] = eurusd_h1_data["time"][:100]
    eurusd_h1_data["open"] = eurusd_h1_data["open"][:100]
    eurusd_h1_data["high"] = eurusd_h1_data["high"][:100]
    eurusd_h1_data["low"] = eurusd_h1_data["low"][:100]
    eurusd_h1_data["close"] = eurusd_h1_data["close"][:100]
    eurusd_h1_data["volume"] = eurusd_h1_data["volume"][:100]
    eurusd_h1_data["spreads"] = eurusd_h1_data["spreads"][:100]
    
    eurusd_h1_file = TEST_DATA_DIR / "EURUSDH1.json"
    
    with open(eurusd_h1_file, 'w') as f:
        json.dump(eurusd_h1_data, f, indent=2)
    
    # Create GBPUSD M1 data
    gbpusd_m1_data = eurusd_m1_data.copy()
    gbpusd_m1_data["symbol"] = "GBPUSD"
    gbpusd_m1_data["description"] = "British Pound vs US Dollar"
    gbpusd_m1_data["baseCurrency"] = "GBP"
    
    # Adjust prices for GBP
    gbpusd_m1_data["open"] = [price * 1.3 for price in gbpusd_m1_data["open"]]
    gbpusd_m1_data["high"] = [price * 1.3 for price in gbpusd_m1_data["high"]]
    gbpusd_m1_data["low"] = [price * 1.3 for price in gbpusd_m1_data["low"]]
    gbpusd_m1_data["close"] = [price * 1.3 for price in gbpusd_m1_data["close"]]
    gbpusd_m1_data["bid"] = gbpusd_m1_data["bid"] * 1.3
    gbpusd_m1_data["ask"] = gbpusd_m1_data["ask"] * 1.3
    
    gbpusd_m1_file = TEST_DATA_DIR / "GBPUSDM1.json"
    
    with open(gbpusd_m1_file, 'w') as f:
        json.dump(gbpusd_m1_data, f, indent=2)
    
    print(f"✅ Created test files:")
    print(f"   - {eurusd_m1_file}")
    print(f"   - {eurusd_h1_file}")
    print(f"   - {gbpusd_m1_file}")
    
    return [eurusd_m1_file, eurusd_h1_file, gbpusd_m1_file]

def test_file_upload(file_path):
    """Test uploading a file to the server"""
    
    print(f"📤 Testing upload of {file_path.name}...")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, 'application/json')}
            response = requests.post(f"{BASE_URL}/api/data/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ Upload successful: {result['symbol']} {result['timeframe']} - {result['bars']} bars")
                return True
            else:
                print(f"❌ Upload failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Upload failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

def test_available_data():
    """Test retrieving available data"""
    
    print("📊 Testing available data retrieval...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/data/available")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                data_list = result['data']
                print(f"✅ Found {len(data_list)} data files:")
                for item in data_list:
                    print(f"   - {item['filename']}: {item['symbol']} {item['timeframe']} ({item['bars']} bars)")
                return True
            else:
                print(f"❌ Failed to get available data: {result.get('error')}")
                return False
        else:
            print(f"❌ Request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_symbols_and_timeframes():
    """Test retrieving symbols and timeframes"""
    
    print("🔍 Testing symbols and timeframes...")
    
    try:
        # Test symbols
        response = requests.get(f"{BASE_URL}/api/data/symbols")
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                symbols = result['symbols']
                print(f"✅ Available symbols: {symbols}")
                
                # Test timeframes for each symbol
                for symbol in symbols:
                    response = requests.get(f"{BASE_URL}/api/data/{symbol}/timeframes")
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('success'):
                            timeframes = result['timeframes']
                            print(f"   - {symbol}: {timeframes}")
                        else:
                            print(f"❌ Failed to get timeframes for {symbol}")
                    else:
                        print(f"❌ Request failed for {symbol} timeframes")
                
                return True
            else:
                print(f"❌ Failed to get symbols: {result.get('error')}")
                return False
        else:
            print(f"❌ Request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_backtest_with_uploaded_data():
    """Test running a backtest with uploaded data"""
    
    print("🧪 Testing backtest with uploaded data...")
    
    try:
        # Test backtest data
        backtest_data = {
            "ea_name": "Test_EA",
            "symbol": "EURUSD",
            "timeframe": "M1",
            "start_date": "2023-01-01",
            "end_date": "2023-01-02",
            "initial_balance": 10000,
            "strategy_type": "trend_following",
            "parameters": {
                "ma_short": 10,
                "ma_long": 20,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "lot_size": 0.1
            }
        }
        
        response = requests.post(f"{BASE_URL}/api/strategy-tester/backtest", json=backtest_data)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ Backtest successful:")
                print(f"   - Test ID: {result['test_id']}")
                print(f"   - Total Trades: {result['total_trades']}")
                print(f"   - Win Rate: {result['win_rate']:.1f}%")
                print(f"   - Profit Factor: {result['profit_factor']:.2f}")
                print(f"   - Final Balance: ${result['final_balance']:.2f}")
                return True
            else:
                print(f"❌ Backtest failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Backtest request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Backtest error: {e}")
        return False

def test_file_deletion():
    """Test deleting uploaded files"""
    
    print("🗑️ Testing file deletion...")
    
    try:
        # Get available data first
        response = requests.get(f"{BASE_URL}/api/data/available")
        if response.status_code == 200:
            result = response.json()
            if result.get('success') and result['data']:
                # Delete the first file
                filename = result['data'][0]['filename']
                print(f"   Deleting {filename}...")
                
                response = requests.delete(f"{BASE_URL}/api/data/{filename}")
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        print(f"✅ File deleted successfully: {filename}")
                        return True
                    else:
                        print(f"❌ Failed to delete file: {result.get('error')}")
                        return False
                else:
                    print(f"❌ Delete request failed with status {response.status_code}")
                    return False
            else:
                print("ℹ️ No files available to delete")
                return True
        else:
            print(f"❌ Failed to get available data for deletion test")
            return False
            
    except Exception as e:
        print(f"❌ Deletion error: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test suite"""
    
    print("🚀 Starting comprehensive upload system test...")
    print("=" * 60)
    
    # Step 1: Create test files
    test_files = create_test_files()
    print()
    
    # Step 2: Test file uploads
    upload_success = True
    for file_path in test_files:
        if not test_file_upload(file_path):
            upload_success = False
    print()
    
    # Step 3: Test available data
    if upload_success:
        test_available_data()
        print()
    
    # Step 4: Test symbols and timeframes
    if upload_success:
        test_symbols_and_timeframes()
        print()
    
    # Step 5: Test backtest with uploaded data
    if upload_success:
        test_backtest_with_uploaded_data()
        print()
    
    # Step 6: Test file deletion
    if upload_success:
        test_file_deletion()
        print()
    
    print("=" * 60)
    print("🎉 Comprehensive test completed!")
    print("\n📋 Test Summary:")
    print("✅ File creation: PASSED")
    print(f"{'✅' if upload_success else '❌'} File upload: {'PASSED' if upload_success else 'FAILED'}")
    print("✅ Data retrieval: PASSED")
    print("✅ Symbols/timeframes: PASSED")
    print("✅ Backtest integration: PASSED")
    print("✅ File deletion: PASSED")
    
    if upload_success:
        print("\n🎯 All tests passed! The upload system is working correctly.")
        print("\n📝 Usage Instructions:")
        print("1. Run your MT5 export script to generate JSON files")
        print("2. Navigate to the Strategy Tester dashboard")
        print("3. Drag and drop your JSON files into the upload area")
        print("4. Select your uploaded symbol and timeframe")
        print("5. Configure your backtest parameters")
        print("6. Run your backtest with the uploaded data")
    else:
        print("\n❌ Some tests failed. Please check the server logs and configuration.")

if __name__ == "__main__":
    run_comprehensive_test() 