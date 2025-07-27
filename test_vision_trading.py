#!/usr/bin/env python3
"""
Test Vision Trading System
Demonstrates automated trading from SMC analysis
"""

import requests
import json
import time
from datetime import datetime

# Test analysis text from user's example
TEST_ANALYSIS = """
## 1. MARKET STRUCTURE ANALYSIS
- **HTF Trend Direction:** Bearish
- **Market Phase:** Transitional
- **Key Structural Evidence:** 
  - BOS at 3,311.160 (Bearish)
  - CHoCH at 3,338.240 (Bullish)

## 2. SMC ELEMENTS IDENTIFIED
- **Change of Character (CHoCH):** 
  - 3,338.240 on 4H timeframe
- **Break of Structure (BOS):** 
  - 3,311.160 (Bearish)
- **Order Blocks (OB):** 
  - Bearish OB: 3,338.240 - 3,360.000
  - Bullish OB: 3,280.000 - 3,300.000
- **Fair Value Gaps (FVG):** 
  - 3,320.000 - 3,330.000 (Unmitigated)
- **Liquidity Zones:** 
  - Equal highs at 3,338.240
  - Sell-side liquidity below 3,311.160
- **Volume Imbalance:** 
  - Around 3,320.000

## 3. DIRECTIONAL BIAS & CONFLUENCE
- **Primary Bias:** Short (High conviction)
- **Structural Evidence:** 
  - BOS at 3,311.160 supports bearish bias
- **Confluence Factors:** 
  - Bearish OB and BOS alignment
  - Unmitigated FVG in bearish direction

## 4. TRADING PLAN
### Entry Strategy:
- **Entry Zone:** 
  - 3,338.240 - 3,360.000
- **Confirmation Criteria:** 
  - Rejection from OB with bearish candlestick pattern
- **Entry Type:** 
  - Limit order within OB

### Risk Management:
- **Stop Loss:** 
  - 3,365.000 (Above OB)
- **Risk/Reward Ratio:** 
  - 1:3

### Profit Targets:
- **TP1:** 
  - 3,311.160 (Previous BOS level)
- **TP2:** 
  - 3,300.000 (Bullish OB)
- **TP3:** 
  - 3,280.000 (Extended target near liquidity zone)

## 5. RISK FACTORS & INVALIDATION
- **Invalidation Level:** 
  - Above 3,365.000
- **Risk Zones:** 
  - Above 3,360.000 where bearish setup fails
- **Key Considerations:** 
  - Monitor for macroeconomic news impacting gold prices
  - Watch for strong bullish momentum above OB
"""

def test_vision_trading_system(base_url="http://localhost:5003"):
    """Test the complete vision trading workflow"""
    
    print("🎯 QNTI Vision Trading System Test")
    print("=" * 50)
    
    # Step 1: Check vision trading status
    print("\n1. Checking vision trading system status...")
    try:
        response = requests.get(f"{base_url}/api/vision-trading/status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Vision Trading Status: {status}")
        else:
            print(f"❌ Failed to get status: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return

    # Step 2: Test analysis processing (preview mode)
    print("\n2. Testing analysis processing (preview mode)...")
    try:
        payload = {
            "analysis_text": TEST_ANALYSIS,
            "symbol": "XAUUSD",
            "auto_submit": False
        }
        
        response = requests.post(
            f"{base_url}/api/vision-trading/process-analysis",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print(f"✅ Trade Preview Created Successfully!")
                print(f"   Trade ID: {result['trade_id']}")
                print(f"   Symbol: {result['symbol']}")
                print(f"   Direction: {result['direction']}")
                print(f"   Entry Zone: {result['entry_zone']}")
                print(f"   Stop Loss: {result['stop_loss']}")
                print(f"   Take Profits: TP1={result['take_profits']['tp1']}, TP2={result['take_profits']['tp2']}")
                print(f"   Lot Size: {result['lot_size']}")
                print(f"   Confidence: {result['confidence']*100:.1f}%")
                print(f"   Status: {result['status']}")
            else:
                print(f"❌ Trade creation failed: {result['message']}")
                return
        else:
            print(f"❌ Request failed: {response.status_code} - {response.text}")
            return
            
    except Exception as e:
        print(f"❌ Error processing analysis: {e}")
        return

    # Step 3: Get vision trades
    print("\n3. Checking vision trades list...")
    try:
        response = requests.get(f"{base_url}/api/vision-trading/trades")
        if response.status_code == 200:
            trades = response.json()
            print(f"✅ Found {len(trades)} vision trades")
            for trade in trades:
                print(f"   • {trade['id']}: {trade['symbol']} {trade['direction']} ({trade['status']})")
        else:
            print(f"❌ Failed to get trades: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting trades: {e}")

    # Step 4: Test configuration
    print("\n4. Testing configuration management...")
    try:
        response = requests.get(f"{base_url}/api/vision-trading/config")
        if response.status_code == 200:
            config = response.json()
            print(f"✅ Current Configuration:")
            for key, value in config.items():
                print(f"   • {key}: {value}")
        else:
            print(f"❌ Failed to get config: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting config: {e}")

    # Step 5: Test auto-submit (if user confirms)
    print("\n5. Test auto-submit functionality? (y/N):", end=" ")
    user_input = input().strip().lower()
    
    if user_input == 'y':
        print("   Creating and auto-submitting trade...")
        try:
            payload = {
                "analysis_text": TEST_ANALYSIS,
                "symbol": "XAUUSD", 
                "auto_submit": True
            }
            
            response = requests.post(
                f"{base_url}/api/vision-trading/process-analysis",
                headers={"Content-Type": "application/json"},
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    print(f"✅ Trade Auto-Submitted Successfully!")
                    print(f"   Trade ID: {result['trade_id']}")
                    print(f"   Status: {result['status']}")
                else:
                    print(f"❌ Auto-submit failed: {result['message']}")
            else:
                print(f"❌ Request failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error auto-submitting: {e}")
    else:
        print("   Skipping auto-submit test")

    print("\n" + "=" * 50)
    print("🎯 Vision Trading Test Complete!")
    print("""
Next Steps:
1. Open your QNTI dashboard at http://localhost:5003
2. Go to the AI Vision Analysis panel
3. Upload a chart and get analysis
4. Use the new vision trading buttons to create automated trades
5. Monitor trades in real-time through the dashboard
    """)

def test_direct_integration():
    """Test direct integration with the vision trading module"""
    print("\n🧪 Testing Direct Vision Trading Integration")
    print("-" * 40)
    
    try:
        # Import the vision trading module directly
        from qnti_vision_trading import QNTIVisionTrader, process_vision_analysis_for_trading
        from qnti_core_system import QNTITradeManager
        from qnti_mt5_integration import QNTIMT5Bridge
        
        print("✅ Imports successful")
        
        # Initialize components (mock for testing)
        trade_manager = QNTITradeManager()
        print("✅ Trade Manager initialized")
        
        # Note: MT5 Bridge will fail if MT5 not connected, but that's expected
        try:
            mt5_bridge = QNTIMT5Bridge(trade_manager)
            print("✅ MT5 Bridge initialized")
        except Exception as e:
            print(f"⚠️  MT5 Bridge failed (expected if MT5 not running): {e}")
            # Create a mock MT5 bridge for testing
            class MockMT5Bridge:
                def __init__(self, trade_manager):
                    self.trade_manager = trade_manager
                    self.symbols = {}
                def get_mt5_status(self):
                    return {"account": {"balance": 10000}}
            mt5_bridge = MockMT5Bridge(trade_manager)
            print("✅ Mock MT5 Bridge created for testing")
        
        # Initialize vision trader
        vision_trader = QNTIVisionTrader(trade_manager, mt5_bridge)
        print("✅ Vision Trader initialized")
        
        # Test analysis processing
        vision_trade = process_vision_analysis_for_trading(
            analysis_text=TEST_ANALYSIS,
            symbol="XAUUSD",
            vision_trader=vision_trader,
            auto_submit=False
        )
        
        if vision_trade:
            print("✅ Vision trade created successfully!")
            print(f"   Analysis ID: {vision_trade.analysis_id}")
            print(f"   Symbol: {vision_trade.symbol}")
            print(f"   Direction: {vision_trade.direction}")
            print(f"   Entry Zone: {vision_trade.entry_zone_min} - {vision_trade.entry_zone_max}")
            print(f"   Stop Loss: {vision_trade.stop_loss}")
            print(f"   Take Profits: TP1={vision_trade.take_profit_1}, TP2={vision_trade.take_profit_2}")
            print(f"   Lot Size: {vision_trade.lot_size}")
            print(f"   Confidence: {vision_trade.confidence:.2f}")
        else:
            print("❌ Failed to create vision trade")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the vision trading module is in the same directory")
    except Exception as e:
        print(f"❌ Error in direct integration test: {e}")

if __name__ == "__main__":
    print("QNTI Vision Trading System - Test Suite")
    print("=====================================")
    
    # Test 1: Direct integration
    test_direct_integration()
    
    # Test 2: API integration
    print("\n🌐 Testing API Integration")
    print("-" * 40)
    print("Make sure QNTI system is running on port 5003")
    print("Press Enter to continue or Ctrl+C to exit...")
    try:
        input()
        test_vision_trading_system()
    except KeyboardInterrupt:
        print("\n👋 Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

    print("\n🎉 All tests completed!") 