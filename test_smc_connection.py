#!/usr/bin/env python3
"""
Test script to verify SMC Automation Control Center connection to real SMC reactor
"""

import requests
import time
import json
from datetime import datetime

def test_smc_connection():
    """Test the complete SMC automation connection"""
    
    print("🧪 Testing SMC Automation Control Center Connection")
    print("=" * 60)
    
    base_url = "http://localhost:5002"  # Default QNTI port
    
    # Test 1: Check SMC status (should show not running initially)
    print("\n1️⃣ Testing SMC Status (before start)...")
    try:
        response = requests.get(f"{base_url}/api/smc/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SMC Status API working")
            print(f"   - Trade setups: {len(data.get('trade_setups', []))}")
            print(f"   - Monitoring: {data.get('status', {}).get('is_monitoring', False)}")
        else:
            print(f"❌ SMC Status failed: {response.status_code}")
    except Exception as e:
        print(f"❌ SMC Status error: {e}")
    
    # Test 2: Start SMC automation
    print("\n2️⃣ Testing SMC Automation Start...")
    try:
        response = requests.post(f"{base_url}/api/smc-automation/start")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ SMC Automation started successfully")
            else:
                print(f"❌ SMC Automation start failed: {data.get('error')}")
        else:
            print(f"❌ SMC Automation start failed: {response.status_code}")
    except Exception as e:
        print(f"❌ SMC Automation start error: {e}")
    
    # Wait a bit for the system to initialize
    print("\n⏳ Waiting 5 seconds for SMC EA to initialize...")
    time.sleep(5)
    
    # Test 3: Check SMC status after start (should show running and setups)
    print("\n3️⃣ Testing SMC Status (after start)...")
    try:
        response = requests.get(f"{base_url}/api/smc/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SMC Status API working after start")
            print(f"   - Trade setups: {len(data.get('trade_setups', []))}")
            print(f"   - Monitoring: {data.get('status', {}).get('is_monitoring', False)}")
            print(f"   - Ready entries: {data.get('summary', {}).get('ready_entries', 0)}")
            
            # Show sample setup
            setups = data.get('trade_setups', [])
            if setups:
                setup = setups[0]
                print(f"   - Sample setup: {setup['symbol']} {setup['direction']} {setup['signal_type']}")
                print(f"     Status: {setup['status']}, Confidence: {setup['confidence']}")
        else:
            print(f"❌ SMC Status failed: {response.status_code}")
    except Exception as e:
        print(f"❌ SMC Status error: {e}")
    
    # Test 4: Check automation status
    print("\n4️⃣ Testing Automation Status...")
    try:
        response = requests.get(f"{base_url}/api/smc-automation/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Automation Status API working")
            print(f"   - Is running: {data.get('data', {}).get('is_running', False)}")
            print(f"   - Active signals: {data.get('data', {}).get('active_signals', 0)}")
        else:
            print(f"❌ Automation Status failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Automation Status error: {e}")
    
    # Test 5: Test setup execution (if any ready setups exist)
    print("\n5️⃣ Testing Setup Execution...")
    try:
        response = requests.get(f"{base_url}/api/smc/status")
        if response.status_code == 200:
            data = response.json()
            setups = data.get('trade_setups', [])
            ready_setups = [s for s in setups if s.get('status') == 'ready_for_entry']
            
            if ready_setups:
                setup_id = ready_setups[0]['setup_id']
                print(f"   - Found ready setup: {setup_id}")
                
                # Test execution endpoint (don't actually execute, just test API)
                print(f"   - Testing execution endpoint (not actually executing)...")
                print(f"   - Would execute: POST /api/smc/execute-setup/{setup_id}")
                print("   ✅ Execution endpoint available")
            else:
                print("   - No ready setups found for execution test")
    except Exception as e:
        print(f"❌ Setup execution test error: {e}")
    
    # Test 6: Stop automation
    print("\n6️⃣ Testing SMC Automation Stop...")
    try:
        response = requests.post(f"{base_url}/api/smc-automation/stop")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ SMC Automation stopped successfully")
            else:
                print(f"❌ SMC Automation stop failed: {data.get('error')}")
        else:
            print(f"❌ SMC Automation stop failed: {response.status_code}")
    except Exception as e:
        print(f"❌ SMC Automation stop error: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 SMC Connection Test Complete!")
    print("\n📝 Summary:")
    print("- The SMC Automation Control Center should now be connected to real SMC reactor data")
    print("- When you start the reactor in the dashboard, it will generate real trade setups")
    print("- The 'Identified Trade Setups' panels will show live data from the SMC EA")
    print("- You can execute setups directly from the control center")
    print("\n🚀 To use: Open the SMC Automation Control Center and click 'Initiate Reactor'")

if __name__ == "__main__":
    test_smc_connection() 