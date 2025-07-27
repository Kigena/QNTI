#!/usr/bin/env python3
"""
Test Advisor Status - Check if forex advisor is properly integrated
"""

import requests
import json

def test_advisor_status():
    """Test the advisor status endpoint"""
    print("📊 Testing Advisor Status")
    print("=" * 50)
    
    # Test the status endpoint
    status_url = "http://localhost:5002/api/advisor/status"
    
    print(f"📡 Testing: {status_url}")
    
    try:
        response = requests.get(status_url, timeout=10)
        
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Advisor Status Response:")
            print(json.dumps(data, indent=2))
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test if there's another endpoint
    print(f"\n📡 Testing chat endpoint with simple message...")
    
    chat_url = "http://localhost:5002/advisor/chat"
    simple_data = {
        "message": "hello",
        "user_id": "test",
        "session_id": "test"
    }
    
    try:
        response = requests.post(chat_url, json=simple_data, timeout=10)
        
        print(f"📊 Chat Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            response_text = data.get('response', '')
            
            # Check if it's the hardcoded response or proper advisor
            if "QNTI Professional Market Analysis" in response_text:
                print("❌ STILL getting hardcoded market analysis response")
            elif "forex" in response_text.lower() and "advisor" in response_text.lower():
                print("✅ Getting proper Forex Advisor response")
            else:
                print("🤔 Getting unknown response type")
                print(f"Preview: {response_text[:200]}...")
        else:
            print(f"❌ Chat Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Chat Error: {e}")

if __name__ == "__main__":
    test_advisor_status() 