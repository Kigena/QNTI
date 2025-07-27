#!/usr/bin/env python3
"""
Simple test for Forex Advisor API endpoint
"""

import requests
import json

def test_forex_advisor_api():
    """Test the forex advisor API endpoint"""
    print("🧪 Testing Forex Advisor API")
    print("=" * 50)
    
    # Test the chat endpoint
    url = "http://localhost:5002/advisor/chat"
    
    test_messages = [
        "What's the current outlook for gold?",
        "How should I manage my forex trading risk?",
        "What are your thoughts on EURUSD?"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}️⃣ Testing: {message}")
        
        try:
            response = requests.post(url, 
                json={
                    "message": message,
                    "session_id": f"test_session_{i}"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    print(f"   ✅ Response: {data.get('response', '')[:150]}...")
                else:
                    print(f"   ❌ API Error: {data.get('error', 'Unknown error')}")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                
        except requests.exceptions.ConnectionError:
            print(f"   ⚠️ Connection error - make sure QNTI system is running on port 5002")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n{'='*50}")
    print("🎯 Test complete!")
    print("\n💡 To start QNTI system:")
    print("   python qnti_main.py --port 5002")

if __name__ == "__main__":
    test_forex_advisor_api() 