#!/usr/bin/env python3
"""
Final Chat Bot Test - Verify all issues are resolved
Tests account progress questions and other chat functionality
"""

import requests
import json
import time

def test_final_chat_fix():
    """Test the fully fixed chat bot system"""
    print("🎯 FINAL CHAT BOT TEST - Verify All Fixes")
    print("=" * 60)
    
    base_url = "http://localhost:5002"
    
    # Test 1: Check if system is running
    print("\n1️⃣ Testing System Status...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ System is running properly")
        else:
            print(f"❌ System status issue: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ System not accessible: {e}")
        return False
    
    # Test 2: Check advisor status endpoint
    print("\n2️⃣ Testing Advisor Status...")
    try:
        response = requests.get(f"{base_url}/api/advisor/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Advisor Status: Online={status.get('advisor_online')}")
        else:
            print(f"❌ Advisor status error: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Advisor status check failed: {e}")
    
    # Test 3: Account progress questions
    print("\n3️⃣ Testing Account Progress Questions...")
    account_questions = [
        "what do you think of my account progress",
        "how is my trading performance", 
        "can you analyze my account progress",
        "what are my trading results"
    ]
    
    for question in account_questions:
        print(f"\n📝 Testing: '{question}'")
        try:
            response = requests.post(f"{base_url}/advisor/chat", 
                json={
                    "message": question,
                    "user_id": "test_user",
                    "session_id": f"test_{int(time.time())}"
                },
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    response_text = data.get('response', '')
                    
                    # Check if it's giving proper account-focused response
                    if any(keyword in response_text.lower() for keyword in ['account', 'performance', 'trading', 'progress']):
                        print("✅ Proper account-focused response")
                    else:
                        print("⚠️ Response may not be account-focused")
                        print(f"Response preview: {response_text[:200]}...")
                else:
                    print(f"❌ Chat failed: {data.get('error')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
        
        time.sleep(2)  # Brief pause between requests
    
    # Test 4: Gold analysis (check for format errors)
    print("\n4️⃣ Testing Gold Analysis...")
    try:
        response = requests.post(f"{base_url}/advisor/chat", 
            json={
                "message": "what's the current gold price and analysis",
                "user_id": "test_user", 
                "session_id": f"gold_test_{int(time.time())}"
            },
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                response_text = data.get('response', '')
                if 'Invalid format specifier' in response_text:
                    print("❌ Still has format specifier errors")
                else:
                    print("✅ Gold analysis working without format errors")
            else:
                print(f"❌ Gold analysis failed: {data.get('error')}")
        else:
            print(f"❌ Gold analysis HTTP error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Gold analysis request failed: {e}")
    
    # Test 5: Check response times
    print("\n5️⃣ Testing Response Times...")
    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/advisor/chat", 
            json={
                "message": "hello",
                "user_id": "speed_test",
                "session_id": f"speed_{int(time.time())}"
            },
            timeout=30
        )
        end_time = time.time()
        response_time = end_time - start_time
        
        if response_time < 10:
            print(f"✅ Good response time: {response_time:.2f}s")
        else:
            print(f"⚠️ Slow response time: {response_time:.2f}s")
            
    except Exception as e:
        print(f"❌ Speed test failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 FINAL TEST COMPLETE!")
    print("\n🌐 Access your chat interface at:")
    print(f"   {base_url}/dashboard/forex_advisor_chat.html")
    
if __name__ == "__main__":
    test_final_chat_fix() 