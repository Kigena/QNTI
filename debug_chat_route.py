#!/usr/bin/env python3
"""
Debug Chat Route - Test which handler is processing the /advisor/chat endpoint
"""

import requests
import json

def debug_chat_route():
    """Test the chat route to see which handler is processing requests"""
    print("🔍 Debugging Chat Route Handler")
    print("=" * 60)
    
    url = "http://localhost:5002/advisor/chat"
    
    test_data = {
        "message": "what do you think of account progress", 
        "user_id": "debug_user",
        "session_id": "debug_session"
    }
    
    print(f"📡 Sending request to: {url}")
    print(f"📝 Payload: {json.dumps(test_data, indent=2)}")
    
    try:
        response = requests.post(url, json=test_data, timeout=10)
        
        print(f"\n📊 Response Status: {response.status_code}")
        print(f"📋 Response Headers:")
        for key, value in response.headers.items():
            if key.lower() in ['content-type', 'server', 'content-length']:
                print(f"  {key}: {value}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"\n✅ JSON Response:")
                print(f"  Success: {data.get('success')}")
                print(f"  Type: {data.get('type')}")
                print(f"  Data Sources: {data.get('data_sources', 'N/A')}")
                
                response_text = data.get('response', '')
                
                # Check response signature to identify which handler processed it
                if "QNTI Professional Market Analysis" in response_text:
                    print(f"\n🎯 HANDLER IDENTIFIED: Web Interface (OLD HARDCODED HANDLER)")
                    print(f"❌ This is the WRONG handler! Should be using Forex Advisor.")
                elif "forex" in response_text.lower() and "advisor" in response_text.lower():
                    print(f"\n🎯 HANDLER IDENTIFIED: Forex Financial Advisor (CORRECT)")
                    print(f"✅ This is the RIGHT handler!")
                else:
                    print(f"\n🎯 HANDLER IDENTIFIED: Unknown/Other")
                
                # Show first 300 chars of response
                preview = response_text[:300] + "..." if len(response_text) > 300 else response_text
                print(f"\n💬 Response Preview:")
                print(preview)
                
            except json.JSONDecodeError:
                print(f"\n❌ Response is not valid JSON:")
                print(response.text[:500])
        else:
            print(f"\n❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_chat_route() 