#!/usr/bin/env python3
"""
Detailed Chat Debug - Full response analysis
"""

import requests
import json

def debug_detailed_chat():
    """Get full detailed response from chat endpoint"""
    print("🔍 DETAILED Chat Response Debug")
    print("=" * 70)
    
    url = "http://localhost:5002/advisor/chat"
    
    test_data = {
        "message": "what do you think of my account progress", 
        "user_id": "debug_user",
        "session_id": "debug_session_detailed"
    }
    
    print(f"📡 Request URL: {url}")
    print(f"📝 Payload: {json.dumps(test_data, indent=2)}")
    
    try:
        response = requests.post(url, json=test_data, timeout=15)
        
        print(f"\n📊 Response Status: {response.status_code}")
        print(f"📋 Full Response Headers:")
        for key, value in response.headers.items():
            print(f"  {key}: {value}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"\n✅ Full JSON Response:")
                print(json.dumps(data, indent=2))
                
            except json.JSONDecodeError as e:
                print(f"\n❌ JSON Decode Error: {e}")
                print(f"Raw Response Text:")
                print(response.text)
        else:
            print(f"\n❌ HTTP Error: {response.status_code}")
            print(f"Response Text: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_detailed_chat() 