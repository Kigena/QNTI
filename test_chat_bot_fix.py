#!/usr/bin/env python3
"""
Test Chat Bot Fix - Verify Account Progress Questions
Test that the chat bot now properly handles account progress questions
instead of giving generic market analysis responses.
"""

import requests
import json
import time

def test_chat_bot_fix():
    """Test the fixed chat bot with account progress questions"""
    print("🤖 Testing FIXED Chat Bot - Account Progress Questions")
    print("=" * 80)
    
    base_url = "http://localhost:5002"
    
    # Test questions that should trigger account_progress conversation type
    test_questions = [
        "what do you think of account progress",
        "how am I doing with my trading performance", 
        "can you review my account performance",
        "what are my trading results",
        "how is my profit and loss looking",
        "give me a performance review"
    ]
    
    print("🧪 Testing Account Progress Questions...")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Test {i}: '{question}'")
        
        try:
            response = requests.post(
                f"{base_url}/advisor/chat",
                json={
                    "message": question,
                    "user_id": "test_user",
                    "session_id": f"test_session_{int(time.time())}"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("success"):
                    response_text = data.get("response", "")
                    
                    # Check if response is relevant to account progress
                    account_keywords = [
                        "account", "performance", "trading", "profit", "loss", 
                        "equity", "balance", "progress", "results"
                    ]
                    
                    # Check if response contains generic market analysis (bad)
                    market_keywords = [
                        "gold", "eurusd", "market analysis", "technical analysis",
                        "rsi", "macd", "support", "resistance"
                    ]
                    
                    account_mentions = sum(1 for keyword in account_keywords if keyword.lower() in response_text.lower())
                    market_mentions = sum(1 for keyword in market_keywords if keyword.lower() in response_text.lower())
                    
                    print(f"✅ Status: SUCCESS")
                    print(f"📊 Account mentions: {account_mentions}")
                    print(f"📈 Market mentions: {market_mentions}")
                    
                    # Show first 200 characters of response
                    preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
                    print(f"💬 Response preview: {preview}")
                    
                    # Determine if response is appropriate
                    if account_mentions >= market_mentions:
                        print("🎯 RESULT: ✅ Response focused on account/performance (GOOD)")
                    else:
                        print("🎯 RESULT: ❌ Response focused on market analysis (BAD)")
                    
                else:
                    print(f"❌ API Error: {data.get('error', 'Unknown error')}")
                    
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print("⏰ Request timed out (chat bot may be slow)")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Brief pause between tests
        if i < len(test_questions):
            time.sleep(2)
    
    print(f"\n🏁 Test Summary:")
    print("If responses are focused on account/performance instead of market analysis,")
    print("then the chat bot fix was successful!")
    
    print(f"\n🌐 You can also test manually at:")
    print(f"http://localhost:5002/dashboard/forex_advisor_chat.html")

if __name__ == "__main__":
    test_chat_bot_fix() 