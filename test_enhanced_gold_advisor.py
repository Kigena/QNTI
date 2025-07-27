#!/usr/bin/env python3
"""
Test Enhanced Gold Advisor Functionality
"""

import requests
import json
import time

def test_gold_advisor():
    """Test the enhanced gold advisor with improved data integration"""
    print("🧪 Testing Enhanced Gold Advisor")
    print("=" * 60)
    
    # Test the advisor endpoint
    url = "http://localhost:5002/advisor/chat"
    
    gold_questions = [
        "What's the current outlook for gold?",
        "How should I trade gold today?",
        "What are the key drivers for gold prices?",
        "Should I buy gold now?",
        "Tell me about gold market analysis"
    ]
    
    for i, question in enumerate(gold_questions, 1):
        print(f"\n🏆 Test {i}: {question}")
        print("-" * 50)
        
        try:
            response = requests.post(url, 
                json={
                    "message": question,
                    "session_id": f"test_gold_{i}",
                    "user_id": "test_user"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("success"):
                    advisor_response = data.get("response", "")
                    
                    # Check for gold-specific content
                    gold_indicators = [
                        "gold", "precious", "XAU", "$3,", "$2,", 
                        "bullish", "bearish", "support", "resistance",
                        "Federal Reserve", "inflation", "safe haven"
                    ]
                    
                    found_indicators = [ind for ind in gold_indicators if ind.lower() in advisor_response.lower()]
                    
                    print(f"✅ Response received ({len(advisor_response)} chars)")
                    print(f"📊 Gold indicators found: {len(found_indicators)}")
                    print(f"🔍 Indicators: {', '.join(found_indicators[:5])}")
                    
                    # Check for live data
                    if any(term in advisor_response for term in ["Live", "Real-Time", "$3,", "2025-"]):
                        print("🔄 ✅ Contains live data")
                    else:
                        print("🔄 ❌ Missing live data")
                    
                    # Check for research insights
                    if any(term in advisor_response for term in ["Research", "Federal Reserve", "Central Bank"]):
                        print("📚 ✅ Contains research insights")
                    else:
                        print("📚 ❌ Missing research insights")
                    
                    # Show first 200 chars of response
                    print(f"📝 Preview: {advisor_response[:200]}...")
                    
                else:
                    print(f"❌ API Error: {data.get('error', 'Unknown error')}")
                    
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
        
        # Pause between requests
        if i < len(gold_questions):
            time.sleep(2)
    
    print(f"\n{'='*60}")
    print("🏆 Gold Advisor Testing Complete!")
    print("\n📊 Expected Enhancements:")
    print("✅ Live Yahoo Finance gold prices ($3,400+ range)")
    print("✅ Enhanced research with gold-focused sources")
    print("✅ Priority scoring for gold-related content")
    print("✅ Professional risk management guidance")
    print("✅ Technical analysis with support/resistance")

if __name__ == "__main__":
    test_gold_advisor() 