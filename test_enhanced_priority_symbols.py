#!/usr/bin/env python3
"""
Test Enhanced Priority Symbol Analysis
Focuses on: Gold, EURUSD, USDJPY, USDCAD, US100, US30, US500, BTC, GBPUSD, GBPJPY
"""

import asyncio
import requests
import json
from datetime import datetime

def test_priority_symbol_analysis():
    """Test enhanced analysis for priority symbols"""
    print("🎯 Testing Enhanced Priority Symbol Analysis")
    print("=" * 80)
    
    # User's priority symbols with their expected Yahoo Finance mappings
    priority_symbols = {
        'Gold': 'GC=F',
        'EURUSD': 'EURUSD=X', 
        'USDJPY': 'USDJPY=X',
        'USDCAD': 'USDCAD=X',
        'US100': '^IXIC',
        'US30': '^DJI', 
        'US500': '^GSPC',
        'Bitcoin': 'BTC-USD',
        'GBPUSD': 'GBPUSD=X',
        'GBPJPY': 'GBPJPY=X'
    }
    
    print("🔍 Priority Symbols Configuration:")
    for name, symbol in priority_symbols.items():
        print(f"   • {name:12} → {symbol}")
    
    # Test live data integration
    print(f"\n📊 Testing Live Data Integration...")
    try:
        from get_live_market_data import get_live_market_data, get_market_analysis
        
        sample_symbols = ['gold', 'eurusd', 'btc']
        for symbol in sample_symbols:
            print(f"\n🧪 Testing {symbol.upper()}:")
            data = get_live_market_data(symbol)
            
            if not data.get('error'):
                analysis = get_market_analysis(symbol, data)
                print(f"   ✅ Price: ${data['price']:.2f if symbol == 'gold' else data['price']:.5f}")
                print(f"   ✅ Change: {data['change_percent']:+.2f}%")
                print(f"   ✅ Analysis: {len(analysis)} chars of detailed analysis")
                print(f"   📝 Preview: {analysis[:100]}...")
            else:
                print(f"   ❌ Error: {data['error']}")
                
    except Exception as e:
        print(f"   ❌ Live data error: {e}")
    
    # Test enhanced market intelligence
    print(f"\n🧠 Testing Enhanced Market Intelligence...")
    try:
        from qnti_enhanced_market_intelligence import QNTIEnhancedMarketIntelligence
        
        intelligence = QNTIEnhancedMarketIntelligence()
        print(f"   ✅ Intelligence engine initialized")
        print(f"   📊 Forex pairs: {intelligence.forex_pairs}")
        print(f"   📊 Commodities: {intelligence.commodities}")
        print(f"   📊 Indices: {intelligence.indices}")
        print(f"   📊 Crypto: {intelligence.crypto}")
        
        # Test insights generation
        insights = intelligence.get_insights(limit=10)
        if insights:
            print(f"   ✅ Generated {len(insights)} insights")
            
            # Check for priority symbol insights
            priority_insights = [i for i in insights if any(sym in str(i.get('symbol', '')) for sym in priority_symbols.values())]
            print(f"   🎯 Priority symbol insights: {len(priority_insights)}")
            
            for insight in priority_insights[:3]:
                symbol = insight.get('symbol', 'Unknown')
                title = insight.get('title', 'No title')
                print(f"      • {symbol}: {title}")
        else:
            print(f"   ⚠️  No insights generated yet")
            
    except Exception as e:
        print(f"   ❌ Intelligence error: {e}")
    
    # Test Forex Advisor with priority symbols
    print(f"\n💼 Testing Forex Advisor with Priority Symbols...")
    
    advisor_url = "http://localhost:5002/advisor/chat"
    test_questions = [
        "What's the current analysis for Gold with specific price levels?",
        "Give me detailed EURUSD analysis with targets and stops",
        "What's happening with US30 and US100 indices today?",
        "Bitcoin price action analysis with support and resistance",
        "USDJPY and GBPJPY detailed technical analysis"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🧪 Test {i}: {question}")
        
        try:
            response = requests.post(advisor_url, 
                json={
                    "message": question,
                    "session_id": f"priority_test_{i}",
                    "user_id": "priority_test"
                },
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("success"):
                    advisor_response = data.get("response", "")
                    
                    # Check for specific analysis elements
                    has_prices = any(term in advisor_response for term in ["$", "price", "current"])
                    has_levels = any(term in advisor_response for term in ["support", "resistance", "target", "stop"])
                    has_data = any(term in advisor_response for term in ["Live", "Real-time", "2025-"])
                    has_specifics = any(term in advisor_response for term in ["%", "momentum", "technical"])
                    
                    print(f"   ✅ Response: {len(advisor_response)} chars")
                    print(f"   📊 Has prices: {'✅' if has_prices else '❌'}")
                    print(f"   📈 Has levels: {'✅' if has_levels else '❌'}")
                    print(f"   🔄 Has live data: {'✅' if has_data else '❌'}")
                    print(f"   🎯 Has specifics: {'✅' if has_specifics else '❌'}")
                    
                    # Show preview
                    print(f"   📝 Preview: {advisor_response[:150]}...")
                    
                else:
                    print(f"   ❌ API Error: {data.get('error', 'Unknown')}")
                    
            else:
                print(f"   ❌ HTTP {response.status_code}: {response.text[:100]}")
                
        except Exception as e:
            print(f"   ❌ Request error: {e}")
        
        # Small delay between requests
        if i < len(test_questions):
            import time
            time.sleep(1)
    
    # Test Market Intelligence endpoint
    print(f"\n🧠 Testing Market Intelligence API...")
    
    try:
        intelligence_url = "http://localhost:5002/api/market-intelligence/insights"
        response = requests.get(intelligence_url, timeout=10)
        
        if response.status_code == 200:
            insights = response.json()
            print(f"   ✅ Retrieved {len(insights)} insights")
            
            # Check for priority symbols
            priority_insights = []
            for insight in insights:
                insight_text = str(insight.get('title', '')) + str(insight.get('description', ''))
                if any(sym.replace('=X', '').replace('^', '').replace('-USD', '') in insight_text 
                       for sym in priority_symbols.values()):
                    priority_insights.append(insight)
            
            print(f"   🎯 Priority symbol insights: {len(priority_insights)}")
            
            for insight in priority_insights[:3]:
                title = insight.get('title', 'No title')
                priority = insight.get('priority', 'unknown')
                print(f"      • {priority.upper()}: {title}")
                
        else:
            print(f"   ❌ HTTP {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Intelligence API error: {e}")
    
    print(f"\n{'='*80}")
    print("🎯 Enhanced Priority Symbol Analysis Test Complete!")
    print("\n📊 Expected Enhancements:")
    print("✅ Focus on user's 10 priority symbols")
    print("✅ Detailed price levels with specific targets/stops") 
    print("✅ Real-time Yahoo Finance data integration")
    print("✅ Symbol-specific analysis (Gold, Forex, Indices, Crypto)")
    print("✅ Actionable insights with concrete price levels")
    print("✅ Professional risk management guidance")

if __name__ == "__main__":
    test_priority_symbol_analysis() 