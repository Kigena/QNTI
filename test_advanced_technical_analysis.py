#!/usr/bin/env python3
"""
Test Advanced Technical Analysis System
Verify detailed metrics, Fibonacci analysis, and entry zones are working
"""

import requests
import json
from datetime import datetime

def test_advanced_technical_analysis():
    """Test the advanced technical analysis system"""
    print("🔬 Testing Advanced Technical Analysis System")
    print("=" * 80)
    
    # Priority symbols to test
    test_symbols = {
        'EURUSD': 'EUR/USD',
        'USDJPY': 'USD/JPY', 
        'Gold': 'Gold',
        'US30': 'Dow Jones',
        'Bitcoin': 'Bitcoin'
    }
    
    # Test the enhanced market intelligence
    print("📊 Testing Enhanced Market Intelligence API...")
    
    try:
        response = requests.get("http://localhost:5002/api/market-intelligence/insights", timeout=30)
        
        if response.status_code == 200:
            insights = response.json()
            print(f"✅ Retrieved {len(insights)} insights")
            
            # Check for advanced technical analysis
            advanced_insights = [i for i in insights if i.get('source') == 'advanced_technical_analysis']
            print(f"🎯 Advanced technical insights: {len(advanced_insights)}")
            
            # Analyze the quality of insights
            for i, insight in enumerate(advanced_insights[:3]):
                print(f"\n🔍 Insight {i+1}:")
                title = insight.get('title', 'No title')
                description = insight.get('description', '')
                priority = insight.get('priority', 'unknown')
                symbol = insight.get('symbol', 'unknown')
                
                print(f"   Symbol: {symbol}")
                print(f"   Title: {title}")
                print(f"   Priority: {priority.upper()}")
                
                # Check for detailed metrics
                has_rsi = 'RSI' in description
                has_macd = 'MACD' in description
                has_fibonacci = 'Fibonacci' in description
                has_entry_zones = 'Entry Zones' in description
                has_targets = 'Target' in description
                has_support_resistance = 'Support:' in description or 'Resistance:' in description
                has_pullback_analysis = 'Pullback Probability' in description
                has_specific_numbers = any(c.isdigit() for c in description)
                
                print(f"   📈 Has RSI: {'✅' if has_rsi else '❌'}")
                print(f"   📊 Has MACD: {'✅' if has_macd else '❌'}")
                print(f"   📐 Has Fibonacci: {'✅' if has_fibonacci else '❌'}")
                print(f"   🎯 Has Entry Zones: {'✅' if has_entry_zones else '❌'}")
                print(f"   🏹 Has Targets: {'✅' if has_targets else '❌'}")
                print(f"   📏 Has Support/Resistance: {'✅' if has_support_resistance else '❌'}")
                print(f"   📉 Has Pullback Analysis: {'✅' if has_pullback_analysis else '❌'}")
                print(f"   🔢 Has Specific Numbers: {'✅' if has_specific_numbers else '❌'}")
                
                # Calculate quality score
                quality_score = sum([
                    has_rsi, has_macd, has_fibonacci, has_entry_zones,
                    has_targets, has_support_resistance, has_pullback_analysis, has_specific_numbers
                ])
                
                print(f"   🏆 Quality Score: {quality_score}/8")
                
                if quality_score >= 6:
                    print("   🎉 HIGH QUALITY - Detailed technical analysis")
                elif quality_score >= 4:
                    print("   👍 GOOD QUALITY - Adequate technical analysis")
                else:
                    print("   ⚠️  LOW QUALITY - Missing detailed metrics")
                
                # Show a preview of the description
                preview = description[:200] + "..." if len(description) > 200 else description
                print(f"   📝 Preview: {preview}")
        else:
            print(f"❌ HTTP {response.status_code}: {response.text[:100]}")
            
    except Exception as e:
        print(f"❌ Market Intelligence API error: {e}")
    
    # Test the Forex Advisor with detailed analysis requests
    print(f"\n💼 Testing Forex Advisor with Advanced Analysis Requests...")
    
    detailed_questions = [
        "Give me detailed technical analysis for EURUSD with RSI, MACD, Fibonacci levels, entry zones, and specific price targets",
        "What are the exact Fibonacci retracement levels for Gold and what's the pullback probability?",
        "Provide specific entry zones and risk-reward ratios for US30 trend continuation",
        "What do the indicators say about Bitcoin? Give me RSI, Stochastic, support/resistance levels",
        "Analyze USDJPY with detailed moving average slopes, Bollinger Band position, and volume analysis"
    ]
    
    for i, question in enumerate(detailed_questions, 1):
        print(f"\n🧪 Test {i}: Advanced Analysis Request")
        print(f"Question: {question[:80]}...")
        
        try:
            response = requests.post("http://localhost:5002/advisor/chat",
                json={
                    "message": question,
                    "session_id": f"advanced_test_{i}",
                    "user_id": "advanced_test"
                },
                timeout=20
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("success"):
                    advisor_response = data.get("response", "")
                    
                    # Check for advanced analysis elements
                    has_specific_prices = any(char.isdigit() and '.' in advisor_response[max(0, i-10):i+10] 
                                            for i, char in enumerate(advisor_response))
                    has_indicators = any(indicator in advisor_response.upper() 
                                       for indicator in ['RSI', 'MACD', 'STOCHASTIC', 'WILLIAMS'])
                    has_fibonacci = any(fib in advisor_response 
                                      for fib in ['Fibonacci', 'fib', '23.6', '38.2', '61.8'])
                    has_entry_zones = any(entry in advisor_response.lower() 
                                        for entry in ['entry', 'zone', 'target', 'stop'])
                    has_percentages = '%' in advisor_response
                    has_support_resistance = any(level in advisor_response.lower() 
                                               for level in ['support', 'resistance', 'level'])
                    has_trend_analysis = any(trend in advisor_response.lower() 
                                           for trend in ['trend', 'bullish', 'bearish', 'momentum'])
                    has_real_time_data = any(date_indicator in advisor_response 
                                           for date_indicator in ['2025', 'Live', 'Real-time', 'Current'])
                    
                    print(f"   Response Length: {len(advisor_response)} chars")
                    print(f"   📊 Specific Prices: {'✅' if has_specific_prices else '❌'}")
                    print(f"   🎯 Technical Indicators: {'✅' if has_indicators else '❌'}")
                    print(f"   📐 Fibonacci Analysis: {'✅' if has_fibonacci else '❌'}")
                    print(f"   🎪 Entry Zones/Targets: {'✅' if has_entry_zones else '❌'}")
                    print(f"   📈 Percentages: {'✅' if has_percentages else '❌'}")
                    print(f"   📏 Support/Resistance: {'✅' if has_support_resistance else '❌'}")
                    print(f"   📊 Trend Analysis: {'✅' if has_trend_analysis else '❌'}")
                    print(f"   🔄 Real-time Data: {'✅' if has_real_time_data else '❌'}")
                    
                    # Calculate analysis quality
                    analysis_quality = sum([
                        has_specific_prices, has_indicators, has_fibonacci, has_entry_zones,
                        has_percentages, has_support_resistance, has_trend_analysis, has_real_time_data
                    ])
                    
                    print(f"   🏆 Analysis Quality: {analysis_quality}/8")
                    
                    if analysis_quality >= 6:
                        print("   🎉 EXCELLENT - Detailed technical analysis with metrics")
                    elif analysis_quality >= 4:
                        print("   👍 GOOD - Adequate technical detail")
                    else:
                        print("   ⚠️  POOR - Generic response without detail")
                    
                    # Show response preview
                    preview = advisor_response[:300] + "..." if len(advisor_response) > 300 else advisor_response
                    print(f"   📝 Response Preview: {preview}")
                    
                else:
                    print(f"   ❌ API Error: {data.get('error', 'Unknown')}")
                    
            else:
                print(f"   ❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Request error: {e}")
    
    # Test the live market data with fixed formatting
    print(f"\n📊 Testing Live Market Data (Fixed Formatting)...")
    
    try:
        from get_live_market_data import get_live_market_data, get_market_analysis
        
        test_symbols = ['gold', 'eurusd', 'btc']
        
        for symbol in test_symbols:
            print(f"\n🧪 Testing {symbol.upper()}:")
            
            try:
                data = get_live_market_data(symbol)
                
                if not data.get('error'):
                    analysis = get_market_analysis(symbol, data)
                    
                    print(f"   ✅ Price: {data['price']}")
                    print(f"   ✅ Change: {data['change_percent']:+.2f}%")
                    print(f"   ✅ Analysis Generated: {len(analysis)} chars")
                    
                    # Check if the analysis has technical levels
                    has_support = 'Support' in analysis
                    has_resistance = 'Resistance' in analysis
                    has_trend = any(trend in analysis for trend in ['Bullish', 'Bearish', 'Neutral'])
                    has_price_levels = any(char.isdigit() for char in analysis)
                    
                    print(f"   📈 Has Support: {'✅' if has_support else '❌'}")
                    print(f"   📈 Has Resistance: {'✅' if has_resistance else '❌'}")
                    print(f"   📊 Has Trend: {'✅' if has_trend else '❌'}")
                    print(f"   🔢 Has Price Levels: {'✅' if has_price_levels else '❌'}")
                    
                else:
                    print(f"   ❌ Error: {data['error']}")
                    
            except Exception as e:
                print(f"   ❌ Error testing {symbol}: {e}")
                
    except Exception as e:
        print(f"❌ Live data module error: {e}")
    
    print(f"\n{'='*80}")
    print("🔬 Advanced Technical Analysis Test Complete!")
    print("\n📊 Expected Improvements:")
    print("✅ RSI, MACD, Stochastic, Williams %R readings with exact values")
    print("✅ Fibonacci retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)")
    print("✅ Specific entry zones for trend continuation")
    print("✅ Pullback probability calculations")
    print("✅ Risk/Reward ratios with exact targets")
    print("✅ Support/Resistance levels with precise prices")
    print("✅ Moving average slopes and alignment analysis")
    print("✅ Bollinger Band position and squeeze detection")
    print("✅ Volume analysis and volatility percentiles")
    print("✅ Real-time price levels instead of generic statements")

if __name__ == "__main__":
    test_advanced_technical_analysis() 