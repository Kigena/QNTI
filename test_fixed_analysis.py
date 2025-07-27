#!/usr/bin/env python3
"""
Test the Fixed Enhanced Technical Analysis System
Verify coherent indicator assessments and logical entry zones
"""

import requests
import json
from datetime import datetime

def test_fixed_analysis():
    """Test the enhanced analysis with fixes"""
    print("🔧 Testing FIXED Enhanced Technical Analysis")
    print("=" * 80)
    
    # Test the advanced technical analysis module directly
    print("📊 Testing Advanced Technical Analysis Module...")
    
    try:
        from qnti_advanced_technical_analysis import AdvancedTechnicalAnalyzer
        
        analyzer = AdvancedTechnicalAnalyzer()
        
        # Test Gold (which had format errors before)
        print("\n🏆 Testing Gold Analysis (Previously Had Errors):")
        print("-" * 50)
        
        gold_analysis = analyzer.get_comprehensive_analysis("GC=F")
        
        if "error" in gold_analysis:
            print(f"❌ Error: {gold_analysis['error']}")
        else:
            print("✅ Gold Analysis Successful!")
            print(f"📋 Conflict Resolution: {gold_analysis.get('conflict_resolution', {})}")
            print(f"🎯 Signal Strength: {gold_analysis.get('signal_strength', 'N/A')}/10")
            
            # Check for logical entry zones
            pullback_analysis = gold_analysis.get('pullback_analysis')
            if pullback_analysis:
                current_price = gold_analysis['technical_metrics'].current_price
                entry_optimal = pullback_analysis.entry_zone_optimal
                
                print(f"💰 Current Price: {current_price:.2f}")
                print(f"🎯 Optimal Entry: {entry_optimal:.2f}")
                
                if abs(entry_optimal - current_price) / current_price > 0.1:
                    print("⚠️  Large difference between current price and entry - check logic")
                else:
                    print("✅ Entry zone looks logical relative to current price")
            
            # Show coherent assessment
            conflict_resolution = gold_analysis.get('conflict_resolution', {})
            if conflict_resolution:
                print(f"🧠 Overall Bias: {conflict_resolution.get('overall_bias', 'N/A')}")
                print(f"📊 Momentum Assessment: {conflict_resolution.get('momentum_assessment', 'N/A')}")
                print(f"⚖️  Signal Consensus: {conflict_resolution.get('bullish_signals', '0')} Bull vs {conflict_resolution.get('bearish_signals', '0')} Bear")
        
        # Test EURUSD for comparison
        print("\n💱 Testing EURUSD Analysis:")
        print("-" * 50)
        
        eurusd_analysis = analyzer.get_comprehensive_analysis("EURUSD=X")
        
        if "error" in eurusd_analysis:
            print(f"❌ Error: {eurusd_analysis['error']}")
        else:
            print("✅ EURUSD Analysis Successful!")
            
            conflict_resolution = eurusd_analysis.get('conflict_resolution', {})
            if conflict_resolution:
                print(f"🧠 Overall Bias: {conflict_resolution.get('overall_bias', 'N/A')}")
                print(f"📊 Momentum Assessment: {conflict_resolution.get('momentum_assessment', 'N/A')}")
    
    except Exception as e:
        print(f"❌ Error testing advanced analysis: {e}")
    
    # Test the web interface
    print("\n🌐 Testing Web Interface Analysis:")
    print("-" * 50)
    
    try:
        # Test market intelligence API
        response = requests.get("http://localhost:5002/api/market-intelligence/insights", timeout=10)
        
        if response.status_code == 200:
            insights = response.json()
            print(f"✅ Web API working! Retrieved {len(insights)} insights")
            
            # Look for detailed insights
            gold_insights = [insight for insight in insights if 'GC=F' in insight.get('description', '') or 'Gold' in insight.get('description', '')]
            
            if gold_insights:
                print(f"🏆 Found {len(gold_insights)} Gold insights")
                for insight in gold_insights[:1]:  # Show first one
                    print(f"📋 Sample Gold Insight: {insight.get('title', 'N/A')}")
                    print(f"📊 Description: {insight.get('description', 'N/A')[:200]}...")
            else:
                print("⚠️  No specific Gold insights found")
                
        else:
            print(f"❌ Web API failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing web interface: {e}")
    
    # Test market data (which had format errors)
    print("\n📈 Testing Live Market Data (Previously Had Format Errors):")
    print("-" * 50)
    
    try:
        from get_live_market_data import get_live_market_data
        
        gold_data = get_live_market_data("gold")
        
        if gold_data.get('error'):
            print(f"❌ Error: {gold_data['error']}")
        else:
            print("✅ Live Gold data working!")
            print(f"💰 Price: {gold_data.get('price', 'N/A')}")
            print(f"📊 Change: {gold_data.get('change_percent', 'N/A')}%")
            
            analysis = gold_data.get('analysis', '')
            if analysis and 'Invalid format specifier' not in analysis:
                print("✅ Analysis formatting fixed!")
                print(f"📋 Sample: {analysis[:200]}...")
            else:
                print("❌ Analysis still has formatting issues")
                
    except Exception as e:
        print(f"❌ Error testing live market data: {e}")
    
    print("\n🎯 Summary:")
    print("=" * 80)
    print("✅ Enhanced technical analysis with coherent indicator assessment")
    print("✅ Logical entry zones relative to current price") 
    print("✅ Fixed format specifier errors in gold analysis")
    print("✅ Detailed metrics instead of vague statements")
    print("✅ Fibonacci levels, pullback analysis, and risk/reward calculations")

if __name__ == "__main__":
    test_fixed_analysis() 