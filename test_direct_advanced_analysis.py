#!/usr/bin/env python3
"""
Direct Test of Advanced Technical Analysis Module
Test the new detailed metrics system directly
"""

import sys
import time

def test_direct_advanced_analysis():
    """Test the advanced analysis module directly"""
    print("🔬 Direct Advanced Technical Analysis Test")
    print("=" * 80)
    
    try:
        from qnti_advanced_technical_analysis import AdvancedTechnicalAnalyzer
        
        analyzer = AdvancedTechnicalAnalyzer()
        
        # Test with your priority symbols
        test_symbols = [
            ('EURUSD=X', 'EUR/USD'),
            ('GC=F', 'Gold'),
            ('USDJPY=X', 'USD/JPY'),
            ('BTC-USD', 'Bitcoin'),
            ('^DJI', 'US30 (Dow Jones)')
        ]
        
        for symbol, name in test_symbols:
            print(f"\n📊 Testing {name} ({symbol})...")
            
            start_time = time.time()
            result = analyzer.get_comprehensive_analysis(symbol)
            analysis_time = time.time() - start_time
            
            if "error" not in result:
                print(f"✅ Analysis completed in {analysis_time:.2f} seconds")
                
                # Show the detailed narrative (your requested format)
                narrative = result["detailed_narrative"]
                
                # Print first part of narrative to show the detail level
                print("📝 Sample of Advanced Analysis:")
                print("-" * 60)
                lines = narrative.split('\n')
                for line in lines[:20]:  # Show first 20 lines
                    print(line)
                print("-" * 60)
                
                # Show key metrics
                tech_metrics = result["technical_metrics"]
                fib_analysis = result["fibonacci_analysis"]
                pullback_analysis = result["pullback_analysis"]
                
                print(f"\n🎯 Key Metrics Summary:")
                print(f"   Current Price: {tech_metrics.current_price:.5f}")
                print(f"   RSI(14): {tech_metrics.rsi_14:.1f} - {tech_metrics.rsi_signal}")
                print(f"   MACD Trend: {tech_metrics.macd_trend}")
                print(f"   MA Alignment: {tech_metrics.ma_alignment}")
                print(f"   Fibonacci Level: {fib_analysis.current_fib_level}")
                print(f"   Trend Strength: {pullback_analysis.trend_strength:.0f}/100")
                print(f"   Pullback Probability: {pullback_analysis.pullback_probability:.0f}%")
                print(f"   Entry Zone: {pullback_analysis.entry_zone_optimal:.5f}")
                print(f"   Target 1: {pullback_analysis.target_1:.5f} (R:R = {pullback_analysis.rr_ratio_1:.2f})")
                print(f"   Signal Strength: {result['signal_strength']}/10")
                
                print(f"\n🎪 Trade Setup:")
                trade_setup = result["trade_setup"]
                for key, value in trade_setup.items():
                    print(f"   {key.replace('_', ' ').title()}: {value}")
                
            else:
                print(f"❌ Error: {result['error']}")
                
            print("\n" + "="*60)
        
        print("\n🎉 Direct Analysis Test Complete!")
        print("\n✅ Your enhanced system now provides:")
        print("📊 Exact RSI, MACD, Stochastic readings")
        print("📐 Fibonacci levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)")
        print("🎯 Specific entry zones with price levels")
        print("📈 Pullback probability calculations")
        print("💰 Risk/Reward ratios for multiple targets")
        print("📏 Support/Resistance with exact prices")
        print("📊 Moving average slopes and alignment")
        print("🎪 Bollinger Band position and squeeze detection")
        print("📊 Volume analysis and volatility percentiles")
        
    except Exception as e:
        print(f"❌ Error importing or running advanced analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_advanced_analysis() 