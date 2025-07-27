#!/usr/bin/env python3
"""
Debug script for Vision Trading parsing logic
"""

import re
from qnti_vision_trading import QNTIVisionTrader

def test_parsing_debug():
    """Test the parsing logic step by step"""
    
    # Sample analysis in your exact format
    sample_analysis = """
**Market Structure Analysis (HTF):**
- **Trend:** Bearish on Daily/H4 timeframes
- **Market Phase:** Range-bound correction within downtrend  
- **Structural Evidence:** Clear BOS to the downside confirmed

**SMC Elements:**
- **CHoCH:** Confirmed on H1 with price breaking previous structure
- **BOS:** Strong bearish break validated on H4
- **Order Blocks:** Bearish OB at 1.0950-1.0970 acting as resistance
- **FVG:** Fair Value Gap identified at 1.0920-1.0935
- **Liquidity Zones:** Sell-side liquidity targeted below 1.0880

**Directional Bias & Confluence:**
- **Primary Bias:** Bearish (Conviction: 80%)
- **Secondary Consideration:** Range-bound scenario (20%)

**Trading Plan:**
- **Entry Zone:** 1.0950-1.0970 (resistance area)
- **Stop Loss:** 1.0985 (above resistance structure)
- **Take Profit 1:** 1.0920 (FVG fill)
- **Take Profit 2:** 1.0880 (liquidity target)

**Risk Factors & Invalidation:**
- **Invalidation:** Above 1.0985 (structure break)
- **Risk Level:** Medium (clear structure but range-bound)
"""

    print("🔍 VISION TRADING PARSING DEBUG")
    print("=" * 50)
    
    # Create trader instance (with mock dependencies)
    try:
        from qnti_main import QNTI_Main
        main_system = QNTI_Main()
        trader = QNTIVisionTrader(main_system.trade_manager, main_system.mt5_bridge)
    except:
        print("❌ Cannot create trader instance - need running QNTI system")
        print("Please run this in the QNTI environment")
        return
    
    # Test each parsing method individually
    print("\n1️⃣ Testing Directional Bias Extraction:")
    bias = trader._extract_directional_bias(sample_analysis)
    print(f"   Result: {bias}")
    
    print("\n2️⃣ Testing Trading Plan Extraction:")
    plan = trader._extract_trading_plan(sample_analysis)
    print(f"   Result: {plan}")
    
    print("\n3️⃣ Testing Risk Factors Extraction:")
    risks = trader._extract_risk_factors(sample_analysis)
    print(f"   Result: {risks}")
    
    print("\n4️⃣ Testing Confidence Extraction:")
    confidence = trader._extract_confidence(sample_analysis)
    print(f"   Result: {confidence}")
    
    # Test manual regex patterns to see what's working
    print("\n🧪 Manual Pattern Testing:")
    
    # Entry zone patterns
    entry_patterns = [
        r"\*\*Entry Zone:\*\*\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",
        r"Entry Zone:\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",
        r"Entry:\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",
        r"entry.*?(\d+\.?\d*)\s*-\s*(\d+\.?\d*)"
    ]
    
    print("   Entry Zone Patterns:")
    for i, pattern in enumerate(entry_patterns):
        match = re.search(pattern, sample_analysis, re.IGNORECASE)
        print(f"     Pattern {i+1}: {'✅' if match else '❌'} {pattern}")
        if match:
            print(f"       Found: {match.groups()}")
    
    # Stop Loss patterns
    sl_patterns = [
        r"\*\*Stop Loss:\*\*\s*(\d+\.?\d*)",
        r"Stop Loss:\s*(\d+\.?\d*)",
        r"SL:\s*(\d+\.?\d*)",
        r"stop.*?(\d+\.?\d*)\s*\(",
        r"Stop:\s*(\d+\.?\d*)"
    ]
    
    print("   Stop Loss Patterns:")
    for i, pattern in enumerate(sl_patterns):
        match = re.search(pattern, sample_analysis, re.IGNORECASE)
        print(f"     Pattern {i+1}: {'✅' if match else '❌'} {pattern}")
        if match:
            print(f"       Found: {match.groups()}")
    
    # Full analysis parsing
    print("\n5️⃣ Full Analysis Parsing:")
    try:
        analysis = trader.parse_smc_analysis(sample_analysis, "EURUSD")
        print(f"   Parsed Analysis: {analysis}")
        
        print("\n6️⃣ Vision Trade Creation:")
        trade = trader.create_vision_trade_from_analysis(analysis)
        if trade:
            print(f"   ✅ Trade Created: {trade.analysis_id}")
            print(f"       Direction: {trade.direction}")
            print(f"       Entry: {trade.entry_zone_lower}-{trade.entry_zone_upper}")
            print(f"       Stop Loss: {trade.stop_loss}")
            print(f"       Take Profits: {trade.take_profit_1}, {trade.take_profit_2}")
        else:
            print("   ❌ Failed to create trade")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("Debug complete!")

if __name__ == "__main__":
    test_parsing_debug() 