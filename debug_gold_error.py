#!/usr/bin/env python3

from get_live_market_data import get_live_market_data, get_market_analysis
import traceback

print("🔍 Debugging Gold Analysis Error")
print("=" * 40)

try:
    print("1. Getting gold data...")
    data = get_live_market_data('gold')
    print(f"✅ Data received: {data}")
    
    print("\n2. Generating analysis...")
    analysis = get_market_analysis('gold', data)
    print(f"✅ Analysis generated successfully!")
    print(f"Analysis preview: {analysis[:200]}...")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n🔍 Full traceback:")
    traceback.print_exc() 