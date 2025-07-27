#!/usr/bin/env python3

from get_live_market_data import get_market_analysis
import traceback

print("🔍 Testing Gold Analysis Edge Cases")
print("=" * 50)

# Test various data scenarios that might cause the error
test_cases = [
    # Normal case
    {
        'symbol': 'gold',
        'price': 3442.3,
        'change_percent': 1.054,
        'high_24h': 3447.5,
        'low_24h': 3394.9,
        'timestamp': '2025-07-22 21:01:13'
    },
    # None values
    {
        'symbol': 'gold',
        'price': 3442.3,
        'change_percent': None,
        'high_24h': None,
        'low_24h': None,
        'timestamp': '2025-07-22 21:01:13'
    },
    # String price
    {
        'symbol': 'gold',
        'price': 'N/A',
        'change_percent': 1.054,
        'high_24h': 3447.5,
        'low_24h': 3394.9,
        'timestamp': '2025-07-22 21:01:13'
    },
    # Missing fields
    {
        'symbol': 'gold',
        'price': 3442.3,
        'change_percent': 1.054
    },
    # Zero price
    {
        'symbol': 'gold',
        'price': 0,
        'change_percent': 1.054,
        'high_24h': 3447.5,
        'low_24h': 3394.9,
        'timestamp': '2025-07-22 21:01:13'
    }
]

for i, test_data in enumerate(test_cases, 1):
    print(f"\n🧪 Test Case {i}: {list(test_data.keys())}")
    
    try:
        analysis = get_market_analysis('gold', test_data)
        print(f"✅ Success: {len(analysis)} chars")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔍 Full traceback:")
        traceback.print_exc()
        print(f"🔍 Test data: {test_data}")
        break 