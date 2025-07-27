#!/usr/bin/env python3
"""
Live Market Data Fetcher for QNTI Forex Advisor
Gets real-time data from Yahoo Finance
"""

import yfinance as yf
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_live_market_data(symbol: str) -> Dict[str, Any]:
    """Get live market data for a symbol"""
    try:
        # Map common symbols to Yahoo Finance format
        symbol_map = {
            'gold': 'GC=F',
            'xau': 'GC=F',
            'xauusd': 'GC=F',
            'eurusd': 'EURUSD=X',
            'eur/usd': 'EURUSD=X',
            'gbpusd': 'GBPUSD=X',
            'gbp/usd': 'GBPUSD=X',
            'usdjpy': 'USDJPY=X',
            'usd/jpy': 'USDJPY=X',
            'silver': 'SI=F',
            'xag': 'SI=F',
            'btc': 'BTC-USD',
            'bitcoin': 'BTC-USD',
            'eth': 'ETH-USD',
            'ethereum': 'ETH-USD'
        }
        
        # Get the proper Yahoo symbol
        yahoo_symbol = symbol_map.get(symbol.lower(), symbol.upper())
        
        # Get the ticker
        ticker = yf.Ticker(yahoo_symbol)
        
        # Get current data
        info = ticker.info
        hist = ticker.history(period="1d", interval="1m")
        
        if hist.empty:
            hist = ticker.history(period="5d")
        
        if hist.empty:
            return {
                'symbol': symbol,
                'price': 'N/A',
                'change': 0,
                'change_percent': 0,
                'error': 'No data available'
            }
        
        current_price = hist['Close'].iloc[-1]
        prev_close = info.get('previousClose', hist['Close'].iloc[0])
        
        if prev_close and prev_close > 0:
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
        else:
            change = 0
            change_percent = 0
        
        # Get additional data
        volume = hist['Volume'].iloc[-1] if 'Volume' in hist else 0
        high_24h = hist['High'].max()
        low_24h = hist['Low'].min()
        
        return {
            'symbol': symbol,
            'yahoo_symbol': yahoo_symbol,
            'price': round(current_price, 5 if 'USD' in yahoo_symbol else 2),
            'change': round(change, 5 if 'USD' in yahoo_symbol else 2),
            'change_percent': round(change_percent, 3),
            'volume': int(volume) if volume > 0 else None,
            'high_24h': round(high_24h, 5 if 'USD' in yahoo_symbol else 2),
            'low_24h': round(low_24h, 5 if 'USD' in yahoo_symbol else 2),
            'prev_close': round(prev_close, 5 if 'USD' in yahoo_symbol else 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'market_cap': info.get('marketCap'),
            'currency': info.get('currency', 'USD')
        }
        
    except Exception as e:
        logger.error(f"Error getting live data for {symbol}: {e}")
        return {
            'symbol': symbol,
            'price': 'N/A',
            'change': 0,
            'change_percent': 0,
            'error': str(e)
        }

def get_market_analysis(symbol: str, market_data: Dict[str, Any]) -> str:
    """Generate market analysis based on live data"""
    try:
        if market_data.get('error'):
            return f"Unable to get current market data for {symbol}: {market_data['error']}"
        
        price = market_data['price']
        change_percent = market_data.get('change_percent', 0)
        high_24h = market_data.get('high_24h', 'N/A')
        low_24h = market_data.get('low_24h', 'N/A')
        
        # Ensure change_percent is a number
        if change_percent is None:
            change_percent = 0
        
        # Determine trend
        if change_percent > 1:
            trend = "📈 Strong Bullish"
        elif change_percent > 0.1:
            trend = "🟢 Bullish"
        elif change_percent > -0.1:
            trend = "⚪ Neutral"
        elif change_percent > -1:
            trend = "🔴 Bearish"
        else:
            trend = "📉 Strong Bearish"
        
        # Calculate support/resistance levels
        if isinstance(price, (int, float)) and price > 0:
            if symbol.lower() in ['gold', 'xau', 'xauusd']:
                support_1 = price - (price * 0.015)  # 1.5% below
                resistance_1 = price + (price * 0.015)  # 1.5% above
                support_2 = price - (price * 0.03)    # 3% below
                resistance_2 = price + (price * 0.03)  # 3% above
            else:
                support_1 = price - (price * 0.005)   # 0.5% below
                resistance_1 = price + (price * 0.005) # 0.5% above
                support_2 = price - (price * 0.01)    # 1% below
                resistance_2 = price + (price * 0.01)  # 1% above
        else:
            support_1 = support_2 = resistance_1 = resistance_2 = 'N/A'
        
        # Safe formatting for price levels
        def safe_format(value, decimal_places=2):
            try:
                if isinstance(value, (int, float)) and not (value != value):  # Check for NaN
                    if decimal_places == 5:
                        return f"{value:.5f}"
                    elif decimal_places == 2:
                        return f"{value:.2f}"
                    else:
                        return f"{value:.{decimal_places}f}"
                else:
                    return str(value) if value is not None else 'N/A'
            except (ValueError, TypeError):
                return 'N/A'
        
        # Safe format change_percent 
        try:
            change_percent_str = f"{change_percent:+.2f}" if isinstance(change_percent, (int, float)) and not (change_percent != change_percent) else "+0.00"
        except (ValueError, TypeError):
            change_percent_str = "+0.00"
        
        # Pre-calculate decimal places to avoid f-string issues
        # Check the mapped symbol (yahoo_symbol) rather than original symbol
        decimal_places = 5 if any(x in str(symbol).upper() for x in ['USD=X', 'JPY=X', 'EUR', 'GBP']) else 2
        
        # Validate all values before formatting
        price_str = safe_format(price, 2)
        high_str = safe_format(high_24h, 2)
        low_str = safe_format(low_24h, 2)
        support_1_str = safe_format(support_1, decimal_places)
        resistance_1_str = safe_format(resistance_1, decimal_places)
        support_2_str = safe_format(support_2, decimal_places)
        resistance_2_str = safe_format(resistance_2, decimal_places)
        
        analysis = f"""**Live Market Analysis for {symbol.upper()}**

**Current Status**: {trend}
**Price**: {price_str} ({change_percent_str}%)
**24h Range**: {low_str} - {high_str}
**Last Updated**: {market_data.get('timestamp', 'Unknown')}

**Technical Levels**:
• **Immediate Support**: {support_1_str}
• **Immediate Resistance**: {resistance_1_str}
• **Extended Support**: {support_2_str}
• **Extended Resistance**: {resistance_2_str}"""

        return analysis
        
    except Exception as e:
        logger.error(f"Error generating analysis for {symbol}: {e}")
        return f"Error generating analysis for {symbol}: {str(e)}"

def test_live_data():
    """Test the live data functionality"""
    symbols = ['gold', 'eurusd', 'btc']
    
    print("🧪 Testing Live Market Data")
    print("=" * 50)
    
    for symbol in symbols:
        print(f"\n📊 Testing {symbol.upper()}:")
        data = get_live_market_data(symbol)
        print(f"Price: {data['price']}")
        print(f"Change: {data['change_percent']:+.2f}%")
        print(f"Status: {'✅ Success' if not data.get('error') else '❌ Error: ' + data['error']}")

if __name__ == "__main__":
    test_live_data() 