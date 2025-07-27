#!/usr/bin/env python3
"""
QNTI System Status Checker
Diagnoses system integration and URL accessibility
"""

import requests
import time
import json
from datetime import datetime

def check_url(url, description):
    """Check if a URL is accessible and return status"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return f"✅ {description}: {response.status_code} OK"
        else:
            return f"⚠️  {description}: {response.status_code} {response.reason}"
    except requests.exceptions.RequestException as e:
        return f"❌ {description}: {str(e)}"

def check_api_endpoint(url, description):
    """Check API endpoint and try to parse JSON response"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            try:
                data = response.json()
                if isinstance(data, dict):
                    success = data.get('success', False)
                    if success:
                        if 'insights' in data:
                            count = len(data['insights']) if data['insights'] else 0
                            return f"✅ {description}: {count} insights available"
                        else:
                            return f"✅ {description}: API working"
                    else:
                        return f"⚠️  {description}: API returned success=false"
                else:
                    return f"✅ {description}: Non-standard response format"
            except json.JSONDecodeError:
                return f"⚠️  {description}: Invalid JSON response"
        else:
            return f"❌ {description}: {response.status_code} {response.reason}"
    except requests.exceptions.RequestException as e:
        return f"❌ {description}: {str(e)}"

def main():
    """Main diagnostic function"""
    print("🔍 QNTI SYSTEM STATUS CHECKER")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Base URLs to check
    base_url = "http://localhost:5003"
    
    # Dashboard Pages
    print("📊 DASHBOARD PAGES:")
    dashboard_pages = [
        ("", "Root Dashboard"),
        ("/dashboard/main_dashboard.html", "Main Dashboard"),
        ("/dashboard/market_intelligence_board.html", "Market Intelligence Board"),
        ("/dashboard/forex_advisor_chat.html", "Forex Advisor Chat"),
        ("/dashboard/trading_center.html", "Trading Center"),
        ("/dashboard/ea_management.html", "EA Management"),
        ("/dashboard/strategy_tester.html", "Strategy Tester"),
        ("/dashboard/analytics_reports.html", "Analytics Reports"),
    ]
    
    for path, name in dashboard_pages:
        url = f"{base_url}{path}"
        print(f"  {check_url(url, name)}")
    
    print()
    
    # API Endpoints
    print("🔌 API ENDPOINTS:")
    api_endpoints = [
        ("/api/health", "Health Check"),
        ("/api/market-intelligence/insights", "Market Intelligence Insights"),
        ("/api/market-intelligence/real-time-data", "Real-time Market Data"),
        ("/api/ai/market-insight", "AI Market Insights"),
        ("/api/system/status", "System Status"),
    ]
    
    for path, name in api_endpoints:
        url = f"{base_url}{path}"
        if "insights" in path or "data" in path:
            print(f"  {check_api_endpoint(url, name)}")
        else:
            print(f"  {check_url(url, name)}")
    
    print()
    
    # Enhanced Intelligence Specific Test
    print("🧠 ENHANCED INTELLIGENCE TEST:")
    try:
        # Test the insights endpoint specifically
        insights_url = f"{base_url}/api/market-intelligence/insights"
        response = requests.get(insights_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ API Response: {response.status_code}")
            print(f"  📈 Success: {data.get('success', False)}")
            print(f"  📊 Insights Count: {len(data.get('insights', []))}")
            print(f"  📋 Stats Available: {'stats' in data}")
            
            # Test if insights have proper structure
            insights = data.get('insights', [])
            if insights:
                first_insight = insights[0]
                required_fields = ['id', 'title', 'description', 'symbol', 'timestamp']
                has_all_fields = all(field in first_insight for field in required_fields)
                print(f"  🏗️  Data Structure: {'Valid' if has_all_fields else 'Missing fields'}")
                print(f"  🔧 Sample Insight: {first_insight.get('title', 'No title')[:50]}...")
            else:
                print(f"  ⚠️  No insights returned")
                
        else:
            print(f"  ❌ API Failed: {response.status_code} {response.reason}")
            
    except Exception as e:
        print(f"  ❌ Enhanced Intelligence Test Failed: {str(e)}")
    
    print()
    print("🎯 RECOMMENDED ACTIONS:")
    print("1. If all dashboard pages show ✅, navigation should work properly")
    print("2. If Market Intelligence API shows ✅ with insights, backend is working") 
    print("3. If Market Intelligence Board page loads but shows no data, check browser console")
    print("4. Access dashboard via: http://localhost:5003")
    print("5. Direct market intelligence: http://localhost:5003/dashboard/market_intelligence_board.html")

if __name__ == "__main__":
    main() 