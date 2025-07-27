#!/usr/bin/env python3
"""
QNTI Vision Trading System - Complete Demonstration
==================================================

This script demonstrates the complete workflow of using the vision trading system
from SMC analysis to automated trade execution.
"""

import requests
import json
import time
from typing import Dict, Any

class VisionTradingDemo:
    def __init__(self, base_url: str = "http://localhost:5003"):
        self.base_url = base_url
        
    def print_header(self, title: str):
        """Print a formatted header"""
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
    
    def print_step(self, step: str, description: str):
        """Print a formatted step"""
        print(f"\n🔹 {step}: {description}")
    
    def make_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
        """Make a request to the QNTI API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url)
            elif method == "POST":
                response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "success": False}
    
    def check_system_status(self):
        """Check if QNTI system is running"""
        self.print_step("STEP 1", "Checking QNTI System Status")
        
        try:
            response = requests.get(f"{self.base_url}/api/health")
            if response.status_code == 200:
                print("✅ QNTI System is running and healthy")
                return True
            else:
                print("❌ QNTI System is not responding properly")
                return False
        except:
            print("❌ Cannot connect to QNTI System")
            print(f"   Make sure QNTI is running on {self.base_url}")
            return False
    
    def initialize_vision_trading(self):
        """Initialize the vision trading system"""
        self.print_step("STEP 2", "Initializing Vision Trading System")
        
        # First check current status
        status_response = self.make_request("/api/vision-trading/status")
        print(f"   Current Status: {status_response}")
        
        if not status_response.get("enabled", False):
            print("   🔧 Initializing vision trading...")
            init_response = self.make_request("/api/vision-trading/initialize", "POST")
            print(f"   Initialization Result: {init_response}")
        else:
            print("   ✅ Vision trading already initialized")
    
    def demonstrate_smc_analysis_parsing(self):
        """Demonstrate SMC analysis parsing"""
        self.print_step("STEP 3", "SMC Analysis Parsing Demo")
        
        # Sample SMC analysis in your format
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
        
        print("   📝 Sample SMC Analysis:")
        print("   " + sample_analysis.replace("\n", "\n   "))
        
        return sample_analysis
    
    def process_analysis_for_trading(self, analysis_text: str):
        """Process analysis and create vision trade"""
        self.print_step("STEP 4", "Processing Analysis for Trading")
        
        data = {
            "analysis_text": analysis_text,
            "symbol": "EURUSD"
        }
        
        print("   🔄 Sending analysis to vision trading system...")
        response = self.make_request("/api/vision-trading/process-analysis", "POST", data)
        
        if response.get("success"):
            print("   ✅ Analysis processed successfully!")
            print(f"   📊 Parsed Elements:")
            
            parsed = response.get("parsed_analysis", {})
            print(f"      • Bias: {parsed.get('directional_bias', {}).get('primary_bias', 'N/A')}")
            print(f"      • Conviction: {parsed.get('directional_bias', {}).get('conviction', 'N/A')}%")
            
            trading_plan = parsed.get('trading_plan', {})
            if trading_plan.get('entry_zone'):
                entry_zone = trading_plan['entry_zone']
                print(f"      • Entry Zone: {entry_zone.get('lower', 'N/A')} - {entry_zone.get('upper', 'N/A')}")
            
            if trading_plan.get('stop_loss'):
                print(f"      • Stop Loss: {trading_plan['stop_loss']}")
            
            take_profits = trading_plan.get('take_profits', {})
            for tp_key, tp_value in take_profits.items():
                print(f"      • {tp_key.upper()}: {tp_value}")
            
            # Check if trade was created
            if response.get("trade_created"):
                trade_id = response.get("trade_id")
                print(f"   🎯 Vision Trade Created: {trade_id}")
            else:
                print(f"   ⚠️  Trade not created: {response.get('message', 'Unknown reason')}")
        else:
            print(f"   ❌ Analysis processing failed: {response.get('error', 'Unknown error')}")
    
    def demonstrate_trade_management(self):
        """Demonstrate trade management features"""
        self.print_step("STEP 5", "Trade Management Features")
        
        # Get current vision trades
        trades_response = self.make_request("/api/vision-trading/trades")
        print("   📋 Current Vision Trades:")
        
        if trades_response.get("success"):
            trades = trades_response.get("trades", [])
            if trades:
                for trade in trades:
                    print(f"      • Trade ID: {trade.get('analysis_id', 'N/A')}")
                    print(f"        Symbol: {trade.get('symbol', 'N/A')}")
                    print(f"        Direction: {trade.get('direction', 'N/A')}")
                    print(f"        Status: {trade.get('status', 'N/A')}")
                    print(f"        Entry: {trade.get('entry_zone_lower', 'N/A')}-{trade.get('entry_zone_upper', 'N/A')}")
            else:
                print("      No active vision trades found")
        else:
            print(f"      ❌ Failed to get trades: {trades_response.get('error', 'Unknown error')}")
    
    def demonstrate_configuration(self):
        """Demonstrate configuration management"""
        self.print_step("STEP 6", "Configuration Management")
        
        # Get current configuration
        config_response = self.make_request("/api/vision-trading/config")
        print("   ⚙️  Current Configuration:")
        
        if config_response.get("success"):
            config = config_response.get("config", {})
            print(f"      • Risk Percentage: {config.get('risk_percentage', 'N/A')}%")
            print(f"      • Min Confidence: {config.get('min_confidence_threshold', 'N/A')}")
            print(f"      • Max Concurrent Trades: {config.get('max_concurrent_trades', 'N/A')}")
            print(f"      • Timeout: {config.get('trade_timeout_hours', 'N/A')} hours")
        else:
            print(f"      ❌ Failed to get config: {config_response.get('error', 'Unknown error')}")
    
    def show_dashboard_integration(self):
        """Show how vision trading integrates with dashboard"""
        self.print_step("STEP 7", "Dashboard Integration")
        
        print("   🖥️  Dashboard Integration Features:")
        print("      • Vision Analysis Panel with trading buttons")
        print("      • 'Preview Trade' - Shows parsed trade parameters")
        print("      • 'Create Trade' - Creates vision trade from analysis")
        print("      • 'Auto Trade' - Enables automatic trade creation")
        print("      • 'Settings' - Configure risk management")
        print()
        print("   📱 Access the dashboard at: http://localhost:5003")
        print("      1. Upload a chart image")
        print("      2. Run AI vision analysis")
        print("      3. Use vision trading buttons when analysis contains SMC elements")
    
    def show_usage_examples(self):
        """Show practical usage examples"""
        self.print_step("STEP 8", "Practical Usage Examples")
        
        examples = [
            {
                "scenario": "Manual Trade Preview",
                "steps": [
                    "1. Upload chart to dashboard",
                    "2. Run vision analysis",
                    "3. Click 'Preview Trade' to see parsed parameters",
                    "4. Review entry zones, stop loss, take profits",
                    "5. Click 'Create Trade' if satisfied"
                ]
            },
            {
                "scenario": "Automated Trading",
                "steps": [
                    "1. Configure risk settings via API or dashboard",
                    "2. Enable 'Auto Trade' mode",
                    "3. Upload charts with SMC analysis",
                    "4. System automatically creates trades from valid analyses",
                    "5. Monitor trades via dashboard or API"
                ]
            },
            {
                "scenario": "API Integration",
                "steps": [
                    "1. POST analysis text to /api/vision-trading/process-analysis",
                    "2. GET trade status from /api/vision-trading/trades",
                    "3. Update config via /api/vision-trading/config",
                    "4. Control monitoring via /api/vision-trading/start|stop"
                ]
            }
        ]
        
        for example in examples:
            print(f"\n   🎯 {example['scenario']}:")
            for step in example['steps']:
                print(f"      {step}")
    
    def run_complete_demo(self):
        """Run the complete demonstration"""
        self.print_header("QNTI VISION TRADING SYSTEM DEMONSTRATION")
        
        # Check system
        if not self.check_system_status():
            print("\n❌ Cannot proceed without QNTI system running")
            print("   Please start QNTI with: python qnti_main.py --port 5003")
            return
        
        # Initialize vision trading
        self.initialize_vision_trading()
        
        # Demonstrate parsing
        sample_analysis = self.demonstrate_smc_analysis_parsing()
        
        # Process analysis
        self.process_analysis_for_trading(sample_analysis)
        
        # Show trade management
        self.demonstrate_trade_management()
        
        # Show configuration
        self.demonstrate_configuration()
        
        # Show dashboard integration
        self.show_dashboard_integration()
        
        # Show usage examples
        self.show_usage_examples()
        
        self.print_header("DEMONSTRATION COMPLETE")
        print("🎉 Vision trading system is ready for use!")
        print("📖 Refer to the examples above for different usage scenarios")
        print("🔧 Check the dashboard at http://localhost:5003 for GUI interface")

def main():
    """Main demonstration function"""
    demo = VisionTradingDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main() 