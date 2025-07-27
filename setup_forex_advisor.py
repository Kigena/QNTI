#!/usr/bin/env python3
"""
QNTI Forex Financial Advisor Setup Script
Automated setup and integration of the Forex Financial Advisor chatbot
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ForexAdvisorSetup:
    """Setup and configuration manager for the Forex Financial Advisor"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.requirements_file = self.project_root / "requirements_forex_advisor.txt"
        self.config_file = self.project_root / "qnti_llm_config.json"
        self.advisor_file = self.project_root / "qnti_forex_financial_advisor.py"
        self.chat_interface = self.project_root / "dashboard" / "forex_advisor_chat.html"
        
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed"""
        logger.info("Checking dependencies...")
        
        required_packages = [
            "ollama",
            "chromadb", 
            "pandas",
            "numpy",
            "yfinance",
            "flask",
            "requests"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✓ {package} is installed")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"✗ {package} is missing")
        
        if missing_packages:
            logger.error(f"Missing packages: {', '.join(missing_packages)}")
            return False
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install required dependencies"""
        logger.info("Installing Forex Advisor dependencies...")
        
        # Create requirements file
        requirements = """# QNTI Forex Financial Advisor Dependencies
ollama>=0.2.0
chromadb>=0.4.0
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.0
flask>=2.3.0
requests>=2.31.0
pydantic>=2.0.0
python-dateutil>=2.8.0
asyncio>=3.4.3

# Optional but recommended
newsapi-python>=0.2.6
plotly>=5.0.0
matplotlib>=3.7.0
seaborn>=0.12.0"""

        try:
            with open(self.requirements_file, 'w') as f:
                f.write(requirements)
            
            # Install packages
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)
            ])
            
            logger.info("✓ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
        except Exception as e:
            logger.error(f"Error creating requirements file: {e}")
            return False
    
    def setup_ollama(self) -> bool:
        """Setup Ollama LLM service"""
        logger.info("Setting up Ollama...")
        
        try:
            # Check if Ollama is installed
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✓ Ollama is installed")
            else:
                logger.error("Ollama is not installed. Please install from https://ollama.ai/download")
                return False
            
            # Check if llama3 model is available
            result = subprocess.run(["ollama", "list"], 
                                  capture_output=True, text=True)
            
            if "llama3" in result.stdout:
                logger.info("✓ llama3 model is available")
            else:
                logger.info("Installing llama3 model...")
                subprocess.run(["ollama", "pull", "llama3"])
                logger.info("✓ llama3 model installed")
            
            # Test Ollama connection
            import ollama
            try:
                ollama.list()
                logger.info("✓ Ollama service is running")
                return True
            except Exception as e:
                logger.error(f"Ollama service connection failed: {e}")
                logger.info("Please start Ollama service manually")
                return False
                
        except FileNotFoundError:
            logger.error("Ollama command not found. Please install Ollama first.")
            return False
        except Exception as e:
            logger.error(f"Error setting up Ollama: {e}")
            return False
    
    def setup_configuration(self) -> bool:
        """Setup configuration files"""
        logger.info("Setting up configuration...")
        
        # Enhanced configuration for forex advisor
        config = {
            "llm": {
                "model": "llama3",
                "base_url": "http://localhost:11434",
                "temperature": 0.7,
                "max_tokens": 2000,
                "timeout": 60
            },
            "chroma": {
                "path": "./qnti_memory",
                "collection_name": "qnti_context",
                "persist_directory": "./qnti_memory"
            },
            "advisor": {
                "model": "llama3",
                "temperature": 0.7,
                "max_tokens": 2000,
                "specialization": "forex_trading",
                "risk_tolerance": "moderate",
                "response_style": "professional_friendly"
            },
            "features": {
                "market_analysis": True,
                "risk_assessment": True,
                "educational_content": True,
                "trade_suggestions": True,
                "news_integration": True,
                "enable_news_analysis": True,
                "enable_market_sentiment": True,
                "enable_trade_correlation": True,
                "enable_performance_insights": True,
                "enable_risk_assessment": True
            },
            "memory": {
                "conversation_retention_days": 30,
                "max_context_messages": 50,
                "enable_learning": True,
                "max_context_documents": 1000,
                "context_retention_days": 30
            },
            "news": {
                "api_key": "",
                "update_interval": 60,
                "sources": ["reuters", "bloomberg", "cnbc", "forexfactory"],
                "queries": ["forex trading", "market analysis", "economic indicators", "central bank", "currency"]
            },
            "market_data": {
                "symbols": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X"],
                "update_interval": 30,
                "major_pairs": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD"]
            },
            "scheduling": {
                "daily_brief_hour": 6,
                "daily_brief_minute": 0,
                "news_update_interval": 60,
                "market_data_interval": 30,
                "context_cleanup_hour": 2
            },
            "security": {
                "api_key": "qnti-secret-key",
                "jwt_secret": "qnti-jwt-secret-2024",
                "token_expiry_hours": 24
            },
            "integration": {
                "trade_log_path": "./trade_log.csv",
                "open_trades_path": "./open_trades.json",
                "backup_path": "./qnti_backups",
                "enable_auto_context": True,
                "context_window_size": 20
            },
            "limits": {
                "max_daily_requests": 10000,
                "max_response_length": 5000,
                "max_session_messages": 100
            }
        }
        
        try:
            # Update existing config or create new one
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    existing_config = json.load(f)
                
                # Merge configurations
                existing_config.update(config)
                config = existing_config
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info("✓ Configuration file updated")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up configuration: {e}")
            return False
    
    def setup_web_integration(self) -> bool:
        """Setup web interface integration"""
        logger.info("Setting up web interface integration...")
        
        try:
            # Update main dashboard navigation
            main_dashboard = self.project_root / "dashboard" / "main_dashboard.html"
            
            if main_dashboard.exists():
                with open(main_dashboard, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Add forex advisor link to navigation
                if "forex_advisor_chat.html" not in content:
                    # Find nav menu and add link
                    nav_pattern = r'(<a href="analytics_reports\.html" class="nav-link">📊 Analytics</a>)'
                    replacement = r'\1\n                <a href="forex_advisor_chat.html" class="nav-link">💬 Forex Advisor</a>'
                    
                    updated_content = content.replace(nav_pattern, replacement)
                    
                    if updated_content != content:
                        with open(main_dashboard, 'w', encoding='utf-8') as f:
                            f.write(updated_content)
                        logger.info("✓ Main dashboard navigation updated")
                    else:
                        logger.info("ℹ️ Main dashboard navigation already includes forex advisor")
            
            # Check if chat interface file exists
            if self.chat_interface.exists():
                logger.info("✓ Forex advisor chat interface is ready")
            else:
                logger.warning("Chat interface file not found. Please ensure forex_advisor_chat.html is in the dashboard folder.")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up web integration: {e}")
            return False
    
    def setup_qnti_integration(self) -> bool:
        """Setup integration with QNTI main system"""
        logger.info("Setting up QNTI integration...")
        
        try:
            # Check if main system file exists
            qnti_main = self.project_root / "qnti_main_system.py"
            
            if not qnti_main.exists():
                logger.error("qnti_main_system.py not found. Please ensure you're in the QNTI project directory.")
                return False
            
            # Create integration code snippet
            integration_code = '''
# Add this to your qnti_main_system.py in the _setup_web_interface method:

# Initialize Forex Financial Advisor
try:
    from qnti_forex_financial_advisor import integrate_forex_advisor_with_qnti
    self.forex_advisor = integrate_forex_advisor_with_qnti(self)
    if self.forex_advisor:
        logger.info("Forex Financial Advisor integrated successfully")
    else:
        logger.warning("Forex Financial Advisor integration failed")
except ImportError:
    logger.info("Forex Financial Advisor not available (module not found)")
except Exception as e:
    logger.warning(f"Forex Financial Advisor integration failed: {e}")
'''
            
            # Save integration snippet
            integration_file = self.project_root / "forex_advisor_integration_snippet.py"
            with open(integration_file, 'w') as f:
                f.write(integration_code)
            
            logger.info("✓ QNTI integration snippet created")
            logger.info("📝 Please manually add the integration code to qnti_main_system.py")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up QNTI integration: {e}")
            return False
    
    def create_test_script(self) -> bool:
        """Create a test script for the forex advisor"""
        logger.info("Creating test script...")
        
        test_script = '''#!/usr/bin/env python3
"""
Test script for QNTI Forex Financial Advisor
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from qnti_forex_financial_advisor import QNTIForexFinancialAdvisor

async def test_forex_advisor():
    """Test the forex advisor functionality"""
    print("🧪 Testing QNTI Forex Financial Advisor...")
    
    # Initialize advisor
    advisor = QNTIForexFinancialAdvisor()
    
    # Test questions
    test_questions = [
        "Hello! I'm new to forex trading. Can you help me understand the basics?",
        "What is proper risk management in forex?",
        "How should I analyze EURUSD for trading opportunities?",
        "What are the best trading sessions for forex?",
        "Can you explain position sizing?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\\n{'='*60}")
        print(f"Test {i}: {question}")
        print('='*60)
        
        try:
            response = await advisor.chat(question, user_id="test_user")
            print(f"✓ Response ({response['type']}):")
            print(response['content'][:300] + "..." if len(response['content']) > 300 else response['content'])
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Test market analysis
    print(f"\\n{'='*60}")
    print("Testing Market Analysis")
    print('='*60)
    
    try:
        analysis = await advisor.analyze_market("EURUSD", "1h")
        print(f"✓ Market Analysis for EURUSD:")
        print(f"  Trend: {analysis.trend_direction}")
        print(f"  Sentiment: {analysis.market_sentiment}")
        print(f"  Risk Level: {analysis.risk_level}")
        print(f"  Recommendation: {analysis.recommendation[:200]}...")
        
    except Exception as e:
        print(f"✗ Market Analysis Error: {e}")
    
    print(f"\\n{'='*60}")
    print("🎉 Forex Advisor Testing Complete!")
    print('='*60)

if __name__ == "__main__":
    asyncio.run(test_forex_advisor())
'''
        
        try:
            test_file = self.project_root / "test_forex_advisor.py"
            with open(test_file, 'w') as f:
                f.write(test_script)
            
            logger.info("✓ Test script created: test_forex_advisor.py")
            return True
            
        except Exception as e:
            logger.error(f"Error creating test script: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run basic tests to verify installation"""
        logger.info("Running installation tests...")
        
        try:
            # Test imports
            from qnti_forex_financial_advisor import QNTIForexFinancialAdvisor
            logger.info("✓ Forex advisor module imports successfully")
            
            # Test advisor initialization
            advisor = QNTIForexFinancialAdvisor()
            logger.info("✓ Forex advisor initializes successfully")
            
            # Test configuration
            if advisor.config:
                logger.info("✓ Configuration loaded successfully")
            
            return True
            
        except ImportError as e:
            logger.error(f"Import test failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Initialization test failed: {e}")
            return False
    
    def print_next_steps(self):
        """Print next steps for the user"""
        logger.info("\n🎉 QNTI Forex Financial Advisor Setup Complete!")
        
        print(f"""
{'='*70}
🚀 NEXT STEPS TO COMPLETE INTEGRATION
{'='*70}

1. 📝 MANUAL INTEGRATION REQUIRED:
   Add the following to your qnti_main_system.py file in the _setup_web_interface method:
   
   # Initialize Forex Financial Advisor
   try:
       from qnti_forex_financial_advisor import integrate_forex_advisor_with_qnti
       self.forex_advisor = integrate_forex_advisor_with_qnti(self)
       logger.info("Forex Financial Advisor integrated successfully")
   except Exception as e:
       logger.warning(f"Forex Financial Advisor integration failed: {{e}}")

2. 🔄 RESTART QNTI SYSTEM:
   python qnti_main.py --port 5003

3. 🌐 ACCESS FOREX ADVISOR:
   Open your browser to: http://localhost:5003/dashboard/forex_advisor_chat.html

4. 🧪 TEST THE SYSTEM:
   python test_forex_advisor.py

5. ⚙️ OPTIONAL CONFIGURATION:
   - Edit qnti_llm_config.json to customize advisor behavior
   - Add NewsAPI key for enhanced market analysis
   - Configure model parameters in the advisor section

{'='*70}
🎯 FEATURES AVAILABLE:
{'='*70}

✓ Professional Forex Trading Advisor Chat
✓ Real-time Market Analysis & Insights  
✓ Risk Management Guidance
✓ Educational Trading Content
✓ Personalized Trading Advice
✓ Integration with QNTI Trading Data
✓ Memory-based Conversation Context
✓ Multi-timeframe Technical Analysis

{'='*70}
📞 SUPPORT:
{'='*70}

If you encounter issues:
1. Check that Ollama service is running
2. Verify all dependencies are installed
3. Ensure QNTI main system is running
4. Check logs in qnti_main.log for errors

Happy Trading! 🚀📈
""")

def main():
    """Main setup function"""
    print("🧠 QNTI Forex Financial Advisor Setup")
    print("="*50)
    
    setup = ForexAdvisorSetup()
    
    # Check if we're in the right directory
    if not (setup.project_root / "qnti_main_system.py").exists():
        logger.error("❌ Not in QNTI project directory. Please run from QNTI root folder.")
        sys.exit(1)
    
    # Step 1: Check/Install dependencies
    if not setup.check_dependencies():
        logger.info("Installing missing dependencies...")
        if not setup.install_dependencies():
            logger.error("❌ Failed to install dependencies")
            sys.exit(1)
    
    # Step 2: Setup Ollama
    if not setup.setup_ollama():
        logger.warning("⚠️ Ollama setup incomplete. Please install Ollama manually.")
        print("Download from: https://ollama.ai/download")
    
    # Step 3: Setup configuration
    if not setup.setup_configuration():
        logger.error("❌ Failed to setup configuration")
        sys.exit(1)
    
    # Step 4: Setup web integration
    if not setup.setup_web_integration():
        logger.warning("⚠️ Web integration incomplete")
    
    # Step 5: Setup QNTI integration
    if not setup.setup_qnti_integration():
        logger.warning("⚠️ QNTI integration setup incomplete")
    
    # Step 6: Create test script
    if not setup.create_test_script():
        logger.warning("⚠️ Test script creation failed")
    
    # Step 7: Run tests
    if not setup.run_tests():
        logger.warning("⚠️ Some tests failed")
    
    # Step 8: Print next steps
    setup.print_next_steps()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "deps":
            setup = ForexAdvisorSetup()
            setup.install_dependencies()
        elif sys.argv[1] == "test":
            setup = ForexAdvisorSetup()
            setup.run_tests()
        else:
            print("Usage: python setup_forex_advisor.py [deps|test]")
    else:
        main() 