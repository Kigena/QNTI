#!/usr/bin/env python3
"""
QNTI Research Agent Monitoring Starter
Starts the automated research monitoring as a background service
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qnti_research.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ResearchMonitoringService:
    """Research monitoring service manager"""
    
    def __init__(self):
        self.running = False
        self.research_agent = None
        
    async def start(self):
        """Start the research monitoring service"""
        try:
            from qnti_research_agent import QNTIResearchAgent
            
            self.research_agent = QNTIResearchAgent()
            self.running = True
            
            logger.info("🚀 Starting QNTI Research Agent Monitoring...")
            logger.info("📊 Sources configured: Fed, ECB, BoE, IMF, BIS, Trading Economics")
            logger.info("🤖 AI summaries enabled with local LLM")
            logger.info("🔍 RAG system active for intelligent queries")
            
            # Start monitoring
            await self.research_agent.start_monitoring()
            
        except KeyboardInterrupt:
            logger.info("🛑 Received shutdown signal")
            await self.shutdown()
        except Exception as e:
            logger.error(f"❌ Error starting research monitoring: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the service gracefully"""
        logger.info("🔄 Shutting down research monitoring...")
        self.running = False
        
        if self.research_agent and self.research_agent.session:
            await self.research_agent.session.close()
        
        logger.info("✅ Research monitoring stopped")

def signal_handler(service):
    """Handle shutdown signals"""
    def handler(signum, frame):
        logger.info(f"Received signal {signum}")
        asyncio.create_task(service.shutdown())
    return handler

async def main():
    """Main entry point"""
    print("=" * 70)
    print("🔬 QNTI Automated Research Agent")
    print("=" * 70)
    print("\n🎯 Features:")
    print("• Automatic research downloads from major financial institutions")
    print("• AI-powered summaries and analysis")
    print("• Real-time RAG integration with market intelligence")
    print("• Continuous monitoring and indexing")
    print("\n📡 Sources:")
    print("• Federal Reserve (Monetary Policy, Speeches)")
    print("• European Central Bank (Press Releases)")  
    print("• Bank of England (Policy Updates)")
    print("• International Monetary Fund (Global Analysis)")
    print("• Bank for International Settlements (Central Banking)")
    print("• Trading Economics (Economic Indicators)")
    print("\n🔧 Requirements:")
    print("• Ollama server running: ollama serve")
    print("• LLaMA3 model available: ollama pull llama3")
    print("• Internet connection for research downloads")
    
    input("\nPress Enter to start monitoring...")
    
    service = ResearchMonitoringService()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler(service))
    signal.signal(signal.SIGTERM, signal_handler(service))
    
    try:
        await service.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 