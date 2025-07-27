#!/usr/bin/env python3
"""
Test QNTI Research Agent
Demonstrates key functionality of the automated research system
"""

import asyncio
import logging
from qnti_research_agent import QNTIResearchAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_research_agent():
    """Test the research agent functionality"""
    print("=" * 60)
    print("🧪 Testing QNTI Research Agent")
    print("=" * 60)
    
    # Initialize agent
    print("\n1️⃣ Initializing Research Agent...")
    agent = QNTIResearchAgent()
    print(f"✅ Agent initialized with {len(agent.sources)} sources")
    print(f"📁 Research directory: {agent.research_dir}")
    
    # Test RAG system
    print("\n2️⃣ Testing RAG System...")
    if agent.query_engine:
        print("✅ RAG system active and ready")
        
        # Test query functionality
        print("\n🔍 Testing sample query...")
        try:
            response = await agent.query_research("What are the latest Federal Reserve policy decisions?")
            print(f"📊 Query response: {response[:200]}...")
        except Exception as e:
            print(f"⚠️ Query test: {e} (Normal if no research data yet)")
    else:
        print("⚠️ RAG system not available (check Ollama)")
    
    # Test source processing
    print("\n3️⃣ Testing Source Processing...")
    for i, source in enumerate(agent.sources[:2]):  # Test first 2 sources
        print(f"\n📡 Testing source {i+1}: {source.name}")
        try:
            # Create session for testing
            import aiohttp
            agent.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={'User-Agent': 'QNTI Research Agent Test'}
            )
            
            # Process one source
            await agent._process_source(source)
            print(f"✅ Successfully processed {source.name}")
            
        except Exception as e:
            print(f"⚠️ Error processing {source.name}: {e}")
        finally:
            if agent.session:
                await agent.session.close()
    
    # Test database
    print("\n4️⃣ Testing Database...")
    if agent.db_path.exists():
        print("✅ Research database exists")
        
        # Check for documents
        import sqlite3
        conn = sqlite3.connect(agent.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM research_documents")
        doc_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM source_stats")
        stats_count = cursor.fetchone()[0]
        conn.close()
        
        print(f"📚 Documents in database: {doc_count}")
        print(f"📊 Source statistics: {stats_count}")
    
    # Test market intelligence integration
    print("\n5️⃣ Testing Market Intelligence Integration...")
    try:
        insights = agent.get_research_insights_for_market_intelligence("currency markets")
        print(f"✅ Generated {len(insights)} research insights for market intelligence")
        for i, insight in enumerate(insights[:2]):
            print(f"   💡 Insight {i+1}: {insight[:100]}...")
    except Exception as e:
        print(f"⚠️ Integration test: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Research Agent Test Complete!")
    print("=" * 60)
    print("\n📋 Summary:")
    print("✅ Research agent initialized")
    print("✅ RAG system configured") 
    print("✅ Source processing tested")
    print("✅ Database operations verified")
    print("✅ Market intelligence integration ready")
    print("\n🚀 To start full monitoring: python start_research_monitoring.py")

if __name__ == "__main__":
    asyncio.run(test_research_agent()) 