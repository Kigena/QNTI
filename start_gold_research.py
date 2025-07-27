#!/usr/bin/env python3
"""
Start Enhanced Gold Research Monitoring
"""

from qnti_research_agent import get_research_agent
import asyncio
import time

async def start_gold_research():
    """Start enhanced research monitoring with gold focus"""
    print("🏆 Starting Enhanced Gold Research Monitoring")
    print("=" * 60)
    
    # Initialize the research agent
    print("1️⃣ Initializing Research Agent...")
    agent = get_research_agent()
    print(f"✅ Agent initialized with {len(agent.research_sources)} sources")
    
    # Show gold-focused sources
    gold_sources = [s for s in agent.research_sources if 'gold' in s.name.lower() or any('gold' in kw for kw in s.keywords)]
    print(f"🏆 Gold-focused sources: {len(gold_sources)}")
    for source in gold_sources:
        print(f"   • {source.name} (Priority: {source.priority})")
    
    # Start monitoring (this will run in background)
    print("\n2️⃣ Starting Background Research Monitoring...")
    await agent.start_monitoring()
    
    # Give it time to process some sources
    print("⏳ Processing initial sources (60 seconds)...")
    await asyncio.sleep(60)
    
    # Test gold research queries
    print("\n3️⃣ Testing Gold Research Queries...")
    test_queries = [
        "gold market analysis",
        "federal reserve gold policy", 
        "precious metals outlook",
        "gold price drivers",
        "central bank gold purchases"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        try:
            insights = agent.get_research_insights_for_market_intelligence(query)
            print(f"   📊 Results: {len(insights)} insights")
            for i, insight in enumerate(insights[:2], 1):
                print(f"   {i}. {insight[:100]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n{'='*60}")
    print("🏆 Gold Research Monitoring is Running!")
    print("💡 The system will continue to collect and prioritize gold research.")
    print("📊 Ask about gold in the Forex Advisor for enhanced insights!")

if __name__ == "__main__":
    asyncio.run(start_gold_research()) 