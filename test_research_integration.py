#!/usr/bin/env python3
"""
Test Research Integration with Market Intelligence
"""

def test_research_integration():
    try:
        from qnti_enhanced_market_intelligence import enhanced_intelligence
        
        print("🔍 Testing Research Integration...")
        
        # Get current insights
        insights = enhanced_intelligence.get_insights(limit=20)
        print(f"Total insights: {len(insights)}")
        
        # Filter for research insights
        research_insights = [i for i in insights if i.get('insight_type') == 'research']
        print(f"Research insights: {len(research_insights)}")
        
        # Show research insights
        for i, insight in enumerate(research_insights[:3]):
            print(f"\n📚 Research Insight {i+1}:")
            print(f"   Title: {insight.get('title', 'No title')}")
            print(f"   Source: {insight.get('source', 'Unknown')}")
            print(f"   Description: {insight.get('description', 'No description')[:150]}...")
        
        # Test direct research agent
        print("\n🔬 Testing Research Agent Directly...")
        from qnti_research_agent import get_research_agent
        agent = get_research_agent()
        
        # Check database content
        import sqlite3
        conn = sqlite3.connect(agent.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM research_documents")
        doc_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM research_documents WHERE summary IS NOT NULL AND summary != ''")
        summary_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT title, summary FROM research_documents WHERE summary IS NOT NULL AND summary != '' LIMIT 1")
        sample_doc = cursor.fetchone()
        
        conn.close()
        
        print(f"   Database documents: {doc_count}")
        print(f"   Documents with summaries: {summary_count}")
        
        if sample_doc:
            print(f"   Sample document: {sample_doc[0]}")
            print(f"   Sample summary: {sample_doc[1][:100]}...")
        
        # Test research query
        test_insights = agent.get_research_insights_for_market_intelligence("currency markets")
        print(f"   Direct query returned: {len(test_insights)} insights")
        
        return len(research_insights) > 0
        
    except Exception as e:
        print(f"❌ Error testing research integration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_research_integration()
    print(f"\n{'✅ Integration working!' if success else '❌ Integration needs attention'}") 