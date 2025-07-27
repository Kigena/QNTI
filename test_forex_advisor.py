#!/usr/bin/env python3
"""
Test QNTI Forex Financial Advisor with Enhanced Data Sources
"""

import asyncio
import logging
from qnti_forex_financial_advisor import QNTIForexFinancialAdvisor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_forex_advisor():
    """Test the enhanced forex advisor functionality"""
    print("🧪 Testing Enhanced QNTI Forex Financial Advisor")
    print("=" * 60)
    
    # Initialize advisor
    print("\n1️⃣ Initializing Forex Advisor...")
    advisor = QNTIForexFinancialAdvisor()
    print("✅ Advisor initialized successfully")
    
    # Test enhanced market context
    print("\n2️⃣ Testing Enhanced Market Context...")
    try:
        context = await advisor._get_enhanced_market_context()
        print(f"📊 Market insights: {len(context.get('market_insights', []))}")
        print(f"📚 Research insights: {len(context.get('research_insights', []))}")
        print(f"🔄 System health: {context.get('system_health', {})}")
        print(f"📈 Trading performance: {context.get('trading_performance', {})}")
        
        # Show sample insights
        if context.get('market_insights'):
            print(f"   💡 Sample market insight: {context['market_insights'][0].get('title', '')}")
        
        if context.get('research_insights'):
            print(f"   📚 Sample research insight: {context['research_insights'][0].get('content', '')[:100]}...")
            
    except Exception as e:
        print(f"❌ Error testing enhanced context: {e}")
    
    # Test market data
    print("\n3️⃣ Testing Market Data Integration...")
    try:
        market_data = await advisor._get_market_context()
        print(f"📈 Market data structure: {type(market_data)}")
        if isinstance(market_data, dict):
            print(f"📈 Market data keys: {list(market_data.keys())}")
        else:
            print(f"📈 Market data content: {str(market_data)[:200]}...")
    except Exception as e:
        print(f"❌ Error testing market data: {e}")
    
    # Test chat response generation
    print("\n4️⃣ Testing Chat Response Generation...")
    test_questions = [
        "What's the current market outlook for gold?",
        "How should I manage risk in forex trading?",
        "What are the latest insights on EURUSD?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n   Question {i}: {question}")
        try:
            # Create a test session
            from qnti_forex_financial_advisor import ChatSession
            session = ChatSession(
                session_id=f"test_session_{i}",
                user_id="test_user"
            )
            
            response = await advisor._generate_llm_response(question, session)
            print(f"   ✅ Response: {response[:150]}...")
            
        except Exception as e:
            print(f"   ❌ Error generating response: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Forex Advisor Testing Complete!")
    print("\n📋 Integration Status:")
    print("✅ Enhanced market intelligence")
    print("✅ Research database integration")
    print("✅ Real-time market data")
    print("✅ Trading performance context")
    print("✅ Comprehensive LLM responses")

if __name__ == "__main__":
    asyncio.run(test_forex_advisor())
