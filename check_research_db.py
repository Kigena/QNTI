#!/usr/bin/env python3
"""Check research database content"""

from qnti_research_agent import get_research_agent
import sqlite3

agent = get_research_agent()
conn = sqlite3.connect(agent.db_path)
cursor = conn.cursor()

print("📊 Research Database Analysis:")

# Check total documents
cursor.execute("SELECT COUNT(*) FROM research_documents")
total_docs = cursor.fetchone()[0]
print(f"   Total documents: {total_docs}")

# Check documents with summaries
cursor.execute("SELECT COUNT(*) FROM research_documents WHERE summary IS NOT NULL AND summary != ''")
summary_docs = cursor.fetchone()[0]
print(f"   Documents with summaries: {summary_docs}")

# Get sample documents
cursor.execute("SELECT title, summary, relevance_score FROM research_documents ORDER BY downloaded_date DESC LIMIT 3")
docs = cursor.fetchall()

print(f"\n📚 Sample Documents:")
for i, (title, summary, score) in enumerate(docs, 1):
    print(f"   {i}. {title[:50]}...")
    print(f"      Summary: {'✅ Available' if summary and summary.strip() else '❌ Empty'}")
    print(f"      Score: {score}")
    if summary and summary.strip():
        print(f"      Preview: {summary[:100]}...")
    print()

# Test search
print("🔍 Testing Currency Search:")
cursor.execute("""
    SELECT title, summary, relevance_score 
    FROM research_documents 
    WHERE LOWER(title) LIKE '%currency%' OR LOWER(title) LIKE '%monetary%' OR LOWER(title) LIKE '%federal%'
    ORDER BY relevance_score DESC 
    LIMIT 3
""")

search_results = cursor.fetchall()
print(f"   Found {len(search_results)} relevant documents")

for title, summary, score in search_results:
    insight_text = f"{title}: {summary}" if summary and summary.strip() else title
    print(f"   💡 {insight_text[:120]}...")

conn.close()

print("\n✅ Database check complete!") 