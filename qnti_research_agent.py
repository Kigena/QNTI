#!/usr/bin/env python3
"""
QNTI Automated Research Agent
Automatically downloads, processes, and summarizes latest financial research
Integrates with local RAG stack for intelligent market insights
"""

import os
import json
import logging
import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
import sqlite3
from urllib.parse import urljoin, urlparse
import re

# Web scraping and parsing
import requests
from bs4 import BeautifulSoup
import feedparser
import PyPDF2
import docx2txt
from markdownify import markdownify

# LLM and RAG
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from llama_index.core import VectorStoreIndex, Document, Settings
    from llama_index.core.storage.storage_context import StorageContext
    from llama_index.core.storage.docstore import SimpleDocumentStore
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResearchSource:
    """Research source configuration"""
    name: str
    url: str
    source_type: str  # 'rss', 'web', 'api'
    selectors: Dict[str, str]  # CSS selectors for web scraping
    update_frequency: int  # minutes
    keywords: List[str]
    priority: int  # 1-10, higher = more important
    enabled: bool = True

@dataclass
class ResearchDocument:
    """Research document structure"""
    id: str
    title: str
    source: str
    url: str
    content: str
    summary: str
    keywords: List[str]
    published_date: datetime
    downloaded_date: datetime
    file_path: Optional[str] = None
    relevance_score: float = 0.0

class QNTIResearchAgent:
    """Automated research agent with RAG integration"""
    
    def __init__(self, research_dir: str = "qnti_research"):
        self.research_dir = Path(research_dir)
        self.research_dir.mkdir(exist_ok=True)
        
        # Setup subdirectories
        self.raw_dir = self.research_dir / "raw"
        self.processed_dir = self.research_dir / "processed"
        self.summaries_dir = self.research_dir / "summaries"
        self.index_dir = self.research_dir / "rag_index"
        
        for dir_path in [self.raw_dir, self.processed_dir, self.summaries_dir, self.index_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Database for tracking research
        self.db_path = self.research_dir / "research.db"
        self._init_database()
        
        # RAG components
        self.rag_index = None
        self.query_engine = None
        self._init_rag_system()
        
        # Research sources configuration
        self.sources = self._load_research_sources()
        
        # Download session
        self.session = None
        
        logger.info(f"Research Agent initialized with {len(self.sources)} sources")
    
    def _init_database(self):
        """Initialize research tracking database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS research_documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                url TEXT UNIQUE,
                content_hash TEXT,
                summary TEXT,
                keywords TEXT,
                published_date TEXT,
                downloaded_date TEXT,
                file_path TEXT,
                relevance_score REAL DEFAULT 0.0,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS source_stats (
                source_name TEXT PRIMARY KEY,
                last_check TEXT,
                documents_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                avg_relevance REAL DEFAULT 0.0
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Research database initialized")
    
    def _init_rag_system(self):
        """Initialize LlamaIndex RAG system"""
        if not LLAMAINDEX_AVAILABLE:
            logger.warning("LlamaIndex not available. Install with: pip install llama-index")
            return
        
        try:
            # Configure LlamaIndex with local Ollama
            Settings.llm = Ollama(model="llama3", request_timeout=120.0)
            Settings.embed_model = OllamaEmbedding(model_name="llama3")
            
            # Try to load existing index
            if (self.index_dir / "index.json").exists():
                self.rag_index = VectorStoreIndex.load_from_disk(str(self.index_dir))
                logger.info("Loaded existing RAG index")
            else:
                # Create new index
                self.rag_index = VectorStoreIndex([])
                logger.info("Created new RAG index")
            
            # Create query engine
            self.query_engine = self.rag_index.as_query_engine(
                similarity_top_k=5,
                response_mode="compact"
            )
            
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            self.rag_index = None
            self.query_engine = None
    
    def _load_research_sources(self) -> List[ResearchSource]:
        """Load research sources configuration"""
        sources = [
            ResearchSource(
                name="Federal Reserve",
                url="https://www.federalreserve.gov/feeds/press_all.xml",
                source_type="rss",
                selectors={},
                update_frequency=60,
                keywords=["monetary policy", "interest rates", "inflation", "employment"],
                priority=10
            ),
            ResearchSource(
                name="ECB Press",
                url="https://www.ecb.europa.eu/rss/press.html",
                source_type="rss",
                selectors={},
                update_frequency=60,
                keywords=["monetary policy", "euro", "banking", "financial stability"],
                priority=9
            ),
            ResearchSource(
                name="Bank of England",
                url="https://www.bankofengland.co.uk/news/speeches",
                source_type="web",
                selectors={
                    "title": "h1.page-title",
                    "content": ".article-content",
                    "date": ".publication-date"
                },
                update_frequency=120,
                keywords=["monetary policy", "inflation", "GBP", "financial stability"],
                priority=8
            ),
            ResearchSource(
                name="IMF Research",
                url="https://www.imf.org/en/News/rss?language=eng&series=IMF%20News",
                source_type="rss",
                selectors={},
                update_frequency=240,
                keywords=["global economy", "financial stability", "crisis", "growth"],
                priority=7
            ),
            ResearchSource(
                name="BIS Papers",
                url="https://www.bis.org/list/cbspeeches/index.htm",
                source_type="web",
                selectors={
                    "title": ".item_title a",
                    "content": ".item_abstract",
                    "date": ".item_date"
                },
                update_frequency=360,
                keywords=["central banking", "financial markets", "regulation"],
                priority=8
            ),
            ResearchSource(
                name="Trading Economics",
                url="https://tradingeconomics.com/stream",
                source_type="web",
                selectors={
                    "title": ".stream-title",
                    "content": ".stream-content",
                    "date": ".stream-date"
                },
                update_frequency=30,
                keywords=["economic indicators", "GDP", "inflation", "employment"],
                priority=6
            ),
            # Gold-focused research sources
            ResearchSource(
                name="World Gold Council",
                url="https://www.gold.org/news-and-insight",
                source_type="web",
                selectors={
                    "title": ".card-title a",
                    "content": ".card-excerpt",
                    "date": ".card-date"
                },
                update_frequency=60,
                keywords=["gold", "precious metals", "investment", "central bank", "demand", "supply"],
                priority=10  # High priority for gold
            ),
            ResearchSource(
                name="COMEX Gold Analysis",
                url="https://www.cmegroup.com/markets/metals/precious/gold.html",
                source_type="web",
                selectors={
                    "title": ".cmeContentTitle",
                    "content": ".cmeContentBody",
                    "date": ".cmeContentDate"
                },
                update_frequency=120,
                keywords=["gold futures", "precious metals", "COMEX", "trading", "prices"],
                priority=9
            ),
            ResearchSource(
                name="Kitco Gold News",
                url="https://www.kitco.com/news/rss/KitcoNewsRSS.xml",
                source_type="rss",
                selectors={},
                update_frequency=30,
                keywords=["gold", "precious metals", "mining", "investment", "forecast"],
                priority=10  # High priority for gold
            ),
            ResearchSource(
                name="Gold Price Analysis",
                url="https://www.investing.com/news/commodities-news",
                source_type="web",
                selectors={
                    "title": ".title",
                    "content": ".articlePage",
                    "date": ".date"
                },
                update_frequency=60,
                keywords=["gold", "precious metals", "commodities", "analysis", "forecast"],
                priority=9
            )
        ]
        
        return sources
    
    async def start_monitoring(self):
        """Start continuous research monitoring"""
        logger.info("Starting research monitoring...")
        
        # Create aiohttp session
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'QNTI Research Agent 1.0 (Educational Trading Research)'
            }
        )
        
        # Schedule periodic updates for each source
        tasks = []
        for source in self.sources:
            if source.enabled:
                task = asyncio.create_task(self._monitor_source(source))
                tasks.append(task)
        
        # Wait for all monitoring tasks
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in monitoring tasks: {e}")
        finally:
            if self.session:
                await self.session.close()
    
    async def _monitor_source(self, source: ResearchSource):
        """Monitor a single research source"""
        logger.info(f"Starting monitoring for {source.name}")
        
        while True:
            try:
                await self._process_source(source)
                await asyncio.sleep(source.update_frequency * 60)  # Convert to seconds
            except Exception as e:
                logger.error(f"Error monitoring {source.name}: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _process_source(self, source: ResearchSource):
        """Process a single research source"""
        logger.info(f"Processing source: {source.name}")
        
        try:
            if source.source_type == "rss":
                documents = await self._process_rss_source(source)
            elif source.source_type == "web":
                documents = await self._process_web_source(source)
            elif source.source_type == "api":
                documents = await self._process_api_source(source)
            else:
                logger.warning(f"Unknown source type: {source.source_type}")
                return
            
            # Process and store new documents
            for doc in documents:
                if await self._is_new_document(doc):
                    await self._store_document(doc)
                    await self._generate_summary(doc)
                    await self._add_to_rag_index(doc)
            
            # Update source statistics
            await self._update_source_stats(source, len(documents))
            
            logger.info(f"Processed {len(documents)} documents from {source.name}")
            
        except Exception as e:
            logger.error(f"Error processing source {source.name}: {e}")
    
    async def _process_rss_source(self, source: ResearchSource) -> List[ResearchDocument]:
        """Process RSS feed source"""
        documents = []
        
        try:
            async with self.session.get(source.url) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    for entry in feed.entries[:10]:  # Limit to latest 10
                        doc = ResearchDocument(
                            id=self._generate_doc_id(entry.link),
                            title=entry.title,
                            source=source.name,
                            url=entry.link,
                            content=entry.get('summary', ''),
                            summary='',
                            keywords=source.keywords,
                            published_date=self._parse_date(entry.get('published', '')),
                            downloaded_date=datetime.now()
                        )
                        
                        # Fetch full content if available
                        full_content = await self._fetch_full_content(doc.url)
                        if full_content:
                            doc.content = full_content
                        
                        documents.append(doc)
                        
        except Exception as e:
            logger.error(f"Error processing RSS {source.name}: {e}")
        
        return documents
    
    async def _process_web_source(self, source: ResearchSource) -> List[ResearchDocument]:
        """Process web scraping source"""
        documents = []
        
        try:
            async with self.session.get(source.url) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Extract articles based on selectors
                    articles = soup.select(source.selectors.get('title', 'article'))[:5]
                    
                    for article in articles:
                        title_elem = article if article.name in ['h1', 'h2', 'h3'] else article.select_one('h1, h2, h3, a')
                        title = title_elem.get_text(strip=True) if title_elem else "Untitled"
                        
                        # Get article URL
                        link_elem = article.find('a') or article
                        url = link_elem.get('href', '') if link_elem else source.url
                        if url and not url.startswith('http'):
                            url = urljoin(source.url, url)
                        
                        # Get content
                        content_elem = article.select_one(source.selectors.get('content', ''))
                        content = content_elem.get_text(strip=True) if content_elem else title
                        
                        doc = ResearchDocument(
                            id=self._generate_doc_id(url),
                            title=title,
                            source=source.name,
                            url=url,
                            content=content,
                            summary='',
                            keywords=source.keywords,
                            published_date=datetime.now(),  # Approximate
                            downloaded_date=datetime.now()
                        )
                        
                        documents.append(doc)
                        
        except Exception as e:
            logger.error(f"Error processing web source {source.name}: {e}")
        
        return documents
    
    async def _process_api_source(self, source: ResearchSource) -> List[ResearchDocument]:
        """Process API-based source"""
        # Placeholder for API sources (NewsAPI, Alpha Vantage, etc.)
        documents = []
        logger.info(f"API source processing not yet implemented for {source.name}")
        return documents
    
    async def _fetch_full_content(self, url: str) -> str:
        """Fetch full content from URL"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Get text content
                    text = soup.get_text()
                    
                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = '\n'.join(chunk for chunk in chunks if chunk)
                    
                    return text[:10000]  # Limit to 10k characters
                    
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
        
        return ""
    
    async def _generate_summary(self, doc: ResearchDocument):
        """Generate AI summary of document"""
        if not OLLAMA_AVAILABLE:
            doc.summary = doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
            return
        
        try:
            prompt = f"""
Summarize the following financial/economic research document for a trading system:

Title: {doc.title}
Source: {doc.source}
Content: {doc.content[:3000]}

Provide a concise summary focusing on:
1. Key market implications
2. Trading/investment insights
3. Economic indicators mentioned
4. Risk factors or opportunities

Summary (max 300 words):
"""
            
            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "max_tokens": 400}
            )
            
            doc.summary = response["message"]["content"]
            
            # Calculate relevance score based on keywords
            doc.relevance_score = self._calculate_relevance_score(doc)
            
            logger.info(f"Generated summary for: {doc.title[:50]}...")
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            doc.summary = doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
    
    def _calculate_relevance_score(self, doc: ResearchDocument) -> float:
        """Calculate relevance score with gold prioritization"""
        score = 0.0
        content_lower = (doc.title + " " + doc.content).lower()
        
        # GOLD-PRIORITY keywords (highest value)
        gold_keywords = [
            "gold", "precious metals", "bullion", "xau", "gold price", "gold mining",
            "gold demand", "gold supply", "gold reserves", "central bank gold",
            "gold investment", "safe haven", "inflation hedge"
        ]
        
        # High-value trading keywords
        high_value_keywords = [
            "interest rates", "monetary policy", "inflation", "gdp", "employment",
            "central bank", "fed", "ecb", "boe", "crisis", "volatility",
            "forex", "currency", "dollar", "euro", "pound", "yen",
            "recession", "growth", "stimulus", "quantitative easing"
        ]
        
        # GOLD gets massive priority boost
        gold_score = 0
        for keyword in gold_keywords:
            if keyword in content_lower:
                gold_score += 3.0  # Much higher weight for gold
        
        # If gold content, start with high base score
        if gold_score > 0:
            score = 5.0 + gold_score  # High base score for gold content
        
        # Add regular keyword scores
        for keyword in high_value_keywords:
            if keyword in content_lower:
                score += 1.0
        
        # Enhanced source priority with gold sources
        source_priority_map = {
            "World Gold Council": 3.0,       # Highest priority for gold
            "Kitco Gold News": 2.8,          # High priority for gold
            "COMEX Gold Analysis": 2.6,      # High priority for gold  
            "Gold Price Analysis": 2.4,      # High priority for gold
            "Federal Reserve": 2.0,
            "ECB Press": 1.8,
            "Bank of England": 1.6,
            "IMF Research": 1.4,
            "BIS Papers": 1.5
        }
        
        source_multiplier = source_priority_map.get(doc.source, 1.0)
        score *= source_multiplier
        
        # Extra boost for gold content from any source
        if gold_score > 0:
            score *= 1.5  # Additional 50% boost for gold content
        
        return min(score, 15.0)  # Higher cap to accommodate gold priority
    
    async def _add_to_rag_index(self, doc: ResearchDocument):
        """Add document to RAG index"""
        if not self.rag_index:
            return
        
        try:
            # Create LlamaIndex document
            llamaindex_doc = Document(
                text=f"{doc.title}\n\n{doc.summary}\n\n{doc.content}",
                metadata={
                    "title": doc.title,
                    "source": doc.source,
                    "url": doc.url,
                    "published_date": doc.published_date.isoformat(),
                    "relevance_score": doc.relevance_score,
                    "keywords": ",".join(doc.keywords)
                }
            )
            
            # Add to index
            self.rag_index.insert(llamaindex_doc)
            
            # Save index
            self.rag_index.storage_context.persist(persist_dir=str(self.index_dir))
            
            logger.info(f"Added to RAG index: {doc.title[:50]}...")
            
        except Exception as e:
            logger.error(f"Error adding to RAG index: {e}")
    
    async def query_research(self, question: str) -> str:
        """Query the research database using RAG"""
        if not self.query_engine:
            return "RAG system not available"
        
        try:
            response = self.query_engine.query(question)
            return str(response)
        except Exception as e:
            logger.error(f"Error querying research: {e}")
            return f"Error querying research: {e}"
    
    def get_research_insights_for_market_intelligence(self, market_context: str) -> List[str]:
        """Get relevant research insights for market intelligence with gold prioritization"""
        insights = []
        
        try:
            # Enhanced queries with gold prioritization
            if any(term in market_context.lower() for term in ['gold', 'xau', 'precious', 'metal']):
                # Gold-specific queries
                queries = [
                    f"Gold price drivers and central bank policies affecting precious metals",
                    f"Federal Reserve monetary policy impact on gold prices",
                    f"Gold demand supply dynamics and inflation hedging",
                    f"Central bank gold purchases and reserves",
                    f"Geopolitical factors affecting gold markets"
                ]
            else:
                # General market queries with gold emphasis
                queries = [
                    f"Recent central bank policy changes affecting {market_context} and gold",
                    f"Economic indicators impacting {market_context} and precious metals markets",
                    f"Risk factors for {market_context} trading and gold correlation",
                    f"Federal Reserve policy impact on {market_context} and safe haven assets"
                ]
            
            for query in queries:
                if self.query_engine:
                    response = self.query_engine.query(query)
                    if response and len(str(response)) > 50:
                        insights.append(str(response)[:300] + "...")
                        if len(insights) >= 3:  # Get enough insights
                            break
            
            # Fallback to enhanced database query if RAG not available
            if not insights:
                insights = self._get_database_insights_enhanced(market_context)
            
        except Exception as e:
            logger.error(f"Error getting research insights: {e}")
        
        return insights[:3]  # Return top 3 insights
    
    def _get_database_insights(self, market_context: str) -> List[str]:
        """Get insights from database as fallback"""
        insights = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Search for relevant documents using actual column names
            search_terms = market_context.lower().split()
            search_conditions = []
            
            for term in search_terms[:3]:  # Limit to 3 terms to avoid overly complex queries
                search_conditions.append(f"(LOWER(title) LIKE '%{term}%' OR LOWER(summary) LIKE '%{term}%')")
            
            search_query = " OR ".join(search_conditions)
            
            cursor.execute(f"""
                SELECT title, summary, relevance_score 
                FROM research_documents 
                WHERE {search_query}
                ORDER BY relevance_score DESC, downloaded_date DESC 
                LIMIT 5
            """)
            
            results = cursor.fetchall()
            for title, summary, score in results:
                # Use summary if available, otherwise use title
                if summary and summary.strip():
                    insight_text = f"{title}: {summary}"
                else:
                    insight_text = title
                
                if insight_text and len(insight_text) > 30:
                    insights.append(insight_text)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error querying database insights: {e}")
        
        return insights

    def _get_database_insights_enhanced(self, market_context: str) -> List[str]:
        """Enhanced database search with gold prioritization"""
        insights = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Gold-priority search terms
            gold_terms = ['gold', 'precious', 'metal', 'xau', 'bullion', 'mining']
            fed_terms = ['federal', 'reserve', 'fed', 'fomc', 'monetary', 'policy']
            
            search_terms = market_context.lower().split()
            
            # Build prioritized search query
            search_conditions = []
            
            # Priority 1: Gold-specific content
            for term in gold_terms:
                search_conditions.append(f"(LOWER(title) LIKE '%{term}%' OR LOWER(summary) LIKE '%{term}%')")
            
            # Priority 2: Federal Reserve and monetary policy
            for term in fed_terms:
                search_conditions.append(f"(LOWER(title) LIKE '%{term}%' OR LOWER(summary) LIKE '%{term}%')")
            
            # Priority 3: Original market context terms
            for term in search_terms[:2]:
                search_conditions.append(f"(LOWER(title) LIKE '%{term}%' OR LOWER(summary) LIKE '%{term}%')")
            
            search_query = " OR ".join(search_conditions)
            
            # Enhanced query with priority scoring
            cursor.execute(f"""
                SELECT title, summary, relevance_score,
                       CASE 
                           WHEN LOWER(title) LIKE '%gold%' OR LOWER(summary) LIKE '%gold%' THEN relevance_score + 3
                           WHEN LOWER(title) LIKE '%precious%' OR LOWER(summary) LIKE '%precious%' THEN relevance_score + 2
                           WHEN LOWER(title) LIKE '%federal%' OR LOWER(summary) LIKE '%federal%' THEN relevance_score + 1
                           ELSE relevance_score 
                       END as priority_score
                FROM research_documents 
                WHERE {search_query}
                ORDER BY priority_score DESC, downloaded_date DESC 
                LIMIT 5
            """)
            
            results = cursor.fetchall()
            for title, summary, score, priority_score in results:
                # Use summary if available, otherwise use title
                if summary and summary.strip():
                    insight_text = f"{title}: {summary}"
                else:
                    insight_text = title
                
                if insight_text and len(insight_text) > 30:
                    insights.append(insight_text[:400] + "..." if len(insight_text) > 400 else insight_text)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error querying enhanced database insights: {e}")
            # Fallback to standard search
            return self._get_database_insights(market_context)
        
        return insights
    
    # Helper methods
    def _generate_doc_id(self, url: str) -> str:
        """Generate unique document ID"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime"""
        try:
            # Try common date formats
            formats = [
                "%a, %d %b %Y %H:%M:%S %Z",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%d %b %Y"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str[:25], fmt)
                except ValueError:
                    continue
                    
        except Exception:
            pass
        
        return datetime.now()
    
    async def _is_new_document(self, doc: ResearchDocument) -> bool:
        """Check if document is new"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id FROM research_documents WHERE url = ?", (doc.url,))
            result = cursor.fetchone()
            
            conn.close()
            return result is None
            
        except Exception as e:
            logger.error(f"Error checking document: {e}")
            return True
    
    async def _store_document(self, doc: ResearchDocument):
        """Store document in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO research_documents 
                (id, title, source, url, content_hash, summary, keywords, 
                 published_date, downloaded_date, relevance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.id, doc.title, doc.source, doc.url,
                hashlib.md5(doc.content.encode()).hexdigest(),
                doc.summary, ",".join(doc.keywords),
                doc.published_date.isoformat(),
                doc.downloaded_date.isoformat(),
                doc.relevance_score
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored document: {doc.title[:50]}...")
            
        except Exception as e:
            logger.error(f"Error storing document: {e}")
    
    async def _update_source_stats(self, source: ResearchSource, doc_count: int):
        """Update source statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO source_stats 
                (source_name, last_check, documents_count)
                VALUES (?, ?, ?)
            """, (source.name, datetime.now().isoformat(), doc_count))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating source stats: {e}")

# Global research agent instance
research_agent = None

def get_research_agent() -> QNTIResearchAgent:
    """Get global research agent instance"""
    global research_agent
    if research_agent is None:
        research_agent = QNTIResearchAgent()
    return research_agent

async def start_research_monitoring():
    """Start research monitoring in background"""
    agent = get_research_agent()
    await agent.start_monitoring()

if __name__ == "__main__":
    # Test the research agent
    async def main():
        agent = QNTIResearchAgent()
        await agent.start_monitoring()
    
    asyncio.run(main()) 