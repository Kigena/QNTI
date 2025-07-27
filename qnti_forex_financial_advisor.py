"""
QNTI Forex Financial Advisor Chatbot
Advanced AI Financial Advisor specializing in Forex Trading with integrated market analysis
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import requests
import yfinance as yf
import time
from flask import Blueprint, jsonify, request

# Import existing QNTI components
try:
    import ollama
    from qnti_llm_mcp_integration import LLMConfig
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Ollama not available. Please install with: pip install ollama")

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not available. Please install with: pip install chromadb")

# Configure logging
# Import centralized logging
from qnti_logging_utils import get_qnti_logger
logger = get_qnti_logger(__name__)

@dataclass
class ChatSession:
    """User chat session data"""
    session_id: str
    user_id: str
    started_at: datetime
    last_activity: datetime
    conversation_history: List[Dict]
    user_profile: Dict
    trading_context: Dict

@dataclass
class MarketAnalysis:
    """Market analysis result"""
    symbol: str
    timeframe: str
    trend_direction: str
    strength: float
    support_levels: List[float]
    resistance_levels: List[float]
    key_levels: List[float]
    market_sentiment: str
    recommendation: str
    risk_level: str
    confidence: float

class ForexKnowledgeBase:
    """Comprehensive Forex Trading Knowledge Base"""
    
    def __init__(self):
        self.trading_concepts = {
            "risk_management": {
                "definition": "The practice of identifying, analyzing and taking steps to reduce or eliminate exposures to loss",
                "rules": [
                    "Never risk more than 1-2% of account per trade",
                    "Use proper position sizing",
                    "Always set stop losses",
                    "Diversify across currency pairs",
                    "Maintain risk-reward ratio of at least 1:2"
                ],
                "tools": ["Position sizing calculator", "Risk-reward calculator", "Stop loss placement"]
            },
            "technical_analysis": {
                "definition": "Analysis of statistical trends gathered from trading activity",
                "key_concepts": [
                    "Support and resistance levels",
                    "Trend identification",
                    "Chart patterns",
                    "Technical indicators",
                    "Volume analysis"
                ],
                "popular_indicators": {
                    "RSI": "Relative Strength Index - momentum oscillator (0-100)",
                    "MACD": "Moving Average Convergence Divergence - trend following",
                    "EMA": "Exponential Moving Average - trend identification",
                    "Bollinger Bands": "Volatility indicator with price bands",
                    "Fibonacci": "Retracement and extension levels"
                }
            },
            "fundamental_analysis": {
                "definition": "Analysis of economic, social and political forces that affect currency values",
                "key_factors": [
                    "Interest rates and monetary policy",
                    "Economic indicators (GDP, inflation, employment)",
                    "Political stability and events",
                    "Trade balance and current account",
                    "Market sentiment and risk appetite"
                ],
                "economic_calendar": [
                    "Non-Farm Payrolls (USD)",
                    "CPI Inflation data",
                    "Central bank meetings",
                    "GDP releases",
                    "PMI manufacturing data"
                ]
            },
            "currency_pairs": {
                "majors": {
                    "EURUSD": {"description": "Euro vs US Dollar", "characteristics": "Most liquid pair, low spreads"},
                    "GBPUSD": {"description": "British Pound vs US Dollar", "characteristics": "High volatility, news sensitive"},
                    "USDJPY": {"description": "US Dollar vs Japanese Yen", "characteristics": "Safe haven flows, carry trade"},
                    "USDCHF": {"description": "US Dollar vs Swiss Franc", "characteristics": "Safe haven, low volatility"},
                    "AUDUSD": {"description": "Australian Dollar vs US Dollar", "characteristics": "Commodity currency"},
                    "USDCAD": {"description": "US Dollar vs Canadian Dollar", "characteristics": "Oil correlation"}
                },
                "trading_sessions": {
                    "Sydney": "21:00-06:00 GMT",
                    "Tokyo": "23:00-08:00 GMT", 
                    "London": "07:00-16:00 GMT",
                    "New York": "12:00-21:00 GMT"
                }
            },
            "trading_psychology": {
                "key_principles": [
                    "Emotional discipline and control",
                    "Patience and waiting for setups",
                    "Accepting losses as part of trading",
                    "Avoiding revenge trading",
                    "Maintaining consistent approach"
                ],
                "common_mistakes": [
                    "Overtrading and excessive risk",
                    "FOMO (Fear of Missing Out)",
                    "Moving stop losses against you",
                    "Not taking profits at targets",
                    "Trading without a plan"
                ]
            }
        }
        
        self.market_conditions = {
            "trending": {
                "characteristics": "Clear directional movement, higher highs/lower lows",
                "best_strategies": ["Trend following", "Breakout trading", "Momentum strategies"],
                "indicators": ["Moving averages", "MACD", "ADX"]
            },
            "ranging": {
                "characteristics": "Sideways movement between support/resistance",
                "best_strategies": ["Mean reversion", "Range trading", "Oscillator strategies"],
                "indicators": ["RSI", "Stochastic", "Bollinger Bands"]
            },
            "volatile": {
                "characteristics": "High price swings, increased spreads",
                "best_strategies": ["Breakout trading", "News trading"],
                "risk_management": ["Reduce position sizes", "Wider stops", "Quick exits"]
            }
        }

class QNTIForexFinancialAdvisor:
    """Advanced Forex Financial Advisor Chatbot"""
    
    def __init__(self, qnti_system=None, config_file: str = "qnti_llm_config.json"):
        self.qnti_system = qnti_system
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
        # Initialize components
        self.knowledge_base = ForexKnowledgeBase()
        self.sessions: Dict[str, ChatSession] = {}
        self.memory_client = None
        self.collection = None
        
        # Data directories
        self.data_dir = Path("qnti_data")
        self.conversations_dir = self.data_dir / "conversations"
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB for conversation memory
        self._init_memory_database()
        
        # Forex-specific prompts
        self.system_prompts = self._load_system_prompts()
        
        logger.info("QNTI Forex Financial Advisor initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            else:
                return self._create_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict:
        """Create default configuration"""
        default_config = {
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
                "news_integration": True
            },
            "memory": {
                "conversation_retention_days": 30,
                "max_context_messages": 50,
                "enable_learning": True
            }
        }
        
        # Save default config
        try:
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving default config: {e}")
        
        return default_config
    
    def _init_memory_database(self):
        """Initialize ChromaDB for conversation memory"""
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available. Conversation memory disabled.")
            return
        
        try:
            memory_path = self.config.get("chroma", {}).get("path", "./qnti_memory")
            self.memory_client = chromadb.PersistentClient(path=memory_path)
            self.collection = self.memory_client.get_or_create_collection(
                name="forex_advisor_conversations",
                metadata={"description": "Forex Financial Advisor conversation history"}
            )
            logger.info("Conversation memory database initialized")
        except Exception as e:
            logger.error(f"Error initializing memory database: {e}")
    
    def _load_system_prompts(self) -> Dict[str, str]:
        """Load specialized system prompts for different scenarios"""
        return {
            "general_advisor": """You are a professional Forex Financial Advisor and trading expert with 15+ years of experience. 

Your expertise includes:
- Advanced technical and fundamental analysis
- Risk management and position sizing
- Market psychology and trading discipline
- Currency pair characteristics and correlations
- Economic indicators and news impact
- Multiple trading strategies and timeframes

Your personality:
- Professional yet approachable
- Educational and patient
- Risk-aware and conservative
- Evidence-based recommendations
- Clear and actionable advice

Always consider:
1. Risk management first
2. Market conditions and volatility
3. User's experience level and risk tolerance
4. Current economic environment
5. Proper position sizing

Provide specific, actionable advice with clear reasoning. When discussing trades, always include risk management parameters.""",

            "market_analysis": """You are conducting detailed forex market analysis. Focus on:

1. Technical Analysis:
   - Price action and chart patterns
   - Key support/resistance levels
   - Trend analysis across timeframes
   - Technical indicator signals

2. Fundamental Analysis:
   - Economic data and calendar events
   - Central bank policies and statements
   - Geopolitical factors
   - Market sentiment and risk appetite

3. Risk Assessment:
   - Volatility conditions
   - Correlation analysis
   - Market liquidity
   - Session overlaps and timing

Provide clear directional bias with confidence levels and specific entry/exit strategies.""",

            "risk_management": """You are a risk management specialist. Always emphasize:

1. Position Sizing:
   - Never risk more than 1-2% per trade
   - Calculate lot sizes based on stop loss distance
   - Consider account size and leverage

2. Stop Loss Placement:
   - Based on technical levels, not arbitrary percentages
   - Account for market volatility and spread
   - Never move stops against your position

3. Risk-Reward Analysis:
   - Minimum 1:2 risk-reward ratio
   - Clear take profit targets
   - Partial profit taking strategies

4. Portfolio Management:
   - Diversification across pairs
   - Correlation awareness
   - Maximum exposure limits

Be conservative and prioritize capital preservation.""",

            "educational": """You are a forex trading educator. Focus on:

1. Clear Explanations:
   - Break down complex concepts
   - Use practical examples
   - Relate to real market situations

2. Progressive Learning:
   - Start with basics if needed
   - Build upon previous knowledge
   - Provide actionable next steps

3. Practical Application:
   - Demo account practice
   - Paper trading exercises
   - Real-world examples

4. Common Pitfalls:
   - Highlight typical mistakes
   - Explain psychological challenges
   - Provide prevention strategies

Make learning engaging and immediately applicable."""
        }
    
    async def chat(self, message: str, user_id: str = "default", 
                  session_id: Optional[str] = None, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Main chat function with forex expertise"""
        
        # Get or create session
        if not session_id:
            session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session = self._get_or_create_session(user_id, session_id)
        
        # Determine conversation type
        conversation_type = self._classify_conversation(message)
        
        # Get enhanced context with REAL market intelligence
        enhanced_market_context = await self._get_enhanced_market_context()
        market_context = await self._get_market_context()
        trading_context = self._get_trading_context(session)
        knowledge_context = self._get_knowledge_context(message)
        
        # Build comprehensive prompt with enhanced intelligence
        system_prompt = self._build_enhanced_system_prompt(conversation_type, enhanced_market_context, market_context, trading_context)
        
        # Get conversation history
        conversation_history = self._get_conversation_history(session)
        
        # Prepare LLM messages
        messages = [
            {"role": "system", "content": system_prompt},
            *conversation_history,
            {"role": "user", "content": message}
        ]
        
        # Generate response
        response = await self._generate_response(messages, conversation_type)
        
        # Add market data if relevant
        if conversation_type in ["market_analysis", "trade_suggestion"]:
            response["market_data"] = market_context
        
        # Save conversation
        self._save_conversation(session, message, response["content"], conversation_type)
        
        # Update session
        session.last_activity = datetime.now()
        session.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_message": message,
            "assistant_response": response["content"],
            "type": conversation_type
        })
        
        return response
    
    def _classify_conversation(self, message: str) -> str:
        """Classify the type of conversation based on message content"""
        message_lower = message.lower()
        
        # Account progress/performance keywords - PRIORITY CHECK FIRST
        if any(phrase in message_lower for phrase in ["account progress", "account performance", "trading progress", "trading performance", "my progress", "how am i doing", "performance review", "account review", "trading results", "my results"]):
            return "account_progress"
        
        # Performance-related keywords
        elif any(word in message_lower for word in ["profit", "loss", "win rate", "drawdown", "returns", "pnl", "p&l"]):
            return "account_progress"
        
        # Market analysis keywords
        elif any(word in message_lower for word in ["analyze", "analysis", "market", "chart", "trend", "forecast"]):
            return "market_analysis"
        
        # Risk management keywords
        elif any(word in message_lower for word in ["risk", "position size", "stop loss", "money management"]):
            return "risk_management"
        
        # Educational keywords
        elif any(word in message_lower for word in ["explain", "how to", "what is", "learn", "teach", "understand"]):
            return "educational"
        
        # Trade suggestion keywords
        elif any(word in message_lower for word in ["trade", "buy", "sell", "entry", "signal", "setup"]):
            return "trade_suggestion"
        
        # News and fundamental keywords
        elif any(word in message_lower for word in ["news", "economic", "fed", "inflation", "gdp", "nfp"]):
            return "fundamental_analysis"
        
        else:
            return "general_advisor"
    
    async def _get_enhanced_market_context(self) -> Dict[str, Any]:
        """Get comprehensive market context including research insights"""
        context = {
            "market_data": {},
            "market_insights": [],
            "research_insights": [],
            "system_health": {},
            "trading_performance": {}
        }
        
        try:
            # Get market data using existing method
            context["market_data"] = await self._get_market_context()
            
            # Get enhanced market intelligence
            try:
                from qnti_enhanced_market_intelligence import enhanced_intelligence
                insights = enhanced_intelligence.get_insights(limit=15)
                context["market_insights"] = [
                    {
                        "title": insight.get("title", ""),
                        "description": insight.get("description", ""),
                        "priority": insight.get("priority", ""),
                        "symbol": insight.get("symbol", ""),
                        "confidence": insight.get("confidence", 0.0),
                        "insight_type": insight.get("insight_type", ""),
                        "timestamp": insight.get("timestamp", "")
                    }
                    for insight in insights if insight
                ]
                logger.info(f"Retrieved {len(context['market_insights'])} market insights")
            except Exception as e:
                logger.warning(f"Could not get enhanced market intelligence: {e}")
            
            # Get research insights
            try:
                from qnti_research_agent import get_research_agent
                agent = get_research_agent()
                
                # Get research insights for major markets
                research_queries = [
                    "currency markets monetary policy",
                    "gold precious metals analysis", 
                    "global financial markets outlook"
                ]
                
                for query in research_queries:
                    insights = agent.get_research_insights_for_market_intelligence(query)
                    for insight in insights:
                        context["research_insights"].append({
                            "query": query,
                            "content": insight,
                            "source": "research_database"
                        })
                
                logger.info(f"Retrieved {len(context['research_insights'])} research insights")
            except Exception as e:
                logger.warning(f"Could not get research insights: {e}")
            
            # Get system health
            if self.qnti_system:
                try:
                    if hasattr(self.qnti_system, 'trade_manager') and self.qnti_system.trade_manager:
                        health = self.qnti_system.trade_manager.get_system_health()
                        context["system_health"] = health
                        
                        # Get trading performance
                        context["trading_performance"] = {
                            "open_trades": health.get("open_trades", 0),
                            "total_profit": health.get("total_profit", 0.0),
                            "active_eas": health.get("active_eas", 0),
                            "success_rate": health.get("success_rate", 0.0)
                        }
                except Exception as e:
                    logger.warning(f"Could not get system health: {e}")
            
        except Exception as e:
            logger.error(f"Error getting enhanced market context: {e}")
        
        return context

    async def _get_market_context(self) -> Dict[str, Any]:
        """Enhanced market context combining QNTI MT5 data with real-time intelligence"""
        market_data = {}
        
        try:
            # First try FRESH yfinance data (prioritize over cached data)
            fresh_data = {}
            try:
                logger.info("🔄 PRIORITY: Fetching FRESH real-time market data from yfinance...")
                logger.info("🚨 FORCING FRESH DATA COLLECTION - BYPASSING ALL CACHE!")
                
                # Map symbols to yfinance format for fresh data
                yf_symbols = {
                    'EURUSD': 'EURUSD=X',
                    'GBPUSD': 'GBPUSD=X', 
                    'USDJPY': 'USDJPY=X',
                    'USDCHF': 'USDCHF=X',
                    'AUDUSD': 'AUDUSD=X',
                    'USDCAD': 'USDCAD=X',
                    'XAUUSD': 'GC=F',  # Gold futures for latest prices
                    'XAGUSD': 'SI=F',  # Silver
                    'BTCUSD': 'BTC-USD'
                }
                
                for advisor_symbol, yf_symbol in yf_symbols.items():
                    try:
                        # Get fresh data from yfinance
                        ticker = yf.Ticker(yf_symbol)
                        hist = ticker.history(period="2d", interval="1h")
                        info = ticker.info
                        
                        if not hist.empty:
                            current_price = hist['Close'].iloc[-1]
                            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                            change_24h = ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0
                            
                            fresh_data[advisor_symbol] = {
                                "price": float(current_price),
                                "change_24h": float(change_24h),
                                "volume": float(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
                                "high_52w": float(hist['High'].max()),
                                "low_52w": float(hist['Low'].min()),
                                "rsi": None,  # Can be calculated if needed
                                "macd": None,
                                "volatility": float(hist['Close'].std()) if len(hist) > 5 else None,
                                "bollinger_upper": None,
                                "bollinger_lower": None,
                                "last_update": datetime.now().isoformat(),
                                "source": "yfinance_fresh"
                            }
                            
                            logger.info(f"✅ FRESH {advisor_symbol}: ${current_price:.4f} ({change_24h:+.2f}%) from yfinance")
                    
                    except Exception as e:
                        logger.warning(f"❌ Failed to get fresh yfinance data for {advisor_symbol}: {e}")
                
                if fresh_data:
                    market_data.clear()  # Clear ALL old data first
                    market_data.update(fresh_data)
                    logger.info(f"🎯 SUCCESS: Using FRESH yfinance data for {len(fresh_data)} symbols")
                    logger.info("🔥 OLD CACHED DATA CLEARED - USING ONLY FRESH DATA!")
                
            except Exception as e:
                logger.warning(f"Fresh yfinance data collection failed: {e}")
            
            # Fallback to enhanced intelligence ONLY if fresh data failed
            if not fresh_data:
                enhanced_data = {}
                try:
                    from qnti_enhanced_market_intelligence import enhanced_intelligence
                    
                    # Get enhanced data for major symbols (as fallback)
                    enhanced_symbols = {
                        'EURUSD': 'EURUSD=X',
                        'GBPUSD': 'GBPUSD=X', 
                        'USDJPY': 'USDJPY=X',
                        'USDCHF': 'USDCHF=X',
                        'AUDUSD': 'AUDUSD=X',
                        'USDCAD': 'USDCAD=X',
                        'XAUUSD': 'GC=F',  # Gold
                        'XAGUSD': 'SI=F',  # Silver
                        'BTCUSD': 'BTC-USD'
                    }
                    
                    for advisor_symbol, enhanced_symbol in enhanced_symbols.items():
                        if enhanced_symbol in enhanced_intelligence.market_data:
                            data = enhanced_intelligence.market_data[enhanced_symbol]
                            tech = enhanced_intelligence.technical_indicators.get(enhanced_symbol)
                            
                            enhanced_data[advisor_symbol] = {
                                "price": data.price,
                                "change_24h": data.change_percent,
                                "volume": data.volume,
                                "high_52w": data.high_52w,
                                "low_52w": data.low_52w,
                                "rsi": getattr(tech, 'rsi', None) if tech else None,
                                "macd": getattr(tech, 'macd', None) if tech else None,
                                "volatility": getattr(tech, 'volatility', None) if tech else None,
                                "bollinger_upper": getattr(tech, 'bollinger_upper', None) if tech else None,
                                "bollinger_lower": getattr(tech, 'bollinger_lower', None) if tech else None,
                                "last_update": data.timestamp.isoformat() if data.timestamp else datetime.now().isoformat(),
                                "source": "enhanced_intelligence_fallback"
                            }
                    
                    logger.info(f"⚠️ Using FALLBACK enhanced intelligence for {len(enhanced_data)} symbols")
                    for symbol, data in enhanced_data.items():
                        logger.info(f"Fallback data for {symbol}: Price=${data.get('price', 'N/A')}, Source={data.get('source', 'N/A')}")
                    market_data.update(enhanced_data)
                
                except Exception as e:
                    logger.warning(f"Enhanced intelligence not available: {e}")
            
            # Get real-time MT5 data if available (as backup/supplement)
            if hasattr(self, 'qnti_system') and self.qnti_system and hasattr(self.qnti_system, 'mt5_bridge'):
                mt5_bridge = self.qnti_system.mt5_bridge
                if mt5_bridge and mt5_bridge.symbols:
                    
                    # Map MT5 symbols to forex advisor format
                    symbol_mapping = {
                        "EURUSD": "EURUSD",
                        "GBPUSD": "GBPUSD", 
                        "USDJPY": "USDJPY",
                        "USDCHF": "USDCHF",
                        "AUDUSD": "AUDUSD",
                        "USDCAD": "USDCAD",
                        "GOLD": "XAUUSD",  # GOLD -> XAUUSD mapping
                        "SILVER": "XAGUSD",
                        "BTCUSD": "BTCUSD"
                    }
                    
                    for mt5_symbol, advisor_symbol in symbol_mapping.items():
                        if mt5_symbol in mt5_bridge.symbols:
                            # ONLY use MT5 data if enhanced intelligence data is not available
                            if advisor_symbol not in market_data:
                                symbol_data = mt5_bridge.symbols[mt5_symbol]
                                
                                # Calculate spread
                                spread = (symbol_data.ask - symbol_data.bid) / symbol_data.bid * 100 if symbol_data.bid > 0 else 0
                                
                                market_data[advisor_symbol] = {
                                    "price": symbol_data.last,
                                    "bid": symbol_data.bid,
                                    "ask": symbol_data.ask,
                                    "change_24h": symbol_data.daily_change_percent,
                                    "spread": spread,
                                    "volume": symbol_data.volume,
                                    "last_update": symbol_data.time.isoformat() if symbol_data.time else datetime.now().isoformat(),
                                    "source": "mt5_fallback"
                                }
                            else:
                                # Enhanced data available - just add MT5-specific fields if needed
                                if mt5_symbol in mt5_bridge.symbols:
                                    symbol_data = mt5_bridge.symbols[mt5_symbol]
                                    spread = (symbol_data.ask - symbol_data.bid) / symbol_data.bid * 100 if symbol_data.bid > 0 else 0
                                    market_data[advisor_symbol].update({
                                        "bid": symbol_data.bid,
                                        "ask": symbol_data.ask,
                                        "spread": spread
                                    })
                    
                        logger.info(f"Retrieved real-time MT5 data for {len(market_data)} symbols")
                    for symbol, data in market_data.items():
                        logger.info(f"Final data for {symbol}: Price=${data.get('price', 'N/A')}, Source={data.get('source', 'N/A')}")
                    
                else:
                    logger.warning("MT5 bridge symbols not available, using fallback data")
                    market_data = self._get_fallback_market_data()
            else:
                logger.warning("QNTI MT5 system not available, using fallback data")
                market_data = self._get_fallback_market_data()
            
            # Add market sentiment indicators
            try:
                # Use MT5 account info for market sentiment if available
                if hasattr(self, 'qnti_system') and self.qnti_system and hasattr(self.qnti_system, 'mt5_bridge'):
                    mt5_bridge = self.qnti_system.mt5_bridge
                    account_info = mt5_bridge.get_account_info()
                    if account_info:
                        market_data["market_sentiment"] = {
                            "account_equity": account_info.get('equity', 0),
                            "account_balance": account_info.get('balance', 0),
                                                         "margin_level": account_info.get('margin_level', 0),
                             "sentiment": "Active Trading" if account_info.get('equity', 0) > account_info.get('balance', 0) else "Conservative"
                         }
            except:
                pass
            
            # Add session information
            current_hour = datetime.now().hour
            if 7 <= current_hour < 16:
                active_session = "London"
            elif 12 <= current_hour < 21:
                active_session = "New York" if current_hour >= 12 else "London"
            elif 21 <= current_hour or current_hour < 6:
                active_session = "Sydney/Tokyo"
            else:
                active_session = "Asian"
            
            market_data["trading_session"] = {
                "current": active_session,
                "hour_utc": current_hour,
                "overlaps": "London-NY" if 12 <= current_hour < 16 else None
            }
            
        except Exception as e:
            logger.error(f"Error getting market context: {e}")
        
        return market_data
    
    def _get_trading_context(self, session: ChatSession) -> Dict[str, Any]:
        """Get trading context from QNTI system"""
        context = {}
        
        if self.qnti_system and hasattr(self.qnti_system, 'trade_manager'):
            try:
                # Get open trades
                open_trades = [
                    trade for trade in self.qnti_system.trade_manager.trades.values() 
                    if trade.status.name == 'OPEN'
                ]
                
                context["open_trades"] = {
                    "count": len(open_trades),
                    "symbols": list(set([trade.symbol for trade in open_trades])),
                    "total_risk": sum([getattr(trade, 'risk_amount', 0) for trade in open_trades])
                }
                
                # Get account info
                if hasattr(self.qnti_system, 'mt5_bridge') and self.qnti_system.mt5_bridge:
                    account_status = self.qnti_system.mt5_bridge.get_mt5_status()
                    context["account"] = account_status.get("account_info", {})
                
                # Get recent performance
                all_trades = list(self.qnti_system.trade_manager.trades.values())
                if all_trades:
                    recent_trades = [t for t in all_trades if t.close_time and 
                                   t.close_time > datetime.now() - timedelta(days=7)]
                    
                    if recent_trades:
                        total_profit = sum([t.profit for t in recent_trades if t.profit])
                        winning_trades = len([t for t in recent_trades if t.profit and t.profit > 0])
                        
                        context["recent_performance"] = {
                            "trades_count": len(recent_trades),
                            "win_rate": (winning_trades / len(recent_trades) * 100) if recent_trades else 0,
                            "total_profit": total_profit,
                            "avg_profit": total_profit / len(recent_trades) if recent_trades else 0
                        }
                
            except Exception as e:
                logger.error(f"Error getting trading context: {e}")
        
        return context
    
    def _get_knowledge_context(self, message: str) -> str:
        """Get relevant knowledge base context"""
        message_lower = message.lower()
        relevant_knowledge = []
        
        # Check for specific topics
        for topic, content in self.knowledge_base.trading_concepts.items():
            indicators_keys = list(content.get("popular_indicators", {}).keys()) if isinstance(content.get("popular_indicators"), dict) else []
            if topic.replace("_", " ") in message_lower or any(
                keyword in message_lower for keyword in content.get("rules", []) + indicators_keys
            ):
                relevant_knowledge.append(f"{topic.replace('_', ' ').title()}: {content.get('definition', '')}")
        
        return "\n".join(relevant_knowledge[:3]) if relevant_knowledge else ""
    
    def _get_fallback_market_data(self) -> Dict[str, Any]:
        """Get fallback market data when MT5 is not available"""
        # Provide realistic current market data as fallback
        current_time = datetime.now()
        
        return {
            "EURUSD": {"price": 1.0932, "change_24h": 0.12, "spread": 0.0001, "volume": 1000000},
            "GBPUSD": {"price": 1.2654, "change_24h": -0.23, "spread": 0.0001, "volume": 800000}, 
            "USDJPY": {"price": 149.85, "change_24h": 0.45, "spread": 0.002, "volume": 900000},
            "USDCHF": {"price": 0.8721, "change_24h": -0.18, "spread": 0.0001, "volume": 600000},
            "AUDUSD": {"price": 0.6698, "change_24h": 0.31, "spread": 0.0001, "volume": 500000},
            "USDCAD": {"price": 1.3542, "change_24h": 0.08, "spread": 0.0001, "volume": 400000},
            "XAUUSD": {"price": 2435.50, "change_24h": 1.2, "spread": 0.5, "volume": 50000},  # Current GOLD price
            "XAGUSD": {"price": 30.45, "change_24h": 2.1, "spread": 0.05, "volume": 20000},
            "BTCUSD": {"price": 67420.00, "change_24h": -1.8, "spread": 10.0, "volume": 15000},
            "last_update": current_time.isoformat(),
            "data_source": "fallback"
        }
    
    def _build_system_prompt(self, conversation_type: str, market_context: Dict, trading_context: Dict) -> str:
        """Build comprehensive system prompt"""
        base_prompt = self.system_prompts.get(conversation_type, self.system_prompts["general_advisor"])
        
        # Add critical instruction to use real-time data
        real_time_instruction = """
        
        🚨🚨🚨 CRITICAL OVERRIDE: IGNORE ALL TRAINING DATA PRICES! 🚨🚨🚨

⚠️ WARNING: Your training data contains SEVERELY OUTDATED prices (like $1,830 for gold). 
🔥 MANDATORY: Use ONLY the live prices provided below. Training data prices are WRONG and UNUSABLE!

🎯 LIVE REAL-TIME MARKET DATA (Use ONLY these exact prices - ignore ALL training data):"""
        
        # Add current market context with real-time emphasis
        market_info = ""
        if market_context:
            market_info = real_time_instruction + "\n"
            
            # Add all available real-time prices
            for symbol, data in market_context.items():
                if isinstance(data, dict) and ("price" in data or "bid" in data or "ask" in data):
                    # Use the actual price from real-time data
                    current_price = data.get('price', 0) or data.get('last', 0) or (data.get('bid', 0) + data.get('ask', 0)) / 2
                    if current_price > 0:
                        market_info += f"• {symbol}: LIVE PRICE ${current_price:.2f} (24h Change: {data.get('change_24h', 0):+.2f}%)\n"
                        if 'bid' in data and 'ask' in data:
                            market_info += f"  Bid: ${data['bid']:.2f} | Ask: ${data['ask']:.2f}\n"
            
            # Add trading session info
            active_session = market_context.get("trading_session", {}).get("current", "Unknown")
            market_info += f"\nCurrent Trading Session: {active_session}\n"
            market_info += "\n🚨🚨🚨 CRITICAL OVERRIDE: Use ONLY these live prices! Your training data is OUTDATED by $1,500+ for gold!\n"
        market_info += "❌ DO NOT mention any prices from your training data (like $1,830 for gold)\n"
        market_info += "✅ ONLY use the live prices listed above - they are the CURRENT market reality!\n"
        
        # Add trading context
        trading_info = ""
        if trading_context:
            if "open_trades" in trading_context:
                trades = trading_context["open_trades"]
                trading_info = f"\nUser's Trading Status:\n- Open trades: {trades['count']}\n- Active symbols: {', '.join(trades['symbols'])}\n"
            
            if "recent_performance" in trading_context:
                perf = trading_context["recent_performance"]
                trading_info += f"- Recent win rate: {perf['win_rate']:.1f}% ({perf['trades_count']} trades)\n"
        
        return f"{base_prompt}\n\n{market_info}{trading_info}\nProvide specific, actionable advice with clear reasoning."
    
    def _build_enhanced_system_prompt(self, conversation_type: str, enhanced_market_context: str, 
                                     market_context: Dict, trading_context: Dict) -> str:
        """Build enhanced system prompt with real-time market intelligence"""
        
        base_prompt = f"""You are a professional Forex Financial Advisor with access to REAL-TIME market intelligence and comprehensive trading analysis.

**YOUR EXPERTISE**:
- Advanced technical and fundamental analysis
- Risk management and position sizing
- Real-time market intelligence interpretation  
- Professional trading psychology
- Regulatory compliance and best practices

**CURRENT MARKET INTELLIGENCE**:
{enhanced_market_context}

**CONVERSATION TYPE**: {conversation_type}

**REAL-TIME TRADING DATA**:
"""
        
        # Add trading context
        if trading_context.get('open_trades'):
            base_prompt += f"- Open Trades: {trading_context['open_trades']['count']} positions\n"
            base_prompt += f"- Active Symbols: {', '.join(trading_context['open_trades']['symbols'])}\n"
        
        if trading_context.get('account'):
            account = trading_context['account']
            base_prompt += f"- Account Equity: ${account.get('equity', 0):,.2f}\n"
            base_prompt += f"- Available Margin: ${account.get('margin_free', 0):,.2f}\n"
        
        # Add enhanced market data summary
        if market_context:
            enhanced_symbols = [symbol for symbol, data in market_context.items() 
                              if isinstance(data, dict) and data.get('source') == 'enhanced_intelligence']
            if enhanced_symbols:
                base_prompt += f"- Enhanced Intelligence Active: {len(enhanced_symbols)} symbols\n"
        
        # Conversation-specific instructions
        if conversation_type == "market_analysis":
            base_prompt += """
**ANALYSIS INSTRUCTIONS**:
- Use REAL-TIME technical indicators (RSI, MACD, volatility) in your analysis
- Reference specific insights from the market intelligence data
- Provide concrete price levels and timeframes
- Include risk assessment based on current volatility
- Give actionable recommendations with clear entry/exit criteria
"""
        elif conversation_type == "trade_suggestion":
            base_prompt += """
**TRADE SUGGESTION INSTRUCTIONS**:
- Base suggestions on REAL market insights and technical indicators
- Provide specific entry prices, stop losses, and profit targets
- Include position sizing recommendations (1-2% account risk)
- Reference current RSI/MACD conditions for timing
- Warn about high volatility periods and adjust accordingly
"""
        elif conversation_type == "risk_management":
            base_prompt += """
**RISK MANAGEMENT INSTRUCTIONS**:
- Calculate position sizes based on account equity and current volatility
- Use REAL volatility data to adjust stop loss distances
- Consider correlation between open positions
- Provide specific risk-reward ratios based on market conditions
"""
        elif conversation_type == "account_progress":
            base_prompt += """
**ACCOUNT PROGRESS ANALYSIS INSTRUCTIONS**:
- Focus on ACTUAL account performance using the real trading data provided above
- Calculate and discuss specific metrics: profit/loss, win rate, recent performance
- Reference actual account equity, balance, and margin levels
- Analyze the user's open positions and recent trading activity
- Provide constructive feedback on trading performance and areas for improvement
- Include specific recommendations for account growth and risk optimization
- Compare current performance to professional trading benchmarks
- DO NOT discuss market analysis unless directly related to account performance
"""
        
        base_prompt += """
**RESPONSE GUIDELINES**:
- Always reference REAL market data when available
- Use professional, confident but measured tone
- Include specific numbers and price levels
- Provide actionable, implementable advice
- Add appropriate risk disclaimers
- Format responses with clear sections and bullet points
- If asked about a specific symbol, use its real-time data from the intelligence system

**🚨 FINAL WARNING 🚨**: Base your advice EXCLUSIVELY on the REAL-TIME market data provided above. 

❌ FORBIDDEN: Do NOT reference ANY prices from your training data (especially the outdated $1,830 gold price)
✅ REQUIRED: Use ONLY the live prices shown above (like $3428+ for gold)
🎯 MANDATE: If you mention any price, it MUST be from the live data section above - NO EXCEPTIONS!

The real-time data includes actual current prices, RSI values, and market insights. Your training data is severely outdated and must be completely ignored for price information.
"""
        
        return base_prompt

    def _get_conversation_history(self, session: ChatSession) -> List[Dict[str, str]]:
        """Get formatted conversation history for LLM"""
        history = []
        max_messages = self.config.get("memory", {}).get("max_context_messages", 10)
        
        # Get recent messages from session
        recent_messages = session.conversation_history[-max_messages:] if session.conversation_history else []
        
        for msg in recent_messages:
            history.extend([
                {"role": "user", "content": msg["user_message"]},
                {"role": "assistant", "content": msg["assistant_response"]}
            ])
        
        return history
    
    async def _generate_response(self, messages: List[Dict], conversation_type: str) -> Dict[str, Any]:
        """Generate LLM response with forex expertise"""
        
        if not OLLAMA_AVAILABLE:
            return {
                "content": "I'm a Forex Financial Advisor, but I need Ollama to be installed to provide intelligent responses. Please install Ollama to enable full functionality.",
                "type": conversation_type,
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Configure model parameters based on conversation type
            temperature = 0.6 if conversation_type in ["market_analysis", "risk_management"] else 0.7
            max_tokens = 1500 if conversation_type == "educational" else 1000
            
            response = ollama.chat(
                model=self.config.get("advisor", {}).get("model", "llama3"),
                messages=messages,
                options={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": 0.9
                }
            )
            
            content = response["message"]["content"]
            
            # Add specialized formatting based on conversation type
            if conversation_type == "market_analysis":
                content = self._format_market_analysis(content)
            elif conversation_type == "risk_management":
                content = self._format_risk_advice(content)
            
            return {
                "content": content,
                "type": conversation_type,
                "model": self.config.get("advisor", {}).get("model", "llama3"),
                "timestamp": datetime.now().isoformat(),
                "tokens_used": len(content.split())
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "content": f"I apologize, but I'm experiencing technical difficulties. As your Forex Financial Advisor, let me provide some general guidance: Always prioritize risk management, use proper position sizing (1-2% per trade), and never trade without a clear plan. Please try your question again.",
                "type": conversation_type,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _format_market_analysis(self, content: str) -> str:
        """Format market analysis responses with structure"""
        if "analysis" not in content.lower():
            return content
        
        # Add professional formatting
        formatted = f"📊 **MARKET ANALYSIS**\n\n{content}\n\n"
        formatted += "⚠️ **Risk Disclaimer**: This analysis is for educational purposes. Always use proper risk management and consider your risk tolerance before trading."
        
        return formatted
    
    def _format_risk_advice(self, content: str) -> str:
        """Format risk management advice with emphasis"""
        formatted = f"🛡️ **RISK MANAGEMENT ADVICE**\n\n{content}\n\n"
        formatted += "💡 **Remember**: Capital preservation is more important than profit maximization. Never risk money you cannot afford to lose."
        
        return formatted
    
    def _get_or_create_session(self, user_id: str, session_id: str) -> ChatSession:
        """Get existing session or create new one"""
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        # Create new session
        session = ChatSession(
            session_id=session_id,
            user_id=user_id,
            started_at=datetime.now(),
            last_activity=datetime.now(),
            conversation_history=[],
            user_profile={},
            trading_context={}
        )
        
        self.sessions[session_id] = session
        return session
    
    def _save_conversation(self, session: ChatSession, user_message: str, 
                          assistant_response: str, conversation_type: str):
        """Save conversation to memory database and file"""
        
        # Save to ChromaDB if available
        if self.collection:
            try:
                doc_id = f"{session.session_id}_{len(session.conversation_history)}"
                self.collection.add(
                    documents=[f"User: {user_message}\nAssistant: {assistant_response}"],
                    metadatas=[{
                        "session_id": session.session_id,
                        "user_id": session.user_id,
                        "timestamp": datetime.now().isoformat(),
                        "type": conversation_type
                    }],
                    ids=[doc_id]
                )
            except Exception as e:
                logger.error(f"Error saving to ChromaDB: {e}")
        
        # Save to file
        try:
            session_file = self.conversations_dir / f"{session.session_id}.json"
            session_data = asdict(session)
            session_data["started_at"] = session_data["started_at"].isoformat()
            session_data["last_activity"] = session_data["last_activity"].isoformat()
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving session file: {e}")
    
    async def analyze_market(self, symbol: str, timeframe: str = "1h", lookback_months: int = 6) -> MarketAnalysis:
        """Enhanced comprehensive market analysis using REAL-TIME intelligence data with extended historical lookback"""
        try:
            # First try to get data from enhanced market intelligence
            enhanced_data = {}
            real_insights = []
            
            try:
                from qnti_enhanced_market_intelligence import enhanced_intelligence
                
                # Map symbol to enhanced intelligence format
                symbol_mapping = {
                    'EURUSD': 'EURUSD=X',
                    'GBPUSD': 'GBPUSD=X', 
                    'USDJPY': 'USDJPY=X',
                    'USDCHF': 'USDCHF=X',
                    'AUDUSD': 'AUDUSD=X',
                    'USDCAD': 'USDCAD=X',
                    'NZDUSD': 'NZDUSD=X',
                    'GOLD': 'GC=F',
                    'SILVER': 'SI=F',
                    'OIL': 'CL=F',
                    'BTCUSD': 'BTC-USD',
                    'ETHUSD': 'ETH-USD',
                    'SPX500': '^GSPC',
                    'NASDAQ': '^IXIC',
                    'DOW': '^DJI'
                }
                enhanced_symbol = symbol_mapping.get(symbol, symbol)
                
                # Get real-time market data and insights
                if enhanced_symbol in enhanced_intelligence.market_data:
                    market_data = enhanced_intelligence.market_data[enhanced_symbol]
                    technical_indicators = enhanced_intelligence.technical_indicators.get(enhanced_symbol)
                    
                    # Get specific insights for this symbol
                    symbol_insights = enhanced_intelligence.get_insights(symbol=enhanced_symbol, limit=10)
                    real_insights = [insight.get('description', '') for insight in symbol_insights if isinstance(insight, dict) and insight.get('description')]
                    
                    enhanced_data = {
                        'current_price': market_data.price,
                        'change_percent': market_data.change_percent,
                        'volume': market_data.volume,
                        'high_52w': market_data.high_52w,
                        'low_52w': market_data.low_52w,
                        'rsi': getattr(technical_indicators, 'rsi', None) if technical_indicators else None,
                        'macd': getattr(technical_indicators, 'macd', None) if technical_indicators else None,
                        'sma_20': getattr(technical_indicators, 'sma_20', None) if technical_indicators else None,
                        'ema_20': getattr(technical_indicators, 'ema_20', None) if technical_indicators else None,
                        'bollinger_upper': getattr(technical_indicators, 'bollinger_upper', None) if technical_indicators else None,
                        'bollinger_lower': getattr(technical_indicators, 'bollinger_lower', None) if technical_indicators else None,
                        'volatility': getattr(technical_indicators, 'volatility', None) if technical_indicators else None,
                        'atr': getattr(technical_indicators, 'atr', None) if technical_indicators else None
                    }
                    
                    logger.info(f"Enhanced market data found for {symbol}: {len(real_insights)} insights available")
                else:
                    logger.warning(f"No enhanced data for {symbol}, falling back to Yahoo Finance")
            except Exception as e:
                logger.warning(f"Enhanced intelligence not available for {symbol}: {e}")
            
            # Get extended historical data for deeper analysis (up to 6 months default)
            ticker_symbol = f"{symbol}=X" if "USD" in symbol else symbol
            ticker = yf.Ticker(ticker_symbol)
            
            # Extended historical data collection
            periods = {
                "1h": f"{min(30, lookback_months * 7)}d",  # Up to 30 days for hourly
                "4h": f"{min(90, lookback_months * 15)}d",  # Up to 90 days for 4h  
                "1d": f"{lookback_months}mo",  # Full months for daily
                "1w": f"{lookback_months * 2}mo"  # Extended for weekly
            }
            
            period = periods.get(timeframe, f"{lookback_months}mo")
            interval = "1h" if timeframe in ["1h", "4h"] else "1d"
            
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                raise ValueError(f"No historical data available for {symbol}")
            
            # Use enhanced data if available, otherwise calculate from historical
            if enhanced_data and enhanced_data.get('current_price'):
                current_price = enhanced_data['current_price']
                rsi = enhanced_data.get('rsi')
                macd = enhanced_data.get('macd')
                volatility = enhanced_data.get('volatility', 0)
                ema_20 = enhanced_data.get('ema_20')
            else:
                current_price = hist['Close'].iloc[-1]
                # Calculate RSI
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs)).iloc[-1] if not rs.empty else 50
                
                # Calculate MACD
                ema_12 = hist['Close'].ewm(span=12).mean()
                ema_26 = hist['Close'].ewm(span=26).mean()
                macd = (ema_12 - ema_26).iloc[-1] if len(hist) > 26 else 0
                
                # Calculate volatility
                volatility = hist['Close'].pct_change().std() * 100 * np.sqrt(252)
                ema_20 = hist['Close'].ewm(span=20).mean().iloc[-1] if len(hist) > 20 else current_price
            
            # Enhanced trend analysis with multiple timeframes
            ema_50 = hist['Close'].ewm(span=50).mean().iloc[-1] if len(hist) > 50 else current_price
            ema_200 = hist['Close'].ewm(span=200).mean().iloc[-1] if len(hist) > 200 else current_price
            
            # Multi-timeframe trend strength
            trend_factors = []
            if current_price > ema_20: trend_factors.append(0.3)
            if ema_20 > ema_50: trend_factors.append(0.3)
            if ema_50 > ema_200: trend_factors.append(0.4)
            
            bullish_strength = sum(trend_factors)
            
            if bullish_strength >= 0.7:
                trend = "Strong Bullish"
                strength = bullish_strength
            elif bullish_strength >= 0.3:
                trend = "Bullish"
                strength = bullish_strength
            elif bullish_strength <= 0.3:
                trend = "Bearish" 
                strength = 1 - bullish_strength
            else:
                trend = "Sideways"
                strength = 0.5
            
            # Enhanced support and resistance with longer lookback
            lookback_days = min(len(hist), lookback_months * 30)
            recent_data = hist.tail(lookback_days)
            
            # Pivot points analysis
            highs = recent_data['High'].rolling(window=20).max()
            lows = recent_data['Low'].rolling(window=20).min()
            
            # Get significant levels
            resistance_levels = sorted(highs.dropna().unique())[-5:]  # Top 5 resistance
            support_levels = sorted(lows.dropna().unique())[:5]  # Bottom 5 support
            
            # Enhanced market sentiment using real insights
            if real_insights:
                # Analyze sentiment from real insights
                bullish_keywords = ['buy', 'bullish', 'support', 'oversold', 'breakout', 'rally']
                bearish_keywords = ['sell', 'bearish', 'resistance', 'overbought', 'breakdown', 'decline']
                
                sentiment_score = 0
                for insight in real_insights:
                    insight_lower = insight.lower()
                    sentiment_score += sum(1 for word in bullish_keywords if word in insight_lower)
                    sentiment_score -= sum(1 for word in bearish_keywords if word in insight_lower)
                
                if sentiment_score > 2:
                    sentiment = "Strong Bullish"
                elif sentiment_score > 0:
                    sentiment = "Bullish"
                elif sentiment_score < -2:
                    sentiment = "Strong Bearish"
                elif sentiment_score < 0:
                    sentiment = "Bearish"
                else:
                    sentiment = "Neutral"
            else:
                # Fallback sentiment analysis
                price_change_24h = (current_price - hist['Close'].iloc[-24]) / hist['Close'].iloc[-24] * 100 if len(hist) > 24 else 0
                
                if abs(price_change_24h) > 2:
                    sentiment = "High Volatility"
                elif price_change_24h > 1:
                    sentiment = "Strong Bullish"
                elif price_change_24h > 0.5:
                    sentiment = "Bullish"
                elif price_change_24h < -1:
                    sentiment = "Strong Bearish"
                elif price_change_24h < -0.5:
                    sentiment = "Bearish"
                else:
                    sentiment = "Neutral"
            
            # Enhanced risk assessment
            if volatility > 30:
                risk_level = "Very High"
            elif volatility > 20:
                risk_level = "High"
            elif volatility > 15:
                risk_level = "Medium"
            elif volatility > 10:
                risk_level = "Low"
            else:
                risk_level = "Very Low"
            
            # Generate enhanced recommendation with real data
            recommendation = await self._generate_enhanced_trading_recommendation(
                symbol, trend, strength, sentiment, risk_level, current_price, 
                enhanced_data, real_insights, rsi, macd
            )
            
            # Enhanced confidence calculation
            confidence = 0.5  # Base confidence
            if enhanced_data:
                confidence += 0.2  # Bonus for real-time data
            if real_insights:
                confidence += 0.15  # Bonus for real insights
            if len(hist) > 200:
                confidence += 0.1  # Bonus for sufficient historical data
            if volatility < 20:
                confidence += 0.05  # Bonus for stable conditions
            
            confidence = min(confidence, 0.95)  # Cap at 95%
            
            return MarketAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                trend_direction=trend,
                strength=strength,
                support_levels=support_levels[-3:],
                resistance_levels=resistance_levels[-3:],
                key_levels=[current_price, ema_20, ema_50, ema_200],
                market_sentiment=sentiment,
                recommendation=recommendation,
                risk_level=risk_level,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error analyzing market for {symbol}: {e}")
            return MarketAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                trend_direction="Unknown",
                strength=0.0,
                support_levels=[],
                resistance_levels=[],
                key_levels=[],
                market_sentiment="Unknown",
                recommendation="Unable to analyze market data. Please check symbol and try again.",
                risk_level="Unknown",
                confidence=0.0
            )
    
    async def _generate_enhanced_trading_recommendation(self, symbol: str, trend: str, strength: float,
                                                      sentiment: str, risk_level: str, current_price: float,
                                                      enhanced_data: dict, real_insights: list, rsi: float, macd: float) -> str:
        """Generate enhanced trading recommendation using real-time market intelligence"""
        
        if not OLLAMA_AVAILABLE:
            # Enhanced fallback recommendation
            recommendation = f"📊 **{symbol} Analysis**\n\n"
            recommendation += f"**Trend**: {trend} (strength: {strength:.1f})\n"
            recommendation += f"**Sentiment**: {sentiment}\n" 
            recommendation += f"**Risk Level**: {risk_level}\n"
            
            if enhanced_data:
                recommendation += f"**Current Price**: {current_price:.5f}\n"
                if enhanced_data.get('rsi'):
                    rsi_status = "Overbought" if enhanced_data['rsi'] > 70 else "Oversold" if enhanced_data['rsi'] < 30 else "Neutral"
                    recommendation += f"**RSI**: {enhanced_data['rsi']:.1f} ({rsi_status})\n"
                if enhanced_data.get('volatility'):
                    recommendation += f"**Volatility**: {enhanced_data['volatility']:.1f}%\n"
                    
            if real_insights:
                recommendation += f"\n**Latest Market Insights**:\n"
                for insight in real_insights[:3]:
                    recommendation += f"• {insight}\n"
                    
            recommendation += f"\n⚠️ **Risk Management**: Use proper position sizing (1-2% risk per trade) and always set stop losses."
            return recommendation
        
        try:
            # Prepare enhanced market context
            insights_context = ""
            if real_insights:
                insights_context = f"\n**REAL-TIME MARKET INSIGHTS**:\n" + "\n".join([f"• {insight}" for insight in real_insights[:5]])
            
            technical_context = ""
            if enhanced_data:
                technical_context = f"""
**REAL-TIME TECHNICAL DATA**:
- RSI: {enhanced_data.get('rsi', 'N/A')} {'(Overbought)' if enhanced_data.get('rsi', 50) > 70 else '(Oversold)' if enhanced_data.get('rsi', 50) < 30 else '(Neutral)'}
- MACD: {enhanced_data.get('macd', 'N/A')}
- Volatility: {enhanced_data.get('volatility', 'N/A')}%
- 52W High: {enhanced_data.get('high_52w', 'N/A')}
- 52W Low: {enhanced_data.get('low_52w', 'N/A')}
- Volume: {enhanced_data.get('volume', 'N/A')}
"""
            
            prompt = f"""
            As a professional forex financial advisor with access to REAL-TIME market data, analyze {symbol} and provide a comprehensive trading recommendation:
            
            **CURRENT MARKET ANALYSIS**:
            - Trend: {trend} (strength: {strength:.1f})
            - Market Sentiment: {sentiment}
            - Risk Level: {risk_level}
            - Current Price: {current_price:.5f}
            {technical_context}
            {insights_context}
            
            **REQUIRED ANALYSIS**:
            1. **Directional Bias**: Clear bullish/bearish/neutral stance with reasoning
            2. **Entry Strategy**: Specific price levels and confirmation signals
            3. **Risk Management**: Stop loss placement and position sizing
            4. **Profit Targets**: 2-3 realistic target levels
            5. **Time Horizon**: Expected trade duration
            6. **Market Context**: How current conditions affect the trade
            
            **FORMAT**: Provide a professional, actionable trading plan that incorporates the real-time market insights. Be specific about price levels and risk management.
            
            **CRITICAL**: Base your analysis on the REAL market insights provided, not general assumptions. If RSI shows overbought/oversold conditions, factor that into your recommendation.
            """
            
            response = ollama.chat(
                model=self.config.get("advisor", {}).get("model", "llama3"),
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.6, "max_tokens": 500}
            )
            
            return response["message"]["content"]
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return f"Based on {trend} trend and {sentiment} sentiment, consider risk level {risk_level} in your trading decisions."
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about active sessions"""
        active_sessions = len(self.sessions)
        total_conversations = sum(len(session.conversation_history) for session in self.sessions.values())
        
        return {
            "active_sessions": active_sessions,
            "total_conversations": total_conversations,
            "memory_status": "Available" if self.collection else "Unavailable",
            "last_activity": max([s.last_activity for s in self.sessions.values()]).isoformat() if self.sessions else None
        }

    async def _generate_llm_response(self, message: str, session: ChatSession) -> str:
        """Generate response using Ollama with comprehensive market context"""
        try:
            # Get comprehensive context
            enhanced_context = await self._get_enhanced_market_context()
            market_context = await self._get_market_context()
            trading_context = self._get_trading_context(session)
            conversation_context = self._get_conversation_context(session)
            
            # Build comprehensive prompt with ALL data sources
            system_prompt = self._build_comprehensive_system_prompt(enhanced_context)
            
            # Create detailed prompt with context
            detailed_prompt = f"""
{system_prompt}

**CURRENT MARKET INTELLIGENCE**:
• Total Market Insights: {len(enhanced_context.get('market_insights', []))}
• Research Database Insights: {len(enhanced_context.get('research_insights', []))}
• System Health: {enhanced_context.get('trading_performance', {}).get('open_trades', 0)} open trades

**KEY MARKET INSIGHTS**:
"""
            
            # Add market insights
            for insight in enhanced_context.get('market_insights', [])[:5]:
                priority_icon = "🔴" if insight.get('priority') == 'high' else "🟡" if insight.get('priority') == 'medium' else "🟢"
                detailed_prompt += f"{priority_icon} {insight.get('title', '')}: {insight.get('description', '')[:100]}...\n"
            
            # Add research insights
            if enhanced_context.get('research_insights'):
                detailed_prompt += "\n**INSTITUTIONAL RESEARCH INSIGHTS**:\n"
                for research in enhanced_context.get('research_insights', [])[:3]:
                    detailed_prompt += f"📚 {research.get('content', '')[:150]}...\n"
            
            # Add market data
            detailed_prompt += f"\n**LIVE MARKET DATA**:\n"
            for symbol, data in enhanced_context.get('market_data', {}).items():
                if isinstance(data, dict):
                    change = data.get('change_percent', 0)
                    change_icon = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    detailed_prompt += f"{change_icon} {symbol}: ${data.get('price', 'N/A')} ({change:+.2f}%)\n"
            
            # Add trading performance
            if enhanced_context.get('trading_performance'):
                perf = enhanced_context['trading_performance']
                detailed_prompt += f"""
**YOUR TRADING PERFORMANCE**:
• Open Trades: {perf.get('open_trades', 0)}
• Total P&L: ${perf.get('total_profit', 0):.2f}
• Active EAs: {perf.get('active_eas', 0)}
• Success Rate: {perf.get('success_rate', 0):.1f}%
"""
            
            detailed_prompt += f"""
**CONVERSATION CONTEXT**:
{conversation_context}

**USER QUESTION**: {message}

**INSTRUCTIONS**: 
Provide a comprehensive, professional response as a Forex Financial Advisor using ALL the above real-time data. Include:
1. Direct answer to the user's question
2. Relevant market analysis from the insights
3. Research-backed recommendations when applicable
4. Risk management advice
5. Specific trading opportunities if relevant

Keep response conversational but professional, under 400 words.
"""
            
            # Generate response
            response = ollama.chat(
                model=self.config.get("llm_model", "llama3"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": detailed_prompt}
                ],
                options={
                    "temperature": 0.7,
                    "max_tokens": 500,
                    "timeout": 30
                }
            )
            
            generated_response = response["message"]["content"]
            
            # Store conversation
            self._store_conversation(session, message, generated_response)
            
            return generated_response
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return self._get_fallback_response()
    
    def _build_comprehensive_system_prompt(self, enhanced_context: Dict[str, Any]) -> str:
        """Build comprehensive system prompt with all available data"""
        
        market_insights_count = len(enhanced_context.get('market_insights', []))
        research_insights_count = len(enhanced_context.get('research_insights', []))
        
        return f"""You are a professional Forex Financial Advisor with access to REAL-TIME market intelligence and comprehensive research database.

**YOUR CAPABILITIES**:
• Access to {market_insights_count} live market insights
• {research_insights_count} institutional research insights  
• Real-time trading performance data
• Technical analysis and market sentiment
• Federal Reserve and central bank research
• Risk management expertise

**YOUR PERSONALITY**:
• Professional but friendly and approachable
• Evidence-based recommendations
• Always prioritize risk management
• Provide actionable insights
• Use market data to support advice

**DATA SOURCES**:
• Enhanced Yahoo Finance real-time data
• Federal Reserve research database
• European Central Bank publications
• Bank of England policy updates
• IMF global economic analysis
• Live trading system performance

**TRADING PHILOSOPHY**:
• Risk management is paramount (1-2% per trade)
• Evidence-based decision making
• Diversification and position sizing
• Stay informed on central bank policies
• Adapt to changing market conditions

Respond as a professional advisor who has access to institutional-grade research and real-time market intelligence."""

# Global instance for Flask integration
forex_advisor = None

def integrate_forex_advisor_with_flask(app, qnti_system=None):
    """Integrate Forex Financial Advisor with QNTI Flask app"""
    global forex_advisor
    
    try:
        # Initialize advisor if not already done
        if forex_advisor is None:
            forex_advisor = QNTIForexFinancialAdvisor(qnti_system=qnti_system)
        
        # Check if route is already registered to avoid conflicts
        if '/advisor/chat' not in [rule.rule for rule in app.url_map.iter_rules()]:
            @app.route('/advisor/chat', methods=['POST'])
            def advisor_chat():
                """Handle chat requests from the frontend"""
                try:
                    data = request.get_json()
                    
                    if not data:
                        return jsonify({
                            "success": False,
                            "error": "No data provided"
                        }), 400
                    
                    message = data.get('message', '').strip()
                    user_id = data.get('user_id', 'web_user')
                    session_id = data.get('session_id', f'session_{int(time.time())}')
                    
                    if not message:
                        return jsonify({
                            "success": False,
                            "error": "Message is required"
                        }), 400
                    
                    # Generate response using the enhanced advisor
                    import asyncio
                    response = asyncio.run(forex_advisor.chat(message, user_id, session_id))
                    
                    return jsonify({
                        "success": True,
                        "response": response,
                        "type": "advisor_response",
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    logger.error(f"Error in advisor chat endpoint: {e}")
                    return jsonify({
                        "success": False,
                        "error": "I'm experiencing technical difficulties. Please try again in a moment.",
                        "fallback_response": "As your Forex Financial Advisor, I recommend always prioritizing risk management (1-2% per trade), using stop losses, and having a clear trading plan. Please try your question again."
                    }), 500
        
        # Check if status route is already registered to avoid conflicts  
        if '/api/advisor/status' not in [rule.rule for rule in app.url_map.iter_rules()]:
            @app.route('/api/advisor/status', methods=['GET'])
            def advisor_status():
                """Get advisor system status"""
                try:
                    status = {
                        "advisor_online": forex_advisor is not None,
                        "ollama_available": False,
                        "enhanced_intelligence": False,
                        "research_database": False,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Check Ollama availability
                    try:
                        import ollama
                        models = ollama.list()
                        status["ollama_available"] = True
                        status["available_models"] = [m['name'] for m in models.get('models', [])]
                    except:
                        pass
                    
                    # Check enhanced intelligence
                    try:
                        from qnti_enhanced_market_intelligence import enhanced_intelligence
                        insights = enhanced_intelligence.get_insights(limit=1)
                        status["enhanced_intelligence"] = len(insights) > 0
                    except:
                        pass
                    
                    # Check research database
                    try:
                        from qnti_research_agent import get_research_agent
                        agent = get_research_agent()
                        status["research_database"] = True
                    except:
                        pass
                    
                    return jsonify(status)
                    
                except Exception as e:
                    logger.error(f"Error getting advisor status: {e}")
                    return jsonify({
                        "advisor_online": False,
                        "error": str(e)
                    }), 500
        
        logger.info("Forex Financial Advisor integrated with QNTI system")
        return True
        
    except Exception as e:
        logger.error(f"Error integrating Forex Financial Advisor: {e}")
        return False

if __name__ == "__main__":
    # Test the advisor
    advisor = QNTIForexFinancialAdvisor()
    
    async def test_advisor():
        response = await advisor.chat("Hello! I'm new to forex trading. Can you help me understand risk management?")
        print("Advisor Response:", response["content"])
        
        # Test market analysis
        analysis = await advisor.analyze_market("EURUSD")
        print("Market Analysis:", analysis.recommendation)
    
    import asyncio
    asyncio.run(test_advisor()) 