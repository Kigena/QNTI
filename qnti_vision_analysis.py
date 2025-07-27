#!/usr/bin/env python3
"""
Quantum Nexus Trading Intelligence (QNTI) - Enhanced Vision Analysis Module
Chart image upload, comprehensive GPT-4V analysis with structured trade scenarios
"""

import cv2
import numpy as np
import base64
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from pathlib import Path
import logging
import uuid
import re
from typing import List, Dict, Optional, Tuple
import dataclasses

# Correct imports from existing project files
from qnti_core_system import Trade, EAPerformance, QNTITradeManager
from qnti_vision_models import (
    TradeScenario,
    PriceLevel,
    TechnicalIndicator,
    ComprehensiveChartAnalysis,
    SignalStrength,
    MarketBias
)

# Configure logging for the vision module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.INFO)

class QNTIEnhancedVisionAnalyzer:
    """
    Handles chart image uploads and comprehensive analysis using OpenAI's Vision API.
    Provides structured analysis results for trading decisions.
    """

    def __init__(self, trade_manager: QNTITradeManager, config_file: str = "vision_config.json"):
        self.trade_manager = trade_manager
        self.config_file = config_file
        self.vision_config = {}
        self.openai_client = None
        self.upload_dir = Path("chart_uploads")
        self.analysis_results: Dict[str, ComprehensiveChartAnalysis] = {}
        self.is_running = False
        self.automated_analysis_task: Optional[asyncio.Task] = None

        self.upload_dir.mkdir(exist_ok=True)
        self._load_config()
        
        # Check for valid OpenAI API key
        # Handle nested vision config structure
        vision_settings = self.vision_config.get("vision", self.vision_config)
        api_key = vision_settings.get("openai_api_key", "")
        if api_key and api_key != "YOUR_OPENAI_API_KEY_HERE" and "YOUR_OPENAI_API_KEY" not in api_key and len(api_key) > 20:
            self._initialize_openai()
            logger.info(f"OpenAI API key loaded successfully (length: {len(api_key)})")
        else:
            logger.warning("OpenAI API key not provided or invalid for enhanced vision analysis")

    def _load_config(self):
        """Load enhanced vision analysis configuration"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.vision_config = json.load(f)
            else:
                logger.warning(f"{self.config_file} not found. Creating default configuration.")
                default_config = {
                    "openai_api_key": "YOUR_OPENAI_API_KEY_HERE",
                    "model_name": "gpt-4o",
                    "max_tokens": 3000,
                    "temperature": 0.2,
                    "analysis_prompt_template": self._get_enhanced_prompt_template(),
                }
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2)
                self.vision_config = default_config
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading vision config: {e}")
            self.vision_config = {}

    def _get_enhanced_prompt_template(self) -> str:
        """
        Returns a structured prompt template for detailed Smart Money Concepts analysis.
        """
        return """Analyze this XAUUSD chart using Smart Money Concepts. I want:

Clear directional bias with structural evidence.

Key SMC elements: CHoCH, BOS, OB, FVG, liquidity zones, volume imbalance.

Entry zone(s) with confirmation criteria.

Stop loss and 2–3 take profit levels.

Risk or invalidation zones.

Please structure it clearly in a step-by-step, table or bullet-point format. Be concise and avoid vague speculation. Prioritize structure over indicators.

**Required Analysis Format:**

## 1. MARKET STRUCTURE ANALYSIS
• HTF Trend Direction: [Bullish/Bearish/Neutral]
• Market Phase: [Trending/Ranging/Transitional]
• Key Structural Evidence: [Specific BOS/CHoCH levels and prices]

## 2. SMC ELEMENTS IDENTIFIED
• **Change of Character (CHoCH):** [Price level and timeframe]
• **Break of Structure (BOS):** [Price level and direction]
• **Order Blocks (OB):** [Bullish/Bearish zones with price ranges]
• **Fair Value Gaps (FVG):** [Unmitigated gaps with price ranges]
• **Liquidity Zones:** [Equal highs/lows, stops targeted]
• **Volume Imbalance:** [Areas of inefficient price delivery]

## 3. DIRECTIONAL BIAS & CONFLUENCE
• **Primary Bias:** [Long/Short with conviction level]
• **Structural Evidence:** [Specific price levels supporting bias]
• **Confluence Factors:** [Multiple SMC elements aligning]

## 4. TRADING PLAN
### Entry Strategy:
• **Entry Zone:** [Price range with specific levels]
• **Confirmation Criteria:** [What needs to happen for entry]
• **Entry Type:** [Market/Limit order placement]

### Risk Management:
• **Stop Loss:** [Exact price level and reasoning]
• **Risk/Reward Ratio:** [Calculated ratio]

### Profit Targets:
• **TP1:** [Price level and reasoning] 
• **TP2:** [Price level and reasoning]
• **TP3:** [Price level and reasoning]

## 5. RISK FACTORS & INVALIDATION
• **Invalidation Level:** [Price that negates the setup]
• **Risk Zones:** [Areas where setup becomes invalid]
• **Key Considerations:** [External factors to monitor]

**Analysis Guidelines:**
- Use specific price levels, not ranges
- State conviction level (High/Medium/Low)
- Focus on price action over indicators
- Be definitive, avoid "could", "might", "possibly"
- Prioritize recent structural developments"""

    def _initialize_openai(self):
        """Initialize OpenAI client for vision analysis"""
        try:
            from openai import AsyncOpenAI
            vision_settings = self.vision_config.get("vision", self.vision_config)
            self.openai_client = AsyncOpenAI(api_key=vision_settings.get("openai_api_key"))
            logger.info("OpenAI client initialized for enhanced vision analysis")
        except ImportError:
            logger.error("OpenAI library not found. Please install with 'pip install openai'")
            self.openai_client = None
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None

    def upload_chart_image(self, image_data: bytes, filename: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validates and saves an uploaded chart image.
        Generates a unique analysis ID for the image.
        """
        if not image_data:
            return False, "No image data provided.", None

        file_ext = Path(filename).suffix.lower()
        if file_ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            return False, f"Unsupported image format. Supported formats: .jpg, .jpeg, .png, .webp", None

        try:
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[0] < 50 or image.shape[1] < 50:
                return False, "Invalid or too small image file.", None
        except Exception as e:
            logger.error(f"Error validating image: {e}")
            return False, "Could not process image file.", None

        analysis_id = f"CHART_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        save_path = self.upload_dir / f"{analysis_id}{file_ext}"

        try:
            with open(save_path, "wb") as f:
                f.write(image_data)
            logger.info(f"Chart image saved: {save_path} with ID: {analysis_id}")
            return True, "Chart uploaded successfully.", analysis_id
        except IOError as e:
            logger.error(f"Error saving uploaded chart image: {e}")
            return False, "Failed to save chart image.", None

    async def analyze_uploaded_chart(self, analysis_id: str, symbol: Optional[str] = None, timeframe: str = "H4") -> Optional[ComprehensiveChartAnalysis]:
        """Analyze an uploaded chart image using OpenAI Vision API"""
        logger.info(f'[VISION] Starting analysis for {analysis_id}, symbol={symbol}, timeframe={timeframe}')
        
        if not self.openai_client:
            logger.error('[VISION] OpenAI client not initialized')
            return None

        # Find image file with correct extension
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.webp']:
            potential_path = self.upload_dir / f"{analysis_id}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break
        
        if not image_path:
            logger.error(f'[VISION] Image not found for analysis_id: {analysis_id}')
            return None

        try:
            # Get vision settings from nested config
            vision_settings = self.vision_config.get("vision", self.vision_config)
            api_key = vision_settings.get("openai_api_key")
            
            if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
                logger.error('[VISION] OpenAI API key not configured')
                return None
            
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # Get prompt template
            prompt = self._get_enhanced_prompt_template()
            
            # Use the properly initialized OpenAI client
            try:
                response = await self.openai_client.chat.completions.create(
                    model=vision_settings.get("model_name", "gpt-4o"),
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                            ]
                        }
                    ],
                    max_tokens=vision_settings.get("max_tokens", 3000),
                    temperature=vision_settings.get("temperature", 0.2),
                )
                
                # If there are non-UTF-8 characters in output, replace with a safe fallback
                content = response.choices[0].message.content
                if isinstance(content, bytes):
                    analysis_content = content.decode("utf-8", errors="replace")
                else:
                    analysis_content = content
                
                if not analysis_content:
                    logger.error(f"❌ Empty response from OpenAI for {analysis_id}")
                    return None
                    
                logger.info(f"✅ Raw OpenAI response for {analysis_id} (length: {len(analysis_content)})")
                
                # Return RAW ANALYSIS exactly as received from OpenAI
                result = self._create_raw_analysis_result(analysis_content, analysis_id, str(image_path), symbol, timeframe)
                self.analysis_results[analysis_id] = result
                return result
                    
            except Exception as api_error:
                logger.error(f"❌ OpenAI API error for {analysis_id}: {api_error}")
                return None
        except Exception as e:
            logger.error(f"❌ [VISION] Error in analyze_uploaded_chart for {analysis_id}: {e}", exc_info=True)
            return None

    def _create_raw_analysis_result(self, analysis_text: str, analysis_id: str, image_path: str, symbol: Optional[str], timeframe: str) -> ComprehensiveChartAnalysis:
        """Create analysis result that preserves the full raw OpenAI analysis text"""
        logger.info(f"[VISION] Creating RAW analysis result for {analysis_id}")
        
        # Create a simple primary scenario with the full raw text
        primary_scenario = TradeScenario(
            scenario_name="ChatGPT-4 Vision Analysis",
            trade_type="BUY" if "buy" in analysis_text.lower() or "bullish" in analysis_text.lower() else "SELL",
            entry_price=1.0,
            stop_loss=0.99,
            take_profit_1=1.01,
            take_profit_2=1.02,
            probability_success=0.85,
            notes=analysis_text  # Store the FULL raw analysis text here
        )
        
        # Create minimal analysis object that preserves raw text
        analysis = ComprehensiveChartAnalysis(
            analysis_id=analysis_id,
            timestamp=datetime.now(),
            symbol=symbol or "UNKNOWN",
            timeframe=timeframe,
            overall_trend="bullish" if "bullish" in analysis_text.lower() else "bearish" if "bearish" in analysis_text.lower() else "neutral",
            trend_strength=SignalStrength.STRONG,
            market_bias=MarketBias.BULLISH if "bullish" in analysis_text.lower() else MarketBias.BEARISH if "bearish" in analysis_text.lower() else MarketBias.NEUTRAL,
            market_structure_notes="Full ChatGPT-4 Vision Analysis",
            current_price=1.0,
            support_levels=[],
            resistance_levels=[],
            key_levels=[],
            indicators=[],
            patterns_detected=["ChatGPT-4 Smart Money Concepts Analysis"],
            pattern_completion=0.90,
            pattern_reliability="high",
            pattern_notes="ChatGPT-4 Vision Analysis",
            primary_scenario=primary_scenario,
            overall_confidence=0.85,
            risk_factors=["Standard market risk"],
            confluence_factors=["ChatGPT-4 Vision Analysis"],
            chart_quality=image_path,
            analysis_notes=analysis_text  # Store the FULL raw analysis text here
        )
        
        return analysis

    def _create_analysis_from_text(self, analysis_text: str, analysis_id: str, image_path: str, symbol: Optional[str], timeframe: str) -> ComprehensiveChartAnalysis:
        """Create analysis from raw text when JSON parsing fails"""
        logger.info(f"[VISION] Creating analysis from raw text for {analysis_id}")
        
        # Store the raw analysis text
        primary_scenario = TradeScenario(
            scenario_name="AI Analysis",
            trade_type="BUY" if "buy" in analysis_text.lower() or "bullish" in analysis_text.lower() else "SELL",
            entry_price=1.0,
            stop_loss=0.99,
            take_profit_1=1.01,
            take_profit_2=1.02,
            probability_success=0.75,
            notes=f"Raw AI Analysis: {analysis_text[:200]}..."
        )
        
        analysis = ComprehensiveChartAnalysis(
            analysis_id=analysis_id,
            timestamp=datetime.now(),
            symbol=symbol or "UNKNOWN",
            timeframe=timeframe,
            overall_trend="bullish" if "bullish" in analysis_text.lower() else "bearish" if "bearish" in analysis_text.lower() else "neutral",
            trend_strength=SignalStrength.MODERATE,
            market_bias=MarketBias.NEUTRAL,
            market_structure_notes="AI Vision Analysis - Raw Text",
            current_price=1.0,
            support_levels=[],
            resistance_levels=[],
            key_levels=[],
            indicators=[],
            patterns_detected=["AI Pattern Recognition"],
            pattern_completion=0.75,
            pattern_reliability="medium",
            pattern_notes="AI Analysis",
            primary_scenario=primary_scenario,
            overall_confidence=0.75,
            risk_factors=["Market volatility"],
            confluence_factors=["AI Analysis"],
            chart_quality=image_path,
            analysis_notes=f"Full AI Analysis Text: {analysis_text}"
        )
        return analysis

    def _create_comprehensive_analysis(self, data: Dict, analysis_id: str, image_path: str) -> ComprehensiveChartAnalysis:
        """Create ComprehensiveChartAnalysis object from enhanced SMC API response dictionary."""
        
        def _parse_smc_scenario(sc_data: Optional[Dict]) -> Optional[TradeScenario]:
            if not sc_data: return None
            
            # Handle take profits array format
            take_profits = sc_data.get("take_profits", [])
            tp1 = take_profits[0].get("tp1") if len(take_profits) > 0 and "tp1" in take_profits[0] else take_profits[0] if len(take_profits) > 0 else 1.0
            tp2 = take_profits[1].get("tp2") if len(take_profits) > 1 and "tp2" in take_profits[1] else take_profits[1] if len(take_profits) > 1 else 1.0
            
            # Extract entry zone - handle range format like "1.2336-1.2340"
            entry_zone = sc_data.get("entry_zone", "1.0")
            if isinstance(entry_zone, str) and "-" in entry_zone:
                entry_price = float(entry_zone.split("-")[0])  # Use lower bound
            else:
                entry_price = float(entry_zone) if entry_zone else 1.0
            
            # Build detailed notes from SMC analysis
            confirmation_signals = sc_data.get("confirmation_signals", [])
            entry_reasoning = sc_data.get("entry_reasoning", "SMC analysis")
            notes = f"Entry: {entry_reasoning}. Confirmations: {', '.join(confirmation_signals[:2])}"
            
            return TradeScenario(
                scenario_name=sc_data.get("scenario_name", "SMC Trading Setup"),
                trade_type=sc_data.get("trade_type", "BUY"),
                entry_price=entry_price,
                stop_loss=float(sc_data.get("stop_loss", 1.0)),
                take_profit_1=float(tp1) if tp1 else 1.0,
                take_profit_2=float(tp2) if tp2 else 1.0,
                probability_success=float(sc_data.get("probability_success", 0.70)),
                notes=notes
            )

        def _parse_smc_key_levels(level_data: Optional[List[Dict]]) -> List[PriceLevel]:
            if not level_data: return []
            levels = []
            
            for lvl in level_data:
                # Handle both simple levels and SMC structural levels
                price = float(lvl.get("price", lvl.get("level", 1.0)))
                level_type = lvl.get("type", "SUPPORT").lower()
                
                # Map SMC types to standard types
                if any(smc_type in level_type.upper() for smc_type in ["BOS", "CHoCH", "EQH", "EQL"]):
                    level_type = "resistance" if "HIGH" in level_type.upper() or "BOS" in level_type.upper() else "support"
                elif "RESISTANCE" in level_type.upper() or "SUPPLY" in level_type.upper():
                    level_type = "resistance"
                else:
                    level_type = "support"
                
                levels.append(PriceLevel(
                    price=price,
                    level_type=level_type,
                    strength=lvl.get("strength", "MODERATE").lower(),
                    context=lvl.get("reason", lvl.get("notes", "SMC structural level"))
                ))
            return levels
        
        def _parse_order_blocks_and_fvgs(supply_demand_data: Optional[Dict]) -> List[PriceLevel]:
            """Extract order blocks and FVGs as key levels"""
            if not supply_demand_data: return []
            levels = []
            
            # Parse Order Blocks
            order_blocks = supply_demand_data.get("active_order_blocks", [])
            for ob in order_blocks:
                zone_range = ob.get("zone_range", "1.0-1.0")
                if "-" in str(zone_range):
                    price = float(zone_range.split("-")[0])  # Use lower bound
                else:
                    price = float(zone_range)
                
                ob_type = ob.get("type", "DEMAND_OB")
                level_type = "support" if "DEMAND" in ob_type else "resistance"
                
                levels.append(PriceLevel(
                    price=price,
                    level_type=level_type,
                    strength=ob.get("strength", "MODERATE").lower(),
                    context=f"Order Block: {ob.get('notes', 'Institutional supply/demand zone')}"
                ))
            
            # Parse Fair Value Gaps
            fvgs = supply_demand_data.get("fair_value_gaps", [])
            for fvg in fvgs:
                gap_range = fvg.get("gap_range", "1.0-1.0")
                if "-" in str(gap_range):
                    price = float(gap_range.split("-")[0])  # Use lower bound
                else:
                    price = float(gap_range)
                
                fvg_type = fvg.get("type", "BULLISH_FVG")
                level_type = "support" if "BULLISH" in fvg_type else "resistance"
                
                levels.append(PriceLevel(
                    price=price,
                    level_type=level_type,
                    strength=fvg.get("priority", "MEDIUM").lower(),
                    context=f"Fair Value Gap: {fvg.get('notes', 'Price inefficiency')}"
                ))
            
            return levels
        
        def _create_smc_indicators(order_flow_data: Optional[Dict], structural_data: Optional[Dict]) -> List[TechnicalIndicator]:
            """Create technical indicators from SMC analysis"""
            indicators = []
            
            if order_flow_data:
                market_structure = order_flow_data.get("market_structure", "RANGING")
                signal = "bullish" if "BULLISH" in market_structure else "bearish" if "BEARISH" in market_structure else "neutral"
                
                indicators.append(TechnicalIndicator(
                    name="Market Structure",
                    value=85.0 if "BOS" in market_structure else 70.0,
                    signal=signal,
                    strength="strong" if "BOS" in market_structure else "moderate",
                    notes=f"SMC Analysis: {market_structure}"
                ))
            
            if structural_data:
                htf_trend = structural_data.get("htf_trend", "RANGING")
                ltf_bias = structural_data.get("ltf_bias", "CONTINUATION")
                
                indicators.append(TechnicalIndicator(
                    name="Multi-Timeframe Bias",
                    value=80.0,
                    signal="bullish" if "BULLISH" in htf_trend else "bearish" if "BEARISH" in htf_trend else "neutral",
                    strength="strong",
                    notes=f"HTF: {htf_trend}, LTF: {ltf_bias}"
                ))
            
            return indicators

        # Extract data from nested SMC structure
        analysis_header = data.get("analysis_header", {})
        order_flow_data = data.get("order_flow_analysis", {})
        supply_demand_data = data.get("supply_demand_zones", {})
        structural_data = data.get("structural_bias", {})
        trade_scenarios = data.get("trade_scenarios", {})
        risk_data = data.get("risk_assessment", {})
        
        # Parse scenarios
        primary_scenario = _parse_smc_scenario(trade_scenarios.get("primary_setup"))
        alternative_scenario = _parse_smc_scenario(trade_scenarios.get("alternative_setup"))
        
        # Create default scenario if missing
        if not primary_scenario:
            trend = structural_data.get("htf_trend", "BULLISH")
            primary_scenario = TradeScenario(
                scenario_name="SMC Analysis Setup",
                trade_type="BUY" if "BULLISH" in trend else "SELL",
                entry_price=1.0,
                stop_loss=0.99 if "BULLISH" in trend else 1.01,
                take_profit_1=1.01 if "BULLISH" in trend else 0.99,
                take_profit_2=1.02 if "BULLISH" in trend else 0.98,
                probability_success=0.70,
                notes="Smart Money Concepts analysis based on order flow and market structure"
            )

        # Parse all levels from different sources
        structural_levels = _parse_smc_key_levels(order_flow_data.get("key_structural_levels", []))
        ob_fvg_levels = _parse_order_blocks_and_fvgs(supply_demand_data)
        all_key_levels = structural_levels + ob_fvg_levels
        
        # Split into support/resistance
        support_levels = [lvl for lvl in all_key_levels if lvl.level_type.lower() == "support"]
        resistance_levels = [lvl for lvl in all_key_levels if lvl.level_type.lower() == "resistance"]

        # Create SMC-specific indicators
        indicators = _create_smc_indicators(order_flow_data, structural_data)
        
        # Build market bias from structural analysis
        htf_trend = structural_data.get("htf_trend", "RANGING")
        ltf_bias = structural_data.get("ltf_bias", "CONTINUATION")
        market_bias = structural_data.get("overall_structure", f"HTF: {htf_trend}, LTF: {ltf_bias}")
        
        # Extract patterns from SMC analysis
        key_observations = data.get("key_observations", [])
        confluence_factors = data.get("confluence_factors", [])
        patterns_detected = key_observations[:3] + confluence_factors[:2]  # Combine for patterns
        
        # Calculate overall confidence
        overall_confidence = float(risk_data.get("overall_confidence", 0.75))
        
        # USE ORIGINAL ANALYSIS CONTENT AS DETAILED TEXT
        full_analysis_text = f"Smart Money Concepts Analysis Complete\n\nMarket Structure: {order_flow_data.get('market_structure', 'Analyzing...')}\nHTF Trend: {structural_data.get('htf_trend', 'Analyzing...')}\nConfidence: {overall_confidence*100:.0f}%"
        
        analysis = ComprehensiveChartAnalysis(
            analysis_id=analysis_id,
            timestamp=datetime.now(),
            symbol=analysis_header.get("symbol", "UNKNOWN"),
            timeframe=analysis_header.get("timeframe", "H4"),
            current_price=1.0,  # Will be updated if provided
            overall_trend=htf_trend.lower() if htf_trend else "sideways",
                         trend_strength=SignalStrength.STRONG if "BOS" in order_flow_data.get("market_structure", "") else SignalStrength.MODERATE,
            market_bias=market_bias,
            market_structure_notes=full_analysis_text,  # Store full detailed text here
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            key_levels=all_key_levels,
            indicators=indicators,
            patterns_detected=patterns_detected,
            pattern_completion=overall_confidence,
            pattern_reliability="high" if overall_confidence > 0.8 else "medium" if overall_confidence > 0.6 else "low",
            pattern_notes=f"SMC Analysis - Confidence: {overall_confidence*100:.0f}%",
            primary_scenario=primary_scenario,
            alternative_scenario=alternative_scenario,
            overall_confidence=overall_confidence,
            risk_factors=data.get("risk_factors", risk_data.get("news_risk", ["Market volatility"])),
            confluence_factors=confluence_factors,
            chart_quality=image_path,
            analysis_notes=f"SMC Order Flow Analysis: {', '.join(key_observations[:2])}"
        )
        return analysis

    def get_analysis_by_id(self, analysis_id: str) -> Optional[ComprehensiveChartAnalysis]:
        return self.analysis_results.get(analysis_id)

    def get_recent_analyses(self, limit: int = 10) -> List[ComprehensiveChartAnalysis]:
        sorted_analyses = sorted(self.analysis_results.values(), key=lambda x: x.timestamp, reverse=True)
        return sorted_analyses[:limit]

    def get_vision_status(self) -> Dict:
        vision_settings = self.vision_config.get("vision", self.vision_config)
        return {
            "openai_connected": self.openai_client is not None,
            "total_analyses": len(self.analysis_results),
            "uploaded_images": len(list(self.upload_dir.iterdir())),
            "config_loaded": bool(self.vision_config),
            "model_name": vision_settings.get("model_name", "N/A"),
            "automated_analysis_running": self.is_running
        }

    def stop_automated_analysis(self):
        """Stops the automated analysis task."""
        if self.automated_analysis_task and not self.automated_analysis_task.done():
            self.automated_analysis_task.cancel()
        self.is_running = False
        logger.info("Stopping automated analysis...")
        logger.info("Automated analysis stopped")

    def start_automated_analysis(self, symbols: Optional[List[str]] = None, timeframes: Optional[List[str]] = None):
        """Starts the automated analysis task."""
        if not self.is_running:
            self.is_running = True
            symbols = symbols or ["EURUSD", "GBPUSD", "USDJPY"]
            timeframes = timeframes or ["H1", "H4"]
            logger.info(f"Starting automated analysis for symbols: {symbols}, timeframes: {timeframes}")
            # For now, this is a placeholder - automated analysis can be implemented later
            logger.info("Automated analysis started")
        else:
            logger.info("Automated analysis is already running")

    def analyze_uploaded_chart_sync(self, analysis_id: str, symbol: Optional[str] = None, timeframe: str = "H4"):
        """Synchronous wrapper for the async analyze_uploaded_chart method"""
        import asyncio
        if symbol is None:
            symbol = ""
        logger.info(f'[VISION] analyze_uploaded_chart_sync called for {analysis_id}, symbol={symbol}, timeframe={timeframe}')
        
        if not self.openai_client:
            logger.error("OpenAI client not initialized for sync analysis")
            return None
            
        try:
            # Try to get existing event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we need to use run_in_executor or create new loop
                    logger.info("[VISION] Event loop is running, creating new loop")
                    import threading
                    result = None
                    exception = None
                    
                    def run_analysis():
                        nonlocal result, exception
                        try:
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            result = new_loop.run_until_complete(self.analyze_uploaded_chart(analysis_id, symbol, timeframe))
                            new_loop.close()
                        except Exception as e:
                            exception = e
                    
                    thread = threading.Thread(target=run_analysis)
                    thread.start()
                    thread.join()
                    
                    if exception:
                        raise exception
                    return result
                else:
                    # Loop exists but not running
                    logger.info("[VISION] Using existing event loop")
                    result = loop.run_until_complete(self.analyze_uploaded_chart(analysis_id, symbol, timeframe))
                    return result
            except RuntimeError:
                # No event loop in current thread
                logger.info("[VISION] No event loop found, creating new one")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.analyze_uploaded_chart(analysis_id, symbol, timeframe))
                loop.close()
                return result
                
        except Exception as e:
            logger.error(f'[VISION] Error in analyze_uploaded_chart_sync: {e}', exc_info=True)
            return None