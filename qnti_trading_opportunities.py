#!/usr/bin/env python3
"""
QNTI Trading Opportunities Management System
Aggregates and scores trading opportunities from all sources
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import threading

# Import centralized logging
from qnti_logging_utils import get_qnti_logger
logger = get_qnti_logger('QNTI_OPPORTUNITIES')

class OpportunityType(Enum):
    SMC_SIGNAL = "smc_signal"
    EA_STRATEGY = "ea_strategy"
    VISION_ANALYSIS = "vision_analysis"
    BACKTEST_RESULT = "backtest_result"
    MANUAL_ANALYSIS = "manual_analysis"

class OpportunityStatus(Enum):
    ACTIVE = "active"
    EXECUTED = "executed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class TradingOpportunity:
    """Standardized trading opportunity with probability scoring"""
    opportunity_id: str
    opportunity_type: OpportunityType
    symbol: str
    timeframe: str
    
    # Core trading data
    trade_direction: str  # BUY, SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    
    # Probability and scoring
    probability_score: float  # 0.0 to 1.0
    confidence_level: float   # 0.0 to 1.0
    risk_level: RiskLevel
    
    # Market context
    market_conditions: Dict[str, Any]
    supporting_factors: List[str]
    risk_factors: List[str]
    
    # Metadata
    created_at: datetime
    expires_at: Optional[datetime]
    status: OpportunityStatus
    source_data: Dict[str, Any]
    
    # Performance tracking
    win_rate_historical: Optional[float] = None
    profit_factor_historical: Optional[float] = None
    max_drawdown_historical: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['opportunity_type'] = self.opportunity_type.value
        result['status'] = self.status.value
        result['risk_level'] = self.risk_level.value
        result['created_at'] = self.created_at.isoformat()
        if self.expires_at:
            result['expires_at'] = self.expires_at.isoformat()
        return result
    
    @property
    def is_expired(self) -> bool:
        """Check if opportunity has expired"""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at
    
    @property
    def time_to_expiry(self) -> Optional[timedelta]:
        """Time remaining until expiry"""
        if not self.expires_at:
            return None
        return self.expires_at - datetime.now()

class QNTITradingOpportunitiesManager:
    """
    Centralized manager for all trading opportunities
    Aggregates, scores, and ranks opportunities from multiple sources
    """
    
    def __init__(self, qnti_system):
        self.qnti_system = qnti_system
        self.opportunities: Dict[str, TradingOpportunity] = {}
        self.historical_performance: Dict[str, Dict] = {}
        
        # Scoring weights for different factors
        self.scoring_weights = {
            'historical_performance': 0.25,
            'market_conditions': 0.20,
            'technical_confluence': 0.20,
            'risk_reward': 0.15,
            'timeframe_validity': 0.10,
            'source_reliability': 0.10
        }
        
        # Initialize monitoring thread
        self.is_monitoring = False
        self._monitor_thread = None
        
        logger.info("QNTI Trading Opportunities Manager initialized")
    
    def start_monitoring(self):
        """Start monitoring for new opportunities"""
        if self.is_monitoring:
            logger.warning("Opportunity monitoring already running")
            return
        
        self.is_monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("📊 Started trading opportunities monitoring")
    
    def stop_monitoring(self):
        """Stop monitoring for opportunities"""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("⏹️ Stopped trading opportunities monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                self._update_opportunities()
                self._cleanup_expired_opportunities()
                asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                asyncio.sleep(60)  # Wait longer on error
    
    def _update_opportunities(self):
        """Update opportunities from all sources"""
        try:
            # Update from SMC automation
            self._update_smc_opportunities()
            
            # Update from EA strategies
            self._update_ea_opportunities()
            
            # Update from vision analysis
            self._update_vision_opportunities()
            
            # Update from backtest results
            self._update_backtest_opportunities()
            
            logger.debug(f"Updated opportunities: {len(self.opportunities)} active")
            
        except Exception as e:
            logger.error(f"Error updating opportunities: {e}")
    
    def _update_smc_opportunities(self):
        """Extract opportunities from SMC automation"""
        try:
            # Check if SMC automation is available and running
            if not hasattr(self.qnti_system, 'smc_automation') or not self.qnti_system.smc_automation:
                logger.debug("SMC automation not available")
                return
            
            smc_automation = self.qnti_system.smc_automation
            if not hasattr(smc_automation, 'active_signals'):
                logger.debug("SMC automation has no active_signals attribute")
                return
            
            # Get active signals from SMC automation
            active_signals = smc_automation.active_signals
            if not active_signals:
                logger.debug("No active SMC signals found")
                return
            
            # Process each active signal
            for signal_id, signal in active_signals.items():
                opportunity_id = f"smc_{signal_id}"
                
                # Check if we already have this opportunity
                if opportunity_id not in self.opportunities:
                    # Create new opportunity from SMC signal
                    opportunity = self._create_smc_opportunity(signal)
                    if opportunity:
                        self.opportunities[opportunity_id] = opportunity
                        logger.info(f"📈 New SMC opportunity: {signal.symbol} {signal.signal_type.value}")
                else:
                    # Update existing opportunity
                    existing_opportunity = self.opportunities[opportunity_id]
                    # Update confidence and other dynamic fields
                    existing_opportunity.confidence_level = signal.confidence
                    existing_opportunity.probability_score = self._calculate_smc_probability(signal)
                
        except Exception as e:
            logger.error(f"Error updating SMC opportunities: {e}")
    
    def _create_smc_opportunity(self, smc_signal) -> Optional[TradingOpportunity]:
        """Create trading opportunity from SMC signal"""
        try:
            # Calculate probability score based on SMC factors
            probability_score = self._calculate_smc_probability(smc_signal)
            
            # Determine trade direction
            signal_type_str = smc_signal.signal_type.value if hasattr(smc_signal.signal_type, 'value') else str(smc_signal.signal_type)
            trade_direction = "BUY" if "buy" in signal_type_str.lower() else "SELL"
            
            # Calculate risk level
            risk_level = self._calculate_risk_level(smc_signal.risk_reward_ratio, smc_signal.confidence)
            
            # Get market conditions from SMC signal
            market_conditions = self._get_smc_market_conditions(smc_signal)
            
            # Supporting and risk factors
            supporting_factors = self._get_smc_supporting_factors(smc_signal)
            risk_factors = self._get_smc_risk_factors(smc_signal)
            
            # Calculate expiry time (SMC signals typically valid for 6-24 hours)
            expires_at = smc_signal.timestamp + timedelta(hours=12)
            
            opportunity = TradingOpportunity(
                opportunity_id=f"smc_{smc_signal.signal_id}",
                opportunity_type=OpportunityType.SMC_SIGNAL,
                symbol=smc_signal.symbol,
                timeframe=smc_signal.timeframe,
                trade_direction=trade_direction,
                entry_price=float(smc_signal.entry_price),
                stop_loss=float(smc_signal.stop_loss),
                take_profit=float(smc_signal.take_profit),
                risk_reward_ratio=float(smc_signal.risk_reward_ratio),
                probability_score=probability_score,
                confidence_level=float(smc_signal.confidence),
                risk_level=risk_level,
                market_conditions=market_conditions,
                supporting_factors=supporting_factors,
                risk_factors=risk_factors,
                created_at=smc_signal.timestamp,
                expires_at=expires_at,
                status=OpportunityStatus.ACTIVE,
                source_data=smc_signal.to_dict() if hasattr(smc_signal, 'to_dict') else {}
            )
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error creating SMC opportunity: {e}")
            return None
    
    def _update_ea_opportunities(self):
        """Extract opportunities from EA strategies"""
        try:
            if not hasattr(self.qnti_system, 'trade_manager'):
                return
            
            trade_manager = self.qnti_system.trade_manager
            if not hasattr(trade_manager, 'ea_profiles'):
                return
            
            for ea_name, ea_data in trade_manager.ea_profiles.items():
                opportunity_id = f"ea_{ea_name}"
                
                # Only create opportunity if EA is active and performing well
                if ea_data.get('status') == 'active':
                    performance = ea_data.get('performance', {})
                    
                    # Check if EA meets minimum performance criteria
                    if self._ea_meets_criteria(performance):
                        if opportunity_id not in self.opportunities:
                            opportunity = self._create_ea_opportunity(ea_name, ea_data)
                            if opportunity:
                                self.opportunities[opportunity_id] = opportunity
                                logger.info(f"🤖 New EA opportunity: {ea_name}")
                
        except Exception as e:
            logger.error(f"Error updating EA opportunities: {e}")
    
    def _create_ea_opportunity(self, ea_name: str, ea_data: Dict) -> Optional[TradingOpportunity]:
        """Create trading opportunity from EA strategy"""
        try:
            performance = ea_data.get('performance', {})
            
            # Calculate probability based on EA performance metrics
            probability_score = self._calculate_ea_probability(performance)
            
            # Get EA configuration
            symbols = ea_data.get('symbols', ['EURUSD'])
            timeframes = ea_data.get('timeframes', ['H1'])
            
            # Estimate trade parameters from EA profile
            entry_price = 1.0  # Placeholder - would need current market price
            stop_loss = 0.99   # Based on EA's typical stop loss
            take_profit = 1.02 # Based on EA's typical take profit
            
            risk_reward_ratio = performance.get('profit_factor', 1.0)
            confidence_level = performance.get('win_rate', 50.0) / 100.0
            
            risk_level = self._calculate_risk_level(risk_reward_ratio, confidence_level)
            
            # Market conditions for primary symbol
            market_conditions = self._get_current_market_conditions(symbols[0])
            
            # Supporting factors from EA analysis
            supporting_factors = self._get_ea_supporting_factors(ea_data)
            risk_factors = self._get_ea_risk_factors(ea_data)
            
            opportunity = TradingOpportunity(
                opportunity_id=f"ea_{ea_name}",
                opportunity_type=OpportunityType.EA_STRATEGY,
                symbol=symbols[0],
                timeframe=timeframes[0],
                trade_direction="BUY",  # Would need EA's current signal
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                probability_score=probability_score,
                confidence_level=confidence_level,
                risk_level=risk_level,
                market_conditions=market_conditions,
                supporting_factors=supporting_factors,
                risk_factors=risk_factors,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=7),  # EA opportunities last longer
                status=OpportunityStatus.ACTIVE,
                source_data=ea_data,
                win_rate_historical=performance.get('win_rate'),
                profit_factor_historical=performance.get('profit_factor'),
                max_drawdown_historical=performance.get('max_drawdown')
            )
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error creating EA opportunity: {e}")
            return None
    
    def _update_vision_opportunities(self):
        """Extract opportunities from vision analysis"""
        try:
            if not hasattr(self.qnti_system, 'vision_analyzer'):
                return
            
            vision_analyzer = self.qnti_system.vision_analyzer
            if not hasattr(vision_analyzer, 'recent_analyses'):
                return
            
            # Get recent vision analyses
            for analysis_id, analysis in vision_analyzer.recent_analyses.items():
                opportunity_id = f"vision_{analysis_id}"
                
                if opportunity_id not in self.opportunities and hasattr(analysis, 'primary_scenario'):
                    opportunity = self._create_vision_opportunity(analysis)
                    if opportunity:
                        self.opportunities[opportunity_id] = opportunity
                        logger.info(f"👁️ New Vision opportunity: {analysis.symbol} {analysis.primary_scenario.trade_type}")
                
        except Exception as e:
            logger.error(f"Error updating vision opportunities: {e}")
    
    def _create_vision_opportunity(self, analysis) -> Optional[TradingOpportunity]:
        """Create trading opportunity from vision analysis"""
        try:
            scenario = analysis.primary_scenario
            
            # Use the probability from vision analysis
            probability_score = scenario.probability_success
            confidence_level = analysis.overall_confidence
            
            risk_level = self._calculate_risk_level(scenario.risk_reward_ratio, confidence_level)
            
            # Market conditions
            market_conditions = self._get_current_market_conditions(analysis.symbol)
            
            # Extract supporting factors from analysis
            supporting_factors = analysis.confluence_factors + [f"Pattern: {', '.join(analysis.patterns_detected)}"]
            risk_factors = analysis.risk_factors
            
            opportunity = TradingOpportunity(
                opportunity_id=f"vision_{analysis.analysis_id}",
                opportunity_type=OpportunityType.VISION_ANALYSIS,
                symbol=analysis.symbol,
                timeframe=scenario.time_frame_validity,
                trade_direction=scenario.trade_type,
                entry_price=scenario.entry_price,
                stop_loss=scenario.stop_loss,
                take_profit=scenario.take_profit_1,
                risk_reward_ratio=scenario.risk_reward_ratio,
                probability_score=probability_score,
                confidence_level=confidence_level,
                risk_level=risk_level,
                market_conditions=market_conditions,
                supporting_factors=supporting_factors,
                risk_factors=risk_factors,
                created_at=analysis.timestamp,
                expires_at=analysis.timestamp + timedelta(hours=12),  # Vision analysis expires in 12h
                status=OpportunityStatus.ACTIVE,
                source_data=asdict(analysis)
            )
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error creating vision opportunity: {e}")
            return None
    
    def _update_backtest_opportunities(self):
        """Extract opportunities from backtest results"""
        # Implementation would depend on backtest results storage
        pass
    
    # Scoring and calculation methods
    def _calculate_smc_probability(self, smc_signal) -> float:
        """Calculate probability score for SMC signal"""
        try:
            base_score = smc_signal.confidence
            
            # Adjust based on signal type strength
            signal_type_multiplier = {
                'structure_break_buy': 0.9,
                'structure_break_sell': 0.9,
                'order_block_buy': 0.8,
                'order_block_sell': 0.8,
                'fvg_fill_buy': 0.7,
                'fvg_fill_sell': 0.7,
                'buy_zone_entry': 0.6,
                'sell_zone_entry': 0.6
            }
            
            multiplier = signal_type_multiplier.get(smc_signal.signal_type.value, 0.5)
            
            # Adjust based on risk-reward ratio
            rr_bonus = min(smc_signal.risk_reward_ratio * 0.1, 0.2)
            
            final_score = min(base_score * multiplier + rr_bonus, 1.0)
            return round(final_score, 3)
            
        except Exception as e:
            logger.error(f"Error calculating SMC probability: {e}")
            return 0.5
    
    def _calculate_ea_probability(self, performance: Dict) -> float:
        """Calculate probability score for EA strategy"""
        try:
            win_rate = performance.get('win_rate', 50.0) / 100.0
            profit_factor = performance.get('profit_factor', 1.0)
            max_drawdown = performance.get('max_drawdown', 50.0)
            
            # Weighted scoring
            win_rate_score = win_rate
            pf_score = min(profit_factor / 3.0, 1.0)  # Normalize profit factor
            dd_score = max(1.0 - (max_drawdown / 50.0), 0.0)  # Penalty for high drawdown
            
            final_score = (win_rate_score * 0.4 + pf_score * 0.4 + dd_score * 0.2)
            return round(min(final_score, 1.0), 3)
            
        except Exception as e:
            logger.error(f"Error calculating EA probability: {e}")
            return 0.5
    
    def _calculate_risk_level(self, risk_reward_ratio: float, confidence: float) -> RiskLevel:
        """Calculate risk level based on RR ratio and confidence"""
        try:
            # Higher RR ratio = lower risk, higher confidence = lower risk
            risk_score = (1.0 / max(risk_reward_ratio, 0.5)) * (1.0 - confidence)
            
            if risk_score < 0.3:
                return RiskLevel.LOW
            elif risk_score < 0.6:
                return RiskLevel.MEDIUM
            elif risk_score < 0.8:
                return RiskLevel.HIGH
            else:
                return RiskLevel.EXTREME
                
        except Exception as e:
            logger.error(f"Error calculating risk level: {e}")
            return RiskLevel.MEDIUM
    
    def _get_current_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Get current market conditions for symbol"""
        try:
            # Get market data from MT5 bridge or other sources
            conditions = {
                'volatility': 'normal',
                'trend': 'ranging',
                'session': 'london',
                'spread': 'normal',
                'last_updated': datetime.now().isoformat()
            }
            
            # Add specific conditions from market analysis
            if hasattr(self.qnti_system, 'intelligent_ea_manager'):
                market_condition = self.qnti_system.intelligent_ea_manager.analyze_current_market(symbol)
                conditions.update({
                    'volatility': getattr(market_condition, 'volatility', 'normal'),
                    'session': getattr(market_condition, 'session', 'unknown'),
                    'spread_level': getattr(market_condition, 'spread_level', 'normal')
                })
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error getting market conditions: {e}")
            return {'error': str(e)}
    
    def _get_smc_market_conditions(self, smc_signal) -> Dict[str, Any]:
        """Get market conditions from SMC signal"""
        try:
            conditions = {
                'signal_type': smc_signal.signal_type.value if hasattr(smc_signal.signal_type, 'value') else str(smc_signal.signal_type),
                'alert_level': smc_signal.alert_level.value if hasattr(smc_signal.alert_level, 'value') else str(smc_signal.alert_level),
                'zone_info': smc_signal.zone_info if hasattr(smc_signal, 'zone_info') else {},
                'timeframe': smc_signal.timeframe,
                'last_updated': datetime.now().isoformat()
            }
            
            # Add additional market data if available
            if hasattr(smc_signal, 'additional_data') and smc_signal.additional_data:
                market_structure = smc_signal.additional_data.get('market_structure', 'ranging')
                conditions['market_structure'] = market_structure
                conditions['volatility'] = smc_signal.additional_data.get('volatility', 'normal')
                conditions['trend'] = smc_signal.additional_data.get('current_trend', 'sideways')
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error getting SMC market conditions: {e}")
            return {'error': str(e)}
    
    def _get_smc_supporting_factors(self, smc_signal) -> List[str]:
        """Get supporting factors for SMC signal"""
        try:
            factors = []
            
            # Signal type specific factors
            signal_type = smc_signal.signal_type.value if hasattr(smc_signal.signal_type, 'value') else str(smc_signal.signal_type)
            factors.append(f"SMC Signal: {signal_type.replace('_', ' ').title()}")
            
            # Alert level
            alert_level = smc_signal.alert_level.value if hasattr(smc_signal.alert_level, 'value') else str(smc_signal.alert_level)
            factors.append(f"Alert Level: {alert_level.upper()}")
            
            # Confidence level
            confidence_pct = int(smc_signal.confidence * 100)
            if confidence_pct >= 80:
                factors.append(f"High Confidence: {confidence_pct}%")
            elif confidence_pct >= 70:
                factors.append(f"Good Confidence: {confidence_pct}%")
            
            # Risk-reward ratio
            if smc_signal.risk_reward_ratio >= 2.0:
                factors.append(f"Excellent R:R Ratio: 1:{smc_signal.risk_reward_ratio:.1f}")
            elif smc_signal.risk_reward_ratio >= 1.5:
                factors.append(f"Good R:R Ratio: 1:{smc_signal.risk_reward_ratio:.1f}")
            
            # Enhanced HTF zone information
            if hasattr(smc_signal, 'zone_info') and smc_signal.zone_info:
                # HTF signal name
                signal_name = smc_signal.zone_info.get('signal_name', '')
                if signal_name and 'HTF' in signal_name:
                    factors.append(f"📊 {signal_name.replace('_', ' ')}")
                
                # HTF bias
                htf_bias = smc_signal.zone_info.get('htf_bias', '')
                if htf_bias and htf_bias != 'neutral':
                    factors.append(f"🔄 HTF Bias: {htf_bias.title()}")
                
                # Structure type
                structure_type = smc_signal.zone_info.get('structure_type', '')
                if structure_type:
                    factors.append(f"🏗️ Structure: {structure_type.replace('_', ' ').title()}")
                
                # HTF zones
                htf_zones = smc_signal.zone_info.get('htf_zones', {})
                if htf_zones:
                    demand_zones = htf_zones.get('demand', [])
                    supply_zones = htf_zones.get('supply', [])
                    if demand_zones:
                        factors.append(f"🟢 HTF Demand Zones: {len(demand_zones)}")
                    if supply_zones:
                        factors.append(f"🔴 HTF Supply Zones: {len(supply_zones)}")
                
                # Liquidity sweeps
                liquidity_sweeps = smc_signal.zone_info.get('liquidity_sweeps', [])
                swept_liquidity = [sweep for sweep in liquidity_sweeps if sweep.get('swept', False)]
                if swept_liquidity:
                    factors.append(f"💧 Liquidity Taken: {len(swept_liquidity)} levels")
                
                # Structure events
                structure_events = smc_signal.zone_info.get('structure_events', {})
                if structure_events:
                    recent_choch = structure_events.get('recent_choch')
                    previous_bos = structure_events.get('previous_bos')
                    if recent_choch:
                        factors.append(f"🔄 Recent CHoCH: {recent_choch.get('level', 'N/A')}")
                    if previous_bos:
                        factors.append(f"🔥 Previous BOS: {previous_bos.get('level', 'N/A')}")
            
            # Enhanced market structure from additional data
            if hasattr(smc_signal, 'additional_data') and smc_signal.additional_data:
                htf_analysis = smc_signal.additional_data.get('htf_analysis', {})
                if htf_analysis:
                    # Retracement context
                    retracement_ctx = htf_analysis.get('retracement_context', {})
                    if retracement_ctx:
                        setup_quality = retracement_ctx.get('setup_quality', '')
                        if setup_quality:
                            factors.append(f"⭐ Setup Quality: {setup_quality.title()}")
                        
                        if retracement_ctx.get('inside_htf_structure'):
                            factors.append("🎯 Inside HTF Structure")
                        
                        if retracement_ctx.get('in_demand_zone'):
                            factors.append("🟢 Price in HTF Demand Zone")
                        
                        if retracement_ctx.get('in_supply_zone'):
                            factors.append("🔴 Price in HTF Supply Zone")
                        
                        if retracement_ctx.get('liquidity_taken'):
                            factors.append("💧 Liquidity Already Taken")
                
                # Signal reasoning
                signal_reasoning = smc_signal.additional_data.get('signal_reasoning', '')
                if signal_reasoning and 'HTF' in signal_reasoning:
                    factors.append(f"📋 {signal_reasoning}")
                
                # Legacy order blocks (fallback)
                order_blocks = smc_signal.additional_data.get('order_blocks', [])
                if len(order_blocks) > 0:
                    factors.append(f"📦 Active Order Blocks: {len(order_blocks)}")
                
                # Fair value gaps
                fvgs = smc_signal.additional_data.get('fair_value_gaps', [])
                if len(fvgs) > 0:
                    factors.append(f"Fair Value Gaps: {len(fvgs)}")
            
            return factors[:6]  # Limit to top 6 factors
            
        except Exception as e:
            logger.error(f"Error getting SMC supporting factors: {e}")
            return [f"SMC Signal: {smc_signal.symbol}"]
    
    def _get_smc_risk_factors(self, smc_signal) -> List[str]:
        """Get risk factors for SMC signal"""
        try:
            risk_factors = []
            
            # Low confidence warning
            if smc_signal.confidence < 0.7:
                risk_factors.append(f"Lower confidence signal: {int(smc_signal.confidence * 100)}%")
            
            # Poor risk-reward ratio
            if smc_signal.risk_reward_ratio < 1.5:
                risk_factors.append(f"Below optimal R:R: 1:{smc_signal.risk_reward_ratio:.1f}")
            
            # Critical alert level
            alert_level = smc_signal.alert_level.value if hasattr(smc_signal.alert_level, 'value') else str(smc_signal.alert_level)
            if alert_level.lower() == 'critical':
                risk_factors.append("Critical market conditions")
            
            # Market volatility warnings
            if hasattr(smc_signal, 'additional_data') and smc_signal.additional_data:
                volatility = smc_signal.additional_data.get('volatility', 1.0)
                if isinstance(volatility, (int, float)) and volatility > 1.8:
                    risk_factors.append("High market volatility")
                elif isinstance(volatility, str) and volatility == 'high':
                    risk_factors.append("High market volatility")
            
            # Time-based risk
            signal_age = (datetime.now() - smc_signal.timestamp).total_seconds() / 3600  # hours
            if signal_age > 6:
                risk_factors.append(f"Signal aging: {signal_age:.1f} hours old")
            
            # News/session risk (simplified)
            current_hour = datetime.now().hour
            if 13 <= current_hour <= 15:  # News hours (example)
                risk_factors.append("Active news/session period")
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Error getting SMC risk factors: {e}")
            return []
    
    def _get_ea_supporting_factors(self, ea_data: Dict) -> List[str]:
        """Get supporting factors for EA strategy"""
        factors = []
        
        performance = ea_data.get('performance', {})
        
        if performance.get('win_rate', 0) > 60:
            factors.append(f"High win rate: {performance.get('win_rate'):.1f}%")
        
        if performance.get('profit_factor', 0) > 1.5:
            factors.append(f"Good profit factor: {performance.get('profit_factor'):.2f}")
        
        factors.append(f"Strategy type: {ea_data.get('strategy_type', 'unknown')}")
        
        return factors
    
    def _get_ea_risk_factors(self, ea_data: Dict) -> List[str]:
        """Get risk factors for EA strategy"""
        risk_factors = []
        
        performance = ea_data.get('performance', {})
        
        if performance.get('max_drawdown', 0) > 20:
            risk_factors.append(f"High drawdown: {performance.get('max_drawdown'):.1f}%")
        
        if performance.get('win_rate', 100) < 40:
            risk_factors.append(f"Low win rate: {performance.get('win_rate'):.1f}%")
        
        return risk_factors
    
    def _ea_meets_criteria(self, performance: Dict) -> bool:
        """Check if EA meets minimum performance criteria"""
        try:
            win_rate = performance.get('win_rate', 0)
            profit_factor = performance.get('profit_factor', 0)
            max_drawdown = performance.get('max_drawdown', 100)
            
            # Minimum criteria
            return (win_rate > 45 and 
                   profit_factor > 1.2 and 
                   max_drawdown < 30)
                   
        except Exception as e:
            logger.error(f"Error checking EA criteria: {e}")
            return False
    
    def _cleanup_expired_opportunities(self):
        """Remove expired opportunities"""
        try:
            expired_ids = []
            for opp_id, opportunity in self.opportunities.items():
                if opportunity.is_expired:
                    expired_ids.append(opp_id)
                    opportunity.status = OpportunityStatus.EXPIRED
            
            for opp_id in expired_ids:
                del self.opportunities[opp_id]
                logger.debug(f"Removed expired opportunity: {opp_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up expired opportunities: {e}")
    
    # Public API methods
    def get_all_opportunities(self, include_expired: bool = False) -> List[Dict[str, Any]]:
        """Get all trading opportunities"""
        try:
            opportunities = []
            
            for opportunity in self.opportunities.values():
                if include_expired or opportunity.status == OpportunityStatus.ACTIVE:
                    opportunities.append(opportunity.to_dict())
            
            # Sort by probability score (highest first)
            opportunities.sort(key=lambda x: x.get('probability_score', 0), reverse=True)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error getting opportunities: {e}")
            return []
    
    def get_opportunities_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Get opportunities for specific symbol"""
        all_opportunities = self.get_all_opportunities()
        return [opp for opp in all_opportunities if opp['symbol'] == symbol]
    
    def get_opportunities_by_type(self, opportunity_type: OpportunityType) -> List[Dict[str, Any]]:
        """Get opportunities by type"""
        all_opportunities = self.get_all_opportunities()
        return [opp for opp in all_opportunities if opp['opportunity_type'] == opportunity_type.value]
    
    def get_top_opportunities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top N opportunities by probability score"""
        all_opportunities = self.get_all_opportunities()
        return all_opportunities[:limit]
    
    def get_opportunity_summary(self) -> Dict[str, Any]:
        """Get summary statistics of opportunities"""
        try:
            opportunities = self.get_all_opportunities()
            
            if not opportunities:
                return {
                    'total_opportunities': 0,
                    'by_type': {},
                    'by_risk_level': {},
                    'avg_probability': 0.0,
                    'symbols': []
                }
            
            # Count by type
            by_type = {}
            for opp in opportunities:
                opp_type = opp['opportunity_type']
                by_type[opp_type] = by_type.get(opp_type, 0) + 1
            
            # Count by risk level
            by_risk = {}
            for opp in opportunities:
                risk_level = opp['risk_level']
                by_risk[risk_level] = by_risk.get(risk_level, 0) + 1
            
            # Calculate average probability
            avg_prob = sum(opp['probability_score'] for opp in opportunities) / len(opportunities)
            
            # Get unique symbols
            symbols = list(set(opp['symbol'] for opp in opportunities))
            
            return {
                'total_opportunities': len(opportunities),
                'by_type': by_type,
                'by_risk_level': by_risk,
                'avg_probability': round(avg_prob, 3),
                'symbols': symbols,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting opportunity summary: {e}")
            return {'error': str(e)}

# Global instance
_opportunities_manager = None

def get_opportunities_manager(qnti_system) -> QNTITradingOpportunitiesManager:
    """Get or create the global opportunities manager"""
    global _opportunities_manager
    if _opportunities_manager is None:
        _opportunities_manager = QNTITradingOpportunitiesManager(qnti_system)
    return _opportunities_manager 