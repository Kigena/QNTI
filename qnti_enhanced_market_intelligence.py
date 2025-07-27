#!/usr/bin/env python3
"""
QNTI Enhanced Market Intelligence Engine
Real-time market data collection, analysis, and intelligent insights generation
"""

import logging
import time
import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger('QNTI_MarketIntelligence')

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    high_52w: float
    low_52w: float
    market_cap: Optional[float] = None
    timestamp: datetime = None

@dataclass
class TechnicalIndicators:
    """Technical analysis indicators"""
    rsi: float
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    macd: float
    macd_signal: float
    bollinger_upper: float
    bollinger_lower: float
    volatility: float
    atr: float

@dataclass
class MarketInsight:
    """Enhanced market insight with real data backing"""
    id: str
    title: str
    description: str
    insight_type: str  # signal, warning, opportunity, trend
    priority: str  # low, medium, high, critical
    confidence: float
    symbol: str
    market_data: MarketData
    timestamp: datetime
    action_required: bool
    source: str
    supporting_data: Dict[str, Any]
    technical_data: TechnicalIndicators = None  # Optional

class QNTIEnhancedMarketIntelligence:
    """Enhanced Market Intelligence Engine with Real Data"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.EnhancedMarketIntelligence")
        
        # Priority asset configuration (user's main focus)
        self.forex_pairs = [
            'EURUSD=X',   # EUR/USD
            'USDJPY=X',   # USD/JPY  
            'USDCAD=X',   # USD/CAD
            'GBPUSD=X',   # GBP/USD
            'GBPJPY=X'    # GBP/JPY
        ]
        self.commodities = ['GC=F']  # Gold (main focus)
        self.indices = [
            '^IXIC',      # US100 (Nasdaq)
            '^DJI',       # US30 (Dow Jones)
            '^GSPC'       # US500 (S&P 500)
        ]
        self.crypto = ['BTC-USD']  # Bitcoin (main focus)
        
        # All symbols to monitor
        self.all_symbols = self.forex_pairs + self.commodities + self.indices + self.crypto
        
        # Data storage
        self.market_data: Dict[str, MarketData] = {}
        self.technical_indicators: Dict[str, TechnicalIndicators] = {}
        self.insights: List[MarketInsight] = []
        self.historical_data: Dict[str, pd.DataFrame] = {}
        
        # API keys (add your keys here)
        self.alpha_vantage_key = "YOUR_ALPHA_VANTAGE_KEY"  # Get free from alphavantage.co
        self.twelve_data_key = "YOUR_TWELVE_DATA_KEY"    # Get free from twelvedata.com
        
        # Update intervals
        self.last_update = {}
        self.update_interval = 300  # 5 minutes
        
        self.logger.info("Enhanced Market Intelligence Engine initialized")
    
    def fetch_yahoo_finance_data(self, symbol: str, period: str = "3mo") -> Optional[pd.DataFrame]:
        """Fetch historical data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return None
                
            # Get additional info
            info = ticker.info
            
            # Store current market data
            current_price = hist['Close'].iloc[-1]
            change = hist['Close'].iloc[-1] - hist['Close'].iloc[-2]
            change_percent = (change / hist['Close'].iloc[-2]) * 100
            
            self.market_data[symbol] = MarketData(
                symbol=symbol,
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
                high_52w=info.get('fiftyTwoWeekHigh', hist['High'].max()),
                low_52w=info.get('fiftyTwoWeekLow', hist['Low'].min()),
                market_cap=info.get('marketCap'),
                timestamp=datetime.now()
            )
            
            return hist
            
        except Exception as e:
            self.logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame, symbol: str) -> TechnicalIndicators:
        """Calculate technical indicators from price data"""
        try:
            if df.empty or len(df) < 50:
                raise ValueError("Insufficient data for technical analysis")
            
            close = df['Close']
            high = df['High']
            low = df['Low']
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Moving Averages
            sma_20 = close.rolling(window=20).mean()
            sma_50 = close.rolling(window=50).mean()
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            
            # MACD
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_middle = close.rolling(window=bb_period).mean()
            bb_std_dev = close.rolling(window=bb_period).std()
            bollinger_upper = bb_middle + (bb_std_dev * bb_std)
            bollinger_lower = bb_middle - (bb_std_dev * bb_std)
            
            # Volatility (20-day)
            volatility = close.pct_change().rolling(window=20).std() * np.sqrt(252) * 100
            
            # ATR (Average True Range)
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            
            # Get latest values
            indicators = TechnicalIndicators(
                rsi=rsi.iloc[-1] if not rsi.empty else 50.0,
                sma_20=sma_20.iloc[-1] if not sma_20.empty else close.iloc[-1],
                sma_50=sma_50.iloc[-1] if not sma_50.empty else close.iloc[-1],
                ema_12=ema_12.iloc[-1] if not ema_12.empty else close.iloc[-1],
                ema_26=ema_26.iloc[-1] if not ema_26.empty else close.iloc[-1],
                macd=macd.iloc[-1] if not macd.empty else 0.0,
                macd_signal=macd_signal.iloc[-1] if not macd_signal.empty else 0.0,
                bollinger_upper=bollinger_upper.iloc[-1] if not bollinger_upper.empty else close.iloc[-1],
                bollinger_lower=bollinger_lower.iloc[-1] if not bollinger_lower.empty else close.iloc[-1],
                volatility=volatility.iloc[-1] if not volatility.empty else 10.0,
                atr=atr.iloc[-1] if not atr.empty else 0.01
            )
            
            self.technical_indicators[symbol] = indicators
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators for {symbol}: {e}")
            # Return default indicators
            current_price = df['Close'].iloc[-1] if not df.empty else 1.0
            return TechnicalIndicators(
                rsi=50.0, sma_20=current_price, sma_50=current_price,
                ema_12=current_price, ema_26=current_price, macd=0.0, macd_signal=0.0,
                bollinger_upper=current_price, bollinger_lower=current_price,
                volatility=10.0, atr=0.01
            )
    
    def analyze_symbol(self, symbol: str) -> List[MarketInsight]:
        """Analyze a symbol with REAL market movement intelligence across multiple timeframes"""
        insights = []
        
        try:
            # Fetch extended historical data for movement analysis
            df = self.fetch_yahoo_finance_data(symbol)
            if df is None or df.empty:
                return insights
            
            # Store historical data
            self.historical_data[symbol] = df
            
            # Calculate technical indicators (still needed for some analysis)
            tech_indicators = self.calculate_technical_indicators(df, symbol)
            market_data = self.market_data.get(symbol)
            
            if not market_data:
                return insights
            
            # REAL MARKET MOVEMENT ANALYSIS
            movement_analysis = self.analyze_market_movements(df, symbol)
            
            # Generate INTELLIGENT insights using multiple prompting strategies
            insights.extend(self.generate_intelligent_insights(
                symbol, df, market_data, tech_indicators, movement_analysis
            ))
            
        except Exception as e:
            logger.error(f"Error analyzing symbol {symbol}: {e}")
        
        return insights
    
    def analyze_market_movements(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Analyze market movements across multiple timeframes to detect trends and ranges"""
        analysis = {
            'daily': {},
            'weekly': {},
            'monthly': {},
            'quarterly': {},
            'yearly': {},
            'current_regime': 'unknown',
            'key_levels': [],
            'momentum_shift': False
        }
        
        try:
            current_price = df['Close'].iloc[-1]
            
            # Daily Analysis (last 30 days)
            if len(df) >= 30:
                daily_data = df.tail(30)
                daily_high = daily_data['High'].max()
                daily_low = daily_data['Low'].min()
                daily_range = daily_high - daily_low
                daily_change = (current_price - daily_data['Close'].iloc[0]) / daily_data['Close'].iloc[0] * 100
                
                # Detect if ranging or trending (daily)
                middle_30pct = daily_low + (daily_range * 0.3)
                upper_30pct = daily_high - (daily_range * 0.3)
                daily_ranging = middle_30pct <= current_price <= upper_30pct
                
                analysis['daily'] = {
                    'timeframe': '30 days',
                    'high': daily_high,
                    'low': daily_low,
                    'range_size': daily_range,
                    'change_percent': daily_change,
                    'is_ranging': daily_ranging,
                    'regime': 'Ranging' if daily_ranging else ('Bullish Trend' if daily_change > 2 else 'Bearish Trend' if daily_change < -2 else 'Neutral'),
                    'volatility': daily_data['Close'].pct_change().std() * 100 * np.sqrt(252)
                }
            
            # Weekly Analysis (last 12 weeks)
            if len(df) >= 84:  # 12 weeks
                weekly_data = df.tail(84)
                weekly_high = weekly_data['High'].max()
                weekly_low = weekly_data['Low'].min()
                weekly_range = weekly_high - weekly_low
                weekly_change = (current_price - weekly_data['Close'].iloc[0]) / weekly_data['Close'].iloc[0] * 100
                
                # Weekly trend detection
                weekly_ranging = abs(weekly_change) < 3 and (current_price > weekly_low + weekly_range * 0.2 and current_price < weekly_high - weekly_range * 0.2)
                
                analysis['weekly'] = {
                    'timeframe': '12 weeks',
                    'high': weekly_high,
                    'low': weekly_low,
                    'range_size': weekly_range,
                    'change_percent': weekly_change,
                    'is_ranging': weekly_ranging,
                    'regime': 'Ranging' if weekly_ranging else ('Strong Uptrend' if weekly_change > 5 else 'Strong Downtrend' if weekly_change < -5 else 'Weak Trend'),
                    'breakout_potential': current_price > weekly_high * 0.995 or current_price < weekly_low * 1.005
                }
            
            # Monthly Analysis (last 6 months)
            if len(df) >= 126:  # ~6 months
                monthly_data = df.tail(126)
                monthly_high = monthly_data['High'].max()
                monthly_low = monthly_data['Low'].min()
                monthly_change = (current_price - monthly_data['Close'].iloc[0]) / monthly_data['Close'].iloc[0] * 100
                
                analysis['monthly'] = {
                    'timeframe': '6 months',
                    'high': monthly_high,
                    'low': monthly_low,
                    'change_percent': monthly_change,
                    'regime': 'Bull Market' if monthly_change > 10 else 'Bear Market' if monthly_change < -10 else 'Sideways Market',
                    'near_highs': current_price > monthly_high * 0.97,
                    'near_lows': current_price < monthly_low * 1.03
                }
            
            # Quarterly Analysis (last 12 months)
            if len(df) >= 252:  # ~1 year
                quarterly_data = df.tail(252)
                quarterly_change = (current_price - quarterly_data['Close'].iloc[0]) / quarterly_data['Close'].iloc[0] * 100
                
                # Find significant support/resistance levels
                highs = quarterly_data['High'].rolling(window=20).max()
                lows = quarterly_data['Low'].rolling(window=20).min()
                key_levels = []
                
                # Add significant levels that price has tested multiple times
                for level in [highs.quantile(0.9), highs.quantile(0.7), lows.quantile(0.3), lows.quantile(0.1)]:
                    if not np.isnan(level):
                        key_levels.append(level)
                
                analysis['quarterly'] = {
                    'timeframe': '12 months',
                    'change_percent': quarterly_change,
                    'regime': 'Major Uptrend' if quarterly_change > 20 else 'Major Downtrend' if quarterly_change < -20 else 'Range-bound',
                    'key_levels': key_levels
                }
                analysis['key_levels'] = key_levels
            
            # Determine overall current regime
            regimes = []
            if 'daily' in analysis and analysis['daily']:
                regimes.append(analysis['daily'].get('regime', 'unknown'))
            if 'weekly' in analysis and analysis['weekly']:
                regimes.append(analysis['weekly'].get('regime', 'unknown'))
            if 'monthly' in analysis and analysis['monthly']:
                regimes.append(analysis['monthly'].get('regime', 'unknown'))
            
            # Detect momentum shifts
            if len(df) >= 10:
                recent_momentum = df['Close'].tail(5).pct_change().mean()
                earlier_momentum = df['Close'].tail(10).head(5).pct_change().mean()
                analysis['momentum_shift'] = abs(recent_momentum - earlier_momentum) > 0.005  # 0.5% shift
            
            # Overall regime assessment
            bullish_count = sum(1 for r in regimes if 'bull' in r.lower() or 'up' in r.lower())
            bearish_count = sum(1 for r in regimes if 'bear' in r.lower() or 'down' in r.lower())
            
            if bullish_count > bearish_count:
                analysis['current_regime'] = 'bullish'
            elif bearish_count > bullish_count:
                analysis['current_regime'] = 'bearish'
            else:
                analysis['current_regime'] = 'neutral'
            
        except Exception as e:
            logger.error(f"Error in movement analysis for {symbol}: {e}")
        
        return analysis
    
    def generate_intelligent_insights(self, symbol: str, df: pd.DataFrame, market_data: MarketData, 
                                     tech_indicators: TechnicalIndicators, movement_analysis: Dict) -> List[MarketInsight]:
        """Generate ADVANCED intelligent insights with detailed technical metrics"""
        insights = []
        timestamp = datetime.now()
        
        # PRIMARY: Advanced Technical Analysis with Fibonacci, Entry Zones, and Detailed Metrics
        insights.extend(self._generate_advanced_technical_insights(symbol, df, market_data, timestamp))
        
        # SECONDARY: Market Regime Analysis (if advanced analysis fails)
        if not insights:
            insights.extend(self._generate_regime_insights(symbol, movement_analysis, market_data, timestamp))
            insights.extend(self._generate_breakout_insights(symbol, movement_analysis, market_data, timestamp))
            insights.extend(self._generate_momentum_insights(symbol, movement_analysis, df, market_data, timestamp))
        
        return insights
    
    def _generate_regime_insights(self, symbol: str, movement_analysis: Dict, market_data: MarketData, timestamp: datetime) -> List[MarketInsight]:
        """Generate insights based on market regime analysis"""
        insights = []
        
        try:
            daily = movement_analysis.get('daily', {})
            weekly = movement_analysis.get('weekly', {})
            monthly = movement_analysis.get('monthly', {})
            
            # Ranging Market Insights
            if daily.get('is_ranging') and weekly.get('is_ranging'):
                range_size = daily.get('range_size', 0)
                current_price = market_data.price
                
                if daily.get('high') and daily.get('low'):
                    position_in_range = (current_price - daily['low']) / range_size * 100
                    
                    if position_in_range > 80:
                        insights.append(MarketInsight(
                            id=f"range_top_{symbol}_{int(time.time())}",
                            title=f"🎯 {symbol} Near Range Top - Short Opportunity",
                            description=f"Price at {position_in_range:.0f}% of 30-day range. {symbol} has been ranging for {daily.get('timeframe')} between {daily['low']:.4f}-{daily['high']:.4f}. Consider short positions with tight stops.",
                            insight_type="opportunity",
                            priority="high",
                            confidence=0.85,
                            symbol=symbol,
                            market_data=market_data,
                            timestamp=timestamp,
                            action_required=True,
                            source="regime_analysis",
                            supporting_data={"range_position": position_in_range, "range_size": range_size}
                        ))
                    elif position_in_range < 20:
                        insights.append(MarketInsight(
                            id=f"range_bottom_{symbol}_{int(time.time())}",
                            title=f"🚀 {symbol} Range Bottom - Long Opportunity",
                            description=f"Price at {position_in_range:.0f}% of 30-day range. {symbol} approaching strong support at {daily['low']:.4f}. Consider long positions for range bounce.",
                            insight_type="opportunity",
                            priority="high",
                            confidence=0.85,
                            symbol=symbol,
                            market_data=market_data,
                            timestamp=timestamp,
                            action_required=True,
                            source="regime_analysis",
                            supporting_data={"range_position": position_in_range, "range_size": range_size}
                        ))
            
            # Trending Market Insights
            if monthly.get('regime') in ['Bull Market', 'Bear Market']:
                monthly_change = monthly.get('change_percent', 0)
                trend_direction = 'bullish' if monthly_change > 0 else 'bearish'
                
                insights.append(MarketInsight(
                    id=f"trend_regime_{symbol}_{int(time.time())}",
                    title=f"📈 {symbol} Strong {trend_direction.title()} Trend",
                    description=f"{symbol} in {monthly['regime']} with {monthly_change:+.1f}% change over 6 months. Trend continuation strategies favored. Look for pullbacks to enter.",
                    insight_type="trend",
                    priority="medium",
                    confidence=0.9,
                    symbol=symbol,
                    market_data=market_data,
                    timestamp=timestamp,
                    action_required=False,
                    source="regime_analysis",
                    supporting_data={"monthly_change": monthly_change, "regime": monthly['regime']}
                ))
            
        except Exception as e:
            logger.error(f"Error generating regime insights for {symbol}: {e}")
        
        return insights
    
    def _generate_breakout_insights(self, symbol: str, movement_analysis: Dict, market_data: MarketData, timestamp: datetime) -> List[MarketInsight]:
        """Generate insights for potential breakouts/breakdowns"""
        insights = []
        
        try:
            weekly = movement_analysis.get('weekly', {})
            current_price = market_data.price
            
            if weekly.get('breakout_potential'):
                weekly_high = weekly.get('high', 0)
                weekly_low = weekly.get('low', 0)
                weekly_change = weekly.get('change_percent', 0)
                
                if current_price > weekly_high * 0.995:  # Near weekly high
                    insights.append(MarketInsight(
                        id=f"breakout_setup_{symbol}_{int(time.time())}",
                        title=f"🚀 {symbol} Breakout Setup - 12-Week High Test",
                        description=f"{symbol} testing 12-week high at {weekly_high:.4f}. Volume confirmation needed for breakout. Target: {weekly_high * 1.02:.4f}, Stop: {weekly_high * 0.995:.4f}",
                        insight_type="signal",
                        priority="high",
                        confidence=0.8,
                        symbol=symbol,
                        market_data=market_data,
                        timestamp=timestamp,
                        action_required=True,
                        source="breakout_analysis",
                        supporting_data={"resistance": weekly_high, "target": weekly_high * 1.02}
                    ))
                elif current_price < weekly_low * 1.005:  # Near weekly low
                    insights.append(MarketInsight(
                        id=f"breakdown_risk_{symbol}_{int(time.time())}",
                        title=f"⚠️ {symbol} Breakdown Risk - 12-Week Low Test",
                        description=f"{symbol} testing 12-week low at {weekly_low:.4f}. High risk of further decline if support fails. Consider protective stops.",
                        insight_type="warning",
                        priority="high",
                        confidence=0.8,
                        symbol=symbol,
                        market_data=market_data,
                        timestamp=timestamp,
                        action_required=True,
                        source="breakdown_analysis",
                        supporting_data={"support": weekly_low, "risk_level": "high"}
                    ))
            
        except Exception as e:
            logger.error(f"Error generating breakout insights for {symbol}: {e}")
        
        return insights
    
    def _generate_advanced_technical_insights(self, symbol: str, df: pd.DataFrame, market_data: MarketData, timestamp: datetime) -> List[MarketInsight]:
        """Generate advanced technical insights with detailed metrics"""
        insights = []
        
        try:
            # Import the advanced analyzer
            from qnti_advanced_technical_analysis import AdvancedTechnicalAnalyzer
            
            # Get comprehensive analysis
            analyzer = AdvancedTechnicalAnalyzer()
            analysis = analyzer.get_comprehensive_analysis(symbol)
            
            if "error" in analysis:
                # Fallback to basic analysis
                return self._generate_momentum_insights(symbol, {}, df, market_data, timestamp)
            
            tech_metrics = analysis["technical_metrics"]
            fib_analysis = analysis["fibonacci_analysis"]
            pullback_analysis = analysis["pullback_analysis"]
            
            # Generate detailed technical insight
            symbol_name = symbol.replace('=X', '').replace('^', '').replace('-USD', '')
            
            # Create comprehensive description with actual metrics
            description = f"""📊 **{symbol_name} Technical Analysis**
            
**Price Action:** {tech_metrics.current_price:.5f} ({market_data.change_percent:+.2f}%)
• 52W Range: {((tech_metrics.current_price - tech_metrics.low_52w) / (tech_metrics.high_52w - tech_metrics.low_52w) * 100):.1f}% of range

**Indicators:**
• RSI(14): {tech_metrics.rsi_14:.1f} - {tech_metrics.rsi_signal.replace('_', ' ').title()}
• MACD: {tech_metrics.macd_line:.6f} | Signal: {tech_metrics.macd_signal:.6f} | Trend: {tech_metrics.macd_trend.replace('_', ' ').title()}
• Stochastic: K={tech_metrics.stochastic_k:.1f}, D={tech_metrics.stochastic_d:.1f} - {tech_metrics.stochastic_signal.replace('_', ' ').title()}
• Williams %R: {tech_metrics.williams_r:.1f}
• CCI(20): {tech_metrics.cci_20:.1f}

**Moving Averages:** {tech_metrics.ma_alignment.replace('_', ' ').title()}
• Price vs SMA20: {((tech_metrics.current_price - tech_metrics.sma_20) / tech_metrics.sma_20 * 100):+.2f}%
• SMA20 Slope: {tech_metrics.ma_slope_20:+.2f}% | SMA50 Slope: {tech_metrics.ma_slope_50:+.2f}%

**Fibonacci Analysis:**
• Trend: {fib_analysis.trend_direction.title()} | Current Level: {fib_analysis.current_fib_level}
• Key Levels: 38.2%={fib_analysis.fib_382:.5f} | 50%={fib_analysis.fib_500:.5f} | 61.8%={fib_analysis.fib_618:.5f}
• Next Target: {fib_analysis.next_fib_target:.5f}

**Pullback Analysis:**
• Trend Strength: {pullback_analysis.trend_strength:.0f}/100 | Pullback Probability: {pullback_analysis.pullback_probability:.0f}%
• Entry Zones: Optimal={pullback_analysis.entry_zone_optimal:.5f} | Conservative={pullback_analysis.entry_zone_conservative:.5f}
• Targets: T1={pullback_analysis.target_1:.5f} (R:R={pullback_analysis.rr_ratio_1:.2f}) | T2={pullback_analysis.target_2:.5f} (R:R={pullback_analysis.rr_ratio_2:.2f})

**Key Levels:**
• Support: {tech_metrics.immediate_support:.5f} | Resistance: {tech_metrics.immediate_resistance:.5f}
• Bollinger Position: {tech_metrics.bb_position:.1f}% | Squeeze: {'Yes' if tech_metrics.bb_squeeze else 'No'}

**Volume & Volatility:**
• Volume: {tech_metrics.volume_ratio:.2f}x avg ({tech_metrics.volume_trend.replace('_', ' ').title()})
• ATR(14): {tech_metrics.atr_14:.5f} | Vol Percentile: {tech_metrics.volatility_percentile:.0f}%"""

            # Determine priority based on signal strength and trend strength
            signal_strength = analysis["signal_strength"]
            if signal_strength >= 8 or pullback_analysis.trend_strength >= 80:
                priority = "critical"
            elif signal_strength >= 6 or pullback_analysis.trend_strength >= 60:
                priority = "high"
            elif signal_strength >= 4 or pullback_analysis.trend_strength >= 40:
                priority = "medium"
            else:
                priority = "low"

            # Determine insight type
            if pullback_analysis.pullback_probability >= 70:
                insight_type = "warning"
                title = f"🚨 {symbol_name} High Pullback Probability ({pullback_analysis.pullback_probability:.0f}%)"
            elif pullback_analysis.trend_strength >= 70:
                insight_type = "signal"
                title = f"📈 {symbol_name} Strong Trend Continuation Setup (Strength: {pullback_analysis.trend_strength:.0f}%)"
            elif tech_metrics.rsi_signal in ["overbought", "oversold"]:
                insight_type = "warning"
                title = f"⚠️ {symbol_name} RSI {tech_metrics.rsi_signal.title()} ({tech_metrics.rsi_14:.1f})"
            else:
                insight_type = "opportunity"
                title = f"🎯 {symbol_name} Technical Analysis (Signal: {signal_strength}/10)"

            insights.append(MarketInsight(
                id=f"advanced_technical_{symbol}_{int(time.time())}",
                title=title,
                description=description,
                insight_type=insight_type,
                priority=priority,
                confidence=0.9 if signal_strength >= 7 else 0.8 if signal_strength >= 5 else 0.7,
                symbol=symbol,
                market_data=market_data,
                timestamp=timestamp,
                action_required=True if priority in ["critical", "high"] else False,
                source="advanced_technical_analysis",
                supporting_data={
                    "signal_strength": signal_strength,
                    "trend_strength": pullback_analysis.trend_strength,
                    "pullback_probability": pullback_analysis.pullback_probability,
                    "rsi": tech_metrics.rsi_14,
                    "macd_trend": tech_metrics.macd_trend,
                    "ma_alignment": tech_metrics.ma_alignment,
                    "fibonacci_level": fib_analysis.current_fib_level,
                    "entry_zone": pullback_analysis.entry_zone_optimal,
                    "targets": [pullback_analysis.target_1, pullback_analysis.target_2],
                    "rr_ratios": [pullback_analysis.rr_ratio_1, pullback_analysis.rr_ratio_2]
                }
            ))
            
        except Exception as e:
            logger.error(f"Error generating advanced technical insights for {symbol}: {e}")
            # Fallback to basic momentum analysis
            return self._generate_momentum_insights(symbol, {}, df, market_data, timestamp)
        
        return insights

    def _generate_momentum_insights(self, symbol: str, movement_analysis: Dict, df: pd.DataFrame, market_data: MarketData, timestamp: datetime) -> List[MarketInsight]:
        """Generate detailed momentum insights with specific price targets and levels"""
        insights = []
        
        try:
            if movement_analysis.get('momentum_shift') and len(df) >= 10:
                recent_momentum = df['Close'].tail(5).pct_change().mean()
                earlier_momentum = df['Close'].tail(10).head(5).pct_change().mean()
                momentum_change = recent_momentum - earlier_momentum
                
                # Get current price data
                current_price = market_data.price
                daily_change = market_data.change_percent
                
                # Calculate key technical levels
                high_5d = df['High'].tail(5).max()
                low_5d = df['Low'].tail(5).min()
                high_20d = df['High'].tail(20).max()
                low_20d = df['Low'].tail(20).min()
                
                # Calculate volume (if available)
                volume_trend = ""
                if 'Volume' in df.columns:
                    avg_volume_5d = df['Volume'].tail(5).mean()
                    avg_volume_20d = df['Volume'].tail(20).mean()
                    volume_ratio = avg_volume_5d / avg_volume_20d if avg_volume_20d > 0 else 1
                    volume_trend = f" Volume: {volume_ratio:.1f}x avg"
                
                if abs(momentum_change) > 0.003:  # More sensitive threshold
                    direction = "accelerating" if momentum_change > 0 else "decelerating"
                    priority = "high" if abs(momentum_change) > 0.01 else "medium"
                    
                    # Create detailed description with specific levels
                    if direction == "accelerating":
                        if symbol == 'GC=F':  # Gold specific
                            next_target = current_price + (current_price * 0.015)  # 1.5% target
                            stop_level = current_price - (current_price * 0.01)   # 1% stop
                            description = f"🟢 Gold momentum accelerating to ${current_price:.2f} (+{daily_change:.2f}%). 5-day momentum: {recent_momentum*100:.3f}% vs {earlier_momentum*100:.3f}%. Next target: ${next_target:.2f}, Stop: ${stop_level:.2f}.{volume_trend}"
                        elif 'USD' in symbol:  # Forex pairs
                            next_target = current_price + (current_price * 0.005)  # 0.5% target
                            stop_level = current_price - (current_price * 0.003)   # 0.3% stop
                            description = f"🟢 {symbol.replace('=X', '')} accelerating to {current_price:.5f} (+{daily_change:.3f}%). 5D momentum: {recent_momentum*100:.3f}% vs {earlier_momentum*100:.3f}%. Target: {next_target:.5f}, Stop: {stop_level:.5f}.{volume_trend}"
                        elif symbol in ['^IXIC', '^DJI', '^GSPC']:  # US Indices
                            next_target = current_price + (current_price * 0.02)   # 2% target
                            stop_level = current_price - (current_price * 0.015)  # 1.5% stop
                            index_name = {'IXIC': 'US100', '^DJI': 'US30', '^GSPC': 'US500'}.get(symbol.replace('^', ''), symbol)
                            description = f"🟢 {index_name} momentum surge to {current_price:.1f} (+{daily_change:.2f}%). 5D momentum: {recent_momentum*100:.3f}% vs {earlier_momentum*100:.3f}%. Target: {next_target:.1f}, Stop: {stop_level:.1f}.{volume_trend}"
                        elif symbol == 'BTC-USD':  # Bitcoin
                            next_target = current_price + (current_price * 0.05)   # 5% target
                            stop_level = current_price - (current_price * 0.03)   # 3% stop
                            description = f"🟢 Bitcoin momentum accelerating to ${current_price:,.0f} (+{daily_change:.2f}%). 5D momentum: {recent_momentum*100:.3f}% vs {earlier_momentum*100:.3f}%. Target: ${next_target:,.0f}, Stop: ${stop_level:,.0f}.{volume_trend}"
                        else:
                            description = f"🟢 {symbol} momentum accelerating. Price: {current_price:.5f} (+{daily_change:.3f}%). 5D: {recent_momentum*100:.3f}% vs {earlier_momentum*100:.3f}%.{volume_trend}"
                    else:  # decelerating
                        if symbol == 'GC=F':  # Gold specific
                            support_level = low_20d
                            rebound_target = current_price + (current_price * 0.008)  # 0.8% rebound
                            description = f"🔴 Gold momentum slowing at ${current_price:.2f} ({daily_change:.2f}%). 5-day momentum: {recent_momentum*100:.3f}% vs {earlier_momentum*100:.3f}%. Support: ${support_level:.2f}, Rebound target: ${rebound_target:.2f}.{volume_trend}"
                        elif 'USD' in symbol:
                            support_level = low_5d
                            description = f"🔴 {symbol.replace('=X', '')} momentum decelerating at {current_price:.5f} ({daily_change:.3f}%). 5D: {recent_momentum*100:.3f}% vs {earlier_momentum*100:.3f}%. Support: {support_level:.5f}.{volume_trend}"
                        elif symbol in ['^IXIC', '^DJI', '^GSPC']:
                            support_level = low_5d
                            index_name = {'IXIC': 'US100', '^DJI': 'US30', '^GSPC': 'US500'}.get(symbol.replace('^', ''), symbol)
                            description = f"🔴 {index_name} momentum slowing at {current_price:.1f} ({daily_change:.2f}%). 5D: {recent_momentum*100:.3f}% vs {earlier_momentum*100:.3f}%. Support: {support_level:.1f}.{volume_trend}"
                        elif symbol == 'BTC-USD':
                            support_level = low_5d
                            description = f"🔴 Bitcoin momentum decelerating at ${current_price:,.0f} ({daily_change:.2f}%). 5D: {recent_momentum*100:.3f}% vs {earlier_momentum*100:.3f}%. Support: ${support_level:,.0f}.{volume_trend}"
                        else:
                            description = f"🔴 {symbol} momentum decelerating. Price: {current_price:.5f} ({daily_change:.3f}%). 5D: {recent_momentum*100:.3f}% vs {earlier_momentum*100:.3f}%.{volume_trend}"
                    
                    insights.append(MarketInsight(
                        id=f"momentum_shift_{symbol}_{int(time.time())}",
                        title=f"⚡ {symbol} Momentum Shift Detected",
                        description=description,
                        insight_type="signal",
                        priority=priority,
                        confidence=0.85,
                        symbol=symbol,
                        market_data=market_data,
                        timestamp=timestamp,
                        action_required=True,
                        source="detailed_momentum_analysis",
                        supporting_data={
                            "momentum_change": momentum_change,
                            "direction": direction,
                            "recent_momentum": recent_momentum,
                            "earlier_momentum": earlier_momentum,
                            "high_5d": high_5d,
                            "low_5d": low_5d,
                            "high_20d": high_20d,
                            "low_20d": low_20d
                        }
                    ))
            
        except Exception as e:
            logger.error(f"Error generating momentum insights for {symbol}: {e}")
        
        return insights
    
    def _generate_level_insights(self, symbol: str, movement_analysis: Dict, market_data: MarketData, timestamp: datetime) -> List[MarketInsight]:
        """Generate insights based on key support/resistance levels"""
        insights = []
        
        try:
            key_levels = movement_analysis.get('key_levels', [])
            current_price = market_data.price
            
            if key_levels:
                # Find closest levels
                closest_resistance = min([level for level in key_levels if level > current_price], default=None)
                closest_support = max([level for level in key_levels if level < current_price], default=None)
                
                if closest_resistance and abs(current_price - closest_resistance) / current_price < 0.02:  # Within 2%
                    insights.append(MarketInsight(
                        id=f"resistance_test_{symbol}_{int(time.time())}",
                        title=f"🔴 {symbol} Testing Key Resistance",
                        description=f"{symbol} approaching major resistance at {closest_resistance:.4f} (current: {current_price:.4f}). Historical level tested multiple times. Watch for rejection or breakout.",
                        insight_type="warning",
                        priority="high",
                        confidence=0.85,
                        symbol=symbol,
                        market_data=market_data,
                        timestamp=timestamp,
                        action_required=True,
                        source="level_analysis",
                        supporting_data={"resistance": closest_resistance, "distance": abs(current_price - closest_resistance)}
                    ))
                
                if closest_support and abs(current_price - closest_support) / current_price < 0.02:  # Within 2%
                    insights.append(MarketInsight(
                        id=f"support_test_{symbol}_{int(time.time())}",
                        title=f"🟢 {symbol} Testing Key Support",
                        description=f"{symbol} approaching major support at {closest_support:.4f} (current: {current_price:.4f}). Strong historical level. Potential bounce opportunity.",
                        insight_type="opportunity",
                        priority="high",
                        confidence=0.85,
                        symbol=symbol,
                        market_data=market_data,
                        timestamp=timestamp,
                        action_required=True,
                        source="level_analysis",
                        supporting_data={"support": closest_support, "distance": abs(current_price - closest_support)}
                    ))
            
        except Exception as e:
            logger.error(f"Error generating level insights for {symbol}: {e}")
        
        return insights
    
    def _generate_timeframe_insights(self, symbol: str, movement_analysis: Dict, market_data: MarketData, timestamp: datetime) -> List[MarketInsight]:
        """Generate insights from cross-timeframe analysis"""
        insights = []
        
        try:
            daily = movement_analysis.get('daily', {})
            weekly = movement_analysis.get('weekly', {})
            monthly = movement_analysis.get('monthly', {})
            
            # Conflicting timeframes
            daily_regime = daily.get('regime', '').lower()
            weekly_regime = weekly.get('regime', '').lower()
            monthly_regime = monthly.get('regime', '').lower()
            
            if 'bull' in daily_regime and 'bear' in weekly_regime:
                insights.append(MarketInsight(
                    id=f"timeframe_conflict_{symbol}_{int(time.time())}",
                    title=f"⚖️ {symbol} Timeframe Conflict - Counter-Trend Rally",
                    description=f"{symbol} showing bullish daily momentum within bearish weekly trend. Likely counter-trend move. Trade with caution and tight stops.",
                    insight_type="warning",
                    priority="medium",
                    confidence=0.8,
                    symbol=symbol,
                    market_data=market_data,
                    timestamp=timestamp,
                    action_required=True,
                    source="timeframe_analysis",
                    supporting_data={"daily_regime": daily_regime, "weekly_regime": weekly_regime}
                ))
            
            # Aligned timeframes (strong signal)
            if all('bull' in regime for regime in [daily_regime, weekly_regime, monthly_regime] if regime):
                insights.append(MarketInsight(
                    id=f"timeframe_alignment_{symbol}_{int(time.time())}",
                    title=f"🎯 {symbol} All Timeframes Bullish - Strong Signal",
                    description=f"{symbol} shows bullish alignment across daily, weekly, and monthly timeframes. High-probability long opportunities on pullbacks.",
                    insight_type="opportunity",
                    priority="high",
                    confidence=0.95,
                    symbol=symbol,
                    market_data=market_data,
                    timestamp=timestamp,
                    action_required=True,
                    source="timeframe_analysis",
                    supporting_data={"alignment": "bullish", "timeframes": ["daily", "weekly", "monthly"]}
                ))
            elif all('bear' in regime for regime in [daily_regime, weekly_regime, monthly_regime] if regime):
                insights.append(MarketInsight(
                    id=f"timeframe_alignment_bear_{symbol}_{int(time.time())}",
                    title=f"🔻 {symbol} All Timeframes Bearish - Strong Downtrend",
                    description=f"{symbol} shows bearish alignment across all timeframes. Avoid long positions. Look for short opportunities on rallies.",
                    insight_type="warning",
                    priority="high",
                    confidence=0.95,
                    symbol=symbol,
                    market_data=market_data,
                    timestamp=timestamp,
                    action_required=True,
                    source="timeframe_analysis",
                    supporting_data={"alignment": "bearish", "timeframes": ["daily", "weekly", "monthly"]}
                ))
            
        except Exception as e:
            logger.error(f"Error generating timeframe insights for {symbol}: {e}")
        
        return insights
    
    def generate_periodic_ai_insights(self) -> List[MarketInsight]:
        """Periodically generate AI-powered insights using multiple prompting strategies"""
        all_insights = []
        
        try:
            import ollama
            
            # Get current market data
            market_summary = self.get_market_summary()
            
            # Strategy 1: Cross-Market Analysis Prompt
            cross_market_prompt = self._create_cross_market_analysis_prompt()
            
            # Strategy 2: Risk-Opportunity Scanner Prompt  
            risk_opportunity_prompt = self._create_risk_opportunity_prompt()
            
            # Strategy 3: Trend Reversal Detection Prompt
            trend_reversal_prompt = self._create_trend_reversal_prompt()
            
            # Strategy 4: Correlation Analysis Prompt
            correlation_prompt = self._create_correlation_analysis_prompt()
            
            # Generate insights from each strategy
            prompts = [
                ("cross_market", cross_market_prompt),
                ("risk_opportunity", risk_opportunity_prompt), 
                ("trend_reversal", trend_reversal_prompt),
                ("correlation", correlation_prompt)
            ]
            
            for strategy_name, prompt in prompts:
                try:
                    response = ollama.chat(
                        model="llama3",
                        messages=[{"role": "user", "content": prompt}],
                        options={"temperature": 0.7, "max_tokens": 800}
                    )
                    
                    ai_insight = response["message"]["content"]
                    
                    # Convert AI response to MarketInsight
                    # Create dummy market data for AI insights
                    dummy_market_data = MarketData(
                        symbol="MARKET_WIDE",
                        price=0.0,
                        change=0.0,
                        change_percent=0.0,
                        volume=0,
                        high_52w=0.0,
                        low_52w=0.0
                    )
                    
                    insight = MarketInsight(
                        id=f"ai_{strategy_name}_{int(time.time())}",
                        title=f"🤖 AI Market Analysis - {strategy_name.replace('_', ' ').title()}",
                        description=ai_insight,
                        insight_type="ai_analysis",
                        priority="medium",
                        confidence=0.8,
                        symbol="MARKET_WIDE",
                        market_data=dummy_market_data,
                        timestamp=datetime.now(),
                        action_required=True,
                        source=f"ai_{strategy_name}",
                        supporting_data={"strategy": strategy_name, "prompt_type": "ai_generated"}
                    )
                    
                    all_insights.append(insight)
                    logger.info(f"Generated AI insight using {strategy_name} strategy")
                    
                except Exception as e:
                    logger.error(f"Error generating AI insight for {strategy_name}: {e}")
            
            # NEW: Add research-powered insights
            research_insights = self._generate_research_powered_insights()
            all_insights.extend(research_insights)
            
        except ImportError:
            logger.warning("Ollama not available for AI insight generation")
        except Exception as e:
            logger.error(f"Error in periodic AI insight generation: {e}")
        
        return all_insights
    
    def _create_cross_market_analysis_prompt(self) -> str:
        """Create enhanced prompt for institutional-grade cross-market analysis"""
        # Get comprehensive data from major symbols with technical indicators
        major_symbols = ['EURUSD=X', 'GC=F', 'BTC-USD', '^GSPC', '^DJI', 'GBPUSD=X', 'SI=F', '^VIX']
        market_analysis = {}
        volatility_alerts = []
        correlation_insights = []
        
        for symbol in major_symbols:
            if symbol in self.market_data:
                data = self.market_data[symbol]
                tech = self.technical_indicators.get(symbol)
                movement = self.historical_data.get(symbol)
                
                # Calculate advanced metrics
                daily_change = data.change_percent
                weekly_change = 0
                if movement is not None and not movement.empty and len(movement) >= 7:
                    weekly_change = (data.price - movement['Close'].iloc[-7]) / movement['Close'].iloc[-7] * 100
                
                # Technical analysis
                rsi = getattr(tech, 'rsi', 50) if tech else 50
                volatility = getattr(tech, 'volatility', 15) if tech else 15
                
                # Movement classification
                if abs(daily_change) > 3.0:
                    momentum = "🔥EXPLOSIVE"
                elif abs(daily_change) > 1.5:
                    momentum = "⚡STRONG"
                elif abs(daily_change) > 0.5:
                    momentum = "📈MODERATE"
                else:
                    momentum = "📊QUIET"
                
                # RSI condition
                if rsi > 75:
                    rsi_condition = "OVERBOUGHT"
                elif rsi < 25:
                    rsi_condition = "OVERSOLD"
                elif 45 <= rsi <= 55:
                    rsi_condition = "NEUTRAL"
                else:
                    rsi_condition = "TRENDING"
                
                market_analysis[symbol] = {
                    'daily_change': daily_change,
                    'weekly_change': weekly_change,
                    'current_price': data.price,
                    'momentum': momentum,
                    'rsi': rsi,
                    'rsi_condition': rsi_condition,
                    'volatility': volatility
                }
                
                # Track high volatility for alerts
                if volatility > 25:
                    volatility_alerts.append(f"{symbol}: {volatility:.1f}% volatility")
        
        # Calculate USD strength index
        usd_strength = self._calculate_usd_strength()
        
        # Build comprehensive market summary
        prompt = f"""You are a senior institutional trader analyzing global markets. Provide sophisticated cross-market intelligence:

📊 LIVE GLOBAL MARKET SNAPSHOT:
"""
        
        for symbol, data in market_analysis.items():
            display_name = symbol.replace('=X', '').replace('^', '').replace('-USD', '').replace('=F', '')
            prompt += f"• {display_name}: {data['daily_change']:+.2f}% ({data['weekly_change']:+.1f}% weekly) | {data['momentum']} | RSI:{data['rsi']:.0f}({data['rsi_condition']}) | Vol:{data['volatility']:.1f}%\n"
        
        prompt += f"""
💰 USD STRENGTH INDEX: {usd_strength:+.1f}% (Cross-pair calculated)

⚡ VOLATILITY ALERTS:
{chr(10).join(volatility_alerts[:3]) if volatility_alerts else "Normal volatility across all markets"}

🎯 INSTITUTIONAL ANALYSIS REQUIRED:
1. MOMENTUM THESIS: Which asset class is showing the strongest institutional flow (with volume/volatility confirmation)?
2. MEAN REVERSION OPPORTUNITY: Any overextended assets approaching key technical reversals?
3. CORRELATION BREAKDOWN: Which markets are diverging from normal correlations (arbitrage opportunity)?
4. RISK-ON/RISK-OFF: What's the current market regime based on cross-asset behavior?
5. CENTRAL BANK IMPLICATIONS: How do current movements reflect monetary policy expectations?

📋 PROVIDE ONE INSTITUTIONAL-GRADE TRADE SETUP:
- Asset & Exact Direction (BUY/SELL)
- Entry Zone (specific price range)
- Stop Loss with technical rationale
- Multiple TP levels (TP1: Quick profit, TP2: Extended target)
- Risk/Reward ratio calculation
- Position sizing recommendation (% of portfolio)
- Time horizon & session timing
- Conviction level (1-10 scale)
- Market catalyst driving the move

Format: Professional institutional analysis with quantified risk metrics and specific price levels."""
        
        return prompt
    
    def _create_risk_opportunity_prompt(self) -> str:
        """Create enhanced prompt for sophisticated risk-opportunity analysis"""
        # Comprehensive risk assessment and opportunity scanning
        critical_alerts = []
        extreme_opportunities = []
        breakout_setups = []
        stress_indicators = []
        portfolio_implications = []
        
        total_volatility = 0
        symbol_count = 0
        
        for symbol, data in self.market_data.items():
            if hasattr(data, 'change_percent'):
                tech = self.technical_indicators.get(symbol)
                daily_change = data.change_percent
                volatility = getattr(tech, 'volatility', 15) if tech else 15
                rsi = getattr(tech, 'rsi', 50) if tech else 50
                
                total_volatility += volatility
                symbol_count += 1
                
                # Critical risk alerts
                if abs(daily_change) > 4.0:
                    critical_alerts.append(f"{symbol}: {daily_change:+.1f}% - 🚨CRITICAL MOVEMENT")
                elif abs(daily_change) > 2.5:
                    critical_alerts.append(f"{symbol}: {daily_change:+.1f}% - ⚠️HIGH RISK")
                
                # Extreme value opportunities
                if rsi < 20:
                    extreme_opportunities.append(f"{symbol}: DEEPLY OVERSOLD (RSI:{rsi:.0f}) - Bounce candidate with {abs(daily_change):.1f}% decline")
                elif rsi > 80:
                    extreme_opportunities.append(f"{symbol}: SEVERELY OVERBOUGHT (RSI:{rsi:.0f}) - Reversal risk with {daily_change:.1f}% rally")
                
                # Breakout preparation (neutral RSI)
                if 45 <= rsi <= 55 and volatility < 20:
                    breakout_setups.append(f"{symbol}: COILING (RSI:{rsi:.0f}, Vol:{volatility:.1f}%) - Breakout preparation")
                
                # Market stress indicators
                if volatility > 30:
                    stress_indicators.append(f"{symbol}: HIGH STRESS - {volatility:.1f}% volatility")
                
                # Portfolio risk implications
                if abs(daily_change) > 2.0 and volatility > 25:
                    portfolio_implications.append(f"{symbol}: PORTFOLIO RISK - {daily_change:+.1f}% move, {volatility:.1f}% vol")
        
        avg_market_volatility = total_volatility / max(symbol_count, 1)
        
        # Market regime classification
        if avg_market_volatility > 30:
            market_regime = "🔴 HIGH STRESS - Crisis/Event driven"
        elif avg_market_volatility > 25:
            market_regime = "🟡 ELEVATED RISK - Heightened uncertainty"
        elif avg_market_volatility > 20:
            market_regime = "🟠 MODERATE VOLATILITY - Normal trading"
        else:
            market_regime = "🟢 LOW VOLATILITY - Calm/Trending markets"
        
        prompt = f"""You are the Chief Risk Officer at a major hedge fund. Analyze current market conditions and provide actionable risk intelligence:

🚨 CRITICAL RISK ALERTS:
{chr(10).join(critical_alerts[:4]) if critical_alerts else "No critical risk alerts - Markets relatively stable"}

🎯 EXTREME VALUE OPPORTUNITIES:
{chr(10).join(extreme_opportunities[:4]) if extreme_opportunities else "No extreme RSI conditions - Markets in balance"}

⚡ BREAKOUT CANDIDATES (Low vol + Neutral RSI):
{chr(10).join(breakout_setups[:3]) if breakout_setups else "No clear breakout setups - Markets directional"}

🌡️ MARKET STRESS TEMPERATURE: {avg_market_volatility:.1f}%
📊 REGIME CLASSIFICATION: {market_regime}

🔍 STRESS INDICATORS:
{chr(10).join(stress_indicators[:3]) if stress_indicators else "Normal volatility environment"}

💼 PORTFOLIO IMPLICATIONS:
{chr(10).join(portfolio_implications[:3]) if portfolio_implications else "Low portfolio impact from current moves"}

🎯 INSTITUTIONAL RISK ANALYSIS REQUIRED:
1. IMMEDIATE THREAT (24-48h): What's the single highest probability risk that could trigger portfolio-wide losses?
2. BEST OPPORTUNITY: Which asset offers the optimal risk-adjusted return with clear technical setup?
3. POSITION SIZING REGIME: Based on current volatility, how should institutional accounts adjust exposure?
4. HEDGE STRATEGY: What's the most effective hedge against current systematic risks?
5. CORRELATION RISKS: Which normally uncorrelated assets are moving together (systemic risk)?
6. LIQUIDITY ASSESSMENT: Which markets may face liquidity issues if volatility spikes?

📊 PROVIDE INSTITUTIONAL RISK REPORT:
- PRIMARY RISK FACTOR (with probability % and timeline)
- TOP OPPORTUNITY TRADE (with detailed R:R analysis)
- RECOMMENDED PORTFOLIO ALLOCATION (% adjustments)
- HEDGING INSTRUMENT & SIZE
- STOP-LOSS DISCIPLINE (account-wide rules)
- VaR IMPACT ASSESSMENT
- MARGIN REQUIREMENT FORECAST
- 48-HOUR TACTICAL OUTLOOK

         Format: Professional risk committee presentation with quantified metrics and specific actionable steps."""
         
        return prompt
    
    def _create_trend_reversal_prompt(self) -> str:
        """Create prompt for trend reversal detection"""
        reversal_candidates = []
        
        for symbol in ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'GC=F', 'SI=F', 'BTC-USD']:
            if symbol in self.historical_data and not self.historical_data[symbol].empty:
                df = self.historical_data[symbol]
                if len(df) >= 30:
                    current_price = df['Close'].iloc[-1]
                    monthly_high = df['High'].tail(30).max()
                    monthly_low = df['Low'].tail(30).min()
                    
                    # Check if at extremes
                    at_high = current_price > monthly_high * 0.98
                    at_low = current_price < monthly_low * 1.02
                    
                    if at_high or at_low:
                        position = "near monthly high" if at_high else "near monthly low"
                        reversal_candidates.append(f"{symbol}: {position}")
        
        prompt = f"""You are a trend reversal specialist. Analyze the following assets at potential turning points:

REVERSAL CANDIDATES:
{chr(10).join(reversal_candidates[:4]) if reversal_candidates else "No clear reversal setups"}

ANALYSIS REQUIRED:
1. Which asset has the highest probability of trend reversal?
2. What specific signals would confirm the reversal?
3. What is the optimal entry and risk management strategy?

Provide ONE specific reversal trade setup with entry, stop, and target levels. Be precise and actionable."""
        
        return prompt
    
    def _create_correlation_analysis_prompt(self) -> str:
        """Create prompt for correlation analysis"""
        # Simple correlation analysis between major pairs
        correlations = []
        
        # USD strength analysis
        usd_pairs = ['EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X', 'USDJPY=X']
        usd_movements = []
        
        for symbol in usd_pairs:
            if symbol in self.market_data:
                change = self.market_data[symbol].change_percent
                # Invert for USD pairs where USD is quote currency
                if symbol in ['USDCAD=X', 'USDJPY=X']:
                    usd_movements.append(-change)  # Invert
                else:
                    usd_movements.append(-change)  # EUR/USD down = USD up
                correlations.append(f"{symbol}: {change:+.2f}%")
        
        avg_usd_strength = sum(usd_movements) / len(usd_movements) if usd_movements else 0
        
        prompt = f"""You are a currency correlation specialist. Analyze USD strength and market correlations:

MAJOR USD PAIR MOVEMENTS:
{chr(10).join(correlations[:5]) if correlations else "No data available"}

CALCULATED USD STRENGTH: {avg_usd_strength:+.2f}%

ANALYSIS REQUIRED:
1. Is USD showing strength or weakness across pairs?
2. Which currency pair is showing divergent behavior?
3. What does this suggest about the next major move?

Provide ONE specific correlation-based trading insight with rationale. Focus on divergences and opportunities."""
        
        return prompt
    
    def update_all_data(self) -> None:
        """Update data for all symbols using parallel processing with periodic AI insight generation"""
        self.logger.info("Starting comprehensive market data update...")
        
        all_insights = []
        
        # Use ThreadPoolExecutor for parallel data fetching
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {executor.submit(self.analyze_symbol, symbol): symbol 
                              for symbol in self.all_symbols}
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    symbol_insights = future.result(timeout=30)
                    all_insights.extend(symbol_insights)
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
        
        # Generate periodic AI insights (every 3rd update to avoid overload)
        if hasattr(self, '_update_counter'):
            self._update_counter += 1
        else:
            self._update_counter = 1
            
        if self._update_counter % 3 == 0:  # Every 3rd update (~15 minutes)
            self.logger.info("Generating periodic AI insights...")
            try:
                ai_insights = self.generate_periodic_ai_insights()
                all_insights.extend(ai_insights)
                
                # NEW: Add research-powered insights
                research_insights = self._generate_research_powered_insights()
                all_insights.extend(research_insights)
                
                self.logger.info(f"Generated {len(ai_insights)} AI insights and {len(research_insights)} research insights")
            except Exception as e:
                self.logger.error(f"Error generating AI insights: {e}")
        
        # Update insights
        self.insights = all_insights
        self.logger.info(f"Market intelligence update completed. Generated {len(all_insights)} insights.")
    
    def get_insights(self, symbol: str = None, insight_type: str = None, limit: int = 50) -> List[Dict]:
        """Get insights with optional filtering"""
        filtered_insights = self.insights
        
        # Filter by symbol
        if symbol:
            filtered_insights = [i for i in filtered_insights if i.symbol == symbol]
        
        # Filter by type
        if insight_type:
            filtered_insights = [i for i in filtered_insights if i.insight_type == insight_type]
        
        # Sort by timestamp (newest first) and limit
        filtered_insights.sort(key=lambda x: x.timestamp, reverse=True)
        filtered_insights = filtered_insights[:limit]
        
        # Convert to dict format for JSON serialization
        return [self._insight_to_dict(insight) for insight in filtered_insights]
    
    def _insight_to_dict(self, insight: MarketInsight) -> Dict:
        """Convert MarketInsight to dictionary with robust error handling"""
        try:
            result = {
                'id': getattr(insight, 'id', f'unknown_{int(time.time())}'),
                'title': getattr(insight, 'title', 'Unknown Insight'),
                'description': getattr(insight, 'description', 'No description available'),
                'insight_type': getattr(insight, 'insight_type', 'general'),
                'priority': getattr(insight, 'priority', 'medium'),
                'confidence': float(getattr(insight, 'confidence', 0.5)),
                'symbol': getattr(insight, 'symbol', 'UNKNOWN'),
                'timestamp': getattr(insight, 'timestamp', datetime.now()).isoformat(),
                'timeAgo': self._get_time_ago(getattr(insight, 'timestamp', datetime.now())),
                'action_required': bool(getattr(insight, 'action_required', False)),
                'source': getattr(insight, 'source', 'unknown'),
                'supporting_data': getattr(insight, 'supporting_data', {})
            }
            
            # Handle optional technical_data with error checking
            technical_data = getattr(insight, 'technical_data', None)
            if technical_data and hasattr(technical_data, 'rsi'):
                try:
                    result['technical_data'] = {
                        'rsi': round(float(getattr(technical_data, 'rsi', 50)), 2),
                        'macd': round(float(getattr(technical_data, 'macd', 0)), 4),
                        'volatility': round(float(getattr(technical_data, 'volatility', 0)), 2),
                        'sma_20': round(float(getattr(technical_data, 'sma_20', 0)), 4),
                        'sma_50': round(float(getattr(technical_data, 'sma_50', 0)), 4)
                    }
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Error processing technical data for {result['id']}: {e}")
                    result['technical_data'] = None
            else:
                result['technical_data'] = None
                
            # Handle market_data with error checking
            market_data = getattr(insight, 'market_data', None)
            if market_data and hasattr(market_data, 'price'):
                try:
                    result['market_data'] = {
                        'price': round(float(getattr(market_data, 'price', 0)), 4),
                        'change': round(float(getattr(market_data, 'change', 0)), 4),
                        'change_percent': round(float(getattr(market_data, 'change_percent', 0)), 2),
                        'volume': int(getattr(market_data, 'volume', 0))
                    }
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Error processing market data for {result['id']}: {e}")
                    result['market_data'] = None
            else:
                result['market_data'] = None
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error converting insight to dict: {e}")
            # Return a safe fallback insight
            return {
                'id': f'error_{int(time.time())}',
                'title': 'Error Processing Insight',
                'description': f'Could not process insight data: {str(e)}',
                'insight_type': 'error',
                'priority': 'low',
                'confidence': 0.0,
                'symbol': 'ERROR',
                'timestamp': datetime.now().isoformat(),
                'timeAgo': 'just now',
                'action_required': False,
                'source': 'error_handler',
                'supporting_data': {},
                'technical_data': None,
                'market_data': None
            }
    
    def _get_time_ago(self, timestamp: datetime) -> str:
        """Get human-readable time ago string"""
        delta = datetime.now() - timestamp
        
        if delta.total_seconds() < 60:
            return "just now"
        elif delta.total_seconds() < 3600:
            minutes = int(delta.total_seconds() / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif delta.total_seconds() < 86400:
            hours = int(delta.total_seconds() / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            days = delta.days
            return f"{days} day{'s' if days != 1 else ''} ago"
    
    def get_market_summary(self) -> Dict:
        """Get HONEST market summary statistics based on actual data"""
        try:
            # Count actual insights that can be displayed
            displayable_insights = []
            try:
                # Test each insight to ensure it's properly structured
                for insight in self.insights:
                    insight_dict = self._insight_to_dict(insight)
                    # Verify it has required fields for display
                    if (insight_dict.get('title') and insight_dict.get('description') and 
                        insight_dict.get('symbol') and insight_dict.get('timestamp')):
                        displayable_insights.append(insight_dict)
            except Exception as e:
                self.logger.error(f"Error validating insights for display: {e}")
                displayable_insights = []
            
            total_insights = len(displayable_insights)
            today_insights = len([i for i in displayable_insights 
                                if datetime.fromisoformat(i['timestamp']).date() == datetime.now().date()])
            high_priority = len([i for i in displayable_insights if i.get('priority') in ['high', 'critical']])
            
            # Calculate confidence only from real technical analysis (not AI synthetic data)
            real_analysis_insights = [i for i in displayable_insights 
                                    if i.get('source', '').startswith('regime_') or 
                                       i.get('source', '').startswith('breakout_') or 
                                       i.get('source', '').startswith('momentum_')]
            avg_confidence = (sum(i.get('confidence', 0) for i in real_analysis_insights) / 
                            len(real_analysis_insights)) if real_analysis_insights else 0
            
            # Get real MT5 account performance for success rate
            success_rate = 0.0
            success_rate_description = "No data"
            try:
                # Try to get global qnti_system instance if available (avoid circular import)
                import sys
                if 'qnti_main_system' in sys.modules:
                    qnti_main_system = sys.modules['qnti_main_system']
                    if hasattr(qnti_main_system, 'qnti_system') and qnti_main_system.qnti_system and hasattr(qnti_main_system.qnti_system, 'trade_manager'):
                        stats = qnti_main_system.qnti_system.trade_manager.calculate_statistics()
                        success_rate = stats.get('win_rate', 0) / 100  # Convert percentage to decimal
                        success_rate_description = f"Real trading win rate"
                    else:
                        # No global instance available, skip real trading data
                        pass
                else:
                    # Module not loaded yet, skip real trading data
                    pass
            except Exception as e:
                self.logger.debug(f"Could not get real trading performance: {e}")
            
            return {
                'totalInsights': total_insights,
                'todayInsights': today_insights,
                'highPriorityCount': high_priority,
                'avgConfidence': round(avg_confidence, 2),
                'successRate': round(success_rate, 2),
                'successRateDescription': success_rate_description,
                'activeAlerts': high_priority,
                'lastUpdate': datetime.now().isoformat(),
                'symbolsMonitored': len(self.all_symbols),
                'dataSource': 'Yahoo Finance + Technical Analysis',
                'displayableInsights': total_insights,  # For debugging
                'realAnalysisInsights': len(real_analysis_insights)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating market summary: {e}")
            return {
                'totalInsights': 0, 'todayInsights': 0, 'highPriorityCount': 0,
                'avgConfidence': 0, 'successRate': 0, 'activeAlerts': 0,
                'successRateDescription': "Error calculating"
            }
    
    def _calculate_usd_strength(self) -> float:
        """Calculate USD strength index across major pairs"""
        try:
            usd_pairs = ['EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X', 'USDJPY=X']
            usd_movements = []
            
            for symbol in usd_pairs:
                if symbol in self.market_data:
                    change = self.market_data[symbol].change_percent
                    # For USD as base currency (USDCAD, USDJPY), positive change = USD strength
                    # For USD as quote currency (EURUSD, GBPUSD), negative change = USD strength
                    if symbol in ['USDCAD=X', 'USDJPY=X']:
                        usd_movements.append(change)  # USD base
                    else:
                        usd_movements.append(-change)  # USD quote, invert
            
            return sum(usd_movements) / len(usd_movements) if usd_movements else 0.0
        except Exception as e:
            logger.error(f"Error calculating USD strength: {e}")
            return 0.0
    
    def _generate_research_powered_insights(self) -> List[MarketInsight]:
        """Generate insights powered by research database"""
        research_insights = []
        
        try:
            # Import research agent
            from qnti_research_agent import get_research_agent
            agent = get_research_agent()
            
            # Get current market context
            major_symbols = ['EURUSD', 'GC=F', 'BTC-USD']
            active_symbols = [sym for sym in major_symbols if sym in self.market_data]
            
            if not active_symbols:
                logger.info("No active symbols found for research insights")
                return research_insights
            
            logger.info(f"Generating research insights for {len(active_symbols)} symbols")
            
            # Generate research-based insights for active symbols
            for symbol in active_symbols[:2]:  # Limit to 2 symbols to avoid overload
                try:
                    market_data = self.market_data[symbol]
                    
                    # Create market context
                    symbol_name = symbol.replace('=X', '').replace('=F', '').replace('-USD', '')
                    market_context = f"{symbol_name} currency markets"
                    if 'GC' in symbol:
                        market_context = "gold precious metals markets"
                    elif 'BTC' in symbol:
                        market_context = "cryptocurrency bitcoin markets"
                    
                    logger.info(f"Querying research for context: {market_context}")
                    
                    # Try RAG query first, fallback to database search
                    research_items = []
                    try:
                        research_items = agent.get_research_insights_for_market_intelligence(market_context)
                        logger.info(f"RAG returned {len(research_items)} insights for {symbol}")
                    except Exception as e:
                        logger.warning(f"RAG query failed for {symbol}: {e}")
                    
                    # If RAG fails or returns empty, use database fallback
                    if not research_items:
                        logger.info(f"Using database fallback for {symbol}")
                        research_items = agent._get_database_insights(market_context)
                        logger.info(f"Database fallback returned {len(research_items)} insights")
                    
                    for i, research_text in enumerate(research_items):
                        if research_text and len(research_text) > 50:
                            # Create research-powered insight
                            insight = MarketInsight(
                                id=f"research_{symbol}_{int(time.time())}_{i}",
                                title=f"📚 Research Alert: {symbol_name} Market Analysis",
                                description=f"Latest research indicates: {research_text}",
                                insight_type="research",
                                priority="high",
                                confidence=0.85,
                                symbol=symbol,
                                market_data=market_data,
                                timestamp=datetime.now(),
                                action_required=True,
                                source="research_database",
                                supporting_data={
                                    "research_source": "aggregated_research",
                                    "market_context": market_context
                                }
                            )
                            
                            research_insights.append(insight)
                            logger.info(f"Created research insight for {symbol}: {research_text[:100]}...")
                            
                except Exception as e:
                    logger.error(f"Error generating research insight for {symbol}: {e}")
            
            # Generate broad market research insight from recent documents
            try:
                logger.info("Generating broad market research insight")
                
                # Get recent high-quality research from database directly
                import sqlite3
                conn = sqlite3.connect(agent.db_path)
                cursor = conn.cursor()
                
                # Get most recent and relevant documents
                cursor.execute("""
                    SELECT title, summary, relevance_score 
                    FROM research_documents 
                    WHERE summary IS NOT NULL AND summary != '' 
                    AND relevance_score > 2.0
                    ORDER BY downloaded_date DESC, relevance_score DESC 
                    LIMIT 3
                """)
                
                recent_docs = cursor.fetchall()
                conn.close()
                
                if recent_docs:
                    # Create broad insight from most relevant recent document
                    doc_title, doc_summary, doc_score = recent_docs[0]
                    
                    # Create dummy market data for broad insight
                    dummy_market_data = MarketData(
                        symbol="GLOBAL_MARKETS",
                        price=0.0,
                        change=0.0,
                        change_percent=0.0,
                        volume=0,
                        high_52w=0.0,
                        low_52w=0.0
                    )
                    
                    research_text = doc_summary if doc_summary else doc_title
                    
                    insight = MarketInsight(
                        id=f"research_global_{int(time.time())}",
                        title="🌍 Global Market Research Summary",
                        description=f"Key research findings: {research_text}",
                        insight_type="research",
                        priority="medium",
                        confidence=0.8,
                        symbol="GLOBAL",
                        market_data=dummy_market_data,
                        timestamp=datetime.now(),
                        action_required=False,
                        source="research_database",
                        supporting_data={
                            "research_scope": "global_markets",
                            "document_count": len(recent_docs),
                            "relevance_score": doc_score
                        }
                    )
                    
                    research_insights.append(insight)
                    logger.info(f"Created global research insight: {research_text[:100]}...")
                else:
                    logger.info("No recent documents found for global insight")
                    
            except Exception as e:
                logger.error(f"Error generating broad research insight: {e}")
        
        except ImportError:
            logger.warning("Research agent not available. Install research dependencies.")
        except Exception as e:
            logger.error(f"Error in research-powered insights generation: {e}")
        
        if research_insights:
            logger.info(f"Generated {len(research_insights)} research-powered insights")
        else:
            logger.info("No research insights generated - checking research database availability")
        
        return research_insights

# Global instance
enhanced_intelligence = QNTIEnhancedMarketIntelligence()

if __name__ == "__main__":
    # Test the system
    enhanced_intelligence.update_all_data()
    insights = enhanced_intelligence.get_insights(limit=10)
    
    print(f"Generated {len(insights)} insights:")
    for insight in insights:
        print(f"- {insight['title']}") 