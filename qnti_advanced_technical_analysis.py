#!/usr/bin/env python3
"""
Advanced Technical Analysis Module
Provides detailed technical metrics, Fibonacci analysis, entry zones, and pullback probabilities
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class AdvancedTechnicalMetrics:
    """Comprehensive technical analysis metrics"""
    # Price Action
    current_price: float
    high_52w: float
    low_52w: float
    price_vs_52w_high: float  # percentage
    price_vs_52w_low: float   # percentage
    
    # Moving Averages with detailed analysis
    sma_20: float
    sma_50: float
    sma_200: float
    ema_9: float
    ema_21: float
    ema_50: float
    ma_alignment: str  # "bullish", "bearish", "mixed"
    ma_slope_20: float  # slope of 20-day MA
    ma_slope_50: float  # slope of 50-day MA
    
    # Oscillators with specific readings
    rsi_14: float
    rsi_signal: str  # "oversold", "overbought", "neutral"
    stochastic_k: float
    stochastic_d: float
    stochastic_signal: str
    williams_r: float
    cci_20: float
    
    # MACD detailed analysis
    macd_line: float
    macd_signal: float
    macd_histogram: float
    macd_trend: str  # "bullish", "bearish", "neutral"
    macd_divergence: str  # "bullish_div", "bearish_div", "none"
    
    # Bollinger Bands
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_position: float  # 0-100, where price sits in bands
    bb_squeeze: bool   # True if bands are contracting
    
    # Volume Analysis
    volume_sma_20: float
    volume_ratio: float  # current volume vs 20-day average
    volume_trend: str   # "increasing", "decreasing", "stable"
    
    # Volatility
    atr_14: float
    volatility_percentile: float  # where current volatility sits historically
    
    # Support/Resistance
    immediate_support: float
    immediate_resistance: float
    major_support: float
    major_resistance: float
    
@dataclass 
class FibonacciAnalysis:
    """Fibonacci retracement and extension analysis"""
    swing_high: float
    swing_low: float
    trend_direction: str  # "up", "down"
    
    # Retracement levels
    fib_236: float
    fib_382: float
    fib_500: float
    fib_618: float
    fib_786: float
    
    # Extension levels
    fib_ext_127: float
    fib_ext_162: float
    fib_ext_200: float
    fib_ext_262: float
    
    # Current position
    current_fib_level: str  # which level price is near
    next_fib_target: float
    fib_support: float
    fib_resistance: float

@dataclass
class PullbackAnalysis:
    """Detailed pullback probability and entry zone analysis"""
    trend_strength: float  # 0-100 scale
    pullback_probability: float  # 0-100 percentage
    pullback_target_1: float  # 23.6% retracement
    pullback_target_2: float  # 38.2% retracement  
    pullback_target_3: float  # 50% retracement
    
    # Entry zones for trend continuation
    entry_zone_optimal: float
    entry_zone_aggressive: float
    entry_zone_conservative: float
    
    # Stop loss levels
    stop_loss_tight: float
    stop_loss_normal: float
    stop_loss_wide: float
    
    # Targets for trend continuation
    target_1: float  # next resistance/support
    target_2: float  # major level
    target_3: float  # extension target
    
    # Risk/Reward ratios
    rr_ratio_1: float
    rr_ratio_2: float
    rr_ratio_3: float

class AdvancedTechnicalAnalyzer:
    """Advanced technical analysis with detailed metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.AdvancedTechnicalAnalyzer")
        
    def get_comprehensive_analysis(self, symbol: str, period: str = "1y") -> Dict:
        """Get comprehensive technical analysis for a symbol"""
        try:
            # Fetch historical data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                return {"error": f"No data available for {symbol}"}
            
            # Calculate all technical metrics
            technical_metrics = self._calculate_technical_metrics(df)
            fibonacci_analysis = self._calculate_fibonacci_analysis(df)
            pullback_analysis = self._calculate_pullback_analysis(df, technical_metrics, fibonacci_analysis)
            
            # Resolve indicator conflicts for coherent analysis
            conflict_resolution = self._resolve_indicator_conflicts(technical_metrics)
            
            # Generate detailed narrative with conflict resolution
            narrative = self._generate_detailed_narrative(symbol, technical_metrics, fibonacci_analysis, pullback_analysis, conflict_resolution)
            
            return {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "technical_metrics": technical_metrics,
                "fibonacci_analysis": fibonacci_analysis,
                "pullback_analysis": pullback_analysis,
                "conflict_resolution": conflict_resolution,
                "detailed_narrative": narrative,
                "signal_strength": self._calculate_signal_strength(technical_metrics, fibonacci_analysis),
                "trade_setup": self._generate_trade_setup(pullback_analysis, technical_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return {"error": str(e)}
    
    def _calculate_technical_metrics(self, df: pd.DataFrame) -> AdvancedTechnicalMetrics:
        """Calculate comprehensive technical indicators"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume'] if 'Volume' in df.columns else pd.Series([1] * len(df))
        
        # Price metrics
        current_price = close.iloc[-1]
        high_52w = close.tail(252).max()
        low_52w = close.tail(252).min()
        
        # Moving Averages
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        ema_9 = close.ewm(span=9).mean()
        ema_21 = close.ewm(span=21).mean()
        ema_50 = close.ewm(span=50).mean()
        
        # MA alignment and slopes
        ma_alignment = self._determine_ma_alignment(current_price, sma_20.iloc[-1], sma_50.iloc[-1], sma_200.iloc[-1])
        ma_slope_20 = (sma_20.iloc[-1] - sma_20.iloc[-5]) / sma_20.iloc[-5] * 100
        ma_slope_50 = (sma_50.iloc[-1] - sma_50.iloc[-10]) / sma_50.iloc[-10] * 100
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1]
        
        # Stochastic
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        stoch_k = ((close - low_14) / (high_14 - low_14)) * 100
        stoch_d = stoch_k.rolling(3).mean()
        
        # Williams %R
        williams_r = ((high_14 - close) / (high_14 - low_14)) * -100
        
        # CCI (Commodity Channel Index)
        tp = (high + low + close) / 3
        cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9).mean()
        macd_histogram = macd_line - macd_signal
        
        # Bollinger Bands
        bb_middle = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        bb_position = ((current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])) * 100
        
        # Volume analysis
        volume_sma_20 = volume.rolling(20).mean()
        volume_ratio = volume.iloc[-1] / volume_sma_20.iloc[-1] if volume_sma_20.iloc[-1] > 0 else 1
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        # Support/Resistance (simplified - using recent highs/lows)
        recent_highs = high.tail(50)
        recent_lows = low.tail(50)
        immediate_resistance = recent_highs.quantile(0.8)
        immediate_support = recent_lows.quantile(0.2)
        major_resistance = recent_highs.max()
        major_support = recent_lows.min()
        
        return AdvancedTechnicalMetrics(
            current_price=current_price,
            high_52w=high_52w,
            low_52w=low_52w,
            price_vs_52w_high=((current_price - high_52w) / high_52w) * 100,
            price_vs_52w_low=((current_price - low_52w) / low_52w) * 100,
            sma_20=sma_20.iloc[-1],
            sma_50=sma_50.iloc[-1],
            sma_200=sma_200.iloc[-1] if len(sma_200) > 0 else current_price,
            ema_9=ema_9.iloc[-1],
            ema_21=ema_21.iloc[-1],
            ema_50=ema_50.iloc[-1],
            ma_alignment=ma_alignment,
            ma_slope_20=ma_slope_20,
            ma_slope_50=ma_slope_50,
            rsi_14=rsi_val,
            rsi_signal=self._interpret_rsi(rsi_val),
            stochastic_k=stoch_k.iloc[-1],
            stochastic_d=stoch_d.iloc[-1],
            stochastic_signal=self._interpret_stochastic(stoch_k.iloc[-1], stoch_d.iloc[-1]),
            williams_r=williams_r.iloc[-1],
            cci_20=cci.iloc[-1],
            macd_line=macd_line.iloc[-1],
            macd_signal=macd_signal.iloc[-1],
            macd_histogram=macd_histogram.iloc[-1],
            macd_trend=self._interpret_macd(macd_line.iloc[-1], macd_signal.iloc[-1], macd_histogram.iloc[-1]),
            macd_divergence="none",  # Would need more complex calculation
            bb_upper=bb_upper.iloc[-1],
            bb_middle=bb_middle.iloc[-1],
            bb_lower=bb_lower.iloc[-1],
            bb_position=bb_position,
            bb_squeeze=self._detect_bb_squeeze(bb_upper, bb_lower),
            volume_sma_20=volume_sma_20.iloc[-1],
            volume_ratio=volume_ratio,
            volume_trend=self._interpret_volume_trend(volume, volume_sma_20),
            atr_14=atr.iloc[-1],
            volatility_percentile=self._calculate_volatility_percentile(atr),
            immediate_support=immediate_support,
            immediate_resistance=immediate_resistance,
            major_support=major_support,
            major_resistance=major_resistance
        )
    
    def _calculate_fibonacci_analysis(self, df: pd.DataFrame) -> FibonacciAnalysis:
        """Calculate Fibonacci retracements and extensions"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Find significant swing high and low (last 50 periods)
        recent_data = df.tail(50)
        swing_high = recent_data['High'].max()
        swing_low = recent_data['Low'].min()
        
        # Determine trend direction
        trend_direction = "up" if close.iloc[-1] > close.iloc[-20] else "down"
        
        # Calculate Fibonacci levels
        fib_range = swing_high - swing_low
        
        if trend_direction == "up":
            # Retracements from swing high
            fib_236 = swing_high - (fib_range * 0.236)
            fib_382 = swing_high - (fib_range * 0.382)
            fib_500 = swing_high - (fib_range * 0.500)
            fib_618 = swing_high - (fib_range * 0.618)
            fib_786 = swing_high - (fib_range * 0.786)
            
            # Extensions above swing high
            fib_ext_127 = swing_high + (fib_range * 0.272)
            fib_ext_162 = swing_high + (fib_range * 0.618)
            fib_ext_200 = swing_high + (fib_range * 1.000)
            fib_ext_262 = swing_high + (fib_range * 1.618)
        else:
            # Retracements from swing low
            fib_236 = swing_low + (fib_range * 0.236)
            fib_382 = swing_low + (fib_range * 0.382)
            fib_500 = swing_low + (fib_range * 0.500)
            fib_618 = swing_low + (fib_range * 0.618)
            fib_786 = swing_low + (fib_range * 0.786)
            
            # Extensions below swing low
            fib_ext_127 = swing_low - (fib_range * 0.272)
            fib_ext_162 = swing_low - (fib_range * 0.618)
            fib_ext_200 = swing_low - (fib_range * 1.000)
            fib_ext_262 = swing_low - (fib_range * 1.618)
        
        current_price = close.iloc[-1]
        
        # Find nearest Fibonacci level
        fib_levels = [fib_236, fib_382, fib_500, fib_618, fib_786]
        nearest_fib = min(fib_levels, key=lambda x: abs(x - current_price))
        
        # Determine current Fibonacci level
        current_fib_level = self._identify_current_fib_level(current_price, fib_levels)
        
        return FibonacciAnalysis(
            swing_high=swing_high,
            swing_low=swing_low,
            trend_direction=trend_direction,
            fib_236=fib_236,
            fib_382=fib_382,
            fib_500=fib_500,
            fib_618=fib_618,
            fib_786=fib_786,
            fib_ext_127=fib_ext_127,
            fib_ext_162=fib_ext_162,
            fib_ext_200=fib_ext_200,
            fib_ext_262=fib_ext_262,
            current_fib_level=current_fib_level,
            next_fib_target=self._get_next_fib_target(current_price, fib_levels, trend_direction),
            fib_support=min([f for f in fib_levels if f < current_price], default=swing_low),
            fib_resistance=min([f for f in fib_levels if f > current_price], default=swing_high)
        )
    
    def _calculate_pullback_analysis(self, df: pd.DataFrame, tech_metrics: AdvancedTechnicalMetrics, fib_analysis: FibonacciAnalysis) -> PullbackAnalysis:
        """Calculate LOGICAL pullback probability and entry zones"""
        close = df['Close']
        current_price = tech_metrics.current_price
        atr = tech_metrics.atr_14
        
        # Calculate trend strength (0-100)
        trend_strength = self._calculate_trend_strength(tech_metrics)
        
        # Calculate pullback probability based on technical factors
        pullback_probability = self._calculate_pullback_probability(tech_metrics, fib_analysis)
        
        # Calculate pullback targets (always same regardless of trend direction)
        if fib_analysis.trend_direction == "up":
            pullback_target_1 = fib_analysis.fib_236  # Shallow pullback
            pullback_target_2 = fib_analysis.fib_382  # Medium pullback
            pullback_target_3 = fib_analysis.fib_500  # Deep pullback
        else:
            pullback_target_1 = fib_analysis.fib_236  # Shallow bounce
            pullback_target_2 = fib_analysis.fib_382  # Medium bounce
            pullback_target_3 = fib_analysis.fib_500  # Strong bounce
        
        # ⚡ LOGICAL ENTRY ZONES - Fixed the major bug!
        if fib_analysis.trend_direction == "up":
            # UPTREND - Long positions
            if pullback_probability > 65:  # Expecting significant pullback
                # Wait for pullback TO these levels, then go long
                entry_zone_optimal = pullback_target_2      # Enter near 38.2% retracement
                entry_zone_aggressive = pullback_target_1   # Enter near 23.6% retracement  
                entry_zone_conservative = pullback_target_3 # Enter near 50% retracement
            else:  # Low pullback probability - breakout/momentum entries
                # Enter ABOVE current price on breakout
                entry_zone_optimal = max(current_price * 1.002, tech_metrics.immediate_resistance * 1.001)
                entry_zone_aggressive = current_price * 1.001  # Immediate momentum entry
                entry_zone_conservative = tech_metrics.immediate_resistance * 1.003  # Confirmed breakout
            
            # Targets for uptrend continuation
            target_1 = max(tech_metrics.immediate_resistance * 1.01, fib_analysis.fib_ext_127)
            target_2 = max(tech_metrics.major_resistance * 1.005, fib_analysis.fib_ext_162)
            target_3 = fib_analysis.fib_ext_200
            
            # Stop losses (always below entry for longs)
            stop_loss_tight = min(entry_zone_optimal * 0.995, current_price * 0.99)
            stop_loss_normal = min(entry_zone_optimal * 0.985, fib_analysis.fib_618)
            stop_loss_wide = min(fib_analysis.swing_low, current_price * 0.97)
            
        else:  # DOWNTREND - Short positions
            if pullback_probability > 65:  # Expecting significant bounce
                # Wait for bounce TO these levels, then go short
                entry_zone_optimal = pullback_target_2      # Enter near 38.2% bounce
                entry_zone_aggressive = pullback_target_1   # Enter near 23.6% bounce
                entry_zone_conservative = pullback_target_3 # Enter near 50% bounce
            else:  # Low bounce probability - breakdown entries
                # Enter BELOW current price on breakdown
                entry_zone_optimal = min(current_price * 0.998, tech_metrics.immediate_support * 0.999)
                entry_zone_aggressive = current_price * 0.999  # Immediate breakdown entry
                entry_zone_conservative = tech_metrics.immediate_support * 0.997  # Confirmed breakdown
            
            # Targets for downtrend continuation
            target_1 = min(tech_metrics.immediate_support * 0.99, fib_analysis.fib_ext_127)
            target_2 = min(tech_metrics.major_support * 0.995, fib_analysis.fib_ext_162)
            target_3 = fib_analysis.fib_ext_200
            
            # Stop losses (always above entry for shorts)
            stop_loss_tight = max(entry_zone_optimal * 1.005, current_price * 1.01)
            stop_loss_normal = max(entry_zone_optimal * 1.015, fib_analysis.fib_618)
            stop_loss_wide = max(fib_analysis.swing_high, current_price * 1.03)
        
        # Calculate CORRECT Risk/Reward ratios
        risk_1 = abs(entry_zone_optimal - stop_loss_normal)
        risk_2 = abs(entry_zone_optimal - stop_loss_normal)
        risk_3 = abs(entry_zone_optimal - stop_loss_normal)
        
        reward_1 = abs(target_1 - entry_zone_optimal)
        reward_2 = abs(target_2 - entry_zone_optimal)
        reward_3 = abs(target_3 - entry_zone_optimal)
        
        rr_ratio_1 = reward_1 / risk_1 if risk_1 > 0 else 0
        rr_ratio_2 = reward_2 / risk_2 if risk_2 > 0 else 0
        rr_ratio_3 = reward_3 / risk_3 if risk_3 > 0 else 0
        
        return PullbackAnalysis(
            trend_strength=trend_strength,
            pullback_probability=pullback_probability,
            pullback_target_1=pullback_target_1,
            pullback_target_2=pullback_target_2,
            pullback_target_3=pullback_target_3,
            entry_zone_optimal=entry_zone_optimal,
            entry_zone_aggressive=entry_zone_aggressive,
            entry_zone_conservative=entry_zone_conservative,
            stop_loss_tight=stop_loss_tight,
            stop_loss_normal=stop_loss_normal,
            stop_loss_wide=stop_loss_wide,
            target_1=target_1,
            target_2=target_2,
            target_3=target_3,
            rr_ratio_1=rr_ratio_1,
            rr_ratio_2=rr_ratio_2,
            rr_ratio_3=rr_ratio_3
        )
    
    # Helper methods
    def _determine_ma_alignment(self, price: float, sma20: float, sma50: float, sma200: float) -> str:
        """Determine moving average alignment"""
        if price > sma20 > sma50 > sma200:
            return "strong_bullish"
        elif price > sma20 > sma50:
            return "bullish"
        elif price < sma20 < sma50 < sma200:
            return "strong_bearish"
        elif price < sma20 < sma50:
            return "bearish"
        else:
            return "mixed"
    
    def _interpret_rsi(self, rsi: float) -> str:
        """Interpret RSI reading with clearer thresholds"""
        if rsi >= 75:
            return "extremely_overbought"
        elif rsi >= 70:
            return "overbought" 
        elif rsi >= 55:
            return "bullish"
        elif rsi >= 45:
            return "neutral"
        elif rsi >= 30:
            return "bearish"
        elif rsi >= 25:
            return "oversold"
        else:
            return "extremely_oversold"
    
    def _interpret_stochastic(self, k: float, d: float) -> str:
        """Interpret Stochastic reading"""
        if k >= 80 and d >= 80:
            return "overbought"
        elif k <= 20 and d <= 20:
            return "oversold"
        elif k > d:
            return "bullish_cross"
        elif k < d:
            return "bearish_cross"
        else:
            return "neutral"
    
    def _interpret_macd(self, macd_line: float, signal: float, histogram: float) -> str:
        """Interpret MACD trend"""
        if macd_line > signal and histogram > 0:
            return "strong_bullish"
        elif macd_line > signal:
            return "bullish"
        elif macd_line < signal and histogram < 0:
            return "strong_bearish"
        elif macd_line < signal:
            return "bearish"
        else:
            return "neutral"
    
    def _detect_bb_squeeze(self, bb_upper: pd.Series, bb_lower: pd.Series) -> bool:
        """Detect Bollinger Band squeeze"""
        current_width = bb_upper.iloc[-1] - bb_lower.iloc[-1]
        avg_width = (bb_upper - bb_lower).tail(20).mean()
        return current_width < (avg_width * 0.8)
    
    def _interpret_volume_trend(self, volume: pd.Series, volume_sma: pd.Series) -> str:
        """Interpret volume trend"""
        recent_avg = volume.tail(5).mean()
        older_avg = volume.tail(20).head(15).mean()
        
        if recent_avg > older_avg * 1.2:
            return "increasing"
        elif recent_avg < older_avg * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_volatility_percentile(self, atr: pd.Series) -> float:
        """Calculate where current volatility sits historically"""
        current_atr = atr.iloc[-1]
        historical_atr = atr.tail(252)  # 1 year
        return (historical_atr < current_atr).sum() / len(historical_atr) * 100
    
    def _identify_current_fib_level(self, price: float, fib_levels: List[float]) -> str:
        """Identify which Fibonacci level price is near"""
        tolerance = 0.02  # 2% tolerance
        
        for i, level in enumerate(fib_levels):
            if abs(price - level) / level < tolerance:
                fib_names = ["23.6%", "38.2%", "50%", "61.8%", "78.6%"]
                return fib_names[i]
        
        return "between_levels"
    
    def _get_next_fib_target(self, price: float, fib_levels: List[float], trend: str) -> float:
        """Get next Fibonacci target"""
        if trend == "up":
            targets = [f for f in fib_levels if f > price]
            return min(targets) if targets else max(fib_levels)
        else:
            targets = [f for f in fib_levels if f < price]
            return max(targets) if targets else min(fib_levels)
    
    def _calculate_trend_strength(self, tech_metrics: AdvancedTechnicalMetrics) -> float:
        """Calculate trend strength on 0-100 scale"""
        strength = 0
        
        # MA alignment (30 points max)
        if tech_metrics.ma_alignment == "strong_bullish":
            strength += 30
        elif tech_metrics.ma_alignment == "bullish":
            strength += 20
        elif tech_metrics.ma_alignment == "strong_bearish":
            strength += 30  # Strong trend regardless of direction
        elif tech_metrics.ma_alignment == "bearish":
            strength += 20
        
        # MA slopes (20 points max)
        if abs(tech_metrics.ma_slope_20) > 2:
            strength += 10
        if abs(tech_metrics.ma_slope_50) > 1:
            strength += 10
        
        # MACD (20 points max)
        if tech_metrics.macd_trend in ["strong_bullish", "strong_bearish"]:
            strength += 20
        elif tech_metrics.macd_trend in ["bullish", "bearish"]:
            strength += 10
        
        # RSI (15 points max)
        if 30 < tech_metrics.rsi_14 < 70:  # Not overbought/oversold
            strength += 15
        elif tech_metrics.rsi_14 > 50:  # Bullish bias
            strength += 10
        
        # Volume (15 points max)
        if tech_metrics.volume_ratio > 1.5:
            strength += 15
        elif tech_metrics.volume_ratio > 1.2:
            strength += 10
        
        return min(strength, 100)
    
    def _calculate_pullback_probability(self, tech_metrics: AdvancedTechnicalMetrics, fib_analysis: FibonacciAnalysis) -> float:
        """Calculate probability of pullback"""
        probability = 0
        
        # RSI overbought/oversold (30 points max)
        if tech_metrics.rsi_14 >= 70:
            probability += 30
        elif tech_metrics.rsi_14 >= 60:
            probability += 15
        elif tech_metrics.rsi_14 <= 30:
            probability += 30
        elif tech_metrics.rsi_14 <= 40:
            probability += 15
        
        # Stochastic (20 points max)  
        if "overbought" in tech_metrics.stochastic_signal:
            probability += 20
        elif "oversold" in tech_metrics.stochastic_signal:
            probability += 20
        
        # Bollinger Band position (20 points max)
        if tech_metrics.bb_position > 90:
            probability += 20
        elif tech_metrics.bb_position < 10:
            probability += 20
        elif tech_metrics.bb_position > 80:
            probability += 10
        elif tech_metrics.bb_position < 20:
            probability += 10
        
        # Distance from moving averages (15 points max)
        price_vs_sma20 = abs(tech_metrics.current_price - tech_metrics.sma_20) / tech_metrics.sma_20
        if price_vs_sma20 > 0.05:  # More than 5% from 20 SMA
            probability += 15
        elif price_vs_sma20 > 0.03:
            probability += 10
        
        # Volume divergence (15 points max)
        if tech_metrics.volume_ratio < 0.8:  # Low volume on move
            probability += 15
        elif tech_metrics.volume_ratio < 1.0:
            probability += 10
        
        return min(probability, 100)
    
    def _calculate_signal_strength(self, tech_metrics: AdvancedTechnicalMetrics, fib_analysis: FibonacciAnalysis) -> int:
        """Calculate overall signal strength 1-10"""
        strength = 0
        
        # Trend alignment
        if tech_metrics.ma_alignment in ["strong_bullish", "strong_bearish"]:
            strength += 3
        elif tech_metrics.ma_alignment in ["bullish", "bearish"]:
            strength += 2
        
        # MACD confirmation
        if tech_metrics.macd_trend in ["strong_bullish", "strong_bearish"]:
            strength += 2
        
        # RSI not extreme
        if 40 < tech_metrics.rsi_14 < 60:
            strength += 2
        
        # Volume support
        if tech_metrics.volume_ratio > 1.2:
            strength += 1
        
        # Fibonacci confluence
        if fib_analysis.current_fib_level != "between_levels":
            strength += 2
        
        return min(strength, 10)
    
    def _generate_trade_setup(self, pullback_analysis: PullbackAnalysis, tech_metrics: AdvancedTechnicalMetrics) -> Dict:
        """Generate specific trade setup"""
        return {
            "setup_type": "trend_continuation" if pullback_analysis.trend_strength > 60 else "range_trade",
            "entry_price": pullback_analysis.entry_zone_optimal,
            "stop_loss": pullback_analysis.stop_loss_normal,
            "target_1": pullback_analysis.target_1,
            "target_2": pullback_analysis.target_2,
            "risk_reward": pullback_analysis.rr_ratio_1,
            "confidence": "high" if pullback_analysis.trend_strength > 70 else "medium" if pullback_analysis.trend_strength > 50 else "low"
        }
    
    def _generate_detailed_narrative(self, symbol: str, tech_metrics: AdvancedTechnicalMetrics, fib_analysis: FibonacciAnalysis, pullback_analysis: PullbackAnalysis, conflict_resolution: Dict[str, str]) -> str:
        """Generate detailed technical narrative"""
        
        # Symbol name formatting
        symbol_name = symbol.replace('=X', '').replace('^', '').replace('-USD', '')
        
        narrative = f"📊 **{symbol_name} Advanced Technical Analysis**\n\n"
        
        # Price position
        narrative += f"**Current Price:** {tech_metrics.current_price:.5f} "
        narrative += f"({tech_metrics.price_vs_52w_low:+.1f}% from 52W low, {tech_metrics.price_vs_52w_high:+.1f}% from 52W high)\n\n"
        
        # Moving Average Analysis
        narrative += f"**Moving Average Analysis:**\n"
        narrative += f"• Alignment: {tech_metrics.ma_alignment.replace('_', ' ').title()}\n"
        narrative += f"• SMA20: {tech_metrics.sma_20:.5f} (slope: {tech_metrics.ma_slope_20:+.2f}%)\n"
        narrative += f"• SMA50: {tech_metrics.sma_50:.5f} (slope: {tech_metrics.ma_slope_50:+.2f}%)\n"
        narrative += f"• Price vs SMA20: {((tech_metrics.current_price - tech_metrics.sma_20) / tech_metrics.sma_20 * 100):+.2f}%\n\n"
        
        # Oscillator Analysis
        narrative += f"**Oscillator Readings:**\n"
        narrative += f"• RSI(14): {tech_metrics.rsi_14:.1f} - {tech_metrics.rsi_signal.replace('_', ' ').title()}\n"
        narrative += f"• Stochastic: K={tech_metrics.stochastic_k:.1f}, D={tech_metrics.stochastic_d:.1f} - {tech_metrics.stochastic_signal.replace('_', ' ').title()}\n"
        narrative += f"• Williams %R: {tech_metrics.williams_r:.1f}\n"
        narrative += f"• CCI(20): {tech_metrics.cci_20:.1f}\n\n"
        
        # MACD Analysis
        narrative += f"**MACD Analysis:**\n"
        narrative += f"• MACD Line: {tech_metrics.macd_line:.6f}\n"
        narrative += f"• Signal Line: {tech_metrics.macd_signal:.6f}\n"
        narrative += f"• Histogram: {tech_metrics.macd_histogram:.6f}\n"
        narrative += f"• Trend: {tech_metrics.macd_trend.replace('_', ' ').title()}\n\n"
        
        # ⚡ COHERENT ASSESSMENT (Resolves Conflicts)
        narrative += f"**📋 Coherent Technical Assessment:**\n"
        narrative += f"• Overall Bias: {conflict_resolution['overall_bias'].title()}\n"
        narrative += f"• Momentum Assessment: {conflict_resolution['momentum_assessment'].replace('_', ' ').title()}\n"
        narrative += f"• Extremes Status: {conflict_resolution['extremes_status'].title()}\n"
        narrative += f"• Signal Consensus: {conflict_resolution['bullish_signals']} Bullish vs {conflict_resolution['bearish_signals']} Bearish signals\n"
        narrative += f"• Signal Strength: {conflict_resolution['signal_strength']}/5\n\n"
        
        # Bollinger Bands
        narrative += f"**Bollinger Bands:**\n"
        narrative += f"• Upper: {tech_metrics.bb_upper:.5f}\n"
        narrative += f"• Middle: {tech_metrics.bb_middle:.5f}\n"
        narrative += f"• Lower: {tech_metrics.bb_lower:.5f}\n"
        narrative += f"• Position: {tech_metrics.bb_position:.1f}% (0=lower band, 100=upper band)\n"
        narrative += f"• Squeeze: {'Yes' if tech_metrics.bb_squeeze else 'No'}\n\n"
        
        # Fibonacci Analysis
        narrative += f"**Fibonacci Analysis:**\n"
        narrative += f"• Trend Direction: {fib_analysis.trend_direction.title()}\n"
        narrative += f"• Swing High: {fib_analysis.swing_high:.5f}\n"
        narrative += f"• Swing Low: {fib_analysis.swing_low:.5f}\n"
        narrative += f"• Current Level: {fib_analysis.current_fib_level}\n"
        narrative += f"• Key Levels:\n"
        narrative += f"  - 23.6%: {fib_analysis.fib_236:.5f}\n"
        narrative += f"  - 38.2%: {fib_analysis.fib_382:.5f}\n"
        narrative += f"  - 50.0%: {fib_analysis.fib_500:.5f}\n"
        narrative += f"  - 61.8%: {fib_analysis.fib_618:.5f}\n"
        narrative += f"• Next Target: {fib_analysis.next_fib_target:.5f}\n\n"
        
        # Pullback Analysis
        narrative += f"**Pullback Analysis:**\n"
        narrative += f"• Trend Strength: {pullback_analysis.trend_strength:.0f}/100\n"
        narrative += f"• Pullback Probability: {pullback_analysis.pullback_probability:.0f}%\n"
        narrative += f"• Pullback Targets:\n"
        narrative += f"  - Target 1 (23.6%): {pullback_analysis.pullback_target_1:.5f}\n"
        narrative += f"  - Target 2 (38.2%): {pullback_analysis.pullback_target_2:.5f}\n"
        narrative += f"  - Target 3 (50.0%): {pullback_analysis.pullback_target_3:.5f}\n\n"
        
        # Entry Zones
        narrative += f"**Entry Zones for Trend Continuation:**\n"
        narrative += f"• Optimal Entry: {pullback_analysis.entry_zone_optimal:.5f}\n"
        narrative += f"• Aggressive Entry: {pullback_analysis.entry_zone_aggressive:.5f}\n"
        narrative += f"• Conservative Entry: {pullback_analysis.entry_zone_conservative:.5f}\n"
        narrative += f"• Stop Loss (Normal): {pullback_analysis.stop_loss_normal:.5f}\n"
        narrative += f"• Target 1: {pullback_analysis.target_1:.5f} (R:R = {pullback_analysis.rr_ratio_1:.2f})\n"
        narrative += f"• Target 2: {pullback_analysis.target_2:.5f} (R:R = {pullback_analysis.rr_ratio_2:.2f})\n\n"
        
        # Support/Resistance
        narrative += f"**Key Levels:**\n"
        narrative += f"• Immediate Support: {tech_metrics.immediate_support:.5f}\n"
        narrative += f"• Immediate Resistance: {tech_metrics.immediate_resistance:.5f}\n"
        narrative += f"• Major Support: {tech_metrics.major_support:.5f}\n"
        narrative += f"• Major Resistance: {tech_metrics.major_resistance:.5f}\n\n"
        
        # Volume & Volatility
        narrative += f"**Volume & Volatility:**\n"
        narrative += f"• Volume Ratio: {tech_metrics.volume_ratio:.2f}x average\n"
        narrative += f"• Volume Trend: {tech_metrics.volume_trend.replace('_', ' ').title()}\n"
        narrative += f"• ATR(14): {tech_metrics.atr_14:.5f}\n"
        narrative += f"• Volatility Percentile: {tech_metrics.volatility_percentile:.0f}%\n"
        
        return narrative

    def _resolve_indicator_conflicts(self, tech_metrics: AdvancedTechnicalMetrics) -> Dict[str, str]:
        """Resolve conflicts between indicators and provide coherent assessment"""
        
        # Count bullish vs bearish signals
        bullish_signals = 0
        bearish_signals = 0
        
        # RSI analysis
        if "bullish" in tech_metrics.rsi_signal or tech_metrics.rsi_14 > 50:
            bullish_signals += 1
        elif "bearish" in tech_metrics.rsi_signal or tech_metrics.rsi_14 < 50:
            bearish_signals += 1
            
        # Stochastic analysis
        if "bullish" in tech_metrics.stochastic_signal:
            bullish_signals += 1
        elif "bearish" in tech_metrics.stochastic_signal or "overbought" in tech_metrics.stochastic_signal:
            bearish_signals += 1
            
        # Williams %R analysis (inverted scale)
        if tech_metrics.williams_r > -50:  # Above -50 is bullish
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        # MACD analysis
        if "bullish" in tech_metrics.macd_trend:
            bullish_signals += 1
        elif "bearish" in tech_metrics.macd_trend:
            bearish_signals += 1
            
        # Moving average analysis
        if "bullish" in tech_metrics.ma_alignment:
            bullish_signals += 2  # Double weight for trend
        elif "bearish" in tech_metrics.ma_alignment:
            bearish_signals += 2
            
        # Determine overall momentum
        if bullish_signals > bearish_signals + 1:
            overall_bias = "bullish"
        elif bearish_signals > bullish_signals + 1:
            overall_bias = "bearish"
        else:
            overall_bias = "neutral"
            
        # Check for overbought/oversold conditions
        overbought_signals = 0
        oversold_signals = 0
        
        if tech_metrics.rsi_14 >= 70:
            overbought_signals += 1
        elif tech_metrics.rsi_14 <= 30:
            oversold_signals += 1
            
        if tech_metrics.stochastic_k >= 80 and tech_metrics.stochastic_d >= 80:
            overbought_signals += 1
        elif tech_metrics.stochastic_k <= 20 and tech_metrics.stochastic_d <= 20:
            oversold_signals += 1
            
        if tech_metrics.williams_r >= -20:  # Very overbought
            overbought_signals += 1
        elif tech_metrics.williams_r <= -80:  # Very oversold
            oversold_signals += 1
            
        # Determine overbought/oversold status
        if overbought_signals >= 2:
            extremes_status = "overbought"
        elif oversold_signals >= 2:
            extremes_status = "oversold"
        else:
            extremes_status = "normal"
            
        # Create coherent assessment
        if extremes_status == "overbought":
            if overall_bias == "bullish":
                momentum_assessment = "bullish_but_overbought"
            else:
                momentum_assessment = "overbought_reversal_risk"
        elif extremes_status == "oversold":
            if overall_bias == "bearish":
                momentum_assessment = "bearish_but_oversold"
            else:
                momentum_assessment = "oversold_bounce_potential"
        else:
            momentum_assessment = f"{overall_bias}_momentum"
            
        return {
            "overall_bias": overall_bias,
            "extremes_status": extremes_status,
            "momentum_assessment": momentum_assessment,
            "bullish_signals": str(bullish_signals),
            "bearish_signals": str(bearish_signals),
            "signal_strength": str(abs(bullish_signals - bearish_signals))
        }

# Usage example
if __name__ == "__main__":
    analyzer = AdvancedTechnicalAnalyzer()
    
    # Test with EURUSD
    result = analyzer.get_comprehensive_analysis("EURUSD=X")
    
    if "error" not in result:
        print(result["detailed_narrative"])
        print(f"\nSignal Strength: {result['signal_strength']}/10")
        print(f"Trade Setup: {result['trade_setup']}")
    else:
        print(f"Error: {result['error']}") 