#!/usr/bin/env python3
"""
QNTI Smart Money Concepts (SMC) Analyzer
Converted from LuxAlgo Pine Script to Python

This module provides comprehensive Smart Money Concepts analysis including:
- Market Structure (BOS/CHoCH)
- Order Blocks
- Fair Value Gaps (FVGs)
- Equal Highs/Lows
- Premium/Discount Zones
- Swing Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TrendDirection(Enum):
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0

class StructureType(Enum):
    BOS = "BOS"  # Break of Structure
    CHOCH = "CHoCH"  # Change of Character

class OrderBlockType(Enum):
    BULLISH = "Bullish"
    BEARISH = "Bearish"

@dataclass
class Pivot:
    """Represents a pivot point (swing high/low)"""
    current_level: float = None
    last_level: float = None
    crossed: bool = False
    bar_time: pd.Timestamp = None
    bar_index: int = None

@dataclass
class OrderBlock:
    """Represents an order block"""
    high: float
    low: float
    timestamp: pd.Timestamp
    bias: OrderBlockType
    mitigated: bool = False
    
@dataclass
class FairValueGap:
    """Represents a Fair Value Gap"""
    top: float
    bottom: float
    timestamp: pd.Timestamp
    bias: TrendDirection
    filled: bool = False

@dataclass
class StructureBreakout:
    """Represents a market structure breakout"""
    level: float
    timestamp: pd.Timestamp
    structure_type: StructureType
    direction: TrendDirection

@dataclass
class SMCResult:
    """Complete SMC analysis result"""
    # Market Structure
    swing_high: Pivot = field(default_factory=Pivot)
    swing_low: Pivot = field(default_factory=Pivot)
    internal_high: Pivot = field(default_factory=Pivot)
    internal_low: Pivot = field(default_factory=Pivot)
    
    # Trends
    swing_trend: TrendDirection = TrendDirection.NEUTRAL
    internal_trend: TrendDirection = TrendDirection.NEUTRAL
    
    # Order Blocks
    swing_order_blocks: List[OrderBlock] = field(default_factory=list)
    internal_order_blocks: List[OrderBlock] = field(default_factory=list)
    
    # Fair Value Gaps
    fair_value_gaps: List[FairValueGap] = field(default_factory=list)
    
    # Structure Breakouts
    structure_breakouts: List[StructureBreakout] = field(default_factory=list)
    
    # Equal Highs/Lows
    equal_highs: List[float] = field(default_factory=list)
    equal_lows: List[float] = field(default_factory=list)
    
    # Premium/Discount Zones
    premium_zone: Tuple[float, float] = None  # (top, bottom)
    equilibrium_zone: Tuple[float, float] = None
    discount_zone: Tuple[float, float] = None
    
    # Alerts
    alerts: Dict[str, bool] = field(default_factory=dict)

class SmartMoneyConcepts:
    """
    Smart Money Concepts Analyzer for QNTI Trading System
    """
    
    def __init__(self, 
                 swing_length: int = 50,
                 internal_length: int = 5,
                 equal_hl_threshold: float = 0.1,
                 equal_hl_confirmation: int = 3,
                 order_block_count: int = 5,
                 atr_period: int = 200):
        
        self.swing_length = swing_length
        self.internal_length = internal_length
        self.equal_hl_threshold = equal_hl_threshold
        self.equal_hl_confirmation = equal_hl_confirmation
        self.order_block_count = order_block_count
        self.atr_period = atr_period
        
        # Initialize state
        self.reset_state()
        
    def reset_state(self):
        """Reset internal state"""
        self.result = SMCResult()
        self.parsed_highs = []
        self.parsed_lows = []
        self.highs = []
        self.lows = []
        self.timestamps = []
        
    def analyze(self, df: pd.DataFrame) -> SMCResult:
        """
        Main analysis function
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume, timestamp)
            
        Returns:
            SMCResult containing all SMC analysis
        """
        try:
            if len(df) < max(self.swing_length, self.atr_period):
                logger.warning(f"Insufficient data for SMC analysis. Need at least {max(self.swing_length, self.atr_period)} bars")
                return self.result
                
            # Reset state for new analysis
            self.reset_state()
            
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"DataFrame must contain columns: {required_cols}")
                
            # Copy and prepare data
            self.df = df.copy()
            if 'timestamp' not in self.df.columns:
                self.df['timestamp'] = pd.to_datetime(self.df.index)
                
            # Calculate technical indicators
            self._calculate_indicators()
            
            # Parse volatility data
            self._parse_volatility_data()
            
            # Analyze market structure
            self._analyze_swing_structure()
            self._analyze_internal_structure()
            
            # Detect order blocks
            self._detect_order_blocks()
            
            # Detect fair value gaps
            self._detect_fair_value_gaps()
            
            # Detect equal highs/lows
            self._detect_equal_highs_lows()
            
            # Calculate premium/discount zones
            self._calculate_premium_discount_zones()
            
            # Check for structure breakouts
            self._check_structure_breakouts()
            
            logger.info(f"SMC Analysis completed for {len(df)} bars")
            return self.result
            
        except Exception as e:
            logger.error(f"Error in SMC analysis: {e}")
            return self.result
    
    def _calculate_indicators(self):
        """Calculate required technical indicators"""
        # ATR for volatility measurement
        self.df['tr'] = np.maximum(
            self.df['high'] - self.df['low'],
            np.maximum(
                abs(self.df['high'] - self.df['close'].shift(1)),
                abs(self.df['low'] - self.df['close'].shift(1))
            )
        )
        self.df['atr'] = self.df['tr'].rolling(window=self.atr_period).mean()
        
        # Cumulative mean range
        self.df['cmr'] = self.df['tr'].expanding().mean()
        
    def _parse_volatility_data(self):
        """Parse data based on volatility to filter out noise"""
        # Identify high volatility bars
        volatility_threshold = 2 * self.df['atr']
        high_vol_bars = (self.df['high'] - self.df['low']) >= volatility_threshold
        
        # For high volatility bars, use inverted values to filter extremes
        self.parsed_highs = np.where(high_vol_bars, self.df['low'], self.df['high'])
        self.parsed_lows = np.where(high_vol_bars, self.df['high'], self.df['low'])
        
        # Store regular values too
        self.highs = self.df['high'].values
        self.lows = self.df['low'].values
        self.timestamps = self.df['timestamp'].values
        
    def _get_leg(self, size: int, index: int) -> int:
        """
        Determine the current leg direction
        
        Args:
            size: lookback period
            index: current index
            
        Returns:
            1 for bullish leg, 0 for bearish leg
        """
        if index < size:
            return 0
            
        lookback_start = max(0, index - size)
        
        highest_in_period = np.max(self.df['high'].iloc[lookback_start:index])
        lowest_in_period = np.min(self.df['low'].iloc[lookback_start:index])
        
        new_leg_high = self.df['high'].iloc[index] > highest_in_period
        new_leg_low = self.df['low'].iloc[index] < lowest_in_period
        
        if new_leg_high:
            return 0  # Bearish leg (new high indicates potential reversal)
        elif new_leg_low:
            return 1  # Bullish leg (new low indicates potential reversal)
        
        return -1  # No change
    
    def _analyze_swing_structure(self):
        """Analyze swing market structure"""
        leg_changes = []
        current_leg = 0
        
        for i in range(len(self.df)):
            leg = self._get_leg(self.swing_length, i)
            if leg != -1:  # Valid leg change
                if leg != current_leg:
                    leg_changes.append((i, leg, current_leg))
                    current_leg = leg
        
        # Process leg changes to identify swing points
        for i, (index, new_leg, old_leg) in enumerate(leg_changes):
            if new_leg == 1:  # Start of bullish leg (found swing low)
                self.result.swing_low.last_level = self.result.swing_low.current_level
                self.result.swing_low.current_level = self.df['low'].iloc[index]
                self.result.swing_low.crossed = False
                self.result.swing_low.bar_time = self.df['timestamp'].iloc[index]
                self.result.swing_low.bar_index = index
                
            elif new_leg == 0:  # Start of bearish leg (found swing high)
                self.result.swing_high.last_level = self.result.swing_high.current_level
                self.result.swing_high.current_level = self.df['high'].iloc[index]
                self.result.swing_high.crossed = False
                self.result.swing_high.bar_time = self.df['timestamp'].iloc[index]
                self.result.swing_high.bar_index = index
    
    def _analyze_internal_structure(self):
        """Analyze internal market structure (shorter timeframe)"""
        leg_changes = []
        current_leg = 0
        
        for i in range(len(self.df)):
            leg = self._get_leg(self.internal_length, i)
            if leg != -1:
                if leg != current_leg:
                    leg_changes.append((i, leg, current_leg))
                    current_leg = leg
        
        # Process internal leg changes
        for i, (index, new_leg, old_leg) in enumerate(leg_changes):
            if new_leg == 1:  # Internal swing low
                self.result.internal_low.last_level = self.result.internal_low.current_level
                self.result.internal_low.current_level = self.df['low'].iloc[index]
                self.result.internal_low.crossed = False
                self.result.internal_low.bar_time = self.df['timestamp'].iloc[index]
                self.result.internal_low.bar_index = index
                
            elif new_leg == 0:  # Internal swing high
                self.result.internal_high.last_level = self.result.internal_high.current_level
                self.result.internal_high.current_level = self.df['high'].iloc[index]
                self.result.internal_high.crossed = False
                self.result.internal_high.bar_time = self.df['timestamp'].iloc[index]
                self.result.internal_high.bar_index = index
    
    def _detect_order_blocks(self):
        """Detect order blocks based on swing points"""
        # Swing order blocks
        if self.result.swing_high.current_level is not None:
            self._find_order_blocks_for_pivot(
                self.result.swing_high, 
                OrderBlockType.BEARISH, 
                self.result.swing_order_blocks,
                is_internal=False
            )
        
        if self.result.swing_low.current_level is not None:
            self._find_order_blocks_for_pivot(
                self.result.swing_low, 
                OrderBlockType.BULLISH, 
                self.result.swing_order_blocks,
                is_internal=False
            )
        
        # Internal order blocks
        if self.result.internal_high.current_level is not None:
            self._find_order_blocks_for_pivot(
                self.result.internal_high, 
                OrderBlockType.BEARISH, 
                self.result.internal_order_blocks,
                is_internal=True
            )
        
        if self.result.internal_low.current_level is not None:
            self._find_order_blocks_for_pivot(
                self.result.internal_low, 
                OrderBlockType.BULLISH, 
                self.result.internal_order_blocks,
                is_internal=True
            )
    
    def _find_order_blocks_for_pivot(self, pivot: Pivot, bias: OrderBlockType, 
                                   storage: List[OrderBlock], is_internal: bool = False):
        """Find order blocks around a pivot point"""
        if pivot.bar_index is None:
            return
            
        start_idx = max(0, pivot.bar_index - 10)
        end_idx = min(len(self.df), pivot.bar_index + 1)
        
        if bias == OrderBlockType.BEARISH:
            # Find the highest bar in the range for bearish OB
            search_range = self.parsed_highs[start_idx:end_idx]
            if len(search_range) > 0:
                max_idx = start_idx + np.argmax(search_range)
                ob = OrderBlock(
                    high=self.df['high'].iloc[max_idx],
                    low=self.df['low'].iloc[max_idx],
                    timestamp=self.df['timestamp'].iloc[max_idx],
                    bias=bias
                )
                storage.append(ob)
        else:
            # Find the lowest bar in the range for bullish OB
            search_range = self.parsed_lows[start_idx:end_idx]
            if len(search_range) > 0:
                min_idx = start_idx + np.argmin(search_range)
                ob = OrderBlock(
                    high=self.df['high'].iloc[min_idx],
                    low=self.df['low'].iloc[min_idx],
                    timestamp=self.df['timestamp'].iloc[min_idx],
                    bias=bias
                )
                storage.append(ob)
        
        # Keep only the most recent order blocks
        if len(storage) > self.order_block_count:
            storage[:] = storage[-self.order_block_count:]
    
    def _detect_fair_value_gaps(self):
        """Detect Fair Value Gaps (FVGs)"""
        for i in range(2, len(self.df)):
            current_high = self.df['high'].iloc[i]
            current_low = self.df['low'].iloc[i]
            prev_high = self.df['high'].iloc[i-1]
            prev_low = self.df['low'].iloc[i-1]
            prev2_high = self.df['high'].iloc[i-2]
            prev2_low = self.df['low'].iloc[i-2]
            
            # Bullish FVG: current low > previous 2 bars high
            if current_low > prev2_high and prev_low > prev2_high:
                fvg = FairValueGap(
                    top=current_low,
                    bottom=prev2_high,
                    timestamp=self.df['timestamp'].iloc[i],
                    bias=TrendDirection.BULLISH
                )
                self.result.fair_value_gaps.append(fvg)
            
            # Bearish FVG: current high < previous 2 bars low
            elif current_high < prev2_low and prev_high < prev2_low:
                fvg = FairValueGap(
                    top=prev2_low,
                    bottom=current_high,
                    timestamp=self.df['timestamp'].iloc[i],
                    bias=TrendDirection.BEARISH
                )
                self.result.fair_value_gaps.append(fvg)
    
    def _detect_equal_highs_lows(self):
        """Detect Equal Highs and Equal Lows"""
        if len(self.df) < self.equal_hl_confirmation:
            return
            
        # Find potential equal highs
        for i in range(self.equal_hl_confirmation, len(self.df)):
            current_high = self.df['high'].iloc[i]
            
            # Look back for similar highs
            lookback_start = max(0, i - 50)  # Look back 50 bars
            for j in range(lookback_start, i - self.equal_hl_confirmation):
                compare_high = self.df['high'].iloc[j]
                
                # Check if highs are approximately equal
                if abs(current_high - compare_high) < (self.equal_hl_threshold * self.df['atr'].iloc[i]):
                    # Confirm with required number of bars
                    confirmed = True
                    for k in range(1, self.equal_hl_confirmation):
                        if i + k >= len(self.df):
                            confirmed = False
                            break
                    
                    if confirmed and current_high not in self.result.equal_highs:
                        self.result.equal_highs.append(current_high)
                        self.result.alerts['equal_highs'] = True
        
        # Find potential equal lows
        for i in range(self.equal_hl_confirmation, len(self.df)):
            current_low = self.df['low'].iloc[i]
            
            # Look back for similar lows
            lookback_start = max(0, i - 50)
            for j in range(lookback_start, i - self.equal_hl_confirmation):
                compare_low = self.df['low'].iloc[j]
                
                if abs(current_low - compare_low) < (self.equal_hl_threshold * self.df['atr'].iloc[i]):
                    confirmed = True
                    for k in range(1, self.equal_hl_confirmation):
                        if i + k >= len(self.df):
                            confirmed = False
                            break
                    
                    if confirmed and current_low not in self.result.equal_lows:
                        self.result.equal_lows.append(current_low)
                        self.result.alerts['equal_lows'] = True
    
    def _calculate_premium_discount_zones(self):
        """Calculate premium and discount zones based on swing extremes"""
        if (self.result.swing_high.current_level is not None and 
            self.result.swing_low.current_level is not None):
            
            swing_high = self.result.swing_high.current_level
            swing_low = self.result.swing_low.current_level
            
            # Premium zone: 95-100% of range
            premium_bottom = 0.95 * swing_high + 0.05 * swing_low
            self.result.premium_zone = (swing_high, premium_bottom)
            
            # Equilibrium zone: 47.5-52.5% of range
            eq_top = 0.525 * swing_high + 0.475 * swing_low
            eq_bottom = 0.525 * swing_low + 0.475 * swing_high
            self.result.equilibrium_zone = (eq_top, eq_bottom)
            
            # Discount zone: 0-5% of range
            discount_top = 0.95 * swing_low + 0.05 * swing_high
            self.result.discount_zone = (discount_top, swing_low)
    
    def _check_structure_breakouts(self):
        """Check for structure breakouts and classify as BOS or CHoCH"""
        current_close = self.df['close'].iloc[-1]
        
        # Check swing high breakout
        if (self.result.swing_high.current_level is not None and 
            not self.result.swing_high.crossed):
            
            if current_close > self.result.swing_high.current_level:
                structure_type = (StructureType.CHOCH if self.result.swing_trend == TrendDirection.BEARISH 
                                else StructureType.BOS)
                
                breakout = StructureBreakout(
                    level=self.result.swing_high.current_level,
                    timestamp=self.df['timestamp'].iloc[-1],
                    structure_type=structure_type,
                    direction=TrendDirection.BULLISH
                )
                self.result.structure_breakouts.append(breakout)
                self.result.swing_high.crossed = True
                self.result.swing_trend = TrendDirection.BULLISH
                
                # Set alerts
                if structure_type == StructureType.BOS:
                    self.result.alerts['swing_bullish_bos'] = True
                else:
                    self.result.alerts['swing_bullish_choch'] = True
        
        # Check swing low breakout
        if (self.result.swing_low.current_level is not None and 
            not self.result.swing_low.crossed):
            
            if current_close < self.result.swing_low.current_level:
                structure_type = (StructureType.CHOCH if self.result.swing_trend == TrendDirection.BULLISH 
                                else StructureType.BOS)
                
                breakout = StructureBreakout(
                    level=self.result.swing_low.current_level,
                    timestamp=self.df['timestamp'].iloc[-1],
                    structure_type=structure_type,
                    direction=TrendDirection.BEARISH
                )
                self.result.structure_breakouts.append(breakout)
                self.result.swing_low.crossed = True
                self.result.swing_trend = TrendDirection.BEARISH
                
                # Set alerts
                if structure_type == StructureType.BOS:
                    self.result.alerts['swing_bearish_bos'] = True
                else:
                    self.result.alerts['swing_bearish_choch'] = True
    
    def get_summary(self) -> Dict:
        """Get a summary of the SMC analysis"""
        return {
            'swing_trend': self.result.swing_trend.name,
            'internal_trend': self.result.internal_trend.name,
            'swing_high': self.result.swing_high.current_level,
            'swing_low': self.result.swing_low.current_level,
            'order_blocks_count': {
                'swing': len(self.result.swing_order_blocks),
                'internal': len(self.result.internal_order_blocks)
            },
            'fair_value_gaps_count': len(self.result.fair_value_gaps),
            'equal_highs_count': len(self.result.equal_highs),
            'equal_lows_count': len(self.result.equal_lows),
            'structure_breakouts_count': len(self.result.structure_breakouts),
            'zones': {
                'premium': self.result.premium_zone,
                'equilibrium': self.result.equilibrium_zone,
                'discount': self.result.discount_zone
            },
            'alerts': self.result.alerts
        }
    
    def get_trading_signals(self) -> Dict:
        """Get trading signals based on SMC analysis"""
        signals = {
            'trend_direction': self.result.swing_trend.name,
            'strength': 'NEUTRAL',
            'key_levels': [],
            'order_blocks': [],
            'fair_value_gaps': [],
            'structure_signals': []
        }
        
        # Determine trend strength
        if (len(self.result.structure_breakouts) > 0 and 
            self.result.structure_breakouts[-1].structure_type == StructureType.BOS):
            signals['strength'] = 'STRONG'
        elif len(self.result.structure_breakouts) > 0:
            signals['strength'] = 'WEAK'
        
        # Add key levels
        if self.result.swing_high.current_level is not None:
            signals['key_levels'].append({
                'level': self.result.swing_high.current_level,
                'type': 'resistance',
                'importance': 'high'
            })
        
        if self.result.swing_low.current_level is not None:
            signals['key_levels'].append({
                'level': self.result.swing_low.current_level,
                'type': 'support',
                'importance': 'high'
            })
        
        # Add recent order blocks
        for ob in self.result.swing_order_blocks[-3:]:  # Last 3 order blocks
            signals['order_blocks'].append({
                'high': ob.high,
                'low': ob.low,
                'bias': ob.bias.value,
                'timestamp': ob.timestamp.isoformat() if hasattr(ob.timestamp, 'isoformat') else str(ob.timestamp)
            })
        
        # Add unfilled fair value gaps
        for fvg in self.result.fair_value_gaps:
            if not fvg.filled:
                signals['fair_value_gaps'].append({
                    'top': fvg.top,
                    'bottom': fvg.bottom,
                    'bias': fvg.bias.name,
                    'timestamp': fvg.timestamp.isoformat() if hasattr(fvg.timestamp, 'isoformat') else str(fvg.timestamp)
                })
        
        # Add recent structure signals
        for breakout in self.result.structure_breakouts[-2:]:  # Last 2 breakouts
            signals['structure_signals'].append({
                'level': breakout.level,
                'type': breakout.structure_type.value,
                'direction': breakout.direction.name,
                'timestamp': breakout.timestamp.isoformat() if hasattr(breakout.timestamp, 'isoformat') else str(breakout.timestamp)
            })
        
        return signals


def test_smc_analyzer():
    """Test function for SMC analyzer"""
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1H')
    
    # Generate realistic OHLC data
    close_prices = 2000 + np.cumsum(np.random.randn(1000) * 0.5)
    opens = close_prices + np.random.randn(1000) * 0.2
    highs = np.maximum(opens, close_prices) + np.abs(np.random.randn(1000) * 0.3)
    lows = np.minimum(opens, close_prices) - np.abs(np.random.randn(1000) * 0.3)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': close_prices,
        'volume': np.random.randint(100, 1000, 1000)
    })
    
    # Initialize and run SMC analyzer
    smc = SmartMoneyConcepts(
        swing_length=50,
        internal_length=5,
        equal_hl_threshold=0.1,
        order_block_count=5
    )
    
    result = smc.analyze(df)
    summary = smc.get_summary()
    signals = smc.get_trading_signals()
    
    print("=== SMC Analysis Results ===")
    print(f"Swing Trend: {summary['swing_trend']}")
    print(f"Internal Trend: {summary['internal_trend']}")
    print(f"Swing High: {summary['swing_high']}")
    print(f"Swing Low: {summary['swing_low']}")
    print(f"Order Blocks: {summary['order_blocks_count']}")
    print(f"Fair Value Gaps: {summary['fair_value_gaps_count']}")
    print(f"Structure Breakouts: {summary['structure_breakouts_count']}")
    print(f"Alerts: {summary['alerts']}")
    
    print("\n=== Trading Signals ===")
    print(f"Trend Direction: {signals['trend_direction']}")
    print(f"Trend Strength: {signals['strength']}")
    print(f"Key Levels: {len(signals['key_levels'])}")
    print(f"Active Order Blocks: {len(signals['order_blocks'])}")
    print(f"Unfilled FVGs: {len(signals['fair_value_gaps'])}")
    
    return result, summary, signals


if __name__ == "__main__":
    # Run test
    test_smc_analyzer() 