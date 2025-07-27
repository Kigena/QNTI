"""
SMC (Smart Money Concepts) Expert Advisor for QNTI Trading System
Advanced implementation with order blocks, fair value gaps, and market structure analysis
"""

import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from qnti_core import (
    Trade, TradeType, TradeSource, TradingSignal, SignalType,
    QNTITradeManager, QNTIMT5Bridge
)


class SMCSignalType(Enum):
    """SMC-specific signal types"""
    ORDER_BLOCK_LONG = "order_block_long"
    ORDER_BLOCK_SHORT = "order_block_short"
    FVG_LONG = "fvg_long"
    FVG_SHORT = "fvg_short"
    BOS_LONG = "bos_long"
    BOS_SHORT = "bos_short"
    CHOCH_LONG = "choch_long"
    CHOCH_SHORT = "choch_short"
    LIQUIDITY_GRAB_LONG = "liquidity_grab_long"
    LIQUIDITY_GRAB_SHORT = "liquidity_grab_short"


@dataclass
class OrderBlock:
    """Order block structure"""
    start_time: datetime
    end_time: datetime
    high: float
    low: float
    volume: float
    bias: str  # 'bullish' or 'bearish'
    strength: float
    tested: bool = False
    broken: bool = False


@dataclass
class FairValueGap:
    """Fair Value Gap structure"""
    start_time: datetime
    top: float
    bottom: float
    bias: str  # 'bullish' or 'bearish'
    filled: bool = False
    strength: float


@dataclass
class MarketStructure:
    """Market structure analysis"""
    trend: str  # 'bullish', 'bearish', 'neutral'
    last_bos: Optional[datetime] = None
    last_choch: Optional[datetime] = None
    swing_highs: List[Tuple[datetime, float]] = None
    swing_lows: List[Tuple[datetime, float]] = None
    
    def __post_init__(self):
        if self.swing_highs is None:
            self.swing_highs = []
        if self.swing_lows is None:
            self.swing_lows = []


class SMCEA:
    """Smart Money Concepts Expert Advisor"""
    
    def __init__(self):
        # Core components
        self.trade_manager: Optional[QNTITradeManager] = None
        self.qnti_mt5_bridge: Optional[QNTIMT5Bridge] = None
        self.qnti_main_system = None
        
        # EA parameters
        self.lookback_period = 50
        self.swing_length = 5
        self.min_order_block_size = 10  # pips
        self.min_fvg_size = 5  # pips
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.reward_ratio = 2.0  # Risk:Reward 1:2
        self.max_spread = 3.0  # Maximum spread in pips
        
        # Structure sensitivity
        self.structure_sensitivity = "Medium"  # High, Medium, Low
        self.show_order_blocks = True
        self.show_fvg = True
        self.show_liquidity_zones = True
        
        # Trading state
        self.is_monitoring = False
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        self.timeframe = "H1"
        
        # SMC data storage
        self.order_blocks: Dict[str, List[OrderBlock]] = {}
        self.fair_value_gaps: Dict[str, List[FairValueGap]] = {}
        self.market_structure: Dict[str, MarketStructure] = {}
        self.liquidity_zones: Dict[str, List[Dict]] = {}
        
        # Performance tracking
        self.signals_generated = 0
        self.trades_executed = 0
        self.last_signal_time = None
        self.last_analysis_time = None
        
        # Threading
        self.monitor_thread = None
        self.stop_event = threading.Event()
    
    def set_qnti_integration(self, trade_manager: QNTITradeManager, 
                           mt5_bridge: QNTIMT5Bridge, qnti_main_system):
        """Set QNTI integration components"""
        self.trade_manager = trade_manager
        self.qnti_mt5_bridge = mt5_bridge
        self.qnti_main_system = qnti_main_system
        
        # Initialize data structures for symbols
        for symbol in self.symbols:
            self.order_blocks[symbol] = []
            self.fair_value_gaps[symbol] = []
            self.market_structure[symbol] = MarketStructure(trend="neutral")
            self.liquidity_zones[symbol] = []
    
    def update_parameters(self, params: Dict[str, Any]):
        """Update SMC EA parameters"""
        if 'lookback_period' in params:
            self.lookback_period = max(10, min(200, params['lookback_period']))
        if 'swing_length' in params:
            self.swing_length = max(3, min(15, params['swing_length']))
        if 'structure_sensitivity' in params:
            self.structure_sensitivity = params['structure_sensitivity']
        if 'show_order_blocks' in params:
            self.show_order_blocks = params['show_order_blocks']
        if 'show_fvg' in params:
            self.show_fvg = params['show_fvg']
        if 'show_liquidity_zones' in params:
            self.show_liquidity_zones = params['show_liquidity_zones']
        if 'risk_per_trade' in params:
            self.risk_per_trade = max(0.001, min(0.1, params['risk_per_trade']))
        if 'reward_ratio' in params:
            self.reward_ratio = max(1.0, min(5.0, params['reward_ratio']))
        if 'symbols' in params:
            self.symbols = params['symbols']
    
    def get_market_data(self, symbol: str, timeframe: str = "H1", count: int = 500) -> Optional[pd.DataFrame]:
        """Get market data from MT5"""
        try:
            if not self.qnti_mt5_bridge:
                return None
            
            rates = self.qnti_mt5_bridge.get_rates(symbol, timeframe, count)
            if rates is None or len(rates) == 0:
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
        except Exception as e:
            print(f"Error getting market data for {symbol}: {e}")
            return None
    
    def identify_swing_points(self, df: pd.DataFrame) -> Tuple[List[Tuple[datetime, float]], List[Tuple[datetime, float]]]:
        """Identify swing highs and lows"""
        swing_highs = []
        swing_lows = []
        
        if len(df) < self.swing_length * 2 + 1:
            return swing_highs, swing_lows
        
        for i in range(self.swing_length, len(df) - self.swing_length):
            # Check for swing high
            is_swing_high = True
            for j in range(i - self.swing_length, i + self.swing_length + 1):
                if j != i and df.iloc[j]['high'] >= df.iloc[i]['high']:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append((df.index[i], df.iloc[i]['high']))
            
            # Check for swing low
            is_swing_low = True
            for j in range(i - self.swing_length, i + self.swing_length + 1):
                if j != i and df.iloc[j]['low'] <= df.iloc[i]['low']:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append((df.index[i], df.iloc[i]['low']))
        
        return swing_highs, swing_lows
    
    def detect_order_blocks(self, df: pd.DataFrame, swing_highs: List, swing_lows: List) -> List[OrderBlock]:
        """Detect order blocks"""
        order_blocks = []
        
        # Look for bearish order blocks (at swing highs)
        for swing_time, swing_high in swing_highs[-10:]:  # Last 10 swing highs
            # Find the candle that created the swing high
            swing_idx = df.index.get_loc(swing_time)
            
            # Look for the last bullish candle before the swing
            for i in range(swing_idx - 1, max(0, swing_idx - 20), -1):
                candle = df.iloc[i]
                if candle['close'] > candle['open']:  # Bullish candle
                    # Check if this could be an order block
                    block_size = (candle['high'] - candle['low']) * 100000  # Convert to pips
                    
                    if block_size >= self.min_order_block_size:
                        # Check if price has moved away and come back
                        future_low = df.iloc[i+1:swing_idx+10]['low'].min() if i+1 < len(df) else candle['low']
                        
                        if future_low < candle['low'] - block_size * 0.0001:  # Moved away
                            order_block = OrderBlock(
                                start_time=df.index[i],
                                end_time=df.index[i],
                                high=candle['high'],
                                low=candle['low'],
                                volume=candle.get('tick_volume', 1000),
                                bias='bearish',
                                strength=self._calculate_order_block_strength(candle, df.iloc[i:swing_idx+1])
                            )
                            order_blocks.append(order_block)
                    break
        
        # Look for bullish order blocks (at swing lows)
        for swing_time, swing_low in swing_lows[-10:]:  # Last 10 swing lows
            swing_idx = df.index.get_loc(swing_time)
            
            # Look for the last bearish candle before the swing
            for i in range(swing_idx - 1, max(0, swing_idx - 20), -1):
                candle = df.iloc[i]
                if candle['close'] < candle['open']:  # Bearish candle
                    block_size = (candle['high'] - candle['low']) * 100000
                    
                    if block_size >= self.min_order_block_size:
                        future_high = df.iloc[i+1:swing_idx+10]['high'].max() if i+1 < len(df) else candle['high']
                        
                        if future_high > candle['high'] + block_size * 0.0001:
                            order_block = OrderBlock(
                                start_time=df.index[i],
                                end_time=df.index[i],
                                high=candle['high'],
                                low=candle['low'],
                                volume=candle.get('tick_volume', 1000),
                                bias='bullish',
                                strength=self._calculate_order_block_strength(candle, df.iloc[i:swing_idx+1])
                            )
                            order_blocks.append(order_block)
                    break
        
        return order_blocks
    
    def detect_fair_value_gaps(self, df: pd.DataFrame) -> List[FairValueGap]:
        """Detect Fair Value Gaps (FVG)"""
        fair_value_gaps = []
        
        for i in range(2, len(df)):
            candle1 = df.iloc[i-2]  # First candle
            candle2 = df.iloc[i-1]  # Middle candle (gap candle)
            candle3 = df.iloc[i]    # Third candle
            
            # Bullish FVG: candle1.high < candle3.low
            if candle1['high'] < candle3['low']:
                gap_size = (candle3['low'] - candle1['high']) * 100000  # Convert to pips
                
                if gap_size >= self.min_fvg_size:
                    fvg = FairValueGap(
                        start_time=df.index[i-1],
                        top=candle3['low'],
                        bottom=candle1['high'],
                        bias='bullish',
                        strength=gap_size / 10  # Normalize strength
                    )
                    fair_value_gaps.append(fvg)
            
            # Bearish FVG: candle1.low > candle3.high
            elif candle1['low'] > candle3['high']:
                gap_size = (candle1['low'] - candle3['high']) * 100000
                
                if gap_size >= self.min_fvg_size:
                    fvg = FairValueGap(
                        start_time=df.index[i-1],
                        top=candle1['low'],
                        bottom=candle3['high'],
                        bias='bearish',
                        strength=gap_size / 10
                    )
                    fair_value_gaps.append(fvg)
        
        return fair_value_gaps
    
    def detect_break_of_structure(self, df: pd.DataFrame, swing_highs: List, swing_lows: List) -> Dict:
        """Detect Break of Structure (BOS) and Change of Character (CHoCH)"""
        structure_breaks = {
            'bos_signals': [],
            'choch_signals': [],
            'trend': 'neutral'
        }
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return structure_breaks
        
        # Sort swings by time
        all_swings = [(time, price, 'high') for time, price in swing_highs] + \
                     [(time, price, 'low') for time, price in swing_lows]
        all_swings.sort(key=lambda x: x[0])
        
        # Analyze last few swings for structure breaks
        recent_swings = all_swings[-10:]  # Last 10 swings
        
        for i in range(2, len(recent_swings)):
            current_swing = recent_swings[i]
            prev_swing = recent_swings[i-1]
            prev_prev_swing = recent_swings[i-2]
            
            # BOS: Break of previous swing level in trend direction
            if current_swing[2] == 'high' and prev_swing[2] == 'low':
                # Potential bullish BOS
                if current_swing[1] > prev_prev_swing[1] and prev_prev_swing[2] == 'high':
                    structure_breaks['bos_signals'].append({
                        'time': current_swing[0],
                        'type': 'bullish_bos',
                        'price': current_swing[1]
                    })
                    structure_breaks['trend'] = 'bullish'
            
            elif current_swing[2] == 'low' and prev_swing[2] == 'high':
                # Potential bearish BOS
                if current_swing[1] < prev_prev_swing[1] and prev_prev_swing[2] == 'low':
                    structure_breaks['bos_signals'].append({
                        'time': current_swing[0],
                        'type': 'bearish_bos',
                        'price': current_swing[1]
                    })
                    structure_breaks['trend'] = 'bearish'
        
        return structure_breaks
    
    def detect_liquidity_zones(self, df: pd.DataFrame, swing_highs: List, swing_lows: List) -> List[Dict]:
        """Detect liquidity zones around swing points"""
        liquidity_zones = []
        
        # Create liquidity zones around swing highs (potential sell-side liquidity)
        for swing_time, swing_high in swing_highs[-5:]:
            zone = {
                'time': swing_time,
                'type': 'sell_side',
                'price': swing_high,
                'zone_high': swing_high + 0.0010,  # 10 pips above
                'zone_low': swing_high - 0.0005,   # 5 pips below
                'strength': 'medium'
            }
            liquidity_zones.append(zone)
        
        # Create liquidity zones around swing lows (potential buy-side liquidity)
        for swing_time, swing_low in swing_lows[-5:]:
            zone = {
                'time': swing_time,
                'type': 'buy_side',
                'price': swing_low,
                'zone_high': swing_low + 0.0005,   # 5 pips above
                'zone_low': swing_low - 0.0010,    # 10 pips below
                'strength': 'medium'
            }
            liquidity_zones.append(zone)
        
        return liquidity_zones
    
    def _calculate_order_block_strength(self, candle: pd.Series, context_df: pd.DataFrame) -> float:
        """Calculate order block strength based on various factors"""
        base_strength = 1.0
        
        # Volume factor
        avg_volume = context_df.get('tick_volume', pd.Series([1000])).mean()
        volume_factor = min(2.0, candle.get('tick_volume', 1000) / avg_volume)
        
        # Size factor
        candle_size = abs(candle['high'] - candle['low'])
        avg_size = (context_df['high'] - context_df['low']).mean()
        size_factor = min(2.0, candle_size / avg_size)
        
        # Body factor (how much of the candle is body vs wick)
        body_size = abs(candle['close'] - candle['open'])
        body_factor = min(2.0, body_size / candle_size) if candle_size > 0 else 1.0
        
        return base_strength * volume_factor * size_factor * body_factor
    
    def analyze_market_structure(self, symbol: str) -> Optional[TradingSignal]:
        """Perform complete SMC market structure analysis"""
        try:
            df = self.get_market_data(symbol, self.timeframe, self.lookback_period)
            if df is None or len(df) < 20:
                return None
            
            # Get current price
            current_price = df.iloc[-1]['close']
            
            # Identify swing points
            swing_highs, swing_lows = self.identify_swing_points(df)
            
            # Update market structure
            self.market_structure[symbol].swing_highs = swing_highs
            self.market_structure[symbol].swing_lows = swing_lows
            
            # Detect order blocks
            if self.show_order_blocks:
                order_blocks = self.detect_order_blocks(df, swing_highs, swing_lows)
                self.order_blocks[symbol] = order_blocks
            
            # Detect fair value gaps
            if self.show_fvg:
                fvgs = self.detect_fair_value_gaps(df)
                self.fair_value_gaps[symbol] = fvgs
            
            # Detect break of structure
            structure_analysis = self.detect_break_of_structure(df, swing_highs, swing_lows)
            self.market_structure[symbol].trend = structure_analysis['trend']
            
            # Detect liquidity zones
            if self.show_liquidity_zones:
                liquidity_zones = self.detect_liquidity_zones(df, swing_highs, swing_lows)
                self.liquidity_zones[symbol] = liquidity_zones
            
            # Generate trading signal based on SMC analysis
            signal = self._generate_smc_signal(symbol, current_price, df)
            
            self.last_analysis_time = datetime.now()
            
            return signal
            
        except Exception as e:
            print(f"Error analyzing market structure for {symbol}: {e}")
            return None
    
    def _generate_smc_signal(self, symbol: str, current_price: float, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate trading signal based on SMC analysis"""
        try:
            # Check for order block signals
            for ob in self.order_blocks.get(symbol, []):
                if not ob.tested:
                    # Check if price is testing the order block
                    if (ob.bias == 'bullish' and 
                        ob.low <= current_price <= ob.high and 
                        current_price < ob.high):
                        
                        # Confirm with market structure
                        if self.market_structure[symbol].trend in ['bullish', 'neutral']:
                            confidence = min(0.95, 0.7 + (ob.strength / 10))
                            
                            signal = TradingSignal(
                                symbol=symbol,
                                signal_type=SignalType.BUY,
                                price=current_price,
                                timestamp=datetime.now(),
                                confidence=confidence,
                                reason=f"SMC: Bullish Order Block test at {ob.low:.5f}-{ob.high:.5f}",
                                metadata={
                                    'smc_signal_type': SMCSignalType.ORDER_BLOCK_LONG.value,
                                    'order_block_strength': ob.strength,
                                    'stop_loss': ob.low - 0.0010,  # 10 pips below OB
                                    'take_profit': current_price + (current_price - (ob.low - 0.0010)) * self.reward_ratio
                                }
                            )
                            
                            ob.tested = True
                            self.signals_generated += 1
                            self.last_signal_time = datetime.now()
                            
                            return signal
                    
                    elif (ob.bias == 'bearish' and 
                          ob.low <= current_price <= ob.high and 
                          current_price > ob.low):
                        
                        if self.market_structure[symbol].trend in ['bearish', 'neutral']:
                            confidence = min(0.95, 0.7 + (ob.strength / 10))
                            
                            signal = TradingSignal(
                                symbol=symbol,
                                signal_type=SignalType.SELL,
                                price=current_price,
                                timestamp=datetime.now(),
                                confidence=confidence,
                                reason=f"SMC: Bearish Order Block test at {ob.low:.5f}-{ob.high:.5f}",
                                metadata={
                                    'smc_signal_type': SMCSignalType.ORDER_BLOCK_SHORT.value,
                                    'order_block_strength': ob.strength,
                                    'stop_loss': ob.high + 0.0010,  # 10 pips above OB
                                    'take_profit': current_price - ((ob.high + 0.0010) - current_price) * self.reward_ratio
                                }
                            )
                            
                            ob.tested = True
                            self.signals_generated += 1
                            self.last_signal_time = datetime.now()
                            
                            return signal
            
            # Check for FVG signals
            for fvg in self.fair_value_gaps.get(symbol, []):
                if not fvg.filled:
                    # Check if price is in the FVG
                    if fvg.bottom <= current_price <= fvg.top:
                        
                        if (fvg.bias == 'bullish' and 
                            self.market_structure[symbol].trend in ['bullish', 'neutral']):
                            
                            confidence = min(0.9, 0.6 + (fvg.strength / 20))
                            
                            signal = TradingSignal(
                                symbol=symbol,
                                signal_type=SignalType.BUY,
                                price=current_price,
                                timestamp=datetime.now(),
                                confidence=confidence,
                                reason=f"SMC: Bullish FVG fill at {fvg.bottom:.5f}-{fvg.top:.5f}",
                                metadata={
                                    'smc_signal_type': SMCSignalType.FVG_LONG.value,
                                    'fvg_strength': fvg.strength,
                                    'stop_loss': fvg.bottom - 0.0005,
                                    'take_profit': current_price + (current_price - (fvg.bottom - 0.0005)) * self.reward_ratio
                                }
                            )
                            
                            self.signals_generated += 1
                            self.last_signal_time = datetime.now()
                            
                            return signal
                        
                        elif (fvg.bias == 'bearish' and 
                              self.market_structure[symbol].trend in ['bearish', 'neutral']):
                            
                            confidence = min(0.9, 0.6 + (fvg.strength / 20))
                            
                            signal = TradingSignal(
                                symbol=symbol,
                                signal_type=SignalType.SELL,
                                price=current_price,
                                timestamp=datetime.now(),
                                confidence=confidence,
                                reason=f"SMC: Bearish FVG fill at {fvg.bottom:.5f}-{fvg.top:.5f}",
                                metadata={
                                    'smc_signal_type': SMCSignalType.FVG_SHORT.value,
                                    'fvg_strength': fvg.strength,
                                    'stop_loss': fvg.top + 0.0005,
                                    'take_profit': current_price - ((fvg.top + 0.0005) - current_price) * self.reward_ratio
                                }
                            )
                            
                            self.signals_generated += 1
                            self.last_signal_time = datetime.now()
                            
                            return signal
            
            return None
            
        except Exception as e:
            print(f"Error generating SMC signal for {symbol}: {e}")
            return None
    
    def execute_trade(self, signal: TradingSignal) -> bool:
        """Execute trade based on SMC signal"""
        try:
            if not self.trade_manager or not self.qnti_mt5_bridge:
                return False
            
            # Calculate position size based on risk
            account_balance = self.qnti_mt5_bridge.get_account_info().get('balance', 10000)
            risk_amount = account_balance * self.risk_per_trade
            
            # Get symbol info for lot size calculation
            symbol_info = self.qnti_mt5_bridge.get_symbol_info(signal.symbol)
            if not symbol_info:
                return False
            
            # Calculate stop loss and take profit from signal metadata
            stop_loss = signal.metadata.get('stop_loss')
            take_profit = signal.metadata.get('take_profit')
            
            if not stop_loss or not take_profit:
                return False
            
            # Calculate lot size based on risk
            pip_value = symbol_info.get('pip_value', 1.0)
            stop_loss_pips = abs(signal.price - stop_loss) * 100000
            
            if stop_loss_pips > 0:
                lot_size = risk_amount / (stop_loss_pips * pip_value)
                lot_size = max(0.01, min(1.0, round(lot_size, 2)))  # Clamp between 0.01 and 1.0
            else:
                lot_size = 0.01
            
            # Determine trade type
            trade_type = TradeType.BUY if signal.signal_type == SignalType.BUY else TradeType.SELL
            
            # Create trade
            trade = Trade(
                trade_id=f"SMC_{signal.symbol}_{int(time.time())}",
                magic_number=55555,  # SMC EA magic number
                symbol=signal.symbol,
                trade_type=trade_type,
                lot_size=lot_size,
                open_price=signal.price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                source=TradeSource.EXPERT_ADVISOR,
                ea_name="SMC_EA",
                ai_confidence=signal.confidence,
                strategy_tags=["smc", signal.metadata.get('smc_signal_type', 'unknown'), signal.signal_type.value.lower()]
            )
            
            # Execute trade
            success, message = self.qnti_mt5_bridge.execute_trade(trade)
            
            if success:
                # Register trade with trade manager
                self.trade_manager.add_trade(trade)
                self.trades_executed += 1
                print(f"SMC EA: Successfully executed {trade_type.value} trade for {signal.symbol}")
                
                # Notify control panel of new trade
                self._notify_trade_executed(trade, signal)
                
                return True
            else:
                print(f"SMC EA: Failed to execute trade for {signal.symbol}: {message}")
                return False
                
        except Exception as e:
            print(f"SMC EA: Error executing trade for {signal.symbol}: {e}")
            return False
    
    def _notify_trade_executed(self, trade: Trade, signal: TradingSignal):
        """Notify the system that a trade was executed from SMC setup"""
        try:
            # Store trade execution info for the control panel
            if not hasattr(self, 'executed_trades'):
                self.executed_trades = []
            
            trade_info = {
                'trade_id': trade.trade_id,
                'setup_id': signal.metadata.get('setup_id', f"smc_auto_{int(time.time())}"),
                'symbol': trade.symbol,
                'direction': trade.trade_type.value.lower(),
                'signal_type': signal.metadata.get('smc_signal_type', 'unknown'),
                'entry_price': trade.open_price,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'lot_size': trade.lot_size,
                'confidence': signal.confidence,
                'executed_at': datetime.now().isoformat(),
                'auto_executed': True
            }
            
            self.executed_trades.append(trade_info)
            
            # Keep only last 100 executed trades
            if len(self.executed_trades) > 100:
                self.executed_trades = self.executed_trades[-100:]
                
        except Exception as e:
            print(f"SMC EA: Error notifying trade execution: {e}")
    
    def get_trade_setups(self) -> List[Dict]:
        """Get current trade setups for the control panel"""
        setups = []
        
        try:
            current_time = datetime.now()
            
            for symbol in self.symbols:
                # Check for order block setups
                for ob in self.order_blocks.get(symbol, []):
                    if not ob.broken and not ob.tested:
                        # Get current price to check if setup is ready
                        df = self.get_market_data(symbol, self.timeframe, 10)
                        if df is None:
                            continue
                            
                        current_price = df.iloc[-1]['close']
                        
                        # Determine if setup is ready for entry
                        is_ready = False
                        entry_price = current_price
                        direction = 'buy' if ob.bias == 'bullish' else 'sell'
                        
                        if ob.bias == 'bullish' and ob.low <= current_price <= ob.high:
                            is_ready = True
                            entry_price = current_price
                        elif ob.bias == 'bearish' and ob.low <= current_price <= ob.high:
                            is_ready = True
                            entry_price = current_price
                        
                        # Calculate SL and TP
                        if direction == 'buy':
                            stop_loss = ob.low - 0.0010
                            take_profit = entry_price + (entry_price - stop_loss) * self.reward_ratio
                        else:
                            stop_loss = ob.high + 0.0010
                            take_profit = entry_price - (stop_loss - entry_price) * self.reward_ratio
                        
                        setup = {
                            'setup_id': f"ob_{symbol}_{int(ob.start_time.timestamp())}",
                            'symbol': symbol,
                            'direction': direction,
                            'signal_type': 'order_block',
                            'status': 'ready_for_entry' if is_ready else 'analyzing',
                            'confidence': min(0.95, 0.7 + (ob.strength / 10)),
                            'entry_price': f"{entry_price:.5f}",
                            'stop_loss': f"{stop_loss:.5f}",
                            'take_profit': f"{take_profit:.5f}",
                            'risk_reward': f"{self.reward_ratio:.1f}",
                            'created_at': ob.start_time.isoformat(),
                            'expires_at': (ob.start_time + timedelta(hours=8)).isoformat(),
                            'is_new': (current_time - ob.start_time).seconds < 300,  # New if created in last 5 minutes
                            'details': {
                                'ob_range': f"{ob.low:.5f} - {ob.high:.5f}",
                                'bias': ob.bias,
                                'strength': int(ob.strength)
                            }
                        }
                        setups.append(setup)
                
                # Check for FVG setups
                for fvg in self.fair_value_gaps.get(symbol, []):
                    if not fvg.filled:
                        df = self.get_market_data(symbol, self.timeframe, 10)
                        if df is None:
                            continue
                            
                        current_price = df.iloc[-1]['close']
                        
                        # Check if price is in FVG
                        is_ready = fvg.bottom <= current_price <= fvg.top
                        direction = 'buy' if fvg.bias == 'bullish' else 'sell'
                        entry_price = current_price
                        
                        # Calculate SL and TP
                        if direction == 'buy':
                            stop_loss = fvg.bottom - 0.0005
                            take_profit = entry_price + (entry_price - stop_loss) * self.reward_ratio
                        else:
                            stop_loss = fvg.top + 0.0005
                            take_profit = entry_price - (stop_loss - entry_price) * self.reward_ratio
                        
                        gap_size = abs(fvg.top - fvg.bottom) * 100000  # Convert to pips
                        fill_percentage = 0
                        if fvg.bottom <= current_price <= fvg.top:
                            fill_percentage = ((current_price - fvg.bottom) / (fvg.top - fvg.bottom)) * 100
                        
                        setup = {
                            'setup_id': f"fvg_{symbol}_{int(fvg.start_time.timestamp())}",
                            'symbol': symbol,
                            'direction': direction,
                            'signal_type': 'fvg',
                            'status': 'ready_for_entry' if is_ready else 'analyzing',
                            'confidence': min(0.9, 0.6 + (fvg.strength / 20)),
                            'entry_price': f"{entry_price:.5f}",
                            'stop_loss': f"{stop_loss:.5f}",
                            'take_profit': f"{take_profit:.5f}",
                            'risk_reward': f"{self.reward_ratio:.1f}",
                            'created_at': fvg.start_time.isoformat(),
                            'expires_at': (fvg.start_time + timedelta(hours=6)).isoformat(),
                            'is_new': (current_time - fvg.start_time).seconds < 300,
                            'details': {
                                'fvg_range': f"{fvg.bottom:.5f} - {fvg.top:.5f}",
                                'gap_size': int(gap_size),
                                'fill_percentage': int(fill_percentage)
                            }
                        }
                        setups.append(setup)
        
        except Exception as e:
            print(f"SMC EA: Error getting trade setups: {e}")
        
        return setups
    
    def get_market_structure_data(self) -> Dict:
        """Get market structure data for the control panel"""
        structure_data = {}
        
        try:
            for symbol in self.symbols:
                if symbol in self.market_structure:
                    ms = self.market_structure[symbol]
                    
                    structure_data[symbol] = {
                        'trend': ms.trend,
                        'swing_highs': [
                            {
                                'time': sh[0].isoformat() if isinstance(sh[0], datetime) else str(sh[0]),
                                'price': f"{sh[1]:.5f}"
                            }
                            for sh in ms.swing_highs[-5:]  # Last 5 swing highs
                        ],
                        'swing_lows': [
                            {
                                'time': sl[0].isoformat() if isinstance(sl[0], datetime) else str(sl[0]),
                                'price': f"{sl[1]:.5f}"
                            }
                            for sl in ms.swing_lows[-5:]  # Last 5 swing lows
                        ],
                        'last_bos': ms.last_bos.isoformat() if ms.last_bos else None,
                        'last_choch': ms.last_choch.isoformat() if ms.last_choch else None
                    }
        
        except Exception as e:
            print(f"SMC EA: Error getting market structure data: {e}")
        
        return structure_data
    
    def start_monitoring(self, symbols: List[str] = None):
        """Start SMC monitoring"""
        if self.is_monitoring:
            return
        
        if symbols:
            self.symbols = symbols
        
        self.is_monitoring = True
        self.stop_event.clear()
        
        # Initialize data structures for new symbols
        for symbol in self.symbols:
            if symbol not in self.order_blocks:
                self.order_blocks[symbol] = []
                self.fair_value_gaps[symbol] = []
                self.market_structure[symbol] = MarketStructure(trend="neutral")
                self.liquidity_zones[symbol] = []
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        print(f"SMC EA: Started monitoring {len(self.symbols)} symbols")
    
    def stop_monitoring(self):
        """Stop SMC monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        print("SMC EA: Stopped monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        print("SMC EA: Monitoring loop started")
        
        while self.is_monitoring and not self.stop_event.is_set():
            try:
                for symbol in self.symbols:
                    if self.stop_event.is_set():
                        break
                    
                    # Update market structure and generate SMC structures
                    self._update_market_structures(symbol)
                    
                    # Analyze market structure and generate signals
                    signal = self.analyze_market_structure(symbol)
                    
                    if signal and signal.confidence >= 0.6:  # Minimum confidence threshold
                        print(f"SMC EA: Generated {signal.signal_type.value} signal for {symbol} "
                              f"(confidence: {signal.confidence:.2f})")
                        
                        # Execute trade if auto-trading is enabled
                        if hasattr(self, 'auto_trading_enabled') and self.auto_trading_enabled:
                            success = self.execute_trade(signal)
                            if success:
                                print(f"SMC EA: Trade executed successfully for {symbol}")
                            else:
                                print(f"SMC EA: Failed to execute trade for {symbol}")
                    
                    self.last_analysis_time = datetime.now()
                
                # Wait before next analysis cycle
                if not self.stop_event.wait(60):  # 1 minute interval
                    continue
                    
            except Exception as e:
                print(f"SMC EA: Error in monitoring loop: {e}")
                if not self.stop_event.wait(30):  # Wait 30 seconds on error
                    continue
        
        print("SMC EA: Monitoring loop stopped")
    
    def _update_market_structures(self, symbol: str):
        """Update market structures for a symbol with realistic SMC data"""
        try:
            current_time = datetime.now()
            
            # Get some market data (use mock data if MT5 not available)
            df = self.get_market_data(symbol, self.timeframe, self.lookback_period)
            if df is None:
                # Generate mock market structure for demonstration
                self._generate_mock_structures(symbol, current_time)
                return
            
            # Analyze for order blocks
            self._identify_order_blocks(symbol, df)
            
            # Analyze for Fair Value Gaps
            self._identify_fvg(symbol, df)
            
            # Update market structure
            self._update_swing_structure(symbol, df)
            
        except Exception as e:
            print(f"SMC EA: Error updating market structures for {symbol}: {e}")
    
    def _generate_mock_structures(self, symbol: str, current_time: datetime):
        """Generate mock SMC structures for demonstration when real data unavailable"""
        import random
        
        # Generate mock order block
        if random.random() < 0.3:  # 30% chance
            base_price = 1.0850 if symbol == 'EURUSD' else 1.2650 if symbol == 'GBPUSD' else 150.25
            variance = 0.001 if 'USD' in symbol else 0.01
            
            ob_low = base_price + random.uniform(-variance, variance)
            ob_high = ob_low + random.uniform(0.0005, 0.002)
            
            order_block = OrderBlock(
                start_time=current_time - timedelta(minutes=random.randint(30, 180)),
                end_time=current_time - timedelta(minutes=random.randint(5, 30)),
                high=ob_high,
                low=ob_low,
                volume=random.uniform(1000, 5000),
                bias=random.choice(['bullish', 'bearish']),
                strength=random.uniform(6, 9),
                tested=False,
                broken=False
            )
            
            # Add to order blocks list, keep only recent ones
            if symbol not in self.order_blocks:
                self.order_blocks[symbol] = []
            self.order_blocks[symbol].append(order_block)
            self.order_blocks[symbol] = self.order_blocks[symbol][-10:]  # Keep last 10
            
        # Generate mock Fair Value Gap
        if random.random() < 0.2:  # 20% chance
            base_price = 1.0850 if symbol == 'EURUSD' else 1.2650 if symbol == 'GBPUSD' else 150.25
            variance = 0.001 if 'USD' in symbol else 0.01
            
            fvg_bottom = base_price + random.uniform(-variance, variance)
            fvg_top = fvg_bottom + random.uniform(0.0008, 0.0015)
            
            fvg = FairValueGap(
                start_time=current_time - timedelta(minutes=random.randint(15, 120)),
                top=fvg_top,
                bottom=fvg_bottom,
                bias=random.choice(['bullish', 'bearish']),
                filled=False,
                strength=random.uniform(5, 8)
            )
            
            # Add to FVG list, keep only recent ones
            if symbol not in self.fair_value_gaps:
                self.fair_value_gaps[symbol] = []
            self.fair_value_gaps[symbol].append(fvg)
            self.fair_value_gaps[symbol] = self.fair_value_gaps[symbol][-8:]  # Keep last 8
    
    def get_status(self) -> Dict[str, Any]:
        """Get SMC EA status"""
        return {
            "is_monitoring": self.is_monitoring,
            "symbols": self.symbols,
            "signals_generated": self.signals_generated,
            "trades_executed": self.trades_executed,
            "last_signal_time": self.last_signal_time.isoformat() if self.last_signal_time else None,
            "last_analysis_time": self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            "parameters": {
                "lookback_period": self.lookback_period,
                "swing_length": self.swing_length,
                "structure_sensitivity": self.structure_sensitivity,
                "risk_per_trade": self.risk_per_trade,
                "reward_ratio": self.reward_ratio,
                "show_order_blocks": self.show_order_blocks,
                "show_fvg": self.show_fvg,
                "show_liquidity_zones": self.show_liquidity_zones
            },
            "market_structure": {
                symbol: {
                    "trend": ms.trend,
                    "swing_highs_count": len(ms.swing_highs),
                    "swing_lows_count": len(ms.swing_lows)
                }
                for symbol, ms in self.market_structure.items()
            },
            "order_blocks_count": {symbol: len(obs) for symbol, obs in self.order_blocks.items()},
            "fvg_count": {symbol: len(fvgs) for symbol, fvgs in self.fair_value_gaps.items()},
            "liquidity_zones_count": {symbol: len(lzs) for symbol, lzs in self.liquidity_zones.items()}
        } 