import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Import QNTI core components
from qnti_core_system import Trade, TradeSource, TradeStatus, logger

@dataclass
class NRBreakoutSignal:
    """NR4/NR7 Breakout signal data"""
    timestamp: datetime
    symbol: str
    signal_type: str  # "NR4_BREAK", "NR7_BREAK", "LONG_BREAK", "SHORT_BREAK"
    price: float
    nr_type: str  # "NR4", "NR7", "NR14"
    high_break: float
    low_break: float
    range_size: float
    avg_range: float
    volume_confirmation: bool
    confidence: float

class NR4NR7BreakoutEA:
    """
    NR4/NR7 Breakout Expert Advisor
    
    Features:
    - NR4 (Narrow Range 4) Detection
    - NR7 (Narrow Range 7) Detection  
    - NR14 (Narrow Range 14) Detection
    - Breakout Trading with Volume Confirmation
    - Dynamic Stop Loss and Take Profit
    - False Breakout Protection
    - Time-based Filters
    """
    
    def __init__(self, 
                 enable_nr4=True,
                 enable_nr7=True, 
                 enable_nr14=False,
                 volume_confirmation=True,
                 breakout_buffer=0.0001,
                 max_wait_hours=24):
        
        self.enable_nr4 = enable_nr4
        self.enable_nr7 = enable_nr7
        self.enable_nr14 = enable_nr14
        self.volume_confirmation = volume_confirmation
        self.breakout_buffer = breakout_buffer  # Buffer above/below high/low for breakout
        self.max_wait_hours = max_wait_hours  # Maximum hours to wait for breakout
        
        # Position tracking
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = None
        self.entry_time = None
        
        # Threading control
        self.running = False
        self.monitoring_thread = None
        
        # QNTI Integration
        self.qnti_trade_manager = None
        self.qnti_mt5_bridge = None
        self.qnti_main_system = None
        
        # Signal history
        self.signals_history = []
        
        # NR tracking
        self.active_nr_patterns = {}  # Track active NR patterns waiting for breakout
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def set_qnti_integration(self, trade_manager, mt5_bridge, main_system):
        """Set QNTI system components"""
        self.qnti_trade_manager = trade_manager
        self.qnti_mt5_bridge = mt5_bridge
        self.qnti_main_system = main_system
        self.logger.info("QNTI integration components set")
        
    def update_parameters(self, parameters: Dict[str, Any]):
        """Update strategy parameters"""
        self.enable_nr4 = bool(parameters.get('enable_nr4', self.enable_nr4))
        self.enable_nr7 = bool(parameters.get('enable_nr7', self.enable_nr7))
        self.enable_nr14 = bool(parameters.get('enable_nr14', self.enable_nr14))
        self.volume_confirmation = bool(parameters.get('volume_confirmation', self.volume_confirmation))
        self.breakout_buffer = float(parameters.get('breakout_buffer', self.breakout_buffer))
        self.max_wait_hours = int(parameters.get('max_wait_hours', self.max_wait_hours))
        self.logger.info("Parameters updated successfully")
        
    def calculate_true_range(self, high: np.array, low: np.array, close: np.array) -> np.array:
        """Calculate True Range for each bar"""
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]  # Handle first element
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    def detect_nr_patterns(self, high: np.array, low: np.array, close: np.array, volume: np.array) -> List[Dict]:
        """Detect NR4, NR7, and NR14 patterns"""
        patterns = []
        
        if len(high) < 15:  # Need at least 15 bars for NR14
            return patterns
        
        # Calculate true range for all bars
        true_range = self.calculate_true_range(high, low, close)
        
        # Check for patterns at the most recent bar
        current_idx = len(high) - 1
        current_range = high[current_idx] - low[current_idx]
        
        # NR4 Detection
        if self.enable_nr4 and current_idx >= 3:
            lookback_ranges = [high[i] - low[i] for i in range(current_idx - 3, current_idx)]
            if current_range < min(lookback_ranges):
                patterns.append({
                    'type': 'NR4',
                    'index': current_idx,
                    'high': high[current_idx],
                    'low': low[current_idx],
                    'range': current_range,
                    'avg_range': np.mean(lookback_ranges),
                    'volume': volume[current_idx] if len(volume) > current_idx else 0,
                    'detected_time': datetime.now()
                })
        
        # NR7 Detection
        if self.enable_nr7 and current_idx >= 6:
            lookback_ranges = [high[i] - low[i] for i in range(current_idx - 6, current_idx)]
            if current_range < min(lookback_ranges):
                patterns.append({
                    'type': 'NR7',
                    'index': current_idx,
                    'high': high[current_idx],
                    'low': low[current_idx],
                    'range': current_range,
                    'avg_range': np.mean(lookback_ranges),
                    'volume': volume[current_idx] if len(volume) > current_idx else 0,
                    'detected_time': datetime.now()
                })
        
        # NR14 Detection
        if self.enable_nr14 and current_idx >= 13:
            lookback_ranges = [high[i] - low[i] for i in range(current_idx - 13, current_idx)]
            if current_range < min(lookback_ranges):
                patterns.append({
                    'type': 'NR14',
                    'index': current_idx,
                    'high': high[current_idx],
                    'low': low[current_idx],
                    'range': current_range,
                    'avg_range': np.mean(lookback_ranges),
                    'volume': volume[current_idx] if len(volume) > current_idx else 0,
                    'detected_time': datetime.now()
                })
        
        return patterns
    
    def check_breakout(self, current_price: float, current_volume: float, nr_pattern: Dict) -> Optional[str]:
        """Check if current price breaks out of NR pattern"""
        high_break_level = nr_pattern['high'] + self.breakout_buffer
        low_break_level = nr_pattern['low'] - self.breakout_buffer
        
        # Check for breakout
        if current_price > high_break_level:
            # Upward breakout
            if self.volume_confirmation:
                # Check if volume is higher than NR bar volume
                if current_volume > nr_pattern['volume'] * 1.5:  # 50% higher volume
                    return "LONG_BREAK"
            else:
                return "LONG_BREAK"
        
        elif current_price < low_break_level:
            # Downward breakout
            if self.volume_confirmation:
                # Check if volume is higher than NR bar volume
                if current_volume > nr_pattern['volume'] * 1.5:  # 50% higher volume
                    return "SHORT_BREAK"
            else:
                return "SHORT_BREAK"
        
        return None
    
    def calculate_position_size(self, nr_pattern: Dict, confidence: float) -> float:
        """Calculate position size based on NR pattern and confidence"""
        base_lot_size = 0.01
        
        # Adjust size based on pattern type
        size_multiplier = 1.0
        if nr_pattern['type'] == 'NR4':
            size_multiplier = 1.2  # NR4 is more frequent, slightly larger size
        elif nr_pattern['type'] == 'NR7':
            size_multiplier = 1.5  # NR7 is less frequent, larger size
        elif nr_pattern['type'] == 'NR14':
            size_multiplier = 2.0  # NR14 is rare, largest size
        
        # Adjust based on range compression
        range_ratio = nr_pattern['range'] / nr_pattern['avg_range']
        compression_multiplier = max(0.5, 2.0 - range_ratio)  # More compression = larger size
        
        return base_lot_size * size_multiplier * compression_multiplier * confidence
    
    def calculate_stop_loss_take_profit(self, signal_type: str, entry_price: float, nr_pattern: Dict) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit"""
        range_size = nr_pattern['range']
        avg_range = nr_pattern['avg_range']
        
        # Stop loss: Place beyond the opposite side of NR bar
        if signal_type == "LONG_BREAK":
            stop_loss = nr_pattern['low'] - (range_size * 0.2)  # 20% buffer below NR low
            take_profit = entry_price + (avg_range * 2)  # Target 2x average range
        else:  # SHORT_BREAK
            stop_loss = nr_pattern['high'] + (range_size * 0.2)  # 20% buffer above NR high
            take_profit = entry_price - (avg_range * 2)  # Target 2x average range
        
        return stop_loss, take_profit
    
    def clean_expired_patterns(self, symbol: str):
        """Remove expired NR patterns that haven't broken out"""
        if symbol not in self.active_nr_patterns:
            return
        
        current_time = datetime.now()
        expired_patterns = []
        
        for pattern_id, pattern in self.active_nr_patterns[symbol].items():
            time_diff = current_time - pattern['detected_time']
            if time_diff.total_seconds() > (self.max_wait_hours * 3600):
                expired_patterns.append(pattern_id)
        
        for pattern_id in expired_patterns:
            del self.active_nr_patterns[symbol][pattern_id]
            self.logger.info(f"Expired NR pattern {pattern_id} for {symbol}")
    
    def generate_signal(self, symbol: str, timeframe: str = "M15") -> Optional[NRBreakoutSignal]:
        """Generate trading signal based on NR breakout analysis"""
        try:
            if not self.qnti_mt5_bridge:
                return None
            
            # Get market data
            symbol_data = None
            for sym in self.qnti_mt5_bridge.symbols.values():
                if sym.name == symbol:
                    symbol_data = sym
                    break
            
            if not symbol_data:
                return None
            
            # Get historical data (last 50 bars for analysis)
            import MetaTrader5 as mt5
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 50)
            
            if rates is None or len(rates) < 15:
                return None
            
            # Convert to numpy arrays
            high_prices = np.array([rate['high'] for rate in rates])
            low_prices = np.array([rate['low'] for rate in rates])
            close_prices = np.array([rate['close'] for rate in rates])
            volumes = np.array([rate['tick_volume'] for rate in rates])
            
            current_price = close_prices[-1]
            current_volume = volumes[-1]
            
            # Initialize active patterns for symbol if not exists
            if symbol not in self.active_nr_patterns:
                self.active_nr_patterns[symbol] = {}
            
            # Clean expired patterns
            self.clean_expired_patterns(symbol)
            
            # Detect new NR patterns
            new_patterns = self.detect_nr_patterns(high_prices, low_prices, close_prices, volumes)
            
            # Add new patterns to active tracking
            for pattern in new_patterns:
                pattern_id = f"{pattern['type']}_{pattern['index']}_{int(pattern['detected_time'].timestamp())}"
                self.active_nr_patterns[symbol][pattern_id] = pattern
                self.logger.info(f"New {pattern['type']} pattern detected for {symbol} at {pattern['detected_time']}")
            
            # Check for breakouts of active patterns
            signal_type = None
            triggered_pattern = None
            confidence = 0.5
            
            for pattern_id, pattern in self.active_nr_patterns[symbol].items():
                breakout = self.check_breakout(current_price, current_volume, pattern)
                if breakout:
                    signal_type = breakout
                    triggered_pattern = pattern
                    
                    # Calculate confidence based on pattern type and volume
                    base_confidence = 0.6
                    if pattern['type'] == 'NR7':
                        base_confidence = 0.7
                    elif pattern['type'] == 'NR14':
                        base_confidence = 0.8
                    
                    # Volume confirmation boost
                    if self.volume_confirmation and current_volume > pattern['volume'] * 2:
                        base_confidence += 0.2
                    
                    confidence = min(1.0, base_confidence)
                    
                    # Remove triggered pattern from active tracking
                    del self.active_nr_patterns[symbol][pattern_id]
                    break
            
            # Exit conditions for existing positions
            if self.position != 0 and not signal_type:
                # Check if price reverses back into the NR range
                for pattern_id, pattern in list(self.active_nr_patterns[symbol].items()):
                    if (pattern['low'] <= current_price <= pattern['high']):
                        if self.position == 1:
                            signal_type = "LONG_EXIT"
                        elif self.position == -1:
                            signal_type = "SHORT_EXIT"
                        confidence = 0.8
                        triggered_pattern = pattern
                        break
            
            if signal_type and triggered_pattern:
                signal = NRBreakoutSignal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type=signal_type,
                    price=current_price,
                    nr_type=triggered_pattern['type'],
                    high_break=triggered_pattern['high'] + self.breakout_buffer,
                    low_break=triggered_pattern['low'] - self.breakout_buffer,
                    range_size=triggered_pattern['range'],
                    avg_range=triggered_pattern['avg_range'],
                    volume_confirmation=current_volume > triggered_pattern['volume'] * 1.5,
                    confidence=confidence
                )
                
                self.signals_history.append(signal)
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating NR breakout signal: {e}")
            return None
    
    def execute_trade(self, signal: NRBreakoutSignal) -> bool:
        """Execute trade based on signal"""
        try:
            if not self.qnti_trade_manager or not self.qnti_mt5_bridge:
                return False
            
            if signal.signal_type in ["LONG_BREAK", "SHORT_BREAK"]:
                # Find the triggered pattern for stop loss calculation
                pattern = {
                    'type': signal.nr_type,
                    'high': signal.high_break - self.breakout_buffer,
                    'low': signal.low_break + self.breakout_buffer,
                    'range': signal.range_size,
                    'avg_range': signal.avg_range
                }
                
                # Calculate position size
                lot_size = self.calculate_position_size(pattern, signal.confidence)
                
                # Calculate stop loss and take profit
                trade_type = "BUY" if signal.signal_type == "LONG_BREAK" else "SELL"
                stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                    signal.signal_type, signal.price, pattern)
                
                # Create trade
                trade = Trade(
                    trade_id=f"NR_{signal.symbol}_{int(time.time())}",
                    magic_number=22222,  # NR Breakout EA magic number
                    symbol=signal.symbol,
                    trade_type=trade_type,
                    lot_size=lot_size,
                    open_price=signal.price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    source=TradeSource.EXPERT_ADVISOR,
                    ea_name="NR4_NR7_Breakout_EA",
                    ai_confidence=signal.confidence,
                    strategy_tags=["nr_breakout", signal.nr_type.lower(), signal.signal_type.lower()]
                )
                
                # Execute trade
                success, message = self.qnti_mt5_bridge.execute_trade(trade)
                
                if success:
                    self.position = 1 if trade_type == "BUY" else -1
                    self.entry_price = signal.price
                    self.entry_time = datetime.now()
                    self.logger.info(f"NR Breakout trade executed: {signal.signal_type} ({signal.nr_type}) at {signal.price}")
                
                return success
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing NR breakout trade: {e}")
            return False
    
    def start_monitoring(self, symbols: List[str] = ["EURUSD"]):
        """Start NR breakout monitoring and trading"""
        if self.running:
            self.logger.warning("NR4/NR7 Breakout EA already running")
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(symbols,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("NR4/NR7 Breakout EA monitoring started")
    
    def stop_monitoring(self):
        """Stop NR breakout monitoring"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("NR4/NR7 Breakout EA monitoring stopped")
    
    def _monitoring_loop(self, symbols: List[str]):
        """Main monitoring loop"""
        while self.running:
            try:
                for symbol in symbols:
                    signal = self.generate_signal(symbol)
                    if signal:
                        self.logger.info(f"NR Breakout Signal: {signal.signal_type} ({signal.nr_type}) for {symbol} at {signal.price}")
                        
                        # Execute trade if it's a breakout signal and we have no position
                        if signal.signal_type in ["LONG_BREAK", "SHORT_BREAK"] and self.position == 0:
                            self.execute_trade(signal)
                        elif signal.signal_type in ["LONG_EXIT", "SHORT_EXIT"] and self.position != 0:
                            # Handle exit logic
                            self.position = 0
                            self.entry_price = None
                            self.entry_time = None
                
                # Sleep between analysis cycles
                time.sleep(60)  # Check every minute for NR patterns
                
            except Exception as e:
                self.logger.error(f"Error in NR breakout monitoring loop: {e}")
                time.sleep(5)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current EA status"""
        active_patterns_count = sum(len(patterns) for patterns in self.active_nr_patterns.values())
        
        return {
            "name": "NR4/NR7 Breakout EA",
            "running": self.running,
            "position": self.position,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "parameters": {
                "enable_nr4": self.enable_nr4,
                "enable_nr7": self.enable_nr7,
                "enable_nr14": self.enable_nr14,
                "volume_confirmation": self.volume_confirmation,
                "breakout_buffer": self.breakout_buffer,
                "max_wait_hours": self.max_wait_hours
            },
            "signals_count": len(self.signals_history),
            "last_signal": self.signals_history[-1].__dict__ if self.signals_history else None,
            "active_patterns": {
                "total_count": active_patterns_count,
                "by_symbol": {symbol: len(patterns) for symbol, patterns in self.active_nr_patterns.items()}
            }
        } 