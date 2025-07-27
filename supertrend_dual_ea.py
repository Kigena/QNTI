import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Import QNTI core components
from qnti_core_system import Trade, TradeSource, TradeStatus, logger

@dataclass
class SuperTrendSignal:
    """SuperTrend signal data"""
    timestamp: datetime
    symbol: str
    signal_type: str  # "LONG_ENTRY", "SHORT_ENTRY", "LONG_EXIT", "SHORT_EXIT"
    price: float
    st_direction: float
    cl_direction: float
    confidence: float

class SuperTrendDualEA:
    """
    Expert Advisor that combines two SuperTrend indicators:
    1. Standard SuperTrend (on price)
    2. Centerline SuperTrend (normalized oscillator)
    
    Entry: Both indicators must agree (both bullish or both bearish)
    Exit: When indicators no longer agree
    """
    
    def __init__(self, 
                 # Standard SuperTrend parameters
                 st_period=7, 
                 st_multiplier=7.0,
                 # Centerline SuperTrend parameters
                 cl_period=22, 
                 cl_multiplier=3.0,
                 cl_use_wicks=False,
                 # Trading parameters
                 symbol="EURUSD",
                 magic_number=77777,
                 lot_size=0.1):
        
        self.st_period = st_period
        self.st_multiplier = st_multiplier
        self.cl_period = cl_period
        self.cl_multiplier = cl_multiplier
        self.cl_use_wicks = cl_use_wicks
        
        # Trading parameters
        self.symbol = symbol
        self.magic_number = magic_number
        self.lot_size = lot_size
        
        # Position tracking
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = None
        self.entry_time = None
        self.current_trade_id = None
        
        # Live data tracking
        self.is_running = False
        self.monitoring_thread = None
        self.last_signals = []
        self.total_signals = 0
        
        # Data buffer for calculations
        self.data_buffer = pd.DataFrame()
        self.buffer_size = max(st_period, cl_period) * 3  # Keep enough data for calculations
        
        # QNTI Integration
        self.trade_manager = None
        self.mt5_bridge = None
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.SuperTrendDualEA")
        
    def set_qnti_integration(self, trade_manager, mt5_bridge):
        """Set QNTI system components for integration"""
        self.trade_manager = trade_manager
        self.mt5_bridge = mt5_bridge
        self.logger.info("QNTI integration components set")
        
    def calculate_atr(self, high, low, close, period):
        """Calculate Average True Range"""
        try:
            return talib.ATR(high, low, close, timeperiod=period)
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return np.full(len(close), np.nan)
    
    def calculate_standard_supertrend(self, high, low, close):
        """
        Calculate standard SuperTrend indicator
        Returns: direction (1 for long, -1 for short), longStop, shortStop
        """
        try:
            hl2 = (high + low) / 2
            atr = self.calculate_atr(high, low, close, self.st_period)
            atr_bands = atr * self.st_multiplier
            
            # Initialize arrays
            longStop = hl2 - atr_bands
            shortStop = hl2 + atr_bands
            dir_array = np.zeros(len(close))
            
            # Calculate SuperTrend
            for i in range(1, len(close)):
                # Long stop calculation
                if close[i-1] > longStop[i-1]:
                    longStop[i] = max(longStop[i], longStop[i-1])
                
                # Short stop calculation
                if close[i-1] < shortStop[i-1]:
                    shortStop[i] = min(shortStop[i], shortStop[i-1])
                
                # Direction calculation
                if i == 1:
                    dir_array[i] = 1
                else:
                    if dir_array[i-1] == -1 and close[i] > shortStop[i-1]:
                        dir_array[i] = 1
                    elif dir_array[i-1] == 1 and close[i] < longStop[i-1]:
                        dir_array[i] = -1
                    else:
                        dir_array[i] = dir_array[i-1]
            
            return dir_array, longStop, shortStop
        except Exception as e:
            self.logger.error(f"Error calculating standard SuperTrend: {e}")
            return np.zeros(len(close)), np.zeros(len(close)), np.zeros(len(close))
    
    def calculate_centerline_supertrend(self, high, low, close):
        """
        Calculate Centerline SuperTrend (normalized oscillator version)
        Returns: direction (1 for long, -1 for short)
        """
        try:
            hl2 = (high + low) / 2
            atr = self.calculate_atr(high, low, close, self.cl_period)
            atr_bands = atr * self.cl_multiplier
            
            # Initialize arrays
            longStop = hl2 - atr_bands
            shortStop = hl2 + atr_bands
            dir_array = np.zeros(len(close))
            
            # Calculate SuperTrend with wicks option
            for i in range(1, len(close)):
                # Choose price based on wicks setting
                if self.cl_use_wicks:
                    long_ref_price = low[i-1]
                    short_ref_price = high[i-1]
                else:
                    long_ref_price = close[i-1]
                    short_ref_price = close[i-1]
                
                # Long stop calculation
                if long_ref_price > longStop[i-1]:
                    longStop[i] = max(longStop[i], longStop[i-1])
                
                # Short stop calculation
                if short_ref_price < shortStop[i-1]:
                    shortStop[i] = min(shortStop[i], shortStop[i-1])
                
                # Direction calculation
                if i == 1:
                    dir_array[i] = 1
                else:
                    ref_high = high[i] if self.cl_use_wicks else close[i]
                    ref_low = low[i] if self.cl_use_wicks else close[i]
                    
                    if dir_array[i-1] == -1 and ref_high > shortStop[i-1]:
                        dir_array[i] = 1
                    elif dir_array[i-1] == 1 and ref_low < longStop[i-1]:
                        dir_array[i] = -1
                    else:
                        dir_array[i] = dir_array[i-1]
            
            return dir_array, longStop, shortStop
        except Exception as e:
            self.logger.error(f"Error calculating centerline SuperTrend: {e}")
            return np.zeros(len(close)), np.zeros(len(close)), np.zeros(len(close))
    
    def analyze_current_data(self, ohlc_data):
        """Analyze current market data and generate signals"""
        try:
            if len(ohlc_data) < self.buffer_size:
                self.logger.warning(f"Insufficient data for analysis: {len(ohlc_data)} bars")
                return None
            
            # Calculate both SuperTrend indicators
            st_dir, st_long, st_short = self.calculate_standard_supertrend(
                ohlc_data['high'].values, ohlc_data['low'].values, ohlc_data['close'].values
            )
            
            cl_dir, cl_long, cl_short = self.calculate_centerline_supertrend(
                ohlc_data['high'].values, ohlc_data['low'].values, ohlc_data['close'].values
            )
            
            # Get current values (last bar)
            current_st_dir = st_dir[-1]
            current_cl_dir = cl_dir[-1]
            current_price = ohlc_data['close'].iloc[-1]
            
            # Get previous values for change detection
            prev_st_dir = st_dir[-2] if len(st_dir) > 1 else current_st_dir
            prev_cl_dir = cl_dir[-2] if len(cl_dir) > 1 else current_cl_dir
            
            # Determine signal type
            signal_type = None
            confidence = 0.0
            
            # Agreement status
            both_bullish = (current_st_dir == 1) and (current_cl_dir == 1)
            both_bearish = (current_st_dir == -1) and (current_cl_dir == -1)
            
            # Entry signals (both must agree and at least one changed)
            if both_bullish and (prev_st_dir != 1 or prev_cl_dir != 1):
                signal_type = "LONG_ENTRY"
                confidence = 0.8 if (prev_st_dir != 1 and prev_cl_dir != 1) else 0.6
                
            elif both_bearish and (prev_st_dir != -1 or prev_cl_dir != -1):
                signal_type = "SHORT_ENTRY"
                confidence = 0.8 if (prev_st_dir != -1 and prev_cl_dir != -1) else 0.6
            
            # Exit signals (indicators no longer agree with current position)
            elif self.position == 1 and not both_bullish:
                signal_type = "LONG_EXIT"
                confidence = 0.7
                
            elif self.position == -1 and not both_bearish:
                signal_type = "SHORT_EXIT"
                confidence = 0.7
            
            if signal_type:
                signal = SuperTrendSignal(
                    timestamp=datetime.now(),
                    symbol=self.symbol,
                    signal_type=signal_type,
                    price=current_price,
                    st_direction=current_st_dir,
                    cl_direction=current_cl_dir,
                    confidence=confidence
                )
                
                self.logger.info(f"Generated signal: {signal_type} at {current_price:.5f} "
                               f"(ST: {current_st_dir}, CL: {current_cl_dir}, Confidence: {confidence:.2f})")
                
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing current data: {e}")
            return None
    
    def execute_signal(self, signal: SuperTrendSignal):
        """Execute a trading signal through QNTI system"""
        try:
            if not self.trade_manager or not self.mt5_bridge:
                self.logger.error("QNTI integration not available for trade execution")
                return False
            
            # Handle entry signals
            if signal.signal_type in ["LONG_ENTRY", "SHORT_ENTRY"]:
                if self.position != 0:
                    self.logger.warning(f"Already in position {self.position}, skipping entry signal")
                    return False
                
                # Create trade object
                trade_type = "BUY" if signal.signal_type == "LONG_ENTRY" else "SELL"
                trade_id = f"ST_DUAL_{int(time.time())}"
                
                trade = Trade(
                    trade_id=trade_id,
                    magic_number=self.magic_number,
                    symbol=self.symbol,
                    trade_type=trade_type,
                    lot_size=self.lot_size,
                    open_price=signal.price,
                    source=TradeSource.EXPERT_ADVISOR,
                    ai_confidence=signal.confidence,
                    strategy_tags=["supertrend_dual", "trend_following"],
                    ea_name="SuperTrend_Dual_EA"
                )
                
                # Execute trade
                success, message = self.mt5_bridge.execute_trade(trade)
                
                if success:
                    self.position = 1 if trade_type == "BUY" else -1
                    self.entry_price = signal.price
                    self.entry_time = signal.timestamp
                    self.current_trade_id = trade_id
                    self.total_signals += 1
                    
                    self.logger.info(f"Trade executed successfully: {trade_id}")
                    return True
                else:
                    self.logger.error(f"Trade execution failed: {message}")
                    return False
            
            # Handle exit signals
            elif signal.signal_type in ["LONG_EXIT", "SHORT_EXIT"]:
                if self.position == 0:
                    self.logger.warning("No position to exit")
                    return False
                
                if self.current_trade_id:
                    success, message = self.mt5_bridge.close_trade(self.current_trade_id)
                    
                    if success:
                        self.logger.info(f"Position closed: {self.current_trade_id}")
                        self.position = 0
                        self.entry_price = None
                        self.entry_time = None
                        self.current_trade_id = None
                        return True
                    else:
                        self.logger.error(f"Position close failed: {message}")
                        return False
                        
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
            return False
    
    def start_monitoring(self):
        """Start live monitoring and trading"""
        if self.is_running:
            self.logger.warning("SuperTrend Dual EA already running")
            return False
        
        if not self.trade_manager or not self.mt5_bridge:
            self.logger.error("QNTI integration required for live monitoring")
            return False
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("SuperTrend Dual EA monitoring started")
        return True
    
    def stop_monitoring(self):
        """Stop live monitoring"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("SuperTrend Dual EA monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        self.logger.info("SuperTrend Dual EA monitoring loop started")
        
        while self.is_running:
            try:
                # Get latest market data from MT5
                ohlc_data = self._get_latest_data()
                
                if ohlc_data is not None and len(ohlc_data) >= self.buffer_size:
                    # Analyze for signals
                    signal = self.analyze_current_data(ohlc_data)
                    
                    if signal:
                        # Store signal
                        self.last_signals.append(signal)
                        if len(self.last_signals) > 10:  # Keep last 10 signals
                            self.last_signals.pop(0)
                        
                        # Execute signal
                        self.execute_signal(signal)
                
                # Sleep for 1 minute between checks
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Shorter sleep on error
    
    def _get_latest_data(self):
        """Get latest OHLC data from MT5"""
        try:
            if not self.mt5_bridge:
                return None
            
            # Get symbol data
            if self.symbol not in self.mt5_bridge.symbols:
                self.logger.warning(f"Symbol {self.symbol} not available")
                return None
            
            # For now, return a simple DataFrame with current price
            # In full implementation, this would get historical OHLC data
            symbol_data = self.mt5_bridge.symbols[self.symbol]
            
            # Create a simple OHLC record (this is simplified)
            current_time = datetime.now()
            current_price = symbol_data.last
            
            # Build buffer with current data point
            new_row = pd.DataFrame({
                'timestamp': [current_time],
                'open': [current_price],
                'high': [current_price + 0.0001],
                'low': [current_price - 0.0001],
                'close': [current_price]
            })
            
            # Append to buffer
            self.data_buffer = pd.concat([self.data_buffer, new_row], ignore_index=True)
            
            # Keep only required size
            if len(self.data_buffer) > self.buffer_size:
                self.data_buffer = self.data_buffer.tail(self.buffer_size).reset_index(drop=True)
            
            return self.data_buffer.copy()
            
        except Exception as e:
            self.logger.error(f"Error getting latest data: {e}")
            return None
    
    def get_status(self):
        """Get current EA status"""
        return {
            'is_running': self.is_running,
            'symbol': self.symbol,
            'position': self.position,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'total_signals': self.total_signals,
            'last_signals': [
                {
                    'timestamp': s.timestamp.isoformat(),
                    'signal_type': s.signal_type,
                    'price': s.price,
                    'confidence': s.confidence
                } for s in self.last_signals[-5:]  # Last 5 signals
            ],
            'parameters': {
                'st_period': self.st_period,
                'st_multiplier': self.st_multiplier,
                'cl_period': self.cl_period,
                'cl_multiplier': self.cl_multiplier,
                'cl_use_wicks': self.cl_use_wicks
            }
        }
    
    def update_parameters(self, parameters: Dict[str, Any]):
        """Update strategy parameters"""
        try:
            if 'st_period' in parameters:
                self.st_period = int(parameters['st_period'])
            if 'st_multiplier' in parameters:
                self.st_multiplier = float(parameters['st_multiplier'])
            if 'cl_period' in parameters:
                self.cl_period = int(parameters['cl_period'])
            if 'cl_multiplier' in parameters:
                self.cl_multiplier = float(parameters['cl_multiplier'])
            if 'cl_use_wicks' in parameters:
                self.cl_use_wicks = bool(parameters['cl_use_wicks'])
            
            # Update buffer size if periods changed
            self.buffer_size = max(self.st_period, self.cl_period) * 3
            
            self.logger.info("Parameters updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating parameters: {e}")
            return False


# Global instance for the unified automation system
_supertrend_dual_instance = None

def get_supertrend_dual_instance():
    """Get or create the global SuperTrend Dual EA instance"""
    global _supertrend_dual_instance
    if _supertrend_dual_instance is None:
        _supertrend_dual_instance = SuperTrendDualEA()
    return _supertrend_dual_instance 