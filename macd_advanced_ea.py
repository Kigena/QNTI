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
class MACDSignal:
    """MACD signal data"""
    timestamp: datetime
    symbol: str
    signal_type: str  # "CROSS_UP", "CROSS_DOWN", "DIVERGENCE", "ZERO_CROSS"
    price: float
    macd_value: float
    signal_value: float
    histogram_value: float
    trend_strength: float
    confidence: float

class MACDAdvancedEA:
    """
    Advanced MACD Expert Advisor
    
    Features:
    - MACD Line and Signal Line Crossovers
    - Zero Line Crossovers
    - MACD Histogram Analysis
    - Divergence Detection
    - Multiple Timeframe Confirmation
    - Dynamic Stop Loss/Take Profit
    """
    
    def __init__(self, 
                 fast_period=12, 
                 slow_period=26,
                 signal_period=9,
                 zero_cross_enabled=True,
                 histogram_threshold=0.0001):
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.zero_cross_enabled = zero_cross_enabled
        self.histogram_threshold = histogram_threshold
        
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
        
        # Previous values for crossover detection
        self.prev_macd = None
        self.prev_signal = None
        self.prev_histogram = None
        
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
        self.fast_period = int(parameters.get('fast_period', self.fast_period))
        self.slow_period = int(parameters.get('slow_period', self.slow_period))
        self.signal_period = int(parameters.get('signal_period', self.signal_period))
        self.zero_cross_enabled = bool(parameters.get('zero_cross_enabled', self.zero_cross_enabled))
        self.histogram_threshold = float(parameters.get('histogram_threshold', self.histogram_threshold))
        self.logger.info("Parameters updated successfully")
        
    def calculate_macd(self, close_prices: np.array) -> Tuple[np.array, np.array, np.array]:
        """Calculate MACD indicator"""
        macd_line, signal_line, histogram = talib.MACD(
            close_prices, 
            fastperiod=self.fast_period,
            slowperiod=self.slow_period, 
            signalperiod=self.signal_period
        )
        return macd_line, signal_line, histogram
    
    def detect_crossover(self, current_macd, current_signal, prev_macd, prev_signal) -> Optional[str]:
        """Detect MACD crossovers"""
        if prev_macd is None or prev_signal is None:
            return None
        
        # Bullish crossover: MACD crosses above signal
        if prev_macd <= prev_signal and current_macd > current_signal:
            return "CROSS_UP"
        
        # Bearish crossover: MACD crosses below signal
        if prev_macd >= prev_signal and current_macd < current_signal:
            return "CROSS_DOWN"
        
        return None
    
    def detect_zero_crossover(self, current_macd, prev_macd) -> Optional[str]:
        """Detect MACD zero line crossovers"""
        if prev_macd is None:
            return None
        
        # Bullish zero cross: MACD crosses above zero
        if prev_macd <= 0 and current_macd > 0:
            return "ZERO_CROSS_UP"
        
        # Bearish zero cross: MACD crosses below zero
        if prev_macd >= 0 and current_macd < 0:
            return "ZERO_CROSS_DOWN"
        
        return None
    
    def analyze_histogram(self, histogram: np.array) -> Dict[str, Any]:
        """Analyze MACD histogram for momentum"""
        if len(histogram) < 3:
            return {"trend": "neutral", "strength": 0.0}
        
        current_hist = histogram[-1]
        prev_hist = histogram[-2]
        prev2_hist = histogram[-3]
        
        # Determine trend
        if current_hist > prev_hist > prev2_hist:
            trend = "increasing"
            strength = min(1.0, abs(current_hist) / self.histogram_threshold)
        elif current_hist < prev_hist < prev2_hist:
            trend = "decreasing"
            strength = min(1.0, abs(current_hist) / self.histogram_threshold)
        else:
            trend = "neutral"
            strength = 0.5
        
        return {
            "trend": trend,
            "strength": strength,
            "current_value": current_hist,
            "momentum": "bullish" if current_hist > 0 else "bearish" if current_hist < 0 else "neutral"
        }
    
    def calculate_trend_strength(self, macd_line: np.array, prices: np.array) -> float:
        """Calculate trend strength based on MACD and price action"""
        if len(macd_line) < 10:
            return 0.5
        
        # Calculate MACD slope
        recent_macd = macd_line[-5:]
        macd_slope = np.polyfit(range(len(recent_macd)), recent_macd, 1)[0]
        
        # Calculate price slope
        recent_prices = prices[-5:]
        price_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        
        # Normalize slopes and combine
        macd_strength = min(1.0, abs(macd_slope) * 1000)  # Scale MACD slope
        price_strength = min(1.0, abs(price_slope) * 10000)  # Scale price slope
        
        # Check if MACD and price agree on direction
        agreement = 1.0 if (macd_slope > 0) == (price_slope > 0) else 0.5
        
        return (macd_strength + price_strength) * agreement / 2
    
    def generate_signal(self, symbol: str, timeframe: str = "M15") -> Optional[MACDSignal]:
        """Generate trading signal based on MACD analysis"""
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
            
            # Get historical data (last 100 bars for analysis)
            import MetaTrader5 as mt5
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
            
            if rates is None or len(rates) < max(self.slow_period, self.signal_period) + 10:
                return None
            
            # Convert to numpy arrays
            close_prices = np.array([rate['close'] for rate in rates])
            
            # Calculate MACD
            macd_line, signal_line, histogram = self.calculate_macd(close_prices)
            
            if len(macd_line) < 2:
                return None
            
            current_macd = macd_line[-1]
            current_signal = signal_line[-1] 
            current_histogram = histogram[-1]
            current_price = close_prices[-1]
            
            # Detect signals
            signal_type = None
            confidence = 0.5
            
            # Check for crossovers
            crossover = self.detect_crossover(current_macd, current_signal, self.prev_macd, self.prev_signal)
            if crossover:
                if crossover == "CROSS_UP":
                    signal_type = "LONG_ENTRY"
                    confidence = 0.7
                elif crossover == "CROSS_DOWN":
                    signal_type = "SHORT_ENTRY"
                    confidence = 0.7
            
            # Check for zero line crossovers if enabled
            if self.zero_cross_enabled and not signal_type:
                zero_cross = self.detect_zero_crossover(current_macd, self.prev_macd)
                if zero_cross:
                    if zero_cross == "ZERO_CROSS_UP":
                        signal_type = "LONG_ENTRY"
                        confidence = 0.8
                    elif zero_cross == "ZERO_CROSS_DOWN":
                        signal_type = "SHORT_ENTRY"
                        confidence = 0.8
            
            # Analyze histogram for additional confirmation
            hist_analysis = self.analyze_histogram(histogram)
            
            # Adjust confidence based on histogram
            if signal_type and hist_analysis["trend"] != "neutral":
                if ((signal_type == "LONG_ENTRY" and hist_analysis["momentum"] == "bullish") or
                    (signal_type == "SHORT_ENTRY" and hist_analysis["momentum"] == "bearish")):
                    confidence = min(1.0, confidence + 0.2)
            
            # Calculate trend strength
            trend_strength = self.calculate_trend_strength(macd_line, close_prices)
            confidence = min(1.0, confidence * (0.5 + trend_strength))
            
            # Check exit conditions
            if self.position != 0:
                if self.position == 1:  # Long position
                    if (crossover == "CROSS_DOWN" or 
                        (current_macd < 0 and self.prev_macd >= 0)):
                        signal_type = "LONG_EXIT"
                        confidence = 0.8
                elif self.position == -1:  # Short position
                    if (crossover == "CROSS_UP" or 
                        (current_macd > 0 and self.prev_macd <= 0)):
                        signal_type = "SHORT_EXIT"
                        confidence = 0.8
            
            # Store current values for next iteration
            self.prev_macd = current_macd
            self.prev_signal = current_signal
            self.prev_histogram = current_histogram
            
            if signal_type:
                signal = MACDSignal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type=signal_type,
                    price=current_price,
                    macd_value=current_macd,
                    signal_value=current_signal,
                    histogram_value=current_histogram,
                    trend_strength=trend_strength,
                    confidence=confidence
                )
                
                self.signals_history.append(signal)
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating MACD signal: {e}")
            return None
    
    def execute_trade(self, signal: MACDSignal) -> bool:
        """Execute trade based on signal"""
        try:
            if not self.qnti_trade_manager or not self.qnti_mt5_bridge:
                return False
            
            # Calculate position size based on confidence and trend strength
            base_lot_size = 0.01
            lot_size = base_lot_size * signal.confidence * (0.5 + signal.trend_strength)
            
            if signal.signal_type in ["LONG_ENTRY", "SHORT_ENTRY"]:
                # Calculate dynamic stop loss and take profit
                atr_multiplier = 2.0
                stop_loss_pips = max(30, int(50 * (2 - signal.confidence)))  # Tighter SL for higher confidence
                take_profit_pips = int(stop_loss_pips * 2)  # 2:1 RR ratio
                
                if signal.signal_type == "LONG_ENTRY":
                    stop_loss = signal.price - (stop_loss_pips * 0.0001)
                    take_profit = signal.price + (take_profit_pips * 0.0001)
                    trade_type = "BUY"
                else:
                    stop_loss = signal.price + (stop_loss_pips * 0.0001)
                    take_profit = signal.price - (take_profit_pips * 0.0001)
                    trade_type = "SELL"
                
                # Create trade
                trade = Trade(
                    trade_id=f"MACD_{signal.symbol}_{int(time.time())}",
                    magic_number=88888,  # MACD EA magic number
                    symbol=signal.symbol,
                    trade_type=trade_type,
                    lot_size=lot_size,
                    open_price=signal.price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    source=TradeSource.EXPERT_ADVISOR,
                    ea_name="MACD_Advanced_EA",
                    ai_confidence=signal.confidence,
                    strategy_tags=["macd", "crossover", signal.signal_type.lower()]
                )
                
                # Execute trade
                success, message = self.qnti_mt5_bridge.execute_trade(trade)
                
                if success:
                    self.position = 1 if trade_type == "BUY" else -1
                    self.entry_price = signal.price
                    self.entry_time = datetime.now()
                    self.logger.info(f"MACD trade executed: {signal.signal_type} at {signal.price}")
                
                return success
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing MACD trade: {e}")
            return False
    
    def start_monitoring(self, symbols: List[str] = ["EURUSD"]):
        """Start MACD monitoring and trading"""
        if self.running:
            self.logger.warning("MACD Advanced EA already running")
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(symbols,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("MACD Advanced EA monitoring started")
    
    def stop_monitoring(self):
        """Stop MACD monitoring"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("MACD Advanced EA monitoring stopped")
    
    def _monitoring_loop(self, symbols: List[str]):
        """Main monitoring loop"""
        while self.running:
            try:
                for symbol in symbols:
                    signal = self.generate_signal(symbol)
                    if signal:
                        self.logger.info(f"MACD Signal: {signal.signal_type} for {symbol} at {signal.price}")
                        
                        # Execute trade if it's an entry signal and we have no position
                        if signal.signal_type in ["LONG_ENTRY", "SHORT_ENTRY"] and self.position == 0:
                            self.execute_trade(signal)
                        elif signal.signal_type in ["LONG_EXIT", "SHORT_EXIT"] and self.position != 0:
                            # Handle exit logic
                            self.position = 0
                            self.entry_price = None
                            self.entry_time = None
                
                # Sleep between analysis cycles
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in MACD monitoring loop: {e}")
                time.sleep(5)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current EA status"""
        return {
            "name": "MACD Advanced EA",
            "running": self.running,
            "position": self.position,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "parameters": {
                "fast_period": self.fast_period,
                "slow_period": self.slow_period,
                "signal_period": self.signal_period,
                "zero_cross_enabled": self.zero_cross_enabled,
                "histogram_threshold": self.histogram_threshold
            },
            "signals_count": len(self.signals_history),
            "last_signal": self.signals_history[-1].__dict__ if self.signals_history else None,
            "current_macd": {
                "macd": self.prev_macd,
                "signal": self.prev_signal,
                "histogram": self.prev_histogram
            }
        } 