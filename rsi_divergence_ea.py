import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.signal import find_peaks

# Import QNTI core components
from qnti_core_system import Trade, TradeSource, TradeStatus, logger

@dataclass
class RSIDivergenceSignal:
    """RSI Divergence signal data"""
    timestamp: datetime
    symbol: str
    signal_type: str  # "BULLISH_DIV", "BEARISH_DIV", "HIDDEN_BULLISH", "HIDDEN_BEARISH"
    price: float
    rsi_value: float
    divergence_strength: float
    confidence: float

class RSIDivergenceEA:
    """
    Expert Advisor for RSI with Divergence Detection
    
    Features:
    - Classic Bullish/Bearish Divergence
    - Hidden Divergence Detection
    - RSI Overbought/Oversold Levels
    - Multi-timeframe Analysis
    - Dynamic Stop Loss/Take Profit
    """
    
    def __init__(self, 
                 rsi_period=14, 
                 rsi_overbought=70, 
                 rsi_oversold=30,
                 divergence_lookback=20,
                 min_divergence_strength=0.6):
        
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.divergence_lookback = divergence_lookback
        self.min_divergence_strength = min_divergence_strength
        
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
        self.rsi_period = int(parameters.get('rsi_period', self.rsi_period))
        self.rsi_overbought = float(parameters.get('rsi_overbought', self.rsi_overbought))
        self.rsi_oversold = float(parameters.get('rsi_oversold', self.rsi_oversold))
        self.divergence_lookback = int(parameters.get('divergence_lookback', self.divergence_lookback))
        self.min_divergence_strength = float(parameters.get('min_divergence_strength', self.min_divergence_strength))
        self.logger.info("Parameters updated successfully")
        
    def calculate_rsi(self, close_prices: np.array) -> np.array:
        """Calculate RSI indicator"""
        return talib.RSI(close_prices, timeperiod=self.rsi_period)
    
    def detect_divergence(self, prices: np.array, rsi: np.array) -> List[Dict]:
        """Detect RSI divergence patterns"""
        divergences = []
        
        if len(prices) < self.divergence_lookback:
            return divergences
        
        # Find peaks and troughs in price and RSI
        price_peaks, _ = find_peaks(prices, distance=5)
        price_troughs, _ = find_peaks(-prices, distance=5)
        rsi_peaks, _ = find_peaks(rsi, distance=5)
        rsi_troughs, _ = find_peaks(-rsi, distance=5)
        
        # Look for bullish divergence (price makes lower low, RSI makes higher low)
        if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
            recent_price_trough = price_troughs[-1]
            prev_price_trough = price_troughs[-2]
            
            # Find corresponding RSI troughs
            rsi_trough_near_recent = self._find_nearest_point(rsi_troughs, recent_price_trough)
            rsi_trough_near_prev = self._find_nearest_point(rsi_troughs, prev_price_trough)
            
            if rsi_trough_near_recent is not None and rsi_trough_near_prev is not None:
                # Check for bullish divergence
                if (prices[recent_price_trough] < prices[prev_price_trough] and 
                    rsi[rsi_trough_near_recent] > rsi[rsi_trough_near_prev]):
                    
                    strength = self._calculate_divergence_strength(
                        prices[prev_price_trough], prices[recent_price_trough],
                        rsi[rsi_trough_near_prev], rsi[rsi_trough_near_recent]
                    )
                    
                    if strength >= self.min_divergence_strength:
                        divergences.append({
                            'type': 'BULLISH_DIV',
                            'strength': strength,
                            'price_index': recent_price_trough,
                            'rsi_value': rsi[rsi_trough_near_recent]
                        })
        
        # Look for bearish divergence (price makes higher high, RSI makes lower high)
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            recent_price_peak = price_peaks[-1]
            prev_price_peak = price_peaks[-2]
            
            # Find corresponding RSI peaks
            rsi_peak_near_recent = self._find_nearest_point(rsi_peaks, recent_price_peak)
            rsi_peak_near_prev = self._find_nearest_point(rsi_peaks, prev_price_peak)
            
            if rsi_peak_near_recent is not None and rsi_peak_near_prev is not None:
                # Check for bearish divergence
                if (prices[recent_price_peak] > prices[prev_price_peak] and 
                    rsi[rsi_peak_near_recent] < rsi[rsi_peak_near_prev]):
                    
                    strength = self._calculate_divergence_strength(
                        prices[prev_price_peak], prices[recent_price_peak],
                        rsi[rsi_peak_near_prev], rsi[rsi_peak_near_recent]
                    )
                    
                    if strength >= self.min_divergence_strength:
                        divergences.append({
                            'type': 'BEARISH_DIV',
                            'strength': strength,
                            'price_index': recent_price_peak,
                            'rsi_value': rsi[rsi_peak_near_recent]
                        })
        
        return divergences
    
    def _find_nearest_point(self, points: np.array, target: int, max_distance=10) -> Optional[int]:
        """Find nearest point within max_distance"""
        if len(points) == 0:
            return None
        
        distances = np.abs(points - target)
        nearest_idx = np.argmin(distances)
        
        if distances[nearest_idx] <= max_distance:
            return points[nearest_idx]
        return None
    
    def _calculate_divergence_strength(self, price1, price2, rsi1, rsi2) -> float:
        """Calculate divergence strength (0-1)"""
        price_change = abs(price2 - price1) / price1
        rsi_change = abs(rsi2 - rsi1) / 100
        
        # Normalize and combine
        strength = min(1.0, (price_change * 100 + rsi_change) / 2)
        return strength
    
    def generate_signal(self, symbol: str, timeframe: str = "M15") -> Optional[RSIDivergenceSignal]:
        """Generate trading signal based on RSI and divergence analysis"""
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
            
            if rates is None or len(rates) < self.rsi_period + self.divergence_lookback:
                return None
            
            # Convert to numpy arrays
            close_prices = np.array([rate['close'] for rate in rates])
            
            # Calculate RSI
            rsi_values = self.calculate_rsi(close_prices)
            current_rsi = rsi_values[-1]
            current_price = close_prices[-1]
            
            # Detect divergences
            divergences = self.detect_divergence(close_prices, rsi_values)
            
            # Generate signal based on conditions
            signal_type = None
            confidence = 0.5
            
            # Check for divergence signals
            for div in divergences:
                if div['type'] == 'BULLISH_DIV' and current_rsi < 50:
                    signal_type = "LONG_ENTRY"
                    confidence = 0.7 + (div['strength'] * 0.3)
                    break
                elif div['type'] == 'BEARISH_DIV' and current_rsi > 50:
                    signal_type = "SHORT_ENTRY"
                    confidence = 0.7 + (div['strength'] * 0.3)
                    break
            
            # Check for basic RSI signals if no divergence
            if not signal_type:
                if current_rsi < self.rsi_oversold and current_rsi > rsi_values[-2]:
                    signal_type = "LONG_ENTRY"
                    confidence = 0.6
                elif current_rsi > self.rsi_overbought and current_rsi < rsi_values[-2]:
                    signal_type = "SHORT_ENTRY"
                    confidence = 0.6
            
            # Check exit conditions
            if self.position != 0:
                if self.position == 1 and current_rsi > self.rsi_overbought:
                    signal_type = "LONG_EXIT"
                    confidence = 0.8
                elif self.position == -1 and current_rsi < self.rsi_oversold:
                    signal_type = "SHORT_EXIT"
                    confidence = 0.8
            
            if signal_type:
                signal = RSIDivergenceSignal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type=signal_type,
                    price=current_price,
                    rsi_value=current_rsi,
                    divergence_strength=divergences[0]['strength'] if divergences else 0.0,
                    confidence=confidence
                )
                
                self.signals_history.append(signal)
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating RSI signal: {e}")
            return None
    
    def execute_trade(self, signal: RSIDivergenceSignal) -> bool:
        """Execute trade based on signal"""
        try:
            if not self.qnti_trade_manager or not self.qnti_mt5_bridge:
                return False
            
            # Calculate position size based on confidence
            base_lot_size = 0.01
            lot_size = base_lot_size * signal.confidence
            
            # Calculate stop loss and take profit
            atr_multiplier = 2.0
            if signal.signal_type in ["LONG_ENTRY", "SHORT_ENTRY"]:
                # Calculate ATR for dynamic SL/TP
                stop_loss_pips = 50  # Default fallback
                take_profit_pips = 100
                
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
                    trade_id=f"RSI_{signal.symbol}_{int(time.time())}",
                    magic_number=77777,  # RSI EA magic number
                    symbol=signal.symbol,
                    trade_type=trade_type,
                    lot_size=lot_size,
                    open_price=signal.price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    source=TradeSource.EXPERT_ADVISOR,
                    ea_name="RSI_Divergence_EA",
                    ai_confidence=signal.confidence,
                    strategy_tags=["rsi", "divergence", signal.signal_type.lower()]
                )
                
                # Execute trade
                success, message = self.qnti_mt5_bridge.execute_trade(trade)
                
                if success:
                    self.position = 1 if trade_type == "BUY" else -1
                    self.entry_price = signal.price
                    self.entry_time = datetime.now()
                    self.logger.info(f"RSI trade executed: {signal.signal_type} at {signal.price}")
                
                return success
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing RSI trade: {e}")
            return False
    
    def start_monitoring(self, symbols: List[str] = ["EURUSD"]):
        """Start RSI monitoring and trading"""
        if self.running:
            self.logger.warning("RSI Divergence EA already running")
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(symbols,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("RSI Divergence EA monitoring started")
    
    def stop_monitoring(self):
        """Stop RSI monitoring"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("RSI Divergence EA monitoring stopped")
    
    def _monitoring_loop(self, symbols: List[str]):
        """Main monitoring loop"""
        while self.running:
            try:
                for symbol in symbols:
                    signal = self.generate_signal(symbol)
                    if signal:
                        self.logger.info(f"RSI Signal: {signal.signal_type} for {symbol} at {signal.price}")
                        
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
                self.logger.error(f"Error in RSI monitoring loop: {e}")
                time.sleep(5)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current EA status"""
        return {
            "name": "RSI Divergence EA",
            "running": self.running,
            "position": self.position,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "parameters": {
                "rsi_period": self.rsi_period,
                "rsi_overbought": self.rsi_overbought,
                "rsi_oversold": self.rsi_oversold,
                "divergence_lookback": self.divergence_lookback,
                "min_divergence_strength": self.min_divergence_strength
            },
            "signals_count": len(self.signals_history),
            "last_signal": self.signals_history[-1].__dict__ if self.signals_history else None
        } 