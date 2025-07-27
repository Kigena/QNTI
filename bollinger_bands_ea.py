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
class BollingerBandsSignal:
    """Bollinger Bands signal data"""
    timestamp: datetime
    symbol: str
    signal_type: str  # "BAND_BOUNCE", "BAND_BREAK", "SQUEEZE", "EXPANSION"
    price: float
    upper_band: float
    middle_band: float
    lower_band: float
    bandwidth: float
    percent_b: float
    squeeze_detected: bool
    confidence: float

class BollingerBandsEA:
    """
    Bollinger Bands Expert Advisor
    
    Features:
    - Mean Reversion (Band Bounces)
    - Breakout Trading (Band Breaks)
    - Bollinger Band Squeeze Detection
    - %B and Bandwidth Analysis
    - Multiple Timeframe Confirmation
    - Dynamic Position Sizing
    """
    
    def __init__(self, 
                 period=20, 
                 std_dev=2.0,
                 squeeze_threshold=0.1,
                 breakout_confirmation=True,
                 mean_reversion_mode=True):
        
        self.period = period
        self.std_dev = std_dev
        self.squeeze_threshold = squeeze_threshold
        self.breakout_confirmation = breakout_confirmation
        self.mean_reversion_mode = mean_reversion_mode
        
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
        
        # Historical data for analysis
        self.prev_bandwidth = None
        self.squeeze_active = False
        
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
        self.period = int(parameters.get('period', self.period))
        self.std_dev = float(parameters.get('std_dev', self.std_dev))
        self.squeeze_threshold = float(parameters.get('squeeze_threshold', self.squeeze_threshold))
        self.breakout_confirmation = bool(parameters.get('breakout_confirmation', self.breakout_confirmation))
        self.mean_reversion_mode = bool(parameters.get('mean_reversion_mode', self.mean_reversion_mode))
        self.logger.info("Parameters updated successfully")
        
    def calculate_bollinger_bands(self, close_prices: np.array) -> Tuple[np.array, np.array, np.array]:
        """Calculate Bollinger Bands"""
        upper_band, middle_band, lower_band = talib.BBANDS(
            close_prices,
            timeperiod=self.period,
            nbdevup=self.std_dev,
            nbdevdn=self.std_dev,
            matype=0
        )
        return upper_band, middle_band, lower_band
    
    def calculate_percent_b(self, price: float, upper_band: float, lower_band: float) -> float:
        """Calculate %B indicator"""
        if upper_band == lower_band:
            return 0.5
        return (price - lower_band) / (upper_band - lower_band)
    
    def calculate_bandwidth(self, upper_band: float, middle_band: float, lower_band: float) -> float:
        """Calculate Bollinger Band Width"""
        if middle_band == 0:
            return 0
        return (upper_band - lower_band) / middle_band
    
    def detect_squeeze(self, bandwidth: float) -> bool:
        """Detect Bollinger Band Squeeze"""
        return bandwidth < self.squeeze_threshold
    
    def analyze_price_position(self, price: float, upper_band: float, middle_band: float, lower_band: float) -> Dict[str, Any]:
        """Analyze price position relative to bands"""
        percent_b = self.calculate_percent_b(price, upper_band, lower_band)
        
        if price >= upper_band:
            position = "above_upper"
            signal_strength = min(1.0, (price - upper_band) / upper_band * 100)
        elif price <= lower_band:
            position = "below_lower"
            signal_strength = min(1.0, (lower_band - price) / lower_band * 100)
        elif price > middle_band:
            position = "upper_half"
            signal_strength = 0.5 + (percent_b - 0.5)
        else:
            position = "lower_half"
            signal_strength = 0.5 - (0.5 - percent_b)
        
        return {
            "position": position,
            "percent_b": percent_b,
            "signal_strength": signal_strength,
            "distance_to_middle": abs(price - middle_band) / middle_band
        }
    
    def detect_band_break(self, prices: np.array, upper_bands: np.array, lower_bands: np.array) -> Optional[str]:
        """Detect band breakouts"""
        if len(prices) < 3:
            return None
        
        current_price = prices[-1]
        prev_price = prices[-2]
        current_upper = upper_bands[-1]
        current_lower = lower_bands[-1]
        prev_upper = upper_bands[-2]
        prev_lower = lower_bands[-2]
        
        # Upward breakout
        if prev_price <= prev_upper and current_price > current_upper:
            return "BREAK_UPPER"
        
        # Downward breakout
        if prev_price >= prev_lower and current_price < current_lower:
            return "BREAK_LOWER"
        
        return None
    
    def analyze_momentum(self, prices: np.array, middle_bands: np.array) -> Dict[str, Any]:
        """Analyze momentum using price vs middle band"""
        if len(prices) < 5:
            return {"momentum": "neutral", "strength": 0.5}
        
        # Calculate slopes
        price_slope = np.polyfit(range(5), prices[-5:], 1)[0]
        band_slope = np.polyfit(range(5), middle_bands[-5:], 1)[0]
        
        # Determine momentum
        if price_slope > band_slope and price_slope > 0:
            momentum = "bullish"
            strength = min(1.0, abs(price_slope) * 10000)
        elif price_slope < band_slope and price_slope < 0:
            momentum = "bearish"
            strength = min(1.0, abs(price_slope) * 10000)
        else:
            momentum = "neutral"
            strength = 0.5
        
        return {
            "momentum": momentum,
            "strength": strength,
            "price_slope": price_slope,
            "band_slope": band_slope
        }
    
    def generate_signal(self, symbol: str, timeframe: str = "M15") -> Optional[BollingerBandsSignal]:
        """Generate trading signal based on Bollinger Bands analysis"""
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
            
            if rates is None or len(rates) < self.period + 10:
                return None
            
            # Convert to numpy arrays
            close_prices = np.array([rate['close'] for rate in rates])
            
            # Calculate Bollinger Bands
            upper_bands, middle_bands, lower_bands = self.calculate_bollinger_bands(close_prices)
            
            if len(upper_bands) < 2:
                return None
            
            current_price = close_prices[-1]
            current_upper = upper_bands[-1]
            current_middle = middle_bands[-1]
            current_lower = lower_bands[-1]
            
            # Calculate indicators
            bandwidth = self.calculate_bandwidth(current_upper, current_middle, current_lower)
            percent_b = self.calculate_percent_b(current_price, current_upper, current_lower)
            squeeze_detected = self.detect_squeeze(bandwidth)
            
            # Analyze price position
            price_analysis = self.analyze_price_position(current_price, current_upper, current_middle, current_lower)
            
            # Detect band breaks
            band_break = self.detect_band_break(close_prices, upper_bands, lower_bands)
            
            # Analyze momentum
            momentum_analysis = self.analyze_momentum(close_prices, middle_bands)
            
            # Generate signals
            signal_type = None
            confidence = 0.5
            
            # Check for squeeze breakouts
            if self.squeeze_active and not squeeze_detected:
                # Squeeze is ending - look for direction
                if band_break == "BREAK_UPPER":
                    signal_type = "LONG_ENTRY"
                    confidence = 0.8
                elif band_break == "BREAK_LOWER":
                    signal_type = "SHORT_ENTRY"
                    confidence = 0.8
            
            # Mean reversion signals
            elif self.mean_reversion_mode and not squeeze_detected:
                if price_analysis["position"] == "below_lower" and percent_b < 0.1:
                    signal_type = "LONG_ENTRY"  # Oversold bounce
                    confidence = 0.7 + (0.1 - percent_b) * 2  # Higher confidence for deeper oversold
                elif price_analysis["position"] == "above_upper" and percent_b > 0.9:
                    signal_type = "SHORT_ENTRY"  # Overbought reversal
                    confidence = 0.7 + (percent_b - 0.9) * 2  # Higher confidence for more overbought
            
            # Breakout signals (when not in mean reversion mode)
            elif not self.mean_reversion_mode and band_break:
                if band_break == "BREAK_UPPER" and momentum_analysis["momentum"] == "bullish":
                    signal_type = "LONG_ENTRY"
                    confidence = 0.6 + momentum_analysis["strength"] * 0.3
                elif band_break == "BREAK_LOWER" and momentum_analysis["momentum"] == "bearish":
                    signal_type = "SHORT_ENTRY"
                    confidence = 0.6 + momentum_analysis["strength"] * 0.3
            
            # Exit conditions
            if self.position != 0:
                if self.position == 1:  # Long position
                    if (price_analysis["position"] == "above_upper" and 
                        momentum_analysis["momentum"] == "bearish"):
                        signal_type = "LONG_EXIT"
                        confidence = 0.8
                    elif current_price < current_middle and percent_b < 0.5:
                        signal_type = "LONG_EXIT"
                        confidence = 0.7
                elif self.position == -1:  # Short position
                    if (price_analysis["position"] == "below_lower" and 
                        momentum_analysis["momentum"] == "bullish"):
                        signal_type = "SHORT_EXIT"
                        confidence = 0.8
                    elif current_price > current_middle and percent_b > 0.5:
                        signal_type = "SHORT_EXIT"
                        confidence = 0.7
            
            # Update squeeze state
            self.squeeze_active = squeeze_detected
            self.prev_bandwidth = bandwidth
            
            if signal_type:
                signal = BollingerBandsSignal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type=signal_type,
                    price=current_price,
                    upper_band=current_upper,
                    middle_band=current_middle,
                    lower_band=current_lower,
                    bandwidth=bandwidth,
                    percent_b=percent_b,
                    squeeze_detected=squeeze_detected,
                    confidence=confidence
                )
                
                self.signals_history.append(signal)
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating Bollinger Bands signal: {e}")
            return None
    
    def execute_trade(self, signal: BollingerBandsSignal) -> bool:
        """Execute trade based on signal"""
        try:
            if not self.qnti_trade_manager or not self.qnti_mt5_bridge:
                return False
            
            # Calculate position size based on confidence and band width
            base_lot_size = 0.01
            size_multiplier = signal.confidence * (1.0 + signal.bandwidth)  # Larger size during expansion
            lot_size = base_lot_size * size_multiplier
            
            if signal.signal_type in ["LONG_ENTRY", "SHORT_ENTRY"]:
                # Calculate dynamic stop loss and take profit based on band width
                band_range = signal.upper_band - signal.lower_band
                
                if signal.signal_type == "LONG_ENTRY":
                    if self.mean_reversion_mode:
                        # Mean reversion: target middle band, stop below lower band
                        stop_loss = signal.lower_band - (band_range * 0.1)
                        take_profit = signal.middle_band
                    else:
                        # Breakout: wider targets
                        stop_loss = signal.price - (band_range * 0.5)
                        take_profit = signal.price + (band_range * 1.0)
                    trade_type = "BUY"
                else:
                    if self.mean_reversion_mode:
                        # Mean reversion: target middle band, stop above upper band
                        stop_loss = signal.upper_band + (band_range * 0.1)
                        take_profit = signal.middle_band
                    else:
                        # Breakout: wider targets
                        stop_loss = signal.price + (band_range * 0.5)
                        take_profit = signal.price - (band_range * 1.0)
                    trade_type = "SELL"
                
                # Create trade
                trade = Trade(
                    trade_id=f"BB_{signal.symbol}_{int(time.time())}",
                    magic_number=99999,  # Bollinger Bands EA magic number
                    symbol=signal.symbol,
                    trade_type=trade_type,
                    lot_size=lot_size,
                    open_price=signal.price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    source=TradeSource.EXPERT_ADVISOR,
                    ea_name="Bollinger_Bands_EA",
                    ai_confidence=signal.confidence,
                    strategy_tags=["bollinger_bands", "mean_reversion" if self.mean_reversion_mode else "breakout", signal.signal_type.lower()]
                )
                
                # Execute trade
                success, message = self.qnti_mt5_bridge.execute_trade(trade)
                
                if success:
                    self.position = 1 if trade_type == "BUY" else -1
                    self.entry_price = signal.price
                    self.entry_time = datetime.now()
                    self.logger.info(f"Bollinger Bands trade executed: {signal.signal_type} at {signal.price}")
                
                return success
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing Bollinger Bands trade: {e}")
            return False
    
    def start_monitoring(self, symbols: List[str] = ["EURUSD"]):
        """Start Bollinger Bands monitoring and trading"""
        if self.running:
            self.logger.warning("Bollinger Bands EA already running")
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(symbols,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("Bollinger Bands EA monitoring started")
    
    def stop_monitoring(self):
        """Stop Bollinger Bands monitoring"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Bollinger Bands EA monitoring stopped")
    
    def _monitoring_loop(self, symbols: List[str]):
        """Main monitoring loop"""
        while self.running:
            try:
                for symbol in symbols:
                    signal = self.generate_signal(symbol)
                    if signal:
                        self.logger.info(f"Bollinger Bands Signal: {signal.signal_type} for {symbol} at {signal.price}")
                        
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
                self.logger.error(f"Error in Bollinger Bands monitoring loop: {e}")
                time.sleep(5)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current EA status"""
        return {
            "name": "Bollinger Bands EA",
            "running": self.running,
            "position": self.position,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "parameters": {
                "period": self.period,
                "std_dev": self.std_dev,
                "squeeze_threshold": self.squeeze_threshold,
                "breakout_confirmation": self.breakout_confirmation,
                "mean_reversion_mode": self.mean_reversion_mode
            },
            "signals_count": len(self.signals_history),
            "last_signal": self.signals_history[-1].__dict__ if self.signals_history else None,
            "current_state": {
                "squeeze_active": self.squeeze_active,
                "prev_bandwidth": self.prev_bandwidth
            }
        } 