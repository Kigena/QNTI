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
class IchimokuSignal:
    """Ichimoku signal data"""
    timestamp: datetime
    symbol: str
    signal_type: str  # "CLOUD_BREAK", "CROSS_SIGNAL", "FUTURE_CLOUD", "TK_CROSS"
    price: float
    tenkan_sen: float
    kijun_sen: float
    chikou_span: float
    senkou_span_a: float
    senkou_span_b: float
    cloud_color: str  # "bullish", "bearish", "neutral"
    trend_strength: float
    confidence: float

class IchimokuCloudEA:
    """
    Ichimoku Cloud Expert Advisor
    
    Features:
    - Tenkan-Sen / Kijun-Sen Crossovers
    - Cloud (Kumo) Break Analysis
    - Chikou Span Confirmation
    - Future Cloud Analysis
    - Multiple Timeframe Confluence
    - Dynamic Risk Management
    """
    
    def __init__(self, 
                 tenkan_period=9, 
                 kijun_period=26,
                 senkou_span_b_period=52,
                 chikou_confirmation=True,
                 cloud_filter=True):
        
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_span_b_period = senkou_span_b_period
        self.chikou_confirmation = chikou_confirmation
        self.cloud_filter = cloud_filter
        
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
        self.prev_tenkan = None
        self.prev_kijun = None
        self.prev_price_vs_cloud = None
        
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
        self.tenkan_period = int(parameters.get('tenkan_period', self.tenkan_period))
        self.kijun_period = int(parameters.get('kijun_period', self.kijun_period))
        self.senkou_span_b_period = int(parameters.get('senkou_span_b_period', self.senkou_span_b_period))
        self.chikou_confirmation = bool(parameters.get('chikou_confirmation', self.chikou_confirmation))
        self.cloud_filter = bool(parameters.get('cloud_filter', self.cloud_filter))
        self.logger.info("Parameters updated successfully")
        
    def calculate_ichimoku(self, high_prices: np.array, low_prices: np.array, close_prices: np.array) -> Dict[str, np.array]:
        """Calculate all Ichimoku components"""
        # Tenkan-Sen (Conversion Line)
        tenkan_highs = pd.Series(high_prices).rolling(window=self.tenkan_period).max().values
        tenkan_lows = pd.Series(low_prices).rolling(window=self.tenkan_period).min().values
        tenkan_sen = (tenkan_highs + tenkan_lows) / 2
        
        # Kijun-Sen (Base Line)
        kijun_highs = pd.Series(high_prices).rolling(window=self.kijun_period).max().values
        kijun_lows = pd.Series(low_prices).rolling(window=self.kijun_period).min().values
        kijun_sen = (kijun_highs + kijun_lows) / 2
        
        # Senkou Span A (Leading Span A) - projected 26 periods ahead
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        
        # Senkou Span B (Leading Span B) - projected 26 periods ahead
        senkou_b_highs = pd.Series(high_prices).rolling(window=self.senkou_span_b_period).max().values
        senkou_b_lows = pd.Series(low_prices).rolling(window=self.senkou_span_b_period).min().values
        senkou_span_b = (senkou_b_highs + senkou_b_lows) / 2
        
        # Chikou Span (Lagging Span) - close price shifted back 26 periods
        chikou_span = np.concatenate([close_prices[self.kijun_period:], np.full(self.kijun_period, np.nan)])
        
        return {
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_span_a": senkou_span_a,
            "senkou_span_b": senkou_span_b,
            "chikou_span": chikou_span
        }
    
    def analyze_cloud(self, senkou_span_a: float, senkou_span_b: float, price: float) -> Dict[str, Any]:
        """Analyze cloud (Kumo) characteristics"""
        if np.isnan(senkou_span_a) or np.isnan(senkou_span_b):
            return {"color": "neutral", "thickness": 0, "position": "unknown"}
        
        # Determine cloud color
        if senkou_span_a > senkou_span_b:
            cloud_color = "bullish"  # Green/bullish cloud
        elif senkou_span_a < senkou_span_b:
            cloud_color = "bearish"  # Red/bearish cloud
        else:
            cloud_color = "neutral"
        
        # Calculate cloud thickness (support/resistance strength)
        cloud_top = max(senkou_span_a, senkou_span_b)
        cloud_bottom = min(senkou_span_a, senkou_span_b)
        thickness = (cloud_top - cloud_bottom) / ((cloud_top + cloud_bottom) / 2) if cloud_top + cloud_bottom > 0 else 0
        
        # Determine price position relative to cloud
        if price > cloud_top:
            position = "above"
        elif price < cloud_bottom:
            position = "below"
        else:
            position = "inside"
        
        return {
            "color": cloud_color,
            "thickness": thickness,
            "position": position,
            "top": cloud_top,
            "bottom": cloud_bottom,
            "support_resistance": cloud_top if position == "above" else cloud_bottom
        }
    
    def detect_tk_cross(self, current_tenkan: float, current_kijun: float, 
                       prev_tenkan: float, prev_kijun: float) -> Optional[str]:
        """Detect Tenkan-Sen / Kijun-Sen crossovers"""
        if (prev_tenkan is None or prev_kijun is None or 
            np.isnan(prev_tenkan) or np.isnan(prev_kijun)):
            return None
        
        # Bullish TK cross: Tenkan crosses above Kijun
        if prev_tenkan <= prev_kijun and current_tenkan > current_kijun:
            return "TK_CROSS_UP"
        
        # Bearish TK cross: Tenkan crosses below Kijun
        if prev_tenkan >= prev_kijun and current_tenkan < current_kijun:
            return "TK_CROSS_DOWN"
        
        return None
    
    def analyze_chikou_span(self, chikou_span: np.array, prices: np.array, current_index: int) -> Dict[str, Any]:
        """Analyze Chikou Span for confirmation"""
        if current_index < self.kijun_period or len(chikou_span) <= current_index:
            return {"confirmation": "neutral", "strength": 0.5}
        
        current_chikou = chikou_span[current_index - self.kijun_period]
        current_price = prices[current_index - self.kijun_period]
        
        if np.isnan(current_chikou):
            return {"confirmation": "neutral", "strength": 0.5}
        
        # Chikou above price = bullish confirmation
        if current_chikou > current_price:
            strength = min(1.0, (current_chikou - current_price) / current_price * 100)
            return {"confirmation": "bullish", "strength": 0.5 + strength * 0.5}
        
        # Chikou below price = bearish confirmation
        elif current_chikou < current_price:
            strength = min(1.0, (current_price - current_chikou) / current_price * 100)
            return {"confirmation": "bearish", "strength": 0.5 + strength * 0.5}
        
        return {"confirmation": "neutral", "strength": 0.5}
    
    def calculate_trend_strength(self, ichimoku_data: Dict[str, np.array], cloud_analysis: Dict[str, Any]) -> float:
        """Calculate overall trend strength based on Ichimoku components"""
        strength_factors = []
        
        # Factor 1: TK relationship
        current_tenkan = ichimoku_data["tenkan_sen"][-1]
        current_kijun = ichimoku_data["kijun_sen"][-1]
        
        if not (np.isnan(current_tenkan) or np.isnan(current_kijun)):
            tk_strength = abs(current_tenkan - current_kijun) / max(current_tenkan, current_kijun)
            strength_factors.append(min(1.0, tk_strength * 100))
        
        # Factor 2: Cloud thickness
        strength_factors.append(min(1.0, cloud_analysis["thickness"] * 10))
        
        # Factor 3: Price distance from cloud
        if cloud_analysis["position"] != "inside":
            distance = abs(cloud_analysis["top"] - cloud_analysis["bottom"])
            price_distance = distance / max(cloud_analysis["top"], cloud_analysis["bottom"]) * 100
            strength_factors.append(min(1.0, price_distance))
        else:
            strength_factors.append(0.3)  # Weak when inside cloud
        
        return sum(strength_factors) / len(strength_factors) if strength_factors else 0.5
    
    def generate_signal(self, symbol: str, timeframe: str = "M15") -> Optional[IchimokuSignal]:
        """Generate trading signal based on Ichimoku analysis"""
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
            
            if rates is None or len(rates) < max(self.senkou_span_b_period, self.kijun_period) + 10:
                return None
            
            # Convert to numpy arrays
            high_prices = np.array([rate['high'] for rate in rates])
            low_prices = np.array([rate['low'] for rate in rates])
            close_prices = np.array([rate['close'] for rate in rates])
            current_price = close_prices[-1]
            
            # Calculate Ichimoku components
            ichimoku_data = self.calculate_ichimoku(high_prices, low_prices, close_prices)
            
            # Current values
            current_tenkan = ichimoku_data["tenkan_sen"][-1]
            current_kijun = ichimoku_data["kijun_sen"][-1]
            current_senkou_a = ichimoku_data["senkou_span_a"][-1]
            current_senkou_b = ichimoku_data["senkou_span_b"][-1]
            current_chikou = ichimoku_data["chikou_span"][-1]
            
            # Analyze cloud
            cloud_analysis = self.analyze_cloud(current_senkou_a, current_senkou_b, current_price)
            
            # Detect TK crossover
            tk_cross = self.detect_tk_cross(current_tenkan, current_kijun, self.prev_tenkan, self.prev_kijun)
            
            # Analyze Chikou Span
            chikou_analysis = self.analyze_chikou_span(ichimoku_data["chikou_span"], close_prices, len(close_prices) - 1)
            
            # Calculate trend strength
            trend_strength = self.calculate_trend_strength(ichimoku_data, cloud_analysis)
            
            # Generate signals
            signal_type = None
            confidence = 0.5
            
            # Primary signal: TK Cross with cloud confirmation
            if tk_cross:
                if tk_cross == "TK_CROSS_UP":
                    # Bullish TK cross
                    if not self.cloud_filter or cloud_analysis["color"] == "bullish":
                        signal_type = "LONG_ENTRY"
                        confidence = 0.7
                        
                        # Boost confidence if price is above cloud
                        if cloud_analysis["position"] == "above":
                            confidence += 0.1
                            
                elif tk_cross == "TK_CROSS_DOWN":
                    # Bearish TK cross
                    if not self.cloud_filter or cloud_analysis["color"] == "bearish":
                        signal_type = "SHORT_ENTRY"
                        confidence = 0.7
                        
                        # Boost confidence if price is below cloud
                        if cloud_analysis["position"] == "below":
                            confidence += 0.1
            
            # Secondary signal: Cloud breaks
            elif cloud_analysis["position"] != self.prev_price_vs_cloud:
                if (self.prev_price_vs_cloud == "below" and cloud_analysis["position"] == "above" and
                    cloud_analysis["color"] == "bullish"):
                    signal_type = "LONG_ENTRY"
                    confidence = 0.6
                elif (self.prev_price_vs_cloud == "above" and cloud_analysis["position"] == "below" and
                      cloud_analysis["color"] == "bearish"):
                    signal_type = "SHORT_ENTRY"
                    confidence = 0.6
            
            # Apply Chikou confirmation if enabled
            if signal_type and self.chikou_confirmation:
                if signal_type == "LONG_ENTRY" and chikou_analysis["confirmation"] != "bullish":
                    confidence *= 0.7  # Reduce confidence without confirmation
                elif signal_type == "SHORT_ENTRY" and chikou_analysis["confirmation"] != "bearish":
                    confidence *= 0.7  # Reduce confidence without confirmation
                elif ((signal_type == "LONG_ENTRY" and chikou_analysis["confirmation"] == "bullish") or
                      (signal_type == "SHORT_ENTRY" and chikou_analysis["confirmation"] == "bearish")):
                    confidence = min(1.0, confidence + 0.2)  # Boost with confirmation
            
            # Apply trend strength
            confidence = min(1.0, confidence * (0.5 + trend_strength))
            
            # Exit conditions
            if self.position != 0:
                if self.position == 1:  # Long position
                    if (tk_cross == "TK_CROSS_DOWN" or 
                        cloud_analysis["position"] == "below" or
                        (current_price < current_kijun and not np.isnan(current_kijun))):
                        signal_type = "LONG_EXIT"
                        confidence = 0.8
                elif self.position == -1:  # Short position
                    if (tk_cross == "TK_CROSS_UP" or 
                        cloud_analysis["position"] == "above" or
                        (current_price > current_kijun and not np.isnan(current_kijun))):
                        signal_type = "SHORT_EXIT"
                        confidence = 0.8
            
            # Store current values for next iteration
            self.prev_tenkan = current_tenkan
            self.prev_kijun = current_kijun
            self.prev_price_vs_cloud = cloud_analysis["position"]
            
            if signal_type:
                signal = IchimokuSignal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type=signal_type,
                    price=current_price,
                    tenkan_sen=current_tenkan,
                    kijun_sen=current_kijun,
                    chikou_span=current_chikou,
                    senkou_span_a=current_senkou_a,
                    senkou_span_b=current_senkou_b,
                    cloud_color=cloud_analysis["color"],
                    trend_strength=trend_strength,
                    confidence=confidence
                )
                
                self.signals_history.append(signal)
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating Ichimoku signal: {e}")
            return None
    
    def execute_trade(self, signal: IchimokuSignal) -> bool:
        """Execute trade based on signal"""
        try:
            if not self.qnti_trade_manager or not self.qnti_mt5_bridge:
                return False
            
            # Calculate position size based on confidence and trend strength
            base_lot_size = 0.01
            lot_size = base_lot_size * signal.confidence * (0.5 + signal.trend_strength)
            
            if signal.signal_type in ["LONG_ENTRY", "SHORT_ENTRY"]:
                # Calculate dynamic stop loss and take profit using Ichimoku levels
                
                if signal.signal_type == "LONG_ENTRY":
                    # Use Kijun-Sen or cloud as support
                    if not np.isnan(signal.kijun_sen):
                        stop_loss = signal.kijun_sen - (abs(signal.price - signal.kijun_sen) * 0.1)
                    else:
                        stop_loss = signal.price - (signal.price * 0.02)  # 2% fallback
                    
                    # Target above cloud or Tenkan-Sen
                    if not np.isnan(signal.tenkan_sen):
                        take_profit = signal.price + (abs(signal.price - signal.tenkan_sen) * 2)
                    else:
                        take_profit = signal.price + (signal.price * 0.03)  # 3% fallback
                    
                    trade_type = "BUY"
                else:
                    # Use Kijun-Sen or cloud as resistance
                    if not np.isnan(signal.kijun_sen):
                        stop_loss = signal.kijun_sen + (abs(signal.price - signal.kijun_sen) * 0.1)
                    else:
                        stop_loss = signal.price + (signal.price * 0.02)  # 2% fallback
                    
                    # Target below cloud or Tenkan-Sen
                    if not np.isnan(signal.tenkan_sen):
                        take_profit = signal.price - (abs(signal.price - signal.tenkan_sen) * 2)
                    else:
                        take_profit = signal.price - (signal.price * 0.03)  # 3% fallback
                    
                    trade_type = "SELL"
                
                # Create trade
                trade = Trade(
                    trade_id=f"ICHI_{signal.symbol}_{int(time.time())}",
                    magic_number=11111,  # Ichimoku EA magic number
                    symbol=signal.symbol,
                    trade_type=trade_type,
                    lot_size=lot_size,
                    open_price=signal.price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    source=TradeSource.EXPERT_ADVISOR,
                    ea_name="Ichimoku_Cloud_EA",
                    ai_confidence=signal.confidence,
                    strategy_tags=["ichimoku", "cloud", signal.signal_type.lower()]
                )
                
                # Execute trade
                success, message = self.qnti_mt5_bridge.execute_trade(trade)
                
                if success:
                    self.position = 1 if trade_type == "BUY" else -1
                    self.entry_price = signal.price
                    self.entry_time = datetime.now()
                    self.logger.info(f"Ichimoku trade executed: {signal.signal_type} at {signal.price}")
                
                return success
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing Ichimoku trade: {e}")
            return False
    
    def start_monitoring(self, symbols: List[str] = ["EURUSD"]):
        """Start Ichimoku monitoring and trading"""
        if self.running:
            self.logger.warning("Ichimoku Cloud EA already running")
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(symbols,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("Ichimoku Cloud EA monitoring started")
    
    def stop_monitoring(self):
        """Stop Ichimoku monitoring"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Ichimoku Cloud EA monitoring stopped")
    
    def _monitoring_loop(self, symbols: List[str]):
        """Main monitoring loop"""
        while self.running:
            try:
                for symbol in symbols:
                    signal = self.generate_signal(symbol)
                    if signal:
                        self.logger.info(f"Ichimoku Signal: {signal.signal_type} for {symbol} at {signal.price}")
                        
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
                self.logger.error(f"Error in Ichimoku monitoring loop: {e}")
                time.sleep(5)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current EA status"""
        return {
            "name": "Ichimoku Cloud EA",
            "running": self.running,
            "position": self.position,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "parameters": {
                "tenkan_period": self.tenkan_period,
                "kijun_period": self.kijun_period,
                "senkou_span_b_period": self.senkou_span_b_period,
                "chikou_confirmation": self.chikou_confirmation,
                "cloud_filter": self.cloud_filter
            },
            "signals_count": len(self.signals_history),
            "last_signal": self.signals_history[-1].__dict__ if self.signals_history else None,
            "current_levels": {
                "tenkan_sen": self.prev_tenkan,
                "kijun_sen": self.prev_kijun,
                "price_vs_cloud": self.prev_price_vs_cloud
            }
        } 