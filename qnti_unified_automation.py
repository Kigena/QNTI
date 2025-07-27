"""
QNTI Unified Automation System
Dynamic Pine Script Indicator Management
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import importlib
import inspect
import os

logger = logging.getLogger(__name__)

@dataclass
class IndicatorConfig:
    """Configuration for a Pine Script indicator"""
    name: str
    type: str
    parameters: Dict[str, Any]
    signals: Dict[str, Any]
    risk_management: Dict[str, Any]
    alerts: Dict[str, bool]
    symbols: List[str]
    timeframes: List[str]
    enabled: bool = True

@dataclass
class Signal:
    """Trading signal from an indicator"""
    indicator: str
    symbol: str
    timeframe: str
    signal_type: str  # 'BUY', 'SELL', 'CLOSE'
    confidence: float
    price: float
    timestamp: datetime
    metadata: Dict[str, Any]

class BaseIndicator(ABC):
    """Base class for all Pine Script indicators"""
    
    def __init__(self, config: IndicatorConfig):
        self.config = config
        self.name = config.name
        self.type = config.type
        self.enabled = config.enabled
        self.last_signals: Dict[str, Signal] = {}
        
    @abstractmethod
    async def analyze(self, symbol: str, timeframe: str, data: Dict) -> Optional[Signal]:
        """Analyze market data and return signal if any"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return indicator-specific parameters schema"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate indicator configuration"""
        pass
    
    def is_enabled(self) -> bool:
        return self.enabled
    
    def get_supported_symbols(self) -> List[str]:
        return self.config.symbols
    
    def get_supported_timeframes(self) -> List[str]:
        return self.config.timeframes

class SMCIndicator(BaseIndicator):
    """Smart Money Concepts Indicator"""
    
    def __init__(self, config: IndicatorConfig):
        super().__init__(config)
        self.swing_length = config.parameters.get('swing_length', 10)
        self.fvg_threshold = config.parameters.get('fvg_threshold', 0.5)
        self.ob_lookback = config.parameters.get('ob_lookback', 20)
    
    async def analyze(self, symbol: str, timeframe: str, data: Dict) -> Optional[Signal]:
        """Analyze for SMC patterns"""
        try:
            # Extract OHLC data
            opens = data.get('open', [])
            highs = data.get('high', [])
            lows = data.get('low', [])
            closes = data.get('close', [])
            
            if len(closes) < self.ob_lookback:
                return None
            
            # SMC Analysis Logic
            current_price = closes[-1]
            
            # Order Block Detection
            order_blocks = self._detect_order_blocks(opens, highs, lows, closes)
            
            # Fair Value Gap Detection  
            fvgs = self._detect_fvg(opens, highs, lows, closes)
            
            # Liquidity Analysis
            liquidity_levels = self._detect_liquidity(highs, lows)
            
            # Generate signal
            signal_type = self._generate_signal(order_blocks, fvgs, liquidity_levels, current_price)
            
            if signal_type:
                confidence = self._calculate_confidence(order_blocks, fvgs, liquidity_levels)
                
                return Signal(
                    indicator="smc",
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_type=signal_type,
                    confidence=confidence,
                    price=current_price,
                    timestamp=datetime.now(),
                    metadata={
                        'order_blocks': len(order_blocks),
                        'fvgs': len(fvgs),
                        'liquidity_levels': len(liquidity_levels)
                    }
                )
                
        except Exception as e:
            logger.error(f"SMC analysis error for {symbol}: {e}")
            
        return None
    
    def _detect_order_blocks(self, opens, highs, lows, closes):
        """Detect order blocks"""
        order_blocks = []
        for i in range(self.ob_lookback, len(closes)):
            # Simplified order block logic
            if closes[i] > closes[i-1] and lows[i] < lows[i-1]:
                order_blocks.append({
                    'type': 'bullish',
                    'price': lows[i],
                    'index': i
                })
        return order_blocks
    
    def _detect_fvg(self, opens, highs, lows, closes):
        """Detect Fair Value Gaps"""
        fvgs = []
        for i in range(2, len(closes)):
            # Bullish FVG
            if lows[i] > highs[i-2] and (lows[i] - highs[i-2]) > self.fvg_threshold:
                fvgs.append({
                    'type': 'bullish',
                    'top': lows[i],
                    'bottom': highs[i-2],
                    'index': i
                })
            # Bearish FVG
            elif highs[i] < lows[i-2] and (lows[i-2] - highs[i]) > self.fvg_threshold:
                fvgs.append({
                    'type': 'bearish',
                    'top': lows[i-2],
                    'bottom': highs[i],
                    'index': i
                })
        return fvgs
    
    def _detect_liquidity(self, highs, lows):
        """Detect liquidity levels"""
        liquidity = []
        for i in range(self.swing_length, len(highs) - self.swing_length):
            # High liquidity
            if all(highs[i] >= highs[i-j] for j in range(1, self.swing_length)) and \
               all(highs[i] >= highs[i+j] for j in range(1, self.swing_length)):
                liquidity.append({'type': 'resistance', 'price': highs[i], 'index': i})
            
            # Low liquidity
            if all(lows[i] <= lows[i-j] for j in range(1, self.swing_length)) and \
               all(lows[i] <= lows[i+j] for j in range(1, self.swing_length)):
                liquidity.append({'type': 'support', 'price': lows[i], 'index': i})
                
        return liquidity
    
    def _generate_signal(self, order_blocks, fvgs, liquidity_levels, current_price):
        """Generate trading signal based on SMC analysis"""
        buy_signals = 0
        sell_signals = 0
        
        # Order block signals
        for ob in order_blocks[-3:]:  # Recent order blocks
            if ob['type'] == 'bullish' and current_price > ob['price']:
                buy_signals += 1
        
        # FVG signals
        for fvg in fvgs[-2:]:  # Recent FVGs
            if fvg['type'] == 'bullish' and fvg['bottom'] <= current_price <= fvg['top']:
                buy_signals += 1
        
        # Liquidity signals
        for liq in liquidity_levels[-3:]:
            if liq['type'] == 'support' and current_price > liq['price']:
                buy_signals += 1
        
        if buy_signals >= 2:
            return "BUY"
        elif sell_signals >= 2:
            return "SELL"
        
        return None
    
    def _calculate_confidence(self, order_blocks, fvgs, liquidity_levels):
        """Calculate signal confidence"""
        total_confirmations = len(order_blocks) + len(fvgs) + len(liquidity_levels)
        return min(total_confirmations * 0.15, 1.0)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'swing_length': {'type': 'int', 'min': 5, 'max': 50, 'default': 10},
            'fvg_threshold': {'type': 'float', 'min': 0.1, 'max': 2.0, 'default': 0.5},
            'ob_lookback': {'type': 'int', 'min': 10, 'max': 100, 'default': 20}
        }
    
    def validate_config(self) -> bool:
        return True

class RSIIndicator(BaseIndicator):
    """RSI with Divergence Detection"""
    
    def __init__(self, config: IndicatorConfig):
        super().__init__(config)
        self.period = config.parameters.get('rsi_period', 14)
        self.overbought = config.parameters.get('overbought', 70)
        self.oversold = config.parameters.get('oversold', 30)
    
    async def analyze(self, symbol: str, timeframe: str, data: Dict) -> Optional[Signal]:
        """Analyze RSI with divergence"""
        try:
            closes = data.get('close', [])
            if len(closes) < self.period + 10:
                return None
            
            # Calculate RSI
            rsi_values = self._calculate_rsi(closes)
            current_rsi = rsi_values[-1]
            
            # Detect divergence
            divergence = self._detect_divergence(closes, rsi_values)
            
            # Generate signal
            signal_type = None
            confidence = 0.5
            
            if current_rsi < self.oversold:
                signal_type = "BUY"
                confidence += 0.2
            elif current_rsi > self.overbought:
                signal_type = "SELL"
                confidence += 0.2
            
            if divergence:
                if divergence['type'] == 'bullish':
                    signal_type = "BUY"
                    confidence += 0.3
                elif divergence['type'] == 'bearish':
                    signal_type = "SELL"
                    confidence += 0.3
            
            if signal_type:
                return Signal(
                    indicator="rsi",
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_type=signal_type,
                    confidence=min(confidence, 1.0),
                    price=closes[-1],
                    timestamp=datetime.now(),
                    metadata={
                        'rsi': current_rsi,
                        'divergence': divergence
                    }
                )
                
        except Exception as e:
            logger.error(f"RSI analysis error for {symbol}: {e}")
            
        return None
    
    def _calculate_rsi(self, closes):
        """Calculate RSI values"""
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        rsi_values = []
        for i in range(self.period - 1, len(gains)):
            avg_gain = sum(gains[i-self.period+1:i+1]) / self.period
            avg_loss = sum(losses[i-self.period+1:i+1]) / self.period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        return rsi_values
    
    def _detect_divergence(self, prices, rsi_values):
        """Detect price-RSI divergence"""
        if len(prices) < 20 or len(rsi_values) < 20:
            return None
        
        # Simplified divergence detection
        recent_prices = prices[-10:]
        recent_rsi = rsi_values[-10:]
        
        price_trend = recent_prices[-1] - recent_prices[0]
        rsi_trend = recent_rsi[-1] - recent_rsi[0]
        
        if price_trend > 0 and rsi_trend < -5:
            return {'type': 'bearish', 'strength': abs(rsi_trend)}
        elif price_trend < 0 and rsi_trend > 5:
            return {'type': 'bullish', 'strength': abs(rsi_trend)}
        
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'rsi_period': {'type': 'int', 'min': 5, 'max': 50, 'default': 14},
            'overbought': {'type': 'int', 'min': 60, 'max': 90, 'default': 70},
            'oversold': {'type': 'int', 'min': 10, 'max': 40, 'default': 30}
        }
    
    def validate_config(self) -> bool:
        return True

class NR4NR7Indicator(BaseIndicator):
    """NR4 & NR7 Narrow Range Indicator with Breakout Detection"""
    
    def __init__(self, config: IndicatorConfig):
        super().__init__(config)
        self.pattern_type = config.parameters.get('pattern_type', 'NR4/NR7')  # NR4/NR7, NR4, NR7
        self.breakout_signals = config.parameters.get('breakout_signals', True)
        self.signal_strength = config.parameters.get('signal_strength', 0.7)
        
        # State tracking for breakouts
        self.active_ranges = {}  # symbol -> range data
        self.range_history = {}  # symbol -> list of historical ranges
        
    async def analyze(self, symbol: str, timeframe: str, data: Dict) -> Optional[Signal]:
        """Analyze for NR4/NR7 patterns and breakouts"""
        try:
            opens = data.get('open', [])
            highs = data.get('high', [])
            lows = data.get('low', [])
            closes = data.get('close', [])
            
            if len(closes) < 10:
                return None
            
            # Calculate ranges for recent bars
            ranges = [highs[i] - lows[i] for i in range(len(highs))]
            
            if len(ranges) < 7:
                return None
            
            # Check for NR4 and NR7 patterns
            current_range = ranges[-1]
            
            # NR7 check (current range is smallest of last 7)
            nr7_detected = False
            if self.pattern_type in ['NR4/NR7', 'NR7']:
                last_7_ranges = ranges[-7:]
                nr7_detected = current_range == min(last_7_ranges)
            
            # NR4 check (current range is smallest of last 4, but not NR7)
            nr4_detected = False
            if self.pattern_type in ['NR4/NR7', 'NR4'] and not nr7_detected:
                last_4_ranges = ranges[-4:]
                nr4_detected = current_range == min(last_4_ranges)
            
            pattern_detected = nr4_detected or nr7_detected
            
            if pattern_detected:
                # Store the range for breakout monitoring
                range_data = {
                    'high': highs[-1],
                    'low': lows[-1],
                    'mid': (highs[-1] + lows[-1]) / 2,
                    'type': 'NR7' if nr7_detected else 'NR4',
                    'timestamp': datetime.now(),
                    'breakout_up': False,
                    'breakout_down': False,
                    'price_above_mid': closes[-1] > (highs[-1] + lows[-1]) / 2,
                    'price_below_mid': closes[-1] < (highs[-1] + lows[-1]) / 2
                }
                
                self.active_ranges[symbol] = range_data
                
                # Initialize range history if needed
                if symbol not in self.range_history:
                    self.range_history[symbol] = []
                
                # Add to history (keep last 50)
                self.range_history[symbol].append(range_data.copy())
                if len(self.range_history[symbol]) > 50:
                    self.range_history[symbol] = self.range_history[symbol][-50:]
                
                logger.info(f"{range_data['type']} pattern detected for {symbol}")
            
            # Check for breakouts if we have an active range
            if symbol in self.active_ranges and self.breakout_signals:
                range_data = self.active_ranges[symbol]
                current_price = closes[-1]
                
                # Breakout detection logic
                signal_type = None
                confidence = self.signal_strength
                metadata = {
                    'pattern_type': range_data['type'],
                    'range_high': range_data['high'],
                    'range_low': range_data['low'],
                    'range_mid': range_data['mid']
                }
                
                # Upward breakout
                if (current_price > range_data['high'] and 
                    not range_data['breakout_up'] and
                    range_data['price_below_mid']):
                    
                    signal_type = "BUY"
                    confidence += 0.2
                    range_data['breakout_up'] = True
                    metadata['breakout_type'] = 'upward'
                    
                # Downward breakout  
                elif (current_price < range_data['low'] and 
                      not range_data['breakout_down'] and
                      range_data['price_above_mid']):
                    
                    signal_type = "SELL"
                    confidence += 0.2
                    range_data['breakout_down'] = True
                    metadata['breakout_type'] = 'downward'
                
                # Update price position relative to mid
                if current_price > range_data['mid']:
                    range_data['price_above_mid'] = True
                    range_data['price_below_mid'] = False
                elif current_price < range_data['mid']:
                    range_data['price_below_mid'] = True
                    range_data['price_above_mid'] = False
                
                # Generate signal if breakout detected
                if signal_type:
                    return Signal(
                        indicator="nr4nr7",
                        symbol=symbol,
                        timeframe=timeframe,
                        signal_type=signal_type,
                        confidence=min(confidence, 1.0),
                        price=current_price,
                        timestamp=datetime.now(),
                        metadata=metadata
                    )
                    
        except Exception as e:
            logger.error(f"NR4/NR7 analysis error for {symbol}: {e}")
            
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'pattern_type': {
                'type': 'select', 
                'options': ['NR4/NR7', 'NR4', 'NR7'], 
                'default': 'NR4/NR7',
                'description': 'Type of narrow range pattern to detect'
            },
            'breakout_signals': {
                'type': 'bool', 
                'default': True,
                'description': 'Generate signals on range breakouts'
            },
            'signal_strength': {
                'type': 'float', 
                'min': 0.1, 
                'max': 1.0, 
                'default': 0.7,
                'description': 'Base confidence level for signals'
            }
        }
    
    def validate_config(self) -> bool:
        pattern_type = self.config.parameters.get('pattern_type', 'NR4/NR7')
        return pattern_type in ['NR4/NR7', 'NR4', 'NR7']
    
    def get_active_ranges(self, symbol: str = None) -> Dict:
        """Get currently active ranges for monitoring"""
        if symbol:
            return self.active_ranges.get(symbol, {})
        return self.active_ranges.copy()
    
    def get_range_history(self, symbol: str = None) -> Dict:
        """Get historical range data"""
        if symbol:
            return self.range_history.get(symbol, [])
        return self.range_history.copy()

class UnifiedAutomationEngine:
    """Main engine for managing multiple indicators"""
    
    def __init__(self, qnti_system=None):
        self.qnti_system = qnti_system
        self.indicators: Dict[str, BaseIndicator] = {}
        self.active_configs: Dict[str, IndicatorConfig] = {}
        self.signals_history: List[Signal] = []
        self.is_running = False
        self.automation_task = None
        
        # Available indicator classes
        self.indicator_classes = {
            'smc': SMCIndicator,
            'rsi': RSIIndicator,
            'nr4nr7': NR4NR7Indicator, # Register NR4NR7Indicator
            # Add more indicators here
        }
        
    def register_indicator(self, indicator_type: str, indicator_class: type):
        """Register a new indicator type"""
        self.indicator_classes[indicator_type] = indicator_class
        
    def load_config(self, config_data: Dict) -> bool:
        """Load automation configuration"""
        try:
            for indicator_name in config_data.get('indicators', []):
                if indicator_name in self.indicator_classes:
                    # Create indicator config
                    indicator_config = IndicatorConfig(
                        name=indicator_name,
                        type=indicator_name,
                        parameters=config_data.get('parameters', {}).get(indicator_name, {}),
                        signals=config_data.get('signals', {}),
                        risk_management=config_data.get('risk', {}),
                        alerts=config_data.get('alerts', {}),
                        symbols=config_data.get('symbols', ['EURUSD']),
                        timeframes=['M15', 'H1'],  # Default timeframes
                        enabled=True
                    )
                    
                    # Create indicator instance
                    indicator_class = self.indicator_classes[indicator_name]
                    indicator = indicator_class(indicator_config)
                    
                    if indicator.validate_config():
                        self.indicators[indicator_name] = indicator
                        self.active_configs[indicator_name] = indicator_config
                        logger.info(f"Loaded indicator: {indicator_name}")
                    else:
                        logger.error(f"Invalid config for indicator: {indicator_name}")
                        return False
                        
            return True
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return False
    
    async def start_automation(self, config_data: Dict) -> Dict[str, Any]:
        """Start the automation system"""
        try:
            if self.is_running:
                return {'success': False, 'message': 'Automation already running'}
            
            # Load configuration
            if not self.load_config(config_data):
                return {'success': False, 'message': 'Invalid configuration'}
            
            if not self.indicators:
                return {'success': False, 'message': 'No valid indicators configured'}
            
            # Start automation loop
            self.is_running = True
            self.automation_task = asyncio.create_task(self._automation_loop())
            
            logger.info(f"Started automation with {len(self.indicators)} indicators")
            return {
                'success': True, 
                'message': f'Automation started with {len(self.indicators)} indicators',
                'indicators': list(self.indicators.keys())
            }
            
        except Exception as e:
            logger.error(f"Error starting automation: {e}")
            return {'success': False, 'message': str(e)}
    
    async def stop_automation(self) -> Dict[str, Any]:
        """Stop the automation system"""
        try:
            self.is_running = False
            
            if self.automation_task:
                self.automation_task.cancel()
                try:
                    await self.automation_task
                except asyncio.CancelledError:
                    pass
                self.automation_task = None
            
            logger.info("Automation stopped")
            return {'success': True, 'message': 'Automation stopped'}
            
        except Exception as e:
            logger.error(f"Error stopping automation: {e}")
            return {'success': False, 'message': str(e)}
    
    async def _automation_loop(self):
        """Main automation loop"""
        while self.is_running:
            try:
                # Process each symbol for each indicator
                for indicator_name, indicator in self.indicators.items():
                    if not indicator.is_enabled():
                        continue
                    
                    for symbol in indicator.get_supported_symbols():
                        # Get market data
                        market_data = await self._get_market_data(symbol, 'H1')
                        
                        if market_data:
                            # Analyze with indicator
                            signal = await indicator.analyze(symbol, 'H1', market_data)
                            
                            if signal:
                                await self._process_signal(signal)
                
                # Wait before next iteration
                await asyncio.sleep(30)  # 30 seconds interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in automation loop: {e}")
                await asyncio.sleep(5)
    
    async def _get_market_data(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get market data for analysis"""
        try:
            if self.qnti_system and hasattr(self.qnti_system, 'get_symbol_data'):
                return await self.qnti_system.get_symbol_data(symbol, timeframe)
            
            # Fallback to dummy data for testing
            import random
            base_price = 1.1000 if 'USD' in symbol else 1500
            return {
                'open': [base_price + random.uniform(-0.01, 0.01) for _ in range(100)],
                'high': [base_price + random.uniform(0, 0.02) for _ in range(100)],
                'low': [base_price + random.uniform(-0.02, 0) for _ in range(100)],
                'close': [base_price + random.uniform(-0.01, 0.01) for _ in range(100)],
                'volume': [random.randint(1000, 10000) for _ in range(100)]
            }
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    async def _process_signal(self, signal: Signal):
        """Process a trading signal"""
        try:
            # Add to history
            self.signals_history.append(signal)
            
            # Keep only recent signals
            if len(self.signals_history) > 1000:
                self.signals_history = self.signals_history[-1000:]
            
            logger.info(f"Signal: {signal.signal_type} {signal.symbol} @ {signal.price} "
                       f"({signal.confidence:.2%} confidence)")
            
            # Send alerts if configured
            config = self.active_configs.get(signal.indicator)
            if config and config.alerts.get('web', False):
                await self._send_web_alert(signal)
            
            # Execute trade if auto-trading is enabled
            # This would integrate with your trading system
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    async def _send_web_alert(self, signal: Signal):
        """Send web notification"""
        # Implementation for web alerts
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get automation status"""
        return {
            'is_running': self.is_running,
            'active_indicators': list(self.indicators.keys()),
            'total_signals': len(self.signals_history),
            'recent_signals': [asdict(s) for s in self.signals_history[-10:]],
            'uptime': time.time() if self.is_running else 0
        }
    
    def get_available_indicators(self) -> Dict[str, Any]:
        """Get available indicator types and their parameters"""
        indicators_info = {}
        
        for indicator_type, indicator_class in self.indicator_classes.items():
            # Create a temporary instance to get parameters
            temp_config = IndicatorConfig(
                name=indicator_type,
                type=indicator_type,
                parameters={},
                signals={},
                risk_management={},
                alerts={},
                symbols=[],
                timeframes=[]
            )
            
            try:
                temp_indicator = indicator_class(temp_config)
                indicators_info[indicator_type] = {
                    'name': indicator_class.__name__,
                    'parameters': temp_indicator.get_parameters(),
                    'description': temp_indicator.__doc__ or f"{indicator_type} indicator"
                }
            except Exception as e:
                logger.error(f"Error getting info for {indicator_type}: {e}")
        
        return indicators_info

# Global automation engine instance
automation_engine = None

def get_automation_engine(qnti_system=None):
    """Get or create automation engine instance"""
    global automation_engine
    if automation_engine is None:
        automation_engine = UnifiedAutomationEngine(qnti_system)
    return automation_engine 